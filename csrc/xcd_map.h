#pragma once
// Job-to-XCD mapping for the causal kernels, plus host helpers for the persistent grid and the scale checks.
#include <cstdint>
#include <c10/util/Exception.h>
#include <cstring>
#include <hip/hip_runtime.h>

// Workgroup i runs on XCD i % nxcd (each XCD has its own L2). Remap i so each XCD walks consecutive Q-block pairs of
// one head instead of every nxcd-th job; a bijection on (qx, bh).
__host__ __device__ __forceinline__ void fmha_xcd_map(int nq_pairs, int BH, int nxcd, int &qx, int &bh, int i) {
    if (nxcd <= 1) {
        qx = i % nq_pairs;
        bh = i / nq_pairs;
        return;
    }
    const int total = nq_pairs * BH;
    const int n_per = total / nxcd;
    const int extra = total % nxcd;
    const int xcd = i % nxcd;
    const int seq = i / nxcd;
    const int off = xcd * n_per + (xcd < extra ? xcd : extra);
    const int L = off + seq;
    qx = L % nq_pairs;
    bh = L / nq_pairs;
}

__device__ __forceinline__ void fmha_xcd_map(int nq_pairs, int BH, int nxcd, int &qx, int &bh) {
    fmha_xcd_map(nq_pairs, BH, nxcd, qx, bh, (int)blockIdx.x);
}

// XCD count of the current device (8 if the query fails).
__host__ inline int fmha_device_xcds() {
    int dev = 0, v = 0;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipDeviceGetAttribute(&v, hipDeviceAttributeNumberOfXccs, dev) == hipSuccess && v > 0)
        return v;
    return 8;
}

// Same, queried once.
__host__ inline int fmha_num_xcds() {
    static int cached = -1;
    if (cached < 0)
        cached = fmha_device_xcds();
    return cached;
}

__host__ inline int fmha_num_cus() {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    int n = 256;
    int dev = 0, v = 0;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev) == hipSuccess && v > 0)
        n = v;
    cached = n;
    return n;
}

// Persistent grid size: at most one workgroup per CU, a multiple of nxcd so a grid-stride job walk stays on one XCD.
__host__ inline int fmha_persist_cap(int nxcd) {
    const int nxc = nxcd > 1 ? nxcd : 1;
    int cap = fmha_num_cus();
    cap -= cap % nxc;
    if (cap < nxc)
        cap = nxc;
    return cap;
}

__host__ inline int fmha_persist_grid(int n_work, int nxcd) {
    const int cap = fmha_persist_cap(nxcd);
    return n_work < cap ? n_work : cap;
}

// Scale checks on the bit patterns: the extension builds with -ffast-math, which folds std::isfinite / isnan to true.
__host__ inline uint64_t fmha_f64_bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
// |scale| * log2(e) is finite in fp32 (rejects inf and nan).
__host__ inline bool fmha_scale_finite(double scale) {
    return (fmha_f64_bits(scale) & 0x7fffffffffffffffull) < fmha_f64_bits(3.4028234663852886e38 / 1.4426950408889634);
}
// FP8: also positive and an fp32 normal (a denormal scale flushes to 0 in the kernel and masked -inf * 0 = NaN).
__host__ inline bool fmha_scale_fp8_ok(double scale) {
    const uint64_t u = fmha_f64_bits(scale);
    return (u >> 63) == 0 && u >= fmha_f64_bits(1.1754943508222875e-38) && fmha_scale_finite(scale);
}

// Split B > 1 into B = 1 launches only at long S (M >= 8192) when one batch already fills the persistent grid.
__host__ inline bool fmha_split_batch(int B, int nq_pairs, int nheads, int nxcd, int M) {
    if (B <= 1)
        return false;
    if (M < 8192)
        return false;
    return nq_pairs * nheads >= fmha_persist_cap(nxcd);
}

// Throws on a failed HIP runtime call.
inline void hip_check(hipError_t e, const char *what) { TORCH_CHECK(e == hipSuccess, what, ": ", hipGetErrorString(e)); }
