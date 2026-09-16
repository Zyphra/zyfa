#pragma once
// Local to the hip research family (not the production fmha tree).
// Hardware round-robins flattened block IDs across XCDs (CTA i -> XCD i % nxcd),
// each with a 4 MB L2. The default 2D grid (q-pair, head) is x-fast, so one XCD
// sees many heads' K/V. Remap so each XCD walks consecutive Q-pairs of one head.
// Bijection on (qx, bh). ZK_FMHA_NXCD overrides.
#include <cstdlib>
#include <hip/hip_runtime.h>

__device__ __forceinline__ void fmha_xcd_map(int nq_pairs, int BH, int nxcd, int &qx, int &bh, int i) {
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

__host__ inline int fmha_num_xcds() {
    static int cached = -1;
    if (cached >= 0)
        return cached;
    int n = 8;
    if (const char *e = std::getenv("ZK_FMHA_NXCD")) {
        const int v = std::atoi(e);
        if (v > 0)
            n = v;
    } else {
        int dev = 0, v = 0;
        if (hipGetDevice(&dev) == hipSuccess &&
            hipDeviceGetAttribute(&v, hipDeviceAttributeNumberOfXccs, dev) == hipSuccess && v > 0)
            n = v;
    }
    cached = n;
    return n;
}

// DeepSeek-style persistent grid: one WG per CU (multiple of nxcd so
// grid-stride keeps block i on XCD i % nxcd). No atomics.
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

// Split B>1 into B=1 launches only when one batch already saturates the persist
// CU cap. Fusing when nq_pairs*nheads < cap occupies idle CUs (B=4 H=8 S=8K
// is 128 jobs). ZK_FMHA_FUSE_B=1 forces a single fused grid.
__host__ inline bool fmha_split_batch(int B, int nq_pairs, int nheads, int nxcd) {
    if (B <= 1)
        return false;
    if (const char *e = std::getenv("ZK_FMHA_FUSE_B"); e && std::atoi(e) == 1)
        return false;
    return nq_pairs * nheads >= fmha_persist_cap(nxcd);
}
