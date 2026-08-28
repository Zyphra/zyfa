#pragma once
// Local to the hip research family (not the production fmha tree).
// Hardware round-robins flattened block IDs across XCDs (CTA i -> XCD i % nxcd),
// each with a 4 MB L2. The default 2D grid (q-pair, head) is x-fast, so one XCD
// sees many heads' K/V. Remap so each XCD walks consecutive Q-pairs of one head.
// Bijection on (qx, bh). ZK_FMHA_NXCD overrides.
#include <cstdlib>
#include <hip/hip_runtime.h>

__device__ __forceinline__ void fmha_xcd_map(int nq_pairs, int BH, int nxcd, int &qx, int &bh) {
    const int i = (int)blockIdx.x;
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
