// HK GQA causal + XCD pair, fp8 fat 32x32x64, KN=64, streamed LDS K/V.
// KV loop unrolled 8. QK at lgkmcnt(2); next-K under last PV aliases k_frag.
// One bar per tile (after softmax, before PV). D=192 / Dv=128 / 8-wave / causal only.
// RESCALE_SKIP: true (default) = HK Δmax≤8 delayed update. false = always rescale.
#ifndef RESCALE_SKIP
#define RESCALE_SKIP true
#endif
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_fp8_d192.h"
#include "xcd_map.h"

namespace fmha_fp8_d192 {

using namespace hk;
using namespace hk8_192;

constexpr int Q_BLOCK = 32;
constexpr int KV_BLOCK = 64;
constexpr int NW = 8;
constexpr int NT = 512;
constexpr float L2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ __attribute__((convergent)) void wg_bar() { __syncthreads(); }

struct alignas(16) Smem {
    st_k k[2];
    st_v v[2];
};

__device__ __forceinline__ const unsigned char *kv_tile_ptr(const unsigned char *base, int tile, int dim) {
    return base + (size_t)tile * KV_BLOCK * dim;
}

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    if (!__any(corr != 1.f))
        return;
    o.a = mulsc(o.a, corr);
    o.b = mulsc(o.b, corr);
    o.c = mulsc(o.c, corr);
    o.d = mulsc(o.d, corr);
}

__device__ __forceinline__ void softmax_pv(o_tile &o, s2_tile &s, uint32_t vbase, float scale, float &m_, float &l_,
                                          k_frag &kf, uint32_t k_next_off) {
    const float mx = col_max2_sc(s, scale);
    const float mnew = fmaxf(m_, mx);
#if RESCALE_SKIP
    const float dmax = mnew - m_;
    const int skip = __all(dmax <= RESCALE_THRESHOLD);
    float corr = 1.f;
    float mref = m_;
    if (!__builtin_expect(skip, 1)) {
        corr = (dmax > 0.f) ? zk_exp2(-dmax) : 1.f;
        if (__builtin_expect(m_ != neg_inf(), 1))
            rescale_o(o, corr);
        mref = mnew;
        m_ = mnew;
    }
#else
    const float corr = (mnew > m_) ? zk_exp2(m_ - mnew) : 1.f;
    if (__builtin_expect(m_ != neg_inf(), 1))
        rescale_o(o, corr);
    const float mref = mnew;
    m_ = mnew;
#endif
    zk_iglp_smx();
    hk8::exp2_sc(s.a, scale, mref);
    hk8::exp2_sc(s.b, scale, mref);
    l_ = l_ * corr + col_sum2(s);
    const i32x8 ps = pack_p(s.a, s.b);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    wg_bar();
    zk_setprio_hi();
    mma_pv_lds_khead(o, vbase, ps, kf, k_next_off);
    zk_setprio_lo();
}

template <bool PairHeads, bool Bshd>
__global__ void __launch_bounds__(NT, ZK_LAUNCH_BLOCKS) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, bf16 *O, float *Lse, int M, int N, int Hq,
                                                    int Hkv, float scale, int nxcd) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 192, Dv = 128;
    // Keep stride in an SGPR for both instantiations so BHSD does not
    // constant-fold D and take a different pack_p / QK schedule than BSHD.
    int q_stride = Bshd ? Hq * D : D;
    int k_stride = Bshd ? Hkv * D : D;
    int v_stride = Bshd ? Hkv * Dv : Dv;
    int o_stride = Bshd ? Hq * Dv : Dv;
    asm volatile("" : "+s"(q_stride), "+s"(k_stride), "+s"(v_stride), "+s"(o_stride));
    const int GROUP = Hq / Hkv;
    constexpr int QW = PairHeads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + QW - 1) / QW;
    const int nq_pairs = (numQ + 1) / 2;
    const int n_head_units = (int)gridDim.x / nq_pairs;
    int qx, hu;
    fmha_xcd_map(nq_pairs, n_head_units, nxcd, qx, hu);

    int batch_idx, head_idx, head_kv, qwarp;
    if constexpr (PairHeads) {
        // Eight physical waves become two barrier-synchronous 4-wave
        // Q-head groups. The two heads share one KV head and identical
        // q positions, so every K/V G->L transfer services both heads.
        const int pairs_per_kv = GROUP / 2;
        const int units_per_batch = Hkv * pairs_per_kv;
        batch_idx = hu / units_per_batch;
        const int u = hu - batch_idx * units_per_batch;
        head_kv = u / pairs_per_kv;
        const int pair = u - head_kv * pairs_per_kv;
        const int head_slot = warpid() / QW;
        qwarp = warpid() - head_slot * QW;
        head_idx = head_kv * GROUP + pair * 2 + head_slot;
    } else {
        // Identity Q-head order keeps GROUP consecutive Q heads on one KV
        // head so fmha_xcd_map locality extends across GQA.
        batch_idx = hu / Hq;
        head_idx = hu - batch_idx * Hq;
        head_kv = head_idx / GROUP;
        qwarp = warpid();
    }

    const unsigned char *Qh, *Kh, *Vh;
    bf16 *Oh;
    if constexpr (Bshd) {
        Qh = Q + ((size_t)batch_idx * M * Hq + (size_t)head_idx) * D;
        Kh = K + ((size_t)batch_idx * N * Hkv + (size_t)head_kv) * D;
        Vh = V + ((size_t)batch_idx * N * Hkv + (size_t)head_kv) * Dv;
        Oh = O + ((size_t)batch_idx * M * Hq + (size_t)head_idx) * Dv;
    } else {
        Qh = Q + ((size_t)batch_idx * Hq + head_idx) * M * D;
        Kh = K + ((size_t)batch_idx * Hkv + head_kv) * N * D;
        Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * Dv;
        Oh = O + ((size_t)batch_idx * Hq + head_idx) * M * Dv;
    }
    float *Lh = Lse ? Lse + ((size_t)batch_idx * Hq + head_idx) * M : nullptr;

    const int num_tiles = N / KV_BLOCK;
    const int q_lo = qx, q_hi = numQ - 1 - qx;

    for (int pp = 0; pp < 2; ++pp) {
        const int qblk = pp ? q_lo : q_hi;
        if (pp && q_lo == q_hi)
            break;
        const int tile_idx = qblk * QW + qwarp;
        const int max_tile_idx = qblk * QW + QW - 1;
        const int max_q_end = (max_tile_idx + 1) * Q_BLOCK;
        int max_num_tiles = (max_q_end + KV_BLOCK - 1) / KV_BLOCK;
        if (max_num_tiles > num_tiles)
            max_num_tiles = num_tiles;
        const int q_start = tile_idx * Q_BLOCK;

        q8_tile192 qT;
        o_tile oR;
        k_frag kf;
        zero_o(oR);
        load_q8(qT, Qh + (size_t)tile_idx * Q_BLOCK * q_stride, q_stride);
        pin_q192(qT);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp) {
            if constexpr (PairHeads)
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            wg_bar();
        }
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, k_stride), k_stride);
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, v_stride), v_stride);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        wg_bar();
        issue_k_pair(kf, k_off(lds_u32(s.k[0].data), 0));
        pin_kf(kf);

        auto one_tile = [&](int kb) {
            // Required for BSHD vs BHSD bitwise match: without it pack_p
            // takes a different schedule on the two instantiations.
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            wg_bar();
            const int cur = kb & 1, nx = cur ^ 1;
            const uint32_t kbase = lds_u32(s.k[cur].data);
            const uint32_t vbase = lds_u32(s.v[cur].data);
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            const unsigned char *gKn = kv_tile_ptr(Kh, kb_n, k_stride);
            const unsigned char *gVn = kv_tile_ptr(Vh, kb_n, v_stride);
            s2_tile att;
            const uint32_t k_next_off = k_off(lds_u32(s.k[nx].data), 0);
            zk_iglp_qk();
            if (warpid() < 4)
                zk_setprio_hi();
            else
                zk_setprio_lo();
            prefetch_k_pass<0>(s.k[nx], gKn, k_stride);
            qk_step<false>(att, kf, qT.d0, k_off(kbase, 1));
            prefetch_k_pass<1>(s.k[nx], gKn, k_stride);
            qk_step<true>(att, kf, qT.d1, k_off(kbase, 2));
            prefetch_v(s.v[nx], gVn, v_stride);
            qk_tail<true>(att, kf, qT.d2);
            zk_qk_group_fp8_d192();
            zk_split_region();
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s2(att, q_start, kb);
            softmax_pv(oR, att, vbase, scale, m_, l_, kf, k_next_off);
        };

        int kb = 0;
        const int n8 = max_num_tiles & ~7;
#pragma unroll 1
        for (; kb < n8; kb += 8) {
#pragma unroll
            for (int u = 0; u < 8; ++u)
                one_tile(kb + u);
        }
        for (; kb < max_num_tiles; ++kb)
            one_tile(kb);

        store_o_fp8(Oh, q_start, M, oR, l_, o_stride);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, bf16 *O, float *Lse, int M,
                int N, int B, int Hq, int Hkv, float scale, bool bshd, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    constexpr int D = 192, Dv = 128;
    bool split = B > 1 && M >= 8192;
    if (const char *e = std::getenv("ZK_FMHA_FUSE_B"); e && std::atoi(e) == 1)
        split = false;
    const int batches = split ? B : 1;
    const int n_head_units = nheads * (split ? 1 : B);
    const int grid = nq_pairs * n_head_units;
    const size_t q_batch = bshd ? (size_t)M * Hq * D : (size_t)Hq * M * D;
    const size_t k_batch = bshd ? (size_t)N * Hkv * D : (size_t)Hkv * N * D;
    const size_t v_batch = bshd ? (size_t)N * Hkv * Dv : (size_t)Hkv * N * Dv;
    const size_t o_batch = bshd ? (size_t)M * Hq * Dv : (size_t)Hq * M * Dv;
    if (pair_heads && bshd)
        hipFuncSetAttribute((const void *)attend_ker<true, true>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    else if (pair_heads)
        hipFuncSetAttribute((const void *)attend_ker<true, false>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    else if (bshd)
        hipFuncSetAttribute((const void *)attend_ker<false, true>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    else
        hipFuncSetAttribute((const void *)attend_ker<false, false>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536);
    // S>=8K: one B=1 grid per batch. Short S keeps one fused grid.
    // Full grid (no persist loop): a job loop in attend_ker breaks pack_p bitwise.
    for (int b = 0; b < batches; ++b) {
        const unsigned char *Qb = Q + b * q_batch;
        const unsigned char *Kb = K + b * k_batch;
        const unsigned char *Vb = V + b * v_batch;
        bf16 *Ob = O + b * o_batch;
        float *Lb = Lse ? Lse + (size_t)b * Hq * M : nullptr;
        if (pair_heads && bshd)
            attend_ker<true, true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd);
        else if (pair_heads)
            attend_ker<true, false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd);
        else if (bshd)
            attend_ker<false, true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd);
        else
            attend_ker<false, false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                              bool return_lse, bool bshd) {
    TORCH_CHECK(causal, "fmha_fp8_d192 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "fmha_fp8_d192 needs float8_e4m3fn");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                bshd ? "q,k,v must be BSHD [B,S,H,D]" : "q,k,v must be BHSD [B,H,S,D]");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int D = q.size(3), Dv = v.size(3);
    const int Hq = bshd ? q.size(2) : q.size(1);
    const int M = bshd ? q.size(1) : q.size(2);
    const int Hkv = bshd ? k.size(2) : k.size(1);
    const int N = bshd ? k.size(1) : k.size(2);
    const int B = q.size(0);
    TORCH_CHECK(D == 192 && Dv == 128 && M % 256 == 0 && N % 64 == 0);
    TORCH_CHECK(Hq % Hkv == 0 && k.size(3) == D && q.size(0) == k.size(0) && q.size(0) == v.size(0));
    if (bshd) {
        TORCH_CHECK(v.size(1) == N && v.size(2) == Hkv);
    } else {
        TORCH_CHECK(v.size(1) == Hkv && v.size(2) == N);
    }
    auto O = bshd ? torch::empty({B, M, Hq, Dv}, q.options().dtype(at::kBFloat16))
                  : torch::empty({B, Hq, M, Dv}, q.options().dtype(at::kBFloat16));
    torch::Tensor Lse;
    float *Lp = nullptr;
    if (return_lse) {
        Lse = torch::empty({B * Hq, M}, q.options().dtype(at::kFloat));
        Lp = Lse.data_ptr<float>();
    }
    run((const unsigned char *)q.data_ptr(), (const unsigned char *)k.data_ptr(),
        (const unsigned char *)v.data_ptr(), reinterpret_cast<bf16 *>(O.data_ptr()), Lp, M, N, B, Hq, Hkv,
        (float)scale * L2E, bshd, at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_fp8_d192
