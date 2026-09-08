// HK GQA causal + XCD pair, bf16 32x32x16, KN=128, streamed LDS K/V.
// One bar per tile (next iter's bar covers V lifetime). Research kernel. D=Dv=128 / 8-wave / causal only. Do not fold into fmha32.hip.
// RESCALE_SKIP: true (default) = HK Δmax≤8 delayed update. false = always rescale.
#ifndef RESCALE_SKIP
#define RESCALE_SKIP true
#endif
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_bf16_d128.h"
#include "xcd_map.h"

namespace fmha_bf16_d128 {

using namespace hk;
using namespace hk16;

constexpr int Q_BLOCK = 32;
constexpr int KV_BLOCK = 128;
constexpr int NW = 8;
constexpr int NT = 512;
constexpr float L2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ __attribute__((convergent)) void wg_bar() { __syncthreads(); }

struct alignas(16) Smem {
    st16 k[2];
    st16 v[2];
};

__device__ __forceinline__ const bf16 *kv_tile_ptr(const bf16 *base, int tile, int seq_stride) {
    return base + (size_t)tile * KV_BLOCK * seq_stride;
}

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    if (!__any(corr != 1.f))
        return;
    o.a = mulsc(o.a, corr);
    o.b = mulsc(o.b, corr);
    o.c = mulsc(o.c, corr);
    o.d = mulsc(o.d, corr);
}

struct SmxAcc {
    float mref, corr;
};

__device__ __forceinline__ SmxAcc smx_prep(o_tile &o, s4_tile &s, float &m_) {
    const float mx = col_max4(s);
    const float mnew = fmaxf(m_, mx);
    SmxAcc a;
#if RESCALE_SKIP
    const float dmax = mnew - m_;
    const int skip = __all(dmax <= RESCALE_THRESHOLD);
    a.corr = 1.f;
    a.mref = m_;
    if (!__builtin_expect(skip, 1)) {
        a.corr = (dmax > 0.f) ? zk_exp2(-dmax) : 1.f;
        if (__builtin_expect(m_ != neg_inf(), 1))
            rescale_o(o, a.corr);
        a.mref = mnew;
        m_ = mnew;
    }
#else
    a.corr = (mnew > m_) ? zk_exp2(m_ - mnew) : 1.f;
    if (__builtin_expect(m_ != neg_inf(), 1))
        rescale_o(o, a.corr);
    a.mref = mnew;
    m_ = mnew;
#endif
    return a;
}

__device__ __forceinline__ void smx_pv(o_tile &o, s4_tile &s, const st16 &vsm, SmxAcc a, float &l_) {
    const auto &vlo = *reinterpret_cast<const st<64, 128> *>(&vsm.data[0]);
    const auto &vhi = *reinterpret_cast<const st<64, 128> *>(&vsm.data[64 * Dv]);
    zk_iglp_smx();
    exp2_h<0>(s.a, a.mref);
    zk_setprio_hi();
    mma_pv_one<0>(o, vlo, cvt8(s.a, 0));
    exp2_h<8>(s.a, a.mref);
    mma_pv_one<1>(o, vlo, cvt8(s.a, 8));
    exp2_h<0>(s.b, a.mref);
    mma_pv_one<2>(o, vlo, cvt8(s.b, 0));
    exp2_h<8>(s.b, a.mref);
    mma_pv_one<3>(o, vlo, cvt8(s.b, 8));
    exp2_h<0>(s.c, a.mref);
    mma_pv_one<0>(o, vhi, cvt8(s.c, 0));
    exp2_h<8>(s.c, a.mref);
    mma_pv_one<1>(o, vhi, cvt8(s.c, 8));
    exp2_h<0>(s.d, a.mref);
    mma_pv_one<2>(o, vhi, cvt8(s.d, 0));
    exp2_h<8>(s.d, a.mref);
    mma_pv_one<3>(o, vhi, cvt8(s.d, 8));
    zk_setprio_lo();
    l_ = l_ * a.corr + col_sum4(s);
}

__device__ __forceinline__ void softmax_pv(o_tile &o, s4_tile &s, const st16 &vsm, float &m_, float &l_) {
    smx_pv(o, s, vsm, smx_prep(o, s, m_), l_);
}

__device__ __forceinline__ void qk_prefetch(s4_tile &att, const st16 &kcur, st16 &knx, st16 &vnx, const bf16 *gKn,
                                           const bf16 *gVn, int kv_stride, const q_tile &qT) {
    zero_s4(att);
    pin_s4(att);
    zk_iglp_qk();
    prefetch_k_pass<0>(knx, gKn, kv_stride);
    mma_qk_quad_c(att, kcur, 0, qT.d0);
    mma_qk_quad_c(att, kcur, 1, qT.d1);
    prefetch_k_pass<1>(knx, gKn, kv_stride);
    mma_qk_quad_c(att, kcur, 2, qT.d2);
    prefetch_v_pass<0>(vnx, gVn, kv_stride);
    mma_qk_quad_c(att, kcur, 3, qT.d3);
    prefetch_k_pass<2>(knx, gKn, kv_stride);
    mma_qk_quad_c(att, kcur, 4, qT.d4);
    prefetch_v_pass<1>(vnx, gVn, kv_stride);
    mma_qk_quad_c(att, kcur, 5, qT.d5);
    prefetch_k_pass<3>(knx, gKn, kv_stride);
    mma_qk_quad_c(att, kcur, 6, qT.d6);
    prefetch_v_pass<2>(vnx, gVn, kv_stride);
    mma_qk_quad_c(att, kcur, 7, qT.d7);
    prefetch_v_pass<3>(vnx, gVn, kv_stride);
    zk_qk_group_bf16_d128();
    zk_split_region();
}

// Bshd=false: BHSD [B,H,S,D], seq stride D (current benches).
// Bshd=true:  BSHD [B,S,H,D], seq stride H*D. D-axis still packed.
template <bool PairHeads, bool Bshd>
__global__ void __launch_bounds__(NT, ZK_LAUNCH_BLOCKS) attend_ker(const bf16 *Q, const bf16 *K, const bf16 *V, bf16 *O, float *Lse,
                                                    int M, int N, int Hq, int Hkv, float scale, int nxcd) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 128;
    const int q_stride = Bshd ? Hq * D : D;
    const int kv_stride = Bshd ? Hkv * D : D;
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

    const bf16 *Qh, *Kh, *Vh;
    bf16 *Oh;
    if constexpr (Bshd) {
        Qh = Q + ((size_t)batch_idx * M * Hq + (size_t)head_idx) * D;
        Kh = K + ((size_t)batch_idx * N * Hkv + (size_t)head_kv) * D;
        Vh = V + ((size_t)batch_idx * N * Hkv + (size_t)head_kv) * D;
        Oh = O + ((size_t)batch_idx * M * Hq + (size_t)head_idx) * D;
    } else {
        Qh = Q + ((size_t)batch_idx * Hq + head_idx) * M * D;
        Kh = K + ((size_t)batch_idx * Hkv + head_kv) * N * D;
        Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * D;
        Oh = O + ((size_t)batch_idx * Hq + head_idx) * M * D;
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

        q_tile qT;
        load_q(qT, Qh + (size_t)tile_idx * Q_BLOCK * q_stride, scale, q_stride);
        pin_q(qT);
        o_tile oR;
        zero_o(oR);
        pin_o(oR);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp)
            wg_bar();
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, kv_stride), kv_stride);
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, kv_stride), kv_stride);

        for (int kb = 0; kb < max_num_tiles; ++kb) {
            const int cur = kb & 1, nx = cur ^ 1;
            wg_bar();
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            s4_tile att;
            qk_prefetch(att, s.k[cur], s.k[nx], s.v[nx], kv_tile_ptr(Kh, kb_n, kv_stride),
                        kv_tile_ptr(Vh, kb_n, kv_stride), kv_stride, qT);
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s4(att, q_start, kb);
            softmax_pv(oR, att, s.v[cur], m_, l_);
        }

        store_o(Oh, q_start, M, oR, l_, q_stride);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const bf16 *Q, const bf16 *K, const bf16 *V, bf16 *O, float *Lse, int M, int N, int B, int Hq, int Hkv,
                float scale, bool bshd, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    if (pair_heads && bshd) {
        hipFuncSetAttribute((const void *)attend_ker<true, true>, hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem);
        attend_ker<true, true>
            <<<nq_pairs * B * nheads, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    } else if (pair_heads) {
        hipFuncSetAttribute((const void *)attend_ker<true, false>, hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem);
        attend_ker<true, false>
            <<<nq_pairs * B * nheads, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    } else if (bshd) {
        hipFuncSetAttribute((const void *)attend_ker<false, true>, hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem);
        attend_ker<false, true>
            <<<nq_pairs * B * nheads, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    } else {
        hipFuncSetAttribute((const void *)attend_ker<false, false>, hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem);
        attend_ker<false, false>
            <<<nq_pairs * B * nheads, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                          bool return_lse, bool bshd) {
    TORCH_CHECK(causal, "fmha_bf16 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "fmha_bf16 needs bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                bshd ? "q,k,v must be BSHD [B,S,H,D]" : "q,k,v must be BHSD [B,H,S,D]");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int B = q.size(0), D = q.size(3), Dv = v.size(3);
    const int Hq = bshd ? q.size(2) : q.size(1);
    const int M = bshd ? q.size(1) : q.size(2);
    const int Hkv = bshd ? k.size(2) : k.size(1);
    const int N = bshd ? k.size(1) : k.size(2);
    TORCH_CHECK(D == 128 && Dv == 128 && M % 256 == 0 && N % 128 == 0);
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
    run(reinterpret_cast<const bf16 *>(q.data_ptr()), reinterpret_cast<const bf16 *>(k.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()), reinterpret_cast<bf16 *>(O.data_ptr()), Lp, M, N, B, Hq, Hkv,
        (float)scale * L2E,
        bshd, at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_bf16_d128
