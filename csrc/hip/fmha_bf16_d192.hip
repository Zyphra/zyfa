// HK GQA causal + XCD pair, bf16 32x32x16, KN=64, streamed LDS K/V.
// One bar per tile (next iter's bar covers V lifetime). Research kernel. D=192 / Dv=128 / 8-wave / causal only. Do not fold into fmha32.hip.
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_bf16_d192.h"
#include "xcd_map.h"

namespace fmha_bf16_d192 {

using namespace hk;
using namespace hk192;

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

__device__ __forceinline__ const bf16 *kv_tile_ptr(const bf16 *base, int tile, int dim) {
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

__device__ __forceinline__ void softmax_pv(o_tile &o, s_tile &s, const st_v &vsm, float &m_, float &l_) {
    const float mx = col_max2(s);
    const float mnew = fmaxf(m_, mx);
    const float corr = (mnew > m_) ? exp2f(m_ - mnew) : 1.f;
    // o is known-zero before the first KV tile; do not broadcast corr=0
    // and multiply the entire output accumulator by zero.
    if (__builtin_expect(m_ != neg_inf(), 1))
        rescale_o(o, corr);
    exp2_m(s.a, mnew);
    p_tile p;
    p.a = cvt8(s.a, 0);
    p.b = cvt8(s.a, 8);
    __builtin_amdgcn_s_setprio(1);
    mma_pv_sl<0>(o, *reinterpret_cast<const st<64, 128> *>(&vsm), p);
    mma_pv_sl<1>(o, *reinterpret_cast<const st<64, 128> *>(&vsm), p);
    exp2_m(s.b, mnew);
    p.c = cvt8(s.b, 0);
    p.d = cvt8(s.b, 8);
    mma_pv_sl<2>(o, *reinterpret_cast<const st<64, 128> *>(&vsm), p);
    mma_pv_sl<3>(o, *reinterpret_cast<const st<64, 128> *>(&vsm), p);
    __builtin_amdgcn_s_setprio(0);
    l_ = l_ * corr + col_sum2(s);
    m_ = mnew;
}

template <bool PairHeads>
__global__ void __launch_bounds__(NT, 1) attend_ker(const bf16 *Q, const bf16 *K, const bf16 *V, float *O, float *Lse,
                                                    int M, int N, int Hq, int Hkv, float scale, int nxcd) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 192, Dv = 128;
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

    const bf16 *Qh = Q + ((size_t)batch_idx * Hq + head_idx) * M * D;
    const bf16 *Kh = K + ((size_t)batch_idx * Hkv + head_kv) * N * D;
    const bf16 *Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * Dv;
    float *Oh = O + ((size_t)batch_idx * Hq + head_idx) * M * Dv;
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

        q_tile192 qT;
        load_q(qT, Qh + (size_t)tile_idx * Q_BLOCK * D, scale);
        pin_q192(qT);
        o_tile oR;
        zero_o(oR);
        pin_o(oR);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp)
            wg_bar();
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, D));
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, Dv));

        for (int kb = 0; kb < max_num_tiles; ++kb) {
            const int cur = kb & 1, nx = cur ^ 1;
            if (kb == 0)
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            else
                asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
            wg_bar();
            uint32_t kbase = lds_u32(s.k[cur].data);
            asm volatile("" : "+v"(kbase));
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            const bf16 *gKn = kv_tile_ptr(Kh, kb_n, D);
            const bf16 *gVn = kv_tile_ptr(Vh, kb_n, Dv);
            s_tile att;
            zero_s(att);
            pin_s(att);
            prefetch_k_pass<0>(s.k[nx], gKn);
            mma_qk_pair(att, k_off(kbase, 0), qT.d0);
            mma_qk_pair(att, k_off(kbase, 1), qT.d1);
            mma_qk_pair(att, k_off(kbase, 2), qT.d2);
            mma_qk_pair(att, k_off(kbase, 3), qT.d3);
            prefetch_k_pass<1>(s.k[nx], gKn);
            mma_qk_pair(att, k_off(kbase, 4), qT.d4);
            mma_qk_pair(att, k_off(kbase, 5), qT.d5);
            mma_qk_pair(att, k_off(kbase, 6), qT.d6);
            mma_qk_pair(att, k_off(kbase, 7), qT.d7);
            prefetch_k_pass<2>(s.k[nx], gKn);
            mma_qk_pair(att, k_off(kbase, 8), qT.d8);
            prefetch_k_pass<3>(s.k[nx], gKn);
            prefetch_v_pass<0>(s.v[nx], gVn);
            mma_qk_pair(att, k_off(kbase, 9), qT.d9);
            mma_qk_pair(att, k_off(kbase, 10), qT.d10);
            prefetch_v_pass<1>(s.v[nx], gVn);
            mma_qk_pair(att, k_off(kbase, 11), qT.d11);
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s2(att, q_start, kb);
            softmax_pv(oR, att, s.v[cur], m_, l_);
        }

        store_o(Oh, q_start, M, oR, l_);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const bf16 *Q, const bf16 *K, const bf16 *V, float *O, float *Lse, int M, int N, int B, int Hq, int Hkv,
                float scale, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    if (pair_heads) {
        hipFuncSetAttribute((const void *)attend_ker<true>, hipFuncAttributeMaxDynamicSharedMemorySize, 131072);
        attend_ker<true><<<nq_pairs * B * (Hq / 2), NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    } else {
        hipFuncSetAttribute((const void *)attend_ker<false>, hipFuncAttributeMaxDynamicSharedMemorySize, 131072);
        attend_ker<false><<<nq_pairs * B * Hq, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale,
                                               bool causal, bool return_lse) {
    TORCH_CHECK(causal, "fmha_bf16_d192 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "fmha_bf16_d192 needs bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be [B,H,S,D]");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int B = q.size(0), Hq = q.size(1), M = q.size(2), D = q.size(3);
    const int Hkv = k.size(1), N = k.size(2), Dv = v.size(3);
    TORCH_CHECK(D == 192 && Dv == 128 && M % 256 == 0 && N % 64 == 0);
    TORCH_CHECK(Hq % Hkv == 0 && k.size(3) == D && v.size(1) == Hkv && v.size(2) == N);
    auto O = torch::empty({B, Hq, M, Dv}, q.options().dtype(at::kFloat));
    torch::Tensor Lse;
    float *Lp = nullptr;
    if (return_lse) {
        Lse = torch::empty({B * Hq, M}, q.options().dtype(at::kFloat));
        Lp = Lse.data_ptr<float>();
    }
    run(reinterpret_cast<const bf16 *>(q.data_ptr()), reinterpret_cast<const bf16 *>(k.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()), O.data_ptr<float>(), Lp, M, N, B, Hq, Hkv, (float)scale * L2E,
        at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_bf16_d192
