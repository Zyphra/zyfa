// HK GQA causal + XCD pair, bf16 32x32x16, KN=64, streamed LDS K/V.
// One bar per tile (next iter's bar covers V lifetime). Research kernel. D=Dv=128 / 8-wave / causal only. Do not fold into fmha32.hip.
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_bf16_d128.h"
#include "xcd_map.h"

namespace fmha_bf16_d128 {

using namespace hk;
using namespace hk16;

constexpr int Q_BLOCK = 32;
constexpr int KV_BLOCK = 64;
constexpr int NW = 8;
constexpr int NT = 512;
constexpr float L2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ __attribute__((convergent)) void wg_bar() { __syncthreads(); }

struct alignas(16) Smem {
    st16 k[2];
    st16 v[2];
};

__device__ __forceinline__ const bf16 *kv_tile_ptr(const bf16 *base, int tile, int D) {
    return base + (size_t)tile * KV_BLOCK * D;
}

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    if (!__any(corr != 1.f))
        return;
    o.a = mulsc(o.a, corr);
    o.b = mulsc(o.b, corr);
    o.c = mulsc(o.c, corr);
    o.d = mulsc(o.d, corr);
}

__device__ __forceinline__ void softmax_pv(o_tile &o, s_tile &s, const st16 &vsm, float &m_, float &l_) {
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

__global__ void __launch_bounds__(NT, 1) attend_ker(const bf16 *Q, const bf16 *K, const bf16 *V, float *O, float *Lse,
                                                    int M, int N, int Hq, int Hkv, float scale, int nxcd) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 128;
    const int GROUP = Hq / Hkv;
    const int numQ = (M / Q_BLOCK + NW - 1) / NW;
    const int nq_pairs = (numQ + 1) / 2;
    const int n_bh = (int)gridDim.x / nq_pairs;
    int qx, bh;
    fmha_xcd_map(nq_pairs, n_bh, nxcd, qx, bh);

    const int batch_idx = bh / Hq;
    const int h_lin = bh % Hq;
    const int head_idx = (h_lin % Hkv) * GROUP + (h_lin / Hkv);
    const int head_kv = head_idx / GROUP;

    const bf16 *Qh = Q + ((size_t)batch_idx * Hq + head_idx) * M * D;
    const bf16 *Kh = K + ((size_t)batch_idx * Hkv + head_kv) * N * D;
    const bf16 *Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * D;
    float *Oh = O + ((size_t)batch_idx * Hq + head_idx) * M * D;
    float *Lh = Lse ? Lse + ((size_t)batch_idx * Hq + head_idx) * M : nullptr;

    const int num_tiles = N / KV_BLOCK;
    const int q_lo = qx, q_hi = numQ - 1 - qx;

    for (int pp = 0; pp < 2; ++pp) {
        const int qblk = pp ? q_lo : q_hi;
        if (pp && q_lo == q_hi)
            break;
        const int tile_idx = qblk * NW + warpid();
        const int max_tile_idx = qblk * NW + NW - 1;
        const int max_q_end = (max_tile_idx + 1) * Q_BLOCK;
        int max_num_tiles = (max_q_end + KV_BLOCK - 1) / KV_BLOCK;
        if (max_num_tiles > num_tiles)
            max_num_tiles = num_tiles;
        const int q_start = tile_idx * Q_BLOCK;

        q_tile qT;
        load_q(qT, Qh + (size_t)tile_idx * Q_BLOCK * D, scale);
        pin_q(qT);
        o_tile oR;
        zero_o(oR);
        pin_o(oR);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp)
            wg_bar();
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, D));
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, D));

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
            const bf16 *gVn = kv_tile_ptr(Vh, kb_n, D);
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
            prefetch_v_pass<0>(s.v[nx], gVn);
            mma_qk_pair(att, k_off(kbase, 5), qT.d5);
            mma_qk_pair(att, k_off(kbase, 6), qT.d6);
            prefetch_v_pass<1>(s.v[nx], gVn);
            mma_qk_pair(att, k_off(kbase, 7), qT.d7);
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
    const int numQ = (M / Q_BLOCK + NW - 1) / NW;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    hipFuncSetAttribute((const void *)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attend_ker<<<nq_pairs * B * Hq, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                          bool return_lse) {
    TORCH_CHECK(causal, "fmha_bf16 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "fmha_bf16 needs bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be [B,H,S,D]");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int B = q.size(0), Hq = q.size(1), M = q.size(2), D = q.size(3);
    const int Hkv = k.size(1), N = k.size(2), Dv = v.size(3);
    TORCH_CHECK(D == 128 && Dv == 128 && M % 256 == 0 && N % 64 == 0);
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

} // namespace fmha_bf16_d128
