// HK GQA causal + XCD pair, fp8 fat 32x32x64, KN=128, streamed LDS K/V.
// Research kernel. D=Dv=128 / 8-wave / causal only. Do not fold into fmha_fp8.hip.
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "hk_rt_fp8_d128.h"
#include "xcd_map.h"

namespace fmha32_hkfp8 {

using namespace hk;
using namespace hk8;

constexpr int Q_BLOCK = 32;
constexpr int KV_BLOCK = 128;
constexpr int NW = 8;
constexpr int NT = 512;
constexpr float L2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ __attribute__((convergent)) void wg_bar() { __syncthreads(); }

struct alignas(16) Smem {
    st8 k[2];
    st8 v[2];
};

__device__ __forceinline__ const unsigned char *kv_tile_ptr(const unsigned char *base, int tile, int D) {
    return base + (size_t)tile * KV_BLOCK * D;
}

__global__ void __launch_bounds__(NT, 1) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, float *O, float *Lse, int M, int N,
                                                    int Hq, int Hkv, float scale, int nxcd) {
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

    const unsigned char *Qh = Q + ((size_t)batch_idx * Hq + head_idx) * M * D;
    const unsigned char *Kh = K + ((size_t)batch_idx * Hkv + head_kv) * N * D;
    const unsigned char *Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * D;
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

        q8_tile qT;
        o_tile oR;
        zero_o(oR);
        load_q8(qT, Qh + (size_t)tile_idx * Q_BLOCK * D);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp)
            wg_bar();
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, D));
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, D));

        for (int kb = 0; kb < max_num_tiles; ++kb) {
            const int cur = kb & 1, nx = cur ^ 1;
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            wg_bar();
            const uint32_t kbase = lds_u32(s.k[cur].data);
            const uint32_t vbase = lds_u32(s.v[cur].data);
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            const unsigned char *gKn = kv_tile_ptr(Kh, kb_n, D);
            const unsigned char *gVn = kv_tile_ptr(Vh, kb_n, D);
            s4_tile att;
            const uint32_t off0 = k_off(kbase, 0), off1 = k_off(kbase, 1);
            prefetch_k_pass<0>(s.k[nx], gKn);
            mma_qk_pair<false>(att, off0, qT.d0, 0);
            prefetch_k_pass<1>(s.k[nx], gKn);
            mma_qk_pair<false>(att, off0, qT.d0, 2);
            prefetch_v_pass<0>(s.v[nx], gVn);
            mma_qk_pair<true>(att, off1, qT.d1, 0);
            prefetch_v_pass<1>(s.v[nx], gVn);
            mma_qk_pair<true>(att, off1, qT.d1, 2);
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s4(att, q_start, kb);

            const float mx = col_max4_sc(att, scale);
            const float mnew = fmaxf(m_, mx);
            const float corr = exp2f(m_ - mnew);
            rescale_o(oR, corr);
            exp2_sc(att.a, scale, mnew);
            exp2_sc(att.b, scale, mnew);
            __builtin_amdgcn_s_setprio(1);
            mma_pv_lds(oR, vbase, pack_p(att.a, att.b), 0);
            exp2_sc(att.c, scale, mnew);
            exp2_sc(att.d, scale, mnew);
            mma_pv_lds(oR, vbase, pack_p(att.c, att.d), 2);
            __builtin_amdgcn_s_setprio(0);
            l_ = l_ * corr + col_sum4(att);
            m_ = mnew;
            wg_bar();
        }

        store_o_fp8(Oh, q_start, M, oR, l_);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, float *O, float *Lse,
                int M, int N, int B, int Hq, int Hkv, float scale, hipStream_t stream) {
    const int numQ = (M / Q_BLOCK + NW - 1) / NW;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    hipFuncSetAttribute((const void *)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attend_ker<<<nq_pairs * B * Hq, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
}

std::vector<torch::Tensor> forward_hkfp8(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale,
                                         bool causal, bool return_lse) {
    TORCH_CHECK(causal, "fmha32_hkfp8 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "fmha32_hkfp8 needs float8_e4m3fn");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q,k,v must be [B,H,S,D]");
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int B = q.size(0), Hq = q.size(1), M = q.size(2), D = q.size(3);
    const int Hkv = k.size(1), N = k.size(2), Dv = v.size(3);
    TORCH_CHECK(D == 128 && Dv == 128 && M % 256 == 0 && N % 128 == 0);
    TORCH_CHECK(Hq % Hkv == 0 && k.size(3) == D && v.size(1) == Hkv && v.size(2) == N);
    auto O = torch::empty({B, Hq, M, Dv}, q.options().dtype(at::kFloat));
    torch::Tensor Lse;
    float *Lp = nullptr;
    if (return_lse) {
        Lse = torch::empty({B * Hq, M}, q.options().dtype(at::kFloat));
        Lp = Lse.data_ptr<float>();
    }
    run((const unsigned char *)q.data_ptr(), (const unsigned char *)k.data_ptr(),
        (const unsigned char *)v.data_ptr(), O.data_ptr<float>(), Lp, M, N, B, Hq, Hkv, (float)scale * L2E,
        at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha32_hkfp8
