// HK GQA causal + XCD pair, fp8 fat 32x32x64, KN=64, streamed LDS K/V.
// D=192 / Dv=128 / 8-wave / causal only.
// V issue under last QK; col_sum under late PV.
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_fp8_d192.h"
#include "xcd_map.h"

namespace fmha_fp8_d192 {

using namespace hk;
using namespace hk8_192;
using hk8::i32x8;
using hk8::v2i32;
using hk8::kh_mma;
using hk8::pack_p;
using hk8::pack_v2x4;
using hk8::ds_tr8;

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

struct v8 {
    v2i32 a0, a1, a2, a3, b0, b1, b2, b3;
};

__device__ __forceinline__ const unsigned char *kv_tile_ptr(const unsigned char *base, int tile, int dim) {
    return base + (size_t)tile * KV_BLOCK * dim;
}

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    if (!__any(corr != 1.f))
        return;
    const int gb = laneid() >> 5;
    float ct[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        ct[r] = __shfl(corr, fmha_qoff(r, gb));
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        o.a[r] *= ct[r];
        o.b[r] *= ct[r];
        o.c[r] *= ct[r];
        o.d[r] *= ct[r];
    }
}

__device__ __forceinline__ void issue_v8(uint32_t vbase, v8 &v) {
    v.a0 = ds_tr8(v_tr8_off(vbase, 0, 0, 0));
    v.a1 = ds_tr8(v_tr8_off(vbase, 0, 0, 1));
    v.a2 = ds_tr8(v_tr8_off(vbase, 0, 1, 0));
    v.a3 = ds_tr8(v_tr8_off(vbase, 0, 1, 1));
    v.b0 = ds_tr8(v_tr8_off(vbase, 1, 0, 0));
    v.b1 = ds_tr8(v_tr8_off(vbase, 1, 0, 1));
    v.b2 = ds_tr8(v_tr8_off(vbase, 1, 1, 0));
    v.b3 = ds_tr8(v_tr8_off(vbase, 1, 1, 1));
    asm volatile("" : "+v"(v.a0), "+v"(v.a1), "+v"(v.a2), "+v"(v.a3), "+v"(v.b0), "+v"(v.b1), "+v"(v.b2),
                      "+v"(v.b3)::"memory");
}

__device__ __forceinline__ void softmax_pv(o_tile &o, s2_tile &s, v8 &v, uint32_t vbase, float scale, float &m_,
                                          float &l_) {
    const float mx = col_max2_sc(s, scale);
    const float mnew = fmaxf(m_, mx);
    const float corr = (mnew > m_) ? exp2f(m_ - mnew) : 1.f;
    rescale_o(o, corr);
    hk8::exp2_sc(s.a, scale, mnew);
    __builtin_amdgcn_s_setprio(1);
    hk8::exp2_sc(s.b, scale, mnew);
    const i32x8 ps = pack_p(s.a, s.b);
    const float ladd = col_sum2(s);

    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(v.a0), "+v"(v.a1), "+v"(v.a2), "+v"(v.a3), "+v"(v.b0), "+v"(v.b1), "+v"(v.b2), "+v"(v.b3));
    o.a = kh_mma(ps, pack_v2x4(v.a0, v.a1, v.a2, v.a3), o.a);
    v2i32 c0 = ds_tr8(v_tr8_off(vbase, 2, 0, 0)), c1 = ds_tr8(v_tr8_off(vbase, 2, 0, 1));
    v2i32 c2 = ds_tr8(v_tr8_off(vbase, 2, 1, 0)), c3 = ds_tr8(v_tr8_off(vbase, 2, 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(v.b0), "+v"(v.b1), "+v"(v.b2), "+v"(v.b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma(ps, pack_v2x4(v.b0, v.b1, v.b2, v.b3), o.b);
    v2i32 d0 = ds_tr8(v_tr8_off(vbase, 3, 0, 0)), d1 = ds_tr8(v_tr8_off(vbase, 3, 0, 1));
    v2i32 d2 = ds_tr8(v_tr8_off(vbase, 3, 1, 0)), d3 = ds_tr8(v_tr8_off(vbase, 3, 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma(ps, pack_v2x4(c0, c1, c2, c3), o.c);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.d = kh_mma(ps, pack_v2x4(d0, d1, d2, d3), o.d);
    __builtin_amdgcn_s_setprio(0);
    l_ = l_ * corr + ladd;
    m_ = mnew;
}

__global__ void __launch_bounds__(NT, 1) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, float *O, float *Lse, int M, int N, int Hq,
                                                    int Hkv, float scale, int nxcd) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 192, Dv = 128;
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
    const unsigned char *Vh = V + ((size_t)batch_idx * Hkv + head_kv) * N * Dv;
    float *Oh = O + ((size_t)batch_idx * Hq + head_idx) * M * Dv;
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

        q8_tile192 qT;
        o_tile oR;
        zero_o(oR);
        load_q8(qT, Qh + (size_t)tile_idx * Q_BLOCK * D);
        pin_q192(qT);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp)
            wg_bar();
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, D));
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, Dv));

        for (int kb = 0; kb < max_num_tiles; ++kb) {
            const int cur = kb & 1, nx = cur ^ 1;
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            wg_bar();
            const uint32_t kbase = lds_u32(s.k[cur].data);
            const uint32_t vbase = lds_u32(s.v[cur].data);
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            const unsigned char *gKn = kv_tile_ptr(Kh, kb_n, D);
            const unsigned char *gVn = kv_tile_ptr(Vh, kb_n, Dv);
            s2_tile att;
            v8 vv;
            prefetch_k_pass<0>(s.k[nx], gKn);
            mma_qk_pair<false>(att, k_off(kbase, 0, 0), k_off(kbase, 0, 1), qT.d0);
            prefetch_k_pass<1>(s.k[nx], gKn);
            mma_qk_pair<true>(att, k_off(kbase, 1, 0), k_off(kbase, 1, 1), qT.d1);
            prefetch_v(s.v[nx], gVn);
            issue_v8(vbase, vv);
            mma_qk_pair<true>(att, k_off(kbase, 2, 0), k_off(kbase, 2, 1), qT.d2);
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s2(att, q_start, kb);
            pin_s2(att);
            softmax_pv(oR, att, vv, vbase, scale, m_, l_);
            wg_bar();
        }

        store_o_fp8(Oh, q_start, M, oR, l_);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, float *O, float *Lse, int M,
                int N, int B, int Hq, int Hkv, float scale, hipStream_t stream) {
    const int numQ = (M / Q_BLOCK + NW - 1) / NW;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    hipFuncSetAttribute((const void *)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attend_ker<<<nq_pairs * B * Hq, NT, smem, stream>>>(Q, K, V, O, Lse, M, N, Hq, Hkv, scale, nxcd);
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale,
                                              bool causal, bool return_lse) {
    TORCH_CHECK(causal, "fmha_fp8_d192 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "fmha_fp8_d192 needs float8_e4m3fn");
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
    run((const unsigned char *)q.data_ptr(), (const unsigned char *)k.data_ptr(),
        (const unsigned char *)v.data_ptr(), O.data_ptr<float>(), Lp, M, N, B, Hq, Hkv, (float)scale * L2E,
        at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_fp8_d192
