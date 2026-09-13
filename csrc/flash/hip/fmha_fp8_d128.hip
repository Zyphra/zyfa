// HK GQA causal + XCD pair, fp8 fat 32x32x64, KN=128, streamed LDS K/V.
// KV loop unrolled 8. QK stays at lgkmcnt(2); next-K under last PV aliases k_frag.
// One bar per tile (after softmax, before last PV). D=Dv=128 / 8-wave / causal only.
// RESCALE_SKIP: true (default) = HK Δmax≤8 delayed update. false = always rescale.
#ifndef RESCALE_SKIP
#define RESCALE_SKIP true
#endif
#ifndef ZK_EXP_SCHRAUDOLPH
#define ZK_EXP_SCHRAUDOLPH 1
#endif
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include "rt_fp8_d128.h"
#include "xcd_map.h"

namespace fmha_fp8_d128 {

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

__device__ __forceinline__ const unsigned char *kv_tile_ptr(const unsigned char *base, int tile, int seq_stride) {
    return base + (size_t)tile * KV_BLOCK * seq_stride;
}

// AITER fwd_hd128_fp8_causal.co: no Δm skip. After the 64 score v_exp,
// corr = exp2(scale*(m_old-m_new)) with v_cndmask(delta, 0) on the
// first-tile sentinel, then l *= corr, sum P, 64 v_mul on O, then PV.
// Always issue the O muls (corr=1 is exp2(0)); no __all / __any.
__device__ __forceinline__ void rescale_o_aiter(o_tile &o, float corr) {
    o.a = mulsc(o.a, corr);
    o.b = mulsc(o.b, corr);
    o.c = mulsc(o.c, corr);
    o.d = mulsc(o.d, corr);
}

__device__ __forceinline__ void softmax_pv(o_tile &o, s4_tile &s, uint32_t vbase, float scale, float &m_, float &l_,
                                          k_frag &kf, uint32_t k_next_off, bool have_mx_ab, float mx_ab) {
    const float mx = have_mx_ab ? col_max4_sc_finish(s, mx_ab, scale) : col_max4_sc(s, scale);
    const float mnew = fmaxf(m_, mx);
#if RESCALE_SKIP
    const float dmax = mnew - m_;
    const int skip = __all(dmax <= RESCALE_THRESHOLD);
    float corr = 1.f;
    float mref = m_;
    if (!__builtin_expect(skip, 1)) {
        // AITER corr: cndmask first-tile delta to 0 → exp2(0)=1.
        // O *= corr runs after the score exps, same as their .co.
        float delta = m_ - mnew;
        delta = (m_ == neg_inf()) ? 0.f : delta;
        corr = zk_exp2(delta);
        mref = mnew;
        m_ = mnew;
    }
#else
    // m_ is already scaled. AITER keeps raw max and multiplies by s26;
    // same corr = exp2(m_old - m_new) in this convention.
    float delta = m_ - mnew;
    delta = (m_ == neg_inf()) ? 0.f : delta;
    const float corr = zk_exp2(delta);
    const float mref = mnew;
    m_ = mnew;
#endif
    // Current V is already in LDS (prior tile bar). First 4 PV run under
    // exp(c,d); vmcnt+bar only before next-K issue in the last 4 PV.
    pv_v_head vh = preissue_pv_v(vbase, 0);
    zk_iglp_smx();
    exp2_sc(s.a, scale, mref);
    exp2_sc(s.b, scale, mref);
    const float sum_ab = sum2_local(s.a, s.b);
    const i32x8 p_lo = pack_p_consume(s.a, s.b);
    // Force exp(c,d) before the first 4 PV so VALU sits under those MFMAs.
    // Source order was PV then exp; setprio around PV blocked the overlap.
    exp2_sc(s.c, scale, mref);
    exp2_sc(s.d, scale, mref);
    l_ = l_ * corr + xor32_sum(sum_ab + sum2_local(s.c, s.d));
#if RESCALE_SKIP
    if (!__builtin_expect(skip, 1))
        rescale_o_aiter(o, corr);
#else
    rescale_o_aiter(o, corr);
#endif
    zk_setprio_hi();
    mma_pv_lds_preissued(o, vbase, p_lo, 0, vh);
    zk_setprio_lo();
    const i32x8 p_hi = pack_p_consume(s.c, s.d);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    wg_bar();
    zk_setprio_hi();
    mma_pv_lds_khead(o, vbase, p_hi, 2, kf, k_next_off);
    zk_setprio_lo();
}

template <bool PairHeads>
__global__ void __launch_bounds__(NT, ZK_LAUNCH_BLOCKS) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, bf16 *O, float *Lse, int M, int N,
                                                    int Hq, int Hkv, float scale, int nxcd, int q_stride,
                                                    int kv_stride, int bshd, int n_head_units) {
    extern __shared__ Smem sm[];
    Smem &s = sm[0];

    constexpr int D = 128;
    const int GROUP = Hq / Hkv;
    constexpr int QW = PairHeads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + QW - 1) / QW;
    const int nq_pairs = (numQ + 1) / 2;
    const int n_jobs = nq_pairs * n_head_units;
    const int stride = (int)gridDim.x;
    const int num_tiles = N / KV_BLOCK;

    for (int job = (int)blockIdx.x; job < n_jobs; job += stride) {
    if (job != (int)blockIdx.x) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        wg_bar();
    }
    int qx, hu;
    fmha_xcd_map(nq_pairs, n_head_units, nxcd, qx, hu, job);

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

    const size_t q_sel = bshd ? ((size_t)batch_idx * M * Hq + (size_t)head_idx) * D
                              : ((size_t)batch_idx * Hq + head_idx) * (size_t)M * D;
    const size_t kv_sel = bshd ? ((size_t)batch_idx * N * Hkv + (size_t)head_kv) * D
                               : ((size_t)batch_idx * Hkv + head_kv) * (size_t)N * D;
    const unsigned char *Qh = Q + q_sel;
    const unsigned char *Kh = K + kv_sel;
    const unsigned char *Vh = V + kv_sel;
    bf16 *Oh = O + q_sel;
    float *Lh = Lse ? Lse + ((size_t)batch_idx * Hq + head_idx) * M : nullptr;

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

        q8_tile qT;
        o_tile oR;
        k_frag kf;
        zero_o(oR);
        load_q8(qT, Qh + (size_t)tile_idx * Q_BLOCK * q_stride, q_stride);
        pin_q8(qT);
        float m_ = neg_inf(), l_ = 0.f;

        if (pp) {
            // Pass 0's last G2S targets s.k[0] when tile count is even.
            // Drain it before pass 1 prefetches tile 0 into the same slot.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            wg_bar();
        }
        prefetch_k(s.k[0], kv_tile_ptr(Kh, 0, kv_stride), kv_stride);
        prefetch_v(s.v[0], kv_tile_ptr(Vh, 0, kv_stride), kv_stride);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        wg_bar();
        issue_k_pair<0>(kf, k_off(lds_u32(s.k[0].data), 0));
        pin_kf(kf);

        auto one_tile = [&](int kb) {
            if constexpr (PairHeads) {
                // Two 4-wave groups share K/V. Drain this wave's G2S before
                // the other group reads the ping-pong slot.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                wg_bar();
            }
            const int cur = kb & 1, nx = cur ^ 1;
            const uint32_t kbase = lds_u32(s.k[cur].data);
            const uint32_t vbase = lds_u32(s.v[cur].data);
            const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
            const unsigned char *gKn = kv_tile_ptr(Kh, kb_n, kv_stride);
            const unsigned char *gVn = kv_tile_ptr(Vh, kb_n, kv_stride);
            s4_tile att;
            const uint32_t off0 = k_off(kbase, 0), off1 = k_off(kbase, 1);
            const uint32_t k_next_off = k_off(lds_u32(s.k[nx].data), 0);
            zk_iglp_qk();
            if (warpid() < 4)
                zk_setprio_hi();
            else
                zk_setprio_lo();
            prefetch_k_pass<0>(s.k[nx], gKn, kv_stride);
            qk_step<false, 8192>(att, kf, qT.d0, 0, off0);
            prefetch_k_pass<1>(s.k[nx], gKn, kv_stride);
            qk_step<false, 0>(att, kf, qT.d0, 2, off1);
            prefetch_v_pass<0>(s.v[nx], gVn, kv_stride);
            qk_step<true, 8192>(att, kf, qT.d1, 0, off1);
            prefetch_v_pass<1>(s.v[nx], gVn, kv_stride);
            const bool causal_tile = q_start < (kb + 1) * KV_BLOCK;
            float mx_ab = neg_inf();
            qk_tail<true, true>(att, kf, qT.d1, 2, &mx_ab);
            zk_qk_group_fp8_d128();
            zk_split_region();
            if (__builtin_expect(causal_tile, 0))
                mask_s4(att, q_start, kb);
            // Causal tiles must re-reduce after mask (pre-mask max includes
            // future keys). Hot tiles finish c,d on top of the folded a,b max.
            softmax_pv(oR, att, vbase, scale, m_, l_, kf, k_next_off, !causal_tile, mx_ab);
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

        store_o_fp8_qrow(Oh, q_start, M, oR, l_, q_stride);
        if (Lh) {
            const float lse = m_ * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, bf16 *O, float *Lse,
                int M, int N, int B, int Hq, int Hkv, float scale, bool bshd, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    constexpr int D = 128;
    const int q_stride = bshd ? Hq * D : D;
    const int kv_stride = bshd ? Hkv * D : D;
    const int bshd_i = bshd ? 1 : 0;
    bool split = B > 1 && M >= 8192;
    if (const char *e = std::getenv("ZK_FMHA_FUSE_B"); e && std::atoi(e) == 1)
        split = false;
    const int batches = split ? B : 1;
    const int n_head_units = nheads * (split ? 1 : B);
    const int grid = fmha_persist_grid(nq_pairs * n_head_units, nxcd);
    const size_t q_batch = bshd ? (size_t)M * Hq * D : (size_t)Hq * M * D;
    const size_t kv_batch = bshd ? (size_t)N * Hkv * D : (size_t)Hkv * N * D;
    if (pair_heads)
        hipFuncSetAttribute((const void *)attend_ker<true>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    else
        hipFuncSetAttribute((const void *)attend_ker<false>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    // S>=8K: one B=1 grid per batch (fused B>1 thrashes XCD L2). Short S keeps one grid.
    for (int b = 0; b < batches; ++b) {
        const unsigned char *Qb = Q + b * q_batch;
        const unsigned char *Kb = K + b * kv_batch;
        const unsigned char *Vb = V + b * kv_batch;
        bf16 *Ob = O + b * q_batch;
        float *Lb = Lse ? Lse + (size_t)b * Hq * M : nullptr;
        if (pair_heads)
            attend_ker<true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, q_stride,
                                                         kv_stride, bshd_i, n_head_units);
        else
            attend_ker<false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, q_stride,
                                                          kv_stride, bshd_i, n_head_units);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                         bool return_lse, bool bshd) {
    TORCH_CHECK(causal, "fmha_fp8 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "fmha_fp8 needs float8_e4m3fn");
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
    run((const unsigned char *)q.data_ptr(), (const unsigned char *)k.data_ptr(),
        (const unsigned char *)v.data_ptr(), reinterpret_cast<bf16 *>(O.data_ptr()), Lp, M, N, B, Hq, Hkv,
        (float)scale * L2E, bshd,
        at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_fp8_d128
