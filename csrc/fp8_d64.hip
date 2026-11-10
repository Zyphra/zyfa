// FP8 (e4m3) causal forward attention, d_qk = d_v = 64. Same lean P, per-block scale, and row-sum MFMA as
// fp8_d128.hip. One 32x32x64 MFMA covers D, and PV writes 64 outputs. This is the kernel flash_attn_func calls.
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "descale.h"
#include "rt_fp8_d64.h"
#include "xcd_map.h"

namespace fmha_fp8_d64 {

using namespace hk;
using namespace hk8_64;

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

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    o.a = mulsc(o.a, corr);
    o.b = mulsc(o.b, corr);
    asm volatile("s_nop 1" : "+v"(o.a), "+v"(o.b));
}

// Online softmax and PV of one 128-key tile. Each 64-key half gets its own P block scale; l is summed by MFMA
// from the same rounded P as O.
__device__ __forceinline__ void softmax_pv(o_tile &o, s4_tile &s, uint32_t vbase, float scale, float &m_, float &l_,
                                          const i32x8 &sel, k_frag &kf, uint32_t k_next_off, bool have_mx_ab,
                                          float mx_ab, const pv_v_head &vh) {
    const float lo_raw = xor32_max(have_mx_ab ? mx_ab : lane_max2(s.a, s.b));
    const float hi_raw = xor32_max(lane_max2(s.c, s.d));
    const float mx = fmaxf(lo_raw, hi_raw) * scale;
    const float mnew = fmaxf(m_, mx);
    const float dmax = mnew - m_;
    const int skip = __all(dmax <= LEAN_THR);
    float corr = 1.f;
    float mref = m_;
    if (!__builtin_expect(skip, 1)) {
        float delta = m_ - mnew;
        delta = (m_ == neg_inf()) ? 0.f : delta;
        corr = exp2_hw(delta);
        mref = mnew;
        m_ = mnew;
    }
    const int off_lo = block_off(lo_raw, scale, mref), off_hi = block_off(hi_raw, scale, mref);
    const int sb_lo = 127 + off_lo, sb_hi = 127 + off_hi;
    const float c8 = 8.f * scale;
    const float b8 = __builtin_fmaf(-8.f, mref, 8.f * LEAN_SHIFT + 56.f - LEAN_BIAS);
    const float b8_lo = __builtin_fmaf(-8.f, (float)off_lo, b8), b8_hi = __builtin_fmaf(-8.f, (float)off_hi, b8);
    iglp_softmax();
    p_code(s.a, c8, b8_lo);
    p_code(s.b, c8, b8_lo);
    const i32x8 p_lo = pack_code_consume(s.a, s.b);
    p_code(s.c, c8, b8_hi);
    p_code(s.d, c8, b8_hi);
    if (!__builtin_expect(skip, 1)) {
        rescale_o(o, corr);
        l_ *= corr;
    }
    mma_pv_lds_preissued(o, vbase, p_lo, 0, vh, sb_lo);
    const float l_lo = block_rowsum(sel, p_lo, sb_lo);
    const i32x8 p_hi = pack_code_consume(s.c, s.d);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    wg_bar();
    mma_pv_lds_khead(o, vbase, p_hi, 2, kf, k_next_off, sb_hi);
    l_ += l_lo + block_rowsum(sel, p_hi, sb_hi);
}

template <bool PairHeads>
__global__ void __launch_bounds__(NT, 1) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, bf16 *O, float *Lse, int M, int N,
                                                    int Hq, int Hkv, float scale, int nxcd, int q_stride,
                                                    int kv_stride, int bshd, int n_head_units, const float *qd,
                                                    const float *kd, const float *vd, float v_scale) {
    extern __shared__ Smem sm[];
    float qk = (qd ? *qd : 1.f) * (kd ? *kd : 1.f);
    asm volatile("" : "+v"(qk));
    scale *= qk;
    float ov = (vd ? *vd : 1.f) * v_scale;
    asm volatile("" : "+v"(ov));
    Smem &s = sm[0];

    constexpr int D = 64;
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
            const i32x8 sel = rowsum_sel();

            if (pp) {
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
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                wg_bar();
                const int cur = kb & 1, nx = cur ^ 1;
                const uint32_t kbase = lds_u32(s.k[cur].data);
                const uint32_t vbase = lds_u32(s.v[cur].data);
                const int kb_n = (kb + 1 < max_num_tiles) ? kb + 1 : kb;
                const unsigned char *gKn = kv_tile_ptr(Kh, kb_n, kv_stride);
                const unsigned char *gVn = kv_tile_ptr(Vh, kb_n, kv_stride);
                s4_tile att;
                const uint32_t off0 = k_off(kbase, 0);
                const uint32_t k_next_off = k_off(lds_u32(s.k[nx].data), 0);
                iglp_qk();
                // D = 64 is one MFMA. Base 4096 is keys 64-127 of this tile.
                prefetch_k_pass<0>(s.k[nx], gKn, kv_stride);
                qk_step<false, 4096>(att, kf, qT.d0, 0, off0);
                prefetch_v_pass<0>(s.v[nx], gVn, kv_stride);
                const bool causal_tile = q_start < (kb + 1) * KV_BLOCK;
                float mx_ab = neg_inf();
                qk_tail<false, true>(att, kf, qT.d0, 2, &mx_ab);
                const pv_v_head vh = preissue_pv_v(vbase, 0);
                mfma_rd_fence(att.c, att.d);
                if (__builtin_expect(causal_tile, 0))
                    mask_s4(att, q_start, kb);
                softmax_pv(oR, att, vbase, scale, m_, l_, sel, kf, k_next_off, !causal_tile, mx_ab, vh);
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

            drain_kf(kf);
            mfma_rd_fence(oR);
            const float m_lse = m_ - LEAN_SHIFT;
            store_o_fp8_qrow(Oh, q_start, M, oR, l_, ov, q_stride);
            if (Lh) {
                const float lse = m_lse * LN2 + logf(l_);
                store_lse(Lh, q_start, M, lse);
            }
        }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, bf16 *O, float *Lse,
                int M, int N, int B, int Hq, int Hkv, float scale, bool bshd, const float *qd, const float *kd,
                const float *vd, float v_scale, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    constexpr int D = 64;
    const int q_stride = bshd ? Hq * D : D;
    const int kv_stride = bshd ? Hkv * D : D;
    const int bshd_i = bshd ? 1 : 0;
    const bool split = fmha_split_batch(B, nq_pairs, nheads, nxcd, M);
    const int batches = split ? B : 1;
    const int n_head_units = nheads * (split ? 1 : B);
    const int grid = fmha_persist_grid(nq_pairs * n_head_units, nxcd);
    const size_t q_batch = bshd ? (size_t)M * Hq * D : (size_t)Hq * M * D;
    const size_t kv_batch = bshd ? (size_t)N * Hkv * D : (size_t)Hkv * N * D;
    if (pair_heads)
        hip_check(hipFuncSetAttribute((const void *)attend_ker<true>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    else
        hip_check(hipFuncSetAttribute((const void *)attend_ker<false>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    for (int b = 0; b < batches; ++b) {
        const unsigned char *Qb = Q + b * q_batch;
        const unsigned char *Kb = K + b * kv_batch;
        const unsigned char *Vb = V + b * kv_batch;
        bf16 *Ob = O + b * q_batch;
        float *Lb = Lse ? Lse + (size_t)b * Hq * M : nullptr;
        if (pair_heads)
            attend_ker<true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, q_stride,
                                                         kv_stride, bshd_i, n_head_units, qd, kd, vd, v_scale);
        else
            attend_ker<false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, q_stride,
                                                          kv_stride, bshd_i, n_head_units, qd, kd, vd, v_scale);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                   bool return_lse, bool bshd, double v_scale,
                                   c10::optional<torch::Tensor> q_descale, c10::optional<torch::Tensor> k_descale,
                                   c10::optional<torch::Tensor> v_descale) {
    TORCH_CHECK(causal, "fmha_fp8 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q, k, v must be on the same device");
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && k.scalar_type() == at::kFloat8_e4m3fn &&
                    v.scalar_type() == at::kFloat8_e4m3fn,
                "fmha_fp8 needs float8_e4m3fn");
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
    TORCH_CHECK(D == 64 && Dv == 64 && M % 256 == 0 && N % 128 == 0);
    TORCH_CHECK(M == N && scale > 0, "fmha_fp8 needs M == N and scale > 0");
    TORCH_CHECK(fmha_scale_fp8_ok(scale), "fmha_fp8 needs a positive normal fp32 scale");
    fmha_check_v_scale(v_scale);
    const float *qd = fmha_descale_ptr(q_descale, q, "q_descale"), *kd = fmha_descale_ptr(k_descale, q, "k_descale"),
                *vd = fmha_descale_ptr(v_descale, q, "v_descale");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0 && k.size(3) == D && q.size(0) == k.size(0) && q.size(0) == v.size(0));
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
    if (B == 0 || Hq == 0 || M == 0)
        return return_lse ? std::vector<torch::Tensor>{O, Lse} : std::vector<torch::Tensor>{O};
    run((const unsigned char *)q.data_ptr(), (const unsigned char *)k.data_ptr(),
        (const unsigned char *)v.data_ptr(), reinterpret_cast<bf16 *>(O.data_ptr()), Lp, M, N, B, Hq, Hkv,
        (float)scale * L2E, bshd, qd, kd, vd, (float)v_scale, at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_fp8_d64
