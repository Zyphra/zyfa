// FP8 (e4m3) causal forward attention for MLA, d_qk = 192 / d_v = 128: HIP kernel with 64-key tiles and exact P.
// Also the fallback of fmha_fp8_d192_asm for tensors past the assembly kernel's 32-bit offsets.
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
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

// Online softmax and PV of one 64-key tile: P = 2^(scale * s - mref + LEAN_SHIFT) on v_exp, RNE to e4m3 with a block
// scale; l is summed by MFMA from the same rounded P as O.
__device__ __forceinline__ void softmax_pv(o_tile &o, s2_tile &s, uint32_t vbase, float scale, float &m_, float &l_,
                                          const i32x8 &sel, k_frag &kf, uint32_t k_next_off) {
    const float lane_raw = xor32_max(lane_max2(s.a, s.b));
    const float mx = lane_raw * scale;
    const float mnew = fmaxf(m_, mx);
    const float dmax = mnew - m_;
    const int skip = __all(dmax <= LEAN_THR);
    float mref = m_;
    if (!__builtin_expect(skip, 1)) {
        const float corr = (dmax > 0.f) ? exp2_hw(-dmax) : 1.f;
        if (__builtin_expect(m_ != neg_inf(), 1)) {
            rescale_o(o, corr);
            l_ *= corr;
        }
        mref = mnew;
        m_ = mnew;
    }
    const int off = block_off(lane_raw, scale, mref);
    const int sb = 127 + off;
    // LEAN_SHIFT - (mref + off); the asm keeps fast-math from reassociating it.
    float mo = mref + (float)off;
    asm volatile("" : "+v"(mo));
    const float bx = LEAN_SHIFT - mo;
    iglp_softmax();
    p_exp_shifted(s.a, scale, bx);
    p_exp_shifted(s.b, scale, bx);
    const i32x8 ps = pack_p(s.a, s.b);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    wg_bar();
    __builtin_amdgcn_s_setprio(1);
    mma_pv_lds_khead(o, vbase, ps, kf, k_next_off, sb);
    l_ += block_rowsum(sel, ps, sb);
    __builtin_amdgcn_s_setprio(0);
}

template <bool PairHeads, bool Bshd>
__global__ void __launch_bounds__(NT, 1) attend_ker(const unsigned char *Q, const unsigned char *K,
                                                    const unsigned char *V, bf16 *O, float *Lse, int M, int N, int Hq,
                                                    int Hkv, float scale, int nxcd, const float *qd, const float *kd,
                                                    const float *vd, float v_scale) {
    extern __shared__ Smem sm[];
    // Descales from device memory (null reads as 1), in the assembly kernel's order (the fences stop fast-math
    // from reassociating it).
    float qk = (qd ? *qd : 1.f) * (kd ? *kd : 1.f);
    asm volatile("" : "+v"(qk));
    scale *= qk;
    float ov = (vd ? *vd : 1.f) * v_scale;
    asm volatile("" : "+v"(ov));
    Smem &s = sm[0];

    constexpr int D = 192, Dv = 128;
    // Strides stay in SGPRs in both layouts, so BHSD does not constant-fold them into a different schedule.
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
        // Two 4-wave groups run two Q heads that share a KV head, so every K/V tile load serves both.
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
        float m_ = neg_inf(), l_ = 0.f;
        const i32x8 sel = rowsum_sel();

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
            // Also keeps the BSHD and BHSD instantiations on the same schedule (bitwise-equal results).
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
            iglp_qk();
            if (warpid() < 4)
                __builtin_amdgcn_s_setprio(1);
            else
                __builtin_amdgcn_s_setprio(0);
            // QK, with the next tile's K/V prefetch between the MFMA pairs.
            prefetch_k_pass<0>(s.k[nx], gKn, k_stride);
            qk_step<false>(att, kf, qT.d0, k_off(kbase, 1));
            prefetch_k_pass<1>(s.k[nx], gKn, k_stride);
            qk_step<true>(att, kf, qT.d1, k_off(kbase, 2));
            prefetch_v(s.v[nx], gVn, v_stride);
            qk_tail(att, kf, qT.d2);
            if (__builtin_expect(q_start < (kb + 1) * KV_BLOCK, 0))
                mask_s2(att, q_start, kb);
            softmax_pv(oR, att, vbase, scale, m_, l_, sel, kf, k_next_off);
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
        // P and l carry 2^LEAN_SHIFT; their ratio does not.
        const float m_lse = m_ - LEAN_SHIFT;
        store_o_fp8_qrow(Oh, q_start, M, oR, l_, ov, o_stride);
        if (Lh) {
            const float lse = m_lse * LN2 + logf(l_);
            store_lse(Lh, q_start, M, lse);
        }
    }
}

static void run(const unsigned char *Q, const unsigned char *K, const unsigned char *V, bf16 *O, float *Lse, int M,
                int N, int B, int Hq, int Hkv, float scale, bool bshd, const float *qd, const float *kd,
                const float *vd, float v_scale, hipStream_t stream) {
    const int GROUP = Hq / Hkv;
    const bool pair_heads = GROUP >= 2 && (GROUP & 1) == 0;
    const int q_waves = pair_heads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + q_waves - 1) / q_waves;
    const int nq_pairs = (numQ + 1) / 2;
    const int nxcd = fmha_num_xcds();
    const size_t smem = sizeof(Smem);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    constexpr int D = 192, Dv = 128;
    const bool split = fmha_split_batch(B, nq_pairs, nheads, nxcd, M);
    const int batches = split ? B : 1;
    const int n_head_units = nheads * (split ? 1 : B);
    // One workgroup per job (no persistent loop).
    const int grid = nq_pairs * n_head_units;
    const size_t q_batch = bshd ? (size_t)M * Hq * D : (size_t)Hq * M * D;
    const size_t k_batch = bshd ? (size_t)N * Hkv * D : (size_t)Hkv * N * D;
    const size_t v_batch = bshd ? (size_t)N * Hkv * Dv : (size_t)Hkv * N * Dv;
    const size_t o_batch = bshd ? (size_t)M * Hq * Dv : (size_t)Hq * M * Dv;
    if (pair_heads && bshd)
        hip_check(hipFuncSetAttribute((const void *)attend_ker<true, true>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536), "hipFuncSetAttribute");
    else if (pair_heads)
        hip_check(hipFuncSetAttribute((const void *)attend_ker<true, false>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536), "hipFuncSetAttribute");
    else if (bshd)
        hip_check(hipFuncSetAttribute((const void *)attend_ker<false, true>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536), "hipFuncSetAttribute");
    else
        hip_check(hipFuncSetAttribute((const void *)attend_ker<false, false>, hipFuncAttributeMaxDynamicSharedMemorySize, 65536), "hipFuncSetAttribute");
    for (int b = 0; b < batches; ++b) {
        const unsigned char *Qb = Q + b * q_batch;
        const unsigned char *Kb = K + b * k_batch;
        const unsigned char *Vb = V + b * v_batch;
        bf16 *Ob = O + b * o_batch;
        float *Lb = Lse ? Lse + (size_t)b * Hq * M : nullptr;
        if (pair_heads && bshd)
            attend_ker<true, true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, qd, kd,
                                                               vd, v_scale);
        else if (pair_heads)
            attend_ker<true, false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, qd, kd,
                                                                vd, v_scale);
        else if (bshd)
            attend_ker<false, true><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, qd, kd,
                                                                vd, v_scale);
        else
            attend_ker<false, false><<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, qd,
                                                                 kd, vd, v_scale);
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                   bool return_lse, bool bshd, double v_scale, const float *qd, const float *kd,
                                   const float *vd) {
    TORCH_CHECK(causal, "fmha_fp8_d192 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q, k, v must be on the same device");
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && k.scalar_type() == at::kFloat8_e4m3fn &&
                    v.scalar_type() == at::kFloat8_e4m3fn,
                "fmha_fp8_d192 needs float8_e4m3fn");
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
    TORCH_CHECK(M == N && scale > 0, "fmha_fp8_d192 needs M == N and scale > 0");
    TORCH_CHECK(fmha_scale_fp8_ok(scale), "fmha_fp8_d192 needs a positive normal fp32 scale");
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

} // namespace fmha_fp8_d192
