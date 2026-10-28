// BF16 causal forward attention for MLA, d_qk = 192 / d_v = 128: an 8-stage software pipeline per 64-key tile,
// with waves 4-7 one stage behind waves 0-3 so one wave's MFMAs overlap the other's LDS reads on every SIMD.
// The pipeline follows the stage structure of AMD's OPUS hd192_v128 kernel in AITER
// (fmha_fwd_hd192_v128_bf16_opus_kernel.hpp, MIT License, Copyright (C) 2024-2026 Advanced Micro Devices, Inc.).
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <type_traits>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
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

// Pipeline of tile t (two S buffers; K(t+2) streams into K(t)'s buffer once both of its halves are read):
//   st0 mem  K(t) half 0; prefetch V(t)        st1 cmp  QK half 0 (12 MFMA); exp 24 of S(t-1)
//   st2 mem  K(t) half 1; V(t-1) landed        st3 cmp  QK half 1; exp 8, l += sum, P = bf16 S(t-1)
//   st4 mem  V(t-1) half 0; prefetch K(t+2)    st5 cmp  PV half 0 (8 MFMA); row max of S(t), lazy decision
//   st6 mem  V(t-1) half 1; K(t+1) landed      st7 cmp  PV half 1; s*c - m of S(t); rescale O, l if moved
constexpr int K_VM = 3, V_VM = 2, KEEP_VM = K_VM + V_VM;  // buffer_load ... lds per thread per K / V tile

// End of a pipeline stage: workgroup barrier, with the scheduler fenced on both sides.
__device__ __forceinline__ void stage_end() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}
template <int N>
__device__ __forceinline__ void wait_vm() {
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}
// lgkmcnt(0) as the builtin, so LLVM knows the fragments have landed and adds no waits of its own.
__device__ __forceinline__ void wait_lgkm0() {
    __builtin_amdgcn_s_waitcnt(0xC07F);
}

struct alignas(16) SmemP {
    st_kp k[2];
    st_v v[2];
    bf16 ost[NW][32 * 128];    // per-wave O staging (32 rows x 256 B)
};

// The O epilogue goes through LDS so the global stores are row-contiguous 16 B per lane; the flush of a pass is
// deferred to after the next pass's prologue loads.
// Lane id from an asm v_mbcnt, so the epilogue's lane addresses are not hoisted out of the main loop.
__device__ __forceinline__ int lane_novec() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(l));
    return l;
}

// O / l as bf16 into the wave's 8 KB of LDS (16 B slot XOR row & 15, 8 B half XOR row bit 4).
__device__ __forceinline__ void stage_o192_write(const o_tile &o, float l_, uint32_t stage) {
    const float inv = (l_ > 0.f) ? __builtin_amdgcn_rcpf(l_) : 0.f;
    const int lane = lane_novec();
    const int row = lane & 31, g = lane >> 5;
    const uint32_t wrow = stage + row * 256, wx = row & 15, wh = (g ^ ((row >> 4) & 1)) * 8;
    // One 16-VGPR block at a time (a scaled copy of all of O would spill).
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const fx16 b = mulsc(i == 0 ? o.a : i == 1 ? o.b : i == 2 ? o.c : o.d, inv);
        uint64_t d[4];
        uint32_t a[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            bf16 x[4] = {(bf16)b[4 * m], (bf16)b[4 * m + 1], (bf16)b[4 * m + 2], (bf16)b[4 * m + 3]};
            d[m] = *reinterpret_cast<uint64_t *>(x);
            a[m] = wrow + (((4 * i + m) ^ wx) * 16) + wh;     // cols 32i + 4g + 8m .. +3
        }
        asm volatile("ds_write_b64 %0, %4\n\tds_write_b64 %1, %5\n\tds_write_b64 %2, %6\n\tds_write_b64 %3, %7"
                     :: "v"(a[0]), "v"(a[1]), "v"(a[2]), "v"(a[3]), "v"(d[0]), "v"(d[1]), "v"(d[2]), "v"(d[3])
                     : "memory");
    }
}

// Read the staged block back 16 B per lane, 4 rows per instruction, and store it.
__device__ __forceinline__ void stage_o192_flush(bf16 *Oh, int q_start, int M, int o_stride, uint32_t stage) {
    const int lane = lane_novec();
    const int s = lane & 15, rq = lane >> 4;
    const uint32_t Br = stage + rq * 256, sx = s ^ rq;
    const uint32_t B = Br + sx * 16, B1 = Br + (sx ^ 4) * 16, B2 = Br + (sx ^ 8) * 16, B3 = Br + (sx ^ 12) * 16;
    __uint128_t rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7;
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                 "ds_read_b128 %0, %8\n\tds_read_b128 %1, %9 offset:1024\n\t"
                 "ds_read_b128 %2, %10 offset:2048\n\tds_read_b128 %3, %11 offset:3072\n\t"
                 "ds_read_b128 %4, %8 offset:4096\n\tds_read_b128 %5, %9 offset:5120\n\t"
                 "ds_read_b128 %6, %10 offset:6144\n\tds_read_b128 %7, %11 offset:7168\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&v"(rd0), "=&v"(rd1), "=&v"(rd2), "=&v"(rd3), "=&v"(rd4), "=&v"(rd5), "=&v"(rd6), "=&v"(rd7)
                 : "v"(B), "v"(B1), "v"(B2), "v"(B3) : "memory");
    const __uint128_t rd[8] = {rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int q = q_start + 4 * c + rq;
        __uint128_t v = rd[c];
        if (c >= 4) {   // rows 16-31 keep their 8 B halves swapped in LDS
            uint64_t *h = reinterpret_cast<uint64_t *>(&v);
            const uint64_t t0 = h[0];
            h[0] = h[1];
            h[1] = t0;
        }
        if (q < M)
            *reinterpret_cast<__uint128_t *>(Oh + (size_t)q * o_stride + 8 * s) = v;
    }
}

template <bool PairHeads, bool Bshd>
__global__ void __launch_bounds__(NT, 1) attend_pp(const bf16 *Q, const bf16 *K, const bf16 *V, bf16 *O, float *Lse,
                                                   int M, int N, int Hq, int Hkv, float scale, int nxcd,
                                                   int n_head_units) {
    extern __shared__ SmemP smp[];
    SmemP &s = smp[0];

    constexpr int D = 192, Dv = 128;
    const int q_stride = Bshd ? Hq * D : D;
    const int k_stride = Bshd ? Hkv * D : D;
    const int v_stride = Bshd ? Hkv * Dv : Dv;
    const int o_stride = Bshd ? Hq * Dv : Dv;
    const int GROUP = Hq / Hkv;
    constexpr int QW = PairHeads ? (NW / 2) : NW;
    const int numQ = (M / Q_BLOCK + QW - 1) / QW;
    const int nq_pairs = (numQ + 1) / 2;
    const int n_jobs = nq_pairs * n_head_units;
    const int stride = (int)gridDim.x;
    const int num_tiles = N / KV_BLOCK;
    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    const bool stagger = (wave >> 2) != 0;
    const float c = scale;
    // Lane address parts, computed once: LDS fragment reads and global->LDS offsets are these plus immediates.
    const int lane = laneid();
    const uint32_t k_rd = lds_off(s.k[0].data) + kp_read_lane(lane);
    const uint32_t v_rd = lds_off(s.v[0].data) + v_read_lane(lane);
    const uint32_t k_go = kp_global_lane(lane, wave, k_stride);
    const uint32_t v_go = v_global_lane((int)threadIdx.x, v_stride);

    const uint32_t ost_stage = lds_off(&s.ost[wave][0]);
    // O / LSE of the previous pass, stored after the next pass's prologue.
    int ost_pend = 0, ost_q = 0;
    bf16 *ost_O = nullptr;
    float *ost_L = nullptr;
    float ost_lse = 0.f;
    // Persistent grid: each workgroup walks jobs (a causal Q-block pair of one head unit).
    for (int job = (int)blockIdx.x; job < n_jobs; job += stride) {
        int qx, hu;
        fmha_xcd_map(nq_pairs, n_head_units, nxcd, qx, hu, job);

        int batch_idx, head_idx, head_kv, qwarp;
        if constexpr (PairHeads) {
            // Two 4-wave groups run two Q heads that share a KV head, so every K/V tile load serves both.
            const int pairs_per_kv = GROUP / 2;
            const int units_per_batch = Hkv * pairs_per_kv;
            batch_idx = hu / units_per_batch;
            const int u = hu - batch_idx * units_per_batch;
            head_kv = u / pairs_per_kv;
            const int pair = u - head_kv * pairs_per_kv;
            const int head_slot = wave / QW;
            qwarp = wave - head_slot * QW;
            head_idx = head_kv * GROUP + pair * 2 + head_slot;
        } else {
            batch_idx = hu / Hq;
            head_idx = hu - batch_idx * Hq;
            head_kv = head_idx / GROUP;
            qwarp = wave;
        }

        const bf16 *Qh, *Kh, *Vh;
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

        const int q_lo = qx, q_hi = numQ - 1 - qx;

        for (int pp = 0; pp < 2; ++pp) {
            const int qblk = pp ? q_lo : q_hi;
            if (pp && q_lo == q_hi)
                break;
            const int tile_idx = qblk * QW + qwarp;
            const int max_tile_idx = qblk * QW + QW - 1;
            const int max_q_end = (max_tile_idx + 1) * Q_BLOCK;
            int T = (max_q_end + KV_BLOCK - 1) / KV_BLOCK;
            if (T > num_tiles)
                T = num_tiles;
            const int q_start = tile_idx * Q_BLOCK;
            auto pf_k = [&](int buf, int t) {
                prefetch_kp(s.k[buf], kv_tile_ptr(Kh, t, k_stride), k_go, wave, k_stride);
            };
            auto pf_k0 = [&](int buf, int t) {
                prefetch_kp<0, 1>(s.k[buf], kv_tile_ptr(Kh, t, k_stride), k_go, wave, k_stride);
            };
            auto pf_k12 = [&](int buf, int t) {
                prefetch_kp<1, 3>(s.k[buf], kv_tile_ptr(Kh, t, k_stride), k_go, wave, k_stride);
            };
            auto pf_v = [&](int buf, int t) {
                prefetch_vp(s.v[buf], kv_tile_ptr(Vh, t, v_stride), v_go, wave, v_stride);
            };

            // Pass boundary: the previous pass's clamped K prefetch may still be in flight.
            wait_vm<0>();
            stage_end();
            q_tile192 qT;
            load_q_raw(qT, Qh + (size_t)tile_idx * Q_BLOCK * q_stride, q_stride);
            pf_k(0, 0);
            pf_k(1, T > 1 ? 1 : 0);
            pf_v(0, 0);
            o_tile oR;
            zero_o(oR);
            float m_, l_ = 0.f;
            s_tile S0, S1;
            k_su kf;
            v_su vf;
            p_tile P;
            wait_vm<0>();
            stage_end();
            if (stagger)
                stage_end();
            if (ost_pend) {
                stage_o192_flush(ost_O, ost_q, M, o_stride, ost_stage);
                if (ost_L)
                    store_lse(ost_L, ost_q, M, ost_lse);
                ost_pend = 0;
            }

            // Tile 0: QK and the scaled, max-subtracted scores; its exp runs in the next phase.
            read_kp_su<0>(kf, k_rd);
            wait_lgkm0();
            stage_end();
            S0.a = qk_su(kf, qT);
            stage_end();
            read_kp_su<1>(kf, k_rd);
            wait_lgkm0();
            stage_end();
            S0.b = qk_su(kf, qT);
            if (__builtin_expect(q_start < KV_BLOCK, 0))
                mask_s2(S0, q_start, 0);
            m_ = c * col_max2(S0);
            fma_sub<0, 32>(S0, c, m_);
            // scale = 0: fma(-inf, 0, -m) is NaN, so the masked scores go back to -inf (c is uniform).
            if (__builtin_expect(c == 0.f, 0) && q_start < KV_BLOCK)
                mask_s2(S0, q_start, 0);
            stage_end();
            pf_k(0, T > 2 ? 2 : T - 1);
            stage_end();

            // Buffer parity is a template constant so every LDS offset is an immediate.
            auto phase = [&](auto cur_c, s_tile &Sc, s_tile &Sp, int t) {
                constexpr int cur = decltype(cur_c)::value, prv = cur ^ 1;
                constexpr uint32_t KB = cur * sizeof(st_kp), VP = prv * sizeof(st_v);
                const int tk = t + 2 < T ? t + 2 : T - 1;
                // st0 mem: K(t) half 0.
                read_kp_su<0>(kf, k_rd + KB);
                wait_lgkm0();
                stage_end();
                // st1 cmp: QK half 0, 2 exp of S(t-1) per MFMA.
                Sc.a = qk_su(kf, qT);
                exp_range<0, 24>(Sp);
                sched_exp_pairs<12, 2, 1>();
                stage_end();
                // st2 mem: K(t) half 1; prefetch V(t) into V(t-2)'s buffer (last read in phase t-1 st6); V(t-1) landed.
                read_kp_su<1>(kf, k_rd + KB);
                pf_v(cur, t);
                wait_lgkm0();
                wait_vm<KEEP_VM>();
                stage_end();
                // st3 cmp: QK half 1; rest of the exp, row sum, P.
                Sc.b = qk_su(kf, qT);
                exp_range<24, 32>(Sp);
                l_ += row_sum(Sp);
                copy_p(P, Sp);
                asm volatile("" : "+v"(P.a), "+v"(P.b), "+v"(P.c), "+v"(P.d), "+v"(l_));
                sched_exp_pairs<4, 2, 3>();
                sched_valu_pairs<8, 6, 3>();
                stage_end();
                // st4 mem: V(t-1) half 0; K(t+2) chunk 0 into K(t)'s buffer (both halves read by now).
                read_vp_su<0>(vf, v_rd + VP);
                pf_k0(cur, tk);
                if (__builtin_expect(q_start < (t + 1) * KV_BLOCK, 0))
                    mask_s2(Sc, q_start, t);
                wait_lgkm0();
                stage_end();
                // st5 cmp: PV half 0; row max of S(t) and the lazy decision; s*c - m on 12 scores.
                pv_su(oR.a, oR.b, vf, P);
                // The vote fma(x, c, -m) <= T equals max(m, cx) - m <= T up to one rounding, and is one op shorter.
                const float cmx = col_max2(Sc);
                const float mnew = fmaxf(m_, c * cmx);
                const bool skip = __all(__builtin_fmaf(cmx, c, -m_) <= RESCALE_THRESHOLD);
                const float mref = skip ? m_ : mnew;
                fma_sub<0, 12>(Sc, c, mref);
                sched_valu_pairs<8, 5, 5>();
                stage_end();
                // st6 mem: V(t-1) half 1; K(t+2) chunks 1-2; K(t+1) landed.
                read_vp_su<1>(vf, v_rd + VP);
                pf_k12(cur, tk);
                wait_lgkm0();
                wait_vm<KEEP_VM>();
                stage_end();
                // st7 cmp: PV half 1; s*c - m on the other 20; rescale O and l if the max moved.
                pv_su(oR.c, oR.d, vf, P);
                fma_sub<12, 32>(Sc, c, mref);
                // Before the branch below: in another basic block the hints would schedule nothing.
                sched_valu_pairs<8, 3, 7>();
                if (__builtin_expect(c == 0.f, 0) && q_start < (t + 1) * KV_BLOCK)
                    mask_s2(Sc, q_start, t);
                if (!__builtin_expect(skip, 1)) {
                    const float corr = exp2_hw(m_ - mref);
                    rescale_o(oR, corr);
                    l_ *= corr;
                    m_ = mref;
                }
                // Keep l's multiply and the next add as separate roundings in every instantiation (no FMA fusion).
                asm volatile("" : "+v"(l_));
                stage_end();
            };

            // Tiles past this wave's last key (q_start + 31) have no visible key. Phase L + 1 runs in full; later
            // phases keep only the prefetches, waits and barriers, with S = -inf so the last exp gives P = 0.
            // Pair-head kernels keep every phase.
            const int L = PairHeads ? T - 1 : (q_start + Q_BLOCK - 1) / KV_BLOCK;
            const int TL = T < L + 2 ? T : L + 2;
            auto phase_dead = [&](auto cur_c, s_tile &Sc, int t) {
                constexpr int cur = decltype(cur_c)::value;
                const int tk = t + 2 < T ? t + 2 : T - 1;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    Sc.a[i] = neg_inf();
                    Sc.b[i] = neg_inf();
                }
                stage_end();
                stage_end();
                pf_v(cur, t);
                wait_vm<KEEP_VM>();
                stage_end();
                stage_end();
                pf_k0(cur, tk);
                stage_end();
                stage_end();
                pf_k12(cur, tk);
                wait_vm<KEEP_VM>();
                stage_end();
                stage_end();
            };

            int t = 1;
            // Four phases per iteration (each taken loop back-edge costs a slow first QK stage).
#pragma unroll 1
            for (; t + 3 < TL; t += 4) {
                phase(std::integral_constant<int, 1>{}, S1, S0, t);
                phase(std::integral_constant<int, 0>{}, S0, S1, t + 1);
                phase(std::integral_constant<int, 1>{}, S1, S0, t + 2);
                phase(std::integral_constant<int, 0>{}, S0, S1, t + 3);
            }
            // At most one more pair (kept as a loop: as an if, LLVM allocates the kernel's registers differently).
#pragma unroll 1
            for (; t + 1 < TL; t += 2) {
                phase(std::integral_constant<int, 1>{}, S1, S0, t);
                phase(std::integral_constant<int, 0>{}, S0, S1, t + 1);
            }
            if (t < TL) {
                phase(std::integral_constant<int, 1>{}, S1, S0, t);
                ++t;
            }
            if constexpr (!PairHeads) {
#pragma unroll 1
                for (; t < T; ++t) {
                    if (t & 1)
                        phase_dead(std::integral_constant<int, 1>{}, S1, t);
                    else
                        phase_dead(std::integral_constant<int, 0>{}, S0, t);
                }
            }

            // Last tile: exp, P, and its PV.
            wait_vm<K_VM>();
            stage_end();
            auto last = [&](auto vb_c, s_tile &Sl) {
                constexpr uint32_t VP = decltype(vb_c)::value * sizeof(st_v);
                exp_range<0, 32>(Sl);
                l_ += row_sum(Sl);
                asm volatile("" : "+v"(l_));
                copy_p(P, Sl);
                stage_end();
                read_vp_su<0>(vf, v_rd + VP);
                wait_lgkm0();
                stage_end();
                pv_su(oR.a, oR.b, vf, P);
                stage_end();
                read_vp_su<1>(vf, v_rd + VP);
                wait_lgkm0();
                stage_end();
                pv_su(oR.c, oR.d, vf, P);
            };
            if ((T & 1) == 0)
                last(std::integral_constant<int, 1>{}, S1);
            else
                last(std::integral_constant<int, 0>{}, S0);
            if (!stagger)
                stage_end();

            stage_o192_write(oR, l_, ost_stage);
            ost_O = Oh; ost_L = Lh; ost_q = q_start; ost_pend = 1;
            if (Lh)
                ost_lse = m_ * LN2 + logf(l_);
        }
    }
    if (ost_pend) {
        stage_o192_flush(ost_O, ost_q, M, o_stride, ost_stage);
        if (ost_L)
            store_lse(ost_L, ost_q, M, ost_lse);
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
    const size_t smem = sizeof(SmemP);
    const int nheads = pair_heads ? (Hq / 2) : Hq;
    constexpr int D = 192, Dv = 128;
    const bool split = fmha_split_batch(B, nq_pairs, nheads, nxcd, M);
    const int batches = split ? B : 1;
    const int n_head_units = nheads * (split ? 1 : B);
    const int grid = fmha_persist_grid(nq_pairs * n_head_units, nxcd);
    const size_t q_batch = bshd ? (size_t)M * Hq * D : (size_t)Hq * M * D;
    const size_t k_batch = bshd ? (size_t)N * Hkv * D : (size_t)Hkv * N * D;
    const size_t v_batch = bshd ? (size_t)N * Hkv * Dv : (size_t)Hkv * N * Dv;
    const size_t o_batch = bshd ? (size_t)M * Hq * Dv : (size_t)Hq * M * Dv;
    if (pair_heads && bshd)
        hip_check(hipFuncSetAttribute((const void *)attend_pp<true, true>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    else if (pair_heads)
        hip_check(hipFuncSetAttribute((const void *)attend_pp<true, false>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    else if (bshd)
        hip_check(hipFuncSetAttribute((const void *)attend_pp<false, true>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    else
        hip_check(hipFuncSetAttribute((const void *)attend_pp<false, false>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem), "hipFuncSetAttribute");
    for (int b = 0; b < batches; ++b) {
        const bf16 *Qb = Q + b * q_batch;
        const bf16 *Kb = K + b * k_batch;
        const bf16 *Vb = V + b * v_batch;
        bf16 *Ob = O + b * o_batch;
        float *Lb = Lse ? Lse + (size_t)b * Hq * M : nullptr;
        if (pair_heads && bshd)
            attend_pp<true, true>
                <<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, n_head_units);
        else if (pair_heads)
            attend_pp<true, false>
                <<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, n_head_units);
        else if (bshd)
            attend_pp<false, true>
                <<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, n_head_units);
        else
            attend_pp<false, false>
                <<<grid, NT, smem, stream>>>(Qb, Kb, Vb, Ob, Lb, M, N, Hq, Hkv, scale, nxcd, n_head_units);
        const hipError_t err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, "fmha_bf16_d192 launch: ", hipGetErrorString(err));
    }
}

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                   bool return_lse, bool bshd) {
    TORCH_CHECK(causal, "fmha_bf16_d192 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q, k, v must be on the same device");
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "fmha_bf16_d192 needs bfloat16");
    TORCH_CHECK(fmha_scale_finite(scale), "fmha_bf16_d192 needs a finite scale");
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
    TORCH_CHECK(D == 192 && Dv == 128 && M % 256 == 0 && N % 64 == 0);
    TORCH_CHECK(M == N, "fmha_bf16_d192 needs equal query and key lengths, got M=", M, " N=", N);
    TORCH_CHECK(scale >= 0, "fmha_bf16_d192 needs scale >= 0 (masked logits are -inf before scaling)");
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
    run(reinterpret_cast<const bf16 *>(q.data_ptr()), reinterpret_cast<const bf16 *>(k.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()), reinterpret_cast<bf16 *>(O.data_ptr()), Lp, M, N, B, Hq, Hkv,
        (float)scale * L2E, bshd, at::cuda::getCurrentCUDAStream());
    if (return_lse)
        return {O, Lse};
    return {O};
}

} // namespace fmha_bf16_d192
