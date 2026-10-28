// BF16 causal forward attention, d = 128: HipKittens' 8-wave ping-pong kernel with 64-key tiles, a
// persistent grid, causal Q-block pairing and native BHSD / BSHD layouts.
//
// Adapted from HipKittens kernels/cdna4/attn/gqa_causal/kernel.cpp (https://github.com/HazyResearch/HipKittens).
// MIT License, Copyright (c) 2024 HazyResearch; see third_party/hipkittens/LICENSE.
#include "hip/hip_runtime.h"
#include "tiles.cuh"
#include "xcd_map.h"
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>
#include <c10/hip/HIPGuard.h>
#include <cstdint>
#include <type_traits>

constexpr int Q_BLOCK_SIZE = 32;
constexpr int KV_BLOCK_SIZE = 64;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

using namespace kittens;

// Logical coords stay {b, s, h, c}; the physical packing is BHSD or BSHD. stride<1>() is the sequence pitch.
template<typename T, bool Bshd>
struct gl_view {
    using identifier = ducks::gl::identifier;
    using dtype = T;
    T *raw_ptr;
    int B, S, H, D;

    __host__ __device__ gl_view() : raw_ptr(nullptr), B(0), S(0), H(0), D(0) {}
    __host__ __device__ gl_view(T *p, int b, int s, int h, int d)
        : raw_ptr(p), B(b), S(s), H(h), D(d) {}

    __host__ __device__ int batch() const { return B; }
    __host__ __device__ int depth() const { return S; }
    __host__ __device__ int rows() const { return H; }
    __host__ __device__ int cols() const { return D; }

    template<int axis>
    __host__ __device__ size_t shape() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be 0..3");
        if constexpr (axis == 0) return (size_t)B;
        else if constexpr (axis == 1) return (size_t)S;
        else if constexpr (axis == 2) return (size_t)H;
        else return (size_t)D;
    }
    template<int axis>
    __host__ __device__ size_t stride() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be 0..3");
        if constexpr (axis == 0) return (size_t)S * H * D;
        else if constexpr (axis == 1) return Bshd ? (size_t)H * D : (size_t)D;
        else if constexpr (axis == 2) return Bshd ? (size_t)D : (size_t)S * D;
        else return 1;
    }
    __device__ T &operator[](const coord<ducks::default_type> &idx) const {
        const int64_t b = idx.b, s = idx.d, h = idx.r, c = idx.c;
        if constexpr (Bshd)
            return raw_ptr[((b * S + s) * H + h) * D + c];
        return raw_ptr[((b * H + h) * S + s) * D + c];
    }
    __device__ int64_t idx(const coord<ducks::default_type> &idx) const {
        const int64_t b = idx.b, s = idx.d, h = idx.r, c = idx.c;
        if constexpr (Bshd)
            return ((b * S + s) * H + h) * D + c;
        return ((b * H + h) * S + s) * D + c;
    }
};

// HipKittens sizes buffer resources as uint32(B*S*H*D*elem), which wraps at 4 GiB. The tile pointer is already
// tile-local, so this view reports a large in-range size instead.
template<typename GL>
struct gl_u32_range {
    using identifier = ducks::gl::identifier;
    using dtype = typename GL::dtype;
    const GL &g;
    __device__ __forceinline__ int batch() const { return 1; }
    __device__ __forceinline__ int depth() const { return 1; }
    __device__ __forceinline__ int rows() const { return 1; }
    __device__ __forceinline__ int cols() const { return (int)(0x7FFFFFFFu / sizeof(dtype)); }
    template<int axis>
    __device__ __forceinline__ size_t stride() const { return g.template stride<axis>(); }
    template<typename Idx>
    __device__ __forceinline__ dtype &operator[](const Idx &idx) const { return g[idx]; }
};
template<typename GL>
__device__ __forceinline__ gl_u32_range<GL> sat_gl(const GL &g) { return {g}; }

// Buffer range in bytes; 0 or anything past 4 GiB becomes the maximum.
__device__ __forceinline__ uint32_t sat_u32_bytes(uint64_t n) {
    return (n == 0ull || n > 0xFFFFFFFFull) ? 0xFFFFFFFFu : (uint32_t)n;
}

using G = kittens::group<NUM_WARPS>;

// Pairs x (one MFMA, then VALU_CNT VALU instructions).
template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
    __builtin_amdgcn_sched_group_barrier(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

// Pairs x (one MFMA, then EXP_CNT transcendental instructions).
template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
    __builtin_amdgcn_sched_group_barrier(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2_rt_base(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    static_assert(std::is_same_v<shape, rt_32x32_s>, "Only 32x32 tiles are supported");

    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }

}

// kittens::{zero, ones, copy, sub, mul, exp2, log, add} are ambiguous between the tile and vector overloads on
// ROCm 7.2's clang; these dispatch explicitly.
namespace hk_fix {
template<typename T>
__device__ __forceinline__ void zero(T &dst) {
    if constexpr (ducks::rv::all<T>) unary_op<base_ops::zero, T>(dst, dst);
    else unary_map<base_ops::zero, T>(dst, dst);
}
template<typename T>
__device__ __forceinline__ void ones(T &dst) {
    if constexpr (ducks::rv::all<T>) unary_op<base_ops::ones, T>(dst, dst);
    else unary_map<base_ops::ones, T>(dst, dst);
}
template<typename D, typename S>
__device__ __forceinline__ void copy(D &dst, const S &src) {
    if constexpr (ducks::rv::all<D>) {
        bin_op<base_ops::copy2, D>(dst, dst, src);
    } else {
        using T2 = typename D::dtype;
        using U2 = typename S::dtype;
        #pragma unroll
        for (int i = 0; i < dst.height; i++) {
            #pragma unroll
            for (int j = 0; j < dst.width; j++) {
                #pragma unroll
                for (int k = 0; k < dst.packed_per_base_tile; k++) {
                    dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(src.tiles[i][j].data[k]);
                }
            }
        }
    }
}
template<typename T, typename U>
__device__ __forceinline__ void sub(T &dst, const T &lhs, const U &rhs) {
    if constexpr (ducks::rv::all<T>) bin_op<base_ops::sub, T>(dst, lhs, rhs);
    else bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
template<typename T, typename U>
__device__ __forceinline__ void mul(T &dst, const T &lhs, const U &rhs) {
    if constexpr (ducks::rv::all<T>) bin_op<base_ops::mul, T>(dst, lhs, rhs);
    else bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
template<typename T>
__device__ __forceinline__ void exp2(T &dst, const T &src) {
    if constexpr (ducks::rv::all<T>) unary_op<base_ops::exp2, T>(dst, src);
    else exp2_rt_base(dst, src);
}
template<typename T>
__device__ __forceinline__ void log(T &dst, const T &src) {
    if constexpr (ducks::rv::all<T>) unary_op<base_ops::log, T>(dst, src);
    else unary_map<base_ops::log, T>(dst, src);
}
template<typename T, typename U>
__device__ __forceinline__ void add(T &dst, const T &lhs, const U &rhs) {
    if constexpr (ducks::rv::all<T>) bin_op<base_ops::sum, T>(dst, lhs, rhs);
    else bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
}

#define zero hk_fix::zero
#define ones hk_fix::ones
#define copy hk_fix::copy
#define sub  hk_fix::sub
#define mul  hk_fix::mul
#define exp2 hk_fix::exp2
#define log  hk_fix::log
#define add  hk_fix::add

// The scale is applied to the fp32 scores, x = fma(s, c, -m), not folded into Q. c = scale * log2 e > 0, so
// max(c * s) = c * max(s) and masked -inf scores stay -inf.
template<typename RV>
__device__ __forceinline__ int rv_all_below_fma(const RV &prev, const RV &raw, float c, float T) {
    int ok = 1;
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; ++i)
            ok &= (__builtin_fmaf(float(raw.data[o][i]), c, -float(prev.data[o][i])) <= T);
    }
    return ok;
}
template<typename RV>
__device__ __forceinline__ void rv_scale_merge(RV &cur, const RV &prev, float c) {   // cur: raw max in, running max out
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; ++i)
            cur.data[o][i] = fmaxf(float(prev.data[o][i]), c * float(cur.data[o][i]));
    }
}
template<typename RV>
__device__ __forceinline__ void rv_scale(RV &v, float c) {
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; ++i)
            v.data[o][i] = c * float(v.data[o][i]);
    }
}
// x = s*c - m per score, one scalar FMA each (pinned: a paired v_pk_fma_f32 cannot issue while an MFMA runs).
template<typename T, typename V>
__device__ __forceinline__ void sub_col_fma(T &dst, const T &src, float c, const V &col_values) {
    #pragma unroll
    for (int j = 0; j < dst.width; j++) {
        const float m = col_values[j][0];
        #pragma unroll
        for (int i = 0; i < dst.height; i++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_base_tile; k++) {
                float x = __builtin_fmaf(src.tiles[i][j].data[k].x, c, -m);
                float y = __builtin_fmaf(src.tiles[i][j].data[k].y, c, -m);
                asm volatile("" : "+v"(x), "+v"(y));
                dst.tiles[i][j].data[k].x = x;
                dst.tiles[i][j].data[k].y = y;
            }
        }
    }
}
// Lazy rescale: O and l are rescaled only when some row's max grows by more than this (log2 units, wave vote).
constexpr float RESCALE_THRESHOLD = 0.5f;

// True iff every lane votes "all below".
__device__ __forceinline__ int wave_all_ok(int lane_ok) {
    return __all(lane_ok);
}

template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s>
using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s>
using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s>
using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

// LDS -> registers, every element pinned after the wait so LLVM cannot sink a read past the next LDS fill.
template<ducks::rt::all RT, ducks::st::all ST>
__device__ __forceinline__ void load_pinned(RT &dst, const ST &src) {
    load(dst, src);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    #pragma unroll
    for (int i = 0; i < RT::height; ++i)
        #pragma unroll
        for (int j = 0; j < RT::width; ++j)
            #pragma unroll
            for (int p = 0; p < RT::packed_per_base_tile; ++p)
                asm volatile("" : "+v"(dst.tiles[i][j].data[p]) :: "memory");
}

// x = -inf if rel < THR_X, y = -inf if rel < THR_Y.
template<int THR_X, int THR_Y>
__device__ inline void mask_vec2_imm(uint32_t rel_vgpr, uint32_t neg_inf_vgpr,
                                     uint32_t& x_ref, uint32_t& y_ref) {
    uint64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
}

// Causal mask of an S^T tile: key k is masked for query q when k > q.
template<ducks::rt::col_layout RT>
__device__ inline void mask_kv_tile(RT &dst, int q_abs, int k_abs, uint32_t neg_inf_v, int lane) {
    const int col  = lane & 31;
    const int q_base = q_abs * Q_BLOCK_SIZE;
    const int k_base = k_abs * KV_BLOCK_SIZE;
    const int q_pos  = q_base + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        // First key row of this lane's elements in the 32-row MFMA chunk.
        const int row_base = (i * 32) + ((lane >> 5) << 2);
        const int rel0 = q_pos - (k_base + row_base);
        const uint32_t rel = static_cast<uint32_t>(rel0);

        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].x);
            auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].y);
            auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].x);
            auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].y);
            auto& d2x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].x);
            auto& d2y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].y);
            auto& d3x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].x);
            auto& d3y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].y);
            auto& d4x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].x);
            auto& d4y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].y);
            auto& d5x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].x);
            auto& d5y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].y);
            auto& d6x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].x);
            auto& d6y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].y);
            auto& d7x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].x);
            auto& d7y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].y);

            mask_vec2_imm< 0, 1 >(rel, neg_inf_v, d0x, d0y);
            mask_vec2_imm< 2, 3 >(rel, neg_inf_v, d1x, d1y);
            mask_vec2_imm< 8, 9 >(rel, neg_inf_v, d2x, d2y);
            mask_vec2_imm<10,11 >(rel, neg_inf_v, d3x, d3y);
            mask_vec2_imm<16,17 >(rel, neg_inf_v, d4x, d4y);
            mask_vec2_imm<18,19 >(rel, neg_inf_v, d5x, d5y);
            mask_vec2_imm<24,25 >(rel, neg_inf_v, d6x, d6y);
            mask_vec2_imm<26,27 >(rel, neg_inf_v, d7x, d7y);
        }
    }
}

template<int D, bool Bshd> struct attn_globals {
    gl_view<bf16, Bshd> Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    hipStream_t stream;
    int nxcd;
    float scale;
    int pair; // 1: a job is a causal Q-block pair (q_lo, q_hi). 0: one Q block per job.
    dim3 grid() {
        const int Nseq = Qg.depth(), H = Qg.rows(), B = Qg.batch();
        const int numQ = (Nseq / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS;
        const int nq_map = pair ? (numQ + 1) / 2 : numQ;
        const int n_work = nq_map * H * B;
        // Persistent grid: at most one workgroup per CU, each walking jobs.
        return dim3(fmha_persist_grid(n_work, nxcd));
    }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// Lane id from an asm v_mbcnt, so the epilogue's lane addresses are not hoisted out of the main loop.
__device__ __forceinline__ int lane_novec() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(l));
    return l;
}

// Q tile load with HipKittens' load<1, qo_tile> lane mapping, through a wave-uniform buffer resource and with all
// chunks in flight at once; issued before the first K wait so the two latencies overlap.
template<int D>
struct q_raw {
    float4 c[D / 16];
};
template<int D, typename GL>
__device__ __forceinline__ void qload_issue(q_raw<D> &raw, const GL &gq, int b, int tile, int h) {
    using RT = rt<float, Q_BLOCK_SIZE, D, row_l, rt_32x16_s>;
    static_assert(RT::height == 1 && RT::width == D / 16 && RT::base_tile_rows == 32 && RT::base_tile_stride == 8 &&
                  RT::base_tile_num_strides == 1 && RT::base_tile_cols == 16, "q_raw assumes HK's 32x16 row tiles");
    const bf16 *p = &gq[(coord<RT>{b, tile, h, 0}.template unit_coord<1, 3>())];
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 rs;
    rs[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a);
    rs[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32));
    rs[2] = 0x7FFFFFFE;                 // gl_u32_range: cols() * sizeof(bf16)
    rs[3] = 0x00020000;
    const int lane = lane_novec();
    const uint32_t voff =
        (uint32_t)(((lane % 32) * (int)gq.template stride<1>() + 8 * (lane / 32)) * (int)sizeof(bf16));
    #pragma unroll
    for (int j = 0; j < D / 16; ++j)
        raw.c[j] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(rs, voff + 32u * j, 0, 0));
}
template<int D>
__device__ __forceinline__ void qload_finish(rt<bf16, Q_BLOCK_SIZE, D, row_l, rt_32x16_s> &q, const q_raw<D> &raw) {
    #pragma unroll
    for (int j = 0; j < D / 16; ++j) {
        const bf16_2 *t = reinterpret_cast<const bf16_2 *>(&raw.c[j]);
        #pragma unroll
        for (int l = 0; l < 4; ++l)
            q.tiles[0][j].data[l] = t[l];
    }
}

// O epilogue (D = 128) through LDS: each wave writes its bf16 O block (32 rows x 256 B) to its own 8 KB (16 B slot
// XOR row & 15, 8 B half XOR row bit 4), then stores it row-contiguously, 16 B per lane and 4 rows per instruction.
// The store of a pass is deferred to the next pass's prologue, after its loads.
template<typename RT>
__device__ inline void stage_o_write(const RT &src, uint32_t stage) {
    static_assert(RT::rows == 32 && RT::cols == 128 && RT::base_tile_rows == 32 && RT::base_tile_stride == 4,
                  "D=128 only");
    const int lane = lane_novec();
    const int row = lane & 31, half = lane >> 5;
    // element (row, col) at row*256 + 16*((col>>3) ^ (row&15)) + 8*(((col>>2)&1) ^ ((row>>4)&1))
    const uint32_t wrow = stage + row * 256, wx = row & 15, wh = (half ^ ((row >> 4) & 1)) * 8;
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
        uint64_t d[4];
        uint32_t a[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            bf16_2 tmp[2];
            tmp[0] = base_types::convertor<bf16_2, float2>::convert(src.tiles[0][j].data[2 * k]);
            tmp[1] = base_types::convertor<bf16_2, float2>::convert(src.tiles[0][j].data[2 * k + 1]);
            d[k] = *reinterpret_cast<uint64_t *>(tmp);
            a[k] = wrow + (((4 * j + k) ^ wx) * 16) + wh;
        }
        asm volatile("ds_write_b64 %0, %4\n\tds_write_b64 %1, %5\n\tds_write_b64 %2, %6\n\tds_write_b64 %3, %7"
                     :: "v"(a[0]), "v"(a[1]), "v"(a[2]), "v"(a[3]), "v"(d[0]), "v"(d[1]), "v"(d[2]), "v"(d[3])
                     : "memory");
    }
}

// Reads the staged block back and stores it (row-contiguous 16 B per lane, 4 rows per instruction).
template<typename RT, typename GL, typename COORD>
__device__ inline void stage_o_flush(const GL &dst, const COORD &idx, uint32_t stage) {
    using U = bf16;
    U *dst_ptr = (U *)&dst[(idx.template unit_coord<1, 3>())];
    const int row_stride = dst.template stride<1>();
    const int lane = lane_novec();
    // Chunk c = rows 4c..4c+3; lane (rq = lane>>4, s = lane&15) reads row 4c+rq, LDS slot s ^ ((4c+rq)&15).
    const int s = lane & 15, rq = lane >> 4;
    const uint32_t Br = stage + rq * 256, sx = s ^ rq;      // XOR the slot index only (stage need not be 256-aligned)
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
    __uint128_t rd[8] = {rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7};
    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
    const buffer_resource br = make_buffer_resource(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(dst_ptr)),
                                                    buffer_size, 0x00020000);
    i32x4 rs = std::bit_cast<i32x4>(br);
    // The O tile pointer is wave-uniform: say so, or every store sits in a readfirstlane loop.
    #pragma unroll
    for (int w = 0; w < 4; ++w)
        rs[w] = __builtin_amdgcn_readfirstlane(rs[w]);
#pragma unroll
    for (int c = 0; c < 8; c++) {
        const int r = 4 * c + rq;
        __uint128_t v = rd[c];
        if (c >= 4) {   // rows 16-31 keep their 8 B halves swapped in LDS
            uint64_t *h = reinterpret_cast<uint64_t *>(&v);
            const uint64_t t0 = h[0];
            h[0] = h[1];
            h[1] = t0;
        }
        llvm_amdgcn_raw_buffer_store_b128(v, rs, (r * row_stride + 8 * s) * sizeof(U), 0, 0);
    }
}

template<int D, bool Bshd> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D, Bshd> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, D, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, D, st_8x32_s> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, D, st_8x32_s>, 2>();
    st_bf<Q_BLOCK_SIZE, D, st_32x32_s> (&o_stage)[NUM_WARPS] =
        al.allocate<st_bf<Q_BLOCK_SIZE, D, st_32x32_s>, NUM_WARPS>();
    static_assert(D != 128 || sizeof(st_bf<Q_BLOCK_SIZE, D, st_32x32_s>) == 8192, "8 KB per wave");

    const int H = g.Qg.rows();
    const int H_KV = g.Kg.rows();
    const int GROUP = H / H_KV;
    const int Nseq = g.Qg.depth();
    const int B = g.Qg.batch();
    const int numQ = (Nseq / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS;
    const int nq_map = g.pair ? (numQ + 1) / 2 : numQ;
    const int n_jobs = nq_map * B * H;
    const int stride = (int)gridDim.x;
    const int num_tiles = Nseq / KV_BLOCK_SIZE;
    const float TEMPERATURE_SCALE = g.scale;
    const int stagger = warpid() / 4;
    const int lane = laneid();
    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t k_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t neg_inf_v = 0xff800000;
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);

    qo_tile<D, bf16> q_reg;
    kv_tile<D, bf16> k_reg;
    kv_tile<D, bf16, col_l, rt_16x32_4_s> v_reg;
    qo_tile_transposed<D, float, col_l, rt_32x32_s> o_reg;
    attn_tile<D, float, col_l, rt_32x32_s> att_block[2];
    attn_tile<D, bf16, col_l, rt_32x32_s> att_block_bf16;
    attn_tile<D, bf16, col_l, rt_16x32_4_s> att_block_bf16_in;
    typename attn_tile<D, float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;

    using T = typename st_bf<KV_BLOCK_SIZE, D, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * D * sizeof(T) / bytes_per_memcpy;
    static_assert(KV_BLOCK_SIZE * D * sizeof(T) % bytes_per_memcpy == 0,
                  "D must divide HK memcpy tile size");
    // The vmcnt waits below count G::load issues: memcpy_per_tile per K or V tile (2 at D = 128, 1 at D = 64).
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    // Deferred O / LSE stores of the previous pass (D = 128).
    int ost_pend = 0, ost_b = 0, ost_t = 0, ost_h = 0;
    typename attn_tile<D, float, col_l, rt_32x32_s>::row_vec ost_lse;
    const uint32_t ost_stage =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&o_stage[0].data[0])) + 8192u * warpid();
    // Persistent grid: each workgroup walks jobs after the prologue.
    for (int job = (int)blockIdx.x; job < n_jobs; job += stride) {
        if (job != (int)blockIdx.x) {
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
        int qx, bh;
        fmha_xcd_map(nq_map, B * H, g.nxcd, qx, bh, job);
        const int batch_idx = bh / H;
        const int head_idx = bh % H;
        const int head_idx_kv = head_idx / GROUP;
        const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
        const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
        i32x4 k_srsrc_base = make_srsrc(
            k_base, sat_u32_bytes((uint64_t)(uint32_t)k_row_stride * (uint64_t)(uint32_t)Nseq), k_row_stride);
        i32x4 v_srsrc_base = make_srsrc(
            v_base, sat_u32_bytes((uint64_t)(uint32_t)v_row_stride * (uint64_t)(uint32_t)Nseq), v_row_stride);

        const int q_lo = qx, q_hi = g.pair ? (numQ - 1 - qx) : qx;
        for (int pp = 0; pp < 2; ++pp) {
            const int qblk = pp ? q_lo : q_hi;
            if (pp && q_lo == q_hi)
                break;
            const int tile_idx = qblk * NUM_WARPS + warpid();
            const int max_tile_idx = qblk * NUM_WARPS + NUM_WARPS - 1;
            const int max_q_end_pos = (max_tile_idx + 1) * Q_BLOCK_SIZE;
            int max_num_tiles = (max_q_end_pos + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
            max_num_tiles = min(max_num_tiles, num_tiles);
            const int q_start_pos = tile_idx * Q_BLOCK_SIZE;
            zero(o_reg);
            zero(norm_vec);
            ones(scale_vec);
            if (pp)
                __builtin_amdgcn_s_barrier();

            q_raw<D> q_raw_;
            qload_issue(q_raw_, g.Qg, batch_idx, tile_idx, head_idx);
            G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base,
                              k_lds_base_0);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();

            {
                qload_finish(q_reg, q_raw_);
                #pragma unroll
                for (int i = 0; i < q_reg.height; ++i)
                    #pragma unroll
                    for (int j = 0; j < q_reg.width; ++j)
                        #pragma unroll
                        for (int p = 0; p < q_reg.packed_per_base_tile; ++p)
                            asm volatile("" : "+v"(q_reg.tiles[i][j].data[p]) :: "memory");
            }

            // K1 and V0 into shared memory, K0 into registers.
            G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base,
                              k_lds_base_1);
            G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base,
                              v_lds_base_0);
            {
                load_pinned(k_reg, k_smem[0]);
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            if constexpr (D == 128) {
                if (ost_pend) {
                    stage_o_flush<qo_tile<D, float, row_l, rt_32x32_s>>(sat_gl(g.Og),
                        coord<qo_tile<D, float, row_l, rt_32x32_s>>{ost_b, ost_t, ost_h, 0}, ost_stage);
                    store(g.L_vec, ost_lse, {ost_b, ost_h, 0, ost_t});
                    ost_pend = 0;
                }
                __builtin_amdgcn_sched_barrier(0);
            }

            // QK0 and the first half of its softmax.
            zero(att_block[0]);
            mma_ABt(att_block[0], k_reg, q_reg, att_block[0]);
            __builtin_amdgcn_sched_barrier(0);
            {
                const int kv_end_pos = (1) * KV_BLOCK_SIZE;
                if (__builtin_expect(q_start_pos < kv_end_pos, 0)) {
                    mask_kv_tile(att_block[0], tile_idx, 0, neg_inf_v, lane);
                }
            }
            col_max(max_vec, att_block[0]);
            rv_scale(max_vec, TEMPERATURE_SCALE);
            copy(max_vec_prev, max_vec);
            sub_col_fma(att_block[0], att_block[0], TEMPERATURE_SCALE, max_vec);
            exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
            __builtin_amdgcn_sched_barrier(0);

            // Waves 4-7 run one cluster behind waves 0-3.
            if (stagger) {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
            }

            // K1 into registers; K2 and V1 into shared memory.
            {
                load_pinned(k_reg, k_smem[1]);
            }
            G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base,
                              k_lds_base_0);
            G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base,
                              v_lds_base_1);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();

            int pending_scale = 0;

            // Hot loop: two key tiles per iteration, eight clusters separated by workgroup barriers.
            #pragma unroll
            for (int j = 3; j < max_num_tiles - 1; j += 2) {
                // Cluster 0: QK1; finish the softmax of QK0.
                zero(att_block[1]);
                mma_ABt(att_block[1], k_reg, q_reg, att_block[1]);
                exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
                if (pending_scale) {
                    mul(norm_vec, norm_vec, scale_vec);
                }
                col_sum(norm_vec, att_block[0], norm_vec);
                copy(att_block_bf16, att_block[0]);
                att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
                sched_barrier_exp_pairs<6, 3, 1>();
                sched_barrier_pairs<10, 5, 1>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 1: K3 into shared memory, V0 into registers; rescale O for the max update of cluster 6.
                G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base,
                                  k_base, k_lds_base_1);
                load(v_reg, v_smem[0]);
                if (pending_scale) {
                    mul_col(o_reg, o_reg, scale_vec);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 2: A0V0; partial softmax of QK1.
                __builtin_amdgcn_s_setprio(1);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
                col_max(max_vec, att_block[1]);
                sched_barrier_pairs<4, 5, 2>();
                int lane_ok = rv_all_below_fma(max_vec_prev, max_vec, TEMPERATURE_SCALE, RESCALE_THRESHOLD);
                rv_scale_merge(max_vec, max_vec_prev, TEMPERATURE_SCALE);
                int all_ok  = wave_all_ok(lane_ok);
                if (__builtin_expect(all_ok, 1)) {
                    copy(max_vec, max_vec_prev);
                    pending_scale = 0;
                } else {
                    sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
                    exp2(scale_vec, scale_vec);
                    copy(max_vec_prev, max_vec);
                    pending_scale = 1;
                }
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
                sub_col_fma(att_block[1], att_block[1], TEMPERATURE_SCALE, max_vec);
                exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
                sched_barrier_pairs<6, 5, 2>();
                sched_barrier_exp_pairs<6, 3, 2>();
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 3: V2 into shared memory, K2 into registers.
                G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V,
                                  v_srsrc_base, v_base, v_lds_base_0);
                {
                    load_pinned(k_reg, k_smem[0]);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 4: QK2; finish the softmax of QK1.
                zero(att_block[0]);
                mma_ABt(att_block[0], k_reg, q_reg, att_block[0]);
                exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
                if (pending_scale) {
                    mul(norm_vec, norm_vec, scale_vec);
                }
                col_sum(norm_vec, att_block[1], norm_vec);
                copy(att_block_bf16, att_block[1]);
                att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
                sched_barrier_exp_pairs<6, 3, 3>();
                sched_barrier_pairs<10, 5, 3>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 5: K4 into shared memory, V1 into registers; mask QK2; rescale O for cluster 2's update.
                G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K,
                                  k_srsrc_base, k_base, k_lds_base_0);
                load(v_reg, v_smem[1]);
                {
                    const int kv_end_pos = (j) * KV_BLOCK_SIZE;
                    if (q_start_pos < kv_end_pos) {
                        mask_kv_tile(att_block[0], tile_idx, j - 1, neg_inf_v, lane);
                    }
                }
                if (pending_scale) {
                    mul_col(o_reg, o_reg, scale_vec);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 6: A1V1; partial softmax of QK2.
                __builtin_amdgcn_s_setprio(1);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
                col_max(max_vec, att_block[0]);
                sched_barrier_pairs<4, 5, 4>();
                lane_ok = rv_all_below_fma(max_vec_prev, max_vec, TEMPERATURE_SCALE, RESCALE_THRESHOLD);
                rv_scale_merge(max_vec, max_vec_prev, TEMPERATURE_SCALE);
                all_ok  = wave_all_ok(lane_ok);
                if (__builtin_expect(all_ok, 1)) {
                    copy(max_vec, max_vec_prev);
                    pending_scale = 0;
                } else {
                    sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
                    exp2(scale_vec, scale_vec);
                    copy(max_vec_prev, max_vec);
                    pending_scale = 1;
                }
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
                mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
                sub_col_fma(att_block[0], att_block[0], TEMPERATURE_SCALE, max_vec);
                exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
                sched_barrier_pairs<6, 5, 4>();
                sched_barrier_exp_pairs<6, 3, 4>();
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // Cluster 7: V3 into shared memory, K3 into registers.
                G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base,
                                  v_base, v_lds_base_1);
                {
                    load_pinned(k_reg, k_smem[1]);
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }

            // Epilogue: the last three tiles (QK3..QK5 = tiles max_num_tiles-3 .. -1). NL = how many of them hold a
            // key visible to this wave (wave-uniform); the others skip their QK and PV MFMAs, mask and softmax (P
            // would be 0). Tile zeroing, register loads, the bf16 P copy, LDS loads, waits and barriers always run.
            const int NL = __builtin_amdgcn_readfirstlane(
                min(3, max(0, (q_start_pos + Q_BLOCK_SIZE - 1) / KV_BLOCK_SIZE - (max_num_tiles - 4))));
            // Cluster 0: QK3; finish the softmax of QK2.
            zero(att_block[1]);
            if (NL >= 1)
                mma_ABt(att_block[1], k_reg, q_reg, att_block[1]);
            exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
            if (pending_scale) {
                mul(norm_vec, norm_vec, scale_vec);
            }

            col_sum(norm_vec, att_block[0], norm_vec);
            copy(att_block_bf16, att_block[0]);
            att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
            sched_barrier_exp_pairs<6, 3, 5>();
            if (NL >= 1)
                sched_barrier_pairs<10, 5, 5>();
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 1: K5 into shared memory, V2 into registers; mask QK3.
            G::load<1, false>(k_smem[1], g.Kg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K,
                              k_srsrc_base, k_base, k_lds_base_1);
            load(v_reg, v_smem[0]);
            if (NL >= 1) {
                const int kv_end_pos = (max_num_tiles - 2) * KV_BLOCK_SIZE;
                if (__builtin_expect(q_start_pos < kv_end_pos, 0)) {
                    mask_kv_tile(att_block[1], tile_idx, max_num_tiles - 3, neg_inf_v, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 2: rescale O for the last hot-loop max update, then A2V2; partial softmax of QK3.
            if (pending_scale) {
                mul_col(o_reg, o_reg, scale_vec);
            }
            __builtin_amdgcn_s_setprio(1);
            mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
            if (NL >= 1) {
                col_max(max_vec, att_block[1]);
                rv_scale_merge(max_vec, max_vec_prev, TEMPERATURE_SCALE);
                sub(scale_vec, max_vec_prev, max_vec);
                copy(max_vec_prev, max_vec);
                exp2(scale_vec, scale_vec);
                sub_col_fma(att_block[1], att_block[1], TEMPERATURE_SCALE, max_vec);
                exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
                sched_barrier_pairs<10, 5, 6>();
                sched_barrier_exp_pairs<6, 3, 6>();
                __builtin_amdgcn_sched_barrier(0);
                mul_col(o_reg, o_reg, scale_vec);
            }
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 3: V4 into shared memory, K4 into registers.
            G::load<1, false>(v_smem[0], g.Vg, {batch_idx, max_num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V,
                              v_srsrc_base, v_base, v_lds_base_0);
            load_pinned(k_reg, k_smem[0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(2 * memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 4: QK4; finish the softmax of QK3.
            zero(att_block[0]);
            if (NL >= 2)
                mma_ABt(att_block[0], k_reg, q_reg, att_block[0]);
            if (NL >= 1) {
                exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
                mul(norm_vec, norm_vec, scale_vec);
                col_sum(norm_vec, att_block[1], norm_vec);
            }
            copy(att_block_bf16, att_block[1]);
            att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
            sched_barrier_exp_pairs<6, 3, 7>();
            sched_barrier_pairs<10, 5, 7>();
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 5: V3 into registers; mask QK4.
            load(v_reg, v_smem[1]);
            if (NL >= 2) {
                const int kv_end_pos = (max_num_tiles - 1) * KV_BLOCK_SIZE;
                if (__builtin_expect(q_start_pos < kv_end_pos, 1)) {
                    mask_kv_tile(att_block[0], tile_idx, max_num_tiles - 2, neg_inf_v, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 6: A3V3; partial softmax of QK4.
            __builtin_amdgcn_s_setprio(1);
            if (NL >= 1)
                mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
            if (NL >= 2) {
                col_max(max_vec, att_block[0]);
                rv_scale_merge(max_vec, max_vec_prev, TEMPERATURE_SCALE);
                sub(scale_vec, max_vec_prev, max_vec);
                copy(max_vec_prev, max_vec);
                exp2(scale_vec, scale_vec);
                sub_col_fma(att_block[0], att_block[0], TEMPERATURE_SCALE, max_vec);
                exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
                sched_barrier_pairs<10, 5, 8>();
                sched_barrier_exp_pairs<6, 3, 8>();
                __builtin_amdgcn_sched_barrier(0);
                mul_col(o_reg, o_reg, scale_vec);
            }
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 7: V5 into shared memory, K5 into registers.
            G::load<1, false>(v_smem[1], g.Vg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V,
                              v_srsrc_base, v_base, v_lds_base_1);
            load_pinned(k_reg, k_smem[1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(%0)" ::"n"(memcpy_per_tile));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 8: QK5; finish the softmax of QK4.
            zero(att_block[1]);
            if (NL >= 3)
                mma_ABt(att_block[1], k_reg, q_reg, att_block[1]);
            if (NL >= 2) {
                exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
                mul(norm_vec, norm_vec, scale_vec);
                col_sum(norm_vec, att_block[0], norm_vec);
            }
            copy(att_block_bf16, att_block[0]);
            att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
            sched_barrier_exp_pairs<6, 3, 9>();
            sched_barrier_pairs<10, 5, 9>();
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 9: V4 into registers; mask QK5.
            load(v_reg, v_smem[0]);
            if (NL >= 3) {
                const int kv_end_pos = (max_num_tiles) * KV_BLOCK_SIZE;
                if (__builtin_expect(q_start_pos < kv_end_pos, 1)) {
                    mask_kv_tile(att_block[1], tile_idx, max_num_tiles - 1, neg_inf_v, lane);
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 10: A4V4; full softmax of QK5.
            if (NL >= 2)
                mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
            if (NL >= 3) {
                col_max(max_vec, att_block[1]);
                rv_scale_merge(max_vec, max_vec_prev, TEMPERATURE_SCALE);
                sub(scale_vec, max_vec_prev, max_vec);
                copy(max_vec_prev, max_vec);
                exp2(scale_vec, scale_vec);

                sub_col_fma(att_block[1], att_block[1], TEMPERATURE_SCALE, max_vec);
                exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
                sched_barrier_pairs<10, 5, 10>();
                sched_barrier_exp_pairs<6, 3, 10>();
                __builtin_amdgcn_sched_barrier(0);

                exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
                mul(norm_vec, norm_vec, scale_vec);

                col_sum(norm_vec, att_block[1], norm_vec);
                __builtin_amdgcn_sched_barrier(0);
                mul_col(o_reg, o_reg, scale_vec);
            }
            copy(att_block_bf16, att_block[1]);
            att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 11: V5 into registers.
            load(v_reg, v_smem[1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 12: A5V5, then O / l.
            if (NL >= 3)
                mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
            div_col(o_reg, o_reg, norm_vec);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Realign the two wave groups.
            if (!stagger) {
                __builtin_amdgcn_s_barrier();
            }

            qo_tile<D, float, row_l, rt_32x32_s> o_reg_transposed;
            transpose(o_reg_transposed, o_reg);
            if constexpr (D == 128) {
                stage_o_write(o_reg_transposed, ost_stage);
            } else
                store<1>(sat_gl(g.Og), o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

            // LSE = m * ln 2 + log(l).
            mul(max_vec, max_vec, 0.69314718056f);
            log(norm_vec, norm_vec);
            add(norm_vec, norm_vec, max_vec);
            if constexpr (D == 128) {
                ost_lse = norm_vec;
                ost_b = batch_idx; ost_t = tile_idx; ost_h = head_idx;
                ost_pend = 1;
            } else
                store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
        } // Q-pair pass
    } // persistent job
    if constexpr (D == 128) {
        if (ost_pend) {
            stage_o_flush<qo_tile<D, float, row_l, rt_32x32_s>>(sat_gl(g.Og),
                coord<qo_tile<D, float, row_l, rt_32x32_s>>{ost_b, ost_t, ost_h, 0}, ost_stage);
            store(g.L_vec, ost_lse, {ost_b, ost_h, 0, ost_t});
        }
    }
}


template<int D, bool Bshd>
void dispatch_micro(attn_globals<D, Bshd> g) {
    g.nxcd = fmha_device_xcds();
    // D = 64 runs one Q block per job (no causal pairing).
    g.pair = (D >= 128) ? 1 : 0;
    unsigned long mem_size = g.dynamic_shared_memory();
    hip_check(hipFuncSetAttribute((void*)attend_ker<D, Bshd>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size), "hipFuncSetAttribute");
    hipLaunchKernelGGL((attend_ker<D, Bshd>), dim3(g.grid()), dim3(g.block()), mem_size, g.stream, g);
}

namespace fmha_bf16_d128 {

constexpr float L2E = 1.4426950408889634f;

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal,
                                   bool return_lse, bool bshd) {
    TORCH_CHECK(causal, "fmha_bf16 is causal-only");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda());
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q, k, v must be on the same device");
    const c10::cuda::OptionalCUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "fmha_bf16 needs bfloat16");
    TORCH_CHECK(fmha_scale_finite(scale), "fmha_bf16 needs a finite scale");
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
    TORCH_CHECK(D == 128 && Dv == 128 && M % 256 == 0 && N % 64 == 0 && (N >= 256 || N == 0));
    TORCH_CHECK(M == N, "fmha_bf16 needs equal query and key lengths, got M=", M, " N=", N);
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0 && k.size(3) == D && q.size(0) == k.size(0) && q.size(0) == v.size(0));
    if (bshd) {
        TORCH_CHECK(v.size(1) == N && v.size(2) == Hkv);
    } else {
        TORCH_CHECK(v.size(1) == Hkv && v.size(2) == N);
    }
    auto O = bshd ? torch::empty({B, M, Hq, Dv}, q.options().dtype(at::kBFloat16))
                  : torch::empty({B, Hq, M, Dv}, q.options().dtype(at::kBFloat16));
    torch::Tensor Lse = torch::empty({B, Hq, 1, M}, q.options().dtype(at::kFloat));
    if (B == 0 || Hq == 0 || M == 0)
        return return_lse ? std::vector<torch::Tensor>{O, Lse.reshape({B * Hq, M})} : std::vector<torch::Tensor>{O};
    const int numQ = (M / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS;
    const int nq_map = (numQ + 1) / 2;
    const bool split = fmha_split_batch(B, nq_map, Hq, fmha_num_xcds(), M);
    const int batches = split ? B : 1;
    const int B_launch = split ? 1 : B;
    const size_t q_batch = (size_t)M * Hq * D;
    const size_t kv_batch = (size_t)N * Hkv * D;
    // The kernel scales the row max of the raw scores, so scale >= 0. c = scale * log2(e) is kept >= FLT_MIN: a zero
    // or denormal c flushes to 0 on the GPU and masked -inf * 0 is NaN.
    TORCH_CHECK(scale >= 0, "fmha_bf16 d128 needs scale >= 0");
    const float sc_raw = (float)scale * L2E;
    const float sc = sc_raw < 1.17549435e-38f ? 1.17549435e-38f : sc_raw;
    auto launch = [&](auto bshd_tag) {
        constexpr bool Bshd = decltype(bshd_tag)::value;
        for (int b = 0; b < batches; ++b) {
            const uint64_t qoff = (uint64_t)b * q_batch * sizeof(uint16_t);
            const uint64_t kvoff = (uint64_t)b * kv_batch * sizeof(uint16_t);
            auto *qp = reinterpret_cast<bf16 *>(reinterpret_cast<uint8_t *>(q.data_ptr()) + qoff);
            auto *kp = reinterpret_cast<bf16 *>(reinterpret_cast<uint8_t *>(k.data_ptr()) + kvoff);
            auto *vp = reinterpret_cast<bf16 *>(reinterpret_cast<uint8_t *>(v.data_ptr()) + kvoff);
            auto *op = reinterpret_cast<bf16 *>(reinterpret_cast<uint8_t *>(O.data_ptr()) + qoff);
            attn_globals<128, Bshd> g{
                gl_view<bf16, Bshd>(qp, B_launch, M, Hq, D),
                gl_view<bf16, Bshd>(kp, B_launch, N, Hkv, D),
                gl_view<bf16, Bshd>(vp, B_launch, N, Hkv, Dv),
                gl_view<bf16, Bshd>(op, B_launch, M, Hq, Dv),
                make_gl<gl<float, -1, -1, -1, -1>>(
                    reinterpret_cast<uint64_t>(Lse.data_ptr()) + (uint64_t)b * Hq * 1 * M * sizeof(float), B_launch, Hq,
                    1, M),
                at::cuda::getCurrentCUDAStream(),
                8,
                sc,
            };
            dispatch_micro<128, Bshd>(g);
            const hipError_t err = hipGetLastError();
            TORCH_CHECK(err == hipSuccess, "fmha_bf16_d128 launch: ", hipGetErrorString(err));
        }
    };
    if (bshd)
        launch(std::true_type{});
    else
        launch(std::false_type{});
    if (return_lse)
        return {O, Lse.reshape({B * Hq, M})};
    return {O};
}

} // namespace fmha_bf16_d128
