#pragma once
// Register tiles and helpers for the gfx950 kernels built on rt*.h (BF16 d192 and FP8 d128 / d192).
// Tiles are structs of named vector members, not arrays: LLVM keeps arrays of ext_vectors in scratch.

#include <hip/hip_runtime.h>
#include <cstdint>
#include "llvm.h"

namespace hk {

using bf16 = __bf16;
using fx16 = float __attribute__((ext_vector_type(16)));
using v8b = bf16 __attribute__((ext_vector_type(8)));
using v4b = bf16 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int laneid() { return (int)threadIdx.x & 63; }
__device__ __forceinline__ int warpid() { return (int)threadIdx.x >> 6; }

__device__ __forceinline__ float neg_inf() {
    uint32_t u = 0xff800000u;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ fx16 mfma16(v8b a, v8b b, fx16 c) {
    return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
}

// Written as fmaxf so the hazard recognizer sees the read; LLVM still emits v_max3_f32.
__device__ __forceinline__ float max3(float a, float b, float c) { return fmaxf(fmaxf(a, b), c); }

// Max / sum with the other 32-lane half (v_permlane32_swap).
typedef int _v2i __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float xor32_max(float x) {
    const int xi = __float_as_int(x);
    const _v2i r = __builtin_amdgcn_permlane32_swap(xi, xi, false, false);
    return fmaxf(__int_as_float(r[0]), __int_as_float(r[1]));
}
__device__ __forceinline__ float xor32_sum(float x) {
    const int xi = __float_as_int(x);
    const _v2i r = __builtin_amdgcn_permlane32_swap(xi, xi, false, false);
    return __int_as_float(r[0]) + __int_as_float(r[1]);
}

// Lazy rescale: the running max is kept while no row's max grows by more than this (log2 units).
constexpr float RESCALE_THRESHOLD = 8.f;

// Instruction classes of llvm.amdgcn.sched.group.barrier.
constexpr int MFMA_MASK = 0x08;
constexpr int VALU_MASK = 0x02;
constexpr int EXP_MASK = 0x400;

#define SCHED_GROUP(mask, cnt, id) __builtin_amdgcn_sched_group_barrier(mask, cnt, id)
#define SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)

// Pairs x (one MFMA, then OtherCnt instructions of OtherMask).
template <int Pairs, int OtherCnt, int OtherMask, int Id>
__device__ __forceinline__ void sched_mfma_pairs() {
    SCHED_GROUP(MFMA_MASK, 1, Id);
    SCHED_GROUP(OtherMask, OtherCnt, Id);
    if constexpr (Pairs > 1)
        sched_mfma_pairs<Pairs - 1, OtherCnt, OtherMask, Id>();
}

template <int Pairs, int ValuCnt, int Id>
__device__ __forceinline__ void sched_valu_pairs() {
    sched_mfma_pairs<Pairs, ValuCnt, VALU_MASK, Id>();
}

template <int Pairs, int ExpCnt, int Id>
__device__ __forceinline__ void sched_exp_pairs() {
    sched_mfma_pairs<Pairs, ExpCnt, EXP_MASK, Id>();
}

// Global -> LDS copies with buffer_load_dwordx4 ... lds (SGPR resource + 32-bit voffset).
using g2s_i32x4 = int32_t __attribute__((ext_vector_type(4)));
using g2s_as3 = uint32_t __attribute__((address_space(3))) *;

struct g2s_rsrc {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ __forceinline__ g2s_i32x4 make_srsrc(const void *ptr, uint32_t range_bytes) {
    g2s_rsrc r{reinterpret_cast<uint64_t>(ptr), range_bytes, 0x110000u};
    g2s_i32x4 out;
    __builtin_memcpy(&out, &r, 16);
    return out;
}

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(g2s_i32x4 rsrc, g2s_as3 lds_ptr, int size, int voffset,
                                                          int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ __forceinline__ void g2s16(const void *gbase, void *lds, uint32_t voff, uint32_t range) {
    llvm_amdgcn_raw_buffer_load_lds(make_srsrc(gbase, range), (g2s_as3)(uintptr_t)lds, 16, (int)voff, 0, 0, 0);
}

__device__ __forceinline__ fx16 fx16_zero() {
    fx16 z;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        z[i] = 0.f;
    return z;
}

// O^T 128x32 = 4 fx16.
struct o_tile {
    fx16 a, b, c, d;
};

// S^T 64x32 = 2 fx16 (two 32-row halves).
struct s_tile {
    fx16 a, b;
};

// P 64x32 = 4 v8b (16-row PV slices).
struct p_tile {
    v8b a, b, c, d;
};

__device__ __forceinline__ void zero_o(o_tile &o) {
    o.a = fx16_zero();
    o.b = fx16_zero();
    o.c = fx16_zero();
    o.d = fx16_zero();
}

__device__ __forceinline__ v8b cvt8(fx16 v, int off) {
    v8b f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
        f[e] = (bf16)v[off + e];
    return f;
}

__device__ __forceinline__ void copy_p(p_tile &d, const s_tile &s) {
    d.a = cvt8(s.a, 0);
    d.b = cvt8(s.a, 8);
    d.c = cvt8(s.b, 0);
    d.d = cvt8(s.b, 8);
}

__device__ __forceinline__ fx16 mulsc(fx16 v, float m) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        v[r] *= m;
    return v;
}

// Rows of accumulator elements 2k and 2k+1 in a 32x32 MFMA C tile, relative to 4 * (lane >> 5).
inline constexpr int kCOff[8][2] = {{0, 1}, {2, 3}, {8, 9}, {10, 11}, {16, 17}, {18, 19}, {24, 25}, {26, 27}};

// Set the scores of keys after q_pos to -inf in one 32-key C tile.
__device__ __forceinline__ fx16 mask_half(fx16 x, int row_base, int q_pos, int k_base, float ninf) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k_base + row_base + kCOff[k][0] > q_pos)
            x[k * 2] = ninf;
        if (k_base + row_base + kCOff[k][1] > q_pos)
            x[k * 2 + 1] = ninf;
    }
    return x;
}

// Element-wise stores of one 32-wide C tile of O^T (row q of O, columns i*32 ...).
__device__ __forceinline__ void store_half(bf16 *O, int o_stride, int q, int i, fx16 v) {
    const int lane = laneid();
    const int d0 = i * 32 + ((lane >> 5) << 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        O[(size_t)q * o_stride + d0 + kCOff[k][0]] = (bf16)v[k * 2];
        O[(size_t)q * o_stride + d0 + kCOff[k][1]] = (bf16)v[k * 2 + 1];
    }
}

__device__ __forceinline__ void store_o(bf16 *O, int o_stride, int q_base, int M, const o_tile &o) {
    const int q = q_base + (laneid() & 31);
    if (q >= M)
        return;
    store_half(O, o_stride, q, 0, o.a);
    store_half(O, o_stride, q, 1, o.b);
    store_half(O, o_stride, q, 2, o.c);
    store_half(O, o_stride, q, 3, o.d);
}

__device__ __forceinline__ void store_lse(float *Lse, int q_base, int M, float lse) {
    const int q = q_base + (laneid() & 31);
    if (q < M)
        Lse[q] = lse;
}

} // namespace hk
