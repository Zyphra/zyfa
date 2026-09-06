#pragma once
// Minimal HipKittens-style register tiles for gfx950. No HipKittens dependency.
//
// Tiles are structs of *named* fx16 / v8b members, not arrays. LLVM will not
// SROA arrays of ext_vectors (they stay in scratch); named members promote
// to VGPRs the same way fmha32's `fx16 acc_o0, acc_o1, ...` do.

#include <hip/hip_runtime.h>
#include <cstdint>

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

// HipKittens delayed rescale: if every lane's Δmax ≤ T, keep old m (no O/L scale).
// RESCALE_SKIP=true (default). RESCALE_SKIP=false always rescales.
#ifndef RESCALE_SKIP
#define RESCALE_SKIP true
#endif
constexpr float RESCALE_THRESHOLD = 8.f;

// llvm.amdgcn.sched.barrier / sched.group.barrier (AMDGPUUsage).
constexpr int MFMA_MASK = 0x08;
constexpr int VALU_MASK = 0x02;
constexpr int VMEM_MASK = 0x10; // includes LDSDMA (buffer_load … lds)
constexpr int EXP_MASK = 0x400;

#define SCHED_GROUP(mask, cnt, id) __builtin_amdgcn_sched_group_barrier(mask, cnt, id)
#define SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)

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

template <int Pairs, int MfmaCnt, int Id>
__device__ __forceinline__ void sched_vmem_mfma() {
    SCHED_GROUP(VMEM_MASK, 1, Id);
    SCHED_GROUP(MFMA_MASK, MfmaCnt, Id);
    if constexpr (Pairs > 1)
        sched_vmem_mfma<Pairs - 1, MfmaCnt, Id>();
}

// HK/aiter G2S: buffer_load_dwordx4 … lds (SGPR rsrc + 32-bit voffset).
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

__device__ __forceinline__ int rv_all_below(float prev, float cur, float T = RESCALE_THRESHOLD) {
    return (cur - prev <= T);
}
__device__ __forceinline__ int wave_all_ok(int lane_ok) { return __all(lane_ok); }

__device__ __forceinline__ v8b v8b_zero() {
    v8b z;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        z[i] = (bf16)0;
    return z;
}
__device__ __forceinline__ fx16 fx16_zero() {
    fx16 z;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        z[i] = 0.f;
    return z;
}

// Q^T 128x32 = 8 v8b (D-chunks).
struct q_tile {
    v8b d0, d1, d2, d3, d4, d5, d6, d7;
};

// K^T 128x64 or V 64x128 = 16 v8b. Same storage; schedule never needs both.
// K index: d*2+n  (d=0..7 D-chunk, n=0..1 K-half).
// V index: sl*4+n (sl=0..3 16-row strip, n=0..3 D-tile).
struct kv_tile {
    v8b d0, d1, d2, d3, d4, d5, d6, d7;
    v8b d8, d9, d10, d11, d12, d13, d14, d15;
};

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

__device__ __forceinline__ void zero_s(s_tile &s) {
    s.a = fx16_zero();
    s.b = fx16_zero();
}
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

// Empty on ROCm 7.2 / clang 22: structs stay in VGPR; the old asm was a
// scheduling barrier on the hottest registers.
__device__ __forceinline__ void pin_kv(kv_tile &) {}
__device__ __forceinline__ void pin_q(q_tile &) {}
__device__ __forceinline__ void pin_s(s_tile &) {}
__device__ __forceinline__ void pin_o(o_tile &) {}

__device__ __forceinline__ v8b p_strip(const s_tile &s, int sl) {
    return sl == 0 ? cvt8(s.a, 0) : sl == 1 ? cvt8(s.a, 8) : sl == 2 ? cvt8(s.b, 0) : cvt8(s.b, 8);
}

__device__ __forceinline__ void mma_pv16_sl(o_tile &o, v8b v0, v8b v1, v8b v2, v8b v3, const s_tile &s, int sl) {
    const v8b ps = p_strip(s, sl);
    o.a = mfma16(v0, ps, o.a);
    o.b = mfma16(v1, ps, o.b);
    o.c = mfma16(v2, ps, o.c);
    o.d = mfma16(v3, ps, o.d);
}

__device__ __forceinline__ void mma_pv_s(o_tile &o, const kv_tile &v, const s_tile &s, int sl) {
    const v8b ps = p_strip(s, sl);
    const v8b v0 = sl == 0 ? v.d0 : sl == 1 ? v.d4 : sl == 2 ? v.d8 : v.d12;
    const v8b v1 = sl == 0 ? v.d1 : sl == 1 ? v.d5 : sl == 2 ? v.d9 : v.d13;
    const v8b v2 = sl == 0 ? v.d2 : sl == 1 ? v.d6 : sl == 2 ? v.d10 : v.d14;
    const v8b v3 = sl == 0 ? v.d3 : sl == 1 ? v.d7 : sl == 2 ? v.d11 : v.d15;
    o.a = mfma16(v0, ps, o.a);
    o.b = mfma16(v1, ps, o.b);
    o.c = mfma16(v2, ps, o.c);
    o.d = mfma16(v3, ps, o.d);
}
__device__ __forceinline__ void mma_pv_s(o_tile &o, const kv_tile &v, const s_tile &s) {
    mma_pv_s(o, v, s, 0);
    mma_pv_s(o, v, s, 1);
    mma_pv_s(o, v, s, 2);
    mma_pv_s(o, v, s, 3);
}

__device__ __forceinline__ fx16 exp2_fx(fx16 v) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        v[r] = exp2f(v[r]);
    return v;
}
__device__ __forceinline__ void exp2_frag(s_tile &s, int i) {
    if (i == 0)
        s.a = exp2_fx(s.a);
    else
        s.b = exp2_fx(s.b);
}

__device__ __forceinline__ void mma_qk16(s_tile &s, v8b d0, v8b d1, v8b d2, v8b d3, v8b d4, v8b d5, v8b d6, v8b d7,
                                        v8b d8, v8b d9, v8b d10, v8b d11, v8b d12, v8b d13, v8b d14, v8b d15,
                                        const q_tile &q) {
    s.a = mfma16(d0, q.d0, s.a);
    s.a = mfma16(d2, q.d1, s.a);
    s.a = mfma16(d4, q.d2, s.a);
    s.a = mfma16(d6, q.d3, s.a);
    s.a = mfma16(d8, q.d4, s.a);
    s.a = mfma16(d10, q.d5, s.a);
    s.a = mfma16(d12, q.d6, s.a);
    s.a = mfma16(d14, q.d7, s.a);
    s.b = mfma16(d1, q.d0, s.b);
    s.b = mfma16(d3, q.d1, s.b);
    s.b = mfma16(d5, q.d2, s.b);
    s.b = mfma16(d7, q.d3, s.b);
    s.b = mfma16(d9, q.d4, s.b);
    s.b = mfma16(d11, q.d5, s.b);
    s.b = mfma16(d13, q.d6, s.b);
    s.b = mfma16(d15, q.d7, s.b);
}

// QK: S += K^T @ Q. k[d*2+n] × q[d] → s[n]
__device__ __forceinline__ void mma_qk(s_tile &s, const kv_tile &k, const q_tile &q) {
    s.a = mfma16(k.d0, q.d0, s.a);
    s.a = mfma16(k.d2, q.d1, s.a);
    s.a = mfma16(k.d4, q.d2, s.a);
    s.a = mfma16(k.d6, q.d3, s.a);
    s.a = mfma16(k.d8, q.d4, s.a);
    s.a = mfma16(k.d10, q.d5, s.a);
    s.a = mfma16(k.d12, q.d6, s.a);
    s.a = mfma16(k.d14, q.d7, s.a);
    s.b = mfma16(k.d1, q.d0, s.b);
    s.b = mfma16(k.d3, q.d1, s.b);
    s.b = mfma16(k.d5, q.d2, s.b);
    s.b = mfma16(k.d7, q.d3, s.b);
    s.b = mfma16(k.d9, q.d4, s.b);
    s.b = mfma16(k.d11, q.d5, s.b);
    s.b = mfma16(k.d13, q.d6, s.b);
    s.b = mfma16(k.d15, q.d7, s.b);
}

// PV: O += V^T @ P. v[sl*4+n] × p[sl] → o[n]
__device__ __forceinline__ void mma_pv(o_tile &o, const kv_tile &v, const p_tile &p) {
    o.a = mfma16(v.d0, p.a, o.a);
    o.a = mfma16(v.d4, p.b, o.a);
    o.a = mfma16(v.d8, p.c, o.a);
    o.a = mfma16(v.d12, p.d, o.a);
    o.b = mfma16(v.d1, p.a, o.b);
    o.b = mfma16(v.d5, p.b, o.b);
    o.b = mfma16(v.d9, p.c, o.b);
    o.b = mfma16(v.d13, p.d, o.b);
    o.c = mfma16(v.d2, p.a, o.c);
    o.c = mfma16(v.d6, p.b, o.c);
    o.c = mfma16(v.d10, p.c, o.c);
    o.c = mfma16(v.d14, p.d, o.c);
    o.d = mfma16(v.d3, p.a, o.d);
    o.d = mfma16(v.d7, p.b, o.d);
    o.d = mfma16(v.d11, p.c, o.d);
    o.d = mfma16(v.d15, p.d, o.d);
}

__device__ __forceinline__ void mma_pv(o_tile &o, const kv_tile &v, const p_tile &p, int sl) {
    // strip 0 → v.d0..d3 × p.a; 1 → d4..d7 × p.b; ...
    const v8b ps = sl == 0 ? p.a : sl == 1 ? p.b : sl == 2 ? p.c : p.d;
    const v8b v0 = sl == 0 ? v.d0 : sl == 1 ? v.d4 : sl == 2 ? v.d8 : v.d12;
    const v8b v1 = sl == 0 ? v.d1 : sl == 1 ? v.d5 : sl == 2 ? v.d9 : v.d13;
    const v8b v2 = sl == 0 ? v.d2 : sl == 1 ? v.d6 : sl == 2 ? v.d10 : v.d14;
    const v8b v3 = sl == 0 ? v.d3 : sl == 1 ? v.d7 : sl == 2 ? v.d11 : v.d15;
    o.a = mfma16(v0, ps, o.a);
    o.b = mfma16(v1, ps, o.b);
    o.c = mfma16(v2, ps, o.c);
    o.d = mfma16(v3, ps, o.d);
}

__device__ __forceinline__ float _fx_max(fx16 v, float mx) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = fmaxf(mx, v[r]);
    return mx;
}
__device__ __forceinline__ float _fx_sum(fx16 v, float t) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        t += v[r];
    return t;
}

__device__ __forceinline__ float col_max(const s_tile &s) {
    float mx = _fx_max(s.b, _fx_max(s.a, neg_inf()));
    return fmaxf(mx, __shfl_xor(mx, 32));
}
__device__ __forceinline__ float col_max(const s_tile &s, float prev) { return fmaxf(col_max(s), prev); }
__device__ __forceinline__ float col_sum(const s_tile &s, float prev) {
    float t = _fx_sum(s.b, _fx_sum(s.a, 0.f));
    return prev + t + __shfl_xor(t, 32);
}

__device__ __forceinline__ fx16 addsc(fx16 v, float m) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        v[r] += m;
    return v;
}
__device__ __forceinline__ fx16 mulsc(fx16 v, float m) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        v[r] *= m;
    return v;
}

__device__ __forceinline__ void sub_col(s_tile &d, const s_tile &s, float m) {
    d.a = addsc(s.a, -m);
    d.b = addsc(s.b, -m);
}
__device__ __forceinline__ void mul_col(o_tile &d, const o_tile &s, float m) {
    d.a = mulsc(s.a, m);
    d.b = mulsc(s.b, m);
    d.c = mulsc(s.c, m);
    d.d = mulsc(s.d, m);
}
__device__ __forceinline__ void div_col(o_tile &d, const o_tile &s, float v) {
    mul_col(d, s, 1.f / v);
}

inline constexpr int kCOff[8][2] = {{0, 1}, {2, 3}, {8, 9}, {10, 11}, {16, 17}, {18, 19}, {24, 25}, {26, 27}};

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

__device__ __forceinline__ void mask_kv(s_tile &dst, int q_abs, int k_abs) {
    const int lane = laneid();
    const int q_pos = q_abs * 32 + (lane & 31);
    const int k_base = k_abs * 64;
    const int rb = ((lane >> 5) << 2);
    const float ninf = neg_inf();
    dst.a = mask_half(dst.a, rb, q_pos, k_base, ninf);
    dst.b = mask_half(dst.b, 32 + rb, q_pos, k_base, ninf);
}

template <int Rows, int Cols>
struct alignas(16) st {
    bf16 data[Rows * Cols];
    __device__ __forceinline__ bf16 *ptr(int r, int c) { return data + r * Cols + c; }
    __device__ __forceinline__ const bf16 *ptr(int r, int c) const { return data + r * Cols + c; }
};

#define HK_GLD16(g, l) __builtin_amdgcn_global_load_lds(                                           \
    (const __attribute__((address_space(1))) void *)(g),                                           \
    (__attribute__((address_space(3))) void *)(l), 16, 0, 0)

// fmha32 K bank-swizzle. 64x128 bf16, 16 chunks/row.
// Write: logical chunk L lives at physical P = (L + key) % 16.
// Read:  P = (L + ky) % 16, L = ds*2 + g.
__device__ __forceinline__ void prefetch_k(st<64, 128> &sm, const bf16 *g, int /*stride*/) {
    constexpr int D = 128, KN = 64, CPR = 16;
#pragma unroll
    for (int c = (int)threadIdx.x; c < KN * D / 8; c += 512) {
        const int key = c / CPR, P = c % CPR;
        const int L = ((P - key) % CPR + CPR) % CPR;
        HK_GLD16(g + (size_t)key * D + L * 8, sm.data + c * 8);
    }
}

// fmha32 V swizzle for ds_read_tr16. Inverse is vphys().
__device__ __forceinline__ void prefetch_v(st<64, 128> &sm, const bf16 *g, int /*stride*/) {
    constexpr int Dv = 128, KN = 64;
#pragma unroll
    for (int c = (int)threadIdx.x; c < KN * Dv / 8; c += 512) {
        const int pp = c * 8, tl = pp / 64, wi = pp % 64;
        const int ky = (tl / (Dv / 16)) * 4 + wi / 16;
        const int dvv = (tl % (Dv / 16)) * 16 + wi % 16;
        HK_GLD16(g + (size_t)ky * Dv + dvv, sm.data + c * 8);
    }
}

__device__ __forceinline__ const bf16 *vphys(const bf16 *lds, int ky, int dv) {
    const int wi = (ky & 3) * 16 + (dv & 15);
    const int tl = (ky >> 2) * 8 + (dv >> 4);
    return lds + tl * 64 + wi;
}

__device__ __forceinline__ v8b load_k_sw(const bf16 *Kc, int ky, int ds, int g) {
    v8b ka;
    const int P = ((ds * 2 + g) + ky) % 16;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        ka[i] = Kc[ky * 128 + P * 8 + i];
    return ka;
}

__device__ __forceinline__ v8b gld_scale(const bf16 *p, float sc) {
    v8b tmp;
    __builtin_memcpy(&tmp, p, 16);
#pragma unroll
    for (int e = 0; e < 8; ++e)
        tmp[e] = (bf16)((float)tmp[e] * sc);
    return tmp;
}

__device__ __forceinline__ void load_q_col(q_tile &dst, const bf16 *g, int stride, float sc) {
    const int lane = laneid();
    const int row_off = lane % 32;
    const int col_off = 8 * (lane / 32);
    dst.d0 = gld_scale(g + (size_t)row_off * stride + 0 * 16 + col_off, sc);
    dst.d1 = gld_scale(g + (size_t)row_off * stride + 1 * 16 + col_off, sc);
    dst.d2 = gld_scale(g + (size_t)row_off * stride + 2 * 16 + col_off, sc);
    dst.d3 = gld_scale(g + (size_t)row_off * stride + 3 * 16 + col_off, sc);
    dst.d4 = gld_scale(g + (size_t)row_off * stride + 4 * 16 + col_off, sc);
    dst.d5 = gld_scale(g + (size_t)row_off * stride + 5 * 16 + col_off, sc);
    dst.d6 = gld_scale(g + (size_t)row_off * stride + 6 * 16 + col_off, sc);
    dst.d7 = gld_scale(g + (size_t)row_off * stride + 7 * 16 + col_off, sc);
}

__device__ __forceinline__ void load_k16(v8b &d0, v8b &d1, v8b &d2, v8b &d3, v8b &d4, v8b &d5, v8b &d6, v8b &d7, v8b &d8,
                                        v8b &d9, v8b &d10, v8b &d11, v8b &d12, v8b &d13, v8b &d14, v8b &d15,
                                        const st<64, 128> &sm) {
    const int c32 = laneid() & 31, g = laneid() >> 5;
    d0 = load_k_sw(sm.data, 0 * 32 + c32, 0, g);
    d1 = load_k_sw(sm.data, 1 * 32 + c32, 0, g);
    d2 = load_k_sw(sm.data, 0 * 32 + c32, 1, g);
    d3 = load_k_sw(sm.data, 1 * 32 + c32, 1, g);
    d4 = load_k_sw(sm.data, 0 * 32 + c32, 2, g);
    d5 = load_k_sw(sm.data, 1 * 32 + c32, 2, g);
    d6 = load_k_sw(sm.data, 0 * 32 + c32, 3, g);
    d7 = load_k_sw(sm.data, 1 * 32 + c32, 3, g);
    d8 = load_k_sw(sm.data, 0 * 32 + c32, 4, g);
    d9 = load_k_sw(sm.data, 1 * 32 + c32, 4, g);
    d10 = load_k_sw(sm.data, 0 * 32 + c32, 5, g);
    d11 = load_k_sw(sm.data, 1 * 32 + c32, 5, g);
    d12 = load_k_sw(sm.data, 0 * 32 + c32, 6, g);
    d13 = load_k_sw(sm.data, 1 * 32 + c32, 6, g);
    d14 = load_k_sw(sm.data, 0 * 32 + c32, 7, g);
    d15 = load_k_sw(sm.data, 1 * 32 + c32, 7, g);
}

__device__ __forceinline__ void load_k_col(kv_tile &dst, const st<64, 128> &sm) {
    const int c32 = laneid() & 31, g = laneid() >> 5;
    dst.d0 = load_k_sw(sm.data, 0 * 32 + c32, 0, g);
    dst.d1 = load_k_sw(sm.data, 1 * 32 + c32, 0, g);
    dst.d2 = load_k_sw(sm.data, 0 * 32 + c32, 1, g);
    dst.d3 = load_k_sw(sm.data, 1 * 32 + c32, 1, g);
    dst.d4 = load_k_sw(sm.data, 0 * 32 + c32, 2, g);
    dst.d5 = load_k_sw(sm.data, 1 * 32 + c32, 2, g);
    dst.d6 = load_k_sw(sm.data, 0 * 32 + c32, 3, g);
    dst.d7 = load_k_sw(sm.data, 1 * 32 + c32, 3, g);
    dst.d8 = load_k_sw(sm.data, 0 * 32 + c32, 4, g);
    dst.d9 = load_k_sw(sm.data, 1 * 32 + c32, 4, g);
    dst.d10 = load_k_sw(sm.data, 0 * 32 + c32, 5, g);
    dst.d11 = load_k_sw(sm.data, 1 * 32 + c32, 5, g);
    dst.d12 = load_k_sw(sm.data, 0 * 32 + c32, 6, g);
    dst.d13 = load_k_sw(sm.data, 1 * 32 + c32, 6, g);
    dst.d14 = load_k_sw(sm.data, 0 * 32 + c32, 7, g);
    dst.d15 = load_k_sw(sm.data, 1 * 32 + c32, 7, g);
}

__device__ __forceinline__ v4b trd16(const v4b *p) {
    return __builtin_amdgcn_ds_read_tr16_b64_v4bf16((__attribute__((address_space(3))) v4b *)p);
}

__device__ __forceinline__ v8b load_v_frag(const st<64, 128> &sm, int i, int j, int row_off, int col_off) {
    const int r = i * 16 + row_off, c = j * 32 + col_off;
    v4b r0 = trd16((const v4b *)vphys(sm.data, r, c));
    v4b r1 = trd16((const v4b *)vphys(sm.data, r + 8, c));
    v8b f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        f[t] = r0[t];
        f[4 + t] = r1[t];
    }
    return f;
}

__device__ __forceinline__ void load_v16(v8b &d0, v8b &d1, v8b &d2, v8b &d3, v8b &d4, v8b &d5, v8b &d6, v8b &d7, v8b &d8,
                                        v8b &d9, v8b &d10, v8b &d11, v8b &d12, v8b &d13, v8b &d14, v8b &d15,
                                        const st<64, 128> &sm) {
    const int lane = laneid();
    const int row_off = ((lane % 16) / 4) + ((lane / 32) * 4);
    const int col_off = ((lane % 4) * 4) + (16 * ((lane % 32) / 16));
    d0 = load_v_frag(sm, 0, 0, row_off, col_off);
    d1 = load_v_frag(sm, 0, 1, row_off, col_off);
    d2 = load_v_frag(sm, 0, 2, row_off, col_off);
    d3 = load_v_frag(sm, 0, 3, row_off, col_off);
    d4 = load_v_frag(sm, 1, 0, row_off, col_off);
    d5 = load_v_frag(sm, 1, 1, row_off, col_off);
    d6 = load_v_frag(sm, 1, 2, row_off, col_off);
    d7 = load_v_frag(sm, 1, 3, row_off, col_off);
    d8 = load_v_frag(sm, 2, 0, row_off, col_off);
    d9 = load_v_frag(sm, 2, 1, row_off, col_off);
    d10 = load_v_frag(sm, 2, 2, row_off, col_off);
    d11 = load_v_frag(sm, 2, 3, row_off, col_off);
    d12 = load_v_frag(sm, 3, 0, row_off, col_off);
    d13 = load_v_frag(sm, 3, 1, row_off, col_off);
    d14 = load_v_frag(sm, 3, 2, row_off, col_off);
    d15 = load_v_frag(sm, 3, 3, row_off, col_off);
}

__device__ __forceinline__ void load_v(kv_tile &dst, const st<64, 128> &sm) {
    const int lane = laneid();
    const int row_off = ((lane % 16) / 4) + ((lane / 32) * 4);
    const int col_off = ((lane % 4) * 4) + (16 * ((lane % 32) / 16));
    dst.d0 = load_v_frag(sm, 0, 0, row_off, col_off);
    dst.d1 = load_v_frag(sm, 0, 1, row_off, col_off);
    dst.d2 = load_v_frag(sm, 0, 2, row_off, col_off);
    dst.d3 = load_v_frag(sm, 0, 3, row_off, col_off);
    dst.d4 = load_v_frag(sm, 1, 0, row_off, col_off);
    dst.d5 = load_v_frag(sm, 1, 1, row_off, col_off);
    dst.d6 = load_v_frag(sm, 1, 2, row_off, col_off);
    dst.d7 = load_v_frag(sm, 1, 3, row_off, col_off);
    dst.d8 = load_v_frag(sm, 2, 0, row_off, col_off);
    dst.d9 = load_v_frag(sm, 2, 1, row_off, col_off);
    dst.d10 = load_v_frag(sm, 2, 2, row_off, col_off);
    dst.d11 = load_v_frag(sm, 2, 3, row_off, col_off);
    dst.d12 = load_v_frag(sm, 3, 0, row_off, col_off);
    dst.d13 = load_v_frag(sm, 3, 1, row_off, col_off);
    dst.d14 = load_v_frag(sm, 3, 2, row_off, col_off);
    dst.d15 = load_v_frag(sm, 3, 3, row_off, col_off);
}

// Pack two f32 → one bf16 pair (lo=a, hi=b). Matches AITER v_cvt_pk_bf16_f32.
__device__ __forceinline__ uint32_t cvt_pk_bf16(float a, float b) {
    uint32_t p;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(p) : "v"(a), "v"(b));
    return p;
}

// C-layout groups {0,1,2,3}, {8,9,10,11}, … — 4 contiguous d, 4× dwordx2 of bf16.
__device__ __forceinline__ void store_half(bf16 *O, int o_stride, int q, int i, fx16 v) {
    const int lane = laneid();
    const int d0 = i * 32 + ((lane >> 5) << 2);
    bf16 *row = O + (size_t)q * o_stride + d0;
    typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int k = g * 2;
        const u32x2 pk = {cvt_pk_bf16(v[k * 2], v[k * 2 + 1]),
                          cvt_pk_bf16(v[(k + 1) * 2], v[(k + 1) * 2 + 1])};
        *reinterpret_cast<u32x2 *>(row + g * 8) = pk;
    }
}

// Stream K/V from LDS during MMA so the 64-VGPR kv tile never lives in registers.
__device__ __forceinline__ void mma_qk(s_tile &s, const st<64, 128> &sm, const q_tile &q) {
    const int c32 = laneid() & 31, g = laneid() >> 5;
    const bf16 *Kc = sm.data;
    const int ky0 = c32, ky1 = 32 + c32;
    s.a = mfma16(load_k_sw(Kc, ky0, 0, g), q.d0, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 1, g), q.d1, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 2, g), q.d2, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 3, g), q.d3, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 4, g), q.d4, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 5, g), q.d5, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 6, g), q.d6, s.a);
    s.a = mfma16(load_k_sw(Kc, ky0, 7, g), q.d7, s.a);
    s.b = mfma16(load_k_sw(Kc, ky1, 0, g), q.d0, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 1, g), q.d1, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 2, g), q.d2, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 3, g), q.d3, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 4, g), q.d4, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 5, g), q.d5, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 6, g), q.d6, s.b);
    s.b = mfma16(load_k_sw(Kc, ky1, 7, g), q.d7, s.b);
}

__device__ __forceinline__ void wait_lgkm0() { asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); }

// Drain tr16 into v0..v3. +v pins the fragments so LLVM cannot issue MFMA first.
__device__ __forceinline__ void wait_vfrag(v8b &v0, v8b &v1, v8b &v2, v8b &v3) {
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3)::"memory");
}

template <int SL>
__device__ __forceinline__ void mma_pv_sl(o_tile &o, const st<64, 128> &sm, const p_tile &p) {
    const int lane = laneid();
    const int row_off = ((lane % 16) / 4) + ((lane / 32) * 4);
    const int col_off = ((lane % 4) * 4) + (16 * ((lane % 32) / 16));
    const v8b ps = SL == 0 ? p.a : SL == 1 ? p.b : SL == 2 ? p.c : p.d;
    v8b v0 = load_v_frag(sm, SL, 0, row_off, col_off);
    v8b v1 = load_v_frag(sm, SL, 1, row_off, col_off);
    v8b v2 = load_v_frag(sm, SL, 2, row_off, col_off);
    v8b v3 = load_v_frag(sm, SL, 3, row_off, col_off);
    wait_vfrag(v0, v1, v2, v3);
    o.a = mfma16(v0, ps, o.a);
    o.b = mfma16(v1, ps, o.b);
    o.c = mfma16(v2, ps, o.c);
    o.d = mfma16(v3, ps, o.d);
}

__device__ __forceinline__ void mma_pv(o_tile &o, const st<64, 128> &sm, const p_tile &p) {
    mma_pv_sl<0>(o, sm, p);
    mma_pv_sl<1>(o, sm, p);
    mma_pv_sl<2>(o, sm, p);
    mma_pv_sl<3>(o, sm, p);
}

__device__ __forceinline__ void mma_pv(o_tile &o, const st<64, 128> &sm, const p_tile &p, int sl) {
    if (sl == 0)
        mma_pv_sl<0>(o, sm, p);
    else if (sl == 1)
        mma_pv_sl<1>(o, sm, p);
    else if (sl == 2)
        mma_pv_sl<2>(o, sm, p);
    else
        mma_pv_sl<3>(o, sm, p);
}

// One PV slice with the P strip by value so the caller can produce each strip
// immediately before its MFMA. No hand-written s_waitcnt: load_v_frag is a
// builtin, so the compiler batches the lgkmcnt drains.
template <int SL>
__device__ __forceinline__ void mma_pv_one(o_tile &o, const st<64, 128> &sm, v8b ps) {
    const int lane = laneid();
    const int row_off = ((lane % 16) / 4) + ((lane / 32) * 4);
    const int col_off = ((lane % 4) * 4) + (16 * ((lane % 32) / 16));
    const v8b v0 = load_v_frag(sm, SL, 0, row_off, col_off);
    const v8b v1 = load_v_frag(sm, SL, 1, row_off, col_off);
    const v8b v2 = load_v_frag(sm, SL, 2, row_off, col_off);
    const v8b v3 = load_v_frag(sm, SL, 3, row_off, col_off);
    o.a = mfma16(v0, ps, o.a);
    o.b = mfma16(v1, ps, o.b);
    o.c = mfma16(v2, ps, o.c);
    o.d = mfma16(v3, ps, o.d);
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

#undef HK_GLD16

} // namespace hk
