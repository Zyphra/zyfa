#pragma once
// FP8 fat 32x32x64 tiles for HK-schedule D=128. KN=128.
// Stream K/V from LDS (no kv register tile) to keep VGPR/spills down.
#include "rt.h"
#include <hip/hip_fp8.h>

namespace hk8 {

using namespace hk;
using fp8 = __hip_fp8_e4m3;
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef int v2i32 __attribute__((ext_vector_type(2)));

constexpr int KN = 128, D = 128, Dv = 128, NKB = 4, NDVB = 4;

__device__ __forceinline__ v2i32 trd8(const v2i32 *q) {
    return __builtin_amdgcn_ds_read_tr8_b64_v2i32((__attribute__((address_space(3))) v2i32 *)q);
}

typedef int i32x4 __attribute__((ext_vector_type(4)));

typedef float f2 __attribute__((ext_vector_type(2)));

// Aiter ASM uses unscaled v_mfma_f32_32x32x64_f8f6f4, not the scale opcode.
__device__ __forceinline__ fx16 kh_mma(i32x8 a, i32x8 b, fx16 c) {
    fx16 o = c;
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0" : "+v"(o) : "v"(a), "v"(b));
    return o;
}

// First QK fragment: acc is the literal 0 (aiter). Avoids a 64-vmov zero_s4 block.
__device__ __forceinline__ fx16 kh_mma0(i32x8 a, i32x8 b) {
    fx16 o;
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, 0" : "=v"(o) : "v"(a), "v"(b));
    return o;
}

__device__ __forceinline__ float max3(float a, float b, float c) {
    float r;
    asm volatile("v_max3_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

#define HK8_GLD16(g, l) __builtin_amdgcn_global_load_lds(                                           \
    (const __attribute__((address_space(1))) void *)(g),                                           \
    (__attribute__((address_space(3))) void *)(l), 16, 0, 0)

struct alignas(16) st8 {
    unsigned char data[KN * D];
};

struct q8_tile {
    i32x8 d0, d1;
};

// 128 keys = four 32-key C tiles. Named so LLVM SROAs (arrays of fx16 spill).
struct s4_tile {
    fx16 a, b, c, d;
};

__device__ __forceinline__ void pin_q8(q8_tile &t) { asm volatile("" : "+v"(t.d0), "+v"(t.d1)); }
__device__ __forceinline__ void pin_s4(s4_tile &t) { asm volatile("" : "+v"(t.a), "+v"(t.b), "+v"(t.c), "+v"(t.d)); }

__device__ __forceinline__ int fmha_qoff(int r, int g) { return (r / 4) * 8 + g * 4 + (r % 4); }

__device__ __forceinline__ i32x8 load32(const unsigned char *p) {
    i32x8 x;
    __builtin_memcpy(&x, p, 32);
    return x;
}

// Concat two b128 into the 8-VGPR MFMA A operand. LLVM usually emits a
// free REG_SEQUENCE rather than v_mov if the halves stay in VGPRs.
__device__ __forceinline__ i32x8 pack8(i32x4 lo, i32x4 hi) {
    return (i32x8){lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}

// Inline-asm DS reads are invisible to SIInsertWaitcnts (it only scores
// real DS opcodes). That is what lets us keep lgkmcnt(2) in front of MFMA
// instead of the compiler injecting lgkmcnt(0).
__device__ __forceinline__ uint32_t lds_u32(const unsigned char *p) {
    uint32_t off;
    asm volatile("v_mov_b32 %0, %1"
                 : "=v"(off)
                 : "v"((const __attribute__((address_space(3))) unsigned char *)p));
    return off;
}

template <int Off>
__device__ __forceinline__ void ds_b128(i32x4 &dst, uint32_t addr) {
    if constexpr (Off == 0)
        asm volatile("ds_read_b128 %0, %1" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 16)
        asm volatile("ds_read_b128 %0, %1 offset:16" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 4096)
        asm volatile("ds_read_b128 %0, %1 offset:4096" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 4112)
        asm volatile("ds_read_b128 %0, %1 offset:4112" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 8192)
        asm volatile("ds_read_b128 %0, %1 offset:8192" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 8208)
        asm volatile("ds_read_b128 %0, %1 offset:8208" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 12288)
        asm volatile("ds_read_b128 %0, %1 offset:12288" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 12304)
        asm volatile("ds_read_b128 %0, %1 offset:12304" : "=&v"(dst) : "v"(addr));
    else
        static_assert(Off == 0, "ds_b128 offset");
}

__device__ __forceinline__ v2i32 ds_tr8(uint32_t addr) {
    v2i32 r;
    asm volatile("ds_read_b64_tr_b8 %0, %1" : "=&v"(r) : "v"(addr));
    return r;
}

__device__ __forceinline__ void zero_s4(s4_tile &s) {
    s.a = fx16_zero();
    s.b = fx16_zero();
    s.c = fx16_zero();
    s.d = fx16_zero();
}

// One GLD wave (512×16B). Two passes cover KN×D. Issued between QK MFMAs.
template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st8 &sm, const unsigned char *g) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / (D / 16);
    const int off16 = (c % (D / 16)) * 16;
    const int P = off16 / 32, r = off16 % 32;
    const int L = (P - key) & 3;
    HK8_GLD16(g + (size_t)key * D + L * 32 + r, sm.data + key * D + P * 32 + r);
}

template <int Pass>
__device__ __forceinline__ void prefetch_v_pass(st8 &sm, const unsigned char *g) {
    constexpr int NK_BITS = 2;
    const int c = (int)threadIdx.x + Pass * 512;
    const int pp = c * 16, tile = pp / 128, t = (pp % 128) / 16, dg = tile % (Dv / 16),
              rest = tile / (Dv / 16);
    const int half = rest & 1, nk = (rest >> 1) & (NKB - 1), gt = (rest >> (1 + NK_BITS)) & 1;
    const int key = nk * 32 + gt * 4 + half * 16 + (t / 4) * 8 + (t % 4);
    HK8_GLD16(g + (size_t)key * Dv + dg * 16, sm.data + c * 16);
}

__device__ __forceinline__ void prefetch_k(st8 &sm, const unsigned char *g) {
    prefetch_k_pass<0>(sm, g);
    prefetch_k_pass<1>(sm, g);
}

__device__ __forceinline__ void prefetch_v(st8 &sm, const unsigned char *g) {
    prefetch_v_pass<0>(sm, g);
    prefetch_v_pass<1>(sm, g);
}

__device__ __forceinline__ void load_q8(q8_tile &dst, const unsigned char *g) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load32(g + (size_t)c32 * D + 0 * 64 + gb * 32);
    dst.d1 = load32(g + (size_t)c32 * D + 1 * 64 + gb * 32);
}

__device__ __forceinline__ i32x8 pack_v2x4(v2i32 r0, v2i32 r1, v2i32 r2, v2i32 r3) {
    return (i32x8){r0[0], r0[1], r1[0], r1[1], r2[0], r2[1], r3[0], r3[1]};
}

__device__ __forceinline__ uint32_t v_tr8_off(uint32_t vbase, int j, int nk, int half) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int dg = j * 2 + (c32 >> 4);
    const int tile = ((gb * NKB + nk) * 2 + half) * (Dv / 16) + dg;
    return vbase + (uint32_t)(tile * 128 + (c32 & 15) * 8);
}

__device__ __forceinline__ uint32_t k_off(uint32_t kbase, int ds) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int P = ((ds * 2 + gb) + c32) & 3;
    return kbase + (uint32_t)(c32 * D + P * 32);
}

// 4× ds_read + lgkmcnt(2) in one asm so LLVM cannot split the pipe.
template <int Base>
__device__ __forceinline__ void ds4_wait2(i32x4 &a0, i32x4 &a1, i32x4 &b0, i32x4 &b1, uint32_t off) {
    if constexpr (Base == 0)
        asm volatile("ds_read_b128 %[a0], %[addr]\n\t"
                     "ds_read_b128 %[a1], %[addr] offset:16\n\t"
                     "ds_read_b128 %[b0], %[addr] offset:4096\n\t"
                     "ds_read_b128 %[b1], %[addr] offset:4112\n\t"
                     "s_waitcnt lgkmcnt(2)"
                     : [a0] "=&v"(a0), [a1] "=&v"(a1), [b0] "=&v"(b0), [b1] "=&v"(b1)
                     : [addr] "v"(off));
    else
        asm volatile("ds_read_b128 %[a0], %[addr] offset:8192\n\t"
                     "ds_read_b128 %[a1], %[addr] offset:8208\n\t"
                     "ds_read_b128 %[b0], %[addr] offset:12288\n\t"
                     "ds_read_b128 %[b1], %[addr] offset:12304\n\t"
                     "s_waitcnt lgkmcnt(2)"
                     : [a0] "=&v"(a0), [a1] "=&v"(a1), [b0] "=&v"(b0), [b1] "=&v"(b1)
                     : [addr] "v"(off));
}

template <bool Acc>
__device__ __forceinline__ void mma_qk_pair(s4_tile &s, uint32_t off, i32x8 qb, int nk0) {
    i32x4 a0, a1, b0, b1;
    if (nk0 == 0)
        ds4_wait2<0>(a0, a1, b0, b1, off);
    else
        ds4_wait2<8192>(a0, a1, b0, b1, off);
    fx16 &lo = nk0 ? s.c : s.a;
    fx16 &hi = nk0 ? s.d : s.b;
    if constexpr (Acc)
        lo = kh_mma(pack8(a0, a1), qb, lo);
    else
        lo = kh_mma0(pack8(a0, a1), qb);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(b0), "+v"(b1));
    if constexpr (Acc)
        hi = kh_mma(pack8(b0, b1), qb, hi);
    else
        hi = kh_mma0(pack8(b0, b1), qb);
}

__device__ __forceinline__ void mma_pv_lds(o_tile &o, uint32_t vbase, i32x8 ps, int nk0) {
    v2i32 a0 = ds_tr8(v_tr8_off(vbase, 0, nk0, 0)), a1 = ds_tr8(v_tr8_off(vbase, 0, nk0, 1));
    v2i32 a2 = ds_tr8(v_tr8_off(vbase, 0, nk0 + 1, 0)), a3 = ds_tr8(v_tr8_off(vbase, 0, nk0 + 1, 1));
    v2i32 b0 = ds_tr8(v_tr8_off(vbase, 1, nk0, 0)), b1 = ds_tr8(v_tr8_off(vbase, 1, nk0, 1));
    v2i32 b2 = ds_tr8(v_tr8_off(vbase, 1, nk0 + 1, 0)), b3 = ds_tr8(v_tr8_off(vbase, 1, nk0 + 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3));
    o.a = kh_mma(ps, pack_v2x4(a0, a1, a2, a3), o.a);
    v2i32 c0 = ds_tr8(v_tr8_off(vbase, 2, nk0, 0)), c1 = ds_tr8(v_tr8_off(vbase, 2, nk0, 1));
    v2i32 c2 = ds_tr8(v_tr8_off(vbase, 2, nk0 + 1, 0)), c3 = ds_tr8(v_tr8_off(vbase, 2, nk0 + 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma(ps, pack_v2x4(b0, b1, b2, b3), o.b);
    v2i32 d0 = ds_tr8(v_tr8_off(vbase, 3, nk0, 0)), d1 = ds_tr8(v_tr8_off(vbase, 3, nk0, 1));
    v2i32 d2 = ds_tr8(v_tr8_off(vbase, 3, nk0 + 1, 0)), d3 = ds_tr8(v_tr8_off(vbase, 3, nk0 + 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma(ps, pack_v2x4(c0, c1, c2, c3), o.c);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.d = kh_mma(ps, pack_v2x4(d0, d1, d2, d3), o.d);
}

__device__ __forceinline__ void mask_s4(s4_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
    s.c = mask_half(s.c, rb, q_pos, k_base + 64, ninf);
    s.d = mask_half(s.d, rb, q_pos, k_base + 96, ninf);
}

__device__ __forceinline__ float col_max4_sc(const s4_tile &s, float scale) {
    // scale is positive for attention. max(scale*x) == scale*max(x),
    // so keep the reduction in raw QK space and pay one multiply after
    // the cross-half reduction instead of 64 multiplies per lane.
    float mx = neg_inf();
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = max3(max3(s.a[r], s.b[r], s.c[r]), s.d[r], mx);
    return fmaxf(mx, __shfl_xor(mx, 32)) * scale;
}

// Packed mul+add, not fma: fmaf(-inf, scale, -mnew) nans when mnew is -inf.
__device__ __forceinline__ void exp2_sc(fx16 &s, float scale, float mnew) {
    const f2 sc = {scale, scale};
    const f2 nm = {-mnew, -mnew};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f2 v = {s[2 * i], s[2 * i + 1]};
        asm volatile("v_pk_mul_f32 %0, %1, %2" : "+v"(v) : "v"(sc));
        asm volatile("v_pk_add_f32 %0, %1, %2" : "+v"(v) : "v"(nm));
        s[2 * i] = exp2f(v[0]);
        s[2 * i + 1] = exp2f(v[1]);
    }
}

__device__ __forceinline__ float col_sum4(const s4_tile &s) {
    float t = _fx_sum(s.d, _fx_sum(s.c, _fx_sum(s.b, _fx_sum(s.a, 0.f))));
    return t + __shfl_xor(t, 32);
}

__device__ __forceinline__ void sub_s4(s4_tile &s, float m) {
    s.a = addsc(s.a, -m);
    s.b = addsc(s.b, -m);
    s.c = addsc(s.c, -m);
    s.d = addsc(s.d, -m);
}

__device__ __forceinline__ void exp2_s4(s4_tile &s) {
    s.a = exp2_fx(s.a);
    s.b = exp2_fx(s.b);
    s.c = exp2_fx(s.c);
    s.d = exp2_fx(s.d);
}

__device__ __forceinline__ int pk_fp8_4(float a, float b, float c, float d) {
    const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
}

__device__ __forceinline__ i32x8 pack_p(const fx16 &lo, const fx16 &hi) {
    i32x8 pa;
    pa[0] = pk_fp8_4(lo[0], lo[1], lo[2], lo[3]);
    pa[1] = pk_fp8_4(lo[4], lo[5], lo[6], lo[7]);
    pa[2] = pk_fp8_4(lo[8], lo[9], lo[10], lo[11]);
    pa[3] = pk_fp8_4(lo[12], lo[13], lo[14], lo[15]);
    pa[4] = pk_fp8_4(hi[0], hi[1], hi[2], hi[3]);
    pa[5] = pk_fp8_4(hi[4], hi[5], hi[6], hi[7]);
    pa[6] = pk_fp8_4(hi[8], hi[9], hi[10], hi[11]);
    pa[7] = pk_fp8_4(hi[12], hi[13], hi[14], hi[15]);
    return pa;
}

__device__ __forceinline__ void store_o_tile(float *O, int q_base, int M, const fx16 &t, int j, const float *inv) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int q = q_base + fmha_qoff(r, gb);
        if (q < M)
            O[(size_t)q * Dv + j * 32 + c32] = t[r] * inv[r];
    }
}

__device__ __forceinline__ void store_o_fp8(float *O, int q_base, int M, const o_tile &o, float l_) {
    const int gb = laneid() >> 5;
    float inv[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        inv[r] = __builtin_amdgcn_rcpf(__shfl(l_, fmha_qoff(r, gb)));
    store_o_tile(O, q_base, M, o.a, 0, inv);
    store_o_tile(O, q_base, M, o.b, 1, inv);
    store_o_tile(O, q_base, M, o.c, 2, inv);
    store_o_tile(O, q_base, M, o.d, 3, inv);
}

#undef HK8_GLD16

} // namespace hk8
