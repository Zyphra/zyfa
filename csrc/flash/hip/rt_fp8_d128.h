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

// Compiler still emits v_max3_f32; the read is visible to the hazard recogniser.
__device__ __forceinline__ float max3(float a, float b, float c) { return fmaxf(fmaxf(a, b), c); }

// gfx950: two copies of x → {[lo,lo],[hi,hi]}. Builtin only — inline asm nans.
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

// G2S via hk::g2s16 (buffer_load … lds).

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

// One QK pair (4x b128 = 16 VGPR). Reused: K -> (dead in softmax) -> next K.
// Aiter aliases these onto v[20:35]. Named members so they stay in VGPR.
struct k_frag {
    i32x4 a0, a1, b0, b1;
};

__device__ __forceinline__ void pin_q8(q8_tile &q) {
    asm volatile("" : "+v"(q.d0), "+v"(q.d1));
}
__device__ __forceinline__ void pin_s4(s4_tile &) {}
__device__ __forceinline__ void pin_kf(k_frag &f) {
    asm volatile("" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
}

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
__device__ __forceinline__ void prefetch_k_pass(st8 &sm, const unsigned char *g, int seq_stride = D) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / (D / 16);
    const int off16 = (c % (D / 16)) * 16;
    const int P = off16 / 32, r = off16 % 32;
    // HBM-only (G2S dest stays c*16). XOR (key>>2) so keys 4 apart do not
    // share a 32B slot; 4-way → 2-way on CDNA4 ds_read_b128 phases.
    const int L = (P ^ (key >> 2)) & 3;
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride;
    const uint32_t voff = (uint32_t)key * (uint32_t)seq_stride + (uint32_t)L * 32u + (uint32_t)r;
    g2s16(g, sm.data + key * D + P * 32 + r, voff, range);
}

template <int Pass>
__device__ __forceinline__ void prefetch_v_pass(st8 &sm, const unsigned char *g, int seq_stride = Dv) {
    constexpr int NK_BITS = 2;
    const int c = (int)threadIdx.x + Pass * 512;
    const int pp = c * 16, tile = pp / 128, t = (pp % 128) / 16, dg = tile % (Dv / 16),
              rest = tile / (Dv / 16);
    const int half = rest & 1, nk = (rest >> 1) & (NKB - 1), gt = (rest >> (1 + NK_BITS)) & 1;
    const int key = nk * 32 + gt * 4 + half * 16 + (t / 4) * 8 + (t % 4);
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride;
    const uint32_t voff = (uint32_t)key * (uint32_t)seq_stride + (uint32_t)dg * 16u;
    g2s16(g, sm.data + c * 16, voff, range);
}

__device__ __forceinline__ void prefetch_k(st8 &sm, const unsigned char *g, int seq_stride = D) {
    prefetch_k_pass<0>(sm, g, seq_stride);
    prefetch_k_pass<1>(sm, g, seq_stride);
}

__device__ __forceinline__ void prefetch_v(st8 &sm, const unsigned char *g, int seq_stride = Dv) {
    prefetch_v_pass<0>(sm, g, seq_stride);
    prefetch_v_pass<1>(sm, g, seq_stride);
}

__device__ __forceinline__ void load_q8(q8_tile &dst, const unsigned char *g, int seq_stride = D) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load32(g + (size_t)c32 * seq_stride + 0 * 64 + gb * 32);
    dst.d1 = load32(g + (size_t)c32 * seq_stride + 1 * 64 + gb * 32);
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
    const int P = ((ds * 2 + gb) ^ (c32 >> 2)) & 3;
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

// Issue one K pair (2+2 b128). No wait — consumer uses lgkmcnt(2).
template <int Base>
__device__ __forceinline__ void issue_k_a(k_frag &f, uint32_t off) {
    if constexpr (Base == 0) {
        ds_b128<0>(f.a0, off);
        ds_b128<16>(f.a1, off);
    } else {
        ds_b128<8192>(f.a0, off);
        ds_b128<8208>(f.a1, off);
    }
}
template <int Base>
__device__ __forceinline__ void issue_k_b(k_frag &f, uint32_t off) {
    if constexpr (Base == 0) {
        ds_b128<4096>(f.b0, off);
        ds_b128<4112>(f.b1, off);
    } else {
        ds_b128<12288>(f.b0, off);
        ds_b128<12304>(f.b1, off);
    }
}
template <int Base>
__device__ __forceinline__ void issue_k_pair(k_frag &f, uint32_t off) {
    issue_k_a<Base>(f, off);
    issue_k_b<Base>(f, off);
}

// Drain LGKM before the first MFMA. lgkmcnt(2) is only safe when no other DS
// (spills, leftover PV) is in flight; those leak into the count and QK races.
template <bool Acc, int NextBase>
__device__ __forceinline__ void qk_step(s4_tile &s, k_frag &f, const i32x8 &qb, int nk0, uint32_t next_off) {
    fx16 &lo = nk0 ? s.c : s.a;
    fx16 &hi = nk0 ? s.d : s.b;
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        lo = kh_mma(pack8(f.a0, f.a1), qb, lo);
    else
        lo = kh_mma0(pack8(f.a0, f.a1), qb);
    issue_k_a<NextBase>(f, next_off);
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.b0), "+v"(f.b1), "+v"(f.a0), "+v"(f.a1));
    if constexpr (Acc)
        hi = kh_mma(pack8(f.b0, f.b1), qb, hi);
    else
        hi = kh_mma0(pack8(f.b0, f.b1), qb);
    issue_k_b<NextBase>(f, next_off);
}

// Last QK pair: no next-K issue (PV refills k_frag). Keep wait on hi.
template <bool Acc>
__device__ __forceinline__ void qk_tail(s4_tile &s, k_frag &f, const i32x8 &qb, int nk0) {
    fx16 &lo = nk0 ? s.c : s.a;
    fx16 &hi = nk0 ? s.d : s.b;
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        lo = kh_mma(pack8(f.a0, f.a1), qb, lo);
    else
        lo = kh_mma0(pack8(f.a0, f.a1), qb);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        hi = kh_mma(pack8(f.b0, f.b1), qb, hi);
    else
        hi = kh_mma0(pack8(f.b0, f.b1), qb);
}

template <bool Acc>
__device__ __forceinline__ void mma_qk_pair(s4_tile &s, uint32_t off, i32x8 qb, int nk0) {
    k_frag f;
    if (nk0 == 0)
        issue_k_pair<0>(f, off);
    else
        issue_k_pair<8192>(f, off);
    qk_tail<Acc>(s, f, qb, nk0);
}

struct pv_v_head {
    v2i32 a0, a1, a2, a3, b0, b1, b2, b3;
};

__device__ __forceinline__ pv_v_head preissue_pv_v(uint32_t vbase, int nk0) {
    const uint32_t a = v_tr8_off(vbase, 0, nk0, 0);
    pv_v_head h;
    h.a0 = ds_tr8(a);
    h.a1 = ds_tr8(a + 1024u);
    h.a2 = ds_tr8(a + 2048u);
    h.a3 = ds_tr8(a + 3072u);
    h.b0 = ds_tr8(a + 256u);
    h.b1 = ds_tr8(a + 1280u);
    h.b2 = ds_tr8(a + 2304u);
    h.b3 = ds_tr8(a + 3328u);
    return h;
}

__device__ __forceinline__ void mma_pv_lds_preissued(o_tile &o, uint32_t vbase, i32x8 ps, int nk0, pv_v_head h) {
    const uint32_t a = v_tr8_off(vbase, 0, nk0, 0);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.a0), "+v"(h.a1), "+v"(h.a2), "+v"(h.a3), "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3));
    o.a = kh_mma(pack_v2x4(h.a0, h.a1, h.a2, h.a3), ps, o.a);
    v2i32 c0 = ds_tr8(a + 512u), c1 = ds_tr8(a + 1536u);
    v2i32 c2 = ds_tr8(a + 2560u), c3 = ds_tr8(a + 3584u);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma(pack_v2x4(h.b0, h.b1, h.b2, h.b3), ps, o.b);
    v2i32 d0 = ds_tr8(a + 768u), d1 = ds_tr8(a + 1792u);
    v2i32 d2 = ds_tr8(a + 2816u), d3 = ds_tr8(a + 3840u);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma(pack_v2x4(c0, c1, c2, c3), ps, o.c);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.d = kh_mma(pack_v2x4(d0, d1, d2, d3), ps, o.d);
}

// Last 4 PV + next-K ds_read into kf (aiter: dests alias next QK A-ops).
__device__ __forceinline__ void mma_pv_lds_khead(o_tile &o, uint32_t vbase, i32x8 ps, int nk0, k_frag &kf,
                                                uint32_t k_next_off) {
    const uint32_t a = v_tr8_off(vbase, 0, nk0, 0);
    pv_v_head h = preissue_pv_v(vbase, nk0);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.a0), "+v"(h.a1), "+v"(h.a2), "+v"(h.a3), "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3));
    o.a = kh_mma(pack_v2x4(h.a0, h.a1, h.a2, h.a3), ps, o.a);
    v2i32 c0 = ds_tr8(a + 512u), c1 = ds_tr8(a + 1536u);
    v2i32 c2 = ds_tr8(a + 2560u), c3 = ds_tr8(a + 3584u);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma(pack_v2x4(h.b0, h.b1, h.b2, h.b3), ps, o.b);
    v2i32 d0 = ds_tr8(a + 768u), d1 = ds_tr8(a + 1792u);
    v2i32 d2 = ds_tr8(a + 2816u), d3 = ds_tr8(a + 3840u);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma(pack_v2x4(c0, c1, c2, c3), ps, o.c);
    issue_k_a<0>(kf, k_next_off);
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3), "+v"(kf.a0), "+v"(kf.a1));
    o.d = kh_mma(pack_v2x4(d0, d1, d2, d3), ps, o.d);
    issue_k_b<0>(kf, k_next_off);
    pin_kf(kf);
}

__device__ __forceinline__ void mma_pv_lds(o_tile &o, uint32_t vbase, i32x8 ps, int nk0) {
    mma_pv_lds_preissued(o, vbase, ps, nk0, preissue_pv_v(vbase, nk0));
}

__device__ __forceinline__ void mask_s4(s4_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    if (k_base + 0 + 31 > q_start)
        s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    if (k_base + 32 + 31 > q_start)
        s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
    if (k_base + 64 + 31 > q_start)
        s.c = mask_half(s.c, rb, q_pos, k_base + 64, ninf);
    if (k_base + 96 + 31 > q_start)
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
    return xor32_max(mx) * scale;
}

// Causal tiles always contain at least one unmasked key for every query row,
// so mnew is finite on the hot path.  Fuse score*scale-mnew with V_PK_FMA_F32
// instead of issuing separate packed MUL and ADD instructions.
__device__ __forceinline__ void exp2_sc(fx16 &s, float scale, float mnew) {
    const f2 sc = {scale, scale};
    const f2 nm = {-mnew, -mnew};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f2 v = {s[2 * i], s[2 * i + 1]};
        asm volatile("v_pk_fma_f32 %0, %0, %1, %2" : "+v"(v) : "v"(sc), "v"(nm));
        s[2 * i] = exp2f(v[0]);
        s[2 * i + 1] = exp2f(v[1]);
    }
}

__device__ __forceinline__ float col_sum4(const s4_tile &s) {
    f2 acc = {0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const f2 a = {s.a[2 * i], s.a[2 * i + 1]};
        const f2 b = {s.b[2 * i], s.b[2 * i + 1]};
        const f2 c = {s.c[2 * i], s.c[2 * i + 1]};
        const f2 d = {s.d[2 * i], s.d[2 * i + 1]};
        acc += (a + b) + (c + d);
    }
    return xor32_sum(acc[0] + acc[1]);
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

// Aiter: cvt_pk overwrites the score VGPRs. lo/hi are dead after; dest may reuse them.
__device__ __forceinline__ i32x8 pack_p_consume(fx16 &lo, fx16 &hi) {
    i32x8 pa = pack_p(lo, hi);
    asm volatile("" : "+v"(pa) : "v"(lo), "v"(hi));
    asm volatile("" : "=v"(lo), "=v"(hi));
    return pa;
}

__device__ __forceinline__ void store_o_tile(bf16 *O, int q_base, int M, const fx16 &t, int j, const float *inv,
                                            int seq_stride = Dv) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int q = q_base + fmha_qoff(r, gb);
        if (q < M)
            O[(size_t)q * seq_stride + j * 32 + c32] = (bf16)(t[r] * inv[r]);
    }
}

__device__ __forceinline__ void store_o_fp8(bf16 *O, int q_base, int M, const o_tile &o, float l_,
                                           int seq_stride = Dv) {
    const int gb = laneid() >> 5;
    float inv[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        inv[r] = __builtin_amdgcn_rcpf(__shfl(l_, fmha_qoff(r, gb)));
    store_o_tile(O, q_base, M, o.a, 0, inv, seq_stride);
    store_o_tile(O, q_base, M, o.b, 1, inv, seq_stride);
    store_o_tile(O, q_base, M, o.c, 2, inv, seq_stride);
    store_o_tile(O, q_base, M, o.d, 3, inv, seq_stride);
}

// A=V B=P: query stays in the lane, dests are d_v (kCOff). Same store as BF16.
__device__ __forceinline__ void store_o_fp8_qrow(bf16 *O, int q_base, int M, const o_tile &o, float l_,
                                                int seq_stride = Dv) {
    const float inv = (l_ > 0.f) ? __builtin_amdgcn_rcpf(l_) : 0.f;
    o_tile t;
    t.a = mulsc(o.a, inv);
    t.b = mulsc(o.b, inv);
    t.c = mulsc(o.c, inv);
    t.d = mulsc(o.d, inv);
    hk::store_o(O, seq_stride, q_base, M, t);
}

#undef HK8_GLD16

} // namespace hk8
