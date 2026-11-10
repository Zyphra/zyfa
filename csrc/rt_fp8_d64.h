#pragma once
// FP8 (e4m3) tiles for d_qk = d_v = 64, 128-key tiles. One 32x32x64 MFMA covers D, so there is no second
// D-half. K swizzle is 2-way (D/32 = 2). PV writes two 32-wide O fragments. Lean P is the same as d = 128.
#include "rt.h"

namespace hk8_64 {

using namespace hk;
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int v2i32 __attribute__((ext_vector_type(2)));

constexpr int KN = 128, D = 64, Dv = 64, NKB = 4;

__device__ __forceinline__ fx16 kh_mma(i32x8 a, i32x8 b, fx16 c) {
    fx16 o = c;
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0" : "+v"(o) : "v"(a), "v"(b));
    return o;
}

__device__ __forceinline__ fx16 kh_mma0(i32x8 a, i32x8 b) {
    fx16 o;
    asm volatile("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, 0" : "=v"(o) : "v"(a), "v"(b));
    return o;
}

// PV with a per-lane E8M0 scale on P (B): the lane's 32-key block times 2^(sb - 127).
__device__ __forceinline__ fx16 kh_mma_sb(i32x8 a, i32x8 b, fx16 c, int sb) {
    fx16 o = c;
    asm volatile("v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4 op_sel_hi:[0,0,0]"
                 : "+v"(o) : "v"(a), "v"(b), "v"(127), "v"(sb));
    return o;
}

__device__ __forceinline__ void mfma_rd_fence(fx16 &a, fx16 &b) {
    asm volatile("s_nop 15\n\ts_nop 3" : "+v"(a), "+v"(b));
}

struct alignas(16) st8 {
    unsigned char data[KN * D];
};

struct q8_tile {
    i32x8 d0;
};

struct s4_tile {
    fx16 a, b, c, d;
};

struct k_frag {
    i32x4 a0, a1, b0, b1;
};

__device__ __forceinline__ void pin_q8(q8_tile &q) { asm volatile("" : "+v"(q.d0)); }
__device__ __forceinline__ void pin_kf(k_frag &f) {
    asm volatile("" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
}
__device__ __forceinline__ void drain_kf(k_frag &f) {
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
}
__device__ __forceinline__ void mfma_rd_fence(o_tile &o) {
    asm volatile("s_nop 15\n\ts_nop 3" : "+v"(o.a), "+v"(o.b));
}

__device__ __forceinline__ i32x8 load32(const unsigned char *p) {
    i32x8 x;
    __builtin_memcpy(&x, p, 32);
    return x;
}

__device__ __forceinline__ i32x8 pack8(i32x4 lo, i32x4 hi) {
    return (i32x8){lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}

__device__ __forceinline__ uint32_t lds_u32(const unsigned char *p) {
    uint32_t off;
    asm volatile("v_mov_b32 %0, %1"
                 : "=v"(off)
                 : "v"((const __attribute__((address_space(3))) unsigned char *)p));
    return off;
}

// Offsets: 0/16 and 32*D=2048 are keys 0-31; 64*D=4096 and 96*D=6144 are keys 64-95.
template <int Off>
__device__ __forceinline__ void ds_b128(i32x4 &dst, uint32_t addr) {
    if constexpr (Off == 0)
        asm volatile("ds_read_b128 %0, %1" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 16)
        asm volatile("ds_read_b128 %0, %1 offset:16" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 2048)
        asm volatile("ds_read_b128 %0, %1 offset:2048" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 2064)
        asm volatile("ds_read_b128 %0, %1 offset:2064" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 4096)
        asm volatile("ds_read_b128 %0, %1 offset:4096" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 4112)
        asm volatile("ds_read_b128 %0, %1 offset:4112" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 6144)
        asm volatile("ds_read_b128 %0, %1 offset:6144" : "=&v"(dst) : "v"(addr));
    else if constexpr (Off == 6160)
        asm volatile("ds_read_b128 %0, %1 offset:6160" : "=&v"(dst) : "v"(addr));
    else
        static_assert(Off == 0, "ds_b128 offset");
}

__device__ __forceinline__ v2i32 ds_tr8(uint32_t addr) {
    v2i32 r;
    asm volatile("ds_read_b64_tr_b8 %0, %1" : "=&v"(r) : "v"(addr));
    return r;
}

// One pass of 512 threads x 16 B covers a KN x D tile.
template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st8 &sm, const unsigned char *g, int seq_stride = D) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / (D / 16);
    const int off16 = (c % (D / 16)) * 16;
    const int P = off16 / 32, r = off16 % 32;
    const int L = (P ^ (key >> 2)) & 1;   // 2-way: D/32 = 2
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
}
__device__ __forceinline__ void prefetch_v(st8 &sm, const unsigned char *g, int seq_stride = Dv) {
    prefetch_v_pass<0>(sm, g, seq_stride);
}

__device__ __forceinline__ void load_q8(q8_tile &dst, const unsigned char *g, int seq_stride = D) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load32(g + (size_t)c32 * seq_stride + gb * 32);
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

__device__ __forceinline__ uint32_t k_off(uint32_t kbase, int) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int P = (gb ^ (c32 >> 2)) & 1;
    return kbase + (uint32_t)(c32 * D + P * 32);
}

template <int Base>
__device__ __forceinline__ void issue_k_a(k_frag &f, uint32_t off) {
    if constexpr (Base == 0) {
        ds_b128<0>(f.a0, off);
        ds_b128<16>(f.a1, off);
    } else {
        ds_b128<4096>(f.a0, off);
        ds_b128<4112>(f.a1, off);
    }
}
template <int Base>
__device__ __forceinline__ void issue_k_b(k_frag &f, uint32_t off) {
    if constexpr (Base == 0) {
        ds_b128<2048>(f.b0, off);
        ds_b128<2064>(f.b1, off);
    } else {
        ds_b128<6144>(f.b0, off);
        ds_b128<6160>(f.b1, off);
    }
}
template <int Base>
__device__ __forceinline__ void issue_k_pair(k_frag &f, uint32_t off) {
    issue_k_a<Base>(f, off);
    issue_k_b<Base>(f, off);
}

template <bool Acc, int NextBase>
__device__ __forceinline__ void qk_step(s4_tile &s, k_frag &f, const i32x8 &qb, int nk0, uint32_t next_off) {
    fx16 &lo = nk0 ? s.c : s.a;
    fx16 &hi = nk0 ? s.d : s.b;
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
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

__device__ __forceinline__ float max2_local(fx16 &a, fx16 &b, float mx) {
    asm volatile("" : "+v"(a), "+v"(b), "+v"(mx));
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const float ar = a[r], br = b[r];
        asm volatile("v_max3_f32 %0, %1, %2, %0" : "+v"(mx) : "v"(ar), "v"(br), "v"(a), "v"(b));
    }
    return mx;
}

// Last QK pair. Keys 0-63 are already final, so their row max runs under this pair's first MFMA.
template <bool Acc, bool MaxAB>
__device__ __forceinline__ void qk_tail(s4_tile &s, k_frag &f, const i32x8 &qb, int nk0, float *mx_ab) {
    fx16 &lo = nk0 ? s.c : s.a;
    fx16 &hi = nk0 ? s.d : s.b;
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        lo = kh_mma(pack8(f.a0, f.a1), qb, lo);
    else
        lo = kh_mma0(pack8(f.a0, f.a1), qb);
    if constexpr (MaxAB) {
        SCHED_FENCE();
        *mx_ab = max2_local(s.a, s.b, *mx_ab);
    }
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        hi = kh_mma(pack8(f.b0, f.b1), qb, hi);
    else
        hi = kh_mma0(pack8(f.b0, f.b1), qb);
}

struct pv_v_head {
    v2i32 a0, a1, a2, a3, b0, b1, b2, b3;
};

// Dv = 64: step between the four reads of one 32-wide fragment is (Dv/16)*128 = 512; the next fragment is +256.
__device__ __forceinline__ pv_v_head preissue_pv_v(uint32_t vbase, int nk0) {
    const uint32_t a = v_tr8_off(vbase, 0, nk0, 0);
    pv_v_head h;
    h.a0 = ds_tr8(a);
    h.a1 = ds_tr8(a + 512u);
    h.a2 = ds_tr8(a + 1024u);
    h.a3 = ds_tr8(a + 1536u);
    h.b0 = ds_tr8(a + 256u);
    h.b1 = ds_tr8(a + 768u);
    h.b2 = ds_tr8(a + 1280u);
    h.b3 = ds_tr8(a + 1792u);
    return h;
}

__device__ __forceinline__ void mma_pv_lds_preissued(o_tile &o, uint32_t, i32x8 ps, int, pv_v_head h, int sb) {
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.a0), "+v"(h.a1), "+v"(h.a2), "+v"(h.a3), "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3));
    o.a = kh_mma_sb(pack_v2x4(h.a0, h.a1, h.a2, h.a3), ps, o.a, sb);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3));
    o.b = kh_mma_sb(pack_v2x4(h.b0, h.b1, h.b2, h.b3), ps, o.b, sb);
}

__device__ __forceinline__ void mma_pv_lds_khead(o_tile &o, uint32_t vbase, i32x8 ps, int nk0, k_frag &kf,
                                                uint32_t k_next_off, int sb) {
    pv_v_head h = preissue_pv_v(vbase, nk0);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(h.a0), "+v"(h.a1), "+v"(h.a2), "+v"(h.a3), "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3));
    o.a = kh_mma_sb(pack_v2x4(h.a0, h.a1, h.a2, h.a3), ps, o.a, sb);
    issue_k_a<0>(kf, k_next_off);
    asm volatile("s_waitcnt lgkmcnt(2)"
                 : "+v"(h.b0), "+v"(h.b1), "+v"(h.b2), "+v"(h.b3), "+v"(kf.a0), "+v"(kf.a1));
    o.b = kh_mma_sb(pack_v2x4(h.b0, h.b1, h.b2, h.b3), ps, o.b, sb);
    issue_k_b<0>(kf, k_next_off);
    pin_kf(kf);
}

__device__ __forceinline__ void mask_s4(s4_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    if (k_base + 31 > q_start)
        s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    if (k_base + 32 + 31 > q_start)
        s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
    if (k_base + 64 + 31 > q_start)
        s.c = mask_half(s.c, rb, q_pos, k_base + 64, ninf);
    if (k_base + 96 + 31 > q_start)
        s.d = mask_half(s.d, rb, q_pos, k_base + 96, ninf);
}

constexpr float LEAN_SHIFT = 6.f, LEAN_THR = 0.125f;
constexpr float LEAN_BIAS = 0.35f;

__device__ __forceinline__ void p_code(fx16 &s, float c8, float b8) {
#pragma unroll
    for (int r = 0; r < 16; ++r)
        s[r] = __builtin_fmaf(s[r], c8, b8);
}

__device__ __forceinline__ int pk_code4(float a, float b, float c, float d) {
    unsigned w = __builtin_amdgcn_cvt_pk_u8_f32(a, 0, 0u);
    w = __builtin_amdgcn_cvt_pk_u8_f32(b, 1, w);
    w = __builtin_amdgcn_cvt_pk_u8_f32(c, 2, w);
    w = __builtin_amdgcn_cvt_pk_u8_f32(d, 3, w);
    return (int)w;
}

__device__ __forceinline__ i32x8 pack_code_consume(fx16 &lo, fx16 &hi) {
    i32x8 pa;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        pa[j] = pk_code4(lo[4 * j], lo[4 * j + 1], lo[4 * j + 2], lo[4 * j + 3]);
        pa[j + 4] = pk_code4(hi[4 * j], hi[4 * j + 1], hi[4 * j + 2], hi[4 * j + 3]);
    }
    asm volatile("" : "+v"(pa) : "v"(lo), "v"(hi));
    asm volatile("" : "=v"(lo), "=v"(hi));
    return pa;
}

typedef float fx4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ i32x8 rowsum_sel() {
    const int l = laneid();
    const int v = (((l >> 4) ^ (l >> 2)) & 1) ? 0 : 0x38383838;
    return (i32x8){v, v, v, v, v, v, v, v};
}

__device__ __forceinline__ float block_rowsum(const i32x8 &sel, const i32x8 &p, int sb) {
    const fx4 z = {};
    const fx4 r = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(sel, p, z, 0, 0, 0, 0, 0, 0);
    return r[0] * __int_as_float(sb << 23);
}

__device__ __forceinline__ float lane_max2(const fx16 &a, const fx16 &b) {
    float mx = neg_inf();
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = max3(a[r], b[r], mx);
    return mx;
}

__device__ __forceinline__ int block_off(float raw_max, float scale, float mref) {
    const float d = __builtin_fmaf(raw_max, scale, -mref);
    return (int)fmaxf(fminf(floorf(d), 0.f), -127.f);
}

// d_v = 64 is the first two 32-wide fragments. P and l carry 2^LEAN_SHIFT; ov is the V descale.
__device__ __forceinline__ void store_o_fp8_qrow(bf16 *O, int q_base, int M, const o_tile &o, float l_, float ov,
                                                int seq_stride = Dv) {
    float inv = (l_ > 0.f) ? __builtin_amdgcn_rcpf(l_) * ov : 0.f;
    asm volatile("" : "+v"(inv));
    const int q = q_base + (laneid() & 31);
    if (q >= M)
        return;
    store_half(O, seq_stride, q, 0, mulsc(o.a, inv));
    store_half(O, seq_stride, q, 1, mulsc(o.b, inv));
}

} // namespace hk8_64
