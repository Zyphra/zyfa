#pragma once
// FP8 (e4m3) tiles and helpers for MLA, d_qk = 192 / d_v = 128, with 64-key tiles (32x32x64 MFMA).
// K rows are padded to 256 B in LDS (chunks 12-15 unused), so a K fragment address is the same for both 32-key halves.
#include "rt_fp8_d128.h"

namespace hk8_192 {

using namespace hk;
using hk8::block_off;
using hk8::block_rowsum;
using hk8::drain_kf;
using hk8::ds_b128;
using hk8::ds_tr8o;
using hk8::i32x4;
using hk8::i32x8;
using hk8::k_frag;
using hk8::kh_mma;
using hk8::kh_mma0;
using hk8::kh_mma_sb;
using hk8::lane_max2;
using hk8::lds_u32;
using hk8::LEAN_SHIFT;
using hk8::LEAN_THR;
using hk8::load32;
using hk8::mfma_rd_fence;
using hk8::p_exp_shifted;
using hk8::pack8;
using hk8::pack_p;
using hk8::pack_v2x4;
using hk8::pin_kf;
using hk8::rowsum_sel;
using hk8::store_o_fp8_qrow;
using hk8::v2i32;

constexpr int KN = 64, D = 192, Dv = 128, CPR = 8, KSTRIDE = 256, NKB = 2;

struct q8_tile192 {
    i32x8 d0, d1, d2;
};

struct s2_tile {
    fx16 a, b;
};

struct alignas(16) st_k {
    unsigned char data[KN * KSTRIDE];
};
struct alignas(16) st_v {
    unsigned char data[KN * Dv];
};

// Half of a K tile; LDS 32-byte chunk P of a key holds global chunk (P - key) mod 8 (bank swizzle).
template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st_k &sm, const unsigned char *g, int seq_stride = D) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / 16, chunk = c & 15;
    const int P = chunk >> 1, r = (chunk & 1) * 16;
    const int L = (P - key) & (CPR - 1);
    const int Ls = (L < 6) ? L : 0;
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride;
    const uint32_t voff = (uint32_t)key * (uint32_t)seq_stride + (uint32_t)Ls * 32u + (uint32_t)r;
    g2s16(g, sm.data + c * 16, voff, range);
}

// V tile, laid out for ds_read_b64_tr_b8.
__device__ __forceinline__ void prefetch_v(st_v &sm, const unsigned char *g, int seq_stride = Dv) {
    constexpr int NK_BITS = 1;
    const int c = (int)threadIdx.x;
    const int pp = c * 16, tile = pp / 128, t = (pp % 128) / 16, dg = tile % (Dv / 16), rest = tile / (Dv / 16);
    const int half = rest & 1, nk = (rest >> 1) & (NKB - 1), gt = (rest >> (1 + NK_BITS)) & 1;
    const int key = nk * 32 + gt * 4 + half * 16 + (t / 4) * 8 + (t % 4);
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride;
    const uint32_t voff = (uint32_t)key * (uint32_t)seq_stride + (uint32_t)dg * 16u;
    g2s16(g, sm.data + c * 16, voff, range);
}

__device__ __forceinline__ void prefetch_k(st_k &sm, const unsigned char *g, int seq_stride = D) {
    prefetch_k_pass<0>(sm, g, seq_stride);
    prefetch_k_pass<1>(sm, g, seq_stride);
}

__device__ __forceinline__ void load_q8(q8_tile192 &dst, const unsigned char *g, int seq_stride = D) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load32(g + (size_t)c32 * seq_stride + 0 * 64 + gb * 32);
    dst.d1 = load32(g + (size_t)c32 * seq_stride + 1 * 64 + gb * 32);
    dst.d2 = load32(g + (size_t)c32 * seq_stride + 2 * 64 + gb * 32);
}

__device__ __forceinline__ uint32_t k_off(uint32_t kbase, int ds) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int P = ((ds * 2 + gb) + c32) & (CPR - 1);
    return kbase + (uint32_t)(c32 * KSTRIDE + P * 32);
}

__device__ __forceinline__ void issue_k_a(k_frag &f, uint32_t off) {
    ds_b128<0>(f.a0, off);
    ds_b128<16>(f.a1, off);
}
__device__ __forceinline__ void issue_k_b(k_frag &f, uint32_t off) {
    ds_b128<8192>(f.b0, off);
    ds_b128<8208>(f.b1, off);
}
__device__ __forceinline__ void issue_k_pair(k_frag &f, uint32_t off) {
    issue_k_a(f, off);
    issue_k_b(f, off);
}

// One 64-wide d step: consume f at lgkmcnt(2); the next step's reads overlap the second MFMA.
template <bool Acc>
__device__ __forceinline__ void qk_step(s2_tile &s, k_frag &f, i32x8 qb, uint32_t next_off) {
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
    if constexpr (Acc)
        s.a = kh_mma(pack8(f.a0, f.a1), qb, s.a);
    else
        s.a = kh_mma0(pack8(f.a0, f.a1), qb);
    issue_k_a(f, next_off);
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.b0), "+v"(f.b1), "+v"(f.a0), "+v"(f.a1));
    if constexpr (Acc)
        s.b = kh_mma(pack8(f.b0, f.b1), qb, s.b);
    else
        s.b = kh_mma0(pack8(f.b0, f.b1), qb);
    issue_k_b(f, next_off);
}

// Last d step (no next reads).
__device__ __forceinline__ void qk_tail(s2_tile &s, k_frag &f, i32x8 qb) {
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(f.a0), "+v"(f.a1), "+v"(f.b0), "+v"(f.b1));
    s.a = kh_mma(pack8(f.a0, f.a1), qb, s.a);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(f.b0), "+v"(f.b1));
    s.b = kh_mma(pack8(f.b0, f.b1), qb, s.b);
    mfma_rd_fence(s.a, s.b);
}

__device__ __forceinline__ uint32_t v_tr8_off(uint32_t vbase, int j, int nk, int half) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int dg = j * 2 + (c32 >> 4);
    const int tile = ((gb * NKB + nk) * 2 + half) * (Dv / 16) + dg;
    return vbase + (uint32_t)(tile * 128 + (c32 & 15) * 8);
}

// PV (4 MFMAs, P block scale sb); the next tile's first K pair is read into kf under the last two MFMAs.
__device__ __forceinline__ void mma_pv_lds_khead(o_tile &o, uint32_t vbase, i32x8 ps, k_frag &kf, uint32_t k_next_off,
                                                int sb) {
    const uint32_t va = v_tr8_off(vbase, 0, 0, 0);
    v2i32 a0 = ds_tr8o<0>(va), a1 = ds_tr8o<1024>(va);
    v2i32 a2 = ds_tr8o<2048>(va), a3 = ds_tr8o<3072>(va);
    v2i32 b0 = ds_tr8o<256>(va), b1 = ds_tr8o<1280>(va);
    v2i32 b2 = ds_tr8o<2304>(va), b3 = ds_tr8o<3328>(va);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3));
    o.a = kh_mma_sb(pack_v2x4(a0, a1, a2, a3), ps, o.a, sb);
    v2i32 c0 = ds_tr8o<512>(va), c1 = ds_tr8o<1536>(va);
    v2i32 c2 = ds_tr8o<2560>(va), c3 = ds_tr8o<3584>(va);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma_sb(pack_v2x4(b0, b1, b2, b3), ps, o.b, sb);
    v2i32 d0 = ds_tr8o<768>(va), d1 = ds_tr8o<1792>(va);
    v2i32 d2 = ds_tr8o<2816>(va), d3 = ds_tr8o<3840>(va);
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma_sb(pack_v2x4(c0, c1, c2, c3), ps, o.c, sb);
    issue_k_a(kf, k_next_off);
    asm volatile("s_waitcnt lgkmcnt(2)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3), "+v"(kf.a0), "+v"(kf.a1));
    o.d = kh_mma_sb(pack_v2x4(d0, d1, d2, d3), ps, o.d, sb);
    issue_k_b(kf, k_next_off);
    pin_kf(kf);
}

__device__ __forceinline__ void mask_s2(s2_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    if (k_base + 0 + 31 > q_start)
        s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    if (k_base + 32 + 31 > q_start)
        s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
}

} // namespace hk8_192
