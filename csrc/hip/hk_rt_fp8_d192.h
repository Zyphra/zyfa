#pragma once
// FP8 fat 32x32x64 tiles for HK-schedule D=192 / Dv=128. KN=64 (d128 HK is KN=128;
// 4 score tiles + 3 Q chunks spill). CPR=6; nk1 P is (P0+32)%6.
// V/O stay 128-wide. Do not fold into hk_rt_fp8_d128.h.
#include "hk_rt_fp8_d128.h"

namespace hk8_192 {

using namespace hk;
using hk8::i32x4;
using hk8::i32x8;
using hk8::v2i32;
using hk8::kh_mma;
using hk8::kh_mma0;
using hk8::pack8;
using hk8::pack_p;
using hk8::pack_v2x4;
using hk8::ds_tr8;
using hk8::load32;
using hk8::lds_u32;
using hk8::fmha_qoff;
using hk8::max3;

constexpr int KN = 64, D = 192, Dv = 128, CPR = D / 32, NKB = 2, NDVB = 4;

struct q8_tile192 {
    i32x8 d0, d1, d2;
};

struct s2_tile {
    fx16 a, b;
};

__device__ __forceinline__ void pin_q192(q8_tile192 &t) { asm volatile("" : "+v"(t.d0), "+v"(t.d1), "+v"(t.d2)); }
__device__ __forceinline__ void pin_s2(s2_tile &t) { asm volatile("" : "+v"(t.a), "+v"(t.b)); }

#define HK8_192_GLD16(g, l)                                                                                            \
    __builtin_amdgcn_global_load_lds((const __attribute__((address_space(1))) void *)(g),                              \
                                     (__attribute__((address_space(3))) void *)(l), 16, 0, 0)

struct alignas(16) st_k {
    unsigned char data[KN * D];
};
struct alignas(16) st_v {
    unsigned char data[KN * Dv];
};

template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st_k &sm, const unsigned char *g) {
    const int c = (int)threadIdx.x + Pass * 512;
    if (c >= KN * (D / 16))
        return;
    const int key = c / (D / 16);
    const int off16 = (c % (D / 16)) * 16;
    const int P = off16 / 32, r = off16 % 32;
    const int L = ((P - key) % CPR + CPR) % CPR;
    HK8_192_GLD16(g + (size_t)key * D + L * 32 + r, sm.data + key * D + P * 32 + r);
}

__device__ __forceinline__ void prefetch_v(st_v &sm, const unsigned char *g) {
    constexpr int NK_BITS = 1;
    const int c = (int)threadIdx.x;
    const int pp = c * 16, tile = pp / 128, t = (pp % 128) / 16, dg = tile % (Dv / 16), rest = tile / (Dv / 16);
    const int half = rest & 1, nk = (rest >> 1) & (NKB - 1), gt = (rest >> (1 + NK_BITS)) & 1;
    const int key = nk * 32 + gt * 4 + half * 16 + (t / 4) * 8 + (t % 4);
    HK8_192_GLD16(g + (size_t)key * Dv + dg * 16, sm.data + c * 16);
}

__device__ __forceinline__ void prefetch_k(st_k &sm, const unsigned char *g) {
    prefetch_k_pass<0>(sm, g);
    prefetch_k_pass<1>(sm, g);
}

__device__ __forceinline__ void load_q8(q8_tile192 &dst, const unsigned char *g) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load32(g + (size_t)c32 * D + 0 * 64 + gb * 32);
    dst.d1 = load32(g + (size_t)c32 * D + 1 * 64 + gb * 32);
    dst.d2 = load32(g + (size_t)c32 * D + 2 * 64 + gb * 32);
}

__device__ __forceinline__ uint32_t k_off(uint32_t kbase, int ds, int nk) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int ky = nk * 32 + c32;
    const int P = ((ds * 2 + gb) + ky) % CPR;
    return kbase + (uint32_t)(ky * D + P * 32);
}

template <bool Acc>
__device__ __forceinline__ void mma_qk_pair(s2_tile &s, uint32_t off0, uint32_t off1, i32x8 qb) {
    i32x4 a0, a1, b0, b1;
    asm volatile("ds_read_b128 %[a0], %[a]\n\t"
                 "ds_read_b128 %[a1], %[a] offset:16\n\t"
                 "ds_read_b128 %[b0], %[b]\n\t"
                 "ds_read_b128 %[b1], %[b] offset:16\n\t"
                 "s_waitcnt lgkmcnt(2)"
                 : [a0] "=&v"(a0), [a1] "=&v"(a1), [b0] "=&v"(b0), [b1] "=&v"(b1)
                 : [a] "v"(off0), [b] "v"(off1));
    if constexpr (Acc)
        s.a = kh_mma(pack8(a0, a1), qb, s.a);
    else
        s.a = kh_mma0(pack8(a0, a1), qb);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(b0), "+v"(b1));
    if constexpr (Acc)
        s.b = kh_mma(pack8(b0, b1), qb, s.b);
    else
        s.b = kh_mma0(pack8(b0, b1), qb);
}

__device__ __forceinline__ uint32_t v_tr8_off(uint32_t vbase, int j, int nk, int half) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int dg = j * 2 + (c32 >> 4);
    const int tile = ((gb * NKB + nk) * 2 + half) * (Dv / 16) + dg;
    return vbase + (uint32_t)(tile * 128 + (c32 & 15) * 8);
}

__device__ __forceinline__ void mma_pv_lds(o_tile &o, uint32_t vbase, i32x8 ps) {
    v2i32 a0 = ds_tr8(v_tr8_off(vbase, 0, 0, 0)), a1 = ds_tr8(v_tr8_off(vbase, 0, 0, 1));
    v2i32 a2 = ds_tr8(v_tr8_off(vbase, 0, 1, 0)), a3 = ds_tr8(v_tr8_off(vbase, 0, 1, 1));
    v2i32 b0 = ds_tr8(v_tr8_off(vbase, 1, 0, 0)), b1 = ds_tr8(v_tr8_off(vbase, 1, 0, 1));
    v2i32 b2 = ds_tr8(v_tr8_off(vbase, 1, 1, 0)), b3 = ds_tr8(v_tr8_off(vbase, 1, 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3));
    o.a = kh_mma(ps, pack_v2x4(a0, a1, a2, a3), o.a);
    v2i32 c0 = ds_tr8(v_tr8_off(vbase, 2, 0, 0)), c1 = ds_tr8(v_tr8_off(vbase, 2, 0, 1));
    v2i32 c2 = ds_tr8(v_tr8_off(vbase, 2, 1, 0)), c3 = ds_tr8(v_tr8_off(vbase, 2, 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3), "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3));
    o.b = kh_mma(ps, pack_v2x4(b0, b1, b2, b3), o.b);
    v2i32 d0 = ds_tr8(v_tr8_off(vbase, 3, 0, 0)), d1 = ds_tr8(v_tr8_off(vbase, 3, 0, 1));
    v2i32 d2 = ds_tr8(v_tr8_off(vbase, 3, 1, 0)), d3 = ds_tr8(v_tr8_off(vbase, 3, 1, 1));
    asm volatile("s_waitcnt lgkmcnt(4)"
                 : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3), "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.c = kh_mma(ps, pack_v2x4(c0, c1, c2, c3), o.c);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(d0), "+v"(d1), "+v"(d2), "+v"(d3));
    o.d = kh_mma(ps, pack_v2x4(d0, d1, d2, d3), o.d);
}

__device__ __forceinline__ void mask_s2(s2_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
}

__device__ __forceinline__ float col_max2_sc(const s2_tile &s, float scale) {
    float mx = neg_inf();
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = max3(s.a[r] * scale, s.b[r] * scale, mx);
    return fmaxf(mx, __shfl_xor(mx, 32));
}

__device__ __forceinline__ float col_sum2(const s2_tile &s) {
    float t = _fx_sum(s.b, _fx_sum(s.a, 0.f));
    return t + __shfl_xor(t, 32);
}

__device__ __forceinline__ void rescale_o(o_tile &o, float corr) {
    if (!__any(corr != 1.f))
        return;
    const int gb = laneid() >> 5;
    float ct[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        ct[r] = __shfl(corr, fmha_qoff(r, gb));
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        o.a[r] *= ct[r];
        o.b[r] *= ct[r];
        o.c[r] *= ct[r];
        o.d[r] *= ct[r];
    }
}

__device__ __forceinline__ void store_o_fp8(float *O, int q_base, int M, const o_tile &o, float l_) {
    hk8::store_o_fp8(O, q_base, M, o, l_);
}

#undef HK8_192_GLD16

} // namespace hk8_192
