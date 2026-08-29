#pragma once
// BF16 32x32x16 tiles for HK-schedule D=192 / Dv=128. KN=64.
// K LDS is stride 512 B (32×16B/key) so dest=c*16 for global_load_lds
// and P is invariant across +32 keys (offset:16384). Global K stays D=192;
// pairs 24-31 are dummy. V/O stay 128-wide.
#include "rt.h"

namespace hk192 {

using namespace hk;

typedef float f2 __attribute__((ext_vector_type(2)));

constexpr int KN = 64, D = 192, Dv = 128, CPR = 32, KSTRIDE_BF16 = 256;

struct q_tile192 {
    v8b d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
};

__device__ __forceinline__ void pin_q192(q_tile192 &t) {
    asm volatile("" : "+v"(t.d0), "+v"(t.d1), "+v"(t.d2), "+v"(t.d3), "+v"(t.d4), "+v"(t.d5), "+v"(t.d6), "+v"(t.d7),
                          "+v"(t.d8), "+v"(t.d9), "+v"(t.d10), "+v"(t.d11));
}

__device__ __forceinline__ float max3(float a, float b, float c) {
    float r;
    asm volatile("v_max3_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

#define HK192_GLD16(g, l)                                                                                              \
    __builtin_amdgcn_global_load_lds((const __attribute__((address_space(1))) void *)(g),                              \
                                     (__attribute__((address_space(3))) void *)(l), 16, 0, 0)

struct alignas(16) st_k {
    bf16 data[KN * KSTRIDE_BF16];
};
struct alignas(16) st_v {
    bf16 data[KN * Dv];
};

template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st_k &sm, const bf16 *g) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / CPR, P = c % CPR;
    const int L = (P - key) & (CPR - 1);
    const int Ls = (L < 24) ? L : 0;
    HK192_GLD16(g + (size_t)key * D + Ls * 8, sm.data + c * 8);
}

template <int Pass>
__device__ __forceinline__ void prefetch_v_pass(st_v &sm, const bf16 *g) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int pp = c * 8, tl = pp / 64, wi = pp % 64;
    const int ky = (tl / (Dv / 16)) * 4 + wi / 16;
    const int dvv = (tl % (Dv / 16)) * 16 + wi % 16;
    HK192_GLD16(g + (size_t)ky * Dv + dvv, sm.data + c * 8);
}

__device__ __forceinline__ void prefetch_k(st_k &sm, const bf16 *g) {
    prefetch_k_pass<0>(sm, g);
    prefetch_k_pass<1>(sm, g);
    prefetch_k_pass<2>(sm, g);
    prefetch_k_pass<3>(sm, g);
}

__device__ __forceinline__ void prefetch_v(st_v &sm, const bf16 *g) {
    prefetch_v_pass<0>(sm, g);
    prefetch_v_pass<1>(sm, g);
}

__device__ __forceinline__ v8b load16(const bf16 *p) {
    v8b x;
    __builtin_memcpy(&x, p, 16);
    return x;
}

__device__ __forceinline__ v8b load16_scale(const bf16 *p, float sc) {
    v8b x = load16(p);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        x[i] = (bf16)((float)x[i] * sc);
    return x;
}

__device__ __forceinline__ void load_q(q_tile192 &dst, const bf16 *g, float sc) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load16_scale(g + (size_t)c32 * D + 0 * 16 + gb * 8, sc);
    dst.d1 = load16_scale(g + (size_t)c32 * D + 1 * 16 + gb * 8, sc);
    dst.d2 = load16_scale(g + (size_t)c32 * D + 2 * 16 + gb * 8, sc);
    dst.d3 = load16_scale(g + (size_t)c32 * D + 3 * 16 + gb * 8, sc);
    dst.d4 = load16_scale(g + (size_t)c32 * D + 4 * 16 + gb * 8, sc);
    dst.d5 = load16_scale(g + (size_t)c32 * D + 5 * 16 + gb * 8, sc);
    dst.d6 = load16_scale(g + (size_t)c32 * D + 6 * 16 + gb * 8, sc);
    dst.d7 = load16_scale(g + (size_t)c32 * D + 7 * 16 + gb * 8, sc);
    dst.d8 = load16_scale(g + (size_t)c32 * D + 8 * 16 + gb * 8, sc);
    dst.d9 = load16_scale(g + (size_t)c32 * D + 9 * 16 + gb * 8, sc);
    dst.d10 = load16_scale(g + (size_t)c32 * D + 10 * 16 + gb * 8, sc);
    dst.d11 = load16_scale(g + (size_t)c32 * D + 11 * 16 + gb * 8, sc);
}

__device__ __forceinline__ uint32_t lds_u32(const bf16 *p) {
    uint32_t off;
    asm volatile("v_mov_b32 %0, %1"
                 : "=v"(off)
                 : "v"((const __attribute__((address_space(3))) unsigned char *)p));
    return off;
}

__device__ __forceinline__ uint32_t k_off(uint32_t kbase, int ds) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int P = ((ds * 2 + gb) + c32) & (CPR - 1);
    return kbase + (uint32_t)(c32 * (KSTRIDE_BF16 * 2) + P * 16);
}

__device__ __forceinline__ void mma_qk_pair(s_tile &s, uint32_t off, v8b qb) {
    v8b a, b;
    asm volatile("ds_read_b128 %[a], %[addr]\n\t"
                 "ds_read_b128 %[b], %[addr] offset:16384\n\t"
                 "s_waitcnt lgkmcnt(1)"
                 : [a] "=&v"(a), [b] "=&v"(b)
                 : [addr] "v"(off));
    s.a = mfma16(a, qb, s.a);
    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(b));
    s.b = mfma16(b, qb, s.b);
}

__device__ __forceinline__ void mask_s2(s_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
}

__device__ __forceinline__ float col_max2(const s_tile &s) {
    float mx = neg_inf();
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = max3(s.a[r], s.b[r], mx);
    return fmaxf(mx, __shfl_xor(mx, 32));
}

__device__ __forceinline__ void exp2_m(fx16 &s, float mnew) {
    const f2 nm = {-mnew, -mnew};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f2 v = {s[2 * i], s[2 * i + 1]};
        asm volatile("v_pk_add_f32 %0, %1, %2" : "+v"(v) : "v"(nm));
        s[2 * i] = exp2f(v[0]);
        s[2 * i + 1] = exp2f(v[1]);
    }
}

__device__ __forceinline__ float col_sum2(const s_tile &s) {
    float t = _fx_sum(s.b, _fx_sum(s.a, 0.f));
    return t + __shfl_xor(t, 32);
}

__device__ __forceinline__ void store_o(float *O, int q_base, int M, const o_tile &o, float l_) {
    const float inv = (l_ > 0.f) ? __builtin_amdgcn_rcpf(l_) : 0.f;
    o_tile t;
    t.a = mulsc(o.a, inv);
    t.b = mulsc(o.b, inv);
    t.c = mulsc(o.c, inv);
    t.d = mulsc(o.d, inv);
    hk::store_o(O, Dv, q_base, M, t);
}

#undef HK192_GLD16

} // namespace hk192
