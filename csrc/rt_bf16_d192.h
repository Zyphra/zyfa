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

__device__ __forceinline__ void pin_q192(q_tile192 &) {}

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

struct alignas(16) st_k {
    bf16 data[KN * KSTRIDE_BF16];
};
struct alignas(16) st_v {
    bf16 data[KN * Dv];
};

template <int Pass>
__device__ __forceinline__ void prefetch_k_pass(st_k &sm, const bf16 *g, int seq_stride = D) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int key = c / CPR, P = c % CPR;
    const int L = (P - key) & (CPR - 1);
    const int Ls = (L < 24) ? L : 0;
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride * (uint32_t)sizeof(bf16);
    const uint32_t voff = ((uint32_t)key * (uint32_t)seq_stride + (uint32_t)Ls * 8u) * (uint32_t)sizeof(bf16);
    g2s16(g, sm.data + c * 8, voff, range);
}

template <int Pass>
__device__ __forceinline__ void prefetch_v_pass(st_v &sm, const bf16 *g, int seq_stride = Dv) {
    const int c = (int)threadIdx.x + Pass * 512;
    const int pp = c * 8, tl = pp / 64, wi = pp % 64;
    const int ky = (tl / (Dv / 16)) * 4 + wi / 16;
    const int dvv = (tl % (Dv / 16)) * 16 + wi % 16;
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride * (uint32_t)sizeof(bf16);
    const uint32_t voff = ((uint32_t)ky * (uint32_t)seq_stride + (uint32_t)dvv) * (uint32_t)sizeof(bf16);
    g2s16(g, sm.data + c * 8, voff, range);
}

__device__ __forceinline__ void prefetch_k(st_k &sm, const bf16 *g, int seq_stride = D) {
    prefetch_k_pass<0>(sm, g, seq_stride);
    prefetch_k_pass<1>(sm, g, seq_stride);
    prefetch_k_pass<2>(sm, g, seq_stride);
    prefetch_k_pass<3>(sm, g, seq_stride);
}

__device__ __forceinline__ void prefetch_v(st_v &sm, const bf16 *g, int seq_stride = Dv) {
    prefetch_v_pass<0>(sm, g, seq_stride);
    prefetch_v_pass<1>(sm, g, seq_stride);
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

__device__ __forceinline__ void load_q(q_tile192 &dst, const bf16 *g, float sc, int seq_stride = D) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    dst.d0 = load16_scale(g + (size_t)c32 * seq_stride + 0 * 16 + gb * 8, sc);
    dst.d1 = load16_scale(g + (size_t)c32 * seq_stride + 1 * 16 + gb * 8, sc);
    dst.d2 = load16_scale(g + (size_t)c32 * seq_stride + 2 * 16 + gb * 8, sc);
    dst.d3 = load16_scale(g + (size_t)c32 * seq_stride + 3 * 16 + gb * 8, sc);
    dst.d4 = load16_scale(g + (size_t)c32 * seq_stride + 4 * 16 + gb * 8, sc);
    dst.d5 = load16_scale(g + (size_t)c32 * seq_stride + 5 * 16 + gb * 8, sc);
    dst.d6 = load16_scale(g + (size_t)c32 * seq_stride + 6 * 16 + gb * 8, sc);
    dst.d7 = load16_scale(g + (size_t)c32 * seq_stride + 7 * 16 + gb * 8, sc);
    dst.d8 = load16_scale(g + (size_t)c32 * seq_stride + 8 * 16 + gb * 8, sc);
    dst.d9 = load16_scale(g + (size_t)c32 * seq_stride + 9 * 16 + gb * 8, sc);
    dst.d10 = load16_scale(g + (size_t)c32 * seq_stride + 10 * 16 + gb * 8, sc);
    dst.d11 = load16_scale(g + (size_t)c32 * seq_stride + 11 * 16 + gb * 8, sc);
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

// Plain-C K fragments. Asm form waited after every pair.
__device__ __forceinline__ void mma_qk_pair_c(s_tile &s, const st_k &sm, int ds, v8b qb) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const int P = ((ds * 2 + gb) + c32) & (CPR - 1);
    const bf16 *p = sm.data + (size_t)c32 * KSTRIDE_BF16 + P * 8;
    const v8b a = *reinterpret_cast<const v8b *>(p);
    const v8b b = *reinterpret_cast<const v8b *>(p + 32 * KSTRIDE_BF16);
    s.a = mfma16(a, qb, s.a);
    s.b = mfma16(b, qb, s.b);
}

// Half-tile exp so the first PV MFMA can issue after 8 v_exp instead of 16.
template <int OFF>
__device__ __forceinline__ void exp2_h(fx16 &s, float mnew) {
    const f2 nm = {-mnew, -mnew};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f2 v = {s[OFF + 2 * i], s[OFF + 2 * i + 1]};
        v = v + nm;
        s[OFF + 2 * i] = zk_exp2(v[0]);
        s[OFF + 2 * i + 1] = zk_exp2(v[1]);
    }
}

__device__ __forceinline__ void mask_s2(s_tile &s, int q_start, int kb) {
    const int q_pos = q_start + (laneid() & 31);
    const int k_base = kb * KN;
    const int rb = ((laneid() >> 5) << 2);
    const float ninf = neg_inf();
    if (k_base + 0 + 31 > q_start)
        s.a = mask_half(s.a, rb, q_pos, k_base + 0, ninf);
    if (k_base + 32 + 31 > q_start)
        s.b = mask_half(s.b, rb, q_pos, k_base + 32, ninf);
}

__device__ __forceinline__ float col_max2(const s_tile &s) {
    float mx = neg_inf();
#pragma unroll
    for (int r = 0; r < 16; ++r)
        mx = max3(s.a[r], s.b[r], mx);
    return xor32_max(mx);
}

__device__ __forceinline__ void exp2_m(fx16 &s, float mnew) {
    const f2 nm = {-mnew, -mnew};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        f2 v = {s[2 * i], s[2 * i + 1]};
        asm volatile("v_pk_add_f32 %0, %1, %2" : "+v"(v) : "v"(nm));
        s[2 * i] = zk_exp2(v[0]);
        s[2 * i + 1] = zk_exp2(v[1]);
    }
}

__device__ __forceinline__ float col_sum2(const s_tile &s) {
    f2 acc = {0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const f2 a = {s.a[2 * i], s.a[2 * i + 1]};
        const f2 b = {s.b[2 * i], s.b[2 * i + 1]};
        acc += a + b;
    }
    return xor32_sum(acc[0] + acc[1]);
}

__device__ __forceinline__ void store_o(bf16 *O, int q_base, int M, const o_tile &o, float l_,
                                        int seq_stride = Dv) {
    const float inv = (l_ > 0.f) ? __builtin_amdgcn_rcpf(l_) : 0.f;
    o_tile t;
    t.a = mulsc(o.a, inv);
    t.b = mulsc(o.b, inv);
    t.c = mulsc(o.c, inv);
    t.d = mulsc(o.d, inv);
    hk::store_o(O, seq_stride, q_base, M, t);
}

#undef HK192_GLD16

} // namespace hk192
