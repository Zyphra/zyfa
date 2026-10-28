#pragma once
// BF16 tiles and helpers for MLA, d_qk = 192 / d_v = 128, with 64-key tiles (32x32x16 MFMA): operand halves are
// read whole into VGPRs in a memory stage and consumed by back-to-back MFMAs in a compute stage.
#include "rt.h"

namespace hk192 {

using namespace hk;

constexpr int KN = 64, D = 192, Dv = 128;

struct q_tile192 {
    v8b d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
};

struct alignas(16) st_v {
    bf16 data[KN * Dv];
};

__device__ __forceinline__ v8b load16(const bf16 *p) {
    v8b x;
    __builtin_memcpy(&x, p, 16);
    return x;
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

// Unscaled Q (the scale is applied to the fp32 scores).
__device__ __forceinline__ void load_q_raw(q_tile192 &dst, const bf16 *g, int seq_stride = D) {
    const int c32 = laneid() & 31, gb = laneid() >> 5;
    const bf16 *r = g + (size_t)c32 * seq_stride + gb * 8;
    dst.d0 = load16(r + 0 * 16);
    dst.d1 = load16(r + 1 * 16);
    dst.d2 = load16(r + 2 * 16);
    dst.d3 = load16(r + 3 * 16);
    dst.d4 = load16(r + 4 * 16);
    dst.d5 = load16(r + 5 * 16);
    dst.d6 = load16(r + 6 * 16);
    dst.d7 = load16(r + 7 * 16);
    dst.d8 = load16(r + 8 * 16);
    dst.d9 = load16(r + 9 * 16);
    dst.d10 = load16(r + 10 * 16);
    dst.d11 = load16(r + 11 * 16);
}

// K half h (keys 32h..32h+31) of a 64-key tile: one A fragment per 16-wide d step.
struct k_su {
    v8b d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
};

// Padded K layout: a tile is 24 blocks of 1040 B; block (r, w) holds d chunk r (64 d) of keys w, w+8, ..., w+56
// (128 B each) plus 16 B of padding, so 16 lanes of a fragment read touch all 64 banks once.
constexpr int KP_BLK = 1040, KP_TILE = 24 * KP_BLK;
struct alignas(16) st_kp {
    unsigned char data[KP_TILE];
};

using lds_v8b = const __attribute__((address_space(3))) v8b *;

__device__ __forceinline__ uint32_t lds_off(const void *p) { return (uint32_t)(uintptr_t)p; }

// Lane part of a K fragment address (key 32h + n, d 16 ds + 8 gb is this + 512 h + 8320 (ds / 4) + 32 (ds % 4)).
__device__ __forceinline__ uint32_t kp_read_lane(int lane) {
    return (uint32_t)((lane & 7) * KP_BLK + ((lane >> 3) & 3) * 128 + (lane >> 5) * 16);
}

// Lane part of the global byte offset of this thread's 16 B (d chunk 0); chunk r adds 128 B (soffset).
__device__ __forceinline__ uint32_t kp_global_lane(int lane, int wave, int seq_stride) {
    return (uint32_t)((((lane >> 3) * 8 + wave) * seq_stride + (lane & 7) * 8) * (int)sizeof(bf16));
}

// d chunks [B, E) of a K tile; wave w fills blocks (r, w).
template <int B = 0, int E = 3>
__device__ __forceinline__ void prefetch_kp(st_kp &sm, const bf16 *g, uint32_t voff, int wave, int seq_stride) {
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride * (uint32_t)sizeof(bf16);
    const g2s_i32x4 rsrc = make_srsrc(g, range);
#pragma unroll
    for (int r = B; r < E; ++r)
        // The chunk offset goes in soffset: the instruction offset would also move the LDS destination.
        llvm_amdgcn_raw_buffer_load_lds(rsrc, (g2s_as3)(uintptr_t)(sm.data + (r * 8 + wave) * KP_BLK), 16, (int)voff,
                                        r * 128, 0, 0);
}

template <int OFF>
__device__ __forceinline__ v8b kp_frag(uint32_t a) {
    return *(lds_v8b)(uintptr_t)(a + (uint32_t)OFF);
}

// a = buffer base + kp_read_lane(lane). H = key half.
template <int H>
__device__ __forceinline__ void read_kp_su(k_su &k, uint32_t a) {
    k.d0 = kp_frag<H * 512 + 0 * 8320 + 0 * 32>(a);
    k.d1 = kp_frag<H * 512 + 0 * 8320 + 1 * 32>(a);
    k.d2 = kp_frag<H * 512 + 0 * 8320 + 2 * 32>(a);
    k.d3 = kp_frag<H * 512 + 0 * 8320 + 3 * 32>(a);
    k.d4 = kp_frag<H * 512 + 1 * 8320 + 0 * 32>(a);
    k.d5 = kp_frag<H * 512 + 1 * 8320 + 1 * 32>(a);
    k.d6 = kp_frag<H * 512 + 1 * 8320 + 2 * 32>(a);
    k.d7 = kp_frag<H * 512 + 1 * 8320 + 3 * 32>(a);
    k.d8 = kp_frag<H * 512 + 2 * 8320 + 0 * 32>(a);
    k.d9 = kp_frag<H * 512 + 2 * 8320 + 1 * 32>(a);
    k.d10 = kp_frag<H * 512 + 2 * 8320 + 2 * 32>(a);
    k.d11 = kp_frag<H * 512 + 2 * 8320 + 3 * 32>(a);
}

// S^T half (32 keys x 32 queries) = K half . Q^T, 12 dependent MFMAs.
__device__ __forceinline__ fx16 qk_su(const k_su &k, const q_tile192 &q) {
    fx16 s = mfma16(k.d0, q.d0, fx16_zero());
    s = mfma16(k.d1, q.d1, s);
    s = mfma16(k.d2, q.d2, s);
    s = mfma16(k.d3, q.d3, s);
    s = mfma16(k.d4, q.d4, s);
    s = mfma16(k.d5, q.d5, s);
    s = mfma16(k.d6, q.d6, s);
    s = mfma16(k.d7, q.d7, s);
    s = mfma16(k.d8, q.d8, s);
    s = mfma16(k.d9, q.d9, s);
    s = mfma16(k.d10, q.d10, s);
    s = mfma16(k.d11, q.d11, s);
    return s;
}

// V half h (dv 64h..64h+63) of a 64-key tile: fragments for 16-key strip sl and 32-wide dv block j.
struct v_su {
    v8b s0j0, s0j1, s1j0, s1j1, s2j0, s2j1, s3j0, s3j1;
};

// Transposed V read as inline asm: LLVM would wait vmcnt(0) for the in-flight V prefetch before the intrinsic.
// The caller waits lgkmcnt(0) and fences the scheduler (stage_end) before any use.
template <int OFF>
__device__ __forceinline__ v8b v_frag_at(uint32_t a) {
    v4b r0, r1;
    asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2" : "=v"(r0) : "v"(a), "i"(OFF) : "memory");
    asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2" : "=v"(r1) : "v"(a), "i"(OFF + 2048) : "memory");
    v8b f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        f[t] = r0[t];
        f[4 + t] = r1[t];
    }
    return f;
}

// Lane part of the transposed V read address.
__device__ __forceinline__ uint32_t v_read_lane(int lane_) {
    const uint32_t lane = (uint32_t)lane_;
    const uint32_t row_off = ((lane & 15u) >> 2) + ((lane >> 5) << 2);
    const uint32_t col_off = ((lane & 3u) << 2) + (((lane & 31u) >> 4) << 4);
    return 2u * (((row_off >> 2) * 8u + (col_off >> 4)) * 64u + (row_off & 3u) * 16u + (col_off & 15u));
}

// Lane part of the global V offset for prefetch pass 0 (pass 1 is +32 keys, passed as soffset).
__device__ __forceinline__ uint32_t v_global_lane(int tid, int seq_stride) {
    const int pp = tid * 8, tl = pp / 64, wi = pp % 64;
    const int ky = (tl / (Dv / 16)) * 4 + wi / 16;
    const int dvv = (tl % (Dv / 16)) * 16 + wi % 16;
    return (uint32_t)((ky * seq_stride + dvv) * (int)sizeof(bf16));
}

__device__ __forceinline__ void prefetch_vp(st_v &sm, const bf16 *g, uint32_t voff, int wave, int seq_stride) {
    const uint32_t range = (uint32_t)KN * (uint32_t)seq_stride * (uint32_t)sizeof(bf16);
    const g2s_i32x4 rsrc = make_srsrc(g, range);
#pragma unroll
    for (int p = 0; p < 2; ++p)
        llvm_amdgcn_raw_buffer_load_lds(rsrc, (g2s_as3)(uintptr_t)(sm.data + (p * 512 + wave * 64) * 8), 16, (int)voff,
                                        p * 32 * seq_stride * (int)sizeof(bf16), 0, 0);
}

// a = buffer base + v_read_lane(lane). H = dv half.
template <int H>
__device__ __forceinline__ void read_vp_su(v_su &v, uint32_t a) {
    v.s0j0 = v_frag_at<H * 512 + 0>(a);
    v.s0j1 = v_frag_at<H * 512 + 256>(a);
    v.s1j0 = v_frag_at<H * 512 + 4096>(a);
    v.s1j1 = v_frag_at<H * 512 + 4096 + 256>(a);
    v.s2j0 = v_frag_at<H * 512 + 8192>(a);
    v.s2j1 = v_frag_at<H * 512 + 8192 + 256>(a);
    v.s3j0 = v_frag_at<H * 512 + 12288>(a);
    v.s3j1 = v_frag_at<H * 512 + 12288 + 256>(a);
}

// O^T dv blocks (2h, 2h+1) += V half h . P^T.
__device__ __forceinline__ void pv_su(fx16 &o0, fx16 &o1, const v_su &v, const p_tile &p) {
    o0 = mfma16(v.s0j0, p.a, o0);
    o1 = mfma16(v.s0j1, p.a, o1);
    o0 = mfma16(v.s1j0, p.b, o0);
    o1 = mfma16(v.s1j1, p.b, o1);
    o0 = mfma16(v.s2j0, p.c, o0);
    o1 = mfma16(v.s2j1, p.c, o1);
    o0 = mfma16(v.s3j0, p.d, o0);
    o1 = mfma16(v.s3j1, p.d, o1);
}

// Element i of the 32 scores a lane holds (s.a then s.b).
#define HK192_S(s, i) ((i) < 16 ? (s).a[(i)] : (s).b[(i) - 16])

// x = s*c - m on elements [B, E), one scalar FMA each (pinned: a paired v_pk_fma_f32 cannot issue next to an MFMA).
template <int B, int E>
__device__ __forceinline__ void fma_sub(s_tile &s, float c, float m) {
#pragma unroll
    for (int i = B; i < E; ++i) {
        float x = __builtin_fmaf(HK192_S(s, i), c, -m);
        asm volatile("" : "+v"(x));
        if (i < 16)
            s.a[i] = x;
        else
            s.b[i - 16] = x;
    }
}

template <int B, int E>
__device__ __forceinline__ void exp_range(s_tile &s) {
#pragma unroll
    for (int i = B; i < E; ++i) {
        if (i < 16)
            s.a[i] = exp2_hw(s.a[i]);
        else
            s.b[i - 16] = exp2_hw(s.b[i - 16]);
    }
}

// Row sum of the 32 fp32 P values, scalar adds.
__device__ __forceinline__ float row_sum(const s_tile &s) {
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        a0 += s.a[i] + s.b[i];
        a1 += s.a[i + 1] + s.b[i + 1];
        asm volatile("" : "+v"(a0), "+v"(a1));
    }
    return xor32_sum(a0 + a1);
}

#undef HK192_S

} // namespace hk192
