#pragma once
// Compile-time knobs for AMDGPU scheduler hints and exp lowering.
// Defaults match the shipped kernel. Override with -D on hipcc.
//
//   ZK_IGLP_QK      -1 off, 0 multi-wave GEMM, 1 single-wave GEMM
//   ZK_IGLP_SMX     -1 off, 2 attention + preds, 3 attention TRANS/MFMA only
//   ZK_SETPRIO       0 off, 1 emit s_setprio (constant 0/1 only)
//   ZK_SPLIT_REGION  1 emit sched_barrier(0) between QK and softmax
//   ZK_QK_GROUP      1 emit sched_group_barrier after QK (counts are per-kernel)
//   ZK_EXP_BUILTIN     1 default: __builtin_amdgcn_exp2f; 0 = libm exp2f
//   ZK_EXP_SCHRAUDOLPH  0 default. 1 = fused IEEE exp2 in zk_exp2_fma2 only
//                     (FP8 D=128 scores). zk_exp2 stays hardware: BF16 and
//                     rescale corr did not gain (MLA BF16 regressed).
//   ZK_LAUNCH_BLOCKS   1 (us) or 2 (HK attn __launch_bounds__ min blocks)

#ifndef ZK_IGLP_QK
#define ZK_IGLP_QK 0
#endif
#ifndef ZK_IGLP_SMX
#define ZK_IGLP_SMX 2
#endif
#ifndef ZK_SETPRIO
#define ZK_SETPRIO 1
#endif
#ifndef ZK_SPLIT_REGION
#define ZK_SPLIT_REGION 0
#endif
#ifndef ZK_QK_GROUP
#define ZK_QK_GROUP 0
#endif
#ifndef ZK_EXP_BUILTIN
#define ZK_EXP_BUILTIN 1
#endif
#ifndef ZK_EXP_SCHRAUDOLPH
#define ZK_EXP_SCHRAUDOLPH 0
#endif
#ifndef ZK_LAUNCH_BLOCKS
#define ZK_LAUNCH_BLOCKS 1
#endif

__device__ __forceinline__ void zk_iglp_qk() {
#if ZK_IGLP_QK >= 0
    __builtin_amdgcn_iglp_opt(ZK_IGLP_QK);
#endif
}

__device__ __forceinline__ void zk_iglp_smx() {
#if ZK_IGLP_SMX >= 0
    __builtin_amdgcn_iglp_opt(ZK_IGLP_SMX);
#endif
}

__device__ __forceinline__ void zk_setprio_hi() {
#if ZK_SETPRIO
    __builtin_amdgcn_s_setprio(1);
#endif
}

__device__ __forceinline__ void zk_setprio_lo() {
#if ZK_SETPRIO
    __builtin_amdgcn_s_setprio(0);
#endif
}

__device__ __forceinline__ void zk_split_region() {
#if ZK_SPLIT_REGION
    __builtin_amdgcn_sched_barrier(0);
#endif
}

typedef float zk_f2 __attribute__((ext_vector_type(2)));

// 2^x. Always hardware / libm. Do not put Schraudolph here: BF16 MLA
// and unfused score exp both lost to extra VALU.
__device__ __forceinline__ float zk_exp2(float x) {
#if ZK_EXP_BUILTIN == 0
    return exp2f(x);
#else
    return __builtin_amdgcn_exp2f(x);
#endif
}

// 2^(a*x + b) in one packed FMA in IEEE exponent space.
// Hot path: a = scale, b = -mnew. Off: scale FMA then zk_exp2.
__device__ __forceinline__ zk_f2 zk_exp2_fma2(zk_f2 x, float a, float b) {
#if ZK_EXP_SCHRAUDOLPH
    constexpr float A = 8388608.f;
    constexpr float B = 1065353216.f;
    const zk_f2 mul = {a * A, a * A};
    const float bias = B + b * A;
    const zk_f2 add = {bias, bias};
    const zk_f2 lo = {B - 80.f * A, B - 80.f * A};
    asm volatile("v_pk_fma_f32 %0, %0, %1, %2" : "+v"(x) : "v"(mul), "v"(add));
    x[0] = fmaxf(x[0], lo[0]);
    x[1] = fmaxf(x[1], lo[1]);
    zk_f2 y;
    y[0] = __int_as_float((int)x[0]);
    y[1] = __int_as_float((int)x[1]);
    return y;
#else
    const zk_f2 mul = {a, a};
    const zk_f2 add = {b, b};
    asm volatile("v_pk_fma_f32 %0, %0, %1, %2" : "+v"(x) : "v"(mul), "v"(add));
    zk_f2 y;
    y[0] = zk_exp2(x[0]);
    y[1] = zk_exp2(x[1]);
    return y;
#endif
}
