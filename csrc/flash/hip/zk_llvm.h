#pragma once
// Compile-time knobs for AMDGPU scheduler hints and exp lowering.
// Defaults match the shipped kernel. Override with -D on hipcc.
//
//   ZK_IGLP_QK      -1 off, 0 multi-wave GEMM, 1 single-wave GEMM
//   ZK_IGLP_SMX     -1 off, 2 attention + preds, 3 attention TRANS/MFMA only
//   ZK_SETPRIO       0 off, 1 emit s_setprio (constant 0/1 only)
//   ZK_SPLIT_REGION  1 emit sched_barrier(0) between QK and softmax
//   ZK_QK_GROUP      1 emit sched_group_barrier after QK (counts are per-kernel)
//   ZK_EXP_BUILTIN    1 shipped: __builtin_amdgcn_exp2f; 0 = libm exp2f
//   ZK_LAUNCH_BLOCKS  1 (us) or 2 (HK attn __launch_bounds__ min blocks)

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

__device__ __forceinline__ float zk_exp2(float x) {
#if ZK_EXP_BUILTIN
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}
