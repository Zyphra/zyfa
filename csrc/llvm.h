#pragma once
// AMDGPU scheduler hints and the hardware exp2 used by the rt*.h kernels.

// iglp_opt(0): interleave the QK MFMAs with their operand loads.
__device__ __forceinline__ void iglp_qk() { __builtin_amdgcn_iglp_opt(0); }

// iglp_opt(2): interleave MFMAs with the softmax exp / VALU work.
__device__ __forceinline__ void iglp_softmax() { __builtin_amdgcn_iglp_opt(2); }

// 2^x on v_exp_f32.
__device__ __forceinline__ float exp2_hw(float x) { return __builtin_amdgcn_exp2f(x); }
