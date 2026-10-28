#pragma once
// The subset of HipKittens (CDNA4) used by bf16_d128.hip, plus an N-warp group with the global-to-shared loads.
// Adapted from HipKittens (MIT License, Copyright (c) 2024 HazyResearch; see third_party/hipkittens/LICENSE).
#ifndef KITTENS_CDNA4
#define KITTENS_CDNA4
#endif

#include "cdna4/common/common.cuh"
#include "cdna4/types/types.cuh"
#include "cdna4/ops/warp/register/register.cuh"
#include "cdna4/ops/warp/memory/util/util.cuh"
#include "cdna4/ops/warp/memory/tile/tile.cuh"
#include "cdna4/ops/warp/memory/vec/vec.cuh"

namespace kittens {

template<int N_WARPS>
struct group {
    static constexpr int GROUP_WARPS = N_WARPS;
    static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS;
    __device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
    __device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
    __device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }
#include "cdna4/ops/group/memory/tile/global_to_shared.cuh"
};

} // namespace kittens
