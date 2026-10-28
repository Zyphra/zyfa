#pragma once
// FP8 descales passed as tensors: the kernels read one float32 value from device memory at launch (a null pointer
// reads as 1), so a call never syncs with the host and can be captured in a CUDA graph.
#include <torch/extension.h>

#include "xcd_map.h"

inline const float *fmha_descale_ptr(const c10::optional<torch::Tensor> &t, const torch::Tensor &q, const char *name) {
    if (!t.has_value())
        return nullptr;
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->device() == q.device(), name,
                " must be a float32 tensor on q's device");
    return t->numel() ? t->data_ptr<float>() : nullptr;
}

// Bit patterns, not std::isfinite: the build uses -ffast-math, which folds isfinite to true.
inline void fmha_check_v_scale(double v_scale) {
    const uint64_t u = fmha_f64_bits(v_scale);
    TORCH_CHECK((u >> 63) == 0 && u >= fmha_f64_bits(1.1754943508222875e-38) &&
                    u <= fmha_f64_bits(3.4028234663852886e38),
                "v_scale must be a positive normal fp32 value");
}
