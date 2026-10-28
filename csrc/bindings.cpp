#include <torch/extension.h>

#include <vector>

#define ZYFA_KERNEL(ns)                                                                                         \
    namespace ns {                                                                                              \
    std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale,        \
                                       bool causal, bool return_lse, bool bshd);                               \
    }

ZYFA_KERNEL(fmha_bf16_d128)
ZYFA_KERNEL(fmha_bf16_d192)
ZYFA_KERNEL(fmha_fp8_d128_asm)
ZYFA_KERNEL(fmha_fp8_d192_asm)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_d128", &fmha_bf16_d128::forward, "BF16, d=128");
    m.def("bf16_d192", &fmha_bf16_d192::forward, "BF16, d_qk=192, d_v=128");
    m.def("fp8_d128", &fmha_fp8_d128_asm::forward, "FP8, d=128");
    m.def("fp8_d192", &fmha_fp8_d192_asm::forward, "FP8, d_qk=192, d_v=128");
}
