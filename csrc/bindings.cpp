#include <torch/extension.h>
#include <vector>

namespace fmha_fp8_d64 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_fp8_d128 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_fp8_d192 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_bf16_d64 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_bf16_d128 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_bf16_d192 {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}
namespace fmha_bf16_d192sq {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse, bool bshd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "zyphra_kernels hip family: streamed FMHA research kernels (gfx950)";
  m.def("fmha_fp8_d64_fwd", &fmha_fp8_d64::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal fp8 FMHA, fat 32x32x64 MFMA, D=64; bshd=false BHSD, true BSHD");
  m.def("fmha_fp8_d128_fwd", &fmha_fp8_d128::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal fp8 FMHA, fat 32x32x64 MFMA, D=128; bshd=false BHSD, true BSHD");
  m.def("fmha_fp8_d192_fwd", &fmha_fp8_d192::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal fp8 FMHA, fat 32x32x64 MFMA, D=192 Dv=128; bshd=false BHSD, true BSHD");
  m.def("fmha_bf16_d64_fwd", &fmha_bf16_d64::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal bf16 FMHA, 32x32x16 MFMA, D=64; bshd=false BHSD, true BSHD");
  m.def("fmha_bf16_d128_fwd", &fmha_bf16_d128::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal bf16 FMHA, 32x32x16 MFMA, D=128; bshd=false BHSD, true BSHD");
  m.def("fmha_bf16_d192_fwd", &fmha_bf16_d192::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal bf16 FMHA, 32x32x16 MFMA, D=192 Dv=128; bshd=false BHSD, true BSHD");
  m.def("fmha_bf16_d192sq_fwd", &fmha_bf16_d192sq::forward,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false, py::arg("bshd") = false,
        "causal bf16 FMHA, 32x32x16 MFMA, square D=Dv=192; bshd=false BHSD, true BSHD");
}
