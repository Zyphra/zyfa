#include <torch/extension.h>
#include <vector>

namespace fmha32_hkfp8 {
std::vector<torch::Tensor> forward_hkfp8(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse);
}
namespace fmha32_hkfp8_d192 {
std::vector<torch::Tensor> forward_hkfp8_d192(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse);
}
namespace fmha32_hkbf16 {
std::vector<torch::Tensor> forward_hkbf16(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse);
}
namespace fmha32_hkbf16_d192 {
std::vector<torch::Tensor> forward_hkbf16_d192(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale, bool causal, bool return_lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "zyphra_kernels hip family: HK-schedule streamed FMHA research kernels (gfx950)";
  m.def("fmha32_hkfp8_fwd", &fmha32_hkfp8::forward_hkfp8,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false,
        "HK GQA causal attend_ker schedule with fat 32x32x64 fp8 MFMA (D=128 causal; not production)");
  m.def("fmha32_hkfp8_d192_fwd", &fmha32_hkfp8_d192::forward_hkfp8_d192,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false,
        "HK GQA causal attend_ker schedule with fat 32x32x64 fp8 MFMA (D=192 Dv=128 causal; not production)");
  m.def("fmha32_hkbf16_fwd", &fmha32_hkbf16::forward_hkbf16,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false,
        "HK GQA causal attend_ker schedule with 32x32x16 bf16 MFMA (D=128 causal; not production)");
  m.def("fmha32_hkbf16_d192_fwd", &fmha32_hkbf16_d192::forward_hkbf16_d192,
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true,
        py::arg("return_lse") = false,
        "HK GQA causal attend_ker schedule with 32x32x16 bf16 MFMA (D=192 Dv=128 causal; not production)");
}
