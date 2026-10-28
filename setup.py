import os
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "build" / "include"

if not torch.version.hip:
    sys.exit("zyfa needs a ROCm build of PyTorch")

# The kernels are written for gfx950 (MI355X) only.
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

hip_flags = ["-O3", "-std=c++20", "-ffast-math", "-fgpu-flush-denormals-to-zero", "-Wno-unused-function",
             "-Wno-nan-infinity-disabled", "-DKITTENS_CDNA4", "-U__HIP_NO_HALF_CONVERSIONS__",
             "-U__HIP_NO_HALF_OPERATORS__"]


class BuildExt(BuildExtension):
    def run(self):
        # Generate the two assembly kernels and embed their code objects as C headers.
        subprocess.run([sys.executable, str(ROOT / "csrc" / "asm" / "build.py"), "--out-dir", str(GEN / "generated"),
                        "--rocm", os.environ.get("ROCM_PATH", "/opt/rocm")], check=True)
        super().run()


setup(
    name="zyfa",
    version="0.1.0",
    description="Flash attention for AMD MI355X from the Zyphra Optimization Engine (ZOE)",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zyphra/zyfa",
    license="Apache-2.0",
    license_files=["LICENSE", "third_party/hipkittens/LICENSE"],
    packages=find_packages(include=["zyfa"]),
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            name="zyfa._C",
            sources=[f"csrc/{f}" for f in ("bindings.cpp", "bf16_d128.hip", "bf16_d192.hip", "fp8_d128.hip",
                                           "fp8_d192.hip", "fp8_d128_asm.hip", "fp8_d192_asm.hip")],
            include_dirs=[str(ROOT / "csrc"), str(GEN), str(ROOT / "third_party" / "hipkittens" / "include")],
            extra_compile_args={"cxx": ["-O3", "-std=c++20"], "nvcc": hip_flags},
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
