# ZYFA

Flash attention for AMD Instinct MI355X, from the Zyphra Optimization Engine (ZOE).

ZYFA is ZOE's attention kernel engine for MI355X. ZOE is the engine behind [Zyphra Cloud](https://cloud.zyphra.com)
and spans other hardware too, including AMD MI300X and AWS Trainium2. ZYFA is a causal forward FlashAttention for
CDNA4 (gfx950) in BF16 and FP8. The BF16 kernels are HIP, and the FP8 kernels are hand-scheduled gfx950 assembly.

On MI355X, ZYFA's FP8 kernels are 1.3-1.9x faster than AITER's on random inputs and 1.16-1.76x on activations
captured from real models. Its BF16 kernels match AITER.

## Performance

TF/s on one MI355X, batch 1, causal, multi-head. ZYFA / AITER (ratio):

| config | S = 2K | S = 16K | S = 32K |
|---|---|---|---|
| BF16, d=128 | 983 / 941 (1.04x) | 1233 / 1200 (1.03x) | 1250 / 1219 (1.03x) |
| BF16, d_qk=192, d_v=128 | 959 / 917 (1.05x) | 1290 / 1288 (1.00x) | 1311 / 1316 (1.00x) |
| FP8, d=128 | 1804 / 1313 (1.37x) | 2549 / 1882 (1.35x) | 2601 / 1937 (1.34x) |
| FP8, d_qk=192, d_v=128 | 1707 / 930 (1.83x) | 2338 / 1232 (1.90x) | 2375 / 1224 (1.94x) |

With grouped-query attention (4 query heads per KV head), FP8 d=128 is 1.27-1.37x AITER and BF16 d=128 is
1.02-1.05x.

Setup: `benchmarks/benchmark.py` on one GPU, Oct 28, 2026. Time is the GPU time of every kernel a call launches, the
mean of 100 calls after 500 warm-up calls, with 5 s idle before each S >= 16K. Heads = max(32, 262144 / S),
q, k, v ~ N(0, 0.1²), TF/s = B·H·S²·(d_qk + d_v) / time. FP8 calls pass all three descales as GPU tensors. AITER
0.1.19 with its default kernel for each config: assembly at d=128, OPUS for BF16 d_qk=192, Composable Kernel for FP8
d_qk=192. ROCm 7.2.3, PyTorch 2.12. Runs on other GPUs of the same node differ by a few percent.

N(0, 0.1²) inputs give near-uniform attention. On q, k, v captured from real models at S = 16K, with sustained clocks,
FP8 d=128 is 1.16x (Mistral-7B-v0.3) and 1.17x (Qwen3-8B) AITER, FP8 d_qk=192 is 1.76x (DeepSeek-V2-Lite) and BF16 is
1.00-1.02x.

## Install

Needs an MI355X and a ROCm build of PyTorch. Tested with ROCm 7.2.3 and PyTorch 2.12.

```sh
git clone https://github.com/Zyphra/zyfa
cd zyfa
pip install . --no-build-isolation
```

The build compiles against the PyTorch you already have, so keep `--no-build-isolation`.

## Usage

`flash_attn_func` takes the arguments of `flash_attn.flash_attn_func`.

```python
import torch
from zyfa import flash_attn_func, fp8_quantize

# (batch, seqlen, heads, head_dim); 32 query heads over 8 KV heads
q = torch.randn(1, 8192, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)

# FP8: per-tensor scales, BF16 output
(q8, q_s), (k8, k_s), (v8, v_s) = fp8_quantize(q), fp8_quantize(k), fp8_quantize(v)
out = flash_attn_func(q8, k8, v8, causal=True, q_descale=q_s, k_descale=k_s, v_descale=v_s)
```

- `layout="bhsd"` takes (batch, heads, seqlen, head_dim) tensors.
- `return_lse=True` also returns the log-sum-exp, shape (batch, heads, seqlen).
- FP8 descales are per tensor: a float, a float32 GPU tensor with one value, or one value expanded to
  (batch, nheads_k), the shape flash-attn 3 takes. The kernel reads tensor descales from GPU memory and applies all
  three itself, so a call never waits on the host. Checking a tensor's value would need that wait, so it must hold a
  positive normal float32. `fp8_quantize` returns its descale as a GPU tensor.
- `flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func` and `flash_attn_varlen_func` take the same arguments as
  flash-attn's. The varlen one needs sequences of equal length.
- Any sequence length works. Lengths that are not a multiple of 256 are zero-padded up to one, which is exact for
  causal attention but copies q, k and v.
- The kernels are registered as `torch.ops.zyfa.fwd`, so `torch.compile` traces through them, and calls can be
  captured in CUDA graphs. Run each new FP8 shape once before capturing: its first call builds a job table on the host.

## Differences from flash-attn

- Causal only: pass `causal=True`. `window_size` (-1, 0) and (None, None) mean the same mask and are accepted.
- Arguments follow flash-attn 2's order (q, k, v, dropout_p, softmax_scale, causal, ...). A positional call in
  flash-attn 3's order (q, k, v, softmax_scale, causal) raises a TypeError: pass those by keyword.
- `return_attn_probs=True` returns (out, lse, None). flash-attn 2 returns an empty tensor in the last place, and
  flash-attn 3 returns (out, lse), which is what `return_lse=True` gives here.
- FP8 descales cannot differ per head.
- No backward. Inputs that require grad run forward, eagerly or under `torch.compile`, and backward raises.
- Non-contiguous inputs, including the slices of packed qkv and kv, are copied before the kernel runs.
- Not supported: KV cache and decode, paged KV, sequences of unequal length, dropout, softcap, ALiBi, sliding
  windows, FP16.

## What is supported

- MI355X (gfx950).
- Forward, causal self-attention with seqlen_q == seqlen_k.
- Head dims 128, and d_qk=192 with d_v=128 (MLA), in BF16 and in FP8 (e4m3).
- MHA, GQA and MQA; batched sequences of equal length.
- BF16 output, bitwise identical from run to run.

## Accuracy

Relative RMS error of the output against fp64 attention on the same inputs, N(0, 1) q, k, v: about 2e-3 in BF16 (the
rounding of a BF16 output) and 2e-2 in FP8. In FP8, P is rounded to e4m3, and at d=128 its exponential comes from a
piecewise-linear exp2. The FP8 log-sum-exp is computed from the row sums of that rounded P, so it differs from the
exact value by up to a few hundredths.

## How it works

- Scores are computed transposed (K is the MFMA A operand), so P goes straight from registers into the PV MFMA.
- BF16 d=128 uses HipKittens' 8-wave ping-pong with two score tiles in flight. BF16 d_qk=192 uses an 8-stage pipeline
  whose stages follow AMD's OPUS kernel.
- The FP8 kernels are generated by Python into gfx950 assembly (`csrc/asm`). The two waves on each SIMD alternate
  between MFMA and softmax phases, so the MFMA pipe and the vector pipe work at the same time.
- FP8 d=128 writes P straight as e4m3 codes from a piecewise-linear exp2: one FMA and one round-to-nearest-even
  convert per score, with per-block scales and the row sum done on the MFMA. FP8 d_qk=192 uses `v_exp`.
- The output is rescaled only when a row's max grows past a threshold, in log2 units: 0.5 for BF16 d=128, 8 for BF16
  d_qk=192, 0.125 for FP8.
- K and V stream through LDS: double buffers in the BF16 kernels, 4-tile rings in the FP8 kernels, with swizzled
  layouts.
- Causal query tiles are paired, first with last, so jobs carry equal work. A persistent grid assigns jobs to the 8
  chiplets (XCDs).
- Tensors with offsets past 32 bits run a HIP version of the FP8 kernel with the same output (the LSE can differ by a
  few millionths).
- Every conversion rounds to nearest even.

## Tests and benchmarks

```sh
python -m pytest tests
python benchmarks/benchmark.py --dtype fp8 --dims 128
```

The tests compare against fp64 attention. The benchmark uses the protocol of the table above and compares against
AITER when it is installed. After editing anything under `csrc/`, delete `build/` before rebuilding: the build does
not track headers or the assembly generators.

## License

Apache 2.0. `csrc/bf16_d128.hip` and `csrc/tiles.cuh` are adapted from
[HipKittens](https://github.com/HazyResearch/HipKittens) (MIT, Copyright (c) 2024 HazyResearch), and
`third_party/hipkittens` is vendored from it under the same license. The BF16 d_qk=192 pipeline follows the stages of
AMD's OPUS kernel in [AITER](https://github.com/ROCm/aiter) (MIT).
