# ZYFA

FlashAttention kernels for the AMD Instinct MI355X, from the Zyphra Optimization Engine (ZOE).

ZYFA is the attention engine of ZOE, which runs [Zyphra Cloud](https://cloud.zyphra.com). It computes causal forward
attention for the prefill phase of inference on CDNA4 GPUs (gfx950), in BF16 and FP8. The BF16 kernels are written in
HIP, AMD's C++ kernel language, and the FP8 kernels are hand-scheduled gfx950 assembly emitted by a Python generator.
The Python interface takes the arguments of `flash_attn.flash_attn_func`.

Our paper, [*ZYFA: Speed and Accuracy of FP8 FlashAttention on AMD CDNA4*](#citation) (arXiv link pending), describes
the design and the measurements summarized here.

- On Gaussian inputs (random q, k, and v, as in kernel benchmarks) at 2K–32K tokens, ZYFA's FP8 kernels achieve
  1.26–2.02× the throughput of the default FP8 kernels of AITER, AMD's library of inference kernels. ZYFA's BF16
  kernels are within 3% of AITER's throughput at 16K and 32K tokens and up to 7% higher at 2K.
- On activations captured from one layer each of Mistral-7B, Qwen3-8B, and DeepSeek-V2-Lite at 16K tokens, timed at
  sustained clocks (the steady clock after seconds of back-to-back launches), the FP8 kernels achieve 1.16–1.76× the
  throughput of AITER's. The gain is smaller than on Gaussian inputs because the sustained clock of the MI355X depends
  on the operand values ([details](#clock-speed-and-input-values)).
- The output error of one FP8 attention call is 62–78% lower than that of AITER's FP8 kernel on Mistral-7B and
  Qwen3-8B, and within a few percent of that of AMD's Composable Kernel and Triton kernels.
- Outputs are bitwise identical from run to run, and calls work under `torch.compile` and in CUDA graphs.

## Install

ZYFA needs an MI355X and a ROCm build of PyTorch. We test it with ROCm 7.2.3 and PyTorch 2.12.

```sh
git clone https://github.com/Zyphra/zyfa
cd zyfa
pip install . --no-build-isolation
```

The build compiles against the PyTorch that is already installed, so keep `--no-build-isolation`. It also assembles
the FP8 kernels with the ROCm compiler, which it looks for in `$ROCM_PATH` (default `/opt/rocm`).

## Usage

```python
import torch
from zyfa import flash_attn_func, fp8_quantize

# (batch, seqlen, heads, head_dim), with 32 query heads over 8 KV heads
q = torch.randn(1, 8192, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)

# BF16
out = flash_attn_func(q, k, v, causal=True)

# FP8 with one power-of-two scale per tensor, BF16 output
(q8, q_s), (k8, k_s), (v8, v_s) = fp8_quantize(q), fp8_quantize(k), fp8_quantize(v)
out = flash_attn_func(q8, k8, v8, causal=True, q_descale=q_s, k_descale=k_s, v_descale=v_s)
```

- `layout="bhsd"` takes (batch, heads, seqlen, head_dim) tensors instead.
- `return_lse=True` also returns the log-sum-exp of each row of scaled scores (natural log), with shape
  (batch, heads, seqlen).
- A descale is the factor that maps an FP8 tensor back to its original range, and ZYFA takes one descale per tensor.
  Each one is a Python float, a float32 GPU tensor with one element, or such a tensor expanded to (batch, nheads_k),
  the shape that flash-attn 3 takes. The kernel reads tensor descales from GPU memory and applies all three itself, so
  a call never waits for the host. For the same reason the values of tensor descales are not checked, so each one must
  hold a positive normal float32. `fp8_quantize` returns its descale as a GPU tensor.
- `flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func`, and `flash_attn_varlen_func` take the same arguments as
  their flash-attn counterparts. The varlen function needs sequences of equal length.
- Any sequence length works. A length that is not a multiple of 256 is padded with zeros up to the next multiple of
  256, which gives the exact result for causal attention (no real query can see the padded keys) but copies q, k,
  and v.
- The kernels are registered as the custom operator `torch.ops.zyfa.fwd`, so `torch.compile` traces through them,
  and calls can be captured in CUDA graphs. Run each new FP8 shape once before capturing it, because the first call
  for a shape builds a job table (the list of work items of the kernel) on the host and copies it to the GPU, which a
  graph capture cannot record.

### Supported configurations

| | |
|---|---|
| GPU | MI355X (gfx950) |
| Pass | forward only |
| Mask | causal, with seqlen_q = seqlen_k |
| Head dimensions (d_qk of q and k, d_v of v) | d_qk = d_v = 128, or d_qk = 192 with d_v = 128 (the per-head form of multi-head latent attention that prefill uses) |
| Heads | MHA, GQA, and MQA |
| Inputs | BF16, or FP8 e4m3 (`torch.float8_e4m3fn`, with 4 exponent and 3 mantissa bits) with one descale per tensor |
| Outputs | BF16 output and, on request, the FP32 log-sum-exp |
| Batches | sequences of equal length |
| Determinism | bitwise identical outputs from run to run |

Not supported are decoding and KV caches, paged KV, batches of sequences with unequal lengths, non-causal masks and
sliding windows, dropout, softcap, ALiBi, FP16 inputs, per-head FP8 descales, and the backward pass.

### Differences from flash-attn

- Only causal attention is supported, so pass `causal=True`. A `window_size` of (-1, -1), (-1, 0), or (None, None)
  is accepted and gives the causal mask.
- Arguments follow flash-attn 2's order (q, k, v, dropout_p, softmax_scale, causal, ...). A positional call in
  flash-attn 3's order (q, k, v, softmax_scale, causal) raises a `TypeError`, so pass those arguments by keyword.
- `return_attn_probs=True` returns (out, lse, None). flash-attn 2 returns an empty tensor in the last place, and
  flash-attn 3 returns (out, lse), which is what `return_lse=True` gives here.
- There is no backward pass. Inputs that require gradients still run forward, eagerly or under `torch.compile`, and
  calling backward raises an error.
- Non-contiguous inputs, including the slices of packed qkv and kv tensors, are copied before the kernel runs.

## Performance

### Kernel benchmarks

For causal multi-head attention on Gaussian inputs, ZYFA's FP8 kernels achieve 1.34–2.02× the throughput of AITER's
default kernels, and ZYFA's BF16 kernels achieve 1.00–1.05×. The table gives the throughput in TFLOP/s on one MI355X
at batch 1 as ZYFA / AITER, with AITER's time divided by ZYFA's in parentheses.

| kernel | S = 2K | S = 16K | S = 32K |
|---|---|---|---|
| BF16, d = 128 | 1000 / 950 (1.05×) | 1237 / 1199 (1.03×) | 1265 / 1236 (1.02×) |
| BF16, d_qk = 192, d_v = 128 | 1008 / 967 (1.04×) | 1296 / 1284 (1.01×) | 1330 / 1329 (1.00×) |
| FP8, d = 128 | 1942 / 1347 (1.44×) | 2597 / 1938 (1.34×) | 2672 / 1944 (1.37×) |
| FP8, d_qk = 192, d_v = 128 | 1929 / 955 (2.02×) | 2391 / 1264 (1.89×) | 2464 / 1237 (1.99×) |

With grouped-query attention (four query heads per KV head), ZYFA achieves 1.26–1.42× the throughput of AITER in FP8
at d = 128, 1.82–1.94× in FP8 at d_qk = 192, and 1.00–1.07× in BF16. At 32K tokens the FP8 kernel at d = 128 reaches
53% of the FP8 datasheet peak, against 39% for AITER.

The baseline is the kernel that AITER 0.1.19 selects by default: its assembly kernel at d = 128, its OPUS kernel (a
HIP kernel written with AITER's C++ template library) for BF16 at d_qk = 192, and Composable Kernel for FP8 at
d_qk = 192. Composable Kernel issues the FP8 matrix instruction of CDNA3, which has half the throughput per clock of
the CDNA4 instruction that ZYFA and AITER's assembly kernel use, so part of the gain at d_qk = 192 comes from the
instruction alone. ZYFA's base HIP kernel at d_qk = 192, without the row sum on the matrix unit, per-block scaling, or
the assembly schedule, already achieves 1.51× the throughput of Composable Kernel at 32K.

Times are kernel-only (the device time of the kernel alone). Each measured time is the mean of 100 calls after 500
warm-up calls, averaged over at least three runs with the order of the kernels rotated between runs. The GPU idles for 5 s
before each configuration with S ≥ 16K. Each call has max(32, 262144 / S) query heads and Gaussian q, k, and v with
standard deviation 0.1, and throughput is B·H·S²·(d_qk + d_v) divided by the time, for batch size B, H query heads,
and sequence length S. The software is ROCm 7.2.3, PyTorch 2.12, and AITER 0.1.19, with the default clock policy and
a 1400 W power cap. Throughput follows the clock of the slowest chiplet, which differs between GPUs, so from 4K tokens
up the throughput of one configuration differs between the GPUs of one node by a median of 4% and at most 8%. Each
comparison therefore uses runs on one GPU.
[`benchmarks/benchmark.py`](benchmarks/benchmark.py) follows the same protocol, but it times kernels with
`torch.profiler` instead of `rocprofv3` and runs each configuration once, so its numbers can differ from the table by
a few percent.

### Captured activations at sustained clocks

On captured activations the FP8 gain over AITER is smaller than on Gaussian inputs, and the BF16 kernels achieve
1.01–1.02× the throughput of AITER's. The table gives AITER's time divided by ZYFA's on the post-RoPE q, k, and v
(after the rotary position embedding, as the attention kernel receives them) of one layer per model, captured on
WikiText-2 at S = 16K, as the median of three runs. After 2 s of idle time each kernel runs back to back for 1 s of
warm-up and at least 3 s of timing, so that the clock settles as it would in a long prefill.

| model (layer) | FP8 | BF16 |
|---|---|---|
| Mistral-7B-v0.3 (layer 20), d = 128 | 1.16× | 1.02× |
| Qwen3-8B (layer 18), d = 128 | 1.17× | 1.01× |
| DeepSeek-V2-Lite (layer 13), d_qk = 192 | 1.76× | 1.01× |

#### Clock speed and input values

Under sustained load the MI355X lowers its clock, and the clock falls further the more operand bits change from one
instruction to the next. Horace He showed the same effect for matrix multiplications on NVIDIA GPUs,
where inputs whose bits change less draw less power and therefore run at a higher clock
([Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data!](https://www.thonking.ai/p/strangely-matrix-multiplications)).
Gaussian inputs with standard deviation 0.1 make attention nearly uniform,
so every probability is close to the row maximum and every kernel writes normal FP8 values. In real activations most
probabilities lie far below the row maximum. In Mistral-7B layer 20, for 91% of the pairs of a row and a 64-key
block, the largest probability of the block lies more than 8 binades (a factor of 256) below the row maximum. AITER's
assembly kernel, which does not scale these blocks up, writes subnormal values or zeros there. Subnormal values and
zeros switch few bits in the matrix products, which is one reason why the clock of AITER's assembly kernel rises on the
Mistral-7B and Qwen3-8B layers. ZYFA multiplies each 64-key block of probabilities by a power of two before rounding
(per-block scaling), which keeps the same values normal and accurate but costs clock speed. ZYFA still needs fewer
cycles than AITER on both kinds of input. Benchmarks on Gaussian inputs therefore overstate the gain, and we
recommend timing attention kernels on captured activations at sustained clocks.

## Accuracy

ZYFA's FP8 kernels add less error than AITER's default FP8 kernels on all three models, and the error of ZYFA's BF16
kernels is 1.00–1.02× that of AITER's. The table gives the output error of one attention call as the relative RMS
error against fp64 attention on the same quantized inputs, so it counts only the error that each kernel adds. Values
are averaged over all layers of each model at S = 4K on WikiText-2.

| model | ZYFA FP8 | AITER FP8 | ZYFA BF16 | AITER BF16 |
|---|---|---|---|---|
| Mistral-7B-v0.3 | 1.20e-2 | 6.41e-2 | 1.81e-3 | 1.77e-3 |
| Qwen3-8B | 9.11e-3 | 3.17e-2 | 1.75e-3 | 1.73e-3 |
| DeepSeek-V2-Lite | 8.36e-3 | 1.02e-2 (Composable Kernel) | 1.94e-3 | 1.94e-3 |

On Mistral-7B and Qwen3-8B, ZYFA's FP8 error is 62–78% lower than that of AITER's assembly kernel and within a few
percent of that of Composable Kernel and Triton. AITER's assembly kernel divides each output by the sum of the
unrounded probabilities, so probabilities that underflow to zero in the numerator still count in the denominator, and
every output row shrinks slightly. ZYFA divides by the sum of the same rounded probabilities that multiply V, so each
output remains a weighted average whose weights sum to one.

At d = 128 the FP8 kernel computes probabilities with lean P, which writes the bit pattern (the code) of each FP8
probability with one multiply-add instead of an exponential ([below](#kernel-design)). On the 68 captured layers at
d = 128, lean P raises the error of one call to 1.22–1.36× that of exact P, which evaluates the exponential. Lean P
also raises the KL divergence of Mistral-7B from a model with fp64 attention by 1.7% relative to exact P. The FP8
log-sum-exp comes from the rounded probabilities, so that results computed over different key ranges can be merged
consistently. With lean P the log-sum-exp differs from the fp64 value by 0.02–0.04 nats in the worst row of our
synthetic inputs.

ZYFA's outputs are identical in every launch. AITER's FP8 assembly kernel changes its output in a few launches when a
GEMM (a matrix multiplication) runs between them.

## Using FP8 attention in a model

In full models most of the error of FP8 attention comes from quantizing q, k, and v rather than from rounding the
probabilities. On Mistral-7B, quantizing the inputs adds 8× the error of rounding the probabilities with lean P. A
Hadamard rotation of q and k before quantization roughly halves the input error. The rotation matrix R is orthogonal,
so (qR)(kR)ᵀ = qkᵀ and only the rounding error changes. The rotation spreads channels with large values over all
channels, which makes the rounding errors of the products smaller. Across the three models the rotation lowers the KL
divergence from the BF16 model by 27–54%, and it adds 0.1–3.4% to the attention time when it is fused into the pass
that quantizes q.

```python
def random_hadamard(d, device, seed=0):
    """Orthogonal d x d matrix diag(s) H / sqrt(d) with random signs s, for d a power of two."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(device)
    return s[:, None] * h / d ** 0.5

R = random_hadamard(128, q.device)  # one matrix for every layer and head
# apply after RoPE, and leave v unrotated
q8, q_s = fp8_quantize(q.float() @ R)
k8, k_s = fp8_quantize(k.float() @ R)
v8, v_s = fp8_quantize(v)
out = flash_attn_func(q8, k8, v8, causal=True, q_descale=q_s, k_descale=k_s, v_descale=v_s)
```

- For d_qk = 192, which is not a power of two, the paper combines a 3-point discrete cosine transform with a 64-point
  Hadamard matrix.
- Subtracting the mean key of each head over the sequence before the rotation (K smoothing) lowers the KL divergence
  further on all three models. The subtracted mean cancels in the softmax only if every key of a row uses the same
  mean, so chunked prefill and prefix caching need a mean that is fixed in advance.
- Static scales, set per layer from calibration data, give the same accuracy as per-call scales, and they let a fused
  kernel quantize q without first computing its maximum.
- Keeping the most sensitive heads in BF16 is not a useful middle ground, because the error is spread over most heads.
  On Mistral-7B and Qwen3-8B, even 90% of heads in BF16 does not match BF16 accuracy, and that mix already takes as
  long as all-BF16 attention.
- Measure FP8 accuracy as the KL divergence from the BF16 model rather than as a change in perplexity. In our runs
  AITER's FP8 kernel lowered the negative log-likelihood of Qwen3-8B below that of BF16 while it had the largest KL
  divergence of all FP8 variants.

## Kernel design

The paper describes each part in detail. The main ideas are the following.

- **Lean P (FP8, d = 128).** Read as an integer, the 8-bit code of a positive e4m3 number grows by 8 each time its
  value doubles, so the code is close to 8·log2 of the value plus a constant. The kernel therefore writes the FP8 code
  of each probability p = 2^x (x is the score minus the row maximum in base-2 units) directly with one multiply-add
  and one integer conversion, instead of an exponential and an FP8 conversion. For 32 queries against a
  128-key block, lean P lowers the vector-pipe cycles of the softmax from 896 to 384, against 1024 cycles of matrix
  work. At d_qk = 192 the kernel uses exact P, because lean P falls below the signal-to-noise threshold of our
  correctness test in 1 of 24 configurations there.
- **Per-block scaling.** Most probabilities in a long row lie many binades below the row maximum, where e4m3 values
  become subnormal (below 2^-6) or zero (below 2^-10). The FP8 kernels multiply each 64-key block of probabilities by a
  power of two so that the largest value of the block gets the code of 1, and the scaled FP8 matrix instruction of
  CDNA4 removes the factor at no extra cost.
- **Row sum on the matrix unit.** One extra matrix instruction (MFMA, matrix fused multiply-add) per 64-key block
  multiplies the FP8 probabilities by a vector of ones, so the denominator is the sum of the same rounded values as the
  numerator. The extra MFMA adds 6% to the matrix work and removes vector additions from the vector unit, which is the
  bottleneck of FP8 attention.
- **Ping-pong schedule.** The FP8 kernels are generated as gfx950 assembly by the Python scripts in `csrc/asm`. The two
  waves (groups of 64 threads that issue instructions together) that share each SIMD unit alternate between issuing
  the matrix instructions and computing the softmax, so the matrix pipe and the vector pipe work at the same time. The
  MI355X does not track the completion of matrix instructions in hardware, so this overlap depends entirely on the
  order of the instructions. With lean P the matrix pipe is busy for 86% of the cycles in the steady state of the key
  loop.
- **Work distribution.** Causal query tiles are paired first with last, so every job (one pair of tiles) does the same
  amount of work. A persistent grid of one workgroup per compute unit (CU) runs the jobs, and each workgroup loops over
  many jobs instead of exiting after one. The grid is remapped so that each of the eight chiplets (XCDs) processes
  consecutive tile pairs of one head, which then share K and V in that chiplet's L2 cache.
- **Lazy rescaling.** The output is rescaled only when the running maximum of a row grows by more than a threshold.
  The threshold, in base-2 units of the score, is 0.125 in FP8, 0.5 in BF16 at d = 128, and 8 in BF16 at d_qk = 192.
- **BF16 kernels.** BF16 at d = 128 uses the 8-wave ping-pong schedule of HipKittens (a library of attention and matrix
  kernels for AMD GPUs) and keeps two score tiles in flight. BF16 at d_qk = 192 uses an 8-stage pipeline whose stages
  follow AMD's OPUS kernel. Waves 4–7 run one stage behind waves 0–3, so on each SIMD the matrix instructions of one
  wave overlap the reads of the other wave from the local data share (LDS), the shared memory of a CU.
- Scores are computed transposed (K is the A operand of the MFMA), so each lane holds the scores of one query, and P
  goes straight from registers into the PV MFMA. K and V stream through LDS, in double buffers in the BF16 kernels and
  4-slot rings in the FP8 kernels.
- Tensors whose byte offsets exceed 32 bits run a HIP version of the FP8 kernel, which computes the same output. The
  log-sum-exp of the HIP version can differ by a few millionths.
- Every conversion rounds to nearest even.

## Tests and benchmarks

```sh
python -m pytest tests
python benchmarks/benchmark.py --dtype fp8 --dims 128
```

The tests compare every kernel with fp64 attention. The benchmark follows the protocol of the
[kernel benchmarks](#kernel-benchmarks) and compares with AITER when it is installed. After editing anything under
`csrc/`, delete `build/` before rebuilding, because the build does not track the headers or the assembly generators.

## Repository layout

| path | contents |
|---|---|
| `zyfa/` | Python interface (`flash_attn_func` and its variants, `fp8_quantize`) |
| `csrc/` | HIP kernels, launchers, and the PyTorch bindings |
| `csrc/asm/` | Python generators of the FP8 assembly kernels |
| `tests/` | correctness tests against fp64 attention |
| `benchmarks/` | throughput benchmark against AITER and PyTorch scaled dot-product attention (SDPA) |
| `third_party/hipkittens/` | vendored HipKittens headers |

## Citation

```bibtex
@misc{anthony2026zyfa,
  title  = {{ZYFA}: Speed and Accuracy of {FP8} {FlashAttention} on {AMD} {CDNA4}},
  author = {Anthony, Quentin and Bikshandi, Ganesh and Medepalli, Praneeth and Ebert, Stephen},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Apache 2.0. `csrc/bf16_d128.hip` and `csrc/tiles.cuh` are adapted from
[HipKittens](https://github.com/HazyResearch/HipKittens) (MIT, Copyright (c) 2024 HazyResearch), and
`third_party/hipkittens` is vendored from it under the same license. The BF16 d_qk = 192 pipeline follows the stages
of AMD's OPUS kernel in [AITER](https://github.com/ROCm/aiter) (MIT).
