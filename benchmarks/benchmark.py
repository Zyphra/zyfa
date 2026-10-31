"""Causal attention forward throughput in TFLOP/s for ZYFA, AITER (if installed), and PyTorch SDPA (with --sdpa).

    python benchmarks/benchmark.py --dtype fp8 --dims 128 --seqlens 2048 16384 32768

The protocol follows the kernel benchmarks in the README. Each call has max(32, 262144 / seqlen) query heads at batch 1
(GQA with --group) and q, k, and v drawn from N(0, 0.1^2). The time of a call is the GPU time of every kernel it
launches, measured with torch.profiler, as the mean of 100 calls after 500 warm-up calls, with 5 s of idle time before
each seqlen >= 16K. FLOPs = B * H * S^2 * (d_qk + d_v). FP8 calls pass all three descales as GPU tensors.

Gaussian inputs make attention nearly uniform, and on activations captured from real models at sustained clocks the
FP8 gain over AITER is smaller (see the README).
"""
import argparse
import os
import time

os.environ.setdefault("KINETO_LOG_LEVEL", "5")   # quiet torch.profiler

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from zyfa import flash_attn_func, fp8_quantize

try:
    import aiter
except ImportError:
    aiter = None


def kernel_time(fn, warmup, iters):
    """Mean GPU time per call, in seconds, of the kernels fn launches."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    us = sum(e.device_time_total for e in prof.key_averages() if e.device_time_total > 0 and e.device_type.name in
             ("CUDA", "HIP"))
    return us * 1e-6 / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--dims", type=int, choices=[128, 192], default=128, help="d_qk; d_v is 128")
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=0, help="query heads, 0 = max(32, 262144 / seqlen)")
    p.add_argument("--group", type=int, default=1, help="query heads per KV head")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--cooldown", type=float, default=5.0, help="seconds idle before each seqlen >= 16K")
    p.add_argument("--sdpa", action="store_true", help="also time torch SDPA (BF16, d=128)")
    a = p.parse_args()
    with profile(activities=[ProfilerActivity.CUDA]):   # a process's first profiler session reads kernels ~15% slow
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()

    dqk, dv = a.dims, 128
    sdpa = a.sdpa and a.dtype == "bf16" and dqk == dv
    cols = ["zyfa", "aiter"] + (["sdpa"] if sdpa else [])
    print(f"{'seqlen':>7} {'heads':>6} " + " ".join(f"{c:>8}" for c in cols) + "  TF/s")
    for s in a.seqlens:
        hq = a.heads or max(32, 262144 // s)
        hkv = hq // a.group
        q = torch.randn(a.batch, s, hq, dqk, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(a.batch, s, hkv, dqk, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(a.batch, s, hkv, dv, device="cuda", dtype=torch.bfloat16) * 0.1
        flops = a.batch * hq * s * s * (dqk + dv)
        row = []

        def tf(fn):
            if s >= 16384:
                time.sleep(a.cooldown)
            row.append(f"{flops / kernel_time(fn, a.warmup, a.iters) / 1e12:8.1f}")

        if a.dtype == "fp8":
            (q8, sq), (k8, sk), (v8, sv) = fp8_quantize(q), fp8_quantize(k), fp8_quantize(v)
            tf(lambda: flash_attn_func(q8, k8, v8, causal=True, q_descale=sq, k_descale=sk, v_descale=sv))
            if aiter:
                d = [x.reshape(1) for x in (sq, sk, sv)]
                tf(lambda: aiter.flash_attn_fp8_pertensor_func(q8, k8, v8, *d, causal=True))
            else:
                row.append(f"{'-':>8}")
        else:
            tf(lambda: flash_attn_func(q, k, v, causal=True))
            tf(lambda: aiter.flash_attn_func(q, k, v, causal=True)) if aiter else row.append(f"{'-':>8}")
            if sdpa:
                qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                tf(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=hkv != hq))
        print(f"{s:>7} {hq:>6} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
