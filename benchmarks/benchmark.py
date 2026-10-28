"""Causal attention forward TF/s: ZYFA vs AITER (if installed) and PyTorch SDPA.

python benchmarks/benchmark.py --dtype fp8 --dims 128 --seqlens 2048 8192 32768
Heads default to max(32, 262144 / seqlen) at batch 1, GQA with --kv-heads. Inputs are randn * 0.1 as in the README
table. FLOPs = B * H * S^2 * (d_qk + d_v). Times are CUDA events around back-to-back calls, so they include launch
gaps the README's kernel-only numbers leave out. For FP8, "zyfa" leaves v_descale out (as the README table does) and
"zyfa+v" applies it, which adds one elementwise multiply on the output.
"""
import argparse

import torch
import torch.nn.functional as F

from zyfa import flash_attn_func, fp8_quantize

try:
    import aiter
except ImportError:
    aiter = None


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--dims", type=int, choices=[128, 192], default=128, help="d_qk; d_v is 128")
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=0, help="query heads, 0 = max(32, 262144 / seqlen)")
    p.add_argument("--kv-heads", type=int, default=0, help="0 = same as --heads")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=100)
    a = p.parse_args()

    dqk, dv = a.dims, 128
    cols = ["zyfa", "zyfa+v", "aiter"] if a.dtype == "fp8" else ["zyfa", "aiter", "sdpa"]
    print(f"{'seqlen':>7} {'heads':>6} " + " ".join(f"{c:>8}" for c in cols) + "  TF/s")
    for s in a.seqlens:
        hq = a.heads or max(32, 262144 // s)
        hkv = a.kv_heads or hq
        q = torch.randn(a.batch, s, hq, dqk, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(a.batch, s, hkv, dqk, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(a.batch, s, hkv, dv, device="cuda", dtype=torch.bfloat16) * 0.1
        flops = a.batch * hq * s * s * (dqk + dv)
        tf = lambda fn: f"{flops / bench(fn, a.warmup, a.iters) / 1e12:8.1f}"
        row = []
        if a.dtype == "fp8":
            (q8, sq), (k8, sk), (v8, sv) = fp8_quantize(q), fp8_quantize(k), fp8_quantize(v)
            row.append(tf(lambda: flash_attn_func(q8, k8, v8, causal=True, q_descale=sq, k_descale=sk)))
            row.append(tf(lambda: flash_attn_func(q8, k8, v8, causal=True, q_descale=sq, k_descale=sk, v_descale=sv)))
            if aiter:
                d = [torch.full((1,), x, device="cuda") for x in (sq, sk, sv)]
                row.append(tf(lambda: aiter.flash_attn_fp8_pertensor_func(q8, k8, v8, *d, causal=True)))
            else:
                row.append(f"{'-':>8}")
        else:
            row.append(tf(lambda: flash_attn_func(q, k, v, causal=True)))
            row.append(tf(lambda: aiter.flash_attn_func(q, k, v, causal=True)) if aiter else f"{'-':>8}")
            if dqk == dv:
                qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                row.append(tf(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=hkv != hq)))
            else:
                row.append(f"{'-':>8}")
        print(f"{s:>7} {hq:>6} " + " ".join(row))


if __name__ == "__main__":
    main()
