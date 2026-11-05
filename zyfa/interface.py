"""Python interface to the ZYFA kernels, with the arguments of flash-attn's functions.

The kernels are registered as the custom operator ``torch.ops.zyfa.fwd``, so ``torch.compile`` traces through them and
calls can be captured in CUDA graphs.
"""
import glob
import math
import os
from typing import Optional

import torch

try:
    from . import _C
except ImportError as e:
    if glob.glob(os.path.join(os.path.dirname(__file__), "_C*.so")):
        raise ImportError(f"zyfa's extension failed to load ({e}). Rebuild it against this PyTorch with "
                          "`pip install . --no-build-isolation`") from e
    raise ImportError("zyfa's extension is not built. Run `pip install . --no-build-isolation`, and import zyfa from "
                      "outside the source tree") from e

FP8 = torch.float8_e4m3fn
_KERNELS = {(torch.bfloat16, 128, 128): _C.bf16_d128, (torch.bfloat16, 192, 128): _C.bf16_d192,
            (FP8, 128, 128): _C.fp8_d128, (FP8, 192, 128): _C.fp8_d192}
_ARCH_OK = set()


def _lse_shape(q, bshd):
    return (q.shape[0], q.shape[2], q.shape[1]) if bshd else tuple(q.shape[:3])


@torch.library.custom_op("zyfa::fwd", mutates_args=())
def _fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, v_scale: float, return_lse: bool,
         bshd: bool, q_descale: Optional[torch.Tensor], k_descale: Optional[torch.Tensor],
         v_descale: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _KERNELS[(q.dtype, q.shape[-1], v.shape[-1])]
    if q.dtype == FP8:
        res = fn(q, k, v, scale, True, return_lse, bshd, v_scale, q_descale, k_descale, v_descale)
    else:
        res = fn(q, k, v, scale, True, return_lse, bshd)
    lse = res[1].view(_lse_shape(q, bshd)) if return_lse else q.new_empty(0, dtype=torch.float32)
    return res[0], lse


@_fwd.register_fake
def _(q, k, v, scale, v_scale, return_lse, bshd, q_descale, k_descale, v_descale):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=torch.bfloat16)
    return out, q.new_empty(_lse_shape(q, bshd) if return_lse else 0, dtype=torch.float32)


@torch.library.custom_op("zyfa::no_backward", mutates_args=())
def _no_backward(grad: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("zyfa is forward only and has no backward pass")


@_no_backward.register_fake
def _(grad, like):
    return torch.empty_like(like)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3])


def _backward(ctx, dout, dlse):
    # The gradients are ops that raise when they run (they take dout, so they stay in the backward graph), so
    # torch.compile can trace a function whose backward is never called.
    return (*(_no_backward(dout, x) for x in ctx.saved_tensors), None, None, None, None, None, None, None)


_fwd.register_autograd(_backward, setup_context=_setup_context)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
                    alibi_slopes=None, deterministic=False, return_attn_probs=False, *, return_lse=False,
                    q_descale=None, k_descale=None, v_descale=None, layout="bshd"):
    """Causal attention forward pass with the arguments of ``flash_attn.flash_attn_func`` (flash-attn 2 order).

    Args:
        q: (batch, seqlen, nheads, d_qk), or (batch, nheads, seqlen, d_qk) with ``layout="bhsd"``.
        k: (batch, seqlen, nheads_k, d_qk). nheads must be a multiple of nheads_k (MHA, GQA, or MQA).
        v: (batch, seqlen, nheads_k, d_v). (d_qk, d_v) is (128, 128) or (192, 128), and q, k, and v are all
            bfloat16 or all float8_e4m3fn. Any seqlen works, and lengths that are not a multiple of 256 are padded
            with zeros, which copies q, k, and v.
        dropout_p: must be 0.
        softmax_scale: scale applied to q k^T before the softmax. Defaults to 1 / sqrt(d_qk).
        causal: must be True.
        window_size: (-1, -1), (-1, 0), or (None, None), which all give the causal mask.
        softcap, alibi_slopes: not supported, so they must keep their defaults.
        deterministic: accepted for compatibility. The output is always bitwise identical from run to run.
        return_attn_probs: also return the log-sum-exp, as (out, lse, None) like flash-attn 2.
        return_lse: also return the log-sum-exp, as (out, lse) like flash-attn 3.
        q_descale, k_descale, v_descale: FP8 only. Per-tensor factors that map the FP8 values back to their
            original range, as returned by ``fp8_quantize``. Each one is a float, a float32 tensor with one element
            on q's device, or such a tensor expanded to (batch, nheads_k), the shape that flash-attn 3 takes. The
            kernel reads tensor descales on the GPU and does not check their values, so they must be positive
            normal float32 values.
        layout: "bshd" (the default) or "bhsd".

    Returns:
        out: bfloat16, with the shape of q and d_v as its last dimension.
        lse: float32 (batch, nheads, seqlen), the natural-log log-sum-exp of each row of scaled scores, returned only
            with return_lse or return_attn_probs. In FP8 it is computed from the rounded probabilities, so that
            results over different key ranges can be merged consistently with the output.
    """
    _check_args(dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes)
    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")
    bshd = layout == "bshd"
    _check(q, k, v, bshd)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    v_scale = 1.0
    if q.dtype == FP8:
        hk = k.shape[2 if bshd else 1]
        q_f, q_descale = _descale(q_descale, "q_descale", q, hk)
        k_f, k_descale = _descale(k_descale, "k_descale", q, hk)
        v_scale, v_descale = _descale(v_descale, "v_descale", q, hk)
        scale *= q_f * k_f
    elif q_descale is not None or k_descale is not None or v_descale is not None:
        raise ValueError("descales apply to float8_e4m3fn inputs only")

    want_lse = return_lse or return_attn_probs
    sdim = 1 if bshd else 2
    s = q.shape[sdim]
    if s == 0:
        out = q.new_empty((*q.shape[:-1], v.shape[-1]), dtype=torch.bfloat16)
        lse = q.new_empty(_lse_shape(q, bshd), dtype=torch.float32)
    else:
        pad = -s % 256   # zero keys past the last query are masked, so padding is exact
        if pad:
            q, k, v = (torch.cat([x, x.new_zeros((*x.shape[:sdim], pad, *x.shape[sdim + 1:]))], sdim)
                       for x in (q, k, v))
        out, lse = torch.ops.zyfa.fwd(q, k, v, scale, v_scale, want_lse, bshd, q_descale, k_descale, v_descale)
        if pad:
            out = out.narrow(sdim, 0, s).contiguous()
            lse = lse.narrow(2, 0, s).contiguous() if want_lse else lse
    if not want_lse:
        return out
    return (out, lse, None) if return_attn_probs else (out, lse)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                              softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False, *,
                              return_lse=False, q_descale=None, k_descale=None, v_descale=None):
    """Attention over packed q, k, and v, the same as ``flash_attn_func`` on the three slices.

    qkv: (batch, seqlen, 3, nheads, 128). The slices are copied before the kernel runs. The other arguments are those
    of ``flash_attn_func``.
    """
    q, k, v = qkv.unbind(2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
                           deterministic, return_attn_probs, return_lse=return_lse, q_descale=q_descale,
                           k_descale=k_descale, v_descale=v_descale)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                             softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False, *,
                             return_lse=False, q_descale=None, k_descale=None, v_descale=None):
    """Attention over q and packed k and v, the same as ``flash_attn_func`` on q and the two slices.

    q: (batch, seqlen, nheads, 128), kv: (batch, seqlen, 2, nheads_k, 128). The slices are copied before the kernel
    runs. The other arguments are those of ``flash_attn_func``.
    """
    k, v = kv.unbind(2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
                           deterministic, return_attn_probs, return_lse=return_lse, q_descale=q_descale,
                           k_descale=k_descale, v_descale=v_descale)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                           deterministic=False, return_attn_probs=False, block_table=None, *, return_lse=False,
                           q_descale=None, k_descale=None, v_descale=None):
    """``flash_attn.flash_attn_varlen_func`` for batches of sequences of equal length.

    q: (total, nheads, d_qk), k: (total, nheads_k, d_qk), v: (total, nheads_k, d_v), with total = batch * max_seqlen
    and max_seqlen_q == max_seqlen_k. The lengths are checked from the shapes, so cu_seqlens is never read back to the
    host. block_table (a paged KV cache) is not supported. The other arguments are those of ``flash_attn_func``.

    Returns out (total, nheads, d_v) and, if requested, lse (nheads, total).
    """
    if block_table is not None:
        raise NotImplementedError("zyfa has no paged KV cache")
    b = cu_seqlens_q.shape[0] - 1
    if not (cu_seqlens_k.shape[0] == b + 1 and max_seqlen_q == max_seqlen_k and q.shape[0] == b * max_seqlen_q
            and k.shape[0] == b * max_seqlen_k and v.shape[0] == k.shape[0]):
        raise NotImplementedError("zyfa's varlen path needs sequences of equal length: total == batch * max_seqlen "
                                  "for q and k, and max_seqlen_q == max_seqlen_k")
    res = flash_attn_func(*(x.reshape(b, max_seqlen_q, *x.shape[1:]) for x in (q, k, v)), dropout_p, softmax_scale,
                          causal, window_size, softcap, alibi_slopes, deterministic, return_attn_probs,
                          return_lse=return_lse, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
    if not isinstance(res, tuple):
        return res.reshape(q.shape[0], *res.shape[2:])
    out = res[0].reshape(q.shape[0], *res[0].shape[2:])
    lse = res[1].transpose(0, 1).reshape(res[1].shape[1], q.shape[0])
    return (out, lse, None) if return_attn_probs else (out, lse)


def fp8_quantize(x):
    """Quantizes x to float8_e4m3fn with one power-of-two scale, on the GPU and without a host sync.

    Returns (x_fp8, descale), where descale is a 0-d float32 tensor and x is approximately x_fp8 * descale. The descale
    is the smallest power of two that brings the largest absolute value of x within the e4m3 range (at most 448), and
    it is 1 if that value is zero or not finite. A power of two changes only the exponent, so scaling adds no rounding
    error of its own.
    """
    amax = x.abs().amax().float()
    ok = (amax > 0) & torch.isfinite(amax)
    descale = torch.where(ok, torch.exp2(torch.ceil(torch.log2(amax / 448.0))), torch.ones_like(amax))
    return (x.float() / descale).to(FP8), descale


def _check_args(dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes):
    if dropout_p is None or isinstance(softmax_scale, bool):
        raise TypeError("these arguments look like flash-attn 3's order (q, k, v, softmax_scale, causal), but zyfa "
                        "uses flash-attn 2's (q, k, v, dropout_p, softmax_scale, causal), so pass them by keyword")
    if dropout_p:
        raise NotImplementedError("zyfa has no dropout (if this value is softmax_scale, pass it by keyword)")
    if not causal:
        raise NotImplementedError("zyfa is causal only, so pass causal=True")
    left, right = (-1 if w is None else w for w in window_size)
    if left >= 0 or right not in (-1, 0):
        raise NotImplementedError("zyfa has no sliding window")
    if softcap or alibi_slopes is not None:
        raise NotImplementedError("zyfa has no softcap or ALiBi")


def _check(q, k, v, bshd):
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise ValueError(f"q, k, v must be 4-D, got {q.dim()}, {k.dim()}, {v.dim()} dims")
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise ValueError("q, k, v must be on the same GPU")
    if not torch.compiler.is_compiling():
        _check_arch(q.device.index if q.device.index is not None else torch.cuda.current_device())
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must have one dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in (torch.bfloat16, FP8):
        raise TypeError(f"zyfa takes bfloat16 or float8_e4m3fn, got {q.dtype}")
    if (q.dtype, q.shape[-1], v.shape[-1]) not in _KERNELS:
        raise ValueError(f"no kernel for {q.dtype} with (d_qk, d_v) = ({q.shape[-1]}, {v.shape[-1]}). The kernels take "
                         "bfloat16 or float8_e4m3fn with (128, 128) or (192, 128)")
    if k.shape[-1] != q.shape[-1]:
        raise ValueError(f"q and k head dims differ: {q.shape[-1]} and {k.shape[-1]}")
    s, h = (1, 2) if bshd else (2, 1)
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError("q, k, v must have the same batch size")
    if not (q.shape[s] == k.shape[s] == v.shape[s]):
        raise NotImplementedError(f"zyfa needs seqlen_q == seqlen_k == seqlen_v, got {q.shape[s]}, {k.shape[s]}, "
                                  f"{v.shape[s]}")
    if k.shape[h] == 0 or k.shape[h] != v.shape[h] or q.shape[h] % k.shape[h]:
        raise ValueError(f"nheads ({q.shape[h]}) must be a multiple of nheads_k ({k.shape[h]}), and k and v must "
                         "have the same number of heads")


def _check_arch(device):
    if device not in _ARCH_OK:
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if not arch.startswith("gfx950"):
            raise RuntimeError(f"zyfa's kernels are built for gfx950 (MI355X), but this GPU is {arch}")
        _ARCH_OK.add(device)


def _descale(x, name, q, hk):
    """-> (factor applied on the host, tensor read by the kernel)."""
    if x is None:
        return 1.0, None
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.float32 or x.device != q.device:
            raise TypeError(f"{name} must be a float32 tensor on q's device")
        expanded = x.dim() == 2 and tuple(x.shape) == (q.shape[0], hk) and all(
            st == 0 for n, st in zip(x.shape, x.stride()) if n > 1)
        if x.numel() != 1 and not expanded:
            raise NotImplementedError(f"zyfa takes one {name} per tensor, so pass a 1-element tensor or a 1-element "
                                      "tensor expanded to (batch, nheads_k) with .expand()")
        return 1.0, x
    x = float(x)
    if not 1.1754943508222875e-38 <= x <= 3.4028234663852886e38:
        raise ValueError(f"FP8 descales must be positive normal fp32 values, got {x}")
    return x, None
