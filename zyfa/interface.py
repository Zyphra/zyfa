import functools
import math

import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError("zyfa's extension is not built: run `pip install . --no-build-isolation`, and import zyfa from "
                      "outside the source tree") from e

_KERNELS = {(torch.bfloat16, 128, 128): _C.bf16_d128, (torch.bfloat16, 192, 128): _C.bf16_d192,
            (torch.float8_e4m3fn, 128, 128): _C.fp8_d128, (torch.float8_e4m3fn, 192, 128): _C.fp8_d192}


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
                    alibi_slopes=None, deterministic=False, return_attn_probs=False, *, return_lse=False,
                    q_descale=None, k_descale=None, v_descale=None, layout="bshd"):
    """Causal attention forward with the arguments of flash_attn.flash_attn_func.

    q: (batch, seqlen, nheads, d_qk), k: (batch, seqlen, nheads_k, d_qk), v: (batch, seqlen, nheads_k, d_v),
    or (batch, nheads, seqlen, d) with layout="bhsd". bf16 or float8_e4m3fn inputs, bf16 output.
    FP8 descales are per tensor. Returns out, (out, lse) with return_lse, or (out, lse, None) with return_attn_probs.
    """
    if not causal:
        raise NotImplementedError("zyfa is causal only, pass causal=True")
    if dropout_p or tuple(window_size) != (-1, -1) or softcap or alibi_slopes is not None:
        raise NotImplementedError("zyfa has no dropout, sliding window, softcap or ALiBi")
    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")
    bshd = layout == "bshd"
    _check(q, k, v, bshd)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if q.dtype == torch.float8_e4m3fn:
        scale *= _descale(q_descale) * _descale(k_descale)
    elif q_descale is not None or k_descale is not None or v_descale is not None:
        raise ValueError("descales apply to float8_e4m3fn inputs only")
    fn = _KERNELS[(q.dtype, q.shape[-1], v.shape[-1])]

    res = fn(q, k, v, scale, True, return_lse or return_attn_probs, bshd)
    out = res[0]
    if v_descale is not None:
        out.mul_(_descale(v_descale))
    if not (return_lse or return_attn_probs):
        return out
    batch, nheads, seqlen = (q.shape[0], q.shape[2], q.shape[1]) if bshd else q.shape[:3]
    lse = res[1].view(batch, nheads, seqlen)
    return (out, lse, None) if return_attn_probs else (out, lse)


def fp8_quantize(x):
    """Per-tensor power-of-two scaling to float8_e4m3fn. Returns (x_fp8, descale), x ~= x_fp8 * descale."""
    amax = x.abs().max().item()
    descale = 2.0 ** math.ceil(math.log2(amax / 448.0)) if amax > 0 and math.isfinite(amax) else 1.0
    return (x.float() / descale).to(torch.float8_e4m3fn), descale


def _check(q, k, v, bshd):
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise ValueError(f"q, k, v must be 4-D, got {q.dim()}, {k.dim()}, {v.dim()} dims")
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise ValueError("q, k, v must be on the same GPU")
    _check_arch(q.device.index if q.device.index is not None else torch.cuda.current_device())
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q, k, v must have one dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(f"zyfa takes bfloat16 or float8_e4m3fn, got {q.dtype}")
    if (q.dtype, q.shape[-1], v.shape[-1]) not in _KERNELS:
        raise ValueError(f"no kernel for {q.dtype} with (d_qk, d_v) = ({q.shape[-1]}, {v.shape[-1]}); supported: "
                         "bfloat16 or float8_e4m3fn with (128, 128) or (192, 128)")
    if k.shape[-1] != q.shape[-1]:
        raise ValueError(f"q and k head dims differ: {q.shape[-1]} and {k.shape[-1]}")
    s, h = (1, 2) if bshd else (2, 1)
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError("q, k, v must have the same batch size")
    if not (q.shape[s] == k.shape[s] == v.shape[s]):
        raise NotImplementedError(f"zyfa needs seqlen_q == seqlen_k == seqlen_v, got {q.shape[s]}, {k.shape[s]}, "
                                  f"{v.shape[s]}")
    if q.shape[s] % 256:
        raise ValueError(f"seqlen must be a multiple of 256, got {q.shape[s]}")
    if k.shape[h] != v.shape[h] or q.shape[h] % k.shape[h]:
        raise ValueError(f"nheads ({q.shape[h]}) must be a multiple of nheads_k ({k.shape[h]}), the same for k and v")
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError("zyfa is forward only: run it under torch.no_grad() or with inputs that do not "
                                  "require grad")


@functools.lru_cache(maxsize=None)
def _check_arch(device):
    arch = torch.cuda.get_device_properties(device).gcnArchName
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"zyfa's kernels are built for gfx950 (MI355X), this GPU is {arch}")


def _descale(x):
    if x is None:
        return 1.0
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise NotImplementedError("zyfa takes per-tensor FP8 descales only")
        x = x.item()
    x = float(x)
    if not (x > 0 and math.isfinite(x)):
        raise ValueError(f"FP8 descales must be positive and finite, got {x}")
    return x
