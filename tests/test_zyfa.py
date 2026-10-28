import math

import pytest
import torch

from zyfa import flash_attn_func, fp8_quantize

BF16_DIMS = [(128, 128), (192, 128)]
FP8_DIMS = [(128, 128), (192, 128)]
HEADS = [(8, 8), (8, 2)]


def reference(q, k, v, scale):
    """fp64 causal attention, BHSD; returns O and the natural-log LSE."""
    q, k, v = q.double(), k.double(), v.double()
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = q @ k.transpose(-1, -2) * scale
    n = s.shape[-1]
    s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    lse = s.logsumexp(-1)
    return (s - lse[..., None]).exp() @ v, lse


def inputs(b, hq, hkv, s, dqk, dv, seed=0, grow=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rn = lambda *shape: torch.randn(*shape, generator=g, device="cuda")
    q, k, v = rn(b, hq, s, dqk), rn(b, hkv, s, dqk), rn(b, hkv, s, dv)
    if grow:  # logits of later keys keep rising, so the running max grows and O is rescaled
        k[..., 0] += torch.arange(s, device="cuda") * (8.0 / 64)
        q[..., 0] = 1.0
    return q, k, v


def rel_rms(x, ref):
    return ((x.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def run(q, k, v, layout, **kw):
    t = (lambda x: x.transpose(1, 2).contiguous()) if layout == "bshd" else (lambda x: x)
    out = flash_attn_func(t(q), t(k), t(v), causal=True, layout=layout, **kw)
    if isinstance(out, tuple):
        return t(out[0]), out[1]
    return t(out)


def run_fp8(q, k, v, layout="bhsd"):
    (q8, sq), (k8, sk), (v8, sv) = fp8_quantize(q), fp8_quantize(k), fp8_quantize(v)
    out, lse = run(q8, k8, v8, layout, return_lse=True, q_descale=sq, k_descale=sk, v_descale=sv)
    ref, ref_lse = reference(q8.double() * sq, k8.double() * sk, v8.double() * sv, 1 / math.sqrt(q.shape[-1]))
    return out, lse, ref, ref_lse


@pytest.mark.parametrize("dims", BF16_DIMS)
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("b,s,grow", [(1, 1024, False), (2, 512, False), (1, 8192, False), (1, 2048, True)])
def test_bf16(dims, heads, layout, b, s, grow):
    q, k, v = (x.bfloat16() for x in inputs(b, *heads, s, *dims, grow=grow))
    out, lse = run(q, k, v, layout, return_lse=True)
    ref, ref_lse = reference(q, k, v, 1 / math.sqrt(dims[0]))
    assert out.dtype == torch.bfloat16
    assert rel_rms(out, ref) < 5e-3
    assert (lse.double() - ref_lse).abs().max().item() < 1e-4


@pytest.mark.parametrize("dims", FP8_DIMS)
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("b,s,grow", [(1, 1024, False), (2, 512, False), (1, 8192, False), (1, 2048, True)])
def test_fp8(dims, heads, layout, b, s, grow):
    out, lse, ref, ref_lse = run_fp8(*inputs(b, *heads, s, *dims, grow=grow), layout)
    assert rel_rms(out, ref) < 5e-2
    # the LSE comes from the row sum of the e4m3 P, so it carries P's rounding
    assert (lse.double() - ref_lse).abs().max().item() < 6e-2


@pytest.mark.parametrize("dtype,dims", [(torch.bfloat16, (128, 128)), (torch.bfloat16, (192, 128)),
                                        (torch.float8_e4m3fn, (128, 128)), (torch.float8_e4m3fn, (192, 128))])
def test_deterministic(dtype, dims):
    q, k, v = (x.to(dtype) for x in inputs(1, 8, 2, 4096, *dims))
    a = flash_attn_func(q, k, v, causal=True, layout="bhsd")
    b = flash_attn_func(q, k, v, causal=True, layout="bhsd")
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_flash_attn_api():
    q, k, v = (x.bfloat16().transpose(1, 2).contiguous() for x in inputs(1, 8, 2, 512, 128, 128))
    out = flash_attn_func(q, k, v, 0.0, None, True)
    out2, lse, probs = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    assert torch.equal(out, out2) and lse.shape == (1, 8, 512) and probs is None


def test_rejects_unsupported():
    q, k, v = (x.bfloat16() for x in inputs(1, 8, 8, 512, 128, 128))
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, causal=False, layout="bhsd")
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, causal=True, window_size=(128, 0), layout="bhsd")
    with pytest.raises(ValueError):
        flash_attn_func(q[:, :, :300], k[:, :, :300], v[:, :, :300], causal=True, layout="bhsd")
    with pytest.raises(ValueError):
        flash_attn_func(q[0], k[0], v[0], causal=True)
    with pytest.raises(ValueError):
        flash_attn_func(q, k, v, causal=True, layout="bhsd", q_descale=1.0)
    with pytest.raises(TypeError):
        flash_attn_func(q.half(), k.half(), v.half(), causal=True, layout="bhsd")
    with pytest.raises(NotImplementedError):
        flash_attn_func(q.requires_grad_(), k, v, causal=True, layout="bhsd")
