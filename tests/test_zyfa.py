import math

import pytest
import torch

from zyfa import (flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func, flash_attn_varlen_func,
                  fp8_quantize)

FP8 = torch.float8_e4m3fn
DIMS = [(128, 128), (192, 128)]
FP8_DIMS = [(64, 64), *DIMS]
HEADS = [(8, 8), (8, 2), (8, 1)]
SHAPES = [(1, 1024, False), (2, 512, False), (1, 8192, False), (1, 2048, True)]


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


def inputs(b, hq, hkv, s, dqk, dv, seed=0, grow=False, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    rn = lambda *shape: torch.randn(*shape, generator=g, device=device)
    q, k, v = rn(b, hq, s, dqk), rn(b, hkv, s, dqk), rn(b, hkv, s, dv)
    if grow:  # logits of later keys keep rising, so the running max grows and O is rescaled
        k[..., 0] += torch.arange(s, device=device) * (8.0 / 64)
        q[..., 0] = 1.0
    return q, k, v


def rel_rms(x, ref):
    return ((x.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def run(q, k, v, layout, **kw):
    t = (lambda x: x.transpose(1, 2).contiguous()) if layout == "bshd" else (lambda x: x)
    out = flash_attn_func(t(q), t(k), t(v), causal=True, layout=layout, **kw)
    if isinstance(out, tuple):
        return t(out[0]), out[1]
    return t(out)


def quantize(q, k, v, mult=(1.0, 1.0, 1.0)):
    """FP8 inputs with descales; mult (>= 1, so nothing saturates) moves each descale off a power of two."""
    out = []
    for x, m in zip((q, k, v), mult):
        x8, d = fp8_quantize(x)
        d = d * m
        out.append(((x.float() / d).to(FP8), d))
    return out


def check_fp8(out, lse, q8, k8, v8, dq, dk, dv):
    ref, ref_lse = reference(q8.double() * dq, k8.double() * dk, v8.double() * dv, 1 / math.sqrt(q8.shape[-1]))
    assert rel_rms(out, ref) < 5e-2
    # the LSE comes from the row sum of the e4m3 P, so it carries P's rounding
    assert (lse.double() - ref_lse).abs().max().item() < 6e-2


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("layout", ["bshd", "bhsd"])
@pytest.mark.parametrize("b,s,grow", SHAPES)
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
@pytest.mark.parametrize("b,s,grow", SHAPES)
def test_fp8(dims, heads, layout, b, s, grow):
    (q8, dq), (k8, dk), (v8, dv) = quantize(*inputs(b, *heads, s, *dims, grow=grow), mult=(1.3, 1.6, 1.1))
    out, lse = run(q8, k8, v8, layout, return_lse=True, q_descale=dq, k_descale=dk, v_descale=dv)
    check_fp8(out, lse, q8, k8, v8, dq, dk, dv)


@pytest.mark.parametrize("dims", FP8_DIMS)
@pytest.mark.parametrize("kind", ["float", "tensor", "expanded", "none"])
def test_fp8_descale_kinds(dims, kind):
    b, hq, hkv, s = 2, 8, 2, 1024
    (q8, dq), (k8, dk), (v8, dv) = quantize(*inputs(b, hq, hkv, s, *dims), mult=(1.4, 1.3, 1.9))
    if kind == "none":
        q8, k8, v8 = (x.to(FP8) for x in inputs(b, hq, hkv, s, *dims))
        dq, dk, dv = (torch.ones((), device="cuda") for _ in range(3))
        kw = {}
    elif kind == "float":
        kw = dict(q_descale=dq.item(), k_descale=dk.item(), v_descale=dv.item())
    elif kind == "tensor":
        kw = dict(q_descale=dq.reshape(1), k_descale=dk.reshape(1, 1), v_descale=dv)
    else:
        kw = {n: d.expand(b, hkv) for n, d in zip(("q_descale", "k_descale", "v_descale"), (dq, dk, dv))}
    out, lse = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", return_lse=True, **kw)
    check_fp8(out, lse, q8, k8, v8, dq, dk, dv)


@pytest.mark.parametrize("dims", FP8_DIMS)
def test_fp8_unit_descales_are_exact(dims):
    q8, k8, v8 = (x.to(FP8) for x in inputs(1, 8, 2, 2048, *dims))
    one = torch.ones(1, device="cuda")
    a, la = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", return_lse=True)
    b, lb = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", return_lse=True, q_descale=one, k_descale=one,
                            v_descale=one)
    assert torch.equal(bits(a), bits(b)) and torch.equal(bits(la), bits(lb))


@pytest.mark.parametrize("dims", FP8_DIMS)
def test_fp8_cuda_graph_reads_descales_on_device(dims):
    q8, k8, v8 = (x.to(FP8) for x in inputs(1, 8, 2, 2048, *dims))
    dq, dk, dv = (torch.ones(1, device="cuda") for _ in range(3))
    f = lambda: flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=dq, k_descale=dk, v_descale=dv)
    ref = f()   # the first call for a shape builds its job table, outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = f()
    graph.replay()
    assert torch.equal(bits(out), bits(ref))
    dv.fill_(2.0)
    dq.fill_(0.5)
    graph.replay()
    eager = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=0.5, v_descale=2.0)
    assert (out.float() - eager.float()).abs().max().item() <= 2 ** -6 * eager.float().abs().max().item()
    assert (out.float() - 2 * ref.float()).abs().max().item() > 0.01   # the new q_descale changed the softmax


@pytest.mark.parametrize("b,hkv", [(2, 1), (1, 2), (3, 2)])
def test_fp8_expanded_descale_shapes(b, hkv):
    q8, k8, v8 = (x.to(FP8) for x in inputs(b, 8, hkv, 512, 128, 128))
    ref = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=0.5, v_descale=2.0)
    for one in (torch.full((1,), 0.5, device="cuda"), torch.full((1, 1), 0.5, device="cuda")):
        out = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=one.expand(b, hkv),
                              v_descale=(one * 4).expand(b, hkv))
        assert torch.equal(bits(out), bits(ref))


def test_fp8_rejects_bad_descales():
    q8, k8, v8 = (x.to(FP8) for x in inputs(2, 8, 2, 512, 128, 128))
    with pytest.raises(NotImplementedError):
        flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=torch.ones(2, 2, device="cuda"))
    with pytest.raises(TypeError):
        flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=torch.ones(1, device="cuda").double())
    for bad in (-1.0, 0.0, float("inf"), float("nan"), 1e300, 1e-40):
        with pytest.raises(ValueError):
            flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", v_descale=bad)
    from zyfa import _C
    with pytest.raises(RuntimeError):
        _C.fp8_d128(q8, k8, v8, 0.088, True, False, False, v_scale=float("inf"))


def test_fp8_empty_batch_with_descales():
    q8, k8, v8 = (x.to(FP8) for x in inputs(0, 8, 2, 512, 128, 128))
    d = torch.ones(1, device="cuda").expand(0, 2)
    out = flash_attn_func(q8, k8, v8, causal=True, layout="bhsd", q_descale=d, k_descale=d, v_descale=d)
    assert out.shape == (0, 8, 512, 128)


@pytest.mark.parametrize("dtype,dims", [(torch.bfloat16, (128, 128)), (torch.bfloat16, (192, 128)),
                                        (FP8, (64, 64)), (FP8, (128, 128)), (FP8, (192, 128))])
@pytest.mark.parametrize("s", [1, 100, 300, 768, 1280])
def test_any_seqlen(dtype, dims, s):
    q, k, v = (x.to(dtype) for x in inputs(2, 8, 2, s, *dims))
    out, lse = flash_attn_func(q, k, v, causal=True, layout="bhsd", return_lse=True)
    ref, ref_lse = reference(q, k, v, 1 / math.sqrt(dims[0]))
    assert out.shape == (2, 8, s, dims[1]) and lse.shape == (2, 8, s) and out.is_contiguous()
    assert rel_rms(out, ref) < (5e-3 if dtype == torch.bfloat16 else 5e-2)
    assert (lse.double() - ref_lse).abs().max().item() < (1e-4 if dtype == torch.bfloat16 else 6e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_seqlen_zero(dtype):
    q, k, v = (x.to(dtype) for x in inputs(2, 8, 2, 0, 128, 128))
    out, lse = flash_attn_func(q, k, v, causal=True, layout="bhsd", return_lse=True)
    assert out.shape == (2, 8, 0, 128) and lse.shape == (2, 8, 0)


@pytest.mark.parametrize("dtype,dims", [(torch.bfloat16, (128, 128)), (torch.bfloat16, (192, 128)),
                                        (FP8, (64, 64)), (FP8, (128, 128)), (FP8, (192, 128))])
def test_deterministic(dtype, dims):
    q, k, v = (x.to(dtype) for x in inputs(1, 8, 2, 4096, *dims))
    a = flash_attn_func(q, k, v, causal=True, layout="bhsd")
    b = flash_attn_func(q, k, v, causal=True, layout="bhsd")
    assert torch.equal(bits(a), bits(b))


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_noncontiguous(dtype):
    q, k, v = (x.to(dtype) for x in inputs(1, 8, 2, 1024, 128, 128))
    qkv = torch.cat([q, k, v], 1).transpose(1, 2)   # (b, s, 12, d): q, k, v are strided views
    qs, ks, vs = qkv[:, :, :8], qkv[:, :, 8:10], qkv[:, :, 10:]
    a = flash_attn_func(qs, ks, vs, causal=True)
    b = flash_attn_func(qs.contiguous(), ks.contiguous(), vs.contiguous(), causal=True)
    assert torch.equal(bits(a), bits(b))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_second_gpu():
    q, k, v = (x.bfloat16() for x in inputs(1, 8, 2, 1024, 128, 128))
    q8, k8, v8 = (x.to(FP8) for x in (q, k, v))
    for x in ((q, k, v), (q8, k8, v8)):
        a = flash_attn_func(*x, causal=True, layout="bhsd")
        b = flash_attn_func(*(t.to("cuda:1") for t in x), causal=True, layout="bhsd")
        assert b.device == torch.device("cuda:1") and torch.equal(bits(a), bits(b.cpu().cuda()))


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_torch_compile(dtype):
    q, k, v = (x.to(dtype).transpose(1, 2).contiguous() for x in inputs(1, 8, 2, 1000, 128, 128))
    kw = dict(q_descale=torch.full((), 0.5, device="cuda"), v_descale=2.0) if dtype == FP8 else {}
    f = lambda q, k, v: flash_attn_func(q, k, v, causal=True, return_lse=True, **kw)
    a, la = f(q, k, v)
    b, lb = torch.compile(f, fullgraph=True)(q, k, v)
    assert torch.equal(bits(a), bits(b)) and torch.equal(bits(la), bits(lb))


def test_opcheck():
    q, k, v = (x.bfloat16().transpose(1, 2).contiguous() for x in inputs(1, 4, 2, 256, 128, 128))
    torch.library.opcheck(torch.ops.zyfa.fwd.default, (q, k, v, 0.088, 1.0, True, True, None, None, None))
    q8, k8, v8 = (x.to(FP8) for x in (q, k, v))
    d = torch.ones(1, device="cuda")
    torch.library.opcheck(torch.ops.zyfa.fwd.default, (q8, k8, v8, 0.088, 1.0, False, True, d, d, d),
                          test_utils="test_faketensor")   # opcheck's schema check multiplies inputs, which float8 lacks


@pytest.mark.parametrize("compiled", [False, True])
def test_forward_with_grad_backward_raises(compiled):
    q, k, v = (x.bfloat16().requires_grad_() for x in inputs(1, 8, 2, 512, 128, 128))
    f = lambda q, k, v: flash_attn_func(q, k, v, causal=True, layout="bhsd")
    out = (torch.compile(f, fullgraph=True) if compiled else f)(q, k, v)
    with torch.no_grad():
        ref = f(q, k, v)
    assert out.requires_grad and torch.equal(bits(out.detach()), bits(ref))
    with pytest.raises(NotImplementedError):
        out.sum().backward()


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_packed_and_varlen(dtype):
    b, s, h, hk = 2, 512, 8, 2
    q, k, v = (x.to(dtype).transpose(1, 2).contiguous() for x in inputs(b, h, hk, s, 128, 128))
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    raw, back = ((lambda x: x.view(torch.uint8)), (lambda x: x.view(FP8))) if dtype == FP8 else ((lambda x: x),) * 2
    qkv = back(torch.stack([raw(q), *(raw(x).repeat_interleave(h // hk, 2) for x in (k, v))], 2))
    full = flash_attn_func(*qkv.unbind(2), causal=True)
    assert torch.equal(bits(flash_attn_qkvpacked_func(qkv, causal=True)), bits(full))
    kv = back(torch.stack([raw(k), raw(v)], 2))
    assert torch.equal(bits(flash_attn_kvpacked_func(q, kv, causal=True)), bits(out))
    cu = torch.arange(0, (b + 1) * s, s, device="cuda", dtype=torch.int32)
    vo, vl = flash_attn_varlen_func(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), cu, cu, s, s, causal=True,
                                    return_lse=True)
    assert torch.equal(bits(vo), bits(out.flatten(0, 1)))
    assert torch.equal(bits(vl), bits(lse.transpose(0, 1).reshape(h, b * s)))
    with pytest.raises(NotImplementedError):
        flash_attn_varlen_func(q.flatten(0, 1)[:-1], k.flatten(0, 1)[:-1], v.flatten(0, 1)[:-1], cu, cu, s, s,
                               causal=True)


def test_flash_attn_api():
    q, k, v = (x.bfloat16().transpose(1, 2).contiguous() for x in inputs(1, 8, 2, 512, 128, 128))
    out = flash_attn_func(q, k, v, 0.0, None, True)
    out2, lse, probs = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    assert torch.equal(out, out2) and lse.shape == (1, 8, 512) and probs is None
    for ws in [(-1, 0), (None, None), [None, 0]]:
        assert torch.equal(flash_attn_func(q, k, v, causal=True, window_size=ws), out)
    with pytest.raises(TypeError):   # flash-attn 3 positional order
        flash_attn_func(q, k, v, None, True)


def test_rejects_unsupported():
    q, k, v = (x.bfloat16() for x in inputs(1, 8, 8, 512, 128, 128))
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, causal=False, layout="bhsd")
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k, v, causal=True, window_size=(128, 0), layout="bhsd")
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, k[:, :, :256], v[:, :, :256], causal=True, layout="bhsd")
    with pytest.raises(ValueError):
        flash_attn_func(q[0], k[0], v[0], causal=True)
    with pytest.raises(ValueError):
        flash_attn_func(q, k[:, :0], v[:, :0], causal=True, layout="bhsd")
    with pytest.raises(ValueError):
        flash_attn_func(q, k, v, causal=True, layout="bhsd", q_descale=1.0)
    with pytest.raises(TypeError):
        flash_attn_func(q.half(), k.half(), v.half(), causal=True, layout="bhsd")
