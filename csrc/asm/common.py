"""Shared parts of the FP8 attention kernel generators (d128.py, d192.py): register-name helpers, the instruction
emitter with its gfx950 hazard checks, the in-order LDS read queue, and the code-object descriptor."""
import argparse
import pathlib
import re
import struct
import sys

KARG_BYTES = 128
NSGPR = 96
RESCALE_THR = 0.125   # O and l are rescaled only when the row max (log2 units) grows by more than this


def s(i):
    return f"s{i}"


def ss(i, n):
    return f"s[{i}:{i + n - 1}]"


def v(i):
    return f"v{i}"


def vv(i, n):
    return f"v[{i}:{i + n - 1}]"


def rescale_vote(dmax):
    """vcc = RESCALE_THR < dmax (lanes whose row max grew past the threshold)."""
    thr = struct.unpack("<I", struct.pack("<f", RESCALE_THR))[0]
    return f"v_cmp_lt_f32_e32 vcc, 0x{thr:x}, {v(dmax)}"


# ---------------------------------------------------------------- instruction model
MFMA_PASSES = {"v_mfma_f32_32x32x64_f8f6f4": 16, "v_mfma_scale_f32_32x32x64_f8f6f4": 16,
               "v_mfma_f32_16x16x128_f8f6f4": 8}
TRANS_OPS = ("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt", "v_sin", "v_cos")
VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")
SREG = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b|\b(vcc)\b|\b(m0)\b")


def sregs_of(tok):
    out = []
    for m in SREG.finditer(tok):
        if m.group(4):
            out.append("vcc")
        elif m.group(5):
            out.append("m0")
        elif m.group(3) is not None:
            out.append(int(m.group(3)))
        else:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def regs_of(tok):
    out = []
    for m in VREG.finditer(tok):
        if m.group(3) is not None:
            out.append(int(m.group(3)))
        else:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


class Gen:
    """Emits instructions and pads with s_nop where a hazard window is still open (hand-written assembly gets no
    hazard recognizer). Windows, in wait states counted from the producing instruction (s_nop n = n + 1):
    MFMA result -> VALU access (20 after a 16-pass MFMA, 12 after an 8-pass one) or MFMA source (20); VALU write ->
    MFMA or permlane read (2); VALU write of an in-flight MFMA's srcC (15); transcendental -> dependent VALU (2);
    VALU write -> readlane/readfirstlane (4); VALU-written SGPR -> any read (4); SALU M0 write -> LDS DMA (1).
    A later MFMA issue from the same wave covers an earlier MFMA's window except for a residual 4."""

    def __init__(self):
        self.lines = []
        self.ws = 0          # running wait-state counter
        self.mfma_w = {}     # vgpr -> (ws at MFMA issue, required) for MFMA destinations
        self.mfma_c = {}     # srcC vgpr of the in-flight MFMA -> (ws at issue, required)
        self.valu_w = {}     # vgpr -> ws at VALU write
        self.valu_s = {}     # sgpr written by a VALU -> ws
        self.trans_w = {}    # vgpr written by a transcendental -> ws
        self.m0_w = -10      # ws after the last SALU write of M0
        self.pads = []       # hazard paddings inserted (for the build report)

    def label(self, name):
        # The hazard state runs straight through labels; branch joins are handled with snapshot/merge or nops.
        self.lines.append(f"{name}:")

    def nop(self, n):
        while n > 0:
            k = min(n, 16)
            self.lines.append(f"  s_nop {k - 1}")
            self.ws += k
            n -= k

    def snapshot(self):
        return (self.ws, dict(self.mfma_w), dict(self.mfma_c), dict(self.valu_w), dict(self.valu_s),
                dict(self.trans_w), self.m0_w)

    def merge(self, snap):
        """A label also reached by a branch taken at snapshot `snap`: the taken path skips everything emitted since, so
        its windows are shorter by the skipped wait states. Keep, per register, the window that needs the most padding
        (the straight-line state, or the snapshot's shifted forward by the skipped wait states). Windows already closed
        at the branch stay closed."""
        w0 = snap[0]
        d = self.ws - w0
        if d <= 0:
            return
        for cur, old in ((self.mfma_w, snap[1]), (self.mfma_c, snap[2])):
            for r, (t, q) in old.items():
                if w0 - t < q and (r not in cur or q - (w0 - t) > cur[r][1] - (self.ws - cur[r][0])):
                    cur[r] = (t + d, q)
        for cur, old in ((self.valu_w, snap[3]), (self.valu_s, snap[4]), (self.trans_w, snap[5])):
            for r, t in old.items():
                if w0 - t < 4:
                    cur[r] = max(cur.get(r, t + d), t + d)
        self.m0_w = max(self.m0_w, snap[6] + d)

    def reset_hazards(self):
        """After a barrier: all earlier MFMA and VALU results have retired."""
        self.mfma_w.clear()
        self.mfma_c.clear()
        self.valu_w.clear()

    def _need(self, req_ws, since):
        return max(0, req_ws - (self.ws - since))

    def _pad(self, n, msg):
        self.pads.append(msg)
        self.nop(n)

    def emit(self, text):
        op = text.split()[0]
        ops = text[len(op):].strip()
        toks = [t.strip() for t in ops.split(",")] if ops else []
        # VALU-written SGPR -> any later read
        rd_s = [r for t in (toks[1:] if op.startswith(("v_", "s_")) else toks) for r in sregs_of(t)]
        if op.startswith("s_cbranch_vcc"):
            rd_s = ["vcc"]
        spad = 0
        for r in rd_s:
            if r in self.valu_s:
                spad = max(spad, self._need(4, self.valu_s[r]))
        if spad and not op.startswith("s_nop"):
            self._pad(spad, f"sgpr hazard pad {spad} before: {text}")
        if op.startswith("v_mfma"):
            self._emit_mfma(op, toks, text)
        elif op.startswith("v_"):
            self._emit_valu(op, toks, text)
        else:
            # SALU write of M0 -> LDS DMA (buffer_load ... lds) reading it
            if op.startswith(("buffer_", "global_")) and text.rstrip().endswith(" lds") and self.ws - self.m0_w < 1:
                self._pad(1, f"m0 hazard pad 1 before: {text}")
            self.lines.append("  " + text)
            self.ws += int(toks[0]) + 1 if op == "s_nop" else 1
            if op.startswith("s_") and toks and toks[0] == "m0":
                self.m0_w = self.ws

    def _emit_valu(self, op, toks, text):
        dst = [] if op.startswith("v_cmp") or not toks else regs_of(toks[0])
        srcs = [r for t in toks[1:] for r in regs_of(t)]
        if op.startswith("v_permlane"):
            srcs = srcs + dst
        pad = 0
        for r in set(srcs) | set(dst):
            if r in self.mfma_w:
                since, req = self.mfma_w[r]
                pad = max(pad, self._need(req, since))
        for r in dst:
            if r in self.mfma_c:
                since, req = self.mfma_c[r]
                pad = max(pad, self._need(req, since))
        if op.startswith("v_permlane"):
            for r in srcs:
                if r in self.valu_w:
                    pad = max(pad, self._need(2, self.valu_w[r]))
        for r in srcs:
            if r in self.trans_w:
                pad = max(pad, self._need(2, self.trans_w[r]))
        if op.startswith(("v_readfirstlane", "v_readlane")):
            for r in srcs:
                if r in self.valu_w:
                    pad = max(pad, self._need(4, self.valu_w[r]))
        if pad:
            self._pad(pad, f"hazard pad {pad} before: {text}")
        self.lines.append("  " + text)
        self.ws += 1
        for r in dst:
            self.valu_w[r] = self.ws
            self.mfma_w.pop(r, None)
            self.trans_w.pop(r, None)
            if op.startswith(TRANS_OPS):
                self.trans_w[r] = self.ws
        # VCC is interlocked (v_cmp -> v_cndmask may be back to back)
        for r in (sregs_of(toks[0]) if toks else []):
            if r != "vcc":
                self.valu_s[r] = self.ws

    def _emit_mfma(self, op, toks, text):
        dst = regs_of(toks[0])
        srcs = [r for t in toks[1:] for r in regs_of(t)]
        pad = 0
        for r in srcs:
            if r in self.valu_w:
                pad = max(pad, self._need(2, self.valu_w[r]))
            if r in self.mfma_w and r not in dst:
                since, req = self.mfma_w[r]
                pad = max(pad, self._need(20, since))
        if pad:
            self._pad(pad, f"hazard pad {pad} before: {text}")
        self.lines.append("  " + text)
        self.ws += 1
        # MFMA results are written progressively: an intervening MFMA leaves a residual window of 4 from its issue
        self.mfma_w = {r: (self.ws, 4) for r, (t, q) in self.mfma_w.items() if self.ws - 1 - t < q}
        self.mfma_c = {}
        req = 20 if MFMA_PASSES[op] == 16 else 12
        for r in dst:
            self.mfma_w[r] = (self.ws, req)
        if len(toks) >= 4:
            for r in regs_of(toks[3]):
                if r not in dst:
                    self.mfma_c[r] = (self.ws, 15)

    def text(self):
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------- LDS read queue
class LdsQ:
    """In-order LDS read queue of one wave: entries (fragment id, number of reads). wait_for(f) emits the lgkmcnt
    that guarantees fragment f's reads have returned."""

    def __init__(self, g):
        self.g = g
        self.q = []

    def outstanding(self):
        return sum(n for _, n in self.q)

    def push(self, fid, n):
        self.q.append([fid, n])
        assert self.outstanding() <= 15, f"lgkm outstanding {self.outstanding()} > 15"

    def wait_for(self, fid):
        idx = None
        for i, (f, _) in enumerate(self.q):
            if f == fid:
                idx = i
        if idx is None:
            return
        self.g.emit(f"s_waitcnt lgkmcnt({sum(n for _, n in self.q[idx + 1:])})")
        self.q = self.q[idx + 1:]

    def drain(self):
        if self.q:
            self.g.emit("s_waitcnt lgkmcnt(0)")
        self.q = []


# ---------------------------------------------------------------- kernel base
class KernelBase:
    """Label allocation, emission of instruction lists with pseudo-ops, barriers."""

    def __init__(self):
        self.g = Gen()
        self.uid = 0
        self._snaps = {}

    def L(self, base):
        self.uid += 1
        return f".L{base}_{self.uid}"

    def emit_list(self, lst):
        """Emit instruction strings; one string may hold several lines. Pseudo-ops: "LABEL x"; "NOP2" (s_nop 1);
        "SNAP x" / "MERGE x" / "DROP x": save the hazard state at a branch / merge it at the branch target / drop it."""
        for t in lst:
            if "\n" in t:
                self.emit_list(t.split("\n"))
            elif t.startswith("LABEL "):
                self.g.label(t[6:])
            elif t == "NOP2":
                self.g.nop(2)
            elif t.startswith("SNAP "):
                self._snaps[t[5:]] = self.g.snapshot()
            elif t.startswith("MERGE "):
                self.g.merge(self._snaps.pop(t[6:]))
            elif t.startswith("DROP "):
                self._snaps.pop(t[5:])
            else:
                self.g.emit(t)

    def bar(self):
        self.g.emit("s_barrier")
        self.g.reset_hazards()


def kernel_text(name, body, lds_bytes):
    """Assembly file: the kernel body plus its descriptor and metadata (512 threads, 256 VGPRs, 128 B kernarg)."""
    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {name}
.p2align 8
.type {name},@function
{name}:
{body}
.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size {lds_bytes}
  .amdhsa_kernarg_size {KARG_BYTES}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr {NSGPR}
  .amdhsa_accum_offset 256
  .amdhsa_ieee_mode 0
  .amdhsa_dx10_clamp 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: {KARG_BYTES}
    .group_segment_fixed_size: {lds_bytes}
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: {NSGPR}
    .vgpr_count: 256
    .agpr_count: 0
    .max_flat_workgroup_size: 512
    .args:
      - {{ .offset: 0, .size: {KARG_BYTES}, .value_kind: by_value }}
...
.end_amdgpu_metadata
"""


def main(generate, default_out):
    """Command line of a generator: write the kernel's assembly to --out."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    a = ap.parse_args()
    text, pads = generate()
    pathlib.Path(a.out).write_text(text)
    print(f"{a.out}: {len(pads)} hazard paddings inserted", file=sys.stderr)
