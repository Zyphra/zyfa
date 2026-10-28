#!/usr/bin/env python3
"""Generates the FP8 causal forward attention kernel for MLA shapes, d_qk = 192 and d_v = 128, on gfx950
(`fmha_fp8pp192`) as hand-scheduled assembly. Same ping-pong structure as d128.py with 64-key tiles, K rows padded to
256 B in LDS, and P = exp2 via v_exp_f32 with RNE conversion to e4m3."""
from common import KernelBase, LdsQ, kernel_text, main, rescale_vote, s, ss, v, vv

# ---------------------------------------------------------------- registers
O = [0, 16, 32, 48]            # O^T accumulators (fp32, d_v rows x 32 queries)
S = [64, 80]                   # S^T blocks (32 keys x 32 queries) of the 64-key tile
P = 96                         # e4m3 P of the tile (64 keys x 32 queries), 8 VGPRs
Q = [104, 112, 120]            # Q, d 0-63 / 64-127 / 128-191
SEL = 128                      # 0/1 selector, A operand of the row-sum MFMA
FR = 136                       # LDS fragment ring: NSLOT x 8 VGPRs
NSLOT = 6
RS = 184                       # row sum of P
V_TID = 188
V_KA = [189, 190, 191]         # K LDS read addresses per 64-d chunk (ring slot 0)
V_VA = 192                     # V LDS read address (ring base + lane part)
V_TKA = [193, 194, 195]        # K LDS read addresses for a runtime ring slot
V_TVA = 196
V_GK0, V_GK1, V_GV0 = 197, 198, 199      # global -> LDS voffsets
V_M, V_L, V_MREF = 200, 201, 202         # running row max (log2 units), row sum, reference max of the coded tile
SB = [203, 204]                # E8M0 scale of P per tile, two parity sets
V_BX = 205                     # exp2 argument bias of the tile being coded
V_127 = 206                    # E8M0 scale 1.0 of the V operand
V_T0, V_T1, V_T2, V_T3 = 207, 208, 209, 210
V_THR, V_NINF, V_TMP, V_CORR = 211, 212, 213, 214   # causal mask threshold, -inf, temporary, rescale factor
V_EA, V_ELA, V_LSE = 215, 216, 217       # epilogue: O and LSE store addresses, LSE value
V_KL = [218, 219, 220]         # copies of V_KA, not read
V_M8 = 221                     # -8.0, not read

S_KARG, S_WG = 0, 2
S_QH, S_OH = 4, 6              # s[4:5], s[6:7]: Q and O bases of the pass
S_KR, S_VR = 8, 12             # K and V buffer resources s[8:11], s[12:15]
S_A = 16                       # kernarg dwords 0-31 in s[16:47]
S_TAB = 48                     # job table entry s[48:55]
S_JOB, S_W, S_G, S_QWARP = 56, 57, 58, 59                      # job, wave, wave group, query block of the wave
S_PP, S_QBLK, S_QSTART, S_N, S_FM, S_K = 60, 61, 62, 63, 64, 65  # pass, Q block, first query, tiles, first masked, k
S_KVT, S_VVT = 66, 67          # bytes per K / V tile (64 * k_ss, 64 * v_ss)
S_KMAX, S_VMAX = 68, 69        # offsets of the pass's last K / V tile
S_DONE = 70                    # no next pass
S_KSOFF, S_VSOFF = 71, 72      # global offsets of the next K and V tile loads
S_FLAG = 73                    # rescale vote
S_RMASK = 74                   # s[74:75]
S_LSE = 76                     # s[76:77]
S_T0, S_T1, S_T2, S_T3 = 78, 79, 80, 81
S_M127 = 82                    # -127.0
S_M0K, S_M0V = 83, 84          # per-wave LDS DMA bases of the K/V rings
S_U = 85                       # base + k
S_KRT, S_VRT, S_KLT, S_VLT = 86, 87, 88, 89   # runtime ring slots: K/V read bases, K/V load M0 bases
S_BASE, S_NBASE = 90, 91       # ring slot of tile 0 of this pass / of the next pass
S_EOH, S_ELSE = 92, 94         # s[92:93], s[94:95]: O / LSE bases of the pass being stored

# kernarg layout (dwords from S_A)
KA = dict(Q=0, K=2, V=4, O=6, LSE=8, TAB=10, SL2=12, NJOBS=13, NUMQ=14, QW=15, PAIR=16, Q_SS=17, K_SS=18, V_SS=19,
          O_SS=20, Q_SH2=21, O_SH2=22, L_SH2=23, NKEYS=24, GRID=25)

LDS_K = [16384 * i for i in range(4)]      # K ring slots (64 keys x 256 B)
LDS_V0 = 65536                             # V ring: 4 slots of 8 KB
LDS_BYTES = 98304                          # end of the rings; the epilogue stages O above (8 KB per wave of a group)
LDS_EPI = 32768

QK_ORDER = [(b, d) for d in range(3) for b in range(2)]   # (key block, 64-d chunk) of the 6 QK MFMAs
NPRE = 4            # K fragments read at the end of SM, ahead of the first QK MFMA
DEPTH = 5           # fragment read-ahead (MFMAs)
VALU_PER_SLOT = 8   # VALU instructions placed after each MFMA of MM (x3 while masking)
LOAD_AT = 2         # MFMA after which the tile loads of MM start


def arg(name):
    return s(S_A + KA[name])


def argh(name):
    return s(S_A + KA[name] + 1)


class Kernel(KernelBase):
    def bar_group(self, grp):
        """One extra barrier for wave group grp only."""
        g = self.g
        lab = self.L("nobar")
        g.emit(f"s_cmp_eq_u32 {s(S_G)}, {grp}")
        g.emit(f"s_cbranch_scc0 {lab}")
        self.bar()
        g.label(lab)

    # ------------------------------------------------------------ entry
    def kernel_entry(self):
        g = self.g
        g.emit(f"s_load_dwordx16 {ss(S_A, 16)}, {ss(S_KARG, 2)}, 0")
        g.emit(f"s_load_dwordx16 {ss(S_A + 16, 16)}, {ss(S_KARG, 2)}, 64")
        g.emit(f"v_mov_b32 {v(V_TID)}, v0")
        g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 6, {v(V_TID)}")
        g.emit(f"v_readfirstlane_b32 {s(S_W)}, {v(V_TMP)}")
        g.emit(f"s_lshr_b32 {s(S_G)}, {s(S_W)}, 2")
        g.emit(f"v_and_b32 {v(V_T0)}, 63, {v(V_TID)}")          # lane
        g.emit(f"v_and_b32 {v(V_T1)}, 31, {v(V_TID)}")          # c32
        g.emit(f"v_bfe_u32 {v(V_T2)}, {v(V_TID)}, 5, 1")        # gb
        # row-sum selector: (((l >> 4) ^ (l >> 2)) & 1) ? 0 : 0x38383838
        g.emit(f"v_lshrrev_b32 {v(V_T3)}, 4, {v(V_T0)}")
        g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 2, {v(V_T0)}")
        g.emit(f"v_xor_b32 {v(V_T3)}, {v(V_T3)}, {v(V_TMP)}")
        g.emit(f"v_and_b32 {v(V_T3)}, 1, {v(V_T3)}")
        g.emit(f"v_cmp_eq_u32_e32 vcc, 0, {v(V_T3)}")
        g.emit(f"v_mov_b32 {v(V_TMP)}, 0x38383838")
        g.emit(f"v_cndmask_b32_e64 {v(SEL)}, 0, {v(V_TMP)}, vcc")
        for i in range(1, 8):
            g.emit(f"v_mov_b32 {v(SEL + i)}, {v(SEL)}")
        # K lane read offsets: c32 * 256 + 32 * ((2 ds + gb + c32) & 7) (32 B rotation swizzle)
        for ds_ in range(3):
            d = V_KA[ds_]
            g.emit(f"v_add_u32 {v(d)}, {2 * ds_}, {v(V_T2)}")
            g.emit(f"v_add_u32 {v(d)}, {v(d)}, {v(V_T1)}")
            g.emit(f"v_and_b32 {v(d)}, 7, {v(d)}")
            g.emit(f"v_lshlrev_b32 {v(d)}, 5, {v(d)}")
            g.emit(f"v_lshl_add_u32 {v(d)}, {v(V_T1)}, 8, {v(d)}")
            g.emit(f"v_mov_b32 {v(V_KL[ds_])}, {v(d)}")
        # V lane read offset: V ring + (gb * 32 + (c32 >> 4)) * 128 + (c32 & 15) * 8
        g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 4, {v(V_T1)}")
        g.emit(f"v_lshl_add_u32 {v(V_VA)}, {v(V_T2)}, 5, {v(V_TMP)}")
        g.emit(f"v_lshlrev_b32 {v(V_VA)}, 7, {v(V_VA)}")
        g.emit(f"v_and_b32 {v(V_TMP)}, 15, {v(V_T1)}")
        g.emit(f"v_lshl_add_u32 {v(V_VA)}, {v(V_TMP)}, 3, {v(V_VA)}")
        g.emit(f"v_add_u32 {v(V_VA)}, {LDS_V0}, {v(V_VA)}")
        g.emit("s_waitcnt lgkmcnt(0)")
        g.emit(f"v_mov_b32 {v(V_127)}, 0x7f")
        g.emit(f"v_mov_b32 {v(V_NINF)}, 0xff800000")
        g.emit(f"v_mov_b32 {v(V_M8)}, 0xc1000000")
        g.emit(f"s_mov_b32 {s(S_M127)}, 0xc2fe0000")
        g.emit(f"s_lshl_b32 {s(S_KVT)}, {arg('K_SS')}, 6")
        g.emit(f"s_lshl_b32 {s(S_VVT)}, {arg('V_SS')}, 6")
        # K G2L voffsets, pass p: c = tid + 512 p, key = c >> 4, sl = c & 15, L = ((sl >> 1) - key) & 7,
        # chunk pair = L < 6 ? L : 0 (the two pad slots hold copies), voff = key * k_ss + 32 pair + 16 (sl & 1)
        for p_, dst in ((0, V_GK0), (1, V_GK1)):
            g.emit(f"v_add_u32 {v(V_T3)}, {512 * p_}, {v(V_TID)}")
            g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 4, {v(V_T3)}")
            g.emit(f"v_bfe_u32 {v(dst)}, {v(V_T3)}, 1, 3")
            g.emit(f"v_sub_u32 {v(dst)}, {v(dst)}, {v(V_TMP)}")
            g.emit(f"v_and_b32 {v(dst)}, 7, {v(dst)}")
            g.emit(f"v_cmp_gt_u32_e32 vcc, 6, {v(dst)}")
            g.emit(f"v_cndmask_b32_e64 {v(dst)}, 0, {v(dst)}, vcc")
            g.emit(f"v_lshlrev_b32 {v(dst)}, 5, {v(dst)}")
            g.emit(f"v_bfe_u32 {v(V_T0)}, {v(V_T3)}, 0, 1")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_T0)}, 4, {v(dst)}")
            g.emit(f"v_mad_u32_u24 {v(dst)}, {v(V_TMP)}, {arg('K_SS')}, {v(dst)}")
        # V G2L voffset (one pass): t = tid & 7, dg = (tid >> 3) & 7, half = bit 6, nk = bit 7, gt = bit 8;
        # key = 32 nk + 4 gt + 16 half + 8 (t >> 2) + (t & 3), voff = key * v_ss + 16 dg
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 7, 1")
        g.emit(f"v_lshlrev_b32 {v(V_GV0)}, 5, {v(V_TMP)}")
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 8, 1")
        g.emit(f"v_lshl_add_u32 {v(V_GV0)}, {v(V_TMP)}, 2, {v(V_GV0)}")
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 6, 1")
        g.emit(f"v_lshl_add_u32 {v(V_GV0)}, {v(V_TMP)}, 4, {v(V_GV0)}")
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 2, 1")
        g.emit(f"v_lshl_add_u32 {v(V_GV0)}, {v(V_TMP)}, 3, {v(V_GV0)}")
        g.emit(f"v_and_b32 {v(V_TMP)}, 3, {v(V_TID)}")
        g.emit(f"v_add_u32 {v(V_GV0)}, {v(V_GV0)}, {v(V_TMP)}")
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 3, 3")
        g.emit(f"v_lshlrev_b32 {v(V_TMP)}, 4, {v(V_TMP)}")
        g.emit(f"v_mad_u32_u24 {v(V_GV0)}, {v(V_GV0)}, {arg('V_SS')}, {v(V_TMP)}")
        g.emit(f"s_mov_b32 {s(S_JOB)}, {s(S_WG)}")
        g.emit(f"s_lshl_b32 {s(S_M0K)}, {s(S_W)}, 10")
        g.emit(f"s_add_u32 {s(S_M0V)}, {s(S_M0K)}, {LDS_V0}")

    # ------------------------------------------------------------ fragments
    def read_kfrag(self, slot, blk, ds_, kb):
        """K fragment (key block blk, 64-d chunk ds_) of ring slot kb (None: runtime slot) into ring register slot."""
        if kb is None:
            base, off = V_TKA[ds_], 8192 * blk
        else:
            base, off = V_KA[ds_], 16384 * kb + 8192 * blk
        r = FR + 8 * slot
        self.g.emit(f"ds_read_b128 {vv(r, 4)}, {v(base)} offset:{off}")
        self.g.emit(f"ds_read_b128 {vv(r + 4, 4)}, {v(base)} offset:{off + 16}")

    def read_vfrag(self, slot, j, vb):
        r = FR + 8 * slot
        base, b0 = (V_TVA, 0) if vb is None else (V_VA, 8192 * vb)
        for i in range(4):
            self.g.emit(f"ds_read_b64_tr_b8 {vv(r + 2 * i, 2)}, {v(base)} offset:{b0 + 256 * j + 1024 * i}")

    def pre_read_k(self, kb):
        for i in range(NPRE):
            b, d = QK_ORDER[i]
            self.read_kfrag(i % NSLOT, b, d, kb)

    # ------------------------------------------------------------ softmax pieces
    def stats_instrs(self, first, wpar):
        """Statistics of S(k+1): row max, rescale vote, E8M0 offset of the tile (written to SB[wpar]), exp2 bias.
        Returns the per-block max chains (each needs its S block final) and the tail."""
        sl2 = arg("SL2")
        groups = []
        for base, a in ((S[0], V_T0), (S[1], V_T1)):
            lst = [f"v_max_f32 {v(a)}, {v(base)}, {v(base + 1)}"]
            for r in range(2, 16, 2):
                lst.append(f"v_max3_f32 {v(a)}, {v(base + r)}, {v(base + r + 1)}, {v(a)}")
            groups.append(lst)
        tail = [f"v_max_f32 {v(V_T0)}, {v(V_T0)}, {v(V_T1)}",
                f"v_mov_b32 {v(V_T1)}, {v(V_T0)}",
                "NOP2",
                f"v_permlane32_swap_b32 {v(V_T0)}, {v(V_T1)}",
                f"v_max_f32 {v(V_T0)}, {v(V_T0)}, {v(V_T1)}",     # raw max of the query's 64 keys
                f"v_mul_f32 {v(V_T1)}, {sl2}, {v(V_T0)}"]         # tile max (log2 units)
        if first:
            tail += [f"v_mov_b32 {v(V_M)}, {v(V_T1)}", f"v_mov_b32 {v(V_MREF)}, {v(V_T1)}"]
        else:
            tail += [f"v_max_f32 {v(V_T3)}, {v(V_M)}, {v(V_T1)}",   # mnew
                     f"v_sub_f32 {v(V_T1)}, {v(V_T3)}, {v(V_M)}",   # dmax
                     rescale_vote(V_T1),
                     "s_cmp_lg_u64 vcc, 0",
                     f"s_cselect_b64 {ss(S_RMASK, 2)}, -1, 0",
                     f"s_cselect_b32 {s(S_FLAG)}, 1, 0",
                     f"v_cndmask_b32_e64 {v(V_MREF)}, {v(V_M)}, {v(V_T3)}, {ss(S_RMASK, 2)}"]
        sbn = SB[wpar]
        # off = floor(tile max - mref) in [-127, 0] (the vote keeps P within e4m3 range); bx = 6 - (mref + off)
        tail += [f"v_fma_f32 {v(V_T0)}, {v(V_T0)}, {sl2}, -{v(V_MREF)}",
                 f"v_floor_f32 {v(V_T0)}, {v(V_T0)}",
                 f"v_med3_f32 {v(V_T0)}, {v(V_T0)}, {s(S_M127)}, 0",
                 f"v_cvt_i32_f32 {v(sbn)}, {v(V_T0)}",
                 f"v_add_u32_e32 {v(sbn)}, 0x7f, {v(sbn)}",
                 f"v_add_f32_e32 {v(V_T1)}, {v(V_MREF)}, {v(V_T0)}",
                 f"v_sub_f32_e32 {v(V_BX)}, 0x40c00000, {v(V_T1)}"]
        return groups, tail

    def mask_thr_instrs(self):
        """S_T3 = q_start - 64 t for the tile t = k + 1 being masked; V_THR = S_T3 + c32 - 4 gb."""
        return [f"s_add_u32 {s(S_T3)}, {s(S_K)}, 1",
                f"s_lshl_b32 {s(S_T3)}, {s(S_T3)}, 6",
                f"s_sub_u32 {s(S_T3)}, {s(S_QSTART)}, {s(S_T3)}",
                f"v_and_b32 {v(V_TMP)}, 31, {v(V_TID)}",
                f"v_add_u32 {v(V_THR)}, {s(S_T3)}, {v(V_TMP)}",
                f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 5, 1",
                f"v_lshlrev_b32 {v(V_TMP)}, 2, {v(V_TMP)}",
                f"v_sub_u32 {v(V_THR)}, {v(V_THR)}, {v(V_TMP)}"]

    def mask_instrs(self, blk):
        """Causal mask of S block blk (keys 32 blk .. +31 of the tile vs the wave's 32 queries): nothing when fully
        visible, -inf when fully masked, elementwise (key > query) on the diagonal. Returned as one unit, so the
        branches stay within one MFMA slot."""
        lv, lf, le = self.L("mvis"), self.L("mfull"), self.L("mend")
        out = [f"s_sub_i32 {s(S_T2)}, {s(S_T3)}, {32 * blk}",
               f"s_cmp_ge_i32 {s(S_T2)}, 32", f"s_cbranch_scc1 {lv}", "SNAP vis",
               f"s_cmp_lt_i32 {s(S_T2)}, 0", f"s_cbranch_scc1 {lf}"]
        for j4 in range(4):
            for i in range(4):
                e = 32 * blk + 8 * j4 + i
                r = S[blk] + 4 * j4 + i
                out.append(f"v_cmp_gt_i32_e32 vcc, {e}, {v(V_THR)}")
                out.append(f"v_cndmask_b32_e64 {v(r)}, {v(r)}, {v(V_NINF)}, vcc")
        # the hazard counter ran through the elementwise path: pad the -inf writes for the QK MFMA that wrote S_blk
        out += [f"s_branch {le}", f"LABEL {lf}", "s_nop 15", "s_nop 3"]
        out += [f"v_mov_b32 {v(S[blk] + r)}, {v(V_NINF)}" for r in range(16)]
        if blk == 1:
            # the fully visible path reaches the S reads of the statistics soon after the QK MFMA: pad it
            lj = self.L("mjoin")
            out += [f"s_branch {lj}", f"LABEL {lv}", "s_nop 15", "s_nop 3", f"LABEL {lj}", f"LABEL {le}",
                    "DROP vis", "NOP2"]
        else:
            # the fully visible path skips the elementwise path: count its windows from the branch
            out += [f"LABEL {le}", f"LABEL {lv}", "MERGE vis", "NOP2"]
        return ["\n".join(out)]

    def codes(self):
        """P = exp2(x) in place, x = s * sl2 + bx computed at the end of the previous MM; P dword j of block b =
        cvt_pk_fp8(p[4j], p[4j+1]) | cvt_pk_fp8(p[4j+2], p[4j+3]) << 16 (RNE). Interleaved: the exps of block b
        (4 behind) with the conversions of block b - 1."""
        g = self.g
        work = []
        for blk in range(2):
            e = [f"v_exp_f32 {v(S[blk] + r)}, {v(S[blk] + r)}" for r in range(16)]
            c = []
            for j in range(4):
                r0 = S[blk] + 4 * j
                d = P + 4 * blk + j
                c.append(f"v_cvt_pk_fp8_f32 {v(d)}, {v(r0)}, {v(r0 + 1)}")
                c.append(f"v_cvt_pk_fp8_f32 {v(d)}, {v(r0 + 2)}, {v(r0 + 3)} op_sel:[0,0,1]")
            work.append((e, c))
        for blk in range(2):
            e = work[blk][0]
            prev = work[blk - 1][1] if blk else []
            for i in range(16):
                if i >= 4:
                    g.emit(e[i - 4])
                if i < len(prev):
                    g.emit(prev[i])
            for i in range(12, 16):
                g.emit(e[i])
        for t in work[1][1]:
            g.emit(t)

    # ------------------------------------------------------------ phases
    def seg_sm(self, kb_next):
        """SM(k): P(k), then the first K(k+1) fragment reads from ring slot kb_next (None: runtime slot, False: no next
        tile)."""
        self.g.emit("s_setprio 0")
        self.codes()
        if kb_next is not False:
            self.pre_read_k(kb_next)

    def loads_static(self, kslot, vslot):
        """K(k+4) -> ring slot kslot, V(k+2) -> vslot, offsets clamped to the pass's last tile: 3 instruction groups."""
        grp = []
        for p_, voff in enumerate((V_GK0, V_GK1)):
            pre = [f"s_min_u32 {s(S_T0)}, {s(S_KSOFF)}, {s(S_KMAX)}"] if p_ == 0 else []
            grp.append(pre + [f"s_add_u32 m0, {s(S_M0K)}, {LDS_K[kslot] + 8192 * p_}",
                              f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T0)} offen lds"])
        grp.append([f"s_min_u32 {s(S_T1)}, {s(S_VSOFF)}, {s(S_VMAX)}",
                    f"s_add_u32 m0, {s(S_M0V)}, {8192 * vslot}",
                    f"buffer_load_dwordx4 {v(V_GV0)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds",
                    f"s_add_u32 {s(S_KSOFF)}, {s(S_KSOFF)}, {s(S_KVT)}",
                    f"s_add_u32 {s(S_VSOFF)}, {s(S_VSOFF)}, {s(S_VVT)}"])
        return grp

    def loads_rt(self):
        """K(k+4) and V(k+2) into the runtime slots, only when inside the pass: 2 instruction groups."""
        lk, lv = self.L("tk"), self.L("tv")
        k = [f"s_add_u32 {s(S_T0)}, {s(S_K)}, 4", f"s_cmp_ge_u32 {s(S_T0)}, {s(S_N)}", f"s_cbranch_scc1 {lk}",
             f"s_mul_i32 {s(S_T0)}, {s(S_T0)}, {s(S_KVT)}"]
        for p_, voff in enumerate((V_GK0, V_GK1)):
            k += [f"s_add_u32 m0, {s(S_KLT)}, {8192 * p_}",
                  f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T0)} offen lds"]
        k += [f"LABEL {lk}"]
        vl = [f"s_add_u32 {s(S_T1)}, {s(S_K)}, 2", f"s_cmp_ge_u32 {s(S_T1)}, {s(S_N)}", f"s_cbranch_scc1 {lv}",
              f"s_mul_i32 {s(S_T1)}, {s(S_T1)}, {s(S_VVT)}",
              f"s_mov_b32 m0, {s(S_VLT)}",
              f"buffer_load_dwordx4 {v(V_GV0)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds", f"LABEL {lv}"]
        return [k, vl]

    def seg_mm(self, variant, kb, vb, par, loads):
        """MM(k): row sum of P(k), QK(k+1), PV(k), and under the MFMAs the statistics of S(k+1).
        variant: qk0 / qk0m (the pass's first QK, unmasked / masked), mm / mmm (unmasked / masked S(k+1)), pvo (PV(k)
        only: tile k+1 has no visible key), last. kb / vb: ring slots of K(k+1) / V(k) (None: runtime slots).
        par: scale parity set of tile k (the statistics write the other). loads: (kslot, vslot), "rt" or None."""
        g = self.g
        g.emit("s_setprio 1")
        if loads == "rt":
            g.emit("s_waitcnt vmcnt(0)")
        with_qk = variant not in ("last", "pvo")
        with_pv = variant not in ("qk0", "qk0m")
        masked = variant in ("qk0m", "mmm")
        first = variant in ("qk0", "qk0m")
        frags = ([("K", b, d) for b, d in QK_ORDER] if with_qk else []) + \
                ([("V", j) for j in range(4)] if with_pv else [])
        lq = LdsQ(g)
        nxt = min(NPRE, len(frags)) if with_qk else 0    # the first K fragments were read at the end of SM
        for i in range(nxt):
            lq.push(i, 2)

        def refill(upto):
            nonlocal nxt
            while nxt < len(frags) and nxt <= upto and lq.outstanding() + (2 if frags[nxt][0] == "K" else 4) <= 15:
                f = frags[nxt]
                if f[0] == "K":
                    self.read_kfrag(nxt % NSLOT, f[1], f[2], kb)
                    lq.push(nxt, 2)
                else:
                    self.read_vfrag(nxt % NSLOT, f[1], vb)
                    lq.push(nxt, 4)
                nxt += 1

        mf = [("R",)] if with_pv else []
        if with_qk:
            mf += [("QK", b, d, i) for i, (b, d) in enumerate(QK_ORDER)]
        if with_pv:
            f0 = len(QK_ORDER) if with_qk else 0
            mf += [("PV", j, f0 + j) for j in range(4)]
        # VALU plan: block b's statistics go after the MFMA that follows its last QK, at most `cap` per MFMA slot
        valu_slots = [[] for _ in mf]
        flat = []
        if with_qk:
            groups, tail = self.stats_instrs(first, 1 - par)
            idx_last = {b: max(i for i, m in enumerate(mf) if m[0] == "QK" and m[1] == b) for b in range(2)}
            if masked:
                flat += [(0, t) for t in self.mask_thr_instrs()]
            for bi, lst in enumerate(groups):
                flat += [(idx_last[bi] + 1, t) for t in (self.mask_instrs(bi) if masked else []) + lst]
            flat += [(0, t) for t in tail]
        last_slot = len(mf) - 1
        cap = VALU_PER_SLOT * (3 if masked else 1)
        p_ = 0
        for e, t in flat:
            p_ = max(p_, min(e, last_slot))
            while p_ < last_slot and sum(1 for x in valu_slots[p_] if x != "NOP2") >= cap:
                p_ += 1
            valu_slots[p_].append(t)
        if with_pv:
            # l += rowsum * 2^off
            tgt = min(len(mf) - 1, 2)
            valu_slots[tgt] = [f"v_lshlrev_b32 {v(V_TMP)}, 23, {v(SB[par])}",
                               f"v_fma_f32 {v(V_L)}, {v(RS)}, {v(V_TMP)}, {v(V_L)}"] + valu_slots[tgt]
        lg = None
        if loads is not None:
            lg = self.loads_rt() if loads == "rt" else self.loads_static(*loads)
        for i, m in enumerate(mf):
            if m[0] == "R":
                g.emit(f"v_mfma_f32_16x16x128_f8f6f4 {vv(RS, 4)}, {vv(SEL, 8)}, {vv(P, 8)}, 0")
                refill(DEPTH - 1)
            else:
                f = m[-1]
                # even fragment: wait for the pair (f, f + 1) if f + 1 is already in flight
                if f % 2 == 0 and f + 1 < len(frags) and any(x[0] == f + 1 for x in lq.q):
                    lq.wait_for(f + 1)
                else:
                    lq.wait_for(f)
                a = vv(FR + 8 * (f % NSLOT), 8)
                if m[0] == "QK":
                    b, d = m[1], m[2]
                    sd = vv(S[b], 16)
                    acc = "0" if d == 0 else sd
                    g.emit(f"v_mfma_f32_32x32x64_f8f6f4 {sd}, {a}, {vv(Q[d], 8)}, {acc}")
                else:
                    od = vv(O[m[1]], 16)
                    g.emit(f"v_mfma_scale_f32_32x32x64_f8f6f4 {od}, {a}, {vv(P, 8)}, {od}, {v(V_127)}, {v(SB[par])} "
                           f"op_sel_hi:[0,0,0]")
                refill(f + DEPTH)
            if lg is not None and 0 <= i - LOAD_AT < len(lg):
                self.emit_list(lg[i - LOAD_AT])
            self.emit_list(valu_slots[i])
        assert nxt == len(frags), (nxt, len(frags))
        lq.drain()
        if with_qk:
            # exp2 arguments of S(k+1) as packed fmas, after the last MFMA: v_pk_fma_f32 cannot run beside an MFMA
            g.emit(f"v_mov_b32 {v(V_T1)}, {v(V_BX)}")
            assert (S_A + KA["SL2"]) % 2 == 0 and V_T1 % 2 == 0
            for blk in range(2):
                for r in range(0, 16, 2):
                    g.emit(f"v_pk_fma_f32 {vv(S[blk] + r, 2)}, {vv(S[blk] + r, 2)}, {ss(S_A + KA['SL2'], 2)}, "
                           f"{vv(V_T1, 2)} op_sel:[0,0,0] op_sel_hi:[1,0,0]")
        if with_qk and not first:
            # rescale voted by this MM's statistics: O *= corr as packed muls (the other wave is in SM: no MFMAs)
            lskip = self.L("norescale")
            g.emit(f"s_cmp_eq_u32 {s(S_FLAG)}, 0")
            g.emit(f"s_cbranch_scc1 {lskip}")
            g.emit(f"v_sub_f32 {v(V_CORR)}, {v(V_M)}, {v(V_MREF)}")
            g.emit(f"v_exp_f32 {v(V_CORR)}, {v(V_CORR)}")
            g.emit(f"v_mov_b32 {v(V_M)}, {v(V_MREF)}")
            assert V_CORR % 2 == 0
            for j in range(4):
                for r in range(0, 16, 2):
                    g.emit(f"v_pk_mul_f32 {vv(O[j] + r, 2)}, {vv(O[j] + r, 2)}, {vv(V_CORR, 2)} "
                           f"op_sel:[0,0] op_sel_hi:[1,0]")
            g.emit(f"v_mul_f32 {v(V_L)}, {v(V_CORR)}, {v(V_L)}")
            g.label(lskip)
        if loads not in (None, "rt"):
            g.emit("s_waitcnt vmcnt(3)")

    # ------------------------------------------------------------ pass-level
    def tail_state(self):
        """Runtime LDS bases for tile k (S_K) in ring slot u = (S_BASE + k) & 3: read K(k+1) and V(k), load K(k+4)
        into slot u and V(k+2) into slot u + 2."""
        g = self.g
        g.emit(f"s_add_u32 {s(S_U)}, {s(S_K)}, {s(S_BASE)}")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_U)}, 1")
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_T0)}, 3")
        g.emit(f"s_lshl_b32 {s(S_KRT)}, {s(S_T0)}, 14")
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_U)}, 3")
        g.emit(f"s_lshl_b32 {s(S_T1)}, {s(S_T0)}, 14")
        g.emit(f"s_add_u32 {s(S_KLT)}, {s(S_M0K)}, {s(S_T1)}")
        g.emit(f"s_lshl_b32 {s(S_VRT)}, {s(S_T0)}, 13")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_U)}, 2")
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_T0)}, 3")
        g.emit(f"s_lshl_b32 {s(S_T1)}, {s(S_T0)}, 13")
        g.emit(f"s_add_u32 {s(S_VLT)}, {s(S_M0V)}, {s(S_T1)}")
        for d in range(3):
            g.emit(f"v_add_u32 {v(V_TKA[d])}, {s(S_KRT)}, {v(V_KA[d])}")
        g.emit(f"v_add_u32 {v(V_TVA)}, {s(S_VRT)}, {v(V_VA)}")

    def load_job(self):
        """Job table entry: [qx, lse_off, q_off (2), k_off (2), v_off >> 7, o_off >> 8]."""
        g = self.g
        g.emit(f"s_lshl_b32 {s(S_T0)}, {s(S_JOB)}, 5")
        g.emit(f"s_add_u32 {s(S_T2)}, {arg('TAB')}, {s(S_T0)}")
        g.emit(f"s_addc_u32 {s(S_T3)}, {argh('TAB')}, 0")
        g.emit(f"s_load_dwordx8 {ss(S_TAB, 8)}, {ss(S_T2, 2)}, 0")
        g.emit("s_waitcnt lgkmcnt(0)")

    def pass_params(self):
        """Per-pass scalars from S_TAB and S_PP. In pair mode group 1 takes the second head of the pair."""
        g = self.g
        # qblk = pp ? qx : num_q - 1 - qx; q_start = (qblk * QW + qwarp) * 32
        g.emit(f"s_sub_u32 {s(S_T0)}, {arg('NUMQ')}, 1")
        g.emit(f"s_sub_u32 {s(S_T0)}, {s(S_T0)}, {s(S_TAB)}")
        g.emit(f"s_cmp_eq_u32 {s(S_PP)}, 0")
        g.emit(f"s_cselect_b32 {s(S_QBLK)}, {s(S_T0)}, {s(S_TAB)}")
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_W)}, 3")
        g.emit(f"s_cmp_eq_u32 {arg('PAIR')}, 0")
        g.emit(f"s_cselect_b32 {s(S_QWARP)}, {s(S_W)}, {s(S_T0)}")
        g.emit(f"s_mul_i32 {s(S_T0)}, {s(S_QBLK)}, {arg('QW')}")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_T0)}, {s(S_QWARP)}")
        g.emit(f"s_lshl_b32 {s(S_QSTART)}, {s(S_T0)}, 5")
        # n = min(((qblk + 1) * QW * 32 + 63) >> 6, n_keys >> 6); fm = q_start >> 6
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_QBLK)}, 1")
        g.emit(f"s_mul_i32 {s(S_T0)}, {s(S_T0)}, {arg('QW')}")
        g.emit(f"s_lshl_b32 {s(S_T0)}, {s(S_T0)}, 5")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_T0)}, 63")
        g.emit(f"s_lshr_b32 {s(S_N)}, {s(S_T0)}, 6")
        g.emit(f"s_lshr_b32 {s(S_T0)}, {arg('NKEYS')}, 6")
        g.emit(f"s_min_u32 {s(S_N)}, {s(S_N)}, {s(S_T0)}")
        g.emit(f"s_lshr_b32 {s(S_FM)}, {s(S_QSTART)}, 6")
        g.emit(f"s_and_b32 {s(S_T1)}, {s(S_G)}, {arg('PAIR')}")
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('Q_SH2')}")
        g.emit(f"s_add_u32 {s(S_QH)}, {s(S_TAB + 2)}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_QH + 1)}, {s(S_TAB + 3)}, 0")
        g.emit(f"s_add_u32 {s(S_QH)}, {s(S_QH)}, {arg('Q')}")
        g.emit(f"s_addc_u32 {s(S_QH + 1)}, {s(S_QH + 1)}, {argh('Q')}")
        g.emit(f"s_mov_b32 {s(S_T3)}, 0")
        g.emit(f"s_mov_b32 {s(S_T2)}, {s(S_TAB + 7)}")
        g.emit(f"s_lshl_b64 {ss(S_T2, 2)}, {ss(S_T2, 2)}, 8")                # o_off
        g.emit(f"s_add_u32 {s(S_OH)}, {s(S_T2)}, {arg('O')}")
        g.emit(f"s_addc_u32 {s(S_OH + 1)}, {s(S_T3)}, {argh('O')}")
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('O_SH2')}")
        g.emit(f"s_add_u32 {s(S_OH)}, {s(S_OH)}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_OH + 1)}, {s(S_OH + 1)}, 0")
        # LSE base (bytes): LSE + 4 * (lse_off + pair_head * L_SH2), 0 when there is no LSE output
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('L_SH2')}")
        g.emit(f"s_add_u32 {s(S_T2)}, {s(S_T2)}, {s(S_TAB + 1)}")
        g.emit(f"s_lshl_b32 {s(S_T2)}, {s(S_T2)}, 2")
        g.emit(f"s_cmp_eq_u64 {ss(S_A + KA['LSE'], 2)}, 0")
        g.emit(f"s_cselect_b32 {s(S_T2)}, 0, {s(S_T2)}")
        g.emit(f"s_add_u32 {s(S_LSE)}, {arg('LSE')}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_LSE + 1)}, {argh('LSE')}, 0")
        # K/V buffer resources: base + offset, num_records = n_keys * row stride
        g.emit(f"s_add_u32 {s(S_KR)}, {arg('K')}, {s(S_TAB + 4)}")
        g.emit(f"s_addc_u32 {s(S_KR + 1)}, {argh('K')}, {s(S_TAB + 5)}")
        g.emit(f"s_mul_i32 {s(S_KR + 2)}, {arg('NKEYS')}, {arg('K_SS')}")
        g.emit(f"s_mov_b32 {s(S_KR + 3)}, 0x110000")
        g.emit(f"s_mov_b32 {s(S_T3)}, 0")
        g.emit(f"s_mov_b32 {s(S_T2)}, {s(S_TAB + 6)}")
        g.emit(f"s_lshl_b64 {ss(S_T2, 2)}, {ss(S_T2, 2)}, 7")                # v_off
        g.emit(f"s_add_u32 {s(S_VR)}, {s(S_T2)}, {arg('V')}")
        g.emit(f"s_addc_u32 {s(S_VR + 1)}, {s(S_T3)}, {argh('V')}")
        g.emit(f"s_mul_i32 {s(S_VR + 2)}, {arg('NKEYS')}, {arg('V_SS')}")
        g.emit(f"s_mov_b32 {s(S_VR + 3)}, 0x110000")

    def q_load(self):
        g = self.g
        g.emit(f"v_and_b32 {v(V_TMP)}, 31, {v(V_TID)}")
        g.emit(f"v_add_u32 {v(V_TMP)}, {s(S_QSTART)}, {v(V_TMP)}")
        g.emit(f"v_mul_lo_u32 {v(V_TMP)}, {v(V_TMP)}, {arg('Q_SS')}")
        g.emit(f"v_bfe_u32 {v(V_T0)}, {v(V_TID)}, 5, 1")
        g.emit(f"v_lshl_add_u32 {v(V_TMP)}, {v(V_T0)}, 5, {v(V_TMP)}")
        for d in range(3):
            for h in range(2):
                g.emit(f"global_load_dwordx4 {vv(Q[d] + 4 * h, 4)}, {v(V_TMP)}, {ss(S_QH, 2)} offset:{64 * d + 16 * h}")

    def soff_init(self):
        """Tile offset clamps (the pass's last tile) and the offsets of the first loads in MM: K(4), V(2)."""
        g = self.g
        g.emit(f"s_sub_u32 {s(S_T0)}, {s(S_N)}, 1")
        g.emit(f"s_mul_i32 {s(S_KMAX)}, {s(S_T0)}, {s(S_KVT)}")
        g.emit(f"s_mul_i32 {s(S_VMAX)}, {s(S_T0)}, {s(S_VVT)}")
        g.emit(f"s_lshl_b32 {s(S_KSOFF)}, {s(S_KVT)}, 2")
        g.emit(f"s_lshl_b32 {s(S_VSOFF)}, {s(S_VVT)}, 1")

    def slot_m0(self, kind, bsgpr, dt, r_bytes, dst):
        """dst = this wave's M0 base in ring slot (bsgpr + dt) & 3, plus r_bytes."""
        g = self.g
        g.emit(f"s_add_u32 {s(dst)}, {s(bsgpr)}, {dt}")
        g.emit(f"s_and_b32 {s(dst)}, {s(dst)}, 3")
        g.emit(f"s_lshl_b32 {s(dst)}, {s(dst)}, {14 if kind == 'K' else 13}")
        g.emit(f"s_add_u32 {s(dst)}, {s(dst)}, {s(S_M0K if kind == 'K' else S_M0V)}")
        if r_bytes:
            g.emit(f"s_add_u32 {s(dst)}, {s(dst)}, {r_bytes}")

    def tile_loads_full(self, tiles, bsgpr):
        """Whole tiles loaded by the 256 lanes of group 0, into ring slots (bsgpr + t) & 3: K in 4 rounds (keys
        + 16 r), V in 2 rounds (keys + 4 r)."""
        g = self.g
        lskip = self.L("tskip")
        for kind, t in tiles:
            if t > 0:
                g.emit(f"s_cmpk_le_u32 {s(S_N)}, {t}")
                g.emit(f"s_cbranch_scc1 {lskip}")
            for r in range(4 if kind == "K" else 2):
                dk = 16 * r if kind == "K" else 4 * r
                g.emit(f"s_mul_i32 {s(S_T1)}, {dk}, {arg('K_SS' if kind == 'K' else 'V_SS')}")
                g.emit(f"s_mul_i32 {s(S_T0)}, {t}, {s(S_KVT if kind == 'K' else S_VVT)}")
                g.emit(f"s_add_u32 {s(S_T1)}, {s(S_T1)}, {s(S_T0)}")
                self.slot_m0(kind, bsgpr, t, 4096 * r, S_T2)
                g.emit(f"s_mov_b32 m0, {s(S_T2)}")
                if kind == "K":
                    g.emit(f"buffer_load_dwordx4 {v(V_GK0)}, {ss(S_KR, 4)}, {s(S_T1)} offen lds")
                else:
                    g.emit(f"buffer_load_dwordx4 {v(V_GV0)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds")
        g.label(lskip)

    def advance(self):
        """Next pass's params (S_DONE = 1 if there is none): pass 1 of this job if its Q block differs from pass 0's,
        else pass 0 of job + grid. Sets S_NBASE = (S_BASE + n) & 3 first."""
        g = self.g
        l_next_job, l_have, l_end = self.L("nextjob"), self.L("have"), self.L("advend")
        g.emit(f"s_add_u32 {s(S_NBASE)}, {s(S_BASE)}, {s(S_N)}")
        g.emit(f"s_and_b32 {s(S_NBASE)}, {s(S_NBASE)}, 3")
        g.emit(f"s_cmp_eq_u32 {s(S_PP)}, 0")
        g.emit(f"s_cbranch_scc0 {l_next_job}")
        g.emit(f"s_mov_b32 {s(S_PP)}, 1")
        g.emit(f"s_sub_u32 {s(S_T0)}, {arg('NUMQ')}, 1")
        g.emit(f"s_sub_u32 {s(S_T0)}, {s(S_T0)}, {s(S_TAB)}")
        g.emit(f"s_cmp_eq_u32 {s(S_T0)}, {s(S_TAB)}")
        g.emit(f"s_cbranch_scc0 {l_have}")
        g.label(l_next_job)
        g.emit(f"s_add_u32 {s(S_JOB)}, {s(S_JOB)}, {arg('GRID')}")
        g.emit(f"s_mov_b32 {s(S_PP)}, 0")
        g.emit(f"s_cmp_ge_u32 {s(S_JOB)}, {arg('NJOBS')}")
        g.emit(f"s_cselect_b32 {s(S_DONE)}, 1, 0")
        g.emit(f"s_cbranch_scc1 {l_end}")
        self.load_job()
        g.label(l_have)
        self.pass_params()
        g.label(l_end)

    # ------------------------------------------------------------ epilogue
    def epi_addr(self):
        """Store addresses of the finishing pass, taken before the next pass's params overwrite S_QSTART/S_OH/S_LSE.
        Lane l stores row q_start + 4 k + (l >> 4), bytes 16 (l & 15) .. +15."""
        g = self.g
        g.emit(f"v_and_b32 {v(V_TMP)}, 31, {v(V_TID)}")
        g.emit(f"v_add_u32 {v(V_TMP)}, {s(S_QSTART)}, {v(V_TMP)}")
        g.emit(f"v_lshlrev_b32 {v(V_ELA)}, 2, {v(V_TMP)}")
        g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_TID)}, 4, 2")
        g.emit(f"v_add_u32 {v(V_TMP)}, {s(S_QSTART)}, {v(V_TMP)}")
        g.emit(f"v_mul_lo_u32 {v(V_EA)}, {v(V_TMP)}, {arg('O_SS')}")
        g.emit(f"v_and_b32 {v(V_TMP)}, 15, {v(V_TID)}")
        g.emit(f"v_lshl_add_u32 {v(V_EA)}, {v(V_TMP)}, 4, {v(V_EA)}")
        for i in range(2):
            g.emit(f"s_mov_b32 {s(S_EOH + i)}, {s(S_OH + i)}")
            g.emit(f"s_mov_b32 {s(S_ELSE + i)}, {s(S_LSE + i)}")

    def epi_store(self):
        """O * 1/l -> bf16, staged through LDS and stored as 1 KB row-contiguous chunks, then the LSE. The stores stay
        outstanding."""
        g = self.g
        g.nop(20)
        g.emit(f"v_rcp_f32 {v(V_TMP)}, {v(V_L)}")
        g.emit(f"v_log_f32 {v(V_LSE)}, {v(V_L)}")
        g.nop(2)
        for j in range(4):
            for r in range(16):
                g.emit(f"v_mul_f32 {v(O[j] + r)}, {v(V_TMP)}, {v(O[j] + r)}")
        for j in range(4):
            for i4 in range(4):
                base = O[j] + 4 * i4
                d = 8 * j + 2 * i4
                g.emit(f"v_cvt_pk_bf16_f32 {v(d)}, {v(base)}, {v(base + 1)}")
                g.emit(f"v_cvt_pk_bf16_f32 {v(d + 1)}, {v(base + 2)}, {v(base + 3)}")
        # dword pair 8j + 2 i4 holds d_v 32 j + 8 i4 + 4 gb + 0..3: swap halves so each lane owns 8 contiguous d_v
        for j in range(4):
            for i4 in (0, 2):
                a, b = 8 * j + 2 * (i4 + 1), 8 * j + 2 * i4
                g.emit(f"v_permlane32_swap_b32 {v(b)}, {v(a)}")
                g.emit(f"v_permlane32_swap_b32 {v(b + 1)}, {v(a + 1)}")
        # stage the wave's 32 rows x 256 B in its own 8 KB of LDS; 16 B slots XOR-swizzled by row (conflict-free)
        T0, T1, T2, T3 = V_T0, V_T1, V_T2, V_T3
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_W)}, 3")
        g.emit(f"s_lshl_b32 {s(S_T0)}, {s(S_T0)}, 13")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_T0)}, {LDS_BYTES}")
        g.emit(f"s_lshl_b32 {s(S_T1)}, {arg('O_SS')}, 2")
        g.emit(f"v_and_b32 {v(T0)}, 31, {v(V_TID)}")                 # q
        g.emit(f"v_lshlrev_b32 {v(T1)}, 8, {v(T0)}")
        g.emit(f"v_add_u32 {v(T1)}, {s(S_T0)}, {v(T1)}")             # base + 256 q
        g.emit(f"v_bfe_u32 {v(T2)}, {v(V_TID)}, 5, 1")               # gb
        g.emit(f"v_and_b32 {v(T3)}, 15, {v(T0)}")
        g.emit(f"v_xor_b32 {v(T2)}, {v(T2)}, {v(T3)}")
        g.emit(f"v_lshl_or_b32 {v(T1)}, {v(T2)}, 4, {v(T1)}")        # slot (4 j + i4 + gb) ^ (q & 15)
        for j in range(4):
            for i4 in (0, 2):
                c = (4 * j + i4) << 4
                g.emit(f"v_xor_b32 {v(T2)}, {c}, {v(T1)}" if c <= 64 else f"v_xor_b32 {v(T2)}, 0x{c:x}, {v(T1)}")
                g.emit(f"ds_write_b128 {v(T2)}, {vv(8 * j + 2 * i4, 4)}")
        g.emit(f"v_bfe_u32 {v(T0)}, {v(V_TID)}, 4, 2")               # h = row within a 4-row chunk
        g.emit(f"v_and_b32 {v(T3)}, 15, {v(V_TID)}")                 # 16 B slot
        g.emit(f"v_xor_b32 {v(T3)}, {v(T3)}, {v(T0)}")
        g.emit(f"v_lshlrev_b32 {v(T0)}, 8, {v(T0)}")
        g.emit(f"v_add_u32 {v(T0)}, {s(S_T0)}, {v(T0)}")
        g.emit(f"v_lshl_or_b32 {v(T0)}, {v(T3)}, 4, {v(T0)}")
        g.emit(f"v_xor_b32 {v(T1)}, 64, {v(T0)}")
        g.emit(f"v_xor_b32 {v(T2)}, 0x80, {v(T0)}")
        g.emit(f"v_xor_b32 {v(T3)}, 0xc0, {v(T0)}")
        bx = [T0, T1, T2, T3]
        for k in range(8):   # chunk k = rows 4k .. 4k + 3
            g.emit(f"ds_read_b128 {vv(O[2] + 4 * k, 4)}, {v(bx[k & 3])} offset:{1024 * k}")
        g.emit(f"v_mov_b32 {v(P)}, {v(V_EA)}")       # store addresses in P (free until the next SM's codes)
        for k in range(1, 8):
            g.emit(f"v_add_u32 {v(P + k)}, {s(S_T1)}, {v(P + k - 1)}")
        for k in range(8):
            g.emit(f"s_waitcnt lgkmcnt({7 - k})")
            g.emit(f"global_store_dwordx4 {v(P + k)}, {vv(O[2] + 4 * k, 4)}, {ss(S_EOH, 2)}")
        # LSE (natural log) = (m - 6 + log2 l) * ln 2
        g.emit(f"v_add_f32 {v(V_LSE)}, {v(V_M)}, {v(V_LSE)}")
        g.emit(f"v_add_f32 {v(V_LSE)}, -4.0, {v(V_LSE)}")
        g.emit(f"v_add_f32 {v(V_LSE)}, -2.0, {v(V_LSE)}")
        g.emit(f"v_mul_f32 {v(V_LSE)}, 0x3f317218, {v(V_LSE)}")
        lno = self.L("nolse")
        g.emit(f"s_cmp_eq_u64 {ss(S_ELSE, 2)}, 0")
        g.emit(f"s_cbranch_scc1 {lno}")
        g.emit(f"global_store_dword {v(V_ELA)}, {v(V_LSE)}, {ss(S_ELSE, 2)}")
        g.label(lno)

    # ------------------------------------------------------------ kernel
    def gen(self):
        g = self.g
        self.kernel_entry()
        l_end, l_top = ".Lend", ".Lpasstop"
        g.emit(f"s_mov_b32 {s(S_PP)}, 0")
        g.emit(f"s_mov_b32 {s(S_BASE)}, 0")
        g.emit(f"s_mov_b32 {s(S_DONE)}, 0")
        g.emit(f"s_mov_b32 {s(S_FLAG)}, 0")
        g.emit(f"s_cmp_ge_u32 {s(S_JOB)}, {arg('NJOBS')}")
        g.emit(f"s_cbranch_scc1 {l_end}")
        self.load_job()
        self.pass_params()
        self.q_load()
        # first pass: every wave loads its share of K0, V0, K1 (ring slots 0, 0, 1)
        lsk = self.L("p0skip")
        for kind, t in (("K", 0), ("V", 0), ("K", 1)):
            if t > 0:
                g.emit(f"s_cmpk_le_u32 {s(S_N)}, {t}")
                g.emit(f"s_cbranch_scc1 {lsk}")
            if kind == "K":
                g.emit(f"s_mul_i32 {s(S_T1)}, {t}, {s(S_KVT)}")
                for p_, voff in enumerate((V_GK0, V_GK1)):
                    g.emit(f"s_add_u32 m0, {s(S_M0K)}, {LDS_K[t] + 8192 * p_}")
                    g.emit(f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T1)} offen lds")
            else:
                g.emit(f"s_mul_i32 {s(S_T1)}, {t}, {s(S_VVT)}")
                g.emit(f"s_add_u32 m0, {s(S_M0V)}, {8192 * t}")
                g.emit(f"buffer_load_dwordx4 {v(V_GV0)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds")
        g.label(lsk)
        for j in range(4):
            for r in range(16):
                g.emit(f"v_mov_b32 {v(O[j] + r)}, 0")
        g.emit("s_waitcnt vmcnt(0)")
        self.bar()
        self.bar_group(1)      # group 1 runs one phase behind group 0
        l_first = self.L("first")
        g.emit(f"s_branch {l_first}")
        g.label(l_top)
        self.epi_store()       # the previous pass's epilogue
        g.label(l_first)
        self.pass_body()
        g.emit(f"s_cmp_eq_u32 {s(S_DONE)}, 0")
        g.emit(f"s_cbranch_scc1 {l_top}")
        self.epi_store()
        self.bar_group(0)
        g.label(l_end)
        g.emit("s_waitcnt vmcnt(0)")
        g.emit("s_endpgm")

    def pass_body(self):
        """One pass: QK0, the steady loop over unmasked tiles (unrolled x4 over the ring slots), the tail over the
        masked tiles, and the transition to the next pass."""
        g = self.g
        g.emit("s_waitcnt vmcnt(8)")       # this pass's Q; only the previous epilogue's stores may be outstanding
        g.emit(f"s_mov_b32 {s(S_K)}, -1")
        g.emit(f"s_lshl_b32 {s(S_T0)}, {s(S_BASE)}, 14")
        for d in range(3):
            g.emit(f"v_add_u32 {v(V_TKA[d])}, {s(S_T0)}, {v(V_KA[d])}")
        self.soff_init()
        self.pre_read_k(None)
        lm, lj = self.L("qk0m"), self.L("qk0j")
        g.emit(f"s_cmp_eq_u32 {s(S_FM)}, 0")
        g.emit(f"s_cbranch_scc1 {lm}")
        self.seg_mm("qk0", None, None, 1, None)
        g.emit(f"s_branch {lj}")
        g.label(lm)
        self.seg_mm("qk0m", None, None, 1, None)
        g.label(lj)
        # the statistics wrote parity set 0; tile 0 uses the parity of its slot (S_BASE & 1)
        lpe = self.L("pareven")
        g.emit(f"s_bitcmp1_b32 {s(S_BASE)}, 0")
        g.emit(f"s_cbranch_scc0 {lpe}")
        g.emit(f"v_mov_b32 {v(SB[1])}, {v(SB[0])}")
        g.label(lpe)
        g.emit("s_waitcnt vmcnt(0)")       # the previous pass's O stores (data in O[0:31])
        for j in range(4):
            for r in range(16):
                g.emit(f"v_mov_b32 {v(O[j] + r)}, 0")
        g.emit(f"v_mov_b32 {v(V_L)}, 0")
        g.emit(f"s_mov_b32 {s(S_FLAG)}, 0")
        lg1 = self.L("nog0")
        g.emit(f"s_cmp_eq_u32 {s(S_G)}, 0")
        g.emit(f"s_cbranch_scc0 {lg1}")
        self.tile_loads_full((("V", 1), ("K", 2), ("K", 3)), S_BASE)
        g.label(lg1)
        self.bar()
        g.emit(f"s_mov_b32 {s(S_K)}, 0")
        l_tail = self.L("tail")
        lu = [self.L(f"st{u}") for u in range(4)]
        # enter the unrolled loop at u = S_BASE
        for u in range(3):
            g.emit(f"s_cmp_eq_u32 {s(S_BASE)}, {u}")
            g.emit(f"s_cbranch_scc1 {lu[u]}")
        g.emit(f"s_branch {lu[3]}")
        for u in range(4):
            g.label(lu[u])
            g.emit(f"s_add_u32 {s(S_T0)}, {s(S_K)}, 1")
            g.emit(f"s_cmp_ge_u32 {s(S_T0)}, {s(S_FM)}")
            g.emit(f"s_cbranch_scc1 {l_tail}")
            self.seg_sm((u + 1) % 4)
            self.bar()
            self.seg_mm("mm", (u + 1) % 4, u, u & 1, (u, (u + 2) % 4))
            self.bar()
            g.emit(f"s_add_u32 {s(S_K)}, {s(S_K)}, 1")
        g.emit(f"s_branch {lu[0]}")
        # tail: masked tiles and the last tile with runtime slots, dispatched on the slot parity
        l_t0, l_t1, l_last0, l_last1, l_done = (self.L("tail0"), self.L("tail1"), self.L("last0"), self.L("last1"),
                                                self.L("pdone"))
        g.label(l_tail)
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_K)}, {s(S_BASE)}")
        g.emit(f"s_bitcmp1_b32 {s(S_T0)}, 0")
        g.emit(f"s_cbranch_scc1 {l_t1}")
        for par, lab, llast, lother in ((0, l_t0, l_last0, l_t1), (1, l_t1, l_last1, l_t0)):
            g.label(lab)
            g.emit(f"s_add_u32 {s(S_T0)}, {s(S_K)}, 1")
            g.emit(f"s_cmp_ge_u32 {s(S_T0)}, {s(S_N)}")
            g.emit(f"s_cbranch_scc1 {llast}")
            self.tail_state()
            # tiles past S_FM hold no key visible to this wave: k == S_FM runs PV(k) only, k > S_FM only the loads
            lnorm, lpvo = self.L("tnorm"), self.L("tpvo")
            g.emit(f"s_cmp_lt_u32 {s(S_K)}, {s(S_FM)}")
            g.emit(f"s_cbranch_scc1 {lnorm}")
            g.emit(f"s_cmp_eq_u32 {s(S_K)}, {s(S_FM)}")
            g.emit(f"s_cbranch_scc1 {lpvo}")
            self.bar()
            g.emit("s_waitcnt vmcnt(0)")
            for grp in self.loads_rt():
                self.emit_list(grp)
            self.bar()
            g.emit(f"s_add_u32 {s(S_K)}, {s(S_K)}, 1")
            g.emit(f"s_branch {lother}")
            g.label(lpvo)
            self.seg_sm(False)
            self.bar()
            self.seg_mm("pvo", None, None, par, "rt")
            self.bar()
            g.emit(f"s_add_u32 {s(S_K)}, {s(S_K)}, 1")
            g.emit(f"s_branch {lother}")
            g.label(lnorm)
            self.seg_sm(None)
            self.bar()
            self.seg_mm("mmm", None, None, par, "rt")
            self.bar()
            g.emit(f"s_add_u32 {s(S_K)}, {s(S_K)}, 1")
            g.emit(f"s_branch {lother}")
            g.label(llast)
            self.tail_state()
            ldead_last, ljoin = self.L("deadlast"), self.L("lastjoin")
            g.emit(f"s_cmp_lt_u32 {s(S_FM)}, {s(S_K)}")     # last tile dead for this wave: no codes, no PV
            g.emit(f"s_cbranch_scc1 {ldead_last}")
            self.seg_sm(False)
            g.emit("s_waitcnt vmcnt(0)")
            self.bar()
            self.transition(par, with_pv=True)
            g.emit(f"s_branch {ljoin}")
            g.label(ldead_last)
            g.emit("s_waitcnt vmcnt(0)")
            self.bar()
            self.transition(par, with_pv=False)
            g.label(ljoin)
            g.emit("s_waitcnt vmcnt(6)")     # the next pass's tiles have landed; its 6 Q loads are waited before QK0
            g.emit(f"s_mov_b32 {s(S_BASE)}, {s(S_NBASE)}")
            self.bar()
            if par == 0:
                g.emit(f"s_branch {l_done}")
        g.label(l_done)

    def transition(self, par, with_pv):
        """Start of the last MM: epilogue addresses, next pass's params; group 0 loads the next pass's K0, V0, K1
        (ring slots from S_NBASE), each wave its next Q; then PV(n-1) unless the last tile is dead for this wave."""
        g = self.g
        self.epi_addr()
        self.advance()
        lnn = self.L("nonext")
        g.emit(f"s_cmp_eq_u32 {s(S_DONE)}, 0")
        g.emit(f"s_cbranch_scc0 {lnn}")
        lgg = self.L("nog0b")
        g.emit(f"s_cmp_eq_u32 {s(S_G)}, 0")
        g.emit(f"s_cbranch_scc0 {lgg}")
        self.tile_loads_full((("K", 0), ("V", 0), ("K", 1)), S_NBASE)
        g.label(lgg)
        self.q_load()
        g.label(lnn)
        if with_pv:
            self.seg_mm("last", None, None, par, None)


def generate(name="fmha_fp8pp192"):
    """Returns the assembly text and the list of hazard paddings inserted."""
    k = Kernel()
    k.gen()
    return kernel_text(name, k.g.text(), LDS_BYTES + LDS_EPI), k.g.pads


if __name__ == "__main__":
    main(generate, "fmha_fp8_d192_pp.s")
