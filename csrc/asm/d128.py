#!/usr/bin/env python3
"""Generates the FP8 causal forward attention kernel for d_qk = d_v = 128 on gfx950 (`fmha_fp8pp`) as hand-scheduled
assembly. Eight waves in two groups run one phase apart, so on each SIMD one wave issues the MFMAs (QK, PV) while the
other computes the softmax statistics and the e4m3 codes of P. 128-key K and V tiles go through 4-slot LDS rings."""
from common import KernelBase, LdsQ, kernel_text, main, rescale_vote, s, ss, v, vv

# ---------------------------------------------------------------- registers
O = [0, 16, 32, 48]            # O^T accumulators (fp32, d_v rows x 32 queries)
S = [64, 80, 96, 112]          # S^T blocks (32 keys x 32 queries)
P_LO, P_HI = 128, 136          # e4m3 codes of P, keys 0-63 / 64-127
Q0, Q1 = 144, 152              # Q, d 0-63 / 64-127 (MFMA B operand)
SEL = 160                      # 0/1 selector, A operand of the row-sum MFMA
FR = 168                       # LDS fragment ring: NSLOT x 8 VGPRs
NSLOT = 5
RS_LO, RS_HI = 208, 212        # row sums of P
V_TID = 216
V_TKA0, V_TKA1, V_TVA = 217, 218, 219    # K/V LDS read addresses for a runtime ring slot
V_MREF = 220                   # reference max of the tile being coded
V_EA, V_ELA, V_LSE = 221, 222, 223       # epilogue: O and LSE store addresses, LSE value
V_KA0, V_KA1, V_VA = 224, 225, 226       # K/V LDS read addresses (V_VA includes the V ring base)
V_GK0, V_GK1, V_GV0, V_GV1 = 227, 228, 229, 230   # global -> LDS voffsets
V_M, V_L = 231, 232            # running row max (log2 units) and row sum
SB = [(233, 234), (235, 236)]  # E8M0 scale of P per 64-key half, two parity sets
P2 = [(237, 238), (239, 240)]  # 2^offset per 64-key half, two parity sets
V_B8LO, V_B8HI = 241, 242      # P code bias per 64-key half
V_127 = 243                    # E8M0 scale 1.0 of the V operand
V_T0, V_T1, V_T2, V_T3 = 244, 245, 246, 247
V_CORR = 248                   # rescale factor
V_THR = 249                    # causal mask threshold
V_NINF = 250                   # -inf
V_M8 = 251                     # -8.0
V_KL0, V_KL1, V_VL = 252, 253, 254       # lane parts of the LDS read addresses
V_TMP = 255

S_KARG = 0                     # s[0:1]
S_WG = 2
S_QH, S_OH = 4, 6              # s[4:5], s[6:7]: Q and O bases of the pass
S_KR, S_VR = 8, 12             # K and V buffer resources s[8:11], s[12:15]
S_A = 16                       # kernarg dwords 0-31 in s[16:47] (dwords 29-31 are not read: S_U, S_DONE, S_NBASE)
S_TAB = 48                     # job table entry s[48:55]
S_JOB, S_W, S_G, S_QWARP = 56, 57, 58, 59                      # job, wave, wave group, query block of the wave
S_PP, S_QBLK, S_QSTART, S_N, S_FM, S_K = 60, 61, 62, 63, 64, 65  # pass, Q block, first query, tiles, first masked, k
S_KVT = 66                     # bytes per tile (128 * kv_ss)
S_KSOFF, S_VSOFF = 71, 72      # global offsets of the next K and V tile loads
S_FLAG = 73                    # rescale vote
S_RMASK = 74                   # s[74:75]
S_LSE = 76                     # s[76:77]
S_T0, S_T1, S_T2, S_T3 = 78, 79, 80, 81
S_M127 = 82                    # -127.0
S_M0K, S_M0V, S_KMAX = 83, 84, 85        # per-wave LDS DMA bases of the K/V rings, offset of the pass's last tile
S_KRT, S_VRT, S_KLT, S_VLT = 86, 87, 88, 89   # runtime ring slots: K/V read bases, K/V load M0 bases
S_PERM = 90                    # set at entry, not read
S_BASE = 91                    # ring slot of the pass's tile 0
S_U, S_DONE, S_NBASE = 45, 46, 47        # base + k, "no next pass", ring slot of the next pass's tile 0
S_EOH, S_ELSE = 92, 94         # s[92:93], s[94:95]: O / LSE bases of the pass being stored

# kernarg layout (dwords from S_A)
KA = dict(Q=0, K=2, V=4, O=6, LSE=8, TAB=10, C8=12, SL2=13, B8K=14, NJOBS=15, NUMQ=16, QW=17, PAIR=18, Q_SS=19,
          KV_SS=20, O_SS=21, Q_SH2=22, O_SH2=23, L_SH2=24, NKEYS=25, GRID=26)

LDS_K = [0, 16384, 32768, 49152]   # K ring slots
LDS_V0 = 65536                     # V ring: 4 slots of 16 KB
LDS_BYTES = 131072                 # end of the rings; the epilogue stages O above (8 KB per wave of a group)
LDS_EPI = 32768

QK_ORDER = [(b, d) for d in (0, 1) for b in range(4)]   # (key block, d half) of the 8 QK MFMAs
NPRE = 2            # K fragments read at the end of SM, ahead of the first QK MFMA
DEPTH = 4           # fragment read-ahead (MFMAs)
VALU_PER_SLOT = 8   # VALU instructions placed after each MFMA of MM (x3 while masking)
LOAD_AT = 4         # MFMA after which the tile loads of MM start


def arg(name):
    return s(S_A + KA[name])


def argh(name):
    return s(S_A + KA[name] + 1)


class Kernel(KernelBase):
    # ------------------------------------------------------------ entry and per-pass scalars
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
        # K lane read offsets: c32 * 128 + 32 * (((2 dh + gb) ^ (c32 >> 2)) & 3)
        g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 2, {v(V_T1)}")
        for dh, dst in ((0, V_KL0), (1, V_KL1)):
            g.emit(f"v_add_u32 {v(dst)}, {dh * 2}, {v(V_T2)}")
            g.emit(f"v_xor_b32 {v(dst)}, {v(dst)}, {v(V_TMP)}")
            g.emit(f"v_and_b32 {v(dst)}, 3, {v(dst)}")
            g.emit(f"v_lshlrev_b32 {v(dst)}, 5, {v(dst)}")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_T1)}, 7, {v(dst)}")
        # V lane read offset: (gb * 64 + (c32 >> 4)) * 128 + (c32 & 15) * 8
        g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 4, {v(V_T1)}")
        g.emit(f"v_lshl_add_u32 {v(V_VL)}, {v(V_T2)}, 6, {v(V_TMP)}")
        g.emit(f"v_lshlrev_b32 {v(V_VL)}, 7, {v(V_VL)}")
        g.emit(f"v_and_b32 {v(V_TMP)}, 15, {v(V_T1)}")
        g.emit(f"v_lshl_add_u32 {v(V_VL)}, {v(V_TMP)}, 3, {v(V_VL)}")
        g.emit("s_waitcnt lgkmcnt(0)")
        g.emit(f"v_mov_b32 {v(V_127)}, 0x7f")
        g.emit(f"v_mov_b32 {v(V_NINF)}, 0xff800000")
        g.emit(f"v_mov_b32 {v(V_M8)}, 0xc1000000")
        g.emit(f"s_mov_b32 {s(S_M127)}, 0xc2fe0000")
        g.emit(f"s_lshl_b32 {s(S_KVT)}, {arg('KV_SS')}, 7")
        # K G2L voffsets, pass p: c = tid + 512 p, key = c >> 3, ch = c & 7, L = ((ch >> 1) ^ (key >> 2)) & 3,
        # voff = key * kv_ss + 32 L + 16 (ch & 1)
        for p, dst in ((0, V_GK0), (1, V_GK1)):
            g.emit(f"v_add_u32 {v(V_T3)}, {512 * p}, {v(V_TID)}")
            g.emit(f"v_lshrrev_b32 {v(V_TMP)}, 3, {v(V_T3)}")
            g.emit(f"v_bfe_u32 {v(dst)}, {v(V_T3)}, 1, 2")
            g.emit(f"v_lshrrev_b32 {v(V_T0)}, 2, {v(V_TMP)}")
            g.emit(f"v_xor_b32 {v(dst)}, {v(dst)}, {v(V_T0)}")
            g.emit(f"v_and_b32 {v(dst)}, 3, {v(dst)}")
            g.emit(f"v_lshlrev_b32 {v(dst)}, 5, {v(dst)}")
            g.emit(f"v_bfe_u32 {v(V_T0)}, {v(V_T3)}, 0, 1")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_T0)}, 4, {v(dst)}")
            g.emit(f"v_mad_u32_u24 {v(dst)}, {v(V_TMP)}, {arg('KV_SS')}, {v(dst)}")
        # V G2L voffsets, pass p: c = tid + 512 p, t = c & 7, dg = (c >> 3) & 7, half = bit 6, nk = bits 7-8,
        # gt = bit 9; key = 32 nk + 4 gt + 16 half + 8 (t >> 2) + (t & 3), voff = key * kv_ss + 16 dg
        for p, dst in ((0, V_GV0), (1, V_GV1)):
            g.emit(f"v_add_u32 {v(V_T3)}, {512 * p}, {v(V_TID)}")
            g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_T3)}, 7, 2")
            g.emit(f"v_lshlrev_b32 {v(dst)}, 5, {v(V_TMP)}")
            g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_T3)}, 9, 1")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_TMP)}, 2, {v(dst)}")
            g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_T3)}, 6, 1")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_TMP)}, 4, {v(dst)}")
            g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_T3)}, 2, 1")
            g.emit(f"v_lshl_add_u32 {v(dst)}, {v(V_TMP)}, 3, {v(dst)}")
            g.emit(f"v_and_b32 {v(V_TMP)}, 3, {v(V_T3)}")
            g.emit(f"v_add_u32 {v(dst)}, {v(dst)}, {v(V_TMP)}")
            g.emit(f"v_bfe_u32 {v(V_TMP)}, {v(V_T3)}, 3, 3")
            g.emit(f"v_lshlrev_b32 {v(V_TMP)}, 4, {v(V_TMP)}")
            g.emit(f"v_mad_u32_u24 {v(dst)}, {v(dst)}, {arg('KV_SS')}, {v(V_TMP)}")
        g.emit(f"s_mov_b32 {s(S_JOB)}, {s(S_WG)}")
        g.emit(f"v_add_u32 {v(V_VA)}, {LDS_V0}, {v(V_VL)}")
        g.emit(f"v_mov_b32 {v(V_KA0)}, {v(V_KL0)}")
        g.emit(f"v_mov_b32 {v(V_KA1)}, {v(V_KL1)}")
        g.emit(f"s_lshl_b32 {s(S_M0K)}, {s(S_W)}, 10")
        g.emit(f"s_add_u32 {s(S_M0V)}, {s(S_M0K)}, {LDS_V0}")
        g.emit(f"s_mov_b32 {s(S_PERM)}, 0x07050301")

    def load_job(self):
        """Job table entry: [qx, lse_off, q_off (2), kv_off (2), o_off (2)]."""
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
        # n = min(((qblk + 1) * QW * 32 + 127) >> 7, n_keys >> 7); fm = q_start >> 7
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_QBLK)}, 1")
        g.emit(f"s_mul_i32 {s(S_T0)}, {s(S_T0)}, {arg('QW')}")
        g.emit(f"s_lshl_b32 {s(S_T0)}, {s(S_T0)}, 5")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_T0)}, 127")
        g.emit(f"s_lshr_b32 {s(S_N)}, {s(S_T0)}, 7")
        g.emit(f"s_lshr_b32 {s(S_T0)}, {arg('NKEYS')}, 7")
        g.emit(f"s_min_u32 {s(S_N)}, {s(S_N)}, {s(S_T0)}")
        g.emit(f"s_lshr_b32 {s(S_FM)}, {s(S_QSTART)}, 7")
        g.emit(f"s_and_b32 {s(S_T1)}, {s(S_G)}, {arg('PAIR')}")
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('Q_SH2')}")
        g.emit(f"s_add_u32 {s(S_QH)}, {s(S_TAB + 2)}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_QH + 1)}, {s(S_TAB + 3)}, 0")
        g.emit(f"s_add_u32 {s(S_QH)}, {s(S_QH)}, {arg('Q')}")
        g.emit(f"s_addc_u32 {s(S_QH + 1)}, {s(S_QH + 1)}, {argh('Q')}")
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('O_SH2')}")
        g.emit(f"s_add_u32 {s(S_OH)}, {s(S_TAB + 6)}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_OH + 1)}, {s(S_TAB + 7)}, 0")
        g.emit(f"s_add_u32 {s(S_OH)}, {s(S_OH)}, {arg('O')}")
        g.emit(f"s_addc_u32 {s(S_OH + 1)}, {s(S_OH + 1)}, {argh('O')}")
        # LSE base (bytes): LSE + 4 * (lse_off + pair_head * L_SH2), 0 when there is no LSE output
        g.emit(f"s_mul_i32 {s(S_T2)}, {s(S_T1)}, {arg('L_SH2')}")
        g.emit(f"s_add_u32 {s(S_T2)}, {s(S_T2)}, {s(S_TAB + 1)}")
        g.emit(f"s_lshl_b32 {s(S_T2)}, {s(S_T2)}, 2")
        g.emit(f"s_cmp_eq_u64 {ss(S_A + KA['LSE'], 2)}, 0")
        g.emit(f"s_cselect_b32 {s(S_T2)}, 0, {s(S_T2)}")
        g.emit(f"s_add_u32 {s(S_LSE)}, {arg('LSE')}, {s(S_T2)}")
        g.emit(f"s_addc_u32 {s(S_LSE + 1)}, {argh('LSE')}, 0")
        # K/V buffer resources: base + kv_off, num_records = n_keys * kv_ss
        for rs, nm in ((S_KR, "K"), (S_VR, "V")):
            g.emit(f"s_add_u32 {s(rs)}, {arg(nm)}, {s(S_TAB + 4)}")
            g.emit(f"s_addc_u32 {s(rs + 1)}, {argh(nm)}, {s(S_TAB + 5)}")
            g.emit(f"s_mul_i32 {s(rs + 2)}, {arg('NKEYS')}, {arg('KV_SS')}")
            g.emit(f"s_mov_b32 {s(rs + 3)}, 0x110000")

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

    def q_load(self):
        g = self.g
        g.emit(f"v_and_b32 {v(V_TMP)}, 31, {v(V_TID)}")
        g.emit(f"v_add_u32 {v(V_TMP)}, {s(S_QSTART)}, {v(V_TMP)}")
        g.emit(f"v_mul_lo_u32 {v(V_TMP)}, {v(V_TMP)}, {arg('Q_SS')}")
        g.emit(f"v_bfe_u32 {v(V_T0)}, {v(V_TID)}, 5, 1")
        g.emit(f"v_lshl_add_u32 {v(V_TMP)}, {v(V_T0)}, 5, {v(V_TMP)}")
        for off, dst in ((0, Q0), (16, Q0 + 4), (64, Q1), (80, Q1 + 4)):
            g.emit(f"global_load_dwordx4 {vv(dst, 4)}, {v(V_TMP)}, {ss(S_QH, 2)} offset:{off}")

    def soff_init(self):
        """Tile offset clamp (the pass's last tile) and the offsets of the first loads in MM: K(4), V(2)."""
        g = self.g
        g.emit(f"s_sub_u32 {s(S_T0)}, {s(S_N)}, 1")
        g.emit(f"s_mul_i32 {s(S_KMAX)}, {s(S_T0)}, {s(S_KVT)}")
        g.emit(f"s_lshl_b32 {s(S_KSOFF)}, {s(S_KVT)}, 2")
        g.emit(f"s_lshl_b32 {s(S_VSOFF)}, {s(S_KVT)}, 1")

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
        g.emit(f"s_mov_b32 {s(S_VRT)}, {s(S_T1)}")
        g.emit(f"s_add_u32 {s(S_T0)}, {s(S_U)}, 2")
        g.emit(f"s_and_b32 {s(S_T0)}, {s(S_T0)}, 3")
        g.emit(f"s_lshl_b32 {s(S_T1)}, {s(S_T0)}, 14")
        g.emit(f"s_add_u32 {s(S_VLT)}, {s(S_M0V)}, {s(S_T1)}")
        g.emit(f"v_add_u32 {v(V_TKA0)}, {s(S_KRT)}, {v(V_KA0)}")
        g.emit(f"v_add_u32 {v(V_TKA1)}, {s(S_KRT)}, {v(V_KA1)}")
        g.emit(f"v_add_u32 {v(V_TVA)}, {s(S_VRT)}, {v(V_VA)}")

    def bar_group(self, grp):
        """One extra barrier for wave group grp only."""
        g = self.g
        lab = self.L("nobar")
        g.emit(f"s_cmp_eq_u32 {s(S_G)}, {grp}")
        g.emit(f"s_cbranch_scc0 {lab}")
        self.bar()
        g.label(lab)

    # ------------------------------------------------------------ global -> LDS
    def slot_m0(self, kind, bsgpr, dt, r_bytes, dst):
        """dst = this wave's M0 base in ring slot (bsgpr + dt) & 3, plus r_bytes."""
        g = self.g
        g.emit(f"s_add_u32 {s(dst)}, {s(bsgpr)}, {dt}")
        g.emit(f"s_and_b32 {s(dst)}, {s(dst)}, 3")
        g.emit(f"s_lshl_b32 {s(dst)}, {s(dst)}, 14")
        g.emit(f"s_add_u32 {s(dst)}, {s(dst)}, {s(S_M0K if kind == 'K' else S_M0V)}")
        if r_bytes:
            g.emit(f"s_add_u32 {s(dst)}, {s(dst)}, {r_bytes}")

    def tile_loads_full(self, tiles, bsgpr):
        """Whole tiles loaded by the 256 lanes of group 0 in 4 rounds of 4 KB, into ring slots (bsgpr + t) & 3."""
        g = self.g
        lskip = self.L("tskip")
        for kind, t in tiles:
            if t > 0:
                g.emit(f"s_cmpk_le_u32 {s(S_N)}, {t}")
                g.emit(f"s_cbranch_scc1 {lskip}")
            for r in range(4):
                dk = 32 * r if kind == "K" else [0, 64, 4, 68][r]   # key offset of round r (V layout bits 8, 9)
                g.emit(f"s_mul_i32 {s(S_T1)}, {dk}, {arg('KV_SS')}")
                g.emit(f"s_mul_i32 {s(S_T0)}, {t}, {s(S_KVT)}")
                g.emit(f"s_add_u32 {s(S_T1)}, {s(S_T1)}, {s(S_T0)}")
                self.slot_m0(kind, bsgpr, t, 4096 * r, S_T2)
                g.emit(f"s_mov_b32 m0, {s(S_T2)}")
                if kind == "K":
                    g.emit(f"buffer_load_dwordx4 {v(V_GK0)}, {ss(S_KR, 4)}, {s(S_T1)} offen lds")
                else:
                    g.emit(f"buffer_load_dwordx4 {v(V_GV0)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds")
        g.label(lskip)

    def loads_static(self, kslot, vslot):
        """K(k+4) -> ring slot kslot, V(k+2) -> vslot, offsets clamped to the pass's last tile: 4 instruction groups."""
        grp = []
        for p, voff in enumerate((V_GK0, V_GK1)):
            pre = [f"s_min_u32 {s(S_T0)}, {s(S_KSOFF)}, {s(S_KMAX)}"] if p == 0 else []
            grp.append(pre + [f"s_add_u32 m0, {s(S_M0K)}, {LDS_K[kslot] + 8192 * p}",
                              f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T0)} offen lds"])
        for p, voff in enumerate((V_GV0, V_GV1)):
            pre = [f"s_min_u32 {s(S_T1)}, {s(S_VSOFF)}, {s(S_KMAX)}"] if p == 0 else []
            post = [f"s_add_u32 {s(S_KSOFF)}, {s(S_KSOFF)}, {s(S_KVT)}",
                    f"s_add_u32 {s(S_VSOFF)}, {s(S_VSOFF)}, {s(S_KVT)}"] if p == 1 else []
            grp.append(pre + [f"s_add_u32 m0, {s(S_M0V)}, {16384 * vslot + 8192 * p}",
                              f"buffer_load_dwordx4 {v(voff)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds"] + post)
        return grp

    def loads_rt(self):
        """K(k+4) and V(k+2) into the runtime slots, only when inside the pass: 2 instruction groups."""
        lk, lv = self.L("tk"), self.L("tv")
        k = [f"s_add_u32 {s(S_T0)}, {s(S_K)}, 4", f"s_cmp_ge_u32 {s(S_T0)}, {s(S_N)}", f"s_cbranch_scc1 {lk}",
             f"s_mul_i32 {s(S_T0)}, {s(S_T0)}, {s(S_KVT)}"]
        for p, voff in enumerate((V_GK0, V_GK1)):
            k += [f"s_add_u32 m0, {s(S_KLT)}, {8192 * p}",
                  f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T0)} offen lds"]
        k += [f"LABEL {lk}"]
        vl = [f"s_add_u32 {s(S_T1)}, {s(S_K)}, 2", f"s_cmp_ge_u32 {s(S_T1)}, {s(S_N)}", f"s_cbranch_scc1 {lv}",
              f"s_mul_i32 {s(S_T1)}, {s(S_T1)}, {s(S_KVT)}"]
        for p, voff in enumerate((V_GV0, V_GV1)):
            vl += [f"s_add_u32 m0, {s(S_VLT)}, {8192 * p}",
                   f"buffer_load_dwordx4 {v(voff)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds"]
        vl += [f"LABEL {lv}"]
        return [k, vl]

    # ------------------------------------------------------------ LDS fragments
    def read_kfrag(self, slot, blk, dh, kb):
        """K fragment (key block blk, d half dh) of ring slot kb (None: runtime slot) into ring register slot."""
        if kb is None:
            base, off = (V_TKA0 if dh == 0 else V_TKA1), 4096 * blk
        else:
            base, off = (V_KA0 if dh == 0 else V_KA1), 16384 * kb + 4096 * blk
        r = FR + 8 * slot
        self.g.emit(f"ds_read_b128 {vv(r, 4)}, {v(base)} offset:{off}")
        self.g.emit(f"ds_read_b128 {vv(r + 4, 4)}, {v(base)} offset:{off + 16}")

    def read_vfrag(self, slot, half, j, vb):
        r = FR + 8 * slot
        base, b0 = (V_TVA, 0) if vb is None else (V_VA, 16384 * vb)
        for i in range(4):
            off = b0 + 4096 * half + 256 * j + 1024 * i
            self.g.emit(f"ds_read_b64_tr_b8 {vv(r + 2 * i, 2)}, {v(base)} offset:{off}")

    def pre_read_k(self, kb):
        for i in range(NPRE):
            b, d = QK_ORDER[i]
            self.read_kfrag(i % NSLOT, b, d, kb)

    # ------------------------------------------------------------ softmax pieces
    def stats_instrs(self, first, wpar):
        """Statistics of S(k+1): row max, rescale vote, per-half E8M0 offsets (written to parity set wpar), code bias.
        Returns the per-block max chains (each needs its S block final) and the tail."""
        sl2 = arg("SL2")
        groups = []
        for base, a in ((S[0], V_T0), (S[1], V_T1), (S[2], V_T2), (S[3], V_T3)):
            lst = [f"v_max_f32 {v(a)}, {v(base)}, {v(base + 1)}"]
            for r in range(2, 16, 2):
                lst.append(f"v_max3_f32 {v(a)}, {v(base + r)}, {v(base + r + 1)}, {v(a)}")
            groups.append(lst)
        tail = [f"v_max_f32 {v(V_T0)}, {v(V_T0)}, {v(V_T1)}", f"v_max_f32 {v(V_T2)}, {v(V_T2)}, {v(V_T3)}",
                f"v_mov_b32 {v(V_T1)}, {v(V_T0)}", f"v_mov_b32 {v(V_T3)}, {v(V_T2)}",
                "NOP2",
                f"v_permlane32_swap_b32 {v(V_T0)}, {v(V_T1)}", f"v_permlane32_swap_b32 {v(V_T2)}, {v(V_T3)}",
                f"v_max_f32 {v(V_T0)}, {v(V_T0)}, {v(V_T1)}",     # raw max, keys 0-63
                f"v_max_f32 {v(V_T2)}, {v(V_T2)}, {v(V_T3)}",     # raw max, keys 64-127
                f"v_max_f32 {v(V_T1)}, {v(V_T0)}, {v(V_T2)}",
                f"v_mul_f32 {v(V_T1)}, {sl2}, {v(V_T1)}"]         # tile max (log2 units)
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
        sb, p2 = SB[wpar], P2[wpar]
        for raw, sbn, p2n in ((V_T0, sb[0], p2[0]), (V_T2, sb[1], p2[1])):
            # off = floor(half max - mref) in [-127, 0] (the vote keeps the codes in e4m3 range); scale = 2^off
            tail += [f"v_fma_f32 {v(raw)}, {v(raw)}, {sl2}, -{v(V_MREF)}",
                     f"v_floor_f32 {v(raw)}, {v(raw)}",
                     f"v_med3_f32 {v(raw)}, {v(raw)}, {s(S_M127)}, 0",
                     f"v_cvt_i32_f32 {v(sbn)}, {v(raw)}",
                     f"v_add_u32_e32 {v(sbn)}, 0x7f, {v(sbn)}",
                     f"v_lshlrev_b32 {v(p2n)}, 23, {v(sbn)}"]
        # code bias: b8 = b8k - 8 mref - 8 off
        tail += [f"v_fma_f32 {v(V_T1)}, {v(V_MREF)}, {v(V_M8)}, {arg('B8K')}",
                 f"v_fma_f32 {v(V_B8LO)}, {v(V_T0)}, {v(V_M8)}, {v(V_T1)}",
                 f"v_fma_f32 {v(V_B8HI)}, {v(V_T2)}, {v(V_M8)}, {v(V_T1)}"]
        return groups, tail

    def mask_thr_instrs(self):
        """S_T3 = q_start - 128 t for the tile t = k + 1 being masked; V_THR = S_T3 + c32 - 4 gb."""
        return [f"s_add_u32 {s(S_T3)}, {s(S_K)}, 1",
                f"s_lshl_b32 {s(S_T3)}, {s(S_T3)}, 7",
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
        if blk == 3:
            # the fully visible path reaches the S reads of the statistics soon after the QK MFMA: pad it
            lj = self.L("mjoin")
            out += [f"s_branch {lj}", f"LABEL {lv}", "s_nop 15", "s_nop 3", f"LABEL {lj}", f"LABEL {le}",
                    "DROP vis", "NOP2"]
        else:
            # the fully visible path skips the elementwise path: count its windows from the branch
            out += [f"LABEL {le}", f"LABEL {lv}", "MERGE vis", "NOP2"]
        return ["\n".join(out)]

    def codes(self):
        """e4m3 codes of P(k) in place: the byte RNE(s * c8 + b8), saturated by v_cvt_pk_u8_f32, is the e4m3 code of
        a piecewise-linear 2^x. Interleaved: the fmas of block b with the conversions of block b - 1."""
        g = self.g
        c8 = arg("C8")
        fmas = []
        for blk in range(4):
            b8 = V_B8LO if blk < 2 else V_B8HI
            for r in range(16):
                fmas.append(f"v_fma_f32 {v(S[blk] + r)}, {v(S[blk] + r)}, {c8}, {v(b8)}")
        cvts = []
        for blk in range(4):
            pbase = (P_LO if blk < 2 else P_HI) + 4 * (blk & 1)
            lst = []
            for b in range(4):
                for j in range(4):
                    acc = "0" if b == 0 else v(pbase + j)
                    lst.append(f"v_cvt_pk_u8_f32 {v(pbase + j)}, {v(S[blk] + 4 * j + b)}, {b}, {acc}")
            cvts.append(lst)
        for blk in range(4):
            prev = cvts[blk - 1] if blk else []
            for i in range(16):
                g.emit(fmas[16 * blk + i])
                if i < len(prev):
                    g.emit(prev[i])
        for t in cvts[3]:
            g.emit(t)

    # ------------------------------------------------------------ phases
    def seg_sm(self, kb_next):
        """SM(k): codes of P(k), then the first K(k+1) fragment reads from ring slot kb_next (None: runtime slot,
        False: no next tile)."""
        self.g.emit("s_setprio 0")
        self.codes()
        if kb_next is not False:
            self.pre_read_k(kb_next)

    def seg_mm(self, variant, kb, vb, par, loads):
        """MM(k): row sums of P(k), QK(k+1), PV(k), and under the MFMAs the statistics of S(k+1).
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
                ([("V", h, j) for h in (0, 1) for j in range(4)] if with_pv else [])
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
                    self.read_vfrag(nxt % NSLOT, f[1], f[2], vb)
                    lq.push(nxt, 4)
                nxt += 1

        mf = [("R", 0), ("R", 1)] if with_pv else []
        if with_qk:
            mf += [("QK", b, d, i) for i, (b, d) in enumerate(QK_ORDER)]
        if with_pv:
            f0 = len(QK_ORDER) if with_qk else 0
            mf += [("PV", h, j, f0 + 4 * h + j) for h in (0, 1) for j in range(4)]
        # VALU plan: block b's statistics go after the MFMA that follows its last QK, at most `cap` per MFMA slot
        valu_slots = [[] for _ in mf]
        flat = []
        if with_qk:
            groups, tail = self.stats_instrs(first, 1 - par)
            idx_of_qk = {(m[1], m[2]): i for i, m in enumerate(mf) if m[0] == "QK"}
            if masked:
                flat += [(0, t) for t in self.mask_thr_instrs()]
            for bi, lst in enumerate(groups):
                start = idx_of_qk[(bi, 1)] + 1
                flat += [(start, t) for t in (self.mask_instrs(bi) if masked else []) + lst]
            flat += [(0, t) for t in tail]
        last_slot = len(mf) - 1
        cap = VALU_PER_SLOT * (3 if masked else 1)
        p = 0
        for e, t in flat:
            p = max(p, min(e, last_slot))
            while p < last_slot and sum(1 for x in valu_slots[p] if x != "NOP2") >= cap:
                p += 1
            valu_slots[p].append(t)
        if with_pv:
            # l = (l + r_lo * 2^off_lo) + r_hi * 2^off_hi, the HIP kernel's order of operations
            tgt = min(len(mf) - 1, 4)
            valu_slots[tgt] = [f"v_mul_f32 {v(V_TMP)}, {v(RS_LO)}, {v(P2[par][0])}",
                               f"v_add_f32 {v(V_L)}, {v(V_L)}, {v(V_TMP)}",
                               f"v_mul_f32 {v(V_TMP)}, {v(RS_HI)}, {v(P2[par][1])}",
                               f"v_add_f32 {v(V_L)}, {v(V_L)}, {v(V_TMP)}"] + valu_slots[tgt]
        for i, m in enumerate(mf):
            if m[0] == "R":
                src, dst = (P_LO, RS_LO) if m[1] == 0 else (P_HI, RS_HI)
                g.emit(f"v_mfma_f32_16x16x128_f8f6f4 {vv(dst, 4)}, {vv(SEL, 8)}, {vv(src, 8)}, 0")
                if i == 1:
                    refill(DEPTH - 1)
            else:
                f = m[3]
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
                    g.emit(f"v_mfma_f32_32x32x64_f8f6f4 {sd}, {a}, {vv(Q0 if d == 0 else Q1, 8)}, {acc}")
                else:
                    h, j = m[1], m[2]
                    od = vv(O[j], 16)
                    g.emit(f"v_mfma_scale_f32_32x32x64_f8f6f4 {od}, {a}, {vv(P_LO if h == 0 else P_HI, 8)}, {od}, "
                           f"{v(V_127)}, {v(SB[par][h])} op_sel_hi:[0,0,0]")
                refill(f + DEPTH)
            if loads is not None:
                lg = self.loads_rt() if loads == "rt" else self.loads_static(*loads)
                if 0 <= i - LOAD_AT < len(lg):
                    self.emit_list(lg[i - LOAD_AT])
            self.emit_list(valu_slots[i])
        assert nxt == len(frags), (nxt, len(frags))
        lq.drain()
        if with_qk and not first:
            # rescale voted by this MM's statistics: O *= corr as packed muls (the other wave is in SM: no MFMAs)
            lskip = self.L("norescale")
            g.emit(f"s_cmp_eq_u32 {s(S_FLAG)}, 0")
            g.emit(f"s_cbranch_scc1 {lskip}")
            g.emit(f"v_sub_f32 {v(V_CORR)}, {v(V_M)}, {v(V_MREF)}")
            g.emit(f"v_exp_f32 {v(V_CORR)}, {v(V_CORR)}")
            g.emit(f"v_mov_b32 {v(V_M)}, {v(V_MREF)}")
            for j in range(4):
                for r in range(0, 16, 2):
                    g.emit(f"v_pk_mul_f32 {vv(O[j] + r, 2)}, {vv(O[j] + r, 2)}, {vv(V_CORR, 2)} "
                           f"op_sel:[0,0] op_sel_hi:[1,0]")
            g.emit(f"v_mul_f32 {v(V_L)}, {v(V_CORR)}, {v(V_L)}")
            g.label(lskip)
        if loads not in (None, "rt"):
            g.emit("s_waitcnt vmcnt(4)")

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
        # LSE (natural log) = (m - 6 + log2 l) * ln 2
        g.emit(f"v_add_f32 {v(V_LSE)}, {v(V_M)}, {v(V_LSE)}")
        g.emit(f"v_add_f32 {v(V_LSE)}, -4.0, {v(V_LSE)}")
        g.emit(f"v_add_f32 {v(V_LSE)}, -2.0, {v(V_LSE)}")
        g.emit(f"v_mul_f32 {v(V_LSE)}, 0x3f317218, {v(V_LSE)}")
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
        # store addresses in P_LO.. (free until the next SM's codes)
        g.emit(f"v_mov_b32 {v(P_LO)}, {v(V_EA)}")
        for k in range(1, 8):
            g.emit(f"v_add_u32 {v(P_LO + k)}, {s(S_T1)}, {v(P_LO + k - 1)}")
        for k in range(8):
            g.emit(f"s_waitcnt lgkmcnt({7 - k})")
            g.emit(f"global_store_dwordx4 {v(P_LO + k)}, {vv(O[2] + 4 * k, 4)}, {ss(S_EOH, 2)}")
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
            g.emit(f"s_mul_i32 {s(S_T1)}, {t}, {s(S_KVT)}")
            for p, voff in enumerate((V_GK0, V_GK1) if kind == "K" else (V_GV0, V_GV1)):
                if kind == "K":
                    g.emit(f"s_add_u32 m0, {s(S_M0K)}, {LDS_K[t] + 8192 * p}")
                    g.emit(f"buffer_load_dwordx4 {v(voff)}, {ss(S_KR, 4)}, {s(S_T1)} offen lds")
                else:
                    g.emit(f"s_add_u32 m0, {s(S_M0V)}, {16384 * t + 8192 * p}")
                    g.emit(f"buffer_load_dwordx4 {v(voff)}, {ss(S_VR, 4)}, {s(S_T1)} offen lds")
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
        g.emit(f"v_add_u32 {v(V_TKA0)}, {s(S_T0)}, {v(V_KA0)}")
        g.emit(f"v_add_u32 {v(V_TKA1)}, {s(S_T0)}, {v(V_KA1)}")
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
        for a, b in ((SB[1][0], SB[0][0]), (SB[1][1], SB[0][1]), (P2[1][0], P2[0][0]), (P2[1][1], P2[0][1])):
            g.emit(f"v_mov_b32 {v(a)}, {v(b)}")
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
        # lmm / ltmm / ltrans mark the MM phases and the transition in the listing (no branch targets them)
        lmm = [self.L(f"mm{u}") for u in range(4)]
        ltmm = [self.L("tmm0"), self.L("tmm1")]
        ltrans = [self.L("trans0"), self.L("trans1")]
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
            g.label(lmm[u])
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
            g.label(ltmm[par])
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
            g.label(ltrans[par])
            self.transition(par, with_pv=True)
            g.emit(f"s_branch {ljoin}")
            g.label(ldead_last)
            g.emit("s_waitcnt vmcnt(0)")
            self.bar()
            self.transition(par, with_pv=False)
            g.label(ljoin)
            g.emit("s_waitcnt vmcnt(4)")     # the next pass's tiles have landed; its 4 Q loads are waited before QK0
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


def generate(name="fmha_fp8pp"):
    """Returns the assembly text and the list of hazard paddings inserted."""
    k = Kernel()
    k.gen()
    return kernel_text(name, k.g.text(), LDS_BYTES + LDS_EPI), k.g.pads


if __name__ == "__main__":
    main(generate, "fmha_fp8_d128_pp.s")
