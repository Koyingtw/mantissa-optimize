"""FP8 (E4M3) causal flash attention for Hopper, Triton >= 3.5 (TMA tensor descriptors), with a
Delta-Matching backward.  v2 of fp8attn.py (same math, faster plumbing).

Inputs: q [S, HQ, D], k/v [S, HKV, D] bf16 (the sbhd tensors of Megatron/TE with B == 1 squeezed).

Quantization (all E4M3):
  * Q, K, V, dO: one fp32 dequant scale per (head, 128-row block) (`quantize`).
  * P: fixed scale 256 (P <= 1), with the V/K/dO per-block scale folded in so that the
    accumulator never needs a per-iteration rescale.
  * dS: one scale per output row, fixed for the whole loop (DS_SCALE=1, a Cauchy-Schwarz bound
    computed from row norms), or per-row amax per tile (DS_SCALE=0, needs an extra FMA per
    accumulator element per tile).

Backward = quantize kernels + dq kernel (Q-outer) + dkdv kernel (KV-outer, loops over the GROUP
query heads sharing a KV head; no atomics anywhere).

Delta-Matching: the dq kernel accumulates delta_i = sum_j P_ij dP^_ij (P recomputed in FP32,
dP^ = dO^ V^T with FP8 operands / FP32 accumulation) while it streams over KV, and uses linearity
to finish dQ in the same pass:
    dQ_i = sum_j P_ij (dP^_ij - delta0_i) k_j  -  (delta_i - delta0_i) * sum_j P_ij k_j
delta0 = <dO, O> only centres dS so that the FP8 quantization of the first term is well scaled.
The dkdv kernel then uses the matched delta directly: dS = P (dP^ - delta).
MATCHED=False is the stale-delta control (delta = delta0, no P K product).
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _desc(t, rows_blk, cols_blk):
    """2D TMA descriptor over a contiguous [A, B, C] fp8 tensor viewed as [A*B, C]."""
    t2 = t.reshape(-1, t.shape[-1])
    return TensorDescriptor.from_tensor(t2, [rows_blk, cols_blk])

FP8 = torch.float8_e4m3fn
QBLK = 128
LOG2E = 1.4426950408889634
LAST = {}          # name -> (n_regs, n_spills, shared) of the last launch, for reporting


def _alloc(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc)


def _note(name, k):
    try:
        LAST[name] = (k.n_regs, k.n_spills, k.metadata.shared)
    except Exception:  # noqa
        pass


# --------------------------------------------------------------------------------------------
@triton.jit
def _quant_kernel(X, X8, X8T, SC, NRM, O, DELTA0, ROWMUL,
                  stride_xs, stride_xh, stride_os, stride_oh, S,
                  D: tl.constexpr, BLK: tl.constexpr, WRITE_T: tl.constexpr,
                  WITH_DELTA: tl.constexpr, WITH_NRM: tl.constexpr, WRITE_N: tl.constexpr,
                  WITH_ROWMUL: tl.constexpr):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = pid_s * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    x = tl.load(X + offs_s[:, None] * stride_xs + h * stride_xh + offs_d[None, :]).to(tl.float32)
    if WITH_ROWMUL:
        x = x * tl.load(ROWMUL + h * S + offs_s)[:, None]
    amax = tl.max(tl.max(tl.abs(x), 1), 0)
    sc = tl.maximum(amax, 1e-30) / 448.0
    x8 = (x * (1.0 / sc)).to(tl.float8e4nv)
    base = h.to(tl.int64) * S * D
    if WRITE_N:
        tl.store(X8 + base + offs_s[:, None] * D + offs_d[None, :], x8)
    if WRITE_T:
        tl.store(X8T + base + offs_d[None, :] * S + offs_s[:, None], x8)
    tl.store(SC + h * (S // BLK) + pid_s, sc)
    if WITH_NRM:
        tl.store(NRM + h * S + offs_s, tl.sqrt(tl.sum(x * x, 1)))
    if WITH_DELTA:
        o = tl.load(O + offs_s[:, None] * stride_os + h * stride_oh + offs_d[None, :]).to(tl.float32)
        tl.store(DELTA0 + h * S + offs_s, tl.sum(o * x, 1))


def quantize(x, write_t=True, o=None, nrm=False, rowmul=None, write_n=True):
    S_, H, D_ = x.shape
    assert x.stride(2) == 1 and S_ % QBLK == 0
    x8 = torch.empty(H, S_, D_, device=x.device, dtype=FP8) if write_n else None
    x8t = torch.empty(H, D_, S_, device=x.device, dtype=FP8) if write_t else x8
    if x8 is None:
        x8 = x8t
    sc = torch.empty(H, S_ // QBLK, device=x.device, dtype=torch.float32)
    delta0 = torch.empty(H, S_, device=x.device, dtype=torch.float32) if o is not None else sc
    nr = torch.empty(H, S_, device=x.device, dtype=torch.float32) if nrm else sc
    oo = o if o is not None else x
    k = _quant_kernel[(S_ // QBLK, H)](x, x8, x8t, sc, nr, oo, delta0, rowmul if rowmul is not None else sc,
                                       x.stride(0), x.stride(1), oo.stride(0), oo.stride(1), S_,
                                       D=D_, BLK=QBLK, WRITE_T=write_t, WITH_DELTA=o is not None,
                                       WITH_NRM=nrm, WRITE_N=write_n, WITH_ROWMUL=rowmul is not None,
                                       num_warps=8)
    _note("quant", k)
    return x8, x8t, sc, (delta0 if o is not None else None), (nr if nrm else None)


# --------------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------------
@triton.jit
def _fwd_inner(acc, l_i, m_i, q8, dk_, dvt, SK, SV, svmax, g, c0, lo, hi, offs_m, offs_n, S, NQB,
               D: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr, MASK: tl.constexpr,
               WS: tl.constexpr):
    for start_n in tl.range(lo, hi, BN, warp_specialize=WS):
        kT = dk_.load([g * S + start_n, 0]).T                                # [D, BN]
        sk = tl.load(SK + g * NQB + start_n // QB)
        sv = tl.load(SV + g * NQB + start_n // QB)
        s = tl.dot(q8, kT) * (c0 * sk)
        if MASK:
            s = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        acc = acc * alpha[:, None]
        v = dvt.load([g * D, start_n]).T                                     # [BN, D] K-major
        p8 = (p * (256.0 * sv / svmax)).to(tl.float8e4nv)
        acc = tl.dot(p8, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(dq_, dk_, dvt, SQ, SK, SV, SVMAX, Out, LSE, MROW,
                stride_os, stride_oh, S, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr, WS: tl.constexpr):
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
    h = tl.program_id(1)
    g = h // GROUP
    NQB = S // QB
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    q8 = dq_.load([h * S + pid_m * BM, 0])
    sq = tl.load(SQ + h * NQB + (pid_m * BM) // QB)
    svmax = tl.load(SVMAX + g)
    m_i = tl.full([BM], -1.0e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    c0 = sq * qk_scale
    lo_diag = (pid_m * BM) // BN * BN
    acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q8, dk_, dvt, SK, SV, svmax, g, c0, 0, lo_diag,
                               offs_m, offs_n, S, NQB, D, BN, QB, False, WS)
    acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q8, dk_, dvt, SK, SV, svmax, g, c0, lo_diag,
                               (pid_m + 1) * BM, offs_m, offs_n, S, NQB, D, BN, QB, True, False)
    o = acc * (svmax / 256.0) / l_i[:, None]
    offs_d = tl.arange(0, D)
    tl.store(Out + offs_m[:, None] * stride_os + h * stride_oh + offs_d[None, :], o.to(tl.bfloat16))
    tl.store(LSE + h * S + offs_m, m_i + tl.math.log2(l_i))
    tl.store(MROW + h * S + offs_m, m_i)


# --------------------------------------------------------------------------------------------
# Backward dq (Q-outer)
# --------------------------------------------------------------------------------------------
@triton.jit
def _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, dk_, dv_, dkt, SK, SV, skmax, g, c0, sdo,
              lse, d0, rrow, lo, hi, offs_m, offs_n, S, NQB, D: tl.constexpr, BN: tl.constexpr,
              QB: tl.constexpr, MATCHED: tl.constexpr, WRITE_RES: tl.constexpr,
              DS_SCALE: tl.constexpr, MASK: tl.constexpr, WS: tl.constexpr):
    for start_n in tl.range(lo, hi, BN, warp_specialize=WS):
        kT = dk_.load([g * S + start_n, 0]).T
        vT = dv_.load([g * S + start_n, 0]).T
        sk = tl.load(SK + g * NQB + start_n // QB)
        sv = tl.load(SV + g * NQB + start_n // QB)
        s = tl.dot(q8, kT)
        if DS_SCALE == 3:
            # lazy per-row scale: rrow only ever shrinks (power-of-2 steps) when a tile would overflow
            p_rs = tl.math.exp2(s * (c0 * sk) - (lse - tl.math.log2(rrow * (sk / skmax)))[:, None])
            if MASK:
                p_rs = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], p_rs, 0.0)
            dpr = tl.dot(do8, vT)
            t = dpr * (sdo * sv) - d0[:, None]
            dsr = p_rs * t
            amax = tl.max(tl.abs(dsr), 1)
            if tl.max(amax, 0) > 448.0:
                f = tl.where(amax > 448.0, tl.math.exp2(tl.floor(tl.math.log2(448.0 / amax))), 1.0)
                dsr = dsr * f[:, None]
                p_rs = p_rs * f[:, None]
                accA = accA * f[:, None]
                rrow = rrow * f
            inv = (skmax / sk) / rrow
            if MATCHED:
                dacc += tl.sum(p_rs * dpr, 1) * (sdo * sv) * inv
            if WRITE_RES:
                rsum += tl.sum(dsr, 1) * inv
                psum += tl.sum(p_rs, 1) * inv
            kb = dkt.load([g * D, start_n]).T
            accA = tl.dot(dsr.to(tl.float8e4nv), kb, accA)
            if MATCHED:
                accB = tl.dot((p_rs * (256.0 / rrow)[:, None]).to(tl.float8e4nv), kb, accB)
        elif DS_SCALE == 2:
            # folded: p_rs = P * rrow * sk/skmax straight out of exp2 (lse here is lse - log2(rrow))
            p_rs = tl.math.exp2(s * (c0 * sk) - (lse - tl.math.log2(sk / skmax))[:, None])
            if MASK:
                p_rs = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], p_rs, 0.0)
            dpr = tl.dot(do8, vT)
            t = dpr * (sdo * sv) - d0[:, None]
            dsr = p_rs * t
            if MATCHED:
                dacc += tl.sum(p_rs * dpr, 1) * (sdo * sv * skmax / sk)
            if WRITE_RES:
                rsum += tl.sum(dsr, 1) * (skmax / sk)
                psum += tl.sum(p_rs, 1) * (skmax / sk)
            kb = dkt.load([g * D, start_n]).T
            accA = tl.dot(dsr.to(tl.float8e4nv), kb, accA)
            if MATCHED:
                accB = tl.dot((p_rs * rrow[:, None]).to(tl.float8e4nv), kb, accB)
        else:
            p = tl.math.exp2(s * (c0 * sk) - lse[:, None])
            if MASK:
                p = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], p, 0.0)
            dp = tl.dot(do8, vT) * (sdo * sv)
            if MATCHED:
                dacc += tl.sum(p * dp, 1)
            ds = p * (dp - d0[:, None])
            if WRITE_RES:
                rsum += tl.sum(ds, 1)
                psum += tl.sum(p, 1)
            kb = dkt.load([g * D, start_n]).T                                    # [BN, D] K-major
            if DS_SCALE == 0:
                amax = tl.max(tl.abs(ds), 1)
                r = 448.0 / tl.maximum(amax, 1e-30)
                ds8 = (ds * r[:, None]).to(tl.float8e4nv)
                accA += tl.dot(ds8, kb) * (sk / r)[:, None]
            else:
                ds8 = (ds * (rrow * (sk / skmax))[:, None]).to(tl.float8e4nv)
                accA = tl.dot(ds8, kb, accA)
            if MATCHED:
                p8 = (p * (256.0 * sk / skmax)).to(tl.float8e4nv)
                accB = tl.dot(p8, kb, accB)
    return accA, accB, dacc, rsum, psum, rrow


@triton.jit
def _bwd_dq_kernel(dq_, ddo, dk_, dv_, dkt, SQ, SK, SKMAX, SV, SDO, LSE, MROW, DELTA0, NDO, VNMAX,
                   DELTA, DQ, RES, stride_dqs, stride_dqh, S, qk_scale, sm_scale,
                   HQ: tl.constexpr, HKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr, MATCHED: tl.constexpr,
                   WRITE_RES: tl.constexpr, DS_SCALE: tl.constexpr, WS: tl.constexpr):
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
    h = tl.program_id(1)
    g = h // GROUP
    NQB = S // QB
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    q8 = dq_.load([h * S + pid_m * BM, 0])
    do8 = ddo.load([h * S + pid_m * BM, 0])
    sq = tl.load(SQ + h * NQB + (pid_m * BM) // QB)
    sdo = tl.load(SDO + h * NQB + (pid_m * BM) // QB)
    skmax = tl.load(SKMAX + g)
    lse = tl.load(LSE + h * S + offs_m)
    d0 = tl.load(DELTA0 + h * S + offs_m)
    if DS_SCALE >= 1:
        # |dS_ij| <= Pmax_i (|dO_i| |v|max + |delta0_i|) (Cauchy-Schwarz), x1.25 margin for FP8 rounding
        pmax = tl.math.exp2(tl.load(MROW + h * S + offs_m) - lse)
        bound = pmax * (tl.load(NDO + h * S + offs_m) * tl.load(VNMAX + g) + tl.abs(d0)) * 1.25
        rrow = 448.0 / tl.maximum(bound, 1e-30)
    else:
        rrow = lse
    if DS_SCALE == 3:
        rrow = rrow * 256.0          # optimistic start, x2^8 over the Cauchy-Schwarz bound
    rrow_keep = rrow
    bsc = tl.zeros([BM], tl.float32) + 1.0
    if DS_SCALE == 2:
        lse = lse - tl.math.log2(rrow)
        bsc = pmax                       # accB is built from P' = P / pmax (row max 1)
        rrow = 256.0 / (rrow * pmax)
    accA = tl.zeros([BM, D], tl.float32)
    accB = tl.zeros([BM, D], tl.float32)
    dacc = tl.zeros([BM], tl.float32)
    rsum = tl.zeros([BM], tl.float32)
    psum = tl.zeros([BM], tl.float32)
    c0 = sq * qk_scale
    lo_diag = (pid_m * BM) // BN * BN
    accA, accB, dacc, rsum, psum, rrow = _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, dk_, dv_, dkt, SK, SV,
                                             skmax, g, c0, sdo, lse, d0, rrow, 0, lo_diag, offs_m, offs_n,
                                             S, NQB, D, BN, QB, MATCHED, WRITE_RES, DS_SCALE, False, WS)
    accA, accB, dacc, rsum, psum, rrow = _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, dk_, dv_, dkt, SK, SV,
                                             skmax, g, c0, sdo, lse, d0, rrow, lo_diag, (pid_m + 1) * BM,
                                             offs_m, offs_n, S, NQB, D, BN, QB, MATCHED, WRITE_RES, DS_SCALE,
                                             True, False)
    if DS_SCALE == 3:
        rrow_keep = rrow
    if DS_SCALE >= 1:
        accA = accA * (skmax / rrow_keep)[:, None]
    if DS_SCALE == 2:
        dacc = dacc / rrow_keep
        rsum = rsum / rrow_keep
        psum = psum / rrow_keep
    if MATCHED:
        eps = dacc - d0
        dq = accA - eps[:, None] * (accB * (skmax / 256.0) * bsc[:, None])
        tl.store(DELTA + h * S + offs_m, dacc)
        if WRITE_RES:
            rsum = rsum - eps * psum
    else:
        dq = accA
    dq = dq * sm_scale
    offs_d = tl.arange(0, D)
    tl.store(DQ + offs_m[:, None] * stride_dqs + h * stride_dqh + offs_d[None, :], dq.to(tl.bfloat16))
    if WRITE_RES:
        tl.store(RES + h * S + offs_m, rsum)


# --------------------------------------------------------------------------------------------
# Backward dK dV (KV-outer, loops over the GROUP query heads of the KV head)
# --------------------------------------------------------------------------------------------
@triton.jit
def _kv_inner(accK, accV, rrow, k8, v8, dq_, ddo, dqt, ddot, SQ, SDO, SQT, SDOT, LSE, DELTA, sqh, sdoh,
              h, c0, sv, lo, hi, offs_n, S, NQB, D: tl.constexpr, BM: tl.constexpr, QB: tl.constexpr,
              DS_SCALE: tl.constexpr, MASK: tl.constexpr, WS: tl.constexpr):
    # Units: true dV = accV * sdoh/256 ; true dK = accK * sqh/rrow (DS_SCALE >= 1) or accK (DS_SCALE 0)
    for start_m in tl.range(lo, hi, BM, warp_specialize=WS):
        rows = start_m + tl.arange(0, BM)
        qT = dq_.load([h * S + start_m, 0]).T                               # [D, BM]
        doT = ddo.load([h * S + start_m, 0]).T
        sq = tl.load(SQ + h * NQB + start_m // QB)          # scale of q8 (S = K Q^T)
        sdo = tl.load(SDO + h * NQB + start_m // QB)        # scale of do8 (dP = V dO^T)
        sqt = tl.load(SQT + h * NQB + start_m // QB)        # scale of (q / l) in q8t   (dK operand)
        sdot = tl.load(SDOT + h * NQB + start_m // QB)      # scale of (dO / l) in do8t (dV operand)
        lse = tl.load(LSE + h * S + rows)                   # here: the row max m_i, so pT = P * l_i <= 1
        dl = tl.load(DELTA + h * S + rows)
        sT = tl.dot(k8, qT)
        pT = tl.math.exp2(sT * (c0 * sq) - lse[None, :])
        if MASK:
            pT = tl.where(rows[None, :] >= offs_n[:, None], pT, 0.0)
        dpT = tl.dot(v8, doT) * (sv * sdo)
        dsT = pT * (dpT - dl[None, :])
        dob = ddot.load([h * D, start_m]).T                                 # [BM, D] K-major
        p8 = (pT * (256.0 * sdot / sdoh)).to(tl.float8e4nv)
        accV = tl.dot(p8, dob, accV)
        qb = dqt.load([h * D, start_m]).T
        if DS_SCALE == 0:
            amax = tl.max(tl.abs(dsT), 1)
            r = 448.0 / tl.maximum(amax, 1e-30)
            ds8 = (dsT * r[:, None]).to(tl.float8e4nv)
            accK += tl.dot(ds8, qb) * (sqt / r)[:, None]
        elif DS_SCALE == 3:
            dsr = dsT * (rrow * (sqt / sqh))[:, None]
            amax = tl.max(tl.abs(dsr), 1)
            if tl.max(amax, 0) > 448.0:
                f = tl.where(amax > 448.0, tl.math.exp2(tl.floor(tl.math.log2(448.0 / amax))), 1.0)
                dsr = dsr * f[:, None]
                accK = accK * f[:, None]
                rrow = rrow * f
            accK = tl.dot(dsr.to(tl.float8e4nv), qb, accK)
        else:
            ds8 = (dsT * (rrow * (sqt / sqh))[:, None]).to(tl.float8e4nv)
            accK = tl.dot(ds8, qb, accK)
    return accK, accV, rrow


@triton.jit
def _bwd_dkdv_kernel(dq_, dqt, dk_, dv_, ddo, ddot, SQ, SQT, SQH, SK, SV, SDO, SDOT, SDOH, LSE, DELTA,
                     VNRM, NDOH, DH, DK, DV, stride_dks, stride_dkh, S, qk_scale, sm_scale,
                     HQ: tl.constexpr, HKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr, DS_SCALE: tl.constexpr,
                     WS: tl.constexpr):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    NQB = S // QB
    offs_n = pid_n * BN + tl.arange(0, BN)
    k8 = dk_.load([g * S + pid_n * BN, 0])
    v8 = dv_.load([g * S + pid_n * BN, 0])
    sk = tl.load(SK + g * NQB + (pid_n * BN) // QB)
    sv = tl.load(SV + g * NQB + (pid_n * BN) // QB)
    vn = tl.load(VNRM + g * S + offs_n)
    accK = tl.zeros([BN, D], tl.float32)
    accV = tl.zeros([BN, D], tl.float32)
    c0 = sk * qk_scale
    lo = (pid_n * BN) // BM * BM
    hi_diag = tl.minimum(((pid_n + 1) * BN + BM - 1) // BM * BM, S)
    rrow = tl.zeros([BN], tl.float32) + 1.0
    sqh_prev = 1.0
    sdoh_prev = 1.0
    for hh in range(GROUP):
        h = g * GROUP + hh
        sqh = tl.load(SQH + h)
        sdoh = tl.load(SDOH + h)
        # per-head units for the accumulators (rescale what has been accumulated so far)
        if DS_SCALE == 1 or DS_SCALE == 2:
            # |dS_ij| <= |dO_i| |v_j| + |delta_i|   (P <= 1), x1.25 margin, per query head
            rnew = 448.0 / tl.maximum((tl.load(NDOH + h) * vn + tl.load(DH + h)) * 1.25, 1e-30)
            if hh > 0:
                accK = accK * ((rnew / sqh) / (rrow / sqh_prev))[:, None]
            rrow = rnew
        elif DS_SCALE == 3:
            if hh == 0:   # optimistic start (x2^8 over the bound); overflow is caught per tile
                rrow = 256.0 * 448.0 / tl.maximum((tl.load(NDOH + h) * vn + tl.load(DH + h)) * 1.25, 1e-30)
            else:
                accK = accK * (sqh_prev / sqh)
        if hh > 0:
            accV = accV * (sdoh_prev / sdoh)
        accK, accV, rrow = _kv_inner(accK, accV, rrow, k8, v8, dq_, ddo, dqt, ddot, SQ, SDO, SQT, SDOT, LSE, DELTA, sqh,
                                     sdoh, h, c0, sv, lo, hi_diag, offs_n, S, NQB, D, BM, QB, DS_SCALE, True, False)
        accK, accV, rrow = _kv_inner(accK, accV, rrow, k8, v8, dq_, ddo, dqt, ddot, SQ, SDO, SQT, SDOT, LSE, DELTA, sqh,
                                     sdoh, h, c0, sv, hi_diag, S, offs_n, S, NQB, D, BM, QB, DS_SCALE, False, WS)
        sqh_prev = sqh
        sdoh_prev = sdoh
    if DS_SCALE >= 1:
        accK = accK * (sqh_prev / rrow)[:, None]
    dk = accK * sm_scale
    dv = accV * (sdoh_prev / 256.0)
    offs_d = tl.arange(0, D)
    tl.store(DK + offs_n[:, None] * stride_dks + g * stride_dkh + offs_d[None, :], dk.to(tl.bfloat16))
    tl.store(DV + offs_n[:, None] * stride_dks + g * stride_dkh + offs_d[None, :], dv.to(tl.bfloat16))


# --------------------------------------------------------------------------------------------
CFG = dict(fwd_BM=128, fwd_BN=64, fwd_warps=4, fwd_stages=3, fwd_WS=False,
           dq_BM=64, dq_BN=64, dq_warps=4, dq_stages=3, dq_WS=False,
           kv_BM=64, kv_BN=128, kv_warps=8, kv_stages=3, kv_WS=False, DS_SCALE=2)


def fp8_attn_fwd(q, k, v, sm_scale=None, cfg=CFG):
    S_, HQ, D_ = q.shape
    HKV = k.shape[1]
    sm_scale = sm_scale or D_ ** -0.5
    q8, _, sq, _, _ = quantize(q, write_t=False)
    k8, _, sk, _, _ = quantize(k, write_t=False)
    _, v8t, sv, _, _ = quantize(v, write_t=True)
    o = torch.empty(S_, HQ, D_, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(HQ, S_, device=q.device, dtype=torch.float32)
    mrow = torch.empty_like(lse)
    BM, BN = cfg["fwd_BM"], cfg["fwd_BN"]
    kk = _fwd_kernel[(S_ // BM, HQ)](_desc(q8, BM, D_), _desc(k8, BN, D_), _desc(v8t, D_, BN), sq, sk, sv, sv.amax(1).contiguous(), o, lse, mrow,
                                     o.stride(0), o.stride(1), S_, sm_scale * LOG2E,
                                     HQ=HQ, HKV=HKV, GROUP=HQ // HKV, D=D_, BM=BM, BN=BN, QB=QBLK,
                                     WS=cfg["fwd_WS"], num_warps=cfg["fwd_warps"],
                                     num_stages=cfg["fwd_stages"])
    _note("fwd", kk)
    return o, lse, mrow


def fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=None, matched=True, cfg=CFG, return_res=False):
    S_, HQ, D_ = q.shape
    HKV = k.shape[1]
    GROUP = HQ // HKV
    sm_scale = sm_scale or D_ ** -0.5
    ds_scale = cfg["DS_SCALE"]
    q8, _, sq, _, _ = quantize(q, write_t=False)
    k8, k8t, sk, _, _ = quantize(k, write_t=True)
    v8, _, sv, _, vnrm = quantize(v, write_t=False, nrm=True)
    do8, _, sdo, delta0, ndo = quantize(do, write_t=False, o=o, nrm=True)
    # dK / dV operands: rows pre-multiplied by 1/l_i = exp2(m_i - lse_i), so the kv kernel can use
    # P' = exp2(s - m_i) (row max exactly 1) instead of the tiny normalized P of diffuse rows.
    pmax = torch.exp2(mrow - lse)
    _, q8t, sqt, _, _ = quantize(q, write_t=True, rowmul=pmax, write_n=False)
    _, do8t, sdot, _, _ = quantize(do, write_t=True, rowmul=pmax, write_n=False)
    dq = torch.empty_like(q)
    dk = torch.empty(S_, HKV, D_, device=q.device, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    delta = torch.empty(HQ, S_, device=q.device, dtype=torch.float32) if matched else delta0
    res = torch.empty(HQ, S_, device=q.device, dtype=torch.float32) if return_res else delta
    BM, BN = cfg["dq_BM"], cfg["dq_BN"]
    kk = _bwd_dq_kernel[(S_ // BM, HQ)](_desc(q8, BM, D_), _desc(do8, BM, D_), _desc(k8, BN, D_),
                                        _desc(v8, BN, D_), _desc(k8t, D_, BN), sq, sk, sk.amax(1).contiguous(), sv, sdo,
                                        lse, mrow, delta0, ndo, vnrm.amax(1).contiguous(), delta, dq, res,
                                        dq.stride(0), dq.stride(1), S_, sm_scale * LOG2E, sm_scale,
                                        HQ=HQ, HKV=HKV, GROUP=GROUP, D=D_, BM=BM, BN=BN, QB=QBLK,
                                        MATCHED=matched, WRITE_RES=return_res, DS_SCALE=ds_scale,
                                        WS=cfg["dq_WS"], num_warps=cfg["dq_warps"], num_stages=cfg["dq_stages"])
    _note("dq", kk)
    # per-query-head maxima used for the accumulator units / dS row scale in the dkdv kernel
    ndoh = ndo.amax(1).contiguous()
    dh = delta.abs().amax(1).contiguous()
    BM, BN = cfg["kv_BM"], cfg["kv_BN"]
    kk = _bwd_dkdv_kernel[(S_ // BN, HKV)](_desc(q8, BM, D_), _desc(q8t, D_, BM), _desc(k8, BN, D_),
                                           _desc(v8, BN, D_), _desc(do8, BM, D_), _desc(do8t, D_, BM), sq, sqt,
                                           sqt.amax(1).contiguous(), sk, sv, sdo, sdot, sdot.amax(1).contiguous(),
                                           mrow, delta, vnrm, ndoh, dh, dk, dv,
                                           dk.stride(0), dk.stride(1), S_, sm_scale * LOG2E, sm_scale,
                                           HQ=HQ, HKV=HKV, GROUP=GROUP, D=D_, BM=BM, BN=BN, QB=QBLK,
                                           DS_SCALE=ds_scale, WS=cfg["kv_WS"], num_warps=cfg["kv_warps"],
                                           num_stages=cfg["kv_stages"])
    _note("kv", kk)
    if return_res:
        return dq, dk, dv, res, delta, delta0
    return dq, dk, dv


class FP8FlashAttnFunc(torch.autograd.Function):
    """Core attention: q [S, 1, HQ, D], k/v [S, 1, HKV, D] bf16 (sbhd/bshd with B == 1), causal."""

    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None, matched=True):
        assert q.shape[1] == 1, "B == 1 only"
        q3, k3, v3 = q[:, 0], k[:, 0], v[:, 0]
        o, lse, mrow = fp8_attn_fwd(q3, k3, v3, sm_scale)
        ctx.save_for_backward(q3, k3, v3, o, lse, mrow)
        ctx.sm_scale, ctx.matched = sm_scale, matched
        return o.unsqueeze(1)

    @staticmethod
    def backward(ctx, do):
        q3, k3, v3, o, lse, mrow = ctx.saved_tensors
        dq, dk, dv = fp8_attn_bwd(q3, k3, v3, o, do[:, 0], lse, mrow, ctx.sm_scale, ctx.matched)
        return dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1), None, None
