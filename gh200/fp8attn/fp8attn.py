"""FP8 (E4M3) causal flash attention for Hopper in Triton, with a Delta-Matching backward.

Shapes follow Megatron/TE `sbhd` with B == 1: q [S, 1, HQ, D], k/v [S, 1, HKV, D] (GQA), bf16.

Quantization: per-block E4M3, one fp32 dequant scale per (head, 128-row block) for Q, K, V, dO.
P is quantized with a fixed scale (P <= 1); dS is quantized with a per-row amax scale.

Backward (two main kernels, no atomics):
  1. dq kernel (Q-outer): for each Q block it recomputes P (FP32) and dP^ = dO^ V^T (FP8 operands,
     FP32 accumulate), accumulates the matched delta_i = sum_j P_ij dP^_ij on the fly, and builds
       dQ = sum_j P(dP^ - delta0) K  -  (delta - delta0) * sum_j P K
     where delta0 = <dO, O> (the stale delta) is only a centering constant. In exact arithmetic this
     is sum_j P (dP^ - delta) K, i.e. Delta-Matching, in a single pass over KV.
  2. dkdv kernel (KV-outer, loops over the HQ/HKV query heads sharing a KV head): uses the matched
     delta from (1) directly: dS = P (dP^ - delta), dV = P^T dO, dK = dS^T Q.
  MATCHED=False gives the stale-delta control (delta = delta0 everywhere, no P K product).
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
E4M3_MAX = 448.0
QBLK = 128  # quantization block (rows along S); all kernel tiles must divide it
LOG2E = 1.4426950408889634


# --------------------------------------------------------------------------------------------
# Quantization: bf16 [S, H, D] (arbitrary strides) -> fp8 [H, S, D] (+ optional [H, D, S])
# --------------------------------------------------------------------------------------------
@triton.jit
def _quant_kernel(X, X8, X8T, SC, O, DELTA0,
                  stride_xs, stride_xh, stride_os, stride_oh, S,
                  D: tl.constexpr, BLK: tl.constexpr, WRITE_T: tl.constexpr, WITH_DELTA: tl.constexpr):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = pid_s * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    x = tl.load(X + offs_s[:, None] * stride_xs + h * stride_xh + offs_d[None, :]).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(x), 1), 0)
    sc = tl.maximum(amax, 1e-30) / 448.0           # dequant scale
    x8 = (x / sc).to(tl.float8e4nv)
    base = h.to(tl.int64) * S * D
    tl.store(X8 + base + offs_s[:, None] * D + offs_d[None, :], x8)
    if WRITE_T:
        tl.store(X8T + base + offs_d[None, :] * S + offs_s[:, None], x8)
    tl.store(SC + h * (S // BLK) + pid_s, sc)
    if WITH_DELTA:
        o = tl.load(O + offs_s[:, None] * stride_os + h * stride_oh + offs_d[None, :]).to(tl.float32)
        tl.store(DELTA0 + h * S + offs_s, tl.sum(o * x, 1))


def quantize(x, write_t=True, o=None):
    """x: [S, H, D] bf16 view (strides arbitrary, last dim contiguous)."""
    S_, H, D_ = x.shape
    assert x.stride(2) == 1 and S_ % QBLK == 0
    x8 = torch.empty(H, S_, D_, device=x.device, dtype=FP8)
    x8t = torch.empty(H, D_, S_, device=x.device, dtype=FP8) if write_t else x8
    sc = torch.empty(H, S_ // QBLK, device=x.device, dtype=torch.float32)
    delta0 = torch.empty(H, S_, device=x.device, dtype=torch.float32) if o is not None else sc
    oo = o if o is not None else x
    _quant_kernel[(S_ // QBLK, H)](x, x8, x8t, sc, oo, delta0,
                                   x.stride(0), x.stride(1), oo.stride(0), oo.stride(1), S_,
                                   D=D_, BLK=QBLK, WRITE_T=write_t, WITH_DELTA=o is not None,
                                   num_warps=8)
    return x8, x8t, sc, (delta0 if o is not None else None)


# --------------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------------
@triton.jit
def _fwd_inner(acc, l_i, m_i, q8, kbase, vbase, SK, SV, svmax, g, c0, lo, hi,
               offs_m, offs_n, offs_d, S, NQB, D: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr,
               MASK: tl.constexpr):
    for start_n in range(lo, hi, BN):
        cols = start_n + offs_n
        kT = tl.load(kbase + cols[None, :] * D + offs_d[:, None])           # [D, BN], D-contig
        sk = tl.load(SK + g * NQB + start_n // QB)
        sv = tl.load(SV + g * NQB + start_n // QB)
        s = tl.dot(q8, kT) * (c0 * sk)
        if MASK:
            s = tl.where(offs_m[:, None] >= cols[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        p8 = (p * (256.0 * sv / svmax)).to(tl.float8e4nv)
        v = tl.load(vbase + offs_d[None, :] * S + cols[:, None])             # [BN, D], S-contig
        acc = acc * alpha[:, None]
        acc = tl.dot(p8, v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(Q8, K8, V8T, SQ, SK, SV, SVMAX, Out, LSE,
                stride_os, stride_oh, S, qk_scale,
                HQ: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr):
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)   # heavy (long) rows first
    h = tl.program_id(1)
    g = h // GROUP
    NQB = S // QB
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q8 = tl.load(Q8 + h.to(tl.int64) * S * D + offs_m[:, None] * D + offs_d[None, :])
    sq = tl.load(SQ + h * NQB + (pid_m * BM) // QB)
    svmax = tl.load(SVMAX + g)
    kbase = K8 + g.to(tl.int64) * S * D
    vbase = V8T + g.to(tl.int64) * S * D
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    c0 = sq * qk_scale
    lo_diag = (pid_m * BM) // BN * BN
    acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q8, kbase, vbase, SK, SV, svmax, g, c0, 0, lo_diag,
                               offs_m, offs_n, offs_d, S, NQB, D, BN, QB, False)
    acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q8, kbase, vbase, SK, SV, svmax, g, c0, lo_diag,
                               (pid_m + 1) * BM, offs_m, offs_n, offs_d, S, NQB, D, BN, QB, True)
    o = acc * (svmax / 256.0) / l_i[:, None]
    tl.store(Out + offs_m[:, None] * stride_os + h * stride_oh + offs_d[None, :], o.to(tl.bfloat16))
    tl.store(LSE + h * S + offs_m, m_i + tl.math.log2(l_i))


# --------------------------------------------------------------------------------------------
# Backward: dq (Q-outer) with on-the-fly matched delta
# --------------------------------------------------------------------------------------------
@triton.jit
def _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, K8, K8T, V8, SK, SV, skmax, g, gb, c0, sdo, lse, d0,
              lo, hi, offs_m, offs_n, offs_d, S, NQB, D: tl.constexpr, BN: tl.constexpr, QB: tl.constexpr,
              MATCHED: tl.constexpr, WRITE_RES: tl.constexpr, MASK: tl.constexpr):
    for start_n in range(lo, hi, BN):
        cols = start_n + offs_n
        kT = tl.load(K8 + gb + cols[None, :] * D + offs_d[:, None])    # [D, BN] D-contig
        vT = tl.load(V8 + gb + cols[None, :] * D + offs_d[:, None])    # [D, BN] D-contig
        sk = tl.load(SK + g * NQB + start_n // QB)
        sv = tl.load(SV + g * NQB + start_n // QB)
        s = tl.dot(q8, kT) * (c0 * sk)
        if MASK:
            s = tl.where(offs_m[:, None] >= cols[None, :], s, -float("inf"))
        p = tl.math.exp2(s - lse[:, None])
        dp = tl.dot(do8, vT) * (sdo * sv)
        if MATCHED:
            dacc += tl.sum(p * dp, 1)
        ds = p * (dp - d0[:, None])
        if WRITE_RES:
            rsum += tl.sum(ds, 1)
            psum += tl.sum(p, 1)
        amax = tl.max(tl.abs(ds), 1)
        r = 448.0 / tl.maximum(amax, 1e-30)
        ds8 = (ds * r[:, None]).to(tl.float8e4nv)
        kb = tl.load(K8T + gb + offs_d[None, :] * S + cols[:, None])    # [BN, D] S-contig
        accA += tl.dot(ds8, kb) * (sk / r)[:, None]
        if MATCHED:
            p8 = (p * (256.0 * sk / skmax)).to(tl.float8e4nv)
            accB = tl.dot(p8, kb, accB)
    return accA, accB, dacc, rsum, psum


@triton.jit
def _bwd_dq_kernel(Q8, K8, K8T, V8, DO8, SQ, SK, SKMAX, SV, SDO, LSE, DELTA0, DELTA, DQ, RES,
                   stride_dqs, stride_dqh, S, qk_scale, sm_scale,
                   GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   QB: tl.constexpr, MATCHED: tl.constexpr, WRITE_RES: tl.constexpr):
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
    h = tl.program_id(1)
    g = h // GROUP
    NQB = S // QB
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    hb = h.to(tl.int64) * S * D
    q8 = tl.load(Q8 + hb + offs_m[:, None] * D + offs_d[None, :])
    do8 = tl.load(DO8 + hb + offs_m[:, None] * D + offs_d[None, :])
    sq = tl.load(SQ + h * NQB + (pid_m * BM) // QB)
    sdo = tl.load(SDO + h * NQB + (pid_m * BM) // QB)
    skmax = tl.load(SKMAX + g)
    lse = tl.load(LSE + h * S + offs_m)
    d0 = tl.load(DELTA0 + h * S + offs_m)
    gb = g.to(tl.int64) * S * D
    accA = tl.zeros([BM, D], tl.float32)
    accB = tl.zeros([BM, D], tl.float32)
    dacc = tl.zeros([BM], tl.float32)
    rsum = tl.zeros([BM], tl.float32)
    psum = tl.zeros([BM], tl.float32)
    c0 = sq * qk_scale
    lo_diag = (pid_m * BM) // BN * BN
    accA, accB, dacc, rsum, psum = _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, K8, K8T, V8, SK, SV, skmax,
                                             g, gb, c0, sdo, lse, d0, 0, lo_diag, offs_m, offs_n, offs_d,
                                             S, NQB, D, BN, QB, MATCHED, WRITE_RES, False)
    accA, accB, dacc, rsum, psum = _dq_inner(accA, accB, dacc, rsum, psum, q8, do8, K8, K8T, V8, SK, SV, skmax,
                                             g, gb, c0, sdo, lse, d0, lo_diag, (pid_m + 1) * BM, offs_m, offs_n,
                                             offs_d, S, NQB, D, BN, QB, MATCHED, WRITE_RES, True)
    if MATCHED:
        eps = dacc - d0
        dq = accA - eps[:, None] * (accB * (skmax / 256.0))
        tl.store(DELTA + h * S + offs_m, dacc)
        if WRITE_RES:
            rsum = rsum - eps * psum
    else:
        dq = accA
    dq = dq * sm_scale
    tl.store(DQ + offs_m[:, None] * stride_dqs + h * stride_dqh + offs_d[None, :], dq.to(tl.bfloat16))
    if WRITE_RES:
        tl.store(RES + h * S + offs_m, rsum)


# --------------------------------------------------------------------------------------------
# Backward: dK, dV (KV-outer, loops over the GROUP query heads of this KV head)
# --------------------------------------------------------------------------------------------
@triton.jit
def _kv_inner(accK, accV, k8, v8, Q8, Q8T, DO8, DO8T, SQ, SDO, LSE, DELTA, sdomax, h, hb, c0, sv,
              lo, hi, offs_m, offs_n, offs_d, S, NQB, D: tl.constexpr, BM: tl.constexpr, QB: tl.constexpr,
              MASK: tl.constexpr):
    for start_m in range(lo, hi, BM):
        rows = start_m + offs_m
        qT = tl.load(Q8 + hb + rows[None, :] * D + offs_d[:, None])     # [D, BM] D-contig
        doT = tl.load(DO8 + hb + rows[None, :] * D + offs_d[:, None])
        sq = tl.load(SQ + h * NQB + start_m // QB)
        sdo = tl.load(SDO + h * NQB + start_m // QB)
        lse = tl.load(LSE + h * S + rows)
        dl = tl.load(DELTA + h * S + rows)
        sT = tl.dot(k8, qT) * (c0 * sq)
        if MASK:
            sT = tl.where(rows[None, :] >= offs_n[:, None], sT, -float("inf"))
        pT = tl.math.exp2(sT - lse[None, :])
        dpT = tl.dot(v8, doT) * (sv * sdo)
        dsT = pT * (dpT - dl[None, :])
        p8 = (pT * (256.0 * sdo / sdomax)).to(tl.float8e4nv)
        dob = tl.load(DO8T + hb + offs_d[None, :] * S + rows[:, None])  # [BM, D] S-contig
        accV = tl.dot(p8, dob, accV)
        amax = tl.max(tl.abs(dsT), 1)
        r = 448.0 / tl.maximum(amax, 1e-30)
        ds8 = (dsT * r[:, None]).to(tl.float8e4nv)
        qb = tl.load(Q8T + hb + offs_d[None, :] * S + rows[:, None])
        accK += tl.dot(ds8, qb) * (sq / r)[:, None]
    return accK, accV


@triton.jit
def _bwd_dkdv_kernel(Q8, Q8T, K8, V8, DO8, DO8T, SQ, SQMAX, SK, SV, SDO, SDOMAX, LSE, DELTA,
                     DK, DV, stride_dks, stride_dkh, S, qk_scale, sm_scale,
                     GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     QB: tl.constexpr):
    pid_n = tl.program_id(0)          # light blocks (late keys) have few queries; schedule heavy first
    g = tl.program_id(1)
    NQB = S // QB
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    gb = g.to(tl.int64) * S * D
    k8 = tl.load(K8 + gb + offs_n[:, None] * D + offs_d[None, :])     # [BN, D]
    v8 = tl.load(V8 + gb + offs_n[:, None] * D + offs_d[None, :])
    sk = tl.load(SK + g * NQB + (pid_n * BN) // QB)
    sv = tl.load(SV + g * NQB + (pid_n * BN) // QB)
    sqmax = tl.load(SQMAX)
    sdomax = tl.load(SDOMAX)
    accK = tl.zeros([BN, D], tl.float32)
    accV = tl.zeros([BN, D], tl.float32)
    c0 = sk * qk_scale
    lo = (pid_n * BN) // BM * BM
    hi_diag = tl.minimum(((pid_n + 1) * BN + BM - 1) // BM * BM, S)
    for hh in range(GROUP):
        h = g * GROUP + hh
        hb = h.to(tl.int64) * S * D
        accK, accV = _kv_inner(accK, accV, k8, v8, Q8, Q8T, DO8, DO8T, SQ, SDO, LSE, DELTA, sdomax, h, hb, c0, sv,
                               lo, hi_diag, offs_m, offs_n, offs_d, S, NQB, D, BM, QB, True)
        accK, accV = _kv_inner(accK, accV, k8, v8, Q8, Q8T, DO8, DO8T, SQ, SDO, LSE, DELTA, sdomax, h, hb, c0, sv,
                               hi_diag, S, offs_m, offs_n, offs_d, S, NQB, D, BM, QB, False)
    dk = accK * sm_scale
    dv = accV * (sdomax / 256.0)
    tl.store(DK + offs_n[:, None] * stride_dks + g * stride_dkh + offs_d[None, :], dk.to(tl.bfloat16))
    tl.store(DV + offs_n[:, None] * stride_dks + g * stride_dkh + offs_d[None, :], dv.to(tl.bfloat16))


# --------------------------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------------------------
CFG = dict(fwd_BM=128, fwd_BN=128, fwd_warps=8, fwd_stages=3,
           dq_BM=64, dq_BN=128, dq_warps=4, dq_stages=2,
           kv_BM=128, kv_BN=64, kv_warps=4, kv_stages=2)


def fp8_attn_fwd(q, k, v, sm_scale=None, cfg=CFG):
    """q [S, HQ, D], k/v [S, HKV, D] bf16 (views of sbhd with B==1). Returns o [S, HQ, D], lse2 [HQ, S]."""
    S_, HQ, D_ = q.shape
    HKV = k.shape[1]
    sm_scale = sm_scale or D_ ** -0.5
    q8, _, sq, _ = quantize(q, write_t=False)
    k8, _, sk, _ = quantize(k, write_t=False)
    _, v8t, sv, _ = quantize(v, write_t=True)
    o = torch.empty(S_, HQ, D_, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(HQ, S_, device=q.device, dtype=torch.float32)
    BM, BN = cfg["fwd_BM"], cfg["fwd_BN"]
    _fwd_kernel[(S_ // BM, HQ)](q8, k8, v8t, sq, sk, sv, sv.amax(1).contiguous(), o, lse,
                                o.stride(0), o.stride(1), S_, sm_scale * LOG2E,
                                HQ=HQ, GROUP=HQ // HKV, D=D_, BM=BM, BN=BN, QB=QBLK,
                                num_warps=cfg["fwd_warps"], num_stages=cfg["fwd_stages"])
    return o, lse


def fp8_attn_bwd(q, k, v, o, do, lse, sm_scale=None, matched=True, cfg=CFG, return_res=False):
    S_, HQ, D_ = q.shape
    HKV = k.shape[1]
    GROUP = HQ // HKV
    sm_scale = sm_scale or D_ ** -0.5
    q8, q8t, sq, _ = quantize(q, write_t=True)
    k8, k8t, sk, _ = quantize(k, write_t=True)
    v8, _, sv, _ = quantize(v, write_t=False)
    do8, do8t, sdo, delta0 = quantize(do, write_t=True, o=o)
    dq = torch.empty_like(q)
    dk = torch.empty(S_, HKV, D_, device=q.device, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    delta = torch.empty(HQ, S_, device=q.device, dtype=torch.float32) if matched else delta0
    res = torch.empty(HQ, S_, device=q.device, dtype=torch.float32) if return_res else delta
    BM, BN = cfg["dq_BM"], cfg["dq_BN"]
    _bwd_dq_kernel[(S_ // BM, HQ)](q8, k8, k8t, v8, do8, sq, sk, sk.amax(1).contiguous(), sv, sdo,
                                   lse, delta0, delta, dq, res,
                                   dq.stride(0), dq.stride(1), S_, sm_scale * LOG2E, sm_scale,
                                   GROUP=GROUP, D=D_, BM=BM, BN=BN, QB=QBLK,
                                   MATCHED=matched, WRITE_RES=return_res,
                                   num_warps=cfg["dq_warps"], num_stages=cfg["dq_stages"])
    BM, BN = cfg["kv_BM"], cfg["kv_BN"]
    _bwd_dkdv_kernel[(S_ // BN, HKV)](q8, q8t, k8, v8, do8, do8t, sq, sq.amax().reshape(1), sk, sv,
                                      sdo, sdo.amax().reshape(1), lse, delta, dk, dv,
                                      dk.stride(0), dk.stride(1), S_, sm_scale * LOG2E, sm_scale,
                                      GROUP=GROUP, D=D_, BM=BM, BN=BN, QB=QBLK,
                                      num_warps=cfg["kv_warps"], num_stages=cfg["kv_stages"])
    if return_res:
        return dq, dk, dv, res, delta, delta0
    return dq, dk, dv


class FP8FlashAttnFunc(torch.autograd.Function):
    """Drop-in core attention: q [S, 1, HQ, D], k/v [S, 1, HKV, D] bf16 (sbhd), causal, no dropout."""

    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None, matched=True):
        assert q.shape[1] == 1, "B == 1 only"
        q3, k3, v3 = q[:, 0], k[:, 0], v[:, 0]
        o, lse = fp8_attn_fwd(q3, k3, v3, sm_scale)
        ctx.save_for_backward(q3, k3, v3, o, lse)
        ctx.sm_scale, ctx.matched = sm_scale, matched
        return o.unsqueeze(1)

    @staticmethod
    def backward(ctx, do):
        q3, k3, v3, o, lse = ctx.saved_tensors
        dq, dk, dv = fp8_attn_bwd(q3, k3, v3, o, do[:, 0], lse, ctx.sm_scale, ctx.matched)
        return dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1), None, None
