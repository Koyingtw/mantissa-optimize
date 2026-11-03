"""Mantissa reduction of FP8 attention operands x stale delta / Delta-Matching, on MI350X (AI paper, gap G1).

Reruns gh200/fp8attn/test_mant_trunc.py (the Triton FP8 flash attention fp8attn2 with forward + backward, stale or
matched delta) with one addition: the kept E4M3 bits can be rounded to nearest (ROUND=nearest) instead of truncated
toward zero (ROUND=trunc, the GH200 study). Truncation toward zero shrinks every |dP| a little, so the stale-delta row
residual turns into a bias with the sign of -delta; rounding to nearest zeroes the same low bits (same switching saving)
without a systematic shrink, so the question is whether the bias is a property of truncation or of fewer bits.

Each FP8 operand (Q, K, V, dO and the transposed / row-scaled copies the backward uses) keeps BITS of 3 mantissa bits
after quantization; P and dS inside the kernel are not reduced. Shape as on GH200 (S=2048, 8 q / 1 kv heads, d=128,
causal), compared with an FP64 reference on the unreduced BF16 inputs. Metrics as on GH200:
  resid  mean|sum_j dS_ij| / mean|delta_i|           (row-sum residual)
  bias   mean(sum_j dS_ij * sign(delta_i)) / mean|delta_i|
  dQ dK dV relative errors against the reference
usage: ROUND=trunc|nearest INPUTS=randn[,<name of a .pt dump in DUMPS>] fp8attn_trunc.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gh200", "fp8attn"))
import fp8attn2 as fa  # noqa: E402

S, HQ, HKV, D = 2048, 8, 1, 128
scale = D ** -0.5
dev = "cuda"
INPUTS = os.environ.get("INPUTS", "randn").split(",")
ROUND = os.environ.get("ROUND", "trunc")
DUMPS = os.environ.get("DUMPS", os.path.expanduser("~/koying/sw/fp8attn-dumps"))
_orig_quantize = fa.quantize
KEEP = {"bits": 3}


def _reduce(t):
    """In place on the E4M3 bytes: keep KEEP['bits'] mantissa bits, truncating or rounding to nearest even
    (magnitude saturates below the NaN code 0x7F, as gh200/infer/ppl_eval.reduce_e4m3 does)."""
    if t is None or KEEP["bits"] >= 3:
        return
    drop = 3 - KEEP["bits"]
    low = (1 << drop) - 1
    u = t.view(torch.uint8)
    if ROUND == "trunc":
        u.bitwise_and_(0xFF ^ low)
        return
    v = u.to(torch.int16)
    sign, mag = v & 0x80, v & 0x7F
    mag = mag + ((1 << (drop - 1)) - 1) + ((mag >> drop) & 1)
    mag = torch.clamp(mag & ~low, max=0x7E & ~low)
    u.copy_((sign | mag).to(torch.uint8))


def quantize_reduced(*a, **kw):
    x8, x8t, sc, d0, nr = _orig_quantize(*a, **kw)
    _reduce(x8)
    if x8t is not x8:
        _reduce(x8t)
    return x8, x8t, sc, d0, nr


fa.quantize = quantize_reduced


def load(name):
    torch.manual_seed(0)
    if name == "randn":
        return [torch.randn(S, h, D, device=dev).to(torch.bfloat16) for h in (HQ, HKV, HKV, HQ)]
    d = torch.load(f"{DUMPS}/{name}.pt", map_location="cpu", weights_only=False)
    q, k, v, do = (d[n][:S, :HQ if n in ("q", "do") else HKV].to(dev, torch.bfloat16).contiguous()
                   for n in ("q", "k", "v", "do"))
    do = (do.float() / do.float().pow(2).mean().sqrt()).to(torch.bfloat16)
    return [q, k, v, do]


@torch.no_grad()
def reference(q, k, v, do):
    g = HQ // HKV
    mask = torch.ones(S, S, device=dev, dtype=torch.bool).tril_()
    dq = torch.empty(S, HQ, D, device=dev)
    dk = torch.zeros(S, HKV, D, device=dev)
    dv = torch.zeros_like(dk)
    for h in range(HQ):
        qh, kh, vh, doh = (t.double() for t in (q[:, h], k[:, h // g], v[:, h // g], do[:, h]))
        p = torch.softmax(((qh @ kh.T) * scale).masked_fill(~mask, float("-inf")), -1)
        oh = p @ vh
        ds = p * (doh @ vh.T - (oh * doh).sum(-1, keepdim=True))
        dq[:, h] = (ds @ kh * scale).float()
        dk[:, h // g] += (ds.T @ qh * scale).float()
        dv[:, h // g] += (p.T @ doh).float()
    return dq, dk, dv


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


cfg = dict(fa.CFG, DS_SCALE=0)
print(f"ROUND={ROUND} device={torch.cuda.get_device_name()}")
print("| input | round | bits | backward | resid | bias | dQ | dK | dV |\n|---|---|---|---|---|---|---|---|---|")
for name in INPUTS:
    q, k, v, do = load(name)
    ref = reference(q, k, v, do)
    for bits in (3, 2, 1, 0):
        KEEP["bits"] = bits
        o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=cfg)
        for matched in (False, True):
            dq, dk, dv, res, delta, _ = fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale,
                                                        matched=matched, cfg=cfg, return_res=True)
            md = delta.abs().mean()
            resid = (res.abs().mean() / md).item()
            bias = ((res * delta.sign()).mean() / md).item()
            print(f"| {name} | {ROUND} | {bits} | {'matched' if matched else 'stale'} | {resid:.2e} | {bias:+.2e} | "
                  f"{rel(dq, ref[0]):.3e} | {rel(dk, ref[1]):.3e} | {rel(dv, ref[2]):.3e} |", flush=True)
