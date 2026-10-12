"""attention 運算元截尾數 × Delta-Matching：截得越多，stale δ 是否越糟、Delta-Matching 是否仍乾淨。

自寫 Triton FP8 attention（fp8attn2）的每個 FP8 運算元（Q、K、V、dO 與 backward 用的轉置／列縮放副本）
在量化後把 E4M3 尾數從 3 位截到 BITS 位（位元遮罩、朝零截斷，就是減少切換位元的作法）。
kernel 內部的 P、dS 量化不截。形狀與 test_minimal.py 相同（S=2048、8 q / 1 kv head、d=128、causal），
對 FP64 參考（用未截斷的 bf16 輸入，即「真正的梯度」）比較。

指標：
  resid      mean|Σ_j dS_ij| / mean|δ_i|          列和殘差大小
  bias       mean(Σ_j dS_ij · sign(δ_i)) / mean|δ_i|  與 δ 同號的系統偏差（截斷讓 |dP̂| 系統性變小時，
                                                    stale 的殘差 ≈ −c·δ，這裡會是負值且隨截斷變大）
  dQ/dK/dV   相對 FP64 參考的相對誤差
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8attn2 as fa  # noqa

S, HQ, HKV, D = 2048, 8, 1, 128
scale = D ** -0.5
dev = "cuda"
INPUTS = os.environ.get("INPUTS", "randn,L40_step6,L00_step10,L79_step30").split(",")
DUMPS = "/nfs/home/koying/fp8attn-dumps"
_orig_quantize = fa.quantize
KEEP = {"bits": 3}


def _mask(t):
    if t is not None and KEEP["bits"] < 3:
        drop = 3 - KEEP["bits"]
        t.view(torch.uint8).bitwise_and_(0xFF ^ ((1 << drop) - 1))


def quantize_trunc(*a, **kw):
    x8, x8t, sc, d0, nr = _orig_quantize(*a, **kw)
    _mask(x8)
    if x8t is not x8:
        _mask(x8t)
    return x8, x8t, sc, d0, nr


fa.quantize = quantize_trunc


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
    dk = torch.zeros(S, HKV, D, device=dev); dv = torch.zeros_like(dk)
    for h in range(HQ):
        qh, kh, vh, doh = (t.double() for t in (q[:, h], k[:, h // g], v[:, h // g], do[:, h]))
        p = torch.softmax(((qh @ kh.T) * scale).masked_fill(~mask, float("-inf")), -1)
        oh = p @ vh
        ds = p * (doh @ vh.T - (oh * doh).sum(-1, keepdim=True))
        dq[:, h] = (ds @ kh * scale).float()
        dk[:, h // g] += (ds.T @ qh * scale).float(); dv[:, h // g] += (p.T @ doh).float()
    return dq, dk, dv


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


cfg = dict(fa.CFG, DS_SCALE=0)
print("| 輸入 | 尾數位元 | backward | resid | bias | dQ | dK | dV |\n|---|---|---|---|---|---|---|---|")
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
            print(f"| {name} | {bits} | {'matched' if matched else 'stale'} | {resid:.2e} | {bias:+.2e} | "
                  f"{rel(dq, ref[0]):.3e} | {rel(dk, ref[1]):.3e} | {rel(dv, ref[2]):.3e} |", flush=True)
