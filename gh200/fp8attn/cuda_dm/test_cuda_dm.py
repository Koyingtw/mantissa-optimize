"""第 1 關正確性：CUDA（CuTe）FP8 attention backward + Delta-Matching 對 FP64 參考，並與 Triton 版（fp8attn2）比較。

和 test_minimal.py 同樣的指標與形狀（預設 S=2048、8 q heads / 1 kv head、d=128、causal）：
  1. dS 列和殘差 mean|Σ_j dS_ij| / mean|δ_i|：CUDA 版的殘差由 dK/dV kernel 重算的 dS（真正進 dK 的那個）
     對 dq kernel 給的 matched δ 求列和，matched 應 < 1e-5，stale 應 > 1e-3。
  2. dQ / dK / dV 對 FP64 參考的相對誤差：不劣於 Triton 版（DS_SCALE=0 較準、=2 為訓練預設）。
forward 用 Triton 的 fp8attn2.fp8_attn_fwd（o、lse 兩邊共用）。
環境變數：INPUT=randn | dump:<path>，S、HQ、HKV。
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fp8attn2 as fa  # noqa
import fp8dm_cuda as fc  # noqa

S = int(os.environ.get("S", 2048))
HQ = int(os.environ.get("HQ", 8))
HKV = int(os.environ.get("HKV", 1))
D = 128
scale = D ** -0.5
dev = "cuda"
torch.manual_seed(0)
INPUT = os.environ.get("INPUT", "randn")
if INPUT.startswith("dump:"):
    d = torch.load(INPUT[5:], map_location="cpu", weights_only=False)
    q, k, v, do = (d[n][:S, :HQ if n in ("q", "do") else HKV].to(dev, torch.bfloat16).contiguous()
                   for n in ("q", "k", "v", "do"))
    do = (do.float() / do.float().pow(2).mean().sqrt()).to(torch.bfloat16)
else:
    q, k, v, do = (torch.randn(S, h, D, device=dev).to(torch.bfloat16) for h in (HQ, HKV, HKV, HQ))
print(f"INPUT={INPUT}  q{tuple(q.shape)} k{tuple(k.shape)}", flush=True)


@torch.no_grad()
def reference():
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


ref = reference()


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=fa.CFG)
rows = []
for impl in ("cuda", "triton DS0", "triton DS2"):
    for matched in (True, False):
        if impl == "cuda":
            dq, dk, dv, res, delta, _ = fc.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched,
                                                        return_res=True)
        else:
            cfg = dict(fa.CFG, DS_SCALE=int(impl[-1]))
            dq, dk, dv, res, delta, _ = fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched,
                                                        cfg=cfg, return_res=True)
        torch.cuda.synchronize()
        bad = any(not torch.isfinite(t.float()).all() for t in (dq, dk, dv))
        md = delta.abs().mean()
        rows.append(dict(impl=impl, matched=matched, resid=(res.abs().mean() / md).item(),
                         bias=((res * delta.sign()).mean() / md).item(),
                         dq=rel(dq, ref[0]), dk=rel(dk, ref[1]), dv=rel(dv, ref[2]), nan=bad))

print("\n| 實作 | backward | 列和殘差（平均） | 有號偏差 | dQ | dK | dV |\n|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['impl']} | {'Delta-Matching' if r['matched'] else 'stale δ'} | {r['resid']:.2e} | {r['bias']:+.1e} | "
          f"{r['dq']:.3e} | {r['dk']:.3e} | {r['dv']:.3e} |" + (" NaN!" if r["nan"] else ""))


def get(impl, matched):
    return next(r for r in rows if r["impl"] == impl and r["matched"] == matched)


cm, cs, t2 = get("cuda", True), get("cuda", False), get("triton DS2", True)
checks = {
    "CUDA matched 列和殘差 < 1e-5": cm["resid"] < 1e-5,
    "CUDA stale 列和殘差 > 1e-3": cs["resid"] > 1e-3,
    "CUDA matched dQ ≤ stale dQ": cm["dq"] <= cs["dq"] * 1.001,
    "CUDA matched dQ/dK/dV 不劣於 Triton DS2（容差 2%）": all(cm[x] <= t2[x] * 1.02 for x in ("dq", "dk", "dv")),
    "沒有 NaN/Inf": not any(r["nan"] for r in rows),
}
for k_, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + k_)
print("ALL PASS" if all(checks.values()) else "SOME FAILED")
