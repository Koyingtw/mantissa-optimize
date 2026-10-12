"""第 1 關（完整形狀）：真實 dump 的整層張量（S=8192、64 q / 8 kv heads、d=128、causal），原始 dO 不正規化。
比較 CUDA 版（Delta-Matching / stale）與 Triton 版（DS0 / DS2，Delta-Matching）對 FP64 參考的 dQ/dK/dV 相對誤差，
以及列和殘差 mean|Σ_j dS_ij| / mean|δ_i|。forward 一律用 Triton fp8attn2.fp8_attn_fwd（o、lse 共用）。
用法：python test_full.py L40_step6 L00_step10 ...（README：step 1–3 不用）
"""
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fp8attn2 as fa  # noqa
import fp8dm_cuda as fc  # noqa

DUMPS = "/nfs/home/koying/fp8attn-dumps"
D = 128
scale = D ** -0.5
dev = "cuda"


@torch.no_grad()
def reference(q, k, v, do):
    S, HQ, _ = q.shape
    HKV = k.shape[1]
    g = HQ // HKV
    mask = torch.ones(S, S, device=dev, dtype=torch.bool).tril_()
    dq = torch.empty(S, HQ, D, device=dev)
    dk = torch.zeros(S, HKV, D, device=dev, dtype=torch.float64)
    dv = torch.zeros_like(dk)
    for h in range(HQ):
        qh, kh, vh, doh = (t.double() for t in (q[:, h], k[:, h // g], v[:, h // g], do[:, h]))
        p = torch.softmax(((qh @ kh.T) * scale).masked_fill_(~mask, float("-inf")), -1)
        oh = p @ vh
        dp = doh @ vh.T
        dp -= (oh * doh).sum(-1, keepdim=True)
        dp *= p
        dq[:, h] = (dp @ kh * scale).float()
        dk[:, h // g] += dp.T @ qh * scale
        dv[:, h // g] += p.T @ doh
        del p, dp
    return dq, dk.float(), dv.float()


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


print("| dump | 實作 | backward | 列和殘差 | 有號偏差 | dQ | dK | dV |\n|---|---|---|---|---|---|---|---|", flush=True)
for name in sys.argv[1:]:
    d = torch.load(f"{DUMPS}/{name}.pt", map_location="cpu", weights_only=False)
    q, k, v, do = (d[n].to(dev, torch.bfloat16).contiguous() for n in ("q", "k", "v", "do"))
    del d
    t0 = time.time()
    ref = reference(q, k, v, do)
    o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=fa.CFG)
    runs = [("cuda", True, None), ("cuda", False, None), ("triton DS0", True, 0), ("triton DS2", True, 2)]
    for impl, matched, dss in runs:
        if impl == "cuda":
            dq, dk, dv, res, delta, _ = fc.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched,
                                                        return_res=True)
        else:
            dq, dk, dv, res, delta, _ = fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched,
                                                        cfg=dict(fa.CFG, DS_SCALE=dss), return_res=True)
        md = delta.abs().mean()
        bad = " NaN!" if any(not torch.isfinite(t.float()).all() for t in (dq, dk, dv)) else ""
        print(f"| {name} | {impl} | {'Delta-Matching' if matched else 'stale δ'} | {(res.abs().mean() / md).item():.2e} | "
              f"{((res * delta.sign()).mean() / md).item():+.1e} | {rel(dq, ref[0]):.3e} | {rel(dk, ref[1]):.3e} | "
              f"{rel(dv, ref[2]):.3e} |{bad}", flush=True)
    print(f"<!-- {name}: {time.time() - t0:.0f} s -->", flush=True)
