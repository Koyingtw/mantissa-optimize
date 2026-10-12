"""最小 kernel 測試：論文 Delta-Matching（arXiv 2609.37852）在自寫 Triton FP8 attention 上是否成立。

小形狀（S=2048、8 個 q head 共用 1 個 kv head、d=128、causal），對 FP32 參考比較：
  1. dS 列和殘差 |Σ_j dS_ij| / |δ_i|：matched 應在 FP32 捨入級（< 1e-5），stale 應在 1e-3 以上。
  2. dQ / dK / dV 相對誤差：matched 不應比 stale 差，dQ 應明顯較好。
  3. autograd 路徑（FP8FlashAttnFunc，訓練用的入口）與直接呼叫 matched 的結果一致。
輸入：INPUT=randn（預設）或 dump:<path.pt>（取真實訓練張量的前 S 個 token、head 0-7 / kv head 0；
causal 下前綴是合法的子問題）。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton  # noqa
from triton.runtime import driver  # noqa

print(f"torch {torch.__version__}  triton {triton.__version__} from {os.path.dirname(triton.__file__)}", flush=True)
print("triton active driver:", type(driver.active).__name__, flush=True)  # 訓練中 D 崩潰的地方

import fp8attn2 as fa  # noqa

S, HQ, HKV, D = 2048, 8, 1, 128
scale = D ** -0.5
dev = "cuda"
torch.manual_seed(0)
INPUT = os.environ.get("INPUT", "randn")
if INPUT.startswith("dump:"):
    d = torch.load(INPUT[5:], map_location="cpu", weights_only=False)
    q, k, v, do = (d[n][:S, :HQ if n in ("q", "do") else HKV].to(dev, torch.bfloat16).contiguous()
                   for n in ("q", "k", "v", "do"))
    # dump 的 dO 是整個 64 head 的梯度，量級可能很小；正規化到單位 RMS，不改變方向與相對誤差。
    do = (do.float() / do.float().pow(2).mean().sqrt()).to(torch.bfloat16)
else:
    q, k, v, do = (torch.randn(S, h, D, device=dev).to(torch.bfloat16) for h in (HQ, HKV, HKV, HQ))
print(f"INPUT={INPUT}  q{tuple(q.shape)} k{tuple(k.shape)}", flush=True)


@torch.no_grad()
def reference():
    g = HQ // HKV
    mask = torch.ones(S, S, device=dev, dtype=torch.bool).tril_()
    o = torch.empty(S, HQ, D, device=dev); dq = torch.empty_like(o)
    dk = torch.zeros(S, HKV, D, device=dev); dv = torch.zeros_like(dk)
    for h in range(HQ):
        qh, kh, vh, doh = (t.double() for t in (q[:, h], k[:, h // g], v[:, h // g], do[:, h]))
        s = ((qh @ kh.T) * scale).masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1)
        oh = p @ vh
        ds = p * (doh @ vh.T - (oh * doh).sum(-1, keepdim=True))
        o[:, h] = oh.float(); dq[:, h] = (ds @ kh * scale).float()
        dk[:, h // g] += (ds.T @ qh * scale).float(); dv[:, h // g] += (p.T @ doh).float()
    return o, dq, dk, dv


ref = reference()


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


# DS_SCALE：dS 量化成 FP8 時的縮放方式。0 = 每 tile 每列 amax（較準），2 = 訓練入口的預設（較快）。
o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=fa.CFG)
out = {}
for dss in (0, fa.CFG["DS_SCALE"]):
    cfg = dict(fa.CFG, DS_SCALE=dss)
    for matched in (True, False):
        dq, dk, dv, res, delta, _ = fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched,
                                                    cfg=cfg, return_res=True)
        r = (res.abs() / delta.abs().clamp_min(1e-30))
        out[(dss, matched)] = dict(resid_med=r.median().item(), resid_mean=(res.abs().mean() / delta.abs().mean()).item(),
                                   o=rel(o, ref[0]), dq=rel(dq, ref[1]), dk=rel(dk, ref[2]), dv=rel(dv, ref[3]), grads=(dq, dk, dv))

# 訓練入口：autograd.Function（預設 matched=True）
qa, ka, va = (t.detach().clone().unsqueeze(1).requires_grad_() for t in (q, k, v))
oa = fa.FP8FlashAttnFunc.apply(qa, ka, va, scale)
oa.backward(do.unsqueeze(1))
m = out[(fa.CFG["DS_SCALE"], True)]["grads"]
same = max(rel(qa.grad[:, 0], m[0].float()), rel(ka.grad[:, 0], m[1].float()), rel(va.grad[:, 0], m[2].float()))

print("\n| backward | DS_SCALE | 列和殘差（中位數） | 列和殘差（平均） | O | dQ | dK | dV |\n|---|---|---|---|---|---|---|---|")
for (dss, matched), x in out.items():
    print(f"| {'Delta-Matching' if matched else 'stale δ'} | {dss} | {x['resid_med']:.2e} | {x['resid_mean']:.2e} | "
          f"{x['o']:.3e} | {x['dq']:.3e} | {x['dk']:.3e} | {x['dv']:.3e} |")
print(f"\nautograd 路徑 vs 直接呼叫 matched 的最大相對差：{same:.2e}")

checks = {}
for dss in (0, fa.CFG["DS_SCALE"]):
    a, b = out[(dss, True)], out[(dss, False)]
    checks.update({
        f"[DS_SCALE={dss}] matched 列和殘差 < 1e-5": a["resid_mean"] < 1e-5,
        f"[DS_SCALE={dss}] stale 列和殘差 > 1e-3": b["resid_mean"] > 1e-3,
        f"[DS_SCALE={dss}] matched dQ 誤差 ≤ stale": a["dq"] <= b["dq"] * 1.001,
        f"[DS_SCALE={dss}] matched dK 誤差 ≤ stale": a["dk"] <= b["dk"] * 1.001,
    })
checks["autograd 路徑與直接呼叫一致（< 1e-3）"] = same < 1e-3
for k_, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + k_)
print("ALL PASS" if all(checks.values()) else "SOME FAILED")
