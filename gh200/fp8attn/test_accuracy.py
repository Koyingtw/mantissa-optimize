"""Numerical accuracy vs an FP32 reference for: our FP8 kernel (matched / stale delta), TE cuDNN
bf16, TE cuDNN fp8 (fp8 bwd), TE fp8 fwd + bf16 bwd, and FA3 bf16 if importable.

INPUT=randn (default) | peaky (q,k x3: sharper softmax) | dump:<path.pt> (dict with q,k,v,do [S,1,H,D])
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import S, HQ, HKV, D  # noqa
import importlib  # noqa
import json  # noqa
fp8attn = importlib.import_module(os.environ.get("IMPL", "fp8attn2"))
CFG = dict(fp8attn.CFG)
CFG.update(json.loads(os.environ.get("CFG", "{}")))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
INPUT = os.environ.get("INPUT", "randn")
SEED = int(os.environ.get("SEED", 0))
torch.manual_seed(SEED)


DUMP_GRADS = {}


def load_inputs():
    if INPUT.startswith("dump:"):
        d = torch.load(INPUT[5:], map_location="cpu", weights_only=False)
        out = []
        for n in ("q", "k", "v", "do"):
            t = d[n].to(dev, torch.bfloat16)
            out.append((t.unsqueeze(1) if t.dim() == 3 else t).contiguous())
        if "dq" in d:   # what TE/cuDNN FP8 returned inside the real training step
            DUMP_GRADS.update({n: d[n].to(dev) for n in ("o", "dq", "dk", "dv")})
        return out
    q = torch.randn(S, 1, HQ, D, device=dev)
    k = torch.randn(S, 1, HKV, D, device=dev)
    v = torch.randn(S, 1, HKV, D, device=dev)
    do = torch.randn(S, 1, HQ, D, device=dev)
    if INPUT == "peaky":
        q, k = q * 3, k * 3
    return [t.to(torch.bfloat16) for t in (q, k, v, do)]


q, k, v, do = load_inputs()
S_, _, HQ_, D_ = q.shape
HKV_ = k.shape[2]
G = HQ_ // HKV_
scale = D_ ** -0.5
print(f"INPUT={INPUT} shape q={tuple(q.shape)} k={tuple(k.shape)}  "
      f"|q|max={q.abs().max().item():.3g} |k|max={k.abs().max().item():.3g} "
      f"|v|max={v.abs().max().item():.3g} |do|max={do.abs().max().item():.3g}", flush=True)


# ---------------- FP32 reference (bf16 inputs promoted, exact math per head) ----------------
@torch.no_grad()
def reference():
    o = torch.empty(S_, HQ_, D_, device=dev)
    dq = torch.empty(S_, HQ_, D_, device=dev)
    dk = torch.zeros(S_, HKV_, D_, device=dev)
    dv = torch.zeros(S_, HKV_, D_, device=dev)
    mask = torch.ones(S_, S_, device=dev, dtype=torch.bool).tril_()
    ent = []
    for h in range(HQ_):
        g = h // G
        qh, kh, vh, doh = (t.float() for t in (q[:, 0, h], k[:, 0, g], v[:, 0, g], do[:, 0, h]))
        s = (qh @ kh.T) * scale
        s.masked_fill_(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        del s
        oh = p @ vh
        dp = doh @ vh.T
        delta = (oh * doh).sum(-1, keepdim=True)
        ds = p * (dp - delta)
        del dp
        dq[:, h] = (ds @ kh) * scale
        dk[:, g] += (ds.T @ qh) * scale
        dv[:, g] += p.T @ doh
        o[:, h] = oh
        if h < 4:
            pl = p.clamp_min(1e-30)
            ent.append((-(p * pl.log()).sum(-1)).mean().item())
        del p, ds
    print(f"ref: mean softmax entropy (heads 0-3) = {sum(ent)/len(ent):.2f} nats (uniform over 4096 ~ {torch.log(torch.tensor(4096.)).item():.2f})")
    return o, dq, dk, dv


ref = reference()
torch.cuda.empty_cache()


def rel(a, b):
    a = a.float().reshape(b.shape)
    return ((a - b).norm() / b.norm()).item()


rows = []


def report(name, o, dq, dk, dv, extra=""):
    r = [rel(o, ref[0]), rel(dq, ref[1]), rel(dk, ref[2]), rel(dv, ref[3])]
    rows.append((name, r, extra))
    print(f"{name:28s} o={r[0]:.3e} dq={r[1]:.3e} dk={r[2]:.3e} dv={r[3]:.3e} {extra}", flush=True)


if DUMP_GRADS:
    report("TE fp8 in training (dump)", DUMP_GRADS["o"], DUMP_GRADS["dq"], DUMP_GRADS["dk"], DUMP_GRADS["dv"])

# ---------------- ours ----------------
VARIANTS = [(m, d) for d in json.loads(os.environ.get("DS_SCALES", "[1, 0]")) for m in (True, False)]
for matched, ds_scale in VARIANTS:
    cfg = dict(CFG, DS_SCALE=ds_scale)
    fo = fp8attn.fp8_attn_fwd(q[:, 0], k[:, 0], v[:, 0], scale, cfg=cfg)
    o, lse, extra = fo[0], fo[1], fo[2:]
    dq, dk, dv, res, delta, delta0 = fp8attn.fp8_attn_bwd(q[:, 0], k[:, 0], v[:, 0], o, do[:, 0], lse, *extra,
                                                          sm_scale=scale, matched=matched, cfg=cfg,
                                                          return_res=True)
    # residual of sum_j dS_ij (FP32 dS before FP8 quantization), relative to |delta|
    rr = (res.abs().mean() / delta.abs().mean()).item()
    dd = ((delta - delta0).abs().mean() / delta0.abs().mean()).item()
    report(f"ours fp8 {'matched' if matched else 'stale'} ds{ds_scale}", o, dq, dk, dv,
           f"rowsum|dS|/|delta|={rr:.2e} |delta-delta0|/|delta0|={dd:.2e}")
    del dq, dk, dv
torch.cuda.empty_cache()

# ---------------- TE / cuDNN ----------------
os.environ.setdefault("NVTE_FUSED_ATTN", "1")
os.environ.setdefault("NVTE_FLASH_ATTN", "0")
import transformer_engine.pytorch as te  # noqa
from transformer_engine.common.recipe import DelayedScaling, Format  # noqa

attn = te.DotProductAttention(HQ_, D_, num_gqa_groups=HKV_, attention_dropout=0.0,
                              qkv_format="sbhd", attn_mask_type="causal").to(dev)
for name, fp8, bwd8 in (("TE cuDNN bf16", False, "1"), ("TE cuDNN fp8 (fp8 bwd)", True, "1"),
                        ("TE cuDNN fp8 fwd, bf16 bwd", True, "0")):
    os.environ["NVTE_FP8_DPA_BWD"] = bwd8
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16,
                            amax_compute_algo="max", fp8_dpa=fp8)
    for it in range(3):  # let delayed scaling amax history settle
        qq, kk, vv = (t.detach().clone().requires_grad_() for t in (q, k, v))
        with te.fp8_autocast(enabled=fp8, fp8_recipe=recipe):
            out = attn(qq, kk, vv)
        out.backward(do.reshape(S_, 1, HQ_ * D_))
    report(name, out.view(S_, HQ_, D_).detach(), qq.grad[:, 0], kk.grad[:, 0], vv.grad[:, 0])
    del qq, kk, vv, out
    torch.cuda.empty_cache()

# ---------------- FA3 (if built) ----------------
try:
    import flash_attn_interface as fa3
    qq, kk, vv = (t.detach().permute(1, 0, 2, 3).contiguous().requires_grad_() for t in (q, k, v))
    out = fa3.flash_attn_func(qq, kk, vv, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    out.backward(do.permute(1, 0, 2, 3))
    report("FA3 bf16", out[0], qq.grad[0], kk.grad[0], vv.grad[0])
except Exception as e:  # noqa
    print("FA3 skipped:", repr(e)[:200])

print("\n| method | O | dQ | dK | dV | note |\n|---|---|---|---|---|---|")
for n, r, e in rows:
    print(f"| {n} | {r[0]:.2e} | {r[1]:.2e} | {r[2]:.2e} | {r[3]:.2e} | {e} |")
