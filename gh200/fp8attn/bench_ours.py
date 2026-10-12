"""Timing of our Triton FP8 attention (fwd, bwd matched, bwd stale) with per-kernel breakdown.
CFG overrides via env CFG='{"dq_BM":64,...}' (json)."""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import S, HQ, HKV, D, bench, flops_fwd, flops_bwd, tflops, sm_clock  # noqa
import importlib  # noqa
fp8attn = importlib.import_module(os.environ.get("IMPL", "fp8attn2"))
V2 = hasattr(fp8attn, "LAST")

cfg = dict(fp8attn.CFG)
cfg.update(json.loads(os.environ.get("CFG", "{}")))
torch.manual_seed(0)
q = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
print(f"cfg={cfg}  clock/temp/power: {sm_clock()}", flush=True)
fo = fp8attn.fp8_attn_fwd(q, k, v, cfg=cfg)
o, lse = fo[0], fo[1]
extra = fo[2:]
tf, _ = bench(lambda: fp8attn.fp8_attn_fwd(q, k, v, cfg=cfg), iters=30)
res = {"fwd": tf}
for matched in (True, False):
    tb, _ = bench(lambda: fp8attn.fp8_attn_bwd(q, k, v, o, do, lse, *extra, matched=matched, cfg=cfg), iters=30)
    res["bwd_matched" if matched else "bwd_stale"] = tb
print(f"clock/temp/power after: {sm_clock()}")
print(f"RESULT ours fwd_ms={res['fwd']:.3f} ({tflops(flops_fwd(), res['fwd']):.0f} TF)  "
      f"bwd_matched_ms={res['bwd_matched']:.3f} ({tflops(flops_bwd(), res['bwd_matched']):.0f} TF)  "
      f"bwd_stale_ms={res['bwd_stale']:.3f} ({tflops(flops_bwd(), res['bwd_stale']):.0f} TF)", flush=True)

from torch.profiler import profile, ProfilerActivity  # noqa
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    fp8attn.fp8_attn_fwd(q, k, v, cfg=cfg)
    fp8attn.fp8_attn_bwd(q, k, v, o, do, lse, *extra, matched=True, cfg=cfg)
    torch.cuda.synchronize()
for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:10]:
    print(f"  kernel {e.key[:60]:60s} n={e.count} total={e.self_device_time_total/1e3:.3f} ms")
if V2:
    print("  regs/spills/smem:", fp8attn.LAST)
