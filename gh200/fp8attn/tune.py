"""Per-kernel config sweep. KERNEL=fwd|dq|kv ; configs = cartesian product of the lists below.
Times the kernel by torch.profiler (mean over 10 calls)."""
import itertools
import json
import os
import sys
import traceback
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(__file__))
from common import S, HQ, HKV, D, flops_fwd, tflops, sm_clock  # noqa
import importlib  # noqa
fp8attn = importlib.import_module(os.environ.get("IMPL", "fp8attn2"))

KERNEL = os.environ.get("KERNEL", "fwd")
space = json.loads(os.environ.get("SPACE", "{}"))
torch.manual_seed(0)
q = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
fo = fp8attn.fp8_attn_fwd(q, k, v)
o, lse, extra = fo[0], fo[1], fo[2:]
name = {"fwd": "_fwd_kernel", "dq": "_bwd_dq_kernel", "kv": "_bwd_dkdv_kernel"}[KERNEL]
mm = {"fwd": 2, "dq": 4, "kv": 4}[KERNEL]   # fp8 matmuls per tile (for an "executed TFLOPS" figure)
print(f"KERNEL={KERNEL} space={space} clock={sm_clock()}", flush=True)
keys = list(space)
for vals in itertools.product(*[space[k_] for k_ in keys]):
    cfg = dict(fp8attn.CFG)
    upd = {f"{KERNEL}_{k_}": v_ for k_, v_ in zip(keys, vals)}
    cfg.update(upd)
    try:
        if KERNEL == "fwd":
            fn = lambda: fp8attn.fp8_attn_fwd(q, k, v, cfg=cfg)  # noqa
        else:
            fn = lambda: fp8attn.fp8_attn_bwd(q, k, v, o, do, lse, *extra, matched=True, cfg=cfg)  # noqa
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        t = [e.self_device_time_total / e.count for e in prof.key_averages() if e.key == name][0] / 1e3
        rg = getattr(fp8attn, "LAST", {}).get(KERNEL)
        print(f"  {upd} -> {t:.3f} ms  ({tflops(mm / 2 * flops_fwd(), t):.0f} TF executed) regs/spill/smem={rg}", flush=True)
    except Exception as e:  # noqa
        print(f"  {upd} -> FAIL {type(e).__name__}: {str(e)[:150]}", flush=True)
print("clock after", sm_clock())
