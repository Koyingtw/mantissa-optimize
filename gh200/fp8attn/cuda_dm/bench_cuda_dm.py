"""第 2 關效能：完整形狀（S=8192、64/8 heads、d=128、causal、B=1）attention backward 計時。

IMPL（逗號分隔）：cuda（本 CUDA 版，含量化前處理，matched 與 stale）、triton（fp8attn2，DS2）、
cudnn（TE 2.2 cuDNN FP8 fwd+bwd，NVTE_FP8_DPA_BWD=1，即訓練現況）、cudnn_bf16、fa3（FA3 bf16）。
LOAD=1：每次 attention bwd 之前在同一個 stream 先跑 LOAD_MS 毫秒的 FP8/bf16 GEMM（模擬訓練中凍結權重的
GEMM），讓模組功耗撞到上限、時脈被壓到訓練時的水準；只計 attention bwd 本身（CUDA events）。
每次計時同時用 NVML 讀 SM 時脈（bwd 結束時）；印中位數。
"""
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import S, HQ, HKV, D, flops_bwd, tflops  # noqa

IMPLS = os.environ.get("IMPL", "cuda,triton").split(",")
LOAD = os.environ.get("LOAD", "0") == "1"
LOAD_MS = float(os.environ.get("LOAD_MS", "30"))
ITERS = int(os.environ.get("ITERS", "30"))
WARM = int(os.environ.get("WARM", "5"))
scale = D ** -0.5
dev = "cuda"

try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def clk():
        return pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_SM)

    def pw():
        return pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0
except Exception:  # noqa
    def clk():
        return -1

    def pw():
        return -1

torch.manual_seed(0)
q = torch.randn(S, HQ, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(S, HKV, D, device=dev, dtype=torch.bfloat16)
do = torch.randn(S, HQ, D, device=dev, dtype=torch.bfloat16)

# background GEMM load (same stream): 8192 x 8192 x 8192 bf16 ~ 1.1 TFLOP each
A = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
Bm = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
C = torch.empty(8192, 8192, device=dev, dtype=torch.bfloat16)


def gemm_load(ms):
    if ms <= 0:
        return
    n = max(1, int(round(ms / 2.0)))   # ~2 ms per 8k^3 bf16 GEMM at 1300-1980 MHz
    for _ in range(n):
        torch.mm(A, Bm, out=C)


def timed(fn, setup=None):
    ts, cs, ps = [], [], []
    for i in range(WARM + ITERS):
        if setup:
            setup()
        if LOAD:
            gemm_load(LOAD_MS)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        if LOAD:
            gemm_load(3)       # keep power up while we read the clock
        torch.cuda.synchronize()
        if i >= WARM:
            ts.append(a.elapsed_time(b))
            cs.append(clk())
            ps.append(pw())
    ts.sort(); cs.sort(); ps.sort()
    m = len(ts) // 2
    return ts[m], ts[0], cs[m], ps[m]


def report(name, r):
    t, tmin, c, p = r
    print(f"RESULT {name:28s} bwd_ms={t:7.3f} (min {tmin:7.3f}) {tflops(flops_bwd(), t):4.0f} TF  "
          f"sm_clock_med={c} MHz power_med={p:.0f} W  load={'on' if LOAD else 'off'}", flush=True)


for impl in IMPLS:
    if impl in ("cuda", "triton"):
        import fp8attn2 as fa
        o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=fa.CFG)
        if impl == "cuda":
            import fp8dm_cuda as fc
            for matched in (True, False):
                r = timed(lambda: fc.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=matched))
                report(f"cuda {'matched' if matched else 'stale'}", r)
            # per-kernel breakdown
            from torch.profiler import profile, ProfilerActivity
            fc.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fc.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale)
                torch.cuda.synchronize()
            tot = 0.0
            for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total):
                tot += e.self_device_time_total / 1e3
                print(f"  kernel {e.key[:70]:70s} n={e.count:2d} {e.self_device_time_total / 1e3:7.3f} ms")
            print(f"  kernel sum {tot:.3f} ms")
        else:
            r = timed(lambda: fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=True))
            report("triton matched DS2", r)
    elif impl in ("cudnn", "cudnn_bf16"):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        os.environ["NVTE_FP8_DPA_BWD"] = "1"
        attn = te.DotProductAttention(HQ, D, num_gqa_groups=HKV, attention_dropout=0.0, qkv_format="sbhd",
                                      attn_mask_type="causal").to(dev)
        use_fp8 = impl == "cudnn"
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=128, amax_compute_algo="most_recent",
                                fp8_dpa=use_fp8)
        qq, kk, vv = (t.unsqueeze(1).detach().requires_grad_() for t in (q, k, v))
        dd = do.reshape(S, 1, HQ * D)
        hold = {}

        def setup():
            for t in (qq, kk, vv):
                t.grad = None
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=recipe):
                hold["o"] = attn(qq, kk, vv)

        r = timed(lambda: hold["o"].backward(dd), setup)
        report(f"cuDNN {'FP8' if use_fp8 else 'bf16'} (TE)", r)
    elif impl == "fa3":
        import flash_attn_interface as fa3
        qb, kb, vb = (t.unsqueeze(0).detach().requires_grad_() for t in (q, k, v))
        dob = do.unsqueeze(0)
        hold = {}

        def setup():
            for t in (qb, kb, vb):
                t.grad = None
            out = fa3.flash_attn_func(qb, kb, vb, causal=True)
            hold["o"] = out[0] if isinstance(out, tuple) else out

        r = timed(lambda: hold["o"].backward(dob), setup)
        report("FA3 bf16", r)
