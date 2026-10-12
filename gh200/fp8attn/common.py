"""Shared helpers for the FP8 attention study (shapes, timing, FLOP counts)."""
import os
import subprocess
import torch

# Llama2-70B LoRA (MLPerf), MBS=1: seq 8192, 64 q heads, 8 kv heads, head_dim 128, causal.
S = int(os.environ.get("ATTN_S", 8192))
B = int(os.environ.get("ATTN_B", 1))
HQ = int(os.environ.get("ATTN_HQ", 64))
HKV = int(os.environ.get("ATTN_HKV", 8))
D = int(os.environ.get("ATTN_D", 128))
SCALE = D ** -0.5


def flops_fwd(s=S, b=B, h=HQ, d=D, causal=True):
    f = 4.0 * b * h * s * s * d
    return f / 2 if causal else f


def flops_bwd(s=S, b=B, h=HQ, d=D, causal=True):
    return 2.5 * flops_fwd(s, b, h, d, causal)


def bench(fn, warmup=5, iters=20, setup=None):
    """Median / min time in ms with CUDA events. `setup` runs (untimed) before each call."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if setup:
            setup()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def sm_clock():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception as e:  # noqa
        return f"n/a ({e})"


def tflops(flop, ms):
    return flop / (ms * 1e-3) / 1e12
