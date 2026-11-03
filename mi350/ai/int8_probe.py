"""Can decode_bench use a real INT8 GEMM with int32 accumulation on MI350X (asked by the cold-gemm session)?
Tries torch._int_mm (int8 x int8 -> int32) at decode-like shapes (M tokens x K=4096, weight K x N=11008 and the
transposed-weight layout F.linear would use), checks it against an exact float64 reference, and times it next to BF16.
usage: int8_probe.py
"""
import time

import torch

dev = "cuda"
K, N = 4096, 11008
torch.manual_seed(0)
print("torch", torch.__version__, "hip", torch.version.hip, torch.cuda.get_device_name(), flush=True)


def timeit(fn, reps=50):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.time() - t) / reps


for M in (1, 8, 16, 17, 32, 64, 256):
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)  # weights as stored by nn.Linear: N x K
    xb, wb = a.to(torch.bfloat16), w.to(torch.bfloat16)
    ref = a.double() @ w.double().t()
    for layout, fn in (("w.t()", lambda: torch._int_mm(a, w.t())),
                       ("contig KxN", lambda: torch._int_mm(a, w.t().contiguous()))):
        try:
            y = fn()
            ok = bool((y.double() == ref).all())
            ms = timeit(fn)
            print(f"M={M:4d} {layout:11s} ok={ok} dtype={y.dtype} int8 {ms:.3f} ms  bf16 "
                  f"{timeit(lambda: torch.nn.functional.linear(xb, wb)):.3f} ms", flush=True)
            # which GPU kernels the call dispatches to (the cold-gemm session compares with its hipBLASLt runs)
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            names = sorted({e.key for e in prof.key_averages() if e.self_device_time_total > 0})
            print(f"M={M:4d} {layout:11s} kernels: {' | '.join(names)}", flush=True)
        except Exception as e:  # report the constraint, keep probing
            print(f"M={M:4d} {layout:11s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
