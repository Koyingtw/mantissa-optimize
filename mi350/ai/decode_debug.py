"""Step-by-step check of the pieces of decode_bench.py on one GPU (it segfaulted after loading, job 375):
BF16 linear eager / in a HIP graph, FP8 _scaled_mm eager / in a HIP graph, for batch 1 and 64, on a few real-size
weights. Prints a line before and after each step, so the last line shows where it dies.
usage: decode_debug.py [--device 0]
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gh200", "infer"))
from ppl_eval import E4M3_KMAX, reduce_bf16, to_fp8  # noqa: E402

dev = torch.device("cuda", int(sys.argv[sys.argv.index("--device") + 1]) if "--device" in sys.argv else 0)
torch.cuda.set_device(dev)
ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for n, k in ((4096, 4096), (11008, 4096), (4096, 11008))]


def step(name, fn):
    print("BEGIN", name, flush=True)
    fn()
    torch.cuda.synchronize()
    print("OK   ", name, flush=True)


for batch in (1, 64):
    xs = {w.shape[1]: torch.randn(batch, w.shape[1], device=dev, dtype=torch.bfloat16) for w in ws}
    wr = [reduce_bf16(w, 3, "nearest") for w in ws]

    def bf16():
        for w in wr:
            F.linear(xs[w.shape[1]], w)

    step(f"bf16 eager b={batch}", bf16)

    def graph(fn):
        def go():
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                fn()
            g.replay()
        return go

    step(f"bf16 graph b={batch}", graph(bf16))
    m = batch + (-batch) % 16
    xp = {K: F.pad(x, (0, 0, 0, m - batch)) for K, x in xs.items()}
    wq, sc = [], []
    for w in ws:
        s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_KMAX[3]
        wq.append(to_fp8(w, s, 3, "nearest"))
        sc.append(s.reshape(1, -1).contiguous())

    def fp8():
        for w, s in zip(wq, sc):
            x = xp[w.shape[1]]
            sa = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_KMAX[3]
            xq = (x.float() / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
            torch._scaled_mm(xq, w.t(), scale_a=sa, scale_b=s, out_dtype=torch.bfloat16)

    step(f"fp8 eager b={batch}", fp8)
    step(f"fp8 graph b={batch}", graph(fp8))
print("ALL OK", flush=True)
