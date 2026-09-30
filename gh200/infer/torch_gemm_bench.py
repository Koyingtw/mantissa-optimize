"""Prefill GEMM throughput on real Llama2-70B operands (dump_act.py) with reduced mantissa.

Same command line / RESULT line as bench/gemm_mant_bench so slurm/e1_gemm_tdp.sbatch can run it
(bin/torch_gemm_bench).  dtype rbf16: BF16 GEMM (cuBLAS, FP32 accumulate) on X @ W^T with <frac> of
7 mantissa bits kept; dtype rfp8: E4M3 per-tensor scales (max of the reduced format) and
torch._scaled_mm, <frac> of 3 bits kept.  --low wonly (or --only-b) reduces only the weights.
Round-to-nearest by default.  X is the layer-40 MLP input (tokens x 8192), W the stacked
gate/up weights (57344 x 8192).  --check: relative error vs the FP32 product of the original data.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_eval import E4M3_KMAX, reduce_bf16, to_fp8  # noqa: E402


def load(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.int16)).view(torch.bfloat16).view(*shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="rfp8")
    ap.add_argument("--frac", type=int, default=-1)
    ap.add_argument("--round", default="nearest")
    ap.add_argument("--low", default="zero")
    ap.add_argument("--only-b", action="store_true")
    ap.add_argument("--meta", default="/nfs/home/koying/models/llama2_70b_operands/operands_L40.json")
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="use only the first ROWS tokens (e.g. 64 = decode)")
    a, _ = ap.parse_known_args()
    wonly = a.only_b or a.low == "wonly"
    native = 3 if a.dtype == "rfp8" else 7
    k = native if a.frac < 0 or a.frac > native else a.frac
    kx = native if wonly else k
    meta = json.load(open(a.meta))
    x0 = load(meta["x"], meta["x_shape"])
    if a.rows:
        x0 = x0[:a.rows]
    x0 = x0.contiguous().cuda()
    w0 = load(meta["w"], meta["w_shape"]).cuda()
    m, kk = x0.shape
    n = w0.shape[0]
    if a.dtype == "rbf16":
        x, w = reduce_bf16(x0, kx, a.round), reduce_bf16(w0, k, a.round)
        run = lambda: x @ w.t()  # noqa: E731
    else:
        sx = (x0.float().abs().amax() / E4M3_KMAX[kx]).reshape(1, 1)
        sw = (w0.float().abs().amax() / E4M3_KMAX[k]).reshape(1, 1)
        x, w = to_fp8(x0, sx, kx, a.round), to_fp8(w0, sw, k, a.round)
        run = lambda: torch._scaled_mm(x, w.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)  # noqa: E731
    flop = 2.0 * m * n * kk
    t = time.time()
    while time.time() - t < a.warmup:
        run()
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rates, busy, calls = [], 0.0, 0
    t_begin = time.time()
    while time.time() - t_begin < a.seconds:
        e0.record()
        for _ in range(5):
            out = run()
        e1.record()
        e1.synchronize()
        ms = e0.elapsed_time(e1)
        busy += ms
        calls += 5
        rates.append(5 * flop / (ms * 1e-3) / 1e12)
    t_end = time.time()
    rel = -1.0
    if a.check:
        ref = x0.float() @ w0.float().t()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"RESULT tag={a.tag} dtype={a.dtype} frac={k} only_a=0 only_b={int(wonly)} low={'wonly' if wonly else 'zero'} "
          f"round={a.round} m={m} n={n} k={kk} calls={calls} tflops={calls * flop / (busy * 1e-3) / 1e12:.3f} "
          f"chunk_min={min(rates):.3f} chunk_max={max(rates):.3f} t_begin={t_begin:.3f} t_end={t_end:.3f} "
          f"rel_err={rel:.3e} emu_bits=-1", flush=True)


if __name__ == "__main__":
    main()
