"""3xTF32 GEMM (FP32-accurate GEMM on TF32 tensor cores) under a power cap, for the equal-accuracy
comparison with FP64 truncation and cuBLAS FP64 emulation (reviewer request).

Each FP32 operand is split into big = tf32(x) and small = tf32(x - big) (round to nearest); then
C = big_a @ big_b + big_a @ small_b + small_a @ big_b with TF32 tensor-core GEMMs and FP32
accumulation (the small @ small term is dropped, as in CUTLASS's 3xTF32).  dtype tf32x3 (this),
tf32x1 (one TF32 GEMM) or fp32 (plain SGEMM).  Operands are uniform [-1, 1) like gemm_mant_bench;
--check reports the relative Frobenius error vs the FP64 product of the original FP64 data (the
FP32 conversion is part of the method's error, as for an FP64 user).  RESULT line as
gemm_mant_bench so e1_gemm_tdp.sbatch can run it (bin/tf32x3_bench).
"""
import argparse
import time

import torch


def tf32_round(x):
    """Round FP32 to TF32 (10 explicit mantissa bits, nearest-even) and keep it in FP32 storage."""
    u = x.view(torch.int32)
    u = u + 0xFFF + ((u >> 13) & 1)
    return (u & ~0x1FFF).view(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="tf32x3")
    ap.add_argument("--frac", type=int, default=23)
    ap.add_argument("--m", type=int, default=32768)
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--low", default="zero")
    ap.add_argument("--check", action="store_true")
    a, _ = ap.parse_known_args()

    g = torch.Generator(device="cuda").manual_seed(1)
    A64 = torch.rand(a.m, a.k, device="cuda", dtype=torch.float64, generator=g) * 2 - 1
    B64 = torch.rand(a.k, a.n, device="cuda", dtype=torch.float64, generator=g) * 2 - 1
    A, B = A64.float(), B64.float()
    if not a.check:
        del A64, B64
    if a.dtype == "tf32x3":
        torch.backends.cuda.matmul.allow_tf32 = True
        Ab, Bb = tf32_round(A), tf32_round(B)
        As, Bs = tf32_round(A - Ab), tf32_round(B - Bb)
        run = lambda: Ab @ Bb + (Ab @ Bs + As @ Bb)  # noqa: E731  (small terms summed first)
        gemms = 3
    elif a.dtype == "tf32x1":
        torch.backends.cuda.matmul.allow_tf32 = True
        run = lambda: A @ B  # noqa: E731
        gemms = 1
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        run = lambda: A @ B  # noqa: E731
        gemms = 1
    flop = 2.0 * a.m * a.n * a.k  # useful flops (one product), not the 3x executed
    t = time.time()
    while time.time() - t < a.warmup:
        run()
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rates, busy, calls = [], 0.0, 0
    t_begin = time.time()
    while time.time() - t_begin < a.seconds:
        e0.record()
        for _ in range(3):
            out = run()
        e1.record()
        e1.synchronize()
        ms = e0.elapsed_time(e1)
        busy += ms
        calls += 3
        rates.append(3 * flop / (ms * 1e-3) / 1e12)
    t_end = time.time()
    rel = -1.0
    if a.check:
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = A64 @ B64  # error vs the original FP64 data, so FP32 input rounding counts
        rel = ((out.double() - ref).norm() / ref.norm()).item()
    print(f"RESULT tag={a.tag} dtype={a.dtype} frac=23 only_a=0 low=zero round=nearest m={a.m} n={a.n} k={a.k} "
          f"calls={calls} tflops={calls * flop / (busy * 1e-3) / 1e12:.3f} chunk_min={min(rates):.3f} "
          f"chunk_max={max(rates):.3f} t_begin={t_begin:.3f} t_end={t_end:.3f} rel_err={rel:.3e} emu_bits=-1 "
          f"gemms={gemms}", flush=True)


if __name__ == "__main__":
    main()
