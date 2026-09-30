"""E4: numerical error of GEMM when operand mantissas are reduced.

A, B ~ U[-1, 1) in FP64.  Each operand keeps `frac` explicit fraction bits
(FP64: of 52; the BF16 family: the float32 value keeps `frac` of 23, frac <= 7
is exactly a BF16).  The error of C = A_k B_k is computed as
dA B + A dB + dA dB in FP64, where dA = A_k - A is exact, so it is accurate to
FP64 relative to the error itself rather than to C.

Outputs one CSV row per (family, rounding, frac, K, trial):
  op_rms      RMS relative operand error   |x_k - x| / |x|
  op_bias     mean signed relative operand error (x_k - x) / x
  c_rel_fro   ||dC||_F / ||C||_F
  c_scale     least-squares slope of C_k on C, minus 1 (systematic scaling bias)
"""
import argparse
import csv
import sys

import numpy as np


def reduce_bits(x, frac, mode, rng, family):
    if family == "fp64":
        width, native, view, one = 52, 52, np.uint64, np.uint64(1)
        y = x.copy()
    else:
        width, native, view, one = 23, 23, np.uint32, np.uint32(1)
        y = x.astype(np.float32)
    if frac >= native:
        return y.astype(np.float64)
    drop = width - frac
    u = y.view(view)
    if mode == "nearest":
        u += view(1) << view(drop - 1)
    elif mode == "stochastic":
        u += rng.integers(0, 1 << drop, size=u.shape, dtype=view)
    u &= ~((one << view(drop)) - one)
    return u.view(y.dtype).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--ks", default="256,1024,4096,16384")
    args = ap.parse_args()

    fam_fracs = {
        "fp64": [52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0],
        "bf16": [23, 7, 6, 5, 4, 3, 2, 1, 0],
    }
    ks = [int(k) for k in args.ks.split(",")]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "rounding", "frac", "K", "trial", "op_rms", "op_bias", "c_rel_fro", "c_scale"])
        for family, fracs in fam_fracs.items():
            for K in ks:
                for trial in range(args.trials):
                    rng = np.random.default_rng(1000 * K + trial)
                    A = rng.uniform(-1, 1, (args.m, K))
                    B = rng.uniform(-1, 1, (K, args.m))
                    C = A @ B
                    cc = float(np.vdot(C, C))
                    for mode in ["trunc", "nearest", "stochastic"]:
                        for frac in fracs:
                            Ak = reduce_bits(A, frac, mode, rng, family)
                            Bk = reduce_bits(B, frac, mode, rng, family)
                            dA, dB = Ak - A, Bk - B
                            dC = dA @ B + A @ dB + dA @ dB
                            rel = np.concatenate([(dA / A).ravel(), (dB / B).ravel()])
                            w.writerow([family, mode, frac, K, trial,
                                        float(np.sqrt(np.mean(rel * rel))), float(np.mean(rel)),
                                        float(np.sqrt(np.vdot(dC, dC) / cc)),
                                        float(np.vdot(C, dC) / cc)])
                    f.flush()
                    print(f"{family} K={K} trial={trial} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
