"""Prefill attention throughput with BF16 Q/K/V whose mantissa is reduced (FlashAttention backend).

Same command line and RESULT line as bench/gemm_mant_bench so slurm/e1_gemm_tdp.sbatch can run it
(dtype=attn, via bin/attn_bench): --frac keeps that many of the 7 BF16 mantissa bits in Q, K and V;
--m = sequence length, --n = heads, --k = head dim (defaults: Llama2-70B, 64 heads x 128, causal).
`tflops` counts the two attention matmuls (4 * L^2 * heads * dim / 2 for causal).  --check reports
the relative error of the output vs the unreduced inputs.  Unknown options are ignored.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_eval import reduce_bf16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="attn")
    ap.add_argument("--frac", type=int, default=7)
    ap.add_argument("--round", default="trunc")
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--low", default="zero")
    ap.add_argument("--check", action="store_true")
    a, _ = ap.parse_known_args()

    torch.manual_seed(1)
    shape = (a.batch, a.n, a.m, a.k)
    q0, k0, v0 = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q, k, v = (reduce_bf16(t, a.frac, a.round) for t in (q0, k0, v0))
    flop = 4.0 * a.batch * a.n * a.m * a.m * a.k / 2  # causal: half the score matrix

    from torch.nn.attention import SDPBackend, sdpa_kernel

    def run():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ref = F.scaled_dot_product_attention(q0, k0, v0, is_causal=True).float()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"RESULT tag={a.tag} dtype=attn frac={a.frac} only_a=0 low=zero round={a.round} m={a.m} n={a.n} k={a.k} "
          f"calls={calls} tflops={calls * flop / (busy * 1e-3) / 1e12:.3f} chunk_min={min(rates):.3f} "
          f"chunk_max={max(rates):.3f} t_begin={t_begin:.3f} t_end={t_end:.3f} rel_err={rel:.3e} emu_bits=-1",
          flush=True)


if __name__ == "__main__":
    main()
