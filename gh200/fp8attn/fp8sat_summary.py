#!/usr/bin/env python3
"""Summarise fp8sat.jsonl files written by fp8_scale_fix.py (FP8SAT_LOG=1).

    python3 fp8sat_summary.py RUN_DIR [RUN_DIR ...]

Per run: steps 1-3 in detail (layers with dO/dQKV all zero, layers saturated, max ratio),
then steps 4..end: fraction of (layer, step) pairs with r > 1 per slot, and the median / p99 of
the max ratio. r = amax * scale / fmax; r > 1 means the tensor was clipped at fmax that step.
For backward slots, r < 2^-16 (E5M2 min subnormal / max normal, about 2.7e-10 relative) means the
tensor flushed to zero entirely.
"""
import json
import os
import sys

SLOTS = (("QKV", "fwd_r", 0), ("O", "fwd_r", 1), ("S", "fwd_r", 2),
         ("dO", "bwd_r", 0), ("dP", "bwd_r", 1), ("dQKV", "bwd_r", 2))


THR = 1.001  # S (=P, amax 1) sits at r = 1 + float rounding; that is not clipping


def load(d):
    p = os.path.join(d, "fp8sat.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    for d in sys.argv[1:]:
        recs = load(d)
        print(f"## {os.path.basename(d.rstrip('/'))}  ({len(recs)} steps)")
        if not recs:
            continue
        print("| step | dO=0 layers | dQKV=0 layers | " + " | ".join(f"{n} nsat / rmax" for n, _, _ in SLOTS) + " |")
        print("|" + "---|" * (3 + len(SLOTS)))
        for r in recs[:3]:
            s = r["summary"]
            cells = []
            for n, key, i in SLOTS:
                rs = [L[key][i] for L in r["layers"].values() if key in L]
                cells.append(f"{sum(1 for x in rs if x > THR)} / {max(rs):.3g}" if rs else "n/a")
            print(f"| {s['step']} | {s['dO_zero_layers']} | {s['dQKV_zero_layers']} | " + " | ".join(cells) + " |")
        late = recs[3:]
        if late:
            cells = []
            for n, key, i in SLOTS:
                rs = [L[key][i] for r in late for L in r["layers"].values() if key in L]
                if rs:
                    frac = sum(1 for x in rs if x > THR) / len(rs)
                    cells.append(f"{100 * frac:.1f}% / p50 {pct(rs, .5):.2g} / p99 {pct(rs, .99):.3g} / max {max(rs):.3g}")
                else:
                    cells.append("n/a")
            print(f"| steps 4-{recs[-1]['summary']['step']} | | | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
