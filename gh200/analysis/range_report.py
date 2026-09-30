"""Summarise mantshim statistics: does an application use FP64's range or its precision?

Input: one or more `MANT_STATS` JSON files (one per process). Histograms from all
files given for one application are summed.

Range metrics (per operand A/B, over nonzero finite values):
  span_binades      log2(max) - log2(min) of the central 99.9% (p0.05..p99.95)
  fit_<fmt>         fraction of values that fall inside the best-placed window
                    of that format's dynamic range, i.e. what a per-tensor scale
                    could keep representable: fp8 e4m3 18 binades (2^-9..2^8.8),
                    fp8 e5m2 32, fp16 40, bf16/fp32 254
Precision metrics:
  sig_median        median number of significant fraction bits (52 - trailing zeros)
  sig_le_<k>        fraction of values that already fit in k fraction bits
usage: range_report.py name=file[,file...] [name=...] > table.csv
"""
import csv
import json
import sys

import numpy as np

WINDOWS = {"fit_fp8_e4m3": 18, "fit_fp8_e5m2": 32, "fit_fp16": 40, "fit_bf16": 254}


def best_window_fraction(exp_hist, width):
    """Largest fraction of values inside any window of `width` consecutive binades."""
    total = exp_hist.sum()
    if total == 0:
        return float("nan")
    c = np.concatenate([[0], np.cumsum(exp_hist)])
    w = min(width, len(exp_hist))
    return float(np.max(c[w:] - c[:-w]) / total)


def quantile_binade(exp_hist, q):
    c = np.cumsum(exp_hist) / exp_hist.sum()
    return int(np.searchsorted(c, q)) - 1023


def summarize(name, files):
    acc = None
    for f in files:
        d = json.load(open(f))
        if acc is None:
            acc = {k: (np.array(v, dtype=np.float64) if isinstance(v, list) else v) for k, v in d.items()}
            acc["calls"] = dict(d["calls"])
            acc["flops"] = dict(d["flops"])
        else:
            for k in ("a_exp", "a_sig", "b_exp", "b_sig"):
                acc[k] += np.array(d[k], dtype=np.float64)
            for k in ("a_zero", "b_zero", "a_nonfinite", "b_nonfinite"):
                acc[k] += d[k]
            for k, v in d["calls"].items():
                acc["calls"][k] = acc["calls"].get(k, 0) + v
            for k, v in d["flops"].items():
                acc["flops"][k] = acc["flops"].get(k, 0) + v
    rows = []
    for op in ("a", "b"):
        e, s = acc[f"{op}_exp"], acc[f"{op}_sig"]
        n = e.sum()
        if n == 0:
            continue
        lo, hi = quantile_binade(e, 0.0005), quantile_binade(e, 0.9995)
        row = {"app": name, "operand": op.upper(), "samples": int(n),
               "zero_frac": acc[f"{op}_zero"] / (n + acc[f"{op}_zero"]),
               "log2_p0.05": lo, "log2_median": quantile_binade(e, 0.5), "log2_p99.95": hi,
               "span_binades": hi - lo,
               "subnormal_frac": float(e[0] / n)}
        for key, width in WINDOWS.items():
            row[key] = best_window_fraction(e, width)
        cs = np.cumsum(s) / s.sum()
        row["sig_median"] = int(np.searchsorted(cs, 0.5))
        for k in (7, 10, 23, 36):
            row[f"sig_le_{k}"] = float(cs[k])
        row["gemm_calls"] = int(sum(acc["calls"].values()))
        row["gemm_tflop"] = float(sum(acc["flops"].values()) / 1e12)
        rows.append(row)
    return rows


def main():
    out = []
    for arg in sys.argv[1:]:
        name, files = arg.split("=", 1)
        out += summarize(name, files.split(","))
    w = csv.DictWriter(sys.stdout, fieldnames=list(out[0]))
    w.writeheader()
    for r in out:
        w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
