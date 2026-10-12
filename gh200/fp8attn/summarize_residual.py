#!/usr/bin/env python3
"""Compact table from residual_measure.py JSON output (no torch needed).

    python3 gh200/fp8attn/summarize_residual.py [results_dir] > results/summary_compact.md
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
fs = sorted(glob.glob(os.path.join(d, "L*_step*.json")),
            key=lambda f: (int(os.path.basename(f).split("step")[1][:-5]), f))
print("| dump | TE dq 相對誤差 (dk, dv) | TE vs 模擬 stale / DM（dq） | (TE−DM) 對 stale 預測斜率 | stale 項 / ‖dq‖ "
      "| 列和殘差 stale：中位 / 平均 / p99 | 有號平均 (t) | DM 量化 dS 後：中位 / p99 | dq 誤差：stale → DM "
      "| dS 不量化：stale → DM | block E4M3：stale → DM | block 列和殘差中位 |")
print("|" + "---|" * 12)
for f in fs:
    s = json.load(open(f))
    M, e = s["metrics"], s["te_vs_emulation"]
    a, st, dm = M["te_actual"], M["te_stale"], M["te_dm"]
    r = st["rowres_pre"]
    print(f"| L{s['layer']} s{s['step']} | {a['dq_relerr']:.3f} ({a['dk_relerr']:.3f}, {a['dv_relerr']:.3f}) "
          f"| {e['dq_te_vs_emul_stale_rel']:.3f} / {e['dq_te_vs_emul_dm_rel']:.3f} "
          f"| {e['dq_TEminusDM_on_stalepred_slope']:.2f} | {e['stale_term_norm_rel_to_dq_ref']:.3f} "
          f"| {r['median_abs']:.1e} / {r['mean_abs']:.1e} / {r['p99_abs']:.2f} | {r['mean_signed']:+.1e} ({r['t_stat']:+.1f}) "
          f"| {dm['rowres_post']['median_abs']:.1e} / {dm['rowres_post']['p99_abs']:.3f} "
          f"| {st['dq_relerr']:.3f} → {dm['dq_relerr']:.3f} "
          f"| {M['te_stale_dSfp32']['dq_relerr']:.3f} → {M['te_dm_dSfp32']['dq_relerr']:.3f} "
          f"| {M['blk_stale']['dq_relerr']:.3f} → {M['blk_dm']['dq_relerr']:.3f} "
          f"| {M['blk_stale']['rowres_pre']['median_abs']:.1e} |")
