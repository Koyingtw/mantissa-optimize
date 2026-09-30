#!/bin/bash
# One-off: finish the bits=1 run (job 1746) that the killed driver was waiting on,
# restore the caps, then start the 800 W rounding/sensitivity series from a copy.
set -uo pipefail
ROOT=/nfs/home/koying/mantissa-optimize
OUT=$ROOT/gh200/logs/lora
RUN=/nfs/home/koying/mlperf-llama2-70b-lora/runs/a3-bits1-seed1234-p575
while squeue -h -j 1746 2>/dev/null | grep -q .; do sleep 30; done
tr '\r' '\n' < "$RUN/slurm-1746.out" | grep '^:::MLLOG' > "$OUT/result_bits1_seed1234_p575.txt"
python3 - "$OUT/result_bits1_seed1234_p575.txt" <<'PY'
import json, sys
ev = [json.loads(l[l.index("{"):]) for l in open(sys.argv[1])]
t0 = next((e["time_ms"] for e in ev if e["key"] == "run_start"), None)
stop = next((e for e in ev if e["key"] == "run_stop"), None)
evals = [(e["metadata"]["samples_count"], round(e["value"], 4)) for e in ev if e["key"] == "eval_accuracy"]
print(f"variant=bits1 status={stop['metadata']['status'] if stop else 'no run_stop'}",
      f"converged_at={next((s for s, v in evals if v <= 0.925), '-')}",
      f"ttt_min={(stop['time_ms'] - t0) / 60000:.2f}" if stop and t0 else "ttt=-", f"evals={evals}")
PY
for f in "$RUN"/gpumon-*.csv; do cp "$f" "$OUT/bits1_p575_seed1234_$(basename "$f")"; done
# caps: job 1747 (14 nodes, 575 W) may start right after 1746 and sets/restores its own caps; the 800 W series sets its own
cp "$ROOT/gh200/slurm/lora_runs.sh" /tmp/lora_runs_p800.sh
VARIANTS="3 2:trunc 2:nearest 2:stochastic 2:trunc:mlp 2:trunc:self_attention" SEED=1234 POWER=800 bash /tmp/lora_runs_p800.sh
