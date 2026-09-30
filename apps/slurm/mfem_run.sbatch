#!/bin/bash
#SBATCH -J mfem-run
#SBATCH -p compute
#SBATCH -w work10
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/mfem_run_%j.log
# 用法:
#   sbatch --export=ALL,BIN=ex1_bench.build-cuda,ARGS="-n 64 -o 3 -rep 5" mfem_run.sbatch
#   截斷：再加 MANT_BITS=36[,MANT_ROUND=nearest]（未設定 = 原生，不做任何修改）
#   NSYS=1 → 用 nsys 收 kernel 統計（apps/logs/mfem_nsys_<jobid>.*）
#   RUNS="a|b|c"：同一個 job 內依序跑多組 "MANT_BITS:MANT_ROUND"（空字串 = 原生），例如 RUNS="|36:trunc|36:nearest"
set -e
module load nvhpc/26.5
BENCH=/nfs/home/koying/mantissa-optimize/apps/mfem-bench
LOGS=/nfs/home/koying/mantissa-optimize/apps/logs
BIN=${BIN:-ex1_bench.build-cuda}
ARGS=${ARGS:-"-n 64 -o 3"}
cd $BENCH
nvidia-smi --query-gpu=name,power.limit,temperature.gpu,clocks.max.sm --format=csv
run_one () {
  local tag=$1
  local CSV=$LOGS/mfem_run_${SLURM_JOB_ID}_${tag}.gpu.csv
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used,temperature.gpu --format=csv -lms 500 > $CSV &
  local SMI=$!
  echo "=== RUN tag=$tag MANT_BITS=${MANT_BITS:-<unset>} MANT_ROUND=${MANT_ROUND:-<unset>} BIN=$BIN ARGS=$ARGS"
  if [ -n "$NSYS" ]; then
    nsys profile -t cuda --stats=false -o $LOGS/mfem_nsys_${SLURM_JOB_ID}_${tag} -f true ./$BIN $ARGS
    nsys stats -r cuda_gpu_kern_sum -f csv -o $LOGS/mfem_nsys_${SLURM_JOB_ID}_${tag} $LOGS/mfem_nsys_${SLURM_JOB_ID}_${tag}.nsys-rep > /dev/null
  else
    /usr/bin/time -f "ELAPSED %e s  MAXRSS %M KB" ./$BIN $ARGS
  fi
  kill $SMI; wait $SMI 2>/dev/null || true
  python3 - $CSV <<'PY'
import csv,sys
r=list(csv.reader(open(sys.argv[1])))[1:]
p=[float(x[1].split()[0]) for x in r if len(x)>3 and 'W' in x[1]]
c=[float(x[2].split()[0]) for x in r if len(x)>3 and 'MHz' in x[2]]
u=[float(x[3].split()[0]) for x in r if len(x)>3 and '%' in x[3]]
# 只看「忙碌」樣本（util>=50%）
busy=[i for i,v in enumerate(u) if v>=50]
bp=[p[i] for i in busy]; bc=[c[i] for i in busy]; bu=[u[i] for i in busy]
print(f"GPU samples={len(p)} meanPower={sum(p)/len(p):.1f}W maxPower={max(p):.1f}W meanUtil={sum(u)/len(u):.1f}%")
if busy:
  print(f"GPU busy(util>=50%) samples={len(busy)} ({len(busy)*0.5:.1f}s) meanPower={sum(bp)/len(bp):.1f}W maxPower={max(bp):.1f}W meanSM={sum(bc)/len(bc):.0f}MHz meanUtil={sum(bu)/len(bu):.1f}%")
PY
}
if [ -n "$RUNS" ]; then
  IFS='|' read -ra LIST <<< "$RUNS"
  i=0
  for spec in "${LIST[@]}"; do
    if [ -z "$spec" ]; then unset MANT_BITS MANT_ROUND; tag="r${i}_native";
    else export MANT_BITS=${spec%%:*}; export MANT_ROUND=${spec##*:}; tag="r${i}_m${MANT_BITS}${MANT_ROUND}"; fi
    run_one $tag
    i=$((i+1))
  done
else
  run_one single
fi
