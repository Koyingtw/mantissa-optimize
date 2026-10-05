#!/bin/bash
#SBATCH -J exageostat-run
#SBATCH -p compute
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/exageostat_run_%j.log
# ExaGeoStatCPP（StarPU + Chameleon + cuBLAS）合成資料 MLE，單一 GH200
# 用法:
#   sbatch --export=ALL,N=40000,DTS=960,ITERS=5 apps/slurm/exageostat_run.sbatch
#   PRELOAD=/nfs/home/koying/mantissa-optimize/gh200/shim/libmantshim.span.so  → LD_PRELOAD shim（再加 MANT_BITS=…、MANT_VERBOSE=1 …）
#   參數（皆有預設）:
#     N=20000 DTS=960 CORES=64 GPUS=1 ITERS=5（NLopt BOBYQA 最大評估次數）TOL=9（ftol_abs = 1e-TOL）
#     THETA=1:0.1:0.5（產生合成資料用的真值 σ²:β:ν）LB=0.1:0.1:0.1 UB=5:5:5
#     KERNEL=univariate_matern_stationary  SEED=0  VERBOSE=detailed
#     STARPU_SCHED（預設 dmdas）、STARPU_CALIBRATE、STARPU_HOME（預設 apps/exageostat-deps/starpu-home，效能模型存這裡）
#   RUNS="a|b|c"：同一 job 依序跑多組 "MANT_BITS:MANT_ROUND"（空字串 = 原生），需搭配 PRELOAD
set -e
D=/nfs/home/koying/mantissa-optimize/apps/exageostat-deps
INST=$D/install
LOGS=/nfs/home/koying/mantissa-optimize/apps/logs
BIN=${BIN:-$D/build/exageostatcpp/examples/end-to-end/Example_Data_Generation_and_Modeling}
CUDA=/usr/local/cuda-13.3
export LD_LIBRARY_PATH=$INST/OPENBLAS/lib:$INST/HWLOC/lib:$INST/GSL/lib:$INST/NLOPT/lib:$INST/STARPU/lib:$INST/CHAMELEON/lib:$CUDA/lib64:$LD_LIBRARY_PATH
export STARPU_HOME=${STARPU_HOME:-$D/starpu-home}
export STARPU_SCHED=${STARPU_SCHED:-dmdas}
# 效能模型檔名預設含主機名；各 GH200 節點硬體相同，統一名稱讓校正結果可跨節點共用
export STARPU_HOSTNAME=${STARPU_HOSTNAME:-gh200}
export STARPU_WORKER_STATS=${STARPU_WORKER_STATS:-1}
export STARPU_SILENT=${STARPU_SILENT:-0}
export OPENBLAS_NUM_THREADS=1
mkdir -p $STARPU_HOME
N=${N:-20000}; DTS=${DTS:-960}; CORES=${CORES:-64}; GPUS=${GPUS:-1}; ITERS=${ITERS:-5}; TOL=${TOL:-9}
THETA=${THETA:-1:0.1:0.5}; LB=${LB:-0.1:0.1:0.1}; UB=${UB:-5:5:5}
KERNEL=${KERNEL:-univariate_matern_stationary}; SEED=${SEED:-0}; VERBOSE=${VERBOSE:-detailed}
ARGS="--N=$N --dts=$DTS --kernel=$KERNEL --computation=exact --precision=double --cores=$CORES --gpus=$GPUS \
 --initial_theta=$THETA --lb=$LB --ub=$UB --max_mle_iterations=$ITERS --tolerance=$TOL --seed=$SEED --verbose=$VERBOSE"
WD=$LOGS/exageostat_run_${SLURM_JOB_ID}
mkdir -p $WD && cd $WD
echo "== host $(hostname) $(date)"
nvidia-smi --query-gpu=name,driver_version,power.limit,temperature.gpu,clocks.max.sm --format=csv
run_one () {
  local tag=$1
  local CSV=$WD/${tag}.gpu.csv ERR=$WD/${tag}.stderr
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,memory.used,temperature.gpu --format=csv -lms 500 > $CSV &
  local SMI=$!
  echo "=== RUN tag=$tag PRELOAD=${PRELOAD:-<none>} MANT_BITS=${MANT_BITS:-<unset>} MANT_ROUND=${MANT_ROUND:-<unset>} MANT_VERBOSE=${MANT_VERBOSE:-<unset>}"
  echo "    $BIN $ARGS"
  local t0=$(date +%s.%N)
  if [ -n "$PRELOAD" ]; then
    env LD_PRELOAD=$PRELOAD /usr/bin/time -f "ELAPSED %e s  MAXRSS %M KB" $BIN $ARGS 2> $ERR
  else
    /usr/bin/time -f "ELAPSED %e s  MAXRSS %M KB" $BIN $ARGS 2> $ERR
  fi
  local rc=$?
  local t1=$(date +%s.%N)
  kill $SMI; wait $SMI 2>/dev/null || true
  echo "    rc=$rc wall=$(python3 -c "print(f'{$t1-$t0:.1f}')") s"
  grep -v '^\[mantshim\]' $ERR | tail -40
  if [ -n "$PRELOAD" ]; then
    echo "    [mantshim] lines: $(grep -c '^\[mantshim\]' $ERR || true)  gemm-lines: $(grep -c '^\[mantshim\] gemm ' $ERR || true)  cublasDgemm-lines: $(grep -c '^\[mantshim\] cublasDgemm ' $ERR || true)"
    grep '^\[mantshim\] gemm ' $ERR | awk '{print $4,$5,$6}' | sort | uniq -c | sort -rn | head -8
  fi
  python3 - $CSV <<'PY'
import csv,sys
r=list(csv.reader(open(sys.argv[1])))[1:]
p=[float(x[1].split()[0]) for x in r if len(x)>3 and 'W' in x[1]]
c=[float(x[2].split()[0]) for x in r if len(x)>3 and 'MHz' in x[2]]
u=[float(x[3].split()[0]) for x in r if len(x)>3 and '%' in x[3]]
m=[float(x[4].split()[0]) for x in r if len(x)>4 and 'MiB' in x[4]]
if p:
  print(f"    GPU samples={len(p)} meanPower={sum(p)/len(p):.1f}W maxPower={max(p):.1f}W meanUtil={sum(u)/len(u):.1f}% maxMem={max(m):.0f}MiB")
busy=[i for i,v in enumerate(u) if v>=50]
if busy:
  bp=[p[i] for i in busy]; bc=[c[i] for i in busy]; bu=[u[i] for i in busy]
  print(f"    GPU busy(util>=50%) samples={len(busy)} ({len(busy)*0.5:.1f}s) meanPower={sum(bp)/len(bp):.1f}W maxPower={max(bp):.1f}W meanSM={sum(bc)/len(bc):.0f}MHz")
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
echo "== done $(date)"
