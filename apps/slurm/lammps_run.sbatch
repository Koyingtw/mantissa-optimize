#!/bin/bash
#SBATCH -J lammps-run
#SBATCH -p compute
#SBATCH -w work11
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/lammps_run_%j.log
# LAMMPS KOKKOS CUDA (FP64) 單 GPU 執行 + nvidia-smi 功耗記錄
# 用法:
#   sbatch --export=ALL,CASE=lj  apps/slurm/lammps_run.sbatch
#   sbatch --export=ALL,CASE=eam,X=8,NSTEPS=1000 apps/slurm/lammps_run.sbatch
#   截斷: --export=ALL,CASE=eam,MANT_BITS=36,MANT_ROUND=nearest
#   多次（同一 job 內依序）: RUNS="native:unset trunc36:36:trunc near36:36:nearest"
#   第 4 欄可指定該次用的執行檔: "orig:unset:trunc:/path/lmp_native_unmodified"
#   逐位比對（GPU 版因 atomic binning 不可逐位重現）: SF=kk/host X=1 NSTEPS=200
#   逐原子力逐位比對: DUMP=1 NSTEPS=0 X=2（run 0 後 write_dump %.17g，cmp 比較）
#   其他: LMP=<binary>  NEIGH=full|half  NEWTON=off|on  NTHERMO=100  LVARS="-var eps 1.1"
set -e
module load nvhpc/26.5
export OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads
CASE=${CASE:-lj}; X=${X:-8}; NSTEPS=${NSTEPS:-1000}; NTHERMO=${NTHERMO:-100}
NEIGH=${NEIGH:-full}; NEWTON=${NEWTON:-off}; SF=${SF:-kk}   # SF=kk/host → 同一 Kokkos 模板的 host 實例（1 thread，可逐位重現）
LMP=${LMP:-/nfs/home/koying/mantissa-optimize/apps/lammps/build-kokkos-cuda/lmp}
IN=/nfs/home/koying/mantissa-optimize/apps/lammps-inputs
RUNS=${RUNS:-"run:${MANT_BITS:-unset}:${MANT_ROUND:-trunc}"}
cd $IN
TAG=${CASE}_x${X}_${SLURM_JOB_ID}
nvidia-smi --query-gpu=name,power.limit,enforced.power.limit,clocks.max.sm --format=csv
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown --format=csv -lms 500 > $TAG.gpu.csv &
SMI=$!
for R in $RUNS; do
  IFS=: read -r NAME MB MR RBIN <<< "$R"; RBIN=${RBIN:-$LMP}
  unset MANT_BITS MANT_ROUND
  [ "$MB" != "unset" ] && export MANT_BITS=$MB MANT_ROUND=$MR
  echo "=== $NAME  bin=$RBIN  MANT_BITS=${MANT_BITS:-<unset>} MANT_ROUND=${MANT_ROUND:-<unset>}  start $(date +%H:%M:%S.%N)"
  date +%s.%N > $TAG.$NAME.t0
  mpirun -np 1 --bind-to none ${MANT_BITS:+-x MANT_BITS -x MANT_ROUND} $RBIN -k on g 1 t 1 -sf $SF \
     -pk kokkos newton $NEWTON neigh $NEIGH -in in.$CASE.kk \
     -var x $X -var y $X -var z $X -var nsteps $NSTEPS -var nthermo $NTHERMO $LVARS ${DUMP:+-var dodump 1 -var dumpf $TAG.$NAME.dump} \
     -log $TAG.$NAME.log -screen none || echo "lmp exit $?"
  date +%s.%N > $TAG.$NAME.t1
  echo "=== $NAME end $(date +%H:%M:%S.%N)"
  grep -E "^\[MANT\]|atoms in group|Loop time|Performance|Total wall|^ +[0-9]+ +[-0-9]" $TAG.$NAME.log | head -40
done
kill $SMI
if [ -n "$DUMP" ]; then   # 逐位比對每個原子的力（第一個 run 當基準）
  set -- $RUNS; REF=${1%%:*}
  for R in $RUNS; do N=${R%%:*}; [ $N = $REF ] && continue
    if cmp -s $TAG.$REF.dump $TAG.$N.dump; then echo "[bitwise] $N vs $REF: IDENTICAL forces (cmp)"; else
    /nfs/home/koying/mantissa-optimize/gh200/.venv/bin/python - $TAG.$REF.dump $TAG.$N.dump $N $REF <<'PY2'
import sys, numpy as np
a=np.loadtxt(sys.argv[1],skiprows=9); b=np.loadtxt(sys.argv[2],skiprows=9)
fa=a[:,1:4]; fb=b[:,1:4]; d=np.abs(fa-fb); nz=np.count_nonzero(d)
rel=np.max(d)/np.max(np.abs(fa))
print(f"[bitwise] {sys.argv[3]} vs {sys.argv[4]}: DIFFERENT components={nz}/{d.size} max|df|={np.max(d):.3e} max|df|/max|f|={rel:.3e} rms_rel={np.sqrt(np.mean(d**2))/np.sqrt(np.mean(fa**2)):.3e}")
PY2
    fi
  done
fi
/nfs/home/koying/mantissa-optimize/gh200/.venv/bin/python - "$TAG" $RUNS <<'PY'
import csv, sys
tag = sys.argv[1]; runs = [r.split(':')[0] for r in sys.argv[2:]]
from datetime import datetime
rows = list(csv.reader(open(f"{tag}.gpu.csv")))[1:]
def ts(s): return datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
for name in runs:
    t0 = float(open(f"{tag}.{name}.t0").read()); t1 = float(open(f"{tag}.{name}.t1").read())
    sel = [r for r in rows if t0 <= ts(r[0]) <= t1]
    # 去掉前 20% 的啟動/建構階段只留穩態（loop 階段）
    p = [float(r[1].split()[0]) for r in sel]; c = [float(r[2].split()[0]) for r in sel]
    u = [float(r[3].split()[0]) for r in sel]; tmp = [float(r[4]) for r in sel]
    cap = sum(1 for r in sel if 'Active' in r[5] and 'Not' not in r[5])
    hw = sum(1 for r in sel if 'Active' in r[6] and 'Not' not in r[6])
    busy = [i for i, x in enumerate(u) if x >= 50]
    pb = [p[i] for i in busy]; cb = [c[i] for i in busy]; ub = [u[i] for i in busy]
    print(f"[{name}] wall={t1-t0:.1f}s samples={len(p)} meanP={sum(p)/len(p):.0f}W maxP={max(p):.0f}W "
          f"meanUtil={sum(u)/len(u):.0f}% | busy(util>=50%) n={len(pb)} meanP={sum(pb)/max(1,len(pb)):.0f}W "
          f"meanSM={sum(cb)/max(1,len(cb)):.0f}MHz meanUtil={sum(ub)/max(1,len(ub)):.0f}% maxT={max(tmp):.0f}C "
          f"sw_power_cap_samples={cap} hw_slowdown_samples={hw}")
PY
