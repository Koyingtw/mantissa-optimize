#!/bin/bash
# E2/E3（MI350X）：單節點 8 GPU 的 HPL-MxP（BF16 storage）或 FP64 HPL（rocHPL），× PPT0 上限 × 尾數設定。
# 每個上限把變體清單正向再反向各跑一次（ABBA）；REPEAT=r 重複 r 個 ABBA 區塊。
#   BENCH=mxp nohup mi350/scripts/hpl_tdp.sh >/dev/null 2>&1 &
#   BENCH=hpl nohup mi350/scripts/hpl_tdp.sh >/dev/null 2>&1 &
# 環境變數：CAPS、VARIANTS（mxp：保留的 BF16 小數位元 native|3|0…，對應 build_mi350_bf16_mant<k>；
#   hpl：A_BITS:B_BITS，53 = 原生 binary64；g8m<k> = GEMMul8 k moduli）、N、NB、REPEAT、START_TEMP、COOL_MAX、TAG、WARMUP=1（先跑一次不計）
# 每次執行的輸出以 "=== RUN ..." / "=== END ..." 包起來，附 Unix 時間，供 analysis 與功耗 log 對齊。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$ROOT/.." && pwd)
cd "$ROOT"
source env.sh
BENCH=${BENCH:-mxp}
CAPS=${CAPS:-"1000 800 650 500"}
REPEAT=${REPEAT:-1}
START_TEMP=${START_TEMP:-63}
COOL_MAX=${COOL_MAX:-120}
ID=${TAG:-$BENCH}_$(date +%m%d_%H%M)
OUT=logs/$ID.out
mkdir -p logs
exec >"$OUT" 2>&1
export UCX_WARN_UNUSED_ENV_VARS=n UCX_TLS=sysv,self,rocm_ipc,rocm_copy
MPI_ARGS=(--map-by ppr:8:node --rank-by slot --bind-to none --mca pml ucx --mca pml_ucx_tls "$UCX_TLS"
          --mca pml_ucx_devices any --mca btl '^vader,tcp,openib,uct')

if [[ $BENCH == mxp ]]; then
  N=${N:-1034752}; NB=${NB:-5504}; P=${P:-4}; Q=${Q:-2}
  VARIANTS=${VARIANTS:-"native 3 0"}
  export HPLMXP_TRAILING_HIPBLAS=1 HPLMXP_IR_RICHARDSON=1 HPLMXP_PRECOND_FP32=1 HPLMXP_PROF=${HPLMXP_PROF:-0} HPLMXP_PROGRESS=0
  export HPLMXP_ROCTX=0 HPLMXP_TRAILING_TN=1 HPLMXP_TN_DIRECT_PANEL=1
  export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8
  export HPLMXP_DEVICE_MAP=${HPLMXP_DEVICE_MAP:-23106754}
  DIAG_ARG=${DIAG_ARG:---approx-diag}
  run_variant() {
    mpirun -np 8 "${MPI_ARGS[@]}" -x HPLMXP_DEVICE_MAP bash -c \
      'export ROCR_VISIBLE_DEVICES=${HPLMXP_DEVICE_MAP:$OMPI_COMM_WORLD_LOCAL_RANK:1}; exec "$@"' wrap \
      "$REPO/rocHPL-MxP/build_mi350_bf16_mant$1/xhplmxp-mi350" --n "$N" --nb "$NB" --p "$P" --q "$Q" \
      --fp8 0 --itcheck 1 "$DIAG_ARG"
  }
else
  # rocHPL feat/adaptive-mixed-storage（worktree ~/koying/sw/rocHPL-amd），先前 MI350 調校的形狀（docs/16、18）
  N=${N:-540672}; NB=${NB:-1024}; P=${P:-1}; Q=${Q:-8}; F=${F:-0.1}
  VARIANTS=${VARIANTS:-"53:53 36:36"}
  BIN=${HPL_BIN:-$SW/rocHPL-amd/build-mi350/bin/rochpl}
  export HPL_QUANT_C_BITS=53 HPL_QUANT_ROUND=${HPL_QUANT_ROUND:-nearest} HPL_QUANT_MODE=scalar HPL_HIPBLASLT=0
  export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=spread
  # g8m<k>：同一 commit 以 -DHPL_USE_GEMMUL8=ON 建的執行檔，trailing update 由 GEMMul8（Ozaki-II，INT8，k 個 moduli）
  # 計算，運算元保留 53 bits；HPL_GEMMUL8_NT（預設 12288，先前 MI350 紀錄 docs/13 的設定）
  BIN_G8=${HPL_BIN_G8:-$SW/rocHPL-amd/build-mi350-g8/bin/rochpl}
  run_variant() {
    local bin=$BIN
    if [[ $1 == g8m* ]]; then
      bin=$BIN_G8
      export HPL_QUANT_A_BITS=53 HPL_QUANT_B_BITS=53 HPL_QUANT_B_BITS_ALT=53
      export HPL_GEMMUL8=1 HPL_GEMMUL8_NUM_MOD=${1#g8m} HPL_GEMMUL8_FASTMODE=1 HPL_GEMMUL8_NT=${HPL_GEMMUL8_NT:-12288}
    else
      # B_BITS_ALT 未設時預設等於 B（見 HPL_quantize_device.cpp），這裡明確給值以免沿用舊環境
      export HPL_QUANT_A_BITS=${1%:*} HPL_QUANT_B_BITS=${1#*:} HPL_QUANT_B_BITS_ALT=${1#*:}
      unset HPL_GEMMUL8 HPL_GEMMUL8_NUM_MOD
    fi
    sleep 20   # 上一輪約 250 GiB 的裝置記憶體要等 driver 回收，否則 HPL_pdmatgen 配置失敗
    # rochpl 只把結果寫到工作目錄的 HPL.out（每次覆寫）：在各自的目錄執行，再把 HPL.out 併進 log
    local d="logs/${ID}_run/$(date +%s)"
    mkdir -p "$d"
    (cd "$d" && mpirun -np 8 "${MPI_ARGS[@]}" bash -c 'export ROCR_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; exec "$@"' \
      wrap "$bin" -P "$P" -Q "$Q" -p "$P" -q "$Q" -N "$N" -NB "$NB" -f "$F")
    local rc=$?
    cat "$d/HPL.out"
    return $rc
  }
fi

restore() { tools/pcap.sh 1000 || true; kill "${SMI_PID:-0}" 2>/dev/null || true; }
trap restore EXIT
.venv/bin/python tools/smi_sampler.py "logs/${ID}_smi.csv" 0.1 &
SMI_PID=$!
hot_max() { local m=0 t; for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input; do
  t=$(( $(cat "$f") / 1000 )); (( t > m )) && m=$t; done; echo $m; }

echo "id=$ID bench=$BENCH N=$N NB=$NB P=$P Q=$Q caps=[$CAPS] variants=[$VARIANTS] repeat=$REPEAT"
read -ra var <<<"$VARIANTS"
rev=()
for ((i = ${#var[@]} - 1; i >= 0; i--)); do rev+=("${var[i]}"); done
passes=()
[[ ${WARMUP:-0} == 1 ]] && passes+=(W)
for ((r = 0; r < REPEAT; r++)); do passes+=("A$r" "B$r"); done

# PAIRWISE=1：每個非參考變體各自和參考（VARIANTS 的第一個）組成兩變體的 ABBA 區塊（ref v v ref），每個變體在區塊內
# 的位置平衡。三個以上變體的 ABBA（v1 v2 v3 | v3 v2 v1）會讓中間的變體永遠緊接在另一次執行之後；n01 的 HPL-MxP
# 800 W 實測到機箱風扇隨前一次執行的溫度而振盪，中間那個變體因此系統性地涼了約 12 °C（見實驗日誌 19:20）。
# MIN_REST：每次執行前至少休息幾秒，再等溫度降到 START_TEMP。PRECOND=<秒>：改為每次執行前跑同一個固定負載（熱預處理）。
MIN_REST=${MIN_REST:-0}
blocks=()
if [[ ${PAIRWISE:-0} == 1 ]]; then
  for v in "${var[@]:1}"; do blocks+=("${var[0]} $v"); done
else
  blocks=("$VARIANTS")
fi
for cap in $CAPS; do
  tools/pcap.sh "$cap"
  for bi in "${!blocks[@]}"; do
  read -ra bvar <<<"${blocks[bi]}"
  brev=()
  for ((i = ${#bvar[@]} - 1; i >= 0; i--)); do brev+=("${bvar[i]}"); done
  for pass in "${passes[@]}"; do
    list=("${bvar[@]}")
    [[ $pass == B* ]] && list=("${brev[@]}")
    [[ $pass == W ]] && list=("${bvar[0]}")
    for v in "${list[@]}"; do
      if (( ${PRECOND:-0} > 0 )); then
        # 熱預處理：每次執行前，所有變體都先跑同一個固定負載（8 張 GPU 原生 BF16 GEMM，PRECOND 秒），讓起始溫度與
        # 風扇狀態一致，取代「降溫到 START_TEMP」（那會讓起始狀態取決於前一次執行，見實驗日誌 00:55）
        pre=()
        for g in 0 1 2 3 4 5 6 7; do
          bin/gemm_mant_bench --device "$g" --dtype bf16 --k 8192 --warmup 0 --seconds "$PRECOND" --tag precond >/dev/null &
          pre+=($!)
        done
        wait "${pre[@]}"
        waited=0
      else
        sleep "$MIN_REST"
        waited=$MIN_REST
        while (( $(hot_max) > START_TEMP && waited < COOL_MAX + MIN_REST )); do sleep 5; waited=$((waited + 5)); done
      fi
      tools/pcap.sh show | awk -v w="${cap}W" '$2 != w {bad=1} END {exit bad}' || { echo "CAP RESET detected"; tools/pcap.sh "$cap"; }
      echo "=== RUN bench=$BENCH cap=$cap blk=$bi pass=$pass variant=$v t=$(date +%s.%N) cool=${waited}s hot=$(hot_max)"
      rc=0
      run_variant "$v" || rc=$?
      echo "=== END bench=$BENCH cap=$cap blk=$bi pass=$pass variant=$v t=$(date +%s.%N) rc=$rc"
    done
  done
  done
done
echo "$BENCH done"
