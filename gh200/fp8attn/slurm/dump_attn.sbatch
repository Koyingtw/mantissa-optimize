#!/bin/bash
#SBATCH --job-name=fp8attn-dump
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --exclude=head
#SBATCH --output=/nfs/home/koying/mantissa-optimize/gh200/fp8attn/logs/fp8attn-dump_%j.log
#
# 1 節點、真實 checkpoint、幾步訓練，從 TE FP8 attention（fwd+bwd FP8，與提交設定相同）
# 擷取 q/k/v/o/dO/dq/dk/dv。仿 SC26-MLPerf/nvidia/launch.sh，但：
#   - IMPL 用獨立工作副本（gh200/fp8attn/prepare_work.sh 產生），不碰生產 nvidia_impl
#   - 不動風扇、不設功耗上限
# 用法： sbatch gh200/fp8attn/slurm/dump_attn.sbatch
#   可用環境變數：DUMP_LAYERS DUMP_STEPS MAX_STEPS_SMOKE SEED DUMP_DIR
set -euo pipefail

SIF=/nfs/home/koying/sc26-mlperf-build/containers/llama2_70b_lora_ngc25.04.sif
BUILD_DIR=/nfs/home/koying/sc26-mlperf-build
WORK=/nfs/home/koying/mlperf-fp8attn-work
IMPL=$WORK/nvidia_impl
DATA_ROOT=/data/mlperf/llama2_70b_lora
DATADIR=$DATA_ROOT/nvidia_npy
CKPT=$DATA_ROOT/nemo_torch_dist
DUMP_DIR=${DUMP_DIR:-/nfs/home/koying/fp8attn-dumps}
CONFIG=config_GH200_8x1x1xtp1pp1cp1_offload.sh
SEED=${SEED:-1234}
RUN_DIR=$WORK/runs/dump-$SLURM_JOB_ID
mkdir -p "$RUN_DIR" "$DUMP_DIR"
for i in $(seq 0 $((SLURM_NTASKS - 1))); do mkdir -p "$RUN_DIR/hydra/rank$i"; done

cat > "$RUN_DIR/overrides.sh" <<EOF
export MAX_STEPS=${MAX_STEPS_SMOKE:-9} VAL_CHECK_INTERVAL=3 LIMIT_VAL_BATCHES=0 WARMUP=False
export NVTE_FP8_DPA_BWD=1
export FP8ATTN_DUMP=1 FP8ATTN_DUMP_DIR=/dumps
export FP8ATTN_DUMP_LAYERS=${DUMP_LAYERS:-0,20,40,79} FP8ATTN_DUMP_STEPS=${DUMP_STEPS:-1,2,6}
EOF
echo "節點 $(hostname)  job $SLURM_JOB_ID  run dir $RUN_DIR  dump $DUMP_DIR  開始 $(date -u +%FT%TZ)"
cat "$RUN_DIR/overrides.sh"
nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used --format=csv

export MASTER_ADDR=$(hostname) MASTER_PORT=29511 SEED
export NCCL_IB_DISABLE=1
export DATA_ROOT=/data CKPT_ROOT=/ckpt
export APPTAINERENV_PREPEND_PATH= APPTAINERENV_PYTHONPATH=/pyextra
export CACHE_CLEARED=0 CONFIG_FILENAME=$CONFIG
export MLPERF_SUBMISSION_ORG=NTHU MLPERF_SUBMISSION_PLATFORM=GH200

srun --kill-on-bad-exit=1 apptainer exec --nv \
  --bind "$DATADIR:/data:ro,$CKPT:/ckpt:ro,$RUN_DIR:/results,$IMPL:/workspace/ft-llm,$BUILD_DIR/containers/pyextra:/pyextra:ro,$DUMP_DIR:/dumps" \
  --pwd /workspace/ft-llm "$SIF" bash -c '
    source ./'"$CONFIG"'
    export DGXNNODES=1
    source /results/overrides.sh
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    export NCCL_SOCKET_IFNAME=$(python ib_ifname.py 10.18.18.)
    export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
    exec python -u train.py hydra.run.dir=/results/hydra/rank$SLURM_PROCID
  '
echo "結束 $(date -u +%FT%TZ)"
ls -la "$DUMP_DIR"
