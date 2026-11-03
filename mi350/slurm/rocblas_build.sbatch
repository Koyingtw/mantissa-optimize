#!/bin/bash
# 從 rocm-libraries（worktree 內的 clone，branch fix/rocblas-gemm-ex-batch-stride-2e32-7.2.3）編譯 rocBLAS 7.2.3，
# 只為 gfx950，用 monorepo 內的 Tensile；不佔滿節點（48 CPU、不要 GPU、不 exclusive），讓其他 GPU 工作照常排程。
#   sbatch -w p01-r01-n01 mi350/slurm/rocblas_build.sbatch
#SBATCH -J rocblas-build
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=192G
#SBATCH -o /home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow/mi350/logs/slurm_%x_%j.out
set -euo pipefail
W=/home/cust_nthu/koying/mantissa-optimize/.claude/worktrees/g8-overflow
SRC=$W/rocm-libraries
B=$W/rocblas-build
ROCM=/opt/rocm-7.2.3
echo "job=$SLURM_JOB_ID host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
export PATH=$ROCM/bin:$ROCM/llvm/bin:$PATH
cmake -S "$SRC/projects/rocblas" -B "$B" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$B/install" \
  -DCMAKE_CXX_COMPILER=$ROCM/llvm/bin/amdclang++ -DCMAKE_C_COMPILER=$ROCM/llvm/bin/amdclang \
  -DROCM_PATH=$ROCM -DCMAKE_PREFIX_PATH=$ROCM \
  -DGPU_TARGETS=gfx950 -DAMDGPU_TARGETS=gfx950 \
  -DBUILD_WITH_TENSILE=ON -DBUILD_WITH_HIPBLASLT=ON \
  -DTensile_TEST_LOCAL_PATH="$SRC/shared/tensile" -DTensile_CPU_THREADS="$SLURM_CPUS_PER_TASK" \
  -DTensile_LIBRARY_FORMAT=msgpack -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF \
  -DBUILD_CLIENTS_SAMPLES=OFF
cmake --build "$B" -j "$SLURM_CPUS_PER_TASK"
cmake --install "$B"
ls -la "$B/install/lib" "$B/install/lib/rocblas/library" | head -20
echo "rc=$? end=$(date)"
