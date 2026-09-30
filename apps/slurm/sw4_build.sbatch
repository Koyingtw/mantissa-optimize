#!/bin/bash
#SBATCH -J sw4-build
#SBATCH -p compute
#SBATCH -w work13
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 1:30:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/sw4_build_%j.log
# 用法: sbatch [--export=ALL,STEP=raja|sw4|all] apps/slurm/sw4_build.sbatch
set -ex
module load nvhpc/26.5
STEP=${STEP:-all}
APPS=/nfs/home/koying/mantissa-optimize/apps
TPL=$APPS/sw4-tpl
CUDA=/software/nvhpc/Linux_aarch64/26.5/cuda/13.2
export PATH=$CUDA/bin:$PATH

if [[ $STEP == raja || $STEP == all ]]; then
  rm -rf $TPL/build-raja
  rm -rf $TPL/install; cmake -S $TPL/RAJA-2026.07.1 -B $TPL/build-raja -DCMAKE_INSTALL_PREFIX=$TPL/install -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCUDAToolkit_ROOT=$CUDA -DCUB_DIR=$CUDA/include/cccl -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DRAJA_ENABLE_CUDA=On -DENABLE_CUDA=On -DRAJA_ENABLE_OPENMP=Off -DENABLE_OPENMP=Off \
    -DRAJA_ENABLE_TESTS=Off -DENABLE_TESTS=Off -DRAJA_ENABLE_EXAMPLES=Off -DRAJA_ENABLE_EXERCISES=Off \
    -DRAJA_ENABLE_BENCHMARKS=Off -DENABLE_BENCHMARKS=Off -DRAJA_ENABLE_REPRODUCERS=Off \
    -DBLT_CXX_STD=c++20 -DCAMP_USE_PLATFORM_DEFAULT_STREAM=On -DCMAKE_CUDA_FLAGS="-Xcompiler=-Wno-template-body" 2>&1 | tail -30
  cmake --build $TPL/build-raja -j 72 2>&1 | tail -30
  cmake --install $TPL/build-raja | tail -5
fi

if [[ $STEP == sw4 || $STEP == all ]]; then
  cd $APPS/sw4
  cp configs/make.gh200 configs/make.inc
  time make -j 72 sw4 2>&1 | tail -60 || true
  # SW4 文件提到平行 make 偶有 race condition，失敗就再跑一次
  [[ -x optimize/sw4 ]] || time make -j 16 sw4 2>&1 | tail -60
  # 原生對照版：以 git HEAD 的 setupRun.C（無截斷程式碼）連結 optimize/sw4_native
  make -f ../sw4-tpl/native/native.mk sw4_native 2>&1 | tail -5
  ls -l optimize/sw4 optimize/sw4_native
  ldd optimize/sw4 | grep -E "cuda|mpi|lapack|blas|nvidia"
fi
