#!/bin/bash
#SBATCH -J exageostat-build
#SBATCH -p compute
#SBATCH -N 1 -c 72 --exclusive
#SBATCH -t 2:00:00
#SBATCH -o /nfs/home/koying/mantissa-optimize/apps/logs/exageostat_build_%j.log
# ExaGeoStatCPP (apps/exageostatcpp, StarPU + Chameleon, CUDA) 及其相依套件，全部裝在 apps/exageostat-deps/install/<NAME>
# 用法: sbatch [--export=ALL,STAGES="starpu chameleon exageostat"] apps/slurm/exageostat_build.sbatch
#   STAGES   要跑的階段（預設全部；已完成的階段有 install/.done_<stage> 標記會跳過，FORCE=1 強制重做）
#   stage: hwloc openblas gsl nlopt json starpu chameleon exageostat
# 原始碼（登入節點先下載好）: apps/exageostat-deps/src/
#   hwloc-2.10.0.tar.gz OpenBLAS-0.3.34.tar.gz gsl-2.7.1.tar.gz nlopt-2.7.1.tar.gz json-3.11.2.tar.gz
#   starpu-1.3.11/（已套 CUDA 13 修改）chameleon-1.1.0/（git v1.1.0 + submodules，已套 CUDA 12+ 修改）
# 各階段完整輸出: apps/exageostat-deps/build/logs/
set -e -o pipefail
D=/nfs/home/koying/mantissa-optimize/apps/exageostat-deps
SRC=$D/src
INST=$D/install
LOGD=$D/build/logs
EXA=/nfs/home/koying/mantissa-optimize/apps/exageostatcpp
CUDA=/usr/local/cuda-13.3
NJ=72
STAGES=${STAGES:-"hwloc openblas gsl nlopt json starpu chameleon exageostat"}
TMPB=/tmp/exageostat-build-$SLURM_JOB_ID
mkdir -p $INST $TMPB $LOGD
trap 'rm -rf $TMPB' EXIT

module load cmake/3.31.12
export CC=gcc CXX=g++ FC=gfortran
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$INST/OPENBLAS/lib:$INST/HWLOC/lib:$INST/GSL/lib:$INST/NLOPT/lib:$INST/STARPU/lib:$INST/CHAMELEON/lib:$CUDA/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$INST/OPENBLAS/lib:$INST/GSL/lib:$INST/NLOPT/lib:$INST/HWLOC/lib:$INST/STARPU/lib:$INST/CHAMELEON/lib:$CUDA/lib64:$LIBRARY_PATH
export CPATH=$INST/OPENBLAS/include:$CPATH
export PKG_CONFIG_PATH=$INST/HWLOC/lib/pkgconfig:$INST/OPENBLAS/lib/pkgconfig:$INST/GSL/lib/pkgconfig:$INST/NLOPT/lib/pkgconfig:$INST/NLOPT/lib64/pkgconfig:$INST/NLOHMANN_JSON/share/pkgconfig:$INST/STARPU/lib/pkgconfig:$INST/CHAMELEON/lib/pkgconfig:$PKG_CONFIG_PATH
# gcc 15 預設 C23；舊 C 程式碼統一用 gnu17
CSTD="-O2 -fPIC -std=gnu17"
# gcc ≥ 14 把下列警告升級成錯誤；Chameleon 1.1.0 的 z/c herk codelet 會踩到（只影響複數版本）
GCC14_PERMISSIVE="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=int-conversion"

echo "== host $(hostname)  $(date)"
gcc --version | head -1; gfortran --version | head -1; nvcc --version | tail -2; cmake --version | head -1

want() { case " $STAGES " in *" $1 "*) ;; *) return 1;; esac; [ -n "$FORCE" ] || [ ! -f $INST/.done_$1 ]; }
done_() { touch $INST/.done_$1; echo "== stage $1 done $(date)"; }
# run <log> <cmd...>: 輸出寫到 $LOGD/<log>，失敗時印出錯誤附近
run() { local l=$LOGD/$1; shift; echo "-- $* (> $l)"; if ! "$@" > $l 2>&1; then
          echo "!! FAILED: $*"; grep -n -B3 -A10 -iE '(^|[^a-z_])error' $l | head -120 || tail -60 $l; exit 1; fi; }

if want hwloc; then
  echo "== stage hwloc $(date)"
  cd $TMPB && tar xzf $SRC/hwloc-2.10.0.tar.gz && cd hwloc-2.10.0
  run hwloc.cfg.log ./configure --prefix=$INST/HWLOC CFLAGS="$CSTD" --disable-cuda --disable-nvml --disable-opencl \
      --disable-rsmi --disable-levelzero --disable-gl --disable-libxml2 --disable-cairo
  run hwloc.make.log make -j $NJ
  run hwloc.install.log make install
  $INST/HWLOC/bin/lstopo-no-graphics --version
  done_ hwloc
fi

if want openblas; then
  echo "== stage openblas $(date)"
  cd $TMPB && tar xzf $SRC/OpenBLAS-0.3.34.tar.gz && cd OpenBLAS-0.3.34
  # 單執行緒、可重入（StarPU 的每個 CPU worker 各自呼叫 BLAS）
  run openblas.make.log make -j $NJ TARGET=NEOVERSEV2 USE_THREAD=0 USE_LOCKING=1 NO_AFFINITY=1 DYNAMIC_ARCH=0 CC=gcc FC=gfortran
  run openblas.install.log make PREFIX=$INST/OPENBLAS install
  ls $INST/OPENBLAS/lib $INST/OPENBLAS/include
  done_ openblas
fi

if want gsl; then
  echo "== stage gsl $(date)"
  cd $TMPB && tar xzf $SRC/gsl-2.7.1.tar.gz && cd gsl-2.7.1
  run gsl.cfg.log ./configure --prefix=$INST/GSL CFLAGS="$CSTD"
  run gsl.make.log make -j $NJ
  run gsl.install.log make install
  done_ gsl
fi

if want nlopt; then
  echo "== stage nlopt $(date)"
  cd $TMPB && tar xzf $SRC/nlopt-2.7.1.tar.gz
  run nlopt.cfg.log cmake -S nlopt-2.7.1 -B nlopt-build -DCMAKE_INSTALL_PREFIX=$INST/NLOPT -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON -DNLOPT_PYTHON=OFF -DNLOPT_SWIG=OFF -DNLOPT_OCTAVE=OFF \
        -DNLOPT_MATLAB=OFF -DNLOPT_GUILE=OFF -DNLOPT_TESTS=OFF
  run nlopt.make.log cmake --build nlopt-build -j $NJ
  run nlopt.install.log cmake --install nlopt-build
  done_ nlopt
fi

if want json; then
  echo "== stage json $(date)"
  cd $TMPB && tar xzf $SRC/json-3.11.2.tar.gz
  run json.cfg.log cmake -S json-3.11.2 -B json-build -DCMAKE_INSTALL_PREFIX=$INST/NLOHMANN_JSON -DJSON_BuildTests=OFF -DJSON_Install=ON
  run json.install.log cmake --install json-build
  done_ json
fi

if want starpu; then
  echo "== stage starpu $(date)"
  rm -rf $TMPB/starpu-build && mkdir -p $TMPB/starpu-build && cd $TMPB/starpu-build
  run starpu.cfg.log $SRC/starpu-1.3.11/configure --prefix=$INST/STARPU \
      --enable-cuda --with-cuda-dir=$CUDA --disable-opencl --disable-mpi --disable-fortran \
      --disable-build-doc --disable-build-examples --disable-build-tests --disable-export-dynamic \
      --enable-shared --disable-static --disable-socl --disable-starpufft --disable-mlr --disable-hdf5 \
      --disable-simgrid --disable-openmp --enable-maxcpus=128 --enable-maxcudadev=4 \
      --with-hwloc=$INST/HWLOC \
      CFLAGS="$CSTD" CXXFLAGS="-O2 -fPIC" NVCCFLAGS="-arch=sm_90"
  tail -45 $LOGD/starpu.cfg.log
  run starpu.make.log make -j $NJ
  run starpu.install.log make install
  grep -E 'define STARPU_USE_CUDA|define STARPU_MAXCPUS|define STARPU_HAVE_HWLOC|define STARPU_HAVE_CUDA_MEMCPY_PEER|define STARPU_HAVE_LIBNVIDIA_ML' $INST/STARPU/include/starpu/1.3/starpu_config.h || true
  done_ starpu
fi

if want chameleon; then
  echo "== stage chameleon $(date)"
  # ExaGeoStatCPP 會 include Chameleon 的內部標頭（control/*.h），路徑寫死為 <prefix>/chameleon-src
  mkdir -p $INST/CHAMELEON && ln -sfn $SRC/chameleon-1.1.0 $INST/CHAMELEON/chameleon-src
  rm -rf $TMPB/chameleon-build
  run chameleon.cfg.log cmake -S $SRC/chameleon-1.1.0 -B $TMPB/chameleon-build -DCMAKE_INSTALL_PREFIX=$INST/CHAMELEON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -std=gnu17 $GCC14_PERMISSIVE" -DBUILD_SHARED_LIBS=ON \
        -DCHAMELEON_SCHED=STARPU -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=OFF \
        -DCHAMELEON_ENABLE_TESTING=OFF -DCHAMELEON_ENABLE_EXAMPLE=OFF -DCHAMELEON_ENABLE_DOC=OFF \
        -DBLA_VENDOR=OpenBLAS -DBLAS_DIR=$INST/OPENBLAS -DCBLAS_DIR=$INST/OPENBLAS -DLAPACKE_DIR=$INST/OPENBLAS \
        -DLAPACK_DIR=$INST/OPENBLAS -DTMG_DIR=$INST/OPENBLAS -DSTARPU_DIR=$INST/STARPU -DHWLOC_DIR=$INST/HWLOC \
        -DCUDA_TOOLKIT_ROOT_DIR=$CUDA
  grep -iE 'cuda|starpu|blas|lapack|warn' $LOGD/chameleon.cfg.log | head -40 || true
  run chameleon.make.log cmake --build $TMPB/chameleon-build -j $NJ
  run chameleon.install.log cmake --install $TMPB/chameleon-build
  grep -E 'CHAMELEON_USE_CUDA|CHAMELEON_USE_CUBLAS_V2|CHAMELEON_SCHED' $INST/CHAMELEON/include/chameleon/config.h || true
  ldd $INST/CHAMELEON/lib/libcudablas.so | grep -E 'cublas|cudart' || true
  done_ chameleon
fi

if want exageostat; then
  echo "== stage exageostat $(date)"
  BLD=$D/build/exageostatcpp
  [ -n "$FORCE" ] && rm -rf $BLD
  export CMAKE_PREFIX_PATH=$INST/NLOHMANN_JSON:$INST/OPENBLAS:$INST/NLOPT:$INST/GSL:$INST/HWLOC:$INST/STARPU:$INST/CHAMELEON
  export GSL_ROOT_DIR=$INST/GSL
  run exageostat.cfg.log cmake -Wno-dev -S $EXA -B $BLD -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INST \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -Wl,--no-as-needed -w -fpic" \
        -DUSE_CUDA=ON -DEXAGEOSTAT_ALLOW_NEW_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DUSE_MPI=OFF -DUSE_HICMA=OFF -DUSE_R=OFF -DUSE_CLIMATE_EMULATOR=OFF -DRUNTIME_TYPE=starpu \
        -DBUILD_EXAMPLES=ON -DBUILD_TESTS=OFF -DBUILD_HEAVY_TESTS=OFF -DBUILD_DOCS=OFF -DCREATE_PACKAGE=OFF \
        -DBLA_VENDOR=OpenBLAS -DBLA_STATIC=OFF -DBUILD_SHARED_LIBS=OFF -Dgpu_backend=none
  grep -E 'Found|found|Install|Can.t find|CHAMELEON_DIR|done|Error|WARN' $LOGD/exageostat.cfg.log | head -60 || true
  run exageostat.make.log cmake --build $BLD -j $NJ
  ls -l $BLD/examples/end-to-end/ || true
  done_ exageostat
fi
echo "== all done $(date)"
