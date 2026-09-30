# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This top-level repo is only a wrapper around two git submodules (both `NTHU-SC` forks); all
real code, history, and docs live inside them. Commits to code must be made **inside the
submodule**, then the submodule pointer bumped here.

| Submodule | Checked-out branch | What it is |
|---|---|---|
| `rocHPL/` | `feat/cuda-port` | HPL (FP64 Linpack). This branch is a **CUDA/Hopper port** of rocHPL |
| `rocHPL-MxP/` | `perf/h200-mantissa` | HPL-MxP (mixed-precision Linpack), tuned for MI350X and H200 |

The project theme ("mantissa-optimize") is reducing the number of significand bits flowing
through the GEMM hot path. Measured finding: fewer toggling mantissa bits → lower GEMM power
→ less throttling → higher SM clock → higher score, while the FP64 residual check still passes.
Research docs in both `docs/` dirs are mostly Traditional Chinese.

Results from mantissa truncation are **research / compliance-grey**, never an official
submission number. Always keep native and reduced-mantissa results clearly labelled
separately, and only count runs whose residual check printed PASSED (threshold 16).

The H200 work ran on a Slurm cluster (account `acd110018`, partitions `dev`, `8gpus`); jobs
write to `logs/` inside each submodule. Benchmarks are launched with `sbatch`, not locally.

## GH200 cluster (current)

The current machine is a 14-node GH200 cluster (`head`, `work1-13`, partition `compute`, one
GH200 per node, aarch64, IB 200 Gb/s). Cross-cutting GH200 experiments live in top-level `gh200/`:
`bench/` (cuBLAS mantissa GEMM microbenchmark), `slurm/` (E1 `e1_gemm_tdp.sbatch`, E2/E3
`hpl_tdp.sbatch` with `BENCH=hpl|mxp`, E4), `analysis/` (`summarize.py` joins run windows with
nvidia-smi power logs and computes ΔW), `logs/`, `results/`, and `實驗日誌.md` (process log).
The user wants progress output and logs written in Traditional Chinese.

- Toolchain: `module load nvhpc/26.5` (CUDA 13.2, cuBLAS in `math_libs/13.2`, NCCL, HPC-X MPI).
  rocHPL: `cmake -S . -B build-gh200 -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CXX_COMPILER=g++`.
  MxP: `make -C src_cuda STORAGE=bf16 MANT_BITS=<k> B=../build_gh200_mant<k> CUDA_HOME=$NH/cuda/13.2
  MATH_HOME=$NH/math_libs/13.2 NVTX_LIB= MPICXX=mpicxx NCCL_HOME=$NH/comm_libs/nccl`
  (`NH=/software/nvhpc/Linux_aarch64/26.5`). Python analysis uses `gh200/.venv`.
- Power: `nv_power [--hosts h1,h2] <W>` sets the whole-module limit (200–1000 W, default 1000);
  scripts set it only on their own exclusive nodes and restore 1000 W on exit. Chassis fans:
  `fan_speed` (only change when the user asks).
- Pitfalls found: at low fan speed GEMM throttles on temperature (84–87 °C) before the power cap;
  at 100% fans native HPL-MxP at 1000 W trips the chassis hardware power brake on work4–7
  (check `clocks_event_reasons_counters.hw_power_brake_slowdown`, `COUNTERS=1` in `hpl_tdp.sbatch`).
  Start each measurement from a cool GPU (the scripts wait for ≤ 45 °C) and use ABBA ordering.

## rocHPL-MxP (`rocHPL-MxP/`)

Two independent codebases live here:

- `src/`, `include/` + CMake/`install.sh` → the original upstream `rochplmxp` (ROCm only).
  Rarely touched in current work.
- `src_mi350/` → the self-contained "hplmxp-nthu" backend where active work happens
  (`main.cu` args/driver, `matgen.cu`, `panel.cu` diag-tile factor & pack kernels, `lu.cu`
  LDU LU without pivoting + RCCL/NCCL panel bcast + trailing GEMM, `solve.cu` iterative
  refinement + FP64 residual). It is written against HIP APIs; `src_cuda/Makefile` compiles
  the **same `src_mi350/*.cu` sources** with nvcc using `-DHPLMXP_CUDA`, which pulls in
  `cuda_compat.hpp` (hip*/hipblas*/rccl names aliased to CUDA/cuBLAS(Lt)/NCCL). Any change to
  `src_mi350` therefore affects both MI350 and H200 builds — keep new HIP API uses mirrored
  in `cuda_compat.hpp`.

Build (Makefiles record flags in `$(B)/.build-config` so changing ROCm/arch/flags forces rebuild):
```bash
# MI350X (gfx950), default STORAGE=fp16, output build_mi350_fp16/xhplmxp-mi350
make -C src_mi350 ROCM=/opt/rocm-7.2.0 GPU_ARCH=gfx950 STORAGE=bf16 -j
# H200 (sm_90a), default STORAGE=bf16, output build_cuda_hopper/xhplmxp-cuda
module load gcc/11.5 cuda/12.6
make -C src_cuda -j STORAGE=bf16 MANT_BITS=0 B=../build_cuda_mant0 \
  MPICXX=/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpicxx \
  NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/comm_libs/13.0/nccl
```
- `MANT_BITS=<n>` (compile-time `HPLMXP_MANT_BITS`) zeroes low float mantissa bits in
  `storage_from_float()` (`src_mi350/hplmxp.h`) before every write-back to FP16/BF16 storage.
  `native` = 7 for BF16, 10 for FP16; anything lower prints `*** RESEARCH MODE ***`.
- Use a separate `B=` build dir per `MANT_BITS`/`STORAGE` combo (the sweep does `build_cuda_mant<n>`).

Run: `xhplmxp-* --n N --nb NB --p P --q Q [--fp8 0] [--approx-diag] [--itcheck 1] [--verbose]`,
one MPI rank per GPU. Reference configs and env defaults are in `scripts/run_h200.sh`
(8/12/16 H200 GPUs) and `scripts/slurm/*.sbatch`. Key sbatch entry points:
- `h200_cuda_smoke.sbatch` — small correctness check (run first after changes)
- `h200_mantissa_sweep_8gpu.sbatch` — builds+runs MANT_BITS 7..0 with nvidia-smi power/clock CSV
- `h200_mantissa_abba_8gpu.sbatch` — ABBA A/B for noise-controlled comparisons
- `fp16_ldu_smoke.sbatch` / `fp16_ldu_bench.sbatch` — MI350X equivalents

Runtime tuning is via `HPLMXP_*` env vars read with `getenv` in `src_mi350/*.cu` (e.g.
`HPLMXP_TRAILING_TN`, `HPLMXP_TRAILING_HIPBLAS`, `HPLMXP_PRECOND_FP32`, `HPLMXP_IR_RICHARDSON`,
`HPLMXP_HBM_RESERVE_GIB`, `HPLMXP_PROF=1|2` for phase breakdown, `HPLMXP_DEBUG`). Output reports
both a *reportable* rate (includes refinement) and an *LU-only* rate — compare reportable.
Lower mantissa can raise refinement cost, so LU-only gains don't translate monotonically.

Current state of results: `docs/h200-mantissa-sweep-zh-TW.md` (H200, 0 bits best: +10–14%),
`docs/cuda-h200-port-zh-TW.md`, `docs/mi350-*.md`.

## rocHPL (`rocHPL/`)

Has its own `rocHPL/CLAUDE.md`, but it describes the AMD/GEMMul8 branch and is **stale for
the checked-out `feat/cuda-port` branch**: on this branch `CMakeLists.txt` is CUDA-only
(`project(... LANGUAGES CXX CUDA)`), HIP/rocBLAS calls are shimmed through
`include/hip/*.h`, `include/rocblas/rocblas.h`, `include/hpl_cuda_ex.hpp` onto CUDA/cuBLAS,
and `HPL_USE_GEMMUL8`/`HPL_USE_HIPBLASLT` are rejected at configure time.

```bash
cmake -S . -B build-cuda-fast -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DHPL_DETAILED_TIMING=OFF -DHPL_PROGRESS_REPORT=OFF
cmake --build build-cuda-fast --parallel 4
sbatch scripts/slurm/cudahpl_smoke.sbatch        # 1 GPU, N=8192
sbatch scripts/slurm/cudahpl_bench.sbatch        # 8/12-GPU record configs
```
Mantissa experiment here is runtime, not compile-time: `MODE=quant` in the sbatch sets
`HPL_QUANT_A_BITS` / `HPL_QUANT_B_BITS` / `HPL_QUANT_C_BITS` (read in
`src/blas/HPL_quantize_device.cpp`) to round binary64 operand significands before a still-FP64
cuBLAS DGEMM; `53` = native. Native records and quant results are tracked separately in
`docs/19_cuda_h200_results.md`. `docs/18_bit_entropy_ceiling.md` explains the bit-entropy
→ power → clock mechanism, and `docs/11_real_data_dgemm_rate.md` why GEMM microbenchmarks
must use random (not zero/constant) data.
