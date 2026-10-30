# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project theme

"mantissa-optimize" studies cutting the number of significand bits that flow through hot
kernels (mostly GEMM). The measured mechanism: fewer toggling mantissa bits → lower power →
less throttling under a power cap → higher SM clock → more throughput, while accuracy checks
still pass. Main results are expressed as ΔW (equivalent power saved at equal performance).

Results from mantissa truncation are **research / compliance-grey**, never an official
submission number. Always keep native and reduced-mantissa results clearly labelled
separately, and for HPL/HPL-MxP only count runs whose residual check printed PASSED (threshold 16).
The user wants progress output, logs and docs written in **Traditional Chinese**; most docs
(`*.md` logs, `paper/zh*`) already are. Machine-parsed log markers (`=== RUN`, `=== END`,
`RESULT`) stay in English.

## Repository layout

| Path | What it is |
|---|---|
| `gh200/` | Top-level GH200 study: scripts, shim, analysis, results, logs (most current work) |
| `apps/` | Third-party applications run under truncation (QE, gpu4pyscf, ExaGeoStat, LAMMPS, MFEM, SW4, CP2K) |
| `paper/` | English paper (`main_sc.tex` + `supplement.tex`) and Chinese report book (`zh_report/`) |
| `rocHPL/`, `rocHPL-MxP/` | Submodules (`NTHU-SC` forks): HPL CUDA port (`feat/cuda-port`), HPL-MxP (`perf/h200-mantissa`) |
| `SC26-MLPerf/` | Submodule: MLPerf Llama2-70B LoRA submission tree (used by the LoRA / FP8 attention studies) |

Submodules are `rocHPL`, `rocHPL-MxP`, `SC26-MLPerf`, `apps/q-e`, `apps/gpu4pyscf`,
`apps/exageostatcpp`. They may not be initialized in a given checkout (`git submodule status`
shows `-`); run `git submodule update --init <path>` before touching them. Code changes to the
NTHU-SC forks are committed **inside the submodule**, then the pointer is bumped here. Upstream
third-party apps are *not* forked: their modifications live as patches in `apps/patches/`
(`apps/patches/README.md` lists base commits; `git -C apps/<name> apply ../patches/<name>.patch`),
and builds are documented in `apps/建置日誌.md`. Untracked third-party trees (`apps/lammps`,
`apps/mfem`, `apps/sw4`, `apps/cp2k`, …) are gitignored — after changing them, regenerate the patch.

Some scripts (`gh200/slurm/lora_*.sh`, `gh200/fp8attn/*`, `gh200/bin/*`) hard-code the cluster
path `/nfs/home/koying/...` (the repo, the MLPerf work dir and build dir); adjust if running elsewhere.

## GH200 cluster

14-node GH200 cluster (`head`, `work1-13`, partition `compute`, one GH200 per node, aarch64,
IB 200 Gb/s). Benchmarks are launched with `sbatch`, never run locally on the head node.

`gh200/` structure (scripts are run/submitted **from `gh200/`**; they `cd` to `SLURM_SUBMIT_DIR`
and write `logs/<job-name>_<jobid>.out` plus nvidia-smi CSVs):
- `bench/` — `gemm_mant_bench.cu` (E1 cuBLAS GEMM with k mantissa bits, random data) and
  `fp64_kernel_bench.cu`; `make -C gh200/bench` → `gh200/bin/`.
- `shim/mantshim.cu` — `LD_PRELOAD` interposer for cuBLAS/CPU-BLAS DGEMM/ZGEMM. This is how
  unmodified FP64 apps (ExaGeoStat, PySCF, HPL via `hpl_shim.sbatch`) get truncated: env
  `MANT_BITS`, `MANT_ROUND=trunc|nearest`, `MANT_STATS` (exponent/fraction-bit histograms),
  `MANT_SPAN`, `MANT_EMU_BITS`/`MANT_EMU_DYN` (cuBLAS Ozaki emulation), `MANT_F32=fp32|tf32|tf32x3`.
  The full env list is the header comment of `mantshim.cu`; `test_shim.cu`/`test_f32.cu` test it.
- `slurm/` — one sbatch per experiment: E1 `e1_gemm_tdp`, E2/E3 `hpl_tdp` (`BENCH=hpl|mxp`,
  `TDPS`, `VARIANTS`, `REPEAT`, `WARMUP`, `COUNTERS=1`), `hpl_tune`, `range_hpl`, E4 `e4_error`,
  apps `app_tdp` (`APP=mfem|sw4|lammps`), `mp2_tdp`, `gp_tdp`, `exageostat_tdp`, `sci_sweep`,
  `a2_accuracy`, `infer_ppl`; LoRA drivers `lora_runs.sh` (run with `nohup`, not sbatch).
  Usage and env knobs are in each script's header comment. App build/run sbatches are in `apps/slurm/`.
- `analysis/` — Python (use `gh200/.venv`), one summarizer per experiment (`summarize.py` for
  E1–E3: joins run windows with nvidia-smi power logs and computes ΔW against the native curve;
  `app_summary.py`, `mp2_summary.py`, `gp_summary.py`, `paired_ci.py`, …). Outputs go to `results/*.csv`.
  `paper_figures.py` → `paper/figs/`, `report_figs_zh.py`/`report_static_figs.py` → `paper/zh_report/figs/`;
  figures read only `results/`, so regenerate them after re-analysis.
- `infer/`, `sci/`, `fp8attn/` — LLM inference (ppl, GEMM speed), CG solver, and the FP8 attention
  numerics study for MLPerf LoRA (`fp8attn/FP8注意力數值研究.md`; research only, not in the submission).
- `實驗日誌.md` (HPL/GEMM experiments) and `應用實驗日誌.md` (applications) are the running process
  logs, with job ids, decisions and the user's requests quoted. Append to them when running experiments.
- `refs/` — citation notes; `refs/pdf/` is gitignored (copyrighted).

Paper builds use tectonic on the GH200 cluster (`cd paper && ~/.local/bin/tectonic -X compile main.tex`) and
TeX Live 2023 XeLaTeX on MI350X node n01 (`paper/build.sh [main_sc|supplement|main]`, installed with apt). The current
paper is the two-platform SC-length `main_sc.tex` (≤ 10 pages + refs) with `supplement.tex` (section references to the
paper are hard-coded numerals, update them if sections move); `main.tex` is the old GH200-only version and `main_v2.tex`
an abandoned full-length draft (not updated). Chinese report: `paper/zh_report/build.sh` (fetches Noto CJK fonts, then checks the log for
overfull lines / undefined refs). Numbers must agree across `results/`, the paper and the report.

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

## MI350X nodes (`mi350/`)

Second platform for the paper: Slurm partition `gpu`, nodes `p01-r01-n01` and `p01-r01-n02`, each 8× MI350X
(gfx950, PPT0 cap 1000 W), ROCm 7.2.3, passwordless sudo. Home directories are **not shared** and ssh to n02
is refused: copy code with `tar c … | srun -w p01-r01-n02 tar x -C ~/koying` and fetch logs back the same way
(n02 logs live in `mi350/logs/n02/`). `mi350/實驗日誌.md` is the process log (append to it).

- `source mi350/env.sh` first: ROCm 7.2.3, OpenMPI 5.0.7+UCX copied from the admin's build into `~/koying/sw`
  (downloading/building MPI from the internet is blocked by permission rules). It **unsets
  `HIP_VISIBLE_DEVICES`** that `/etc/profile.d/env.sh` sets; without that HPL-MxP finds no GPU. Native HIP
  order is KFD order (≠ BDF order), so analysis always joins on the BDF the benchmark prints.
- Power: `mi350/tools/pcap.sh <W>` (sudo amd-smi PPT0, read back), `pcap.sh show`; scripts restore 1000 W.
  `mi350/tools/smi_sampler.py` logs amdsmi gpu_metrics every 100 ms incl. the hardware energy accumulator
  (15.259 µJ/count) and PPT/thermal residency counters; committed sampler logs are `*_smi.csv.xz`.
- Experiments: `mi350/scripts/*.sh` (header comments list env knobs), submitted with
  `sbatch -w <node> --export=ALL,SCRIPT=<script>,… mi350/slurm/run.sbatch`. `e1_gemm_tdp.sh` (HIP GEMM bench
  `mi350/bench/gemm_mant_bench.hip`, `make -C mi350/bench`), `e1b_fixedclk.sh` (SCLK lock, fill controls),
  `hpl_tdp.sh` (`BENCH=mxp` uses `rocHPL-MxP/build_mi350_bf16_mant<k>`; `BENCH=hpl` uses rocHPL
  `feat/adaptive-mixed-storage` in a worktree at `~/koying/sw/rocHPL-amd/build-mi350`; rochpl writes only
  `HPL.out`), `gp_tdp.sh` (`mi350/apps/gp_chol.hip`, HIP port of `apps/geostat/gp_mle.py`).
- Analysis (`mi350/.venv`): `analysis/e1_summary.py`, `hpl_summary.py`, `gp_summary.py` → `mi350/results/`.
- Findings so far: MI350X is power-bound at the stock 1000 W, so truncation gains there already; HPL-MxP
  0 kept bits does not converge (use 3–4); FP64 HPL 30/28 bits fail at every N tested (36 is safe).

## rocHPL-MxP (`rocHPL-MxP/`)

Earlier H200/MI350X work (before GH200) ran on a different Slurm cluster (account `acd110018`,
partitions `dev`, `8gpus`); the sbatch scripts inside the submodules target that cluster and write
to the submodule's own `logs/`. Submodule research docs (`docs/`) are mostly Traditional Chinese.

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
