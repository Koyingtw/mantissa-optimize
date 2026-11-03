# Fix wrong results on ROCm when a moduli batch stride reaches 2^32 elements

## Problem

`oz2_core.hpp` computes the INT8 products of several moduli with one call to
`call_gemm_tn_strided_batched_raw` (strideC = m*n). On ROCm the batch strides reach
the Tensile/TensileLite kernels as 32-bit values. Once m*n >= 2^32, every batch after
the first is read from and written to the wrong place, and the call still returns
success. `gemmul8::gemm` then returns finite but wrong results: the relative error is
about 31 for m = n = 65536, k = 1024.

- The trigger is a single stride of 2^32 elements or more. The total size and the last
  offset do not matter: 50000^2 with 3 batches (last offset 5e9) is correct.
- At 65536^2 the stride truncates to 0, so batch 1 is never written. At 65537^2,
  batch 1 lands on C + 131073 and overwrites most of batch 0.
- ROCm 7.2.3 on MI350X (gfx950) is affected through every layer: hipBLAS
  `hipblasGemmStridedBatchedEx`, rocBLAS `rocblas_gemm_strided_batched_ex` with
  `ROCBLAS_USE_HIPBLASLT=0` and `=1`, and direct `hipblasLtMatmul`.
- Single-batch GEMMs of the same size are correct.
- ROCm/rocm-libraries#13216 makes hipBLASLt (develop) refuse such problems. rocBLAS
  still returns wrong results, and I am sending a separate fix there.
- CUDA builds are unaffected, and `batch_count()` already returns 1 for n > 2048 on
  sm_90.

## Change

In `src/oz2/common/matmult.hpp`, on HIP builds, `call_gemm_tn_strided_batched_raw`
issues the batches one GEMM at a time with host-side 64-bit pointer offsets when any
stride is 2^32 elements or more. Otherwise the batched path is unchanged.

## Test (MI350X, ROCm 7.2.3, OZ2_BACKENDS=INT8)

Direct API, 14 moduli, k = 1024, relative error against FP64 DGEMM:

| m = n | before | after |
|---|---|---|
| 65535 | 1.6e-15 | 1.6e-15 |
| 65536 | 31 | 1.6e-15 |
| 65537 | 31 | 1.6e-15 |
| 78976 | 31 | 1.6e-15 |

Hook (`GEMMUL8_BACKEND_GEMM=INT8`), 12 moduli, DGEMM m = n = 65536:

| k | before | after |
|---|---|---|
| 1024 | rel. err 48, 50.6 TFLOP/s | 4.0e-13, 50.4 TFLOP/s |
| 4096 | rel. err 97, 107.6 TFLOP/s | 8.0e-13, 106.6 TFLOP/s |

At 65535^2 the results are unchanged: 49.9 vs 49.0 and 106.2 vs 105.5 TFLOP/s, with the same errors.

A minimal reproducer of the library bug (int8 strided batched with batch stride
>= 2^32 through hipBLAS, rocBLAS and hipBLASLt) is in the rocBLAS PR (link to be added).
