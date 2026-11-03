Title: fix(rocblas): split strided-batched gemm_ex when a batch stride reaches 2^32 elements

JIRA ID : N/A (related: ROCM-32140 / #13216 for hipBLASLt)

## Motivation

`rocblas_gemm_strided_batched_ex` (and hipBLAS `hipblasGemmStridedBatchedEx` on top of it)
returns `rocblas_status_success` but produces wrong results when `batch_count > 1` and any
batch stride is 2^32 elements or more. Example: int8 inputs with int32 output, m = n = 65536,
k = 1024, `stride_c = m * n = 2^32`, batch 2. The stride truncates to 0, so batch 1 is never
written. At 65537^2 the truncated stride is 131073, so batch 1 overwrites most of batch 0.
65535^2 (stride 2^32 - 2^17 + 1) is correct, as is 50000^2 with 3 batches, whose last offset is 5e9.
What triggers the bug is the size of a single stride, not the size of the allocation.

Seen on MI350X (gfx950) with ROCm 7.2.3, with `ROCBLAS_USE_HIPBLASLT=0` and `=1`. Both Tensile
(`shared/tensile/.../ContractionSolution.cpp`) and TensileLite append the strides as `uint32_t`
kernel arguments. Tensile's `BufferLoad/StoreOffsetLimitCheck` predicates look at the leading
dimension only. #13216 makes hipBLASLt refuse such problems (`problemFitsKernelArgs`), but rocBLAS has
no equivalent guard, and the Tensile path still runs them.

We hit this in practice with GEMMul8 (Ozaki-II INT8 emulation of DGEMM), which batches the per-modulus
products with stride m*n. Any DGEMM with m*n >= 2^32 came back finite but wrong (relative error ~31).

## Technical Details

In `rocblas_gemm_ex_template`, for the strided (non-pointer-array) form with `batch_count > 1` and
any of `stride_a/b/c/d >= 2^32`, call the template once per batch with `batch_count = 1` and
the batch selected through the 64-bit element offsets (`offsetAin + i * stride_a`, ...). With one batch
the strides are unused. The offsets are 64-bit throughout: `uint64_t` in Tensile, added to the pointer on
the host (7.2.3) or as an `int64_t` element offset converted to `size_t` bytes (develop) for hipBLASLt.
Problems below the limit are unaffected (one compare per call).

Alternatives considered: making the kernel stride arguments 64-bit (Tensile/TensileLite codegen
change across all kernels, much larger) or returning an error (would turn currently-"working" calls into
failures; splitting keeps them correct). Splitting costs one extra launch per batch, which
is negligible at these sizes (each batch is >= 2^32 output elements).

## Test Plan

- Reproducer (`int8_stride_layers.hip`, below): int8 -> int32 C_b = A_b^T B_b, batch 2 with B_1 = -B_0,
  7 probe elements per batch checked against a host dot product, through hipBLAS, rocBLAS and hipBLASLt.
- Built rocBLAS from the `rocm-7.2.3` tag plus this patch (gfx950, Tensile + hipBLASLt), ran the reproducer
  against stock and patched `librocblas.so`. The patch cherry-picks onto develop without conflicts.
  I did not build develop, because it needs a newer hipBLASLt than ROCm 7.2.3 ships.

## Test Result

MI350X, ROCm 7.2.3, k = 1024, batch 2, number of the 7 probes per batch that are wrong (batch0 / batch1):

| m = n | stride / 2^32 | stock, HIPBLASLT=0 | stock, HIPBLASLT=1 | patched, both |
|---|---|---|---|---|
| 65535 | 0.99997 | 0 / 0 | 0 / 0 | 0 / 0 |
| 65536 | 1.0 | 7 / 7 | 7 / 7 | 0 / 0 |
| 65537 | 1.00003 | 5 / 7 | 3 / 7 | 0 / 0 |

These are the same for the rocBLAS and hipBLAS entry points. All calls return status 0.

## Submission Checklist
- [x] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.

## Risk level

Low. The new branch runs only when a batch stride is at least 2^32 elements, which currently gives wrong
results. Other calls take the existing path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_016pHscGyK39vkQ1dhLvWhdu
