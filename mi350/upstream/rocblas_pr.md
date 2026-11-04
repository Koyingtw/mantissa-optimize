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
- Regression gtest `gemm_strided_batched_ex_stride_2e32` (category `stress`, Linux) in
  `gemm_strided_batched_gtest.yaml`: 64x64x64, batch 2, `rand_int` data (different results per batch),
  stride_c = stride_d = 2^32 + 1 (batch 1 overwrites batch 0 without the fix), int8 -> int32 and half with float compute,
  `use_hipblaslt: [0, 1]`. Only the device pointer mode runs, because `testing_gemm_strided_batched_ex`
  computes the reference and both modes' checks inside its `pointer_mode_device` branch (a host-only
  entry is never checked). Needs about 128 GB of host and 32 GB of device memory.
  Run: `rocblas-test --gtest_filter='*stride_2e32*'`.

## Test Result

MI350X, ROCm 7.2.3, k = 1024, batch 2, number of the 7 probes per batch that are wrong (batch0 / batch1):

| m = n | stride / 2^32 | stock, HIPBLASLT=0 | stock, HIPBLASLT=1 | patched, both |
|---|---|---|---|---|
| 65535 | 0.99997 | 0 / 0 | 0 / 0 | 0 / 0 |
| 65536 | 1.0 | 7 / 7 | 7 / 7 | 0 / 0 |
| 65537 | 1.00003 | 5 / 7 | 3 / 7 | 0 / 0 |

These are the same for the rocBLAS and hipBLAS entry points. All calls return status 0.

`rocblas-test --gtest_filter='*stride_2e32*'` (4 tests: int8 and half, `use_hipblaslt` 0 and 1, stride 2^32 + 1):
stock 7.2.3 `librocblas.so` 0 of 4 pass, patched 4 of 4 pass. An earlier version with 2^32 - 1 (control) and 2^32
as well gave: stock passes only the 2^32 - 1 cases (4 of 12), patched 12 of 12.

Every failure is a `unit_check` mismatch (status is success), for both `use_hipblaslt` values and both types.

## Submission Checklist
- [x] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.

## Risk level

Low. The new branch runs only when a batch stride is at least 2^32 elements, which currently gives wrong
results. Other calls take the existing path.

<details><summary>Reproducer <code>int8_stride_layers.hip</code> (build: <code>hipcc -O2 int8_stride_layers.hip -lhipblas -lrocblas -lhipblaslt</code>)</summary>

```cpp
// Which ROCm layer truncates the batch stride of a strided-batched INT8 GEMM to 32 bits?
// The same C_b = A_b^T B_b (int8 x int8 -> int32, A_b k x m, B_b k x n, batch b = 0, 1 with B_1 = -B_0) is issued through
//   --api hipblas   hipblasGemmStridedBatchedEx
//   --api rocblas   rocblas_gemm_strided_batched_ex   (ROCBLAS_USE_HIPBLASLT=0/1 picks Tensile or hipBLASLt inside)
//   --api lt        hipblasLtMatmul with HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT / STRIDED_BATCH_OFFSET
// and a few entries of both batches are compared with a host dot product. C is prefilled with 0x7f7f7f7f.
//   int8_stride_layers --api rocblas --sizes "65535 65536" [--k 1024]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s\n", hipGetErrorString(e_)); exit(1); } } while (0)

static int8_t val(uint64_t s, size_t i) {
  uint64_t z = (s ^ (i * 0x9E3779B97F4A7C15ull)) * 0xBF58476D1CE4E5B9ull;
  return int8_t((z >> 40) % 128) - 64;
}

int main(int argc, char** argv) {
  size_t k = 1024;
  int dev = 0;
  std::string sizes = "65535 65536", api = "hipblas";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--k")) k = atol(argv[++i]);
    else if (!strcmp(argv[i], "--device")) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--sizes")) sizes = argv[++i];
    else if (!strcmp(argv[i], "--api")) api = argv[++i];
  }
  CK(hipSetDevice(dev));
  hipblasHandle_t hb;
  rocblas_handle hr;
  hipblasLtHandle_t hl;
  hipblasCreate(&hb);
  rocblas_create_handle(&hr);
  hipblasLtCreate(&hl);
  std::istringstream is(sizes);
  size_t m;
  while (is >> m) {
    const size_t n = m, batch = 2, sA = k * m, sB = k * n, sC = m * n;
    std::vector<int8_t> hA(sA * batch), hB(sB * batch);
    for (size_t i = 0; i < sA; ++i) hA[i] = hA[sA + i] = val(1, i);
    for (size_t i = 0; i < sB; ++i) { hB[i] = val(2, i); hB[sB + i] = int8_t(-hB[i]); }
    int8_t *A, *B;
    int32_t* C;
    CK(hipMalloc(&A, hA.size()));
    CK(hipMalloc(&B, hB.size()));
    CK(hipMalloc(&C, sC * batch * 4));
    CK(hipMemcpy(A, hA.data(), hA.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(B, hB.data(), hB.size(), hipMemcpyHostToDevice));
    CK(hipMemset(C, 0x7f, sC * batch * 4));
    const int32_t one = 1, zero = 0;
    int status = -1;
    void* ws = nullptr;
    if (api == "hipblas") {
      status = hipblasGemmStridedBatchedEx(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, (int)m, (int)n, (int)k, &one, A, HIP_R_8I, (int)k,
                                           (int64_t)sA, B, HIP_R_8I, (int)k, (int64_t)sB, &zero, C, HIP_R_32I, (int)m,
                                           (int64_t)sC, (int)batch, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
    } else if (api == "rocblas") {
      status = rocblas_gemm_strided_batched_ex(hr, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int)m,
                                               (rocblas_int)n, (rocblas_int)k, &one, A, rocblas_datatype_i8_r, (rocblas_int)k,
                                               (rocblas_stride)sA, B, rocblas_datatype_i8_r, (rocblas_int)k, (rocblas_stride)sB,
                                               &zero, C, rocblas_datatype_i32_r, (rocblas_int)m, (rocblas_stride)sC, C,
                                               rocblas_datatype_i32_r, (rocblas_int)m, (rocblas_stride)sC, (rocblas_int)batch,
                                               rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0);
    } else if (api == "lt") {
      hipblasLtMatmulDesc_t desc;
      hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
      hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
      hipblasLtMatrixLayout_t la, lb, lc;
      hipblasLtMatrixLayoutCreate(&la, HIP_R_8I, k, m, k);
      hipblasLtMatrixLayoutCreate(&lb, HIP_R_8I, k, n, k);
      hipblasLtMatrixLayoutCreate(&lc, HIP_R_32I, m, n, m);
      const int32_t bc = (int32_t)batch;
      const int64_t stA = (int64_t)sA, stB = (int64_t)sB, stC = (int64_t)sC;
      for (auto& [l, st] : {std::pair{la, stA}, std::pair{lb, stB}, std::pair{lc, stC}}) {
        hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
        hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof(st));
      }
      const size_t ws_size = size_t(64) << 20;
      CK(hipMalloc(&ws, ws_size));
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));
      hipblasLtMatmulHeuristicResult_t heur[1];
      int found = 0;
      hipblasLtMatmulAlgoGetHeuristic(hl, desc, la, lb, lc, lc, pref, 1, heur, &found);
      if (found == 0) {
        fprintf(stderr, "hipBLASLt: no algorithm\n");
      } else {
        status = hipblasLtMatmul(hl, desc, &one, A, la, B, lb, &zero, C, lc, C, lc, &heur[0].algo, ws, ws_size, 0);
      }
    }
    CK(hipDeviceSynchronize());
    size_t bad[2] = {0, 0}, checked = 0;
    const size_t probes[][2] = {{0, 0}, {m - 1, 0}, {0, n - 1}, {m - 1, n - 1}, {m / 2, n / 2}, {123, n - 2}, {m - 7, n / 3}};
    for (size_t b = 0; b < batch; ++b)
      for (auto& p : probes) {
        const size_t i = p[0], j = p[1];
        int64_t ref = 0;
        for (size_t l = 0; l < k; ++l) ref += int64_t(hA[b * sA + i * k + l]) * hB[b * sB + j * k + l];
        int32_t got;
        CK(hipMemcpy(&got, C + b * sC + j * m + i, 4, hipMemcpyDeviceToHost));
        ++checked;
        if (got != ref) ++bad[b];
      }
    printf("RESULT_LAYER api=%s m=n=%zu k=%zu stride=%zu stride_over_2^32=%.4f status=%d checked=%zu bad_batch0=%zu bad_batch1=%zu\n",
           api.c_str(), m, k, sC, double(sC) / 4294967296.0, status, checked, bad[0], bad[1]);
    fflush(stdout);
    if (ws) CK(hipFree(ws));
    CK(hipFree(A));
    CK(hipFree(B));
    CK(hipFree(C));
  }
  return 0;
}
```

</details>
