// Is hipBLAS's INT8 GEMM (int8 x int8 -> int32, op(A)=A^T, as GEMMul8 calls it) correct when m*n >= 2^32?
// C = A^T B with A (k x m), B (k x n) int8 in [-64, 63]; a few entries of C, including the last column, are checked
// against a host dot product. --batch 2 runs a strided batch with strideC = m*n (how GEMMul8 batches its moduli; the
// second product uses A and B negated, so a wrong batch offset shows up).
//   int8_gemm_check --sizes "65535 65536" [--k 1024] [--batch 2] [--device 0]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s\n", hipGetErrorString(e_)); exit(1); } } while (0)

static int8_t val(uint64_t s, size_t i) { uint64_t z = (s ^ (i * 0x9E3779B97F4A7C15ull)) * 0xBF58476D1CE4E5B9ull; return int8_t((z >> 40) % 128) - 64; }

int main(int argc, char** argv) {
  size_t k = 1024; int dev = 0, batch = 1; std::string sizes = "65535 65536";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--k")) k = atol(argv[++i]);
    else if (!strcmp(argv[i], "--device")) dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--sizes")) sizes = argv[++i];
    else if (!strcmp(argv[i], "--batch")) batch = atoi(argv[++i]);
  }
  CK(hipSetDevice(dev));
  hipblasHandle_t h; hipblasCreate(&h);
  std::istringstream is(sizes); size_t m;
  while (is >> m) {
    const size_t n = m;
    const size_t sA = k * m, sB = k * n, sC = m * n;
    std::vector<int8_t> hA(sA * batch), hB(sB * batch);
    for (size_t i = 0; i < sA; ++i) for (int b = 0; b < batch; ++b) hA[b * sA + i] = int8_t(b % 2 ? -val(1, i) : val(1, i));
    for (size_t i = 0; i < sB; ++i) for (int b = 0; b < batch; ++b) hB[b * sB + i] = int8_t(b % 2 ? -val(2, i) : val(2, i) + (b ? 1 : 0));
    int8_t *A, *B; int32_t* C;
    CK(hipMalloc(&A, hA.size())); CK(hipMalloc(&B, hB.size())); CK(hipMalloc(&C, sC * batch * 4));
    CK(hipMemcpy(A, hA.data(), hA.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(B, hB.data(), hB.size(), hipMemcpyHostToDevice));
    CK(hipMemset(C, 0x7f, sC * batch * 4));
    const int32_t one = 1, zero = 0;
    hipblasStatus_t st = hipblasGemmStridedBatchedEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, (int)m, (int)n, (int)k, &one, A, HIP_R_8I, (int)k,
                                                     (int64_t)sA, B, HIP_R_8I, (int)k, (int64_t)sB, &zero, C, HIP_R_32I, (int)m,
                                                     (int64_t)sC, batch, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
    CK(hipDeviceSynchronize());
    size_t bad = 0, checked = 0;
    const size_t probes[][2] = {{0, 0}, {m - 1, 0}, {0, n - 1}, {m - 1, n - 1}, {m / 2, n / 2}, {123, n - 2}, {m - 7, n / 3}};
    for (int b = 0; b < batch; ++b)
    for (auto& p : probes) {
      const size_t i = p[0], j = p[1];
      int64_t ref = 0;
      for (size_t l = 0; l < k; ++l) ref += int64_t(hA[b * sA + i * k + l]) * hB[b * sB + j * k + l];
      int32_t got;
      CK(hipMemcpy(&got, C + b * sC + j * m + i, 4, hipMemcpyDeviceToHost));
      ++checked; if (got != ref) { ++bad; printf("  mismatch batch %d C[%zu,%zu] got=%d ref=%lld\n", b, i, j, got, (long long)ref); }
    }
    printf("RESULT_I8 m=n=%zu k=%zu batch=%d mn_over_2^32=%.4f status=%d checked=%zu bad=%zu\n", m, k, batch,
           double(m * n) / 4294967296.0, (int)st, checked, bad);
    fflush(stdout);
    CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C));
  }
  return 0;
}
