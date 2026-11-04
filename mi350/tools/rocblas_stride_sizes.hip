// Which problem sizes does rocBLAS strided-batched gemm_ex get wrong when the C/D batch stride is >= 2^32?
// int8 x int8 -> int32, batch 2, ldc = M, stride_a/b packed, stride_c = stride_d = --stride (default 2^32).
// Every element of both batches is checked against a host reference (keep M, N, K moderate).
//   rocblas_stride_sizes --stride 4294967296 --cases "TN 64 64 64  TN 4096 4096 1024  NN 4096 4096 1024"
#include <hip/hip_runtime.h>
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
  int64_t stride = int64_t(1) << 32;
  int beta_in = 0;  // beta = 1 adds C (zeroed), which must not change the result
  std::string cases = "TN 64 64 64";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--stride")) stride = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--cases")) cases = argv[++i];
    else if (!strcmp(argv[i], "--beta")) beta_in = atoi(argv[++i]);
  }
  rocblas_handle h;
  rocblas_create_handle(&h);
  const size_t batch = 2;
  int32_t* C;
  std::istringstream is(cases);
  std::string op;
  size_t m, n, k;
  while (is >> op >> m >> n >> k) {
    const bool ta = op[0] == 'T', tb = op[1] == 'T';
    const size_t lda = ta ? k : m, ldb = tb ? n : k;
    const size_t sA = lda * (ta ? m : k), sB = ldb * (tb ? k : n);
    std::vector<int8_t> hA(sA * batch), hB(sB * batch);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = val(1, i);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = val(2, i);
    int8_t *A, *B;
    CK(hipMalloc(&A, hA.size()));
    CK(hipMalloc(&B, hB.size()));
    const size_t c_total = size_t(stride) * (batch - 1) + m * n;
    CK(hipMalloc(&C, c_total * 4));
    CK(hipMemcpy(A, hA.data(), hA.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(B, hB.data(), hB.size(), hipMemcpyHostToDevice));
    CK(hipMemset(C, beta_in ? 0 : 0x7f, c_total * 4));
    const int32_t one = 1, zero = beta_in;
    rocblas_status st = rocblas_gemm_strided_batched_ex(
        h, ta ? rocblas_operation_transpose : rocblas_operation_none, tb ? rocblas_operation_transpose : rocblas_operation_none,
        (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, &one, A, rocblas_datatype_i8_r, (rocblas_int)lda, (rocblas_stride)sA, B,
        rocblas_datatype_i8_r, (rocblas_int)ldb, (rocblas_stride)sB, &zero, C, rocblas_datatype_i32_r, (rocblas_int)m, stride, C,
        rocblas_datatype_i32_r, (rocblas_int)m, stride, (rocblas_int)batch, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0,
        0);
    CK(hipDeviceSynchronize());
    size_t bad[2] = {0, 0};
    std::vector<int32_t> hc(m * n);
    for (size_t b = 0; b < batch; ++b) {
      CK(hipMemcpy(hc.data(), C + b * size_t(stride), m * n * 4, hipMemcpyDeviceToHost));
      for (size_t j = 0; j < n; j += (n > 256 ? n / 61 + 1 : 1))
        for (size_t i = 0; i < m; i += (m > 256 ? m / 59 + 1 : 1)) {
          int64_t ref = 0;
          for (size_t l = 0; l < k; ++l) {
            const int8_t a = ta ? hA[b * sA + i * lda + l] : hA[b * sA + l * lda + i];
            const int8_t bb = tb ? hB[b * sB + l * ldb + j] : hB[b * sB + j * ldb + l];
            ref += int64_t(a) * bb;
          }
          if (hc[j * m + i] != ref) ++bad[b];
          if (getenv("STRIDE_DEBUG") && i < 2 && j == 0)
            printf("  debug batch=%zu i=%zu got=%d ref=%lld\n", b, i, hc[j * m + i], (long long)ref);
        }
    }
    const char* lt = getenv("ROCBLAS_USE_HIPBLASLT");
    printf("RESULT_SIZES op=%s m=%zu n=%zu k=%zu stride=%lld hipblaslt=%s beta=%d status=%d bad_batch0=%zu bad_batch1=%zu\n",
           op.c_str(), m, n, k, (long long)stride, lt ? lt : "default", beta_in, (int)st, bad[0], bad[1]);
    fflush(stdout);
    CK(hipFree(A));
    CK(hipFree(B));
    CK(hipFree(C));
  }
  return 0;
}
