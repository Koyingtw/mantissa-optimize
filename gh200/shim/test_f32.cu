// Checks mantshim's MANT_F32 path against an exact host product (long double accumulation):
// DGEMM (T, N) and ZGEMM (C, N), both with non-trivial alpha/beta and padded leading dimensions.
// Prints the relative Frobenius error of C, so fp32 / tf32 / tf32x3 can be compared with the
// truncated-FP64 path (run without MANT_F32, with MANT_BITS set).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cd = std::complex<double>;
using cld = std::complex<long double>;

int main() {
  const int m = 300, n = 200, k = 2048, lda = k + 8, ldb = k + 16, ldc = m + 4;  // A stored k x m (op T/C)
  srand(1);
  auto rnd = [] { return 2.0 * rand() / RAND_MAX - 1.0; };
  cublasHandle_t h;
  cublasCreate(&h);

  {  // real
    std::vector<double> A((size_t)lda * m), B((size_t)ldb * n), C((size_t)ldc * n), C0;
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C) x = rnd();
    C0 = C;
    double *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 8); cudaMalloc(&dB, B.size() * 8); cudaMalloc(&dC, C.size() * 8);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size() * 8, cudaMemcpyHostToDevice);
    const double alpha = 0.75, beta = -0.5;
    cublasStatus_t s = cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0, pad = 0;
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        long double acc = 0;
        for (int p = 0; p < k; ++p) acc += (long double)A[p + (size_t)i * lda] * B[p + (size_t)j * ldb];
        const long double ref = alpha * acc + beta * (long double)C0[i + (size_t)j * ldc];
        const double d = C[i + (size_t)j * ldc] - (double)ref;
        num += d * d;
        den += (double)(ref * ref);
      }
      for (int i = m; i < ldc; ++i) pad = fmax(pad, fabs(C[i + (size_t)j * ldc] - C0[i + (size_t)j * ldc]));
    }
    printf("dgemm(T,N) status=%d rel_err=%.3e padding_untouched=%d\n", (int)s, sqrt(num / den), pad == 0);
  }

  {  // complex, conjugate transpose
    std::vector<cd> A((size_t)lda * m), B((size_t)ldb * n), C((size_t)ldc * n), C0;
    for (auto& x : A) x = cd(rnd(), rnd());
    for (auto& x : B) x = cd(rnd(), rnd());
    for (auto& x : C) x = cd(rnd(), rnd());
    C0 = C;
    cuDoubleComplex *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 16); cudaMalloc(&dB, B.size() * 16); cudaMalloc(&dC, C.size() * 16);
    cudaMemcpy(dA, A.data(), A.size() * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size() * 16, cudaMemcpyHostToDevice);
    const cuDoubleComplex alpha = {0.5, 0.25}, beta = {1.0, -0.5};
    cublasStatus_t s = cublasZgemm(h, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cudaMemcpy(C.data(), dC, C.size() * 16, cudaMemcpyDeviceToHost);
    const cld al(alpha.x, alpha.y), be(beta.x, beta.y);
    double num = 0, den = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cld acc = 0;
        for (int p = 0; p < k; ++p) acc += std::conj((cld)A[p + (size_t)i * lda]) * (cld)B[p + (size_t)j * ldb];
        const cld ref = al * acc + be * (cld)C0[i + (size_t)j * ldc];
        const cld d = (cld)C[i + (size_t)j * ldc] - ref;
        num += (double)std::norm(d);
        den += (double)std::norm(ref);
      }
    printf("zgemm(C,N) status=%d rel_err=%.3e\n", (int)s, sqrt(num / den));
  }
  return 0;
}
