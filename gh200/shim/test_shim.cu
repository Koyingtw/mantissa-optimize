// Checks mantshim: reduced result equals a GEMM on explicitly reduced inputs,
// and the caller's A/B are left untouched.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
static double red(double v, int bits) { uint64_t u; memcpy(&u, &v, 8); u &= ~((1ull << (52 - bits)) - 1); memcpy(&v, &u, 8); return v; }
int main(int argc, char** argv) {
  const int bits = atoi(argv[1]);  // what the shim was told, for the reference
  const int m = 300, n = 200, k = 500, lda = 320, ldb = 520;
  std::vector<double> A(lda * k), B(ldb * n), Ar(A.size()), Br(B.size());
  srand(1);
  for (auto& x : A) x = rand() / (double)RAND_MAX - 0.5;
  for (auto& x : B) x = rand() / (double)RAND_MAX - 0.5;
  for (size_t i = 0; i < A.size(); ++i) Ar[i] = red(A[i], bits);
  for (size_t i = 0; i < B.size(); ++i) Br[i] = red(B[i], bits);
  double *dA, *dB, *dC, *dAr, *dBr, *dCr;
  cudaMalloc(&dA, A.size() * 8); cudaMalloc(&dB, B.size() * 8); cudaMalloc(&dC, m * n * 8);
  cudaMalloc(&dAr, A.size() * 8); cudaMalloc(&dBr, B.size() * 8); cudaMalloc(&dCr, m * n * 8);
  cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice); cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dAr, Ar.data(), A.size() * 8, cudaMemcpyHostToDevice); cudaMemcpy(dBr, Br.data(), B.size() * 8, cudaMemcpyHostToDevice);
  cublasHandle_t h; cublasCreate(&h);
  const double one = 1, zero = 0;
  cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dA, lda, dB, ldb, &zero, dC, m);    // intercepted
  cudaDeviceSynchronize();
  // reference: shim is active for this call too, but inputs are already reduced so it is a no-op
  cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dAr, lda, dBr, ldb, &zero, dCr, m);
  std::vector<double> C(m * n), Cr(m * n), A2(A.size());
  cudaMemcpy(C.data(), dC, m * n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(Cr.data(), dCr, m * n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(A2.data(), dA, A.size() * 8, cudaMemcpyDeviceToHost);
  double md = 0; for (int i = 0; i < m * n; ++i) md = fmax(md, fabs(C[i] - Cr[i]));
  // exact reference on the host (original, unreduced inputs), long double accumulation
  double num = 0, den = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      long double acc = 0;
      for (int p = 0; p < k; ++p) acc += (long double)A[i + (size_t)p * lda] * B[p + (size_t)j * ldb];
      const double d = C[i + (size_t)j * m] - (double)acc;
      num += d * d;
      den += (double)acc * (double)acc;
    }
  printf("bits=%d max|C_shim - C_ref|=%.3e  A_untouched=%d  rel_err_vs_exact=%.3e\n", bits, md,
         memcmp(A.data(), A2.data(), A.size() * 8) == 0, sqrt(num / den));
  // transposed + Zgemm smoke
  cuDoubleComplex a = {1, 0}, b = {0, 0};
  cublasZgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, 50, 40, 60, &a, (cuDoubleComplex*)dA, 64, (cuDoubleComplex*)dB, 64, &b, (cuDoubleComplex*)dC, 50);
  printf("zgemm status ok=%d\n", cudaDeviceSynchronize() == cudaSuccess);
  return 0;
}
