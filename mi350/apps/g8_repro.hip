// Minimal reproducer for the GEMMul8 failure seen in the GP likelihood at n=80000 (trailing update m=n=78976, k=1024):
// C = A B^T with A (m x k) and B (n x k) uniform in [-1,1), computed by native FP64 (hipblasDgemm, reference) and by
// GEMMul8's direct API (gemmul8::gemm, INT8 backend, no hook), for each m = n given. Prints the relative Frobenius error,
// the number of non-finite entries and the largest elementwise error, so a size threshold (e.g. m*n crossing 2^32)
// shows up directly.
//   g8_repro --moduli 14 --k 1024 --sizes "65000 65535 65536 65537 66000 70000 78976"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "gemmul8.hpp"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void fill(double* x, size_t n, uint64_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t z = seed ^ (i * 0xd1b54a32d192ed03ull);
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    x[i] = (double)(z >> 11) * 0x1.0p-52 - 1.0;
  }
}

// acc[0] = sum (c-r)^2, acc[1] = sum r^2, acc[2] = non-finite count, acc[3] = max |c-r|
__global__ void compare(const double* c, const double* r, size_t n, double* acc) {
  __shared__ double s0[256], s1[256], s2[256], s3[256];
  double d = 0, q = 0, bad = 0, mx = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    if (!isfinite(c[i])) { bad += 1; continue; }
    const double e = c[i] - r[i];
    d += e * e;
    q += r[i] * r[i];
    mx = fmax(mx, fabs(e));
  }
  s0[threadIdx.x] = d; s1[threadIdx.x] = q; s2[threadIdx.x] = bad; s3[threadIdx.x] = mx;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) {
      s0[threadIdx.x] += s0[threadIdx.x + k]; s1[threadIdx.x] += s1[threadIdx.x + k];
      s2[threadIdx.x] += s2[threadIdx.x + k]; s3[threadIdx.x] = fmax(s3[threadIdx.x], s3[threadIdx.x + k]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(acc, s0[0]); atomicAdd(acc + 1, s1[0]); atomicAdd(acc + 2, s2[0]);
    // max via CAS on the bit pattern (values are non-negative, so integer order matches)
    unsigned long long* p = (unsigned long long*)(acc + 3);
    unsigned long long old = *p, v = __double_as_longlong(s3[0]);
    while (v > old) { unsigned long long prev = atomicCAS(p, old, v); if (prev == old) break; old = prev; }
  }
}

int main(int argc, char** argv) {
  int moduli = 14, device = 0;
  size_t k = 1024;
  std::string sizes = "65000 65535 65536 65537 66000 70000 78976";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--moduli")) moduli = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--k")) k = strtoull(argv[++i], nullptr, 10);
    else if (!strcmp(argv[i], "--sizes")) sizes = argv[++i];
    else if (!strcmp(argv[i], "--device")) device = atoi(argv[++i]);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  CK(hipSetDevice(device));
  hipblasHandle_t h;
  if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "hipblasCreate failed\n"); return 1; }
  std::istringstream is(sizes);
  size_t m;
  while (is >> m) {
    const size_t n = m;
    double *A, *B, *C, *R, *acc;
    CK(hipMalloc(&A, m * k * 8));
    CK(hipMalloc(&B, n * k * 8));
    CK(hipMalloc(&C, m * n * 8));
    CK(hipMalloc(&R, m * n * 8));
    CK(hipMalloc(&acc, 4 * 8));
    fill<<<4096, 256>>>(A, m * k, 1);
    fill<<<4096, 256>>>(B, n * k, 2);
    const double one = 1.0, zero = 0.0;
    if (hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, &one, A, m, B, n, &zero, R, m) != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "reference dgemm failed\n"); return 1;
    }
    const size_t ws = gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(m, n, k, moduli);
    void* work = nullptr;
    const hipError_t we = hipMalloc(&work, ws);
    std::string status = "ok";
    if (we != hipSuccess) {
      status = "workspace_alloc_failed";
      (void)hipGetLastError();
    } else {
      CK(hipMemset(C, 0, m * n * 8));
      gemmul8::gemm<double, gemmul8::Backend::INT8>(h, HIPBLAS_OP_N, HIPBLAS_OP_T, m, n, k, &one, A, m, B, n, &zero, C, m, moduli, true, work);
      const hipError_t ke = hipDeviceSynchronize();
      if (ke != hipSuccess) status = std::string("kernel_error:") + hipGetErrorString(ke);
    }
    double hacc[4] = {0, 0, 0, 0};
    if (status == "ok") {
      CK(hipMemset(acc, 0, 4 * 8));
      compare<<<4096, 256>>>(C, R, m * n, acc);
      CK(hipMemcpy(hacc, acc, sizeof(hacc), hipMemcpyDeviceToHost));
    }
    printf("RESULT_G8 m=%zu n=%zu k=%zu mn=%zu mn_over_2^32=%.4f moduli=%d workspace_GB=%.2f status=%s rel_err=%.3e "
           "nonfinite=%.0f max_abs_err=%.3e\n", m, n, k, m * n, (double)(m * n) / 4294967296.0, moduli, ws / 1e9,
           status.c_str(), hacc[1] > 0 ? std::sqrt(hacc[0] / hacc[1]) : NAN, hacc[2], hacc[3]);
    fflush(stdout);
    if (work) CK(hipFree(work));
    CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C)); CK(hipFree(R)); CK(hipFree(acc));
  }
  hipblasDestroy(h);
  return 0;
}
