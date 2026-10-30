// Gaussian-process (geostatistics) log-likelihood with a blocked Cholesky on one MI350X: the HIP/rocBLAS
// port of apps/geostat/gp_mle.py (default mode, order=none) for the MI350 study.
//
// n random 2-D locations in [0,1)^2, Matern-3/2 covariance with a nugget, right-looking blocked Cholesky:
// rocSOLVER POTRF on the diagonal block, rocBLAS TRSM for the panel L21, then the panel is copied into a
// scratch buffer (keeping only k fraction bits, round-to-nearest or truncation, one fused coalesced pass)
// and the trailing update A22 -= L21 L21^T runs as one FP64 DGEMM over the full square block (only the
// lower part is used), as in the CUDA version. Observations are drawn from the model at the middle range
// (factor once in native FP64, y = L z).
//
// usage: gp_chol --device D --variants "native t36n t28n t28n t36n native" [--n 80000] [--nb 1024]
//                [--ranges "0.02 0.03 0.04 0.05 0.06 0.08"] [--nugget 1e-4] [--seed 1] [--tag s]
//   variant: native | t<k><n|t> (k fraction bits kept, n = nearest, t = truncate); a variant listed twice
//   is its B pass. --hipblas runs the trailing DGEMM through hipblasDgemm instead of rocblas_dgemm, so that
//   GEMMul8's LD_PRELOAD hook (which intercepts hipBLAS) can replace it with INT8 emulation.
//   Prints one RESULT_GP line per variant (log-likelihoods, deviation from the first native
//   run, wall time, Unix start/end).
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CR(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
  fprintf(stderr, "rocBLAS %s:%d %s\n", __FILE__, __LINE__, rocblas_status_to_string(s_)); exit(1); } } while (0)

static double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

__global__ void covariance(double* K, const double* xy, long n, double s, double nugget) {
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < n * n; idx += (long)gridDim.x * blockDim.x) {
    const long i = idx % n, j = idx / n;
    const double dx = xy[2 * i] - xy[2 * j], dy = xy[2 * i + 1] - xy[2 * j + 1];
    const double d = s * sqrt(dx * dx + dy * dy);
    K[idx] = (1 + d) * exp(-d) + (i == j ? nugget : 0.0);
  }
}

// Copy the r x b panel (column-major inside the n x n matrix) into Lt (ld r), keeping `bits` of the 52 fraction
// bits: add half a kept ulp (nearest; ties away from zero in magnitude) or nothing (truncate), then mask.
__global__ void copy_reduce(const double* A, long lda, double* Lt, long r, long b, uint64_t add, uint64_t mask) {
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < r * b; idx += (long)gridDim.x * blockDim.x) {
    const long i = idx % r, c = idx / r;
    uint64_t u;
    const double v = A[i + c * lda];
    memcpy(&u, &v, 8);
    u = (u + add) & mask;
    double w;
    memcpy(&w, &u, 8);
    Lt[idx] = w;
  }
}

__global__ void log_diag_sum(const double* A, long n, double* out) {
  __shared__ double s[256];
  double acc = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    acc += log(A[i + i * n]);
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

struct Ctx {
  rocblas_handle h;
  hipblasHandle_t hb = nullptr;  // set when the trailing DGEMM goes through hipBLAS
  long n, nb;
  double* Lt;      // panel scratch, (n - nb) x nb
  rocblas_int* info;
};

// Returns false if the matrix is not positive definite (POTRF info != 0).
static bool cholesky(Ctx& c, double* A, int bits, bool nearest) {
  const long n = c.n;
  const double one = 1.0, mone = -1.0;
  uint64_t add = 0, mask = ~0ull;
  if (bits < 52) {
    const int drop = 52 - bits;
    add = nearest ? 1ull << (drop - 1) : 0;
    mask = ~((1ull << drop) - 1ull);
  }
  for (long j = 0; j < n; j += c.nb) {
    const long b = std::min(c.nb, n - j);
    double* D = A + j + j * n;
    CR(rocsolver_dpotrf(c.h, rocblas_fill_lower, b, D, n, c.info));
    rocblas_int info;
    CK(hipMemcpy(&info, c.info, sizeof(info), hipMemcpyDeviceToHost));
    if (info != 0) return false;
    const long r = n - j - b;
    if (r == 0) break;
    // L21 = A21 L11^-T
    CR(rocblas_dtrsm(c.h, rocblas_side_right, rocblas_fill_lower, rocblas_operation_transpose,
                     rocblas_diagonal_non_unit, r, b, &one, D, n, A + (j + b) + j * n, n));
    copy_reduce<<<4096, 256>>>(A + (j + b) + j * n, n, c.Lt, r, b, add, mask);
    if (c.hb) {
      if (hipblasDgemm(c.hb, HIPBLAS_OP_N, HIPBLAS_OP_T, r, r, b, &mone, c.Lt, r, c.Lt, r, &one,
                       A + (j + b) + (j + b) * n, n) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "hipblasDgemm failed\n"); exit(1); }
    } else {
      CR(rocblas_dgemm(c.h, rocblas_operation_none, rocblas_operation_transpose, r, r, b, &mone, c.Lt, r, c.Lt, r,
                       &one, A + (j + b) + (j + b) * n, n));
    }
  }
  CK(hipDeviceSynchronize());
  return true;
}

static double loglik(Ctx& c, const double* A, const double* y, double* z, double* acc) {
  const long n = c.n;
  CK(hipMemset(acc, 0, sizeof(double)));
  log_diag_sum<<<256, 256>>>(A, n, acc);
  double logdet;
  CK(hipMemcpy(&logdet, acc, sizeof(double), hipMemcpyDeviceToHost));
  logdet *= 2;
  CK(hipMemcpy(z, y, n * sizeof(double), hipMemcpyDeviceToDevice));
  CR(rocblas_dtrsv(c.h, rocblas_fill_lower, rocblas_operation_none, rocblas_diagonal_non_unit, n, A, n, z, 1));
  double q;
  CR(rocblas_ddot(c.h, n, z, 1, z, 1, &q));
  return -0.5 * (q + logdet + n * std::log(2 * M_PI));
}

int main(int argc, char** argv) {
  long n = 80000, nb = 1024;
  int device = 0, use_hipblas = 0;
  uint64_t seed = 1;
  double nugget = 1e-4;
  std::string variants = "native", ranges_s = "0.02 0.03 0.04 0.05 0.06 0.08", tag = "run";
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
    if (!strcmp(argv[i], "--n")) n = atol(next());
    else if (!strcmp(argv[i], "--nb")) nb = atol(next());
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(next(), nullptr, 10);
    else if (!strcmp(argv[i], "--nugget")) nugget = atof(next());
    else if (!strcmp(argv[i], "--variants")) variants = next();
    else if (!strcmp(argv[i], "--ranges")) ranges_s = next();
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else if (!strcmp(argv[i], "--hipblas")) use_hipblas = 1;
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  std::vector<double> ranges;
  { std::istringstream is(ranges_s); double r; while (is >> r) ranges.push_back(r); }
  CK(hipSetDevice(device));
  Ctx c;
  c.n = n;
  c.nb = nb;
  CR(rocblas_create_handle(&c.h));
  CR(rocblas_set_pointer_mode(c.h, rocblas_pointer_mode_host));

  double *A, *xy, *y, *z, *acc;
  CK(hipMalloc(&A, (size_t)n * n * 8));
  CK(hipMalloc(&c.Lt, (size_t)(n - nb) * nb * 8));
  CK(hipMalloc(&c.info, sizeof(rocblas_int)));
  CK(hipMalloc(&xy, (size_t)n * 2 * 8));
  CK(hipMalloc(&y, (size_t)n * 8));
  CK(hipMalloc(&z, (size_t)n * 8));
  CK(hipMalloc(&acc, 8));

  std::mt19937_64 rs(seed);
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> G(0, 1);
  std::vector<double> h(2 * n);
  for (auto& v : h) v = U(rs);
  CK(hipMemcpy(xy, h.data(), h.size() * 8, hipMemcpyHostToDevice));
  // observations from the model at the middle range: y = L z, native FP64 factor
  const double mid = ranges[ranges.size() / 2];
  covariance<<<8192, 256>>>(A, xy, n, std::sqrt(3.0) / mid, nugget);
  if (!cholesky(c, A, 52, true)) { fprintf(stderr, "data covariance not positive definite\n"); return 1; }
  std::vector<double> zh(n);
  for (auto& v : zh) v = G(rs);
  CK(hipMemcpy(y, zh.data(), n * 8, hipMemcpyHostToDevice));
  CR(rocblas_dtrmv(c.h, rocblas_fill_lower, rocblas_operation_none, rocblas_diagonal_non_unit, n, A, n, y, 1));
  CK(hipDeviceSynchronize());
  char bdf[32];
  CK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  printf("n=%ld nb=%ld ranges=[%s] nugget=%g seed=%llu gpu=%d bdf=%s\n", n, nb, ranges_s.c_str(), nugget,
         (unsigned long long)seed, device, bdf);
  fflush(stdout);

  // the observations are generated with rocBLAS above, so an emulated run sees exactly the same data
  if (use_hipblas && hipblasCreate(&c.hb) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "hipblasCreate failed\n"); return 1; }
  std::vector<double> ref;
  std::map<std::string, int> seen;
  std::istringstream vs(variants);
  std::string v;
  while (vs >> v) {
    const char pas = "ABCDEFGH"[seen[v]++];
    int bits = 52;
    bool nearest = true;
    if (v != "native") {
      if (v[0] != 't') { fprintf(stderr, "bad variant %s\n", v.c_str()); return 2; }
      bits = atoi(v.c_str() + 1);
      nearest = v.back() == 'n';
    }
    std::vector<double> lls;
    bool ok = true;
    const double t0 = now_unix();
    for (double rg : ranges) {
      covariance<<<8192, 256>>>(A, xy, n, std::sqrt(3.0) / rg, nugget);
      if (!cholesky(c, A, bits, nearest)) { lls.push_back(NAN); ok = false; continue; }
      lls.push_back(loglik(c, A, y, z, acc));
    }
    const double t1 = now_unix();
    if (v == "native" && ref.empty()) ref = lls;
    std::ostringstream ll, dl;
    for (size_t i = 0; i < lls.size(); ++i) {
      char b1[64], b2[64];
      snprintf(b1, sizeof b1, "%s%.10f", i ? " " : "", lls[i]);
      snprintf(b2, sizeof b2, "%s%.3e", i ? " " : "", ref.empty() ? NAN : lls[i] - ref[i]);
      ll << b1;
      dl << b2;
    }
    printf("RESULT_GP tag=%s gpu=%d variant=%s pass=%c status=%s loglik=[%s] dll=[%s] wall=%.3fs t_begin=%.3f t_end=%.3f\n",
           tag.c_str(), device, v.c_str(), pas, ok ? "ok" : "notpd", ll.str().c_str(), dl.str().c_str(), t1 - t0, t0, t1);
    fflush(stdout);
  }
  rocblas_destroy_handle(c.h);
  return 0;
}
