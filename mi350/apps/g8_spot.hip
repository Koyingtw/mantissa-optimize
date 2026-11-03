// Post-hoc FP64 spot check of an emulated GEMM (review round 7, P3): after GEMMul8 (INT8 backend, direct API) computes
// C = op(A) op(B), recompute s randomly chosen rows of C with a native DGEMM and compare. The spot estimate
// ||dC_S||_F / ||C_S||_F is set against the full normwise error ||dC||_F / ||C||_F (from a full native reference) and
// against the per-row span statistic of the span check (share of op(A) elements more than 2^-32 below their row maximum).
// The spot check's cost is its gather plus an s x n x k DGEMM, timed with HIP events against the emulated call.
// It samples rows when m >= n and columns (an m x s x k DGEMM) otherwise, so its cost stays a small share of the call.
//
// Operands: --a / --b FILE[@offset] (raw FP64, column-major in the stored layout, as bench/gemm_mant_bench --load-a),
// or --synthetic hidden|benign:<binades>: op(A) = [A1, 2^-e A2], op(B) = [2^-e B1; B2] (hidden: every product term is
// 2^-e small, while each row of A and column of B has its maximum in the large half) or [B1; B2] (benign: the small half
// contributes 2^-e of C). Both have half of their elements 2^-e below the row maximum; only "hidden" loses C.
//
// usage: g8_spot --m M --n N --k K [--ta] [--tb] (--a F --b F | --synthetic hidden:40) [--moduli "10 12 14"]
//                [--rows 64] [--seed 1] [--device 0] [--tag s]
// One SPOT line per moduli count: full error, spot error, ratio, span share, times of native / emulated / spot.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "gemmul8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "hipBLAS %s:%d %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static void load64(const std::string& spec, double* d, size_t n) {
  const size_t at = spec.find('@');
  const std::string path = spec.substr(0, at);
  const size_t off = at == std::string::npos ? 0 : strtoull(spec.c_str() + at + 1, nullptr, 10);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fseeko(f, (off_t)(off * 8), SEEK_SET) != 0) { fprintf(stderr, "cannot open %s\n", spec.c_str()); exit(1); }
  std::vector<double> h(n);
  if (fread(h.data(), 8, n, f) != n) { fprintf(stderr, "%s: too short\n", path.c_str()); exit(1); }
  fclose(f);
  CK(hipMemcpy(d, h.data(), n * 8, hipMemcpyHostToDevice));
}

__device__ __forceinline__ double unif(uint64_t seed, size_t i) {
  uint64_t x = seed ^ (i * 0xd1b54a32d192ed03ull);
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x ^= x >> 31;
  return (double)(x >> 11) * 0x1.0p-52 - 1.0;
}

// A: m x k (ld m), B: k x n (ld k); the second half of k is small in A; in B the first half is small (hidden) or not.
__global__ void synth(double* A, double* B, long m, long n, long k, double small, int hidden) {
  const size_t na = (size_t)m * k, nb = (size_t)k * n;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < na + nb; i += (size_t)gridDim.x * blockDim.x) {
    if (i < na) {
      const long kk = (long)(i / m);
      A[i] = unif(11, i) * (kk >= k / 2 ? small : 1.0);
    } else {
      const size_t j = i - na;
      const long kk = (long)(j % k);
      B[j] = unif(23, j) * (hidden && kk < k / 2 ? small : 1.0);
    }
  }
}

// share of op(A) elements more than 2^-32 below their row maximum (row i of op(A): stored row i, or column i with ta)
__global__ void row_span(const double* A, long m, long k, int ta, unsigned long long* cnt) {
  for (long i = blockIdx.x; i < m; i += gridDim.x) {
    __shared__ double smax[256];
    double mx = 0;
    for (long j = threadIdx.x; j < k; j += blockDim.x) mx = fmax(mx, fabs(ta ? A[j + i * k] : A[i + j * m]));
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + s]);
      __syncthreads();
    }
    const double lim = smax[0] * 0x1.0p-32;
    unsigned long long c = 0;
    for (long j = threadIdx.x; j < k; j += blockDim.x) {
      const double v = fabs(ta ? A[j + i * k] : A[i + j * m]);
      c += v != 0 && v < lim;
    }
    atomicAdd(cnt, c);
    __syncthreads();
  }
}

// sums of dC^2 and C^2 over all of C and over the sampled rows (bycol = 0) or columns (bycol = 1)
__global__ void err_sums(const double* C, const double* R, long m, size_t mn, const unsigned char* row_on, int bycol,
                         double* acc) {
  double s[4] = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < mn; i += (size_t)gridDim.x * blockDim.x) {
    const double d = C[i] - R[i], r = R[i];
    s[0] += d * d;
    s[1] += r * r;
    if (row_on[bycol ? i / m : i % m]) { s[2] += d * d; s[3] += r * r; }
  }
  for (int j = 0; j < 4; ++j) atomicAdd(acc + j, s[j]);
}

__global__ void gather_rows(const double* A, long m, long k, int ta, const int* rows, int s, double* G) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)s * k; t += (size_t)gridDim.x * blockDim.x) {
    const int r = (int)(t % s);
    const long j = (long)(t / s);
    G[t] = ta ? A[j + (size_t)rows[r] * k] : A[rows[r] + (size_t)j * m];
  }
}

// columns j of op(B) into G (k x s): op(B) = B (k x n, ld k) or B^T (B stored n x k, ld n)
__global__ void gather_cols(const double* B, long n, long k, int tb, const int* cols, int s, double* G) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < (size_t)s * k; t += (size_t)gridDim.x * blockDim.x) {
    const long l = (long)(t % k);
    const int c = (int)(t / k);
    G[t] = tb ? B[cols[c] + (size_t)l * n] : B[l + (size_t)cols[c] * k];
  }
}

int main(int argc, char** argv) {
  long m = 0, n = 0, k = 0;
  int ta = 0, tb = 0, s = 64, device = 0;
  uint64_t seed = 1;
  std::string fa, fb, syn, mods = "10 12 14", tag = "run";
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
    if (!strcmp(argv[i], "--m")) m = atol(next());
    else if (!strcmp(argv[i], "--n")) n = atol(next());
    else if (!strcmp(argv[i], "--k")) k = atol(next());
    else if (!strcmp(argv[i], "--ta")) ta = 1;
    else if (!strcmp(argv[i], "--tb")) tb = 1;
    else if (!strcmp(argv[i], "--a")) fa = next();
    else if (!strcmp(argv[i], "--b")) fb = next();
    else if (!strcmp(argv[i], "--synthetic")) syn = next();
    else if (!strcmp(argv[i], "--moduli")) mods = next();
    else if (!strcmp(argv[i], "--rows")) s = atoi(next());
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(next(), nullptr, 10);
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  if (!m || !n || !k || (syn.empty() && (fa.empty() || fb.empty()))) { fprintf(stderr, "see the header for usage\n"); return 2; }
  CK(hipSetDevice(device));
  hipblasHandle_t h;
  CB(hipblasCreate(&h));
  double *A, *B, *C, *R, *G, *CS, *acc;
  unsigned char* row_on;
  int* rows_d;
  unsigned long long* cnt;
  CK(hipMalloc(&A, (size_t)m * k * 8));
  CK(hipMalloc(&B, (size_t)k * n * 8));
  CK(hipMalloc(&C, (size_t)m * n * 8));
  CK(hipMalloc(&R, (size_t)m * n * 8));
  const int bycol = n > m;
  const long dim = bycol ? n : m;
  CK(hipMalloc(&G, (size_t)s * k * 8));
  CK(hipMalloc(&CS, (size_t)s * (bycol ? m : n) * 8));
  CK(hipMalloc(&acc, 4 * 8));
  CK(hipMalloc(&row_on, std::max(m, n)));
  CK(hipMalloc(&rows_d, s * sizeof(int)));
  CK(hipMalloc(&cnt, 8));
  if (!syn.empty()) {
    if (ta || tb) { fprintf(stderr, "--synthetic uses the N N layout\n"); return 2; }
    const size_t c = syn.find(':');
    const double e = c == std::string::npos ? 40 : atof(syn.c_str() + c + 1);
    synth<<<4096, 256>>>(A, B, m, n, k, std::ldexp(1.0, -(int)e), syn.rfind("hidden", 0) == 0);
  } else {
    load64(fa, A, (size_t)m * k);
    load64(fb, B, (size_t)k * n);
  }
  CK(hipMemset(cnt, 0, 8));
  row_span<<<1024, 256>>>(A, m, k, ta, cnt);
  unsigned long long nspan;
  CK(hipMemcpy(&nspan, cnt, 8, hipMemcpyDeviceToHost));
  const double span = (double)nspan / ((double)m * k);

  std::mt19937_64 rng(seed);
  std::vector<int> rows;
  std::vector<unsigned char> on(dim, 0);
  while ((int)rows.size() < s) {
    const int r = (int)(rng() % dim);
    if (!on[r]) { on[r] = 1; rows.push_back(r); }
  }
  CK(hipMemcpy(rows_d, rows.data(), s * sizeof(int), hipMemcpyHostToDevice));
  CK(hipMemcpy(row_on, on.data(), dim, hipMemcpyHostToDevice));

  const double one = 1.0, zero = 0.0;
  const hipblasOperation_t oa = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N, ob = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const long lda = ta ? k : m, ldb = tb ? n : k;
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  auto timed = [&](auto f) {
    f();  // warm-up (and GEMMul8's per-shape setup)
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0));
    for (int r = 0; r < 3; ++r) f();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    return ms / 3;
  };
  const float t_nat = timed([&] { CB(hipblasDgemm(h, oa, ob, m, n, k, &one, A, lda, B, ldb, &zero, R, m)); });
  const float t_spot = timed([&] {
    if (bycol) {
      gather_cols<<<1024, 256>>>(B, n, k, tb, rows_d, s, G);
      CB(hipblasDgemm(h, oa, HIPBLAS_OP_N, m, s, k, &one, A, lda, G, k, &zero, CS, m));
    } else {
      gather_rows<<<1024, 256>>>(A, m, k, ta, rows_d, s, G);
      CB(hipblasDgemm(h, HIPBLAS_OP_N, ob, s, n, k, &one, G, s, B, ldb, &zero, CS, s));
    }
  });
  std::istringstream ms(mods);
  for (int M; ms >> M;) {
    const size_t ws = gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(m, n, k, M);
    void* work;
    CK(hipMalloc(&work, ws));
    const float t_emu = timed([&] {
      gemmul8::gemm<double, gemmul8::Backend::INT8>(h, oa, ob, m, n, k, &one, A, lda, B, ldb, &zero, C, m, M, true, work);
    });
    CK(hipMemset(acc, 0, 4 * 8));
    err_sums<<<2048, 256>>>(C, R, m, (size_t)m * n, row_on, bycol, acc);
    double a[4];
    CK(hipMemcpy(a, acc, sizeof(a), hipMemcpyDeviceToHost));
    const double full = std::sqrt(a[0] / a[1]), spot = std::sqrt(a[2] / a[3]);
    printf("SPOT tag=%s m=%ld n=%ld k=%ld moduli=%d %s=%d span=%.4f err_full=%.3e err_spot=%.3e ratio=%.3f "
           "t_native_ms=%.3f t_emu_ms=%.3f t_spot_ms=%.3f spot_over_emu=%.4f\n",
           tag.c_str(), m, n, k, M, bycol ? "cols" : "rows", s, span, full, spot, spot / full, t_nat, t_emu, t_spot, t_spot / t_emu);
    fflush(stdout);
    CK(hipFree(work));
  }
  return 0;
}
