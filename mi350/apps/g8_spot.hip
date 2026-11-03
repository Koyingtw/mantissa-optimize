// Post-hoc FP64 spot check of an emulated GEMM (review rounds 7-8, P3/Q1): after GEMMul8 (INT8 backend, direct API)
// computes C = op(A) op(B), recompute s sampled rows of C with a native DGEMM and compare. The spot estimate
// ||dC_S||_F / ||C_S||_F is set against the full normwise error ||dC||_F / ||C||_F (from a full native reference) and
// against the per-row span statistic of the span check (share of op(A) elements more than 2^-32 below their row maximum).
// The spot check's cost is its gather plus an s x n x k DGEMM, timed with HIP events against the emulated call.
// It samples rows when m >= n and columns (an m x s x k DGEMM) otherwise, so its cost stays a small share of the call.
// --stratify takes half of the samples from the rows (columns) with the widest span and half at random.
// Per sampled unit (row or column) the tool also reports how many units exceed a relative error of 1e-10 / 1e-8 / 1e-6
// (a failure confined to a share p of the units is found by s random samples with probability 1 - (1 - p)^s).
// --cw adds the componentwise error max |dC| / (|op(A)| |op(B)|), full and on the sampled units, and times the spot check
// with it (one more s x n x k DGEMM on absolute values). --trunc "36 32" evaluates truncation (round to nearest) on the
// same data for comparison (TRUNC lines).
//
// Operands: --a / --b FILE[@offset] (raw FP64, column-major in the stored layout, as bench/gemm_mant_bench --load-a),
// or --synthetic hidden|benign:<binades>: op(A) = [A1, 2^-e A2], op(B) = [2^-e B1; B2] (hidden: every product term is
// 2^-e small, while each row of A and column of B has its maximum in the large half) or [B1; B2] (benign: the small half
// contributes 2^-e of C). Both have half of their elements 2^-e below the row maximum; only "hidden" loses C.
// benign:0 gives plain uniform operands (used for the m n >= 2^32 size failure).
//
// usage: g8_spot --m M --n N --k K [--ta] [--tb] (--a F --b F | --synthetic hidden:40) [--moduli "10 12 14"]
//                [--rows 64] [--stratify] [--cw] [--trunc "36"] [--seed 1] [--device 0] [--tag s]
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

// Span statistic of nvec vectors of length len, element l of vector v at X[v * sv + l * se]: the number of nonzero
// elements more than 2^-32 below the vector's largest magnitude, per vector (cnt_v, may be null) and in total.
__global__ void vec_span(const double* X, long nvec, long len, long sv, long se, unsigned long long* total, int* cnt_v) {
  __shared__ double smax[256];
  __shared__ unsigned long long scnt[256];
  for (long v = blockIdx.x; v < nvec; v += gridDim.x) {
    double mx = 0;
    for (long l = threadIdx.x; l < len; l += blockDim.x) mx = fmax(mx, fabs(X[v * sv + l * se]));
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + s]);
      __syncthreads();
    }
    const double lim = smax[0] * 0x1.0p-32;
    unsigned long long c = 0;
    for (long l = threadIdx.x; l < len; l += blockDim.x) {
      const double x = fabs(X[v * sv + l * se]);
      c += x != 0 && x < lim;
    }
    scnt[threadIdx.x] = c;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) scnt[threadIdx.x] += scnt[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(total, scnt[0]);
      if (cnt_v) cnt_v[v] = (int)scnt[0];
    }
    __syncthreads();
  }
}

// per unit u (row of C if !bycol, else column): sums of dC^2 and C^2; C is m x n with ld m. Rows: one thread per row
// (neighbouring threads read neighbouring addresses); columns: one block per contiguous column.
__global__ void unit_sums(const double* C, const double* R, long m, long n, int bycol, double* d2, double* r2) {
  if (!bycol) {
    for (long u = blockIdx.x * (long)blockDim.x + threadIdx.x; u < m; u += (long)gridDim.x * blockDim.x) {
      double a = 0, b = 0;
      for (long l = 0; l < n; ++l) {
        const size_t i = (size_t)u + (size_t)l * m;
        const double d = C[i] - R[i];
        a += d * d;
        b += R[i] * R[i];
      }
      d2[u] = a;
      r2[u] = b;
    }
    return;
  }
  __shared__ double sd[256], sr[256];
  for (long u = blockIdx.x; u < n; u += gridDim.x) {
    double a = 0, b = 0;
    for (long l = threadIdx.x; l < m; l += blockDim.x) {
      const size_t i = (size_t)u * m + l;
      const double d = C[i] - R[i];
      a += d * d;
      b += R[i] * R[i];
    }
    sd[threadIdx.x] = a;
    sr[threadIdx.x] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) { sd[threadIdx.x] += sd[threadIdx.x + s]; sr[threadIdx.x] += sr[threadIdx.x + s]; }
      __syncthreads();
    }
    if (threadIdx.x == 0) { d2[u] = sd[0]; r2[u] = sr[0]; }
    __syncthreads();
  }
}

// largest |dC| / (|A||B|) over all of C (out[0]) and over the sampled units (out[1]); non-negative doubles compare as
// their bit patterns
__global__ void cw_max(const double* C, const double* R, const double* AB, long m, size_t mn, const unsigned char* on,
                       int bycol, unsigned long long* out) {
  double a = 0, b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < mn; i += (size_t)gridDim.x * blockDim.x) {
    if (AB[i] > 0) {
      const double e = fabs(C[i] - R[i]) / AB[i];
      a = fmax(a, e);
      if (on[bycol ? i / m : i % m]) b = fmax(b, e);
    }
  }
  atomicMax(out, (unsigned long long)__double_as_longlong(a));
  atomicMax(out + 1, (unsigned long long)__double_as_longlong(b));
}

// counts over all of C: entries with |dC| / (|A||B|) above 1e-9 / 1e-6 / 1e-3, and entries whose bound |A||B| is below
// 1e-290 (near the bottom of the FP64 range, where relative errors lose meaning)
__global__ void cw_count(const double* C, const double* R, const double* AB, size_t mn, unsigned long long* cnt) {
  unsigned long long c[4] = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < mn; i += (size_t)gridDim.x * blockDim.x) {
    if (AB[i] < 1e-290) { c[3] += AB[i] > 0; continue; }
    const double e = fabs(C[i] - R[i]) / AB[i];
    c[0] += e > 1e-9;
    c[1] += e > 1e-6;
    c[2] += e > 1e-3;
  }
  for (int j = 0; j < 4; ++j) atomicAdd(cnt + j, c[j]);
}

__global__ void abs_copy(const double* X, double* Y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) Y[i] = fabs(X[i]);
}

__global__ void round_copy(const double* X, double* Y, size_t n, int bits) {
  const int drop = 52 - bits;
  const uint64_t add = 1ull << (drop - 1), mask = ~((1ull << drop) - 1ull);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    Y[i] = __longlong_as_double((long long)(((uint64_t)__double_as_longlong(X[i]) + add) & mask));
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
  int ta = 0, tb = 0, s = 64, device = 0, stratify = 0, cw = 0;
  uint64_t seed = 1;
  std::string fa, fb, syn, mods = "10 12 14", tag = "run", truncs;
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
    else if (!strcmp(argv[i], "--stratify")) stratify = 1;
    else if (!strcmp(argv[i], "--cw")) cw = 1;
    else if (!strcmp(argv[i], "--trunc")) truncs = next();
    else if (!strcmp(argv[i], "--seed")) seed = strtoull(next(), nullptr, 10);
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  if (!m || !n || !k || (syn.empty() && (fa.empty() || fb.empty()))) { fprintf(stderr, "see the header for usage\n"); return 2; }
  CK(hipSetDevice(device));
  hipblasHandle_t h;
  CB(hipblasCreate(&h));
  const size_t mk = (size_t)m * k, kn = (size_t)k * n, mn = (size_t)m * n;
  double *A, *B, *C, *R, *G, *CS, *acc, *d2, *r2;
  unsigned char* row_on;
  int *rows_d, *cnt_v;
  unsigned long long *cnt, *cwo;
  CK(hipMalloc(&A, mk * 8));
  CK(hipMalloc(&B, kn * 8));
  CK(hipMalloc(&C, mn * 8));
  CK(hipMalloc(&R, mn * 8));
  const int bycol = n > m;
  const long dim = bycol ? n : m;
  CK(hipMalloc(&G, (size_t)s * k * 8));
  CK(hipMalloc(&CS, (size_t)s * (bycol ? m : n) * 8));
  CK(hipMalloc(&acc, 4 * 8));
  CK(hipMalloc(&row_on, dim));
  CK(hipMalloc(&rows_d, s * sizeof(int)));
  CK(hipMalloc(&cnt, 8));
  CK(hipMalloc(&cwo, 16));
  CK(hipMalloc(&cnt_v, dim * sizeof(int)));
  CK(hipMalloc(&d2, dim * 8));
  CK(hipMalloc(&r2, dim * 8));
  if (!syn.empty()) {
    if (ta || tb) { fprintf(stderr, "--synthetic uses the N N layout\n"); return 2; }
    const size_t c = syn.find(':');
    const double e = c == std::string::npos ? 40 : atof(syn.c_str() + c + 1);
    synth<<<4096, 256>>>(A, B, m, n, k, std::ldexp(1.0, -(int)e), syn.rfind("hidden", 0) == 0);
  } else {
    load64(fa, A, mk);
    load64(fb, B, kn);
  }
  // span statistics: rows of op(A) (row i, element l: A[i + l m], or A[l + i k] with ta) and columns of op(B)
  // (column j, element l: B[l + j k], or B[j + l n] with tb)
  auto span_of = [&](int ofA, int* per) {
    CK(hipMemset(cnt, 0, 8));
    if (ofA) vec_span<<<1024, 256>>>(A, m, k, ta ? k : 1, ta ? 1 : m, cnt, per);
    else vec_span<<<1024, 256>>>(B, n, k, tb ? 1 : k, tb ? n : 1, cnt, per);
    unsigned long long t;
    CK(hipMemcpy(&t, cnt, 8, hipMemcpyDeviceToHost));
    return (double)t / (double)(ofA ? mk : kn);
  };
  const double span = span_of(1, bycol ? nullptr : cnt_v);
  const double span_b = span_of(0, bycol ? cnt_v : nullptr);

  std::mt19937_64 rng(seed);
  std::vector<int> rows;
  std::vector<unsigned char> on(dim, 0);
  if (stratify) {  // half from the units with the widest span
    std::vector<int> cv(dim), idx(dim);
    CK(hipMemcpy(cv.data(), cnt_v, dim * sizeof(int), hipMemcpyDeviceToHost));
    for (long i = 0; i < dim; ++i) idx[i] = (int)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cv[a] > cv[b]; });
    for (int i = 0; i < s / 2; ++i) { on[idx[i]] = 1; rows.push_back(idx[i]); }
  }
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
  // componentwise reference |op(A)| |op(B)| and the cost of the spot check with it
  double *AB = nullptr, *Aa = nullptr, *Ba = nullptr;
  float t_spot_cw = 0;
  if (cw) {
    CK(hipMalloc(&AB, mn * 8));
    CK(hipMalloc(&Aa, mk * 8));
    CK(hipMalloc(&Ba, kn * 8));
    abs_copy<<<4096, 256>>>(A, Aa, mk);
    abs_copy<<<4096, 256>>>(B, Ba, kn);
    CB(hipblasDgemm(h, oa, ob, m, n, k, &one, Aa, lda, Ba, ldb, &zero, AB, m));
    double* G2;
    CK(hipMalloc(&G2, (size_t)s * k * 8));
    t_spot_cw = timed([&] {  // gather, |gathered| and |other operand|, two DGEMMs
      if (bycol) {
        gather_cols<<<1024, 256>>>(B, n, k, tb, rows_d, s, G);
        abs_copy<<<1024, 256>>>(G, G2, (size_t)s * k);
        abs_copy<<<4096, 256>>>(A, Aa, mk);
        CB(hipblasDgemm(h, oa, HIPBLAS_OP_N, m, s, k, &one, A, lda, G, k, &zero, CS, m));
        CB(hipblasDgemm(h, oa, HIPBLAS_OP_N, m, s, k, &one, Aa, lda, G2, k, &zero, CS, m));
      } else {
        gather_rows<<<1024, 256>>>(A, m, k, ta, rows_d, s, G);
        abs_copy<<<1024, 256>>>(G, G2, (size_t)s * k);
        abs_copy<<<4096, 256>>>(B, Ba, kn);
        CB(hipblasDgemm(h, HIPBLAS_OP_N, ob, s, n, k, &one, G, s, B, ldb, &zero, CS, s));
        CB(hipblasDgemm(h, HIPBLAS_OP_N, ob, s, n, k, &one, G2, s, Ba, ldb, &zero, CS, s));
      }
    });
    CK(hipFree(G2));
  }
  // error report of a result X against the native reference R
  auto report = [&](const char* kind, const char* setting, const double* X, float t_x) {
    CK(hipMemset(acc, 0, 4 * 8));
    err_sums<<<2048, 256>>>(X, R, m, mn, row_on, bycol, acc);
    double a[4];
    CK(hipMemcpy(a, acc, sizeof(a), hipMemcpyDeviceToHost));
    unit_sums<<<bycol ? 1024 : (int)((m + 255) / 256), 256>>>(X, R, m, n, bycol, d2, r2);
    std::vector<double> hd(dim), hr(dim);
    CK(hipMemcpy(hd.data(), d2, dim * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr.data(), r2, dim * 8, hipMemcpyDeviceToHost));
    long bad[3] = {0, 0, 0}, nu = 0;
    double umax = 0;
    for (long u = 0; u < dim; ++u) {
      if (hr[u] <= 0) continue;
      const double e = std::sqrt(hd[u] / hr[u]);
      ++nu;
      umax = std::max(umax, e);
      bad[0] += e > 1e-10;
      bad[1] += e > 1e-8;
      bad[2] += e > 1e-6;
    }
    double cwf = -1, cws = -1;
    unsigned long long cc[4] = {0, 0, 0, 0};
    if (cw) {
      unsigned long long* dcc;
      CK(hipMalloc(&dcc, 32));
      CK(hipMemset(dcc, 0, 32));
      cw_count<<<2048, 256>>>(X, R, AB, mn, dcc);
      CK(hipMemcpy(cc, dcc, 32, hipMemcpyDeviceToHost));
      CK(hipFree(dcc));
      CK(hipMemset(cwo, 0, 16));
      cw_max<<<2048, 256>>>(X, R, AB, m, mn, row_on, bycol, cwo);
      unsigned long long hc[2];
      CK(hipMemcpy(hc, cwo, 16, hipMemcpyDeviceToHost));
      memcpy(&cwf, hc, 8);
      memcpy(&cws, hc + 1, 8);
    }
    const double full = std::sqrt(a[0] / a[1]), spot = std::sqrt(a[2] / a[3]);
    printf("%s tag=%s m=%ld n=%ld k=%ld %s %s=%d stratify=%d span=%.4f span_b=%.4f err_full=%.3e err_spot=%.3e ratio=%.3f "
           "units=%ld bad10=%ld bad8=%ld bad6=%ld unit_max=%.3e cw_full=%.3e cw_spot=%.3e cw9=%llu cw6=%llu cw3=%llu tiny=%llu "
           "t_native_ms=%.3f t_emu_ms=%.3f t_spot_ms=%.3f spot_over_emu=%.4f t_spot_cw_ms=%.3f\n",
           kind, tag.c_str(), m, n, k, setting, bycol ? "cols" : "rows", s, stratify, span, span_b, full, spot,
           spot / full, nu, bad[0], bad[1], bad[2], umax, cwf, cws, cc[0], cc[1], cc[2], cc[3], t_nat, t_x, t_spot, t_x > 0 ? t_spot / t_x : 0.0,
           t_spot_cw);
    fflush(stdout);
  };
  std::istringstream ms(mods);
  for (int M; ms >> M;) {
    const size_t ws = gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(m, n, k, M);
    void* work;
    if (hipMalloc(&work, ws) != hipSuccess) {
      printf("SPOT tag=%s moduli=%d status=no_workspace bytes=%zu\n", tag.c_str(), M, ws);
      fflush(stdout);
      continue;
    }
    const float t_emu = timed([&] {
      gemmul8::gemm<double, gemmul8::Backend::INT8>(h, oa, ob, m, n, k, &one, A, lda, B, ldb, &zero, C, m, M, true, work);
    });
    char set[32];
    snprintf(set, sizeof set, "moduli=%d", M);
    report("SPOT", set, C, t_emu);
    CK(hipFree(work));
  }
  std::istringstream ts(truncs);
  for (int bits; ts >> bits;) {  // truncation (round to nearest) of both operands on the same data
    double *At, *Bt;
    CK(hipMalloc(&At, mk * 8));
    CK(hipMalloc(&Bt, kn * 8));
    round_copy<<<4096, 256>>>(A, At, mk, bits);
    round_copy<<<4096, 256>>>(B, Bt, kn, bits);
    const float t_tr = timed([&] { CB(hipblasDgemm(h, oa, ob, m, n, k, &one, At, lda, Bt, ldb, &zero, C, m)); });
    char set[32];
    snprintf(set, sizeof set, "bits=%d", bits);
    report("TRUNC", set, C, t_tr);
    CK(hipFree(At));
    CK(hipFree(Bt));
  }
  return 0;
}
