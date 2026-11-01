// GEMM throughput under a reduced operand mantissa on AMD Instinct MI350X (HIP port of
// gh200/bench/gemm_mant_bench.cu, E1 of the MI350 study).
//
// A and B are filled with uniform random values in [-1, 1), then the low fraction bits are cleared
// (or rounded) so only `--frac` explicit fraction bits remain. FP64 uses hipblasDgemm (rocBLAS FP64
// MFMA, the HPL trailing update); BF16 uses hipblasGemmEx with BF16 inputs, FP32 output and FP32
// accumulation (the HPL-MxP trailing update); FP32 uses hipblasSgemm; FP8 (OCP E4M3) uses hipBLASLt
// in the TN layout, FP32 accumulation and BF16 output, `--frac` keeping 0..3 of the E4M3 fraction bits.
// The measured window's wall-clock bounds are printed as Unix times so the amdsmi power log
// (mi350/tools/smi_sampler.py) can be joined to it.
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "hipBLAS(Lt) %s:%d %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

// What the dropped low fraction bits are filled with (mechanism control):
// 0 = zero (truncation), 1 = all ones, 2 = one fixed nonzero pattern shared by
// every element. 1 and 2 remove bit toggling without making the bits zero.
// FP64 only, temporal-toggling controls (review: separate toggling from values and Hamming weight):
// 3 = krep: the dropped bits are random per row of A / column of B but repeated along k, so they are
//     neither zero nor constant across the matrix, yet do not change between consecutive k;
// 6 = mrep: the transpose of krep, random along k but the same for every row of A / column of B at a given k;
// 4 = sorted, 5 = shuffled: the same multiset of values per row of A / column of B (stratified uniform in
//     [-1, 1) with random low bits), placed in increasing order along k (4) or in a pseudo-random order (5);
//     values and Hamming weights are identical, only the bit changes between consecutive k differ.
__constant__ int g_low_mode = 0;
static const char* kLowNames[] = {"zero", "ones", "const", "krep", "sorted", "shuffled", "mrep"};
__constant__ long g_kdim = 0;
constexpr uint64_t kConstPattern = 0xA5C3F0965A3C0F69ull;

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ double uniform_pm1(uint64_t seed, size_t i) {
  return (double)(splitmix64(seed ^ (i * 0xd1b54a32d192ed03ull)) >> 11) * 0x1.0p-52 - 1.0;
}

// Keep `frac` of the 52 FP64 fraction bits. round=1 adds half a kept ulp first.
// keybits: the krep pattern of this element's row (A) or column (B).
__device__ __forceinline__ double reduce64(double v, int frac, int round, uint64_t keybits = 0) {
  if (frac >= 52) return v;
  uint64_t u;
  memcpy(&u, &v, 8);
  const uint64_t drop = 52 - frac;
  if (round) u += 1ull << (drop - 1);
  const uint64_t low = (1ull << drop) - 1ull;
  u &= ~low;
  if (g_low_mode == 1) u |= low;
  else if (g_low_mode == 2) u |= kConstPattern & low;
  else if (g_low_mode == 3 || g_low_mode == 6) u |= keybits & low;
  memcpy(&v, &u, 8);
  return v;
}

__device__ __forceinline__ float reduce32(float v, int frac, int round) {
  if (frac >= 23) return v;
  uint32_t u;
  memcpy(&u, &v, 4);
  const uint32_t drop = 23 - frac;
  if (round) u += 1u << (drop - 1);
  const uint32_t low = (1u << drop) - 1u;
  u &= ~low;
  if (g_low_mode == 1) u |= low;
  else if (g_low_mode == 2) u |= (uint32_t)kConstPattern & low;
  memcpy(&v, &u, 4);
  return v;
}

// Exponent spread (range experiment, FP64 only): value = uniform[-1, 1) * 2^e with e ~ U(-S/2, S/2), i.e.
// log-uniform magnitudes over S binades. Mode 0 draws e per element, so every row and column spans S binades;
// mode 1 draws one e per row of A and per column of B, so each row/column is narrow (what per-row scaled
// emulation sees) while the matrix as a whole still spans S binades (what a per-tensor format sees).
__constant__ double g_spread = 0;
__constant__ int g_spread_mode = 0;

// col_key: the operand's scaling unit is a column (B, k x n) rather than a row (A, m x k); ld = leading dim.
__device__ __forceinline__ double spread_scale(uint64_t seed, size_t i, size_t ld, int col_key) {
  if (g_spread == 0) return 1.0;
  const size_t key = g_spread_mode == 0 ? i : (col_key ? i / ld : i % ld);
  const double u = (double)(splitmix64(splitmix64(seed + 0x5eedull) ^ (key * 0xd1b54a32d192ed03ull)) >> 11) * 0x1.0p-53;
  return exp2((u - 0.5) * g_spread);
}

__global__ void fill64(double* x, size_t n, uint64_t seed, int frac, int round, size_t ld, int col_key) {
  const int mode = g_low_mode;
  const size_t K = (size_t)g_kdim;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    // A (m x k, col-major, ld = m): row = i % ld, k index = i / ld; B (k x n, ld = k): column = i / ld, k index = i % ld
    const size_t key = col_key ? i / ld : i % ld, kidx = col_key ? i % ld : i / ld;
    const uint64_t kh = splitmix64(seed * 0x9E3779B97F4A7C15ull ^ (key + 1) * 0xC2B2AE3D27D4EB4Full);
    if (mode == 4 || mode == 5) {
      // K is a power of two, so j -> (a j + b) mod K with odd a is a bijection on 0..K-1
      const size_t jj = mode == 4 ? kidx : ((kh | 1) * kidx + (kh >> 32)) & (K - 1);
      const double u = (double)(splitmix64(kh ^ (jj * 0xd1b54a32d192ed03ull)) >> 11) * 0x1.0p-53;
      x[i] = reduce64(-1.0 + 2.0 * ((double)jj + u) / (double)K, frac, round);
    } else {
      const uint64_t pat = mode == 6 ? splitmix64(0x51ED270B27AB9C1Dull ^ (kidx + 1) * 0xC2B2AE3D27D4EB4Full) : kh;
      x[i] = reduce64(uniform_pm1(seed, i) * spread_scale(seed, i, ld, col_key), frac, round, pat);
    }
  }
}

// Bit statistics of an FP64 operand: acc[0] += popcount(x ^ previous element along k), acc[1] += popcount(x),
// acc[2] = pairs along k, acc[3] = elements, acc[4] += popcount(x ^ neighbour in the other dimension: previous row of
// A, previous column of B), acc[5] = pairs in that dimension.
__global__ void bitstats64(const double* x, size_t n, size_t ld, int col_key, unsigned long long* acc) {
  unsigned long long t = 0, h = 0, pairs = 0, cnt = 0, tx = 0, px = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t u = __double_as_longlong(x[i]);
    h += __popcll(u);
    ++cnt;
    const size_t kidx = col_key ? i % ld : i / ld, oidx = col_key ? i / ld : i % ld;
    if (kidx > 0) {
      const size_t prev = col_key ? i - 1 : i - ld;
      t += __popcll(u ^ (uint64_t)__double_as_longlong(x[prev]));
      ++pairs;
    }
    if (oidx > 0) {
      const size_t prev = col_key ? i - ld : i - 1;
      tx += __popcll(u ^ (uint64_t)__double_as_longlong(x[prev]));
      ++px;
    }
  }
  atomicAdd(acc, t); atomicAdd(acc + 1, h); atomicAdd(acc + 2, pairs); atomicAdd(acc + 3, cnt);
  atomicAdd(acc + 4, tx); atomicAdd(acc + 5, px);
}

__global__ void fill32(float* x, size_t n, uint64_t seed, int frac, int round) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = reduce32((float)uniform_pm1(seed, i), frac, round);
}

// frac <= 7 keeps the value exactly representable in BF16, so the final conversion does not round
// again; the dropped BF16 bits (16..22 of the FP32 pattern) take the low-mode fill.
__global__ void fill16(__hip_bfloat16* x, size_t n, uint64_t seed, int frac, int round) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = (float)uniform_pm1(seed, i);
    if (frac < 7) {
      const int mode = g_low_mode;
      uint32_t u;
      memcpy(&u, &v, 4);
      const uint32_t drop = 23 - frac;
      if (round) u += 1u << (drop - 1);
      u &= ~((1u << drop) - 1u);
      const uint32_t low = ((1u << drop) - 1u) & 0xFFFF0000u;
      if (mode) u |= (mode == 1 ? 0xFFFFFFFFu : (uint32_t)kConstPattern) & low;
      memcpy(&v, &u, 4);
    }
    x[i] = __float2bfloat16(v);
  }
}

// OCP E4M3 with only `frac` of its 3 fraction bits kept (truncation of the raw byte).
__global__ void fill8(__hip_fp8_e4m3* x, size_t n, uint64_t seed, int frac) {
  const uint8_t mask = (uint8_t)~((1u << (3 - frac)) - 1u);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    __hip_fp8_e4m3 v((float)uniform_pm1(seed, i));
    v.__x &= mask;
    if (g_low_mode == 1) v.__x |= (uint8_t)~mask;
    else if (g_low_mode == 2) v.__x |= (uint8_t)(0x5 & ~mask);
    x[i] = v;
  }
}

// Elementwise relative error |c - ref| / |ref| at ns evenly spaced entries (-1 where ref == 0).
__global__ void relerr_sample(const double* c, const double* ref, size_t stride, size_t ns, double* out) {
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < ns; j += (size_t)gridDim.x * blockDim.x) {
    const double r = ref[j * stride];
    out[j] = r == 0 ? -1.0 : fabs(c[j * stride] - r) / fabs(r);
  }
}

// Accuracy check: sum (c - ref)^2 and sum ref^2 with one atomic per block.
template <typename T>
__device__ __forceinline__ double as_double(T v) { return (double)v; }
template <>
__device__ __forceinline__ double as_double<__hip_bfloat16>(__hip_bfloat16 v) { return (double)__bfloat162float(v); }
template <typename T>
__global__ void err_kernel(const T* c, const double* ref, size_t n, double* acc) {
  __shared__ double sd[256], sr[256];
  double d = 0, r = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const double e = as_double(c[i]) - ref[i];
    d += e * e;
    r += ref[i] * ref[i];
  }
  sd[threadIdx.x] = d;
  sr[threadIdx.x] = r;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { sd[threadIdx.x] += sd[threadIdx.x + s]; sr[threadIdx.x] += sr[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { atomicAdd(acc, sd[0]); atomicAdd(acc + 1, sr[0]); }
}

static double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  std::string dtype = "fp64", tag = "run";
  int frac = -1, round = 0, only_a = 0, only_b = 0, low_mode = 0, check = 0, device = 0, ta = 0, tb = 0, spread_mode = 0;
  double spread = 0;
  long m = 32768, n = 32768, k = 2048;
  // FP64: bit flips between consecutive k (tog) and between neighbouring rows of A / columns of B (tox), set bits (hw)
  double tog_a = -1, tog_b = -1, hw_a = -1, hw_b = -1, tox_a = -1, tox_b = -1;
  double warmup_s = 10, measure_s = 30;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
    if (!strcmp(argv[i], "--dtype")) dtype = next();
    else if (!strcmp(argv[i], "--frac")) frac = atoi(next());
    else if (!strcmp(argv[i], "--round")) round = !strcmp(next(), "nearest");
    else if (!strcmp(argv[i], "--m")) m = atol(next());
    else if (!strcmp(argv[i], "--n")) n = atol(next());
    else if (!strcmp(argv[i], "--k")) k = atol(next());
    else if (!strcmp(argv[i], "--warmup")) warmup_s = atof(next());
    else if (!strcmp(argv[i], "--seconds")) measure_s = atof(next());
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--only-a")) only_a = 1;  // B keeps the native mantissa
    else if (!strcmp(argv[i], "--only-b")) only_b = 1;  // A keeps it
    else if (!strcmp(argv[i], "--ta")) ta = 1;  // FP64 only: A stored k x m, op(A) = A^T
    else if (!strcmp(argv[i], "--tb")) tb = 1;  // FP64 only: B stored n x k, op(B) = B^T
    else if (!strcmp(argv[i], "--check")) check = 1;    // relative error vs FP64 on the unreduced data
    else if (!strcmp(argv[i], "--spread")) spread = atof(next());  // exponent spread in binades (see g_spread)
    else if (!strcmp(argv[i], "--spread-mode")) spread_mode = !strcmp(next(), "row");  // elem | row
    else if (!strcmp(argv[i], "--low")) {  // fill of the dropped bits: zero | ones | const | krep | mrep | sorted | shuffled
      const char* s = next();
      low_mode = !strcmp(s, "ones") ? 1 : !strcmp(s, "const") ? 2 : !strcmp(s, "krep") ? 3
               : !strcmp(s, "sorted") ? 4 : !strcmp(s, "shuffled") ? 5 : !strcmp(s, "mrep") ? 6 : 0;
    }
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  const bool is64 = dtype == "fp64", is32 = dtype == "fp32", is8 = dtype == "fp8";
  if (!is64 && !is32 && !is8 && dtype != "bf16") {
    fprintf(stderr, "--dtype must be fp64, fp32, bf16 or fp8\n");
    return 2;
  }
  const int native = is64 ? 52 : is32 ? 23 : is8 ? 3 : 7;
  if (frac < 0 || frac > native) frac = native;

  CK(hipSetDevice(device));
  char bdf[32];
  CK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  if (spread != 0 && (!is64 || ta || tb)) { fprintf(stderr, "--spread needs --dtype fp64 without --ta/--tb\n"); return 2; }
  if (low_mode >= 3 && (!is64 || ta || tb || spread != 0 || check || (k & (k - 1)))) {
    fprintf(stderr, "--low krep|mrep|sorted|shuffled needs --dtype fp64, a power-of-two k, no --ta/--tb/--spread/--check\n");
    return 2;
  }
  CK(hipMemcpyToSymbol(HIP_SYMBOL(g_kdim), &k, sizeof(long)));
  CK(hipMemcpyToSymbol(HIP_SYMBOL(g_low_mode), &low_mode, sizeof(int)));
  CK(hipMemcpyToSymbol(HIP_SYMBOL(g_spread), &spread, sizeof(double)));
  CK(hipMemcpyToSymbol(HIP_SYMBOL(g_spread_mode), &spread_mode, sizeof(int)));
  hipblasHandle_t h;
  CB(hipblasCreate(&h));
  const size_t esz = is64 ? 8 : is32 ? 4 : is8 ? 1 : 2, csz = is64 ? 8 : is8 ? 2 : 4;
  void *A, *B, *C;
  CK(hipMalloc(&A, (size_t)m * k * esz));
  CK(hipMalloc(&B, (size_t)k * n * esz));
  CK(hipMalloc(&C, (size_t)m * n * csz));
  CK(hipMemset(C, 0, (size_t)m * n * csz));
  if (is64) {
    fill64<<<4096, 256>>>((double*)A, (size_t)m * k, 1, only_b ? native : frac, round, m, 0);
    fill64<<<4096, 256>>>((double*)B, (size_t)k * n, 2, only_a ? native : frac, round, k, 1);
    if (!ta && !tb) {
      unsigned long long* acc;
      CK(hipMalloc(&acc, 12 * sizeof(unsigned long long)));
      CK(hipMemset(acc, 0, 12 * sizeof(unsigned long long)));
      bitstats64<<<4096, 256>>>((double*)A, (size_t)m * k, m, 0, acc);
      bitstats64<<<4096, 256>>>((double*)B, (size_t)k * n, k, 1, acc + 6);
      unsigned long long h[12];
      CK(hipMemcpy(h, acc, sizeof(h), hipMemcpyDeviceToHost));
      CK(hipFree(acc));
      tog_a = (double)h[0] / h[2]; hw_a = (double)h[1] / h[3]; tox_a = (double)h[4] / h[5];
      tog_b = (double)h[6] / h[8]; hw_b = (double)h[7] / h[9]; tox_b = (double)h[10] / h[11];
    }
  } else if (is32) {
    fill32<<<4096, 256>>>((float*)A, (size_t)m * k, 1, only_b ? native : frac, round);
    fill32<<<4096, 256>>>((float*)B, (size_t)k * n, 2, only_a ? native : frac, round);
  } else if (is8) {
    fill8<<<4096, 256>>>((__hip_fp8_e4m3*)A, (size_t)m * k, 1, only_b ? native : frac);
    fill8<<<4096, 256>>>((__hip_fp8_e4m3*)B, (size_t)k * n, 2, only_a ? native : frac);
  } else {
    fill16<<<4096, 256>>>((__hip_bfloat16*)A, (size_t)m * k, 1, only_b ? native : frac, round);
    fill16<<<4096, 256>>>((__hip_bfloat16*)B, (size_t)k * n, 2, only_a ? native : frac, round);
  }
  CK(hipDeviceSynchronize());

  const double one = 1.0, zero = 0.0;
  const float onef = 1.0f, zerof = 0.0f;
  // FP8: D[m,n] = A^T B with A stored k x m and B stored k x n.
  hipblasLtHandle_t lt = nullptr;
  hipblasLtMatmulDesc_t op = nullptr;
  hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  hipblasLtMatmulHeuristicResult_t heur = {};
  void* ws = nullptr;
  const size_t ws_size = 64ull << 20;
  if (is8) {
    CB(hipblasLtCreate(&lt));
    CB(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t t = HIPBLAS_OP_T, nn = HIPBLAS_OP_N;
    CB(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CB(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn)));
    CB(hipblasLtMatrixLayoutCreate(&la, HIP_R_8F_E4M3, k, m, k));
    CB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_8F_E4M3, k, n, k));
    CB(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, m, n, m));
    CK(hipMalloc(&ws, ws_size));
    hipblasLtMatmulPreference_t pref;
    CB(hipblasLtMatmulPreferenceCreate(&pref));
    CB(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
    int found = 0;
    CB(hipblasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &found));
    if (!found) { fprintf(stderr, "no FP8 algorithm\n"); return 1; }
  }
  auto gemm = [&]() {
    if (is8)
      CB(hipblasLtMatmul(lt, op, &onef, A, la, B, lb, &zerof, C, lc, C, lc, &heur.algo, ws, ws_size, 0));
    else if (is64)
      CB(hipblasDgemm(h, ta ? HIPBLAS_OP_T : HIPBLAS_OP_N, tb ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one,
                      (double*)A, ta ? k : m, (double*)B, tb ? n : k, &zero, (double*)C, m));
    else if (is32)
      CB(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &onef, (float*)A, m, (float*)B, k, &zerof, (float*)C, m));
    else
      CB(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &onef, A, HIP_R_16BF, m, B, HIP_R_16BF, k,
                       &zerof, C, HIP_R_32F, m, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  };

  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  const double flop = 2.0 * m * n * k;

  // Warm up until the clock has settled, then time fixed-size chunks.
  double t = now_unix();
  while (now_unix() - t < warmup_s) { gemm(); CK(hipDeviceSynchronize()); }

  std::vector<double> rates;
  double busy_ms = 0;
  long calls = 0;
  const double t_begin = now_unix();
  while (now_unix() - t_begin < measure_s) {
    CK(hipEventRecord(e0));
    for (int r = 0; r < 5; ++r) gemm();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    busy_ms += ms;
    calls += 5;
    rates.push_back(5 * flop / (ms * 1e-3) / 1e12);
  }
  const double t_end = now_unix();
  // Accuracy of this path: FP64 GEMM on the original, unreduced values in the same layout.
  double rel_err = -1, err_p50 = -1, err_p99 = -1;
  if (check) {
    double *Ar, *Br, *Cr, *acc;
    CK(hipMalloc(&Ar, (size_t)m * k * 8));
    CK(hipMalloc(&Br, (size_t)k * n * 8));
    CK(hipMalloc(&Cr, (size_t)m * n * 8));
    CK(hipMalloc(&acc, 2 * sizeof(double)));
    CK(hipMemset(acc, 0, 2 * sizeof(double)));
    const int zero_mode = 0;  // the reference always holds the random low bits
    CK(hipMemcpyToSymbol(HIP_SYMBOL(g_low_mode), &zero_mode, sizeof(int)));
    fill64<<<4096, 256>>>(Ar, (size_t)m * k, 1, 52, 0, m, 0);
    fill64<<<4096, 256>>>(Br, (size_t)k * n, 2, 52, 0, k, 1);
    // The reference uses the strided-batched entry point (batch 1) so an interposer on hipblasDgemm /
    // rocblas_dgemm does not also reduce the reference.
    if (is8)  // D = A^T B with A stored k x m
      CB(hipblasDgemmStridedBatched(h, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &one, Ar, k, 0, Br, k, 0, &zero, Cr, m, 0, 1));
    else
      CB(hipblasDgemmStridedBatched(h, ta ? HIPBLAS_OP_T : HIPBLAS_OP_N, tb ? HIPBLAS_OP_T : HIPBLAS_OP_N, m, n, k, &one,
                                    Ar, ta ? k : m, 0, Br, tb ? n : k, 0, &zero, Cr, m, 0, 1));
    const size_t nn = (size_t)m * n;
    if (is64) err_kernel<double><<<2048, 256>>>((const double*)C, Cr, nn, acc);
    else if (is8) err_kernel<__hip_bfloat16><<<2048, 256>>>((const __hip_bfloat16*)C, Cr, nn, acc);
    else err_kernel<float><<<2048, 256>>>((const float*)C, Cr, nn, acc);
    double hacc[2];
    CK(hipMemcpy(hacc, acc, sizeof(hacc), hipMemcpyDeviceToHost));
    rel_err = sqrt(hacc[0] / hacc[1]);
    // elementwise relative error on up to 2^20 evenly spaced entries (inf/nan count as inf): the Frobenius error
    // is dominated by the largest entries, while wide spreads hurt the small ones
    if (is64) {
      const size_t ns = nn < (1u << 20) ? nn : (1u << 20), stride = nn / ns;
      double* es;
      CK(hipMalloc(&es, ns * sizeof(double)));
      relerr_sample<<<1024, 256>>>((const double*)C, Cr, stride, ns, es);
      std::vector<double> hs(ns), v;
      CK(hipMemcpy(hs.data(), es, ns * sizeof(double), hipMemcpyDeviceToHost));
      CK(hipFree(es));
      for (double x : hs) if (!(x < 0)) v.push_back(std::isnan(x) ? INFINITY : x);
      if (!v.empty()) {
        std::sort(v.begin(), v.end());
        err_p50 = v[v.size() / 2];
        err_p99 = v[(size_t)(v.size() * 0.99)];
      }
    }
    CK(hipFree(Ar)); CK(hipFree(Br)); CK(hipFree(Cr)); CK(hipFree(acc));
  }
  double lo = rates[0], hi = rates[0];
  for (double r : rates) { lo = r < lo ? r : lo; hi = r > hi ? r : hi; }
  printf("RESULT tag=%s gpu=%d bdf=%s dtype=%s frac=%d only_a=%d only_b=%d ta=%d tb=%d low=%s round=%s m=%ld n=%ld k=%ld calls=%ld tflops=%.3f chunk_min=%.3f chunk_max=%.3f t_begin=%.3f t_end=%.3f rel_err=%.3e spread=%g smode=%s err_p50=%.3e err_p99=%.3e tog_a=%.3f tog_b=%.3f hw_a=%.3f hw_b=%.3f tox_a=%.3f tox_b=%.3f\n",
         tag.c_str(), device, bdf, dtype.c_str(), frac, only_a, only_b, ta, tb,
         kLowNames[low_mode], round ? "nearest" : "trunc", m, n, k, calls,
         calls * flop / (busy_ms * 1e-3) / 1e12, lo, hi, t_begin, t_end, rel_err, spread, spread_mode ? "row" : "elem",
         err_p50, err_p99, tog_a, tog_b, hw_a, hw_b, tox_a, tox_b);
  CB(hipblasDestroy(h));
  return 0;
}
