// GEMM throughput under a reduced operand mantissa (E1 of the GH200 study).
//
// A and B are filled with uniform random values in [-1, 1), then the low
// fraction bits are cleared (or rounded) so only `--frac` explicit fraction
// bits remain.  FP64 uses cublasDgemm; BF16 uses cublasGemmEx with BF16
// inputs, FP32 output and FP32 accumulation (the HPL-MxP trailing update).
// FP8 (E4M3) uses cublasLt in the TN layout Hopper requires, FP32 accumulation
// and BF16 output; `--frac` then keeps 0..3 of the E4M3 fraction bits.
// The measured window's wall-clock bounds are printed as Unix times so an
// external nvidia-smi power log can be joined to it.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS %s:%d %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

// What the dropped low fraction bits are filled with (mechanism control):
// 0 = zero (truncation), 1 = all ones, 2 = one fixed nonzero pattern shared by
// every element. 1 and 2 remove bit toggling without making the bits zero.
__constant__ int g_low_mode = 0;
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

// Exponent spread (range experiment): value = uniform[-1, 1) * 2^e with e ~ U(-S/2, S/2), i.e. log-uniform
// magnitudes over S binades. Mode 0 draws e per element, so every row and column spans S binades; mode 1
// draws one e per row of A and per column of B, so each row/column is narrow (what per-row scaled
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

// Keep `frac` of the 52 FP64 fraction bits. round=1 adds half a kept ulp first.
__device__ __forceinline__ double reduce64(double v, int frac, int round) {
  if (frac >= 52) return v;
  uint64_t u;
  memcpy(&u, &v, 8);
  const uint64_t drop = 52 - frac;
  if (round) u += 1ull << (drop - 1);
  const uint64_t low = (1ull << drop) - 1ull;
  u &= ~low;
  if (g_low_mode == 1) u |= low;
  else if (g_low_mode == 2) u |= kConstPattern & low;
  memcpy(&v, &u, 8);
  return v;
}

__device__ __forceinline__ float reduce32(float v, int frac, int round) {
  if (frac >= 23) return v;
  uint32_t u;
  memcpy(&u, &v, 4);
  const uint32_t drop = 23 - frac;
  if (round) u += 1u << (drop - 1);
  u &= ~((1u << drop) - 1u);
  memcpy(&v, &u, 4);
  return v;
}

__global__ void fill64(double* x, size_t n, uint64_t seed, int frac, int round, size_t ld, int col_key) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = reduce64(uniform_pm1(seed, i) * spread_scale(seed, i, ld, col_key), frac, round);
}

// frac <= 7 keeps the value exactly representable in BF16, so the final
// conversion does not round again.
__global__ void fill16(__nv_bfloat16* x, size_t n, uint64_t seed, int frac, int round, size_t ld, int col_key) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = (float)(uniform_pm1(seed, i) * spread_scale(seed, i, ld, col_key));
    if (frac < 7) {
      v = reduce32(v, frac, round);
      if (g_low_mode) {
        uint32_t u;
        memcpy(&u, &v, 4);
        const uint32_t low = ((1u << (23 - frac)) - 1u) & 0xFFFF0000u;  // dropped BF16 bits
        u |= (g_low_mode == 1 ? 0xFFFFFFFFu : (uint32_t)kConstPattern) & low;
        memcpy(&v, &u, 4);
      }
    }
    x[i] = __float2bfloat16_rn(v);
  }
}

// FP32 operands (for the fp32/tf32 paths) with `frac` of 23 fraction bits kept. With a spread, values
// outside FP32's range become inf or (sub)normal/zero, as a plain conversion would.
__global__ void fill32(float* x, size_t n, uint64_t seed, int frac, int round, size_t ld, int col_key) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = reduce32((float)(uniform_pm1(seed, i) * spread_scale(seed, i, ld, col_key)), frac, round);
}

// Accuracy check: sum (c - ref)^2 and sum ref^2 with one atomic per block.
template <typename T>
__device__ __forceinline__ double as_double(T v) { return (double)v; }
template <>
__device__ __forceinline__ double as_double<__nv_bfloat16>(__nv_bfloat16 v) { return (double)__bfloat162float(v); }
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

// Elementwise relative error |c - ref| / |ref| at ns evenly spaced entries (-1 where ref == 0). The
// Frobenius error above is dominated by the largest entries; with a wide exponent spread the small
// entries are where per-row scaled emulation and short formats lose accuracy.
template <typename T>
__global__ void relerr_sample(const T* c, const double* ref, size_t stride, size_t ns, double* out) {
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < ns; j += (size_t)gridDim.x * blockDim.x) {
    const double r = ref[j * stride];
    out[j] = r == 0 ? -1.0 : fabs(as_double(c[j * stride]) - r) / fabs(r);
  }
}

// E4M3 with only `frac` of its 3 fraction bits kept (truncation of the raw byte).
__global__ void fill8(__nv_fp8_e4m3* x, size_t n, uint64_t seed, int frac) {
  const uint8_t mask = (uint8_t)~((1u << (3 - frac)) - 1u);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    __nv_fp8_e4m3 v((float)uniform_pm1(seed, i));
    v.__x &= mask;
    if (g_low_mode == 1) v.__x |= (uint8_t)~mask;
    else if (g_low_mode == 2) v.__x |= (uint8_t)(0x5 & ~mask);
    x[i] = v;
  }
}

#define CL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLASLt %s:%d %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  std::string dtype = "fp64", tag = "run";
  int frac = -1, round = 0, only_a = 0, only_b = 0, low_mode = 0, check = 0, spread_mode = 0;
  double spread = 0;
  long m = 32768, n = 32768, k = 2048;
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
    else if (!strcmp(argv[i], "--only-a")) only_a = 1;  // B keeps the native mantissa
    else if (!strcmp(argv[i], "--only-b")) only_b = 1;  // A keeps it: B = weights in the inference shapes
    else if (!strcmp(argv[i], "--check")) check = 1;    // relative error vs FP64 on the unreduced data
    else if (!strcmp(argv[i], "--spread")) spread = atof(next());  // exponent spread in binades (see g_spread)
    else if (!strcmp(argv[i], "--spread-mode")) spread_mode = !strcmp(next(), "row");  // elem | row
    else if (!strcmp(argv[i], "--low")) {  // fill of the dropped bits: zero | ones | const
      const char* m = next();
      low_mode = !strcmp(m, "ones") ? 1 : !strcmp(m, "const") ? 2 : 0;
    }
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  // fp64emu: cuBLAS FP64 fixed-point (Ozaki-type) emulation on integer tensor cores,
  //          --frac = maximum retained mantissa bits; operands keep all 52 bits.
  // fp64emud: the same emulation with dynamic mantissa control: cuBLAS picks the number of bits at run
  //          time for FP64-equivalent accuracy (--frac = the maximum it may use; reported as emu_bits).
  // fp32:    SGEMM on FP32 cores; tf32: SGEMM on TF32 tensor cores (10-bit mantissa inputs).
  const bool emud = dtype == "fp64emud", emu = dtype == "fp64emu" || emud, is32 = dtype == "fp32" || dtype == "tf32";
  const bool is64 = dtype == "fp64" || emu, is8 = dtype == "fp8";
  if (!is64 && !is8 && !is32 && dtype != "bf16") {
    fprintf(stderr, "--dtype must be fp64, fp64emu, fp64emud, fp32, tf32, bf16 or fp8\n");
    return 2;
  }
  if (is8 && spread != 0) {
    fprintf(stderr, "--spread is not supported for fp8 (it would need per-tensor scaling)\n");
    return 2;
  }
  const int native = emu ? 53 : is64 ? 52 : is32 ? 23 : is8 ? 3 : 7;
  if (frac < 0 || (frac > native && !emud)) frac = native;

  CK(cudaMemcpyToSymbol(g_low_mode, &low_mode, sizeof(int)));
  CK(cudaMemcpyToSymbol(g_spread, &spread, sizeof(double)));
  CK(cudaMemcpyToSymbol(g_spread_mode, &spread_mode, sizeof(int)));
  cublasHandle_t h;
  CB(cublasCreate(&h));
  const size_t esz = is64 ? 8 : is32 ? 4 : is8 ? 1 : 2, csz = is64 ? 8 : is8 ? 2 : 4;
  int* emu_bits_dev = nullptr;
  if (emu) {
    CB(cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    CB(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER));
    CB(cublasSetFixedPointEmulationMantissaControl(h, emud ? CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
                                                            : CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    CB(cublasSetFixedPointEmulationMaxMantissaBitCount(h, frac));
    CK(cudaMalloc(&emu_bits_dev, sizeof(int)));
    CK(cudaMemset(emu_bits_dev, 0xff, sizeof(int)));
    CB(cublasSetFixedPointEmulationMantissaBitCountPointer(h, emu_bits_dev));
  }
  if (dtype == "tf32") CB(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
  void *A, *B, *C;
  CK(cudaMalloc(&A, (size_t)m * k * esz));
  CK(cudaMalloc(&B, (size_t)k * n * esz));
  CK(cudaMalloc(&C, (size_t)m * n * csz));
  CK(cudaMemset(C, 0, (size_t)m * n * csz));
  if (is64) {
    const int f = emu ? 52 : frac;  // emulation reduces precision itself
    fill64<<<4096, 256>>>((double*)A, (size_t)m * k, 1, only_b && !emu ? 52 : f, round, m, 0);
    fill64<<<4096, 256>>>((double*)B, (size_t)k * n, 2, only_a || emu ? 52 : f, round, k, 1);
  } else if (is32) {
    fill32<<<4096, 256>>>((float*)A, (size_t)m * k, 1, only_b ? native : frac, round, m, 0);
    fill32<<<4096, 256>>>((float*)B, (size_t)k * n, 2, only_a ? native : frac, round, k, 1);
  } else if (is8) {
    fill8<<<4096, 256>>>((__nv_fp8_e4m3*)A, (size_t)m * k, 1, only_b ? native : frac);
    fill8<<<4096, 256>>>((__nv_fp8_e4m3*)B, (size_t)k * n, 2, only_a ? native : frac);
  } else {
    fill16<<<4096, 256>>>((__nv_bfloat16*)A, (size_t)m * k, 1, only_b ? native : frac, round, m, 0);
    fill16<<<4096, 256>>>((__nv_bfloat16*)B, (size_t)k * n, 2, only_a ? native : frac, round, k, 1);
  }
  CK(cudaDeviceSynchronize());

  const double one = 1.0, zero = 0.0;
  const float onef = 1.0f, zerof = 0.0f;
  // FP8: D[m,n] = A^T B with A stored k x m and B stored k x n.
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulHeuristicResult_t heur = {};
  void* ws = nullptr;
  const size_t ws_size = 64ull << 20;
  if (is8) {
    CL(cublasLtCreate(&lt));
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, nn = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn)));
    CL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, m, k));
    CL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, n, k));
    CL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, m, n, m));
    CK(cudaMalloc(&ws, ws_size));
    cublasLtMatmulPreference_t pref;
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
    int found = 0;
    CL(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &found));
    if (!found) { fprintf(stderr, "no FP8 algorithm\n"); return 1; }
  }
  auto gemm = [&]() {
    if (is8)
      CL(cublasLtMatmul(lt, op, &onef, A, la, B, lb, &zerof, C, lc, C, lc, &heur.algo, ws, ws_size, 0));
    else if (is64)
      CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, (double*)A, m, (double*)B, k, &zero, (double*)C, m));
    else if (is32)
      CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &onef, (float*)A, m, (float*)B, k, &zerof, (float*)C, m));
    else
      CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &onef, A, CUDA_R_16BF, m, B, CUDA_R_16BF, k,
                      &zerof, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const double flop = 2.0 * m * n * k;

  // Warm up until the clock has settled, then time fixed-size chunks.
  double t = now_unix();
  while (now_unix() - t < warmup_s) { gemm(); CK(cudaDeviceSynchronize()); }

  std::vector<double> rates;
  double busy_ms = 0;
  long calls = 0;
  const double t_begin = now_unix();
  while (now_unix() - t_begin < measure_s) {
    CK(cudaEventRecord(e0));
    for (int r = 0; r < 5; ++r) gemm();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    busy_ms += ms;
    calls += 5;
    rates.push_back(5 * flop / (ms * 1e-3) / 1e12);
  }
  const double t_end = now_unix();
  // Accuracy of this path: FP64 GEMM on the original, unreduced values in the same layout.
  double rel_err = -1, err_p50 = -1, err_p99 = -1;
  int emu_bits_used = -2;
  if (emu) CK(cudaMemcpy(&emu_bits_used, emu_bits_dev, sizeof(int), cudaMemcpyDeviceToHost));
  if (check) {
    double *Ar, *Br, *Cr, *acc;
    CK(cudaMalloc(&Ar, (size_t)m * k * 8));
    CK(cudaMalloc(&Br, (size_t)k * n * 8));
    CK(cudaMalloc(&Cr, (size_t)m * n * 8));
    CK(cudaMalloc(&acc, 2 * sizeof(double)));
    CK(cudaMemset(acc, 0, 2 * sizeof(double)));
    const int keep = low_mode;  // the reference always holds the random low bits
    CK(cudaMemcpyToSymbol(g_low_mode, &(low_mode = 0), sizeof(int)));
    fill64<<<4096, 256>>>(Ar, (size_t)m * k, 1, 52, 0, is8 ? k : m, is8 ? 1 : 0);
    fill64<<<4096, 256>>>(Br, (size_t)k * n, 2, 52, 0, k, 1);
    low_mode = keep;
    cublasHandle_t hr;
    CB(cublasCreate(&hr));
    // The reference uses the strided-batched entry point (batch 1): interposers such as the GEMMul8
    // hook or libmantshim intercept cublasDgemm, so a plain cublasDgemm here would be emulated too.
    if (is8)  // D = A^T B with A stored k x m
      CB(cublasDgemmStridedBatched(hr, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, Ar, k, 0, Br, k, 0, &zero, Cr, m, 0, 1));
    else
      CB(cublasDgemmStridedBatched(hr, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Ar, m, 0, Br, k, 0, &zero, Cr, m, 0, 1));
    const size_t nn = (size_t)m * n;
    if (is64) err_kernel<double><<<2048, 256>>>((const double*)C, Cr, nn, acc);
    else if (is8) err_kernel<__nv_bfloat16><<<2048, 256>>>((const __nv_bfloat16*)C, Cr, nn, acc);
    else err_kernel<float><<<2048, 256>>>((const float*)C, Cr, nn, acc);
    double hacc[2];
    CK(cudaMemcpy(hacc, acc, sizeof(hacc), cudaMemcpyDeviceToHost));
    rel_err = sqrt(hacc[0] / hacc[1]);
    // elementwise relative error on up to 2^20 evenly spaced entries (inf/nan count as inf)
    const size_t ns = nn < (1u << 20) ? nn : (1u << 20), stride = nn / ns;
    double* es;
    CK(cudaMalloc(&es, ns * sizeof(double)));
    if (is64) relerr_sample<double><<<1024, 256>>>((const double*)C, Cr, stride, ns, es);
    else if (is8) relerr_sample<__nv_bfloat16><<<1024, 256>>>((const __nv_bfloat16*)C, Cr, stride, ns, es);
    else relerr_sample<float><<<1024, 256>>>((const float*)C, Cr, stride, ns, es);
    std::vector<double> hs(ns);
    CK(cudaMemcpy(hs.data(), es, ns * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(es);
    std::vector<double> v;
    v.reserve(ns);
    for (double x : hs) if (!(x < 0)) v.push_back(std::isnan(x) ? INFINITY : x);
    if (!v.empty()) {
      std::sort(v.begin(), v.end());
      err_p50 = v[v.size() / 2];
      err_p99 = v[(size_t)(v.size() * 0.99)];
    }
    cublasDestroy(hr);
    cudaFree(Ar); cudaFree(Br); cudaFree(Cr); cudaFree(acc);
  }
  double lo = rates[0], hi = rates[0];
  for (double r : rates) { lo = r < lo ? r : lo; hi = r > hi ? r : hi; }
  printf("RESULT tag=%s dtype=%s frac=%d only_a=%d only_b=%d low=%s round=%s m=%ld n=%ld k=%ld calls=%ld tflops=%.3f chunk_min=%.3f chunk_max=%.3f t_begin=%.3f t_end=%.3f rel_err=%.3e emu_bits=%d spread=%g smode=%s err_p50=%.3e err_p99=%.3e\n",
         tag.c_str(), dtype.c_str(), frac, only_a, only_b, low_mode == 1 ? "ones" : low_mode == 2 ? "const" : "zero", round ? "nearest" : "trunc", m, n, k, calls,
         calls * flop / (busy_ms * 1e-3) / 1e12, lo, hi, t_begin, t_end, rel_err, emu_bits_used,
         spread, spread_mode ? "row" : "elem", err_p50, err_p99);
  cublasDestroy(h);
  return 0;
}
