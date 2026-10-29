// Non-GEMM FP64 kernels under a reduced input mantissa: HIP port of gh200/bench/fp64_kernel_bench.cu
// (M4 of the GH200 study) for the MI350 study; adds --device and prints the device's PCI BDF.
//
// Ozaki-type emulation only replaces dense GEMM; these kernels run on the FP64
// vector ALUs (V_FMA_F64 on MI350X, DFMA on Hopper), where no emulation applies.  Kernels:
//   dfma  - register-blocked SIMT outer-product loop (8x8 per thread): isolates
//           the DFMA pipe with GEMM-like operand traffic but no tensor cores.
//   nbody - all-pairs gravitational acceleration (softened, FP64 rsqrt): a real
//           compute-bound kernel whose intermediates (r^2, 1/r^3) regain full
//           mantissa even when the inputs are reduced.  --inner k also reduces
//           dx, dy, dz and 1/r^3 inside the loop (in-code truncation).
//   copy / triad - STREAM-style data movement (TB/s reported in the tflops field).
// Command line and RESULT line follow gemm_mant_bench so e1_gemm_tdp.sbatch and
// analysis/e1_load.py work unchanged; `tflops` counts 2 per DFMA for dfma and
// 20 flops per interaction for nbody (the usual convention).
#include <hip/hip_runtime.h>

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

__constant__ int g_low_mode = 0;  // 0 zero, 1 ones, 2 fixed pattern (as in gemm_mant_bench)
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

__device__ __forceinline__ double reduce64(double v, int frac, int round) {
  if (frac >= 52) return v;
  uint64_t u = __double_as_longlong(v);
  const uint64_t drop = 52 - frac;
  if (round) u += 1ull << (drop - 1);
  const uint64_t low = (1ull << drop) - 1ull;
  u &= ~low;
  if (g_low_mode == 1) u |= low;
  else if (g_low_mode == 2) u |= kConstPattern & low;
  return __longlong_as_double(u);
}

// In-loop truncation (plain mask, no rounding): what an application would add to its kernel.
__device__ __forceinline__ double mask64(double v, uint64_t keep) {
  return __longlong_as_double(__double_as_longlong(v) & keep);
}

__global__ void fill64(double* x, size_t n, uint64_t seed, int frac, int round) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = reduce64(uniform_pm1(seed, i), frac, round);
}

// Bodies: positions uniform in [-1, 1)^3, masses in [0.5, 1.5) (w component).
__global__ void fill_bodies(double4* p, int n, int frac, int round) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    p[i] = make_double4(reduce64(uniform_pm1(11, 4 * (size_t)i), frac, round),
                        reduce64(uniform_pm1(11, 4 * (size_t)i + 1), frac, round),
                        reduce64(uniform_pm1(11, 4 * (size_t)i + 2), frac, round),
                        reduce64(1.0 + 0.5 * uniform_pm1(11, 4 * (size_t)i + 3), frac, round));
}

constexpr int kThreads = 128, kRA = 8, kRB = 8, kKS = 4;  // 32 KB static smem per block

// A: per block kKS x (kRA/2) x kThreads double2 (conflict-free), B: per block kKS x kRB (broadcast).
__global__ void __launch_bounds__(kThreads) dfma_kernel(const double* __restrict__ A, const double* __restrict__ B,
                                                        double* __restrict__ out, int reps) {
  __shared__ double2 sa[kKS * (kRA / 2) * kThreads];
  __shared__ double sb[kKS * kRB];
  const int t = threadIdx.x;
  const double2* ga = reinterpret_cast<const double2*>(A) + (size_t)blockIdx.x * kKS * (kRA / 2) * kThreads;
  for (int i = t; i < kKS * (kRA / 2) * kThreads; i += kThreads) sa[i] = ga[i];
  for (int i = t; i < kKS * kRB; i += kThreads) sb[i] = B[(size_t)blockIdx.x * kKS * kRB + i];
  __syncthreads();
  double acc[kRA][kRB];
#pragma unroll
  for (int i = 0; i < kRA; ++i)
#pragma unroll
    for (int j = 0; j < kRB; ++j) acc[i][j] = 0;
  for (int rep = 0; rep < reps; ++rep) {
#pragma unroll 2
    for (int k = 0; k < kKS; ++k) {
      double a[kRA], b[kRB];
#pragma unroll
      for (int q = 0; q < kRA / 2; ++q) {
        const double2 v = sa[(k * (kRA / 2) + q) * kThreads + t];
        a[2 * q] = v.x;
        a[2 * q + 1] = v.y;
      }
#pragma unroll
      for (int j = 0; j < kRB; ++j) b[j] = sb[k * kRB + j];
#pragma unroll
      for (int i = 0; i < kRA; ++i)
#pragma unroll
        for (int j = 0; j < kRB; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
    }
  }
  double s = 0;
#pragma unroll
  for (int i = 0; i < kRA; ++i)
#pragma unroll
    for (int j = 0; j < kRB; ++j) s += acc[i][j];
  out[(size_t)blockIdx.x * kThreads + t] = s;
}

constexpr int kNbThreads = 256;

template <bool kInner>
__global__ void __launch_bounds__(kNbThreads) nbody_kernel(const double4* __restrict__ p, double* __restrict__ acc,
                                                           int n, double eps2, uint64_t keep) {
  __shared__ double4 sp[kNbThreads];
  const int i = blockIdx.x * kNbThreads + threadIdx.x;
  const double4 pi = p[i];
  double ax = 0, ay = 0, az = 0;
  for (int tile = 0; tile < n; tile += kNbThreads) {
    sp[threadIdx.x] = p[tile + threadIdx.x];
    __syncthreads();
#pragma unroll 8
    for (int j = 0; j < kNbThreads; ++j) {
      const double4 pj = sp[j];
      double dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
      if (kInner) { dx = mask64(dx, keep); dy = mask64(dy, keep); dz = mask64(dz, keep); }
      const double r2 = fma(dx, dx, fma(dy, dy, fma(dz, dz, eps2)));
      const double inv = rsqrt(r2);
      double s = inv * inv * inv * pj.w;
      if (kInner) s = mask64(s, keep);
      ax = fma(dx, s, ax);
      ay = fma(dy, s, ay);
      az = fma(dz, s, az);
    }
    __syncthreads();
  }
  acc[3 * (size_t)i] = ax;
  acc[3 * (size_t)i + 1] = ay;
  acc[3 * (size_t)i + 2] = az;
}

// Memory-bound data movement (STREAM style): copy a = b, triad a = b + 3 c.  The inputs carry the
// reduced mantissa; isolates HBM / on-chip data-path toggling from arithmetic.
__global__ void copy_kernel(double* __restrict__ a, const double* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] = b[i];
}
__global__ void triad_kernel(double* __restrict__ a, const double* __restrict__ b, const double* __restrict__ c, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    a[i] = b[i] + 3.0 * c[i];
}

__global__ void err_kernel(const double* c, const double* ref, size_t n, double* out) {
  __shared__ double sd[256], sr[256];
  double d = 0, r = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const double e = c[i] - ref[i];
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
  if (threadIdx.x == 0) { atomicAdd(out, sd[0]); atomicAdd(out + 1, sr[0]); }
}

static double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  std::string dtype = "dfma", tag = "run";
  int frac = 52, round = 0, low_mode = 0, check = 0, inner = 52, device = 0;
  long nbodies = 131072, reps = 80000;
  double warmup_s = 10, measure_s = 30;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
    if (!strcmp(argv[i], "--dtype")) dtype = next();
    else if (!strcmp(argv[i], "--frac")) frac = atoi(next());
    else if (!strcmp(argv[i], "--round")) round = !strcmp(next(), "nearest");
    else if (!strcmp(argv[i], "--inner")) inner = atoi(next());   // nbody: bits kept inside the loop
    else if (!strcmp(argv[i], "--n")) nbodies = atol(next());
    else if (!strcmp(argv[i], "--reps")) reps = atol(next());
    else if (!strcmp(argv[i], "--warmup")) warmup_s = atof(next());
    else if (!strcmp(argv[i], "--seconds")) measure_s = atof(next());
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else if (!strcmp(argv[i], "--check")) check = 1;
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--low")) {
      const char* m = next();
      low_mode = !strcmp(m, "ones") ? 1 : !strcmp(m, "const") ? 2 : 0;
    }
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  const bool mem = dtype == "copy" || dtype == "triad";
  if (dtype != "dfma" && dtype != "nbody" && !mem) { fprintf(stderr, "--dtype must be dfma, nbody, copy or triad\n"); return 2; }
  if (frac < 0 || frac > 52) frac = 52;
  if (inner < 0 || inner > 52) inner = 52;
  const bool nb = dtype == "nbody";
  if (nb && nbodies % kNbThreads) { fprintf(stderr, "--n must be a multiple of %d\n", kNbThreads); return 2; }
  CK(hipSetDevice(device));
  char bdf[32];
  CK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  CK(hipMemcpyToSymbol(HIP_SYMBOL(g_low_mode), &low_mode, sizeof(int)));
  int sms = 0;
  CK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, device));

  const int blocks = sms * 3;  // dfma: 3 blocks x 4 warps per SM
  const size_t na = (size_t)blocks * kKS * kRA * kThreads, nb_ = (size_t)blocks * kKS * kRB;
  double *A = nullptr, *B = nullptr, *out = nullptr;
  double4* P = nullptr;
  const uint64_t keep = inner >= 52 ? ~0ull : ~((1ull << (52 - inner)) - 1ull);
  const double eps2 = 1e-4;
  const size_t nmem = (size_t)1 << 28;  // 2 GiB per array
  double *Ma = nullptr, *Mb = nullptr, *Mc = nullptr;
  if (mem) {
    CK(hipMalloc(&Ma, nmem * 8));
    CK(hipMalloc(&Mb, nmem * 8));
    CK(hipMalloc(&Mc, nmem * 8));
    fill64<<<1024, 256>>>(Ma, nmem, 3, 52, 0);
    fill64<<<1024, 256>>>(Mb, nmem, 4, frac, round);
    fill64<<<1024, 256>>>(Mc, nmem, 5, frac, round);
  } else if (nb) {
    CK(hipMalloc(&P, nbodies * sizeof(double4)));
    CK(hipMalloc(&out, 3 * nbodies * sizeof(double)));
    fill_bodies<<<1024, 256>>>(P, (int)nbodies, frac, round);
  } else {
    CK(hipMalloc(&A, na * 8));
    CK(hipMalloc(&B, nb_ * 8));
    CK(hipMalloc(&out, (size_t)blocks * kThreads * 8));
    fill64<<<1024, 256>>>(A, na, 1, frac, round);
    fill64<<<1024, 256>>>(B, nb_, 2, frac, round);
  }
  CK(hipDeviceSynchronize());
  auto launch = [&]() {
    if (dtype == "copy") copy_kernel<<<sms * 8, 512>>>(Ma, Mb, nmem);
    else if (dtype == "triad") triad_kernel<<<sms * 8, 512>>>(Ma, Mb, Mc, nmem);
    else if (!nb) dfma_kernel<<<blocks, kThreads>>>(A, B, out, (int)reps);
    else if (inner < 52) nbody_kernel<true><<<nbodies / kNbThreads, kNbThreads>>>(P, out, (int)nbodies, eps2, keep);
    else nbody_kernel<false><<<nbodies / kNbThreads, kNbThreads>>>(P, out, (int)nbodies, eps2, keep);
  };
  // for copy / triad the "tflops" field carries HBM traffic in TB/s (16 or 24 bytes per element)
  const double flop = mem ? (dtype == "copy" ? 16.0 : 24.0) * nmem
                    : nb ? 20.0 * nbodies * nbodies : 2.0 * blocks * kThreads * (double)reps * kKS * kRA * kRB;

  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  double t = now_unix();
  while (now_unix() - t < warmup_s) { launch(); CK(hipDeviceSynchronize()); }
  CK(hipGetLastError());
  std::vector<double> rates;
  double busy_ms = 0;
  long calls = 0;
  const double t_begin = now_unix();
  while (now_unix() - t_begin < measure_s) {
    CK(hipEventRecord(e0));
    for (int r = 0; r < 5; ++r) launch();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    busy_ms += ms;
    calls += 5;
    rates.push_back(5 * flop / (ms * 1e-3) / 1e12);
  }
  const double t_end = now_unix();

  // nbody accuracy: accelerations vs the same kernel on the unreduced bodies.
  double rel_err = -1;
  if (check && nb) {
    double *ref, *e;
    double4* Pr;
    CK(hipMalloc(&Pr, nbodies * sizeof(double4)));
    CK(hipMalloc(&ref, 3 * nbodies * sizeof(double)));
    CK(hipMalloc(&e, 2 * sizeof(double)));
    CK(hipMemset(e, 0, 2 * sizeof(double)));
    const int zero = 0;
    CK(hipMemcpyToSymbol(HIP_SYMBOL(g_low_mode), &zero, sizeof(int)));
    fill_bodies<<<1024, 256>>>(Pr, (int)nbodies, 52, 0);
    nbody_kernel<false><<<nbodies / kNbThreads, kNbThreads>>>(Pr, ref, (int)nbodies, eps2, ~0ull);
    err_kernel<<<1024, 256>>>(out, ref, 3 * nbodies, e);
    double h[2];
    CK(hipMemcpy(h, e, sizeof(h), hipMemcpyDeviceToHost));
    rel_err = sqrt(h[0] / h[1]);
    (void)hipFree(Pr); (void)hipFree(ref); (void)hipFree(e);
  }
  double lo = rates[0], hi = rates[0];
  for (double r : rates) { lo = r < lo ? r : lo; hi = r > hi ? r : hi; }
  printf("RESULT tag=%s gpu=%d bdf=%s dtype=%s frac=%d only_a=0 low=%s round=%s m=%ld n=%d k=%ld calls=%ld tflops=%.3f chunk_min=%.3f chunk_max=%.3f t_begin=%.3f t_end=%.3f rel_err=%.3e emu_bits=-1\n",
         tag.c_str(), device, bdf, dtype.c_str(), frac, low_mode == 1 ? "ones" : low_mode == 2 ? "const" : "zero",
         round ? "nearest" : "trunc", nb ? nbodies : (long)blocks * kThreads, nb ? inner : kRA * kRB, nb ? nbodies : reps,
         calls, calls * flop / (busy_ms * 1e-3) / 1e12, lo, hi, t_begin, t_end, rel_err);
  return 0;
}
