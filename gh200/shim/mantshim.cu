// mantshim: LD_PRELOAD interposer for FP64 GEMM (cuBLAS and CPU BLAS).
//
// For every intercepted DGEMM/ZGEMM it can
//   * record operand statistics: a histogram of binary exponents (how much of
//     FP64's *range* the data uses) and of the number of significant fraction
//     bits, i.e. 52 minus trailing zero bits (how much of FP64's *precision*
//     the data already uses), plus call counts and flops;
//   * reduce the operand mantissas: A and B are copied to scratch buffers with
//     only MANT_BITS fraction bits kept, and the real GEMM runs on the copies,
//     so the caller's data is never modified.
//
// Environment:
//   MANT_BITS=<0..52>          fraction bits kept (default 52 = unchanged)
//   MANT_ROUND=trunc|nearest   how the dropped bits are removed (default trunc)
//   MANT_MIN_MNK=<n>           only touch GEMMs with m*n*k >= n (default 0)
//   MANT_STATS=<path>          write statistics as JSON to <path>.<pid>.json at exit
//   MANT_STATS_SAMPLE=<n>      max elements sampled per operand per call (default 1<<20)
//   MANT_VERBOSE=1             print one line per intercepted call to stderr
//   MANT_FORCE_COPY=1          take the copy path even when nothing is dropped (bits=52),
//                              so a baseline carries the same copy overhead as a reduced run
//   MANT_SPAN=<path>           per GEMM shape, count the nonzero elements of sampled rows of op(A) and
//                              columns of op(B) that lie more than 24/32/40/48/56 binades below their
//                              row's (column's) maximum: what per-row scaled INT8 emulation loses
//                              (MANT_SPAN_LINES lines per operand per call, MANT_SPAN_MIN_MNK)
//   MANT_EMU_BITS=<k>          instead of reducing, run each FP64 GEMM with cuBLAS fixed-point
//                              (Ozaki-type, integer tensor core) emulation retaining at most k
//                              mantissa bits; no copies are made
//   MANT_EMU_DYN=1             with MANT_EMU_BITS: dynamic mantissa control, i.e. cuBLAS picks the bit
//                              count per call from the operands' exponent spans for FP64-equivalent
//                              accuracy and runs native FP64 when that needs more than k bits
//   MANT_EMU_LOG=1             read back the bit count cuBLAS used for every emulated GEMM (synchronizes
//                              after each call) and print a histogram (calls, flops) at exit
//   mantshim_set_bits(k)       exported C function: change MANT_BITS at run time (e.g. from a
//                              solver callback through ctypes); -1 returns the current value
//   MANT_REDUCE_64=1           also reduce the 64-bit-index entry points (cublasDgemm_v2_64,
//                              cublasZgemm_v2_64). These are what cuSOLVER calls internally
//                              (e.g. inside eigensolvers), so by default they are only counted.
//   MANT_F32=fp32|tf32|tf32x3  instead of reducing, run each FP64 (real or complex) GEMM in single
//                              precision: A and B are rounded to FP32 (tf32: also to TF32; tf32x3:
//                              split into big = tf32(x) and small = tf32(x - big) and summed as
//                              big*small + small*big + big*big, dropping small*small), multiplied
//                              with FP32 accumulation, and C = alpha*C32 + beta*C is formed in FP64.
//                              The equal-accuracy alternative a reviewer asked about.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kExpBins = 2048;  // raw 11-bit exponent field
constexpr int kSigBins = 53;    // 0..52 significant fraction bits

struct Config {
  int bits = 52;
  bool nearest = false;
  double min_mnk = 0;
  const char* stats = nullptr;
  long sample = 1 << 20;
  bool verbose = false;
  bool reduce64 = false;
  bool force_copy = false;
  int emu_bits = -1;
  bool emu_dyn = false;         // MANT_EMU_DYN
  bool emu_log = false;         // MANT_EMU_LOG
  int f32 = 0;                  // MANT_F32: 1 = fp32, 2 = tf32, 3 = tf32x3
  const char* dump = nullptr;   // MANT_DUMP=<dir>: write the operands of the first MANT_DUMP_N GEMMs
  long dump_n = 4;
  double dump_min = 1e9;        // MANT_DUMP_MIN_MNK: only GEMMs with m*n*k at least this
  const char* span = nullptr;   // MANT_SPAN=<path>: per-row exponent span per GEMM shape -> <path>.<pid>.txt
  double span_min = 1e8;        // MANT_SPAN_MIN_MNK
  long span_lines = 256;        // MANT_SPAN_LINES: rows / columns sampled per operand per call
  Config() {
    span = getenv("MANT_SPAN");
    if (const char* e = getenv("MANT_SPAN_MIN_MNK")) span_min = atof(e);
    if (const char* e = getenv("MANT_SPAN_LINES")) span_lines = atol(e);
    dump = getenv("MANT_DUMP");
    if (const char* e = getenv("MANT_DUMP_N")) dump_n = atol(e);
    if (const char* e = getenv("MANT_DUMP_MIN_MNK")) dump_min = atof(e);
    if (const char* e = getenv("MANT_EMU_BITS")) emu_bits = atoi(e);
    if (const char* e = getenv("MANT_EMU_DYN")) emu_dyn = atoi(e) != 0;
    if (const char* e = getenv("MANT_EMU_LOG")) emu_log = atoi(e) != 0;
    if (const char* e = getenv("MANT_F32"))
      f32 = !strcmp(e, "fp32") ? 1 : !strcmp(e, "tf32") ? 2 : !strcmp(e, "tf32x3") ? 3 : 0;
    if (const char* e = getenv("MANT_FORCE_COPY")) force_copy = atoi(e) != 0;
    if (const char* e = getenv("MANT_REDUCE_64")) reduce64 = atoi(e) != 0;
    if (const char* e = getenv("MANT_BITS")) bits = atoi(e);
    if (bits < 0) bits = 0;
    if (bits > 52) bits = 52;
    if (const char* e = getenv("MANT_ROUND")) nearest = !strcmp(e, "nearest");
    if (const char* e = getenv("MANT_MIN_MNK")) min_mnk = atof(e);
    stats = getenv("MANT_STATS");
    if (const char* e = getenv("MANT_STATS_SAMPLE")) sample = atol(e);
    if (const char* e = getenv("MANT_VERBOSE")) verbose = atoi(e) != 0;
  }
};
Config& cfg_mut() {
  static Config c;
  return c;
}
const Config& cfg() { return cfg_mut(); }

// MANT_EMU_LOG: cuBLAS writes the bit count of each emulated GEMM to emu_bits_dev; it is read back when the
// next emulated GEMM is set up (or at exit) and accumulated per bit count as (calls, flops).
int* emu_bits_dev = nullptr;
cudaStream_t emu_stream = nullptr;
double emu_pending_flop = 0;
std::map<int, std::pair<long, double>> emu_hist;

void emu_collect() {
  if (!emu_bits_dev || emu_pending_flop == 0) return;
  if (cudaStreamSynchronize(emu_stream) != cudaSuccess) {
    cudaGetLastError();
    cudaDeviceSynchronize();
  }
  int b = -1;
  cudaMemcpy(&b, emu_bits_dev, sizeof(int), cudaMemcpyDeviceToHost);
  auto& e = emu_hist[b];
  e.first += 1;
  e.second += emu_pending_flop;
  emu_pending_flop = 0;
}

void write_emu() {
  if (!cfg().emu_log) return;
  emu_collect();
  double tot = 0;
  for (auto& kv : emu_hist) tot += kv.second.second;
  for (auto& kv : emu_hist)
    fprintf(stderr, "[mantshim] emu bits=%d calls=%ld gflop=%.1f flop_share=%.4f\n", kv.first, kv.second.first,
            kv.second.second / 1e9, kv.second.second / std::max(tot, 1.0));
}

// Put the handle into fixed-point FP64 emulation with at most `bits` mantissa bits (MANT_EMU_DYN: as many as
// cuBLAS finds necessary, native FP64 above `bits`). `flop` and `s` are only used by MANT_EMU_LOG.
void set_emulation(cublasHandle_t h, int bits, double flop, cudaStream_t s) {
  cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
  cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER);
  cublasSetFixedPointEmulationMantissaControl(
      h, cfg().emu_dyn ? CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC : CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
  cublasSetFixedPointEmulationMaxMantissaBitCount(h, bits);
  if (cfg().emu_log) {
    emu_collect();
    if (!emu_bits_dev) cudaMalloc(&emu_bits_dev, sizeof(int));
    cudaMemsetAsync(emu_bits_dev, 0xff, sizeof(int), s);
    cublasSetFixedPointEmulationMantissaBitCountPointer(h, emu_bits_dev);
    emu_stream = s;
    emu_pending_flop = flop;
  }
}

// ---- statistics ----------------------------------------------------------
struct HostHist {
  std::vector<unsigned long long> exp = std::vector<unsigned long long>(kExpBins);
  std::vector<unsigned long long> sig = std::vector<unsigned long long>(kSigBins);
  unsigned long long zeros = 0, nonfinite = 0;
};

struct Stats {
  std::mutex mu;
  std::map<std::string, unsigned long long> calls;
  std::map<std::string, double> flops;
  std::map<std::string, double> flops_reduced;
  HostHist host_a, host_b;  // CPU BLAS operands
  // device histograms, one set per device: [A exp | A sig | B exp | B sig | A zero,nonfinite | B zero,nonfinite]
  std::map<int, unsigned long long*> dev_hist;
  bool atexit_registered = false;
};
Stats& st() {
  static Stats s;
  return s;
}
constexpr int kDevHistLen = 2 * (kExpBins + kSigBins) + 4;

__device__ __forceinline__ void classify(double v, unsigned long long* exp_h, unsigned long long* sig_h,
                                         unsigned long long* zero_nf) {
  uint64_t u = __double_as_longlong(v);
  const int e = (int)((u >> 52) & 0x7ff);
  const uint64_t frac = u & 0xfffffffffffffull;
  if (e == 0x7ff) { atomicAdd(zero_nf + 1, 1ull); return; }
  if (e == 0 && frac == 0) { atomicAdd(zero_nf, 1ull); return; }
  atomicAdd(exp_h + e, 1ull);
  const int sig = frac ? 52 - __ffsll((long long)frac) + 1 : 0;
  atomicAdd(sig_h + sig, 1ull);
}

// Sample `total` elements of a column-major rows x cols matrix (element = `per` doubles).
__global__ void hist_kernel(const double* x, long rows, long cols, long ld, int per, long stride,
                            unsigned long long* exp_h, unsigned long long* sig_h, unsigned long long* zero_nf) {
  const long total = rows * cols;
  for (long s = blockIdx.x * (long)blockDim.x + threadIdx.x; s * stride < total; s += (long)gridDim.x * blockDim.x) {
    const long idx = s * stride, c = idx / rows, r = idx % rows;
    for (int p = 0; p < per; ++p) classify(x[(c * ld + r) * per + p], exp_h, sig_h, zero_nf);
  }
}

__device__ __forceinline__ double reduce64(double v, int bits, bool nearest) {
  uint64_t u = __double_as_longlong(v);
  if (((u >> 52) & 0x7ff) == 0x7ff) return v;
  const int drop = 52 - bits;
  if (drop == 0) return v;
  if (nearest) u += 1ull << (drop - 1);
  u &= ~((1ull << drop) - 1ull);
  return __longlong_as_double(u);
}

// Copy a strided column-major matrix into a compact one, reducing each double.
// grid.y walks columns, grid.x the (rows*per) doubles of a column: no divisions.
__global__ void reduce_copy(const double* __restrict__ src, long rows, long cols, long ld, int per,
                            double* __restrict__ dst, int bits, bool nearest) {
  const long len = rows * per, sld = ld * per;
  for (long c = blockIdx.y; c < cols; c += gridDim.y) {
    const double* s = src + c * sld;
    double* d = dst + c * len;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < len; i += (long)gridDim.x * blockDim.x)
      d[i] = reduce64(s[i], bits, nearest);
  }
}

void launch_reduce(const double* src, long rows, long cols, long ld, int per, double* dst, cudaStream_t st) {
  const long len = rows * per;
  dim3 grid((unsigned)std::min<long>((len + 255) / 256, 64), (unsigned)std::min<long>(cols, 8192));
  reduce_copy<<<grid, 256, 0, st>>>(src, rows, cols, ld, per, dst, cfg().bits, cfg().nearest);
}

// ---- per-row exponent span (MANT_SPAN) -------------------------------------
// Per-row scaled emulation (Ozaki-type: cuBLAS fixed-point, GEMMul8) normalises every row of op(A) and
// every column of op(B) (both run along k) by its largest element, so an element e binades below that
// maximum keeps only about (bits - e) bits and is lost once e exceeds the retained bits. For sampled lines
// we count the nonzero elements more than T binades below their line's maximum, per GEMM shape.
constexpr int kSpanT[] = {24, 32, 40, 48, 56};   // = 24 + 8 t (the device code uses that form)
constexpr int kNSpan = sizeof(kSpanT) / sizeof(int);
constexpr int kSpanSlot = 2 + kNSpan;  // lines, nonzeros, count above each threshold
constexpr int kSpanClasses = 512;       // distinct shapes tracked; the last class collects the rest

__global__ void span_kernel(const double* x, long len, long line_stride, long elem_stride, int per, long nlines,
                            long line_step, unsigned long long* out) {
  __shared__ int smax[256];
  __shared__ unsigned long long scnt[256][kNSpan + 1];
  for (long ln = blockIdx.x; ln * line_step < nlines; ln += gridDim.x) {
    const long base = ln * line_step * line_stride;
    int mx = -100000;
    for (long e = threadIdx.x; e < len; e += blockDim.x)
      for (int p = 0; p < per; ++p) {
        const double v = x[(base + e * elem_stride) * per + p];
        if (v != 0 && isfinite(v)) mx = max(mx, ilogb(v));
      }
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + s]);
      __syncthreads();
    }
    mx = smax[0];
    unsigned long long c[kNSpan + 1] = {};
    for (long e = threadIdx.x; e < len; e += blockDim.x)
      for (int p = 0; p < per; ++p) {
        const double v = x[(base + e * elem_stride) * per + p];
        if (v == 0 || !isfinite(v)) continue;
        c[0]++;
        const int below = mx - ilogb(v);
        for (int t = 0; t < kNSpan; ++t) c[t + 1] += below > 24 + 8 * t;   // = kSpanT[t]
      }
    for (int t = 0; t <= kNSpan; ++t) scnt[threadIdx.x][t] = c[t];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s)
        for (int t = 0; t <= kNSpan; ++t) scnt[threadIdx.x][t] += scnt[threadIdx.x + s][t];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(out, 1ull);
      for (int t = 0; t <= kNSpan; ++t) atomicAdd(out + 1 + t, scnt[0][t]);
    }
    __syncthreads();
  }
}

struct SpanClass { long m, n, k; int ta, tb, per; unsigned long long calls = 0; double flops = 0; };
std::mutex g_span_mu;
std::vector<SpanClass> g_span_cls;
std::map<std::tuple<long, long, long, int, int, int>, int> g_span_idx;
std::map<int, unsigned long long*> g_span_dev;  // per device: kSpanClasses x 2 operands x kSpanSlot

void span_record(bool ta, bool tb, long m, long n, long k, const void* A, long lda, const void* B, long ldb, int per,
                 cudaStream_t s) {
  if (!cfg().span || (double)m * n * k < cfg().span_min || m <= 0 || n <= 0 || k <= 0) return;
  int dev = 0;
  cudaGetDevice(&dev);
  int ci;
  unsigned long long* buf;
  {
    std::lock_guard<std::mutex> g(g_span_mu);
    auto key = std::make_tuple(m, n, k, (int)ta, (int)tb, per);
    auto it = g_span_idx.find(key);
    if (it != g_span_idx.end()) ci = it->second;
    else if ((int)g_span_cls.size() < kSpanClasses - 1) {
      ci = (int)g_span_cls.size();
      g_span_idx[key] = ci;
      g_span_cls.push_back(SpanClass{m, n, k, (int)ta, (int)tb, per});
    } else {
      if ((int)g_span_cls.size() < kSpanClasses) g_span_cls.push_back(SpanClass{-1, -1, -1, 0, 0, per});
      ci = kSpanClasses - 1;
    }
    g_span_cls[ci].calls++;
    g_span_cls[ci].flops += 2.0 * m * n * k * (per == 2 ? 4 : 1);
    auto d = g_span_dev.find(dev);
    if (d == g_span_dev.end()) {
      unsigned long long* p = nullptr;
      cudaMalloc(&p, (size_t)kSpanClasses * 2 * kSpanSlot * 8);
      cudaMemset(p, 0, (size_t)kSpanClasses * 2 * kSpanSlot * 8);
      d = g_span_dev.emplace(dev, p).first;
    }
    buf = d->second + (size_t)ci * 2 * kSpanSlot;
  }
  const long lines = cfg().span_lines;
  // op(A): m lines (rows) of length k; op(B): n lines (columns) of length k
  const long a_ls = ta ? lda : 1, a_es = ta ? 1 : lda, b_ls = tb ? 1 : ldb, b_es = tb ? ldb : 1;
  const long a_step = std::max(1L, m / lines), b_step = std::max(1L, n / lines);
  span_kernel<<<(unsigned)std::min(m, lines), 256, 0, s>>>((const double*)A, k, a_ls, a_es, per, m, a_step, buf);
  span_kernel<<<(unsigned)std::min(n, lines), 256, 0, s>>>((const double*)B, k, b_ls, b_es, per, n, b_step,
                                                          buf + kSpanSlot);
}

void write_span() {
  if (!cfg().span) return;
  std::lock_guard<std::mutex> g(g_span_mu);
  std::vector<unsigned long long> tot((size_t)kSpanClasses * 2 * kSpanSlot, 0);
  for (auto& kv : g_span_dev) {
    std::vector<unsigned long long> tmp(tot.size());
    cudaSetDevice(kv.first);
    cudaDeviceSynchronize();
    if (cudaMemcpy(tmp.data(), kv.second, tmp.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess)
      for (size_t i = 0; i < tot.size(); ++i) tot[i] += tmp[i];
  }
  std::string path = std::string(cfg().span) + "." + std::to_string(getpid()) + ".txt";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;
  fprintf(f, "# per GEMM shape (m=-1: all further shapes); op A = rows of op(A), op B = columns of op(B), along k\n");
  fprintf(f, "# op m n k ta tb per calls gflop lines nonzero");
  for (int t : kSpanT) fprintf(f, " below%d", t);
  fprintf(f, "\n");
  for (size_t c = 0; c < g_span_cls.size(); ++c)
    for (int o = 0; o < 2; ++o) {
      const unsigned long long* v = &tot[(c * 2 + o) * kSpanSlot];
      const SpanClass& sc = g_span_cls[c];
      fprintf(f, "%s %ld %ld %ld %d %d %d %llu %.3f", o ? "B" : "A", sc.m, sc.n, sc.k, sc.ta, sc.tb, sc.per, sc.calls,
              sc.flops / 1e9);
      for (int t = 0; t < kSpanSlot; ++t) fprintf(f, " %llu", v[t]);
      fprintf(f, "\n");
    }
  fclose(f);
}

void write_stats();
void write_all() {
  write_stats();
  write_span();
  write_emu();
}
void register_atexit() {
  // Registered lazily (after the CUDA runtime has initialised), so our handler
  // runs before the runtime tears its context down.
  Stats& s = st();
  if (!s.atexit_registered) {
    s.atexit_registered = true;
    atexit(write_all);
  }
}

unsigned long long* device_hist() {
  int dev = 0;
  cudaGetDevice(&dev);
  Stats& s = st();
  auto it = s.dev_hist.find(dev);
  if (it != s.dev_hist.end()) return it->second;
  unsigned long long* h = nullptr;
  cudaMalloc(&h, kDevHistLen * sizeof(unsigned long long));
  cudaMemset(h, 0, kDevHistLen * sizeof(unsigned long long));
  s.dev_hist[dev] = h;
  return h;
}

void dev_record(const void* x, long rows, long cols, long ld, int per, bool is_b, cudaStream_t stream) {
  if (!cfg().stats || rows <= 0 || cols <= 0) return;
  unsigned long long* h = device_hist();
  const long total = rows * cols;
  const long stride = total > cfg().sample ? (total + cfg().sample - 1) / cfg().sample : 1;
  unsigned long long* base = h + (is_b ? kExpBins + kSigBins : 0);
  unsigned long long* zn = h + 2 * (kExpBins + kSigBins) + (is_b ? 2 : 0);
  hist_kernel<<<256, 256, 0, stream>>>((const double*)x, rows, cols, ld, per, stride, base, base + kExpBins, zn);
}

void host_record(const double* x, long rows, long cols, long ld, int per, HostHist& hh) {
  const long total = rows * cols;
  const long stride = total > cfg().sample ? (total + cfg().sample - 1) / cfg().sample : 1;
  for (long idx = 0; idx < total; idx += stride) {
    const long c = idx / rows, r = idx % rows;
    for (int p = 0; p < per; ++p) {
      uint64_t u;
      memcpy(&u, &x[(c * ld + r) * per + p], 8);
      const int e = (int)((u >> 52) & 0x7ff);
      const uint64_t frac = u & 0xfffffffffffffull;
      if (e == 0x7ff) { hh.nonfinite++; continue; }
      if (e == 0 && frac == 0) { hh.zeros++; continue; }
      hh.exp[e]++;
      hh.sig[frac ? 52 - __builtin_ctzll(frac) : 0]++;
    }
  }
}

void count(const char* name, double flops, bool reduced) {
  std::lock_guard<std::mutex> g(st().mu);
  st().calls[name]++;
  st().flops[name] += flops;
  if (reduced) st().flops_reduced[name] += flops;
}

void dump_hist(FILE* f, const char* key, const unsigned long long* v, int n) {
  fprintf(f, "\"%s\":[", key);
  for (int i = 0; i < n; ++i) fprintf(f, "%s%llu", i ? "," : "", v[i]);
  fprintf(f, "]");
}

void write_stats() {
  if (!cfg().stats) return;
  Stats& s = st();
  std::lock_guard<std::mutex> g(s.mu);
  std::string path = std::string(cfg().stats) + "." + std::to_string(getpid()) + ".json";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;
  fprintf(f, "{\"mant_bits\":%d,\"round\":\"%s\",\"calls\":{", cfg().bits, cfg().nearest ? "nearest" : "trunc");
  bool first = true;
  for (auto& kv : s.calls) { fprintf(f, "%s\"%s\":%llu", first ? "" : ",", kv.first.c_str(), kv.second); first = false; }
  fprintf(f, "},\"flops\":{");
  first = true;
  for (auto& kv : s.flops) { fprintf(f, "%s\"%s\":%.6e", first ? "" : ",", kv.first.c_str(), kv.second); first = false; }
  fprintf(f, "},\"flops_reduced\":{");
  first = true;
  for (auto& kv : s.flops_reduced) { fprintf(f, "%s\"%s\":%.6e", first ? "" : ",", kv.first.c_str(), kv.second); first = false; }
  fprintf(f, "}");
  // Sum device histograms with the host ones.
  std::vector<unsigned long long> tot(kDevHistLen, 0);
  for (auto& kv : s.dev_hist) {
    std::vector<unsigned long long> tmp(kDevHistLen);
    cudaSetDevice(kv.first);
    cudaDeviceSynchronize();
    if (cudaMemcpy(tmp.data(), kv.second, kDevHistLen * 8, cudaMemcpyDeviceToHost) == cudaSuccess)
      for (int i = 0; i < kDevHistLen; ++i) tot[i] += tmp[i];
  }
  for (int i = 0; i < kExpBins; ++i) tot[i] += s.host_a.exp[i], tot[kExpBins + kSigBins + i] += s.host_b.exp[i];
  for (int i = 0; i < kSigBins; ++i)
    tot[kExpBins + i] += s.host_a.sig[i], tot[2 * kExpBins + kSigBins + i] += s.host_b.sig[i];
  const int z = 2 * (kExpBins + kSigBins);
  tot[z] += s.host_a.zeros, tot[z + 1] += s.host_a.nonfinite, tot[z + 2] += s.host_b.zeros, tot[z + 3] += s.host_b.nonfinite;
  fprintf(f, ",");
  dump_hist(f, "a_exp", tot.data(), kExpBins);
  fprintf(f, ",");
  dump_hist(f, "a_sig", tot.data() + kExpBins, kSigBins);
  fprintf(f, ",");
  dump_hist(f, "b_exp", tot.data() + kExpBins + kSigBins, kExpBins);
  fprintf(f, ",");
  dump_hist(f, "b_sig", tot.data() + 2 * kExpBins + kSigBins, kSigBins);
  fprintf(f, ",\"a_zero\":%llu,\"a_nonfinite\":%llu,\"b_zero\":%llu,\"b_nonfinite\":%llu}\n", tot[z], tot[z + 1],
          tot[z + 2], tot[z + 3]);
  fclose(f);
}

template <typename F>
F real(const char* name) {
  void* p = dlsym(RTLD_NEXT, name);
  if (!p) {
    fprintf(stderr, "[mantshim] cannot resolve %s\n", name);
    abort();
  }
  return reinterpret_cast<F>(p);
}

bool want_reduce(double mnk) { return (cfg().bits < 52 || cfg().force_copy) && mnk >= cfg().min_mnk; }
thread_local bool g_stats_only = false;  // set around the _64 entry points unless MANT_REDUCE_64

// Stored shape of op(X): rows x cols in the caller's column-major layout.
inline void stored(bool trans, long r, long c, long& rows, long& cols) {
  rows = trans ? c : r;
  cols = trans ? r : c;
}

// One grow-only scratch buffer per stream. Reuse is safe because every use is
// stream-ordered: the next GEMM on the same stream runs after the previous one.
struct StreamBuf { void* p = nullptr; size_t size = 0; };
std::mutex g_buf_mu;
std::map<cudaStream_t, StreamBuf> g_bufs;

void* stream_scratch(cudaStream_t s, size_t need) {
  std::lock_guard<std::mutex> g(g_buf_mu);
  StreamBuf& b = g_bufs[s];
  if (b.size < need) {
    if (b.p) cudaFreeAsync(b.p, s);
    b.p = nullptr;
    b.size = 0;
    const size_t want = need + need / 4;
    if (cudaMallocAsync(&b.p, want, s) != cudaSuccess) return nullptr;
    b.size = want;
  }
  return b.p;
}

struct Scratch {
  void* a = nullptr;
  void* b = nullptr;
  cudaStream_t s = nullptr;
};

// MANT_DUMP: copy the (first batch entry of the) original operands of a few large GEMMs to host
// files, column-major and compact (rows x cols x per doubles), plus one index line per GEMM.
std::mutex g_dump_mu;
long g_dumped = 0;
void dump_one(const char* path, const void* x, long rows, long cols, long ld, int per, cudaStream_t s) {
  std::vector<double> h((size_t)rows * cols * per);
  cudaMemcpy2DAsync(h.data(), rows * per * 8, x, ld * per * 8, rows * per * 8, cols, cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  if (FILE* f = fopen(path, "wb")) { fwrite(h.data(), 8, h.size(), f); fclose(f); }
}
void dump_operands(bool ta, bool tb, long m, long n, long k, const void* A, long ar, long ac, long lda,
                   const void* B, long br, long bc, long ldb, int per, cudaStream_t s) {
  if (!cfg().dump || (double)m * n * k < cfg().dump_min) return;
  std::lock_guard<std::mutex> g(g_dump_mu);
  if (g_dumped >= cfg().dump_n) return;
  const long i = g_dumped++;
  char pa[4096], pb[4096], pi[4096];
  snprintf(pa, sizeof pa, "%s/gemm%ld_A.f64", cfg().dump, i);
  snprintf(pb, sizeof pb, "%s/gemm%ld_B.f64", cfg().dump, i);
  snprintf(pi, sizeof pi, "%s/index.txt", cfg().dump);
  dump_one(pa, A, ar, ac, lda, per, s);
  dump_one(pb, B, br, bc, ldb, per, s);
  if (FILE* f = fopen(pi, "a")) {
    fprintf(f, "%ld per=%d ta=%d tb=%d m=%ld n=%ld k=%ld Arows=%ld Acols=%ld Brows=%ld Bcols=%ld\n", i, per, ta, tb,
            m, n, k, ar, ac, br, bc);
    fclose(f);
  }
}

// Record statistics and, if requested, build reduced compact copies of A and B.
// Returns true when `sa`/`sb` hold the reduced operands to use instead.
bool prepare_device(cublasHandle_t h, bool ta, bool tb, long m, long n, long k, const void* A, long lda,
                    const void* B, long ldb, int per, long batch, long sA, long sB, Scratch& sc, long& ldA,
                    long& ldB) {
  register_atexit();
  cublasGetStream(h, &sc.s);
  if (cfg().verbose)  // every entry point passes through here: log the GEMM shape once per call
    fprintf(stderr, "[mantshim] gemm per=%d m=%ld n=%ld k=%ld batch=%ld stats_only=%d\n", per, m, n, k, batch,
            (int)g_stats_only);
  long ar, ac, br, bc;
  stored(ta, m, k, ar, ac);
  stored(tb, k, n, br, bc);
  dump_operands(ta, tb, m, n, k, A, ar, ac, lda, B, br, bc, ldb, per, sc.s);
  span_record(ta, tb, m, n, k, A, lda, B, ldb, per, sc.s);   // first batch entry only
  for (long i = 0; i < batch; ++i) {
    dev_record((const double*)A + i * sA * per, ar, ac, lda, per, false, sc.s);
    dev_record((const double*)B + i * sB * per, br, bc, ldb, per, true, sc.s);
  }
  if (cfg().emu_bits > 0 && !g_stats_only) {
    set_emulation(h, cfg().emu_bits, 2.0 * m * n * k * batch * (per == 2 ? 4 : 1), sc.s);
    return false;
  }
  if (g_stats_only || !want_reduce((double)m * n * k)) return false;
  const size_t asz = (size_t)ar * ac * per * 8 * batch, bsz = (size_t)br * bc * per * 8 * batch;
  void* buf = stream_scratch(sc.s, asz + bsz);
  if (!buf) {
    fprintf(stderr, "[mantshim] scratch allocation failed; GEMM left unreduced\n");
    return false;
  }
  sc.a = buf;
  sc.b = (char*)buf + asz;
  for (long i = 0; i < batch; ++i) {
    launch_reduce((const double*)A + i * sA * per, ar, ac, lda, per, (double*)sc.a + i * ar * ac * per, sc.s);
    launch_reduce((const double*)B + i * sB * per, br, bc, ldb, per, (double*)sc.b + i * br * bc * per, sc.s);
  }
  ldA = ar;
  ldB = br;
  return true;
}

bool is_t(cublasOperation_t op) { return op != CUBLAS_OP_N; }

// ---- MANT_F32: single-precision replacement of an FP64 GEMM -----------------
__device__ __forceinline__ float tf32_round(float x) {  // nearest-even to 10 fraction bits
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u += 0xfffu + ((u >> 13) & 1u);
  return __uint_as_float(u & ~0x1fffu);
}

// Strided column-major doubles -> compact floats: big (FP32 or TF32) and, if `small`, tf32(x - big).
__global__ void split_copy(const double* __restrict__ src, long rows, long cols, long ld, int per,
                           float* __restrict__ big, float* __restrict__ small, bool tf32) {
  const long len = rows * per, sld = ld * per;
  for (long c = blockIdx.y; c < cols; c += gridDim.y) {
    const double* s = src + c * sld;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < len; i += (long)gridDim.x * blockDim.x) {
      const float f = (float)s[i];
      const float b = tf32 ? tf32_round(f) : f;
      big[c * len + i] = b;
      if (small) small[c * len + i] = tf32_round(f - b);
    }
  }
}

// C = alpha * C32 + beta * C for an m x n (x per) result; C32 is compact.
__global__ void combine(const float* __restrict__ c32, long m, long n, int per, double2 alpha, double2 beta,
                        bool beta_zero, double* __restrict__ C, long ldc) {
  for (long c = blockIdx.y; c < n; c += gridDim.y)
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
      double* d = C + (c * ldc + i) * per;
      const float* s = c32 + (c * m + i) * per;
      if (per == 1) {
        d[0] = alpha.x * s[0] + (beta_zero ? 0.0 : beta.x * d[0]);
      } else {
        const double sr = s[0], si = s[1];
        double r = alpha.x * sr - alpha.y * si, im = alpha.x * si + alpha.y * sr;
        if (!beta_zero) {
          const double dr = d[0], di = d[1];
          r += beta.x * dr - beta.y * di;
          im += beta.x * di + beta.y * dr;
        }
        d[0] = r;
        d[1] = im;
      }
    }
}

dim3 grid_for(long len, long cols) {
  return dim3((unsigned)std::min<long>((len + 255) / 256, 64), (unsigned)std::min<long>(cols, 8192));
}

// Runs one FP64 GEMM (per = 1 real, 2 complex) through MANT_F32. Returns false when the mode is off.
bool f32_gemm(const char* name, cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, long m, long n, long k,
              const void* alpha, const void* A, long lda, const void* B, long ldb, const void* beta, void* C, long ldc,
              int per, cublasStatus_t& status) {
  const int mode = cfg().f32;
  if (!mode || g_stats_only) return false;
  using GemmEx = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
                                    const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*,
                                    cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
  static GemmEx gemm = real<GemmEx>("cublasGemmEx");  // the real one: our cublasGemmEx would recurse
  register_atexit();
  count(name, (per == 2 ? 8.0 : 2.0) * m * n * k, true);
  cudaStream_t s;
  cublasGetStream(h, &s);
  long ar, ac, br, bc;
  stored(is_t(ta), m, k, ar, ac);
  stored(is_t(tb), k, n, br, bc);
  const size_t na = (size_t)ar * ac * per, nb = (size_t)br * bc * per, nc = (size_t)m * n * per;
  const bool x3 = mode == 3;
  float* buf = (float*)stream_scratch(s, ((x3 ? 2 : 1) * (na + nb) + nc) * sizeof(float));
  if (!buf) {
    fprintf(stderr, "[mantshim] MANT_F32 scratch allocation failed\n");
    abort();
  }
  float *Ab = buf, *Bb = Ab + na, *As = x3 ? Bb + nb : nullptr, *Bs = x3 ? As + na : nullptr;
  float* C32 = (x3 ? Bs + nb : Bb + nb);
  split_copy<<<grid_for(ar * per, ac), 256, 0, s>>>((const double*)A, ar, ac, lda, per, Ab, As, mode >= 2);
  split_copy<<<grid_for(br * per, bc), 256, 0, s>>>((const double*)B, br, bc, ldb, per, Bb, Bs, mode >= 2);
  // alpha/beta may live on the device; our own scalars are passed from the host.
  cublasPointerMode_t pm;
  cublasGetPointerMode(h, &pm);
  double2 al = {0, 0}, be = {0, 0};
  const size_t ssz = per * sizeof(double);
  if (pm == CUBLAS_POINTER_MODE_DEVICE) {
    cudaMemcpyAsync(&al, alpha, ssz, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&be, beta, ssz, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  } else {
    memcpy(&al, alpha, ssz);
    memcpy(&be, beta, ssz);
  }
  const float one[2] = {1.f, 0.f}, zero[2] = {0.f, 0.f};
  const cudaDataType dt = per == 2 ? CUDA_C_32F : CUDA_R_32F;
  const cublasComputeType_t ct = mode == 1 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F_FAST_TF32;
  auto mm = [&](const float* a, const float* b, const float* bta) {
    return gemm(h, ta, tb, (int)m, (int)n, (int)k, one, a, dt, (int)ar, b, dt, (int)br, bta, C32, dt, (int)m, ct,
                CUBLAS_GEMM_DEFAULT);
  };
  if (x3) {  // small terms first, then the big product
    status = mm(Ab, Bs, zero);
    if (status == CUBLAS_STATUS_SUCCESS) status = mm(As, Bb, one);
    if (status == CUBLAS_STATUS_SUCCESS) status = mm(Ab, Bb, one);
  } else {
    status = mm(Ab, Bb, zero);
  }
  if (pm == CUBLAS_POINTER_MODE_DEVICE) cublasSetPointerMode(h, pm);
  const bool bz = be.x == 0 && be.y == 0;
  combine<<<grid_for(m, n), 256, 0, s>>>(C32, m, n, per, al, be, bz, (double*)C, ldc);
  if (cfg().verbose)
    fprintf(stderr, "[mantshim] %s f32 mode=%d m=%ld n=%ld k=%ld status=%d\n", name, mode, m, n, k, (int)status);
  return true;
}

}  // namespace

extern "C" {

int mantshim_set_bits(int bits) {
  Config& c = cfg_mut();
  if (bits >= 0) c.bits = bits > 52 ? 52 : bits;
  return c.bits;
}

// ---- cuBLAS ---------------------------------------------------------------
cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                              const double* alpha, const double* A, int lda, const double* B, int ldb,
                              const double* beta, double* C, int ldc) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*,
                                const double*, int, const double*, int, const double*, double*, int);
  static Fn fn = real<Fn>("cublasDgemm_v2");
  cublasStatus_t fs;
  if (f32_gemm("cublasDgemm", h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, fs)) return fs;
  Scratch sc;
  long la = lda, lb = ldb;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 1, 1, 0, 0, sc, la, lb);
  count("cublasDgemm", 2.0 * m * n * k, red);
  if (cfg().verbose) fprintf(stderr, "[mantshim] cublasDgemm %d %d %d reduced=%d\n", m, n, k, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return fn(h, ta, tb, m, n, k, alpha, (const double*)sc.a, (int)la, (const double*)sc.b, (int)lb, beta, C, ldc);
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                              const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
                              const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C,
                              int ldc) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int,
                                const cuDoubleComplex*, cuDoubleComplex*, int);
  static Fn fn = real<Fn>("cublasZgemm_v2");
  cublasStatus_t fs;
  if (f32_gemm("cublasZgemm", h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 2, fs)) return fs;
  Scratch sc;
  long la = lda, lb = ldb;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 2, 1, 0, 0, sc, la, lb);
  count("cublasZgemm", 8.0 * m * n * k, red);
  if (cfg().verbose) fprintf(stderr, "[mantshim] cublasZgemm %d %d %d reduced=%d\n", m, n, k, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return fn(h, ta, tb, m, n, k, alpha, (const cuDoubleComplex*)sc.a, (int)la, (const cuDoubleComplex*)sc.b, (int)lb,
            beta, C, ldc);
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                         int k, const double* alpha, const double* A, int lda, long long sA,
                                         const double* B, int ldb, long long sB, const double* beta, double* C,
                                         int ldc, long long sC, int batch) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*,
                                const double*, int, long long, const double*, int, long long, const double*, double*,
                                int, long long, int);
  static Fn fn = real<Fn>("cublasDgemmStridedBatched");
  if (cfg().f32) {
    cublasStatus_t fs = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < batch && fs == CUBLAS_STATUS_SUCCESS; ++i)
      f32_gemm("cublasDgemmStridedBatched", h, ta, tb, m, n, k, alpha, A + i * sA, lda, B + i * sB, ldb, beta,
               C + i * sC, ldc, 1, fs);
    return fs;
  }
  Scratch sc;
  long la = lda, lb = ldb;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 1, batch, sA, sB, sc, la, lb);
  count("cublasDgemmStridedBatched", 2.0 * m * n * k * batch, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
  long ar, ac, br, bc;
  stored(is_t(ta), m, k, ar, ac);
  stored(is_t(tb), k, n, br, bc);
  return fn(h, ta, tb, m, n, k, alpha, (const double*)sc.a, (int)la, ar * ac, (const double*)sc.b, (int)lb, br * bc,
            beta, C, ldc, sC, batch);
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                         int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
                                         long long sA, const cuDoubleComplex* B, int ldb, long long sB,
                                         const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long sC,
                                         int batch) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                const cuDoubleComplex*, const cuDoubleComplex*, int, long long,
                                const cuDoubleComplex*, int, long long, const cuDoubleComplex*, cuDoubleComplex*, int,
                                long long, int);
  static Fn fn = real<Fn>("cublasZgemmStridedBatched");
  if (cfg().f32) {
    cublasStatus_t fs = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < batch && fs == CUBLAS_STATUS_SUCCESS; ++i)
      f32_gemm("cublasZgemmStridedBatched", h, ta, tb, m, n, k, alpha, A + i * sA, lda, B + i * sB, ldb, beta,
               C + i * sC, ldc, 2, fs);
    return fs;
  }
  Scratch sc;
  long la = lda, lb = ldb;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 2, batch, sA, sB, sc, la, lb);
  count("cublasZgemmStridedBatched", 8.0 * m * n * k * batch, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
  long ar, ac, br, bc;
  stored(is_t(ta), m, k, ar, ac);
  stored(is_t(tb), k, n, br, bc);
  return fn(h, ta, tb, m, n, k, alpha, (const cuDoubleComplex*)sc.a, (int)la, ar * ac, (const cuDoubleComplex*)sc.b,
            (int)lb, br * bc, beta, C, ldc, sC, batch);
}

cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                            const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t ct, cublasGemmAlgo_t algo) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
                                const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*,
                                cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
  static Fn fn = real<Fn>("cublasGemmEx");
  const bool f64 = Atype == CUDA_R_64F && Btype == CUDA_R_64F;
  const bool z64 = Atype == CUDA_C_64F && Btype == CUDA_C_64F;
  if (!f64 && !z64) return fn(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, ct, algo);
  cublasStatus_t fs;
  if (f32_gemm(z64 ? "cublasGemmEx_Z" : "cublasGemmEx_D", h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
               z64 ? 2 : 1, fs))
    return fs;
  Scratch sc;
  long la = lda, lb = ldb;
  const int per = z64 ? 2 : 1;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, per, 1, 0, 0, sc, la, lb);
  count(z64 ? "cublasGemmEx_Z" : "cublasGemmEx_D", (z64 ? 8.0 : 2.0) * m * n * k, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, ct, algo);
  return fn(h, ta, tb, m, n, k, alpha, sc.a, Atype, (int)la, sc.b, Btype, (int)lb, beta, C, Ctype, ldc, ct, algo);
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                          int k, const void* alpha, const void* A, cudaDataType Atype, int lda,
                                          long long sA, const void* B, cudaDataType Btype, int ldb, long long sB,
                                          const void* beta, void* C, cudaDataType Ctype, int ldc, long long sC,
                                          int batch, cublasComputeType_t ct, cublasGemmAlgo_t algo) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
                                const void*, cudaDataType, int, long long, const void*, cudaDataType, int, long long,
                                const void*, void*, cudaDataType, int, long long, int, cublasComputeType_t,
                                cublasGemmAlgo_t);
  static Fn fn = real<Fn>("cublasGemmStridedBatchedEx");
  const bool f64 = Atype == CUDA_R_64F && Btype == CUDA_R_64F;
  const bool z64 = Atype == CUDA_C_64F && Btype == CUDA_C_64F;
  if (!f64 && !z64)
    return fn(h, ta, tb, m, n, k, alpha, A, Atype, lda, sA, B, Btype, ldb, sB, beta, C, Ctype, ldc, sC, batch, ct, algo);
  if (cfg().f32) {
    const int per = z64 ? 2 : 1;
    cublasStatus_t fs = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < batch && fs == CUBLAS_STATUS_SUCCESS; ++i)
      f32_gemm(z64 ? "cublasGemmStridedBatchedEx_Z" : "cublasGemmStridedBatchedEx_D", h, ta, tb, m, n, k, alpha,
               (const double*)A + i * sA * per, lda, (const double*)B + i * sB * per, ldb, beta,
               (double*)C + i * sC * per, ldc, per, fs);
    return fs;
  }
  Scratch sc;
  long la = lda, lb = ldb;
  const int per = z64 ? 2 : 1;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, per, batch, sA, sB, sc, la, lb);
  count(z64 ? "cublasGemmStridedBatchedEx_Z" : "cublasGemmStridedBatchedEx_D", (z64 ? 8.0 : 2.0) * m * n * k * batch,
        red);
  if (!red)
    return fn(h, ta, tb, m, n, k, alpha, A, Atype, lda, sA, B, Btype, ldb, sB, beta, C, Ctype, ldc, sC, batch, ct, algo);
  long ar, ac, br, bc;
  stored(is_t(ta), m, k, ar, ac);
  stored(is_t(tb), k, n, br, bc);
  return fn(h, ta, tb, m, n, k, alpha, sc.a, Atype, (int)la, ar * ac, sc.b, Btype, (int)lb, br * bc, beta, C, Ctype,
            ldc, sC, batch, ct, algo);
}

// 64-bit-index variants (cuSOLVER uses these internally).
cublasStatus_t cublasDgemm_v2_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m, int64_t n,
                                 int64_t k, const double* alpha, const double* A, int64_t lda, const double* B,
                                 int64_t ldb, const double* beta, double* C, int64_t ldc) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t,
                                const double*, const double*, int64_t, const double*, int64_t, const double*, double*,
                                int64_t);
  static Fn fn = real<Fn>("cublasDgemm_v2_64");
  Scratch sc;
  long la = lda, lb = ldb;
  g_stats_only = !cfg().reduce64;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 1, 1, 0, 0, sc, la, lb);
  g_stats_only = false;
  count("cublasDgemm_64", 2.0 * m * n * k, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return fn(h, ta, tb, m, n, k, alpha, (const double*)sc.a, la, (const double*)sc.b, lb, beta, C, ldc);
}

cublasStatus_t cublasZgemm_v2_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m, int64_t n,
                                 int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda,
                                 const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta,
                                 cuDoubleComplex* C, int64_t ldc) {
  using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t,
                                const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*,
                                int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t);
  static Fn fn = real<Fn>("cublasZgemm_v2_64");
  Scratch sc;
  long la = lda, lb = ldb;
  g_stats_only = !cfg().reduce64;
  const bool red = prepare_device(h, is_t(ta), is_t(tb), m, n, k, A, lda, B, ldb, 2, 1, 0, 0, sc, la, lb);
  g_stats_only = false;
  count("cublasZgemm_64", 8.0 * m * n * k, red);
  if (!red) return fn(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return fn(h, ta, tb, m, n, k, alpha, (const cuDoubleComplex*)sc.a, la, (const cuDoubleComplex*)sc.b, lb, beta, C,
            ldc);
}

}  // extern "C"

// ---- CPU BLAS (Fortran interface) -----------------------------------------
static void host_reduce(const double* src, long rows, long cols, long ld, int per, std::vector<double>& dst) {
  dst.resize((size_t)rows * cols * per);
  const int drop = 52 - cfg().bits;
  if (drop == 0) {  // forced copy, nothing to drop
    for (long c = 0; c < cols; ++c)
      memcpy(&dst[(size_t)c * rows * per], &src[c * ld * per], (size_t)rows * per * 8);
    return;
  }
  for (long c = 0; c < cols; ++c)
    for (long r = 0; r < rows; ++r)
      for (int p = 0; p < per; ++p) {
        uint64_t u;
        memcpy(&u, &src[(c * ld + r) * per + p], 8);
        if (((u >> 52) & 0x7ff) != 0x7ff) {
          if (cfg().nearest) u += 1ull << (drop - 1);
          u &= ~((1ull << drop) - 1ull);
        }
        memcpy(&dst[((size_t)c * rows + r) * per + p], &u, 8);
      }
}

template <int PER, typename T>
static void host_gemm(const char* name, const char* sym, const char* ta, const char* tb, const int* m, const int* n,
                      const int* k, const T* alpha, const T* A, const int* lda, const T* B, const int* ldb,
                      const T* beta, T* C, const int* ldc) {
  using Fn = void (*)(const char*, const char*, const int*, const int*, const int*, const T*, const T*, const int*,
                      const T*, const int*, const T*, T*, const int*);
  static Fn fn = real<Fn>(sym);
  const bool tA = (*ta != 'N' && *ta != 'n'), tB = (*tb != 'N' && *tb != 'n');
  long ar, ac, br, bc;
  stored(tA, *m, *k, ar, ac);
  stored(tB, *k, *n, br, bc);
  if (cfg().stats) {
    std::lock_guard<std::mutex> g(st().mu);
    if (!st().atexit_registered) { st().atexit_registered = true; atexit(write_stats); }
    host_record((const double*)A, ar, ac, *lda, PER, st().host_a);
    host_record((const double*)B, br, bc, *ldb, PER, st().host_b);
  }
  const bool red = want_reduce((double)*m * *n * *k) && ar > 0 && br > 0;
  count(name, (PER == 2 ? 8.0 : 2.0) * *m * *n * *k, red);
  if (!red) return fn(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  thread_local std::vector<double> ra, rb;
  host_reduce((const double*)A, ar, ac, *lda, PER, ra);
  host_reduce((const double*)B, br, bc, *ldb, PER, rb);
  const int la = (int)ar, lb = (int)br;
  fn(ta, tb, m, n, k, alpha, (const T*)ra.data(), &la, (const T*)rb.data(), &lb, beta, C, ldc);
}

extern "C" {

void dgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k, const double* alpha,
            const double* A, const int* lda, const double* B, const int* ldb, const double* beta, double* C,
            const int* ldc) {
  host_gemm<1, double>("dgemm", "dgemm_", ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void zgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k,
            const std::complex<double>* alpha, const std::complex<double>* A, const int* lda,
            const std::complex<double>* B, const int* ldb, const std::complex<double>* beta, std::complex<double>* C,
            const int* ldc) {
  host_gemm<2, std::complex<double>>("zgemm", "zgemm_", ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // extern "C"
