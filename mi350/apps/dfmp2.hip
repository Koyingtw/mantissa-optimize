// DF-MP2 correlation energy on one MI350X from a precomputed three-index tensor (mi350/apps/mp2_prep.py):
// the GEMM-dominated part of GPU4PySCF's DF-MP2, which does not run on ROCm.
//
// B[i][a][P] (nocc x nvir x naux, row-major) is read once to the host and copied to the GPU for every variant.
// For each occupied i, one FP64 DGEMM forms (ia|jb) = sum_P B[i][a][P] B[j][b][P] for all j <= i at once
// (m = nvir, n = (i+1) nvir, k = naux), then a kernel accumulates
//   E = sum_{j<=i} w_ij sum_ab (ia|jb) [2 (ia|jb) - (ib|ja)] / (e_i + e_j - e_a - e_b),  w = 2 (j<i) or 1 (j=i).
// Truncation variants keep k of the 52 fraction bits of B, in place, once after the copy (as on GH200, where B
// is reduced once after it is built). INT8 emulation runs the same binary under GEMMul8's LD_PRELOAD hook.
//
// usage: dfmp2 --dir D --device G --variants "native t32n t20n t20n t32n native" [--tag s]
// Prints one RESULT_MP2 line per variant: correlation energy, deviation from the PySCF CPU reference and from
// this process's first native run, wall time of the contraction, Unix start/end.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "hipBLAS %s:%d %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

__global__ void reduce_bits(uint64_t* x, size_t n, uint64_t add, uint64_t mask) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = (x[i] + add) & mask;
}

// C: nvir x ((i+1) nvir), column-major; C[a + (j nvir + b) nvir] = (ia|jb).
__global__ void mp2_energy(const double* C, const double* eps, long nocc, long nvir, long i, double* acc) {
  __shared__ double s[256];
  double sum = 0;
  const long total = (i + 1) * nvir * nvir;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
    const long a = idx % nvir, jb = idx / nvir, j = jb / nvir, b = jb % nvir;
    const double t = C[idx], x = C[b + (j * nvir + a) * nvir];
    const double d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
    sum += (j < i ? 2.0 : 1.0) * t * (2 * t - x) / d;
  }
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(acc, s[0]);
}

int main(int argc, char** argv) {
  std::string dir, variants = "native", tag = "run";
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
    if (!strcmp(argv[i], "--dir")) dir = next();
    else if (!strcmp(argv[i], "--device")) device = atoi(next());
    else if (!strcmp(argv[i], "--variants")) variants = next();
    else if (!strcmp(argv[i], "--tag")) tag = next();
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  long nocc = 0, nvir = 0, naux = 0;
  double e_ref = NAN;
  {
    std::ifstream m(dir + "/meta.txt");
    std::string w;
    while (m >> w) {
      auto eq = w.find('=');
      if (eq == std::string::npos) continue;
      const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
      if (k == "nocc") nocc = atol(v.c_str());
      else if (k == "nvir") nvir = atol(v.c_str());
      else if (k == "naux") naux = atol(v.c_str());
      else if (k == "e_corr_ref") e_ref = atof(v.c_str());
    }
  }
  if (!nocc || !nvir || !naux) { fprintf(stderr, "bad %s/meta.txt\n", dir.c_str()); return 1; }
  const size_t nB = (size_t)nocc * nvir * naux;
  std::vector<double> hB(nB), heps(nocc + nvir);
  {
    FILE* f = fopen((dir + "/B.bin").c_str(), "rb");
    if (!f || fread(hB.data(), 8, nB, f) != nB) { fprintf(stderr, "cannot read B.bin\n"); return 1; }
    fclose(f);
    f = fopen((dir + "/eps.bin").c_str(), "rb");
    if (!f || fread(heps.data(), 8, heps.size(), f) != heps.size()) { fprintf(stderr, "cannot read eps.bin\n"); return 1; }
    fclose(f);
  }
  CK(hipSetDevice(device));
  char bdf[32];
  CK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), device));
  double *B, *C, *eps, *acc;
  CK(hipMalloc(&B, nB * 8));
  CK(hipMalloc(&C, (size_t)nvir * nvir * nocc * 8));
  CK(hipMalloc(&eps, heps.size() * 8));
  CK(hipMalloc(&acc, 8));
  CK(hipMemcpy(eps, heps.data(), heps.size() * 8, hipMemcpyHostToDevice));
  hipblasHandle_t h;
  CB(hipblasCreate(&h));
  printf("nocc=%ld nvir=%ld naux=%ld e_ref=%.12f gpu=%d bdf=%s tflop=%.1f\n", nocc, nvir, naux, e_ref, device, bdf,
         (double)nvir * nvir * naux * nocc * (nocc + 1) / 1e12);
  fflush(stdout);

  const double one = 1.0, zero = 0.0;
  double e_native = NAN;
  std::map<std::string, int> seen;
  std::istringstream vs(variants);
  std::string v;
  while (vs >> v) {
    const char pas = "ABCDEFGH"[seen[v]++];
    CK(hipMemcpy(B, hB.data(), nB * 8, hipMemcpyHostToDevice));
    if (v != "native") {
      if (v[0] != 't') { fprintf(stderr, "bad variant %s\n", v.c_str()); return 2; }
      const int bits = atoi(v.c_str() + 1), drop = 52 - bits;
      const uint64_t add = v.back() == 'n' ? 1ull << (drop - 1) : 0, mask = ~((1ull << drop) - 1ull);
      reduce_bits<<<8192, 256>>>((uint64_t*)B, nB, add, mask);
    }
    CK(hipMemset(acc, 0, 8));
    CK(hipDeviceSynchronize());
    const double t0 = now_unix();
    for (long i = 0; i < nocc; ++i) {
      // C = B_i^T B_{0..i} in the column-major view X = naux x (nocc nvir)
      CB(hipblasDgemm(h, HIPBLAS_OP_T, HIPBLAS_OP_N, nvir, (i + 1) * nvir, naux, &one, B + (size_t)i * nvir * naux,
                      naux, B, naux, &zero, C, nvir));
      mp2_energy<<<4096, 256>>>(C, eps, nocc, nvir, i, acc);
    }
    CK(hipDeviceSynchronize());
    const double t1 = now_unix();
    double e;
    CK(hipMemcpy(&e, acc, 8, hipMemcpyDeviceToHost));
    // the denominators e_i + e_j - e_a - e_b are negative, so the sum is already the (negative) correlation energy
    if (v == "native" && std::isnan(e_native)) e_native = e;
    printf("RESULT_MP2 tag=%s gpu=%d variant=%s pass=%c e_corr=%.12f de_ref=%.3e de_native=%.3e wall=%.3fs t_begin=%.3f t_end=%.3f\n",
           tag.c_str(), device, v.c_str(), pas, e, e - e_ref, e - e_native, t1 - t0, t0, t1);
    fflush(stdout);
  }
  CB(hipblasDestroy(h));
  return 0;
}
