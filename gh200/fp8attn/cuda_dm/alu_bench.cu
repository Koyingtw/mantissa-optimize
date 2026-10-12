// micro-benchmark: cycles for the dkdv elementwise mix (32 elements / thread) with 2 or more warpgroups per SM
#include <cstdio>
#include <cstdint>
#include <cuda_fp8.h>
template <int MODE>
__global__ void k(float* out, int iters, long long* cyc) {
  float S[32], P[32];
  for (int i = 0; i < 32; ++i) { S[i] = threadIdx.x * 0.001f + i; P[i] = i * 0.5f; }
  float c1 = 0.01f, c2 = 0.3f, w0 = 1.1f, w1 = 0.9f, acc = 0.f;
  uint32_t pk = 0;
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    float ad0 = 0.f, ad1 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float mr = 0.1f * (i & 7), dl = 0.2f * (i & 3);
      float pv;
      if (MODE == 0) pv = exp2f(fmaf(S[i], c1, -mr));
      else pv = fmaf(S[i], c1, -mr);
      float ds = pv * fmaf(P[i], c2, -dl);
      float dsr = ds * ((i & 1) ? w1 : w0);
      if (i & 1) ad1 = fmaxf(ad1, fabsf(dsr)); else ad0 = fmaxf(ad0, fabsf(dsr));
      S[i] = pv; P[i] = dsr;
    }
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
      __nv_fp8x2_storage_t a = __nv_cvt_float2_to_fp8x2(make_float2(S[i], S[i + 1]), __NV_SATFINITE, __NV_E4M3);
      __nv_fp8x2_storage_t b = __nv_cvt_float2_to_fp8x2(make_float2(P[i], P[i + 1]), __NV_SATFINITE, __NV_E4M3);
      pk ^= (uint32_t(a) << 16) | b;
    }
    acc += ad0 + ad1;
    // perturb so the loop is not hoisted
#pragma unroll
    for (int i = 0; i < 32; ++i) { S[i] = S[i] * 0.999f + 0.001f * i; P[i] = P[i] * 0.5f + 1.f; }
  }
  long long t1 = clock64();
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc + pk;
  if (threadIdx.x == 0 && blockIdx.x == 0) *cyc = t1 - t0;
}
int main() {
  float* out; long long* cyc; cudaMalloc(&out, 132 * 1024 * 4); cudaMallocManaged(&cyc, 8);
  for (int threads : {128, 256, 512}) {
    for (int mode : {0, 1}) {
      if (mode == 0) k<0><<<132, threads>>>(out, 1000, cyc); else k<1><<<132, threads>>>(out, 1000, cyc);
      cudaDeviceSynchronize();
      printf("threads/SM=%d mode=%s  cycles per iteration (32 elem/thread) = %.0f\n", threads, mode ? "no-exp" : "exp", *cyc / 1000.0);
    }
  }
}
