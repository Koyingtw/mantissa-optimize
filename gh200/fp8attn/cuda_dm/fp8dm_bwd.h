// FP8 (E4M3) causal attention backward with Delta-Matching, Hopper (sm_90a), CUTLASS/CuTe.
// Host-side launch API (plain pointers; the PyTorch binding is in fp8dm_api.cpp).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace fp8dm {

// Quantize x [S, H, 128] bf16 (strides sxs, sxh, 1) in (head, 128-row) blocks to E4M3.
//   x8  [H, S, 128]        (if write_n)
//   x8t [H, 128, S]        transposed copy, sequence index permuted inside each 16-group so that it
//                          matches the FP8 RS-wgmma register layout of the accumulator-derived A operand
//                          (if write_t)
//   sc  [H, S/128]         dequant scale = amax / 448
//   rowmul [H, S] (opt)    rows are multiplied by rowmul before quantization
//   fixed_sc [H] (opt)     use this per-head scale instead of the block amax (must bound |x|/448)
//   o (opt) -> delta0 [H, S] = sum_d x * o  (fp32, from the unquantized bf16 values)
void quant(const void* x, int64_t sxs, int64_t sxh, const float* rowmul, const float* fixed_sc,
           const void* o, int64_t sos, int64_t soh,
           void* x8, void* x8t, float* sc, float* delta0, int S, int H, cudaStream_t st);

struct DqArgs {
  const void *q8, *do8, *k8, *v8, *k8t;
  const float *sq, *sdo, *sk, *sv, *skmax, *lse2, *delta0;
  const float* mrow_in;   // optional: forward row max of the log2-domain scores [HQ, S]
  void* dq; int64_t dq_s, dq_h;
  float *delta, *mrow2;
  int S, HQ, HKV;
  float scale;          // softmax scale (1/sqrt(d))
  bool matched;
};
void dq(const DqArgs& a, cudaStream_t st);

struct DkvArgs {
  const void *k8, *v8, *q8, *do8, *q8t, *do8t;
  const float *sk, *sv, *sq, *sdo, *sqt, *sdot, *hmax, *mrow2, *delta;
  void *dk, *dv; int64_t dk_s, dk_h;
  float* res;           // optional [HQ, S] row sums of dS / Pmax (test only, atomics)
  int S, HQ, HKV;
  float scale;
};
void dkdv(const DkvArgs& a, cudaStream_t st);

}  // namespace fp8dm
