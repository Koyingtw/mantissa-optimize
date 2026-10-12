// FP8 (E4M3) causal attention backward with Delta-Matching for Hopper (sm_90a), CUTLASS 3.6 / CuTe.
//
// Design (same math as the Triton prototype gh200/fp8attn/fp8attn2.py, "approach (1)"):
//   quant  : Q, K, V, dO -> E4M3 with one scale per (head, 128 rows); K^T copy for dQ.
//   dq     : Q-outer (128 query rows / CTA, 2 consumer warpgroups x 64 rows), streams K, V, K^T.
//            S = Q K^T, dP = dO V^T (FP8 SS-wgmma), P = exp2(S c - lse) in FP32,
//            dS0 = P (dP - delta0)            -> accA += dS0 K   (FP8 RS-wgmma)
//            P                                -> accB += P K     (FP8 RS-wgmma, Delta-Matching only)
//            delta = sum_j P dP  (matched, FP32);  dQ = accA - (delta - delta0) accB.
//            Also writes delta and the row max of P (mrow2 = lse + log2 Pmax).
//   quant2 : Q, dO rows times Pmax -> transposed E4M3 copies for dK / dV.
//   dkdv   : KV-outer (128 keys / CTA, 2 consumer WGs x 64 keys), loops over the 8 query heads of the
//            KV head (dK, dV stay in registers, no atomics), streams Q, dO, Q'^T, dO'^T, mrow2, delta.
//            P' = P / Pmax (row max 1), dS' = P'(dP - delta) -> dK += dS' Q', dV += P' dO'.
// Every GEMM is FP8 x FP8 -> FP32 on wgmma; both operands K-major.  P / dS go from accumulator
// registers straight into the A operand of the next wgmma (permute_Cregs_fp8 = a pure register
// renaming); the resulting permutation of the contraction index is baked into the transposed copies
// written by the quant kernel (perm16 below).
// dS quantization: per-row scale, chosen lazily = power of two that fits the running row amax
// (accumulator rows are rescaled by exact powers of two when the scale has to shrink).

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "sm90_pipeline_no_cluster.hpp"   // FA3 hopper/
#include "utils.h"                        // FA3 hopper/: flash::gemm, layouts, permute_Cregs_fp8

#include "fp8dm_bwd.h"

#define FP8DM_CHECK(x)                                                                          \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "fp8dm CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    }                                                                                           \
  } while (0)

namespace fp8dm {

using namespace cute;
using Elem = cutlass::float_e4m3_t;
static constexpr int kD = 128;
static constexpr float kFp8Max = 448.f;
static constexpr float kLr0 = 24.f;       // initial log2 of the lazy dS scale (forces a fit on 1st tile)
static constexpr int kMaxNB = 128;        // S <= 16384
static constexpr int kMaxGroup = 8;       // query heads per KV head
#ifdef FP8DM_X_NOEXP
#define FP8DM_EXP2(x) fmaf(x, 0.001f, 0.5f)
#endif
#ifndef FP8DM_EXP2
#define FP8DM_EXP2(x) exp2f(x)
#endif
#ifndef FP8DM_SCHED
#define FP8DM_SCHED 0
#endif
#ifndef FP8DM_DELAY
#define FP8DM_DELAY 0
#endif
#ifndef FP8DM_SQH
#define FP8DM_SQH p.hmax[h]
#endif
#ifdef FP8DM_SQH_POW2
#undef FP8DM_SQH
#define FP8DM_SQH exp2f(rintf(log2f(p.hmax[h])))
#endif

#ifdef FP8DM_PROF
#define FP8DM_T(i) pt_[i] = clock64()
#define FP8DM_ACC(k, a, b) pr_[k] += pt_[b] - pt_[a]
#else
#define FP8DM_T(i)
#define FP8DM_ACC(k, a, b)
#endif

// position a inside a 16-group of the contraction index  ->  actual row/col held there
__host__ __device__ __forceinline__ int perm16(int a) { return 2 * (a >> 2) + (a & 1) + 8 * ((a >> 1) & 1); }

// ============================================================================================
// Quantization
// ============================================================================================
template <bool WriteN, bool WriteT, bool RowMul, bool WithDelta, bool Fixed>
__global__ void __launch_bounds__(256) quant_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t sxs, int64_t sxh, const float* __restrict__ rowmul,
    const float* __restrict__ fixed_sc,
    const __nv_bfloat16* __restrict__ o, int64_t sos, int64_t soh,
    uint8_t* __restrict__ x8, uint8_t* __restrict__ x8t, float* __restrict__ sc, float* __restrict__ delta0, int S) {
  __shared__ uint8_t tile[128][128 + 16];
  __shared__ float red[8];
  const int blk = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int row0 = blk * 128;
  const int c0 = (tid & 15) * 8;
  float v[8][8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (tid >> 4) + 16 * i;
    const uint4 u = *reinterpret_cast<const uint4*>(x + (int64_t)(row0 + r) * sxs + (int64_t)h * sxh + c0);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
    float rm = 1.f;
    if constexpr (RowMul) { rm = rowmul[(int64_t)h * S + row0 + r]; }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(b[j]);
      v[i][2 * j] = f.x * rm;
      v[i][2 * j + 1] = f.y * rm;
    }
    if constexpr (WithDelta) {
      const uint4 uo = *reinterpret_cast<const uint4*>(o + (int64_t)(row0 + r) * sos + (int64_t)h * soh + c0);
      const __nv_bfloat162* bo = reinterpret_cast<const __nv_bfloat162*>(&uo);
      float d = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(bo[j]);
        d += f.x * v[i][2 * j] + f.y * v[i][2 * j + 1];
      }
#pragma unroll
      for (int off = 8; off >= 1; off >>= 1) d += __shfl_xor_sync(0xffffffff, d, off);
      if ((tid & 15) == 0) delta0[(int64_t)h * S + row0 + r] = d;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[i][j]));
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  if ((tid & 31) == 0) red[tid >> 5] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
  const float s = Fixed ? fixed_sc[h] : fmaxf(amax, 1e-30f) / kFp8Max;
  const float inv = 1.f / s;
  if (tid == 0) sc[(int64_t)h * (S / 128) + blk] = s;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (tid >> 4) + 16 * i;
    uint32_t w[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[i][4 * j] * inv, v[i][4 * j + 1] * inv), __NV_SATFINITE, __NV_E4M3);
      __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[i][4 * j + 2] * inv, v[i][4 * j + 3] * inv), __NV_SATFINITE, __NV_E4M3);
      w[j] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
    if constexpr (WriteN) {
      *reinterpret_cast<uint2*>(x8 + ((int64_t)h * S + row0 + r) * kD + c0) = make_uint2(w[0], w[1]);
    }
    if constexpr (WriteT) {
      *reinterpret_cast<uint2*>(&tile[r][c0]) = make_uint2(w[0], w[1]);
    }
  }
  if constexpr (WriteT) {
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int task = tid + 256 * k;
      const int d = task >> 3, ch = task & 7;
      uint32_t w[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t acc = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) acc |= uint32_t(tile[ch * 16 + perm16(4 * q + b)][d]) << (8 * b);
        w[q] = acc;
      }
      *reinterpret_cast<uint4*>(x8t + ((int64_t)h * kD + d) * S + row0 + ch * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

void quant(const void* x, int64_t sxs, int64_t sxh, const float* rowmul, const float* fixed_sc, const void* o, int64_t sos,
           int64_t soh, void* x8, void* x8t, float* sc, float* delta0, int S, int H, cudaStream_t st) {
  dim3 grid(S / 128, H);
  auto X = static_cast<const __nv_bfloat16*>(x);
  auto O = static_cast<const __nv_bfloat16*>(o);
  auto X8 = static_cast<uint8_t*>(x8);
  auto X8T = static_cast<uint8_t*>(x8t);
#define FP8DM_Q(WN, WT, RM, WD) quant_kernel<WN, WT, RM, WD, false><<<grid, 256, 0, st>>>(X, sxs, sxh, rowmul, fixed_sc, O, sos, soh, X8, X8T, sc, delta0, S)
  const bool wn = x8 != nullptr, wt = x8t != nullptr, rm = rowmul != nullptr, wd = o != nullptr;
  if (wn && !wt && !rm && !wd) FP8DM_Q(true, false, false, false);
  else if (wn && wt && !rm && !wd) FP8DM_Q(true, true, false, false);
  else if (wn && !wt && !rm && wd) FP8DM_Q(true, false, false, true);
  else if (!wn && wt && rm && !wd && fixed_sc)
    quant_kernel<false, true, true, false, true><<<grid, 256, 0, st>>>(X, sxs, sxh, rowmul, fixed_sc, O, sos, soh, X8, X8T, sc, delta0, S);
  else if (!wn && wt && rm && !wd) FP8DM_Q(false, true, true, false);
  else if (!wn && wt && !rm && !wd) FP8DM_Q(false, true, false, false);
  else fprintf(stderr, "fp8dm::quant: unsupported combination\n");
#undef FP8DM_Q
  FP8DM_CHECK(cudaGetLastError());
}

// ============================================================================================
// helpers
// ============================================================================================
template <class TA, class TB>
__device__ __forceinline__ void fake_cvt(TA const& a, TB& b) {   // timing experiment only: byte pack without F2FP
  auto bb = recast<uint32_t>(b);
#pragma unroll
  for (int i = 0; i < size(bb); ++i) {
    const uint32_t x0 = __float_as_uint(a(4 * i)), x1 = __float_as_uint(a(4 * i + 1));
    const uint32_t x2 = __float_as_uint(a(4 * i + 2)), x3 = __float_as_uint(a(4 * i + 3));
    bb(i) = __byte_perm(__byte_perm(x0, x1, 0x0073), __byte_perm(x2, x3, 0x0073), 0x5410);
  }
}

template <typename T>
__device__ __forceinline__ T quad_max(T x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
  return x;
}
__device__ __forceinline__ float quad_sum(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1);
  x += __shfl_xor_sync(0xffffffff, x, 2);
  return x;
}

template <class TMA, class GT, class ST>
__device__ __forceinline__ auto tma_part(TMA const& tma, GT const& gt, ST const& st) {
  return tma_partition(tma, _0{}, Layout<_1>{}, group_modes<0, 2>(st), group_modes<0, 2>(gt));
}

using ShapeT = Shape<int32_t, int32_t, int32_t>;
using StrideT = Stride<int64_t, _1, int64_t>;

template <class SL, class Tile>
using TmaT = decltype(make_tma_copy(SM90_TMA_LOAD{},
                                    make_tensor(make_gmem_ptr(static_cast<Elem const*>(nullptr)), ShapeT{}, StrideT{}),
                                    SL{}, Tile{}, _1{}));

template <class SL, class Tile>
TmaT<SL, Tile> make_tma(const void* ptr, int rows, int cols, int heads) {
  auto t = make_tensor(make_gmem_ptr(static_cast<Elem const*>(ptr)), make_shape(rows, cols, heads),
                       make_stride(int64_t(cols), _1{}, int64_t(rows) * cols));
  return make_tma_copy(SM90_TMA_LOAD{}, t, SL{}, Tile{}, _1{});
}

// ============================================================================================
// dQ kernel (Q-outer) with Delta-Matching
// ============================================================================================
template <int kBlockN_, int kStages_>
struct DqCfg {
  static constexpr int kBlockM = 128, kBlockN = kBlockN_, kStages = kStages_;
  using TileS = Shape<Int<kBlockM>, Int<kBlockN>, Int<kD>>;
  using TiledQ = Shape<Int<kBlockM>, Int<kD>, Int<kBlockN>>;
  using AtomLayout = Layout<Shape<_2, _1, _1>>;
  using TiledMmaS = decltype(make_tiled_mma(GMMA::ss_op_selector<Elem, Elem, float, TileS>(), AtomLayout{}));
  using TiledMmadQ = decltype(make_tiled_mma(
      GMMA::rs_op_selector<Elem, Elem, float, TiledQ, GMMA::Major::K, GMMA::Major::K>(), AtomLayout{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kBlockM>, Int<kD>>(),
      Shape<Int<kBlockM>, Int<kD>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kBlockN>, Int<kD>>(),
      Shape<Int<kBlockN>, Int<kD>, Int<kStages>>{}));
  using SmemLayoutKt = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kD>, Int<kBlockN>>(),
      Shape<Int<kD>, Int<kBlockN>, Int<kStages>>{}));
  using TMA_Q = TmaT<SmemLayoutQ, Shape<Int<kBlockM>, Int<kD>>>;
  using TMA_K = TmaT<decltype(take<0, 2>(SmemLayoutK{})), Shape<Int<kBlockN>, Int<kD>>>;
  using TMA_Kt = TmaT<decltype(take<0, 2>(SmemLayoutKt{})), Shape<Int<kD>, Int<kBlockN>>>;
  using Pipeline = cutlass::PipelineTmaAsyncNoCluster<kStages>;
  static constexpr uint32_t kBytesQ = kBlockM * kD;
  static constexpr uint32_t kBytesStage = 3 * kBlockN * kD;

  struct SharedStorage {
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQ>, 1024> q;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQ>, 1024> dO;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutK>, 1024> k;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutK>, 1024> v;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutKt>, 1024> kt;
    alignas(16) float skv[3][kMaxNB];   // per 128-key block: sk, sv, log2(sk / skmax)
    alignas(16) cutlass::arch::ClusterTransactionBarrier barQ;
    alignas(16) typename Pipeline::SharedStorage pipe;
  };
  struct Params {
    TMA_Q tma_q, tma_do;
    TMA_K tma_k, tma_v;
    TMA_Kt tma_kt;
    int S, HQ, HKV, group;
    float scale_log2, scale;
    const float *sq, *sdo, *sk, *sv, *skmax, *lse2, *delta0;
    const float* mrow_in;   // optional forward row max of the log2 scores (then P's FP8 scale is known)
    __nv_bfloat16* dq;
    int64_t dq_s, dq_h;
    float *delta, *mrow2;
  };
};

template <class C, bool Matched, bool KnownMax>
__global__ void __launch_bounds__(384, 1) dq_kernel(CUTE_GRID_CONSTANT typename C::Params const p) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN;
  using Pipeline = typename C::Pipeline;
  extern __shared__ __align__(1024) char smem_raw[];
  auto& ss = *reinterpret_cast<typename C::SharedStorage*>(smem_raw);

  const int m_block = gridDim.x - 1 - blockIdx.x;   // heaviest (longest causal row range) first
  const int h = blockIdx.y;
  const int g = h / p.group;
  const int NB = p.S / 128;
  const int n_blocks = (m_block + 1) * (kBlockM / kBlockN);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int wg_idx = cutlass::canonical_warp_group_idx();
  const int lane_pred = cute::elect_one_sync();

  if (warp_idx == 0 && lane_pred) {
    cute::prefetch_tma_descriptor(p.tma_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_do.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_kt.get_tma_descriptor());
    ss.barQ.init(1);
  }
  typename Pipeline::Params pp;
  pp.transaction_bytes = C::kBytesStage;
  pp.role = wg_idx == 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
  pp.is_leader = (threadIdx.x % 128) == 0;
  pp.num_consumers = 256;
  Pipeline pipe(ss.pipe, pp, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ = make_tensor(make_smem_ptr(ss.q.data()), typename C::SmemLayoutQ{});
  Tensor sdO = make_tensor(make_smem_ptr(ss.dO.data()), typename C::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(ss.k.data()), typename C::SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(ss.v.data()), typename C::SmemLayoutK{});
  Tensor sKt = make_tensor(make_smem_ptr(ss.kt.data()), typename C::SmemLayoutKt{});

  if (wg_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_idx == 0 && lane_pred) {
      Tensor mQ = p.tma_q.get_tma_tensor(make_shape(p.S, int(kD), p.HQ));
      Tensor mdO = p.tma_do.get_tma_tensor(make_shape(p.S, int(kD), p.HQ));
      Tensor mK = p.tma_k.get_tma_tensor(make_shape(p.S, int(kD), p.HKV));
      Tensor mV = p.tma_v.get_tma_tensor(make_shape(p.S, int(kD), p.HKV));
      Tensor mKt = p.tma_kt.get_tma_tensor(make_shape(int(kD), p.S, p.HKV));
      Tensor gQ = local_tile(mQ(_, _, h), Shape<Int<kBlockM>, Int<kD>>{}, make_coord(m_block, _0{}));
      Tensor gdO = local_tile(mdO(_, _, h), Shape<Int<kBlockM>, Int<kD>>{}, make_coord(m_block, _0{}));
      Tensor gK = local_tile(mK(_, _, g), Shape<Int<kBlockN>, Int<kD>>{}, make_coord(_, _0{}));
      Tensor gV = local_tile(mV(_, _, g), Shape<Int<kBlockN>, Int<kD>>{}, make_coord(_, _0{}));
      Tensor gKt = local_tile(mKt(_, _, g), Shape<Int<kD>, Int<kBlockN>>{}, make_coord(_0{}, _));
      auto [tQg, tQs] = tma_part(p.tma_q, gQ, sQ);
      auto [tdOg, tdOs] = tma_part(p.tma_do, gdO, sdO);
      auto [tKg, tKs] = tma_part(p.tma_k, gK, sK);
      auto [tVg, tVs] = tma_part(p.tma_v, gV, sV);
      auto [tKtg, tKts] = tma_part(p.tma_kt, gKt, sKt);
      using BarT = cutlass::arch::ClusterTransactionBarrier::ValueType;
      ss.barQ.arrive_and_expect_tx(2 * C::kBytesQ);
      copy(p.tma_q.with(reinterpret_cast<BarT&>(ss.barQ), 0), tQg, tQs);
      copy(p.tma_do.with(reinterpret_cast<BarT&>(ss.barQ), 0), tdOg, tdOs);
      auto ws = cutlass::make_producer_start_state<Pipeline>();
      for (int n = 0; n < n_blocks; ++n) {
        pipe.producer_acquire(ws);
        auto* bar = pipe.producer_get_barrier(ws);
        const int s = ws.index();
        copy(p.tma_k.with(*bar, 0), tKg(_, n), tKs(_, s));
        copy(p.tma_v.with(*bar, 0), tVg(_, n), tVs(_, s));
        copy(p.tma_kt.with(*bar, 0), tKtg(_, n), tKts(_, s));
        ++ws;
      }
      pipe.producer_tail(ws);
    }
    return;
  }

  // ---------------------------------------------------------------- consumers
  cutlass::arch::warpgroup_reg_alloc<240>();
  const int tidx = threadIdx.x - 128;
  const int cwg = tidx / 128;
  typename C::TiledMmaS mmaS;
  typename C::TiledMmadQ mmadQ;
  Layout wgl = make_layout(make_shape(_2{}), make_stride(Int<128>{}));
  auto wgS = mmaS.get_slice(wgl(cwg));
  auto wgdQ = mmadQ.get_slice(wgl(cwg));
  auto thrS = mmaS.get_thread_slice(tidx);
  auto thrdQ = mmadQ.get_thread_slice(tidx);
  Tensor tSrQ = wgS.partition_fragment_A(sQ);
  Tensor tSrdO = wgS.partition_fragment_A(sdO);
  Tensor tSrK = wgS.partition_fragment_B(sK);
  Tensor tSrV = wgS.partition_fragment_B(sV);
  Tensor tdQrKt = wgdQ.partition_fragment_B(sKt);

  Tensor accA = partition_fragment_C(mmadQ, Shape<Int<kBlockM>, Int<kD>>{});
  Tensor accB = partition_fragment_C(mmadQ, Shape<Int<kBlockM>, Int<kD>>{});
  clear(accA);
  clear(accB);
  Tensor accA_rc = make_tensor(accA.data(), flash::convert_layout_acc_rowcol(accA.layout()));
  Tensor accB_rc = make_tensor(accB.data(), flash::convert_layout_acc_rowcol(accB.layout()));

  Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tScS = thrS.partition_C(cS);
  Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int NR = decltype(size<0>(tScS_rc))::value;   // 2
  constexpr int NC = decltype(size<1>(tScS_rc))::value;   // kBlockN / 4
  static_assert(NR == 2);

  // Row scales (lazy, shrink-only, the row max lands exactly on 448):
  //   pr = P * rP * sk/skmax   (exp2 bias)            -> FP8 A operand of accB += P K
  //   ds = pr * c * (dP - delta0)                     -> FP8 A operand of accA += dS K
  // true sum_j P (dP - delta0) K = accA * skmax / (rP c),  true sum_j P K = accB * skmax / rP.
  float lse2[NR], d0[NR], lrp[NR], cc[NR], dpart[NR], psum[NR], pmax[NR];
  int qrow[NR];
#pragma unroll
  for (int mi = 0; mi < NR; ++mi) {
    qrow[mi] = m_block * kBlockM + int(get<0>(tScS_rc(mi, 0)));
    lse2[mi] = p.lse2[(int64_t)h * p.S + qrow[mi]];
    d0[mi] = p.delta0[(int64_t)h * p.S + qrow[mi]];
    // KnownMax: rP = 448 / Pmax (row max of P lands on 448); else lazy like c
    lrp[mi] = KnownMax ? (8.807354922057604f + lse2[mi] - p.mrow_in[(int64_t)h * p.S + qrow[mi]]) : kLr0;
    cc[mi] = exp2f(kLr0);
    dpart[mi] = 0.f;
    psum[mi] = 0.f;
    pmax[mi] = 0.f;
  }
  const float sq = p.sq[h * NB + m_block];
  const float sdo = p.sdo[h * NB + m_block];
  const float skmax = p.skmax[g];
  const float cq = sq * p.scale_log2;
  const int tcol0 = 2 * (threadIdx.x % 4);
  static_assert(NC == kBlockN / 4);

  // ping-pong between the two consumer warpgroups: each WG issues its GEMMs only on its turn
  // FP8DM_SCHED 1: strict ping-pong of GEMM issue between the 2 consumer WGs; 0: free running, WG1 delayed
  // by FP8DM_DELAY cycles at the start (de-phases the two WGs)
  auto sched_sync = [&]() { if constexpr (FP8DM_SCHED == 1) cutlass::arch::NamedBarrier::sync(256, 1 + cwg); };
  auto sched_arrive = [&]() { if constexpr (FP8DM_SCHED == 1) cutlass::arch::NamedBarrier::arrive(256, 1 + (1 - cwg)); };
  if constexpr (FP8DM_SCHED == 1) {
    if (cwg == 1) cutlass::arch::NamedBarrier::arrive(256, 1);   // WG0 goes first
  } else if (FP8DM_DELAY > 0 && cwg == 1) {
    const long long t0 = clock64();
    while (clock64() - t0 < FP8DM_DELAY) { }
  }

  for (int i = tidx; i < NB; i += 256) {
    const float a = p.sk[g * NB + i];
    ss.skv[0][i] = a;
    ss.skv[1][i] = p.sv[g * NB + i];
    ss.skv[2][i] = __log2f(a / skmax);
  }
  cutlass::arch::NamedBarrier::sync(256, 3);
  ss.barQ.wait(0);
  typename Pipeline::PipelineState rs;
#ifdef FP8DM_PROF
  long long pt_[6]; long long pr_[5] = {0, 0, 0, 0, 0};
#endif
  const int n_diag = (m_block * kBlockM) / kBlockN;   // first tile that touches the diagonal
  Tensor tSrS = partition_fragment_C(mmaS, Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tdPrdP = partition_fragment_C(mmaS, Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor S_rc = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
  Tensor dP_rc = make_tensor(tdPrdP.data(), flash::convert_layout_acc_rowcol(tdPrdP.layout()));
  CUTLASS_PRAGMA_NO_UNROLL
  for (int n = 0; n < n_blocks; ++n) {
    const int stg = rs.index();
    FP8DM_T(0);
    {
      auto tok = pipe.consumer_try_wait(rs);
      pipe.consumer_wait(rs, tok);
    }
    sched_sync();
#ifndef FP8DM_X_NOG1
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(mmaS, tSrQ, tSrK(_, _, _, stg), tSrS);
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(mmaS, tSrdO, tSrV(_, _, _, stg), tdPrdP);
#endif
    sched_arrive();
    const int nb = (n * kBlockN) / 128;
    const float sk = ss.skv[0][nb];
    const float sv = ss.skv[1][nb];
    const float cs = cq * sk;
    const float cp = sdo * sv;
    const float lks = ss.skv[2][nb];
    warpgroup_wait<0>();
    FP8DM_T(1); FP8DM_ACC(0, 0, 1);
    float bias[NR], cpc[NR], d0c[NR];
#pragma unroll
    for (int mi = 0; mi < NR; ++mi) {
      bias[mi] = lrp[mi] + lks - lse2[mi];
      cpc[mi] = cp * cc[mi];
      d0c[mi] = d0[mi] * cc[mi];
    }
    float es[NR], ps[NR], ad[NR], ap[NR];
#pragma unroll
    for (int mi = 0; mi < NR; ++mi) { es[mi] = 0.f; ps[mi] = 0.f; ad[mi] = 0.f; ap[mi] = 0.f; }
#ifndef FP8DM_X_NOALU
    auto elem = [&](auto diag_c) {
      constexpr bool Diag = decltype(diag_c)::value;
      int lim[NR];   // mask cols with local index > lim (closed form of the wgmma C layout: col = 2 (lane%4) + (ni&1) + 8 (ni>>1))
#pragma unroll
      for (int mi = 0; mi < NR; ++mi) lim[mi] = qrow[mi] - n * kBlockN - tcol0;
#pragma unroll
      for (int mi = 0; mi < NR; ++mi)
#pragma unroll
        for (int ni = 0; ni < NC; ++ni) {
          float pr = FP8DM_EXP2(fmaf(S_rc(mi, ni), cs, bias[mi]));
          if constexpr (Diag) { if (((ni & 1) + 8 * (ni >> 1)) > lim[mi]) pr = 0.f; }
          const float ds = pr * fmaf(dP_rc(mi, ni), cpc[mi], -d0c[mi]);
          es[mi] += ds;
          ps[mi] += pr;
          ad[mi] = fmaxf(ad[mi], fabsf(ds));
          if constexpr (!KnownMax) ap[mi] = fmaxf(ap[mi], pr);
          S_rc(mi, ni) = pr;
          dP_rc(mi, ni) = ds;
        }
    };
    if (n >= n_diag) elem(cute::true_type{}); else elem(cute::false_type{});
    FP8DM_T(2); FP8DM_ACC(1, 1, 2);
#pragma unroll
    for (int mi = 0; mi < NR; ++mi) {
      ad[mi] = quad_max(ad[mi]);
      if constexpr (!KnownMax) ap[mi] = quad_max(ap[mi]);
      if (!KnownMax && ap[mi] > kFp8Max) {          // shrink rP (scales both P and dS of this row)
        const float f = kFp8Max / ap[mi];
        lrp[mi] += __log2f(f);
#pragma unroll
        for (int ni = 0; ni < NC; ++ni) { S_rc(mi, ni) *= f; dP_rc(mi, ni) *= f; }
#pragma unroll
        for (int c = 0; c < size<1>(accA_rc); ++c) { accA_rc(mi, c) *= f; accB_rc(mi, c) *= f; }
        es[mi] *= f; ps[mi] *= f; ad[mi] *= f; ap[mi] = kFp8Max;
      }
      if (ad[mi] > kFp8Max) {                       // shrink c (dS only)
        const float f = kFp8Max / ad[mi];
        cc[mi] *= f;
#pragma unroll
        for (int ni = 0; ni < NC; ++ni) dP_rc(mi, ni) *= f;
#pragma unroll
        for (int c = 0; c < size<1>(accA_rc); ++c) accA_rc(mi, c) *= f;
        es[mi] *= f;
      }
      const float invp = exp2f(-lks - lrp[mi]);     // skmax / (sk rP)
      dpart[mi] = fmaf(es[mi], invp / cc[mi], dpart[mi]);
      psum[mi] = fmaf(ps[mi], invp, psum[mi]);
      if constexpr (!KnownMax) pmax[mi] = fmaxf(pmax[mi], ap[mi] * invp);
    }
#endif
    FP8DM_T(3); FP8DM_ACC(2, 2, 3);
    // dS, P -> FP8 A operands (register renaming only; the K^T copy carries the matching permutation)
    flash::permute_Cregs_fp8(tdPrdP);
    Tensor tdS_a = make_tensor(tdPrdP.data(), flash::convert_layout_acc_Aregs<typename C::TiledMmadQ>(tdPrdP.layout()));
    Tensor rdS = make_tensor_like<Elem>(tdS_a);
    flash::convert_type_out(tdS_a, rdS);
    flash::permute_Cregs_fp8(tSrS);
    Tensor tP_a = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<typename C::TiledMmadQ>(tSrS.layout()));
    Tensor rP = make_tensor_like<Elem>(tP_a);
    if constexpr (Matched) flash::convert_type_out(tP_a, rP);
    FP8DM_T(4); FP8DM_ACC(3, 3, 4);
    sched_sync();
#ifndef FP8DM_X_NOG2
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(mmadQ, rdS, tdQrKt(_, _, _, stg), accA);
    if constexpr (Matched) flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(mmadQ, rP, tdQrKt(_, _, _, stg), accB);
#endif
    sched_arrive();
    warpgroup_wait<0>();
    pipe.consumer_release(rs);
    ++rs;
    FP8DM_T(5); FP8DM_ACC(4, 4, 5);
  }

#ifdef FP8DM_PROF
  if (blockIdx.x < 2 && blockIdx.y < 2 && (tidx % 128) == 0)
    printf("PROF dq m_block=%d h=%d wg=%d n=%d  wait_gemm=%lld alu=%lld lazy=%lld cvt_waitdata=%lld turn+issue=%lld per-iter total=%lld\n", m_block, h, cwg, n_blocks, pr_[0]/n_blocks, pr_[1]/n_blocks, pr_[2]/n_blocks, pr_[3]/n_blocks, pr_[4]/n_blocks, (pr_[0]+pr_[1]+pr_[2]+pr_[3]+pr_[4])/n_blocks);
#endif
  // ---------------------------------------------------------------- epilogue
  Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kD>>{});
  Tensor tOcO = thrdQ.partition_C(cO);
  Tensor tOcO_rc = make_tensor(tOcO.data(), flash::convert_layout_acc_rowcol(tOcO.layout()));
#pragma unroll
  for (int mi = 0; mi < NR; ++mi) {
    const float dp = quad_sum(dpart[mi]);
    const float psm = quad_sum(psum[mi]);
    const float delta = Matched ? fmaf(d0[mi], psm, dp) : d0[mi];
    const float eps = delta - d0[mi];
    const float cb0 = p.scale * skmax * exp2f(-lrp[mi]);
    const float ca = cb0 / cc[mi];
    const float cb = Matched ? cb0 * eps : 0.f;
    const int row = m_block * kBlockM + int(get<0>(tOcO_rc(mi, 0)));
    __nv_bfloat16* out = p.dq + (int64_t)row * p.dq_s + (int64_t)h * p.dq_h;
#pragma unroll
    for (int c = 0; c < size<1>(accA_rc); c += 2) {
      const int col = int(get<1>(tOcO_rc(mi, c)));
      float x0 = accA_rc(mi, c) * ca, x1 = accA_rc(mi, c + 1) * ca;
      if constexpr (Matched) { x0 -= accB_rc(mi, c) * cb; x1 -= accB_rc(mi, c + 1) * cb; }
      *reinterpret_cast<__nv_bfloat162*>(out + col) = __floats2bfloat162_rn(x0, x1);
    }
    if ((tidx & 3) == 0) {
      p.delta[(int64_t)h * p.S + row] = delta;
      p.mrow2[(int64_t)h * p.S + row] = KnownMax ? p.mrow_in[(int64_t)h * p.S + row] : lse2[mi] + __log2f(pmax[mi]);
    }
  }
}

template <int kBlockN, int kStages, bool Matched, bool KnownMax>
void run_dq(const DqArgs& a, cudaStream_t st) {
  using C = DqCfg<kBlockN, kStages>;
  typename C::Params p{
      make_tma<typename C::SmemLayoutQ, Shape<Int<C::kBlockM>, Int<kD>>>(a.q8, a.S, kD, a.HQ),
      make_tma<typename C::SmemLayoutQ, Shape<Int<C::kBlockM>, Int<kD>>>(a.do8, a.S, kD, a.HQ),
      make_tma<decltype(take<0, 2>(typename C::SmemLayoutK{})), Shape<Int<kBlockN>, Int<kD>>>(a.k8, a.S, kD, a.HKV),
      make_tma<decltype(take<0, 2>(typename C::SmemLayoutK{})), Shape<Int<kBlockN>, Int<kD>>>(a.v8, a.S, kD, a.HKV),
      make_tma<decltype(take<0, 2>(typename C::SmemLayoutKt{})), Shape<Int<kD>, Int<kBlockN>>>(a.k8t, kD, a.S, a.HKV),
      a.S, a.HQ, a.HKV, a.HQ / a.HKV, float(a.scale * M_LOG2E), a.scale,
      a.sq, a.sdo, a.sk, a.sv, a.skmax, a.lse2, a.delta0, a.mrow_in,
      static_cast<__nv_bfloat16*>(a.dq), a.dq_s, a.dq_h, a.delta, a.mrow2};
  auto kern = dq_kernel<C, Matched, KnownMax>;
  const int smem = sizeof(typename C::SharedStorage);
  FP8DM_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kern<<<dim3(a.S / C::kBlockM, a.HQ), 384, smem, st>>>(p);
  FP8DM_CHECK(cudaGetLastError());
}

void dq(const DqArgs& a, cudaStream_t st) {
  if (a.S / 128 > kMaxNB || a.S % 128) { fprintf(stderr, "fp8dm::dq: S must be a multiple of 128 and <= %d\n", kMaxNB * 128); return; }
  if (a.mrow_in) {
    if (a.matched) run_dq<64, 4, true, true>(a, st);
    else run_dq<64, 4, false, true>(a, st);
  } else {
    if (a.matched) run_dq<64, 4, true, false>(a, st);
    else run_dq<64, 4, false, false>(a, st);
  }
}

// ============================================================================================
// dK dV kernel (KV-outer, loops over the query heads of the KV head)
// ============================================================================================
template <int kBlockM_, int kStages_>
struct DkvCfg {
  static constexpr int kBlockN = 128, kBlockM = kBlockM_, kStages = kStages_;
  using TileS = Shape<Int<kBlockN>, Int<kBlockM>, Int<kD>>;      // keys x queries x d
  using TiledKV = Shape<Int<kBlockN>, Int<kD>, Int<kBlockM>>;    // keys x d x queries
  using AtomLayout = Layout<Shape<_2, _1, _1>>;
  using TiledMmaS = decltype(make_tiled_mma(GMMA::ss_op_selector<Elem, Elem, float, TileS>(), AtomLayout{}));
  using TiledMmadKV = decltype(make_tiled_mma(
      GMMA::rs_op_selector<Elem, Elem, float, TiledKV, GMMA::Major::K, GMMA::Major::K>(), AtomLayout{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kBlockN>, Int<kD>>(),
      Shape<Int<kBlockN>, Int<kD>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kBlockM>, Int<kD>>(),
      Shape<Int<kBlockM>, Int<kD>, Int<kStages>>{}));
  using SmemLayoutQt = decltype(tile_to_shape(
      cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Elem, Int<kD>, Int<kBlockM>>(),
      Shape<Int<kD>, Int<kBlockM>, Int<kStages>>{}));
  using TMA_KV = TmaT<SmemLayoutKV, Shape<Int<kBlockN>, Int<kD>>>;
  using TMA_Q = TmaT<decltype(take<0, 2>(SmemLayoutQ{})), Shape<Int<kBlockM>, Int<kD>>>;
  using TMA_Qt = TmaT<decltype(take<0, 2>(SmemLayoutQt{})), Shape<Int<kD>, Int<kBlockM>>>;
  using Pipeline = cutlass::PipelineTmaAsyncNoCluster<kStages>;
  static constexpr uint32_t kBytesKV = kBlockN * kD;
  static constexpr uint32_t kBytesStage = 4 * kBlockM * kD + 2 * kBlockM * 4;

  struct SharedStorage {
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutKV>, 1024> k;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutKV>, 1024> v;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQ>, 1024> q;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQ>, 1024> dO;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQt>, 1024> qt;
    cute::array_aligned<Elem, cute::cosize_v<SmemLayoutQt>, 1024> dOt;
    alignas(128) float mrow[kStages][kBlockM];
    alignas(128) float delta[kStages][kBlockM];
    alignas(16) float qsc[5][kMaxGroup * kMaxNB];   // per (query head, 128-row block): c1, kv, c2, lkv, wq
    alignas(16) cutlass::arch::ClusterTransactionBarrier barKV;
    alignas(16) typename Pipeline::SharedStorage pipe;
  };
  struct Params {
    TMA_KV tma_k, tma_v;
    TMA_Q tma_q, tma_do;
    TMA_Qt tma_qt, tma_dot;
    int S, HQ, HKV, group;
    float scale_log2, scale;
    const float *sk, *sv, *sq, *sdo, *sqt, *sdot, *hmax, *mrow2, *delta;   // hmax [2, HQ]: per-head max of sqt, sdot
    __nv_bfloat16 *dk, *dv;
    int64_t dk_s, dk_h;
    float* res;
  };
};

template <class C, bool WriteRes>
__global__ void __launch_bounds__(384, 1) dkdv_kernel(CUTE_GRID_CONSTANT typename C::Params const p) {
  constexpr int kBlockM = C::kBlockM, kBlockN = C::kBlockN;
  using Pipeline = typename C::Pipeline;
  extern __shared__ __align__(1024) char smem_raw[];
  auto& ss = *reinterpret_cast<typename C::SharedStorage*>(smem_raw);

  const int n_block = blockIdx.x;   // heaviest first (n_block 0 sees every query block)
  const int g = blockIdx.y;
  const int NB = p.S / 128;
  const int m_start = n_block * kBlockN / kBlockM;
  const int m_cnt = p.S / kBlockM - m_start;
  const int iters = p.group * m_cnt;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int wg_idx = cutlass::canonical_warp_group_idx();
  const int lane_pred = cute::elect_one_sync();

  if (warp_idx == 0 && lane_pred) {
    cute::prefetch_tma_descriptor(p.tma_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_do.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_qt.get_tma_descriptor());
    cute::prefetch_tma_descriptor(p.tma_dot.get_tma_descriptor());
    ss.barKV.init(1);
  }
  typename Pipeline::Params pp;
  pp.transaction_bytes = C::kBytesStage;
  pp.role = wg_idx == 0 ? Pipeline::ThreadCategory::Producer : Pipeline::ThreadCategory::Consumer;
  pp.is_leader = (threadIdx.x % 128) == 0;
  pp.num_consumers = 256;
  Pipeline pipe(ss.pipe, pp, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sK = make_tensor(make_smem_ptr(ss.k.data()), typename C::SmemLayoutKV{});
  Tensor sV = make_tensor(make_smem_ptr(ss.v.data()), typename C::SmemLayoutKV{});
  Tensor sQ = make_tensor(make_smem_ptr(ss.q.data()), typename C::SmemLayoutQ{});
  Tensor sdO = make_tensor(make_smem_ptr(ss.dO.data()), typename C::SmemLayoutQ{});
  Tensor sQt = make_tensor(make_smem_ptr(ss.qt.data()), typename C::SmemLayoutQt{});
  Tensor sdOt = make_tensor(make_smem_ptr(ss.dOt.data()), typename C::SmemLayoutQt{});

  if (wg_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_idx == 0 && lane_pred) {
      Tensor mK = p.tma_k.get_tma_tensor(make_shape(p.S, int(kD), p.HKV));
      Tensor mV = p.tma_v.get_tma_tensor(make_shape(p.S, int(kD), p.HKV));
      Tensor gK = local_tile(mK(_, _, g), Shape<Int<kBlockN>, Int<kD>>{}, make_coord(n_block, _0{}));
      Tensor gV = local_tile(mV(_, _, g), Shape<Int<kBlockN>, Int<kD>>{}, make_coord(n_block, _0{}));
      auto [tKg, tKs] = tma_part(p.tma_k, gK, sK);
      auto [tVg, tVs] = tma_part(p.tma_v, gV, sV);
      using BarT = cutlass::arch::ClusterTransactionBarrier::ValueType;
      ss.barKV.arrive_and_expect_tx(2 * C::kBytesKV);
      copy(p.tma_k.with(reinterpret_cast<BarT&>(ss.barKV), 0), tKg, tKs);
      copy(p.tma_v.with(reinterpret_cast<BarT&>(ss.barKV), 0), tVg, tVs);
      Tensor mQ = p.tma_q.get_tma_tensor(make_shape(p.S, int(kD), p.HQ));
      Tensor mdO = p.tma_do.get_tma_tensor(make_shape(p.S, int(kD), p.HQ));
      Tensor mQt = p.tma_qt.get_tma_tensor(make_shape(int(kD), p.S, p.HQ));
      Tensor mdOt = p.tma_dot.get_tma_tensor(make_shape(int(kD), p.S, p.HQ));
      auto ws = cutlass::make_producer_start_state<Pipeline>();
      for (int hh = 0; hh < p.group; ++hh) {
        const int h = g * p.group + hh;
        Tensor gQ = local_tile(mQ(_, _, h), Shape<Int<kBlockM>, Int<kD>>{}, make_coord(_, _0{}));
        Tensor gdO = local_tile(mdO(_, _, h), Shape<Int<kBlockM>, Int<kD>>{}, make_coord(_, _0{}));
        Tensor gQt = local_tile(mQt(_, _, h), Shape<Int<kD>, Int<kBlockM>>{}, make_coord(_0{}, _));
        Tensor gdOt = local_tile(mdOt(_, _, h), Shape<Int<kD>, Int<kBlockM>>{}, make_coord(_0{}, _));
        auto [tQg, tQs] = tma_part(p.tma_q, gQ, sQ);
        auto [tdOg, tdOs] = tma_part(p.tma_do, gdO, sdO);
        auto [tQtg, tQts] = tma_part(p.tma_qt, gQt, sQt);
        auto [tdOtg, tdOts] = tma_part(p.tma_dot, gdOt, sdOt);
        for (int m = m_start; m < m_start + m_cnt; ++m) {
          pipe.producer_acquire(ws);
          auto* bar = pipe.producer_get_barrier(ws);
          const int s = ws.index();
          copy(p.tma_q.with(*bar, 0), tQg(_, m), tQs(_, s));
          copy(p.tma_do.with(*bar, 0), tdOg(_, m), tdOs(_, s));
          copy(p.tma_qt.with(*bar, 0), tQtg(_, m), tQts(_, s));
          copy(p.tma_dot.with(*bar, 0), tdOtg(_, m), tdOts(_, s));
          SM90_BULK_COPY_G2S::copy(p.mrow2 + (int64_t)h * p.S + m * kBlockM, bar, &ss.mrow[s][0], kBlockM * 4);
          SM90_BULK_COPY_G2S::copy(p.delta + (int64_t)h * p.S + m * kBlockM, bar, &ss.delta[s][0], kBlockM * 4);
          ++ws;
        }
      }
      pipe.producer_tail(ws);
    }
    return;
  }

  // ---------------------------------------------------------------- consumers
  cutlass::arch::warpgroup_reg_alloc<240>();
  const int tidx = threadIdx.x - 128;
  const int cwg = tidx / 128;
  typename C::TiledMmaS mmaS;
  typename C::TiledMmadKV mmadKV;
  Layout wgl = make_layout(make_shape(_2{}), make_stride(Int<128>{}));
  auto wgS = mmaS.get_slice(wgl(cwg));
  auto wgKV = mmadKV.get_slice(wgl(cwg));
  auto thrS = mmaS.get_thread_slice(tidx);
  auto thrKV = mmadKV.get_thread_slice(tidx);
  Tensor tSrK = wgS.partition_fragment_A(sK);
  Tensor tSrV = wgS.partition_fragment_A(sV);
  Tensor tSrQ = wgS.partition_fragment_B(sQ);
  Tensor tSrdO = wgS.partition_fragment_B(sdO);
  Tensor tKVrQt = wgKV.partition_fragment_B(sQt);
  Tensor tKVrdOt = wgKV.partition_fragment_B(sdOt);

  Tensor accK = partition_fragment_C(mmadKV, Shape<Int<kBlockN>, Int<kD>>{});
  Tensor accV = partition_fragment_C(mmadKV, Shape<Int<kBlockN>, Int<kD>>{});
  clear(accK);
  clear(accV);
  Tensor accK_rc = make_tensor(accK.data(), flash::convert_layout_acc_rowcol(accK.layout()));
  Tensor accV_rc = make_tensor(accV.data(), flash::convert_layout_acc_rowcol(accV.layout()));

  Tensor cS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
  Tensor tScS = thrS.partition_C(cS);
  Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int NR = decltype(size<0>(tScS_rc))::value;   // 2 key rows
  constexpr int NC = decltype(size<1>(tScS_rc))::value;   // kBlockM / 4 query cols
  static_assert(NR == 2);

  int krow[NR];
  float rr[NR];
#pragma unroll
  for (int mi = 0; mi < NR; ++mi) {
    krow[mi] = n_block * kBlockN + int(get<0>(tScS_rc(mi, 0)));
    rr[mi] = exp2f(kLr0);
  }
  const float sk = p.sk[g * NB + n_block];
  const float sv = p.sv[g * NB + n_block];
  float sqh = 1.f, sdoh = 1.f;   // units of the current head
  const float ck = sk * p.scale_log2;
  const int tcol0 = 2 * (threadIdx.x % 4);
  static_assert(NC == kBlockM / 4);

  // FP8DM_SCHED 1: strict ping-pong of GEMM issue between the 2 consumer WGs; 0: free running, WG1 delayed
  // by FP8DM_DELAY cycles at the start (de-phases the two WGs)
  auto sched_sync = [&]() { if constexpr (FP8DM_SCHED == 1) cutlass::arch::NamedBarrier::sync(256, 1 + cwg); };
  auto sched_arrive = [&]() { if constexpr (FP8DM_SCHED == 1) cutlass::arch::NamedBarrier::arrive(256, 1 + (1 - cwg)); };
  if constexpr (FP8DM_SCHED == 1) {
    if (cwg == 1) cutlass::arch::NamedBarrier::arrive(256, 1);   // WG0 goes first
  } else if (FP8DM_DELAY > 0 && cwg == 1) {
    const long long t0 = clock64();
    while (clock64() - t0 < FP8DM_DELAY) { }
  }

  for (int i = tidx; i < p.group * NB; i += 256) {
    const int hh_ = i / NB, b_ = i % NB, h_ = g * p.group + hh_, qb_ = h_ * NB + b_;
    const float kv_ = kFp8Max * p.sdot[qb_] / p.hmax[p.HQ + h_];
    ss.qsc[0][i] = ck * p.sq[qb_];
    ss.qsc[1][i] = kv_;
    ss.qsc[2][i] = sv * p.sdo[qb_] / kv_;
    ss.qsc[3][i] = __log2f(kv_);
    { const int h = h_; ss.qsc[4][i] = p.sqt[qb_] / FP8DM_SQH; }
  }
  cutlass::arch::NamedBarrier::sync(256, 3);
  ss.barKV.wait(0);
  typename Pipeline::PipelineState rs;
  Tensor tSrS = partition_fragment_C(mmaS, Shape<Int<kBlockN>, Int<kBlockM>>{});
  Tensor tdPrdP = partition_fragment_C(mmaS, Shape<Int<kBlockN>, Int<kBlockM>>{});
  Tensor S_rc = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
  Tensor dP_rc = make_tensor(tdPrdP.data(), flash::convert_layout_acc_rowcol(tdPrdP.layout()));
  int hh = 0, m = m_start;
#ifdef FP8DM_PROF
  long long pt_[7]; long long pr_[6] = {0, 0, 0, 0, 0, 0};
#endif
  CUTLASS_PRAGMA_NO_UNROLL
  for (int it = 0; it < iters; ++it) {
    const int stg = rs.index();
    const int h = g * p.group + hh;
    const int qi = hh * NB + (m * kBlockM) / 128;
    const float c1 = ss.qsc[0][qi];   // sk sq scale log2(e)
    const float kv = ss.qsc[1][qi];   // P' -> FP8 with row max 1 -> 448 sdot / sdoh
    const float c2 = ss.qsc[2][qi];   // sv sdo / kv
    const float lkv = ss.qsc[3][qi];
    const float wq = ss.qsc[4][qi];   // sqt / sqh
    FP8DM_T(0);
    {
      auto tok = pipe.consumer_try_wait(rs);
      pipe.consumer_wait(rs, tok);
    }
    FP8DM_T(1); FP8DM_ACC(0, 0, 1);
    sched_sync();
#ifndef FP8DM_X_NOG1
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(mmaS, tSrK, tSrQ(_, _, _, stg), tSrS);
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(mmaS, tSrV, tSrdO(_, _, _, stg), tdPrdP);
#else
    if (it == 0) { for (int i = 0; i < size(tSrS); ++i) { tSrS(i) = 0.01f * (i & 7); tdPrdP(i) = 0.02f * (i & 3); } }
#endif
    sched_arrive();
    FP8DM_T(2); FP8DM_ACC(1, 1, 2);
    warpgroup_wait<0>();
    FP8DM_T(3); FP8DM_ACC(2, 2, 3);
    if (m == m_start) {
      // new query head: per-head units (true dK = accK * sqh / rr, true dV = accV * sdoh / 448) and a fresh
      // lazy dS row scale
      const float sqh_new = FP8DM_SQH, sdoh_new = p.hmax[p.HQ + h];
      if (hh > 0) {
        const float fk = sqh / sqh_new, fv = sdoh / sdoh_new;
#pragma unroll
        for (int mi = 0; mi < NR; ++mi) {
          const float f = fk * (exp2f(kLr0) / rr[mi]);
#pragma unroll
          for (int c = 0; c < size<1>(accK_rc); ++c) { accK_rc(mi, c) *= f; accV_rc(mi, c) *= fv; }
          rr[mi] = exp2f(kLr0);
        }
      }
      sqh = sqh_new;
      sdoh = sdoh_new;
    }
    const float* smr = &ss.mrow[stg][0];
    const float* sdl = &ss.delta[stg][0];
    const bool diag = m * kBlockM < (n_block + 1) * kBlockN;
    float ad[NR] = {0.f, 0.f};
    float w[NR];
#pragma unroll
    for (int mi = 0; mi < NR; ++mi) w[mi] = wq * rr[mi];
    auto elem = [&](auto diag_c) {
      constexpr bool Diag = decltype(diag_c)::value;
#pragma unroll
    for (int ni = 0; ni < NC; ++ni) {
      const int lc = tcol0 + (ni & 1) + 8 * (ni >> 1);   // closed form of the wgmma C layout
      const float mr = smr[lc] - lkv;
      const float dl = sdl[lc] / kv;
      float colsum = 0.f;
#pragma unroll
      for (int mi = 0; mi < NR; ++mi) {
        float pv = FP8DM_EXP2(fmaf(S_rc(mi, ni), c1, -mr));          // P' * kv
        if constexpr (Diag) { if (m * kBlockM + lc < krow[mi]) pv = 0.f; }
        const float ds = pv * fmaf(dP_rc(mi, ni), c2, -dl);     // P' (dP - delta)
        if constexpr (WriteRes) colsum += ds;
        const float dsr = ds * w[mi];
        ad[mi] = fmaxf(ad[mi], fabsf(dsr));
        dP_rc(mi, ni) = dsr;
        S_rc(mi, ni) = pv;
      }
      if constexpr (WriteRes) {
        colsum += __shfl_xor_sync(0xffffffff, colsum, 4);
        colsum += __shfl_xor_sync(0xffffffff, colsum, 8);
        colsum += __shfl_xor_sync(0xffffffff, colsum, 16);
        if ((tidx & 31) < 4) atomicAdd(p.res + (int64_t)h * p.S + m * kBlockM + lc, colsum);
      }
    }
    };
    if (diag) elem(cute::true_type{}); else elem(cute::false_type{});
#pragma unroll
    for (int mi = 0; mi < NR; ++mi) {
      ad[mi] = quad_max(ad[mi]);
      if (ad[mi] > kFp8Max) {                     // shrink the dS scale of this key row: max -> 448
        const float f = kFp8Max / ad[mi];
        rr[mi] *= f;
#pragma unroll
        for (int ni = 0; ni < NC; ++ni) dP_rc(mi, ni) *= f;
#pragma unroll
        for (int c = 0; c < size<1>(accK_rc); ++c) accK_rc(mi, c) *= f;
      }
    }
    flash::permute_Cregs_fp8(tSrS);
    Tensor tP_a = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<typename C::TiledMmadKV>(tSrS.layout()));
    Tensor rP = make_tensor_like<Elem>(tP_a);
#ifndef FP8DM_X_NOCVT
    flash::convert_type_out(tP_a, rP);
#else
    fake_cvt(tP_a, rP);
#endif
    flash::permute_Cregs_fp8(tdPrdP);
    Tensor tdS_a = make_tensor(tdPrdP.data(), flash::convert_layout_acc_Aregs<typename C::TiledMmadKV>(tdPrdP.layout()));
    Tensor rdS = make_tensor_like<Elem>(tdS_a);
#ifndef FP8DM_X_NOCVT
    flash::convert_type_out(tdS_a, rdS);
#else
    fake_cvt(tdS_a, rdS);
#endif
    if (++m == m_start + m_cnt) { m = m_start; ++hh; }
    FP8DM_T(4); FP8DM_ACC(3, 3, 4);
    sched_sync();
#ifndef FP8DM_X_NOG2
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(mmadKV, rP, tKVrdOt(_, _, _, stg), accV);
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(mmadKV, rdS, tKVrQt(_, _, _, stg), accK);
#else
    { auto x = recast<uint32_t>(rP); auto y = recast<uint32_t>(rdS); accV(0) += __uint_as_float(x(0) ^ x(3) ^ y(1)); }
#endif
    sched_arrive();
    FP8DM_T(5); FP8DM_ACC(4, 4, 5);
    warpgroup_wait<0>();
    pipe.consumer_release(rs);
    FP8DM_T(6); FP8DM_ACC(5, 5, 6);
    ++rs;
  }

#ifdef FP8DM_PROF
  if (blockIdx.x < 2 && blockIdx.y < 1 && (tidx % 128) == 0)
    printf("PROF dkdv n_block=%d g=%d wg=%d it=%d  wait_data=%lld turn1=%lld gemm1=%lld alu=%lld turn2=%lld gemm2=%lld total=%lld\n", n_block, g, cwg, iters, pr_[0]/iters, pr_[1]/iters, pr_[2]/iters, pr_[3]/iters, pr_[4]/iters, pr_[5]/iters, (pr_[0]+pr_[1]+pr_[2]+pr_[3]+pr_[4]+pr_[5])/iters);
#endif
  // ---------------------------------------------------------------- epilogue
  Tensor cO = make_identity_tensor(Shape<Int<kBlockN>, Int<kD>>{});
  Tensor tOcO = thrKV.partition_C(cO);
  Tensor tOcO_rc = make_tensor(tOcO.data(), flash::convert_layout_acc_rowcol(tOcO.layout()));
  const float cv = sdoh / kFp8Max;
#pragma unroll
  for (int mi = 0; mi < NR; ++mi) {
    const float ckk = p.scale * sqh / rr[mi];
    const int row = n_block * kBlockN + int(get<0>(tOcO_rc(mi, 0)));
    __nv_bfloat16* outk = p.dk + (int64_t)row * p.dk_s + (int64_t)g * p.dk_h;
    __nv_bfloat16* outv = p.dv + (int64_t)row * p.dk_s + (int64_t)g * p.dk_h;
#pragma unroll
    for (int c = 0; c < size<1>(accK_rc); c += 2) {
      const int col = int(get<1>(tOcO_rc(mi, c)));
      *reinterpret_cast<__nv_bfloat162*>(outk + col) = __floats2bfloat162_rn(accK_rc(mi, c) * ckk, accK_rc(mi, c + 1) * ckk);
      *reinterpret_cast<__nv_bfloat162*>(outv + col) = __floats2bfloat162_rn(accV_rc(mi, c) * cv, accV_rc(mi, c + 1) * cv);
    }
  }
}

template <int kBlockM, int kStages, bool WriteRes>
void run_dkdv(const DkvArgs& a, cudaStream_t st) {
  using C = DkvCfg<kBlockM, kStages>;
  using LQ = decltype(take<0, 2>(typename C::SmemLayoutQ{}));
  using LQt = decltype(take<0, 2>(typename C::SmemLayoutQt{}));
  typename C::Params p{
      make_tma<typename C::SmemLayoutKV, Shape<Int<C::kBlockN>, Int<kD>>>(a.k8, a.S, kD, a.HKV),
      make_tma<typename C::SmemLayoutKV, Shape<Int<C::kBlockN>, Int<kD>>>(a.v8, a.S, kD, a.HKV),
      make_tma<LQ, Shape<Int<kBlockM>, Int<kD>>>(a.q8, a.S, kD, a.HQ),
      make_tma<LQ, Shape<Int<kBlockM>, Int<kD>>>(a.do8, a.S, kD, a.HQ),
      make_tma<LQt, Shape<Int<kD>, Int<kBlockM>>>(a.q8t, kD, a.S, a.HQ),
      make_tma<LQt, Shape<Int<kD>, Int<kBlockM>>>(a.do8t, kD, a.S, a.HQ),
      a.S, a.HQ, a.HKV, a.HQ / a.HKV, float(a.scale * M_LOG2E), a.scale,
      a.sk, a.sv, a.sq, a.sdo, a.sqt, a.sdot, a.hmax, a.mrow2, a.delta,
      static_cast<__nv_bfloat16*>(a.dk), static_cast<__nv_bfloat16*>(a.dv), a.dk_s, a.dk_h, a.res};
  auto kern = dkdv_kernel<C, WriteRes>;
  const int smem = sizeof(typename C::SharedStorage);
  FP8DM_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kern<<<dim3(a.S / C::kBlockN, a.HKV), 384, smem, st>>>(p);
  FP8DM_CHECK(cudaGetLastError());
}

void dkdv(const DkvArgs& a, cudaStream_t st) {
  if (a.S / 128 > kMaxNB || a.S % 128 || a.HQ / a.HKV > kMaxGroup) { fprintf(stderr, "fp8dm::dkdv: unsupported shape\n"); return; }
  if (a.res) run_dkdv<64, 3, true>(a, st);
  else run_dkdv<64, 3, false>(a, st);
}

}  // namespace fp8dm
