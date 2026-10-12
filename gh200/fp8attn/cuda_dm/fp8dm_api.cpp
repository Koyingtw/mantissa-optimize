// PyTorch binding for the CUDA FP8 Delta-Matching attention backward (fp8dm_bwd.cu).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fp8dm_bwd.h"

namespace {

#define CHECK_CUDA_T(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

const float* fptr(const c10::optional<at::Tensor>& t) { return t.has_value() ? t->data_ptr<float>() : nullptr; }

// x [S, H, 128] bf16 (stride(2) == 1).  Returns (x8 [H,S,128] or empty, x8t [H,128,S] or empty, sc [H,S/128], delta0 or empty)
std::vector<at::Tensor> quant(at::Tensor x, bool write_n, bool write_t, c10::optional<at::Tensor> rowmul,
                              c10::optional<at::Tensor> o, c10::optional<at::Tensor> fixed_sc) {
  CHECK_CUDA_T(x);
  TORCH_CHECK(x.dim() == 3 && x.size(2) == 128 && x.stride(2) == 1 && x.scalar_type() == at::kBFloat16);
  const int S = x.size(0), H = x.size(1);
  TORCH_CHECK(S % 128 == 0);
  TORCH_CHECK(x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0, "16-byte aligned rows required");
  c10::cuda::CUDAGuard guard(x.device());
  auto opt8 = x.options().dtype(at::kFloat8_e4m3fn);
  auto optf = x.options().dtype(at::kFloat);
  at::Tensor x8 = write_n ? at::empty({H, S, 128}, opt8) : at::Tensor();
  at::Tensor x8t = write_t ? at::empty({H, 128, S}, opt8) : at::Tensor();
  at::Tensor sc = at::empty({H, S / 128}, optf);
  at::Tensor d0 = o.has_value() ? at::empty({H, S}, optf) : at::Tensor();
  if (rowmul.has_value()) { CHECK_CONTIG(*rowmul); TORCH_CHECK(rowmul->numel() == (int64_t)H * S); }
  if (o.has_value()) { TORCH_CHECK(o->stride(2) == 1 && o->sizes() == x.sizes() && o->scalar_type() == at::kBFloat16); }
  if (fixed_sc.has_value()) { CHECK_CONTIG(*fixed_sc); TORCH_CHECK(fixed_sc->numel() == H); }
  fp8dm::quant(x.data_ptr(), x.stride(0), x.stride(1), fptr(rowmul), fptr(fixed_sc),
               o.has_value() ? o->data_ptr() : nullptr, o.has_value() ? o->stride(0) : 0, o.has_value() ? o->stride(1) : 0,
               write_n ? x8.data_ptr() : nullptr, write_t ? x8t.data_ptr() : nullptr, sc.data_ptr<float>(),
               o.has_value() ? d0.data_ptr<float>() : nullptr, S, H, at::cuda::getCurrentCUDAStream());
  return {x8, x8t, sc, d0};
}

// returns dq [S, HQ, 128] bf16, delta [HQ, S], mrow2 [HQ, S]
std::vector<at::Tensor> dq(at::Tensor q8, at::Tensor do8, at::Tensor k8, at::Tensor v8, at::Tensor k8t,
                           at::Tensor sq, at::Tensor sdo, at::Tensor sk, at::Tensor sv, at::Tensor skmax,
                           at::Tensor lse2, at::Tensor delta0, c10::optional<at::Tensor> mrow, double scale, bool matched) {
  if (mrow.has_value()) { CHECK_CUDA_T(*mrow); CHECK_CONTIG(*mrow); }
  for (auto* t : {&q8, &do8, &k8, &v8, &k8t, &sq, &sdo, &sk, &sv, &skmax, &lse2, &delta0}) { CHECK_CUDA_T(*t); CHECK_CONTIG(*t); }
  const int HQ = q8.size(0), S = q8.size(1), HKV = k8.size(0);
  c10::cuda::CUDAGuard guard(q8.device());
  auto dqo = at::empty({S, HQ, 128}, q8.options().dtype(at::kBFloat16));
  auto delta = at::empty({HQ, S}, q8.options().dtype(at::kFloat));
  auto mrow2 = at::empty({HQ, S}, q8.options().dtype(at::kFloat));
  fp8dm::DqArgs a{q8.data_ptr(), do8.data_ptr(), k8.data_ptr(), v8.data_ptr(), k8t.data_ptr(),
                  sq.data_ptr<float>(), sdo.data_ptr<float>(), sk.data_ptr<float>(), sv.data_ptr<float>(),
                  skmax.data_ptr<float>(), lse2.data_ptr<float>(), delta0.data_ptr<float>(), fptr(mrow),
                  dqo.data_ptr(), dqo.stride(0), dqo.stride(1), delta.data_ptr<float>(), mrow2.data_ptr<float>(),
                  S, HQ, HKV, float(scale), matched};
  fp8dm::dq(a, at::cuda::getCurrentCUDAStream());
  return {dqo, delta, mrow2};
}

// returns dk, dv [S, HKV, 128] bf16 (and res [HQ,S] if want_res)
std::vector<at::Tensor> dkdv(at::Tensor k8, at::Tensor v8, at::Tensor q8, at::Tensor do8, at::Tensor q8t, at::Tensor do8t,
                             at::Tensor sk, at::Tensor sv, at::Tensor sq, at::Tensor sdo, at::Tensor sqt, at::Tensor sdot,
                             at::Tensor hmax, at::Tensor mrow2, at::Tensor delta, double scale, bool want_res) {
  for (auto* t : {&k8, &v8, &q8, &do8, &q8t, &do8t, &sk, &sv, &sq, &sdo, &sqt, &sdot, &hmax, &mrow2, &delta}) {
    CHECK_CUDA_T(*t); CHECK_CONTIG(*t);
  }
  const int HQ = q8.size(0), S = q8.size(1), HKV = k8.size(0);
  c10::cuda::CUDAGuard guard(q8.device());
  auto dk = at::empty({S, HKV, 128}, q8.options().dtype(at::kBFloat16));
  auto dv = at::empty({S, HKV, 128}, q8.options().dtype(at::kBFloat16));
  at::Tensor res = want_res ? at::zeros({HQ, S}, q8.options().dtype(at::kFloat)) : at::Tensor();
  fp8dm::DkvArgs a{k8.data_ptr(), v8.data_ptr(), q8.data_ptr(), do8.data_ptr(), q8t.data_ptr(), do8t.data_ptr(),
                   sk.data_ptr<float>(), sv.data_ptr<float>(), sq.data_ptr<float>(), sdo.data_ptr<float>(),
                   sqt.data_ptr<float>(), sdot.data_ptr<float>(), hmax.data_ptr<float>(), mrow2.data_ptr<float>(),
                   delta.data_ptr<float>(), dk.data_ptr(), dv.data_ptr(), dk.stride(0), dk.stride(1),
                   want_res ? res.data_ptr<float>() : nullptr, S, HQ, HKV, float(scale)};
  fp8dm::dkdv(a, at::cuda::getCurrentCUDAStream());
  return {dk, dv, res};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant", &quant, "E4M3 block quantization (+ permuted transposed copy, delta0)");
  m.def("dq", &dq, "FP8 dQ kernel with Delta-Matching");
  m.def("dkdv", &dkdv, "FP8 dK/dV kernel");
}
