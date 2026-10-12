"""除錯：CUDA kernel 的 dV 與「同一組 FP8 運算元、FP64 累加」的模擬比較，判斷誤差來源（累加精度或量化方式）。"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fp8attn2 as fa  # noqa
import fp8dm_cuda as fc  # noqa

S, HQ, HKV, D = int(os.environ.get("S", 2048)), 8, 1, 128
G = HQ // HKV
scale = D ** -0.5
torch.manual_seed(0)
q, k, v, do = (torch.randn(S, h, D, device="cuda").to(torch.bfloat16) for h in (HQ, HKV, HKV, HQ))
o, lse, mrow = fa.fp8_attn_fwd(q, k, v, scale, cfg=fa.CFG)
E = fc.ext()
q8, _, sq, _ = E.quant(q, True, False, None, None, None)
k8, k8t, sk, _ = E.quant(k, True, True, None, None, None)
v8, _, sv, _ = E.quant(v, True, False, None, None, None)
do8, _, sdo, d0 = E.quant(do, True, False, None, o, None)
skmax = sk.amax(1).contiguous()
dq, delta, mrow2 = E.dq(q8, do8, k8, v8, k8t, sq, sdo, sk, sv, skmax, lse, d0, None, scale, True)
pmax = torch.exp2(mrow2 - lse)
_, q8t, sqt, _ = E.quant(q, False, True, pmax, None, None)
_, do8t, sdot, _ = E.quant(do, False, True, pmax, None, None)
hmax = torch.stack([sqt.amax(1), sdot.amax(1)]).contiguous()
cdo = sdot.amax().view(1)
dk, dv, _ = E.dkdv(k8, v8, q8, do8, q8t, do8t, sk, sv, sq, sdo, sqt, sdot, hmax, mrow2, delta, scale, False)

perm = torch.tensor([2 * (a >> 2) + (a & 1) + 8 * ((a >> 1) & 1) for a in range(16)], device="cuda")
idx = (torch.arange(S, device="cuda") // 16 * 16).view(-1, 16) + perm  # position a -> actual row
idx = idx.view(-1)
inv_idx = torch.empty_like(idx); inv_idx[idx] = torch.arange(S, device="cuda")


def deq_rows(x8, sc):            # x8 [H,S,D], sc [H,S/128] -> fp64 [H,S,D]
    return x8.double() * sc.double().repeat_interleave(128, 1)[..., None]


def deq_t(x8t, sc):              # x8t [H,D,S] permuted -> fp64 [H,S,D] actual order
    x = x8t.double()[:, :, inv_idx]  # actual row r sits at position inv_idx[r]
    return x.transpose(1, 2) * sc.double().repeat_interleave(128, 1)[..., None]


K = deq_rows(k8, sk)[0]
Qd = deq_rows(q8, sq)
dOt = deq_t(do8t, sdot)          # = dO * pmax (quantized)
dOt_raw = deq_t(do8t, torch.ones_like(sdot))
mask = torch.ones(S, S, device="cuda", dtype=torch.bool).tril_()
dv_emu = torch.zeros(S, D, device="cuda", dtype=torch.float64)
dv_exact_p = torch.zeros_like(dv_emu)
dv_T = torch.zeros_like(dv_emu); dv_R = torch.zeros_like(dv_emu)
sdoh = sdot.amax(1)
for h in range(HQ):
    s = (Qd[h] @ K.T) * (scale * 1.4426950408889634)
    pp = torch.exp2(s - mrow2[h].double()[:, None]).masked_fill(~mask, 0)   # P' [q, k]
    kv = 256.0 * sdot[h].double().repeat_interleave(128)[:, None] / cdo[0].double()  # per query row
    pv = (pp * kv).float().to(torch.float8_e4m3fn).double()
    dv_emu += pv.T @ dOt_raw[h] * (cdo[0].double() / 256.0)
    dv_exact_p += pp.T @ dOt[h]
    kvT = 256.0 * sdot[h].double().repeat_interleave(128)[:, None] / sdoh[h].double()
    dv_T += (pp * kvT).float().to(torch.float8_e4m3fn).double().T @ dOt_raw[h] * (sdoh[h].double() / 256.0)
    dv_R += (pp * 256).float().to(torch.float8_e4m3fn).double().T @ dOt[h] / 256.0
# FP64 truth
ref = torch.zeros(S, D, device="cuda", dtype=torch.float64)
for h in range(HQ):
    s = (q[:, h].double() @ k[:, 0].double().T) * scale
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
    ref += p.T @ do[:, h].double()


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


print(f"S={S}")
print(f"kernel dV vs emulation (same FP8 operands, fp64 acc): {rel(dv[:, 0], dv_emu):.3e}")
print(f"kernel dV vs FP64 truth: {rel(dv[:, 0], ref):.3e}")
print(f"emulation vs truth: {rel(dv_emu, ref):.3e}")
print(f"Triton-style kv (per-head sdoh) emu vs truth: {rel(dv_T, ref):.3e}   kv=256 (scale after GEMM) emu vs truth: {rel(dv_R, ref):.3e}")
print(f"unquantized-P' emulation vs truth (only dO' quantized): {rel(dv_exact_p, ref):.3e}")
cfg = dict(fa.CFG, DS_SCALE=0)
_, _, dvt = fa.fp8_attn_bwd(q, k, v, o, do, lse, mrow, sm_scale=scale, matched=True, cfg=cfg)
print(f"triton dV vs truth: {rel(dvt[:, 0], ref):.3e}   triton vs emulation: {rel(dvt[:, 0], dv_emu):.3e}")
