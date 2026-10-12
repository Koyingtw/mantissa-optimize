"""CUDA (CUTLASS/CuTe, Hopper) FP8 attention backward with Delta-Matching.

Python front end for fp8dm_bwd.cu.  The extension is JIT-built once into $FP8DM_BUILD
(default /nfs/home/koying/fp8attn-build/cuda_dm); later imports load the prebuilt .so directly
(FP8DM_NO_BUILD=1 forbids building, e.g. inside training jobs).

Interface (same as fp8attn2.fp8_attn_bwd so the existing tests can use it):
    fp8_attn_bwd(q, k, v, o, do, lse, mrow=None, sm_scale=None, matched=True, cfg=None, return_res=False)
q/o/do [S, HQ, 128] bf16, k/v [S, HKV, 128] bf16 (stride(-1) == 1, rows 16-byte aligned),
lse [HQ, S] fp32 in the log2 domain of the scaled scores (as written by fp8attn2.fp8_attn_fwd:
P = exp2(S * sm_scale * log2(e) - lse)).  mrow (forward row max of the log2 scores, as fp8attn2 returns it) is optional; if given, P's
FP8 scale is known up front (cheaper dq kernel).  cfg is ignored.
return_res=True additionally returns (res, delta, delta0): res[h, i] = sum_j dS_ij as recomputed
by the dK/dV kernel (the dS that actually feeds dK), delta = matched delta, delta0 = <dO, O>.
"""
import os
import sys
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("FP8DM_BUILD", "/nfs/home/koying/fp8attn-build/cuda_dm")
FA3 = os.environ.get("FP8DM_FA3", "/nfs/home/koying/fp8attn-build/fa3src")
_EXT = None
USE_MROW = os.environ.get("FP8DM_USE_MROW", "1") == "1"   # use the forward row max if given
HEAD_SCALE_T = os.environ.get("FP8DM_HEAD_SCALE_T", "0") == "1"   # 0 = per-(head,128-row) scales (better on real data)


def ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if os.environ.get("FP8DM_NO_BUILD", "0") == "1":
        if BUILD not in sys.path:
            sys.path.insert(0, BUILD)
        import fp8dm_cuda_ext  # noqa
        _EXT = fp8dm_cuda_ext
        return _EXT
    from torch.utils.cpp_extension import load
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.makedirs(BUILD, exist_ok=True)
    _EXT = load(
        name="fp8dm_cuda_ext",
        sources=[f"{SRC}/fp8dm_api.cpp", f"{SRC}/fp8dm_bwd.cu"],
        build_directory=BUILD,
        extra_include_paths=[SRC, f"{FA3}/hopper", f"{FA3}/csrc/cutlass/include"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                           "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                           "-DNDEBUG", "-lineinfo", "--resource-usage",
                           "-gencode", "arch=compute_90a,code=sm_90a"],
        verbose=os.environ.get("FP8DM_VERBOSE", "0") == "1",
    )
    return _EXT


def fp8_attn_bwd(q, k, v, o, do, lse, mrow=None, sm_scale=None, matched=True, cfg=None, return_res=False):
    S, HQ, D = q.shape
    HKV = k.shape[1]
    G = HQ // HKV
    assert D == 128 and S % 128 == 0 and HQ % HKV == 0
    sm_scale = sm_scale or D ** -0.5
    E = ext()
    lse = lse.contiguous()
    q8, _, sq, _ = E.quant(q, True, False, None, None, None)
    k8, k8t, sk, _ = E.quant(k, True, True, None, None, None)
    v8, _, sv, _ = E.quant(v, True, False, None, None, None)
    do8, _, sdo, d0 = E.quant(do, True, False, None, o, None)
    skmax = sk.amax(1).contiguous()
    mr = mrow.contiguous() if (mrow is not None and USE_MROW) else None
    dq, delta, mrow2 = E.dq(q8, do8, k8, v8, k8t, sq, sdo, sk, sv, skmax, lse, d0, mr, float(sm_scale), bool(matched))
    pmax = torch.exp2(mrow2 - lse)
    # Q' = Q * Pmax, dO' = dO * Pmax (transposed copies for dK / dV) with ONE scale per head: an upper
    # bound of amax(x * Pmax) from the block amax of x and the block max of Pmax (the rows stay in the
    # E4M3 normal range); the dK / dV kernels then quantize P' (row max 1) and dS' with a constant factor.
    if HEAD_SCALE_T:
        pmb = pmax.view(HQ, S // 128, 128).amax(2)
        fq, fdo = (sq * pmb).amax(1).contiguous(), (sdo * pmb).amax(1).contiguous()
    else:   # one scale per (head, 128 rows) block
        fq = fdo = None
    _, q8t, sqt, _ = E.quant(q, False, True, pmax, None, fq)
    _, do8t, sdot, _ = E.quant(do, False, True, pmax, None, fdo)
    hmax = torch.stack([sqt.amax(1), sdot.amax(1)]).contiguous()
    dk, dv, res = E.dkdv(k8, v8, q8, do8, q8t, do8t, sk, sv, sq, sdo, sqt, sdot, hmax, mrow2, delta,
                         float(sm_scale), bool(return_res))
    if return_res:
        return dq, dk, dv, res * pmax, delta, d0
    return dq, dk, dv
