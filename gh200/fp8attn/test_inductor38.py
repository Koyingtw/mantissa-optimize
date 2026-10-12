"""容器的 torch 2.7 inductor（torch.compile）能否搭配 Triton 3.8 運作（D 變體的前提）。

Triton 3.8 把 triton_key 從 triton.compiler.compiler 移到 triton.runtime.cache；先補上這個別名，
再 torch.compile 幾個訓練中會出現的融合運算（bias+dropout+add、SwiGLU、RMSNorm 類 reduction），
forward + backward 對照 eager。
"""
import torch
import triton
import triton.compiler.compiler as _cc

if not hasattr(_cc, "triton_key"):
    from triton.runtime.cache import triton_key
    _cc.triton_key = triton_key
    print("shim: triton.compiler.compiler.triton_key -> triton.runtime.cache.triton_key")
print(f"torch {torch.__version__}  triton {triton.__version__} ({triton.__file__})", flush=True)

dev = "cuda"
torch.manual_seed(0)


def bias_dropout_add(x, bias, residual, p: float, training: bool):
    return residual + torch.nn.functional.dropout(x + bias, p=p, training=training)


def swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def rmsnorm(x, w, eps: float = 1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w


cases = {
    "bias_dropout_add(p=0)": (bias_dropout_add, lambda: (torch.randn(4096, 8192, device=dev, dtype=torch.bfloat16),
                                                          torch.randn(8192, device=dev, dtype=torch.bfloat16),
                                                          torch.randn(4096, 8192, device=dev, dtype=torch.bfloat16)),
                              dict(p=0.0, training=True)),
    "swiglu": (swiglu, lambda: (torch.randn(4096, 2 * 8192, device=dev, dtype=torch.bfloat16),), {}),
    "rmsnorm": (rmsnorm, lambda: (torch.randn(4096, 8192, device=dev, dtype=torch.bfloat16),
                                  torch.randn(8192, device=dev, dtype=torch.bfloat16)), {}),
}
ok = True
for name, (fn, mk, kw) in cases.items():
    try:
        args = [a.requires_grad_() for a in mk()]
        ref_args = [a.detach().clone().requires_grad_() for a in args]
        out = torch.compile(fn)(*args, **kw)
        ref = fn(*ref_args, **kw)
        g = torch.randn_like(ref)
        out.backward(g); ref.backward(g)
        err = max([((out - ref).float().norm() / ref.float().norm()).item()] +
                  [((a.grad - r.grad).float().norm() / r.grad.float().norm()).item() for a, r in zip(args, ref_args)])
        good = err < 2e-2
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {name}: max rel err fwd/bwd = {err:.2e}", flush=True)
    except Exception as e:  # noqa
        ok = False
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
print("ALL PASS" if ok else "SOME FAILED")
