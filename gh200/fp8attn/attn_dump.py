"""Dump real attention operands/gradients from the MLPerf Llama2-70B LoRA run.

Imported by the patched train.py of the *separate* work copy
(gh200/fp8attn/prepare_work.sh -> /nfs/home/koying/mlperf-fp8attn-work/nvidia_impl);
the production nvidia_impl is never touched. Enabled only when FP8ATTN_DUMP=1.

For each selected decoder layer (Megatron `TEDotProductAttention`, which calls
TE 2.2 `DotProductAttention` -> `FusedAttnFunc`, cuDNN FP8 fwd + FP8 bwd when
NVTE_FP8_DPA_BWD=1) and each selected training step it saves, on rank 0:

  q, k, v   inputs of core attention (post-RoPE, bf16)          [S, H, D]
  o         output of core attention (bf16, = dequant(O_fp8))   [S, Hq, D]
  do        gradient arriving at the attention output (bf16)    [S, Hq, D]
  dq,dk,dv  gradients TE returned for q, k, v (bf16, dequantised
            from cuDNN's E5M2 output)                            [S, H, D]
  meta      dict: layer, step, softmax_scale, dropout, mask type, recipe, and the
            DPA module's FP8 delayed-scaling state (scale / amax_history for the
            forward and backward slots, captured at fwd, before bwd and after bwd).

Layout: Megatron passes sbhd tensors with b=1; the batch dim is squeezed, so every
tensor is [seq=8192, heads, head_dim=128] in natural (contiguous) order.

Env:
  FP8ATTN_DUMP=1                  enable
  FP8ATTN_DUMP_DIR=/dumps         output directory (one .pt per layer/step)
  FP8ATTN_DUMP_LAYERS=0,20,40,79  0-based decoder layer indices
  FP8ATTN_DUMP_STEPS=1,2,6        1-based training step (count of grad-enabled forwards)
"""
import os
import time

import torch

_LOG = "fp8attn_dump:"


def _ints(s):
    return {int(x) for x in s.split(",") if x.strip()}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _fp8_state(mod):
    """Copy of the DPA module's delayed-scaling state (small tensors)."""
    out = {}
    meta = getattr(mod, "fp8_meta", None)
    if not meta:
        return out
    for key in ("scaling_fwd", "scaling_bwd"):
        st = meta.get(key)
        if st is None:
            continue
        d = {}
        for attr in ("scale", "amax_history"):
            t = getattr(st, attr, None)
            if isinstance(t, torch.Tensor):
                d[attr] = t.detach().float().cpu().clone()
        out[key] = d
    return out


def _recipe(mod):
    meta = getattr(mod, "fp8_meta", None) or {}
    r = meta.get("recipe")
    if r is None:
        return None
    keys = ("fp8_format", "margin", "amax_history_len", "amax_compute_algo", "fp8_dpa", "fp8_mha",
            "reduce_amax", "interval")
    return {k: str(getattr(r, k, None)) for k in keys} | {"repr": repr(r)}


def _meta_indices():
    try:
        from transformer_engine.pytorch.cpp_extensions import fused_attn as fa
        return {n: int(getattr(fa, n)) for n in
                ("META_QKV", "META_O", "META_S", "META_DQKV", "META_DO", "META_DP")}
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}


def _sq(t):
    """[S, 1, ...] -> [S, ...], detached CPU copy in the native dtype."""
    t = t.detach()
    if t.dim() >= 3 and t.shape[1] == 1:
        t = t[:, 0]
    return t.contiguous().cpu().clone()


class _Dumper:
    def __init__(self, layer, mod, steps, outdir):
        self.layer, self.mod, self.steps, self.outdir = layer, mod, steps, outdir
        self.count = 0
        self.cur = None

    def pre(self, mod, args, kwargs):
        if not (mod.training and torch.is_grad_enabled()):
            return
        if getattr(_TRAINER.get("t"), "warmup", False):  # WARMUP=True: synthetic warmup steps are not counted
            return
        self.count += 1
        if self.count not in self.steps:
            return
        names = ("query", "key", "value", "attention_mask", "attn_mask_type")
        vals = list(args) + [None] * 5
        get = lambda i: kwargs.get(names[i], vals[i])
        q, k, v = get(0), get(1), get(2)
        rec = {
            "q": _sq(q), "k": _sq(k), "v": _sq(v),
            "meta": {
                "layer": self.layer, "step": self.count,
                "raw_shapes": {"q": tuple(q.shape), "k": tuple(k.shape), "v": tuple(v.shape)},
                "attention_mask_is_none": get(3) is None,
                "attn_mask_type": str(get(4)),
                "packed_seq_params": repr(kwargs.get("packed_seq_params")),
                "softmax_scale": getattr(mod, "softmax_scale", None),
                "attention_dropout": getattr(mod, "attention_dropout", None),
                "attention_dropout_p": getattr(getattr(mod, "attention_dropout_ctx", None), "p", None),
                "qkv_format_megatron": getattr(mod, "qkv_format", None),
                "num_attention_heads": getattr(mod, "num_attention_heads", None),
                "num_gqa_groups": getattr(mod, "num_gqa_groups", None),
                "NVTE_FP8_DPA_BWD": os.environ.get("NVTE_FP8_DPA_BWD"),
                "recipe": _recipe(mod),
                "meta_indices": _meta_indices(),
            },
            "pending": {"do", "dq", "dk", "dv"},
        }
        self.cur = rec
        for name, t in (("dq", q), ("dk", k), ("dv", v)):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(self._grad_hook(rec, name))
            else:
                rec["pending"].discard(name)
                rec["meta"][f"{name}_missing"] = True

    def post(self, mod, args, kwargs, out):
        rec = self.cur
        if rec is None or rec["meta"]["step"] != self.count or "o" in rec:
            return
        o = out[0] if isinstance(out, (tuple, list)) else out
        rec["meta"]["o_raw_shape"] = tuple(o.shape)
        on = _sq(o)
        rec["o"] = on.view(on.shape[0], -1, rec["q"].shape[-1])
        rec["meta"]["fp8_state_fwd"] = _fp8_state(mod)
        if o.requires_grad:
            o.register_hook(self._grad_hook(rec, "do"))
        else:
            rec["pending"].discard("do")

    def _grad_hook(self, rec, name):
        def hook(g):
            with torch.no_grad():
                t = _sq(g)
                if name == "do":
                    t = t.view(t.shape[0], -1, rec["q"].shape[-1])
                    rec["meta"]["fp8_state_before_bwd"] = _fp8_state(self.mod)
                rec[name] = t
                rec["pending"].discard(name)
                if not rec["pending"]:
                    rec["meta"]["fp8_state_after_bwd"] = _fp8_state(self.mod)
                    self._save(rec)
            return None
        return hook

    def _save(self, rec):
        rec.pop("pending", None)
        path = os.path.join(self.outdir, f"L{self.layer:02d}_step{rec['meta']['step']}.pt")
        t0 = time.time()
        torch.save(rec, path)
        print(f"{_LOG} saved {path} ({time.time() - t0:.1f}s) "
              f"q{tuple(rec['q'].shape)} k{tuple(rec['k'].shape)} o{tuple(rec['o'].shape)}", flush=True)
        if self.cur is rec:
            self.cur = None


_TRAINER = {}


def install(pl_module):
    if os.environ.get("FP8ATTN_DUMP", "0") != "1" or _rank() != 0:
        return 0
    outdir = os.environ.get("FP8ATTN_DUMP_DIR", "/dumps")
    os.makedirs(outdir, exist_ok=True)
    layers = _ints(os.environ.get("FP8ATTN_DUMP_LAYERS", "0,20,40,79"))
    steps = _ints(os.environ.get("FP8ATTN_DUMP_STEPS", "1,2,6"))
    n = 0
    for name, mod in pl_module.named_modules():
        if type(mod).__name__ != "TEDotProductAttention" or ".layers." not in name:
            continue
        idx = int(name.split(".layers.")[1].split(".")[0])
        if idx not in layers:
            continue
        d = _Dumper(idx, mod, steps, outdir)
        mod.register_forward_pre_hook(d.pre, with_kwargs=True)
        mod.register_forward_hook(d.post, with_kwargs=True)
        n += 1
        print(f"{_LOG} hooked {name} ({type(mod).__module__}.{type(mod).__name__})", flush=True)
    print(f"{_LOG} {n} modules hooked, layers={sorted(layers)} steps={sorted(steps)} -> {outdir}", flush=True)
    return n


def make_callback():
    import lightning.pytorch as pl

    class _AttnDump(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            _TRAINER["t"] = trainer
            install(pl_module)

    return _AttnDump()
