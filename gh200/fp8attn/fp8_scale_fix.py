"""FP8 delayed-scaling cold-start experiments for the MLPerf Llama2-70B LoRA work copy.

Imported by the patched train.py of the *work copy*
(/nfs/home/koying/mlperf-fp8attn-work/nvidia_impl); production nvidia_impl is never touched.
Every switch defaults to off, so with no env set the run is identical to the original code.

Env switches (read once at import / install time):

  FP8_BOOTSTRAP_PASSES=N   (candidate b') On the first *timed* training step, run N extra
                           forward+backward passes on that same batch before the real one.
                           Their gradients are discarded (MegatronStrategy.training_step zeroes
                           the grad buffers at the start of every call), only the FP8 amax
                           history / scales they produce are kept. N=2 is enough for the cascade
                           o_proj dgrad -> attention dO -> attention dQKV (see 實驗日誌.md).
  FP8_CS_DPA=1             (candidate c, with FP8_RECIPE=tensorwise) Megatron builds
                           Float8CurrentScaling(fp8_format=...) without fp8_dpa, i.e. attention
                           silently falls back to BF16. This forces fp8_dpa=True on that recipe.
  FP8SAT_LOG=1             Cheap monitor: per training step and per decoder layer, the FP8
                           saturation ratio r = amax * scale / fmax of every attention slot
                           (fwd QKV/O/S in E4M3, bwd dO/dP/dQKV in E5M2), plus whether dO and
                           dQKV were all zero. r > 1 means this step's tensor was clipped.
                           Rank 0 writes JSON lines to $FP8SAT_LOG_FILE (default
                           /results/fp8sat.jsonl). Adds one small D2H copy per layer per step.

Recipe / margin switches (FP8_RECIPE, FP8_MARGIN) are plumbed in train.py; the reset knob is
NVIDIA's own RESET_FP8_STATS_AFTER_WARMUP (conf/megatron_gpt_peft_tuning_config.yaml).
"""
import json
import os
import time

import torch

_LOG = "fp8_scale_fix:"
FMAX_FWD, FMAX_BWD = 448.0, 57344.0  # HYBRID: E4M3 forward, E5M2 backward


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


# --------------------------------------------------------------------------- bootstrap (b')
class _Recorder:
    """Iterator wrapper that remembers every item it hands out."""

    def __init__(self, it):
        self._it, self.items = it, []

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self._it)
        self.items.append(x)
        return x

    def __getattr__(self, name):
        return getattr(self._it, name)


class _Replay:
    """Replays recorded items; falls back to the real iterator if more are requested."""

    def __init__(self, items, it):
        self._items, self._i, self._it = items, 0, it

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < len(self._items):
            self._i += 1
            return self._items[self._i - 1]
        raise RuntimeError("fp8 bootstrap: replay asked for more micro-batches than were recorded")

    def __getattr__(self, name):
        return getattr(self._it, name)


def install_bootstrap():
    n = int(os.environ.get("FP8_BOOTSTRAP_PASSES", "0") or 0)
    if n <= 0:
        return
    from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy

    orig = MegatronStrategy.training_step
    state = {"done": False}

    def training_step(self, dataloader_iter, *args, **kwargs):
        if state["done"] or getattr(self.trainer, "warmup", False):
            return orig(self, dataloader_iter, *args, **kwargs)
        state["done"] = True
        torch.cuda.synchronize()
        t0 = time.time()
        rec = _Recorder(dataloader_iter)
        orig(self, rec, *args, **kwargs)  # calibration pass 1 (consumes the real batch)
        for _ in range(n - 1):
            orig(self, _Replay(rec.items, dataloader_iter), *args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.time()
        if _rank() == 0:
            print(f"{_LOG} bootstrap: {n} calibration pass(es) on step-1 batch "
                  f"({len(rec.items)} micro-batch(es)), {t1 - t0:.2f}s, grads discarded", flush=True)
        return orig(self, _Replay(rec.items, dataloader_iter), *args, **kwargs)

    MegatronStrategy.training_step = training_step
    if _rank() == 0:
        print(f"{_LOG} FP8_BOOTSTRAP_PASSES={n} installed", flush=True)


# --------------------------------------------------------------------------- current scaling DPA
def install_cs_dpa():
    if os.environ.get("FP8_CS_DPA", "0") != "1":
        return
    from transformer_engine.common import recipe as R

    orig = R.Float8CurrentScaling.__post_init__

    def post(self):
        orig(self)
        self.fp8_dpa = True

    R.Float8CurrentScaling.__post_init__ = post
    if _rank() == 0:
        print(f"{_LOG} Float8CurrentScaling.fp8_dpa forced True", flush=True)


# --------------------------------------------------------------------------- monitor
def _slots():
    from transformer_engine.pytorch.cpp_extensions import fused_attn as fa
    return (tuple(int(x) for x in (fa.META_QKV, fa.META_O, fa.META_S)),
            tuple(int(x) for x in (fa.META_DO, fa.META_DP, fa.META_DQKV)))


class _Mon:
    def __init__(self, layer, mod, sink):
        self.layer, self.mod, self.sink = layer, mod, sink
        self.trainer = None

    def _st(self, key):
        st = (getattr(self.mod, "fp8_meta", None) or {}).get(key)
        if st is None or not hasattr(st, "scale") or not hasattr(st, "amax_history"):
            return None
        return st

    def post(self, mod, args, kwargs, out):
        if not (mod.training and torch.is_grad_enabled()) or getattr(self.trainer, "warmup", False):
            return
        fw, bw = _slots()
        rec = {"layer": self.layer}
        st = self._st("scaling_fwd")
        if st is not None:
            idx = list(fw)
            rec["f"] = torch.stack([st.scale[idx], st.amax_history[0][idx]]).detach().clone()
        self.sink.cur[self.layer] = rec
        o = out[0] if isinstance(out, (tuple, list)) else out
        q = kwargs.get("query", args[0] if args else None)
        if isinstance(o, torch.Tensor) and o.requires_grad:
            o.register_hook(self._before_bwd(rec))
        if isinstance(q, torch.Tensor) and q.requires_grad:
            q.register_hook(self._after_bwd(rec))

    def _before_bwd(self, rec):
        def hook(g):
            st = self._st("scaling_bwd")
            if st is not None:
                _, bw = _slots()
                rec["bs"] = st.scale[list(bw)].detach().clone()
            return None
        return hook

    def _after_bwd(self, rec):
        def hook(g):
            st = self._st("scaling_bwd")
            if st is not None:
                _, bw = _slots()
                rec["ba"] = st.amax_history[0][list(bw)].detach().clone()
            return None
        return hook


class _Sink:
    def __init__(self, path):
        self.path, self.cur, self.step = path, {}, 0

    def flush(self):
        self.step += 1
        layers = {}
        for l, r in sorted(self.cur.items()):
            d = {}
            if "f" in r:
                s, a = r["f"].float().cpu().tolist()
                d["fwd_scale"], d["fwd_amax"] = s, a
                d["fwd_r"] = [x * y / FMAX_FWD for x, y in zip(a, s)]
            if "bs" in r and "ba" in r:
                s, a = r["bs"].float().cpu().tolist(), r["ba"].float().cpu().tolist()
                d["bwd_scale"], d["bwd_amax"] = s, a
                d["bwd_r"] = [x * y / FMAX_BWD for x, y in zip(a, s)]
            layers[l] = d
        self.cur = {}
        # per-step summary over layers; slot order fwd (QKV,O,S), bwd (dO,dP,dQKV)
        summ = {"step": self.step, "n_layers": len(layers)}
        bw = [d for d in layers.values() if "bwd_amax" in d]
        summ["dO_zero_layers"] = sum(1 for d in bw if d["bwd_amax"][0] == 0.0)
        summ["dQKV_zero_layers"] = sum(1 for d in bw if d["bwd_amax"][2] == 0.0)
        for name, key, i in (("QKV", "fwd_r", 0), ("O", "fwd_r", 1), ("S", "fwd_r", 2),
                             ("dO", "bwd_r", 0), ("dP", "bwd_r", 1), ("dQKV", "bwd_r", 2)):
            rs = [d[key][i] for d in layers.values() if key in d]
            if rs:
                summ[f"{name}_rmax"] = max(rs)
                summ[f"{name}_nsat"] = sum(1 for r in rs if r > 1.0)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"summary": summ, "layers": layers}) + "\n")
        print(f"{_LOG} sat " + json.dumps(summ), flush=True)


def make_monitor_callback():
    import lightning.pytorch as pl

    class _SatMon(pl.Callback):
        def __init__(self):
            self.sink, self.mons = None, []

        def on_train_start(self, trainer, pl_module):
            if _rank() != 0:
                return
            self.sink = _Sink(os.environ.get("FP8SAT_LOG_FILE", "/results/fp8sat.jsonl"))
            for name, mod in pl_module.named_modules():
                if type(mod).__name__ != "TEDotProductAttention" or ".layers." not in name:
                    continue
                idx = int(name.split(".layers.")[1].split(".")[0])
                m = _Mon(idx, mod, self.sink)
                m.trainer = trainer
                mod.register_forward_hook(m.post, with_kwargs=True)
                self.mons.append(m)
            print(f"{_LOG} monitor on {len(self.mons)} attention modules -> {self.sink.path}", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.sink is not None and self.sink.cur:
                self.sink.flush()

    return _SatMon()


def install_early():
    """Called from train.py before the model is built."""
    install_bootstrap()
    install_cs_dpa()
