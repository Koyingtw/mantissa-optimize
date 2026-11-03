"""Decode-like weight streaming with an LLM's real weights (AI paper, W4): tokens/s and energy per token.

A decode step of a dense LLM reads every linear weight once per generated token batch. This bench loads the decoder
linear weights of a Hugging Face checkpoint (q/k/v/o/gate/up/down of every layer) onto one GPU and runs y = x W^T for
all of them in layer order, with x of B tokens; one pass = one decode step of B sequences, without attention and the
KV cache (which truncating weights does not change). One pass is captured as a HIP graph and replayed, so small batches
measure the GPU, not Python launch overhead.

Formats (one per measurement window):
  bf16:<k>   BF16 weights keeping k of 7 mantissa bits (round to nearest), BF16 activations, F.linear
  fp8:<k>    FP8 E4M3 weights per output channel keeping k of 3 bits, activations quantized per token on every call,
             torch._scaled_mm (the dynamic FP8 recipe of gh200/infer/ppl_eval.py, fmt fp8c, weights only reduced)
Every GPU runs the same schedule from a shared start time T0, so all GPUs of the node are in the same format at the same
time (mixing formats across GPUs of one node biased the LoRA runs); a window is warm-up + measurement + idle gap.

Prints one RESULT line (JSON) per window: format, batch, passes, tokens/s, Unix bounds for the power log.
usage: decode_bench.py --model meta-llama/Llama-2-7b-hf --device 0 --t0 <unix> --formats "bf16:7 bf16:3 ..."
                       --batches "1 16 64 256" [--warmup 5 --seconds 15 --gap 10]
Schedule: for each batch, the formats in order then reversed (ABBA).
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gh200", "infer"))
from ppl_eval import E4M3_KMAX, reduce_bf16, to_fp8  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_linears(model, dev):
    from safetensors import safe_open
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/koying/sw/hf")), "hub",
                        "models--" + model.replace("/", "--"))
    files = sorted(glob.glob(f"{root}/snapshots/*/*.safetensors"))
    ws = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as s:
            for k in s.keys():
                if ".layers." in k and k.endswith(".weight") and any(p in k for p in PROJ):
                    ws[k] = s.get_tensor(k).to(torch.bfloat16)
    def order(k):  # layer index, then projection order
        layer = int(k.split(".layers.")[1].split(".")[0])
        return layer, next(i for i, p in enumerate(PROJ) if p in k)
    return [ws[k].to(dev) for k in sorted(ws, key=order)]


def build(ws, fmt, batch, dev):
    """Return a zero-argument callable that runs one pass, plus the tensors it needs (kept alive)."""
    kind, k = fmt.split(":")
    k = int(k)
    xs = {w.shape[1]: torch.randn(batch, w.shape[1], device=dev, dtype=torch.bfloat16) for w in ws}
    if kind == "bf16":
        wr = [reduce_bf16(w, k, "nearest") for w in ws]

        def run():
            for w in wr:
                F.linear(xs[w.shape[1]], w)
        return run, (wr, xs)
    # fp8: per-channel weight scales, per-token activation scales computed on every call
    fmax = E4M3_KMAX[k]
    wq, sc = [], []
    for w in ws:
        s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fmax
        wq.append(to_fp8(w, s, k, "nearest"))
        sc.append(s.reshape(1, -1).contiguous())
    m = batch + (-batch) % 16  # _scaled_mm needs rows in multiples of 16
    xp = {K: F.pad(x, (0, 0, 0, m - batch)) for K, x in xs.items()}

    def run():
        for w, s in zip(wq, sc):
            x = xp[w.shape[1]]
            sa = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_KMAX[3]
            xq = (x.float() / sa).clamp(-448, 448).to(torch.float8_e4m3fn)
            torch._scaled_mm(xq, w.t(), scale_a=sa, scale_b=s, out_dtype=torch.bfloat16)
    return run, (wq, sc, xp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--formats", default="bf16:7 bf16:4 bf16:3 fp8:3 fp8:2")
    ap.add_argument("--batches", default="1 16 64 256")
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--gap", type=float, default=10)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    dev = torch.device("cuda", a.device)
    pr = torch.cuda.get_device_properties(a.device)
    bdf = f"{pr.pci_domain_id:04x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
    ws = load_linears(a.model, dev)
    gbytes = sum(w.numel() for w in ws) * 2 / 1e9
    fmts = a.formats.split()
    sched = [(int(b), f) for b in a.batches.split() for f in fmts + fmts[::-1]]
    period = a.warmup + a.seconds + a.gap
    print("START", json.dumps(dict(tag=a.tag, model=a.model, device=a.device, bdf=bdf, linears=len(ws),
                                   bf16_gbytes=gbytes, windows=len(sched), period=period)), flush=True)
    for i, (batch, fmt) in enumerate(sched):
        start = a.t0 + i * period
        run, keep = build(ws, fmt, batch, dev)  # prepared during the previous window's gap
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run()  # warm the kernels outside the graph
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            run()
        torch.cuda.synchronize()
        now = time.time()
        if now > start:
            print("LATE", json.dumps(dict(window=i, seconds=now - start)), flush=True)
        else:
            time.sleep(start - now)
        t_end_warm = time.time() + a.warmup
        while time.time() < t_end_warm:
            g.replay()
            torch.cuda.synchronize()
        n = 0
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            n += 10
        t1 = time.time()
        print("RESULT", json.dumps(dict(tag=a.tag, bdf=bdf, window=i, fmt=fmt, batch=batch, passes=n,
                                        tokens_per_s=batch * n / (t1 - t0), pass_ms=1e3 * (t1 - t0) / n,
                                        t_begin=t0, t_end=t1)), flush=True)
        del g, run, keep
        torch.cuda.empty_cache()
    print("END", flush=True)


if __name__ == "__main__":
    main()
