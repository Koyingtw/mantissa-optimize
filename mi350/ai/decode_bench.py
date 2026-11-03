"""Decode-like weight streaming with an LLM's real weights (AI paper, W4): tokens/s and energy per token.

A decode step of a dense LLM reads every linear weight once per generated token batch. This bench loads the decoder
linear weights of a Hugging Face checkpoint (q/k/v/o/gate/up/down of every layer) onto one GPU and runs y = x W^T for
all of them in layer order, with x of B tokens; one pass = one decode step of B sequences, without attention and the
KV cache (which truncating weights does not change). One pass is captured as HIP graphs (one per layer by default; --mode
graph = one graph for the whole pass, eager = no graph) and replayed, so small batches measure the GPU, not Python
launch overhead.

Formats (one per measurement window):
  bf16:<k>   BF16 weights keeping k of 7 mantissa bits (round to nearest), BF16 activations, F.linear
  fp8:<k>    FP8 E4M3 weights per output channel keeping k of 3 bits, activations quantized per token once (outside
             the pass), torch._scaled_mm (the dynamic FP8 recipe of gh200/infer/ppl_eval.py, fmt fp8c, weights only
             reduced)
  int8:<enc> INT8 x INT8 -> INT32 GEMMs, enc = w8a8 | w7 | w7off | w7offc (see build_int8; for the cold-gemm session)
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


def pad_rows(x, m):
    """Pad a batch of token rows to m rows by repeating the real rows cyclically (not zeros: zero rows switch less
    and would bias the padded formats, as the cold-gemm session pointed out)."""
    return x if x.shape[0] == m else x[torch.arange(m, device=x.device) % x.shape[0]]


def build(ws, fmt, batch, dev):
    """Return one zero-argument callable per weight (in layer order), plus the tensors they need (kept alive)."""
    kind, k = fmt.split(":")
    xs = {w.shape[1]: torch.randn(batch, w.shape[1], device=dev, dtype=torch.bfloat16) for w in ws}
    if kind == "int8":
        return build_int8(ws, k, batch, xs, dev)
    k = int(k)
    if kind == "bf16":
        wr = [reduce_bf16(w, k, "nearest") for w in ws]

        return [lambda w=w: F.linear(xs[w.shape[1]], w) for w in wr], (wr, xs)
    # fp8: per-channel weight scales, per-token activation scales computed on every call
    fmax = E4M3_KMAX[k]
    wq, sc = [], []
    for w in ws:
        s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fmax
        wq.append(to_fp8(w, s, k, "nearest"))
        sc.append(s.reshape(1, -1).contiguous())
    m = batch + (-batch) % 16  # _scaled_mm needs rows in multiples of 16 (repeated rows; job 391 padded with zeros)
    # activations quantized once per input width, outside the timed pass: an inference engine fuses this into the
    # preceding kernel, and quantizing per call in eager mode costs 5 extra launches per GEMM (FP8 then measured 4x
    # slower than BF16, job 387); the pass then reads every weight once with one kernel per weight, as for BF16
    xq, sa = {}, {}
    for K, x in xs.items():
        xpad = pad_rows(x, m)
        sa[K] = xpad.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_KMAX[3]
        xq[K] = (xpad.float() / sa[K]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return [lambda w=w, s=s: torch._scaled_mm(xq[w.shape[1]], w.t(), scale_a=sa[w.shape[1]], scale_b=s,
                                               out_dtype=torch.bfloat16) for w, s in zip(wq, sc)], (wq, sc, xq, sa)


WFIRST = False  # --int8-wfirst: weights as the first torch operand (hipBLASLt src0), Y^T = Wq Xq^T


def imm(x, w):
    """Integer product of the token rows x [m, K] and a weight w [N, K]: Y = x w^T as _int_mm(x, w^T) (default), or
    Y^T = _int_mm(w, x^T) with --int8-wfirst. torch is row-major, so hipBLASLt sees the transposed problem and its B
    argument, which lands on the MFMA src0 input, is the first torch operand: X by default, the weights with
    --int8-wfirst (the cold-gemm session's layout, where w7off gains +5-8% at large M)."""
    return torch._int_mm(w, x.t()) if WFIRST else torch._int_mm(x, w.t())


def build_int8(ws, enc, batch, xs, dev):
    """INT8 GEMMs with int32 accumulation (torch._int_mm -> hipBLASLt I8II), for the cold-gemm session:
      w8a8    weights per output channel symmetric 8-bit, round(W/s) in [-127, 127]
      w7      control: symmetric 7-bit, round(W/s) in [-63, 63] (same lost bit as w7off, signed)
      w7off   w7 plus 64, W' = round(W/s) + 64 in [1, 127] (non-negative); the correction
              y[i, j] -= 64 * r[i], r[i] = sum_k Xq[i, k] (int32), is assumed fused into the epilogue and not run
      w7offc  as w7off with the correction run as a separate kernel on every call
    Activations: per-token symmetric 8-bit, quantized once outside the pass (as for FP8); the activations X are
    standard normal BF16 (torch.randn, unseeded), not real model activations. torch._int_mm needs more than
    16 rows (int8_probe.py), so the batch is padded to max(batch, 17) rounded up to a multiple of 8 (24 for batch <= 16)
    by repeating the real rows; decode reads the same weights, so the padding costs compute, not memory traffic.
    CHECK lines, before any measurement: for w7off / w7offc the corrected int32 result must equal w7 bit for bit for
    every weight; for w8a8 / w7 the first weight against the exact float64 product."""
    if WFIRST:  # the weight is the M > 16 side; pad the tokens to a multiple of 8 (24 for batch < 8, as in job 417)
        m = 24 if batch < 8 else batch + (-batch) % 8
    else:
        m = max(batch, 17)
        m += (-m) % 8
    bits, off = (8, 0) if enc == "w8a8" else (7, 64 if enc.startswith("w7off") else 0)
    qmax = 2 ** (bits - 1) - 1
    wq, w7 = [], []
    for w in ws:
        s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
        q = (w.float() / s).round().clamp(-qmax, qmax)
        wq.append((q + off).to(torch.int8))
        if off:
            w7.append(q.to(torch.int8))
    xq, rs = {}, {}
    for K, x in xs.items():
        xp = pad_rows(x, m).float()
        sa = xp.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
        xq[K] = (xp / sa).round().clamp(-127, 127).to(torch.int8)
        r = off * xq[K].to(torch.int32).sum(dim=1, keepdim=True)  # [m, 1]: per token, broadcast over the channels j
        rs[K] = (r.t() if WFIRST else r).contiguous()
    if off:
        bad = sum(int(not torch.equal(imm(xq[a.shape[1]], a) - rs[a.shape[1]], imm(xq[b.shape[1]], b)))
                  for a, b in zip(wq, w7))
        print("CHECK", json.dumps(dict(enc=enc, batch=batch, rows=m, vs="w7 int32", wfirst=WFIRST, weights=len(wq),
                                       mismatched=bad, exact=bad == 0)), flush=True)
        if bad:
            raise SystemExit(f"{enc}: corrected result differs from w7 for {bad} weights")
        del w7
    else:
        w0 = wq[0]
        ref = xq[w0.shape[1]].double() @ w0.double().t()
        y0 = imm(xq[w0.shape[1]], w0).double()
        print("CHECK", json.dumps(dict(enc=enc, batch=batch, rows=m, vs="float64", wfirst=WFIRST,
                                       exact=bool(((y0.t() if WFIRST else y0) == ref).all()))), flush=True)
    if enc == "w7offc":
        def op(w):
            y = imm(xq[w.shape[1]], w)
            y.sub_(rs[w.shape[1]])
        return [lambda w=w: op(w) for w in wq], (wq, xq, rs)
    return [lambda w=w: imm(xq[w.shape[1]], w) for w in wq], (wq, xq, rs)


def executor(ops, mode, chunk=7):
    """One pass over all ops: eager launches, one HIP graph for the whole pass (graph), or one graph per chunk of
    ops, i.e. per layer (layergraph; the single large graph segfaulted on replay in job 384)."""
    if mode == "eager":
        def step():
            for o in ops:
                o()
        return step, None
    groups = [ops] if mode == "graph" else [ops[i:i + chunk] for i in range(0, len(ops), chunk)]
    graphs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for o in ops:  # warm the kernels outside the graph
            o()
    torch.cuda.current_stream().wait_stream(s)
    for grp in groups:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for o in grp:
                o()
        graphs.append(g)

    def step():
        for g in graphs:
            g.replay()
    return step, graphs


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
    ap.add_argument("--mode", default="eager", choices=["eager", "graph", "layergraph"],
                    help="graph and layergraph segfault on replay with ROCm 7.1 PyTorch 2.13 (jobs 384, 387)")
    ap.add_argument("--profile", action="store_true", help="per window also print the summed GPU kernel time of 5 passes")
    ap.add_argument("--int8-wfirst", action="store_true", help="INT8: weights as the first _int_mm operand (src0)")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    global WFIRST
    WFIRST = a.int8_wfirst
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
        ops, keep = build(ws, fmt, batch, dev)  # prepared during the previous window's gap
        step, graphs = executor(ops, a.mode)
        torch.cuda.synchronize()
        if a.profile:  # is a pass GPU-bound? summed kernel time against wall time per pass
            from torch.profiler import ProfilerActivity, profile
            step()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                tw = time.time()
                for _ in range(5):
                    step()
                torch.cuda.synchronize()
                tw = time.time() - tw
            ka = [e for e in prof.key_averages() if e.self_device_time_total > 0]
            ktot = sum(e.self_device_time_total for e in ka) / 1e3 / 5
            print("PROF", json.dumps(dict(window=i, fmt=fmt, batch=batch, kernel_ms=ktot, wall_ms=1e3 * tw / 5,
                                          kernels=sorted(e.key for e in ka))), flush=True)
        now = time.time()
        if now > start:
            print("LATE", json.dumps(dict(window=i, seconds=now - start)), flush=True)
        else:
            time.sleep(start - now)
        t_end_warm = time.time() + a.warmup
        while time.time() < t_end_warm:
            step()
            torch.cuda.synchronize()
        n = 0
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            n += 10
        t1 = time.time()
        print("RESULT", json.dumps(dict(tag=a.tag, bdf=bdf, window=i, fmt=fmt, batch=batch, mode=a.mode, passes=n,
                                        tokens_per_s=batch * n / (t1 - t0), pass_ms=1e3 * (t1 - t0) / n,
                                        t_begin=t0, t_end=t1)), flush=True)
        del step, graphs, ops, keep
        torch.cuda.empty_cache()
    print("END", flush=True)


if __name__ == "__main__":
    main()
