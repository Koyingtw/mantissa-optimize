"""Copy of cold-gemm/llm/ppl.py (2026-11-03, for the cold-gemm handoff) with one change: the decoder layers are found as
model.model.layers or, for OPT, model.model.decoder.layers, and the original weights are kept on the host (copied to the
GPU layer by layer when a scheme is applied) so Llama 2 70B fits on one GPU. Quantizers and evaluation are unchanged.

E10: WikiText-2 perplexity of Llama-3.1-8B under the integer encodings studied for power.

Simulated (fake) quantization of every decoder nn.Linear (lm_head and embeddings stay BF16):
  weights, per output channel
    w8sym   round(W/s) in [-127, 127], s = max|W| / 127              (standard symmetric INT8)
    w7sym   round(W/s) in [-63, 63],   s = max|W| / 63
            == the non-negative encoding q + 64 in [1, 127] with a fixed zero-point: the integer
            correction is exact, so its accuracy is exactly that of w7sym
    w7asym  round((W - min)/s) in [0, 127], s = (max - min) / 127     (unsigned 7-bit, per-channel
            zero-point; also non-negative in the GEMM)
    w8asym  round((W - min)/s) in [0, 255]: unsigned 8-bit with zero point. Needs a u8 x s8 / u8 x u8
            matrix instruction: MI350X MFMA has only signed int8 (no U8 opcode in ROCm LLVM), so it is
            an accuracy reference for hardware that has one (to be checked on H200)
  activations (input of each Linear), per token
    a8sym   symmetric INT8
    a7asym  unsigned 7-bit with per-token zero-point: X and W both non-negative -> every product >= 0
    a8asym  unsigned 8-bit with per-token zero-point (u8 hardware only, as w8asym)
Scheme names combine as w<..>[-a<..>], e.g. w7asym-a7asym; "bf16" is the unquantized model.

  uv run python ppl.py --schemes bf16,w8sym,w7sym,w7asym,w8sym-a8sym,w7asym-a7asym --out ../results/llm/ppl
"""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
ap.add_argument("--schemes", default="bf16,w8sym,w7sym,w7asym,w8sym-a8sym,w7asym-a7asym")
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--max-windows", type=int, default=0, help="0 = whole test set")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def q_sym(x, bits, dim):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s


def q_asym(x, bits, dim):
    qmax = 2 ** bits - 1
    lo, hi = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
    s = (hi - lo).clamp(min=1e-8) / qmax
    z = (-lo / s).round()
    return ((x / s).round() + z).clamp(0, qmax).sub(z) * s


QW = {"w8sym": lambda w: q_sym(w, 8, 1), "w7sym": lambda w: q_sym(w, 7, 1), "w7asym": lambda w: q_asym(w, 7, 1),
      "w8asym": lambda w: q_asym(w, 8, 1)}
QA = {"a8sym": lambda x: q_sym(x, 8, -1), "a7asym": lambda x: q_asym(x, 7, -1), "a8asym": lambda x: q_asym(x, 8, -1)}

tok = AutoTokenizer.from_pretrained(a.model)
path = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-2-raw-v1/*"])
test = pd.read_parquet(next(Path(path).glob("wikitext-2-raw-v1/test-*.parquet")))
ids = tok("\n\n".join(test.text), return_tensors="pt").input_ids
nwin = ids.shape[1] // a.seqlen
if a.max_windows:
    nwin = min(nwin, a.max_windows)

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
model.eval()
layers = model.model.layers if hasattr(model.model, "layers") else model.model.decoder.layers  # OPT: decoder.layers
linears = {n: m for n, m in layers.named_modules() if isinstance(m, torch.nn.Linear)}
w_orig = {n: m.weight.data.to("cpu", copy=True) for n, m in linears.items()}  # on the host: 70B would not fit twice
rows = []
for scheme in a.schemes.split(","):
    ws, _, as_ = scheme.partition("-")
    for n, m in linears.items():
        w0 = w_orig[n].to(m.weight.device)  # quantized on the GPU, layer by layer
        m.weight.data.copy_(w0 if ws == "bf16" else QW[ws](w0.float()).to(torch.bfloat16))
    hooks = []
    if as_:
        qa = QA[as_]
        hooks = [m.register_forward_pre_hook(lambda mod, inp: (qa(inp[0].float()).to(inp[0].dtype),))
                 for m in linears.values()]
    t0, nll = time.time(), 0.0
    with torch.no_grad():
        for i in range(nwin):
            x = ids[:, i * a.seqlen:(i + 1) * a.seqlen].cuda()
            nll += model(x, labels=x).loss.float().item()
    for h in hooks:
        h.remove()
    r = {"scheme": scheme, "ppl": math.exp(nll / nwin), "windows": nwin, "seqlen": a.seqlen, "model": a.model,
         "seconds": time.time() - t0}
    rows.append(r)
    print(json.dumps(r), flush=True)
    pd.DataFrame(rows).to_csv(out / "ppl.csv", index=False)
