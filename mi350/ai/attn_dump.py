"""Dump attention inputs and output gradients (q, k, v, dO) of chosen layers from one forward + backward of an HF causal
LM on the MLPerf GovReport data (AI paper, gap G1), in the format of the GH200 dumps (gh200/fp8attn/attn_dump.py):
one file per layer, dict q / k / v / do as BF16 [seq, heads, head_dim] (k, v with the KV heads, not repeated) + meta.

The model is the base model (LoRA at initialization is zero, so this is step 0 of the fine-tuning). The input is the
first --seq tokens of a training sequence: with causal attention, q, k and v at these positions equal those of the full
8192-token sequence; dO differs slightly because the loss of the later tokens is left out. No gradient checkpointing
(the attention function then runs once per layer); gradients reach the activations through enable_input_require_grads.
usage: attn_dump.py --model meta-llama/Llama-2-70b-hf --layers 0 20 40 79 --seq 2048 --sample 0 --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 20, 40, 79])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/koying/sw/fp8attn-dumps"))
    ap.add_argument("--tag", default="s0")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    store = {}
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def wrapped(module, q, k, v, mask, **kw):
        out, w = orig(module, q, k, v, mask, **kw)  # out: [B, S, H, D]
        li = module.layer_idx
        if li in a.layers and out.requires_grad:
            store[li] = dict(q=q.detach()[0].transpose(0, 1).cpu(), k=k.detach()[0].transpose(0, 1).cpu(),
                             v=v.detach()[0].transpose(0, 1).cpu())

            def hook(g, li=li):
                store[li]["do"] = g.detach()[0].cpu()  # [S, H, D]
            out.register_hook(hook)
        return out, w

    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    model.enable_input_require_grads()
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/koying/sw/hf")), "hub",
                        "datasets--regisss--scrolls_gov_report_preprocessed_mlperf_2")
    d = pd.read_parquet(sorted(glob.glob(f"{root}/snapshots/*/data/train-*.parquet"))[-1])
    ids = torch.tensor(np.asarray(d.input_ids.iloc[a.sample][:a.seq]), dtype=torch.long)[None].cuda()
    labels = torch.tensor(np.asarray(d.labels.iloc[a.sample][:a.seq]), dtype=torch.long)[None].cuda()
    if (labels != -100).sum() < 2:  # the prompt part has no labels: use next-token prediction on the input itself
        labels = ids.clone()
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    os.makedirs(a.out, exist_ok=True)
    name = a.model.split("/")[-1]
    for li in a.layers:
        s = store[li]
        meta = dict(model=a.model, layer=li, seq=a.seq, sample=a.sample, loss=float(out.loss),
                    q=list(s["q"].shape), k=list(s["k"].shape), do=list(s["do"].shape))
        f = f"{a.out}/{name}_L{li:02d}_{a.tag}.pt"
        torch.save(dict(s, meta=meta), f)
        print("DUMP", json.dumps(dict(meta, file=f)), flush=True)


if __name__ == "__main__":
    main()
