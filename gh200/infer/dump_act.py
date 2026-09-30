"""Dump real Llama2-70B operands for the real-data GEMM benchmark.

Runs WikiText-2 windows through decoder layers 0..LAYER (BF16, layer-wise as ppl_eval_layerwise.py)
and saves, as raw little-endian BF16 row-major files:
  x_mlp_L<LAYER>.bf16  - input of that layer's MLP (post-attention RMSNorm output), tokens x 8192
  w_gateup_L<LAYER>.bf16 - that layer's gate_proj and up_proj weights stacked, 57344 x 8192
plus a small JSON with the shapes.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_eval_layerwise import build_layer, load_weights, make_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/mlperf/llama2_70b_lora/model")
    ap.add_argument("--tokenizer", default="/data/mlperf/llama2_70b_lora/tokenizer")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", default="/nfs/home/koying/models/llama2_70b_operands")
    ap.add_argument("--data-cache", default="/nfs/home/koying/models/datasets")
    a, _ = ap.parse_known_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=a.data_cache)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0, :a.windows * a.seqlen].view(a.windows, a.seqlen)
    cfg = make_config(a.model)
    w = load_weights(a.model)
    hidden = w["model.embed_tokens.weight"][ids].cuda()
    pe = LlamaRotaryEmbedding(config=cfg, device="cuda")(hidden[:1], torch.arange(a.seqlen, device="cuda").unsqueeze(0))
    with torch.no_grad():
        for i in range(a.layer + 1):
            layer = build_layer(cfg, i, w)
            if i == a.layer:
                xs = []
                for j in range(a.windows):
                    h = hidden[j:j + 1]
                    attn_out = layer.self_attn(hidden_states=layer.input_layernorm(h), position_embeddings=pe, attention_mask=None)[0]
                    xs.append(layer.post_attention_layernorm(h + attn_out)[0])
                x = torch.cat(xs).contiguous()
                break
            for j in range(a.windows):
                hidden[j:j + 1] = layer(hidden[j:j + 1], position_embeddings=pe)[0]
            del layer
    p = f"model.layers.{a.layer}.mlp."
    wg = torch.cat([w[p + "gate_proj.weight"], w[p + "up_proj.weight"]]).contiguous()
    fx, fw = f"{a.out}/x_mlp_L{a.layer}.bf16", f"{a.out}/w_gateup_L{a.layer}.bf16"
    x.cpu().view(torch.int16).numpy().tofile(fx)
    wg.view(torch.int16).numpy().tofile(fw)
    meta = dict(x=fx, x_shape=list(x.shape), w=fw, w_shape=list(wg.shape), layer=a.layer,
                x_absmax=x.abs().max().item(), x_absmed=x.float().abs().median().item(),
                w_absmax=wg.abs().max().item(), w_absmed=wg.float().abs().median().item())
    json.dump(meta, open(f"{a.out}/operands_L{a.layer}.json", "w"), indent=1)
    print("DUMPED " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
