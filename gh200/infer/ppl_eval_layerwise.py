"""Layer-wise WikiText-2 perplexity for a Llama model too large for one GPU in BF16.

Built for the MLPerf Llama2-70B checkpoint (fused qkv_proj, custom model code): all weights are
read once into CPU memory (GH200 LPDDR5X), then for every spec the evaluation windows pass
through one decoder layer at a time, each layer copied to the GPU once.  The fused qkv weight is
split back into q/k/v so layer names match the standard HF Llama (self_attn.q_proj, ...).
Formats and --specs are those of ppl_eval.py; the final norm and lm_head stay BF16.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppl_eval import QLinear, parse_spec, replace_linears  # noqa: E402


def load_weights(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    w = {}
    for f in sorted(set(idx.values())):
        with safe_open(os.path.join(model_dir, f), "pt", device="cpu") as sf:
            for k in sf.keys():
                w[k] = sf.get_tensor(k)
    return w


def make_config(model_dir):
    from transformers import LlamaConfig
    c = json.load(open(os.path.join(model_dir, "config.json")))
    keys = ["hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
            "num_hidden_layers", "rms_norm_eps", "rope_theta", "max_position_embeddings", "vocab_size"]
    cfg = LlamaConfig(**{k: c[k] for k in keys}, rope_scaling=c.get("rope_scaling"), torch_dtype=torch.bfloat16)
    cfg._attn_implementation = "sdpa"
    return cfg


def build_layer(cfg, i, w):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    with torch.device("meta"):
        layer = LlamaDecoderLayer(cfg, i)
    layer = layer.to_empty(device="cuda").to(torch.bfloat16)
    p = f"model.layers.{i}."
    hd = cfg.hidden_size // cfg.num_attention_heads
    nq, nkv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    sd = {}
    if p + "self_attn.qkv_proj.weight" in w:
        qkv = w[p + "self_attn.qkv_proj.weight"]
        sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"], sd["self_attn.v_proj.weight"] = \
            qkv[:nq], qkv[nq:nq + nkv], qkv[nq + nkv:nq + 2 * nkv]
    for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        if p + name + ".weight" in w:
            sd[name + ".weight"] = w[p + name + ".weight"]
    sd["input_layernorm.weight"] = w[p + "input_layernorm.weight"]
    sd["post_attention_layernorm.weight"] = w[p + "post_attention_layernorm.weight"]
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return layer.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/mlperf/llama2_70b_lora/model")
    ap.add_argument("--tokenizer", default="/data/mlperf/llama2_70b_lora/tokenizer")
    ap.add_argument("--specs", nargs="+", default=["bf16"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--layer-stats", action="store_true")
    ap.add_argument("--data-cache", default="/nfs/home/koying/models/datasets")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=a.data_cache)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    nwin = ids.shape[1] // a.seqlen
    if a.max_windows:
        nwin = min(nwin, a.max_windows)
    windows = ids[0, :nwin * a.seqlen].view(nwin, a.seqlen)

    t_load = time.time()
    cfg = make_config(a.model)
    w = load_weights(a.model)
    print(f"loaded {len(w)} tensors in {time.time() - t_load:.0f}s", flush=True)
    embed = w["model.embed_tokens.weight"]
    norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().to(torch.bfloat16)
    norm.weight.data.copy_(w["model.norm.weight"])
    head = w["lm_head.weight"].cuda()
    rot = LlamaRotaryEmbedding(config=cfg, device="cuda")
    pos = torch.arange(a.seqlen, device="cuda").unsqueeze(0)

    for spec in a.specs:
        c = parse_spec(spec)
        t0 = time.time()
        hidden = embed[windows].cuda()  # (nwin, L, H) bf16, ~4.7 GB for the full test set at 70B
        pe = rot(hidden[:1], pos)
        layer_err = {}
        with torch.no_grad():
            for i in range(cfg.num_hidden_layers):
                layer = build_layer(cfg, i, w)
                replace_linears(layer, c["fmt"], c["wa"], c["keep"], c["keep_fmt"], True, a.layer_stats)
                for j in range(nwin):
                    hidden[j:j + 1] = layer(hidden[j:j + 1], position_embeddings=pe)[0]
                if a.layer_stats:
                    for m in layer.modules():
                        if isinstance(m, QLinear) and m.stats is not None:
                            layer_err[f"layers.{i}.{m.name}"] = m.stats
                del layer
            nll = 0.0
            for j in range(nwin):
                logits = F.linear(norm(hidden[j]), head).float()
                x = windows[j].cuda()
                nll += F.cross_entropy(logits[:-1], x[1:], reduction="sum").item()
        ntok = nwin * (a.seqlen - 1)
        res = dict(tag=a.tag, model="Llama2-70B", spec=spec, **c, seqlen=a.seqlen, windows=nwin, tokens=ntok,
                   ppl=math.exp(nll / ntok), seconds=round(time.time() - t0, 1))
        if a.layer_stats:
            res["layer_err"] = layer_err
        print("RESULT " + json.dumps(res), flush=True)
        del hidden
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
