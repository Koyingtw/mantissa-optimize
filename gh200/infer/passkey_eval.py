"""Long-context passkey retrieval under reduced attention / linear-layer precision.

A random 5-digit key is hidden at a given depth inside repeated filler text of about L tokens; the
prompt ends with "What is the pass key? The pass key is" and is followed by the key. One forward
pass (teacher forcing) checks whether the model's argmax predicts every key token.  Specs as in
ppl_eval.py (e.g. 'bf16;attn=fp8', 'bf16;attn=bf16:1', 'fp8c'); prints RESULT lines with accuracy
per (spec, length).
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppl_eval  # noqa: E402

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def build(tok, length, depth, key):
    n_fill = max(1, length // len(tok(FILLER, add_special_tokens=False).input_ids))
    pre = FILLER * int(n_fill * depth)
    post = FILLER * (n_fill - int(n_fill * depth))
    needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
    prompt = ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. "
              + pre + needle + post + "What is the pass key? The pass key is")
    p_ids = tok(prompt, return_tensors="pt").input_ids
    a_ids = tok(" " + str(key), add_special_tokens=False, return_tensors="pt").input_ids
    return torch.cat([p_ids, a_ids], dim=1), a_ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--specs", nargs="+", default=["bf16"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--keys", type=int, default=4, help="random keys per (length, depth)")
    ap.add_argument("--tag", default="")
    a, _ = ap.parse_known_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cpu").eval()
    rng = random.Random(0)
    cases = [(L, d, rng.randint(10000, 99999)) for L in a.lengths for d in a.depths for _ in range(a.keys)]

    for spec in a.specs:
        c = ppl_eval.parse_spec(spec)
        model = copy.deepcopy(base).cuda()
        ppl_eval.replace_linears(model, c["fmt"], c["wa"], c["keep"], c["keep_fmt"], c["head"], False)
        F.scaled_dot_product_attention = ppl_eval.make_attention(c["attn"]) if c["attn"] else ppl_eval._SDPA
        hits = {}
        t0 = time.time()
        with torch.no_grad():
            for L, d, key in cases:
                ids, na = build(tok, L, d, key)
                ids = ids.cuda()
                logits = model(ids).logits[0, -na - 1:-1]
                ok = bool((logits.argmax(-1) == ids[0, -na:]).all())
                hits.setdefault(L, []).append(ok)
        for L, v in hits.items():
            print("RESULT " + json.dumps(dict(tag=a.tag, model=os.path.basename(a.model.rstrip("/")), spec=spec, **c,
                                              length=L, n=len(v), accuracy=sum(v) / len(v),
                                              seconds=round(time.time() - t0, 1))), flush=True)
        F.scaled_dot_product_attention = ppl_eval._SDPA
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
