"""SQuAD v1.1 dev F1/EM of BERT-large under reduced linear-layer precision (MLPerf BERT-99 / 99.9 style).

MLPerf Inference accepts a BERT submission at 99% (or 99.9%) of the FP32 reference F1; here the
reference is this checkpoint's own FP32 F1.  Specs: 'fp32' (reference, no replacement), 'fp16',
'bf16', 'fp16:<k>' / 'bf16:<k>' (k mantissa bits in weights and activations), 'fp8' / 'fp8c'
(E4M3 per tensor / per channel+token), with the ';wa=w' and ';attn=' options of ppl_eval.py.
The model runs in FP32, FP16 (fp16 formats) or BF16 (all others).
"""
import argparse
import collections
import copy
import json
import os
import re
import string
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppl_eval  # noqa: E402


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_em(pred, answers):
    best_f1, em = 0.0, 0.0
    for gt in answers:
        p, g = normalize(pred).split(), normalize(gt).split()
        common = collections.Counter(p) & collections.Counter(g)
        same = sum(common.values())
        f1 = 0.0 if same == 0 else 2 * (same / len(p)) * (same / len(g)) / (same / len(p) + same / len(g))
        best_f1 = max(best_f1, f1)
        em = max(em, float(normalize(pred) == normalize(gt)))
    return best_f1, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/nfs/home/koying/models/bert-large-squad")
    ap.add_argument("--specs", nargs="+", default=["fp32", "fp16", "bf16", "fp8", "fp8c"])
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data-cache", default="/nfs/home/koying/models/datasets")
    ap.add_argument("--tag", default="")
    a, _ = ap.parse_known_args()

    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("rajpurkar/squad", split="validation", cache_dir=a.data_cache)
    if a.max_examples:
        ds = ds.select(range(a.max_examples))
    enc = tok(list(ds["question"]), list(ds["context"]), truncation="only_second", max_length=384, stride=128,
              return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    feat_ex = enc["overflow_to_sample_mapping"]
    ids = torch.tensor(enc["input_ids"])
    tt = torch.tensor(enc["token_type_ids"])
    am = torch.tensor(enc["attention_mask"])
    base = AutoModelForQuestionAnswering.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    ref_f1 = None

    for spec in a.specs:
        c = ppl_eval.parse_spec(spec) if spec != "fp32" else dict(fmt="fp32", wa="wa", keep="", keep_fmt="bf16", head=False, attn="")
        kind = c["fmt"].split(":")[0]
        dt = {"fp32": torch.float32, "fp16": torch.float16}.get(kind, torch.bfloat16)
        model = copy.deepcopy(base).to(dt).cuda()
        if kind != "fp32" and ":" in c["fmt"] or kind.startswith("fp8"):
            ppl_eval.replace_linears(model, c["fmt"], c["wa"], c["keep"], c["keep_fmt"], True, False)
        F.scaled_dot_product_attention = ppl_eval.make_attention(c["attn"]) if c["attn"] else ppl_eval._SDPA
        t0 = time.time()
        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, len(ids), a.batch):
                out = model(input_ids=ids[i:i + a.batch].cuda(), token_type_ids=tt[i:i + a.batch].cuda(),
                            attention_mask=am[i:i + a.batch].cuda())
                starts.append(out.start_logits.float().cpu())
                ends.append(out.end_logits.float().cpu())
        F.scaled_dot_product_attention = ppl_eval._SDPA
        starts, ends = torch.cat(starts).numpy(), torch.cat(ends).numpy()
        best = {}
        for fi, ex in enumerate(feat_ex):
            offs, seq = enc["offset_mapping"][fi], enc.sequence_ids(fi)
            s_idx = np.argsort(starts[fi])[-20:][::-1]
            e_idx = np.argsort(ends[fi])[-20:][::-1]
            for si in s_idx:
                for ei in e_idx:
                    if seq[si] != 1 or seq[ei] != 1 or ei < si or ei - si + 1 > 30:
                        continue
                    score = starts[fi][si] + ends[fi][ei]
                    if ex not in best or score > best[ex][0]:
                        best[ex] = (score, ds[ex]["context"][offs[si][0]:offs[ei][1]])
        f1s, ems = zip(*(f1_em(best.get(i, (0, ""))[1], ds[i]["answers"]["text"]) for i in range(len(ds))))
        f1, em = 100 * np.mean(f1s), 100 * np.mean(ems)
        if spec == "fp32":
            ref_f1 = f1
        res = dict(tag=a.tag, model="bert-large-squad", spec=spec, **c, examples=len(ds), f1=f1, em=em,
                   rel_f1=(f1 / ref_f1 * 100) if ref_f1 else None, seconds=round(time.time() - t0, 1))
        print("RESULT " + json.dumps(res), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
