"""LoRA fine-tuning on the MLPerf Training GovReport data with the frozen base weights' mantissas reduced (AI paper, W5).

Follows the MLPerf Llama 2 70B LoRA reference recipe, scaled to one GPU per run: LoRA r=16, alpha=32, dropout 0.1 on
q/k/v/o projections, AdamW (lr 4e-4, weight decay 1e-4, cosine to 0 over --max-steps, no warmup), gradient clipping
0.3, global batch 8 sequences of 8192 tokens (--micro sequences per step, the rest by accumulation), evaluation loss on
the 173 validation sequences every --eval-every steps (MLPerf: every 48 steps = 384 samples). The base model runs in
BF16 (LoRA weights in FP32 under BF16 autocast), with gradient checkpointing.

Truncation: once, after loading and before LoRA is attached, every nn.Linear weight inside the decoder blocks keeps
--bits of the 7 BF16 mantissa bits (round to nearest even, or truncation with --round trunc); embeddings, norms and
lm_head stay native. The frozen weights enter the forward GEMM and the input-gradient GEMM of every step, so a GPU that is
power-bound on BF16 matrix work can clock higher; the cost is a weaker base model, paid in samples to converge.

Prints STEP lines (step, samples, train loss, step time, Unix bounds for the power log) and EVAL lines (eval loss).
usage: lora.py --model meta-llama/Llama-2-7b-hf --bits 3 --seed 1 --device 0 [--max-steps 1024] [--target 0.925]
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = "datasets--regisss--scrolls_gov_report_preprocessed_mlperf_2"


def reduce_bf16_(w, k, rnd):
    """In place: keep k of the 7 BF16 mantissa bits (sign-magnitude; a carry into the exponent is correct)."""
    if k >= 7:
        return
    u = w.data.view(torch.int16).to(torch.int32) & 0xFFFF
    drop = 7 - k
    if rnd == "nearest":
        u = u + ((1 << (drop - 1)) - 1) + ((u >> drop) & 1)
    u = u & ~((1 << drop) - 1)
    u = torch.where(u >= 0x8000, u - 0x10000, u)
    w.data.copy_(u.to(torch.int16).view(torch.bfloat16))


def load_split(name):
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/koying/sw/hf")), "hub", DATA)
    f = sorted(glob.glob(f"{root}/snapshots/*/data/{name}-*.parquet"))[-1]
    d = pd.read_parquet(f)
    return (torch.tensor(np.stack(d.input_ids.values), dtype=torch.long),
            torch.tensor(np.stack(d.labels.values), dtype=torch.long))


def log(kind, **kw):
    print(kind, json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=7)
    ap.add_argument("--round", default="nearest", choices=["nearest", "trunc"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--gbs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--eval-every", type=int, default=48)
    ap.add_argument("--eval-samples", type=int, default=173)
    ap.add_argument("--target", type=float, default=0.0, help="stop once the eval loss is at or below this")
    ap.add_argument("--no-eval", action="store_true", help="speed runs: train only")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.cuda.set_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda", a.device)
    # HIP device order is KFD order, not BDF order: the power log is joined on the BDF
    pr = torch.cuda.get_device_properties(a.device)
    bdf = f"{pr.pci_domain_id:04x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    n_red = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name:
            reduce_bf16_(mod.weight, a.bits, a.round)
            n_red += 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.max_steps) / a.max_steps)))

    tr_x, tr_y = load_split("train")
    ev_x, ev_y = load_split("validation")
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(tr_x), generator=g)
    log("START", tag=a.tag, model=a.model, bits=a.bits, round=a.round, seed=a.seed, device=a.device, bdf=bdf,
        reduced_linears=n_red, trainable=sum(p.numel() for p in params), gbs=a.gbs, micro=a.micro,
        max_steps=a.max_steps, torch=torch.__version__)

    def evaluate(step):
        model.eval()
        tot, cnt = 0.0, 0
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(min(a.eval_samples, len(ev_x))):
                x, y = ev_x[i:i + 1].to(dev), ev_y[i:i + 1].to(dev)
                n = int((y[:, 1:] != -100).sum())
                out = model(input_ids=x, labels=y)
                tot += float(out.loss) * n
                cnt += n
        model.train()
        loss = tot / cnt
        log("EVAL", tag=a.tag, step=step, samples=step * a.gbs, eval_loss=loss, seconds=time.time() - t0)
        return loss

    model.train()
    pos = 0
    t_start = time.time()
    for step in range(1, a.max_steps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        acc = a.gbs // a.micro
        loss_sum = 0.0
        for _ in range(acc):
            idx = order[[(pos + j) % len(order) for j in range(a.micro)]]
            pos += a.micro
            x, y = tr_x[idx].to(dev), tr_y[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=y)
            (out.loss / acc).backward()
            loss_sum += float(out.loss) / acc
        torch.nn.utils.clip_grad_norm_(params, 0.3)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        t1 = time.time()
        log("STEP", tag=a.tag, step=step, samples=step * a.gbs, loss=loss_sum, seconds=t1 - t0, t_begin=t0, t_end=t1)
        if not a.no_eval and step % a.eval_every == 0:
            if evaluate(step) <= a.target:
                log("CONVERGED", tag=a.tag, step=step, samples=step * a.gbs, minutes=(time.time() - t_start) / 60)
                break
    log("END", tag=a.tag, minutes=(time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
