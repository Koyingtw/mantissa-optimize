"""WikiText-2 perplexity of an HF causal LM with its linear layers run in a chosen number format.

Formats (the fmt field of --specs, applied to every nn.Linear inside the decoder blocks; lm_head and
embeddings stay BF16 unless head=1):
  bf16                 native BF16 GEMM (FP32 accumulate)
  bf16:<k>[:<round>]   weights and activations keep <k> of the 7 BF16 mantissa bits, then a
                       native BF16 GEMM (the hardware path of mantissa zeroing)
  fp8[:<k>[:<round>]]  E4M3 on the FP8 tensor cores (torch._scaled_mm), per-tensor dynamic
                       activation scale and per-tensor weight scale (the common "dynamic FP8"
                       recipe); <k> of the 3 E4M3 mantissa bits kept after the cast
  fp8c[:<k>[:<round>]] as fp8 but per-output-channel weight and per-token activation scales
round = trunc | nearest (half to even, default).  wa=w keeps activations native (weights only).
Each --specs entry is 'fmt[;wa=w][;keep=REGEX][;keep_fmt=FMT][;head=1]': layers whose name matches
REGEX use keep_fmt instead (default bf16), e.g. keep the FP8-sensitive layers in BF16 or in truncated
BF16; head=1 also converts lm_head; attn=bf16:<k> truncates Q/K/V of the attention to k mantissa
bits, attn=fp8[:<k>] emulates FP8 attention (Q, K, V and softmax probabilities in E4M3).
--layer-stats: also report each layer's relative output error vs BF16 on the first window.
Prints one JSON line (RESULT {...}).
"""
import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0
# Largest finite E4M3 value once only k mantissa bits are kept (codes above it are not
# representable): scales map amax onto this, not onto 448, so the top of the range is not clipped.
E4M3_KMAX = {3: 448.0, 2: 448.0, 1: 384.0, 0: 256.0}


def reduce_bf16(x, k, rnd):
    """Keep k of the 7 BF16 mantissa bits (sign-magnitude, carry into the exponent is correct)."""
    if k >= 7:
        return x
    u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    drop = 7 - k
    if rnd == "nearest":
        u = u + ((1 << (drop - 1)) - 1) + ((u >> drop) & 1)
    u = u & ~((1 << drop) - 1)
    u = torch.where(u >= 0x8000, u - 0x10000, u)  # back to signed 16-bit range
    return u.to(torch.int16).view(torch.bfloat16).view(x.shape)


def reduce_fp16(x, k, rnd):
    """Keep k of the 10 FP16 mantissa bits (same bit trick as reduce_bf16)."""
    if k >= 10:
        return x
    u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    drop = 10 - k
    if rnd == "nearest":
        u = u + ((1 << (drop - 1)) - 1) + ((u >> drop) & 1)
    u = u & ~((1 << drop) - 1)
    u = torch.where(u >= 0x8000, u - 0x10000, u)
    return u.to(torch.int16).view(torch.float16).view(x.shape)


def reduce_e4m3(q, k, rnd):
    """Keep k of the 3 E4M3 mantissa bits on the raw byte; saturate below the NaN code."""
    if k >= 3:
        return q
    u = q.view(torch.uint8).to(torch.int16)
    sign, mag = u & 0x80, u & 0x7F
    drop = 3 - k
    low = (1 << drop) - 1
    if rnd == "nearest":
        mag = mag + ((1 << (drop - 1)) - 1) + ((mag >> drop) & 1)
    mag = torch.clamp(mag & ~low, max=0x7E & ~low)
    return (sign | mag).to(torch.uint8).view(torch.float8_e4m3fn)


def to_fp8(x, scale, k, rnd):
    q = (x.float() / scale).clamp(-E4M3_KMAX[k], E4M3_KMAX[k]).to(torch.float8_e4m3fn)
    return reduce_e4m3(q, k, rnd)


def parse_fmt(s):
    p = s.split(":")
    kind = p[0]
    native = {"bf16": 7, "fp16": 10, "fp8": 3, "fp8c": 3, "fp8t": 3, "fp8p": 3}[kind]
    k = int(p[1]) if len(p) > 1 and p[1] != "" else native
    rnd = p[2] if len(p) > 2 else "nearest"
    return kind, k, rnd


class QLinear(nn.Module):
    def __init__(self, lin, fmt, wa, name):
        super().__init__()
        self.kind, self.k, self.rnd = parse_fmt(fmt)
        self.wa, self.name = wa, name
        self.bias = lin.bias
        w = lin.weight.data
        self.out_features, self.in_features = w.shape
        self.w_ref = None
        if self.kind in ("bf16", "fp16"):
            self.red = reduce_bf16 if self.kind == "bf16" else reduce_fp16
            self.weight = nn.Parameter(self.red(w, self.k, self.rnd), requires_grad=False)
        else:
            fmax = E4M3_KMAX[self.k]
            if self.kind == "fp8":
                s = w.float().abs().amax().clamp(min=1e-12) / fmax
                self.w_scale = s.reshape(1, 1)
            else:  # per output channel
                s = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fmax
                self.w_scale = s.reshape(1, -1)  # (1, N) for _scaled_mm
            self.w_q = to_fp8(w, s, self.k, self.rnd)  # (N, K) row-major
        self.stats = None

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if self.kind in ("bf16", "fp16"):
            xa = x2 if self.wa == "w" else self.red(x2, self.k, self.rnd)
            y = F.linear(xa, self.weight, self.bias)
        else:
            m = x2.shape[0]
            pad = (-m) % 16
            xp = F.pad(x2, (0, 0, 0, pad)) if pad else x2
            ka = 3 if self.wa == "w" else self.k
            if self.kind == "fp8":
                sa = xp.float().abs().amax().clamp(min=1e-12) / E4M3_KMAX[ka]
                sa = sa.reshape(1, 1)
            else:
                sa = xp.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / E4M3_KMAX[ka]
            xq = to_fp8(xp, sa, ka, self.rnd)
            y = torch._scaled_mm(xq, self.w_q.t(), scale_a=sa, scale_b=self.w_scale, out_dtype=torch.bfloat16)
            y = y[:m]
            if self.bias is not None:
                y = y + self.bias
        if self.w_ref is not None and self.stats is None:
            ref = F.linear(x2, self.w_ref, self.bias).float()
            self.stats = ((y.float() - ref).norm() / ref.norm()).item()
            self.w_ref = None
        return y.reshape(*shp[:-1], y.shape[-1])


def replace_linears(model, fmt, wa, keep, keep_fmt, include_head, layer_stats):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if not isinstance(child, nn.Linear):
                continue
            if not include_head and ("lm_head" in full or "embed" in full):
                continue
            f = keep_fmt if keep and re.search(keep, full) else fmt
            if f.startswith("fp8") and (child.out_features % 16 or child.in_features % 16):
                continue  # FP8 tensor-core GEMM needs multiples of 16 (e.g. BERT's 2-wide qa_outputs)
            q = QLinear(child, f, wa, full)
            if layer_stats:
                q.w_ref = child.weight.data
            setattr(mod, cname, q)
            n += 1
    torch.cuda.empty_cache()
    return n


_SDPA = F.scaled_dot_product_attention


def fake_e4m3(x, k=3, rnd="nearest", dim=None):
    """Quantize-dequantize to E4M3 keeping k mantissa bits (per-tensor, or per slice along dim)."""
    amax = x.float().abs().amax() if dim is None else x.float().abs().amax(dim=dim, keepdim=True)
    s = amax.clamp(min=1e-12) / E4M3_KMAX[k]
    return (to_fp8(x, s, k, rnd).float() * s).to(x.dtype)


def make_attention(mode):
    """Replacement for F.scaled_dot_product_attention used by HF 'sdpa' attention.
    mode 'bf16:<k>[:round]': Q, K, V keep k BF16 mantissa bits, then the normal kernel.
    mode 'fp8[:<k>]': emulated FP8 attention (FlashAttention-3 style): Q, K, V per-head E4M3,
    softmax in FP32, probabilities P quantized to E4M3 before P @ V; 'fp8t' uses one scale per
    whole Q / K / V tensor instead of per head; 'fp8p' is 'fp8t' but casts the softmax probabilities P
    (in [0, 1]) to E4M3 with no rescaling -- a naive FP8 attention used as a controlled experiment
    on range: probabilities below the E4M3 subnormal range (~2e-3) vanish.  (Not FlashAttention-3's
    documented scheme, which quantizes Q/K/V per block; SageAttention2 scales P by 448.)"""
    kind, k, rnd = parse_fmt(mode)

    def attn(q, key, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if kind == "bf16":
            q, key, v = (reduce_bf16(t, k, rnd) for t in (q, key, v))
            return _SDPA(q, key, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        if key.shape[1] != q.shape[1]:  # grouped-query attention
            rep = q.shape[1] // key.shape[1]
            key, v = key.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        qdim = None if kind in ("fp8t", "fp8p") else (2, 3)
        q, key, v = (fake_e4m3(t, k, rnd, dim=qdim) for t in (q, key, v))
        sc = scale if scale is not None else q.shape[-1] ** -0.5
        out = torch.empty_like(q)
        L = q.shape[2]
        S = key.shape[2]
        blk = 2048  # query rows per block: one head and block at a time keeps the scores small at 64k context
        for h in range(q.shape[1]):
            kh, vh = key[:, h].float(), v[:, h].float()
            for r0 in range(0, L, blk):
                r1 = min(L, r0 + blk)
                sco = (q[:, h, r0:r1].float() @ kh.transpose(-1, -2)) * sc
                if is_causal:
                    causal = torch.ones(r1 - r0, S, device=q.device, dtype=torch.bool).tril(S - L + r0)
                    sco = sco.masked_fill(~causal, float("-inf"))
                elif attn_mask is None:
                    pass  # full (bidirectional) attention, e.g. diffusion transformers or a 1-token decode step
                elif attn_mask.dtype == torch.bool:
                    sco = sco.masked_fill(~attn_mask[:, 0, r0:r1], float("-inf"))
                else:
                    sco = sco + attn_mask[:, 0, r0:r1].float()
                pr = torch.softmax(sco, dim=-1)
                if kind == "fp8p":  # direct cast, scale 1: E4M3 min subnormal 2^-9, anything smaller -> 0
                    pr = reduce_e4m3(pr.clamp(max=E4M3_KMAX[k]).to(torch.float8_e4m3fn), k, rnd).float()
                else:
                    pr = fake_e4m3(pr, k, rnd, dim=(-2, -1)).float()  # per-block P scale
                out[:, h, r0:r1] = (pr @ vh).to(q.dtype)
        return out
    return attn


def parse_spec(spec):
    """'fmt[;wa=w][;keep=REGEX][;keep_fmt=FMT][;head=1][;attn=MODE]' -> dict."""
    parts = spec.split(";")
    d = dict(fmt=parts[0], wa="wa", keep="", keep_fmt="bf16", head=False, attn="")
    for p in parts[1:]:
        key, val = p.split("=", 1)
        d[key] = (val == "1") if key == "head" else val
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--specs", nargs="+", default=["bf16"],
                    help="one or more 'fmt[;wa=w][;keep=REGEX][;keep_fmt=FMT][;head=1]'; the model is "
                         "loaded once (CPU) and copied to the GPU for each spec")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--layer-stats", action="store_true")
    ap.add_argument("--data-cache", default="/nfs/home/koying/models/datasets")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import copy

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=a.data_cache)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cpu")
    base.eval()
    nwin = ids.shape[1] // a.seqlen
    if a.max_windows:
        nwin = min(nwin, a.max_windows)

    for spec in a.specs:
        c = parse_spec(spec)
        model = copy.deepcopy(base).cuda()
        nq = replace_linears(model, c["fmt"], c["wa"], c["keep"], c["keep_fmt"], c["head"], a.layer_stats)
        F.scaled_dot_product_attention = make_attention(c["attn"]) if c["attn"] else _SDPA
        nll, ntok = 0.0, 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(nwin):
                x = ids[:, i * a.seqlen:(i + 1) * a.seqlen].cuda()
                logits = model(x).logits.float()
                nll += F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
                ntok += a.seqlen - 1
        res = dict(tag=a.tag, model=os.path.basename(a.model.rstrip("/")), spec=spec, **c, seqlen=a.seqlen,
                   windows=nwin, tokens=ntok, ppl=math.exp(nll / ntok), n_linear=nq,
                   seconds=round(time.time() - t0, 1))
        if a.layer_stats:
            res["layer_err"] = {m.name: m.stats for m in model.modules() if isinstance(m, QLinear) and m.stats is not None}
        print("RESULT " + json.dumps(res), flush=True)
        F.scaled_dot_product_attention = _SDPA
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
