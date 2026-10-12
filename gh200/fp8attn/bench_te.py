"""Stage 0 baseline: TE 2.2 DotProductAttention (cuDNN fused attn) on the Llama2-70B LoRA shape.

Usage (one mode per process, since NVTE_* env vars are read inside TE):
  MODE=bf16      -> bf16 fwd + bf16 bwd
  MODE=fp8       -> fp8 fwd + fp8 bwd   (NVTE_FP8_DPA_BWD=1, what training ran with)
  MODE=fp8_bf16bwd -> fp8 fwd + bf16 bwd (NVTE_FP8_DPA_BWD=0)
  MODE=fa2       -> flash-attn 2.7.3 bf16 (via flash_attn_func directly)
"""
import os
import sys
import torch

MODE = os.environ.get("MODE", "bf16")
if MODE == "fp8":
    os.environ["NVTE_FP8_DPA_BWD"] = "1"
elif MODE == "fp8_bf16bwd":
    os.environ["NVTE_FP8_DPA_BWD"] = "0"
os.environ.setdefault("NVTE_FUSED_ATTN", "1")
os.environ.setdefault("NVTE_FLASH_ATTN", "0")

sys.path.insert(0, os.path.dirname(__file__))
from common import S, B, HQ, HKV, D, bench, flops_fwd, flops_bwd, tflops, sm_clock  # noqa

torch.manual_seed(0)
dev = "cuda"
q = torch.randn(S, B, HQ, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(S, B, HKV, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(S, B, HKV, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
do = torch.randn(S, B, HQ * D, device=dev, dtype=torch.bfloat16)

if MODE in ("fa2", "fa3", "fa3fp8"):
    if MODE == "fa2":
        from flash_attn import flash_attn_func
    else:
        import flash_attn_interface as _fa3

        def flash_attn_func(a, b, c, causal):
            if MODE == "fa3fp8":
                a, b, c = (t.to(torch.float8_e4m3fn) for t in (a, b, c))
            o = _fa3.flash_attn_func(a, b, c, causal=causal)
            return o[0] if isinstance(o, tuple) else o
    qb = q.detach().permute(1, 0, 2, 3).contiguous().requires_grad_()
    kb = k.detach().permute(1, 0, 2, 3).contiguous().requires_grad_()
    vb = v.detach().permute(1, 0, 2, 3).contiguous().requires_grad_()
    dob = torch.randn(B, S, HQ, D, device=dev, dtype=torch.bfloat16)

    def fwd():
        return flash_attn_func(qb, kb, vb, causal=True)
    out_holder = {}

    def setup_bwd():
        if MODE == "fa3fp8":
            raise SystemExit(f"RESULT fa3fp8 fwd_ms={tf:.3f} fwd_TFLOPS={tflops(flops_fwd(), tf):.0f} (no FP8 bwd in FA3)")
        for t in (qb, kb, vb):
            t.grad = None
        out_holder["o"] = fwd()

    def bwd():
        out_holder["o"].backward(dob)
else:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    attn = te.DotProductAttention(HQ, D, num_gqa_groups=HKV, attention_dropout=0.0,
                                  qkv_format="sbhd", attn_mask_type="causal").to(dev)
    use_fp8 = MODE.startswith("fp8")
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=128,
                            amax_compute_algo="most_recent", fp8_dpa=use_fp8)

    def fwd():
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=recipe):
            return attn(q, k, v)
    out_holder = {}

    def setup_bwd():
        for t in (q, k, v):
            t.grad = None
        out_holder["o"] = fwd()

    def bwd():
        out_holder["o"].backward(do)

print(f"MODE={MODE} S={S} B={B} HQ={HQ} HKV={HKV} D={D}  clock/temp/power before: {sm_clock()}", flush=True)
with torch.no_grad():
    tf, tf_min = bench(fwd, warmup=5, iters=30)
tb, tb_min = bench(bwd, warmup=5, iters=30, setup=setup_bwd)
print(f"clock/temp/power after: {sm_clock()}")
print(f"RESULT {MODE} fwd_ms={tf:.3f} (min {tf_min:.3f}) fwd_TFLOPS={tflops(flops_fwd(), tf):.0f}  "
      f"bwd_ms={tb:.3f} (min {tb_min:.3f}) bwd_TFLOPS={tflops(flops_bwd(), tb):.0f}", flush=True)

# Kernel-level breakdown of one fwd+bwd (which kernels, how long)
from torch.profiler import profile, ProfilerActivity
setup_bwd(); bwd(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    setup_bwd()
    bwd()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=90))
