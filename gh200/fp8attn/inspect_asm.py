"""Compile the v2 kernels with the current CFG and print an instruction mix of the SASS
(no Nsight Compute on this cluster: ERR_NVGPUCTRPERM), plus TTGIR layout conversions."""
import collections
import json
import os
import re
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
import importlib  # noqa
fp8attn = importlib.import_module(os.environ.get("IMPL", "fp8attn2"))
from common import S, HQ, HKV, D  # noqa

cfg = dict(fp8attn.CFG)
cfg.update(json.loads(os.environ.get("CFG", "{}")))
torch.manual_seed(0)
q = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(S, HKV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(S, HQ, D, device="cuda", dtype=torch.bfloat16)
fo = fp8attn.fp8_attn_fwd(q, k, v, cfg=cfg)
fp8attn.fp8_attn_bwd(q, k, v, fo[0], do, *fo[1:], matched=True, cfg=cfg)
torch.cuda.synchronize()

OUT = os.environ.get("ASM_DIR", "/nfs/home/koying/fp8attn-build/asm")
os.makedirs(OUT, exist_ok=True)
for name in ("_fwd_kernel", "_bwd_dq_kernel", "_bwd_dkdv_kernel"):
    fn = getattr(fp8attn, name)
    for dev_cache in fn.device_caches.values() if hasattr(fn, "device_caches") else [fn.cache]:
        cache = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
        for key, kern in list(cache.items())[-1:]:
            asm = kern.asm
            txt = {}
            for ext in ("ttgir", "ptx", "sass"):
                try:
                    txt[ext] = asm[ext]
                    open(f"{OUT}/{name}.{ext}", "w").write(txt[ext])
                except Exception as e:  # noqa
                    print("   no", ext, repr(e)[:100])
            ttgir = txt.get("ttgir", "")
            print(f"== {name}: regs={kern.n_regs} spills={kern.n_spills} smem={kern.metadata.shared}")
            print("   ttgir convert_layout:", ttgir.count("ttg.convert_layout"), " local_alloc:", ttgir.count("ttg.local_alloc"),
                  " warp_group_dot:", ttgir.count("warp_group_dot "), " local_load:", ttgir.count("ttg.local_load"))
            sass = txt.get("sass", "")
            ops = collections.Counter()
            for line in sass.splitlines():
                m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?", line)
                if m:
                    ops[m.group(2)] += 1
            top = ", ".join(f"{o}:{c}" for o, c in ops.most_common(40))
            print("   SASS static mix:", top)
