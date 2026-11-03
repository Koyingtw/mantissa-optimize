"""Per-step speed of LoRA runs (ai/lora.py via scripts/ai_lora.sh): median step time from step 4 on, mean power and
gfxclk over the training window of each GPU (sampler joined by BDF), and the speedup of each truncated run over the
native run with the same seed. Several run ids pool into one table, so a run and its GPU-swapped repeat (SWAP=1)
separate the bit count from the GPU.
usage: lora_speed.py <run id> [...]   (logs in ../logs) → prints the table; results/ai_lora_speed.csv
"""
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from e1_summary import BDF  # noqa: E402
from smi import load_smi  # noqa: E402


def main():
    out = []
    for rid in sys.argv[1:]:
        smi = load_smi(f"{LOG}/{rid}_smi.csv")
        for f in sorted(glob.glob(f"{LOG}/{rid}_gpu*.log")):
            st, steps = None, []
            for line in open(f):
                if line.startswith("START"):
                    st = json.loads(line[6:])
                elif line.startswith("STEP"):
                    steps.append(json.loads(line[5:]))
            s = pd.DataFrame(steps)
            s = s[s.step >= 4]
            g = BDF.index(st["bdf"])
            w = smi[(smi.gpu == g) & (smi.t >= s.t_begin.min()) & (smi.t <= s.t_end.max())]
            out.append(dict(run=rid, bits=st["bits"], seed=st["seed"], bdf=st["bdf"], steps=len(s),
                            step_s=s.seconds.median(), power_w=w.power_w.mean(), gfxclk=w.gfxclk.mean(),
                            t_hot=w.t_hot.mean()))
    d = pd.DataFrame(out)
    pairs = []
    for (rid, seed), g in d.groupby(["run", "seed"]):
        n = g[g.bits == 7]
        for r in g[g.bits < 7].itertuples():
            pairs.append(dict(run=rid, seed=seed, bits=r.bits, bdf_trunc=r.bdf, bdf_native=n.bdf.iloc[0],
                              speedup_pct=100 * (n.step_s.iloc[0] / r.step_s - 1)))
    p = pd.DataFrame(pairs)
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    d.to_csv(os.path.join(HERE, "..", "results", "ai_lora_speed.csv"), index=False)
    pd.set_option("display.width", 200)
    print(d.sort_values(["run", "bdf"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(p.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    # per GPU: step time with each bit count across runs (separates the GPU from the bit count)
    print(d.pivot_table(index="bdf", columns="bits", values="step_s").round(3).to_string())


if __name__ == "__main__":
    main()
