"""WikiText-2 perplexity on MI350X (scripts/ai_ppl.sh) against GH200 (gh200/results/infer_ppl.csv).

For every model and format: perplexity, change relative to that platform's native BF16 (percent), and the GH200 change
for the same model and format where it was measured. FP8 runs on each platform's own FP8 matrix engines (torch._scaled_mm),
so small cross-platform differences in the FP8 rows are expected; the BF16 rows should agree closely.
usage: ppl_summary.py <run id> [...]   (logs in ../logs) → results/ai_ppl.csv, prints the delta table
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
GH = os.path.join(HERE, "..", "..", "gh200", "results", "infer_ppl.csv")
GH_NAME = {"Llama-3.1-8B": "Meta-Llama-3.1-8B", "Llama-2-70b-hf": "Llama2-70B"}  # model names used on GH200


def main():
    rows = []
    for rid in sys.argv[1:]:
        for line in open(f"{LOG}/{rid}.out"):
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                rows.append(dict(model=r["model"], spec=r["spec"], ppl=r["ppl"]))
    d = pd.DataFrame(rows).drop_duplicates(["model", "spec"], keep="last")
    base = d[d.spec == "bf16"].set_index("model").ppl
    d["delta_pct"] = 100 * (d.ppl / d.model.map(base) - 1)
    gh = pd.read_csv(GH)
    gh = gh[gh["attn"].isna() & gh["keep"].isna() & (gh["head"].astype(str) == "False")]
    ghd = gh.drop_duplicates(["model", "spec"], keep="last").set_index(["model", "spec"])
    d["gh200_delta_pct"] = [ghd.delta_pct.get((GH_NAME.get(m, m), s)) for m, s in zip(d.model, d.spec)]
    d["gh200_ppl_bf16"] = [ghd.ppl_bf16.get((GH_NAME.get(m, m), "bf16")) for m in d.model]
    d.to_csv(os.path.join(HERE, "..", "results", "ai_ppl.csv"), index=False)
    order = list(dict.fromkeys(d.spec))
    t = d.pivot_table(index="spec", columns="model", values="delta_pct").reindex(order)
    g = d.pivot_table(index="spec", columns="model", values="gh200_delta_pct").reindex(order)
    pd.set_option("display.width", 220)
    print("native BF16 perplexity (MI350X / GH200):")
    print(d[d.spec == "bf16"][["model", "ppl", "gh200_ppl_bf16"]].to_string(index=False))
    print("\nchange vs native BF16 (%), MI350X:")
    print(t.round(2).to_string())
    print("\nsame on GH200 (%):")
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
