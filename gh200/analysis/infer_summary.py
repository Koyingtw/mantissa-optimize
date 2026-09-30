"""Collect infer/ppl_eval RESULT lines from logs/infer/ppl_*.out into results/infer_ppl.csv and print
a model x spec table of perplexity change (%) vs that model's native BF16 at the same sequence length."""
import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("logs/infer/ppl_*.out")):
    for line in open(f):
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r.pop("layer_err", None)
            r["job"] = int(f.split("_")[-1].split(".")[0])
            rows.append(r)
d = pd.DataFrame(rows)
d = d[d.windows > 8]  # drop smoke tests
d = d.sort_values("job").drop_duplicates(["model", "seqlen", "spec"], keep="last")
base = d[d.spec == "bf16"].set_index(["model", "seqlen"]).ppl
d["ppl_bf16"] = [base.get((m, s)) for m, s in zip(d.model, d.seqlen)]
d["delta_pct"] = (d.ppl / d.ppl_bf16 - 1) * 100
d.to_csv("results/infer_ppl.csv", index=False)
d["col"] = d.model + "@" + d.seqlen.astype(str)
t = d.pivot_table(index="spec", columns="col", values="delta_pct", sort=False)
pd.set_option("display.width", 250)
print("ppl bf16:", base.round(4).to_dict())
print(t.round(2).to_string())
