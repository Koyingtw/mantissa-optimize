"""Per-call method selector for FP64 GEMMs under a power cap: native, truncation or INT8 emulation (GEMMul8)
(review round 5, M4).

Time model, relative to native FP64 at the same cap P (all fitted on GEMM sweeps only, never on applications):
  truncation to b fraction bits:  r_t(P)       = 1 / (1 + g_t(P)),   g_t = mean measured speedup over the fitted shapes
  emulation with M moduli:        r_e(m,n,k,P) = a_PM + b_PM / k + c_PM (1/m + 1/n)
                                  (+ a one-time setup s_M seconds the first time a shape is seen)
Accuracy: the caller gives the tolerated relative error of the product (eps); truncation needs b with the measured
error of that bit count <= eps, emulation the fewest moduli whose measured error <= eps.
Memory: emulation is ruled out if GEMMul8's workspace does not fit (not modelled here; the HPL cases are marked).
Decision: the method with the smallest predicted time among those that meet eps.

usage:
  selector.py fit <e1 id> [...]          fit and cross-validate (leave one shape out) → results/selector_fit.csv,
                                         results/selector_cv.csv
  selector.py dfmp2 <cap> <eps> [setup]  per-call decisions for the DF-MP2 shapes (m=2002, n=(i+1)2002, k=5477):
                                         threshold n* above which emulation is predicted faster
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIT = os.path.join(RES, "selector_model.json")
EMU = ("g8n", "g8")  # preferred emulation build first (v3.5.4), then 3.2.0


def load(ids):
    s = pd.concat([pd.read_csv(f"{RES}/{i}_summary.csv").assign(id=i) for i in ids])
    s = s[(s.low == "zero") & (s.spread == 0)]
    s["r"] = 1 / (1 + s.speedup_pct / 100)  # time relative to native at the same cap and shape
    return s


def design(m, n, k):
    return np.column_stack([np.ones(len(m)), 1 / np.asarray(k, float), 1 / np.asarray(m, float) + 1 / np.asarray(n, float)])


def fit(s):
    model = {"emu": {}, "trunc": {}, "err": {}}
    for (dt, M, cap), g in s[s.dtype.isin(EMU)].groupby(["dtype", "frac", "cap"]):
        X = design(g.m, g.n, g.k)
        coef, *_ = np.linalg.lstsq(X, g.r.values, rcond=None)
        pred = X @ coef
        r2 = 1 - ((g.r - pred) ** 2).sum() / max(((g.r - g.r.mean()) ** 2).sum(), 1e-12)
        model["emu"][f"{dt}:{M}:{cap}"] = dict(coef=coef.tolist(), r2=float(r2), n=len(g))
    for (b, cap), g in s[(s.dtype == "fp64") & (s.frac < 52)].groupby(["frac", "cap"]):
        model["trunc"][f"{b}:{cap}"] = dict(gain=float(g.speedup_pct.mean() / 100), sd=float(g.speedup_pct.std() / 100),
                                            n=len(g))
    for (dt, M), g in s[s.dtype.isin(EMU + ("fp64",))].groupby(["dtype", "frac"]):
        if dt == "fp64" and M == 52:
            continue
        model["err"][f"{dt}:{M}"] = float(g.rel_err.max())
    return model


def r_emu(model, dt, M, cap, m, n, k):
    c = model["emu"].get(f"{dt}:{M}:{cap}")
    return None if c is None else float((design([m], [n], [k]) @ np.array(c["coef"]))[0])


def choose(model, m, n, k, cap, eps, t_native=None, setup=0.0, calls=1, emu=None):
    """Return (method, relative time) for one GEMM shape; t_native (s) is needed only to weigh the setup cost."""
    if emu is None:  # the preferred emulation build that the model has data for
        emu = next((e for e in EMU if any(key.startswith(f"{e}:") for key in model["err"])), EMU[0])
    cands = [("native", 1.0)]
    bits = [int(key.split(":")[1]) for key in model["err"] if key.startswith("fp64:") and model["err"][key] <= eps]
    for b in sorted(bits):
        g = model["trunc"].get(f"{b}:{cap}")
        if g:
            cands.append((f"trunc{b}", 1 / (1 + g["gain"])))
            break
    mods = sorted(int(key.split(":")[1]) for key in model["err"] if key.startswith(f"{emu}:") and model["err"][key] <= eps)
    if mods:
        r = r_emu(model, emu, mods[0], cap, m, n, k)
        if r is not None:
            if setup and t_native:
                r += setup / (calls * t_native)
            cands.append((f"emu{mods[0]}", r))
    return min(cands, key=lambda c: c[1])


def cmd_fit(ids):
    s = load(ids)
    model = fit(s)
    json.dump(model, open(FIT, "w"), indent=1)
    rows = []
    for key, c in model["emu"].items():
        dt, M, cap = key.split(":")
        rows.append(dict(method=dt, moduli=int(M), cap=int(cap), a=c["coef"][0], b=c["coef"][1], c=c["coef"][2],
                         r2=c["r2"], n=c["n"]))
    pd.DataFrame(rows).to_csv(f"{RES}/selector_fit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    # leave one shape out: refit without the shape, predict the winner among truncation (35) and emulation (12 moduli)
    shapes = s[["m", "n", "k"]].drop_duplicates().values.tolist()
    cv = []
    for m, n, k in shapes:
        tr = s[~((s.m == m) & (s.n == n) & (s.k == k))]
        te = s[(s.m == m) & (s.n == n) & (s.k == k)]
        if tr[["m", "n", "k"]].drop_duplicates().shape[0] < 3:
            continue
        mod = fit(tr)
        for cap, g in te.groupby("cap"):
            meas = {}
            for r in g.itertuples():
                if r.dtype == "fp64" and r.frac == 52:
                    meas["native"] = 1.0
                elif r.dtype == "fp64":
                    meas[f"trunc{r.frac}"] = r.r
                elif r.dtype == EMU[0] or (r.dtype in EMU and EMU[0] not in set(g.dtype)):
                    meas[f"emu{r.frac}"] = r.r
            for eps in (1e-10, 1e-12):
                pick, pr = choose(mod, m, n, k, cap, eps, emu=EMU[0] if EMU[0] in set(g.dtype) else EMU[1])
                ok = [x for x in meas if x == "native" or
                      (x.startswith("trunc") and mod["err"].get(f"fp64:{x[5:]}", 1) <= eps) or
                      (x.startswith("emu") and mod["err"].get(f"{EMU[0]}:{x[3:]}", mod["err"].get(f"{EMU[1]}:{x[3:]}", 1)) <= eps)]
                best = min(ok, key=lambda x: meas[x])
                got = meas.get(pick, np.nan)
                cv.append(dict(m=m, n=n, k=k, cap=cap, eps=eps, pick=pick, pred_r=pr, best=best, best_r=meas[best],
                               pick_r=got, regret_pct=100 * (got / meas[best] - 1), correct=pick == best))
    c = pd.DataFrame(cv)
    c.to_csv(f"{RES}/selector_cv.csv", index=False)
    if len(c):
        print(c.groupby(["eps", "cap"]).agg(n=("correct", "size"), correct=("correct", "mean"),
                                            mean_regret=("regret_pct", "mean"), max_regret=("regret_pct", "max")))


def cmd_dfmp2(cap, eps, setup=0.0):
    model = json.load(open(FIT))
    nvir, naux, nocc = 2002, 5477, 226
    rows = []
    for i in range(nocc):
        n = (i + 1) * nvir
        rows.append((i, n) + choose(model, nvir, n, naux, cap, eps))
    d = pd.DataFrame(rows, columns=["i", "n", "pick", "r"])
    emu = d[d.pick.str.startswith("emu")]
    print(d.pick.value_counts().to_string())
    print("n* =", int(emu.n.min()) if len(emu) else "never")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "fit":
        cmd_fit(sys.argv[2:])
    elif sys.argv[1] == "dfmp2":
        cmd_dfmp2(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
