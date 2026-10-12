#!/usr/bin/env python3
"""Offline measurement of FP8 attention-backward error and the stale-delta row-sum residual
on real tensors dumped from our MLPerf Llama2-70B LoRA run (gh200/fp8attn/attn_dump.py).

    python residual_measure.py /nfs/home/koying/fp8attn-dumps/L40_step6.pt [...] --out results/

Per dump (one layer, one training step; q/k/v/o/dO/dq/dk/dv, see the dump README):

  ref        FP64 attention fwd+bwd from the dumped bf16 q, k, v, dO (causal, GQA 64/8,
             head_dim 128, softmax scale 1/sqrt(128), dropout 0 -- all read from the dump meta)
  te_*       emulation of TE 2.2 / cuDNN FP8 attention with the *actual* delayed-scaling scales
             captured from the run: HYBRID recipe = E4M3 for Q,K,V,P(S),O; E5M2 for dO, dS(dP), dQKV.
             fwd: P_unnorm=exp(S-m) quantised with s_S, O=(P^ V^)/l quantised with s_O.
             bwd: P recomputed in fp32 from Q^K^T, dP = dO^ V^T (fp32 acc.), dS = P*(dP - delta),
             dS quantised with s_dP, dQ = scale*dS^ K^, dK = scale*dS^T Q^, dV = P^T dO^,
             outputs quantised with s_dQKV (E5M2) -- exactly the tensors TE hands cuDNN.
             te_stale:  delta_i = <dO^_i, O^_i>, O^ = TE's own O_fp8 (reconstructed from the dump).
             te_dm:     delta_i = sum_j P_ij dP^_ij   (Delta-Matching).
  cur_*      same formats, per-tensor *current* scaling (scale = fmax / amax of this step's tensor).
  blk_*      block-scaled E4M3 everywhere (1x128 blocks along the last/contraction dim, current
             amax per block), the paper's setting.
  te_actual  dq/dk/dv that TE/cuDNN actually returned in the run.

Metrics: normalised row-sum residual r_i = sum_j dS_ij / sum_j |dS_ij| (before and after dS
quantisation), its median/p99 of |r| and signed mean; rel. error / cosine of dq, dk, dv vs ref;
projection of the dq error on Kbar_i = sum_j P_ij K_j (the direction a row-sum residual eps_i
writes into: dq_i gets -scale*eps_i*Kbar_i), incl. the regression of TE's actual dq error on the
stale-delta prediction -scale*e_i*Kbar_i, and on the global mean-K direction.
"""
import argparse
import glob
import json
import math
import os
import time

import torch

E4M3, E5M2 = "e4m3", "e5m2"
FMAX = {E4M3: 448.0, E5M2: 57344.0}
FDT = {E4M3: torch.float8_e4m3fn, E5M2: torch.float8_e5m2}
dev = "cuda"
f64 = torch.float64


def qdq(x, scale, fmt):
    """Quantise-dequantise x (fp64) with a scale (float or broadcastable tensor)."""
    y = (x * scale).float().clamp_(-FMAX[fmt], FMAX[fmt]).to(FDT[fmt]).to(f64)
    return y / scale


def qdq_block(x, fmt, bs=128):
    """Block quantisation along the last dim, blocks of bs, current amax per block."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // bs, bs)
    amax = xb.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, FMAX[fmt] / amax, torch.ones_like(amax))
    return qdq(xb, scale, fmt).reshape(shp)


class Scheme:
    """Quantisers for one emulated FP8 scheme. kind: ref | te | cur | blk."""

    def __init__(self, kind, s=None):
        self.kind, self.s = kind, s or {}

    def _pt(self, x, name, fmt):
        return qdq(x, self.s[name], fmt)

    def q(self, x, name):  # forward operands / outputs
        if self.kind == "ref":
            return x
        if self.kind == "blk":
            return qdq_block(x, E4M3)
        fmt = E4M3 if name in ("q", "k", "v", "s", "o") else E5M2
        return self._pt(x, name, fmt)


def stats_rows(r):
    a = r.abs()
    n = r.numel()
    if n == 0:  # e.g. dS flushed to zero entirely (FP8 cold start)
        return {"n": 0, "median_abs": float("nan"), "p90_abs": float("nan"), "p99_abs": float("nan"),
                "mean_abs": float("nan"), "mean_signed": float("nan"), "frac_pos": float("nan"),
                "t_stat": float("nan")}
    sd = r.std().item() if n > 1 else float("nan")
    return {
        "n": n,
        "median_abs": a.median().item(),
        "p90_abs": torch.quantile(a[: 16_000_000].float(), 0.90).item(),
        "p99_abs": torch.quantile(a[: 16_000_000].float(), 0.99).item(),
        "mean_abs": a.mean().item(),
        "mean_signed": r.mean().item(),
        "frac_pos": (r > 0).double().mean().item(),
        "t_stat": r.mean().item() / (sd / math.sqrt(n)) if sd and sd > 0 else float("nan"),
    }


def run(path, out_dir, heads=None, row_min=128):
    t0 = time.time()
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = d["meta"]
    S_, Hq, D = d["q"].shape
    Hk = d["k"].shape[1]
    G = Hq // Hk
    assert m["attention_dropout"] in (0, 0.0, None), m["attention_dropout"]
    assert "causal" in m["attn_mask_type"], m["attn_mask_type"]
    sm = m["softmax_scale"] or 1.0 / math.sqrt(D)
    idx = m["meta_indices"]
    sf = m["fp8_state_fwd"]["scaling_fwd"]["scale"]
    sb = m["fp8_state_before_bwd"]["scaling_bwd"]["scale"]
    ab = m["fp8_state_after_bwd"]["scaling_bwd"]["amax_history"][0]
    af = m["fp8_state_fwd"]["scaling_fwd"]["amax_history"][0]
    te_s = {"q": sf[idx["META_QKV"]].item(), "k": sf[idx["META_QKV"]].item(), "v": sf[idx["META_QKV"]].item(),
            "s": sf[idx["META_S"]].item(), "o": sf[idx["META_O"]].item(),
            "do": sb[idx["META_DO"]].item(), "dp": sb[idx["META_DP"]].item(), "dqkv": sb[idx["META_DQKV"]].item()}
    X = {k: d[k].to(dev, f64) for k in ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
    if X["do"].abs().max() == 0:
        print(f"{path}: dO is all zero (FP8 delayed-scaling cold start), skipped", flush=True)
        return None
    heads = list(range(Hq)) if heads is None else heads
    full = len(heads) == Hq
    mask = torch.ones(S_, S_, dtype=torch.bool, device=dev).tril_()
    rows = torch.arange(S_, device=dev)

    # ---------------- reference pass (also collects current-step amaxes) -------------
    ref = {"dq": torch.zeros(S_, Hq, D, dtype=f64, device=dev),
           "dk": torch.zeros(S_, Hk, D, dtype=f64, device=dev),
           "dv": torch.zeros(S_, Hk, D, dtype=f64, device=dev),
           "o": torch.zeros(S_, Hq, D, dtype=f64, device=dev),
           "kbar": torch.zeros(S_, Hq, D, dtype=f64, device=dev)}
    amax = {"o": 0.0, "dp": 0.0, "ds": 0.0, "ds_scaled": 0.0}
    for h in heads:
        g = h // G
        q, k, v, do = X["q"][:, h], X["k"][:, g], X["v"][:, g], X["do"][:, h]
        S = (q @ k.T) * sm
        S.masked_fill_(~mask, float("-inf"))
        P = torch.softmax(S, dim=1)
        del S
        O = P @ v
        dP = do @ v.T
        delta = (do * O).sum(1)
        dS = P * (dP - delta[:, None])
        ref["o"][:, h] = O
        ref["kbar"][:, h] = P @ k
        ref["dq"][:, h] = sm * dS @ k
        ref["dk"][:, g] += sm * dS.T @ q
        ref["dv"][:, g] += P.T @ do
        amax["o"] = max(amax["o"], O.abs().max().item())
        amax["dp"] = max(amax["dp"], dP.abs().max().item())
        amax["ds"] = max(amax["ds"], dS.abs().max().item())
        amax["ds_scaled"] = max(amax["ds_scaled"], sm * dS.abs().max().item())
        del P, dP, dS, O
    amax["dq"], amax["dk"], amax["dv"] = (ref[n].abs().max().item() for n in ("dq", "dk", "dv"))
    amax["dqkv"] = max(amax["dq"], amax["dk"], amax["dv"])
    # Which tensor does TE's "dP" quantiser see?  Compare with cuDNN's recorded amax of this step.
    te_amax_dp = ab[idx["META_DP"]].item()
    cands = {"dS": amax["ds"], "scale*dS": amax["ds_scaled"], "dP": amax["dp"]}
    dp_is = min(cands, key=lambda c: abs(math.log(max(cands[c], 1e-30) / max(te_amax_dp, 1e-30))))
    dp_mult = sm if dp_is == "scale*dS" else 1.0

    cur_s = {"q": FMAX[E4M3] / X["q"].abs().max().item(), "k": FMAX[E4M3] / X["k"].abs().max().item(),
             "v": FMAX[E4M3] / X["v"].abs().max().item(), "s": FMAX[E4M3] / 1.0,
             "o": FMAX[E4M3] / amax["o"], "do": FMAX[E5M2] / X["do"].abs().max().item(),
             "dp": FMAX[E5M2] / (amax["ds"] * dp_mult), "dqkv": FMAX[E5M2] / amax["dqkv"]}
    schemes = {"te": Scheme("te", te_s), "cur": Scheme("cur", cur_s), "blk": Scheme("blk")}

    # TE's own O_fp8, reconstructed from the dumped bf16 dequantised output
    o_te_q = qdq(X["o"], te_s["o"], E4M3)
    recon_o_err = ((o_te_q - X["o"]).norm() / X["o"].norm()).item()

    res = {}
    for sname, sch in schemes.items():
        acc = {dl: {"dq": torch.zeros(S_, Hq, D, dtype=f64, device=dev),
                    "dk": torch.zeros(S_, Hk, D, dtype=f64, device=dev),
                    "dv": torch.zeros(S_, Hk, D, dtype=f64, device=dev),
                    "r_pre": [], "r_post": [], "e_rel": [], "e": []} for dl in ("stale", "dm")}
        if sname == "te":  # decomposition: same pipeline but dS kept in fp32 (no E5M2 on dS)
            for dl in ("stale", "dm"):
                acc[f"{dl}_dSfp32"] = {"dq": torch.zeros(S_, Hq, D, dtype=f64, device=dev),
                                       "dk": torch.zeros(S_, Hk, D, dtype=f64, device=dev),
                                       "dv": torch.zeros(S_, Hk, D, dtype=f64, device=dev)}
        o_emul = torch.zeros(S_, Hq, D, dtype=f64, device=dev)
        Q = sch.q(X["q"], "q")
        K = sch.q(X["k"], "k")
        V = sch.q(X["v"], "v")
        dO = sch.q(X["do"], "do")
        for h in heads:
            g = h // G
            q, k, v, do = Q[:, h], K[:, g], V[:, g], dO[:, h]
            S = (q @ k.T) * sm
            S.masked_fill_(~mask, float("-inf"))
            mx = S.amax(1, keepdim=True)
            E = torch.exp(S - mx)
            del S
            l = E.sum(1, keepdim=True)
            P = E / l
            # forward output as cuDNN produces it: unnormalised P quantised (S quantiser), then 1/l
            O = (sch.q(E, "s") @ v) / l if sname != "blk" else (qdq_block(E, E4M3) @ v) / l
            del E
            Oq = sch.q(O, "o")
            o_emul[:, h] = Oq
            O_for_delta = o_te_q[:, h] if sname == "te" else Oq
            dP = do @ v.T
            Pb = sch.q(P, "s") if sname != "blk" else qdq_block(P, E4M3)
            dv_h = Pb.T @ do
            del Pb
            delta_dm = (P * dP).sum(1)
            delta_st = (do * O_for_delta).sum(1)
            for dl, delta in (("stale", delta_st), ("dm", delta_dm)):
                a = acc[dl]
                dS = P * (dP - delta[:, None])
                absum = dS.abs().sum(1)
                ok = (absum > 0) & (rows >= row_min)
                a["r_pre"].append((dS.sum(1)[ok] / absum[ok]).float())
                a["e"].append((delta - delta_dm).float())
                a["e_rel"].append(((delta - delta_dm)[ok] / absum[ok]).float())
                if sname == "te":
                    a2 = acc[f"{dl}_dSfp32"]
                    a2["dq"][:, h] = sm * dS @ k
                    a2["dk"][:, g] += sm * dS.T @ q
                    a2["dv"][:, g] += dv_h
                if sname == "blk":
                    dSq = qdq_block(dS, E4M3)
                else:
                    dSq = qdq(dS * dp_mult, sch.s["dp"], E5M2) / dp_mult
                del dS
                absq = dSq.abs().sum(1)
                okq = (absq > 0) & (rows >= row_min)
                a["r_post"].append((dSq.sum(1)[okq] / absq[okq]).float())
                a["dq"][:, h] = sm * dSq @ k
                a["dk"][:, g] += sm * dSq.T @ q
                a["dv"][:, g] += dv_h
                del dSq
            del P, dP, O, Oq
        for dl, a in acc.items():
            for n in ("dq", "dk", "dv"):
                if sname == "blk":
                    a[n] = qdq_block(a[n], E4M3)
                else:
                    a[n] = qdq(a[n], sch.s["dqkv"], E5M2)
            res[f"{sname}_{dl}"] = a
        hs = torch.tensor(heads, device=dev)
        res[f"{sname}_o_vs_dump"] = ((o_emul - X["o"])[:, hs].norm() / X["o"][:, hs].norm()).item()
        res[f"{sname}_o_vs_ref"] = ((o_emul - ref["o"])[:, hs].norm() / ref["o"][:, hs].norm()).item()
        del o_emul, Q, K, V, dO
    res["te_actual"] = {"dq": X["dq"], "dk": X["dk"], "dv": X["dv"]}

    # ---------------- metrics ----------------
    kbar = ref["kbar"]
    kb_n2 = (kbar * kbar).sum(-1).clamp_min(1e-300)
    kmean = X["k"].mean(0)  # [Hk, D] global mean-K direction per kv head
    kmean_q = kmean.repeat_interleave(G, 0)  # [Hq, D]
    kmean_q = kmean_q / kmean_q.norm(dim=-1, keepdim=True)
    e_pred = torch.cat(res["te_stale"]["e"]).view(len(heads), S_).T  # [S, H] (rows ordered h-major)
    hsel = torch.tensor(heads, device=dev)
    pred_c = -sm * e_pred  # predicted coefficient on Kbar for the stale-delta bias

    def errs(name, a):
        out = {}
        for n in (("dq", "dk", "dv") if full else ("dq",)):
            x, r = a[n], ref[n]
            if n == "dq":
                x, r = x[:, hsel], r[:, hsel]
            out[f"{n}_relerr"] = ((x - r).norm() / r.norm()).item()
            out[f"{n}_cos"] = (torch.nn.functional.cosine_similarity(x.flatten(), r.flatten(), dim=0)).item()
            out[f"{n}_zero_frac"] = (x == 0).double().mean().item()
        E = (a["dq"] - ref["dq"])[:, hsel]  # [S, H, D]
        kb, kn2 = kbar[:, hsel], kb_n2[:, hsel]
        c = (E * kb).sum(-1) / kn2  # coefficient of the error on Kbar_i
        rows_ok = rows >= row_min
        Eo = E[rows_ok]
        e2 = (Eo * Eo).sum()
        out["dq_err_energy_on_Kbar"] = ((c[rows_ok] ** 2) * kn2[rows_ok]).sum().item() / e2.item()
        cg = (E * kmean_q[hsel][None]).sum(-1)
        out["dq_err_energy_on_meanK"] = (cg[rows_ok] ** 2).sum().item() / e2.item()
        pc = pred_c[rows_ok]
        cc = c[rows_ok]
        out["c_vs_pred_corr"] = torch.corrcoef(torch.stack([cc.flatten(), pc.flatten()]))[0, 1].item()
        out["c_vs_pred_slope"] = ((cc * pc).sum() / (pc * pc).sum()).item()
        resid = Eo - (pc[..., None] * kb[rows_ok])
        out["dq_err_explained_by_stale_pred"] = 1.0 - (resid * resid).sum().item() / e2.item()
        out["c_mean_signed_over_rms"] = (cc.mean() / cc.pow(2).mean().sqrt()).item()
        if "r_pre" in a:
            out["rowres_pre"] = stats_rows(torch.cat(a["r_pre"]))
            out["rowres_post"] = stats_rows(torch.cat(a["r_post"]))
            out["delta_mismatch_rel"] = stats_rows(torch.cat(a["e_rel"]))
        return out

    metrics = {k: errs(k, v) for k, v in res.items() if isinstance(v, dict)}
    # TE actual vs our exact-recipe emulation (validates the emulation)
    emu = {}
    for n in (("dq", "dk", "dv") if full else ("dq",)):
        sel = (lambda t: t[:, hsel]) if n == "dq" else (lambda t: t)
        x = sel(res["te_actual"][n])
        for v in ("te_stale", "te_dm"):
            y = sel(res[v][n])
            emu[f"{n}_te_vs_emul_{v[3:]}_rel"] = ((x - y).norm() / x.norm().clamp_min(1e-300)).item()
            emu[f"{n}_te_vs_emul_{v[3:]}_cos"] = torch.nn.functional.cosine_similarity(
                x.flatten(), y.flatten(), dim=0).item()
    # Is TE's dq = DM-emulation + stale-delta term?  Regress (dq_TE - dq_te_dm) on -scale*e_i*Kbar_i.
    rows_ok = rows >= row_min
    Dd = (res["te_actual"]["dq"] - res["te_dm"]["dq"])[:, hsel][rows_ok]
    Pk = (pred_c[..., None] * kbar[:, hsel])[rows_ok]
    emu["dq_TEminusDM_on_stalepred_slope"] = ((Dd * Pk).sum() / (Pk * Pk).sum()).item()
    emu["dq_TEminusDM_explained_by_stalepred"] = 1 - ((Dd - Pk) ** 2).sum().item() / max((Dd * Dd).sum().item(), 1e-300)
    emu["stale_term_norm_rel_to_dq_ref"] = (Pk.norm() / ref["dq"][:, hsel][rows_ok].norm()).item()
    summary = {
        "file": os.path.basename(path), "layer": m["layer"], "step": m["step"], "heads": len(heads),
        "softmax_scale": sm, "row_min": row_min,
        "te_scales": te_s, "cur_scales": cur_s,
        "te_recorded_amax": {"qkv_fwd": af[idx["META_QKV"]].item(), "o": af[idx["META_O"]].item(),
                             "s": af[idx["META_S"]].item(), "do": ab[idx["META_DO"]].item(),
                             "dp": te_amax_dp, "dqkv": ab[idx["META_DQKV"]].item()},
        "ref_amax": amax, "te_dP_quantiser_sees": dp_is,
        "o_recon_err": recon_o_err,
        "o_emul": {k: v for k, v in res.items() if k.endswith("_o_vs_dump") or k.endswith("_o_vs_ref")},
        "te_vs_emulation": emu, "metrics": metrics, "seconds": time.time() - t0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, os.path.basename(path).replace(".pt", ".json")), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{path}: done in {summary['seconds']:.0f}s", flush=True)
    return summary


def table(summaries):
    lines = []
    hdr = ("| dump | variant | rowres pre: median / p99 / mean signed | rowres post: median / p99 | "
           "dq rel / cos | dk rel | dv rel | dq-err on Kbar | expl. by stale pred | corr(c, pred) |")
    lines += [hdr, "|" + "---|" * 10]
    for s in summaries:
        for v in ("te_actual", "te_stale", "te_dm", "te_stale_dSfp32", "te_dm_dSfp32",
                  "cur_stale", "cur_dm", "blk_stale", "blk_dm"):
            x = s["metrics"][v]
            pre = x.get("rowres_pre")
            post = x.get("rowres_post")
            if "dk_relerr" not in x:
                x = dict(x, dk_relerr=float("nan"), dv_relerr=float("nan"))
            pre_s = f"{pre['median_abs']:.2e} / {pre['p99_abs']:.2e} / {pre['mean_signed']:+.1e}" if pre else "n/a (no dS)"
            post_s = f"{post['median_abs']:.2e} / {post['p99_abs']:.2e}" if post else "n/a"
            lines.append(
                f"| L{s['layer']} s{s['step']} | {v} | {pre_s} | {post_s} | {x['dq_relerr']:.3f} / {x['dq_cos']:.4f} | "
                f"{x['dk_relerr']:.3f} | {x['dv_relerr']:.3f} | {x['dq_err_energy_on_Kbar']:.3f} | "
                f"{x['dq_err_explained_by_stale_pred']:+.3f} | {x['c_vs_pred_corr']:+.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--heads", default=None, help="comma list of q heads (default all 64)")
    ap.add_argument("--row-min", type=int, default=128, help="ignore the first rows (few keys) in row stats")
    a = ap.parse_args()
    heads = [int(x) for x in a.heads.split(",")] if a.heads else None
    files = sorted({f for p in a.dumps for f in glob.glob(p)})
    sums = []
    for f in files:
        try:
            s = run(f, a.out, heads, a.row_min)
        except Exception as e:  # keep going over the other dumps
            print(f"{f}: FAILED {e!r}", flush=True)
            s = None
        if s:
            sums.append(s)
            torch.cuda.empty_cache()
    if sums:
        t = table(sums)
        with open(os.path.join(a.out, "summary_table.md"), "a") as fh:
            fh.write(f"\n<!-- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} -->\n" + t + "\n")
        print(t)


if __name__ == "__main__":
    main()
