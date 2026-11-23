import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
OUT = ROOT / "round2" / "results_p12"


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    o = np.argsort(p)
    r = p[o] * (m - np.arange(m))
    adj = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


def stats_reliability():
    cal = pd.read_csv(OUT / "reliability_calib.csv")
    rows = []
    for fold, g in cal.groupby("fold"):
        if not str(fold).isdigit():
            continue
        for a, b in [("gbm", "logistic_l1")]:
            m = g.pivot_table(index=g.index % len(g), columns="model", values="p")
    tr = cal[cal.fold.astype(str).str.isdigit()]
    per_env = tr.groupby(["fold", "model"]).apply(
        lambda g: pd.Series({
            "brier": ((g.y - g.p) ** 2).mean(),
            "pos": g.y.mean(),
        })).reset_index()
    piv = per_env.pivot(index="fold", columns="model", values="brier")
    d = (piv["logistic_l1"] - piv["gbm"]).dropna()
    w = st.wilcoxon(d)
    res = [{"comparison": "gbm vs logistic (Brier, CV folds)",
            "n_pairs": int(len(d)), "mean_diff": float(d.mean()),
            "wilcoxon_p": float(w.pvalue),
            "cohen_dz": float(d.mean() / d.std(ddof=1))}]
    te = cal[cal.fold == "test"]
    met = pd.read_csv(OUT / "reliability_metrics.csv")
    out = pd.DataFrame(res)
    out["p_holm"] = holm(out["wilcoxon_p"])
    out.to_csv(OUT / "stats_p1_reliability.csv", index=False)
    return out


def stats_strategies():
    s = pd.read_csv(OUT / "reobservation_sims.csv")
    comp = [("thresh_0.1", "fixed_1.0"), ("thresh_0.05", "fixed_1.0"),
            ("thresh_0.4", "fixed_4.0"), ("thresh_0.1", "no_request")]
    rows = []
    for a, b in comp:
        if b == "no_request":
            d_req = s[f"{a}_req_per_min"]
            d_err = s[f"{a}_mean_err"] - s["no_request_mean"]
        else:
            d_req = s[f"{a}_req_per_min"] - s[f"{b}_req_per_min"]
            d_err = s[f"{a}_mean_err"] - s[f"{b}_mean_err"]
        wq = st.wilcoxon(d_req)
        we = st.wilcoxon(d_err)
        rows.append({"comparison": f"{a} vs {b}",
                     "mean_diff_req_per_min": float(d_req.mean()),
                     "wilcoxon_p_req": float(wq.pvalue),
                     "mean_diff_err": float(d_err.mean()),
                     "wilcoxon_p_err": float(we.pvalue),
                     "n_pairs": int(len(d_req))})
    out = pd.DataFrame(rows)
    out["p_holm_req"] = holm(out["wilcoxon_p_req"])
    out["p_holm_err"] = holm(out["wilcoxon_p_err"])
    out.to_csv(OUT / "stats_p2_strategies.csv", index=False)
    return out


if __name__ == "__main__":
    print(stats_reliability().round(6).to_string())
    print()
    print(stats_strategies().round(6).to_string())
