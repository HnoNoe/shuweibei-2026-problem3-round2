import json

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS

M = "weighted_on_time_rate"

VAL_CONTRASTS = [
    ("B2_WSPT_MEAN", "B1_EDD", "EDD to WSPT scoring"),
    ("B2b_WSPT_MEDIAN", "B2_WSPT_MEAN", "mean to median anchor"),
    ("B3_WSPT_Q90", "B2b_WSPT_MEDIAN", "median to Q90 robust anchor"),
    ("C1_atc", "B2_WSPT_MEAN", "WSPT to ATC slack rule"),
    ("C2_guard", "C1_atc", "hard censoring guard (rejected)"),
    ("C3_budget", "C1_atc", "budget-aware trade-off"),
    ("C4_roll", "C1_atc", "rollout lookahead"),
    ("C5_online", "B3_WSPT_Q90", "full online two-stage engine"),
    ("C6_full", "C5_online", "offline sequence search (rejected)"),
    ("RM_buytop", "C5_online", "adversarial buy-top review arm"),
]

TEST_CONTRASTS = [
    ("C5_online", "B3_WSPT_Q90", "full engine vs strongest naive (test)"),
    ("RM_buytop", "C5_online", "adversarial arm vs zero-review (test)"),
]


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip("\ufeff") for c in df.columns]
    return df


def paired(df, a, b):
    x = df[df.arm == a].set_index(["case", "seed", "alpha", "beta"])[M]
    y = df[df.arm == b].set_index(["case", "seed", "alpha", "beta"])[M]
    common = x.index.intersection(y.index)
    return (x.loc[common] - y.loc[common]).dropna()


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def audit(df, contrasts):
    rows = []
    for a, b, name in contrasts:
        d = paired(df, a, b)
        n = len(d)
        mean = float(d.mean())
        hw = float(stats.t.interval(0.95, n - 1, loc=mean,
                                    scale=d.std(ddof=1) / np.sqrt(n))[1] - mean)
        t_stat, p_t = stats.ttest_rel(
            df[df.arm == a].set_index(["case", "seed", "alpha", "beta"])[M]
            .loc[d.index],
            df[df.arm == b].set_index(["case", "seed", "alpha", "beta"])[M]
            .loc[d.index])
        try:
            p_w = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            p_w = 1.0
        dz = float(mean / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
        rows.append(dict(contrast=name, arm=a, base=b, n=n, mean=mean,
                         ci95_halfwidth=hw, t_stat=float(t_stat), p_t=float(p_t),
                         p_wilcoxon=p_w, cohens_dz=dz))
    adj = holm([r["p_wilcoxon"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return rows


def main():
    out = {
        "metric": M,
        "validation_family": audit(load(RESULTS / "q7_ablation.csv"),
                                   VAL_CONTRASTS),
        "test_family": audit(load(RESULTS / "q8_test.csv"), TEST_CONTRASTS),
    }
    with open(RESULTS / "q9_stats_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for fam in ("validation_family", "test_family"):
        print(f"== {fam}")
        for r in out[fam]:
            print(f"{r['contrast']:45s} d={r['mean']:+.4f} "
                  f"p_w={r['p_wilcoxon']:.2e} holm={r['p_holm']:.2e} "
                  f"dz={r['cohens_dz']:+.2f}")


if __name__ == "__main__":
    main()
