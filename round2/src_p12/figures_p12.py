import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
OUT = ROOT / "round2" / "results_p12"
FIG = ROOT / "round2" / "figures3"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
    "axes.edgecolor": "0.25", "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
C = {"hold": "#A9563B", "cv": "#7A8B5C", "ridge": "#386795", "grey": "0.55",
     "blue": "#386795", "green": "#4E9A6F", "amber": "#C9962E"}


def fig_reliability():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    df = pd.read_pickle(OUT / "frames.pkl").dropna(subset=["v", "spread", "dist"])
    feats = ["v", "spread", "dist", "x", "y", "z"]
    within = []
    for env, g in df.groupby("environment_id"):
        if g.bad.sum() < 20 or len(g) < 400:
            continue
        aucs = []
        pos = np.where(g.bad.to_numpy() == 1)[0]
        for s in range(10):
            m = LogisticRegression(class_weight="balanced", max_iter=1000)
            idx = np.concatenate([rng_choice(g, pos, s)])
            m.fit(g[feats].iloc[idx], g.bad.iloc[idx])
            rest = np.setdiff1d(np.arange(len(g)), idx)
            aucs.append(roc_auc_score(g.bad.iloc[rest], m.predict_proba(g[feats].iloc[rest])[:, 1]))
        within.append(np.mean(aucs))
    rm = pd.read_csv(OUT / "reliability_metrics.csv")
    across = rm[(rm.model == "logistic_l1") & (rm.fold.isin([0, 1, 2, 3, 4]))]["auroc"].to_numpy()
    te = rm[(rm.model == "logistic_l1") & (rm.fold == "test")]["auroc"].iloc[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bp = ax.boxplot([within, across], positions=[1, 2], widths=0.5, patch_artist=True,
                    medianprops=dict(color="0.1", lw=1.2),
                    showcaps=False, showfliers=False)
    for p, c in zip(bp["boxes"], [C["green"], C["hold"]]):
        p.set_facecolor(c)
        p.set_alpha(0.55)
        p.set_edgecolor("0.2")
        p.set_linewidth(0.8)
    rngn = np.random.default_rng(3)
    for x, vals in [(1, within), (2, across)]:
        ax.scatter(x + rngn.uniform(-0.09, 0.09, len(vals)), vals, s=11, color="0.25", zorder=3, lw=0)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=(0, (4, 3)))
    ax.axhline(te, color=C["ridge"], lw=0.9, ls=(0, (1, 2)))
    ax.text(2.35, te + 0.012, "held-out test\n0.61", fontsize=7.5, color=C["ridge"], ha="right")
    ax.text(0.62, 0.505, "chance", fontsize=7.5, color="0.45")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["within\nenvironment", "across\nenvironments"], fontsize=8.5)
    ax.set_ylabel("Frame-level reliability AUROC", fontsize=8.5)
    ax.set_ylim(0.15, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=10)
    fig.savefig(FIG / "fig_p1_reliability.png")
    plt.close(fig)


def rng_choice(g, pos, s):
    rng = np.random.default_rng(100 + s)
    neg = np.where(g.bad.to_numpy() == 0)[0]
    n_neg = min(len(neg), 10 * len(pos))
    return np.concatenate([rng.choice(pos, len(pos), replace=False),
                           rng.choice(neg, n_neg, replace=False)])


def fig_growth():
    cur = pd.read_csv(OUT / "error_by_delta.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    style = {"hold": (C["hold"], "o", "HOLD (last state)"),
             "cv": (C["cv"], "s", "Constant velocity"),
             "ridge": (C["ridge"], "^", "Family ridge (ours)")}
    for meth, (col, mk, lab) in style.items():
        d = cur[cur.method == meth].sort_values("delta")
        ax.fill_between(d["delta"], d["ci_lo"], d["ci_hi"], color=col, alpha=0.16, lw=0)
        ax.plot(d["delta"], d["mae"], color=col, lw=1.4, marker=mk, ms=4, label=lab,
                markerfacecolor="white", markeredgewidth=1.1)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3, 5])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "3", "5"])
    ax.set_xlabel("Missing-observation window $\\Delta$ (s)", fontsize=8.5)
    ax.set_ylabel("Mean 3D position error (m)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=10)
    fig.savefig(FIG / "fig_p2_growth.png")
    plt.close(fig)


def fig_tradeoff():
    s = pd.read_csv(OUT / "reobservation_sims.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    def pt(name, col, mk, lab, filled=False):
        e = s[f"{name}_mean_err"].mean()
        q = s[f"{name}_req_per_min"].mean()
        ax.scatter(q, e, s=42, color=col, marker=mk, zorder=3,
                   facecolor=col if filled else "white", edgecolor=col, linewidth=1.2,
                   label=lab)

    fx = [0.5, 1.0, 2.0, 4.0]
    es = [s[f"fixed_{k}_mean_err"].mean() for k in fx]
    qs = [s[f"fixed_{k}_req_per_min"].mean() for k in fx]
    ax.plot(qs, es, color=C["grey"], lw=1.1, marker="o", ms=4,
            markerfacecolor="white", markeredgewidth=1.0, label="Fixed interval", zorder=2)
    pt("thresh_0.05", C["ridge"], "^", "Uncertainty threshold", True)
    pt("thresh_0.4", C["ridge"], "^", None, True)
    pt("oracle_0.1", C["amber"], "D", "Oracle bound", False)
    ax.scatter([0.5], [s["no_request_mean"].mean()], s=42, color="0.25", marker="x",
               linewidth=1.4, label="No re-observation", zorder=3)
    ax.annotate("0.05 m", (s["thresh_0.05_req_per_min"].mean(), s["thresh_0.05_mean_err"].mean()),
                textcoords="offset points", xytext=(7, 5), fontsize=7.5, color=C["ridge"])
    ax.annotate("0.40 m", (s["thresh_0.4_req_per_min"].mean(), s["thresh_0.4_mean_err"].mean()),
                textcoords="offset points", xytext=(7, -2), fontsize=7.5, color=C["ridge"])
    ax.set_xscale("log")
    ax.set_xlabel("Re-observation requests per minute", fontsize=8.5)
    ax.set_ylabel("Time-mean 3D error (m)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=10)
    fig.savefig(FIG / "fig_p2_tradeoff.png")
    plt.close(fig)


def fig_audit():
    ar = pd.read_csv(OUT / "audit_records.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    ax = axes[0]
    d = ar.groupby("dataset").agg(bad=("bad_ratio", "mean"), dur=("duration_s", "median"),
                                  n=("episode_uuid", "count"))
    x = np.arange(len(d))
    ax.bar(x - 0.18, d["bad"] * 100, width=0.36, color=C["hold"], alpha=0.85, label="Bad-frame ratio (%)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, d["dur"], width=0.36, color=C["blue"], alpha=0.8, label="Median duration (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, fontsize=8.5)
    ax.set_ylabel("Bad frames (%)", fontsize=8.5)
    ax2.set_ylabel("Median record duration (s)", fontsize=8.5, color=C["blue"])
    ax2.tick_params(axis="y", colors=C["blue"])
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(d["bad"] * 100) * 1.35)

    ax = axes[1]
    iv = pd.read_csv(OUT / "interval_features.csv")
    order = sorted(iv["family_id"].unique())
    data = [iv[iv.family_id == f]["speed_mean"].dropna() for f in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="0.1", lw=1.2),
                    showcaps=False)
    cols = {"A": C["blue"], "L": C["green"], "P": C["amber"]}
    for p, f in zip(bp["boxes"], order):
        p.set_facecolor(cols.get(f, "0.5"))
        p.set_alpha(0.6)
        p.set_edgecolor("0.2")
        p.set_linewidth(0.8)
    rngn = np.random.default_rng(5)
    for k, f in enumerate(order):
        v = iv[iv.family_id == f]["speed_mean"].dropna()
        ax.scatter(k + 1 + rngn.uniform(-0.14, 0.14, len(v)), v, s=7, color="0.3", lw=0, zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"Family {f}" for f in order], fontsize=8.5)
    ax.set_ylabel("Mean hand speed (m/s)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG / "fig_p1_audit.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_audit()
    fig_reliability()
    fig_growth()
    fig_tradeoff()
    print("figures done")
