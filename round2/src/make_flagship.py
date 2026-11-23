import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

from common import RESULTS, ROOT

FIGDIR = ROOT / "round2/figures3"

C_REF = "#386795"
C_BASE = "#B0413E"
C_GAIN = "#4E9A6F"
C_GREY = "#8C8C8C"
C_GRID = "0.85"


def box(ax, x, y, w, h, fc, ec, lw=0.9, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def arrow(ax, x0, x1, y, color="0.55"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                shrinkA=0, shrinkB=0), zorder=1)


def mini_bars_r2(ax, x0, x1, y0, y1, labels, values, colors):
    vals = np.asarray(values, dtype=float)
    lo, hi = vals.min(), max(vals.max(), 0.0)
    span = (hi - lo) or 1.0
    zy = y0 + (0.0 - lo) / span * (y1 - y0)
    ax.plot([x0, x1], [zy, zy], color="black", lw=0.6, zorder=3)
    n = len(vals)
    bw = (x1 - x0) / (n * 1.6)
    for i, v in enumerate(vals):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        top = y0 + (v - lo) / span * (y1 - y0)
        ax.bar(cx, top - zy, bw, bottom=zy, color=colors[i], zorder=3)
        ax.text(cx, min(top, zy) - 1.0, f"{v:.2f}", ha="center",
                va="top", fontsize=5.6, zorder=4)


def mini_bars_bias(ax, x0, x1, y0, y1, labels, values, colors):
    vals = np.asarray(values, dtype=float)
    lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.1)
    span = (hi - lo) or 1.0
    zy = y0 + (0.0 - lo) / span * (y1 - y0)
    ax.plot([x0, x1], [zy, zy], color="black", lw=0.6, zorder=3)
    n = len(vals)
    bw = (x1 - x0) / (n * 2.6)
    for i, v in enumerate(vals):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        top = y0 + (v - lo) / span * (y1 - y0)
        ax.bar(cx, top - zy, bw, bottom=zy, color=colors[i], zorder=3)
        off = 1.0 if v >= 0 else -1.0
        ax.text(cx, top + off, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.0, zorder=4)


def main():
    q2 = pd.read_csv(RESULTS / "q2_predictability.csv")
    q2.columns = [c.strip("\ufeff") for c in q2.columns]
    ridge = q2[q2.model == "岭回归"].sort_values("window")
    fam_mean = float(q2[q2.model == "family均值基线"]["R2"].mean())
    cen = pd.read_csv(RESULTS / "censor_study.csv")
    cen.columns = [c.strip("\ufeff") for c in cen.columns]
    cen = cen.sort_values("n").iloc[-1]
    sv = json.load(open(RESULTS / "q7_ablation_summary.json", encoding="utf-8"))
    c5 = [p for p in sv["pairs"] if p["label"] == "C5-B3 在线全家vs最强朴素"][0]
    saa = json.load(open(RESULTS / "q7_saa_gap.json", encoding="utf-8"))
    st = json.load(open(RESULTS / "q8_test_summary.json", encoding="utf-8"))
    t_mean = st["C5_online_minus_B3_WSPT_Q90"]["mean"]
    t_lo, t_hi = st["C5_online_minus_B3_WSPT_Q90"]["ci95"]

    W, H = 160.0, 100.0
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(20, 96.5, "baseline pipeline", ha="center", fontsize=8,
            fontweight="bold", color=C_BASE)
    ax.text(80, 96.5, "the evidence (CRN-paired)", ha="center", fontsize=8,
            fontweight="bold", color="0.25")
    ax.text(140, 96.5, "refined pipeline", ha="center", fontsize=8,
            fontweight="bold", color=C_REF)

    rows = [
        dict(y=76.0, tag="1  information",
             left="ridge prefix-duration model +\nselective-review experiment\n(never falsified)",
             right="zero-review: prefix R² < 0 for\nevery window and model falsifies\nbuying information"),
        dict(y=50.0, tag="2  truncation",
             left="naïvely drop censored jobs",
             right="EM censoring-aware update:\ntruncation is information,\nnot garbage"),
        dict(y=24.0, tag="3  decision",
             left="deterministic ATC score,\nmean anchor, one mapping,\n90 trials",
             right="two-stage stochastic engine:\nmedian→Q90 anchor, soft censoring\npenalty, budget trade-off,\nrollout lookahead"),
    ]

    lw_, lh = 37.0, 15.5
    lx, rx = 1.5, 121.5
    cx0, cx1 = 45.0, 115.0
    ch = 17.5

    for r in rows:
        yc = r["y"]
        ax.text(0.5, yc + lh / 2 + 3.2, r["tag"], fontsize=6.8,
                fontweight="bold", color="0.3", va="center")
        box(ax, lx, yc - lh / 2, lw_, lh, "#FBEFEF", C_BASE, lw=1.0)
        ax.text(lx + lw_ / 2, yc, r["left"], ha="center", va="center",
                fontsize=6.6, color="0.15", zorder=3)
        box(ax, rx, yc - lh / 2, lw_, lh, "#EDF2F8", C_REF, lw=1.0)
        ax.text(rx + lw_ / 2, yc, r["right"], ha="center", va="center",
                fontsize=6.6, color="0.15", zorder=3)
        box(ax, cx0, yc - ch / 2, cx1 - cx0, ch, "white", C_GRID, lw=0.9)
        arrow(ax, lx + lw_ + 0.4, cx0 - 0.4, yc)
        arrow(ax, cx1 + 0.4, rx - 0.4, yc, color=C_REF)

    yc = rows[0]["y"]
    mini_bars_r2(ax, cx0 + 5, cx1 - 5, yc - 3.2, yc + 7.0,
                 None,
                 [fam_mean] + ridge["R2"].tolist(),
                 [C_GREY, C_BASE, C_BASE, C_BASE])
    ax.text((cx0 + cx1) / 2, yc - ch / 2 + 1.2,
            "out-of-sample R²: family mean vs ridge (0.2/0.6/1.0 s prefixes)",
            ha="center", fontsize=6.0, color="0.25")

    yc = rows[1]["y"]
    mini_bars_bias(ax, cx0 + 8, cx1 - 8, yc - 3.2, yc + 7.0,
                   None,
                   [float(cen["naive_mu_bias"]), float(cen["em_mu_bias"])],
                   [C_BASE, C_REF])
    ax.text((cx0 + cx1) / 2, yc - ch / 2 + 1.2,
            f"log-mean bias, known-truth simulation (n = {int(cen['n'])})",
            ha="center", fontsize=6.0, color="0.25")

    yc = rows[2]["y"]
    gx0, gx1 = cx0 + 14, cx1 - 14
    gz = yc - 2.5
    gtop = yc + 6.5
    ax.plot([gx0, gx1], [gz, gz], color="black", lw=0.6, zorder=3)
    m, hw = c5["mean_diff"], c5["ci95_halfwidth"]
    span = (gtop - gz) / 0.09
    btop = gz + m * span
    ax.bar((gx0 + gx1) / 2, btop - gz, 8.0, bottom=gz, color=C_GAIN, zorder=3)
    ax.errorbar([(gx0 + gx1) / 2], [btop], yerr=[hw * span], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2.4, zorder=4)
    ax.text((gx0 + gx1) / 2 + 6.0, btop, f"{m:+.4f}\n± {hw:.4f}",
            fontsize=6.4, va="center")
    ax.text((cx0 + cx1) / 2, yc - ch / 2 + 1.2,
            "paired on-time gain over strongest naive rule, 450 validation trials",
            ha="center", fontsize=6.0, color="0.25")

    by0, bh = 1.5, 11.5
    box(ax, 1.5, by0, 157.0, bh, "#F4F4F4", "0.75", lw=0.8, r=1.0)
    ax.text(4.5, by0 + bh / 2, "certified:", fontsize=7.5, fontweight="bold",
            va="center", color="0.15")
    s1 = saa["S1"]
    bx0 = 25.0
    ax.text(bx0, by0 + bh - 2.4, "SAA optimality gap (S1, exact)", fontsize=6.0,
            color="0.35", va="center")
    yb = by0 + 3.4
    lo_x = bx0 + (s1["lb"] - 0.72) / 0.27 * 20
    ub_x = bx0 + (s1["ub"] - 0.72) / 0.27 * 20
    ax.plot([lo_x, ub_x], [yb, yb], color=C_GREY, lw=4.5, alpha=0.5,
            solid_capstyle="round")
    ax.plot([lo_x], [yb], "o", color=C_REF, ms=4)
    ax.plot([ub_x], [yb], "s", color=C_GREY, ms=3.5)
    ax.text(bx0 + 8.5, yb, f"{s1['gap_rel']*100:.1f}%", fontsize=7,
            fontweight="bold", color=C_REF, va="center")
    ax.text(56, by0 + bh / 2, f"validation  {c5['mean_diff']:+.4f} ± "
            f"{c5['ci95_halfwidth']:.4f}", fontsize=7, va="center",
            color="0.15")
    ax.text(94, by0 + bh / 2, f"test  {t_mean:+.4f} ± "
            f"{(t_hi - t_lo) / 2:.4f}", fontsize=7, va="center", color="0.15")
    ax.text(130, by0 + bh / 2, "adversarial buy-top arm: −0.0504",
            fontsize=6.5, va="center", color=C_BASE)
    ax.text(130, by0 + 2.4, "reviews bought: 0", fontsize=7, va="center",
            fontweight="bold", color=C_GAIN)

    fig.savefig(FIGDIR / "figZ_flagship.png")
    plt.close(fig)
    print("figZ_flagship.png")


if __name__ == "__main__":
    main()
