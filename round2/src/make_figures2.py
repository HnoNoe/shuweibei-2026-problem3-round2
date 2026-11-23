
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from common import FIGURES, RESULTS, ROOT, load_source_index, eligible_pool
from duration_model3 import DurationModel

FIGDIR = ROOT / "round2/figures"
captions = {}

def fig1_family_durations():
    pool = eligible_pool(load_source_index())
    model = DurationModel.from_gofit()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    for ax, fam in zip(axes, ("A", "L", "P")):
        ds = [r["d"] for k, rows in pool.items() if k[1] == fam
              for r in rows]
        ax.hist(ds, bins=25, density=True, alpha=0.6, color="#4C72B0",
                label="eligible 样本")
        mu, sig = model.params[fam]
        xs = np.linspace(0.01, max(ds) * 1.05, 400)
        pdf = np.exp(-(np.log(xs) - mu) ** 2 / (2 * sig ** 2)) \
            / (xs * sig * math.sqrt(2 * math.pi))
        ax.plot(xs, pdf, "r-", lw=2, label="对数正态拟合")
        ax.set_title(f"{fam} 族 (n={len(ds)}, σ={sig:.2f})")
        ax.set_xlabel("作业时长 d (s)")
    axes[0].set_ylabel("密度")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_family_durations.png", dpi=200)
    plt.close(fig)
    captions["fig1_family_durations"] = (
        "图 三族作业时长分布与对数正态拟合（eligible 池，train/validation 合并展示）")

def fig2_censoring():
    p = RESULTS / "censor_study.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df.columns = [c.strip("\ufeff") for c in df.columns]

    nrows = len(df)
    fig, axes = plt.subplots(1, nrows, figsize=(6.4 * nrows, 3.6), sharey=True)
    if nrows == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, df.iterrows()):
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, [r["naive_mu_bias"], r["naive_sigma_bias"]], w,
               label="朴素丢弃", color="#C44E52", alpha=0.85)
        ax.bar(x + w / 2, [r["em_mu_bias"], r["em_sigma_bias"]], w,
               label="EM 含删失", color="#55A868", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["μ 偏差", "σ 偏差"])
        ax.set_title(f"子样本 n={int(r['n'])}")
        ax.set_ylabel("估计值 − 真值")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_censoring.png", dpi=200)
    plt.close(fig)
    captions["fig2_censoring"] = (
        "图 截断样本处理对照：朴素丢弃 vs EM 含删失的 μ/σ 偏差（已知真值模拟）")

def fig3_prefix_r2():
    df = pd.read_csv(RESULTS / "q2_predictability.csv")
    df.columns = [c.strip("\ufeff") for c in df.columns]
    piv = df.pivot(index="model", columns="window", values="R2")
    order = ["family均值基线", "岭回归", "梯度提升树"]
    wins = [c.strip("\ufeff") for c in df.columns if c.startswith("p")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(order))
    w = 0.25
    for i, win in enumerate(sorted(piv.columns)):
        ax.bar(x + (i - 1) * w, piv.loc[order, win], w, label=win)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylabel("R²（负 = 不如猜族均值）")
    ax.legend(title="前缀窗口")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_prefix_r2.png", dpi=200)
    plt.close(fig)
    captions["fig3_prefix_r2"] = (
        "图 前缀运动特征预测总时长的 R²：全窗口全模型为负，预测力证伪")

def fig4_ablation():
    s = json.load(open(RESULTS / "q7_ablation_summary.json", encoding="utf-8"))
    pairs = [p for p in s["pairs"] if not p["label"].startswith("主对比")
             and "RM" not in p["label"]]
    labels = [p["label"] for p in pairs]
    vals = [p["mean_diff"] for p in pairs]
    cis = [p["ci95_halfwidth"] for p in pairs]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in vals]
    ax.barh(range(len(vals)), vals, xerr=cis, color=colors, alpha=0.85,
            error_kw=dict(lw=1.2))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("配对差值（臂A − 臂B，加权准时率）±95% CI")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_ablation.png", dpi=200)
    plt.close(fig)
    captions["fig4_ablation"] = (
        "图 消融归因链：相邻配置配对差值与 95% CI（450 局/臂，CRN 配对）")

def fig5_btradeoff():
    p = RESULTS / "q8_btradeoff_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(df["b"], df["cost"], "o-", color="#4C72B0")
    axes[0].set_title("总费用")
    axes[1].plot(df["b"], df["overrun"], "o-", color="#DD8452")
    axes[1].set_title("观测 overrun (s)")
    axes[2].plot(df["b"], df["ot"], "o-", color="#55A868")
    axes[2].set_title("加权准时率")
    for ax in axes:
        ax.set_xlabel("预留 b (s)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_btradeoff.png", dpi=200)
    plt.close(fig)
    captions["fig5_btradeoff"] = (
        "图 预留 b 的权衡扫描：准时率与 overrun 不变、费用单调上升（b=0.2 弱占优的量化验证）")

def fig6_paired_hist():
    df = pd.read_csv(RESULTS / "q7_ablation.csv")
    df.columns = [c.strip("\ufeff") for c in df.columns]
    piv = df.pivot_table(index=["alpha", "beta", "case", "seed"], columns="arm",
                         values="weighted_on_time_rate")
    d = (piv["C6_full"] - piv["B3_WSPT_Q90"]).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(d, bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(d.mean(), color="r", ls="--", lw=1.5,
               label=f"均值 {d.mean():+.4f}")
    ax.set_xlabel("逐 trial 配对差值（C6 − B3）")
    ax.set_ylabel("频次")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_paired_hist.png", dpi=200)
    plt.close(fig)
    captions["fig6_paired_hist"] = (
        "图 主对比逐 trial 配对差分分布：正值占优且均值显著离 0")

if __name__ == "__main__":
    fig1_family_durations()
    fig2_censoring()
    fig3_prefix_r2()
    fig4_ablation()
    fig5_btradeoff()
    fig6_paired_hist()
    with open(FIGDIR / "fig_captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(json.dumps(captions, ensure_ascii=False, indent=2))
