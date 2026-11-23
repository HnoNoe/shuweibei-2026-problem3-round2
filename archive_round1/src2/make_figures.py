"""生成论文配图（6 张）。用托管 Python 运行（内置中文字体）。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
RES = ROOT / "results2"
FIG = ROOT / "figures2"
FIG.mkdir(exist_ok=True)

C0, C1, C2 = "#8c8c8c", "#4C9BD4", "#E07B39"
POLICY_LABEL = {"P0_基线WSPT": "基线 WSPT", "P1_ATC无防护": "ATC(无防护)",
                "P2_主策略冻结": "主策略(冻结)"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


# ---- 图1: 时长分布 + 前缀可预测性 ----
def fig1():
    df = pd.read_csv(ROOT / "prefix_features.csv")
    tr = df[df.split == "train"]
    pred = pd.read_csv(RES / "predictability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    colors = {"P": C2, "A": C1, "L": "#5AA46A"}
    for fam, sub in tr.groupby("family_id"):
        ax.hist(np.log(sub["duration_s"]), bins=18, alpha=0.55,
                label=f"族 {fam} (n={len(sub)})", color=colors[fam])
    ax.set_xlabel("log(源区间时长 d / s)")
    ax.set_ylabel("频数")
    ax.set_title("(a) 训练集时长分布（对数正态近似）", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    sub = pred[pred.model != "family均值基线"]
    x = np.arange(3)
    w = 0.36
    for i, (name, cc) in enumerate([("岭回归", C1), ("梯度提升树", C2)]):
        g = sub[sub.model == name]
        ax.bar(x + (i - 0.5) * w, g.R2, w, label=name, color=cc)
    base = pred[pred.model == "family均值基线"].R2.iloc[0]
    ax.axhline(base, ls="--", color=C0, lw=1.2)
    ax.text(2.35, base - 0.06, "family 基线", fontsize=8.5, color=C0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["0.2 s", "0.6 s", "1.0 s"])
    ax.set_xlabel("已释放前缀窗口")
    ax.set_ylabel("分组 5 折 CV 的 $R^2$")
    ax.set_title("(b) 前缀特征预测总时长的 $R^2$", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_duration.png", dpi=200)
    plt.close(fig)


# ---- 图2: 各案例 on-time 对比 ----
def fig2():
    val = pd.read_csv(RES / "validation_all.csv")
    nom = val[val.mapping == "a1.0_b0.5"]
    cases = ["S1", "S2", "S3"]
    pols = ["P0_基线WSPT", "P1_ATC无防护", "P2_主策略冻结"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(cases))
    w = 0.26
    for i, p in enumerate(pols):
        means, lo, hi = [], [], []
        for c in cases:
            v = nom[(nom.policy == p) & (nom.case == c
                    )]["weighted_on_time_rate"].values
            means.append(v.mean())
            l, h = boot_ci(v)
            lo.append(v.mean() - l)
            hi.append(h - v.mean())
        ax.bar(x + (i - 1) * w, means, w, yerr=[lo, hi], capsize=3,
               label=POLICY_LABEL[p],
               color=[C0, C1, C2][i], error_kw=dict(lw=1))
        for xi, m in zip(x + (i - 1) * w, means):
            dy = 0.030 + 0.028 * i if i > 0 else 0.030
            ax.text(xi, m + dy, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, ls=":", color="r", lw=1.1)
    ax.text(2.42, 0.51, "0.5 约束线", color="r", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"案例 {c}" for c in cases])
    ax.set_ylabel("平均加权准时率")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_policy.png", dpi=200)
    plt.close(fig)


# ---- 图3: 配对差异 ----
def fig3():
    val = pd.read_csv(RES / "validation_all.csv")
    nom = val[val.mapping == "a1.0_b0.5"]
    p2 = nom[nom.policy == "P2_主策略冻结"][["case", "seed",
            "weighted_on_time_rate"]].rename(
            columns={"weighted_on_time_rate": "y2"})
    p0 = nom[nom.policy == "P0_基线WSPT"][["case", "seed",
            "weighted_on_time_rate"]].rename(
            columns={"weighted_on_time_rate": "y0"})
    mg = p2.merge(p0, on=["case", "seed"])
    mg["d"] = mg.y2 - mg.y0
    cases = ["S1", "S2", "S3"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = [mg[mg.case == c]["d"].values for c in cases] + [mg["d"].values]
    bp = ax.boxplot(data, labels=[f"{c}\n(n=30)" for c in cases] + ["全部\n(n=90)"],
                    showmeans=True, meanprops=dict(marker="D", mfc=C2, mec=C2),
                    medianprops=dict(color=C1), widths=0.55)
    for i, v in enumerate(data):
        m = v.mean()
        l, h = boot_ci(v)
        ax.errorbar(i + 1, m, yerr=[[m - l], [h - m]], fmt="none", ecolor=C2,
                    capsize=4, lw=1.4)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("配对差异：主策略 − 基线（加权准时率）")
    ax.set_title("同一案例、同一随机种子的 30 次配对 trial", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pairdiff.png", dpi=200)
    plt.close(fig)


# ---- 图4: 5 映射敏感性 ----
def fig4():
    val = pd.read_csv(RES / "validation_all.csv")
    order = ["a0.75_b0.5", "a1.0_b0.0", "a1.0_b0.5", "a1.0_b1.0", "a1.5_b0.5"]
    labels = ["α=0.75\nβ=0.5", "α=1.0\nβ=0.0", "α=1.0\nβ=0.5\n(名义)",
              "α=1.0\nβ=1.0", "α=1.5\nβ=0.5"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for p, cc, mk in [("P0_基线WSPT", C0, "o"),
                      ("P2_主策略冻结", C2, "s")]:
        ys = []
        for mp in order:
            v = val[(val.policy == p) & (val.mapping == mp
                    )]["weighted_on_time_rate"].mean()
            ys.append(v)
        ax.plot(range(5), ys, mk + "-", color=cc, label=POLICY_LABEL[p],
                lw=1.6, ms=6)
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.3f}", (i, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=cc)
    ax.axhline(0.5, ls=":", color="r", lw=1.1)
    ax.text(4.05, 0.515, "0.5 约束线", color="r", fontsize=8.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("平均加权准时率")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_mapping.png", dpi=200)
    plt.close(fig)


# ---- 图5: b 的多目标权衡 ----
def fig5():
    tab = pd.read_csv(RES / "tradeoff_b.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    xs = np.arange(len(tab))
    ax.bar(xs - 0.18, tab.overrun, 0.36, label="observed_overrun", color=C2)
    ax.bar(xs + 0.18, tab.idle, 0.36, label="reserved_idle", color=C1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.20(最小)", "中位数", "Q75", "Q90"], fontsize=9)
    ax.set_xlabel("预留时长 b 的取法")
    ax.set_ylabel("平均秒数 / trial")
    ax.set_title("(a) 超时占用 vs 预留空闲", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(xs, tab.ontime, "s-", color=C2, lw=1.6, ms=6, label="加权准时率")
    for i, v in enumerate(tab.ontime):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=C2)
    ax2 = ax.twinx()
    ax2.plot(xs, tab.cost, "o--", color=C1, lw=1.4, ms=5, label="平均成本")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.20(最小)", "中位数", "Q75", "Q90"], fontsize=9)
    ax.set_xlabel("预留时长 b 的取法")
    ax.set_ylabel("加权准时率")
    ax2.set_ylabel("平均成本 / trial")
    ax.set_ylim(0.4, 1.0)
    ax2.set_ylim(20, 32)
    ax.set_title("(b) 准时率与成本的此消彼长", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_tradeoff.png", dpi=200)
    plt.close(fig)


# ---- 图6: 示例 trial 时间线 ----
def fig6():
    log = pd.read_csv(RES / "sample_trial_log.csv")
    tasks = pd.read_csv(ROOT / "problem" / "tasks.csv")
    due = dict(zip(tasks.job_id, tasks.due_s))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ylabels = []
    for i, r in log.iterrows():
        if r.action != "execute":
            continue
        jid = r.job
        k = len(ylabels)
        ylabels.append(jid.replace("S2_", ""))
        ax.barh(k, r.service_start - r.time, left=r.time, color="#d9d9d9",
                height=0.6)
        ontime = r.completion <= due.get(jid, 1e9) and not bool(r.censored)
        ax.barh(k, r.completion - r.service_start, left=r.service_start,
                color=(C1 if ontime else "#C0504D"), height=0.6)
        if r.censored:
            ax.barh(k, 100 - r.completion, left=r.completion,
                    color="#C0504D", alpha=0.35, height=0.6,
                    hatch="//", edgecolor="white")
    stop_t = log[log.action == "stop"].time.iloc[0]
    ax.axvline(stop_t, color="k", ls="--", lw=1.2)
    ax.text(stop_t + 1, len(ylabels) - 0.4, f"停止 t={stop_t:.1f}",
            fontsize=9)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color="#d9d9d9", label="移动+换型"),
                       mp.Patch(color=C1, label="服务(准时)"),
                       mp.Patch(color="#C0504D", label="服务(延误/截断)")],
              frameon=False, fontsize=9, loc="lower right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("场景时间 t / s")
    ax.set_ylabel("作业（案例 S2，种子 2026111705）")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_timeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))
