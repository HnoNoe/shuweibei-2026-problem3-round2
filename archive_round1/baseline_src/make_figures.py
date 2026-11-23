"""生成论文所需图表。"""
import sys
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from config import FIGURES, RESULTS

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

comp = pd.read_csv(RESULTS / "comparison_validation.csv")
test = pd.read_csv(RESULTS / "comparison_test.csv")

# 图1: 两种策略各case的on-time率（名义映射）
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, title in zip(axes, [comp, test], ["验证集 validation", "测试集 test"]):
    nom = data[data.mapping == "a1.0_b0.5"]
    pivot = nom.groupby(["case", "strategy"]).weighted_on_time_rate.mean().unstack()
    pivot.plot(kind="bar", ax=ax, color=["#2E86AB", "#A23B72"], width=0.7)
    ax.axhline(0.5, color="red", ls="--", lw=1.5, label="约束线 0.50")
    ax.set_title(title + "：各案例准时完成率")
    ax.set_ylabel("加权准时完成率"); ax.set_xlabel("案例")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "fig1_ontime_by_case.png", dpi=150)
plt.close()

# 图2: 5种映射下两策略的on-time率
fig, ax = plt.subplots(figsize=(10, 4.5))
g = comp.groupby(["mapping", "strategy"]).weighted_on_time_rate.mean().unstack()
g.plot(kind="bar", ax=ax, color=["#2E86AB", "#A23B72"], width=0.75)
ax.axhline(0.5, color="red", ls="--", lw=1.5, label="约束线 0.50")
ax.set_title("五种时长映射下的平均准时完成率（验证集）")
ax.set_ylabel("加权准时完成率"); ax.set_xlabel("映射 (α, β)")
ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(FIGURES / "fig2_ontime_by_mapping.png", dpi=150)
plt.close()

# 图3: 多指标雷达/柱状对比（名义映射，validation）
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
nom = comp[comp.mapping == "a1.0_b0.5"].groupby("strategy").mean(numeric_only=True)
metrics = [("weighted_completion_rate", "完成率"), ("observed_overrun_s", "超期量(s)"),
           ("reserved_idle_s", "预留空闲(s)")]
for ax, (col, label) in zip(axes, metrics):
    vals = nom[col]
    bars = ax.bar(range(len(vals)), vals.values, color=["#2E86AB", "#A23B72"])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(["准时导向", "风险规避"], fontsize=9)
    ax.set_title(label); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "fig3_tradeoff.png", dpi=150)
plt.close()

# 图4: 时长分布诊断（前缀与时长弱相关）
feat = pd.read_csv("data/prefix_features.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(feat.duration_s.clip(0, 30), bins=50, color="#2E86AB", edgecolor="white")
axes[0].set_title("源区间作业时长分布（截断至30s）")
axes[0].set_xlabel("时长 d (s)"); axes[0].set_ylabel("频次")
inten = feat.apply(lambda r: np.nanmean([
    r.get("p02_vall_std", np.nan), r.get("p02_head_std", np.nan)]), axis=1)
m = np.isfinite(inten) & (feat.duration_s < 30)
axes[1].scatter(inten[m], feat.duration_s[m], alpha=0.3, s=12, color="#A23B72")
corr = np.corrcoef(inten[m], feat.duration_s[m])[0, 1]
axes[1].set_title(f"前缀运动强度 vs 时长 (r={corr:.2f})")
axes[1].set_xlabel("前缀运动强度"); axes[1].set_ylabel("时长 d (s)")
plt.tight_layout()
plt.savefig(FIGURES / "fig4_duration_diagnosis.png", dpi=150)
plt.close()
print("figures done")
