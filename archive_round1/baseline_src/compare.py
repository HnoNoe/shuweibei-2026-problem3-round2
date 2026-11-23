"""多策略对比：主策略(准时导向) vs 风险规避策略 vs 基线。

在 validation 上冻结策略后，用 test split + 30 个冻结种子做最终测试。
题目规定最终测试种子由组委会冻结，这里用公共验证种子做验证报告，
test split 结果作为"模拟最终测试"并明确标注。
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from config import VAL_SEEDS, CASES, SENSITIVITY, RESULTS
from duration_model import DurationEstimator
from scheduler import Scheduler, default_params
from runner import SourceSampler, load_tasks, run_trial

df = pd.read_csv("data/prefix_features.csv")
est = DurationEstimator(rho_shrink=0.5).fit(df[df.split == "train"])
sampler = SourceSampler("data/prefix_features.csv")
tasks = load_tasks()

# 两种策略（在 validation 上确定，此处冻结）
STRATEGIES = {
    "A_准时导向": dict(reserve_mode="min", review_mode="none", order_rule="wspt"),
    "B_风险规避": dict(reserve_mode="median", review_mode="none", order_rule="wspt"),
}

def run(split, seeds=VAL_SEEDS):
    rows = []
    for name, over in STRATEGIES.items():
        p = default_params(); p.update(over)
        sched = Scheduler(est, p)
        for (a, b) in SENSITIVITY:
            for case in CASES:
                for seed in seeds:
                    tr = run_trial(case, seed, a, b, sampler, tasks, sched, split)
                    m = tr.metrics(); m["strategy"] = name
                    m["mapping"] = f"a{a}_b{b}"; m["split"] = split
                    rows.append(m)
    return pd.DataFrame(rows)

for split in ["validation", "test"]:
    res = run(split)
    res.to_csv(RESULTS / f"comparison_{split}.csv", index=False)
    print(f"\n===== {split} split ({len(res)} trials) =====")
    nom = res[res.mapping == "a1.0_b0.5"]
    tab = nom.groupby("strategy").agg(
        completion=("weighted_completion_rate", "mean"),
        ontime=("weighted_on_time_rate", "mean"),
        tardiness=("weighted_tardiness_s", "mean"),
        overrun=("observed_overrun_s", "mean"),
        idle=("reserved_idle_s", "mean"),
        cost=("cost", "mean"),
        censored=("censored_units", "mean"),
    ).round(3)
    print(tab.to_string())
    # 各策略每case ontime（约束检验）
    print("各case on-time:")
    print(nom.groupby(["strategy", "case"]).weighted_on_time_rate.mean().round(3).to_string())
    # 全映射均值
    print("全5映射 on-time 均值:")
    print(res.groupby("strategy").weighted_on_time_rate.mean().round(3).to_string())
