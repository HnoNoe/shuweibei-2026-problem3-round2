"""验证评估：5 种映射 × 3 案例 × 30 公共种子，产出 trial-level 全指标与汇总。

合规要求（题目第4/5节）：
- 30 trials/case，90 across cases，450 across five mappings。
- 共享策略用相同源分配（同 seed）配对。
- 报告均值、变异、不确定性。
- 若以成本/风险为目标，检验 mean weighted on-time rate >= 0.5。
"""
import json, sys
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from config import VAL_SEEDS, SENSITIVITY, CASES, RESULTS
from duration_model import DurationEstimator
from scheduler import Scheduler, default_params
from runner import SourceSampler, load_tasks, run_trial

def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (0, 0)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))

def main(feat_csv="data/prefix_features.csv", split="validation"):
    df = pd.read_csv(feat_csv)
    train = df[df["split"] == "train"]
    # 冻结时长模型（只用 train 拟合）
    est = DurationEstimator(rho_shrink=0.5).fit(train)
    print("时长模型训练完成。train评估:", est.evaluate(train))
    va = df[df["split"] == "validation"]
    if len(va):
        print("validation评估:", est.evaluate(va))

    sampler = SourceSampler(feat_csv)
    task_by_case = load_tasks()
    sched = Scheduler(est, default_params())

    rows = []
    for (alpha, beta) in SENSITIVITY:
        for case in CASES:
            for seed in VAL_SEEDS:
                tr = run_trial(case, seed, alpha, beta, sampler, task_by_case, sched, split)
                m = tr.metrics()
                m["mapping"] = f"a{alpha}_b{beta}"
                m["split"] = split
                rows.append(m)
    trials = pd.DataFrame(rows)
    trials.to_csv(RESULTS / f"trial_results_{split}.csv", index=False)
    print("trial结果:", trials.shape, "->", RESULTS / f"trial_results_{split}.csv")

    # 汇总：按 mapping × case
    key = ["mapping", "case"]
    metrics = ["weighted_completion_rate", "weighted_on_time_rate",
               "weighted_tardiness_s", "cost", "observations",
               "observed_overrun_s", "reserved_idle_s", "censored_units",
               "elapsed_s", "valid_protocol"]
    agg = trials.groupby(key)[metrics].agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS / f"summary_{split}.csv")
    print("\n=== 汇总（均值）===")
    mean_tab = trials.groupby(key)[metrics].mean()
    print(mean_tab.round(3).to_string())

    # on-time >= 0.5 约束检验（名义映射）
    nom = trials[trials["mapping"] == "a1.0_b0.5"]
    print("\n=== on-time>=0.5 约束检验（名义, 每case 30 trials）===")
    for case in CASES:
        x = nom[nom["case"] == case]["weighted_on_time_rate"].values
        ci = bootstrap_ci(x)
        print(f"  {case}: mean={x.mean():.3f} 95%CI=({ci[0]:.3f},{ci[1]:.3f}) "
              f"min={x.min():.3f} PASS={x.mean() >= 0.5}")
    # 总体
    print("\n=== 全局指标（名义映射）===")
    for m in metrics:
        x = nom[m].values
        print(f"  {m}: {x.mean():.3f} +/- {x.std():.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/prefix_features.csv")
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()
    main(a.feat, a.split)
