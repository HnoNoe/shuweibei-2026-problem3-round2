"""策略参数搜索：在 validation 上找到满足 on-time>=0.5 且综合指标最优的配置。

合规：只用 validation seeds 调参，test 前冻结。
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from config import VAL_SEEDS, CASES, NOMINAL_ALPHA, NOMINAL_BETA, RESULTS
from duration_model import DurationEstimator
from scheduler import Scheduler, default_params
from runner import SourceSampler, load_tasks, run_trial

df = pd.read_csv("data/prefix_features.csv")
est = DurationEstimator(rho_shrink=0.5).fit(df[df.split == "train"])
sampler = SourceSampler("data/prefix_features.csv")
task_by_case = load_tasks()

def evaluate_params(p, seeds=VAL_SEEDS):
    sched = Scheduler(est, p)
    rows = []
    for case in CASES:
        for seed in seeds:
            tr = run_trial(case, seed, NOMINAL_ALPHA, NOMINAL_BETA,
                           sampler, task_by_case, sched, "validation")
            rows.append(tr.metrics())
    d = pd.DataFrame(rows)
    return dict(
        ontime=d.weighted_on_time_rate.mean(),
        ontime_min_case=d.groupby("case").weighted_on_time_rate.mean().min(),
        completion=d.weighted_completion_rate.mean(),
        tardiness=d.weighted_tardiness_s.mean(),
        idle=d.reserved_idle_s.mean(),
        overrun=d.observed_overrun_s.mean(),
        cost=d.cost.mean(),
    )

# 搜索：风险系数 risk 与 各优先级权重
grid = []
for risk in [0.0, 0.25, 0.5, 0.75]:
    for a_u in [0.5, 1.2, 2.0, 3.0]:
        for a_d in [0.0, 0.5, 1.5]:
            p = default_params()
            p.update(risk=risk, a_u=a_u, a_d=a_d)
            grid.append((risk, a_u, a_d, p))

results = []
for risk, a_u, a_d, p in grid:
    r = evaluate_params(p)
    r.update(risk=risk, a_u=a_u, a_d=a_d)
    results.append(r)
    print(f"risk={risk:.2f} a_u={a_u:.1f} a_d={a_d:.1f} -> ontime={r['ontime']:.3f} "
          f"min_case={r['ontime_min_case']:.3f} comp={r['completion']:.3f} "
          f"tard={r['tardiness']:.0f} idle={r['idle']:.1f}")

res = pd.DataFrame(results)
res.to_csv(RESULTS / "param_search.csv", index=False)
# 筛选满足约束的组合，按 ontime 最大
ok = res[res.ontime_min_case >= 0.50].sort_values("ontime", ascending=False)
print("\n=== 满足 on-time>=0.5 的 Top10 ===")
print(ok[["risk","a_u","a_d","ontime","ontime_min_case","completion","tardiness","idle"]].head(10).to_string(index=False))
print("\n满足约束组合数:", len(ok), "/", len(res))
