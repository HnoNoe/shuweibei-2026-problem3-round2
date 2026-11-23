"""策略搜索 v2：reserve_mode × order_rule × review_mode。"""
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
    bycase = d.groupby("case").weighted_on_time_rate.mean()
    return dict(ontime=d.weighted_on_time_rate.mean(),
                ontime_min_case=bycase.min(),
                completion=d.weighted_completion_rate.mean(),
                tardiness=d.weighted_tardiness_s.mean(),
                idle=d.reserved_idle_s.mean(),
                overrun=d.observed_overrun_s.mean(),
                cost=d.cost.mean(), obs=d.observations.mean(),
                valid=d.valid_protocol.mean())

grid = []
for reserve in ["min", "median", "q75"]:
    for rule in ["combined", "edd", "wspt", "slack"]:
        for review in ["none", "selective"]:
            grid.append((reserve, rule, review))

results = []
for reserve, rule, review in grid:
    p = default_params(); p.update(reserve_mode=reserve, order_rule=rule, review_mode=review)
    r = evaluate_params(p)
    r.update(reserve=reserve, rule=rule, review=review)
    results.append(r)
    print(f"{reserve:7s} {rule:8s} {review:9s} -> "
          f"on={r['ontime']:.3f} min={r['ontime_min_case']:.3f} "
          f"comp={r['completion']:.3f} tard={r['tardiness']:6.0f} "
          f"idle={r['idle']:5.1f} over={r['overrun']:4.1f} cost={r['cost']:.1f}")

res = pd.DataFrame(results)
res.to_csv(RESULTS / "param_search_v2.csv", index=False)
ok = res[res.ontime_min_case >= 0.50].sort_values("ontime", ascending=False)
print("\n=== 满足 min_case on-time>=0.5 的配置 ===")
print(ok[["reserve","rule","review","ontime","ontime_min_case","completion","tardiness","idle","overrun"]].to_string(index=False))
print("\n满足约束数:", len(ok), "/", len(res))
