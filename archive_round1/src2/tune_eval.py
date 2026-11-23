"""调参与评估主驱动。

阶段1（调参，仅用 validation split、名义映射 90 trials）：
    网格搜索 ATC 的 kappa、截断防护分位 guard_q、预算取舍开关；
    Rollout 策略一并参加比较。
阶段2（冻结后验证全量）：5 映射 × 3 案例 × 30 种子 = 450 trials，
    主策略 vs 基线 WSPT 配对比较 + bootstrap 置信区间。
阶段3（模拟终测）：test split + 同一批公共种子（组委会种子未公开，
    此处按基线同样约定明确标注为模拟终测）。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import (ROOT, RESULTS, CASES, MAPPINGS, NOMINAL, VAL_SEEDS)
from duration import DurationModel
from policy import DispatchPolicy, RolloutPolicy, BaselineWSPT
from sampler import SourceSampler, run_trial

FEAT = ROOT / "prefix_features.csv"

_model = None
_sampler = None

def get_model():
    global _model
    if _model is None:
        df = pd.read_csv(FEAT)
        _model = DurationModel().fit(df[df.split == "train"])
    return _model

def get_sampler():
    global _sampler
    if _sampler is None:
        _sampler = SourceSampler(str(FEAT))
    return _sampler

def run_set(policy, split, mappings=(NOMINAL,), seeds=VAL_SEEDS, tag=""):
    rows = []
    sam = get_sampler()
    for (a, b) in mappings:
        for case in CASES:
            for seed in seeds:
                tr = run_trial(case, seed, a, b, sam, policy, split)
                m = tr.metrics()
                m["mapping"] = f"a{a}_b{b}"
                m["split"] = split
                m["policy"] = tag or policy.__class__.__name__
                rows.append(m)
    return pd.DataFrame(rows)

def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(x, size=len(x), replace=True))
             for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))

# ---------- 阶段1：调参 ----------
def phase1():
    model = get_model()
    grid = []
    for kappa in [3.0, 4.0, 5.0, 8.0]:
        for gq in [0.80, 0.90]:
            for sel in [False]:
                grid.append(("Dispatch", dict(kappa=kappa, guard_q=gq,
                                              selection=sel)))
    grid.append(("BaselineWSPT", {}))
    grid.append(("Rollout", dict(kappa=3.0, guard_q=0.90, selection=True,
                                 top_k=4, n_scen=48)))
    rows = []
    for name, kw in grid:
        t0 = time.time()
        pol = (BaselineWSPT(model) if name == "BaselineWSPT"
               else RolloutPolicy(model, **kw) if name == "Rollout"
               else DispatchPolicy(model, **kw))
        res = run_set(pol, "validation")
        nom = res[res.mapping == "a1.0_b0.5"]
        bycase = nom.groupby("case").weighted_on_time_rate.mean()
        rows.append(dict(
            name=name, kappa=kw.get("kappa"), guard_q=kw.get("guard_q"),
            selection=kw.get("selection"),
            ontime=nom.weighted_on_time_rate.mean(),
            min_case=bycase.min(),
            completion=nom.weighted_completion_rate.mean(),
            cost=nom.cost.mean(), overrun=nom.observed_overrun_s.mean(),
            valid=nom.valid_protocol.mean(),
            sec=round(time.time() - t0, 1)))
        r = rows[-1]
        print(f"{name:14s} k={r['kappa']} gq={r['guard_q']} sel={r['selection']}"
              f" -> on={r['ontime']:.3f} min={r['min_case']:.3f} "
              f"comp={r['completion']:.3f} cost={r['cost']:.1f} "
              f"over={r['overrun']:.1f} valid={r['valid']:.2f} "
              f"[{r['sec']}s]")
    tab = pd.DataFrame(rows).sort_values("ontime", ascending=False)
    tab.to_csv(RESULTS / "tune_grid.csv", index=False)
    print("\n=== 满足 min_case>=0.5 的排名 ===")
    ok = tab[tab.min_case >= 0.50]
    print(ok.to_string(index=False))
    return tab

# ---------- 阶段2：冻结策略验证全量（5 映射 × 3 案例 × 30 种子）----------
FROZEN = dict(kappa=4.0, guard_q=0.90, selection=True)

def phase2():
    model = get_model()
    policies = {
        "P0_基线WSPT": BaselineWSPT(model),
        "P1_ATC无防护": DispatchPolicy(model, kappa=FROZEN["kappa"],
                                       guard_q=0.01,
                                       selection=FROZEN["selection"]),
        "P2_主策略冻结": DispatchPolicy(model, **FROZEN),
    }
    all_rows = []
    for name, pol in policies.items():
        res = run_set(pol, "validation", mappings=MAPPINGS, tag=name)
        all_rows.append(res)
        res.to_csv(RESULTS / f"validation_{name}.csv", index=False)
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(RESULTS / "validation_all.csv", index=False)

    nom = full[full.mapping == "a1.0_b0.5"]
    print("\n=== 名义映射 验证集（90 trials/策略）===")
    tab = nom.groupby("policy").agg(
        completion=("weighted_completion_rate", "mean"),
        ontime=("weighted_on_time_rate", "mean"),
        tardiness=("weighted_tardiness_s", "mean"),
        cost=("cost", "mean"),
        overrun=("observed_overrun_s", "mean"),
        idle=("reserved_idle_s", "mean"),
        censored=("censored_units", "mean"),
        valid=("valid_protocol", "mean"),
    ).round(3)
    print(tab.to_string())

    # 消融 + 配对差异（名义映射，按案例）
    print("\n=== 每案例加权准时率（名义映射，均值±95%CI）===")
    for case in CASES:
        line = f"  {case}: "
        for name in policies:
            x = nom[(nom.policy == name) & (nom.case == case
                    )]["weighted_on_time_rate"].values
            lo, hi = bootstrap_ci(x)
            line += f"{name.split('_')[0]}={x.mean():.3f}({lo:.3f},{hi:.3f})  "
        print(line)

    print("\n=== 配对差异 P2−P0（同 case+seed）名义映射 ===")
    p2 = nom[nom.policy == "P2_主策略冻结"][["case", "seed",
            "weighted_on_time_rate"]].rename(
            columns={"weighted_on_time_rate": "y2"})
    p0 = nom[nom.policy == "P0_基线WSPT"][["case", "seed",
            "weighted_on_time_rate"]].rename(
            columns={"weighted_on_time_rate": "y0"})
    merged = p2.merge(p0, on=["case", "seed"])
    merged["d"] = merged.y2 - merged.y0
    for case in CASES:
        x = merged[merged.case == case]["d"].values
        lo, hi = bootstrap_ci(x)
        print(f"  {case}: mean diff={x.mean():+.4f}, 95%CI=({lo:+.4f},{hi:+.4f}),"
              f" win_rate={(x>0).mean():.2f}")
    x = merged["d"].values
    lo, hi = bootstrap_ci(x)
    print(f"  全部: mean diff={x.mean():+.4f}, 95%CI=({lo:+.4f},{hi:+.4f}),"
          f" win_rate={(x>0).mean():.2f}")

    # 约束检验：名义映射每案例 mean on-time >= 0.5
    print("\n=== 0.5 约束检验（P2 名义映射）===")
    for case in CASES:
        x = nom[(nom.policy == "P2_主策略冻结") & (nom.case == case
                )]["weighted_on_time_rate"].values
        lo, hi = bootstrap_ci(x)
        print(f"  {case}: {x.mean():.3f} [{lo:.3f},{hi:.3f}]"
              f" {'PASS' if x.mean() >= 0.5 else 'FAIL'} min={x.min():.3f}")

    # 全映射稳健性
    print("\n=== P2 全 5 映射（每映射 90 trials）===")
    g = full[full.policy == "P2_主策略冻结"].groupby("mapping").agg(
        ontime=("weighted_on_time_rate", "mean"),
        completion=("weighted_completion_rate", "mean"),
        cost=("cost", "mean"),
        overrun=("observed_overrun_s", "mean"),
        valid=("valid_protocol", "mean")).round(3)
    print(g.to_string())
    return full

# ---------- 阶段3：test split 模拟终测（冻结策略）----------
def phase3():
    model = get_model()
    pol = DispatchPolicy(model, **FROZEN)
    res = run_set(pol, "test", mappings=MAPPINGS, tag="P2_主策略冻结")
    res.to_csv(RESULTS / "test_frozen.csv", index=False)
    nom = res[res.mapping == "a1.0_b0.5"]
    print("\n=== 模拟终测 test split（名义映射，90 trials）===")
    print(nom.groupby("case").agg(
        ontime=("weighted_on_time_rate", "mean"),
        completion=("weighted_completion_rate", "mean"),
        cost=("cost", "mean"),
        valid=("valid_protocol", "mean")).round(3).to_string())
    print("\n全映射:")
    print(res.groupby("mapping").agg(
        ontime=("weighted_on_time_rate", "mean"),
        completion=("weighted_completion_rate", "mean"),
        valid=("valid_protocol", "mean")).round(3).to_string())
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="1")
    a = ap.parse_args()
    if a.phase == "1":
        phase1()
    elif a.phase == "2":
        phase2()
    elif a.phase == "3":
        phase3()
