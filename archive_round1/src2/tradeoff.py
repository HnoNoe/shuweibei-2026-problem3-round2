"""预留时长 b 的多目标权衡扫描（问题三内容）。

b 不影响完成时刻（C=s+p），只改变：费用（0.1/s）、reserved_idle（b>p 时）、
observed_overrun（p>b 时）。主策略 b=0.2 最大化准时率/最小化成本；
风险导向策略取 b = Q_q(p)，用 overrun 换 idle 的下降。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import RESULTS, NOMINAL
from policy import DispatchPolicy
from tune_eval import get_model, run_set, FROZEN, bootstrap_ci

def main():
    model = get_model()
    rows = []
    for rq, label in [(None, "b=0.20(最小)"), (0.50, "b=中位数"),
                      (0.75, "b=Q75"), (0.90, "b=Q90")]:
        pol = DispatchPolicy(model, kappa=FROZEN["kappa"],
                             guard_q=FROZEN["guard_q"],
                             selection=FROZEN["selection"], reserve_q=rq)
        res = run_set(pol, "validation", mappings=(NOMINAL,), tag=label)
        nom = res[res.mapping == "a1.0_b0.5"]
        rows.append(dict(
            reserve=label,
            ontime=nom.weighted_on_time_rate.mean(),
            completion=nom.weighted_completion_rate.mean(),
            cost=nom.cost.mean(),
            overrun=nom.observed_overrun_s.mean(),
            idle=nom.reserved_idle_s.mean(),
            censored=nom.censored_units.mean(),
            valid=nom.valid_protocol.mean()))
        r = rows[-1]
        print(f"{label:14s} on={r['ontime']:.3f} comp={r['completion']:.3f} "
              f"cost={r['cost']:.2f} over={r['overrun']:.1f} "
              f"idle={r['idle']:.1f} cens={r['censored']:.2f}")
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(RESULTS / "tradeoff_b.csv", index=False)

if __name__ == "__main__":
    main()
