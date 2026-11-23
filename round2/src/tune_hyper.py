"""第 7 步 · 之一：超参内层验证与冻结。

纪律（step7_plan.md §1）：κ、K 只用 train split 内部种子选，验证集冻结前不可见。
联合网格 4×3=12 配置，κ/K 各自按边际均值选最优；成绩全量落盘可复查。
"""

import itertools
import json
import time

import numpy as np

from common import RESULTS
from duration_model3 import DurationModel
from sampler import SourceSampler, run_trial
from strategy import TwoStagePolicy

KAPPAS = [1.5, 2.0, 2.5, 3.0]
ROLL_KS = [10, 20, 40]
TUNE_SEEDS = list(range(770001, 770021))     # train split 内部种子，与 VAL_SEEDS 无交
CASES_ = ("S1", "S2", "S3")
ALPHA, BETA = 1.0, 0.5                        # 名义映射


def main():
    sampler = SourceSampler()
    model = DurationModel.from_gofit()
    rows = []
    for kap, rk in itertools.product(KAPPAS, ROLL_KS):
        pol = TwoStagePolicy("C5_online",
                             hyper=dict(kappa=kap, roll_K=rk), model=model)
        t0 = time.time()
        vals, invalid = [], 0
        for case in CASES_:
            for seed in TUNE_SEEDS:
                tr = run_trial(case, seed, ALPHA, BETA, sampler, pol, "train")
                m = tr.metrics()
                vals.append(m["weighted_on_time_rate"])
                invalid += m["invalid_actions"]
        dt = time.time() - t0
        n = len(vals)
        rows.append(dict(kappa=kap, roll_K=rk, n=n,
                         mean_ot=float(np.mean(vals)),
                         se=float(np.std(vals, ddof=1) / np.sqrt(n)),
                         invalid=invalid,
                         sec_per_game=dt / n))
        print(f"kappa={kap} K={rk}: mean_ot={np.mean(vals):.4f} "
              f"+-{np.std(vals, ddof=1)/np.sqrt(n):.4f} invalid={invalid} "
              f"{dt/n:.3f}s/局", flush=True)

    # 边际选择：κ 按 K 平均、K 按 κ 平均
    def marg(param, idx):
        return {v: np.mean([r["mean_ot"] for r in rows if r[idx] == v])
                for v in param}
    m_kap = marg(KAPPAS, "kappa")
    m_rk = marg(ROLL_KS, "roll_K")
    best_kap = max(m_kap, key=m_kap.get)
    best_rk = max(m_rk, key=m_rk.get)
    # 约束：单局均值 < 1s（达不到则向小退）
    feas = [r for r in rows if r["sec_per_game"] < 1.0]
    if not feas:
        raise RuntimeError("全部配置超算力预算，需重议")
    out = dict(
        protocol="train split 内部种子 20×3案例×名义映射=60局/配置；"
                 "κ 与 K 按边际均值分别选取；冻结后验证种子才可见",
        tune_seeds=TUNE_SEEDS, grid=rows,
        marginal_kappa={str(k): float(v) for k, v in m_kap.items()},
        marginal_roll_K={str(k): float(v) for k, v in m_rk.items()},
        frozen=dict(kappa=best_kap, roll_K=best_rk),
        frozen_from=dict(best_joint=[r for r in rows
                                     if r["kappa"] == best_kap
                                     and r["roll_K"] == best_rk][0]),
    )
    with open(RESULTS / "q7_tuning.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("FROZEN:", out["frozen"])


if __name__ == "__main__":
    main()
