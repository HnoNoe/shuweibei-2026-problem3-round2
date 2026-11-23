"""第 8 步 · 之二：b 权衡扫描（b_dominance.md §5 承诺的量化对照）。

b ∈ {0.2, 0.5, 1.0, 2.0}，其余决策与 C6 完全一致（决策仍按 b=0.2 的合法性
世界做，仅在发出 execute 时改预留值，且须过 s+b≤H 与预算两道真实检查，
过不了就退回 0.2）。名义映射 × 3 案例 × 30 验证种子。
预期（解析结论）：准时率/overrun 不变，费用单调上升 —— 曲线画出来给论文。
"""

import json
import time

import numpy as np
import pandas as pd

from common import (B_MIN, CHANGEOVER, FEE_FIXED, FEE_RESERVE_PER_S,
                    FEE_DIST_PER_UNIT, MAPPINGS, RESULTS, ROOT, VAL_SEEDS,
                    zone_dist)
from duration_model3 import DurationModel
from sampler import SourceSampler, run_trial
from strategy import EPS, TwoStagePolicy

B_VALUES = [0.2, 0.5, 1.0, 2.0]


class BScanPolicy:
    """C6 包装：execute 时把预留改成 b_scan（受真实合法性约束）。"""

    def __init__(self, b_scan, hyper, model):
        self.inner = TwoStagePolicy("C6_full", hyper=hyper, model=model)
        self.b = b_scan

    def reset(self, trial):
        self.inner.reset(trial)

    def decide(self, snap):
        act = self.inner.decide(snap)
        if not act or act[0] != "execute" or abs(self.b - B_MIN) < EPS:
            return act
        _, jid, _ = act
        info = snap["jobs"][jid]
        D = zone_dist(snap["zone"], info["zone"])
        s = snap["t"] + D + CHANGEOVER
        fee = FEE_FIXED + FEE_RESERVE_PER_S * self.b + FEE_DIST_PER_UNIT * D
        if s + self.b <= snap["H"] + EPS and snap["cost"] + fee <= snap["B"] + EPS:
            return ("execute", jid, self.b)
        return act                           # 过不了真实检查就退回 0.2


def main():
    frozen = json.load(open(ROOT / "round2/results/q7_tuning.json",
                            encoding="utf-8"))["frozen"]
    model = DurationModel.from_gofit()
    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))
    rows = []
    t0 = time.time()
    for b in B_VALUES:
        for case in ("S1", "S2", "S3"):
            for seed in VAL_SEEDS:
                pol = BScanPolicy(b, frozen, model)
                tr = run_trial(case, seed, 1.0, 0.5, sampler, pol, "validation")
                m = tr.metrics()
                rows.append(dict(b=b, **m))   # metrics 已含 case/seed
        print(f"b={b} done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    bad = int((df["valid_protocol"] != 1).sum())
    assert bad == 0, f"b 扫描协议不合规 {bad} 局"
    df.to_csv(RESULTS / "q8_btradeoff.csv", index=False, encoding="utf-8-sig")
    agg = df.groupby("b").agg(
        ot=("weighted_on_time_rate", "mean"),
        completion=("weighted_completion_rate", "mean"),
        cost=("cost", "mean"),
        overrun=("observed_overrun_s", "mean"),
        idle=("reserved_idle_s", "mean")).reset_index()
    agg.to_csv(RESULTS / "q8_btradeoff_summary.csv", index=False,
               encoding="utf-8-sig")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
