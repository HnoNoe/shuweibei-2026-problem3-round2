"""第 8 步 · 补丁：重跑 RM_buytop 臂（validation）。

背景（2026-11-23）：review_max.py 的 GBM 特征键不匹配（训练列带 "p10_" 前缀、
快照特征键已剥离）导致 _gbm_d 恒返回 None —— "买到顶"反证的**信息通道从未生效**，
旧 q7_ablation.csv 的 RM 行是失效版结果。本脚本只重跑 RM 单臂
（5 映射 × 3 案例 × 30 验证种子 = 450 局，CRN 种子与 run_ablation 完全一致），
输出 q7_ablation_rm_fixed.csv；由 merge_rm_fix.py 替换进主表后再跑 analyze。
"""

import csv
import json
import time

from common import MAPPINGS, ROOT, VAL_SEEDS
from duration_model3 import DurationModel
from review_max import ReviewMaxPolicy
from sampler import SourceSampler, run_trial


def main():
    frozen = json.load(open(ROOT / "round2/results/q7_tuning.json",
                            encoding="utf-8"))["frozen"]
    print("FROZEN hyper:", frozen, flush=True)
    model = DurationModel.from_gofit()
    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))

    out = ROOT / "round2/results/q7_ablation_rm_fixed.csv"
    t0 = time.time()
    n = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "alpha", "beta", "case", "seed",
                    "weighted_on_time_rate", "weighted_completion_rate",
                    "weighted_tardiness_s", "cost", "observations",
                    "censored_units", "invalid_actions", "valid_protocol",
                    "observed_overrun_s", "reserved_idle_s"])
        for alpha, beta in MAPPINGS:
            for case in ("S1", "S2", "S3"):
                for seed in VAL_SEEDS:
                    pol = ReviewMaxPolicy(hyper=frozen, model=model)
                    tr = run_trial(case, seed, alpha, beta, sampler,
                                   pol, "validation")
                    m = tr.metrics()
                    w.writerow(["RM_buytop", alpha, beta, case, seed,
                                m["weighted_on_time_rate"],
                                m["weighted_completion_rate"],
                                m["weighted_tardiness_s"], m["cost"],
                                m["observations"], m["censored_units"],
                                m["invalid_actions"], m["valid_protocol"],
                                m["observed_overrun_s"],
                                m["reserved_idle_s"]])
                    n += 1
                    if n % 50 == 0:
                        print(f"{n}/450, {time.time()-t0:.0f}s", flush=True)
                        f.flush()
    print(f"DONE {n} trials in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
