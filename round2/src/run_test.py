
import csv
import json
import time

import numpy as np
from scipy import stats

from common import MAPPINGS, RESULTS, ROOT
from duration_model3 import DurationModel
from review_max import ReviewMaxPolicy
from sampler import SourceSampler, run_trial
from strategy import TwoStagePolicy, baseline

TEST_SEEDS = list(range(990001, 990031))

ARMS = ["C5_online", "C6_full", "B3_WSPT_Q90", "RM_buytop"]

def main():
    frozen = json.load(open(ROOT / "round2/results/q7_tuning.json",
                            encoding="utf-8"))["frozen"]
    model = DurationModel.from_gofit()
    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))
    rows = []
    t0 = time.time()
    for alpha, beta in MAPPINGS:
        for case in ("S1", "S2", "S3"):
            for seed in TEST_SEEDS:
                for arm in ARMS:
                    if arm == "C6_full":
                        pol = TwoStagePolicy("C6_full", hyper=frozen, model=model)
                    elif arm == "C5_online":
                        pol = TwoStagePolicy("C5_online", hyper=frozen, model=model)
                    elif arm == "B3_WSPT_Q90":
                        pol = baseline("B3_WSPT_Q90", model=model)
                    else:
                        pol = ReviewMaxPolicy(hyper=frozen, model=model)
                    tr = run_trial(case, seed, alpha, beta, sampler, pol, "test")
                    m = tr.metrics()
                    rows.append(dict(arm=arm, **m))

                if len(rows) % 90 == 0:
                    print(f"{len(rows)//(len(ARMS)*3)}/450 trials "
                          f"{time.time()-t0:.0f}s", flush=True)
    bad = [r for r in rows if r["valid_protocol"] != 1]
    assert not bad, f"test 集协议不合规 {len(bad)} 局"

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "q8_test.csv", index=False, encoding="utf-8-sig")

    piv = df.pivot_table(index=["alpha", "beta", "case", "seed"], columns="arm",
                         values="weighted_on_time_rate")
    summary = dict(n_games=int(len(rows) // len(ARMS)),
                   protocol_violations=0, frozen=frozen)
    for a, b in [("C5_online", "B3_WSPT_Q90"), ("C6_full", "C5_online"),
                 ("RM_buytop", "C5_online")]:
        d = (piv[a] - piv[b]).dropna()
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=d.std(ddof=1) / np.sqrt(len(d)))
        summary[f"{a}_minus_{b}"] = dict(
            mean=float(d.mean()), ci95=[float(ci[0]), float(ci[1])],
            median=float(d.median()), n=int(len(d)))
    for arm in ARMS:
        sub = df[df.arm == arm]
        summary.setdefault("by_arm", {})[arm] = dict(
            ot=float(sub["weighted_on_time_rate"].mean()),
            completion=float(sub["weighted_completion_rate"].mean()),
            cost=float(sub["cost"].mean()))
    with open(RESULTS / "q8_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
