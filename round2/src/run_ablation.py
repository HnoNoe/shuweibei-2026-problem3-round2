
import csv
import json
import time

from common import MAPPINGS, ROOT, VAL_SEEDS
from duration_model3 import DurationModel
from review_max import ReviewMaxPolicy
from sampler import SourceSampler, run_trial
from strategy import TwoStagePolicy, baseline

ARMS = ["B1_EDD", "B2_WSPT_MEAN", "B2b_WSPT_MEDIAN", "B3_WSPT_Q90",
        "C0_bare", "C1_atc", "C2_guard", "C3_budget", "C4_roll",
        "C5_online", "C6_full", "RM_buytop"]

def make_arm(name, model, frozen):
    if name.startswith("B"):
        return baseline(name, model=model)
    if name == "RM_buytop":
        return ReviewMaxPolicy(hyper=frozen, model=model)
    return TwoStagePolicy(name, hyper=frozen, model=model)

def main():
    frozen = json.load(open(ROOT / "round2/results/q7_tuning.json",
                            encoding="utf-8"))["frozen"]
    print("FROZEN hyper:", frozen, flush=True)
    model = DurationModel.from_gofit()
    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))

    out = ROOT / "round2/results/q7_ablation.csv"
    t_start = time.time()
    n_done = 0
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
                    for arm in ARMS:
                        pol = make_arm(arm, model, frozen)
                        tr = run_trial(case, seed, alpha, beta, sampler,
                                       pol, "validation")
                        m = tr.metrics()
                        w.writerow([arm, alpha, beta, case, seed,
                                    m["weighted_on_time_rate"],
                                    m["weighted_completion_rate"],
                                    m["weighted_tardiness_s"], m["cost"],
                                    m["observations"], m["censored_units"],
                                    m["invalid_actions"], m["valid_protocol"],
                                    m["observed_overrun_s"],
                                    m["reserved_idle_s"]])
                    n_done += 1
                    if n_done % 25 == 0:
                        el = time.time() - t_start
                        print(f"{n_done}/450 trials, {el:.0f}s, "
                              f"ETA {el/n_done*(450-n_done):.0f}s", flush=True)
                        f.flush()
    print(f"DONE {n_done} trials in {time.time()-t_start:.0f}s -> {out}")

if __name__ == "__main__":
    main()
