
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from common import (B_MIN, CHANGEOVER, PREFIXES, PROBLEM, REVIEW_COST, ROOT,
                    zone_dist)
from strategy import CONFIGS, EPS, FastSim, TwoStagePolicy, fee_of

MODELS = {}

def _train_gbms():

    global MODELS
    if MODELS:
        return MODELS
    feats = pd.read_csv(ROOT / "prefix_features.csv")
    df = feats[feats["split"] == "train"]
    MODELS = {}
    for lvl, pre in [(PREFIXES[0], "p02"), (PREFIXES[1], "p06"),
                     (PREFIXES[2], "p10")]:
        cols = [c for c in df.columns if c.startswith(pre + "_")]
        sub = df[cols + ["duration_s"]].dropna()
        g = GradientBoostingRegressor(random_state=7)

        g.fit(sub[cols].to_numpy(), sub["duration_s"].to_numpy())

        MODELS[lvl] = (g, cols, {c: c[len(pre) + 1:] for c in cols},
                       sub[cols].median())
    return MODELS

class ReviewMaxPolicy(TwoStagePolicy):

    def __init__(self, hyper=None, model=None):
        super().__init__(CONFIGS["C5_online"], hyper=hyper, model=model)
        _train_gbms()
        self._snap = None
        self._gbm_cache = {}

    def _est(self, jid, info, q=None):
        if q is not None:
            return self._d_hat(info["family"], q=q)
        pred = self._gbm_d(jid)
        if pred is not None:
            return min(max(pred, 0.05), 300.0)
        return self._d_hat(info["family"])

    def _gbm_d(self, jid):
        if self._snap is None:
            return None
        per = self._snap.get("prefix_feat", {}).get(jid, {})
        for lvl in sorted(MODELS, reverse=True):
            if lvl in per and per[lvl]:
                g, cols, mapping, med = MODELS[lvl]
                f = per[lvl]
                if all(mapping[c] in f for c in cols):

                    key = (jid, lvl)
                    if key not in self._gbm_cache:
                        x = [f[mapping[c]] if pd.notna(f[mapping[c]])
                             else med[c] for c in cols]
                        self._gbm_cache[key] = float(g.predict([x])[0])
                    return self._gbm_cache[key]
        return None

    def _review_trigger(self, snap):
        jobs = snap["jobs"]
        t = snap["t"]
        cands = []
        for jid, info in jobs.items():
            if jid in snap["executed"]:
                continue
            if t + EPS < info["release_s"]:
                continue
            if any(p not in snap["completed"] for p in info["predecessors"]):
                continue
            D = zone_dist(snap["zone"], info["zone"])
            if snap["cost"] + fee_of(D) > snap["B"] + EPS:
                continue
            if t + D + CHANGEOVER + B_MIN > snap["H"] + EPS:
                continue
            cands.append(jid)
        if not cands:
            return None
        sim_now = self._sim_now(snap, cands)
        top = max(cands, key=lambda j: self._score(sim_now, j, jobs))
        if snap["review_count"][top] < 2 and snap["reviews"] < snap["K"] \
                and snap["cost"] + REVIEW_COST <= snap["B"] + EPS \
                and t + 1.0 <= snap["H"] + EPS:
            return ("review", top)
        return None

    def _sim_now(self, snap, cands):
        sim = FastSim(snap, {}, snap["alpha"], snap["beta"], snap["H"], snap["B"])
        sim.dur = {j: snap["alpha"] * self._est(j, snap["jobs"][j]) + snap["beta"]
                   for j in cands}
        return sim

    def decide(self, snap):
        self._snap = snap
        act = self._review_trigger(snap)
        if act is not None:
            return act
        return super().decide(snap)

if __name__ == "__main__":

    from common import CASES, VAL_SEEDS
    from duration_model3 import DurationModel
    from sampler import SourceSampler, run_trial

    sampler = SourceSampler(feat_csv=str(ROOT / "prefix_features.csv"))
    pol = ReviewMaxPolicy(model=DurationModel.from_gofit())
    for case in ("S1", "S3"):
        tr = run_trial(case, VAL_SEEDS[0], 1.0, 0.5, sampler, pol, "validation")
        m = tr.metrics()
        print(case, "reviews=", m["observations"], "cost=", m["cost"],
              "ot=", round(m["weighted_on_time_rate"], 4),
              "invalid=", m["invalid_actions"])
    print("REVIEW_MAX SMOKE OK")
