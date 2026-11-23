"""源区间采样与 trial 装配（严格按赛题第 1 节）。

- random.Random(int(seed)) 每案例重新初始化；
- 按 job_id 排序依次 rng.choice(split, family) 池（保持源索引行顺序）；
- 采样与策略后续动作无关（同 seed 的策略共享同一源分配，可配对比较）。
"""
import random
import pandas as pd
from common import (load_tasks, load_source_index, eligible_pool, PREFIXES,
                    MAX_ACTIONS)
from simulator import Job, Trial

_FEAT_CSV = None
_FEAT_MAP = None


def load_features(csv_path):
    """unit_id -> {0.2: {feat:val}, 0.6: {...}, 1.0: {...}}（仅前缀特征，无真值）。"""
    global _FEAT_CSV, _FEAT_MAP
    df = pd.read_csv(csv_path)
    meta = {"unit_id", "dataset", "sample", "environment_id", "split",
            "family_id", "source_segment_index", "start_s", "end_s",
            "duration_s", "available"}
    feat_cols = [c for c in df.columns if c not in meta]
    m = {}
    for _, r in df.iterrows():
        per = {}
        for lvl, pre in [(PREFIXES[0], "p02"), (PREFIXES[1], "p06"),
                         (PREFIXES[2], "p10")]:
            d = {c[len(pre) + 1:]: float(r[c])
                 for c in feat_cols if c.startswith(pre + "_")}
            per[lvl] = d
        m[r["unit_id"]] = per
    _FEAT_CSV, _FEAT_MAP = csv_path, m
    return m


class SourceSampler:
    def __init__(self, feat_csv=None):
        self.tasks = load_tasks()
        self.pool = eligible_pool(load_source_index())
        if feat_csv:
            load_features(feat_csv)
        self.feat_map = _FEAT_MAP or {}

    def sample_jobs(self, case, split, seed):
        rng = random.Random(int(seed))
        jobs = []
        for row in self.tasks[case]:
            src = rng.choice(self.pool[(split, row["family"])])
            jobs.append(Job(
                job_id=row["job_id"], family=row["family"],
                release_s=row["release_s"], due_s=row["due_s"],
                weight=row["weight"], zone=row["zone"],
                predecessors=row["predecessors"], d=src["d"],
                source_ref=src["unit_id"],
                prefix_feat=self.feat_map.get(src["unit_id"])))
        return jobs


def run_trial(case, seed, alpha, beta, sampler, policy, split):
    from common import CASES
    cfg = CASES[case]
    jobs = sampler.sample_jobs(case, split, seed)
    trial_seeds = dict(case=case, seed=seed, alpha=alpha, beta=beta,
                       H=cfg["H"], B=cfg["B"], K=cfg["K"])
    tr = Trial(jobs, H=cfg["H"], B=cfg["B"], K=cfg["K"],
               alpha=alpha, beta=beta, seed=seed, case=case)
    policy.reset(trial_seeds)
    for _ in range(MAX_ACTIONS + 10):
        if tr.terminated:
            break
        act = policy.decide(tr.snapshot())
        if act is None or act[0] == "stop":
            tr.stop()
            break
        name = act[0]
        if name == "execute":
            ok, _ = tr.execute(act[1], act[2])
        elif name == "review":
            ok, _ = tr.review(act[1])
        elif name == "wait":
            ok, _ = tr.wait(act[1])
        else:
            tr.stop()
            break
        if not ok and tr.terminated:
            break
    if not tr.terminated:
        tr.stop()
    return tr
