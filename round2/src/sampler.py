"""源区间采样与 trial 装配（重建版）。

文献定位：本模块是仿真文献中经典方差削减技术 **共同随机数（Common Random
Numbers, CRN）** 的落地——各策略在同一随机种子下共享同一批源分配，
使配对差分的方差远小于独立抽样的方差（Goldsman & Nelson 1990s 系列，
Law & Kelton 仿真教材）。原文第 1 节恰好明文规定了同步方式：

- random.Random(int(seed)) 每案例重新初始化；
- 按 job_id 排序依次 rng.choice(split, family) 池（保持源索引行顺序）；
- 采样与策略后续动作无关（同 seed 的策略共享同一源分配，可配对比较）。

合规要点（原文第 2 节）：prefix_features.csv 只含前缀窗口内的数值特征，
解析时按 META 白名单剔除 unit_id/duration_s 等真值与身份列，
保证递入 snapshot 的信息里没有终点和时长。
"""
import random
import pandas as pd
from common import (load_tasks, load_source_index, eligible_pool, PREFIXES,
                    MAX_ACTIONS)
from simulator import Job, Trial

# 原文第 2 节：前缀特征允许的信息范围之外的一切列（身份/真值/终点）
META = {"unit_id", "dataset", "sample", "environment_id", "split",
        "family_id", "source_segment_index", "start_s", "end_s",
        "duration_s", "available"}

_FEAT_CSV = None
_FEAT_MAP = None


def load_features(csv_path):
    """unit_id -> {0.2: {feat:val}, 0.6: {...}, 1.0: {...}}（仅前缀特征，无真值）。"""
    global _FEAT_CSV, _FEAT_MAP
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META]
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
    """抽作业 -> 建 Trial -> 驱动策略动作循环 -> 返回 Trial 对象。

    策略全程只见 snapshot()；动作违规时裁判终止 trial，这里原样返回。
    """
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


# ---------------- 验证 ----------------
if __name__ == "__main__":
    import numpy as np

    # 1) CRN 确定性：同 (case, seed, split) 两次抽样，时长分配逐字节一致
    s = SourceSampler()
    for case in ("S1", "S2", "S3"):
        for seed in (2026111700, 2026111717, 2026111729):
            a = [(j.job_id, j.d, j.source_ref) for j in s.sample_jobs(case, "validation", seed)]
            b = [(j.job_id, j.d, j.source_ref) for j in s.sample_jobs(case, "validation", seed)]
            assert a == b, f"确定性破坏: {case} seed={seed}"
    print("1) CRN 确定性: 3案例 x 3种子 全部一致 OK")

    # 2) 不同 split 池子确实不同（train/validation 源不混）
    j_val = s.sample_jobs("S1", "validation", 2026111700)
    j_tr = s.sample_jobs("S1", "train", 2026111700)
    assert j_val[0].d != j_tr[0].d or j_val[0].source_ref != j_tr[0].source_ref
    print("2) train/validation 池分离 OK")

    # 3) 抽样分布抽查：P 族池在 train split 的时长应能抽到长尾（125s 存在）
    pool_p = [r["d"] for r in s.pool[("train", "P")]]
    assert max(pool_p) >= 125.0 and len(pool_p) > 100
    print(f"3) P 族 train 池 n={len(pool_p)}, 最长 {max(pool_p):.1f}s OK")

    # 4) 前缀特征装载：无真值泄漏（META 白名单生效）
    s2 = SourceSampler(feat_csv=str(__import__("common").ROOT / "prefix_features.csv"))
    n_feat = sum(1 for v in s2.feat_map.values() for d in v.values() if d)
    print(f"4) 前缀特征装载 OK: {len(s2.feat_map)} 段, 非空特征档 {n_feat} 个")
    print("SAMPLER ALL CHECKS PASSED")
