"""试验运行器：源区间采样 + 策略驱动 + trial 执行。

源采样规则（题目第1节，严格执行）：
- 按 job_id 排序；每个 job 从「指定 split + family」的 eligible 区间池中
  uniform with replacement 采样一个；保持 source_segment_index 行顺序；
  每个 case 重新初始化 random.Random(int(seed))；choice 依次调用。
- 验证用 validation split，测试用 test split。
"""
import csv, random
from collections import defaultdict
from typing import Dict, List
import sys
sys.path.insert(0, "src")
from config import CONSTRAINTS, CASES, DATA
from simulator import Job, Trial
import pandas as pd
sys.path.insert(0, "src")
from duration_model import DurationEstimator

IDX = CONSTRAINTS / "source_segment_index.csv"
TASKS = CONSTRAINTS / "tasks.csv"

class SourceSampler:
    def __init__(self, feat_csv=None):
        rows = list(csv.DictReader(open(IDX, encoding="utf-8-sig")))
        self.pool = defaultdict(list)
        for r in rows:
            if r["eligible"] != "1":
                continue
            key = (r["split"], r["family_id"])
            self.pool[key].append(dict(
                unit_id=r["unit_id"],
                d=float(r["end_s"]) - float(r["start_s"]),
                start=float(r["start_s"]), end=float(r["end_s"]),
                dataset=r["dataset"], sample=r["sample"],
            ))
        # 建立 unit_id -> 前缀强度 映射（策略可见的特征，非真值）
        self.intensity = {}
        if feat_csv is not None:
            import os
            if not os.path.exists(feat_csv):
                feat_csv = str(DATA / "prefix_features.csv")
            fdf = pd.read_csv(feat_csv)
            for _, r in fdf.iterrows():
                d = {}
                for lvl, pre in [(0.2, "p02"), (0.6, "p06"), (1.0, "p10")]:
                    d[lvl] = float(DurationEstimator._intensity(r, pre))
                self.intensity[r["unit_id"]] = d
        for k in self.pool:  # 保持行顺序（读取顺序即文件顺序）
            pass

    def sample_jobs(self, case_jobs, split: str, seed: int) -> List[Job]:
        rng = random.Random(int(seed))
        jobs = []
        for row in sorted(case_jobs, key=lambda r: r["job_id"]):
            fam = row["family_id"]
            pool = self.pool[(split, fam)]
            src = rng.choice(pool)   # uniform with replacement, 保持池顺序
            preds = [x for x in row["predecessors"].split("|") if x.strip()]
            jobs.append(Job(
                job_id=row["job_id"], family=fam,
                release_s=float(row["release_s"]), due_s=float(row["due_s"]),
                weight=float(row["priority_weight"]), zone=row["zone_id"],
                predecessors=preds, d=src["d"],
                source_ref=src["unit_id"],
                intensity=self.intensity.get(src["unit_id"], {}),
            ))
        return jobs

def load_tasks():
    rows = list(csv.DictReader(open(TASKS, encoding="utf-8-sig")))
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    return by_case

def run_trial(case: str, seed: int, alpha: float, beta: float,
              sampler: SourceSampler, task_by_case, scheduler, split="validation"):
    cfg = CASES[case]
    jobs = sampler.sample_jobs(task_by_case[case], split, seed)
    tr = Trial(jobs, H=cfg["H"], B=cfg["B"], K=cfg["K"],
               alpha=alpha, beta=beta, seed=seed, case=case)
    # 把 alpha/beta 注入策略（映射条件）
    scheduler.p["alpha"], scheduler.p["beta"] = alpha, beta
    max_steps = 220
    for _ in range(max_steps):
        if tr.terminated:
            break
        action, jid, val = scheduler.decide(tr)
        if action == "review":
            r = tr.do_review(jid)
        elif action == "execute":
            r = tr.do_execute(jid, val)
        elif action == "wait":
            r = tr.do_wait(val)
        elif action == "stop":
            r = tr.do_stop()
        else:
            break
        if not r.ok and tr.terminated:
            break
    if not tr.terminated:
        tr.do_stop()
    return tr
