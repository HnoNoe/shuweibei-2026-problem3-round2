"""确定性仿真器：严格按照 Problem 3 Requirements 的协议模拟单个 trial。

关键规则（摘自题目）：
- 单设备；执行/审查/等待不可重叠；等待免费且必须为正、不跨越 H。
- 执行：预留 b>=0.2；服务开始 s=t+D+0.5；成本=2+0.1*b+0.2*D；要求累计成本<=B 且 s+b<=H。
- 完成 C=s+p (p=alpha*d+beta)；若 C<=H 完成，设备下次可用 s+max(p,b)；否则 H 处截断(censored)。
- 审查：成本1、占用1秒、延迟1秒返回；前缀 0.2->0.6->1.0；每 job 最多2次；受 K,B 约束。
- 目标：最大化加权准时率（主），满足 on-time>=0.5；记录全部公共指标与过程日志。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np

ZONES = {"Z0": (0, 0), "Z1": (1, 0), "Z2": (0, 1), "Z3": (1, 1), "Z4": (2, 0)}

def zone_dist(a: str, b: str) -> float:
    xa, ya = ZONES[a]; xb, yb = ZONES[b]
    return math.hypot(xa - xb, ya - yb)

@dataclass
class Job:
    job_id: str
    family: str
    release_s: float
    due_s: float
    weight: float
    zone: str
    predecessors: List[str]
    d: float                    # 原始标注时长（真值，仅用于计算 p 和日志）
    source_ref: str = ""
    intensity: dict = field(default_factory=dict)  # 前缀强度 {0.2:..,0.6:..,1.0:..}，策略可见

@dataclass
class ActionResult:
    ok: bool
    reason: str = ""

class Trial:
    """单个 trial 的状态机。策略通过 step() 提交动作，仿真器强制合规。"""
    def __init__(self, jobs: List[Job], H: float, B: float, K: float,
                 alpha: float, beta: float, seed: int, case: str):
        self.jobs = {j.job_id: j for j in jobs}
        self.order = sorted(self.jobs.keys())
        self.H, self.B, self.K = H, B, K
        self.alpha, self.beta = alpha, beta
        self.seed, self.case = seed, case
        # 状态
        self.t = 0.0
        self.zone = "Z0"
        self.cost = 0.0
        self.reviews = 0
        self.prefix_level = {j: 0.2 for j in self.order}   # 当前可见累计前缀
        self.review_count = {j: 0 for j in self.order}
        self.completed: set = set()
        self.executed: set = set()
        self.pending_review: Dict[str, float] = {}   # job -> 返回时间
        self.n_actions = 0
        self.violations = 0
        self.terminated = False
        self.stopped = False
        # 记录
        self.completion_time: Dict[str, float] = {}
        self.log: List[dict] = []
        self.trial_rows: List[dict] = []

    @property
    def W(self):
        return sum(j.weight for j in self.jobs.values())

    def p_of(self, jid: str) -> float:
        return self.alpha * self.jobs[jid].d + self.beta

    def released(self, jid: str) -> bool:
        return self.t >= self.jobs[jid].release_s - 1e-9

    def preds_done(self, jid: str) -> bool:
        return all(p in self.completed for p in self.jobs[jid].predecessors)

    def available_jobs(self) -> List[str]:
        return [j for j in self.order
                if j not in self.executed and self.released(j)
                and self.prefix_level[j] >= 0.2]

    def do_review(self, jid: str) -> ActionResult:
        if self.terminated:
            return ActionResult(False, "terminated")
        if jid not in self.jobs:
            return self._violate("unknown job")
        if jid in self.executed:
            return self._violate("job already attempted")
        if not self.released(jid):
            return self._violate("job not released")
        if self.review_count[jid] >= 2:
            return self._violate("review limit per job")
        if self.reviews >= self.K:
            return self._violate("review limit per case")
        if self.cost + 1.0 > self.B + 1e-9:
            return self._violate("budget exceeded")
        ret = self.t + 1.0
        if ret > self.H + 1e-9:
            return self._violate("review crosses H")
        self.n_actions += 1
        if self.n_actions > 200:
            return self._violate("too many actions")
        # 提交
        self.cost += 1.0
        self.reviews += 1
        self.review_count[jid] += 1
        old = self.prefix_level[jid]
        new = 0.6 if old < 0.6 else 1.0
        self.pending_review[jid] = ret
        self._log(dict(time=self.t, action="review", job=jid,
                       prefix_before=old, prefix_after=new, return_time=ret,
                       cost=self.cost, zone=self.zone))
        # 时间推进 1 秒（审查占用设备）
        self.t = ret
        # 返回信息
        self.prefix_level[jid] = new
        del self.pending_review[jid]
        return ActionResult(True)

    def do_wait(self, w: float) -> ActionResult:
        if self.terminated:
            return ActionResult(False, "terminated")
        if not (w > 0 and math.isfinite(w)):
            return self._violate("wait must be positive finite")
        if self.t + w > self.H + 1e-9:
            return self._violate("wait crosses H")
        self.n_actions += 1
        if self.n_actions > 200:
            return self._violate("too many actions")
        self._log(dict(time=self.t, action="wait", wait=w,
                       return_time=self.t + w, cost=self.cost, zone=self.zone))
        self.t += w
        return ActionResult(True)

    def do_execute(self, jid: str, b: float) -> ActionResult:
        if self.terminated:
            return ActionResult(False, "terminated")
        if jid not in self.jobs:
            return self._violate("unknown job")
        if jid in self.executed:
            return self._violate("repeated job")
        if not self.released(jid):
            return self._violate("job not released")
        if not self.preds_done(jid):
            return self._violate("predecessor unmet")
        if not (math.isfinite(b) and b >= 0.2 - 1e-9):
            return self._violate("insufficient reservation")
        D = zone_dist(self.zone, self.jobs[jid].zone)
        s = self.t + D + 0.5
        fee = 2.0 + 0.1 * b + 0.2 * D
        if self.cost + fee > self.B + 1e-9:
            return self._violate("budget exceeded")
        if s + b > self.H + 1e-9:
            return self._violate("planned action crosses H")
        self.n_actions += 1
        if self.n_actions > 200:
            return self._violate("too many actions")
        p = self.p_of(jid)
        C = s + p
        fee_paid = fee
        self.cost += fee_paid
        self.executed.add(jid)
        # overrun 对所有已执行作业计算（题目定义），idle 仅对完成作业计算
        overrun = max(min(p, self.H - s) - b, 0.0)
        idle = 0.0; censored = False
        if C <= self.H + 1e-9:
            self.completed.add(jid)
            self.completion_time[jid] = C
            idle = max(b - p, 0.0)
            next_avail = s + max(p, b)
        else:
            censored = True
            next_avail = self.H
        self._log(dict(time=self.t, action="execute", job=jid, reserve=b,
                       service_start=s, completion=min(C, self.H),
                       p=p, overrun=overrun, idle=idle, censored=censored,
                       fee=fee_paid, cost=self.cost, zone=self.jobs[jid].zone,
                       source=self.jobs[jid].source_ref, next_available=next_avail))
        self.trial_rows.append(dict(job=jid, completed=not censored,
                                    C=min(C, self.H), p=p, b=b, censored=censored,
                                    overrun=overrun, idle=idle))
        # 时间/位置推进
        self.t = min(next_avail, self.H)
        self.zone = self.jobs[jid].zone
        if self.t >= self.H - 1e-9:
            self.terminated = True
        return ActionResult(True)

    def do_stop(self) -> ActionResult:
        self.stopped = True
        self.terminated = True
        self._log(dict(time=self.t, action="stop", cost=self.cost, zone=self.zone))
        return ActionResult(True)

    def _log(self, d):
        self.log.append(d)

    def _violate(self, reason):
        self.violations += 1
        self.terminated = True
        self._log(dict(time=self.t, action="INVALID", reason=reason,
                       cost=self.cost, zone=self.zone))
        return ActionResult(False, reason)

    def metrics(self) -> dict:
        W = self.W
        wc = wt = tard = 0.0
        for j in self.order:
            job = self.jobs[j]
            C = self.completion_time.get(j, self.H)
            if j in self.completed:
                wc += job.weight
                if C <= job.due_s + 1e-9:
                    wt += job.weight
            tard += job.weight * max(C - job.due_s, 0.0)
        overrun = sum(r["overrun"] for r in self.trial_rows)
        idle = sum(r["idle"] for r in self.trial_rows if r["completed"])
        censored = sum(1 for r in self.trial_rows if r["censored"])
        return dict(
            case=self.case, seed=self.seed, alpha=self.alpha, beta=self.beta,
            weighted_completion_rate=wc / W,
            weighted_on_time_rate=wt / W,
            weighted_tardiness_s=tard,
            cost=self.cost, observations=self.reviews,
            elapsed_s=self.t,
            observed_overrun_s=overrun,
            reserved_idle_s=idle,
            censored_units=censored,
            invalid_actions=self.violations,
            valid_protocol=1 if self.violations == 0 else 0,
            jobs_completed=len(self.completed),
        )
