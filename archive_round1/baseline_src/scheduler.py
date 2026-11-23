"""调度策略 v2：滚动时域启发式 + 可配置预留/审查/排序规则。

关键洞察（题目规则推导）：
- 完成时刻 C = s + p，与预留 b 无关。
- 设备下次可用 = s + max(p,b)。当 b<=p 时占用为 s+p，b 再小也不缩短。
- 因此 b 取最小值 0.2 可最小化费用(0.1*b)与 reserved_idle，
  且不改变完成/占用时间 —— 对以完成率/准时率为目标的策略是最优的。
- b>p 只制造 reserved_idle 并增加费用，仅在需要压低 observed_overrun 时才有意义。

策略通过 reserve_mode / review_mode / order_rule 三个旋钮搜索。
"""
from typing import Tuple
import numpy as np
from simulator import zone_dist

class Scheduler:
    def __init__(self, estimator, params: dict):
        self.est = estimator
        self.p = params

    def _est_job(self, tr, jid):
        job = tr.jobs[jid]
        lvl = tr.prefix_level[jid]
        inten = job.intensity.get(lvl)
        if inten is None:
            inten = self.est.params.get(job.family, {}).get("z_mean", 0.5)
        return self.est.predict(job.family, inten, lvl)

    def _priority(self, tr, jid, est) -> float:
        job = tr.jobs[jid]
        D = zone_dist(tr.zone, job.zone)
        p_hat = self.p["alpha"] * est["d_mean"] + self.p["beta"]
        travel = D + 0.5
        slack = job.due_s - tr.t - travel - p_hat
        rule = self.p.get("order_rule", "combined")
        if rule == "edd":
            return -job.due_s
        if rule == "wspt":
            return job.weight / max(p_hat + travel, 0.5)
        if rule == "slack":
            return -slack
        if rule == "weight":
            return job.weight
        # combined: 权重、紧迫度、时长效率、距离、解锁价值
        urg = np.exp(-slack / max(job.due_s, 20.0) * 3.0)
        dur = 1.0 / (1.0 + p_hat)
        dist = 1.0 / (1.0 + D)
        unlock = sum(1 for j in tr.jobs.values() if jid in j.predecessors)
        return (self.p["a_w"] * job.weight + self.p["a_u"] * urg +
                self.p["a_d"] * dur + self.p["a_dist"] * dist +
                self.p["a_unlock"] * unlock)

    def _reserve(self, tr, jid, est) -> float:
        mode = self.p.get("reserve_mode", "min")
        D = zone_dist(tr.zone, tr.jobs[jid].zone)
        max_b = tr.H - (tr.t + D + 0.5)
        if max_b <= 0.2:
            return 0.2
        if mode == "min":
            return 0.2
        p_hat = self.p["alpha"] * est["d_mean"] + self.p["beta"]
        if mode == "median":
            b = p_hat
        elif mode == "q75":
            b = self.p["alpha"] * est["d_q75"] + self.p["beta"]
        elif mode == "q90":
            b = self.p["alpha"] * est["d_q90"] + self.p["beta"]
        else:  # risk blend
            b = self.p["alpha"] * (est["d_q75"] + self.p["risk"] *
                 (est["d_q90"] - est["d_q75"])) + self.p["beta"]
        return float(np.clip(b, 0.2, max_b))

    def _worth_review(self, tr, jid, est) -> bool:
        mode = self.p.get("review_mode", "none")
        if mode == "none":
            return False
        if tr.reviews >= tr.K or tr.review_count[jid] >= 2:
            return False
        if tr.cost + 3.0 > tr.B:
            return False
        if mode == "all":
            return True
        # selective: 不确定性大且重要，且为前序解锁
        spread = (est["d_q90"] - est["d_q10"]) / max(est["d_mean"], 1.0)
        unlock = any(jid in j.predecessors for j in tr.jobs.values())
        return (spread > self.p.get("review_spread", 0.8) and
                (tr.jobs[jid].weight >= 2 or unlock))

    def decide(self, tr) -> Tuple[str, object, object]:
        # 1) 审查
        for jid in tr.order:
            if jid not in tr.executed and tr.released(jid):
                est = self._est_job(tr, jid)
                if self._worth_review(tr, jid, est):
                    return ("review", jid, None)
        # 2) 可执行作业
        cands = []
        for jid in tr.order:
            if jid in tr.executed or not tr.released(jid) or not tr.preds_done(jid):
                continue
            est = self._est_job(tr, jid)
            D = zone_dist(tr.zone, tr.jobs[jid].zone)
            s = tr.t + D + 0.5
            b = self._reserve(tr, jid, est)
            fee = 2 + 0.1 * b + 0.2 * D
            if tr.cost + fee > tr.B + 1e-9 or s + b > tr.H + 1e-9:
                continue
            cands.append((self._priority(tr, jid, est), jid, b))
        if cands:
            cands.sort(reverse=True)
            return ("execute", cands[0][1], cands[0][2])
        # 3) 等待
        future = [j for j in tr.order if j not in tr.executed]
        if future:
            nxt = min(tr.jobs[j].release_s for j in future)
            if tr.t < nxt <= tr.H:
                return ("wait", None, min(nxt - tr.t, tr.H - tr.t))
        return ("stop", None, None)

def default_params():
    return dict(a_w=1.0, a_u=1.2, a_d=0.5, a_dist=0.3, a_unlock=0.8,
    order_rule="wspt", review_spread=0.8,
    reserve_mode="min", review_mode="none")
