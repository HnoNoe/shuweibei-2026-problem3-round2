
import math
from common import (ZONES, FEE_FIXED, FEE_RESERVE_PER_S, FEE_DIST_PER_UNIT,
                    CHANGEOVER, REVIEW_COST, REVIEW_OCCUPY, REVIEW_DELAY,
                    REVIEW_MAX_PER_JOB, PREFIXES, B_MIN, MAX_ACTIONS,
                    zone_dist)

class Job:
    __slots__ = ("job_id", "family", "release_s", "due_s", "weight", "zone",
                 "predecessors", "d", "source_ref", "prefix_feat")

    def __init__(self, job_id, family, release_s, due_s, weight, zone,
                 predecessors, d, source_ref, prefix_feat=None):
        self.job_id = job_id
        self.family = family
        self.release_s = release_s
        self.due_s = due_s
        self.weight = weight
        self.zone = zone
        self.predecessors = list(predecessors)
        self.d = d
        self.source_ref = source_ref
        self.prefix_feat = prefix_feat or {lvl: None for lvl in PREFIXES}

class Trial:
    def __init__(self, jobs, H, B, K, alpha, beta, seed, case):
        self.jobs = {j.job_id: j for j in jobs}
        self.order = sorted(self.jobs)
        self.H, self.B, self.K = H, B, K
        self.alpha, self.beta = alpha, beta
        self.seed, self.case = seed, case

        self.t = 0.0
        self.zone = "Z0"
        self.cost = 0.0
        self.reviews = 0
        self.prefix_level = {j: PREFIXES[0] for j in self.order}
        self.review_count = {j: 0 for j in self.order}
        self.completed = set()
        self.executed = set()
        self.completion_time = {}
        self.n_actions = 0
        self.violations = 0
        self.terminated = False
        self.stopped = False
        self._rows = []
        self.log = []

    def snapshot(self):
        return dict(t=self.t, zone=self.zone, cost=self.cost,
                    reviews=self.reviews, H=self.H, B=self.B, K=self.K,
                    alpha=self.alpha, beta=self.beta,
                    jobs={j: dict(family=self.jobs[j].family,
                                  release_s=self.jobs[j].release_s,
                                  due_s=self.jobs[j].due_s,
                                  weight=self.jobs[j].weight,
                                  zone=self.jobs[j].zone,
                                  predecessors=list(self.jobs[j].predecessors))
                          for j in self.order},
                    executed=set(self.executed), completed=set(self.completed),
                    completion_time=dict(self.completion_time),
                    prefix_level=dict(self.prefix_level),
                    review_count=dict(self.review_count),

                    prefix_feat={j: {lvl: f for lvl, f in
                                     self.jobs[j].prefix_feat.items()
                                     if f is not None and lvl <= self.prefix_level[j]}
                                 for j in self.order})

    def _released(self, jid):
        return self.t >= self.jobs[jid].release_s - 1e-9

    def _preds_done(self, jid):
        return all(p in self.completed for p in self.jobs[jid].predecessors)

    def _p(self, jid):
        return self.alpha * self.jobs[jid].d + self.beta

    def _violate(self, reason):
        self.violations += 1
        self.terminated = True
        self.log.append(dict(time=self.t, action="INVALID", reason=reason,
                             cost=self.cost, zone=self.zone))
        return False, reason

    def _count_action(self):
        self.n_actions += 1
        return self.n_actions <= MAX_ACTIONS

    def execute(self, jid, b):
        if self.terminated:
            return False, "terminated"
        if jid not in self.jobs:
            return self._violate("unknown job")
        if jid in self.executed:
            return self._violate("repeated job")
        if not self._released(jid):
            return self._violate("job not released")
        if not self._preds_done(jid):
            return self._violate("predecessor unmet")
        if not (math.isfinite(b) and b >= B_MIN - 1e-9):
            return self._violate("insufficient reservation")
        D = zone_dist(self.zone, self.jobs[jid].zone)
        s = self.t + D + CHANGEOVER
        fee = FEE_FIXED + FEE_RESERVE_PER_S * b + FEE_DIST_PER_UNIT * D
        if self.cost + fee > self.B + 1e-9:
            return self._violate("budget exceeded")
        if s + b > self.H + 1e-9:
            return self._violate("planned action crosses H")
        if not self._count_action():
            return self._violate("too many actions")
        p = self._p(jid)
        C = s + p
        self.cost += fee
        self.executed.add(jid)
        overrun = max(min(p, self.H - s) - b, 0.0)
        idle, censored = 0.0, False
        if C <= self.H + 1e-9:
            self.completed.add(jid)
            self.completion_time[jid] = C
            idle = max(b - p, 0.0)
            next_avail = s + max(p, b)
        else:
            censored = True
            next_avail = self.H
        self._rows.append(dict(job=jid, completed=not censored,
                               C=min(C, self.H), p=p, b=b,
                               overrun=overrun, idle=idle))
        self.log.append(dict(time=self.t, action="execute", job=jid, reserve=b,
                             info_level=self.prefix_level[jid],
                             travel_D=D, service_start=s,
                             completion=min(C, self.H), p=p,
                             overrun=overrun, idle=idle, censored=censored,
                             fee=fee, cost=self.cost,
                             zone_from=self.zone, zone_to=self.jobs[jid].zone,
                             next_available=min(next_avail, self.H),
                             source_ref=self.jobs[jid].source_ref))
        self.t = min(next_avail, self.H)
        self.zone = self.jobs[jid].zone
        if self.t >= self.H - 1e-9:
            self.terminated = True
        return True, "ok"

    def review(self, jid):
        if self.terminated:
            return False, "terminated"
        if jid not in self.jobs:
            return self._violate("unknown job")
        if jid in self.executed:
            return self._violate("job already attempted")
        if not self._released(jid):
            return self._violate("job not released")
        if self.review_count[jid] >= REVIEW_MAX_PER_JOB:
            return self._violate("review limit per job")
        if self.reviews >= self.K:
            return self._violate("review limit per case")
        if self.cost + REVIEW_COST > self.B + 1e-9:
            return self._violate("budget exceeded")
        ret = self.t + REVIEW_DELAY
        if ret > self.H + 1e-9:
            return self._violate("review crosses H")
        if not self._count_action():
            return self._violate("too many actions")
        old = self.prefix_level[jid]
        new = PREFIXES[1] if old < PREFIXES[1] else PREFIXES[2]
        self.cost += REVIEW_COST
        self.reviews += 1
        self.review_count[jid] += 1
        self.t = ret
        self.prefix_level[jid] = new
        self.log.append(dict(time=ret - REVIEW_OCCUPY, action="review", job=jid,
                             info_level=old, info_after=new, return_time=ret,
                             fee=REVIEW_COST, cost=self.cost, zone=self.zone))
        return True, "ok"

    def wait(self, w):
        if self.terminated:
            return False, "terminated"
        if not (math.isfinite(w) and w > 0):
            return self._violate("wait must be positive finite")
        if self.t + w > self.H + 1e-9:
            return self._violate("wait crosses H")
        if not self._count_action():
            return self._violate("too many actions")
        self.log.append(dict(time=self.t, action="wait", wait=w,
                             return_time=self.t + w, cost=self.cost,
                             zone=self.zone))
        self.t += w
        return True, "ok"

    def stop(self):
        self.stopped = True
        self.terminated = True
        self.log.append(dict(time=self.t, action="stop", cost=self.cost,
                             zone=self.zone))
        return True, "ok"

    def metrics(self):
        W = sum(self.jobs[j].weight for j in self.order)
        wc = wt = tard = 0.0
        for j in self.order:
            job = self.jobs[j]
            C = self.completion_time.get(j, self.H)
            if j in self.completed:
                wc += job.weight
                if C <= job.due_s + 1e-9:
                    wt += job.weight
            tard += job.weight * max(C - job.due_s, 0.0)
        return dict(
            case=self.case, seed=self.seed,
            alpha=self.alpha, beta=self.beta,
            weighted_completion_rate=wc / W,
            weighted_on_time_rate=wt / W,
            weighted_tardiness_s=tard,
            cost=self.cost, observations=self.reviews,
            elapsed_s=self.t,
            observed_overrun_s=sum(r["overrun"] for r in self._rows),
            reserved_idle_s=sum(r["idle"] for r in self._rows if r["completed"]),
            censored_units=sum(1 for r in self._rows if not r["completed"]),
            invalid_actions=self.violations,
            valid_protocol=1 if self.violations == 0 else 0,
        )
