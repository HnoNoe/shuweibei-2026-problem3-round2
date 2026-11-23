
import itertools
import math

import numpy as np

from common import CASES, CHANGEOVER, load_tasks, zone_dist
from duration_model3 import DurationModel
from strategy import EPS, FastSim, fee_of, B_MIN

def public_jobs(case):

    return {j["job_id"]: dict(family=j["family"], release_s=j["release_s"],
                              due_s=j["due_s"], weight=j["weight"],
                              zone=j["zone"], predecessors=list(j["predecessors"]))
            for j in load_tasks()[case]}

def feasible_permutations(jobs):

    ids = list(jobs)
    preds = {j: set(jobs[j]["predecessors"]) for j in ids}
    done, seq = set(), []

    def dfs():
        if len(seq) == len(ids):
            yield list(seq)
            return
        for j in ids:
            if j in done or not preds[j] <= done:
                continue
            seq.append(j)
            done.add(j)
            yield from dfs()
            done.remove(j)
            seq.pop()

    yield from dfs()

def eval_sequence(seq, jobs, durs, alpha, beta, H, B):

    snap0 = dict(t=0.0, zone="Z0", cost=0.0, H=H, B=B, alpha=alpha, beta=beta,
                 jobs=jobs, executed=set(), completed=set(), completion_time={})
    sim = FastSim(snap0, durs, alpha, beta, H, B)
    for jid in seq:
        if jid in sim.executed:
            continue
        info = jobs[jid]
        if sim.t + EPS < info["release_s"]:
            sim.wait_to(info["release_s"])
        if sim.t >= sim.H - EPS:
            break
        if all(p in sim.completed for p in info["predecessors"]):
            sim.exec_ok(jid)
    return sim.weighted_on_time()

def sample_scenarios(jobs, params, N, rng):

    out = []
    for _ in range(N):
        out.append({j: float(np.exp(rng.normal(*params[info["family"]])))
                    for j, info in jobs.items()})
    return out

def local_search(seq0, jobs, scen, alpha, beta, H, B, max_steps=40):

    def ev(seq):
        return sum(eval_sequence(seq, jobs, d, alpha, beta, H, B) for d in scen) / len(scen)

    def neighbors(seq):
        for i in range(len(seq) - 1):
            s2 = seq[:]
            s2[i], s2[i + 1] = s2[i + 1], s2[i]
            yield s2
        for i in range(1, len(seq)):
            for j in range(0, i - 1):
                s2 = seq[:]
                x = s2.pop(i)
                s2.insert(j, x)
                yield s2

    best, best_ev = list(seq0), ev(seq0)
    for _ in range(max_steps):
        improved = False
        for cand in neighbors(best):
            e = ev(cand)
            if e > best_ev + 1e-12:
                best, best_ev = cand, e
                improved = True
                break
        if not improved:
            break
    return best, best_ev

def atc_scan_init(jobs, model, alpha, beta, H, B, kappa=2.0):

    snap0 = dict(t=0.0, zone="Z0", cost=0.0, H=H, B=B, alpha=alpha, beta=beta,
                 jobs=jobs, executed=set(), completed=set(), completion_time={})
    mean_d = {f: math.exp(mu + 0.5 * sig ** 2) for f, (mu, sig) in model.params.items()}
    dur = {j: alpha * mean_d[info["family"]] + beta for j, info in jobs.items()}
    sim = FastSim(snap0, dur, alpha, beta, H, B)
    seq = []
    guard = 400
    while guard > 0:
        guard -= 1
        cands = []
        for jid, info in jobs.items():
            if jid in sim.executed:
                continue
            if sim.t + EPS < info["release_s"]:
                continue
            if any(p not in sim.completed for p in info["predecessors"]):
                continue
            D = zone_dist(sim.zone, info["zone"])
            if sim.cost + fee_of(D) > sim.B + EPS:
                continue
            if sim.t + D + CHANGEOVER + B_MIN > sim.H + EPS:
                continue
            cands.append(jid)
        if not cands:
            future = [i["release_s"] for j, i in jobs.items() if j not in sim.executed]
            if not future:
                break
            sim.wait_to(min(future))
            if sim.t >= sim.H - EPS:
                break
            continue
        def score(j):
            info = jobs[j]
            D = zone_dist(sim.zone, info["zone"])
            s = sim.t + D + CHANGEOVER
            p = dur[j]
            rest = [x for x in jobs if x not in sim.executed]
            p_bar = sum(dur[x] for x in rest) / len(rest) if rest else p
            slack = info["due_s"] - (s + p)
            return (info["weight"] / max(p, EPS)) * math.exp(-max(slack, 0.0) / (kappa * max(p_bar, EPS)))
        best = max(cands, key=score)
        if not sim.exec_ok(best):
            break
        seq.append(best)
    return seq

def saa_gap(case, alpha=1.0, beta=0.5, candidate_seq=None, M=10, N=20,
            n_eval=5000, model=None, seed0=777000, ls_solver=True):

    model = model or DurationModel.from_gofit()
    cfg = CASES[case]
    H, B = cfg["H"], cfg["B"]
    jobs = public_jobs(case)
    params = model.params

    exact = (case == "S1")
    batch_vals, batch_seqs = [], []
    for m in range(M):
        rng = np.random.default_rng(seed0 + m)
        scen = sample_scenarios(jobs, params, N, rng)
        if exact:
            best_v, best_seq = -1.0, None
            for seq in feasible_permutations(jobs):
                v = sum(eval_sequence(seq, jobs, d, alpha, beta, H, B)
                        for d in scen) / N
                if v > best_v + 1e-12:
                    best_v, best_seq = v, seq
        else:
            seq0 = atc_scan_init(jobs, model, alpha, beta, H, B)
            if ls_solver:
                best_seq, best_v = local_search(seq0, jobs, scen, alpha, beta, H, B)
            else:
                best_seq = seq0
                best_v = sum(eval_sequence(seq0, jobs, d, alpha, beta, H, B)
                             for d in scen) / N
        batch_vals.append(best_v)
        batch_seqs.append(best_seq)

    lb = float(np.mean(batch_vals))
    lb_sd = float(np.std(batch_vals, ddof=1)) if M > 1 else 0.0
    lb_ci = 1.96 * lb_sd / math.sqrt(M) if M > 1 else 0.0

    if candidate_seq is None:
        candidate_seq = atc_scan_init(jobs, model, alpha, beta, H, B)
    rng_eval = np.random.default_rng(seed0 + 999999)
    scen_eval = sample_scenarios(jobs, params, n_eval, rng_eval)
    ub = sum(eval_sequence(candidate_seq, jobs, d, alpha, beta, H, B)
             for d in scen_eval) / n_eval

    return dict(case=case, alpha=alpha, beta=beta,
                exact_lower_bound=exact, M=M, N=N, n_eval=n_eval,
                lb=lb, lb_ci95=lb_ci, ub=ub,
                gap=max(ub - lb, 0.0), gap_rel=max((ub - lb) / max(ub, EPS), 0.0),
                batch_vals=batch_vals,
                candidate_seq=list(candidate_seq),
                batch_seqs=[list(s) for s in batch_seqs])

if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, __file__.rsplit("\\", 1)[0])
    r = saa_gap("S1", M=5, N=10, n_eval=2000)
    print(json.dumps({k: v for k, v in r.items() if k not in
                      ("batch_seqs", "candidate_seq")}, indent=2))
    assert r["gap"] >= -1e-9 and r["gap_rel"] < 0.3, "SAA 间隙冒烟失败"
    print("SAA GAP SMOKE OK: gap_rel=%.4f (exact LB=%s)" %
          (r["gap_rel"], r["exact_lower_bound"]))
