
from pathlib import Path
import csv
import json
import math

from scipy.stats import norm

from common import RESULTS, ROOT

SIGMA_FLOOR = 0.35
GOFIT = RESULTS / "q1_gofit.csv"

def _milliers(alpha):
    return norm.pdf(alpha) / (1.0 - norm.cdf(alpha))

class DurationModel:

    def __init__(self, params, frozen=True):

        self.params = {f: (float(m), max(float(s), SIGMA_FLOOR))
                       for f, (m, s) in params.items()}
        self.frozen = frozen

    @classmethod
    def from_gofit(cls, path=GOFIT):
        params = {}
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                params[r["family"]] = (float(r["lognormal_mu"]),
                                       float(r["lognormal_sigma"]))
        return cls(params, frozen=True)

    def quantile(self, family, q):
        mu, sigma = self.params[family]
        return math.exp(mu + sigma * norm.ppf(q))

    def mean(self, family):
        mu, sigma = self.params[family]
        return math.exp(mu + 0.5 * sigma ** 2)

    def sample(self, family, rng):
        mu, sigma = self.params[family]
        return rng.lognormal(mu, sigma)

    def updater(self):
        if self.frozen:
            raise RuntimeError(
                "DurationModel 已冻结：trial 内更新请用 trial 自己的 "
                "MixedLikelihoodUpdater（model.updater()），禁止改共享底账。")
        return MixedLikelihoodUpdater(self.params)

class MixedLikelihoodUpdater:

    def __init__(self, base_params):
        self._base = {f: (m, max(s, SIGMA_FLOOR)) for f, (m, s) in base_params.items()}
        self.reset()

    def reset(self):

        self.params = dict(self._base)
        self._exact = {f: [] for f in self._base}
        self._cens = {f: [] for f in self._base}
        self.n_obs = 0

    def add_completed(self, family, d):

        if d <= 0:
            raise ValueError("d 必须为正")
        self._exact[family].append(math.log(d))
        self.n_obs += 1

    def add_censored(self, family, thr):

        if thr <= 0:
            raise ValueError("删失阈值必须为正")
        self._cens[family].append(math.log(thr))
        self.n_obs += 1

    def fit(self, max_iter=100, tol=1e-6):

        if self.n_obs == 0:
            return 0
        it = 0
        for it in range(1, max_iter + 1):
            delta = 0.0
            for fam in self.params:
                mu, sigma = self.params[fam]
                zs, cs = self._exact[fam], self._cens[fam]
                n = len(zs) + len(cs)
                if n == 0:
                    continue

                ez = sum(zs)
                e2 = sum((z - mu) ** 2 for z in zs)
                for c in cs:
                    alpha = (c - mu) / sigma
                    lam = _milliers(alpha)
                    ez += mu + sigma * lam
                    e2 += sigma ** 2 * (1 + alpha * lam)

                mu_new = ez / n
                sigma_new = math.sqrt(e2 / n)
                sigma_new = max(sigma_new, SIGMA_FLOOR)
                delta = max(delta, abs(mu_new - mu), abs(sigma_new - sigma))
                self.params[fam] = (mu_new, sigma_new)
            if delta < tol:
                break
        return it

def _run_checks():
    import numpy as np

    checks = []

    def rec(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    model = DurationModel.from_gofit()

    raw = {r["family"]: (float(r["lognormal_mu"]), float(r["lognormal_sigma"]))
           for r in csv.DictReader(open(GOFIT, encoding="utf-8-sig"))}
    ok, detail = True, []
    for fam, (m_ref, s_ref) in raw.items():
        m, s = model.params[fam]
        m_ok = abs(m - m_ref) <= 0.01

        s_ok = (abs(s - s_ref) <= 1e-9) if s_ref >= SIGMA_FLOOR + 0.01 \
            else (s >= SIGMA_FLOOR - 0.01)
        ok &= m_ok and s_ok
        detail.append(f"{fam}: mu {m:.3f} vs {m_ref:.3f} {'OK' if m_ok else 'FAIL'}; "
                      f"sigma {s:.3f} vs {s_ref:.3f} {'OK' if s_ok else 'FAIL'}")
    rec("1 参数与 q1_gofit 一致", ok, "; ".join(detail))

    ok = True
    for fam in model.params:
        mu, _ = model.params[fam]
        q25, q50, q75 = (model.quantile(fam, q) for q in (0.25, 0.5, 0.75))
        grid = [model.quantile(fam, q / 100) for q in range(1, 100)]
        mono = all(grid[i] <= grid[i + 1] + 1e-12 for i in range(len(grid) - 1))
        ok &= abs(q50 - math.exp(mu)) < 1e-9 and q25 <= q50 <= q75 and mono
    rec("2 分位数自洽", ok, "中位数=exp(mu)、单调、q25<=q50<=q75，全族通过" if ok else "见参数")

    rng = np.random.default_rng(20261123)
    MU_T, SIG_T, N = 1.588, 0.76, 124
    ds = rng.lognormal(MU_T, SIG_T, N).tolist()
    thr = sorted(ds)[N // 2]
    zs = [math.log(d) for d in ds if d <= thr]
    cs = [math.log(thr)] * (N - len(zs))

    from scipy.optimize import minimize

    def mixed_nll(theta):
        mu, ls = theta
        sig = math.exp(ls)
        nll = sum(0.5 * ((z - mu) / sig) ** 2 + ls + z for z in zs)
        nll += sum(-math.log(1.0 - norm.cdf((c - mu) / sig)) for c in cs)
        return nll

    mle = minimize(mixed_nll, [1.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-8, fatol=1e-10))
    mu_mle, sig_mle = mle.x[0], math.exp(mle.x[1])

    up = MixedLikelihoodUpdater({"P": (MU_T, SIG_T)})
    for z in zs:
        up.add_completed("P", math.exp(z))
    for c in cs:
        up.add_censored("P", math.exp(c))
    it = up.fit()
    mu_hat, sig_hat = up.params["P"]
    em_ok = abs(mu_hat - mu_mle) < 1e-3 and abs(sig_hat - sig_mle) < 1e-3

    meds = []
    for rep in range(100):
        r2 = np.random.default_rng(90000 + rep)
        d2 = r2.lognormal(MU_T, SIG_T, N).tolist()
        t2 = sorted(d2)[N // 2]
        u2 = MixedLikelihoodUpdater({"P": (MU_T, SIG_T)})
        for d in d2:
            if d <= t2:
                u2.add_completed("P", d)
            else:
                u2.add_censored("P", t2)
        u2.fit()
        meds.append(u2.params["P"][0])
    meds.sort()
    med_mu = meds[50]
    mc_ok = abs(med_mu - MU_T) < 0.05
    ok = em_ok and mc_ok and it <= 30
    rec("3 EM 正确且收敛", ok,
        f"EM({mu_hat:.4f},{sig_hat:.4f}) vs 数值MLE({mu_mle:.4f},{sig_mle:.4f}) 差<1e-3；"
        f"100 重复中位 mu={med_mu:.4f} vs 真值 {MU_T}；{it} 步收敛")

    up2 = MixedLikelihoodUpdater({"P": (MU_T, SIG_T)})
    for d in ds[:60]:
        up2.add_completed("P", d)
    up2.fit()
    mu_dirty, _ = up2.params["P"]
    up2.reset()
    mu_back = up2.params["P"][0]
    ok = (mu_back == MU_T and mu_dirty != MU_T and up2.n_obs == 0
          and up2._exact["P"] == [] and up2._cens["P"] == [])
    rec("4 reset 防泄漏", ok,
        f"污染后 mu={mu_dirty:.3f}，reset 后回到基线 {mu_back:.3f}、缓存清零")

    comp = [d for d in ds if d <= thr]
    mu_comp_only = sum(math.log(d) for d in comp) / len(comp)
    MU_BASE_WRONG, SIG_BASE_WRONG = 1.0, 0.5
    up3 = MixedLikelihoodUpdater({"P": (MU_BASE_WRONG, SIG_BASE_WRONG)})
    for d in comp:
        up3.add_completed("P", d)
    for _ in ds:
        if _ > thr:
            up3.add_censored("P", thr)
    up3.fit()
    mu_mixed = up3.params["P"][0]
    err_mix = abs(mu_mixed - MU_T)
    err_comp = abs(mu_comp_only - MU_T)
    err_base = abs(MU_BASE_WRONG - MU_T)
    ok = err_mix < err_comp and err_mix < err_base
    rec("5 双信息通道", ok,
        f"|估计-真值|：混合 {err_mix:.3f} vs 仅完成子集 {err_comp:.3f} vs 不更新 {err_base:.3f}；"
        f"删失通道把 mu 从 {mu_comp_only:.3f} 拉回 {mu_mixed:.3f}（真值 {MU_T}）")

    try:
        model.updater()
        rec("6 冻结拒绝更新", False, "updater() 未抛错！")
    except RuntimeError:
        rec("6 冻结拒绝更新", True, "frozen 模型调 updater() 正确抛 RuntimeError")

    out = dict(step="q5 时长模型+混合似然更新器",
               sigma_floor=SIGMA_FLOOR,
               params={f: list(v) for f, v in model.params.items()},
               checks=checks,
               all_passed=all(c["passed"] for c in checks))
    with open(RESULTS / "q5_model_checks.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["name"])
    print("ALL PASS" if out["all_passed"] else "SOME FAILED")
    return out["all_passed"]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    _run_checks()
