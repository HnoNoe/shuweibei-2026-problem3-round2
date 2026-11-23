
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import ROOT, RESULTS

FEAT = ROOT / "prefix_features.csv"
META = {"unit_id", "dataset", "sample", "environment_id", "split",
        "family_id", "source_segment_index", "start_s", "end_s",
        "duration_s", "available"}

def load():
    df = pd.read_csv(FEAT)
    feat_cols = [c for c in df.columns if c not in META]
    return df, feat_cols

def q1_stats(df):
    tr = df[df.split == "train"]
    print("=" * 72)
    print("Q1a 分族时长统计（train, 秒）")
    g = tr.groupby("family_id")["duration_s"]
    stats = g.agg(["count", "mean", "std", "median", "min", "max"]).round(2)
    print(stats.to_string())
    stats.to_csv(RESULTS / "q1_family_stats.csv", encoding="utf-8-sig")
    ly = np.log(tr["duration_s"].values)
    from scipy import stats as st
    print(f"\nlog d 整体: mean={ly.mean():.3f} std={ly.std():.3f} "
          f"skew={st.skew(ly):.3f} kurt={st.kurtosis(ly):.3f}")
    for fam, sub in tr.groupby("family_id"):
        l = np.log(sub["duration_s"].values)
        print(f"  {fam}: log d mean={l.mean():.3f} std={l.std():.3f} skew={st.skew(l):.3f}")
    groups = [np.log(s["duration_s"].values) for _, s in tr.groupby("family_id")]
    f, p = st.f_oneway(*groups)
    print(f"\n族间 log d 均值差异 ANOVA: F={f:.2f}, p={p:.4f} "
          f"-> {'分层显著成立' if p < 0.05 else '分层不显著'}")
    return stats

def q1_gofit(df):

    from scipy import stats as st
    tr = df[df.split == "train"]
    print("=" * 72)
    print("Q1b 拟合优度: lognormal vs Weibull（按族 AIC，越小越好）")
    rows = []
    for fam, sub in tr.groupby("family_id"):
        y = sub["duration_s"].values

        lly = np.log(y)
        mu, sig = lly.mean(), lly.std()
        ll_ln = float(np.sum(st.norm.logpdf(lly, mu, sig)))
        k_ln = 2
        aic_ln = 2 * k_ln - 2 * ll_ln

        c, loc, scale = st.weibull_min.fit(y, floc=0)
        ll_wb = float(np.sum(st.weibull_min.logpdf(y, c, loc, scale)))
        aic_wb = 2 * 2 - 2 * ll_wb
        rows.append(dict(family=fam, n=len(y),
                         lognormal_mu=round(mu, 3), lognormal_sigma=round(sig, 3),
                         weibull_k=round(c, 3), weibull_lambda=round(scale, 3),
                         AIC_lognormal=round(aic_ln, 1), AIC_weibull=round(aic_wb, 1),
                         winner="lognormal" if aic_ln < aic_wb else "weibull"))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    r.to_csv(RESULTS / "q1_gofit.csv", index=False, encoding="utf-8-sig")
    return r

def censor_study(df, reps=500, seed=42):

    tr = df[(df.split == "train") & (df.family_id == "P")]
    TRUE_MU = float(np.log(tr["duration_s"].values).mean())
    TRUE_SIG = float(np.log(tr["duration_s"].values).std())

    def em_censored(lx, cens, thr, iters=500):

        from scipy import stats as st
        mu, s = lx[~cens].mean(), lx[~cens].std()
        for _ in range(iters):
            a = (thr - mu) / s
            lam = np.exp(st.norm.logpdf(a) - st.norm.logsf(a))
            ez = lx.copy()
            ez[cens] = mu + s * lam
            var = (lx - mu) ** 2
            var[cens] = s * s * (1 + a * lam)
            mu_n, s_n = ez.mean(), np.sqrt(var.mean())
            if abs(mu_n - mu) < 1e-12 and abs(s_n - s) < 1e-12:
                mu, s = mu_n, s_n
                break
            mu, s = mu_n, s_n
        return mu, s

    rng = np.random.default_rng(seed)
    print("=" * 72)
    print(f"截断对照（模拟真值 mu={TRUE_MU:.3f} sigma={TRUE_SIG:.3f}, "
          f"{reps} 次重复）")
    rows = []
    for n_sub in (60, 124):
        nb, cb = [], []
        for _ in range(reps):
            ly = rng.normal(TRUE_MU, TRUE_SIG, n_sub)
            thr = np.median(ly)
            cens = ly > thr
            nv = ly[~cens]
            nb.append([nv.mean(), nv.std(ddof=1)])
            cb.append(em_censored(ly, cens, thr))
        nb = np.array(nb) - [TRUE_MU, TRUE_SIG]
        cb = np.array(cb) - [TRUE_MU, TRUE_SIG]
        print(f"n={n_sub}: 朴素丢弃 mu {nb[:,0].mean():+.3f} σ {nb[:,1].mean():+.3f} | "
              f"EM含删失 mu {cb[:,0].mean():+.3f}±{cb[:,0].std():.3f} "
              f"σ {cb[:,1].mean():+.3f}±{cb[:,1].std():.3f}")
        rows.append(dict(n=n_sub,
                         naive_mu_bias=round(nb[:,0].mean(), 3),
                         naive_sigma_bias=round(nb[:,1].mean(), 3),
                         em_mu_bias=round(cb[:,0].mean(), 3),
                         em_mu_sd=round(cb[:,0].std(), 3),
                         em_sigma_bias=round(cb[:,1].mean(), 3),
                         em_sigma_sd=round(cb[:,1].std(), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "censor_study.csv", index=False, encoding="utf-8-sig")
    return out

def grouped_kfold(df, n_splits=5, seed=0):
    envs = df["environment_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(envs)
    folds = np.array_split(envs, n_splits)
    for i in range(n_splits):
        va_envs = set(folds[i])
        yield df[~df["environment_id"].isin(va_envs)], df[df["environment_id"].isin(va_envs)]

def q2_predictability(df, feat_cols):

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score, mean_absolute_error

    tr_all = df[df.split == "train"].reset_index(drop=True)
    print("=" * 72)
    print("Q2 前缀特征预测总时长（按环境分组 5 折 CV, train split）")
    results = []
    for pre in ["p02", "p06", "p10"]:
        cols = [c for c in feat_cols if c.startswith(pre + "_")]
        for name, maker in [
            ("family均值基线", None),
            ("岭回归", lambda: make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))),
            ("梯度提升树", lambda: HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_depth=3, random_state=0)),
        ]:
            r2s, maes, rs = [], [], []
            for tr, va in grouped_kfold(tr_all):
                ytr = np.log(tr["duration_s"].values)
                yva = np.log(va["duration_s"].values)
                if name == "family均值基线":
                    mu = tr.groupby("family_id")["duration_s"].apply(
                        lambda s: np.log(s.mean()))
                    pred = va["family_id"].map(mu).fillna(np.log(
                        tr["duration_s"].mean())).values
                else:
                    m = maker()
                    m.fit(tr[cols].values, ytr)
                    pred = m.predict(va[cols].values)
                r2s.append(r2_score(yva, pred))
                maes.append(mean_absolute_error(np.exp(yva), np.exp(pred)))
                rs.append(np.corrcoef(pred, yva)[0, 1])
            results.append(dict(window=pre, model=name,
                                R2=round(np.mean(r2s), 3),
                                MAE_s=round(np.mean(maes), 2),
                                r=round(np.mean(rs), 3)))
    res = pd.DataFrame(results)
    print(res.to_string(index=False))
    res.to_csv(RESULTS / "q2_predictability.csv", index=False, encoding="utf-8-sig")

    print("\n单特征 |r| 前 8（p10 窗口）:")
    tr_all2 = df[df.split == "train"]
    y = np.log(tr_all2["duration_s"].values)
    cols = [c for c in feat_cols if c.startswith("p10_")]
    cors = []
    for c in cols:
        x = tr_all2[c].values
        if np.std(x) < 1e-12:
            continue
        cors.append((c, abs(np.corrcoef(x, y)[0, 1])))
    cors.sort(key=lambda t: -t[1])
    for c, r in cors[:8]:
        print(f"  {c}: |r|={r:.3f}")
    return res

if __name__ == "__main__":
    df, feat_cols = load()
    q1_stats(df)
    q1_gofit(df)
    censor_study(df)
    q2_predictability(df, feat_cols)
    print("\nDIAGNOSIS DONE")
