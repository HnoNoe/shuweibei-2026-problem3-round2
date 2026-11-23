"""数据诊断（第 4 步）：Q1 分布形态 + 拟合优度 / 截断对照实验 / Q2 前缀可预测性。

文献锚点：
- Strum et al. 2000（手术时长拟合对数正态，被引 182）、2017 Weibull 任务时长
  -> Q1 不止看形状，正式做 lognormal vs Weibull 的 AIC 对比
- Fleming 2013（生存分析经典）、Machine Learning for Survival Analysis 2019
  -> 截断对照：朴素丢弃 vs 含删失 MLE，量化偏差
- Early Prediction on Time Series 2009 / 2020 综述（被引 190/202）
  -> Q2 本质是"早期时间序列预测"：前缀够不够早、够不够准

纪律：全部统计只在 train split 上做；按 environment_id 分组 CV 防泄漏
（同一操作者跨环境出现，随机折会泄漏，见原文第 2 节）。
"""
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


# ---------------- Q1 分布诊断 ----------------
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
    """lognormal vs Weibull 按族 AIC 对比（文献: Strum 2000; Weibull 任务时长 2017）。"""
    from scipy import stats as st
    tr = df[df.split == "train"]
    print("=" * 72)
    print("Q1b 拟合优度: lognormal vs Weibull（按族 AIC，越小越好）")
    rows = []
    for fam, sub in tr.groupby("family_id"):
        y = sub["duration_s"].values
        # lognormal: log 数据的 MLE
        lly = np.log(y)
        mu, sig = lly.mean(), lly.std()
        ll_ln = float(np.sum(st.norm.logpdf(lly, mu, sig)))
        k_ln = 2
        aic_ln = 2 * k_ln - 2 * ll_ln
        # Weibull: 三参数 MLE（floc=0 两参数）
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


# ---------------- 截断对照实验 ----------------
def censor_study(df, reps=500, seed=42):
    """朴素丢弃截断样本 vs 含删失 MLE（EM 算法）的参数偏差对比。

    实验设计：从 Q1b 拟合的 P 族参数 (mu, sigma) 作为【已知真值】生成模拟数据，
    按样本中位数右截断（模拟 trial 内只能看到 p 是否越过 H-s 的场景），
    两种估计量各自与真值比较——隔离出"截断处理"本身的偏差，不受有限池子抽样噪声污染。

    文献：Fleming 2013（生存分析经典）；EM 更新式见标准教材。
    """
    tr = df[(df.split == "train") & (df.family_id == "P")]
    TRUE_MU = float(np.log(tr["duration_s"].values).mean())
    TRUE_SIG = float(np.log(tr["duration_s"].values).std())

    def em_censored(lx, cens, thr, iters=500):
        """右删失正态 EM：删失点条件为 Z > thr（阈值，非观测值）。"""
        from scipy import stats as st
        mu, s = lx[~cens].mean(), lx[~cens].std()
        for _ in range(iters):
            a = (thr - mu) / s
            lam = np.exp(st.norm.logpdf(a) - st.norm.logsf(a))
            ez = lx.copy()
            ez[cens] = mu + s * lam                 # E[Z | Z>thr]
            var = (lx - mu) ** 2
            var[cens] = s * s * (1 + a * lam)       # E[(Z-mu)^2 | Z>thr]
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


# ---------------- Q2 前缀可预测性 ----------------
def grouped_kfold(df, n_splits=5, seed=0):
    envs = df["environment_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(envs)
    folds = np.array_split(envs, n_splits)
    for i in range(n_splits):
        va_envs = set(folds[i])
        yield df[~df["environment_id"].isin(va_envs)], df[df["environment_id"].isin(va_envs)]


def q2_predictability(df, feat_cols):
    """按环境分组 5 折 CV；family 均值基线 / 岭回归 / GBM；R2、MAE、Pearson r。"""
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
