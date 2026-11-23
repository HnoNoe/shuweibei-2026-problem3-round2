"""时长分布诊断与前缀可预测性研究。

回答两个决定建模路线的问题：
Q1  源区间时长 d 的分布形态？（决定 p=αd+β 的不确定性结构）
Q2  已释放前缀（0.2/0.6/1.0s）的运动特征能否预测总时长 d？
    —— 决定花钱买审查是否有价值（value of information）

方法：按环境分组的 K 折交叉验证（防止同一操作者/环境泄漏），
模型：family 均值基线 vs 岭回归 vs 梯度提升树；评价 R²、MAE、Pearson r。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import ROOT, RESULTS, FIGURES

FEAT = ROOT / "prefix_features.csv"

META = {"unit_id", "dataset", "sample", "environment_id", "split",
        "family_id", "source_segment_index", "start_s", "end_s",
        "duration_s", "available"}


def load():
    df = pd.read_csv(FEAT)
    feat_cols = [c for c in df.columns if c not in META]
    return df, feat_cols


def q1_distribution(df):
    print("=" * 70)
    print("Q1: 时长分布（按 family × split 统计，单位 s）")
    tr = df[df.split == "train"]
    g = tr.groupby("family_id")["duration_s"]
    stats = g.agg(["count", "mean", "std", "median", "min", "max"]).round(2)
    print(stats.to_string())
    y = tr["duration_s"].values
    print(f"\n全 train: n={len(y)}, mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")
    # 对数正态适配: 检验 log d 的偏度/峰度
    from scipy import stats as st
    ly = np.log(y)
    print(f"log d: mean={ly.mean():.3f}, std={ly.std():.3f}, "
          f"skew={st.skew(ly):.3f}, kurtosis={st.kurtosis(ly):.3f}")
    # Shapiro 不太可能过（n 大），看 QQ 相关系数即可
    for fam, sub in tr.groupby("family_id"):
        l = np.log(sub["duration_s"].values)
        print(f"  {fam}: log d mean={l.mean():.3f} std={l.std():.3f} "
              f"skew={st.skew(l):.3f}")
    # 方差齐性：族间均值差异
    groups = [np.log(s["duration_s"].values) for _, s in tr.groupby("family_id")]
    f, p = st.f_oneway(*groups)
    print(f"族间 log d 均值差异 ANOVA: F={f:.2f}, p={p:.4f}")
    return stats


def grouped_kfold(df, n_splits=5, seed=0):
    """按 environment_id 分组 K 折。"""
    envs = df["environment_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(envs)
    folds = np.array_split(envs, n_splits)
    for i in range(n_splits):
        va_envs = set(folds[i])
        tr_mask = ~df["environment_id"].isin(va_envs)
        va_mask = df["environment_id"].isin(va_envs)
        yield df[tr_mask], df[va_mask]


def q2_predictability(df, feat_cols):
    print("=" * 70)
    print("Q2: 前缀特征对总时长 d 的预测力（按环境分组 5 折 CV，train split）")
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score, mean_absolute_error

    tr_all = df[df.split == "train"].reset_index(drop=True)
    y_all = np.log(tr_all["duration_s"].values)

    results = []
    for pre in ["p02", "p06", "p10"]:
        cols = [c for c in feat_cols if c.startswith(pre + "_")]
        for model_name, maker in [
            ("family均值基线", None),
            ("岭回归", lambda: make_pipeline(
                SimpleImputer(strategy="median"), Ridge(alpha=1.0))),
            ("梯度提升树", lambda: HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_depth=3,
                random_state=0)),
        ]:
            r2s, maes, rs = [], [], []
            for tr, va in grouped_kfold(tr_all):
                ytr = np.log(tr["duration_s"].values)
                yva = np.log(va["duration_s"].values)
                if model_name == "family均值基线":
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
            results.append(dict(window=pre, model=model_name,
                                R2=np.mean(r2s), MAE_s=np.mean(maes),
                                r=np.mean(rs)))
    res = pd.DataFrame(results).round(3)
    print(res.to_string(index=False))
    res.to_csv(RESULTS / "predictability.csv", index=False)

    # 单特征最大相关（看看什么信息最强）
    print("\n单特征 |r| 前 8（p10 窗口）:")
    cols = [c for c in feat_cols if c.startswith("p10_")]
    cors = []
    for c in cols:
        x = tr_all[c].values
        if np.std(x) < 1e-12:
            continue
        r = np.corrcoef(x, y_all)[0, 1]
        cors.append((c, abs(r)))
    cors.sort(key=lambda t: -t[1])
    for c, r in cors[:8]:
        print(f"  {c}: |r|={r:.3f}")
    return res


if __name__ == "__main__":
    df, feat_cols = load()
    q1_distribution(df)
    q2_predictability(df, feat_cols)
