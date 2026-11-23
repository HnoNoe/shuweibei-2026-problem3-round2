"""时长不确定性模型（重做版）。

事实基础（analyze_duration.py 验证）：
- 前缀特征对 d 无预测力（分组 CV 下 GBM R²≈−0.5，不优于 family 基线）；
- d 重右尾（1.2~125s），log d 仍右偏（偏度 1.14），族间差异显著（p=0.001）。

模型：family 分层对数正态  log d ~ N(mu_f, sig_f^2)
- mu_f, sig_f 用 train split 的 log d 矩估计，sig 设下限 0.35 抗小样本（L 族 n=10）；
- 提供解析分位数与向量化抽样；p = alpha*d + beta 的分布由此诱导。
- 策略接口只暴露分位数/均值/抽样，不暴露任何单条真值。
"""
import numpy as np
import pandas as pd

SIGMA_FLOOR = 0.35


class DurationModel:
    def __init__(self):
        self.fam = {}          # family -> (mu, sigma)
        self.global_mu = 0.0
        self.global_sigma = 1.0

    def fit(self, train_df: pd.DataFrame):
        y = np.log(train_df["duration_s"].clip(lower=0.2).values)
        self.global_mu, self.global_sigma = float(y.mean()), float(y.std())
        for fam, sub in train_df.groupby("family_id"):
            ly = np.log(sub["duration_s"].clip(lower=0.2).values)
            self.fam[fam] = (float(ly.mean()),
                             max(float(ly.std()), SIGMA_FLOOR))
        return self

    def _params(self, family):
        return self.fam.get(family,
                            (self.global_mu, self.global_sigma))

    # ---- d 的分位数（秒）----
    def d_quantile(self, family, q):
        mu, sig = self._params(family)
        from scipy.stats import norm
        return float(np.exp(mu + sig * norm.ppf(q)))

    def d_median(self, family):
        return self.d_quantile(family, 0.5)

    def d_mean(self, family):
        mu, sig = self._params(family)
        return float(np.exp(mu + 0.5 * sig * sig))

    # ---- p = alpha*d + beta ----
    def p_mean(self, family, alpha, beta):
        return alpha * self.d_mean(family) + beta

    def p_median(self, family, alpha, beta):
        return alpha * self.d_median(family) + beta

    def p_quantile(self, family, alpha, beta, q):
        return alpha * self.d_quantile(family, q) + beta

    def sample_d(self, family, n, rng):
        mu, sig = self._params(family)
        return np.exp(rng.normal(mu, sig, size=n))

    def sample_p(self, family, alpha, beta, n, rng):
        return alpha * self.sample_d(family, n, rng) + beta

    def table(self):
        rows = []
        for fam, (mu, sig) in sorted(self.fam.items()):
            rows.append(dict(family=fam, mu=mu, sigma=sig,
                             d_median=np.exp(mu), d_mean=np.exp(mu + sig**2/2),
                             d_q90=np.exp(mu + 1.2816 * sig),
                             d_q99=np.exp(mu + 2.3263 * sig)))
        return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    from common import ROOT
    df = pd.read_csv(ROOT / "prefix_features.csv")
    m = DurationModel().fit(df[df.split == "train"])
    print(m.table().to_string(index=False))
