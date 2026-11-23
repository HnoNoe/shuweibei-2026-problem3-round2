"""时长估计与不确定性建模。

诊断结论：0.2-1.0s 前缀的运动学特征对作业总时长 d 的线性/树模型预测力很弱
（分组 CV 下 R2<=0）。因此本题的重点不是精确点预测，而是：
1) 用 family 分层 + 前缀强度做 d 的条件分布估计（贝叶斯/分位数）；
2) 显式输出不确定性区间，供调度策略选择预留时长 b（风险对冲）。

模型冻结协议：只在 train split 拟合，validation 选超参，test 前冻结。
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class DurationEstimator:
    """基于 family 先验 + 前缀运动强度的时长条件分布估计器。

    采用对数正态先验（family 分层），前缀强度作为弱似然做贝叶斯更新：
      log d ~ N(mu_family, sigma_family^2)
      观测前缀运动强度 z -> 与 log d 的相关系数 rho（在 train 上估计）
      后验均值 = mu + rho*sigma*(z-mu_z)/sigma_z
    输出点估计(中位数)、分位数区间。
    """
    def __init__(self, rho_shrink: float = 0.5):
        self.rho_shrink = rho_shrink   # 收缩系数：前缀证据的信任度
        self.params: Dict[str, dict] = {}
        self.global_mu = None
        self.global_sigma = None

    @staticmethod
    def _intensity(row, prefix="p02") -> float:
        """前缀综合运动强度：各速度/加速度通道 std 的均值（稳健）。"""
        keys = [f"{prefix}_vall_std", f"{prefix}_wristL_std", f"{prefix}_wristR_std",
                f"{prefix}_accel_mean", f"{prefix}_head_std", f"{prefix}_qL_std"]
        vals = []
        for k in keys:
            v = row.get(k, np.nan)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else 0.0

    def fit(self, train_df: pd.DataFrame):
        y = np.log(train_df["duration_s"].clip(lower=0.2).values)
        self.global_mu, self.global_sigma = float(np.mean(y)), float(np.std(y))
        train = train_df.copy()
        train["_int"] = train.apply(self._intensity, axis=1)
        for fam, sub in train.groupby("family_id"):
            ly = np.log(sub["duration_s"].clip(lower=0.2).values)
            mu, sig = float(np.mean(ly)), max(float(np.std(ly)), 0.3)
            z = sub["_int"].values
            rho = 0.0
            if len(sub) > 5 and z.std() > 1e-9:
                rho = float(np.corrcoef(z, ly)[0, 1])
            rho = self.rho_shrink * np.clip(rho, -0.6, 0.6)
            self.params[fam] = dict(mu=mu, sigma=sig, rho=rho,
                                    z_mean=float(np.mean(z)),
                                    z_std=max(float(np.std(z)), 1e-6),
                                    n=len(sub))
        return self

    def predict(self, family: str, intensity: float, prefix_level: float = 0.2
                ) -> Dict[str, float]:
        pr = self.params.get(family, dict(mu=self.global_mu, sigma=self.global_sigma,
                                          rho=0.0, z_mean=0.0, z_std=1.0, n=0))
        mu, sig, rho = pr["mu"], pr["sigma"], pr["rho"]
        # 证据信任度随前缀长度增加（0.2/0.6/1.0 -> 0.5/0.75/1.0）
        trust = {0.2: 0.5, 0.6: 0.75, 1.0: 1.0}.get(prefix_level, 0.5)
        zscore = (intensity - pr["z_mean"]) / pr["z_std"]
        post_mu = mu + trust * rho * sig * np.clip(zscore, -3, 3)
        # 观测越多，后验方差略降
        post_sigma = sig * (1 - 0.1 * trust * abs(rho))
        return dict(
            d_median=float(np.exp(post_mu)),
            d_mean=float(np.exp(post_mu + 0.5 * post_sigma**2)),
            d_q10=float(np.exp(post_mu - 1.2816 * post_sigma)),
            d_q90=float(np.exp(post_mu + 1.2816 * post_sigma)),
            d_q75=float(np.exp(post_mu + 0.6745 * post_sigma)),
            mu=post_mu, sigma=post_sigma, family=family,
        )

    def evaluate(self, df: pd.DataFrame) -> dict:
        """在给定 split 上评估点预测与覆盖率。"""
        rows = []
        for _, r in df.iterrows():
            fam = r["family_id"]
            inten = self._intensity(r, "p02")
            pred = self.predict(fam, inten, 0.2)
            rows.append((r["duration_s"], pred))
        y = np.array([a for a, _ in rows])
        med = np.array([b["d_median"] for _, b in rows])
        q10 = np.array([b["d_q10"] for _, b in rows])
        q90 = np.array([b["d_q90"] for _, b in rows])
        mae = float(np.mean(np.abs(y - med)))
        mape = float(np.mean(np.abs(y - med) / y))
        cover = float(np.mean((y >= q10) & (y <= q90)))
        return dict(n=len(y), mae_s=mae, mape=mape, q10q90_coverage=cover,
                    median_pred=np.median(med), median_true=np.median(y))

if __name__ == "__main__":
    df = pd.read_csv("data/prefix_features_EgoDemo.csv")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "validation"]
    est = DurationEstimator(rho_shrink=0.5).fit(tr)
    print("train:", est.evaluate(tr))
    print("valid:", est.evaluate(va))
    print("params:", est.params)
