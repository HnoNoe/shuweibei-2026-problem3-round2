"""导出论文图表所需的中间数据（时长模型表 + 示例 trial 日志）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import ROOT, RESULTS, NOMINAL
from duration import DurationModel
from policy import DispatchPolicy
from sampler import SourceSampler, run_trial
from tune_eval import get_model, FROZEN

df = pd.read_csv(ROOT / "prefix_features.csv")
m = DurationModel().fit(df[df.split == "train"])
m.table().to_csv(RESULTS / "duration_table.csv", index=False)

sam = SourceSampler(str(ROOT / "prefix_features.csv"))
pol = DispatchPolicy(get_model(), **FROZEN)
tr = run_trial("S2", 2026111705, NOMINAL[0], NOMINAL[1], sam, pol, "validation")
log = pd.DataFrame(tr.log)
log.to_csv(RESULTS / "sample_trial_log.csv", index=False)
print(log[["time", "action", "job", "reserve", "service_start",
           "completion", "cost"]].to_string(index=False))
print("\nmetrics:", tr.metrics())
