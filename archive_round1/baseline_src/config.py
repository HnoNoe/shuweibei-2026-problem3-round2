"""数维杯 C 题 (Problem 3) —— 全局配置与路径常量。

数据来源为只读的本地数据集，脚本不修改原始数据。
"""
from pathlib import Path

# ---- 原始数据根目录（只读） ----
DATA_ROOT = Path(
    r"C:/Users/Camevian/Downloads/C_Problem_Data/"
    r"Restocking_Problem_and_Data_EN/Restocking_Problem_and_Data_EN"
)
CONSTRAINTS = DATA_ROOT / "Restocking_Tasks_and_Constraints"
RAW = DATA_ROOT / "Raw_Data" / "EgoDemo" / "EgoDemo"

# ---- 工程输出目录 ----
WORK = Path(r"C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37")
SRC = WORK / "src"
RESULTS = WORK / "results"
FIGURES = WORK / "figures"
REPORT = WORK / "report"
DATA = WORK / "data"
for _d in (RESULTS, FIGURES, REPORT, DATA):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 题目常量 ----
PREFIXES = [0.2, 0.6, 1.0]          # 累计可见源时间窗口（秒）
NOMINAL_ALPHA = 1.0
NOMINAL_BETA = 0.5
SENSITIVITY = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
VAL_SEEDS = list(range(2026111700, 2026111730))  # 30 个公共验证种子

CASES = {
    "S1": dict(jobs=8, H=100.0, B=32.0, K=4),
    "S2": dict(jobs=12, H=100.0, B=34.0, K=6),
    "S3": dict(jobs=16, H=125.0, B=40.0, K=8),
}

# ---- 特征列 ----
SIMPLE_NUM = ["timestamp", "frame_index"]
