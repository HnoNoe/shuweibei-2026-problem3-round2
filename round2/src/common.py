"""数维杯 C 题 (Problem 3) 重建版 —— 公共常量与数据装载。

设计原则：题目参数的唯一出处。任何脚本要用题里的数字，一律从这里 import，
不许散落魔法数。原文依据：Problem_3_Requirements.txt 第 1 节 + common_conditions.json。
"""
from pathlib import Path
import csv, json, math
from collections import defaultdict

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
PROBLEM = ROOT / "problem"
RESULTS = ROOT / "round2/results"
FIGURES = ROOT / "round2/figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ---- 区域坐标（原文第 3 节：Z0=(0,0) ... Z4=(2,0)）----
ZONES = {"Z0": (0.0, 0.0), "Z1": (1.0, 0.0), "Z2": (0.0, 1.0),
         "Z3": (1.0, 1.0), "Z4": (2.0, 0.0)}

# ---- 三个案例（原文第 1 节表：S1/S2/S3 的 H, B, K）----
CASES = {
    "S1": dict(H=100.0, B=32.0, K=4),
    "S2": dict(H=100.0, B=34.0, K=6),
    "S3": dict(H=125.0, B=40.0, K=8),
}
# 五种映射：名义 + alpha 敏感性 + beta 敏感性（原文第 3 节，非全因子）
MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
NOMINAL = (1.0, 0.5)
# 30 个公开验证种子（common_conditions.json）
VAL_SEEDS = list(range(2026111700, 2026111730))

# ---- 费用与动作常量（common_conditions.json 一致）----
FEE_FIXED = 2.0            # 每单固定费
FEE_RESERVE_PER_S = 0.1    # 预留每秒费用
FEE_DIST_PER_UNIT = 0.2    # 每单位距离费用
CHANGEOVER = 0.5           # 每次执行换型
REVIEW_COST = 1.0          # 每次审查费用
REVIEW_OCCUPY = 1.0        # 审查占用设备秒数
REVIEW_DELAY = 1.0         # 审查返回延迟
REVIEW_MAX_PER_JOB = 2     # 每作业审查上限
PREFIXES = (0.2, 0.6, 1.0)  # 三档前缀（累计窗口）
B_MIN = 0.2                # 最小预留
MAX_ACTIONS = 200          # 每 trial 动作上限
MIN_ELIGIBLE_S = 1.2       # eligible 长度条件


def zone_dist(a, b):
    xa, ya = ZONES[a]
    xb, yb = ZONES[b]
    return math.hypot(xa - xb, ya - yb)


def load_tasks():
    """tasks.csv -> {case: [job dict]}（按 job_id 排序）。"""
    rows = list(csv.DictReader(open(PROBLEM / "tasks.csv", encoding="utf-8-sig")))
    by_case = defaultdict(list)
    for r in rows:
        preds = [x for x in r["predecessors"].split("|") if x.strip()]
        by_case[r["case_id"]].append(dict(
            job_id=r["job_id"], family=r["family_id"],
            release_s=float(r["release_s"]), due_s=float(r["due_s"]),
            weight=float(r["priority_weight"]), zone=r["zone_id"],
            predecessors=preds))
    for c in by_case:
        by_case[c].sort(key=lambda r: r["job_id"])
    return dict(by_case)


def load_source_index():
    """source_segment_index.csv -> 全行列表（保持文件顺序）。"""
    return list(csv.DictReader(
        open(PROBLEM / "source_segment_index.csv", encoding="utf-8-sig")))


def eligible_pool(index_rows):
    """{(split, family): [row dict（含 d）]}，保持源池行顺序。"""
    pool = defaultdict(list)
    for r in index_rows:
        if r["eligible"] != "1":
            continue
        pool[(r["split"], r["family_id"])].append(dict(
            unit_id=r["unit_id"], d=float(r["duration_s"]),
            dataset=r["dataset"], sample=r["sample"]))
    return dict(pool)


def load_conditions():
    return json.load(open(PROBLEM / "common_conditions.json", encoding="utf-8"))
