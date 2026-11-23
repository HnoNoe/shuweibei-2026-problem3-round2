
from pathlib import Path
import csv, json, math
from collections import defaultdict

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
PROBLEM = ROOT / "problem"
RESULTS = ROOT / "round2/results"
FIGURES = ROOT / "round2/figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

ZONES = {"Z0": (0.0, 0.0), "Z1": (1.0, 0.0), "Z2": (0.0, 1.0),
         "Z3": (1.0, 1.0), "Z4": (2.0, 0.0)}

CASES = {
    "S1": dict(H=100.0, B=32.0, K=4),
    "S2": dict(H=100.0, B=34.0, K=6),
    "S3": dict(H=125.0, B=40.0, K=8),
}

MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
NOMINAL = (1.0, 0.5)

VAL_SEEDS = list(range(2026111700, 2026111730))

FEE_FIXED = 2.0
FEE_RESERVE_PER_S = 0.1
FEE_DIST_PER_UNIT = 0.2
CHANGEOVER = 0.5
REVIEW_COST = 1.0
REVIEW_OCCUPY = 1.0
REVIEW_DELAY = 1.0
REVIEW_MAX_PER_JOB = 2
PREFIXES = (0.2, 0.6, 1.0)
B_MIN = 0.2
MAX_ACTIONS = 200
MIN_ELIGIBLE_S = 1.2

def zone_dist(a, b):
    xa, ya = ZONES[a]
    xb, yb = ZONES[b]
    return math.hypot(xa - xb, ya - yb)

def load_tasks():

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

    return list(csv.DictReader(
        open(PROBLEM / "source_segment_index.csv", encoding="utf-8-sig")))

def eligible_pool(index_rows):

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
