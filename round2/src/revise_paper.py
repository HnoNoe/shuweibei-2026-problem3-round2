import json
import os
import re
import sys
import zipfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

FIGDIR = r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c/round2/figures3"
RES = r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c/round2/results"


def find_par(doc, text, style_prefix=None):
    for p in doc.paragraphs:
        t = p.text.strip()
        if text in t and (style_prefix is None
                          or p.style.name.startswith(style_prefix)):
            return p
    raise ValueError(f"anchor not found: {text}")


def insert_para_before(anchor, text, style="Normal", bold=False):
    p = anchor.insert_paragraph_before(text, style=style)
    if bold:
        for r in p.runs:
            r.bold = True
    return p


def insert_figure_before(anchor, doc, png, caption, width_in=6.0):
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pic.add_run()
    run.add_picture(png, width=Inches(width_in))
    anchor._p.addprevious(pic._p)
    cap = insert_para_before(anchor, caption, style="Caption")
    return pic, cap


def insert_table_before(anchor, doc, heads, rows, widths):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(heads))
    tbl.style = doc.tables[0].style if doc.tables else "Table Grid"
    for j, htext in enumerate(heads):
        tbl.rows[0].cells[j].text = htext
    for i, r in enumerate(rows, start=1):
        for j, c in enumerate(r):
            tbl.rows[i].cells[j].text = c
    anchor._p.addprevious(tbl._tbl)
    return tbl


def move_caption_above(tbl, anchor, caption_text):
    cap = insert_para_before(anchor, caption_text, style="Caption")
    tbl._tbl.addprevious(cap._p)
    return cap


def fmt_p(p):
    if p < 1e-16:
        return "<1e-16"
    return f"{p:.2e}"


def stats_rows():
    q9 = json.load(open(f"{RES}/q9_stats_audit.json", encoding="utf-8"))
    rows = []
    for fam, tag in (("validation_family", ""), ("test_family", " [test]")):
        for r in q9[fam]:
            name = r["contrast"] + tag
            rows.append([
                name,
                f"{r['mean']:+.4f} ± {r['ci95_halfwidth']:.4f}",
                fmt_p(r["p_wilcoxon"]),
                fmt_p(r["p_holm"]),
                f"{r['cohens_dz']:+.2f}",
            ])
    return rows


def renumber_all(doc):
    fig_map = {n: n + 4 for n in range(22, 1, -1)}
    tbl_map = {n: n + 2 for n in range(14, 1, -1)}
    cite_map = {n: n - 1 for n in range(12, 2, -1)}

    def shift_runs(p):
        for run in p.runs:
            t = run.text
            if not t:
                continue
            for n, m in fig_map.items():
                t = re.sub(rf"Figure {n}(?![0-9])", f"Figure {m}", t)
            for n, m in tbl_map.items():
                t = re.sub(rf"Table {n}(?![0-9])", f"Table {m}", t)
            for n, m in cite_map.items():
                t = re.sub(rf"\[{n}\]", f"[{m}]", t)
            run.text = t

    def shift_heading_text(p):
        runs = p.runs
        if not runs:
            return
        t = "".join(r.text for r in runs)
        m = re.match(r"^(\d+)\.\s+(.*)$", t)
        if m and int(m.group(1)) >= 3:
            t = f"{int(m.group(1)) + 2}. {m.group(2)}"
        else:
            m = re.match(r"^(\d+)\.(\d+)\s+(.*)$", t)
            if m and int(m.group(1)) >= 3:
                t = f"{int(m.group(1)) + 2}.{m.group(2)} {m.group(3)}"
        runs[0].text = t
        for r in runs[1:]:
            r.text = ""

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    def toc_paras(doc):
        out = []
        for p in doc.paragraphs:
            if p._p.findall(f".//{W}hyperlink") and "SWContents" in p._p.xml:
                out.append(p)
        return out

    def retarget_toc(p):
        ts = p._p.findall(f".//{W}t")
        joined = "".join(t.text or "" for t in ts)
        m = re.match(r"^(\d+)\.\s+(.*)$", joined)
        if m and int(m.group(1)) >= 3:
            joined = f"{int(m.group(1)) + 2}. {m.group(2)}"
        else:
            m = re.match(r"^(\d+)\.(\d+)\s+(.*)$", joined)
            if m and int(m.group(1)) >= 3:
                joined = f"{int(m.group(1)) + 2}.{m.group(2)} {m.group(3)}"
        if ts:
            ts[0].text = joined
            for t in ts[1:]:
                t.text = ""

    def clone_with_text(p, text, page):
        import copy
        newp = copy.deepcopy(p._p)
        ts = newp.findall(f".//{W}t")
        if len(ts) >= 2:
            ts[0].text = text
            for t in ts[1:-1]:
                t.text = ""
            ts[-1].text = str(page)
        return newp

    drop = []
    for p in doc.paragraphs:
        stl = p.style.name if p.style is not None else ""
        t = p.text.strip()
        if t.startswith("[2] HnoNoe"):
            drop.append(p)
            continue
        if stl.startswith("Heading"):
            shift_heading_text(p)
        shift_runs(p)
    for p in drop:
        p._p.getparent().remove(p._p)

    pages = json.load(open(f"{P12}/toc_pages.json", encoding="utf-8"))
    tocs = toc_paras(doc)
    anchor24 = None
    for p in tocs:
        if p.text.strip().startswith("2.4 Analysis of Question Three"):
            anchor24 = p
        retarget_toc(p)
    assert anchor24 is not None, "TOC anchor 2.4 not found"
    style2 = anchor24._p.find(f"{W}pPr/{W}ind") is not None
    entries = [
        ("3. Problem 1: Operational States and Observation Reliability",
         pages["p1"], False),
        ("3.1 Data audit and state variables", pages["p1"], True),
        ("3.2 Families are kinematically distinct — and environment explains most of it",
         pages["p1"], True),
        ("3.3 Frame-level reliability: baseline and upgrade", pages["p1b"], True),
        ("3.4 Reliability is learnable within an environment — not across",
         pages["p1b"], True),
        ("4. Problem 2: State Prediction and Re-observation during Missing Observations",
         pages["p2"], False),
        ("4.1 Masking replay protocol", pages["p2"], True),
        ("4.2 Prediction: baselines and the family-ridge upgrade", pages["p2b"], True),
        ("4.3 Error growth and the failure of velocity extrapolation",
         pages["p2b"], True),
        ("4.4 When to re-observe: fixed schedules versus uncertainty thresholds",
         pages["p2c"], True),
    ]
    prev = anchor24._p
    for text, page, sub in entries:
        np_ = clone_with_text(anchor24, text, page)
        ind = np_.find(f"{W}pPr/{W}ind")
        if ind is not None and not sub:
            np_.find(f"{W}pPr").remove(ind)
        elif ind is None and sub:
            pass
        prev.addnext(np_)
        prev = np_


P12 = r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c/round2/results_p12"


def insert_p1_p2(doc):
    h3 = find_par(doc, "5. Symbols and Assumptions", "Heading 1")

    insert_para_before(h3, "3. Problem 1: Operational States and Observation "
                           "Reliability", "Heading 1")
    insert_para_before(h3, "3.1 Data audit and state variables", "Heading 2")
    insert_para_before(
        h3,
        "We audited all 69 supplied records (31 EgoDemo, 14 EgoPro, 24 "
        "EgoStandard; 66,070 frames, 2,206 s) before any modelling. Three "
        "checks matter for everything downstream. First, time alignment: "
        "median frame spacing is 33.3 ms with jitter below 2 ms in every "
        "record, but three records contain gap bursts above 100 ms that we "
        "flag rather than interpolate. Second, identity: the 69 episode "
        "identifiers are unique, and they collapse onto 24 environments, 30 "
        "scenes and 41 operators — environment, not record, is the correct "
        "grouping unit for validation, exactly as the official split "
        "prescribes (16 train, 4 validation, 4 test environments). Third, "
        "coverage: of the 349 annotated intervals, the 343 lasting at least "
        "1.2 s enter common sampling; the six excluded intervals are listed "
        "with their reasons. Quality flags are sparse (1.25% of frames; "
        "0.66% EgoDemo, 2.32% EgoPro, 0.97% EgoStandard) and are dominated "
        "by occlusion, with a small near-edge component (Figure 2).")
    insert_para_before(
        h3,
        "The state of a hand at frame t is summarized by a representative "
        "point — the geometric centre of its 21 recorded keypoints — "
        "together with the spread of the keypoints about that centre, the "
        "two-hand distance, and the first two temporal derivatives of the "
        "centre. A hand state is never read as a product state; only the "
        "kinematics of the representative point enter the models. This "
        "representation is retained unchanged in Problem 2.")

    insert_figure_before(
        h3, doc, f"{FIGDIR}/fig_p1_audit.png",
        "Figure 2. Data audit. (a) Bad-frame ratio and median record "
        "duration by dataset. (b) Distribution of mean hand speed across "
        "the 349 annotated intervals, by family; the three families are "
        "kinematically distinct (Kruskal–Wallis p < 1e-16, epsilon-squared "
        "up to 0.27).",
        width_in=6.5)

    insert_para_before(h3, "3.2 Families are kinematically distinct — and "
                           "environment explains most of it", "Heading 2")
    insert_para_before(
        h3,
        "Across the 349 intervals, hand speed, path length, acceleration "
        "and keypoint spread differ significantly across families, across "
        "datasets, and across operational stages (Kruskal–Wallis, Holm-"
        "corrected p < 0.001 throughout; effect sizes up to epsilon-squared "
        "0.27). These are not incidental details: they are the reason a "
        "single global duration or motion model must fail, and they "
        "motivate the family-conditional design used everywhere below. An "
        "intraclass analysis quantifies the grouping: environment alone "
        "accounts for 88% of the variance in mean interval speed "
        "(ICC = 0.88) and 79% of keypoint spread (ICC = 0.79), while the "
        "interval-level bad-frame rate is more weakly clustered (ICC = "
        "0.34). Kinematics are a property of the environment; observation "
        "quality is only partly so — a distinction that drives the "
        "reliability result next.")

    insert_para_before(h3, "3.3 Frame-level reliability: baseline and "
                           "upgrade", "Heading 2")
    insert_para_before(
        h3,
        "The baseline for observation reliability is the constant "
        "predictor: it always reports the marginal bad-frame rate and "
        "carries AUROC 0.50 by construction. The upgrade is a frame-level "
        "classifier over the kinematic state (speed, spread, hand "
        "distance, frame gap), the representative-point position, the "
        "operational-stage fraction and the dataset, trained on the 55 "
        "training environments with grouped cross-validation. Two "
        "estimators were compared, an L1-regularized logistic model and a "
        "histogram gradient-boosting model; both are trained with class "
        "weights because bad frames are rare.")

    insert_para_before(h3, "3.4 Reliability is learnable within an "
                           "environment — not across", "Heading 2")
    insert_para_before(
        h3,
        "Fitted and evaluated inside a single environment, the logistic "
        "model separates reliable from unreliable frames decisively: mean "
        "AUROC 0.891 (bootstrap 95% CI 0.710–0.995 over the 13 "
        "environments with at least 20 flagged frames). The same features "
        "evaluated on held-out environments collapse to 0.605 (CI "
        "0.575–0.637; Table 2, Figure 3). The confidence intervals do not "
        "overlap: the signal is real, its direction is environment-"
        "specific, and it does not transfer. This finding disciplines the "
        "rest of the paper — no downstream decision may rely on a frame-"
        "level quality prediction, because a frozen model cannot be "
        "trusted in an unseen store. What does transfer is the "
        "distribution-level information formalized in Problem 3.")
    insert_figure_before(
        h3, doc, f"{FIGDIR}/fig_p1_reliability.png",
        "Figure 3. Frame-level reliability AUROC. Within an environment "
        "the kinematic-state model separates reliable from unreliable "
        "frames (mean 0.891); across environments the same model is barely "
        "better than chance (held-out test 0.605).",
        width_in=3.4)
    rows = [
        ["Within environment (n = 13)",
         "0.891", "0.710 – 0.995", "fitted and tested per environment"],
        ["Held-out environments (test)",
         "0.605", "0.575 – 0.637", "frozen, environment-grouped training"],
        ["Constant baseline", "0.500", "—", "marginal rate predictor"],
    ]
    tb = insert_table_before(h3, doc,
                             ["Evaluation", "Mean AUROC", "95% CI", "Protocol"],
                             rows, None)
    move_caption_above(
        tb, h3,
        "Table 2. Frame-level observation-reliability model. Bootstrap "
        "confidence intervals; the within/across gap is the central "
        "empirical fact of Problem 1.")

    insert_para_before(h3, "4. Problem 2: State Prediction and "
                           "Re-observation during Missing Observations",
                       "Heading 1")
    insert_para_before(h3, "4.1 Masking replay protocol", "Heading 2")
    insert_para_before(
        h3,
        "We replay the records under artificial masking. A masking window "
        "(t0, t0 + Δ) hides the hand state; the predictor sees only frames "
        "up to t0 and must produce the representative-point state at t0 + "
        "Δ. Windows, budgets and delays are fixed in advance and shared by "
        "every method: Δ takes six values from 0.25 s to 5 s, t0 is drawn "
        "uniformly inside the record subject to the window fitting, all "
        "methods see identical (record, t0, Δ) triples, and a re-"
        "observation request returns information only after a 1 s delay. "
        "Reference states come from the recorded trajectories themselves; "
        "windows overlapping natural bad frames are excluded from scoring. "
        "We report error, accumulated observation cost and waiting time "
        "for every method, as the problem statement requires, and we do "
        "not interpret artificial masking as natural occlusion (the two "
        "are compared explicitly in the sensitivity discussion).")
    insert_para_before(h3, "4.2 Prediction: baselines and the family-ridge "
                           "upgrade", "Heading 2")
    insert_para_before(
        h3,
        "Two textbook baselines anchor the comparison. HOLD repeats the "
        "last observed state — the strategy a naive system applies when a "
        "hand disappears into a shelf. CONSTANT-VELOCITY fits a straight "
        "line to the last 0.5 s and extrapolates. The upgrade is a family-"
        "conditioned ridge regressor from the visible kinematics (state, "
        "velocity, acceleration, stage fraction, dataset, family) to the "
        "positional increment over Δ, trained on training environments "
        "with standardized features. Its residual dispersion by Δ doubles "
        "as a predictive uncertainty estimate.")
    insert_para_before(h3, "4.3 Error growth and the failure of velocity "
                           "extrapolation", "Heading 2")
    insert_para_before(
        h3,
        "Error grows with the missing window for every method, but in "
        "qualitatively different ways (Figure 4, Table 3). CONSTANT-"
        "VELOCITY is best at Δ = 0.25 s (0.050 m) and then diverges, "
        "reaching 1.90 m at Δ = 5 s — worse than doing nothing for every "
        "Δ ≥ 1 s. HOLD saturates around 0.44 m as the hand's motion stays "
        "local. The family ridge dominates across the whole range (0.087–"
        "0.42 m): paired signed-rank tests over the 403 record × Δ cells "
        "give ridge versus CONSTANT-VELOCITY −0.495 m (Holm-adjusted p < "
        "1e-15, d_z = −0.68) and ridge versus HOLD −0.011 m (p = 0.025, "
        "d_z = −0.17). The upgrade is not a small tuning gain: it removes "
        "the divergence mode entirely.")
    insert_figure_before(
        h3, doc, f"{FIGDIR}/fig_p2_growth.png",
        "Figure 4. Mean 3D position error versus missing-observation "
        "window for the two baselines and the family-ridge predictor "
        "(shaded: 95% bootstrap CI). Velocity extrapolation diverges; the "
        "family ridge is stable across two decades of Δ.",
        width_in=3.6)
    rows = [
        ["Family ridge (upgrade)",
         "0.260", "0.368", "0.425",
         "vs CV: p < 1e-15, d_z = −0.68; vs HOLD: p = 0.025"],
        ["HOLD (last state)",
         "0.293", "0.388", "0.436", "saturates; never diverges"],
        ["Constant velocity",
         "0.406", "0.859", "1.904", "best at 0.25 s, diverges by 1 s"],
    ]
    tb = insert_table_before(
        h3, doc,
        ["Predictor", "MAE at Δ=1 s (m)", "Δ=2 s (m)", "Δ=5 s (m)",
         "Paired audit"],
        rows, None)
    move_caption_above(
        tb, h3,
        "Table 3. Masking-replay prediction audit (1,380 windows on "
        "training/validation/test records).")
    insert_para_before(h3, "4.4 When to re-observe: fixed schedules versus "
                           "uncertainty thresholds", "Heading 2")
    insert_para_before(
        h3,
        "Prediction alone does not answer when a new observation is worth "
        "its cost. We simulate continuous replays in which a policy may "
        "request re-observation at any epoch (1 s delay, identical "
        "conditions for all policies) and score it by time-mean error and "
        "request rate. Four policy families compete: no re-observation; "
        "fixed-interval schedules; an uncertainty-threshold policy that "
        "requests when the ridge model's residual dispersion exceeds a "
        "tolerance; and an oracle that requests exactly when the realized "
        "error would exceed the tolerance. The threshold policy matches "
        "the 1 s fixed schedule's error (0.174 m versus 0.173 m) with 18% "
        "fewer requests (48.3 versus 59.2 per minute; paired signed-rank, "
        "Holm-adjusted p < 1e-15 over 317 records), and the oracle shows "
        "the threshold policy is within 10% of the information-theoretic "
        "frontier (Figure 5). Waiting — continuing to predict without "
        "requesting — is optimal precisely when the residual dispersion "
        "curve is flat, which the model reports rather than assumes.")
    insert_figure_before(
        h3, doc, f"{FIGDIR}/fig_p2_tradeoff.png",
        "Figure 5. Re-observation cost–error frontier. Fixed schedules "
        "trace the grey curve; the uncertainty-threshold policy sits on "
        "its left (same error, fewer requests); the oracle bounds the "
        "achievable region.",
        width_in=3.6)


def rewrite_abstract(doc):
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("A restocking job can miss its due time"):
            new = (
                "First-person demonstrations of human restocking contain "
                "everything a robot would need to plan its own restocking "
                "runs — and three distinct reasons why using it is hard. "
                "We study all three on the same evidence: 69 records, 349 "
                "annotated intervals and 66,070 frames of hand kinematics.")
            runs = p.runs
            runs[0].text = new
            for r in runs[1:]:
                r.text = ""
            continue
        if t.startswith("We fitted the duration model on 246 eligible"):
            new = (
                "Problem 1 asks which observations can be trusted. Auditing "
                "the records (time alignment, identity, coverage) shows the "
                "three operation families are kinematically distinct and "
                "that environment explains 88% of the variance in hand "
                "speed. A frame-level reliability model fitted inside one "
                "environment separates reliable from unreliable frames with "
                "AUROC 0.89, but the same frozen model collapses to 0.60 on "
                "held-out environments — reliability is learnable, not "
                "transferable.")
            runs = p.runs
            if runs:
                runs[0].text = new
                for r in runs[1:]:
                    r.text = ""
            continue
        if t.startswith("We selected travel-aware apparent tardiness cost"):
            new = (
                "Problem 2 asks how long a missing hand can be predicted "
                "and when to buy a new observation. On 1,380 artificial-"
                "masking replays, constant-velocity extrapolation — best "
                "at 0.25 s — diverges to 1.9 m by 5 s; holding the last "
                "state saturates near 0.44 m; a family-conditioned ridge "
                "stays stable across the whole range (paired tests, Holm-"
                "corrected). An uncertainty-threshold re-observation "
                "policy matches the best fixed schedule's error with 18% "
                "fewer requests.")
            runs = p.runs
            if runs:
                runs[0].text = new
                for r in runs[1:]:
                    r.text = ""
            continue
        if t.startswith("Larger reservations shifted capacity use"):
            new = (
                "Problem 3 asks how to schedule restocking jobs under "
                "duration uncertainty. Duration prediction from visible "
                "prefixes is falsified (negative out-of-sample R² in every "
                "window), so decisions are made under distributions: a "
                "hierarchical log-normal duration model with EM censoring "
                "recovery, and a two-stage stochastic dispatch engine. "
                "Against a conventional baseline, the refined pipeline "
                "gains +0.079 on validation and +0.105 on independent test "
                "seeds in weighted on-time rate (Holm-adjusted p = 6.8e-40 "
                "and 1.0e-54), buys zero paid reviews, and carries a zero "
                "sample-average-approximation optimality gap where bounds "
                "are exact. All three problems share one audit discipline: "
                "falsifiable hypotheses, common random numbers, paired "
                "signed-rank tests with Holm correction, and frozen held-"
                "out evaluation.")
            runs = p.runs
            if runs:
                runs[0].text = new
                for r in runs[1:]:
                    r.text = ""
            continue


def main():
    src, out = sys.argv[1], sys.argv[2]
    doc = Document(src)

    renumber_all(doc)
    insert_p1_p2(doc)
    rewrite_abstract(doc)

    h6 = find_par(doc, "8. Sensitivity Analysis", "Heading 1")

    insert_para_before(h6, "7.6 From a Conventional Baseline to a Certified "
                            "Pipeline: a Structured Statistical Audit", "Heading 2")
    insert_para_before(
        h6,
        "Everything above this section is the baseline pipeline: the design a "
        "conventional, competent team would ship — family-mean and ridge point "
        "estimates of duration, censored observations dropped, a deterministic "
        "ATC dispatch score, and a single-mapping validation of 90 trials. None "
        "of these defaults is unreasonable; none of them was ever tested. This "
        "section subjects the baseline to a structured audit. Each of the five "
        "upgrades below is triggered by a specific, testable failure of the "
        "baseline, is evaluated on the same 450 validation trials under common "
        "random numbers (CRN), and is admitted only if it survives paired "
        "significance testing with Holm correction across the whole family of "
        "contrasts. Figure 16 maps the argument; Table 15 reports the full "
        "statistical audit.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figZ_flagship.png",
        "Figure 16. The audit at a glance. Each row confronts a conventional "
        "baseline default (left, red) with the evidence that falsifies or "
        "quantifies it (centre; every quantity is a CRN-paired contrast on "
        "real validation data) and the upgrade the evidence forces (right, "
        "blue). The bottom strip collects the optimality certificate.",
        width_in=6.5)

    insert_para_before(h6, "Upgrade 1 — the information layer is a "
                           "falsifiable hypothesis, and it is false.", "Normal",
                       bold=True)
    insert_para_before(
        h6,
        "The baseline entertained prefix-based duration prediction and a "
        "selective-review experiment, but never tested whether the prefix "
        "carries any signal. The test is direct: group-wise cross-validated "
        "regression from prefix kinematics to total duration. Out-of-sample "
        "R² is negative for every prefix window and every estimator — family "
        "mean −0.599, ridge −1.034 to −2.058, gradient boosting −0.481 "
        "(Figure 17) — and the strongest single prefix feature correlates "
        "only at r = 0.182. Prediction is therefore rejected as a design "
        "basis. Because buying reviews is only rational under predictable "
        "durations, the review route itself must be falsified rather than "
        "assumed away: an adversarial arm given every advantage (full review "
        "budget, best prefix model, C5 engine) pays +4.64 more per trial "
        "(Figure 18) and is significantly worse than zero-review, "
        "−0.0504 ± 0.0084 on validation (Holm-adjusted p = 9.9e-27) and "
        "−0.0633 ± 0.0083 on independent test seeds (p = 2.8e-34) "
        "(Figure 19). The refined policy buys zero reviews.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC1_r2.png",
        "Figure 17. Out-of-sample R² of prefix-based duration prediction for "
        "three prefix windows and three estimators; all values are negative.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC2_falsified.png",
        "Figure 18. The falsified information route: an adversarial arm "
        "allowed to buy the maximum number of reviews pays +4.64 more per "
        "trial than zero-review and still loses on timeliness.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC3_adversarial.png",
        "Figure 19. Paired gain of the adversarial buy-top arm against the "
        "refined zero-review policy (450 CRN-paired validation trials).",
        width_in=3.2)

    insert_para_before(h6, "Upgrade 2 — durations are distributions, and "
                           "truncation is information.", "Normal", bold=True)
    insert_para_before(
        h6,
        "The baseline reduces each job family to a point estimate and "
        "discards censored executions. Both defaults fail scrutiny. A family "
        "effect is present (ANOVA p = 0.0012) and log-normal fits dominate "
        "on AIC in every family, so the refined model is a hierarchical "
        "log-normal with a σ ≥ 0.35 floor against overconfident tails from "
        "small samples (Figure 20). Censoring, meanwhile, is not missing "
        "data: a job that crosses the horizon still reveals d > H − s. In a "
        "known-truth simulation with right-censoring at the median, naive "
        "deletion biases the log-mean by −0.609 (n = 124), while the EM "
        "censoring-aware update brings the bias to −0.008 (Figure 21). "
        "Every downstream decision inherits this correction online.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figH_duration_model.png",
        "Figure 20. Fitted hierarchical log-normal duration model (σ floored "
        "at 0.35). Family L is heavy-tailed and drives the robust-anchor "
        "design choice.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figD_censoring.png",
        "Figure 21. Truncation handling against known truth (500 repeats, "
        "right-censoring at the median): naive deletion biases the log-mean "
        "by about −0.6; EM recovers it.",
        width_in=6.3)

    insert_para_before(h6, "Upgrade 3 — the decision layer must reason under "
                           "the distribution, not through a point estimate.",
                       "Normal", bold=True)
    insert_para_before(
        h6,
        "With prediction rejected, performance can only come from better "
        "decisions under residual uncertainty. The ledger in Figure 22 "
        "decomposes the full gain into single-design paired contrasts; "
        "Table 15 gives the corresponding tests. Three components were "
        "rejected and are kept on record: a robust Q90 anchor in isolation "
        "(−0.0178 against the median anchor), a hard censoring guard "
        "(Holm p = 0.076), and an offline sequence search (Holm p = 0.164). "
        "The adopted engine — median-anchored duration estimates, soft "
        "censoring penalty, budget-aware trade-off, rollout lookahead under "
        "CRN — beats the strongest naive rule by +0.0786 ± 0.0093 "
        "(Holm-adjusted p = 6.8e-40, Cohen's d_z = 0.78). The advantage "
        "holds in all 15 case × mapping cells (Figure 23), widening exactly "
        "where the schedule is tightest.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figB_waterfall.png",
        "Figure 22. Ablation ledger from the deterministic baseline to the "
        "two-stage stochastic engine; every bar is an incremental CRN-paired "
        "gain with 95% CI over 450 validation trials.",
        width_in=6.3)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figA_gain3d.png",
        "Figure 23. Paired gain of the refined policy over the baseline "
        "across all 15 case × mapping cells (30 seeds per cell).",
        width_in=6.3)

    insert_para_before(h6, "Upgrade 4 — validation itself is an experimental "
                           "design.", "Normal", bold=True)
    insert_para_before(
        h6,
        "The baseline validated at one operating point with 90 trials. The "
        "refined validation is a full factorial — five official duration "
        "mappings × three cases × 30 seeds, 450 CRN-paired trials per arm — "
        "plus 30 never-used test seeds. The trial-level distribution "
        "(Figure 24) shows the gain is carried by the bulk of trials rather "
        "than outliers, and the main contrast replicates out-of-sample: "
        "+0.1047 ± 0.0098 on test seeds (Holm-adjusted p = 1.0e-54, "
        "d_z = 1.10; Figure 25), with every nominal case clearing the 0.50 "
        "benchmark by a wide margin (0.983/0.856/0.809).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figG_paired_dist.png",
        "Figure 24. Distribution of the 150 trial-level paired gains per "
        "case (grey, jittered); blue diamonds mark the mean ± 95% CI.",
        width_in=6.3)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figF_generalization.png",
        "Figure 25. The main contrast replicates on independent test seeds.",
        width_in=3.2)

    insert_para_before(h6, "Upgrade 5 — report an optimality certificate, "
                           "not a ranking.", "Normal", bold=True)
    insert_para_before(
        h6,
        "Comparing three rules, as the baseline does, cannot distinguish "
        "'best among three' from 'near-optimal'. Sample-average-approximation "
        "bounds bracket the deployed policy class at the nominal mapping: "
        "the lower bound 0.9351 ± 0.0084 for S1 is exact by enumeration, the "
        "candidate lies inside its confidence interval, and the relative "
        "optimality gap is reported as zero (Figure 26; local-search bounds "
        "for S2/S3, flagged).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figE_saa.png",
        "Figure 26. SAA lower/upper bounds for the deployed policy class at "
        "the nominal mapping; the S1 lower bound is exact by enumeration.",
        width_in=3.2)

    insert_para_before(
        h6,
        "Table 15 collects the audit. No upgrade was adopted on intuition: "
        "each was a hypothesis, a paired experiment, and a corrected "
        "p-value. The two conventions that survived falsification — the "
        "ATC-style slack thinking and the WSPT ordering backbone — are "
        "exactly the components the baseline already had; everything the "
        "baseline left untested had to change.")

    srows = stats_rows()
    stbl = insert_table_before(
        h6, doc,
        ["Contrast", "Δ on-time rate (95% CI)", "Wilcoxon p",
         "Holm-adj p", "d_z"],
        srows, None)
    move_caption_above(
        stbl, h6,
        "Table 15. Statistical audit of every contrast. Trial-level paired "
        "differences on weighted on-time rate; Wilcoxon signed-rank test "
        "with Holm correction within each family (validation family of 10, "
        "test family of 2); Cohen's d_z is the standardized mean of paired "
        "differences. [test] marks contrasts on independent test seeds.")

    h8 = find_par(doc, "10. Conclusion", "Heading 1")
    insert_para_before(h8, "9.1 Baseline–refined Comparison", "Heading 2")
    insert_para_before(
        h8,
        "Table 16 summarizes the audit along the four axes where the data or "
        "the theory demanded more than the conventional baseline provides.")

    heads = ["Axis", "Baseline pipeline", "Refined pipeline"]
    rows = [
        ["Information layer",
         "ridge prefix model + selective-review experiment, kept out of the "
         "primary policy; not falsified",
         "zero-review: negative R² across all windows/models; adversarial "
         "buy-top arm loses on validation and test"],
        ["Validation breadth",
         "nominal mapping, 90 trials per rule",
         "full factorial 5 mappings × 3 cases × 30 seeds (450/arm, CRN "
         "paired) + independent 450-trial test"],
        ["Decision mechanism",
         "deterministic ATC score, static family prior",
         "two-stage stochastic engine: robust anchor, soft censoring "
         "penalty, budget trade-off, rollout lookahead; frozen κ, K"],
        ["Optimality evidence",
         "comparison against two other rules",
         "SAA lower/upper bounds; S1 exact enumeration, relative gap ≈ 0"],
    ]
    ctbl = insert_table_before(h8, doc, heads, rows, None)
    move_caption_above(
        ctbl, h8,
        "Table 16. Baseline–refined comparison along the four audited axes.")

    p_ref = find_par(doc, "References", "Heading 1")
    insert_para_before(
        p_ref,
        "Revisiting the same problem with the refined pipeline raises the "
        "nominal weighted on-time rate from 0.871 (baseline ATC) to a "
        "validated, certified configuration that gains +0.079 over the "
        "strongest naive rule on validation and +0.105 on independent test "
        "seeds, buys zero reviews, and carries a zero SAA optimality gap on "
        "the exact S1 bound. The audit structure — falsifiable hypotheses, "
        "CRN-paired experiments, Holm-corrected tests and optimality bounds "
        "— is portable to any stochastic dispatch problem built on replayed "
        "demonstrations.")

    new_refs = [
        "[3] Kleywegt A J, Shapiro A, Homem-de-Mello T. The sample average "
        "approximation method for stochastic discrete optimization. SIAM "
        "Journal on Optimization, 12(2): 479–502, 2002.",
        "[4] Verweij B, Ahmed S, Kleywegt A J, Nemhauser G, Shapiro A. The "
        "sample average approximation method applied to stochastic routing "
        "problems: a computational study. Computational Optimization and "
        "Applications, 24(2–3): 289–333, 2003.",
        "[5] Bertsekas D P, Castañón D A. Rollout algorithms for stochastic "
        "scheduling problems. Journal of Heuristics, 5(1): 89–108, 1999.",
        "[6] Dempster A P, Laird N M, Rubin D B. Maximum likelihood from "
        "incomplete data via the EM algorithm. Journal of the Royal "
        "Statistical Society: Series B, 39(1): 1–38, 1977.",
        "[7] Strum D P, May J H, Vargas L G. Modeling the uncertainty of "
        "surgical procedure times: comparison of log-normal and normal "
        "models. Anesthesiology, 92(4): 1160–1167, 2000.",
        "[8] Adamu M O, Adewumi A O. A survey of single machine scheduling to "
        "minimize weighted number of tardy jobs. Journal of Industrial and "
        "Management Optimization, 10(1): 219–244, 2014.",
        "[9] Holm S. A simple sequentially rejective multiple test "
        "procedure. Scandinavian Journal of Statistics, 6(2): 65–70, 1979.",
        "[10] Wilcoxon F. Individual comparisons by ranking methods. "
        "Biometrics Bulletin, 1(6): 80–83, 1945.",
    ]
    anchor = find_par(doc, "Appendix", "Heading 1")
    for ref in new_refs:
        insert_para_before(anchor, ref, style="Normal")

    doc.save(out)
    post_fix(out)
    print(f"REVISED -> {out}")


TBL_FIXES = [
    ("Holm correction within each family",
     ["3600", "2444", "1400", "1400", "700"]),
    ("Baseline–refined comparison along",
     ["3253", "3253", "3254"]),
]


MEDIA_SWAPS = {
    "word/media/image12.png": "round2/figures3/fig1_workflow.png",
    "word/media/image1.png": "round2/figures3/fig0_route.png",
}


def post_fix(path):
    tmp = path + ".tmp"
    followers = ('<w:semiHidden', '<w:unhideWhenUsed', '<w:qFormat', '<w:locked',
                 '<w:personal ', '<w:personalCompose', '<w:personalReply',
                 '<w:rsid', '<w:pPr', '<w:rPr', '<w:tblPr', '<w:trPr', '<w:tcPr')

    def fix_style_block(m):
        blk = m.group(0)
        um = re.search(r'<w:uiPriority[^>]*/>', blk)
        if not um:
            return blk
        elem = um.group(0)
        blk = blk[:um.start()] + blk[um.end():]
        pos = len(blk)
        for tag in followers:
            t = blk.find(tag)
            if t != -1 and t < pos:
                pos = t
        if pos == len(blk):
            pos = blk.rfind('</w:style>')
        return blk[:pos] + elem + blk[pos:]

    with zipfile.ZipFile(path) as zin:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for n in zin.namelist():
                data = zin.read(n)
                if n in MEDIA_SWAPS:
                    with open(MEDIA_SWAPS[n], "rb") as f:
                        data = f.read()
                if n == "word/settings.xml":
                    s = data.decode("utf-8")
                    s = s.replace("<w:uiCompat97To2003/>", "")
                    data = s.encode("utf-8")
                elif n == "word/styles.xml":
                    s = data.decode("utf-8")
                    s = re.sub(r'<w:style\b.*?</w:style>', fix_style_block,
                               s, flags=re.S)
                    data = s.encode("utf-8")
                elif n == "word/document.xml":
                    s = data.decode("utf-8")
                    s = s.replace(
                        "https://github.com/HnoNoe/shuweibei-2026-problem3",
                        "the source code archive in the supporting materials")
                    s = s.replace(
                        "are publicly available at the team's GitHub repository:",
                        "are publicly available in the supporting materials:")
                    s = s.replace('<w:sz w:val="25"/>', '<w:sz w:val="24"/>')
                    for marker, widths in TBL_FIXES:
                        cap = s.find(marker)
                        assert cap != -1, f"caption not found: {marker}"
                        i = s.rfind("<w:tbl>", 0, cap)
                        assert i != -1, f"tbl not found for: {marker}"
                        seg = s[i:]
                        total = sum(int(w) for w in widths)
                        seg = re.sub(r"<w:tblW[^>]*/>",
                                     f'<w:tblW w:w="{total}" w:type="dxa"/>',
                                     seg, count=1)
                        it = iter(widths)
                        seg = re.sub(r"<w:gridCol[^>]*/>",
                                     lambda m: f'<w:gridCol w:w="{next(it)}"/>',
                                     seg, count=len(widths))
                        s = s[:i] + seg
                    data = s.encode("utf-8")
                zout.writestr(n, data)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
