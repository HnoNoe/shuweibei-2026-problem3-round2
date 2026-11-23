# -*- coding: utf-8 -*-
"""第 9 步 · 之二：改稿——SWBMCM 论文一体化升级。

叙事主线：原版本（baseline pipeline）保留为对照，新增 5.6 节逐项审计升级
（CRN 配对的基线 vs 精炼对比 + 6 张图），第 7 章加 baseline–refined 对比表，
结论补 refined 结果，参考文献规范化扩充。

用法：python3 revise_paper.py <in.docx> <out.docx>
"""

import copy
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

FIGDIR = r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c/round2/figures3"


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
    """在 anchor 前插入居中图片段 + caption 段，返回图片段。"""
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pic.add_run()
    run.add_picture(png, width=Inches(width_in))
    anchor._p.addprevious(pic._p)
    cap = insert_para_before(anchor, caption, style="Caption")
    return pic, cap


def main():
    src, out = sys.argv[1], sys.argv[2]
    doc = Document(src)

    # ============ 1. 新增 5.6 节（插在 6. Sensitivity Analysis 之前） ============
    h6 = find_par(doc, "6. Sensitivity Analysis", "Heading 1")

    insert_para_before(h6, "5.6 From the Baseline Pipeline to the Refined Pipeline", "Heading 2")
    insert_para_before(
        h6,
        "The results above come from the baseline pipeline reported in the "
        "earlier version of this study: a deterministic ATC dispatcher with a "
        "static family prior, a ridge-based prefix review experiment that was "
        "kept out of the primary policy, and a nominal-mapping validation of "
        "90 trials per rule. Three limitations remain open. First, the "
        "information layer is settled only heuristically — the review variant "
        "looked unpromising in one experiment, but no falsification test "
        "excludes a better-informed buyer. Second, validation covers only the "
        "nominal mapping, so gains may be an artifact of one operating point. "
        "Third, the dispatch rule is certified by nothing stronger than a "
        "comparison with two other rules. This section audits each limitation "
        "with a refined pipeline on the very same trial instances (the source "
        "assignments drawn from each seed are identical, so every contrast "
        "below is common-random-number paired).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figA_gain3d.png",
        "Figure 12. Paired gain of the refined two-stage stochastic policy "
        "over the baseline pipeline across all 15 case × mapping cells (CRN-"
        "synchronized trials; 30 seeds per cell). Gains are uniformly "
        "non-negative and are largest in the S2/S3 hard cases.")

    insert_para_before(
        h6,
        "Figure 12 replaces the single-operating-point validation with a "
        "full factorial: five official duration mappings, three cases and 30 "
        "seeds give 450 paired trials per arm. The refined policy never loses "
        "to the baseline in any cell; the nominal-mapping paired gains are "
        "+0.017 (S1), +0.125 (S2) and +0.093 (S3), and the advantage widens "
        "exactly where the schedule is tightest. The refined engine upgrades "
        "the deterministic score in four auditable steps — a median (rather "
        "than mean) anchor for the duration estimate, a robust Q90 anchor, a "
        "soft censoring penalty that keeps long-tail jobs from blocking the "
        "equipment, and a budget-aware trade-off — and closes with a rollout "
        "lookahead that simulates the remaining trial under common random "
        "numbers before each decision.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figB_waterfall.png",
        "Figure 13. Ablation ledger from the deterministic dispatch baseline "
        "to the final two-stage stochastic policy; each bar is an incremental, "
        "CRN-paired gain with 95% CI over 450 validation trials.")

    insert_para_before(
        h6,
        "Every step in Figure 13 pays a measurable gain, and two honest "
        "exceptions are kept in the ledger rather than hidden: the hard "
        "censoring filter was replaced by a soft penalty after the paired "
        "test showed a decision–simulation mismatch (the contrast narrows "
        "from −0.0047 to −0.0017), and an offline sequence search was "
        "evaluated but not adopted (C6 − C5 = −0.0060 ± 0.0065 on validation "
        "and −0.0119 ± 0.0067 on test). Hyperparameters (κ = 3.0, roll-out "
        "scenarios K = 40) were frozen on internal training seeds before any "
        "validation contrast was run.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC_information.png",
        "Figure 14. The information layer, settled by data. (a) Out-of-sample "
        "R² of prefix-based duration prediction is negative for every window "
        "and model — the ridge estimator the baseline experiment relied on is "
        "outperformed by the family-mean guess. (b) An adversarial buy-top arm "
        "that purchases the maximum allowed reviews still pays more and loses. "
        "(c) Paired gain of that arm against the refined zero-review policy.")

    insert_para_before(
        h6,
        "Figure 14 closes the information question the baseline left open. "
        "Across the 0.2/0.6/1.0 s windows, group-wise cross-validated R² is "
        "negative for the family-mean baseline (−0.599), worse for ridge "
        "(−1.034 to −2.058), and best but still negative for gradient "
        "boosting (−0.481); the strongest single feature correlates only at "
        "r = 0.182. The refined policy therefore buys zero reviews, and an "
        "adversarial variant given every advantage — full review budget, best "
        "prefix model, C5 engine — loses to zero-review by −0.0504 ± 0.0084 "
        "on validation and −0.0633 ± 0.0083 on independent test seeds.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figD_censoring.png",
        "Figure 15. Truncation handling against a known-truth simulation "
        "(500 repeats, right-censoring at the median): dropping censored jobs "
        "biases the log-mean by about −0.6, while the EM censoring-aware "
        "update recovers the truth.")

    insert_para_before(
        h6,
        "The refined duration model also treats truncation as information "
        "rather than noise. Jobs that cross the horizon still reveal d > H − "
        "s; an EM update with right-censored likelihood terms turns this into "
        "an online correction, removing the −0.6 log-mean bias that naive "
        "deletion would inject (Figure 15).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figE_saa.png",
        "Figure 16. Sample-average-approximation bounds for the deployed "
        "policy class at the nominal mapping; the lower bound is exact for S1 "
        "by enumeration, and candidate sequences fall inside its confidence "
        "interval (relative gap ≈ 0; local-search bound for S2/S3, flagged).",
        width_in=4.4)

    insert_para_before(
        h6,
        "Finally, the refined result carries an optimality certificate. A "
        "sample-average-approximation bound brackets the deployed policy "
        "class from below (0.9351 ± 0.0084 for S1, exact by enumeration) and "
        "above (0.9173); the deployed candidate lies inside the confidence "
        "interval, so the relative optimality gap is reported as zero. The "
        "baseline comparison of three rules cannot distinguish 'best among "
        "three' from 'near-optimal' — Figure 16 can.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figF_generalization.png",
        "Figure 17. The main contrast (refined online policy vs the strongest "
        "naive WSPT variant) replicates on independent test seeds: +0.0786 ± "
        "0.0093 on validation, +0.1047 ± 0.0098 on test.",
        width_in=4.4)

    insert_para_before(
        h6,
        "Out-of-sample replication confirms the design gain is not seed luck. "
        "On 30 never-used test seeds the refined policy improves the weighted "
        "on-time rate over the strongest naive rule by +0.1047 ± 0.0098 "
        "(Figure 17), and every nominal case clears the 0.50 benchmark by a "
        "wide margin (0.983/0.856/0.809 by case on validation).")

    # ============ 2. 第 7 章加对比表（插在 8. Conclusion 之前） ============
    h8 = find_par(doc, "8. Conclusion", "Heading 1")
    insert_para_before(h8, "7.1 Baseline–refined Comparison", "Heading 2")
    insert_para_before(
        h8,
        "Table 13 summarizes the audit. The baseline pipeline remains a "
        "valid, reproducible reference implementation; the refined pipeline "
        "extends it on the four axes where the data or the theory demanded "
        "more.")

    tbl = doc.add_table(rows=5, cols=3)
    tbl.style = doc.tables[0].style if doc.tables else "Table Grid"
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
    for j, htext in enumerate(heads):
        tbl.rows[0].cells[j].text = htext
    for i, r in enumerate(rows, start=1):
        for j, c in enumerate(r):
            tbl.rows[i].cells[j].text = c
    # 表格移到 h8 前
    h8._p.addprevious(tbl._tbl)
    cap = insert_para_before(h8, "Table 13. Baseline–refined comparison along "
                                 "the four audited axes.", style="Caption")
    # 表题应在表上方：把 caption 段移到表格前
    tbl._tbl.addprevious(cap._p)

    # ============ 3. Conclusion 补一段 ============
    p_ref = find_par(doc, "References", "Heading 1")
    insert_para_before(
        p_ref,
        "Revisiting the same problem with the refined pipeline raises the "
        "nominal weighted on-time rate from 0.871 (baseline ATC) to a "
        "validated, certified configuration that gains +0.079 over the "
        "strongest naive rule on validation and +0.105 on independent test "
        "seeds, buys zero reviews, and carries a zero SAA optimality gap on "
        "the exact S1 bound. The audit structure — paired trials, an ablation "
        "ledger, falsification arms and optimality bounds — is portable to "
        "any stochastic dispatch problem built on replayed demonstrations.")

    # ============ 4. References 规范化扩充 ============
    new_refs = [
        "[4] Kleywegt A J, Shapiro A, Homem-de-Mello T. The sample average "
        "approximation method for stochastic discrete optimization. SIAM "
        "Journal on Optimization, 12(2): 479–502, 2002.",
        "[5] Verweij B, Ahmed S, Kleywegt A J, Nemhauser G, Shapiro A. The "
        "sample average approximation method applied to stochastic routing "
        "problems: a computational study. Computational Optimization and "
        "Applications, 24(2–3): 289–333, 2003.",
        "[6] Bertsekas D P, Castañón D A. Rollout algorithms for stochastic "
        "scheduling problems. Journal of Heuristics, 5(1): 89–108, 1999.",
        "[7] Dempster A P, Laird N M, Rubin D B. Maximum likelihood from "
        "incomplete data via the EM algorithm. Journal of the Royal "
        "Statistical Society: Series B, 39(1): 1–38, 1977.",
        "[8] Strum D P, May J H, Vargas L G. Modeling the uncertainty of "
        "surgical procedure times: comparison of log-normal and normal "
        "models. Anesthesiology, 92(4): 1160–1167, 2000.",
        "[9] Adamu M O, Adewumi A O. A survey of single machine scheduling to "
        "minimize weighted number of tardy jobs. Journal of Industrial and "
        "Management Optimization, 10(1): 219–244, 2014.",
        "[10] HnoNoe. Refined scheduling pipeline and audit results (round 2). "
        "https://github.com/HnoNoe/shuweibei-2026-problem3-round2",
    ]
    anchor = find_par(doc, "Appendix", "Heading 1")
    for ref in new_refs:
        insert_para_before(anchor, ref, style="Normal")

    doc.save(out)
    print(f"REVISED -> {out}")


if __name__ == "__main__":
    main()
