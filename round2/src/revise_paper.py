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


def main():
    src, out = sys.argv[1], sys.argv[2]
    doc = Document(src)

    h6 = find_par(doc, "6. Sensitivity Analysis", "Heading 1")

    insert_para_before(h6, "5.6 From a Conventional Baseline to a Certified "
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
        "contrasts. Figure 12 maps the argument; Table 13 reports the full "
        "statistical audit.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figZ_flagship.png",
        "Figure 12. The audit at a glance. Each row confronts a conventional "
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
        "(Figure 13) — and the strongest single prefix feature correlates "
        "only at r = 0.182. Prediction is therefore rejected as a design "
        "basis. Because buying reviews is only rational under predictable "
        "durations, the review route itself must be falsified rather than "
        "assumed away: an adversarial arm given every advantage (full review "
        "budget, best prefix model, C5 engine) pays +4.64 more per trial "
        "(Figure 14) and is significantly worse than zero-review, "
        "−0.0504 ± 0.0084 on validation (Holm-adjusted p = 9.9e-27) and "
        "−0.0633 ± 0.0083 on independent test seeds (p = 2.8e-34) "
        "(Figure 15). The refined policy buys zero reviews.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC1_r2.png",
        "Figure 13. Out-of-sample R² of prefix-based duration prediction for "
        "three prefix windows and three estimators; all values are negative.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC2_falsified.png",
        "Figure 14. The falsified information route: an adversarial arm "
        "allowed to buy the maximum number of reviews pays +4.64 more per "
        "trial than zero-review and still loses on timeliness.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figC3_adversarial.png",
        "Figure 15. Paired gain of the adversarial buy-top arm against the "
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
        "small samples (Figure 16). Censoring, meanwhile, is not missing "
        "data: a job that crosses the horizon still reveals d > H − s. In a "
        "known-truth simulation with right-censoring at the median, naive "
        "deletion biases the log-mean by −0.609 (n = 124), while the EM "
        "censoring-aware update brings the bias to −0.008 (Figure 17). "
        "Every downstream decision inherits this correction online.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figH_duration_model.png",
        "Figure 16. Fitted hierarchical log-normal duration model (σ floored "
        "at 0.35). Family L is heavy-tailed and drives the robust-anchor "
        "design choice.",
        width_in=3.2)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figD_censoring.png",
        "Figure 17. Truncation handling against known truth (500 repeats, "
        "right-censoring at the median): naive deletion biases the log-mean "
        "by about −0.6; EM recovers it.",
        width_in=6.3)

    insert_para_before(h6, "Upgrade 3 — the decision layer must reason under "
                           "the distribution, not through a point estimate.",
                       "Normal", bold=True)
    insert_para_before(
        h6,
        "With prediction rejected, performance can only come from better "
        "decisions under residual uncertainty. The ledger in Figure 18 "
        "decomposes the full gain into single-design paired contrasts; "
        "Table 13 gives the corresponding tests. Three components were "
        "rejected and are kept on record: a robust Q90 anchor in isolation "
        "(−0.0178 against the median anchor), a hard censoring guard "
        "(Holm p = 0.076), and an offline sequence search (Holm p = 0.164). "
        "The adopted engine — median-anchored duration estimates, soft "
        "censoring penalty, budget-aware trade-off, rollout lookahead under "
        "CRN — beats the strongest naive rule by +0.0786 ± 0.0093 "
        "(Holm-adjusted p = 6.8e-40, Cohen's d_z = 0.78). The advantage "
        "holds in all 15 case × mapping cells (Figure 19), widening exactly "
        "where the schedule is tightest.")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figB_waterfall.png",
        "Figure 18. Ablation ledger from the deterministic baseline to the "
        "two-stage stochastic engine; every bar is an incremental CRN-paired "
        "gain with 95% CI over 450 validation trials.",
        width_in=6.3)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figA_gain3d.png",
        "Figure 19. Paired gain of the refined policy over the baseline "
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
        "(Figure 20) shows the gain is carried by the bulk of trials rather "
        "than outliers, and the main contrast replicates out-of-sample: "
        "+0.1047 ± 0.0098 on test seeds (Holm-adjusted p = 1.0e-54, "
        "d_z = 1.10; Figure 21), with every nominal case clearing the 0.50 "
        "benchmark by a wide margin (0.983/0.856/0.809).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figG_paired_dist.png",
        "Figure 20. Distribution of the 150 trial-level paired gains per "
        "case (grey, jittered); blue diamonds mark the mean ± 95% CI.",
        width_in=6.3)
    insert_figure_before(
        h6, doc, f"{FIGDIR}/figF_generalization.png",
        "Figure 21. The main contrast replicates on independent test seeds.",
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
        "optimality gap is reported as zero (Figure 22; local-search bounds "
        "for S2/S3, flagged).")

    insert_figure_before(
        h6, doc, f"{FIGDIR}/figE_saa.png",
        "Figure 22. SAA lower/upper bounds for the deployed policy class at "
        "the nominal mapping; the S1 lower bound is exact by enumeration.",
        width_in=3.2)

    insert_para_before(
        h6,
        "Table 13 collects the audit. No upgrade was adopted on intuition: "
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
        "Table 13. Statistical audit of every contrast. Trial-level paired "
        "differences on weighted on-time rate; Wilcoxon signed-rank test "
        "with Holm correction within each family (validation family of 10, "
        "test family of 2); Cohen's d_z is the standardized mean of paired "
        "differences. [test] marks contrasts on independent test seeds.")

    h8 = find_par(doc, "8. Conclusion", "Heading 1")
    insert_para_before(h8, "7.1 Baseline–refined Comparison", "Heading 2")
    insert_para_before(
        h8,
        "Table 14 summarizes the audit along the four axes where the data or "
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
        "Table 14. Baseline–refined comparison along the four audited axes.")

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
        "[10] Holm S. A simple sequentially rejective multiple test "
        "procedure. Scandinavian Journal of Statistics, 6(2): 65–70, 1979.",
        "[11] Wilcoxon F. Individual comparisons by ranking methods. "
        "Biometrics Bulletin, 1(6): 80–83, 1945.",
        "[12] HnoNoe. Refined scheduling pipeline and audit results (round "
        "2). https://github.com/HnoNoe/shuweibei-2026-problem3-round2",
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
