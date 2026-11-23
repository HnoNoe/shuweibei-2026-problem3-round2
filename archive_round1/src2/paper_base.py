# -*- coding: utf-8 -*-
"""构建数维杯 C 题参赛论文 DOCX（原生 python-docx 排版，不用 HTML 中间层）。

格式依据 output/refs 官方规范：
- 摘要专用页（题目三号黑体、摘  要三号黑体加粗、内容宋体小四、关键词四号黑体）
- 目录（域，打开 Word 自动更新）
- 正文宋体小四、1.5 倍行距；标题三号黑体；图表按章编号；表题上方、图题下方、五号宋体居中
- 页眉 Team # + Page X of Y（域）；页脚居中页码（正文从 1 起编）
- 参考文献另起页（5 号宋体）；附录另起页（5 号）
"""
import os
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(r"C:/Users/Camevian/Documents/Kimi/Workspaces/大学学习/shuwei_c")
FIG = ROOT / "figures2"
OUT = ROOT / "数维杯C题_论文_重做版.docx"
TEAM = "2026112100001"   # 占位队号，提交前替换

SONG, HEI, TIMES = "宋体", "黑体", "Times New Roman"


def set_run(run, cn=SONG, en=TIMES, size=12, bold=False, italic=False,
            color=None):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), cn)


def para(doc, text="", cn=SONG, en=TIMES, size=12, bold=False,
         align=None, indent_chars=2, line=1.5, before=0, after=0,
         style=None):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if line:
        pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        set_run(p.add_run(text), cn, en, size, bold)
    return p


def h1(doc, text):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(13)
    set_run(p.add_run(text), HEI, TIMES, 16, bold=False)
    return p


def h2(doc, text):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    set_run(p.add_run(text), HEI, TIMES, 14, bold=False)
    return p


def h3(doc, text):
    p = doc.add_paragraph(style="Heading 3")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    set_run(p.add_run(text), HEI, TIMES, 12, bold=False)
    return p


def figure(doc, img, caption, width_cm=14.5, chap="5"):
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
             line=1.0, before=6, after=0)
    p.add_run().add_picture(str(img), width=Cm(width_cm))
    c = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
             line=1.0, before=3, after=10)
    set_run(c.add_run(caption), SONG, TIMES, 10.5)


def table(doc, caption, headers, rows, chap="5", size=10.5,
          align_first_left=False):
    c = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
             line=1.0, before=8, after=3)
    set_run(c.add_run(caption), SONG, TIMES, size)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(headers):
        cell = t.cell(0, j)
        cell.paragraphs[0].paragraph_format.alignment = \
            WD_ALIGN_PARAGRAPH.CENTER
        set_run(cell.paragraphs[0].add_run(str(htxt)), SONG, TIMES, size)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.paragraphs[0].paragraph_format.alignment = \
                WD_ALIGN_PARAGRAPH.LEFT if (align_first_left and j == 0) \
                else WD_ALIGN_PARAGRAPH.CENTER
            set_run(cell.paragraphs[0].add_run(str(v)), SONG, TIMES, size)
    para(doc, indent_chars=0, line=1.0, before=0, after=6)
    return t


def formula(doc, expr, number):
    """公式段：居中式子 + 右端编号。_{x} 记法渲染为真下标。"""
    import re
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
             line=1.5, before=4, after=4)
    tab = p.paragraph_format.tab_stops
    tab.add_tab_stop(Cm(7.25))
    tab.add_tab_stop(Cm(14.5), 2)  # 2 = right
    for seg in re.split(r"(_\{[^}]*\})", expr):
        if not seg:
            continue
        if seg.startswith("_{"):
            r = p.add_run(seg[2:-1])
            set_run(r, SONG, TIMES, 12)
            r.font.subscript = True
        else:
            set_run(p.add_run(seg), SONG, TIMES, 12)
    set_run(p.add_run("\t\t(" + number + ")"), SONG, TIMES, 12)


def field(p, instr):
    """插入 Word 域（PAGE / NUMPAGES / TOC）。begin/instr/end 必须分属不同 run。"""
    r1 = p.add_run()
    fb = OxmlElement("w:fldChar"); fb.set(qn("w:fldCharType"), "begin")
    r1._element.append(fb)
    r2 = p.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = instr
    r2._element.append(it)
    r3 = p.add_run()
    fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end")
    r3._element.append(fe)
    return r2


def body_header_footer(section):
    """正文节页眉 Team # + Page X of Y；页脚居中页码；正文从 1 起编。"""
    # 起始页码 1
    sectPr = section._sectPr
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:start"), "1")
    sectPr.append(pg)
    hd = section.header
    hd.is_linked_to_previous = False
    hp = hd.paragraphs[0]
    hp.paragraph_format.tab_stops.add_tab_stop(Cm(14.5), 2)
    set_run(hp.add_run(f"Team # {TEAM}"), SONG, TIMES, 9)
    hp.add_run("\t")
    set_run(hp.add_run("Page "), SONG, TIMES, 9)
    fr = field(hp, "PAGE")
    set_run(hp.add_run(" of "), SONG, TIMES, 9)
    field(hp, "NUMPAGES")
    for r in hp.runs:
        set_run(r, SONG, TIMES, 9)
    ft = section.footer
    ft.is_linked_to_previous = False
    fp = ft.paragraphs[0]
    fp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr2 = field(fp, "PAGE")
    for r in fp.runs:
        set_run(r, SONG, TIMES, 9)


def enable_update_fields(doc):
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)
