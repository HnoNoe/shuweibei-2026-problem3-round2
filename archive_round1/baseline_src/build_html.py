"""从 final_draft.md 生成符合 html_rules.md 的完整 HTML（academic-paper）。

层级映射：md 文档标题(#) → 封面承载不重复；md 章(##) → html h1；md 节(###) → html h2。
"""
import re, html
from pathlib import Path

SRC = Path("output/q3-20261121/stage1/final_draft.md")
OUT = Path("output/q3-20261121/stage2/formatted-数维杯C题建模论文.html")

lines = SRC.read_text(encoding="utf-8").split("\n")

# ---- 解析 markdown ----
blocks = []  # dict
i = 0
while i < len(lines):
    ln = lines[i]
    if not ln.strip():
        i += 1; continue
    m = re.match(r'^(#{1,4})\s+(.*)', ln)
    if m:
        blocks.append(dict(t="h", text=m.group(2).strip(), md_level=len(m.group(1))))
        i += 1; continue
    if ln.strip().startswith("|"):
        table = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            table.append(lines[i].strip()); i += 1
        blocks.append(dict(t="table", rows=table)); continue
    if ln.strip().startswith("!["):
        m = re.match(r'!\[(.*?)\]\((.*?)\)', ln.strip())
        blocks.append(dict(t="img", alt=m.group(1), src=m.group(2)))
        i += 1; continue
    para = [ln.strip()]
    i += 1
    while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,4})\s|^\||^!\[', lines[i].strip()):
        para.append(lines[i].strip()); i += 1
    blocks.append(dict(t="p", text=" ".join(para)))

# ---- 确定 HTML 层级与 id，同时构建目录 ----
h1_no = 0
h2_no = 0
toc = []  # (level, text, id)
for b in blocks:
    if b["t"] != "h":
        continue
    ml = b["md_level"]
    if ml == 1:
        # 文档标题由封面承载，正文不输出，也不进目录
        b["skip"] = True
        continue
    if ml == 2:
        h1_no += 1
        h2_no = 0
        b["html_level"] = 1
        b["id"] = f"section-{h1_no}"
    elif ml == 3:
        h1_no = max(h1_no, 1)
        h2_no += 1
        b["html_level"] = 2
        b["id"] = f"section-{h1_no}-{h2_no}"
    else:  # md h4 -> html h3
        b["html_level"] = 3
        b["id"] = f"section-{h1_no}-{h2_no}-{b['id'] if 'id' in b else h2_no}"
    toc.append((b["html_level"], b["text"], b["id"]))

# ---- 行内格式 ----
def inline(s):
    s = html.escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
    s = re.sub(r'`(.+?)`', r'<code>\1</code>', s)
    return s

# ---- 正文 ----
LEAD = {
    "二、问题分析": "本章按照三个核心问题依次展开分析，并在最后给出整体技术路线。",
    "六、模型的建立与求解": "本章针对三个核心问题分别建立模型并给出求解结果。",
    "七、模型的评价及优化": "本章从误差与合规、优点、缺点和推广四个方面对模型进行评价。",
    "6.1 问题一：时长估计与不确定性模型": "本节建立作业族分层的对数正态先验模型，并以前缀强度进行弱贝叶斯更新。",
    "6.2 问题二：调度决策模型": "本节推导预留时长的占优性质，并建立 WSPT 滚动时域调度策略。",
    "6.3 问题三：多目标权衡与约束检验": "本节在多组参数与多种映射下检验策略的多目标表现与约束满足情况。",
}
body = []
last_h1 = None
last_h2 = None
for b in blocks:
    t = b["t"]
    if t == "h":
        if b.get("skip"):
            continue
        lvl = b["html_level"]
        txt = inline(b["text"])
        sid = b["id"]
        # h1 后紧跟 h2 时补一句章导语，避免"章标题直接接节标题"
        if lvl == 2 and last_h1 is not None:
            lead = LEAD.get(last_h1, "本章对相关内容作具体展开。")
            body.append(f'<p style="font-family:var(--ff-body);font-size:var(--fs-body);line-height:var(--lh-body);text-indent:2em;text-align:justify;margin-bottom:var(--spacing-paragraph)">{inline(lead)}</p>')
            last_h1 = None
        if lvl == 3 and last_h2 is not None:
            lead = LEAD.get(last_h2, "本节对相关内容作具体展开。")
            body.append(f'<p style="font-family:var(--ff-body);font-size:var(--fs-body);line-height:var(--lh-body);text-indent:2em;text-align:justify;margin-bottom:var(--spacing-paragraph)">{inline(lead)}</p>')
            last_h2 = None
        # academic-paper 法定要素：摘要区/参考文献区加语义 class
        extra = ""
        if b["text"].strip() == "摘要":
            extra = ' class="abstract"'
        elif b["text"].strip() == "参考文献":
            extra = ' class="references"'
        if lvl == 1:
            body.append(f'<h1 id="{sid}"{extra} style="font-size:var(--fs-h1);font-weight:var(--fw-bold);color:var(--color-primary);margin-top:var(--spacing-section);margin-bottom:var(--spacing-block)">{txt}</h1>')
            last_h1 = re.sub(r'\*\*', '', b["text"]).strip()
        elif lvl == 2:
            body.append(f'<h2 id="{sid}" style="font-size:var(--fs-h2);font-weight:var(--fw-bold);color:var(--color-primary);margin-top:var(--spacing-block);margin-bottom:var(--spacing-paragraph)">{txt}</h2>')
            last_h2 = re.sub(r'\*\*', '', b["text"]).strip()
        else:
            body.append(f'<h3 id="{sid}" style="font-size:var(--fs-h3);font-weight:var(--fw-bold);color:var(--color-text);margin-top:var(--spacing-block);margin-bottom:var(--spacing-paragraph)">{txt}</h3>')
    elif t == "p":
        raw = b["text"]
        # 图题行：图 X-X 标题（path） -> 生成图片 + 居中图题（五号）
        mfig = re.match(r'^(图\s*\d+-\d+.*?)（([^）]+\.(?:png|jpg|jpeg))）$', raw)
        if mfig:
            cap, src = mfig.group(1).strip(), mfig.group(2)
            if not src.startswith(("/", "file://")):
                src = "C:/Users/Camevian/WorkBuddy/2026-11-21-08-56-37/" + src
            body.append(f'<figure style="margin:var(--spacing-block) 0;text-align:center">'
                        f'<img src="{src}" alt="{html.escape(cap)}" width="460" style="max-width:100%">'
                        f'<figcaption style="font-size:var(--fs-caption);font-family:var(--ff-body);text-align:center;margin-top:0.4em">{inline(cap)}</figcaption></figure>')
            continue
        # 表题行：表 X-X 标题 -> 居中表题（五号），置于表格上方
        if re.match(r'^表\s*\d+-\d+', raw):
            body.append(f'<p class="table-caption" style="font-size:var(--fs-caption);font-family:var(--ff-body);text-align:center;font-weight:var(--fw-bold);margin:var(--spacing-block) 0 0.4em">{inline(raw)}</p>')
            continue
        txt = inline(raw)
        body.append(f'<p style="font-family:var(--ff-body);font-size:var(--fs-body);line-height:var(--lh-body);text-indent:2em;text-align:justify;margin-bottom:var(--spacing-paragraph)">{txt}</p>')
    elif t == "table":
        rows = b["rows"]
        data_rows = [r for r in rows if not re.match(r'^\|[\s\-:|]+\|$', r)]
        if not data_rows:
            continue
        def cells(r):
            return [c.strip() for c in r.strip().strip("|").split("|")]
        head = cells(data_rows[0])
        htm = ['<table class="data-table" style="width:100%;border-collapse:collapse;margin:var(--spacing-block) 0">']
        htm.append('<thead><tr>' + ''.join(f'<th style="border:1px solid var(--color-border);background:var(--color-bg);padding:6px;font-size:var(--fs-small);font-weight:var(--fw-bold);text-align:center">{inline(c)}</th>' for c in head) + '</tr></thead>')
        htm.append('<tbody>')
        for r in data_rows[1:]:
            cs = cells(r)
            htm.append('<tr>' + ''.join(f'<td style="border:1px solid var(--color-border);padding:6px;font-size:var(--fs-small);text-align:left">{inline(c)}</td>' for c in cs) + '</tr>')
        htm.append('</tbody></table>')
        body.append(''.join(htm))
    elif t == "img":
        body.append(f'<figure><p style="text-align:center"><img src="{b["src"]}" alt="{html.escape(b["alt"])}" width="460" style="max-width:100%"></p></figure>')

body_html = "\n".join(body)

# ---- 目录（正确嵌套）----
toc_items = []
i = 0
while i < len(toc):
    lvl, txt, sid = toc[i]
    item = f'<li><a href="#{sid}">{inline(txt)}</a>'
    # 收集紧随其后的更深层级条目
    subs = []
    j = i + 1
    while j < len(toc) and toc[j][0] > lvl:
        subs.append(toc[j]); j += 1
    if subs:
        item += '<ol class="toc-list">'
        k = 0
        while k < len(subs):
            sl, st, ss = subs[k]
            item += f'<li><a href="#{ss}">{inline(st)}</a></li>'
            k += 1
        item += '</ol>'
    item += '</li>'
    toc_items.append(item)
    i = j
toc_html = "\n".join(toc_items)

doc = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="docx-page-size" content="A4">
<title>数维杯C题：补货作业时长估计与随机调度建模论文</title>
<style>
:root {{
  --typography-fontFamily-heading: "SimHei","黑体",sans-serif;
  --typography-fontFamily-body: "SimSun","宋体",serif;
  --typography-fontSize-h1: 16pt;
  --typography-fontSize-h2: 14pt;
  --typography-fontSize-h3: 12pt;
  --typography-fontSize-body: 12pt;
  --typography-fontSize-small: 10.5pt;
  --typography-lineHeight-body: 1.6;
  --typography-fontWeight-bold: 700;
  --typography-fontWeight-normal: 400;
  --fs-h1: var(--typography-fontSize-h1);
  --fs-h2: var(--typography-fontSize-h2);
  --fs-h3: var(--typography-fontSize-h3);
  --fs-body: var(--typography-fontSize-body);
  --fs-small: var(--typography-fontSize-small);
  --fs-caption: 10.5pt;
  --fs-header: 9pt;
  --ff-heading: var(--typography-fontFamily-heading);
  --ff-body: var(--typography-fontFamily-body);
  --ff-mono: monospace;
  --lh-body: var(--typography-lineHeight-body);
  --fw-bold: var(--typography-fontWeight-bold);
  --fw-normal: var(--typography-fontWeight-normal);
  --color-primary: #1F4E79;
  --color-text: #212121;
  --color-muted: #616161;
  --color-border: #BDBDBD;
  --color-bg: #EEF3F8;
  --fs-cover-title: 26pt;
  --spacing-paragraph: 0.7em;
  --spacing-section: 1.8em;
  --spacing-block: 1.0em;
  --margin-page: 2.5cm;
  --page-content-width: 16.0cm;
}}
@page {{ margin: var(--margin-page); @top-center {{ content: "Team # 2026112100001   Page " counter(page); font-size:var(--fs-header); font-family:var(--ff-body); color:var(--color-muted); }} @bottom-center {{ content: counter(page); }} }}
@page cover {{ @top-center {{ content: none; }} @bottom-center {{ content: none; }} }}
section[role="cover"] {{ page: cover; }}
body {{ max-width: var(--page-content-width); margin-left:auto; margin-right:auto; font-family:var(--ff-body); }}
.cover-title {{ text-align:center; font-family:var(--ff-heading); font-size:var(--fs-cover-title); font-weight:var(--fw-bold); color:var(--color-primary); margin-top:5cm; line-height:1.4; }}
.cover-sub {{ text-align:center; font-family:var(--ff-heading); font-size:var(--fs-h2); color:var(--color-muted); margin-top:1.5cm; }}
.cover-meta {{ text-align:center; font-family:var(--ff-body); font-size:var(--fs-body); color:var(--color-muted); margin-top:3cm; }}
.toc-title {{ text-align:center; font-family:var(--ff-heading); font-size:var(--fs-h1); font-weight:var(--fw-bold); color:var(--color-primary); margin-bottom:var(--spacing-block); }}
.toc-list, .toc-list ol, .toc-list ul {{ list-style:none; }}
.toc-list li {{ font-family:var(--ff-body); font-size:var(--fs-body); line-height:1.9; }}
.toc-list ol li {{ font-size:var(--fs-small); color:var(--color-muted); padding-left:1.5em; }}
.toc-list a {{ color:var(--color-text); text-decoration:none; }}
h1,h2,h3 {{ font-family:var(--ff-heading); }}
p {{ font-family:var(--ff-body); }}
code {{ font-family:var(--ff-mono); font-size:var(--fs-small); }}
figure {{ margin:var(--spacing-block) 0; }}
</style>
</head>
<body>
<section role="cover">
  <p class="cover-title">数维杯 C 题<br>补货作业时长估计与随机调度决策</p>
  <p class="cover-sub">基于第一人称演示数据的建模与求解</p>
  <p class="cover-meta">建模论文<br>2026 年 11 月</p>
</section>
<section role="cover">
<nav class="doc-toc" aria-label="文档目录">
  <p class="toc-title">目录</p>
  <ol class="toc-list">
{toc_html}
  </ol>
</nav>
</section>
<section role="body" data-page-restart="1">
{body_html}
</section>
</body>
</html>
'''
OUT.write_text(doc, encoding="utf-8")
print("written; toc items", len(toc), "blocks", len(blocks), "anchors", doc.count('<a href="#section'))
