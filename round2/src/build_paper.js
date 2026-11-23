// 第 8 步 · 之四：论文 docx 生成（docx-js 原生，弃用 round1 的 md→html→docx 流水线）。
//
// 输入：docs/paper_content.md（标记：#TITLE / #ABSTRACT / #KEYWORDS / #H1 / ##H2 / ###H3 /
//       !FIG[key]（图占位，题注取 figures/fig_captions.json，按章编号"图 5-1"）/
//       !TCAP 表x-y 文字（表格题注，表上方）/ markdown 表格 / 行内 **加粗**）
// 输出：../数维杯C题_建模论文.docx（shuwei_c 项目根）
// 版式：正文宋体小四(12pt)+Times New Roman，首行缩进 2 字符，1.5 倍行距；
//       H1 三号(16pt)黑体居中，H2 四号(14pt)黑体左，H3 小四(12pt)黑体左；
//       图表题五号(10.5pt)宋体居中（表上、图下），按章编号；
//       页眉 Team # {队号} 右对齐 + Page N of M，页脚页码居中；
//       摘要控制一页，参考文献另起页，目录用 Word TOC 域（打开时刷新）。
// 用法：node build_paper.js <abs output.docx>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel, ImageRun, Packer,
  PageNumber, Paragraph, TabStopType, Table, TableCell, TableRow,
  TableOfContents, TextRun, WidthType,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node build_paper.js /abs/out.docx");

const SRC = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SRC, "..");              // round2/
const PROJ = path.resolve(ROOT, "..");             // shuwei_c/
const MD = fs.readFileSync(path.join(ROOT, "docs/paper_content.md"), "utf-8");
const CAPTIONS = JSON.parse(fs.readFileSync(
  path.join(ROOT, "figures/fig_captions.json"), "utf-8"));

const TEAM_ID = "2026112100001";   // 占位队号，交付前必须换真号
const PAGE_W = 11906, MARGIN = 1440, BODY_W = PAGE_W - 2 * MARGIN;

// ---- 字号（半磅） ----
const SZ = { TITLE: 44, H1: 32, H2: 28, H3: 24, BODY: 24, CAP: 21, SMALL: 18 };

const fontSong = { ascii: "Times New Roman", hAnsi: "Times New Roman",
                   cs: "Times New Roman", eastAsia: "SimSun" };
const fontHei  = { ascii: "Times New Roman", hAnsi: "Times New Roman",
                   cs: "Times New Roman", eastAsia: "SimHei" };

// ---- 行内 **加粗** 解析 ----
function runs(text, base = {}) {
  const out = [];
  for (const seg of text.split(/(\*\*[^*]+\*\*)/g)) {
    if (!seg) continue;
    if (seg.startsWith("**") && seg.endsWith("**")) {
      out.push(new TextRun({ text: seg.slice(2, -2), bold: true, ...base }));
    } else {
      out.push(new TextRun({ text: seg, ...base }));
    }
  }
  return out;
}

const bodyPara = (text) => new Paragraph({
  children: runs(text, { font: fontSong, size: SZ.BODY }),
  indent: { firstLine: 480 },                       // 小四两字符缩进
  spacing: { line: 360, lineRule: "auto" },          // 1.5 倍行距
  alignment: AlignmentType.JUSTIFIED,
});

const h1 = (text) => new Paragraph({
  children: runs(text, { font: fontHei, size: SZ.H1, bold: true }),
  heading: HeadingLevel.HEADING_1,
  alignment: AlignmentType.CENTER,
  spacing: { before: 240, after: 180, line: 360, lineRule: "auto" },
  pageBreakBefore: true,                             // 每章另起页
});

const h2 = (text) => new Paragraph({
  children: runs(text, { font: fontHei, size: SZ.H2, bold: true }),
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 120, line: 360, lineRule: "auto" },
});

const h3 = (text) => new Paragraph({
  children: runs(text, { font: fontHei, size: SZ.H3, bold: true }),
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 160, after: 100, line: 360, lineRule: "auto" },
});

const caption = (text) => new Paragraph({
  children: runs(text, { font: fontSong, size: SZ.CAP }),
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 160, line: 300, lineRule: "auto" },
});

const centered = (text, opts = {}) => new Paragraph({
  children: runs(text, { font: fontSong, size: SZ.BODY, ...opts }),
  alignment: AlignmentType.CENTER,
  spacing: { line: 360, lineRule: "auto", ...opts.spacing },
});

// ---- 章号映射（中文数字 -> 阿拉伯），用于图表按章编号 ----
const CN_NUM = { "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                 "七": 7, "八": 8, "九": 9, "十": 10 };
let chapter = 0, figSeq = 0, tblSeq = 0;

function figBlock(key) {
  figSeq += 1;
  const file = path.join(ROOT, "figures", `${key}.png`);
  const buf = fs.readFileSync(file);
  // 宽高比决定图宽：扁图全页宽，方图 75%
  const dims = pngSize(buf);
  const wide = dims.w / dims.h > 2.2;
  const w = wide ? BODY_W : Math.round(BODY_W * 0.75);
  const h = Math.round(w * dims.h / dims.w);
  const capText = (CAPTIONS[key] || key).replace(/^图\s*/, "");
  return [
    new Paragraph({
      children: [new ImageRun({
        type: "drawing",
        data: buf,
        transformation: { width: Math.round(w / 15), height: Math.round(h / 15) },
      })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 0 },
    }),
    caption(`图 ${chapter}-${figSeq}  ${capText}`),
  ];
}

function pngSize(buf) {
  // PNG: 宽在字节 16-19、高在 20-23（大端）
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function tableBlock(rows, capText) {
  tblSeq += 1;
  const ncol = rows[0].length;
  const w = Math.floor(BODY_W / ncol);
  const trs = rows.map((r, ri) => new TableRow({
    children: r.map((c) => new TableCell({
      children: [new Paragraph({
        children: runs(c, { font: fontSong, size: SZ.CAP }),
        alignment: AlignmentType.CENTER,
        spacing: { line: 300, lineRule: "auto" },
      })],
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      shading: ri === 0 ? { type: "clear", fill: "EEF3F6" } : undefined,
    })),
  }));
  const blocks = [];
  if (capText) blocks.push(caption(capText.replace(/^表\s*/, `表 ${chapter}-${tblSeq}  `)));
  blocks.push(new Table({
    rows: trs,
    width: { size: BODY_W, type: WidthType.DXA },
    columnWidths: Array(ncol).fill(w),
    borders: true,
    alignment: AlignmentType.CENTER,
  }));
  blocks.push(new Paragraph({  // 表后空一行
    children: [], spacing: { after: 120 },
  }));
  return blocks;
}

// ---- 封面 ----
function cover(title) {
  const blank = () => new Paragraph({ children: [], spacing: { after: 200 } });
  return [
    blank(), blank(), blank(),
    centered("2026 年数维杯大学生数学建模竞赛", { size: SZ.H2, bold: true }),
    blank(),
    centered("C 题  Problem 3", { size: SZ.H2, bold: true }),
    blank(), blank(),
    ...title.split("").length
      ? [centered(title, { size: SZ.TITLE, bold: true, font: fontHei,
                           spacing: { line: 480, lineRule: "auto", after: 240 } })]
      : [],
    blank(), blank(), blank(), blank(), blank(),
    centered(`队号：${TEAM_ID}`, { size: SZ.H3, bold: true }),
    blank(), blank(), blank(), blank(),
    centered("2026 年 11 月", { size: SZ.BODY }),
  ];
}

// ---- 主解析 ----
const children = [];
let mode = null;          // title/abstract/keywords
let pendingCap = null;    // !TCAP 等待下一个表格
const lines = MD.split(/\r?\n/);
let titleText = "";

for (const raw of lines) {
  const line = raw.trimEnd();
  const t = line.trim();
  if (!t) { mode = null; continue; }

  if (t.startsWith("#TITLE ")) { titleText = t.slice(7); continue; }
  if (t === "#ABSTRACT") {
    mode = "abstract";
    children.push(new Paragraph({
      children: [new TextRun({ text: "摘  要", font: fontHei, size: SZ.H1,
                               bold: true })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 180 },
    }));
    continue;
  }
  if (t === "#KEYWORDS") { mode = "keywords"; continue; }
  if (t.startsWith("!TCAP ")) { pendingCap = t.slice(6); continue; }
  if (t.startsWith("!FIG[")) {
    children.push(...figBlock(t.slice(5, -1))); continue;
  }
  if (t.startsWith("#H1 ")) {
    const head = t.slice(4);
    const cn = CN_NUM[head[0]];
    if (cn) { chapter = cn; figSeq = 0; tblSeq = 0; }
    // 参考文献/附录另起页由 h1 的 pageBreakBefore 统一处理
    children.push(h1(head)); continue;
  }
  if (t.startsWith("##H2 ")) { children.push(h2(t.slice(5))); continue; }
  if (t.startsWith("###H3 ")) { children.push(h3(t.slice(6))); continue; }
  if (t.startsWith("|")) {
    const rows = [];
    for (const seg of t.replace(/^\|/, "").replace(/\|$/, "").split("|")) {
      rows.push(seg.trim());
    }
    // markdown 表：第一行表头、第二行 --- 分隔；单行表全部当数据行
    if (!children._tblBuf) children._tblBuf = { head: rows, sep: false, body: [] };
    else if (!children._tblBuf.sep && rows.every((c) => /^:?-+:?$/.test(c))) {
      children._tblBuf.sep = true;
    } else if (children._tblBuf.sep) {
      children._tblBuf.body.push(rows);
    } else {
      // 没有分隔行：把上一行降成数据行，当前行当表头
      children._tblBuf.body.push(children._tblBuf.head);
      children._tblBuf.head = rows;
    }
    continue;
  }
  // 遇到非表格行时冲刷表格缓冲
  if (children._tblBuf) {
    const tb = children._tblBuf;
    children._tblBuf = null;
    children.push(...tableBlock([tb.head, ...tb.body], pendingCap));
    pendingCap = null;
  }

  if (mode === "abstract") {
    children.push(new Paragraph({
      children: runs(t, { font: fontSong, size: SZ.BODY }),
      indent: { firstLine: 480 },
      spacing: { line: 360, lineRule: "auto" },
      alignment: AlignmentType.JUSTIFIED,
    }));
  } else if (mode === "keywords") {
    children.push(new Paragraph({
      children: runs(`关键词：${t}`, { font: fontSong, size: SZ.BODY, bold: false }),
      spacing: { line: 360, lineRule: "auto", before: 120 },
    }));
    mode = null;
  } else {
    children.push(bodyPara(t));
  }
}
if (children._tblBuf) {
  const tb = children._tblBuf;
  children.push(...tableBlock([tb.head, ...tb.body], pendingCap));
}

// ---- 参考文献条目（内容稿中 [n] 开头的行渲染为小字悬挂）----
// （已按正文段落处理，若需悬挂缩进可在此特判）

// ---- 文档装配 ----
const doc = new Document({
  creator: "shuwei_c round2",
  title: titleText,
  features: { updateFields: true },   // Word 打开时刷新目录域
  styles: {
    default: {
      document: {
        run: { font: fontSong, size: SZ.BODY },
        paragraph: { spacing: { line: 360, lineRule: "auto" } },
      },
    },
  },
  sections: [{
    properties: {
      titlePage: true,                   // 封面首页不显示页眉页脚
      page: {
        size: { width: PAGE_W, height: 16838 },   // A4
        margin: { top: MARGIN, right: MARGIN,
                  bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      first: new Header({ children: [new Paragraph({ children: [] })] }),
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: `Team # ${TEAM_ID}`, font: fontSong, size: SZ.CAP }),
            new TextRun({ text: "\tPage ", font: fontSong, size: SZ.CAP,
                          children: [PageNumber.CURRENT] }),
            new TextRun({ text: " of ", font: fontSong, size: SZ.CAP }),
            new TextRun({ text: "", font: fontSong, size: SZ.CAP,
                          children: [PageNumber.TOTAL_PAGES] }),
          ],
          alignment: AlignmentType.RIGHT,
          tabStops: [{ type: TabStopType.RIGHT, position: BODY_W }],
        })],
      }),
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({
        children: [new Paragraph({
          children: [new TextRun({ children: [PageNumber.CURRENT],
                                   font: fontSong, size: SZ.CAP })],
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children: [
      // 封面（首页不显示页眉页脚）
      ...cover(titleText),
      new Paragraph({ children: [], pageBreakBefore: true }),
      // 目录（Word 打开时按提示刷新域；updateFields 已开）
      new TableOfContents("目  录", {
        hyperlink: true,
        headingStyleRange: "1-3",
      }),
      new Paragraph({ children: [], pageBreakBefore: true }),
      // 正文（标题/摘要/关键词/章节）
      ...children,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outputPath, buf);
  console.log(`BUILT ${outputPath} (${buf.length} bytes)`);
});
