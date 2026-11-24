import os
import re
import sys
import zipfile

from docx import Document

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def qn(tag):
    return f"{W}{tag}"


def para_text(p):
    return "".join(t.text or "" for t in p.findall(f".//{W}t"))


def find_body_heading(doc, prefix):
    for p in doc.paragraphs:
        if p._p.findall(f".//{W}hyperlink"):
            continue
        if para_text(p._p).strip().startswith(prefix):
            return p._p
    return None


def fix_header(doc):
    for sec in doc.sections:
        for hdr in (sec.header, sec.first_page_header, sec.even_page_header):
            for p in hdr.paragraphs:
                for t in p._p.findall(f".//{W}t"):
                    if "Team # SWBMCM2611888" in (t.text or ""):
                        t.text = t.text.replace("Team # SWBMCM2611888",
                                                "Team CSWBMCM2611888")
                    elif "Team # " in (t.text or ""):
                        t.text = t.text.replace("Team # ", "Team C")


def compress_toc(doc):
    n = 0
    for p in doc.paragraphs:
        if "SWContents" not in p._p.xml:
            continue
        if not p._p.findall(f".//{W}hyperlink"):
            continue
        pPr = p._p.find(qn("pPr"))
        if pPr is None:
            continue
        sp = pPr.find(qn("spacing"))
        if sp is None:
            sp = pPr.makeelement(qn("spacing"), {})
            pPr.append(sp)
        sp.set(qn("line"), "240")
        sp.set(qn("lineRule"), "exact")
        sp.set(qn("before"), "0")
        sp.set(qn("after"), "0")
        n += 1
    return n


def drop_toc_entries(doc, prefixes):
    drop = []
    for p in doc.paragraphs:
        if "SWContents" not in p._p.xml:
            continue
        if not p._p.findall(f".//{W}hyperlink"):
            continue
        txt = para_text(p._p).strip()
        if any(txt.startswith(pre) for pre in prefixes):
            drop.append(p._p)
    for el in drop:
        el.getparent().remove(el)
    return len(drop)


def retitle(p_el, new_text):
    ts = p_el.findall(f".//{W}t")
    if ts:
        ts[0].text = new_text
        for t in ts[1:]:
            t.text = ""


def move_block(doc):
    h74 = find_body_heading(doc, "7.4 Comparison Across")
    h75 = find_body_heading(doc, "7.5 Case-level")
    h76 = find_body_heading(doc, "7.6 From a Conventional")
    h8 = find_body_heading(doc, "8. Sensitivity")
    h_appc = find_body_heading(doc, "Appendix C")
    assert h74 is not None and h75 is not None and h8 is not None \
        and h_appc is not None, "move anchors missing"
    body = h74.getparent()
    kids = list(body)
    i0 = kids.index(h74)
    # 上界用 7.6 标题而不是 8：否则 7.4~8 之间的 7.6 叙事会被一起扫进附录
    i1 = kids.index(h76) if h76 is not None and kids.index(h76) > i0 \
        else kids.index(h8)
    block = kids[i0:i1]
    for el in block:
        body.remove(el)
    anchor = find_body_heading(doc, "Appendix C")
    for el in block:
        anchor.addprevious(el)
    retitle(h74, "B.1 Comparison Across Dispatch Policies")
    retitle(h75, "B.2 Case-level Validation and Paired Variability")
    lead = block[0].makeelement(qn("p"), {})
    r = lead.makeelement(qn("r"), {})
    t = r.makeelement(qn("t"), {})
    t.text = ("The dispatch-policy comparison and the case-level validation "
              "summarised in Sections 7.3 and 8 are collected here in full.")
    r.append(t)
    lead.append(r)
    h74.addprevious(lead)
    return len(block)


def strip_bad_settings(dst):
    tmp = dst + ".tmp"
    with zipfile.ZipFile(dst) as zin:
        names = zin.namelist()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                if name == "word/settings.xml":
                    xml = data.decode("utf-8")
                    xml = re.sub(r"<w:uiCompat97To2003[^>]*/>", "", xml)
                    data = xml.encode("utf-8")
                zout.writestr(name, data)
    os.replace(tmp, dst)


def main(src, dst):
    doc = Document(src)
    fix_header(doc)
    n_toc = compress_toc(doc)
    n_drop = drop_toc_entries(doc, ["7.4 Comparison", "7.5 Case-level"])
    n_blk = move_block(doc)
    doc.save(dst)
    strip_bad_settings(dst)
    print(f"header fixed, toc compressed x{n_toc}, toc entries dropped x{n_drop}, "
          f"moved elements x{n_blk}, uiCompat97To2003 stripped")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
