"""
Span/block merging for layout + OCR + formula LaTeX → readable text.

Ported from PDF-Extract-Kit / MinerU merge logic, adapted for ai-tutor-backend.
"""

from __future__ import annotations

import re
from typing import Any


def _overlap_y_ratio(bbox1: list[float], bbox2: list[float], threshold: float = 0.8) -> bool:
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2
    overlap = max(0.0, min(y1_1, y1_2) - max(y0_1, y0_2))
    h1, h2 = y1_1 - y0_1, y1_2 - y0_2
    if min(h1, h2) <= 0:
        return False
    return (overlap / min(h1, h2)) > threshold


def _overlap_area_ratio(span_bbox: list[float], block_bbox: list[float]) -> float:
    x_left = max(span_bbox[0], block_bbox[0])
    y_top = max(span_bbox[1], block_bbox[1])
    x_right = min(span_bbox[2], block_bbox[2])
    y_bottom = min(span_bbox[3], block_bbox[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    block_area = max(1.0, (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1]))
    return inter / block_area


def _merge_spans_to_line(spans: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s["bbox"][1])
    lines: list[list[dict[str, Any]]] = []
    current = [spans[0]]
    for span in spans[1:]:
        if span["type"] in ("isolated",) or any(s["type"] == "isolated" for s in current):
            lines.append(current)
            current = [span]
            continue
        if _overlap_y_ratio(span["bbox"], current[-1]["bbox"]):
            current.append(span)
        else:
            lines.append(current)
            current = [span]
    if current:
        lines.append(current)
    return lines


def _line_sort_spans(lines: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        line.sort(key=lambda s: s["bbox"][0])
        line_bbox = [
            min(s["bbox"][0] for s in line),
            min(s["bbox"][1] for s in line),
            max(s["bbox"][2] for s in line),
            max(s["bbox"][3] for s in line),
        ]
        out.append({"bbox": line_bbox, "spans": line})
    return out


def _fix_text_block(block: dict[str, Any]) -> dict[str, Any]:
    for span in block["spans"]:
        if span["type"] == "isolated":
            span["type"] = "inline"
    block["lines"] = _line_sort_spans(_merge_spans_to_line(block["spans"]))
    del block["spans"]
    return block


def _fix_interline_block(block: dict[str, Any]) -> dict[str, Any]:
    block["lines"] = _line_sort_spans(_merge_spans_to_line(block["spans"]))
    del block["spans"]
    return block


def fill_spans_in_blocks(
    blocks: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    overlap_ratio: float = 0.6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remaining = list(spans)
    block_with_spans: list[dict[str, Any]] = []
    for block in blocks:
        poly = block["poly"]
        block_bbox = [min(poly[0], poly[2]), min(poly[1], poly[5]), max(poly[2], poly[4]), max(poly[1], poly[5])]
        block_dict = {
            "type": block["category_type"],
            "bbox": block_bbox,
            "saved_info": block,
            "spans": [],
        }
        matched: list[dict[str, Any]] = []
        for span in remaining:
            if _overlap_area_ratio(span["bbox"], block_bbox) > overlap_ratio:
                matched.append(span)
        block_dict["spans"] = matched
        block_with_spans.append(block_dict)
        for span in matched:
            remaining.remove(span)
    return block_with_spans, remaining


def fix_block_spans(block_with_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed: list[dict[str, Any]] = []
    for block in block_with_spans:
        if block["type"] == "isolate_formula":
            fixed.append(_fix_interline_block(block))
        else:
            fixed.append(_fix_text_block(block))
    return fixed


def _detect_lang(text: str) -> str:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _escape_md(text: str) -> str:
    for ch in ("*", "`", "~", "$"):
        text = text.replace(ch, "\\" + ch)
    return text


def merge_para_with_text(para_block: dict[str, Any]) -> str:
    para_text = ""
    for line in para_block["lines"]:
        line_text = "".join(s["content"].strip() for s in line["spans"] if s["type"] == "text")
        line_lang = _detect_lang(line_text) if line_text else "en"
        for span in line["spans"]:
            stype = span["type"]
            content = ""
            if stype == "text":
                content = _escape_md(span["content"])
            elif stype in ("inline", "ignore-formula"):
                content = f" ${span['content'].strip('$')}$ "
            elif stype == "isolated":
                content = f"\n$$\n{span['content'].strip('$')}\n$$\n"
            if not content:
                continue
            if line_lang == "zh":
                para_text += content.strip()
            else:
                para_text += content.strip() + " "
    return para_text.strip()


def latex_rm_whitespace(s: str) -> str:
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter, noletter = "[a-zA-Z]", r"[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda _: str(names.pop(0)), s)
    while True:
        news = s
        news = re.sub(rf"(?!\\ )({noletter})\s+?({noletter})", r"\1\2", s)
        news = re.sub(rf"(?!\\ )({noletter})\s+?({letter})", r"\1\2", news)
        news = re.sub(rf"({letter})\s+?({noletter})", r"\1\2", news)
        if news == s:
            break
        s = news
    return s


def page_to_markdown(layout_dets: list[dict[str, Any]]) -> str:
    blocks: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    text_block_types = {
        "title", "plain text", "figure_caption", "table_caption",
        "table_footnote", "isolate_formula", "formula_caption",
    }
    for item in layout_dets:
        cat = item["category_type"]
        if cat in ("inline", "text", "isolated"):
            key = "text" if cat == "text" else "latex"
            xmin, ymin, _, _, xmax, ymax, _, _ = item["poly"]
            spans.append({
                "type": cat,
                "bbox": [xmin, ymin, xmax, ymax],
                "content": item.get(key, ""),
            })
            if cat == "isolated":
                item = {**item, "category_type": "isolate_formula"}
                blocks.append(item)
        else:
            blocks.append(item)

    need_fix = [b for b in blocks if b["category_type"] in text_block_types]
    final_block = [b for b in blocks if b["category_type"] not in text_block_types]
    block_with_spans, _ = fill_spans_in_blocks(need_fix, spans, 0.6)
    for para in fix_block_spans(block_with_spans):
        result = merge_para_with_text(para)
        if para["type"] == "isolate_formula":
            para["saved_info"]["latex"] = result
        else:
            para["saved_info"]["text"] = result
        final_block.append(para["saved_info"])

    def _order(poly: list[float]) -> float:
        return poly[1] * 3000 + poly[0]

    final_block.sort(key=lambda b: _order(b["poly"]))
    md_parts: list[str] = []
    for block in final_block:
        cat = block["category_type"]
        if cat == "title":
            md_parts.append("\n# " + block.get("text", "") + "\n")
        elif cat == "isolate_formula":
            md_parts.append("\n" + block.get("latex", "") + "\n")
        elif cat in ("plain text", "figure_caption", "table_caption"):
            md_parts.append(" " + block.get("text", "") + " ")
    return "".join(md_parts).strip()
