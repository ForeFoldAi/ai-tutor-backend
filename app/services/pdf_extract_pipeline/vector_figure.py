"""
Figure geometry from the PDF itself: vector drawings, embedded images and short labels.

The layout model often misses thin line diagrams (geometry, ray optics, circuits,
graphs). Before this module, a missed diagram fell back to a blind fixed-size window
above its "Fig. X" label, which cut figures in half or pulled in paragraphs. Here the
figure is located from what the PDF actually draws:

1. Collect page elements: vector paths and embedded raster images, minus page furniture
   (running header/footer rules, full-page backgrounds, filled text boxes, watermarks).
2. Cluster elements that touch (within a small gap).
3. Choose the cluster(s) that belong to the "Fig. X" label: nearest above the label (or
   beside it), never crossing another figure label or a paragraph.
4. Pull in short text labels (A, B, 30°, 5 cm, Hypotenuse) that sit on/around the drawing.
5. Render with prose lines that fall inside the crop painted out.

Everything is in PDF points and relative to the page, so it works for any board,
class, subject and page size.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field

import fitz

PANEL_MARK_RE = re.compile(r"^\(?(?:[ivx]{1,4}|[a-h])\)$|^\((?:[ivx]{1,4}|[a-h])\)?$", re.I)
FIG_LABEL_RE = re.compile(r"(?i)^\s*fig(?:ure)?\.?\s*(\d+(?:\.\d+)*)\b")

# Gap (pt) under which two elements belong to the same drawing.
CLUSTER_GAP = 10.0
# Max vertical distance (pt) between a figure label and its drawing.
LABEL_MAX_GAP = 80.0
# Short labels this close (pt) to the drawing become part of the figure.
LABEL_ATTACH_GAP = 14.0
# Side-by-side / stacked panels of one figure are at most this far apart (pt).
PANEL_GAP = 60.0
# Ignore elements in the running header/footer bands (fraction of page height).
HEADER_FRAC = 0.1
FOOTER_FRAC = 0.06
# A recurring image on this share of pages is a page template / watermark.
RECURRING_PAGE_FRAC = 0.4


def is_figure_label_text(text: str) -> str | None:
    """Figure number when *text* is a figure label line ("Fig. 6.14", "Figure 3.2: ...")."""
    m = FIG_LABEL_RE.match(text or "")
    return m.group(1) if m else None


def is_prose_line(text: str) -> bool:
    """Body-text line (sentence) rather than a diagram label."""
    t = (text or "").strip()
    if not t:
        return False
    if is_figure_label_text(t):
        return False
    words = t.split()
    return len(t) > 40 or len(words) > 6


@dataclass
class TextLine:
    rect: fitz.Rect
    text: str
    size: float = 0.0


def page_text_lines(page: fitz.Page) -> list[TextLine]:
    out: list[TextLine] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if text:
                sizes = [float(s.get("size") or 0) for s in spans if (s.get("text") or "").strip()]
                out.append(TextLine(fitz.Rect(line["bbox"]), text, max(sizes) if sizes else 0.0))
    return out


def body_font_size(lines: list[TextLine]) -> float:
    """Most common size among prose lines (the page's body text size)."""
    sizes = Counter(round(ln.size, 0) for ln in lines if is_prose_line(ln.text) and ln.size > 0)
    return float(sizes.most_common(1)[0][0]) if sizes else 0.0


def _is_fraction_bar(r: fitz.Rect, lines: list[TextLine]) -> bool:
    """Thin horizontal rule with text right above and below it: an equation, not a figure."""
    if r.height > 1.5 or r.width > 90:
        return False
    above = below = False
    for ln in lines:
        if _h_overlap(ln.rect, r) < 0.4 * min(r.width, ln.rect.width):
            continue
        if 0 <= r.y0 - ln.rect.y1 <= 6:
            above = True
        elif 0 <= ln.rect.y0 - r.y1 <= 6:
            below = True
    return above and below


def _furniture_band(page: fitz.Page) -> tuple[float, float]:
    h = page.rect.height
    return h * HEADER_FRAC, h * (1.0 - FOOTER_FRAC)


def figure_elements(
    page: fitz.Page,
    *,
    lines: list[TextLine] | None = None,
    skip_image_digests: frozenset[bytes] = frozenset(),
) -> tuple[list[fitz.Rect], int]:
    """Drawing and image rects that can belong to a figure, and how many are vector paths."""
    W, H = page.rect.width, page.rect.height
    top, bottom = _furniture_band(page)
    lines = lines if lines is not None else page_text_lines(page)
    prose = [ln.rect for ln in lines if is_prose_line(ln.text)]
    rects: list[fitz.Rect] = []
    n_vector = 0
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.width <= 0 and r.height <= 0:
            continue
        # Pad by half the stroke width: a horizontal/vertical line has a zero-height/width
        # box, which PyMuPDF treats as empty (never intersects), so axes and number lines
        # would otherwise drop out of every cluster.
        half = max(0.5, float(d.get("width") or 1.0) / 2.0)
        r = fitz.Rect(r.x0 - half, r.y0 - half, r.x1 + half, r.y1 + half)
        if r.y1 < top or r.y0 > bottom:
            continue
        if r.width > W * 0.6 and r.height < 4:  # header/footer/section rules
            continue
        if r.height < 3 and r.width > W * 0.25 and (r.y1 < H * 0.15 or r.y0 > H * 0.88):
            continue  # running-header / footer rule next to the page title or number
        if r.width * r.height > W * H * 0.35:  # page / column backgrounds
            continue
        if d.get("fill") is not None and sum(1 for p in prose if r.contains(p)) >= 2:
            continue  # filled box behind a paragraph (activity / note box)
        if _is_fraction_bar(r, lines):
            continue
        rects.append(r)
        n_vector += 1
    for info in page.get_image_info(hashes=True):
        if info.get("digest") in skip_image_digests:
            continue
        if int(info.get("width") or 0) <= 2 or int(info.get("height") or 0) <= 2:
            continue  # placeholder left by strip_recurring_images / spacer pixels
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.width < 8 or r.height < 8:
            continue
        if r.width * r.height > W * H * 0.6:
            continue
        rects.append(r)
    return rects, n_vector


def cluster_rects(rects: list[fitz.Rect], gap: float = CLUSTER_GAP) -> list[fitz.Rect]:
    """Union rects that lie within *gap* of each other (transitively)."""
    out = [fitz.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        result: list[fitz.Rect] = []
        while out:
            a = out.pop()
            i = 0
            while i < len(out):
                b = out[i]
                if fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap).intersects(b):
                    a |= b
                    out.pop(i)
                    merged = True
                else:
                    i += 1
            result.append(a)
        out = result
    return out


def _h_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _v_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _contains_prose(rect: fitz.Rect, lines: list[TextLine], *, min_lines: int = 2) -> bool:
    inner = rect + (2, 2, -2, -2)
    return sum(1 for ln in lines if is_prose_line(ln.text) and inner.contains(ln.rect)) >= min_lines


def _crosses_other_label(rect: fitz.Rect, label_rects: list[fitz.Rect]) -> bool:
    return any(rect.intersects(lr) for lr in label_rects)


@dataclass
class VectorFigure:
    rect: fitz.Rect          # figure region (PDF points), label line excluded
    core: fitz.Rect          # drawing/image elements only (no attached labels)
    vector_paths: int        # number of vector path elements inside the core
    labels: list[fitz.Rect] = field(default_factory=list)  # attached text labels


def locate_figure(
    page: fitz.Page,
    label_rect: fitz.Rect,
    *,
    lines: list[TextLine] | None = None,
    other_label_rects: list[fitz.Rect] | None = None,
    skip_image_digests: frozenset[bytes] = frozenset(),
) -> VectorFigure | None:
    """Region of the figure that the "Fig. X" label at *label_rect* belongs to."""
    lines = lines if lines is not None else page_text_lines(page)
    body = body_font_size(lines)
    others = [r for r in (other_label_rects or []) if not r.intersects(label_rect)]
    elements, _ = figure_elements(page, lines=lines, skip_image_digests=skip_image_digests)
    clusters = [
        c for c in cluster_rects(elements)
        if c.width * c.height >= 300 and not c.intersects(label_rect)
        and not _contains_prose(c, lines)
    ]
    if not clusters:
        return None

    def score(c: fitz.Rect) -> float | None:
        # Drawing above the label (most common), or beside it on the same row.
        if c.y1 <= label_rect.y0 + 4:
            v_gap = label_rect.y0 - c.y1
            if v_gap > LABEL_MAX_GAP:
                return None
            h_gap = max(0.0, max(c.x0, label_rect.x0) - min(c.x1, label_rect.x1))
            if h_gap > 40:
                return None
            between = fitz.Rect(min(c.x0, label_rect.x0), c.y1, max(c.x1, label_rect.x1), label_rect.y0)
            if _crosses_other_label(between, others):
                return None
            return v_gap + h_gap * 0.5
        if _v_overlap(c, label_rect) > 0 or (c.y0 < label_rect.y1 and c.y1 > label_rect.y0 - 4):
            h_gap = max(0.0, max(c.x0, label_rect.x0) - min(c.x1, label_rect.x1))
            if h_gap > 30:
                return None
            return 20.0 + h_gap
        return None

    scored = [(s, c) for c in clusters if (s := score(c)) is not None]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    core = fitz.Rect(scored[0][1])

    # Multi-panel figures: (i)/(ii) side by side, or (a)/(b) stacked above the label.
    changed = True
    rounds = 0
    # Each round must grow the core; the cap is a backstop. An empty/degenerate
    # cluster is never "contained" yet adds nothing — counting that as a change
    # looped forever (Class 6 Maths "Lines and Angles" hung extraction for hours).
    while changed and rounds <= len(clusters):
        changed = False
        rounds += 1
        for c in clusters:
            if c.is_empty or core.contains(c):
                continue
            side_by_side = _v_overlap(c, core) >= 0.5 * min(c.height, core.height) and (
                max(c.x0, core.x0) - min(c.x1, core.x1) <= PANEL_GAP
            )
            stacked = (
                _h_overlap(c, core) >= 0.5 * min(c.width, core.width)
                and 0 <= core.y0 - c.y1 <= CLUSTER_GAP * 3
            )
            if not (side_by_side or stacked):
                continue
            cand = core | c
            if cand == core:
                continue
            if cand.intersects(label_rect) or _crosses_other_label(cand, others) or _contains_prose(cand, lines):
                continue
            core = cand
            changed = True

    n_vector = 0
    probe = core + (-2, -2, 2, 2)
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if probe.contains(fitz.Rect(r.x0, r.y0, max(r.x1, r.x0 + 0.1), max(r.y1, r.y0 + 0.1))):
            n_vector += 1

    region = fitz.Rect(core)
    attached: list[fitz.Rect] = []
    H = page.rect.height
    for ln in lines:
        # Running headers/footers (book title, page number) beside a figure at the page edge.
        if (ln.rect.y1 < H * 0.15 and ln.rect.y1 <= core.y0 + 2) or (
            ln.rect.y0 > H * 0.88 and ln.rect.y0 >= core.y1 - 2
        ):
            continue
        if is_diagram_label(ln, core, label_rect, body):
            region |= ln.rect
            attached.append(ln.rect)
    return VectorFigure(rect=region + (-4, -4, 4, 4), core=core, vector_paths=n_vector, labels=attached)


def is_diagram_label(ln: TextLine, core: fitz.Rect, label_rect: fitz.Rect, body_size: float) -> bool:
    """Short text that belongs to the drawing (part names, values), not body text."""
    if is_prose_line(ln.text) or is_figure_label_text(ln.text) or ln.rect.intersects(label_rect):
        return False
    if core.contains(ln.rect):
        return True
    zone_panel = core + (-LABEL_ATTACH_GAP, -LABEL_ATTACH_GAP, LABEL_ATTACH_GAP, LABEL_ATTACH_GAP)
    if PANEL_MARK_RE.match(ln.text.strip()) and zone_panel.intersects(ln.rect):
        return True  # (i), (ii), (a), (b) panel markers
    zone = core + (-LABEL_ATTACH_GAP, -LABEL_ATTACH_GAP, LABEL_ATTACH_GAP, LABEL_ATTACH_GAP)
    if not zone.intersects(ln.rect):
        return False
    # Body-size text next to a drawing is usually a sentence fragment ("(Why?)",
    # "(Theorem 6.1)"); diagram callouts are set smaller than body text.
    return body_size <= 0 or ln.size < body_size - 0.5


def prose_mask_rects(page: fitz.Page, fig: VectorFigure, *, lines: list[TextLine] | None = None) -> list[fitz.Rect]:
    """Text lines inside the crop that are not part of the figure (to paint out)."""
    lines = lines if lines is not None else page_text_lines(page)
    out: list[fitz.Rect] = []
    for ln in lines:
        if not ln.rect.intersects(fig.rect):
            continue
        if fig.core.contains(ln.rect) or any(ln.rect.intersects(a) for a in fig.labels):
            continue  # text that is part of the diagram
        out.append(ln.rect)
    return out


def render_region(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    zoom: float = 2.5,
    mask: list[fitz.Rect] | None = None,
    trim: bool = True,
) -> bytes:
    """PNG of *rect* rendered from the vector page, *mask* rects painted white, and the
    surrounding white margin trimmed."""
    from PIL import Image, ImageChops, ImageDraw

    clip = rect & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if mask:
        draw = ImageDraw.Draw(img)
        for m in mask:
            r = m & clip
            if r.is_empty:
                continue
            draw.rectangle(
                [
                    (r.x0 - clip.x0) * zoom - 1,
                    (r.y0 - clip.y0) * zoom - 1,
                    (r.x1 - clip.x0) * zoom + 1,
                    (r.y1 - clip.y0) * zoom + 1,
                ],
                fill="white",
            )
    if trim:
        # Near-white (>= 245) counts as background so faint scanner/anti-alias noise is ignored.
        gray = img.convert("L").point(lambda v: 255 if v >= 245 else 0)
        bbox = ImageChops.invert(gray).getbbox()
        if bbox:
            pad = int(3 * zoom)
            x0, y0, x1, y1 = bbox
            img = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def recurring_image_digests(doc: fitz.Document, *, page_frac: float = RECURRING_PAGE_FRAC) -> frozenset[bytes]:
    """Digests of images repeated on many pages: watermarks, page backgrounds, borders."""
    counts: Counter[bytes] = Counter()
    for page in doc:
        for d in {i["digest"] for i in page.get_image_info(hashes=True)}:
            counts[d] += 1
    threshold = max(3, int(doc.page_count * page_frac))
    return frozenset(d for d, n in counts.items() if n >= threshold)


def strip_recurring_images(doc: fitz.Document) -> tuple[fitz.Document, int]:
    """In-memory copy of *doc* without watermark / page-template images.

    NCERT pages carry a "© NCERT not to be republished" stamp and a page background as
    images repeated on every page; they bleed into every figure crop and confuse the
    layout model. The source document is not modified.
    """
    digests = recurring_image_digests(doc)
    if not digests:
        return doc, 0
    clean = fitz.open(stream=doc.tobytes(), filetype="pdf")
    removed = 0
    for page in clean:
        for info in page.get_image_info(hashes=True, xrefs=True):
            if info.get("digest") in digests and info.get("xref"):
                try:
                    page.delete_image(info["xref"])
                    removed += 1
                except Exception:
                    continue
    return clean, removed


def printed_caption(page: fitz.Page, label_rect: fitz.Rect, *, max_lines: int = 6) -> str:
    """The caption as printed: the "Fig. X ..." line plus its continuation lines.

    Continuation lines are taken while they stay in the caption's own style (not larger
    than the label line and smaller than body text when the label itself is smaller), sit
    directly below (line spacing ≤ 1.6× line height) and overlap horizontally. Lines the PDF
    prints several times on top of each other (overprinted labels) are kept once.
    """
    lines = page_text_lines(page)
    body = body_font_size(lines)
    start = None
    for ln in lines:
        if ln.rect.intersects(label_rect) and is_figure_label_text(ln.text):
            if start is None or ln.rect.y0 < start.rect.y0:
                start = ln
    if start is None:
        hits = [ln for ln in lines if ln.rect.intersects(label_rect)]
        if not hits:
            return ""
        start = hits[0]

    out = [start.text]
    seen = {re.sub(r"\s+", " ", start.text).strip().lower()}
    caption_style = body > 0 and start.size < body - 0.5
    prev = start
    following = sorted(
        (ln for ln in lines if ln.rect.y0 >= start.rect.y1 - 2 and ln is not start),
        key=lambda ln: (ln.rect.y0, ln.rect.x0),
    )
    for ln in following:
        if len(out) >= max_lines:
            break
        key = re.sub(r"\s+", " ", ln.text).strip().lower()
        if key in seen:
            continue  # overprinted duplicate
        gap = ln.rect.y0 - prev.rect.y1
        if gap > 1.6 * max(prev.rect.height, 6.0):
            break
        if _h_overlap(ln.rect, start.rect | prev.rect) <= 0:
            continue
        if is_figure_label_text(ln.text):
            break  # next figure's caption
        if not caption_style or ln.size > start.size + 0.5:
            break  # body text resumes
        out.append(ln.text)
        seen.add(key)
        prev = ln
    return re.sub(r"\s+", " ", " ".join(out)).strip()
