"""Ground interactive panels on the final LLM answer (pass-2 + affinity gate).

ponytail: one module for math+science finalize — no second framework.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

AFFINITY_THRESHOLD = 40

_SECTION_RE = re.compile(
    r"(?:\*\*)?(?:Final Answer|Solution|The Formula|Given Information|Concept Behind It)"
    r"(?:\*\*)?\s*\n([\s\S]*?)(?=\n(?:\*\*)?(?:To Find|Quick Check|Key Takeaway|Practice Question|Final Answer|Solution|\Z))",
    re.I,
)
_NUM_RE = re.compile(
    r"(?<![A-Za-z_])(\d+(?:\.\d+)?)(?:\s*(?:cm(?:\s*[²³23])?|m(?:\s*[²2])?|N(?:/cm[²2])?|Pa|°|degrees?))?",
    re.I,
)
_POWER_RE = re.compile(
    r"\b(to\s+the\s+power|exponent|exponents?|powers?\s+of|multipl(?:y|ied)\s+by\s+itself)\b|"
    r"\b\d+\s*\^\s*\d+\b|\b\d+[²³]\b",
    re.I,
)
_QUADRATIC_RE = re.compile(r"\b(quadratic|parabola|discriminant|ax\s*\^?\s*2|x\s*[²2])\b", re.I)
_OPEN_BOX_RE = re.compile(
    r"\b(open\s+box|cut\s+from\s+(?:its\s+)?(?:four\s+)?corners?|folded?\s+upwards?|"
    r"squares?\s+of\s+side|sheet\s+of\s+paper.*cut)\b",
    re.I,
)
_PRESSURE_RE = re.compile(
    r"\b(pressure|N\s*/\s*cm|force\s+and\s+area|P\s*=\s*F\s*/\s*A|exerted\s+by\s+the\s+block)\b",
    re.I,
)
_DISEASE_RE = re.compile(
    r"\b(transmission|germs?\b|pathogen|wash\s+(?:my\s+)?hands?|"
    r"spread\s+(?:of\s+)?(?:disease|infection|illness|germs?)|"
    r"how\s+(?:do\s+)?(?:germs|diseases?)\s+spread|airborne|vector-borne|contagious)\b",
    re.I,
)
_HEALTH_HABITS_RE = re.compile(
    r"\b(stay\s+healthy|keep\s+\w+\s+healthy|nutritious|nutrition|balanced\s+diet|"
    r"healthy\s+(?:food|habit|lifestyle)|clean\s+water|exercise|"
    r"body\s+need)\b",
    re.I,
)
_SKIP_ANSWER_RE = re.compile(r"^\s*(hi|hello|hey|thanks|ok|okay|yes|no)\b", re.I)

SubjectKind = Literal["math", "science"]

_SCI_SHORT_MAP = {
    "photosynthesis": "plant-anatomy-lab",
    "respiration": "human-body-system-3d",
    "force-motion": "force-pressure-lab",
    "magnetism": "magnet-field-visualizer",
    "electricity": "circuit-builder",
    "acids-bases": "acid-base-indicator-lab",
    "chemical-reaction": "reaction-simulator",
    "states-of-matter": "states-of-matter-lab",
    "water-cycle": "water-cycle-animator",
    "sound": "sound-wave-lab",
    "solar-system": "solar-system-3d",
    "disease": "disease-transmission-simulator",
    "light-reflection": "light-optics-bench",
    "refraction": "light-optics-bench",
    "heat-transfer": "states-of-matter-lab",
    "digestion": "human-body-system-3d",
    "blood-circulation": "human-body-system-3d",
    "human-organs": "human-body-system-3d",
}


def match_text(answer: str, query: str = "") -> str:
    """Prefer clean answer; append query when answer is thin."""
    a = (answer or "").strip()
    q = (query or "").strip()
    if len(a) >= 80:
        return f"{a}\n{q}".strip() if q and q.lower() not in a.lower() else a
    return f"{a} {q}".strip()


def extract_answer_numbers(answer: str) -> list[float]:
    """Numbers from Solution/Final Answer first, else whole answer."""
    text = answer or ""
    chunks: list[str] = []
    for m in _SECTION_RE.finditer(text):
        chunks.append(m.group(1))
    blob = "\n".join(chunks) if chunks else text
    out: list[float] = []
    seen: set[float] = set()
    for m in _NUM_RE.finditer(blob):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _panel_sliders(panel: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not panel:
        return []
    viz = panel.get("visualization") or panel.get("experiment") or {}
    if not isinstance(viz, dict):
        return []
    return list(viz.get("sliders") or [])


def panel_type(panel: dict[str, Any] | None, kind: SubjectKind) -> str:
    if not panel:
        return ""
    if kind == "math":
        viz = panel.get("visualization") or {}
        return str(viz.get("visualizationType") or panel.get("visualizationType") or "")
    exp = panel.get("experiment") or panel
    return str(exp.get("experimentType") or panel.get("experimentType") or "")


def expand_slider_ranges(
    panel: dict[str, Any] | None, answer_nums: list[float]
) -> dict[str, Any] | None:
    """Ensure slider min/max can hold answer quantities (e.g. area 200 cm²)."""
    if not panel or not answer_nums:
        return panel
    out = dict(panel)
    key = "visualization" if "visualization" in out else ("experiment" if "experiment" in out else None)
    if not key:
        return panel
    block = dict(out.get(key) or {})
    range_ids = {
        "area", "force", "length", "width", "height", "cut",
        "sheetLength", "sheetWidth", "s", "r", "h",
    }
    sliders = []
    for s in block.get("sliders") or []:
        if not isinstance(s, dict):
            sliders.append(s)
            continue
        s2 = dict(s)
        try:
            default = float(s2.get("default", 0))
            smin = float(s2.get("min", 0))
            smax = float(s2.get("max", default or 1))
        except (TypeError, ValueError):
            sliders.append(s2)
            continue
        sid = str(s2.get("id") or "")
        for n in answer_nums:
            if n <= 0:
                continue
            if default > 0 and 0.2 * default <= n <= 50 * max(default, 1):
                smax = max(smax, n)
            elif sid in range_ids and n <= 2000:
                smax = max(smax, n)
        if default > smax:
            smax = default
        if default < smin:
            smin = min(smin, default)
        s2["min"] = smin
        s2["max"] = smax
        sliders.append(s2)
    block["sliders"] = sliders
    out[key] = block
    return out


def seed_sliders_from_numbers(
    panel: dict[str, Any] | None, answer_nums: list[float]
) -> dict[str, Any] | None:
    """Set slider defaults from answer numbers when possible."""
    if not panel or not answer_nums:
        return panel
    out = expand_slider_ranges(panel, answer_nums)
    if not out:
        return panel
    key = "visualization" if "visualization" in out else ("experiment" if "experiment" in out else None)
    if not key:
        return out
    block = dict(out.get(key) or {})
    sliders = []
    pool = list(answer_nums)
    for s in block.get("sliders") or []:
        if not isinstance(s, dict):
            sliders.append(s)
            continue
        s2 = dict(s)
        try:
            smin = float(s2.get("min", 0))
            smax = float(s2.get("max", 100))
            cur = float(s2.get("default", smin))
        except (TypeError, ValueError):
            sliders.append(s2)
            continue
        pick = None
        sid = str(s2.get("id") or "").lower()
        prefer: dict[str, tuple[float, ...]] = {
            "force": (40.0,),
            "area": (200.0, 100.0),
            "cut": (5.0,),
            "height": (5.0, 10.0),
            "h": (5.0,),
            "length": (20.0, 30.0, 10.0),
            "width": (10.0, 20.0),
            "sheetlength": (30.0,),
            "sheetwidth": (20.0,),
        }
        wanted = prefer.get(sid, ())
        for w in wanted:
            if w in pool and smin <= w <= smax:
                pick = w
                break
        if pick is None:
            for n in pool:
                if smin <= n <= smax and n != cur:
                    pick = n
                    break
        if pick is not None:
            s2["default"] = pick
            if pick in pool:
                pool.remove(pick)
        sliders.append(s2)
    block["sliders"] = sliders
    if block.get("visualizationType") == "linear-graph" or block.get("curveType") == "quadratic":
        if len(answer_nums) >= 3 and not block.get("coefficients"):
            block["coefficients"] = [answer_nums[0], answer_nums[1], answer_nums[2]]
    out[key] = block
    return out


def type_affinity_score(text: str, viz_type: str, kind: SubjectKind) -> int:
    """Higher = better match between answer text and panel type. 0 = conflict/no signal."""
    t = (text or "").strip()
    vt = (viz_type or "").strip().lower()
    if not t or not vt:
        return 0

    score = 0

    if kind == "math":
        from app.services.math_lesson.visual_matcher import _AFFINITY

        for affinity_re, affinity_type, boost in _AFFINITY:
            if vt == affinity_type and affinity_re.search(t):
                score += boost
        if _OPEN_BOX_RE.search(t) and vt in (
            "area-resizer",
            "mensuration-cube",
            "mensuration-cylinder",
            "concept-explorer",
        ):
            score += 95 if vt == "area-resizer" else 70
        if _POWER_RE.search(t) and not _QUADRATIC_RE.search(t):
            if vt in ("linear-graph", "statistics-lab"):
                return 0
            # Don't boost Power Play when text is √n / construction.
            from app.services.interactive_templates import query_blocks_powers

            if query_blocks_powers(t) and vt == "concept-explorer":
                pass
            elif vt in ("concept-explorer", "number-line", "algebra-stepper"):
                score += 80
        if vt == "sqrt-number-line" and (
            re.search(r"\b(number\s*line|√|sqrt|square\s*root|construct)\b", t, re.I)
        ):
            score += 100
        if _QUADRATIC_RE.search(t) and vt == "linear-graph":
            score += 90
    else:
        from app.services.science_experiment.visual_matcher import _AFFINITY

        for affinity_re, affinity_type, boost in _AFFINITY:
            target = _SCI_SHORT_MAP.get(affinity_type, affinity_type)
            if vt in (affinity_type, target) and affinity_re.search(t):
                score += boost
        if _PRESSURE_RE.search(t) and vt == "force-pressure-lab":
            score += 100
        if _HEALTH_HABITS_RE.search(t) and not _DISEASE_RE.search(t):
            if vt == "disease-transmission-simulator":
                return 0
            if vt == "human-body-system-3d":
                score += 90
        if _DISEASE_RE.search(t) and vt == "disease-transmission-simulator":
            score += 95
        if _PRESSURE_RE.search(t) and vt not in ("force-pressure-lab", "force-motion"):
            if score < AFFINITY_THRESHOLD:
                return min(score, 10)

    return score


def panel_numbers_agree(panel: dict[str, Any] | None, answer_nums: list[float]) -> bool:
    """True if key slider defaults are grounded in the answer (or answer has no numbers)."""
    if not answer_nums:
        return True
    if not panel:
        return False
    defaults: list[float] = []
    for s in _panel_sliders(panel):
        if not isinstance(s, dict):
            continue
        try:
            defaults.append(float(s["default"]))
        except (KeyError, TypeError, ValueError):
            pass
    if not defaults:
        return True

    def in_answer(d: float) -> bool:
        for a in answer_nums:
            if abs(a - d) < 1e-6:
                return True
            if max(abs(a), abs(d), 1) and abs(a - d) / max(abs(a), abs(d), 1) < 0.02:
                return True
        return False

    hits = sum(1 for d in defaults if in_answer(d))
    need = 1 if len(defaults) == 1 else max(1, (len(defaults) + 1) // 2)
    return hits >= need


def _force_seed_defaults(
    panel: dict[str, Any] | None, nums: list[float]
) -> dict[str, Any] | None:
    if not panel or not nums:
        return panel
    out = dict(panel)
    key = "visualization" if "visualization" in out else ("experiment" if "experiment" in out else None)
    if not key:
        return out
    block = dict(out.get(key) or {})
    sliders = []
    pool = [n for n in nums if n > 0]
    for s in block.get("sliders") or []:
        if not isinstance(s, dict):
            sliders.append(s)
            continue
        s2 = dict(s)
        if pool:
            n = pool.pop(0)
            s2["default"] = n
            try:
                s2["max"] = max(float(s2.get("max", n)), n)
                s2["min"] = min(float(s2.get("min", 0)), n)
            except (TypeError, ValueError):
                pass
        sliders.append(s2)
    block["sliders"] = sliders
    out[key] = block
    return out


def should_emit_panel(
    panel: dict[str, Any] | None,
    answer: str,
    *,
    kind: SubjectKind,
    query: str = "",
) -> dict[str, Any] | None:
    """Affinity + number gate. Returns panel (possibly seeded) or None."""
    from app.services.interactive_templates import query_blocks_powers

    if not panel:
        return None
    text = match_text(answer, query)
    vt = panel_type(panel, kind)
    if not vt or vt in ("generic", ""):
        return None
    score = type_affinity_score(text, vt, kind)
    if score < AFFINITY_THRESHOLD and (query or "").strip():
        # Answer may paraphrase without catalog trigger words; re-score the query.
        score = max(score, type_affinity_score(query, vt, kind))
    if score < AFFINITY_THRESHOLD:
        if (
            kind == "math"
            and vt == "concept-explorer"
            and len((answer or "").strip()) > 120
            and (_OPEN_BOX_RE.search(text) or _POWER_RE.search(text))
            and not query_blocks_powers(query)
        ):
            score = AFFINITY_THRESHOLD
        elif kind == "math" and vt in (
            "sqrt-number-line",
            "geometry-construction",
            "number-line",
            "factor-tree",
        ):
            # Keep Interactive Exploration when query/catalog chose a specific lab.
            score = AFFINITY_THRESHOLD
        else:
            return None

    nums = extract_answer_numbers(answer)
    # Don't clobber template defaults that already match the answer.
    if nums and panel_numbers_agree(panel, nums):
        panel = expand_slider_ranges(panel, nums)
        return panel
    panel = seed_sliders_from_numbers(panel, nums)
    panel = expand_slider_ranges(panel, nums)
    if nums and not panel_numbers_agree(panel, nums):
        panel = _force_seed_defaults(panel, nums)
        panel = expand_slider_ranges(panel, nums)
        if not panel_numbers_agree(panel, nums):
            logger.info("Suppress interactive: numbers disagree type=%s", vt)
            return None
    return panel


def _allow_list_math() -> list[str]:
    from app.services.math_lesson.topic_ontology import KNOWN_VISUALIZATION_TYPES

    return sorted(KNOWN_VISUALIZATION_TYPES)


def _allow_list_science() -> list[str]:
    from app.services.science_experiment.topic_ontology import KNOWN_VISUALIZATION_TYPES

    return sorted(KNOWN_VISUALIZATION_TYPES)


_PASS2_MATH_SYSTEM = """You build ONE interactive math-lesson JSON that illustrates the tutor ANSWER.
Rules:
- Output ONLY a ```math-lesson JSON fence (or {"skip":true} if no suitable type).
- Pick visualizationType from ALLOW_LIST only.
- Copy EVERY important number and formula from the ANSWER into slider defaults and liveCalculations.
- Open box / cut corners / folded sheet → visualizationType "area-resizer" with sliders length, width, height (folded box) and liveCalculations volume = length * width * height. Include sheetLength/sheetWidth/cut when in the answer.
- Powers/exponents → "concept-explorer" or "algebra-stepper", NEVER linear-graph or statistics-lab.
- Construct √n / square root on a number line / geometrical construction of surds → "sqrt-number-line", NEVER concept-explorer Power Play.
- Do not invent numbers that are not in the ANSWER."""

_PASS2_SCIENCE_SYSTEM = """You build ONE interactive science-experiment JSON that illustrates the tutor ANSWER.
Rules:
- Output ONLY a ```science-experiment JSON fence (or {"skip":true} if no suitable type).
- Pick experimentType from ALLOW_LIST only.
- Copy EVERY important number and formula from the ANSWER into slider defaults and liveCalculations / threeViews.scientific.equation.
- Force/pressure / P=F/A / block on table → experimentType "force-pressure-lab" with force and area from the answer; set area max >= largest area in the answer.
- Disease/germs/transmission / wash hands → "disease-transmission-simulator".
- Stay healthy / nutritious food / exercise (without germs/transmission) → "human-body-system-3d", NOT disease.
- Include threeViews (realWorld, microscopic, scientific).
- Virtual lab only. Do not invent numbers absent from the ANSWER."""


def llm_build_panel_from_answer(
    answer: str,
    *,
    kind: SubjectKind,
    query: str = "",
    class_level: str = "",
    enabled: bool = True,
) -> dict[str, Any] | None:
    """Pass-2 LLM: answer → interactive JSON. Returns None on skip/failure."""
    if not enabled:
        return None
    a = (answer or "").strip()
    if len(a) < 40 or _SKIP_ANSWER_RE.match(a):
        return None
    allow = _allow_list_math() if kind == "math" else _allow_list_science()
    hinted = _hint_types(a, kind)
    allow_show = hinted + [t for t in allow if t not in hinted][:40]
    system = _PASS2_MATH_SYSTEM if kind == "math" else _PASS2_SCIENCE_SYSTEM
    user = (
        f"CLASS_LEVEL: {class_level or 'unknown'}\n"
        f"QUERY: {query[:400]}\n"
        f"ALLOW_LIST: {', '.join(allow_show)}\n\n"
        f"ANSWER:\n{a[:6000]}\n"
    )
    try:
        from app.services.llm_client import complete_sync

        raw = complete_sync(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            feature="lesson",
            max_tokens=2200,
            timeout=90.0,
        )
    except Exception as exc:
        logger.warning("pass-2 interactive LLM failed: %s", exc)
        return None
    return _parse_pass2(raw, kind)


def _hint_types(answer: str, kind: SubjectKind) -> list[str]:
    a = answer or ""
    if kind == "math":
        out: list[str] = []
        if _OPEN_BOX_RE.search(a):
            out += ["area-resizer", "mensuration-cube"]
        if _POWER_RE.search(a):
            out += ["concept-explorer", "algebra-stepper"]
        if re.search(r"\b(perfect\s+square|prime\s+factor)\b", a, re.I):
            out += ["factor-tree"]
        if re.search(r"\bfraction", a, re.I):
            out += ["fractions"]
        return out
    out: list[str] = []
    if _PRESSURE_RE.search(a):
        out.append("force-pressure-lab")
    if _DISEASE_RE.search(a):
        out.append("disease-transmission-simulator")
    if re.search(r"\bphotosynthesis\b", a, re.I):
        out.append("plant-anatomy-lab")
    return out


def _parse_pass2(raw: str, kind: SubjectKind) -> dict[str, Any] | None:
    text = raw or ""
    if re.search(r'"skip"\s*:\s*true', text, re.I):
        return None
    fence = (
        r"```(?:math-lesson|json:math-lesson|math_lesson)\s*\n([\s\S]*?)```"
        if kind == "math"
        else (
            r"```(?:science-experiment|science-lesson|json:science-experiment|"
            r"science_experiment|science_lesson)\s*\n([\s\S]*?)```"
        )
    )
    m = re.search(fence, text, re.I)
    blob = m.group(1).strip() if m else None
    if not blob:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start : end + 1]
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("skip") is True:
        return None
    try:
        if kind == "math":
            from app.services.math_lesson.schemas import MathLesson

            return MathLesson.model_validate(data).model_dump(mode="json")
        from app.services.science_experiment.schemas import ScienceExperiment

        return ScienceExperiment.model_validate(data).model_dump(mode="json")
    except Exception as exc:
        logger.debug("pass-2 schema validate failed: %s", exc)
        return data if panel_type(data, kind) else None


def resolve_grounded_panel(
    *,
    answer: str,
    query: str,
    kind: SubjectKind,
    class_level: str = "",
    first_panel: dict[str, Any] | None = None,
    catalog_panel: dict[str, Any] | None = None,
    allow_llm_pass2: bool = True,
) -> dict[str, Any] | None:
    """Tiered resolve: typed extract→template first; pass-2 only if incomplete."""
    from app.services.interactive_templates import (
        PRODUCTION_TYPES,
        build_template_panel,
        extract_typed_params,
        normalize_to_template,
        params_complete,
        query_blocks_fractions,
        query_blocks_powers,
    )

    text = match_text(answer, query)
    if len(text) < 20:
        return None

    preferred, params, pack = extract_typed_params(text, kind)

    # Dual-signal: student question wins on type — never Power Play for roots/construction.
    if kind == "math" and pack == "powers" and query_blocks_powers(query):
        preferred, params, pack = None, {}, ""
        q_vt, q_params, q_pack = extract_typed_params(query or "", kind)
        if q_pack == "sqrt_line":
            preferred, params, pack = q_vt, q_params, q_pack
        else:
            a_vt, a_params, a_pack = extract_typed_params(
                f"{query}\n{answer}", kind
            )
            if a_pack and a_pack != "powers":
                preferred, params, pack = a_vt, a_params, a_pack

    # Dual-signal: algebra/%/probability questions must not become fractions from bare a/b.
    if kind == "math" and pack == "fractions" and query_blocks_fractions(query):
        preferred, params, pack = None, {}, ""

    # Tier A — deterministic template (no extra LLM) when params are complete.
    if preferred and params_complete(preferred, params, pack=pack):
        built = build_template_panel(
            preferred, params, kind=kind, class_level=class_level, pack=pack
        )
        gated = should_emit_panel(built, answer or text, kind=kind, query=query)
        if gated is not None:
            return gated

    need_pass2 = not (preferred and params_complete(preferred, params, pack=pack))
    pass2 = llm_build_panel_from_answer(
        answer if len((answer or "").strip()) >= 40 else text,
        kind=kind,
        query=query,
        class_level=class_level,
        enabled=allow_llm_pass2 and need_pass2,
    )

    def _norm(p: dict[str, Any] | None) -> dict[str, Any] | None:
        return normalize_to_template(
            p,
            kind=kind,
            class_level=class_level,
            preferred_type=preferred,
            params=params,
            pack=pack,
        )

    templated_preferred = None
    if preferred and preferred in PRODUCTION_TYPES:
        templated_preferred = build_template_panel(
            preferred, params, kind=kind, class_level=class_level, pack=pack
        )

    # Prefer catalog / first panel when query intent blocks powers or fractions steals.
    candidates: tuple[dict[str, Any] | None, ...]
    if kind == "math" and (query_blocks_powers(query) or query_blocks_fractions(query)):
        candidates = (
            _norm(catalog_panel),
            catalog_panel,
            _norm(first_panel),
            first_panel,
            templated_preferred,
            _norm(pass2),
        )
    else:
        candidates = (
            _norm(pass2),
            templated_preferred,
            _norm(first_panel),
            _norm(catalog_panel),
            catalog_panel,
        )

    for candidate in candidates:
        gated = should_emit_panel(candidate, answer or text, kind=kind, query=query)
        if gated is not None:
            if params:
                gated = normalize_to_template(
                    gated,
                    kind=kind,
                    class_level=class_level,
                    preferred_type=preferred or panel_type(gated, kind),
                    params=params,
                    pack=pack,
                ) or gated
                gated = should_emit_panel(gated, answer or text, kind=kind, query=query)
                if gated is None:
                    continue
            return gated

    if catalog_panel and not extract_answer_numbers(answer or text):
        vt = panel_type(catalog_panel, kind)
        if type_affinity_score(text, vt, kind) >= AFFINITY_THRESHOLD:
            return catalog_panel
    # Keep Interactive Exploration for math: emit catalog/first even if affinity was soft.
    if kind == "math":
        for fallback in (catalog_panel, first_panel):
            if fallback and panel_type(fallback, kind) not in ("", "generic"):
                return fallback
    return None
