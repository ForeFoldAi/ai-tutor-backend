"""Production interactive labs: typed templates + answer→param binding.

ponytail: fill catalog shells from extracted params — LLM must not invent slider shapes.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

SubjectKind = Literal["math", "science"]

PRODUCTION_TYPES: frozenset[str] = frozenset(
    {
        "area-resizer",
        "factor-tree",
        "mensuration-cube",
        "mensuration-cylinder",
        "fractions",
        "algebra-stepper",
        "concept-explorer",
        "number-line",
        "sqrt-number-line",
        "circle",
        "linear-graph",
        "statistics-lab",
        "shape-lab",
        "force-pressure-lab",
        "disease-transmission-simulator",
        "plant-anatomy-lab",
        "circuit-builder",
        "acid-base-indicator-lab",
        "reaction-simulator",
        "states-of-matter-lab",
        "water-cycle-animator",
        "human-body-system-3d",
        "magnet-field-visualizer",
        "sound-wave-lab",
        "solar-system-3d",
    }
)

# Query intent beats answer-only powers cues (Pythagoras “multiplied by itself”, 2²=4).
_POWERS_BLOCK_INTENT_RE = re.compile(
    r"\b(number\s*line|geometrical?\s+construction|compass|construct|"
    r"perpendicular\s+bisector|surd|irrational|"
    r"square\s*root|cube\s*root|√|sqrt)\b",
    re.I,
)
# Query is clearly not about fractions — bare a/b in the answer must not steal the lab.
_FRACTIONS_BLOCK_INTENT_RE = re.compile(
    r"\b(solve|equation|linear\s+equation|algebra|percent|percentage|"
    r"probability|ratio\b|proportion|profit|interest)\b",
    re.I,
)
_FRACTION_TOPIC_RE = re.compile(
    r"\b(fractions?|numerator|denominator|pizza|half|halves|quarter|thirds?|"
    r"part\s+of\s+(?:a\s+)?whole)\b",
    re.I,
)
_SQRT_ON_LINE_RE = re.compile(
    r"(?:√\s*\(?\s*(\d+)|sqrt\s*\(\s*(\d+)\s*\)|square\s*root\s+of\s+(\d+))",
    re.I,
)
_LEG_PAIR_RE = re.compile(
    r"(\d+)\s*[²2]\s*\+\s*(\d+)\s*[²2]|sides?\s+(\d+)\s+and\s+(\d+)|legs?\s+(\d+)\s+and\s+(\d+)",
    re.I,
)
# Do not rebuild Power Play over these when normalize sees pack=powers.
_KEEP_OVER_POWERS: frozenset[str] = frozenset(
    {
        "sqrt-number-line",
        "geometry-construction",
        "number-line",
        "pythagoras",
        "factor-tree",
        "geometry-basics",
        "angle-explorer",
        "parallel-transversal",
        "triangle-explorer",
        "triangle-angle-sum",
        "circle-tangent",
    }
)

_FORCE_N = re.compile(r"(\d+(?:\.\d+)?)\s*N\b", re.I)
_AREA_CM = re.compile(r"(\d+(?:\.\d+)?)\s*cm\s*[²2]", re.I)
_SHEET_LW = re.compile(
    r"(?:sheet|paper|rectangle)[^\d]{0,40}?(\d+(?:\.\d+)?)\s*cm[^\d]{0,40}?(\d+(?:\.\d+)?)\s*cm",
    re.I,
)
_CUT_SIDE = re.compile(
    r"(?:squares?\s+of\s+side|cut[^\d]{0,24}?side)\s*=?\s*(\d+(?:\.\d+)?)\s*cm",
    re.I,
)
# Optional intermediate "30 − 2×5 =" then the result before cm.
_BOX_DIM = re.compile(
    r"(?:base\s+length|box\s+length|new\s+length|length\s+of\s+(?:the\s+)?box)"
    r"\s*=\s*(?:[0-9.\s+\-−×x*/]+?\s*=\s*)?(\d+(?:\.\d+)?)\s*cm",
    re.I,
)
_BOX_W = re.compile(
    r"(?:base\s+width|box\s+width|new\s+width|width\s+of\s+(?:the\s+)?box)"
    r"\s*=\s*(?:[0-9.\s+\-−×x*/]+?\s*=\s*)?(\d+(?:\.\d+)?)\s*cm",
    re.I,
)
_BOX_H = re.compile(
    r"(?:height\s+of\s+(?:the\s+)?box|box\s+height)"
    r"\s*=\s*(?:[0-9.\s+\-−×x*/]+?\s*=\s*)?(\d+(?:\.\d+)?)\s*cm",
    re.I,
)
_VOLUME = re.compile(r"(?:volume|V)\s*=\s*[^\n]{0,60}?(\d+(?:\.\d+)?)\s*cm", re.I)


def params_complete(viz_type: str, params: dict[str, float], *, pack: str = "") -> bool:
    vt = (viz_type or "").strip()
    if vt == "force-pressure-lab":
        return "force" in params and "area" in params
    if vt == "area-resizer" and pack == "open_box":
        has_sheet = {"sheetLength", "sheetWidth", "cut"} <= params.keys()
        has_box = {"length", "width", "height"} <= params.keys()
        return has_sheet or has_box
    if vt == "concept-explorer" and pack == "powers":
        return {"base", "exponent", "result"} <= params.keys()
    if vt == "sqrt-number-line":
        return "n" in params
    if vt == "factor-tree":
        return "x" in params
    if vt == "mensuration-cube":
        return "s" in params
    if vt == "fractions":
        return {"numerator", "denominator"} <= params.keys()
    if vt in {
        "disease-transmission-simulator",
        "human-body-system-3d",
        "plant-anatomy-lab",
        "circuit-builder",
        "water-cycle-animator",
        "acid-base-indicator-lab",
        "reaction-simulator",
    }:
        return True
    return False


def _core_for_extract(text: str) -> str:
    """Drop trailing practice/extension so example numbers don't overwrite the worked solution."""
    t = text or ""
    t = re.split(
        r"(?is)\n\s*(?:\*\*)?(?:Practice Question|Try This|Extension|Another Example)(?:\*\*)?",
        t,
        maxsplit=1,
    )[0]
    return t.strip()


def query_blocks_powers(query: str) -> bool:
    """True when the student question is roots/construction, not exponents."""
    return bool(_POWERS_BLOCK_INTENT_RE.search(query or ""))


def query_blocks_fractions(query: str) -> bool:
    """True when the question is algebra/%/probability — not a fractions lesson."""
    q = query or ""
    if _FRACTION_TOPIC_RE.search(q):
        return False
    return bool(_FRACTIONS_BLOCK_INTENT_RE.search(q))


def extract_typed_params(text: str, kind: SubjectKind) -> tuple[str | None, dict[str, float], str]:
    t = _core_for_extract(text)
    if kind == "science":
        return _extract_science(t)
    return _extract_math(t)


def _extract_science(t: str) -> tuple[str | None, dict[str, float], str]:
    params: dict[str, float] = {}
    if re.search(r"\b(pressure|P\s*=\s*F\s*/\s*A|N\s*/\s*cm)\b", t, re.I):
        forces = [float(m.group(1)) for m in _FORCE_N.finditer(t)]
        areas = [float(m.group(1)) for m in _AREA_CM.finditer(t)]
        if forces:
            params["force"] = forces[0]
        if areas:
            params["area"] = max(areas)
            if len(areas) > 1:
                params["area_b"] = min(areas)
        return "force-pressure-lab", params, "pressure"
    # Healthy habits (food/exercise) ≠ disease-spread lab — only match spread when clearly about germs/transmission
    health_habits = re.search(
        r"\b(stay\s+healthy|keep\s+\w+\s+healthy|nutritious|nutrition|balanced\s+diet|"
        r"healthy\s+(?:food|habit|lifestyle)|clean\s+water|exercise|"
        r"(?:body|we|you)\s+need\s+to\s+stay\s+healthy|what\s+(?:does|do)\s+(?:our|the|my|your)?\s*body\s+need)\b",
        t,
        re.I,
    )
    disease_spread = re.search(
        r"\b(transmission|germs?\b|pathogen|wash\s+(?:my\s+)?hands?|"
        r"spread\s+(?:of\s+)?(?:disease|infection|illness|germs?)|"
        r"how\s+(?:do\s+)?(?:germs|diseases?)\s+spread|airborne|vector-borne|contagious)\b",
        t,
        re.I,
    )
    if health_habits and not disease_spread:
        return "human-body-system-3d", params, "health"
    if disease_spread or (
        re.search(r"\b(disease|hygiene)\b", t, re.I) and not health_habits
    ):
        return "disease-transmission-simulator", params, "disease"
    if re.search(r"\bphotosynthesis\b", t, re.I):
        return "plant-anatomy-lab", params, "photosynthesis"
    if re.search(r"\b(circuit|lamp|bulb|conductor|battery)\b", t, re.I):
        return "circuit-builder", params, "circuit"
    if re.search(r"\b(acid|base|ph\b|indicator)\b", t, re.I):
        return "acid-base-indicator-lab", params, "acid_base"
    return None, params, ""


def _extract_math(t: str) -> tuple[str | None, dict[str, float], str]:
    params: dict[str, float] = {}
    if re.search(
        r"\b(open\s+box|cut\s+from\s+(?:its\s+)?(?:four\s+)?corners?|folded?\s+upwards?|"
        r"squares?\s+of\s+side\s+\d+)\b",
        t,
        re.I,
    ):
        m = _SHEET_LW.search(t)
        if m:
            params["sheetLength"] = float(m.group(1))
            params["sheetWidth"] = float(m.group(2))
        cut = _CUT_SIDE.search(t)
        if cut:
            params["cut"] = float(cut.group(1))
            params["height"] = float(cut.group(1))
        bl, bw, bh = _BOX_DIM.search(t), _BOX_W.search(t), _BOX_H.search(t)
        if bl:
            params["length"] = float(bl.group(1))
        if bw:
            params["width"] = float(bw.group(1))
        if bh:
            params["height"] = float(bh.group(1))
        # Derived folded dims win — avoids trapping on "Base length = 30 − 2×5 = 20".
        if "sheetLength" in params and "cut" in params:
            params["length"] = params["sheetLength"] - 2 * params["cut"]
        if "sheetWidth" in params and "cut" in params:
            params["width"] = params["sheetWidth"] - 2 * params["cut"]
        vol = _VOLUME.search(t)
        if vol:
            params["volume"] = float(vol.group(1))
        return "area-resizer", params, "open_box"

    # √n on number line / geometrical construction — before powers (answer often says “× itself”).
    if re.search(
        r"\b(number\s*line)\b.*\b(construct|construction|geometr|square\s*root|√|sqrt|surd)\b|"
        r"\b(construct|construction|geometr|square\s*root|√|sqrt|surd)\b.*\b(number\s*line)\b|"
        r"\b(mark|represent|locate)\b.*(?:√|sqrt|square\s*root)",
        t,
        re.I,
    ) or (
        _SQRT_ON_LINE_RE.search(t)
        and re.search(r"\b(number\s*line|construct|construction|compass|geometr)\b", t, re.I)
    ):
        m = _SQRT_ON_LINE_RE.search(t)
        if m:
            params["n"] = float(next(g for g in m.groups() if g))
        else:
            params["n"] = 5.0
        legs = _LEG_PAIR_RE.search(t)
        if legs:
            vals = [g for g in legs.groups() if g]
            if len(vals) >= 2:
                x, y = float(vals[0]), float(vals[1])
                # Larger leg along the number line (textbook OA = max).
                params["a"], params["b"] = (x, y) if x >= y else (y, x)
        if "a" not in params or "b" not in params:
            n = int(params["n"])
            # ponytail: pick first a,b with a²+b²=n (a≥b≥1); upgrade if answer always supplies legs
            found = False
            for a in range(int(n**0.5), 0, -1):
                b2 = n - a * a
                b = int(b2**0.5)
                if b * b == b2 and b >= 1:
                    params["a"] = float(a)
                    params["b"] = float(b)
                    found = True
                    break
            if not found:
                params["a"] = 2.0
                params["b"] = 1.0
        return "sqrt-number-line", params, "sqrt_line"

    if (
        re.search(
            r"\b(to\s+the\s+power|exponent|powers?\s+of|multipl(?:y|ied)\s+by\s+itself)\b|\d+\s*\^\s*\d+",
            t,
            re.I,
        )
        and not re.search(r"\b(quadratic|parabola|slope|y\s*=\s*mx)\b", t, re.I)
        and not _POWERS_BLOCK_INTENT_RE.search(t)
    ):
        m = re.search(
            r"(\d+)\s*(?:\^|to\s+the\s+power(?:\s+of)?)\s*(\d+).*?(?:=\s*|is\s+|equals\s+)(\d+)",
            t,
            re.I | re.S,
        )
        if m:
            params["base"] = float(m.group(1))
            params["exponent"] = float(m.group(2))
            params["result"] = float(m.group(3))
        else:
            m2 = re.search(r"(\d+)\s*(?:\^|to\s+the\s+power(?:\s+of)?)\s*(\d+)", t, re.I)
            if m2:
                params["base"] = float(m2.group(1))
                params["exponent"] = float(m2.group(2))
        # Prefer explicit a^b = c, else compute for small integer exponents (avoid 2x2=4 trap).
        m3 = re.search(r"(\d+)\s*\^\s*(\d+)\s*=\s*(\d+)", t)
        if m3:
            params["base"] = float(m3.group(1))
            params["exponent"] = float(m3.group(2))
            params["result"] = float(m3.group(3))
        elif "base" in params and "exponent" in params and params["exponent"] <= 12:
            try:
                params["result"] = float(params["base"] ** params["exponent"])
            except OverflowError:
                pass
        return "concept-explorer", params, "powers"

    if re.search(r"\b(perfect\s+square|prime\s+factor)\b", t, re.I):
        root = re.search(r"(\d+)\s*[×x\*]\s*\1\s*=\s*(\d+)", t)
        if root:
            params["x"] = float(root.group(2))
            params["y"] = float(root.group(2))
            params["root"] = float(root.group(1))
        else:
            for m in re.finditer(r"\b(\d{2,5})\b", t):
                n = float(m.group(1))
                if n >= 4:
                    params["x"] = n
                    params["y"] = n
                    break
        return "factor-tree", params, "perfect_square"

    if re.search(r"\b(cube|volume)\b.*\bside\b|\bside\b.*\bcube\b", t, re.I) and not re.search(
        r"\bcube\s*root\b", t, re.I
    ):
        m = re.search(
            r"side\s*(?:length\s*)?(?:of\s*)?(?:a\s+)?(?:cube\s*)?(?:=|is|:)?\s*(\d+(?:\.\d+)?)",
            t,
            re.I,
        )
        if m:
            params["s"] = float(m.group(1))
            return "mensuration-cube", params, "cube"

    # Fractions only when topic cues present — bare 3/4 in an equation answer must not win.
    if _FRACTION_TOPIC_RE.search(t) and not _FRACTIONS_BLOCK_INTENT_RE.search(t):
        m = re.search(r"(?:\\frac\{(\d+)\}\{(\d+)\}|(\d+)\s*/\s*(\d+))", t)
        if m:
            params["numerator"] = float(m.group(1) or m.group(3))
            params["denominator"] = float(m.group(2) or m.group(4))
        return "fractions", params, "fractions"

    return None, params, ""


def build_template_panel(
    viz_type: str,
    params: dict[str, float],
    *,
    kind: SubjectKind,
    class_level: str = "",
    pack: str = "",
) -> dict[str, Any] | None:
    vt = (viz_type or "").strip()
    if not vt or vt not in PRODUCTION_TYPES:
        return None
    shell = _shell_for(vt, kind, class_level, pack=pack)
    if not shell:
        return None
    return apply_params(shell, params, kind=kind, pack=pack)


def _shell_for(
    vt: str, kind: SubjectKind, class_level: str, *, pack: str = ""
) -> dict[str, Any] | None:
    if kind == "science":
        from app.services.science_experiment.visual_catalog import build_for_type

        return copy.deepcopy(build_for_type(vt, class_level))
    if pack == "open_box" and vt == "area-resizer":
        from app.services.math_lesson.textbook_catalog import _lesson, _open_box_spec

        return copy.deepcopy(
            _lesson(
                "Open Box from a Rectangular Sheet",
                "Cut equal squares from corners and fold to make an open box.",
                "Box length = sheet length − 2×cut; volume = l × w × h.",
                _open_box_spec(),
                class_level or "Class 8",
            )
        )
    if pack == "powers" and vt == "concept-explorer":
        return _powers_shell(class_level)
    if pack == "sqrt_line" and vt == "sqrt-number-line":
        return _sqrt_number_line_shell(class_level)
    from app.services.math_lesson.visual_catalog import match_math_visualization

    hint = {
        "factor-tree": "perfect square prime factorization factor tree 81",
        "mensuration-cube": "cube side volume surface area",
        "fractions": "fractions numerator denominator pizza",
        "circle": "quarter turns half turn circle",
        "algebra-stepper": "solve linear equation 2x + 5 = 15",
        "linear-graph": "slope intercept linear graph y = mx + c",
        "area-resizer": "area of rectangle and triangle",
        "concept-explorer": "mathematics concept",
        "number-line": "integers on number line",
        "sqrt-number-line": "construct square root 5 on the number line geometrical construction",
        "shape-lab": "square rectangle sides corners",
        "statistics-lab": "mean median mode statistics",
        "mensuration-cylinder": "cylinder radius height volume",
    }.get(vt, vt)
    lesson = copy.deepcopy(match_math_visualization(hint, class_level))
    got = str((lesson.get("visualization") or {}).get("visualizationType") or "")
    if got != vt and pack == "powers":
        return _powers_shell(class_level)
    if got != vt and pack == "sqrt_line":
        return _sqrt_number_line_shell(class_level)
    if got != vt:
        lesson.setdefault("visualization", {})["visualizationType"] = vt
    return lesson


def _sqrt_number_line_shell(class_level: str = "") -> dict[str, Any]:
    return {
        "conceptName": "Square Root on the Number Line",
        "classLevel": class_level or "Class 8",
        "learningObjective": "Construct √n on the number line with a right triangle and compass arc.",
        "conceptExplanation": "If a² + b² = n, the hypotenuse length is √n; transfer it to the line with an arc.",
        "visualization": {
            "visualizationType": "sqrt-number-line",
            "title": "Construct √n on the number line",
            "description": "Build a right triangle with legs a and b, then swing an arc to mark √n.",
            "renderMode": "2d",
            "sliders": [
                {"id": "n", "label": "n (√n)", "min": 2, "max": 50, "step": 1, "default": 5},
                {"id": "a", "label": "Leg a", "min": 1, "max": 20, "step": 1, "default": 2},
                {"id": "b", "label": "Leg b", "min": 1, "max": 20, "step": 1, "default": 1},
            ],
            "liveCalculations": [
                {"id": "check", "label": "a² + b²", "formula": "a * a + b * b", "unit": ""},
            ],
            "buttons": [{"id": "reset", "label": "Reset", "action": "reset"}],
        },
        "guidedExploration": [
            "Why is OB equal to √n?",
            "Where does the arc meet the number line?",
        ],
    }


def _powers_shell(class_level: str = "") -> dict[str, Any]:
    return {
        "conceptName": "Exponents and Powers",
        "classLevel": class_level or "Class 8",
        "learningObjective": "See base multiplied by itself exponent times.",
        "conceptExplanation": "a^n means multiply a by itself n times.",
        "visualization": {
            "visualizationType": "concept-explorer",
            "title": "Power Play",
            "description": "Change base and exponent; compare to the solution value.",
            "renderMode": "2d",
            "sliders": [
                {"id": "base", "label": "Base", "min": 1, "max": 12, "step": 1, "default": 2},
                {"id": "exponent", "label": "Exponent", "min": 0, "max": 8, "step": 1, "default": 5},
                {
                    "id": "result",
                    "label": "Value (from solution)",
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "default": 32,
                },
            ],
            "liveCalculations": [
                {"id": "check", "label": "base^exponent", "formula": "base ** exponent", "unit": ""},
            ],
            "buttons": [
                {"id": "animate", "label": "▶ Expand", "action": "animate"},
                {"id": "reset", "label": "Reset", "action": "reset"},
            ],
        },
        "guidedExploration": ["What happens when the exponent increases by 1?"],
    }


def apply_params(
    panel: dict[str, Any] | None,
    params: dict[str, float],
    *,
    kind: SubjectKind,
    pack: str = "",
) -> dict[str, Any] | None:
    if not panel:
        return None
    if not params:
        return panel
    out = copy.deepcopy(panel)
    if kind == "math" or "visualization" in out:
        key = "visualization"
    else:
        key = "experiment"
    if key not in out and "experiment" in out:
        key = "experiment"
    if key not in out:
        return out
    block = dict(out.get(key) or {})
    sliders = []
    for s in block.get("sliders") or []:
        if not isinstance(s, dict):
            sliders.append(s)
            continue
        s2 = dict(s)
        sid = str(s2.get("id") or "")
        if sid in params:
            val = float(params[sid])
            s2["default"] = val
            try:
                smin = float(s2.get("min", 0))
                smax = float(s2.get("max", val))
            except (TypeError, ValueError):
                smin, smax = 0.0, val
            smin = min(smin, val)
            smax = max(smax, val)
            if sid == "area" and "area_b" in params:
                smax = max(smax, float(params["area"]), float(params["area_b"]))
            if sid == "result":
                smax = max(smax, val, 1000)
            s2["min"] = smin
            s2["max"] = smax
        sliders.append(s2)
    block["sliders"] = sliders
    if kind == "science" and block.get("experimentType") == "force-pressure-lab":
        views = dict(block.get("threeViews") or {})
        sci = dict(views.get("scientific") or {})
        sci["equation"] = sci.get("equation") or "P = F / A"
        if "force" in params and "area" in params and params["area"]:
            p = params["force"] / params["area"]
            sci["description"] = f"P = {params['force']} / {params['area']} = {p:g}"
        views["scientific"] = sci
        block["threeViews"] = views
    if pack == "powers" and "base" in params and "exponent" in params and "result" in params:
        block["description"] = (
            f"{int(params['base'])}^{int(params['exponent'])} = {int(params['result'])}"
        )
    out[key] = block
    return out


def normalize_to_template(
    panel: dict[str, Any] | None,
    *,
    kind: SubjectKind,
    class_level: str,
    preferred_type: str | None,
    params: dict[str, float],
    pack: str,
) -> dict[str, Any] | None:
    panel_vt = ""
    if panel:
        if kind == "math":
            panel_vt = str((panel.get("visualization") or {}).get("visualizationType") or "")
        else:
            panel_vt = str((panel.get("experiment") or panel).get("experimentType") or "")

    # Dual-signal: question-chosen geometry / √n lab wins over answer-only Power Play.
    if pack == "powers" and panel_vt in _KEEP_OVER_POWERS:
        return apply_params(panel, params, kind=kind, pack="") or panel

    if preferred_type and preferred_type in PRODUCTION_TYPES:
        built = build_template_panel(
            preferred_type, params, kind=kind, class_level=class_level, pack=pack
        )
        if built is not None:
            return built
    if not panel:
        return None
    if panel_vt in PRODUCTION_TYPES and params:
        return apply_params(panel, params, kind=kind, pack=pack) or panel
    return panel
