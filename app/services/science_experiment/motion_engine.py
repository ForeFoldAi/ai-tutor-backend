"""
Science motion engine — computes derived observables for interactive experiments.

Formulas mirror frontend experiment-utils.ts so sliders produce consistent
real-world, microscopic, and scientific readouts across grade bands.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MotionResult:
    values: dict[str, float]
    labels: dict[str, str]
    color_hint: str = ""


def _num(class_level: str) -> int:
    m = re.search(r"class[_\s]*(\d{1,2})", (class_level or "").replace("_", " "), re.I)
    if m:
        return max(1, min(12, int(m.group(1))))
    return 8


def grade_tier(class_level: str) -> str:
    n = _num(class_level)
    if n <= 3:
        return "elementary"
    if n <= 5:
        return "primary"
    if n <= 8:
        return "middle"
    return "advanced"


def compute_motion(
    experiment_type: str,
    variables: dict[str, float],
    *,
    class_level: str = "",
) -> MotionResult:
    """Return computed observables for the active experiment type."""
    v = variables
    et = (experiment_type or "").lower()
    tier = grade_tier(class_level)

    if et == "photosynthesis":
        light = v.get("light", 50)
        co2 = v.get("co2", 50)
        rate = round(light * co2 / 100, 1)
        o2 = round(rate * 0.8, 1)
        return MotionResult(
            {"rate": rate, "oxygen": o2, "glucose": round(rate * 0.6, 1)},
            {"rate": "Photosynthesis rate", "oxygen": "O₂ released", "glucose": "Glucose formed"},
        )

    if et == "respiration":
        activity = v.get("activity", 50)
        o2_in = round(activity * 0.5, 1)
        co2_out = round(activity * 0.45, 1)
        return MotionResult(
            {"oxygen_in": o2_in, "co2_out": co2_out},
            {"oxygen_in": "O₂ absorbed", "co2_out": "CO₂ released"},
        )

    if et == "acids-bases":
        ph = v.get("ph", 7)
        if ph < 3:
            color = "#ef4444"
        elif ph < 6:
            color = "#f97316"
        elif ph <= 8:
            color = "#22c55e"
        elif ph <= 11:
            color = "#3b82f6"
        else:
            color = "#8b5cf6"
        return MotionResult({"ph": ph}, {"ph": "pH level"}, color_hint=color)

    if et == "chemical-reaction":
        temp = v.get("temperature", 25)
        conc = v.get("concentration", 50)
        rate = round((temp / 25) * (conc / 50) * 10, 2)
        return MotionResult(
            {"reaction_rate": rate, "energy": round(rate * 4, 1)},
            {"reaction_rate": "Reaction rate", "energy": "Energy released (kJ)"},
        )

    if et == "states-of-matter":
        temp = v.get("temperature", 20)
        if temp < 0:
            state, speed = "solid", 0.2
        elif temp < 100:
            state, speed = "liquid", 0.5 + temp / 200
        else:
            state, speed = "gas", 1.0 + (temp - 100) / 50
        return MotionResult(
            {"particle_speed": round(speed, 2), "state_index": {"solid": 0, "liquid": 1, "gas": 2}[state]},
            {"particle_speed": "Particle speed", "state_index": f"State: {state}"},
        )

    if et == "electricity":
        voltage = v.get("voltage", 6)
        resistance = max(1, v.get("resistance", 10))
        current = round(voltage / resistance, 2)
        power = round(voltage * current, 2)
        return MotionResult(
            {"current": current, "power": power},
            {"current": "Current (A)", "power": "Power (W)"},
        )

    if et == "force-motion":
        force = v.get("force", 10)
        mass = max(1, v.get("mass", 5))
        accel = round(force / mass, 2)
        return MotionResult(
            {"acceleration": accel, "velocity": round(accel * 2, 2)},
            {"acceleration": "Acceleration (m/s²)", "velocity": "Velocity after 2s (m/s)"},
        )

    if et == "heat-transfer":
        hot = v.get("hot_temp", 80)
        cold = v.get("cold_temp", 20)
        cond = v.get("conductivity", 50)
        flow = round(abs(hot - cold) * cond / 100, 1)
        return MotionResult(
            {"heat_flow": flow, "equilibrium": round((hot + cold) / 2, 1)},
            {"heat_flow": "Heat flow rate", "equilibrium": "Equilibrium temp (°C)"},
        )

    if et == "sound":
        freq = v.get("frequency", 440)
        amplitude = v.get("amplitude", 50)
        wavelength = round(343 / max(1, freq), 3)
        return MotionResult(
            {"wavelength": wavelength, "loudness": amplitude},
            {"wavelength": "Wavelength (m)", "loudness": "Loudness"},
        )

    if et == "plant-growth":
        days = v.get("days", 7)
        water = v.get("water", 50)
        height = round(days * water / 20, 1)
        return MotionResult(
            {"height": height, "days": days},
            {"height": "Plant height (cm)", "days": "Days elapsed"},
        )

    if et == "seed-germination":
        days = v.get("days", 5)
        moisture = v.get("moisture", 60)
        sprout = min(100, round(days * moisture / 8, 0))
        return MotionResult(
            {"germination": sprout, "root_length": round(sprout / 10, 1)},
            {"germination": "Germination (%)", "root_length": "Root length (cm)"},
        )

    if et == "blood-circulation":
        heart_rate = v.get("heart_rate", 72)
        return MotionResult(
            {"heart_rate": heart_rate, "cycles": round(heart_rate / 60, 2)},
            {"heart_rate": "Heart rate (bpm)", "cycles": "Beats per second"},
        )

    if et == "magnetism":
        distance = v.get("distance", 5)
        strength = v.get("strength", 80)
        field = round(strength / max(1, distance ** 2) * 10, 1)
        return MotionResult(
            {"field_strength": field},
            {"field_strength": "Field strength (relative)"},
        )

    if et == "water-cycle":
        temp = v.get("temperature", 30)
        humidity = v.get("humidity", 60)
        evap = round(temp * humidity / 100, 1)
        return MotionResult(
            {"evaporation": evap, "condensation": round(evap * 0.7, 1)},
            {"evaporation": "Evaporation rate", "condensation": "Condensation rate"},
        )

    if et == "refraction":
        angle = v.get("incident_angle", 45)
        n = v.get("refractive_index", 1.33)
        rad = math.radians(angle)
        refracted = round(math.degrees(math.asin(min(1, math.sin(rad) / n))), 1)
        return MotionResult(
            {"incident": angle, "refracted": refracted},
            {"incident": "Incident angle (°)", "refracted": "Refracted angle (°)"},
        )

    if et == "light-reflection":
        angle = v.get("angle", 45)
        return MotionResult(
            {"incident": angle, "reflected": angle},
            {"incident": "Incident angle (°)", "reflected": "Reflected angle (°)"},
        )

    # Generic fallback
    a = v.get("value1", 5)
    b = v.get("value2", 3)
    return MotionResult(
        {"sum": a + b, "product": a * b},
        {"sum": "Combined effect", "product": "Interaction strength"},
    )


def enrich_live_calculations(
    experiment_type: str,
    variables: dict[str, float],
    *,
    class_level: str = "",
) -> list[dict[str, Any]]:
    """Build live calculation rows for the experiment panel."""
    motion = compute_motion(experiment_type, variables, class_level=class_level)
    rows: list[dict[str, Any]] = []
    for key, val in motion.values.items():
        label = motion.labels.get(key, key)
        unit = ""
        if "°" in label or key.endswith("angle"):
            unit = "°"
        elif "rate" in key or "flow" in key:
            unit = tier_unit(tier)
        rows.append({"id": key, "label": label, "formula": str(val), "unit": unit, "_computed": val})
    return rows


def tier_unit(tier: str) -> str:
    return {"elementary": "", "primary": "", "middle": "units", "advanced": "SI units"}[tier]
