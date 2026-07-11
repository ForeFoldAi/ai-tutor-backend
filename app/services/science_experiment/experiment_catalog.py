"""
Science experiment catalog — Class 1–10 experiment type matching.

Each experiment provides three synchronized views:
  realWorld → what students see in a lab
  microscopic → particles / cells / charges
  scientific → concepts, equations, reasoning
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.science_experiment.motion_engine import grade_tier

_CLASS_NUM_RE = re.compile(r"class[_\s]*(\d{1,2})", re.I)


def _parse_class_num(class_level: str) -> int:
    if not class_level:
        return 6
    m = _CLASS_NUM_RE.search(class_level.replace("_", " "))
    if m:
        return max(1, min(10, int(m.group(1))))
    return 6


def _display_level(class_level: str) -> str:
    return class_level.replace("_", " ") if class_level else "Class 6"


def _views(
    real: tuple[str, str],
    micro: tuple[str, str],
    sci: tuple[str, str],
    *,
    equation: str = "",
) -> dict[str, Any]:
    return {
        "realWorld": {"title": real[0], "description": real[1], "narration": real[1]},
        "microscopic": {"title": micro[0], "description": micro[1], "narration": micro[1]},
        "scientific": {
            "title": sci[0],
            "description": sci[1],
            "narration": sci[1],
            "equation": equation,
        },
    }


def _exp(
    etype: str,
    title: str,
    description: str,
    *,
    class_level: str = "",
    sliders: list[dict] | None = None,
    buttons: list[dict] | None = None,
    calcs: list[dict] | None = None,
    views: dict | None = None,
    safety: list[str] | None = None,
    procedure: list[str] | None = None,
) -> dict[str, Any]:
    tier = grade_tier(class_level)
    return {
        "experimentType": etype,
        "title": title,
        "description": description,
        "gradeTier": tier,
        "threeViews": views or _views(
            ("Lab observation", description),
            ("Microscopic view", "See particles and interactions."),
            ("Scientific explanation", "Understand the concept behind the observation."),
        ),
        "sliders": sliders or [],
        "buttons": buttons or [
            {"id": "animate", "label": "▶ Play", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
        "liveCalculations": calcs or [],
        "colors": {
            "primary": "#0EA5E9",
            "secondary": "#10B981",
            "accent": "#F59E0B",
            "background": "#F0F9FF",
            "text": "#0F172A",
        },
        "studentInteractions": [
            {"id": "observe", "type": "watch", "description": "Watch all three views sync as you change variables."}
        ],
        "safetyNotes": safety or [],
        "procedure": procedure or [],
        "hypothesisPrompt": "What do you predict will happen when you change the variables?",
    }


def _lesson(
    concept: str,
    objective: str,
    explanation: str,
    experiment: dict,
    level: str,
) -> dict[str, Any]:
    return {
        "conceptName": concept,
        "classLevel": level,
        "learningObjective": objective,
        "conceptExplanation": explanation,
        "experiment": experiment,
        "guidedExploration": [
            "Switch between Real World, Microscopic, and Scientific views.",
            "Change one variable at a time — what do you observe?",
            "Can you predict the outcome before pressing Play?",
        ],
    }


# --- Experiment spec builders ---

def _plant_growth(class_level: str) -> dict:
    return _exp(
        "plant-growth",
        "Plant Growth Time-lapse",
        "Watch a seedling grow as water and time increase.",
        class_level=class_level,
        sliders=[
            {"id": "days", "label": "Days", "min": 1, "max": 30, "step": 1, "default": 7, "unit": "days"},
            {"id": "water", "label": "Water", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[{"id": "height", "label": "Plant height", "formula": "days * water / 20", "unit": "cm"}],
        views=_views(
            ("Growing plant in soil", "A seedling rises taller each day with enough water."),
            ("Root hair cells absorbing water", "Water moves into root cells by osmosis."),
            ("Growth = f(water, light, nutrients)", "Plants need water, sunlight, and minerals for photosynthesis-driven growth."),
        ),
        procedure=["Place seed in moist soil.", "Provide light daily.", "Measure height over time."],
    )


def _seed_germination(class_level: str) -> dict:
    return _exp(
        "seed-germination",
        "Seed Germination Timeline",
        "Track how moisture and warmth trigger sprouting.",
        class_level=class_level,
        sliders=[
            {"id": "days", "label": "Days", "min": 1, "max": 14, "step": 1, "default": 5},
            {"id": "moisture", "label": "Soil moisture", "min": 0, "max": 100, "step": 5, "default": 60, "unit": "%"},
        ],
        calcs=[{"id": "germination", "label": "Germination", "formula": "days * moisture / 8", "unit": "%"}],
        views=_views(
            ("Seed swelling and sprouting", "The seed coat cracks; a root emerges first."),
            ("Embryo activating cell division", "Stored food breaks down; cells divide rapidly."),
            ("Germination needs water + oxygen + suitable temperature", "Enzymes activate stored food for the embryo."),
        ),
    )


def _photosynthesis(class_level: str) -> dict:
    return _exp(
        "photosynthesis",
        "Photosynthesis Lab",
        "Molecules enter and leave the leaf as you adjust light and CO₂.",
        class_level=class_level,
        sliders=[
            {"id": "light", "label": "Light intensity", "min": 0, "max": 100, "step": 5, "default": 70, "unit": "%"},
            {"id": "co2", "label": "CO₂ level", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[
            {"id": "rate", "label": "Photosynthesis rate", "formula": "light * co2 / 100", "unit": ""},
            {"id": "oxygen", "label": "O₂ released", "formula": "rate * 0.8", "unit": ""},
        ],
        views=_views(
            ("Green leaf in sunlight", "CO₂ enters stomata; O₂ bubbles out in water."),
            ("Chloroplasts capturing light", "Light energy splits water; CO₂ is fixed into glucose."),
            ("6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂", "Light energy converts to chemical energy in chloroplasts."),
            equation="6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂",
        ),
    )


def _respiration(class_level: str) -> dict:
    return _exp(
        "respiration",
        "Cellular Respiration",
        "Animated O₂ and CO₂ exchange in living cells.",
        class_level=class_level,
        sliders=[
            {"id": "activity", "label": "Activity level", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[
            {"id": "oxygen_in", "label": "O₂ absorbed", "formula": "activity * 0.5", "unit": ""},
            {"id": "co2_out", "label": "CO₂ released", "formula": "activity * 0.45", "unit": ""},
        ],
        views=_views(
            ("Breathing and gas exchange", "We inhale O₂ and exhale CO₂ during respiration."),
            ("Mitochondria releasing energy", "Glucose is broken down; ATP is produced."),
            ("C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + Energy", "Respiration releases stored chemical energy."),
            equation="C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP",
        ),
    )


def _digestion(class_level: str) -> dict:
    return _exp(
        "digestion",
        "Digestive Journey",
        "Follow food through organs and see chemical breakdown.",
        class_level=class_level,
        sliders=[
            {"id": "stage", "label": "Digestion stage", "min": 0, "max": 5, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            ("Food traveling through organs", "Mouth → oesophagus → stomach → intestines."),
            ("Enzymes breaking macromolecules", "Proteins, fats, and starch are hydrolysed."),
            ("Mechanical + chemical digestion", "Teeth and enzymes convert food into absorbable nutrients."),
        ),
        procedure=["Chew food (mechanical).", "Stomach acid and enzymes act.", "Nutrients absorbed in small intestine."],
    )


def _blood_circulation(class_level: str) -> dict:
    return _exp(
        "blood-circulation",
        "Blood Circulation",
        "Red and blue flows show oxygenated vs deoxygenated blood.",
        class_level=class_level,
        sliders=[
            {"id": "heart_rate", "label": "Heart rate", "min": 60, "max": 120, "step": 1, "default": 72, "unit": "bpm"},
        ],
        calcs=[{"id": "cycles", "label": "Beats per second", "formula": "heart_rate / 60", "unit": "Hz"}],
        views=_views(
            ("Heart pumping blood", "Red = oxygen-rich; blue = oxygen-poor blood."),
            ("Red blood cells carrying O₂", "Haemoglobin binds oxygen in lungs, releases in tissues."),
            ("Double circulation: pulmonary + systemic", "Heart is a four-chambered pump."),
        ),
    )


def _magnetism(class_level: str) -> dict:
    return _exp(
        "magnetism",
        "Magnetic Field Explorer",
        "Moving field lines around a bar magnet.",
        class_level=class_level,
        sliders=[
            {"id": "strength", "label": "Magnet strength", "min": 10, "max": 100, "step": 5, "default": 80},
            {"id": "distance", "label": "Distance", "min": 1, "max": 10, "step": 0.5, "default": 3, "unit": "cm"},
        ],
        calcs=[{"id": "field_strength", "label": "Field strength", "formula": "strength / (distance * distance) * 10", "unit": ""}],
        views=_views(
            ("Iron filings around a magnet", "Field lines run from N pole to S pole."),
            ("Aligned atomic magnetic domains", "Domains align in ferromagnetic materials."),
            ("Field strength ∝ 1/r²", "Like poles repel; unlike poles attract."),
        ),
    )


def _electricity(class_level: str) -> dict:
    return _exp(
        "electricity",
        "Electric Circuit Lab",
        "Glowing electron flow through wires and components.",
        class_level=class_level,
        sliders=[
            {"id": "voltage", "label": "Voltage", "min": 1, "max": 12, "step": 1, "default": 6, "unit": "V"},
            {"id": "resistance", "label": "Resistance", "min": 1, "max": 50, "step": 1, "default": 10, "unit": "Ω"},
        ],
        calcs=[
            {"id": "current", "label": "Current", "formula": "voltage / resistance", "unit": "A"},
            {"id": "power", "label": "Power", "formula": "voltage * current", "unit": "W"},
        ],
        views=_views(
            ("Bulb glowing in a circuit", "Electrons flow from negative to positive terminal."),
            ("Free electrons drifting in wire", "Drift velocity is slow; energy transfers quickly."),
            ("Ohm's law: V = IR", "Current is proportional to voltage at constant resistance."),
            equation="V = I × R",
        ),
        safety=["Never connect high voltage without supervision.", "Dry hands before touching circuits."],
    )


def _light_reflection(class_level: str) -> dict:
    return _exp(
        "light-reflection",
        "Light Reflection",
        "Animated rays bouncing off a mirror.",
        class_level=class_level,
        sliders=[
            {"id": "angle", "label": "Incident angle", "min": 0, "max": 80, "step": 5, "default": 45, "unit": "°"},
        ],
        calcs=[
            {"id": "incident", "label": "Incident angle", "formula": "angle", "unit": "°"},
            {"id": "reflected", "label": "Reflected angle", "formula": "angle", "unit": "°"},
        ],
        views=_views(
            ("Ray hitting a plane mirror", "Angle of incidence equals angle of reflection."),
            ("Photons bouncing off smooth surface", "Specular reflection from smooth surfaces."),
            ("∠i = ∠r", "Law of reflection."),
            equation="∠incident = ∠reflected",
        ),
    )


def _refraction(class_level: str) -> dict:
    return _exp(
        "refraction",
        "Light Refraction",
        "Ray bending when entering water or glass.",
        class_level=class_level,
        sliders=[
            {"id": "incident_angle", "label": "Incident angle", "min": 10, "max": 80, "step": 5, "default": 45, "unit": "°"},
            {"id": "refractive_index", "label": "Refractive index", "min": 1.0, "max": 2.0, "step": 0.05, "default": 1.33},
        ],
        calcs=[{"id": "refracted", "label": "Refracted angle", "formula": "asin(sin(incident_angle) / refractive_index)", "unit": "°"}],
        views=_views(
            ("Pencil appearing bent in water", "Light slows in denser medium; ray bends."),
            ("Light wavefronts slowing at boundary", "Speed change causes direction change."),
            ("Snell's law: n₁ sin θ₁ = n₂ sin θ₂", "Refraction depends on refractive indices."),
            equation="n₁ sin θ₁ = n₂ sin θ₂",
        ),
    )


def _acids_bases(class_level: str) -> dict:
    return _exp(
        "acids-bases",
        "Acids & Bases Indicator Lab",
        "Color-changing solutions as pH shifts.",
        class_level=class_level,
        sliders=[
            {"id": "ph", "label": "pH", "min": 0, "max": 14, "step": 0.5, "default": 7, "unit": ""},
        ],
        calcs=[{"id": "ph", "label": "pH level", "formula": "ph", "unit": ""}],
        views=_views(
            ("Indicator colour in beaker", "Red in acid, green neutral, blue/purple in base."),
            ("H⁺ and OH⁻ ion concentration", "pH = −log[H⁺]; more H⁺ means more acidic."),
            ("Acid + Base → Salt + Water", "Neutralisation forms salt and water."),
            equation="pH = −log₁₀[H⁺]",
        ),
        safety=["Wear goggles when handling acids and bases.", "Never taste lab chemicals."],
    )


def _chemical_reaction(class_level: str) -> dict:
    return _exp(
        "chemical-reaction",
        "Chemical Reaction Simulator",
        "Particle-level bond breaking and forming.",
        class_level=class_level,
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": 10, "max": 100, "step": 5, "default": 25, "unit": "°C"},
            {"id": "concentration", "label": "Concentration", "min": 10, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[
            {"id": "reaction_rate", "label": "Reaction rate", "formula": "(temperature / 25) * (concentration / 50) * 10", "unit": ""},
            {"id": "energy", "label": "Energy released", "formula": "reaction_rate * 4", "unit": "kJ"},
        ],
        views=_views(
            ("Magnesium ribbon burning brightly", "A bright white flame and white ash form."),
            ("Mg atoms reacting with O₂ molecules", "Bonds break and new MgO ionic lattice forms."),
            ("2Mg + O₂ → 2MgO (combination reaction)", "Heat and light energy are released."),
            equation="2Mg + O₂ → 2MgO",
        ),
        safety=["Keep flammable materials away from flame.", "Use tongs to hold burning metals."],
    )


def _states_of_matter(class_level: str) -> dict:
    return _exp(
        "states-of-matter",
        "States of Matter",
        "Molecules move faster or slower with temperature.",
        class_level=class_level,
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": -20, "max": 150, "step": 5, "default": 25, "unit": "°C"},
        ],
        calcs=[{"id": "particle_speed", "label": "Particle speed", "formula": "temperature_based", "unit": "rel"}],
        views=_views(
            ("Ice, water, and steam", "Same substance in three states."),
            ("Particles vibrating, sliding, or flying", "Kinetic energy increases with temperature."),
            ("Heating adds kinetic energy → phase change", "Melting and boiling are phase transitions."),
        ),
    )


def _heat_transfer(class_level: str) -> dict:
    return _exp(
        "heat-transfer",
        "Heat Transfer Lab",
        "Animated conduction, convection, and radiation.",
        class_level=class_level,
        sliders=[
            {"id": "hot_temp", "label": "Hot side", "min": 40, "max": 120, "step": 5, "default": 80, "unit": "°C"},
            {"id": "cold_temp", "label": "Cold side", "min": 0, "max": 39, "step": 1, "default": 20, "unit": "°C"},
            {"id": "conductivity", "label": "Conductivity", "min": 10, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[{"id": "heat_flow", "label": "Heat flow", "formula": "abs(hot_temp - cold_temp) * conductivity / 100", "unit": ""}],
        views=_views(
            ("Heat moving through a metal rod", "Hot particles collide with cooler neighbours."),
            ("Molecular collisions transferring energy", "Kinetic energy transfers particle to particle."),
            ("Heat flows hot → cold until equilibrium", "Q = mcΔT for absorbed heat."),
            equation="Q = mcΔT",
        ),
    )


def _water_cycle(class_level: str) -> dict:
    return _exp(
        "water-cycle",
        "Water Cycle Animation",
        "Continuous evaporation, condensation, and precipitation.",
        class_level=class_level,
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": 0, "max": 45, "step": 1, "default": 30, "unit": "°C"},
            {"id": "humidity", "label": "Humidity", "min": 0, "max": 100, "step": 5, "default": 60, "unit": "%"},
        ],
        calcs=[{"id": "evaporation", "label": "Evaporation rate", "formula": "temperature * humidity / 100", "unit": ""}],
        views=_views(
            ("Sun heating ocean and lakes", "Water evaporates, forms clouds, falls as rain."),
            ("Water molecules escaping liquid surface", "Evaporation from high-energy surface molecules."),
            ("Solar energy drives the hydrologic cycle", "Energy from the Sun powers evaporation."),
        ),
    )


def _sound(class_level: str) -> dict:
    return _exp(
        "sound",
        "Sound Waves Lab",
        "Vibrating particles and wave propagation.",
        class_level=class_level,
        sliders=[
            {"id": "frequency", "label": "Frequency", "min": 100, "max": 2000, "step": 50, "default": 440, "unit": "Hz"},
            {"id": "amplitude", "label": "Amplitude", "min": 10, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[
            {"id": "wavelength", "label": "Wavelength", "formula": "343 / frequency", "unit": "m"},
            {"id": "loudness", "label": "Loudness", "formula": "amplitude", "unit": "%"},
        ],
        views=_views(
            ("Tuning fork vibrating", "Compressions and rarefactions travel through air."),
            ("Particles oscillating back and forth", "Longitudinal wave — particles vibrate parallel to direction."),
            ("v = fλ", "Speed of sound ≈ 343 m/s in air at 20°C."),
            equation="v = f × λ",
        ),
    )


def _force_motion(class_level: str) -> dict:
    return _exp(
        "force-motion",
        "Force & Motion Simulator",
        "Motion vectors and graphs from applied force.",
        class_level=class_level,
        sliders=[
            {"id": "force", "label": "Force", "min": 0, "max": 50, "step": 1, "default": 10, "unit": "N"},
            {"id": "mass", "label": "Mass", "min": 1, "max": 20, "step": 1, "default": 5, "unit": "kg"},
        ],
        calcs=[
            {"id": "acceleration", "label": "Acceleration", "formula": "force / mass", "unit": "m/s²"},
            {"id": "velocity", "label": "Velocity (2s)", "formula": "acceleration * 2", "unit": "m/s"},
        ],
        views=_views(
            ("Cart accelerating on a track", "Bigger force → faster acceleration."),
            ("Net force causes change in motion", "Friction opposes motion."),
            ("F = ma (Newton's 2nd law)", "Acceleration is proportional to net force."),
            equation="F = m × a",
        ),
    )


def _solar_system(class_level: str) -> dict:
    return _exp(
        "solar-system",
        "Solar System Explorer",
        "Interactive orbits of planets around the Sun.",
        class_level=class_level,
        sliders=[
            {"id": "planet", "label": "Planet index", "min": 0, "max": 7, "step": 1, "default": 2, "unit": ""},
            {"id": "speed", "label": "Orbit speed", "min": 0.2, "max": 2, "step": 0.1, "default": 1, "unit": "×"},
        ],
        views=_views(
            ("Planets orbiting the Sun", "Inner planets orbit faster than outer ones."),
            ("Gravitational attraction", "The Sun's mass curves spacetime; planets follow orbits."),
            ("Kepler's laws of planetary motion", "Orbit period increases with distance from the Sun."),
        ),
    )


def _human_organs(class_level: str) -> dict:
    return _exp(
        "human-organs",
        "Human Body Explorer",
        "Rotatable organ systems with labels.",
        class_level=class_level,
        sliders=[
            {"id": "system", "label": "Body system", "min": 0, "max": 4, "step": 1, "default": 0, "unit": ""},
            {"id": "rotation", "label": "Rotation", "min": 0, "max": 360, "step": 15, "default": 0, "unit": "°"},
        ],
        views=_views(
            ("Organ system overview", "Digestive, circulatory, respiratory, nervous, skeletal."),
            ("Tissues and organs working together", "Organs are made of specialised tissues."),
            ("Structure relates to function", "Each organ's shape supports its role."),
        ),
    )


def _concept_explorer(class_level: str) -> dict:
    return _exp(
        "concept-explorer",
        "Science Concept Explorer",
        "Explore variables and observe outcomes.",
        class_level=class_level,
        sliders=[
            {"id": "value1", "label": "Variable A", "min": 0, "max": 100, "step": 5, "default": 50},
            {"id": "value2", "label": "Variable B", "min": 0, "max": 100, "step": 5, "default": 30},
        ],
        calcs=[{"id": "effect", "label": "Combined effect", "formula": "value1 + value2", "unit": ""}],
    )


# (pattern, spec_builder, concept, objective, explanation)
_EXPERIMENT_RULES: list[tuple[re.Pattern[str], Callable[[str], dict], str, str, str]] = [
    (re.compile(r"\b(plant\s+growth|growing\s+plant|seedling)\b", re.I), _plant_growth, "Plant Growth", "Observe how plants grow over time.", "Water and light drive growth."),
    (re.compile(r"\b(seed\s+germinat|sprout|germinat)\b", re.I), _seed_germination, "Seed Germination", "Track seed sprouting.", "Moisture triggers embryo growth."),
    (re.compile(r"\b(photosynthesis|chlorophyll|stomata)\b", re.I), _photosynthesis, "Photosynthesis", "See molecules enter and leave leaves.", "Plants make food using light."),
    (re.compile(r"\b(respiration|breathing|mitochondria)\b", re.I), _respiration, "Respiration", "Gas exchange in living cells.", "Cells release energy from glucose."),
    (re.compile(r"\b(digestion|digestive|stomach|intestine|enzyme)\b", re.I), _digestion, "Digestion", "Follow food through the body.", "Food is broken into absorbable nutrients."),
    (re.compile(r"\b(blood\s+circulat|heart\s+beat|cardiovascular|artery|vein)\b", re.I), _blood_circulation, "Blood Circulation", "See blood flow through the heart.", "Heart pumps blood in two circuits."),
    (re.compile(r"\b(magnet|magnetic\s+field|magnetism|compass)\b", re.I), _magnetism, "Magnetism", "Explore magnetic field lines.", "Magnets have north and south poles."),
    (re.compile(r"\b(electric|circuit|current|voltage|resistance|bulb|lamp|wire|conductor|insulator|battery|glow|torch)\b", re.I), _electricity, "Electricity", "Build and observe a circuit.", "Current flows when circuit is complete."),
    (re.compile(r"\b(reflect|reflection|mirror)\b", re.I), _light_reflection, "Light Reflection", "Rays bouncing off surfaces.", "Angle of incidence equals reflection."),
    (re.compile(r"\b(refract|refraction|bending\s+of\s+light|prism)\b", re.I), _refraction, "Refraction", "Light bending in water or glass.", "Light slows in denser media."),
    (re.compile(r"\b(acid|base|alkali|ph\s*indicator|litmus|neutrali[sz])\b", re.I), _acids_bases, "Acids & Bases", "Watch colour change with pH.", "pH measures acidity."),
    (re.compile(r"\b(chemical\s+reaction|burning|combustion|magnesium|bond|react)\b", re.I), _chemical_reaction, "Chemical Reactions", "Particle-level bond changes.", "Reactants form new products."),
    (re.compile(r"\b(solid|liquid|gas|states?\s+of\s+matter|melting|boiling|freezing)\b", re.I), _states_of_matter, "States of Matter", "See molecules speed up or slow down.", "Temperature changes kinetic energy."),
    (re.compile(r"\b(heat\s+transfer|conduction|convection|radiation|thermal)\b", re.I), _heat_transfer, "Heat Transfer", "Heat flowing between objects.", "Heat moves from hot to cold."),
    (re.compile(r"\b(water\s+cycle|evaporation|condensation|precipitation|rain\s+cycle)\b", re.I), _water_cycle, "Water Cycle", "Continuous environmental water movement.", "Solar energy drives the cycle."),
    (re.compile(r"\b(sound\s+wave|vibration|frequency|pitch|loudness|echo)\b", re.I), _sound, "Sound", "Vibrating particles and waves.", "Sound is a longitudinal wave."),
    (re.compile(r"\b(force|motion|newton|friction|accelerat|velocity|momentum)\b", re.I), _force_motion, "Force & Motion", "See how force changes motion.", "F = ma governs acceleration."),
    (re.compile(r"\b(solar\s+system|planet|orbit|earth\s+revolution|sun\s+and\s+moon)\b", re.I), _solar_system, "Solar System", "Planets orbiting the Sun.", "Gravity keeps planets in orbit."),
    (re.compile(r"\b(human\s+organ|body\s+system|anatomy|heart|lung|kidney|brain|organs?\s+and\s+systems?)\b", re.I), _human_organs, "Human Organs", "Explore organ systems.", "Organs work together in systems."),
    (re.compile(r"\b(experiment|lab|observe|demonstration|activity)\b", re.I), _concept_explorer, "Science Experiment", "Hands-on exploration.", "Observe, predict, test."),
]


def match_science_experiment(query: str, class_level: str = "") -> dict[str, Any]:
    """Match query to the best science experiment. Always returns a lesson dict."""
    from app.services.science_experiment.visual_matcher import pick_best_experiment_rule

    q = (query or "").strip()
    level = _display_level(class_level)
    if not q:
        lesson = _lesson("Science Explorer", "Explore science interactively.", "Change variables and observe.", _concept_explorer(class_level), level)
        return lesson

    picked = pick_best_experiment_rule(q, _EXPERIMENT_RULES)
    if picked:
        spec_fn, concept, objective, explanation = picked
        return _lesson(concept, objective, explanation, spec_fn(class_level), level)

    return _lesson(
        "Science Explorer",
        "Explore this science topic interactively.",
        "Use the sliders and watch all three views.",
        _concept_explorer(class_level),
        level,
    )
