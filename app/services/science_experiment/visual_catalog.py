"""Catalog builders for every science visualizationType in the ontology."""

from __future__ import annotations

from typing import Any, Callable

from app.services.science_experiment.topic_ontology import canonicalize_type


def _tier(class_level: str) -> str:
    from app.services.math_lesson.math_tokens import parse_class_num

    n = parse_class_num(class_level)
    if n <= 3:
        return "elementary"
    if n <= 5:
        return "primary"
    if n <= 8:
        return "middle"
    return "advanced"


def _display(class_level: str) -> str:
    from app.services.math_lesson.math_tokens import parse_class_num

    return f"Class {parse_class_num(class_level)}"


def _views(rw: str, micro: str, sci: str, equation: str = "") -> dict:
    return {
        "realWorld": {"title": "What you see", "description": rw, "narration": rw},
        "microscopic": {"title": "What's inside", "description": micro, "narration": micro},
        "scientific": {
            "title": "Why it happens",
            "description": sci,
            "narration": sci,
            "equation": equation,
        },
    }


def _exp(
    etype: str,
    title: str,
    description: str,
    class_level: str,
    *,
    subject: str = "evs",
    kind: str = "concept",
    sliders: list | None = None,
    calcs: list | None = None,
    views: dict | None = None,
    procedure: list | None = None,
    safety: list | None = None,
    apparatus: list | None = None,
    aim: str = "",
    hypothesis: str = "",
) -> dict:
    buttons = [
        {"id": "animate", "label": "▶ Play", "action": "animate"},
        {"id": "reset", "label": "Reset", "action": "reset"},
    ]
    steps = []
    if procedure:
        for i, p in enumerate(procedure):
            steps.append(
                {
                    "id": f"s{i+1}",
                    "instruction": p if p.startswith("In this simulation") else f"In this simulation: {p}",
                    "safetyNote": None,
                    "durationSeconds": None,
                }
            )
    return {
        "experimentType": etype,
        "title": title,
        "description": description,
        "gradeTier": _tier(class_level),
        "subject": subject,
        "kind": kind,
        "aim": aim or description,
        "apparatus": apparatus or [],
        "safetyLevel": "caution" if kind == "experiment" else "none",
        "threeViews": views
        or _views(
            "Observe the interactive model.",
            "Look at the hidden structure or particles.",
            "Connect the observation to the science idea.",
        ),
        "sliders": sliders or [
            {"id": "intensity", "label": "Explore", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"}
        ],
        "buttons": buttons,
        "liveCalculations": calcs or [],
        "colors": {
            "primary": "#2d70b3",
            "secondary": "#388c46",
            "accent": "#e08a2b",
            "background": "#fafafa",
            "text": "#1a1a1f",
        },
        "safetyNotes": safety
        or (
            ["This is a virtual simulation — do not try hazardous steps at home."]
            if kind == "experiment"
            else []
        ),
        "procedure": procedure or [],
        "procedureSteps": steps,
        "hypothesisPrompt": hypothesis
        or "What do you predict will change when you move the slider?",
        "expectedObservation": "Watch labels and the Observe box update as you explore.",
        "explanation": description,
        "relatedVisualizationType": etype,
    }


def _lesson(
    concept: str,
    objective: str,
    explanation: str,
    experiment: dict,
    class_level: str,
    *,
    subject: str = "evs",
    kind: str = "concept",
    guided: list | None = None,
) -> dict[str, Any]:
    return {
        "conceptName": concept,
        "classLevel": _display(class_level),
        "subject": subject,
        "kind": kind,
        "learningObjective": objective,
        "conceptExplanation": explanation,
        "experiment": experiment,
        "guidedExploration": guided
        or [
            "Move each slider slowly and watch what changes.",
            "Switch between What you see / What's inside / Why it happens.",
            "Say the big idea in your own words.",
        ],
    }


# --- builders per visualizationType -------------------------------------------------

def plant_anatomy(cl: str) -> dict:
    return _exp(
        "plant-anatomy-lab",
        "Plant parts & photosynthesis",
        "Explore root, stem, leaf, and how light and water help plants make food.",
        cl,
        subject="biology",
        kind="experiment",
        sliders=[
            {"id": "light", "label": "Sunlight", "min": 0, "max": 100, "step": 5, "default": 70, "unit": "%"},
            {"id": "water", "label": "Water", "min": 0, "max": 100, "step": 5, "default": 60, "unit": "%"},
            {"id": "co2", "label": "CO₂", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[{"id": "rate", "label": "Food-making rate", "formula": "light * water * co2 / 10000", "unit": ""}],
        views=_views(
            "A plant with labeled root, stem, and leaf.",
            "Chloroplasts in the leaf use light to make sugar.",
            "Photosynthesis needs light, water, and carbon dioxide.",
            "6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂",
        ),
        procedure=["Increase sunlight and watch the leaf glow.", "Lower water — notice the rate drop.", "Compare all three views."],
        apparatus=["Virtual plant", "Light control", "Water control"],
        aim="See how plants need light, water, and air to make food.",
    )


def human_body_basics(cl: str) -> dict:
    return _exp(
        "human-body-basics",
        "Our body — senses & parts",
        "Learn the main body parts and the five senses with clear labels.",
        cl,
        subject="evs",
        sliders=[{"id": "focus", "label": "Highlight part", "min": 0, "max": 5, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "A simple labeled body outline.",
            "Sense organs collect information.",
            "Each sense helps us understand the world safely.",
        ),
    )


def animal_habitat(cl: str) -> dict:
    return _exp(
        "animal-habitat-explorer",
        "Animals & habitats",
        "Match animals to land, water, or air homes.",
        cl,
        subject="evs",
        sliders=[{"id": "habitat", "label": "Habitat", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Animals living in different places.",
            "Body features help animals survive (fins, wings, fur).",
            "Habitat = the place that provides food, water, and shelter.",
        ),
    )


def food_chain(cl: str) -> dict:
    return _exp(
        "food-chain-simple",
        "Food chain",
        "Energy flows from sun → plant → herbivore → carnivore.",
        cl,
        subject="biology",
        sliders=[{"id": "level", "label": "Chain step", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "A simple food chain you can step through.",
            "Energy packets move from one living thing to the next.",
            "Producers make food; consumers eat; decomposers recycle.",
        ),
    )


def water_cycle(cl: str) -> dict:
    return _exp(
        "water-cycle-animator",
        "Water cycle",
        "Watch evaporation, condensation, and precipitation.",
        cl,
        subject="evs",
        kind="concept",
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": 0, "max": 40, "step": 1, "default": 28, "unit": "°C"},
            {"id": "humidity", "label": "Humidity", "min": 20, "max": 100, "step": 5, "default": 60, "unit": "%"},
        ],
        calcs=[{"id": "evap", "label": "Evaporation", "formula": "temperature * humidity / 100", "unit": ""}],
        views=_views(
            "Sun heats water → clouds form → rain returns.",
            "Water molecules speed up when heated and rise as vapor.",
            "The Sun drives the continuous water cycle.",
        ),
    )


def human_system(cl: str) -> dict:
    return _exp(
        "human-body-system-3d",
        "Human body systems",
        "Switch systems: digestion, breathing, circulation, and more.",
        cl,
        subject="biology",
        sliders=[
            {"id": "system", "label": "System", "min": 0, "max": 4, "step": 1, "default": 0, "unit": ""},
            {"id": "flow", "label": "Flow speed", "min": 1, "max": 10, "step": 1, "default": 5, "unit": ""},
        ],
        views=_views(
            "Body outline with the selected system highlighted.",
            "Pathways carry food, air, blood, or signals.",
            "Organs work together as systems to keep us alive.",
        ),
    )


def life_cycle(cl: str) -> dict:
    return _exp(
        "life-cycle-animator",
        "Life cycle",
        "Step through stages of growth and reproduction (age-appropriate).",
        cl,
        subject="biology",
        sliders=[{"id": "stage", "label": "Stage", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Circular stages from start to adult.",
            "Each stage has a job in the life cycle.",
            "Living things grow, reproduce, and continue the cycle.",
        ),
    )


def weather(cl: str) -> dict:
    return _exp(
        "weather-climate-simulator",
        "Weather & climate",
        "Change temperature and pressure to see weather patterns.",
        cl,
        subject="physics",
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": -5, "max": 45, "step": 1, "default": 25, "unit": "°C"},
            {"id": "pressure", "label": "Pressure", "min": 980, "max": 1040, "step": 1, "default": 1010, "unit": "hPa"},
        ],
        views=_views(
            "Icons for sun, cloud, rain, and wind respond to controls.",
            "Warm moist air rises; cool air can bring rain.",
            "Weather is day-to-day; climate is the long-term pattern.",
        ),
    )


def simple_machines(cl: str) -> dict:
    return _exp(
        "simple-machines-lab",
        "Simple machines",
        "Explore lever and wheel — how they make work easier.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "effort", "label": "Effort force", "min": 1, "max": 20, "step": 1, "default": 5, "unit": "N"},
            {"id": "distance", "label": "Effort arm", "min": 1, "max": 10, "step": 1, "default": 4, "unit": ""},
        ],
        calcs=[{"id": "load", "label": "Load moved", "formula": "effort * distance", "unit": ""}],
        views=_views(
            "A lever and a wheel you can adjust.",
            "Force is spread over a longer distance.",
            "Machines change force or direction — they do not create energy.",
        ),
        procedure=["Increase effort arm and notice load capacity.", "Compare short vs long lever."],
        apparatus=["Virtual lever", "Virtual wheel"],
    )


def states_matter(cl: str) -> dict:
    return _exp(
        "states-of-matter-lab",
        "States of matter",
        "Heat particles and watch solid → liquid → gas.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[{"id": "temperature", "label": "Temperature", "min": -20, "max": 120, "step": 2, "default": 20, "unit": "°C"}],
        calcs=[{"id": "speed", "label": "Particle speed", "formula": "(temperature + 20) / 40", "unit": ""}],
        views=_views(
            "Particles packed, sliding, or flying free.",
            "Heat raises kinetic energy of particles.",
            "State depends on how strongly particles hold together.",
        ),
        procedure=["Warm slowly past melting, then boiling.", "Cool again and watch packing return."],
        apparatus=["Virtual particle box", "Temperature slider"],
    )


def ecosystem(cl: str) -> dict:
    return _exp(
        "ecosystem-lab",
        "Ecosystem & environment",
        "See how plants, animals, and waste connect in an ecosystem.",
        cl,
        subject="biology",
        sliders=[
            {"id": "plants", "label": "Plants", "min": 0, "max": 10, "step": 1, "default": 5, "unit": ""},
            {"id": "animals", "label": "Animals", "min": 0, "max": 10, "step": 1, "default": 3, "unit": ""},
        ],
        views=_views(
            "A mini habitat with living and non-living parts.",
            "Energy and materials cycle between organisms.",
            "Removing one part can affect the whole system.",
        ),
    )


def material_sorting(cl: str) -> dict:
    return _exp(
        "material-sorting-lab",
        "Sorting materials",
        "Sort materials by properties: metal / non-metal, hard / soft.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[{"id": "item", "label": "Sample", "min": 0, "max": 5, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Cards of everyday materials to classify.",
            "Properties come from how atoms are arranged.",
            "Grouping by properties helps us choose materials wisely.",
        ),
        procedure=["Select a sample.", "Decide the group from its properties.", "Check the feedback."],
        apparatus=["Material cards", "Sort bins"],
    )


def separation(cl: str) -> dict:
    return _exp(
        "separation-techniques-lab",
        "Separation techniques",
        "Filtration and evaporation step by step (virtual lab).",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[{"id": "step", "label": "Procedure step", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Mixture → filter → filtrate and residue.",
            "Particles larger than pores stay behind.",
            "We separate mixtures using physical differences.",
        ),
        procedure=["Pour mixture into funnel (simulation).", "Collect filtrate.", "Evaporate to recover dissolved solid."],
        apparatus=["Beaker", "Funnel", "Filter paper"],
        safety=["Virtual lab only — no real chemicals."],
    )


def reaction(cl: str) -> dict:
    return _exp(
        "reaction-simulator",
        "Chemical reactions",
        "Watch reactants recombine into products.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[
            {"id": "temperature", "label": "Temperature", "min": 10, "max": 80, "step": 5, "default": 25, "unit": "°C"},
            {"id": "concentration", "label": "Concentration", "min": 10, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[{"id": "rate", "label": "Reaction rate", "formula": "temperature * concentration / 50", "unit": ""}],
        views=_views(
            "Molecules meet and rearrange.",
            "Bonds break and new bonds form.",
            "A chemical change makes new substances.",
            "reactants → products",
        ),
        procedure=["Raise temperature — rate increases.", "Compare before/after particles."],
        apparatus=["Virtual molecules"],
    )


def motion_grapher(cl: str) -> dict:
    return _exp(
        "motion-grapher",
        "Motion & graphs",
        "An object moves on a track while a distance–time graph updates.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "speed", "label": "Speed", "min": 0, "max": 20, "step": 1, "default": 5, "unit": "m/s"},
            {"id": "time", "label": "Time", "min": 0, "max": 10, "step": 0.5, "default": 4, "unit": "s"},
        ],
        calcs=[{"id": "distance", "label": "Distance", "formula": "speed * time", "unit": "m"}],
        views=_views(
            "Cart on a track + live graph.",
            "Steeper graph line means faster motion.",
            "Distance = speed × time (uniform motion).",
            "s = v t",
        ),
        procedure=["Set speed, scrub time, read distance.", "Compare steep vs flat graph."],
    )


def light_optics(cl: str) -> dict:
    return _exp(
        "light-optics-bench",
        "Light: reflection & refraction",
        "Drag the angle and see reflected / refracted rays.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "angle", "label": "Incident angle", "min": 5, "max": 80, "step": 1, "default": 40, "unit": "°"},
            {"id": "mode", "label": "Mode (0=mirror,1=lens)", "min": 0, "max": 1, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Light source, mirror/lens, and screen.",
            "Rays bounce or bend at the surface.",
            "i = r for mirrors; denser media bend light toward the normal.",
        ),
        procedure=["Change angle of incidence.", "Toggle mirror vs lens mode."],
        apparatus=["Ray box", "Mirror/lens", "Screen"],
    )


def circuit(cl: str) -> dict:
    return _exp(
        "circuit-builder",
        "Electric circuits",
        "Complete the circuit — current, bulb brightness, and Ohm's law.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "voltage", "label": "Voltage", "min": 1, "max": 12, "step": 1, "default": 6, "unit": "V"},
            {"id": "resistance", "label": "Resistance", "min": 1, "max": 20, "step": 1, "default": 6, "unit": "Ω"},
            {"id": "switch", "label": "Switch (0=off,1=on)", "min": 0, "max": 1, "step": 1, "default": 1, "unit": ""},
        ],
        calcs=[
            {"id": "current", "label": "Current", "formula": "voltage / resistance", "unit": "A"},
            {"id": "power", "label": "Power", "formula": "voltage * voltage / resistance", "unit": "W"},
        ],
        views=_views(
            "Battery, switch, bulb, and wires.",
            "Charges flow when the path is closed.",
            "Ohm's law: V = I R",
            "V = I R",
        ),
        procedure=["Close the switch.", "Raise resistance — bulb dims.", "Read ammeter values."],
        apparatus=["Battery", "Bulb", "Switch", "Resistor"],
        safety=["Virtual circuit only."],
    )


def magnet_field(cl: str) -> dict:
    return _exp(
        "magnet-field-visualizer",
        "Magnetic fields",
        "See field lines around a bar magnet; change strength and distance.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "strength", "label": "Magnet strength", "min": 20, "max": 100, "step": 5, "default": 70, "unit": "%"},
            {"id": "distance", "label": "Distance", "min": 1, "max": 8, "step": 1, "default": 3, "unit": ""},
        ],
        calcs=[{"id": "field", "label": "Field strength", "formula": "strength / (distance * distance)", "unit": ""}],
        views=_views(
            "Bar magnet with field lines to compass needles.",
            "Field is strongest near the poles.",
            "Field strength falls quickly with distance.",
        ),
    )


def microscope(cl: str) -> dict:
    return _exp(
        "microscope-lab",
        "Microscope lab",
        "Focus and magnify onion cell / microbe specimens.",
        cl,
        subject="biology",
        kind="experiment",
        sliders=[
            {"id": "focus", "label": "Focus", "min": 0, "max": 100, "step": 5, "default": 70, "unit": "%"},
            {"id": "magnification", "label": "Magnification", "min": 40, "max": 400, "step": 40, "default": 100, "unit": "×"},
            {"id": "specimen", "label": "Specimen", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Eyepiece view of a slide.",
            "Cells and tiny organisms become visible.",
            "Microscopes reveal structure below naked-eye scale.",
        ),
        procedure=["Start low power.", "Focus until sharp.", "Increase magnification carefully."],
    )


def acid_base(cl: str) -> dict:
    return _exp(
        "acid-base-indicator-lab",
        "Acids, bases & indicators",
        "Slide pH and watch indicator colour change.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[{"id": "ph", "label": "pH", "min": 0, "max": 14, "step": 0.5, "default": 7, "unit": ""}],
        views=_views(
            "Beaker with universal indicator colour.",
            "H⁺ / OH⁻ balance sets pH.",
            "Acids pH < 7, bases pH > 7, neutral = 7.",
        ),
        procedure=["Move pH from acid to base.", "Note colour at 7."],
        apparatus=["Beaker", "Virtual indicator"],
        safety=["Virtual chemicals only — never taste unknowns."],
    )


def cell_structure(cl: str) -> dict:
    return _exp(
        "cell-structure-3d",
        "Cell structure",
        "Plant vs animal cell — click organelles to learn their jobs.",
        cl,
        subject="biology",
        sliders=[
            {"id": "plant", "label": "Plant cell (0=animal,1=plant)", "min": 0, "max": 1, "step": 1, "default": 1, "unit": ""},
            {"id": "organelle", "label": "Highlight organelle", "min": 0, "max": 5, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Labeled cell with nucleus and organelles.",
            "Each organelle has a specialized function.",
            "The cell is the basic unit of life.",
        ),
    )


def crystal(cl: str) -> dict:
    return _exp(
        "crystal-lattice-3d",
        "Crystal lattices & metals",
        "Compare lattice packing and why metals conduct.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[{"id": "lattice", "label": "Structure", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Repeating unit cells you can rotate (2D schematic).",
            "Free electrons explain metallic conduction.",
            "Structure decides hardness and conductivity.",
        ),
    )


def combustion(cl: str) -> dict:
    return _exp(
        "combustion-flame-lab",
        "Combustion & flame",
        "See luminous vs non-luminous flame zones.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[
            {"id": "air", "label": "Air supply", "min": 0, "max": 100, "step": 5, "default": 40, "unit": "%"},
            {"id": "fuel", "label": "Fuel", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Candle/Bunsen-style flame zones.",
            "Complete burning needs enough oxygen.",
            "Combustion is a rapid reaction with oxygen releasing heat/light.",
        ),
        safety=["Virtual flame only."],
        procedure=["Increase air — flame turns blue.", "Compare yellow sooty flame."],
    )


def electrolysis(cl: str) -> dict:
    return _exp(
        "electrolysis-lab",
        "Electrolysis",
        "Ions move; gases form at electrodes when current is on.",
        cl,
        subject="chemistry",
        kind="experiment",
        sliders=[
            {"id": "current", "label": "Current on", "min": 0, "max": 1, "step": 1, "default": 1, "unit": ""},
            {"id": "voltage", "label": "Voltage", "min": 1, "max": 12, "step": 1, "default": 6, "unit": "V"},
        ],
        views=_views(
            "Cell with anode and cathode.",
            "Positive ions move one way; negative the other.",
            "Electrical energy drives a non-spontaneous chemical change.",
        ),
        safety=["Virtual electrolysis only."],
    )


def force_pressure(cl: str) -> dict:
    return _exp(
        "force-pressure-lab",
        "Force, pressure & friction",
        "Change force and area — pressure updates live.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "force", "label": "Force", "min": 1, "max": 200, "step": 1, "default": 20, "unit": "N"},
            {"id": "area", "label": "Area", "min": 1, "max": 500, "step": 1, "default": 100, "unit": "cm²"},
            {"id": "friction", "label": "Roughness", "min": 0, "max": 10, "step": 1, "default": 3, "unit": ""},
        ],
        calcs=[{"id": "pressure", "label": "Pressure", "formula": "force / area", "unit": "N/cm²"}],
        views=_views(
            "Block on a surface with force arrow.",
            "Same force on smaller area → larger pressure.",
            "P = F / A; friction opposes sliding.",
            "P = F / A",
        ),
    )


def sound_wave(cl: str) -> dict:
    return _exp(
        "sound-wave-lab",
        "Sound waves",
        "Oscilloscope view — pitch vs loudness.",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "frequency", "label": "Frequency (pitch)", "min": 100, "max": 800, "step": 10, "default": 440, "unit": "Hz"},
            {"id": "amplitude", "label": "Amplitude (loudness)", "min": 10, "max": 100, "step": 5, "default": 50, "unit": "%"},
        ],
        calcs=[{"id": "wavelength", "label": "Wavelength", "formula": "34300 / frequency", "unit": "cm"}],
        views=_views(
            "Wave on a screen like an oscilloscope.",
            "Particles vibrate; energy travels as a wave.",
            "Higher frequency → higher pitch; larger amplitude → louder.",
            "v = f λ",
        ),
    )


def solar_system(cl: str) -> dict:
    return _exp(
        "solar-system-3d",
        "Solar system",
        "Planets orbit the Sun (illustrative scale, not true-to-scale).",
        cl,
        subject="physics",
        sliders=[{"id": "speed", "label": "Orbit speed", "min": 1, "max": 10, "step": 1, "default": 3, "unit": ""}],
        views=_views(
            "Sun at center with orbiting planets.",
            "Gravity keeps planets in orbit.",
            "Illustrative distances — textbooks also use not-to-scale diagrams.",
        ),
    )


def molecule(cl: str) -> dict:
    return _exp(
        "molecule-builder-3d",
        "Atoms & molecules",
        "Ball-and-stick models with CPK colours (O red, H white, C grey, N blue).",
        cl,
        subject="chemistry",
        sliders=[{"id": "molecule", "label": "Molecule", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "H₂O, CO₂, CH₄, or NaCl fragment.",
            "Atoms bond in fixed geometries.",
            "Molecules are groups of atoms held by chemical bonds.",
        ),
    )


def atom_structure(cl: str) -> dict:
    return _exp(
        "atom-structure-3d",
        "Structure of the atom",
        "Bohr model — change atomic number and watch shells fill.",
        cl,
        subject="chemistry",
        sliders=[{"id": "Z", "label": "Atomic number", "min": 1, "max": 20, "step": 1, "default": 6, "unit": ""}],
        views=_views(
            "Nucleus with electrons in shells.",
            "Protons = electrons in a neutral atom.",
            "NCERT uses shell model (not quantum orbitals) at this level.",
        ),
    )


def gravitation(cl: str) -> dict:
    return _exp(
        "gravitation-orbit-simulator",
        "Gravitation & orbits",
        "Change mass and distance — orbit path updates (illustrative).",
        cl,
        subject="physics",
        sliders=[
            {"id": "mass", "label": "Central mass", "min": 1, "max": 10, "step": 1, "default": 5, "unit": ""},
            {"id": "distance", "label": "Orbit radius", "min": 2, "max": 8, "step": 1, "default": 4, "unit": ""},
        ],
        views=_views(
            "Central body and orbiting satellite.",
            "Stronger gravity pulls harder; closer orbits are tighter.",
            "Gravity provides the centripetal force for orbit.",
        ),
    )


def energy_xform(cl: str) -> dict:
    return _exp(
        "energy-transformation-lab",
        "Energy transformations",
        "Follow energy packets along a chain (e.g. Sun → panel → bulb).",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[{"id": "stage", "label": "Stage", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Animated flow of energy forms.",
            "Energy changes form but total is conserved (ideal).",
            "Useful energy can be stored, transferred, or wasted as heat.",
        ),
    )


def tissue(cl: str) -> dict:
    return _exp(
        "tissue-explorer-3d",
        "Tissues",
        "Compare plant and animal tissue types with labels.",
        cl,
        subject="biology",
        sliders=[
            {"id": "kind", "label": "0=plant,1=animal", "min": 0, "max": 1, "step": 1, "default": 0, "unit": ""},
            {"id": "type", "label": "Tissue type", "min": 0, "max": 3, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Micrograph-style panels of tissues.",
            "Similar cells work together as a tissue.",
            "Structure matches function (e.g. muscle contracts).",
        ),
    )


def disease(cl: str) -> dict:
    return _exp(
        "disease-transmission-simulator",
        "How germs can spread",
        "Pick air, touch, or insect to see how germs can spread — and how to stop them.",
        cl,
        subject="biology",
        sliders=[{"id": "mode", "label": "How they spread (0 air · 1 touch · 2 insect)", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""}],
        views=_views(
            "Two people — germs travel along the dashed path.",
            "Germs are tiny living things that can make us ill.",
            "Block the path (wash hands, vaccines) and spread drops.",
        ),
    )


def periodic(cl: str) -> dict:
    return _exp(
        "periodic-table-explorer",
        "Periodic table",
        "Browse groups and periods; open an element card.",
        cl,
        subject="chemistry",
        sliders=[
            {"id": "Z", "label": "Atomic number", "min": 1, "max": 20, "step": 1, "default": 11, "unit": ""},
        ],
        views=_views(
            "Interactive mini periodic grid (first 20).",
            "Properties repeat in periods/groups.",
            "Atomic number orders the modern table.",
        ),
    )


def eye_optics(cl: str) -> dict:
    return _exp(
        "eye-optics-3d",
        "Human eye & vision",
        "Ray paths for normal, myopia, hyperopia + corrective lens.",
        cl,
        subject="physics",
        sliders=[
            {"id": "defect", "label": "0=normal,1=myopia,2=hyperopia", "min": 0, "max": 2, "step": 1, "default": 0, "unit": ""},
            {"id": "lens", "label": "Corrective lens on", "min": 0, "max": 1, "step": 1, "default": 0, "unit": ""},
        ],
        views=_views(
            "Eye cross-section with light rays to retina.",
            "Lens focuses rays; defects miss the retina.",
            "Spectacles add a lens to correct focus.",
        ),
    )


def electromagnet(cl: str) -> dict:
    return _exp(
        "electromagnet-induction-lab",
        "Magnetic effects of current",
        "Move a magnet through a coil — induced current direction (Lenz).",
        cl,
        subject="physics",
        kind="experiment",
        sliders=[
            {"id": "position", "label": "Magnet position", "min": 0, "max": 100, "step": 5, "default": 20, "unit": "%"},
            {"id": "turns", "label": "Coil turns", "min": 10, "max": 100, "step": 10, "default": 50, "unit": ""},
        ],
        views=_views(
            "Coil and bar magnet.",
            "Changing flux induces current.",
            "Induced current opposes the change (Lenz's law).",
        ),
    )


def punnett(cl: str) -> dict:
    return _exp(
        "heredity-punnett-lab",
        "Heredity — Punnett square",
        "Place alleles and see genotype / phenotype ratios.",
        cl,
        subject="biology",
        sliders=[
            {"id": "parentA", "label": "Parent A (0=AA,1=Aa,2=aa)", "min": 0, "max": 2, "step": 1, "default": 1, "unit": ""},
            {"id": "parentB", "label": "Parent B (0=AA,1=Aa,2=aa)", "min": 0, "max": 2, "step": 1, "default": 1, "unit": ""},
        ],
        views=_views(
            "Punnett square fills as parents change.",
            "Alleles combine randomly in offspring.",
            "Genotype vs phenotype — ratios from the square.",
        ),
    )


def concept(cl: str) -> dict:
    return _exp(
        "concept-explorer",
        "Science explorer",
        "A clear labeled diagram to explore this idea with controls.",
        cl,
        subject="evs",
        sliders=[
            {"id": "value1", "label": "Factor A", "min": 0, "max": 100, "step": 5, "default": 40, "unit": ""},
            {"id": "value2", "label": "Factor B", "min": 0, "max": 100, "step": 5, "default": 60, "unit": ""},
        ],
        calcs=[{"id": "effect", "label": "Combined effect", "formula": "(value1 + value2) / 2", "unit": ""}],
    )


BUILDERS: dict[str, Callable[[str], dict]] = {
    "plant-anatomy-lab": plant_anatomy,
    "human-body-basics": human_body_basics,
    "animal-habitat-explorer": animal_habitat,
    "food-chain-simple": food_chain,
    "water-cycle-animator": water_cycle,
    "human-body-system-3d": human_system,
    "life-cycle-animator": life_cycle,
    "weather-climate-simulator": weather,
    "simple-machines-lab": simple_machines,
    "states-of-matter-lab": states_matter,
    "ecosystem-lab": ecosystem,
    "material-sorting-lab": material_sorting,
    "separation-techniques-lab": separation,
    "reaction-simulator": reaction,
    "motion-grapher": motion_grapher,
    "light-optics-bench": light_optics,
    "circuit-builder": circuit,
    "electricity-circuit-lab": circuit,
    "magnet-field-visualizer": magnet_field,
    "microscope-lab": microscope,
    "acid-base-indicator-lab": acid_base,
    "titration-lab": acid_base,
    "cell-structure-3d": cell_structure,
    "crystal-lattice-3d": crystal,
    "combustion-flame-lab": combustion,
    "electrolysis-lab": electrolysis,
    "force-pressure-lab": force_pressure,
    "sound-wave-lab": sound_wave,
    "solar-system-3d": solar_system,
    "molecule-builder-3d": molecule,
    "atom-structure-3d": atom_structure,
    "gravitation-orbit-simulator": gravitation,
    "energy-transformation-lab": energy_xform,
    "tissue-explorer-3d": tissue,
    "disease-transmission-simulator": disease,
    "periodic-table-explorer": periodic,
    "eye-optics-3d": eye_optics,
    "electromagnet-induction-lab": electromagnet,
    "heredity-punnett-lab": punnett,
    "concept-explorer": concept,
}

_META: dict[str, tuple[str, str, str, str]] = {
    # etype: concept, objective, explanation, subject
    "plant-anatomy-lab": ("Plants & photosynthesis", "See how plants use light and water.", "Leaves make food using sunlight.", "biology"),
    "human-body-basics": ("Human body basics", "Name parts and senses.", "Sense organs help us explore safely.", "evs"),
    "animal-habitat-explorer": ("Animals & habitats", "Link animals to habitats.", "Habitat meets needs of living things.", "evs"),
    "food-chain-simple": ("Food chain", "Trace energy in a chain.", "Energy flows from producers to consumers.", "biology"),
    "water-cycle-animator": ("Water cycle", "Follow water through the cycle.", "The Sun drives evaporation and rainfall.", "evs"),
    "human-body-system-3d": ("Body systems", "Explore organ systems.", "Systems work together to keep us alive.", "biology"),
    "life-cycle-animator": ("Life cycles", "Step through life stages.", "Organisms grow and continue life cycles.", "biology"),
    "weather-climate-simulator": ("Weather", "Relate controls to weather.", "Weather changes with heat and pressure.", "physics"),
    "simple-machines-lab": ("Simple machines", "See how levers help.", "Machines make work easier.", "physics"),
    "states-of-matter-lab": ("States of matter", "Heat particles through states.", "Temperature changes particle motion.", "chemistry"),
    "ecosystem-lab": ("Ecosystems", "See living links in nature.", "Parts of an ecosystem depend on each other.", "biology"),
    "material-sorting-lab": ("Materials", "Sort by properties.", "Properties decide how we use materials.", "chemistry"),
    "separation-techniques-lab": ("Separation", "Run a virtual filtration.", "We separate mixtures by physical differences.", "chemistry"),
    "reaction-simulator": ("Reactions", "Watch chemical change.", "New substances form in reactions.", "chemistry"),
    "motion-grapher": ("Motion", "Read a distance–time graph.", "Graphs show how position changes with time.", "physics"),
    "light-optics-bench": ("Light", "Explore reflection/refraction.", "Light travels in rays and can bend or bounce.", "physics"),
    "circuit-builder": ("Circuits", "Build and measure a circuit.", "Current needs a closed path.", "physics"),
    "magnet-field-visualizer": ("Magnets", "Visualize field lines.", "Magnets have poles and fields.", "physics"),
    "microscope-lab": ("Microscope", "Focus a virtual slide.", "Microscopes reveal tiny structures.", "biology"),
    "acid-base-indicator-lab": ("Acids & bases", "Read pH with colour.", "Indicators show acidic or basic nature.", "chemistry"),
    "cell-structure-3d": ("Cells", "Tour a cell.", "Organelles have specialized jobs.", "biology"),
    "crystal-lattice-3d": ("Lattices", "Compare structures.", "Atomic arrangement decides properties.", "chemistry"),
    "combustion-flame-lab": ("Combustion", "Compare flame zones.", "Burning needs fuel and oxygen.", "chemistry"),
    "electrolysis-lab": ("Electrolysis", "See ions move.", "Current can drive chemical change.", "chemistry"),
    "force-pressure-lab": ("Force & pressure", "Link force, area, pressure.", "Pressure = force ÷ area.", "physics"),
    "sound-wave-lab": ("Sound", "Change pitch and loudness.", "Sound is a vibration traveling as a wave.", "physics"),
    "solar-system-3d": ("Solar system", "Watch orbits.", "Gravity keeps planets orbiting the Sun.", "physics"),
    "molecule-builder-3d": ("Molecules", "Inspect molecule models.", "Atoms bond to form molecules.", "chemistry"),
    "atom-structure-3d": ("Atoms", "Build Bohr shells.", "Electrons occupy shells around the nucleus.", "chemistry"),
    "gravitation-orbit-simulator": ("Gravitation", "Tune an orbit.", "Gravity governs motion of planets.", "physics"),
    "energy-transformation-lab": ("Energy", "Follow energy forms.", "Energy changes form along a chain.", "physics"),
    "tissue-explorer-3d": ("Tissues", "Compare tissue types.", "Tissues are groups of similar cells.", "biology"),
    "disease-transmission-simulator": ("Health", "See how germs can spread.", "Stopping the path keeps people healthy.", "biology"),
    "periodic-table-explorer": ("Periodic table", "Explore elements.", "Elements are arranged by atomic number.", "chemistry"),
    "eye-optics-3d": ("Human eye", "Correct vision defects.", "The eye lens focuses light on the retina.", "physics"),
    "electromagnet-induction-lab": ("Induction", "Induce a current.", "Changing magnetism can create current.", "physics"),
    "heredity-punnett-lab": ("Heredity", "Fill a Punnett square.", "Alleles combine to give offspring traits.", "biology"),
    "concept-explorer": ("Science idea", "Explore with controls.", "Observe, predict, and explain.", "evs"),
}


def build_for_type(etype: str, class_level: str = "", *, kind: str | None = None) -> dict[str, Any]:
    canon = canonicalize_type(etype)
    builder = BUILDERS.get(canon, concept)
    exp = builder(class_level)
    if kind:
        exp["kind"] = kind
    meta = _META.get(canon, _META["concept-explorer"])
    concept_name, objective, explanation, subject = meta
    return _lesson(
        concept_name,
        objective,
        explanation,
        exp,
        class_level,
        subject=subject,
        kind=kind or exp.get("kind") or "concept",
    )


def registered_visualization_types() -> frozenset[str]:
    return frozenset(BUILDERS.keys())
