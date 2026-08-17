#!/usr/bin/env python3
"""Deterministic visual prompt corpus for Anima text-encoder bridge calibration.

The corpus is deliberately model-agnostic: it mixes Danbooru-like short tags,
compact natural-language relations, multi-subject spatial descriptions, camera,
composition, materials, lighting, text/rendering concepts, and longer prompts.
It does not depend on copyrighted character or artist-name lists.
"""
from __future__ import annotations

import random
import re
from typing import Any, Iterable

SUBJECTS = [
    "1girl", "1boy", "1woman", "1man", "two women", "two men", "woman and man",
    "solo", "duo", "trio", "group portrait", "three women", "two girls and one boy",
    "crowd of students", "android girl", "humanoid robot",
    "elf woman", "witch", "knight", "pilot", "scientist", "mechanic", "musician",
    "dancer", "office worker", "traveler", "athlete", "chef", "astronaut",
]
HAIR = [
    "black hair", "brown hair", "blonde hair", "white hair", "silver hair", "red hair",
    "orange hair", "blue hair", "green hair", "purple hair", "pink hair", "short hair",
    "long hair", "bob cut", "ponytail", "side ponytail", "twintails", "braided hair",
    "wavy hair", "curly hair", "messy hair", "windblown hair", "choppy bangs", "long sidelocks",
]
FACE = [
    "blue eyes", "green eyes", "brown eyes", "red eyes", "purple eyes", "golden eyes",
    "looking at viewer", "looking away", "profile", "parted lips", "closed-mouth smile",
    "wide smile", "serious expression", "focused expression", "surprised expression",
    "raised inner eyebrows", "half-lidded eyes", "round glasses", "freckles", "blush",
]
BODY = [
    "slender build", "athletic build", "soft build", "curvy figure", "broad shoulders",
    "narrow waist", "wide hips", "long legs", "thick thighs", "petite build", "tall stature",
]
CLOTHES = [
    "white turtleneck dress", "black tailored suit", "red pilot suit", "blue summer dress",
    "green field jacket", "oversized hoodie", "leather jacket", "school-style blazer",
    "pleated skirt", "wide-leg trousers", "cargo pants", "long coat", "raincoat", "lab coat",
    "knitted sweater", "sleeveless top", "detached sleeves", "armored bodysuit", "formal gown",
    "utility belt", "fingerless gloves", "lace collar", "puffed sleeves", "frilled cuffs",
    "embroidered fabric", "metallic trim", "transparent outer layer", "layered clothing",
]
POSES = [
    "standing", "sitting", "kneeling", "crouching", "walking forward", "running", "jumping",
    "leaning against a wall", "leaning against a tree", "turning toward the viewer",
    "torso twist", "arched back", "contrapposto", "one knee raised", "legs apart",
    "arms crossed", "one hand on hip", "hand near face", "open palm toward viewer",
    "reaching into the foreground", "holding an object with both hands", "looking over shoulder",
    "hair and clothing moving in the wind", "dynamic action pose", "relaxed seated pose",
]
CAMERA = [
    "eye-level view", "from above", "from below", "low angle", "high angle", "three-quarter view",
    "side view", "rear three-quarter view", "dutch angle", "extreme dutch angle", "fisheye perspective",
    "wide-angle lens", "telephoto compression", "close-up", "upper body portrait", "cowboy shot",
    "full body", "wide shot", "panoramic composition", "square composition", "vertical composition",
    "strong foreshortening", "controlled perspective", "one-point perspective", "two-point perspective",
]
COMPOSITION = [
    "face in the upper third", "subject centered", "subject offset to the right", "subject offset to the left",
    "diagonal composition", "triangular composition", "circular eye path", "clear focal hierarchy",
    "foreground hand as focal point", "large readable face", "clean silhouette", "layered depth",
    "foreground framing", "leading lines", "negative space", "symmetrical layout", "asymmetrical balance",
    "overlapping depth without covering faces", "small central gap between subjects",
    "clear separation between subjects", "dynamic crop",
]
ENVIRONMENTS = [
    "sunlit forest", "autumn woodland", "rainy city street", "neon alley", "school classroom",
    "modern apartment", "traditional room", "industrial workshop", "mechanical hangar", "laboratory",
    "rooftop at sunset", "night observatory", "beach", "rocky coast", "snowy mountain", "desert ruins",
    "flower field", "train platform", "subway station", "library", "cafe", "concert stage", "art studio",
    "space station interior", "fantasy castle", "stone temple", "greenhouse", "underwater tunnel",
]
LIGHTING = [
    "soft daylight", "dappled sunlight", "golden hour", "blue hour", "moonlight", "overcast light",
    "strong backlighting", "rim light", "sidelighting", "volumetric lighting", "god rays", "neon lighting",
    "warm lantern light", "cold fluorescent light", "dramatic shadow", "high-key lighting", "low-key lighting",
    "reflected light", "colored bounce light", "subsurface scattering", "lens flare", "bokeh",
]
COLORS = [
    "warm palette", "cool palette", "complementary orange and blue", "red and cyan accents",
    "pastel palette", "muted earth tones", "high saturation", "controlled saturation",
    "low saturation", "neutral palette", "monochrome",
    "black and white with one red accent", "gold and navy palette", "mint and pink palette",
]
STYLES = [
    "anime illustration", "clean cel shading", "soft painterly shading", "graphic poster style",
    "visual novel event cg", "retro game illustration", "y2k futurism", "vaporwave", "frutiger aero",
    "chromecore", "metalheart", "pop art", "acid graphics", "minimalist design", "editorial illustration",
    "high-detail digital painting", "flat color design", "glossy 3d-like highlights", "inked line art",
]
OBJECTS = [
    "glowing wrench", "transparent umbrella", "book", "smartphone", "camera", "helmet", "sword",
    "staff", "microphone", "guitar", "coffee cup", "flower bouquet", "lantern", "crystal", "map",
    "toolbox", "mechanical part", "floating interface panel", "holographic display", "paper airplane",
]
RELATIONS = [
    "standing behind a chair", "sitting beside a window", "walking toward the camera",
    "reaching past the camera", "one subject in front of the other", "two subjects facing each other",
    "two subjects back to back", "one subject looking at the other", "hands meeting at the center",
    "left subject wears white and right subject wears black",
    "foreground subject has short hair and background subject has long hair",
    "front-left subject points at the viewer while rear-right subject looks away",
    "object between the subject and the viewer", "foreground object partially occluding the lower frame",
]
MATERIALS = [
    "brushed metal", "polished chrome", "matte plastic", "glossy vinyl", "soft cotton", "rough linen",
    "wet fabric", "transparent glass", "frosted glass", "ceramic surface", "weathered wood", "stone texture",
]
RENDER = [
    "sharp eyes", "detailed hands", "fine fabric texture", "shallow depth of field", "deep depth of field",
    "motion blur in the background", "foreground blur", "atmospheric perspective", "clean linework",
    "soft bloom", "specular highlights", "reflective surfaces", "subtle film grain", "crisp edges",
]
TEXT_CONCEPTS = [
    "simple logo with one readable letter", "large readable sign", "small interface labels",
    "poster with geometric typography", "no readable text", "blank signboard", "minimal icon design",
]


def _pick(rng: random.Random, seq: list[str]) -> str:
    return seq[rng.randrange(len(seq))]


def _unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out



def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    idx = int(round((len(ordered) - 1) * max(0.0, min(1.0, float(q)))))
    return ordered[idx]


def analyze_bridge_calibration_prompts(prompts: Iterable[str]) -> dict[str, Any]:
    """Return tokenizer-free diagnostics for a calibration corpus.

    The report is intentionally cheap so it can be used in Jupyter before any
    model or tokenizer is loaded. It does not judge semantic quality; it flags
    structural properties that commonly make representation calibration less
    useful, such as duplicates or prompt-syntax control tokens.
    """

    lines = [str(item).strip() for item in prompts if str(item).strip()]
    unique = []
    seen: set[str] = set()
    duplicates = 0
    for line in lines:
        key = line.casefold()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
            unique.append(line)

    clause_counts = [len([p for p in re.split(r"[,;\n]+", line) if p.strip()]) for line in lines]
    char_lengths = [len(line) for line in lines]
    atomic = sum(c <= 1 for c in clause_counts)
    short = sum(2 <= c <= 8 for c in clause_counts)
    medium = sum(9 <= c <= 16 for c in clause_counts)
    long = sum(c >= 17 for c in clause_counts)
    weighted = sum(bool(re.search(r"[\(\[][^\n]*?:\s*[-+]?\d+(?:\.\d+)?[\)\]]", line)) for line in lines)
    break_lines = sum(bool(re.search(r"(?:^|[,;\s])BREAK(?:$|[,;\s])", line, flags=re.I)) for line in lines)
    and_lines = sum(" AND " in line for line in lines)
    non_ascii = sum(any(ord(ch) > 127 for ch in line) for line in lines)

    n = max(1, len(lines))
    warnings: list[str] = []
    if len(lines) < 256:
        warnings.append("fewer than 256 prompt lines: calibration is too small")
    elif len(lines) < 1024:
        warnings.append("fewer than 1024 prompt lines: usable for experiments, but 4096+ is recommended")
    if duplicates / n > 0.05:
        warnings.append("more than 5% duplicate lines: repeated prompts add little new alignment information")
    if (weighted + break_lines + and_lines) / n > 0.05:
        warnings.append("more than 5% of lines contain weighting/BREAK/AND syntax; prefer semantic text without frontend control syntax")
    if long == 0:
        warnings.append("no long prompts detected: add coherent long-context examples if long-prompt use is a goal")
    if atomic == 0:
        warnings.append("no atomic concepts detected: add single-concept anchors for local semantic directions")

    return {
        "prompt_lines": len(lines),
        "unique_lines": len(unique),
        "duplicate_lines": duplicates,
        "duplicate_ratio": duplicates / n,
        "atomic_lines": atomic,
        "short_lines": short,
        "medium_lines": medium,
        "long_lines": long,
        "char_length_p50": _percentile(char_lengths, 0.50),
        "char_length_p90": _percentile(char_lengths, 0.90),
        "char_length_p99": _percentile(char_lengths, 0.99),
        "clauses_p50": _percentile(clause_counts, 0.50),
        "clauses_p90": _percentile(clause_counts, 0.90),
        "weighted_syntax_lines": weighted,
        "break_syntax_lines": break_lines,
        "and_syntax_lines": and_lines,
        "non_ascii_lines": non_ascii,
        "warnings": warnings,
    }

def build_default_bridge_calibration_prompts(count: int = 4096, seed: int = 3571) -> list[str]:
    """Build a deterministic calibration corpus.

    The first section deliberately contains atomic/short concepts; the rest
    combines them into progressively richer prompts. Phrase expansion in the
    calibration script turns these lines into tens of thousands of paired
    anchors without needing a huge checked-in text asset.
    """
    if count < 256:
        raise ValueError("default bridge calibration corpus requires count >= 256")
    rng = random.Random(int(seed))
    prompts: list[str] = []

    # Atomic anchors ensure common visual directions are represented directly.
    atomic_banks = [
        SUBJECTS, HAIR, FACE, BODY, CLOTHES, POSES, CAMERA, COMPOSITION,
        ENVIRONMENTS, LIGHTING, COLORS, STYLES, OBJECTS, RELATIONS, MATERIALS,
        RENDER, TEXT_CONCEPTS,
    ]
    for bank in atomic_banks:
        prompts.extend(bank)

    # Short tag-style prompts.
    while len(prompts) < min(count, 1536):
        clauses = [
            _pick(rng, SUBJECTS), _pick(rng, HAIR), _pick(rng, FACE),
            _pick(rng, CLOTHES), _pick(rng, POSES), _pick(rng, CAMERA),
            _pick(rng, ENVIRONMENTS), _pick(rng, LIGHTING),
        ]
        if rng.random() < 0.55:
            clauses.append(_pick(rng, COMPOSITION))
        if rng.random() < 0.45:
            clauses.append(_pick(rng, STYLES))
        prompts.append(", ".join(_unique(clauses)))

    # Medium prompts with relations/materials.
    while len(prompts) < min(count, 3072):
        clauses = [
            _pick(rng, SUBJECTS), _pick(rng, HAIR), _pick(rng, FACE), _pick(rng, BODY),
            _pick(rng, CLOTHES), _pick(rng, MATERIALS), _pick(rng, POSES),
            _pick(rng, RELATIONS), _pick(rng, CAMERA), _pick(rng, COMPOSITION),
            _pick(rng, ENVIRONMENTS), _pick(rng, LIGHTING), _pick(rng, COLORS),
            _pick(rng, RENDER), _pick(rng, STYLES),
        ]
        prompts.append(", ".join(_unique(clauses)))

    # Long prompts exercise contextual drift and ensure calibration is not only
    # based on isolated tags. They intentionally retain one coherent composition.
    while len(prompts) < count:
        clauses = [
            _pick(rng, SUBJECTS), _pick(rng, HAIR), _pick(rng, FACE), _pick(rng, BODY),
            _pick(rng, CLOTHES), _pick(rng, CLOTHES), _pick(rng, MATERIALS),
            _pick(rng, POSES), _pick(rng, POSES), _pick(rng, RELATIONS),
            _pick(rng, OBJECTS), _pick(rng, CAMERA), _pick(rng, CAMERA),
            _pick(rng, COMPOSITION), _pick(rng, COMPOSITION), _pick(rng, ENVIRONMENTS),
            _pick(rng, LIGHTING), _pick(rng, LIGHTING), _pick(rng, COLORS),
            _pick(rng, RENDER), _pick(rng, RENDER), _pick(rng, STYLES),
        ]
        if rng.random() < 0.20:
            clauses.append(_pick(rng, TEXT_CONCEPTS))
        prompts.append(", ".join(_unique(clauses)))

    return _unique(prompts)[:count]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=3571)
    args = parser.parse_args()
    lines = build_default_bridge_calibration_prompts(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} prompts to {args.output}")
