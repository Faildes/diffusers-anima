#!/usr/bin/env python3
"""Corpus preparation for Anima-native text-encoder training.

The input can be the user's existing one-prompt-per-line calibration file.  This
module preserves the original lines and adds deterministic minimal-pair /
binding groups so the trainer can explicitly identify concepts for which the
0.8B source separates meanings better than the 0.6B Anima reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class NativePromptGroup:
    texts: tuple[str, ...]
    category: str = "base"
    focus: str = ""

    @property
    def paired(self) -> bool:
        return len(self.texts) == 2


# Pairs are intentionally directional only for generation; the trainer evaluates
# the representation delta symmetrically.  Longest phrases should appear first.
_MINIMAL_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("in front of", "behind", "spatial"),
    ("left of", "right of", "spatial"),
    ("from above", "from below", "camera"),
    ("high angle", "low angle", "camera"),
    ("sitting", "standing", "pose"),
    ("kneeling", "standing", "pose"),
    ("walking", "running", "pose"),
    ("red hair", "blue hair", "appearance"),
    ("black hair", "blonde hair", "appearance"),
    ("green eyes", "blue eyes", "appearance"),
    ("white dress", "black dress", "clothing"),
    ("white jacket", "black jacket", "clothing"),
    ("1girl", "2girls", "count"),
    ("1boy", "2boys", "count"),
    ("one woman", "two women", "count"),
    ("one man", "two men", "count"),
    ("close-up", "full body", "framing"),
    ("close up", "full body", "framing"),
    ("warm palette", "cool palette", "color"),
    ("high saturation", "low saturation", "color"),
)



# High-level sampling buckets deliberately do not mirror corpus frequency.  The
# calibration file can grow arbitrarily large without allowing common solo /
# colour-heavy prompt families to dominate the optimiser budget.
_SAMPLING_BUCKET_WEIGHTS: dict[str, float] = {
    "general": 0.20,
    "composition": 0.12,
    "binding": 0.22,
    "count": 0.12,
    "multilingual": 0.10,
    "color": 0.08,
    "style": 0.04,
    # v5 explicitly reserves fixed-budget optimiser pressure for prompt
    # obedience instead of hoping it emerges from raw corpus frequency.
    "instruction": 0.07,
    "equivalence": 0.05,
}

_MULTI_SUBJECT_RE = re.compile(
    r"(?i)\b(?:two women|two men|woman and man|duo|trio|group portrait|anthro couple|anthro trio|"
    r"[2-9](?:girls?|boys?|women|men|people|persons?|characters?)|"
    r"exactly\s+(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:people|persons?|characters?|girls?|boys?|women|men))\b"
)
_COUNT_RE = re.compile(
    r"(?i)\b(?:[2-9]\d?\s*(?:girls?|boys?|women|men)|exactly\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:people|persons?|characters?|girls?|boys?|women|men))\b"
)
_COLOR_RE = re.compile(
    r"(?i)\b(?:high saturation|controlled saturation|low saturation|pastel palette|muted earth tones|"
    r"monochrome|black and white|warm palette|cool palette|red and cyan accents|gold and navy palette|"
    r"mint and pink palette|complementary orange and blue|neon lighting|colored bounce light)\b"
)
_STYLE_RE = re.compile(
    r"(?i)\b(?:anime illustration|clean cel shading|soft painterly shading|graphic poster style|"
    r"visual novel event cg|retro game illustration|y2k futurism|vaporwave|frutiger aero|chromecore|"
    r"metalheart|pop art|acid graphics|editorial illustration|digital painting|flat color design)\b"
)
_COMPOSITION_RE = re.compile(
    r"(?i)\b(?:from above|from below|low angle|high angle|dutch angle|fisheye|wide-angle|telephoto|"
    r"close-up|full body|wide shot|panoramic|foreshortening|perspective|composition|foreground|background|"
    r"standing|sitting|kneeling|walking|running|jumping|crouching|contrapposto|looking over shoulder)\b"
)

def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in str(text))

def prompt_sampling_bucket(text: str) -> str:
    """Return a coarse, deterministic sampling bucket for one prompt.

    This is intentionally lexical rather than embedding-based: corpus analysis
    must stay cheap even for hundreds of thousands of calibration rows.
    """
    raw = str(text or "")
    if _MULTI_SUBJECT_RE.search(raw):
        return "binding"
    if _COUNT_RE.search(raw):
        return "count"
    if _contains_non_ascii(raw):
        return "multilingual"
    if _COLOR_RE.search(raw):
        return "color"
    if _STYLE_RE.search(raw):
        return "style"
    if _COMPOSITION_RE.search(raw):
        return "composition"
    return "general"

def group_sampling_bucket(group: NativePromptGroup) -> str:
    category = str(group.category)
    if category in {"binding", "attribute_swap"}:
        return "binding"
    if category in {"count", "count_binding"}:
        return "count"
    if category == "multilingual_binding":
        return "multilingual"
    if category in {"directive_contrast", "instruction_fidelity"}:
        return "instruction"
    if category == "tag_nl_equivalence":
        return "equivalence"
    if category in {"color", "color_control"}:
        return "color"
    if category in {"spatial", "camera", "pose", "framing"}:
        return "composition"
    if category.startswith("base:"):
        bucket = category.split(":", 1)[1]
        return bucket if bucket in _SAMPLING_BUCKET_WEIGHTS else "general"
    if category in _SAMPLING_BUCKET_WEIGHTS:
        return category
    return prompt_sampling_bucket(group.texts[0]) if group.texts else "general"

def default_sampling_bucket_weights() -> dict[str, float]:
    return dict(_SAMPLING_BUCKET_WEIGHTS)

def split_validation_lines(
    lines: Sequence[str],
    *,
    validation_size: int = 192,
    seed: int = 3571,
) -> tuple[list[str], list[str]]:
    """Deterministically hold out a small stratified validation subset.

    The holdout has a fixed *count*, not a fixed fraction, so growing a 40k
    corpus to 400k does not make validation or training more expensive.
    """
    wanted = max(0, min(int(validation_size), max(0, len(lines) - 1)))
    if wanted <= 0:
        return list(lines), []
    by_bucket: dict[str, list[tuple[str, str]]] = {}
    salt = f"anima-native-val-v3:{int(seed)}:"
    for text in lines:
        bucket = prompt_sampling_bucket(text)
        digest = hashlib.sha256((salt + text.casefold()).encode("utf-8")).hexdigest()
        by_bucket.setdefault(bucket, []).append((digest, text))
    for values in by_bucket.values():
        values.sort(key=lambda item: item[0])

    weights = default_sampling_bucket_weights()
    selected: list[str] = []
    # Allocate approximately by the target sampler distribution, never by raw
    # corpus frequency.  Empty buckets simply donate capacity to the fill pass.
    for bucket, weight in weights.items():
        take = max(1, round(wanted * weight)) if by_bucket.get(bucket) else 0
        selected.extend(text for _digest, text in by_bucket.get(bucket, [])[:take])
    # Stable fill/trim by salted hash.
    selected_map = {x.casefold(): x for x in selected}
    if len(selected_map) < wanted:
        remainder: list[tuple[str, str]] = []
        for values in by_bucket.values():
            for digest, text in values:
                if text.casefold() not in selected_map:
                    remainder.append((digest, text))
        remainder.sort(key=lambda item: item[0])
        for _digest, text in remainder:
            selected_map.setdefault(text.casefold(), text)
            if len(selected_map) >= wanted:
                break
    selected = list(selected_map.values())
    if len(selected) > wanted:
        selected.sort(key=lambda text: hashlib.sha256((salt + text.casefold()).encode("utf-8")).hexdigest())
        selected = selected[:wanted]
    held = {x.casefold() for x in selected}
    train = [text for text in lines if text.casefold() not in held]
    return train, selected

# Fixed Anima-compatibility anchors.  These never scale with corpus size and
# intentionally include neutral/explicit-colour triplets, multi-person binding,
# Danbooru tags, natural language and multiple languages.
ANIMA_COMPAT_ANCHOR_PROMPTS: tuple[str, ...] = (
    "1girl, solo, red hair, blue eyes, standing, school classroom, soft daylight, anime illustration",
    "1girl, solo, red hair, blue eyes, standing, school classroom, soft daylight, anime illustration, high saturation",
    "1girl, solo, red hair, blue eyes, standing, school classroom, soft daylight, anime illustration, controlled saturation",
    "1girl, solo, black hair, green eyes, sitting, library, overcast light, clean cel shading",
    "1boy, solo, brown hair, full body, train platform, blue hour, clean linework",
    "1woman, white hair, formal gown, rooftop at sunset, rim light, painterly illustration",
    "1man, black tailored suit, industrial workshop, sidelighting, detailed hands",
    "furry, anthro fox, orange fur, standing, forest, soft daylight",
    "humanoid robot, metallic body, mechanical hangar, reflected light",
    "elf woman, long hair, green eyes, fantasy castle, moonlight",
    "two women; subject 1: far left, red hair, black dress; subject 2: far right, blue hair, white jacket",
    "two men; subject 1: far left, black hair, grey suit; subject 2: far right, blonde hair, blue coat",
    "exactly three characters, no additional people; subject 1: far left, red hair; subject 2: center, blue hair; subject 3: far right, green hair",
    "exactly four characters, no additional people; subject 1: far left, black hair, red coat; subject 2: center-left, blonde hair, blue jacket; subject 3: center-right, white hair, green hoodie; subject 4: far right, purple hair, black dress",
    "woman and man, facing each other, small central gap, cafe, warm lantern light",
    "trio, overlapping depth without covering faces, wide shot, school classroom",
    "group portrait, clear focal hierarchy, controlled perspective, soft daylight",
    "1girl, sitting, from above, close-up, cool palette",
    "1girl, sitting, from below, full body, warm palette",
    "1girl, dynamic action pose, strong foreshortening, dutch angle",
    "1girl, monochrome, clean linework, simple background",
    "1girl, pastel palette, soft painterly shading, flower field",
    "1girl, muted earth tones, autumn woodland, overcast light",
    "1girl, neon alley, neon lighting, reflective surfaces",
    "1girl, vaporwave, square composition, pink and cyan lighting",
    "1girl, frutiger aero, clean glossy interface, bright daylight",
    "1girl, chromecore, polished chrome, controlled saturation",
    "1girl, metalheart, brushed metal, dramatic shadow",
    "正確に2人の人物、subject 1: 左、赤い髪、黒い服; subject 2: 右、青い髪、白い服",
    "恰好3个人物，没有其他人物; subject 1: left, red hair; subject 2: center, blue hair; subject 3: right, green hair",
    "정확히 2명의 인물, 다른 인물 없음; subject 1: left, black hair; subject 2: right, blonde hair",
    "exactamente 2 personajes, ninguna persona adicional; subject 1: left, red hair; subject 2: right, blue hair",
)

def build_color_control_groups(*, count: int = 128, seed: int = 76543) -> list[NativePromptGroup]:
    """Create explicit saturation-control pairs without increasing step budget."""
    rng = random.Random(int(seed))
    subjects = ["1girl", "1boy", "1woman", "1man", "android girl", "elf woman", "anthro fox"]
    scenes = ["school classroom", "library", "rooftop at sunset", "rainy city street", "sunlit forest", "mechanical hangar"]
    styles = ["anime illustration", "clean cel shading", "soft painterly shading", "high-detail digital painting"]
    variants = ["high saturation", "controlled saturation", "muted earth tones", "monochrome", "pastel palette"]
    groups: list[NativePromptGroup] = []
    for i in range(max(0, int(count))):
        base = f"{rng.choice(subjects)}, standing, {rng.choice(scenes)}, soft daylight, {rng.choice(styles)}"
        variant = variants[i % len(variants)]
        groups.append(NativePromptGroup((base, f"{base}, {variant}"), category="color_control", focus=variant))
    return groups

def read_prompt_lines(path: str | Path) -> list[str]:
    path = Path(path)
    lines: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
    if not lines:
        raise ValueError(f"No usable prompt lines found in {path}")
    return lines


def corpus_sha256(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(str(line).strip().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _replace_first_case_insensitive(text: str, a: str, b: str) -> str | None:
    pattern = re.compile(re.escape(a), flags=re.IGNORECASE)
    if pattern.search(text):
        return pattern.sub(b, text, count=1)
    pattern = re.compile(re.escape(b), flags=re.IGNORECASE)
    if pattern.search(text):
        return pattern.sub(a, text, count=1)
    return None


def make_minimal_pair(text: str) -> tuple[str, str] | None:
    for a, b, category in _MINIMAL_REPLACEMENTS:
        mutated = _replace_first_case_insensitive(text, a, b)
        if mutated is not None and mutated.casefold() != text.casefold():
            return mutated, category
    return None


def build_binding_groups(*, count: int = 256, seed: int = 99173) -> list[NativePromptGroup]:
    """Generate hard multi-subject ownership/count/multilingual pairs.

    The total number of generated groups remains ``count`` so stronger binding
    coverage does not make training longer.  Every synthetic multi-person row
    uses explicit ``subject N:`` clauses; the trainer and sd_embed runtime use
    the same structural convention to construct subject-slot IDs.
    """
    rng = random.Random(int(seed))
    hairs = ["red", "blue", "green", "black", "blonde", "white", "silver", "purple"]
    eyes = ["blue", "green", "brown", "red", "golden", "purple", "amber", "grey"]
    clothes = [
        "white jacket", "black jacket", "blue coat", "red dress", "green hoodie",
        "grey suit", "yellow cardigan", "purple sweater",
    ]
    poses = ["standing", "sitting", "looking at the viewer", "looking away", "raising one hand", "walking"]
    positions = ["far left", "left", "center-left", "center", "center-right", "right", "far right"]
    number_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

    def make_subjects(n: int) -> list[dict[str, str]]:
        # Distinct values by construction make ownership swaps unambiguous.
        chosen_hair = rng.sample(hairs, n)
        chosen_eye = rng.sample(eyes, n)
        chosen_clothes = rng.sample(clothes, n)
        chosen_pose = [rng.choice(poses) for _ in range(n)]
        chosen_pos = positions[:n] if n <= len(positions) else [f"position {i+1}" for i in range(n)]
        return [
            {"hair": chosen_hair[i], "eyes": chosen_eye[i], "clothes": chosen_clothes[i],
             "pose": chosen_pose[i], "position": chosen_pos[i]}
            for i in range(n)
        ]

    def render(subjects: list[dict[str, str]], *, language: str = "en") -> str:
        n = len(subjects)
        if language == "ja":
            prefix = f"正確に{n}人の人物、他の人物はいない"
            clauses = [
                f"subject {i+1}: {x['position']}、{x['hair']} hair、{x['eyes']} eyes、{x['clothes']}、{x['pose']}"
                for i, x in enumerate(subjects)
            ]
        elif language == "zh":
            prefix = f"恰好{n}个人物，没有其他人物"
            clauses = [
                f"subject {i+1}: {x['position']}, {x['hair']} hair, {x['eyes']} eyes, {x['clothes']}, {x['pose']}"
                for i, x in enumerate(subjects)
            ]
        elif language == "ko":
            prefix = f"정확히 {n}명의 인물, 다른 인물 없음"
            clauses = [
                f"subject {i+1}: {x['position']}, {x['hair']} hair, {x['eyes']} eyes, {x['clothes']}, {x['pose']}"
                for i, x in enumerate(subjects)
            ]
        elif language == "es":
            prefix = f"exactamente {n} personajes, ninguna persona adicional"
            clauses = [
                f"subject {i+1}: {x['position']}, pelo {x['hair']}, ojos {x['eyes']}, {x['clothes']}, {x['pose']}"
                for i, x in enumerate(subjects)
            ]
        else:
            prefix = f"exactly {number_words[n]} characters, no additional people"
            clauses = [
                f"subject {i+1}: {x['position']}, {x['hair']} hair, {x['eyes']} eyes, {x['clothes']}, {x['pose']}"
                for i, x in enumerate(subjects)
            ]
        return prefix + "; " + "; ".join(clauses) + "; preserve each subject's own attributes"

    groups: list[NativePromptGroup] = []
    languages = ("ja", "zh", "ko", "es")
    for i in range(max(0, int(count))):
        mode = i % 5
        n = rng.randint(2, 6)
        subjects = make_subjects(n)
        base = render(subjects)

        if mode in (0, 1):
            # One-attribute mutation: retain the old useful minimal-binding task,
            # now across 2-6 people rather than only two women.
            changed = [dict(x) for x in subjects]
            target = rng.randrange(n)
            alternatives = [x for x in hairs if x not in {s["hair"] for s in subjects}]
            changed[target]["hair"] = rng.choice(alternatives or hairs)
            groups.append(NativePromptGroup((base, render(changed)), category="binding", focus=f"subject_{target+1}"))
        elif mode == 2:
            # Hard negative with exactly the same bag of words: only ownership
            # changes.  This directly trains against clothes/hair swapping.
            swapped = [dict(x) for x in subjects]
            a, b = rng.sample(range(n), 2)
            swapped[a]["clothes"], swapped[b]["clothes"] = swapped[b]["clothes"], swapped[a]["clothes"]
            swapped[a]["hair"], swapped[b]["hair"] = swapped[b]["hair"], swapped[a]["hair"]
            groups.append(NativePromptGroup((base, render(swapped)), category="attribute_swap", focus=f"subject_{a+1}_{b+1}"))
        elif mode == 3:
            # Count hard negative.  The shared subjects remain unchanged and a
            # single clearly distinct subject is added/removed.
            n2 = n + 1 if n < 6 else n - 1
            other = make_subjects(n2)
            if n2 > n:
                other[:n] = [dict(x) for x in subjects]
            else:
                other = [dict(x) for x in subjects[:n2]]
            groups.append(NativePromptGroup((base, render(other)), category="count_binding", focus=f"count_{n}_{n2}"))
        else:
            # Multilingual understanding must remain sourced from Qwen3.5, not
            # collapsed into the 0.6B reference.  Use the same ownership change
            # in a non-English instruction while keeping visual tags explicit.
            lang = languages[(i // 5) % len(languages)]
            changed = [dict(x) for x in subjects]
            target = rng.randrange(n)
            alternatives = [x for x in clothes if x not in {s["clothes"] for s in subjects}]
            changed[target]["clothes"] = rng.choice(alternatives or clothes)
            groups.append(NativePromptGroup((render(subjects, language=lang), render(changed, language=lang)), category="multilingual_binding", focus=f"{lang}_subject_{target+1}"))
    return groups



def build_instruction_fidelity_groups(*, count: int = 256, seed: int = 73321) -> list[NativePromptGroup]:
    """Create hard prompt-obedience contrasts with minimal lexical change.

    These pairs target the failure modes that matter for image prompting: exact
    count, ownership, pose, spatial relation, camera direction and explicit
    inclusion/exclusion.  They are generated independently of corpus size and
    therefore do not increase the v3/v4 fixed optimiser-step budget.
    """
    rng = random.Random(int(seed))
    colors = ["red", "blue", "green", "black", "white", "purple", "blonde"]
    clothes = ["black dress", "white jacket", "green hoodie", "blue coat", "red uniform"]
    poses = ["standing", "sitting", "kneeling", "walking", "looking at viewer"]
    groups: list[NativePromptGroup] = []
    for i in range(max(0, int(count))):
        mode = i % 6
        a, b = rng.sample(colors, 2)
        c1, c2 = rng.sample(clothes, 2)
        p1, p2 = rng.sample(poses, 2)
        if mode == 0:
            n = rng.randint(2, 7)
            m = n + 1 if n < 8 else n - 1
            left = f"exactly {n} girls, no additional people, group portrait, each face visible"
            right = f"exactly {m} girls, no additional people, group portrait, each face visible"
            focus = f"count_{n}_{m}"
        elif mode == 1:
            left = f"two girls; subject 1: left, {a} hair, {c1}; subject 2: right, {b} hair, {c2}"
            right = f"two girls; subject 1: left, {a} hair, {c2}; subject 2: right, {b} hair, {c1}"
            focus = "clothing_ownership"
        elif mode == 2:
            left = f"two girls; subject 1: left, {a} hair, {p1}; subject 2: right, {b} hair, {p2}"
            right = f"two girls; subject 1: left, {a} hair, {p2}; subject 2: right, {b} hair, {p1}"
            focus = "pose_ownership"
        elif mode == 3:
            left = f"{a}-haired girl left of {b}-haired girl, both full body"
            right = f"{a}-haired girl right of {b}-haired girl, both full body"
            focus = "spatial_direction"
        elif mode == 4:
            left = f"1girl, {a} hair, {c1}, low angle, full body"
            right = f"1girl, {a} hair, {c1}, high angle, full body"
            focus = "camera_direction"
        else:
            left = f"1girl, {a} hair, {c1}, wearing glasses, {p1}"
            right = f"1girl, {a} hair, {c1}, no glasses, {p1}"
            focus = "explicit_presence"
        groups.append(NativePromptGroup((left, right), category="directive_contrast", focus=focus))
    return groups


def build_tag_nl_equivalence_groups(*, count: int = 160, seed: int = 73357) -> list[NativePromptGroup]:
    """Pair Danbooru-like tags with natural-language prompts of identical intent."""
    rng = random.Random(int(seed))
    hairs = ["red", "blue", "black", "blonde", "white", "green"]
    eyes = ["blue", "green", "brown", "amber", "purple"]
    clothes = ["black dress", "white jacket", "green hoodie", "blue coat"]
    scenes = ["classroom", "library", "city street", "forest", "train platform"]
    poses = ["standing", "sitting", "walking", "kneeling"]
    groups: list[NativePromptGroup] = []
    for i in range(max(0, int(count))):
        if i % 3 == 0:
            hair, eye = rng.choice(hairs), rng.choice(eyes)
            cloth, scene, pose = rng.choice(clothes), rng.choice(scenes), rng.choice(poses)
            tag = f"1girl, solo, {hair} hair, {eye} eyes, {cloth}, {pose}, {scene}, full body"
            natural = (
                f"A single girl with {hair} hair and {eye} eyes is {pose} in a {scene}. "
                f"She wears a {cloth}. Show her full body and no other person."
            )
        else:
            h1, h2 = rng.sample(hairs, 2)
            c1, c2 = rng.sample(clothes, 2)
            tag = (
                f"2girls; subject 1: left, {h1} hair, {c1}; "
                f"subject 2: right, {h2} hair, {c2}; no other people"
            )
            natural = (
                f"There are exactly two girls and nobody else. The girl on the left has {h1} hair "
                f"and wears a {c1}. The girl on the right has {h2} hair and wears a {c2}."
            )
        groups.append(NativePromptGroup((tag, natural), category="tag_nl_equivalence", focus="same_visual_intent"))
    return groups


def build_training_groups(
    lines: Sequence[str],
    *,
    pair_fraction: float = 0.35,
    binding_pairs: int = 256,
    seed: int = 3571,
) -> list[NativePromptGroup]:
    rng = random.Random(int(seed))

    candidates: list[NativePromptGroup] = []
    for text in lines:
        pair = make_minimal_pair(text)
        if pair is None:
            continue
        mutated, category = pair
        candidates.append(NativePromptGroup((text, mutated), category=category, focus="minimal_pair"))
    rng.shuffle(candidates)
    desired = min(len(candidates), max(0, round(len(lines) * max(0.0, float(pair_fraction)))))
    paired = candidates[:desired]
    # A selected minimal pair already contains its original corpus row.  Do not
    # run that identical text through the 0.8B backbone again as a singleton.
    # This removes duplicate compute without dropping a single unique prompt.
    paired_originals = {group.texts[0].casefold() for group in paired}
    base_groups = [
        NativePromptGroup((text,), category=f"base:{prompt_sampling_bucket(text)}")
        for text in lines
        if text.casefold() not in paired_originals
    ]
    bindings = build_binding_groups(count=binding_pairs, seed=seed + 17)
    color_controls = build_color_control_groups(count=max(64, min(256, binding_pairs // 4)), seed=seed + 29)
    instruction_groups = build_instruction_fidelity_groups(
        count=max(192, int(binding_pairs)), seed=seed + 41
    )
    equivalence_groups = build_tag_nl_equivalence_groups(
        count=max(128, int(binding_pairs) // 2), seed=seed + 53
    )
    groups = base_groups + paired + bindings + color_controls + instruction_groups + equivalence_groups
    rng.shuffle(groups)
    return groups


def preview_training_groups(groups: Sequence[NativePromptGroup]) -> dict[str, object]:
    categories: dict[str, int] = {}
    rows = 0
    pairs = 0
    for group in groups:
        categories[group.category] = categories.get(group.category, 0) + 1
        rows += len(group.texts)
        pairs += int(group.paired)
    return {
        "groups": len(groups),
        "rows": rows,
        "paired_groups": pairs,
        "categories": dict(sorted(categories.items())),
    }


__all__ = [
    "NativePromptGroup",
    "ANIMA_COMPAT_ANCHOR_PROMPTS",
    "build_binding_groups",
    "build_color_control_groups",
    "build_instruction_fidelity_groups",
    "build_tag_nl_equivalence_groups",
    "build_training_groups",
    "default_sampling_bucket_weights",
    "group_sampling_bucket",
    "prompt_sampling_bucket",
    "split_validation_lines",
    "corpus_sha256",
    "make_minimal_pair",
    "preview_training_groups",
    "read_prompt_lines",
]
