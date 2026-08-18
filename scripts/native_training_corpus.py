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
    """Generate subject-binding pairs where only one subject changes."""
    rng = random.Random(int(seed))
    colors = ["red", "blue", "green", "black", "blonde", "white", "silver", "purple"]
    eyes = ["blue", "green", "brown", "red", "golden", "purple"]
    clothes = ["white jacket", "black jacket", "blue coat", "red dress", "green hoodie", "grey suit"]
    poses = ["standing", "sitting", "looking at the viewer", "looking away", "raising one hand"]
    groups: list[NativePromptGroup] = []
    for i in range(max(0, int(count))):
        left_hair, changed_hair, right_hair = rng.sample(colors, 3)
        left_eye, right_eye = rng.sample(eyes, 2)
        left_cloth, right_cloth = rng.sample(clothes, 2)
        left_pose, right_pose = rng.sample(poses, 2)
        base = (
            f"two women, left woman has {left_hair} hair and {left_eye} eyes, {left_cloth}, {left_pose}; "
            f"right woman has {right_hair} hair and {right_eye} eyes, {right_cloth}, {right_pose}, "
            "clear separation between subjects"
        )
        changed = (
            f"two women, left woman has {changed_hair} hair and {left_eye} eyes, {left_cloth}, {left_pose}; "
            f"right woman has {right_hair} hair and {right_eye} eyes, {right_cloth}, {right_pose}, "
            "clear separation between subjects"
        )
        groups.append(NativePromptGroup((base, changed), category="binding", focus="left_subject"))
    return groups


def build_training_groups(
    lines: Sequence[str],
    *,
    pair_fraction: float = 0.35,
    binding_pairs: int = 256,
    seed: int = 3571,
) -> list[NativePromptGroup]:
    rng = random.Random(int(seed))
    base_groups = [NativePromptGroup((text,), category="base") for text in lines]

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
    bindings = build_binding_groups(count=binding_pairs, seed=seed + 17)
    groups = base_groups + paired + bindings
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
    "build_binding_groups",
    "build_training_groups",
    "corpus_sha256",
    "make_minimal_pair",
    "preview_training_groups",
    "read_prompt_lines",
]
