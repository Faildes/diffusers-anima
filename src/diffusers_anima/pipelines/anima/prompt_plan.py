"""Structured prompt metadata shared by sd_embed and diffusers-anima.

A prompt plan keeps the user's full visible text and attaches weighting metadata
to character spans.  It deliberately does not pre-build multiple conditioning
tensors for AND/BREAK groups; all groups become one Qwen semantic memory and
one Anima adapter pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AnimaPromptSpan:
    start: int
    end: int
    qwen_factor: float = 1.0
    t5_factor: float = 1.0
    group: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "AnimaPromptSpan":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                start=int(value["start"]),
                end=int(value["end"]),
                qwen_factor=float(value.get("qwen_factor", 1.0)),
                t5_factor=float(value.get("t5_factor", 1.0)),
                group=int(value.get("group", 0)),
            )
        raise TypeError(f"Unsupported prompt span type: {type(value).__name__}")


@dataclass(frozen=True)
class AnimaPromptPlan:
    text: str
    spans: tuple[AnimaPromptSpan, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "AnimaPromptPlan":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, Mapping):
            spans_value = value.get("spans", ())
            return cls(
                text=str(value.get("text", "")),
                spans=tuple(AnimaPromptSpan.from_value(item) for item in spans_value),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError(f"Unsupported prompt plan type: {type(value).__name__}")

    def validated(self) -> "AnimaPromptPlan":
        n = len(self.text)
        for span in self.spans:
            if span.start < 0 or span.end < span.start or span.end > n:
                raise ValueError(
                    f"Prompt span [{span.start}, {span.end}) is outside text length {n}."
                )
        return self


def coerce_prompt_plans(values: Sequence[Any]) -> list[AnimaPromptPlan]:
    return [AnimaPromptPlan.from_value(value).validated() for value in values]


__all__ = ["AnimaPromptPlan", "AnimaPromptSpan", "coerce_prompt_plans"]
