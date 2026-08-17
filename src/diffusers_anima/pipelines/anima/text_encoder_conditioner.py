"""Final Anima text-encoder conditioning head.

The v3 *final encoder* format keeps the complete source encoder weights and a
single global Anima compatibility head in one safetensors file.  Unlike a bare
bridge, it may also append a small number of source-geometry-preserving semantic
summary slots.  The primary token memory is never replaced: it stays fully
aligned to the Qwen3-0.6B representation expected by Anima's LLM adapter.

This design intentionally prioritises capability retention.  Qwen3.5-0.8B stays
unchanged as the semantic backbone, while the Anima head supplies compatibility
and exposes a conservative extra view of source semantics to cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch

from .text_encoder_bridge import AnimaTextEncoderBridge, read_text_encoder_profile_metadata

_FINAL_ENCODER_FORMAT_V3 = "anima_text_encoder_v3"


def is_final_text_encoder_file(path: str | Path) -> bool:
    try:
        md = read_text_encoder_profile_metadata(path)
    except Exception:
        return False
    return md.get("format") == _FINAL_ENCODER_FORMAT_V3 and md.get("artifact_kind") == "final_encoder"


def _metadata_float(md: dict[str, str], key: str, default: float) -> float:
    try:
        return float(md.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _metadata_int(md: dict[str, str], key: str, default: int) -> int:
    try:
        return int(md.get(key, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class AnimaTextEncoderConditioner:
    """One global compatibility head plus optional semantic-expansion memory."""

    alignment: AnimaTextEncoderBridge
    semantic_expansion_strength: float = 0.25
    semantic_expansion_max_tokens: int = 16
    semantic_expansion_chunk_size: int = 64
    semantic_expansion_min_source_tokens: int = 16
    semantic_expansion_residual_clip: float = 0.30
    # v4: expansion slots must not become a second global soup. When PromptPlan
    # group ids are available, build slots inside each group, and suppress
    # residual summaries whose token directions disagree.
    semantic_expansion_group_aware: bool = True
    semantic_expansion_coherence_power: float = 1.0
    semantic_expansion_min_coherence: float = 0.15
    # v5: a short prompt should not cross the adapter's native 512-source
    # boundary merely because semantic slots were appended. Long prompts are
    # handled by the transformer-side windowed router instead.
    semantic_expansion_preserve_native_window: bool = True
    semantic_expansion_native_window: int = 512
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.semantic_expansion_strength = max(0.0, float(self.semantic_expansion_strength))
        self.semantic_expansion_max_tokens = max(0, int(self.semantic_expansion_max_tokens))
        self.semantic_expansion_chunk_size = max(1, int(self.semantic_expansion_chunk_size))
        self.semantic_expansion_min_source_tokens = max(1, int(self.semantic_expansion_min_source_tokens))
        self.semantic_expansion_residual_clip = max(0.0, float(self.semantic_expansion_residual_clip))
        self.semantic_expansion_group_aware = bool(self.semantic_expansion_group_aware)
        self.semantic_expansion_coherence_power = max(0.0, float(self.semantic_expansion_coherence_power))
        self.semantic_expansion_min_coherence = max(0.0, min(1.0, float(self.semantic_expansion_min_coherence)))
        self.semantic_expansion_preserve_native_window = bool(self.semantic_expansion_preserve_native_window)
        self.semantic_expansion_native_window = max(1, int(self.semantic_expansion_native_window))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        center_strength: float | None = None,
        variance_strength: float | None = None,
        rms_strength: float | None = None,
        delta_clip_ratio: float | None = None,
        token_rms_strength: float | None = None,
        token_rms_min_ratio: float | None = None,
        token_rms_max_ratio: float | None = None,
        semantic_expansion_strength: float | None = None,
        semantic_expansion_max_tokens: int | None = None,
        semantic_expansion_chunk_size: int | None = None,
        semantic_expansion_min_source_tokens: int | None = None,
        semantic_expansion_residual_clip: float | None = None,
        semantic_expansion_group_aware: bool | None = None,
        semantic_expansion_coherence_power: float | None = None,
        semantic_expansion_min_coherence: float | None = None,
        semantic_expansion_preserve_native_window: bool | None = None,
        semantic_expansion_native_window: int | None = None,
    ) -> "AnimaTextEncoderConditioner":
        path = Path(path)
        md = read_text_encoder_profile_metadata(path)
        if md.get("format") != _FINAL_ENCODER_FORMAT_V3 or md.get("artifact_kind") != "final_encoder":
            raise ValueError(f"Not an Anima final text encoder v3 artifact: {path}")
        tensors = load_file(str(path), device="cpu")
        required = ("head.rotation", "head.source_mean", "head.target_mean")
        missing = [key for key in required if key not in tensors]
        if missing:
            raise ValueError(f"Final encoder is missing conditioning tensors: {', '.join(missing)}")

        # Reuse the proven v2 alignment implementation. The runtime metadata is
        # intentionally marked as v2 because the mathematical head has the same
        # transform semantics; v3 changes packaging/memory behaviour, not the
        # primary alignment equation.
        bridge_md = dict(md)
        bridge_md["format"] = "anima_text_encoder_profile_v2"
        bridge_md["artifact_kind"] = "final_encoder"
        bridge_md["contains_encoder_weights"] = "true"
        def md_float(key: str, default: float) -> float:
            return _metadata_float(md, key, default)
        def md_bool(key: str, default: bool) -> bool:
            value = str(md.get(key, "true" if default else "false")).strip().lower()
            return value in {"1", "true", "yes", "on"}

        # v4 intentionally does not reuse the calibration-only variance optimum
        # as the final-encoder default. Pooled cosine can prefer full variance
        # matching even when token-level energy and multi-subject binding degrade.
        final_center = md_float("final_center_strength", 1.0)
        final_variance = md_float("final_variance_strength", 0.0)
        final_rms = md_float("final_rms_strength", 0.0)
        alignment = AnimaTextEncoderBridge(
            rotation=tensors["head.rotation"],
            source_mean=tensors["head.source_mean"],
            target_mean=tensors["head.target_mean"],
            variance_scale=tensors.get("head.variance_scale"),
            input_projection=tensors.get("head.input_projection"),
            rms_scale=tensors.get("head.rms_scale"),
            center_strength=(final_center if center_strength is None else float(center_strength)),
            variance_strength=(final_variance if variance_strength is None else float(variance_strength)),
            rms_strength=(final_rms if rms_strength is None else float(rms_strength)),
            delta_clip_ratio=(
                md_float("final_delta_clip_ratio", 0.30)
                if delta_clip_ratio is None else float(delta_clip_ratio)
            ),
            token_rms_strength=(
                md_float("final_token_rms_strength", 0.70)
                if token_rms_strength is None else float(token_rms_strength)
            ),
            token_rms_min_ratio=(
                md_float("final_token_rms_min_ratio", 0.85)
                if token_rms_min_ratio is None else float(token_rms_min_ratio)
            ),
            token_rms_max_ratio=(
                md_float("final_token_rms_max_ratio", 1.10)
                if token_rms_max_ratio is None else float(token_rms_max_ratio)
            ),
            metadata=bridge_md,
        )
        return cls(
            alignment=alignment,
            semantic_expansion_strength=(
                md_float("semantic_expansion_strength", 0.25)
                if semantic_expansion_strength is None else float(semantic_expansion_strength)
            ),
            semantic_expansion_max_tokens=(
                _metadata_int(md, "semantic_expansion_max_tokens", 16)
                if semantic_expansion_max_tokens is None else int(semantic_expansion_max_tokens)
            ),
            semantic_expansion_chunk_size=(
                _metadata_int(md, "semantic_expansion_chunk_size", 64)
                if semantic_expansion_chunk_size is None else int(semantic_expansion_chunk_size)
            ),
            semantic_expansion_min_source_tokens=(
                _metadata_int(md, "semantic_expansion_min_source_tokens", 16)
                if semantic_expansion_min_source_tokens is None else int(semantic_expansion_min_source_tokens)
            ),
            semantic_expansion_residual_clip=(
                md_float("semantic_expansion_residual_clip", 0.30)
                if semantic_expansion_residual_clip is None else float(semantic_expansion_residual_clip)
            ),
            semantic_expansion_group_aware=(
                md_bool("semantic_expansion_group_aware", True)
                if semantic_expansion_group_aware is None else bool(semantic_expansion_group_aware)
            ),
            semantic_expansion_coherence_power=(
                md_float("semantic_expansion_coherence_power", 1.0)
                if semantic_expansion_coherence_power is None else float(semantic_expansion_coherence_power)
            ),
            semantic_expansion_min_coherence=(
                md_float("semantic_expansion_min_coherence", 0.15)
                if semantic_expansion_min_coherence is None else float(semantic_expansion_min_coherence)
            ),
            semantic_expansion_preserve_native_window=(
                md_bool("semantic_expansion_preserve_native_window", True)
                if semantic_expansion_preserve_native_window is None else bool(semantic_expansion_preserve_native_window)
            ),
            semantic_expansion_native_window=(
                _metadata_int(md, "semantic_expansion_native_window", 512)
                if semantic_expansion_native_window is None else int(semantic_expansion_native_window)
            ),
            metadata=md,
        )

    def set_runtime_stability(
        self,
        *,
        center_strength: float | None = None,
        variance_strength: float | None = None,
        rms_strength: float | None = None,
        delta_clip_ratio: float | None = None,
        token_rms_strength: float | None = None,
        token_rms_min_ratio: float | None = None,
        token_rms_max_ratio: float | None = None,
        semantic_expansion_strength: float | None = None,
        semantic_expansion_residual_clip: float | None = None,
        semantic_expansion_group_aware: bool | None = None,
        semantic_expansion_coherence_power: float | None = None,
        semantic_expansion_min_coherence: float | None = None,
        semantic_expansion_preserve_native_window: bool | None = None,
        semantic_expansion_native_window: int | None = None,
    ) -> "AnimaTextEncoderConditioner":
        self.alignment.set_runtime_stability(
            center_strength=center_strength,
            variance_strength=variance_strength,
            rms_strength=rms_strength,
            delta_clip_ratio=delta_clip_ratio,
            token_rms_strength=token_rms_strength,
            token_rms_min_ratio=token_rms_min_ratio,
            token_rms_max_ratio=token_rms_max_ratio,
        )
        if semantic_expansion_strength is not None:
            self.semantic_expansion_strength = max(0.0, float(semantic_expansion_strength))
        if semantic_expansion_residual_clip is not None:
            self.semantic_expansion_residual_clip = max(0.0, float(semantic_expansion_residual_clip))
        if semantic_expansion_group_aware is not None:
            self.semantic_expansion_group_aware = bool(semantic_expansion_group_aware)
        if semantic_expansion_coherence_power is not None:
            self.semantic_expansion_coherence_power = max(0.0, float(semantic_expansion_coherence_power))
        if semantic_expansion_min_coherence is not None:
            self.semantic_expansion_min_coherence = max(0.0, min(1.0, float(semantic_expansion_min_coherence)))
        if semantic_expansion_preserve_native_window is not None:
            self.semantic_expansion_preserve_native_window = bool(semantic_expansion_preserve_native_window)
        if semantic_expansion_native_window is not None:
            self.semantic_expansion_native_window = max(1, int(semantic_expansion_native_window))
        return self

    def validate_encoder(self, encoder: Any, *, strict_fingerprint: bool = True) -> None:
        self.alignment.validate_encoder(encoder, strict_fingerprint=strict_fingerprint)

    def clear_runtime_cache(self) -> None:
        self.alignment.clear_runtime_cache()

    def align(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.alignment.apply(hidden_states)

    def _clip_residual(self, base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        clip = float(self.semantic_expansion_residual_clip)
        if clip <= 0.0:
            return residual
        base_norm = base.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        residual_norm = residual.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp((base_norm * clip) / residual_norm, max=1.0).to(residual.dtype)
        return residual * scale

    @staticmethod
    def _contiguous_segments(valid: torch.Tensor, group_ids: torch.Tensor | None) -> list[torch.Tensor]:
        if valid.numel() == 0:
            return []
        if group_ids is None:
            return [valid]
        groups = group_ids[valid].to(dtype=torch.long)
        segments: list[torch.Tensor] = []
        start = 0
        for i in range(1, int(valid.numel())):
            if int(groups[i].item()) != int(groups[i - 1].item()):
                segments.append(valid[start:i])
                start = i
        segments.append(valid[start:])
        return [seg for seg in segments if seg.numel() > 0]

    def _coherence_scale(self, residual_tokens: torch.Tensor) -> float:
        if residual_tokens.ndim != 2 or residual_tokens.shape[0] <= 1:
            return 1.0
        rt = residual_tokens.float()
        mean_vec = rt.mean(dim=0)
        mean_norm = torch.linalg.vector_norm(mean_vec)
        individual = torch.linalg.vector_norm(rt, dim=-1).mean().clamp_min(1e-6)
        coherence = float(torch.clamp(mean_norm / individual, 0.0, 1.0).item())
        if coherence < float(self.semantic_expansion_min_coherence):
            return 0.0
        return coherence ** float(self.semantic_expansion_coherence_power)

    def build_memory(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return stable aligned tokens plus binding-preserving semantic slots.

        v4 keeps v3's capability-retention idea, but expansion is no longer a
        blind equal-token summary. PromptPlan groups (BREAK/AND/semicolon groups)
        are kept separate when available, and incoherent residual directions are
        attenuated instead of being averaged into a cross-character mixture.
        """
        primary = self.align(hidden_states)
        strength = float(self.semantic_expansion_strength)
        if strength <= 0.0 or self.semantic_expansion_max_tokens <= 0:
            return primary, attention_mask
        source_view = self.alignment.apply_source_preserving(hidden_states)

        bsz, seq_len, dim = primary.shape
        slot_budget = int(self.semantic_expansion_max_tokens)
        if self.semantic_expansion_preserve_native_window and seq_len < int(self.semantic_expansion_native_window):
            slot_budget = min(slot_budget, int(self.semantic_expansion_native_window) - int(seq_len))
        if slot_budget <= 0:
            return primary, attention_mask
        if attention_mask is None:
            mask = torch.ones((bsz, seq_len), dtype=torch.long, device=primary.device)
        else:
            mask = attention_mask.to(device=primary.device)
        if group_ids is not None:
            if group_ids.shape[:2] != mask.shape[:2]:
                raise ValueError("group_ids must match the original source token mask shape")
            groups = group_ids.to(device=primary.device, dtype=torch.long)
        else:
            groups = None

        per_sample: list[list[torch.Tensor]] = []
        max_slots = 0
        for b in range(bsz):
            valid = torch.nonzero(mask[b] > 0, as_tuple=False).flatten()
            n = int(valid.numel())
            slots: list[torch.Tensor] = []
            if n >= self.semantic_expansion_min_source_tokens:
                sample_groups = groups[b] if (groups is not None and self.semantic_expansion_group_aware) else None
                segments = self._contiguous_segments(valid, sample_groups)
                for segment in segments:
                    if len(slots) >= slot_budget:
                        break
                    seg_n = int(segment.numel())
                    requested = max(1, math.ceil(seg_n / self.semantic_expansion_chunk_size))
                    requested = min(requested, slot_budget - len(slots), seg_n)
                    for slot_idx in range(requested):
                        lo = (slot_idx * seg_n) // requested
                        hi = ((slot_idx + 1) * seg_n) // requested
                        idx = segment[lo:hi]
                        if idx.numel() == 0:
                            continue
                        base_tokens = primary[b, idx]
                        residual_tokens = source_view[b, idx] - base_tokens
                        coherence_scale = self._coherence_scale(residual_tokens)
                        if coherence_scale <= 0.0:
                            continue
                        base = base_tokens.mean(dim=0)
                        residual = residual_tokens.mean(dim=0)
                        residual = self._clip_residual(base.unsqueeze(0), residual.unsqueeze(0))[0]
                        slots.append(base + residual * (strength * coherence_scale))
            per_sample.append(slots)
            max_slots = max(max_slots, len(slots))
        if max_slots == 0:
            return primary, attention_mask

        extra = torch.zeros((bsz, max_slots, dim), dtype=primary.dtype, device=primary.device)
        extra_mask = torch.zeros((bsz, max_slots), dtype=mask.dtype, device=mask.device)
        for b, slots in enumerate(per_sample):
            if not slots:
                continue
            stacked = torch.stack(slots, dim=0)
            extra[b, : stacked.shape[0]] = stacked
            extra_mask[b, : stacked.shape[0]] = 1
        return torch.cat([primary, extra], dim=1), torch.cat([mask, extra_mask], dim=1)

    def describe(self) -> dict[str, Any]:
        return {
            "format": self.metadata.get("format", _FINAL_ENCODER_FORMAT_V3),
            "artifact_kind": self.metadata.get("artifact_kind", "final_encoder"),
            "conditioning_mode": self.metadata.get("conditioning_mode", "aligned_plus_semantic_slots"),
            "source_family": self.metadata.get("source_family", "unknown"),
            "target_family": self.metadata.get("target_family", "unknown"),
            "anima_ready": True,
            "center_strength": self.alignment.center_strength,
            "variance_strength": self.alignment.variance_strength,
            "semantic_expansion_strength": self.semantic_expansion_strength,
            "semantic_expansion_max_tokens": self.semantic_expansion_max_tokens,
            "semantic_expansion_chunk_size": self.semantic_expansion_chunk_size,
            "semantic_expansion_min_source_tokens": self.semantic_expansion_min_source_tokens,
            "semantic_expansion_residual_clip": self.semantic_expansion_residual_clip,
            "semantic_expansion_group_aware": self.semantic_expansion_group_aware,
            "semantic_expansion_coherence_power": self.semantic_expansion_coherence_power,
            "semantic_expansion_min_coherence": self.semantic_expansion_min_coherence,
            "semantic_expansion_preserve_native_window": self.semantic_expansion_preserve_native_window,
            "semantic_expansion_native_window": self.semantic_expansion_native_window,
            "delta_clip_ratio": self.alignment.delta_clip_ratio,
            "token_rms_strength": self.alignment.token_rms_strength,
            "token_rms_min_ratio": self.alignment.token_rms_min_ratio,
            "token_rms_max_ratio": self.alignment.token_rms_max_ratio,
            "validation_cosine_before": self.metadata.get("validation_cosine_before"),
            "validation_cosine_after": self.metadata.get("validation_cosine_after"),
        }


__all__ = [
    "AnimaTextEncoderConditioner",
    "is_final_text_encoder_file",
]
