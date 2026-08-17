#!/usr/bin/env python3
"""Package a calibrated v2 bridge into a performance-first v3 final encoder.

This step is intentionally non-destructive: it does not rewrite or distil the
Qwen3.5 backbone.  The complete source encoder is retained and the calibrated
Anima compatibility transform becomes an embedded conditioning head.  Optional
semantic-expansion slots expose a clipped source-geometry-preserving view as
*additional* adapter memory, so compatibility remains the primary signal.

The output is one safetensors file and can be passed directly as encoder_path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def read_text_encoder_profile_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _strip_model_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("model.") for key in state_dict):
        return {key[6:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _extract_qwen35_text_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in ("model.language_model.", "language_model.", "model."):
        extracted = {
            key[len(prefix):]: value for key, value in state_dict.items()
            if key.startswith(prefix) and not key.startswith(prefix + "visual.")
        }
        if "embed_tokens.weight" in extracted and any(key.startswith("layers.") for key in extracted):
            return extracted
    if "embed_tokens.weight" in state_dict and any(key.startswith("layers.") for key in state_dict):
        return dict(state_dict)
    raise RuntimeError("Could not find the Qwen3.5 text backbone in source_model.")


@dataclass
class FinalEncoderConfig:
    bridge_profile: str | Path
    output: str | Path
    source_model: str | Path | None = None
    semantic_expansion_strength: float = 0.25
    semantic_expansion_max_tokens: int = 16
    semantic_expansion_chunk_size: int = 64
    semantic_expansion_min_source_tokens: int = 16
    semantic_expansion_residual_clip: float = 0.30
    semantic_expansion_group_aware: bool = True
    semantic_expansion_coherence_power: float = 1.0
    semantic_expansion_min_coherence: float = 0.15
    # v4 final-runtime stability defaults. Calibration metrics are preserved
    # separately; these values are chosen for token-level image stability.
    final_center_strength: float = 1.0
    final_variance_strength: float = 0.0
    final_rms_strength: float = 0.0
    final_delta_clip_ratio: float = 0.30
    final_token_rms_strength: float = 0.70
    final_token_rms_min_ratio: float = 0.85
    final_token_rms_max_ratio: float = 1.10

    def normalized(self) -> "FinalEncoderConfig":
        return FinalEncoderConfig(
            bridge_profile=str(self.bridge_profile),
            output=str(self.output),
            source_model=None if self.source_model is None else str(self.source_model),
            semantic_expansion_strength=float(self.semantic_expansion_strength),
            semantic_expansion_max_tokens=int(self.semantic_expansion_max_tokens),
            semantic_expansion_chunk_size=int(self.semantic_expansion_chunk_size),
            semantic_expansion_min_source_tokens=int(self.semantic_expansion_min_source_tokens),
            semantic_expansion_residual_clip=float(self.semantic_expansion_residual_clip),
            semantic_expansion_group_aware=bool(self.semantic_expansion_group_aware),
            semantic_expansion_coherence_power=float(self.semantic_expansion_coherence_power),
            semantic_expansion_min_coherence=float(self.semantic_expansion_min_coherence),
            final_center_strength=float(self.final_center_strength),
            final_variance_strength=float(self.final_variance_strength),
            final_rms_strength=float(self.final_rms_strength),
            final_delta_clip_ratio=float(self.final_delta_clip_ratio),
            final_token_rms_strength=float(self.final_token_rms_strength),
            final_token_rms_min_ratio=float(self.final_token_rms_min_ratio),
            final_token_rms_max_ratio=float(self.final_token_rms_max_ratio),
        )


@dataclass(frozen=True)
class FinalEncoderResult:
    output: Path
    source_family: str
    encoder_tensors: int
    head_tensors: int
    metadata: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["output"] = str(self.output)
        out["metadata"] = dict(self.metadata)
        return out

    def summary(self) -> str:
        return (
            f"final_encoder: {self.output}\n"
            f"source={self.source_family}; encoder_tensors={self.encoder_tensors}; "
            f"head_tensors={self.head_tensors}; expansion={self.metadata.get('semantic_expansion_strength')} "
            f"x <= {self.metadata.get('semantic_expansion_max_tokens')} slots"
        )


def _extract_source_weights(
    profile_tensors: dict[str, Any],
    metadata: dict[str, str],
    source_model: str | Path | None,
) -> dict[str, Any]:
    bundled = {
        key[len("encoder."):]: value
        for key, value in profile_tensors.items()
        if key.startswith("encoder.")
    }
    if bundled:
        return bundled
    if source_model is None:
        raise ValueError(
            "The v2 profile does not contain encoder.* weights. Pass source_model=... / --source-model "
            "or first create an aligned_encoder profile with --bundle-source-weights."
        )
    raw = load_file(str(source_model), device="cpu")
    family = metadata.get("source_family", "qwen3.5")
    if family == "qwen3.5":
        return _extract_qwen35_text_state_dict(raw)
    if family == "qwen3":
        return _strip_model_prefix(raw)
    raise ValueError(f"Unsupported source_family for final packaging: {family!r}")


def finalize_anima_text_encoder(config: FinalEncoderConfig) -> FinalEncoderResult:
    config = config.normalized()
    profile_path = Path(config.bridge_profile)
    md = read_text_encoder_profile_metadata(profile_path)
    if md.get("format") != "anima_text_encoder_profile_v2":
        raise ValueError("bridge_profile must be an anima_text_encoder_profile_v2 artifact")
    profile_tensors = load_file(str(profile_path), device="cpu")
    required = ("bridge.rotation", "bridge.source_mean", "bridge.target_mean")
    missing = [key for key in required if key not in profile_tensors]
    if missing:
        raise ValueError(f"Profile is missing required tensors: {', '.join(missing)}")

    source = _extract_source_weights(profile_tensors, md, config.source_model)
    tensors: dict[str, Any] = {f"encoder.{key}": value.contiguous() for key, value in source.items()}
    head_map = {
        "bridge.rotation": "head.rotation",
        "bridge.source_mean": "head.source_mean",
        "bridge.target_mean": "head.target_mean",
        "bridge.variance_scale": "head.variance_scale",
        "bridge.rms_scale": "head.rms_scale",
        "bridge.input_projection": "head.input_projection",
    }
    head_count = 0
    for old, new in head_map.items():
        if old in profile_tensors:
            tensors[new] = profile_tensors[old].contiguous()
            head_count += 1

    metadata = dict(md)
    metadata.update({
        "format": "anima_text_encoder_v3",
        "artifact_kind": "final_encoder",
        "contains_encoder_weights": "true",
        "anima_ready": "true",
        "conditioning_head": "global_alignment_v1",
        "conditioning_mode": "aligned_plus_semantic_slots",
        "semantic_expansion_mode": "clipped_source_residual_slots",
        "semantic_expansion_strength": f"{max(0.0, config.semantic_expansion_strength):.8g}",
        "semantic_expansion_max_tokens": str(max(0, config.semantic_expansion_max_tokens)),
        "semantic_expansion_chunk_size": str(max(1, config.semantic_expansion_chunk_size)),
        "semantic_expansion_min_source_tokens": str(max(1, config.semantic_expansion_min_source_tokens)),
        "semantic_expansion_residual_clip": f"{max(0.0, config.semantic_expansion_residual_clip):.8g}",
        "semantic_expansion_group_aware": "true" if config.semantic_expansion_group_aware else "false",
        "semantic_expansion_coherence_power": f"{max(0.0, config.semantic_expansion_coherence_power):.8g}",
        "semantic_expansion_min_coherence": f"{max(0.0, min(1.0, config.semantic_expansion_min_coherence)):.8g}",
        "stability_profile": "binding_preserving_v4",
        "final_center_strength": f"{config.final_center_strength:.8g}",
        "final_variance_strength": f"{config.final_variance_strength:.8g}",
        "final_rms_strength": f"{config.final_rms_strength:.8g}",
        "final_delta_clip_ratio": f"{max(0.0, config.final_delta_clip_ratio):.8g}",
        "final_token_rms_strength": f"{max(0.0, min(1.0, config.final_token_rms_strength)):.8g}",
        "final_token_rms_min_ratio": f"{max(1e-3, config.final_token_rms_min_ratio):.8g}",
        "final_token_rms_max_ratio": f"{max(config.final_token_rms_min_ratio, config.final_token_rms_max_ratio):.8g}",
        "calibration_recommended_center_strength": md.get("recommended_center_strength", "unknown"),
        "calibration_recommended_variance_strength": md.get("recommended_variance_strength", "unknown"),
        "finalization_policy": "preserve_source_backbone_add_anima_head_binding_stability_v4",
        "parent_profile_format": md.get("format", "unknown"),
        "parent_artifact_kind": md.get("artifact_kind", "unknown"),
    })
    # Keep metadata strings compact and valid for safetensors.
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={str(k): str(v) for k, v in metadata.items()})
    return FinalEncoderResult(
        output=output,
        source_family=metadata.get("source_family", "unknown"),
        encoder_tensors=len(source),
        head_tensors=head_count,
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--bridge-profile", required=True)
    p.add_argument("--source-model", default=None,
                   help="raw source encoder; optional when profile already bundles encoder.*")
    p.add_argument("--output", required=True)
    p.add_argument("--semantic-expansion-strength", type=float, default=0.25)
    p.add_argument("--semantic-expansion-max-tokens", type=int, default=16)
    p.add_argument("--semantic-expansion-chunk-size", type=int, default=64)
    p.add_argument("--semantic-expansion-min-source-tokens", type=int, default=16)
    p.add_argument("--semantic-expansion-residual-clip", type=float, default=0.30)
    p.add_argument("--semantic-expansion-group-aware", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--semantic-expansion-coherence-power", type=float, default=1.0)
    p.add_argument("--semantic-expansion-min-coherence", type=float, default=0.15)
    p.add_argument("--final-center-strength", type=float, default=1.0)
    p.add_argument("--final-variance-strength", type=float, default=0.0)
    p.add_argument("--final-rms-strength", type=float, default=0.0)
    p.add_argument("--final-delta-clip-ratio", type=float, default=0.30)
    p.add_argument("--final-token-rms-strength", type=float, default=0.70)
    p.add_argument("--final-token-rms-min-ratio", type=float, default=0.85)
    p.add_argument("--final-token-rms-max-ratio", type=float, default=1.10)
    return p


def main() -> None:
    a = _parser().parse_args()
    result = finalize_anima_text_encoder(FinalEncoderConfig(
        bridge_profile=a.bridge_profile,
        source_model=a.source_model,
        output=a.output,
        semantic_expansion_strength=a.semantic_expansion_strength,
        semantic_expansion_max_tokens=a.semantic_expansion_max_tokens,
        semantic_expansion_chunk_size=a.semantic_expansion_chunk_size,
        semantic_expansion_min_source_tokens=a.semantic_expansion_min_source_tokens,
        semantic_expansion_residual_clip=a.semantic_expansion_residual_clip,
        semantic_expansion_group_aware=a.semantic_expansion_group_aware,
        semantic_expansion_coherence_power=a.semantic_expansion_coherence_power,
        semantic_expansion_min_coherence=a.semantic_expansion_min_coherence,
        final_center_strength=a.final_center_strength,
        final_variance_strength=a.final_variance_strength,
        final_rms_strength=a.final_rms_strength,
        final_delta_clip_ratio=a.final_delta_clip_ratio,
        final_token_rms_strength=a.final_token_rms_strength,
        final_token_rms_min_ratio=a.final_token_rms_min_ratio,
        final_token_rms_max_ratio=a.final_token_rms_max_ratio,
    ))
    print(result.summary())


if __name__ == "__main__":
    main()
