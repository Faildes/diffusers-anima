"""Representation-space alignment profile for alternate Anima text encoders.

The v2 artifact is intentionally more than a bare matrix.  It is a self-
describing encoder-compatibility profile that can be used in two forms:

* ``bridge_profile``: small file containing only calibration tensors.  The
  source Qwen checkpoint is loaded separately.
* ``aligned_encoder``: the same bridge tensors plus ``encoder.*`` weights.  The
  single safetensors file can be supplied as ``encoder_path`` and the pipeline
  automatically attaches the embedded bridge.

This allows today's Qwen3.5-0.8B + bridge setup to evolve into an aligned text
encoder artifact without changing the downstream Anima conditioning API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file
import torch


_BRIDGE_FORMAT_V1 = "anima_text_encoder_bridge_v1"
_PROFILE_FORMAT_V2 = "anima_text_encoder_profile_v2"
_SUPPORTED_FORMATS = {_BRIDGE_FORMAT_V1, _PROFILE_FORMAT_V2}


def encoder_config_fingerprint(config: Any) -> str:
    """Return a stable architecture fingerprint without hashing model weights."""
    if hasattr(config, "to_dict"):
        data = dict(config.to_dict())
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        data = {
            key: getattr(config, key)
            for key in dir(config)
            if not key.startswith("_") and isinstance(getattr(config, key, None), (str, int, float, bool, type(None)))
        }
    # Keep architecture-defining fields and any Qwen3.5 layer-type schedule.
    keys = (
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "max_position_embeddings", "rope_theta", "layer_types", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
    )
    reduced = {key: data.get(key) for key in keys if key in data}
    payload = json.dumps(reduced, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_text_encoder_profile_metadata(path: str | Path) -> dict[str, str]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def is_text_encoder_profile_file(path: str | Path) -> bool:
    try:
        metadata = read_text_encoder_profile_metadata(path)
    except Exception:
        return False
    return metadata.get("format") == _PROFILE_FORMAT_V2


@dataclass
class AnimaTextEncoderBridge:
    rotation: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    variance_scale: torch.Tensor | None = None
    input_projection: torch.Tensor | None = None
    rms_scale: torch.Tensor | None = None
    center_strength: float = 1.0
    variance_strength: float = 0.0
    rms_strength: float = 0.0
    # v4 stability controls. These are deliberately runtime-only transforms
    # around the global alignment, so the source encoder weights remain intact.
    delta_clip_ratio: float = 0.0
    token_rms_strength: float = 0.0
    token_rms_min_ratio: float = 0.85
    token_rms_max_ratio: float = 1.10
    metadata: dict[str, str] = field(default_factory=dict)
    _cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor | None, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.rotation = self.rotation.detach().cpu().float().contiguous()
        self.source_mean = self.source_mean.detach().cpu().float().reshape(-1).contiguous()
        self.target_mean = self.target_mean.detach().cpu().float().reshape(-1).contiguous()
        if self.variance_scale is not None:
            self.variance_scale = self.variance_scale.detach().cpu().float().reshape(-1).contiguous()
        if self.input_projection is not None:
            self.input_projection = self.input_projection.detach().cpu().float().contiguous()
        if self.rms_scale is not None:
            self.rms_scale = self.rms_scale.detach().cpu().float().reshape(-1).contiguous()

        bridge_dim = int(self.source_mean.numel())
        if tuple(self.rotation.shape) != (bridge_dim, bridge_dim):
            raise ValueError(
                f"Bridge rotation must have shape ({bridge_dim}, {bridge_dim}), got {tuple(self.rotation.shape)}."
            )
        if int(self.target_mean.numel()) != bridge_dim:
            raise ValueError("Bridge source/target means must have the same bridge dimension.")
        if self.variance_scale is not None and int(self.variance_scale.numel()) != bridge_dim:
            raise ValueError("Bridge variance_scale dimension does not match the bridge dimension.")
        if self.rms_scale is not None and int(self.rms_scale.numel()) not in {1, bridge_dim}:
            raise ValueError("Bridge rms_scale must be scalar or match the bridge dimension.")
        if self.input_projection is not None:
            if self.input_projection.ndim != 2 or int(self.input_projection.shape[1]) != bridge_dim:
                raise ValueError(
                    "Bridge input_projection must have shape (source_hidden_size, bridge_hidden_size)."
                )
        self.center_strength = float(self.center_strength)
        self.variance_strength = float(self.variance_strength)
        self.rms_strength = float(self.rms_strength)
        self.delta_clip_ratio = max(0.0, float(self.delta_clip_ratio))
        self.token_rms_strength = max(0.0, min(1.0, float(self.token_rms_strength)))
        self.token_rms_min_ratio = max(1e-3, float(self.token_rms_min_ratio))
        self.token_rms_max_ratio = max(self.token_rms_min_ratio, float(self.token_rms_max_ratio))

    @property
    def bridge_hidden_size(self) -> int:
        return int(self.source_mean.numel())

    @property
    def hidden_size(self) -> int:
        # Backward-compatible alias for the Anima/reference-side width.
        return self.bridge_hidden_size

    @property
    def source_hidden_size(self) -> int:
        if self.input_projection is not None:
            return int(self.input_projection.shape[0])
        value = int(self.metadata.get("source_hidden_size", "0") or 0)
        return value or self.bridge_hidden_size

    @property
    def artifact_kind(self) -> str:
        return self.metadata.get("artifact_kind", "bridge_profile")

    @property
    def contains_encoder_weights(self) -> bool:
        return self.metadata.get("contains_encoder_weights", "false").lower() == "true"

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
    ) -> "AnimaTextEncoderBridge":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Anima text-encoder bridge/profile not found: {path}")
        tensors = load_file(str(path), device="cpu")
        metadata = read_text_encoder_profile_metadata(path)
        format_name = metadata.get("format")
        if format_name not in {None, *_SUPPORTED_FORMATS}:
            raise ValueError(
                f"Unsupported Anima text-encoder bridge/profile format {format_name!r}; "
                f"supported={sorted(_SUPPORTED_FORMATS)}."
            )

        if format_name == _PROFILE_FORMAT_V2:
            prefix = "bridge."
            def get_tensor(name: str):
                return tensors.get(prefix + name)
            rotation = get_tensor("rotation")
            source_mean = get_tensor("source_mean")
            target_mean = get_tensor("target_mean")
            variance_scale_tensor = get_tensor("variance_scale")
            input_projection = get_tensor("input_projection")
            rms_scale = get_tensor("rms_scale")
        else:
            rotation = tensors.get("rotation")
            source_mean = tensors.get("source_mean")
            target_mean = tensors.get("target_mean")
            variance_scale_tensor = tensors.get("variance_scale")
            input_projection = tensors.get("input_projection")
            rms_scale = tensors.get("rms_scale")

        missing = [
            name for name, value in (
                ("rotation", rotation), ("source_mean", source_mean), ("target_mean", target_mean)
            ) if value is None
        ]
        if missing:
            raise ValueError(f"Bridge/profile file is missing tensors: {', '.join(missing)}")

        def metadata_float(key: str, fallback: float) -> float:
            try:
                return float(metadata.get(key, fallback))
            except (TypeError, ValueError):
                return float(fallback)

        if center_strength is None:
            # v4 separates calibration-space optimum from image-runtime defaults.
            # Full centering is retained, but per-dimension variance matching is
            # opt-in because it was observed to amplify colour/style channels.
            center_strength = metadata_float("runtime_recommended_center_strength", 1.0)
        if variance_strength is None:
            variance_strength = metadata_float("runtime_recommended_variance_strength", 0.0)
        if rms_strength is None:
            rms_strength = metadata_float("runtime_recommended_rms_strength", 0.0)
        if delta_clip_ratio is None:
            delta_clip_ratio = metadata_float("recommended_delta_clip_ratio", 0.30)
        if token_rms_strength is None:
            token_rms_strength = metadata_float("recommended_token_rms_strength", 0.70)
        if token_rms_min_ratio is None:
            token_rms_min_ratio = metadata_float("recommended_token_rms_min_ratio", 0.85)
        if token_rms_max_ratio is None:
            token_rms_max_ratio = metadata_float("recommended_token_rms_max_ratio", 1.10)

        return cls(
            rotation=rotation,  # type: ignore[arg-type]
            source_mean=source_mean,  # type: ignore[arg-type]
            target_mean=target_mean,  # type: ignore[arg-type]
            variance_scale=variance_scale_tensor,
            input_projection=input_projection,
            rms_scale=rms_scale,
            center_strength=center_strength,
            variance_strength=variance_strength,
            rms_strength=rms_strength,
            metadata=metadata,
        )

    def validate_encoder(self, encoder: Any, *, strict_fingerprint: bool = True) -> None:
        config = getattr(encoder, "config", None)
        active_hidden = int(getattr(config, "hidden_size", 0) or 0)
        if active_hidden and active_hidden != self.source_hidden_size:
            raise ValueError(
                f"Bridge source hidden size {self.source_hidden_size} does not match active encoder {active_hidden}."
            )
        expected = self.metadata.get("source_architecture_fingerprint", "")
        if expected and config is not None:
            actual = encoder_config_fingerprint(config)
            if actual != expected:
                message = (
                    "Bridge source architecture fingerprint does not match the active encoder: "
                    f"bridge={expected[:12]}..., active={actual[:12]}...."
                )
                if strict_fingerprint:
                    raise ValueError(message)
                import warnings
                warnings.warn(message, stacklevel=2)

    def _runtime_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Alignment is numerically more stable in fp32. Cache per device while
        # keeping the artifact itself on CPU so model CPU offload does not mutate it.
        key = (str(device), torch.float32)
        cached = self._cache.get(key)
        if cached is None:
            projection = (
                self.input_projection.to(device=device, dtype=torch.float32)
                if self.input_projection is not None else None
            )
            rotation = self.rotation.to(device=device, dtype=torch.float32)
            source_mean = self.source_mean.to(device=device, dtype=torch.float32)
            target_mean = self.target_mean.to(device=device, dtype=torch.float32)
            variance_scale = (
                self.variance_scale.to(device=device, dtype=torch.float32)
                if self.variance_scale is not None else None
            )
            rms_scale = (
                self.rms_scale.to(device=device, dtype=torch.float32)
                if self.rms_scale is not None else None
            )
            cached = (projection, rotation, source_mean, target_mean, variance_scale, rms_scale)
            self._cache[key] = cached
        return cached  # type: ignore[return-value]

    def clear_runtime_cache(self) -> None:
        self._cache.clear()

    def _apply_delta_clip(self, reference: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Limit only the alignment displacement, not the source representation itself."""
        if self.delta_clip_ratio <= 0.0:
            return out
        delta = out - reference
        delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True).clamp_min(1e-6)
        ref_norm = torch.linalg.vector_norm(reference.float(), dim=-1, keepdim=True).clamp_min(1e-6)
        allowed = ref_norm * float(self.delta_clip_ratio)
        scale = torch.clamp(allowed / delta_norm, max=1.0).to(delta.dtype)
        return reference + delta * scale

    def _apply_token_rms_preservation(self, reference: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Partially preserve per-token source energy after rotation.

        The old bridge can match pooled cosine while still changing the norm of
        individual tokens enough to over-drive colour/style channels. This
        correction is local per token and deliberately bounded.
        """
        if self.token_rms_strength <= 0.0:
            return out
        ref_rms = reference.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        out_rms = out.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        ratio = (ref_rms / out_rms).clamp(
            min=float(self.token_rms_min_ratio), max=float(self.token_rms_max_ratio)
        )
        scale = 1.0 + (ratio - 1.0) * float(self.token_rms_strength)
        return out * scale.to(out.dtype)

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
    ) -> "AnimaTextEncoderBridge":
        if center_strength is not None:
            self.center_strength = float(center_strength)
        if variance_strength is not None:
            self.variance_strength = float(variance_strength)
        if rms_strength is not None:
            self.rms_strength = float(rms_strength)
        if delta_clip_ratio is not None:
            self.delta_clip_ratio = max(0.0, float(delta_clip_ratio))
        if token_rms_strength is not None:
            self.token_rms_strength = max(0.0, min(1.0, float(token_rms_strength)))
        if token_rms_min_ratio is not None:
            self.token_rms_min_ratio = max(1e-3, float(token_rms_min_ratio))
        if token_rms_max_ratio is not None:
            self.token_rms_max_ratio = max(self.token_rms_min_ratio, float(token_rms_max_ratio))
        self.clear_runtime_cache()
        return self

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("AnimaTextEncoderBridge expects (batch, sequence, hidden) hidden states.")
        if int(hidden_states.shape[-1]) != self.source_hidden_size:
            raise ValueError(
                f"Bridge source hidden size is {self.source_hidden_size}, got {int(hidden_states.shape[-1])}."
            )
        output_dtype = hidden_states.dtype
        projection, rotation, source_mean, target_mean, variance_scale, rms_scale = self._runtime_tensors(
            hidden_states.device, hidden_states.dtype
        )
        x = hidden_states.float()
        if projection is not None:
            x = torch.matmul(x, projection)

        # Orthogonal rotation is the least destructive common coordinate change
        # and therefore acts as the stability reference for all v4 guards.
        rotated_uncentered = torch.matmul(x, rotation)
        format_name = self.metadata.get("format", _BRIDGE_FORMAT_V1)
        if format_name == _PROFILE_FORMAT_V2:
            aligned_centered = torch.matmul(x - source_mean, rotation)
            if variance_scale is not None and self.variance_strength != 0.0:
                scale = 1.0 + (variance_scale - 1.0) * self.variance_strength
                aligned_centered = aligned_centered * scale
            fully_centered = aligned_centered + target_mean
            out = rotated_uncentered + (fully_centered - rotated_uncentered) * self.center_strength
        else:
            aligned_centered = torch.matmul(x - source_mean, rotation)
            if variance_scale is not None and self.variance_strength != 0.0:
                scale = 1.0 + (variance_scale - 1.0) * self.variance_strength
                aligned_centered = aligned_centered * scale
            center = source_mean + (target_mean - source_mean) * self.center_strength
            out = aligned_centered + center

        # Keep pooled-space alignment but stop individual tokens from being
        # displaced far enough to collapse identity/binding or over-drive style.
        out = self._apply_delta_clip(rotated_uncentered, out)

        if rms_scale is not None and self.rms_strength != 0.0:
            scale = 1.0 + (rms_scale - 1.0) * self.rms_strength
            out = out * scale
        out = self._apply_token_rms_preservation(rotated_uncentered, out)
        return out.to(dtype=output_dtype)

    def apply_source_preserving(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map source features into Anima coordinates without covariance matching.

        This view uses the same global projection/rotation and target centering as
        the compatibility bridge, but deliberately skips variance/RMS correction.
        For the common 1024->1024 case the transform is an orthogonal rotation
        plus translation, so relative source geometry is preserved.  Final Anima
        encoders use this only for optional semantic-expansion memory; the primary
        token stream remains fully reference-aligned.
        """
        if hidden_states.ndim != 3:
            raise ValueError("AnimaTextEncoderBridge expects (batch, sequence, hidden) hidden states.")
        if int(hidden_states.shape[-1]) != self.source_hidden_size:
            raise ValueError(
                f"Bridge source hidden size is {self.source_hidden_size}, got {int(hidden_states.shape[-1])}."
            )
        output_dtype = hidden_states.dtype
        projection, rotation, source_mean, target_mean, _variance_scale, _rms_scale = self._runtime_tensors(
            hidden_states.device, hidden_states.dtype
        )
        x = hidden_states.float()
        if projection is not None:
            x = torch.matmul(x, projection)
        out = torch.matmul(x - source_mean, rotation) + target_mean
        return out.to(dtype=output_dtype)

    def describe(self) -> dict[str, Any]:
        return {
            "format": self.metadata.get("format", _BRIDGE_FORMAT_V1),
            "artifact_kind": self.artifact_kind,
            "source_family": self.metadata.get("source_family", "unknown"),
            "target_family": self.metadata.get("target_family", "unknown"),
            "source_model": self.metadata.get("source_model", "unknown"),
            "reference_model": self.metadata.get("reference_model", "unknown"),
            "samples": int(self.metadata.get("samples", "0") or 0),
            "corpus_lines": int(self.metadata.get("corpus_lines", "0") or 0),
            "source_hidden_size": self.source_hidden_size,
            "bridge_hidden_size": self.bridge_hidden_size,
            "center_strength": self.center_strength,
            "variance_strength": self.variance_strength,
            "rms_strength": self.rms_strength,
            "has_variance_scale": self.variance_scale is not None,
            "has_input_projection": self.input_projection is not None,
            "delta_clip_ratio": self.delta_clip_ratio,
            "token_rms_strength": self.token_rms_strength,
            "token_rms_min_ratio": self.token_rms_min_ratio,
            "token_rms_max_ratio": self.token_rms_max_ratio,
            "contains_encoder_weights": self.contains_encoder_weights,
            "validation_cosine_before": self.metadata.get("validation_cosine_before"),
            "validation_cosine_after": self.metadata.get("validation_cosine_after"),
            "calibration_corpus_sha256": self.metadata.get("calibration_corpus_sha256"),
        }


__all__ = [
    "AnimaTextEncoderBridge",
    "encoder_config_fingerprint",
    "is_text_encoder_profile_file",
    "read_text_encoder_profile_metadata",
]
