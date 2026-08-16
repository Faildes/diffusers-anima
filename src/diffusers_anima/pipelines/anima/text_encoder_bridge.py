"""Representation-space alignment bridge for alternate Anima text encoders.

Anima's LLM adapter was trained against Qwen3-0.6B hidden states.  A text
encoder with the same hidden width (for example Qwen3.5-0.8B) is shape
compatible but not necessarily representation compatible.  This module keeps
that concern separate from prompt length handling: the source encoder may read
long prompts, then this bridge maps its hidden states into the reference
Qwen3-0.6B space before the existing Anima adapter sees them.

The bridge is calibration-only: no gradients or trainable runtime weights are
required.  ``scripts/calibrate_text_encoder_bridge.py`` produces the safetensors
artifact consumed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch


_BRIDGE_FORMAT = "anima_text_encoder_bridge_v1"


@dataclass
class AnimaTextEncoderBridge:
    rotation: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    variance_scale: torch.Tensor | None = None
    center_strength: float = 0.5
    variance_strength: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    _cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.rotation = self.rotation.detach().cpu().float().contiguous()
        self.source_mean = self.source_mean.detach().cpu().float().reshape(-1).contiguous()
        self.target_mean = self.target_mean.detach().cpu().float().reshape(-1).contiguous()
        if self.variance_scale is not None:
            self.variance_scale = self.variance_scale.detach().cpu().float().reshape(-1).contiguous()
        dim = int(self.source_mean.numel())
        if tuple(self.rotation.shape) != (dim, dim):
            raise ValueError(
                f"Bridge rotation must have shape ({dim}, {dim}), got {tuple(self.rotation.shape)}."
            )
        if int(self.target_mean.numel()) != dim:
            raise ValueError("Bridge source/target means must have the same dimension.")
        if self.variance_scale is not None and int(self.variance_scale.numel()) != dim:
            raise ValueError("Bridge variance_scale dimension does not match the hidden size.")
        self.center_strength = float(self.center_strength)
        self.variance_strength = float(self.variance_strength)

    @property
    def hidden_size(self) -> int:
        return int(self.source_mean.numel())

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        center_strength: float = 0.5,
        variance_strength: float = 0.0,
    ) -> "AnimaTextEncoderBridge":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Anima text-encoder bridge not found: {path}")
        tensors = load_file(str(path), device="cpu")
        metadata: dict[str, str] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        format_name = metadata.get("format")
        if format_name not in {None, _BRIDGE_FORMAT}:
            raise ValueError(
                f"Unsupported Anima text-encoder bridge format {format_name!r}; expected {_BRIDGE_FORMAT!r}."
            )
        required = {"rotation", "source_mean", "target_mean"}
        missing = sorted(required - set(tensors))
        if missing:
            raise ValueError(f"Bridge file is missing tensors: {', '.join(missing)}")
        return cls(
            rotation=tensors["rotation"],
            source_mean=tensors["source_mean"],
            target_mean=tensors["target_mean"],
            variance_scale=tensors.get("variance_scale"),
            center_strength=center_strength,
            variance_strength=variance_strength,
            metadata=metadata,
        )

    def _runtime_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Alignment is numerically more stable in fp32.  Cache per device while
        # keeping the artifact itself on CPU so model CPU offload does not mutate it.
        key = (str(device), torch.float32)
        cached = self._cache.get(key)
        if cached is None:
            rotation = self.rotation.to(device=device, dtype=torch.float32)
            source_mean = self.source_mean.to(device=device, dtype=torch.float32)
            target_mean = self.target_mean.to(device=device, dtype=torch.float32)
            variance_scale = (
                self.variance_scale.to(device=device, dtype=torch.float32)
                if self.variance_scale is not None
                else None
            )
            cached = (rotation, source_mean, target_mean, variance_scale)
            self._cache[key] = cached
        return cached  # type: ignore[return-value]

    def clear_runtime_cache(self) -> None:
        self._cache.clear()

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "AnimaTextEncoderBridge expects hidden states shaped (batch, sequence, hidden)."
            )
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"Bridge hidden size is {self.hidden_size}, got {int(hidden_states.shape[-1])}."
            )
        output_dtype = hidden_states.dtype
        rotation, source_mean, target_mean, variance_scale = self._runtime_tensors(
            hidden_states.device, hidden_states.dtype
        )
        x = hidden_states.float()
        centered = x - source_mean
        aligned_centered = torch.matmul(centered, rotation)
        if variance_scale is not None and self.variance_strength != 0.0:
            scale = 1.0 + (variance_scale - 1.0) * self.variance_strength
            aligned_centered = aligned_centered * scale
        center = source_mean + (target_mean - source_mean) * self.center_strength
        return (aligned_centered + center).to(dtype=output_dtype)

    def describe(self) -> dict[str, Any]:
        return {
            "format": self.metadata.get("format", _BRIDGE_FORMAT),
            "source_family": self.metadata.get("source_family", "unknown"),
            "target_family": self.metadata.get("target_family", "unknown"),
            "samples": int(self.metadata.get("samples", "0") or 0),
            "hidden_size": self.hidden_size,
            "center_strength": self.center_strength,
            "variance_strength": self.variance_strength,
            "has_variance_scale": self.variance_scale is not None,
        }


__all__ = ["AnimaTextEncoderBridge"]
