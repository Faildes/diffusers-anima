"""Anima-native Qwen3.5 text encoder.

This module is the bridge-free endpoint for the alternate-encoder work.  The
Qwen3.5-0.8B text backbone is kept as the knowledge source, but the final
representation is produced by a trainable Anima-native head that is part of the
encoder itself:

    selected Qwen3.5 hidden layers
        -> token-dependent layer mixer
        -> residual semantic block
        -> learned Anima projection / bias
        -> optional learned RMS stabilisation
        -> [B, N, 1024] Anima source memory

No runtime Procrustes bridge is required for an ``anima_native_text_encoder_v1``
artifact.  A bridge profile may still be used *during training* as a bootstrap
teacher/initialiser, but no bridge tensors are needed by the final checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast


_NATIVE_ENCODER_FORMAT_V1 = "anima_native_text_encoder_v1"
_NATIVE_ENCODER_KIND = "native_encoder"


def is_anima_native_text_encoder_metadata(metadata: dict[str, str] | None) -> bool:
    md = metadata or {}
    return (
        md.get("format") == _NATIVE_ENCODER_FORMAT_V1
        and md.get("artifact_kind") == _NATIVE_ENCODER_KIND
    )


def _parse_layer_indices(value: str | Sequence[int] | None, *, num_layers: int) -> tuple[int, ...]:
    if value is None:
        # hidden_states[0] is the embedding output.  These are therefore roughly
        # 25/50/75/100% depth for the 24-layer Qwen3.5-0.8B backbone.
        # Qwen3.5 uses a 3:1 linear/full-attention stack.  For the 24-layer
        # 0.8B model the full-attention milestones are 4/8/12/16/20/24.
        # Sampling six roughly-equal depth milestones preserves lexical/local
        # features from the middle of the network while still exposing the
        # strongest final semantic representation.
        steps = min(6, max(1, int(num_layers)))
        raw = [max(1, round(num_layers * i / steps)) for i in range(1, steps + 1)]
    elif isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("native_layer_indices_json must decode to a list")
        raw = [int(x) for x in parsed]
    else:
        raw = [int(x) for x in value]

    out: list[int] = []
    for idx in raw:
        # Indexing refers to the hidden-state tuple, whose valid backbone layer
        # outputs are 1..num_layers.  Negative indices are accepted for tools.
        resolved = idx if idx >= 0 else (num_layers + 1 + idx)
        resolved = max(1, min(num_layers, resolved))
        if resolved not in out:
            out.append(resolved)
    if not out:
        raise ValueError("Anima-native layer mixer needs at least one hidden layer")
    return tuple(out)


class _NativeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype=dtype)


@dataclass(frozen=True)
class AnimaNativeHeadConfig:
    hidden_size: int = 1024
    intermediate_size: int = 1536
    layer_indices: tuple[int, ...] = (4, 8, 12, 16, 20, 24)
    norm_eps: float = 1e-6
    final_layer_logit_bias: float = 3.0
    max_subject_slots: int = 16
    max_separator_types: int = 3
    binding_rms_min_ratio: float = 0.92
    binding_rms_max_ratio: float = 1.08

    @classmethod
    def from_metadata(cls, metadata: dict[str, str], *, num_layers: int, hidden_size: int) -> "AnimaNativeHeadConfig":
        def _int(key: str, default: int) -> int:
            try:
                return int(metadata.get(key, default))
            except (TypeError, ValueError):
                return int(default)

        def _float(key: str, default: float) -> float:
            try:
                return float(metadata.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        return cls(
            hidden_size=int(hidden_size),
            intermediate_size=_int("native_intermediate_size", max(1536, hidden_size)),
            layer_indices=_parse_layer_indices(
                metadata.get("native_layer_indices_json"), num_layers=int(num_layers)
            ),
            norm_eps=_float("native_norm_eps", 1e-6),
            final_layer_logit_bias=_float("native_final_layer_logit_bias", 3.0),
            max_subject_slots=max(2, _int("native_max_subject_slots", 16)),
            max_separator_types=max(3, _int("native_max_separator_types", 3)),
            binding_rms_min_ratio=_float("native_binding_rms_min_ratio", 0.92),
            binding_rms_max_ratio=_float("native_binding_rms_max_ratio", 1.08),
        )


class AnimaNativeQwen35Head(nn.Module):
    """Trainable, knowledge-preserving Anima output head.

    The token-dependent layer gate is intentionally initialised to select the
    final Qwen layer almost exclusively.  ``output_proj`` starts as identity (or
    can be initialised from a previous bridge during training), and both
    residual gates start at zero.  Training therefore begins from the original
    0.8B representation instead of destroying it at initialisation time.
    """

    def __init__(self, config: AnimaNativeHeadConfig):
        super().__init__()
        self.native_config = config
        dim = int(config.hidden_size)
        k = len(config.layer_indices)

        logits = torch.zeros(k, dtype=torch.float32)
        logits[-1] = float(config.final_layer_logit_bias)
        self.layer_logits = nn.Parameter(logits)
        self.layer_gate = nn.Linear(dim, k, bias=True)
        nn.init.zeros_(self.layer_gate.weight)
        nn.init.zeros_(self.layer_gate.bias)

        self.pre_norm = _NativeRMSNorm(dim, eps=config.norm_eps)
        self.fc1 = nn.Linear(dim, int(config.intermediate_size), bias=True)
        self.fc2 = nn.Linear(int(config.intermediate_size), dim, bias=True)
        self.residual_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # A full learned projection is part of the final encoder rather than a
        # runtime bridge.  Identity initialisation preserves the 0.8B source;
        # training may optionally bootstrap it from a historical bridge.
        self.output_proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.output_norm = _NativeRMSNorm(dim, eps=config.norm_eps)
        self.output_norm_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # Multi-subject ownership head.  These signals are deliberately small
        # residuals on top of the already Anima-compatible representation:
        # - slot embeddings distinguish character/subject clauses,
        # - count embeddings make exact cardinality an explicit global cue,
        # - binding_gate amplifies each slot's residual from the global centroid.
        # Gates start at zero so old behaviour is the exact initial condition.
        max_slots = int(config.max_subject_slots)
        self.slot_embedding = nn.Embedding(max_slots + 1, dim, padding_idx=0)
        self.count_embedding = nn.Embedding(max_slots + 1, dim, padding_idx=0)
        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.count_embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.slot_embedding.weight[0].zero_()
            self.count_embedding.weight[0].zero_()
        self.slot_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.count_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.binding_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # Separator provenance is distinct from subject identity. A weak
        # boundary residual lets AND/BREAK/semicolon survive the bridge-free
        # path without turning style/background clauses into character slots.
        self.separator_embedding = nn.Embedding(int(config.max_separator_types) + 1, dim, padding_idx=0)
        nn.init.normal_(self.separator_embedding.weight, mean=0.0, std=0.005)
        with torch.no_grad():
            self.separator_embedding.weight[0].zero_()
        self.separator_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(self.native_config.layer_indices)

    def initialise_from_linear_alignment(
        self,
        rotation: torch.Tensor,
        source_mean: torch.Tensor,
        target_mean: torch.Tensor,
        *,
        strength: float = 1.0,
    ) -> None:
        """Blend the learned output projection toward a centred Procrustes map.

        This is a *training initialiser*.  The resulting native checkpoint stores
        only ordinary learned encoder parameters; it does not require the bridge
        profile at runtime.
        """
        strength = max(0.0, min(1.0, float(strength)))
        if tuple(rotation.shape) != tuple(self.output_proj.weight.shape):
            raise ValueError(
                f"alignment rotation shape {tuple(rotation.shape)} does not match native projection "
                f"{tuple(self.output_proj.weight.shape)}"
            )
        r = rotation.detach().float()
        mu_s = source_mean.detach().float().reshape(-1)
        mu_t = target_mean.detach().float().reshape(-1)
        target_weight = r.T.contiguous()
        target_bias = (mu_t - torch.matmul(mu_s, r)).contiguous()
        with torch.no_grad():
            self.output_proj.weight.lerp_(target_weight.to(self.output_proj.weight), strength)
            self.output_proj.bias.lerp_(target_bias.to(self.output_proj.bias), strength)

    def _apply_binding_controls(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None,
        subject_counts: torch.Tensor | None,
        separator_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Inject slot/count ownership without shifting Anima activation scale.

        PromptPlan groups are treated as subject/semantic slots when supplied by
        sd_embed.  The operation is residual and followed by a per-token RMS
        guard so stronger identity binding cannot become a global saturation or
        contrast shift in the image model.
        """
        if group_ids is None and subject_counts is None and separator_ids is None:
            zero = hidden.new_zeros(())
            return hidden, {
                "slot_gate": zero, "count_gate": zero, "binding_gate": zero, "separator_gate": zero
            }

        base = hidden
        out = hidden
        bsz, seq_len, dim = hidden.shape
        if attention_mask is None:
            mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=hidden.device)
        else:
            mask = attention_mask.to(device=hidden.device).bool()
            if mask.shape[1] != seq_len:
                mask = mask[:, :seq_len]

        slot_strength = torch.tanh(self.slot_gate).to(dtype=hidden.dtype)
        count_strength = torch.tanh(self.count_gate).to(dtype=hidden.dtype)
        binding_strength = torch.tanh(self.binding_gate).to(dtype=hidden.dtype)
        separator_strength = torch.tanh(self.separator_gate).to(dtype=hidden.dtype)

        if separator_ids is not None:
            separators = separator_ids.to(device=hidden.device, dtype=torch.long)
            if separators.shape[1] < seq_len:
                separators = F.pad(separators, (0, seq_len - separators.shape[1]), value=0)
            elif separators.shape[1] > seq_len:
                separators = separators[:, :seq_len]
            separators = separators.clamp_(0, self.separator_embedding.num_embeddings - 1)
            boundary = self.separator_embedding(separators).to(dtype=hidden.dtype)
            out = out + separator_strength * boundary * mask.unsqueeze(-1).to(hidden.dtype)

        if group_ids is not None:
            groups = group_ids.to(device=hidden.device, dtype=torch.long)
            if groups.shape[1] < seq_len:
                groups = F.pad(groups, (0, seq_len - groups.shape[1]), value=0)
            elif groups.shape[1] > seq_len:
                groups = groups[:, :seq_len]
            groups = groups.clamp_(0, self.slot_embedding.num_embeddings - 1)
            slot = self.slot_embedding(groups).to(dtype=hidden.dtype)
            out = out + slot_strength * slot * mask.unsqueeze(-1).to(hidden.dtype)

            # Keep each subject clause direction distinguishable even when the
            # same attribute words occur elsewhere.  This is intentionally
            # centroid-residual based rather than orthogonalisation, which would
            # distort the 0.6B-compatible geometry too aggressively.
            if bool(mask.any()):
                separated = out.clone()
                for b in range(bsz):
                    valid = mask[b]
                    if not bool(valid.any()):
                        continue
                    global_center = out[b, valid].mean(dim=0, keepdim=True)
                    for gid in torch.unique(groups[b, valid]):
                        idx = valid & (groups[b] == gid)
                        if not bool(idx.any()):
                            continue
                        group_center = out[b, idx].mean(dim=0, keepdim=True)
                        separated[b, idx] = separated[b, idx] + binding_strength * (group_center - global_center)
                out = separated

        if subject_counts is not None:
            counts = subject_counts.to(device=hidden.device, dtype=torch.long).reshape(-1)
            if counts.numel() == 1 and bsz > 1:
                counts = counts.expand(bsz)
            if counts.numel() != bsz:
                raise ValueError(f"subject_counts must have batch size {bsz}, got {counts.numel()}")
            counts = counts.clamp_(0, self.count_embedding.num_embeddings - 1)
            count_vec = self.count_embedding(counts).to(dtype=hidden.dtype).unsqueeze(1)
            out = out + count_strength * count_vec * mask.unsqueeze(-1).to(hidden.dtype)

        # Saturation/contrast guard: binding may change direction but is not
        # allowed to inflate or collapse per-token RMS outside a narrow band
        # around the already-trained Anima-compatible base representation.
        base_rms = base.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        out_rms = out.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        ratio = (out_rms / base_rms).clamp(
            min=float(self.native_config.binding_rms_min_ratio),
            max=float(self.native_config.binding_rms_max_ratio),
        )
        target_rms = base_rms * ratio
        out = (out.float() * (target_rms / out_rms)).to(dtype=hidden.dtype)
        out = torch.where(mask.unsqueeze(-1), out, base)
        return out, {
            "slot_gate": slot_strength.float(),
            "count_gate": count_strength.float(),
            "binding_gate": binding_strength.float(),
            "separator_gate": separator_strength.float(),
        }

    def _select_layers(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not hidden_states:
            raise ValueError("Qwen3.5 did not return hidden_states for Anima-native mixing")
        selected: list[torch.Tensor] = []
        for idx in self.layer_indices:
            if idx >= len(hidden_states):
                raise ValueError(
                    f"native layer index {idx} is out of range for {len(hidden_states)} hidden states"
                )
            selected.append(hidden_states[idx])
        return selected

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        subject_counts: torch.Tensor | None = None,
        separator_ids: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selected = self._select_layers(hidden_states)
        final_hidden = hidden_states[-1]
        stacked = torch.stack(selected, dim=-2)  # [B, N, K, D]

        # Keep the Linear matmul in the head parameter dtype (normally BF16 on
        # CUDA) and promote only the tiny K-way gate logits to FP32 for the
        # numerically sensitive softmax.  Casting ``final_hidden`` to FP32 before
        # a BF16 Linear raises: mat1 and mat2 must have the same dtype.
        gate_input = final_hidden.to(dtype=self.layer_gate.weight.dtype)
        token_logits = self.layer_gate(gate_input).float()  # ANIMA_NATIVE_BF16_SAFE_GATE_V1
        token_logits = token_logits + self.layer_logits.float().view(1, 1, -1)
        layer_weights = torch.softmax(token_logits, dim=-1).to(dtype=stacked.dtype)
        mixed = (stacked * layer_weights.unsqueeze(-1)).sum(dim=-2)

        residual = self.fc2(F.silu(self.fc1(self.pre_norm(mixed))))
        semantic = mixed + torch.tanh(self.residual_gate).to(mixed.dtype) * residual
        projected = self.output_proj(semantic)
        normed = self.output_norm(projected)
        out = projected + torch.tanh(self.output_norm_gate).to(projected.dtype) * (normed - projected)
        out, binding_details = self._apply_binding_controls(
            out,
            attention_mask=attention_mask,
            group_ids=group_ids,
            subject_counts=subject_counts,
            separator_ids=separator_ids,
        )

        if not return_details:
            return out
        return out, {
            "mixed": mixed,
            "final_hidden": final_hidden,
            "semantic": semantic,
            "projected": projected,
            "layer_weights": layer_weights,
            **binding_details,
        }


def _module_runtime_dtype(module: nn.Module) -> torch.dtype:
    """Return a Diffusers/Transformers-compatible runtime dtype for ``module``.

    Plain ``torch.nn.Module`` intentionally has no ``dtype`` property, while
    both Diffusers and Transformers model classes expose one.  The native Qwen
    wrapper is registered as a pipeline component, so DiffusionPipeline.to()
    also expects that contract.  Prefer an existing model property and fall
    back to the first floating/complex parameter or buffer.
    """
    try:
        dtype = getattr(module, "dtype", None)
    except (AttributeError, RuntimeError):
        dtype = None
    if isinstance(dtype, torch.dtype):
        return dtype

    fallback: torch.dtype | None = None
    for tensors in (module.parameters(), module.buffers()):
        for tensor in tensors:
            fallback = tensor.dtype
            if tensor.is_floating_point() or tensor.is_complex():
                return tensor.dtype
    return fallback or torch.float32


def _module_runtime_device(module: nn.Module) -> torch.device:
    """Return a Diffusers/Transformers-compatible runtime device for ``module``."""
    try:
        device = getattr(module, "device", None)
    except (AttributeError, RuntimeError):
        device = None
    if isinstance(device, torch.device):
        return device
    if isinstance(device, (str, int)):
        try:
            return torch.device(device)
        except (TypeError, RuntimeError):
            pass

    for tensors in (module.parameters(), module.buffers()):
        for tensor in tensors:
            return tensor.device
    return torch.device("cpu")


class AnimaNativeQwen35Encoder(nn.Module):
    """Qwen3.5-0.8B backbone with an integrated Anima-native output head.

    This is intentionally a lightweight ``nn.Module`` wrapper rather than a
    second ``PreTrainedModel``.  It nevertheless exposes the small runtime
    interface (notably ``dtype`` and ``device``) expected from registered
    Diffusers/Transformers model components.
    """

    main_input_name = "input_ids"

    def __init__(self, backbone: nn.Module, native_head: AnimaNativeQwen35Head):
        super().__init__()
        self.backbone = backbone
        self.native_head = native_head
        self.config = getattr(backbone, "config", None)
        self._anima_text_encoder_family = "qwen3.5"
        self._anima_source_text_encoder_family = "qwen3.5"
        self._anima_native_encoder = True
        self._anima_conditioning_ready = True

    @property
    def dtype(self) -> torch.dtype:
        # DiffusionPipeline.to() unconditionally inspects ``module.dtype`` for
        # registered torch modules after moving them.  Delegate to the Qwen
        # backbone so the wrapper behaves like a Transformers model.
        return _module_runtime_dtype(self.backbone)

    @property
    def device(self) -> torch.device:
        # DiffusionPipeline.device and several offload/device helpers expect the
        # same property that PreTrainedModel exposes.
        return _module_runtime_device(self.backbone)

    @property
    def is_gradient_checkpointing(self) -> bool:
        # Keep training/tooling introspection compatible with the wrapped Qwen
        # model without changing the wrapper's state-dict layout.
        return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

    def get_input_embeddings(self):
        getter = getattr(self.backbone, "get_input_embeddings", None)
        return getter() if callable(getter) else None

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any):
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self):
        fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if callable(fn):
            return fn()
        return None

    def enable_input_require_grads(self):
        fn = getattr(self.backbone, "enable_input_require_grads", None)
        if callable(fn):
            return fn()
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = False,
        return_native_details: bool = False,
        anima_group_ids: torch.Tensor | None = None,
        anima_subject_counts: torch.Tensor | None = None,
        anima_separator_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # The native head always requires the backbone hidden-state stack.  Cache
        # states are not useful for image-conditioning feature extraction.
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        native = self.native_head(
            backbone_out.hidden_states,
            attention_mask=attention_mask,
            group_ids=anima_group_ids,
            subject_counts=anima_subject_counts,
            separator_ids=anima_separator_ids,
            return_details=bool(return_native_details),
        )
        if return_native_details:
            native_hidden, details = native
            return native_hidden, details, backbone_out
        native_hidden = native

        want_dict = True if return_dict is None else bool(return_dict)
        if not want_dict:
            # Match the common first-tensor convention used by text_encoding.py.
            return (native_hidden,)

        exposed_hidden_states = None
        if output_hidden_states:
            exposed_hidden_states = tuple(backbone_out.hidden_states or ()) + (native_hidden,)
        return BaseModelOutputWithPast(
            last_hidden_state=native_hidden,
            past_key_values=None,
            hidden_states=exposed_hidden_states,
            attentions=getattr(backbone_out, "attentions", None),
        )

    def native_description(self) -> dict[str, Any]:
        return {
            "format": _NATIVE_ENCODER_FORMAT_V1,
            "artifact_kind": _NATIVE_ENCODER_KIND,
            "hidden_size": int(self.native_head.native_config.hidden_size),
            "intermediate_size": int(self.native_head.native_config.intermediate_size),
            "layer_indices": list(self.native_head.layer_indices),
            "max_subject_slots": int(self.native_head.native_config.max_subject_slots),
            "binding_head": "slot_count_group_separator_v3",
            "bridge_required": False,
        }


def native_head_metadata(config: AnimaNativeHeadConfig) -> dict[str, str]:
    return {
        "native_hidden_size": str(int(config.hidden_size)),
        "native_intermediate_size": str(int(config.intermediate_size)),
        "native_layer_indices_json": json.dumps(list(config.layer_indices), separators=(",", ":")),
        "native_norm_eps": f"{float(config.norm_eps):.8g}",
        "native_final_layer_logit_bias": f"{float(config.final_layer_logit_bias):.8g}",
        "native_max_subject_slots": str(int(config.max_subject_slots)),
        "native_max_separator_types": str(int(config.max_separator_types)),
        "native_binding_rms_min_ratio": f"{float(config.binding_rms_min_ratio):.8g}",
        "native_binding_rms_max_ratio": f"{float(config.binding_rms_max_ratio):.8g}",
        "native_binding_head": "slot_count_group_separator_v3",
        "native_layer_mixer": "token_dependent_softmax_v2_full_attention_milestones",
        "native_semantic_block": "rmsnorm_silu_residual_mlp_v1",
        "native_output_projection": "learned_linear_bias_v1",
    }


__all__ = [
    "_NATIVE_ENCODER_FORMAT_V1",
    "_NATIVE_ENCODER_KIND",
    "AnimaNativeHeadConfig",
    "AnimaNativeQwen35Head",
    "AnimaNativeQwen35Encoder",
    "is_anima_native_text_encoder_metadata",
    "native_head_metadata",
]
