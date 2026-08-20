"""TCAtria1B text encoder runtime for Anima / Anima 2.9B.

The checkpoint format is the one produced by the Team-C TCAtria1B trainer:
  hybrid_config.json
  model.safetensors
  tokenizer.json / tokenizer_config.json / vocab.json ...

TCAtria1B emits 1024-dimensional source hidden states.  Anima's existing LLM
adapter consumes those states and maps them to the 512-token final conditioning
used by both 28-block Anima and 40-block Anima 2.9B transformers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3_5TextConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


@dataclass
class TCAtria1BConfig:
    architecture: str = "TCAtria1B"
    hidden_size: int = 1024
    vocab_size: int = 248320
    num_macro_blocks: int = 6
    qwen3_gqa_source_layers: tuple[int, ...] = (0, 5, 11, 16, 22, 27)
    qwen35_layers_per_macro: int = 4
    adapter_intermediate_size: int = 1728
    fusion_init_scale: float = 1.0e-3
    adapter_init_scale: float = 1.0e-3
    qwen3_config: dict[str, Any] | None = None
    qwen35_text_config: dict[str, Any] | None = None
    model_type: str = "tcatria1b"
    default_source_max_length: int = 1024

    @classmethod
    def load(cls, path: str | Path) -> "TCAtria1BConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "qwen3_gqa_source_layers" in data:
            data["qwen3_gqa_source_layers"] = tuple(data["qwen3_gqa_source_layers"])
        # Older training outputs used the working architecture name.  It does not
        # affect tensor names; normalize it at runtime to the public model name.
        data["architecture"] = "TCAtria1B"
        supported = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in supported})

    def q3(self) -> Qwen3Config:
        if self.qwen3_config is None:
            raise ValueError("TCAtria1B config is missing qwen3_config")
        return Qwen3Config.from_dict(self.qwen3_config)

    def q35(self) -> Qwen3_5TextConfig:
        if self.qwen35_text_config is None:
            raise ValueError("TCAtria1B config is missing qwen35_text_config")
        return Qwen3_5TextConfig.from_dict(self.qwen35_text_config)


@dataclass
class TCAtria1BOutput:
    last_hidden_state: torch.Tensor
    pre_adapter_hidden_state: torch.Tensor | None = None
    macro_hidden_states: tuple[torch.Tensor, ...] | None = None
    pre_fusion_macro_hidden_states: tuple[torch.Tensor, ...] | None = None

    # Transformers-style tuple compatibility used by generic callers.
    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.last_hidden_state
        raise IndexError(index)


class _DualAttentionFusionLayer(nn.Module):
    def __init__(self, cfg: TCAtria1BConfig, macro_idx: int):
        super().__init__()
        q3_cfg = cfg.q3()
        q35_cfg = cfg.q35()
        q3_source_idx = cfg.qwen3_gqa_source_layers[macro_idx]
        q35_full_idx = macro_idx * 4 + 3

        self.input_layernorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=q35_cfg.rms_norm_eps)
        self.gqa = Qwen3Attention(q3_cfg, layer_idx=q3_source_idx)
        self.full_attn = Qwen3_5Attention(q35_cfg, layer_idx=q35_full_idx)
        self.fusion_gate = nn.Linear(cfg.hidden_size, 2, bias=True)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.fusion_init_scale)))
        self.post_attention_layernorm = Qwen3_5RMSNorm(cfg.hidden_size, eps=q35_cfg.rms_norm_eps)
        self.mlp = Qwen3_5MLP(q35_cfg, q35_cfg.intermediate_size)
        self.ffn_scale = nn.Parameter(torch.tensor(float(cfg.fusion_init_scale)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        q3_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        q35_position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        gqa_out, _ = self.gqa(
            hidden_states=h,
            position_embeddings=q3_position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
        )
        full_out, _ = self.full_attn(
            hidden_states=h,
            position_embeddings=q35_position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
        )
        weights = torch.softmax(self.fusion_gate(h).float(), dim=-1).to(h.dtype)
        mixed = weights[..., :1] * gqa_out + weights[..., 1:] * full_out
        hidden_states = residual + self.attn_scale.to(mixed.dtype) * mixed

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        return residual + self.ffn_scale.to(h.dtype) * self.mlp(h)


class _AnimaConditioningAdapter(nn.Module):
    def __init__(self, cfg: TCAtria1BConfig):
        super().__init__()
        q35_cfg = cfg.q35()
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=q35_cfg.rms_norm_eps)
        mid = cfg.adapter_intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, mid, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, mid, bias=False)
        self.down_proj = nn.Linear(mid, cfg.hidden_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(cfg.adapter_init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        return x + self.scale.to(h.dtype) * h


class TCAtria1BModel(nn.Module):
    """Team-C Atria 1B deployment model.

    Structure:
      6 x [GQA -> 3 Linear-Attention -> Full-Attention -> Dual Fusion]

    ``last_hidden_state`` is always 1024 wide and is intentionally compatible
    with the existing Anima LLM adapter.
    """

    def __init__(self, cfg: TCAtria1BConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        q3_cfg = cfg.q3()
        q35_cfg = cfg.q35()

        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=q35_cfg.pad_token_id
        )
        self.gqa_layers = nn.ModuleList(
            [Qwen3DecoderLayer(q3_cfg, layer_idx=i) for i in cfg.qwen3_gqa_source_layers]
        )
        self.q35_layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(q35_cfg, layer_idx=i) for i in range(q35_cfg.num_hidden_layers)]
        )
        self.fusion_layers = nn.ModuleList(
            [_DualAttentionFusionLayer(cfg, macro_idx=i) for i in range(cfg.num_macro_blocks)]
        )
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=q35_cfg.rms_norm_eps)
        self.conditioning_adapter = _AnimaConditioningAdapter(cfg)
        self.q3_rotary = Qwen3RotaryEmbedding(q3_cfg)
        self.q35_rotary = Qwen3_5TextRotaryEmbedding(q35_cfg)
        self.gradient_checkpointing = False
        self.name_or_path = "TCAtria1B"

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.enable_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        self.enable_gradient_checkpointing(False)

    @staticmethod
    def _linear_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if bool(torch.all(attention_mask == 1)):
            return None
        return attention_mask

    def _run(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_macro_states: bool = False,
        **_: Any,
    ) -> TCAtria1BOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seq_len, _ = hidden_states.shape

        if attention_mask is None:
            attention_mask = torch.ones(
                batch, seq_len, device=hidden_states.device, dtype=torch.long
            )
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch, -1)

        causal_mask = create_causal_mask(
            config=self.cfg.q35(),
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        linear_mask = self._linear_mask(attention_mask)
        q3_pos = self.q3_rotary(hidden_states, position_ids)
        q35_pos = self.q35_rotary(hidden_states, position_ids)

        macro_states: list[torch.Tensor] = []
        pre_fusion_macro_states: list[torch.Tensor] = []
        for macro_idx in range(self.cfg.num_macro_blocks):
            gqa_layer = self.gqa_layers[macro_idx]
            hidden_states = self._run(
                lambda h, layer=gqa_layer: layer(
                    h,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=q3_pos,
                    past_key_values=None,
                    use_cache=False,
                ),
                hidden_states,
            )

            q35_base = macro_idx * 4
            for local_idx in range(4):
                layer_idx = q35_base + local_idx
                layer = self.q35_layers[layer_idx]
                mask = linear_mask if local_idx < 3 else causal_mask
                hidden_states = self._run(
                    lambda h, layer=layer, mask=mask: layer(
                        h,
                        position_embeddings=q35_pos,
                        attention_mask=mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                    ),
                    hidden_states,
                )

            if return_macro_states:
                pre_fusion_macro_states.append(hidden_states)

            fusion = self.fusion_layers[macro_idx]
            hidden_states = self._run(
                lambda h, fusion=fusion: fusion(
                    h,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    q3_position_embeddings=q3_pos,
                    q35_position_embeddings=q35_pos,
                ),
                hidden_states,
            )
            if return_macro_states:
                macro_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        pre_adapter = hidden_states
        hidden_states = self.conditioning_adapter(hidden_states)
        return TCAtria1BOutput(
            last_hidden_state=hidden_states,
            pre_adapter_hidden_state=pre_adapter,
            macro_hidden_states=tuple(macro_states) if return_macro_states else None,
            pre_fusion_macro_hidden_states=(
                tuple(pre_fusion_macro_states) if return_macro_states else None
            ),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> "TCAtria1BModel":
        path = Path(model_dir).expanduser().resolve()
        if not is_tcatria1b_directory(path):
            raise ValueError(
                f"Not a TCAtria1B directory: {path}. Expected hybrid_config.json and model.safetensors."
            )
        cfg = TCAtria1BConfig.load(path / "hybrid_config.json")
        if cfg.hidden_size != 1024:
            raise ValueError(f"TCAtria1B must emit hidden_size=1024 for Anima, got {cfg.hidden_size}")
        old_dtype = torch.get_default_dtype()
        try:
            if dtype is not None:
                torch.set_default_dtype(dtype)
            model = cls(cfg)
        finally:
            torch.set_default_dtype(old_dtype)
        load_model(model, str(path / "model.safetensors"), strict=True)
        if dtype is not None:
            model.to(dtype=dtype)
        model.eval().requires_grad_(False)
        model.to(device=device)
        return model


def is_tcatria1b_directory(path: str | Path) -> bool:
    p = Path(path).expanduser()
    return p.is_dir() and (p / "hybrid_config.json").is_file() and (p / "model.safetensors").is_file()
