from __future__ import annotations

import numbers
import re
from typing import Any

from diffusers import CosmosTransformer3DModel, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffusers.utils import USE_PEFT_BACKEND, set_weights_and_activate_adapters
import torch
from torch import nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    feature_dim = x.shape[-1]
    if feature_dim % 2 != 0:
        raise ValueError(
            f"RoPE rotate_half expects even feature dim, got {feature_dim}."
        )
    half = feature_dim // 2
    paired = x.reshape(*x.shape[:-1], 2, half)
    first, second = paired.unbind(dim=-2)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


def _expand_attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    casted = mask.to(torch.bool)
    if casted.ndim == 2:
        return casted[:, None, None, :]
    return casted


def _build_position_ids(
    batch_size: int, length: int, device: torch.device
) -> torch.Tensor:
    base = torch.arange(length, device=device, dtype=torch.long)
    return base.unsqueeze(0).expand(batch_size, -1)


def _build_source_position_ids(
    batch_size: int,
    length: int,
    device: torch.device,
    *,
    trained_length: int = 512,
) -> torch.Tensor:
    """Keep long source memory inside the adapter's trained RoPE position range.

    The Qwen encoder may expose more than 512 source tokens.  The adapter was
    trained with a bounded conditioning window, so raw source positions far past
    that range would introduce a second out-of-distribution change on top of the
    alternate encoder.  For long sources we retain every key/value token but
    continuously compress their adapter-side RoPE coordinates into [0, 511].
    """
    if length <= int(trained_length):
        return _build_position_ids(batch_size, length, device)
    base = torch.linspace(
        0.0,
        float(int(trained_length) - 1),
        steps=int(length),
        device=device,
        dtype=torch.float32,
    )
    return base.unsqueeze(0).expand(batch_size, -1)


def _build_long_context_window_starts(
    length: int,
    *,
    window_size: int,
    overlap: int,
) -> list[int]:
    """Return monotonically increasing source-window starts with no gaps.

    The long-source path intentionally keeps each cross-attention competition
    inside a window no larger than the adapter's native 512-token training
    range.  A modest overlap protects concepts that cross a boundary.
    """
    length = int(length)
    window_size = max(1, int(window_size))
    overlap = max(0, min(int(overlap), window_size - 1))
    if length <= window_size:
        return [0]
    stride = max(1, window_size - overlap)
    starts: list[int] = []
    start = 0
    while start < length:
        starts.append(start)
        if start + window_size >= length:
            break
        start += stride
    return starts


def _pad_to_length(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    pad_tokens = target_length - hidden_states.shape[1]
    if pad_tokens <= 0:
        return hidden_states
    return F.pad(hidden_states, (0, 0, 0, pad_tokens))


def _default_padding_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (1, 1, hidden_states.shape[-2], hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


class _AnimaRMSNorm(nn.Module):
    """RMSNorm implementation used by the Anima adapter blocks."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_diffusers(cls, module: DiffusersRMSNorm) -> "_AnimaRMSNorm":
        patched = cls(
            tuple(module.dim),
            eps=float(module.eps),
            elementwise_affine=module.weight is not None,
            bias=module.bias is not None,
        )
        with torch.no_grad():
            if module.weight is not None:
                patched.weight.copy_(module.weight)
            if module.bias is not None and patched.bias is not None:
                patched.bias.copy_(module.bias)
        return patched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            out = F.rms_norm(x, self.normalized_shape, eps=self.eps)
        else:
            out = F.rms_norm(
                x,
                self.normalized_shape,
                weight=self.weight.to(dtype=x.dtype, device=x.device),
                eps=self.eps,
            )
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype, device=out.device)
        return out


def _patch_diffusers_rmsnorm_to_anima(module: nn.Module) -> None:
    """Recursively replace Diffusers RMSNorm modules with Anima RMSNorm."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, DiffusersRMSNorm):
            setattr(module, child_name, _AnimaRMSNorm.from_diffusers(child))
            continue
        _patch_diffusers_rmsnorm_to_anima(child)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        half_dim = head_dim // 2
        index = torch.arange(half_dim, dtype=torch.float32)
        exponent = (2.0 / float(head_dim)) * index
        inv = torch.reciprocal(
            torch.pow(torch.tensor(theta, dtype=torch.float32), exponent)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(device=x.device, dtype=torch.float32)
        inv = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bl,d->bld", pos, inv)
        emb = freqs.repeat(1, 1, 2)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class _AdapterAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        inner = query_dim
        head_dim = inner // heads
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.q_norm = _AnimaRMSNorm(head_dim)
        self.k_norm = _AnimaRMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def _project_query(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.q_proj(x)
            .view(x.shape[0], x.shape[1], self.heads, self.head_dim)
            .transpose(1, 2)
        )

    def _project_key_value(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = (
            self.k_proj(context)
            .view(context.shape[0], context.shape[1], self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context)
            .view(context.shape[0], context.shape[1], self.heads, self.head_dim)
            .transpose(1, 2)
        )
        return self.k_norm(k), v

    def _windowed_source_attention(
        self,
        q: torch.Tensor,
        *,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None,
        rope: _RotaryEmbedding,
        window_size: int,
        overlap: int,
        router_top_k: int,
        router_temperature: float,
        locality_strength: float,
    ) -> torch.Tensor:
        """Attend to arbitrary source length through native-size memory banks.

        A single global softmax over >512 source tokens changes the entropy and
        positional regime that the frozen Anima adapter saw during training.
        Instead, every bank performs ordinary <=512-token attention with local
        RoPE positions.  A second, length-normalised router chooses a small
        number of relevant banks per target token/head.  The result still has
        the native target length, so the downstream DiT contract is unchanged.
        """
        source_len = int(context.shape[1])
        starts = _build_long_context_window_starts(
            source_len, window_size=int(window_size), overlap=int(overlap)
        )
        if len(starts) <= 1:
            k, v = self._project_key_value(context)
            local_ids = _build_position_ids(context.shape[0], context.shape[1], context.device)
            k = _apply_rope(k, *rope(context, local_ids))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            return y

        q_len = int(q.shape[-2])
        q_rel = torch.linspace(
            0.0, 1.0, steps=max(1, q_len), device=q.device, dtype=torch.float32
        ).view(1, 1, q_len)
        if q_len == 1:
            q_rel.zero_()

        bank_outputs: list[torch.Tensor] = []
        bank_evidence: list[torch.Tensor] = []
        score_scale = float(self.head_dim) ** -0.5
        neg_large = -1.0e4

        for start in starts:
            end = min(source_len, int(start) + int(window_size))
            ctx = context[:, start:end]
            k, v = self._project_key_value(ctx)
            local_ids = _build_position_ids(ctx.shape[0], ctx.shape[1], ctx.device)
            k = _apply_rope(k, *rope(ctx, local_ids))

            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * score_scale
            local_mask: torch.Tensor | None = None
            if attn_mask is not None:
                local_mask = attn_mask[..., start:end].to(device=scores.device, dtype=torch.bool)
                masked_scores = scores.masked_fill(~local_mask, float('-inf'))
                valid_count = local_mask.sum(dim=-1).clamp_min(1)
            else:
                masked_scores = scores
                valid_count = torch.full(
                    (*scores.shape[:-1],),
                    max(1, end - start),
                    device=scores.device,
                    dtype=torch.long,
                )

            # Within-bank competition is the same softmax shape the adapter was
            # trained on. Explicit mask renormalisation avoids NaNs for padded
            # banks in mixed-length batches.
            local_prob = torch.softmax(masked_scores, dim=-1)
            if local_mask is not None:
                local_prob = torch.where(local_mask, local_prob, torch.zeros_like(local_prob))
                denom = local_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                local_prob = local_prob / denom
            bank_y = torch.matmul(local_prob.to(dtype=v.dtype), v)

            # log-mean-exp removes the artificial preference for a bank merely
            # because it contains more tokens. This is the routing signal only;
            # token attention above remains ordinary softmax.
            evidence = torch.logsumexp(masked_scores, dim=-1) - torch.log(
                valid_count.to(dtype=masked_scores.dtype)
            )
            if local_mask is not None:
                valid_bank = local_mask.any(dim=-1)
                evidence = torch.where(valid_bank, evidence, torch.full_like(evidence, neg_large))

            if locality_strength != 0.0 and source_len > 1:
                center = (float(start) + float(max(start, end - 1))) * 0.5 / float(source_len - 1)
                locality = torch.abs(q_rel - float(center))
                evidence = evidence - locality * float(locality_strength)

            bank_outputs.append(bank_y)
            bank_evidence.append(evidence)

        stacked_y = torch.stack(bank_outputs, dim=0)
        logits = torch.stack(bank_evidence, dim=0)
        temperature = max(1e-4, float(router_temperature))
        logits = logits / temperature

        top_k = int(router_top_k)
        if 0 < top_k < int(logits.shape[0]):
            _top_values, top_indices = torch.topk(logits, k=top_k, dim=0)
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(0, top_indices, True)
            logits = logits.masked_fill(~keep, float('-inf'))

        gates = torch.softmax(logits, dim=0)
        gates = torch.nan_to_num(gates, nan=0.0, posinf=0.0, neginf=0.0)
        gate_sum = gates.sum(dim=0, keepdim=True).clamp_min(1e-6)
        gates = gates / gate_sum
        return (stacked_y * gates.unsqueeze(-1).to(dtype=stacked_y.dtype)).sum(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_q: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_k: tuple[torch.Tensor, torch.Tensor] | None = None,
        long_context_options: dict[str, Any] | None = None,
        rope: _RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context

        q = self.q_norm(self._project_query(x))
        if pos_q is not None:
            q = _apply_rope(q, *pos_q)

        options = long_context_options or {}
        use_windowed = (
            bool(options.get('enabled', False))
            and rope is not None
            and int(context.shape[1]) > int(options.get('threshold', 512))
        )
        if use_windowed:
            y = self._windowed_source_attention(
                q,
                context=context,
                attn_mask=attn_mask,
                rope=rope,
                window_size=int(options.get('window_size', 384)),
                overlap=int(options.get('overlap', 64)),
                router_top_k=int(options.get('router_top_k', 2)),
                router_temperature=float(options.get('router_temperature', 0.8)),
                locality_strength=float(options.get('locality_strength', 1.0)),
            )
        else:
            k, v = self._project_key_value(context)
            if pos_k is not None:
                k = _apply_rope(k, *pos_k)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1).contiguous()
        return self.o_proj(y)


class _AdapterBlock(nn.Module):
    def __init__(self, model_dim: int = 1024, context_dim: int = 1024, heads: int = 16):
        super().__init__()
        self.norm_self_attn = _AnimaRMSNorm(model_dim)
        self.self_attn = _AdapterAttention(model_dim, model_dim, heads)
        self.norm_cross_attn = _AnimaRMSNorm(model_dim)
        self.cross_attn = _AdapterAttention(model_dim, context_dim, heads)
        self.norm_mlp = _AnimaRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        target_mask: torch.Tensor | None,
        source_mask: torch.Tensor | None,
        pos_target: tuple[torch.Tensor, torch.Tensor],
        pos_source: tuple[torch.Tensor, torch.Tensor] | None,
        long_context_options: dict[str, Any] | None = None,
        rope: _RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm_self_attn(x),
            attn_mask=target_mask,
            pos_q=pos_target,
            pos_k=pos_target,
        )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            context=context,
            attn_mask=source_mask,
            pos_q=pos_target,
            pos_k=pos_source,
            long_context_options=long_context_options,
            rope=rope,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class _LLMAdapter(nn.Module):
    def __init__(
        self, vocab_size: int = 32128, dim: int = 1024, layers: int = 6, heads: int = 16
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                _AdapterBlock(model_dim=dim, context_dim=dim, heads=heads)
                for _ in range(layers)
            ]
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm = _AnimaRMSNorm(dim)
        self.rope = _RotaryEmbedding(dim // heads)
        # Runtime-only compatibility policies. No weights depend on these flags.
        # `source_position_mode` is retained for <=512/legacy A-B paths.
        self.source_position_mode = "compress"
        # v5: sources beyond the native training window are paged into local
        # memory banks rather than exposed to one >512-key softmax.
        self.long_context_mode = "windowed"
        self.long_context_threshold = 512
        self.long_context_window_size = 384
        self.long_context_overlap = 64
        self.long_context_router_top_k = 2
        self.long_context_router_temperature = 0.8
        self.long_context_locality_strength = 1.0

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_target_mask = _expand_attention_mask(target_attention_mask)
        normalized_source_mask = _expand_attention_mask(source_attention_mask)

        target_hidden_states = self.embed(target_input_ids)
        source_context = source_hidden_states

        target_position_ids = _build_position_ids(
            batch_size=target_hidden_states.shape[0],
            length=target_hidden_states.shape[1],
            device=target_hidden_states.device,
        )
        use_windowed_long_context = (
            self.long_context_mode == "windowed"
            and int(source_context.shape[1]) > int(self.long_context_threshold)
        )
        source_position_embed: tuple[torch.Tensor, torch.Tensor] | None
        if use_windowed_long_context:
            # Every source bank receives native local 0..window-1 RoPE inside
            # `_AdapterAttention`; do not pre-compress the complete sequence.
            source_position_embed = None
        else:
            if self.source_position_mode == "raw":
                source_position_ids = _build_position_ids(
                    batch_size=source_context.shape[0],
                    length=source_context.shape[1],
                    device=source_context.device,
                )
            else:
                source_position_ids = _build_source_position_ids(
                    batch_size=source_context.shape[0],
                    length=source_context.shape[1],
                    device=source_context.device,
                    trained_length=512,
                )
            source_position_embed = self.rope(source_context, source_position_ids)

        target_position_embed = self.rope(target_hidden_states, target_position_ids)
        long_context_options = {
            "enabled": use_windowed_long_context,
            "threshold": int(self.long_context_threshold),
            "window_size": int(self.long_context_window_size),
            "overlap": int(self.long_context_overlap),
            "router_top_k": int(self.long_context_router_top_k),
            "router_temperature": float(self.long_context_router_temperature),
            "locality_strength": float(self.long_context_locality_strength),
        }

        hidden_states = target_hidden_states
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                context=source_context,
                target_mask=normalized_target_mask,
                source_mask=normalized_source_mask,
                pos_target=target_position_embed,
                pos_source=source_position_embed,
                long_context_options=long_context_options,
                rope=self.rope,
            )
        return self.norm(self.out_proj(hidden_states))


class AnimaTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    @register_to_config
    def __init__(
        self,
        # CosmosTransformer3DModel core parameters
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] | list[int] = (128, 240, 240),
        patch_size: tuple[int, int, int] | list[int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] | list[float] = (1.0, 4.0, 4.0),
        # LLMAdapter parameters
        adapter_vocab_size: int = 32128,
        adapter_dim: int = 1024,
        adapter_layers: int = 6,
        adapter_heads: int = 16,
    ):
        super().__init__()
        core = _create_anima_transformer_core_model(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=tuple(max_size),
            patch_size=tuple(patch_size),
            rope_scale=tuple(rope_scale),
        )
        _patch_diffusers_rmsnorm_to_anima(core)
        self.core = core
        self.llm_adapter = _LLMAdapter(
            vocab_size=adapter_vocab_size,
            dim=adapter_dim,
            layers=adapter_layers,
            heads=adapter_heads,
        )

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor | None,
        t5xxl_weights: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_ids is None:
            return text_embeds

        # Qwen source memory may be longer than 512.  Only the T5/query/output
        # side is constrained to Anima's trained 512-position contract.
        if text_ids.shape[1] > 512:
            text_ids = text_ids[:, :512]
            if t5xxl_weights is not None:
                t5xxl_weights = t5xxl_weights[:, :512]
            if target_attention_mask is not None:
                target_attention_mask = target_attention_mask[:, :512]
        adapted = self.llm_adapter(
            text_embeds,
            text_ids,
            target_attention_mask=target_attention_mask,
            source_attention_mask=source_attention_mask,
        )
        if t5xxl_weights is not None:
            adapted = adapted * t5xxl_weights
        adapted = adapted[:, :512]
        return _pad_to_length(adapted, 512)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        t5xxl_weights = kwargs.pop("t5xxl_weights", None)
        if t5xxl_ids is not None:
            encoder_hidden_states = self.preprocess_text_embeds(
                encoder_hidden_states, t5xxl_ids, t5xxl_weights=t5xxl_weights
            )

        padding_mask = kwargs.pop("padding_mask", None)
        if padding_mask is None:
            # CosmosTransformer3DModel internally repeats this per batch, so keep batch=1 here.
            padding_mask = _default_padding_mask(hidden_states)

        sample = self.core(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)

    def set_adapters(
        self,
        adapter_names: list[str] | str,
        weights: float
        | dict[str, float]
        | list[float | dict[str, float] | None]
        | None = None,
    ) -> None:
        """Set active LoRA adapters without relying on Diffusers private model-name mappings."""
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        normalized_names = (
            [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        )
        if not isinstance(weights, list):
            normalized_weights = [weights] * len(normalized_names)
        else:
            normalized_weights = list(weights)

        if len(normalized_names) != len(normalized_weights):
            raise ValueError(
                f"Length of adapter names {len(normalized_names)} is not equal to the length of their weights "
                f"{len(normalized_weights)}."
            )

        resolved_weights = [
            weight if weight is not None else 1.0 for weight in normalized_weights
        ]
        set_weights_and_activate_adapters(self, normalized_names, resolved_weights)


def _create_anima_transformer_core_model(
    in_channels: int = 16,
    out_channels: int = 16,
    num_attention_heads: int = 16,
    attention_head_dim: int = 128,
    num_layers: int = 28,
    mlp_ratio: float = 4.0,
    text_embed_dim: int = 1024,
    adaln_lora_dim: int = 256,
    max_size: tuple[int, int, int] = (128, 240, 240),
    patch_size: tuple[int, int, int] = (1, 2, 2),
    rope_scale: tuple[float, float, float] = (1.0, 4.0, 4.0),
) -> CosmosTransformer3DModel:
    return CosmosTransformer3DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        text_embed_dim=text_embed_dim,
        adaln_lora_dim=adaln_lora_dim,
        max_size=max_size,
        patch_size=patch_size,
        rope_scale=rope_scale,
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )


def _convert_anima_state_dict_to_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    core: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}

    root_map = {
        "x_embedder.proj.1.weight": "core.patch_embed.proj.weight",
        "t_embedder.1.linear_1.weight": "core.time_embed.t_embedder.linear_1.weight",
        "t_embedder.1.linear_2.weight": "core.time_embed.t_embedder.linear_2.weight",
        "t_embedding_norm.weight": "core.time_embed.norm.weight",
        "final_layer.adaln_modulation.1.weight": "core.norm_out.linear_1.weight",
        "final_layer.adaln_modulation.2.weight": "core.norm_out.linear_2.weight",
        "final_layer.linear.weight": "core.proj_out.weight",
    }

    block_maps = {
        "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
        "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
        "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
        "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
        "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
        "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
        "self_attn.q_norm.weight": "attn1.norm_q.weight",
        "self_attn.k_norm.weight": "attn1.norm_k.weight",
        "self_attn.q_proj.weight": "attn1.to_q.weight",
        "self_attn.k_proj.weight": "attn1.to_k.weight",
        "self_attn.v_proj.weight": "attn1.to_v.weight",
        "self_attn.output_proj.weight": "attn1.to_out.0.weight",
        "cross_attn.q_norm.weight": "attn2.norm_q.weight",
        "cross_attn.k_norm.weight": "attn2.norm_k.weight",
        "cross_attn.q_proj.weight": "attn2.to_q.weight",
        "cross_attn.k_proj.weight": "attn2.to_k.weight",
        "cross_attn.v_proj.weight": "attn2.to_v.weight",
        "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
        "mlp.layer1.weight": "ff.net.0.proj.weight",
        "mlp.layer2.weight": "ff.net.2.weight",
    }

    # Cosmos/Anima checkpoints persist positional-index helper buffers which
    # Diffusers reconstructs from the transformer config at runtime. They are
    # not trainable model weights and have no destination in
    # CosmosTransformer3DModel.state_dict(). Keep this list explicit so an
    # actually unknown checkpoint key still fails loudly below.
    ignored_checkpoint_keys = {
        "pos_embedder.dim_spatial_range",
        "pos_embedder.dim_temporal_range",
        "pos_embedder.seq",
    }

    block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for key, value in state_dict.items():
        if key in ignored_checkpoint_keys:
            continue
        if key.startswith("llm_adapter."):
            adapter[".".join(["llm_adapter", key.removeprefix("llm_adapter.")])] = value
            continue

        mapped = root_map.get(key)
        if mapped is not None:
            core[mapped] = value
            continue

        m = block_re.match(key)
        if m is not None:
            block_index = m.group(1)
            tail = m.group(2)
            mapped_tail = block_maps.get(tail)
            if mapped_tail is None:
                raise RuntimeError(f"Unsupported Anima checkpoint key in blocks: {key}")
            core[f"core.transformer_blocks.{block_index}.{mapped_tail}"] = value
            continue

        raise RuntimeError(f"Unsupported Anima checkpoint key: {key}")

    return core, adapter
