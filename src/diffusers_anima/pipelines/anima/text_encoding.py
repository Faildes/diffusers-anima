"""Anima prompt tokenization and text conditioning utilities."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import ModelMixin
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

import torch

# Legacy fallback used only when a tokenizer does not expose pad/eos ids.
# Qwen3.5 tokenizers normally provide their own values, so model-specific IDs are
# never forced when they are available.
_QWEN3_DEFAULT_PAD_TOKEN_ID: int = 151643

# Maximum sequence length the LLM adapter conditioning tensor is padded / truncated to.
_CONDITIONING_MAX_LENGTH: int = 512


class AnimaPromptTokenizer:
    """Prompt tokenizer for Anima dual-encoder conditioning (Qwen3 + T5-XXL).

    Produces token IDs and per-token weights consumed by the pipeline.
    All weights are fixed at ``1.0`` in the current implementation (no
    parenthesis-weighted prompt syntax).
    """

    def __init__(
        self,
        qwen_tokenizer: "PreTrainedTokenizer" | "PreTrainedTokenizerFast",
        t5_tokenizer: "PreTrainedTokenizer" | "PreTrainedTokenizerFast",
    ) -> None:
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer

    def tokenize_with_weights(
        self, text: str
    ) -> dict[str, list[list[tuple[int, float]]]]:
        qwen_ids = (
            self.qwen_tokenizer(
                [text],
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
            .input_ids[0]
            .tolist()
        )
        t5_ids = (
            self.t5_tokenizer(
                [text],
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
            .input_ids[0]
            .tolist()
        )

        qwen_pad = self.qwen_tokenizer.pad_token_id
        if qwen_pad is None:
            qwen_pad = self.qwen_tokenizer.eos_token_id
        if qwen_pad is None:
            qwen_pad = _QWEN3_DEFAULT_PAD_TOKEN_ID
        if len(qwen_ids) == 0:
            qwen_ids = [int(qwen_pad)]

        t5_eos = self.t5_tokenizer.eos_token_id
        if t5_eos is None:
            t5_eos = 1
        if len(t5_ids) == 0:
            t5_ids = [int(t5_eos)]
        elif int(t5_ids[-1]) != int(t5_eos):
            t5_ids = [*t5_ids, int(t5_eos)]

        return {
            "qwen": [[(int(token_id), 1.0) for token_id in qwen_ids]],
            # Backward-compatible alias for callers that still inspect this key.
            "qwen3_06b": [[(int(token_id), 1.0) for token_id in qwen_ids]],
            "t5xxl": [[(int(token_id), 1.0) for token_id in t5_ids]],
        }


def _extract_ids_and_weights(
    token_weight_pairs: list[tuple[int | str, float]],
) -> tuple[list[int], list[float]]:
    token_ids: list[int] = []
    token_weights: list[float] = []
    for token, weight, *rest in token_weight_pairs:
        del rest
        if not isinstance(token, numbers.Integral):
            raise RuntimeError(
                "Prompt tokenizer returned a non-integer token, which is not supported in this pipeline."
            )
        token_ids.append(int(token))
        token_weights.append(float(weight))
    return token_ids, token_weights


def resolve_text_encoder_backbone(text_encoder: "PreTrainedModel") -> "PreTrainedModel":
    """Return the text backbone that yields token hidden states.

    Qwen3 is already a text-only model. Qwen3.5 may be represented either by a
    full multimodal wrapper (``model.language_model``) or, preferably, by the
    text-only causal-LM wrapper (``model``). Both expose the same language
    backbone to Anima without requiring a second text model.
    """
    direct = getattr(text_encoder, "language_model", None)
    if direct is not None:
        return direct
    model = getattr(text_encoder, "model", None)
    nested = getattr(model, "language_model", None) if model is not None else None
    if nested is not None:
        return nested
    # Qwen3.5 text-only causal LM: ``text_encoder.model`` is the text backbone.
    if model is not None and hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    return text_encoder


def prepare_condition_inputs(
    prompt_tokenizer: AnimaPromptTokenizer,
    text_encoder: "PreTrainedModel",
    prompt: list[str],
    *,
    execution_device: str,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize and encode a batch of prompts into conditioning tensors.

    Returns:
        qwen_hidden: Qwen hidden states, shape ``(B, T_q, D)``.
        qwen_mask: Source-token attention mask, shape ``(B, T_q)``.
        t5_ids: T5 token IDs, shape ``(B, T_t5)``.
        t5_weights: Per-token T5 weights, shape ``(B, T_t5, 1)``.
    """
    if len(prompt) == 0:
        raise ValueError("`prompt` batch must not be empty.")

    qwen_pad = prompt_tokenizer.qwen_tokenizer.pad_token_id
    if qwen_pad is None:
        qwen_pad = prompt_tokenizer.qwen_tokenizer.eos_token_id
    if qwen_pad is None:
        qwen_pad = _QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_pad = prompt_tokenizer.t5_tokenizer.pad_token_id
    if t5_pad is None:
        t5_pad = 0

    qwen_token_batches: list[list[int]] = []
    t5_token_batches: list[list[int]] = []
    t5_weight_batches: list[list[float]] = []
    max_qwen_len = 0
    max_t5_len = 0

    for text in prompt:
        tokenized = prompt_tokenizer.tokenize_with_weights(text)
        qwen_pairs = tokenized.get("qwen", tokenized["qwen3_06b"])[0]
        qwen_token_ids, _ = _extract_ids_and_weights(qwen_pairs)
        t5_token_ids, t5_token_weights = _extract_ids_and_weights(tokenized["t5xxl"][0])

        if len(qwen_token_ids) == 0:
            qwen_token_ids = [int(qwen_pad)]
        if len(t5_token_ids) == 0:
            t5_token_ids = [1]
            t5_token_weights = [1.0]

        qwen_token_batches.append(qwen_token_ids)
        t5_token_batches.append(t5_token_ids)
        t5_weight_batches.append(t5_token_weights)
        max_qwen_len = max(max_qwen_len, len(qwen_token_ids))
        max_t5_len = max(max_t5_len, len(t5_token_ids))

    batch_size = len(prompt)
    qwen_ids = torch.full(
        (batch_size, max_qwen_len),
        int(qwen_pad),
        dtype=torch.long,
        device=execution_device,
    )
    qwen_mask = torch.zeros(
        (batch_size, max_qwen_len), dtype=torch.long, device=execution_device
    )
    t5_ids = torch.full(
        (batch_size, max_t5_len),
        int(t5_pad),
        dtype=torch.int32,
        device=execution_device,
    )
    t5_weights = torch.zeros(
        (batch_size, max_t5_len, 1),
        dtype=torch.float32,
        device=execution_device,
    )

    for idx, (qwen_ids_item, t5_ids_item, t5_weights_item) in enumerate(
        zip(qwen_token_batches, t5_token_batches, t5_weight_batches, strict=True)
    ):
        q_len = len(qwen_ids_item)
        t_len = len(t5_ids_item)
        qwen_ids[idx, :q_len] = torch.tensor(
            qwen_ids_item, dtype=torch.long, device=execution_device
        )
        qwen_mask[idx, :q_len] = 1
        t5_ids[idx, :t_len] = torch.tensor(
            t5_ids_item, dtype=torch.int32, device=execution_device
        )
        t5_weights[idx, :t_len, 0] = torch.tensor(
            t5_weights_item, dtype=torch.float32, device=execution_device
        )

    with torch.inference_mode():
        text_backbone = resolve_text_encoder_backbone(text_encoder)
        text_encoder_out = text_backbone(input_ids=qwen_ids, attention_mask=qwen_mask)
        if isinstance(text_encoder_out, tuple):
            qwen_hidden = text_encoder_out[0]
        else:
            qwen_hidden = text_encoder_out.last_hidden_state
        qwen_hidden = qwen_hidden.to(model_dtype)
    return qwen_hidden, qwen_mask, t5_ids, t5_weights


def build_condition(
    transformer: "ModelMixin",
    *,
    qwen_hidden: torch.Tensor,
    qwen_mask: torch.Tensor | None,
    t5_ids: torch.Tensor,
    t5_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the LLM adapter and pad the conditioning sequence to 512 tokens."""
    with torch.inference_mode():
        cond = transformer.preprocess_text_embeds(
            qwen_hidden,
            t5_ids,
            t5xxl_weights=t5_weights,
            source_attention_mask=qwen_mask,
        )
    # The semantic frontend is expected to fit prompts before this point; this
    # slice is a final contract guard so the DiT never receives >512 positions.
    cond = cond[:, :_CONDITIONING_MAX_LENGTH]
    pad_len = max(0, _CONDITIONING_MAX_LENGTH - cond.shape[1])
    if pad_len > 0:
        cond = torch.nn.functional.pad(cond, (0, 0, 0, pad_len))
    return cond
