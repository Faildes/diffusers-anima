"""Anima prompt tokenization and text conditioning utilities.

The Qwen source-memory length and Anima target-conditioning length are
intentionally separate.  Qwen may read a long prompt while the existing Anima
LLM adapter keeps its 512-position T5/query side.  Alternate Qwen families can
be representation-aligned with :class:`AnimaTextEncoderBridge` before entering
the adapter.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers import ModelMixin
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )
    from .text_encoder_bridge import AnimaTextEncoderBridge
    from .text_encoder_conditioner import AnimaTextEncoderConditioner

import torch

_QWEN3_DEFAULT_PAD_TOKEN_ID: int = 151643
_CONDITIONING_MAX_LENGTH: int = 512
_T5_DEFAULT_QUERY_MAX_LENGTH: int = 224
_T5_QUERY_STRATEGIES = {"head", "uniform", "group_aware"}


def _uniform_indices(length: int, target: int) -> list[int]:
    """Return monotonically increasing indices spanning the entire sequence."""
    length = int(length)
    target = int(target)
    if target <= 0 or length <= 0:
        return []
    if length <= target:
        return list(range(length))
    if target == 1:
        return [0]
    # Integer interpolation includes both endpoints and stays unique when
    # length >= target.
    return [(i * (length - 1)) // (target - 1) for i in range(target)]


def _select_query_items(items: list[Any], *, target: int, strategy: str) -> list[Any]:
    if len(items) <= int(target):
        return list(items)
    if strategy == "head":
        return list(items[: int(target)])
    if strategy in {"uniform", "group_aware"}:
        # Plain-text calls have no PromptPlan groups. ``group_aware`` therefore
        # falls back to full-range uniform anchors while the structured path
        # below preserves every semantic/subject group explicitly.
        return [items[i] for i in _uniform_indices(len(items), int(target))]
    raise ValueError(f"Unsupported T5 query strategy: {strategy!r}")


def _select_group_aware_query_indices(
    offsets: list[tuple[int, int]],
    spans: Any,
    *,
    target: int,
    subject_group_ids: Any = (),
) -> list[int]:
    """Choose bounded T5 *query anchors* without truncating Qwen memory.

    The Qwen sequence remains complete and is the semantic memory.  T5 tokens
    merely provide adapter queries, and dense 256..512-query occupancy is an
    empirically unstable Anima regime.  Preserve coverage by reserving anchors
    for every PromptPlan group (and extra endpoints for subject groups), then
    fill the remaining budget uniformly over the complete T5 stream.
    """
    length = len(offsets)
    target = max(1, int(target))
    if length <= target:
        return list(range(length))

    groups = _groups_from_spans(offsets, tuple(spans or ()))
    try:
        subject_set = {int(x) for x in subject_group_ids or ()}
    except (TypeError, ValueError):
        subject_set = set()

    by_group: dict[int, list[int]] = {}
    for idx, gid in enumerate(groups):
        by_group.setdefault(int(gid), []).append(idx)

    chosen: set[int] = {0, length - 1}
    # Every semantic group gets at least one centre anchor while budget allows.
    # Subject groups get both ends first, preserving ownership boundaries.
    ordered_groups = list(by_group.items())
    mandatory: list[int] = []
    for gid, indices in ordered_groups:
        if not indices:
            continue
        if gid in subject_set:
            mandatory.extend((indices[0], indices[-1]))
        mandatory.append(indices[len(indices) // 2])
    for idx in mandatory:
        if len(chosen) >= target:
            break
        chosen.add(int(idx))

    # Fill all remaining capacity across the whole prompt. This is deterministic
    # and never tail-truncates long natural-language/tag hybrids.
    if len(chosen) < target:
        for idx in _uniform_indices(length, target):
            chosen.add(int(idx))
            if len(chosen) >= target:
                break
    if len(chosen) < target:
        for idx in range(length):
            chosen.add(idx)
            if len(chosen) >= target:
                break
    return sorted(chosen)[:target]


class AnimaPromptTokenizer:
    """Prompt tokenizer for Anima dual-encoder conditioning.

    ``qwen_source_max_length=None`` means the Qwen source is not truncated by
    Anima.  This is deliberate: the source encoder acts as semantic memory.
    The T5/query side uses a conservative 224-query budget by default because
    dense ~256..512 target occupancy is empirically unstable in stock Anima.
    Qwen source memory remains independent and may be arbitrarily longer.
    """

    def __init__(
        self,
        qwen_tokenizer: "PreTrainedTokenizer" | "PreTrainedTokenizerFast",
        t5_tokenizer: "PreTrainedTokenizer" | "PreTrainedTokenizerFast",
        *,
        qwen_source_max_length: int | None = None,
        t5_query_strategy: str = "group_aware",
        t5_query_max_length: int = _T5_DEFAULT_QUERY_MAX_LENGTH,
    ) -> None:
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.qwen_source_max_length = (
            None if qwen_source_max_length is None else int(qwen_source_max_length)
        )
        if str(t5_query_strategy) not in _T5_QUERY_STRATEGIES:
            raise ValueError(
                f"t5_query_strategy must be one of {sorted(_T5_QUERY_STRATEGIES)}, got {t5_query_strategy!r}"
            )
        self.t5_query_strategy = str(t5_query_strategy)
        self.t5_query_max_length = int(t5_query_max_length)
        if not 16 <= self.t5_query_max_length <= _CONDITIONING_MAX_LENGTH:
            raise ValueError("t5_query_max_length must be in [16, 512]")

    def tokenize_with_weights(
        self, text: str
    ) -> dict[str, list[list[tuple[int, float]]]]:
        qwen_kwargs: dict[str, Any] = {
            "add_special_tokens": False,
            "truncation": self.qwen_source_max_length is not None,
            "return_tensors": "pt",
        }
        if self.qwen_source_max_length is not None:
            qwen_kwargs["max_length"] = int(self.qwen_source_max_length)
        qwen_ids = self.qwen_tokenizer([text], **qwen_kwargs).input_ids[0].tolist()

        # The adapter target/query side is fixed at 512 positions, but do not
        # blindly drop the tail of a long prompt.  Tokenize the full T5 stream
        # and, when needed, choose query anchors spanning the complete text.
        # Qwen still carries every source token; T5 is only the bounded query set.
        t5_all_ids = (
            self.t5_tokenizer(
                [text],
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
            .input_ids[0]
            .tolist()
        )
        t5_ids = _select_query_items(
            [int(x) for x in t5_all_ids],
            target=max(1, int(self.t5_query_max_length) - 1),
            strategy=self.t5_query_strategy,
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
        t5_ids = t5_ids[: int(self.t5_query_max_length)]
        if t5_ids and int(t5_ids[-1]) != int(t5_eos):
            t5_ids[-1] = int(t5_eos)

        qwen_pairs = [[(int(token_id), 1.0) for token_id in qwen_ids]]
        return {
            "qwen": qwen_pairs,
            # Backward-compatible key used by existing sd_embed integrations.
            "qwen3_06b": qwen_pairs,
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


def apply_text_encoder_bridge(
    bridge: "AnimaTextEncoderBridge | None", hidden_states: torch.Tensor
) -> torch.Tensor:
    if bridge is None:
        return hidden_states
    return bridge.apply(hidden_states)


def apply_text_encoder_conditioning(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    conditioner: "AnimaTextEncoderConditioner | None" = None,
    bridge: "AnimaTextEncoderBridge | None" = None,
    group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the final encoder head when present, otherwise the legacy bridge."""
    if conditioner is not None:
        return conditioner.build_memory(hidden_states, attention_mask, group_ids=group_ids)
    return apply_text_encoder_bridge(bridge, hidden_states), attention_mask


def prepare_condition_inputs(
    prompt_tokenizer: AnimaPromptTokenizer,
    text_encoder: "PreTrainedModel",
    prompt: list[str],
    *,
    execution_device: str,
    model_dtype: torch.dtype,
    bridge: "AnimaTextEncoderBridge | None" = None,
    conditioner: "AnimaTextEncoderConditioner | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize/encode prompts while preserving long Qwen source memory.

    Returns ``(qwen_hidden, qwen_mask, t5_ids, t5_mask, t5_weights)``.
    Source and target masks are kept all the way into the Anima LLM adapter so
    padded source tokens never become cross-attention memory.
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
        qwen_entries = tokenized.get("qwen", tokenized.get("qwen3_06b"))
        if qwen_entries is None:
            raise RuntimeError("Prompt tokenizer did not return Qwen token IDs.")
        qwen_token_ids, _ = _extract_ids_and_weights(qwen_entries[0])
        t5_token_ids, t5_token_weights = _extract_ids_and_weights(tokenized["t5xxl"][0])

        if len(qwen_token_ids) == 0:
            qwen_token_ids = [int(qwen_pad)]
        if len(t5_token_ids) == 0:
            t5_token_ids = [1]
            t5_token_weights = [1.0]
        if len(t5_token_ids) > _CONDITIONING_MAX_LENGTH:
            raise RuntimeError(
                f"T5 target/query length exceeded {_CONDITIONING_MAX_LENGTH}: {len(t5_token_ids)}"
            )

        qwen_token_batches.append(qwen_token_ids)
        t5_token_batches.append(t5_token_ids)
        t5_weight_batches.append(t5_token_weights)
        max_qwen_len = max(max_qwen_len, len(qwen_token_ids))
        max_t5_len = max(max_t5_len, len(t5_token_ids))

    batch_size = len(prompt)
    qwen_ids = torch.full(
        (batch_size, max_qwen_len), int(qwen_pad), dtype=torch.long, device=execution_device
    )
    qwen_mask = torch.zeros(
        (batch_size, max_qwen_len), dtype=torch.long, device=execution_device
    )
    t5_ids = torch.full(
        (batch_size, max_t5_len), int(t5_pad), dtype=torch.int32, device=execution_device
    )
    t5_mask = torch.zeros(
        (batch_size, max_t5_len), dtype=torch.long, device=execution_device
    )
    t5_weights = torch.zeros(
        (batch_size, max_t5_len, 1), dtype=torch.float32, device=execution_device
    )

    for idx, (qwen_ids_item, t5_ids_item, t5_weights_item) in enumerate(
        zip(qwen_token_batches, t5_token_batches, t5_weight_batches, strict=True)
    ):
        q_len = len(qwen_ids_item)
        t_len = len(t5_ids_item)
        qwen_ids[idx, :q_len] = torch.tensor(qwen_ids_item, dtype=torch.long, device=execution_device)
        qwen_mask[idx, :q_len] = 1
        t5_ids[idx, :t_len] = torch.tensor(t5_ids_item, dtype=torch.int32, device=execution_device)
        t5_mask[idx, :t_len] = 1
        t5_weights[idx, :t_len, 0] = torch.tensor(
            t5_weights_item, dtype=torch.float32, device=execution_device
        )

    with torch.inference_mode():
        # Qwen3.5 linear-attention cache objects are unnecessary for feature
        # extraction and may retain large long-context states.
        try:
            text_encoder_out = text_encoder(
                input_ids=qwen_ids, attention_mask=qwen_mask, use_cache=False
            )
        except TypeError:
            text_encoder_out = text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask)
        if isinstance(text_encoder_out, tuple):
            qwen_hidden = text_encoder_out[0]
        else:
            qwen_hidden = text_encoder_out.last_hidden_state
        qwen_hidden = qwen_hidden.to(model_dtype)
        qwen_hidden, qwen_mask = apply_text_encoder_conditioning(
            qwen_hidden, qwen_mask, conditioner=conditioner, bridge=bridge
        )
    return qwen_hidden, qwen_mask, t5_ids, t5_mask, t5_weights


def build_condition(
    transformer: "ModelMixin",
    *,
    qwen_hidden: torch.Tensor,
    qwen_mask: torch.Tensor | None,
    t5_ids: torch.Tensor,
    t5_mask: torch.Tensor | None,
    t5_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the existing Anima LLM adapter with long source / 512 target."""
    with torch.inference_mode():
        cond = transformer.preprocess_text_embeds(
            qwen_hidden,
            t5_ids,
            t5xxl_weights=t5_weights,
            source_attention_mask=qwen_mask,
            target_attention_mask=t5_mask,
        )
    # Target-side contract is always 512 regardless of Qwen source length.
    cond = cond[:, :_CONDITIONING_MAX_LENGTH]
    pad_len = max(0, _CONDITIONING_MAX_LENGTH - cond.shape[1])
    if pad_len > 0:
        cond = torch.nn.functional.pad(cond, (0, 0, 0, pad_len))
    return cond


def _flatten_tokenizer_field(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return list(value or [])


def _tokenize_with_offsets(
    tokenizer: Any,
    text: str,
    *,
    max_length: int | None,
    add_eos: bool,
    query_strategy: str = "head",
) -> tuple[list[int], list[tuple[int, int]]]:
    # For uniform T5 query selection we must see the complete token stream
    # before selecting bounded query anchors. Qwen uses head truncation only when
    # the caller explicitly configures a source cap.
    pretruncate = max_length is not None and query_strategy == "head"
    kwargs: dict[str, Any] = {
        "add_special_tokens": False,
        "truncation": pretruncate,
        "return_offsets_mapping": True,
        "return_tensors": None,
    }
    if pretruncate:
        kwargs["max_length"] = int(max_length)
    try:
        encoded = tokenizer(text, verbose=False, **kwargs)
    except TypeError:
        encoded = tokenizer(text, **kwargs)
    ids = _flatten_tokenizer_field(
        getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else [])
    )
    offsets = _flatten_tokenizer_field(
        getattr(encoded, "offset_mapping", encoded.get("offset_mapping") if isinstance(encoded, dict) else [])
    )
    offsets = [tuple(map(int, item)) for item in offsets]
    ids = [int(x) for x in ids]
    if len(ids) != len(offsets):
        raise RuntimeError(
            "Prompt-plan encoding requires a fast tokenizer with offset_mapping support."
        )
    if max_length is not None and len(ids) > int(max_length):
        selected_indices = (
            list(range(int(max_length)))
            if query_strategy == "head"
            else _uniform_indices(len(ids), int(max_length))
        )
        if query_strategy not in _T5_QUERY_STRATEGIES:
            raise ValueError(f"Unsupported T5 query strategy: {query_strategy!r}")
        ids = [ids[i] for i in selected_indices]
        offsets = [offsets[i] for i in selected_indices]
    if add_eos:
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = 1
        if not ids or int(ids[-1]) != int(eos):
            ids.append(int(eos))
            offsets.append((len(text), len(text)))
        if len(ids) > _CONDITIONING_MAX_LENGTH:
            ids = ids[:_CONDITIONING_MAX_LENGTH]
            offsets = offsets[:_CONDITIONING_MAX_LENGTH]
            ids[-1] = int(eos)
            offsets[-1] = (len(text), len(text))
    return ids, offsets


def _factors_from_spans(
    offsets: list[tuple[int, int]],
    spans: Any,
    *,
    attribute: str,
) -> list[float]:
    factors: list[float] = []
    span_list = list(spans or ())
    for start, end in offsets:
        if end <= start:
            factors.append(1.0)
            continue
        weighted = 0.0
        covered = 0
        for span in span_list:
            overlap = max(0, min(end, int(span.end)) - max(start, int(span.start)))
            if overlap <= 0:
                continue
            weighted += float(getattr(span, attribute)) * overlap
            covered += overlap
        if covered <= 0:
            factors.append(1.0)
        else:
            # Uncovered characters keep factor 1.0.
            uncovered = max(0, (end - start) - covered)
            factors.append((weighted + uncovered) / float(end - start))
    return factors


def _groups_from_spans(offsets: list[tuple[int, int]], spans: tuple[Any, ...]) -> list[int]:
    """Assign each source token to the PromptPlan group with maximum character overlap."""
    groups: list[int] = []
    last_group = 0
    for start, end in offsets:
        best_group = last_group
        best_overlap = 0
        for span in spans:
            overlap = max(0, min(end, int(span.end)) - max(start, int(span.start)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_group = int(getattr(span, "group", 0))
        if best_overlap > 0:
            last_group = best_group
        groups.append(best_group)
    return groups


def _apply_qwen_plan_factors(
    hidden_states: torch.Tensor,
    factors: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Final v3 encoders may append semantic expansion slots after the original
    # Qwen tokens. Prompt-plan weights apply to visible text tokens only; the
    # summary slots deliberately receive neutral factor 1.0.
    if factors.shape[1] < hidden_states.shape[1]:
        pad = hidden_states.shape[1] - factors.shape[1]
        factors = torch.nn.functional.pad(factors, (0, pad), value=1.0)
    elif factors.shape[1] > hidden_states.shape[1]:
        factors = factors[:, : hidden_states.shape[1]]
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    factors = factors.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    factors = torch.where(mask > 0, factors, torch.ones_like(factors))
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    anchor = (hidden_states * mask).sum(dim=1, keepdim=True) / denom
    return anchor + (hidden_states - anchor) * factors


def _native_subject_group_ids(
    plans: list[Any],
    qwen_groups: torch.Tensor,
) -> torch.Tensor | None:
    """Map semantic PromptPlan groups to dense native subject slots.

    sd_embed can use groups for AND/BREAK/semantic weighting that are not
    people.  Only metadata-declared ``subject_group_ids`` become ownership
    slots; everything else remains global slot 0.
    """
    mapped = torch.zeros_like(qwen_groups)
    any_subject = False
    for b, plan in enumerate(plans):
        raw_ids = getattr(plan, "metadata", {}).get("subject_group_ids", ())
        try:
            subject_ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            subject_ids = []
        for slot, original_group in enumerate(subject_ids, start=1):
            mapped[b] = torch.where(
                qwen_groups[b] == int(original_group),
                torch.full_like(mapped[b], int(slot)),
                mapped[b],
            )
            any_subject = True
    return mapped if any_subject else None


def _native_separator_ids(
    plans: list[Any],
    qwen_groups: torch.Tensor,
) -> torch.Tensor | None:
    """Mark the first source token after each AND/BREAK/semicolon boundary.

    v5 preserved group ids but the bridge-free native head only consumed
    *subject* groups. v6 keeps all separator boundaries as a separate, weak
    signal so style/background groups are never mislabelled as people.
    """
    type_ids = {"semicolon": 1, "and": 2, "break": 3}
    out = torch.zeros_like(qwen_groups, dtype=torch.long)
    any_boundary = False
    for b, plan in enumerate(plans):
        metadata = getattr(plan, "metadata", {}) or {}
        raw = metadata.get("group_separator_types", {}) or {}
        try:
            mapping = {int(k): str(v).casefold() for k, v in dict(raw).items()}
        except (TypeError, ValueError):
            mapping = {}
        for gid, kind in mapping.items():
            sep_id = int(type_ids.get(kind, 0))
            if gid <= 0 or sep_id <= 0:
                continue
            positions = torch.nonzero(qwen_groups[b] == int(gid), as_tuple=False).flatten()
            if positions.numel() <= 0:
                continue
            out[b, int(positions[0].item())] = sep_id
            any_boundary = True
    return out if any_boundary else None


def _native_subject_counts(plans: list[Any], *, device: str) -> torch.Tensor | None:
    """Return exact subject counts declared by structured prompt metadata.

    A value of zero means "unspecified" and therefore contributes only the
    learned neutral count slot.  Keeping this signal out of free-text parsing
    in diffusers-anima makes sd_embed the single owner of prompt semantics.
    """
    values: list[int] = []
    has_declared = False
    for plan in plans:
        raw = getattr(plan, "metadata", {}).get("subject_count", 0)
        try:
            value = max(0, int(raw or 0))
        except (TypeError, ValueError):
            value = 0
        has_declared = has_declared or value > 0
        values.append(value)
    if not has_declared:
        return None
    return torch.tensor(values, dtype=torch.long, device=device)


def prepare_condition_inputs_from_plans(
    prompt_tokenizer: AnimaPromptTokenizer,
    text_encoder: "PreTrainedModel",
    plans: list[Any],
    *,
    execution_device: str,
    model_dtype: torch.dtype,
    bridge: "AnimaTextEncoderBridge | None" = None,
    conditioner: "AnimaTextEncoderConditioner | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode structured prompt plans in one source-memory / one-adapter pass."""
    from .prompt_plan import coerce_prompt_plans

    normalized = coerce_prompt_plans(plans)
    if not normalized:
        raise ValueError("prompt plan batch must not be empty")

    qwen_pad = prompt_tokenizer.qwen_tokenizer.pad_token_id
    if qwen_pad is None:
        qwen_pad = prompt_tokenizer.qwen_tokenizer.eos_token_id
    if qwen_pad is None:
        qwen_pad = _QWEN3_DEFAULT_PAD_TOKEN_ID
    t5_pad = prompt_tokenizer.t5_tokenizer.pad_token_id
    if t5_pad is None:
        t5_pad = 0

    q_ids_batch: list[list[int]] = []
    q_factor_batch: list[list[float]] = []
    q_group_batch: list[list[int]] = []
    t_ids_batch: list[list[int]] = []
    t_factor_batch: list[list[float]] = []
    for plan in normalized:
        q_ids, q_offsets = _tokenize_with_offsets(
            prompt_tokenizer.qwen_tokenizer,
            plan.text,
            max_length=prompt_tokenizer.qwen_source_max_length,
            add_eos=False,
            query_strategy="head",
        )
        if not q_ids:
            q_ids = [int(qwen_pad)]
            q_offsets = [(0, 0)]
        t_all_ids, t_all_offsets = _tokenize_with_offsets(
            prompt_tokenizer.t5_tokenizer,
            plan.text,
            max_length=None,
            add_eos=False,
            query_strategy="head",
        )
        t_content_budget = max(1, int(prompt_tokenizer.t5_query_max_length) - 1)
        if len(t_all_ids) > t_content_budget:
            if prompt_tokenizer.t5_query_strategy == "head":
                t_selected = list(range(t_content_budget))
            elif prompt_tokenizer.t5_query_strategy == "uniform":
                t_selected = _uniform_indices(len(t_all_ids), t_content_budget)
            else:
                t_selected = _select_group_aware_query_indices(
                    t_all_offsets,
                    plan.spans,
                    target=t_content_budget,
                    subject_group_ids=getattr(plan, "metadata", {}).get("subject_group_ids", ()),
                )
            t_ids = [t_all_ids[i] for i in t_selected]
            t_offsets = [t_all_offsets[i] for i in t_selected]
        else:
            t_ids = list(t_all_ids)
            t_offsets = list(t_all_offsets)
        t5_eos = prompt_tokenizer.t5_tokenizer.eos_token_id
        if t5_eos is None:
            t5_eos = 1
        if not t_ids or int(t_ids[-1]) != int(t5_eos):
            t_ids.append(int(t5_eos))
            t_offsets.append((len(plan.text), len(plan.text)))
        if len(t_ids) > int(prompt_tokenizer.t5_query_max_length):
            t_ids = t_ids[: int(prompt_tokenizer.t5_query_max_length)]
            t_offsets = t_offsets[: int(prompt_tokenizer.t5_query_max_length)]
            t_ids[-1] = int(t5_eos)
            t_offsets[-1] = (len(plan.text), len(plan.text))
        q_ids_batch.append(q_ids)
        q_factor_batch.append(_factors_from_spans(q_offsets, plan.spans, attribute="qwen_factor"))
        q_group_batch.append(_groups_from_spans(q_offsets, plan.spans))
        t_ids_batch.append(t_ids)
        t_factor_batch.append(_factors_from_spans(t_offsets, plan.spans, attribute="t5_factor"))

    bsz = len(normalized)
    qmax = max(len(x) for x in q_ids_batch)
    tmax = max(len(x) for x in t_ids_batch)
    qwen_ids = torch.full((bsz, qmax), int(qwen_pad), dtype=torch.long, device=execution_device)
    qwen_mask = torch.zeros((bsz, qmax), dtype=torch.long, device=execution_device)
    qwen_factors = torch.ones((bsz, qmax), dtype=torch.float32, device=execution_device)
    qwen_groups = torch.zeros((bsz, qmax), dtype=torch.long, device=execution_device)
    t5_ids = torch.full((bsz, tmax), int(t5_pad), dtype=torch.int32, device=execution_device)
    t5_mask = torch.zeros((bsz, tmax), dtype=torch.long, device=execution_device)
    t5_weights = torch.zeros((bsz, tmax, 1), dtype=torch.float32, device=execution_device)

    for i, (qids, qf, qg, tids, tf) in enumerate(
        zip(q_ids_batch, q_factor_batch, q_group_batch, t_ids_batch, t_factor_batch, strict=True)
    ):
        qlen = len(qids)
        tlen = len(tids)
        qwen_ids[i, :qlen] = torch.tensor(qids, dtype=torch.long, device=execution_device)
        qwen_mask[i, :qlen] = 1
        qwen_factors[i, :qlen] = torch.tensor(qf, dtype=torch.float32, device=execution_device)
        qwen_groups[i, :qlen] = torch.tensor(qg, dtype=torch.long, device=execution_device)
        t5_ids[i, :tlen] = torch.tensor(tids, dtype=torch.int32, device=execution_device)
        t5_mask[i, :tlen] = 1
        t5_weights[i, :tlen, 0] = torch.tensor(tf, dtype=torch.float32, device=execution_device)

    with torch.inference_mode():
        native_kwargs: dict[str, Any] = {}
        if bool(getattr(text_encoder, "_anima_native_encoder", False)):
            native_groups = _native_subject_group_ids(normalized, qwen_groups)
            if native_groups is not None:
                native_kwargs["anima_group_ids"] = native_groups
            separator_ids = _native_separator_ids(normalized, qwen_groups)
            if separator_ids is not None:
                native_kwargs["anima_separator_ids"] = separator_ids
            subject_counts = _native_subject_counts(normalized, device=execution_device)
            if subject_counts is not None:
                native_kwargs["anima_subject_counts"] = subject_counts
        try:
            out = text_encoder(
                input_ids=qwen_ids,
                attention_mask=qwen_mask,
                use_cache=False,
                **native_kwargs,
            )
        except TypeError:
            # Non-native third-party encoders can have a narrower forward
            # signature; never drop structured controls from a native encoder.
            if native_kwargs:
                raise
            out = text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask)
        qwen_hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        qwen_hidden = qwen_hidden.to(dtype=model_dtype)
        qwen_hidden, qwen_mask = apply_text_encoder_conditioning(
            qwen_hidden, qwen_mask, conditioner=conditioner, bridge=bridge, group_ids=qwen_groups
        )
        qwen_hidden = _apply_qwen_plan_factors(qwen_hidden, qwen_factors, qwen_mask)
    return qwen_hidden, qwen_mask, t5_ids, t5_mask, t5_weights
