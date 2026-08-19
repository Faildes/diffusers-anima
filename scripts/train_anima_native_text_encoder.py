#!/usr/bin/env python3
"""Train a bridge-free Anima-native encoder from Qwen3.5-0.8B-Base.

The final artifact is a single ``anima_native_text_encoder_v1`` safetensors file
containing the Qwen3.5 text backbone plus an integrated Anima-native head.  The
runtime does not load or apply a bridge.

Training intentionally uses two references with different jobs:

* Qwen3-0.6B-Base: teaches the representation/distribution that the frozen Anima
  LLM adapter can consume.
* Qwen3.5-0.8B-Base itself: supplies semantic geometry.  We preserve that
  geometry rather than forcing the student to become a clone of the 0.6B model.

A historical v2 bridge profile is optional.  When supplied it is used only as a
bootstrap initialiser/token teacher.  It is not stored as a runtime bridge in
the final native checkpoint.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from native_training_corpus import (  # noqa: E402
    ANIMA_COMPAT_ANCHOR_PROMPTS,
    NativePromptGroup,
    build_training_groups,
    corpus_sha256,
    default_sampling_bucket_weights,
    group_sampling_bucket,
    preview_training_groups,
    read_prompt_lines,
    split_validation_lines,
)
from diffusers_anima.pipelines.anima.anima_native_text_encoder import (  # noqa: E402
    _NATIVE_ENCODER_FORMAT_V1,
    _NATIVE_ENCODER_KIND,
    AnimaNativeHeadConfig,
    AnimaNativeQwen35Encoder,
    AnimaNativeQwen35Head,
    native_head_metadata,
)
from diffusers_anima.pipelines.anima.loading import load_text_encoder_single_file  # noqa: E402
from diffusers_anima.pipelines.anima.text_encoder_bridge import (  # noqa: E402
    AnimaTextEncoderBridge,
    encoder_config_fingerprint,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class NativeEncoderTrainingConfig:
    source_model: str | Path | None
    reference_model: str | Path
    prompts: str | Path
    output: str | Path
    source_tokenizer: str | Path = "Qwen/Qwen3.5-0.8B-Base"
    reference_tokenizer: str | Path = "Qwen/Qwen3-0.6B-Base"
    resume_native: str | Path | None = None
    bootstrap_bridge_profile: str | Path | None = None
    bootstrap_initialization_strength: float = 1.0
    device: str = "auto"
    reference_device: str = "auto"
    dtype: str = "bfloat16"
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    max_steps: int = 0
    # v3 decouples optimiser work from corpus size. max_steps still overrides
    # this for explicit experiments; otherwise fixed_budget_steps is used.
    fixed_budget_steps: int = 1000
    balanced_sampling: bool = True
    validation_size: int = 192
    validation_anchor_size: int = 32
    validation_every: int = 100
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.002
    restore_best_checkpoint: bool = True
    seed: int = 3571
    head_lr: float = 1.0e-4
    backbone_lr: float = 5.0e-6
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    train_last_n_layers: int = 0
    gradient_checkpointing: bool = True
    native_intermediate_size: int = 1536
    native_layer_indices: tuple[int, ...] | None = None
    native_final_layer_logit_bias: float = 3.0
    native_max_subject_slots: int = 16
    native_binding_rms_min_ratio: float = 0.92
    native_binding_rms_max_ratio: float = 1.08
    pair_fraction: float = 0.35
    binding_pairs: int = 256
    # The 0.6B model is an Anima-compatibility anchor, not the semantic teacher.
    # Running it on a deterministic subset keeps that anchor while avoiding a
    # second LLM forward on every microbatch.
    reference_batch_fraction: float = 0.50
    # Multi-person / exact-count prompts benefit disproportionately from the
    # stable Anima behaviour of Qwen3-0.6B. Use the 0.6B anchor more often for
    # those prompts without letting it dominate normal single-subject prompts.
    multi_person_reference_fraction: float = 1.00
    reference_max_length: int = 256
    fused_adamw: bool = True
    allow_tf32: bool = True
    # Losses.  Compatibility is important, but source geometry deliberately
    # remains substantial so 0.8B-only distinctions are not collapsed into 0.6B.
    anima_compat_weight: float = 1.00
    source_geometry_weight: float = 0.75
    token_geometry_weight: float = 0.50
    knowledge_gain_weight: float = 0.75
    distribution_weight: float = 0.20
    channel_distribution_weight: float = 0.06
    binding_geometry_weight: float = 0.35
    multi_person_compat_weight: float = 0.40
    count_anchor_weight: float = 0.45
    subject_slot_anchor_weight: float = 0.40
    ownership_anchor_weight: float = 0.30
    multi_person_reference_boost: float = 0.50
    layer_prior_weight: float = 0.01
    bootstrap_token_weight: float = 0.50
    bootstrap_final_fraction: float = 0.25
    reference_relax_for_gain: float = 0.50
    log_every: int = 10

    def normalized(self) -> "NativeEncoderTrainingConfig":
        data = asdict(self)
        data["source_model"] = None if self.source_model is None else str(self.source_model)
        data["reference_model"] = str(self.reference_model)
        data["prompts"] = str(self.prompts)
        data["output"] = str(self.output)
        data["source_tokenizer"] = str(self.source_tokenizer)
        data["reference_tokenizer"] = str(self.reference_tokenizer)
        data["resume_native"] = None if self.resume_native is None else str(self.resume_native)
        data["bootstrap_bridge_profile"] = (
            None if self.bootstrap_bridge_profile is None else str(self.bootstrap_bridge_profile)
        )
        if self.native_layer_indices is not None:
            data["native_layer_indices"] = tuple(int(x) for x in self.native_layer_indices)
        return NativeEncoderTrainingConfig(**data)


@dataclass(frozen=True)
class NativeEncoderTrainingResult:
    output: Path
    steps: int
    elapsed_seconds: float
    corpus_lines: int
    training_rows: int
    metadata: Mapping[str, str]
    final_loss: float
    best_step: int = 0
    validation_score: float = float("nan")
    early_stopped: bool = False

    def summary(self) -> str:
        return (
            f"native_encoder: {self.output}\n"
            f"steps={self.steps}; best_step={self.best_step}; corpus={self.corpus_lines}; "
            f"rows={self.training_rows}; final_loss={self.final_loss:.6f}; "
            f"val={self.validation_score:.6f}; early_stopped={self.early_stopped}; "
            f"elapsed={self.elapsed_seconds:.1f}s"
        )


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(dict(payload))
        return
    stage = payload.pop("stage", "train")
    print(f"[anima-native:{stage}] " + " ".join(f"{k}={v}" for k, v in payload.items()))


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "--"
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def make_jupyter_progress_callback(*, history_size: int = 8) -> ProgressCallback:
    """Create a detailed Jupyter progress display with elapsed time and ETA.

    ETA is based on an EMA of completed optimiser-step durations.  Model loading
    and save stages have unknown totals, so they show elapsed time rather than a
    fabricated percentage.  The callback also shows GPU memory when available.
    """
    try:
        from IPython.display import HTML, display
    except Exception:
        return lambda payload: _emit(None, **payload)

    handle: dict[str, Any] = {"display": None}
    created = time.perf_counter()
    stage_started = created
    current_stage: str | None = None
    last_step_time: float | None = None
    last_step: int | None = None
    ema_step_seconds: float | None = None
    history: deque[str] = deque(maxlen=max(1, int(history_size)))

    def _gpu_stats() -> str:
        if not torch.cuda.is_available():
            return "GPU: unavailable"
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        name = torch.cuda.get_device_name(dev)
        return f"GPU {name}: {alloc:.2f} GiB alloc / {reserved:.2f} GiB reserved / {peak:.2f} GiB peak"

    def callback(payload: dict[str, Any]) -> None:
        nonlocal stage_started, current_stage, last_step_time, last_step, ema_step_seconds
        now = time.perf_counter()
        stage = str(payload.get("stage", "train"))
        if stage != current_stage:
            current_stage = stage
            stage_started = now
            last_step_time = None
            last_step = None
            ema_step_seconds = None

        step = payload.get("step")
        total = payload.get("total")
        step_i = int(step) if isinstance(step, (int, float)) else None
        total_i = int(total) if isinstance(total, (int, float)) and int(total) > 0 else None

        # Prefer timing supplied by the trainer; otherwise estimate from callback cadence.
        supplied_step_seconds = payload.get("step_seconds_ema")
        if supplied_step_seconds is not None:
            try:
                ema_step_seconds = float(supplied_step_seconds)
            except Exception:
                pass
        elif step_i is not None and last_step_time is not None and last_step is not None and step_i > last_step:
            sample = (now - last_step_time) / max(1, step_i - last_step)
            ema_step_seconds = sample if ema_step_seconds is None else (0.85 * ema_step_seconds + 0.15 * sample)
        if step_i is not None:
            last_step = step_i
            last_step_time = now

        pct: float | None = None
        eta: float | None = None
        if step_i is not None and total_i is not None:
            pct = 100.0 * max(0, min(step_i, total_i)) / total_i
            supplied_eta = payload.get("eta_seconds")
            if supplied_eta is not None:
                try:
                    eta = max(0.0, float(supplied_eta))
                except Exception:
                    eta = None
            elif ema_step_seconds is not None:
                eta = max(0, total_i - step_i) * ema_step_seconds

        total_elapsed = now - created
        stage_elapsed = now - stage_started
        loss = payload.get("loss")
        epoch = payload.get("epoch")
        lr = payload.get("lr")
        micro = payload.get("micro_step")
        micro_total = payload.get("micro_total")

        title = f"<b>{stage}</b>"
        if step_i is not None and total_i is not None:
            title += f" &nbsp; step {step_i:,}/{total_i:,}"
            if pct is not None:
                title += f" &nbsp; <b>{pct:5.1f}%</b>"
        if epoch is not None:
            title += f" &nbsp; epoch {epoch}"
        if micro is not None and micro_total is not None:
            title += f" &nbsp; micro {micro}/{micro_total}"

        stats = [
            f"Elapsed: <b>{_format_duration(total_elapsed)}</b>",
            f"Stage: {_format_duration(stage_elapsed)}",
            f"ETA: <b>{_format_duration(eta)}</b>",
        ]
        if ema_step_seconds is not None and ema_step_seconds > 0:
            stats.append(f"EMA: {ema_step_seconds:.2f}s/step ({1.0/ema_step_seconds:.3f} step/s)")
        if loss is not None:
            try:
                stats.append(f"loss={float(loss):.6f}")
            except Exception:
                stats.append(f"loss={loss}")
        if lr is not None:
            stats.append(f"lr={lr}")
        if payload.get("gain") is not None:
            stats.append(f"gain={payload.get('gain')}")

        loss_parts = []
        for key in ("compat", "source_geom", "token_geom", "knowledge", "distribution", "bootstrap"):
            if key in payload:
                try:
                    loss_parts.append(f"{key}={float(payload[key]):.4f}")
                except Exception:
                    loss_parts.append(f"{key}={payload[key]}")

        details = " ".join(
            f"{k}={v}" for k, v in payload.items()
            if k not in {
                "stage", "step", "total", "loss", "epoch", "lr", "gain",
                "eta_seconds", "elapsed_seconds", "stage_elapsed_seconds",
                "step_seconds", "step_seconds_ema", "micro_step", "micro_total",
                "compat", "source_geom", "token_geom", "knowledge", "distribution", "bootstrap",
            }
        )
        summary = f"{stage}"
        if step_i is not None and total_i is not None:
            summary += f" {step_i}/{total_i}"
        if loss is not None:
            summary += f" loss={loss}"
        history.append(summary)

        if pct is None:
            bar = (
                '<div style="height:10px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:7px">'
                '<div style="height:100%;width:35%;background:#4c8bf5;opacity:.55"></div></div>'
            )
        else:
            bar = (
                '<div style="height:10px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:7px">'
                f'<div style="height:100%;width:{pct:.2f}%;background:#4c8bf5"></div></div>'
            )

        history_html = "<br>".join(history)
        html = HTML(
            '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.45;'
            'border:1px solid #bbb;border-radius:10px;padding:10px 12px">'
            f'<div style="font-size:15px">{title}</div>{bar}'
            f'<div style="margin-top:8px">{" &nbsp; | &nbsp; ".join(stats)}</div>'
            f'<div style="margin-top:5px">{_gpu_stats()}</div>'
            + (f'<div style="margin-top:5px">{" &nbsp; ".join(loss_parts)}</div>' if loss_parts else "")
            + (f'<div style="margin-top:5px;color:#666">{details}</div>' if details else "")
            + f'<details style="margin-top:7px"><summary>Recent updates</summary>'
              f'<div style="margin-top:5px;color:#666">{history_html}</div></details></div>'
        )
        if handle["display"] is None:
            handle["display"] = display(html, display_id=True)
        else:
            handle["display"].update(html)

    return callback


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    normalized = str(name).lower()
    if device == "cpu":
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _masked_pool(hidden: torch.Tensor, mask: torch.Tensor, *, last_weight: float = 0.65) -> torch.Tensor:
    m = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (hidden * m).sum(dim=1) / denom
    last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    last = hidden[batch_idx, last_idx]
    w = max(0.0, min(1.0, float(last_weight)))
    return mean * (1.0 - w) + last * w


def _normalized_mse_each(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def norm(x: torch.Tensor) -> torch.Tensor:
        return x / x.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    diff = (norm(a.float()) - norm(b.float())).square()
    return diff.mean(dim=-1)


def _normalized_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _normalized_mse_each(a, b).mean()


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)


def _pairwise_geometry_loss(student: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if student.shape[0] < 2:
        return student.new_zeros(())
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(source.float(), dim=-1)
    gs = s @ s.T
    gt = t @ t.T
    n = gs.shape[0]
    keep = ~torch.eye(n, dtype=torch.bool, device=gs.device)
    return F.mse_loss(gs[keep], gt[keep])


def _uniform_token_sample(hidden: torch.Tensor, mask: torch.Tensor, max_tokens: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    samples: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for b in range(hidden.shape[0]):
        length = max(1, int(mask[b].sum().item()))
        take = min(length, int(max_tokens))
        if take == 1:
            idx = torch.zeros(1, device=hidden.device, dtype=torch.long)
        else:
            idx = torch.linspace(0, length - 1, steps=take, device=hidden.device).round().long()
        x = hidden[b, idx]
        if take < max_tokens:
            x = F.pad(x, (0, 0, 0, max_tokens - take))
        samples.append(x)
        valid = torch.zeros(max_tokens, device=hidden.device, dtype=torch.bool)
        valid[:take] = True
        valids.append(valid)
    return torch.stack(samples), torch.stack(valids)


def _token_geometry_loss(student: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    max_tokens = min(48, int(student.shape[1]))
    if max_tokens < 2:
        return student.new_zeros(())
    s, valid = _uniform_token_sample(student, mask, max_tokens=max_tokens)
    t, _ = _uniform_token_sample(source, mask, max_tokens=max_tokens)
    s = F.normalize((s - s.mean(dim=1, keepdim=True)).float(), dim=-1)
    t = F.normalize((t - t.mean(dim=1, keepdim=True)).float(), dim=-1)
    gs = torch.matmul(s, s.transpose(-1, -2))
    gt = torch.matmul(t, t.transpose(-1, -2))
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(max_tokens, dtype=torch.bool, device=student.device).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    if not bool(pair_mask.any()):
        return student.new_zeros(())
    return F.mse_loss(gs[pair_mask], gt[pair_mask])


def _distribution_loss(
    student: torch.Tensor,
    student_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    def stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rms = x.float().square().mean(dim=-1).add(1e-6).sqrt()
        m = mask.float()
        denom = m.sum(dim=1).clamp_min(1.0)
        mean = (rms * m).sum(dim=1) / denom
        centered = (rms - mean.unsqueeze(1)) * m
        std = (centered.square().sum(dim=1) / denom).add(1e-6).sqrt()
        return mean, std
    sm, ss = stats(student, student_mask)
    tm, ts = stats(target, target_mask)
    return F.smooth_l1_loss(torch.log(sm), torch.log(tm)) + 0.5 * F.smooth_l1_loss(torch.log(ss), torch.log(ts))


def _layer_prior_loss(layer_weights: torch.Tensor) -> torch.Tensor:
    k = int(layer_weights.shape[-1])
    if k <= 1:
        return layer_weights.new_zeros(())
    # Preserve a final-layer majority while making upper-middle knowledge
    # accessible.  This is only a very small regulariser.
    base = torch.arange(1, k + 1, device=layer_weights.device, dtype=torch.float32)
    prior = base.square()
    prior = prior / prior.sum()
    observed = layer_weights.float().mean(dim=(0, 1)).clamp_min(1e-6)
    return F.kl_div(observed.log(), prior, reduction="batchmean")


def _flatten_group_batch(groups: Sequence[NativePromptGroup]) -> tuple[list[str], list[tuple[int, int, str]]]:
    texts: list[str] = []
    pairs: list[tuple[int, int, str]] = []
    for group in groups:
        start = len(texts)
        texts.extend(group.texts)
        if len(group.texts) == 2:
            pairs.append((start, start + 1, group.category))
    return texts, pairs


def _iter_group_batches(
    groups: Sequence[NativePromptGroup],
    *,
    row_batch_size: int,
    rng: random.Random,
) -> Iterable[list[NativePromptGroup]]:
    order = list(groups)
    rng.shuffle(order)
    batch: list[NativePromptGroup] = []
    rows = 0
    for group in order:
        needed = len(group.texts)
        if batch and rows + needed > row_batch_size:
            yield batch
            batch = []
            rows = 0
        batch.append(group)
        rows += needed
    if batch:
        yield batch


def _iter_balanced_group_batches(
    groups: Sequence[NativePromptGroup],
    *,
    row_batch_size: int,
    rng: random.Random,
    bucket_weights: Mapping[str, float] | None = None,
    max_batches: int | None = None,
) -> Iterable[list[NativePromptGroup]]:
    """Yield an infinite stratified stream whose mix is corpus-size invariant.

    Each bucket is shuffled into a deck and consumed without replacement until
    exhausted.  Choosing the next deck uses fixed target weights, never the raw
    number of rows in that bucket.  Adding 100k similar calibration lines thus
    increases diversity inside a deck but not its optimiser pressure.
    """
    pools: dict[str, list[NativePromptGroup]] = {}
    for group in groups:
        pools.setdefault(group_sampling_bucket(group), []).append(group)
    if not pools:
        raise ValueError("No training groups available for balanced sampling")
    weights = dict(default_sampling_bucket_weights())
    if bucket_weights:
        weights.update({str(k): max(0.0, float(v)) for k, v in bucket_weights.items()})
    active = [bucket for bucket in weights if pools.get(bucket) and weights.get(bucket, 0.0) > 0.0]
    for bucket in pools:
        if bucket not in active and bucket not in weights:
            active.append(bucket)
            weights[bucket] = 0.02
    if not active:
        active = sorted(pools)
        weights = {bucket: 1.0 for bucket in active}

    decks: dict[str, list[NativePromptGroup]] = {}
    positions: dict[str, int] = {}

    def refill(bucket: str) -> None:
        deck = list(pools[bucket])
        rng.shuffle(deck)
        decks[bucket] = deck
        positions[bucket] = 0

    def draw(bucket: str) -> NativePromptGroup:
        if bucket not in decks or positions[bucket] >= len(decks[bucket]):
            refill(bucket)
        idx = positions[bucket]
        positions[bucket] = idx + 1
        return decks[bucket][idx]

    target_rows = max(2, int(row_batch_size))
    yielded = 0
    while max_batches is None or yielded < max(0, int(max_batches)):
        batch: list[NativePromptGroup] = []
        rows = 0
        # Build one row-bounded microbatch.  Paired groups stay atomic.
        while rows < target_rows:
            current_active = [b for b in active if pools.get(b)]
            current_weights = [max(1e-9, float(weights.get(b, 0.0))) for b in current_active]
            bucket = rng.choices(current_active, weights=current_weights, k=1)[0]
            group = draw(bucket)
            needed = len(group.texts)
            if batch and rows + needed > target_rows:
                break
            batch.append(group)
            rows += needed
            if rows >= target_rows:
                break
        if batch:
            yield batch
            yielded += 1


_SUBJECT_MARKER_RE = re.compile(
    r"(?i)(?:^|[;|\n])\s*(?:subject|character|char|person|woman|man|girl|boy)\s*(?:#?\d+|[A-H])\s*:"
)
_EXACT_COUNT_RE = re.compile(
    r"(?i)(?:\bexactly\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:characters?|people|persons?|girls?|boys?|women|men)\b|\b(?:duo|pair|couple)\b|\btrio\b|\bquartet\b|\bquintet\b|\bsextet\b|(?<!\d)(\d{1,2})\s*(?:girls?|boys?|women|men)(?!\w))"
)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _tokenize(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str,
    max_length: int,
    include_offsets: bool = False,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "max_length": int(max_length),
        "return_tensors": "pt",
    }
    if include_offsets:
        kwargs["return_offsets_mapping"] = True
    encoded = tokenizer(list(texts), **kwargs)
    result = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    if include_offsets and "offset_mapping" in encoded:
        # Character offsets are only used by CPU-side prompt structure parsing;
        # keeping them off the GPU avoids a needless transfer.
        result["offset_mapping"] = encoded["offset_mapping"].cpu()
    return result


def _infer_training_subject_controls(
    texts: Sequence[str],
    offsets: torch.Tensor | None,
    attention_mask: torch.Tensor,
    *,
    device: str,
    max_subject_slots: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Map explicit ``subject N:`` clauses to token slots for binding training."""
    if offsets is None:
        return None, None
    bsz, seq_len = attention_mask.shape
    group_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
    counts = torch.zeros((bsz,), dtype=torch.long, device=device)
    any_structured = False
    for b, text in enumerate(texts):
        matches = list(_SUBJECT_MARKER_RE.finditer(text))
        count = len(matches)
        if count <= 0:
            m = _EXACT_COUNT_RE.search(text)
            if m is not None:
                raw = (m.group(1) or m.group(2) or "").lower()
                token = m.group(0).lower()
                if raw:
                    count = int(raw) if raw.isdigit() else int(_COUNT_WORDS.get(raw, 0))
                elif "duo" in token or "pair" in token or "couple" in token:
                    count = 2
                elif "trio" in token:
                    count = 3
                elif "quartet" in token:
                    count = 4
                elif "quintet" in token:
                    count = 5
                elif "sextet" in token:
                    count = 6
        if count > 0:
            counts[b] = min(int(count), int(max_subject_slots))
        if not matches:
            continue
        any_structured = True
        starts = [m.start() for m in matches]
        for t in range(min(seq_len, offsets.shape[1])):
            if int(attention_mask[b, t].item()) <= 0:
                continue
            start = int(offsets[b, t, 0].item())
            # group 0 is global text. Subject clauses start at slot 1.
            gid = 0
            for idx, marker_start in enumerate(starts, start=1):
                if start >= marker_start:
                    gid = idx
                else:
                    break
            group_ids[b, t] = min(gid, int(max_subject_slots) - 1)
    if not any_structured and not bool((counts > 0).any()):
        return None, None
    return group_ids, counts


def _group_binding_geometry_loss(
    student: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve Qwen3.5 subject-to-subject geometry inside each prompt."""
    if group_ids is None:
        return student.new_zeros(())
    losses: list[torch.Tensor] = []
    for b in range(student.shape[0]):
        valid = mask[b].bool()
        gids = torch.unique(group_ids[b, valid])
        gids = gids[gids > 0]
        if gids.numel() < 2:
            continue
        s_centers: list[torch.Tensor] = []
        t_centers: list[torch.Tensor] = []
        for gid in gids:
            idx = valid & (group_ids[b] == gid)
            if not bool(idx.any()):
                continue
            s_centers.append(student[b, idx].mean(dim=0))
            t_centers.append(source[b, idx].detach().mean(dim=0))
        if len(s_centers) < 2:
            continue
        s = F.normalize(torch.stack(s_centers).float(), dim=-1)
        t = F.normalize(torch.stack(t_centers).float(), dim=-1)
        gs = s @ s.T
        gt = t @ t.T
        keep = ~torch.eye(gs.shape[0], dtype=torch.bool, device=gs.device)
        losses.append(F.mse_loss(gs[keep], gt[keep]))
    return torch.stack(losses).mean() if losses else student.new_zeros(())


def _slot_centroid_anchor_loss(
    student: torch.Tensor,
    target: torch.Tensor,
    student_mask: torch.Tensor,
    target_mask: torch.Tensor,
    student_group_ids: torch.Tensor | None,
    target_group_ids: torch.Tensor | None,
) -> torch.Tensor:
    if student_group_ids is None or target_group_ids is None:
        return student.new_zeros(())
    losses: list[torch.Tensor] = []
    for b in range(student.shape[0]):
        sg = student_group_ids[b]
        tg = target_group_ids[b]
        svalid = student_mask[b].bool()
        tvalid = target_mask[b].bool()
        s_gids = {int(x) for x in torch.unique(sg[svalid]).tolist() if int(x) > 0}
        t_gids = {int(x) for x in torch.unique(tg[tvalid]).tolist() if int(x) > 0}
        gids = sorted(s_gids & t_gids)
        if not gids:
            continue
        per_prompt: list[torch.Tensor] = []
        for gid in gids:
            sidx = svalid & (sg == gid)
            tidx = tvalid & (tg == gid)
            if not bool(sidx.any()) or not bool(tidx.any()):
                continue
            sc = student[b, sidx].mean(dim=0)
            tc = target[b, tidx].detach().mean(dim=0).to(device=student.device, dtype=student.dtype)
            per_prompt.append(_cosine_distance(sc.unsqueeze(0), tc.unsqueeze(0)).mean())
            per_prompt.append(0.25 * _normalized_mse(sc, tc))
        if per_prompt:
            losses.append(torch.stack(per_prompt).mean())
    return torch.stack(losses).mean() if losses else student.new_zeros(())


def _multi_person_mask(subject_counts: torch.Tensor | None) -> torch.Tensor | None:
    if subject_counts is None:
        return None
    return subject_counts.reshape(-1).to(dtype=torch.long) >= 2


def _count_anchor_loss(
    student_pool: torch.Tensor,
    target_pool: torch.Tensor | None,
    subject_counts: torch.Tensor | None,
) -> torch.Tensor:
    if target_pool is None:
        return student_pool.new_zeros(())
    multi_mask = _multi_person_mask(subject_counts)
    if multi_mask is None or int(multi_mask.sum().item()) <= 0:
        return student_pool.new_zeros(())
    sp = student_pool[multi_mask]
    tp = target_pool[multi_mask].detach().to(device=student_pool.device, dtype=student_pool.dtype)
    return _cosine_distance(sp, tp).mean() + 0.25 * _normalized_mse(sp, tp)


def _ownership_anchor_loss(
    student_pool: torch.Tensor,
    target_pool: torch.Tensor | None,
    pairs: Sequence[tuple[int, int, str]],
) -> torch.Tensor:
    if target_pool is None:
        return student_pool.new_zeros(())
    losses: list[torch.Tensor] = []
    for a, b, category in pairs:
        if category not in {"binding", "attribute_swap", "count_binding"}:
            continue
        ds = _cosine_distance(student_pool[a:a+1], student_pool[b:b+1]).squeeze(0)
        dt = _cosine_distance(target_pool[a:a+1], target_pool[b:b+1]).squeeze(0).detach().to(device=student_pool.device, dtype=student_pool.dtype)
        losses.append(F.smooth_l1_loss(ds, dt))
    return torch.stack(losses).mean() if losses else student_pool.new_zeros(())


def _channel_distribution_loss(
    student: torch.Tensor,
    student_mask: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Match channel-level mean/variance to the 0.6B Anima anchor.

    This is deliberately reference-only.  It prevents a learned binding head
    from changing the global conditioning distribution (a common route to
    saturation/contrast drift) without suppressing Qwen3.5 semantic geometry.
    """
    def stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        m = mask.to(device=x.device, dtype=torch.float32).unsqueeze(-1)
        denom = m.sum(dim=(0, 1)).clamp_min(1.0)
        mean = (xf * m).sum(dim=(0, 1)) / denom
        var = ((xf - mean) ** 2 * m).sum(dim=(0, 1)) / denom
        return mean, var.add(1e-6).sqrt()
    sm, ss = stats(student, student_mask)
    tm, ts = stats(target, target_mask)
    # Normalise scale for a stable low-weight regulariser.
    mean_scale = tm.square().mean().sqrt().clamp_min(1e-3)
    std_scale = ts.square().mean().sqrt().clamp_min(1e-3)
    return F.smooth_l1_loss(sm / mean_scale, tm / mean_scale) + 0.5 * F.smooth_l1_loss(ss / std_scale, ts / std_scale)


@dataclass(frozen=True)
class _ValidationTargets:
    prompts: tuple[str, ...]
    source_pool: torch.Tensor
    reference_pool: torch.Tensor
    neutral_mask: torch.Tensor
    binding_mask: torch.Tensor

def _validation_prompt_flags(prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    explicit_color = re.compile(
        r"(?i)\b(?:high saturation|controlled saturation|low saturation|pastel palette|muted earth tones|monochrome|black and white|warm palette|cool palette|neon lighting|colored bounce light)\b"
    )
    binding = re.compile(
        r"(?i)(?:subject\s*(?:#?\d+|[A-H])\s*:|two women|two men|woman and man|duo|trio|quartet|group portrait|(?:[2-9]\d?\s*(?:girls?|boys?|women|men|people|characters?))|exactly\s+(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:characters?|people|girls?|boys?|women|men))"
    )
    neutral = torch.tensor([not bool(explicit_color.search(text)) for text in prompts], dtype=torch.bool)
    binding_mask = torch.tensor([bool(binding.search(text)) for text in prompts], dtype=torch.bool)
    return neutral, binding_mask

def _pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    z = F.normalize(x.float(), dim=-1)
    return 1.0 - z @ z.T

def _pooled_channel_drift(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    s = student.float()
    t = target.float()
    sm, ss = s.mean(dim=0), s.std(dim=0, unbiased=False).clamp_min(1e-5)
    tm, ts = t.mean(dim=0), t.std(dim=0, unbiased=False).clamp_min(1e-5)
    mean_scale = tm.square().mean().sqrt().clamp_min(1e-3)
    std_scale = ts.square().mean().sqrt().clamp_min(1e-3)
    return F.smooth_l1_loss(sm / mean_scale, tm / mean_scale) + 0.5 * F.smooth_l1_loss(ss / std_scale, ts / std_scale)

def _snapshot_trainable_state(model: AnimaNativeQwen35Encoder) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu", copy=True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

def _restore_trainable_state(model: AnimaNativeQwen35Encoder, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

def _build_validation_targets(
    student: AnimaNativeQwen35Encoder,
    reference: torch.nn.Module,
    source_tok: Any,
    ref_tok: Any,
    prompts: Sequence[str],
    *,
    device: str,
    ref_device: str,
    max_length: int,
    reference_max_length: int,
    batch_size: int,
) -> _ValidationTargets:
    source_pools: list[torch.Tensor] = []
    ref_pools: list[torch.Tensor] = []
    source_was_training = student.backbone.training
    student.backbone.eval()
    reference.eval()
    for start in range(0, len(prompts), max(1, int(batch_size))):
        batch = list(prompts[start:start + max(1, int(batch_size))])
        src = _tokenize(source_tok, batch, device=device, max_length=max_length)
        ref = _tokenize(
            ref_tok, batch, device=ref_device,
            max_length=min(max_length, max(32, int(reference_max_length))),
        )
        with torch.inference_mode():
            source_out = student.backbone(
                input_ids=src["input_ids"], attention_mask=src["attention_mask"],
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
            source_pools.append(_masked_pool(source_out.hidden_states[-1], src["attention_mask"]).float().cpu())
            ref_out = reference(
                input_ids=ref["input_ids"], attention_mask=ref["attention_mask"], use_cache=False,
            )
            ref_hidden = ref_out[0] if isinstance(ref_out, tuple) else ref_out.last_hidden_state
            ref_pools.append(_masked_pool(ref_hidden, ref["attention_mask"]).float().cpu())
    if source_was_training:
        student.backbone.train()
    neutral, binding = _validation_prompt_flags(prompts)
    return _ValidationTargets(
        prompts=tuple(prompts),
        source_pool=torch.cat(source_pools, dim=0),
        reference_pool=torch.cat(ref_pools, dim=0),
        neutral_mask=neutral,
        binding_mask=binding,
    )

def _evaluate_validation(
    student: AnimaNativeQwen35Encoder,
    source_tok: Any,
    targets: _ValidationTargets,
    *,
    device: str,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    pools: list[torch.Tensor] = []
    model_was_training = student.training
    backbone_was_training = student.backbone.training
    student.eval()
    for start in range(0, len(targets.prompts), max(1, int(batch_size))):
        batch = list(targets.prompts[start:start + max(1, int(batch_size))])
        src = _tokenize(source_tok, batch, device=device, max_length=max_length, include_offsets=True)
        group_ids, subject_counts = _infer_training_subject_controls(
            batch, src.get("offset_mapping"), src["attention_mask"],
            device=device, max_subject_slots=int(student.native_head.native_config.max_subject_slots),
        )
        with torch.inference_mode():
            out = student(
                input_ids=src["input_ids"], attention_mask=src["attention_mask"],
                anima_group_ids=group_ids, anima_subject_counts=subject_counts,
                use_cache=False, return_dict=True,
            )
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            pools.append(_masked_pool(hidden, src["attention_mask"]).float().cpu())
    if model_was_training:
        student.train()
        if not backbone_was_training:
            student.backbone.eval()

    student_pool = torch.cat(pools, dim=0)
    ref_pool = targets.reference_pool
    source_pool = targets.source_pool
    compat = _cosine_distance(student_pool, ref_pool).mean() + 0.25 * _normalized_mse(student_pool, ref_pool)
    geom = F.smooth_l1_loss(_pairwise_distance_matrix(student_pool), _pairwise_distance_matrix(source_pool))
    neutral = targets.neutral_mask
    if bool(neutral.any()):
        channel = _pooled_channel_drift(student_pool[neutral], ref_pool[neutral])
        srms = student_pool[neutral].square().mean(dim=-1).sqrt().clamp_min(1e-6)
        rrms = ref_pool[neutral].square().mean(dim=-1).sqrt().clamp_min(1e-6)
        rms = torch.log(srms / rrms).abs().mean()
    else:
        channel = student_pool.new_zeros(())
        rms = student_pool.new_zeros(())
    binding_mask = targets.binding_mask
    if int(binding_mask.sum().item()) >= 2:
        binding = F.smooth_l1_loss(
            _pairwise_distance_matrix(student_pool[binding_mask]),
            _pairwise_distance_matrix(source_pool[binding_mask]),
        )
    else:
        binding = student_pool.new_zeros(())
    # Compatibility and neutral distribution dominate the safety score; source
    # geometry/binding protect 0.8B knowledge and concept separation.
    score = compat + 0.75 * geom + 0.35 * channel + 0.20 * rms + 0.35 * binding
    return {
        "score": float(score.item()),
        "compat": float(compat.item()),
        "source_geom": float(geom.item()),
        "neutral_channel": float(channel.item()),
        "neutral_rms": float(rms.item()),
        "binding_geom": float(binding.item()),
    }


def _configure_trainable_backbone(model: AnimaNativeQwen35Encoder, last_n: int) -> list[torch.nn.Parameter]:
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if last_n <= 0:
        return []
    layers = getattr(model.backbone, "layers", None)
    if layers is None:
        raise RuntimeError("Qwen3.5 backbone does not expose .layers; cannot unfreeze upper layers")
    n = min(int(last_n), len(layers))
    for layer in list(layers)[-n:]:
        layer.requires_grad_(True)
    norm = getattr(model.backbone, "norm", None)
    if norm is not None:
        norm.requires_grad_(True)
    return [p for p in model.backbone.parameters() if p.requires_grad]


def _save_native_encoder(
    model: AnimaNativeQwen35Encoder,
    output: Path,
    *,
    config: NativeEncoderTrainingConfig,
    corpus_lines: Sequence[str],
    training_rows: int,
    steps: int,
    final_loss: float,
    best_step: int,
    validation_metrics: Mapping[str, float],
    early_stopped: bool,
    validation_prompts: int,
) -> dict[str, str]:
    model.to("cpu")
    source_config = model.backbone.config.to_dict() if hasattr(model.backbone.config, "to_dict") else {}
    tensors: dict[str, torch.Tensor] = {}
    for key, value in model.backbone.state_dict().items():
        tensors[f"encoder.{key}"] = value.detach().cpu().contiguous()
    for key, value in model.native_head.state_dict().items():
        tensors[f"native.{key}"] = value.detach().cpu().contiguous()

    metadata: dict[str, str] = {
        "format": _NATIVE_ENCODER_FORMAT_V1,
        "artifact_kind": _NATIVE_ENCODER_KIND,
        "contains_encoder_weights": "true",
        "anima_ready": "true",
        "bridge_required": "false",
        "source_family": "qwen3.5",
        "target_family": "anima-qwen3",
        "source_model": str(config.source_model or config.resume_native or "unknown"),
        "reference_model": str(config.reference_model),
        "source_tokenizer": str(config.source_tokenizer),
        "reference_tokenizer": str(config.reference_tokenizer),
        "source_hidden_size": str(int(getattr(model.backbone.config, "hidden_size", 1024))),
        "target_hidden_size": "1024",
        "source_architecture_fingerprint": encoder_config_fingerprint(model.backbone.config),
        "source_config_json": json.dumps(source_config, sort_keys=True, separators=(",", ":")),
        "training_corpus_sha256": corpus_sha256(corpus_lines),
        "training_corpus_lines": str(len(corpus_lines)),
        "training_rows": str(int(training_rows)),
        "training_steps": str(int(steps)),
        "training_final_loss": f"{float(final_loss):.8g}",
        "training_last_n_layers": str(int(config.train_last_n_layers)),
        "training_policy": "qwen35_fixed_budget_balanced_best_validation_v4_partial_qwen06_multi_person_anchor",
        "knowledge_policy": "preserve_qwen35_vocab_knowledge_multilingual_geometry_qwen3_is_compat_anchor",
        "training_budget_mode": "fixed_optimizer_steps",
        "training_fixed_budget_steps": str(int(config.fixed_budget_steps)),
        "training_balanced_sampling": "true" if config.balanced_sampling else "false",
        "training_multi_person_reference_fraction": f"{float(config.multi_person_reference_fraction):.8g}",
        "training_multi_person_compat_weight": f"{float(config.multi_person_compat_weight):.8g}",
        "training_count_anchor_weight": f"{float(config.count_anchor_weight):.8g}",
        "training_subject_slot_anchor_weight": f"{float(config.subject_slot_anchor_weight):.8g}",
        "training_ownership_anchor_weight": f"{float(config.ownership_anchor_weight):.8g}",
        "training_best_step": str(int(best_step)),
        "training_early_stopped": "true" if early_stopped else "false",
        "validation_prompts": str(int(validation_prompts)),
        "validation_score": f"{float(validation_metrics.get('score', float('nan'))):.8g}",
        "validation_compat": f"{float(validation_metrics.get('compat', float('nan'))):.8g}",
        "validation_source_geometry": f"{float(validation_metrics.get('source_geom', float('nan'))):.8g}",
        "validation_neutral_channel": f"{float(validation_metrics.get('neutral_channel', float('nan'))):.8g}",
        "validation_neutral_rms": f"{float(validation_metrics.get('neutral_rms', float('nan'))):.8g}",
        "validation_binding_geometry": f"{float(validation_metrics.get('binding_geom', float('nan'))):.8g}",
        "sampling_bucket_weights_json": json.dumps(default_sampling_bucket_weights(), sort_keys=True, separators=(",", ":")),
        "reference_batch_fraction": f"{float(config.reference_batch_fraction):.8g}",
        "reference_max_length": str(int(config.reference_max_length)),
        "bootstrap_bridge_used": "true" if config.bootstrap_bridge_profile else "false",
        "runtime_conditioning_mode": "native_encoder_direct",
    }
    metadata.update(native_head_metadata(model.native_head.native_config))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={str(k): str(v) for k, v in metadata.items()})
    return metadata


def train_anima_native_text_encoder(
    config: NativeEncoderTrainingConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> NativeEncoderTrainingResult:
    started = time.perf_counter()
    cfg = config.normalized()
    torch.manual_seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    device = _resolve_device(cfg.device)
    ref_device = device if cfg.reference_device == "auto" else _resolve_device(cfg.reference_device)
    dtype = _resolve_dtype(cfg.dtype, device)
    ref_dtype = _resolve_dtype(cfg.dtype, ref_device)
    _emit(
        progress_callback,
        stage="prepare",
        message="initialising corpus/tokenizers/models",
        device=device,
        reference_device=ref_device,
        dtype=str(dtype).replace("torch.", ""),
    )

    lines = read_prompt_lines(cfg.prompts)
    train_lines, validation_lines = split_validation_lines(
        lines, validation_size=int(cfg.validation_size), seed=int(cfg.seed),
    )
    groups = build_training_groups(
        train_lines,
        pair_fraction=float(cfg.pair_fraction),
        binding_pairs=int(cfg.binding_pairs),
        seed=int(cfg.seed),
    )
    preview = preview_training_groups(groups)
    training_rows = int(preview["rows"])
    validation_prompts = list(validation_lines)
    validation_prompts.extend(
        list(ANIMA_COMPAT_ANCHOR_PROMPTS)[: max(0, int(cfg.validation_anchor_size))]
    )
    # Preserve order while removing any overlap between held-out corpus rows and
    # fixed anchors. Held-out rows never re-enter the optimiser stream.
    validation_prompts = list(dict.fromkeys(validation_prompts))
    if not validation_prompts:
        validation_prompts = [ANIMA_COMPAT_ANCHOR_PROMPTS[0]]
    _emit(
        progress_callback,
        stage="corpus",
        prompt_lines=len(lines),
        train_prompt_lines=len(train_lines),
        validation_prompts=len(validation_prompts),
        balanced_sampling=bool(cfg.balanced_sampling),
        **preview,
    )

    _emit(progress_callback, stage="tokenizers", message="loading source tokenizer")
    source_tok = AutoTokenizer.from_pretrained(str(cfg.source_tokenizer))
    _emit(progress_callback, stage="tokenizers", message="loading reference tokenizer")
    ref_tok = AutoTokenizer.from_pretrained(str(cfg.reference_tokenizer))
    if source_tok.pad_token_id is None:
        source_tok.pad_token = source_tok.eos_token
    if ref_tok.pad_token_id is None:
        ref_tok.pad_token = ref_tok.eos_token

    if cfg.resume_native:
        _emit(progress_callback, stage="source_load", message="loading native checkpoint", path=str(cfg.resume_native))
        loaded = load_text_encoder_single_file(str(cfg.resume_native), device=device, dtype=dtype, cache=False)
        if not isinstance(loaded, AnimaNativeQwen35Encoder):
            raise ValueError("resume_native must be an anima_native_text_encoder_v1 checkpoint")
        student = loaded
        fresh_native = False
    else:
        if not cfg.source_model:
            raise ValueError("source_model is required when resume_native is not supplied")
        _emit(progress_callback, stage="source_load", message="loading raw Qwen3.5 source", path=str(cfg.source_model))
        backbone = load_text_encoder_single_file(str(cfg.source_model), device=device, dtype=dtype, cache=False)
        if isinstance(backbone, AnimaNativeQwen35Encoder):
            raise ValueError("source_model should be the raw Qwen3.5 source; use resume_native for native checkpoints")
        num_layers = int(getattr(backbone.config, "num_hidden_layers", 24))
        hidden_size = int(getattr(backbone.config, "hidden_size", 1024))
        layer_indices = cfg.native_layer_indices
        if layer_indices is None:
            # Qwen3.5-0.8B has 24 layers with full-attention milestones every
            # four layers.  Six equal-depth samples therefore preserve both
            # mid-layer lexical/local information and final semantic knowledge.
            steps = min(6, max(1, num_layers))
            layer_indices = tuple(max(1, round(num_layers * i / steps)) for i in range(1, steps + 1))
        head_cfg = AnimaNativeHeadConfig(
            hidden_size=hidden_size,
            intermediate_size=int(cfg.native_intermediate_size),
            layer_indices=tuple(int(x) for x in layer_indices),
            norm_eps=float(getattr(backbone.config, "rms_norm_eps", 1e-6)),
            final_layer_logit_bias=float(cfg.native_final_layer_logit_bias),
            max_subject_slots=int(cfg.native_max_subject_slots),
            binding_rms_min_ratio=float(cfg.native_binding_rms_min_ratio),
            binding_rms_max_ratio=float(cfg.native_binding_rms_max_ratio),
        )
        student = AnimaNativeQwen35Encoder(backbone, AnimaNativeQwen35Head(head_cfg)).to(device=device, dtype=dtype)
        fresh_native = True

    _emit(
        progress_callback,
        stage="source_ready",
        message="source/native student loaded",
        train_last_n_layers=int(cfg.train_last_n_layers),
    )

    bootstrap_bridge: AnimaTextEncoderBridge | None = None
    if cfg.bootstrap_bridge_profile:
        bootstrap_bridge = AnimaTextEncoderBridge.from_file(
            cfg.bootstrap_bridge_profile,
            center_strength=1.0,
            variance_strength=0.0,
            rms_strength=0.0,
            delta_clip_ratio=0.30,
            token_rms_strength=0.70,
            token_rms_min_ratio=0.85,
            token_rms_max_ratio=1.10,
        )
        if fresh_native:
            student.native_head.initialise_from_linear_alignment(
                bootstrap_bridge.rotation,
                bootstrap_bridge.source_mean,
                bootstrap_bridge.target_mean,
                strength=float(cfg.bootstrap_initialization_strength),
            )
        _emit(progress_callback, stage="bootstrap", bridge=str(cfg.bootstrap_bridge_profile))

    _emit(progress_callback, stage="reference_load", message="loading Qwen3-0.6B Anima reference", path=str(cfg.reference_model))
    reference = load_text_encoder_single_file(
        str(cfg.reference_model), device=ref_device, dtype=ref_dtype, cache=False
    )
    _emit(progress_callback, stage="reference_ready", message="reference loaded")
    reference.eval().requires_grad_(False)

    student.native_head.requires_grad_(True)
    backbone_params = _configure_trainable_backbone(student, int(cfg.train_last_n_layers))
    if int(cfg.train_last_n_layers) > 0 and cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        # Frozen embeddings feed trainable upper layers.  Some HF checkpointing
        # implementations require the boundary activation itself to carry grad.
        student.enable_input_require_grads()
    if backbone_params:
        student.train()
    else:
        # Head-only calibration never needs dropout/training-mode work in the
        # frozen 0.8B backbone. inference_mode() below can then use the cheapest
        # safe source forward while gradients still flow through the native head.
        student.backbone.eval()
        student.native_head.train()

    validation_targets = _build_validation_targets(
        student, reference, source_tok, ref_tok, validation_prompts,
        device=device, ref_device=ref_device, max_length=int(cfg.max_length),
        reference_max_length=int(cfg.reference_max_length),
        batch_size=max(4, int(cfg.batch_size)),
    )
    initial_validation = _evaluate_validation(
        student, source_tok, validation_targets,
        device=device, max_length=int(cfg.max_length), batch_size=max(4, int(cfg.batch_size)),
    )
    _emit(progress_callback, stage="validation", step=0, total=0, **{k: f"{v:.6f}" for k, v in initial_validation.items()})

    head_params = [p for p in student.native_head.parameters() if p.requires_grad]
    param_groups: list[dict[str, Any]] = [
        {"params": head_params, "lr": float(cfg.head_lr), "weight_decay": float(cfg.weight_decay)}
    ]
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": float(cfg.backbone_lr),
            "weight_decay": float(cfg.weight_decay),
        })
    if device.startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    adamw_kwargs: dict[str, Any] = {"betas": (0.9, 0.95), "eps": 1e-8}
    if device.startswith("cuda") and bool(cfg.fused_adamw):
        adamw_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
    except (TypeError, RuntimeError):
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

    rows_per_epoch = max(1, training_rows)
    micro_steps_per_epoch = max(1, math.ceil(rows_per_epoch / max(1, int(cfg.batch_size))))
    optimizer_steps_per_epoch = max(
        1, math.ceil(micro_steps_per_epoch / max(1, int(cfg.gradient_accumulation_steps)))
    )
    # v3: corpus size controls diversity, never optimiser magnitude.  Explicit
    # max_steps remains an escape hatch; otherwise the fixed budget is used.
    total_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else max(1, int(cfg.fixed_budget_steps))
    virtual_epochs = max(1, math.ceil(total_steps / optimizer_steps_per_epoch))
    warmup = max(0, round(total_steps * max(0.0, float(cfg.warmup_fraction))))
    _emit(
        progress_callback,
        stage="optimizer",
        step=0,
        total=total_steps,
        rows=training_rows,
        micro_steps_per_epoch=micro_steps_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        fixed_budget_steps=total_steps,
        virtual_epochs=virtual_epochs,
        balanced_sampling=bool(cfg.balanced_sampling),
        gradient_accumulation=int(cfg.gradient_accumulation_steps),
        reference_batch_fraction=f"{float(cfg.reference_batch_fraction):.3f}",
        multi_person_reference_fraction=f"{float(cfg.multi_person_reference_fraction):.3f}",
        reference_max_length=int(cfg.reference_max_length),
        fused_adamw=bool(adamw_kwargs.get("fused", False)),
        warmup_steps=warmup,
    )

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-6, float(step + 1) / float(warmup))
        remain = max(1, total_steps - warmup)
        progress = max(0.0, min(1.0, float(step - warmup) / float(remain)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(int(cfg.seed))
    global_step = 0
    accum = 0
    last_loss = float("nan")
    running: dict[str, float] = {}
    train_started = time.perf_counter()
    last_optimizer_time = train_started
    ema_step_seconds: float | None = None
    micro_step_global = 0
    micro_total = max(1, total_steps * max(1, int(cfg.gradient_accumulation_steps)))
    best_step = 0
    best_metrics = dict(initial_validation)
    best_score = float(initial_validation["score"])
    best_state = _snapshot_trainable_state(student) if bool(cfg.restore_best_checkpoint) else {}
    stale_validations = 0
    early_stopped = False

    for epoch in range(virtual_epochs):
        if global_step >= total_steps or early_stopped:
            break
        if cfg.balanced_sampling:
            batch_iterator = _iter_balanced_group_batches(
                groups, row_batch_size=max(2, int(cfg.batch_size)), rng=rng,
                max_batches=micro_steps_per_epoch,
            )
        else:
            batch_iterator = _iter_group_batches(
                groups, row_batch_size=max(2, int(cfg.batch_size)), rng=rng
            )
        for group_batch in batch_iterator:
            if global_step >= total_steps or early_stopped:
                break
            micro_step_global += 1
            texts, pairs = _flatten_group_batch(group_batch)
            need_offsets = any(_SUBJECT_MARKER_RE.search(text) or _EXACT_COUNT_RE.search(text) for text in texts)
            src = _tokenize(
                source_tok,
                texts,
                device=device,
                max_length=int(cfg.max_length),
                include_offsets=need_offsets,
            )
            group_ids, subject_counts = _infer_training_subject_controls(
                texts,
                src.get("offset_mapping"),
                src["attention_mask"],
                device=device,
                max_subject_slots=int(student.native_head.native_config.max_subject_slots),
            )

            # Qwen3-0.6B is intentionally an Anima-distribution anchor, not the
            # semantic teacher.  Sampling anchor batches avoids a full second
            # language-model forward on every microbatch while inverse-
            # probability scaling keeps its expected loss contribution stable.
            reference_fraction = max(0.05, min(1.0, float(cfg.reference_batch_fraction)))
            multi_person_fraction = max(reference_fraction, min(1.0, float(cfg.multi_person_reference_fraction)))
            batch_bucket = {group_sampling_bucket(group) for group in group_batch}
            batch_has_multi_person = bool(batch_bucket & {"binding", "count", "multilingual"})
            active_reference_fraction = multi_person_fraction if batch_has_multi_person else reference_fraction
            use_reference = micro_step_global == 1 or rng.random() < active_reference_fraction
            ref: dict[str, torch.Tensor] | None = None
            ref_hidden: torch.Tensor | None = None
            ref_pool: torch.Tensor | None = None

            # Head-only stage does not need a graph through the 0.8B backbone.
            backbone_context = torch.enable_grad() if backbone_params else torch.no_grad()
            with backbone_context:
                source_out = student.backbone(
                    input_ids=src["input_ids"],
                    attention_mask=src["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            native_hidden, details = student.native_head(
                source_out.hidden_states,
                attention_mask=src["attention_mask"],
                group_ids=group_ids,
                subject_counts=subject_counts,
                return_details=True,
            )
            source_hidden = details["final_hidden"]

            if use_reference:
                ref = _tokenize(
                    ref_tok,
                    texts,
                    device=ref_device,
                    max_length=min(int(cfg.max_length), max(32, int(cfg.reference_max_length))),
                    include_offsets=need_offsets,
                )
                with torch.inference_mode():
                    ref_out = reference(
                        input_ids=ref["input_ids"],
                        attention_mask=ref["attention_mask"],
                        use_cache=False,
                    )
                    ref_hidden = ref_out[0] if isinstance(ref_out, tuple) else ref_out.last_hidden_state

            student_pool = _masked_pool(native_hidden, src["attention_mask"])
            source_pool = _masked_pool(source_hidden.detach(), src["attention_mask"])
            ref_group_ids: torch.Tensor | None = None
            if ref_hidden is not None and ref is not None:
                ref_pool = _masked_pool(ref_hidden, ref["attention_mask"]).to(
                    device=device, dtype=student_pool.dtype
                )
                ref_group_ids, _ = _infer_training_subject_controls(
                    texts,
                    ref.get("offset_mapping"),
                    ref["attention_mask"],
                    device=ref_device,
                    max_subject_slots=int(student.native_head.native_config.max_subject_slots),
                )

            sample_weights = torch.ones(student_pool.shape[0], device=device, dtype=torch.float32)
            multi_mask = _multi_person_mask(subject_counts)
            if multi_mask is not None and bool(multi_mask.any()):
                sample_weights = sample_weights + multi_mask.to(dtype=sample_weights.dtype) * float(cfg.multi_person_reference_boost)
            knowledge_losses: list[torch.Tensor] = []
            gains: list[torch.Tensor] = []
            for a, b, category in pairs:
                ds = _cosine_distance(source_pool[a:a+1], source_pool[b:b+1]).squeeze(0)
                dstudent = _cosine_distance(student_pool[a:a+1], student_pool[b:b+1]).squeeze(0)
                if ref_pool is not None:
                    dr = _cosine_distance(ref_pool[a:a+1], ref_pool[b:b+1]).squeeze(0)
                    gain = (torch.relu(ds - dr) / ds.clamp_min(1e-4)).clamp(0.0, 1.0).detach()
                    relaxed = 1.0 - float(cfg.reference_relax_for_gain) * float(gain.item())
                    sample_weights[a] *= relaxed
                    sample_weights[b] *= relaxed
                else:
                    # On non-anchor batches Qwen3.5 alone owns semantic truth.
                    # This is especially important for vocabulary, multilingual
                    # distinctions, count and attribute-ownership hard negatives.
                    gain = ds.detach().new_ones(())
                hard_multiplier = 1.35 if category in {"attribute_swap", "count_binding", "multilingual_binding"} else 1.0
                knowledge_losses.append(
                    float(hard_multiplier) * gain * F.smooth_l1_loss(dstudent, ds.detach())
                )
                gains.append(gain)

            reference_scale = (1.0 / active_reference_fraction) if ref_pool is not None else 0.0
            if ref_pool is not None:
                compat_each = _cosine_distance(student_pool, ref_pool)
                compat_each = compat_each + 0.25 * _normalized_mse_each(student_pool, ref_pool)
                compat = reference_scale * (
                    (compat_each * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
                )
            else:
                compat = student_pool.new_zeros(())

            source_geometry = _pairwise_geometry_loss(student_pool, source_pool)
            token_geometry = _token_geometry_loss(native_hidden, source_hidden.detach(), src["attention_mask"])
            binding_geometry = _group_binding_geometry_loss(
                native_hidden,
                source_hidden.detach(),
                src["attention_mask"],
                group_ids,
            )
            multi_person_compat = student_pool.new_zeros(())
            count_anchor = student_pool.new_zeros(())
            slot_anchor = student_pool.new_zeros(())
            ownership_anchor = student_pool.new_zeros(())
            if ref_hidden is not None and ref is not None and ref_pool is not None:
                ref_hidden_device = ref_hidden.to(device=device, dtype=native_hidden.dtype)
                ref_mask = ref["attention_mask"].to(device)
                multi_person_compat = reference_scale * _count_anchor_loss(student_pool, ref_pool, subject_counts)
                count_anchor = reference_scale * _count_anchor_loss(student_pool, ref_pool, subject_counts)
                slot_anchor = reference_scale * _slot_centroid_anchor_loss(
                    native_hidden,
                    ref_hidden_device,
                    src["attention_mask"],
                    ref_mask,
                    group_ids,
                    ref_group_ids.to(device) if ref_group_ids is not None else None,
                )
                ownership_anchor = reference_scale * _ownership_anchor_loss(student_pool, ref_pool, pairs)
            knowledge_gain = (
                torch.stack(knowledge_losses).mean() if knowledge_losses else student_pool.new_zeros(())
            )

            bootstrap_target: torch.Tensor | None = None
            bootstrap_loss = student_pool.new_zeros(())
            if bootstrap_bridge is not None:
                with torch.no_grad():
                    bootstrap_target = bootstrap_bridge.apply(source_hidden.detach())
                bootstrap_loss = _cosine_distance(
                    native_hidden.reshape(-1, native_hidden.shape[-1]),
                    bootstrap_target.reshape(-1, bootstrap_target.shape[-1]),
                )
                flat_mask = src["attention_mask"].reshape(-1).bool()
                bootstrap_loss = bootstrap_loss[flat_mask].mean()
                bootstrap_loss = bootstrap_loss + 0.20 * _normalized_mse(
                    native_hidden[src["attention_mask"].bool()],
                    bootstrap_target[src["attention_mask"].bool()],
                )

            distribution = student_pool.new_zeros(())
            channel_distribution = student_pool.new_zeros(())
            if bootstrap_target is not None:
                distribution = _distribution_loss(
                    native_hidden,
                    src["attention_mask"],
                    bootstrap_target,
                    src["attention_mask"],
                )
            elif ref_hidden is not None and ref is not None:
                ref_target = ref_hidden.to(device=device, dtype=native_hidden.dtype)
                ref_mask = ref["attention_mask"].to(device)
                distribution = reference_scale * _distribution_loss(
                    native_hidden,
                    src["attention_mask"],
                    ref_target,
                    ref_mask,
                )
                channel_distribution = reference_scale * _channel_distribution_loss(
                    native_hidden,
                    src["attention_mask"],
                    ref_target,
                    ref_mask,
                )
            layer_prior = _layer_prior_loss(details["layer_weights"])

            progress = float(global_step) / max(1.0, float(total_steps - 1))
            bootstrap_weight = float(cfg.bootstrap_token_weight) * (
                1.0 - progress * (1.0 - max(0.0, min(1.0, float(cfg.bootstrap_final_fraction))))
            )
            loss = (
                float(cfg.anima_compat_weight) * compat
                + float(cfg.source_geometry_weight) * source_geometry
                + float(cfg.token_geometry_weight) * token_geometry
                + float(cfg.knowledge_gain_weight) * knowledge_gain
                + float(cfg.binding_geometry_weight) * binding_geometry
                + float(cfg.multi_person_compat_weight) * multi_person_compat
                + float(cfg.count_anchor_weight) * count_anchor
                + float(cfg.subject_slot_anchor_weight) * slot_anchor
                + float(cfg.ownership_anchor_weight) * ownership_anchor
                + float(cfg.distribution_weight) * distribution
                + float(cfg.channel_distribution_weight) * channel_distribution
                + float(cfg.layer_prior_weight) * layer_prior
                + bootstrap_weight * bootstrap_loss
            )
            scaled = loss / max(1, int(cfg.gradient_accumulation_steps))
            scaled.backward()
            accum += 1
            last_loss = float(loss.detach().item())

            values = {
                "compat": float(compat.detach().item()),
                "source_geom": float(source_geometry.detach().item()),
                "token_geom": float(token_geometry.detach().item()),
                "knowledge": float(knowledge_gain.detach().item()),
                "binding": float(binding_geometry.detach().item()),
                "multi_compat": float(multi_person_compat.detach().item()),
                "count_anchor": float(count_anchor.detach().item()),
                "slot_anchor": float(slot_anchor.detach().item()),
                "ownership_anchor": float(ownership_anchor.detach().item()),
                "distribution": float(distribution.detach().item()),
                "channel_dist": float(channel_distribution.detach().item()),
                "bootstrap": float(bootstrap_loss.detach().item()),
                "gain": float(torch.stack(gains).mean().item()) if gains else 0.0,
                "reference": 1.0 if ref_pool is not None else 0.0,
            }
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value

            if accum >= max(1, int(cfg.gradient_accumulation_steps)):
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    max_norm=float(cfg.max_grad_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                step_now = time.perf_counter()
                step_seconds = max(1e-9, step_now - last_optimizer_time)
                last_optimizer_time = step_now
                ema_step_seconds = (
                    step_seconds
                    if ema_step_seconds is None
                    else 0.90 * ema_step_seconds + 0.10 * step_seconds
                )
                if global_step == 1 or global_step % max(1, int(cfg.log_every)) == 0 or global_step >= total_steps:
                    interval = max(1, min(int(cfg.log_every), global_step))
                    avg_values = {key: value / float(interval) for key, value in running.items()}
                    elapsed_train = step_now - train_started
                    eta_seconds = max(0, total_steps - global_step) * float(ema_step_seconds)
                    _emit(
                        progress_callback,
                        stage="train",
                        step=global_step,
                        total=total_steps,
                        micro_step=micro_step_global,
                        micro_total=micro_total,
                        epoch=epoch + 1,
                        loss=f"{last_loss:.6f}",
                        lr=f"{scheduler.get_last_lr()[0]:.3e}",
                        gain=f"{values['gain']:.4f}",
                        elapsed_seconds=f"{elapsed_train:.3f}",
                        eta_seconds=f"{eta_seconds:.3f}",
                        step_seconds=f"{step_seconds:.3f}",
                        step_seconds_ema=f"{float(ema_step_seconds):.3f}",
                        compat=f"{avg_values.get('compat', values['compat']):.6f}",
                        source_geom=f"{avg_values.get('source_geom', values['source_geom']):.6f}",
                        token_geom=f"{avg_values.get('token_geom', values['token_geom']):.6f}",
                        knowledge=f"{avg_values.get('knowledge', values['knowledge']):.6f}",
                        distribution=f"{avg_values.get('distribution', values['distribution']):.6f}",
                        binding=f"{avg_values.get('binding', values['binding']):.6f}",
                        multi_compat=f"{avg_values.get('multi_compat', values['multi_compat']):.6f}",
                        count_anchor=f"{avg_values.get('count_anchor', values['count_anchor']):.6f}",
                        slot_anchor=f"{avg_values.get('slot_anchor', values['slot_anchor']):.6f}",
                        ownership_anchor=f"{avg_values.get('ownership_anchor', values['ownership_anchor']):.6f}",
                        reference=f"{avg_values.get('reference', values['reference']):.3f}",
                        bootstrap=f"{avg_values.get('bootstrap', values['bootstrap']):.6f}",
                    )
                    running.clear()

                validation_due = (
                    int(cfg.validation_every) > 0
                    and (global_step % max(1, int(cfg.validation_every)) == 0 or global_step >= total_steps)
                )
                if validation_due:
                    metrics = _evaluate_validation(
                        student, source_tok, validation_targets,
                        device=device, max_length=int(cfg.max_length),
                        batch_size=max(4, int(cfg.batch_size)),
                    )
                    score = float(metrics["score"])
                    improvement = max(0.0, float(cfg.early_stopping_min_delta))
                    threshold = best_score * (1.0 - improvement)
                    improved = score < threshold
                    if improved:
                        best_score = score
                        best_step = global_step
                        best_metrics = dict(metrics)
                        stale_validations = 0
                        if bool(cfg.restore_best_checkpoint):
                            best_state = _snapshot_trainable_state(student)
                    else:
                        stale_validations += 1
                    _emit(
                        progress_callback,
                        stage="validation",
                        step=global_step,
                        total=total_steps,
                        best_step=best_step,
                        improved=improved,
                        stale=stale_validations,
                        **{k: f"{v:.6f}" for k, v in metrics.items()},
                    )
                    if (
                        int(cfg.early_stopping_patience) > 0
                        and stale_validations >= int(cfg.early_stopping_patience)
                    ):
                        early_stopped = True

            del source_out, native_hidden, details, source_hidden
            if ref_hidden is not None:
                del ref_hidden

        # Flush a partial accumulation at the epoch boundary.
        if accum > 0 and global_step < total_steps and not early_stopped:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                max_norm=float(cfg.max_grad_norm),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            global_step += 1
            step_now = time.perf_counter()
            step_seconds = max(1e-9, step_now - last_optimizer_time)
            last_optimizer_time = step_now
            ema_step_seconds = step_seconds if ema_step_seconds is None else 0.90 * ema_step_seconds + 0.10 * step_seconds
            _emit(
                progress_callback,
                stage="train",
                step=global_step,
                total=total_steps,
                micro_step=micro_step_global,
                micro_total=micro_total,
                epoch=epoch + 1,
                loss=f"{last_loss:.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.3e}",
                elapsed_seconds=f"{time.perf_counter() - train_started:.3f}",
                eta_seconds=f"{max(0, total_steps - global_step) * float(ema_step_seconds):.3f}",
                step_seconds=f"{step_seconds:.3f}",
                step_seconds_ema=f"{float(ema_step_seconds):.3f}",
            )

    if bool(cfg.restore_best_checkpoint) and best_state:
        _restore_trainable_state(student, best_state)
        # The final artifact is always the best validation checkpoint, never the
        # last optimiser step. This is the main guard against saturation drift.
        final_metrics = _evaluate_validation(
            student, source_tok, validation_targets,
            device=device, max_length=int(cfg.max_length),
            batch_size=max(4, int(cfg.batch_size)),
        )
        best_metrics = dict(final_metrics)
        best_score = float(final_metrics["score"])
    elif best_step == 0:
        best_step = global_step

    output = Path(cfg.output)
    _emit(progress_callback, stage="save", message="serialising native encoder", output=str(output))
    metadata = _save_native_encoder(
        student,
        output,
        config=cfg,
        corpus_lines=lines,
        training_rows=training_rows,
        steps=global_step,
        final_loss=last_loss,
        best_step=best_step,
        validation_metrics=best_metrics,
        early_stopped=early_stopped,
        validation_prompts=len(validation_targets.prompts),
    )
    elapsed = time.perf_counter() - started
    result = NativeEncoderTrainingResult(
        output=output,
        steps=global_step,
        elapsed_seconds=elapsed,
        corpus_lines=len(lines),
        training_rows=training_rows,
        metadata=metadata,
        final_loss=last_loss,
        best_step=best_step,
        validation_score=best_score,
        early_stopped=early_stopped,
    )
    _emit(
        progress_callback,
        stage="done",
        step=global_step,
        total=max(global_step, total_steps),
        loss=f"{last_loss:.6f}",
        best_step=best_step,
        validation_score=f"{best_score:.6f}",
        early_stopped=early_stopped,
        output=output,
        elapsed_seconds=f"{elapsed:.3f}",
        eta_seconds="0",
    )
    del reference, student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _parse_layer_indices(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", default=None)
    p.add_argument("--reference-model", required=True)
    p.add_argument("--prompts", required=True, help="UTF-8 one-prompt-per-line calibration file")
    p.add_argument("--output", required=True)
    p.add_argument("--source-tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--reference-tokenizer", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--resume-native", default=None)
    p.add_argument("--bootstrap-bridge-profile", default=None)
    p.add_argument("--bootstrap-initialization-strength", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--reference-device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--fixed-budget-steps", type=int, default=1000)
    p.add_argument("--no-balanced-sampling", action="store_true")
    p.add_argument("--validation-size", type=int, default=192)
    p.add_argument("--validation-anchor-size", type=int, default=32)
    p.add_argument("--validation-every", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.002)
    p.add_argument("--no-restore-best-checkpoint", action="store_true")
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--train-last-n-layers", type=int, default=0)
    p.add_argument("--native-intermediate-size", type=int, default=1536)
    p.add_argument("--native-layer-indices", default=None, help="e.g. 4,8,12,16,20,24")
    p.add_argument("--pair-fraction", type=float, default=0.35)
    p.add_argument("--binding-pairs", type=int, default=256)
    p.add_argument("--reference-batch-fraction", type=float, default=0.50)
    p.add_argument("--multi-person-reference-fraction", type=float, default=1.00)
    p.add_argument("--reference-max-length", type=int, default=256)
    p.add_argument("--multi-person-compat-weight", type=float, default=0.40)
    p.add_argument("--count-anchor-weight", type=float, default=0.45)
    p.add_argument("--subject-slot-anchor-weight", type=float, default=0.40)
    p.add_argument("--ownership-anchor-weight", type=float, default=0.30)
    p.add_argument("--multi-person-reference-boost", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=3571)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main() -> None:
    a = _parser().parse_args()
    cfg = NativeEncoderTrainingConfig(
        source_model=a.source_model,
        reference_model=a.reference_model,
        prompts=a.prompts,
        output=a.output,
        source_tokenizer=a.source_tokenizer,
        reference_tokenizer=a.reference_tokenizer,
        resume_native=a.resume_native,
        bootstrap_bridge_profile=a.bootstrap_bridge_profile,
        bootstrap_initialization_strength=a.bootstrap_initialization_strength,
        device=a.device,
        reference_device=a.reference_device,
        dtype=a.dtype,
        max_length=a.max_length,
        batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        epochs=a.epochs,
        max_steps=a.max_steps,
        fixed_budget_steps=a.fixed_budget_steps,
        balanced_sampling=not a.no_balanced_sampling,
        validation_size=a.validation_size,
        validation_anchor_size=a.validation_anchor_size,
        validation_every=a.validation_every,
        early_stopping_patience=a.early_stopping_patience,
        early_stopping_min_delta=a.early_stopping_min_delta,
        restore_best_checkpoint=not a.no_restore_best_checkpoint,
        head_lr=a.head_lr,
        backbone_lr=a.backbone_lr,
        train_last_n_layers=a.train_last_n_layers,
        native_intermediate_size=a.native_intermediate_size,
        native_layer_indices=_parse_layer_indices(a.native_layer_indices),
        pair_fraction=a.pair_fraction,
        binding_pairs=a.binding_pairs,
        reference_batch_fraction=a.reference_batch_fraction,
        multi_person_reference_fraction=a.multi_person_reference_fraction,
        reference_max_length=a.reference_max_length,
        multi_person_compat_weight=a.multi_person_compat_weight,
        count_anchor_weight=a.count_anchor_weight,
        subject_slot_anchor_weight=a.subject_slot_anchor_weight,
        ownership_anchor_weight=a.ownership_anchor_weight,
        multi_person_reference_boost=a.multi_person_reference_boost,
        seed=a.seed,
        log_every=a.log_every,
    )
    result = train_anima_native_text_encoder(cfg)
    print(result.summary())


if __name__ == "__main__":
    main()
