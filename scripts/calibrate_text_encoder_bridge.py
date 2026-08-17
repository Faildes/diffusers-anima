#!/usr/bin/env python3
"""Create an Anima text-encoder compatibility profile.

The calibration is gradient-free. It pairs the same visual prompts through a
source encoder (for example Qwen3.5-0.8B-Base) and the Anima reference encoder
(Qwen3-0.6B-Base), solves an orthogonal Procrustes transform, validates it on a
held-out split, and writes a self-describing safetensors profile.

This module is intentionally dual-use:

* CLI: ``python scripts/calibrate_text_encoder_bridge.py ...``
* Jupyter/Python: import ``BridgeCalibrationConfig`` and
  ``calibrate_text_encoder_bridge`` and call them directly.

``prompts`` / ``--prompts`` is optional. Without it, a deterministic built-in
visual corpus is used. ``bundle_source_weights`` / ``--bundle-source-weights``
additionally stores ``encoder.*`` tensors so the result can later be supplied
directly as ``encoder_path``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Mapping

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

from bridge_calibration_corpus import (  # noqa: E402
    analyze_bridge_calibration_prompts,
    build_default_bridge_calibration_prompts,
)
from diffusers_anima.pipelines.anima.loading import load_text_encoder_single_file  # noqa: E402
from diffusers_anima.pipelines.anima.text_encoder_bridge import encoder_config_fingerprint  # noqa: E402


@dataclass(frozen=True)
class CalibrationProgress:
    """One progress event emitted by the calibrator."""

    stage: str
    current: int | None = None
    total: int | None = None
    message: str = ""

    @property
    def fraction(self) -> float | None:
        if self.current is None or self.total in (None, 0):
            return None
        return max(0.0, min(1.0, float(self.current) / float(self.total)))


ProgressCallback = Callable[[CalibrationProgress], None]


@dataclass
class BridgeCalibrationConfig:
    """Configuration shared by CLI and Jupyter/Python callers."""

    source_model: str | Path
    reference_model: str | Path
    output: str | Path
    source_tokenizer: str | Path = "Qwen/Qwen3.5-0.8B-Base"
    reference_tokenizer: str | Path = "Qwen/Qwen3-0.6B-Base"
    prompts: str | Path | None = None
    dump_prompts: str | Path | None = None
    default_prompt_count: int = 4096
    device: str = "auto"
    batch_size: int = 16
    pooling: str = "both"
    last_weight: float = 0.65
    max_length: int = 2048
    include_phrases: bool = True
    min_phrase_chars: int = 3
    validation_fraction: float = 0.08
    seed: int = 3571
    bundle_source_weights: bool = False
    cleanup_models: bool = True

    def normalized(self) -> "BridgeCalibrationConfig":
        return BridgeCalibrationConfig(
            source_model=str(self.source_model),
            reference_model=str(self.reference_model),
            output=str(self.output),
            source_tokenizer=str(self.source_tokenizer),
            reference_tokenizer=str(self.reference_tokenizer),
            prompts=None if self.prompts is None else str(self.prompts),
            dump_prompts=None if self.dump_prompts is None else str(self.dump_prompts),
            default_prompt_count=int(self.default_prompt_count),
            device=str(self.device),
            batch_size=int(self.batch_size),
            pooling=str(self.pooling),
            last_weight=float(self.last_weight),
            max_length=int(self.max_length),
            include_phrases=bool(self.include_phrases),
            min_phrase_chars=int(self.min_phrase_chars),
            validation_fraction=float(self.validation_fraction),
            seed=int(self.seed),
            bundle_source_weights=bool(self.bundle_source_weights),
            cleanup_models=bool(self.cleanup_models),
        )


@dataclass(frozen=True)
class BridgeCalibrationResult:
    """Summary returned to a notebook after calibration completes."""

    output: Path
    artifact_kind: str
    metadata: Mapping[str, str]
    corpus_source: str
    corpus_lines: int
    expanded_anchors: int
    train_samples: int
    validation_samples: int
    validation_cosine_before: float
    validation_cosine_after: float
    validation_rmse_after: float
    recommended_center_strength: float
    recommended_variance_strength: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["output"] = str(self.output)
        out["metadata"] = dict(self.metadata)
        return out

    def summary(self) -> str:
        return (
            f"{self.artifact_kind}: {self.output}\n"
            f"cosine {self.validation_cosine_before:.6f} -> {self.validation_cosine_after:.6f}; "
            f"rmse={self.validation_rmse_after:.6f}; "
            f"center={self.recommended_center_strength:.2f}; "
            f"variance={self.recommended_variance_strength:.2f}; "
            f"anchors={self.expanded_anchors}; elapsed={self.elapsed_seconds:.1f}s"
        )


class _ConsoleProgress:
    def __init__(self) -> None:
        self._last_bucket: dict[str, int] = {}

    def __call__(self, event: CalibrationProgress) -> None:
        if event.current is None or event.total in (None, 0):
            if event.message:
                print(f"[bridge-calibration] {event.stage}: {event.message}")
            else:
                print(f"[bridge-calibration] {event.stage}")
            return
        bucket = int(20 * event.current / max(1, event.total))
        if bucket != self._last_bucket.get(event.stage) or event.current >= event.total:
            self._last_bucket[event.stage] = bucket
            suffix = f" - {event.message}" if event.message else ""
            print(f"[bridge-calibration] {event.stage} {event.current}/{event.total}{suffix}")


def make_jupyter_progress_callback() -> ProgressCallback:
    """Return a single-cell progress renderer for JupyterLab.

    It uses IPython's display-id update when available and falls back to the
    normal console printer elsewhere. No ipywidgets dependency is required.
    """

    fallback = _ConsoleProgress()
    try:
        from IPython.display import HTML, display
    except Exception:
        return fallback

    handle: dict[str, Any] = {"display": None}

    def callback(event: CalibrationProgress) -> None:
        fraction = event.fraction
        if fraction is None:
            percent = ""
            bar = ""
        else:
            pct = 100.0 * fraction
            percent = f" {pct:5.1f}%"
            bar = (
                '<div style="height:8px;background:#ddd;border-radius:4px;overflow:hidden;margin-top:4px">'
                f'<div style="width:{pct:.2f}%;height:100%;background:#4c8bf5"></div></div>'
            )
        counts = ""
        if event.current is not None and event.total is not None:
            counts = f" &nbsp; {event.current:,}/{event.total:,}"
        message = event.message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = HTML(
            f'<div style="font-family:monospace"><b>{event.stage}</b>{percent}{counts}<br>'
            f'<span>{message}</span>{bar}</div>'
        )
        if handle["display"] is None:
            handle["display"] = display(html, display_id=True)
        else:
            handle["display"].update(html)

    return callback


def _emit(callback: ProgressCallback | None, event: CalibrationProgress) -> None:
    (callback or _ConsoleProgress())(event)


def _device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_calibration_prompts(
    path: str | Path | None = None,
    *,
    default_count: int = 4096,
    seed: int = 3571,
) -> tuple[list[str], str]:
    """Load a user corpus or deterministically build the default corpus."""

    resolved = None if path is None else Path(path)
    if resolved is None:
        prompts = build_default_bridge_calibration_prompts(default_count, seed)
        source = f"builtin:anima_bridge_calibration_v2:{default_count}:{seed}"
    else:
        prompts = [
            line.strip() for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        source = str(resolved)
    if len(prompts) < 256:
        raise ValueError("Calibration needs at least 256 prompt lines; thousands are recommended.")
    return prompts, source


# Backward-compatible private name used by earlier scripts/tests.
def _read_prompt_lines(path: Path | None, *, default_count: int, seed: int) -> tuple[list[str], str]:
    return load_calibration_prompts(path, default_count=default_count, seed=seed)


def _expand_prompt_anchors(prompts: list[str], *, include_phrases: bool, min_phrase_chars: int) -> list[str]:
    if not include_phrases:
        return list(prompts)
    out: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        candidates = [prompt]
        candidates.extend(
            item.strip()
            for item in re.split(r"[,;\n]+", prompt)
            if len(item.strip()) >= min_phrase_chars
        )
        for item in candidates:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def preview_calibration_corpus(
    prompts: str | Path | list[str] | tuple[str, ...] | None = None,
    *,
    default_count: int = 4096,
    seed: int = 3571,
    include_phrases: bool = True,
    min_phrase_chars: int = 3,
) -> dict[str, Any]:
    """Return lightweight corpus statistics before loading either model."""

    if prompts is None or isinstance(prompts, (str, Path)):
        base, source = load_calibration_prompts(prompts, default_count=default_count, seed=seed)
    else:
        base = [str(item).strip() for item in prompts if str(item).strip()]
        source = "python:list"
        if len(base) < 256:
            raise ValueError("Calibration needs at least 256 prompt lines; thousands are recommended.")
    anchors = _expand_prompt_anchors(
        base,
        include_phrases=include_phrases,
        min_phrase_chars=int(min_phrase_chars),
    )
    report = analyze_bridge_calibration_prompts(base)
    report.update({
        "source": source,
        "expanded_anchors": len(anchors),
        "include_phrases": bool(include_phrases),
        "min_phrase_chars": int(min_phrase_chars),
        "corpus_sha256": _corpus_sha256(base),
    })
    return report


def _corpus_sha256(prompts: list[str]) -> str:
    payload = ("\n".join(prompts) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_anchors(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: str,
    pooling: str,
    last_weight: float,
    max_length: int,
) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.inference_mode():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        hidden = hidden.float()
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        if pooling == "mean":
            anchors = mean
        elif pooling == "last":
            anchors = last
        elif pooling == "blend":
            anchors = mean * (1.0 - float(last_weight)) + last * float(last_weight)
        elif pooling == "both":
            anchors = torch.cat([mean, last], dim=0)
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
    return anchors.detach().cpu()


def _collect_train_statistics(
    source_model,
    reference_model,
    source_tok,
    reference_tok,
    texts: list[str],
    *,
    device: str,
    batch_size: int,
    pooling: str,
    last_weight: float,
    max_length: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    dim = int(getattr(source_model.config, "hidden_size", 0))
    sum_x = torch.zeros(dim, dtype=torch.float64)
    sum_y = torch.zeros(dim, dtype=torch.float64)
    sum_x2 = torch.zeros((dim, dim), dtype=torch.float64)
    sum_y_sq = torch.zeros(dim, dtype=torch.float64)
    sum_xy = torch.zeros((dim, dim), dtype=torch.float64)
    count = 0
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        x = _encode_anchors(
            source_model, source_tok, batch_text, device=device,
            pooling=pooling, last_weight=last_weight, max_length=max_length,
        ).double()
        y = _encode_anchors(
            reference_model, reference_tok, batch_text, device=device,
            pooling=pooling, last_weight=last_weight, max_length=max_length,
        ).double()
        if x.shape != y.shape:
            raise RuntimeError(f"Paired representation shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        sum_x += x.sum(dim=0)
        sum_y += y.sum(dim=0)
        sum_x2 += x.T @ x
        sum_y_sq += (y * y).sum(dim=0)
        sum_xy += x.T @ y
        count += int(x.shape[0])
        _emit(progress_callback, CalibrationProgress(
            "train anchors",
            min(start + len(batch_text), len(texts)),
            len(texts),
            f"paired samples={count:,}",
        ))
    mu_x = sum_x / count
    mu_y = sum_y / count
    cross = sum_xy / count - torch.outer(mu_x, mu_y)
    cov_x = sum_x2 / count - torch.outer(mu_x, mu_x)
    var_y = (sum_y_sq / count - mu_y.square()).clamp_min(1e-12)
    return mu_x, mu_y, cross, cov_x, var_y, count


def _encode_validation_pairs(
    source_model,
    reference_model,
    source_tok,
    reference_tok,
    texts: list[str],
    *,
    device: str,
    batch_size: int,
    pooling: str,
    last_weight: float,
    max_length: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        xs.append(_encode_anchors(
            source_model, source_tok, batch_text, device=device,
            pooling=pooling, last_weight=last_weight, max_length=max_length,
        ))
        ys.append(_encode_anchors(
            reference_model, reference_tok, batch_text, device=device,
            pooling=pooling, last_weight=last_weight, max_length=max_length,
        ))
        _emit(progress_callback, CalibrationProgress(
            "validation anchors",
            min(start + len(batch_text), len(texts)),
            len(texts),
        ))
    return torch.cat(xs, dim=0).double(), torch.cat(ys, dim=0).double()


def _apply_candidate(
    x: torch.Tensor,
    *,
    rotation: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    variance_scale: torch.Tensor,
    center_strength: float,
    variance_strength: float,
) -> torch.Tensor:
    rotated_uncentered = x @ rotation
    centered = (x - source_mean) @ rotation
    scale = 1.0 + (variance_scale - 1.0) * float(variance_strength)
    fully_centered = centered * scale + target_mean
    return rotated_uncentered + (fully_centered - rotated_uncentered) * float(center_strength)


def _score_alignment(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    cos = F.cosine_similarity(x.float(), y.float(), dim=-1).mean().item()
    rmse = torch.sqrt(torch.mean((x - y).square())).item()
    return float(cos), float(rmse)


def _select_recommended_strengths(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    rotation: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    variance_scale: torch.Tensor,
) -> tuple[float, float, float, float, float]:
    before_cos, _before_rmse = _score_alignment(x_val, y_val)
    candidates_center = (0.0, 0.25, 0.5, 0.75, 1.0)
    candidates_variance = (0.0, 0.25, 0.5, 1.0)
    best: tuple[float, float, float, float] | None = None
    for center in candidates_center:
        for variance in candidates_variance:
            aligned = _apply_candidate(
                x_val,
                rotation=rotation,
                source_mean=source_mean,
                target_mean=target_mean,
                variance_scale=variance_scale,
                center_strength=center,
                variance_strength=variance,
            )
            cos, rmse = _score_alignment(aligned, y_val)
            candidate = (cos, -rmse, center, variance)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    after_cos, neg_rmse, center, variance = best
    return float(center), float(variance), float(before_cos), float(after_cos), float(-neg_rmse)


def _validate_config(config: BridgeCalibrationConfig) -> None:
    if not 0.01 <= float(config.validation_fraction) <= 0.40:
        raise ValueError("validation_fraction must be in [0.01, 0.40]")
    if int(config.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(config.max_length) < 8:
        raise ValueError("max_length must be >= 8")
    if int(config.default_prompt_count) < 256:
        raise ValueError("default_prompt_count must be >= 256")
    if config.pooling not in {"mean", "last", "blend", "both"}:
        raise ValueError(f"Unsupported pooling mode: {config.pooling}")


def calibrate_text_encoder_bridge(
    config: BridgeCalibrationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> BridgeCalibrationResult:
    """Run calibration directly from Python/Jupyter and return a structured result."""

    started = time.perf_counter()
    config = config.normalized()
    _validate_config(config)
    if progress_callback is None:
        progress_callback = _ConsoleProgress()
    device = _device(config.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _emit(progress_callback, CalibrationProgress("corpus", message="loading / generating prompt corpus"))
    base_prompts, corpus_source = load_calibration_prompts(
        config.prompts,
        default_count=int(config.default_prompt_count),
        seed=int(config.seed),
    )
    if config.dump_prompts is not None:
        dump_path = Path(config.dump_prompts)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("\n".join(base_prompts) + "\n", encoding="utf-8")
    corpus_hash = _corpus_sha256(base_prompts)
    anchors = _expand_prompt_anchors(
        base_prompts,
        include_phrases=config.include_phrases,
        min_phrase_chars=int(config.min_phrase_chars),
    )
    if len(anchors) < 512:
        raise ValueError("Calibration needs at least 512 expanded anchors.")
    report = analyze_bridge_calibration_prompts(base_prompts)
    _emit(progress_callback, CalibrationProgress(
        "corpus",
        len(base_prompts),
        len(base_prompts),
        f"anchors={len(anchors):,}; duplicate_lines={report['duplicate_lines']}; source={corpus_source}",
    ))

    rng = random.Random(int(config.seed))
    rng.shuffle(anchors)
    val_count = max(64, int(len(anchors) * float(config.validation_fraction)))
    val_texts = anchors[:val_count]
    train_texts = anchors[val_count:]

    _emit(progress_callback, CalibrationProgress("tokenizers", message="loading source and reference tokenizers"))
    source_tok = AutoTokenizer.from_pretrained(str(config.source_tokenizer))
    reference_tok = AutoTokenizer.from_pretrained(str(config.reference_tokenizer))

    source_model = None
    reference_model = None
    try:
        _emit(progress_callback, CalibrationProgress(
            "models", message=f"loading source encoder on {device} ({dtype})"
        ))
        source_model = load_text_encoder_single_file(
            str(config.source_model), device=device, dtype=dtype, cache=False
        )
        _emit(progress_callback, CalibrationProgress(
            "models", message=f"loading reference encoder on {device} ({dtype})"
        ))
        reference_model = load_text_encoder_single_file(
            str(config.reference_model), device=device, dtype=dtype, cache=False
        )
        source_family = str(getattr(source_model, "_anima_text_encoder_family", "unknown"))
        target_family = str(getattr(reference_model, "_anima_text_encoder_family", "unknown"))
        dim = int(getattr(source_model.config, "hidden_size", 0))
        target_dim = int(getattr(reference_model.config, "hidden_size", 0))
        if dim != target_dim or dim <= 0:
            raise ValueError(
                "This v2 calibrator currently requires matching hidden sizes; "
                f"got source={dim}, reference={target_dim}. The profile format already reserves "
                "bridge.input_projection for future wider encoders."
            )

        mu_x, mu_y, cross, cov_x, var_y, train_count = _collect_train_statistics(
            source_model, reference_model, source_tok, reference_tok, train_texts,
            device=device, batch_size=int(config.batch_size), pooling=config.pooling,
            last_weight=float(config.last_weight), max_length=int(config.max_length),
            progress_callback=progress_callback,
        )
        _emit(progress_callback, CalibrationProgress(
            "solve", message=f"solving {dim}x{dim} orthogonal Procrustes SVD"
        ))
        u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
        rotation = u @ vh
        cov_rot = rotation.T @ cov_x @ rotation
        var_rot = torch.diagonal(cov_rot).clamp_min(1e-12)
        variance_scale = torch.sqrt(var_y / var_rot).clamp(0.25, 4.0)

        x_val, y_val = _encode_validation_pairs(
            source_model, reference_model, source_tok, reference_tok, val_texts,
            device=device, batch_size=int(config.batch_size), pooling=config.pooling,
            last_weight=float(config.last_weight), max_length=int(config.max_length),
            progress_callback=progress_callback,
        )
        center_strength, variance_strength, cos_before, cos_after, rmse_after = _select_recommended_strengths(
            x_val, y_val,
            rotation=rotation,
            source_mean=mu_x,
            target_mean=mu_y,
            variance_scale=variance_scale,
        )
        _emit(progress_callback, CalibrationProgress(
            "validation",
            int(x_val.shape[0]),
            int(x_val.shape[0]),
            f"cosine={cos_before:.6f}->{cos_after:.6f}; rmse={rmse_after:.6f}; "
            f"center={center_strength:.2f}; variance={variance_strength:.2f}",
        ))

        tensors: dict[str, torch.Tensor] = {
            "bridge.rotation": rotation.float().contiguous(),
            "bridge.source_mean": mu_x.float().contiguous(),
            "bridge.target_mean": mu_y.float().contiguous(),
            "bridge.variance_scale": variance_scale.float().contiguous(),
        }
        contains_encoder = bool(config.bundle_source_weights)
        if contains_encoder:
            _emit(progress_callback, CalibrationProgress("bundle", message="copying source encoder weights into profile"))
            source_model.to("cpu")
            for key, value in source_model.state_dict().items():
                tensors[f"encoder.{key}"] = value.detach().cpu().contiguous()

        source_config = source_model.config.to_dict() if hasattr(source_model.config, "to_dict") else {}
        reference_config = reference_model.config.to_dict() if hasattr(reference_model.config, "to_dict") else {}
        metadata = {
            "format": "anima_text_encoder_profile_v2",
            "artifact_kind": "aligned_encoder" if contains_encoder else "bridge_profile",
            "contains_encoder_weights": "true" if contains_encoder else "false",
            "bridge_type": "orthogonal_procrustes",
            "transform_semantics": "rotation_only_to_centered_target_v2",
            "source_family": source_family,
            "target_family": target_family,
            "source_model": str(config.source_model),
            "reference_model": str(config.reference_model),
            "source_tokenizer": str(config.source_tokenizer),
            "reference_tokenizer": str(config.reference_tokenizer),
            "source_hidden_size": str(dim),
            "target_hidden_size": str(target_dim),
            "source_architecture_fingerprint": encoder_config_fingerprint(source_model.config),
            "target_architecture_fingerprint": encoder_config_fingerprint(reference_model.config),
            "source_config_json": json.dumps(source_config, sort_keys=True, separators=(",", ":")),
            "reference_config_json": json.dumps(reference_config, sort_keys=True, separators=(",", ":")),
            "corpus_source": corpus_source,
            "calibration_corpus_sha256": corpus_hash,
            "corpus_lines": str(len(base_prompts)),
            "expanded_anchors": str(len(anchors)),
            "samples": str(train_count),
            "validation_samples": str(int(x_val.shape[0])),
            "pooling": f"blend:{config.last_weight:.6g}" if config.pooling == "blend" else str(config.pooling),
            "calibration_max_length": str(int(config.max_length)),
            "recommended_center_strength": f"{center_strength:.8g}",
            "recommended_variance_strength": f"{variance_strength:.8g}",
            "recommended_rms_strength": "0",
            # v4 final-runtime stability defaults are intentionally separate from
            # calibration-only pooled cosine optimisation.
            "recommended_delta_clip_ratio": "0.30",
            "recommended_token_rms_strength": "0.70",
            "recommended_token_rms_min_ratio": "0.85",
            "recommended_token_rms_max_ratio": "1.10",
            "runtime_recommended_center_strength": "1.0",
            "runtime_recommended_variance_strength": "0.0",
            "runtime_recommended_rms_strength": "0.0",
            "validation_cosine_before": f"{cos_before:.8g}",
            "validation_cosine_after": f"{cos_after:.8g}",
            "validation_rmse_after": f"{rmse_after:.8g}",
        }

        output = Path(config.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _emit(progress_callback, CalibrationProgress("save", message=f"writing {output}"))
        save_file(tensors, str(output), metadata=metadata)
        elapsed = time.perf_counter() - started
        result = BridgeCalibrationResult(
            output=output,
            artifact_kind=metadata["artifact_kind"],
            metadata=metadata,
            corpus_source=corpus_source,
            corpus_lines=len(base_prompts),
            expanded_anchors=len(anchors),
            train_samples=train_count,
            validation_samples=int(x_val.shape[0]),
            validation_cosine_before=cos_before,
            validation_cosine_after=cos_after,
            validation_rmse_after=rmse_after,
            recommended_center_strength=center_strength,
            recommended_variance_strength=variance_strength,
            elapsed_seconds=elapsed,
        )
        _emit(progress_callback, CalibrationProgress("done", 1, 1, result.summary().replace("\n", " | ")))
        return result
    finally:
        if config.cleanup_models:
            del source_model, reference_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--prompts", type=Path, default=None,
                        help="optional text corpus; built-in visual corpus is used when omitted")
    parser.add_argument("--dump-prompts", type=Path, default=None,
                        help="write the exact base prompt corpus used for reproducibility")
    parser.add_argument("--default-prompt-count", type=int, default=4096)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pooling", choices=("mean", "last", "blend", "both"), default="both")
    parser.add_argument("--last-weight", type=float, default=0.65)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-phrases", action="store_true")
    parser.add_argument("--min-phrase-chars", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=3571)
    parser.add_argument("--bundle-source-weights", action="store_true",
                        help="embed source text-encoder weights as encoder.* so output can be used as encoder_path")
    parser.add_argument("--keep-models", action="store_true",
                        help="do not explicitly clear loaded encoder models after calibration")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = BridgeCalibrationConfig(
        source_model=args.source_model,
        source_tokenizer=args.source_tokenizer,
        reference_model=args.reference_model,
        reference_tokenizer=args.reference_tokenizer,
        prompts=args.prompts,
        dump_prompts=args.dump_prompts,
        default_prompt_count=args.default_prompt_count,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        pooling=args.pooling,
        last_weight=args.last_weight,
        max_length=args.max_length,
        include_phrases=not args.no_phrases,
        min_phrase_chars=args.min_phrase_chars,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        bundle_source_weights=args.bundle_source_weights,
        cleanup_models=not args.keep_models,
    )
    result = calibrate_text_encoder_bridge(config)
    print(result.summary())


if __name__ == "__main__":
    main()
