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
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
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
    NativePromptGroup,
    build_training_groups,
    corpus_sha256,
    preview_training_groups,
    read_prompt_lines,
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
    native_final_layer_logit_bias: float = 6.0
    pair_fraction: float = 0.35
    binding_pairs: int = 256
    # Losses.  Compatibility is important, but source geometry deliberately
    # remains substantial so 0.8B-only distinctions are not collapsed into 0.6B.
    anima_compat_weight: float = 1.00
    source_geometry_weight: float = 0.75
    token_geometry_weight: float = 0.50
    knowledge_gain_weight: float = 0.75
    distribution_weight: float = 0.20
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

    def summary(self) -> str:
        return (
            f"native_encoder: {self.output}\n"
            f"steps={self.steps}; corpus={self.corpus_lines}; rows={self.training_rows}; "
            f"final_loss={self.final_loss:.6f}; elapsed={self.elapsed_seconds:.1f}s"
        )


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(dict(payload))
        return
    stage = payload.pop("stage", "train")
    print(f"[anima-native:{stage}] " + " ".join(f"{k}={v}" for k, v in payload.items()))


def make_jupyter_progress_callback() -> ProgressCallback:
    try:
        from IPython.display import HTML, display
    except Exception:
        return lambda payload: _emit(None, **payload)
    handle: dict[str, Any] = {"display": None}

    def callback(payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage", "train"))
        step = payload.get("step")
        total = payload.get("total")
        loss = payload.get("loss")
        pct = 0.0
        if isinstance(step, int) and isinstance(total, int) and total > 0:
            pct = 100.0 * step / total
        line = f"<b>{stage}</b>"
        if step is not None and total is not None:
            line += f" &nbsp; {step}/{total} ({pct:.1f}%)"
        if loss is not None:
            line += f" &nbsp; loss={float(loss):.6f}"
        details = " ".join(
            f"{k}={v}" for k, v in payload.items()
            if k not in {"stage", "step", "total", "loss"}
        )
        html = HTML(
            f'<div style="font-family:monospace">{line}<br>{details}'
            f'<div style="height:8px;background:#ddd;border-radius:4px;overflow:hidden;margin-top:4px">'
            f'<div style="height:100%;width:{pct:.2f}%;background:#4c8bf5"></div></div></div>'
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


def _tokenize(tokenizer: Any, texts: Sequence[str], *, device: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
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
        "training_policy": "dual_teacher_knowledge_preserving_native_v1",
        "knowledge_policy": "preserve_qwen35_geometry_relax_qwen3_reference_on_gain",
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

    lines = read_prompt_lines(cfg.prompts)
    groups = build_training_groups(
        lines,
        pair_fraction=float(cfg.pair_fraction),
        binding_pairs=int(cfg.binding_pairs),
        seed=int(cfg.seed),
    )
    preview = preview_training_groups(groups)
    training_rows = int(preview["rows"])
    _emit(progress_callback, stage="corpus", **preview)

    source_tok = AutoTokenizer.from_pretrained(str(cfg.source_tokenizer))
    ref_tok = AutoTokenizer.from_pretrained(str(cfg.reference_tokenizer))
    if source_tok.pad_token_id is None:
        source_tok.pad_token = source_tok.eos_token
    if ref_tok.pad_token_id is None:
        ref_tok.pad_token = ref_tok.eos_token

    if cfg.resume_native:
        loaded = load_text_encoder_single_file(str(cfg.resume_native), device=device, dtype=dtype, cache=False)
        if not isinstance(loaded, AnimaNativeQwen35Encoder):
            raise ValueError("resume_native must be an anima_native_text_encoder_v1 checkpoint")
        student = loaded
        fresh_native = False
    else:
        if not cfg.source_model:
            raise ValueError("source_model is required when resume_native is not supplied")
        backbone = load_text_encoder_single_file(str(cfg.source_model), device=device, dtype=dtype, cache=False)
        if isinstance(backbone, AnimaNativeQwen35Encoder):
            raise ValueError("source_model should be the raw Qwen3.5 source; use resume_native for native checkpoints")
        num_layers = int(getattr(backbone.config, "num_hidden_layers", 24))
        hidden_size = int(getattr(backbone.config, "hidden_size", 1024))
        layer_indices = cfg.native_layer_indices
        if layer_indices is None:
            layer_indices = (
                max(1, round(num_layers * 0.25)),
                max(1, round(num_layers * 0.50)),
                max(1, round(num_layers * 0.75)),
                num_layers,
            )
        head_cfg = AnimaNativeHeadConfig(
            hidden_size=hidden_size,
            intermediate_size=int(cfg.native_intermediate_size),
            layer_indices=tuple(int(x) for x in layer_indices),
            norm_eps=float(getattr(backbone.config, "rms_norm_eps", 1e-6)),
            final_layer_logit_bias=float(cfg.native_final_layer_logit_bias),
        )
        student = AnimaNativeQwen35Encoder(backbone, AnimaNativeQwen35Head(head_cfg)).to(device=device, dtype=dtype)
        fresh_native = True

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

    reference = load_text_encoder_single_file(
        str(cfg.reference_model), device=ref_device, dtype=ref_dtype, cache=False
    )
    reference.eval().requires_grad_(False)

    student.native_head.requires_grad_(True)
    backbone_params = _configure_trainable_backbone(student, int(cfg.train_last_n_layers))
    if int(cfg.train_last_n_layers) > 0 and cfg.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        # Frozen embeddings feed trainable upper layers.  Some HF checkpointing
        # implementations require the boundary activation itself to carry grad.
        student.enable_input_require_grads()
    student.train()

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
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    rows_per_epoch = max(1, training_rows)
    micro_steps_per_epoch = max(1, math.ceil(rows_per_epoch / max(1, int(cfg.batch_size))))
    optimizer_steps_per_epoch = max(
        1, math.ceil(micro_steps_per_epoch / max(1, int(cfg.gradient_accumulation_steps)))
    )
    total_steps = int(cfg.max_steps) if int(cfg.max_steps) > 0 else optimizer_steps_per_epoch * max(1, int(cfg.epochs))
    warmup = max(0, round(total_steps * max(0.0, float(cfg.warmup_fraction))))

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

    for epoch in range(max(1, int(cfg.epochs))):
        if global_step >= total_steps:
            break
        for group_batch in _iter_group_batches(groups, row_batch_size=max(2, int(cfg.batch_size)), rng=rng):
            if global_step >= total_steps:
                break
            texts, pairs = _flatten_group_batch(group_batch)
            src = _tokenize(source_tok, texts, device=device, max_length=int(cfg.max_length))
            ref = _tokenize(ref_tok, texts, device=ref_device, max_length=int(cfg.max_length))

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
                return_details=True,
            )
            source_hidden = details["final_hidden"]

            with torch.no_grad():
                ref_out = reference(
                    input_ids=ref["input_ids"],
                    attention_mask=ref["attention_mask"],
                    use_cache=False,
                )
                ref_hidden = ref_out[0] if isinstance(ref_out, tuple) else ref_out.last_hidden_state

            student_pool = _masked_pool(native_hidden, src["attention_mask"])
            source_pool = _masked_pool(source_hidden.detach(), src["attention_mask"])
            ref_pool = _masked_pool(ref_hidden, ref["attention_mask"]).to(device=device, dtype=student_pool.dtype)

            sample_weights = torch.ones(student_pool.shape[0], device=device, dtype=torch.float32)
            knowledge_losses: list[torch.Tensor] = []
            gains: list[torch.Tensor] = []
            for a, b, _category in pairs:
                ds = _cosine_distance(source_pool[a:a+1], source_pool[b:b+1]).squeeze(0)
                dr = _cosine_distance(ref_pool[a:a+1], ref_pool[b:b+1]).squeeze(0)
                gain = torch.relu(ds - dr) / ds.clamp_min(1e-4)
                gain = gain.clamp(0.0, 1.0).detach()
                dstudent = _cosine_distance(student_pool[a:a+1], student_pool[b:b+1]).squeeze(0)
                knowledge_losses.append(gain * F.smooth_l1_loss(dstudent, ds.detach()))
                gains.append(gain)
                relaxed = 1.0 - float(cfg.reference_relax_for_gain) * float(gain.item())
                sample_weights[a] *= relaxed
                sample_weights[b] *= relaxed

            compat_each = _cosine_distance(student_pool, ref_pool)
            compat_each = compat_each + 0.25 * _normalized_mse_each(student_pool, ref_pool)
            # Apply the same knowledge-aware relaxation to the complete 0.6B
            # compatibility term so 0.8B-only distinctions are not erased by
            # the normalised-MSE component either.
            compat = (compat_each * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
            source_geometry = _pairwise_geometry_loss(student_pool, source_pool)
            token_geometry = _token_geometry_loss(native_hidden, source_hidden.detach(), src["attention_mask"])
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

            if bootstrap_target is not None:
                distribution_target = bootstrap_target
                distribution_mask = src["attention_mask"]
            else:
                distribution_target = ref_hidden.to(device=device, dtype=native_hidden.dtype)
                distribution_mask = ref["attention_mask"].to(device)
            distribution = _distribution_loss(
                native_hidden,
                src["attention_mask"],
                distribution_target,
                distribution_mask,
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
                + float(cfg.distribution_weight) * distribution
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
                "distribution": float(distribution.detach().item()),
                "bootstrap": float(bootstrap_loss.detach().item()),
                "gain": float(torch.stack(gains).mean().item()) if gains else 0.0,
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

                if global_step == 1 or global_step % max(1, int(cfg.log_every)) == 0 or global_step >= total_steps:
                    denom = float(max(1, int(cfg.log_every) if global_step > 1 else 1))
                    _emit(
                        progress_callback,
                        stage="train",
                        step=global_step,
                        total=total_steps,
                        epoch=epoch + 1,
                        loss=f"{last_loss:.6f}",
                        lr=f"{scheduler.get_last_lr()[0]:.3e}",
                        gain=f"{values['gain']:.4f}",
                    )
                    running.clear()

            del source_out, native_hidden, details, source_hidden, ref_out, ref_hidden

        # Flush a partial accumulation at the epoch boundary.
        if accum > 0 and global_step < total_steps:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                max_norm=float(cfg.max_grad_norm),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            global_step += 1

    output = Path(cfg.output)
    metadata = _save_native_encoder(
        student,
        output,
        config=cfg,
        corpus_lines=lines,
        training_rows=training_rows,
        steps=global_step,
        final_loss=last_loss,
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
    )
    _emit(progress_callback, stage="done", step=global_step, total=global_step, loss=f"{last_loss:.6f}", output=output)
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
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--train-last-n-layers", type=int, default=0)
    p.add_argument("--native-intermediate-size", type=int, default=1536)
    p.add_argument("--native-layer-indices", default=None, help="e.g. 6,12,18,24")
    p.add_argument("--pair-fraction", type=float, default=0.35)
    p.add_argument("--binding-pairs", type=int, default=256)
    p.add_argument("--seed", type=int, default=3571)
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
        head_lr=a.head_lr,
        backbone_lr=a.backbone_lr,
        train_last_n_layers=a.train_last_n_layers,
        native_intermediate_size=a.native_intermediate_size,
        native_layer_indices=_parse_layer_indices(a.native_layer_indices),
        pair_fraction=a.pair_fraction,
        binding_pairs=a.binding_pairs,
        seed=a.seed,
    )
    result = train_anima_native_text_encoder(cfg)
    print(result.summary())


if __name__ == "__main__":
    main()
