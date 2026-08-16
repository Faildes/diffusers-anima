#!/usr/bin/env python3
"""Calibrate Qwen3.5 (or another 1024-d encoder) into Anima's Qwen3-0.6B space.

This is not gradient training.  It collects paired prompt/phrase representations,
solves one orthogonal Procrustes map, and stores centering/variance statistics in
one small safetensors artifact.

Example:
  python scripts/calibrate_text_encoder_bridge.py \
    --source-model qwen35_model.safetensors \
    --source-tokenizer Qwen/Qwen3.5-0.8B-Base \
    --reference-model qwen3_06b.safetensors \
    --reference-tokenizer Qwen/Qwen3-0.6B-Base \
    --prompts prompts.txt \
    --output qwen35_08b_to_qwen3_06b.safetensors
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusers_anima.pipelines.anima.loading import load_text_encoder_single_file  # noqa: E402


def _device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_prompts(path: Path, *, include_phrases: bool, min_phrase_chars: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not include_phrases:
        return prompts
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


def _encode_anchors(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: str,
    dtype: torch.dtype,
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
            # Mean anchors align global prompt geometry, while last-token anchors
            # align the causal endpoint representations that carry strong phrase
            # semantics.  Both use the same Procrustes map.
            anchors = torch.cat([mean, last], dim=0)
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
    return anchors.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--pooling",
        choices=("mean", "last", "blend", "both"),
        default="both",
        help="paired anchor representation; 'both' combines mean and causal endpoint anchors",
    )
    parser.add_argument("--last-weight", type=float, default=0.65, help="used only with --pooling blend")
    parser.add_argument("--max-length", type=int, default=2048, help="paired calibration context length")
    parser.add_argument("--no-phrases", action="store_true")
    parser.add_argument("--min-phrase-chars", type=int, default=3)
    args = parser.parse_args()

    device = _device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prompts = _load_prompts(
        args.prompts,
        include_phrases=not args.no_phrases,
        min_phrase_chars=args.min_phrase_chars,
    )
    if len(prompts) < 32:
        raise ValueError("Calibration needs at least 32 prompt/phrase anchors; thousands are recommended.")

    source_tok = AutoTokenizer.from_pretrained(args.source_tokenizer)
    reference_tok = AutoTokenizer.from_pretrained(args.reference_tokenizer)
    source_model = load_text_encoder_single_file(
        args.source_model, device=device, dtype=dtype, cache=False
    )
    reference_model = load_text_encoder_single_file(
        args.reference_model, device=device, dtype=dtype, cache=False
    )
    source_family = str(getattr(source_model, "_anima_text_encoder_family", "unknown"))
    target_family = str(getattr(reference_model, "_anima_text_encoder_family", "unknown"))

    dim = int(getattr(source_model.config, "hidden_size", 0))
    target_dim = int(getattr(reference_model.config, "hidden_size", 0))
    if dim != target_dim or dim <= 0:
        raise ValueError(f"This bridge calibrator requires matching hidden sizes, got {dim} and {target_dim}.")

    sum_x = torch.zeros(dim, dtype=torch.float64)
    sum_y = torch.zeros(dim, dtype=torch.float64)
    sum_x2 = torch.zeros((dim, dim), dtype=torch.float64)
    sum_y_sq = torch.zeros(dim, dtype=torch.float64)
    sum_xy = torch.zeros((dim, dim), dtype=torch.float64)
    count = 0

    for start in range(0, len(prompts), args.batch_size):
        texts = prompts[start : start + args.batch_size]
        x = _encode_anchors(
            source_model,
            source_tok,
            texts,
            device=device,
            dtype=dtype,
            pooling=args.pooling,
            last_weight=args.last_weight,
            max_length=args.max_length,
        ).double()
        y = _encode_anchors(
            reference_model,
            reference_tok,
            texts,
            device=device,
            dtype=dtype,
            pooling=args.pooling,
            last_weight=args.last_weight,
            max_length=args.max_length,
        ).double()
        if x.shape != y.shape:
            raise RuntimeError(f"Paired representation shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        sum_x += x.sum(dim=0)
        sum_y += y.sum(dim=0)
        sum_x2 += x.T @ x
        sum_y_sq += (y * y).sum(dim=0)
        sum_xy += x.T @ y
        count += int(x.shape[0])
        if start == 0 or (start // args.batch_size) % 50 == 0:
            print(f"[bridge-calibration] {min(start + len(texts), len(prompts))}/{len(prompts)}")

    mu_x = sum_x / count
    mu_y = sum_y / count
    cross = sum_xy / count - torch.outer(mu_x, mu_y)
    cov_x = sum_x2 / count - torch.outer(mu_x, mu_x)
    var_y = (sum_y_sq / count - mu_y.square()).clamp_min(1e-12)

    print("[bridge-calibration] solving 1024x1024 orthogonal Procrustes SVD...")
    u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    cov_rot = rotation.T @ cov_x @ rotation
    var_rot = torch.diagonal(cov_rot).clamp_min(1e-12)
    variance_scale = torch.sqrt(var_y / var_rot).clamp(0.25, 4.0)

    centered_mse = (
        torch.trace(cov_x)
        + var_y.sum()
        - 2.0 * torch.trace(rotation.T @ cross)
    ).clamp_min(0.0)
    centered_rmse = torch.sqrt(centered_mse / dim).item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "rotation": rotation.float().contiguous(),
            "source_mean": mu_x.float().contiguous(),
            "target_mean": mu_y.float().contiguous(),
            "variance_scale": variance_scale.float().contiguous(),
        },
        str(args.output),
        metadata={
            "format": "anima_text_encoder_bridge_v1",
            "source_family": source_family,
            "target_family": target_family,
            "samples": str(count),
            "pooling": (
                f"blend:{args.last_weight:.6g}" if args.pooling == "blend" else str(args.pooling)
            ),
            "calibration_max_length": str(int(args.max_length)),
            "centered_rmse": f"{centered_rmse:.8g}",
        },
    )
    print(f"[bridge-calibration] wrote {args.output}")
    print(f"[bridge-calibration] samples={count} centered_rmse={centered_rmse:.6g}")


if __name__ == "__main__":
    main()
