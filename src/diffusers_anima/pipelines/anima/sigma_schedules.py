"""Sigma schedule construction for Anima custom samplers."""

from __future__ import annotations

import numpy as np
import torch

from .constants import ANIMA_SAMPLING_MULTIPLIER, FORGE_BETA_ALPHA, FORGE_BETA_BETA


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    numerator = t * alpha
    denominator = t * (alpha - 1.0) + 1.0
    return numerator / denominator


def _build_base_sigmas(
    *,
    num_train_timesteps: int,
    shift: float,
    device: str,
) -> torch.Tensor:
    # 0..1 の正規化時間で統一
    t = (
        torch.arange(1, num_train_timesteps + 1, dtype=torch.float32, device=device)
        / float(num_train_timesteps)
    )
    base_sigmas = _time_snr_shift(shift, t).to(dtype=torch.float32)
    return base_sigmas


def _sanitize_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    sigmas = sigmas.detach().to(dtype=torch.float32)
    sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    if sigmas.numel() == 0:
        return sigmas

    if sigmas[0] < sigmas[-1]:
        sigmas = torch.flip(sigmas, dims=[0])

    fixed = sigmas.clone()
    for i in range(1, fixed.numel()):
        if fixed[i] > fixed[i - 1]:
            fixed[i] = fixed[i - 1]

    if fixed[-1].item() != 0.0:
        fixed = torch.cat([fixed, fixed.new_zeros(1)], dim=0)

    return fixed


def build_simple_sigmas(base_sigmas: torch.Tensor, *, steps: int) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    idx = torch.linspace(
        base_sigmas.numel() - 1,
        0,
        steps,
        device=base_sigmas.device,
        dtype=torch.float32,
    ).round().long()

    sigmas = base_sigmas.index_select(0, idx)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
    return _sanitize_sigmas(sigmas)


def build_beta_sigmas(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    beta_alpha: float,
    beta_beta: float,
    device: str,
) -> torch.Tensor:
    from scipy import stats

    base_sigmas = _build_base_sigmas(
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        device=device,
    )

    total_timesteps = len(base_sigmas) - 1
    ts = 1.0 - np.linspace(0.0, 1.0, num_inference_steps, endpoint=False)
    mapped = stats.beta.ppf(ts, beta_alpha, beta_beta) * float(total_timesteps)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=float(total_timesteps), neginf=0.0)
    indices = np.clip(np.rint(mapped).astype(np.int64), 0, total_timesteps)

    sigmas = []
    last_index = None
    for index in indices:
        if last_index is None or int(index) != last_index:
            sigmas.append(float(base_sigmas[int(index)].item()))
        last_index = int(index)

    sigmas.append(0.0)
    return _sanitize_sigmas(torch.tensor(sigmas, device=device, dtype=torch.float32))


def build_normal_sigmas(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    device: str,
) -> torch.Tensor:
    base_sigmas = _build_base_sigmas(
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        device=device,
    )

    sigma_max = base_sigmas[-1]
    sigma_min = base_sigmas[0]

    sigmas = torch.linspace(
        sigma_max,
        sigma_min,
        num_inference_steps,
        device=device,
        dtype=torch.float32,
    )
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float32)], dim=0)
    return _sanitize_sigmas(sigmas)


def build_sampling_sigmas(
    scheduler: object,
    *,
    num_inference_steps: int,
    sigma_schedule: str,
    beta_alpha: float = FORGE_BETA_ALPHA,
    beta_beta: float = FORGE_BETA_BETA,
    device: str,
) -> torch.Tensor:
    shift = float(scheduler.config.shift)
    num_train_timesteps = int(scheduler.config.num_train_timesteps)

    if sigma_schedule == "normal":
        return build_normal_sigmas(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            device=device,
        )

    if sigma_schedule == "simple":
        base_sigmas = _build_base_sigmas(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            device=device,
        )
        return build_simple_sigmas(base_sigmas, steps=num_inference_steps)

    if sigma_schedule == "beta":
        return build_beta_sigmas(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            device=device,
        )

    scheduler.set_timesteps(num_inference_steps, device=device)
    return _sanitize_sigmas(scheduler.sigmas.to(device=device, dtype=torch.float32))