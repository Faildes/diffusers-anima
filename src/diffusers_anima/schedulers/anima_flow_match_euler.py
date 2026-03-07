"""Anima-specific scheduler wrapper on top of FlowMatch Euler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diffusers import FlowMatchEulerDiscreteScheduler

# Diffusers custom-component loading checks for `SchedulerMixin` by name in this module.
from diffusers import SchedulerMixin as SchedulerMixin  # noqa: F401

from ..pipelines.anima.constants import FORGE_BETA_ALPHA, FORGE_BETA_BETA


# ---------------------------------------------------------------------------
# sampler / schedule support
# ---------------------------------------------------------------------------

_LEGACY_SAMPLER_ALIASES = {
    "euler_a_rf": "euler_a",
    "euler_ancestral_rf": "euler_ancestral",
}

_SUPPORTED_CANONICAL_SAMPLERS = (
    "flowmatch_euler",
    "euler",
    "euler_a",
    "euler_ancestral",
    "er_sde",
)

SUPPORTED_ANIMA_SAMPLERS = (
    *_SUPPORTED_CANONICAL_SAMPLERS,
    *_LEGACY_SAMPLER_ALIASES.keys(),
)

SUPPORTED_ANIMA_SIGMA_SCHEDULES = ("beta", "uniform", "simple", "normal")

# er_sde 用に保持しておく solver 種別
SUPPORTED_ER_SDE_SOLVER_TYPES = ("midpoint", "heun")


def _normalize_sampler_name(sampler: str) -> str:
    sampler = str(sampler).strip().lower()
    return _LEGACY_SAMPLER_ALIASES.get(sampler, sampler)


def _validate_anima_sampler_config(*, sampler: str, sigma_schedule: str) -> None:
    """Validate Anima sampler/sigma schedule combinations."""
    sampler = _normalize_sampler_name(sampler)

    if sampler not in _SUPPORTED_CANONICAL_SAMPLERS:
        raise ValueError(
            "`sampler` must be one of: "
            "flowmatch_euler, euler, euler_a, euler_ancestral, er_sde "
            "(legacy aliases: euler_a_rf, euler_ancestral_rf)."
        )

    if sigma_schedule not in SUPPORTED_ANIMA_SIGMA_SCHEDULES:
        raise ValueError(
            "`sigma_schedule` must be one of: beta, uniform, simple, normal."
        )

    if sampler == "flowmatch_euler" and sigma_schedule != "uniform":
        raise ValueError("`flowmatch_euler` requires `sigma_schedule='uniform'`.")


def _validate_er_sde_config(
    *,
    solver_type: str,
    max_stage: int,
    s_churn: float,
    s_tmin: float,
    s_tmax: float | None,
) -> None:
    solver_type = str(solver_type).strip().lower()
    if solver_type not in SUPPORTED_ER_SDE_SOLVER_TYPES:
        raise ValueError(
            "`solver_type` must be one of: " + ", ".join(SUPPORTED_ER_SDE_SOLVER_TYPES)
        )

    if int(max_stage) < 1:
        raise ValueError("`max_stage` must be >= 1.")

    if float(s_churn) < 0.0:
        raise ValueError("`s_churn` must be >= 0.0.")

    if float(s_tmin) < 0.0:
        raise ValueError("`s_tmin` must be >= 0.0.")

    if s_tmax is not None and float(s_tmax) < float(s_tmin):
        raise ValueError("`s_tmax` must be >= `s_tmin` when provided.")


def _scheduler_config_get(config: Any, *, key: str, default: Any) -> Any:
    """Read scheduler config from dict-like or attribute-like objects."""
    if hasattr(config, "get"):
        value = config.get(key, None)
    else:
        value = getattr(config, key, None)
    return default if value is None else value


@dataclass(frozen=True)
class AnimaSamplingConfig:
    sampler: str
    sigma_schedule: str
    beta_alpha: float
    beta_beta: float
    eta: float
    s_noise: float
    solver_type: str
    max_stage: int
    s_churn: float
    s_tmin: float
    s_tmax: float | None


class AnimaFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """FlowMatch Euler scheduler extended with Anima sampling metadata.

    Sampling knobs are serialized into scheduler config and restored via
    `from_config(...)`. This class does not implement the custom samplers
    themselves; it stores validated runtime metadata for pipeline-side dispatch.
    """

    SUPPORTED_SAMPLERS = SUPPORTED_ANIMA_SAMPLERS
    SUPPORTED_SIGMA_SCHEDULES = SUPPORTED_ANIMA_SIGMA_SCHEDULES
    SUPPORTED_ER_SDE_SOLVER_TYPES = SUPPORTED_ER_SDE_SOLVER_TYPES

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int | None = 256,
        max_image_seq_len: int | None = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool | None = False,
        use_exponential_sigmas: bool | None = False,
        use_beta_sigmas: bool | None = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        sampler: str = "euler_a",
        sigma_schedule: str = "beta",
        beta_alpha: float = FORGE_BETA_ALPHA,
        beta_beta: float = FORGE_BETA_BETA,
        eta: float = 1.0,
        s_noise: float = 1.0,
        solver_type: str = "midpoint",
        max_stage: int = 2,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float | None = None,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=invert_sigmas,
            shift_terminal=shift_terminal,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
            stochastic_sampling=stochastic_sampling,
        )

        self.set_sampling_config(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
            solver_type=solver_type,
            max_stage=max_stage,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
        )

    def set_sampling_config(
        self,
        *,
        sampler: str = "euler_a",
        sigma_schedule: str = "beta",
        beta_alpha: float = FORGE_BETA_ALPHA,
        beta_beta: float = FORGE_BETA_BETA,
        eta: float = 1.0,
        s_noise: float = 1.0,
        solver_type: str = "midpoint",
        max_stage: int = 2,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float | None = None,
    ) -> None:
        sampler = _normalize_sampler_name(sampler)
        solver_type = str(solver_type).strip().lower()

        _validate_anima_sampler_config(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
        )
        _validate_er_sde_config(
            solver_type=solver_type,
            max_stage=max_stage,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
        )

        self.register_to_config(
            sampler=sampler,
            sigma_schedule=str(sigma_schedule),
            beta_alpha=float(beta_alpha),
            beta_beta=float(beta_beta),
            eta=float(eta),
            s_noise=float(s_noise),
            solver_type=solver_type,
            max_stage=int(max_stage),
            s_churn=float(s_churn),
            s_tmin=float(s_tmin),
            s_tmax=None if s_tmax is None else float(s_tmax),
        )

    def get_sampling_config(self) -> AnimaSamplingConfig:
        config = self.config

        sampler = _normalize_sampler_name(
            str(_scheduler_config_get(config, key="sampler", default="euler_a"))
        )
        sigma_schedule = str(
            _scheduler_config_get(config, key="sigma_schedule", default="beta")
        )
        beta_alpha = float(
            _scheduler_config_get(config, key="beta_alpha", default=FORGE_BETA_ALPHA)
        )
        beta_beta = float(
            _scheduler_config_get(config, key="beta_beta", default=FORGE_BETA_BETA)
        )
        eta = float(_scheduler_config_get(config, key="eta", default=1.0))
        s_noise = float(_scheduler_config_get(config, key="s_noise", default=1.0))
        solver_type = str(
            _scheduler_config_get(config, key="solver_type", default="midpoint")
        ).strip().lower()
        max_stage = int(_scheduler_config_get(config, key="max_stage", default=2))
        s_churn = float(_scheduler_config_get(config, key="s_churn", default=0.0))
        s_tmin = float(_scheduler_config_get(config, key="s_tmin", default=0.0))
        s_tmax = _scheduler_config_get(config, key="s_tmax", default=None)

        _validate_anima_sampler_config(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
        )
        _validate_er_sde_config(
            solver_type=solver_type,
            max_stage=max_stage,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
        )

        return AnimaSamplingConfig(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
            solver_type=solver_type,
            max_stage=max_stage,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=None if s_tmax is None else float(s_tmax),
        )