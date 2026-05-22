"""ER-SDE scheduler-side metadata for Anima.

The actual denoising algorithm lives in ``pipelines/anima/sampler_er_sde.py``.
This module only keeps ER-SDE-specific names and config validation separate from
``anima_flow_match_euler.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

ER_SDE_SAMPLER_NAME = "er_sde"
DEFAULT_ER_SDE_MAX_STAGE = 3


def validate_er_sde_max_stage(er_sde_max_stage: int) -> None:
    if er_sde_max_stage < 1 or er_sde_max_stage > 3:
        raise ValueError("`er_sde_max_stage` must be one of: 1, 2, 3.")


@dataclass(frozen=True)
class AnimaERSDESamplingConfig:
    """ER-SDE-specific sampling options stored on the Anima scheduler config."""

    max_stage: int = DEFAULT_ER_SDE_MAX_STAGE
