from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimaComponents:
    """Sources for Anima single-file style loading.

    ``model_path`` is the transformer checkpoint. ``text_encoder_path`` and
    ``vae_path`` are optional override files used by
    ``AnimaPipeline.from_multiple_files``. When they are omitted, loading.py
    falls back to the hardcoded Anima defaults used by ``from_single_file``.
    Tokenizers are still resolved from the fixed Anima tokenizer sources.
    """

    model_path: str
    text_encoder_path: str | None = None
    vae_path: str | None = None


@dataclass(frozen=True)
class AnimaLoaderOptions:
    local_files_only: bool
    cache_dir: str | None = None
    force_download: bool = False
    token: str | bool | None = None
    revision: str | None = None
    proxies: dict[str, str] | None = None
    cache_components: bool = True
    cache_transformer: bool = False


@dataclass(frozen=True)
class AnimaRuntimeOptions:
    device: str = "auto"
    dtype: str = "auto"
    text_encoder_dtype: str = "auto"
    text_encoder_max_sequence_length: int | None = None
