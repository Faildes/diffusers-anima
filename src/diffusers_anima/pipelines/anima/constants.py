from __future__ import annotations

from typing import Any

import torch


LOCAL_QWEN_TOKENIZER_DIR = "prompt_tokenizer_qwen"
LOCAL_T5_TOKENIZER_DIR = "prompt_tokenizer_t5"


ANIMA_VAE_CONFIG: dict[str, Any] = {
    "_class_name": "AutoencoderKLQwenImage",
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "latents_mean": [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ],
    "latents_std": [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    ],
    "num_res_blocks": 2,
    "z_dim": 16,
}

DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DTYPE_NAME_MAP = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
HF_URL_PREFIXES = (
    "https://huggingface.co/",
    "huggingface.co/",
    "https://hf.co/",
    "hf.co/",
)

# Forge's Anima config uses FLOW with multiplier=1.0, i.e. the model receives sigma-space timesteps.
ANIMA_SAMPLING_MULTIPLIER = 1.0
FORGE_BETA_ALPHA = 0.6
FORGE_BETA_BETA = 0.6

QWEN3_06B_CONFIG: dict[str, Any] = {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_bias": False,
}

# Official Qwen3.5-0.8B(-Base) text-backbone configuration.  The released
# checkpoint is multimodal, but Anima only needs ``model.language_model``.
# Keeping this as a text-only config avoids loading the vision tower.
QWEN35_08B_TEXT_CONFIG: dict[str, Any] = {
    "vocab_size": 248320,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "max_position_embeddings": 262144,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
    "mlp_only_layers": [],
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "use_cache": False,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    },
}
