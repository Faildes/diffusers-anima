from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/diffusers_anima/pipelines/anima/anima_native_text_encoder.py"


def _load_native_module():
    # Keep this regression test independent of the installed Transformers
    # version.  The wrapper contract under test only needs this tiny output
    # container at import/forward time.
    transformers = types.ModuleType("transformers")
    modeling_outputs = types.ModuleType("transformers.modeling_outputs")

    class BaseModelOutputWithPast:
        def __init__(
            self,
            *,
            last_hidden_state,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        ):
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states
            self.attentions = attentions

    modeling_outputs.BaseModelOutputWithPast = BaseModelOutputWithPast
    transformers.modeling_outputs = modeling_outputs
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("transformers.modeling_outputs", modeling_outputs)

    spec = importlib.util.spec_from_file_location("anima_native_runtime_contract_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _DummyBackbone(nn.Module):
    def __init__(self, dim: int = 8, layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(32, dim)
        self.proj = nn.Linear(dim, dim)
        self.config = types.SimpleNamespace(hidden_size=dim, num_hidden_layers=layers)
        self._gradient_checkpointing = False
        self.layers = layers

    @property
    def dtype(self):
        return self.embed.weight.dtype

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def is_gradient_checkpointing(self):
        return self._gradient_checkpointing

    def get_input_embeddings(self):
        return self.embed

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        hidden_states = [hidden]
        for _ in range(self.layers):
            hidden = self.proj(hidden)
            hidden_states.append(hidden)
        return types.SimpleNamespace(hidden_states=tuple(hidden_states), attentions=None)


def _make_encoder(dtype=torch.float32):
    module = _load_native_module()
    backbone = _DummyBackbone()
    cfg = module.AnimaNativeHeadConfig(
        hidden_size=8,
        intermediate_size=12,
        layer_indices=(1, 2, 3, 4),
    )
    encoder = module.AnimaNativeQwen35Encoder(backbone, module.AnimaNativeQwen35Head(cfg))
    return encoder.to(dtype=dtype)


def test_native_wrapper_exposes_diffusers_dtype_and_device_contract():
    encoder = _make_encoder(dtype=torch.float64)
    assert encoder.dtype == torch.float64
    assert encoder.device == torch.device("cpu")
    assert encoder.main_input_name == "input_ids"

    # This is the access pattern that crashed DiffusionPipeline.to().
    encoder.to(torch.device("cpu"), None)
    assert encoder.dtype == torch.float64
    assert encoder.device == torch.device("cpu")


def test_native_wrapper_keeps_forward_output_and_checkpointing_contract():
    encoder = _make_encoder()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)

    out = encoder(input_ids=ids, attention_mask=mask)
    assert out.last_hidden_state.shape == (1, 3, 8)
    assert out.last_hidden_state.dtype == encoder.dtype

    assert encoder.is_gradient_checkpointing is False
    encoder.gradient_checkpointing_enable()
    assert encoder.is_gradient_checkpointing is True
    encoder.gradient_checkpointing_disable()
    assert encoder.is_gradient_checkpointing is False


def test_native_binding_head_is_zero_init_compatible_and_rms_bounded():
    encoder = _make_encoder()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones_like(ids)
    groups = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)
    counts = torch.tensor([2], dtype=torch.long)

    plain = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    structured = encoder(
        input_ids=ids,
        attention_mask=mask,
        anima_group_ids=groups,
        anima_subject_counts=counts,
    ).last_hidden_state
    # New controls are a strict no-op before training, preserving legacy/0.6B
    # conditioning behaviour for newly initialised heads.
    assert torch.allclose(plain, structured, atol=0.0, rtol=0.0)

    with torch.no_grad():
        encoder.native_head.slot_gate.fill_(2.0)
        encoder.native_head.count_gate.fill_(2.0)
        encoder.native_head.binding_gate.fill_(2.0)
    bound = encoder(
        input_ids=ids,
        attention_mask=mask,
        anima_group_ids=groups,
        anima_subject_counts=counts,
    ).last_hidden_state
    assert not torch.allclose(plain, bound)
    base_rms = plain.float().square().mean(dim=-1).sqrt()
    bound_rms = bound.float().square().mean(dim=-1).sqrt()
    ratio = bound_rms / base_rms.clamp_min(1e-6)
    assert float(ratio.max()) <= 1.081
    assert float(ratio.min()) >= 0.919
