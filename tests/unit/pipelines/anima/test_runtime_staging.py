"""Stage boundaries must never move the denoiser once it is resident."""

from types import SimpleNamespace

import torch

from diffusers_anima.pipelines.anima.pipeline_anima import (
    AnimaPipeline,
    _resolve_effective_cfg_batch_mode,
)


class RecordingModule:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def to(self, *args, **kwargs):
        self.events.append((self.name, args, kwargs))
        return self


def test_persistent_staging_leaves_only_transformer_on_accelerator():
    events = []
    pipe = SimpleNamespace(
        transformer=RecordingModule("transformer", events),
        text_encoder=RecordingModule("text", events),
        vae=RecordingModule("vae", events),
        model_dtype=torch.float32,
    )
    AnimaPipeline.enable_persistent_transformer_staging(pipe, device="cuda:1")
    assert [entry[0] for entry in events] == ["text", "vae", "transformer"]
    assert events[0][1] == ("cpu",)
    assert events[1][1] == ("cpu",)
    assert events[2][2]["device"] == torch.device("cuda:1")
    assert pipe.execution_device == "cuda:1"
    assert pipe.keep_transformer_on_device
    assert pipe.use_module_cpu_offload


def test_regular_offload_restores_transformer_stage_transfers():
    pipe = SimpleNamespace(
        _anima_execution_device="cuda",
        keep_transformer_on_device=True,
        use_module_cpu_offload=True,
    )
    AnimaPipeline.enable_model_cpu_offload(pipe)
    assert not pipe.keep_transformer_on_device


def test_auto_cfg_recognizes_indexed_cuda_device():
    assert _resolve_effective_cfg_batch_mode(
        "auto", execution_device="cuda:1", transformer_num_layers=28
    ) == "concat"
    assert _resolve_effective_cfg_batch_mode(
        "auto", execution_device="cuda:1", transformer_num_layers=40
    ) == "split"
