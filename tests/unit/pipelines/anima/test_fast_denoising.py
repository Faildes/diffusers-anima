"""Fast denoising config works for both Anima transformer depths."""

from types import SimpleNamespace

import pytest
import torch

from diffusers_anima.pipelines.anima.pipeline_anima import AnimaPipeline


class Block:
    pass


class FakeTransformer:
    def __init__(self, depth):
        self.core = SimpleNamespace(transformer_blocks=[Block() for _ in range(depth)])
        self.compilations = []

    def compile_repeated_blocks(self, **kwargs):
        self.compilations.append(kwargs)


@pytest.mark.parametrize("depth", [28, 40])
def test_regional_compile_targets_repeated_blocks_once(depth):
    transformer = FakeTransformer(depth)
    pipe = SimpleNamespace(transformer=transformer)
    AnimaPipeline.enable_fast_denoising(pipe)
    assert transformer._repeated_blocks == ("Block",)
    assert transformer.compilations == [{"mode": "default", "fullgraph": False}]
    assert pipe.fast_cfg_batch_mode == "concat"
    AnimaPipeline.enable_fast_denoising(
        pipe, cfg_batch_mode="split", compile_blocks=False
    )
    assert pipe.fast_cfg_batch_mode == "split"
    assert len(transformer.compilations) == 1


def test_recompile_requires_a_new_pipeline():
    pipe = SimpleNamespace(transformer=FakeTransformer(28))
    AnimaPipeline.enable_fast_denoising(pipe)
    with pytest.raises(RuntimeError, match="already compiled"):
        AnimaPipeline.enable_fast_denoising(pipe)


class RecordingModule:
    def __init__(self):
        self.moves = []
        self.compilations = []
        self.full_compilations = []
        self.core = SimpleNamespace(transformer_blocks=[Block() for _ in range(40)])

    def modules(self):
        yield self

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def compile(self, **kwargs):
        assert not ({"mode", "options"} <= kwargs.keys())
        self.full_compilations.append(kwargs)

    def compile_repeated_blocks(self, **kwargs):
        assert not ({"mode", "options"} <= kwargs.keys())
        self.compilations.append(kwargs)


def maximum_pipe():
    return SimpleNamespace(
        transformer=RecordingModule(),
        text_encoder=RecordingModule(),
        vae=RecordingModule(),
        model_dtype=torch.bfloat16,
        text_encoder_dtype=torch.float32,
        execution_device="cpu",
        use_module_cpu_offload=True,
        keep_transformer_on_device=False,
    )


def test_maximum_resides_on_gpu_and_compiles_in_place():
    pipe = maximum_pipe()
    transformer = pipe.transformer
    result = AnimaPipeline.enable_fast_denoising(pipe, mode="maximum", device="cuda:1")
    assert result is pipe
    assert pipe.transformer is transformer
    assert pipe.transformer._repeated_blocks == ("Block",)
    assert pipe.transformer.compilations == [
        {"fullgraph": False, "options": {"triton.cudagraphs": False}}
    ]
    assert pipe.transformer.full_compilations == []
    for module, dtype in (
        (pipe.transformer, torch.bfloat16),
        (pipe.text_encoder, torch.float32),
        (pipe.vae, torch.bfloat16),
    ):
        assert module.moves == [((), {"device": torch.device("cuda:1"), "dtype": dtype})]
    assert pipe.execution_device == "cuda:1"
    assert pipe.use_module_cpu_offload is False
    assert pipe.keep_transformer_on_device is True
    assert pipe.fast_cfg_batch_mode == "concat"
    assert pipe.performance_mode == "maximum"
    assert pipe._anima_warmed_shapes == set()


def test_full_model_compile_requires_explicit_opt_in():
    pipe = maximum_pipe()
    AnimaPipeline.enable_fast_denoising(
        pipe, mode="maximum", compile_scope="full", device="cuda:0"
    )
    assert pipe.transformer.compilations == []
    assert pipe.transformer.full_compilations == [
        {
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": False,
        }
    ]


@pytest.mark.parametrize("unsafe_mode", ["reduce-overhead", "max-autotune"])
def test_cuda_graph_modes_refused_before_moving_components(unsafe_mode):
    pipe = maximum_pipe()
    with pytest.raises(ValueError, match="CUDA Graph"):
        AnimaPipeline.enable_fast_denoising(
            pipe, mode="maximum", compile_mode=unsafe_mode
        )
    assert pipe.transformer.moves == []


def test_max_autotune_without_graphs_remains_available():
    pipe = maximum_pipe()
    AnimaPipeline.enable_fast_denoising(
        pipe, mode="maximum", compile_mode="max-autotune-no-cudagraphs"
    )
    assert pipe.transformer.compilations == [
        {
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": False,
        }
    ]


def test_explicit_default_mode_does_not_conflict_with_cudagraph_options():
    pipe = maximum_pipe()
    AnimaPipeline.enable_fast_denoising(
        pipe, mode="maximum", compile_mode="default"
    )
    assert pipe.transformer.compilations == [
        {"fullgraph": False, "options": {"triton.cudagraphs": False}}
    ]


def test_maximum_rejects_invalid_scope_before_moving_components():
    pipe = maximum_pipe()
    with pytest.raises(ValueError, match="compile_scope"):
        AnimaPipeline.enable_fast_denoising(pipe, mode="maximum", compile_scope="unknown")
    assert pipe.transformer.moves == []


def test_maximum_rejects_cpu_before_moving_components():
    pipe = maximum_pipe()
    with pytest.raises(ValueError, match="CUDA"):
        AnimaPipeline.enable_fast_denoising(pipe, mode="maximum", device="cpu")
    assert pipe.transformer.moves == []
    assert pipe.text_encoder.moves == []
    assert pipe.vae.moves == []


def test_maximum_refuses_regional_compiled_pipeline():
    pipe = maximum_pipe()
    pipe._anima_blocks_compiled = True
    with pytest.raises(RuntimeError, match="already compiled"):
        AnimaPipeline.enable_fast_denoising(pipe, mode="maximum")
    assert pipe.transformer.moves == []
