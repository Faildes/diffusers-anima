"""Compilation is displayed before denoising without another model call."""

from types import SimpleNamespace

import torch

from diffusers_anima.pipelines.anima import pipeline_anima, sampling


def test_first_real_forward_has_separate_progress(monkeypatch):
    events = []
    compile_options = {}

    class Bar:
        def __init__(self, name):
            self.name = name
            events.append(name + "_open")

        def update(self, count=1):
            events.append(self.name + "_update")

        def close(self):
            events.append(self.name + "_close")

    def make_compile_bar(**kwargs):
        compile_options.update(kwargs)
        return Bar("compile")

    monkeypatch.setattr(pipeline_anima, "tqdm", make_compile_bar)

    class FakePipeline:
        _progress_bar_config = {}
        _anima_warmed_shapes = set()

        def progress_bar(self, **kwargs):
            return Bar("denoise")

    class FakeTransformer:
        def __call__(self, latents, timestep, *, encoder_hidden_states, return_dict):
            events.append("model")
            return (torch.zeros_like(latents),)

    pipeline = FakePipeline()
    wrapped = pipeline_anima._FirstPredictionProgress(
        FakeTransformer(), pipeline, key=("shape",), forwards=1
    )
    sampling.sample_euler_ancestral_rf(
        wrapped,
        pipeline,
        torch.ones(1, 1),
        sigmas=torch.tensor([0.9, 0.0]),
        pos_cond=torch.zeros(1, 1, 1),
        neg_cond=None,
        guidance_scale=1.0,
        eta=1.0,
        s_noise=1.0,
        generator=torch.Generator(device="cpu").manual_seed(5),
        cfg_batch_mode="split",
        model_dtype=torch.float32,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    assert events == [
        "compile_open", "model", "compile_update", "compile_close",
        "denoise_open", "denoise_update", "denoise_close",
    ]
    assert compile_options["total"] is None
    assert ("shape",) in pipeline._anima_warmed_shapes


def test_split_cfg_waits_for_both_first_prediction_forwards(monkeypatch):
    events = []
    monkeypatch.setattr(
        pipeline_anima,
        "tqdm",
        lambda **kwargs: SimpleNamespace(
            update=lambda count: events.append("done"),
            close=lambda: events.append("closed"),
        ),
    )
    pipe = SimpleNamespace(_progress_bar_config={}, _anima_warmed_shapes=set())
    wrapped = pipeline_anima._FirstPredictionProgress(
        lambda: events.append("forward"), pipe, key=(28,), forwards=2
    )
    wrapped()
    assert (28,) not in pipe._anima_warmed_shapes
    wrapped()
    assert events == ["forward", "forward", "done", "closed"]
    assert (28,) in pipe._anima_warmed_shapes
