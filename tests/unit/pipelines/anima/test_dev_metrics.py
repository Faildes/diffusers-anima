"""Development timing and VRAM reporting for pipeline calls."""

from types import SimpleNamespace

import pytest
import torch

from diffusers_anima.pipelines.anima import dev_metrics
from diffusers_anima.pipelines.anima.pipeline_anima import AnimaPipeline


def _pipe(device="cpu"):
    return SimpleNamespace(
        execution_device=device,
        dev_metrics_calls=0,
        dev_metrics_total_seconds=0.0,
        dev_metrics_total_images=0,
        dev_metrics_last=None,
        _dev_metrics_print=False,
    )


def test_cpu_reports_per_call_and_cumulative_time(monkeypatch):
    pipe = _pipe()
    clock = iter((0.0, 3.0, 3.0, 7.0))
    monkeypatch.setattr(dev_metrics, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    with dev_metrics.GenerationDevMonitor(pipe, 0.1):
        pass
    assert pipe.dev_metrics_last["seconds"] == 3.0
    assert pipe.dev_metrics_last["vram"] is None
    with dev_metrics.GenerationDevMonitor(pipe, 0.1):
        pass
    assert pipe.dev_metrics_last["seconds"] == 4.0
    assert pipe.dev_metrics_last["total_seconds"] == 7.0
    assert pipe.dev_metrics_last["average_seconds_per_call"] == 3.5
    assert pipe.dev_metrics_last["total_images"] == 2


def test_batch_timing_reports_per_image_throughput(monkeypatch):
    pipe = _pipe()
    clock = iter((0.0, 4.0))
    monkeypatch.setattr(dev_metrics, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    with dev_metrics.GenerationDevMonitor(pipe, 0.1, image_count=4):
        pass
    assert pipe.dev_metrics_last["images"] == 4
    assert pipe.dev_metrics_last["seconds_per_image"] == 1.0
    assert pipe.dev_metrics_last["images_per_second"] == 1.0


def test_cuda_reports_time_weighted_average_and_exact_allocator_peak(monkeypatch):
    gib = 1024 ** 3
    pipe = _pipe("cuda:0")
    allocations = iter((1 * gib, 3 * gib, 2 * gib))
    reservations = iter((2 * gib, 4 * gib, 3 * gib))
    available = iter((8 * gib, 6 * gib, 7 * gib))
    events = []
    monkeypatch.setattr(dev_metrics, "time", SimpleNamespace(perf_counter=iter((0.0, 0.0, 1.0, 2.0, 2.0)).__next__))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: next(allocations))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: next(reservations))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (next(available), 12 * gib))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 5 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 8 * gib)
    with dev_metrics.GenerationDevMonitor(pipe, 60.0) as monitor:
        monitor._sample()
    stats = pipe.dev_metrics_last
    assert events == ["sync", "reset", "sync"]
    assert stats["seconds"] == 2.0
    assert stats["samples"] == 3
    assert stats["vram"]["allocated"] == {
        "start_gib": 1.0, "average_gib": 2.0, "peak_gib": 5.0, "end_gib": 2.0
    }
    assert stats["vram"]["reserved"]["peak_gib"] == 8.0
    assert stats["vram"]["device_used"]["average_gib"] == 5.0
    assert stats["vram"]["device_used"]["peak_gib"] == 6.0


def test_dev_metrics_toggle_and_reset():
    pipe = _pipe()
    with pytest.raises(ValueError, match="sample_interval"):
        AnimaPipeline.enable_dev_metrics(pipe, sample_interval=0)
    AnimaPipeline.enable_dev_metrics(pipe, sample_interval=0.2, print_report=False)
    assert pipe._dev_metrics_enabled is True
    assert pipe._dev_metrics_sample_interval == 0.2
    assert pipe._dev_metrics_print is False
    AnimaPipeline.disable_dev_metrics(pipe)
    assert pipe._dev_metrics_enabled is False
    pipe.dev_metrics_calls = 2
    pipe.dev_metrics_total_seconds = 4.0
    pipe.dev_metrics_total_images = 8
    AnimaPipeline.reset_dev_metrics(pipe)
    assert pipe.dev_metrics_calls == 0
    assert pipe.dev_metrics_total_seconds == 0.0
    assert pipe.dev_metrics_total_images == 0
    assert pipe.dev_metrics_last is None
