"""Optional, per-call development statistics for Anima image generation."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import torch
from tqdm.auto import tqdm


_GIB = 1024 ** 3


def _gib(value: float | int) -> float:
    return value / _GIB


class GenerationDevMonitor:
    """Measure one pipeline call, including optional CUDA memory statistics.

    Average usage is time-weighted from periodic observations. PyTorch's exact
    peak counters cover allocations between observations. Device-wide used
    memory is sampled and may include allocations from other processes.
    """

    def __init__(self, pipeline: Any, sample_interval: float, *, image_count: int = 1):
        self.pipeline = pipeline
        self.sample_interval = sample_interval
        self.image_count = image_count
        self.device = torch.device(pipeline.execution_device)
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples = 0
        self.first: dict[str, int] | None = None
        self.last: dict[str, int] | None = None
        self.area: dict[str, float] = {}
        self.sampled_peak: dict[str, int] = {}
        self.last_sample_time: float | None = None
        self.started: float | None = None
        self.device_used_available = True

    def _sample(self) -> None:
        if not self.cuda:
            return
        values = {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
        }
        if self.device_used_available:
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                values["device_used"] = total - free
            except (RuntimeError, NotImplementedError):
                # Device-wide usage is optional; allocator measurements still work.
                self.device_used_available = False
        now = time.perf_counter()
        if self.first is None:
            self.first = values.copy()
            if self.started is not None:
                for key, value in values.items():
                    self.area[key] = value * max(0.0, now - self.started)
        if self.last is not None and self.last_sample_time is not None:
            delta = max(0.0, now - self.last_sample_time)
            for key, value in self.last.items():
                self.area[key] = self.area.get(key, 0.0) + value * delta
        for key, value in values.items():
            self.sampled_peak[key] = max(self.sampled_peak.get(key, 0), value)
        self.last = values
        self.last_sample_time = now
        self.samples += 1

    def _poll(self) -> None:
        while not self.stop_event.wait(self.sample_interval):
            self._sample()

    def __enter__(self) -> GenerationDevMonitor:
        if self.cuda:
            # Include only this call in the CUDA allocator peak counters.
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        if self.cuda:
            self._sample()
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        if self.cuda:
            torch.cuda.synchronize(self.device)
            self._sample()
        finished = time.perf_counter()
        elapsed = finished - self.started if self.started is not None else 0.0
        pipeline = self.pipeline
        pipeline.dev_metrics_calls += 1
        pipeline.dev_metrics_total_seconds += elapsed
        pipeline.dev_metrics_total_images += self.image_count
        report: dict[str, Any] = {
            "call": pipeline.dev_metrics_calls,
            "images": self.image_count,
            "seconds": elapsed,
            "seconds_per_image": elapsed / self.image_count,
            "images_per_second": self.image_count / elapsed if elapsed else 0.0,
            "total_seconds": pipeline.dev_metrics_total_seconds,
            "total_images": pipeline.dev_metrics_total_images,
            "average_seconds_per_image": (
                pipeline.dev_metrics_total_seconds / pipeline.dev_metrics_total_images
            ),
            "average_seconds_per_call": pipeline.dev_metrics_total_seconds / pipeline.dev_metrics_calls,
            "succeeded": exc_type is None,
            "device": str(self.device),
            "samples": self.samples,
            "sample_interval_seconds": self.sample_interval,
            "vram": None,
        }
        if self.first is not None and self.last is not None and self.last_sample_time is not None:
            interval = max(self.last_sample_time - self.started, 0.0)
            vram = {}
            for key, value in self.last.items():
                if key not in self.first:
                    continue
                peak = self.sampled_peak[key]
                if key == "allocated":
                    peak = max(peak, torch.cuda.max_memory_allocated(self.device))
                elif key == "reserved":
                    peak = max(peak, torch.cuda.max_memory_reserved(self.device))
                vram[key] = {
                    "start_gib": _gib(self.first[key]),
                    "average_gib": _gib(self.area.get(key, 0.0) / interval if interval else value),
                    "peak_gib": _gib(peak),
                    "end_gib": _gib(value),
                }
            report["vram"] = vram
        pipeline.dev_metrics_last = report
        if pipeline._dev_metrics_print:
            status = "" if exc_type is None else " (failed)"
            tqdm.write(
                f"[Anima Dev] call {report['call']}{status} ({self.image_count} images): "
                f"{elapsed:.2f}s | total {report['total_seconds']:.2f}s "
                f"| avg/image {report['average_seconds_per_image']:.2f}s "
                f"| {report['images_per_second']:.2f} images/s"
            )
            if report["vram"] is None:
                tqdm.write("[Anima Dev] VRAM: unavailable (non-CUDA device or unsupported CUDA statistics)")
            else:
                for key, name in (
                    ("allocated", "PyTorch allocated"),
                    ("reserved", "PyTorch reserved"),
                    ("device_used", "GPU used (all processes)"),
                ):
                    if key not in report["vram"]:
                        continue
                    v = report["vram"][key]
                    tqdm.write(
                        f"[Anima Dev] {name} GiB: start {v['start_gib']:.2f} "
                        f"| avg~ {v['average_gib']:.2f} "
                        f"| peak{'~' if key == 'device_used' else ''} {v['peak_gib']:.2f} "
                        f"| end {v['end_gib']:.2f}"
                    )
        return False


def validate_sample_interval(value: float) -> float:
    """Bound polling overhead and reject nonsensical sampling intervals."""
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.02:
        raise ValueError("sample_interval must be a finite number >= 0.02 seconds.")
    return float(value)
