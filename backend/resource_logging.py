import contextlib
import functools
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import psutil
import torch

from backend import config

logger = logging.getLogger(__name__)


@dataclass
class _Snapshot:
    ram_rss: int
    vram_allocated: int | None
    vram_free: int | None
    vram_total: int | None


class ResourceLogger:
    def __init__(self, enabled: bool | None = None, interval_s: float | None = None):
        self.enabled = config.RESOURCE_LOGGING_ENABLED if enabled is None else enabled
        self.interval_s = (
            config.RESOURCE_LOGGING_INTERVAL_S if interval_s is None else interval_s
        )
        self._process = psutil.Process()

    def annotate(
        self,
        label: str,
        metadata_builder: Callable[[inspect.BoundArguments], dict[str, Any]] | None = None,
        batch_id_factory: Callable[[], str] | None = None,
        batch_id_arg: str = "batch_id",
    ):
        def decorator(func):
            signature = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if batch_id_factory and batch_id_arg in signature.parameters:
                    if kwargs.get(batch_id_arg) is None:
                        kwargs[batch_id_arg] = batch_id_factory()

                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                metadata = metadata_builder(bound) if metadata_builder else {}
                metadata = metadata or {}
                with self.track(label, **metadata):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def track(self, label: str, **metadata: Any):
        @contextlib.contextmanager
        def _context():
            if not self.enabled:
                yield
                return

            start_time = time.monotonic()
            stop_event = threading.Event()
            stats = _SamplingStats(cuda_available=torch.cuda.is_available())

            stats.capture_start(self._process)
            thread = threading.Thread(
                target=_sampling_loop,
                args=(self._process, self.interval_s, stop_event, stats),
                daemon=True,
            )
            thread.start()
            try:
                yield
            finally:
                stop_event.set()
                thread.join()
                stats.capture_end(self._process)
                duration_s = time.monotonic() - start_time
                payload = stats.summary(duration_s)
                if metadata:
                    payload["context"] = metadata
                logger.info("Resource usage (%s): %s", label, payload)

        return _context()


class _SamplingStats:
    def __init__(self, cuda_available: bool):
        self.cuda_available = cuda_available
        self.start: _Snapshot | None = None
        self.end: _Snapshot | None = None
        self.peak_vram_allocated: int | None = None
        self.ram_sum = 0
        self.ram_max = 0
        self.vram_sum = 0
        self.vram_max = 0
        self.count = 0

    def capture_start(self, process: psutil.Process) -> None:
        if self.cuda_available:
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
        self.start = _snapshot(process, self.cuda_available)

    def capture_end(self, process: psutil.Process) -> None:
        self.end = _snapshot(process, self.cuda_available)
        if self.cuda_available:
            try:
                self.peak_vram_allocated = torch.cuda.max_memory_allocated()
            except RuntimeError:
                self.peak_vram_allocated = None

    def sample(self, process: psutil.Process) -> None:
        snap = _snapshot(process, self.cuda_available)
        self.count += 1
        self.ram_sum += snap.ram_rss
        self.ram_max = max(self.ram_max, snap.ram_rss)
        if snap.vram_allocated is not None:
            self.vram_sum += snap.vram_allocated
            self.vram_max = max(self.vram_max, snap.vram_allocated)

    def summary(self, duration_s: float) -> dict[str, Any]:
        avg_ram = self.ram_sum / self.count if self.count else 0
        avg_vram = self.vram_sum / self.count if self.count else 0
        summary: dict[str, Any] = {
            "duration_s": round(duration_s, 3),
            "ram_bytes": {
                "start": self.start.ram_rss if self.start else None,
                "end": self.end.ram_rss if self.end else None,
                "avg": int(avg_ram),
                "max": self.ram_max,
            },
        }
        if self.cuda_available:
            summary["vram_bytes"] = {
                "start": self.start.vram_allocated if self.start else None,
                "end": self.end.vram_allocated if self.end else None,
                "avg": int(avg_vram),
                "max": self.vram_max,
                "peak_allocated": self.peak_vram_allocated,
                "free_start": self.start.vram_free if self.start else None,
                "free_end": self.end.vram_free if self.end else None,
                "total": self.start.vram_total if self.start else None,
            }
        return summary


def _snapshot(process: psutil.Process, cuda_available: bool) -> _Snapshot:
    ram_rss = process.memory_info().rss
    vram_allocated = None
    vram_free = None
    vram_total = None
    if cuda_available:
        try:
            vram_allocated = torch.cuda.memory_allocated()
            vram_free, vram_total = torch.cuda.mem_get_info()
        except RuntimeError:
            vram_allocated = None
            vram_free = None
            vram_total = None
    return _Snapshot(
        ram_rss=ram_rss,
        vram_allocated=vram_allocated,
        vram_free=vram_free,
        vram_total=vram_total,
    )


def _sampling_loop(
    process: psutil.Process,
    interval_s: float,
    stop_event: threading.Event,
    stats: _SamplingStats,
) -> None:
    while not stop_event.is_set():
        stats.sample(process)
        stop_event.wait(interval_s)


resource_logger = ResourceLogger()
