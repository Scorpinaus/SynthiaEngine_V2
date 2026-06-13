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
_BYTES_PER_MB = 1024**2


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


class SummaryProfiler:
    """Capture a compact one-run resource summary for API results."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval_s: float | None = None,
        nvml_device_index: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.interval_s = config.RESOURCE_LOGGING_INTERVAL_S if interval_s is None else interval_s
        self.nvml_device_index = nvml_device_index
        self.profile: dict[str, Any] | None = None
        self._process = psutil.Process()
        self._start_time: float | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._stats: _SummaryStats | None = None

    def __enter__(self) -> "SummaryProfiler":
        if not self.enabled:
            return self

        cuda_available = _cuda_available()
        _reset_cuda_peak_memory_stats(cuda_available)
        nvml = _NvmlUsedMemorySampler(self.nvml_device_index)
        self._stats = _SummaryStats(cuda_available=cuda_available, nvml=nvml)
        self._stats.capture_start(self._process)
        self._start_time = time.perf_counter()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=_summary_sampling_loop,
            args=(self._stop_event, self._stats, max(0.01, float(self.interval_s))),
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.enabled or self._stats is None or self._start_time is None:
            return

        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        _synchronize_cuda(self._stats.cuda_available)
        elapsed_seconds = time.perf_counter() - self._start_time
        try:
            self._stats.capture_end(self._process)
            self.profile = self._stats.summary(elapsed_seconds)
        finally:
            self._stats.close()


class _SummaryStats:
    def __init__(self, *, cuda_available: bool, nvml: "_NvmlUsedMemorySampler") -> None:
        self.cuda_available = cuda_available
        self.nvml = nvml
        self.rss_start: int | None = None
        self.rss_end: int | None = None
        self.cuda_peak_allocated: int | None = None
        self.cuda_peak_reserved: int | None = None

    def capture_start(self, process: psutil.Process) -> None:
        self.rss_start = process.memory_info().rss
        self.nvml.capture_start()

    def capture_end(self, process: psutil.Process) -> None:
        self.rss_end = process.memory_info().rss
        if self.cuda_available:
            try:
                self.cuda_peak_allocated = torch.cuda.max_memory_allocated()
                self.cuda_peak_reserved = torch.cuda.max_memory_reserved()
            except Exception:
                self.cuda_peak_allocated = None
                self.cuda_peak_reserved = None
        self.nvml.capture_end()

    def sample(self) -> None:
        self.nvml.sample()

    def summary(self, elapsed_seconds: float) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "elapsed_seconds": elapsed_seconds,
            "rss_before_mb": _bytes_to_mb(self.rss_start),
            "rss_after_mb": _bytes_to_mb(self.rss_end),
            "cuda_available": self.cuda_available,
            "cuda_peak_allocated_mb": _bytes_to_mb(self.cuda_peak_allocated),
            "cuda_peak_reserved_mb": _bytes_to_mb(self.cuda_peak_reserved),
            **self.nvml.summary(),
        }

    def close(self) -> None:
        self.nvml.close()


class _NvmlUsedMemorySampler:
    def __init__(self, device_index: int | None) -> None:
        self.available = False
        self.device_index: int | None = None
        self.used_start: int | None = None
        self.used_end: int | None = None
        self.used_peak: int | None = None
        self._pynvml: Any | None = None
        self._handle: Any | None = None
        self._initialized = False

        pynvml = None
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            index = _default_nvml_device_index() if device_index is None else int(device_index)
            if count <= 0 or index < 0 or index >= count:
                pynvml.nvmlShutdown()
                return

            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._initialized = True
            self.available = True
            self.device_index = index
        except Exception:
            if pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._pynvml = None
            self._handle = None
            self._initialized = False

    def capture_start(self) -> None:
        used = self._current_used()
        self.used_start = used
        self.used_peak = used

    def capture_end(self) -> None:
        used = self._current_used()
        self.used_end = used
        if used is not None:
            self.used_peak = max(self.used_peak or 0, used)

    def sample(self) -> None:
        used = self._current_used()
        if used is not None:
            self.used_peak = max(self.used_peak or 0, used)

    def summary(self) -> dict[str, Any]:
        return {
            "nvml_available": self.available,
            "nvml_device_index": self.device_index,
            "nvml_used_start_mb": _bytes_to_mb(self.used_start),
            "nvml_used_end_mb": _bytes_to_mb(self.used_end),
            "nvml_used_peak_sampled_mb": _bytes_to_mb(self.used_peak),
        }

    def close(self) -> None:
        if self._initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._initialized = False

    def _current_used(self) -> int | None:
        if not self.available or self._pynvml is None or self._handle is None:
            return None
        try:
            return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        except Exception:
            return None


def _summary_sampling_loop(
    stop_event: threading.Event,
    stats: _SummaryStats,
    interval_s: float,
) -> None:
    while not stop_event.is_set():
        stats.sample()
        stop_event.wait(interval_s)


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _reset_cuda_peak_memory_stats(cuda_available: bool) -> None:
    if not cuda_available:
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _synchronize_cuda(cuda_available: bool) -> None:
    if not cuda_available:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _default_nvml_device_index() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    return 0


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / _BYTES_PER_MB
