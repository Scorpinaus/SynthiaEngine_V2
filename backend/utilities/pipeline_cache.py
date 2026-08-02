from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar

from backend.settings import PipelineCacheSettings, load_settings

T = TypeVar("T")
_CACHES: "weakref.WeakSet[PipelineCache]" = weakref.WeakSet()


@dataclass
class _Entry(Generic[T]):
    value: T
    cost_mb: int
    release: Callable[[T], None]


class PipelineCache(Generic[T]):
    """Thread-safe LRU cache with entry and estimated-memory budgets."""

    def __init__(self, *, max_entries: int = 0, max_cost_mb: int = 0, name: str = "pipeline"):
        self.name = name
        self.max_entries = max(0, int(max_entries))
        self.max_cost_mb = max(0, int(max_cost_mb))
        self._entries: OrderedDict[Hashable, _Entry[T]] = OrderedDict()
        self._cost_mb = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        _CACHES.add(self)

    @classmethod
    def from_settings(
        cls,
        settings: PipelineCacheSettings | None = None,
        *,
        name: str = "pipeline",
    ) -> "PipelineCache[T]":
        budget = settings or load_settings().pipeline_cache
        return cls(
            max_entries=budget.max_entries,
            max_cost_mb=budget.max_cost_mb,
            name=name,
        )

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_cost_mb > 0

    @property
    def current_cost_mb(self) -> int:
        with self._lock:
            return self._cost_mb

    def acquire(
        self,
        key: Hashable,
        loader: Callable[[], T],
        *,
        cost_mb: int,
        release: Callable[[T], None],
    ) -> tuple[T, bool]:
        """Return `(value, cache_owned)`; false means caller owns cleanup."""
        if not self.enabled or cost_mb <= 0 or cost_mb > self.max_cost_mb:
            self.misses += 1
            return loader(), False
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._entries[key] = entry
                self.hits += 1
                return entry.value, True

            self.misses += 1
            value = loader()
            entry = _Entry(value=value, cost_mb=int(cost_mb), release=release)
            self._entries[key] = entry
            self._cost_mb += entry.cost_mb
            self._evict_to_budget()
            return value, key in self._entries

    def _evict_to_budget(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_entries or self._cost_mb > self.max_cost_mb
        ):
            _, entry = self._entries.popitem(last=False)
            self._cost_mb -= entry.cost_mb
            self.evictions += 1
            entry.release(entry.value)

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._cost_mb = 0
        for entry in entries:
            entry.release(entry.value)

    def discard(self, value: T) -> bool:
        """Remove and release a specific cached value by object identity."""
        entry_to_release = None
        with self._lock:
            for key, entry in self._entries.items():
                if entry.value is value:
                    entry_to_release = self._entries.pop(key)
                    self._cost_mb -= entry_to_release.cost_mb
                    self.evictions += 1
                    break
        if entry_to_release is not None:
            entry_to_release.release(entry_to_release.value)
            return True
        return False

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "cost_mb": self._cost_mb,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


def clear_all_pipeline_caches() -> None:
    for cache in list(_CACHES):
        cache.clear()


def pipeline_cache_stats() -> dict[str, dict[str, int | bool]]:
    return {cache.name: cache.stats() for cache in list(_CACHES)}
