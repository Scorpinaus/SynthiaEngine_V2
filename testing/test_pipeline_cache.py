from backend.utilities.pipeline_cache import PipelineCache
from backend.flux.pipeline import _timed_pipeline_call
from backend.settings import PipelineCacheSettings


def test_cache_uses_typed_budget_settings():
    cache = PipelineCache.from_settings(
        PipelineCacheSettings(max_entries=2, max_cost_mb=512),
        name="test",
    )

    assert cache.name == "test"
    assert cache.max_entries == 2
    assert cache.max_cost_mb == 512


def test_disabled_cache_leaves_value_owned_by_caller():
    cache = PipelineCache(max_entries=0, max_cost_mb=0)
    value, cached = cache.acquire("a", lambda: object(), cost_mb=10, release=lambda _v: None)
    assert value is not None
    assert cached is False
    assert cache.stats()["entries"] == 0


def test_cache_reuses_values_and_tracks_hits():
    cache = PipelineCache(max_entries=2, max_cost_mb=100)
    loaded = []
    first, cached = cache.acquire("a", lambda: loaded.append(object()) or loaded[-1], cost_mb=40, release=lambda _v: None)
    second, cached_again = cache.acquire("a", lambda: object(), cost_mb=40, release=lambda _v: None)
    assert cached is True and cached_again is True
    assert first is second
    assert cache.stats() == {"enabled": True, "entries": 1, "cost_mb": 40, "hits": 1, "misses": 1, "evictions": 0}


def test_lru_eviction_releases_pipeline_and_respects_memory_budget():
    released = []
    cache = PipelineCache(max_entries=3, max_cost_mb=100)
    a = object()
    b = object()
    cache.acquire("a", lambda: a, cost_mb=60, release=released.append)
    cache.acquire("b", lambda: b, cost_mb=60, release=released.append)
    assert released == [a]
    assert cache.current_cost_mb == 60
    assert cache.stats()["evictions"] == 1


def test_oversized_value_is_not_cached():
    cache = PipelineCache(max_entries=2, max_cost_mb=100)
    value, cached = cache.acquire("large", lambda: object(), cost_mb=101, release=lambda _v: None)
    assert value is not None
    assert cached is False
    assert cache.stats()["entries"] == 0


def test_flux_stage_timing_separates_denoise_and_decode_when_callback_is_supported():
    class FakePipe:
        def __call__(self, *, callback_on_step_end=None, **_kwargs):
            callback_on_step_end(self, 0, 1, {})
            return object()

    output, timing = _timed_pipeline_call(FakePipe(), {})
    assert output is not None
    assert timing["inference_seconds"] >= 0
    assert timing["denoise_seconds"] is not None
    assert timing["decode_seconds"] is not None


def test_discard_releases_unhealthy_cached_instance():
    released = []
    cache = PipelineCache(max_entries=1, max_cost_mb=100)
    value = object()
    cache.acquire("a", lambda: value, cost_mb=50, release=released.append)
    assert cache.discard(value) is True
    assert released == [value]
    assert cache.stats()["entries"] == 0
