from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from PIL import Image
import pytest

from backend.lora.registry import LoraRegistryEntry
from backend.qwen_image.lightning import resolve_qwen_image_lightning_profile
from backend.qwen_image import pipeline as qwen_image_pipeline
from backend.registries.model import ModelRegistryEntry
from backend.utilities.subprocess_transport import SubprocessCanceled


class _FakePipeline:
    def __init__(self, events: list[str]):
        self.events = events
        self.scheduler = object()
        self.call_arguments: list[dict[str, object]] = []
        self.inference_error: Exception | None = None
        self.unload_error: Exception | None = None

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.events.append("inference")
        self.call_arguments.append(kwargs)
        if self.inference_error is not None:
            raise self.inference_error
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

    def unload_lora_weights(self) -> None:
        self.events.append("unload")
        if self.unload_error is not None:
            raise self.unload_error


def _normal_resolution(lora_adapters, _model_entry, _task, steps, true_cfg_scale):
    adapters = tuple(
        SimpleNamespace(
            lora_id=adapter["lora_id"],
            entry=SimpleNamespace(lora_id=adapter["lora_id"]),
        )
        for adapter in (lora_adapters or [])
    )
    return SimpleNamespace(
        adapters=adapters,
        model_variant="qwen-image-2512",
        task=_task,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        lightning_profile=None,
    )


class _MixedStackPipeline:
    def __init__(self, reported_active_adapters: list[str] | None = None):
        self.transformer = object()
        self.load_calls: list[tuple[str, dict[str, object]]] = []
        self.set_calls: list[tuple[list[str], list[float]]] = []
        self.fuse_lora = Mock()
        self._reported_active_adapters = reported_active_adapters

    def load_lora_weights(self, file_path: str, **kwargs: object) -> None:
        self.load_calls.append((file_path, kwargs))

    def set_adapters(self, names: list[str], *, adapter_weights: list[float]) -> None:
        self.set_calls.append((names, adapter_weights))
        if self._reported_active_adapters is None:
            self._reported_active_adapters = list(names)

    def get_active_adapters(self) -> list[str]:
        return list(self._reported_active_adapters or [])


class _MissingSetAdaptersPipeline(_MixedStackPipeline):
    set_adapters = None


class _NoActiveAdapterQueryPipeline(_MixedStackPipeline):
    get_active_adapters = None


def _mixed_stack_inputs(order: tuple[str, str], task: str = "text2img") -> tuple[
    list[dict[str, object]], object, dict[int, LoraRegistryEntry]
]:
    lightning = LoraRegistryEntry(
        lora_id=101,
        lora_model_family="qwen-image",
        lora_type="lora",
        lora_location="local",
        file_path="C:/loras/lightning.safetensors",
        name="Shared adapter",
        runtime_profile={
            "kind": "qwen_image_lightning",
            "base_variant": "qwen-image-2512",
            "steps": 4,
            "true_cfg_scale": 1.0,
            "scheduler_profile": "qwen_image_lightning_shift3",
            "adapter_strength": 1.0,
            "supported_tasks": ["text2img", "img2img", "inpaint"],
        },
    )
    companion = LoraRegistryEntry(
        lora_id=202,
        lora_model_family="qwen-image",
        lora_type="lora",
        lora_location="local",
        file_path="C:/loras/companion.safetensors",
        name="Shared adapter",
        compatibility={
            "base_variants": ["qwen-image-2512"],
            "runtime_profile_kinds": ["qwen_image_lightning"],
            "supported_tasks": ["text2img", "img2img", "inpaint"],
        },
    )
    entries = {lightning.lora_id: lightning, companion.lora_id: companion}
    strengths = {lightning.lora_id: 1.0, companion.lora_id: 0.5}
    entries_by_name = {"lightning": lightning, "companion": companion}
    selected = []
    for name in order:
        entry = entries_by_name[name]
        selected.append({"lora_id": entry.lora_id, "strength": strengths[entry.lora_id]})
    model = ModelRegistryEntry(
        name="Qwen-Image-2512-SDNQ-4bit-dynamic",
        family="qwen-image",
        model_type="diffusers",
        location_type="local",
        model_id=1,
        version="local",
        link="C:/models/Qwen-Image-2512-SDNQ-4bit-dynamic",
    )
    resolution = resolve_qwen_image_lightning_profile(
        selected,
        model,
        task,
        4,
        1.0,
        lookup_lora_entry=entries.__getitem__,
    )
    return selected, resolution, entries


class _LifecycleMixedStackPipeline(_MixedStackPipeline):
    def __init__(
        self,
        events: list[str],
        *,
        failure: str | None = None,
        reported_active_adapters: list[str] | None = None,
    ):
        super().__init__(reported_active_adapters=reported_active_adapters)
        self.events = events
        self.failure = failure
        self.scheduler = object()
        self.call_arguments: list[dict[str, object]] = []
        self.unload_calls = 0
        self.active_query_calls = 0

    def load_lora_weights(self, file_path: str, **kwargs: object) -> None:
        super().load_lora_weights(file_path, **kwargs)
        self.events.append("adapter_load")
        if self.failure == "second_adapter_load" and len(self.load_calls) == 2:
            raise RuntimeError("synthetic second adapter load failure")

    def set_adapters(self, names: list[str], *, adapter_weights: list[float]) -> None:
        super().set_adapters(names, adapter_weights=adapter_weights)
        self.events.append("set_adapters")

    def get_active_adapters(self) -> list[str]:
        self.active_query_calls += 1
        return super().get_active_adapters()

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.events.append("inference")
        self.call_arguments.append(kwargs)
        if self.failure == "inference":
            raise RuntimeError("synthetic mixed inference failure")
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

    def unload_lora_weights(self) -> None:
        self.unload_calls += 1
        self.events.append("unload")


@contextmanager
def _patched_mixed_generation(
    loader_name: str,
    pipe: _LifecycleMixedStackPipeline,
    entries: dict[int, LoraRegistryEntry],
    tmp_path,
    *,
    additional_pipelines: list[_LifecycleMixedStackPipeline] | None = None,
):
    events = pipe.events
    mixed_scheduler = object()
    base_scheduler = object()
    loaded_pipelines = iter([pipe, *(additional_pipelines or [])])
    model_entry = ModelRegistryEntry(
        name="Qwen-Image-2512-SDNQ-4bit-dynamic",
        family="qwen-image",
        model_type="diffusers",
        location_type="local",
        model_id=1,
        version="local",
        link="C:/models/Qwen-Image-2512-SDNQ-4bit-dynamic",
    )

    def _resolve(adapters, model_entry, task, steps, true_cfg_scale):
        events.append(f"resolve:{task}")
        return resolve_qwen_image_lightning_profile(
            adapters,
            model_entry,
            task,
            steps,
            true_cfg_scale,
            lookup_lora_entry=entries.__getitem__,
        )

    def _load_pipeline(_model):
        events.append("pipeline_load")
        return next(loaded_pipelines)

    def _select_scheduler(resolution, _pipe):
        events.append(
            "mixed_scheduler" if resolution.lightning_profile is not None else "base_scheduler"
        )
        return mixed_scheduler if resolution.lightning_profile is not None else base_scheduler

    def _create_scheduler(_scheduler, _pipe):
        events.append("base_scheduler")
        return base_scheduler

    def _release_pipeline(released_pipe, *, logger):
        assert released_pipe is not None
        assert logger is qwen_image_pipeline.logger
        events.append("release")

    with (
        patch.object(qwen_image_pipeline, loader_name, side_effect=_load_pipeline),
        patch.object(
            qwen_image_pipeline,
            "_get_qwen_image_model_entry",
            return_value=model_entry,
        ),
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=_resolve,
        ),
        patch.object(
            qwen_image_pipeline,
            "select_qwen_image_scheduler",
            side_effect=_select_scheduler,
        ),
        patch.object(qwen_image_pipeline, "create_scheduler", side_effect=_create_scheduler),
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="mixed"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(
            qwen_image_pipeline,
            "save_generated_image",
            return_value="batch_mixed/output.png",
        ),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=_release_pipeline),
        patch.object(
            qwen_image_pipeline.torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        yield mixed_scheduler, base_scheduler


def _run_generation_with_lora(
    generation_function,
    loader_name: str,
    params: dict[str, object],
    tmp_path,
):
    events: list[str] = []
    pipe = _FakePipeline(events)
    scheduler = object()

    def _apply_lora(*args, **kwargs):
        assert pipe.scheduler is scheduler
        events.append("apply")
        return ["lora_Qwen"]

    def _release_pipeline(released_pipe, *, logger):
        assert released_pipe is pipe
        assert logger is qwen_image_pipeline.logger
        events.append("release")

    with (
        patch.object(qwen_image_pipeline, loader_name, return_value=pipe),
        patch.object(qwen_image_pipeline, "create_scheduler", return_value=scheduler),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", return_value=scheduler),
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=_normal_resolution,
        ),
        patch.object(
            qwen_image_pipeline,
            "_apply_qwen_lora_adapters",
            side_effect=_apply_lora,
        ) as apply_lora,
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="lora"),
        patch.object(
            qwen_image_pipeline,
            "get_batch_output_dir",
            return_value=tmp_path,
        ),
        patch.object(
            qwen_image_pipeline,
            "save_generated_image",
            return_value="batch_lora/output.png",
        ),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(
            qwen_image_pipeline,
            "release_pipeline",
            side_effect=_release_pipeline,
        ) as release_pipeline,
        patch.object(
            qwen_image_pipeline.torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        result = generation_function(params)

    return result, pipe, apply_lora, release_pipeline, events


@contextmanager
def _patched_failure_runtime(loader_name: str, tmp_path, failure: str):
    events: list[str] = []
    pipe = _FakePipeline(events)
    scheduler = object()
    if failure == "inference":
        pipe.inference_error = RuntimeError("synthetic inference failure")
    if failure == "cleanup":
        pipe.unload_error = RuntimeError("synthetic unload failure")

    def _apply_lora(*_args, **_kwargs):
        assert pipe.scheduler is scheduler
        events.append("apply")
        if failure == "adapter_load":
            raise RuntimeError("synthetic adapter load failure")
        return ["lora_Qwen"]

    def _release_pipeline(released_pipe, *, logger):
        assert released_pipe is pipe
        assert logger is qwen_image_pipeline.logger
        events.append("release")

    with (
        patch.object(qwen_image_pipeline, loader_name, return_value=pipe),
        patch.object(qwen_image_pipeline, "create_scheduler", return_value=scheduler),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", return_value=scheduler),
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=_normal_resolution,
        ),
        patch.object(
            qwen_image_pipeline,
            "_apply_qwen_lora_adapters",
            side_effect=_apply_lora,
        ) as apply_lora,
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="failure"),
        patch.object(
            qwen_image_pipeline,
            "get_batch_output_dir",
            return_value=tmp_path,
        ),
        patch.object(qwen_image_pipeline, "save_generated_image"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(
            qwen_image_pipeline,
            "release_pipeline",
            side_effect=_release_pipeline,
        ) as release_pipeline,
        patch.object(
            qwen_image_pipeline.torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        if failure == "cancellation":
            with patch.object(
                qwen_image_pipeline,
                "_raise_if_cancelled",
                side_effect=(None, SubprocessCanceled("Cancel requested")),
            ):
                yield pipe, apply_lora, release_pipeline, events
        else:
            yield pipe, apply_lora, release_pipeline, events


_MODE_CASES = (
    (
        "text2img",
        qwen_image_pipeline.generate_text2img_in_process,
        "load_text2img_pipeline",
        {"prompt": "test", "seed": 11},
    ),
    (
        "img2img",
        qwen_image_pipeline.generate_img2img_in_process,
        "load_img2img_pipeline",
        {
            "prompt": "test",
            "initial_image": Image.new("RGB", (64, 48), "blue"),
            "seed": 11,
        },
    ),
    (
        "inpaint",
        qwen_image_pipeline.generate_inpaint_in_process,
        "load_inpaint_pipeline",
        {
            "prompt": "test",
            "initial_image": Image.new("RGB", (64, 48), "blue"),
            "mask_image": Image.new("L", (64, 48), "white"),
            "seed": 11,
        },
    ),
)


def test_qwen_lora_adapter_list_validation():
    assert qwen_image_pipeline._qwen_lora_adapters({}) is None
    adapters = [{"lora_id": 101, "strength": 0.8}]
    assert qwen_image_pipeline._qwen_lora_adapters(
        {"lora_adapters": adapters}
    ) is adapters

    with pytest.raises(ValueError, match="lora_adapters must be a list"):
        qwen_image_pipeline._qwen_lora_adapters(
            {"lora_adapters": {"lora_id": 101}}
        )


def test_apply_qwen_lora_adapters_uses_transformer_rules(tmp_path):
    pipe = object()
    adapters = [{"lora_id": 101, "strength": 0.8}]
    coverage = {"lora_Qwen": {"transformer": {"adapter_present": True}}}
    report_path = tmp_path / "batch_lora_coverage.json"

    with (
        patch.object(
            qwen_image_pipeline,
            "apply_lora_adapters_with_validation",
            return_value=(["lora_Qwen"], coverage),
        ) as apply_lora,
        patch.object(
            qwen_image_pipeline,
            "write_lora_coverage_report",
            return_value=report_path,
        ) as write_report,
    ):
        adapter_names = qwen_image_pipeline._apply_qwen_lora_adapters(
            pipe,
            adapters,
            batch_output_dir=tmp_path,
            batch_id="batch",
        )

    assert adapter_names == ["lora_Qwen"]
    apply_lora.assert_called_once_with(
        pipe,
        adapters,
        expected_family="qwen-image",
        validate=True,
        allowed_lora_types=("lora",),
        allowed_targets=("both",),
        coverage_components=("transformer",),
        resolved_entries=None,
    )
    write_report.assert_called_once_with(tmp_path, "batch", coverage)


def test_apply_qwen_lora_adapters_skips_empty_list(tmp_path):
    with (
        patch.object(
            qwen_image_pipeline,
            "apply_lora_adapters_with_validation",
        ) as apply_lora,
        patch.object(
            qwen_image_pipeline,
            "write_lora_coverage_report",
        ) as write_report,
    ):
        adapter_names = qwen_image_pipeline._apply_qwen_lora_adapters(
            object(),
            [],
            batch_output_dir=tmp_path,
            batch_id="batch",
        )

    assert adapter_names == []
    apply_lora.assert_not_called()
    write_report.assert_not_called()


@pytest.mark.parametrize(
    ("order", "expected_names", "expected_weights", "expected_paths"),
    (
        (
            ("lightning", "companion"),
            ["lora_Shared_adapter", "lora_Shared_adapter_202"],
            [1.0, 0.5],
            ["C:/loras/lightning.safetensors", "C:/loras/companion.safetensors"],
        ),
        (
            ("companion", "lightning"),
            ["lora_Shared_adapter", "lora_Shared_adapter_101"],
            [0.5, 1.0],
            ["C:/loras/companion.safetensors", "C:/loras/lightning.safetensors"],
        ),
    ),
)
def test_apply_qwen_lora_adapters_activates_resolved_mixed_stack_in_request_order(
    order,
    expected_names,
    expected_weights,
    expected_paths,
    tmp_path,
    caplog,
):
    adapters, resolution, entries = _mixed_stack_inputs(order)
    pipe = _MixedStackPipeline()

    caplog.set_level("INFO", logger=qwen_image_pipeline.logger.name)
    with patch.object(
        qwen_image_pipeline,
        "write_lora_coverage_report",
        return_value=tmp_path / "coverage.json",
    ) as write_report:
        adapter_names = qwen_image_pipeline._apply_qwen_lora_adapters(
            pipe,
            adapters,
            batch_output_dir=tmp_path,
            batch_id="mixed",
            resolved_entries=entries,
            resolution=resolution,
        )

    assert adapter_names == expected_names
    assert pipe.load_calls == [
        (file_path, {"adapter_name": adapter_name})
        for file_path, adapter_name in zip(expected_paths, expected_names, strict=True)
    ]
    assert pipe.set_calls == [(expected_names, expected_weights)]
    assert pipe.get_active_adapters() == expected_names
    pipe.fuse_lora.assert_not_called()
    coverage = write_report.call_args.args[2]
    assert set(coverage) == set(expected_names)
    assert all(set(item) == {"transformer"} for item in coverage.values())
    write_report.assert_called_once_with(tmp_path, "mixed", coverage)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"adapter_ids={[adapter['lora_id'] for adapter in adapters]}" in messages
    assert f"adapter_names={expected_names}" in messages
    assert f"strengths={expected_weights}" in messages
    assert "base_variant=qwen-image-2512 task=text2img lightning_steps=4" in messages


def test_apply_qwen_lora_adapters_requires_set_adapters_before_mixed_load(tmp_path):
    adapters, resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    pipe = _MissingSetAdaptersPipeline()

    with pytest.raises(RuntimeError, match="requires callable pipeline.set_adapters support"):
        qwen_image_pipeline._apply_qwen_lora_adapters(
            pipe,
            adapters,
            batch_output_dir=tmp_path,
            batch_id="mixed",
            resolved_entries=entries,
            resolution=resolution,
        )

    assert pipe.load_calls == []


def test_apply_qwen_lora_adapters_rejects_missing_active_mixed_adapter(tmp_path):
    adapters, resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    pipe = _MixedStackPipeline(reported_active_adapters=["lora_Shared_adapter"])

    with pytest.raises(RuntimeError, match="activation is missing adapter names"):
        qwen_image_pipeline._apply_qwen_lora_adapters(
            pipe,
            adapters,
            batch_output_dir=tmp_path,
            batch_id="mixed",
            resolved_entries=entries,
            resolution=resolution,
        )

    assert len(pipe.load_calls) == 2
    assert pipe.set_calls == [
        (["lora_Shared_adapter", "lora_Shared_adapter_202"], [1.0, 0.5])
    ]


def test_apply_qwen_lora_adapters_allows_missing_active_adapter_query(tmp_path):
    adapters, resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    pipe = _NoActiveAdapterQueryPipeline()

    with patch.object(
        qwen_image_pipeline,
        "write_lora_coverage_report",
        return_value=tmp_path / "coverage.json",
    ):
        adapter_names = qwen_image_pipeline._apply_qwen_lora_adapters(
            pipe,
            adapters,
            batch_output_dir=tmp_path,
            batch_id="mixed",
            resolved_entries=entries,
            resolution=resolution,
        )

    assert adapter_names == ["lora_Shared_adapter", "lora_Shared_adapter_202"]
    assert pipe.set_calls == [(adapter_names, [1.0, 0.5])]


def test_mixed_stack_second_adapter_load_failure_cleans_up_before_release(tmp_path):
    adapters, _resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    events: list[str] = []
    pipe = _LifecycleMixedStackPipeline(events, failure="second_adapter_load")
    params = {
        "prompt": "mixed load failure",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "lora_adapters": adapters,
    }

    with _patched_mixed_generation("load_text2img_pipeline", pipe, entries, tmp_path):
        with pytest.raises(RuntimeError, match="synthetic second adapter load failure"):
            qwen_image_pipeline.generate_text2img_in_process(params)

    assert [path for path, _kwargs in pipe.load_calls] == [
        "C:/loras/lightning.safetensors",
        "C:/loras/companion.safetensors",
    ]
    assert pipe.set_calls == []
    assert pipe.unload_calls == 1
    assert events[-2:] == ["unload", "release"]


def test_mixed_stack_active_adapter_failure_cleans_up_before_release(tmp_path):
    adapters, _resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    events: list[str] = []
    pipe = _LifecycleMixedStackPipeline(
        events,
        reported_active_adapters=["lora_Shared_adapter"],
    )
    params = {
        "prompt": "mixed active failure",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "lora_adapters": adapters,
    }

    with _patched_mixed_generation("load_text2img_pipeline", pipe, entries, tmp_path):
        with pytest.raises(RuntimeError, match="activation is missing adapter names"):
            qwen_image_pipeline.generate_text2img_in_process(params)

    assert pipe.set_calls == [
        (["lora_Shared_adapter", "lora_Shared_adapter_202"], [1.0, 0.5])
    ]
    assert pipe.active_query_calls == 1
    assert pipe.unload_calls == 1
    assert events[-2:] == ["unload", "release"]


@pytest.mark.parametrize(
    ("mode", "generation_function", "loader_name", "base_params"),
    _MODE_CASES,
    ids=[case[0] for case in _MODE_CASES],
)
@pytest.mark.parametrize("failure", ("inference", "cancellation"))
def test_mixed_stack_failure_and_cancellation_clean_up_all_qwen_modes(
    mode,
    generation_function,
    loader_name,
    base_params,
    failure,
    tmp_path,
):
    adapters, _resolution, entries = _mixed_stack_inputs(("companion", "lightning"), mode)
    events: list[str] = []
    pipe = _LifecycleMixedStackPipeline(
        events,
        failure="inference" if failure == "inference" else None,
    )
    params = {
        **base_params,
        "steps": 4,
        "true_cfg_scale": 1.0,
        "lora_adapters": adapters,
    }

    with _patched_mixed_generation(loader_name, pipe, entries, tmp_path):
        if failure == "inference":
            with pytest.raises(RuntimeError, match="synthetic mixed inference failure"):
                generation_function(params)
        else:
            with patch.object(
                qwen_image_pipeline,
                "_raise_if_cancelled",
                side_effect=(None, None, SubprocessCanceled("Cancel requested")),
            ):
                with pytest.raises(SubprocessCanceled, match="Cancel requested"):
                    generation_function(params)

    assert events.count("inference") == 1
    assert pipe.unload_calls == 1
    assert events[-2:] == ["unload", "release"]


@pytest.mark.parametrize(
    ("mode", "generation_function", "loader_name", "base_params"),
    (
        (
            "text2img",
            qwen_image_pipeline.generate_text2img_in_process,
            "load_text2img_pipeline",
            {"prompt": "mixed text", "negative_prompt": "text negative"},
        ),
        (
            "img2img",
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {
                "prompt": "mixed image",
                "negative_prompt": "image negative",
                "initial_image": Image.new("RGB", (64, 48), "blue"),
                "strength": 0.37,
            },
        ),
        (
            "inpaint",
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "prompt": "mixed inpaint",
                "negative_prompt": "inpaint negative",
                "initial_image": Image.new("RGB", (64, 48), "blue"),
                "mask_image": Image.new("L", (64, 48), "white"),
                "strength": 0.42,
                "padding_mask_crop": 9,
            },
        ),
    ),
)
def test_successful_mixed_stack_preserves_qwen_request_data(
    mode,
    generation_function,
    loader_name,
    base_params,
    tmp_path,
):
    adapters, _resolution, entries = _mixed_stack_inputs(("companion", "lightning"), mode)
    events: list[str] = []
    pipe = _LifecycleMixedStackPipeline(events)
    params = {
        **base_params,
        "steps": 4,
        "true_cfg_scale": 1.0,
        "lora_adapters": adapters,
    }

    with _patched_mixed_generation(loader_name, pipe, entries, tmp_path) as (
        mixed_scheduler,
        _base_scheduler,
    ):
        result = generation_function(params)

    assert result == {"images": ["/outputs/batch_mixed/output.png"]}
    assert pipe.scheduler is mixed_scheduler
    assert pipe.load_calls == [
        ("C:/loras/companion.safetensors", {"adapter_name": "lora_Shared_adapter"}),
        ("C:/loras/lightning.safetensors", {"adapter_name": "lora_Shared_adapter_101"}),
    ]
    assert pipe.set_calls == [
        (["lora_Shared_adapter", "lora_Shared_adapter_101"], [0.5, 1.0])
    ]
    assert pipe.unload_calls == 1
    assert events[-2:] == ["unload", "release"]
    call_kwargs = pipe.call_arguments[0]
    assert call_kwargs["prompt"] == base_params["prompt"]
    assert call_kwargs["negative_prompt"] == base_params["negative_prompt"]
    assert call_kwargs["num_inference_steps"] == 4
    assert call_kwargs["true_cfg_scale"] == 1.0
    if mode != "text2img":
        assert call_kwargs["image"] is base_params["initial_image"]
        assert call_kwargs["strength"] == base_params["strength"]
    if mode == "inpaint":
        assert call_kwargs["mask_image"] is base_params["mask_image"]
        assert call_kwargs["padding_mask_crop"] == 9


def test_mixed_request_then_base_request_has_no_adapter_state_leak(tmp_path):
    adapters, _resolution, entries = _mixed_stack_inputs(("lightning", "companion"))
    events: list[str] = []
    mixed_pipe = _LifecycleMixedStackPipeline(events)
    base_pipe = _LifecycleMixedStackPipeline(events)
    mixed_params = {
        "prompt": "mixed first",
        "steps": 4,
        "true_cfg_scale": 1.0,
        "lora_adapters": adapters,
    }
    base_params = {"prompt": "base second"}

    with _patched_mixed_generation(
        "load_text2img_pipeline",
        mixed_pipe,
        entries,
        tmp_path,
        additional_pipelines=[base_pipe],
    ) as (mixed_scheduler, base_scheduler):
        qwen_image_pipeline.generate_text2img_in_process(mixed_params)
        qwen_image_pipeline.generate_text2img_in_process(base_params)

    assert mixed_pipe.scheduler is mixed_scheduler
    assert mixed_pipe.unload_calls == 1
    assert base_pipe.scheduler is base_scheduler
    assert base_pipe.load_calls == []
    assert base_pipe.set_calls == []
    assert base_pipe.active_query_calls == 0
    assert base_pipe.unload_calls == 0
    assert events.index("unload") < events.index("release") < events.index("base_scheduler")


def test_cleanup_qwen_lora_adapters_unloads_requested_weights():
    unload_lora_weights = Mock()
    pipe = SimpleNamespace(unload_lora_weights=unload_lora_weights)

    qwen_image_pipeline._cleanup_qwen_lora_adapters(pipe, requested=True)

    unload_lora_weights.assert_called_once_with()


def test_cleanup_qwen_lora_adapters_is_best_effort():
    unload_lora_weights = Mock(side_effect=RuntimeError("synthetic unload failure"))
    pipe = SimpleNamespace(unload_lora_weights=unload_lora_weights)

    qwen_image_pipeline._cleanup_qwen_lora_adapters(pipe, requested=True)

    unload_lora_weights.assert_called_once_with()


def test_cleanup_qwen_lora_adapters_skips_request_without_lora():
    unload_lora_weights = Mock()
    pipe = SimpleNamespace(unload_lora_weights=unload_lora_weights)

    qwen_image_pipeline._cleanup_qwen_lora_adapters(pipe, requested=False)

    unload_lora_weights.assert_not_called()


def test_text2img_applies_lora_once_and_unloads_after_all_images(tmp_path):
    adapters = [
        {"lora_id": 101, "strength": 0.8},
        {"lora_id": 102, "strength": 0.4},
    ]
    result, pipe, apply_lora, release_pipeline, events = _run_generation_with_lora(
        qwen_image_pipeline.generate_text2img_in_process,
        "load_text2img_pipeline",
        {
            "prompt": "test",
            "seed": 11,
            "num_images": 2,
            "lora_adapters": adapters,
        },
        tmp_path,
    )

    assert result == {
        "images": [
            "/outputs/batch_lora/output.png",
            "/outputs/batch_lora/output.png",
        ]
    }
    apply_lora.assert_called_once_with(
        pipe,
        adapters,
        batch_output_dir=tmp_path,
        batch_id="lora",
        resolved_entries={101: ANY, 102: ANY},
        resolution=ANY,
    )
    assert len(pipe.call_arguments) == 2
    assert events == ["apply", "inference", "inference", "unload", "release"]
    release_pipeline.assert_called_once()


def test_img2img_applies_lora_and_unloads_after_inference(tmp_path):
    adapters = [{"lora_id": 201, "strength": 0.7}]
    initial_image = Image.new("RGB", (64, 48), "blue")
    result, pipe, apply_lora, release_pipeline, events = _run_generation_with_lora(
        qwen_image_pipeline.generate_img2img_in_process,
        "load_img2img_pipeline",
        {
            "prompt": "test",
            "initial_image": initial_image,
            "seed": 11,
            "lora_adapters": adapters,
        },
        tmp_path,
    )

    assert result == {"images": ["/outputs/batch_lora/output.png"]}
    apply_lora.assert_called_once_with(
        pipe,
        adapters,
        batch_output_dir=tmp_path,
        batch_id="lora",
        resolved_entries={201: ANY},
        resolution=ANY,
    )
    assert pipe.call_arguments[0]["image"] is initial_image
    assert events == ["apply", "inference", "unload", "release"]
    release_pipeline.assert_called_once()


def test_inpaint_applies_lora_and_unloads_after_inference(tmp_path):
    adapters = [{"lora_id": 301, "strength": 0.6}]
    initial_image = Image.new("RGB", (64, 48), "blue")
    mask_image = Image.new("L", (64, 48), "white")
    result, pipe, apply_lora, release_pipeline, events = _run_generation_with_lora(
        qwen_image_pipeline.generate_inpaint_in_process,
        "load_inpaint_pipeline",
        {
            "prompt": "test",
            "initial_image": initial_image,
            "mask_image": mask_image,
            "seed": 11,
            "lora_adapters": adapters,
        },
        tmp_path,
    )

    assert result == {"images": ["/outputs/batch_lora/output.png"]}
    apply_lora.assert_called_once_with(
        pipe,
        adapters,
        batch_output_dir=tmp_path,
        batch_id="lora",
        resolved_entries={301: ANY},
        resolution=ANY,
    )
    assert pipe.call_arguments[0]["image"] is initial_image
    assert pipe.call_arguments[0]["mask_image"] is mask_image
    assert events == ["apply", "inference", "unload", "release"]
    release_pipeline.assert_called_once()


@pytest.mark.parametrize(
    ("_mode", "generation_function", "loader_name", "base_params"),
    _MODE_CASES,
    ids=[case[0] for case in _MODE_CASES],
)
def test_qwen_lora_cleanup_after_adapter_load_failure(
    _mode,
    generation_function,
    loader_name,
    base_params,
    tmp_path,
):
    params = {
        **base_params,
        "lora_adapters": [{"lora_id": 401, "strength": 0.8}],
    }
    with _patched_failure_runtime(loader_name, tmp_path, "adapter_load") as runtime:
        _pipe, apply_lora, release_pipeline, events = runtime
        with pytest.raises(RuntimeError, match="synthetic adapter load failure"):
            generation_function(params)

    apply_lora.assert_called_once()
    release_pipeline.assert_called_once()
    assert events == ["apply", "unload", "release"]


@pytest.mark.parametrize(
    ("_mode", "generation_function", "loader_name", "base_params"),
    _MODE_CASES,
    ids=[case[0] for case in _MODE_CASES],
)
def test_qwen_lora_cleanup_after_inference_failure(
    _mode,
    generation_function,
    loader_name,
    base_params,
    tmp_path,
):
    params = {
        **base_params,
        "lora_adapters": [{"lora_id": 402, "strength": 0.8}],
    }
    with _patched_failure_runtime(loader_name, tmp_path, "inference") as runtime:
        _pipe, apply_lora, release_pipeline, events = runtime
        with pytest.raises(RuntimeError, match="synthetic inference failure"):
            generation_function(params)

    apply_lora.assert_called_once()
    release_pipeline.assert_called_once()
    assert events == ["apply", "inference", "unload", "release"]


@pytest.mark.parametrize(
    ("_mode", "generation_function", "loader_name", "base_params"),
    _MODE_CASES,
    ids=[case[0] for case in _MODE_CASES],
)
def test_qwen_lora_cleanup_after_cancellation(
    _mode,
    generation_function,
    loader_name,
    base_params,
    tmp_path,
):
    params = {
        **base_params,
        "lora_adapters": [{"lora_id": 403, "strength": 0.8}],
    }
    with _patched_failure_runtime(loader_name, tmp_path, "cancellation") as runtime:
        _pipe, apply_lora, release_pipeline, events = runtime
        with pytest.raises(SubprocessCanceled, match="Cancel requested"):
            generation_function(params)

    apply_lora.assert_called_once()
    release_pipeline.assert_called_once()
    assert events == ["apply", "unload", "release"]


@pytest.mark.parametrize(
    ("_mode", "generation_function", "loader_name", "base_params"),
    _MODE_CASES,
    ids=[case[0] for case in _MODE_CASES],
)
def test_qwen_pipeline_release_runs_after_lora_cleanup_failure(
    _mode,
    generation_function,
    loader_name,
    base_params,
    tmp_path,
):
    params = {
        **base_params,
        "lora_adapters": [{"lora_id": 404, "strength": 0.8}],
    }
    with _patched_failure_runtime(loader_name, tmp_path, "cleanup") as runtime:
        _pipe, apply_lora, release_pipeline, events = runtime
        generation_function(params)

    apply_lora.assert_called_once()
    release_pipeline.assert_called_once()
    assert events == ["apply", "inference", "unload", "release"]
