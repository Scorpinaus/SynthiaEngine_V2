from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from PIL import Image
import pytest

from backend.qwen_image import pipeline as qwen_image_pipeline
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
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        lightning_profile=None,
    )


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
