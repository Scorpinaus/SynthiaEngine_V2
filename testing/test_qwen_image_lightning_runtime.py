from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from backend.lora.registry import LoraCompatibility, LoraRegistryEntry
from backend.qwen_image.lightning import resolve_qwen_image_lightning_profile as resolve_lightning_profile
from backend.qwen_image import pipeline as qwen_image_pipeline
from backend.registries.model import ModelRegistryEntry


class _FakePipeline:
    def __init__(self, events, *, inference_error=None, unload_error=None):
        self.events = events
        self.scheduler = object()
        self.calls = []
        self.unload_calls = 0
        self.inference_error = inference_error
        self.unload_error = unload_error

    def __call__(self, **kwargs):
        self.events.append("inference")
        self.calls.append(kwargs)
        if self.inference_error is not None:
            raise self.inference_error
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

    def unload_lora_weights(self):
        self.events.append("unload")
        self.unload_calls += 1
        if self.unload_error is not None:
            raise self.unload_error


def _lightning_entry(lora_id, steps):
    return LoraRegistryEntry(
        lora_id=lora_id,
        lora_model_family="qwen-image",
        lora_type="lora",
        lora_location="local",
        file_path=f"C:/loras/lightning-{steps}.safetensors",
        runtime_profile={
            "kind": "qwen_image_lightning",
            "base_variant": "qwen-image-2512",
            "steps": steps,
            "true_cfg_scale": 1.0,
            "scheduler_profile": "qwen_image_lightning_shift3",
            "adapter_strength": 1.0,
            "supported_tasks": ["text2img"],
        },
    )


def _model_entry():
    return ModelRegistryEntry(
        name="Qwen-Image-2512-SDNQ-4bit-dynamic",
        family="qwen-image",
        model_type="diffusers",
        location_type="local",
        model_id=1,
        version="local",
        link="C:/models/Qwen-Image-2512-SDNQ-4bit-dynamic",
    )


def _companion_entry(lora_id, compatibility):
    return LoraRegistryEntry(
        lora_id=lora_id,
        lora_model_family="qwen-image",
        lora_type="lora",
        lora_location="local",
        file_path=f"C:/loras/companion-{lora_id}.safetensors",
        compatibility=compatibility,
    )


def _compatibility(*, supported_tasks=None):
    return {
        "base_variants": ["qwen-image-2512"],
        "runtime_profile_kinds": ["qwen_image_lightning"],
        "supported_tasks": supported_tasks or ["text2img", "img2img", "inpaint"],
    }


def _real_resolver_with_entries(entries, events):
    def resolve(adapters, model_entry, task, steps, true_cfg_scale):
        events.append("resolve")
        return resolve_lightning_profile(
            adapters,
            model_entry,
            task,
            steps,
            true_cfg_scale,
            lookup_lora_entry=entries.__getitem__,
        )

    return resolve


@pytest.mark.parametrize("steps", [4, 8])
def test_text2img_resolves_before_load_and_executes_validated_lightning_values(tmp_path, steps):
    events = []
    entry = _lightning_entry(901 + steps, steps)
    adapters = [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}]
    resolution = SimpleNamespace(
        adapters=(SimpleNamespace(lora_id=entry.lora_id, entry=entry),),
        steps=steps,
        true_cfg_scale=1.0,
        lightning_profile=entry.runtime_profile,
    )
    pipe = _FakePipeline(events)
    fixed_scheduler = object()

    def resolve(*args):
        events.append("resolve")
        assert args[0] == adapters
        assert args[2:] == ("text2img", steps, 1.0)
        return resolution

    def load(_model):
        events.append("load")
        return pipe

    def select(received_resolution, received_pipe):
        events.append("select")
        assert received_resolution is resolution
        assert received_pipe is pipe
        return fixed_scheduler

    def apply(received_pipe, received_adapters, **kwargs):
        events.append("apply")
        assert received_pipe is pipe
        assert received_adapters == adapters
        assert kwargs["resolved_entries"] == {entry.lora_id: entry}
        return ["lora_lightning"]

    with (
        patch.object(qwen_image_pipeline, "resolve_qwen_image_lightning_profile", side_effect=resolve),
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", side_effect=load),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=select),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=apply) as apply_lora,
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="lightning"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "save_generated_image", return_value="batch_lightning/output.png"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda *_args, **_kwargs: events.append("release")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
    ):
        result = qwen_image_pipeline.generate_text2img_in_process(
            {"prompt": "test", "steps": steps, "true_cfg_scale": 1.0, "lora_adapters": adapters}
        )

    assert result == {"images": ["/outputs/batch_lightning/output.png"]}
    assert pipe.scheduler is fixed_scheduler
    assert pipe.calls[0]["num_inference_steps"] == steps
    assert pipe.calls[0]["true_cfg_scale"] == 1.0
    assert events == ["resolve", "load", "select", "apply", "inference", "unload", "release"]
    apply_lora.assert_called_once()


def test_text2img_passes_compatible_lightning_companion_in_request_order(tmp_path):
    events = []
    lightning = _lightning_entry(910, 4)
    companion = _companion_entry(911, _compatibility(supported_tasks=["text2img"]))
    adapters = [
        {"lora_id": companion.lora_id, "strength": 0.5, "target": "both"},
        {"lora_id": lightning.lora_id, "strength": 1.0, "target": "both"},
    ]
    pipe = _FakePipeline(events)
    fixed_scheduler = object()

    def apply(received_pipe, received_adapters, **kwargs):
        events.append("apply")
        assert received_pipe is pipe
        assert received_adapters == adapters
        assert list(kwargs["resolved_entries"]) == [companion.lora_id, lightning.lora_id]
        assert [entry.lora_id for entry in kwargs["resolved_entries"].values()] == [
            companion.lora_id,
            lightning.lora_id,
        ]
        return ["companion", "lightning"]

    with (
        patch.object(qwen_image_pipeline, "_get_qwen_image_model_entry", return_value=_model_entry()),
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=_real_resolver_with_entries(
                {lightning.lora_id: lightning, companion.lora_id: companion}, events
            ),
        ),
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", side_effect=lambda _model: (events.append("load"), pipe)[1]),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=lambda *_args: (events.append("select"), fixed_scheduler)[1]),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=apply),
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="compatible"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "save_generated_image", return_value="batch_compatible/output.png"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda *_args, **_kwargs: events.append("release")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
    ):
        result = qwen_image_pipeline.generate_text2img_in_process(
            {"prompt": "test", "steps": 4, "true_cfg_scale": 1.0, "lora_adapters": adapters}
        )

    assert result == {"images": ["/outputs/batch_compatible/output.png"]}
    assert pipe.scheduler is fixed_scheduler
    assert events == ["resolve", "load", "select", "apply", "inference", "unload", "release"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.mark.parametrize(
    ("failure_stage", "expected_exception", "expected_events"),
    [
        ("adapter", ValueError, ["resolve", "load", "select", "apply", "unload", "release"]),
        (
            "inference",
            RuntimeError,
            ["resolve", "load", "select", "apply", "inference", "unload", "release"],
        ),
        (
            "cancellation",
            qwen_image_pipeline.SubprocessCanceled,
            ["resolve", "load", "select", "apply", "unload", "release"],
        ),
    ],
)
def test_lightning_text2img_failures_unload_before_pipeline_release(
    tmp_path, failure_stage, expected_exception, expected_events
):
    events = []
    entry = _lightning_entry(940, 4)
    adapters = [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}]
    resolution = SimpleNamespace(
        adapters=(SimpleNamespace(lora_id=entry.lora_id, entry=entry),),
        steps=4,
        true_cfg_scale=1.0,
        lightning_profile=entry.runtime_profile,
    )
    pipe = _FakePipeline(
        events,
        inference_error=RuntimeError("inference failed") if failure_stage == "inference" else None,
    )

    def apply(*_args, **_kwargs):
        events.append("apply")
        if failure_stage == "adapter":
            raise ValueError("adapter load failed")

    cancellation = (
        [None, qwen_image_pipeline.SubprocessCanceled("Cancel requested")]
        if failure_stage == "cancellation"
        else None
    )
    with (
        patch.object(qwen_image_pipeline, "resolve_qwen_image_lightning_profile", side_effect=lambda *_args: (events.append("resolve"), resolution)[1]),
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", side_effect=lambda *_args: (events.append("load"), pipe)[1]),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=lambda *_args: (events.append("select"), object())[1]),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=apply),
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="lightning-failure"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda *_args, **_kwargs: events.append("release")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
        patch.object(qwen_image_pipeline, "_raise_if_cancelled", side_effect=cancellation),
    ):
        with pytest.raises(expected_exception):
            qwen_image_pipeline.generate_text2img_in_process(
                {"prompt": "test", "steps": 4, "true_cfg_scale": 1.0, "lora_adapters": adapters}
            )

    assert events == expected_events


def test_lightning_cleanup_failure_still_releases_pipeline(tmp_path):
    events = []
    entry = _lightning_entry(941, 8)
    adapters = [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}]
    resolution = SimpleNamespace(
        adapters=(SimpleNamespace(lora_id=entry.lora_id, entry=entry),),
        steps=8,
        true_cfg_scale=1.0,
        lightning_profile=entry.runtime_profile,
    )
    pipe = _FakePipeline(events, unload_error=RuntimeError("cleanup failed"))

    with (
        patch.object(qwen_image_pipeline, "resolve_qwen_image_lightning_profile", side_effect=lambda *_args: (events.append("resolve"), resolution)[1]),
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", side_effect=lambda *_args: (events.append("load"), pipe)[1]),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=lambda *_args: (events.append("select"), object())[1]),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=lambda *_args, **_kwargs: events.append("apply")),
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="lightning-cleanup"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "save_generated_image", return_value="out.png"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda *_args, **_kwargs: events.append("release")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
    ):
        qwen_image_pipeline.generate_text2img_in_process(
            {"prompt": "test", "steps": 8, "true_cfg_scale": 1.0, "lora_adapters": adapters}
        )

    assert events == ["resolve", "load", "select", "apply", "inference", "unload", "release"]


def test_lightning_request_does_not_leak_adapter_or_scheduler_state_to_base_request(tmp_path):
    events = []
    entry = _lightning_entry(942, 4)
    lightning_adapters = [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}]
    lightning_resolution = SimpleNamespace(
        adapters=(SimpleNamespace(lora_id=entry.lora_id, entry=entry),),
        steps=4,
        true_cfg_scale=1.0,
        lightning_profile=entry.runtime_profile,
    )
    base_resolution = SimpleNamespace(
        adapters=(),
        steps=20,
        true_cfg_scale=4.0,
        lightning_profile=None,
    )
    lightning_pipe = _FakePipeline(events)
    base_pipe = _FakePipeline(events)
    fixed_scheduler = object()
    base_scheduler = object()

    def resolve(adapters, *_args):
        events.append("resolve")
        return lightning_resolution if adapters else base_resolution

    def select(resolution, pipe):
        if resolution.lightning_profile is not None:
            events.append("select-lightning")
            assert pipe is lightning_pipe
            return fixed_scheduler
        events.append("select-base")
        assert pipe is base_pipe
        return base_scheduler

    def apply(pipe, adapters, **_kwargs):
        if adapters:
            events.append("apply-lightning")
            assert pipe is lightning_pipe
        else:
            events.append("apply-base")
            assert pipe is base_pipe
        return []

    loaded_pipes = iter((lightning_pipe, base_pipe))
    with (
        patch.object(qwen_image_pipeline, "resolve_qwen_image_lightning_profile", side_effect=resolve),
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", side_effect=lambda _model: (events.append("load"), next(loaded_pipes))[1]),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=select),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=apply),
        patch.object(qwen_image_pipeline, "make_batch_id", side_effect=("lightning", "base")),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "save_generated_image", return_value="out.png"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda pipe, **_kwargs: events.append("release-lightning" if pipe is lightning_pipe else "release-base")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
    ):
        qwen_image_pipeline.generate_text2img_in_process(
            {"prompt": "lightning", "steps": 4, "true_cfg_scale": 1.0, "lora_adapters": lightning_adapters}
        )
        qwen_image_pipeline.generate_text2img_in_process(
            {"prompt": "base", "steps": 20, "true_cfg_scale": 4.0}
        )

    assert lightning_pipe.scheduler is fixed_scheduler
    assert base_pipe.scheduler is base_scheduler
    assert base_pipe.scheduler is not fixed_scheduler
    assert base_pipe.calls[0]["num_inference_steps"] == 20
    assert base_pipe.calls[0]["true_cfg_scale"] == 4.0
    assert lightning_pipe.unload_calls == 1
    assert base_pipe.unload_calls == 0
    assert events == [
        "resolve", "load", "select-lightning", "apply-lightning", "inference", "unload",
        "release-lightning", "resolve", "load", "select-base", "apply-base", "inference",
        "release-base",
    ]


@pytest.mark.parametrize(
    "message",
    [
        "requires base variant 'qwen-image-2512'",
        "requires steps=4; received 8.",
        "requires true_cfg_scale=1.0; received 2.0.",
        "requires strength=1.0; received 0.8.",
        "at most one standard companion LoRA.",
    ],
)
def test_invalid_text2img_lightning_validation_fails_before_pipeline_load(message):
    loader = Mock()
    release = Mock()
    with (
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=ValueError(message),
        ) as resolve,
        patch.object(qwen_image_pipeline, "load_text2img_pipeline", loader),
        patch.object(qwen_image_pipeline, "release_pipeline", release),
    ):
        with pytest.raises(ValueError, match=message):
            qwen_image_pipeline.generate_text2img_in_process(
                {"prompt": "test", "lora_adapters": [{"lora_id": 999, "strength": 1.0}]}
            )

    assert resolve.call_args.args[2] == "text2img"
    loader.assert_not_called()
    release.assert_not_called()


@pytest.mark.parametrize(
    ("generation", "loader_name", "params", "task", "steps"),
    [
        (
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {"initial_image": Image.new("RGB", (8, 8), "blue"), "strength": 0.35},
            "img2img",
            4,
        ),
        (
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "initial_image": Image.new("RGB", (8, 8), "blue"),
                "mask_image": Image.new("L", (8, 8), "white"),
                "strength": 0.45,
            },
            "inpaint",
            8,
        ),
    ],
)
def test_img2img_and_inpaint_execute_lightning_with_fixed_scheduler(
    tmp_path, generation, loader_name, params, task, steps
):
    events = []
    entry = _lightning_entry(920 + steps, steps)
    adapters = [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}]
    resolution = SimpleNamespace(
        adapters=(SimpleNamespace(lora_id=entry.lora_id, entry=entry),),
        steps=steps,
        true_cfg_scale=1.0,
        lightning_profile=entry.runtime_profile,
    )
    pipe = _FakePipeline(events)
    fixed_scheduler = object()

    def resolve(*args):
        events.append("resolve")
        assert args[2:] == (task, steps, 1.0)
        return resolution

    with (
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=resolve,
        ) as resolve,
        patch.object(qwen_image_pipeline, loader_name, side_effect=lambda _model: (events.append("load"), pipe)[1]),
        patch.object(qwen_image_pipeline, "select_qwen_image_scheduler", side_effect=lambda *_args: (events.append("scheduler"), fixed_scheduler)[1]),
        patch.object(qwen_image_pipeline, "_apply_qwen_lora_adapters", side_effect=lambda *_args, **_kwargs: events.append("apply")),
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="lightning"),
        patch.object(qwen_image_pipeline, "get_batch_output_dir", return_value=tmp_path),
        patch.object(qwen_image_pipeline, "save_generated_image", return_value="out.png"),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline", side_effect=lambda *_args, **_kwargs: events.append("release")),
        patch.object(qwen_image_pipeline.torch, "autocast", side_effect=lambda *_args, **_kwargs: _NullContext()),
    ):
        generation({**params, "steps": steps, "true_cfg_scale": 1.0, "lora_adapters": adapters})

    assert pipe.scheduler is fixed_scheduler
    assert pipe.calls[0]["num_inference_steps"] == steps
    assert pipe.calls[0]["true_cfg_scale"] == 1.0
    assert pipe.calls[0]["strength"] == params["strength"]
    assert pipe.calls[0]["image"] is params["initial_image"]
    if task == "inpaint":
        assert pipe.calls[0]["mask_image"] is params["mask_image"]
    assert events == ["resolve", "load", "scheduler", "apply", "inference", "unload", "release"]


@pytest.mark.parametrize(
    ("generation", "loader_name", "params", "task"),
    [
        (
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {"initial_image": Image.new("RGB", (8, 8), "blue")},
            "img2img",
        ),
        (
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "initial_image": Image.new("RGB", (8, 8), "blue"),
                "mask_image": Image.new("L", (8, 8), "white"),
            },
            "inpaint",
        ),
    ],
)
def test_image_task_lightning_profile_failures_happen_before_pipeline_load(
    generation, loader_name, params, task
):
    loader = Mock()
    release = Mock()
    with (
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=ValueError("requires base variant 'qwen-image-2512'"),
        ) as resolve,
        patch.object(qwen_image_pipeline, loader_name, loader),
        patch.object(qwen_image_pipeline, "release_pipeline", release),
    ):
        with pytest.raises(ValueError, match="requires base variant"):
            generation({**params, "lora_adapters": [{"lora_id": 999, "strength": 1.0}]})

    assert resolve.call_args.args[2] == task
    loader.assert_not_called()
    release.assert_not_called()


@pytest.mark.parametrize(
    ("generation", "loader_name", "params", "task", "invalid_compatibility", "message"),
    [
        (
            qwen_image_pipeline.generate_text2img_in_process,
            "load_text2img_pipeline",
            {"prompt": "test"},
            "text2img",
            {"base_variants": [], "runtime_profile_kinds": ["qwen_image_lightning"], "supported_tasks": ["text2img"]},
            "missing base variant 'qwen-image-2512'",
        ),
        (
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {"initial_image": Image.new("RGB", (8, 8), "blue"), "strength": 0.35},
            "img2img",
            {"base_variants": ["qwen-image-2512"], "runtime_profile_kinds": ["qwen_image_lightning"], "supported_tasks": ["text2img"]},
            "missing task 'img2img'",
        ),
        (
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "initial_image": Image.new("RGB", (8, 8), "blue"),
                "mask_image": Image.new("L", (8, 8), "white"),
                "strength": 0.45,
            },
            "inpaint",
            {"base_variants": ["qwen-image-2512"], "runtime_profile_kinds": [], "supported_tasks": ["inpaint"]},
            "missing runtime profile kind 'qwen_image_lightning'",
        ),
    ],
)
def test_incompatible_lightning_companion_rejects_before_pipeline_load(
    generation,
    loader_name,
    params,
    task,
    invalid_compatibility,
    message,
):
    events = []
    lightning = _lightning_entry(970, 4)
    companion = _companion_entry(971, _compatibility())
    companion.compatibility = LoraCompatibility.model_construct(**invalid_compatibility)
    adapters = [
        {"lora_id": lightning.lora_id, "strength": 1.0, "target": "both"},
        {"lora_id": companion.lora_id, "strength": 0.5, "target": "both"},
    ]
    loader = Mock()
    release = Mock()

    with (
        patch.object(qwen_image_pipeline, "_get_qwen_image_model_entry", return_value=_model_entry()),
        patch.object(
            qwen_image_pipeline,
            "resolve_qwen_image_lightning_profile",
            side_effect=_real_resolver_with_entries(
                {lightning.lora_id: lightning, companion.lora_id: companion}, events
            ),
        ),
        patch.object(qwen_image_pipeline, loader_name, loader),
        patch.object(qwen_image_pipeline, "release_pipeline", release),
    ):
        with pytest.raises(ValueError, match=message):
            generation({**params, "steps": 4, "true_cfg_scale": 1.0, "lora_adapters": adapters})

    assert events == ["resolve"]
    loader.assert_not_called()
    release.assert_not_called()
