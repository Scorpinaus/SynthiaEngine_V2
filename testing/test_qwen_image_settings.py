from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from backend.qwen_image import pipeline as qwen_image_pipeline
from backend.utilities.subprocess_transport import SubprocessCanceled, SubprocessRuntime
from backend.workflow import (
    QwenImageImg2ImgInputs,
    QwenImageInpaintInputs,
    QwenImageText2ImgInputs,
    build_workflow_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


class _FakePipeline:
    def __init__(self) -> None:
        self.call_arguments: dict[str, object] | None = None
        self.scheduler = object()

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.call_arguments = kwargs
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])


def _run_with_fake_pipeline(
    generation_function,
    loader_name: str,
    params: dict[str, object],
) -> tuple[dict[str, list[str]], _FakePipeline, object]:
    fake_pipe = _FakePipeline()
    scheduler = object()
    with (
        patch.object(qwen_image_pipeline, loader_name, return_value=fake_pipe),
        patch.object(
            qwen_image_pipeline,
            "create_scheduler",
            return_value=scheduler,
        ) as create_scheduler,
        patch.object(
            qwen_image_pipeline,
            "select_qwen_image_scheduler",
            return_value=scheduler,
        ) as select_scheduler,
        patch.object(qwen_image_pipeline, "make_batch_id", return_value="settings"),
        patch.object(
            qwen_image_pipeline,
            "get_batch_output_dir",
            return_value=ROOT,
        ),
        patch.object(
            qwen_image_pipeline,
            "save_generated_image",
            return_value="batch_settings/output.png",
        ),
        patch.object(qwen_image_pipeline, "cleanup_memory"),
        patch.object(qwen_image_pipeline, "release_pipeline"),
        patch.object(
            qwen_image_pipeline.torch,
            "autocast",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
    ):
        result = generation_function(params)

    if generation_function is qwen_image_pipeline.generate_text2img_in_process:
        select_scheduler.assert_called_once()
        create_scheduler.assert_not_called()
    else:
        create_scheduler.assert_called_once_with("flowmatch_euler", fake_pipe)
        select_scheduler.assert_not_called()
    return result, fake_pipe, scheduler


def test_qwen_image_workflow_and_catalog_defaults_match_model_card():
    text2img = QwenImageText2ImgInputs()
    img2img = QwenImageImg2ImgInputs(
        initial_image="@artifact:initial",
        prompt="test",
    )
    inpaint = QwenImageInpaintInputs(
        initial_image="@artifact:initial",
        mask_image="@artifact:mask",
        prompt="test",
    )

    for inputs in (text2img, img2img, inpaint):
        serialized_inputs = inputs.model_dump()
        assert inputs.negative_prompt == EXPECTED_NEGATIVE_PROMPT
        assert inputs.steps == 50
        assert inputs.true_cfg_scale == 4.0
        assert serialized_inputs["guidance_scale"] is None
        assert inputs.scheduler == "flowmatch_euler"
        assert inputs.live_preview is True

    assert (text2img.width, text2img.height) == (1328, 1328)
    assert (img2img.width, img2img.height) == (1328, 1328)
    assert img2img.strength == 0.6
    assert inpaint.strength == 0.6
    assert (inpaint.width, inpaint.height) == (1024, 1024)
    assert inpaint.padding_mask_crop is None

    catalog = build_workflow_catalog()
    for task_type in (
        "qwen-image.text2img",
        "qwen-image.img2img",
        "qwen-image.inpaint",
    ):
        task = catalog["tasks"][task_type]
        defaults = task["input_defaults"]
        assert defaults["negative_prompt"] == EXPECTED_NEGATIVE_PROMPT
        assert defaults["steps"] == 50
        assert defaults["true_cfg_scale"] == 4.0
        assert defaults["guidance_scale"] is None
        assert defaults["scheduler"] == "flowmatch_euler"
        assert defaults["live_preview"] is True
        assert task["input_schema"]["properties"]["guidance_scale"]["deprecated"] is True

    inpaint_defaults = catalog["tasks"]["qwen-image.inpaint"]["input_defaults"]
    assert inpaint_defaults["width"] == 1024
    assert inpaint_defaults["height"] == 1024
    assert inpaint_defaults["padding_mask_crop"] is None


def test_qwen_image_runtime_uses_model_settings_without_distilled_guidance():
    initial_image = Image.new("RGB", (64, 48), "blue")
    mask_image = Image.new("L", (64, 48), "white")
    cases = (
        (
            qwen_image_pipeline.generate_text2img_in_process,
            "load_text2img_pipeline",
            {"prompt": "test", "seed": 11, "guidance_scale": 9.0},
            {"width": 1328, "height": 1328},
        ),
        (
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {
                "prompt": "test",
                "initial_image": initial_image,
                "seed": 11,
                "guidance_scale": 9.0,
            },
            {"width": 1328, "height": 1328, "strength": 0.6},
        ),
        (
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "prompt": "test",
                "initial_image": initial_image,
                "mask_image": mask_image,
                "width": 768,
                "height": 1024,
                "padding_mask_crop": 32,
                "seed": 11,
                "guidance_scale": 9.0,
            },
            {
                "width": 768,
                "height": 1024,
                "padding_mask_crop": 32,
                "strength": 0.6,
            },
        ),
    )

    for generation_function, loader_name, params, expected_values in cases:
        result, fake_pipe, scheduler = _run_with_fake_pipeline(
            generation_function,
            loader_name,
            params,
        )

        assert result == {"images": ["/outputs/batch_settings/output.png"]}
        assert fake_pipe.scheduler is scheduler
        call_arguments = fake_pipe.call_arguments
        assert call_arguments is not None
        assert call_arguments["negative_prompt"] == EXPECTED_NEGATIVE_PROMPT
        assert call_arguments["num_inference_steps"] == 50
        assert call_arguments["true_cfg_scale"] == 4.0
        assert "guidance_scale" not in call_arguments
        for name, expected_value in expected_values.items():
            assert call_arguments[name] == expected_value


def test_qwen_image_runtime_preserves_an_explicit_empty_negative_prompt():
    assert qwen_image_pipeline._negative_prompt({"negative_prompt": ""}) == ""


def test_qwen_image_step_callback_writes_a_preview_on_every_intermediate_step(
    tmp_path,
):
    progress_path = tmp_path / "progress.json"
    preview_path = tmp_path / "preview.png"
    runtime = SubprocessRuntime(
        progress_path=progress_path,
        cancel_path=tmp_path / "cancel.requested",
    )
    pipe = SimpleNamespace(num_timesteps=8)
    callback = qwen_image_pipeline._build_step_callback(
        runtime,
        requested_steps=8,
        image_index=0,
        total_images=1,
        width=1328,
        height=1328,
        preview_path=preview_path,
        preview_url="/outputs/batch_test/preview.png",
    )
    callback_kwargs = {"latents": object()}

    with patch.object(
        qwen_image_pipeline,
        "_decode_preview_image",
        return_value=Image.new("RGB", (1024, 512), "blue"),
    ):
        returned = callback(pipe, 0, None, callback_kwargs)

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert returned is callback_kwargs
    assert progress == {
        "phase": "denoising",
        "image_number": 1,
        "total_images": 1,
        "step": 1,
        "total_steps": 8,
        "percent": 12.5,
        "preview_url": "/outputs/batch_test/preview.png",
    }
    with Image.open(preview_path) as preview:
        assert preview.size == (768, 384)


def test_qwen_image_step_callback_skips_preview_decode_when_disabled(tmp_path):
    progress_path = tmp_path / "progress.json"
    runtime = SubprocessRuntime(
        progress_path=progress_path,
        cancel_path=tmp_path / "cancel.requested",
    )
    call_kwargs = {}
    qwen_image_pipeline._install_step_callback(
        call_kwargs,
        runtime,
        requested_steps=8,
        image_index=0,
        total_images=1,
        width=1328,
        height=1328,
        preview_path=tmp_path / "preview.png",
        preview_url="/outputs/batch_test/preview.png",
        live_preview=False,
    )

    assert call_kwargs["callback_on_step_end_tensor_inputs"] == []
    with patch.object(qwen_image_pipeline, "_decode_preview_image") as decode:
        call_kwargs["callback_on_step_end"](
            SimpleNamespace(num_timesteps=8),
            0,
            None,
            {},
        )

    decode.assert_not_called()
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {
        "phase": "denoising",
        "image_number": 1,
        "total_images": 1,
        "step": 1,
        "total_steps": 8,
        "percent": 12.5,
    }


def test_qwen_image_step_callback_stops_on_cancel(tmp_path):
    cancel_path = tmp_path / "cancel.requested"
    cancel_path.touch()
    runtime = SubprocessRuntime(
        progress_path=tmp_path / "progress.json",
        cancel_path=cancel_path,
    )
    callback = qwen_image_pipeline._build_step_callback(
        runtime,
        requested_steps=8,
        image_index=0,
        total_images=1,
        width=1328,
        height=1328,
        preview_path=tmp_path / "preview.png",
        preview_url="/outputs/batch_test/preview.png",
    )

    with pytest.raises(SubprocessCanceled, match="Cancel requested"):
        callback(SimpleNamespace(num_timesteps=8), 0, None, {"latents": object()})


def test_qwen_image_pages_match_backend_defaults():
    for page_name in ("text2img", "img2img", "inpaint"):
        html = (ROOT / "frontend" / "qwen_image" / f"{page_name}.html").read_text(
            encoding="utf-8"
        )
        javascript = (
            ROOT / "frontend" / "qwen_image" / f"{page_name}.js"
        ).read_text(encoding="utf-8")

        assert EXPECTED_NEGATIVE_PROMPT in html
        assert EXPECTED_NEGATIVE_PROMPT in javascript
        assert 'id="steps" type="number" value="50"' in html
        assert 'id="live_preview" type="checkbox" checked' in html
        assert 'data-default-scheduler="flowmatch_euler"' in html
        assert 'data-allowed-schedulers="flowmatch_euler"' in html
        assert 'id="lora-panel-root"' in html
        assert '<link rel="stylesheet" href="../style.css?v=6" />' in html
        assert '<script src="../components/lora_panel.js?v=6"></script>' in html
        assert "Guidance Scale" not in html
        assert 'key: "guidance_scale"' not in javascript
        assert 'key: "scheduler", fallback: "flowmatch_euler"' in javascript
        assert 'key: "steps", type: "number", integer: true, fallback: 50' in javascript
        assert 'key: "live_preview", type: "checkbox", fallback: true' in javascript
        assert "loraEnvelope: false" in javascript
        assert "page.withLora" in javascript

    for page_name in ("text2img", "img2img"):
        html = (ROOT / "frontend" / "qwen_image" / f"{page_name}.html").read_text(
            encoding="utf-8"
        )
        assert 'id="width" type="number" value="1328"' in html
        assert 'id="height" type="number" value="1328"' in html

    inpaint_html = (ROOT / "frontend" / "qwen_image" / "inpaint.html").read_text(
        encoding="utf-8"
    )
    inpaint_javascript = (
        ROOT / "frontend" / "qwen_image" / "inpaint.js"
    ).read_text(encoding="utf-8")
    assert 'id="width" type="number" value="1024"' in inpaint_html
    assert 'id="height" type="number" value="1024"' in inpaint_html
    assert 'id="padding_mask_crop" type="number"' in inpaint_html
    assert 'key: "width", type: "number", integer: true, fallback: 1024' in inpaint_javascript
    assert 'key: "height", type: "number", integer: true, fallback: 1024' in inpaint_javascript
    assert 'key: "padding_mask_crop"' in inpaint_javascript

    scheduler_script = (
        ROOT / "frontend" / "components" / "scheduler_panel.js"
    ).read_text(encoding="utf-8")
    assert "container.dataset.defaultScheduler" in scheduler_script
    assert "container.dataset.allowedSchedulers" in scheduler_script
