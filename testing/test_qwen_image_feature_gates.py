from unittest.mock import patch

import pytest
from PIL import Image
from pydantic import ValidationError

from backend.qwen_image import pipeline as qwen_image_pipeline
from backend.workflow import (
    QwenImageImg2ImgInputs,
    QwenImageInpaintInputs,
    QwenImageText2ImgInputs,
    build_workflow_catalog,
)


_SCHEMA_CASES = (
    (QwenImageText2ImgInputs, {}),
    (
        QwenImageImg2ImgInputs,
        {"initial_image": "@artifact:initial", "prompt": "test"},
    ),
    (
        QwenImageInpaintInputs,
        {
            "initial_image": "@artifact:initial",
            "mask_image": "@artifact:mask",
            "prompt": "test",
        },
    ),
)


@pytest.mark.parametrize(("input_model", "required_inputs"), _SCHEMA_CASES)
def test_qwen_image_schema_locks_scheduler_to_flowmatch_euler(
    input_model,
    required_inputs,
):
    inputs = input_model(**required_inputs, scheduler="FLOWMATCH_EULER")

    assert inputs.scheduler == "flowmatch_euler"

    with pytest.raises(ValidationError, match="supports only scheduler 'flowmatch_euler'"):
        input_model(**required_inputs, scheduler="euler")


def test_qwen_image_schema_accepts_lora_adapters():
    assert QwenImageText2ImgInputs(lora_adapters=[]).lora_adapters == []
    inputs = QwenImageText2ImgInputs(
        lora_adapters=[{"lora_id": 101, "strength": 0.8}],
    )

    assert inputs.lora_adapters[0].model_dump() == {
        "lora_id": 101,
        "strength": 0.8,
        "target": "both",
    }


@pytest.mark.parametrize(
    "generation_function",
    (
        qwen_image_pipeline.generate_text2img,
        qwen_image_pipeline.generate_img2img,
        qwen_image_pipeline.generate_inpaint,
    ),
)
def test_qwen_image_public_runtime_rejects_scheduler_before_subprocess(
    generation_function,
):
    with patch.object(qwen_image_pipeline, "run_subprocess") as run_subprocess:
        with pytest.raises(ValueError, match="supports only scheduler"):
            generation_function({"prompt": "test", "scheduler": "euler"})

    run_subprocess.assert_not_called()


@pytest.mark.parametrize(
    ("generation_function", "operation"),
    (
        (qwen_image_pipeline.generate_text2img, "text2img"),
        (qwen_image_pipeline.generate_img2img, "img2img"),
        (qwen_image_pipeline.generate_inpaint, "inpaint"),
    ),
)
def test_qwen_image_public_runtime_forwards_lora_to_subprocess(
    generation_function,
    operation,
):
    params = {"prompt": "test", "lora_adapters": [{"lora_id": 101}]}
    with patch.object(
        qwen_image_pipeline,
        "run_subprocess",
        return_value={"images": ["/outputs/fake.png"]},
    ) as run_subprocess:
        result = generation_function(params)

    assert result == {"images": ["/outputs/fake.png"]}
    assert run_subprocess.call_args.args[1] == operation
    assert run_subprocess.call_args.args[2] is params


@pytest.mark.parametrize(
    ("generation_function", "loader_name", "params"),
    (
        (
            qwen_image_pipeline.generate_text2img_in_process,
            "load_text2img_pipeline",
            {"prompt": "test"},
        ),
        (
            qwen_image_pipeline.generate_img2img_in_process,
            "load_img2img_pipeline",
            {"prompt": "test", "initial_image": Image.new("RGB", (64, 64))},
        ),
        (
            qwen_image_pipeline.generate_inpaint_in_process,
            "load_inpaint_pipeline",
            {
                "prompt": "test",
                "initial_image": Image.new("RGB", (64, 64)),
                "mask_image": Image.new("L", (64, 64)),
            },
        ),
    ),
)
def test_qwen_image_in_process_rejects_invalid_lora_container_before_model_load(
    generation_function,
    loader_name,
    params,
):
    runtime_params = {**params, "lora_adapters": {"lora_id": 101}}
    with patch.object(qwen_image_pipeline, loader_name) as load_pipeline:
        with pytest.raises(ValueError, match="lora_adapters must be a list"):
            generation_function(runtime_params)

    load_pipeline.assert_not_called()


def test_qwen_image_catalog_marks_fixed_scheduler_and_enabled_lora():
    catalog = build_workflow_catalog()
    features = catalog["capabilities"]["qwen-image"]["features"]

    assert features["text2img"] is True
    assert features["img2img"] is True
    assert features["inpaint"] is True
    assert features["true_cfg_scale"] is True
    assert features["scheduler"] is False
    assert features["lora_adapters"] is True
    for task_type in (
        "qwen-image.text2img",
        "qwen-image.img2img",
        "qwen-image.inpaint",
    ):
        task = catalog["tasks"][task_type]
        assert task["ui_hints"]["inputs"]["scheduler"] == {
            "label": "Scheduler",
            "widget": "select",
            "options": ["flowmatch_euler"],
            "read_only": True,
        }
        lora_hint = task["ui_hints"]["inputs"]["lora_adapters"]
        assert lora_hint["widget"] == "json"
        assert lora_hint.get("supported", True) is True
        assert task["input_schema"]["properties"]["scheduler"]["const"] == "flowmatch_euler"
