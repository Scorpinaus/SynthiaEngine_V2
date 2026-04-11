from backend.workflow import build_workflow_catalog


def test_workflow_catalog_exposes_model_capabilities_matrix():
    catalog = build_workflow_catalog()
    capabilities = catalog.get("capabilities", {})

    assert "sd15" in capabilities
    assert "sdxl" in capabilities
    assert "flux" in capabilities
    assert "qwen-image" in capabilities
    assert "z-image" in capabilities


def test_workflow_capability_features_for_core_families():
    catalog = build_workflow_catalog()
    capabilities = catalog["capabilities"]

    sd15 = capabilities["sd15"]["features"]
    assert sd15["text2img"] is True
    assert sd15["text2video"] is True
    assert sd15["img2img"] is True
    assert sd15["inpaint"] is True
    assert sd15["controlnet"] is True
    assert sd15["hires_fix"] is True
    assert sd15["lora_adapters"] is True

    sdxl = capabilities["sdxl"]["features"]
    assert sdxl["text2video"] is False
    assert sdxl["controlnet"] is True
    assert sdxl["hires_fix"] is False
    assert sdxl["lora_adapters"] is True

    flux = capabilities["flux"]["features"]
    assert flux["text2img"] is True
    assert flux["text2video"] is False
    assert flux["img2img"] is True
    assert flux["inpaint"] is True
    assert flux["controlnet"] is False
    assert flux["lora_adapters"] is True

    qwen = capabilities["qwen-image"]["features"]
    assert qwen["text2video"] is False
    assert qwen["true_cfg_scale"] is True
    assert qwen["inpaint"] is True
    assert qwen["lora_adapters"] is True

    zimage = capabilities["z-image"]["features"]
    assert zimage["text2img"] is True
    assert zimage["text2video"] is False
    assert zimage["img2img"] is True
    assert zimage["inpaint"] is True
    assert zimage["lora_adapters"] is True
