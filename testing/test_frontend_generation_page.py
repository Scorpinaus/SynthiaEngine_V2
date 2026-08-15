from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_migrated_pages_load_shared_controller_before_page_script() -> None:
    for family in ("flux", "qwen_image", "z_image"):
        page_version = 8 if family == "qwen_image" else 3
        for name in ("text2img", "img2img"):
            html = (ROOT / "frontend" / family / f"{name}.html").read_text(encoding="utf-8")
            assert html.index("../generation_page.js") < html.index(
                f'{name}.js?v={page_version}'
            )

    for family in ("anima", "ernie_image"):
        html = (ROOT / "frontend" / family / "text2img.html").read_text(encoding="utf-8")
        assert html.index("../generation_page.js") < html.index('text2img.js?v=3')


def test_optional_generation_features_remain_explicit() -> None:
    anima = (ROOT / "frontend" / "anima" / "text2img.js").read_text(encoding="utf-8")
    ernie = (ROOT / "frontend" / "ernie_image" / "text2img.js").read_text(encoding="utf-8")
    controller = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")

    assert 'lora: false' in anima
    assert 'loraEnvelope: false' in ernie
    assert 'type: "checkbox"' in ernie
    assert 'config.lora !== false' in controller


def test_flux_page_configs_keep_task_contracts_visible() -> None:
    text2img = (ROOT / "frontend" / "flux" / "text2img.js").read_text(encoding="utf-8")
    img2img = (ROOT / "frontend" / "flux" / "img2img.js").read_text(encoding="utf-8")

    assert 'taskType: "flux.text2img"' in text2img
    assert 'taskType: "flux.img2img"' in img2img
    assert 'inputs.initial_image = `@artifact:${artifact.artifact_id}`' in img2img
    assert 'return: "@t1.images"' in (
        ROOT / "frontend" / "generation_page.js"
    ).read_text(encoding="utf-8")


def test_qwen_pages_show_step_progress_previews_and_cancel_control() -> None:
    controller = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
    assert 'progress?.phase === "denoising"' in controller
    assert "progress.preview_url" in controller
    assert "/api/jobs/${activeJobId}/cancel" in controller

    for page_name in ("text2img", "img2img", "inpaint"):
        html = (ROOT / "frontend" / "qwen_image" / f"{page_name}.html").read_text(
            encoding="utf-8"
        )
        assert 'id="generation-progress"' in html
        assert 'id="generation-progress-bar"' in html
        assert 'id="generation-cancel"' in html
        assert '../generation_page.js?v=3' in html


def test_qwen_pages_load_shared_lightning_settings_before_page_scripts() -> None:
    for page_name, task_type, page_version in (
        ("text2img", "qwen-image.text2img", 8),
        ("img2img", "qwen-image.img2img", 8),
        ("inpaint", "qwen-image.inpaint", 9),
    ):
        html = (ROOT / "frontend" / "qwen_image" / f"{page_name}.html").read_text(
            encoding="utf-8"
        )
        script = (ROOT / "frontend" / "qwen_image" / f"{page_name}.js").read_text(
            encoding="utf-8"
        )
        assert 'src="lightning_settings.js?v=2"' in html
        assert html.index("lightning_settings.js") < html.index("../generation_page.js")
        assert html.index("../generation_page.js") < html.index(
            f'{page_name}.js?v={page_version}'
        )
        assert f'taskType: "{task_type}"' in script
        assert "QwenImageLightningSettings.create" in script
        assert "settingsHooks: qwenLightningSettings" in script


def test_simple_inpaint_pages_share_editor_but_keep_task_fields_visible() -> None:
    editor = (ROOT / "frontend" / "components" / "inpaint_editor.js").read_text(encoding="utf-8")
    assert "getBaseImageFile" in editor
    assert "getActiveMaskBlob" in editor
    assert "/create-blur-mask" in editor

    for family, task_type in (
        ("flux", "flux.inpaint"),
        ("qwen_image", "qwen-image.inpaint"),
        ("z_image", "z-image.inpaint"),
    ):
        script = (ROOT / "frontend" / family / "inpaint.js").read_text(encoding="utf-8")
        assert f'taskType: "{task_type}"' in script
        assert "inputs.initial_image" in script
        assert "inputs.mask_image" in script
