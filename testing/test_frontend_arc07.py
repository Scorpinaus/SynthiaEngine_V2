from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


PAGE_CONTRACTS = {
    ("sd15", "text2img"): (
        "text2img.html",
        "text2img.js?v=7",
        ("sd15.text2img", "sd15.controlnet.text2img", "sd15.hires_fix"),
    ),
    ("sd15", "img2img"): ("img2img.html", "img2img.js?v=7", ("sd15.img2img",)),
    ("sd15", "inpainting"): ("inpainting.html", "inpainting.js?v=7", ("sd15.inpaint",)),
    ("sd15", "animatediff"): (
        "animatediff.html",
        "animatediff.js?v=4",
        ("sd15.animatediff.text2video",),
    ),
    ("sdxl", "text2img"): (
        "text2img.html",
        "text2img.js?v=7",
        ("sdxl.text2img", "sdxl.controlnet.text2img"),
    ),
    ("sdxl", "img2img"): ("img2img.html", "img2img.js?v=5", ("sdxl.img2img",)),
    ("sdxl", "inpaint"): ("inpaint.html", "inpaint.js?v=5", ("sdxl.inpaint",)),
}


def test_arc07_entrypoints_are_small_and_keep_task_names_visible() -> None:
    for (family, script_name), (_html, _tag, task_names) in PAGE_CONTRACTS.items():
        script = FRONTEND / family / f"{script_name}.js"
        source = script.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 125
        for task_name in task_names:
            assert task_name in source
        assert "submitWorkflow" not in source
        assert "watchJob" not in source
        assert "/models?family=" not in source


def test_arc07_pages_load_composition_before_entrypoint() -> None:
    shared = (
        "../generation_page.js?v=4",
        "../components/adapter_controller.js?v=1",
        "../components/controlnet_controller.js?v=1",
        "../components/ip_adapter_controller.js?v=1",
    )
    for (family, script_name), (html_name, entry_tag, _tasks) in PAGE_CONTRACTS.items():
        html = (FRONTEND / family / html_name).read_text(encoding="utf-8")
        if script_name == "animatediff":
            assert html.index("../generation_page.js?v=4") < html.index(entry_tag)
            assert html.index("../components/animatediff_controller.js?v=1") < html.index(entry_tag)
            continue
        for script in shared:
            assert html.index(script) < html.index(entry_tag)
        assert html.index("generation_controller.js?v=1") < html.index(entry_tag)
        if "inpaint" in script_name:
            assert html.index("../components/inpaint_editor.js?v=1") < html.index(entry_tag)


def test_arc07_feature_controllers_have_explicit_contract_methods() -> None:
    controlnet = (FRONTEND / "components" / "controlnet_controller.js").read_text(
        encoding="utf-8"
    )
    ip_adapter = (FRONTEND / "components" / "ip_adapter_controller.js").read_text(
        encoding="utf-8"
    )
    inpaint = (FRONTEND / "components" / "inpaint_editor.js").read_text(encoding="utf-8")
    animatediff = (FRONTEND / "components" / "animatediff_controller.js").read_text(
        encoding="utf-8"
    )

    for method in ("attachSd15Text", "attachSd15Image", "attachSdxlText", "attachSdxlImage"):
        assert method in controlnet
    assert "attachEncoded" in ip_adapter
    assert "attachDirect" in ip_adapter
    assert "getBaseImageFile" in inpaint
    assert "getActiveMaskBlob" in inpaint
    assert 'return: "@t1.videos"' in animatediff


def test_arc07_family_controllers_keep_payload_shapes_and_guardrails() -> None:
    sd15 = (FRONTEND / "sd15" / "generation_controller.js").read_text(encoding="utf-8")
    sdxl = (FRONTEND / "sdxl" / "generation_controller.js").read_text(encoding="utf-8")

    for contract in (
        'returnRef = "@t1.images"',
        'returnRef = "@hires.images"',
        'return: "@img2img.images"',
        'return: "@inpaint.images"',
        "inputs.initial_image",
        "inputs.mask_image",
        "lora_enabled",
        "IP-Adapter cannot be combined with Hi-Res Fix yet.",
    ):
        assert contract in sd15

    for contract in (
        'return: "@t1.images"',
        "inputs.initial_image",
        "inputs.mask_image",
        "inputs.Lora",
        "SDXL IP-Adapter cannot be combined with ControlNet yet.",
        "SDXL img2img IP-Adapter cannot be combined with ControlNet yet.",
        "SDXL inpaint IP-Adapter cannot be combined with ControlNet yet.",
    ):
        assert contract in sdxl
