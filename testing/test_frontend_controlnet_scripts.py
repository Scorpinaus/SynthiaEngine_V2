from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendControlNetScriptTests(unittest.TestCase):
    def test_sd15_page_includes_controlnet_scripts_before_sd15(self):
        sd15_html = (ROOT / "frontend" / "sd15.html").read_text(encoding="utf-8")
        validator_tag = '<script src="workflow_input_validator.js?v=1"></script>'
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        sd15_tag = '<script src="sd15.js?v=5"></script>'

        self.assertIn(validator_tag, sd15_html)
        self.assertIn(panel_tag, sd15_html)
        self.assertIn(preprocessor_tag, sd15_html)
        self.assertIn(preset_tag, sd15_html)
        self.assertIn(sd15_tag, sd15_html)
        self.assertLess(sd15_html.index(validator_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(panel_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(preprocessor_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(preset_tag), sd15_html.index(sd15_tag))

    def test_sd15_img2img_page_includes_controlnet_scripts_before_img2img(self):
        sd15_img2img_html = (ROOT / "frontend" / "sd15_img2img.html").read_text(encoding="utf-8")
        validator_tag = '<script src="workflow_input_validator.js?v=1"></script>'
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        img2img_tag = '<script src="sd15_img2img.js?v=5"></script>'

        self.assertIn(validator_tag, sd15_img2img_html)
        self.assertIn(panel_tag, sd15_img2img_html)
        self.assertIn(preprocessor_tag, sd15_img2img_html)
        self.assertIn(preset_tag, sd15_img2img_html)
        self.assertIn(img2img_tag, sd15_img2img_html)
        self.assertLess(sd15_img2img_html.index(validator_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(sd15_img2img_html.index(panel_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(
            sd15_img2img_html.index(preprocessor_tag), sd15_img2img_html.index(img2img_tag)
        )
        self.assertLess(sd15_img2img_html.index(preset_tag), sd15_img2img_html.index(img2img_tag))

    def test_sd15_inpaint_page_includes_controlnet_scripts_before_inpaint(self):
        sd15_inpaint_html = (ROOT / "frontend" / "sd15_inpainting.html").read_text(encoding="utf-8")
        validator_tag = '<script src="workflow_input_validator.js?v=1"></script>'
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        inpaint_tag = '<script src="sd15_inpainting.js?v=5"></script>'

        self.assertIn(validator_tag, sd15_inpaint_html)
        self.assertIn(panel_tag, sd15_inpaint_html)
        self.assertIn(preprocessor_tag, sd15_inpaint_html)
        self.assertIn(preset_tag, sd15_inpaint_html)
        self.assertIn(inpaint_tag, sd15_inpaint_html)
        self.assertLess(sd15_inpaint_html.index(validator_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(sd15_inpaint_html.index(panel_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(
            sd15_inpaint_html.index(preprocessor_tag), sd15_inpaint_html.index(inpaint_tag)
        )
        self.assertLess(sd15_inpaint_html.index(preset_tag), sd15_inpaint_html.index(inpaint_tag))

    def test_sdxl_page_includes_controlnet_scripts_before_sdxl(self):
        sdxl_html = (ROOT / "frontend" / "sdxl.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        sdxl_tag = '<script src="sdxl.js?v=6"></script>'

        self.assertIn(panel_tag, sdxl_html)
        self.assertIn(preprocessor_tag, sdxl_html)
        self.assertIn(lora_tag, sdxl_html)
        self.assertIn(preset_tag, sdxl_html)
        self.assertIn(sdxl_tag, sdxl_html)
        self.assertLess(sdxl_html.index(panel_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(preprocessor_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(lora_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(preset_tag), sdxl_html.index(sdxl_tag))

    def test_sdxl_img2img_page_includes_controlnet_scripts_before_sdxl_img2img(self):
        sdxl_img2img_html = (ROOT / "frontend" / "sdxl_img2img.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        sdxl_img2img_tag = '<script src="sdxl_img2img.js?v=4"></script>'

        self.assertIn(panel_tag, sdxl_img2img_html)
        self.assertIn(preprocessor_tag, sdxl_img2img_html)
        self.assertIn(lora_tag, sdxl_img2img_html)
        self.assertIn(preset_tag, sdxl_img2img_html)
        self.assertIn(sdxl_img2img_tag, sdxl_img2img_html)
        self.assertLess(sdxl_img2img_html.index(panel_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(preprocessor_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(lora_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(preset_tag), sdxl_img2img_html.index(sdxl_img2img_tag))

    def test_sdxl_inpaint_page_includes_controlnet_scripts_before_sdxl_inpaint(self):
        sdxl_inpaint_html = (ROOT / "frontend" / "sdxl_inpaint.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        sdxl_inpaint_tag = '<script src="sdxl_inpaint.js?v=4"></script>'

        self.assertIn(panel_tag, sdxl_inpaint_html)
        self.assertIn(preprocessor_tag, sdxl_inpaint_html)
        self.assertIn(lora_tag, sdxl_inpaint_html)
        self.assertIn(preset_tag, sdxl_inpaint_html)
        self.assertIn(sdxl_inpaint_tag, sdxl_inpaint_html)
        self.assertLess(sdxl_inpaint_html.index(panel_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(preprocessor_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(lora_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(preset_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))

    def test_controlnet_panel_script_exposes_expected_api(self):
        panel_js = (ROOT / "frontend" / "controlnet_panel.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel", panel_js)
        self.assertIn("getState", panel_js)
        self.assertIn("loadPanel", panel_js)
        self.assertIn("updateIndicator", panel_js)
        self.assertIn('fetch("controlnet_panel.html?v=2", { cache: "no-store" })', panel_js)

    def test_controlnet_preprocessor_script_exposes_expected_api(self):
        preprocessor_js = (ROOT / "frontend" / "controlnet_preprocessor.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPreprocessor", preprocessor_js)
        self.assertIn("ensureControlNetUI", preprocessor_js)
        self.assertIn("openPreprocessorModal", preprocessor_js)
        self.assertIn('fetch("controlnet_preprocessor.html?v=2", { cache: "no-store" })', preprocessor_js)
        self.assertIn("ensurePreprocessorLayoutStructure", preprocessor_js)
        self.assertIn("gridTemplateColumns", preprocessor_js)
        self.assertIn("window.innerWidth <= 700", preprocessor_js)

    def test_sd15_controlnet_script_wires_per_item_guidance_timing(self):
        panel_js = (ROOT / "frontend" / "controlnet_panel.js").read_text(encoding="utf-8")
        preprocessor_js = (ROOT / "frontend" / "controlnet_preprocessor.js").read_text(
            encoding="utf-8"
        )
        sd15_js = (ROOT / "frontend" / "sd15.js").read_text(encoding="utf-8")

        self.assertIn("data-guidance-start-id", panel_js)
        self.assertIn("data-guidance-end-id", panel_js)
        self.assertIn("guidanceStart: Number(guidanceStart ?? 0.0)", panel_js)
        self.assertIn("guidanceEnd: Number(guidanceEnd ?? 1.0)", panel_js)
        self.assertIn("setPerItemGuidanceTimingEnabled", panel_js)
        self.assertIn("defaultGuidanceStart", preprocessor_js)
        self.assertIn("defaultGuidanceEnd", preprocessor_js)
        self.assertIn("controlGuidanceStarts", sd15_js)
        self.assertIn("controlGuidanceEnds", sd15_js)
        self.assertIn("guidance_start: controlGuidanceStarts[idx]", sd15_js)
        self.assertIn("guidance_end: controlGuidanceEnds[idx]", sd15_js)
        self.assertIn("inputs.control_guidance_starts = controlGuidanceStarts", sd15_js)
        self.assertIn("inputs.control_guidance_ends = controlGuidanceEnds", sd15_js)
        self.assertIn("setPerItemGuidanceTimingEnabled?.(true)", sd15_js)

    def test_workflow_input_validator_script_exposes_expected_api(self):
        validator_js = (ROOT / "frontend" / "workflow_input_validator.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.WorkflowInputValidator", validator_js)
        self.assertIn("validateTaskInputs", validator_js)
        self.assertIn("assertTaskInputs", validator_js)
        self.assertIn("WorkflowCatalog.load", validator_js)

    def test_lora_panel_html_has_weight_mode_toggle(self):
        lora_html = (ROOT / "frontend" / "lora_panel.html").read_text(encoding="utf-8")
        self.assertIn('id="lora-weight-mode-row"', lora_html)
        self.assertIn('id="lora-weight-mode-basic"', lora_html)
        self.assertIn('id="lora-weight-mode-advanced"', lora_html)

    def test_lora_panel_script_supports_sd15_advanced_component_strengths(self):
        lora_js = (ROOT / "frontend" / "lora_panel.js").read_text(encoding="utf-8")
        self.assertIn("weightMode", lora_js)
        self.assertIn("lora-weight-mode-advanced", lora_js)
        self.assertIn("unet_strength", lora_js)
        self.assertIn("text_encoder_strength", lora_js)

    def test_preset_panel_html_has_mode_specific_controls(self):
        preset_html = (ROOT / "frontend" / "preset_panel.html").read_text(encoding="utf-8")
        self.assertIn('id="preset-load"', preset_html)
        self.assertIn('id="preset-refresh"', preset_html)
        self.assertIn('id="preset-add-new"', preset_html)
        self.assertIn('id="preset-name-field"', preset_html)
        self.assertIn('id="preset-create-actions"', preset_html)
        self.assertIn('id="preset-manage-actions"', preset_html)
        self.assertIn('id="preset-cancel"', preset_html)

    def test_preset_panel_script_supports_create_and_manage_modes(self):
        preset_js = (ROOT / "frontend" / "preset_panel.js").read_text(encoding="utf-8")
        self.assertIn("const UI_MODES", preset_js)
        self.assertIn("setUiMode(UI_MODES.MANAGE)", preset_js)
        self.assertIn('document.getElementById("preset-add-new")', preset_js)
        self.assertIn('document.getElementById("preset-cancel")', preset_js)

    def test_sd15_img2img_script_consumes_controlnet_state(self):
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", img2img_js)
        self.assertIn("controlnetEnabled", img2img_js)
        self.assertIn("control_images", img2img_js)
        self.assertIn("controlnet_models", img2img_js)

    def test_sd15_script_wires_preset_panel(self):
        sd15_js = (ROOT / "frontend" / "sd15.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sd15_js)
        self.assertIn('taskType: "sd15.text2img"', sd15_js)
        self.assertIn("collectSettings: collectSd15PresetSettings", sd15_js)
        self.assertIn("applySettings: applySd15PresetSettings", sd15_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", sd15_js)

    def test_sd15_script_wires_lcm_mode_payload_and_guardrails(self):
        sd15_html = (ROOT / "frontend" / "sd15.html").read_text(encoding="utf-8")
        sd15_js = (ROOT / "frontend" / "sd15.js").read_text(encoding="utf-8")
        scheduler_html = (ROOT / "frontend" / "scheduler_panel.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="lcm_enabled"', sd15_html)
        self.assertIn('<option value="lcm">LCM (SD1.5)</option>', scheduler_html)
        self.assertIn('inputs.lcm = { enabled: true };', sd15_js)
        self.assertIn('inputs.scheduler = DEFAULTS.lcm_scheduler;', sd15_js)
        self.assertIn("LCM mode is currently available for SD1.5 text-to-image only.", sd15_js)
        self.assertIn("lora_enabled: loraAdaptersEnabled", sd15_js)
        self.assertNotIn("cannot combine with selected LoRAs", sd15_js)

    def test_sd15_page_wires_ip_adapter_controls(self):
        sd15_html = (ROOT / "frontend" / "sd15.html").read_text(encoding="utf-8")

        self.assertIn('id="ip_adapter_panel"', sd15_html)
        self.assertIn('id="ip_adapter_toggle"', sd15_html)
        self.assertIn('id="ip_adapter_content"', sd15_html)
        self.assertIn('id="ip_adapter_enabled"', sd15_html)
        self.assertIn('id="ip_adapter_image"', sd15_html)
        self.assertIn('id="ip_adapter_preview"', sd15_html)
        self.assertIn('id="ip_adapter_mask_image"', sd15_html)
        self.assertIn('id="ip_adapter_mask_editor_open"', sd15_html)
        self.assertIn('id="ip_adapter_mask_preview"', sd15_html)
        self.assertIn('id="ip_adapter_scale"', sd15_html)
        self.assertIn("ip_adapter_panel.js?v=1", sd15_html)

    def test_sd15_script_wires_ip_adapter_payload_and_guardrails(self):
        sd15_js = (ROOT / "frontend" / "sd15.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init({", sd15_js)
        self.assertIn("getIpAdapterImageFile", sd15_js)
        self.assertIn("WorkflowClient.uploadArtifact(", sd15_js)
        self.assertIn("window.IpAdapterPanel?.getMaskFile?.()", sd15_js)
        self.assertIn("inputs.ip_adapter = {", sd15_js)
        self.assertIn("inputs.ip_adapter.mask_image", sd15_js)
        self.assertIn('type: "sd15.ip_adapter.encode"', sd15_js)
        self.assertIn('id: "ip_embeds"', sd15_js)
        self.assertIn('image_embeds: "@ip_embeds.image_embeds"', sd15_js)
        self.assertIn('ip_adapter_subfolder: "models"', sd15_js)
        self.assertIn('ip_adapter_weight_name: "ip-adapter_sd15.bin"', sd15_js)
        self.assertIn('model: "h94/IP-Adapter"', sd15_js)
        self.assertIn('weight_name: "ip-adapter_sd15.bin"', sd15_js)
        self.assertIn("IP-Adapter cannot be combined with LCM mode yet.", sd15_js)
        self.assertIn("IP-Adapter cannot be combined with Hi-Res Fix yet.", sd15_js)
        self.assertIn("IP-Adapter is currently available for SD1.5 text-to-image only.", sd15_js)

    def test_sd15_img2img_script_wires_preset_panel(self):
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", img2img_js)
        self.assertIn('taskType: "sd15.img2img"', img2img_js)
        self.assertIn("collectSettings: collectSd15Img2ImgPresetSettings", img2img_js)
        self.assertIn("applySettings: applySd15Img2ImgPresetSettings", img2img_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", img2img_js)

    def test_sd15_img2img_script_wires_lcm_mode_payload_and_guardrails(self):
        img2img_html = (ROOT / "frontend" / "sd15_img2img.html").read_text(encoding="utf-8")
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")

        self.assertIn('id="lcm_enabled"', img2img_html)
        self.assertIn("function applyLcmImg2ImgContract(inputs)", img2img_js)
        self.assertIn("inputs.lcm = { enabled: true };", img2img_js)
        self.assertIn('inputs.scheduler = DEFAULTS.lcm_scheduler;', img2img_js)
        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 img2img yet.", img2img_js)

    def test_sd15_img2img_page_wires_ip_adapter_controls(self):
        img2img_html = (ROOT / "frontend" / "sd15_img2img.html").read_text(encoding="utf-8")

        self.assertIn('id="ip_adapter_panel"', img2img_html)
        self.assertIn('id="ip_adapter_toggle"', img2img_html)
        self.assertIn('id="ip_adapter_content"', img2img_html)
        self.assertIn('id="ip_adapter_enabled"', img2img_html)
        self.assertIn('id="ip_adapter_image"', img2img_html)
        self.assertIn('id="ip_adapter_preview"', img2img_html)
        self.assertIn('id="ip_adapter_mask_image"', img2img_html)
        self.assertIn('id="ip_adapter_mask_editor_open"', img2img_html)
        self.assertIn('id="ip_adapter_mask_preview"', img2img_html)
        self.assertIn('id="ip_adapter_scale"', img2img_html)
        self.assertIn("ip_adapter_panel.js?v=1", img2img_html)

    def test_sd15_img2img_script_wires_ip_adapter_payload_and_guardrails(self):
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init({", img2img_js)
        self.assertIn("getIpAdapterImageFile", img2img_js)
        self.assertIn("WorkflowClient.uploadArtifact(", img2img_js)
        self.assertIn("window.IpAdapterPanel?.getMaskFile?.()", img2img_js)
        self.assertIn("taskInputs.ip_adapter = {", img2img_js)
        self.assertIn("taskInputs.ip_adapter.mask_image", img2img_js)
        self.assertIn('type: "sd15.ip_adapter.encode"', img2img_js)
        self.assertIn('id: "ip_embeds"', img2img_js)
        self.assertIn('image_embeds: "@ip_embeds.image_embeds"', img2img_js)
        self.assertIn('ip_adapter_subfolder: "models"', img2img_js)
        self.assertIn('ip_adapter_weight_name: "ip-adapter_sd15.bin"', img2img_js)
        self.assertIn('model: "h94/IP-Adapter"', img2img_js)
        self.assertIn('weight_name: "ip-adapter_sd15.bin"', img2img_js)
        self.assertIn("IP-Adapter cannot be combined with ControlNet for SD1.5 img2img yet.", img2img_js)
        self.assertIn("IP-Adapter cannot be combined with LCM mode for SD1.5 img2img yet.", img2img_js)

    def test_sd15_inpaint_script_wires_preset_panel(self):
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", inpaint_js)
        self.assertIn('taskType: "sd15.inpaint"', inpaint_js)
        self.assertIn("collectSettings: collectSd15InpaintPresetSettings", inpaint_js)
        self.assertIn("applySettings: applySd15InpaintPresetSettings", inpaint_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", inpaint_js)

    def test_sd15_inpaint_script_wires_lcm_mode_payload_and_guardrails(self):
        inpaint_html = (ROOT / "frontend" / "sd15_inpainting.html").read_text(
            encoding="utf-8"
        )
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")

        self.assertIn('id="lcm_enabled"', inpaint_html)
        self.assertIn("function applyLcmInpaintContract(inputs)", inpaint_js)
        self.assertIn("inputs.lcm = { enabled: true };", inpaint_js)
        self.assertIn("inputs.scheduler = DEFAULTS.lcm_scheduler;", inpaint_js)
        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 inpaint yet.", inpaint_js)

    def test_sd15_inpaint_page_wires_ip_adapter_controls(self):
        inpaint_html = (ROOT / "frontend" / "sd15_inpainting.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="ip_adapter_panel"', inpaint_html)
        self.assertIn('id="ip_adapter_toggle"', inpaint_html)
        self.assertIn('id="ip_adapter_content"', inpaint_html)
        self.assertIn('id="ip_adapter_enabled"', inpaint_html)
        self.assertIn('id="ip_adapter_image"', inpaint_html)
        self.assertIn('id="ip_adapter_preview"', inpaint_html)
        self.assertIn('id="ip_adapter_mask_image"', inpaint_html)
        self.assertIn('id="ip_adapter_mask_editor_open"', inpaint_html)
        self.assertIn('id="ip_adapter_mask_preview"', inpaint_html)
        self.assertIn('id="ip_adapter_scale"', inpaint_html)
        self.assertIn("ip_adapter_panel.js?v=1", inpaint_html)

    def test_sd15_inpaint_script_wires_ip_adapter_payload_and_guardrails(self):
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init({", inpaint_js)
        self.assertIn("getIpAdapterImageFile", inpaint_js)
        self.assertIn("WorkflowClient.uploadArtifact(", inpaint_js)
        self.assertIn("window.IpAdapterPanel?.getMaskFile?.()", inpaint_js)
        self.assertIn("taskInputs.ip_adapter = {", inpaint_js)
        self.assertIn("taskInputs.ip_adapter.mask_image", inpaint_js)
        self.assertIn('type: "sd15.ip_adapter.encode"', inpaint_js)
        self.assertIn('id: "ip_embeds"', inpaint_js)
        self.assertIn('image_embeds: "@ip_embeds.image_embeds"', inpaint_js)
        self.assertIn('ip_adapter_subfolder: "models"', inpaint_js)
        self.assertIn('ip_adapter_weight_name: "ip-adapter_sd15.bin"', inpaint_js)
        self.assertIn('model: "h94/IP-Adapter"', inpaint_js)
        self.assertIn('weight_name: "ip-adapter_sd15.bin"', inpaint_js)
        self.assertIn("IP-Adapter cannot be combined with ControlNet for SD1.5 inpaint yet.", inpaint_js)
        self.assertIn("IP-Adapter cannot be combined with LCM mode for SD1.5 inpaint yet.", inpaint_js)

    def test_ip_adapter_panel_script_previews_selected_reference_image(self):
        panel_js = (ROOT / "frontend" / "ip_adapter_panel.js").read_text(encoding="utf-8")

        self.assertIn("URL.createObjectURL(file)", panel_js)
        self.assertIn("URL.revokeObjectURL(previewUrl)", panel_js)
        self.assertIn("URL.revokeObjectURL(maskPreviewUrl)", panel_js)
        self.assertIn('document.getElementById("ip_adapter_image")', panel_js)
        self.assertIn('document.getElementById("ip_adapter_preview")', panel_js)
        self.assertIn('document.getElementById("ip_adapter_mask_image")', panel_js)
        self.assertIn("window.IpAdapterPanel = { init, getMaskFile }", panel_js)
        self.assertIn('content.classList.toggle("is-open", isOpen)', panel_js)

    def test_sd15_img2img_script_wires_lora_panel_and_payload(self):
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" })', img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", img2img_js)
        self.assertIn("inputs.lora = {", img2img_js)
        self.assertIn("setLoraContract(taskInputs, loraAdapters);", img2img_js)

    def test_sd15_inpaint_script_consumes_controlnet_state(self):
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", inpaint_js)
        self.assertIn("controlnetEnabled", inpaint_js)
        self.assertIn("control_images", inpaint_js)
        self.assertIn("controlnet_models", inpaint_js)

    def test_sd15_inpaint_script_wires_lora_panel_and_payload(self):
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" })', inpaint_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", inpaint_js)
        self.assertIn("inputs.lora = {", inpaint_js)
        self.assertIn("setLoraContract(taskInputs, loraAdapters);", inpaint_js)

    def test_sdxl_script_consumes_controlnet_state(self):
        sdxl_js = (ROOT / "frontend" / "sdxl.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_js)
        self.assertIn("controlnetEnabled", sdxl_js)
        self.assertIn("control_images", sdxl_js)
        self.assertIn("controlnet_models", sdxl_js)

    def test_sdxl_script_wires_lora_panel_and_payload(self):
        sdxl_js = (ROOT / "frontend" / "sdxl.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_js)
        self.assertIn("inputs.lora_adapters = loraAdapters;", sdxl_js)
        self.assertIn("payload.lora_adapters = loraAdapters;", sdxl_js)

    def test_sdxl_page_wires_ip_adapter_controls(self):
        sdxl_html = (ROOT / "frontend" / "sdxl.html").read_text(encoding="utf-8")

        self.assertIn('id="ip_adapter_panel"', sdxl_html)
        self.assertIn('id="ip_adapter_toggle"', sdxl_html)
        self.assertIn('id="ip_adapter_content"', sdxl_html)
        self.assertIn('id="ip_adapter_enabled"', sdxl_html)
        self.assertIn('id="ip_adapter_image"', sdxl_html)
        self.assertIn('id="ip_adapter_preview"', sdxl_html)
        self.assertIn('id="ip_adapter_scale"', sdxl_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_html)

    def test_sdxl_script_wires_ip_adapter_payload_and_guardrails(self):
        sdxl_js = (ROOT / "frontend" / "sdxl.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init()", sdxl_js)
        self.assertIn("getIpAdapterImageFile", sdxl_js)
        self.assertIn("WorkflowClient.uploadArtifact(", sdxl_js)
        self.assertIn("payload.ip_adapter = {", sdxl_js)
        self.assertIn('type: "sdxl.ip_adapter.encode"', sdxl_js)
        self.assertIn('id: "ip_embeds"', sdxl_js)
        self.assertIn('image_embeds: "@ip_embeds.image_embeds"', sdxl_js)
        self.assertIn("ip_adapter_model: \"h94/IP-Adapter\"", sdxl_js)
        self.assertIn('ip_adapter_subfolder: "sdxl_models"', sdxl_js)
        self.assertIn('ip_adapter_weight_name: "ip-adapter_sdxl.bin"', sdxl_js)
        self.assertIn('model: "h94/IP-Adapter"', sdxl_js)
        self.assertIn('subfolder: "sdxl_models"', sdxl_js)
        self.assertIn('weight_name: "ip-adapter_sdxl.bin"', sdxl_js)
        self.assertIn("SDXL IP-Adapter cannot be combined with ControlNet yet.", sdxl_js)

    def test_sdxl_script_wires_preset_panel(self):
        sdxl_js = (ROOT / "frontend" / "sdxl.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_js)
        self.assertIn('taskType: "sdxl.text2img"', sdxl_js)
        self.assertIn("collectSettings: collectSdxlPresetSettings", sdxl_js)
        self.assertIn("applySettings: applySdxlPresetSettings", sdxl_js)

    def test_sdxl_img2img_script_consumes_controlnet_state(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_img2img_js)
        self.assertIn("controlnetEnabled", sdxl_img2img_js)
        self.assertIn("control_images", sdxl_img2img_js)
        self.assertIn("controlnet_models", sdxl_img2img_js)

    def test_sdxl_img2img_script_wires_lora_panel_and_payload(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl_img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_img2img_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", sdxl_img2img_js)

    def test_sdxl_img2img_page_and_script_wire_ip_adapter_payload(self):
        sdxl_img2img_html = (ROOT / "frontend" / "sdxl_img2img.html").read_text(encoding="utf-8")
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl_img2img.js").read_text(encoding="utf-8")

        self.assertIn('id="ip_adapter_panel"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_toggle"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_content"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_enabled"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_image"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_preview"', sdxl_img2img_html)
        self.assertIn('id="ip_adapter_scale"', sdxl_img2img_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_img2img_html)
        self.assertIn("window.IpAdapterPanel?.init()", sdxl_img2img_js)
        self.assertIn("getIpAdapterImageFile", sdxl_img2img_js)
        self.assertIn("taskInputs.ip_adapter = {", sdxl_img2img_js)
        self.assertIn('subfolder: "sdxl_models"', sdxl_img2img_js)
        self.assertIn('weight_name: "ip-adapter_sdxl.bin"', sdxl_img2img_js)
        self.assertIn("SDXL img2img IP-Adapter cannot be combined with ControlNet yet.", sdxl_img2img_js)

    def test_sdxl_img2img_script_wires_preset_panel(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_img2img_js)
        self.assertIn('taskType: "sdxl.img2img"', sdxl_img2img_js)
        self.assertIn("collectSettings: collectSdxlImg2ImgPresetSettings", sdxl_img2img_js)
        self.assertIn("applySettings: applySdxlImg2ImgPresetSettings", sdxl_img2img_js)

    def test_sdxl_inpaint_script_consumes_controlnet_state(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl_inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_inpaint_js)
        self.assertIn("controlnetEnabled", sdxl_inpaint_js)
        self.assertIn("control_images", sdxl_inpaint_js)
        self.assertIn("controlnet_models", sdxl_inpaint_js)

    def test_sdxl_inpaint_script_wires_lora_panel_and_payload(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl_inpaint.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_inpaint_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_inpaint_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", sdxl_inpaint_js)

    def test_sdxl_inpaint_page_and_script_wire_ip_adapter_payload(self):
        sdxl_inpaint_html = (ROOT / "frontend" / "sdxl_inpaint.html").read_text(encoding="utf-8")
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl_inpaint.js").read_text(encoding="utf-8")

        self.assertIn('id="ip_adapter_panel"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_toggle"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_content"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_enabled"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_image"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_preview"', sdxl_inpaint_html)
        self.assertIn('id="ip_adapter_scale"', sdxl_inpaint_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_inpaint_html)
        self.assertIn("window.IpAdapterPanel?.init()", sdxl_inpaint_js)
        self.assertIn("getIpAdapterImageFile", sdxl_inpaint_js)
        self.assertIn("taskInputs.ip_adapter = {", sdxl_inpaint_js)
        self.assertIn('subfolder: "sdxl_models"', sdxl_inpaint_js)
        self.assertIn('weight_name: "ip-adapter_sdxl.bin"', sdxl_inpaint_js)
        self.assertIn("SDXL inpaint IP-Adapter cannot be combined with ControlNet yet.", sdxl_inpaint_js)

    def test_sdxl_inpaint_script_wires_preset_panel(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl_inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_inpaint_js)
        self.assertIn('taskType: "sdxl.inpaint"', sdxl_inpaint_js)
        self.assertIn("collectSettings: collectSdxlInpaintPresetSettings", sdxl_inpaint_js)
        self.assertIn("applySettings: applySdxlInpaintPresetSettings", sdxl_inpaint_js)

    def test_z_image_page_includes_lora_script_before_z_image(self):
        z_image_html = (ROOT / "frontend" / "z_image.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        z_image_tag = '<script src="z_image.js?v=2"></script>'

        self.assertIn(lora_tag, z_image_html)
        self.assertIn(preset_tag, z_image_html)
        self.assertIn(z_image_tag, z_image_html)
        self.assertLess(z_image_html.index(lora_tag), z_image_html.index(z_image_tag))
        self.assertLess(z_image_html.index(preset_tag), z_image_html.index(z_image_tag))

    def test_z_image_script_wires_lora_panel_and_payload(self):
        z_image_js = (ROOT / "frontend" / "z_image.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "z-image" })', z_image_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", z_image_js)
        self.assertIn("inputs.Lora = {", z_image_js)
        self.assertIn("inputs.lora_adapters = loraAdapters;", z_image_js)

    def test_z_image_script_wires_preset_panel(self):
        z_image_js = (ROOT / "frontend" / "z_image.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", z_image_js)
        self.assertIn('taskType: "z-image.text2img"', z_image_js)
        self.assertIn("collectSettings: collectZImagePresetSettings", z_image_js)
        self.assertIn("applySettings: applyZImagePresetSettings", z_image_js)

    def test_z_image_img2img_page_includes_lora_script_before_z_image_img2img(self):
        z_image_img2img_html = (ROOT / "frontend" / "z_image_img2img.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        z_image_img2img_tag = '<script src="z_image_img2img.js?v=2"></script>'

        self.assertIn(lora_tag, z_image_img2img_html)
        self.assertIn(preset_tag, z_image_img2img_html)
        self.assertIn(z_image_img2img_tag, z_image_img2img_html)
        self.assertLess(z_image_img2img_html.index(lora_tag), z_image_img2img_html.index(z_image_img2img_tag))
        self.assertLess(
            z_image_img2img_html.index(preset_tag),
            z_image_img2img_html.index(z_image_img2img_tag),
        )

    def test_z_image_img2img_script_wires_lora_panel_and_payload(self):
        z_image_img2img_js = (ROOT / "frontend" / "z_image_img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "z-image" })', z_image_img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", z_image_img2img_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", z_image_img2img_js)

    def test_z_image_img2img_script_wires_preset_panel(self):
        z_image_img2img_js = (ROOT / "frontend" / "z_image_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", z_image_img2img_js)
        self.assertIn('taskType: "z-image.img2img"', z_image_img2img_js)
        self.assertIn("collectSettings: collectZImageImg2ImgPresetSettings", z_image_img2img_js)
        self.assertIn("applySettings: applyZImageImg2ImgPresetSettings", z_image_img2img_js)

    def test_z_image_inpaint_page_includes_lora_script_before_z_image_inpaint(self):
        z_image_inpaint_html = (ROOT / "frontend" / "z_image_inpaint.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        z_image_inpaint_tag = '<script src="z_image_inpaint.js?v=1"></script>'

        self.assertIn(lora_tag, z_image_inpaint_html)
        self.assertIn(preset_tag, z_image_inpaint_html)
        self.assertIn(z_image_inpaint_tag, z_image_inpaint_html)
        self.assertLess(
            z_image_inpaint_html.index(lora_tag),
            z_image_inpaint_html.index(z_image_inpaint_tag),
        )
        self.assertLess(
            z_image_inpaint_html.index(preset_tag),
            z_image_inpaint_html.index(z_image_inpaint_tag),
        )

    def test_z_image_inpaint_script_wires_lora_panel_and_payload(self):
        z_image_inpaint_js = (ROOT / "frontend" / "z_image_inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'window.LoraPanel?.init({ apiBase: API_BASE, family: "z-image" })',
            z_image_inpaint_js,
        )
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", z_image_inpaint_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", z_image_inpaint_js)

    def test_z_image_inpaint_script_wires_preset_panel(self):
        z_image_inpaint_js = (ROOT / "frontend" / "z_image_inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.PresetPanel?.init({", z_image_inpaint_js)
        self.assertIn('taskType: "z-image.inpaint"', z_image_inpaint_js)
        self.assertIn("collectSettings: collectZImageInpaintPresetSettings", z_image_inpaint_js)
        self.assertIn("applySettings: applyZImageInpaintPresetSettings", z_image_inpaint_js)

    def test_flux_page_includes_lora_script_before_flux(self):
        flux_html = (ROOT / "frontend" / "flux.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        flux_tag = '<script src="flux.js?v=2"></script>'

        self.assertIn(lora_tag, flux_html)
        self.assertIn(preset_tag, flux_html)
        self.assertIn(flux_tag, flux_html)
        self.assertLess(flux_html.index(lora_tag), flux_html.index(flux_tag))
        self.assertLess(flux_html.index(preset_tag), flux_html.index(flux_tag))

    def test_flux_script_wires_lora_panel_and_payload(self):
        flux_js = (ROOT / "frontend" / "flux.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "flux" })', flux_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", flux_js)
        self.assertIn("payload.lora_adapters = loraAdapters;", flux_js)

    def test_flux_script_wires_preset_panel(self):
        flux_js = (ROOT / "frontend" / "flux.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", flux_js)
        self.assertIn('taskType: "flux.text2img"', flux_js)
        self.assertIn("collectSettings: collectFluxPresetSettings", flux_js)
        self.assertIn("applySettings: applyFluxPresetSettings", flux_js)

    def test_flux_img2img_page_includes_lora_script_before_flux_img2img(self):
        flux_img2img_html = (ROOT / "frontend" / "flux_img2img.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        flux_img2img_tag = '<script src="flux_img2img.js?v=2"></script>'

        self.assertIn(lora_tag, flux_img2img_html)
        self.assertIn(preset_tag, flux_img2img_html)
        self.assertIn(flux_img2img_tag, flux_img2img_html)
        self.assertLess(
            flux_img2img_html.index(lora_tag), flux_img2img_html.index(flux_img2img_tag)
        )
        self.assertLess(
            flux_img2img_html.index(preset_tag), flux_img2img_html.index(flux_img2img_tag)
        )

    def test_flux_img2img_script_wires_lora_panel_and_payload(self):
        flux_img2img_js = (ROOT / "frontend" / "flux_img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "flux" })', flux_img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", flux_img2img_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", flux_img2img_js)

    def test_flux_img2img_script_wires_preset_panel(self):
        flux_img2img_js = (ROOT / "frontend" / "flux_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", flux_img2img_js)
        self.assertIn('taskType: "flux.img2img"', flux_img2img_js)
        self.assertIn("collectSettings: collectFluxImg2ImgPresetSettings", flux_img2img_js)
        self.assertIn("applySettings: applyFluxImg2ImgPresetSettings", flux_img2img_js)

    def test_flux_inpaint_page_includes_lora_script_before_flux_inpaint(self):
        flux_inpaint_html = (ROOT / "frontend" / "flux_inpaint.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        flux_inpaint_tag = '<script src="flux_inpaint.js?v=2"></script>'

        self.assertIn(lora_tag, flux_inpaint_html)
        self.assertIn(preset_tag, flux_inpaint_html)
        self.assertIn(flux_inpaint_tag, flux_inpaint_html)
        self.assertLess(
            flux_inpaint_html.index(lora_tag), flux_inpaint_html.index(flux_inpaint_tag)
        )
        self.assertLess(
            flux_inpaint_html.index(preset_tag), flux_inpaint_html.index(flux_inpaint_tag)
        )

    def test_flux_inpaint_script_wires_lora_panel_and_payload(self):
        flux_inpaint_js = (ROOT / "frontend" / "flux_inpaint.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "flux" })', flux_inpaint_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", flux_inpaint_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", flux_inpaint_js)

    def test_flux_inpaint_script_wires_preset_panel(self):
        flux_inpaint_js = (ROOT / "frontend" / "flux_inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", flux_inpaint_js)
        self.assertIn('taskType: "flux.inpaint"', flux_inpaint_js)
        self.assertIn("collectSettings: collectFluxInpaintPresetSettings", flux_inpaint_js)
        self.assertIn("applySettings: applyFluxInpaintPresetSettings", flux_inpaint_js)

    def test_qwen_image_page_includes_lora_script_before_qwen_image(self):
        qwen_image_html = (ROOT / "frontend" / "qwen_image.html").read_text(encoding="utf-8")
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        qwen_image_tag = '<script src="qwen_image.js?v=2"></script>'

        self.assertIn(lora_tag, qwen_image_html)
        self.assertIn(preset_tag, qwen_image_html)
        self.assertIn(qwen_image_tag, qwen_image_html)
        self.assertLess(qwen_image_html.index(lora_tag), qwen_image_html.index(qwen_image_tag))
        self.assertLess(qwen_image_html.index(preset_tag), qwen_image_html.index(qwen_image_tag))

    def test_qwen_image_script_wires_lora_panel_and_payload(self):
        qwen_image_js = (ROOT / "frontend" / "qwen_image.js").read_text(encoding="utf-8")
        self.assertIn(
            'window.LoraPanel?.init({ apiBase: API_BASE, family: "qwen-image" })',
            qwen_image_js,
        )
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", qwen_image_js)
        self.assertIn("payload.lora_adapters = loraAdapters;", qwen_image_js)

    def test_qwen_image_script_wires_preset_panel(self):
        qwen_image_js = (ROOT / "frontend" / "qwen_image.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", qwen_image_js)
        self.assertIn('taskType: "qwen-image.text2img"', qwen_image_js)
        self.assertIn("collectSettings: collectQwenImagePresetSettings", qwen_image_js)
        self.assertIn("applySettings: applyQwenImagePresetSettings", qwen_image_js)

    def test_qwen_image_img2img_page_includes_lora_script_before_qwen_image_img2img(self):
        qwen_image_img2img_html = (ROOT / "frontend" / "qwen_image_img2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        qwen_image_img2img_tag = '<script src="qwen_image_img2img.js?v=2"></script>'

        self.assertIn(lora_tag, qwen_image_img2img_html)
        self.assertIn(preset_tag, qwen_image_img2img_html)
        self.assertIn(qwen_image_img2img_tag, qwen_image_img2img_html)
        self.assertLess(
            qwen_image_img2img_html.index(lora_tag),
            qwen_image_img2img_html.index(qwen_image_img2img_tag),
        )
        self.assertLess(
            qwen_image_img2img_html.index(preset_tag),
            qwen_image_img2img_html.index(qwen_image_img2img_tag),
        )

    def test_qwen_image_img2img_script_wires_lora_panel_and_payload(self):
        qwen_image_img2img_js = (ROOT / "frontend" / "qwen_image_img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'window.LoraPanel?.init({ apiBase: API_BASE, family: "qwen-image" })',
            qwen_image_img2img_js,
        )
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", qwen_image_img2img_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", qwen_image_img2img_js)

    def test_qwen_image_img2img_script_wires_preset_panel(self):
        qwen_image_img2img_js = (ROOT / "frontend" / "qwen_image_img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.PresetPanel?.init({", qwen_image_img2img_js)
        self.assertIn('taskType: "qwen-image.img2img"', qwen_image_img2img_js)
        self.assertIn("collectSettings: collectQwenImageImg2ImgPresetSettings", qwen_image_img2img_js)
        self.assertIn("applySettings: applyQwenImageImg2ImgPresetSettings", qwen_image_img2img_js)

    def test_qwen_image_inpaint_page_includes_lora_script_before_qwen_image_inpaint(self):
        qwen_image_inpaint_html = (ROOT / "frontend" / "qwen_image_inpaint.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        qwen_image_inpaint_tag = '<script src="qwen_image_inpaint.js?v=2"></script>'

        self.assertIn(lora_tag, qwen_image_inpaint_html)
        self.assertIn(preset_tag, qwen_image_inpaint_html)
        self.assertIn(qwen_image_inpaint_tag, qwen_image_inpaint_html)
        self.assertLess(
            qwen_image_inpaint_html.index(lora_tag),
            qwen_image_inpaint_html.index(qwen_image_inpaint_tag),
        )
        self.assertLess(
            qwen_image_inpaint_html.index(preset_tag),
            qwen_image_inpaint_html.index(qwen_image_inpaint_tag),
        )

    def test_qwen_image_inpaint_script_wires_lora_panel_and_payload(self):
        qwen_image_inpaint_js = (ROOT / "frontend" / "qwen_image_inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'window.LoraPanel?.init({ apiBase: API_BASE, family: "qwen-image" })',
            qwen_image_inpaint_js,
        )
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", qwen_image_inpaint_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", qwen_image_inpaint_js)

    def test_qwen_image_inpaint_script_wires_preset_panel(self):
        qwen_image_inpaint_js = (ROOT / "frontend" / "qwen_image_inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.PresetPanel?.init({", qwen_image_inpaint_js)
        self.assertIn('taskType: "qwen-image.inpaint"', qwen_image_inpaint_js)
        self.assertIn("collectSettings: collectQwenImageInpaintPresetSettings", qwen_image_inpaint_js)
        self.assertIn("applySettings: applyQwenImageInpaintPresetSettings", qwen_image_inpaint_js)

    def test_preprocessor_modal_has_two_column_layout_hooks(self):
        preprocessor_html = (ROOT / "frontend" / "controlnet_preprocessor.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('class="modal-body preprocessor-layout"', preprocessor_html)
        self.assertIn('class="preprocessor-settings"', preprocessor_html)
        self.assertIn('class="preprocessor-preview preprocessor-preview-panel"', preprocessor_html)
        self.assertIn("grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);", preprocessor_html)

    def test_preprocessor_modal_styles_define_viewport_height_preview(self):
        style_css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
        preprocessor_html = (ROOT / "frontend" / "controlnet_preprocessor.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("#preprocessor-modal .preprocessor-layout", style_css)
        self.assertIn("#preprocessor-modal .preprocessor-preview-panel img", style_css)
        self.assertIn("max-height: calc(94vh - 220px);", style_css)
        self.assertIn("@media (max-width: 700px)", style_css)
        self.assertIn("max-height: calc(94vh - 220px);", preprocessor_html)


if __name__ == "__main__":
    unittest.main()
