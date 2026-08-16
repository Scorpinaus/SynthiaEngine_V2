from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendControlNetScriptTests(unittest.TestCase):
    def test_sd15_page_includes_controlnet_scripts_before_sd15(self):
        sd15_html = (ROOT / "frontend" / "sd15" / "text2img.html").read_text(encoding="utf-8")
        validator_tag = '<script src="../workflow_input_validator.js?v=1"></script>'
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=3"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=2"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=2"></script>'
        sd15_tag = '<script src="text2img.js?v=6"></script>'

        self.assertIn(validator_tag, sd15_html)
        self.assertIn(adapter_panel_tag, sd15_html)
        self.assertIn(panel_tag, sd15_html)
        self.assertIn(preprocessor_tag, sd15_html)
        self.assertIn(preset_tag, sd15_html)
        self.assertIn(lora_tag, sd15_html)
        self.assertIn(ip_adapter_tag, sd15_html)
        self.assertIn(sd15_tag, sd15_html)
        self.assertLess(sd15_html.index(validator_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(adapter_panel_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(panel_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(preprocessor_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(preset_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(lora_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(ip_adapter_tag), sd15_html.index(sd15_tag))

    def test_sd15_img2img_page_includes_controlnet_scripts_before_img2img(self):
        sd15_img2img_html = (ROOT / "frontend" / "sd15" / "img2img.html").read_text(encoding="utf-8")
        validator_tag = '<script src="../workflow_input_validator.js?v=1"></script>'
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=3"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=2"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=2"></script>'
        img2img_tag = '<script src="img2img.js?v=6"></script>'

        self.assertIn(validator_tag, sd15_img2img_html)
        self.assertIn(adapter_panel_tag, sd15_img2img_html)
        self.assertIn(panel_tag, sd15_img2img_html)
        self.assertIn(preprocessor_tag, sd15_img2img_html)
        self.assertIn(preset_tag, sd15_img2img_html)
        self.assertIn(lora_tag, sd15_img2img_html)
        self.assertIn(ip_adapter_tag, sd15_img2img_html)
        self.assertIn(img2img_tag, sd15_img2img_html)
        self.assertLess(sd15_img2img_html.index(validator_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(sd15_img2img_html.index(adapter_panel_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(sd15_img2img_html.index(panel_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(
            sd15_img2img_html.index(preprocessor_tag), sd15_img2img_html.index(img2img_tag)
        )
        self.assertLess(sd15_img2img_html.index(preset_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(sd15_img2img_html.index(lora_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(sd15_img2img_html.index(ip_adapter_tag), sd15_img2img_html.index(img2img_tag))

    def test_sd15_inpaint_page_includes_controlnet_scripts_before_inpaint(self):
        sd15_inpaint_html = (ROOT / "frontend" / "sd15" / "inpainting.html").read_text(encoding="utf-8")
        validator_tag = '<script src="../workflow_input_validator.js?v=1"></script>'
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=3"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=2"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=2"></script>'
        inpaint_tag = '<script src="inpainting.js?v=6"></script>'

        self.assertIn(validator_tag, sd15_inpaint_html)
        self.assertIn(adapter_panel_tag, sd15_inpaint_html)
        self.assertIn(panel_tag, sd15_inpaint_html)
        self.assertIn(preprocessor_tag, sd15_inpaint_html)
        self.assertIn(preset_tag, sd15_inpaint_html)
        self.assertIn(lora_tag, sd15_inpaint_html)
        self.assertIn(ip_adapter_tag, sd15_inpaint_html)
        self.assertIn(inpaint_tag, sd15_inpaint_html)
        self.assertLess(sd15_inpaint_html.index(validator_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(sd15_inpaint_html.index(adapter_panel_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(sd15_inpaint_html.index(panel_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(
            sd15_inpaint_html.index(preprocessor_tag), sd15_inpaint_html.index(inpaint_tag)
        )
        self.assertLess(sd15_inpaint_html.index(preset_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(sd15_inpaint_html.index(lora_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(sd15_inpaint_html.index(ip_adapter_tag), sd15_inpaint_html.index(inpaint_tag))

    def test_sdxl_page_includes_controlnet_scripts_before_sdxl(self):
        sdxl_html = (ROOT / "frontend" / "sdxl" / "text2img.html").read_text(
            encoding="utf-8"
        )
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        sdxl_tag = '<script src="text2img.js?v=6"></script>'

        self.assertIn(adapter_panel_tag, sdxl_html)
        self.assertIn(panel_tag, sdxl_html)
        self.assertIn(preprocessor_tag, sdxl_html)
        self.assertIn(lora_tag, sdxl_html)
        self.assertIn(ip_adapter_tag, sdxl_html)
        self.assertIn(preset_tag, sdxl_html)
        self.assertNotIn("sdxl_ip_adapter_panel.js", sdxl_html)
        self.assertIn(sdxl_tag, sdxl_html)
        self.assertLess(sdxl_html.index(adapter_panel_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(panel_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(preprocessor_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(lora_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(ip_adapter_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(preset_tag), sdxl_html.index(sdxl_tag))

    def test_sdxl_img2img_page_includes_controlnet_scripts_before_sdxl_img2img(self):
        sdxl_img2img_html = (ROOT / "frontend" / "sdxl" / "img2img.html").read_text(
            encoding="utf-8"
        )
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        sdxl_img2img_tag = '<script src="img2img.js?v=4"></script>'

        self.assertIn(adapter_panel_tag, sdxl_img2img_html)
        self.assertIn(panel_tag, sdxl_img2img_html)
        self.assertIn(preprocessor_tag, sdxl_img2img_html)
        self.assertIn(lora_tag, sdxl_img2img_html)
        self.assertIn(ip_adapter_tag, sdxl_img2img_html)
        self.assertIn(preset_tag, sdxl_img2img_html)
        self.assertNotIn("sdxl_ip_adapter_panel.js", sdxl_img2img_html)
        self.assertIn(sdxl_img2img_tag, sdxl_img2img_html)
        self.assertLess(sdxl_img2img_html.index(adapter_panel_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(panel_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(preprocessor_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(lora_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(ip_adapter_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(preset_tag), sdxl_img2img_html.index(sdxl_img2img_tag))

    def test_sdxl_inpaint_page_includes_controlnet_scripts_before_sdxl_inpaint(self):
        sdxl_inpaint_html = (ROOT / "frontend" / "sdxl" / "inpaint.html").read_text(
            encoding="utf-8"
        )
        adapter_panel_tag = '<script src="../components/adapter_panel.js?v=1"></script>'
        panel_tag = '<script src="../components/controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="../components/controlnet_preprocessor.js?v=3"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        ip_adapter_tag = '<script src="../components/ip_adapter_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        sdxl_inpaint_tag = '<script src="inpaint.js?v=4"></script>'

        self.assertIn(adapter_panel_tag, sdxl_inpaint_html)
        self.assertIn(panel_tag, sdxl_inpaint_html)
        self.assertIn(preprocessor_tag, sdxl_inpaint_html)
        self.assertIn(lora_tag, sdxl_inpaint_html)
        self.assertIn(ip_adapter_tag, sdxl_inpaint_html)
        self.assertIn(preset_tag, sdxl_inpaint_html)
        self.assertNotIn("sdxl_ip_adapter_panel.js", sdxl_inpaint_html)
        self.assertIn(sdxl_inpaint_tag, sdxl_inpaint_html)
        self.assertLess(sdxl_inpaint_html.index(adapter_panel_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(panel_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(preprocessor_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(lora_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(ip_adapter_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(preset_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))

    def test_controlnet_panel_script_exposes_expected_api(self):
        panel_js = (ROOT / "frontend" / "components" / "controlnet_panel.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel", panel_js)
        self.assertIn("getState", panel_js)
        self.assertIn("getSummary", panel_js)
        self.assertIn("loadPanel", panel_js)
        self.assertIn("updateIndicator", panel_js)
        self.assertIn("adapter-summary-changed", panel_js)
        self.assertIn('fetch(resolveAssetUrl("controlnet_panel.html?v=2"), { cache: "no-store" })', panel_js)

    def test_controlnet_preprocessor_script_exposes_expected_api(self):
        preprocessor_js = (ROOT / "frontend" / "components" / "controlnet_preprocessor.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPreprocessor", preprocessor_js)
        self.assertIn("ensureControlNetUI", preprocessor_js)
        self.assertIn("openPreprocessorModal", preprocessor_js)
        self.assertIn('fetch(resolveAssetUrl("controlnet_preprocessor.html?v=2"), { cache: "no-store" })', preprocessor_js)
        self.assertIn("ensurePreprocessorLayoutStructure", preprocessor_js)
        self.assertIn("gridTemplateColumns", preprocessor_js)
        self.assertIn("window.innerWidth <= 700", preprocessor_js)

    def test_controlnet_preprocessor_script_renders_generic_param_schema(self):
        preprocessor_js = (ROOT / "frontend" / "components" / "controlnet_preprocessor.js").read_text(
            encoding="utf-8"
        )
        preprocessor_html = (ROOT / "frontend" / "components" / "controlnet_preprocessor.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="preprocessor-params"', preprocessor_html)
        self.assertIn("renderPreprocessorParams", preprocessor_js)
        self.assertIn("data-preprocessor-param", preprocessor_js)
        self.assertIn("spec?.type === \"bool\"", preprocessor_js)
        self.assertIn("Object.entries(schema).forEach", preprocessor_js)
        self.assertIn("preprocessor.available === false", preprocessor_js)
        self.assertIn("option.disabled = true", preprocessor_js)
        self.assertIn("definition.install_hint", preprocessor_js)
        self.assertNotIn('id="canny-thresholds"', preprocessor_html)

    def test_sd15_controlnet_script_wires_per_item_guidance_timing(self):
        panel_js = (ROOT / "frontend" / "components" / "controlnet_panel.js").read_text(encoding="utf-8")
        preprocessor_js = (ROOT / "frontend" / "components" / "controlnet_preprocessor.js").read_text(
            encoding="utf-8"
        )
        sd15_js = (ROOT / "frontend" / "sd15" / "text2img.js").read_text(encoding="utf-8")

        self.assertIn("data-guidance-start-id", panel_js)
        self.assertIn("data-guidance-end-id", panel_js)
        self.assertIn("guidanceStart: Number(guidanceStart ?? 0.0)", panel_js)
        self.assertIn("guidanceEnd: Number(guidanceEnd ?? 1.0)", panel_js)
        self.assertIn("updateControlItemFromField", panel_js)
        self.assertIn('target.hasAttribute("data-scale-id")', panel_js)
        self.assertIn('target.hasAttribute("data-guidance-start-id")', panel_js)
        self.assertIn('target.hasAttribute("data-guidance-end-id")', panel_js)
        self.assertIn('itemsContainer?.addEventListener("input"', preprocessor_js)
        self.assertIn("updateControlItemFromField?.(event.target)", preprocessor_js)
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
        lora_html = (ROOT / "frontend" / "components" / "lora_panel.html").read_text(encoding="utf-8")
        self.assertIn('id="lora-weight-mode-row"', lora_html)
        self.assertIn('id="lora-weight-mode-basic"', lora_html)
        self.assertIn('id="lora-weight-mode-advanced"', lora_html)

    def test_lora_panel_script_supports_sd15_advanced_component_strengths(self):
        lora_js = (ROOT / "frontend" / "components" / "lora_panel.js").read_text(encoding="utf-8")
        self.assertIn("weightMode", lora_js)
        self.assertIn("lora-weight-mode-advanced", lora_js)
        self.assertIn("unet_strength", lora_js)
        self.assertIn("text_encoder_strength", lora_js)
        self.assertIn("getSummary", lora_js)

    def test_lora_panel_script_uses_qwen_transformer_strength_contract(self):
        lora_js = (ROOT / "frontend" / "components" / "lora_panel.js").read_text(encoding="utf-8")
        self.assertIn('loraState.family === "qwen-image"', lora_js)
        self.assertIn('isQwenImageFamily() ? "Qwen transformer" : "Strength"', lora_js)
        self.assertGreaterEqual(lora_js.count("if (!isQwenImageFamily())"), 2)
        self.assertIn('item.target = lora.target ?? "both"', lora_js)
        self.assertIn('fetch(resolveAssetUrl("lora_panel.html?v=2"))', lora_js)

    def test_lora_panel_script_exposes_prompt_preset_words(self):
        lora_js = (ROOT / "frontend" / "components" / "lora_panel.js").read_text(encoding="utf-8")
        self.assertIn("prompt_presets", lora_js)
        self.assertIn("updateLoraPromptPreset", lora_js)
        self.assertIn("getSelectedPresetWords", lora_js)
        self.assertIn("openPromptPresetModal", lora_js)
        self.assertIn("lora_prompt_preset_editor.js", lora_js)
        self.assertIn("Edit Prompt Presets", lora_js)
        self.assertIn("Add Prompt Presets", lora_js)

    def test_lora_prompt_preset_editor_page_uses_shared_editor(self):
        prompt_preset_html = (ROOT / "frontend" / "models" / "lora" / "prompt_presets.html").read_text(
            encoding="utf-8"
        )
        prompt_preset_js = (ROOT / "frontend" / "models" / "lora" / "prompt_presets.js").read_text(
            encoding="utf-8"
        )
        editor_js = (ROOT / "frontend" / "components" / "lora_prompt_preset_editor.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("lora-prompt-preset-editor-root", prompt_preset_html)
        self.assertIn("lora_prompt_preset_editor.js", prompt_preset_html)
        self.assertIn("window.LoraPromptPresetEditor?.mount", prompt_preset_js)
        self.assertIn("PATCH", editor_js)
        self.assertIn("prompt_presets", editor_js)

    def test_preset_panel_html_has_mode_specific_controls(self):
        preset_html = (ROOT / "frontend" / "components" / "preset_panel.html").read_text(encoding="utf-8")
        self.assertIn('id="preset-load"', preset_html)
        self.assertIn('id="preset-refresh"', preset_html)
        self.assertIn('id="preset-add-new"', preset_html)
        self.assertIn('id="preset-name-field"', preset_html)
        self.assertIn('id="preset-create-actions"', preset_html)
        self.assertIn('id="preset-manage-actions"', preset_html)
        self.assertIn('id="preset-cancel"', preset_html)

    def test_preset_panel_script_supports_create_and_manage_modes(self):
        preset_js = (ROOT / "frontend" / "components" / "preset_panel.js").read_text(encoding="utf-8")
        self.assertIn("const UI_MODES", preset_js)
        self.assertIn("setUiMode(UI_MODES.MANAGE)", preset_js)
        self.assertIn('document.getElementById("preset-add-new")', preset_js)
        self.assertIn('document.getElementById("preset-cancel")', preset_js)

    def test_sd15_img2img_script_consumes_controlnet_state(self):
        img2img_js = (ROOT / "frontend" / "sd15" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", img2img_js)
        self.assertIn("controlnetEnabled", img2img_js)
        self.assertIn("control_images", img2img_js)
        self.assertIn("controlnet_models", img2img_js)

    def test_sd15_script_wires_preset_panel(self):
        sd15_js = (ROOT / "frontend" / "sd15" / "text2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sd15_js)
        self.assertIn('taskType: "sd15.text2img"', sd15_js)
        self.assertIn("collectSettings: collectSd15PresetSettings", sd15_js)
        self.assertIn("applySettings: applySd15PresetSettings", sd15_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", sd15_js)
        self.assertIn("initAdapterModal", sd15_js)
        self.assertIn("updateAdapterSummary", sd15_js)

    def test_sd15_script_wires_lcm_mode_payload_and_guardrails(self):
        sd15_html = (ROOT / "frontend" / "sd15" / "text2img.html").read_text(encoding="utf-8")
        sd15_js = (ROOT / "frontend" / "sd15" / "text2img.js").read_text(encoding="utf-8")
        scheduler_html = (ROOT / "frontend" / "components" / "scheduler_panel.html").read_text(
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
        sd15_html = (ROOT / "frontend" / "sd15" / "text2img.html").read_text(encoding="utf-8")
        adapter_panel_js = (ROOT / "frontend" / "components" / "adapter_panel.js").read_text(
            encoding="utf-8"
        )
        adapter_panel_html = (ROOT / "frontend" / "components" / "adapter_panel.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="adapter-panel-root"', sd15_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt"', sd15_html)
        self.assertIn("ip_adapter_panel.js?v=2", sd15_html)
        self.assertIn('id="adapter-modal-open"', adapter_panel_html)
        self.assertIn('id="adapter-modal"', adapter_panel_html)
        self.assertIn('data-adapter-tab="overview"', adapter_panel_html)
        self.assertIn('data-adapter-tab="controlnet"', adapter_panel_html)
        self.assertIn('data-adapter-tab="lora"', adapter_panel_html)
        self.assertIn('data-adapter-tab="ipadapter"', adapter_panel_html)
        self.assertIn('id="adapter-overview-controlnet-count"', adapter_panel_html)
        self.assertIn('id="controlnet-panel-root"', adapter_panel_html)
        self.assertIn('id="lora-panel-root"', adapter_panel_html)
        self.assertIn('id="ip_adapter_panel"', adapter_panel_html)
        self.assertIn('id="ip_adapter_toggle"', adapter_panel_html)
        self.assertIn('id="ip_adapter_content"', adapter_panel_html)
        self.assertIn('id="ip_adapter_enabled"', adapter_panel_html)
        self.assertIn('id="ip_adapter_image"', adapter_panel_html)
        self.assertIn('id="ip_adapter_preview"', adapter_panel_html)
        self.assertIn('id="ip_adapter_mask_image"', adapter_panel_html)
        self.assertIn('id="ip_adapter_mask_editor_open"', adapter_panel_html)
        self.assertIn('id="ip_adapter_mask_preview"', adapter_panel_html)
        self.assertIn('id="ip_adapter_scale"', adapter_panel_html)
        self.assertIn('adapter_panel.html?v=1', adapter_panel_js)
        self.assertIn("window.AdapterPanel", adapter_panel_js)
        self.assertIn("adapter-panel:loaded", adapter_panel_js)

    def test_adapter_modal_allows_preprocessor_modal_to_stack_above_it(self):
        style_css = (ROOT / "frontend" / "styles" / "generation.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("#adapter-modal {\n    z-index: 1000;", style_css)
        self.assertIn("#preprocessor-modal {\n    z-index: 1010;", style_css)

    def test_adapter_modal_suppresses_gallery_controls_while_open(self):
        adapter_panel_js = (ROOT / "frontend" / "components" / "adapter_panel.js").read_text(
            encoding="utf-8"
        )
        style_css = (ROOT / "frontend" / "styles" / "generation.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("initAdapterModalOpenState", adapter_panel_js)
        self.assertIn('"adapter-modal-open"', adapter_panel_js)
        self.assertIn("new MutationObserver(syncOpenState).observe", adapter_panel_js)
        self.assertIn("body.adapter-modal-open .viewer-controls", style_css)
        self.assertIn("visibility: hidden;", style_css)

    def test_sd15_script_wires_ip_adapter_payload_and_guardrails(self):
        sd15_js = (ROOT / "frontend" / "sd15" / "text2img.js").read_text(encoding="utf-8")

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
        img2img_js = (ROOT / "frontend" / "sd15" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", img2img_js)
        self.assertIn('taskType: "sd15.img2img"', img2img_js)
        self.assertIn("collectSettings: collectSd15Img2ImgPresetSettings", img2img_js)
        self.assertIn("applySettings: applySd15Img2ImgPresetSettings", img2img_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", img2img_js)

    def test_sd15_img2img_script_wires_lcm_mode_payload_and_guardrails(self):
        img2img_html = (ROOT / "frontend" / "sd15" / "img2img.html").read_text(encoding="utf-8")
        img2img_js = (ROOT / "frontend" / "sd15" / "img2img.js").read_text(encoding="utf-8")

        self.assertIn('id="lcm_enabled"', img2img_html)
        self.assertIn("function applyLcmImg2ImgContract(inputs)", img2img_js)
        self.assertIn("inputs.lcm = { enabled: true };", img2img_js)
        self.assertIn('inputs.scheduler = DEFAULTS.lcm_scheduler;', img2img_js)
        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 img2img yet.", img2img_js)

    def test_sd15_img2img_page_wires_ip_adapter_controls(self):
        img2img_html = (ROOT / "frontend" / "sd15" / "img2img.html").read_text(encoding="utf-8")

        self.assertIn('id="adapter-panel-root"', img2img_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt reference"', img2img_html)
        self.assertIn("ip_adapter_panel.js?v=2", img2img_html)

    def test_sd15_img2img_script_wires_ip_adapter_payload_and_guardrails(self):
        img2img_js = (ROOT / "frontend" / "sd15" / "img2img.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init({", img2img_js)
        self.assertIn("function initAdapterModal()", img2img_js)
        self.assertIn("function updateAdapterSummary()", img2img_js)
        self.assertIn('window.addEventListener("adapter-summary-changed", updateAdapterSummary)', img2img_js)
        self.assertIn("initAdapterModal();", img2img_js)
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
        inpaint_js = (ROOT / "frontend" / "sd15" / "inpainting.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", inpaint_js)
        self.assertIn('taskType: "sd15.inpaint"', inpaint_js)
        self.assertIn("collectSettings: collectSd15InpaintPresetSettings", inpaint_js)
        self.assertIn("applySettings: applySd15InpaintPresetSettings", inpaint_js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", inpaint_js)

    def test_sd15_inpaint_script_wires_lcm_mode_payload_and_guardrails(self):
        inpaint_html = (ROOT / "frontend" / "sd15" / "inpainting.html").read_text(
            encoding="utf-8"
        )
        inpaint_js = (ROOT / "frontend" / "sd15" / "inpainting.js").read_text(encoding="utf-8")

        self.assertIn('id="lcm_enabled"', inpaint_html)
        self.assertIn("function applyLcmInpaintContract(inputs)", inpaint_js)
        self.assertIn("inputs.lcm = { enabled: true };", inpaint_js)
        self.assertIn("inputs.scheduler = DEFAULTS.lcm_scheduler;", inpaint_js)
        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 inpaint yet.", inpaint_js)

    def test_sd15_inpaint_page_wires_ip_adapter_controls(self):
        inpaint_html = (ROOT / "frontend" / "sd15" / "inpainting.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="adapter-panel-root"', inpaint_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt reference"', inpaint_html)
        self.assertIn("ip_adapter_panel.js?v=2", inpaint_html)

    def test_sd15_inpaint_script_wires_ip_adapter_payload_and_guardrails(self):
        inpaint_js = (ROOT / "frontend" / "sd15" / "inpainting.js").read_text(encoding="utf-8")

        self.assertIn("window.IpAdapterPanel?.init({", inpaint_js)
        self.assertIn("function initAdapterModal()", inpaint_js)
        self.assertIn("function updateAdapterSummary()", inpaint_js)
        self.assertIn('window.addEventListener("adapter-summary-changed", updateAdapterSummary)', inpaint_js)
        self.assertIn("initAdapterModal();", inpaint_js)
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
        panel_js = (ROOT / "frontend" / "components" / "ip_adapter_panel.js").read_text(encoding="utf-8")

        self.assertIn("URL.createObjectURL(file)", panel_js)
        self.assertIn("URL.revokeObjectURL(previewUrl)", panel_js)
        self.assertIn("URL.revokeObjectURL(maskPreviewUrl)", panel_js)
        self.assertIn('document.getElementById("ip_adapter_image")', panel_js)
        self.assertIn('document.getElementById("ip_adapter_preview")', panel_js)
        self.assertIn('document.getElementById("ip_adapter_mask_image")', panel_js)
        self.assertIn("window.IpAdapterPanel = { init, getMaskFile, getSummary }", panel_js)
        self.assertIn("adapter-summary-changed", panel_js)
        self.assertIn('content.classList.toggle("is-open", isOpen)', panel_js)

    def test_sd15_img2img_script_wires_lora_panel_and_payload(self):
        img2img_js = (ROOT / "frontend" / "sd15" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" })', img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", img2img_js)
        self.assertIn("inputs.lora = {", img2img_js)
        self.assertIn("setLoraContract(taskInputs, loraAdapters);", img2img_js)

    def test_sd15_inpaint_script_consumes_controlnet_state(self):
        inpaint_js = (ROOT / "frontend" / "sd15" / "inpainting.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", inpaint_js)
        self.assertIn("controlnetEnabled", inpaint_js)
        self.assertIn("control_images", inpaint_js)
        self.assertIn("controlnet_models", inpaint_js)
        self.assertIn("controlnet_inpaint_condition", inpaint_js)
        self.assertIn("lllyasviel/control_v11p_sd15_inpaint", inpaint_js)
        self.assertIn("inpaint-condition", inpaint_js)
        self.assertIn("no separate preprocessor image is needed", inpaint_js)

    def test_sd15_inpaint_script_wires_lora_panel_and_payload(self):
        inpaint_js = (ROOT / "frontend" / "sd15" / "inpainting.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" })', inpaint_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", inpaint_js)
        self.assertIn("inputs.lora = {", inpaint_js)
        self.assertIn("setLoraContract(taskInputs, loraAdapters);", inpaint_js)

    def test_sdxl_script_consumes_controlnet_state(self):
        sdxl_js = (ROOT / "frontend" / "sdxl" / "text2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_js)
        self.assertIn("controlnetEnabled", sdxl_js)
        self.assertIn("control_images", sdxl_js)
        self.assertIn("controlnet_models", sdxl_js)

    def test_sdxl_script_wires_lora_panel_and_payload(self):
        sdxl_js = (ROOT / "frontend" / "sdxl" / "text2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_js)
        self.assertIn("inputs.lora_adapters = loraAdapters;", sdxl_js)
        self.assertIn("payload.lora_adapters = loraAdapters;", sdxl_js)

    def test_sdxl_page_wires_ip_adapter_controls(self):
        sdxl_html = (ROOT / "frontend" / "sdxl" / "text2img.html").read_text(encoding="utf-8")
        adapter_panel_js = (ROOT / "frontend" / "components" / "adapter_panel.js").read_text(
            encoding="utf-8"
        )
        adapter_panel_html = (ROOT / "frontend" / "components" / "adapter_panel.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="adapter-panel-root"', sdxl_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt"', sdxl_html)
        self.assertIn('data-ip-adapter-mask-enabled="false"', sdxl_html)
        self.assertIn('id="ip_adapter_panel"', adapter_panel_html)
        self.assertIn('id="ip_adapter_toggle"', adapter_panel_html)
        self.assertIn('id="ip_adapter_content"', adapter_panel_html)
        self.assertIn('id="ip_adapter_enabled"', adapter_panel_html)
        self.assertIn('id="ip_adapter_image"', adapter_panel_html)
        self.assertIn('id="ip_adapter_preview"', adapter_panel_html)
        self.assertIn('id="ip_adapter_scale"', adapter_panel_html)
        self.assertIn('data-ip-adapter-mask-section', adapter_panel_html)
        self.assertIn('container.dataset.ipAdapterMaskEnabled === "false"', adapter_panel_js)
        self.assertIn("adapter_panel.js?v=1", sdxl_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_html)

    def test_sdxl_script_wires_ip_adapter_payload_and_guardrails(self):
        sdxl_js = (ROOT / "frontend" / "sdxl" / "text2img.js").read_text(encoding="utf-8")

        self.assertIn("window.AdapterPanel?.render?.()", sdxl_js)
        self.assertIn("window.IpAdapterPanel?.init()", sdxl_js)
        self.assertIn("initAdapterModal", sdxl_js)
        self.assertIn("updateAdapterSummary", sdxl_js)
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
        sdxl_js = (ROOT / "frontend" / "sdxl" / "text2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_js)
        self.assertIn('taskType: "sdxl.text2img"', sdxl_js)
        self.assertIn("collectSettings: collectSdxlPresetSettings", sdxl_js)
        self.assertIn("applySettings: applySdxlPresetSettings", sdxl_js)

    def test_sdxl_img2img_script_consumes_controlnet_state(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_img2img_js)
        self.assertIn("controlnetEnabled", sdxl_img2img_js)
        self.assertIn("control_images", sdxl_img2img_js)
        self.assertIn("controlnet_models", sdxl_img2img_js)

    def test_sdxl_img2img_script_wires_lora_panel_and_payload(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_img2img_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_img2img_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", sdxl_img2img_js)

    def test_sdxl_img2img_page_and_script_wire_ip_adapter_payload(self):
        sdxl_img2img_html = (ROOT / "frontend" / "sdxl" / "img2img.html").read_text(
            encoding="utf-8"
        )
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl" / "img2img.js").read_text(encoding="utf-8")

        self.assertIn('id="adapter-panel-root"', sdxl_img2img_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt reference"', sdxl_img2img_html)
        self.assertIn('data-ip-adapter-mask-enabled="false"', sdxl_img2img_html)
        self.assertIn("adapter_panel.js?v=1", sdxl_img2img_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_img2img_html)
        self.assertIn("window.AdapterPanel?.render?.()", sdxl_img2img_js)
        self.assertIn("window.IpAdapterPanel?.init()", sdxl_img2img_js)
        self.assertIn("initAdapterModal", sdxl_img2img_js)
        self.assertIn("updateAdapterSummary", sdxl_img2img_js)
        self.assertIn("getIpAdapterImageFile", sdxl_img2img_js)
        self.assertIn("taskInputs.ip_adapter = {", sdxl_img2img_js)
        self.assertIn('subfolder: "sdxl_models"', sdxl_img2img_js)
        self.assertIn('weight_name: "ip-adapter_sdxl.bin"', sdxl_img2img_js)
        self.assertIn("SDXL img2img IP-Adapter cannot be combined with ControlNet yet.", sdxl_img2img_js)

    def test_sdxl_img2img_script_wires_preset_panel(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl" / "img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_img2img_js)
        self.assertIn('taskType: "sdxl.img2img"', sdxl_img2img_js)
        self.assertIn("collectSettings: collectSdxlImg2ImgPresetSettings", sdxl_img2img_js)
        self.assertIn("applySettings: applySdxlImg2ImgPresetSettings", sdxl_img2img_js)

    def test_sdxl_inpaint_script_consumes_controlnet_state(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl" / "inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_inpaint_js)
        self.assertIn("controlnetEnabled", sdxl_inpaint_js)
        self.assertIn("control_images", sdxl_inpaint_js)
        self.assertIn("controlnet_models", sdxl_inpaint_js)

    def test_sdxl_inpaint_script_wires_lora_panel_and_payload(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl" / "inpaint.js").read_text(encoding="utf-8")
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" })', sdxl_inpaint_js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", sdxl_inpaint_js)
        self.assertIn("taskInputs.lora_adapters = loraAdapters;", sdxl_inpaint_js)

    def test_sdxl_inpaint_page_and_script_wire_ip_adapter_payload(self):
        sdxl_inpaint_html = (ROOT / "frontend" / "sdxl" / "inpaint.html").read_text(
            encoding="utf-8"
        )
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl" / "inpaint.js").read_text(encoding="utf-8")

        self.assertIn('id="adapter-panel-root"', sdxl_inpaint_html)
        self.assertIn('data-ip-adapter-toggle-label="Use image prompt reference"', sdxl_inpaint_html)
        self.assertIn('data-ip-adapter-mask-enabled="false"', sdxl_inpaint_html)
        self.assertIn("adapter_panel.js?v=1", sdxl_inpaint_html)
        self.assertIn("ip_adapter_panel.js?v=1", sdxl_inpaint_html)
        self.assertIn("window.AdapterPanel?.render?.()", sdxl_inpaint_js)
        self.assertIn("window.IpAdapterPanel?.init()", sdxl_inpaint_js)
        self.assertIn("initAdapterModal", sdxl_inpaint_js)
        self.assertIn("updateAdapterSummary", sdxl_inpaint_js)
        self.assertIn("getIpAdapterImageFile", sdxl_inpaint_js)
        self.assertIn("taskInputs.ip_adapter = {", sdxl_inpaint_js)
        self.assertIn('subfolder: "sdxl_models"', sdxl_inpaint_js)
        self.assertIn('weight_name: "ip-adapter_sdxl.bin"', sdxl_inpaint_js)
        self.assertIn("SDXL inpaint IP-Adapter cannot be combined with ControlNet yet.", sdxl_inpaint_js)

    def test_sdxl_inpaint_script_wires_preset_panel(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl" / "inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", sdxl_inpaint_js)
        self.assertIn('taskType: "sdxl.inpaint"', sdxl_inpaint_js)
        self.assertIn("collectSettings: collectSdxlInpaintPresetSettings", sdxl_inpaint_js)
        self.assertIn("applySettings: applySdxlInpaintPresetSettings", sdxl_inpaint_js)

    def _assert_arc07_load_order(self):
        pages = {
            ("sd15", "text2img.html"): "text2img.js?v=7",
            ("sd15", "img2img.html"): "img2img.js?v=7",
            ("sd15", "inpainting.html"): "inpainting.js?v=7",
            ("sdxl", "text2img.html"): "text2img.js?v=7",
            ("sdxl", "img2img.html"): "img2img.js?v=5",
            ("sdxl", "inpaint.html"): "inpaint.js?v=5",
        }
        shared_scripts = (
            "../generation_page.js?v=4",
            "../components/adapter_controller.js?v=1",
            "../components/controlnet_controller.js?v=1",
            "../components/ip_adapter_controller.js?v=1",
            "generation_controller.js?v=1",
        )
        for (family, filename), entry_script in pages.items():
            html = (ROOT / "frontend" / family / filename).read_text(encoding="utf-8")
            for shared_script in shared_scripts:
                self.assertIn(shared_script, html)
                self.assertLess(html.index(shared_script), html.index(entry_script))
            if "inpaint" in filename:
                self.assertIn("../components/inpaint_editor.js?v=1", html)
                self.assertLess(
                    html.index("../components/inpaint_editor.js?v=1"),
                    html.index(entry_script),
                )

    def _assert_arc07_feature_composition(self):
        generation_page = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
        adapter = (ROOT / "frontend" / "components" / "adapter_controller.js").read_text(
            encoding="utf-8"
        )
        controlnet = (ROOT / "frontend" / "components" / "controlnet_controller.js").read_text(
            encoding="utf-8"
        )
        ip_adapter = (ROOT / "frontend" / "components" / "ip_adapter_controller.js").read_text(
            encoding="utf-8"
        )
        sd15 = (ROOT / "frontend" / "sd15" / "generation_controller.js").read_text(
            encoding="utf-8"
        )
        sdxl = (ROOT / "frontend" / "sdxl" / "generation_controller.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("createFormController", generation_page)
        self.assertIn("createJobController", generation_page)
        self.assertIn("WorkflowClient.watchJob", generation_page)
        self.assertIn("window.PresetPanel?.init", generation_page)
        self.assertIn("window.LoraPanel?.init", generation_page)
        self.assertIn("window.AdapterController", adapter)
        self.assertIn("adapter-summary-changed", adapter)

        self.assertIn("window.ControlNetPanel?.getState?.()", controlnet)
        self.assertIn("window.ControlNetPreprocessor.init()", controlnet)
        self.assertIn("attachSd15Text", controlnet)
        self.assertIn("attachSd15Image", controlnet)
        self.assertIn("attachSdxlText", controlnet)
        self.assertIn("attachSdxlImage", controlnet)
        self.assertIn("inputs.control_guidance_starts", controlnet)
        self.assertIn("inputs.control_guidance_ends", controlnet)
        self.assertIn("inputs.control_images", controlnet)
        self.assertIn("inputs.controlnet_models", controlnet)

        self.assertIn("window.IpAdapterPanel?.init", ip_adapter)
        self.assertIn("attachEncoded", ip_adapter)
        self.assertIn("attachDirect", ip_adapter)
        self.assertIn('id: "ip_embeds"', ip_adapter)
        self.assertIn('image_embeds: "@ip_embeds.image_embeds"', ip_adapter)
        self.assertIn('const MODEL = "h94/IP-Adapter"', ip_adapter)
        self.assertIn('"ip-adapter_sd15.bin"', ip_adapter)
        self.assertIn('"ip-adapter_sdxl.bin"', ip_adapter)
        self.assertIn("inputs.ip_adapter.mask_image", ip_adapter)

        for task_name in (
            "sd15.text2img",
            "sd15.controlnet.text2img",
            "sd15.hires_fix",
            "sd15.ip_adapter.encode",
            "sd15.img2img",
            "sd15.inpaint",
            "sdxl.text2img",
            "sdxl.controlnet.text2img",
            "sdxl.ip_adapter.encode",
            "sdxl.img2img",
            "sdxl.inpaint",
        ):
            family = "sd15" if task_name.startswith("sd15") else "sdxl"
            page_sources = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (ROOT / "frontend" / family).glob("*.js")
            )
            self.assertIn(task_name, page_sources)

        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 img2img yet.", sd15)
        self.assertIn("LCM mode cannot be combined with ControlNet for SD1.5 inpaint yet.", sd15)
        self.assertIn("IP-Adapter cannot be combined with Hi-Res Fix yet.", sd15)
        self.assertIn("lllyasviel/control_v11p_sd15_inpaint", sd15)
        self.assertIn("no separate preprocessor image is needed", sd15)
        self.assertIn("SDXL IP-Adapter cannot be combined with ControlNet yet.", sdxl)
        self.assertIn("SDXL img2img IP-Adapter cannot be combined with ControlNet yet.", sdxl)
        self.assertIn("SDXL inpaint IP-Adapter cannot be combined with ControlNet yet.", sdxl)

    # ARC-07 moves these contracts from page-local files into composed controllers.
    test_sd15_page_includes_controlnet_scripts_before_sd15 = _assert_arc07_load_order
    test_sd15_img2img_page_includes_controlnet_scripts_before_img2img = _assert_arc07_load_order
    test_sd15_inpaint_page_includes_controlnet_scripts_before_inpaint = _assert_arc07_load_order
    test_sdxl_page_includes_controlnet_scripts_before_sdxl = _assert_arc07_load_order
    test_sdxl_img2img_page_includes_controlnet_scripts_before_sdxl_img2img = _assert_arc07_load_order
    test_sdxl_inpaint_page_includes_controlnet_scripts_before_sdxl_inpaint = _assert_arc07_load_order

    test_sd15_controlnet_script_wires_per_item_guidance_timing = _assert_arc07_feature_composition
    test_sd15_img2img_script_consumes_controlnet_state = _assert_arc07_feature_composition
    test_sd15_img2img_script_wires_ip_adapter_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_img2img_script_wires_lcm_mode_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_img2img_script_wires_lora_panel_and_payload = _assert_arc07_feature_composition
    test_sd15_img2img_script_wires_preset_panel = _assert_arc07_feature_composition
    test_sd15_inpaint_script_consumes_controlnet_state = _assert_arc07_feature_composition
    test_sd15_inpaint_script_wires_ip_adapter_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_inpaint_script_wires_lcm_mode_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_inpaint_script_wires_lora_panel_and_payload = _assert_arc07_feature_composition
    test_sd15_inpaint_script_wires_preset_panel = _assert_arc07_feature_composition
    test_sd15_script_wires_ip_adapter_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_script_wires_lcm_mode_payload_and_guardrails = _assert_arc07_feature_composition
    test_sd15_script_wires_preset_panel = _assert_arc07_feature_composition
    test_sdxl_img2img_page_and_script_wire_ip_adapter_payload = _assert_arc07_feature_composition
    test_sdxl_img2img_script_consumes_controlnet_state = _assert_arc07_feature_composition
    test_sdxl_img2img_script_wires_lora_panel_and_payload = _assert_arc07_feature_composition
    test_sdxl_img2img_script_wires_preset_panel = _assert_arc07_feature_composition
    test_sdxl_inpaint_page_and_script_wire_ip_adapter_payload = _assert_arc07_feature_composition
    test_sdxl_inpaint_script_consumes_controlnet_state = _assert_arc07_feature_composition
    test_sdxl_inpaint_script_wires_lora_panel_and_payload = _assert_arc07_feature_composition
    test_sdxl_inpaint_script_wires_preset_panel = _assert_arc07_feature_composition
    test_sdxl_script_consumes_controlnet_state = _assert_arc07_feature_composition
    test_sdxl_script_wires_ip_adapter_payload_and_guardrails = _assert_arc07_feature_composition
    test_sdxl_script_wires_lora_panel_and_payload = _assert_arc07_feature_composition
    test_sdxl_script_wires_preset_panel = _assert_arc07_feature_composition

    def test_z_image_page_includes_lora_script_before_z_image(self):
        z_image_html = (ROOT / "frontend" / "z_image" / "text2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        z_image_tag = '<script src="text2img.js?v=3"></script>'

        self.assertIn(lora_tag, z_image_html)
        self.assertIn(preset_tag, z_image_html)
        self.assertIn(controller_tag, z_image_html)
        self.assertIn(z_image_tag, z_image_html)
        self.assertLess(z_image_html.index(lora_tag), z_image_html.index(z_image_tag))
        self.assertLess(z_image_html.index(preset_tag), z_image_html.index(z_image_tag))

    def test_z_image_script_wires_lora_panel_and_payload(self):
        z_image_js = (ROOT / "frontend" / "z_image" / "text2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "z-image"', z_image_js)
        self.assertIn("page.withLora", z_image_js)

    def test_z_image_script_wires_preset_panel(self):
        z_image_js = (ROOT / "frontend" / "z_image" / "text2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "z-image.text2img"', z_image_js)

    def test_z_image_img2img_page_includes_lora_script_before_z_image_img2img(self):
        z_image_img2img_html = (ROOT / "frontend" / "z_image" / "img2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        z_image_img2img_tag = '<script src="img2img.js?v=3"></script>'

        self.assertIn(lora_tag, z_image_img2img_html)
        self.assertIn(preset_tag, z_image_img2img_html)
        self.assertIn(controller_tag, z_image_img2img_html)
        self.assertIn(z_image_img2img_tag, z_image_img2img_html)
        self.assertLess(z_image_img2img_html.index(lora_tag), z_image_img2img_html.index(z_image_img2img_tag))
        self.assertLess(
            z_image_img2img_html.index(preset_tag),
            z_image_img2img_html.index(z_image_img2img_tag),
        )

    def test_z_image_img2img_script_wires_lora_panel_and_payload(self):
        z_image_img2img_js = (ROOT / "frontend" / "z_image" / "img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "z-image"', z_image_img2img_js)
        self.assertIn("page.withLora", z_image_img2img_js)
        self.assertIn("inputs.initial_image", z_image_img2img_js)

    def test_z_image_img2img_script_wires_preset_panel(self):
        z_image_img2img_js = (ROOT / "frontend" / "z_image" / "img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "z-image.img2img"', z_image_img2img_js)

    def test_z_image_inpaint_page_includes_lora_script_before_z_image_inpaint(self):
        z_image_inpaint_html = (ROOT / "frontend" / "z_image" / "inpaint.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        editor_tag = '<script src="../components/inpaint_editor.js?v=1"></script>'
        z_image_inpaint_tag = '<script src="inpaint.js?v=3"></script>'

        self.assertIn(lora_tag, z_image_inpaint_html)
        self.assertIn(preset_tag, z_image_inpaint_html)
        self.assertIn(editor_tag, z_image_inpaint_html)
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
        z_image_inpaint_js = (ROOT / "frontend" / "z_image" / "inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "z-image"', z_image_inpaint_js)
        self.assertIn("page.withLora", z_image_inpaint_js)
        self.assertIn("inputs.mask_image", z_image_inpaint_js)

    def test_z_image_inpaint_script_wires_preset_panel(self):
        z_image_inpaint_js = (ROOT / "frontend" / "z_image" / "inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "z-image.inpaint"', z_image_inpaint_js)

    def test_flux_page_includes_lora_script_before_flux(self):
        flux_html = (ROOT / "frontend" / "flux" / "text2img.html").read_text(encoding="utf-8")
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        flux_tag = '<script src="text2img.js?v=3"></script>'

        self.assertIn(lora_tag, flux_html)
        self.assertIn(preset_tag, flux_html)
        self.assertIn(controller_tag, flux_html)
        self.assertIn(flux_tag, flux_html)
        self.assertLess(flux_html.index(lora_tag), flux_html.index(flux_tag))
        self.assertLess(flux_html.index(preset_tag), flux_html.index(flux_tag))
        self.assertLess(flux_html.index(controller_tag), flux_html.index(flux_tag))

    def test_flux_script_wires_lora_panel_and_payload(self):
        flux_js = (ROOT / "frontend" / "flux" / "text2img.js").read_text(encoding="utf-8")
        controller_js = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
        self.assertIn('family: "flux"', flux_js)
        self.assertIn("page.withLora", flux_js)
        self.assertIn("window.LoraPanel?.init", controller_js)
        self.assertIn("inputs.lora_adapters", controller_js)

    def test_flux_script_wires_preset_panel(self):
        flux_js = (ROOT / "frontend" / "flux" / "text2img.js").read_text(encoding="utf-8")
        controller_js = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", controller_js)
        self.assertIn('taskType: "flux.text2img"', flux_js)
        self.assertIn("collectSettings,", controller_js)
        self.assertIn("applySettings,", controller_js)

    def test_flux_img2img_page_includes_lora_script_before_flux_img2img(self):
        flux_img2img_html = (ROOT / "frontend" / "flux" / "img2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        flux_img2img_tag = '<script src="img2img.js?v=3"></script>'

        self.assertIn(lora_tag, flux_img2img_html)
        self.assertIn(preset_tag, flux_img2img_html)
        self.assertIn(controller_tag, flux_img2img_html)
        self.assertIn(flux_img2img_tag, flux_img2img_html)
        self.assertLess(
            flux_img2img_html.index(lora_tag), flux_img2img_html.index(flux_img2img_tag)
        )
        self.assertLess(
            flux_img2img_html.index(preset_tag), flux_img2img_html.index(flux_img2img_tag)
        )
        self.assertLess(
            flux_img2img_html.index(controller_tag), flux_img2img_html.index(flux_img2img_tag)
        )

    def test_flux_img2img_script_wires_lora_panel_and_payload(self):
        flux_img2img_js = (ROOT / "frontend" / "flux" / "img2img.js").read_text(encoding="utf-8")
        controller_js = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
        self.assertIn('family: "flux"', flux_img2img_js)
        self.assertIn("alwaysSendLoraAdapters: true", flux_img2img_js)
        self.assertIn("page.withLora", flux_img2img_js)
        self.assertIn("inputs.lora_adapters", controller_js)

    def test_flux_img2img_script_wires_preset_panel(self):
        flux_img2img_js = (ROOT / "frontend" / "flux" / "img2img.js").read_text(encoding="utf-8")
        controller_js = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")
        self.assertIn("window.PresetPanel?.init({", controller_js)
        self.assertIn('taskType: "flux.img2img"', flux_img2img_js)
        self.assertIn("collectSettings,", controller_js)
        self.assertIn("applySettings,", controller_js)

    def test_flux_inpaint_page_includes_lora_script_before_flux_inpaint(self):
        flux_inpaint_html = (ROOT / "frontend" / "flux" / "inpaint.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        editor_tag = '<script src="../components/inpaint_editor.js?v=1"></script>'
        flux_inpaint_tag = '<script src="inpaint.js?v=3"></script>'

        self.assertIn(lora_tag, flux_inpaint_html)
        self.assertIn(preset_tag, flux_inpaint_html)
        self.assertIn(editor_tag, flux_inpaint_html)
        self.assertIn(flux_inpaint_tag, flux_inpaint_html)
        self.assertLess(
            flux_inpaint_html.index(lora_tag), flux_inpaint_html.index(flux_inpaint_tag)
        )
        self.assertLess(
            flux_inpaint_html.index(preset_tag), flux_inpaint_html.index(flux_inpaint_tag)
        )

    def test_flux_inpaint_script_wires_lora_panel_and_payload(self):
        flux_inpaint_js = (ROOT / "frontend" / "flux" / "inpaint.js").read_text(encoding="utf-8")
        self.assertIn('family: "flux"', flux_inpaint_js)
        self.assertIn("page.withLora", flux_inpaint_js)
        self.assertIn("inputs.mask_image", flux_inpaint_js)

    def test_flux_inpaint_script_wires_preset_panel(self):
        flux_inpaint_js = (ROOT / "frontend" / "flux" / "inpaint.js").read_text(encoding="utf-8")
        self.assertIn('taskType: "flux.inpaint"', flux_inpaint_js)

    def test_qwen_image_page_includes_lora_panel(self):
        qwen_image_html = (ROOT / "frontend" / "qwen_image" / "text2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=6"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        lightning_settings_tag = '<script src="lightning_settings.js?v=2"></script>'
        controller_tag = '<script src="../generation_page.js?v=3"></script>'
        qwen_image_tag = '<script src="text2img.js?v=8"></script>'

        self.assertIn('id="lora-panel-root"', qwen_image_html)
        self.assertIn(lora_tag, qwen_image_html)
        self.assertIn(preset_tag, qwen_image_html)
        self.assertIn(lightning_settings_tag, qwen_image_html)
        self.assertIn(controller_tag, qwen_image_html)
        self.assertIn(qwen_image_tag, qwen_image_html)
        self.assertLess(qwen_image_html.index(lora_tag), qwen_image_html.index(controller_tag))
        self.assertLess(qwen_image_html.index(lightning_settings_tag), qwen_image_html.index(controller_tag))
        self.assertLess(qwen_image_html.index(preset_tag), qwen_image_html.index(qwen_image_tag))

    def test_qwen_image_script_wires_lora_payload(self):
        qwen_image_js = (ROOT / "frontend" / "qwen_image" / "text2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "qwen-image"', qwen_image_js)
        self.assertIn("loraEnvelope: false", qwen_image_js)
        self.assertIn("page.withLora", qwen_image_js)

    def test_qwen_image_script_wires_preset_panel(self):
        qwen_image_js = (ROOT / "frontend" / "qwen_image" / "text2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "qwen-image.text2img"', qwen_image_js)
        self.assertIn('key: "true_cfg_scale"', qwen_image_js)

    def test_qwen_image_img2img_page_includes_lora_panel(self):
        qwen_image_img2img_html = (ROOT / "frontend" / "qwen_image" / "img2img.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=6"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        lightning_settings_tag = '<script src="lightning_settings.js?v=2"></script>'
        controller_tag = '<script src="../generation_page.js?v=3"></script>'
        qwen_image_img2img_tag = '<script src="img2img.js?v=8"></script>'

        self.assertIn('id="lora-panel-root"', qwen_image_img2img_html)
        self.assertIn(lora_tag, qwen_image_img2img_html)
        self.assertIn(preset_tag, qwen_image_img2img_html)
        self.assertIn(lightning_settings_tag, qwen_image_img2img_html)
        self.assertIn(controller_tag, qwen_image_img2img_html)
        self.assertIn(qwen_image_img2img_tag, qwen_image_img2img_html)
        self.assertLess(
            qwen_image_img2img_html.index(lora_tag),
            qwen_image_img2img_html.index(controller_tag),
        )
        self.assertLess(
            qwen_image_img2img_html.index(lightning_settings_tag),
            qwen_image_img2img_html.index(controller_tag),
        )
        self.assertLess(
            qwen_image_img2img_html.index(preset_tag),
            qwen_image_img2img_html.index(qwen_image_img2img_tag),
        )

    def test_qwen_image_img2img_script_wires_lora_payload(self):
        qwen_image_img2img_js = (ROOT / "frontend" / "qwen_image" / "img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "qwen-image"', qwen_image_img2img_js)
        self.assertIn("loraEnvelope: false", qwen_image_img2img_js)
        self.assertIn("page.withLora", qwen_image_img2img_js)
        self.assertIn("inputs.initial_image", qwen_image_img2img_js)

    def test_qwen_image_img2img_script_wires_preset_panel(self):
        qwen_image_img2img_js = (ROOT / "frontend" / "qwen_image" / "img2img.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "qwen-image.img2img"', qwen_image_img2img_js)
        self.assertIn('key: "true_cfg_scale"', qwen_image_img2img_js)

    def test_qwen_image_inpaint_page_includes_lora_panel(self):
        qwen_image_inpaint_html = (ROOT / "frontend" / "qwen_image" / "inpaint.html").read_text(
            encoding="utf-8"
        )
        lora_tag = '<script src="../components/lora_panel.js?v=6"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        editor_tag = '<script src="../components/inpaint_editor.js?v=1"></script>'
        lightning_settings_tag = '<script src="lightning_settings.js?v=2"></script>'
        controller_tag = '<script src="../generation_page.js?v=3"></script>'
        qwen_image_inpaint_tag = '<script src="inpaint.js?v=9"></script>'

        self.assertIn('id="lora-panel-root"', qwen_image_inpaint_html)
        self.assertIn(lora_tag, qwen_image_inpaint_html)
        self.assertIn(preset_tag, qwen_image_inpaint_html)
        self.assertIn(lightning_settings_tag, qwen_image_inpaint_html)
        self.assertIn(controller_tag, qwen_image_inpaint_html)
        self.assertIn(editor_tag, qwen_image_inpaint_html)
        self.assertIn(qwen_image_inpaint_tag, qwen_image_inpaint_html)
        self.assertLess(
            qwen_image_inpaint_html.index(lora_tag),
            qwen_image_inpaint_html.index(controller_tag),
        )
        self.assertLess(
            qwen_image_inpaint_html.index(lightning_settings_tag),
            qwen_image_inpaint_html.index(controller_tag),
        )
        self.assertLess(
            qwen_image_inpaint_html.index(preset_tag),
            qwen_image_inpaint_html.index(qwen_image_inpaint_tag),
        )

    def test_qwen_image_inpaint_script_wires_lora_payload(self):
        qwen_image_inpaint_js = (ROOT / "frontend" / "qwen_image" / "inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('family: "qwen-image"', qwen_image_inpaint_js)
        self.assertIn("loraEnvelope: false", qwen_image_inpaint_js)
        self.assertIn("page.withLora", qwen_image_inpaint_js)
        self.assertIn("inputs.initial_image", qwen_image_inpaint_js)
        self.assertIn("inputs.mask_image", qwen_image_inpaint_js)

    def test_qwen_image_inpaint_script_wires_preset_panel(self):
        qwen_image_inpaint_js = (ROOT / "frontend" / "qwen_image" / "inpaint.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('taskType: "qwen-image.inpaint"', qwen_image_inpaint_js)
        self.assertIn('key: "true_cfg_scale"', qwen_image_inpaint_js)

    def test_preprocessor_modal_has_two_column_layout_hooks(self):
        preprocessor_html = (ROOT / "frontend" / "components" / "controlnet_preprocessor.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('class="modal-body preprocessor-layout"', preprocessor_html)
        self.assertIn('class="preprocessor-settings"', preprocessor_html)
        self.assertIn('class="preprocessor-preview preprocessor-preview-panel"', preprocessor_html)
        self.assertIn("grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);", preprocessor_html)

    def test_preprocessor_modal_styles_define_viewport_height_preview(self):
        style_css = "\n".join(
            (ROOT / "frontend" / "styles" / name).read_text(encoding="utf-8")
            for name in ("components.css", "responsive.css")
        )
        preprocessor_html = (ROOT / "frontend" / "components" / "controlnet_preprocessor.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("#preprocessor-modal .preprocessor-layout", style_css)
        self.assertIn("#preprocessor-modal .preprocessor-preview-panel img", style_css)
        self.assertIn("max-height: calc(94vh - 220px);", style_css)
        self.assertIn("@media (max-width: 700px)", style_css)
        self.assertIn("max-height: calc(94vh - 220px);", preprocessor_html)


if __name__ == "__main__":
    unittest.main()
