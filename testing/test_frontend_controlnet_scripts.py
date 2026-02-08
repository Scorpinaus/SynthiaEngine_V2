from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendControlNetScriptTests(unittest.TestCase):
    def test_sd15_page_includes_controlnet_scripts_before_sd15(self):
        sd15_html = (ROOT / "frontend" / "sd15.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        sd15_tag = '<script src="sd15.js?v=3"></script>'

        self.assertIn(panel_tag, sd15_html)
        self.assertIn(preprocessor_tag, sd15_html)
        self.assertIn(sd15_tag, sd15_html)
        self.assertLess(sd15_html.index(panel_tag), sd15_html.index(sd15_tag))
        self.assertLess(sd15_html.index(preprocessor_tag), sd15_html.index(sd15_tag))

    def test_sd15_img2img_page_includes_controlnet_scripts_before_img2img(self):
        sd15_img2img_html = (ROOT / "frontend" / "sd15_img2img.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        img2img_tag = '<script src="sd15_img2img.js?v=3"></script>'

        self.assertIn(panel_tag, sd15_img2img_html)
        self.assertIn(preprocessor_tag, sd15_img2img_html)
        self.assertIn(img2img_tag, sd15_img2img_html)
        self.assertLess(sd15_img2img_html.index(panel_tag), sd15_img2img_html.index(img2img_tag))
        self.assertLess(
            sd15_img2img_html.index(preprocessor_tag), sd15_img2img_html.index(img2img_tag)
        )

    def test_sd15_inpaint_page_includes_controlnet_scripts_before_inpaint(self):
        sd15_inpaint_html = (ROOT / "frontend" / "sd15_inpainting.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        inpaint_tag = '<script src="sd15_inpainting.js?v=3"></script>'

        self.assertIn(panel_tag, sd15_inpaint_html)
        self.assertIn(preprocessor_tag, sd15_inpaint_html)
        self.assertIn(inpaint_tag, sd15_inpaint_html)
        self.assertLess(sd15_inpaint_html.index(panel_tag), sd15_inpaint_html.index(inpaint_tag))
        self.assertLess(
            sd15_inpaint_html.index(preprocessor_tag), sd15_inpaint_html.index(inpaint_tag)
        )

    def test_sdxl_page_includes_controlnet_scripts_before_sdxl(self):
        sdxl_html = (ROOT / "frontend" / "sdxl.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        sdxl_tag = '<script src="sdxl.js?v=5"></script>'

        self.assertIn(panel_tag, sdxl_html)
        self.assertIn(preprocessor_tag, sdxl_html)
        self.assertIn(sdxl_tag, sdxl_html)
        self.assertLess(sdxl_html.index(panel_tag), sdxl_html.index(sdxl_tag))
        self.assertLess(sdxl_html.index(preprocessor_tag), sdxl_html.index(sdxl_tag))

    def test_sdxl_img2img_page_includes_controlnet_scripts_before_sdxl_img2img(self):
        sdxl_img2img_html = (ROOT / "frontend" / "sdxl_img2img.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        sdxl_img2img_tag = '<script src="sdxl_img2img.js?v=3"></script>'

        self.assertIn(panel_tag, sdxl_img2img_html)
        self.assertIn(preprocessor_tag, sdxl_img2img_html)
        self.assertIn(sdxl_img2img_tag, sdxl_img2img_html)
        self.assertLess(sdxl_img2img_html.index(panel_tag), sdxl_img2img_html.index(sdxl_img2img_tag))
        self.assertLess(sdxl_img2img_html.index(preprocessor_tag), sdxl_img2img_html.index(sdxl_img2img_tag))

    def test_sdxl_inpaint_page_includes_controlnet_scripts_before_sdxl_inpaint(self):
        sdxl_inpaint_html = (ROOT / "frontend" / "sdxl_inpaint.html").read_text(encoding="utf-8")
        panel_tag = '<script src="controlnet_panel.js?v=2"></script>'
        preprocessor_tag = '<script src="controlnet_preprocessor.js?v=3"></script>'
        sdxl_inpaint_tag = '<script src="sdxl_inpaint.js?v=3"></script>'

        self.assertIn(panel_tag, sdxl_inpaint_html)
        self.assertIn(preprocessor_tag, sdxl_inpaint_html)
        self.assertIn(sdxl_inpaint_tag, sdxl_inpaint_html)
        self.assertLess(sdxl_inpaint_html.index(panel_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))
        self.assertLess(sdxl_inpaint_html.index(preprocessor_tag), sdxl_inpaint_html.index(sdxl_inpaint_tag))

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

    def test_sd15_img2img_script_consumes_controlnet_state(self):
        img2img_js = (ROOT / "frontend" / "sd15_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", img2img_js)
        self.assertIn("controlnetEnabled", img2img_js)
        self.assertIn("control_images", img2img_js)
        self.assertIn("controlnet_models", img2img_js)

    def test_sd15_inpaint_script_consumes_controlnet_state(self):
        inpaint_js = (ROOT / "frontend" / "sd15_inpainting.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", inpaint_js)
        self.assertIn("controlnetEnabled", inpaint_js)
        self.assertIn("control_images", inpaint_js)
        self.assertIn("controlnet_models", inpaint_js)

    def test_sdxl_script_consumes_controlnet_state(self):
        sdxl_js = (ROOT / "frontend" / "sdxl.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_js)
        self.assertIn("controlnetEnabled", sdxl_js)
        self.assertIn("control_images", sdxl_js)
        self.assertIn("controlnet_models", sdxl_js)

    def test_sdxl_img2img_script_consumes_controlnet_state(self):
        sdxl_img2img_js = (ROOT / "frontend" / "sdxl_img2img.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_img2img_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_img2img_js)
        self.assertIn("controlnetEnabled", sdxl_img2img_js)
        self.assertIn("control_images", sdxl_img2img_js)
        self.assertIn("controlnet_models", sdxl_img2img_js)

    def test_sdxl_inpaint_script_consumes_controlnet_state(self):
        sdxl_inpaint_js = (ROOT / "frontend" / "sdxl_inpaint.js").read_text(encoding="utf-8")
        self.assertIn("window.ControlNetPanel?.getState?.()", sdxl_inpaint_js)
        self.assertIn("window.ControlNetPreprocessor.init()", sdxl_inpaint_js)
        self.assertIn("controlnetEnabled", sdxl_inpaint_js)
        self.assertIn("control_images", sdxl_inpaint_js)
        self.assertIn("controlnet_models", sdxl_inpaint_js)

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
