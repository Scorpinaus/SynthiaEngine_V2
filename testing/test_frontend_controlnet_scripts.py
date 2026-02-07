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
