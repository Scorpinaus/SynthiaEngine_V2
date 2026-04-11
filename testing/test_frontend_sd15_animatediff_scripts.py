from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendSd15AnimateDiffScriptTests(unittest.TestCase):
    def test_animatediff_page_includes_expected_scripts_in_order(self):
        html = (ROOT / "frontend" / "sd15_animatediff.html").read_text(encoding="utf-8")
        viewer_tag = '<script src="video_gallery.js?v=1"></script>'
        lora_tag = '<script src="lora_panel.js?v=1"></script>'
        preset_tag = '<script src="preset_panel.js?v=1"></script>'
        validator_tag = '<script src="workflow_input_validator.js?v=1"></script>'
        animatediff_tag = '<script src="sd15_animatediff.js?v=1"></script>'

        self.assertIn(viewer_tag, html)
        self.assertIn(lora_tag, html)
        self.assertIn(preset_tag, html)
        self.assertIn(validator_tag, html)
        self.assertIn(animatediff_tag, html)
        self.assertLess(html.index(viewer_tag), html.index(animatediff_tag))
        self.assertLess(html.index(lora_tag), html.index(animatediff_tag))
        self.assertLess(html.index(preset_tag), html.index(animatediff_tag))
        self.assertLess(html.index(validator_tag), html.index(animatediff_tag))

    def test_animatediff_script_wires_expected_task_and_payload(self):
        js = (ROOT / "frontend" / "sd15_animatediff.js").read_text(encoding="utf-8")

        self.assertIn('const TASK_ANIMATEDIFF_TEXT2VIDEO = "sd15.animatediff.text2video";', js)
        self.assertIn("window.PresetPanel?.init({", js)
        self.assertIn("collectSettings: collectAnimateDiffPresetSettings", js)
        self.assertIn("applySettings: applyAnimateDiffPresetSettings", js)
        self.assertIn('window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" })', js)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", js)
        self.assertIn("return: \"@t1.videos\"", js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", js)
        self.assertIn("free_noise_enabled", js)
        self.assertIn("free_noise_context_length", js)
        self.assertIn("free_noise_context_stride", js)

    def test_animatediff_page_includes_free_noise_controls(self):
        html = (ROOT / "frontend" / "sd15_animatediff.html").read_text(encoding="utf-8")

        self.assertIn('id="free_noise_enabled"', html)
        self.assertIn('id="free_noise_context_length"', html)
        self.assertIn('id="free_noise_context_stride"', html)

    def test_video_gallery_exposes_expected_api(self):
        js = (ROOT / "frontend" / "video_gallery.js").read_text(encoding="utf-8")

        self.assertIn("function createVideoGalleryViewer", js)
        self.assertIn("setVideos(videos)", js)
        self.assertIn("viewerVideo.load()", js)

    def test_nav_bar_links_animatediff_page(self):
        js = (ROOT / "frontend" / "nav_bar.js").read_text(encoding="utf-8")

        self.assertIn('href: "sd15_animatediff.html"', js)
        self.assertIn('label: "SD 1.5 AnimateDiff"', js)

    def test_style_supports_video_viewer(self):
        css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

        self.assertIn(".viewer-frame video", css)


if __name__ == "__main__":
    unittest.main()
