from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendSd15AnimateDiffScriptTests(unittest.TestCase):
    def test_animatediff_page_includes_expected_scripts_in_order(self):
        html = (ROOT / "frontend" / "sd15" / "animatediff.html").read_text(encoding="utf-8")
        viewer_tag = '<script src="../components/video_gallery.js?v=1"></script>'
        lora_tag = '<script src="../components/lora_panel.js?v=1"></script>'
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        validator_tag = '<script src="../workflow_input_validator.js?v=1"></script>'
        generation_tag = '<script src="../generation_page.js?v=4"></script>'
        controller_tag = '<script src="../components/animatediff_controller.js?v=1"></script>'
        animatediff_tag = '<script src="animatediff.js?v=4"></script>'

        self.assertIn(viewer_tag, html)
        self.assertIn(lora_tag, html)
        self.assertIn(preset_tag, html)
        self.assertIn(validator_tag, html)
        self.assertIn(generation_tag, html)
        self.assertIn(controller_tag, html)
        self.assertIn(animatediff_tag, html)
        self.assertLess(html.index(viewer_tag), html.index(animatediff_tag))
        self.assertLess(html.index(lora_tag), html.index(animatediff_tag))
        self.assertLess(html.index(preset_tag), html.index(animatediff_tag))
        self.assertLess(html.index(validator_tag), html.index(animatediff_tag))
        self.assertLess(html.index(generation_tag), html.index(controller_tag))
        self.assertLess(html.index(controller_tag), html.index(animatediff_tag))

    def test_animatediff_script_wires_expected_task_and_payload(self):
        entrypoint = (ROOT / "frontend" / "sd15" / "animatediff.js").read_text(
            encoding="utf-8"
        )
        controller = (ROOT / "frontend" / "components" / "animatediff_controller.js").read_text(
            encoding="utf-8"
        )
        generation_page = (ROOT / "frontend" / "generation_page.js").read_text(
            encoding="utf-8"
        )
        js = "\n".join((entrypoint, controller, generation_page))

        self.assertIn('const TASK_ANIMATEDIFF_TEXT2VIDEO = "sd15.animatediff.text2video";', js)
        self.assertIn("AnimateDiffController.create", entrypoint)
        self.assertIn("window.AnimateDiffController", controller)
        self.assertIn("form.initPresets()", controller)
        self.assertIn("form.initLora()", controller)
        self.assertIn("window.LoraPanel?.getSelectedAdapters?.() ?? []", js)
        self.assertIn("return: \"@t1.videos\"", js)
        self.assertIn("GenerationPage.validateTasks(tasks)", controller)
        self.assertIn("free_noise_enabled", js)
        self.assertIn("free_noise_context_length", js)
        self.assertIn("free_noise_context_stride", js)
        self.assertIn("free_init_enabled", js)
        self.assertIn("free_init_num_iters", js)
        self.assertIn("free_init_use_fast_sampling", js)
        self.assertIn("free_init_method", js)
        self.assertIn("free_init_order", js)
        self.assertIn("free_init_spatial_stop_frequency", js)
        self.assertIn("free_init_temporal_stop_frequency", js)

    def test_animatediff_page_includes_free_noise_controls(self):
        html = (ROOT / "frontend" / "sd15" / "animatediff.html").read_text(encoding="utf-8")

        self.assertIn('id="free_noise_enabled"', html)
        self.assertIn('id="free_noise_context_length"', html)
        self.assertIn('id="free_noise_context_stride"', html)

    def test_animatediff_page_includes_free_init_controls(self):
        html = (ROOT / "frontend" / "sd15" / "animatediff.html").read_text(encoding="utf-8")

        self.assertIn('id="free_init_enabled"', html)
        self.assertIn('id="free_init_num_iters"', html)
        self.assertIn('id="free_init_use_fast_sampling"', html)
        self.assertIn('id="free_init_method"', html)
        self.assertIn('id="free_init_order"', html)
        self.assertIn('id="free_init_spatial_stop_frequency"', html)
        self.assertIn('id="free_init_temporal_stop_frequency"', html)

    def test_video_gallery_exposes_expected_api(self):
        js = (ROOT / "frontend" / "components" / "video_gallery.js").read_text(encoding="utf-8")

        self.assertIn("function createVideoGalleryViewer", js)
        self.assertIn("setVideos(videos)", js)
        self.assertIn("viewerVideo.load()", js)

    def test_nav_bar_links_animatediff_page(self):
        js = (ROOT / "frontend" / "components" / "nav_bar.js").read_text(encoding="utf-8")

        self.assertIn('href: "sd15/animatediff.html"', js)
        self.assertIn('label: "SD 1.5 AnimateDiff"', js)

    def test_style_supports_video_viewer(self):
        css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

        self.assertIn(".viewer-frame video", css)


if __name__ == "__main__":
    unittest.main()
