from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendWanScriptTests(unittest.TestCase):
    def test_wan_page_includes_expected_scripts_in_order(self):
        html = (ROOT / "frontend" / "wan" / "text2video.html").read_text(encoding="utf-8")
        viewer_tag = '<script src="../components/video_gallery.js?v=1"></script>'
        validator_tag = '<script src="../workflow_input_validator.js?v=1"></script>'
        wan_tag = '<script src="text2video.js?v=1"></script>'

        self.assertIn(viewer_tag, html)
        self.assertIn(validator_tag, html)
        self.assertIn(wan_tag, html)
        self.assertLess(html.index(viewer_tag), html.index(wan_tag))
        self.assertLess(html.index(validator_tag), html.index(wan_tag))

    def test_wan_script_wires_expected_task_and_payload(self):
        js = (ROOT / "frontend" / "wan" / "text2video.js").read_text(encoding="utf-8")

        self.assertIn('const TASK_WAN_TEXT2VIDEO = "wan.text2video";', js)
        self.assertIn('model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"', js)
        self.assertIn("width: 832", js)
        self.assertIn("height: 480", js)
        self.assertIn("memory_preset: \"safe\"", js)
        self.assertIn("return: \"@t1.videos\"", js)
        self.assertIn("WorkflowInputValidator?.assertTaskInputs", js)
        self.assertIn("WorkflowClient.submitWorkflow", js)
        self.assertIn("videoGallery.setVideos", js)

    def test_wan_page_exposes_frame_options_and_fixed_480p_controls(self):
        html = (ROOT / "frontend" / "wan" / "text2video.html").read_text(encoding="utf-8")

        self.assertIn('id="num_frames"', html)
        self.assertIn('<option value="33">33</option>', html)
        self.assertIn('<option value="49" selected>49</option>', html)
        self.assertIn('<option value="81">81</option>', html)
        self.assertIn('id="width"', html)
        self.assertIn('id="height"', html)
        self.assertIn('id="memory_preset"', html)

    def test_nav_bar_links_wan_page(self):
        js = (ROOT / "frontend" / "components" / "nav_bar.js").read_text(encoding="utf-8")

        self.assertIn('href: "wan/text2video.html"', js)
        self.assertIn('label: "WAN Text2Video"', js)


if __name__ == "__main__":
    unittest.main()
