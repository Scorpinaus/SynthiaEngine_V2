from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendAnimaScriptTests(unittest.TestCase):
    def test_anima_page_includes_shared_scripts_before_page_script(self):
        html = (ROOT / "frontend" / "anima" / "text2img.html").read_text(
            encoding="utf-8"
        )
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        anima_tag = '<script src="text2img.js?v=3"></script>'

        self.assertIn('<script src="../workflow_client.js?v=1"></script>', html)
        self.assertIn('<script src="../workflow_catalog.js?v=1"></script>', html)
        self.assertIn(preset_tag, html)
        self.assertIn(controller_tag, html)
        self.assertIn(anima_tag, html)
        self.assertLess(html.index(controller_tag), html.index(anima_tag))

    def test_anima_script_wires_catalog_preset_and_payload(self):
        js = (ROOT / "frontend" / "anima" / "text2img.js").read_text(
            encoding="utf-8"
        )

        controller = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")

        self.assertIn('family: "anima"', js)
        self.assertIn('taskType: "anima.text2img"', js)
        self.assertIn("lora: false", js)
        self.assertIn("page.collectSettings", js)
        self.assertIn("config.taskType", controller)
        self.assertIn('return: "@t1.images"', controller)
        self.assertIn("memory_preset", js)
        self.assertIn("flowmatch_euler", js)


if __name__ == "__main__":
    unittest.main()
