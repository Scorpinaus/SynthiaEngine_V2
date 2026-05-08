from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendErnieImageScriptTests(unittest.TestCase):
    def test_ernie_image_page_includes_shared_scripts_before_page_script(self):
        html = (ROOT / "frontend" / "ernie_image" / "text2img.html").read_text(
            encoding="utf-8"
        )
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        ernie_tag = '<script src="text2img.js?v=1"></script>'

        self.assertIn('<script src="../workflow_client.js?v=1"></script>', html)
        self.assertIn('<script src="../workflow_catalog.js?v=1"></script>', html)
        self.assertIn(preset_tag, html)
        self.assertIn(ernie_tag, html)
        self.assertLess(html.index(preset_tag), html.index(ernie_tag))

    def test_ernie_image_script_wires_catalog_preset_and_payload(self):
        js = (ROOT / "frontend" / "ernie_image" / "text2img.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('fetch(`${API_BASE}/models?family=ernie-image`)', js)
        self.assertIn('taskType: "ernie-image.text2img"', js)
        self.assertIn("collectErnieImagePresetSettings", js)
        self.assertIn("applyErnieImagePresetSettings", js)
        self.assertIn('WorkflowCatalog.applyDefaultsToForm("ernie-image.text2img"', js)
        self.assertIn('tasks: [{ id: "t1", type: "ernie-image.text2img", inputs }]', js)
        self.assertIn("use_pe", js)
        self.assertIn("load_pe", js)
        self.assertIn("execution_mode", js)
        self.assertIn("memory_preset", js)


if __name__ == "__main__":
    unittest.main()
