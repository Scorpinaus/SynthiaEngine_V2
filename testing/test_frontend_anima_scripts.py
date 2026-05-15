from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendAnimaScriptTests(unittest.TestCase):
    def test_anima_page_includes_shared_scripts_before_page_script(self):
        html = (ROOT / "frontend" / "anima" / "text2img.html").read_text(
            encoding="utf-8"
        )
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        anima_tag = '<script src="text2img.js?v=1"></script>'

        self.assertIn('<script src="../workflow_client.js?v=1"></script>', html)
        self.assertIn('<script src="../workflow_catalog.js?v=1"></script>', html)
        self.assertIn(preset_tag, html)
        self.assertIn(anima_tag, html)
        self.assertLess(html.index(preset_tag), html.index(anima_tag))

    def test_anima_script_wires_catalog_preset_and_payload(self):
        js = (ROOT / "frontend" / "anima" / "text2img.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('fetch(`${API_BASE}/models?family=anima`)', js)
        self.assertIn('taskType: "anima.text2img"', js)
        self.assertIn("collectAnimaPresetSettings", js)
        self.assertIn("applyAnimaPresetSettings", js)
        self.assertIn('WorkflowCatalog.applyDefaultsToForm("anima.text2img"', js)
        self.assertIn('tasks: [{ id: "t1", type: "anima.text2img", inputs }]', js)
        self.assertIn("memory_preset", js)
        self.assertIn("flowmatch_euler", js)


if __name__ == "__main__":
    unittest.main()
