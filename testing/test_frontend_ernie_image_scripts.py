from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendErnieImageScriptTests(unittest.TestCase):
    def test_ernie_image_page_includes_shared_scripts_before_page_script(self):
        html = (ROOT / "frontend" / "ernie_image" / "text2img.html").read_text(
            encoding="utf-8"
        )
        preset_tag = '<script src="../components/preset_panel.js?v=1"></script>'
        controller_tag = '<script src="../generation_page.js?v=1"></script>'
        ernie_tag = '<script src="text2img.js?v=3"></script>'

        self.assertIn('<script src="../workflow_client.js?v=1"></script>', html)
        self.assertIn('<script src="../workflow_catalog.js?v=1"></script>', html)
        self.assertIn('<script src="../components/lora_panel.js?v=2"></script>', html)
        self.assertIn('<script src="../components/adapter_panel.js?v=1"></script>', html)
        self.assertIn('<div id="adapter-panel-root"></div>', html)
        self.assertIn(preset_tag, html)
        self.assertIn(controller_tag, html)
        self.assertIn(ernie_tag, html)
        self.assertLess(html.index(controller_tag), html.index(ernie_tag))

    def test_ernie_image_script_wires_catalog_preset_and_payload(self):
        js = (ROOT / "frontend" / "ernie_image" / "text2img.js").read_text(
            encoding="utf-8"
        )

        controller = (ROOT / "frontend" / "generation_page.js").read_text(encoding="utf-8")

        self.assertIn('family: "ernie-image"', js)
        self.assertIn('taskType: "ernie-image.text2img"', js)
        self.assertIn("page.collectSettings", js)
        self.assertIn("config.taskType", controller)
        self.assertIn('return: "@t1.images"', controller)
        self.assertIn("use_pe", js)
        self.assertIn("load_pe", js)
        self.assertIn("memory_preset", js)
        self.assertIn("negative_prompt", js)
        self.assertIn('window.AdapterPanel?.render?.()', js)
        self.assertIn("page.withLora", js)
        self.assertIn("loraEnvelope: false", js)
        self.assertIn("window.LoraPanel?.init", controller)
        self.assertIn("inputs.lora_adapters", controller)


if __name__ == "__main__":
    unittest.main()
