from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendLoraRegistryScriptTests(unittest.TestCase):
    def test_lora_models_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "lora_models.html").read_text(encoding="utf-8")
        self.assertIn('<script src="api_config.js?v=1"></script>', html)
        self.assertIn('<script src="header.js?v=1"></script>', html)
        self.assertIn('<script src="nav_bar.js?v=2"></script>', html)
        self.assertIn('<script src="lora_models.js?v=1"></script>', html)

    def test_models_page_links_to_lora_registry_only_for_lora_actions(self):
        html = (ROOT / "frontend" / "models.html").read_text(encoding="utf-8")
        self.assertIn('href="lora_models.html"', html)
        self.assertNotIn('href="lora_add.html"', html)

    def test_lora_add_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "lora_add.html").read_text(encoding="utf-8")
        self.assertIn('<script src="api_config.js?v=1"></script>', html)
        self.assertIn('<script src="header.js?v=1"></script>', html)
        self.assertIn('<script src="nav_bar.js?v=2"></script>', html)
        self.assertIn('<script src="lora_add.js?v=1"></script>', html)

    def test_lora_edit_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "lora_edit.html").read_text(encoding="utf-8")
        self.assertIn('<script src="api_config.js?v=1"></script>', html)
        self.assertIn('<script src="header.js?v=1"></script>', html)
        self.assertIn('<script src="nav_bar.js?v=2"></script>', html)
        self.assertIn('<script src="lora_edit.js?v=1"></script>', html)

    def test_lora_models_script_uses_list_filter_and_delete_endpoints(self):
        js = (ROOT / "frontend" / "lora_models.js").read_text(encoding="utf-8")
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("window.confirm(", js)
        self.assertIn("method: \"DELETE\"", js)
        self.assertIn("lora_edit.html?lora_id=", js)

    def test_lora_add_script_uses_create_endpoint(self):
        js = (ROOT / "frontend" / "lora_add.js").read_text(encoding="utf-8")
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("method: \"POST\"", js)
        self.assertIn("LoRA saved successfully.", js)

    def test_lora_edit_script_uses_get_and_patch_endpoints(self):
        js = (ROOT / "frontend" / "lora_edit.js").read_text(encoding="utf-8")
        self.assertIn("new URLSearchParams(window.location.search)", js)
        self.assertIn("${API_BASE}/lora-models/${encodeURIComponent(String(loraId))}", js)
        self.assertIn("method: \"PATCH\"", js)
        self.assertIn("LoRA entry saved successfully.", js)


if __name__ == "__main__":
    unittest.main()
