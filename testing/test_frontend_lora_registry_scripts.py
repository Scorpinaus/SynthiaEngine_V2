from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendLoraRegistryScriptTests(unittest.TestCase):
    def test_lora_models_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "model_page.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="model_page.js?v=1"></script>', html)

    def test_models_page_links_to_lora_registry_only_for_lora_actions(self):
        html = (ROOT / "frontend" / "models" / "base" / "registry.html").read_text(encoding="utf-8")
        self.assertIn('href="../lora/model_page.html"', html)
        self.assertNotIn('href="../lora/add.html"', html)

    def test_lora_add_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "add.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="add.js?v=1"></script>', html)
        self.assertIn('id="select-local-file"', html)
        self.assertIn('id="web-file-input"', html)
        self.assertIn('name="file_path" type="text" placeholder="Select a local file or enter a web link" readonly required', html)

    def test_lora_edit_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "edit.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="edit.js?v=1"></script>', html)
        self.assertIn('id="select-local-file"', html)
        self.assertIn('id="web-file-input"', html)
        self.assertIn('name="file_path" type="text" placeholder="Select a local file or enter a web link" readonly required', html)

    def test_lora_models_script_uses_list_filter_and_delete_endpoints(self):
        js = (ROOT / "frontend" / "models" / "lora" / "model_page.js").read_text(encoding="utf-8")
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("window.confirm(", js)
        self.assertIn("method: \"DELETE\"", js)
        self.assertIn("edit.html?lora_id=", js)

    def test_lora_add_script_uses_create_endpoint(self):
        js = (ROOT / "frontend" / "models" / "lora" / "add.js").read_text(encoding="utf-8")
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("method: \"POST\"", js)
        self.assertIn("LoRA saved successfully.", js)
        self.assertIn("/api/local-path/select", js)
        self.assertIn('selection_type: "file"', js)

    def test_lora_edit_script_uses_get_and_patch_endpoints(self):
        js = (ROOT / "frontend" / "models" / "lora" / "edit.js").read_text(encoding="utf-8")
        self.assertIn("new URLSearchParams(window.location.search)", js)
        self.assertIn("${API_BASE}/lora-models/${encodeURIComponent(String(loraId))}", js)
        self.assertIn("method: \"PATCH\"", js)
        self.assertIn("LoRA entry saved successfully.", js)
        self.assertIn("/api/local-path/select", js)
        self.assertIn('selection_type: "file"', js)


if __name__ == "__main__":
    unittest.main()
