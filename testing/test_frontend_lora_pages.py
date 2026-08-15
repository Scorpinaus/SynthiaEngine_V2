from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendLoraPageTests(unittest.TestCase):
    def test_lora_models_page_includes_expected_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "model_page.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="model_page.js?v=1"></script>', html)
        self.assertIn('href="add.html"', html)

    def test_lora_add_page_includes_expected_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "add.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="add.js?v=1"></script>', html)
        self.assertIn('href="model_page.html"', html)
        self.assertIn('<select name="lora_model_family" required>', html)
        self.assertIn('<option value="sd15">sd15</option>', html)
        self.assertIn('<option value="sdxl">sdxl</option>', html)
        self.assertIn('<option value="flux">flux</option>', html)
        self.assertIn('<option value="qwen-image">qwen-image</option>', html)
        self.assertIn('<option value="z-image">z-image</option>', html)
        self.assertIn('<select name="lora_type" required>', html)
        self.assertIn('<option value="lora">lora</option>', html)
        self.assertIn('<option value="lycoris">lycoris</option>', html)
        self.assertIn('<option value="lokr">lokr</option>', html)
        self.assertIn('id="local-file-panel"', html)
        self.assertIn('id="select-local-file"', html)
        self.assertIn('id="web-file-panel"', html)
        self.assertIn('id="web-file-input"', html)
        self.assertIn('name="file_path" type="text" placeholder="Select a local file or enter a web link" readonly required', html)

    def test_lora_edit_page_includes_expected_scripts(self):
        html = (ROOT / "frontend" / "models" / "lora" / "edit.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="edit.js?v=1"></script>', html)
        self.assertIn('href="model_page.html"', html)
        self.assertIn('<select name="lora_model_family" required>', html)
        self.assertIn('<option value="sd15">sd15</option>', html)
        self.assertIn('<option value="sdxl">sdxl</option>', html)
        self.assertIn('<option value="flux">flux</option>', html)
        self.assertIn('<option value="qwen-image">qwen-image</option>', html)
        self.assertIn('<option value="z-image">z-image</option>', html)
        self.assertIn('<select name="lora_type" required>', html)
        self.assertIn('<option value="lora">lora</option>', html)
        self.assertIn('<option value="lycoris">lycoris</option>', html)
        self.assertIn('<option value="lokr">lokr</option>', html)
        self.assertIn('id="local-file-panel"', html)
        self.assertIn('id="select-local-file"', html)
        self.assertIn('id="web-file-panel"', html)
        self.assertIn('id="web-file-input"', html)
        self.assertIn('name="file_path" type="text" placeholder="Select a local file or enter a web link" readonly required', html)

    def test_lora_models_script_calls_list_and_delete_endpoints(self):
        js = (ROOT / "frontend" / "models" / "lora" / "model_page.js").read_text(encoding="utf-8")
        self.assertIn("window.confirm(", js)
        self.assertIn("method: \"DELETE\"", js)
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("${API_BASE}/lora-models/${encodeURIComponent(String(entry.lora_id))}", js)
        self.assertIn("edit.html?lora_id=", js)

    def test_lora_add_script_calls_create_endpoint(self):
        js = (ROOT / "frontend" / "models" / "lora" / "add.js").read_text(encoding="utf-8")
        self.assertIn("method: \"POST\"", js)
        self.assertIn("${API_BASE}/lora-models", js)
        self.assertIn("LoRA saved successfully.", js)
        self.assertIn("/api/local-path/select", js)
        self.assertIn('selection_type: "file"', js)
        self.assertIn("syncFilePathMode", js)

    def test_lora_add_page_serializes_lightning_metadata_and_hub_coordinates(self):
        html = (ROOT / "frontend" / "models" / "lora" / "add.html").read_text(encoding="utf-8")
        js = (ROOT / "frontend" / "models" / "lora" / "add.js").read_text(encoding="utf-8")
        self.assertIn('id="adapter-use"', html)
        self.assertIn('value="qwen_image_lightning"', html)
        self.assertIn('id="lightning-profile-panel"', html)
        self.assertIn('id="lightning-steps"', html)
        self.assertIn('id="hub-coordinates-panel"', html)
        self.assertIn('id="weight-name-field"', html)
        self.assertIn("function buildRuntimeProfile()", js)
        self.assertIn('kind: "qwen_image_lightning"', js)
        self.assertIn('supported_tasks: ["text2img", "img2img", "inpaint"]', js)
        self.assertIn("runtime_profile: buildRuntimeProfile()", js)
        self.assertIn('supported_tasks: ["text2img", "img2img", "inpaint"]', js)
        self.assertIn("weight_name: isHub ? weightNameField?.value.trim() || null : null", js)
        self.assertIn("syncAdapterUse", js)

    def test_lora_edit_script_calls_detail_and_patch_endpoints(self):
        js = (ROOT / "frontend" / "models" / "lora" / "edit.js").read_text(encoding="utf-8")
        self.assertIn("new URLSearchParams(window.location.search)", js)
        self.assertIn("${API_BASE}/lora-models/${encodeURIComponent(String(loraId))}", js)
        self.assertIn("method: \"PATCH\"", js)
        self.assertIn("LoRA entry saved successfully.", js)
        self.assertIn("/api/local-path/select", js)
        self.assertIn('selection_type: "file"', js)
        self.assertIn("syncFilePathMode", js)
        self.assertIn('select[name="lora_model_family"]', js)
        self.assertIn('select[name="lora_type"]', js)

    def test_lora_edit_page_hydrates_lightning_metadata_and_hub_coordinates(self):
        html = (ROOT / "frontend" / "models" / "lora" / "edit.html").read_text(encoding="utf-8")
        js = (ROOT / "frontend" / "models" / "lora" / "edit.js").read_text(encoding="utf-8")
        self.assertIn('id="adapter-use"', html)
        self.assertIn('id="lightning-profile-panel"', html)
        self.assertIn('id="hub-coordinates-panel"', html)
        self.assertIn('id="revision-field"', html)
        self.assertIn('entry.runtime_profile?.kind === "qwen_image_lightning"', js)
        self.assertIn("lightningStepsField.value = String(entry.runtime_profile?.steps || 4)", js)
        self.assertIn("weightNameField.value = entry.weight_name || \"\"", js)
        self.assertIn("subfolderField.value = entry.subfolder || \"\"", js)
        self.assertIn("revisionField.value = entry.revision || \"\"", js)
        self.assertIn("runtime_profile: buildRuntimeProfile()", js)

    def test_nav_includes_separate_base_and_lora_links(self):
        nav_js = (ROOT / "frontend" / "components" / "nav_bar.js").read_text(encoding="utf-8")
        self.assertIn('{ href: "models/base/registry.html", label: "Base Models" }', nav_js)
        self.assertIn('{ href: "models/lora/model_page.html", label: "LoRA Models" }', nav_js)

    def test_base_model_pages_remain_wired(self):
        models_html = (ROOT / "frontend" / "models" / "base" / "registry.html").read_text(encoding="utf-8")
        base_add_html = (ROOT / "frontend" / "models" / "base" / "add.html").read_text(encoding="utf-8")
        self.assertIn('<script src="registry.js?v=1"></script>', models_html)
        self.assertIn('<script src="add.js?v=1"></script>', base_add_html)
        self.assertIn('href="add.html"', models_html)
        self.assertIn('href="registry.html"', base_add_html)


if __name__ == "__main__":
    unittest.main()
