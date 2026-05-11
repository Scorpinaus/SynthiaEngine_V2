from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendModelRegistryScriptTests(unittest.TestCase):
    def test_model_edit_page_wires_required_scripts(self):
        html = (ROOT / "frontend" / "models" / "base" / "edit.html").read_text(encoding="utf-8")
        self.assertIn('<script src="../../api_config.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/header.js?v=1"></script>', html)
        self.assertIn('<script src="../../components/nav_bar.js?v=3"></script>', html)
        self.assertIn('<script src="edit.js?v=1"></script>', html)
        self.assertIn('<select name="family" required>', html)
        self.assertIn('<option value="sd15">sd15</option>', html)
        self.assertIn('<option value="sdxl">sdxl</option>', html)
        self.assertIn('<option value="flux">flux</option>', html)
        self.assertIn('<option value="qwen-image">qwen-image</option>', html)
        self.assertIn('<option value="z-image">z-image</option>', html)

    def test_model_add_page_family_uses_fixed_dropdown_options(self):
        html = (ROOT / "frontend" / "models" / "base" / "add.html").read_text(encoding="utf-8")
        self.assertIn('<select name="family" required>', html)
        self.assertIn('<option value="sd15">sd15</option>', html)
        self.assertIn('<option value="sdxl">sdxl</option>', html)
        self.assertIn('<option value="flux">flux</option>', html)
        self.assertIn('<option value="qwen-image">qwen-image</option>', html)
        self.assertIn('<option value="z-image">z-image</option>', html)

    def test_model_add_page_uses_location_type_specific_link_controls(self):
        html = (ROOT / "frontend" / "models" / "base" / "add.html").read_text(encoding="utf-8")
        js = (ROOT / "frontend" / "models" / "base" / "add.js").read_text(encoding="utf-8")
        self.assertIn('<select name="location_type" required>', html)
        self.assertIn('<option value="local">Local</option>', html)
        self.assertIn('<option value="hub">Hub</option>', html)
        self.assertIn('id="select-local-link"', html)
        self.assertIn('id="web-link-input"', html)
        self.assertIn('input name="link" type="text"', html)
        self.assertIn("readonly required", html)
        self.assertIn("/api/local-path/select", js)
        self.assertIn('locationTypeField?.value === "hub"', js)
        self.assertIn("linkField.value = webLinkInput.value.trim()", js)

    def test_models_page_keeps_base_add_and_lora_registry_buttons(self):
        html = (ROOT / "frontend" / "models" / "base" / "registry.html").read_text(encoding="utf-8")
        self.assertIn('href="add.html"', html)
        self.assertIn('href="../lora/model_page.html"', html)
        self.assertNotIn('href="../lora/add.html"', html)

    def test_models_script_uses_edit_and_delete_flows(self):
        js = (ROOT / "frontend" / "models" / "base" / "registry.js").read_text(encoding="utf-8")
        self.assertIn("edit.html?name=", js)
        self.assertIn("window.confirm(", js)
        self.assertIn("method: \"DELETE\"", js)
        self.assertIn("${API_BASE}/models/${encodeURIComponent(model.name)}", js)

    def test_models_script_only_builds_huggingface_links_for_hub_models(self):
        js = (ROOT / "frontend" / "models" / "base" / "registry.js").read_text(encoding="utf-8")
        self.assertIn('if (model.location_type !== "hub")', js)
        self.assertIn("return buildCode(link);", js)
        self.assertIn("https://huggingface.co/${link}", js)
        self.assertIn('return model.location_type === "hub" ? "Link" : "File Path";', js)
        self.assertIn("buildDetailRow(getLinkLabel(model), buildLink(model), { stacked: true })", js)

    def test_model_edit_script_uses_get_and_patch_endpoints(self):
        js = (ROOT / "frontend" / "models" / "base" / "edit.js").read_text(encoding="utf-8")
        self.assertIn("new URLSearchParams(window.location.search)", js)
        self.assertIn("${API_BASE}/models/${encodeURIComponent(modelName)}", js)
        self.assertIn("${API_BASE}/models/${encodeURIComponent(currentModelName)}", js)
        self.assertIn("method: \"PATCH\"", js)
        self.assertIn("Model entry saved successfully.", js)

    def test_model_edit_page_uses_location_type_specific_link_controls(self):
        html = (ROOT / "frontend" / "models" / "base" / "edit.html").read_text(encoding="utf-8")
        js = (ROOT / "frontend" / "models" / "base" / "edit.js").read_text(encoding="utf-8")
        self.assertIn('<select name="location_type" required>', html)
        self.assertIn('id="select-local-link"', html)
        self.assertIn('id="web-link-input"', html)
        self.assertIn('input name="link" type="text"', html)
        self.assertIn("readonly required", html)
        self.assertIn('select[name="family"]', js)
        self.assertIn("/api/local-path/select", js)
        self.assertIn("syncLinkMode({ preserveLink: true })", js)
        self.assertIn("webLinkInput.value = locationTypeField.value === \"hub\"", js)


if __name__ == "__main__":
    unittest.main()
