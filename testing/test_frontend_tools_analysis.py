from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendToolsAnalysisTests(unittest.TestCase):
    def test_tools_analysis_page_renders_architecture_summary(self):
        html = (ROOT / "frontend" / "others" / "tools_analysis.html").read_text(encoding="utf-8")

        self.assertIn('id="tools-architecture-value"', html)
        self.assertIn('id="tools-architecture-detail"', html)
        self.assertIn('id="tools-architecture-evidence"', html)
        self.assertIn('id="tools-metadata"', html)
        self.assertIn('class="tools-table tools-metadata-table"', html)
        self.assertIn('id="tools-metadata-head"', html)
        self.assertIn('id="tools-metadata-body"', html)
        self.assertIn('id="tools-table-head"', html)
        self.assertIn('aria-controls="tools-metadata-body"', html)
        self.assertIn('aria-controls="tools-table-body"', html)

    def test_tools_analysis_script_handles_metadata_status(self):
        js = (ROOT / "frontend" / "others" / "tools_analysis.js").read_text(encoding="utf-8")

        self.assertIn("result.metadata_available", js)
        self.assertIn("Safetensors metadata is not present or not available.", js)
        self.assertIn("result.safetensors_metadata", js)
        self.assertIn("result.architecture_evidence", js)
        self.assertIn("function setRowsCollapsed", js)
        self.assertIn("function bindCollapsibleHeader", js)
        self.assertIn("bindCollapsibleHeader(metadataHead, metadataBody)", js)
        self.assertIn("bindCollapsibleHeader(tableHead, tableBody)", js)


if __name__ == "__main__":
    unittest.main()
