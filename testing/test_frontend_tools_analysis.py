from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendToolsAnalysisTests(unittest.TestCase):
    def test_tools_analysis_page_renders_architecture_summary(self):
        html = (ROOT / "frontend" / "others" / "tools_analysis.html").read_text(encoding="utf-8")

        self.assertIn('id="tools-architecture-value"', html)
        self.assertIn('id="tools-architecture-detail"', html)
        self.assertIn('id="tools-architecture-evidence"', html)

    def test_tools_analysis_script_handles_metadata_status(self):
        js = (ROOT / "frontend" / "others" / "tools_analysis.js").read_text(encoding="utf-8")

        self.assertIn("result.metadata_available", js)
        self.assertIn("Safetensors metadata is not present or not available.", js)
        self.assertIn("result.architecture_evidence", js)


if __name__ == "__main__":
    unittest.main()
