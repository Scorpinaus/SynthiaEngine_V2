from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsModelContractTests(unittest.TestCase):
    def test_workflow_api_docs_cover_base_model_registry_crud(self):
        docs = (ROOT / "docs" / "WORKFLOW_API.md").read_text(encoding="utf-8")
        self.assertIn("### Base model registry endpoints", docs)
        self.assertIn("`GET /models`", docs)
        self.assertIn("`POST /models`", docs)
        self.assertIn("`GET /models/{model_name}`", docs)
        self.assertIn("`PATCH /models/{model_name}`", docs)
        self.assertIn("`DELETE /models/{model_name}`", docs)
        self.assertIn("`POST /api/local-path/select`", docs)
        self.assertIn("`frontend/models/base/edit.html`", docs)


if __name__ == "__main__":
    unittest.main()
