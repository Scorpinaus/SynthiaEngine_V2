from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsLoraContractTests(unittest.TestCase):
    def test_workflow_api_docs_cover_lora_crud_contract_and_compatibility(self):
        docs = (ROOT / "docs" / "WORKFLOW_API.md").read_text(encoding="utf-8")

        self.assertIn("### LoRA registry endpoints", docs)
        self.assertIn("`GET /lora-models`", docs)
        self.assertIn("`POST /lora-models`", docs)
        self.assertIn("`GET /lora-models/{lora_id}`", docs)
        self.assertIn("`PATCH /lora-models/{lora_id}`", docs)
        self.assertIn("`DELETE /lora-models/{lora_id}`", docs)

        self.assertIn("Response `200`: `LoraRegistryEntry[]`", docs)
        self.assertIn("Response `200`: created `LoraRegistryEntry`", docs)
        self.assertIn("Response `200`: updated `LoraRegistryEntry`", docs)
        self.assertIn("Returns `204` on success.", docs)

        self.assertIn("Error `400`", docs)
        self.assertIn("Error `404`", docs)
        self.assertIn("Error `422`", docs)
        self.assertIn("LoRA with id <lora_id> already exists.", docs)
        self.assertIn("LoRA with id <lora_id> not found.", docs)
        self.assertIn("prompt_presets", docs)
        self.assertIn("Each prompt preset has a non-empty `name` and a non-empty `words` list.", docs)
        self.assertIn("Preset words are prompt fragments intended for frontend prompt composition", docs)

        self.assertIn("Compatibility guarantees:", docs)
        self.assertIn("Existing `GET /lora-models` and `POST /lora-models` consumers are backward-compatible.", docs)


if __name__ == "__main__":
    unittest.main()
