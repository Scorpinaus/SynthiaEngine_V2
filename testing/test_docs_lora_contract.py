from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsLoraContractTests(unittest.TestCase):
    def test_architecture_docs_cover_qwen_lora_support_and_lifecycle(self):
        docs = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

        self.assertIn(
            "| `qwen-image` | yes | no | yes | yes | no | no | yes | no | yes |",
            docs,
        )
        self.assertIn("transformer-only LoRA", docs)
        self.assertIn("family `qwen-image`, type `lora`", docs)
        self.assertIn("writes transformer coverage", docs)
        self.assertIn("unloads requested adapters in `finally`", docs)
        self.assertIn("`release_pipeline`, including after load failure", docs)

    def test_workflow_api_docs_cover_qwen_transformer_lora_contract(self):
        docs = (ROOT / "docs" / "WORKFLOW_API.md").read_text(encoding="utf-8")

        self.assertIn(
            "| `qwen-image` | yes | no | yes | yes | no | no | yes | no | yes |",
            docs,
        )
        self.assertIn("Qwen-Image LoRA input notes:", docs)
        self.assertIn('{ "lora_id": 101, "strength": 0.8 }', docs)
        self.assertIn('"target": "both"', docs)
        self.assertIn('lora_model_family: "qwen-image"', docs)
        self.assertIn('lora_type: "lora"', docs)
        self.assertIn("Call `load_lora_weights(...)`", docs)
        self.assertIn("Call `set_adapters(...)` once", docs)
        self.assertIn("coverage report for the transformer", docs)
        self.assertIn("Call `unload_lora_weights()` in `finally`", docs)
        self.assertIn("Call `release_pipeline` after adapter cleanup", docs)
        self.assertNotIn(
            "The workflow catalog marks scheduler selection and LoRA as unavailable",
            docs,
        )

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
