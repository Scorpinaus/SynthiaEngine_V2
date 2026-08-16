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
        self.assertIn("coverage report for both transformer adapters", docs)
        self.assertIn("Call `unload_lora_weights()` in `finally`", docs)
        self.assertIn("Call `release_pipeline` after adapter cleanup", docs)
        self.assertIn("Qwen Image Lightning adapter", docs)
        self.assertIn("`qwen-image.img2img`, and `qwen-image.inpaint`", docs)
        self.assertIn("Image-to-image and inpaint Lightning use is experimental", docs)
        self.assertIn("`qwen-image-2512` base variant", docs)
        self.assertIn("Mixed Lightning request example:", docs)
        self.assertIn("at most one explicitly", docs)
        self.assertIn("Missing or mismatched metadata fails", docs)
        self.assertIn("Standard-only multi-LoRA requests keep their existing behavior", docs)
        self.assertIn("The companion strength is user controlled.", docs)
        self.assertIn("profile value (`4` or `8`)", docs)
        self.assertIn("`true_cfg_scale` must be `1.0`", docs)
        self.assertIn("`qwen_image_lightning_shift3` scheduler profile", docs)
        self.assertIn("Hub entries can provide `weight_name`, `subfolder`, and `revision`", docs)
        self.assertNotIn("Do not combine it with a standard or style", docs)
        self.assertNotIn("Lightning uses one adapter only.", docs)
        self.assertNotIn(
            "The workflow catalog marks scheduler selection and LoRA as unavailable",
            docs,
        )

    def test_workflow_api_docs_cover_qwen_lightning_mixed_stack_contract(self):
        docs = (ROOT / "docs" / "WORKFLOW_API.md").read_text(encoding="utf-8")

        self.assertIn("Mixed Lightning request example:", docs)
        self.assertIn('{ "lora_id": 101, "strength": 1.0, "target": "both" }', docs)
        self.assertIn('{ "lora_id": 102, "strength": 0.35, "target": "both" }', docs)
        self.assertIn("same request is valid with the two adapter entries reversed", docs)
        self.assertIn("the requested order.", docs)
        self.assertIn("at most one explicitly", docs)
        self.assertIn("`qwen-image-2512`, `qwen_image_lightning`, and the normalized current task", docs)
        self.assertIn("Missing or mismatched metadata fails", docs)
        self.assertIn("before pipeline load.", docs)
        self.assertIn("Two Lightning adapters and two or more companions with Lightning are not", docs)
        self.assertIn("A non-Qwen entry, a non-standard LoRA type, Qwen Image Edit or", docs)
        self.assertIn("a non-transformer target are not supported", docs)
        self.assertIn("The companion strength is user controlled.", docs)
        self.assertIn("fixed 4- or 8-step profile, CFG `1.0`, and shift-3 scheduler", docs)
        self.assertIn("Mixed Lightning use is experimental.", docs)
        self.assertIn("`compatibility` is an operator declaration.", docs)
        self.assertIn("Official upstream examples do not", docs)
        self.assertIn("GPU quality acceptance has not", docs)
        self.assertIn("yet run.", docs)

        self.assertIn("Qwen-Image LoRA user interface:", docs)
        self.assertIn("Experimental stack: Lightning + 1\n  LoRA", docs)
        self.assertIn("Lightning strength is locked at `1.0`.", docs)
        self.assertIn("Companion strength stays enabled.", docs)
        self.assertIn("Removing only the companion keeps Lightning settings.", docs)
        self.assertIn("Removing Lightning", docs)
        self.assertIn("restores the earlier steps and True CFG values.", docs)
        self.assertIn("A mixed preset restores its adapters in order and sends one profile event.", docs)

        self.assertIn("unique name for each selected adapter", docs)
        self.assertIn("once with ordered names and weights", docs)
        self.assertIn("never fuses adapters", docs)
        self.assertIn("coverage report for both transformer adapters", docs)
        self.assertIn("requires joint activation support", docs)
        self.assertIn("verifies both active names", docs)
        self.assertIn("unloads once in `finally` before release", docs)
        self.assertIn("partial second adapter load", docs)
        self.assertIn("active-adapter verification failure", docs)
        self.assertIn("inference failure, or cancellation", docs)
        self.assertIn("The next base request has no adapter", docs)
        self.assertIn("or Lightning scheduler state.", docs)
        self.assertIn("The existing `lora_adapters` list schema is unchanged", docs)
        self.assertIn("No dependency change", docs)
        self.assertIn("is required.", docs)
        self.assertNotIn("This metadata is declarative in Slice 1.", docs)

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
        self.assertIn("runtime_profile", docs)
        self.assertIn("weight_name", docs)
        self.assertIn("subfolder", docs)
        self.assertIn("revision", docs)
        self.assertIn('"kind": "qwen_image_lightning"', docs)
        self.assertIn('"base_variant": "qwen-image-2512"', docs)
        self.assertIn('"scheduler_profile": "qwen_image_lightning_shift3"', docs)
        self.assertIn('"supported_tasks": ["text2img", "img2img", "inpaint"]', docs)
        self.assertIn("A Hub Lightning entry requires `weight_name`", docs)
        self.assertIn("`compatibility` is optional operator-declared metadata", docs)
        self.assertIn('"base_variants": ["qwen-image-2512"]', docs)
        self.assertIn('"runtime_profile_kinds": ["qwen_image_lightning"]', docs)
        self.assertIn("All three `compatibility` fields are required", docs)
        self.assertIn("non-empty list with no duplicate values", docs)
        self.assertIn("Unknown fields are rejected.", docs)
        self.assertIn("`compatibility` requires `lora_model_family: \"qwen-image\"`", docs)
        self.assertIn("not allowed on a Lightning entry with `runtime_profile`", docs)
        self.assertIn("Existing registry rows remain valid and return `compatibility: null`", docs)
        self.assertIn("Send `compatibility: null` in `PATCH /lora-models/{lora_id}`", docs)
        self.assertIn("`compatibility` is an operator declaration.", docs)
        self.assertNotIn("This metadata is declarative in Slice 1.", docs)
        self.assertIn("`runtime_profile`, `compatibility`, `weight_name`", docs)
        self.assertIn("`runtime_profile`, `compatibility`, `weight_name`, `subfolder`", docs)

        self.assertIn("Compatibility guarantees:", docs)
        self.assertIn("Existing `GET /lora-models` and `POST /lora-models` consumers are backward-compatible.", docs)
        self.assertIn(
            "Lightning runtime metadata, declarative compatibility metadata, and Hub source coordinates are additive nullable request/response fields.",
            docs,
        )


if __name__ == "__main__":
    unittest.main()
