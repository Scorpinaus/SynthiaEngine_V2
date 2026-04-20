import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from backend.ip_adapter_embeds import (
    load_ip_adapter_embeds_artifact,
    save_ip_adapter_embeds_artifact,
)
from backend.workflow_utility import cleanup_artifacts, collect_artifact_ids


class IpAdapterEmbedsArtifactTests(unittest.TestCase):
    def test_embed_artifacts_are_saved_loaded_collected_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with patch("backend.workflow_utility.OUTPUT_DIR", output_dir):
                with patch("backend.ip_adapter_embeds.OUTPUT_DIR", output_dir):
                    artifact = save_ip_adapter_embeds_artifact(
                        [torch.ones((1, 2, 3))],
                        metadata={
                            "adapters": [
                                {
                                    "model": "h94/IP-Adapter",
                                    "subfolder": "sdxl_models",
                                    "weight_name": "ip-adapter_sdxl.bin",
                                    "scale": 0.6,
                                }
                            ],
                            "do_classifier_free_guidance": True,
                            "num_images_per_prompt": 1,
                        },
                    )

                    artifact_id = artifact["artifact_id"]
                    self.assertTrue(artifact_id.startswith("e"))
                    path = output_dir / artifact["path"]
                    self.assertTrue(path.exists())

                    payload = load_ip_adapter_embeds_artifact(artifact)
                    self.assertEqual(payload["family"], "SDXL")
                    self.assertEqual(len(payload["embeds"]), 1)
                    self.assertEqual(collect_artifact_ids({"image_embeds": artifact}), {artifact_id})

                    cleanup_artifacts({artifact_id})
                    self.assertFalse(path.exists())

    def test_embed_artifacts_can_store_sd15_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with patch("backend.workflow_utility.OUTPUT_DIR", output_dir):
                with patch("backend.ip_adapter_embeds.OUTPUT_DIR", output_dir):
                    artifact = save_ip_adapter_embeds_artifact(
                        [torch.ones((1, 2, 3))],
                        family="SD15",
                        metadata={
                            "adapters": [
                                {
                                    "model": "h94/IP-Adapter",
                                    "subfolder": "models",
                                    "weight_name": "ip-adapter_sd15.bin",
                                    "scale": 0.6,
                                }
                            ],
                            "do_classifier_free_guidance": True,
                            "num_images_per_prompt": 1,
                        },
                    )

                    payload = load_ip_adapter_embeds_artifact(artifact)
                    self.assertEqual(payload["family"], "SD15")


if __name__ == "__main__":
    unittest.main()
