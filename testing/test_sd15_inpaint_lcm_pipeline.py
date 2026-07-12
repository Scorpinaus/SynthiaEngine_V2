import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from PIL import Image

from backend.sd15.pipeline import generate_images_inpaint_in_process


class FakeGenerator:
    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeInpaintPipeline:
    def __init__(self):
        self.scheduler = SimpleNamespace(config={"name": "base"})
        self.loaded_loras = []
        self.adapter_calls = []
        self.unloaded = False
        self.calls = []

    def load_lora_weights(self, model_id, adapter_name=None):
        self.loaded_loras.append((model_id, adapter_name))

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.adapter_calls.append((adapter_names, adapter_weights))

    def unload_lora_weights(self):
        self.unloaded = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), color="white")])


class Sd15InpaintLcmPipelineTests(unittest.TestCase):
    def test_lcm_mode_loads_lcm_lora_and_uses_lcm_scheduler(self):
        pipe = FakeInpaintPipeline()
        scheduler_calls = []

        def _fake_create_scheduler(name, scheduler_pipe):
            scheduler_calls.append((name, scheduler_pipe))
            return SimpleNamespace(config={"name": name})

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15.pipeline.OUTPUT_DIR", Path(tmpdir)):
                with patch("backend.sd15.pipeline.load_inpaint_pipeline", return_value=pipe):
                    with patch("backend.sd15.pipeline.create_scheduler", side_effect=_fake_create_scheduler):
                        with patch("backend.sd15.pipeline.torch.Generator", return_value=FakeGenerator()):
                            with patch(
                                "backend.sd15.pipeline.build_prompt_embeddings",
                                return_value=(torch.ones(1, 6, 1), torch.zeros(1, 6, 1), True),
                            ):
                                filenames = generate_images_inpaint_in_process(
                                    {
                                        "initial_image": Image.new("RGB", (16, 16), color="black"),
                                        "mask_image": Image.new("L", (16, 16), color="white"),
                                        "prompt": "test prompt",
                                        "steps": 4,
                                        "cfg": 0.0,
                                        "scheduler": "lcm",
                                        "seed": 123,
                                        "batch_id": "batch_lcm",
                                    }
                                )

        self.assertEqual(scheduler_calls, [("lcm", pipe)])
        self.assertEqual(
            pipe.loaded_loras,
            [("latent-consistency/lcm-lora-sdv1-5", "lcm_lora_sd15")],
        )
        self.assertEqual(pipe.adapter_calls, [(["lcm_lora_sd15"], [1.0])])
        self.assertTrue(pipe.unloaded)
        self.assertEqual(pipe.calls[0]["num_inference_steps"], 4)
        self.assertEqual(pipe.calls[0]["guidance_scale"], 0.0)
        self.assertIsNone(pipe.calls[0]["prompt"])
        self.assertIsNone(pipe.calls[0]["negative_prompt"])
        self.assertIsNone(pipe.calls[0]["clip_skip"])
        self.assertIsNotNone(pipe.calls[0]["prompt_embeds"])
        self.assertIsNotNone(pipe.calls[0]["negative_prompt_embeds"])
        self.assertEqual(filenames, ["batch_batch_lcm/batch_lcm_123.png"])


if __name__ == "__main__":
    unittest.main()
