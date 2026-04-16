import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.sd15_pipeline import generate_images_img2img


class FakeGenerator:
    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeImg2ImgPipeline:
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


class Sd15Img2ImgLcmPipelineTests(unittest.TestCase):
    def test_lcm_mode_loads_lcm_lora_and_uses_lcm_scheduler(self):
        pipe = FakeImg2ImgPipeline()
        scheduler_calls = []

        def _fake_create_scheduler(name, scheduler_pipe):
            scheduler_calls.append((name, scheduler_pipe))
            return SimpleNamespace(config={"name": name})

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("backend.sd15_pipeline.OUTPUT_DIR", Path(tmpdir)):
                with patch("backend.sd15_pipeline.load_img2img_pipeline", return_value=pipe):
                    with patch("backend.sd15_pipeline.create_scheduler", side_effect=_fake_create_scheduler):
                        with patch("backend.sd15_pipeline.torch.Generator", return_value=FakeGenerator()):
                            filenames = generate_images_img2img(
                                {
                                    "initial_image": Image.new("RGB", (16, 16), color="black"),
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
        self.assertEqual(filenames, ["batch_batch_lcm/batch_lcm_123.png"])


if __name__ == "__main__":
    unittest.main()
