from unittest.mock import patch

from PIL import Image

from backend.ernie_image.pipeline import _generate_text2img_subprocess_child


class _FakePipeline:
    def __init__(self):
        self.unloaded = False

    def __call__(self, **_kwargs):
        return type("Result", (), {"images": [Image.new("RGB", (8, 8), "white")]})()

    def unload_lora_weights(self):
        self.unloaded = True


def test_ernie_image_pipeline_applies_lora_adapters_and_relies_on_subprocess_exit(tmp_path):
    fake_pipe = _FakePipeline()
    lora_adapters = [{"lora_id": 101, "strength": 0.8}]

    with patch("backend.ernie_image.pipeline.load_text2img_pipeline", return_value=fake_pipe):
        with patch("backend.ernie_image.pipeline.make_batch_id", return_value="batch123"):
            with patch("backend.ernie_image.pipeline.OUTPUT_DIR", tmp_path):
                with patch("backend.ernie_image.pipeline.cleanup_memory"):
                    with patch(
                        "backend.ernie_image.pipeline.apply_lora_adapters_with_validation",
                        return_value=(["lora_style"], {"lora_style": {"transformer": {}}}),
                    ) as apply_lora:
                        with patch(
                            "backend.ernie_image.pipeline.write_lora_coverage_report",
                            return_value=None,
                        ):
                            result = _generate_text2img_subprocess_child(
                                {
                                    "prompt": "test prompt",
                                    "num_images": 1,
                                    "seed": 123,
                                    "lora_adapters": lora_adapters,
                                }
                            )

    assert result["images"] == ["/outputs/batch_batch123/batch123_123.png"]
    apply_lora.assert_called_once_with(
        fake_pipe,
        lora_adapters,
        expected_family="ernie-image",
        validate=False,
    )
    assert fake_pipe.unloaded is False
