from enum import Enum
import sys
import types
import unittest
from unittest.mock import call, patch

import torch

from backend.qwen_image import pipeline as qwen_image_pipeline


def _model_entry(
    *,
    model_type: str = "diffusers",
    location_type: str = "local",
    version: str = "local",
    link: str = r"D:\models\qwen-image-sdnq",
) -> qwen_image_pipeline.ModelRegistryEntry:
    return qwen_image_pipeline.ModelRegistryEntry(
        name="Test Qwen-Image SDNQ",
        family="qwen-image",
        model_type=model_type,
        location_type=location_type,
        model_id=101,
        version=version,
        link=link,
    )


class _FakeVae:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def enable_slicing(self) -> None:
        self._events.append("vae_slicing")

    def enable_tiling(self) -> None:
        self._events.append("vae_tiling")


class _FakePipeline:
    def __init__(
        self,
        events: list[str],
        *,
        transformer_method: object | None = "sdnq",
        text_encoder_method: object | None = "sdnq",
    ) -> None:
        self.transformer = types.SimpleNamespace(
            config=types.SimpleNamespace(
                quantization_config={
                    "quant_method": transformer_method,
                    "sdnq_version": "0.1.4",
                }
            )
        )
        self.text_encoder = types.SimpleNamespace(
            config={
                "quantization_config": {
                    "quant_method": text_encoder_method,
                    "sdnq_version": "0.1.4",
                }
            }
        )
        self.vae = _FakeVae(events)
        self._events = events

    def enable_attention_slicing(self, setting: str) -> None:
        self._events.append(f"attention_slicing:{setting}")

    def enable_sequential_cpu_offload(self) -> None:
        self._events.append("sequential_cpu_offload")

    def enable_model_cpu_offload(self) -> None:
        raise AssertionError("Model CPU offload must not be used.")

    def to(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("The full pipeline must not move to CUDA.")


class QwenImageSdnqLoaderTests(unittest.TestCase):
    def test_shared_loader_registers_sdnq_before_loading_and_uses_safe_memory(self):
        events: list[str] = []
        loaded: dict[str, object] = {}
        fake_pipe = _FakePipeline(events)
        entry = _model_entry(
            location_type="hub",
            version="51bbb04c6c9664cc226f4403a9175aa2d0b29b9d",
            link="Disty0/Qwen-Image-2512-SDNQ-4bit-dynamic",
        )

        class FakePipelineClass:
            @classmethod
            def from_pretrained(cls, source: str, **kwargs: object) -> _FakePipeline:
                events.append("from_pretrained")
                loaded.update(source=source, kwargs=kwargs)
                return fake_pipe

        def register_sdnq() -> str:
            events.append("register_sdnq")
            return "0.2.2"

        def verify_quantization(pipe: object) -> str:
            self.assertIs(pipe, fake_pipe)
            events.append("verify_quantization")
            return "0.1.4"

        with (
            patch.object(
                qwen_image_pipeline,
                "_get_qwen_image_model_entry",
                return_value=entry,
            ),
            patch.object(
                qwen_image_pipeline,
                "_register_sdnq",
                side_effect=register_sdnq,
            ),
            patch.object(
                qwen_image_pipeline,
                "_verify_sdnq_quantization",
                side_effect=verify_quantization,
            ),
            self.assertLogs(qwen_image_pipeline.logger.name, level="INFO") as logs,
        ):
            result = qwen_image_pipeline._load_qwen_image_pipeline(
                FakePipelineClass,
                None,
            )

        self.assertIs(result, fake_pipe)
        self.assertEqual(
            events,
            [
                "register_sdnq",
                "from_pretrained",
                "verify_quantization",
                "attention_slicing:max",
                "vae_slicing",
                "vae_tiling",
                "sequential_cpu_offload",
            ],
        )
        self.assertEqual(loaded["source"], entry.link)
        load_arguments = loaded["kwargs"]
        self.assertIsInstance(load_arguments, dict)
        self.assertIs(load_arguments["torch_dtype"], torch.bfloat16)
        self.assertIs(load_arguments["low_cpu_mem_usage"], True)
        self.assertEqual(load_arguments["revision"], entry.version)
        self.assertNotIn("device_map", load_arguments)
        log_output = "\n".join(logs.output)
        self.assertIn(f"checkpoint={entry.link}", log_output)
        self.assertIn(f"revision={entry.version}", log_output)
        self.assertIn("sdnq_package_version=0.2.2", log_output)
        self.assertIn("sdnq_checkpoint_version=0.1.4", log_output)
        self.assertIn("memory=sequential_cpu_offload", log_output)

    def test_local_diffusers_load_arguments_do_not_include_revision(self):
        entry = _model_entry(
            version="51bbb04c6c9664cc226f4403a9175aa2d0b29b9d",
        )

        load_arguments = qwen_image_pipeline._build_diffusers_load_arguments(entry)

        self.assertEqual(
            load_arguments,
            {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            },
        )

    def test_quantization_validation_accepts_both_sdnq_components(self):
        fake_pipe = _FakePipeline([])

        checkpoint_version = qwen_image_pipeline._verify_sdnq_quantization(fake_pipe)

        self.assertEqual(checkpoint_version, "0.1.4")

    def test_quantization_validation_accepts_sdnq_enum_values(self):
        class QuantizationMethod(str, Enum):
            SDNQ = "sdnq"

        fake_pipe = _FakePipeline(
            [],
            transformer_method=QuantizationMethod.SDNQ,
            text_encoder_method=QuantizationMethod.SDNQ,
        )

        checkpoint_version = qwen_image_pipeline._verify_sdnq_quantization(fake_pipe)

        self.assertEqual(checkpoint_version, "0.1.4")

    def test_register_sdnq_returns_clear_error_when_package_is_missing(self):
        with patch.dict(sys.modules, {"sdnq": None}):
            with self.assertRaises(RuntimeError) as error:
                qwen_image_pipeline._register_sdnq()

        self.assertEqual(
            str(error.exception),
            "Qwen-Image SDNQ requires the 'sdnq' package. Install the project requirements.",
        )

    def test_quantization_validation_stops_and_releases_pipeline(self):
        events: list[str] = []
        fake_pipe = _FakePipeline(events, transformer_method=None)
        entry = _model_entry()

        class FakePipelineClass:
            @classmethod
            def from_pretrained(cls, _source: str, **_kwargs: object) -> _FakePipeline:
                return fake_pipe

        with (
            patch.object(
                qwen_image_pipeline,
                "_get_qwen_image_model_entry",
                return_value=entry,
            ),
            patch.object(qwen_image_pipeline, "_register_sdnq", return_value="0.2.2"),
            patch.object(qwen_image_pipeline, "release_pipeline") as release_pipeline,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Loading stopped to prevent a full BF16 fallback",
            ):
                qwen_image_pipeline._load_qwen_image_pipeline(FakePipelineClass, None)

        self.assertEqual(events, [])
        release_pipeline.assert_called_once_with(
            fake_pipe,
            logger=qwen_image_pipeline.logger,
        )

    def test_loaded_capability_validation_stops_and_releases_pipeline(self):
        events: list[str] = []
        fake_pipe = _FakePipeline(events)
        fake_pipe.text_encoder = None
        entry = _model_entry()

        class FakePipelineClass:
            @classmethod
            def from_pretrained(cls, _source: str, **_kwargs: object) -> _FakePipeline:
                return fake_pipe

        with (
            patch.object(
                qwen_image_pipeline,
                "_get_qwen_image_model_entry",
                return_value=entry,
            ),
            patch.object(qwen_image_pipeline, "_register_sdnq", return_value="0.2.2"),
            patch.object(
                qwen_image_pipeline,
                "_verify_sdnq_quantization",
            ) as verify_quantization,
            patch.object(qwen_image_pipeline, "release_pipeline") as release_pipeline,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "missing required capabilities: text_encoder",
            ):
                qwen_image_pipeline._load_qwen_image_pipeline(FakePipelineClass, None)

        self.assertEqual(events, [])
        verify_quantization.assert_not_called()
        release_pipeline.assert_called_once_with(
            fake_pipe,
            logger=qwen_image_pipeline.logger,
        )

    def test_model_capability_check_rejects_single_file_before_sdnq_import(self):
        entry = _model_entry(model_type="single-file")

        with (
            patch.object(
                qwen_image_pipeline,
                "_get_qwen_image_model_entry",
                return_value=entry,
            ),
            patch.object(qwen_image_pipeline, "_register_sdnq") as register_sdnq,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "supports only Diffusers model folders or Hub repositories",
            ):
                qwen_image_pipeline._load_qwen_image_pipeline(object, None)

        register_sdnq.assert_not_called()

    def test_all_public_pipeline_loaders_use_the_shared_loader(self):
        marker = object()

        with patch.object(
            qwen_image_pipeline,
            "_load_qwen_image_pipeline",
            return_value=marker,
        ) as shared_loader:
            self.assertIs(qwen_image_pipeline.load_text2img_pipeline("test"), marker)
            self.assertIs(qwen_image_pipeline.load_img2img_pipeline("test"), marker)
            self.assertIs(qwen_image_pipeline.load_inpaint_pipeline("test"), marker)

        self.assertEqual(
            shared_loader.call_args_list,
            [
                call(qwen_image_pipeline.QwenImagePipeline, "test"),
                call(qwen_image_pipeline.QwenImageImg2ImgPipeline, "test"),
                call(qwen_image_pipeline.QwenImageInpaintPipeline, "test"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
