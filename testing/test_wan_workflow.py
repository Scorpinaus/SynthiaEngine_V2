import json
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from backend.workflow import (
    WanImage2VideoInputs,
    WanText2VideoInputs,
    build_workflow_catalog,
)
from backend.workflow.assembly import _wan_image2video, _wan_text2video
from backend.wan.pipeline import (
    generate_image2video,
    generate_image2video_in_process,
    generate_text2video,
    generate_text2video_in_process,
    load_image2video_pipeline,
    _validate_wan_resolution,
    _validate_wan_frame_count,
    _validate_wan_i2v_frame_count,
    _wan_video_metadata_path,
    _write_wan_video_metadata,
)

LOCAL_T2V_MODEL = r"D:\diffusion\diffusers\Wan2.1-T2V-1.3B-Diffusers"
LOCAL_VACE_MODEL = r"D:\diffusion\diffusers\Wan2.1-VACE-1.3B-diffusers"
WAN_I2V_MODEL = r"D:\diffusion\diffusers\Wan2.1-I2V-14B-480P-Diffusers"


class WanText2VideoSchemaTests(unittest.TestCase):
    def test_defaults_target_wan21_t2v_13b_480p_safe_memory(self):
        inputs = WanText2VideoInputs(prompt="test prompt")

        self.assertEqual(inputs.model, LOCAL_T2V_MODEL)
        self.assertEqual(inputs.width, 832)
        self.assertEqual(inputs.height, 480)
        self.assertEqual(inputs.num_frames, 49)
        self.assertEqual(inputs.fps, 16)
        self.assertEqual(inputs.steps, 30)
        self.assertEqual(inputs.guidance_scale, 6.0)
        self.assertEqual(inputs.memory_preset, "safe")
        self.assertEqual(inputs.quantization, "none")
        self.assertEqual(inputs.num_videos, 1)
        self.assertIsNone(inputs.reference_image)
        self.assertIsNone(inputs.mask_image)
        self.assertIsNone(inputs.conditioning_video)
        self.assertEqual(inputs.conditioning_scale, 1.0)

    def test_prompt_is_required(self):
        with self.assertRaises(ValidationError):
            WanText2VideoInputs()

    def test_only_supported_frame_counts_are_allowed(self):
        for frame_count in (33, 49, 81):
            self.assertEqual(
                WanText2VideoInputs(prompt="test", num_frames=frame_count).num_frames,
                frame_count,
            )

        with self.assertRaises(ValidationError):
            WanText2VideoInputs(prompt="test", num_frames=65)

    def test_only_supported_resolutions_are_allowed(self):
        WanText2VideoInputs(prompt="test", width=832, height=480)
        WanText2VideoInputs(prompt="test", width=512, height=512)
        _validate_wan_resolution(832, 480)
        _validate_wan_resolution(512, 512)

        with self.assertRaises(ValidationError):
            WanText2VideoInputs(prompt="test", width=768, height=512)

        with self.assertRaisesRegex(
            ValueError,
            "wan.text2video supports only 832x480 or 512x512 output.",
        ):
            _validate_wan_resolution(768, 512)


class WanImage2VideoSchemaTests(unittest.TestCase):
    def test_defaults_target_wan21_i2v_14b_480p_experimental_offload(self):
        inputs = WanImage2VideoInputs(
            prompt="test prompt",
            image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
        )

        self.assertEqual(inputs.model, WAN_I2V_MODEL)
        self.assertEqual(inputs.width, 832)
        self.assertEqual(inputs.height, 480)
        self.assertEqual(inputs.num_frames, 81)
        self.assertEqual(inputs.fps, 16)
        self.assertEqual(inputs.steps, 50)
        self.assertEqual(inputs.guidance_scale, 5.0)
        self.assertEqual(inputs.memory_preset, "offload")
        self.assertEqual(inputs.num_videos, 1)
        self.assertEqual(inputs.experimental_ack, True)
        self.assertEqual(inputs.quantization, "none")

    def test_bnb_8bit_quantization_is_allowed(self):
        inputs = WanImage2VideoInputs(
            prompt="test prompt",
            image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            quantization="bnb_8bit",
        )

        self.assertEqual(inputs.quantization, "bnb_8bit")

    def test_image_is_required(self):
        with self.assertRaises(ValidationError):
            WanImage2VideoInputs(prompt="test prompt")

    def test_only_480p_resolution_is_allowed(self):
        WanImage2VideoInputs(
            prompt="test",
            image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
            width=832,
            height=480,
        )

        with self.assertRaises(ValidationError):
            WanImage2VideoInputs(
                prompt="test",
                image={"artifact_id": "a0123456789abcdef0123456789abcdef"},
                width=512,
                height=512,
            )


class WanText2VideoWorkflowTests(unittest.TestCase):
    def test_wan_video_metadata_sidecar_uses_batch_filename(self):
        temp_dir = Path("testing/.tmp_wan_metadata")
        temp_dir.mkdir(exist_ok=True)
        metadata_path = _wan_video_metadata_path(temp_dir, "batch123")
        try:

            _write_wan_video_metadata(
                metadata_path,
                {
                    "mode": "wan.text2video",
                    "prompt": "test prompt",
                    "batch_id": "batch123",
                    "source_path": Path("local/model"),
                },
            )

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        finally:
            if metadata_path.exists():
                metadata_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

        self.assertEqual(metadata_path.name, "video_batch123.mp4.json")
        self.assertEqual(payload["mode"], "wan.text2video")
        self.assertEqual(payload["prompt"], "test prompt")
        self.assertEqual(payload["batch_id"], "batch123")
        self.assertEqual(payload["source_path"], str(Path("local") / "model"))

    def test_wan_task_passes_expected_generation_params(self):
        captured = {}

        def _fake_generate_videos(params):
            captured.update(params)
            return ["batch/out.mp4"]

        with patch("backend.workflow.assembly.make_batch_id", return_value="batch123"):
            with patch(
                "backend.workflow.assembly.generate_wan_text2video",
                side_effect=_fake_generate_videos,
            ):
                result = _wan_text2video(
                    {
                        "prompt": "test prompt",
                        "negative_prompt": "bad",
                        "steps": 28,
                        "guidance_scale": 5.5,
                        "width": 832,
                        "height": 480,
                        "seed": 123,
                        "model": LOCAL_T2V_MODEL,
                        "num_frames": 81,
                        "fps": 16,
                        "num_videos": 1,
                        "memory_preset": "safe",
                        "quantization": "bnb_8bit",
                    },
                    _ctx=None,
                )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["videos"], ["/outputs/batch/out.mp4"])
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["negative_prompt"], "bad")
        self.assertEqual(captured["steps"], 28)
        self.assertEqual(captured["guidance_scale"], 5.5)
        self.assertEqual(captured["width"], 832)
        self.assertEqual(captured["height"], 480)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["model"], LOCAL_T2V_MODEL)
        self.assertEqual(captured["num_frames"], 81)
        self.assertEqual(captured["fps"], 16)
        self.assertEqual(captured["num_videos"], 1)
        self.assertEqual(captured["memory_preset"], "safe")
        self.assertEqual(captured["quantization"], "bnb_8bit")
        self.assertEqual(captured["batch_id"], "batch123")

    def test_wan_task_passes_vace_media_refs(self):
        captured = {}
        reference = Image.new("RGB", (32, 32), "blue")
        mask = Image.new("L", (32, 32), 255)
        video_path = Path("conditioning.mp4")

        def _fake_generate_videos(params):
            captured.update(params)
            return ["batch/out.mp4"]

        with patch("backend.workflow.assembly.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.assembly._open_image_ref", side_effect=[reference, mask]):
                with patch("backend.workflow.assembly._open_video_ref", return_value=video_path):
                    with patch(
                        "backend.workflow.assembly.generate_wan_text2video",
                        side_effect=_fake_generate_videos,
                    ):
                        result = _wan_text2video(
                            {
                                "prompt": "test prompt",
                                "width": 512,
                                "height": 512,
                                "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                                "reference_image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                                "mask_image": {"artifact_id": "a1123456789abcdef0123456789abcdef"},
                                "conditioning_video": {"artifact_id": "v0123456789abcdef0123456789abcdef"},
                                "conditioning_scale": 0.75,
                            },
                            _ctx=None,
                        )

        self.assertEqual(result["videos"], ["/outputs/batch/out.mp4"])
        self.assertIs(captured["reference_image"], reference)
        self.assertIs(captured["mask_image"], mask)
        self.assertEqual(captured["conditioning_video"], video_path)
        self.assertEqual(captured["conditioning_scale"], 0.75)
        self.assertEqual(captured["width"], 512)
        self.assertEqual(captured["height"], 512)

    def test_wan_task_is_exposed_in_catalog(self):
        catalog = build_workflow_catalog()

        self.assertIn("wan.text2video", catalog["tasks"])
        self.assertIn("wan.image2video", catalog["tasks"])
        self.assertIn("wan", catalog["capabilities"])
        self.assertIn("wan.text2video", catalog["capabilities"]["wan"]["task_types"])
        self.assertIn("wan.image2video", catalog["capabilities"]["wan"]["task_types"])
        self.assertTrue(catalog["capabilities"]["wan"]["features"]["text2video"])

    def test_wan_frame_count_validation_message_is_actionable(self):
        _validate_wan_frame_count(33)
        _validate_wan_frame_count(49)
        _validate_wan_frame_count(81)

        with self.assertRaisesRegex(
            ValueError,
            "num_frames must be one of 33, 49, 81 for wan.text2video",
        ):
            _validate_wan_frame_count(65)

    def test_vace_requires_reference_image(self):
        with self.assertRaisesRegex(
            ValueError,
            "reference_image is required for Wan VACE generation.",
        ):
            generate_text2video_in_process(
                {
                    "prompt": "test prompt",
                    "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                    "conditioning_video": Path("conditioning.mp4"),
                    "mask_image": Image.new("L", (32, 32), 255),
                    "width": 512,
                    "height": 512,
                }
            )

    def test_vace_generation_uses_local_vace_pipeline_without_t2v_loader(self):
        loaded_models = []

        class _FakeVacePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_vace(model, *, memory_preset, quantization):
            loaded_models.append((model, memory_preset, quantization))
            return _FakeVacePipe()

        with patch(
            "backend.wan.pipeline.load_text2video_pipeline",
            side_effect=AssertionError("T2V loader should not be used for VACE"),
        ):
            with patch("backend.wan.pipeline.load_vace_pipeline", side_effect=_fake_load_vace):
                with patch(
                    "backend.wan.pipeline._prepare_vace_conditions",
                    return_value=(["video-frame"], ["mask-frame"], ["reference-frame"]),
                ):
                    with patch("backend.wan.pipeline.export_to_video"):
                        output = generate_text2video_in_process(
                            {
                                "prompt": "test prompt",
                                "model": LOCAL_T2V_MODEL,
                                "conditioning_video": Path("conditioning.mp4"),
                                "mask_image": Image.new("L", (32, 32), 255),
                                "reference_image": Image.new("RGB", (32, 32), "blue"),
                                "width": 512,
                                "height": 512,
                                "seed": 123,
                                "batch_id": "batch123",
                            }
                        )

        self.assertEqual(loaded_models, [(LOCAL_VACE_MODEL, "safe", "none")])
        self.assertEqual(output, ["batch_batch123/batch123_123.mp4"])

    def test_vace_generation_passes_quantization_to_loader(self):
        loaded_models = []

        class _FakeVacePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_vace(model, *, memory_preset, quantization):
            loaded_models.append((model, memory_preset, quantization))
            return _FakeVacePipe()

        with patch("backend.wan.pipeline.load_vace_pipeline", side_effect=_fake_load_vace):
            with patch(
                "backend.wan.pipeline._prepare_vace_conditions",
                return_value=(["video-frame"], ["mask-frame"], ["reference-frame"]),
            ):
                with patch("backend.wan.pipeline.export_to_video"):
                    output = generate_text2video_in_process(
                        {
                            "prompt": "test prompt",
                            "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                            "conditioning_video": Path("conditioning.mp4"),
                            "mask_image": Image.new("L", (32, 32), 255),
                            "reference_image": Image.new("RGB", (32, 32), "blue"),
                            "width": 512,
                            "height": 512,
                            "seed": 123,
                            "batch_id": "batch123",
                            "quantization": "bnb_8bit",
                        }
                    )

        self.assertEqual(
            loaded_models,
            [("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "safe", "bnb_8bit")],
        )
        self.assertEqual(output, ["batch_batch123/batch123_123.mp4"])

    def test_vace_generation_releases_pipeline_after_render(self):
        released = []

        class _FakeVacePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_vace(_model, *, memory_preset, quantization):
            return _FakeVacePipe()

        with patch("backend.wan.pipeline.load_vace_pipeline", side_effect=_fake_load_vace):
            with patch(
                "backend.wan.pipeline._prepare_vace_conditions",
                return_value=(["video-frame"], ["mask-frame"], ["reference-frame"]),
            ):
                with patch("backend.wan.pipeline.export_to_video"):
                    with patch(
                        "backend.wan.pipeline.release_pipeline",
                        side_effect=lambda pipe, logger=None: released.append(pipe),
                    ):
                        generate_text2video_in_process(
                            {
                                "prompt": "test prompt",
                                "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                                "conditioning_video": Path("conditioning.mp4"),
                                "mask_image": Image.new("L", (32, 32), 255),
                                "reference_image": Image.new("RGB", (32, 32), "blue"),
                                "width": 512,
                                "height": 512,
                                "seed": 123,
                                "batch_id": "batch123",
                            }
                        )

        self.assertEqual(len(released), 1)
        self.assertIsInstance(released[0], _FakeVacePipe)

    def test_wan_t2v_generation_passes_quantization_to_loader(self):
        loaded = []

        class _FakePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_t2v(model, *, memory_preset, quantization):
            loaded.append((model, memory_preset, quantization))
            return _FakePipe()

        with patch("backend.wan.pipeline.load_text2video_pipeline", side_effect=_fake_load_t2v):
            with patch("backend.wan.pipeline.export_to_video"):
                output = generate_text2video_in_process(
                    {
                        "prompt": "test prompt",
                        "model": LOCAL_T2V_MODEL,
                        "seed": 123,
                        "batch_id": "batch123",
                        "quantization": "bnb_8bit",
                    }
                )

        self.assertEqual(loaded, [(LOCAL_T2V_MODEL, "safe", "bnb_8bit")])
        self.assertEqual(output, ["batch_batch123/batch123_123.mp4"])

    def test_wan_t2v_generation_releases_pipeline_after_render(self):
        released = []

        class _FakePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_t2v(_model, *, memory_preset, quantization):
            return _FakePipe()

        with patch("backend.wan.pipeline.load_text2video_pipeline", side_effect=_fake_load_t2v):
            with patch("backend.wan.pipeline.export_to_video"):
                with patch(
                    "backend.wan.pipeline.release_pipeline",
                    side_effect=lambda pipe, logger=None: released.append(pipe),
                ):
                    generate_text2video_in_process(
                        {
                            "prompt": "test prompt",
                            "model": LOCAL_T2V_MODEL,
                            "seed": 123,
                            "batch_id": "batch123",
                        }
                    )

        self.assertEqual(len(released), 1)
        self.assertIsInstance(released[0], _FakePipe)


class WanImage2VideoWorkflowTests(unittest.TestCase):
    def test_wan_i2v_task_passes_expected_generation_params(self):
        captured = {}
        image = Image.new("RGB", (640, 360), "blue")

        def _fake_generate_videos(params):
            captured.update(params)
            return ["batch/out.mp4"]

        with patch("backend.workflow.assembly.make_batch_id", return_value="batch123"):
            with patch("backend.workflow.assembly._open_image_ref", return_value=image):
                with patch(
                    "backend.workflow.assembly.generate_wan_image2video",
                    side_effect=_fake_generate_videos,
                ):
                    result = _wan_image2video(
                        {
                            "prompt": "test prompt",
                            "negative_prompt": "bad",
                            "image": {"artifact_id": "a0123456789abcdef0123456789abcdef"},
                            "steps": 40,
                            "guidance_scale": 4.5,
                            "width": 832,
                            "height": 480,
                            "seed": 123,
                            "model": WAN_I2V_MODEL,
                            "num_frames": 81,
                            "fps": 16,
                            "num_videos": 1,
                            "memory_preset": "offload",
                            "quantization": "bnb_8bit",
                            "experimental_ack": True,
                        },
                        _ctx=None,
                    )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["videos"], ["/outputs/batch/out.mp4"])
        self.assertIs(captured["image"], image)
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["negative_prompt"], "bad")
        self.assertEqual(captured["steps"], 40)
        self.assertEqual(captured["guidance_scale"], 4.5)
        self.assertEqual(captured["width"], 832)
        self.assertEqual(captured["height"], 480)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["model"], WAN_I2V_MODEL)
        self.assertEqual(captured["num_frames"], 81)
        self.assertEqual(captured["fps"], 16)
        self.assertEqual(captured["num_videos"], 1)
        self.assertEqual(captured["memory_preset"], "offload")
        self.assertEqual(captured["quantization"], "bnb_8bit")
        self.assertEqual(captured["batch_id"], "batch123")

    def test_wan_i2v_generation_requires_experimental_ack(self):
        with self.assertRaisesRegex(
            ValueError,
            "experimental_ack must be true for wan.image2video",
        ):
            generate_image2video_in_process(
                {
                    "prompt": "test prompt",
                    "image": Image.new("RGB", (32, 32), "blue"),
                    "experimental_ack": False,
                }
            )

    def test_wan_i2v_frame_count_validation_message_is_actionable(self):
        _validate_wan_i2v_frame_count(33)
        _validate_wan_i2v_frame_count(49)
        _validate_wan_i2v_frame_count(81)

        with self.assertRaisesRegex(
            ValueError,
            "num_frames must be one of 33, 49, 81 for wan.image2video",
        ):
            _validate_wan_i2v_frame_count(65)

        with self.assertRaisesRegex(
            ValueError,
            "num_frames must be one of 33, 49, 81 for wan.image2video",
        ):
            generate_image2video_in_process(
                {
                    "prompt": "test prompt",
                    "image": Image.new("RGB", (32, 32), "blue"),
                    "experimental_ack": True,
                    "num_frames": 65,
                }
            )

    def test_wan_i2v_generation_passes_quantization_to_loader(self):
        loaded = []

        class _FakePipe:
            def __call__(self, **_kwargs):
                return SimpleNamespace(frames=[["frame"]])

        def _fake_load_i2v(model, *, memory_preset, quantization):
            loaded.append((model, memory_preset, quantization))
            return _FakePipe()

        with patch("backend.wan.pipeline.load_image2video_pipeline", side_effect=_fake_load_i2v):
            with patch("backend.wan.pipeline.export_to_video"):
                output = generate_image2video_in_process(
                    {
                        "prompt": "test prompt",
                        "image": Image.new("RGB", (832, 480), "blue"),
                        "seed": 123,
                        "batch_id": "batch123",
                        "quantization": "bnb_8bit",
                        "experimental_ack": True,
                    }
                )

        self.assertEqual(loaded, [(WAN_I2V_MODEL, "offload", "bnb_8bit")])
        self.assertEqual(output, ["batch_batch123/batch123_123.mp4"])

    def test_wan_i2v_loader_quantizes_only_transformer_and_text_encoder(self):
        captured = {}
        quantization_config = object()

        class _FakeAutoencoder:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return object()

        class _FakeImageEncoder:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return object()

        class _FakeScheduler:
            config = {}

            @classmethod
            def from_config(cls, *_args, **_kwargs):
                return cls()

        class _FakePipe:
            def __init__(self):
                self.scheduler = _FakeScheduler()
                self.offload_enabled = False

            def enable_model_cpu_offload(self):
                self.offload_enabled = True

        class _FakeWanImageToVideoPipeline:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                captured["model"] = model
                captured["kwargs"] = kwargs
                return _FakePipe()

        def _fake_build_config(quantization, *, components_to_quantize, task_type):
            captured["quantization"] = quantization
            captured["components_to_quantize"] = components_to_quantize
            captured["task_type"] = task_type
            return quantization_config

        fake_diffusers = SimpleNamespace(
            AutoencoderKLWan=_FakeAutoencoder,
            WanImageToVideoPipeline=_FakeWanImageToVideoPipeline,
        )
        fake_scheduler_module = SimpleNamespace(UniPCMultistepScheduler=_FakeScheduler)
        fake_transformers = SimpleNamespace(CLIPVisionModel=_FakeImageEncoder)

        with patch.dict(
            sys.modules,
            {
                "diffusers": fake_diffusers,
                "diffusers.schedulers.scheduling_unipc_multistep": fake_scheduler_module,
                "transformers": fake_transformers,
            },
        ):
            with patch("torch.cuda.is_available", return_value=True):
                with patch(
                    "backend.wan.pipeline.build_diffusers_pipeline_quantization_config",
                    side_effect=_fake_build_config,
                ):
                    pipe = load_image2video_pipeline(
                        WAN_I2V_MODEL,
                        memory_preset="offload",
                        quantization="bnb_8bit",
                    )

        self.assertTrue(pipe.offload_enabled)
        self.assertEqual(captured["model"], WAN_I2V_MODEL)
        self.assertEqual(captured["quantization"], "bnb_8bit")
        self.assertEqual(captured["components_to_quantize"], ["transformer", "text_encoder"])
        self.assertEqual(captured["task_type"], "wan.image2video")
        self.assertIs(captured["kwargs"]["quantization_config"], quantization_config)


if __name__ == "__main__":
    unittest.main()
