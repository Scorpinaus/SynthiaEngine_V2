import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError

from backend.workflow import (
    Sd15AnimateDiffText2VideoInputs,
    _sd15_animatediff_text2video,
    build_workflow_catalog,
)
from backend.sd15.animatediff_pipeline import _make_animatediff_generator
from backend.sd15.animatediff_pipeline import _validate_animatediff_frame_settings
from backend.sd15.animatediff_pipeline import _validate_free_init_settings
from backend.sd15.animatediff_pipeline import _enable_free_init
from backend.sd15.animatediff_pipeline import _prepare_animatediff_prompt_inputs
from backend.sd15.animatediff_pipeline import _animatediff_video_metadata_path
from backend.sd15.animatediff_pipeline import _write_animatediff_video_metadata


class Sd15AnimateDiffText2VideoSchemaTests(unittest.TestCase):
    def test_defaults_include_motion_adapter_and_video_settings(self):
        inputs = Sd15AnimateDiffText2VideoInputs(prompt="test prompt")

        self.assertEqual(inputs.motion_adapter, "guoyww/animatediff-motion-adapter-v1-5-2")
        self.assertEqual(inputs.scheduler, "ddim")
        self.assertEqual(inputs.steps, 25)
        self.assertEqual(inputs.num_frames, 16)
        self.assertEqual(inputs.fps, 8)
        self.assertEqual(inputs.num_videos, 1)
        self.assertFalse(inputs.free_noise_enabled)
        self.assertEqual(inputs.free_noise_context_length, 16)
        self.assertEqual(inputs.free_noise_context_stride, 4)
        self.assertFalse(inputs.free_init_enabled)
        self.assertEqual(inputs.free_init_num_iters, 3)
        self.assertFalse(inputs.free_init_use_fast_sampling)
        self.assertEqual(inputs.free_init_method, "butterworth")
        self.assertEqual(inputs.free_init_order, 4)
        self.assertEqual(inputs.free_init_spatial_stop_frequency, 0.25)
        self.assertEqual(inputs.free_init_temporal_stop_frequency, 0.25)

    def test_prompt_is_required(self):
        with self.assertRaises(ValidationError):
            Sd15AnimateDiffText2VideoInputs()


class Sd15AnimateDiffText2VideoWorkflowTests(unittest.TestCase):
    def test_animatediff_video_metadata_sidecar_uses_batch_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = _animatediff_video_metadata_path(Path(temp_dir), "batch123")

            _write_animatediff_video_metadata(
                metadata_path,
                {
                    "prompt": "test prompt",
                    "batch_id": "batch123",
                    "source_path": Path("local/model"),
                },
            )

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata_path.name, "video_batch123.mp4.json")
        self.assertEqual(payload["prompt"], "test prompt")
        self.assertEqual(payload["batch_id"], "batch123")
        self.assertEqual(payload["source_path"], str(Path("local") / "model"))

    def test_animatediff_task_passes_expected_generation_params(self):
        captured = {}

        def _fake_generate_videos(params):
            captured.update(params)
            return ["batch/out.mp4"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch(
                "backend.workflow.generate_videos_text2video",
                side_effect=_fake_generate_videos,
            ):
                result = _sd15_animatediff_text2video(
                    {
                        "prompt": "test prompt",
                        "negative_prompt": "bad",
                        "steps": 18,
                        "cfg": 6.5,
                        "width": 640,
                        "height": 384,
                        "seed": 123,
                        "scheduler": "ddim",
                        "model": "stable-diffusion-v1-5",
                        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
                        "num_frames": 24,
                        "fps": 12,
                        "num_videos": 2,
                        "free_noise_enabled": True,
                        "free_noise_context_length": 16,
                        "free_noise_context_stride": 4,
                        "free_init_enabled": True,
                        "free_init_num_iters": 4,
                        "free_init_use_fast_sampling": True,
                        "free_init_method": "gaussian",
                        "free_init_order": 5,
                        "free_init_spatial_stop_frequency": 0.2,
                        "free_init_temporal_stop_frequency": 0.3,
                        "clip_skip": 2,
                        "lora": {
                            "lora_enabled": True,
                            "lora_adapters": [{"lora_id": 101, "strength": 0.75}],
                        },
                        "weighting_policy": "a1111-like",
                    },
                    _ctx=None,
                )

        self.assertEqual(result["batch_id"], "batch123")
        self.assertEqual(result["videos"], ["/outputs/batch/out.mp4"])
        self.assertEqual(captured["prompt"], "test prompt")
        self.assertEqual(captured["negative_prompt"], "bad")
        self.assertEqual(captured["steps"], 18)
        self.assertEqual(captured["cfg"], 6.5)
        self.assertEqual(captured["width"], 640)
        self.assertEqual(captured["height"], 384)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["scheduler"], "ddim")
        self.assertEqual(captured["model"], "stable-diffusion-v1-5")
        self.assertEqual(captured["motion_adapter"], "guoyww/animatediff-motion-adapter-v1-5-3")
        self.assertEqual(captured["num_frames"], 24)
        self.assertEqual(captured["fps"], 12)
        self.assertEqual(captured["num_videos"], 2)
        self.assertTrue(captured["free_noise_enabled"])
        self.assertEqual(captured["free_noise_context_length"], 16)
        self.assertEqual(captured["free_noise_context_stride"], 4)
        self.assertTrue(captured["free_init_enabled"])
        self.assertEqual(captured["free_init_num_iters"], 4)
        self.assertTrue(captured["free_init_use_fast_sampling"])
        self.assertEqual(captured["free_init_method"], "gaussian")
        self.assertEqual(captured["free_init_order"], 5)
        self.assertEqual(captured["free_init_spatial_stop_frequency"], 0.2)
        self.assertEqual(captured["free_init_temporal_stop_frequency"], 0.3)
        self.assertEqual(captured["clip_skip"], 2)
        self.assertEqual(captured["lora_adapters"], [{"lora_id": 101, "strength": 0.75}])
        self.assertEqual(captured["weighting_policy"], "a1111-like")
        self.assertEqual(captured["batch_id"], "batch123")

    def test_animatediff_disables_lora_when_flag_is_false(self):
        captured = {}

        def _fake_generate_videos(params):
            captured.update(params)
            return ["batch/out.mp4"]

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch(
                "backend.workflow.generate_videos_text2video",
                side_effect=_fake_generate_videos,
            ):
                _sd15_animatediff_text2video(
                    {
                        "prompt": "test prompt",
                        "lora": {
                            "lora_enabled": False,
                            "lora_adapters": [{"lora_id": 101, "strength": 0.75}],
                        },
                    },
                    _ctx=None,
                )

        self.assertEqual(captured["lora_adapters"], [])

    def test_animatediff_task_is_exposed_in_catalog(self):
        catalog = build_workflow_catalog()

        self.assertIn("sd15.animatediff.text2video", catalog["tasks"])
        self.assertIn(
            "sd15.animatediff.text2video",
            catalog["capabilities"]["sd15"]["task_types"],
        )
        self.assertTrue(catalog["capabilities"]["sd15"]["features"]["text2video"])

    def test_animatediff_frame_limit_requires_free_noise_for_longer_video(self):
        with self.assertRaisesRegex(
            ValueError,
            "num_frames=48 exceeds motion adapter temporal limit 32",
        ):
            _validate_animatediff_frame_settings(
                num_frames=48,
                free_noise_enabled=False,
                free_noise_context_length=16,
                free_noise_context_stride=4,
                motion_max_seq_length=32,
            )

    def test_animatediff_free_noise_allows_longer_video_with_short_context(self):
        _validate_animatediff_frame_settings(
            num_frames=48,
            free_noise_enabled=True,
            free_noise_context_length=16,
            free_noise_context_stride=4,
            motion_max_seq_length=32,
        )

    def test_animatediff_free_noise_stride_must_not_exceed_context(self):
        with self.assertRaisesRegex(
            ValueError,
            "free_noise_context_stride must be <= free_noise_context_length",
        ):
            _validate_animatediff_frame_settings(
                num_frames=48,
                free_noise_enabled=True,
                free_noise_context_length=16,
                free_noise_context_stride=24,
                motion_max_seq_length=32,
            )

    def test_animatediff_free_noise_generator_is_cpu_for_randperm(self):
        generator = _make_animatediff_generator(seed=123, free_noise_enabled=True)

        self.assertEqual(generator.device.type, "cpu")
        shuffled = torch.randperm(4, generator=generator)
        self.assertEqual(shuffled.device.type, "cpu")

    def test_animatediff_free_init_settings_are_validated(self):
        _validate_free_init_settings(
            num_iters=3,
            method="butterworth",
            order=4,
            spatial_stop_frequency=0.25,
            temporal_stop_frequency=0.25,
        )

        with self.assertRaisesRegex(
            ValueError,
            "free_init_method must be one of butterworth, ideal, gaussian",
        ):
            _validate_free_init_settings(
                num_iters=3,
                method="bad",
                order=4,
                spatial_stop_frequency=0.25,
                temporal_stop_frequency=0.25,
            )

    def test_animatediff_enable_free_init_calls_pipeline(self):
        class FakePipe:
            def __init__(self):
                self.kwargs = None

            def enable_free_init(self, **kwargs):
                self.kwargs = kwargs

        pipe = FakePipe()

        _enable_free_init(
            pipe,
            num_iters=4,
            use_fast_sampling=True,
            method="gaussian",
            order=5,
            spatial_stop_frequency=0.2,
            temporal_stop_frequency=0.3,
        )

        self.assertEqual(
            pipe.kwargs,
            {
                "num_iters": 4,
                "use_fast_sampling": True,
                "method": "gaussian",
                "order": 5,
                "spatial_stop_frequency": 0.2,
                "temporal_stop_frequency": 0.3,
            },
        )

    def test_animatediff_free_noise_uses_raw_prompt_inputs(self):
        with patch("backend.sd15.animatediff_pipeline.build_prompt_embeddings") as mocked:
            prompt_input, negative_prompt_input, prompt_embeds, negative_prompt_embeds = (
                _prepare_animatediff_prompt_inputs(
                    pipe=object(),
                    prompt="long prompt",
                    negative_prompt="long negative",
                    clip_skip=1,
                    weighting_policy="diffusers-like",
                    free_noise_enabled=True,
                )
            )

        mocked.assert_not_called()
        self.assertEqual(prompt_input, "long prompt")
        self.assertEqual(negative_prompt_input, "long negative")
        self.assertIsNone(prompt_embeds)
        self.assertIsNone(negative_prompt_embeds)

    def test_animatediff_normal_mode_keeps_prompt_embedding_path(self):
        with patch(
            "backend.sd15.animatediff_pipeline.build_prompt_embeddings",
            return_value=("pos", "neg", True),
        ) as mocked:
            prompt_input, negative_prompt_input, prompt_embeds, negative_prompt_embeds = (
                _prepare_animatediff_prompt_inputs(
                    pipe=object(),
                    prompt="weighted prompt",
                    negative_prompt="weighted negative",
                    clip_skip=1,
                    weighting_policy="a1111-like",
                    free_noise_enabled=False,
                )
            )

        mocked.assert_called_once()
        self.assertIsNone(prompt_input)
        self.assertIsNone(negative_prompt_input)
        self.assertEqual(prompt_embeds, "pos")
        self.assertEqual(negative_prompt_embeds, "neg")


if __name__ == "__main__":
    unittest.main()
