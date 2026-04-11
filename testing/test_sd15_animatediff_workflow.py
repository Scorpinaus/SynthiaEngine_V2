import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.workflow import (
    Sd15AnimateDiffText2VideoInputs,
    _sd15_animatediff_text2video,
    build_workflow_catalog,
)
from backend.sd15_animatediff_pipeline import _validate_animatediff_frame_settings


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

    def test_prompt_is_required(self):
        with self.assertRaises(ValidationError):
            Sd15AnimateDiffText2VideoInputs()


class Sd15AnimateDiffText2VideoWorkflowTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
