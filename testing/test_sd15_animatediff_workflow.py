import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.workflow import (
    Sd15AnimateDiffText2VideoInputs,
    _sd15_animatediff_text2video,
    build_workflow_catalog,
)


class Sd15AnimateDiffText2VideoSchemaTests(unittest.TestCase):
    def test_defaults_include_motion_adapter_and_video_settings(self):
        inputs = Sd15AnimateDiffText2VideoInputs(prompt="test prompt")

        self.assertEqual(inputs.motion_adapter, "guoyww/animatediff-motion-adapter-v1-5-2")
        self.assertEqual(inputs.scheduler, "ddim")
        self.assertEqual(inputs.steps, 25)
        self.assertEqual(inputs.num_frames, 16)
        self.assertEqual(inputs.fps, 8)
        self.assertEqual(inputs.num_videos, 1)

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


if __name__ == "__main__":
    unittest.main()
