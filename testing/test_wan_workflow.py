import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend.workflow import (
    WanText2VideoInputs,
    _wan_text2video,
    build_workflow_catalog,
)
from backend.wan.pipeline import (
    _validate_wan_frame_count,
    _wan_video_metadata_path,
    _write_wan_video_metadata,
)


class WanText2VideoSchemaTests(unittest.TestCase):
    def test_defaults_target_wan21_t2v_13b_480p_safe_memory(self):
        inputs = WanText2VideoInputs(prompt="test prompt")

        self.assertEqual(inputs.model, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(inputs.width, 832)
        self.assertEqual(inputs.height, 480)
        self.assertEqual(inputs.num_frames, 49)
        self.assertEqual(inputs.fps, 16)
        self.assertEqual(inputs.steps, 30)
        self.assertEqual(inputs.guidance_scale, 6.0)
        self.assertEqual(inputs.memory_preset, "safe")
        self.assertEqual(inputs.num_videos, 1)

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

        with patch("backend.workflow.make_batch_id", return_value="batch123"):
            with patch(
                "backend.workflow.generate_wan_text2video",
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
                        "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        "num_frames": 81,
                        "fps": 16,
                        "num_videos": 1,
                        "memory_preset": "safe",
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
        self.assertEqual(captured["model"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(captured["num_frames"], 81)
        self.assertEqual(captured["fps"], 16)
        self.assertEqual(captured["num_videos"], 1)
        self.assertEqual(captured["memory_preset"], "safe")
        self.assertEqual(captured["batch_id"], "batch123")

    def test_wan_task_is_exposed_in_catalog(self):
        catalog = build_workflow_catalog()

        self.assertIn("wan.text2video", catalog["tasks"])
        self.assertIn("wan", catalog["capabilities"])
        self.assertIn("wan.text2video", catalog["capabilities"]["wan"]["task_types"])
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


if __name__ == "__main__":
    unittest.main()
