import unittest

from PIL import Image

from backend.sd15.pipeline import (
    _enable_xformers_memory_efficient_attention_if_available,
    _make_inpaint_controlnet_condition,
    _resize_control_image_to_target,
)


class FakeXformersPipeline:
    def __init__(self, error=None):
        self.error = error
        self.enabled = False

    def enable_xformers_memory_efficient_attention(self):
        if self.error is not None:
            raise self.error
        self.enabled = True


class Sd15ControlNetResizeTests(unittest.TestCase):
    def test_resize_height_only(self):
        image = Image.new("RGB", (512, 256))
        resized = _resize_control_image_to_target(
            image,
            target_width=512,
            target_height=512,
        )
        self.assertEqual(resized.size, (512, 512))

    def test_resize_width_only(self):
        image = Image.new("RGB", (256, 512))
        resized = _resize_control_image_to_target(
            image,
            target_width=512,
            target_height=512,
        )
        self.assertEqual(resized.size, (512, 512))

    def test_resize_width_and_height(self):
        image = Image.new("RGB", (256, 384))
        resized = _resize_control_image_to_target(
            image,
            target_width=512,
            target_height=512,
        )
        self.assertEqual(resized.size, (512, 512))

    def test_resize_list_of_control_images(self):
        images = [
            Image.new("RGB", (256, 512)),
            Image.new("RGB", (512, 256)),
            Image.new("RGB", (640, 640)),
        ]
        resized = _resize_control_image_to_target(
            images,
            target_width=512,
            target_height=512,
        )
        self.assertIsInstance(resized, list)
        self.assertEqual(len(resized), 3)
        self.assertEqual(resized[0].size, (512, 512))
        self.assertEqual(resized[1].size, (512, 512))
        self.assertEqual(resized[2].size, (512, 512))


class Sd15ControlNetXformersTests(unittest.TestCase):
    def test_missing_xformers_does_not_fail_generation_setup(self):
        pipe = FakeXformersPipeline(ModuleNotFoundError("No module named 'xformers'"))

        with self.assertLogs("backend.sd15.pipeline", level="WARNING") as logs:
            enabled = _enable_xformers_memory_efficient_attention_if_available(pipe)

        self.assertFalse(enabled)
        self.assertTrue(
            any("continuing without it" in message for message in logs.output)
        )

    def test_available_xformers_is_enabled(self):
        pipe = FakeXformersPipeline()

        enabled = _enable_xformers_memory_efficient_attention_if_available(pipe)

        self.assertTrue(enabled)
        self.assertTrue(pipe.enabled)


class Sd15InpaintControlNetConditionTests(unittest.TestCase):
    def test_inpaint_condition_sets_masked_pixels_to_negative_one(self):
        image = Image.new("RGB", (2, 1), color=(255, 128, 0))
        mask = Image.new("L", (2, 1), color=0)
        mask.putpixel((1, 0), 255)

        condition = _make_inpaint_controlnet_condition(image, mask)

        self.assertEqual(tuple(condition.shape), (1, 3, 1, 2))
        self.assertGreaterEqual(float(condition[0, 0, 0, 0]), 0.99)
        self.assertEqual(float(condition[0, 0, 0, 1]), -1.0)
        self.assertEqual(float(condition[0, 1, 0, 1]), -1.0)
        self.assertEqual(float(condition[0, 2, 0, 1]), -1.0)


if __name__ == "__main__":
    unittest.main()
