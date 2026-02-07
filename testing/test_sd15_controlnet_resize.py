import unittest

from PIL import Image

from backend.sd15_pipeline import _resize_control_image_to_target


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


if __name__ == "__main__":
    unittest.main()
