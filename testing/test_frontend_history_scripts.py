from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendHistoryScriptTests(unittest.TestCase):
    def test_history_page_uses_render_language(self):
        html = (ROOT / "frontend" / "others" / "history.html").read_text(encoding="utf-8")

        self.assertIn("view media in the same render batch", html)
        self.assertIn('aria-label="Previous render"', html)
        self.assertIn('aria-label="Next render"', html)

    def test_history_script_renders_video_records(self):
        js = (ROOT / "frontend" / "others" / "history.js").read_text(encoding="utf-8")

        self.assertIn("media_type", js)
        self.assertIn('document.createElement("video")', js)
        self.assertIn("video.controls = true", js)
        self.assertIn('preload = "metadata"', js)
        self.assertIn("history-video-thumb", js)
        self.assertIn("Generate an image or video", js)

    def test_history_styles_support_video_records(self):
        css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

        self.assertIn(".history-video-preview", css)
        self.assertIn(".history-video-thumb", css)
        self.assertIn(".viewer-frame video", css)


if __name__ == "__main__":
    unittest.main()
