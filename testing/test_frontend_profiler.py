from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendProfilerTests(unittest.TestCase):
    def test_profiler_page_wires_live_job_ui(self):
        html = (ROOT / "frontend" / "others" / "profiler.html").read_text(encoding="utf-8")

        self.assertIn('id="profiler-active-jobs"', html)
        self.assertIn('id="profiler-recent-jobs"', html)
        self.assertIn('id="metric-ram-current"', html)
        self.assertIn('id="metric-cuda-current"', html)
        self.assertIn('id="metric-nvml-current"', html)
        self.assertIn('id="chart-ram"', html)
        self.assertIn('id="chart-cuda"', html)
        self.assertIn('id="chart-nvml"', html)
        self.assertIn('<script src="../workflow_client.js?v=1"></script>', html)
        self.assertIn('<script src="profiler.js?v=1"></script>', html)

    def test_profiler_script_uses_jobs_sse_and_profile_fields(self):
        js = (ROOT / "frontend" / "others" / "profiler.js").read_text(encoding="utf-8")

        self.assertIn("/api/jobs?limit=50", js)
        self.assertIn("WorkflowClient.watchJob", js)
        self.assertIn("rss_current_mb", js)
        self.assertIn("cuda_allocated_current_mb", js)
        self.assertIn("cuda_reserved_current_mb", js)
        self.assertIn("nvml_used_current_mb", js)
        self.assertIn("drawChart", js)

    def test_profiler_nav_and_styles_exist(self):
        nav_js = (ROOT / "frontend" / "components" / "nav_bar.js").read_text(encoding="utf-8")
        css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

        self.assertIn('{ href: "others/profiler.html", label: "Profiler" }', nav_js)
        self.assertIn(".profiler-panel", css)
        self.assertIn(".profiler-metrics", css)
        self.assertIn(".profiler-chart canvas", css)


if __name__ == "__main__":
    unittest.main()
