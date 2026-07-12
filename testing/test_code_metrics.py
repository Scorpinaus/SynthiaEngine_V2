from pathlib import Path

from tools.code_metrics import collect_metrics, source_files


def test_source_files_only_returns_supported_source_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "backend"
    source_dir.mkdir()
    (source_dir / "module.py").write_text("# note\n\nvalue = 1\n", encoding="utf-8")
    (source_dir / "data.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in source_files(tmp_path, ("backend",))] == ["module.py"]


def test_collect_metrics_separates_maintained_tests_and_vendor_code(tmp_path: Path) -> None:
    for directory in ("backend", "frontend", "testing", "custom_pipelines", "tools"):
        (tmp_path / directory).mkdir()
    (tmp_path / "backend" / "main.py").write_text("# comment\n\nanswer = 42\n", encoding="utf-8")
    (tmp_path / "frontend" / "app.js").write_text("// comment\nrun();\n", encoding="utf-8")
    (tmp_path / "testing" / "test_app.py").write_text("def test_app():\n    pass\n", encoding="utf-8")
    (tmp_path / "custom_pipelines" / "vendor.py").write_text("vendor = True\n", encoding="utf-8")

    metrics = collect_metrics(tmp_path)

    assert metrics["backend"].code_lines == 1
    assert metrics["frontend"].code_lines == 1
    assert metrics["tests"].code_lines == 2
    assert metrics["vendored_custom_pipelines"].code_lines == 1
    assert metrics["maintained_total"].code_lines == 2
    assert metrics["maintained_total"].files == 2
