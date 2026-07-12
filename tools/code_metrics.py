"""Report repository source-line metrics by ownership category.

This intentionally uses physical lines instead of a language-specific parser so
the same stable measurement works for Python, JavaScript, HTML, and CSS.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SOURCE_SUFFIXES = {".py", ".js", ".html", ".css"}
CATEGORIES = {
    "backend": ("backend",),
    "frontend": ("frontend",),
    "tests": ("testing",),
    "vendored_custom_pipelines": ("custom_pipelines",),
    "tools": ("tools",),
}
EXCLUDED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__", "node_modules"}


@dataclass
class Metrics:
    files: int = 0
    physical_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0

    def add_file(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        self.files += 1
        self.physical_lines += len(lines)
        for line in lines:
            stripped = line.strip()
            if not stripped:
                self.blank_lines += 1
            elif _is_comment(stripped, path.suffix.lower()):
                self.comment_lines += 1
            else:
                self.code_lines += 1


def _is_comment(line: str, suffix: str) -> bool:
    if suffix == ".py":
        return line.startswith("#")
    if suffix in {".js", ".css"}:
        return line.startswith("//") or line.startswith("/*") or line.startswith("*")
    if suffix == ".html":
        return line.startswith("<!--")
    return False


def source_files(root: Path, top_level_dirs: Iterable[str]) -> Iterable[Path]:
    for directory in top_level_dirs:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_SUFFIXES
                and not EXCLUDED_PARTS.intersection(path.parts)
            ):
                yield path


def collect_metrics(root: Path) -> dict[str, Metrics]:
    result: dict[str, Metrics] = {}
    for category, directories in CATEGORIES.items():
        metrics = Metrics()
        for path in source_files(root, directories):
            metrics.add_file(path)
        result[category] = metrics

    maintained = Metrics()
    for category in ("backend", "frontend", "tools"):
        current = result[category]
        for field in Metrics.__dataclass_fields__:
            setattr(maintained, field, getattr(maintained, field) + getattr(current, field))
    result["maintained_total"] = maintained
    return result


def render_table(metrics: dict[str, Metrics]) -> str:
    headers = ("Category", "Files", "Physical", "Code", "Comments", "Blank")
    rows = [
        (
            category,
            str(values.files),
            str(values.physical_lines),
            str(values.code_lines),
            str(values.comment_lines),
            str(values.blank_lines),
        )
        for category, values in metrics.items()
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    output = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers))]
    output.append("  ".join("-" * width for width in widths))
    output.extend("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows)
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    metrics = collect_metrics(args.root.resolve())
    if args.json:
        print(json.dumps({name: asdict(value) for name, value in metrics.items()}, indent=2))
    else:
        print(render_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
