from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
STYLES = FRONTEND / "styles"

EXPECTED_IMPORTS = [
    "tokens.css",
    "base.css",
    "layout.css",
    "components.css",
    "generation.css",
    "registry-tools.css",
    "responsive.css",
]


def test_style_entrypoint_imports_all_layers_in_cascade_order():
    entrypoint = (FRONTEND / "style.css").read_text(encoding="utf-8")
    imports = re.findall(r'@import url\("\./styles/([^\"]+)"\);', entrypoint)

    assert imports == EXPECTED_IMPORTS
    assert all((STYLES / name).is_file() for name in imports)

    code_lines = [
        line
        for line in entrypoint.splitlines()
        if line.strip() and not line.lstrip().startswith("/*")
    ]
    assert all(line.startswith("@import ") for line in code_lines)


def test_each_style_layer_owns_expected_rules():
    layer_hooks = {
        "tokens.css": [":root", "--font-sans", "--color-accent"],
        "base.css": ["body {", "*::before", "box-sizing: border-box"],
        "layout.css": [".header-nav", ".nav-group-menu", ".layout"],
        "components.css": [".jobs-panel", ".controlnet-panel", ".preset-panel"],
        "generation.css": [".gallery-panel", ".viewer-frame video", "#adapter-modal"],
        "registry-tools.css": [
            ".workflow-builder-form",
            ".history-panel",
            ".models-panel",
            ".tools-panel",
            ".profiler-panel",
        ],
        "responsive.css": [
            "@media (max-width: 700px)",
            "@media (max-width: 720px)",
            "@media (max-width: 760px)",
            "@media (max-width: 900px)",
        ],
    }

    for name, hooks in layer_hooks.items():
        css = (STYLES / name).read_text(encoding="utf-8")
        for hook in hooks:
            assert hook in css, f"{hook} must stay in {name}"


def test_responsive_rules_stay_in_the_last_layer():
    for name in EXPECTED_IMPORTS[:-1]:
        css = (STYLES / name).read_text(encoding="utf-8")
        assert "@media" not in css, f"Move responsive rules out of {name}"

    responsive = (STYLES / "responsive.css").read_text(encoding="utf-8")
    assert responsive.count("@media") == 5


def test_html_pages_keep_the_stable_style_entrypoint():
    pages = [
        page
        for page in sorted(FRONTEND.rglob("*.html"))
        if "<!doctype html>" in page.read_text(encoding="utf-8").lower()
    ]
    assert pages

    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert re.search(r'href="(?:\.\./){0,2}style\.css\?v=\d+"', html), page
        assert "styles/" not in html, page
