from pathlib import Path

from PIL import Image

from backend.utilities.pipeline import release_pipeline, resolve_base_seed, save_generated_image


ROOT = Path(__file__).resolve().parents[1]


class _HookedPipeline:
    def __init__(self) -> None:
        self.freed = False
        self.removed = False

    def maybe_free_model_hooks(self) -> None:
        self.freed = True

    def remove_all_hooks(self) -> None:
        self.removed = True


class _FailingHookPipeline:
    def __init__(self) -> None:
        self.removed = False

    def maybe_free_model_hooks(self) -> None:
        raise RuntimeError("hook release failed")

    def remove_all_hooks(self) -> None:
        self.removed = True


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_resolve_base_seed_preserves_explicit_seed(monkeypatch):
    monkeypatch.setattr("backend.utilities.pipeline.torch.randint", lambda *_args: None)
    assert resolve_base_seed(123) == 123


def test_save_generated_image_uses_standard_path_and_metadata(tmp_path):
    image = Image.new("RGB", (2, 3), "white")
    relpath = save_generated_image(
        image,
        tmp_path,
        "b1",
        42,
        {"prompt": "test", "initial_image": object()},
        mode="img2img",
        pipeline="test-family",
        remove_params=("initial_image",),
        size=(2, 3),
    )

    assert relpath == "batch_b1/b1_42.png"
    saved = Image.open(tmp_path / "b1_42.png")
    assert saved.text["mode"] == "img2img"
    assert saved.text["pipeline"] == "test-family"
    assert saved.text["seed"] == "42"
    assert saved.text["width"] == "2"
    assert "initial_image" not in saved.text


def test_release_pipeline_calls_diffusers_hook_cleanup(monkeypatch):
    calls = []
    pipe = _HookedPipeline()

    monkeypatch.setattr(
        "backend.utilities.pipeline.cleanup_memory",
        lambda *, clear_cuda=True: calls.append(clear_cuda),
    )

    release_pipeline(pipe, clear_cuda=False)

    assert pipe.freed is True
    assert pipe.removed is True
    assert calls == [False]


def test_release_pipeline_logs_and_continues_when_hook_cleanup_fails(monkeypatch, caplog):
    calls = []
    pipe = _FailingHookPipeline()

    monkeypatch.setattr(
        "backend.utilities.pipeline.cleanup_memory",
        lambda *, clear_cuda=True: calls.append(clear_cuda),
    )

    release_pipeline(pipe)

    assert pipe.removed is True
    assert calls == [True]
    assert "Failed to free pipeline model hooks." in caplog.text


def test_sd15_pipeline_generation_paths_use_shared_pipeline_release():
    source = _read_repo_file("backend/sd15/pipeline.py")

    assert "release_pipeline" in source
    assert source.count("release_pipeline(pipe, logger=logger)") >= 6


def test_sdxl_pipeline_generation_paths_use_shared_pipeline_release():
    source = _read_repo_file("backend/sdxl/pipeline.py")

    assert "from backend.utilities.pipeline import" in source
    assert "release_pipeline" in source
    assert "def _release_pipeline" not in source
    assert source.count("release_pipeline(pipe, logger=logger)") >= 6


def test_flux_pipeline_generation_paths_use_shared_pipeline_release():
    source = _read_repo_file("backend/flux/pipeline.py")

    assert "from backend.utilities.pipeline import" in source
    assert "release_pipeline" in source
    assert "def _release_pipeline" not in source
    assert source.count("release_pipeline(pipe, logger=logger)") >= 3


def test_qwen_image_pipeline_generation_paths_use_shared_pipeline_release():
    source = _read_repo_file("backend/qwen_image/pipeline.py")

    assert "from backend.utilities.pipeline import" in source
    assert "release_pipeline" in source
    assert source.count("release_pipeline(pipe, logger=logger)") >= 3


def test_z_image_pipeline_generation_paths_use_shared_pipeline_release():
    source = _read_repo_file("backend/z_image/pipeline.py")

    assert "from backend.utilities.pipeline import" in source
    assert "release_pipeline" in source
    assert source.count("release_pipeline(pipe, logger=logger)") >= 3
