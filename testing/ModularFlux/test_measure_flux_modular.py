from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

import measure_flux_modular as harness
from custom_pipelines.FluxModular.before_denoise import FluxInpaintPrepareMaskStep
from custom_pipelines.FluxModular import low_memory
from custom_pipelines.FluxModular import device_placement


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"images": [Image.new("RGB", (8, 8), "green")]}


def test_parse_args_defaults_are_low_memory_safe():
    args = harness.parse_args([])

    assert args.case == "flux-text2img"
    assert args.width == 768
    assert args.height == 768
    assert args.steps == 8
    assert args.num_images == 1
    assert args.offload == "auto"
    assert args.load_strategy == "phased"
    assert args.prompt_cache is True
    assert args.prompt_cache_device == "cpu"
    assert args.cuda_placement == "auto"
    assert args.vram_reserve_margin == "3GB"
    assert args.transformer_stream_blocks == "auto"
    assert args.low_memory_sequential_images is True
    assert args.low_memory_transformer_buffers is True
    assert args.decode_chunk_size == 1
    assert args.low_cpu_mem_usage is None
    assert args.offload_state_dict is None
    assert args.disable_mmap is False
    assert args.device_map is None
    assert args.quantization == "none"
    assert args.bnb_4bit_use_double_quant is True
    assert args.system_ram_limit is None


def test_parse_max_memory_entries():
    parsed = harness.parse_max_memory(["0=10GB", "cpu=48GB"])

    assert parsed == {0: "10GB", "cpu": "48GB"}


def test_parse_memory_bytes():
    assert harness.parse_memory_bytes("16GB") == 16_000_000_000
    assert harness.parse_memory_bytes("1GiB") == 1024**3
    assert harness.parse_memory_bytes(None) is None


def test_prompt_cache_key_changes_with_prompt():
    first = harness.parse_args(["--prompt", "first"])
    second = harness.parse_args(["--prompt", "second"])

    assert harness.prompt_cache_key("flux", "model", first, "dtype") != harness.prompt_cache_key(
        "flux",
        "model",
        second,
        "dtype",
    )


def test_prompt_cache_entry_moves_values_to_cpu():
    class FakeTensor:
        def __init__(self, device):
            self.device = device

        def detach(self):
            return self

        def to(self, *args, **kwargs):
            return FakeTensor(kwargs.get("device", args[0] if args else None))

    entry = harness.make_prompt_cache_entry(
        (FakeTensor("cuda"), FakeTensor("cuda")),
        cache_key=("flux", "model"),
        cache_device="cpu",
    )

    assert entry["prompt_embeds"].device == "cpu"
    assert entry["pooled_prompt_embeds"].device == "cpu"


def test_device_placement_parse_memory_bytes():
    assert device_placement.parse_memory_bytes("3GB") == 3_000_000_000
    assert device_placement.parse_memory_bytes("1GiB") == 1024**3


def test_transformer_streaming_patch_can_be_restored():
    module = torch.nn.Module()
    module.forward = lambda value: value

    device_placement.enable_transformer_block_streaming(module, device="cpu", blocks_per_group=2)

    assert device_placement.transformer_streaming_enabled(module) is True
    assert module._fluxmodular_block_stream_config.blocks_per_group == 2

    device_placement.disable_transformer_block_streaming(module)

    assert device_placement.transformer_streaming_enabled(module) is False


def test_img2img_prepare_latents_moves_inputs_to_denoise_device(monkeypatch):
    step = low_memory.LowMemoryFluxImg2ImgPrepareLatentsStep()
    block_state = SimpleNamespace(
        image_latents=torch.zeros((1, 4, 4)),
        latents=torch.zeros((1, 4, 4)),
        timesteps=torch.zeros((1,)),
        guidance=torch.zeros((1,)),
    )
    captured_devices = {}

    class FakeState:
        def __init__(self):
            self.values = {"image_latents": block_state.image_latents}

        def get(self, _name, default=None):
            return default

    class FakeScheduler:
        def scale_noise(self, image_latents, latent_timestep, latents):
            captured_devices["image_latents"] = image_latents.device
            captured_devices["latent_timestep"] = latent_timestep.device
            captured_devices["latents"] = latents.device
            return latents

    def fake_set_block_state(_state, updated_block_state):
        captured_devices["image_latents"] = block_state.image_latents.device
        captured_devices["latents"] = block_state.latents.device
        captured_devices["timesteps"] = block_state.timesteps.device
        captured_devices["guidance"] = block_state.guidance.device
        captured_devices["initial_noise"] = updated_block_state.initial_noise.device

    monkeypatch.setattr(step, "get_block_state", lambda _state: block_state)
    monkeypatch.setattr(step, "set_block_state", fake_set_block_state)
    monkeypatch.setattr(low_memory, "denoise_execution_device", lambda _components: torch.device("meta"))

    _components, state = step(SimpleNamespace(scheduler=FakeScheduler()), FakeState())

    assert captured_devices == {
        "image_latents": torch.device("meta"),
        "latent_timestep": torch.device("meta"),
        "latents": torch.device("meta"),
        "timesteps": torch.device("meta"),
        "guidance": torch.device("meta"),
        "initial_noise": torch.device("meta"),
    }
    assert "image_latents" not in state.values


def test_inpaint_prepare_latents_keeps_image_latents_for_blending(monkeypatch):
    step = low_memory.LowMemoryFluxInpaintPrepareLatentsStep()
    block_state = SimpleNamespace(
        image_latents=torch.zeros((1, 4, 4)),
        latents=torch.ones((1, 4, 4)),
        timesteps=torch.zeros((1,)),
        guidance=torch.zeros((1,)),
    )

    class FakeState:
        def __init__(self):
            self.values = {"image_latents": block_state.image_latents}

        def get(self, _name, default=None):
            return default

    class FakeScheduler:
        def scale_noise(self, image_latents, _latent_timestep, latents):
            return image_latents + latents

    monkeypatch.setattr(step, "get_block_state", lambda _state: block_state)
    monkeypatch.setattr(step, "set_block_state", lambda _state, _block_state: None)
    monkeypatch.setattr(low_memory, "denoise_execution_device", lambda _components: torch.device("cpu"))

    _components, state = step(SimpleNamespace(scheduler=FakeScheduler()), FakeState())

    assert "image_latents" in state.values
    assert torch.equal(block_state.initial_noise, torch.ones((1, 4, 4)))
    assert torch.equal(block_state.latents, torch.ones((1, 4, 4)))


def test_inpaint_prepare_mask_latents_packs_and_repeats_mask():
    mask_condition = torch.zeros((1, 1, 32, 32))
    mask_condition[:, :, 8:24, 8:24] = 1

    mask = FluxInpaintPrepareMaskStep.prepare_mask_latents(
        SimpleNamespace(vae_scale_factor=8),
        mask_condition,
        batch_size=1,
        num_channels_latents=4,
        num_images_per_prompt=2,
        height=32,
        width=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask.shape == (2, 4, 16)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_resolve_cases_supports_pipeline_all():
    args = harness.parse_args(["--case", "all", "--pipeline", "all"])

    cases = harness.resolve_cases(args)

    assert [case.name for case in cases] == [
        "flux-text2img",
        "flux-img2img",
        "flux-inpaint",
        "flux-embeds2img",
        "flux-img2img-embeds",
        "flux-inpaint-embeds",
        "kontext-text2img",
        "kontext-image",
        "kontext-embeds2img",
        "kontext-image-embeds",
    ]


def test_resolve_cases_supports_short_aliases():
    flux_args = harness.parse_args(["--case", "img2img", "--pipeline", "flux"])
    flux_inpaint_args = harness.parse_args(["--case", "inpaint", "--pipeline", "flux"])
    kontext_args = harness.parse_args(["--case", "image", "--pipeline", "kontext"])

    assert [case.name for case in harness.resolve_cases(flux_args)] == ["flux-img2img"]
    assert [case.name for case in harness.resolve_cases(flux_inpaint_args)] == ["flux-inpaint"]
    assert [case.name for case in harness.resolve_cases(kontext_args)] == ["kontext-image"]


def test_build_case_kwargs_for_flux_img2img_uses_synthetic_image():
    args = harness.parse_args(["--case", "flux-img2img", "--pipeline", "flux", "--seed", "99"])
    case = harness.CASES["flux-img2img"]

    kwargs, stats = harness.build_case_kwargs(args, case, FakePipe(), run_seed=99)

    assert kwargs["prompt"] == args.prompt
    assert kwargs["strength"] == args.strength
    assert kwargs["low_memory_transformer_buffers"] is True
    assert kwargs["low_memory_cuda_placement"] == "auto"
    assert kwargs["low_memory_vram_reserve_margin"] == "3GB"
    assert kwargs["low_memory_transformer_stream_blocks"] == "auto"
    assert kwargs["decode_chunk_size"] == 1
    assert isinstance(kwargs["image"], Image.Image)
    assert stats["prepare_seconds"] >= 0
    assert stats["embed_seconds"] is None


def test_build_case_kwargs_for_flux_inpaint_uses_synthetic_mask():
    args = harness.parse_args(["--case", "flux-inpaint", "--pipeline", "flux", "--seed", "99"])
    case = harness.CASES["flux-inpaint"]

    kwargs, stats = harness.build_case_kwargs(args, case, FakePipe(), run_seed=99)

    assert kwargs["prompt"] == args.prompt
    assert kwargs["strength"] == args.strength
    assert isinstance(kwargs["image"], Image.Image)
    assert isinstance(kwargs["mask_image"], Image.Image)
    assert kwargs["mask_image"].mode == "L"
    assert stats["prepare_seconds"] >= 0
    assert stats["embed_seconds"] is None


def test_default_pipeline_loader_uses_direct_local_constructor(monkeypatch):
    calls = {}

    class FakeFluxModularPipeline:
        def __init__(self, **kwargs):
            calls["constructor_kwargs"] = kwargs
            self.tokenizer = None
            self.transformer = None
            self._component_specs = {
                "tokenizer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
                "transformer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
            }

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("from_pretrained should not be used for the local modular harness")

        def load_components(self, name=None, **kwargs):
            calls.setdefault("component_loads", []).append((name, kwargs))
            setattr(self, name, object())

    fake_module = types.ModuleType("custom_pipelines.FluxModular")
    fake_module.FluxModularPipeline = FakeFluxModularPipeline
    fake_module.FluxKontextModularPipeline = FakeFluxModularPipeline

    def fake_enable_low_memory_flux_modular(_pipe, **kwargs):
        calls["offload_kwargs"] = kwargs
        return "auto"

    fake_module.enable_low_memory_flux_modular = fake_enable_low_memory_flux_modular

    args = harness.parse_args(
        [
            "--model",
            r"D:\diffusion\diffusers\FLUX.1-dev",
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--variant",
            "fp16",
            "--local-files-only",
            "--disable-mmap",
            "--low-cpu-mem-usage",
            "--offload-state-dict",
            "--offload-folder",
            str(Path("C:/tmp/flux-load-offload")),
            "--device-map",
            "cpu",
            "--max-memory",
            "cpu=48GB",
            "--quantization",
            "bnb_8bit",
        ]
    )
    quantization_config = {"text_encoder_2": object(), "transformer": object()}
    monkeypatch.setitem(sys.modules, "custom_pipelines.FluxModular", fake_module)
    monkeypatch.setattr(harness, "build_quantization_config_map", lambda _args: quantization_config)
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)
    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )
    monkeypatch.setattr(harness, "precompute_prompt_embeds", lambda _pipe, _args: ("prompt", "pooled"))

    pipe, load_stats = harness.default_pipeline_loader("flux", args)

    assert isinstance(pipe, FakeFluxModularPipeline)
    assert calls["constructor_kwargs"]["pretrained_model_name_or_path"] == r"D:\diffusion\diffusers\FLUX.1-dev"
    assert calls["constructor_kwargs"]["local_files_only"] is True
    assert "variant" not in calls["constructor_kwargs"]
    assert "device_map" not in calls["constructor_kwargs"]
    assert len(calls["component_loads"]) == 2
    tokenizer_kwargs = calls["component_loads"][0][1]
    transformer_kwargs = calls["component_loads"][1][1]
    assert tokenizer_kwargs["local_files_only"] is True
    assert "device_map" not in tokenizer_kwargs
    assert "quantization_config" not in tokenizer_kwargs
    assert transformer_kwargs["variant"] == "fp16"
    assert transformer_kwargs["local_files_only"] is True
    assert transformer_kwargs["low_cpu_mem_usage"] is True
    assert transformer_kwargs["offload_state_dict"] is True
    assert transformer_kwargs["offload_folder"] == str(Path("C:/tmp/flux-load-offload"))
    assert transformer_kwargs["device_map"] == "cpu"
    assert transformer_kwargs["max_memory"] == {"cpu": "48GB"}
    assert transformer_kwargs["disable_mmap"] is True
    assert transformer_kwargs["quantization_config"] is quantization_config["transformer"]
    assert load_stats["offload_mode"] == "auto"
    assert load_stats["load_strategy"] == "phased"
    assert load_stats["quantization"] == "bnb_8bit"
    assert load_stats["component_load_count"] == 2
    assert load_stats["component_loads"][0]["component"] == "tokenizer"
    assert load_stats["component_loads"][0]["phase"] == "prompt_load"
    assert load_stats["component_loads"][1]["component"] == "transformer"
    assert load_stats["component_loads"][1]["phase"] == "generation_load"
    assert load_stats["component_loads"][1]["status"] == "loaded"
    assert [phase["phase"] for phase in load_stats["phase_loads"]] == [
        "prompt_cache",
        "prompt_load",
        "prompt_encode",
        "prompt_release",
        "generation_load",
    ]
    assert load_stats["phase_loads"][0]["status"] == "miss"
    assert load_stats["phase_loads"][3]["released_components"] == ["tokenizer"]
    assert load_stats["cached_prompt_embeds"] is True
    assert load_stats["prompt_cache_hits"] == 0
    assert load_stats["prompt_cache_misses"] == 1
    assert load_stats["prompt_cache_stores"] == 1
    assert load_stats["prompt_cache_entries"] == 1


def test_default_pipeline_loader_reuses_prompt_cache(monkeypatch):
    calls = {"precompute": 0}

    class FakeFluxModularPipeline:
        def __init__(self, **_kwargs):
            self.tokenizer = None
            self.transformer = None
            self._component_specs = {
                "tokenizer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
                "transformer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
            }

        def load_components(self, name=None, **kwargs):
            calls.setdefault("component_loads", []).append((name, kwargs))
            setattr(self, name, object())

    fake_module = types.ModuleType("custom_pipelines.FluxModular")
    fake_module.FluxModularPipeline = FakeFluxModularPipeline
    fake_module.FluxKontextModularPipeline = FakeFluxModularPipeline
    fake_module.enable_low_memory_flux_modular = lambda _pipe, **_kwargs: "auto"

    def fake_precompute(_pipe, _args):
        calls["precompute"] += 1
        return "prompt", "pooled"

    args = harness.parse_args(["--device", "cpu", "--torch-dtype", "float32"])
    monkeypatch.setitem(sys.modules, "custom_pipelines.FluxModular", fake_module)
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)
    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(harness, "precompute_prompt_embeds", fake_precompute)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )

    _first_pipe, first_stats = harness.default_pipeline_loader("flux", args)
    _second_pipe, second_stats = harness.default_pipeline_loader("flux", args)

    assert calls["precompute"] == 1
    assert [name for name, _kwargs in calls["component_loads"]] == ["tokenizer", "transformer", "transformer"]
    assert first_stats["phase_loads"][0]["status"] == "miss"
    assert second_stats["phase_loads"][0]["status"] == "hit"
    assert second_stats["phase_loads"][1]["status"] == "skipped"
    assert second_stats["phase_loads"][2]["reason"] == "prompt_cache_hit"
    assert second_stats["component_load_count"] == 1
    assert second_stats["component_loads"][0]["component"] == "transformer"
    assert second_stats["prompt_cache_hits"] == 1
    assert second_stats["prompt_cache_misses"] == 1
    assert second_stats["prompt_cache_stores"] == 1
    assert second_stats["prompt_cache_entries"] == 1


def test_default_pipeline_loader_eager_loads_all_components(monkeypatch):
    calls = {}

    class FakeFluxModularPipeline:
        def __init__(self, **kwargs):
            calls["constructor_kwargs"] = kwargs
            self.tokenizer = None
            self.transformer = None
            self._component_specs = {
                "tokenizer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
                "transformer": SimpleNamespace(
                    default_creation_method="from_pretrained",
                    pretrained_model_name_or_path="fake",
                    type_hint=object,
                ),
            }

        def load_components(self, name=None, **kwargs):
            calls.setdefault("component_loads", []).append((name, kwargs))
            setattr(self, name, object())

        def to(self, **kwargs):
            calls["to_kwargs"] = kwargs

    fake_module = types.ModuleType("custom_pipelines.FluxModular")
    fake_module.FluxModularPipeline = FakeFluxModularPipeline
    fake_module.FluxKontextModularPipeline = FakeFluxModularPipeline
    fake_module.enable_low_memory_flux_modular = lambda _pipe, **_kwargs: "auto"

    args = harness.parse_args(["--load-strategy", "eager", "--offload", "none", "--device", "cpu"])
    monkeypatch.setitem(sys.modules, "custom_pipelines.FluxModular", fake_module)
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)
    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )

    pipe, load_stats = harness.default_pipeline_loader("flux", args)

    assert isinstance(pipe, FakeFluxModularPipeline)
    assert calls["component_loads"][0][0] == "tokenizer"
    assert calls["component_loads"][1][0] == "transformer"
    assert [component["phase"] for component in load_stats["component_loads"]] == ["eager_load", "eager_load"]
    assert [phase["phase"] for phase in load_stats["phase_loads"]] == ["eager_load"]
    assert load_stats["cached_prompt_embeds"] is False
    assert calls["to_kwargs"]["device"].type == "cpu"


def test_build_case_kwargs_uses_cached_embeds_in_phased_mode():
    args = harness.parse_args([])
    pipe = FakePipe()
    pipe._modular_flux_cached_prompt_embeds = {
        "prompt_embeds": "prompt",
        "pooled_prompt_embeds": "pooled",
    }

    kwargs, stats = harness.build_case_kwargs(args, harness.CASES["flux-text2img"], pipe, run_seed=1)

    assert "prompt" not in kwargs
    assert "prompt_2" not in kwargs
    assert kwargs["prompt_embeds"] == "prompt"
    assert kwargs["pooled_prompt_embeds"] == "pooled"
    assert stats["embed_seconds"] == 0.0


def test_run_measurement_records_success_and_writes_json(tmp_path, monkeypatch):
    output_json = tmp_path / "flux_modular.json"
    output_dir = tmp_path / "images"
    args = harness.parse_args(
        [
            "--case",
            "flux-text2img",
            "--runs",
            "2",
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
        ]
    )
    fake_pipe = FakePipe()

    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 512.0)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)

    def fake_loader(kind, _args):
        return fake_pipe, {"pipeline": kind, "load_seconds": 0.01, "offload_mode": "fake"}

    result = harness.run_measurement(args, pipeline_loader=fake_loader)

    assert result["summary"]["runs"] == 2
    assert result["summary"]["successes"] == 2
    assert result["summary"]["failures"] == 0
    assert len(fake_pipe.calls) == 2
    assert fake_pipe.calls[0]["low_memory_sequential_images"] is True
    assert fake_pipe.calls[0]["low_memory_transformer_buffers"] is True
    assert result["runs"][0]["image_paths"]

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["summary"] == result["summary"]
    assert Path(result["runs"][0]["image_paths"][0]).exists()


def test_run_single_case_marks_rss_limit_exceeded(monkeypatch, tmp_path):
    args = harness.parse_args(
        [
            "--case",
            "flux-text2img",
            "--output-dir",
            str(tmp_path),
            "--system-ram-limit",
            "64MB",
        ]
    )

    class FakePeakRSSSampler:
        def __init__(self, _interval_seconds):
            self.peak_mb = 128.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr(harness, "PeakRSSSampler", FakePeakRSSSampler)
    monkeypatch.setattr(harness, "get_process_rss_mb", lambda: 128.0)
    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)
    monkeypatch.setattr(
        harness,
        "get_cuda_memory_stats",
        lambda: {
            "cuda_available": False,
            "cuda_max_allocated_mb": None,
            "cuda_max_reserved_mb": None,
            "cuda_allocated_after_mb": None,
            "cuda_reserved_after_mb": None,
        },
    )

    result = harness.run_single_case(
        args,
        pipe=FakePipe(),
        case=harness.CASES["flux-text2img"],
        run_index=1,
        kind="measured",
    )

    assert result["status"] == "rss_limit_exceeded"
    assert result["rss_limit_exceeded"] is True
    assert result["error_type"] == "RSSLimitExceeded"
    assert "Peak process RSS" in result["error"]
    assert harness.summarize_runs([result])["failures"] == 1


def test_result_has_failures_includes_load_status():
    assert harness.result_has_failures({"summary": {"failures": 0}, "loads": [{"status": "success"}]}) is False
    assert (
        harness.result_has_failures(
            {"summary": {"failures": 0}, "loads": [{"status": "rss_limit_exceeded"}]}
        )
        is True
    )


def test_run_measurement_records_inference_failure(monkeypatch, tmp_path):
    args = harness.parse_args(["--case", "flux-text2img", "--output-dir", str(tmp_path)])

    class FailingPipe:
        def __call__(self, **_kwargs):
            raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(harness, "reset_cuda_memory_stats", lambda: None)
    monkeypatch.setattr(harness, "synchronize_cuda", lambda: None)

    def fake_loader(kind, _args):
        return FailingPipe(), {"pipeline": kind, "load_seconds": 0.01, "offload_mode": "fake"}

    result = harness.run_measurement(args, pipeline_loader=fake_loader)

    assert result["summary"]["runs"] == 1
    assert result["summary"]["successes"] == 0
    assert result["summary"]["failures"] == 1
    assert result["runs"][0]["status"] == "error"
    assert result["runs"][0]["phase"] == "inference"
    assert result["runs"][0]["error_type"] == "RuntimeError"
    assert "synthetic OOM" in result["runs"][0]["error"]
