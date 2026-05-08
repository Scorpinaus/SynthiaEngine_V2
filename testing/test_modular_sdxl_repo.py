import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_MODULES_CACHE", str(REPO_ROOT / ".pytest_cache" / "hf_modules"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("diffusers")


def test_sdxl_modular_text2img_parser_defaults_to_builtin_sdxl_repo():
    from backend.modular_diffusers.sdxl import sdxl_modular_text2img

    parser = sdxl_modular_text2img.build_parser()
    args = parser.parse_args([])

    assert args.model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert not hasattr(args, "repo_path")
    assert args.output.name == "sdxl_modular_text2img.png"
    assert args.width == 1024
    assert args.height == 1024
    assert args.lora == []
    assert args.textual_inversion == []


def test_sdxl_modular_text2img_main_uses_diffusers_builtin_modular_pipeline(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_text2img

    output_path = Path("outputs") / "sdxl-test.png"
    saved_paths = []

    class FakeImage:
        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def __init__(self):
            self.load_kwargs = None
            self.device = None
            self.call_kwargs = None

        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    calls = {}
    fake_pipe = FakePipeline()

    def fake_from_pretrained(model):
        calls["model"] = model
        return fake_pipe

    monkeypatch.setattr(sdxl_modular_text2img.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sdxl_modular_text2img.ModularPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_text2img.py",
            "--model",
            "custom/sdxl",
            "--prompt",
            "paint a bright portal",
            "--negative-prompt",
            "blurry",
            "--steps",
            "5",
            "--guidance-scale",
            "6.5",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
            "42",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_text2img.main()

    assert saved_paths == [output_path]
    assert calls["model"] == "custom/sdxl"
    assert fake_pipe.load_kwargs == {"torch_dtype": torch.float32}
    assert fake_pipe.device == "cpu"
    assert fake_pipe.call_kwargs["prompt"] == "paint a bright portal"
    assert fake_pipe.call_kwargs["negative_prompt"] == "blurry"
    assert fake_pipe.call_kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_kwargs["guidance_scale"] == 6.5
    assert fake_pipe.call_kwargs["width"] == 16
    assert fake_pipe.call_kwargs["height"] == 16
    assert fake_pipe.call_kwargs["output"] == "images"
    assert isinstance(fake_pipe.call_kwargs["generator"], torch.Generator)


def test_sdxl_modular_text2img_main_applies_adapter_args(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_text2img

    output_path = Path("outputs") / "sdxl-adapter-test.png"
    applied = {}
    saved_paths = []

    class FakeImage:
        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            return [FakeImage()]

    fake_pipe = FakePipeline()

    def fake_apply(pipe, args):
        applied["pipe"] = pipe
        applied["lora"] = args.lora
        applied["lora_weight"] = args.lora_weight
        applied["textual_inversion"] = args.textual_inversion
        applied["textual_inversion_2"] = args.textual_inversion_2

    monkeypatch.setattr(sdxl_modular_text2img.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sdxl_modular_text2img.ModularPipeline,
        "from_pretrained",
        lambda model: fake_pipe,
    )
    monkeypatch.setattr(sdxl_modular_text2img, "apply_sdxl_modular_adapters_from_args", fake_apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_text2img.py",
            "--lora",
            "style.safetensors",
            "--lora-weight",
            "0.8",
            "--textual-inversion",
            "clip_l.pt",
            "--textual-inversion-2",
            "clip_g.pt",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_text2img.main()

    assert saved_paths == [output_path]
    assert applied == {
        "pipe": fake_pipe,
        "lora": ["style.safetensors"],
        "lora_weight": [0.8],
        "textual_inversion": ["clip_l.pt"],
        "textual_inversion_2": ["clip_g.pt"],
    }


def test_sdxl_modular_img2img_parser_defaults_to_builtin_sdxl_repo():
    from backend.modular_diffusers.sdxl import sdxl_modular_img2img

    parser = sdxl_modular_img2img.build_parser()
    args = parser.parse_args(["--image", "input.png"])

    assert args.image == Path("input.png")
    assert args.model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert not hasattr(args, "repo_path")
    assert args.output.name == "sdxl_modular_img2img.png"
    assert args.strength == 0.75
    assert args.width == 1024
    assert args.height == 1024
    assert args.lora == []


def test_sdxl_modular_img2img_main_uses_diffusers_builtin_modular_pipeline(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_img2img

    output_path = Path("outputs") / "sdxl-img2img-test.png"
    saved_paths = []

    class FakeImage:
        def __init__(self, source=None):
            self.source = source
            self.converted_to = []

        def convert(self, mode):
            self.converted_to.append(mode)
            return self

        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def __init__(self):
            self.load_kwargs = None
            self.device = None
            self.call_kwargs = None

        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    opened_images = []
    calls = {}
    fake_pipe = FakePipeline()

    def fake_open(path):
        image = FakeImage(source=path)
        opened_images.append(image)
        return image

    def fake_from_pretrained(model):
        calls["model"] = model
        return fake_pipe

    monkeypatch.setattr(sdxl_modular_img2img.Image, "open", fake_open)
    monkeypatch.setattr(sdxl_modular_img2img.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sdxl_modular_img2img.ModularPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_img2img.py",
            "--image",
            "input.png",
            "--model",
            "custom/sdxl",
            "--prompt",
            "make this cinematic",
            "--negative-prompt",
            "blurry",
            "--strength",
            "0.6",
            "--steps",
            "5",
            "--guidance-scale",
            "6.5",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
            "42",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_img2img.main()

    assert saved_paths == [output_path]
    assert calls["model"] == "custom/sdxl"
    assert fake_pipe.load_kwargs == {"torch_dtype": torch.float32}
    assert fake_pipe.device == "cpu"
    assert len(opened_images) == 1
    assert opened_images[0].source == Path("input.png")
    assert opened_images[0].converted_to == ["RGB"]
    assert fake_pipe.call_kwargs["prompt"] == "make this cinematic"
    assert fake_pipe.call_kwargs["negative_prompt"] == "blurry"
    assert fake_pipe.call_kwargs["image"] is opened_images[0]
    assert fake_pipe.call_kwargs["strength"] == 0.6
    assert fake_pipe.call_kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_kwargs["guidance_scale"] == 6.5
    assert fake_pipe.call_kwargs["width"] == 16
    assert fake_pipe.call_kwargs["height"] == 16
    assert fake_pipe.call_kwargs["output"] == "images"
    assert isinstance(fake_pipe.call_kwargs["generator"], torch.Generator)


def test_sdxl_modular_inpaint_parser_defaults_to_builtin_sdxl_repo():
    from backend.modular_diffusers.sdxl import sdxl_modular_inpaint

    parser = sdxl_modular_inpaint.build_parser()
    args = parser.parse_args(["--image", "input.png", "--mask-image", "mask.png"])

    assert args.image == Path("input.png")
    assert args.mask_image == Path("mask.png")
    assert args.model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert not hasattr(args, "repo_path")
    assert args.output.name == "sdxl_modular_inpaint.png"
    assert args.strength == 1.0
    assert args.padding_mask_crop is None
    assert args.width == 1024
    assert args.height == 1024
    assert args.textual_inversion_2 == []


def test_sdxl_modular_inpaint_main_uses_diffusers_builtin_modular_pipeline(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_inpaint

    output_path = Path("outputs") / "sdxl-inpaint-test.png"
    saved_paths = []

    class FakeImage:
        def __init__(self, source=None):
            self.source = source
            self.converted_to = []

        def convert(self, mode):
            self.converted_to.append(mode)
            return self

        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def __init__(self):
            self.load_kwargs = None
            self.device = None
            self.call_kwargs = None

        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    opened_images = []
    calls = {}
    fake_pipe = FakePipeline()

    def fake_open(path):
        image = FakeImage(source=path)
        opened_images.append(image)
        return image

    def fake_from_pretrained(model):
        calls["model"] = model
        return fake_pipe

    monkeypatch.setattr(sdxl_modular_inpaint.Image, "open", fake_open)
    monkeypatch.setattr(sdxl_modular_inpaint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sdxl_modular_inpaint.ModularPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_inpaint.py",
            "--image",
            "input.png",
            "--mask-image",
            "mask.png",
            "--model",
            "custom/sdxl",
            "--prompt",
            "paint a portal",
            "--negative-prompt",
            "blurry",
            "--strength",
            "0.85",
            "--steps",
            "5",
            "--guidance-scale",
            "6.5",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
            "42",
            "--padding-mask-crop",
            "12",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_inpaint.main()

    assert saved_paths == [output_path]
    assert calls["model"] == "custom/sdxl"
    assert fake_pipe.load_kwargs == {"torch_dtype": torch.float32}
    assert fake_pipe.device == "cpu"
    assert len(opened_images) == 2
    assert opened_images[0].source == Path("input.png")
    assert opened_images[0].converted_to == ["RGB"]
    assert opened_images[1].source == Path("mask.png")
    assert opened_images[1].converted_to == ["L"]
    assert fake_pipe.call_kwargs["prompt"] == "paint a portal"
    assert fake_pipe.call_kwargs["negative_prompt"] == "blurry"
    assert fake_pipe.call_kwargs["image"] is opened_images[0]
    assert fake_pipe.call_kwargs["mask_image"] is opened_images[1]
    assert fake_pipe.call_kwargs["strength"] == 0.85
    assert fake_pipe.call_kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_kwargs["guidance_scale"] == 6.5
    assert fake_pipe.call_kwargs["width"] == 16
    assert fake_pipe.call_kwargs["height"] == 16
    assert fake_pipe.call_kwargs["padding_mask_crop"] == 12
    assert fake_pipe.call_kwargs["output"] == "images"
    assert isinstance(fake_pipe.call_kwargs["generator"], torch.Generator)


def test_sdxl_modular_adapter_spec_parsing():
    from backend.modular_diffusers.sdxl.adapters import parse_lora_specs, parse_textual_inversion_specs

    args = SimpleNamespace(
        lora=["a.safetensors", "b.safetensors"],
        lora_weight=[0.7, 0.9],
        lora_name=["style_a", "style_b"],
        textual_inversion=["clip_l.pt"],
        textual_inversion_token=["<style>"],
        textual_inversion_2=["clip_g.pt"],
        textual_inversion_2_token=["<style>"],
    )

    lora_specs = parse_lora_specs(args)
    textual_specs = parse_textual_inversion_specs(args)

    assert [(spec.path, spec.weight, spec.adapter_name) for spec in lora_specs] == [
        ("a.safetensors", 0.7, "style_a"),
        ("b.safetensors", 0.9, "style_b"),
    ]
    assert [(spec.path, spec.token, spec.encoder) for spec in textual_specs] == [
        ("clip_l.pt", "<style>", "text_encoder"),
        ("clip_g.pt", "<style>", "text_encoder_2"),
    ]


def test_sdxl_modular_adapter_spec_rejects_mismatched_lora_weights():
    from backend.modular_diffusers.sdxl.adapters import parse_lora_specs

    args = SimpleNamespace(lora=["a.safetensors", "b.safetensors"], lora_weight=[0.7], lora_name=[])

    with pytest.raises(ValueError, match="--lora-weight count"):
        parse_lora_specs(args)


def test_sdxl_modular_adapter_spec_rejects_mismatched_textual_tokens():
    from backend.modular_diffusers.sdxl.adapters import parse_textual_inversion_specs

    args = SimpleNamespace(
        textual_inversion=["clip_l.pt", "other.pt"],
        textual_inversion_token=["<style>"],
        textual_inversion_2=[],
        textual_inversion_2_token=[],
    )

    with pytest.raises(ValueError, match="--textual-inversion-token count"):
        parse_textual_inversion_specs(args)


def test_sdxl_modular_adapter_support_binds_loader_methods():
    from backend.modular_diffusers.sdxl.adapters import enable_sdxl_modular_adapter_support

    pipe = SimpleNamespace()

    enable_sdxl_modular_adapter_support(pipe)

    assert pipe._lora_loadable_modules == ["unet", "text_encoder", "text_encoder_2"]
    assert pipe.unet_name == "unet"
    assert pipe.text_encoder_name == "text_encoder"
    assert callable(pipe.load_lora_weights)
    assert callable(pipe.set_adapters)
    assert callable(pipe.load_textual_inversion)
    assert callable(pipe.maybe_convert_prompt)


def test_sdxl_modular_controlnet_text2img_parser_defaults():
    from backend.modular_diffusers.sdxl import sdxl_modular_controlnet_text2img
    from backend.modular_diffusers.sdxl.controlnet import DEFAULT_CONTROLNET_MODEL

    parser = sdxl_modular_controlnet_text2img.build_parser()
    args = parser.parse_args(["--control-image", "control.png"])

    assert args.control_image == "control.png"
    assert args.controlnet_model == DEFAULT_CONTROLNET_MODEL
    assert args.controlnet_conditioning_scale == 1.0
    assert args.control_guidance_start == 0.0
    assert args.control_guidance_end == 1.0
    assert args.guess_mode is False
    assert args.output.name == "sdxl_modular_controlnet_text2img.png"


def test_sdxl_modular_controlnet_text2img_main_loads_controlnet_and_passes_kwargs(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_controlnet_text2img
    from backend.modular_diffusers.sdxl import controlnet as sdxl_controlnet

    output_path = Path("outputs") / "sdxl-controlnet-test.png"
    saved_paths = []
    applied = {}

    class FakeImage:
        def __init__(self, source=None):
            self.source = source
            self.converted_to = []

        def convert(self, mode):
            self.converted_to.append(mode)
            return self

        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def __init__(self):
            self.load_kwargs = None
            self.updated_components = None
            self.device = None
            self.call_kwargs = None

        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def update_components(self, **kwargs):
            self.updated_components = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    fake_pipe = FakePipeline()
    fake_controlnet = object()
    opened_images = []
    controlnet_calls = {}

    def fake_open(path):
        image = FakeImage(source=path)
        opened_images.append(image)
        return image

    def fake_controlnet_from_pretrained(model, torch_dtype):
        controlnet_calls["model"] = model
        controlnet_calls["torch_dtype"] = torch_dtype
        return fake_controlnet

    def fake_apply(pipe, args):
        applied["pipe"] = pipe
        applied["lora"] = args.lora

    monkeypatch.setattr(sdxl_modular_controlnet_text2img.Image, "open", fake_open)
    monkeypatch.setattr(sdxl_modular_controlnet_text2img.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sdxl_modular_controlnet_text2img.ModularPipeline, "from_pretrained", lambda model: fake_pipe)
    monkeypatch.setattr(sdxl_controlnet.ControlNetModel, "from_pretrained", fake_controlnet_from_pretrained)
    monkeypatch.setattr(sdxl_modular_controlnet_text2img, "apply_sdxl_modular_adapters_from_args", fake_apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_controlnet_text2img.py",
            "--control-image",
            "control.png",
            "--controlnet-model",
            "custom/controlnet",
            "--controlnet-conditioning-scale",
            "0.7",
            "--control-guidance-start",
            "0.2",
            "--control-guidance-end",
            "0.8",
            "--guess-mode",
            "--prompt",
            "control this",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_controlnet_text2img.main()

    assert saved_paths == [output_path]
    assert controlnet_calls == {"model": "custom/controlnet", "torch_dtype": torch.float32}
    assert fake_pipe.updated_components == {"controlnet": fake_controlnet}
    assert applied == {"pipe": fake_pipe, "lora": []}
    assert opened_images[0].source == "control.png"
    assert opened_images[0].converted_to == ["RGB"]
    assert fake_pipe.call_kwargs["prompt"] == "control this"
    assert fake_pipe.call_kwargs["control_image"] is opened_images[0]
    assert fake_pipe.call_kwargs["controlnet_conditioning_scale"] == 0.7
    assert fake_pipe.call_kwargs["guess_mode"] is True
    assert fake_pipe.call_kwargs["control_guidance_start"] == 0.2
    assert fake_pipe.call_kwargs["control_guidance_end"] == 0.8
    assert fake_pipe.call_kwargs["output"] == "images"


def test_sdxl_modular_controlnet_img2img_main_passes_images_and_controlnet(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_controlnet_img2img
    from backend.modular_diffusers.sdxl import controlnet as sdxl_controlnet

    output_path = Path("outputs") / "sdxl-controlnet-img2img-test.png"
    saved_paths = []

    class FakeImage:
        def __init__(self, source=None):
            self.source = source
            self.converted_to = []

        def convert(self, mode):
            self.converted_to.append(mode)
            return self

        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def update_components(self, **kwargs):
            self.updated_components = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    fake_pipe = FakePipeline()
    opened_images = []

    def fake_open(path):
        image = FakeImage(source=path)
        opened_images.append(image)
        return image

    monkeypatch.setattr(sdxl_modular_controlnet_img2img.Image, "open", fake_open)
    monkeypatch.setattr(sdxl_modular_controlnet_img2img.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sdxl_modular_controlnet_img2img.ModularPipeline, "from_pretrained", lambda model: fake_pipe)
    monkeypatch.setattr(sdxl_controlnet.ControlNetModel, "from_pretrained", lambda *args, **kwargs: "controlnet")
    monkeypatch.setattr(sdxl_modular_controlnet_img2img, "apply_sdxl_modular_adapters_from_args", lambda pipe, args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_controlnet_img2img.py",
            "--image",
            "input.png",
            "--control-image",
            "control.png",
            "--strength",
            "0.55",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_controlnet_img2img.main()

    assert saved_paths == [output_path]
    assert [image.source for image in opened_images] == [Path("input.png"), "control.png"]
    assert [image.converted_to for image in opened_images] == [["RGB"], ["RGB"]]
    assert fake_pipe.updated_components == {"controlnet": "controlnet"}
    assert fake_pipe.call_kwargs["image"] is opened_images[0]
    assert fake_pipe.call_kwargs["control_image"] is opened_images[1]
    assert fake_pipe.call_kwargs["strength"] == 0.55
    assert "controlnet_conditioning_scale" in fake_pipe.call_kwargs


def test_sdxl_modular_controlnet_inpaint_main_passes_mask_and_controlnet(monkeypatch):
    from backend.modular_diffusers.sdxl import sdxl_modular_controlnet_inpaint
    from backend.modular_diffusers.sdxl import controlnet as sdxl_controlnet

    output_path = Path("outputs") / "sdxl-controlnet-inpaint-test.png"
    saved_paths = []

    class FakeImage:
        def __init__(self, source=None):
            self.source = source
            self.converted_to = []

        def convert(self, mode):
            self.converted_to.append(mode)
            return self

        def save(self, path):
            saved_paths.append(path)

    class FakePipeline:
        def load_components(self, **kwargs):
            self.load_kwargs = kwargs

        def update_components(self, **kwargs):
            self.updated_components = kwargs

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return [FakeImage()]

    fake_pipe = FakePipeline()
    opened_images = []

    def fake_open(path):
        image = FakeImage(source=path)
        opened_images.append(image)
        return image

    monkeypatch.setattr(sdxl_modular_controlnet_inpaint.Image, "open", fake_open)
    monkeypatch.setattr(sdxl_modular_controlnet_inpaint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sdxl_modular_controlnet_inpaint.ModularPipeline, "from_pretrained", lambda model: fake_pipe)
    monkeypatch.setattr(sdxl_controlnet.ControlNetModel, "from_pretrained", lambda *args, **kwargs: "controlnet")
    monkeypatch.setattr(sdxl_modular_controlnet_inpaint, "apply_sdxl_modular_adapters_from_args", lambda pipe, args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sdxl_modular_controlnet_inpaint.py",
            "--image",
            "input.png",
            "--mask-image",
            "mask.png",
            "--control-image",
            "control.png",
            "--padding-mask-crop",
            "16",
            "--output",
            str(output_path),
        ],
    )

    sdxl_modular_controlnet_inpaint.main()

    assert saved_paths == [output_path]
    assert [image.source for image in opened_images] == [Path("input.png"), Path("mask.png"), "control.png"]
    assert [image.converted_to for image in opened_images] == [["RGB"], ["L"], ["RGB"]]
    assert fake_pipe.updated_components == {"controlnet": "controlnet"}
    assert fake_pipe.call_kwargs["image"] is opened_images[0]
    assert fake_pipe.call_kwargs["mask_image"] is opened_images[1]
    assert fake_pipe.call_kwargs["control_image"] is opened_images[2]
    assert fake_pipe.call_kwargs["padding_mask_crop"] == 16
    assert "controlnet_conditioning_scale" in fake_pipe.call_kwargs
