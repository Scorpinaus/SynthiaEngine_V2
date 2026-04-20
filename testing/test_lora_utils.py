from types import SimpleNamespace

import backend.lora.utils as lora_utils


class _DummyPipe:
    class _DummyComponent:
        def __init__(self):
            self.loaded: list[tuple[str, str, str | None]] = []
            self.adapter_calls: list[tuple[list[str], list[float | dict[str, object]]]] = []

        def load_lora_adapter(self, file_path: str, adapter_name: str, prefix: str | None = None):
            self.loaded.append((file_path, adapter_name, prefix))

        def set_adapters(self, adapter_names: list[str], weights: list[float | dict[str, object]]):
            self.adapter_calls.append((adapter_names, weights))

    def __init__(self):
        self.loaded: list[tuple[str, str]] = []
        self.adapter_calls: list[tuple[list[str], list[float | dict[str, object]]]] = []
        self.unet = self._DummyComponent()
        self.text_encoder = self._DummyComponent()

    def load_lora_weights(self, file_path: str, adapter_name: str):
        self.loaded.append((file_path, adapter_name))

    def set_adapters(self, adapter_names: list[str], adapter_weights: list[float | dict[str, object]]):
        self.adapter_calls.append((adapter_names, adapter_weights))


def test_apply_lora_adapters_sanitizes_dot_names(monkeypatch):
    def _mock_get_lora_entry(_lora_id: int):
        return SimpleNamespace(
            lora_id=101,
            lora_model_family="sdxl",
            name="Pinpin Art Style V3.0",
            file_path="C:/loras/pinpin.safetensors",
        )

    monkeypatch.setattr(lora_utils, "get_lora_entry", _mock_get_lora_entry)
    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[{"lora_id": 101, "strength": 0.8}],
        expected_family="sdxl",
        validate=False,
    )

    assert adapter_names == ["lora_Pinpin_Art_Style_V3_0"]
    assert pipe.loaded == [("C:/loras/pinpin.safetensors", "lora_Pinpin_Art_Style_V3_0")]
    assert pipe.adapter_calls[0][0] == ["lora_Pinpin_Art_Style_V3_0"]


def test_apply_lora_adapters_handles_sanitized_name_collision(monkeypatch):
    entries = {
        201: SimpleNamespace(
            lora_id=201,
            lora_model_family="sdxl",
            name="Same.Name",
            file_path="C:/loras/a.safetensors",
        ),
        202: SimpleNamespace(
            lora_id=202,
            lora_model_family="sdxl",
            name="Same Name",
            file_path="C:/loras/b.safetensors",
        ),
    }

    monkeypatch.setattr(lora_utils, "get_lora_entry", lambda lora_id: entries[lora_id])
    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[{"lora_id": 201, "strength": 1.0}, {"lora_id": 202, "strength": 1.0}],
        expected_family="sdxl",
        validate=False,
    )

    assert adapter_names == ["lora_Same_Name", "lora_Same_Name_202"]
    assert [name for _, name in pipe.loaded] == ["lora_Same_Name", "lora_Same_Name_202"]


def test_apply_lora_adapters_supports_unet_and_text_encoder_scales(monkeypatch):
    def _mock_get_lora_entry(_lora_id: int):
        return SimpleNamespace(
            lora_id=301,
            lora_model_family="sd15",
            name="Layered",
            file_path="C:/loras/layered.safetensors",
        )

    calls: list[dict[str, object]] = []

    def _mock_build_text_encoder_scales_map(
        pipe,
        *,
        adapter_name: str,
        adapter_index: int,
        default_scale: float,
        overrides: dict[str, float],
    ) -> dict[str, float]:
        calls.append(
            {
                "pipe": pipe,
                "adapter_name": adapter_name,
                "adapter_index": adapter_index,
                "default_scale": default_scale,
                "overrides": overrides,
            }
        )
        return {
            "text_model.encoder.layers.0.self_attn.q_proj": overrides["layers.0"] if "layers.0" in overrides else 0.4,
            "text_model.encoder.layers.1.self_attn.q_proj": default_scale,
        }

    monkeypatch.setattr(lora_utils, "get_lora_entry", _mock_get_lora_entry)
    monkeypatch.setattr(lora_utils, "_build_text_encoder_scales_map", _mock_build_text_encoder_scales_map)

    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[
            {
                "lora_id": 301,
                "strength": 0.9,
                "text_encoder_strength": 0.6,
                "text_encoder_scales": {"layers.0": 0.4},
                "unet_scales": {"mid": 0.7, "up": {"block_0": [0.5, 0.6, 0.7]}},
            }
        ],
        expected_family="sd15",
        validate=False,
    )

    assert adapter_names == ["lora_Layered"]
    assert len(calls) == 1
    assert calls[0]["default_scale"] == 0.6
    assert calls[0]["overrides"] == {"layers.0": 0.4}
    _, adapter_weights = pipe.adapter_calls[0]
    assert adapter_weights == [
        {
            "unet": {"mid": 0.7, "up": {"block_0": [0.5, 0.6, 0.7]}},
            "text_encoder": {
                "text_model.encoder.layers.0.self_attn.q_proj": 0.4,
                "text_model.encoder.layers.1.self_attn.q_proj": 0.6,
            },
        }
    ]


def test_apply_lora_adapters_supports_unet_target(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=501,
            lora_model_family="sd15",
            name="UnetOnly",
            file_path="C:/loras/unet_only.safetensors",
        ),
    )

    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[{"lora_id": 501, "strength": 0.6, "target": "unet"}],
        expected_family="sd15",
        validate=False,
    )

    assert adapter_names == ["lora_UnetOnly"]
    assert pipe.loaded == []
    assert pipe.unet.loaded == [("C:/loras/unet_only.safetensors", "lora_UnetOnly", "unet")]
    assert pipe.unet.adapter_calls == [(["lora_UnetOnly"], [0.6])]
    assert pipe.text_encoder.loaded == []
    assert pipe.text_encoder.adapter_calls == []


def test_apply_lora_adapters_supports_text_encoder_target(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=502,
            lora_model_family="sd15",
            name="TextOnly",
            file_path="C:/loras/text_only.safetensors",
        ),
    )

    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[{"lora_id": 502, "strength": 0.7, "target": "text_encoder"}],
        expected_family="sd15",
        validate=False,
    )

    assert adapter_names == ["lora_TextOnly"]
    assert pipe.loaded == []
    assert pipe.unet.loaded == []
    assert pipe.unet.adapter_calls == []
    assert pipe.text_encoder.loaded == [("C:/loras/text_only.safetensors", "lora_TextOnly", "text_encoder")]
    assert pipe.text_encoder.adapter_calls == [(["lora_TextOnly"], [0.7])]


def test_apply_lora_adapters_supports_mixed_targets(monkeypatch):
    entries = {
        601: SimpleNamespace(
            lora_id=601,
            lora_model_family="sd15",
            name="Both",
            file_path="C:/loras/both.safetensors",
        ),
        602: SimpleNamespace(
            lora_id=602,
            lora_model_family="sd15",
            name="Unet",
            file_path="C:/loras/unet.safetensors",
        ),
        603: SimpleNamespace(
            lora_id=603,
            lora_model_family="sd15",
            name="Text",
            file_path="C:/loras/text.safetensors",
        ),
    }
    monkeypatch.setattr(lora_utils, "get_lora_entry", lambda lora_id: entries[lora_id])

    pipe = _DummyPipe()
    adapter_names, _ = lora_utils.apply_lora_adapters_with_validation(
        pipe,
        lora_adapters=[
            {"lora_id": 601, "strength": 0.9, "target": "both"},
            {"lora_id": 602, "strength": 0.6, "target": "unet"},
            {"lora_id": 603, "strength": 0.3, "target": "text_encoder"},
        ],
        expected_family="sd15",
        validate=False,
    )

    assert adapter_names == ["lora_Both", "lora_Unet", "lora_Text"]
    assert pipe.loaded == [("C:/loras/both.safetensors", "lora_Both")]
    assert pipe.adapter_calls == [(["lora_Both"], [0.9])]
    assert pipe.unet.loaded == [("C:/loras/unet.safetensors", "lora_Unet", "unet")]
    assert pipe.unet.adapter_calls == [(["lora_Unet"], [0.6])]
    assert pipe.text_encoder.loaded == [("C:/loras/text.safetensors", "lora_Text", "text_encoder")]
    assert pipe.text_encoder.adapter_calls == [(["lora_Text"], [0.3])]


def test_apply_lora_adapters_rejects_invalid_target(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=701,
            lora_model_family="sd15",
            name="BadTarget",
            file_path="C:/loras/bad_target.safetensors",
        ),
    )

    pipe = _DummyPipe()
    try:
        lora_utils.apply_lora_adapters_with_validation(
            pipe,
            lora_adapters=[{"lora_id": 701, "target": "te"}],
            expected_family="sd15",
            validate=False,
        )
        assert False, "Expected ValueError for invalid target."
    except ValueError as exc:
        assert str(exc) == "LoRA adapter field 'target' must be one of: both, unet, text_encoder."


def test_apply_lora_adapters_rejects_invalid_unet_scales(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=401,
            lora_model_family="sd15",
            name="InvalidUnet",
            file_path="C:/loras/invalid_unet.safetensors",
        ),
    )

    pipe = _DummyPipe()
    try:
        lora_utils.apply_lora_adapters_with_validation(
            pipe,
            lora_adapters=[{"lora_id": 401, "unet_scales": ["bad"]}],
            expected_family="sd15",
            validate=False,
        )
        assert False, "Expected ValueError for invalid unet_scales."
    except ValueError as exc:
        assert str(exc) == "LoRA adapter at index 0 field 'unet_scales' must be a number or an object."


def test_apply_lora_adapters_rejects_invalid_text_encoder_scales(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=402,
            lora_model_family="sd15",
            name="InvalidText",
            file_path="C:/loras/invalid_text.safetensors",
        ),
    )

    pipe = _DummyPipe()
    try:
        lora_utils.apply_lora_adapters_with_validation(
            pipe,
            lora_adapters=[{"lora_id": 402, "text_encoder_scales": {"layers.0": "high"}}],
            expected_family="sd15",
            validate=False,
        )
        assert False, "Expected ValueError for invalid text_encoder_scales."
    except ValueError as exc:
        assert str(exc) == "LoRA adapter at index 0 field 'text_encoder_scales'.layers.0 must be a number."


def test_apply_lora_adapters_text_encoder_scales_require_text_encoder(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_lora_entry",
        lambda _lora_id: SimpleNamespace(
            lora_id=403,
            lora_model_family="sd15",
            name="NeedsTextEncoder",
            file_path="C:/loras/needs_text_encoder.safetensors",
        ),
    )

    pipe = _DummyPipe()
    pipe.text_encoder = None
    try:
        lora_utils.apply_lora_adapters_with_validation(
            pipe,
            lora_adapters=[{"lora_id": 403, "text_encoder_scales": {"layers.0": 0.5}}],
            expected_family="sd15",
            validate=False,
        )
        assert False, "Expected ValueError when pipeline has no text_encoder."
    except ValueError as exc:
        assert (
            str(exc)
            == "LoRA adapter at index 0 provides text_encoder_scales but this pipeline has no text_encoder."
        )
