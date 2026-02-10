from types import SimpleNamespace

import backend.lora_utils as lora_utils


class _DummyPipe:
    def __init__(self):
        self.loaded: list[tuple[str, str]] = []
        self.adapter_calls: list[tuple[list[str], list[float | dict[str, float]]]] = []

    def load_lora_weights(self, file_path: str, adapter_name: str):
        self.loaded.append((file_path, adapter_name))

    def set_adapters(self, adapter_names: list[str], adapter_weights: list[float | dict[str, float]]):
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
