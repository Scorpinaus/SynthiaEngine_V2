from dataclasses import FrozenInstanceError

import pytest

from backend.lora.registry import LoraCompatibility, LoraRegistryEntry
from backend.qwen_image.lightning import (
    get_qwen_image_base_variant,
    resolve_qwen_image_lightning_profile,
)
from backend.registries.model import ModelRegistryEntry
from backend.workflow.schema_input import QwenImageLoraAdapter


def _model(*, name="Qwen-Image-2512-SDNQ-4bit-dynamic", link=None, family="qwen-image"):
    return ModelRegistryEntry(
        name=name,
        family=family,
        model_type="diffusers",
        location_type="local",
        model_id=1,
        version="local",
        link=link or rf"D:\diffusion\diffusers\{name}",
    )


def _profile(steps=4):
    return {
        "kind": "qwen_image_lightning",
        "base_variant": "qwen-image-2512",
        "steps": steps,
        "true_cfg_scale": 1.0,
        "scheduler_profile": "qwen_image_lightning_shift3",
        "adapter_strength": 1.0,
        "supported_tasks": ["text2img"],
    }


def _compatibility(*, supported_tasks=None):
    return {
        "base_variants": ["qwen-image-2512"],
        "runtime_profile_kinds": ["qwen_image_lightning"],
        "supported_tasks": supported_tasks or ["text2img", "img2img", "inpaint"],
    }


def _lora(lora_id, *, runtime_profile=None, compatibility=None, family="qwen-image", lora_type="lora"):
    return LoraRegistryEntry(
        lora_id=lora_id,
        lora_model_family=family,
        lora_type=lora_type,
        lora_location="local",
        file_path=rf"D:\diffusion\loras\{lora_id}.safetensors",
        runtime_profile=runtime_profile,
        compatibility=compatibility,
    )


def _resolve(adapters, entries, *, model=None, task="text2img", steps=4, true_cfg_scale=1.0):
    lookup_calls = []

    def lookup(lora_id):
        lookup_calls.append(lora_id)
        try:
            return entries[lora_id]
        except KeyError as exc:
            raise ValueError(f"LoRA with id {lora_id} not found.") from exc

    result = resolve_qwen_image_lightning_profile(
        adapters,
        model or _model(),
        task,
        steps,
        true_cfg_scale,
        lookup_lora_entry=lookup,
    )
    return result, lookup_calls


def test_base_variant_detection_handles_sdnq_hub_edit_and_other_models():
    assert get_qwen_image_base_variant(_model()) == "qwen-image-2512"
    assert get_qwen_image_base_variant(
        _model(name="Local Qwen", link="Disty0/Qwen-Image-2512-SDNQ-4bit-dynamic")
    ) == "qwen-image-2512"
    assert get_qwen_image_base_variant(_model(name="Qwen-Image-Edit-2511")) == "qwen-image-edit"
    assert get_qwen_image_base_variant(_model(name="Qwen-Image-2509")) == "qwen-image-other"
    with pytest.raises(ValueError, match="not a Qwen-Image model"):
        get_qwen_image_base_variant(_model(family="sdxl"))


def test_resolver_keeps_no_adapter_and_normal_lora_paths_and_result_immutable():
    empty, empty_calls = _resolve(None, {})
    assert empty.adapters == ()
    assert empty.lightning_profile is None
    assert empty.task == "text2img"
    assert empty_calls == []

    normal_entry = _lora(10)
    resolved, lookup_calls = _resolve(
        [{"lora_id": 10, "strength": 0.65, "target": "both"}], {10: normal_entry}
    )
    assert lookup_calls == [10]
    assert resolved.lightning_profile is None
    assert resolved.adapters[0].entry is normal_entry
    assert resolved.adapters[0].strength == 0.65
    with pytest.raises(FrozenInstanceError):
        resolved.steps = 8
    with pytest.raises(FrozenInstanceError):
        resolved.adapters[0].strength = 1.0


@pytest.mark.parametrize("steps", [4, 8])
def test_resolver_accepts_exact_four_and_eight_step_lightning_profiles(steps):
    entry = _lora(20 + steps, runtime_profile=_profile(steps))
    resolved, lookup_calls = _resolve(
        [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}],
        {entry.lora_id: entry},
        steps=steps,
    )
    assert lookup_calls == [entry.lora_id]
    assert resolved.lightning_profile == entry.runtime_profile
    assert resolved.steps == steps


@pytest.mark.parametrize(("task", "steps"), [("img2img", 4), ("qwen-image.inpaint", 8)])
def test_resolver_accepts_lightning_for_experimental_image_tasks(task, steps):
    entry = _lora(70 + steps, runtime_profile=_profile(steps))
    resolved, _lookup_calls = _resolve(
        [{"lora_id": entry.lora_id, "strength": 1.0, "target": "both"}],
        {entry.lora_id: entry},
        task=task,
        steps=steps,
    )
    assert resolved.task in {"img2img", "inpaint"}


def test_resolver_accepts_mapping_and_pydantic_adapter_inputs_and_resolves_all_ids():
    first = _lora(31)
    second = _lora(32)
    resolved, lookup_calls = _resolve(
        [
            {"lora_id": 31, "strength": 0.4, "target": "both"},
            QwenImageLoraAdapter(lora_id=32, strength=0.6),
        ],
        {31: first, 32: second},
    )
    assert lookup_calls == [31, 32]
    assert [adapter.entry for adapter in resolved.adapters] == [first, second]


def test_resolver_keeps_standard_only_multi_lora_behavior_unchanged():
    entries = {lora_id: _lora(lora_id) for lora_id in (33, 34, 35)}
    resolved, lookup_calls = _resolve(
        [
            {"lora_id": 33, "strength": 0.0},
            {"lora_id": 34, "strength": 0.5},
            {"lora_id": 35, "strength": 1.0},
        ],
        entries,
    )
    assert lookup_calls == [33, 34, 35]
    assert resolved.lightning_profile is None
    assert [adapter.lora_id for adapter in resolved.adapters] == [33, 34, 35]
    assert [adapter.strength for adapter in resolved.adapters] == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    ("adapters", "message"),
    [
        ({"lora_id": 1}, "lora_adapters must be a list"),
        ([object()], "must be a mapping"),
        ([{"strength": 1.0}], "invalid lora_id"),
        ([{"lora_id": "1", "strength": 1.0}], "invalid lora_id"),
        ([{"lora_id": True, "strength": 1.0}], "invalid lora_id"),
        ([{"lora_id": 1, "strength": "high"}], "strength must be numeric"),
        ([{"lora_id": 1, "strength": True}], "strength must be numeric"),
        ([{"lora_id": 1, "strength": -0.1}], "from 0.0 through 1.0"),
        ([{"lora_id": 1, "strength": 1.1}], "from 0.0 through 1.0"),
    ],
)
def test_resolver_rejects_malformed_adapter_inputs(adapters, message):
    with pytest.raises(ValueError, match=message):
        _resolve(adapters, {})


@pytest.mark.parametrize(
    ("task", "steps", "true_cfg_scale", "message"),
    [
        ("qwen-image.unknown", 4, 1.0, "Unsupported Qwen Image task"),
        ("text2img", True, 1.0, "requested steps must be an integer"),
        ("text2img", 4.0, 1.0, "requested steps must be an integer"),
        ("text2img", 4, True, "requested true_cfg_scale must be numeric"),
        ("text2img", 4, "1.0", "requested true_cfg_scale must be numeric"),
    ],
)
def test_resolver_rejects_invalid_task_and_requested_settings(task, steps, true_cfg_scale, message):
    with pytest.raises(ValueError, match=message):
        _resolve(None, {}, task=task, steps=steps, true_cfg_scale=true_cfg_scale)


def test_resolver_rejects_unknown_family_type_and_target():
    with pytest.raises(ValueError, match="could not be resolved"):
        _resolve([{"lora_id": 40, "strength": 1.0}], {})
    with pytest.raises(ValueError, match="incompatible family"):
        _resolve([{"lora_id": 41, "strength": 1.0}], {41: _lora(41, family="sdxl")})
    with pytest.raises(ValueError, match="unsupported type"):
        _resolve([{"lora_id": 42, "strength": 1.0}], {42: _lora(42, lora_type="lycoris")})
    with pytest.raises(ValueError, match="target must be 'both'"):
        _resolve([{"lora_id": 43, "strength": 1.0, "target": "unet"}], {43: _lora(43)})


@pytest.mark.parametrize(
    ("steps", "companion_strength", "lightning_first"),
    [
        (4, 0.0, True),
        (4, 0.5, False),
        (4, 1.0, True),
        (8, 0.0, False),
        (8, 0.5, True),
        (8, 1.0, False),
    ],
)
def test_resolver_allows_declared_compatible_lightning_companions(
    steps,
    companion_strength,
    lightning_first,
):
    lightning = _lora(100 + steps, runtime_profile=_profile(steps))
    companion = _lora(
        200 + steps,
        compatibility=_compatibility(supported_tasks=["img2img"]),
    )
    lightning_adapter = {"lora_id": lightning.lora_id, "strength": 1.0}
    companion_adapter = {"lora_id": companion.lora_id, "strength": companion_strength}
    adapters = [lightning_adapter, companion_adapter] if lightning_first else [companion_adapter, lightning_adapter]

    resolved, lookup_calls = _resolve(
        adapters,
        {lightning.lora_id: lightning, companion.lora_id: companion},
        task="qwen-image.img2img",
        steps=steps,
    )

    assert lookup_calls == [adapter["lora_id"] for adapter in adapters]
    assert [adapter.lora_id for adapter in resolved.adapters] == [adapter["lora_id"] for adapter in adapters]
    assert [adapter.strength for adapter in resolved.adapters] == [adapter["strength"] for adapter in adapters]
    assert resolved.lightning_profile == lightning.runtime_profile
    assert resolved.task == "img2img"
    assert resolved.steps == steps


def test_resolver_rejects_two_lightning_adapters_and_two_standard_companions():
    lightning_a = _lora(51, runtime_profile=_profile())
    lightning_b = _lora(52, runtime_profile=_profile())
    with pytest.raises(ValueError, match="supports one Lightning adapter; received IDs: 51, 52"):
        _resolve(
            [{"lora_id": 51, "strength": 1.0}, {"lora_id": 52, "strength": 1.0}],
            {51: lightning_a, 52: lightning_b},
        )
    with pytest.raises(ValueError, match="at most one standard companion LoRA; received IDs: 53, 54"):
        _resolve(
            [
                {"lora_id": 51, "strength": 1.0},
                {"lora_id": 53, "strength": 0.5},
                {"lora_id": 54, "strength": 0.5},
            ],
            {
                51: lightning_a,
                53: _lora(53, compatibility=_compatibility()),
                54: _lora(54, compatibility=_compatibility()),
            },
        )


def test_resolver_rejects_lightning_companion_without_compatibility():
    lightning = _lora(55, runtime_profile=_profile())
    companion = _lora(56)
    with pytest.raises(ValueError, match="companion LoRA 56 has no declared compatibility metadata"):
        _resolve(
            [{"lora_id": 55, "strength": 1.0}, {"lora_id": 56, "strength": 0.5}],
            {55: lightning, 56: companion},
        )


@pytest.mark.parametrize(
    ("field_name", "task", "message"),
    [
        ("base_variants", "text2img", "missing base variant 'qwen-image-2512'"),
        ("runtime_profile_kinds", "text2img", "missing runtime profile kind 'qwen_image_lightning'"),
        ("supported_tasks", "qwen-image.inpaint", "missing task 'inpaint'"),
    ],
)
def test_resolver_rejects_lightning_companion_with_missing_compatibility_value(
    field_name,
    task,
    message,
):
    lightning = _lora(57, runtime_profile=_profile())
    companion = _lora(58, compatibility=_compatibility())
    companion.compatibility = LoraCompatibility.model_construct(
        **{field: [] if field == field_name else value for field, value in _compatibility().items()}
    )
    with pytest.raises(ValueError, match=message):
        _resolve(
            [{"lora_id": 57, "strength": 1.0}, {"lora_id": 58, "strength": 0.5}],
            {57: lightning, 58: companion},
            task=task,
        )


def test_resolver_rejects_invalid_lightning_task_model_steps_cfg_and_strength():
    lightning = _lora(61, runtime_profile=_profile())
    adapter = [{"lora_id": 61, "strength": 1.0, "target": "both"}]
    with pytest.raises(ValueError, match="requires base variant 'qwen-image-2512'; selected 'qwen-image-edit'"):
        _resolve(adapter, {61: lightning}, model=_model(name="Qwen-Image-Edit-2511"))
    with pytest.raises(ValueError, match="requires steps=4; received 8"):
        _resolve(adapter, {61: lightning}, steps=8)
    with pytest.raises(ValueError, match="requires true_cfg_scale=1.0; received 2.0"):
        _resolve(adapter, {61: lightning}, true_cfg_scale=2.0)
    with pytest.raises(ValueError, match="requires strength=1.0; received 0.8"):
        _resolve([{"lora_id": 61, "strength": 0.8}], {61: lightning})
