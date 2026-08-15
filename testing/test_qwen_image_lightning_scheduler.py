import math
from types import SimpleNamespace

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler

from backend.lora.registry import LoraRuntimeProfile
from backend.qwen_image import lightning


EXPECTED_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _profile(steps=4):
    return LoraRuntimeProfile(
        kind="qwen_image_lightning",
        base_variant="qwen-image-2512",
        steps=steps,
        true_cfg_scale=1.0,
        scheduler_profile="qwen_image_lightning_shift3",
        adapter_strength=1.0,
        supported_tasks=["text2img"],
    )


def test_lightning_scheduler_config_has_exact_fixed_values_and_keys():
    config = lightning.get_qwen_image_lightning_scheduler_config()

    assert set(config) == set(EXPECTED_CONFIG)
    assert config == EXPECTED_CONFIG
    assert config["base_shift"] == pytest.approx(math.log(3))
    assert config["max_shift"] == pytest.approx(math.log(3))


def test_lightning_scheduler_config_is_a_fresh_copy_from_an_immutable_source():
    first = lightning.get_qwen_image_lightning_scheduler_config()
    second = lightning.get_qwen_image_lightning_scheduler_config()

    assert first is not second
    first["shift"] = 9.0
    assert second["shift"] == 1.0
    with pytest.raises(TypeError):
        lightning._QWEN_IMAGE_LIGHTNING_SCHEDULER_CONFIG["shift"] = 9.0


@pytest.mark.parametrize("steps", [4, 8])
def test_lightning_profiles_construct_the_same_fixed_flowmatch_scheduler(steps):
    scheduler = lightning.create_qwen_image_lightning_scheduler(_profile(steps))

    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    for key, value in EXPECTED_CONFIG.items():
        assert scheduler.config[key] == value


def test_four_and_eight_step_lightning_profiles_share_one_scheduler_config():
    four_step_scheduler = lightning.create_qwen_image_lightning_scheduler(_profile(4))
    eight_step_scheduler = lightning.create_qwen_image_lightning_scheduler(_profile(8))

    assert dict(four_step_scheduler.config) == dict(eight_step_scheduler.config)


def test_lightning_constructor_rejects_another_scheduler_profile():
    fake_profile = SimpleNamespace(scheduler_profile="another-scheduler")

    with pytest.raises(ValueError, match="requires scheduler_profile 'qwen_image_lightning_shift3'"):
        lightning.create_qwen_image_lightning_scheduler(fake_profile)


def test_scheduler_selector_uses_the_base_path_without_mutating_the_pipeline(monkeypatch):
    pipe = SimpleNamespace(scheduler="base-scheduler")
    expected_scheduler = object()
    calls = []

    def create_base_scheduler(name, received_pipe):
        calls.append((name, received_pipe))
        return expected_scheduler

    monkeypatch.setattr(lightning, "create_scheduler", create_base_scheduler)
    selected = lightning.select_qwen_image_scheduler(
        SimpleNamespace(lightning_profile=None), pipe
    )

    assert selected is expected_scheduler
    assert calls == [("flowmatch_euler", pipe)]
    assert pipe.scheduler == "base-scheduler"


def test_lightning_selector_does_not_read_or_assign_the_base_scheduler():
    class SentinelPipe:
        @property
        def scheduler(self):
            raise AssertionError("Lightning selection must not read the base scheduler config.")

        @scheduler.setter
        def scheduler(self, _value):
            raise AssertionError("Lightning selection must not assign pipe.scheduler.")

    selected = lightning.select_qwen_image_scheduler(
        SimpleNamespace(lightning_profile=_profile()), SentinelPipe()
    )

    assert isinstance(selected, FlowMatchEulerDiscreteScheduler)
    assert selected.config["base_shift"] == pytest.approx(math.log(3))
