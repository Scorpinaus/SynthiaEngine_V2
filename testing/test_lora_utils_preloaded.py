import unittest

from backend.lora.utils import apply_lora_adapters_with_validation


class FakePipeline:
    def __init__(self):
        self.adapter_calls = []

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.adapter_calls.append((adapter_names, adapter_weights))


class LoraUtilsPreloadedAdapterTests(unittest.TestCase):
    def test_preloaded_adapters_activate_without_registry_loras(self):
        pipe = FakePipeline()

        adapter_names, coverage = apply_lora_adapters_with_validation(
            pipe,
            None,
            expected_family="sd15",
            preloaded_adapters=[("lcm_lora_sd15", 1.0)],
        )

        self.assertEqual(adapter_names, ["lcm_lora_sd15"])
        self.assertEqual(coverage, {})
        self.assertEqual(pipe.adapter_calls, [(["lcm_lora_sd15"], [1.0])])


if __name__ == "__main__":
    unittest.main()
