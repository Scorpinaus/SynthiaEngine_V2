from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


class QwenImageLightningSettingsTests(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node.js is required for the Qwen Lightning settings test.")
    def test_lightning_settings_lock_restore_and_preset_hydration(self):
        node_test = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const listeners = new Map();
const inputs = new Map([
    ["steps", { value: "50", disabled: false }],
    ["true_cfg", { value: "4.0", disabled: false }],
    ["strength", { value: "0.6", disabled: false }],
]);
global.window = globalThis;
window.addEventListener = (name, listener) => listeners.set(name, listener);
window.dispatchEvent = (event) => listeners.get(event.type)?.(event);
global.document = { getElementById: (id) => inputs.get(id) ?? null };

const source = fs.readFileSync("frontend/qwen_image/lightning_settings.js", "utf8");
vm.runInThisContext(source, { filename: "lightning_settings.js" });

const lightning4 = {
    kind: "qwen_image_lightning",
    steps: 4,
    true_cfg_scale: 1.0,
};
const lightning8 = { ...lightning4, steps: 8 };
const dispatchProfile = (profile) => window.dispatchEvent({
    type: "qwen-lightning-profile-changed",
    detail: { profile },
});

const controller = window.QwenImageLightningSettings.create({
    taskType: "qwen-image.text2img",
});
dispatchProfile(lightning4);
assert.equal(inputs.get("steps").value, "4");
assert.equal(inputs.get("true_cfg").value, "1");
assert.equal(inputs.get("steps").disabled, true);
assert.equal(inputs.get("true_cfg").disabled, true);

dispatchProfile(lightning8);
assert.equal(inputs.get("steps").value, "8");
dispatchProfile(null);
assert.equal(inputs.get("steps").value, "50");
assert.equal(inputs.get("true_cfg").value, "4.0");
assert.equal(inputs.get("steps").disabled, false);

inputs.get("steps").value = "30";
inputs.get("true_cfg").value = "2.0";
controller.beforeApplySettings();
inputs.get("steps").value = "4";
inputs.get("true_cfg").value = "1";
dispatchProfile(lightning4);
controller.afterApplySettings();
assert.equal(inputs.get("steps").disabled, true);
dispatchProfile(null);
assert.equal(inputs.get("steps").value, "30");
assert.equal(inputs.get("true_cfg").value, "2.0");

dispatchProfile(lightning4);
controller.beforeApplySettings();
inputs.get("steps").value = "35";
inputs.get("true_cfg").value = "3.0";
dispatchProfile(null);
controller.afterApplySettings();
assert.equal(inputs.get("steps").disabled, false);
assert.equal(inputs.get("steps").value, "35");
assert.equal(inputs.get("true_cfg").value, "3.0");

inputs.get("steps").value = "40";
inputs.get("true_cfg").value = "3.5";
const imgController = window.QwenImageLightningSettings.create({
    taskType: "qwen-image.img2img",
});
dispatchProfile(lightning8);
assert.equal(inputs.get("steps").value, "8");
assert.equal(inputs.get("true_cfg").value, "1");
assert.equal(inputs.get("strength").value, "0.6");
assert.equal(inputs.get("strength").disabled, false);
dispatchProfile(null);
assert.equal(inputs.get("steps").value, "40");
assert.equal(inputs.get("true_cfg").value, "3.5");

inputs.get("steps").value = "45";
inputs.get("true_cfg").value = "3.75";
const inpaintController = window.QwenImageLightningSettings.create({
    taskType: "qwen-image.inpaint",
});
dispatchProfile(lightning4);
assert.equal(inputs.get("steps").value, "4");
assert.equal(inputs.get("true_cfg").value, "1");
assert.equal(inputs.get("strength").value, "0.6");
assert.equal(inputs.get("strength").disabled, false);
dispatchProfile(null);
assert.equal(inputs.get("steps").value, "45");
assert.equal(inputs.get("true_cfg").value, "3.75");
"""

        result = subprocess.run(
            [NODE, "-e", node_test],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
