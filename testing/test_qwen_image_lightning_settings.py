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

    @unittest.skipUnless(NODE, "Node.js is required for the Qwen Lightning settings test.")
    def test_preset_hydration_emits_profile_events_for_the_lora_panel(self):
        node_test = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeClassList {
    constructor() {
        this.values = new Set();
    }

    add(...names) {
        names.forEach((name) => this.values.add(name));
    }

    remove(...names) {
        names.forEach((name) => this.values.delete(name));
    }

    contains(name) {
        return this.values.has(name);
    }
}

class FakeElement {
    constructor(tagName = "div") {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.className = "";
        this.classList = new FakeClassList();
        this.listeners = {};
        this._innerHTML = "";
        this.textContent = "";
        this.value = "";
        this.disabled = false;
    }

    set innerHTML(value) {
        this._innerHTML = String(value);
        if (value === "") {
            this.children = [];
        }
    }

    get innerHTML() {
        return this._innerHTML;
    }

    append(...children) {
        children.forEach((child) => this.appendChild(child));
    }

    appendChild(child) {
        this.children.push(child);
        child.parentNode = this;
        return child;
    }

    addEventListener(name, callback) {
        this.listeners[name] = callback;
    }
}

const elements = new Map();
[
    "lora-panel-root",
    "lora-toggle",
    "add-lora",
    "lora-weight-mode-row",
    "lora-weight-mode-basic",
    "lora-weight-mode-advanced",
    "lora-list",
    "lora-empty",
    "lora-stack-status",
    "lora-select",
    "steps",
    "true_cfg",
].forEach((id) => elements.set(id, new FakeElement()));
elements.get("steps").value = "50";
elements.get("true_cfg").value = "4.0";

const listeners = new Map();
const events = [];
global.window = globalThis;
global.CustomEvent = class CustomEvent {
    constructor(name, options) {
        this.type = name;
        this.detail = options?.detail;
    }
};
window.addEventListener = (name, listener) => {
    const callbacks = listeners.get(name) ?? [];
    callbacks.push(listener);
    listeners.set(name, callbacks);
};
window.dispatchEvent = (event) => {
    events.push(event);
    (listeners.get(event.type) ?? []).forEach((listener) => listener(event));
    return true;
};
global.document = {
    currentScript: { src: "https://example.test/components/lora_panel.js" },
    getElementById: (id) => elements.get(id) ?? null,
    createElement: (tagName) => new FakeElement(tagName),
};
global.fetch = async (url) => {
    const requestUrl = String(url);
    if (requestUrl.includes("lora_panel.html")) {
        return { ok: true, text: async () => "" };
    }
    if (requestUrl.includes("/lora-models?family=")) {
        return {
            ok: true,
            json: async () => [
                {
                    lora_id: 102,
                    name: "Qwen Compatible",
                    compatibility: {
                        base_variants: ["qwen-image-2512"],
                        runtime_profile_kinds: ["qwen_image_lightning"],
                        supported_tasks: ["text2img"],
                    },
                },
                {
                    lora_id: 104,
                    name: "Lightning 4",
                    runtime_profile: {
                        kind: "qwen_image_lightning",
                        steps: 4,
                        true_cfg_scale: 1.0,
                        adapter_strength: 1.0,
                    },
                },
            ],
        };
    }
    throw new Error(`Unexpected request: ${requestUrl}`);
};
console.debug = () => {};

function descendants(node) {
    return node.children.flatMap((child) => [child, ...descendants(child)]);
}

function hasClass(element, className) {
    return String(element.className).split(/\s+/).includes(className);
}

function profileEvents() {
    return events.filter((event) => event.type === "qwen-lightning-profile-changed");
}

const panelSource = fs.readFileSync("frontend/components/lora_panel.js", "utf8");
const settingsSource = fs.readFileSync("frontend/qwen_image/lightning_settings.js", "utf8");
vm.runInThisContext(panelSource, { filename: "lora_panel.js" });
vm.runInThisContext(settingsSource, { filename: "lightning_settings.js" });

(async () => {
    await window.LoraPanel.init({
        apiBase: "/api",
        family: "qwen-image",
        taskType: "qwen-image.text2img",
    });
    const controller = window.QwenImageLightningSettings.create({
        taskType: "qwen-image.text2img",
    });
    const inputs = {
        steps: elements.get("steps"),
        trueCfg: elements.get("true_cfg"),
    };
    const mixedPreset = [
        { lora_id: 102, strength: 0.35 },
        { lora_id: 104, strength: 0.2 },
    ];

    function applyPreset(adapters, { steps, trueCfg }) {
        const eventCount = profileEvents().length;
        controller.beforeApplySettings();
        inputs.steps.value = steps;
        inputs.trueCfg.value = trueCfg;
        window.LoraPanel.setSelectedAdapters(adapters);
        controller.afterApplySettings();
        return profileEvents().slice(eventCount);
    }

    let emitted = applyPreset(mixedPreset, { steps: "31", trueCfg: "3.1" });
    assert.equal(emitted.length, 1);
    assert.equal(emitted[0].detail.profile.steps, 4);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 102, strength: 0.35 },
        { lora_id: 104, strength: 1 },
    ]);
    assert.equal(inputs.steps.value, "4");
    assert.equal(inputs.trueCfg.value, "1");
    assert.equal(inputs.steps.disabled, true);
    assert.equal(inputs.trueCfg.disabled, true);

    emitted = applyPreset(mixedPreset, { steps: "32", trueCfg: "3.2" });
    assert.equal(emitted.length, 1);
    assert.equal(emitted[0].detail.profile.steps, 4);
    assert.equal(inputs.steps.value, "4");
    assert.equal(inputs.trueCfg.value, "1");

    let eventCount = profileEvents().length;
    descendants(elements.get("lora-list").children[0])
        .find((element) => hasClass(element, "lora-remove"))
        .listeners.click();
    assert.equal(profileEvents().length, eventCount);
    assert.equal(inputs.steps.disabled, true);
    assert.equal(inputs.trueCfg.disabled, true);

    emitted = applyPreset(mixedPreset, { steps: "33", trueCfg: "3.3" });
    assert.equal(emitted.length, 1);
    eventCount = profileEvents().length;
    descendants(elements.get("lora-list").children[1])
        .find((element) => hasClass(element, "lora-remove"))
        .listeners.click();
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [{ lora_id: 102, strength: 0.35 }]);
    assert.equal(profileEvents().length, eventCount + 1);
    assert.equal(profileEvents().at(-1).detail.profile, null);
    assert.equal(inputs.steps.value, "50");
    assert.equal(inputs.trueCfg.value, "4.0");
    assert.equal(inputs.steps.disabled, false);
    assert.equal(inputs.trueCfg.disabled, false);

    emitted = applyPreset([{ lora_id: 102, strength: 0.6 }], {
        steps: "29",
        trueCfg: "2.5",
    });
    assert.equal(emitted.length, 1);
    assert.equal(emitted[0].detail.profile, null);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [{ lora_id: 102, strength: 0.6 }]);
    assert.equal(inputs.steps.value, "29");
    assert.equal(inputs.trueCfg.value, "2.5");
    assert.equal(inputs.steps.disabled, false);
    assert.equal(inputs.trueCfg.disabled, false);
})();
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
