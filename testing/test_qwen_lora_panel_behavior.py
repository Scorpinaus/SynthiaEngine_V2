from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


class QwenLoraPanelBehaviorTests(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node.js is required for the LoRA panel behavior test.")
    def test_qwen_panel_renders_transformer_strength_and_builds_minimal_payload(self):
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

    toggle(name) {
        if (this.values.has(name)) {
            this.values.delete(name);
            return false;
        }
        this.values.add(name);
        return true;
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
        this.checked = false;
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
].forEach((id) => elements.set(id, new FakeElement()));

global.window = globalThis;
global.CustomEvent = class CustomEvent {
    constructor(name, options) {
        this.type = name;
        this.detail = options?.detail;
    }
};
const events = [];
window.dispatchEvent = (event) => {
    events.push(event);
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
                { lora_id: 101, name: "Qwen Plain" },
                {
                    lora_id: 102,
                    name: "Qwen Compatible",
                    compatibility: {
                        base_variants: ["qwen-image-2512"],
                        runtime_profile_kinds: ["qwen_image_lightning"],
                        supported_tasks: ["text2img", "img2img", "inpaint"],
                    },
                },
                {
                    lora_id: 103,
                    name: "Qwen Text Only",
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
                        adapter_strength: 1.0,
                    },
                },
                {
                    lora_id: 105,
                    name: "Lightning 8",
                    runtime_profile: {
                        kind: "qwen_image_lightning",
                        steps: 8,
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

function lastLightningEvent() {
    return events.filter((event) => event.type === "qwen-lightning-profile-changed").at(-1);
}

function lightningEventCount() {
    return events.filter((event) => event.type === "qwen-lightning-profile-changed").length;
}

function optionFor(loraId) {
    return elements.get("lora-select").children.find((option) => option.value === String(loraId));
}

function addSelected(loraId) {
    const select = elements.get("lora-select");
    select.value = String(loraId);
    elements.get("add-lora").listeners.click();
}

async function initQwen(taskType) {
    await window.LoraPanel.init({ apiBase: "/api", family: "qwen-image", taskType });
    window.LoraPanel.setSelectedAdapters([]);
}

const source = fs.readFileSync("frontend/components/lora_panel.js", "utf8");
vm.runInThisContext(source, { filename: "lora_panel.js" });

(async () => {
    await initQwen("qwen-image.text2img");
    const stackStatus = elements.get("lora-stack-status");
    assert.equal(stackStatus.classList.contains("is-hidden"), true);
    assert.match(optionFor(104).textContent, /Lightning · 4 steps/);
    assert.match(optionFor(105).textContent, /Lightning · 8 steps/);
    assert.match(optionFor(102).textContent, /Lightning-compatible · Qwen Image 2512/);

    window.LoraPanel.setSelectedAdapters([
        { lora_id: 101, strength: 0.65, target: "unet" },
        { lora_id: 102, strength: 0.35, target: "text_encoder" },
    ]);

    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 101, strength: 0.65 },
        { lora_id: 102, strength: 0.35 },
    ]);
    assert.deepEqual(window.LoraPanel.getSummary(), {
        available: 5,
        selected: 2,
        family: "qwen-image",
        weightMode: "basic",
    });
    assert.equal(optionFor(104).disabled, true);
    assert.equal(optionFor(104).title, "Remove extra standard LoRAs before selecting Lightning");
    assert.equal(stackStatus.classList.contains("is-hidden"), true);

    const qwenItems = elements.get("lora-list").children;
    assert.equal(qwenItems.length, 2);
    qwenItems.forEach((item) => {
        const rendered = descendants(item);
        const strength = rendered.find((element) => hasClass(element, "lora-strength"));
        assert.match(strength.innerHTML, /Qwen transformer/);
        assert.equal(rendered.filter((element) => hasClass(element, "lora-target")).length, 0);
        assert.equal(
            rendered.filter(
                (element) => element.tagName === "INPUT" && element.type === "range",
            ).length,
            1,
        );
    });

    window.LoraPanel.setSelectedAdapters([]);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), []);
    assert.equal(elements.get("lora-empty").classList.contains("is-hidden"), false);
    assert.equal(stackStatus.classList.contains("is-hidden"), true);

    addSelected(104);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 104, strength: 1 },
    ]);
    const lightning4 = descendants(elements.get("lora-list").children[0]);
    assert.equal(
        lightning4.find((element) => hasClass(element, "lora-profile-label")).textContent,
        "Lightning · 4 steps",
    );
    assert.equal(
        lightning4.find((element) => element.tagName === "INPUT" && element.type === "range").disabled,
        true,
    );
    assert.equal(lastLightningEvent().detail.profile.steps, 4);
    assert.equal(optionFor(105).disabled, true);
    assert.equal(optionFor(105).title, "Only one Lightning adapter is allowed");
    assert.equal(optionFor(101).disabled, true);
    assert.equal(optionFor(101).title, "No Lightning compatibility metadata");
    assert.equal(optionFor(102).disabled, false);
    assert.equal(stackStatus.classList.contains("is-hidden"), true);

    addSelected(102);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 104, strength: 1 },
        { lora_id: 102, strength: 0.8 },
    ]);
    assert.equal(optionFor(103).disabled, true);
    assert.equal(optionFor(103).title, "Only one Lightning-compatible companion is allowed");
    assert.equal(stackStatus.textContent, "Experimental stack: Lightning + 1 LoRA");
    assert.equal(stackStatus.classList.contains("is-hidden"), false);
    const mixedLightning = descendants(elements.get("lora-list").children[0]);
    const mixedCompanion = descendants(elements.get("lora-list").children[1]);
    assert.equal(
        mixedLightning.find((element) => hasClass(element, "lora-profile-label")).textContent,
        "Lightning · 4 steps",
    );
    assert.equal(
        mixedCompanion.find((element) => hasClass(element, "lora-profile-label")).textContent,
        "Lightning-compatible · Qwen Image 2512",
    );
    assert.equal(
        mixedLightning.find((element) => element.tagName === "INPUT" && element.type === "range").disabled,
        true,
    );
    const companionSlider = mixedCompanion.find(
        (element) => element.tagName === "INPUT" && element.type === "range",
    );
    assert.equal(companionSlider.disabled, false);
    companionSlider.value = "0.5";
    companionSlider.listeners.input({ target: companionSlider });
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 104, strength: 1 },
        { lora_id: 102, strength: 0.5 },
    ]);

    const eventsBeforeCompanionRemoval = lightningEventCount();
    descendants(elements.get("lora-list").children[1])
        .find((element) => hasClass(element, "lora-remove"))
        .listeners.click();
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [{ lora_id: 104, strength: 1 }]);
    assert.equal(optionFor(103).disabled, false);
    assert.equal(stackStatus.classList.contains("is-hidden"), true);
    assert.equal(lightningEventCount(), eventsBeforeCompanionRemoval);

    addSelected(102);
    assert.equal(stackStatus.classList.contains("is-hidden"), false);
    const eventsBeforeLightningRemoval = lightningEventCount();
    descendants(elements.get("lora-list").children[0])
        .find((element) => hasClass(element, "lora-remove"))
        .listeners.click();
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [{ lora_id: 102, strength: 0.8 }]);
    assert.equal(stackStatus.classList.contains("is-hidden"), true);
    assert.equal(
        descendants(elements.get("lora-list").children[0])
            .find((element) => element.tagName === "INPUT" && element.type === "range").disabled,
        false,
    );
    assert.equal(lightningEventCount(), eventsBeforeLightningRemoval + 1);
    assert.equal(lastLightningEvent().detail.profile, null);

    await initQwen("qwen-image.img2img");
    window.LoraPanel.setSelectedAdapters([{ lora_id: 102, strength: 0.5 }]);
    assert.equal(optionFor(104).disabled, false);
    addSelected(104);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 102, strength: 0.5 },
        { lora_id: 104, strength: 1 },
    ]);
    assert.equal(lastLightningEvent().detail.profile.steps, 4);
    await window.LoraPanel.init({
        apiBase: "/api",
        family: "qwen-image",
        taskType: "qwen-image.img2img",
    });
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 102, strength: 0.5 },
        { lora_id: 104, strength: 1 },
    ]);

    await initQwen("qwen-image.img2img");
    window.LoraPanel.setSelectedAdapters([{ lora_id: 103, strength: 0.5 }]);
    assert.equal(optionFor(104).disabled, true);
    assert.equal(
        optionFor(104).title,
        "Selected standard LoRA is not Lightning-compatible for this task",
    );

    for (const [taskType, compatibilityTask, textOnlyEnabled] of [
        ["qwen-image.text2img", "text2img", true],
        ["qwen-image.img2img", "img2img", false],
        ["qwen-image.inpaint", "inpaint", false],
    ]) {
        await initQwen(taskType);
        addSelected(104);
        assert.equal(optionFor(101).disabled, true);
        assert.equal(optionFor(101).title, "No Lightning compatibility metadata");
        assert.equal(optionFor(102).disabled, false);
        assert.equal(optionFor(103).disabled, !textOnlyEnabled);
        if (!textOnlyEnabled) {
            assert.equal(optionFor(103).title, `Not compatible with Qwen task ${compatibilityTask}`);
        }
    }

    await window.LoraPanel.init({ apiBase: "/api", family: "sdxl" });
    window.LoraPanel.setSelectedAdapters([
        { lora_id: 101, strength: 0.5, target: "unet" },
        { lora_id: 102, strength: 0.25, target: "text_encoder" },
    ]);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 101, strength: 0.5, target: "unet" },
        { lora_id: 102, strength: 0.25, target: "text_encoder" },
    ]);
    const sdxlRendered = descendants(elements.get("lora-list").children[0]);
    assert.equal(sdxlRendered.filter((element) => hasClass(element, "lora-target")).length, 1);
    assert.equal(stackStatus.classList.contains("is-hidden"), true);
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
