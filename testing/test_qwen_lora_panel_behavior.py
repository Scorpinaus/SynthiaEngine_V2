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
                { lora_id: 101, name: "Qwen Detail" },
                { lora_id: 102, name: "Qwen Style" },
                {
                    lora_id: 103,
                    name: "Lightning 4",
                    runtime_profile: {
                        kind: "qwen_image_lightning",
                        steps: 4,
                        adapter_strength: 1.0,
                    },
                },
                {
                    lora_id: 104,
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

const source = fs.readFileSync("frontend/components/lora_panel.js", "utf8");
vm.runInThisContext(source, { filename: "lora_panel.js" });

(async () => {
    await window.LoraPanel.init({ apiBase: "/api", family: "qwen-image" });
    window.LoraPanel.setSelectedAdapters([
        { lora_id: 101, strength: 0.65, target: "unet" },
        { lora_id: 102, strength: 0.35, target: "text_encoder" },
    ]);

    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 101, strength: 0.65 },
        { lora_id: 102, strength: 0.35 },
    ]);
    assert.deepEqual(window.LoraPanel.getSummary(), {
        available: 4,
        selected: 2,
        family: "qwen-image",
        weightMode: "basic",
    });

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

    window.LoraPanel.setSelectedAdapters([{ lora_id: 103, strength: 0.25 }]);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 103, strength: 1 },
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

    window.LoraPanel.setSelectedAdapters([{ lora_id: 104, strength: 0.4 }]);
    const lightning8 = descendants(elements.get("lora-list").children[0]);
    assert.equal(
        lightning8.find((element) => hasClass(element, "lora-profile-label")).textContent,
        "Lightning · 8 steps",
    );
    assert.equal(lastLightningEvent().detail.profile.steps, 8);
    window.LoraPanel.setSelectedAdapters([{ lora_id: 101, strength: 0.5 }]);
    assert.equal(lastLightningEvent().detail.profile, null);

    await window.LoraPanel.init({
        apiBase: "/api",
        family: "qwen-image",
        taskType: "qwen-image.img2img",
    });
    const options = elements.get("lora-select").children;
    assert.equal(options.find((option) => option.value === "103").disabled, false);
    window.LoraPanel.setSelectedAdapters([{ lora_id: 103, strength: 1 }]);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [{ lora_id: 103, strength: 1 }]);

    await window.LoraPanel.init({ apiBase: "/api", family: "sdxl" });
    window.LoraPanel.setSelectedAdapters([
        { lora_id: 101, strength: 0.5, target: "unet" },
    ]);
    assert.deepEqual(window.LoraPanel.getSelectedAdapters(), [
        { lora_id: 101, strength: 0.5, target: "unet" },
    ]);
    const sdxlRendered = descendants(elements.get("lora-list").children[0]);
    assert.equal(sdxlRendered.filter((element) => hasClass(element, "lora-target")).length, 1);
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
