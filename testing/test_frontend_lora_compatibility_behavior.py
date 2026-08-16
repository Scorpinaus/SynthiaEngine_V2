from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


class FrontendLoraCompatibilityBehaviorTests(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node.js is required for the LoRA compatibility behavior test.")
    def test_add_and_edit_compatibility_payload_behavior(self):
        node_test = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
    constructor(value = "") {
        this.value = value;
        this.checked = false;
        this.disabled = false;
        this.hidden = false;
        this.className = "";
        this.textContent = "";
        this.listeners = new Map();
    }

    addEventListener(name, listener) {
        this.listeners.set(name, listener);
    }
}

function createPageEnvironment({ edit = false }) {
    const form = new FakeElement();
    const fields = {
        lora_id: new FakeElement(edit ? "" : "1"),
        name: new FakeElement(""),
        lora_model_family: new FakeElement("qwen-image"),
        lora_type: new FakeElement("lora"),
        lora_location: new FakeElement("local"),
        file_path: new FakeElement(""),
    };
    const taskFields = [
        new FakeElement("text2img"),
        new FakeElement("img2img"),
        new FakeElement("inpaint"),
    ];
    const elements = new Map([
        [edit ? "lora-edit-form" : "lora-form", form],
        [edit ? "lora-edit-state" : "lora-form-state", new FakeElement()],
        ["local-file-panel", new FakeElement()],
        ["web-file-panel", new FakeElement()],
        ["web-file-input", new FakeElement()],
        ["select-local-file", new FakeElement()],
        ["adapter-use", new FakeElement("standard")],
        ["lightning-profile-panel", new FakeElement()],
        ["lightning-steps", new FakeElement("4")],
        ["lightning-compatibility-panel", new FakeElement()],
        ["lightning-compatibility-enabled", new FakeElement()],
        ["lightning-compatibility-tasks", new FakeElement()],
        ["hub-coordinates-panel", new FakeElement()],
        ["weight-name-field", new FakeElement()],
        ["subfolder-field", new FakeElement()],
        ["revision-field", new FakeElement()],
        ["manage-prompt-presets", new FakeElement()],
    ]);
    const selectorFields = new Map([
        ['input[name="lora_id"]', fields.lora_id],
        ['input[name="name"]', fields.name],
        ['select[name="lora_model_family"]', fields.lora_model_family],
        ['select[name="lora_type"]', fields.lora_type],
        ['select[name="lora_location"]', fields.lora_location],
        ['input[name="file_path"]', fields.file_path],
        ['button[type="submit"]', new FakeElement()],
    ]);
    form.querySelector = (selector) => selectorFields.get(selector) ?? null;
    form.reset = () => {};

    class FakeFormData {
        constructor(formElement) {
            this.formElement = formElement;
        }

        get(name) {
            return fields[name]?.value ?? null;
        }
    }

    const sandbox = {
        API_BASE: "/api",
        FormData: FakeFormData,
        console: { error: () => {}, log: () => {} },
        fetch: async () => ({ ok: true, json: async () => [] }),
        URLSearchParams,
        document: {
            getElementById: (id) => elements.get(id) ?? null,
            querySelectorAll: (selector) => (
                selector === 'input[name="lightning-compatibility-task"]' ? taskFields : []
            ),
            createElement: () => new FakeElement(),
        },
        location: { search: "" },
    };
    sandbox.window = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(
        fs.readFileSync(edit ? "frontend/models/lora/edit.js" : "frontend/models/lora/add.js", "utf8"),
        sandbox,
        { filename: edit ? "edit.js" : "add.js" },
    );
    return { sandbox, fields, taskFields, elements, form };
}

const expectedTextToImageCompatibility = {
    base_variants: ["qwen-image-2512"],
    runtime_profile_kinds: ["qwen_image_lightning"],
    supported_tasks: ["text2img"],
};
const plain = (value) => JSON.parse(JSON.stringify(value));

const add = createPageEnvironment({ edit: false });
add.fields.file_path.value = "C:/loras/companion.safetensors";
assert.equal(add.sandbox.serializeForm(add.form).compatibility, null);

const addEnabled = add.elements.get("lightning-compatibility-enabled");
addEnabled.checked = true;
add.sandbox.syncLightningCompatibility();
assert.equal(add.taskFields[0].checked, true);
assert.equal(add.elements.get("lightning-compatibility-tasks").disabled, false);
assert.deepEqual(plain(add.sandbox.serializeForm(add.form).compatibility), expectedTextToImageCompatibility);

add.taskFields[0].checked = false;
add.taskFields[1].checked = true;
add.taskFields[2].checked = true;
add.sandbox.syncLightningCompatibility();
assert.deepEqual(plain(add.sandbox.serializeForm(add.form).compatibility), {
    ...expectedTextToImageCompatibility,
    supported_tasks: ["img2img", "inpaint"],
});

add.fields.lora_model_family.value = "sdxl";
add.sandbox.syncLightningCompatibility();
assert.equal(add.elements.get("lightning-compatibility-panel").hidden, true);
assert.equal(addEnabled.checked, false);
assert.deepEqual(add.taskFields.map((field) => field.checked), [false, false, false]);
assert.equal(add.sandbox.serializeForm(add.form).compatibility, null);

add.fields.lora_model_family.value = "qwen-image";
add.fields.lora_type.value = "lora";
add.elements.get("adapter-use").value = "qwen_image_lightning";
add.sandbox.syncAdapterUse();
assert.equal(add.elements.get("lightning-compatibility-panel").hidden, true);
assert.equal(addEnabled.checked, false);
assert.equal(add.sandbox.serializeForm(add.form).compatibility, null);

const edit = createPageEnvironment({ edit: true });
edit.sandbox.fillForm({
    lora_id: 12,
    name: "Companion",
    lora_model_family: "qwen-image",
    lora_type: "lora",
    lora_location: "local",
    file_path: "C:/loras/companion.safetensors",
    runtime_profile: null,
    compatibility: {
        ...expectedTextToImageCompatibility,
        supported_tasks: ["img2img", "inpaint"],
    },
});
assert.equal(edit.elements.get("lightning-compatibility-enabled").checked, true);
assert.deepEqual(edit.taskFields.map((field) => field.checked), [false, true, true]);
assert.deepEqual(plain(edit.sandbox.buildPayload().compatibility), {
    ...expectedTextToImageCompatibility,
    supported_tasks: ["img2img", "inpaint"],
});

edit.elements.get("lightning-compatibility-enabled").checked = false;
edit.sandbox.syncLightningCompatibility();
assert.equal(edit.sandbox.buildPayload().compatibility, null);

edit.sandbox.fillForm({
    lora_id: 13,
    name: "Lightning",
    lora_model_family: "qwen-image",
    lora_type: "lora",
    lora_location: "local",
    file_path: "C:/loras/lightning.safetensors",
    runtime_profile: { kind: "qwen_image_lightning", steps: 4 },
    compatibility: null,
});
assert.equal(edit.elements.get("lightning-compatibility-panel").hidden, true);
assert.deepEqual(edit.taskFields.map((field) => field.checked), [false, false, false]);
assert.equal(edit.sandbox.buildPayload().compatibility, null);
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
