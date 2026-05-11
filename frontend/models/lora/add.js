const form = document.getElementById("lora-form");
const state = document.getElementById("lora-form-state");
const loraIdField = form.querySelector('input[name="lora_id"]');
const locationField = form.querySelector('select[name="lora_location"]');
const filePathField = form.querySelector('input[name="file_path"]');
const localFilePanel = document.getElementById("local-file-panel");
const webFilePanel = document.getElementById("web-file-panel");
const webFileInput = document.getElementById("web-file-input");
const selectLocalFileButton = document.getElementById("select-local-file");
const promptPresetName = document.getElementById("prompt-preset-name");
const promptPresetWords = document.getElementById("prompt-preset-words");
const addPromptPresetButton = document.getElementById("add-prompt-preset");
const promptPresetList = document.getElementById("prompt-preset-list");
let nextLoraId = 1;
let promptPresets = [];

function setState(message, variant = "info") {
    state.textContent = message;
    state.className = `model-form-state ${variant}`;
}

function updateLoraId(value) {
    nextLoraId = value;
    if (loraIdField) {
        loraIdField.value = String(value);
    }
}

function syncFilePathMode() {
    const isHub = locationField?.value === "hub";
    if (localFilePanel) {
        localFilePanel.hidden = isHub;
    }
    if (webFilePanel) {
        webFilePanel.hidden = !isHub;
    }
    if (filePathField) {
        filePathField.value = isHub ? webFileInput.value.trim() : "";
    }
}

function parsePresetWords(value) {
    return String(value || "")
        .split(/[\n,]+/)
        .map((word) => word.trim())
        .filter(Boolean);
}

function renderPromptPresets() {
    if (!promptPresetList) {
        return;
    }
    promptPresetList.innerHTML = "";
    if (promptPresets.length === 0) {
        const empty = document.createElement("div");
        empty.className = "field-hint";
        empty.textContent = "No prompt presets added.";
        promptPresetList.appendChild(empty);
        return;
    }
    promptPresets.forEach((preset, index) => {
        const row = document.createElement("div");
        row.className = "lora-preset-row";

        const summary = document.createElement("div");
        summary.className = "lora-preset-summary";
        const name = document.createElement("strong");
        name.textContent = preset.name;
        const words = document.createElement("span");
        words.textContent = preset.words.join(", ");
        summary.append(name, words);

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
            promptPresets.splice(index, 1);
            renderPromptPresets();
        });

        row.append(summary, remove);
        promptPresetList.appendChild(row);
    });
}

function addPromptPreset() {
    const name = promptPresetName?.value.trim() || "";
    const words = parsePresetWords(promptPresetWords?.value);
    if (!name || words.length === 0) {
        setState("Prompt presets need a name and at least one word.", "error");
        return;
    }
    promptPresets.push({ name, words });
    promptPresetName.value = "";
    promptPresetWords.value = "";
    renderPromptPresets();
    setState("Prompt preset added.", "success");
}

async function selectLocalFile() {
    setState("Selecting local file...");

    try {
        const response = await fetch(`${API_BASE}/api/local-path/select`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection_type: "file" }),
        });
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            const detail = errorBody.detail || "Unable to select local file.";
            throw new Error(detail);
        }

        const data = await response.json();
        const selectedPath = data.path || "";
        if (!selectedPath) {
            setState("No local file selected.", "error");
            return;
        }
        filePathField.value = selectedPath;
        setState("Local file selected.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to select local file.", "error");
    }
}

function serializeForm(formElement) {
    const formData = new FormData(formElement);
    const nameValue = formData.get("name")?.toString().trim() ?? "";
    return {
        lora_id: nextLoraId,
        lora_model_family: formData.get("lora_model_family")?.toString().trim() ?? "",
        lora_type: formData.get("lora_type")?.toString().trim() ?? "",
        lora_location: formData.get("lora_location")?.toString().trim() ?? "",
        file_path: formData.get("file_path")?.toString().trim() ?? "",
        name: nameValue || null,
        prompt_presets: promptPresets,
    };
}

async function fetchNextLoraId() {
    try {
        const response = await fetch(`${API_BASE}/lora-models`);
        if (!response.ok) {
            throw new Error("Failed to load LoRA registry.");
        }
        const data = await response.json();
        const loras = Array.isArray(data) ? data : [];
        const maxId = loras.reduce((max, entry) => {
            const currentId = Number(entry.lora_id);
            return Number.isFinite(currentId) ? Math.max(max, currentId) : max;
        }, 0);
        updateLoraId(maxId + 1);
    } catch (error) {
        console.error(error);
        updateLoraId(1);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setState("Saving LoRA entry...");

    const payload = serializeForm(form);
    if (
        !payload.lora_model_family ||
        !payload.lora_type ||
        !payload.lora_location ||
        !payload.file_path
    ) {
        setState("Please complete all required fields before submitting.", "error");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/lora-models`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || "Unable to save LoRA.");
        }

        setState("LoRA saved successfully.", "success");
        form.reset();
        promptPresets = [];
        renderPromptPresets();
        syncFilePathMode();
        fetchNextLoraId();
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save LoRA.", "error");
    }
});

locationField?.addEventListener("change", syncFilePathMode);
webFileInput?.addEventListener("input", () => {
    if (locationField?.value === "hub") {
        filePathField.value = webFileInput.value.trim();
    }
});
selectLocalFileButton?.addEventListener("click", selectLocalFile);
addPromptPresetButton?.addEventListener("click", addPromptPreset);

syncFilePathMode();
renderPromptPresets();
fetchNextLoraId();
