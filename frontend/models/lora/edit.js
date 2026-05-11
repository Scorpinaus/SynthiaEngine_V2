const form = document.getElementById("lora-edit-form");
const state = document.getElementById("lora-edit-state");
const loraIdField = form.querySelector('input[name="lora_id"]');
const nameField = form.querySelector('input[name="name"]');
const familyField = form.querySelector('select[name="lora_model_family"]');
const typeField = form.querySelector('select[name="lora_type"]');
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

let currentLoraId = null;
let loading = false;
let promptPresets = [];

function setState(message, variant = "info") {
    state.textContent = message;
    state.className = `model-form-state ${variant}`;
}

function setLoading(isLoading) {
    loading = isLoading;
    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = isLoading;
        submitButton.textContent = isLoading ? "Saving..." : "Save LoRA";
    }
}

function getLoraIdFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const rawValue = params.get("lora_id");
    const parsed = Number(rawValue);
    if (!rawValue || !Number.isInteger(parsed) || parsed < 0) {
        return null;
    }
    return parsed;
}

function syncFilePathMode({ resetLocal = false } = {}) {
    const isHub = locationField?.value === "hub";
    if (localFilePanel) {
        localFilePanel.hidden = isHub;
    }
    if (webFilePanel) {
        webFilePanel.hidden = !isHub;
    }
    if (filePathField && isHub) {
        filePathField.value = webFileInput.value.trim();
    } else if (filePathField && resetLocal) {
        filePathField.value = "";
    }
}

function parsePresetWords(value) {
    return String(value || "")
        .split(/[\n,]+/)
        .map((word) => word.trim())
        .filter(Boolean);
}

function normalizePromptPresets(value) {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .map((preset) => ({
            name: String(preset?.name || "").trim(),
            words: Array.isArray(preset?.words)
                ? preset.words.map((word) => String(word || "").trim()).filter(Boolean)
                : [],
        }))
        .filter((preset) => preset.name && preset.words.length > 0);
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

function fillForm(entry) {
    currentLoraId = entry.lora_id;
    loraIdField.value = String(entry.lora_id);
    nameField.value = entry.name || "";
    familyField.value = entry.lora_model_family || "";
    typeField.value = entry.lora_type || "";
    locationField.value = entry.lora_location || "local";
    filePathField.value = entry.file_path || "";
    webFileInput.value = locationField.value === "hub" ? entry.file_path || "" : "";
    promptPresets = normalizePromptPresets(entry.prompt_presets);
    renderPromptPresets();
    syncFilePathMode();
}

async function loadLora() {
    const loraId = getLoraIdFromQuery();
    if (loraId === null) {
        setState("Invalid or missing lora_id query parameter.", "error");
        setLoading(true);
        return;
    }

    setLoading(true);
    setState("Loading LoRA entry...");
    try {
        const response = await fetch(`${API_BASE}/lora-models/${encodeURIComponent(String(loraId))}`);
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || "Unable to load LoRA entry.");
        }
        const entry = await response.json();
        fillForm(entry);
        setState("LoRA entry loaded.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to load LoRA entry.", "error");
    } finally {
        setLoading(false);
    }
}

function buildPayload() {
    const nameValue = nameField.value.trim();
    return {
        lora_model_family: familyField.value.trim(),
        lora_type: typeField.value.trim(),
        lora_location: locationField.value.trim(),
        file_path: filePathField.value.trim(),
        name: nameValue || null,
        prompt_presets: promptPresets,
    };
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (loading) {
        return;
    }
    if (currentLoraId === null) {
        setState("Unable to save: missing lora_id.", "error");
        return;
    }

    const payload = buildPayload();
    if (
        !payload.lora_model_family ||
        !payload.lora_type ||
        !payload.lora_location ||
        !payload.file_path
    ) {
        setState("Please complete all required fields before saving.", "error");
        return;
    }

    setLoading(true);
    setState("Saving LoRA entry...");
    try {
        const response = await fetch(
            `${API_BASE}/lora-models/${encodeURIComponent(String(currentLoraId))}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }
        );
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || "Unable to save LoRA entry.");
        }

        const updated = await response.json();
        fillForm(updated);
        setState("LoRA entry saved successfully.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save LoRA entry.", "error");
    } finally {
        setLoading(false);
    }
});

locationField?.addEventListener("change", () => syncFilePathMode({ resetLocal: true }));
webFileInput?.addEventListener("input", () => {
    if (locationField?.value === "hub") {
        filePathField.value = webFileInput.value.trim();
    }
});
selectLocalFileButton?.addEventListener("click", selectLocalFile);
addPromptPresetButton?.addEventListener("click", addPromptPreset);

syncFilePathMode();
renderPromptPresets();
loadLora();
