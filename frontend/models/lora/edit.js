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
const managePromptPresetsLink = document.getElementById("manage-prompt-presets");

let currentLoraId = null;
let loading = false;

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
    if (managePromptPresetsLink) {
        managePromptPresetsLink.href = `prompt_presets.html?lora_id=${encodeURIComponent(String(entry.lora_id))}`;
    }
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

syncFilePathMode();
loadLora();
