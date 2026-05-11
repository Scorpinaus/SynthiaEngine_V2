const form = document.getElementById("lora-form");
const state = document.getElementById("lora-form-state");
const loraIdField = form.querySelector('input[name="lora_id"]');
const locationField = form.querySelector('select[name="lora_location"]');
const filePathField = form.querySelector('input[name="file_path"]');
const localFilePanel = document.getElementById("local-file-panel");
const webFilePanel = document.getElementById("web-file-panel");
const webFileInput = document.getElementById("web-file-input");
const selectLocalFileButton = document.getElementById("select-local-file");
let nextLoraId = 1;

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

        const created = await response.json();
        setState("LoRA saved successfully.", "success");
        const manageLink = document.createElement("a");
        manageLink.className = "nav-link";
        manageLink.href = `prompt_presets.html?lora_id=${encodeURIComponent(String(created.lora_id))}`;
        manageLink.textContent = "Manage prompt presets";
        state.appendChild(document.createTextNode(" "));
        state.appendChild(manageLink);
        form.reset();
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

syncFilePathMode();
fetchNextLoraId();
