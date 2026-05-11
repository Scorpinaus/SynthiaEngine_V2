const form = document.getElementById("model-form");
const state = document.getElementById("model-form-state");
const modelIdField = form.querySelector('input[name="model_id"]');
const modelTypeField = form.querySelector('select[name="model_type"]');
const locationTypeField = form.querySelector('select[name="location_type"]');
const linkField = form.querySelector('input[name="link"]');
const localLinkPanel = document.getElementById("local-link-panel");
const webLinkPanel = document.getElementById("web-link-panel");
const webLinkInput = document.getElementById("web-link-input");
const selectLocalLinkButton = document.getElementById("select-local-link");
let nextModelId = 1;

function setState(message, variant = "info") {
    state.textContent = message;
    state.className = `model-form-state ${variant}`;
}

function serializeForm(formElement) {
    const formData = new FormData(formElement);
    return {
        name: formData.get("name")?.toString().trim() ?? "",
        family: formData.get("family")?.toString().trim() ?? "",
        model_type: formData.get("model_type")?.toString().trim() ?? "",
        location_type: formData.get("location_type")?.toString().trim() ?? "",
        model_id: nextModelId,
        version: formData.get("version")?.toString().trim() ?? "",
        link: formData.get("link")?.toString().trim() ?? "",
    };
}

function updateModelId(value) {
    nextModelId = value;
    if (modelIdField) {
        modelIdField.value = String(value);
    }
}

function getLocalSelectionType() {
    return modelTypeField?.value === "single_file" ? "file" : "folder";
}

function updateLocalButtonLabel() {
    if (!selectLocalLinkButton) {
        return;
    }
    const label = getLocalSelectionType() === "file" ? "Select local file" : "Select local folder";
    selectLocalLinkButton.textContent = label;
}

function syncLinkMode() {
    const isHub = locationTypeField?.value === "hub";
    if (localLinkPanel) {
        localLinkPanel.hidden = isHub;
    }
    if (webLinkPanel) {
        webLinkPanel.hidden = !isHub;
    }
    if (linkField) {
        linkField.value = isHub ? webLinkInput.value.trim() : "";
    }
    updateLocalButtonLabel();
}

async function selectLocalLink() {
    const selectionType = getLocalSelectionType();
    setState(`Selecting local ${selectionType}...`);

    try {
        const response = await fetch(`${API_BASE}/api/local-path/select`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection_type: selectionType }),
        });
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            const detail = errorBody.detail || "Unable to select local path.";
            throw new Error(detail);
        }

        const data = await response.json();
        const selectedPath = data.path || "";
        if (!selectedPath) {
            setState("No local path selected.", "error");
            return;
        }
        linkField.value = selectedPath;
        setState("Local path selected.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to select local path.", "error");
    }
}

async function fetchNextModelId() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) {
            throw new Error("Failed to load model registry.");
        }
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        const maxId = models.reduce((max, model) => {
            const currentId = Number(model.model_id);
            return Number.isFinite(currentId) ? Math.max(max, currentId) : max;
        }, 0);
        updateModelId(maxId + 1);
    } catch (error) {
        console.error(error);
        updateModelId(1);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setState("Saving model entry…");

    const payload = serializeForm(form);
    if (!payload.name) {
        setState("Please complete all fields before submitting.", "error");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/models`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            const detail = errorBody.detail || "Unable to save model.";
            throw new Error(detail);
        }

        setState("Model saved successfully.", "success");
        form.reset();
        syncLinkMode();
        fetchNextModelId();
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save model.", "error");
    }
});

locationTypeField?.addEventListener("change", syncLinkMode);
modelTypeField?.addEventListener("change", updateLocalButtonLabel);
webLinkInput?.addEventListener("input", () => {
    if (locationTypeField?.value === "hub") {
        linkField.value = webLinkInput.value.trim();
    }
});
selectLocalLinkButton?.addEventListener("click", selectLocalLink);

syncLinkMode();
fetchNextModelId();
