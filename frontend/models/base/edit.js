const form = document.getElementById("model-edit-form");
const state = document.getElementById("model-edit-state");
const nameField = form.querySelector('input[name="name"]');
const familyField = form.querySelector('select[name="family"]');
const modelTypeField = form.querySelector('select[name="model_type"]');
const locationTypeField = form.querySelector('select[name="location_type"]');
const modelIdField = form.querySelector('input[name="model_id"]');
const versionField = form.querySelector('input[name="version"]');
const linkField = form.querySelector('input[name="link"]');
const localLinkPanel = document.getElementById("local-link-panel");
const webLinkPanel = document.getElementById("web-link-panel");
const webLinkInput = document.getElementById("web-link-input");
const selectLocalLinkButton = document.getElementById("select-local-link");

let currentModelName = null;
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
        submitButton.textContent = isLoading ? "Saving..." : "Save model";
    }
}

function getModelNameFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const name = params.get("name");
    if (!name || !name.trim()) {
        return null;
    }
    return name.trim();
}

function getLocalSelectionType() {
    return modelTypeField?.value === "single-file" ? "file" : "folder";
}

function updateLocalButtonLabel() {
    if (!selectLocalLinkButton) {
        return;
    }
    const label = getLocalSelectionType() === "file" ? "Select local file" : "Select local folder";
    selectLocalLinkButton.textContent = label;
}

function syncLinkMode({ preserveLink = false } = {}) {
    const isHub = locationTypeField?.value === "hub";
    const currentLink = linkField?.value.trim() || "";
    if (localLinkPanel) {
        localLinkPanel.hidden = isHub;
    }
    if (webLinkPanel) {
        webLinkPanel.hidden = !isHub;
    }
    if (isHub) {
        if (preserveLink && !webLinkInput.value && currentLink) {
            webLinkInput.value = currentLink;
        }
        linkField.value = webLinkInput.value.trim();
    } else if (!preserveLink) {
        linkField.value = "";
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

function fillForm(entry) {
    currentModelName = entry.name;
    nameField.value = entry.name || "";
    familyField.value = entry.family || "";
    modelTypeField.value = entry.model_type || "diffusers";
    locationTypeField.value = entry.location_type || "local";
    modelIdField.value = String(entry.model_id ?? "");
    versionField.value = entry.version || "";
    linkField.value = entry.link || "";
    webLinkInput.value = locationTypeField.value === "hub" ? entry.link || "" : "";
    syncLinkMode({ preserveLink: true });
}

async function loadModel() {
    const modelName = getModelNameFromQuery();
    if (!modelName) {
        setState("Invalid or missing name query parameter.", "error");
        setLoading(true);
        return;
    }

    setLoading(true);
    setState("Loading model entry...");
    try {
        const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`);
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            const detail = errorBody.detail || "Unable to load model entry.";
            throw new Error(detail);
        }
        const entry = await response.json();
        fillForm(entry);
        setState("Model entry loaded.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to load model entry.", "error");
    } finally {
        setLoading(false);
    }
}

function buildPayload() {
    return {
        family: familyField.value.trim(),
        model_type: modelTypeField.value.trim(),
        location_type: locationTypeField.value.trim(),
        model_id: Number(modelIdField.value),
        version: versionField.value.trim(),
        link: linkField.value.trim(),
    };
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (loading) {
        return;
    }
    if (!currentModelName) {
        setState("Unable to save: missing model name.", "error");
        return;
    }

    const payload = buildPayload();
    if (
        !payload.family ||
        !payload.model_type ||
        !payload.location_type ||
        !Number.isFinite(payload.model_id) ||
        !payload.version ||
        !payload.link
    ) {
        setState("Please complete all fields before saving.", "error");
        return;
    }

    setLoading(true);
    setState("Saving model entry...");
    try {
        const response = await fetch(`${API_BASE}/models/${encodeURIComponent(currentModelName)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            const detail = errorBody.detail || "Unable to save model entry.";
            throw new Error(detail);
        }

        const updated = await response.json();
        fillForm(updated);
        setState("Model entry saved successfully.", "success");
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save model entry.", "error");
    } finally {
        setLoading(false);
    }
});

locationTypeField?.addEventListener("change", () => syncLinkMode());
modelTypeField?.addEventListener("change", updateLocalButtonLabel);
webLinkInput?.addEventListener("input", () => {
    if (locationTypeField?.value === "hub") {
        linkField.value = webLinkInput.value.trim();
    }
});
selectLocalLinkButton?.addEventListener("click", selectLocalLink);

syncLinkMode({ preserveLink: true });
loadModel();
