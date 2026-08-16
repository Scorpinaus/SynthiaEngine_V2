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
const adapterUseField = document.getElementById("adapter-use");
const lightningProfilePanel = document.getElementById("lightning-profile-panel");
const lightningStepsField = document.getElementById("lightning-steps");
const lightningCompatibilityPanel = document.getElementById("lightning-compatibility-panel");
const lightningCompatibilityEnabledField = document.getElementById("lightning-compatibility-enabled");
const lightningCompatibilityTasksField = document.getElementById("lightning-compatibility-tasks");
const lightningCompatibilityTaskFields = Array.from(
    document.querySelectorAll('input[name="lightning-compatibility-task"]')
);
const hubCoordinatesPanel = document.getElementById("hub-coordinates-panel");
const weightNameField = document.getElementById("weight-name-field");
const subfolderField = document.getElementById("subfolder-field");
const revisionField = document.getElementById("revision-field");

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
    if (hubCoordinatesPanel) {
        hubCoordinatesPanel.hidden = !isHub;
    }
    if (filePathField && isHub) {
        filePathField.value = webFileInput.value.trim();
    } else if (filePathField && resetLocal) {
        filePathField.value = "";
    }
}

function isLightningSelected() {
    return adapterUseField?.value === "qwen_image_lightning";
}

function isLightningCompatibilityEligible() {
    return !isLightningSelected()
        && familyField?.value === "qwen-image"
        && typeField?.value === "lora";
}

function selectedLightningCompatibilityTasks() {
    return lightningCompatibilityTaskFields
        .filter((field) => field.checked)
        .map((field) => field.value);
}

function syncLightningCompatibility() {
    const isEligible = isLightningCompatibilityEligible();
    if (lightningCompatibilityPanel) {
        lightningCompatibilityPanel.hidden = !isEligible;
    }
    if (lightningCompatibilityEnabledField) {
        lightningCompatibilityEnabledField.disabled = !isEligible;
        if (!isEligible) {
            lightningCompatibilityEnabledField.checked = false;
            lightningCompatibilityTaskFields.forEach((field) => {
                field.checked = false;
            });
        }
    }
    if (lightningCompatibilityTasksField) {
        lightningCompatibilityTasksField.disabled = !isEligible || !lightningCompatibilityEnabledField?.checked;
    }
    if (isEligible && lightningCompatibilityEnabledField?.checked && !selectedLightningCompatibilityTasks().length) {
        const textToImageTask = lightningCompatibilityTaskFields.find((field) => field.value === "text2img");
        if (textToImageTask) {
            textToImageTask.checked = true;
        }
    }
}

function syncAdapterUse() {
    const isLightning = isLightningSelected();
    if (lightningProfilePanel) {
        lightningProfilePanel.hidden = !isLightning;
    }
    if (isLightning) {
        familyField.value = "qwen-image";
        typeField.value = "lora";
    }
    familyField.disabled = isLightning;
    typeField.disabled = isLightning;
    syncLightningCompatibility();
}

function buildRuntimeProfile() {
    if (!isLightningSelected()) {
        return null;
    }
    return {
        kind: "qwen_image_lightning",
        base_variant: "qwen-image-2512",
        steps: Number(lightningStepsField?.value || 4),
        true_cfg_scale: 1.0,
        scheduler_profile: "qwen_image_lightning_shift3",
        adapter_strength: 1.0,
        supported_tasks: ["text2img", "img2img", "inpaint"],
    };
}

function buildLightningCompatibility() {
    if (!isLightningCompatibilityEligible() || !lightningCompatibilityEnabledField?.checked) {
        return null;
    }
    const supportedTasks = selectedLightningCompatibilityTasks();
    if (!supportedTasks.length) {
        return null;
    }
    return {
        base_variants: ["qwen-image-2512"],
        runtime_profile_kinds: ["qwen_image_lightning"],
        supported_tasks: supportedTasks,
    };
}

function hydrateLightningCompatibility(compatibility) {
    const supportedTasks = Array.isArray(compatibility?.supported_tasks)
        ? compatibility.supported_tasks
        : [];
    if (lightningCompatibilityEnabledField) {
        lightningCompatibilityEnabledField.checked = Boolean(compatibility);
    }
    lightningCompatibilityTaskFields.forEach((field) => {
        field.checked = supportedTasks.includes(field.value);
    });
    syncLightningCompatibility();
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
    adapterUseField.value = entry.runtime_profile?.kind === "qwen_image_lightning"
        ? "qwen_image_lightning"
        : "standard";
    lightningStepsField.value = String(entry.runtime_profile?.steps || 4);
    weightNameField.value = entry.weight_name || "";
    subfolderField.value = entry.subfolder || "";
    revisionField.value = entry.revision || "";
    if (managePromptPresetsLink) {
        managePromptPresetsLink.href = `prompt_presets.html?lora_id=${encodeURIComponent(String(entry.lora_id))}`;
    }
    syncFilePathMode();
    syncAdapterUse();
    hydrateLightningCompatibility(entry.compatibility);
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
    const isHub = locationField.value === "hub";
    return {
        lora_model_family: familyField.value.trim(),
        lora_type: typeField.value.trim(),
        lora_location: locationField.value.trim(),
        file_path: filePathField.value.trim(),
        name: nameValue || null,
        runtime_profile: buildRuntimeProfile(),
        compatibility: buildLightningCompatibility(),
        weight_name: isHub ? weightNameField.value.trim() || null : null,
        subfolder: isHub ? subfolderField.value.trim() || null : null,
        revision: isHub ? revisionField.value.trim() || null : null,
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
    if (payload.runtime_profile && payload.lora_location === "hub" && !payload.weight_name) {
        setState("Hub Qwen Image Lightning entries require a weight name.", "error");
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
adapterUseField?.addEventListener("change", syncAdapterUse);
familyField?.addEventListener("change", syncLightningCompatibility);
typeField?.addEventListener("change", syncLightningCompatibility);
lightningCompatibilityEnabledField?.addEventListener("change", syncLightningCompatibility);
lightningCompatibilityTasksField?.addEventListener("change", syncLightningCompatibility);
webFileInput?.addEventListener("input", () => {
    if (locationField?.value === "hub") {
        filePathField.value = webFileInput.value.trim();
    }
});
selectLocalFileButton?.addEventListener("click", selectLocalFile);

syncFilePathMode();
syncAdapterUse();
loadLora();
