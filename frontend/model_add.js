const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("model-form");
const state = document.getElementById("model-form-state");
const modelIdField = form.querySelector('input[name="model_id"]');
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
        fetchNextModelId();
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save model.", "error");
    }
});

fetchNextModelId();
