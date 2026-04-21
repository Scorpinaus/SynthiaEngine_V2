const form = document.getElementById("lora-form");
const state = document.getElementById("lora-form-state");
const loraIdField = form.querySelector('input[name="lora_id"]');
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

        setState("LoRA saved successfully.", "success");
        form.reset();
        fetchNextLoraId();
    } catch (error) {
        console.error(error);
        setState(error.message || "Unable to save LoRA.", "error");
    }
});

fetchNextLoraId();
