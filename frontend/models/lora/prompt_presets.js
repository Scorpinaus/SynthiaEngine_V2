const editorRoot = document.getElementById("lora-prompt-preset-editor-root");

function getLoraIdFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const rawValue = params.get("lora_id");
    const parsed = Number(rawValue);
    if (!rawValue || !Number.isInteger(parsed) || parsed < 0) {
        return null;
    }
    return parsed;
}

function renderMissingLoraId() {
    if (!editorRoot) {
        return;
    }
    editorRoot.innerHTML = "";
    const message = document.createElement("div");
    message.className = "model-form-state error";
    message.textContent = "Invalid or missing lora_id query parameter.";
    editorRoot.appendChild(message);
}

const loraId = getLoraIdFromQuery();
if (loraId === null) {
    renderMissingLoraId();
} else {
    window.LoraPromptPresetEditor?.mount({
        container: editorRoot,
        apiBase: API_BASE,
        loraId,
    });
}
