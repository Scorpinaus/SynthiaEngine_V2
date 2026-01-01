const API_BASE = "http://127.0.0.1:8000";

const state = {
    loading: false,
    error: null,
    models: [],
};

const modelsState = document.getElementById("models-state");
const modelsGrid = document.getElementById("models-grid");
const refreshButton = document.getElementById("models-refresh");

function setState(next) {
    Object.assign(state, next);
    render();
}

function buildDetailRow(label, value) {
    const row = document.createElement("div");
    row.className = "model-detail-row";
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    row.appendChild(dt);
    row.appendChild(dd);
    return row;
}

function buildLink(link) {
    if (!link) {
        const span = document.createElement("span");
        span.textContent = "Not available";
        return span;
    }
    const anchor = document.createElement("a");
    anchor.href = link.startsWith("http") ? link : `https://huggingface.co/${link}`;
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
    anchor.textContent = link;
    return anchor;
}

function buildCard(model) {
    const card = document.createElement("article");
    card.className = "model-card";

    const header = document.createElement("div");
    header.className = "model-card-header";
    const title = document.createElement("h3");
    title.textContent = model.name || "Unnamed model";
    const family = document.createElement("span");
    family.className = "model-tag";
    family.textContent = model.family || "Unknown family";
    header.appendChild(title);
    header.appendChild(family);

    const meta = document.createElement("dl");
    meta.className = "model-details";
    meta.appendChild(buildDetailRow("Model type", model.model_type || "Unknown"));
    meta.appendChild(buildDetailRow("Location", model.location_type || "Unknown"));
    meta.appendChild(buildDetailRow("Model ID", model.model_id ?? "Unknown"));
    meta.appendChild(buildDetailRow("Version", model.version || "Unknown"));

    const linkRow = document.createElement("div");
    linkRow.className = "model-detail-row";
    const dt = document.createElement("dt");
    dt.textContent = "Link";
    const dd = document.createElement("dd");
    dd.appendChild(buildLink(model.link));
    linkRow.appendChild(dt);
    linkRow.appendChild(dd);
    meta.appendChild(linkRow);

    card.appendChild(header);
    card.appendChild(meta);
    return card;
}

function render() {
    modelsGrid.innerHTML = "";

    if (state.loading) {
        modelsState.textContent = "Loading model registry…";
        modelsState.style.display = "block";
        return;
    }

    if (state.error) {
        modelsState.textContent = state.error;
        modelsState.style.display = "block";
        return;
    }

    if (!state.models.length) {
        modelsState.textContent = "No models found in the registry.";
        modelsState.style.display = "block";
        return;
    }

    modelsState.style.display = "none";
    state.models.forEach((model) => {
        modelsGrid.appendChild(buildCard(model));
    });
}

async function fetchModels() {
    setState({ loading: true, error: null });
    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) {
            throw new Error("Failed to load model registry.");
        }
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        models.sort((a, b) => (a.name || "").localeCompare(b.name || ""));
        setState({ models, loading: false });
    } catch (error) {
        setState({ loading: false, error: "Unable to load models. Try again shortly." });
        console.error(error);
    }
}

refreshButton.addEventListener("click", fetchModels);

fetchModels();
