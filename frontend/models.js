const API_BASE = "http://127.0.0.1:8000";

const state = {
    loading: false,
    error: null,
    models: [],
};

const EMPTY_VALUE = "Unknown";

const modelsState = document.getElementById("models-state");
const modelsGrid = document.getElementById("models-grid");
const refreshButton = document.getElementById("models-refresh");
const refreshLabel = refreshButton.textContent;

function setState(next) {
    Object.assign(state, next);
    render();
}

function getValue(value, fallback = EMPTY_VALUE) {
    if (value === null || value === undefined || value === "") {
        return fallback;
    }
    return value;
}

function buildDetailRow(label, value) {
    const row = document.createElement("div");
    row.className = "model-detail-row";
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    if (value instanceof Node) {
        dd.appendChild(value);
    } else {
        dd.textContent = value;
    }
    row.appendChild(dt);
    row.appendChild(dd);
    return row;
}

function buildCode(value) {
    const code = document.createElement("span");
    code.className = "model-code";
    code.textContent = value;
    return code;
}

function buildPill(label, value) {
    const pill = document.createElement("span");
    pill.className = "model-pill";
    pill.textContent = `${label}: ${getValue(value)}`;
    return pill;
}

function buildLink(link) {
    if (!link) {
        const span = document.createElement("span");
        span.className = "model-muted";
        span.textContent = "Not available";
        return span;
    }
    const anchor = document.createElement("a");
    anchor.className = "model-link";
    anchor.href = link.startsWith("http") ? link : `https://huggingface.co/${link}`;
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
    anchor.textContent = link;
    return anchor;
}

function buildCard(model) {
    const card = document.createElement("article");
    card.className = "model-card";

    const header = document.createElement("header");
    header.className = "model-card-header";
    const titleWrap = document.createElement("div");
    titleWrap.className = "model-card-title";
    const title = document.createElement("h3");
    title.textContent = getValue(model.name, "Unnamed model");
    const family = document.createElement("span");
    family.className = "model-tag";
    family.textContent = getValue(model.family, "Unknown family");
    titleWrap.appendChild(title);
    header.appendChild(titleWrap);
    header.appendChild(family);

    const meta = document.createElement("div");
    meta.className = "model-card-meta";
    meta.appendChild(buildPill("Type", model.model_type));
    meta.appendChild(buildPill("Location", model.location_type));
    meta.appendChild(buildPill("Version", model.version));

    const details = document.createElement("dl");
    details.className = "model-details";
    const modelIdValue = model.model_id === null || model.model_id === undefined || model.model_id === ""
        ? getValue(model.model_id)
        : buildCode(model.model_id);
    details.appendChild(buildDetailRow("Model ID", modelIdValue));
    details.appendChild(buildDetailRow("Link", buildLink(model.link)));

    card.appendChild(header);
    card.appendChild(meta);
    card.appendChild(details);
    return card;
}

function render() {
    modelsGrid.innerHTML = "";
    refreshButton.textContent = state.loading ? "Refreshing..." : refreshLabel;
    refreshButton.disabled = state.loading;

    if (state.loading) {
        modelsState.textContent = "Loading model registry...";
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
