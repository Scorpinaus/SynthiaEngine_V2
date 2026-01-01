const API_BASE = "http://127.0.0.1:8000";

const state = {
    loading: false,
    error: null,
    models: [],
    search: "",
    family: "",
    sort: "name",
    lastUpdated: null,
};

const EMPTY_VALUE = "Unknown";

const modelsState = document.getElementById("models-state");
const modelsGrid = document.getElementById("models-grid");
const refreshButton = document.getElementById("models-refresh");
const modelsCount = document.getElementById("models-count");
const modelsUpdated = document.getElementById("models-updated");
const modelsSearch = document.getElementById("models-search");
const modelsFamily = document.getElementById("models-family");
const modelsSort = document.getElementById("models-sort");

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

function buildPill(text, variant = "default") {
    const pill = document.createElement("span");
    pill.className = `model-pill model-pill-${variant}`;
    pill.textContent = text || "Unknown";
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

    title.textContent = model.name || "Unnamed model";
    const subtitle = document.createElement("p");
    subtitle.className = "model-subtitle";
    subtitle.textContent = model.model_id ? `ID: ${model.model_id}` : "ID: Unknown";
    header.appendChild(title);
    header.appendChild(subtitle);

    const pills = document.createElement("div");
    pills.className = "model-pill-group";
    pills.appendChild(buildPill(model.family || "Unknown family", "family"));
    pills.appendChild(buildPill(model.model_type || "Unknown type", "type"));
    pills.appendChild(buildPill(model.location_type || "Unknown location", "location"));

    const meta = document.createElement("dl");
    meta.className = "model-details";
    meta.appendChild(buildDetailRow("Version", model.version || "Unknown"));
    meta.appendChild(buildDetailRow("Model ID", model.model_id ?? "Unknown"));

    const details = document.createElement("dl");
    details.className = "model-details";
    const modelIdValue = model.model_id === null || model.model_id === undefined || model.model_id === ""
        ? getValue(model.model_id)
        : buildCode(model.model_id);
    details.appendChild(buildDetailRow("Model ID", modelIdValue));
    details.appendChild(buildDetailRow("Link", buildLink(model.link)));

    card.appendChild(header);
    card.appendChild(pills);
    card.appendChild(meta);
    card.appendChild(details);
    return card;
}

function normalize(value) {
    return String(value || "").toLowerCase();
}

function getFilteredModels() {
    const query = normalize(state.search);
    const family = normalize(state.family);
    return state.models.filter((model) => {
        const familyValue = normalize(model.family);
        if (family && familyValue !== family) {
            return false;
        }
        if (!query) {
            return true;
        }
        return [
            model.name,
            model.family,
            model.model_type,
            model.model_id,
            model.location_type,
            model.version,
        ]
            .map(normalize)
            .some((value) => value.includes(query));
    });
}

function sortModels(models) {
    const sortKey = state.sort;
    const getValue = (model) => {
        switch (sortKey) {
            case "family":
                return model.family || "";
            case "type":
                return model.model_type || "";
            case "location":
                return model.location_type || "";
            default:
                return model.name || "";
        }
    };
    return [...models].sort((a, b) => getValue(a).localeCompare(getValue(b)));
}

function updateFamilyOptions(models) {
    const selected = state.family;
    const families = Array.from(
        new Set(models.map((model) => model.family).filter(Boolean))
    ).sort((a, b) => a.localeCompare(b));
    modelsFamily.innerHTML = "";
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = "All families";
    modelsFamily.appendChild(allOption);
    families.forEach((family) => {
        const option = document.createElement("option");
        option.value = family;
        option.textContent = family;
        modelsFamily.appendChild(option);
    });
    modelsFamily.value = selected;
}

function updateSummary(visibleCount, totalCount) {
    modelsCount.textContent = visibleCount;
    const countLabel = document.querySelector(".models-summary-label");
    if (countLabel) {
        countLabel.textContent = totalCount === 1 ? "model" : "models";
    }
    if (state.lastUpdated) {
        modelsUpdated.textContent = `Updated ${state.lastUpdated.toLocaleTimeString()}`;
    } else {
        modelsUpdated.textContent = "";
    }
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
        updateSummary(0, state.models.length);
        return;
    }

    if (!state.models.length) {
        modelsState.textContent = "No models found in the registry.";
        modelsState.style.display = "block";
        updateSummary(0, 0);
        return;
    }

    const filtered = sortModels(getFilteredModels());
    updateSummary(filtered.length, state.models.length);

    if (!filtered.length) {
        modelsState.textContent = "No models match your current filters.";
        modelsState.style.display = "block";
        return;
    }

    modelsState.style.display = "none";
    filtered.forEach((model) => {
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
        setState({ models, loading: false, lastUpdated: new Date() });
        updateFamilyOptions(models);
    } catch (error) {
        setState({ loading: false, error: "Unable to load models. Try again shortly." });
        console.error(error);
    }
}

refreshButton.addEventListener("click", fetchModels);
modelsSearch.addEventListener("input", (event) => {
    setState({ search: event.target.value });
});
modelsFamily.addEventListener("change", (event) => {
    setState({ family: event.target.value });
});
modelsSort.addEventListener("change", (event) => {
    setState({ sort: event.target.value });
});

fetchModels();
