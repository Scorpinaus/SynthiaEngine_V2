const state = {
    loading: false,
    deletingId: null,
    error: null,
    loras: [],
    search: "",
    family: "",
    sort: "name",
    lastUpdated: null,
};

const EMPTY_VALUE = "Unknown";

const loraState = document.getElementById("lora-state");
const loraGrid = document.getElementById("lora-grid");
const refreshButton = document.getElementById("lora-refresh");
const refreshLabel = refreshButton ? refreshButton.textContent : "Refresh";
const loraCount = document.getElementById("lora-count");
const loraUpdated = document.getElementById("lora-updated");
const loraSearch = document.getElementById("lora-search");
const loraFamily = document.getElementById("lora-family");
const loraSort = document.getElementById("lora-sort");

function setState(next) {
    Object.assign(state, next);
    render();
}

function normalize(value) {
    return String(value || "").toLowerCase();
}

function getValue(value, fallback = EMPTY_VALUE) {
    if (value === null || value === undefined || value === "") {
        return fallback;
    }
    return value;
}

function buildDetailRow(label, value, options = {}) {
    const row = document.createElement("div");
    row.className = "model-detail-row";
    if (options.stacked) {
        row.classList.add("model-detail-row-stacked");
    }

    const dt = document.createElement("dt");
    dt.textContent = label;

    const dd = document.createElement("dd");
    if (value instanceof Node) {
        dd.appendChild(value);
    } else {
        dd.textContent = value;
    }

    row.append(dt, dd);
    return row;
}

function buildPill(text, variant = "default") {
    const pill = document.createElement("span");
    pill.className = `model-pill model-pill-${variant}`;
    pill.textContent = text || EMPTY_VALUE;
    return pill;
}

function buildCode(text) {
    const code = document.createElement("code");
    code.className = "model-code";
    code.textContent = text || EMPTY_VALUE;
    return code;
}

function getPromptPresets(entry) {
    return Array.isArray(entry.prompt_presets) ? entry.prompt_presets : [];
}

function formatPromptPresetSummary(entry) {
    const presets = getPromptPresets(entry);
    if (presets.length === 0) {
        return "None";
    }
    const names = presets.map((preset) => preset.name).filter(Boolean);
    return `${presets.length} preset${presets.length === 1 ? "" : "s"}${names.length ? `: ${names.join(", ")}` : ""}`;
}

function buildActionLink(href, text) {
    const link = document.createElement("a");
    link.className = "secondary nav-link";
    link.href = href;
    link.textContent = text;
    return link;
}

function buildDeleteButton(entry) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = state.deletingId === entry.lora_id ? "Deleting..." : "Delete";
    button.disabled = state.deletingId === entry.lora_id || state.loading;
    button.addEventListener("click", () => {
        handleDelete(entry);
    });
    return button;
}

function buildCard(entry) {
    const card = document.createElement("article");
    card.className = "model-card";

    const header = document.createElement("header");
    header.className = "model-card-header";

    const title = document.createElement("h3");
    title.textContent = getValue(entry.name, `LoRA ${entry.lora_id ?? "Unknown"}`);
    const subtitle = document.createElement("p");
    subtitle.className = "model-subtitle";
    subtitle.textContent = `LoRA ID: ${getValue(entry.lora_id)}`;
    header.append(title, subtitle);

    const pills = document.createElement("div");
    pills.className = "model-pill-group";
    pills.appendChild(buildPill(getValue(entry.lora_model_family), "family"));
    pills.appendChild(buildPill(getValue(entry.lora_type), "type"));
    pills.appendChild(buildPill(getValue(entry.lora_location), "location"));

    const details = document.createElement("dl");
    details.className = "model-details";
    details.appendChild(buildDetailRow("LoRA ID", buildCode(String(getValue(entry.lora_id)))));
    details.appendChild(buildDetailRow("Prompt Presets", formatPromptPresetSummary(entry)));
    details.appendChild(buildDetailRow("File Path", buildCode(getValue(entry.file_path)), { stacked: true }));

    const actions = document.createElement("div");
    actions.className = "models-actions";
    actions.appendChild(buildActionLink(`edit.html?lora_id=${encodeURIComponent(String(entry.lora_id))}`, "Edit"));
    actions.appendChild(buildDeleteButton(entry));

    card.append(header, pills, details, actions);
    return card;
}

function getFilteredLoras() {
    const query = normalize(state.search);
    const family = normalize(state.family);
    return state.loras.filter((entry) => {
        if (family && normalize(entry.lora_model_family) !== family) {
            return false;
        }
        if (!query) {
            return true;
        }
        return [
            entry.lora_id,
            entry.name,
            entry.lora_model_family,
            entry.lora_type,
            entry.lora_location,
            entry.file_path,
            ...getPromptPresets(entry).flatMap((preset) => [preset.name, ...(preset.words || [])]),
        ].map(normalize).some((value) => value.includes(query));
    });
}

function sortLoras(entries) {
    const sortKey = state.sort;
    const getSortValue = (entry) => {
        switch (sortKey) {
            case "family":
                return normalize(entry.lora_model_family);
            case "type":
                return normalize(entry.lora_type);
            case "location":
                return normalize(entry.lora_location);
            case "id":
                return String(Number(entry.lora_id) || 0).padStart(12, "0");
            default:
                return normalize(entry.name || entry.file_path);
        }
    };
    return [...entries].sort((a, b) => getSortValue(a).localeCompare(getSortValue(b)));
}

function updateFamilyOptions(entries) {
    const selected = state.family;
    const families = Array.from(
        new Set(entries.map((entry) => entry.lora_model_family).filter(Boolean))
    ).sort((a, b) => a.localeCompare(b));

    loraFamily.innerHTML = "";
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = "All families";
    loraFamily.appendChild(allOption);

    families.forEach((family) => {
        const option = document.createElement("option");
        option.value = family;
        option.textContent = family;
        loraFamily.appendChild(option);
    });
    loraFamily.value = selected;
}

function updateSummary(visibleCount, totalCount) {
    loraCount.textContent = String(visibleCount);
    const countLabel = document.querySelector(".models-summary-label");
    if (countLabel) {
        countLabel.textContent = totalCount === 1 ? "lora" : "loras";
    }
    if (state.lastUpdated) {
        loraUpdated.textContent = `Updated ${state.lastUpdated.toLocaleTimeString()}`;
    } else {
        loraUpdated.textContent = "";
    }
}

function render() {
    loraGrid.innerHTML = "";
    refreshButton.textContent = state.loading ? "Refreshing..." : refreshLabel;
    refreshButton.disabled = state.loading || state.deletingId !== null;

    if (state.loading) {
        loraState.textContent = "Loading LoRA registry...";
        loraState.style.display = "block";
        return;
    }

    if (state.error) {
        loraState.textContent = state.error;
        loraState.style.display = "block";
        updateSummary(0, state.loras.length);
        return;
    }

    if (!state.loras.length) {
        loraState.textContent = "No LoRAs found in the registry.";
        loraState.style.display = "block";
        updateSummary(0, 0);
        return;
    }

    const filtered = sortLoras(getFilteredLoras());
    updateSummary(filtered.length, state.loras.length);

    if (!filtered.length) {
        loraState.textContent = "No LoRAs match your current filters.";
        loraState.style.display = "block";
        return;
    }

    loraState.style.display = "none";
    filtered.forEach((entry) => {
        loraGrid.appendChild(buildCard(entry));
    });
}

async function fetchLoras() {
    setState({ loading: true, error: null });
    try {
        const response = await fetch(`${API_BASE}/lora-models`);
        if (!response.ok) {
            throw new Error("Failed to load LoRA registry.");
        }
        const data = await response.json();
        const loras = Array.isArray(data) ? data : [];
        setState({ loras, loading: false, lastUpdated: new Date() });
        updateFamilyOptions(loras);
    } catch (error) {
        console.error(error);
        setState({ loading: false, error: "Unable to load LoRAs. Try again shortly." });
    }
}

async function handleDelete(entry) {
    const confirmed = window.confirm(
        `Delete LoRA ${entry.lora_id}${entry.name ? ` (${entry.name})` : ""}?`
    );
    if (!confirmed) {
        return;
    }

    setState({ deletingId: entry.lora_id, error: null });
    try {
        const response = await fetch(`${API_BASE}/lora-models/${encodeURIComponent(String(entry.lora_id))}`, {
            method: "DELETE",
        });
        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || "Unable to delete LoRA entry.");
        }
        setState({ deletingId: null });
        await fetchLoras();
    } catch (error) {
        console.error(error);
        setState({
            deletingId: null,
            error: error.message || "Unable to delete LoRA entry.",
        });
    }
}

refreshButton.addEventListener("click", fetchLoras);
loraSearch.addEventListener("input", (event) => {
    setState({ search: event.target.value });
});
loraFamily.addEventListener("change", (event) => {
    setState({ family: event.target.value });
});
loraSort.addEventListener("change", (event) => {
    setState({ sort: event.target.value });
});

fetchLoras();
