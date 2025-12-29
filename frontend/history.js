const API_BASE = "http://127.0.0.1:8000";
const METADATA_FIELDS = [
    { key: "prompt", label: "Prompt" },
    { key: "negative_prompt", label: "Negative Prompt" },
    { key: "steps", label: "Steps" },
    { key: "cfg", label: "CFG" },
    { key: "width", label: "Width" },
    { key: "height", label: "Height" },
    { key: "seed", label: "Seed" },
    { key: "scheduler", label: "Scheduler" },
    { key: "model", label: "Model" },
    { key: "clip_skip", label: "Clip Skip" },
    { key: "mode", label: "Mode" },
    { key: "batch_id", label: "Batch ID" },
];

const state = {
    loading: false,
    error: null,
    records: [],
};

const historyState = document.getElementById("history-state");
const historyGrid = document.getElementById("history-grid");
const refreshButton = document.getElementById("history-refresh");

function setState(next) {
    Object.assign(state, next);
    render();
}

function formatTimestamp(timestamp) {
    if (!timestamp) {
        return "Unknown time";
    }
    return new Date(timestamp * 1000).toLocaleString();
}

function buildMetadata(metadata) {
    const list = document.createElement("dl");
    list.className = "history-meta-list";
    let hasEntries = false;

    METADATA_FIELDS.forEach(({ key, label }) => {
        const value = metadata?.[key];
        if (value === undefined || value === null || value === "") {
            return;
        }
        hasEntries = true;
        const row = document.createElement("div");
        row.className = "history-meta-row";
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        row.appendChild(dt);
        row.appendChild(dd);
        list.appendChild(row);
    });

    return hasEntries ? list : null;
}

function buildCard(record, index) {
    const card = document.createElement("article");
    card.className = "history-card";

    const imageWrap = document.createElement("div");
    imageWrap.className = "history-image";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `${API_BASE}${record.url}`;
    img.alt = `Render ${record.filename || index + 1}`;
    imageWrap.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "history-meta";

    const title = document.createElement("div");
    title.className = "history-meta-title";
    const name = document.createElement("span");
    name.textContent = record.filename || `Render ${index + 1}`;
    const time = document.createElement("span");
    time.className = "history-meta-time";
    time.textContent = formatTimestamp(record.timestamp);
    title.appendChild(name);
    title.appendChild(time);

    const details = document.createElement("details");
    details.className = "history-details";
    const summary = document.createElement("summary");
    summary.textContent = "Prompt metadata";
    details.appendChild(summary);
    const metadataList = buildMetadata(record.metadata || {});
    if (metadataList) {
        details.appendChild(metadataList);
    } else {
        const empty = document.createElement("p");
        empty.className = "history-empty-meta";
        empty.textContent = "No prompt metadata available.";
        details.appendChild(empty);
    }

    meta.appendChild(title);
    meta.appendChild(details);

    card.appendChild(imageWrap);
    card.appendChild(meta);

    return card;
}

function render() {
    historyGrid.innerHTML = "";

    if (state.loading) {
        historyState.textContent = "Loading render history…";
        historyState.style.display = "block";
        return;
    }

    if (state.error) {
        historyState.textContent = state.error;
        historyState.style.display = "block";
        return;
    }

    if (!state.records.length) {
        historyState.textContent = "No renders yet. Generate an image to populate the gallery.";
        historyState.style.display = "block";
        return;
    }

    historyState.style.display = "none";
    state.records.forEach((record, index) => {
        historyGrid.appendChild(buildCard(record, index));
    });
}

async function fetchHistory() {
    setState({ loading: true, error: null });
    try {
        const response = await fetch(`${API_BASE}/history`);
        if (!response.ok) {
            throw new Error("Failed to load history.");
        }
        const data = await response.json();
        const records = Array.isArray(data) ? data : [];
        records.sort((a, b) => (b.timestamp ?? 0) - (a.timestamp ?? 0));
        setState({ records, loading: false });
    } catch (error) {
        setState({ loading: false, error: "Unable to load history. Try again shortly." });
        console.error(error);
    }
}

refreshButton.addEventListener("click", fetchHistory);

fetchHistory();
