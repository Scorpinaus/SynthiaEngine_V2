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
    batches: [],
    selectedBatchId: null,
    selectedIndex: 0,
};

const historyState = document.getElementById("history-state");
const historyLayout = document.getElementById("history-layout");
const historyBatchList = document.getElementById("history-batch-list");
const historyBatchCount = document.getElementById("history-batch-count");
const viewerTitle = document.getElementById("history-viewer-title");
const viewerSubtitle = document.getElementById("history-viewer-subtitle");
const viewerFrame = document.getElementById("history-viewer-frame");
const viewerCount = document.getElementById("history-viewer-count");
const viewerMeta = document.getElementById("history-meta");
const viewerThumbs = document.getElementById("history-thumbs");
const prevButton = document.getElementById("history-prev");
const nextButton = document.getElementById("history-next");
const refreshButton = document.getElementById("history-refresh");
const isMounted = Boolean(
    historyState &&
        historyLayout &&
        historyBatchList &&
        viewerTitle &&
        viewerSubtitle &&
        viewerFrame &&
        viewerCount &&
        viewerMeta &&
        viewerThumbs &&
        prevButton &&
        nextButton &&
        refreshButton
);

function setState(next) {
    Object.assign(state, next);
    if (!isMounted) {
        return;
    }
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

function getBatchId(record) {
    const batchId = record?.metadata?.batch_id;
    if (batchId === undefined || batchId === null || batchId === "") {
        return "Unbatched";
    }
    return String(batchId);
}

function groupRecords(records) {
    const grouped = new Map();

    records.forEach((record) => {
        const id = getBatchId(record);
        if (!grouped.has(id)) {
            grouped.set(id, []);
        }
        grouped.get(id).push(record);
    });

    const batches = Array.from(grouped.entries()).map(([id, items]) => {
        const latest = items.reduce((max, item) => Math.max(max, item.timestamp ?? 0), 0);
        return { id, items, latest };
    });

    batches.sort((a, b) => b.latest - a.latest);
    return batches;
}

function selectBatch(batchId, index) {
    setState({ selectedBatchId: batchId, selectedIndex: index });
}

function renderBatchList() {
    historyBatchList.innerHTML = "";

    if (historyBatchCount) {
        const batchCount = state.batches.length;
        const renderCount = state.records.length;
        historyBatchCount.textContent = `${batchCount} batch${batchCount === 1 ? "" : "es"} | ${renderCount} render${renderCount === 1 ? "" : "s"}`;
    }

    state.batches.forEach((batch) => {
        if (!batch.items.length) {
            return;
        }
        const batchButton = document.createElement("button");
        batchButton.type = "button";
        batchButton.className = "history-batch";
        if (batch.id === state.selectedBatchId) {
            batchButton.classList.add("is-active");
        }
        batchButton.addEventListener("click", () => selectBatch(batch.id, 0));

        const preview = document.createElement("div");
        preview.className = "history-batch-preview";
        const previewImage = document.createElement("img");
        previewImage.loading = "lazy";
        previewImage.src = `${API_BASE}${batch.items[0].url}`;
        previewImage.alt = batch.items[0].filename || `Batch ${batch.id}`;
        preview.appendChild(previewImage);

        batchButton.appendChild(preview);
        historyBatchList.appendChild(batchButton);
    });
}

function updateNavigation(count) {
    const disabled = count <= 1;
    prevButton.disabled = disabled;
    nextButton.disabled = disabled;
}

function renderViewer() {
    viewerFrame.innerHTML = "";
    viewerThumbs.innerHTML = "";
    viewerMeta.innerHTML = "";

    const batch = state.batches.find((item) => item.id === state.selectedBatchId);
    if (!batch || !batch.items.length) {
        viewerSubtitle.textContent = "Pick a batch to inspect prompt metadata.";
        viewerCount.textContent = "";
        const empty = document.createElement("div");
        empty.className = "history-viewer-empty";
        empty.textContent = "Select a batch to preview.";
        viewerFrame.appendChild(empty);
        updateNavigation(0);
        return;
    }

    const selectedIndex = Math.min(state.selectedIndex, batch.items.length - 1);
    const record = batch.items[selectedIndex];

    viewerSubtitle.textContent = `Image ${selectedIndex + 1} of ${batch.items.length} | ${formatTimestamp(record.timestamp)}`;
    viewerCount.textContent = `Image ${selectedIndex + 1} of ${batch.items.length}`;

    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `${API_BASE}${record.url}`;
    img.alt = record.filename || `Render ${selectedIndex + 1}`;
    viewerFrame.appendChild(img);

    const metadataList = buildMetadata(record.metadata || {});
    if (metadataList) {
        viewerMeta.appendChild(metadataList);
    } else {
        const empty = document.createElement("p");
        empty.className = "history-empty-meta";
        empty.textContent = "No prompt metadata available.";
        viewerMeta.appendChild(empty);
    }

    batch.items.forEach((item, index) => {
        const thumb = document.createElement("img");
        thumb.className = "viewer-thumb";
        if (index === selectedIndex) {
            thumb.classList.add("is-active");
        }
        thumb.loading = "lazy";
        thumb.src = `${API_BASE}${item.url}`;
        thumb.alt = item.filename || `Render ${index + 1}`;
        thumb.addEventListener("click", () => selectBatch(batch.id, index));
        viewerThumbs.appendChild(thumb);
    });

    updateNavigation(batch.items.length);
}

function render() {
    if (!isMounted) {
        return;
    }
    if (state.loading) {
        historyState.textContent = "Loading render history...";
        historyState.style.display = "block";
        historyLayout.style.display = "none";
        return;
    }

    if (state.error) {
        historyState.textContent = state.error;
        historyState.style.display = "block";
        historyLayout.style.display = "none";
        return;
    }

    if (!state.records.length) {
        historyState.textContent = "No renders yet. Generate an image to populate the gallery.";
        historyState.style.display = "block";
        historyLayout.style.display = "none";
        return;
    }

    historyState.style.display = "none";
    historyLayout.style.display = "grid";

    renderBatchList();
    renderViewer();
}

function shiftSelection(direction) {
    const batch = state.batches.find((item) => item.id === state.selectedBatchId);
    if (!batch || !batch.items.length) {
        return;
    }
    const count = batch.items.length;
    const nextIndex = (state.selectedIndex + direction + count) % count;
    setState({ selectedIndex: nextIndex });
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
        const batches = groupRecords(records);

        let selectedBatchId = state.selectedBatchId;
        let selectedIndex = state.selectedIndex;
        if (!selectedBatchId || !batches.some((batch) => batch.id === selectedBatchId)) {
            selectedBatchId = batches[0]?.id ?? null;
            selectedIndex = 0;
        } else {
            const selectedBatch = batches.find((batch) => batch.id === selectedBatchId);
            if (selectedBatch && selectedIndex >= selectedBatch.items.length) {
                selectedIndex = 0;
            }
        }

        setState({ records, batches, selectedBatchId, selectedIndex, loading: false });
    } catch (error) {
        setState({ loading: false, error: "Unable to load history. Try again shortly." });
        console.error(error);
    }
}

if (isMounted) {
    refreshButton.addEventListener("click", fetchHistory);
    prevButton.addEventListener("click", () => shiftSelection(-1));
    nextButton.addEventListener("click", () => shiftSelection(1));
    fetchHistory();
} else {
    console.warn("History UI elements not found. Skipping history initialization.");
}
