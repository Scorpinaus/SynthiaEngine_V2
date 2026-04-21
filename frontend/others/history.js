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

const HISTORY_IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp"]);
const HISTORY_VIDEO_EXTENSIONS = new Set([".mp4", ".webm", ".mov"]);

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

function getPathExtension(record) {
    const path = String(record?.filename || record?.url || "").split("?")[0].split("#")[0];
    const dotIndex = path.lastIndexOf(".");
    return dotIndex >= 0 ? path.slice(dotIndex).toLowerCase() : "";
}

function getMediaType(record) {
    const mediaType = String(record?.media_type || "").toLowerCase();
    if (mediaType === "image" || mediaType === "video") {
        return mediaType;
    }

    const extension = getPathExtension(record);
    if (HISTORY_VIDEO_EXTENSIONS.has(extension)) {
        return "video";
    }
    if (HISTORY_IMAGE_EXTENSIONS.has(extension)) {
        return "image";
    }
    return "render";
}

function getMediaLabel(record) {
    const mediaType = getMediaType(record);
    if (mediaType === "video") {
        return "Video";
    }
    if (mediaType === "image") {
        return "Image";
    }
    return "Render";
}

function buildMediaUrl(record) {
    const url = String(record?.url || "");
    if (!url) {
        return "";
    }
    if (/^(https?:)?\/\//.test(url) || url.startsWith("data:") || url.startsWith("blob:")) {
        return url;
    }
    return `${API_BASE}${url}`;
}

function buildMetadata(metadata) {
    const list = document.createElement("dl");
    list.className = "history-meta-list";
    let hasEntries = false;
    const knownKeys = new Set(METADATA_FIELDS.map(({ key }) => key));

    function normalizeLabel(key) {
        return key
            .split("_")
            .filter(Boolean)
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");
    }

    function normalizeValue(value) {
        if (value === undefined || value === null || value === "") {
            return null;
        }
        if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
            return String(value);
        }
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }

    function appendRow(label, rawValue) {
        const value = normalizeValue(rawValue);
        if (value === null) {
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
    }

    METADATA_FIELDS.forEach(({ key, label }) => {
        appendRow(label, metadata?.[key]);
    });

    Object.keys(metadata || {}).forEach((key) => {
        if (knownKeys.has(key)) {
            return;
        }
        appendRow(normalizeLabel(key), metadata[key]);
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
        const firstItem = batch.items[0];
        if (getMediaType(firstItem) === "video") {
            const previewVideo = document.createElement("div");
            previewVideo.className = "history-video-preview";
            previewVideo.textContent = "Video";
            preview.appendChild(previewVideo);
        } else {
            const previewImage = document.createElement("img");
            previewImage.loading = "lazy";
            previewImage.src = buildMediaUrl(firstItem);
            previewImage.alt = firstItem.filename || `Batch ${batch.id}`;
            preview.appendChild(previewImage);
        }

        batchButton.appendChild(preview);
        historyBatchList.appendChild(batchButton);
    });
}

function updateNavigation(count, selectedIndex) {
    if (count <= 1) {
        prevButton.disabled = true;
        nextButton.disabled = true;
        return;
    }
    prevButton.disabled = selectedIndex <= 0;
    nextButton.disabled = selectedIndex >= count - 1;
}

function renderViewer() {
    viewerFrame.innerHTML = "";
    viewerThumbs.innerHTML = "";
    viewerMeta.innerHTML = "";

    const batch = state.batches.find((item) => item.id === state.selectedBatchId);
    if (!batch || !batch.items.length) {
        viewerTitle.textContent = "";
        viewerSubtitle.textContent = "Pick a batch to inspect prompt metadata.";
        viewerCount.textContent = "";
        const empty = document.createElement("div");
        empty.className = "history-viewer-empty";
        empty.textContent = "Select a batch to preview.";
        viewerFrame.appendChild(empty);
        updateNavigation(0, 0);
        return;
    }

    const selectedIndex = Math.min(state.selectedIndex, batch.items.length - 1);
    const record = batch.items[selectedIndex];
    const mediaLabel = getMediaLabel(record);
    const mediaType = getMediaType(record);

    viewerTitle.textContent = batch.id === "Unbatched" ? "Unbatched renders" : `Batch ${batch.id}`;
    viewerSubtitle.textContent = `${mediaLabel} ${selectedIndex + 1} of ${batch.items.length} | ${formatTimestamp(record.timestamp)}`;
    viewerCount.textContent = `${mediaLabel} ${selectedIndex + 1} of ${batch.items.length}`;

    if (mediaType === "video") {
        const video = document.createElement("video");
        video.controls = true;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.src = buildMediaUrl(record);
        viewerFrame.appendChild(video);
    } else {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = buildMediaUrl(record);
        img.alt = record.filename || `Render ${selectedIndex + 1}`;
        viewerFrame.appendChild(img);
    }

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
        const itemMediaType = getMediaType(item);
        const thumb = document.createElement(itemMediaType === "video" ? "button" : "img");
        thumb.className = itemMediaType === "video" ? "viewer-thumb history-video-thumb" : "viewer-thumb";
        if (index === selectedIndex) {
            thumb.classList.add("is-active");
        }
        if (itemMediaType === "video") {
            thumb.type = "button";
            thumb.textContent = `Video ${index + 1}`;
        } else {
            thumb.loading = "lazy";
            thumb.src = buildMediaUrl(item);
            thumb.alt = item.filename || `Render ${index + 1}`;
        }
        thumb.addEventListener("click", () => selectBatch(batch.id, index));
        viewerThumbs.appendChild(thumb);
    });

    updateNavigation(batch.items.length, selectedIndex);
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
        historyState.textContent = "No renders yet. Generate an image or video to populate the gallery.";
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
    const nextIndex = Math.min(Math.max(state.selectedIndex + direction, 0), count - 1);
    if (nextIndex !== state.selectedIndex) {
        setState({ selectedIndex: nextIndex });
    }
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
