const form = document.getElementById("tools-form");
const fileInput = document.getElementById("tools-file");
const limitInput = document.getElementById("tools-limit");
const submitButton = document.getElementById("tools-submit");
const stateMessage = document.getElementById("tools-state");
const resultsSection = document.getElementById("tools-results");
const fileNameLabel = document.getElementById("tools-file-name");
const summaryMeta = document.getElementById("tools-summary-meta");
const summaryCount = document.getElementById("tools-summary-count");
const architectureValue = document.getElementById("tools-architecture-value");
const architectureDetail = document.getElementById("tools-architecture-detail");
const architectureEvidence = document.getElementById("tools-architecture-evidence");
const tableBody = document.getElementById("tools-table-body");

function setState(message, isError = false) {
    stateMessage.textContent = message;
    stateMessage.classList.toggle("is-error", isError);
}

function clearResults() {
    tableBody.innerHTML = "";
    resultsSection.hidden = true;
    fileNameLabel.textContent = "";
    summaryMeta.textContent = "";
    summaryCount.textContent = "";
    architectureValue.textContent = "Unknown";
    architectureDetail.textContent = "";
    architectureEvidence.innerHTML = "";
}

function renderRows(rows) {
    tableBody.innerHTML = "";
    rows.forEach((row) => {
        const tr = document.createElement("tr");
        const keyCell = document.createElement("td");
        keyCell.textContent = row.key;
        const shapeCell = document.createElement("td");
        shapeCell.textContent = row.shape;
        const dtypeCell = document.createElement("td");
        dtypeCell.textContent = row.dtype;
        tr.appendChild(keyCell);
        tr.appendChild(shapeCell);
        tr.appendChild(dtypeCell);
        tableBody.appendChild(tr);
    });
}

function formatArchitecture(value) {
    const labels = {
        sd15: "SD1.5",
        sd2: "SD2.x",
        sdxl: "SDXL",
        flux: "Flux",
        "qwen-image": "Qwen-Image",
        "z-image": "Z-Image",
        "ernie-image": "ERNIE-Image",
    };
    return labels[value] || "Unknown";
}

function renderArchitecture(result) {
    const confidence = result.architecture_confidence || "unknown";
    const metadataMessage = result.metadata_available
        ? `Safetensors metadata available (${(result.metadata_keys || []).length} keys).`
        : "Safetensors metadata is not present or not available.";

    architectureValue.textContent = `${formatArchitecture(result.architecture)} (${confidence} confidence)`;
    architectureDetail.textContent = metadataMessage;
    architectureEvidence.innerHTML = "";

    const evidence = Array.isArray(result.architecture_evidence) ? result.architecture_evidence : [];
    evidence.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        architectureEvidence.appendChild(li);
    });
}

async function analyzeModel(file, limitValue) {
    const formData = new FormData();
    formData.append("file", file);
    if (Number.isFinite(limitValue) && limitValue > 0) {
        formData.append("limit", String(limitValue));
    }

    const response = await fetch(`${API_BASE}/api/tools/analyze-model`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to analyze model.");
    }

    return response.json();
}

form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearResults();

    const file = fileInput?.files?.[0];
    if (!file) {
        setState("Please choose a model file.", true);
        return;
    }

    const limitValue = Number.parseInt(limitInput?.value || "0", 10) || 0;

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
    setState("Analyzing model. This can take a moment for large files.");

    try {
        const result = await analyzeModel(file, limitValue);
        fileNameLabel.textContent = result.file_name;
        summaryMeta.textContent = `Loaded via ${result.loader}`;
        summaryCount.textContent = `${result.returned} of ${result.total} layers shown`;
        renderArchitecture(result);
        renderRows(result.rows || []);
        resultsSection.hidden = false;
        setState("Analysis complete.");
    } catch (error) {
        console.error(error);
        setState(error.message || "Failed to analyze model.", true);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Analyze model";
    }
});
