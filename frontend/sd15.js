const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

const generateButton = document.getElementById("generate-button");
if (generateButton) {
    generateButton.addEventListener("click", () => {
        generate();
    });
}

function makeIdempotencyKey() {
    if (typeof crypto?.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

async function uploadArtifact(blobOrFile, filename = "upload.png") {
    const formData = new FormData();
    formData.append("file", blobOrFile, filename);
    const res = await fetch(`${API_BASE}/api/artifacts`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Artifact upload failed (${res.status}): ${errorText}`);
    }
    return await res.json();
}

async function submitWorkflow(payload, idempotencyKey) {
    const res = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotencyKey,
        },
        body: JSON.stringify({ kind: "workflow", payload }),
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Job submit failed (${res.status}): ${errorText}`);
    }
    return await res.json();
}

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=sd15`);
        const models = await res.json();

        if (!Array.isArray(models) || models.length === 0) {
            throw new Error("No models returned.");
        }

        models.forEach((model, index) => {
            const option = document.createElement("option");
            option.value = model.name ?? "";
            const family = model.family ?? "unknown";
            const modelType = model.model_type ?? "unknown";
            option.textContent = `${model.name} (${family}, ${modelType})`;
            if (index === 0) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    } catch (error) {
        const fallback = document.createElement("option");
        fallback.value = "stable-diffusion-v1-5";
        fallback.textContent = "stable-diffusion-v1-5 (sd15, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

const controlnetState = {
    previewUrl: null,
    previewBlob: null,
    preprocessors: new Map(),
};

function updateControlNetIndicator() {
    const indicator = document.getElementById("controlnet-indicator");
    const enabledToggle = document.getElementById("controlnet-enabled");
    const status = document.getElementById("controlnet-status");
    const isActive = Boolean(enabledToggle?.checked && controlnetState.previewUrl);
    if (indicator) {
        indicator.classList.toggle("is-active", isActive);
    }
    if (status) {
        status.textContent = isActive
            ? "ControlNet preprocessor ready for SD1.5."
            : "No preprocessor applied.";
    }
}

function updateControlNetActiveFlag() {
    const flag = document.getElementById("controlnet-active-flag");
    if (!flag) {
        return;
    }
    const enabledToggle = document.getElementById("controlnet-enabled");
    const isActive = Boolean(enabledToggle?.checked && controlnetState.previewUrl);
    flag.classList.toggle("is-hidden", !isActive);
    flag.style.display = isActive ? "inline-flex" : "none";
}

function toggleControlNetPanel() {
    const content = document.getElementById("controlnet-content");
    const chevron = document.getElementById("controlnet-chevron");
    if (!content || !chevron) {
        return;
    }
    const isOpen = content.classList.toggle("is-open");
    chevron.textContent = isOpen ? "▴" : "▾";
}

function updateDownloadLinkState(isReady) {
    const downloadLink = document.getElementById("download-control-image");
    if (!downloadLink) {
        return;
    }
    downloadLink.setAttribute("aria-disabled", isReady ? "false" : "true");
    downloadLink.classList.toggle("is-disabled", !isReady);
    if (!isReady) {
        downloadLink.href = "#";
    }
}

let controlNetUiReady = false;
let controlNetUiLoading = null;

async function loadControlNetModal() {
    const container = document.getElementById("controlnet-preprocessor-root");
    if (!container) {
        return;
    }
    try {
        const res = await fetch("controlnet_preprocessor.html");
        if (!res.ok) {
            throw new Error(`Failed to load ControlNet preprocessor UI: ${res.status}`);
        }
        container.innerHTML = await res.text();
    } catch (error) {
        console.warn("Failed to load ControlNet preprocessor UI:", error);
    }
}

async function loadControlNetPanel() {
    const container = document.getElementById("controlnet-panel-root");
    if (!container) {
        return;
    }
    try {
        const res = await fetch("controlnet_panel.html");
        if (!res.ok) {
            throw new Error(`Failed to load ControlNet panel UI: ${res.status}`);
        }
        container.innerHTML = await res.text();
    } catch (error) {
        console.warn("Failed to load ControlNet panel UI:", error);
    }
}

async function ensureControlNetUI() {
    if (controlNetUiReady) {
        return;
    }
    if (controlNetUiLoading) {
        return controlNetUiLoading;
    }
    controlNetUiLoading = (async () => {
        await loadControlNetPanel();
        await loadControlNetModal();
        const panel = document.getElementById("controlnet-toggle");
        const modal = document.getElementById("preprocessor-modal");
        if (!panel || !modal) {
            throw new Error("ControlNet preprocessor UI failed to load.");
        }
        initControlNetUI();
        controlNetUiReady = true;
    })()
        .catch((error) => {
            console.warn("ControlNet UI initialization failed:", error);
        })
        .finally(() => {
            controlNetUiLoading = null;
        });
    return controlNetUiLoading;
}
async function loadPreprocessors() {
    const select = document.getElementById("preprocessor-select");
    if (!select) {
        return;
    }
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/api/controlnet/preprocessors`);
        const preprocessors = await res.json();
        preprocessors.forEach((preprocessor) => {
            const option = document.createElement("option");
            option.value = preprocessor.id;
            option.textContent = preprocessor.name;
            select.appendChild(option);
            controlnetState.preprocessors.set(preprocessor.id, preprocessor);
        });
        updatePreprocessorDefaults(select.value);
    } catch (error) {
        const fallback = document.createElement("option");
        fallback.value = "canny";
        fallback.textContent = "Canny";
        select.appendChild(fallback);
        console.warn("Failed to load preprocessors:", error);
    }
}

function updatePreprocessorDefaults(preprocessorId) {
    const definition = controlnetState.preprocessors.get(preprocessorId);
    const defaults = definition?.defaults ?? {};
    const description = definition?.description ?? "";
    const lowThresholdInput = document.getElementById("preprocessor-low-threshold");
    const highThresholdInput = document.getElementById("preprocessor-high-threshold");
    const descriptionNode = document.getElementById("preprocessor-description");
    const cannyRow = document.getElementById("canny-thresholds");
    if (lowThresholdInput) {
        lowThresholdInput.value = Number(defaults.low_threshold ?? 100);
    }
    if (highThresholdInput) {
        highThresholdInput.value = Number(defaults.high_threshold ?? 200);
    }
    if (descriptionNode) {
        descriptionNode.textContent = description;
    }
    if (cannyRow) {
        const isCanny = preprocessorId === "canny";
        cannyRow.classList.toggle("is-hidden", !isCanny);
    }
}

function buildPreprocessorParams(preprocessorId) {
    const definition = controlnetState.preprocessors.get(preprocessorId);
    const params = { ...(definition?.defaults ?? {}) };
    if (preprocessorId === "canny") {
        const lowThresholdInput = document.getElementById("preprocessor-low-threshold");
        const highThresholdInput = document.getElementById("preprocessor-high-threshold");
        params.low_threshold = Number(lowThresholdInput?.value ?? params.low_threshold ?? 100);
        params.high_threshold = Number(highThresholdInput?.value ?? params.high_threshold ?? 200);
    }
    return params;
}

async function openPreprocessorModal() {
    await ensureControlNetUI();
    const modal = document.getElementById("preprocessor-modal");
    if (!modal) {
        return;
    }
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
}

function closePreprocessorModal() {
    const modal = document.getElementById("preprocessor-modal");
    if (!modal) {
        return;
    }
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
}

async function applyPreprocessor() {
    const fileInput = document.getElementById("preprocessor-image");
    const select = document.getElementById("preprocessor-select");
    const preview = document.getElementById("preprocessor-preview");
    const downloadLink = document.getElementById("download-control-image");
    const enabledToggle = document.getElementById("controlnet-enabled");

    if (!fileInput?.files?.length) {
        alert("Please select an input image for the preprocessor.");
        return;
    }
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    const selectedId = select?.value ?? "canny";
    formData.append("preprocessor_id", selectedId);
    formData.append("params", JSON.stringify(buildPreprocessorParams(selectedId)));

    const res = await fetch(`${API_BASE}/api/controlnet/preprocess`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        console.error("Preprocessor failed", res.status);
        alert("Preprocessor failed. Check the backend logs for details.");
        return;
    }

    const blob = await res.blob();
    if (controlnetState.previewUrl) {
        URL.revokeObjectURL(controlnetState.previewUrl);
    }
    controlnetState.previewUrl = URL.createObjectURL(blob);
    controlnetState.previewBlob = blob;
    if (preview) {
        preview.src = controlnetState.previewUrl;
    }
    if (downloadLink) {
        downloadLink.href = controlnetState.previewUrl;
        downloadLink.setAttribute("download", "controlnet_preprocessor.png");
    }
    updateDownloadLinkState(true);
    if (enabledToggle) {
        enabledToggle.checked = true;
    }
    updateControlNetIndicator();
    updateControlNetActiveFlag();
}

function initControlNetUI() {
    const toggleButton = document.getElementById("controlnet-toggle");
    const openButton = document.getElementById("open-preprocessors");
    const closeButton = document.getElementById("close-preprocessors");
    const overlay = document.getElementById("preprocessor-overlay");
    const applyButton = document.getElementById("apply-preprocessor");
    const enabledToggle = document.getElementById("controlnet-enabled");
    const select = document.getElementById("preprocessor-select");
    const fileInput = document.getElementById("preprocessor-image");

    toggleButton?.addEventListener("click", toggleControlNetPanel);
    openButton?.addEventListener("click", openPreprocessorModal);
    closeButton?.addEventListener("click", closePreprocessorModal);
    overlay?.addEventListener("click", closePreprocessorModal);
    applyButton?.addEventListener("click", applyPreprocessor);
    enabledToggle?.addEventListener("change", () => {
        updateControlNetIndicator();
        updateControlNetActiveFlag();
    });
    select?.addEventListener("change", (event) => {
        updatePreprocessorDefaults(event.target.value);
        updateDownloadLinkState(false);
    });
    fileInput?.addEventListener("change", () => {
        controlnetState.previewBlob = null;
        updateDownloadLinkState(false);
        updateControlNetActiveFlag();
    });

    loadPreprocessors();
    updateControlNetIndicator();
    updateControlNetActiveFlag();
    updateDownloadLinkState(false);
}

async function initControlNet() {
    await ensureControlNetUI();
}

initControlNet();

window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" });

async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const cfg = Number(document.getElementById("cfg").value);
    const scheduler = document.getElementById("scheduler").value;
    const seedValue = document.getElementById("seed").value;
    const seedNumber = seedValue === "" ? null : Number(seedValue);
    const seed = Number.isFinite(seedNumber) ? seedNumber : null;
    const negative_prompt = document.getElementById("negative_prompt").value;
    const width = Number(document.getElementById("width").value);
    const height = Number(document.getElementById("height").value);
    const hires_enabled = Boolean(document.getElementById("hires_enabled")?.checked);
    const hiresScaleInput = Number(document.getElementById("hires_scale").value);
    const hires_scale = Number.isFinite(hiresScaleInput) ? hiresScaleInput : 1.0;
    const model = document.getElementById("model_select").value;
    const clip_skip = document.getElementById("clip_skip").value;
    const num_images = Number(document.getElementById("num_images").value);
    const weighting_policy =
        document.getElementById("weighting_policy")?.value ?? "diffusers-like";
    const controlnetEnabled = Boolean(
        document.getElementById("controlnet-enabled")?.checked
    );
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];

    const idempotencyKey = makeIdempotencyKey();

    try {
        const tasks = [];

        if (controlnetEnabled) {
            if (!controlnetState.previewBlob) {
                throw new Error("ControlNet enabled but no preprocessor output image is ready.");
            }
            const uploaded = await uploadArtifact(controlnetState.previewBlob, "controlnet.png");
            const inputs = {
                control_image: `@artifact:${uploaded.artifact_id}`,
                prompt,
                negative_prompt,
                steps,
                cfg,
                scheduler,
                seed,
                width,
                height,
                model,
                num_images,
                clip_skip,
                weighting_policy,
            };
            if (loraAdapters.length > 0) {
                inputs.lora_adapters = loraAdapters;
            }
            tasks.push({ id: "t1", type: "sd15.controlnet.text2img", inputs });
        } else {
            const inputs = {
                prompt,
                negative_prompt,
                steps,
                cfg,
                scheduler,
                seed,
                width,
                height,
                model,
                num_images,
                clip_skip,
                weighting_policy,
            };
            if (loraAdapters.length > 0) {
                inputs.lora_adapters = loraAdapters;
            }
            tasks.push({ id: "t1", type: "sd15.text2img", inputs });
        }

        let returnRef = "@t1.images";
        if (hires_enabled && hires_scale > 1.0) {
            tasks.push({
                id: "hires",
                type: "sd15.hires_fix",
                inputs: {
                    images: "@t1.images",
                    prompt,
                    negative_prompt,
                    steps,
                    cfg,
                    scheduler,
                    seed,
                    model,
                    clip_skip,
                    hires_scale,
                    weighting_policy,
                    lora_adapters: loraAdapters,
                },
            });
            returnRef = "@hires.images";
        }

        const workflowPayload = { tasks, return: returnRef };
        const createdJob = await submitWorkflow(workflowPayload, idempotencyKey);
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        const source = new EventSource(`${API_BASE}/api/jobs/${jobId}/events`);
        activeEventSource = source;

        source.onmessage = (event) => {
            if (token !== activeJobToken) {
                source.close();
                return;
            }

            let job;
            try {
                job = JSON.parse(event.data);
            } catch {
                return;
            }

            const status = job?.status ?? "unknown";
            if (status === "succeeded") {
                const images = job?.result?.outputs;
                gallery.setImages(Array.isArray(images) ? images : []);
                source.close();
            } else if (status === "failed" || status === "canceled") {
                gallery.setImages([]);
                source.close();
            }
        };

        source.onerror = () => {
            if (token !== activeJobToken) {
                source.close();
                return;
            }
            source.close();
        };
    } catch (error) {
        console.warn("Failed to generate SD1.5 images:", error);
        gallery.setImages([]);
    }
}
