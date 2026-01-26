/**
 * SD1.5 (Stable Diffusion 1.5) UI wiring.
 *
 * Responsibilities:
 * - Read values from the SD1.5 form controls.
 * - Submit a workflow job to the backend and stream status updates via SSE.
 * - Populate the gallery with returned images.
 * - Optionally run a ControlNet preprocessor and attach its output to the workflow.
 *
 * This file assumes `API_BASE` and `createGalleryViewer()` are provided globally
 * (typically by the hosting HTML page).
 */

// Gallery viewer for displaying generated outputs (with light cache-busting).
const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

// Token incremented per generate() call to ignore stale SSE events from prior jobs.
let activeJobToken = 0;
// Currently active SSE connection (closed on new generation).
let activeEventSource = null;

/**
 * Close any active SSE connection and clear the local reference.
 * Safe to call multiple times.
 */
function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

/**
 * Populate the SD1.5 model dropdown from the backend model registry.
 * Falls back to a sane default if the request fails.
 */
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

// ControlNet UI state shared across the preprocessor panel and generate() flow.
const controlnetState = {
    previewUrl: null,
    previewBlob: null,
    preprocessors: new Map(),
};

/**
 * Update the ControlNet status indicator based on whether it's enabled and a
 * valid preprocessed image is ready.
 */
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

/**
 * Show/hide the "active" flag/pill for ControlNet in the UI.
 */
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

/**
 * Toggle open/closed state for the ControlNet configuration panel.
 */
function toggleControlNetPanel() {
    const content = document.getElementById("controlnet-content");
    const chevron = document.getElementById("controlnet-chevron");
    if (!content || !chevron) {
        return;
    }
    const isOpen = content.classList.toggle("is-open");
    chevron.textContent = isOpen ? "▴" : "▾";
}

/**
 * Enable/disable the "download preprocessor output" link.
 *
 * When disabled we clear the href to avoid downloading an old preview.
 *
 * @param {boolean} isReady
 */
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

/**
 * Load the preprocessor modal HTML fragment into its container.
 * Split into a separate file to keep the main SD1.5 page lighter.
 */
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

/**
 * Load the ControlNet panel HTML fragment into its container.
 */
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

/**
 * Ensure the ControlNet UI fragments are loaded and event listeners are bound.
 *
 * This function is idempotent and returns a shared promise so multiple callers
 * don't trigger duplicate fetches/initialization.
 */
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

/**
 * Load available ControlNet preprocessors from the backend and populate the select.
 * Caches definitions in `controlnetState.preprocessors` for later defaults rendering.
 */
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

/**
 * Update UI controls to reflect the selected preprocessor's default parameters.
 * Currently only exposes Canny thresholds in the UI, but the backend supports
 * arbitrary parameter objects.
 *
 * @param {string} preprocessorId
 */
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

/**
 * Build the params object sent to the backend `/api/controlnet/preprocess` endpoint.
 *
 * @param {string} preprocessorId
 * @returns {object}
 */
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

/**
 * Open the ControlNet preprocessor modal (ensuring the UI is loaded first).
 */
async function openPreprocessorModal() {
    await ensureControlNetUI();
    const modal = document.getElementById("preprocessor-modal");
    if (!modal) {
        return;
    }
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
}

/**
 * Close the ControlNet preprocessor modal (no-op if missing).
 */
function closePreprocessorModal() {
    const modal = document.getElementById("preprocessor-modal");
    if (!modal) {
        return;
    }
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
}

/**
 * Run the selected preprocessor on the selected image and update preview state.
 *
 * Persists the resulting blob/URL so `generate()` can upload it as an artifact
 * when ControlNet is enabled.
 */
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
        // Avoid leaking object URLs when users iterate on preprocess settings.
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

/**
 * Bind ControlNet UI event handlers and prime initial UI state.
 * Safe to call once after the HTML fragments have been injected.
 */
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

/**
 * Entry-point for ControlNet wiring (loads HTML fragments + binds handlers).
 */
async function initControlNet() {
    await ensureControlNetUI();
}

/**
 * Initialize page-level UI integrations.
 *
 * Centralizes one-time setup calls to keep the global scope tidy.
 */
function initSd15Page() {
    if (didInitSd15Page) {
        return;
    }
    didInitSd15Page = true;

    gallery.render();

    const generateButton = document.getElementById("generate-button");
    generateButton?.addEventListener("click", () => {
        generate();
    });

    void loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm("sd15.text2img", {
                    steps: "steps",
                    cfg: "cfg",
                    width: "width",
                    height: "height",
                    num_images: "num_images",
                    clip_skip: "clip_skip",
                    weighting_policy: "weighting_policy",
                });
                window.WorkflowCatalog.applyDefaultsToForm("sd15.controlnet.text2img", {
                    steps: "steps",
                    cfg: "cfg",
                    width: "width",
                    height: "height",
                    num_images: "num_images",
                    clip_skip: "clip_skip",
                });
                window.WorkflowCatalog.applyDefaultsToForm("sd15.hires_fix", {
                    hires_scale: "hires_scale",
                });
            })
            .catch(() => {});
    }
    initControlNet().catch((error) => {
        console.warn("ControlNet init failed:", error);
    });

    // Optional LoRA panel integration (only active if that script is present on the page).
    window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" });
}

let didInitSd15Page = false;

/**
 * Run an initializer once the DOM is ready (or immediately if already ready).
 *
 * @param {() => void} initFn
 */
function runWhenDomReady(initFn) {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initFn, { once: true });
        return;
    }
    initFn();
}

runWhenDomReady(initSd15Page);

/**
 * Collect inputs, submit a workflow job, and stream results into the gallery.
 *
 * High-level flow:
 * 1) Read form fields and normalize types.
 * 2) Optionally upload ControlNet preprocessor output (artifact).
 * 3) Build workflow tasks (`sd15.text2img` or ControlNet variant; optional hires fix).
 * 4) Submit job and attach SSE listener for status updates.
 */
async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);
    const primaryTaskType = controlnetEnabled ? "sd15.controlnet.text2img" : "sd15.text2img";

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const primaryDefaults = catalog?.tasks?.[primaryTaskType]?.input_defaults ?? {};
    const hiresDefaults = catalog?.tasks?.["sd15.hires_fix"]?.input_defaults ?? {};

    const prompt = WorkflowClient.readTextValue("prompt", "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        primaryDefaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", primaryDefaults.steps ?? 20, {
        integer: true,
    });
    const cfg = WorkflowClient.readNumberValue("cfg", primaryDefaults.cfg ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", primaryDefaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", primaryDefaults.width ?? 512, {
        integer: true,
    });
    const height = WorkflowClient.readNumberValue("height", primaryDefaults.height ?? 512, {
        integer: true,
    });
    const hires_enabled = Boolean(document.getElementById("hires_enabled")?.checked);
    const hires_scale = WorkflowClient.readNumberValue("hires_scale", hiresDefaults.hires_scale ?? 1.0);
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (primaryDefaults.model ?? null);
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", primaryDefaults.clip_skip ?? 1, {
        integer: true,
    });
    const num_images = WorkflowClient.readNumberValue("num_images", primaryDefaults.num_images ?? 1, {
        integer: true,
    });
    const weighting_policy = WorkflowClient.readTextValue(
        "weighting_policy",
        primaryDefaults.weighting_policy ?? "diffusers-like"
    );
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];

    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    try {
        const tasks = [];

        if (controlnetEnabled) {
            if (!controlnetState.previewBlob) {
                throw new Error("ControlNet enabled but no preprocessor output image is ready.");
            }
            // ControlNet tasks reference a stored artifact for the control image input.
            const uploaded = await WorkflowClient.uploadArtifact(
                API_BASE,
                controlnetState.previewBlob,
                "controlnet.png"
            );
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
            // Optional 2nd-pass upscaling/refinement ("hires fix") on the outputs of t1.
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
                },
            });
            if (loraAdapters.length > 0) {
                tasks[tasks.length - 1].inputs.lora_adapters = loraAdapters;
            }
            returnRef = "@hires.images";
        }

        const workflowPayload = { tasks, return: returnRef };
        const createdJob = await WorkflowClient.submitWorkflow(API_BASE, workflowPayload, idempotencyKey);
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        activeEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeJobToken,
            onDone: (job) => {
                const status = job?.status ?? "unknown";
                if (status === "succeeded") {
                    const images = job?.result?.outputs;
                    gallery.setImages(Array.isArray(images) ? images : []);
                } else {
                    gallery.setImages([]);
                }
            },
            onError: () => {
                if (token !== activeJobToken) {
                    return;
                }
                gallery.setImages([]);
            },
        });
    } catch (error) {
        console.warn("Failed to generate SD1.5 images:", error);
        gallery.setImages([]);
    }
}
