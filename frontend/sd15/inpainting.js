/**
 * SD1.5 inpainting UI wiring.
 *
 * Responsibilities:
 * - Load an initial image and render it into a canvas.
 * - Let the user paint an inpainting mask (white = inpaint, black = keep).
 * - Optionally blur mask edges using a backend helper endpoint.
 * - Upload base+mask artifacts, submit an `sd15.inpaint` workflow job,
 *   and stream status updates via SSE into the gallery.
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

const DEFAULTS = {
    steps: 20,
    lcm_steps: 4,
    cfg: 7.5,
    lcm_cfg: 0,
    scheduler: "euler",
    lcm_scheduler: "lcm",
    ip_adapter_scale: 0.6,
};

// Token incremented per generateInpaint() call to ignore stale SSE events from prior jobs.
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

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
}

function getIpAdapterImageFile() {
    return document.getElementById("ip_adapter_image")?.files?.[0] ?? null;
}

let controlNetUiReady = Promise.resolve();
let loraPanelReady = Promise.resolve();

function countLabel(count, singular, plural = `${singular}s`) {
    const value = Number(count);
    const safeCount = Number.isFinite(value) ? value : 0;
    return `${safeCount} ${safeCount === 1 ? singular : plural}`;
}

function setText(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

function collectAdapterSummaries() {
    const controlFallbackState = getControlNetState();
    const loraFallback = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const control = window.ControlNetPanel?.getSummary?.() ?? {
        availablePreprocessors: controlFallbackState?.preprocessors?.size ?? 0,
        totalPreprocessors: controlFallbackState?.preprocessors?.size ?? 0,
        activeItems: controlFallbackState?.controlItems?.length ?? 0,
        enabled: Boolean(document.getElementById("controlnet-enabled")?.checked),
    };
    const lora = window.LoraPanel?.getSummary?.() ?? {
        available: 0,
        selected: Array.isArray(loraFallback) ? loraFallback.length : 0,
    };
    const ipAdapter = window.IpAdapterPanel?.getSummary?.() ?? {
        availableAdapters: 1,
        enabled: Boolean(document.getElementById("ip_adapter_enabled")?.checked),
        hasReference: Boolean(document.getElementById("ip_adapter_image")?.files?.[0]),
        hasMask: Boolean(window.IpAdapterPanel?.getMaskFile?.()),
    };
    return { control, lora, ipAdapter };
}

function updateAdapterSummary() {
    const { control, lora, ipAdapter } = collectAdapterSummaries();
    const availableControl = Number(control.availablePreprocessors ?? control.totalPreprocessors ?? 0);
    const activeControlItems = Number(control.activeItems ?? 0);
    const availableLoras = Number(lora.available ?? 0);
    const selectedLoras = Number(lora.selected ?? 0);
    const availableIpAdapters = Number(ipAdapter.availableAdapters ?? 1);
    const ipActive = Boolean(ipAdapter.enabled);
    const totalAvailable =
        (Number.isFinite(availableControl) ? availableControl : 0) +
        (Number.isFinite(availableLoras) ? availableLoras : 0) +
        (Number.isFinite(availableIpAdapters) ? availableIpAdapters : 0);
    const activeAdapterCount =
        (control.enabled && activeControlItems > 0 ? activeControlItems : 0) +
        (Number.isFinite(selectedLoras) ? selectedLoras : 0) +
        (ipActive ? 1 : 0);

    setText(
        "adapter_summary_label",
        `${countLabel(totalAvailable, "adapter available", "adapters available")} / ${countLabel(activeAdapterCount, "adapter active", "adapters active")}`
    );
    setText("adapter-tab-controlnet-badge", countLabel(activeControlItems, "active", "active"));
    setText("adapter-tab-lora-badge", countLabel(selectedLoras, "selected", "selected"));
    setText("adapter-tab-ipadapter-badge", ipActive ? "on" : "off");

    setText(
        "adapter-overview-controlnet-count",
        countLabel(availableControl, "preprocessor available", "preprocessors available")
    );
    setText(
        "adapter-overview-controlnet-detail",
        control.enabled && activeControlItems > 0
            ? `${countLabel(activeControlItems, "control image")} active.`
            : "No control images active."
    );
    setText("adapter-overview-lora-count", countLabel(availableLoras, "LoRA available", "LoRAs available"));
    setText(
        "adapter-overview-lora-detail",
        selectedLoras > 0 ? `${countLabel(selectedLoras, "LoRA")} selected.` : "No LoRAs selected."
    );
    setText(
        "adapter-overview-ipadapter-count",
        countLabel(availableIpAdapters, "IP-Adapter available", "IP-Adapters available")
    );
    setText(
        "adapter-overview-ipadapter-detail",
        ipActive
            ? `Image prompt enabled${ipAdapter.hasReference ? " with reference image" : ""}${ipAdapter.hasMask ? " and mask" : ""}.`
            : "Image prompt disabled."
    );
}

function setAdapterTab(tabName) {
    const target = String(tabName || "overview");
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
        const isActive = tab.getAttribute("data-adapter-tab") === target;
        tab.classList.toggle("is-active", isActive);
        tab.setAttribute("aria-selected", String(isActive));
    });
    document.querySelectorAll("[data-adapter-panel]").forEach((panel) => {
        const isActive = panel.getAttribute("data-adapter-panel") === target;
        panel.classList.toggle("is-active", isActive);
        panel.toggleAttribute("hidden", !isActive);
    });
    updateAdapterSummary();
}

function setAdapterModalOpen(isOpen) {
    const modal = document.getElementById("adapter-modal");
    if (!modal) {
        return;
    }
    modal.classList.toggle("hidden", !isOpen);
    modal.setAttribute("aria-hidden", String(!isOpen));
    if (isOpen) {
        updateAdapterSummary();
        document.getElementById("adapter-modal-close")?.focus();
    } else {
        document.getElementById("adapter-modal-open")?.focus();
    }
}

function initAdapterModal() {
    const modal = document.getElementById("adapter-modal");
    if (!modal) {
        return;
    }
    document.getElementById("adapter-modal-open")?.addEventListener("click", () => {
        setAdapterModalOpen(true);
    });
    document.getElementById("adapter-modal-close")?.addEventListener("click", () => {
        setAdapterModalOpen(false);
    });
    document.getElementById("adapter-modal-overlay")?.addEventListener("click", () => {
        setAdapterModalOpen(false);
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !modal.classList.contains("hidden")) {
            setAdapterModalOpen(false);
        }
    });
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
        tab.addEventListener("click", () => {
            setAdapterTab(tab.getAttribute("data-adapter-tab"));
        });
    });
    document.querySelectorAll("[data-adapter-tab-jump]").forEach((button) => {
        button.addEventListener("click", () => {
            setAdapterTab(button.getAttribute("data-adapter-tab-jump"));
        });
    });
    window.addEventListener("adapter-summary-changed", updateAdapterSummary);
    modal.addEventListener("change", () => {
        window.setTimeout(updateAdapterSummary, 0);
    });
    modal.addEventListener("click", () => {
        window.setTimeout(updateAdapterSummary, 0);
    });
    updateAdapterSummary();
}

function isLcmModeEnabled() {
    return Boolean(document.getElementById("lcm_enabled")?.checked);
}

function setInputValue(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) {
        return;
    }
    el.value = value === null ? "" : String(value);
}

function setCheckboxValue(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) {
        return;
    }
    el.checked = Boolean(value);
}

function syncLcmModeDefaults() {
    if (!isLcmModeEnabled()) {
        const scheduler = document.getElementById("scheduler");
        if (scheduler?.value === DEFAULTS.lcm_scheduler) {
            setInputValue("scheduler", DEFAULTS.scheduler);
        }
        return;
    }
    setInputValue("steps", DEFAULTS.lcm_steps);
    setInputValue("cfg", DEFAULTS.lcm_cfg);
    setInputValue("scheduler", DEFAULTS.lcm_scheduler);
}

function setModelSelection(value) {
    if (value === undefined) {
        return;
    }
    const select = document.getElementById("model_select");
    if (!select) {
        return;
    }
    if (value === null || value === "") {
        select.value = "";
        return;
    }
    const normalized = String(value);
    const hasOption = Array.from(select.options).some((opt) => opt.value === normalized);
    if (!hasOption) {
        const option = document.createElement("option");
        option.value = normalized;
        option.textContent = `${normalized} (preset)`;
        select.appendChild(option);
    }
    select.value = normalized;
}

function collectSd15InpaintPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 20, { integer: true }),
        cfg: WorkflowClient.readNumberValue("cfg", 7.5),
        scheduler: WorkflowClient.readTextValue("scheduler", "euler"),
        seed: WorkflowClient.readSeedValue("seed"),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
        model: WorkflowClient.readTextValue("model_select", null),
        strength: WorkflowClient.readNumberValue("strength", 0.5),
        padding_mask_crop: WorkflowClient.readNumberValue("padding_mask_crop", 32, {
            integer: true,
        }),
        clip_skip: WorkflowClient.readNumberValue("clip_skip", 1, { integer: true }),
        controlnet_enabled: Boolean(document.getElementById("controlnet-enabled")?.checked),
        controlnet_conditioning_scale: WorkflowClient.readNumberValue(
            "controlnet_conditioning_scale",
            1.0
        ),
        control_guidance_start: WorkflowClient.readNumberValue("control_guidance_start", 0.0),
        control_guidance_end: WorkflowClient.readNumberValue("control_guidance_end", 1.0),
        controlnet_guess_mode: Boolean(document.getElementById("controlnet_guess_mode")?.checked),
        controlnet_compat_mode: WorkflowClient.readTextValue("controlnet_compat_mode", "warn"),
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
        lcm_enabled: isLcmModeEnabled(),
        ip_adapter_enabled: Boolean(document.getElementById("ip_adapter_enabled")?.checked),
        ip_adapter_scale: WorkflowClient.readNumberValue(
            "ip_adapter_scale",
            DEFAULTS.ip_adapter_scale
        ),
    };
}

async function applySd15InpaintPresetSettings(settings) {
    await Promise.all([controlNetUiReady, loraPanelReady]);

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.cfg);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("num_images", settings.num_images);
    setModelSelection(settings.model);
    setInputValue("strength", settings.strength);
    setInputValue("padding_mask_crop", settings.padding_mask_crop);
    setInputValue("clip_skip", settings.clip_skip);
    setCheckboxValue("controlnet-enabled", settings.controlnet_enabled);
    setCheckboxValue("lcm_enabled", settings.lcm_enabled);
    setInputValue("controlnet_conditioning_scale", settings.controlnet_conditioning_scale);
    setInputValue("control_guidance_start", settings.control_guidance_start);
    setInputValue("control_guidance_end", settings.control_guidance_end);
    setCheckboxValue("controlnet_guess_mode", settings.controlnet_guess_mode);
    setInputValue("controlnet_compat_mode", settings.controlnet_compat_mode);
    setCheckboxValue("ip_adapter_enabled", settings.ip_adapter_enabled);
    setInputValue("ip_adapter_scale", settings.ip_adapter_scale);
    if (settings.lcm_enabled) {
        syncLcmModeDefaults();
    }

    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
    }

    window.ControlNetPanel?.clearControlItems?.();
    window.ControlNetPanel?.updateIndicator?.();
    window.ControlNetPanel?.updateActiveFlag?.();
}

// DOM references (expected to exist in the hosting HTML).
const baseCanvas = document.getElementById("base_canvas");
const maskCanvas = document.getElementById("mask_canvas");
const canvasStack = document.querySelector(".canvas-stack");
const canvasScroll = document.querySelector(".canvas-scroll");
const imageInfo = document.getElementById("image_info");
const brushSizeInput = document.getElementById("brush_size");
const brushValue = document.getElementById("brush_value");
const zoomInput = document.getElementById("zoom_level");
const zoomValue = document.getElementById("zoom_value");
const eraseToggle = document.getElementById("erase_toggle");
const initialImageInput = document.getElementById("initial_image");
const maskModal = document.getElementById("mask_modal");
const maskPreview = document.getElementById("mask_preview");
const maskPreviewPanel = document.getElementById("mask_preview_panel");
const maskBlurButton = document.getElementById("mask_blur");
const blurFactorInput = document.getElementById("blur_factor");
const blurToggle = document.getElementById("blur_toggle");

// Current inpainting session state.
let baseImageFile = null;
let baseImage = null;
let isDrawing = false;
let maskBlob = null;
let maskDataUrl = null;
let blurMaskBlob = null;
let blurMaskDataUrl = null;
// Scale factor from image pixels -> on-screen canvas CSS pixels (fit-to-view + zoom).
let displayScale = 1;

// 2D contexts: base shows the uploaded image; mask stores the editable mask.
const baseContext = baseCanvas.getContext("2d");
const maskContext = maskCanvas.getContext("2d");

/**
 * Update the brush size label to mirror the current slider value.
 */
function updateBrushLabel() {
    brushValue.textContent = brushSizeInput.value;
}

/**
 * Update the zoom label to mirror the current slider value.
 */
function updateZoomLabel() {
    zoomValue.textContent = zoomInput.value;
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

let didInitSd15InpaintingPage = false;

/**
 * Initialize page-level UI integrations.
 *
 * Centralizes one-time setup calls to keep the global scope tidy.
 */
function initSd15InpaintingPage() {
    if (didInitSd15InpaintingPage) {
        return;
    }
    didInitSd15InpaintingPage = true;

    // Render the gallery shell immediately (images will be set after a job completes).
    gallery.render();
    window.AdapterPanel?.render?.();
    initAdapterModal();
    window.IpAdapterPanel?.init({
        getMaskBackdropFile: () => baseImageFile,
    });

    updateBrushLabel();
    brushSizeInput.addEventListener("input", updateBrushLabel);
    document.getElementById("lcm_enabled")?.addEventListener("change", syncLcmModeDefaults);

    updateZoomLabel();
    zoomInput.addEventListener("input", () => {
        updateZoomLabel();
        if (baseImage) {
            // Recompute the on-screen canvas size when zoom changes.
            resizeCanvasDisplay(baseImage);
        }
    });
 
    void loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm("sd15.inpaint", {
                    steps: "steps",
                    cfg: "cfg",
                    strength: "strength",
                    num_images: "num_images",
                    padding_mask_crop: "padding_mask_crop",
                    clip_skip: "clip_skip",
                    controlnet_conditioning_scale: "controlnet_conditioning_scale",
                    control_guidance_start: "control_guidance_start",
                    control_guidance_end: "control_guidance_end",
                    controlnet_compat_mode: "controlnet_compat_mode",
                    ip_adapter_scale: "ip_adapter_scale",
                });
            })
            .catch(() => {});
    }
    if (window.ControlNetPreprocessor?.init) {
        controlNetUiReady = window.ControlNetPreprocessor.init().catch((error) => {
            console.warn("ControlNet init failed:", error);
        });
    }
    // Optional LoRA panel integration (only active if that script is present on the page).
    loraPanelReady = window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" }) ?? Promise.resolve();
    loraPanelReady.then(() => {
        updateAdapterSummary();
        window.setTimeout(updateAdapterSummary, 500);
        window.setTimeout(updateAdapterSummary, 1500);
    });
    window.PresetPanel?.init({
        apiBase: API_BASE,
        family: "sd15",
        taskType: "sd15.inpaint",
        collectSettings: collectSd15InpaintPresetSettings,
        applySettings: applySd15InpaintPresetSettings,
    });
 
    // When a new base image is selected, reset mask state and render it on the canvases.
    initialImageInput.addEventListener("change", () => {
        const file = initialImageInput.files[0];
        if (!file) {
            return;
        }

        baseImageFile = file;
        // Reset any previously created masks/blurred masks for the new base image.
        maskBlob = null;
        maskDataUrl = null;
        blurMaskBlob = null;
        blurMaskDataUrl = null;
        blurToggle.checked = false;
        maskPreview.removeAttribute("src");
        maskPreviewPanel.classList.add("hidden");
        updateBlurControls();

        // Decode the local image file for immediate on-canvas editing (no upload yet).
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                baseImage = img;
                resizeCanvasDisplay(img);
                // Open the editor by default to encourage the "upload -> mask -> generate" flow.
                openMaskEditor();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    maskCanvas.addEventListener("pointerdown", (event) => {
        if (!baseImageFile) {
            return;
        }
        isDrawing = true;
        // Capture the pointer so we keep drawing even if the pointer leaves the element.
        maskCanvas.setPointerCapture(event.pointerId);
        drawAt(getCanvasPosition(event));
    });

    maskCanvas.addEventListener("pointermove", (event) => {
        if (!isDrawing) {
            return;
        }
        drawAt(getCanvasPosition(event));
    });

    maskCanvas.addEventListener("pointerup", () => {
        // End the stroke.
        isDrawing = false;
    });

    maskCanvas.addEventListener("pointerleave", () => {
        // Defensive: stop drawing if the pointer leaves unexpectedly.
        isDrawing = false;
    });

    // Expose actions for HTML `onclick` bindings (keeps markup simple).
    window.clearMask = clearMask;
    window.openMaskEditor = openMaskEditor;
    window.closeMaskEditor = closeMaskEditor;
    window.saveMask = saveMask;
    window.toggleMaskPreview = toggleMaskPreview;
    window.generateInpaint = generateInpaint;
    window.generateBlurMask = generateBlurMask;

    blurToggle.addEventListener("change", updateMaskPreview);

    // Ensure controls are in a consistent state on first render.
    updateBlurControls();
}

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

runWhenDomReady(initSd15InpaintingPage);

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

function baseInpaintInputs(inputs, defaults) {
    const prompt = WorkflowClient.readTextValue("prompt", "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        defaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const cfg = WorkflowClient.readNumberValue("cfg", defaults.cfg ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : defaults.model ?? null;
    const strength = WorkflowClient.readNumberValue("strength", defaults.strength ?? 0.5);
    const padding_mask_crop = WorkflowClient.readNumberValue(
        "padding_mask_crop",
        defaults.padding_mask_crop ?? 32,
        { integer: true }
    );
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, {
        integer: true,
    });

    Object.assign(inputs, {
        prompt,
        negative_prompt,
        steps,
        cfg,
        scheduler,
        seed,
        num_images,
        model,
        strength,
        padding_mask_crop,
        clip_skip,
    });
    return inputs;
}

function applyLcmInpaintContract(inputs) {
    inputs.lcm = { enabled: true };
    inputs.scheduler = DEFAULTS.lcm_scheduler;
    if (!Number.isFinite(inputs.steps)) {
        inputs.steps = DEFAULTS.lcm_steps;
    }
    if (!Number.isFinite(inputs.cfg)) {
        inputs.cfg = DEFAULTS.lcm_cfg;
    }
    if (inputs.steps < 1 || inputs.steps > 8) {
        throw new Error("LCM mode requires steps between 1 and 8.");
    }
    if (inputs.cfg < 0 || inputs.cfg > 2) {
        throw new Error("LCM mode requires CFG between 0 and 2.");
    }
}

function setLoraContract(inputs, loraAdapters) {
    const adapters = Array.isArray(loraAdapters) ? loraAdapters : [];
    const enabled = adapters.length > 0;
    inputs.lora = {
        lora_enabled: enabled,
        lora_adapters: enabled ? adapters : [],
    };
}

async function setControlNetInputs(inputs, defaults, controlnetState) {
    const controlnet_conditioning_scale = WorkflowClient.readNumberValue(
        "controlnet_conditioning_scale",
        defaults.controlnet_conditioning_scale ?? 1.0
    );
    const control_guidance_start = WorkflowClient.readNumberValue(
        "control_guidance_start",
        defaults.control_guidance_start ?? 0.0
    );
    const control_guidance_end = WorkflowClient.readNumberValue(
        "control_guidance_end",
        defaults.control_guidance_end ?? 1.0
    );
    const controlnet_guess_mode = Boolean(document.getElementById("controlnet_guess_mode")?.checked);
    const controlnet_compat_mode = WorkflowClient.readTextValue(
        "controlnet_compat_mode",
        defaults.controlnet_compat_mode ?? "warn"
    );

    const controlItems = Array.isArray(controlnetState?.controlItems) ? controlnetState.controlItems : [];
    if (controlItems.length === 0 && !controlnetState?.previewBlob) {
        throw new Error("ControlNet enabled but no preprocessor output image is ready.");
    }

    const effectiveItemsRaw =
        controlItems.length > 0
            ? controlItems
            : [
                  {
                      previewBlob: controlnetState.previewBlob,
                      preprocessorId: controlnetState.preprocessorId ?? null,
                      modelId: "lllyasviel/control_v11p_sd15_canny",
                      conditioningScale: controlnet_conditioning_scale,
                  },
              ];

    const uploadedArtifacts = await Promise.all(
        effectiveItemsRaw.map((item, idx) =>
            WorkflowClient.uploadArtifact(API_BASE, item.previewBlob, `controlnet_${idx + 1}.png`)
        )
    );
    const controlImages = uploadedArtifacts.map((uploaded) => `@artifact:${uploaded.artifact_id}`);
    const controlnetModels = effectiveItemsRaw.map(
        (item) => item.modelId || "lllyasviel/control_v11p_sd15_canny"
    );
    const controlnetScales = effectiveItemsRaw.map((item) => {
        const parsed = Number(item.conditioningScale);
        return Number.isFinite(parsed) ? parsed : controlnet_conditioning_scale;
    });
    const controlnetPreprocessorIds = effectiveItemsRaw.map((item) => item.preprocessorId || null);
    const hasAllPreprocessorIds = controlnetPreprocessorIds.every(
        (value) => typeof value === "string" && value.length > 0
    );

    const controlnetPreprocessors = controlImages.map((controlImage, idx) => ({
        control_image: controlImage,
        model_id: controlnetModels[idx],
        conditioning_scale: controlnetScales[idx],
        preprocessor_id: controlnetPreprocessorIds[idx],
    }));

    inputs.Controlnet = {
        enabled: true,
        controlnetConditioningScale: controlnet_conditioning_scale,
        controlGuidanceStart: control_guidance_start,
        controlGuidanceEnd: control_guidance_end,
        controlnetGuessMode: controlnet_guess_mode,
        controlnetPreprocessors,
    };

    inputs.control_image = controlImages[0];
    inputs.controlnet_model = controlnetModels[0];
    inputs.controlnet_conditioning_scale = controlnetScales[0];
    inputs.controlnet_guess_mode = controlnet_guess_mode;
    inputs.control_guidance_start = control_guidance_start;
    inputs.control_guidance_end = control_guidance_end;
    inputs.controlnet_compat_mode = controlnet_compat_mode;
    if (hasAllPreprocessorIds) {
        inputs.controlnet_preprocessor_id = controlnetPreprocessorIds[0];
    }

    if (effectiveItemsRaw.length > 1) {
        inputs.control_images = controlImages.slice(1);
        inputs.controlnet_models = controlnetModels;
        inputs.controlnet_conditioning_scales = controlnetScales;
        if (hasAllPreprocessorIds) {
            inputs.controlnet_preprocessor_ids = controlnetPreprocessorIds;
        }
    }

    return inputs;
}

/**
 * Resize both canvases to match the image's pixel dimensions and update their
 * CSS sizes to fit the viewport (plus user zoom).
 *
 * The base canvas is re-rendered from the image, and the mask is reset.
 *
 * @param {HTMLImageElement} image
 */
function resizeCanvasDisplay(image) {
    baseCanvas.width = image.width;
    baseCanvas.height = image.height;
    maskCanvas.width = image.width;
    maskCanvas.height = image.height;

    // Compute a "fit-to-view" scale based on container width and a max viewport height,
    // then apply the user-provided zoom multiplier.
    const availableWidth =
        canvasStack.parentElement?.clientWidth || canvasStack.clientWidth || image.width;
    const maxHeight = Math.round(window.innerHeight * 0.7);
    const maxWidth = Math.round(availableWidth);
    const fitScale = Math.min(1, maxWidth / image.width, maxHeight / image.height);
    const zoomScale = Number(zoomInput.value) / 100;
    displayScale = fitScale * zoomScale;
    const displayWidth = Math.round(image.width * displayScale);
    const displayHeight = Math.round(image.height * displayScale);
    const containerWidth = Math.min(maxWidth, displayWidth);
    const containerHeight = Math.min(maxHeight, displayHeight);

    canvasStack.style.width = `${containerWidth}px`;
    canvasStack.style.height = `${containerHeight}px`;
    canvasStack.style.maxWidth = "100%";
    canvasScroll.style.width = `${containerWidth}px`;
    canvasScroll.style.height = `${containerHeight}px`;
    canvasScroll.style.transform = "none";    
    baseCanvas.style.width = `${displayWidth}px`;
    baseCanvas.style.height = `${displayHeight}px`;
    maskCanvas.style.width = `${displayWidth}px`;
    maskCanvas.style.height = `${displayHeight}px`;

    // Redraw base and reset mask to a full black image (meaning "keep original").
    baseContext.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    baseContext.drawImage(image, 0, 0);
    clearMask();
    imageInfo.textContent = `Image size: ${image.width} × ${image.height} (${Math.round(displayScale * 100)}% view)`;
}

/**
 * Convert a pointer event's viewport coordinates into mask-canvas pixel coordinates.
 *
 * This accounts for the canvas being displayed at a CSS-scaled size while its
 * drawing buffer remains at the original image resolution.
 *
 * @param {PointerEvent} event
 * @returns {{x: number, y: number}}
 */
function getCanvasPosition(event) {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

/**
 * Draw a circular dab at a given position on the mask canvas.
 *
 * Mask convention:
 * - white  = inpaint this area
 * - black  = keep original
 *
 * @param {{x: number, y: number}} position
 */
function drawAt(position) {
    const brushSize = Number(brushSizeInput.value);
    const color = eraseToggle.checked ? "#000000" : "#ffffff";
    maskContext.fillStyle = color;
    maskContext.beginPath();
    maskContext.arc(position.x, position.y, brushSize / 2, 0, Math.PI * 2);
    maskContext.fill();
}

/**
 * Reset the mask canvas to solid black and clear any derived mask state.
 */
function clearMask() {
    if (!maskContext) {
        return;
    }
    maskContext.fillStyle = "#000000";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskBlob = null;
    maskDataUrl = null;
    blurMaskBlob = null;
    blurMaskDataUrl = null;
    blurToggle.checked = false;
    maskPreview.removeAttribute("src");
    updateBlurControls();
}

/**
 * Show the mask editor modal/panel.
 */
function openMaskEditor() {
    if (!baseImageFile) {
        alert("Please upload an initial image first.");
        return;
    }
    maskModal.classList.remove("hidden");
}

/**
 * Hide the mask editor modal/panel.
 */
function closeMaskEditor() {
    maskModal.classList.add("hidden");
}

/**
 * Toggle visibility of the mask preview panel.
 */
function toggleMaskPreview() {
    if (maskPreviewPanel.classList.contains("hidden")) {
        maskPreviewPanel.classList.remove("hidden");
    } else {
        maskPreviewPanel.classList.add("hidden");
    }
}

/**
 * Persist the current mask into both a Blob (for upload) and a Data URL (for preview).
 * Also clears any previous blurred-mask result, since it's no longer valid.
 */
async function saveMask() {
    maskBlob = await getMaskBlob();
    if (!maskBlob) {
        alert("Failed to create mask image.");
        return;
    }
    maskDataUrl = maskCanvas.toDataURL("image/png");
    blurMaskBlob = null;
    blurMaskDataUrl = null;
    blurToggle.checked = false;
    maskPreviewPanel.classList.remove("hidden");
    updateMaskPreview();
    updateBlurControls();
    closeMaskEditor();
}

/**
 * Create a PNG Blob of the current mask canvas.
 *
 * @returns {Promise<Blob|null>}
 */
function getMaskBlob() {
    return new Promise((resolve) => {
        maskCanvas.toBlob((blob) => {
            resolve(blob);
        }, "image/png");
    });
}

/**
 * Update the preview image to show either the blurred mask (if enabled) or the
 * raw saved mask.
 */
function updateMaskPreview() {
    if (blurToggle.checked && blurMaskDataUrl) {
        maskPreview.src = blurMaskDataUrl;
        return;
    }
    if (maskDataUrl) {
        maskPreview.src = maskDataUrl;
        return;
    }
    maskPreview.removeAttribute("src");
}

/**
 * Enable/disable blur controls based on whether a saved mask exists and whether
 * a blurred mask has been generated.
 */
function updateBlurControls() {
    const hasMask = Boolean(maskBlob);
    maskBlurButton.disabled = !hasMask;
    if (!hasMask) {
        blurToggle.checked = false;
        blurToggle.disabled = true;
        blurMaskBlob = null;
        blurMaskDataUrl = null;
    } else {
        blurToggle.disabled = !blurMaskDataUrl;
    }
    updateMaskPreview();
}

/**
 * Request a blurred version of the saved mask from the backend.
 *
 * The backend applies a Gaussian blur to soften edges, which can help avoid
 * harsh seams in the final inpaint result.
 */
async function generateBlurMask() {
    if (!maskBlob) {
        alert("Please create and save a mask before blurring.");
        return;
    }
    const blurFactor = Number(blurFactorInput.value);
    if (!Number.isFinite(blurFactor) || blurFactor < 0 || blurFactor > 128) {
        alert("Blur strength must be a number between 0 and 128.");
        return;
    }

    // Prevent repeated clicks while the request is in-flight.
    maskBlurButton.disabled = true;
    maskBlurButton.textContent = "Blurring...";
    try {
        const formData = new FormData();
        formData.append("mask_image", maskBlob, "mask.png");
        formData.append("blur_factor", blurFactor.toString());
        const res = await fetch(`${API_BASE}/create-blur-mask`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            throw new Error("Failed to blur mask.");
        }

        const blob = await res.blob();
        blurMaskBlob = blob;
        // Convert to a Data URL for preview/download without uploading it anywhere.
        blurMaskDataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
        blurToggle.checked = true;
    } catch (error) {
        console.error(error);
        alert("Unable to blur mask. Please try again.");
    } finally {
        maskBlurButton.textContent = "Blur mask edges";
        updateBlurControls();
    }
}

/**
 * Upload base+mask artifacts, submit an inpaint job, and stream results into the gallery.
 *
 * Uses the blurred mask if the toggle is enabled and a blurred mask is available;
 * otherwise uses the raw saved mask.
 */
async function generateInpaint() {
    const token = ++activeJobToken;
    closeActiveEventSource();
    const controlnetState = getControlNetState();
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);
    const lcmEnabled = isLcmModeEnabled();
    const ipAdapterEnabled = Boolean(document.getElementById("ip_adapter_enabled")?.checked);
    const ipAdapterImageFile = getIpAdapterImageFile();
    const ipAdapterMaskFile = window.IpAdapterPanel?.getMaskFile?.() ?? null;

    if (!baseImageFile) {
        alert("Please upload an initial image.");
        return;
    }
    const activeMaskBlob = blurToggle.checked && blurMaskBlob ? blurMaskBlob : maskBlob;
    if (!activeMaskBlob) {
        alert("Please create and save a mask before generating.");
        return;
    }

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["sd15.inpaint"]?.input_defaults ?? {};

    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    try {
        const taskInputs = {};
        baseInpaintInputs(taskInputs, defaults);
        const lcmRequested = lcmEnabled || taskInputs.scheduler === DEFAULTS.lcm_scheduler;
        if (lcmRequested && controlnetEnabled) {
            throw new Error("LCM mode cannot be combined with ControlNet for SD1.5 inpaint yet.");
        }
        if (ipAdapterEnabled && controlnetEnabled) {
            throw new Error("IP-Adapter cannot be combined with ControlNet for SD1.5 inpaint yet.");
        }
        if (ipAdapterEnabled && lcmRequested) {
            throw new Error("IP-Adapter cannot be combined with LCM mode for SD1.5 inpaint yet.");
        }
        if (ipAdapterEnabled && !ipAdapterImageFile) {
            throw new Error("IP-Adapter enabled but no reference image is selected.");
        }
        if (lcmRequested) {
            applyLcmInpaintContract(taskInputs);
        }

        // Upload base and mask concurrently to reduce overall latency.
        const [uploadedBase, uploadedMask] = await Promise.all([
            WorkflowClient.uploadArtifact(API_BASE, baseImageFile, baseImageFile.name || "initial.png"),
            WorkflowClient.uploadArtifact(API_BASE, activeMaskBlob, "mask.png"),
        ]);
        taskInputs.initial_image = `@artifact:${uploadedBase.artifact_id}`;
        taskInputs.mask_image = `@artifact:${uploadedMask.artifact_id}`;

        const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
        setLoraContract(taskInputs, loraAdapters);

        const tasks = [];
        if (ipAdapterEnabled) {
            const uploadedIpAdapterImage = await WorkflowClient.uploadArtifact(
                API_BASE,
                ipAdapterImageFile,
                ipAdapterImageFile.name || "ip_adapter.png"
            );
            if (!uploadedIpAdapterImage?.artifact_id) {
                throw new Error("IP-Adapter image upload did not return an artifact id.");
            }
            const ipAdapterScale = WorkflowClient.readNumberValue(
                "ip_adapter_scale",
                defaults.ip_adapter?.scale ?? DEFAULTS.ip_adapter_scale
            );
            tasks.push({
                id: "ip_embeds",
                type: "sd15.ip_adapter.encode",
                inputs: {
                    image: `@artifact:${uploadedIpAdapterImage.artifact_id}`,
                    model: taskInputs.model,
                    guidance_scale: taskInputs.cfg,
                    ip_adapter_model: "h94/IP-Adapter",
                    ip_adapter_subfolder: "models",
                    ip_adapter_weight_name: "ip-adapter_sd15.bin",
                    ip_adapter_scale: ipAdapterScale,
                },
            });
            taskInputs.ip_adapter = {
                enabled: true,
                image_embeds: "@ip_embeds.image_embeds",
                scale: ipAdapterScale,
                model: "h94/IP-Adapter",
                subfolder: "models",
                weight_name: "ip-adapter_sd15.bin",
            };
            if (ipAdapterMaskFile) {
                const uploadedIpAdapterMask = await WorkflowClient.uploadArtifact(
                    API_BASE,
                    ipAdapterMaskFile,
                    ipAdapterMaskFile.name || "ip_adapter_mask.png"
                );
                if (!uploadedIpAdapterMask?.artifact_id) {
                    throw new Error("IP-Adapter mask upload did not return an artifact id.");
                }
                taskInputs.ip_adapter.mask_image = `@artifact:${uploadedIpAdapterMask.artifact_id}`;
            }
        }

        if (controlnetEnabled) {
            await setControlNetInputs(taskInputs, defaults, controlnetState);
        }

        tasks.push({
            id: "inpaint",
            type: "sd15.inpaint",
            inputs: taskInputs,
        });

        for (const task of tasks) {
            await validateTaskInputsOrThrow(task.type, task.inputs);
        }

        const workflowPayload = {
            tasks,
            return: "@inpaint.images",
        };

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
                    const warnings = job?.result?.tasks?.inpaint?.warnings;
                    if (Array.isArray(warnings) && warnings.length > 0) {
                        console.warn("ControlNet warnings:", warnings);
                        const statusNode = document.getElementById("controlnet-status");
                        if (statusNode) {
                            statusNode.textContent = warnings.join(" ");
                        }
                    }
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
        if (
            error instanceof Error &&
            error.message.startsWith("Input validation failed for ")
        ) {
            alert(error.message);
        }
        console.warn("Failed to run inpaint job:", error);
        gallery.setImages([]);
    }
}

