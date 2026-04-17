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

const TASK_TEXT2IMG = "sd15.text2img";
const TASK_CONTROLNET_TEXT2IMG = "sd15.controlnet.text2img";
const TASK_HIRES_FIX = "sd15.hires_fix";

const DEFAULTS = {
    prompt: "",
    negative_prompt: "",
    steps: 20,
    lcm_steps: 4,
    cfg: 7.5,
    lcm_cfg: 0,
    scheduler: "euler",
    lcm_scheduler: "lcm",
    width: 512,
    height: 512,
    hires_scale: 1.0,
    clip_skip: 1,
    num_images: 1,
    weighting_policy: "diffusers-like",
    ip_adapter_scale: 0.6,
    controlnet_conditioning_scale: 1.0,
    control_guidance_start: 0.0,
    control_guidance_end: 1.0,
    controlnet_compat_mode: "warn",
    model: null,
    modelSelectOption: "stable-diffusion-v1-5",
};

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
    if (!select) {
        return;
    }
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
        fallback.value = DEFAULTS.modelSelectOption;
        fallback.textContent = `${DEFAULTS.modelSelectOption} (sd15, diffusers)`;
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
}

let controlNetUiReady = Promise.resolve();
let loraPanelReady = Promise.resolve();

function isLcmModeEnabled() {
    return Boolean(document.getElementById("lcm_enabled")?.checked);
}

function getIpAdapterImageFile() {
    return document.getElementById("ip_adapter_image")?.files?.[0] ?? null;
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

function collectSd15PresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", DEFAULTS.prompt),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", DEFAULTS.negative_prompt),
        steps: WorkflowClient.readNumberValue("steps", DEFAULTS.steps, { integer: true }),
        cfg: WorkflowClient.readNumberValue("cfg", DEFAULTS.cfg),
        scheduler: WorkflowClient.readTextValue("scheduler", DEFAULTS.scheduler),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", DEFAULTS.width, { integer: true }),
        height: WorkflowClient.readNumberValue("height", DEFAULTS.height, { integer: true }),
        hires_enabled: Boolean(document.getElementById("hires_enabled")?.checked),
        hires_scale: WorkflowClient.readNumberValue("hires_scale", DEFAULTS.hires_scale),
        model: WorkflowClient.readTextValue("model_select", DEFAULTS.model),
        clip_skip: WorkflowClient.readNumberValue("clip_skip", DEFAULTS.clip_skip, {
            integer: true,
        }),
        num_images: WorkflowClient.readNumberValue("num_images", DEFAULTS.num_images, {
            integer: true,
        }),
        weighting_policy: WorkflowClient.readTextValue(
            "weighting_policy",
            DEFAULTS.weighting_policy
        ),
        controlnet_enabled: Boolean(document.getElementById("controlnet-enabled")?.checked),
        controlnet_conditioning_scale: WorkflowClient.readNumberValue(
            "controlnet_conditioning_scale",
            DEFAULTS.controlnet_conditioning_scale
        ),
        control_guidance_start: WorkflowClient.readNumberValue(
            "control_guidance_start",
            DEFAULTS.control_guidance_start
        ),
        control_guidance_end: WorkflowClient.readNumberValue(
            "control_guidance_end",
            DEFAULTS.control_guidance_end
        ),
        controlnet_guess_mode: Boolean(document.getElementById("controlnet_guess_mode")?.checked),
        controlnet_compat_mode: WorkflowClient.readTextValue(
            "controlnet_compat_mode",
            DEFAULTS.controlnet_compat_mode
        ),
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
        ip_adapter_enabled: Boolean(document.getElementById("ip_adapter_enabled")?.checked),
        ip_adapter_scale: WorkflowClient.readNumberValue(
            "ip_adapter_scale",
            DEFAULTS.ip_adapter_scale
        ),
        lcm_enabled: isLcmModeEnabled(),
    };
}

async function applySd15PresetSettings(settings) {
    await Promise.all([controlNetUiReady, loraPanelReady]);

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.cfg);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setInputValue("hires_scale", settings.hires_scale);
    setInputValue("clip_skip", settings.clip_skip);
    setInputValue("num_images", settings.num_images);
    setInputValue("ip_adapter_scale", settings.ip_adapter_scale);
    setInputValue("weighting_policy", settings.weighting_policy);
    setInputValue("controlnet_conditioning_scale", settings.controlnet_conditioning_scale);
    setInputValue("control_guidance_start", settings.control_guidance_start);
    setInputValue("control_guidance_end", settings.control_guidance_end);
    setInputValue("controlnet_compat_mode", settings.controlnet_compat_mode);

    setCheckboxValue("hires_enabled", settings.hires_enabled);
    setCheckboxValue("controlnet-enabled", settings.controlnet_enabled);
    setCheckboxValue("controlnet_guess_mode", settings.controlnet_guess_mode);
    setCheckboxValue("ip_adapter_enabled", settings.ip_adapter_enabled);
    setCheckboxValue("lcm_enabled", settings.lcm_enabled);
    setModelSelection(settings.model);
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
    window.IpAdapterPanel?.init();

    const generateButton = document.getElementById("generate-button");
    generateButton?.addEventListener("click", () => {
        generate();
    });
    document.getElementById("lcm_enabled")?.addEventListener("change", syncLcmModeDefaults);

    void loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(TASK_TEXT2IMG, {
                    steps: "steps",
                    cfg: "cfg",
                    width: "width",
                    height: "height",
                    num_images: "num_images",
                    clip_skip: "clip_skip",
                    weighting_policy: "weighting_policy",
                });
                window.WorkflowCatalog.applyDefaultsToForm(TASK_CONTROLNET_TEXT2IMG, {
                    steps: "steps",
                    cfg: "cfg",
                    width: "width",
                    height: "height",
                    num_images: "num_images",
                    clip_skip: "clip_skip",
                    controlnet_conditioning_scale: "controlnet_conditioning_scale",
                    control_guidance_start: "control_guidance_start",
                    control_guidance_end: "control_guidance_end",
                    controlnet_compat_mode: "controlnet_compat_mode",
                });
                window.WorkflowCatalog.applyDefaultsToForm(TASK_HIRES_FIX, {
                    hires_scale: "hires_scale",
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
    window.PresetPanel?.init({
        apiBase: API_BASE,
        family: "sd15",
        taskType: "sd15.text2img",
        collectSettings: collectSd15PresetSettings,
        applySettings: applySd15PresetSettings,
    });
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

function baseInput(inputs, primaryDefaults) {
    const prompt = WorkflowClient.readTextValue("prompt", DEFAULTS.prompt);
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        primaryDefaults.negative_prompt ?? DEFAULTS.negative_prompt
    );
    const steps = WorkflowClient.readNumberValue("steps", primaryDefaults.steps ?? DEFAULTS.steps, {
        integer: true,
    });
    const cfg = WorkflowClient.readNumberValue("cfg", primaryDefaults.cfg ?? DEFAULTS.cfg);
    const scheduler = WorkflowClient.readTextValue(
        "scheduler",
        primaryDefaults.scheduler ?? DEFAULTS.scheduler
    );
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", primaryDefaults.width ?? DEFAULTS.width, {
        integer: true,
    });
    const height = WorkflowClient.readNumberValue(
        "height",
        primaryDefaults.height ?? DEFAULTS.height,
        {
        integer: true,
        }
    );
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : primaryDefaults.model ?? DEFAULTS.model;
    const clip_skip = WorkflowClient.readNumberValue(
        "clip_skip",
        primaryDefaults.clip_skip ?? DEFAULTS.clip_skip,
        { integer: true }
    );
    const num_images = WorkflowClient.readNumberValue(
        "num_images",
        primaryDefaults.num_images ?? DEFAULTS.num_images,
        { integer: true }
    );
    const weighting_policy = WorkflowClient.readTextValue(
        "weighting_policy",
        primaryDefaults.weighting_policy ?? DEFAULTS.weighting_policy
    );

    Object.assign(inputs, {
        prompt,
        negative_prompt,
        steps,
        cfg,
        scheduler,
        seed,
        width,
        height,
        model,
        clip_skip,
        num_images,
        weighting_policy,
    });
    return inputs;
}

function applyLcmText2ImgContract(inputs) {
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

async function setControlNetInputs(inputs, primaryDefaults, controlnetState) {
    const controlnet_conditioning_scale = WorkflowClient.readNumberValue(
        "controlnet_conditioning_scale",
        primaryDefaults.controlnet_conditioning_scale ?? DEFAULTS.controlnet_conditioning_scale
    );
    const control_guidance_start = WorkflowClient.readNumberValue(
        "control_guidance_start",
        primaryDefaults.control_guidance_start ?? DEFAULTS.control_guidance_start
    );
    const control_guidance_end = WorkflowClient.readNumberValue(
        "control_guidance_end",
        primaryDefaults.control_guidance_end ?? DEFAULTS.control_guidance_end
    );
    const controlnet_guess_mode = Boolean(document.getElementById("controlnet_guess_mode")?.checked);
    const controlnet_compat_mode = WorkflowClient.readTextValue(
        "controlnet_compat_mode",
        primaryDefaults.controlnet_compat_mode ?? DEFAULTS.controlnet_compat_mode
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

    inputs.controlNetEnabled = true;
    inputs.controlnet_conditioning_scale = controlnet_conditioning_scale;
    inputs.control_guidance_start = control_guidance_start;
    inputs.control_guidance_end = control_guidance_end;
    inputs.controlnet_guess_mode = controlnet_guess_mode;
    inputs.controlnet_compat_mode = controlnet_compat_mode;
    inputs.effectiveItems = controlImages.map((controlImage, idx) => ({
        control_image: controlImage,
        model_id: controlnetModels[idx],
        conditioning_scale: controlnetScales[idx],
        preprocessor_id: controlnetPreprocessorIds[idx],
    }));

    inputs.control_image = controlImages[0];
    if (effectiveItemsRaw.length > 1) {
        inputs.control_images = controlImages;
        inputs.controlnet_models = controlnetModels;
        inputs.controlnet_conditioning_scales = controlnetScales;
        if (hasAllPreprocessorIds) {
            inputs.controlnet_preprocessor_ids = controlnetPreprocessorIds;
        }
    } else {
        inputs.controlnet_model = controlnetModels[0];
        inputs.controlnet_conditioning_scale = controlnetScales[0];
        if (hasAllPreprocessorIds) {
            inputs.controlnet_preprocessor_id = controlnetPreprocessorIds[0];
        }
    }

    return inputs;
}

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

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

    // Check which optional features are enabled in the current UI state.
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);
    const hires_enabled = Boolean(document.getElementById("hires_enabled")?.checked);
    const lcmEnabled = isLcmModeEnabled();
    const ipAdapterEnabled = Boolean(document.getElementById("ip_adapter_enabled")?.checked);
    const ipAdapterImageFile = getIpAdapterImageFile();
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const loraAdaptersEnabled = Array.isArray(loraAdapters) && loraAdapters.length > 0;

    // Retrieve catalog and set task type
    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const primaryTaskType = controlnetEnabled ? TASK_CONTROLNET_TEXT2IMG : TASK_TEXT2IMG;
    const primaryDefaults = catalog?.tasks?.[primaryTaskType]?.input_defaults ?? {};
    
    // Check if hires_fix enabled
    const hiresDefaults = catalog?.tasks?.[TASK_HIRES_FIX]?.input_defaults ?? {};
    const hires_scale = WorkflowClient.readNumberValue(
        "hires_scale",
        hiresDefaults.hires_scale ?? DEFAULTS.hires_scale
    );
    const hiresEnabled = hires_enabled && hires_scale > 1.0;
    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    // Build input list and push to FastAPI backend.
    try {
        if (lcmEnabled && controlnetEnabled) {
            throw new Error("LCM mode is currently available for SD1.5 text-to-image only.");
        }
        if (lcmEnabled && hiresEnabled) {
            throw new Error("LCM mode cannot be combined with Hi-Res Fix yet.");
        }
        if (ipAdapterEnabled && controlnetEnabled) {
            throw new Error("IP-Adapter is currently available for SD1.5 text-to-image only.");
        }
        if (ipAdapterEnabled && lcmEnabled) {
            throw new Error("IP-Adapter cannot be combined with LCM mode yet.");
        }
        if (ipAdapterEnabled && hiresEnabled) {
            throw new Error("IP-Adapter cannot be combined with Hi-Res Fix yet.");
        }
        if (ipAdapterEnabled && !ipAdapterImageFile) {
            throw new Error("IP-Adapter enabled but no reference image is selected.");
        }

        const tasks = [];
        const inputs = {};
        baseInput(inputs, primaryDefaults);

        if (lcmEnabled) {
            applyLcmText2ImgContract(inputs);
        }
        inputs.lora = {
            lora_enabled: loraAdaptersEnabled,
            lora_adapters: loraAdaptersEnabled ? loraAdapters : [],
        };
        inputs.hires = {
            enabled: hiresEnabled,
            hires_scale,
        };
        if (ipAdapterEnabled) {
            const uploadedIpAdapterImage = await WorkflowClient.uploadArtifact(
                API_BASE,
                ipAdapterImageFile,
                ipAdapterImageFile.name || "ip_adapter.png"
            );
            if (!uploadedIpAdapterImage?.artifact_id) {
                throw new Error("IP-Adapter image upload did not return an artifact id.");
            }
            inputs.ip_adapter = {
                enabled: true,
                image: `@artifact:${uploadedIpAdapterImage.artifact_id}`,
                scale: WorkflowClient.readNumberValue(
                    "ip_adapter_scale",
                    primaryDefaults.ip_adapter?.scale ?? DEFAULTS.ip_adapter_scale
                ),
                model: "h94/IP-Adapter",
                subfolder: "models",
                weight_name: "ip-adapter_sd15.bin",
            };
        }

        // Check if ControlNet is enabled.
        if (controlnetEnabled) {
            const controlnetState = getControlNetState();
            await setControlNetInputs(inputs, primaryDefaults, controlnetState);
            inputs.Controlnet = {
                enabled: true,
                controlnetConditioningScale: inputs.controlnet_conditioning_scale,
                controlGuidanceStart: inputs.control_guidance_start,
                controlGuidanceEnd: inputs.control_guidance_end,
                controlnetGuessMode: inputs.controlnet_guess_mode,
                controlnetPreprocessors: Array.isArray(inputs.effectiveItems)
                    ? inputs.effectiveItems.map((item) => ({
                          control_image: item.control_image,
                          model_id: item.model_id,
                          conditioning_scale: item.conditioning_scale,
                          preprocessor_id: item.preprocessor_id,
                      }))
                    : [],
            };
            tasks.push({ id: "t1", type: TASK_CONTROLNET_TEXT2IMG, inputs });
        } else {
            tasks.push({ id: "t1", type: TASK_TEXT2IMG, inputs });
        }

        let returnRef = "@t1.images";
        if (hiresEnabled) {
            const hiresInputs = {
                images: "@t1.images",
                prompt: inputs.prompt,
                negative_prompt: inputs.negative_prompt,
                steps: inputs.steps,
                cfg: inputs.cfg,
                scheduler: inputs.scheduler,
                seed: inputs.seed,
                model: inputs.model,
                clip_skip: inputs.clip_skip,
                hires_scale,
                weighting_policy: inputs.weighting_policy,
                hires: {
                    enabled: true,
                    hires_scale,
                },
            };
            hiresInputs.lora = {
                lora_enabled: loraAdaptersEnabled,
                lora_adapters: loraAdaptersEnabled ? loraAdapters : [],
            };

            tasks.push({ id: "hires", type: TASK_HIRES_FIX, inputs: hiresInputs });
            returnRef = "@hires.images";
        }

        for (const task of tasks) {
            await validateTaskInputsOrThrow(task.type, task.inputs);
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
                    const warnings = job?.result?.tasks?.t1?.warnings;
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
        console.warn("Failed to generate SD1.5 images:", error);
        gallery.setImages([]);
    }
}
