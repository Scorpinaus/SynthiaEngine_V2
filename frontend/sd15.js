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

const { SD15_BASE_PAGE_CONFIG } = require("./SD15_BASE_PAGE_CONFIG");

function isPlainObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function deepFreeze(value) {
    if (!isPlainObject(value) && !Array.isArray(value)) {
        return value;
    }
    Object.freeze(value);
    for (const key of Object.keys(value)) {
        const next = value[key];
        if ((isPlainObject(next) || Array.isArray(next)) && !Object.isFrozen(next)) {
            deepFreeze(next);
        }
    }
    return value;
}

function createSd15PageConfig(overridesRaw) {
    const overrides = isPlainObject(overridesRaw) ? overridesRaw : {};
    const mappingsOverride = isPlainObject(overrides.mappings) ? overrides.mappings : {};
    const catalogDefaultsOverride = isPlainObject(mappingsOverride.catalogDefaults)
        ? mappingsOverride.catalogDefaults
        : {};

    return deepFreeze({
        ids: {
            ...SD15_BASE_PAGE_CONFIG.ids,
            ...(isPlainObject(overrides.ids) ? overrides.ids : {}),
        },
        tasks: {
            ...SD15_BASE_PAGE_CONFIG.tasks,
            ...(isPlainObject(overrides.tasks) ? overrides.tasks : {}),
        },
        fallbackDefaults: {
            ...SD15_BASE_PAGE_CONFIG.fallbackDefaults,
            ...(isPlainObject(overrides.fallbackDefaults) ? overrides.fallbackDefaults : {}),
        },
        mappings: {
            catalogDefaults: {
                text2img: {
                    ...SD15_BASE_PAGE_CONFIG.mappings.catalogDefaults.text2img,
                    ...(isPlainObject(catalogDefaultsOverride.text2img)
                        ? catalogDefaultsOverride.text2img
                        : {}),
                },
                controlnetText2img: {
                    ...SD15_BASE_PAGE_CONFIG.mappings.catalogDefaults.controlnetText2img,
                    ...(isPlainObject(catalogDefaultsOverride.controlnetText2img)
                        ? catalogDefaultsOverride.controlnetText2img
                        : {}),
                },
                hiresFix: {
                    ...SD15_BASE_PAGE_CONFIG.mappings.catalogDefaults.hiresFix,
                    ...(isPlainObject(catalogDefaultsOverride.hiresFix)
                        ? catalogDefaultsOverride.hiresFix
                        : {}),
                },
            },
            presetTextOrNumberFields: {
                ...SD15_BASE_PAGE_CONFIG.mappings.presetTextOrNumberFields,
                ...(isPlainObject(mappingsOverride.presetTextOrNumberFields)
                    ? mappingsOverride.presetTextOrNumberFields
                    : {}),
            },
            presetBooleanFields: {
                ...SD15_BASE_PAGE_CONFIG.mappings.presetBooleanFields,
                ...(isPlainObject(mappingsOverride.presetBooleanFields)
                    ? mappingsOverride.presetBooleanFields
                    : {}),
            },
        },
    });
}

const SD15_PAGE = createSd15PageConfig(window.SD15_PAGE_OVERRIDES);
const IDS = SD15_PAGE.ids;
const TASKS = SD15_PAGE.tasks;
const FALLBACKS = SD15_PAGE.fallbackDefaults;
const MAPPINGS = SD15_PAGE.mappings;

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
    const select = document.getElementById(IDS.modelSelect);
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
        fallback.value = FALLBACKS.modelSelectOption;
        fallback.textContent = `${FALLBACKS.modelSelectOption} (sd15, diffusers)`;
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

function setModelSelection(value) {
    if (value === undefined) {
        return;
    }
    const select = document.getElementById(IDS.modelSelect);
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
        prompt: WorkflowClient.readTextValue(IDS.prompt, FALLBACKS.prompt),
        negative_prompt: WorkflowClient.readTextValue(IDS.negativePrompt, FALLBACKS.negative_prompt),
        steps: WorkflowClient.readNumberValue(IDS.steps, FALLBACKS.steps, { integer: true }),
        cfg: WorkflowClient.readNumberValue(IDS.cfg, FALLBACKS.cfg),
        scheduler: WorkflowClient.readTextValue(IDS.scheduler, FALLBACKS.scheduler),
        seed: WorkflowClient.readSeedValue(IDS.seed),
        width: WorkflowClient.readNumberValue(IDS.width, FALLBACKS.width, { integer: true }),
        height: WorkflowClient.readNumberValue(IDS.height, FALLBACKS.height, { integer: true }),
        hires_enabled: Boolean(document.getElementById(IDS.hiresEnabled)?.checked),
        hires_scale: WorkflowClient.readNumberValue(IDS.hiresScale, FALLBACKS.hires_scale),
        model: WorkflowClient.readTextValue(IDS.modelSelect, FALLBACKS.model),
        clip_skip: WorkflowClient.readNumberValue(IDS.clipSkip, FALLBACKS.clip_skip, {
            integer: true,
        }),
        num_images: WorkflowClient.readNumberValue(IDS.numImages, FALLBACKS.num_images, {
            integer: true,
        }),
        weighting_policy: WorkflowClient.readTextValue(
            IDS.weightingPolicy,
            FALLBACKS.weighting_policy
        ),
        controlnet_enabled: Boolean(document.getElementById(IDS.controlnetEnabled)?.checked),
        controlnet_conditioning_scale: WorkflowClient.readNumberValue(
            IDS.controlnetConditioningScale,
            FALLBACKS.controlnet_conditioning_scale
        ),
        control_guidance_start: WorkflowClient.readNumberValue(
            IDS.controlGuidanceStart,
            FALLBACKS.control_guidance_start
        ),
        control_guidance_end: WorkflowClient.readNumberValue(
            IDS.controlGuidanceEnd,
            FALLBACKS.control_guidance_end
        ),
        controlnet_guess_mode: Boolean(document.getElementById(IDS.controlnetGuessMode)?.checked),
        controlnet_compat_mode: WorkflowClient.readTextValue(
            IDS.controlnetCompatMode,
            FALLBACKS.controlnet_compat_mode
        ),
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
    };
}

async function applySd15PresetSettings(settings) {
    await Promise.all([controlNetUiReady, loraPanelReady]);

    for (const [settingKey, elementId] of Object.entries(MAPPINGS.presetTextOrNumberFields)) {
        setInputValue(elementId, settings[settingKey]);
    }
    for (const [settingKey, elementId] of Object.entries(MAPPINGS.presetBooleanFields)) {
        setCheckboxValue(elementId, settings[settingKey]);
    }
    setModelSelection(settings.model);

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

    const generateButton = document.getElementById(IDS.generateButton);
    generateButton?.addEventListener("click", () => {
        generate();
    });

    void loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(TASKS.text2img, {
                    ...MAPPINGS.catalogDefaults.text2img,
                });
                window.WorkflowCatalog.applyDefaultsToForm(TASKS.controlnetText2img, {
                    ...MAPPINGS.catalogDefaults.controlnetText2img,
                });
                window.WorkflowCatalog.applyDefaultsToForm(TASKS.hiresFix, {
                    ...MAPPINGS.catalogDefaults.hiresFix,
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

function baseInput(primaryDefaults) {
    const prompt = WorkflowClient.readTextValue(IDS.prompt, FALLBACKS.prompt);
    const negative_prompt = WorkflowClient.readTextValue(
        IDS.negativePrompt,
        primaryDefaults.negative_prompt ?? FALLBACKS.negative_prompt
    );
    const steps = WorkflowClient.readNumberValue(IDS.steps, primaryDefaults.steps ?? FALLBACKS.steps, {
        integer: true,
    });
    const cfg = WorkflowClient.readNumberValue(IDS.cfg, primaryDefaults.cfg ?? FALLBACKS.cfg);
    const scheduler = WorkflowClient.readTextValue(
        IDS.scheduler,
        primaryDefaults.scheduler ?? FALLBACKS.scheduler
    );
    const seed = WorkflowClient.readSeedValue(IDS.seed);
    const width = WorkflowClient.readNumberValue(IDS.width, primaryDefaults.width ?? FALLBACKS.width, {
        integer: true,
    });
    const height = WorkflowClient.readNumberValue(
        IDS.height,
        primaryDefaults.height ?? FALLBACKS.height,
        {
        integer: true,
        }
    );
    const modelRaw = document.getElementById(IDS.modelSelect)?.value ?? "";
    const model = modelRaw ? modelRaw : primaryDefaults.model ?? FALLBACKS.model;
    const clip_skip = WorkflowClient.readNumberValue(
        IDS.clipSkip,
        primaryDefaults.clip_skip ?? FALLBACKS.clip_skip,
        { integer: true }
    );
    const num_images = WorkflowClient.readNumberValue(
        IDS.numImages,
        primaryDefaults.num_images ?? FALLBACKS.num_images,
        { integer: true }
    );
    const weighting_policy = WorkflowClient.readTextValue(
        IDS.weightingPolicy,
        primaryDefaults.weighting_policy ?? FALLBACKS.weighting_policy
    );

    return {
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
    };
}

async function setControlNetInputs(inputs, primaryDefaults, controlnetState) {
    const controlnet_conditioning_scale = WorkflowClient.readNumberValue(
        IDS.controlnetConditioningScale,
        primaryDefaults.controlnet_conditioning_scale ?? FALLBACKS.controlnet_conditioning_scale
    );
    const control_guidance_start = WorkflowClient.readNumberValue(
        IDS.controlGuidanceStart,
        primaryDefaults.control_guidance_start ?? FALLBACKS.control_guidance_start
    );
    const control_guidance_end = WorkflowClient.readNumberValue(
        IDS.controlGuidanceEnd,
        primaryDefaults.control_guidance_end ?? FALLBACKS.control_guidance_end
    );
    const controlnet_guess_mode = Boolean(document.getElementById(IDS.controlnetGuessMode)?.checked);
    const controlnet_compat_mode = WorkflowClient.readTextValue(
        IDS.controlnetCompatMode,
        primaryDefaults.controlnet_compat_mode ?? FALLBACKS.controlnet_compat_mode
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

    // Check if controlNet active, hiresFix active and if loraAdapters > 0 or not
    const controlnetEnabled = Boolean(document.getElementById(IDS.controlnetEnabled)?.checked);
    const hires_enabled = Boolean(document.getElementById(IDS.hiresEnabled)?.checked);
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const loraAdaptersEnabled = Array.isArray(loraAdapters) && loraAdapters.length > 0;

    // Retrieve catalog and set task type
    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const primaryTaskType = controlnetEnabled ? TASKS.controlnetText2img : TASKS.text2img;
    const primaryDefaults = catalog?.tasks?.[primaryTaskType]?.input_defaults ?? {};
    
    // Check if hires_fix enabled
    const hiresDefaults = catalog?.tasks?.[TASKS.hiresFix]?.input_defaults ?? {};
    const hires_scale = WorkflowClient.readNumberValue(
        IDS.hiresScale,
        hiresDefaults.hires_scale ?? FALLBACKS.hires_scale
    );
    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    // Build input list and push to FastAPI backend 
    try {
        const tasks = [];
        
        // Set base inputs and loraAdapters (if active)
        const primaryInputs = baseInput(primaryDefaults);
        // If loraAdapters enabled, add lora sub-object which sets loraStatus and adapters
        if (loraAdaptersEnabled) {
            primaryInputs.Lora = {
                loraStatus: true,
                adapters: loraAdapters,
            };
            primaryInputs.lora_adapters = loraAdapters;
        } else {
            primaryInputs.Lora = {
                loraStatus: false,
                adapters: [],
            };
        }

        // Check if controlnet is enabled
        // primaryInputs.controlNetEnabled = controlnetEnabled;
        if (controlnetEnabled) {
            const controlnetState = getControlNetState();
            await setControlNetInputs(primaryInputs, primaryDefaults, controlnetState);
            tasks.push({ id: "t1", type: TASKS.controlnetText2img, inputs: primaryInputs });
        } else {
            tasks.push({ id: "t1", type: TASKS.text2img, inputs: primaryInputs });
        }

        let returnRef = "@t1.images";
        if (hires_enabled && hires_scale > 1.0) {
            primaryInputs.hires = {
                hiresEnabled: true,
                hires_scale,
            };
            tasks.push({
                id: "hires",
                type: TASKS.hiresFix,
                inputs: {
                    images: "@t1.images",
                    prompt: primaryInputs.prompt,
                    negative_prompt: primaryInputs.negative_prompt,
                    steps: primaryInputs.steps,
                    cfg: primaryInputs.cfg,
                    scheduler: primaryInputs.scheduler,
                    seed: primaryInputs.seed,
                    model: primaryInputs.model,
                    clip_skip: primaryInputs.clip_skip,
                    hires_scale,
                    weighting_policy: primaryInputs.weighting_policy,
                    hires: {
                        hiresEnabled: true,
                        hires_scale,
                    },
                },
            });
            if (loraAdapters.length > 0) {
                tasks[tasks.length - 1].inputs.lora_adapters = loraAdapters;
                tasks[tasks.length - 1].inputs.Lora = {
                    loraStatus: true,
                    adapters: loraAdapters,
                };
            }
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
                        const statusNode = document.getElementById(IDS.controlnetStatus);
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
