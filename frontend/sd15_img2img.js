/**
 * SD1.5 img2img UI wiring.
 *
 * Responsibilities:
 * - Upload an initial image to the backend artifact store.
 * - Submit an `sd15.img2img` workflow job.
 * - Stream job status updates via SSE and populate the gallery on completion.
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

// Token incremented per generateImg2Img() call to ignore stale SSE events from prior jobs.
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

function collectSd15Img2ImgPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 20, { integer: true }),
        cfg: WorkflowClient.readNumberValue("cfg", 7.5),
        scheduler: WorkflowClient.readTextValue("scheduler", "euler"),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", 512, { integer: true }),
        height: WorkflowClient.readNumberValue("height", 512, { integer: true }),
        strength: WorkflowClient.readNumberValue("strength", 0.75),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
        model: WorkflowClient.readTextValue("model_select", null),
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
    };
}

async function applySd15Img2ImgPresetSettings(settings) {
    await Promise.all([controlNetUiReady, loraPanelReady]);

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.cfg);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setInputValue("strength", settings.strength);
    setInputValue("num_images", settings.num_images);
    setModelSelection(settings.model);
    setInputValue("clip_skip", settings.clip_skip);
    setCheckboxValue("controlnet-enabled", settings.controlnet_enabled);
    setInputValue("controlnet_conditioning_scale", settings.controlnet_conditioning_scale);
    setInputValue("control_guidance_start", settings.control_guidance_start);
    setInputValue("control_guidance_end", settings.control_guidance_end);
    setCheckboxValue("controlnet_guess_mode", settings.controlnet_guess_mode);
    setInputValue("controlnet_compat_mode", settings.controlnet_compat_mode);

    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
    }

    window.ControlNetPanel?.clearControlItems?.();
    window.ControlNetPanel?.updateIndicator?.();
    window.ControlNetPanel?.updateActiveFlag?.();
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

/**
 * Initialize page-level UI integrations.
 *
 * Centralizes one-time setup calls to keep the global scope tidy.
 */
function initSd15Img2ImgPage() {
    gallery.render();
    loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm("sd15.img2img", {
                    steps: "steps",
                    cfg: "cfg",
                    strength: "strength",
                    num_images: "num_images",
                    clip_skip: "clip_skip",
                    controlnet_conditioning_scale: "controlnet_conditioning_scale",
                    control_guidance_start: "control_guidance_start",
                    control_guidance_end: "control_guidance_end",
                    controlnet_compat_mode: "controlnet_compat_mode",
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
        taskType: "sd15.img2img",
        collectSettings: collectSd15Img2ImgPresetSettings,
        applySettings: applySd15Img2ImgPresetSettings,
    });
}

initSd15Img2ImgPage();

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

function baseImg2ImgInputs(inputs, defaults) {
    const prompt = WorkflowClient.readTextValue("prompt", "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        defaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const cfg = WorkflowClient.readNumberValue("cfg", defaults.cfg ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? null, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? null, { integer: true });
    const strength = WorkflowClient.readNumberValue("strength", defaults.strength ?? 0.75);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : defaults.model ?? null;
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
        width,
        height,
        strength,
        num_images,
        model,
        clip_skip,
    });
    return inputs;
}

function setLoraContract(inputs, loraAdapters) {
    const adapters = Array.isArray(loraAdapters) ? loraAdapters : [];
    const enabled = adapters.length > 0;
    inputs.Lora = {
        enabled,
        adapters: enabled ? adapters : [],
    };
    inputs.lora = {
        lora_enabled: enabled,
        lora_adapters: enabled ? adapters : [],
    };
    return enabled;
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
 * Collect inputs, upload the initial image, submit an img2img job, and stream results.
 *
 * High-level flow:
 * 1) Validate the initial image selection.
 * 2) Read form fields and normalize types.
 * 3) Upload the initial image as an artifact reference.
 * 4) Submit the workflow and stream SSE status updates into the gallery.
 */
async function generateImg2Img() {
    const token = ++activeJobToken;
    closeActiveEventSource();
    const controlnetState = getControlNetState();
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);

    const initialImageInput = document.getElementById("initial_image");
    const initialFile = initialImageInput.files[0];

    if (!initialFile) {
        alert("Please select an initial image.");
        return;
    }

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["sd15.img2img"]?.input_defaults ?? {};
    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    try {
        // img2img tasks reference a stored artifact for the initial image input.
        const uploaded = await WorkflowClient.uploadArtifact(
            API_BASE,
            initialFile,
            initialFile.name || "initial.png"
        );

        const taskInputs = {};
        baseImg2ImgInputs(taskInputs, defaults);
        taskInputs.initial_image = `@artifact:${uploaded.artifact_id}`;

        const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
        const loraAdaptersEnabled = setLoraContract(taskInputs, loraAdapters);
        taskInputs.lora_adapters = loraAdapters;
        if (!loraAdaptersEnabled) {
            taskInputs.lora_adapters = [];
        }

        if (controlnetEnabled) {
            await setControlNetInputs(taskInputs, defaults, controlnetState);
        }

        await validateTaskInputsOrThrow("sd15.img2img", taskInputs);

        const workflowPayload = {
            tasks: [
                {
                    id: "img2img",
                    type: "sd15.img2img",
                    inputs: taskInputs,
                },
            ],
            return: "@img2img.images",
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
                    const warnings = job?.result?.tasks?.img2img?.warnings;
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
        console.warn("Failed to run img2img job:", error);
        gallery.setImages([]);
    }
}
