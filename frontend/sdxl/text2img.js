const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

const DEFAULTS = {
    ip_adapter_scale: 0.6,
};

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;
let controlNetUiReady = Promise.resolve();
let loraPanelReady = Promise.resolve();
let ipAdapterPanelReady = Promise.resolve();

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=sdxl`);
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
        fallback.value = "stable-diffusion-xl-base-1.0";
        fallback.textContent = "stable-diffusion-xl-base-1.0 (sdxl, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();
if (window.WorkflowCatalog?.load) {
    void window.WorkflowCatalog
        .load(API_BASE)
        .then(() => {
            window.WorkflowCatalog.applyDefaultsToForm("sdxl.text2img", {
                steps: "steps",
                guidance_scale: "cfg",
                width: "width",
                height: "height",
                num_images: "num_images",
                clip_skip: "clip_skip",
            });
            window.WorkflowCatalog.applyDefaultsToForm("sdxl.controlnet.text2img", {
                steps: "steps",
                guidance_scale: "cfg",
                width: "width",
                height: "height",
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
ipAdapterPanelReady =
    (window.SdxlIpAdapterPanel?.load?.() ?? Promise.resolve())
        .then(() => {
            window.IpAdapterPanel?.init();
        })
        .catch((error) => {
            console.warn("SDXL IP-Adapter UI init failed:", error);
        });
loraPanelReady = window.LoraPanel?.init({ apiBase: API_BASE, family: "sdxl" }) ?? Promise.resolve();
window.PresetPanel?.init({
    apiBase: API_BASE,
    family: "sdxl",
    taskType: "sdxl.text2img",
    collectSettings: collectSdxlPresetSettings,
    applySettings: applySdxlPresetSettings,
});

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
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

function collectSdxlPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 20, { integer: true }),
        guidance_scale: WorkflowClient.readNumberValue("cfg", 7.5),
        scheduler: WorkflowClient.readTextValue("scheduler", "euler"),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", 1024, { integer: true }),
        height: WorkflowClient.readNumberValue("height", 1024, { integer: true }),
        hires_enabled: Boolean(document.getElementById("hires_enabled")?.checked),
        hires_scale: WorkflowClient.readNumberValue("hires_scale", 1.0),
        model: WorkflowClient.readTextValue("model_select", null),
        clip_skip: WorkflowClient.readNumberValue("clip_skip", 1, { integer: true }),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
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
        ip_adapter_enabled: Boolean(document.getElementById("ip_adapter_enabled")?.checked),
        ip_adapter_scale: WorkflowClient.readNumberValue(
            "ip_adapter_scale",
            DEFAULTS.ip_adapter_scale
        ),
    };
}

async function applySdxlPresetSettings(settings) {
    await Promise.all([controlNetUiReady, loraPanelReady, ipAdapterPanelReady]);

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.guidance_scale);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setCheckboxValue("hires_enabled", settings.hires_enabled);
    setInputValue("hires_scale", settings.hires_scale);
    setModelSelection(settings.model);
    setInputValue("clip_skip", settings.clip_skip);
    setInputValue("num_images", settings.num_images);
    setCheckboxValue("controlnet-enabled", settings.controlnet_enabled);
    setInputValue("controlnet_conditioning_scale", settings.controlnet_conditioning_scale);
    setInputValue("control_guidance_start", settings.control_guidance_start);
    setInputValue("control_guidance_end", settings.control_guidance_end);
    setCheckboxValue("controlnet_guess_mode", settings.controlnet_guess_mode);
    setInputValue("controlnet_compat_mode", settings.controlnet_compat_mode);
    setCheckboxValue("ip_adapter_enabled", settings.ip_adapter_enabled);
    setInputValue("ip_adapter_scale", settings.ip_adapter_scale);

    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
    }

    window.ControlNetPanel?.clearControlItems?.();
    window.ControlNetPanel?.updateIndicator?.();
    window.ControlNetPanel?.updateActiveFlag?.();
}

function resolveSdxlControlNetModel(modelId) {
    const normalized = String(modelId || "").trim();
    if (!normalized || normalized.includes("_sd15")) {
        return "diffusers/controlnet-canny-sdxl-1.0";
    }
    return normalized;
}

function baseInput(inputs, defaults) {
    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        defaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 1024, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 1024, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : defaults.model ?? null;
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, {
        integer: true,
    });

    Object.assign(inputs, {
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        scheduler,
        seed,
        width,
        height,
        model,
        num_images,
        clip_skip,
    });

    return inputs;
}

async function setSdxlControlNetInputs(inputs, defaults, controlnetState) {
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
    const effectiveItems =
        controlItems.length > 0
            ? controlItems
            : [
                  {
                      previewBlob: controlnetState.previewBlob,
                      preprocessorId: controlnetState.preprocessorId ?? null,
                      modelId: "diffusers/controlnet-canny-sdxl-1.0",
                      conditioningScale: controlnet_conditioning_scale,
                  },
              ];

    const uploadedArtifacts = await Promise.all(
        effectiveItems.map((item, idx) =>
            WorkflowClient.uploadArtifact(
                API_BASE,
                item.previewBlob,
                `controlnet_${idx + 1}.png`
            )
        )
    );
    const controlImages = uploadedArtifacts.map((uploaded) => `@artifact:${uploaded.artifact_id}`);
    const controlnetModels = effectiveItems.map((item) => resolveSdxlControlNetModel(item.modelId));
    const controlnetScales = effectiveItems.map((item) => {
        const parsed = Number(item.conditioningScale);
        return Number.isFinite(parsed) ? parsed : controlnet_conditioning_scale;
    });
    const controlnetPreprocessorIds = effectiveItems.map((item) => item.preprocessorId || null);
    const hasAllPreprocessorIds = controlnetPreprocessorIds.every(
        (value) => typeof value === "string" && value.length > 0
    );

    Object.assign(inputs, {
        control_image: controlImages[0],
        controlnet_conditioning_scale,
        controlnet_guess_mode,
        control_guidance_start,
        control_guidance_end,
        controlnet_compat_mode,
    });

    if (effectiveItems.length > 1) {
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

    inputs.Controlnet = {
        enabled: true,
        controlnetConditioningScale: controlnet_conditioning_scale,
        controlGuidanceStart: control_guidance_start,
        controlGuidanceEnd: control_guidance_end,
        controlnetGuessMode: controlnet_guess_mode,
        controlnetPreprocessors: effectiveItems.map((item, idx) => ({
            control_image: controlImages[idx],
            model_id: controlnetModels[idx],
            conditioning_scale: controlnetScales[idx],
            preprocessor_id: controlnetPreprocessorIds[idx],
        })),
    };
}

function setJobUiState(isBusy, message) {
    const button = document.getElementById("generate_button");

    if (button) {
        button.disabled = Boolean(isBusy);
        button.textContent = isBusy ? "Generating..." : "Generate";
    }
}

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();
    await ipAdapterPanelReady;
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);
    const ipAdapterEnabled = Boolean(document.getElementById("ip_adapter_enabled")?.checked);
    const ipAdapterImageFile = getIpAdapterImageFile();
    const primaryTaskType = controlnetEnabled ? "sdxl.controlnet.text2img" : "sdxl.text2img";

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.[primaryTaskType]?.input_defaults ?? {};

    await Promise.all([controlNetUiReady, loraPanelReady]);
    const inputs = {};
    baseInput(inputs, defaults);

    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const loraAdaptersEnabled = Array.isArray(loraAdapters) && loraAdapters.length > 0;

    inputs.Lora = {
        enabled: loraAdaptersEnabled,
        adapters: loraAdaptersEnabled ? loraAdapters : [],
    };

    const hiresEnabledEl = document.getElementById("hires_enabled");
    const hiresScaleEl = document.getElementById("hires_scale");
    const hiresUiPresent = Boolean(hiresEnabledEl && hiresScaleEl);
    const hires_enabled = hiresUiPresent ? Boolean(hiresEnabledEl.checked) : false;
    const hires_scale = hiresUiPresent
        ? WorkflowClient.readNumberValue("hires_scale", defaults.hires_scale ?? 1.0)
        : 1.0;
    const hiresEnabled = hiresUiPresent && hires_enabled && hires_scale > 1.0;
    if (hiresEnabled) {
        inputs.hires = {
            enabled: true,
            hires_scale,
        };
    }

    const payload = inputs;
    if (loraAdapters.length > 0) {
        inputs.lora_adapters = loraAdapters;
        payload.lora_adapters = loraAdapters;
    }

    try {
        if (ipAdapterEnabled && controlnetEnabled) {
            throw new Error("SDXL IP-Adapter cannot be combined with ControlNet yet.");
        }
        if (ipAdapterEnabled && !ipAdapterImageFile) {
            throw new Error("IP-Adapter enabled but no reference image is selected.");
        }

        setJobUiState(true, "Submitting job...");
        let workflowPayload;
        if (controlnetEnabled) {
            const controlnetState = getControlNetState();
            await setSdxlControlNetInputs(inputs, defaults, controlnetState);
            workflowPayload = {
                tasks: [{ id: "t1", type: "sdxl.controlnet.text2img", inputs }],
                return: "@t1.images",
            };
        } else {
            if (hiresUiPresent) {
                payload.hires_enabled = hires_enabled;
                payload.hires_scale = hires_scale;
            }
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
                payload.ip_adapter = {
                    enabled: true,
                    image_embeds: "@ip_embeds.image_embeds",
                    scale: ipAdapterScale,
                    model: "h94/IP-Adapter",
                    subfolder: "sdxl_models",
                    weight_name: "ip-adapter_sdxl.bin",
                };
                workflowPayload = {
                    tasks: [
                        {
                            id: "ip_embeds",
                            type: "sdxl.ip_adapter.encode",
                            inputs: {
                                image: `@artifact:${uploadedIpAdapterImage.artifact_id}`,
                                model: payload.model,
                                guidance_scale: payload.guidance_scale,
                                ip_adapter_model: "h94/IP-Adapter",
                                ip_adapter_subfolder: "sdxl_models",
                                ip_adapter_weight_name: "ip-adapter_sdxl.bin",
                                ip_adapter_scale: ipAdapterScale,
                            },
                        },
                        { id: "t1", type: "sdxl.text2img", inputs: payload },
                    ],
                    return: "@t1.images",
                };
            } else {
                workflowPayload = {
                    tasks: [{ id: "t1", type: "sdxl.text2img", inputs: payload }],
                    return: "@t1.images",
                };
            }
        }
        const idempotencyKey = WorkflowClient.makeIdempotencyKey();
        const createdJob = await WorkflowClient.submitWorkflow(API_BASE, workflowPayload, idempotencyKey);
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        setJobUiState(true, `Queued (job ${jobId})`);

        activeEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeJobToken,
            onUpdate: (job) => {
                const status = job?.status ?? "unknown";
                if (status === "queued") {
                    setJobUiState(true, `Queued (job ${jobId})`);
                } else if (status === "running") {
                    setJobUiState(true, `Running (job ${jobId})`);
                } else {
                    setJobUiState(true, `Status: ${status} (job ${jobId})`);
                }
            },
            onDone: (job) => {
                const status = job?.status ?? "unknown";
                if (status === "succeeded") {
                    const images = job?.result?.outputs;
                    gallery.setImages(Array.isArray(images) ? images : []);
                    setJobUiState(false, `Done (job ${jobId})`);
                } else if (status === "failed") {
                    const err = job?.error ?? "Unknown error.";
                    setJobUiState(false, `Failed (job ${jobId})`);
                    gallery.setImages([]);
                    console.warn("Job failed:", err);
                } else if (status === "canceled") {
                    setJobUiState(false, `Canceled (job ${jobId})`);
                    gallery.setImages([]);
                } else {
                    setJobUiState(false, `Done (job ${jobId})`);
                }
            },
            onError: () => {
                if (token !== activeJobToken) {
                    return;
                }
                setJobUiState(false, "Job update stream lost.");
            },
        });
    } catch (error) {
        console.warn("Failed to generate SDXL images:", error);
        gallery.setImages([]);
        setJobUiState(false, "Failed to generate.");
    }
}
