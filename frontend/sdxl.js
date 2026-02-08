const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;

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
    void window.ControlNetPreprocessor.init().catch((error) => {
        console.warn("ControlNet init failed:", error);
    });
}

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
}

function resolveSdxlControlNetModel(modelId) {
    const normalized = String(modelId || "").trim();
    if (!normalized || normalized.includes("_sd15")) {
        return "diffusers/controlnet-canny-sdxl-1.0";
    }
    return normalized;
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
    const controlnetState = getControlNetState();
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);
    const primaryTaskType = controlnetEnabled ? "sdxl.controlnet.text2img" : "sdxl.text2img";

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.[primaryTaskType]?.input_defaults ?? {};

    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue("negative_prompt", defaults.negative_prompt ?? "");
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 1024, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 1024, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, { integer: true });
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, { integer: true });
    const hires_enabled = Boolean(document.getElementById("hires_enabled")?.checked);
    const hires_scale = WorkflowClient.readNumberValue("hires_scale", defaults.hires_scale ?? 1.0);
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

    try {
        setJobUiState(true, "Submitting job...");
        let workflowPayload;
        if (controlnetEnabled) {
            const controlItems = Array.isArray(controlnetState?.controlItems)
                ? controlnetState.controlItems
                : [];
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
            const controlImages = uploadedArtifacts.map(
                (uploaded) => `@artifact:${uploaded.artifact_id}`
            );
            const controlnetModels = effectiveItems.map(
                (item) => resolveSdxlControlNetModel(item.modelId)
            );
            const controlnetScales = effectiveItems.map((item) => {
                const parsed = Number(item.conditioningScale);
                return Number.isFinite(parsed) ? parsed : controlnet_conditioning_scale;
            });
            const controlnetPreprocessorIds = effectiveItems.map(
                (item) => item.preprocessorId || null
            );
            const hasAllPreprocessorIds = controlnetPreprocessorIds.every(
                (value) => typeof value === "string" && value.length > 0
            );
            const inputs = {
                control_image: controlImages[0],
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
                controlnet_conditioning_scale,
                controlnet_guess_mode,
                control_guidance_start,
                control_guidance_end,
                controlnet_compat_mode,
            };
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
            workflowPayload = {
                tasks: [{ id: "t1", type: "sdxl.controlnet.text2img", inputs }],
                return: "@t1.images",
            };
        } else {
            const payload = {
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
                hires_enabled,
                hires_scale,
            };
            workflowPayload = {
                tasks: [{ id: "t1", type: "sdxl.text2img", inputs: payload }],
                return: "@t1.images",
            };
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
