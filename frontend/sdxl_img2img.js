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
            window.WorkflowCatalog.applyDefaultsToForm("sdxl.img2img", {
                steps: "steps",
                guidance_scale: "cfg",
                width: "width",
                height: "height",
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
    void window.ControlNetPreprocessor.init().catch((error) => {
        console.warn("ControlNet init failed:", error);
    });
}

async function generateSdxlImg2Img() {
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
    const defaults = catalog?.tasks?.["sdxl.img2img"]?.input_defaults ?? {};

    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue("negative_prompt", defaults.negative_prompt ?? "");
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 1024, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 1024, { integer: true });
    const strength = WorkflowClient.readNumberValue("strength", defaults.strength ?? 0.75);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, { integer: true });
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
        const uploaded = await WorkflowClient.uploadArtifact(
            API_BASE,
            initialFile,
            initialFile.name || "initial.png",
        );

        const taskInputs = {
            initial_image: `@artifact:${uploaded.artifact_id}`,
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
            scheduler,
            seed,
            width,
            height,
            strength,
            num_images,
            model,
            clip_skip,
        };
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
                (controlUploaded) => `@artifact:${controlUploaded.artifact_id}`
            );
            const controlnetModels = effectiveItems.map((item) =>
                resolveSdxlControlNetModel(item.modelId)
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

            taskInputs.control_image = controlImages[0];
            taskInputs.controlnet_model = controlnetModels[0];
            taskInputs.controlnet_conditioning_scale = controlnetScales[0];
            taskInputs.controlnet_guess_mode = controlnet_guess_mode;
            taskInputs.control_guidance_start = control_guidance_start;
            taskInputs.control_guidance_end = control_guidance_end;
            taskInputs.controlnet_compat_mode = controlnet_compat_mode;
            if (hasAllPreprocessorIds) {
                taskInputs.controlnet_preprocessor_id = controlnetPreprocessorIds[0];
            }
            if (effectiveItems.length > 1) {
                taskInputs.control_images = controlImages.slice(1);
                taskInputs.controlnet_models = controlnetModels;
                taskInputs.controlnet_conditioning_scales = controlnetScales;
                if (hasAllPreprocessorIds) {
                    taskInputs.controlnet_preprocessor_ids = controlnetPreprocessorIds;
                }
            }
        }

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: "sdxl.img2img",
                    inputs: taskInputs,
                },
            ],
            return: "@t1.images",
        };

        const idempotencyKey = WorkflowClient.makeIdempotencyKey();
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
        console.warn("Failed to run SDXL img2img job:", error);
        gallery.setImages([]);
    }
}
