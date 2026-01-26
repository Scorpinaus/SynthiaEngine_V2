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
                cfg: "guidance_scale",
                width: "width",
                height: "height",
                num_images: "num_images",
                clip_skip: "clip_skip",
            });
        })
        .catch(() => {});
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

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["sdxl.text2img"]?.input_defaults ?? {};

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
    console.log("Generate payload", payload);

    try {
        setJobUiState(true, "Submitting job...");
        const workflowPayload = {
            tasks: [{ id: "t1", type: "sdxl.text2img", inputs: payload }],
            return: "@t1.images",
        };
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
