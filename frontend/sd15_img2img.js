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
                });
            })
            .catch(() => {});
    }
    // Optional LoRA panel integration (only active if that script is present on the page).
    window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" });
}

initSd15Img2ImgPage();

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

    const initialImageInput = document.getElementById("initial_image");
    const initialFile = initialImageInput.files[0];

    if (!initialFile) {
        alert("Please select an initial image.");
        return;
    }

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["sd15.img2img"]?.input_defaults ?? {};

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
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, {
        integer: true,
    });
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];

    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    try {
        // img2img tasks reference a stored artifact for the initial image input.
        const uploaded = await WorkflowClient.uploadArtifact(
            API_BASE,
            initialFile,
            initialFile.name || "initial.png"
        );

        const taskInputs = {
            initial_image: `@artifact:${uploaded.artifact_id}`,
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
        };
        if (loraAdapters.length > 0) {
            taskInputs.lora_adapters = loraAdapters;
        }

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
        console.warn("Failed to run img2img job:", error);
        gallery.setImages([]);
    }
}
