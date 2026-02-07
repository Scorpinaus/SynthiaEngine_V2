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

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
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
                    controlnet_conditioning_scale: "controlnet_conditioning_scale",
                    control_guidance_start: "control_guidance_start",
                    control_guidance_end: "control_guidance_end",
                    controlnet_compat_mode: "controlnet_compat_mode",
                });
                window.WorkflowCatalog.applyDefaultsToForm("sd15.hires_fix", {
                    hires_scale: "hires_scale",
                });
            })
            .catch(() => {});
    }
    if (window.ControlNetPreprocessor?.init) {
        window.ControlNetPreprocessor.init().catch((error) => {
            console.warn("ControlNet init failed:", error);
        });
    }

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
    const controlnetState = getControlNetState();

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
    const controlnet_conditioning_scale = WorkflowClient.readNumberValue(
        "controlnet_conditioning_scale",
        primaryDefaults.controlnet_conditioning_scale ?? 1.0
    );
    const control_guidance_start = WorkflowClient.readNumberValue(
        "control_guidance_start",
        primaryDefaults.control_guidance_start ?? 0.0
    );
    const control_guidance_end = WorkflowClient.readNumberValue(
        "control_guidance_end",
        primaryDefaults.control_guidance_end ?? 1.0
    );
    const controlnet_guess_mode = Boolean(document.getElementById("controlnet_guess_mode")?.checked);
    const controlnet_compat_mode = WorkflowClient.readTextValue(
        "controlnet_compat_mode",
        primaryDefaults.controlnet_compat_mode ?? "warn"
    );
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];

    const idempotencyKey = WorkflowClient.makeIdempotencyKey();

    try {
        const tasks = [];

        if (controlnetEnabled) {
            if (!controlnetState?.previewBlob) {
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
                controlnet_conditioning_scale,
                controlnet_guess_mode,
                control_guidance_start,
                control_guidance_end,
                controlnet_compat_mode,
            };
            if (controlnetState?.preprocessorId) {
                inputs.controlnet_preprocessor_id = controlnetState.preprocessorId;
            }
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
