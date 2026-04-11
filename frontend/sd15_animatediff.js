const TASK_ANIMATEDIFF_TEXT2VIDEO = "sd15.animatediff.text2video";

const ANIMATEDIFF_DEFAULTS = {
    prompt: "",
    negative_prompt: "",
    steps: 25,
    cfg: 7.5,
    scheduler: "ddim",
    width: 512,
    height: 512,
    num_frames: 16,
    fps: 8,
    num_videos: 1,
    clip_skip: 1,
    weighting_policy: "diffusers-like",
    motion_adapter: "guoyww/animatediff-motion-adapter-v1-5-2",
    model: null,
    modelSelectOption: "stable-diffusion-v1-5",
};

const videoGallery = createVideoGalleryViewer({
    buildVideoUrl: (path, idx, stamp) => `${API_BASE}${path}?t=${stamp}_${idx}`,
});

let activeAnimatediffJobToken = 0;
let activeAnimatediffEventSource = null;
let loraPanelReady = Promise.resolve();
let didInitAnimateDiffPage = false;

function closeActiveAnimateDiffEventSource() {
    if (activeAnimatediffEventSource) {
        activeAnimatediffEventSource.close();
        activeAnimatediffEventSource = null;
    }
}

function setInputValue(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) {
        return;
    }
    el.value = value === null ? "" : String(value);
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
            throw new Error("No SD1.5 models returned.");
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
        fallback.value = ANIMATEDIFF_DEFAULTS.modelSelectOption;
        fallback.textContent = `${ANIMATEDIFF_DEFAULTS.modelSelectOption} (sd15, diffusers)`;
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load SD1.5 models:", error);
    }
}

function collectAnimateDiffPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ANIMATEDIFF_DEFAULTS.prompt),
        negative_prompt: WorkflowClient.readTextValue(
            "negative_prompt",
            ANIMATEDIFF_DEFAULTS.negative_prompt
        ),
        steps: WorkflowClient.readNumberValue("steps", ANIMATEDIFF_DEFAULTS.steps, {
            integer: true,
        }),
        cfg: WorkflowClient.readNumberValue("cfg", ANIMATEDIFF_DEFAULTS.cfg),
        scheduler: WorkflowClient.readTextValue("scheduler", ANIMATEDIFF_DEFAULTS.scheduler),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", ANIMATEDIFF_DEFAULTS.width, {
            integer: true,
        }),
        height: WorkflowClient.readNumberValue("height", ANIMATEDIFF_DEFAULTS.height, {
            integer: true,
        }),
        num_frames: WorkflowClient.readNumberValue(
            "num_frames",
            ANIMATEDIFF_DEFAULTS.num_frames,
            { integer: true }
        ),
        fps: WorkflowClient.readNumberValue("fps", ANIMATEDIFF_DEFAULTS.fps, {
            integer: true,
        }),
        num_videos: WorkflowClient.readNumberValue(
            "num_videos",
            ANIMATEDIFF_DEFAULTS.num_videos,
            { integer: true }
        ),
        clip_skip: WorkflowClient.readNumberValue(
            "clip_skip",
            ANIMATEDIFF_DEFAULTS.clip_skip,
            { integer: true }
        ),
        weighting_policy: WorkflowClient.readTextValue(
            "weighting_policy",
            ANIMATEDIFF_DEFAULTS.weighting_policy
        ),
        motion_adapter: WorkflowClient.readTextValue(
            "motion_adapter",
            ANIMATEDIFF_DEFAULTS.motion_adapter
        ),
        model: WorkflowClient.readTextValue("model_select", ANIMATEDIFF_DEFAULTS.model),
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
    };
}

async function applyAnimateDiffPresetSettings(settings) {
    await loraPanelReady;

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.cfg);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setInputValue("num_frames", settings.num_frames);
    setInputValue("fps", settings.fps);
    setInputValue("num_videos", settings.num_videos);
    setInputValue("clip_skip", settings.clip_skip);
    setInputValue("weighting_policy", settings.weighting_policy);
    setInputValue("motion_adapter", settings.motion_adapter);
    setModelSelection(settings.model);

    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
    }
}

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

function collectAnimateDiffInputs(defaults) {
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : defaults.model ?? ANIMATEDIFF_DEFAULTS.model;
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const loraEnabled = Array.isArray(loraAdapters) && loraAdapters.length > 0;

    return {
        prompt: WorkflowClient.readTextValue("prompt", defaults.prompt ?? ANIMATEDIFF_DEFAULTS.prompt),
        negative_prompt: WorkflowClient.readTextValue(
            "negative_prompt",
            defaults.negative_prompt ?? ANIMATEDIFF_DEFAULTS.negative_prompt
        ),
        steps: WorkflowClient.readNumberValue(
            "steps",
            defaults.steps ?? ANIMATEDIFF_DEFAULTS.steps,
            { integer: true }
        ),
        cfg: WorkflowClient.readNumberValue("cfg", defaults.cfg ?? ANIMATEDIFF_DEFAULTS.cfg),
        scheduler: WorkflowClient.readTextValue(
            "scheduler",
            defaults.scheduler ?? ANIMATEDIFF_DEFAULTS.scheduler
        ),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue(
            "width",
            defaults.width ?? ANIMATEDIFF_DEFAULTS.width,
            { integer: true }
        ),
        height: WorkflowClient.readNumberValue(
            "height",
            defaults.height ?? ANIMATEDIFF_DEFAULTS.height,
            { integer: true }
        ),
        motion_adapter: WorkflowClient.readTextValue(
            "motion_adapter",
            defaults.motion_adapter ?? ANIMATEDIFF_DEFAULTS.motion_adapter
        ),
        num_frames: WorkflowClient.readNumberValue(
            "num_frames",
            defaults.num_frames ?? ANIMATEDIFF_DEFAULTS.num_frames,
            { integer: true }
        ),
        fps: WorkflowClient.readNumberValue("fps", defaults.fps ?? ANIMATEDIFF_DEFAULTS.fps, {
            integer: true,
        }),
        num_videos: WorkflowClient.readNumberValue(
            "num_videos",
            defaults.num_videos ?? ANIMATEDIFF_DEFAULTS.num_videos,
            { integer: true }
        ),
        clip_skip: WorkflowClient.readNumberValue(
            "clip_skip",
            defaults.clip_skip ?? ANIMATEDIFF_DEFAULTS.clip_skip,
            { integer: true }
        ),
        weighting_policy: WorkflowClient.readTextValue(
            "weighting_policy",
            defaults.weighting_policy ?? ANIMATEDIFF_DEFAULTS.weighting_policy
        ),
        model,
        lora: {
            lora_enabled: loraEnabled,
            lora_adapters: loraEnabled ? loraAdapters : [],
        },
    };
}

async function generate() {
    const token = ++activeAnimatediffJobToken;
    closeActiveAnimateDiffEventSource();

    try {
        const catalog = window.WorkflowCatalog?.load
            ? await window.WorkflowCatalog.load(API_BASE)
            : null;
        const defaults = catalog?.tasks?.[TASK_ANIMATEDIFF_TEXT2VIDEO]?.input_defaults ?? {};
        const inputs = collectAnimateDiffInputs(defaults);

        await validateTaskInputsOrThrow(TASK_ANIMATEDIFF_TEXT2VIDEO, inputs);

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: TASK_ANIMATEDIFF_TEXT2VIDEO,
                    inputs,
                },
            ],
            return: "@t1.videos",
        };
        const createdJob = await WorkflowClient.submitWorkflow(
            API_BASE,
            workflowPayload,
            WorkflowClient.makeIdempotencyKey()
        );
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        activeAnimatediffEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeAnimatediffJobToken,
            onDone: (job) => {
                if (job?.status === "succeeded") {
                    const videos = job?.result?.outputs;
                    videoGallery.setVideos(Array.isArray(videos) ? videos : []);
                } else {
                    videoGallery.setVideos([]);
                }
            },
            onError: () => {
                if (token !== activeAnimatediffJobToken) {
                    return;
                }
                videoGallery.setVideos([]);
            },
        });
    } catch (error) {
        if (
            error instanceof Error &&
            error.message.startsWith("Input validation failed for ")
        ) {
            alert(error.message);
        }
        console.warn("Failed to generate SD1.5 AnimateDiff videos:", error);
        videoGallery.setVideos([]);
    }
}

function initAnimateDiffPage() {
    if (didInitAnimateDiffPage) {
        return;
    }
    didInitAnimateDiffPage = true;

    videoGallery.render();
    document.getElementById("generate-button")?.addEventListener("click", () => {
        generate();
    });

    void loadModels();
    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(TASK_ANIMATEDIFF_TEXT2VIDEO, {
                    steps: "steps",
                    cfg: "cfg",
                    width: "width",
                    height: "height",
                    motion_adapter: "motion_adapter",
                    num_frames: "num_frames",
                    fps: "fps",
                    num_videos: "num_videos",
                    clip_skip: "clip_skip",
                    weighting_policy: "weighting_policy",
                });
            })
            .catch(() => {});
    }

    loraPanelReady = window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" }) ?? Promise.resolve();
    window.PresetPanel?.init({
        apiBase: API_BASE,
        family: "sd15",
        taskType: TASK_ANIMATEDIFF_TEXT2VIDEO,
        collectSettings: collectAnimateDiffPresetSettings,
        applySettings: applyAnimateDiffPresetSettings,
    });
}

function runWhenDomReady(initFn) {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initFn, { once: true });
        return;
    }
    initFn();
}

runWhenDomReady(initAnimateDiffPage);
