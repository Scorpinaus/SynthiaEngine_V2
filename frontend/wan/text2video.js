const TASK_WAN_TEXT2VIDEO = "wan.text2video";

const WAN_DEFAULTS = {
    prompt: "",
    negative_prompt: "",
    steps: 30,
    guidance_scale: 6.0,
    width: 832,
    height: 480,
    num_frames: 49,
    fps: 16,
    num_videos: 1,
    memory_preset: "safe",
    model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
};

const videoGallery = createVideoGalleryViewer({
    buildVideoUrl: (path, idx, stamp) => `${API_BASE}${path}?t=${stamp}_${idx}`,
});

let activeWanJobToken = 0;
let activeWanEventSource = null;
let didInitWanPage = false;

function closeActiveWanEventSource() {
    if (activeWanEventSource) {
        activeWanEventSource.close();
        activeWanEventSource = null;
    }
}

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

function collectWanInputs(defaults) {
    return {
        prompt: WorkflowClient.readTextValue("prompt", defaults.prompt ?? WAN_DEFAULTS.prompt),
        negative_prompt: WorkflowClient.readTextValue(
            "negative_prompt",
            defaults.negative_prompt ?? WAN_DEFAULTS.negative_prompt
        ),
        steps: WorkflowClient.readNumberValue("steps", defaults.steps ?? WAN_DEFAULTS.steps, {
            integer: true,
        }),
        guidance_scale: WorkflowClient.readNumberValue(
            "guidance_scale",
            defaults.guidance_scale ?? WAN_DEFAULTS.guidance_scale
        ),
        width: WorkflowClient.readNumberValue("width", defaults.width ?? WAN_DEFAULTS.width, {
            integer: true,
        }),
        height: WorkflowClient.readNumberValue(
            "height",
            defaults.height ?? WAN_DEFAULTS.height,
            { integer: true }
        ),
        seed: WorkflowClient.readSeedValue("seed"),
        model: WorkflowClient.readTextValue("model", defaults.model ?? WAN_DEFAULTS.model),
        num_frames: WorkflowClient.readNumberValue(
            "num_frames",
            defaults.num_frames ?? WAN_DEFAULTS.num_frames,
            { integer: true }
        ),
        fps: WorkflowClient.readNumberValue("fps", defaults.fps ?? WAN_DEFAULTS.fps, {
            integer: true,
        }),
        num_videos: WAN_DEFAULTS.num_videos,
        memory_preset: WorkflowClient.readTextValue(
            "memory_preset",
            defaults.memory_preset ?? WAN_DEFAULTS.memory_preset
        ),
    };
}

async function generate() {
    const token = ++activeWanJobToken;
    closeActiveWanEventSource();

    try {
        const catalog = window.WorkflowCatalog?.load
            ? await window.WorkflowCatalog.load(API_BASE)
            : null;
        const defaults = catalog?.tasks?.[TASK_WAN_TEXT2VIDEO]?.input_defaults ?? {};
        const inputs = collectWanInputs(defaults);

        await validateTaskInputsOrThrow(TASK_WAN_TEXT2VIDEO, inputs);

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: TASK_WAN_TEXT2VIDEO,
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

        activeWanEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeWanJobToken,
            onDone: (job) => {
                if (job?.status === "succeeded") {
                    const videos = job?.result?.outputs;
                    videoGallery.setVideos(Array.isArray(videos) ? videos : []);
                } else {
                    videoGallery.setVideos([]);
                }
            },
            onError: () => {
                if (token !== activeWanJobToken) {
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
        console.warn("Failed to generate WAN videos:", error);
        videoGallery.setVideos([]);
    }
}

function initWanPage() {
    if (didInitWanPage) {
        return;
    }
    didInitWanPage = true;

    videoGallery.render();
    document.getElementById("generate-button")?.addEventListener("click", () => {
        generate();
    });

    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(TASK_WAN_TEXT2VIDEO, {
                    steps: "steps",
                    guidance_scale: "guidance_scale",
                    width: "width",
                    height: "height",
                    model: "model",
                    num_frames: "num_frames",
                    fps: "fps",
                    memory_preset: "memory_preset",
                });
            })
            .catch(() => {});
    }
}

function runWhenDomReady(initFn) {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initFn, { once: true });
        return;
    }
    initFn();
}

runWhenDomReady(initWanPage);
