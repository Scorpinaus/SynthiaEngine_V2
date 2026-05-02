const TASK_WAN_IMAGE2VIDEO = "wan.image2video";

const WAN_I2V_DEFAULTS = {
    prompt: "",
    negative_prompt: "",
    steps: 50,
    guidance_scale: 5.0,
    width: 832,
    height: 480,
    num_frames: 81,
    fps: 16,
    num_videos: 1,
    memory_preset: "offload",
    quantization: "none",
    experimental_ack: true,
    model: "D:\\diffusion\\diffusers\\Wan2.1-I2V-14B-480P-Diffusers",
};

const videoGallery = createVideoGalleryViewer({
    buildVideoUrl: (path, idx, stamp) => `${API_BASE}${path}?t=${stamp}_${idx}`,
});

let activeWanI2vJobToken = 0;
let activeWanI2vEventSource = null;
let didInitWanI2vPage = false;

function closeActiveWanI2vEventSource() {
    if (activeWanI2vEventSource) {
        activeWanI2vEventSource.close();
        activeWanI2vEventSource = null;
    }
}

async function validateTaskInputsOrThrow(taskType, inputs) {
    if (!window.WorkflowInputValidator?.assertTaskInputs) {
        return;
    }
    await window.WorkflowInputValidator.assertTaskInputs(API_BASE, taskType, inputs);
}

async function uploadRequiredImage() {
    const input = document.getElementById("image");
    const file = input?.files?.[0];
    if (!file) {
        throw new Error("WAN I2V requires an input image.");
    }
    return await WorkflowClient.uploadArtifact(API_BASE, file, file.name || "wan-i2v.png");
}

async function collectWanI2vInputs(defaults) {
    const imageArtifact = await uploadRequiredImage();
    const experimentalAck = Boolean(document.getElementById("experimental_ack")?.checked);
    const inputs = {
        image: { artifact_id: imageArtifact.artifact_id },
        prompt: WorkflowClient.readTextValue("prompt", defaults.prompt ?? WAN_I2V_DEFAULTS.prompt),
        negative_prompt: WorkflowClient.readTextValue(
            "negative_prompt",
            defaults.negative_prompt ?? WAN_I2V_DEFAULTS.negative_prompt
        ),
        steps: WorkflowClient.readNumberValue("steps", defaults.steps ?? WAN_I2V_DEFAULTS.steps, {
            integer: true,
        }),
        guidance_scale: WorkflowClient.readNumberValue(
            "guidance_scale",
            defaults.guidance_scale ?? WAN_I2V_DEFAULTS.guidance_scale
        ),
        width: WAN_I2V_DEFAULTS.width,
        height: WAN_I2V_DEFAULTS.height,
        seed: WorkflowClient.readSeedValue("seed"),
        model: WorkflowClient.readTextValue("model", defaults.model ?? WAN_I2V_DEFAULTS.model),
        num_frames: WorkflowClient.readNumberValue(
            "num_frames",
            defaults.num_frames ?? WAN_I2V_DEFAULTS.num_frames,
            { integer: true }
        ),
        fps: WorkflowClient.readNumberValue("fps", defaults.fps ?? WAN_I2V_DEFAULTS.fps, {
            integer: true,
        }),
        num_videos: WAN_I2V_DEFAULTS.num_videos,
        memory_preset: WorkflowClient.readTextValue(
            "memory_preset",
            defaults.memory_preset ?? WAN_I2V_DEFAULTS.memory_preset
        ),
        quantization: WorkflowClient.readTextValue(
            "quantization",
            defaults.quantization ?? WAN_I2V_DEFAULTS.quantization
        ),
        experimental_ack: experimentalAck,
    };
    if (!inputs.experimental_ack) {
        throw new Error("WAN I2V requires acknowledging the slow experimental warning.");
    }
    return inputs;
}

async function generate() {
    const token = ++activeWanI2vJobToken;
    closeActiveWanI2vEventSource();

    try {
        const catalog = window.WorkflowCatalog?.load
            ? await window.WorkflowCatalog.load(API_BASE)
            : null;
        const defaults = catalog?.tasks?.[TASK_WAN_IMAGE2VIDEO]?.input_defaults ?? {};
        const inputs = await collectWanI2vInputs(defaults);

        await validateTaskInputsOrThrow(TASK_WAN_IMAGE2VIDEO, inputs);

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: TASK_WAN_IMAGE2VIDEO,
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

        activeWanI2vEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeWanI2vJobToken,
            onDone: (job) => {
                if (job?.status === "succeeded") {
                    const videos = job?.result?.outputs;
                    videoGallery.setVideos(Array.isArray(videos) ? videos : []);
                } else {
                    videoGallery.setVideos([]);
                }
            },
            onError: () => {
                if (token !== activeWanI2vJobToken) {
                    return;
                }
                videoGallery.setVideos([]);
            },
        });
    } catch (error) {
        if (
            error instanceof Error &&
            (error.message.startsWith("Input validation failed for ") ||
                error.message.startsWith("WAN I2V requires"))
        ) {
            alert(error.message);
        }
        console.warn("Failed to generate WAN I2V videos:", error);
        videoGallery.setVideos([]);
    }
}

function initWanI2vPage() {
    if (didInitWanI2vPage) {
        return;
    }
    didInitWanI2vPage = true;

    videoGallery.render();
    document.getElementById("generate-button")?.addEventListener("click", () => {
        generate();
    });

    if (window.WorkflowCatalog?.load) {
        void window.WorkflowCatalog
            .load(API_BASE)
            .then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(TASK_WAN_IMAGE2VIDEO, {
                    steps: "steps",
                    guidance_scale: "guidance_scale",
                    model: "model",
                    num_frames: "num_frames",
                    fps: "fps",
                    memory_preset: "memory_preset",
                    quantization: "quantization",
                    experimental_ack: "experimental_ack",
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

runWhenDomReady(initWanI2vPage);
