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
    model: "D:\\diffusion\\diffusers\\Wan2.1-T2V-1.3B-Diffusers",
    vaceModel: "D:\\diffusion\\diffusers\\Wan2.1-VACE-1.3B-diffusers",
    conditioning_scale: 1.0,
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

function readWanMode() {
    return WorkflowClient.readTextValue("wan_mode", "t2v");
}

function readResolution() {
    const value = WorkflowClient.readTextValue("resolution", "832x480");
    if (value === "512x512") {
        return { width: 512, height: 512 };
    }
    return { width: 832, height: 480 };
}

function setWanMode(mode) {
    const isVace = mode === "vace";
    const modelInput = document.getElementById("model");
    const vaceFields = document.getElementById("vace-fields");
    if (modelInput) {
        modelInput.value = isVace ? WAN_DEFAULTS.vaceModel : WAN_DEFAULTS.model;
    }
    vaceFields?.classList.toggle("is-hidden", !isVace);
}

async function uploadOptionalArtifact(elementId, fallbackName) {
    const input = document.getElementById(elementId);
    const file = input?.files?.[0];
    if (!file) {
        return null;
    }
    return await WorkflowClient.uploadArtifact(API_BASE, file, file.name || fallbackName);
}

async function collectWanInputs(defaults) {
    const mode = readWanMode();
    const resolution = readResolution();
    const inputs = {
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
        width: resolution.width,
        height: resolution.height,
        seed: WorkflowClient.readSeedValue("seed"),
        model: WorkflowClient.readTextValue(
            "model",
            mode === "vace" ? WAN_DEFAULTS.vaceModel : defaults.model ?? WAN_DEFAULTS.model
        ),
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

    if (mode === "vace") {
        const referenceArtifact = await uploadOptionalArtifact("reference_image", "reference.png");
        const maskArtifact = await uploadOptionalArtifact("mask_image", "mask.png");
        const videoArtifact = await uploadOptionalArtifact("conditioning_video", "conditioning.mp4");
        if (!referenceArtifact || !maskArtifact || !videoArtifact) {
            throw new Error("WAN VACE requires a reference image, mask image, and conditioning video.");
        }
        inputs.reference_image = { artifact_id: referenceArtifact.artifact_id };
        inputs.mask_image = { artifact_id: maskArtifact.artifact_id };
        inputs.conditioning_video = { artifact_id: videoArtifact.artifact_id };
        inputs.conditioning_scale = WorkflowClient.readNumberValue(
            "conditioning_scale",
            defaults.conditioning_scale ?? WAN_DEFAULTS.conditioning_scale
        );
    }

    return inputs;
}

async function generate() {
    const token = ++activeWanJobToken;
    closeActiveWanEventSource();

    try {
        const catalog = window.WorkflowCatalog?.load
            ? await window.WorkflowCatalog.load(API_BASE)
            : null;
        const defaults = catalog?.tasks?.[TASK_WAN_TEXT2VIDEO]?.input_defaults ?? {};
        const inputs = await collectWanInputs(defaults);

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
        } else if (error instanceof Error && error.message.startsWith("WAN VACE requires")) {
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
    setWanMode(readWanMode());
    document.getElementById("wan_mode")?.addEventListener("change", () => {
        setWanMode(readWanMode());
    });
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
                    conditioning_scale: "conditioning_scale",
                });
                setWanMode(readWanMode());
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
