const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;
let loraPanelReady = Promise.resolve();

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
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

function collectQwenImageImg2ImgPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 30, { integer: true }),
        true_cfg_scale: WorkflowClient.readNumberValue("true_cfg", 4.0),
        guidance_scale: WorkflowClient.readNumberValue("cfg", 7.5),
        scheduler: WorkflowClient.readTextValue("scheduler", "euler"),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", 1024, { integer: true }),
        height: WorkflowClient.readNumberValue("height", 1024, { integer: true }),
        model: WorkflowClient.readTextValue("model_select", null),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
        strength: WorkflowClient.readNumberValue("strength", 0.75),
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
    };
}

async function applyQwenImageImg2ImgPresetSettings(settings) {
    await loraPanelReady;

    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("true_cfg", settings.true_cfg_scale);
    setInputValue("cfg", settings.guidance_scale);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setModelSelection(settings.model);
    setInputValue("num_images", settings.num_images);
    setInputValue("strength", settings.strength);

    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
    }
}

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=qwen-image`);
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
        fallback.value = "qwen-image";
        fallback.textContent = "qwen-image (diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();
loraPanelReady = window.LoraPanel?.init({ apiBase: API_BASE, family: "qwen-image" }) ?? Promise.resolve();
window.PresetPanel?.init({
    apiBase: API_BASE,
    family: "qwen-image",
    taskType: "qwen-image.img2img",
    collectSettings: collectQwenImageImg2ImgPresetSettings,
    applySettings: applyQwenImageImg2ImgPresetSettings,
});
if (window.WorkflowCatalog?.load) {
    void window.WorkflowCatalog
        .load(API_BASE)
        .then(() => {
            window.WorkflowCatalog.applyDefaultsToForm("qwen-image.img2img", {
                steps: "steps",
                true_cfg: "true_cfg_scale",
                cfg: "guidance_scale",
                width: "width",
                height: "height",
                strength: "strength",
                num_images: "num_images",
            });
        })
        .catch(() => {});
}

function baseInput(inputs, defaults) {
    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        defaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 30, { integer: true });
    const true_cfg_scale = WorkflowClient.readNumberValue("true_cfg", defaults.true_cfg_scale ?? 4.0);
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 7.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 1024, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 1024, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const strength = WorkflowClient.readNumberValue("strength", defaults.strength ?? 0.75);

    Object.assign(inputs, {
        prompt,
        negative_prompt,
        steps,
        true_cfg_scale,
        guidance_scale,
        scheduler,
        seed,
        width,
        height,
        model,
        num_images,
        strength,
    });

    return inputs;
}

async function generateQwenImageImg2Img() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["qwen-image.img2img"]?.input_defaults ?? {};
    const inputs = {};
    baseInput(inputs, defaults);
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const loraAdaptersEnabled = Array.isArray(loraAdapters) && loraAdapters.length > 0;
    inputs.Lora = {
        enabled: loraAdaptersEnabled,
        adapters: loraAdaptersEnabled ? loraAdapters : [],
    };
    if (loraAdaptersEnabled) {
        inputs.lora_adapters = loraAdapters;
    }
    const initialImageInput = document.getElementById("initial_image");

    if (!initialImageInput.files || initialImageInput.files.length === 0) {
        alert("Please choose an initial image.");
        return;
    }

    try {
        const initialFile = initialImageInput.files[0];
        const uploaded = await WorkflowClient.uploadArtifact(
            API_BASE,
            initialFile,
            initialFile.name || "initial.png",
        );
        inputs.initial_image = `@artifact:${uploaded.artifact_id}`;

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: "qwen-image.img2img",
                    inputs,
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
        console.warn("Failed to run Qwen-Image img2img job:", error);
        gallery.setImages([]);
    }
}
