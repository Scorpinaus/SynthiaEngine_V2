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

function collectAnimaPresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 35, { integer: true }),
        guidance_scale: WorkflowClient.readNumberValue("cfg", 4.5),
        scheduler: WorkflowClient.readTextValue("scheduler", "flowmatch_euler"),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", 1024, { integer: true }),
        height: WorkflowClient.readNumberValue("height", 1024, { integer: true }),
        model: WorkflowClient.readTextValue("model_select", null),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
        memory_preset: WorkflowClient.readTextValue("memory_preset", "sequential_offload"),
    };
}

async function applyAnimaPresetSettings(settings) {
    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.guidance_scale);
    setInputValue("scheduler", settings.scheduler);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setModelSelection(settings.model);
    setInputValue("num_images", settings.num_images);
    setInputValue("memory_preset", settings.memory_preset);
}

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=anima`);
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
        fallback.value = "Anima-Preview-3";
        fallback.textContent = "Anima-Preview-3 (hub, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();
window.PresetPanel?.init({
    apiBase: API_BASE,
    family: "anima",
    taskType: "anima.text2img",
    collectSettings: collectAnimaPresetSettings,
    applySettings: applyAnimaPresetSettings,
});
if (window.WorkflowCatalog?.load) {
    void window.WorkflowCatalog
        .load(API_BASE)
        .then(() => {
            window.WorkflowCatalog.applyDefaultsToForm("anima.text2img", {
                negative_prompt: "negative_prompt",
                steps: "steps",
                cfg: "guidance_scale",
                width: "width",
                height: "height",
                num_images: "num_images",
                memory_preset: "memory_preset",
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
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 35, { integer: true });
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 4.5);
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "flowmatch_euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 1024, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 1024, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const memory_preset = WorkflowClient.readTextValue(
        "memory_preset",
        defaults.memory_preset ?? "sequential_offload"
    );

    Object.assign(inputs, {
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
        memory_preset,
    });

    return inputs;
}

async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["anima.text2img"]?.input_defaults ?? {};
    const inputs = {};
    baseInput(inputs, defaults);
    console.log("Generate payload", inputs);

    try {
        const workflowPayload = {
            tasks: [{ id: "t1", type: "anima.text2img", inputs }],
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
        console.warn("Failed to generate Anima images:", error);
        gallery.setImages([]);
    }
}
