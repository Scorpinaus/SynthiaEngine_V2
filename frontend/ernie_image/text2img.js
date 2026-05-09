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

function setCheckboxValue(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) {
        return;
    }
    el.checked = Boolean(value);
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

function collectErnieImagePresetSettings() {
    return {
        prompt: WorkflowClient.readTextValue("prompt", ""),
        negative_prompt: WorkflowClient.readTextValue("negative_prompt", ""),
        steps: WorkflowClient.readNumberValue("steps", 8, { integer: true }),
        guidance_scale: WorkflowClient.readNumberValue("cfg", 1.0),
        seed: WorkflowClient.readSeedValue("seed"),
        width: WorkflowClient.readNumberValue("width", 768, { integer: true }),
        height: WorkflowClient.readNumberValue("height", 768, { integer: true }),
        model: WorkflowClient.readTextValue("model_select", null),
        num_images: WorkflowClient.readNumberValue("num_images", 1, { integer: true }),
        use_pe: Boolean(document.getElementById("use_pe")?.checked),
        load_pe: Boolean(document.getElementById("load_pe")?.checked),
        memory_preset: WorkflowClient.readTextValue("memory_preset", "sequential_offload"),
    };
}

async function applyErnieImagePresetSettings(settings) {
    setInputValue("prompt", settings.prompt);
    setInputValue("negative_prompt", settings.negative_prompt);
    setInputValue("steps", settings.steps);
    setInputValue("cfg", settings.guidance_scale);
    setInputValue("seed", settings.seed);
    setInputValue("width", settings.width);
    setInputValue("height", settings.height);
    setModelSelection(settings.model);
    setInputValue("num_images", settings.num_images);
    setCheckboxValue("use_pe", settings.use_pe);
    setCheckboxValue("load_pe", settings.load_pe);
    setInputValue("memory_preset", settings.memory_preset);
}

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=ernie-image`);
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
        fallback.value = "ERNIE-Image-Turbo";
        fallback.textContent = "ERNIE-Image-Turbo (hub, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();
window.PresetPanel?.init({
    apiBase: API_BASE,
    family: "ernie-image",
    taskType: "ernie-image.text2img",
    collectSettings: collectErnieImagePresetSettings,
    applySettings: applyErnieImagePresetSettings,
});
if (window.WorkflowCatalog?.load) {
    void window.WorkflowCatalog
        .load(API_BASE)
        .then(() => {
            window.WorkflowCatalog.applyDefaultsToForm("ernie-image.text2img", {
                negative_prompt: "negative_prompt",
                steps: "steps",
                cfg: "guidance_scale",
                width: "width",
                height: "height",
                num_images: "num_images",
                memory_preset: "memory_preset",
            });
            const defaults =
                window.WorkflowCatalog.getTask?.("ernie-image.text2img")?.input_defaults ?? {};
            setCheckboxValue("use_pe", defaults.use_pe);
            setCheckboxValue("load_pe", defaults.load_pe);
        })
        .catch(() => {});
}

function baseInput(inputs, defaults) {
    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue(
        "negative_prompt",
        defaults.negative_prompt ?? ""
    );
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 8, { integer: true });
    const guidance_scale = WorkflowClient.readNumberValue("cfg", defaults.guidance_scale ?? 1.0);
    const seed = WorkflowClient.readSeedValue("seed");
    const width = WorkflowClient.readNumberValue("width", defaults.width ?? 768, { integer: true });
    const height = WorkflowClient.readNumberValue("height", defaults.height ?? 768, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, {
        integer: true,
    });
    const use_pe = Boolean(document.getElementById("use_pe")?.checked);
    const load_pe = Boolean(document.getElementById("load_pe")?.checked);
    const memory_preset = WorkflowClient.readTextValue(
        "memory_preset",
        defaults.memory_preset ?? "sequential_offload"
    );

    Object.assign(inputs, {
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        seed,
        width,
        height,
        model,
        num_images,
        use_pe,
        load_pe,
        memory_preset,
    });

    return inputs;
}

async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["ernie-image.text2img"]?.input_defaults ?? {};
    const inputs = {};
    baseInput(inputs, defaults);
    console.log("Generate payload", inputs);

    try {
        const workflowPayload = {
            tasks: [{ id: "t1", type: "ernie-image.text2img", inputs }],
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
        console.warn("Failed to generate ERNIE-Image images:", error);
        gallery.setImages([]);
    }
}
