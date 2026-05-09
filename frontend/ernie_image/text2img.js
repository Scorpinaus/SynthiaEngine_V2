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

function setCheckboxValue(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) {
        return;
    }
    el.checked = Boolean(value);
}

function countLabel(count, singular, plural = `${singular}s`) {
    const value = Number(count);
    const safeCount = Number.isFinite(value) ? value : 0;
    return `${safeCount} ${safeCount === 1 ? singular : plural}`;
}

function setText(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

function updateAdapterSummary() {
    const selectedAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    const lora = window.LoraPanel?.getSummary?.() ?? {
        available: 0,
        selected: Array.isArray(selectedAdapters) ? selectedAdapters.length : 0,
    };
    const availableLoras = Number(lora.available ?? 0);
    const selectedLoras = Number(lora.selected ?? 0);

    setText(
        "adapter_summary_label",
        `${countLabel(availableLoras, "adapter available", "adapters available")} / ${countLabel(selectedLoras, "adapter active", "adapters active")}`
    );
    setText("adapter-tab-lora-badge", countLabel(selectedLoras, "selected", "selected"));
    setText("adapter-overview-lora-count", countLabel(availableLoras, "LoRA available", "LoRAs available"));
    setText(
        "adapter-overview-lora-detail",
        selectedLoras > 0 ? `${countLabel(selectedLoras, "LoRA")} selected.` : "No LoRAs selected."
    );
}

function setAdapterTab(tabName) {
    const target = String(tabName || "overview");
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
        const isActive = tab.getAttribute("data-adapter-tab") === target;
        tab.classList.toggle("is-active", isActive);
        tab.setAttribute("aria-selected", String(isActive));
    });
    document.querySelectorAll("[data-adapter-panel]").forEach((panel) => {
        const isActive = panel.getAttribute("data-adapter-panel") === target;
        panel.classList.toggle("is-active", isActive);
        panel.toggleAttribute("hidden", !isActive);
    });
    updateAdapterSummary();
}

function setAdapterModalOpen(isOpen) {
    const modal = document.getElementById("adapter-modal");
    if (!modal) {
        return;
    }
    modal.classList.toggle("hidden", !isOpen);
    modal.setAttribute("aria-hidden", String(!isOpen));
    if (isOpen) {
        updateAdapterSummary();
        document.getElementById("adapter-modal-close")?.focus();
    } else {
        document.getElementById("adapter-modal-open")?.focus();
    }
}

function hideAdapterSection(sectionName) {
    document.querySelector(`[data-adapter-tab="${sectionName}"]`)?.remove();
    document.querySelector(`[data-adapter-panel="${sectionName}"]`)?.remove();
    document.querySelector(`[data-adapter-tab-jump="${sectionName}"]`)?.remove();
}

function initAdapterModal() {
    const modal = document.getElementById("adapter-modal");
    if (!modal) {
        return;
    }
    setText("adapter-modal-subtitle", "ERNIE-Image adapter stack");
    hideAdapterSection("controlnet");
    hideAdapterSection("ipadapter");
    document.getElementById("adapter-modal-open")?.addEventListener("click", () => {
        setAdapterModalOpen(true);
    });
    document.getElementById("adapter-modal-close")?.addEventListener("click", () => {
        setAdapterModalOpen(false);
    });
    document.getElementById("adapter-modal-overlay")?.addEventListener("click", () => {
        setAdapterModalOpen(false);
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !modal.classList.contains("hidden")) {
            setAdapterModalOpen(false);
        }
    });
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
        tab.addEventListener("click", () => {
            setAdapterTab(tab.getAttribute("data-adapter-tab"));
        });
    });
    document.querySelectorAll("[data-adapter-tab-jump]").forEach((button) => {
        button.addEventListener("click", () => {
            setAdapterTab(button.getAttribute("data-adapter-tab-jump"));
        });
    });
    window.addEventListener("adapter-summary-changed", updateAdapterSummary);
    modal.addEventListener("change", () => {
        window.setTimeout(updateAdapterSummary, 0);
    });
    modal.addEventListener("click", () => {
        window.setTimeout(updateAdapterSummary, 0);
    });
    updateAdapterSummary();
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
        lora_adapters: window.LoraPanel?.getSelectedAdapters?.() ?? [],
    };
}

async function applyErnieImagePresetSettings(settings) {
    await loraPanelReady;

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
    if (Array.isArray(settings.lora_adapters)) {
        window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
        updateAdapterSummary();
    }
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
window.AdapterPanel?.render?.();
initAdapterModal();
loraPanelReady = window.LoraPanel?.init({ apiBase: API_BASE, family: "ernie-image" }) ?? Promise.resolve();
loraPanelReady.then(() => {
    updateAdapterSummary();
    window.setTimeout(updateAdapterSummary, 500);
});
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
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
    if (Array.isArray(loraAdapters) && loraAdapters.length > 0) {
        inputs.lora_adapters = loraAdapters;
    }
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
