/** Shared controller for the ordinary single-task image generation pages. */
(function () {
    function setValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element && value !== undefined) {
            element.value = value === null ? "" : String(value);
        }
    }

    function setFieldValue(field, value) {
        const element = document.getElementById(field.element);
        if (!element || value === undefined) return;
        if (field.type === "checkbox") {
            element.checked = Boolean(value);
            return;
        }
        if (field.element === "model_select" && value &&
            !Array.from(element.options).some((option) => option.value === String(value))) {
            element.add(new Option(`${value} (preset)`, String(value)));
        }
        setValue(field.element, value);
    }

    function readField(field, defaults) {
        const fallback = defaults[field.key] ?? field.fallback;
        if (field.type === "checkbox") {
            return Boolean(document.getElementById(field.element)?.checked);
        }
        if (field.type === "seed") {
            return WorkflowClient.readSeedValue(field.element);
        }
        if (field.type === "number") {
            return WorkflowClient.readNumberValue(field.element, fallback, {
                integer: field.integer === true,
            });
        }
        return WorkflowClient.readTextValue(field.element, fallback);
    }

    function createFormController({ family, taskType, fallbackModel, fields }) {
        let loraReady = Promise.resolve();

        function collectSettings(defaults = {}) {
            const settings = {};
            for (const field of fields) {
                settings[field.key] = readField(field, defaults);
            }
            settings.lora_adapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
            return settings;
        }

        async function applySettings(settings) {
            await loraReady;
            for (const field of fields) {
                setFieldValue(field, settings[field.key]);
            }
            if (Array.isArray(settings.lora_adapters)) {
                window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
            }
        }

        async function loadModels() {
            const select = document.getElementById("model_select");
            if (!select) return;
            select.innerHTML = "";
            try {
                const response = await fetch(`${API_BASE}/models?family=${family}`);
                const models = await response.json();
                if (!Array.isArray(models) || models.length === 0) {
                    throw new Error("No models returned.");
                }
                models.forEach((model, index) => {
                    const option = document.createElement("option");
                    option.value = model.name ?? "";
                    option.textContent = `${model.name} (${model.family ?? "unknown"}, ${model.model_type ?? "unknown"})`;
                    option.selected = index === 0;
                    select.appendChild(option);
                });
            } catch (error) {
                const option = document.createElement("option");
                option.value = fallbackModel.value;
                option.textContent = fallbackModel.label;
                option.selected = true;
                select.appendChild(option);
                console.warn("Failed to load models:", error);
            }
        }

        async function defaults(requestedTaskType = taskType) {
            const catalog = window.WorkflowCatalog?.load
                ? await window.WorkflowCatalog.load(API_BASE)
                : null;
            return catalog?.tasks?.[requestedTaskType]?.input_defaults ?? {};
        }

        function applyCatalogDefaults(requestedTaskType = taskType, bindings = null) {
            if (!window.WorkflowCatalog?.load) return Promise.resolve();
            const resolvedBindings = bindings ?? Object.fromEntries(
                fields
                    .filter((field) => field.type !== "checkbox")
                    .map((field) => [field.element, field.key])
            );
            return window.WorkflowCatalog.load(API_BASE).then(() => {
                window.WorkflowCatalog.applyDefaultsToForm(requestedTaskType, resolvedBindings);
                const catalogDefaults = window.WorkflowCatalog.getTask?.(requestedTaskType)?.input_defaults ?? {};
                fields
                    .filter((field) => field.type === "checkbox")
                    .forEach((field) => setFieldValue(field, catalogDefaults[field.key]));
            }).catch(() => {});
        }

        function initLora() {
            loraReady = window.LoraPanel?.init({ apiBase: API_BASE, family }) ?? Promise.resolve();
            return loraReady;
        }

        function initPresets(collect = collectSettings, apply = applySettings) {
            window.PresetPanel?.init({
                apiBase: API_BASE,
                family,
                taskType,
                collectSettings: collect,
                applySettings: apply,
            });
        }

        return {
            applyCatalogDefaults,
            applySettings,
            collectSettings,
            defaults,
            initLora,
            initPresets,
            loadModels,
            ready: () => loraReady,
        };
    }

    function createJobController(setOutputs) {
        let activeToken = 0;
        let activeEventSource = null;

        async function run(payload, errorMessage, callbacks = {}) {
            const token = ++activeToken;
            activeEventSource?.close();
            activeEventSource = null;
            callbacks.onStateChange?.("submitting", null);
            try {
                const job = await WorkflowClient.submitWorkflow(
                    API_BASE,
                    payload,
                    WorkflowClient.makeIdempotencyKey(),
                );
                if (!job?.id) {
                    throw new Error("Job submit did not return an id.");
                }
                callbacks.onStateChange?.("queued", job.id);
                activeEventSource = WorkflowClient.watchJob(API_BASE, job.id, {
                    isStale: () => token !== activeToken,
                    onUpdate: (update) => {
                        callbacks.onStateChange?.(update?.status ?? "unknown", job.id);
                        callbacks.onUpdate?.(update, job.id);
                    },
                    onDone: (update) => {
                        const outputs = update?.status === "succeeded" ? update?.result?.outputs : [];
                        setOutputs(Array.isArray(outputs) ? outputs : []);
                        callbacks.onStateChange?.(update?.status ?? "done", job.id);
                        callbacks.onDone?.(update, job.id);
                    },
                    onError: () => {
                        if (token === activeToken) {
                            setOutputs([]);
                            callbacks.onStateChange?.("stream-error", job.id);
                            callbacks.onError?.(job.id);
                        }
                    },
                });
                return job;
            } catch (error) {
                if (error instanceof Error && error.message.startsWith("Input validation failed for ")) {
                    alert(error.message);
                }
                console.warn(errorMessage, error);
                setOutputs([]);
                callbacks.onStateChange?.("failed", null);
                callbacks.onFailure?.(error);
                return null;
            }
        }

        return {
            clear: () => setOutputs([]),
            close: () => activeEventSource?.close(),
            run,
        };
    }

    function createImageJobs() {
        const gallery = createGalleryViewer({
            buildImageUrl: (path, index, stamp) => API_BASE + path + `?t=${stamp}_${index}`,
        });
        gallery.render();
        return createJobController((outputs) => gallery.setImages(outputs));
    }

    function createVideoJobs() {
        const gallery = createVideoGalleryViewer({
            buildVideoUrl: (path, index, stamp) => API_BASE + path + `?t=${stamp}_${index}`,
        });
        gallery.render();
        return createJobController((outputs) => gallery.setVideos(outputs));
    }

    async function validateTasks(tasks) {
        if (!window.WorkflowInputValidator?.assertTaskInputs) return;
        for (const task of tasks) {
            await window.WorkflowInputValidator.assertTaskInputs(API_BASE, task.type, task.inputs);
        }
    }

    function runWhenDomReady(init) {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", init, { once: true });
            return;
        }
        init();
    }

    function create(config) {
        const gallery = createGalleryViewer({
            buildImageUrl: (path, index, stamp) => API_BASE + path + `?t=${stamp}_${index}`,
        });
        gallery.render();

        let activeToken = 0;
        let activeEventSource = null;
        let loraReady = Promise.resolve();

        function collectSettings(defaults = {}) {
            const settings = {};
            for (const field of config.fields) {
                settings[field.key] = readField(field, defaults);
            }
            if (config.lora !== false) {
                settings.lora_adapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
            }
            return settings;
        }

        async function applySettings(settings) {
            await loraReady;
            for (const field of config.fields) {
                setFieldValue(field, settings[field.key]);
            }
            if (config.lora !== false && Array.isArray(settings.lora_adapters)) {
                window.LoraPanel?.setSelectedAdapters?.(settings.lora_adapters);
            }
        }

        async function loadModels() {
            const select = document.getElementById("model_select");
            if (!select) {
                return;
            }
            select.innerHTML = "";
            try {
                const response = await fetch(`${API_BASE}/models?family=${config.family}`);
                const models = await response.json();
                if (!Array.isArray(models) || models.length === 0) {
                    throw new Error("No models returned.");
                }
                models.forEach((model, index) => {
                    const option = document.createElement("option");
                    option.value = model.name ?? "";
                    option.textContent = `${model.name} (${model.family ?? "unknown"}, ${model.model_type ?? "unknown"})`;
                    option.selected = index === 0;
                    select.appendChild(option);
                });
            } catch (error) {
                const option = document.createElement("option");
                option.value = config.fallbackModel.value;
                option.textContent = config.fallbackModel.label;
                option.selected = true;
                select.appendChild(option);
                console.warn("Failed to load models:", error);
            }
        }

        async function defaults() {
            const catalog = window.WorkflowCatalog?.load
                ? await window.WorkflowCatalog.load(API_BASE)
                : null;
            return catalog?.tasks?.[config.taskType]?.input_defaults ?? {};
        }

        function withLora(inputs) {
            if (config.lora === false) {
                return inputs;
            }
            const adapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];
            const enabled = Array.isArray(adapters) && adapters.length > 0;
            if (config.loraEnvelope !== false) {
                inputs.Lora = { enabled, adapters: enabled ? adapters : [] };
            }
            if (enabled || config.alwaysSendLoraAdapters) {
                inputs.lora_adapters = enabled ? adapters : [];
            } else {
                delete inputs.lora_adapters;
            }
            return inputs;
        }

        async function run(inputs, errorMessage) {
            const token = ++activeToken;
            activeEventSource?.close();
            activeEventSource = null;
            try {
                const payload = {
                    tasks: [{ id: "t1", type: config.taskType, inputs }],
                    return: "@t1.images",
                };
                const job = await WorkflowClient.submitWorkflow(
                    API_BASE,
                    payload,
                    WorkflowClient.makeIdempotencyKey(),
                );
                if (!job?.id) {
                    throw new Error("Job submit did not return an id.");
                }
                activeEventSource = WorkflowClient.watchJob(API_BASE, job.id, {
                    isStale: () => token !== activeToken,
                    onDone: (update) => {
                        const images = update?.status === "succeeded" ? update?.result?.outputs : [];
                        gallery.setImages(Array.isArray(images) ? images : []);
                    },
                    onError: () => {
                        if (token === activeToken) {
                            gallery.setImages([]);
                        }
                    },
                });
            } catch (error) {
                console.warn(errorMessage, error);
                gallery.setImages([]);
            }
        }

        void loadModels();
        if (config.lora !== false) {
            loraReady = window.LoraPanel?.init({ apiBase: API_BASE, family: config.family }) ?? Promise.resolve();
        }
        window.PresetPanel?.init({
            apiBase: API_BASE,
            family: config.family,
            taskType: config.taskType,
            collectSettings,
            applySettings,
        });
        if (window.WorkflowCatalog?.load) {
            void window.WorkflowCatalog.load(API_BASE).then(() => {
                const bindings = Object.fromEntries(config.fields
                    .filter((field) => field.type !== "checkbox")
                    .map((field) => [field.element, field.key]));
                window.WorkflowCatalog.applyDefaultsToForm(config.taskType, bindings);
                const catalogDefaults = window.WorkflowCatalog.getTask?.(config.taskType)?.input_defaults ?? {};
                config.fields.filter((field) => field.type === "checkbox")
                    .forEach((field) => setFieldValue(field, catalogDefaults[field.key]));
            }).catch(() => {});
        }

        return { collectSettings, defaults, run, withLora };
    }

    window.GenerationPage = {
        create,
        createFormController,
        createImageJobs,
        createVideoJobs,
        runWhenDomReady,
        validateTasks,
    };
})();
