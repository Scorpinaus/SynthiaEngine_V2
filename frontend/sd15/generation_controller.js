/** SD1.5 page compositions for text, image, and inpaint tasks. */
(function () {
    const FALLBACK_MODEL = {
        value: "stable-diffusion-v1-5",
        label: "stable-diffusion-v1-5 (sd15, diffusers)",
    };
    const LCM_SCHEDULER = "lcm";
    const INPAINT_CONTROLNET = {
        model: "lllyasviel/control_v11p_sd15_inpaint",
        preprocessorId: "inpaint-condition",
    };

    function createForm(taskType, fields) {
        return GenerationPage.createFormController({
            family: "sd15",
            taskType,
            fallbackModel: FALLBACK_MODEL,
            fields,
        });
    }

    function loraContract(inputs, adapters) {
        const selected = Array.isArray(adapters) ? adapters : [];
        inputs.lora = {
            lora_enabled: selected.length > 0,
            lora_adapters: selected.length > 0 ? selected : [],
        };
    }

    function lcmEnabled() {
        return Boolean(document.getElementById("lcm_enabled")?.checked);
    }

    function syncLcmDefaults() {
        if (!lcmEnabled()) {
            const scheduler = document.getElementById("scheduler");
            if (scheduler?.value === LCM_SCHEDULER) scheduler.value = "euler";
            return;
        }
        document.getElementById("steps").value = "4";
        document.getElementById("cfg").value = "0";
        document.getElementById("scheduler").value = LCM_SCHEDULER;
    }

    function applyLcm(inputs) {
        inputs.lcm = { enabled: true };
        inputs.scheduler = LCM_SCHEDULER;
        if (inputs.steps < 1 || inputs.steps > 8) {
            throw new Error("LCM mode requires steps between 1 and 8.");
        }
        if (inputs.cfg < 0 || inputs.cfg > 2) {
            throw new Error("LCM mode requires CFG between 0 and 2.");
        }
    }

    function baseTextInputs(settings) {
        return {
            prompt: settings.prompt,
            negative_prompt: settings.negative_prompt,
            steps: settings.steps,
            cfg: settings.cfg,
            scheduler: settings.scheduler,
            seed: settings.seed,
            width: settings.width,
            height: settings.height,
            model: settings.model,
            clip_skip: settings.clip_skip,
            num_images: settings.num_images,
            weighting_policy: settings.weighting_policy,
        };
    }

    function baseImg2ImgInputs(settings) {
        return {
            prompt: settings.prompt,
            negative_prompt: settings.negative_prompt,
            steps: settings.steps,
            cfg: settings.cfg,
            scheduler: settings.scheduler,
            seed: settings.seed,
            width: settings.width,
            height: settings.height,
            strength: settings.strength,
            num_images: settings.num_images,
            model: settings.model,
            clip_skip: settings.clip_skip,
        };
    }

    function baseInpaintInputs(settings) {
        return {
            prompt: settings.prompt,
            negative_prompt: settings.negative_prompt,
            steps: settings.steps,
            cfg: settings.cfg,
            scheduler: settings.scheduler,
            seed: settings.seed,
            num_images: settings.num_images,
            model: settings.model,
            strength: settings.strength,
            padding_mask_crop: settings.padding_mask_crop,
            clip_skip: settings.clip_skip,
        };
    }

    function reportFailure(jobs, message, error) {
        if (error instanceof Error && error.message.startsWith("Input validation failed for ")) {
            alert(error.message);
        }
        console.warn(message, error);
        jobs.clear();
    }

    function initCore({ form, controlNet, ipAdapter, adapter, applyPreset, catalogTasks }) {
        adapter.init();
        void form.loadModels();
        controlNet.init({ perItemGuidanceTiming: catalogTasks.perItemGuidanceTiming });
        form.initLora();
        form.initPresets(form.collectSettings, applyPreset);
        for (const taskType of catalogTasks.types) {
            void form.applyCatalogDefaults(taskType);
        }
        document.getElementById("lcm_enabled")?.addEventListener("change", syncLcmDefaults);
        return ipAdapter;
    }

    function createText2Img({
        taskText2Img,
        taskControlNet,
        taskHires,
        taskIpAdapter,
        fields,
    }) {
        const form = createForm(taskText2Img, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sd15");
        const adapter = AdapterController.create();
        let jobs = null;

        async function applyPreset(settings) {
            await form.applySettings(settings);
            if (settings.lcm_enabled) syncLcmDefaults();
            controlNet.reset();
        }

        async function generate() {
            try {
                await Promise.all([controlNet.ready(), form.ready(), ipAdapter.ready()]);
                const controlEnabled = controlNet.enabled();
                const primaryTask = controlEnabled ? taskControlNet : taskText2Img;
                const defaults = await form.defaults(primaryTask);
                const hiresDefaults = await form.defaults(taskHires);
                const settings = form.collectSettings(defaults);
                const hiresScale = WorkflowClient.readNumberValue(
                    "hires_scale",
                    hiresDefaults.hires_scale ?? 1.0
                );
                const hiresEnabled = Boolean(settings.hires_enabled) && hiresScale > 1.0;
                const useLcm = Boolean(settings.lcm_enabled);
                const useIpAdapter = Boolean(settings.ip_adapter_enabled);

                if (useLcm && controlEnabled) {
                    throw new Error("LCM mode is currently available for SD1.5 text-to-image only.");
                }
                if (useLcm && hiresEnabled) {
                    throw new Error("LCM mode cannot be combined with Hi-Res Fix yet.");
                }
                if (useIpAdapter && controlEnabled) {
                    throw new Error("IP-Adapter is currently available for SD1.5 text-to-image only.");
                }
                if (useIpAdapter && useLcm) {
                    throw new Error("IP-Adapter cannot be combined with LCM mode yet.");
                }
                if (useIpAdapter && hiresEnabled) {
                    throw new Error("IP-Adapter cannot be combined with Hi-Res Fix yet.");
                }

                const inputs = baseTextInputs(settings);
                if (useLcm) applyLcm(inputs);
                loraContract(inputs, settings.lora_adapters);
                inputs.hires = { enabled: hiresEnabled, hires_scale: hiresScale };

                const tasks = [];
                if (useIpAdapter) {
                    await ipAdapter.attachEncoded(tasks, taskIpAdapter, inputs, defaults);
                }
                if (controlEnabled) {
                    await controlNet.attachSd15Text(inputs, defaults);
                }
                tasks.push({ id: "t1", type: primaryTask, inputs });

                let returnRef = "@t1.images";
                if (hiresEnabled) {
                    const hiresInputs = {
                        images: "@t1.images",
                        prompt: inputs.prompt,
                        negative_prompt: inputs.negative_prompt,
                        steps: inputs.steps,
                        cfg: inputs.cfg,
                        scheduler: inputs.scheduler,
                        seed: inputs.seed,
                        model: inputs.model,
                        clip_skip: inputs.clip_skip,
                        hires_scale: hiresScale,
                        weighting_policy: inputs.weighting_policy,
                        hires: { enabled: true, hires_scale: hiresScale },
                    };
                    loraContract(hiresInputs, settings.lora_adapters);
                    tasks.push({ id: "hires", type: taskHires, inputs: hiresInputs });
                    returnRef = "@hires.images";
                }

                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: returnRef },
                    "Failed to generate SD1.5 images:",
                    { onDone: (job) => controlNet.showWarnings(job, "t1") }
                );
            } catch (error) {
                reportFailure(jobs, "Failed to generate SD1.5 images:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            ipAdapter.init({
                getMaskSize: () => ({
                    width: WorkflowClient.readNumberValue("width", 512, { integer: true }),
                    height: WorkflowClient.readNumberValue("height", 512, { integer: true }),
                }),
            });
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: {
                    perItemGuidanceTiming: true,
                    types: [taskText2Img, taskControlNet, taskHires],
                },
            });
            document.getElementById("generate-button")?.addEventListener("click", generate);
        }

        return { generate, init };
    }

    function createImg2Img({ taskType, taskIpAdapter, fields }) {
        const form = createForm(taskType, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sd15");
        const adapter = AdapterController.create();
        let jobs = null;

        async function applyPreset(settings) {
            await form.applySettings(settings);
            if (settings.lcm_enabled) syncLcmDefaults();
            controlNet.reset();
        }

        async function generate() {
            const initialFile = document.getElementById("initial_image")?.files?.[0] ?? null;
            if (!initialFile) {
                alert("Please select an initial image.");
                return;
            }
            try {
                await Promise.all([controlNet.ready(), form.ready(), ipAdapter.ready()]);
                const defaults = await form.defaults();
                const settings = form.collectSettings(defaults);
                const controlEnabled = Boolean(settings.controlnet_enabled);
                const useIpAdapter = Boolean(settings.ip_adapter_enabled);
                const useLcm = Boolean(settings.lcm_enabled) || settings.scheduler === LCM_SCHEDULER;
                if (useLcm && controlEnabled) {
                    throw new Error("LCM mode cannot be combined with ControlNet for SD1.5 img2img yet.");
                }
                if (useIpAdapter && controlEnabled) {
                    throw new Error("IP-Adapter cannot be combined with ControlNet for SD1.5 img2img yet.");
                }
                if (useIpAdapter && useLcm) {
                    throw new Error("IP-Adapter cannot be combined with LCM mode for SD1.5 img2img yet.");
                }

                const inputs = baseImg2ImgInputs(settings);
                if (useLcm) applyLcm(inputs);
                const initial = await WorkflowClient.uploadArtifact(
                    API_BASE,
                    initialFile,
                    initialFile.name || "initial.png"
                );
                inputs.initial_image = `@artifact:${initial.artifact_id}`;
                loraContract(inputs, settings.lora_adapters);

                const tasks = [];
                if (useIpAdapter) {
                    await ipAdapter.attachEncoded(tasks, taskIpAdapter, inputs, defaults);
                }
                if (controlEnabled) {
                    await controlNet.attachSd15Image(inputs, defaults);
                }
                tasks.push({ id: "img2img", type: taskType, inputs });
                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: "@img2img.images" },
                    "Failed to run img2img job:",
                    { onDone: (job) => controlNet.showWarnings(job, "img2img") }
                );
            } catch (error) {
                reportFailure(jobs, "Failed to run img2img job:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            ipAdapter.init({
                getMaskBackdropFile: () => document.getElementById("initial_image")?.files?.[0] ?? null,
            });
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: { perItemGuidanceTiming: false, types: [taskType] },
            });
        }

        return { generate, init };
    }

    function installInpaintCondition(adapter) {
        const settings = document.querySelector("#controlnet-content .controlnet-settings");
        if (!settings || document.getElementById("controlnet_inpaint_condition")) return;
        const row = document.createElement("label");
        row.className = "field inline-field";
        row.innerHTML = `
            <input id="controlnet_inpaint_condition" type="checkbox" />
            <span>Use SD1.5 inpaint ControlNet condition</span>
        `;
        const hint = document.createElement("div");
        hint.className = "field-hint";
        hint.textContent = "Uses the current initial image and mask; no separate preprocessor image is needed.";
        settings.insertBefore(hint, settings.firstChild);
        settings.insertBefore(row, hint);
        row.querySelector("input")?.addEventListener("change", (event) => {
            if (event.target?.checked) {
                const toggle = document.getElementById("controlnet-enabled");
                if (toggle) toggle.checked = true;
                const status = document.getElementById("controlnet-status");
                if (status) status.textContent = "Inpaint ControlNet condition ready.";
            }
            window.ControlNetPanel?.updateIndicator?.();
            window.ControlNetPanel?.updateActiveFlag?.();
            adapter.update();
        });
    }

    function createInpaint({ taskType, taskIpAdapter, fields }) {
        const form = createForm(taskType, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sd15");
        const adapter = AdapterController.create({
            adjustControlSummary: (summary) => {
                summary.availablePreprocessors = Number(
                    summary.availablePreprocessors ?? summary.totalPreprocessors ?? 0
                ) + 1;
                summary.totalPreprocessors = Number(summary.totalPreprocessors ?? 0) + 1;
                if (document.getElementById("controlnet_inpaint_condition")?.checked) {
                    summary.activeItems = Number(summary.activeItems ?? 0) + 1;
                    summary.enabled = true;
                }
                return summary;
            },
        });
        let editor = null;
        let jobs = null;

        async function applyPreset(settings) {
            await form.applySettings(settings);
            if (settings.controlnet_inpaint_condition) {
                const toggle = document.getElementById("controlnet-enabled");
                if (toggle) toggle.checked = true;
                const status = document.getElementById("controlnet-status");
                if (status) status.textContent = "Inpaint ControlNet condition ready.";
            }
            if (settings.lcm_enabled) syncLcmDefaults();
            controlNet.reset();
            adapter.update();
        }

        async function generate() {
            const initialFile = editor.getBaseImageFile();
            const maskFile = editor.getActiveMaskBlob();
            if (!initialFile) {
                alert("Please upload an initial image.");
                return;
            }
            if (!maskFile) {
                alert("Please create and save a mask before generating.");
                return;
            }
            try {
                await Promise.all([controlNet.ready(), form.ready(), ipAdapter.ready()]);
                const defaults = await form.defaults();
                const settings = form.collectSettings(defaults);
                const controlEnabled = Boolean(settings.controlnet_enabled);
                const useIpAdapter = Boolean(settings.ip_adapter_enabled);
                const useLcm = Boolean(settings.lcm_enabled) || settings.scheduler === LCM_SCHEDULER;
                if (useLcm && controlEnabled) {
                    throw new Error("LCM mode cannot be combined with ControlNet for SD1.5 inpaint yet.");
                }
                if (useIpAdapter && controlEnabled) {
                    throw new Error("IP-Adapter cannot be combined with ControlNet for SD1.5 inpaint yet.");
                }
                if (useIpAdapter && useLcm) {
                    throw new Error("IP-Adapter cannot be combined with LCM mode for SD1.5 inpaint yet.");
                }

                const inputs = baseInpaintInputs(settings);
                if (useLcm) applyLcm(inputs);
                const [initial, mask] = await Promise.all([
                    WorkflowClient.uploadArtifact(API_BASE, initialFile, initialFile.name || "initial.png"),
                    WorkflowClient.uploadArtifact(API_BASE, maskFile, "mask.png"),
                ]);
                inputs.initial_image = `@artifact:${initial.artifact_id}`;
                inputs.mask_image = `@artifact:${mask.artifact_id}`;
                loraContract(inputs, settings.lora_adapters);

                const tasks = [];
                if (useIpAdapter) {
                    await ipAdapter.attachEncoded(tasks, taskIpAdapter, inputs, defaults);
                }
                if (controlEnabled) {
                    const condition = settings.controlnet_inpaint_condition ? INPAINT_CONTROLNET : null;
                    await controlNet.attachSd15Image(inputs, defaults, condition);
                }
                tasks.push({ id: "inpaint", type: taskType, inputs });
                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: "@inpaint.images" },
                    "Failed to run inpaint job:",
                    { onDone: (job) => controlNet.showWarnings(job, "inpaint") }
                );
            } catch (error) {
                reportFailure(jobs, "Failed to run inpaint job:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            editor = InpaintEditor.create();
            ipAdapter.init({ getMaskBackdropFile: () => editor.getBaseImageFile() });
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: { perItemGuidanceTiming: false, types: [taskType] },
            });
            void controlNet.ready().then(() => installInpaintCondition(adapter));
        }

        return { generate, init };
    }

    window.Sd15GenerationController = { createImg2Img, createInpaint, createText2Img };
})();
