/** SDXL page compositions for text, image, and inpaint tasks. */
(function () {
    const FALLBACK_MODEL = {
        value: "stable-diffusion-xl-base-1.0",
        label: "stable-diffusion-xl-base-1.0 (sdxl, diffusers)",
    };

    function createForm(taskType, fields) {
        return GenerationPage.createFormController({
            family: "sdxl",
            taskType,
            fallbackModel: FALLBACK_MODEL,
            fields,
        });
    }

    function loraContract(inputs, adapters, alwaysSendList) {
        const selected = Array.isArray(adapters) ? adapters : [];
        inputs.Lora = {
            enabled: selected.length > 0,
            adapters: selected.length > 0 ? selected : [],
        };
        if (alwaysSendList || selected.length > 0) {
            inputs.lora_adapters = selected;
        }
    }

    function baseTextInputs(settings) {
        return {
            prompt: settings.prompt,
            negative_prompt: settings.negative_prompt,
            steps: settings.steps,
            guidance_scale: settings.guidance_scale,
            scheduler: settings.scheduler,
            seed: settings.seed,
            width: settings.width,
            height: settings.height,
            model: settings.model,
            num_images: settings.num_images,
            clip_skip: settings.clip_skip,
        };
    }

    function baseImg2ImgInputs(settings) {
        return {
            ...baseTextInputs(settings),
            strength: settings.strength,
        };
    }

    function baseInpaintInputs(settings) {
        return {
            prompt: settings.prompt,
            negative_prompt: settings.negative_prompt,
            steps: settings.steps,
            guidance_scale: settings.guidance_scale,
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
        console.warn(message, error);
        jobs.clear();
    }

    function initCore({ form, controlNet, ipAdapter, adapter, applyPreset, catalogTasks }) {
        adapter.init();
        void form.loadModels();
        controlNet.init();
        ipAdapter.init();
        form.initLora();
        form.initPresets(form.collectSettings, applyPreset);
        for (const taskType of catalogTasks) {
            void form.applyCatalogDefaults(taskType);
        }
    }

    function createText2Img({ taskText2Img, taskControlNet, taskIpAdapter, fields }) {
        const form = createForm(taskText2Img, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sdxl");
        const adapter = AdapterController.create({ subtitle: "SDXL adapter stack" });
        let jobs = null;

        async function applyPreset(settings) {
            await Promise.all([controlNet.ready(), ipAdapter.ready()]);
            await form.applySettings(settings);
            controlNet.reset();
        }

        function updateButton(status) {
            const button = document.getElementById("generate_button");
            if (!button) return;
            const busy = ["submitting", "queued", "running"].includes(status);
            button.disabled = busy;
            button.textContent = busy ? "Generating..." : "Generate";
        }

        async function generate() {
            try {
                await Promise.all([controlNet.ready(), form.ready(), ipAdapter.ready()]);
                const controlEnabled = controlNet.enabled();
                const primaryTask = controlEnabled ? taskControlNet : taskText2Img;
                const defaults = await form.defaults(primaryTask);
                const settings = form.collectSettings(defaults);
                const useIpAdapter = Boolean(settings.ip_adapter_enabled);
                if (useIpAdapter && controlEnabled) {
                    throw new Error("SDXL IP-Adapter cannot be combined with ControlNet yet.");
                }

                const inputs = baseTextInputs(settings);
                loraContract(inputs, settings.lora_adapters, false);
                const hiresUiPresent = Boolean(
                    document.getElementById("hires_enabled") && document.getElementById("hires_scale")
                );
                const hiresEnabled = hiresUiPresent && Boolean(settings.hires_enabled);
                const hiresScale = hiresUiPresent ? settings.hires_scale : 1.0;
                if (hiresEnabled && hiresScale > 1.0) {
                    inputs.hires = { enabled: true, hires_scale: hiresScale };
                }

                const tasks = [];
                if (controlEnabled) {
                    await controlNet.attachSdxlText(inputs, defaults);
                } else {
                    if (hiresUiPresent) {
                        inputs.hires_enabled = hiresEnabled;
                        inputs.hires_scale = hiresScale;
                    }
                    if (useIpAdapter) {
                        await ipAdapter.attachEncoded(tasks, taskIpAdapter, inputs, defaults);
                    }
                }
                tasks.push({ id: "t1", type: primaryTask, inputs });
                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: "@t1.images" },
                    "Failed to generate SDXL images:",
                    {
                        onDone: (job) => controlNet.showWarnings(job, "t1"),
                        onStateChange: (status) => updateButton(status),
                    }
                );
            } catch (error) {
                updateButton("failed");
                reportFailure(jobs, "Failed to generate SDXL images:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: [taskText2Img, taskControlNet],
            });
        }

        return { generate, init };
    }

    function createImg2Img({ taskType, fields }) {
        const form = createForm(taskType, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sdxl");
        const adapter = AdapterController.create({ subtitle: "SDXL adapter stack" });
        let jobs = null;

        async function applyPreset(settings) {
            await Promise.all([controlNet.ready(), ipAdapter.ready()]);
            await form.applySettings(settings);
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
                if (useIpAdapter && controlEnabled) {
                    throw new Error("SDXL img2img IP-Adapter cannot be combined with ControlNet yet.");
                }

                const inputs = baseImg2ImgInputs(settings);
                loraContract(inputs, settings.lora_adapters, true);
                controlNet.disabledEnvelope(inputs, defaults);
                const initial = await WorkflowClient.uploadArtifact(
                    API_BASE,
                    initialFile,
                    initialFile.name || "initial.png"
                );
                inputs.initial_image = `@artifact:${initial.artifact_id}`;
                if (useIpAdapter) await ipAdapter.attachDirect(inputs, defaults);
                if (controlEnabled) await controlNet.attachSdxlImage(inputs, defaults);

                const tasks = [{ id: "t1", type: taskType, inputs }];
                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: "@t1.images" },
                    "Failed to run SDXL img2img job:",
                    { onDone: (job) => controlNet.showWarnings(job, "t1") }
                );
            } catch (error) {
                reportFailure(jobs, "Failed to run SDXL img2img job:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: [taskType],
            });
        }

        return { generate, init };
    }

    function createInpaint({ taskType, fields }) {
        const form = createForm(taskType, fields);
        const controlNet = ControlNetController.create();
        const ipAdapter = IpAdapterController.create("sdxl");
        const adapter = AdapterController.create({ subtitle: "SDXL adapter stack" });
        let editor = null;
        let jobs = null;

        async function applyPreset(settings) {
            await Promise.all([controlNet.ready(), ipAdapter.ready()]);
            await form.applySettings(settings);
            controlNet.reset();
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
                if (useIpAdapter && controlEnabled) {
                    throw new Error("SDXL inpaint IP-Adapter cannot be combined with ControlNet yet.");
                }

                const inputs = baseInpaintInputs(settings);
                loraContract(inputs, settings.lora_adapters, true);
                controlNet.disabledEnvelope(inputs, defaults);
                const [initial, mask] = await Promise.all([
                    WorkflowClient.uploadArtifact(API_BASE, initialFile, initialFile.name || "initial.png"),
                    WorkflowClient.uploadArtifact(API_BASE, maskFile, "mask.png"),
                ]);
                inputs.initial_image = `@artifact:${initial.artifact_id}`;
                inputs.mask_image = `@artifact:${mask.artifact_id}`;
                if (useIpAdapter) await ipAdapter.attachDirect(inputs, defaults);
                if (controlEnabled) await controlNet.attachSdxlImage(inputs, defaults);

                const tasks = [{ id: "t1", type: taskType, inputs }];
                await GenerationPage.validateTasks(tasks);
                await jobs.run(
                    { tasks, return: "@t1.images" },
                    "Failed to run SDXL inpaint job:",
                    { onDone: (job) => controlNet.showWarnings(job, "t1") }
                );
            } catch (error) {
                reportFailure(jobs, "Failed to run SDXL inpaint job:", error);
            }
        }

        function init() {
            jobs = GenerationPage.createImageJobs();
            editor = InpaintEditor.create();
            initCore({
                form,
                controlNet,
                ipAdapter,
                adapter,
                applyPreset,
                catalogTasks: [taskType],
            });
        }

        return { generate, init };
    }

    window.SdxlGenerationController = { createImg2Img, createInpaint, createText2Img };
})();
