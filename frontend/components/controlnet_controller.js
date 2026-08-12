/** ControlNet controller for SD1.5 and SDXL workflow payloads. */
(function () {
    const SD15_CANNY_MODEL = "lllyasviel/control_v11p_sd15_canny";
    const SDXL_CANNY_MODEL = "diffusers/controlnet-canny-sdxl-1.0";

    function readOptions(defaults) {
        return {
            scale: WorkflowClient.readNumberValue(
                "controlnet_conditioning_scale",
                defaults.controlnet_conditioning_scale ?? 1.0
            ),
            guidanceStart: WorkflowClient.readNumberValue(
                "control_guidance_start",
                defaults.control_guidance_start ?? 0.0
            ),
            guidanceEnd: WorkflowClient.readNumberValue(
                "control_guidance_end",
                defaults.control_guidance_end ?? 1.0
            ),
            guessMode: Boolean(document.getElementById("controlnet_guess_mode")?.checked),
            compatMode: WorkflowClient.readTextValue(
                "controlnet_compat_mode",
                defaults.controlnet_compat_mode ?? "warn"
            ),
        };
    }

    function resolveSdxlModel(modelId) {
        const normalized = String(modelId || "").trim();
        return !normalized || normalized.includes("_sd15") ? SDXL_CANNY_MODEL : normalized;
    }

    function create() {
        let ready = Promise.resolve();

        function enabled() {
            return Boolean(document.getElementById("controlnet-enabled")?.checked);
        }

        function state() {
            return window.ControlNetPanel?.getState?.() ?? null;
        }

        function init({ perItemGuidanceTiming = false } = {}) {
            if (perItemGuidanceTiming) {
                window.ControlNetPanel?.setPerItemGuidanceTimingEnabled?.(true);
            }
            if (window.ControlNetPreprocessor?.init) {
                ready = window.ControlNetPreprocessor.init().catch((error) => {
                    console.warn("ControlNet init failed:", error);
                });
            }
            return ready;
        }

        function reset() {
            window.ControlNetPanel?.clearControlItems?.();
            window.ControlNetPanel?.updateIndicator?.();
            window.ControlNetPanel?.updateActiveFlag?.();
        }

        async function uploadItems(family, options) {
            const currentState = state();
            const controlItems = Array.isArray(currentState?.controlItems) ? currentState.controlItems : [];
            if (controlItems.length === 0 && !currentState?.previewBlob) {
                throw new Error("ControlNet enabled but no preprocessor output image is ready.");
            }
            const defaultModel = family === "sdxl" ? SDXL_CANNY_MODEL : SD15_CANNY_MODEL;
            const items = controlItems.length > 0 ? controlItems : [{
                previewBlob: currentState.previewBlob,
                preprocessorId: currentState.preprocessorId ?? null,
                modelId: defaultModel,
                conditioningScale: options.scale,
                guidanceStart: options.guidanceStart,
                guidanceEnd: options.guidanceEnd,
            }];
            const uploads = await Promise.all(items.map((item, index) =>
                WorkflowClient.uploadArtifact(API_BASE, item.previewBlob, `controlnet_${index + 1}.png`)
            ));
            const images = uploads.map((upload) => `@artifact:${upload.artifact_id}`);
            const models = items.map((item) => family === "sdxl"
                ? resolveSdxlModel(item.modelId)
                : item.modelId || SD15_CANNY_MODEL);
            const scales = items.map((item) => {
                const value = Number(item.conditioningScale);
                return Number.isFinite(value) ? value : options.scale;
            });
            const guidanceStarts = items.map((item) => {
                const value = Number(item.guidanceStart);
                return Number.isFinite(value) ? value : options.guidanceStart;
            });
            const guidanceEnds = items.map((item) => {
                const value = Number(item.guidanceEnd);
                return Number.isFinite(value) ? value : options.guidanceEnd;
            });
            const preprocessorIds = items.map((item) => item.preprocessorId || null);
            return {
                guidanceEnds,
                guidanceStarts,
                hasAllPreprocessorIds: preprocessorIds.every(
                    (value) => typeof value === "string" && value.length > 0
                ),
                images,
                items,
                models,
                preprocessorIds,
                scales,
            };
        }

        function envelope(options, controls) {
            return {
                enabled: true,
                controlnetConditioningScale: options.scale,
                controlGuidanceStart: options.guidanceStart,
                controlGuidanceEnd: options.guidanceEnd,
                controlnetGuessMode: options.guessMode,
                controlnetPreprocessors: controls.images.map((image, index) => ({
                    control_image: image,
                    model_id: controls.models[index],
                    conditioning_scale: controls.scales[index],
                    preprocessor_id: controls.preprocessorIds[index],
                })),
            };
        }

        function disabledEnvelope(inputs, defaults) {
            const options = readOptions(defaults);
            inputs.Controlnet = {
                enabled: false,
                controlnetConditioningScale: options.scale,
                controlGuidanceStart: options.guidanceStart,
                controlGuidanceEnd: options.guidanceEnd,
                controlnetGuessMode: options.guessMode,
                controlnetPreprocessors: [],
            };
        }

        async function attachSd15Text(inputs, defaults) {
            const options = readOptions(defaults);
            const controls = await uploadItems("sd15", options);
            inputs.controlNetEnabled = true;
            inputs.controlnet_conditioning_scale = options.scale;
            inputs.control_guidance_start = options.guidanceStart;
            inputs.control_guidance_end = options.guidanceEnd;
            inputs.controlnet_guess_mode = options.guessMode;
            inputs.controlnet_compat_mode = options.compatMode;
            inputs.effectiveItems = controls.images.map((image, index) => ({
                control_image: image,
                model_id: controls.models[index],
                conditioning_scale: controls.scales[index],
                guidance_start: controls.guidanceStarts[index],
                guidance_end: controls.guidanceEnds[index],
                preprocessor_id: controls.preprocessorIds[index],
            }));
            inputs.Controlnet = {
                ...envelope(options, controls),
                controlnetPreprocessors: inputs.effectiveItems.map((item) => ({ ...item })),
            };
            inputs.control_image = controls.images[0];
            if (controls.items.length > 1) {
                inputs.control_images = controls.images;
                inputs.controlnet_models = controls.models;
                inputs.controlnet_conditioning_scales = controls.scales;
                inputs.control_guidance_starts = controls.guidanceStarts;
                inputs.control_guidance_ends = controls.guidanceEnds;
                if (controls.hasAllPreprocessorIds) {
                    inputs.controlnet_preprocessor_ids = controls.preprocessorIds;
                }
            } else {
                inputs.controlnet_model = controls.models[0];
                inputs.controlnet_conditioning_scale = controls.scales[0];
                inputs.control_guidance_start = controls.guidanceStarts[0];
                inputs.control_guidance_end = controls.guidanceEnds[0];
                if (controls.hasAllPreprocessorIds) {
                    inputs.controlnet_preprocessor_id = controls.preprocessorIds[0];
                }
            }
        }

        async function attachSd15Image(inputs, defaults, inpaintCondition = null) {
            const options = readOptions(defaults);
            if (inpaintCondition) {
                const currentState = state();
                const items = Array.isArray(currentState?.controlItems) ? currentState.controlItems : [];
                if (items.length > 0 || currentState?.previewBlob) {
                    throw new Error("SD1.5 inpaint ControlNet condition cannot be combined with preprocessor control images yet.");
                }
                inputs.Controlnet = {
                    enabled: true,
                    controlnetConditioningScale: options.scale,
                    controlGuidanceStart: options.guidanceStart,
                    controlGuidanceEnd: options.guidanceEnd,
                    controlnetGuessMode: options.guessMode,
                    controlnetPreprocessors: [{
                        model_id: inpaintCondition.model,
                        conditioning_scale: options.scale,
                        preprocessor_id: inpaintCondition.preprocessorId,
                    }],
                };
                inputs.controlnet_model = inpaintCondition.model;
                inputs.controlnet_preprocessor_id = inpaintCondition.preprocessorId;
                inputs.controlnet_conditioning_scale = options.scale;
                inputs.controlnet_guess_mode = options.guessMode;
                inputs.control_guidance_start = options.guidanceStart;
                inputs.control_guidance_end = options.guidanceEnd;
                inputs.controlnet_compat_mode = options.compatMode;
                return;
            }
            const controls = await uploadItems("sd15", options);
            inputs.Controlnet = envelope(options, controls);
            inputs.control_image = controls.images[0];
            inputs.controlnet_model = controls.models[0];
            inputs.controlnet_conditioning_scale = controls.scales[0];
            inputs.controlnet_guess_mode = options.guessMode;
            inputs.control_guidance_start = options.guidanceStart;
            inputs.control_guidance_end = options.guidanceEnd;
            inputs.controlnet_compat_mode = options.compatMode;
            if (controls.hasAllPreprocessorIds) {
                inputs.controlnet_preprocessor_id = controls.preprocessorIds[0];
            }
            if (controls.items.length > 1) {
                inputs.control_images = controls.images.slice(1);
                inputs.controlnet_models = controls.models;
                inputs.controlnet_conditioning_scales = controls.scales;
                if (controls.hasAllPreprocessorIds) {
                    inputs.controlnet_preprocessor_ids = controls.preprocessorIds;
                }
            }
        }

        async function attachSdxl(inputs, defaults, textMode) {
            const options = readOptions(defaults);
            const controls = await uploadItems("sdxl", options);
            inputs.Controlnet = envelope(options, controls);
            inputs.control_image = controls.images[0];
            inputs.controlnet_conditioning_scale = textMode ? options.scale : controls.scales[0];
            inputs.controlnet_guess_mode = options.guessMode;
            inputs.control_guidance_start = options.guidanceStart;
            inputs.control_guidance_end = options.guidanceEnd;
            inputs.controlnet_compat_mode = options.compatMode;
            if (controls.items.length > 1) {
                inputs.control_images = textMode ? controls.images : controls.images.slice(1);
                inputs.controlnet_models = controls.models;
                inputs.controlnet_conditioning_scales = controls.scales;
                if (controls.hasAllPreprocessorIds) {
                    inputs.controlnet_preprocessor_ids = controls.preprocessorIds;
                }
            } else {
                inputs.controlnet_model = controls.models[0];
                inputs.controlnet_conditioning_scale = controls.scales[0];
                if (controls.hasAllPreprocessorIds) {
                    inputs.controlnet_preprocessor_id = controls.preprocessorIds[0];
                }
            }
        }

        function showWarnings(job, taskId) {
            const warnings = job?.result?.tasks?.[taskId]?.warnings;
            if (!Array.isArray(warnings) || warnings.length === 0) return;
            console.warn("ControlNet warnings:", warnings);
            const status = document.getElementById("controlnet-status");
            if (status) status.textContent = warnings.join(" ");
        }

        return {
            attachSd15Image,
            attachSd15Text,
            attachSdxlImage: (inputs, defaults) => attachSdxl(inputs, defaults, false),
            attachSdxlText: (inputs, defaults) => attachSdxl(inputs, defaults, true),
            disabledEnvelope,
            enabled,
            init,
            ready: () => ready,
            reset,
            showWarnings,
            state,
        };
    }

    window.ControlNetController = { create };
})();
