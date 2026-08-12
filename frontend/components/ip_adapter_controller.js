/** IP-Adapter controller for SD1.5 and SDXL workflow payloads. */
(function () {
    const MODEL = "h94/IP-Adapter";

    function create(family) {
        const isSdxl = family === "sdxl";
        const subfolder = isSdxl ? "sdxl_models" : "models";
        const weightName = isSdxl ? "ip-adapter_sdxl.bin" : "ip-adapter_sd15.bin";
        let ready = Promise.resolve();

        function enabled() {
            return Boolean(document.getElementById("ip_adapter_enabled")?.checked);
        }

        function imageFile() {
            return document.getElementById("ip_adapter_image")?.files?.[0] ?? null;
        }

        function maskFile() {
            return window.IpAdapterPanel?.getMaskFile?.() ?? null;
        }

        function init(options = undefined) {
            ready = Promise.resolve().then(() => {
                window.IpAdapterPanel?.init(options);
            }).catch((error) => {
                console.warn(`${family} IP-Adapter UI init failed:`, error);
            });
            return ready;
        }

        function scale(defaults) {
            return WorkflowClient.readNumberValue(
                "ip_adapter_scale",
                defaults.ip_adapter?.scale ?? 0.6
            );
        }

        function requireImage() {
            const file = imageFile();
            if (!file) {
                throw new Error("IP-Adapter enabled but no reference image is selected.");
            }
            return file;
        }

        async function upload(file, fallbackName) {
            const artifact = await WorkflowClient.uploadArtifact(
                API_BASE,
                file,
                file.name || fallbackName
            );
            if (!artifact?.artifact_id) {
                throw new Error("IP-Adapter image upload did not return an artifact id.");
            }
            return `@artifact:${artifact.artifact_id}`;
        }

        async function attachEncoded(tasks, encodeTaskType, inputs, defaults) {
            const reference = await upload(requireImage(), "ip_adapter.png");
            const adapterScale = scale(defaults);
            tasks.push({
                id: "ip_embeds",
                type: encodeTaskType,
                inputs: {
                    image: reference,
                    model: inputs.model,
                    guidance_scale: inputs.guidance_scale ?? inputs.cfg,
                    ip_adapter_model: MODEL,
                    ip_adapter_subfolder: subfolder,
                    ip_adapter_weight_name: weightName,
                    ip_adapter_scale: adapterScale,
                },
            });
            inputs.ip_adapter = {
                enabled: true,
                image_embeds: "@ip_embeds.image_embeds",
                scale: adapterScale,
                model: MODEL,
                subfolder,
                weight_name: weightName,
            };
            const mask = maskFile();
            if (!isSdxl && mask) {
                const maskArtifact = await WorkflowClient.uploadArtifact(
                    API_BASE,
                    mask,
                    mask.name || "ip_adapter_mask.png"
                );
                if (!maskArtifact?.artifact_id) {
                    throw new Error("IP-Adapter mask upload did not return an artifact id.");
                }
                inputs.ip_adapter.mask_image = `@artifact:${maskArtifact.artifact_id}`;
            }
        }

        async function attachDirect(inputs, defaults) {
            const reference = await upload(requireImage(), "ip_adapter.png");
            inputs.ip_adapter = {
                enabled: true,
                image: reference,
                scale: scale(defaults),
                model: MODEL,
                subfolder,
                weight_name: weightName,
            };
        }

        return {
            attachDirect,
            attachEncoded,
            enabled,
            imageFile,
            init,
            maskFile,
            ready: () => ready,
        };
    }

    window.IpAdapterController = { create };
})();
