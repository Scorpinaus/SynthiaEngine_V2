(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    let controlNetUiReady = false;
    let controlNetUiLoading = null;
    let layoutResizeBound = false;

    function getPanelApi() {
        return window.ControlNetPanel;
    }

    function getState() {
        return getPanelApi()?.getState?.();
    }

    function updateDownloadLinkState(isReady) {
        const downloadLink = document.getElementById("download-control-image");
        if (!downloadLink) {
            return;
        }
        downloadLink.setAttribute("aria-disabled", isReady ? "false" : "true");
        downloadLink.classList.toggle("is-disabled", !isReady);
        if (!isReady) {
            downloadLink.href = "#";
        }
    }

    function applyPreprocessorLayoutStyles() {
        const modal = document.getElementById("preprocessor-modal");
        const body = modal?.querySelector(".modal-body");
        const settings = modal?.querySelector(".preprocessor-settings");
        const previewPanel = modal?.querySelector(".preprocessor-preview");
        const previewImage = document.getElementById("preprocessor-preview");
        const content = modal?.querySelector(".modal-content");
        if (!modal || !body || !settings || !previewPanel || !content) {
            return;
        }

        content.style.width = "min(94vw, 1100px)";
        content.style.maxHeight = "94vh";
        body.style.display = "grid";
        body.style.gap = "16px";
        body.style.alignItems = "start";
        settings.style.display = "grid";
        settings.style.gap = "12px";
        settings.style.alignContent = "start";

        if (window.innerWidth <= 700) {
            body.style.gridTemplateColumns = "1fr";
        } else {
            body.style.gridTemplateColumns = "minmax(280px, 360px) minmax(0, 1fr)";
        }

        if (previewImage) {
            previewImage.style.maxHeight = window.innerWidth <= 700 ? "55vh" : "calc(94vh - 220px)";
            previewImage.style.minHeight = "240px";
        }
    }

    function ensurePreprocessorLayoutStructure() {
        const modal = document.getElementById("preprocessor-modal");
        const body = modal?.querySelector(".modal-body");
        if (!modal || !body) {
            return;
        }

        body.classList.add("preprocessor-layout");

        const previewPanel =
            body.querySelector(".preprocessor-preview-panel") ?? body.querySelector(".preprocessor-preview");
        if (previewPanel) {
            previewPanel.classList.add("preprocessor-preview-panel");
        }

        let settings = body.querySelector(".preprocessor-settings");
        if (!settings) {
            settings = document.createElement("div");
            settings.className = "preprocessor-settings";

            const children = Array.from(body.children);
            children.forEach((node) => {
                if (node !== previewPanel) {
                    settings.appendChild(node);
                }
            });
            if (previewPanel) {
                body.insertBefore(settings, previewPanel);
            } else {
                body.appendChild(settings);
            }
        }

        let paramsContainer = document.getElementById("preprocessor-params");
        if (!paramsContainer) {
            const staleCannyFields = document.getElementById("canny-thresholds");
            paramsContainer = document.createElement("div");
            paramsContainer.id = "preprocessor-params";
            paramsContainer.className = "preprocessor-param-list";
            const applyButton = document.getElementById("apply-preprocessor");
            if (staleCannyFields) {
                staleCannyFields.replaceWith(paramsContainer);
            } else if (applyButton) {
                applyButton.parentElement?.insertBefore(paramsContainer, applyButton);
            } else {
                settings.appendChild(paramsContainer);
            }
        }

        applyPreprocessorLayoutStyles();
        if (!layoutResizeBound) {
            window.addEventListener("resize", applyPreprocessorLayoutStyles);
            layoutResizeBound = true;
        }
    }

    async function loadControlNetModal() {
        const container = document.getElementById("controlnet-preprocessor-root");
        if (!container) {
            return;
        }
        try {
            const res = await fetch(resolveAssetUrl("controlnet_preprocessor.html?v=2"), { cache: "no-store" });
            if (!res.ok) {
                throw new Error(`Failed to load ControlNet preprocessor UI: ${res.status}`);
            }
            container.innerHTML = await res.text();
            ensurePreprocessorLayoutStructure();
        } catch (error) {
            console.warn("Failed to load ControlNet preprocessor UI:", error);
        }
    }

    async function loadPreprocessors() {
        const select = document.getElementById("preprocessor-select");
        const state = getState();
        if (!select || !state) {
            return;
        }
        select.innerHTML = "";
        try {
            const res = await fetch(`${API_BASE}/api/controlnet/preprocessors`);
            const preprocessors = await res.json();
            preprocessors.forEach((preprocessor) => {
                const option = document.createElement("option");
                option.value = preprocessor.id;
                option.textContent = preprocessor.name;
                select.appendChild(option);
                state.preprocessors.set(preprocessor.id, preprocessor);
            });
            updatePreprocessorDefaults(select.value);
        } catch (error) {
            const fallback = document.createElement("option");
            fallback.value = "canny";
            fallback.textContent = "Canny";
            select.appendChild(fallback);
            console.warn("Failed to load preprocessors:", error);
        }
    }

    function formatParamLabel(key) {
        return key
            .split("_")
            .filter(Boolean)
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");
    }

    function createParamInput(key, spec, value) {
        const input = document.createElement("input");
        input.id = `preprocessor-param-${key}`;
        input.dataset.preprocessorParam = key;
        input.dataset.paramType = spec?.type || "str";

        if (spec?.type === "bool") {
            input.type = "checkbox";
            input.checked = Boolean(value);
            return input;
        }

        if (spec?.type === "int" || spec?.type === "float") {
            input.type = "number";
            input.step = spec.type === "int" ? "1" : "0.01";
            if (spec.minimum !== null && spec.minimum !== undefined) {
                input.min = String(spec.minimum);
            }
            if (spec.maximum !== null && spec.maximum !== undefined) {
                input.max = String(spec.maximum);
            }
            input.value = Number(value ?? 0);
            return input;
        }

        input.type = "text";
        input.value = value ?? "";
        return input;
    }

    function renderPreprocessorParams(preprocessorId) {
        const state = getState();
        const definition = state?.preprocessors.get(preprocessorId);
        const schema = definition?.param_schema ?? {};
        const defaults = definition?.defaults ?? {};
        const paramsContainer = document.getElementById("preprocessor-params");
        if (!paramsContainer) {
            return;
        }
        paramsContainer.innerHTML = "";

        Object.entries(schema).forEach(([key, spec]) => {
            const label = document.createElement("label");
            const input = createParamInput(key, spec, defaults[key]);
            const labelText = document.createElement("span");
            labelText.textContent = formatParamLabel(key);

            if (spec?.type === "bool") {
                label.className = "field inline-field";
                label.append(input, labelText);
            } else {
                label.className = "field";
                label.append(labelText, input);
            }
            paramsContainer.appendChild(label);
        });
    }

    function updatePreprocessorDefaults(preprocessorId) {
        const state = getState();
        const definition = state?.preprocessors.get(preprocessorId);
        const description = definition?.description ?? "";
        const descriptionNode = document.getElementById("preprocessor-description");
        if (descriptionNode) {
            descriptionNode.textContent = description;
        }
        renderPreprocessorParams(preprocessorId);
    }

    function buildPreprocessorParams(preprocessorId) {
        const state = getState();
        const definition = state?.preprocessors.get(preprocessorId);
        const schema = definition?.param_schema ?? {};
        const params = { ...(definition?.defaults ?? {}) };

        Object.entries(schema).forEach(([key, spec]) => {
            const input = document.querySelector(`[data-preprocessor-param="${key}"]`);
            if (!(input instanceof HTMLInputElement)) {
                return;
            }
            if (spec?.type === "bool") {
                params[key] = Boolean(input.checked);
                return;
            }
            if (spec?.type === "int") {
                params[key] = parseInt(input.value, 10);
                return;
            }
            if (spec?.type === "float") {
                params[key] = Number(input.value);
                return;
            }
            params[key] = input.value;
        });
        return params;
    }

    async function openPreprocessorModal() {
        await ensureControlNetUI();
        const modal = document.getElementById("preprocessor-modal");
        if (!modal) {
            return;
        }
        modal.classList.remove("hidden");
        modal.setAttribute("aria-hidden", "false");
    }

    function closePreprocessorModal() {
        const modal = document.getElementById("preprocessor-modal");
        if (!modal) {
            return;
        }
        modal.classList.add("hidden");
        modal.setAttribute("aria-hidden", "true");
    }

    async function applyPreprocessor() {
        const panelApi = getPanelApi();
        const state = getState();
        const fileInput = document.getElementById("preprocessor-image");
        const select = document.getElementById("preprocessor-select");
        const preview = document.getElementById("preprocessor-preview");
        const downloadLink = document.getElementById("download-control-image");
        const enabledToggle = document.getElementById("controlnet-enabled");

        if (!panelApi || !state) {
            console.warn("ControlNet panel API not available.");
            return;
        }
        if (!fileInput?.files?.length) {
            alert("Please select an input image for the preprocessor.");
            return;
        }

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);
        const selectedId = select?.value ?? "canny";
        formData.append("preprocessor_id", selectedId);
        formData.append("params", JSON.stringify(buildPreprocessorParams(selectedId)));

        const res = await fetch(`${API_BASE}/api/controlnet/preprocess`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            console.error("Preprocessor failed", res.status);
            alert("Preprocessor failed. Check the backend logs for details.");
            return;
        }

        const blob = await res.blob();
        const previewUrl = URL.createObjectURL(blob);
        state.previewUrl = previewUrl;
        state.previewBlob = blob;
        state.preprocessorId = selectedId;
        const defaultScaleInput = document.getElementById("controlnet_conditioning_scale");
        const defaultScale = Number(defaultScaleInput?.value ?? 1.0);
        const defaultGuidanceStartInput = document.getElementById("control_guidance_start");
        const defaultGuidanceStart = Number(defaultGuidanceStartInput?.value ?? 0.0);
        const defaultGuidanceEndInput = document.getElementById("control_guidance_end");
        const defaultGuidanceEnd = Number(defaultGuidanceEndInput?.value ?? 1.0);
        const definition = state.preprocessors.get(selectedId);
        const recommendedModel =
            definition?.recommended_sd15_control_models?.[0] ??
            definition?.legacy_aliases?.[0] ??
            "lllyasviel/control_v11p_sd15_canny";
        panelApi.addControlItem({
            previewBlob: blob,
            previewUrl,
            preprocessorId: selectedId,
            modelId: recommendedModel,
            conditioningScale: defaultScale,
            guidanceStart: defaultGuidanceStart,
            guidanceEnd: defaultGuidanceEnd,
        });
        if (preview) {
            preview.src = state.previewUrl;
        }
        if (downloadLink) {
            downloadLink.href = state.previewUrl;
            downloadLink.setAttribute("download", "controlnet_preprocessor.png");
        }
        updateDownloadLinkState(true);
        if (enabledToggle) {
            enabledToggle.checked = true;
        }
        panelApi.updateIndicator();
        panelApi.updateActiveFlag();
    }

    function initControlNetUI() {
        const panelApi = getPanelApi();
        const state = getState();
        const toggleButton = document.getElementById("controlnet-toggle");
        const openButton = document.getElementById("open-preprocessors");
        const closeButton = document.getElementById("close-preprocessors");
        const overlay = document.getElementById("preprocessor-overlay");
        const applyButton = document.getElementById("apply-preprocessor");
        const enabledToggle = document.getElementById("controlnet-enabled");
        const select = document.getElementById("preprocessor-select");
        const fileInput = document.getElementById("preprocessor-image");
        const itemsContainer = document.getElementById("controlnet-items");
        const prevButton = document.getElementById("controlnet-prev");
        const nextButton = document.getElementById("controlnet-next");

        toggleButton?.addEventListener("click", panelApi?.togglePanel);
        openButton?.addEventListener("click", openPreprocessorModal);
        closeButton?.addEventListener("click", closePreprocessorModal);
        overlay?.addEventListener("click", closePreprocessorModal);
        applyButton?.addEventListener("click", applyPreprocessor);
        prevButton?.addEventListener("click", () => {
            panelApi?.showPrevControlItem?.();
        });
        nextButton?.addEventListener("click", () => {
            panelApi?.showNextControlItem?.();
        });
        enabledToggle?.addEventListener("change", () => {
            panelApi?.updateIndicator();
            panelApi?.updateActiveFlag();
        });
        select?.addEventListener("change", (event) => {
            updatePreprocessorDefaults(event.target.value);
            updateDownloadLinkState(false);
        });
        fileInput?.addEventListener("change", () => {
            if (state) {
                state.previewBlob = null;
                state.preprocessorId = null;
            }
            updateDownloadLinkState(false);
            panelApi?.updateActiveFlag();
        });
        itemsContainer?.addEventListener("click", (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }
            const removeId = Number(target.getAttribute("data-remove-id"));
            if (!Number.isFinite(removeId)) {
                return;
            }
            panelApi?.removeControlItem(removeId);
            panelApi?.updateIndicator();
            panelApi?.updateActiveFlag();
        });
        itemsContainer?.addEventListener("change", (event) => {
            const target = event.target;
            if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLSelectElement)) {
                return;
            }

            const modelId = Number(target.getAttribute("data-model-id"));
            if (Number.isFinite(modelId) && target instanceof HTMLSelectElement) {
                panelApi?.updateControlItem(modelId, { modelId: target.value });
                return;
            }

            const scaleId = Number(target.getAttribute("data-scale-id"));
            if (Number.isFinite(scaleId) && target instanceof HTMLInputElement) {
                const scale = Number(target.value);
                if (Number.isFinite(scale)) {
                    panelApi?.updateControlItem(scaleId, { conditioningScale: scale });
                }
                return;
            }

            const guidanceStartId = Number(target.getAttribute("data-guidance-start-id"));
            if (Number.isFinite(guidanceStartId) && target instanceof HTMLInputElement) {
                const guidanceStart = Number(target.value);
                if (Number.isFinite(guidanceStart)) {
                    panelApi?.updateControlItem(guidanceStartId, { guidanceStart });
                }
                return;
            }

            const guidanceEndId = Number(target.getAttribute("data-guidance-end-id"));
            if (Number.isFinite(guidanceEndId) && target instanceof HTMLInputElement) {
                const guidanceEnd = Number(target.value);
                if (Number.isFinite(guidanceEnd)) {
                    panelApi?.updateControlItem(guidanceEndId, { guidanceEnd });
                }
            }
        });

        loadPreprocessors();
        panelApi?.renderControlItems?.();
        panelApi?.updateIndicator();
        panelApi?.updateActiveFlag();
        updateDownloadLinkState(false);
    }

    async function ensureControlNetUI() {
        if (controlNetUiReady) {
            return;
        }
        if (controlNetUiLoading) {
            return controlNetUiLoading;
        }
        controlNetUiLoading = (async () => {
            await getPanelApi()?.loadPanel?.();
            await loadControlNetModal();
            const panel = document.getElementById("controlnet-toggle");
            const modal = document.getElementById("preprocessor-modal");
            if (!panel || !modal) {
                throw new Error("ControlNet preprocessor UI failed to load.");
            }
            ensurePreprocessorLayoutStructure();
            initControlNetUI();
            controlNetUiReady = true;
        })()
            .catch((error) => {
                console.warn("ControlNet UI initialization failed:", error);
            })
            .finally(() => {
                controlNetUiLoading = null;
            });
        return controlNetUiLoading;
    }

    async function init() {
        await ensureControlNetUI();
    }

    window.ControlNetPreprocessor = {
        init,
        ensureControlNetUI,
        openPreprocessorModal,
        closePreprocessorModal,
    };
})();
