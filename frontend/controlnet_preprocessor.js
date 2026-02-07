(() => {
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
            const res = await fetch("controlnet_preprocessor.html?v=2", { cache: "no-store" });
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

    function updatePreprocessorDefaults(preprocessorId) {
        const state = getState();
        const definition = state?.preprocessors.get(preprocessorId);
        const defaults = definition?.defaults ?? {};
        const description = definition?.description ?? "";
        const lowThresholdInput = document.getElementById("preprocessor-low-threshold");
        const highThresholdInput = document.getElementById("preprocessor-high-threshold");
        const descriptionNode = document.getElementById("preprocessor-description");
        const cannyRow = document.getElementById("canny-thresholds");
        if (lowThresholdInput) {
            lowThresholdInput.value = Number(defaults.low_threshold ?? 100);
        }
        if (highThresholdInput) {
            highThresholdInput.value = Number(defaults.high_threshold ?? 200);
        }
        if (descriptionNode) {
            descriptionNode.textContent = description;
        }
        if (cannyRow) {
            const isCanny = preprocessorId === "canny";
            cannyRow.classList.toggle("is-hidden", !isCanny);
        }
    }

    function buildPreprocessorParams(preprocessorId) {
        const state = getState();
        const definition = state?.preprocessors.get(preprocessorId);
        const params = { ...(definition?.defaults ?? {}) };
        if (preprocessorId === "canny") {
            const lowThresholdInput = document.getElementById("preprocessor-low-threshold");
            const highThresholdInput = document.getElementById("preprocessor-high-threshold");
            params.low_threshold = Number(lowThresholdInput?.value ?? params.low_threshold ?? 100);
            params.high_threshold = Number(highThresholdInput?.value ?? params.high_threshold ?? 200);
        }
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
        if (state.previewUrl) {
            // Avoid leaking object URLs when users iterate on preprocess settings.
            URL.revokeObjectURL(state.previewUrl);
        }
        state.previewUrl = URL.createObjectURL(blob);
        state.previewBlob = blob;
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

        toggleButton?.addEventListener("click", panelApi?.togglePanel);
        openButton?.addEventListener("click", openPreprocessorModal);
        closeButton?.addEventListener("click", closePreprocessorModal);
        overlay?.addEventListener("click", closePreprocessorModal);
        applyButton?.addEventListener("click", applyPreprocessor);
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
            }
            updateDownloadLinkState(false);
            panelApi?.updateActiveFlag();
        });

        loadPreprocessors();
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
