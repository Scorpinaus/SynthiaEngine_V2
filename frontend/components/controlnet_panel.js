(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    const DEFAULT_CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny";
    const state = {
        previewUrl: null,
        previewBlob: null,
        preprocessors: new Map(),
        preprocessorId: null,
        controlItems: [],
        nextControlItemId: 1,
        activeControlIndex: 0,
        perItemGuidanceTimingEnabled: false,
    };

    function emitAdapterSummaryChanged() {
        window.dispatchEvent(new CustomEvent("adapter-summary-changed", { detail: { panel: "controlnet" } }));
    }

    function getState() {
        return state;
    }

    function getSummary() {
        const preprocessors = Array.from(state.preprocessors.values());
        const availablePreprocessors = preprocessors.filter(
            (preprocessor) => preprocessor?.available !== false
        ).length;
        const enabledToggle = document.getElementById("controlnet-enabled");
        return {
            availablePreprocessors,
            totalPreprocessors: preprocessors.length,
            activeItems: state.controlItems.length,
            enabled: Boolean(enabledToggle?.checked),
        };
    }

    function updateIndicator() {
        const indicator = document.getElementById("controlnet-indicator");
        const enabledToggle = document.getElementById("controlnet-enabled");
        const status = document.getElementById("controlnet-status");
        const hasItems = state.controlItems.length > 0;
        const isActive = Boolean(enabledToggle?.checked && hasItems);
        if (indicator) {
            indicator.classList.toggle("is-active", isActive);
        }
        if (status) {
            status.textContent = isActive
                ? `ControlNet ready (${state.controlItems.length} image${state.controlItems.length === 1 ? "" : "s"}).`
                : "No preprocessor applied.";
        }
        emitAdapterSummaryChanged();
    }

    function updateActiveFlag() {
        const flag = document.getElementById("controlnet-active-flag");
        if (!flag) {
            return;
        }
        const enabledToggle = document.getElementById("controlnet-enabled");
        const isActive = Boolean(enabledToggle?.checked && state.controlItems.length > 0);
        flag.classList.toggle("is-hidden", !isActive);
        flag.style.display = isActive ? "inline-flex" : "none";
        emitAdapterSummaryChanged();
    }

    function _modelOptionsForItem(item) {
        const definition = item.preprocessorId ? state.preprocessors.get(item.preprocessorId) : null;
        const options = [];
        const seen = new Set();
        const pushUnique = (value) => {
            const normalized = String(value || "").trim();
            if (!normalized || seen.has(normalized)) {
                return;
            }
            seen.add(normalized);
            options.push(normalized);
        };
        (definition?.recommended_sd15_control_models || []).forEach(pushUnique);
        (definition?.legacy_aliases || []).forEach(pushUnique);
        pushUnique(item.modelId || DEFAULT_CONTROLNET_MODEL);
        pushUnique(DEFAULT_CONTROLNET_MODEL);
        return options;
    }

    function _escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function renderControlItems() {
        const container = document.getElementById("controlnet-items");
        const empty = document.getElementById("controlnet-items-empty");
        const count = document.getElementById("controlnet-count");
        const prevButton = document.getElementById("controlnet-prev");
        const nextButton = document.getElementById("controlnet-next");
        if (!container || !empty) {
            return;
        }

        const total = state.controlItems.length;
        if (count) {
            count.textContent = total > 0 ? `${state.activeControlIndex + 1} / ${total}` : "0 / 0";
        }
        if (prevButton) {
            prevButton.disabled = total <= 1;
        }
        if (nextButton) {
            nextButton.disabled = total <= 1;
        }

        if (state.controlItems.length === 0) {
            container.innerHTML = "";
            empty.classList.remove("is-hidden");
            state.activeControlIndex = 0;
            return;
        }

        if (state.activeControlIndex >= total) {
            state.activeControlIndex = total - 1;
        }
        if (state.activeControlIndex < 0) {
            state.activeControlIndex = 0;
        }

        empty.classList.add("is-hidden");
        const item = state.controlItems[state.activeControlIndex];
        const options = _modelOptionsForItem(item)
            .map((modelId) => {
                const selected = modelId === item.modelId ? " selected" : "";
                return `<option value="${_escapeHtml(modelId)}"${selected}>${_escapeHtml(modelId)}</option>`;
            })
            .join("");
        const timingControls = state.perItemGuidanceTimingEnabled
            ? `
                <div class="field-row">
                    <label class="field">
                        <span>Guidance Start</span>
                        <input data-guidance-start-id="${item.id}" type="number" min="0" max="1" step="0.01" value="${Number(item.guidanceStart ?? 0.0)}" />
                    </label>
                    <label class="field">
                        <span>Guidance End</span>
                        <input data-guidance-end-id="${item.id}" type="number" min="0" max="1" step="0.01" value="${Number(item.guidanceEnd ?? 1.0)}" />
                    </label>
                </div>`
            : "";
        container.innerHTML = `
            <div class="controlnet-item" data-control-id="${item.id}">
                <div class="controlnet-item-head">
                    <span>Control #${state.activeControlIndex + 1} (${_escapeHtml(item.preprocessorId || "unknown")})</span>
                    <button type="button" class="secondary controlnet-item-remove" data-remove-id="${item.id}">Remove</button>
                </div>
                <img src="${_escapeHtml(item.previewUrl)}" alt="Control image ${state.activeControlIndex + 1}" />
                <label class="field">
                    <span>ControlNet Model</span>
                    <select data-model-id="${item.id}">${options}</select>
                </label>
                <label class="field">
                    <span>Conditioning Scale</span>
                    <input data-scale-id="${item.id}" type="number" min="0" max="2" step="0.05" value="${Number(item.conditioningScale ?? 1.0)}" />
                </label>
                ${timingControls}
            </div>`;
    }

    function addControlItem({
        previewBlob,
        previewUrl,
        preprocessorId,
        modelId,
        conditioningScale,
        guidanceStart,
        guidanceEnd,
    }) {
        const item = {
            id: state.nextControlItemId++,
            previewBlob,
            previewUrl,
            preprocessorId: preprocessorId || null,
            modelId: modelId || DEFAULT_CONTROLNET_MODEL,
            conditioningScale: Number(conditioningScale ?? 1.0),
            guidanceStart: Number(guidanceStart ?? 0.0),
            guidanceEnd: Number(guidanceEnd ?? 1.0),
        };
        state.controlItems.push(item);
        state.activeControlIndex = state.controlItems.length - 1;
        state.previewBlob = previewBlob ?? null;
        state.previewUrl = previewUrl ?? null;
        state.preprocessorId = preprocessorId || null;
        renderControlItems();
        emitAdapterSummaryChanged();
        return item;
    }

    function removeControlItem(itemId) {
        const index = state.controlItems.findIndex((item) => item.id === itemId);
        if (index < 0) {
            return;
        }
        const [removed] = state.controlItems.splice(index, 1);
        if (removed?.previewUrl) {
            URL.revokeObjectURL(removed.previewUrl);
        }
        if (state.activeControlIndex >= state.controlItems.length) {
            state.activeControlIndex = Math.max(0, state.controlItems.length - 1);
        }
        if (state.controlItems.length === 0) {
            state.previewBlob = null;
            state.previewUrl = null;
            state.preprocessorId = null;
        }
        renderControlItems();
        emitAdapterSummaryChanged();
    }

    function updateControlItem(itemId, patch) {
        const item = state.controlItems.find((entry) => entry.id === itemId);
        if (!item) {
            return;
        }
        Object.assign(item, patch);
        emitAdapterSummaryChanged();
    }

    function updateControlItemFromField(target) {
        if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLSelectElement)) {
            return false;
        }

        if (target instanceof HTMLSelectElement && target.hasAttribute("data-model-id")) {
            updateControlItem(Number(target.getAttribute("data-model-id")), { modelId: target.value });
            return true;
        }

        if (!(target instanceof HTMLInputElement)) {
            return false;
        }

        if (target.hasAttribute("data-scale-id")) {
            const scale = Number(target.value);
            if (Number.isFinite(scale)) {
                updateControlItem(Number(target.getAttribute("data-scale-id")), { conditioningScale: scale });
            }
            return true;
        }

        if (target.hasAttribute("data-guidance-start-id")) {
            const guidanceStart = Number(target.value);
            if (Number.isFinite(guidanceStart)) {
                updateControlItem(Number(target.getAttribute("data-guidance-start-id")), { guidanceStart });
            }
            return true;
        }

        if (target.hasAttribute("data-guidance-end-id")) {
            const guidanceEnd = Number(target.value);
            if (Number.isFinite(guidanceEnd)) {
                updateControlItem(Number(target.getAttribute("data-guidance-end-id")), { guidanceEnd });
            }
            return true;
        }

        return false;
    }

    function clearControlItems() {
        state.controlItems.forEach((item) => {
            if (item.previewUrl) {
                URL.revokeObjectURL(item.previewUrl);
            }
        });
        state.controlItems = [];
        state.activeControlIndex = 0;
        state.previewBlob = null;
        state.previewUrl = null;
        state.preprocessorId = null;
        renderControlItems();
        emitAdapterSummaryChanged();
    }

    function showPrevControlItem() {
        if (state.controlItems.length <= 1) {
            return;
        }
        state.activeControlIndex =
            (state.activeControlIndex - 1 + state.controlItems.length) % state.controlItems.length;
        renderControlItems();
        emitAdapterSummaryChanged();
    }

    function showNextControlItem() {
        if (state.controlItems.length <= 1) {
            return;
        }
        state.activeControlIndex = (state.activeControlIndex + 1) % state.controlItems.length;
        renderControlItems();
        emitAdapterSummaryChanged();
    }

    function setPerItemGuidanceTimingEnabled(enabled) {
        state.perItemGuidanceTimingEnabled = Boolean(enabled);
        renderControlItems();
        emitAdapterSummaryChanged();
    }

    function togglePanel() {
        const content = document.getElementById("controlnet-content");
        const chevron = document.getElementById("controlnet-chevron");
        if (!content || !chevron) {
            return;
        }
        const isOpen = content.classList.toggle("is-open");
        chevron.textContent = isOpen ? "\u25b4" : "\u25be";
    }

    async function loadPanel() {
        const container = document.getElementById("controlnet-panel-root");
        if (!container) {
            return;
        }
        try {
            const res = await fetch(resolveAssetUrl("controlnet_panel.html?v=2"), { cache: "no-store" });
            if (!res.ok) {
                throw new Error(`Failed to load ControlNet panel UI: ${res.status}`);
            }
            container.innerHTML = await res.text();
        } catch (error) {
            console.warn("Failed to load ControlNet panel UI:", error);
        }
    }

    window.ControlNetPanel = {
        getState,
        getSummary,
        updateIndicator,
        updateActiveFlag,
        renderControlItems,
        addControlItem,
        removeControlItem,
        updateControlItem,
        updateControlItemFromField,
        clearControlItems,
        showPrevControlItem,
        showNextControlItem,
        setPerItemGuidanceTimingEnabled,
        togglePanel,
        loadPanel,
    };
})();
