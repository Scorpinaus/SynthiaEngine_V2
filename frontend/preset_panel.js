(() => {
    const state = {
        apiBase: "",
        family: "",
        taskType: "",
        collectSettings: null,
        applySettings: null,
        presets: [],
        selectedId: null,
        loadedPresetId: null,
        uiMode: "default",
    };

    const UI_MODES = {
        DEFAULT: "default",
        CREATE: "create",
        MANAGE: "manage",
    };

    function setStatus(message) {
        const status = document.getElementById("preset-status");
        if (status) {
            status.textContent = message;
        }
    }

    function getErrorMessage(error, fallback) {
        if (!error) {
            return fallback;
        }
        if (typeof error.message === "string" && error.message) {
            return error.message;
        }
        return fallback;
    }

    async function requestJson(url, options = {}) {
        const res = await fetch(url, options);
        if (!res.ok) {
            let detail = "";
            try {
                const body = await res.json();
                detail = typeof body?.detail === "string" ? body.detail : "";
            } catch (_error) {
                detail = "";
            }
            if (!detail) {
                try {
                    detail = await res.text();
                } catch (_error) {
                    detail = "";
                }
            }
            const suffix = detail ? `: ${detail}` : "";
            throw new Error(`Request failed (${res.status})${suffix}`);
        }
        if (res.status === 204) {
            return null;
        }
        return await res.json();
    }

    function getPresetModal() {
        return document.getElementById("preset-modal");
    }

    function closePresetDialog() {
        const modal = getPresetModal();
        if (!modal) {
            return;
        }
        modal.classList.add("hidden");
    }

    function openPresetDialog() {
        const modal = getPresetModal();
        if (!modal) {
            return;
        }
        modal.classList.remove("hidden");
    }

    function ensureDialogShell() {
        const content = document.getElementById("preset-content");
        if (!content || getPresetModal()) {
            return;
        }

        const modal = document.createElement("div");
        modal.id = "preset-modal";
        modal.className = "modal hidden";
        modal.innerHTML = `
            <div class="modal-overlay" id="preset-modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Prompt + Generation Presets</h2>
                    <button class="secondary" id="preset-modal-close" type="button">Close</button>
                </div>
                <div class="modal-body" id="preset-modal-body"></div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.querySelector("#preset-modal-body")?.appendChild(content);
    }

    function getSelectedPreset() {
        if (!Number.isFinite(state.selectedId)) {
            return null;
        }
        return state.presets.find((entry) => entry.preset_id === state.selectedId) ?? null;
    }

    function syncNameInputWithSelected() {
        if (state.uiMode !== UI_MODES.MANAGE) {
            return;
        }
        const nameInput = document.getElementById("preset-name");
        if (!nameInput) {
            return;
        }
        const selected = getSelectedPreset();
        nameInput.value = selected?.name ?? "";
    }

    function setUiMode(mode, options = {}) {
        const { clearName = false, focusName = false } = options;
        const nextMode = Object.values(UI_MODES).includes(mode) ? mode : UI_MODES.DEFAULT;
        state.uiMode = nextMode;

        const nameField = document.getElementById("preset-name-field");
        const createActions = document.getElementById("preset-create-actions");
        const manageActions = document.getElementById("preset-manage-actions");
        const nameInput = document.getElementById("preset-name");

        nameField?.classList.toggle("is-hidden", nextMode === UI_MODES.DEFAULT);
        createActions?.classList.toggle("is-hidden", nextMode !== UI_MODES.CREATE);
        manageActions?.classList.toggle("is-hidden", nextMode !== UI_MODES.MANAGE);

        if (nameInput) {
            if (nextMode === UI_MODES.CREATE && clearName) {
                nameInput.value = "";
            }
            if (nextMode === UI_MODES.MANAGE) {
                const selected = getSelectedPreset();
                nameInput.value = selected?.name ?? "";
            }
            if (focusName) {
                nameInput.focus();
                nameInput.select();
            }
        }
    }

    function renderPresetOptions() {
        const select = document.getElementById("preset-select");
        if (!select) {
            return;
        }

        select.innerHTML = "";
        if (state.presets.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No presets";
            option.selected = true;
            select.appendChild(option);
            state.selectedId = null;
            if (state.uiMode === UI_MODES.MANAGE) {
                setUiMode(UI_MODES.DEFAULT);
            }
            setStatus("No preset selected.");
            updateLoadedIndicator();
            return;
        }

        state.presets.forEach((entry) => {
            const option = document.createElement("option");
            option.value = String(entry.preset_id);
            option.textContent = entry.name;
            if (entry.preset_id === state.selectedId) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        if (!Number.isFinite(state.selectedId) || !state.presets.some((entry) => entry.preset_id === state.selectedId)) {
            state.selectedId = state.presets[0].preset_id;
            select.value = String(state.selectedId);
        }

        if (state.uiMode === UI_MODES.MANAGE) {
            syncNameInputWithSelected();
        }
        const selected = getSelectedPreset();
        setStatus(selected ? `Selected "${selected.name}".` : "No preset selected.");
        updateLoadedIndicator();
    }

    function updateLoadedIndicator() {
        const indicator = document.getElementById("preset-active-indicator");
        if (!indicator) {
            return;
        }
        const loaded = state.presets.find((entry) => entry.preset_id === state.loadedPresetId) ?? null;
        const isActive = Boolean(loaded);
        indicator.classList.toggle("is-hidden", !isActive);
        indicator.textContent = loaded ? `Preset Active: ${loaded.name}` : "Preset Active";
    }

    async function fetchPresets() {
        const params = new URLSearchParams();
        if (state.family) {
            params.set("family", state.family);
        }
        if (state.taskType) {
            params.set("task_type", state.taskType);
        }
        const query = params.toString();
        const url = `${state.apiBase}/api/presets${query ? `?${query}` : ""}`;
        const presets = await requestJson(url);
        state.presets = Array.isArray(presets) ? presets : [];
        if (!state.presets.some((entry) => entry.preset_id === state.loadedPresetId)) {
            state.loadedPresetId = null;
        }
        if (!state.presets.some((entry) => entry.preset_id === state.selectedId)) {
            state.selectedId = state.presets.length > 0 ? state.presets[0].preset_id : null;
        }
        renderPresetOptions();
    }

    function normalizeName() {
        const nameInput = document.getElementById("preset-name");
        const name = (nameInput?.value ?? "").trim();
        if (!name) {
            throw new Error("Preset name is required.");
        }
        return name;
    }

    function readSettingsForSave() {
        if (typeof state.collectSettings !== "function") {
            throw new Error("Preset collectSettings callback is not configured.");
        }
        const settings = state.collectSettings();
        if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
            throw new Error("collectSettings must return an object.");
        }
        return settings;
    }

    async function saveNewPreset() {
        try {
            const body = {
                name: normalizeName(),
                family: state.family,
                task_type: state.taskType,
                settings: readSettingsForSave(),
            };
            const created = await requestJson(`${state.apiBase}/api/presets`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            state.selectedId = Number(created?.preset_id);
            await fetchPresets();
            setUiMode(UI_MODES.MANAGE);
            setStatus(`Saved preset "${created?.name ?? body.name}".`);
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to save preset."));
        }
    }

    async function updatePreset() {
        const selected = getSelectedPreset();
        if (!selected) {
            setStatus("Select a preset to update.");
            return;
        }
        try {
            const body = {
                name: normalizeName(),
                settings: readSettingsForSave(),
            };
            const updated = await requestJson(`${state.apiBase}/api/presets/${selected.preset_id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            state.selectedId = Number(updated?.preset_id ?? selected.preset_id);
            await fetchPresets();
            setUiMode(UI_MODES.MANAGE);
            setStatus(`Updated preset "${updated?.name ?? body.name}".`);
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to update preset."));
        }
    }

    async function deletePreset() {
        const selected = getSelectedPreset();
        if (!selected) {
            setStatus("Select a preset to delete.");
            return;
        }
        if (!window.confirm(`Delete preset "${selected.name}"?`)) {
            return;
        }
        try {
            await requestJson(`${state.apiBase}/api/presets/${selected.preset_id}`, {
                method: "DELETE",
            });
            state.selectedId = null;
            if (state.loadedPresetId === selected.preset_id) {
                state.loadedPresetId = null;
            }
            await fetchPresets();
            setUiMode(getSelectedPreset() ? UI_MODES.MANAGE : UI_MODES.DEFAULT);
            setStatus(`Deleted preset "${selected.name}".`);
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to delete preset."));
        }
    }

    async function loadSelectedPreset() {
        const selected = getSelectedPreset();
        if (!selected) {
            setStatus("Select a preset to load.");
            return;
        }
        if (typeof state.applySettings !== "function") {
            setStatus("Preset applySettings callback is not configured.");
            return;
        }
        try {
            await Promise.resolve(state.applySettings(selected.settings ?? {}));
            state.loadedPresetId = selected.preset_id;
            updateLoadedIndicator();
            setUiMode(UI_MODES.MANAGE);
            setStatus(`Loaded preset "${selected.name}".`);
            closePresetDialog();
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to load preset."));
        }
    }

    function onPresetToggleClick() {
        openPresetDialog();
    }

    function onPresetModalKeydown(event) {
        if (event.key === "Escape") {
            closePresetDialog();
        }
    }

    function bindEvents() {
        document.getElementById("preset-toggle")?.addEventListener("click", onPresetToggleClick);
        document.getElementById("preset-modal-close")?.addEventListener("click", closePresetDialog);
        document.getElementById("preset-modal-overlay")?.addEventListener("click", closePresetDialog);
        document.addEventListener("keydown", onPresetModalKeydown);
        document.getElementById("preset-refresh")?.addEventListener("click", () => {
            void fetchPresets().catch((error) => {
                setStatus(getErrorMessage(error, "Failed to refresh presets."));
            });
        });
        document.getElementById("preset-load")?.addEventListener("click", () => {
            void loadSelectedPreset();
        });
        document.getElementById("preset-add-new")?.addEventListener("click", () => {
            setUiMode(UI_MODES.CREATE, { clearName: true, focusName: true });
            setStatus("Enter a preset name and click Save New.");
        });
        document.getElementById("preset-save-new")?.addEventListener("click", () => {
            void saveNewPreset();
        });
        document.getElementById("preset-cancel")?.addEventListener("click", () => {
            setUiMode(UI_MODES.DEFAULT);
            const selected = getSelectedPreset();
            setStatus(selected ? `Selected "${selected.name}".` : "No preset selected.");
        });
        document.getElementById("preset-update")?.addEventListener("click", () => {
            void updatePreset();
        });
        document.getElementById("preset-delete")?.addEventListener("click", () => {
            void deletePreset();
        });
        document.getElementById("preset-select")?.addEventListener("change", (event) => {
            const value = Number(event.target?.value);
            state.selectedId = Number.isFinite(value) ? value : null;
            const selected = getSelectedPreset();
            if (state.uiMode === UI_MODES.MANAGE) {
                syncNameInputWithSelected();
            }
            setStatus(selected ? `Selected "${selected.name}".` : "No preset selected.");
        });
    }

    async function init({ apiBase, family, taskType, collectSettings, applySettings }) {
        const container = document.getElementById("preset-panel-root");
        if (!container) {
            return;
        }
        state.apiBase = apiBase ?? window.API_BASE ?? "";
        state.family = String(family ?? "").trim().toLowerCase();
        state.taskType = String(taskType ?? "").trim();
        state.collectSettings = collectSettings ?? null;
        state.applySettings = applySettings ?? null;
        state.presets = [];
        state.selectedId = null;
        state.loadedPresetId = null;
        state.uiMode = UI_MODES.DEFAULT;

        try {
            const res = await fetch("preset_panel.html?v=1", { cache: "no-store" });
            if (!res.ok) {
                throw new Error(`Failed to load preset panel UI: ${res.status}`);
            }
            container.innerHTML = await res.text();
        } catch (error) {
            console.warn("Failed to load preset panel UI:", error);
            return;
        }

        ensureDialogShell();
        bindEvents();
        setUiMode(UI_MODES.DEFAULT);
        try {
            await fetchPresets();
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to load presets."));
        }
    }

    window.PresetPanel = {
        init,
        reload: fetchPresets,
    };
})();
