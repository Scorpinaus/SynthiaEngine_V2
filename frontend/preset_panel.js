(() => {
    const state = {
        apiBase: "",
        family: "",
        taskType: "",
        collectSettings: null,
        applySettings: null,
        presets: [],
        selectedId: null,
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

    function togglePresetPanel() {
        const content = document.getElementById("preset-content");
        const chevron = document.getElementById("preset-chevron");
        if (!content || !chevron) {
            return;
        }
        const isOpen = content.classList.toggle("is-open");
        chevron.textContent = isOpen ? "\u25b4" : "\u25be";
    }

    function getSelectedPreset() {
        if (!Number.isFinite(state.selectedId)) {
            return null;
        }
        return state.presets.find((entry) => entry.preset_id === state.selectedId) ?? null;
    }

    function renderPresetOptions() {
        const select = document.getElementById("preset-select");
        const nameInput = document.getElementById("preset-name");
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
            if (nameInput) {
                nameInput.value = "";
            }
            setStatus("No preset selected.");
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

        const selected = getSelectedPreset();
        if (nameInput) {
            nameInput.value = selected?.name ?? "";
        }
        setStatus(selected ? `Selected "${selected.name}".` : "No preset selected.");
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
            await fetchPresets();
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
            setStatus(`Loaded preset "${selected.name}".`);
        } catch (error) {
            setStatus(getErrorMessage(error, "Failed to load preset."));
        }
    }

    function bindEvents() {
        document.getElementById("preset-toggle")?.addEventListener("click", togglePresetPanel);
        document.getElementById("preset-refresh")?.addEventListener("click", () => {
            void fetchPresets().catch((error) => {
                setStatus(getErrorMessage(error, "Failed to refresh presets."));
            });
        });
        document.getElementById("preset-load")?.addEventListener("click", () => {
            void loadSelectedPreset();
        });
        document.getElementById("preset-save-new")?.addEventListener("click", () => {
            void saveNewPreset();
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
            const nameInput = document.getElementById("preset-name");
            if (nameInput) {
                nameInput.value = selected?.name ?? "";
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

        bindEvents();
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
