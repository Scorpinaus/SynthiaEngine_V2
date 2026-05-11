(() => {
    function normalizePromptPresets(value) {
        if (!Array.isArray(value)) {
            return [];
        }
        return value
            .map((preset) => ({
                name: String(preset?.name || "").trim(),
                words: Array.isArray(preset?.words)
                    ? preset.words.map((word) => String(word || "").trim()).filter(Boolean)
                    : [],
            }))
            .filter((preset) => preset.name && preset.words.length > 0);
    }

    function parsePresetWords(value) {
        return String(value || "")
            .split(/[\n,]+/)
            .map((word) => word.trim())
            .filter(Boolean);
    }

    function wordsToText(words) {
        return Array.isArray(words) ? words.join(", ") : "";
    }

    function createEditor({ container, apiBase, loraId, initialEntry = null, onSaved = null, compact = false }) {
        const state = {
            entry: initialEntry,
            presets: normalizePromptPresets(initialEntry?.prompt_presets),
            loading: false,
            saving: false,
            status: "",
            statusVariant: "info",
        };

        function setStatus(message, variant = "info") {
            state.status = message;
            state.statusVariant = variant;
            render();
        }

        function setLoading(value) {
            state.loading = value;
            render();
        }

        function setSaving(value) {
            state.saving = value;
            render();
        }

        function updatePreset(index, field, value) {
            const next = [...state.presets];
            const current = next[index] || { name: "", words: [] };
            next[index] = {
                ...current,
                [field]: field === "words" ? parsePresetWords(value) : String(value || "").trim(),
            };
            state.presets = next;
        }

        function addPreset() {
            state.presets = [...state.presets, { name: "", words: [] }];
            render();
        }

        function removePreset(index) {
            state.presets = state.presets.filter((_, presetIndex) => presetIndex !== index);
            render();
        }

        function validatePresets() {
            return state.presets
                .map((preset) => ({
                    name: String(preset.name || "").trim(),
                    words: Array.isArray(preset.words) ? preset.words.map((word) => String(word || "").trim()).filter(Boolean) : [],
                }))
                .filter((preset) => preset.name || preset.words.length > 0);
        }

        async function load() {
            if (state.entry) {
                render();
                return;
            }
            setLoading(true);
            try {
                const response = await fetch(`${apiBase}/lora-models/${encodeURIComponent(String(loraId))}`);
                if (!response.ok) {
                    const errorBody = await response.json().catch(() => ({}));
                    throw new Error(errorBody.detail || "Unable to load LoRA entry.");
                }
                state.entry = await response.json();
                state.presets = normalizePromptPresets(state.entry.prompt_presets);
                state.status = "Prompt presets loaded.";
                state.statusVariant = "success";
            } catch (error) {
                console.error(error);
                state.status = error.message || "Unable to load prompt presets.";
                state.statusVariant = "error";
            } finally {
                setLoading(false);
            }
        }

        async function save() {
            const normalized = validatePresets();
            const invalid = normalized.find((preset) => !preset.name || preset.words.length === 0);
            if (invalid || normalized.length !== state.presets.filter((preset) => preset.name || preset.words?.length).length) {
                setStatus("Each prompt preset needs a name and at least one word.", "error");
                return;
            }

            setSaving(true);
            try {
                const response = await fetch(`${apiBase}/lora-models/${encodeURIComponent(String(loraId))}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt_presets: normalized }),
                });
                if (!response.ok) {
                    const errorBody = await response.json().catch(() => ({}));
                    throw new Error(errorBody.detail || "Unable to save prompt presets.");
                }
                const updated = await response.json();
                state.entry = updated;
                state.presets = normalizePromptPresets(updated.prompt_presets);
                state.status = "Prompt presets saved.";
                state.statusVariant = "success";
                if (typeof onSaved === "function") {
                    onSaved(updated);
                }
            } catch (error) {
                console.error(error);
                state.status = error.message || "Unable to save prompt presets.";
                state.statusVariant = "error";
            } finally {
                setSaving(false);
            }
        }

        function render() {
            container.innerHTML = "";

            const root = document.createElement("div");
            root.className = compact ? "lora-preset-editor-shell compact" : "lora-preset-editor-shell";

            const header = document.createElement("div");
            header.className = "lora-preset-editor-title";
            const title = document.createElement("h3");
            const entryName = state.entry?.name || `LoRA ${loraId}`;
            title.textContent = compact ? "Prompt Presets" : `${entryName} Prompt Presets`;
            const subtitle = document.createElement("p");
            subtitle.textContent = state.entry
                ? `LoRA ID: ${state.entry.lora_id}`
                : `LoRA ID: ${loraId}`;
            header.append(title, subtitle);

            const list = document.createElement("div");
            list.className = "lora-preset-edit-list";

            if (state.loading) {
                const loading = document.createElement("div");
                loading.className = "field-hint";
                loading.textContent = "Loading prompt presets...";
                list.appendChild(loading);
            } else if (state.presets.length === 0) {
                const empty = document.createElement("div");
                empty.className = "field-hint";
                empty.textContent = "No prompt presets yet.";
                list.appendChild(empty);
            }

            state.presets.forEach((preset, index) => {
                const row = document.createElement("div");
                row.className = "lora-preset-edit-row";

                const nameLabel = document.createElement("label");
                nameLabel.className = "field";
                nameLabel.innerHTML = "<span>Preset Name</span>";
                const nameInput = document.createElement("input");
                nameInput.type = "text";
                nameInput.value = preset.name || "";
                nameInput.placeholder = "Soft watercolor";
                nameInput.addEventListener("input", (event) => {
                    updatePreset(index, "name", event.target.value);
                });
                nameLabel.appendChild(nameInput);

                const wordsLabel = document.createElement("label");
                wordsLabel.className = "field";
                wordsLabel.innerHTML = "<span>Words</span>";
                const wordsInput = document.createElement("textarea");
                wordsInput.rows = compact ? 2 : 3;
                wordsInput.value = wordsToText(preset.words);
                wordsInput.placeholder = "soft watercolor, paper texture, pastel colors";
                wordsInput.addEventListener("input", (event) => {
                    updatePreset(index, "words", event.target.value);
                });
                wordsLabel.appendChild(wordsInput);

                const remove = document.createElement("button");
                remove.type = "button";
                remove.className = "secondary";
                remove.textContent = "Remove";
                remove.disabled = state.saving;
                remove.addEventListener("click", () => removePreset(index));

                row.append(nameLabel, wordsLabel, remove);
                list.appendChild(row);
            });

            const actions = document.createElement("div");
            actions.className = "lora-preset-editor-actions";

            const add = document.createElement("button");
            add.type = "button";
            add.className = "secondary";
            add.textContent = "Add Preset";
            add.disabled = state.loading || state.saving;
            add.addEventListener("click", addPreset);

            const saveButton = document.createElement("button");
            saveButton.type = "button";
            saveButton.className = "primary";
            saveButton.textContent = state.saving ? "Saving..." : "Save Presets";
            saveButton.disabled = state.loading || state.saving;
            saveButton.addEventListener("click", save);

            actions.append(add, saveButton);

            const status = document.createElement("div");
            status.className = `model-form-state ${state.statusVariant}`;
            status.setAttribute("aria-live", "polite");
            status.textContent = state.status;

            root.append(header, list, actions, status);
            container.appendChild(root);
        }

        render();
        void load();

        return {
            load,
            save,
            getEntry: () => state.entry,
            getPresets: () => normalizePromptPresets(state.presets),
        };
    }

    window.LoraPromptPresetEditor = {
        mount: createEditor,
        normalizePromptPresets,
        parsePresetWords,
    };
})();
