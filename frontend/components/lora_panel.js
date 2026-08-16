(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    const LOG_LORA_PANEL = true;
    const DEFAULT_STRENGTH = 0.8;
    const QWEN_LIGHTNING_PROFILE_EVENT = "qwen-lightning-profile-changed";
    let promptPresetEditorScriptPromise = null;

    function logDebug(message, data) {
        if (!LOG_LORA_PANEL) {
            return;
        }
        if (data === undefined) {
            console.debug(`[LoraPanel] ${message}`);
            return;
        }
        console.debug(`[LoraPanel] ${message}`, data);
    }

    const loraState = {
        available: [],
        selected: [],
        family: "",
        taskType: "",
        weightMode: "basic",
        apiBase: "",
        selectedLightningKey: "",
    };

    function emitAdapterSummaryChanged() {
        window.dispatchEvent(new CustomEvent("adapter-summary-changed", { detail: { panel: "lora" } }));
    }

    function getSummary() {
        return {
            available: loraState.available.length,
            selected: loraState.selected.length,
            family: loraState.family,
            weightMode: loraState.weightMode,
        };
    }

    function clampStrength(value) {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) {
            return DEFAULT_STRENGTH;
        }
        return Math.max(0, Math.min(1, parsed));
    }

    function normalizeFamily(value) {
        return String(value || "").trim().toLowerCase();
    }

    function supportsAdvancedWeights() {
        return loraState.family === "sd15";
    }

    function isQwenImageFamily() {
        return loraState.family === "qwen-image";
    }

    function getQwenCompatibilityTask() {
        if (!isQwenImageFamily()) {
            return "";
        }
        const taskType = String(loraState.taskType || "").trim().toLowerCase();
        const normalizedTasks = {
            "qwen-image.text2img": "text2img",
            "qwen-image.img2img": "img2img",
            "qwen-image.inpaint": "inpaint",
        };
        return normalizedTasks[taskType] || "";
    }

    function getLightningProfile(value) {
        const profile = value?.runtime_profile;
        if (!isQwenImageFamily() || profile?.kind !== "qwen_image_lightning") {
            return null;
        }
        const steps = Number(profile.steps);
        const adapterStrength = Number(profile.adapter_strength);
        if ((steps !== 4 && steps !== 8) || adapterStrength !== 1) {
            return null;
        }
        return profile;
    }

    function hasCompatibilityValue(values, expected) {
        return Array.isArray(values) && values.includes(expected);
    }

    function isQwenLightningCompatible(entry) {
        const compatibility = entry?.compatibility;
        const task = getQwenCompatibilityTask();
        return Boolean(
            task &&
            hasCompatibilityValue(compatibility?.base_variants, "qwen-image-2512") &&
            hasCompatibilityValue(compatibility?.runtime_profile_kinds, "qwen_image_lightning") &&
            hasCompatibilityValue(compatibility?.supported_tasks, task),
        );
    }

    function getSelectedQwenMixedStack() {
        if (!isQwenImageFamily()) {
            return null;
        }
        const lightningAdapters = loraState.selected.filter((lora) => getLightningProfile(lora));
        const standardAdapters = loraState.selected.filter((lora) => !getLightningProfile(lora));
        if (
            lightningAdapters.length !== 1 ||
            standardAdapters.length !== 1 ||
            !isQwenLightningCompatible(standardAdapters[0])
        ) {
            return null;
        }
        return { lightning: lightningAdapters[0], companion: standardAdapters[0] };
    }

    function syncQwenMixedStackStatus() {
        const status = document.getElementById("lora-stack-status");
        if (!status) {
            return;
        }
        if (getSelectedQwenMixedStack()) {
            status.textContent = "Experimental stack: Lightning + 1 LoRA";
            status.classList.remove("is-hidden");
            return;
        }
        status.classList.add("is-hidden");
    }

    function getQwenCompatibilityReason(entry) {
        const compatibility = entry?.compatibility;
        const task = getQwenCompatibilityTask();
        if (!compatibility) {
            return "No Lightning compatibility metadata";
        }
        if (!hasCompatibilityValue(compatibility.base_variants, "qwen-image-2512")) {
            return "Not compatible with Qwen Image 2512";
        }
        if (!hasCompatibilityValue(compatibility.runtime_profile_kinds, "qwen_image_lightning")) {
            return "Not compatible with Qwen Image Lightning";
        }
        if (!task || !hasCompatibilityValue(compatibility.supported_tasks, task)) {
            return `Not compatible with Qwen task ${task || "unknown"}`;
        }
        return "Not compatible with selected Lightning";
    }

    function getEntrySelectionState(entry, selected = loraState.selected) {
        if (!isQwenImageFamily()) {
            return { disabled: false, reason: "" };
        }

        const entryId = Number(entry?.lora_id);
        if (selected.some((lora) => Number(lora.lora_id) === entryId)) {
            return { disabled: true, reason: "Already selected" };
        }

        const lightningSelected = selected.filter((lora) => getLightningProfile(lora));
        const standardSelected = selected.filter((lora) => !getLightningProfile(lora));
        const lightningProfile = getLightningProfile(entry);
        const task = getQwenCompatibilityTask();

        if (lightningProfile) {
            if (!task) {
                return { disabled: true, reason: "Lightning requires a supported Qwen task" };
            }
            if (lightningSelected.length > 0) {
                return { disabled: true, reason: "Only one Lightning adapter is allowed" };
            }
            if (standardSelected.length > 1) {
                return { disabled: true, reason: "Remove extra standard LoRAs before selecting Lightning" };
            }
            if (standardSelected.length === 1 && !isQwenLightningCompatible(standardSelected[0])) {
                return {
                    disabled: true,
                    reason: "Selected standard LoRA is not Lightning-compatible for this task",
                };
            }
            return { disabled: false, reason: "" };
        }

        if (lightningSelected.length > 0) {
            if (!isQwenLightningCompatible(entry)) {
                return { disabled: true, reason: getQwenCompatibilityReason(entry) };
            }
            if (standardSelected.length > 0) {
                return { disabled: true, reason: "Only one Lightning-compatible companion is allowed" };
            }
        }
        return { disabled: false, reason: "" };
    }

    function canSelectEntry(entry, selected = loraState.selected) {
        return !getEntrySelectionState(entry, selected).disabled;
    }

    function getOptionLabel(entry, selectionState) {
        const name = entry.name ?? entry.file_path ?? `LoRA ${entry.lora_id}`;
        const lightningProfile = getLightningProfile(entry);
        let label = name;
        if (lightningProfile) {
            label += ` — Lightning · ${lightningProfile.steps} steps`;
        } else if (isQwenLightningCompatible(entry)) {
            label += " — Lightning-compatible · Qwen Image 2512";
        }
        return selectionState.disabled ? `${label} — ${selectionState.reason}` : label;
    }

    function refreshLoraOptions() {
        const select = document.getElementById("lora-select");
        if (!select) {
            return;
        }
        select.innerHTML = "";
        if (loraState.available.length === 0) {
            const fallback = document.createElement("option");
            fallback.value = "";
            fallback.textContent = "No LoRAs available";
            fallback.selected = true;
            select.appendChild(fallback);
            return;
        }
        let selectedOption = false;
        loraState.available.forEach((entry) => {
            const selectionState = getEntrySelectionState(entry);
            const option = document.createElement("option");
            option.value = String(entry.lora_id);
            option.disabled = selectionState.disabled;
            option.title = selectionState.reason;
            option.textContent = getOptionLabel(entry, selectionState);
            if (!selectedOption && !option.disabled) {
                option.selected = true;
                selectedOption = true;
            }
            select.appendChild(option);
        });
    }

    function normalizeSelectedLora(lora) {
        const profile = getLightningProfile(lora);
        if (!profile) {
            return lora;
        }
        const strength = Number(profile.adapter_strength);
        return {
            ...lora,
            runtime_profile: profile,
            strength,
            unet_strength: strength,
            text_encoder_strength: strength,
            target: "both",
        };
    }

    function emitLightningProfileChanged({ force = false } = {}) {
        const selected = loraState.selected.find((lora) => getLightningProfile(lora));
        const profile = selected ? getLightningProfile(selected) : null;
        const nextKey = profile ? `${selected.lora_id}:${profile.steps}` : "";
        if (!force && nextKey === loraState.selectedLightningKey) {
            return;
        }
        loraState.selectedLightningKey = nextKey;
        window.dispatchEvent(new CustomEvent(QWEN_LIGHTNING_PROFILE_EVENT, {
            detail: {
                lora_id: selected?.lora_id ?? null,
                profile,
            },
        }));
    }

    function normalizeWeightMode(mode) {
        return String(mode || "").trim().toLowerCase() === "advanced" ? "advanced" : "basic";
    }

    function isAdvancedWeightMode() {
        return supportsAdvancedWeights() && loraState.weightMode === "advanced";
    }

    function recomputeCombinedStrength(adapter) {
        adapter.strength = clampStrength((adapter.unet_strength + adapter.text_encoder_strength) / 2);
    }

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

    function syncWeightModeUI() {
        const modeRow = document.getElementById("lora-weight-mode-row");
        const basicRadio = document.getElementById("lora-weight-mode-basic");
        const advancedRadio = document.getElementById("lora-weight-mode-advanced");
        if (!modeRow || !basicRadio || !advancedRadio) {
            return;
        }

        if (!supportsAdvancedWeights()) {
            loraState.weightMode = "basic";
            modeRow.classList.add("is-hidden");
            basicRadio.checked = true;
            advancedRadio.checked = false;
            return;
        }

        modeRow.classList.remove("is-hidden");
        const normalized = normalizeWeightMode(loraState.weightMode);
        loraState.weightMode = normalized;
        basicRadio.checked = normalized === "basic";
        advancedRadio.checked = normalized === "advanced";
    }

    function setWeightMode(mode) {
        const nextMode = supportsAdvancedWeights() ? normalizeWeightMode(mode) : "basic";
        if (loraState.weightMode === nextMode) {
            syncWeightModeUI();
            return;
        }
        loraState.weightMode = nextMode;
        syncWeightModeUI();
        logDebug("Updated LoRA weight mode.", { family: loraState.family, weight_mode: loraState.weightMode });
        renderLoraList();
    }

    function bindWeightModeControls() {
        const basicRadio = document.getElementById("lora-weight-mode-basic");
        const advancedRadio = document.getElementById("lora-weight-mode-advanced");
        basicRadio?.addEventListener("change", (event) => {
            if (event.target?.checked) {
                setWeightMode("basic");
            }
        });
        advancedRadio?.addEventListener("change", (event) => {
            if (event.target?.checked) {
                setWeightMode("advanced");
            }
        });
    }

    function toggleLoraPanel() {
        const content = document.getElementById("lora-content");
        const chevron = document.getElementById("lora-chevron");
        if (!content || !chevron) {
            return;
        }
        const isOpen = content.classList.toggle("is-open");
        chevron.textContent = isOpen ? "▴" : "▾";
    }

    function renderLoraList({ forceLightningProfileEvent = false } = {}) {
        syncWeightModeUI();
        refreshLoraOptions();
        const mixedStack = getSelectedQwenMixedStack();
        syncQwenMixedStackStatus();
        const list = document.getElementById("lora-list");
        const emptyState = document.getElementById("lora-empty");
        if (!list || !emptyState) {
            return;
        }
        list.innerHTML = "";
        if (loraState.selected.length === 0) {
            emptyState.classList.remove("is-hidden");
            emitLightningProfileChanged({ force: forceLightningProfileEvent });
            emitAdapterSummaryChanged();
            return;
        }
        emptyState.classList.add("is-hidden");
        loraState.selected.forEach((lora) => {
            const lightningProfile = getLightningProfile(lora);
            if (lightningProfile) {
                Object.assign(lora, normalizeSelectedLora(lora));
            }
            const item = document.createElement("div");
            item.className = "lora-item";

            const header = document.createElement("div");
            header.className = "lora-item-header";

            const name = document.createElement("span");
            name.textContent = lora.lora_name;

            if (lightningProfile) {
                const profileLabel = document.createElement("span");
                profileLabel.className = "lora-profile-label";
                profileLabel.textContent = `Lightning · ${lightningProfile.steps} steps`;
                header.appendChild(profileLabel);
            } else if (Number(mixedStack?.companion.lora_id) === Number(lora.lora_id)) {
                const compatibilityLabel = document.createElement("span");
                compatibilityLabel.className = "lora-profile-label";
                compatibilityLabel.textContent = "Lightning-compatible · Qwen Image 2512";
                header.appendChild(compatibilityLabel);
            }

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "secondary lora-remove";
            remove.textContent = "Remove";
            remove.addEventListener("click", () => removeLora(lora.lora_id));

            header.append(name, remove);

            const strength = clampStrength(lora.strength);
            const unetStrength = clampStrength(lora.unet_strength ?? strength);
            const textEncoderStrength = clampStrength(lora.text_encoder_strength ?? strength);
            lora.strength = strength;
            lora.unet_strength = unetStrength;
            lora.text_encoder_strength = textEncoderStrength;

            const targetWrap = document.createElement("label");
            targetWrap.className = "lora-target";
            targetWrap.innerHTML = "<span>Target</span>";

            const targetSelect = document.createElement("select");
            [
                { value: "both", label: "UNet + Text Encoder" },
                { value: "unet", label: "UNet only" },
                { value: "text_encoder", label: "Text Encoder only" },
            ].forEach((optionData) => {
                const option = document.createElement("option");
                option.value = optionData.value;
                option.textContent = optionData.label;
                if (lora.target === optionData.value) {
                    option.selected = true;
                }
                targetSelect.appendChild(option);
            });
            targetSelect.addEventListener("change", (event) => {
                updateLoraTarget(lora.lora_id, String(event.target.value || "both"));
            });
            targetWrap.appendChild(targetSelect);

            const presets = normalizePromptPresets(lora.prompt_presets);
            let promptPresetWrap = null;
            if (presets.length > 0) {
                promptPresetWrap = document.createElement("label");
                promptPresetWrap.className = "lora-prompt-preset";
                promptPresetWrap.innerHTML = "<span>Prompt Preset</span>";

                const promptPresetSelect = document.createElement("select");
                const emptyOption = document.createElement("option");
                emptyOption.value = "";
                emptyOption.textContent = "No prompt preset";
                promptPresetSelect.appendChild(emptyOption);
                presets.forEach((preset) => {
                    const option = document.createElement("option");
                    option.value = preset.name;
                    option.textContent = preset.name;
                    if (lora.prompt_preset_name === preset.name) {
                        option.selected = true;
                    }
                    promptPresetSelect.appendChild(option);
                });
                promptPresetSelect.addEventListener("change", (event) => {
                    updateLoraPromptPreset(lora.lora_id, String(event.target.value || ""));
                });
                promptPresetWrap.appendChild(promptPresetSelect);
            }

            if (isAdvancedWeightMode()) {
                const advancedStrengthWrap = document.createElement("div");
                advancedStrengthWrap.className = "lora-strength-grid";

                const unetStrengthWrap = document.createElement("label");
                unetStrengthWrap.className = "lora-strength";
                unetStrengthWrap.innerHTML = `<span>UNet Strength: <strong>${unetStrength.toFixed(2)}</strong></span>`;

                const unetSlider = document.createElement("input");
                unetSlider.type = "range";
                unetSlider.min = "0";
                unetSlider.max = "1";
                unetSlider.step = "0.05";
                unetSlider.value = String(unetStrength);
                unetSlider.addEventListener("input", (event) => {
                    const value = Number(event.target.value);
                    updateLoraUnetStrength(lora.lora_id, value);
                });
                unetStrengthWrap.appendChild(unetSlider);

                const textStrengthWrap = document.createElement("label");
                textStrengthWrap.className = "lora-strength";
                textStrengthWrap.innerHTML = `<span>Text Encoder Strength: <strong>${textEncoderStrength.toFixed(2)}</strong></span>`;

                const textSlider = document.createElement("input");
                textSlider.type = "range";
                textSlider.min = "0";
                textSlider.max = "1";
                textSlider.step = "0.05";
                textSlider.value = String(textEncoderStrength);
                textSlider.addEventListener("input", (event) => {
                    const value = Number(event.target.value);
                    updateLoraTextEncoderStrength(lora.lora_id, value);
                });
                textStrengthWrap.appendChild(textSlider);

                advancedStrengthWrap.append(unetStrengthWrap, textStrengthWrap);
                item.append(header, advancedStrengthWrap, targetWrap);
            } else {
                const strengthWrap = document.createElement("label");
                strengthWrap.className = "lora-strength";
                const strengthLabel = isQwenImageFamily() ? "Qwen transformer" : "Strength";
                strengthWrap.innerHTML = `<span>${strengthLabel}: <strong>${strength.toFixed(2)}</strong></span>`;

                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = "0";
                slider.max = "1";
                slider.step = "0.05";
                slider.value = String(strength);
                slider.disabled = Boolean(lightningProfile);
                slider.addEventListener("input", (event) => {
                    const value = Number(event.target.value);
                    updateLoraStrength(lora.lora_id, value);
                });

                strengthWrap.appendChild(slider);
                item.append(header, strengthWrap);
                if (!isQwenImageFamily()) {
                    item.appendChild(targetWrap);
                }
            }
            if (promptPresetWrap) {
                item.appendChild(promptPresetWrap);
            }
            const managePresetButton = document.createElement("button");
            managePresetButton.type = "button";
            managePresetButton.className = "secondary lora-preset-manage";
            managePresetButton.textContent = presets.length > 0 ? "Edit Prompt Presets" : "Add Prompt Presets";
            managePresetButton.addEventListener("click", () => openPromptPresetModal(lora.lora_id));
            item.appendChild(managePresetButton);
            list.appendChild(item);
        });
        emitLightningProfileChanged({ force: forceLightningProfileEvent });
        emitAdapterSummaryChanged();
    }

    function updateLoraStrength(loraId, strength) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        target.strength = clampStrength(strength);
        target.unet_strength = target.strength;
        target.text_encoder_strength = target.strength;
        logDebug("Updated adapter strength.", { lora_id: loraId, strength: target.strength });
        renderLoraList();
    }

    function updateLoraUnetStrength(loraId, strength) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        target.unet_strength = clampStrength(strength);
        recomputeCombinedStrength(target);
        logDebug("Updated adapter UNet strength.", {
            lora_id: loraId,
            unet_strength: target.unet_strength,
            strength: target.strength,
        });
        renderLoraList();
    }

    function updateLoraTextEncoderStrength(loraId, strength) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        target.text_encoder_strength = clampStrength(strength);
        recomputeCombinedStrength(target);
        logDebug("Updated adapter text encoder strength.", {
            lora_id: loraId,
            text_encoder_strength: target.text_encoder_strength,
            strength: target.strength,
        });
        renderLoraList();
    }

    function removeLora(loraId) {
        loraState.selected = loraState.selected.filter((lora) => lora.lora_id !== loraId);
        logDebug("Removed adapter.", { lora_id: loraId, selected_count: loraState.selected.length });
        renderLoraList();
    }

    function updateLoraTarget(loraId, targetValue) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        const normalized = String(targetValue || "both").trim().toLowerCase().replace("-", "_");
        if (normalized === "unet" || normalized === "text_encoder" || normalized === "both") {
            target.target = normalized;
        } else {
            target.target = "both";
        }
        logDebug("Updated adapter target.", { lora_id: loraId, target: target.target });
        renderLoraList();
    }

    function updateLoraPromptPreset(loraId, presetName) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        const presets = normalizePromptPresets(target.prompt_presets);
        const matched = presets.find((preset) => preset.name === presetName);
        target.prompt_preset_name = matched?.name || "";
        target.prompt_preset_words = matched?.words || [];
        logDebug("Updated adapter prompt preset.", {
            lora_id: loraId,
            prompt_preset_name: target.prompt_preset_name,
            word_count: target.prompt_preset_words.length,
        });
        emitAdapterSummaryChanged();
        renderLoraList();
    }

    function updateStoredLoraEntry(updatedEntry) {
        const normalizedPresets = normalizePromptPresets(updatedEntry?.prompt_presets);
        loraState.available = loraState.available.map((entry) =>
            Number(entry.lora_id) === Number(updatedEntry.lora_id)
                ? { ...entry, ...updatedEntry, prompt_presets: normalizedPresets }
                : entry,
        );
        const updatedSelected = loraState.selected.map((selected) => {
            if (Number(selected.lora_id) !== Number(updatedEntry.lora_id)) {
                return selected;
            }
            const matchedPreset = normalizedPresets.find((preset) => preset.name === selected.prompt_preset_name);
            return {
                ...selected,
                lora_name: updatedEntry.name ?? updatedEntry.file_path ?? selected.lora_name,
                prompt_presets: normalizedPresets,
                prompt_preset_name: matchedPreset?.name || "",
                prompt_preset_words: matchedPreset?.words || [],
                runtime_profile: updatedEntry.runtime_profile ?? selected.runtime_profile ?? null,
                compatibility: updatedEntry.compatibility ?? selected.compatibility ?? null,
            };
        });
        const normalizedSelected = [];
        updatedSelected.forEach((selected) => {
            if (canSelectEntry(selected, normalizedSelected)) {
                normalizedSelected.push(normalizeSelectedLora(selected));
            }
        });
        loraState.selected = normalizedSelected;
        renderLoraList();
        emitAdapterSummaryChanged();
    }

    function ensurePromptPresetEditorScript() {
        if (window.LoraPromptPresetEditor) {
            return Promise.resolve(window.LoraPromptPresetEditor);
        }
        if (!promptPresetEditorScriptPromise) {
            promptPresetEditorScriptPromise = new Promise((resolve, reject) => {
                const script = document.createElement("script");
                script.src = resolveAssetUrl("lora_prompt_preset_editor.js?v=1");
                script.onload = () => resolve(window.LoraPromptPresetEditor);
                script.onerror = () => reject(new Error("Unable to load LoRA prompt preset editor."));
                document.head.appendChild(script);
            });
        }
        return promptPresetEditorScriptPromise;
    }

    async function openPromptPresetModal(loraId) {
        const selected = loraState.selected.find((lora) => Number(lora.lora_id) === Number(loraId));
        if (!selected) {
            return;
        }

        const overlay = document.createElement("div");
        overlay.className = "lora-preset-modal-backdrop";

        const modal = document.createElement("div");
        modal.className = "lora-preset-modal";
        modal.setAttribute("role", "dialog");
        modal.setAttribute("aria-modal", "true");

        const header = document.createElement("div");
        header.className = "lora-preset-modal-header";
        const title = document.createElement("h3");
        title.textContent = selected.lora_name || `LoRA ${loraId}`;
        const close = document.createElement("button");
        close.type = "button";
        close.className = "secondary";
        close.textContent = "Close";
        close.addEventListener("click", () => overlay.remove());
        header.append(title, close);

        const body = document.createElement("div");
        body.className = "lora-preset-modal-body";
        const loading = document.createElement("div");
        loading.className = "field-hint";
        loading.textContent = "Loading prompt preset editor...";
        body.appendChild(loading);

        modal.append(header, body);
        overlay.appendChild(modal);
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) {
                overlay.remove();
            }
        });
        document.body.appendChild(overlay);

        try {
            const editor = await ensurePromptPresetEditorScript();
            body.innerHTML = "";
            editor.mount({
                container: body,
                apiBase: loraState.apiBase,
                loraId,
                compact: true,
                onSaved: updateStoredLoraEntry,
            });
        } catch (error) {
            console.error(error);
            body.innerHTML = "";
            const message = document.createElement("div");
            message.className = "model-form-state error";
            message.textContent = error.message || "Unable to open prompt preset editor.";
            body.appendChild(message);
        }
    }

    function addLora() {
        const select = document.getElementById("lora-select");
        if (!select) {
            return;
        }
        const selectedId = Number(select.value);
        if (!Number.isFinite(selectedId)) {
            return;
        }
        const existing = loraState.selected.find((lora) => lora.lora_id === selectedId);
        if (existing) {
            return;
        }
        const entry = loraState.available.find((lora) => lora.lora_id === selectedId);
        if (!entry || !canSelectEntry(entry)) {
            return;
        }
        loraState.selected.push(normalizeSelectedLora({
            lora_id: entry.lora_id,
            lora_name: entry.name ?? entry.file_path ?? `LoRA ${entry.lora_id}`,
            prompt_presets: normalizePromptPresets(entry.prompt_presets),
            prompt_preset_name: "",
            prompt_preset_words: [],
            strength: DEFAULT_STRENGTH,
            unet_strength: DEFAULT_STRENGTH,
            text_encoder_strength: DEFAULT_STRENGTH,
            target: "both",
            runtime_profile: entry.runtime_profile ?? null,
            compatibility: entry.compatibility ?? null,
        }));
        logDebug("Added adapter.", { lora_id: entry.lora_id, selected_count: loraState.selected.length });
        renderLoraList();
    }

    async function loadLoras(apiBase, family) {
        const select = document.getElementById("lora-select");
        if (!select) {
            return;
        }
        try {
            const res = await fetch(`${apiBase}/lora-models?family=${family}`);
            const loras = await res.json();
            if (!Array.isArray(loras) || loras.length === 0) {
                throw new Error("No LoRAs returned.");
            }
            loraState.available = loras;
            const refreshedSelected = loraState.selected.map((selected) => {
                const refreshedEntry = loras.find(
                    (entry) => Number(entry.lora_id) === Number(selected.lora_id),
                );
                if (!refreshedEntry) {
                    return selected;
                }
                const promptPresets = normalizePromptPresets(refreshedEntry.prompt_presets);
                const matchedPreset = promptPresets.find(
                    (preset) => preset.name === selected.prompt_preset_name,
                );
                return {
                    ...selected,
                    lora_name: refreshedEntry.name ?? refreshedEntry.file_path ?? selected.lora_name,
                    prompt_presets: promptPresets,
                    prompt_preset_name: matchedPreset?.name || "",
                    prompt_preset_words: matchedPreset?.words || [],
                    runtime_profile: refreshedEntry.runtime_profile ?? null,
                    compatibility: refreshedEntry.compatibility ?? null,
                };
            });
            const normalizedSelected = [];
            refreshedSelected.forEach((selected) => {
                if (canSelectEntry(selected, normalizedSelected)) {
                    normalizedSelected.push(normalizeSelectedLora(selected));
                }
            });
            loraState.selected = normalizedSelected;
            logDebug("Loaded available LoRAs.", { family, count: loras.length });
        } catch (error) {
            loraState.available = [];
            console.warn("Failed to load LoRAs:", error);
        }
        renderLoraList();
    }

    function buildLoraPayload() {
        const includeAdvancedStrengths = isAdvancedWeightMode();
        const payload = loraState.selected.map((lora) => {
            const item = {
                lora_id: lora.lora_id,
                strength: clampStrength(lora.strength),
            };
            if (!isQwenImageFamily()) {
                item.target = lora.target ?? "both";
            }
            if (includeAdvancedStrengths) {
                item.unet_strength = clampStrength(lora.unet_strength ?? lora.strength);
                item.text_encoder_strength = clampStrength(
                    lora.text_encoder_strength ?? lora.strength,
                );
            }
            return item;
        });
        logDebug("Built LoRA payload.", { count: payload.length, payload });
        return payload;
    }

    function buildPromptPresetWords() {
        const words = loraState.selected.flatMap((lora) =>
            Array.isArray(lora.prompt_preset_words) ? lora.prompt_preset_words : [],
        );
        logDebug("Built LoRA prompt preset words.", { count: words.length, words });
        return words;
    }

    function setSelectedAdapters(adapters) {
        if (!Array.isArray(adapters)) {
            loraState.selected = [];
            renderLoraList({ forceLightningProfileEvent: true });
            return;
        }

        const mapped = [];
        let hasAdvancedStrength = false;
        adapters.forEach((adapter) => {
            const loraId = Number(adapter?.lora_id);
            if (!Number.isFinite(loraId)) {
                return;
            }

            const matched = loraState.available.find((entry) => Number(entry.lora_id) === loraId);
            const strengthRaw = Number(adapter?.strength);
            const unetStrengthRaw = Number(adapter?.unet_strength);
            const textEncoderStrengthRaw = Number(adapter?.text_encoder_strength);
            const hasPerComponentStrength =
                Number.isFinite(unetStrengthRaw) || Number.isFinite(textEncoderStrengthRaw);
            if (hasPerComponentStrength) {
                hasAdvancedStrength = true;
            }
            const fallbackStrength = Number.isFinite(strengthRaw)
                ? clampStrength(strengthRaw)
                : DEFAULT_STRENGTH;
            const unetStrength = Number.isFinite(unetStrengthRaw)
                ? clampStrength(unetStrengthRaw)
                : fallbackStrength;
            const textEncoderStrength = Number.isFinite(textEncoderStrengthRaw)
                ? clampStrength(textEncoderStrengthRaw)
                : fallbackStrength;
            const strength = Number.isFinite(strengthRaw)
                ? clampStrength(strengthRaw)
                : clampStrength((unetStrength + textEncoderStrength) / 2);
            const targetRaw = String(adapter?.target ?? "both").trim().toLowerCase().replace("-", "_");
            const target = targetRaw === "unet" || targetRaw === "text_encoder" ? targetRaw : "both";
            const selected = {
                lora_id: loraId,
                lora_name:
                    matched?.name ??
                    matched?.file_path ??
                    adapter?.lora_name ??
                    `LoRA ${loraId}`,
                prompt_presets: normalizePromptPresets(matched?.prompt_presets),
                prompt_preset_name: "",
                prompt_preset_words: [],
                strength,
                unet_strength: unetStrength,
                text_encoder_strength: textEncoderStrength,
                target,
                runtime_profile: matched?.runtime_profile ?? adapter?.runtime_profile ?? null,
                compatibility: matched?.compatibility ?? adapter?.compatibility ?? null,
            };
            if (canSelectEntry(selected, mapped)) {
                mapped.push(normalizeSelectedLora(selected));
            }
        });
        loraState.selected = mapped;
        loraState.weightMode = hasAdvancedStrength && supportsAdvancedWeights() ? "advanced" : "basic";
        logDebug("Hydrated selected adapters.", { count: mapped.length, adapters: mapped });
        renderLoraList({ forceLightningProfileEvent: true });
    }

    async function initLoraUI({ apiBase, family, taskType = "" }) {
        const container = document.getElementById("lora-panel-root");
        if (!container) {
            return;
        }
        try {
            const res = await fetch(resolveAssetUrl("lora_panel.html?v=2"));
            if (!res.ok) {
                throw new Error(`Failed to load LoRA panel UI: ${res.status}`);
            }
            container.innerHTML = await res.text();
        } catch (error) {
            console.warn("Failed to load LoRA panel UI:", error);
            return;
        }

        const toggleButton = document.getElementById("lora-toggle");
        const addButton = document.getElementById("add-lora");
        loraState.family = normalizeFamily(family);
        loraState.taskType = String(taskType || "").trim().toLowerCase();
        loraState.weightMode = "basic";
        loraState.apiBase = apiBase;
        toggleButton?.addEventListener("click", toggleLoraPanel);
        addButton?.addEventListener("click", addLora);
        bindWeightModeControls();
        syncWeightModeUI();
        await loadLoras(apiBase, family);
        renderLoraList();
    }

    window.LoraPanel = {
        init: initLoraUI,
        getSelectedAdapters: buildLoraPayload,
        getSelectedPresetWords: buildPromptPresetWords,
        setSelectedAdapters,
        getSummary,
    };
})();
