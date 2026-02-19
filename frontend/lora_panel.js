(() => {
    const LOG_LORA_PANEL = true;
    const DEFAULT_STRENGTH = 0.8;

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
        weightMode: "basic",
    };

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

    function normalizeWeightMode(mode) {
        return String(mode || "").trim().toLowerCase() === "advanced" ? "advanced" : "basic";
    }

    function isAdvancedWeightMode() {
        return supportsAdvancedWeights() && loraState.weightMode === "advanced";
    }

    function recomputeCombinedStrength(adapter) {
        adapter.strength = clampStrength((adapter.unet_strength + adapter.text_encoder_strength) / 2);
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

    function renderLoraList() {
        syncWeightModeUI();
        const list = document.getElementById("lora-list");
        const emptyState = document.getElementById("lora-empty");
        if (!list || !emptyState) {
            return;
        }
        list.innerHTML = "";
        if (loraState.selected.length === 0) {
            emptyState.classList.remove("is-hidden");
            return;
        }
        emptyState.classList.add("is-hidden");
        loraState.selected.forEach((lora) => {
            const item = document.createElement("div");
            item.className = "lora-item";

            const header = document.createElement("div");
            header.className = "lora-item-header";

            const name = document.createElement("span");
            name.textContent = lora.lora_name;

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
                strengthWrap.innerHTML = `<span>Strength: <strong>${strength.toFixed(2)}</strong></span>`;

                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = "0";
                slider.max = "1";
                slider.step = "0.05";
                slider.value = String(strength);
                slider.addEventListener("input", (event) => {
                    const value = Number(event.target.value);
                    updateLoraStrength(lora.lora_id, value);
                });

                strengthWrap.appendChild(slider);
                item.append(header, strengthWrap, targetWrap);
            }
            list.appendChild(item);
        });
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
        if (!entry) {
            return;
        }
        loraState.selected.push({
            lora_id: entry.lora_id,
            lora_name: entry.name ?? entry.file_path ?? `LoRA ${entry.lora_id}`,
            strength: DEFAULT_STRENGTH,
            unet_strength: DEFAULT_STRENGTH,
            text_encoder_strength: DEFAULT_STRENGTH,
            target: "both",
        });
        logDebug("Added adapter.", { lora_id: entry.lora_id, selected_count: loraState.selected.length });
        renderLoraList();
    }

    async function loadLoras(apiBase, family) {
        const select = document.getElementById("lora-select");
        if (!select) {
            return;
        }
        select.innerHTML = "";
        try {
            const res = await fetch(`${apiBase}/lora-models?family=${family}`);
            const loras = await res.json();
            if (!Array.isArray(loras) || loras.length === 0) {
                throw new Error("No LoRAs returned.");
            }
            loraState.available = loras;
            logDebug("Loaded available LoRAs.", { family, count: loras.length });
            loras.forEach((lora, index) => {
                const option = document.createElement("option");
                option.value = String(lora.lora_id);
                const name = lora.name ?? lora.file_path ?? `LoRA ${lora.lora_id}`;
                option.textContent = name;
                if (index === 0) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
        } catch (error) {
            const fallback = document.createElement("option");
            fallback.value = "";
            fallback.textContent = "No LoRAs available";
            fallback.selected = true;
            select.appendChild(fallback);
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
                target: lora.target ?? "both",
            };
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

    function setSelectedAdapters(adapters) {
        if (!Array.isArray(adapters)) {
            loraState.selected = [];
            renderLoraList();
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
            mapped.push({
                lora_id: loraId,
                lora_name:
                    matched?.name ??
                    matched?.file_path ??
                    adapter?.lora_name ??
                    `LoRA ${loraId}`,
                strength,
                unet_strength: unetStrength,
                text_encoder_strength: textEncoderStrength,
                target,
            });
        });
        loraState.selected = mapped;
        loraState.weightMode = hasAdvancedStrength && supportsAdvancedWeights() ? "advanced" : "basic";
        logDebug("Hydrated selected adapters.", { count: mapped.length, adapters: mapped });
        renderLoraList();
    }

    async function initLoraUI({ apiBase, family }) {
        const container = document.getElementById("lora-panel-root");
        if (!container) {
            return;
        }
        try {
            const res = await fetch("lora_panel.html");
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
        loraState.weightMode = "basic";
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
        setSelectedAdapters,
    };
})();
