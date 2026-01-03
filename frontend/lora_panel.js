(() => {
    const loraState = {
        available: [],
        selected: [],
    };

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

            const strengthWrap = document.createElement("label");
            strengthWrap.className = "lora-strength";
            strengthWrap.innerHTML = `<span>Strength: <strong>${lora.strength.toFixed(2)}</strong></span>`;

            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0";
            slider.max = "1";
            slider.step = "0.05";
            slider.value = String(lora.strength);
            slider.addEventListener("input", (event) => {
                const value = Number(event.target.value);
                updateLoraStrength(lora.lora_id, value);
            });

            strengthWrap.appendChild(slider);

            item.append(header, strengthWrap);
            list.appendChild(item);
        });
    }

    function updateLoraStrength(loraId, strength) {
        const target = loraState.selected.find((lora) => lora.lora_id === loraId);
        if (!target) {
            return;
        }
        target.strength = Math.max(0, Math.min(1, strength));
        renderLoraList();
    }

    function removeLora(loraId) {
        loraState.selected = loraState.selected.filter((lora) => lora.lora_id !== loraId);
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
            strength: 0.8,
        });
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
        return loraState.selected.map((lora) => ({
            lora_id: lora.lora_id,
            strength: lora.strength,
        }));
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
        toggleButton?.addEventListener("click", toggleLoraPanel);
        addButton?.addEventListener("click", addLora);
        await loadLoras(apiBase, family);
        renderLoraList();
    }

    window.LoraPanel = {
        init: initLoraUI,
        getSelectedAdapters: buildLoraPayload,
    };
})();
