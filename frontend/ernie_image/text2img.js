function countLabel(count, singular, plural = `${singular}s`) {
    const value = Number.isFinite(Number(count)) ? Number(count) : 0;
    return `${value} ${value === 1 ? singular : plural}`;
}

function updateAdapterSummary() {
    const summary = window.LoraPanel?.getSummary?.() ?? { available: 0, selected: 0 };
    const available = Number(summary.available ?? 0);
    const selected = Number(summary.selected ?? 0);
    const setText = (id, text) => {
        const element = document.getElementById(id);
        if (element) element.textContent = text;
    };
    setText("adapter_summary_label", `${countLabel(available, "adapter available", "adapters available")} / ${countLabel(selected, "adapter active", "adapters active")}`);
    setText("adapter-tab-lora-badge", countLabel(selected, "selected", "selected"));
    setText("adapter-overview-lora-count", countLabel(available, "LoRA available", "LoRAs available"));
    setText("adapter-overview-lora-detail", selected ? `${countLabel(selected, "LoRA")} selected.` : "No LoRAs selected.");
}

function setAdapterTab(name) {
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
        const active = tab.dataset.adapterTab === name;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("[data-adapter-panel]").forEach((panel) => {
        const active = panel.dataset.adapterPanel === name;
        panel.classList.toggle("is-active", active);
        panel.toggleAttribute("hidden", !active);
    });
    updateAdapterSummary();
}

function setAdapterModalOpen(open) {
    const modal = document.getElementById("adapter-modal");
    if (!modal) return;
    modal.classList.toggle("hidden", !open);
    modal.setAttribute("aria-hidden", String(!open));
    document.getElementById(open ? "adapter-modal-close" : "adapter-modal-open")?.focus();
    if (open) updateAdapterSummary();
}

function initAdapterModal() {
    const modal = document.getElementById("adapter-modal");
    if (!modal) return;
    const subtitle = document.getElementById("adapter-modal-subtitle");
    if (subtitle) subtitle.textContent = "ERNIE-Image adapter stack";
    for (const name of ["controlnet", "ipadapter"]) {
        document.querySelector(`[data-adapter-tab="${name}"]`)?.remove();
        document.querySelector(`[data-adapter-panel="${name}"]`)?.remove();
        document.querySelector(`[data-adapter-tab-jump="${name}"]`)?.remove();
    }
    document.getElementById("adapter-modal-open")?.addEventListener("click", () => setAdapterModalOpen(true));
    document.getElementById("adapter-modal-close")?.addEventListener("click", () => setAdapterModalOpen(false));
    document.getElementById("adapter-modal-overlay")?.addEventListener("click", () => setAdapterModalOpen(false));
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !modal.classList.contains("hidden")) setAdapterModalOpen(false);
    });
    document.querySelectorAll("[data-adapter-tab]").forEach((tab) =>
        tab.addEventListener("click", () => setAdapterTab(tab.dataset.adapterTab)));
    document.querySelectorAll("[data-adapter-tab-jump]").forEach((button) =>
        button.addEventListener("click", () => setAdapterTab(button.dataset.adapterTabJump)));
    window.addEventListener("adapter-summary-changed", updateAdapterSummary);
    modal.addEventListener("change", () => setTimeout(updateAdapterSummary, 0));
    modal.addEventListener("click", () => setTimeout(updateAdapterSummary, 0));
    updateAdapterSummary();
}

window.AdapterPanel?.render?.();
initAdapterModal();

const page = GenerationPage.create({
    family: "ernie-image",
    taskType: "ernie-image.text2img",
    loraEnvelope: false,
    fallbackModel: { value: "ERNIE-Image-Turbo", label: "ERNIE-Image-Turbo (hub, diffusers)" },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 8 },
        { element: "cfg", key: "guidance_scale", type: "number", fallback: 1.0 },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 768 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 768 },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "use_pe", key: "use_pe", type: "checkbox", fallback: false },
        { element: "load_pe", key: "load_pe", type: "checkbox", fallback: false },
        { element: "memory_preset", key: "memory_preset", fallback: "sequential_offload" },
    ],
});

async function generate() {
    const inputs = page.withLora(page.collectSettings(await page.defaults()));
    await page.run(inputs, "Failed to generate ERNIE-Image images:");
}
