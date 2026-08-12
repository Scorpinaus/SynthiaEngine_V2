/** Shared adapter modal controller for SD1.5 and SDXL generation pages. */
(function () {
    function countLabel(count, singular, plural = `${singular}s`) {
        const value = Number(count);
        const safeCount = Number.isFinite(value) ? value : 0;
        return `${safeCount} ${safeCount === 1 ? singular : plural}`;
    }

    function setText(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) element.textContent = value;
    }

    function create({ subtitle = null, adjustControlSummary = null } = {}) {
        function summaries() {
            const controlState = window.ControlNetPanel?.getState?.() ?? null;
            const selectedLoras = window.LoraPanel?.getSelectedAdapters?.() ?? [];
            const baseControl = window.ControlNetPanel?.getSummary?.() ?? {
                availablePreprocessors: controlState?.preprocessors?.size ?? 0,
                totalPreprocessors: controlState?.preprocessors?.size ?? 0,
                activeItems: controlState?.controlItems?.length ?? 0,
                enabled: Boolean(document.getElementById("controlnet-enabled")?.checked),
            };
            const control = typeof adjustControlSummary === "function"
                ? adjustControlSummary({ ...baseControl })
                : baseControl;
            const lora = window.LoraPanel?.getSummary?.() ?? {
                available: 0,
                selected: Array.isArray(selectedLoras) ? selectedLoras.length : 0,
            };
            const ipAdapter = window.IpAdapterPanel?.getSummary?.() ?? {
                availableAdapters: 1,
                enabled: Boolean(document.getElementById("ip_adapter_enabled")?.checked),
                hasReference: Boolean(document.getElementById("ip_adapter_image")?.files?.[0]),
                hasMask: Boolean(window.IpAdapterPanel?.getMaskFile?.()),
            };
            return { control, ipAdapter, lora };
        }

        function update() {
            const { control, ipAdapter, lora } = summaries();
            const availableControl = Number(control.availablePreprocessors ?? control.totalPreprocessors ?? 0);
            const activeControl = Number(control.activeItems ?? 0);
            const availableLoras = Number(lora.available ?? 0);
            const selectedLoras = Number(lora.selected ?? 0);
            const availableIpAdapters = Number(ipAdapter.availableAdapters ?? 1);
            const ipActive = Boolean(ipAdapter.enabled);
            const totalAvailable =
                (Number.isFinite(availableControl) ? availableControl : 0) +
                (Number.isFinite(availableLoras) ? availableLoras : 0) +
                (Number.isFinite(availableIpAdapters) ? availableIpAdapters : 0);
            const totalActive =
                (control.enabled && activeControl > 0 ? activeControl : 0) +
                (Number.isFinite(selectedLoras) ? selectedLoras : 0) +
                (ipActive ? 1 : 0);

            setText(
                "adapter_summary_label",
                `${countLabel(totalAvailable, "adapter available", "adapters available")} / ${countLabel(totalActive, "adapter active", "adapters active")}`
            );
            setText("adapter-tab-controlnet-badge", countLabel(activeControl, "active", "active"));
            setText("adapter-tab-lora-badge", countLabel(selectedLoras, "selected", "selected"));
            setText("adapter-tab-ipadapter-badge", ipActive ? "on" : "off");
            setText(
                "adapter-overview-controlnet-count",
                countLabel(availableControl, "preprocessor available", "preprocessors available")
            );
            setText(
                "adapter-overview-controlnet-detail",
                control.enabled && activeControl > 0
                    ? `${countLabel(activeControl, "control image")} active.`
                    : "No control images active."
            );
            setText("adapter-overview-lora-count", countLabel(availableLoras, "LoRA available", "LoRAs available"));
            setText(
                "adapter-overview-lora-detail",
                selectedLoras > 0 ? `${countLabel(selectedLoras, "LoRA")} selected.` : "No LoRAs selected."
            );
            setText(
                "adapter-overview-ipadapter-count",
                countLabel(availableIpAdapters, "IP-Adapter available", "IP-Adapters available")
            );
            setText(
                "adapter-overview-ipadapter-detail",
                ipActive
                    ? `Image prompt enabled${ipAdapter.hasReference ? " with reference image" : ""}${ipAdapter.hasMask ? " and mask" : ""}.`
                    : "Image prompt disabled."
            );
        }

        function selectTab(tabName) {
            const target = String(tabName || "overview");
            document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
                const active = tab.getAttribute("data-adapter-tab") === target;
                tab.classList.toggle("is-active", active);
                tab.setAttribute("aria-selected", String(active));
            });
            document.querySelectorAll("[data-adapter-panel]").forEach((panel) => {
                const active = panel.getAttribute("data-adapter-panel") === target;
                panel.classList.toggle("is-active", active);
                panel.toggleAttribute("hidden", !active);
            });
            update();
        }

        function setOpen(open) {
            const modal = document.getElementById("adapter-modal");
            if (!modal) return;
            modal.classList.toggle("hidden", !open);
            modal.setAttribute("aria-hidden", String(!open));
            if (open) {
                update();
                document.getElementById("adapter-modal-close")?.focus();
            } else {
                document.getElementById("adapter-modal-open")?.focus();
            }
        }

        function init() {
            const modal = document.getElementById("adapter-modal");
            if (!modal) return;
            window.AdapterPanel?.render?.();
            if (subtitle) setText("adapter-modal-subtitle", subtitle);
            document.getElementById("adapter-modal-open")?.addEventListener("click", () => setOpen(true));
            document.getElementById("adapter-modal-close")?.addEventListener("click", () => setOpen(false));
            document.getElementById("adapter-modal-overlay")?.addEventListener("click", () => setOpen(false));
            document.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && !modal.classList.contains("hidden")) setOpen(false);
            });
            document.querySelectorAll("[data-adapter-tab]").forEach((tab) => {
                tab.addEventListener("click", () => selectTab(tab.getAttribute("data-adapter-tab")));
            });
            document.querySelectorAll("[data-adapter-tab-jump]").forEach((button) => {
                button.addEventListener("click", () => selectTab(button.getAttribute("data-adapter-tab-jump")));
            });
            window.addEventListener("adapter-summary-changed", update);
            modal.addEventListener("change", () => window.setTimeout(update, 0));
            modal.addEventListener("click", () => window.setTimeout(update, 0));
            update();
        }

        return { init, update };
    }

    window.AdapterController = { create };
})();
