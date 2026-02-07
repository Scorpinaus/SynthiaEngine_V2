(() => {
    const state = {
        previewUrl: null,
        previewBlob: null,
        preprocessors: new Map(),
    };

    function getState() {
        return state;
    }

    function updateIndicator() {
        const indicator = document.getElementById("controlnet-indicator");
        const enabledToggle = document.getElementById("controlnet-enabled");
        const status = document.getElementById("controlnet-status");
        const isActive = Boolean(enabledToggle?.checked && state.previewUrl);
        if (indicator) {
            indicator.classList.toggle("is-active", isActive);
        }
        if (status) {
            status.textContent = isActive
                ? "ControlNet preprocessor ready for SD1.5."
                : "No preprocessor applied.";
        }
    }

    function updateActiveFlag() {
        const flag = document.getElementById("controlnet-active-flag");
        if (!flag) {
            return;
        }
        const enabledToggle = document.getElementById("controlnet-enabled");
        const isActive = Boolean(enabledToggle?.checked && state.previewUrl);
        flag.classList.toggle("is-hidden", !isActive);
        flag.style.display = isActive ? "inline-flex" : "none";
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
            const res = await fetch("controlnet_panel.html?v=2", { cache: "no-store" });
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
        updateIndicator,
        updateActiveFlag,
        togglePanel,
        loadPanel,
    };
})();
