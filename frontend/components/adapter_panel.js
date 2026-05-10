(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    const DEFAULT_IP_ADAPTER_TOGGLE_LABEL = "Use image prompt reference";
    let markup = null;

    function loadMarkup() {
        if (markup !== null) {
            return markup;
        }

        const request = new XMLHttpRequest();
        request.open("GET", resolveAssetUrl("adapter_panel.html?v=1"), false);
        request.send();
        if (request.status >= 400) {
            throw new Error(`Failed to load adapter panel UI: ${request.status}`);
        }
        markup = request.responseText.trim();
        return markup;
    }

    function getConfiguredMarkup(container) {
        const ipAdapterToggleLabel =
            container?.dataset?.ipAdapterToggleLabel?.trim() || DEFAULT_IP_ADAPTER_TOGGLE_LABEL;
        return loadMarkup().replace("{{IP_ADAPTER_TOGGLE_LABEL}}", ipAdapterToggleLabel);
    }

    function renderAdapterPanel() {
        const container = document.getElementById("adapter-panel-root");
        if (!container || container.dataset.adapterPanelLoaded === "true") {
            return;
        }

        try {
            container.innerHTML = getConfiguredMarkup(container);
        } catch (error) {
            console.warn("Failed to load adapter panel UI:", error);
            return;
        }
        if (container.dataset.ipAdapterMaskEnabled === "false") {
            container
                .querySelectorAll("[data-ip-adapter-mask-section]")
                .forEach((element) => element.remove());
        }
        container.dataset.adapterPanelLoaded = "true";
        initAdapterModalOpenState();
        document.dispatchEvent(new CustomEvent("adapter-panel:loaded"));
    }

    function initAdapterModalOpenState() {
        const modal = document.getElementById("adapter-modal");
        if (!modal || modal.dataset.openStateObserverLoaded === "true") {
            return;
        }

        const syncOpenState = () => {
            document.body.classList.toggle(
                "adapter-modal-open",
                !modal.classList.contains("hidden")
            );
        };

        syncOpenState();
        new MutationObserver(syncOpenState).observe(modal, {
            attributes: true,
            attributeFilter: ["class"],
        });
        modal.dataset.openStateObserverLoaded = "true";
    }

    renderAdapterPanel();
    if (document.readyState === "loading" && !document.getElementById("adapter-panel-root")) {
        document.addEventListener("DOMContentLoaded", renderAdapterPanel, { once: true });
    }

    window.AdapterPanel = {
        render: renderAdapterPanel,
    };
})();
