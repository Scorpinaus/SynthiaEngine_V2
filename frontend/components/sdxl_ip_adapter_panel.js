(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    const DEFAULT_IP_ADAPTER_TOGGLE_LABEL = "Use image prompt reference";
    let loadPromise = null;

    async function loadSdxlIpAdapterPanel() {
        const container = document.getElementById("sdxl-ip-adapter-panel-root");
        if (!container || container.dataset.sdxlIpAdapterLoaded === "true") {
            return;
        }
        try {
            const res = await fetch(resolveAssetUrl("sdxl_ip_adapter_panel.html?v=1"), { cache: "no-store" });
            if (!res.ok) {
                throw new Error(`Failed to load SDXL IP-Adapter UI: ${res.status}`);
            }
            const ipAdapterToggleLabel =
                container.dataset.ipAdapterToggleLabel?.trim() || DEFAULT_IP_ADAPTER_TOGGLE_LABEL;
            const markup = (await res.text()).replace("{{IP_ADAPTER_TOGGLE_LABEL}}", ipAdapterToggleLabel);
            container.innerHTML = markup;
            container.dataset.sdxlIpAdapterLoaded = "true";
            document.dispatchEvent(new CustomEvent("sdxl-ip-adapter-panel:loaded"));
        } catch (error) {
            console.warn("Failed to load SDXL IP-Adapter UI:", error);
        }
    }

    function ensureLoaded() {
        if (!loadPromise) {
            loadPromise = loadSdxlIpAdapterPanel();
        }
        return loadPromise;
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", ensureLoaded, { once: true });
    } else {
        ensureLoaded();
    }

    window.SdxlIpAdapterPanel = {
        load: ensureLoaded,
    };
})();
