(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);

    async function loadSchedulerPanel() {
        const container = document.getElementById("scheduler-panel-root");
        if (!container) {
            return;
        }
        try {
            const res = await fetch(resolveAssetUrl("scheduler_panel.html"));
            if (!res.ok) {
                throw new Error(`Failed to load scheduler UI: ${res.status}`);
            }
            container.innerHTML = await res.text();
        } catch (error) {
            console.warn("Failed to load scheduler UI:", error);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", loadSchedulerPanel);
    } else {
        loadSchedulerPanel();
    }
})();
