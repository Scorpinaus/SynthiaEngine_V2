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
            const defaultScheduler = container.dataset.defaultScheduler;
            const allowedSchedulers = (container.dataset.allowedSchedulers || "")
                .split(",")
                .map((value) => value.trim())
                .filter(Boolean);
            const schedulerSelect = container.querySelector("#scheduler");
            if (schedulerSelect && allowedSchedulers.length) {
                Array.from(schedulerSelect.options).forEach((option) => {
                    if (!allowedSchedulers.includes(option.value)) {
                        option.remove();
                    }
                });
            }
            if (defaultScheduler && schedulerSelect) {
                schedulerSelect.value = defaultScheduler;
            }
            if (schedulerSelect?.options.length === 1) {
                schedulerSelect.disabled = true;
                schedulerSelect.title = "This runtime profile uses a fixed scheduler.";
            }
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
