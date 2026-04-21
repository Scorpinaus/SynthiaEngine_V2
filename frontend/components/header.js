(() => {
    const scriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
    const resolveAssetUrl = (path) => (scriptUrl ? new URL(path, scriptUrl).toString() : path);
    const fallbackMarkup = `
<header class="header">
    <h1>Synthia Engine</h1>
    <div id="nav-root"></div>
</header>
`.trim();

    async function loadHeader() {
        const container = document.getElementById("header-root");
        if (!container) {
            return;
        }
        let markup = fallbackMarkup;
        try {
            const res = await fetch(resolveAssetUrl("header.html"));
            if (!res.ok) {
                throw new Error(`Failed to load header UI: ${res.status}`);
            }
            markup = await res.text();
        } catch (error) {
            console.warn("Failed to load header UI:", error);
        }
        container.innerHTML = markup;
        document.dispatchEvent(new CustomEvent("header:loaded"));
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", loadHeader);
    } else {
        loadHeader();
    }
})();
