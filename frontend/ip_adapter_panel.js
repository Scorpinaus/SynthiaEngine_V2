(() => {
    let previewUrl = null;
    let didInit = false;

    function revokePreviewUrl() {
        if (!previewUrl) {
            return;
        }
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    function init() {
        if (didInit) {
            return;
        }

        const panel = document.getElementById("ip_adapter_panel");
        const toggle = document.getElementById("ip_adapter_toggle");
        const content = document.getElementById("ip_adapter_content");
        const chevron = document.getElementById("ip_adapter_chevron");
        const enabled = document.getElementById("ip_adapter_enabled");
        const input = document.getElementById("ip_adapter_image");
        const preview = document.getElementById("ip_adapter_preview");
        const empty = document.getElementById("ip_adapter_preview_empty");

        if (!panel || !toggle || !content) {
            return;
        }
        didInit = true;

        function setOpen(isOpen) {
            content.classList.toggle("is-open", isOpen);
            toggle.setAttribute("aria-expanded", String(isOpen));
            if (chevron) {
                chevron.textContent = isOpen ? "\u25b4" : "\u25be";
            }
        }

        function clearPreview() {
            revokePreviewUrl();
            if (preview) {
                preview.removeAttribute("src");
                preview.classList.add("is-hidden");
            }
            empty?.classList.remove("is-hidden");
        }

        function updatePreview() {
            const file = input?.files?.[0] ?? null;
            clearPreview();
            if (!file || !String(file.type || "").startsWith("image/")) {
                return;
            }

            previewUrl = URL.createObjectURL(file);
            if (preview) {
                preview.src = previewUrl;
                preview.classList.remove("is-hidden");
            }
            empty?.classList.add("is-hidden");
            setOpen(true);
        }

        toggle.addEventListener("click", () => {
            setOpen(!content.classList.contains("is-open"));
        });
        enabled?.addEventListener("change", () => {
            if (enabled.checked) {
                setOpen(true);
            }
        });
        input?.addEventListener("change", updatePreview);
        window.addEventListener("beforeunload", revokePreviewUrl);

        updatePreview();
    }

    window.IpAdapterPanel = { init };
})();
