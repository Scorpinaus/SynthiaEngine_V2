(() => {
    let previewUrl = null;
    let maskPreviewUrl = null;
    let maskBlob = null;
    let modal = null;
    let modalRefs = null;
    let activeOptions = {};
    let displayScale = 1;
    let isDrawing = false;
    let didInit = false;

    function emitAdapterSummaryChanged() {
        window.dispatchEvent(new CustomEvent("adapter-summary-changed", { detail: { panel: "ip_adapter" } }));
    }

    function getSummary() {
        const enabled = Boolean(document.getElementById("ip_adapter_enabled")?.checked);
        const imageInput = document.getElementById("ip_adapter_image");
        const maskInput = document.getElementById("ip_adapter_mask_image");
        const scale = Number(document.getElementById("ip_adapter_scale")?.value);
        return {
            availableAdapters: 1,
            enabled,
            hasReference: Boolean(imageInput?.files?.[0]),
            hasMask: Boolean(maskBlob || maskInput?.files?.[0]),
            scale: Number.isFinite(scale) ? scale : null,
        };
    }

    function revokePreviewUrl() {
        if (!previewUrl) {
            return;
        }
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    function revokeMaskPreviewUrl() {
        if (!maskPreviewUrl) {
            return;
        }
        URL.revokeObjectURL(maskPreviewUrl);
        maskPreviewUrl = null;
    }

    function getMaskFile() {
        const input = document.getElementById("ip_adapter_mask_image");
        return input?.files?.[0] ?? maskBlob;
    }

    function readPositiveInteger(elementId, fallback) {
        const value = Number(document.getElementById(elementId)?.value);
        return Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
    }

    function getEditorSize() {
        if (typeof activeOptions.getMaskSize === "function") {
            const size = activeOptions.getMaskSize();
            const width = Number(size?.width);
            const height = Number(size?.height);
            if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
                return { width: Math.round(width), height: Math.round(height) };
            }
        }
        return {
            width: readPositiveInteger("width", 512),
            height: readPositiveInteger("height", 512),
        };
    }

    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            if (!file || !String(file.type || "").startsWith("image/")) {
                resolve(null);
                return;
            }
            const url = URL.createObjectURL(file);
            const image = new Image();
            image.onload = () => {
                URL.revokeObjectURL(url);
                resolve(image);
            };
            image.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("Unable to load image file."));
            };
            image.src = url;
        });
    }

    function getCanvasPosition(event) {
        const maskCanvas = modalRefs?.maskCanvas;
        const rect = maskCanvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (maskCanvas.width / rect.width),
            y: (event.clientY - rect.top) * (maskCanvas.height / rect.height),
        };
    }

    function drawAt(position) {
        const { maskCanvas, eraseToggle, brushSizeInput } = modalRefs;
        const context = maskCanvas.getContext("2d");
        const brushSize = Number(brushSizeInput.value) || 32;
        context.fillStyle = eraseToggle.checked ? "#000000" : "#ffffff";
        context.beginPath();
        context.arc(position.x, position.y, brushSize / 2, 0, Math.PI * 2);
        context.fill();
    }

    function setModalOpen(isOpen) {
        modal?.classList.toggle("hidden", !isOpen);
    }

    function ensureMaskModal() {
        if (modal) {
            return;
        }

        modal = document.createElement("div");
        modal.id = "ip_adapter_mask_modal";
        modal.className = "modal hidden";
        modal.setAttribute("role", "dialog");
        modal.setAttribute("aria-modal", "true");
        modal.innerHTML = `
            <div class="modal-overlay" data-ip-adapter-mask-close></div>
            <div class="modal-content">
                <div class="modal-header">
                    <div>
                        <h2>IP-Adapter Mask</h2>
                        <p class="modal-subtitle">White applies the image prompt. Black suppresses it.</p>
                    </div>
                    <button id="ip_adapter_mask_editor_close" class="secondary" type="button">Close</button>
                </div>
                <div class="modal-toolbar">
                    <label class="field">
                        <span>Brush size</span>
                        <input id="ip_adapter_mask_brush_size" type="number" min="1" max="256" value="48" />
                    </label>
                    <label class="field inline-field">
                        <input id="ip_adapter_mask_erase" type="checkbox" />
                        <span>Erase to black</span>
                    </label>
                    <label class="field">
                        <span>Zoom</span>
                        <input id="ip_adapter_mask_zoom" type="range" min="25" max="200" step="5" value="100" />
                    </label>
                </div>
                <div id="ip_adapter_mask_info" class="field-hint"></div>
                <div class="canvas-stack">
                    <div id="ip_adapter_mask_canvas_scroll" class="canvas-scroll">
                        <canvas id="ip_adapter_mask_base_canvas" class="mask-canvas"></canvas>
                        <canvas id="ip_adapter_mask_canvas" class="mask-canvas"></canvas>
                    </div>
                </div>
                <div class="field-row">
                    <button id="ip_adapter_mask_save" class="primary" type="button">Save mask</button>
                    <button id="ip_adapter_mask_clear_canvas" class="secondary" type="button">Clear to black</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        modalRefs = {
            baseCanvas: modal.querySelector("#ip_adapter_mask_base_canvas"),
            maskCanvas: modal.querySelector("#ip_adapter_mask_canvas"),
            canvasScroll: modal.querySelector("#ip_adapter_mask_canvas_scroll"),
            info: modal.querySelector("#ip_adapter_mask_info"),
            brushSizeInput: modal.querySelector("#ip_adapter_mask_brush_size"),
            eraseToggle: modal.querySelector("#ip_adapter_mask_erase"),
            zoomInput: modal.querySelector("#ip_adapter_mask_zoom"),
        };

        modal.querySelector("#ip_adapter_mask_editor_close")?.addEventListener("click", () => setModalOpen(false));
        modal.querySelector("[data-ip-adapter-mask-close]")?.addEventListener("click", () => setModalOpen(false));
        modal.querySelector("#ip_adapter_mask_save")?.addEventListener("click", saveCanvasMask);
        modal.querySelector("#ip_adapter_mask_clear_canvas")?.addEventListener("click", clearCanvasMask);
        modalRefs.zoomInput?.addEventListener("input", applyCanvasDisplaySize);

        const maskCanvas = modalRefs.maskCanvas;
        maskCanvas.addEventListener("pointerdown", (event) => {
            isDrawing = true;
            maskCanvas.setPointerCapture(event.pointerId);
            drawAt(getCanvasPosition(event));
        });
        maskCanvas.addEventListener("pointermove", (event) => {
            if (isDrawing) {
                drawAt(getCanvasPosition(event));
            }
        });
        maskCanvas.addEventListener("pointerup", () => {
            isDrawing = false;
        });
        maskCanvas.addEventListener("pointerleave", () => {
            isDrawing = false;
        });
    }

    function applyCanvasDisplaySize() {
        const { baseCanvas, maskCanvas, canvasScroll } = modalRefs;
        const availableWidth = modal.querySelector(".canvas-stack")?.clientWidth || baseCanvas.width;
        const maxHeight = Math.round(window.innerHeight * 0.58);
        const fitScale = Math.min(1, availableWidth / baseCanvas.width, maxHeight / baseCanvas.height);
        const zoomScale = Number(modalRefs.zoomInput.value) / 100;
        displayScale = fitScale * zoomScale;
        const displayWidth = Math.max(1, Math.round(baseCanvas.width * displayScale));
        const displayHeight = Math.max(1, Math.round(baseCanvas.height * displayScale));

        canvasScroll.style.width = `${displayWidth}px`;
        canvasScroll.style.height = `${displayHeight}px`;
        baseCanvas.style.width = `${displayWidth}px`;
        baseCanvas.style.height = `${displayHeight}px`;
        maskCanvas.style.width = `${displayWidth}px`;
        maskCanvas.style.height = `${displayHeight}px`;
    }

    function clearCanvasMask() {
        const context = modalRefs.maskCanvas.getContext("2d");
        context.fillStyle = "#000000";
        context.fillRect(0, 0, modalRefs.maskCanvas.width, modalRefs.maskCanvas.height);
    }

    async function drawExistingMask() {
        const file = getMaskFile();
        const image = await loadImageFromFile(file);
        if (!image) {
            clearCanvasMask();
            return;
        }
        const context = modalRefs.maskCanvas.getContext("2d");
        context.clearRect(0, 0, modalRefs.maskCanvas.width, modalRefs.maskCanvas.height);
        context.drawImage(image, 0, 0, modalRefs.maskCanvas.width, modalRefs.maskCanvas.height);
    }

    async function openMaskEditor() {
        ensureMaskModal();
        const backdropFile =
            typeof activeOptions.getMaskBackdropFile === "function"
                ? activeOptions.getMaskBackdropFile()
                : null;
        const backdropImage = await loadImageFromFile(backdropFile);
        const size = backdropImage
            ? { width: backdropImage.naturalWidth, height: backdropImage.naturalHeight }
            : getEditorSize();

        const { baseCanvas, maskCanvas, info } = modalRefs;
        baseCanvas.width = size.width;
        baseCanvas.height = size.height;
        maskCanvas.width = size.width;
        maskCanvas.height = size.height;

        const baseContext = baseCanvas.getContext("2d");
        baseContext.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
        if (backdropImage) {
            baseContext.drawImage(backdropImage, 0, 0, baseCanvas.width, baseCanvas.height);
        } else {
            baseContext.fillStyle = "#111111";
            baseContext.fillRect(0, 0, baseCanvas.width, baseCanvas.height);
        }
        info.textContent = `Mask size: ${size.width} x ${size.height}`;
        await drawExistingMask();
        applyCanvasDisplaySize();
        setModalOpen(true);
    }

    async function saveCanvasMask() {
        maskBlob = await new Promise((resolve) => {
            modalRefs.maskCanvas.toBlob((blob) => resolve(blob), "image/png");
        });
        if (!maskBlob) {
            alert("Failed to create IP-Adapter mask.");
            return;
        }

        const input = document.getElementById("ip_adapter_mask_image");
        if (input) {
            input.value = "";
        }
        updateMaskPreview();
        setModalOpen(false);
    }

    function clearMaskSelection() {
        const input = document.getElementById("ip_adapter_mask_image");
        if (input) {
            input.value = "";
        }
        maskBlob = null;
        updateMaskPreview();
    }

    function updateMaskPreview() {
        const input = document.getElementById("ip_adapter_mask_image");
        const preview = document.getElementById("ip_adapter_mask_preview");
        const empty = document.getElementById("ip_adapter_mask_preview_empty");
        const file = input?.files?.[0] ?? maskBlob;

        revokeMaskPreviewUrl();
        if (preview) {
            preview.removeAttribute("src");
            preview.classList.add("is-hidden");
        }
        empty?.classList.remove("is-hidden");

        if (!file) {
            emitAdapterSummaryChanged();
            return;
        }
        maskPreviewUrl = URL.createObjectURL(file);
        if (preview) {
            preview.src = maskPreviewUrl;
            preview.classList.remove("is-hidden");
        }
        empty?.classList.add("is-hidden");
        emitAdapterSummaryChanged();
    }

    function init(options = {}) {
        activeOptions = options || {};
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
        const maskInput = document.getElementById("ip_adapter_mask_image");
        const maskEditorOpen = document.getElementById("ip_adapter_mask_editor_open");
        const maskClear = document.getElementById("ip_adapter_mask_clear");

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
            emitAdapterSummaryChanged();
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
            emitAdapterSummaryChanged();
        }

        toggle.addEventListener("click", () => {
            setOpen(!content.classList.contains("is-open"));
        });
        enabled?.addEventListener("change", () => {
            if (enabled.checked) {
                setOpen(true);
            }
            emitAdapterSummaryChanged();
        });
        input?.addEventListener("change", updatePreview);
        maskInput?.addEventListener("change", () => {
            maskBlob = null;
            updateMaskPreview();
            setOpen(true);
        });
        maskEditorOpen?.addEventListener("click", () => {
            void openMaskEditor();
        });
        maskClear?.addEventListener("click", clearMaskSelection);
        document.getElementById("ip_adapter_scale")?.addEventListener("input", emitAdapterSummaryChanged);
        window.addEventListener("beforeunload", () => {
            revokePreviewUrl();
            revokeMaskPreviewUrl();
        });

        updatePreview();
        updateMaskPreview();
        emitAdapterSummaryChanged();
    }

    window.IpAdapterPanel = { init, getMaskFile, getSummary };
})();
