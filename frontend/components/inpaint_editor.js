/** Shared canvas-based mask editor used by simple-family inpainting pages. */
(function () {
    function create() {
        const baseCanvas = document.getElementById("base_canvas");
        const maskCanvas = document.getElementById("mask_canvas");
        const canvasStack = document.querySelector(".canvas-stack");
        const canvasScroll = document.querySelector(".canvas-scroll");
        const imageInfo = document.getElementById("image_info");
        const brushSizeInput = document.getElementById("brush_size");
        const brushValue = document.getElementById("brush_value");
        const zoomInput = document.getElementById("zoom_level");
        const zoomValue = document.getElementById("zoom_value");
        const eraseToggle = document.getElementById("erase_toggle");
        const initialImageInput = document.getElementById("initial_image");
        const maskModal = document.getElementById("mask_modal");
        const maskPreview = document.getElementById("mask_preview");
        const maskPreviewPanel = document.getElementById("mask_preview_panel");
        const maskBlurButton = document.getElementById("mask_blur");
        const blurFactorInput = document.getElementById("blur_factor");
        const blurToggle = document.getElementById("blur_toggle");

        const baseContext = baseCanvas.getContext("2d");
        const maskContext = maskCanvas.getContext("2d");
        let baseImageFile = null;
        let baseImage = null;
        let isDrawing = false;
        let maskBlob = null;
        let maskDataUrl = null;
        let blurMaskBlob = null;
        let blurMaskDataUrl = null;

        function updateMaskPreview() {
            const source = blurToggle.checked && blurMaskDataUrl ? blurMaskDataUrl : maskDataUrl;
            if (source) {
                maskPreview.src = source;
            } else {
                maskPreview.removeAttribute("src");
            }
        }

        function updateBlurControls() {
            maskBlurButton.disabled = !maskBlob;
            if (!maskBlob) {
                blurToggle.checked = false;
                blurToggle.disabled = true;
                blurMaskBlob = null;
                blurMaskDataUrl = null;
            } else {
                blurToggle.disabled = !blurMaskDataUrl;
            }
            updateMaskPreview();
        }

        function clearMask() {
            maskContext.fillStyle = "#000000";
            maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            maskBlob = null;
            maskDataUrl = null;
            blurMaskBlob = null;
            blurMaskDataUrl = null;
            blurToggle.checked = false;
            updateBlurControls();
        }

        function resizeCanvasDisplay(image) {
            baseCanvas.width = image.width;
            baseCanvas.height = image.height;
            maskCanvas.width = image.width;
            maskCanvas.height = image.height;

            const availableWidth = canvasStack.parentElement?.clientWidth || canvasStack.clientWidth || image.width;
            const maxHeight = Math.round(window.innerHeight * 0.7);
            const maxWidth = Math.round(availableWidth);
            const fitScale = Math.min(1, maxWidth / image.width, maxHeight / image.height);
            const displayScale = fitScale * Number(zoomInput.value) / 100;
            const displayWidth = Math.round(image.width * displayScale);
            const displayHeight = Math.round(image.height * displayScale);
            const containerWidth = Math.min(maxWidth, displayWidth);
            const containerHeight = Math.min(maxHeight, displayHeight);

            for (const container of [canvasStack, canvasScroll]) {
                container.style.width = `${containerWidth}px`;
                container.style.height = `${containerHeight}px`;
            }
            canvasStack.style.maxWidth = "100%";
            canvasScroll.style.transform = "none";
            for (const canvas of [baseCanvas, maskCanvas]) {
                canvas.style.width = `${displayWidth}px`;
                canvas.style.height = `${displayHeight}px`;
            }
            baseContext.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
            baseContext.drawImage(image, 0, 0);
            clearMask();
            imageInfo.textContent = `Image size: ${image.width} × ${image.height} (${Math.round(displayScale * 100)}% view)`;
        }

        function openMaskEditor() {
            if (!baseImageFile) {
                alert("Please upload an initial image first.");
                return;
            }
            maskModal.classList.remove("hidden");
        }

        function closeMaskEditor() {
            maskModal.classList.add("hidden");
        }

        function toggleMaskPreview() {
            maskPreviewPanel.classList.toggle("hidden");
        }

        function getMaskBlob() {
            return new Promise((resolve) => maskCanvas.toBlob(resolve, "image/png"));
        }

        async function saveMask() {
            maskBlob = await getMaskBlob();
            if (!maskBlob) {
                alert("Failed to create mask image.");
                return;
            }
            maskDataUrl = maskCanvas.toDataURL("image/png");
            blurMaskBlob = null;
            blurMaskDataUrl = null;
            blurToggle.checked = false;
            maskPreviewPanel.classList.remove("hidden");
            updateBlurControls();
            closeMaskEditor();
        }

        async function generateBlurMask() {
            if (!maskBlob) {
                alert("Please create and save a mask before blurring.");
                return;
            }
            const blurFactor = Number(blurFactorInput.value);
            if (!Number.isFinite(blurFactor) || blurFactor < 0 || blurFactor > 128) {
                alert("Blur strength must be a number between 0 and 128.");
                return;
            }
            maskBlurButton.disabled = true;
            maskBlurButton.textContent = "Blurring...";
            try {
                const formData = new FormData();
                formData.append("mask_image", maskBlob, "mask.png");
                formData.append("blur_factor", String(blurFactor));
                const response = await fetch(`${API_BASE}/create-blur-mask`, { method: "POST", body: formData });
                if (!response.ok) {
                    throw new Error("Failed to blur mask.");
                }
                blurMaskBlob = await response.blob();
                blurMaskDataUrl = await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(blurMaskBlob);
                });
                blurToggle.checked = true;
            } catch (error) {
                console.error(error);
                alert("Unable to blur mask. Please try again.");
            } finally {
                maskBlurButton.textContent = "Blur mask edges";
                updateBlurControls();
            }
        }

        initialImageInput.addEventListener("change", () => {
            const file = initialImageInput.files[0];
            if (!file) return;
            baseImageFile = file;
            maskBlob = null;
            maskDataUrl = null;
            blurMaskBlob = null;
            blurMaskDataUrl = null;
            blurToggle.checked = false;
            maskPreview.removeAttribute("src");
            maskPreviewPanel.classList.add("hidden");
            updateBlurControls();
            const reader = new FileReader();
            reader.onload = (event) => {
                const image = new Image();
                image.onload = () => {
                    baseImage = image;
                    resizeCanvasDisplay(image);
                    openMaskEditor();
                };
                image.src = event.target.result;
            };
            reader.readAsDataURL(file);
        });

        function canvasPosition(event) {
            const rect = maskCanvas.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * maskCanvas.width / rect.width,
                y: (event.clientY - rect.top) * maskCanvas.height / rect.height,
            };
        }

        function drawAt(position) {
            maskContext.fillStyle = eraseToggle.checked ? "#000000" : "#ffffff";
            maskContext.beginPath();
            maskContext.arc(position.x, position.y, Number(brushSizeInput.value) / 2, 0, Math.PI * 2);
            maskContext.fill();
        }

        maskCanvas.addEventListener("pointerdown", (event) => {
            if (!baseImageFile) return;
            isDrawing = true;
            maskCanvas.setPointerCapture(event.pointerId);
            drawAt(canvasPosition(event));
        });
        maskCanvas.addEventListener("pointermove", (event) => {
            if (isDrawing) drawAt(canvasPosition(event));
        });
        maskCanvas.addEventListener("pointerup", () => { isDrawing = false; });
        maskCanvas.addEventListener("pointerleave", () => { isDrawing = false; });
        brushSizeInput.addEventListener("input", () => { brushValue.textContent = brushSizeInput.value; });
        zoomInput.addEventListener("input", () => {
            zoomValue.textContent = zoomInput.value;
            if (baseImage) resizeCanvasDisplay(baseImage);
        });
        blurToggle.addEventListener("change", updateMaskPreview);
        brushValue.textContent = brushSizeInput.value;
        zoomValue.textContent = zoomInput.value;
        updateBlurControls();

        Object.assign(window, { clearMask, openMaskEditor, closeMaskEditor, saveMask, toggleMaskPreview, generateBlurMask });
        return {
            getBaseImageFile: () => baseImageFile,
            getActiveMaskBlob: () => blurToggle.checked && blurMaskBlob ? blurMaskBlob : maskBlob,
        };
    }

    window.InpaintEditor = { create };
})();
