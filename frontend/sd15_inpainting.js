const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return "http://127.0.0.1:8000" + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

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

let baseImageFile = null;
let baseImage = null;
let isDrawing = false;
let maskBlob = null;
let maskDataUrl = null;
let blurMaskBlob = null;
let blurMaskDataUrl = null;
let displayScale = 1;

const baseContext = baseCanvas.getContext("2d");
const maskContext = maskCanvas.getContext("2d");

function updateBrushLabel() {
    brushValue.textContent = brushSizeInput.value;
}

updateBrushLabel();
brushSizeInput.addEventListener("input", updateBrushLabel);

function updateZoomLabel() {
    zoomValue.textContent = zoomInput.value;
}

updateZoomLabel();
zoomInput.addEventListener("input", () => {
    updateZoomLabel();
    if (baseImage) {
        resizeCanvasDisplay(baseImage);
    }
});

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch("http://127.0.0.1:8000/models?family=sd15");
        const models = await res.json();

        if (!Array.isArray(models) || models.length === 0) {
            throw new Error("No models returned.");
        }

        models.forEach((model, index) => {
            const option = document.createElement("option");
            option.value = model.name ?? "";
            const family = model.family ?? "unknown";
            const modelType = model.model_type ?? "unknown";
            option.textContent = `${model.name} (${family}, ${modelType})`;
            if (index === 0) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    } catch (error) {
        const fallback = document.createElement("option");
        fallback.value = "stable-diffusion-v1-5";
        fallback.textContent = "stable-diffusion-v1-5 (sd15, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

function resizeCanvasDisplay(image) {
    baseCanvas.width = image.width;
    baseCanvas.height = image.height;
    maskCanvas.width = image.width;
    maskCanvas.height = image.height;

    const availableWidth =
        canvasStack.parentElement?.clientWidth || canvasStack.clientWidth || image.width;
    const maxHeight = Math.round(window.innerHeight * 0.7);
    const maxWidth = Math.round(availableWidth);
    const fitScale = Math.min(1, maxWidth / image.width, maxHeight / image.height);
    const zoomScale = Number(zoomInput.value) / 100;
    displayScale = fitScale * zoomScale;
    const displayWidth = Math.round(image.width * displayScale);
    const displayHeight = Math.round(image.height * displayScale);
    const containerWidth = Math.min(maxWidth, displayWidth);
    const containerHeight = Math.min(maxHeight, displayHeight);

    canvasStack.style.width = `${containerWidth}px`;
    canvasStack.style.height = `${containerHeight}px`;
    canvasStack.style.maxWidth = "100%";
    canvasScroll.style.width = `${containerWidth}px`;
    canvasScroll.style.height = `${containerHeight}px`;
    canvasScroll.style.transform = "none";    
    baseCanvas.style.width = `${displayWidth}px`;
    baseCanvas.style.height = `${displayHeight}px`;
    maskCanvas.style.width = `${displayWidth}px`;
    maskCanvas.style.height = `${displayHeight}px`;

    baseContext.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    baseContext.drawImage(image, 0, 0);
    clearMask();
    imageInfo.textContent = `Image size: ${image.width} × ${image.height} (${Math.round(displayScale * 100)}% view)`;
}

initialImageInput.addEventListener("change", () => {
    const file = initialImageInput.files[0];
    if (!file) {
        return;
    }

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
        const img = new Image();
        img.onload = () => {
            baseImage = img;
            resizeCanvasDisplay(img);
            openMaskEditor();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

function getCanvasPosition(event) {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
    };
}

function drawAt(position) {
    const brushSize = Number(brushSizeInput.value);
    const color = eraseToggle.checked ? "#000000" : "#ffffff";
    maskContext.fillStyle = color;
    maskContext.beginPath();
    maskContext.arc(position.x, position.y, brushSize / 2, 0, Math.PI * 2);
    maskContext.fill();
}

maskCanvas.addEventListener("pointerdown", (event) => {
    if (!baseImageFile) {
        return;
    }
    isDrawing = true;
    maskCanvas.setPointerCapture(event.pointerId);
    drawAt(getCanvasPosition(event));
});

maskCanvas.addEventListener("pointermove", (event) => {
    if (!isDrawing) {
        return;
    }
    drawAt(getCanvasPosition(event));
});

maskCanvas.addEventListener("pointerup", () => {
    isDrawing = false;
});

maskCanvas.addEventListener("pointerleave", () => {
    isDrawing = false;
});

function clearMask() {
    if (!maskContext) {
        return;
    }
    maskContext.fillStyle = "#000000";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskBlob = null;
    maskDataUrl = null;
    blurMaskBlob = null;
    blurMaskDataUrl = null;
    blurToggle.checked = false;
    maskPreview.removeAttribute("src");
    updateBlurControls();
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
    if (maskPreviewPanel.classList.contains("hidden")) {
        maskPreviewPanel.classList.remove("hidden");
    } else {
        maskPreviewPanel.classList.add("hidden");
    }
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
    updateMaskPreview();
    updateBlurControls();
    closeMaskEditor();
}

window.clearMask = clearMask;
window.openMaskEditor = openMaskEditor;
window.closeMaskEditor = closeMaskEditor;
window.saveMask = saveMask;
window.toggleMaskPreview = toggleMaskPreview;

function getMaskBlob() {
    return new Promise((resolve) => {
        maskCanvas.toBlob((blob) => {
            resolve(blob);
        }, "image/png");
    });
}

function updateMaskPreview() {
    if (blurToggle.checked && blurMaskDataUrl) {
        maskPreview.src = blurMaskDataUrl;
        return;
    }
    if (maskDataUrl) {
        maskPreview.src = maskDataUrl;
        return;
    }
    maskPreview.removeAttribute("src");
}

function updateBlurControls() {
    const hasMask = Boolean(maskBlob);
    maskBlurButton.disabled = !hasMask;
    if (!hasMask) {
        blurToggle.checked = false;
        blurToggle.disabled = true;
        blurMaskBlob = null;
        blurMaskDataUrl = null;
    } else {
        blurToggle.disabled = !blurMaskDataUrl;
    }
    updateMaskPreview();
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
        formData.append("blur_factor", blurFactor.toString());
        const res = await fetch("http://127.0.0.1:8000/create-blur-mask", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            throw new Error("Failed to blur mask.");
        }

        const blob = await res.blob();
        blurMaskBlob = blob;
        blurMaskDataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(blob);
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

async function generateInpaint() {
    if (!baseImageFile) {
        alert("Please upload an initial image.");
        return;
    }
    const activeMaskBlob = blurToggle.checked && blurMaskBlob ? blurMaskBlob : maskBlob;
    if (!activeMaskBlob) {
        alert("Please create and save a mask before generating.");
        return;
    }

    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const cfg = Number(document.getElementById("cfg").value);
    const scheduler = document.getElementById("scheduler").value;
    const seedValue = document.getElementById("seed").value;
    const seedNumber = seedValue === "" ? null : Number(seedValue);
    const seed = Number.isFinite(seedNumber) ? seedNumber : null;
    const negative_prompt = document.getElementById("negative_prompt").value;
    const num_images = Number(document.getElementById("num_images").value);
    const model = document.getElementById("model_select").value;
    const strength = Number(document.getElementById("strength").value);
    const paddingMaskCrop = Number(document.getElementById("padding_mask_crop").value);
    const clip_skip = Number(document.getElementById("clip_skip").value);

    const formData = new FormData();
    formData.append("initial_image", baseImageFile);
    formData.append("mask_image", activeMaskBlob, "mask.png");
    formData.append("prompt", prompt);
    formData.append("negative_prompt", negative_prompt);
    formData.append("steps", steps.toString());
    formData.append("cfg", cfg.toString());
    formData.append("scheduler", scheduler);
    formData.append("seed", seed === null ? "" : seed.toString());
    formData.append("num_images", num_images.toString());
    formData.append("model", model);
    formData.append("strength", strength);
    formData.append("padding_mask_crop", paddingMaskCrop);
    formData.append("clip_skip", clip_skip);

    const res = await fetch("http://127.0.0.1:8000/generate-inpaint", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}

window.generateInpaint = generateInpaint;
window.generateBlurMask = generateBlurMask;
blurToggle.addEventListener("change", updateMaskPreview);
updateBlurControls();
