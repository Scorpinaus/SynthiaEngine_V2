const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return "http://127.0.0.1:8000" + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

const baseCanvas = document.getElementById("base_canvas");
const maskCanvas = document.getElementById("mask_canvas");
const canvasStack = document.querySelector(".canvas-stack");
const imageInfo = document.getElementById("image_info");
const brushSizeInput = document.getElementById("brush_size");
const brushValue = document.getElementById("brush_value");
const eraseToggle = document.getElementById("erase_toggle");
const initialImageInput = document.getElementById("initial_image");
const maskModal = document.getElementById("mask_modal");
const maskPreview = document.getElementById("mask_preview");
const maskPreviewPanel = document.getElementById("mask_preview_panel");

let baseImageFile = null;
let isDrawing = false;
let maskBlob = null;
let maskDataUrl = null;

const baseContext = baseCanvas.getContext("2d");
const maskContext = maskCanvas.getContext("2d");

function updateBrushLabel() {
    brushValue.textContent = brushSizeInput.value;
}

updateBrushLabel();
brushSizeInput.addEventListener("input", updateBrushLabel);

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch("http://127.0.0.1:8000/models");
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

    const availableWidth = canvasStack.clientWidth || image.width;
    const displayWidth = Math.min(availableWidth, image.width);
    const displayHeight = Math.round(displayWidth * (image.height / image.width));

    canvasStack.style.height = `${displayHeight}px`;
    baseCanvas.style.width = "100%";
    baseCanvas.style.height = "100%";
    maskCanvas.style.width = "100%";
    maskCanvas.style.height = "100%";

    baseContext.clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    baseContext.drawImage(image, 0, 0);
    clearMask();
    imageInfo.textContent = `Image size: ${image.width} × ${image.height}`;
}

initialImageInput.addEventListener("change", () => {
    const file = initialImageInput.files[0];
    if (!file) {
        return;
    }

    baseImageFile = file;
    maskBlob = null;
    maskDataUrl = null;
    maskPreview.removeAttribute("src");
    maskPreviewPanel.classList.add("hidden");
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
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
    maskPreview.removeAttribute("src");
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
    maskPreview.src = maskDataUrl;
    maskPreviewPanel.classList.remove("hidden");
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

async function generateInpaint() {
    if (!baseImageFile) {
        alert("Please upload an initial image.");
        return;
    }
    if (!maskBlob) {
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

    const formData = new FormData();
    formData.append("initial_image", baseImageFile);
    formData.append("mask_image", maskBlob, "mask.png");
    formData.append("prompt", prompt);
    formData.append("negative_prompt", negative_prompt);
    formData.append("steps", steps.toString());
    formData.append("cfg", cfg.toString());
    formData.append("scheduler", scheduler);
    formData.append("seed", seed === null ? "" : seed.toString());
    formData.append("num_images", num_images.toString());
    formData.append("model", model);

    const res = await fetch("http://127.0.0.1:8000/generate-inpaint", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}

window.generateInpaint = generateInpaint;
