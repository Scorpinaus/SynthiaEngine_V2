const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return "http://127.0.0.1:8000" + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

const maskCanvas = document.getElementById("mask-canvas");
const maskCtx = maskCanvas.getContext("2d");
const maskBuffer = document.createElement("canvas");
const maskBufferCtx = maskBuffer.getContext("2d");
const imageInput = document.getElementById("image");
const brushInput = document.getElementById("brush_size");
const exportButton = document.getElementById("export_mask");
const widthInput = document.getElementById("width");
const heightInput = document.getElementById("height");
let baseImage = null;
let isDrawing = false;
let lastPoint = null;

function resizeCanvases(width, height) {
    maskCanvas.width = width;
    maskCanvas.height = height;
    maskBuffer.width = width;
    maskBuffer.height = height;
    maskBufferCtx.fillStyle = "black";
    maskBufferCtx.fillRect(0, 0, width, height);
    drawBaseImage();
}

function drawBaseImage() {
    if (!baseImage) {
        return;
    }
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(baseImage, 0, 0, maskCanvas.width, maskCanvas.height);
}

function getBrushSize() {
    const size = Number(brushInput.value);
    return Number.isFinite(size) ? size : 32;
}

function drawMaskStroke(x, y) {
    const brushSize = getBrushSize();
    maskCtx.save();
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.lineCap = "round";
    maskCtx.lineJoin = "round";
    maskCtx.strokeStyle = "rgba(255, 255, 255, 0.7)";
    maskCtx.lineWidth = brushSize;
    maskCtx.beginPath();
    if (lastPoint) {
        maskCtx.moveTo(lastPoint.x, lastPoint.y);
    } else {
        maskCtx.moveTo(x, y);
    }
    maskCtx.lineTo(x, y);
    maskCtx.stroke();
    maskCtx.restore();

    maskBufferCtx.save();
    maskBufferCtx.globalCompositeOperation = "source-over";
    maskBufferCtx.lineCap = "round";
    maskBufferCtx.lineJoin = "round";
    maskBufferCtx.strokeStyle = "white";
    maskBufferCtx.lineWidth = brushSize;
    maskBufferCtx.beginPath();
    if (lastPoint) {
        maskBufferCtx.moveTo(lastPoint.x, lastPoint.y);
    } else {
        maskBufferCtx.moveTo(x, y);
    }
    maskBufferCtx.lineTo(x, y);
    maskBufferCtx.stroke();
    maskBufferCtx.restore();
    lastPoint = { x, y };
}

function handlePointerDown(event) {
    if (!baseImage) {
        alert("Upload an image before painting a mask.");
        return;
    }
    isDrawing = true;
    const rect = maskCanvas.getBoundingClientRect();
    drawMaskStroke(event.clientX - rect.left, event.clientY - rect.top);
}

function handlePointerMove(event) {
    if (!isDrawing) {
        return;
    }
    const rect = maskCanvas.getBoundingClientRect();
    drawMaskStroke(event.clientX - rect.left, event.clientY - rect.top);
}

function handlePointerUp() {
    isDrawing = false;
    lastPoint = null;
}

maskCanvas.addEventListener("pointerdown", handlePointerDown);
maskCanvas.addEventListener("pointermove", handlePointerMove);
maskCanvas.addEventListener("pointerup", handlePointerUp);
maskCanvas.addEventListener("pointerleave", handlePointerUp);

function clearMask() {
    if (!baseImage) {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        return;
    }
    maskBufferCtx.fillStyle = "black";
    maskBufferCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    drawBaseImage();
}

function buildMaskBlob() {
    return new Promise((resolve) => {
        maskBuffer.toBlob((blob) => resolve(blob), "image/png");
    });
}

async function downloadMask() {
    if (!baseImage) {
        alert("Upload an image before exporting a mask.");
        return;
    }
    const maskBlob = await buildMaskBlob();
    if (!maskBlob) {
        return;
    }
    const maskUrl = URL.createObjectURL(maskBlob);
    const link = document.createElement("a");
    link.href = maskUrl;
    link.download = "inpaint-mask.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(maskUrl);
}

imageInput.addEventListener("change", () => {
    const imageFile = imageInput.files[0];
    if (!imageFile) {
        baseImage = null;
        clearMask();
        exportButton.disabled = true;
        return;
    }
    const img = new Image();
    img.onload = () => {
        baseImage = img;
        resizeCanvases(img.width, img.height);
        drawBaseImage();
        widthInput.value = img.width;
        heightInput.value = img.height;
        exportButton.disabled = false;
    };
    img.src = URL.createObjectURL(imageFile);
});

async function generateInpaint() {
    const imageFile = imageInput.files[0];

    if (!imageFile) {
        alert("Please select an image to inpaint.");
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
    const width = baseImage ? baseImage.width : Number(widthInput.value);
    const height = baseImage ? baseImage.height : Number(heightInput.value);
    const strength = Number(document.getElementById("strength").value);
    const num_images = Number(document.getElementById("num_images").value);

    const formData = new FormData();
    formData.append("image", imageFile);
    const maskBlob = await buildMaskBlob();
    if (!maskBlob) {
        alert("Please draw a mask before generating.");
        return;
    }
    formData.append("mask_image", maskBlob, "mask.png");
    formData.append("prompt", prompt);
    formData.append("negative_prompt", negative_prompt);
    formData.append("steps", steps.toString());
    formData.append("cfg", cfg.toString());
    formData.append("scheduler", scheduler);
    formData.append("seed", seed === null ? "" : seed.toString());
    formData.append("width", width.toString());
    formData.append("height", height.toString());
    formData.append("strength", strength.toString());
    formData.append("num_images", num_images.toString());

    const res = await fetch("http://127.0.0.1:8000/generate-inpaint", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}
