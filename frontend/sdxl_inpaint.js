const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

function getControlNetState() {
    return window.ControlNetPanel?.getState?.() ?? null;
}

function resolveSdxlControlNetModel(modelId) {
    const normalized = String(modelId || "").trim();
    if (!normalized || normalized.includes("_sd15")) {
        return "diffusers/controlnet-canny-sdxl-1.0";
    }
    return normalized;
}

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
        const res = await fetch(`${API_BASE}/models?family=sdxl`);
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
        fallback.value = "stable-diffusion-xl-base-1.0";
        fallback.textContent = "stable-diffusion-xl-base-1.0 (sdxl, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();
if (window.WorkflowCatalog?.load) {
    void window.WorkflowCatalog
        .load(API_BASE)
        .then(() => {
            window.WorkflowCatalog.applyDefaultsToForm("sdxl.inpaint", {
                steps: "steps",
                guidance_scale: "guidance_scale",
                strength: "strength",
                num_images: "num_images",
                padding_mask_crop: "padding_mask_crop",
                clip_skip: "clip_skip",
                controlnet_conditioning_scale: "controlnet_conditioning_scale",
                control_guidance_start: "control_guidance_start",
                control_guidance_end: "control_guidance_end",
                controlnet_compat_mode: "controlnet_compat_mode",
            });
        })
        .catch(() => {});
}
if (window.ControlNetPreprocessor?.init) {
    void window.ControlNetPreprocessor.init().catch((error) => {
        console.warn("ControlNet init failed:", error);
    });
}

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
        const res = await fetch(`${API_BASE}/create-blur-mask`, {
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

async function generateSdxlInpaint() {
    const token = ++activeJobToken;
    closeActiveEventSource();
    const controlnetState = getControlNetState();
    const controlnetEnabled = Boolean(document.getElementById("controlnet-enabled")?.checked);

    if (!baseImageFile) {
        alert("Please upload an initial image.");
        return;
    }
    const activeMaskBlob = blurToggle.checked && blurMaskBlob ? blurMaskBlob : maskBlob;
    if (!activeMaskBlob) {
        alert("Please create and save a mask before generating.");
        return;
    }

    const catalog = window.WorkflowCatalog?.load ? await window.WorkflowCatalog.load(API_BASE) : null;
    const defaults = catalog?.tasks?.["sdxl.inpaint"]?.input_defaults ?? {};

    const prompt = WorkflowClient.readTextValue("prompt", defaults.prompt ?? "");
    const negative_prompt = WorkflowClient.readTextValue("negative_prompt", defaults.negative_prompt ?? "");
    const steps = WorkflowClient.readNumberValue("steps", defaults.steps ?? 20, { integer: true });
    const guidanceScale = WorkflowClient.readNumberValue(
        "guidance_scale",
        defaults.guidance_scale ?? 7.5,
    );
    const scheduler = WorkflowClient.readTextValue("scheduler", defaults.scheduler ?? "euler");
    const seed = WorkflowClient.readSeedValue("seed");
    const num_images = WorkflowClient.readNumberValue("num_images", defaults.num_images ?? 1, { integer: true });
    const modelRaw = document.getElementById("model_select")?.value ?? "";
    const model = modelRaw ? modelRaw : (defaults.model ?? null);
    const strength = WorkflowClient.readNumberValue("strength", defaults.strength ?? 0.5);
    const paddingMaskCrop = WorkflowClient.readNumberValue(
        "padding_mask_crop",
        defaults.padding_mask_crop ?? 32,
        { integer: true }
    );
    const clip_skip = WorkflowClient.readNumberValue("clip_skip", defaults.clip_skip ?? 1, { integer: true });
    const controlnet_conditioning_scale = WorkflowClient.readNumberValue(
        "controlnet_conditioning_scale",
        defaults.controlnet_conditioning_scale ?? 1.0
    );
    const control_guidance_start = WorkflowClient.readNumberValue(
        "control_guidance_start",
        defaults.control_guidance_start ?? 0.0
    );
    const control_guidance_end = WorkflowClient.readNumberValue(
        "control_guidance_end",
        defaults.control_guidance_end ?? 1.0
    );
    const controlnet_guess_mode = Boolean(document.getElementById("controlnet_guess_mode")?.checked);
    const controlnet_compat_mode = WorkflowClient.readTextValue(
        "controlnet_compat_mode",
        defaults.controlnet_compat_mode ?? "warn"
    );

    try {
        const [uploadedBase, uploadedMask] = await Promise.all([
            WorkflowClient.uploadArtifact(API_BASE, baseImageFile, baseImageFile.name || "initial.png"),
            WorkflowClient.uploadArtifact(API_BASE, activeMaskBlob, "mask.png"),
        ]);

        const taskInputs = {
            initial_image: `@artifact:${uploadedBase.artifact_id}`,
            mask_image: `@artifact:${uploadedMask.artifact_id}`,
            prompt,
            negative_prompt,
            steps,
            guidance_scale: guidanceScale,
            scheduler,
            seed,
            num_images,
            model,
            strength,
            padding_mask_crop: paddingMaskCrop,
            clip_skip,
        };
        if (controlnetEnabled) {
            const controlItems = Array.isArray(controlnetState?.controlItems)
                ? controlnetState.controlItems
                : [];
            if (controlItems.length === 0 && !controlnetState?.previewBlob) {
                throw new Error("ControlNet enabled but no preprocessor output image is ready.");
            }
            const effectiveItems =
                controlItems.length > 0
                    ? controlItems
                    : [
                        {
                            previewBlob: controlnetState.previewBlob,
                            preprocessorId: controlnetState.preprocessorId ?? null,
                            modelId: "diffusers/controlnet-canny-sdxl-1.0",
                            conditioningScale: controlnet_conditioning_scale,
                        },
                    ];
            const uploadedArtifacts = await Promise.all(
                effectiveItems.map((item, idx) =>
                    WorkflowClient.uploadArtifact(
                        API_BASE,
                        item.previewBlob,
                        `controlnet_${idx + 1}.png`
                    )
                )
            );
            const controlImages = uploadedArtifacts.map(
                (controlUploaded) => `@artifact:${controlUploaded.artifact_id}`
            );
            const controlnetModels = effectiveItems.map((item) =>
                resolveSdxlControlNetModel(item.modelId)
            );
            const controlnetScales = effectiveItems.map((item) => {
                const parsed = Number(item.conditioningScale);
                return Number.isFinite(parsed) ? parsed : controlnet_conditioning_scale;
            });
            const controlnetPreprocessorIds = effectiveItems.map(
                (item) => item.preprocessorId || null
            );
            const hasAllPreprocessorIds = controlnetPreprocessorIds.every(
                (value) => typeof value === "string" && value.length > 0
            );

            taskInputs.control_image = controlImages[0];
            taskInputs.controlnet_model = controlnetModels[0];
            taskInputs.controlnet_conditioning_scale = controlnetScales[0];
            taskInputs.controlnet_guess_mode = controlnet_guess_mode;
            taskInputs.control_guidance_start = control_guidance_start;
            taskInputs.control_guidance_end = control_guidance_end;
            taskInputs.controlnet_compat_mode = controlnet_compat_mode;
            if (hasAllPreprocessorIds) {
                taskInputs.controlnet_preprocessor_id = controlnetPreprocessorIds[0];
            }
            if (effectiveItems.length > 1) {
                taskInputs.control_images = controlImages.slice(1);
                taskInputs.controlnet_models = controlnetModels;
                taskInputs.controlnet_conditioning_scales = controlnetScales;
                if (hasAllPreprocessorIds) {
                    taskInputs.controlnet_preprocessor_ids = controlnetPreprocessorIds;
                }
            }
        }

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: "sdxl.inpaint",
                    inputs: taskInputs,
                },
            ],
            return: "@t1.images",
        };

        const idempotencyKey = WorkflowClient.makeIdempotencyKey();
        const createdJob = await WorkflowClient.submitWorkflow(API_BASE, workflowPayload, idempotencyKey);
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        activeEventSource = WorkflowClient.watchJob(API_BASE, jobId, {
            isStale: () => token !== activeJobToken,
            onDone: (job) => {
                const status = job?.status ?? "unknown";
                if (status === "succeeded") {
                    const images = job?.result?.outputs;
                    gallery.setImages(Array.isArray(images) ? images : []);
                    const warnings = job?.result?.tasks?.t1?.warnings;
                    if (Array.isArray(warnings) && warnings.length > 0) {
                        console.warn("ControlNet warnings:", warnings);
                        const statusNode = document.getElementById("controlnet-status");
                        if (statusNode) {
                            statusNode.textContent = warnings.join(" ");
                        }
                    }
                } else {
                    gallery.setImages([]);
                }
            },
            onError: () => {
                if (token !== activeJobToken) {
                    return;
                }
                gallery.setImages([]);
            },
        });
    } catch (error) {
        console.warn("Failed to run SDXL inpaint job:", error);
        gallery.setImages([]);
    }
}

window.generateSdxlInpaint = generateSdxlInpaint;
window.generateBlurMask = generateBlurMask;
blurToggle.addEventListener("change", updateMaskPreview);
updateBlurControls();
