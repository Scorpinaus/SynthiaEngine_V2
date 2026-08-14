const QWEN_IMAGE_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。";

const page = GenerationPage.create({
    family: "qwen-image",
    taskType: "qwen-image.inpaint",
    fallbackModel: { value: "qwen-image", label: "qwen-image (diffusers)" },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: QWEN_IMAGE_NEGATIVE_PROMPT },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 50 },
        { element: "true_cfg", key: "true_cfg_scale", type: "number", fallback: 4.0 },
        { element: "scheduler", key: "scheduler", fallback: "flowmatch_euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1024 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1024 },
        {
            element: "padding_mask_crop",
            key: "padding_mask_crop",
            type: "number",
            integer: true,
            fallback: null,
        },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "strength", key: "strength", type: "number", fallback: 0.6 },
        { element: "live_preview", key: "live_preview", type: "checkbox", fallback: true },
    ],
});
const editor = InpaintEditor.create();

async function generateQwenImageInpaint() {
    const initialImage = editor.getBaseImageFile();
    const maskImage = editor.getActiveMaskBlob();
    if (!initialImage) {
        alert("Please upload an initial image.");
        return;
    }
    if (!maskImage) {
        alert("Please create and save a mask before generating.");
        return;
    }
    try {
        const [baseArtifact, maskArtifact] = await Promise.all([
            WorkflowClient.uploadArtifact(API_BASE, initialImage, initialImage.name || "initial.png"),
            WorkflowClient.uploadArtifact(API_BASE, maskImage, "mask.png"),
        ]);
        const inputs = page.collectSettings(await page.defaults());
        inputs.initial_image = `@artifact:${baseArtifact.artifact_id}`;
        inputs.mask_image = `@artifact:${maskArtifact.artifact_id}`;
        await page.run(inputs, "Failed to run Qwen-Image inpaint job:");
    } catch (error) {
        console.warn("Failed to upload Qwen-Image inpaint inputs:", error);
    }
}
