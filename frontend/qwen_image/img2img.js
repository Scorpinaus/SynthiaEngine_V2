const QWEN_IMAGE_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。";
const qwenLightningSettings = window.QwenImageLightningSettings.create({
    taskType: "qwen-image.img2img",
});

const page = GenerationPage.create({
    family: "qwen-image",
    taskType: "qwen-image.img2img",
    loraEnvelope: false,
    settingsHooks: qwenLightningSettings,
    fallbackModel: { value: "qwen-image", label: "qwen-image (diffusers)" },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: QWEN_IMAGE_NEGATIVE_PROMPT },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 50 },
        { element: "true_cfg", key: "true_cfg_scale", type: "number", fallback: 4.0 },
        { element: "scheduler", key: "scheduler", fallback: "flowmatch_euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1328 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1328 },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "strength", key: "strength", type: "number", fallback: 0.6 },
        { element: "live_preview", key: "live_preview", type: "checkbox", fallback: true },
    ],
});

async function generateQwenImageImg2Img() {
    const imageInput = document.getElementById("initial_image");
    if (!imageInput.files?.length) {
        alert("Please choose an initial image.");
        return;
    }
    try {
        const file = imageInput.files[0];
        const artifact = await WorkflowClient.uploadArtifact(API_BASE, file, file.name || "initial.png");
        const inputs = page.withLora(page.collectSettings(await page.defaults()));
        inputs.initial_image = `@artifact:${artifact.artifact_id}`;
        await page.run(inputs, "Failed to run Qwen-Image img2img job:");
    } catch (error) {
        console.warn("Failed to upload the Qwen-Image img2img input:", error);
    }
}
