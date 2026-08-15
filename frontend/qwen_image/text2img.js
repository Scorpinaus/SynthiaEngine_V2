const QWEN_IMAGE_NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。";

const page = GenerationPage.create({
    family: "qwen-image",
    taskType: "qwen-image.text2img",
    loraEnvelope: false,
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
        { element: "live_preview", key: "live_preview", type: "checkbox", fallback: true },
    ],
});

async function generate() {
    const inputs = page.withLora(page.collectSettings(await page.defaults()));
    await page.run(inputs, "Failed to generate Qwen-Image images:");
}
