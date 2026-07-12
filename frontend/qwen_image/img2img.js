const page = GenerationPage.create({
    family: "qwen-image",
    taskType: "qwen-image.img2img",
    fallbackModel: { value: "qwen-image", label: "qwen-image (diffusers)" },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 30 },
        { element: "true_cfg", key: "true_cfg_scale", type: "number", fallback: 4.0 },
        { element: "cfg", key: "guidance_scale", type: "number", fallback: 7.5 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1024 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1024 },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "strength", key: "strength", type: "number", fallback: 0.75 },
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
