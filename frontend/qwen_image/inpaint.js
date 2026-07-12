const page = GenerationPage.create({
    family: "qwen-image",
    taskType: "qwen-image.inpaint",
    fallbackModel: { value: "qwen-image", label: "qwen-image (diffusers)" },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 30 },
        { element: "true_cfg", key: "true_cfg_scale", type: "number", fallback: 4.0 },
        { element: "guidance_scale", key: "guidance_scale", type: "number", fallback: 7.5 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "strength", key: "strength", type: "number", fallback: 0.5 },
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
        const inputs = page.withLora(page.collectSettings(await page.defaults()));
        inputs.initial_image = `@artifact:${baseArtifact.artifact_id}`;
        inputs.mask_image = `@artifact:${maskArtifact.artifact_id}`;
        await page.run(inputs, "Failed to run Qwen-Image inpaint job:");
    } catch (error) {
        console.warn("Failed to upload Qwen-Image inpaint inputs:", error);
    }
}
