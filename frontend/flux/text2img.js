const page = GenerationPage.create({
    family: "flux",
    taskType: "flux.text2img",
    fallbackModel: {
        value: "black-forest-labs/FLUX.1-schnell",
        label: "black-forest-labs/FLUX.1-schnell (flux, diffusers)",
    },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 20 },
        { element: "cfg", key: "guidance_scale", type: "number", fallback: 0.0 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1024 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1024 },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
    ],
});

async function generate() {
    const inputs = page.withLora(page.collectSettings(await page.defaults()));
    await page.run(inputs, "Failed to generate Flux images:");
}
