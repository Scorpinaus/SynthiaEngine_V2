const page = GenerationPage.create({
    family: "anima",
    taskType: "anima.text2img",
    lora: false,
    fallbackModel: {
        value: "Anima-Preview-3",
        label: "Anima-Preview-3 (hub, diffusers)",
    },
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 35 },
        { element: "cfg", key: "guidance_scale", type: "number", fallback: 4.5 },
        { element: "scheduler", key: "scheduler", fallback: "flowmatch_euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1024 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1024 },
        { element: "model_select", key: "model", fallback: null },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "memory_preset", key: "memory_preset", fallback: "sequential_offload" },
    ],
});

async function generate() {
    await page.run(page.collectSettings(await page.defaults()), "Failed to generate Anima images:");
}
