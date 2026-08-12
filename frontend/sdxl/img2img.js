const TASK_IMG2IMG = "sdxl.img2img";

const page = SdxlGenerationController.createImg2Img({
    taskType: TASK_IMG2IMG,
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 20 },
        { element: "cfg", key: "guidance_scale", type: "number", fallback: 7.5 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 1024 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 1024 },
        { element: "strength", key: "strength", type: "number", fallback: 0.75 },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "model_select", key: "model", fallback: null },
        { element: "clip_skip", key: "clip_skip", type: "number", integer: true, fallback: 1 },
        { element: "controlnet-enabled", key: "controlnet_enabled", type: "checkbox" },
        {
            element: "controlnet_conditioning_scale",
            key: "controlnet_conditioning_scale",
            type: "number",
            fallback: 1.0,
        },
        { element: "control_guidance_start", key: "control_guidance_start", type: "number", fallback: 0.0 },
        { element: "control_guidance_end", key: "control_guidance_end", type: "number", fallback: 1.0 },
        { element: "controlnet_guess_mode", key: "controlnet_guess_mode", type: "checkbox" },
        { element: "controlnet_compat_mode", key: "controlnet_compat_mode", fallback: "warn" },
        { element: "ip_adapter_enabled", key: "ip_adapter_enabled", type: "checkbox" },
        { element: "ip_adapter_scale", key: "ip_adapter_scale", type: "number", fallback: 0.6 },
    ],
});

window.generateSdxlImg2Img = page.generate;
GenerationPage.runWhenDomReady(page.init);
