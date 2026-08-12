const TASK_INPAINT = "sd15.inpaint";
const TASK_IP_ADAPTER_ENCODE = "sd15.ip_adapter.encode";

const page = Sd15GenerationController.createInpaint({
    taskType: TASK_INPAINT,
    taskIpAdapter: TASK_IP_ADAPTER_ENCODE,
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 20 },
        { element: "cfg", key: "cfg", type: "number", fallback: 7.5 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "model_select", key: "model", fallback: null },
        { element: "strength", key: "strength", type: "number", fallback: 0.5 },
        {
            element: "padding_mask_crop",
            key: "padding_mask_crop",
            type: "number",
            integer: true,
            fallback: 32,
        },
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
        {
            element: "controlnet_inpaint_condition",
            key: "controlnet_inpaint_condition",
            type: "checkbox",
        },
        { element: "lcm_enabled", key: "lcm_enabled", type: "checkbox" },
        { element: "ip_adapter_enabled", key: "ip_adapter_enabled", type: "checkbox" },
        { element: "ip_adapter_scale", key: "ip_adapter_scale", type: "number", fallback: 0.6 },
    ],
});

window.generateInpaint = page.generate;
GenerationPage.runWhenDomReady(page.init);
