const TASK_TEXT2IMG = "sd15.text2img";
const TASK_CONTROLNET_TEXT2IMG = "sd15.controlnet.text2img";
const TASK_HIRES_FIX = "sd15.hires_fix";
const TASK_IP_ADAPTER_ENCODE = "sd15.ip_adapter.encode";

const page = Sd15GenerationController.createText2Img({
    taskText2Img: TASK_TEXT2IMG,
    taskControlNet: TASK_CONTROLNET_TEXT2IMG,
    taskHires: TASK_HIRES_FIX,
    taskIpAdapter: TASK_IP_ADAPTER_ENCODE,
    fields: [
        { element: "prompt", key: "prompt", fallback: "" },
        { element: "negative_prompt", key: "negative_prompt", fallback: "" },
        { element: "steps", key: "steps", type: "number", integer: true, fallback: 20 },
        { element: "cfg", key: "cfg", type: "number", fallback: 7.5 },
        { element: "scheduler", key: "scheduler", fallback: "euler" },
        { element: "seed", key: "seed", type: "seed" },
        { element: "width", key: "width", type: "number", integer: true, fallback: 512 },
        { element: "height", key: "height", type: "number", integer: true, fallback: 512 },
        { element: "hires_enabled", key: "hires_enabled", type: "checkbox" },
        { element: "hires_scale", key: "hires_scale", type: "number", fallback: 1.0 },
        { element: "model_select", key: "model", fallback: null },
        { element: "clip_skip", key: "clip_skip", type: "number", integer: true, fallback: 1 },
        { element: "num_images", key: "num_images", type: "number", integer: true, fallback: 1 },
        { element: "weighting_policy", key: "weighting_policy", fallback: "diffusers-like" },
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
        { element: "lcm_enabled", key: "lcm_enabled", type: "checkbox" },
    ],
});

GenerationPage.runWhenDomReady(page.init);
