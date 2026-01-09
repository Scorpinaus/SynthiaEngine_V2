const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=sd15`);
        const models = await res.json();

        if (!Array.isArray(models) || models.length === 0) {
            throw new Error("No models returned.");
        }

        models.forEach((model, index) => {
            const option = document.createElement("option");
            option.value = model.name ?? "";
            const family = model.family ?? "unknown";
            const modelType = model.model_type ?? "unknown";
            option.textContent = `${model.name} (${family}, ${modelType})`;
            if (index === 0) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    } catch (error) {
        const fallback = document.createElement("option");
        fallback.value = "stable-diffusion-v1-5";
        fallback.textContent = "stable-diffusion-v1-5 (sd15, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

window.LoraPanel?.init({ apiBase: API_BASE, family: "sd15" });

async function generateImg2Img() {
    const initialImageInput = document.getElementById("initial_image");
    const initialFile = initialImageInput.files[0];

    if (!initialFile) {
        alert("Please select an initial image.");
        return;
    }

    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const cfg = Number(document.getElementById("cfg").value);
    const scheduler = document.getElementById("scheduler").value;
    const seedValue = document.getElementById("seed").value;
    const seedNumber = seedValue === "" ? null : Number(seedValue);
    const seed = Number.isFinite(seedNumber) ? seedNumber : null;
    const negative_prompt = document.getElementById("negative_prompt").value;
    const width = Number(document.getElementById("width").value);
    const height = Number(document.getElementById("height").value);
    const strength = Number(document.getElementById("strength").value);
    const num_images = Number(document.getElementById("num_images").value);
    const model = document.getElementById("model_select").value;
    const clip_skip = Number(document.getElementById("clip_skip").value);
    const loraAdapters = window.LoraPanel?.getSelectedAdapters?.() ?? [];

    const formData = new FormData();
    formData.append("initial_image", initialFile);
    formData.append("prompt", prompt);
    formData.append("negative_prompt", negative_prompt);
    formData.append("steps", steps.toString());
    formData.append("cfg", cfg.toString());
    formData.append("scheduler", scheduler);
    formData.append("seed", seed === null ? "" : seed.toString());
    formData.append("width", width.toString());
    formData.append("height", height.toString());
    formData.append("strength", strength.toString());
    formData.append("num_images", num_images.toString());
    formData.append("model", model);
    formData.append("clip_skip", clip_skip);
    if (loraAdapters.length > 0) {
        formData.append("lora_adapters", JSON.stringify(loraAdapters));
    }

    const res = await fetch(`${API_BASE}/generate-img2img`, {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}
