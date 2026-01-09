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
        const res = await fetch(`${API_BASE}/models?family=flux`);
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
        fallback.value = "black-forest-labs/FLUX.1-schnell";
        fallback.textContent = "black-forest-labs/FLUX.1-schnell (flux, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

async function generateFluxImg2Img() {
    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const guidance_scale = Number(document.getElementById("cfg").value);
    const scheduler = document.getElementById("scheduler")?.value ?? "euler";
    const seedValue = document.getElementById("seed").value;
    const seedNumber = seedValue === "" ? null : Number(seedValue);
    const seed = Number.isFinite(seedNumber) ? seedNumber : null;
    const negative_prompt = document.getElementById("negative_prompt").value;
    const width = Number(document.getElementById("width").value);
    const height = Number(document.getElementById("height").value);
    const model = document.getElementById("model_select").value;
    const num_images = Number(document.getElementById("num_images").value);
    const strength = Number(document.getElementById("strength").value);
    const initialImageInput = document.getElementById("initial_image");

    if (!initialImageInput.files || initialImageInput.files.length === 0) {
        alert("Please choose an initial image.");
        return;
    }

    const formData = new FormData();
    formData.append("initial_image", initialImageInput.files[0]);
    formData.append("prompt", prompt);
    formData.append("negative_prompt", negative_prompt);
    formData.append("steps", String(steps));
    formData.append("guidance_scale", String(guidance_scale));
    formData.append("scheduler", scheduler);
    formData.append("width", String(width));
    formData.append("height", String(height));
    formData.append("seed", seed === null ? "" : String(seed));
    formData.append("num_images", String(num_images));
    formData.append("model", model);
    formData.append("strength", String(strength));

    const res = await fetch(`${API_BASE}/api/flux/img2img`, {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}
