const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return "http://127.0.0.1:8000" + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch("http://127.0.0.1:8000/models");
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
        fallback.value = "stable-diffusion-xl-base-1.0";
        fallback.textContent = "stable-diffusion-xl-base-1.0 (sdxl, diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

async function generate() {
    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const guidance_scale = Number(document.getElementById("cfg").value);
    const seedValue = document.getElementById("seed").value;
    const seedNumber = seedValue === "" ? null : Number(seedValue);
    const seed = Number.isFinite(seedNumber) ? seedNumber : null;
    const negative_prompt = document.getElementById("negative_prompt").value;
    const width = Number(document.getElementById("width").value);
    const height = Number(document.getElementById("height").value);
    const model = document.getElementById("model_select").value;
    const num_images = Number(document.getElementById("num_images").value);
    const clip_skip = Number(document.getElementById("clip_skip").value);

    const payload = {
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        seed,
        width,
        height,
        model,
        num_images,
        clip_skip,
    };
    console.log("Generate payload", payload);

    const res = await fetch("http://127.0.0.1:8000/api/sdxl/text2img", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}
