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
        const res = await fetch(`${API_BASE}/models?family=qwen-image`);
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
        fallback.value = "qwen-image";
        fallback.textContent = "qwen-image (diffusers)";
        fallback.selected = true;
        select.appendChild(fallback);
        console.warn("Failed to load models:", error);
    }
}

loadModels();

async function generate() {
    const prompt = document.getElementById("prompt").value;
    const steps = Number(document.getElementById("steps").value);
    const true_cfg_scale = Number(document.getElementById("true_cfg").value);
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

    const payload = {
        prompt,
        negative_prompt,
        steps,
        true_cfg_scale,
        guidance_scale,
        scheduler,
        seed,
        width,
        height,
        model,
        num_images,
    };
    console.log("Generate payload", payload);

    try {
        const res = await fetch(`${API_BASE}/api/qwen-image/text2img`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Qwen-Image request failed (${res.status}): ${errorText}`);
        }

        const data = await res.json();
        gallery.setImages(Array.isArray(data.images) ? data.images : []);
    } catch (error) {
        console.warn("Failed to generate Qwen-Image images:", error);
        gallery.setImages([]);
    }
}
