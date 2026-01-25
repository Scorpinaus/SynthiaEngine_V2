const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

function makeIdempotencyKey() {
    if (typeof crypto?.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

async function uploadArtifact(blobOrFile, filename = "upload.png") {
    const formData = new FormData();
    formData.append("file", blobOrFile, filename);
    const res = await fetch(`${API_BASE}/api/artifacts`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Artifact upload failed (${res.status}): ${errorText}`);
    }
    return await res.json();
}

async function submitWorkflow(payload, idempotencyKey) {
    const res = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotencyKey,
        },
        body: JSON.stringify({ kind: "workflow", payload }),
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Job submit failed (${res.status}): ${errorText}`);
    }
    return await res.json();
}

async function loadModels() {
    const select = document.getElementById("model_select");
    select.innerHTML = "";
    try {
        const res = await fetch(`${API_BASE}/models?family=sdxl`);
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

async function generateSdxlImg2Img() {
    const token = ++activeJobToken;
    closeActiveEventSource();

    const initialImageInput = document.getElementById("initial_image");
    const initialFile = initialImageInput.files[0];

    if (!initialFile) {
        alert("Please select an initial image.");
        return;
    }

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
    const strength = Number(document.getElementById("strength").value);
    const num_images = Number(document.getElementById("num_images").value);
    const model = document.getElementById("model_select").value;
    const clip_skip = Number(document.getElementById("clip_skip").value);

    try {
        const uploaded = await uploadArtifact(initialFile, initialFile.name || "initial.png");

        const workflowPayload = {
            tasks: [
                {
                    id: "t1",
                    type: "sdxl.img2img",
                    inputs: {
                        initial_image: `@artifact:${uploaded.artifact_id}`,
                        prompt,
                        negative_prompt,
                        steps,
                        guidance_scale,
                        scheduler,
                        seed,
                        width,
                        height,
                        strength,
                        num_images,
                        model,
                        clip_skip,
                    },
                },
            ],
            return: "@t1.images",
        };

        const createdJob = await submitWorkflow(workflowPayload, makeIdempotencyKey());
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        const source = new EventSource(`${API_BASE}/api/jobs/${jobId}/events`);
        activeEventSource = source;

        source.onmessage = (event) => {
            if (token !== activeJobToken) {
                source.close();
                return;
            }

            let job;
            try {
                job = JSON.parse(event.data);
            } catch {
                return;
            }

            const status = job?.status ?? "unknown";
            if (status === "succeeded") {
                const images = job?.result?.outputs;
                gallery.setImages(Array.isArray(images) ? images : []);
                source.close();
            } else if (status === "failed" || status === "canceled") {
                gallery.setImages([]);
                source.close();
            }
        };

        source.onerror = () => {
            if (token !== activeJobToken) {
                source.close();
                return;
            }
            source.close();
        };
    } catch (error) {
        console.warn("Failed to run SDXL img2img job:", error);
        gallery.setImages([]);
    }
}
