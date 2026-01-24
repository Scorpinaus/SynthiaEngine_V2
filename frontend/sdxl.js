const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return API_BASE + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

let activeJobToken = 0;
let activeEventSource = null;

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

function setJobUiState(isBusy, message) {
    const button = document.getElementById("generate_button");

    if (button) {
        button.disabled = Boolean(isBusy);
        button.textContent = isBusy ? "Generating..." : "Generate";
    }
}

function closeActiveEventSource() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

async function submitJob(kind, payload) {
    const idempotencyKey = typeof crypto?.randomUUID === "function"
        ? crypto.randomUUID()
        : `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    const res = await fetch(`${API_BASE}/api/jobs`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotencyKey,
        },
        body: JSON.stringify({ kind, payload }),
    });

    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Job submit failed (${res.status}): ${errorText}`);
    }

    return await res.json();
}

async function generate() {
    const token = ++activeJobToken;
    closeActiveEventSource();
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
    const clip_skip = Number(document.getElementById("clip_skip").value);
    const hires_enabled = Boolean(document.getElementById("hires_enabled")?.checked);
    const hiresScaleInput = Number(document.getElementById("hires_scale").value);
    const hires_scale = Number.isFinite(hiresScaleInput) ? hiresScaleInput : 1.0;

    const payload = {
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        scheduler,
        seed,
        width,
        height,
        model,
        num_images,
        clip_skip,
        hires_enabled,
        hires_scale,
    };
    console.log("Generate payload", payload);

    try {
        setJobUiState(true, "Submitting job...");
        const createdJob = await submitJob("sdxl_text2img", payload);
        const jobId = createdJob?.id;
        if (!jobId) {
            throw new Error("Job submit did not return an id.");
        }

        setJobUiState(true, `Queued (job ${jobId})`);

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
            } catch (parseError) {
                console.warn("Failed to parse job event:", parseError);
                return;
            }

            const status = job?.status ?? "unknown";
            if (status === "queued") {
                setJobUiState(true, `Queued (job ${jobId})`);
            } else if (status === "running") {
                setJobUiState(true, `Running (job ${jobId})`);
            } else if (status === "succeeded") {
                const images = job?.result?.images;
                gallery.setImages(Array.isArray(images) ? images : []);
                setJobUiState(false, `Done (job ${jobId})`);
                source.close();
            } else if (status === "failed") {
                const err = job?.error ?? "Unknown error.";
                setJobUiState(false, `Failed (job ${jobId})`);
                gallery.setImages([]);
                source.close();
                console.warn("Job failed:", err);
            } else if (status === "canceled") {
                setJobUiState(false, `Canceled (job ${jobId})`);
                gallery.setImages([]);
                source.close();
            } else {
                setJobUiState(true, `Status: ${status} (job ${jobId})`);
            }
        };

        source.onerror = () => {
            if (token !== activeJobToken) {
                source.close();
                return;
            }
            setJobUiState(false, "Job update stream lost.");
            source.close();
        };
    } catch (error) {
        console.warn("Failed to generate SDXL images:", error);
        gallery.setImages([]);
        setJobUiState(false, "Failed to generate.");
    }
}
