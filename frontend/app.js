const viewerImage = document.getElementById("viewer-image");
const viewerEmpty = document.getElementById("viewer-empty");
const viewerCount = document.getElementById("viewer-count");
const viewerPrev = document.getElementById("viewer-prev");
const viewerNext = document.getElementById("viewer-next");
const viewerThumbs = document.getElementById("viewer-thumbs");

let currentImages = [];
let currentIndex = 0;
let batchStamp = null;

function buildImageUrl(path, idx) {
    return "http://127.0.0.1:8000" + path + `?t=${batchStamp}_${idx}`;
}

function renderViewer() {
    if (!currentImages.length) {
        viewerImage.style.display = "none";
        viewerEmpty.style.display = "block";
        viewerCount.textContent = "0 / 0";
        viewerPrev.disabled = true;
        viewerNext.disabled = true;
        viewerThumbs.innerHTML = "";
        return;
    }

    viewerEmpty.style.display = "none";
    viewerImage.style.display = "block";
    viewerImage.src = buildImageUrl(currentImages[currentIndex], currentIndex);
    viewerCount.textContent = `${currentIndex + 1} / ${currentImages.length}`;
    viewerPrev.disabled = currentIndex === 0;
    viewerNext.disabled = currentIndex === currentImages.length - 1;

    viewerThumbs.innerHTML = "";
    currentImages.forEach((path, idx) => {
        const thumb = document.createElement("img");
        thumb.src = buildImageUrl(path, idx);
        thumb.className = `viewer-thumb${idx === currentIndex ? " is-active" : ""}`;
        thumb.alt = `Thumbnail ${idx + 1}`;
        thumb.addEventListener("click", () => {
            currentIndex = idx;
            renderViewer();
        });
        viewerThumbs.appendChild(thumb);
    });
}

viewerPrev.addEventListener("click", () => {
    if (currentIndex > 0) {
        currentIndex -= 1;
        renderViewer();
    }
});

viewerNext.addEventListener("click", () => {
    if (currentIndex < currentImages.length - 1) {
        currentIndex += 1;
        renderViewer();
    }
});

async function generate() {
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

    const num_images = Number(document.getElementById("num_images").value);

    const payload = {
        prompt,
        negative_prompt,
        steps,
        cfg,
        scheduler,
        seed,
        width,
        height,
        num_images,
    };
    console.log("Generate payload", payload);

    const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    currentImages = Array.isArray(data.images) ? data.images : [];
    currentIndex = 0;
    batchStamp = Date.now();
    renderViewer();
}

renderViewer();
