const gallery = createGalleryViewer({
    buildImageUrl: (path, idx, stamp) => {
        return "http://127.0.0.1:8000" + path + `?t=${stamp}_${idx}`;
    },
});

gallery.render();

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

    const res = await fetch("http://127.0.0.1:8000/img2img", {
        method: "POST",
        body: formData,
    });

    const data = await res.json();
    gallery.setImages(Array.isArray(data.images) ? data.images : []);
}
