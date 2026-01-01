const NAV_ITEMS = [
    { href: "index.html", label: "Text2Img" },
    { href: "img2img.html", label: "Img2Img" },
    { href: "inpaint.html", label: "Inpaint" },
    { href: "sdxl.html", label: "SDXL Text2Img" },
    { href: "sdxl_img2img.html", label: "SDXL Img2Img" },
    { href: "sdxl_inpaint.html", label: "SDXL Inpaint" },
    { href: "z_image.html", label: "Z-Image Text2Img" },
    { href: "z_image_img2img.html", label: "Z-Image Img2Img" },
    { href: "flux.html", label: "Flux Text2Img" },
    { href: "flux_img2img.html", label: "Flux Img2Img" },
    { href: "flux_inpaint.html", label: "Flux Inpaint" },
    { href: "models.html", label: "Models" },
    { href: "model_add.html", label: "Add Model" },
    { href: "history.html", label: "History" },
];

function renderNavBar() {
    const navRoot = document.getElementById("nav-root");
    if (!navRoot) {
        return;
    }

    const nav = document.createElement("nav");
    nav.className = "header-nav";

    const currentPath = window.location.pathname.split("/").pop() || "index.html";

    NAV_ITEMS.forEach((item) => {
        const link = document.createElement("a");
        link.className = "secondary nav-link";
        link.href = item.href;
        link.textContent = item.label;
        if (item.href === currentPath) {
            link.classList.add("is-active");
        }
        nav.appendChild(link);
    });

    navRoot.appendChild(nav);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderNavBar);
} else {
    renderNavBar();
}
