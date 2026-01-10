const NAV_GROUPS = [
    {
        label: "Render",
        items: [
            { href: "sd15.html", label: "SD 1.5 Text2Img" },
            { href: "sd15_img2img.html", label: "SD 1.5 Img2Img" },
            { href: "sd15_inpainting.html", label: "SD 1.5 Inpaint" },
            { href: "sdxl.html", label: "SDXL Text2Img" },
            { href: "sdxl_img2img.html", label: "SDXL Img2Img" },
            { href: "sdxl_inpaint.html", label: "SDXL Inpaint" },
            { href: "z_image.html", label: "Z-Image Text2Img" },
            { href: "z_image_img2img.html", label: "Z-Image Img2Img" },
            { href: "qwen_image.html", label: "Qwen-Image Text2Img" },
            { href: "qwen_image_img2img.html", label: "Qwen-Image Img2Img" },
            { href: "qwen_image_inpaint.html", label: "Qwen-Image Inpaint" },
            { href: "flux.html", label: "Flux Text2Img" },
            { href: "flux_img2img.html", label: "Flux Img2Img" },
        ],
    },
    {
        label: "Models",
        items: [
            { href: "models.html", label: "Models" },
            { href: "model_add.html", label: "Add Model" },
            { href: "tools_analysis.html", label: "Tools & Analysis" },
        ],
    },
    {
        label: "History",
        items: [{ href: "history.html", label: "History" }],
    },
];

function renderNavBar() {
    const navRoot = document.getElementById("nav-root");
    if (!navRoot) {
        return false;
    }
    if (navRoot.querySelector(".header-nav")) {
        return true;
    }
    navRoot.innerHTML = "";

    const nav = document.createElement("nav");
    nav.className = "header-nav";

    const currentPath = window.location.pathname.split("/").pop() || "sd15.html";

    NAV_GROUPS.forEach((group) => {
        const groupWrap = document.createElement("div");
        groupWrap.className = "nav-group";

        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "secondary nav-group-toggle";
        toggle.textContent = group.label;
        toggle.setAttribute("aria-haspopup", "true");
        toggle.setAttribute("aria-expanded", "false");

        const menu = document.createElement("div");
        menu.className = "nav-group-menu";
        menu.setAttribute("role", "menu");

        group.items.forEach((item) => {
            const link = document.createElement("a");
            link.className = "secondary nav-link";
            link.href = item.href;
            link.textContent = item.label;
            link.style.justifyContent = "center";
            link.style.textAlign = "center";
            link.setAttribute("role", "menuitem");
            if (item.href === currentPath) {
                link.classList.add("is-active");
            }
            menu.appendChild(link);
        });

        groupWrap.append(toggle, menu);
        nav.appendChild(groupWrap);
    });

    const groups = Array.from(nav.querySelectorAll(".nav-group"));

    function closeAllGroups() {
        groups.forEach((group) => {
            group.classList.remove("is-open");
            const button = group.querySelector(".nav-group-toggle");
            if (button) {
                button.setAttribute("aria-expanded", "false");
            }
        });
    }

    groups.forEach((group) => {
        const toggle = group.querySelector(".nav-group-toggle");
        toggle?.addEventListener("click", (event) => {
            event.stopPropagation();
            const isOpen = group.classList.toggle("is-open");
            toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
            groups.forEach((other) => {
                if (other !== group) {
                    other.classList.remove("is-open");
                    const otherToggle = other.querySelector(".nav-group-toggle");
                    otherToggle?.setAttribute("aria-expanded", "false");
                }
            });
        });
    });

    document.addEventListener("click", closeAllGroups);
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeAllGroups();
        }
    });

    navRoot.appendChild(nav);
    return true;
}

function initNavBar() {
    if (renderNavBar()) {
        return;
    }
    const observer = new MutationObserver(() => {
        if (renderNavBar()) {
            observer.disconnect();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initNavBar);
} else {
    initNavBar();
}

document.addEventListener("header:loaded", () => {
    renderNavBar();
});
