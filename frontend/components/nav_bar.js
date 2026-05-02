const navScriptUrl = document.currentScript?.src ? new URL(document.currentScript.src) : null;
const frontendRootUrl = navScriptUrl ? new URL("../", navScriptUrl) : null;
const DEFAULT_NAV_PAGE = "sd15/text2img.html";

function resolveNavHref(path) {
    return frontendRootUrl ? new URL(path, frontendRootUrl).toString() : path;
}

function normalizeNavPath(path) {
    try {
        const base = frontendRootUrl?.toString() ?? window.location.href;
        return new URL(path, base).pathname.replace(/^\/+/, "");
    } catch (_error) {
        return path.replace(/^\/+/, "");
    }
}

const NAV_GROUPS = [
    {
        label: "Render",
        items: [
            { href: "sd15/text2img.html", label: "SD 1.5 Text2Img" },
            { href: "sd15/animatediff.html", label: "SD 1.5 AnimateDiff" },
            { href: "wan/text2video.html", label: "WAN Text2Video" },
            { href: "wan/image2video.html", label: "WAN I2V 14B" },
            { href: "sd15/img2img.html", label: "SD 1.5 Img2Img" },
            { href: "sd15/inpainting.html", label: "SD 1.5 Inpaint" },
            { href: "sdxl/text2img.html", label: "SDXL Text2Img" },
            { href: "sdxl/img2img.html", label: "SDXL Img2Img" },
            { href: "sdxl/inpaint.html", label: "SDXL Inpaint" },
            { href: "z_image/text2img.html", label: "Z-Image Text2Img" },
            { href: "z_image/img2img.html", label: "Z-Image Img2Img" },
            { href: "z_image/inpaint.html", label: "Z-Image Inpaint" },
            { href: "qwen_image/text2img.html", label: "Qwen-Image Text2Img" },
            { href: "qwen_image/img2img.html", label: "Qwen-Image Img2Img" },
            { href: "qwen_image/inpaint.html", label: "Qwen-Image Inpaint" },
            { href: "flux/text2img.html", label: "Flux Text2Img" },
            { href: "flux/img2img.html", label: "Flux Img2Img" },
            { href: "flux/inpaint.html", label: "Flux Inpaint" },
        ],
    },
    {
        label: "Models",
        items: [
            { href: "models/base/registry.html", label: "Base Models" },
            { href: "models/lora/model_page.html", label: "LoRA Models" },
            { href: "models/base/add.html", label: "Add Base Model" },
            { href: "models/lora/add.html", label: "Add LoRA" },
            { href: "others/tools_analysis.html", label: "Tools & Analysis" },
        ],
    },
    {
        label: "History",
        items: [{ href: "others/history.html", label: "History" }],
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

    const currentPath = window.location.pathname.replace(/^\/+/, "") || DEFAULT_NAV_PAGE;

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
            link.href = resolveNavHref(item.href);
            link.textContent = item.label;
            link.style.justifyContent = "center";
            link.style.textAlign = "center";
            link.setAttribute("role", "menuitem");
            if (normalizeNavPath(item.href) === currentPath) {
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
