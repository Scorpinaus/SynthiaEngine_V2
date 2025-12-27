function createGalleryViewer(options = {}) {
    const config = {
        imageId: "viewer-image",
        emptyId: "viewer-empty",
        countId: "viewer-count",
        prevId: "viewer-prev",
        nextId: "viewer-next",
        thumbsId: "viewer-thumbs",
        buildImageUrl: null,
        ...options,
    };

    const viewerImage = document.getElementById(config.imageId);
    const viewerEmpty = document.getElementById(config.emptyId);
    const viewerCount = document.getElementById(config.countId);
    const viewerPrev = document.getElementById(config.prevId);
    const viewerNext = document.getElementById(config.nextId);
    const viewerThumbs = document.getElementById(config.thumbsId);

    const state = {
        currentImages: [],
        currentIndex: 0,
        batchStamp: Date.now(),
    };

    function buildUrl(path, idx) {
        if (typeof config.buildImageUrl === "function") {
            return config.buildImageUrl(path, idx, state.batchStamp);
        }
        return path;
    }

    function renderViewer() {
        if (!state.currentImages.length) {
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
        viewerImage.src = buildUrl(state.currentImages[state.currentIndex], state.currentIndex);
        viewerCount.textContent = `${state.currentIndex + 1} / ${state.currentImages.length}`;
        viewerPrev.disabled = state.currentIndex === 0;
        viewerNext.disabled = state.currentIndex === state.currentImages.length - 1;

        viewerThumbs.innerHTML = "";
        state.currentImages.forEach((path, idx) => {
            const thumb = document.createElement("img");
            thumb.src = buildUrl(path, idx);
            thumb.className = `viewer-thumb${idx === state.currentIndex ? " is-active" : ""}`;
            thumb.alt = `Thumbnail ${idx + 1}`;
            thumb.addEventListener("click", () => {
                state.currentIndex = idx;
                renderViewer();
            });
            viewerThumbs.appendChild(thumb);
        });
    }

    viewerPrev.addEventListener("click", () => {
        if (state.currentIndex > 0) {
            state.currentIndex -= 1;
            renderViewer();
        }
    });

    viewerNext.addEventListener("click", () => {
        if (state.currentIndex < state.currentImages.length - 1) {
            state.currentIndex += 1;
            renderViewer();
        }
    });

    return {
        render: renderViewer,
        setImages(images) {
            state.currentImages = Array.isArray(images) ? images : [];
            state.currentIndex = 0;
            state.batchStamp = Date.now();
            renderViewer();
        },
    };
}
