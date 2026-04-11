function createVideoGalleryViewer(options = {}) {
    const config = {
        videoId: "viewer-video",
        emptyId: "viewer-empty",
        countId: "viewer-count",
        prevId: "viewer-prev",
        nextId: "viewer-next",
        thumbsId: "viewer-thumbs",
        buildVideoUrl: null,
        ...options,
    };

    const viewerVideo = document.getElementById(config.videoId);
    const viewerEmpty = document.getElementById(config.emptyId);
    const viewerCount = document.getElementById(config.countId);
    const viewerPrev = document.getElementById(config.prevId);
    const viewerNext = document.getElementById(config.nextId);
    const viewerThumbs = document.getElementById(config.thumbsId);

    const state = {
        currentVideos: [],
        currentIndex: 0,
        batchStamp: Date.now(),
    };

    function buildUrl(path, idx) {
        if (typeof config.buildVideoUrl === "function") {
            return config.buildVideoUrl(path, idx, state.batchStamp);
        }
        return path;
    }

    function renderViewer() {
        if (!state.currentVideos.length) {
            viewerVideo.style.display = "none";
            viewerVideo.removeAttribute("src");
            viewerEmpty.style.display = "block";
            viewerCount.textContent = "0 / 0";
            viewerPrev.disabled = true;
            viewerNext.disabled = true;
            viewerThumbs.innerHTML = "";
            return;
        }

        const currentUrl = buildUrl(state.currentVideos[state.currentIndex], state.currentIndex);
        viewerEmpty.style.display = "none";
        viewerVideo.style.display = "block";
        if (viewerVideo.src !== currentUrl) {
            viewerVideo.src = currentUrl;
            viewerVideo.load();
        }
        viewerCount.textContent = `${state.currentIndex + 1} / ${state.currentVideos.length}`;
        viewerPrev.disabled = state.currentIndex === 0;
        viewerNext.disabled = state.currentIndex === state.currentVideos.length - 1;

        viewerThumbs.innerHTML = "";
        state.currentVideos.forEach((_path, idx) => {
            const thumb = document.createElement("button");
            thumb.type = "button";
            thumb.className = `secondary${idx === state.currentIndex ? " is-active" : ""}`;
            thumb.textContent = `Video ${idx + 1}`;
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
        if (state.currentIndex < state.currentVideos.length - 1) {
            state.currentIndex += 1;
            renderViewer();
        }
    });

    return {
        render: renderViewer,
        setVideos(videos) {
            state.currentVideos = Array.isArray(videos) ? videos : [];
            state.currentIndex = 0;
            state.batchStamp = Date.now();
            renderViewer();
        },
    };
}
