const POLL_MS = 5000;
const MAX_POINTS = 240;
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled"]);

const state = {
    jobs: [],
    selectedJobId: null,
    selectedJob: null,
    samples: [],
    eventSource: null,
    pollTimer: null,
};

const activeJobsEl = document.getElementById("profiler-active-jobs");
const recentJobsEl = document.getElementById("profiler-recent-jobs");
const jobsStatusEl = document.getElementById("profiler-jobs-status");
const subtitleEl = document.getElementById("profiler-subtitle");
const refreshBtn = document.getElementById("profiler-refresh");
const jobTitleEl = document.getElementById("profiler-job-title");
const jobDetailEl = document.getElementById("profiler-job-detail");
const jobStatusEl = document.getElementById("profiler-job-status");

const metricEls = {
    elapsed: document.getElementById("metric-elapsed"),
    elapsedSub: document.getElementById("metric-elapsed-sub"),
    ramCurrent: document.getElementById("metric-ram-current"),
    ramSub: document.getElementById("metric-ram-sub"),
    cudaCurrent: document.getElementById("metric-cuda-current"),
    cudaSub: document.getElementById("metric-cuda-sub"),
    nvmlCurrent: document.getElementById("metric-nvml-current"),
    nvmlSub: document.getElementById("metric-nvml-sub"),
};

const charts = {
    ram: {
        canvas: document.getElementById("chart-ram"),
        range: document.getElementById("chart-ram-range"),
        series: [{ key: "ram", color: "#2563eb", label: "RSS" }],
    },
    cuda: {
        canvas: document.getElementById("chart-cuda"),
        range: document.getElementById("chart-cuda-range"),
        series: [
            { key: "cudaAllocated", color: "#7c3aed", label: "Allocated" },
            { key: "cudaReserved", color: "#0891b2", label: "Reserved" },
        ],
    },
    nvml: {
        canvas: document.getElementById("chart-nvml"),
        range: document.getElementById("chart-nvml-range"),
        series: [{ key: "nvml", color: "#dc2626", label: "Used" }],
    },
};

function setJobsStatus(text) {
    if (jobsStatusEl) {
        jobsStatusEl.textContent = text;
    }
}

function formatJobId(job) {
    const id = String(job?.id || "");
    return id.length > 8 ? id.slice(0, 8) : id || "-";
}

function formatTime(timestamp) {
    if (!timestamp) {
        return "-";
    }
    return new Date(timestamp).toLocaleTimeString();
}

function formatSeconds(value) {
    if (!Number.isFinite(value)) {
        return "-";
    }
    if (value < 60) {
        return `${value.toFixed(1)}s`;
    }
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60).toString().padStart(2, "0");
    return `${minutes}:${seconds}`;
}

function formatMb(value) {
    if (!Number.isFinite(value)) {
        return "-";
    }
    if (value >= 1024) {
        return `${(value / 1024).toFixed(2)} GB`;
    }
    return `${value.toFixed(0)} MB`;
}

function valueOf(profile, keys) {
    for (const key of keys) {
        const value = Number(profile?.[key]);
        if (Number.isFinite(value)) {
            return value;
        }
    }
    return null;
}

function profileToSample(profile) {
    if (!profile) {
        return null;
    }
    const elapsed = valueOf(profile, ["elapsed_seconds"]);
    if (elapsed === null) {
        return null;
    }
    return {
        elapsed,
        ram: valueOf(profile, ["rss_current_mb", "rss_after_mb", "rss_before_mb"]),
        cudaAllocated: valueOf(profile, ["cuda_allocated_current_mb", "cuda_peak_allocated_mb"]),
        cudaReserved: valueOf(profile, ["cuda_reserved_current_mb", "cuda_peak_reserved_mb"]),
        nvml: valueOf(profile, ["nvml_used_current_mb", "nvml_used_end_mb", "nvml_used_start_mb"]),
    };
}

function appendProfile(profile) {
    const sample = profileToSample(profile);
    if (!sample) {
        return;
    }
    const last = state.samples[state.samples.length - 1];
    if (last && Math.abs(last.elapsed - sample.elapsed) < 0.001) {
        state.samples[state.samples.length - 1] = sample;
    } else {
        state.samples.push(sample);
    }
    if (state.samples.length > MAX_POINTS) {
        state.samples.splice(0, state.samples.length - MAX_POINTS);
    }
}

function mergeJob(job) {
    if (!job?.id) {
        return;
    }
    const existingIndex = state.jobs.findIndex((item) => item.id === job.id);
    if (existingIndex >= 0) {
        state.jobs[existingIndex] = { ...state.jobs[existingIndex], ...job };
    } else {
        state.jobs.unshift(job);
    }
    if (state.selectedJobId === job.id) {
        state.selectedJob = { ...(state.selectedJob || {}), ...job };
    }
}

function clearEventSource() {
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }
}

function renderJobList(container, jobs) {
    if (!container) {
        return;
    }
    container.innerHTML = "";
    if (!jobs.length) {
        const empty = document.createElement("div");
        empty.className = "profiler-empty";
        empty.textContent = "No jobs";
        container.appendChild(empty);
        return;
    }
    jobs.forEach((job) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "profiler-job";
        button.classList.toggle("is-active", job.id === state.selectedJobId);
        button.setAttribute("aria-pressed", job.id === state.selectedJobId ? "true" : "false");
        button.addEventListener("click", () => selectJob(job.id));

        const title = document.createElement("span");
        title.className = "profiler-job-name";
        title.textContent = `${job.kind || "job"} ${formatJobId(job)}`;

        const meta = document.createElement("span");
        meta.className = "profiler-job-meta";
        meta.textContent = `${job.status || "unknown"} | ${formatTime(job.updated_at || job.created_at)}`;

        button.append(title, meta);
        container.appendChild(button);
    });
}

function renderJobs() {
    const active = state.jobs.filter((job) => ["queued", "running"].includes(job.status));
    const recent = state.jobs.filter((job) => TERMINAL_STATUSES.has(job.status));
    renderJobList(activeJobsEl, active);
    renderJobList(recentJobsEl, recent.slice(0, 20));
}

function renderSelectedJob() {
    const job = state.selectedJob;
    const profile = job?.result?.profile || null;
    if (!job) {
        subtitleEl.textContent = "No job selected.";
        jobTitleEl.textContent = "No job selected";
        jobDetailEl.textContent = "Waiting for jobs.";
        jobStatusEl.textContent = "Idle";
        jobStatusEl.dataset.status = "idle";
        renderMetrics(null);
        drawCharts();
        return;
    }

    subtitleEl.textContent = `Job ${formatJobId(job)} | ${job.status || "unknown"}`;
    jobTitleEl.textContent = `${job.kind || "job"} ${job.id}`;
    jobDetailEl.textContent = `Created ${formatTime(job.created_at)} | Updated ${formatTime(job.updated_at)}`;
    jobStatusEl.textContent = job.status || "unknown";
    jobStatusEl.dataset.status = job.status || "unknown";
    renderMetrics(profile);
    drawCharts();
}

function renderMetrics(profile) {
    const elapsed = valueOf(profile, ["elapsed_seconds"]);
    const ramCurrent = valueOf(profile, ["rss_current_mb", "rss_after_mb", "rss_before_mb"]);
    const ramPeak = valueOf(profile, ["rss_peak_sampled_mb"]);
    const ramStart = valueOf(profile, ["rss_before_mb"]);
    const ramEnd = valueOf(profile, ["rss_after_mb"]);
    const cudaCurrent = valueOf(profile, ["cuda_allocated_current_mb", "cuda_peak_allocated_mb"]);
    const cudaPeak = valueOf(profile, ["cuda_peak_allocated_mb"]);
    const cudaReserved = valueOf(profile, ["cuda_reserved_current_mb", "cuda_peak_reserved_mb"]);
    const cudaReservedPeak = valueOf(profile, ["cuda_peak_reserved_mb"]);
    const nvmlCurrent = valueOf(profile, ["nvml_used_current_mb", "nvml_used_end_mb", "nvml_used_start_mb"]);
    const nvmlPeak = valueOf(profile, ["nvml_used_peak_sampled_mb"]);
    const nvmlStart = valueOf(profile, ["nvml_used_start_mb"]);
    const nvmlEnd = valueOf(profile, ["nvml_used_end_mb"]);

    metricEls.elapsed.textContent = formatSeconds(elapsed);
    metricEls.elapsedSub.textContent = profile ? `schema v${profile.schema_version || 1}` : "-";
    metricEls.ramCurrent.textContent = formatMb(ramCurrent);
    metricEls.ramSub.textContent = `peak ${formatMb(ramPeak)} | start ${formatMb(ramStart)} | end ${formatMb(ramEnd)}`;
    metricEls.cudaCurrent.textContent = formatMb(cudaCurrent);
    metricEls.cudaSub.textContent = `peak ${formatMb(cudaPeak)} | reserved ${formatMb(cudaReserved)} / ${formatMb(cudaReservedPeak)}`;
    metricEls.nvmlCurrent.textContent = formatMb(nvmlCurrent);
    metricEls.nvmlSub.textContent = profile?.nvml_available
        ? `peak ${formatMb(nvmlPeak)} | start ${formatMb(nvmlStart)} | end ${formatMb(nvmlEnd)}`
        : "NVML unavailable";
}

function renderAll() {
    renderJobs();
    renderSelectedJob();
}

async function refreshJobs({ keepSelection = true } = {}) {
    try {
        setJobsStatus("Refreshing");
        const response = await fetch(`${API_BASE}/api/jobs?limit=50`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Jobs request failed (${response.status}): ${errorText}`);
        }
        state.jobs = await response.json();
        const selectedStillExists = state.jobs.some((job) => job.id === state.selectedJobId);
        if (!keepSelection || !state.selectedJobId || !selectedStillExists) {
            const running = state.jobs.find((job) => job.status === "running");
            const queued = state.jobs.find((job) => job.status === "queued");
            const recent = state.jobs.find((job) => TERMINAL_STATUSES.has(job.status));
            const next = running || queued || recent || state.jobs[0] || null;
            selectJob(next?.id || null, { fromRefresh: true });
        } else {
            state.selectedJob = state.jobs.find((job) => job.id === state.selectedJobId) || state.selectedJob;
            appendProfile(state.selectedJob?.result?.profile);
        }
        setJobsStatus(`Updated ${new Date().toLocaleTimeString()}`);
        renderAll();
    } catch (error) {
        console.warn("Failed to refresh profiler jobs:", error);
        setJobsStatus("Refresh failed");
    }
}

function selectJob(jobId, { fromRefresh = false } = {}) {
    if (state.selectedJobId === jobId && !fromRefresh) {
        return;
    }
    clearEventSource();
    state.selectedJobId = jobId;
    state.selectedJob = state.jobs.find((job) => job.id === jobId) || null;
    state.samples = [];
    appendProfile(state.selectedJob?.result?.profile);

    if (state.selectedJob && !TERMINAL_STATUSES.has(state.selectedJob.status)) {
        state.eventSource = WorkflowClient.watchJob(API_BASE, state.selectedJob.id, {
            isStale: () => state.selectedJobId !== jobId,
            onUpdate: (job) => {
                mergeJob(job);
                appendProfile(job?.result?.profile);
                renderAll();
            },
            onDone: (job) => {
                mergeJob(job);
                appendProfile(job?.result?.profile);
                renderAll();
            },
            onError: () => {
                if (state.selectedJobId === jobId) {
                    setJobsStatus("Stream lost");
                }
            },
        });
    }
    renderAll();
}

function drawCharts() {
    drawChart(charts.ram);
    drawChart(charts.cuda);
    drawChart(charts.nvml);
}

function drawChart(chart) {
    const canvas = chart.canvas;
    if (!canvas) {
        return;
    }
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const points = state.samples;
    const values = [];
    chart.series.forEach((series) => {
        points.forEach((point) => {
            const value = point[series.key];
            if (Number.isFinite(value)) {
                values.push(value);
            }
        });
    });

    if (!points.length || !values.length) {
        drawEmptyChart(ctx, width, height);
        chart.range.textContent = "-";
        return;
    }

    const minElapsed = points[0].elapsed;
    const maxElapsed = Math.max(points[points.length - 1].elapsed, minElapsed + 1);
    const maxValue = Math.max(...values, 1);
    const minValue = Math.min(0, Math.min(...values));
    const padding = 28 * dpr;
    const chartWidth = Math.max(1, width - padding * 2);
    const chartHeight = Math.max(1, height - padding * 2);

    ctx.strokeStyle = "#e3ddd4";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i <= 3; i += 1) {
        const y = padding + (chartHeight * i) / 3;
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
    }
    ctx.stroke();

    chart.series.forEach((series) => {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        let started = false;
        points.forEach((point) => {
            const value = point[series.key];
            if (!Number.isFinite(value)) {
                return;
            }
            const x = padding + ((point.elapsed - minElapsed) / (maxElapsed - minElapsed)) * chartWidth;
            const y = height - padding - ((value - minValue) / (maxValue - minValue || 1)) * chartHeight;
            if (!started) {
                ctx.moveTo(x, y);
                started = true;
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
    });

    ctx.fillStyle = "#6b645c";
    ctx.font = `${11 * dpr}px Space Grotesk, Segoe UI, sans-serif`;
    ctx.fillText(formatMb(maxValue), padding, 16 * dpr);
    ctx.fillText(formatSeconds(maxElapsed), width - padding - 52 * dpr, height - 8 * dpr);
    chart.range.textContent = `${formatMb(minValue)} to ${formatMb(maxValue)}`;
}

function drawEmptyChart(ctx, width, height) {
    ctx.strokeStyle = "#e3ddd4";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(24, height - 28);
    ctx.lineTo(width - 24, height - 28);
    ctx.stroke();
    ctx.fillStyle = "#6b645c";
    ctx.font = "13px Space Grotesk, Segoe UI, sans-serif";
    ctx.fillText("No samples", 24, 28);
}

refreshBtn?.addEventListener("click", () => refreshJobs({ keepSelection: true }));
window.addEventListener("resize", drawCharts);

if (activeJobsEl && recentJobsEl) {
    refreshJobs({ keepSelection: false });
    state.pollTimer = setInterval(() => refreshJobs({ keepSelection: true }), POLL_MS);
} else {
    console.warn("Profiler UI elements not found. Skipping profiler initialization.");
}
