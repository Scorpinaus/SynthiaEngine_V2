(() => {
    const fallbackMarkup = `
<section class="jobs-panel">
    <div class="jobs-header">
        <h3>Job Queue</h3>
        <span id="jobs-status" class="jobs-status">Idle</span>
    </div>
    <div class="jobs-tabs">
        <button class="jobs-tab is-active" type="button" data-tab="current">Current</button>
        <button class="jobs-tab" type="button" data-tab="recent">Recent</button>
    </div>
    <div class="jobs-tab-panel" data-tab-panel="current">
        <div class="jobs-section">
            <h4>Active</h4>
            <ul id="jobs-active" class="jobs-list"></ul>
        </div>
        <div class="jobs-section">
            <h4>Queued</h4>
            <ul id="jobs-queued" class="jobs-list"></ul>
        </div>
    </div>
    <div class="jobs-tab-panel" data-tab-panel="recent" hidden>
        <div class="jobs-section">
            <h4>Recent</h4>
            <ul id="jobs-recent" class="jobs-list"></ul>
        </div>
    </div>
</section>
`.trim();

    const JOB_QUEUE_POLL_MS = 5000;
    let jobQueueTimer = null;

    function setJobsStatus(message) {
        const statusEl = document.getElementById("jobs-status");
        if (statusEl) {
            statusEl.textContent = message;
        }
    }

    function renderJobList(containerId, jobs) {
        const container = document.getElementById(containerId);
        if (!container) {
            return;
        }
        container.innerHTML = "";
        if (!jobs.length) {
            const empty = document.createElement("li");
            empty.className = "job-card";
            empty.textContent = "No jobs";
            container.appendChild(empty);
            return;
        }
        jobs.forEach((job) => {
            const item = document.createElement("li");
            item.className = "job-card";

            const id = document.createElement("div");
            id.textContent = `${job.kind} (${job.id})`;
            const meta = document.createElement("span");
            const created = job.created_at ? new Date(job.created_at).toLocaleTimeString() : "unknown";
            const cancelRequested = Boolean(job.cancel_requested);
            const cancelSuffix = cancelRequested ? " (cancel pending)" : "";
            meta.textContent = `Status: ${job.status}${cancelSuffix} - Created: ${created}`;
            item.appendChild(id);
            item.appendChild(meta);

            const canCancel =
                job &&
                typeof job.id === "string" &&
                ["queued", "running"].includes(job.status) &&
                !cancelRequested;

            const showCancelPending =
                job &&
                typeof job.id === "string" &&
                ["queued", "running"].includes(job.status) &&
                cancelRequested;

            if (canCancel || showCancelPending) {
                const actions = document.createElement("div");
                actions.className = "job-card-actions";

                const cancelBtn = document.createElement("button");
                cancelBtn.type = "button";
                cancelBtn.className = "job-cancel-btn";
                cancelBtn.textContent = showCancelPending ? "Cancel pending" : "Cancel";
                cancelBtn.disabled = Boolean(showCancelPending);
                cancelBtn.addEventListener("click", async () => {
                    cancelBtn.disabled = true;
                    cancelBtn.textContent = "Canceling...";
                    try {
                        const res = await fetch(`${API_BASE}/api/jobs/${job.id}/cancel`, {
                            method: "POST",
                        });
                        if (!res.ok) {
                            const errorText = await res.text();
                            throw new Error(`Cancel failed (${res.status}): ${errorText}`);
                        }
                        await refreshJobQueue();
                    } catch (error) {
                        console.warn("Failed to cancel job:", error);
                        cancelBtn.disabled = false;
                        cancelBtn.textContent = "Cancel";
                    }
                });

                actions.appendChild(cancelBtn);
                item.appendChild(actions);
            }
            container.appendChild(item);
        });
    }

    async function refreshJobQueue() {
        try {
            setJobsStatus("Refreshing...");
            const res = await fetch(`${API_BASE}/api/jobs?limit=10`);
            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`Jobs request failed (${res.status}): ${errorText}`);
            }
            const jobs = await res.json();
            const active = jobs.filter((job) => job.status === "running");
            const queued = jobs.filter((job) => job.status === "queued");
            const recent = jobs.filter((job) => ["succeeded", "failed", "canceled"].includes(job.status));
            renderJobList("jobs-active", active);
            renderJobList("jobs-queued", queued);
            renderJobList("jobs-recent", recent);
            setJobsStatus(`Updated ${new Date().toLocaleTimeString()}`);
        } catch (error) {
            console.warn("Failed to refresh job queue:", error);
            setJobsStatus("Failed to refresh.");
        }
    }

    function startJobQueuePolling() {
        if (jobQueueTimer) {
            clearInterval(jobQueueTimer);
        }
        refreshJobQueue();
        jobQueueTimer = setInterval(refreshJobQueue, JOB_QUEUE_POLL_MS);
    }

    function activateTab(activeTab) {
        document.querySelectorAll(".jobs-tab").forEach((tab) => {
            const isActive = tab.dataset.tab === activeTab;
            tab.classList.toggle("is-active", isActive);
        });
        document.querySelectorAll(".jobs-tab-panel").forEach((panel) => {
            const isActive = panel.dataset.tabPanel === activeTab;
            panel.hidden = !isActive;
        });
    }

    function wireTabs() {
        const tabs = document.querySelectorAll(".jobs-tab");
        tabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                activateTab(tab.dataset.tab);
            });
        });
    }

    async function loadJobQueue() {
        const container = document.getElementById("jobs-queue-root");
        if (!container) {
            return;
        }
        let markup = fallbackMarkup;
        try {
            const res = await fetch("jobs_queue.html?v=2");
            if (!res.ok) {
                throw new Error(`Failed to load jobs queue UI: ${res.status}`);
            }
            markup = await res.text();
        } catch (error) {
            console.warn("Failed to load jobs queue UI:", error);
        }
        container.innerHTML = markup;
        wireTabs();
        activateTab("current");
        startJobQueuePolling();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", loadJobQueue);
    } else {
        loadJobQueue();
    }
})();
