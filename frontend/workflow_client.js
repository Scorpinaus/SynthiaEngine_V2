// Shared workflow client helpers (no bundler required).
// Exposes a global `WorkflowClient` object.

(function () {
    function makeIdempotencyKey() {
        if (typeof crypto?.randomUUID === "function") {
            return crypto.randomUUID();
        }
        return `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    }

    async function uploadArtifact(apiBase, blobOrFile, filename = "upload.png") {
        const base = apiBase ?? window.API_BASE ?? "";
        const formData = new FormData();
        formData.append("file", blobOrFile, filename);
        const res = await fetch(`${base}/api/artifacts`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Artifact upload failed (${res.status}): ${errorText}`);
        }
        return await res.json();
    }

    async function submitWorkflow(apiBase, payload, idempotencyKey) {
        const base = apiBase ?? window.API_BASE ?? "";
        const key = idempotencyKey || makeIdempotencyKey();
        const res = await fetch(`${base}/api/jobs`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Idempotency-Key": key,
            },
            body: JSON.stringify({ kind: "workflow", payload }),
        });
        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Job submit failed (${res.status}): ${errorText}`);
        }
        return await res.json();
    }

    function watchJob(apiBase, jobId, { isStale, onUpdate, onDone, onError } = {}) {
        const base = apiBase ?? window.API_BASE ?? "";
        const source = new EventSource(`${base}/api/jobs/${jobId}/events`);

        source.onmessage = (event) => {
            try {
                if (typeof isStale === "function" && isStale()) {
                    source.close();
                    return;
                }

                const job = JSON.parse(event.data);
                onUpdate?.(job);
                const status = job?.status ?? "unknown";
                if (status === "succeeded" || status === "failed" || status === "canceled") {
                    source.close();
                    onDone?.(job);
                }
            } catch (err) {
                // Parsing errors shouldn't break the connection; log and continue.
                console.warn("Failed to handle job SSE message:", err);
            }
        };

        source.onerror = (event) => {
            try {
                if (typeof isStale === "function" && isStale()) {
                    source.close();
                    return;
                }
                onError?.(event);
            } finally {
                // Avoid infinite reconnect loops in the browser.
                source.close();
            }
        };

        return source;
    }

    function readTextValue(elementId, fallback) {
        const el = document.getElementById(elementId);
        const value = (el?.value ?? "").toString();
        const trimmed = value.trim();
        if (!trimmed) {
            return fallback;
        }
        return trimmed;
    }

    function readNumberValue(elementId, fallback, { integer = false } = {}) {
        const el = document.getElementById(elementId);
        if (!el) {
            return fallback;
        }
        const raw = el.value;
        if (raw === "" || raw === null || raw === undefined) {
            return fallback;
        }
        const parsed = Number(raw);
        if (!Number.isFinite(parsed)) {
            return fallback;
        }
        let value = parsed;
        const minAttr = el.getAttribute("min");
        const maxAttr = el.getAttribute("max");
        const min = minAttr === null ? null : Number(minAttr);
        const max = maxAttr === null ? null : Number(maxAttr);
        if (Number.isFinite(min)) {
            value = Math.max(min, value);
        }
        if (Number.isFinite(max)) {
            value = Math.min(max, value);
        }
        if (integer) {
            value = Math.round(value);
        }
        return value;
    }

    function readSeedValue(elementId) {
        const el = document.getElementById(elementId);
        const raw = el?.value ?? "";
        if (raw === "") {
            return null;
        }
        const parsed = Number(raw);
        return Number.isFinite(parsed) ? parsed : null;
    }

    window.WorkflowClient = {
        makeIdempotencyKey,
        uploadArtifact,
        submitWorkflow,
        watchJob,
        readTextValue,
        readNumberValue,
        readSeedValue,
    };
})();

