/**
 * Shared workflow client helpers (no bundler required).
 *
 * Exposes a global `window.WorkflowClient` object used by workflow UIs to:
 * - Upload artifacts to the backend (`POST /api/artifacts`).
 * - Submit workflow jobs (`POST /api/jobs` with `{ kind: "workflow" }`).
 * - Stream job updates via Server-Sent Events (`GET /api/jobs/:id/events`).
 * - Read/normalize values from HTML form controls.
 *
 * This module is intentionally small and framework-agnostic so it can be used from
 * plain `<script>` tags.
 *
 * Example:
 * - `const artifact = await WorkflowClient.uploadArtifact(API_BASE, file, file.name);`
 * - `const job = await WorkflowClient.submitWorkflow(API_BASE, payload);`
 * - `WorkflowClient.watchJob(API_BASE, job.id, { onUpdate, onDone, onError });`
 */

(function () {
    /**
     * @typedef {Object} ArtifactResponse
     * @property {string=} id
     * @property {string=} path
     * @property {string=} filename
     * @property {string=} content_type
     * @property {number=} size_bytes
     */

    /**
     * @typedef {Object} JobCreateResponse
     * @property {string=} id
     * @property {string=} status
     */

    /**
     * @typedef {Object} JobUpdate
     * A minimal subset of the job object streamed via SSE and returned by the API.
     * Fields are treated as optional because backend schemas can evolve.
     * @property {string=} id
     * @property {string=} status
     * @property {Object=} result
     */

    /**
     * @typedef {Object} WatchJobCallbacks
     * @property {() => boolean=} isStale If true, the connection is closed and updates ignored.
     * @property {(job: JobUpdate) => void=} onUpdate Called for each parsed SSE message.
     * @property {(job: JobUpdate) => void=} onDone Called when status is terminal.
     * @property {(event: any) => void=} onError Called on EventSource error.
     */

    /**
     * Create an idempotency key suitable for the `Idempotency-Key` header.
     *
     * Uses `crypto.randomUUID()` when available; otherwise falls back to a timestamp+random string.
     *
     * @returns {string}
     */
    function makeIdempotencyKey() {
        if (typeof crypto?.randomUUID === "function") {
            return crypto.randomUUID();
        }
        return `idemp_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    }

    /**
     * Upload a Blob/File as an artifact.
     *
     * @param {string=} apiBase Optional API base URL (defaults to `window.API_BASE` or "").
     * @param {Blob|File} blobOrFile File contents to upload.
     * @param {string=} filename Filename to report to the server (used when `blobOrFile` is a Blob).
     * @returns {Promise<ArtifactResponse>}
     * @throws {Error} When the HTTP response is not OK.
     */
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

    /**
     * Submit a workflow job.
     *
     * Adds an `Idempotency-Key` header (generated if not provided) to support safe retries.
     *
     * @param {string=} apiBase Optional API base URL (defaults to `window.API_BASE` or "").
     * @param {any} payload Workflow payload (task list + return ref).
     * @param {string=} idempotencyKey Optional idempotency key override.
     * @returns {Promise<JobCreateResponse>}
     * @throws {Error} When the HTTP response is not OK.
     */
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

    /**
     * Watch a job via Server-Sent Events and invoke callbacks with parsed job updates.
     *
     * Behavior:
     * - `onUpdate(job)` is invoked for every parsed SSE message.
     * - When job status is terminal ("succeeded" | "failed" | "canceled"), the connection closes
     *   and `onDone(job)` is invoked.
     * - On transport errors, `onError(event)` is invoked and the connection is closed to avoid
     *   infinite reconnect loops.
     *
     * @param {string=} apiBase Optional API base URL (defaults to `window.API_BASE` or "").
     * @param {string} jobId Job id returned by `submitWorkflow`.
     * @param {WatchJobCallbacks=} callbacks Callback collection.
     * @returns {EventSource} The active EventSource (caller may close it).
     */
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

    /**
     * Read an element's string value; trims and applies a fallback when blank.
     *
     * @param {string} elementId DOM element id.
     * @param {string} fallback Returned when the element is missing or the trimmed value is blank.
     * @returns {string}
     */
    function readTextValue(elementId, fallback) {
        const el = document.getElementById(elementId);
        const value = (el?.value ?? "").toString();
        const trimmed = value.trim();
        if (!trimmed) {
            return fallback;
        }
        return trimmed;
    }

    /**
     * Read an element's numeric value, honoring optional HTML `min`/`max` attributes.
     *
     * Returns `fallback` when the element is missing, empty, or not a finite number.
     * If `integer` is true, rounds to the nearest integer after applying min/max clamps.
     *
     * @param {string} elementId DOM element id.
     * @param {number} fallback Returned when missing/invalid/empty.
     * @param {{ integer?: boolean }=} options
     * @returns {number}
     */
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

    /**
     * Read a seed value as a number or null.
     *
     * Convention:
     * - Empty string means "random seed" and returns null.
     * - Non-finite values return null.
     *
     * @param {string} elementId DOM element id.
     * @returns {number|null}
     */
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
