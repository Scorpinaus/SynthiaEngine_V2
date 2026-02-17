/**
 * Workflow catalog helpers (no bundler required).
 *
 * Exposes a global `window.WorkflowCatalog` object used by workflow UIs to:
 * - Fetch `/api/workflow/catalog` once (with caching + in-flight de-duping).
 * - Read task definitions and their `input_defaults`.
 * - Apply defaults to HTML form fields.
 *
 * Caching semantics:
 * - First caller triggers a network request; concurrent callers share one promise.
 * - Once resolved, the parsed JSON is cached for the lifetime of the page.
 * - On failure, a safe fallback catalog is returned (`{ version: "error", tasks: {} }`).
 *
 * Expected API response shape (subset used by this file):
 * - `{ version: string, tasks: { [taskType: string]: { input_defaults?: object } } }`
 *
 * Example:
 * - `await WorkflowCatalog.load(API_BASE);`
 * - `const defaults = WorkflowCatalog.getDefaults("sd15.text2img");`
 * - `WorkflowCatalog.applyDefaultsToForm("sd15.text2img", { steps: "steps" });`
 */

(function () {
    /**
     * @typedef {Object} WorkflowCatalogTask
     * @property {Record<string, any>=} input_defaults
     */

    /**
     * @typedef {Object} WorkflowCatalogResponse
     * @property {string} version
     * @property {Record<string, WorkflowCatalogTask>} tasks
     */

    const state = {
        /** @type {WorkflowCatalogResponse|null} */
        catalog: null,
        /** @type {Promise<WorkflowCatalogResponse>|null} */
        promise: null,
    };

    /**
     * Fetch the workflow catalog from the backend and cache it.
     *
     * If the request fails, this returns a fallback "error" catalog instead of throwing
     * (callers can still render UI with hardcoded defaults).
     *
     * @param {string=} apiBase Optional API base URL (defaults to `window.API_BASE` or "").
     * @returns {Promise<WorkflowCatalogResponse>}
     */
    async function load(apiBase) {
        const base = apiBase ?? window.API_BASE ?? "";
        if (state.catalog) {
            return state.catalog;
        }
        if (state.promise) {
            return state.promise;
        }
        // De-dupe concurrent loads by storing the in-flight promise.
        state.promise = (async () => {
            try {
                const res = await fetch(`${base}/api/workflow/catalog`);
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error(`Catalog fetch failed (${res.status}): ${text}`);
                }
                state.catalog = await res.json();
                return state.catalog;
            } finally {
                state.promise = null;
            }
        })().catch((error) => {
            console.warn("Failed to load workflow catalog:", error);
            state.catalog = { version: "error", tasks: {} };
            return state.catalog;
        });
        return state.promise;
    }

    /**
     * Read a task definition from the loaded catalog.
     *
     * @param {string} taskType Task type identifier (e.g. "sd15.text2img").
     * @returns {WorkflowCatalogTask|null} Task definition, or null if missing/not loaded.
     */
    function getTask(taskType) {
        return state.catalog?.tasks?.[taskType] ?? null;
    }

    /**
     * Read the `input_defaults` object for a task type.
     *
     * @param {string} taskType Task type identifier (e.g. "sd15.text2img").
     * @returns {Record<string, any>} Defaults object (empty object if missing/not loaded).
     */
    function getDefaults(taskType) {
        return getTask(taskType)?.input_defaults ?? {};
    }

    /**
     * Apply backend defaults to HTML inputs by element id and workflow field name.
     *
     * Only sets a value when the element appears "unset" (its value is empty or equals
     * its `defaultValue`) to avoid clobbering user edits.
     *
     * Notes:
     * - Values are coerced to strings (`String(def)`).
     * - Missing elements and missing defaults are silently ignored.
     *
     * @param {string} taskType Task type identifier (e.g. "sd15.text2img").
     * @param {Record<string, string>} idToFieldMap Map of `{ elementId: fieldName }`.
     * @returns {void}
     */
    function applyDefaultsToForm(taskType, idToFieldMap) {
        const defaults = getDefaults(taskType);
        for (const [elementId, fieldName] of Object.entries(idToFieldMap)) {
            const el = document.getElementById(elementId);
            if (!el) {
                continue;
            }
            const def = defaults[fieldName];
            if (def === undefined || def === null) {
                continue;
            }
            if (el.value === "" || el.value === el.defaultValue) {
                el.value = String(def);
            }
        }
    }

    window.WorkflowCatalog = {
        load,
        getTask,
        getDefaults,
        applyDefaultsToForm,
    };
})();
