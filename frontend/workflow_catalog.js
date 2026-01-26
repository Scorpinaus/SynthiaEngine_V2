// Minimal helper for fetching/caching the workflow catalog.
// Exposes a global `WorkflowCatalog` object (no bundler required).

(function () {
    const state = {
        catalog: null,
        promise: null,
    };

    async function load(apiBase) {
        const base = apiBase ?? window.API_BASE ?? "";
        if (state.catalog) {
            return state.catalog;
        }
        if (state.promise) {
            return state.promise;
        }
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

    function getTask(taskType) {
        return state.catalog?.tasks?.[taskType] ?? null;
    }

    function getDefaults(taskType) {
        return getTask(taskType)?.input_defaults ?? {};
    }

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

