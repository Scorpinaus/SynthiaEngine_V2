(function () {
    const taskSelect = document.getElementById("builder-task-type");
    const form = document.getElementById("builder-form");
    const submit = document.getElementById("builder-submit");
    const status = document.getElementById("builder-status");
    const result = document.getElementById("builder-result");
    const media = document.getElementById("builder-media");
    let catalog = null;
    let activeSource = null;

    function fieldType(schema, hint, defaultValue) {
        if (["image_list_ref"].includes(hint?.widget)) return "json";
        if (hint?.widget) return hint.widget;
        if (schema?.type === "boolean" || typeof defaultValue === "boolean") return "checkbox";
        if (schema?.type === "number" || schema?.type === "integer" || typeof defaultValue === "number") return "number";
        if (schema?.type === "array" || schema?.type === "object") return "json";
        return "text";
    }

    async function populateModelSelect(select, source) {
        select.innerHTML = '<option value="">Default model</option>';
        const params = new URLSearchParams(source?.params || {});
        try {
            const response = await fetch(`${window.API_BASE || ""}/models?${params}`);
            if (!response.ok) return;
            for (const model of await response.json()) {
                const option = document.createElement("option");
                option.value = model.name || model.model_name || model.path || "";
                option.textContent = model.name || model.model_name || option.value;
                select.appendChild(option);
            }
        } catch (error) {
            console.warn("Unable to load models for workflow builder", error);
        }
    }

    function createControl(name, schema, hint, defaultValue, required) {
        const wrapper = document.createElement(hint?.advanced ? "details" : "label");
        wrapper.className = "field workflow-builder-field";
        if (hint?.advanced) {
            const summary = document.createElement("summary");
            summary.textContent = hint.label || name;
            wrapper.appendChild(summary);
        } else {
            const label = document.createElement("span");
            label.textContent = `${hint?.label || name}${required ? " *" : ""}`;
            wrapper.appendChild(label);
        }

        const kind = fieldType(schema, hint, defaultValue);
        let control;
        if (kind === "textarea" || kind === "json") {
            control = document.createElement("textarea");
            control.className = kind === "json" ? "textarea-small workflow-builder-json" : "";
            if (kind === "json" && defaultValue !== undefined) {
                control.value = JSON.stringify(defaultValue, null, 2);
            } else if (defaultValue !== undefined && defaultValue !== null) {
                control.value = String(defaultValue);
            }
        } else if (kind === "select" || kind === "model_select") {
            control = document.createElement("select");
            for (const value of hint?.options || []) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                control.appendChild(option);
            }
            if (kind === "model_select") populateModelSelect(control, hint?.source);
            if (defaultValue !== undefined && defaultValue !== null) control.value = String(defaultValue);
        } else {
            control = document.createElement("input");
            if (["image_ref", "video_ref"].includes(kind)) {
                control.type = "file";
                control.accept = kind === "video_ref" ? "video/*" : "image/*";
                control.dataset.artifactInput = "true";
            } else if (kind === "checkbox") {
                control.type = "checkbox";
                control.checked = Boolean(defaultValue);
            } else {
                control.type = kind === "number" ? "number" : "text";
                if (defaultValue !== undefined && defaultValue !== null) control.value = String(defaultValue);
                for (const attr of ["min", "max", "step"]) {
                    if (hint?.[attr] !== undefined) control.setAttribute(attr, hint[attr]);
                }
            }
        }
        control.name = name;
        control.dataset.kind = kind;
        control.required = required;
        if (hint?.placeholder) control.placeholder = hint.placeholder;
        wrapper.appendChild(control);
        if (hint?.help) {
            const help = document.createElement("small");
            help.textContent = hint.help;
            wrapper.appendChild(help);
        }
        return wrapper;
    }

    function renderTask(taskType) {
        form.replaceChildren();
        const task = catalog?.tasks?.[taskType];
        submit.disabled = !task;
        if (!task) return;
        const properties = task.input_schema?.properties || {};
        const required = new Set(task.input_schema?.required || []);
        const hints = task.ui_hints?.inputs || {};
        const order = task.ui_hints?.input_order || Object.keys(properties);
        for (const name of order) {
            form.appendChild(createControl(name, properties[name] || {}, hints[name] || {}, task.input_defaults?.[name], required.has(name)));
        }
        status.textContent = `${order.length} inputs generated from ${taskType}.`;
    }

    async function readInputs() {
        const inputs = {};
        for (const control of form.querySelectorAll("[name]")) {
            const kind = control.dataset.kind;
            if (control.dataset.artifactInput === "true") {
                if (!control.files?.length) continue;
                const artifact = await WorkflowClient.uploadArtifact(window.API_BASE, control.files[0], control.files[0].name);
                inputs[control.name] = { artifact_id: artifact.artifact_id };
            } else if (kind === "checkbox") {
                inputs[control.name] = control.checked;
            } else if (kind === "number") {
                if (control.value !== "") inputs[control.name] = Number(control.value);
            } else if (kind === "json") {
                if (control.value.trim()) inputs[control.name] = JSON.parse(control.value);
            } else if (control.value !== "") {
                inputs[control.name] = control.value;
            }
        }
        return inputs;
    }

    function collectMedia(value, found = []) {
        if (typeof value === "string" && value.startsWith("/outputs/")) found.push(value);
        else if (Array.isArray(value)) value.forEach((item) => collectMedia(item, found));
        else if (value && typeof value === "object") Object.values(value).forEach((item) => collectMedia(item, found));
        return [...new Set(found)];
    }

    function showResult(job) {
        const output = job?.result?.outputs ?? job?.result ?? job;
        result.textContent = JSON.stringify(output, null, 2);
        media.replaceChildren();
        for (const path of collectMedia(output)) {
            const element = /\.(mp4|webm|mov)$/i.test(path) ? document.createElement("video") : document.createElement("img");
            element.src = `${window.API_BASE || ""}${path}`;
            element.alt = "Generated workflow output";
            if (element.tagName === "VIDEO") element.controls = true;
            media.appendChild(element);
        }
    }

    submit.addEventListener("click", async () => {
        if (!form.reportValidity()) return;
        submit.disabled = true;
        status.textContent = "Uploading inputs and submitting task…";
        result.textContent = "Waiting for renderer…";
        media.replaceChildren();
        try {
            const taskType = taskSelect.value;
            const job = await WorkflowClient.submitWorkflow(window.API_BASE, {
                tasks: [{ id: "render", type: taskType, inputs: await readInputs() }],
            });
            activeSource?.close();
            activeSource = WorkflowClient.watchJob(window.API_BASE, job.id, {
                onUpdate: (update) => { status.textContent = `Job ${update.status || "running"}…`; },
                onDone: (done) => {
                    status.textContent = `Job ${done.status}.`;
                    showResult(done);
                    submit.disabled = false;
                },
                onError: () => {
                    status.textContent = "Lost the job event stream. Check the queue for status.";
                    submit.disabled = false;
                },
            });
        } catch (error) {
            status.textContent = error.message || String(error);
            result.textContent = String(error.stack || error);
            submit.disabled = false;
        }
    });

    taskSelect.addEventListener("change", () => renderTask(taskSelect.value));

    (async function init() {
        catalog = await WorkflowCatalog.load(window.API_BASE);
        taskSelect.replaceChildren(new Option("Select a task…", ""));
        for (const taskType of Object.keys(catalog.tasks || {}).sort()) {
            taskSelect.appendChild(new Option(catalog.tasks[taskType]?.ui_hints?.title || taskType, taskType));
        }
        status.textContent = Object.keys(catalog.tasks || {}).length
            ? "Select any registered task."
            : "The workflow catalog is unavailable.";
    })();
})();
