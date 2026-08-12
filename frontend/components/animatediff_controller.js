/** AnimateDiff page controller with video result handling. */
(function () {
    function create({ taskType, fields, fallbackModel, catalogBindings }) {
        const form = GenerationPage.createFormController({
            family: "sd15",
            taskType,
            fallbackModel,
            fields,
        });
        let jobs = null;

        async function generate() {
            const defaults = await form.defaults();
            const inputs = form.collectSettings(defaults);
            const adapters = Array.isArray(inputs.lora_adapters) ? inputs.lora_adapters : [];
            delete inputs.lora_adapters;
            inputs.lora = {
                lora_enabled: adapters.length > 0,
                lora_adapters: adapters.length > 0 ? adapters : [],
            };
            const tasks = [{ id: "t1", type: taskType, inputs }];
            try {
                await GenerationPage.validateTasks(tasks);
            } catch (error) {
                if (error instanceof Error && error.message.startsWith("Input validation failed for ")) {
                    alert(error.message);
                }
                console.warn("Failed to validate SD1.5 AnimateDiff inputs:", error);
                return;
            }
            await jobs.run(
                { tasks, return: "@t1.videos" },
                "Failed to generate SD1.5 AnimateDiff videos:"
            );
        }

        function init() {
            jobs = GenerationPage.createVideoJobs();
            document.getElementById("generate-button")?.addEventListener("click", generate);
            void form.loadModels();
            if (window.WorkflowCatalog?.load) {
                void window.WorkflowCatalog.load(API_BASE).then(() => {
                    window.WorkflowCatalog.applyDefaultsToForm(taskType, catalogBindings);
                }).catch(() => {});
            }
            form.initLora();
            form.initPresets();
        }

        return { generate, init };
    }

    window.AnimateDiffController = { create };
})();
