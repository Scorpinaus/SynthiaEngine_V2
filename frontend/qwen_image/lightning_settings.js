(() => {
    const PROFILE_EVENT = "qwen-lightning-profile-changed";

    function validProfile(profile) {
        if (profile?.kind !== "qwen_image_lightning") {
            return null;
        }
        const steps = Number(profile.steps);
        const trueCfgScale = Number(profile.true_cfg_scale);
        if ((steps !== 4 && steps !== 8) || trueCfgScale !== 1) {
            return null;
        }
        return profile;
    }

    function create({ taskType, stepsElementId = "steps", trueCfgElementId = "true_cfg" }) {
        const isQwenTask = [
            "qwen-image.text2img",
            "qwen-image.img2img",
            "qwen-image.inpaint",
        ].includes(taskType);
        let savedValues = null;
        let pendingValues = null;
        let applyingSettings = false;
        let activeProfile = null;

        function getInputs() {
            return {
                steps: document.getElementById(stepsElementId),
                trueCfg: document.getElementById(trueCfgElementId),
            };
        }

        function snapshotValues() {
            const { steps, trueCfg } = getInputs();
            return {
                steps: steps?.value ?? "",
                trueCfg: trueCfg?.value ?? "",
            };
        }

        function setLocked(profile) {
            if (!isQwenTask) {
                return;
            }
            const { steps, trueCfg } = getInputs();
            if (steps) {
                steps.value = String(profile.steps);
                steps.disabled = true;
            }
            if (trueCfg) {
                trueCfg.value = String(profile.true_cfg_scale);
                trueCfg.disabled = true;
            }
        }

        function unlock({ restore }) {
            if (!isQwenTask) {
                return;
            }
            const { steps, trueCfg } = getInputs();
            if (steps) {
                steps.disabled = false;
                if (restore && savedValues) {
                    steps.value = savedValues.steps;
                }
            }
            if (trueCfg) {
                trueCfg.disabled = false;
                if (restore && savedValues) {
                    trueCfg.value = savedValues.trueCfg;
                }
            }
            savedValues = null;
        }

        function applyProfile(profile) {
            const normalized = validProfile(profile);
            if (!isQwenTask) {
                return;
            }
            if (!normalized) {
                unlock({ restore: !applyingSettings });
                return;
            }
            if (!savedValues) {
                savedValues = pendingValues ?? snapshotValues();
            }
            setLocked(normalized);
        }

        function onProfileChanged(event) {
            activeProfile = validProfile(event?.detail?.profile ?? null);
            if (applyingSettings) {
                return;
            }
            applyProfile(activeProfile);
        }

        function beforeApplySettings() {
            if (!isQwenTask) {
                return;
            }
            applyingSettings = true;
            pendingValues = snapshotValues();
        }

        function afterApplySettings() {
            if (!isQwenTask) {
                return;
            }
            applyingSettings = false;
            if (activeProfile) {
                applyProfile(activeProfile);
            } else {
                unlock({ restore: false });
            }
            pendingValues = null;
        }

        window.addEventListener(PROFILE_EVENT, onProfileChanged);
        return { beforeApplySettings, afterApplySettings };
    }

    window.QwenImageLightningSettings = { create };
})();
