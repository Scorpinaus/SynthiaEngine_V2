(() => {
    const DEFAULT_IP_ADAPTER_TOGGLE_LABEL = "Use image prompt reference";
    const markup = `
<div class="adapter-launch">
    <button id="adapter-modal-open" class="secondary adapter-launch-button" type="button">
        <span>Adapters</span>
        <span id="adapter_summary_label" class="adapter-launch-summary">Loading adapter counts...</span>
    </button>
</div>

<div id="adapter-modal" class="modal hidden" aria-hidden="true">
    <div class="modal-overlay" id="adapter-modal-overlay"></div>
    <div class="modal-content adapter-modal-content" role="dialog" aria-modal="true" aria-labelledby="adapter-modal-title">
        <div class="modal-header">
            <div>
                <h2 id="adapter-modal-title">Adapters</h2>
                <p class="modal-subtitle" id="adapter-modal-subtitle">SD1.5 adapter stack</p>
            </div>
            <button id="adapter-modal-close" class="secondary" type="button">Close</button>
        </div>

        <div class="adapter-tabs" role="tablist" aria-label="Adapter panels">
            <button id="adapter-tab-overview" class="adapter-tab is-active" type="button" role="tab" aria-selected="true" aria-controls="adapter-panel-overview" data-adapter-tab="overview">
                <span>Overview</span>
            </button>
            <button id="adapter-tab-controlnet" class="adapter-tab" type="button" role="tab" aria-selected="false" aria-controls="adapter-panel-controlnet" data-adapter-tab="controlnet">
                <span>ControlNet Preprocessors</span>
                <span id="adapter-tab-controlnet-badge" class="adapter-tab-badge">0</span>
            </button>
            <button id="adapter-tab-lora" class="adapter-tab" type="button" role="tab" aria-selected="false" aria-controls="adapter-panel-lora" data-adapter-tab="lora">
                <span>LoRA Adapters</span>
                <span id="adapter-tab-lora-badge" class="adapter-tab-badge">0</span>
            </button>
            <button id="adapter-tab-ipadapter" class="adapter-tab" type="button" role="tab" aria-selected="false" aria-controls="adapter-panel-ipadapter" data-adapter-tab="ipadapter">
                <span>IP-Adapters</span>
                <span id="adapter-tab-ipadapter-badge" class="adapter-tab-badge">1</span>
            </button>
        </div>

        <div class="adapter-tab-panels">
            <section id="adapter-panel-overview" class="adapter-tab-panel is-active" role="tabpanel" aria-labelledby="adapter-tab-overview" data-adapter-panel="overview">
                <div class="adapter-overview-grid">
                    <button class="adapter-overview-card" type="button" data-adapter-tab-jump="controlnet">
                        <span class="adapter-overview-title">ControlNet Preprocessors</span>
                        <strong id="adapter-overview-controlnet-count">0 available</strong>
                        <span id="adapter-overview-controlnet-detail">No control images active.</span>
                    </button>
                    <button class="adapter-overview-card" type="button" data-adapter-tab-jump="lora">
                        <span class="adapter-overview-title">LoRA Adapters</span>
                        <strong id="adapter-overview-lora-count">0 available</strong>
                        <span id="adapter-overview-lora-detail">No LoRAs selected.</span>
                    </button>
                    <button class="adapter-overview-card" type="button" data-adapter-tab-jump="ipadapter">
                        <span class="adapter-overview-title">IP-Adapters</span>
                        <strong id="adapter-overview-ipadapter-count">1 available</strong>
                        <span id="adapter-overview-ipadapter-detail">Image prompt disabled.</span>
                    </button>
                </div>
            </section>

            <section id="adapter-panel-controlnet" class="adapter-tab-panel" role="tabpanel" aria-labelledby="adapter-tab-controlnet" data-adapter-panel="controlnet" hidden>
                <div id="controlnet-panel-root"></div>
            </section>

            <section id="adapter-panel-lora" class="adapter-tab-panel" role="tabpanel" aria-labelledby="adapter-tab-lora" data-adapter-panel="lora" hidden>
                <div id="lora-panel-root"></div>
            </section>

            <section id="adapter-panel-ipadapter" class="adapter-tab-panel" role="tabpanel" aria-labelledby="adapter-tab-ipadapter" data-adapter-panel="ipadapter" hidden>
                <div id="ip_adapter_panel" class="ip-adapter-panel">
                    <button id="ip_adapter_toggle" class="ip-adapter-toggle" type="button" aria-expanded="false" aria-controls="ip_adapter_content">
                        <span>IP-Adapter</span>
                        <span id="ip_adapter_chevron" aria-hidden="true">&#9662;</span>
                    </button>
                    <div id="ip_adapter_content" class="ip-adapter-content">
                        <label class="field">
                            <span>Image Prompt Reference</span>
                            <span class="hires-toggle">
                                <span>{{IP_ADAPTER_TOGGLE_LABEL}}</span>
                                <input id="ip_adapter_enabled" type="checkbox" />
                            </span>
                        </label>

                        <label class="field">
                            <span>IP-Adapter Image</span>
                            <input id="ip_adapter_image" type="file" accept="image/*" />
                        </label>

                        <div id="ip_adapter_preview_panel" class="ip-adapter-preview-panel">
                            <div id="ip_adapter_preview_empty" class="field-hint">No reference image selected.</div>
                            <img id="ip_adapter_preview" class="ip-adapter-preview is-hidden" alt="IP-Adapter reference preview" />
                        </div>

                        <label class="field" data-ip-adapter-mask-section>
                            <span>IP-Adapter Mask</span>
                            <input id="ip_adapter_mask_image" type="file" accept="image/*" />
                        </label>

                        <div id="ip_adapter_mask_preview_panel" class="ip-adapter-preview-panel" data-ip-adapter-mask-section>
                            <div id="ip_adapter_mask_preview_empty" class="field-hint">No IP-Adapter mask selected.</div>
                            <img id="ip_adapter_mask_preview" class="ip-adapter-preview is-hidden" alt="IP-Adapter mask preview" />
                        </div>

                        <div class="field-row" data-ip-adapter-mask-section>
                            <button id="ip_adapter_mask_editor_open" class="secondary" type="button">Create mask</button>
                            <button id="ip_adapter_mask_clear" class="secondary" type="button">Clear mask</button>
                        </div>
                        <div class="field-hint" data-ip-adapter-mask-section>White applies the image prompt. Black suppresses it.</div>

                        <label class="field">
                            <span>IP-Adapter Scale</span>
                            <input id="ip_adapter_scale" type="number" value="0.6" step="0.05" min="0" max="1" />
                        </label>
                    </div>
                </div>
            </section>
        </div>
    </div>
</div>
`.trim();

    function getConfiguredMarkup(container) {
        const ipAdapterToggleLabel =
            container?.dataset?.ipAdapterToggleLabel?.trim() || DEFAULT_IP_ADAPTER_TOGGLE_LABEL;
        return markup.replace("{{IP_ADAPTER_TOGGLE_LABEL}}", ipAdapterToggleLabel);
    }

    function renderAdapterPanel() {
        const container = document.getElementById("adapter-panel-root");
        if (!container || container.dataset.adapterPanelLoaded === "true") {
            return;
        }

        container.innerHTML = getConfiguredMarkup(container);
        if (container.dataset.ipAdapterMaskEnabled === "false") {
            container
                .querySelectorAll("[data-ip-adapter-mask-section]")
                .forEach((element) => element.remove());
        }
        container.dataset.adapterPanelLoaded = "true";
        initAdapterModalOpenState();
        document.dispatchEvent(new CustomEvent("adapter-panel:loaded"));
    }

    function initAdapterModalOpenState() {
        const modal = document.getElementById("adapter-modal");
        if (!modal || modal.dataset.openStateObserverLoaded === "true") {
            return;
        }

        const syncOpenState = () => {
            document.body.classList.toggle(
                "adapter-modal-open",
                !modal.classList.contains("hidden")
            );
        };

        syncOpenState();
        new MutationObserver(syncOpenState).observe(modal, {
            attributes: true,
            attributeFilter: ["class"],
        });
        modal.dataset.openStateObserverLoaded = "true";
    }

    renderAdapterPanel();
    if (document.readyState === "loading" && !document.getElementById("adapter-panel-root")) {
        document.addEventListener("DOMContentLoaded", renderAdapterPanel, { once: true });
    }

    window.AdapterPanel = {
        render: renderAdapterPanel,
    };
})();
