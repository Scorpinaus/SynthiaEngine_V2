from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_builder_is_catalog_driven_and_submits_registered_task():
    script = (ROOT / "frontend" / "workflow_builder.js").read_text(encoding="utf-8")
    assert "WorkflowCatalog.load" in script
    assert "catalog.tasks" in script
    assert "input_schema" in script
    assert "input_defaults" in script
    assert "ui_hints" in script
    assert "WorkflowClient.submitWorkflow" in script
    assert "WorkflowClient.uploadArtifact" in script
    assert "WorkflowClient.watchJob" in script


def test_workflow_builder_page_loads_shared_clients_and_is_in_navigation():
    page = (ROOT / "frontend" / "workflow_builder.html").read_text(encoding="utf-8")
    nav = (ROOT / "frontend" / "components" / "nav_bar.js").read_text(encoding="utf-8")
    assert 'src="workflow_catalog.js' in page
    assert 'src="workflow_client.js' in page
    assert 'id="builder-form"' in page
    assert 'href: "workflow_builder.html"' in nav
