from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.workflow import (
    TASK_REGISTRY,
    WorkflowRequest,
    WorkflowTask,
    build_workflow_catalog,
)

router = APIRouter(prefix="/api/workflow", tags=["workflow"])


class WorkflowTaskTypesResponse(BaseModel):
    task_types: list[str]


class WorkflowSchemaResponse(BaseModel):
    workflow_request_schema: dict[str, Any]
    workflow_task_schema: dict[str, Any]


class WorkflowCatalogTask(BaseModel):
    input_schema: dict[str, Any]
    input_defaults: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    ui_hints: dict[str, Any] | None = None


class WorkflowModelCapabilities(BaseModel):
    label: str
    aliases: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    features: dict[str, bool] = Field(default_factory=dict)


class WorkflowCatalogResponse(BaseModel):
    version: str
    tasks: dict[str, WorkflowCatalogTask]
    capabilities: dict[str, WorkflowModelCapabilities] = Field(default_factory=dict)


@router.get("/task-types", response_model=WorkflowTaskTypesResponse)
async def list_workflow_task_types():
    return WorkflowTaskTypesResponse(task_types=sorted(TASK_REGISTRY))


@router.get("/schema", response_model=WorkflowSchemaResponse)
async def get_workflow_schema():
    return WorkflowSchemaResponse(
        workflow_request_schema=WorkflowRequest.model_json_schema(by_alias=True),
        workflow_task_schema=WorkflowTask.model_json_schema(by_alias=True),
    )


@router.get("/catalog", response_model=WorkflowCatalogResponse)
async def get_workflow_catalog():
    return WorkflowCatalogResponse(**build_workflow_catalog())
