"""Host-local path selection endpoint and native dialog service."""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.api.dependencies import get_app_settings


router = APIRouter(tags=["local-paths"])


class LocalPathSelectRequest(BaseModel):
    selection_type: Literal["file", "folder"]


class LocalPathSelectResponse(BaseModel):
    path: str


def open_local_path_dialog(selection_type: Literal["file", "folder"]) -> str:
    """Open a native local picker on the backend host and return the path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Local path picker is unavailable on this host.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if selection_type == "file":
            selected_path = filedialog.askopenfilename(
                parent=root,
                title="Select local model file",
            )
        else:
            selected_path = filedialog.askdirectory(
                parent=root,
                title="Select local model folder",
            )
    finally:
        root.destroy()

    return os.path.normpath(selected_path) if selected_path else ""


@router.post("/api/local-path/select", response_model=LocalPathSelectResponse)
def select_local_path(req: LocalPathSelectRequest, request: Request):
    """Open a native local path picker on the API host."""
    settings = get_app_settings(request)
    client_host = request.client.host if request.client else ""
    if not settings.api.allow_remote_path_picker and client_host not in {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
    }:
        raise HTTPException(
            status_code=403,
            detail="Local path selection is restricted to loopback clients.",
        )
    try:
        return LocalPathSelectResponse(path=open_local_path_dialog(req.selection_type))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
