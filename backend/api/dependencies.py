"""Shared FastAPI dependencies for application-owned resources."""

from fastapi import Request

from backend.settings import AppSettings


def get_app_settings(request: Request) -> AppSettings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, AppSettings):
        raise RuntimeError("Application settings are not initialized.")
    return settings
