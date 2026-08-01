"""Repository-wide pytest environment setup."""

from __future__ import annotations

import os
import tempfile


_HF_MODULES_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None

if "HF_MODULES_CACHE" not in os.environ:
    _HF_MODULES_TEMP_DIR = tempfile.TemporaryDirectory(prefix="syntha_hf_modules_")
    os.environ["HF_MODULES_CACHE"] = _HF_MODULES_TEMP_DIR.name
