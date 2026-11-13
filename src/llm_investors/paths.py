"""Project root: config/, data/, logs/, cache/ and results/ live here, outside the package.

Defaults to the repository root (src/llm_investors/paths.py -> two levels up). Set
LLM_INVESTORS_ROOT to keep data elsewhere, e.g. when the package is installed non-editable.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("LLM_INVESTORS_ROOT") or Path(__file__).resolve().parents[2])
