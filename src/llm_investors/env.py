"""Load secrets from a local, git-ignored `.env` file at the project root.

    cp .env.example .env        # then put your key in .env
    chmod 600 .env

Format: `KEY=value` per line; blank lines, `# comments`, an optional `export ` prefix and
quotes around the value are allowed. A variable already set in the environment wins over the
file, so `export OPENAI_API_KEY=...` still works. Values are never printed or logged.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from llm_investors.paths import ROOT

ENV_FILE = ROOT / ".env"


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:                     # trailing comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


def load_env(path: Path | None = None) -> list[str]:
    """Set variables from `path` (default: <root>/.env) that are not already set; return their
    names, never their values."""
    path = Path(path) if path is not None else ENV_FILE
    if not path.exists():
        return []
    if os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"WARNING: {path} is readable by other users; run: chmod 600 {path}",
              file=sys.stderr)
    loaded = []
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        if value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def require_api_key() -> None:
    """Load .env and stop with a clear message if there is still no OpenAI key."""
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Put it in the git-ignored file "
            f"{ENV_FILE} (copy .env.example), or `export OPENAI_API_KEY=...`. "
            "Free alternatives that need no key: --dry-run, --replay, --backend dummy.")
