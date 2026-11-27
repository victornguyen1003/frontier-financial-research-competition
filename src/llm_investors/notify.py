"""Phone notifications through ntfy (https://ntfy.sh) when a command finishes.

Set NTFY_TOPIC in the git-ignored .env (or the environment). An ntfy topic is readable by
anyone who knows its name, so it is not committed, and messages carry only run summaries
(statuses, cost, projections), never keys or prompts. Without NTFY_TOPIC nothing is sent.
A failed notification never affects the run.
"""
from __future__ import annotations

import os
import urllib.request

from llm_investors.env import load_env


def notify(title: str, message: str, tags: str = "", priority: str = "default") -> bool:
    load_env()
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(
        f"{server}/{topic}", data=message.encode("utf-8"), method="POST",
        headers={"Title": title.encode("ascii", "replace").decode(), "Tags": tags,
                 "Priority": priority})
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:  # noqa: BLE001
        return False
