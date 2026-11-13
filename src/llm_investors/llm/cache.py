"""Disk cache for LLM replies, one JSON file per request hash (safe across threads/processes).

The cache is committed to git (cache/<study>/<2 hex>/<sha256>.json) so every collaborator can
replay paid runs for free. Files are content-addressed by the request, so new replies from
different people never collide. Each record stores `paid_usd`, what that reply cost (incomplete
attempts included), so the budget cap can be computed from the shared cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path


def request_key(**fields) -> str:
    blob = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class CacheCorrupt(RuntimeError):
    """A cache file exists but is not valid JSON (e.g. left with git merge-conflict markers).
    Raised instead of treating it as a miss, which would silently pay for the call again."""


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CacheCorrupt(f"{p} is not valid JSON ({e.msg}); if it has git conflict "
                               "markers, keep the version already on GitHub (see README, "
                               "'Sharing the cache')") from None

    def put(self, key: str, rec: dict):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        # fsync before the rename (and the directory after it): a paid reply must survive a
        # power loss or freeze, not come back as an empty file after the rename.
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
        try:
            fd = os.open(p.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass                                    # e.g. platforms without directory fsync

    def entries(self):
        return self.root.glob("??/*.json")

    def stats(self, price_fn=None) -> dict:
        """Entry counts and the total paid for the cached replies (budget accounting)."""
        n = stopped = corrupt = 0
        paid = 0.0
        size = 0
        for p in self.entries():
            n += 1
            size += p.stat().st_size
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                corrupt += 1
                continue
            stopped += rec.get("status") == "incomplete_limit"
            if "paid_usd" in rec:
                paid += float(rec["paid_usd"])
            elif price_fn and rec.get("usage"):        # records written before paid_usd existed
                paid += price_fn(rec["usage"]) + price_fn(rec.get("incomplete_usage") or {})
        return {"entries": n, "incomplete_limit_entries": stopped, "corrupt": corrupt,
                "bytes": size, "paid_usd": paid}
