"""`cache-stats`: what is in the shared reply cache and what still needs to be committed.

    cache-stats            # both studies
    cache-stats study1

The cache (cache/<study>/) is what makes `--replay` free for every collaborator, so commit and
push it after every paid run (README, "Sharing the cache").
"""
from __future__ import annotations

import subprocess
import sys

from llm_investors.llm.cache import DiskCache
from llm_investors.llm.cost import price_usage
from llm_investors.paths import ROOT
from llm_investors.sim.common import load_config


def _git(*args: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [line for line in out.splitlines() if line.strip()]


def unbacked(study: str) -> dict | None:
    """Cache files not yet committed, and commits touching the cache not yet pushed."""
    path = f"cache/{study}"
    status = _git("status", "--porcelain", "--untracked-files=all", "--", path)
    if status is None:
        return None
    files = [s for s in status if s.endswith(".json") and not s.endswith("spend.json")]
    ahead = _git("log", "--oneline", "@{upstream}..HEAD", "--", path)
    return {"uncommitted_files": len(files), "unpushed_commits": len(ahead or [])}


def report(study: str) -> dict:
    cfg = load_config(study)
    cache = DiskCache(ROOT / "cache" / study)
    st = cache.stats(lambda u: price_usage(cfg["llm"]["price_per_mtok"], u))
    st["backup"] = unbacked(study)
    return st


def reminder(study: str) -> str | None:
    """One-line nudge printed after a paid run when the cache has unbacked entries."""
    b = unbacked(study)
    if not b or not (b["uncommitted_files"] or b["unpushed_commits"]):
        return None
    return (f"Back up the reply cache so collaborators can replay for free: "
            f"{b['uncommitted_files']} uncommitted file(s) in cache/{study}"
            + (f", {b['unpushed_commits']} unpushed commit(s)" if b["unpushed_commits"] else "")
            + f". Run: git add cache/{study} && git commit -m 'cache: {study}' && git push")


def main():
    studies = sys.argv[1:] or ["study1", "study2"]
    for study in studies:
        st = report(study)
        print(f"{study}: {st['entries']:,} cached replies ({st['bytes'] / 1e6:.1f} MB), "
              f"{st['incomplete_limit_entries']} incomplete-limit stops, "
              f"paid ${st['paid_usd']:.4f}" + (f", {st['corrupt']} CORRUPT file(s)"
                                                 if st["corrupt"] else ""))
        b = st["backup"]
        if b is None:
            print("  (not a git checkout: cannot check the backup)")
        elif b["uncommitted_files"] or b["unpushed_commits"]:
            print(f"  NOT BACKED UP: {b['uncommitted_files']} uncommitted file(s), "
                  f"{b['unpushed_commits']} unpushed commit(s)")
        else:
            print("  backed up: everything is committed and pushed")


if __name__ == "__main__":
    main()
