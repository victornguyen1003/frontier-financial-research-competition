"""`publish-logs`: copy run logs into the committed `published/logs/` folder, gzipped.

`logs/` is git-ignored (logs can be regenerated for free with `--replay`), so a deployed
viewer (e.g. Streamlit Community Cloud, which serves the git checkout) would find no logs at
all. Each `logs/<name>/run_*.jsonl` is written to `published/logs/<name>/run_*.jsonl.gz`; the
viewer reads those whenever `logs/<name>/` is missing. Compression is ~12x.

    publish-logs                 # every real dataset in logs/ (skips *_dummy and *_dryrun)
    publish-logs study2_v2_pilot # just these

Deterministic (gzip mtime 0; a file is rewritten only when its content changed), so publishing
unchanged logs creates no git diff. Runs that are gone from logs/<name>/ are removed from the
published copy, so it mirrors the source. Paid runs publish automatically when they finish.
"""
from __future__ import annotations

import gzip
import io
import shutil
import sys

from llm_investors.paths import ROOT

PUBLISHED = ROOT / "published" / "logs"


def publishable(name: str) -> bool:
    return not (name.endswith("_dummy") or name.endswith("_dryrun"))


def _gz(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as f:
        f.write(data)
    return buf.getvalue()


def publish(name: str) -> dict:
    src, dst = ROOT / "logs" / name, PUBLISHED / name
    if not src.is_dir():
        return {"name": name, "written": 0, "unchanged": 0, "removed": 0, "missing": True}
    dst.mkdir(parents=True, exist_ok=True)
    written = unchanged = removed = 0
    keep = set()
    for p in sorted(src.glob("run_*.jsonl")):
        out = dst / (p.name + ".gz")
        keep.add(out.name)
        data = _gz(p.read_bytes())
        if out.exists() and out.read_bytes() == data:
            unchanged += 1
            continue
        tmp = out.with_suffix(".gz.tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
        written += 1
    fatal = src / "FATAL_ERROR.json"
    if fatal.exists():
        shutil.copyfile(fatal, dst / fatal.name)
        keep.add(fatal.name)
    for old in dst.iterdir():
        if old.name not in keep:
            old.unlink()
            removed += 1
    return {"name": name, "written": written, "unchanged": unchanged, "removed": removed,
            "missing": False}


def summary_line(r: dict) -> str:
    if r["missing"]:
        return f"{r['name']}: no logs/{r['name']}/ (skipped)"
    return (f"{r['name']}: {r['written']} written, {r['unchanged']} unchanged, "
            f"{r['removed']} removed -> published/logs/{r['name']}/")


def main():
    names = sys.argv[1:] or sorted(p.name for p in (ROOT / "logs").glob("study*")
                                   if p.is_dir() and publishable(p.name))
    for n in names:
        print(summary_line(publish(n)))
    print("Commit them so the deployed viewer can show the logs: "
          "git add published && git commit -m 'publish logs' && git push")


if __name__ == "__main__":
    main()
