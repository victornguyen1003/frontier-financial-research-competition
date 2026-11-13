"""Shared simulation plumbing: config, event logs, LLM setup, live terminal table, projections."""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import yaml

from llm_investors.llm.cache import DiskCache
from llm_investors.llm.client import LLMClient
from llm_investors.llm.cost import CostTracker, price_usage

from llm_investors.paths import ROOT  # noqa: E402


def load_config(study: str) -> dict:
    return yaml.safe_load((ROOT / "config" / f"{study}.yaml").read_text())


def base_args(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--backend", choices=["openai", "dummy"],
                    help="override llm.backend (dummy = rule-based agents, no API)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts and projected cost; no API calls (dummy replies)")
    ap.add_argument("--show-prompts", type=int, default=3, help="prompts to print in --dry-run")
    ap.add_argument("--yes", action="store_true", help="skip the full-run cost confirmation")
    ap.add_argument("--no-live", action="store_true", help="disable the live terminal table")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--replay", action="store_true",
                    help="cache-only re-run: never calls the API (no key needed, guaranteed $0); "
                         "stops on a cache miss and keeps the original logs")
    ap.add_argument("--retry-incomplete", action="store_true",
                    help="pay to retry requests that hit the incomplete limit in an earlier run "
                         "(by default that stop is replayed from cache for free)")
    return ap


def output_name(study: str, backend: str, dry_run: bool, stage: str = "full",
                variant: str = "") -> str:
    """Name of the logs/<name>/ and results/<name>/ directories.

    The pilot gets its own directory, so a full run never deletes pilot logs; both share the
    disk cache, so the full run replays the pilot's calls for free. Variants (e.g. the
    `_nopnl` ablation) and dummy / dry runs never mix with the main data.
    """
    name = study + (f"_{variant}" if variant else "") + ("_pilot" if stage == "pilot" else "")
    return f"{name}_dryrun" if dry_run else f"{name}_dummy" if backend == "dummy" else name


class EventLog:
    """Append-only .jsonl log; every event is flushed so a crashed run keeps its record."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, type_: str, **fields):
        rec = {"type": type_, "ts": round(time.time(), 3), **fields}
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()


def _json_default(o):
    try:
        return o.item()  # numpy scalars
    except AttributeError:
        return str(o)


def make_llm(cfg: dict, name: str, dry_run: bool, show_prompts: int, echo=print,
             offline: bool = False, retry_incomplete: bool = False):
    cache = DiskCache(ROOT / "cache" / cfg["study"])
    prices = cfg["llm"]["price_per_mtok"]
    real = cfg["llm"]["backend"] == "openai" and not dry_run
    shared = cache.stats(lambda u: price_usage(prices, u))["paid_usd"] if real else 0.0
    cost = CostTracker(cfg["budget"]["cap_usd"], prices,
                       ROOT / "cache" / cfg["study"] / "spend.json", shared_spent=shared)
    if real:
        echo(f"Budget {cfg['study']}: ${cost.spent_before:.4f} of ${cost.cap:.2f} already "
             f"spent (shared cache ${shared:.4f}, this machine's ledger "
             f"${cost.local_spent:.4f}).")
    return LLMClient(cfg["llm"], cache, cost, dry_run=dry_run, show_prompts=show_prompts,
                     echo=echo, offline=offline, retry_incomplete=retry_incomplete)


def confirm_full_run(cfg: dict, args, projection_file: Path) -> bool:
    if args.yes or args.dry_run or args.replay or cfg["llm"]["backend"] == "dummy":
        return True
    if projection_file.exists():
        p = json.loads(projection_file.read_text())
        print(f"Pilot projection: full run ≈ ${p['projected_full_cost_usd']:.2f} "
              f"(≈ ${p['projected_remaining_cost_usd']:.2f} not yet cached); cap "
              f"${cfg['budget']['cap_usd']:.2f}.")
    else:
        print("No pilot projection found (run --stage pilot first).")
    return input("Start the full run? [y/N] ").strip().lower() == "y"


class Board:
    """Live terminal table (rich): one row per active run."""

    def __init__(self, title: str, columns: list[str], cost: CostTracker, enabled: bool = True):
        self.title, self.columns, self.cost, self.enabled = title, columns, cost, enabled
        self.rows: dict[str, dict] = {}
        self.done = 0
        self.total = 0
        self.lock = threading.Lock()
        self._live = None

    def update(self, run_id: str, **fields):
        with self.lock:
            self.rows.setdefault(run_id, {}).update(fields)

    def finish(self, run_id: str):
        with self.lock:
            self.rows.pop(run_id, None)
            self.done += 1

    def _render(self):
        from rich.table import Table
        s = self.cost.summary()
        t = Table(title=f"{self.title} — runs {self.done}/{self.total} — spent "
                        f"${s['spent_total_usd']:.4f} of ${s['cap_usd']:.2f} — API calls "
                        f"{s['api_calls']}, cache hits {s['cache_hits']}", expand=True)
        t.add_column("run", no_wrap=True)
        for c in self.columns:
            t.add_column(c, overflow="fold")
        with self.lock:
            for rid, r in list(self.rows.items())[-20:]:
                t.add_row(rid, *[str(r.get(c, "")) for c in self.columns])
        return t

    def __enter__(self):
        if self.enabled:
            from rich.live import Live
            self._live = Live(get_renderable=self._render, refresh_per_second=2)
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live:
            self._live.__exit__(*exc)

    def echo(self, msg: str):
        if self._live:
            self._live.console.print(msg)
        else:
            print(msg)


def call_stats(log_dir: Path, price_fn) -> dict:
    """Measured tokens and (usage-priced) cost per call attempt from a stage's logs, cache hits
    included, plus the JSON retry rate and the service tiers that actually served the calls."""
    n = inp = out = reas = invalid = 0
    cost = 0.0
    tiers: dict[str, int] = {}
    for p in log_dir.glob("run_*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "llm_call" or not e.get("usage"):
                continue
            u = e["usage"]
            n += 1
            invalid += not e.get("valid", True)
            tier = str(u.get("service_tier"))
            tiers[tier] = tiers.get(tier, 0) + 1
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)
            reas += u.get("reasoning_tokens", 0)
            cost += price_fn(u)
            if e.get("incomplete_usage"):   # retried incomplete attempts are paid too
                cost += price_fn(e["incomplete_usage"])
    return {"calls": n, "mean_input_tokens": inp / n if n else None,
            "mean_output_tokens": out / n if n else None,
            "mean_reasoning_tokens": reas / n if n else None,
            "mean_cost_usd": cost / n if n else None,
            # extra attempts per first attempt caused by malformed JSON
            "json_retry_rate": invalid / max(1, n - invalid),
            "service_tiers": tiers}


def check_service_tier(st: dict, expected: str | None) -> None:
    """Warn if the API served calls on another tier than configured (prices assume it)."""
    other = {t: c for t, c in st["service_tiers"].items() if t != str(expected)}
    if expected and other:
        print(f"WARNING: {sum(other.values())} of {st['calls']} calls were served on tier(s) "
              f"{other}, not '{expected}'. The cost cap uses the configured '{expected}' "
              f"prices; check them before the full run.")


FATAL_STATUSES = ("quota_stop", "api_error_stop")


def fatal_status(e) -> str:
    return "quota_stop" if e.kind == "quota" else "api_error_stop"


def report_fatal(results: list[dict], log_dir: Path) -> int:
    """Print a report of fatal API errors, save it next to the logs, return the exit code."""
    fatal = [r for r in results if r["status"] in FATAL_STATUSES]
    if not fatal:
        return 0
    first = fatal[0]["fatal"]
    report = {"runs_stopped_by_error": [r["run_id"] for r in fatal],
              "runs_aborted": sum(r["status"] == "aborted" for r in results),
              "runs_complete": sum(r["status"] == "complete" for r in results),
              "error": first}
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "FATAL_ERROR.json").write_text(json.dumps(report, indent=2))
    what = ("QUOTA / BILLING / SPEND LIMIT" if first["kind"] == "quota"
            else "NON-RETRYABLE API ERROR")
    bar = "!" * 72
    print(f"\n{bar}\n{what} - the run was stopped.\n"
          f"  error:      {first['error_type']} (HTTP {first['status']}, code {first['code']})\n"
          f"  message:    {first['message']}\n"
          f"  request id: {first['request_id']}\n"
          f"  runs: {report['runs_complete']} complete, {len(fatal)} stopped by the error, "
          f"{report['runs_aborted']} aborted\n"
          f"  report:     {log_dir / 'FATAL_ERROR.json'}\n"
          + ("  Fix the quota or spend limit in the OpenAI dashboard, then re-run: completed "
             "calls replay from cache at $0.\n" if first["kind"] == "quota" else
             "  Fix the cause, then re-run: completed calls replay from cache at $0.\n")
          + bar)
    return 2


def replay_setup(args, backend: str, name: str) -> str:
    """Validate --replay and return the directory name the runs write to."""
    if not args.replay:
        return name
    if backend != "openai" or args.dry_run:
        raise SystemExit("--replay re-reads cached API replies: it needs the openai backend "
                         "(dummy runs have no cache) and cannot be combined with --dry-run")
    return f"{name}__replay"


def finish_replay(results: list[dict], name: str, log_name: str) -> int:
    """After a --replay: on a full cache hit, swap the replayed logs in for the originals and
    return 0; on any cache miss, keep the originals, report, and return 3."""
    import shutil
    tmp, dest = ROOT / "logs" / log_name, ROOT / "logs" / name
    misses = [r["run_id"] for r in results if r["status"] == "cache_miss"]
    if misses:
        bar = "!" * 72
        print(f"\n{bar}\nREPLAY INCOMPLETE - no API call was made ($0).\n"
              f"  {len(misses)} run(s) needed a reply that is not in the cache, e.g. "
              f"{misses[0]}.\n"
              "  The prompts differ from the cached run (config, prompt text, seed, model or "
              "max_output_tokens changed?).\n"
              f"  Original logs in {dest} are unchanged; the partial replay is in {tmp}.\n{bar}")
        return 3
    dest.mkdir(parents=True, exist_ok=True)
    for old in [*dest.glob("run_*.jsonl"), *dest.glob("FATAL_ERROR.json")]:
        old.unlink()
    for f in tmp.iterdir():
        f.replace(dest / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nReplay complete from cache: 0 API calls, $0. Logs refreshed in {dest}.")
    return 0
