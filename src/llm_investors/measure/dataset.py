"""Build analysis datasets from the .jsonl logs (free to re-run; no API calls).

    build-dataset study1_v3_gpt-6-luna_pilot

Writes results/<name>/:
  runs.*        one row per run: persona/profile, attribute columns, run-level PGR/PLR/DE and
                observability counts
  positions.*   one row per position held at the start of a period (all periods)
  decisions.*   positions on sale days only (Odean's sample; the §5.2 regression rows)
  reasoning.*   one row per run x period: the decision's `reasoning` text (design v3)
  lessons.csv   reflection lessons
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from llm_investors.agents.personas import ATTRIBUTE_KEYS
from llm_investors.paths import ROOT

from . import odean

META_KEYS = ("run_id", "architecture", "persona", "persona_kind", "n_active", "scenario",
             "model", "design")


def read_log(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a crashed run can leave a partial last line
    return out


def run_logs(study: str) -> list[Path]:
    return sorted((ROOT / "logs" / study).glob("run_*.jsonl"))


def _save(df: pd.DataFrame, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    try:
        df.to_parquet(out / f"{name}.parquet", index=False)
    except Exception:
        pass


def observability(events: list[dict]) -> dict:
    calls = [e for e in events if e["type"] == "llm_call"]
    return {
        "llm_calls": len(calls),
        "cache_hits": sum(e["cached"] for e in calls),
        "invalid_replies": sum(not e["valid"] for e in calls),
        "fallbacks": sum(e["type"] == "fallback" for e in events),
        "api_retries": sum(e.get("api_retries", 0) for e in calls),
        # incomplete replies were retried (never parsed or held); counts survive cache replay
        "incomplete": sum(e.get("incomplete_retries", 0) for e in calls),
        "incomplete_stops": sum(e["type"] == "incomplete_stop" for e in events),
        "fatal_api_errors": sum(e["type"] == "fatal_api_error" for e in events),
        # replies that left out a stock (invalid -> retried; never a silent hold)
        "omitted_stock_replies": sum(str(e.get("error") or "").startswith("missing an entry")
                                     for e in calls),
        "reasoning_errors": sum('"reasoning"' in str(e.get("error") or "") for e in calls),
        "orders_rejected": sum(e["type"] == "order_rejected" for e in events),
        "input_tokens": sum(e["usage"].get("input_tokens", 0) for e in calls if e["usage"]),
        "output_tokens": sum(e["usage"].get("output_tokens", 0) for e in calls if e["usage"]),
        "reasoning_tokens": sum(e["usage"].get("reasoning_tokens", 0) for e in calls if e["usage"]),
        "cost_usd": sum(e.get("cost_usd", 0) for e in calls),
    }


def run_meta(start: dict) -> dict:
    meta = {k: start.get(k) for k in META_KEYS if k in start}
    prof = start.get("profile")
    if prof:
        meta |= {f"attr_{k}": int(prof.get(k, 0)) for k in ATTRIBUTE_KEYS}
    return meta


def build(study: str) -> dict[str, pd.DataFrame]:
    pos_rows, run_rows, lesson_rows, reason_rows = [], [], [], []
    for path in run_logs(study):
        ev = read_log(path)
        start = next((e for e in ev if e["type"] == "run_start"), None)
        end = next((e for e in ev if e["type"] == "run_end"), None)
        if start is None:
            continue
        meta = run_meta(start)
        complete = end is not None and end.get("status") == "complete"
        obs = []
        for e in ev:
            if e["type"] != "period_end":
                continue
            obs += e["observations"]
            if "reasoning" in e:
                trades = [o for o in e["orders"] if o["action"] != "hold"]
                reason_rows.append({**meta, "period": e["period"], "reasoning": e["reasoning"],
                                    "fallback": e["reasoning"] is None,
                                    "n_sells": sum(o["action"] == "sell" for o in trades),
                                    "n_buys": sum(o["action"] == "buy" for o in trades),
                                    "actions": "".join(o["action"][0].upper()
                                                       for o in e["orders"])})
        for o in obs:
            o.update(meta)
        pos_rows += obs
        for e in ev:
            if e["type"] == "reflection":
                lesson_rows.append({**meta, "agent": e["agent"], "period": e["period"],
                                    **{f"lesson_{i + 1}": x for i, x in enumerate(e["lessons"])}})
        run_rows.append({**meta, "complete": complete,
                         "status": end.get("status") if end else "crashed",
                         "sale_days": len({o["period"] for o in obs if o["sold"]}),
                         **observability(ev), **odean.disposition(obs, True),
                         **{f"{k}_allperiods": v for k, v in
                            odean.disposition(obs, False).items()},
                         "final_value": end.get("final_value") if end else None})
    dfs = {"positions": pd.DataFrame(pos_rows), "runs": pd.DataFrame(run_rows),
           "lessons": pd.DataFrame(lesson_rows)}
    if reason_rows:
        dfs["reasoning"] = pd.DataFrame(reason_rows)
    if not dfs["positions"].empty:
        p = dfs["positions"]
        dfs["decisions"] = p[p["sale_day"] == 1].reset_index(drop=True)
    return dfs


def main(study: str):
    dfs = build(study)
    out = ROOT / "results" / study
    for name, df in dfs.items():
        _save(df, out, name)
        print(f"{name:<15} {len(df):>7} rows -> {out / name}.csv")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study1_v3_gpt-6-luna")


if __name__ == "__main__":
    cli()
