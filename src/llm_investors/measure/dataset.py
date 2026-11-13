"""Build analysis datasets from the .jsonl logs (free to re-run; no API calls).

    build-dataset study1
    build-dataset study2
Writes results/<study>/positions.parquet (+ .csv), decisions.* (sale days only),
runs.* (run-level PGR/PLR/DE and observability counts), lessons.csv,
and for Study 2 markets.* and market_periods.*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import market_metrics, odean

from llm_investors.paths import ROOT  # noqa: E402


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
        "orders_rejected": sum(e["type"] == "order_rejected" for e in events),
        "input_tokens": sum(e["usage"].get("input_tokens", 0) for e in calls if e["usage"]),
        "output_tokens": sum(e["usage"].get("output_tokens", 0) for e in calls if e["usage"]),
        "reasoning_tokens": sum(e["usage"].get("reasoning_tokens", 0) for e in calls if e["usage"]),
        "cost_usd": sum(e.get("cost_usd", 0) for e in calls),
    }


def build(study: str) -> dict[str, pd.DataFrame]:
    pos_rows, run_rows, lesson_rows, market_rows, mper_rows = [], [], [], [], []
    for path in run_logs(study):
        ev = read_log(path)
        start = next((e for e in ev if e["type"] == "run_start"), None)
        end = next((e for e in ev if e["type"] == "run_end"), None)
        if start is None:
            continue
        meta = {k: start[k] for k in start if k in
                ("run_id", "architecture", "persona", "scenario", "cell", "replicate",
                 "agent_mix", "memory", "price_limit", "transaction_tax", "n_agents")}
        complete = end is not None and end.get("status") == "complete"
        obs = [o for e in ev if e["type"] == "period_end" for o in e["observations"]]
        for o in obs:
            o.update(meta)
        pos_rows += obs
        for e in ev:
            if e["type"] == "reflection":
                lesson_rows.append({**meta, "agent": e["agent"], "period": e["period"],
                                    "persona_of_agent": e["persona"],
                                    **{f"lesson_{i + 1}": x for i, x in enumerate(e["lessons"])}})
        obsv = observability(ev)
        if study.startswith("study1"):
            run_rows.append({**meta, "complete": complete,
                             "status": end.get("status") if end else "crashed", **obsv,
                             **odean.disposition(obs, True),
                             **{f"{k}_allperiods": v for k, v in
                                odean.disposition(obs, False).items()},
                             "final_value": end.get("final_value") if end else None})
        else:
            auctions = [e for e in ev if e["type"] == "auction_cleared"]
            T = len(auctions)
            if T:
                price = [a["price"] for a in auctions]
                fv = [a["fundamental_value"] for a in auctions]
                traded = [a["traded"] for a in auctions]
                for a in auctions:
                    mper_rows.append({**meta, "period": a["period"], "price": a["price"],
                                      "fv": a["fundamental_value"], "volume": a["volume"],
                                      "traded": a["traded"], "imbalance": a["imbalance"],
                                      "dividend": a.get("dividend")})
                market_rows.append({**meta, "complete": complete,
                                    "status": end.get("status") if end else "crashed",
                                    "periods": T, **obsv,
                                    **market_metrics.bubble_measures(price, fv, traded),
                                    **market_metrics.trading_measures(
                                        price, [a["volume"] for a in auctions],
                                        start["total_shares"], traded),
                                    **odean.disposition(obs, False)})
            else:  # stopped before the first auction: keep the row for observability counts
                market_rows.append({**meta, "complete": complete,
                                    "status": end.get("status") if end else "crashed",
                                    "periods": 0, **obsv})
            # agent-level DE inside the market, by persona
            for (agent, persona), g in pd.DataFrame(obs).groupby(["agent", "agent_persona"]) \
                    if obs else []:
                run_rows.append({**meta, "agent": agent, "agent_persona": persona,
                                 "complete": complete,
                                 **odean.disposition(g.to_dict("records"), False)})
    dfs = {"positions": pd.DataFrame(pos_rows), "runs": pd.DataFrame(run_rows),
           "lessons": pd.DataFrame(lesson_rows)}
    if not dfs["positions"].empty:
        p = dfs["positions"]
        dfs["decisions"] = p[p["sale_day"] == 1].reset_index(drop=True)
    if study.startswith("study2"):
        dfs["markets"] = pd.DataFrame(market_rows)
        dfs["market_periods"] = pd.DataFrame(mper_rows)
    return dfs


def main(study: str):
    dfs = build(study)
    out = ROOT / "results" / study
    for name, df in dfs.items():
        _save(df, out, name)
        print(f"{name:<15} {len(df):>7} rows -> {out / name}.csv")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study1")


if __name__ == "__main__":
    cli()
