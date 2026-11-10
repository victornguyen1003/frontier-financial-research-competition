"""Odean (1998) disposition-effect metrics.

On every period in which the agent sells at least one stock, each position held
*before* that period's trades is classified by comparing the current price with
the average purchase price:

    sold (fully or partly) & price > cost  -> realized gain   (RG)
    sold                   & price < cost  -> realized loss   (RL)
    not sold               & price > cost  -> paper gain      (PG)
    not sold               & price < cost  -> paper loss      (PL)
    price == cost                          -> ignored

PGR = RG / (RG + PG),  PLR = RL / (RL + PL),  DE = PGR - PLR.
DE > 0 means winners are realized more readily than losers (disposition effect).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def classify(observations: list[dict]) -> Counter:
    """observations: one dict per (period, stock held before trading):
    {"period", "stock", "price", "cost", "sold"}"""
    c = Counter(RG=0, RL=0, PG=0, PL=0)
    sale_periods = {o["period"] for o in observations if o["sold"]}
    for o in observations:
        if o["period"] not in sale_periods or o["price"] == o["cost"]:
            continue
        gain = o["price"] > o["cost"]
        c[("R" if o["sold"] else "P") + ("G" if gain else "L")] += 1
    return c


def ratios(c: Counter) -> dict:
    pgr = c["RG"] / (c["RG"] + c["PG"]) if c["RG"] + c["PG"] else None
    plr = c["RL"] / (c["RL"] + c["PL"]) if c["RL"] + c["PL"] else None
    de = pgr - plr if pgr is not None and plr is not None else None
    return {**c, "PGR": pgr, "PLR": plr, "DE": de}


def run_metrics(observations: list[dict]) -> dict:
    return ratios(classify(observations))


def pooled(runs: list[dict]) -> dict:
    """Sum RG/RL/PG/PL across runs and recompute the ratios."""
    total = Counter()
    for r in runs:
        total.update({k: r[k] for k in ("RG", "RL", "PG", "PL")})
    return ratios(total)


def from_log(path: Path) -> dict:
    """Recompute metrics for one run from its .jsonl log."""
    obs = [json.loads(l) for l in Path(path).read_text().splitlines()]
    obs = [r for r in obs if r.get("type") == "positions"]
    return run_metrics([o for r in obs for o in r["observations"]])


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, from_log(p))
