"""Odean (1998) PGR / PLR / disposition effect.

Each observation is one position held before trading in a period:
    {"period", "price", "avg_cost", "sold": bool}   (+ any identifiers)
Gain/loss is price vs average purchase price; price == cost is ignored.

sale_days_only=True (Odean; Study 1): count only periods in which the same agent sold
something. Study 2 has a single stock, so a sale day has no paper counterpart; it uses
sale_days_only=False (every period with a position), and Study 1 reports both for comparison.
"""
from __future__ import annotations

from collections import Counter


def counts(obs: list[dict], sale_days_only: bool = True) -> Counter:
    c = Counter(RG=0, RL=0, PG=0, PL=0)
    sale_days = {o["period"] for o in obs if o["sold"]}
    for o in obs:
        if sale_days_only and o["period"] not in sale_days:
            continue
        if o["price"] == o["avg_cost"]:
            continue
        c[("R" if o["sold"] else "P") + ("G" if o["price"] > o["avg_cost"] else "L")] += 1
    return c


def ratios(c) -> dict:
    rg, rl, pg, pl = (int(c[k]) for k in ("RG", "RL", "PG", "PL"))
    pgr = rg / (rg + pg) if rg + pg else None
    plr = rl / (rl + pl) if rl + pl else None
    return {"RG": rg, "RL": rl, "PG": pg, "PL": pl, "PGR": pgr, "PLR": plr,
            "DE": pgr - plr if pgr is not None and plr is not None else None}


def disposition(obs: list[dict], sale_days_only: bool = True) -> dict:
    return ratios(counts(obs, sale_days_only))
