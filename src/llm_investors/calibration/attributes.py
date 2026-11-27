"""Attribute-level calibration (plan §2.7): the two age levels and their sources.

younger_age / older_age are the median ages of investors in the younger and the older group,
by linear interpolation inside the TDCC age brackets (ages treated as continuous, so the
bracket "20--30 Years" covers [20, 31)). The TDCC brackets are 0-19, 20-30, 31-40, 41-50,
51-60 and 61+, so the split the plan calls "under 40 / 40 and over" falls at the nearest
bracket boundary: 40 and under against 41 and over. Institutions are excluded.
"""
from __future__ import annotations

from . import sources

TOP_AGE = 90       # upper end of the open last bracket (only matters if a median falls there)
SPLIT = 41         # first age of the older group (bracket boundary nearest to 40)


def bracket_median(brackets: list[dict], col: str = "total") -> float:
    total = sum(b[col] for b in brackets)
    half, cum = total / 2, 0.0
    for b in brackets:
        lo, hi = b["lo"], (b["hi"] + 1 if b["hi"] is not None else TOP_AGE)
        if cum + b[col] >= half:
            return lo + (half - cum) / b[col] * (hi - lo)
        cum += b[col]
    raise ValueError("empty brackets")


def calibrate_attributes(cal_cfg: dict) -> dict:
    tdcc = sources.load_tdcc_age(cal_cfg["tdcc_age_path"])
    br = tdcc["brackets"]
    young = [b for b in br if b["hi"] is not None and b["hi"] < SPLIT]
    old = [b for b in br if b["lo"] >= SPLIT]
    if not young or not old or len(young) + len(old) != len(br):
        raise ValueError(f"TDCC brackets do not split at {SPLIT}: {[b['label'] for b in br]}")
    y, o = bracket_median(young), bracket_median(old)
    tot = sum(b["total"] for b in br)
    female = sum(b.get("female") or 0 for b in br)
    url = cal_cfg.get("tdcc_source_url") or None
    extra = sources.load_attribute_data(cal_cfg.get("attribute_data_path",
                                                    "data/attribute_data.csv"))
    return {
        "younger_age": int(round(y)), "older_age": int(round(o)),
        "younger_age_exact": round(y, 2), "older_age_exact": round(o, 2),
        "method": ("Median age within each group by linear interpolation inside TDCC age "
                   f"brackets; younger = brackets up to age {SPLIT - 1}, older = {SPLIT} and "
                   "over (TDCC brackets end at 40, so this is the boundary nearest to the "
                   "plan's 40); institutions excluded."),
        "groups": {"younger": {"brackets": [b["label"] for b in young],
                               "investors": sum(b["total"] for b in young)},
                   "older": {"brackets": [b["label"] for b in old],
                             "investors": sum(b["total"] for b in old)}},
        "tdcc_age_distribution": {"title": tdcc["title"], "brackets": br,
                                  "institutions": tdcc["institutions"],
                                  "share_female_individuals": female / tot if tot else None,
                                  "file": str(cal_cfg["tdcc_age_path"]),
                                  "source_url": url},
        "sources": [{"attribute": "age", "statistic": "investors by age bracket",
                     "value": tdcc["title"], "source_url": url,
                     "file": str(cal_cfg["tdcc_age_path"])}]
        + extra.to_dict("records"),
    }
