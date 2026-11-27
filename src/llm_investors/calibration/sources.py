"""Loaders for the raw calibration inputs. Nothing here is written back to the repository.

TEJ daily stock data: TEJ exports come with Chinese or English headers; both are mapped to
the same internal names. `calibration.tej_path` may be a file or a folder (every *.csv in it).

TDCC investor statistics: the "Investors with Trading Accounts by Age" table as downloaded
(one header row of age brackets, rows for male / female / total counts).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from llm_investors.paths import ROOT

# internal name -> accepted headers (Chinese TEJ export, English TEJ export)
TEJ_COLUMNS = {
    "stock": ("公司簡稱", "公司", "CO_ID"),
    "date": ("年月日", "Date"),
    "close": ("收盤價(元)", "Close(NTD)"),
    "ret_ln_pct": ("報酬率-Ln", "報酬率％-Ln", "ROI%-Ln"),
    "market": ("市場別", "Market"),
    "industry": ("TSE 產業別", "TSE產業別", "TSE Industry Name"),
    "limit": ("漲跌停", "LIMIT"),
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return pd.read_csv(p, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{p}: cannot decode as UTF-8 or Big5")


def load_tej(path: str | Path) -> pd.DataFrame:
    """Daily rows with columns stock, date (Timestamp), close, ret_ln (decimal log return),
    market, industry, limit."""
    p = resolve(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files or not files[0].exists():
        raise FileNotFoundError(f"TEJ data not found at {p} (calibration.tej_path)")
    frames = []
    for f in files:
        raw = _read_csv(f)
        raw.columns = [c.strip() for c in raw.columns]
        cols = {}
        for name, options in TEJ_COLUMNS.items():
            hit = next((c for c in options if c in raw.columns), None)
            if hit is None:
                raise KeyError(f"{f.name}: no column for {name!r} (tried {options})")
            cols[name] = raw[hit]
        frames.append(pd.DataFrame(cols))
    d = pd.concat(frames, ignore_index=True)
    d["stock"] = d["stock"].str.strip()
    d["date"] = pd.to_datetime(d["date"].str.strip(), format="%Y%m%d")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["ret_ln"] = pd.to_numeric(d["ret_ln_pct"], errors="coerce") / 100.0
    for c in ("market", "industry", "limit"):
        d[c] = d[c].fillna("").str.strip()
    return (d.drop(columns="ret_ln_pct").drop_duplicates(["stock", "date"])
            .sort_values(["stock", "date"]).reset_index(drop=True))


def _num(s: str) -> float | None:
    s = (s or "").replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def load_tdcc_age(path: str | Path) -> dict:
    """{title, brackets: [{label, lo, hi, male, female, total}], institutions}.
    Bracket ages are inclusive: "20--30 Years" -> lo 20, hi 30; "61Years & Over" -> hi None."""
    p = resolve(path)
    rows = list(csv.reader(p.read_text(encoding="utf-8-sig").splitlines()))
    title = rows[0][0].strip() if rows and rows[0] else ""
    header = next(r for r in rows if r and r[0].strip().lower() == "age")
    by_sex = {}
    for r in rows:
        if len(r) > 1 and r[1].strip() in ("Male", "Female", "Total"):
            by_sex[r[1].strip().lower()] = r
    brackets, inst = [], None
    for j, lab in enumerate(header):
        lab = lab.strip()
        if j < 2 or not lab or lab.lower() == "total":
            continue
        if "institution" in lab.lower():
            inst = _num(by_sex["total"][j])
            continue
        nums = [int(x) for x in re.findall(r"\d+", lab)]
        lo, hi = (nums[0], None) if "over" in lab.lower() else (nums[0], nums[1])
        brackets.append({"label": lab, "lo": lo, "hi": hi,
                         **{s: _num(by_sex[s][j]) for s in ("male", "female", "total")
                            if s in by_sex}})
    return {"title": title, "brackets": brackets, "institutions": inst}


def load_attribute_data(path: str | Path) -> pd.DataFrame:
    p = resolve(path)
    if not p.exists():
        return pd.DataFrame(columns=["attribute", "statistic", "value", "year", "source_url"])
    return pd.read_csv(p, dtype=str).fillna("")
