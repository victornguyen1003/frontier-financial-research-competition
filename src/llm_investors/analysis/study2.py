"""Study 2 analysis: market outcomes by treatment (unit = market run).

    analyze-study2                 # real results
    analyze-study2 study2_dummy

Outputs in results/<name>/analysis/:
  outcomes_by_cell.csv     descriptive: mean, sd, min, max per cell (3-5 runs per cell is too
                           few for bootstrap CIs)
  paired_vs_baseline.csv   cell minus baseline for the same replicate (same dividend draws and
                           endowments: common random numbers), per replicate and averaged
  agent_de.csv             in-market DE by persona x memory, with Study 1's all-periods DE as a
                           reference only (NOT like for like; see AGENT_DE_NOTE)
  stylized_facts.json, summary.json, figures/*.png (one action heatmap per cell)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llm_investors.analysis import style
from llm_investors.measure import market_metrics, odean
from llm_investors.measure.dataset import read_log

from llm_investors.paths import ROOT  # noqa: E402
OUTCOMES = ["RAD", "RD", "peak_overpricing", "crash", "turnover", "volatility",
            "ret_autocorr_1"]


AGENT_DE_NOTE = (
    "In-market DE is not comparable like for like with Study 1: endowed shares are booked at "
    "FV1 (360) while FV falls to 24, so most positions are structural paper losses; gain/loss "
    "uses the clearing price, which the agent had not seen when it decided; and it counts all "
    "periods (one stock, so sale days have no paper counterpart). Study 1's all-periods DE is "
    "shown for reference only.")


def outcomes_by_cell(mk: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: with 3-5 market runs per cell a percentile CI is not meaningful."""
    rows = []
    for cell, g in mk.groupby("cell"):
        r = {"cell": cell, "runs": len(g)}
        for k in OUTCOMES:
            x = g[k].dropna()
            r |= {k: x.mean(), f"{k}_sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                  f"{k}_min": x.min(), f"{k}_max": x.max()}
        rows.append(r)
    return pd.DataFrame(rows)


def paired_vs_baseline(mk: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Cell minus baseline for the same replicate r. Replicate r draws the same dividends and
    endowments in every cell, so this removes that shared noise."""
    if baseline not in set(mk["cell"]):
        return pd.DataFrame()
    base = mk[mk["cell"] == baseline].set_index("replicate")[OUTCOMES]
    rows = []
    for cell, g in mk[mk["cell"] != baseline].groupby("cell"):
        g = g.set_index("replicate")[OUTCOMES]
        common = g.index.intersection(base.index)
        diff = g.loc[common] - base.loc[common]
        for r, row in diff.iterrows():
            rows.append({"cell": cell, "replicate": r, **row.to_dict()})
        rows.append({"cell": cell, "replicate": "mean", "n_pairs": len(common),
                     **diff.mean().to_dict()})
    return pd.DataFrame(rows)


def agent_de(runs: pd.DataFrame, study1: str) -> pd.DataFrame:
    rows = []
    for (mem, pers), g in runs.groupby(["memory", "agent_persona"]):
        p = odean.ratios(g[["RG", "RL", "PG", "PL"]].sum())
        rows.append({"memory": mem, "persona": pers, "agents": len(g),
                     "market_PGR": p["PGR"], "market_PLR": p["PLR"], "market_DE": p["DE"]})
    out = pd.DataFrame(rows)
    s1 = ROOT / "results" / study1 / "runs.csv"
    if s1.exists() and not out.empty:
        r1 = pd.read_csv(s1)
        r1 = r1[r1["complete"]]
        ref = []
        for (a, pers), g in r1.groupby(["architecture", "persona"]):
            c = g[[f"{k}_allperiods" for k in ("RG", "RL", "PG", "PL")]]
            c.columns = ["RG", "RL", "PG", "PL"]
            ref.append({"memory": a, "persona": pers,
                        "study1_DE_allperiods_reference": odean.ratios(c.sum())["DE"]})
        out = out.merge(pd.DataFrame(ref), on=["memory", "persona"], how="left")
    return out


def stylized_facts(mp: pd.DataFrame) -> dict:
    r = []
    acs = []
    for _, g in mp.sort_values("period").groupby("run_id"):
        x = np.diff(np.log(g["price"].to_numpy()))
        r += list(x)
        a = market_metrics._ac(np.abs(x))
        if a is not None:
            acs.append(a)
    return {"n_returns": len(r), "excess_kurtosis_pooled": market_metrics._kurt(r),
            "mean_abs_return_autocorr_1": float(np.mean(acs)) if acs else None,
            "note": "Positive excess kurtosis = fat tails; positive |r| autocorrelation = "
                    "volatility clustering. With 15 periods per market these are weak tests."}


def fig_price_fv(mp: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    cells = sorted(mp["cell"].unique())
    n = len(cells)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.9 * rows), sharey=True,
                             squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        g = mp[mp["cell"] == cell]
        for i, (_, r) in enumerate(g.groupby("run_id")):
            r = r.sort_values("period")
            ax.plot(r["period"], r["price"], color=style.SERIES[0], alpha=0.55, linewidth=1.5,
                    label="price (one line per market run)" if i == 0 else None)
        fv = g.drop_duplicates("period").sort_values("period")
        ax.plot(fv["period"], fv["fv"], color=style.INK, linestyle="--", linewidth=1.5,
                label="fundamental value")
        ax.set_title(cell.replace("_", " "), loc="left")
        ax.set_xlabel("period")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("price")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Market price vs fundamental value by treatment", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def fig_outcomes(mk: pd.DataFrame, path: Path):
    """Every market run as a dot, the cell mean as a bar: no CI from 3-5 runs."""
    import matplotlib.pyplot as plt
    style.apply()
    ks = ["RAD", "RD", "turnover", "volatility"]
    cells = sorted(mk["cell"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(13, 3.2), sharey=True)
    y = np.arange(len(cells))
    for ax, k in zip(axes, ks):
        for i, c in enumerate(cells):
            x = mk.loc[mk["cell"] == c, k].dropna()
            ax.scatter(x, np.full(len(x), i), s=36, color=style.SERIES[0],
                       edgecolor=style.SURFACE, linewidth=1.5, zorder=3,
                       label="market run" if (i == 0 and k == ks[0]) else None)
            if len(x):
                ax.plot([x.mean()] * 2, [i - 0.3, i + 0.3], color=style.INK, linewidth=2,
                        zorder=4, label="cell mean" if (i == 0 and k == ks[0]) else None)
        ax.set_title(k, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [c.replace("_", " ") for c in cells])
    fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Market outcomes by treatment (each dot is one market run; descriptive)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def action_matrix(log_path: Path) -> tuple[np.ndarray, list[str]]:
    """Agents x periods: +1 buy, -1 sell, 0 hold (orders after validation)."""
    ev = [e for e in read_log(log_path) if e["type"] == "period_end"]
    agents = sorted(ev[0]["orders"]) if ev else []
    code = {"buy": 1, "sell": -1, "hold": 0}
    m = np.array([[code[e["orders"][a]["action"]] for e in ev] for a in agents])
    return m, agents


def fig_actions(log_path: Path, path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    style.apply()
    m, agents = action_matrix(log_path)
    if not len(agents):
        return
    start = read_log(log_path)[0]
    pers = {a["agent"]: a["persona"] for a in start["agents"]}
    order = sorted(range(len(agents)), key=lambda i: (pers[agents[i]], agents[i]))
    fig, ax = plt.subplots(figsize=(7, 0.18 * len(agents) + 1.4))
    ax.imshow(m[order], aspect="auto", interpolation="nearest",
              cmap=ListedColormap([style.SELL, style.NEUTRAL, style.BUY]), vmin=-1, vmax=1)
    ax.set_yticks(range(len(agents)),
                  [f"{agents[i]} {pers[agents[i]].replace('_', ' ')}" for i in order],
                  fontsize=7)
    ax.set_xticks(range(m.shape[1]), range(1, m.shape[1] + 1))
    ax.set_xlabel("period")
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.BUY, label="buy"), Patch(color=style.SELL, label="sell"),
                       Patch(facecolor=style.NEUTRAL, edgecolor=style.GRID, label="hold")],
              loc="upper center", bbox_to_anchor=(0.5, -0.08 - 3 / (len(agents) + 8)), ncols=3)
    ax.set_title(f"Orders by agent and period — {start['run_id'].replace('_', ' ')}", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(name: str = "study2"):
    d = ROOT / "results" / name
    mk = pd.read_csv(d / "markets.csv")
    mk = mk[mk["complete"]]
    mp = pd.read_csv(d / "market_periods.csv")
    mp = mp[mp["run_id"].isin(mk["run_id"])]
    runs = pd.read_csv(d / "runs.csv")
    out = d / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cells = outcomes_by_cell(mk)
    cells.to_csv(out / "outcomes_by_cell.csv", index=False)
    paired = paired_vs_baseline(mk)
    paired.to_csv(out / "paired_vs_baseline.csv", index=False)
    ade = agent_de(runs[runs["complete"]], name.replace("study2", "study1"))
    ade.to_csv(out / "agent_de.csv", index=False)
    sf = stylized_facts(mp)
    (out / "stylized_facts.json").write_text(json.dumps(sf, indent=2))
    fig_price_fv(mp, out / "figures" / "price_vs_fv.png")
    fig_outcomes(mk, out / "figures" / "outcomes_by_cell.png")
    for old in (out / "figures").glob("actions_heatmap*.png"):
        old.unlink()
    for cell, g in mk.sort_values("replicate").groupby("cell"):   # first replicate per cell
        log = ROOT / "logs" / name / f"run_{g['run_id'].iloc[0]}.jsonl"
        if log.exists():
            fig_actions(log, out / "figures" / f"actions_heatmap_{cell}.png")
    (out / "summary.json").write_text(json.dumps(
        {"n_markets": len(mk), "cells": cells.to_dict("records"),
         "paired_vs_baseline": paired.to_dict("records"), "agent_de": ade.to_dict("records"),
         "agent_de_note": AGENT_DE_NOTE, "stylized_facts": sf}, indent=2, default=float))
    print(cells[["cell", "runs"] + OUTCOMES].round(3).to_string(index=False))
    if not paired.empty:
        print(paired[paired["replicate"] == "mean"].round(3).to_string(index=False))
    print(ade.round(3).to_string(index=False))
    print("NOTE:", AGENT_DE_NOTE)
    print(sf)
    print(f"-> {out}")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study2")


if __name__ == "__main__":
    cli()
