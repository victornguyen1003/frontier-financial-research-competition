"""Study 1 analysis (free to re-run; reads results/<name>/ built by measure.dataset).

    analyze-study1                 # real results
    analyze-study1 study1_dummy    # rule-based pipeline check

Outputs in results/<name>/analysis/:
  regressions.txt          LPM (main), LPM without `ret`, LPM without the final period, logit
                           (robustness), reflect-vs-none on periods 6+, post-reflection
                           interaction; SEs clustered by matched set (persona x scenario)
  de_pooled.csv            pooled PGR / PLR / DE with 95% cluster-bootstrap CIs, for all
                           periods and excluding the final period
  de_bootstrap.csv         run-level DE by architecture (and by architecture x persona) with
                           cluster-bootstrap 95% CIs over matched persona x scenario sets
  de_differences.csv       log - none, reflect - none with CIs
  friedman.json            Friedman test across architectures on matched sets
  summary.json             everything the Streamlit Findings page shows
  figures/*.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from llm_investors.analysis import style
from llm_investors.measure import odean

from llm_investors.paths import ROOT  # noqa: E402
ARCHS = ["none", "log", "reflect"]
B = 2000


def load(name: str):
    d = ROOT / "results" / name
    runs = pd.read_csv(d / "runs.csv")
    dec = pd.read_parquet(d / "decisions.parquet") if (d / "decisions.parquet").exists() \
        else pd.read_csv(d / "decisions.csv")
    runs = runs[runs["complete"]]
    dec = dec[dec["run_id"].isin(runs["run_id"])]
    return runs, dec


# --------------------------------------------------------------------------- regressions
def regressions(dec: pd.DataFrame) -> tuple[str, dict]:
    """SEs are clustered by matched set (persona x scenario): the three arms of a set share
    prices and, by design, identical early prompts (one shared draw), so their errors are
    correlated. Matched sets nest runs, so this is more conservative than the proposal's
    clustering by run."""
    d = dec.dropna(subset=["gain"]).copy()
    d["gain"] = d["gain"].astype(int)
    d["fe"] = d["persona"] + "_" + d["scenario"].astype(str)
    d["post"] = (d["period"] > 5).astype(int)
    T = int(d["period"].max())
    arch = "C(architecture, Treatment(reference='none'))"
    controls = "ret + holding_period + drawdown"
    out, coefs, failed = [], {}, []

    def clusters(data):
        return data["fe"].astype("category").cat.codes

    def fit(label, formula, data, logit=False):
        g = clusters(data)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = (smf.logit if logit else smf.ols)(formula, data)
                res = model.fit(cov_type="cluster", cov_kwds={"groups": g},
                                **({"disp": 0, "maxiter": 200} if logit else {}))
            if logit and not res.mle_retvals.get("converged", True):
                raise RuntimeError("did not converge (possible separation)")
        except Exception as e:  # noqa: BLE001
            out.append(f"== {label}: FAILED ({type(e).__name__}: {e})\n")
            failed.append({"model": label, "error": f"{type(e).__name__}: {e}"})
            return
        keep = [k for k in res.params.index if "gain" in k or k in ("ret", "holding_period",
                                                                    "drawdown")]
        tbl = pd.DataFrame({"coef": res.params[keep], "se": res.bse[keep],
                            "p": res.pvalues[keep]})
        out.append(f"== {label}\n{formula}\nN = {int(res.nobs)}, clusters (matched sets) = "
                   f"{g.nunique()}\n{tbl.round(4).to_string()}\n")
        coefs[label] = {k: {"coef": float(res.params[k]), "se": float(res.bse[k]),
                            "p": float(res.pvalues[k])} for k in keep}

    fit("LPM (main)", f"sell ~ gain * {arch} + {controls} + C(fe)", d)
    # With `ret` controlled, gain x arch is a jump at zero return; without it, it is the full
    # winner-loser gap (closer to PGR - PLR).
    fit("LPM without ret", f"sell ~ gain * {arch} + holding_period + drawdown + C(fe)", d)
    # Agents know the horizon; liquidating in the last period inflates PGR and PLR alike.
    fit(f"LPM excluding the final period ({T})", f"sell ~ gain * {arch} + {controls} + C(fe)",
        d[d["period"] < T])
    fit("Logit (robustness)", f"sell ~ gain * {arch} + {controls} + C(fe)", d, logit=True)
    nr = d[d["architecture"].isin(["none", "reflect"])]
    fit("LPM reflect vs none, periods 6+",
        f"sell ~ gain * C(architecture, Treatment(reference='none')) + {controls} + C(fe)",
        nr[nr["period"] >= 6])
    nr = nr.assign(reflect=(nr["architecture"] == "reflect").astype(int))
    fit("LPM reflect vs none, post-reflection interaction",
        f"sell ~ gain * reflect * post + {controls} + C(fe)", nr)
    note = ("Gain x architecture coefficients measure how memory changes the gap between the "
            "probability of selling a winner and a loser (the DE) relative to 'none'. "
            "SEs clustered by matched set (persona x scenario).\n")
    if failed:
        note += "FAILED MODELS: " + "; ".join(f["model"] for f in failed) + "\n"
    coefs["_failed"] = failed
    return note + "\n".join(out), coefs


# --------------------------------------------------------------------------- bootstrap
def matched_sets(runs: pd.DataFrame, col: str = "DE") -> pd.DataFrame:
    """Wide table: one row per persona x scenario, one column per architecture; rows where
    the run-level DE is undefined for any architecture are dropped."""
    w = runs.pivot_table(index=["persona", "scenario"], columns="architecture", values=col)
    return w[[a for a in ARCHS if a in w]].dropna()


def boot_ci(w: pd.DataFrame, fn, rng) -> tuple[float, float, float]:
    est = fn(w)
    idx = rng.integers(0, len(w), (B, len(w)))
    vals = np.array([fn(w.iloc[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5], axis=0)
    return est, lo, hi


def bootstrap(runs: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, diffs = [], []
    groups = [("all", runs)] + [(p, g) for p, g in runs.groupby("persona")]
    for pers, g in groups:
        w = matched_sets(g)
        if len(w) < 3:
            continue
        est, lo, hi = boot_ci(w, lambda x: x.mean().to_numpy(), rng)
        for i, a in enumerate(w.columns):
            pooled = odean.ratios(g[g["architecture"] == a][["RG", "RL", "PG", "PL"]].sum())
            rows.append({"persona": pers, "architecture": a, "n_sets": len(w),
                         "n_sets_dropped": int(g.groupby(["persona", "scenario"]).ngroups
                                               - len(w)),
                         "n_runs_pooled": int((g["architecture"] == a).sum()),
                         "mean_run_DE": est[i], "ci_low": lo[i], "ci_high": hi[i],
                         "pooled_PGR": pooled["PGR"], "pooled_PLR": pooled["PLR"],
                         "pooled_DE": pooled["DE"]})
        for a in ("log", "reflect"):
            if a in w and "none" in w:
                e, l, h = boot_ci(w, lambda x: float((x[a] - x["none"]).mean()), rng)
                diffs.append({"persona": pers, "contrast": f"{a} - none", "n_sets": len(w),
                              "diff": e, "ci_low": l, "ci_high": h})
    w = matched_sets(runs)
    fr = None
    if len(w) >= 3 and w.shape[1] == 3:
        s, p = stats.friedmanchisquare(*[w[a] for a in w.columns])
        fr = {"statistic": float(s), "p_value": float(p), "n_sets": len(w)}
    return pd.DataFrame(rows), pd.DataFrame(diffs), fr


def pooled_ci(dec: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Pooled PGR / PLR / DE per architecture (overall and by persona) with 95% CIs from a
    cluster bootstrap over matched sets (persona x scenario), for all periods and excluding
    the final period. Counts come from the decision rows (Odean's sale-day sample)."""
    rng = np.random.default_rng(seed)
    d = dec.dropna(subset=["gain"]).copy()
    T = int(d["period"].max())
    d["cls"] = np.where(d["sell"] == 1, np.where(d["gain"] == 1, "RG", "RL"),
                        np.where(d["gain"] == 1, "PG", "PL"))
    rows = []
    for sample, data in (("all periods", d), (f"excluding period {T}", d[d["period"] < T])):
        for pers, g in [("all", data)] + list(data.groupby("persona")):
            cnt = (g.groupby(["persona", "scenario", "architecture", "cls"]).size()
                   .unstack(["architecture", "cls"], fill_value=0))
            sets = len(cnt)
            idx = rng.integers(0, sets, (B, sets))
            for a in ARCHS:
                cols = [(a, k) for k in ("RG", "RL", "PG", "PL")]
                m = np.stack([cnt[c].to_numpy() if c in cnt else np.zeros(sets)
                              for c in cols], axis=1)                    # sets x 4
                tot = m.sum(axis=0)
                bs = m[idx].sum(axis=1)                                  # B x 4
                with np.errstate(invalid="ignore", divide="ignore"):
                    pgr = bs[:, 0] / (bs[:, 0] + bs[:, 2])
                    plr = bs[:, 1] / (bs[:, 1] + bs[:, 3])
                est = odean.ratios(dict(zip(("RG", "RL", "PG", "PL"), tot)))
                row = {"sample": sample, "persona": pers, "architecture": a, "n_sets": sets,
                       **{k: int(v) for k, v in zip(("RG", "RL", "PG", "PL"), tot)}}
                for k, v in (("PGR", pgr), ("PLR", plr), ("DE", pgr - plr)):
                    lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() \
                        else (np.nan, np.nan)
                    row |= {k: est[k], f"{k}_lo": lo, f"{k}_hi": hi}
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def fig_de(boot: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    pers = [p for p in boot["persona"].unique() if p != "all"]
    cols = ["all"] + pers
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.26
    for i, a in enumerate(ARCHS):
        b = boot[boot["architecture"] == a].set_index("persona").reindex(cols)
        x = np.arange(len(cols)) + (i - 1) * width
        ax.bar(x, b["mean_run_DE"], width - 0.03, color=style.ARCH_COLORS[a], label=a,
               zorder=2)
        ax.errorbar(x, b["mean_run_DE"], yerr=[b["mean_run_DE"] - b["ci_low"],
                                              b["ci_high"] - b["mean_run_DE"]],
                    fmt="none", ecolor=style.INK_2, elinewidth=1, capsize=3, zorder=3)
    ax.axhline(0, color=style.INK_2, linewidth=1)
    ax.set_xticks(np.arange(len(cols)), [c.replace("_", " ") for c in cols])
    ax.set_ylabel("disposition effect (PGR − PLR)")
    ax.set_title("Run-level disposition effect by memory architecture (95% cluster-bootstrap CI)",
                 loc="left")
    ax.legend(title="memory architecture", ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_paths(name: str, persona: str, scenario: int, path: Path):
    """Price paths of one scenario with each architecture's sales marked."""
    import matplotlib.pyplot as plt
    from llm_investors.measure.dataset import read_log
    style.apply()
    logs = {a: ROOT / "logs" / name / f"run_{a}_{persona}_s{scenario:03d}.jsonl" for a in ARCHS}
    logs = {a: p for a, p in logs.items() if p.exists()}
    if not logs:
        return
    first = read_log(next(iter(logs.values())))[0]
    prices, labels = np.array(first["prices"]), first["labels"]
    fig, axes = plt.subplots(1, len(labels), figsize=(12, 2.8), sharey=True)
    for j, (ax, s) in enumerate(zip(axes, labels)):
        ax.plot(range(len(prices)), prices[:, j], color=style.INK_2, linewidth=1.5)
        for k, (a, p) in enumerate(logs.items()):
            sells = [e for e in read_log(p) if e["type"] == "trade" and e["stock"] == s
                     and e["action"] == "sell"]
            ax.scatter([e["period"] for e in sells],
                       [e["price"] * (1 + 0.025 * (k - 1)) for e in sells], s=36,
                       color=style.ARCH_COLORS[a], edgecolor=style.SURFACE, linewidth=1.5,
                       zorder=3, label=f"{a} sells" if j == 0 else None)
        ax.axhline(100, color=style.GRID, linewidth=1, zorder=1)
        ax.set_title(f"Stock {s}", loc="left")
        ax.set_xlabel("period")
    axes[0].set_ylabel("price")
    fig.legend(loc="upper right", ncols=3)
    fig.suptitle(f"Scenario {scenario}, {persona.replace('_', ' ')}: sales by memory "
                 "architecture (markers offset vertically)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def main(name: str = "study1"):
    runs, dec = load(name)
    out = ROOT / "results" / name / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    text, coefs = regressions(dec)
    (out / "regressions.txt").write_text(text)
    boot, diffs, fr = bootstrap(runs)
    pooled = pooled_ci(dec)
    pooled.to_csv(out / "de_pooled.csv", index=False)
    boot.to_csv(out / "de_bootstrap.csv", index=False)
    diffs.to_csv(out / "de_differences.csv", index=False)
    (out / "friedman.json").write_text(json.dumps(fr, indent=2))
    if not boot.empty:
        fig_de(boot, out / "figures" / "de_by_architecture.png")
    for pers in sorted(runs["persona"].unique()):
        fig_paths(name, pers, int(runs["scenario"].min()), out / "figures" / f"paths_{pers}.png")
    summary = {"n_runs": len(runs), "n_decision_rows": len(dec), "regressions": coefs,
               "friedman": fr, "bootstrap": boot.to_dict("records"),
               "pooled": pooled.to_dict("records"),
               "differences": diffs.to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(text)
    print(boot.round(3).to_string(index=False))
    print(pooled[pooled["persona"] == "all"][["sample", "architecture", "PGR", "PLR", "DE",
                                               "DE_lo", "DE_hi"]].round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    print("Friedman:", fr)
    print(f"-> {out}")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study1")


if __name__ == "__main__":
    cli()
