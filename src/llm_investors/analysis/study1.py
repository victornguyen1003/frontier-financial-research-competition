"""Study 1 v3 analysis (plan §5; free to re-run, reads results/<name>/ from build-dataset).

    analyze-study1 study1_v3_gpt-6-luna_pilot

Writes results/<name>/analysis/:
  cells.csv          RG, RL, PG, PL, PGR, PLR, DE, Ratio per persona x arm (+ pooled rows),
                     95% cluster-bootstrap CIs resampling paths
  regression.txt/csv §5.2 decision-level LPM, two-way clustered by path and profile
  tests.json         §5.3 tests, each with its MDE
  blind_spots.csv    d_k with Holm p-values, CIs, MDE and the blind-spot flag
  attribute_de.csv   raw DE difference (attribute at 1 minus 0) with path-bootstrap CIs
  figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from llm_investors.agents.personas import ATTRIBUTE_KEYS, LABELS
from llm_investors.analysis import style
from llm_investors.analysis.ols import holm, mde, ols
from llm_investors.measure import odean
from llm_investors.paths import ROOT

ARCHS = ["none", "log", "reflect"]
MEMS = ["log", "reflect"]
TAGS = ["persona", "reference_point", "past_trades", "forecast", "randomness"]
TAGCOLS = [f"tag_{t}" for t in TAGS]
B = 2000
BAND = 0.03
ALPHA = 0.05
CNT = ["RG", "RL", "PG", "PL"]
CONTROLS = ["ret", "holding_period", "drawdown"]


def benchmark() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "benchmarks" / "barber_2007_table3_panelB.csv")


def _read(d: Path, name: str) -> pd.DataFrame | None:
    for ext, fn in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
        p = d / f"{name}.{ext}"
        if p.exists():
            try:
                return fn(p)
            except Exception:  # noqa: BLE001
                continue
    return None


def load(name: str):
    d = ROOT / "results" / name
    runs = _read(d, "runs")
    if runs is None:
        raise SystemExit(f"no dataset in {d}: run build-dataset {name}")
    runs = runs[runs["complete"].astype(bool)]
    dec = _read(d, "decisions")
    dec = dec[dec["run_id"].isin(runs["run_id"])] if dec is not None else pd.DataFrame()
    reasoning = _read(d, "reasoning")
    tags = _read(d, "reasoning_tags")
    return runs, dec, reasoning, tags


# --------------------------------------------------------------------------- §5.1 cells
def _ratios(c: np.ndarray) -> dict:
    return odean.ratios(dict(zip(CNT, c)))


def cell_table(runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per persona x arm, plus pooled rows ('profiles (pooled)', and per kind), with 95% CIs
    from a cluster bootstrap that resamples paths (scenarios)."""
    rng = np.random.default_rng(seed)
    scen = np.sort(runs["scenario"].unique())
    idx = rng.integers(0, len(scen), (B, len(scen)))
    groups = [(p, g) for p, g in runs.groupby("persona")]
    prof = runs[runs["persona_kind"] == "profile"]
    if not prof.empty:
        groups.append(("profiles (pooled)", prof))
    rows = []
    for pers, g in groups:
        kind = g["persona_kind"].iloc[0] if pers != "profiles (pooled)" else "pooled"
        for arch in ARCHS + ["all arms"]:
            h = g if arch == "all arms" else g[g["architecture"] == arch]
            if h.empty:
                continue
            per = h.groupby("scenario")[CNT].sum().reindex(scen, fill_value=0).to_numpy()
            est = _ratios(per.sum(axis=0))
            bs = per[idx].sum(axis=1)                       # B x 4
            with np.errstate(invalid="ignore", divide="ignore"):
                pgr = bs[:, 0] / (bs[:, 0] + bs[:, 2])
                plr = bs[:, 1] / (bs[:, 1] + bs[:, 3])
                ratio = pgr / plr
            row = {"persona": pers, "kind": kind, "architecture": arch, "runs": len(h),
                   "n_active": h["n_active"].iloc[0] if "n_active" in h and kind != "pooled"
                   else None, **{k: int(v) for k, v in zip(CNT, per.sum(axis=0))}}
            for k, v in (("PGR", pgr), ("PLR", plr), ("DE", pgr - plr), ("Ratio", ratio)):
                f = v[np.isfinite(v)]
                lo, hi = np.percentile(f, [2.5, 97.5]) if len(f) else (np.nan, np.nan)
                row |= {k: est[k], f"{k}_lo": lo, f"{k}_hi": hi}
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- §5.2 regression
def attr_cols() -> list[str]:
    return [f"attr_{k}" for k in ATTRIBUTE_KEYS]


def design_matrix(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """sold = a + b gain + Σ c_k attr_k + Σ d_k gain×attr_k + Σ e_m mem_m + Σ f_m gain×mem_m
    + ret + holding_period + drawdown (plan §5.2)."""
    gain = d["gain"].to_numpy(float)
    cols, names = [np.ones(len(d)), gain], ["const", "gain"]
    for a in attr_cols():
        cols.append(d[a].to_numpy(float))
        names.append(a)
    for a in attr_cols():
        cols.append(gain * d[a].to_numpy(float))
        names.append(f"gain:{a}")
    for m in MEMS:
        mm = (d["architecture"] == m).to_numpy(float)
        cols += [mm]
        names += [f"mem_{m}"]
    for m in MEMS:
        cols.append(gain * (d["architecture"] == m).to_numpy(float))
        names.append(f"gain:mem_{m}")
    for c in CONTROLS:
        cols.append(d[c].to_numpy(float))
        names.append(c)
    return np.column_stack(cols), names


def regression_sample(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec[dec["persona_kind"] == "profile"].dropna(subset=["gain"]).copy()
    d["gain"] = d["gain"].astype(int)
    d["sold"] = d["sell"].astype(int)
    return d


def regression(dec: pd.DataFrame):
    """Returns (fit, sample, note, omitted). `omitted` lists every §5.2 term that is not in the
    fitted model, with the reason, so the regression table can show it.

    With fewer distinct profiles than attribute parameters (e.g. the pilot's 8 profiles
    against 11 attributes + a constant), the attribute terms are not identified; the model then
    uses profile fixed effects instead of attr_k and drops the gain x attr_k terms, so the
    memory terms can still be estimated."""
    d = regression_sample(dec)
    if d.empty or d["persona"].nunique() < 2 or d["scenario"].nunique() < 2:
        return None, d, "not estimable (needs at least 2 profiles and 2 paths with sales)", []
    X, names = design_matrix(d)
    keep = X.std(axis=0) > 0
    keep[0] = True
    omitted = [(n, "omitted: no variation in the sample") for n, k in zip(names, keep) if not k]
    full_names = names
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    note = "full §5.2 model"
    if np.linalg.matrix_rank(X) < X.shape[1]:
        fe = pd.get_dummies(d["persona"], drop_first=True, dtype=float)
        gain = d["gain"].to_numpy(float)
        cols = [np.ones(len(d)), gain, fe.to_numpy()]
        names = ["const", "gain"] + [f"profile_{c}" for c in fe.columns]
        for m in MEMS:
            mm = (d["architecture"] == m).to_numpy(float)
            cols += [mm[:, None], (gain * mm)[:, None]]
            names += [f"mem_{m}", f"gain:mem_{m}"]
        cols += [d[c].to_numpy(float)[:, None] for c in CONTROLS]
        names += CONTROLS
        X = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])
        note = (f"attribute terms NOT IDENTIFIED with {d['persona'].nunique()} profiles "
                f"({len(ATTRIBUTE_KEYS)} attributes + constant); fitted with profile fixed "
                "effects and gain x memory terms only. Attribute effects need the main run.")
        why = (f"omitted: not identified with {d['persona'].nunique()} profiles "
               "(absorbed by the profile fixed effects)")
        omitted = [(n, why) for n in full_names if n.startswith(("attr_", "gain:attr_"))]
        omitted.append((f"profile_{sorted(d['persona'].unique())[0]}",
                        "reference profile (absorbed in const)"))
    fit = ols(d["sold"].to_numpy(float), X, names,
              [d["scenario"].to_numpy(), d["persona"].to_numpy()])
    return fit, d, note, omitted


def table_text(t: pd.DataFrame) -> str:
    """Plain-text table with blanks instead of NaN."""
    return t.round(4).astype(object).where(t.notna(), "").to_string(index=False)


def coef_table(fit, omitted=()) -> pd.DataFrame:
    """Every estimated term, then every §5.2 term left out of the fit (with the reason)."""
    lo, hi = fit.ci()
    t = pd.DataFrame({"term": fit.names, "coef": fit.beta, "se": fit.se, "p": fit.pvalues(),
                      "ci_lo": lo, "ci_hi": hi, "status": "estimated"})
    if omitted:
        t = pd.concat([t, pd.DataFrame([{"term": n, "status": why} for n, why in omitted])],
                      ignore_index=True)
    return t


def attribute_de(runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Raw change in pooled PGR - PLR: profiles with the attribute at 1 minus at 0 (all arms),
    and log/reflect minus none, with path-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    prof = runs[runs["persona_kind"] == "profile"]
    if prof.empty:
        return pd.DataFrame()
    scen = np.sort(prof["scenario"].unique())
    idx = rng.integers(0, len(scen), (B, len(scen)))

    def de(per):
        with np.errstate(invalid="ignore", divide="ignore"):
            return per[..., 0] / (per[..., 0] + per[..., 2]) - per[..., 1] / (per[..., 1] + per[..., 3])

    def contrast(a, b):
        pa = a.groupby("scenario")[CNT].sum().reindex(scen, fill_value=0).to_numpy()
        pb = b.groupby("scenario")[CNT].sum().reindex(scen, fill_value=0).to_numpy()
        est = de(pa.sum(0)) - de(pb.sum(0))
        bs = de(pa[idx].sum(1)) - de(pb[idx].sum(1))
        bs = bs[np.isfinite(bs)]
        lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
        return est, lo, hi

    rows = []
    for k in ATTRIBUTE_KEYS:
        c = f"attr_{k}"
        if prof[c].nunique() < 2:
            continue
        e, lo, hi = contrast(prof[prof[c] == 1], prof[prof[c] == 0])
        rows.append({"term": f"gain:{c}", "label": LABELS[k], "raw_DE_diff": e,
                     "raw_lo": lo, "raw_hi": hi})
    for m in MEMS:
        e, lo, hi = contrast(prof[prof["architecture"] == m], prof[prof["architecture"] == "none"])
        rows.append({"term": f"gain:mem_{m}", "label": f"{m} - none", "raw_DE_diff": e,
                     "raw_lo": lo, "raw_hi": hi})
    return pd.DataFrame(rows)


def de_by_variable(runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Pooled PGR, PLR and DE at each level of each variable, with 95% path-bootstrap CIs:
    the 11 attributes (profiles only, level 0 and 1), the memory arm (profiles only) and the
    persona type (C0, C1, profiles). One row per variable x level, for the selectable chart."""
    rng = np.random.default_rng(seed)
    if runs.empty:
        return pd.DataFrame()
    scen = np.sort(runs["scenario"].unique())
    idx = rng.integers(0, len(scen), (B, len(scen)))
    prof = runs[runs["persona_kind"] == "profile"]
    rows = []

    def add(var, label, level, g, order):
        per = g.groupby("scenario")[CNT].sum().reindex(scen, fill_value=0).to_numpy()
        est = _ratios(per.sum(0))
        bs = per[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            de = bs[:, 0] / (bs[:, 0] + bs[:, 2]) - bs[:, 1] / (bs[:, 1] + bs[:, 3])
        de = de[np.isfinite(de)]
        lo, hi = np.percentile(de, [2.5, 97.5]) if len(de) else (np.nan, np.nan)
        rows.append({"variable": var, "label": label, "level": level, "order": order,
                     "runs": len(g), "PGR": est["PGR"], "PLR": est["PLR"], "DE": est["DE"],
                     "DE_lo": lo, "DE_hi": hi})

    for k in ATTRIBUTE_KEYS:
        c = f"attr_{k}"
        if c in prof:
            for lvl in (0, 1):
                g = prof[prof[c] == lvl]
                if not g.empty:
                    add(k, LABELS[k], f"level {lvl}", g, lvl)
    for i, a in enumerate(ARCHS):
        g = prof[prof["architecture"] == a]
        if not g.empty:
            add("memory", "Memory arm (profiles)", a, g, i)
    for i, (lvl, g) in enumerate((("C0", runs[runs["persona"] == "C0"]),
                                  ("C1", runs[runs["persona"] == "C1"]), ("profiles", prof))):
        if not g.empty:
            add("persona_type", "Persona type", lvl, g, i)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- §5.3 tests
def _path_de(runs: pd.DataFrame, persona: str) -> pd.Series:
    """Per path: DE of the persona pooled over arms."""
    g = runs[runs["persona"] == persona].groupby("scenario")[CNT].sum()
    return g.apply(lambda r: _ratios(r.to_numpy())["DE"], axis=1).astype(float)


def paired(a: pd.Series, b: pd.Series, rng, alternative="two-sided") -> dict:
    d = (a - b).dropna()
    out = {"n_paths": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None}
    if len(d) >= 2:
        bs = rng.choice(d.to_numpy(), (B, len(d))).mean(axis=1)
        out |= {"ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d, alternative=alternative).pvalue) \
                if (d != 0).any() else 1.0
        except ValueError:
            out["p_wilcoxon"] = None
        out["mde"] = mde(out["se"], ALPHA, dof=len(d) - 1) if out["se"] > 0 else None
    return out


def sameness(reasoning: pd.DataFrame | None) -> dict | None:
    """Share of identical decisions (the action on each stock) between two personas on the
    same path, arm and period: profile pairs against the C0-C1 pair as reference. Period 1
    is reported separately because only there are all states identical."""
    if reasoning is None or reasoning.empty or "actions" not in reasoning:
        return None
    r = reasoning.dropna(subset=["actions"])
    out = {}
    for label, sub in (("all periods", r), ("period 1", r[r["period"] == 1])):
        prof_same, prof_n, ref_same, ref_n = 0, 0, 0, 0
        for _, g in sub.groupby(["scenario", "architecture", "period"]):
            p = g[g["persona_kind"] == "profile"]["actions"].to_numpy()
            if len(p) >= 2:
                _, counts = np.unique(p, return_counts=True)
                prof_same += int((counts * (counts - 1) // 2).sum())
                prof_n += len(p) * (len(p) - 1) // 2
            c = g.set_index("persona")["actions"]
            if "C0" in c and "C1" in c:
                ref_n += 1
                ref_same += int(c["C0"] == c["C1"])
        out[label] = {"profile_pairs_identical": prof_same / prof_n if prof_n else None,
                      "profile_pairs": prof_n,
                      "C0_C1_identical": ref_same / ref_n if ref_n else None,
                      "C0_C1_pairs": ref_n}
    return out


def tag_tests(tags: pd.DataFrame | None, reasoning: pd.DataFrame | None,
              dec: pd.DataFrame) -> dict | None:
    if tags is None or tags.empty:
        return None
    t = tags.copy()
    out = {"rate_by_arm": {a: {tag: float(g[f"tag_{tag}"].mean()) for tag in TAGS}
                           for a, g in t.groupby("architecture")},
           "rate_overall": {tag: float(t[f"tag_{tag}"].mean()) for tag in TAGS}}
    # past_trades by arm (RQ3 memory)
    out["past_trades_by_arm"] = {a: float(g["tag_past_trades"].mean())
                                 for a, g in t.groupby("architecture")}
    # persona tag against the number of active attributes (profiles + C0)
    pt = t[t["persona_kind"].isin(["profile", "C0"])].dropna(subset=["n_active"])
    if pt["n_active"].nunique() > 1:
        X = np.column_stack([np.ones(len(pt)), pt["n_active"].to_numpy(float)])
        f = ols(pt["tag_persona"].astype(float).to_numpy(), X, ["const", "n_active"],
                [pt["scenario"].to_numpy(), pt["persona"].to_numpy()])
        out["persona_tag_vs_n_active"] = {"slope": float(f.beta[1]), "se": float(f.se[1]),
                                          "p": float(f.pvalues()[1]),
                                          "rate_by_n_active": pt.groupby("n_active")["tag_persona"]
                                          .mean().astype(float).to_dict()}
    # reference_point tag as a predictor of selling winners (gain positions on sale days)
    if not dec.empty:
        key = t.set_index(["run_id", "period"])["tag_reference_point"]
        g = dec.dropna(subset=["gain"])
        g = g[g["gain"] == 1].join(key, on=["run_id", "period"]).dropna(subset=["tag_reference_point"])
        if len(g) > 10 and g["tag_reference_point"].nunique() > 1:
            X = np.column_stack([np.ones(len(g)), g["tag_reference_point"].astype(float)])
            f = ols(g["sell"].astype(float).to_numpy(), X, ["const", "reference_point"],
                    [g["scenario"].to_numpy(), g["persona"].to_numpy()])
            out["reference_point_predicts_selling_winners"] = {
                "coef": float(f.beta[1]), "se": float(f.se[1]), "p": float(f.pvalues()[1]),
                "n": int(len(g))}
    by_attr = {}
    tp = t[t["persona_kind"] == "profile"]
    for k in ATTRIBUTE_KEYS:
        c = f"attr_{k}"
        if c in tp:
            by_attr[k] = {str(lvl): {tag: float(g[f"tag_{tag}"].mean()) for tag in TAGS}
                          for lvl, g in tp.groupby(c)}
    out["rate_by_attribute"] = by_attr
    return out


def run_tests(runs, dec, fit, reasoning, tags, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict = {"alpha": ALPHA, "blind_spot_band": BAND}
    blind = pd.DataFrame()
    if fit is not None:
        dk = [n for n in fit.names if n.startswith("gain:attr_")]
        fm = [n for n in fit.names if n.startswith("gain:mem_")]
        out["rq3_memory_wald"] = {**fit.wald(fm), "terms": fm} if fm else None
        out["memory_coefs"] = {n: {"coef": float(fit.beta[j]), "se": float(fit.se[j]),
                                   "p": float(fit.pvalues()[j]),
                                   "mde": mde(fit.se[j], ALPHA, dof=fit.dof)}
                               for n, j in zip(fm, fit.idx(fm))}
    if fit is not None and any(n.startswith("gain:attr_") for n in fit.names):
        out["rq1_joint"] = {**fit.wald(dk), "terms": dk}
        i = fit.idx(dk)
        p = fit.pvalues()[i]
        lo, hi = fit.ci()
        m_holm = mde(fit.se[i].mean(), ALPHA / len(dk), dof=fit.dof)
        blind = pd.DataFrame({
            "term": dk, "attribute": [n.removeprefix("gain:attr_") for n in dk],
            "label": [LABELS[n.removeprefix("gain:attr_")] for n in dk],
            "coef": fit.beta[i], "se": fit.se[i], "ci_lo": lo[i], "ci_hi": hi[i], "p": p,
            "p_holm": holm(p),
            "mde_holm": [mde(s, ALPHA / len(dk), dof=fit.dof) for s in fit.se[i]],
            "mde_unadjusted": [mde(s, ALPHA, dof=fit.dof) for s in fit.se[i]]})
        blind["significant_holm"] = blind["p_holm"] < ALPHA
        blind["blind_spot"] = (blind["ci_lo"] > -BAND) & (blind["ci_hi"] < BAND)
        out["rq2"] = {"significant": blind.loc[blind["significant_holm"], "attribute"].tolist(),
                      "blind_spots": blind.loc[blind["blind_spot"], "attribute"].tolist(),
                      "mean_mde_holm": m_holm, "dof": fit.dof, "n_rows": fit.n}
    personas = sorted(runs["persona"].unique())
    if "C0" in personas:
        c0 = _path_de(runs, "C0")
        prof = [p for p in personas if p.startswith("P")]
        pr = {p: paired(_path_de(runs, p), c0, rng) for p in prof}
        ps = [v.get("p_wilcoxon") for v in pr.values()]
        valid = [i for i, v in enumerate(ps) if v is not None]
        adj = holm(np.array([ps[i] for i in valid])) if valid else []
        for j, i in enumerate(valid):
            pr[prof[i]]["p_holm"] = float(adj[j])
        diffs = [v["mean_diff"] for v in pr.values() if v["mean_diff"] is not None]
        out["rq1_paired"] = {"by_profile": pr,
                             "spread_sd_of_profile_minus_C0": float(np.std(diffs, ddof=1))
                             if len(diffs) > 1 else None,
                             "n_significant_holm": int(sum(v.get("p_holm", 1) < ALPHA
                                                           for v in pr.values()))}
        if "C1" in personas:
            out["manipulation_check"] = {**paired(_path_de(runs, "C1"), c0, rng, "greater"),
                                         "hypothesis": "DE(C1) > DE(C0), one-sided"}
        out["C0_vs_profiles_pooled_DE"] = {
            "C0": _ratios(runs[runs["persona"] == "C0"][CNT].sum().to_numpy())["DE"],
            "profiles": _ratios(runs[runs["persona_kind"] == "profile"][CNT].sum()
                                .to_numpy())["DE"]}
    out["rq2_sameness"] = sameness(reasoning)
    out["rq3_tags"] = tag_tests(tags, reasoning, dec)
    return out, blind


# --------------------------------------------------------------------------- figures
def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_cells(cells: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    c = cells[cells["architecture"].isin(ARCHS)]
    order = [p for p in ["C0", "C1", "profiles (pooled)"] if p in set(c["persona"])] + \
        sorted(p for p in c["persona"].unique() if str(p).startswith("P"))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(order) + 2), 3.8))
    w = 0.26
    for i, a in enumerate(ARCHS):
        b = c[c["architecture"] == a].set_index("persona").reindex(order)
        x = np.arange(len(order)) + (i - 1) * w
        ax.bar(x, b["DE"], w - 0.03, color=style.ARCH_COLORS[a], label=a, zorder=2)
        ax.errorbar(x, b["DE"], yerr=[(b["DE"] - b["DE_lo"]).clip(lower=0),
                                      (b["DE_hi"] - b["DE"]).clip(lower=0)],
                    fmt="none", ecolor=style.INK_2, elinewidth=1, capsize=2, zorder=3)
    ax.axhline(0, color=style.INK_2, linewidth=1)
    ax.set_xticks(np.arange(len(order)), order, rotation=45, ha="right")
    ax.set_ylabel("DE = PGR − PLR")
    ax.set_title("Disposition effect by persona and memory arm (95% path-bootstrap CI)",
                 loc="left")
    ax.legend(title="memory arm", ncols=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_ratio(cells: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    bm = benchmark()
    c = cells[cells["architecture"] == "all arms"].dropna(subset=["Ratio"]).copy()
    order = [p for p in ["C0", "C1", "profiles (pooled)"] if p in set(c["persona"])] + \
        sorted(p for p in c["persona"].unique() if str(p).startswith("P"))
    c = c.set_index("persona").reindex(order).dropna(subset=["Ratio"])
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(c) + 3), 3.8))
    x = np.arange(len(c))
    colors = [style.SERIES[1] if p in ("C0", "C1") else style.SERIES[0] for p in c.index]
    ax.scatter(x, c["Ratio"], color=colors, zorder=3)
    ax.errorbar(x, c["Ratio"], yerr=[(c["Ratio"] - c["Ratio_lo"]).clip(lower=0),
                                     (c["Ratio_hi"] - c["Ratio"]).clip(lower=0)],
                fmt="none", ecolor=style.INK_2, elinewidth=1, capsize=2, zorder=2)
    for _, r in bm.iterrows():
        if r["investor_type"] in ("Individuals", "All investors", "Mutual funds",
                                  "Foreign investors"):
            ax.axhline(r["ratio"], color=style.GRID if r["ratio"] < 1 else style.INK_2,
                       linewidth=1, linestyle="--", zorder=1)
            ax.text(len(c) - 0.5, r["ratio"], f" {r['investor_type']} {r['ratio']:.2f}",
                    va="center", fontsize=8, color=style.INK_2)
    ax.axhline(1, color=style.INK_2, linewidth=0.8)
    ax.set_xticks(x, c.index, rotation=45, ha="right")
    ax.set_ylabel("PGR / PLR")
    ax.set_title("Ratio against Barber et al. (2007), Table 3 Panel B (all arms pooled)",
                 loc="left")
    ax.set_xlim(-0.5, len(c) + 2.5)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_forest(blind: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    b = blind.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(b) + 1.5))
    y = np.arange(len(b))
    ax.axvspan(-BAND, BAND, color=style.NEUTRAL, zorder=0, label=f"±{BAND} blind-spot band")
    m = b["mde_holm"].mean()
    for s in (-m, m):
        ax.axvline(s, color=style.SERIES[1], linestyle=":", linewidth=1.2)
    ax.plot([], [], color=style.SERIES[1], linestyle=":", label=f"MDE (Holm) ±{m:.3f}")
    ax.errorbar(b["coef"], y, xerr=[b["coef"] - b["ci_lo"], b["ci_hi"] - b["coef"]], fmt="o",
                color=style.SERIES[0], ecolor=style.SERIES[0], capsize=2, zorder=3)
    ax.axvline(0, color=style.INK_2, linewidth=1)
    ax.set_yticks(y, b["label"])
    ax.set_xlabel("d_k: change in P(sell | gain) − P(sell | loss)  (≈ change in PGR − PLR)")
    ax.set_title("Attribute effects on the disposition effect (95% CI, two-way clustered)",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_memory(cells: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    c = cells[cells["architecture"].isin(ARCHS)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for p, g in c.groupby("persona"):
        g = g.set_index("architecture").reindex(ARCHS)
        if p in ("C0", "C1", "profiles (pooled)"):
            col = {"C0": style.SERIES[1], "C1": style.SELL, "profiles (pooled)": style.INK}[p]
            ax.plot(range(3), g["DE"], marker="o", color=col, linewidth=2.5, label=p, zorder=3)
        else:
            ax.plot(range(3), g["DE"], color=style.SERIES[0], alpha=0.35, linewidth=1, zorder=2)
    ax.plot([], [], color=style.SERIES[0], alpha=0.5, label="each profile")
    ax.set_xticks(range(3), ARCHS)
    ax.axhline(0, color=style.INK_2, linewidth=1)
    ax.set_ylabel("DE = PGR − PLR")
    ax.set_title("Memory effects per persona", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_tags(tags: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    style.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                 gridspec_kw={"width_ratios": [1, 2.2]})
    r = tags.groupby("architecture")[TAGCOLS].mean().reindex(ARCHS)
    x = np.arange(len(TAGS))
    for i, a in enumerate(ARCHS):
        a1.bar(x + (i - 1) * 0.26, r.loc[a], 0.23, color=style.ARCH_COLORS[a], label=a)
    a1.set_xticks(x, [t.replace("_", " ") for t in TAGS], rotation=30, ha="right")
    a1.set_ylabel("share of reasoning texts")
    a1.set_title("Tag rates by memory arm", loc="left")
    a1.legend(fontsize=8)
    tp = tags[tags["persona_kind"] == "profile"]
    diff = pd.DataFrame({k: tp[tp[f"attr_{k}"] == 1][TAGCOLS].mean()
                         - tp[tp[f"attr_{k}"] == 0][TAGCOLS].mean()
                         for k in ATTRIBUTE_KEYS if f"attr_{k}" in tp}).T
    if not diff.empty:
        lim = max(0.05, float(np.nanmax(np.abs(diff.to_numpy()))))
        diff.columns = TAGS
        im = a2.imshow(diff.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        a2.set_yticks(range(len(diff)), [LABELS[k] for k in diff.index])
        a2.set_xticks(range(len(TAGS)), [t.replace("_", " ") for t in TAGS], rotation=30,
                      ha="right")
        a2.grid(False)
        fig.colorbar(im, ax=a2, label="rate at level 1 − level 0")
        a2.set_title("Tag rates by attribute", loc="left")
    _save(fig, path)


def fig_example_path(name: str, runs: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    from llm_investors.measure.dataset import read_log
    style.apply()
    persona = "C1" if "C1" in set(runs["persona"]) else runs["persona"].iloc[0]
    k = int(runs["scenario"].min())
    logs = {a: ROOT / "logs" / name / f"run_{a}_{persona}_s{k:03d}.jsonl" for a in ARCHS}
    logs = {a: p for a, p in logs.items() if p.exists()}
    if not logs:
        return
    first = read_log(next(iter(logs.values())))[0]
    prices, labels = np.array(first["prices"]), first["labels"]
    fig, axes = plt.subplots(1, len(labels), figsize=(13, 2.9), sharey=True)
    for j, (ax, s) in enumerate(zip(axes, labels)):
        ax.plot(range(len(prices)), prices[:, j], color=style.INK_2, linewidth=1.5)
        for i, (a, p) in enumerate(logs.items()):
            tr = [e for e in read_log(p) if e["type"] == "trade" and e["stock"] == s]
            for act, mk in (("sell", "v"), ("buy", "^")):
                e = [x for x in tr if x["action"] == act]
                ax.scatter([x["period"] for x in e], [x["price"] * (1 + 0.03 * (i - 1)) for x in e],
                           s=30, marker=mk, color=style.ARCH_COLORS[a], zorder=3,
                           label=f"{a} {act}" if j == 0 else None)
        ax.axhline(100, color=style.GRID, linewidth=1, zorder=1)
        ax.set_title(f"Stock {s}", loc="left")
        ax.set_xlabel("month")
    axes[0].set_ylabel("price")
    fig.legend(loc="upper right", ncols=6, fontsize=8)
    fig.suptitle(f"Path {k}, {persona}: trades by memory arm (▼ sell, ▲ buy)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(name: str):
    runs, dec, reasoning, tags = load(name)
    out = ROOT / "results" / name / "analysis"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for old in figs.glob("*.png"):
        old.unlink()                      # figures of an earlier run must not linger
    cells = cell_table(runs)
    cells.to_csv(out / "cells.csv", index=False)
    fit, sample, reg_note, omitted = regression(dec) if not dec.empty else \
        (None, dec, "no decisions", [])
    raw = attribute_de(runs)
    raw.to_csv(out / "attribute_de.csv", index=False)
    de_by_variable(runs).to_csv(out / "de_by_variable.csv", index=False)
    if fit is not None:
        tbl = coef_table(fit, omitted)
        tbl.to_csv(out / "regression.csv", index=False)
        text = (f"Model: {reg_note}\n"
                "Decision-level LPM (plan §5.2), profiles only; SEs two-way clustered by path "
                f"and profile; t with {fit.dof} df.\n"
                + ("sold ~ gain + attr_k + gain:attr_k + mem_m + gain:mem_m + ret + "
                   "holding_period + drawdown\n" if reg_note == "full §5.2 model" else
                   "sold ~ gain + profile FE + mem_m + gain:mem_m + ret + holding_period + "
                   "drawdown\n")
                + f"N = {fit.n}, paths = {sample['scenario'].nunique()}, profiles = "
                f"{sample['persona'].nunique()}\n\n"
                "d_k (gain:attr_k) and f_m (gain:mem_m) are the change in P(sell|gain) − "
                "P(sell|loss), the regression counterpart of PGR − PLR. Every term is listed; "
                "terms not in the fitted model are shown with the reason.\n\n"
                "TABLE 1. Regression\n"
                + table_text(tbl[["term", "coef", "se", "p", "ci_lo", "ci_hi"]]
                             .assign(status=tbl["status"].str.replace("omitted: ", "")))
                + "\n\nTABLE 2. Raw DE difference (pooled PGR − PLR, level 1 minus level 0; "
                "memory: arm minus none), 95% path-bootstrap CI\n"
                + ("" if reg_note == "full §5.2 model" else
                   "CAUTION: with this few profiles attributes that are on together cannot be "
                   "separated; read these as descriptive only.\n")
                + (table_text(raw[["label", "raw_DE_diff", "raw_lo", "raw_hi"]])
                   if not raw.empty else "(none)"))
    else:
        text = f"Regression {reg_note}."
    (out / "regression.txt").write_text(text)
    tests, blind = run_tests(runs, dec, fit, reasoning, tags)
    blind.to_csv(out / "blind_spots.csv", index=False)
    tests["n_runs"], tests["n_decision_rows"] = int(len(runs)), int(len(dec))
    tests["regression_model"] = reg_note
    (out / "tests.json").write_text(json.dumps(tests, indent=2, default=float))
    if not cells.empty:
        fig_cells(cells, figs / "de_by_persona_arm.png")
        fig_ratio(cells, figs / "ratio_vs_benchmark.png")
        fig_memory(cells, figs / "memory_effects.png")
    if not blind.empty:
        fig_forest(blind, figs / "forest_attributes.png")
    if tags is not None and not tags.empty:
        fig_tags(tags, figs / "tag_rates.png")
    fig_example_path(name, runs, figs / "example_path.png")
    power_dir = ROOT / "results" / name / "power"
    if (power_dir / "power_curves.png").exists():
        import shutil
        shutil.copy(power_dir / "power_curves.png", figs / "power_curves.png")
    print(text)
    print("\nCells (all arms):")
    print(cells[cells["architecture"] == "all arms"][["persona", "runs", "PGR", "PLR", "DE",
                                                       "DE_lo", "DE_hi", "Ratio"]]
          .round(3).to_string(index=False))
    mc = tests.get("manipulation_check")
    if mc:
        print(f"\nManipulation check DE(C1) − DE(C0): {mc['mean_diff']:+.3f} "
              f"[{mc.get('ci_lo', float('nan')):+.3f}, {mc.get('ci_hi', float('nan')):+.3f}], "
              f"one-sided p {mc.get('p_wilcoxon')}")
    if "rq1_joint" in tests:
        print(f"RQ1 joint Wald (all d_k = 0): p = {tests['rq1_joint']['p']:.4f}")
    print(f"-> {out}")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study1_v3_gpt-6-luna")


if __name__ == "__main__":
    cli()
