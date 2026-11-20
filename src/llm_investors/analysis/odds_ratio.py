"""Odds ratios and marginal effects of a paper gain on selling, by memory architecture, with
standard errors clustered three ways (free to re-run; reads results/<name>/ like study1).

    analyze-odds                   # real results
    analyze-odds study1_v2

Outputs in results/<name>/analysis/:
  odds_ratio.txt           readable report: OR and AME per spec x cluster, paper comparison
  odds_ratios.csv          OR of a gain per architecture (and ratio of ORs vs 'none')
  marginal_effects.csv     average marginal effect of a gain per architecture (and vs 'none')
  cluster_comparison.csv   gain / gain x architecture coefficients under each clustering
  paper_comparison.csv     raw PGR, PLR, PGR/PLR and OR per architecture next to Barber et al.

Barber, Lee, Liu & Odean (2007) report PGR / PLR, not an odds ratio. Their 2.629 for
individual investors is PGR / PLR = 4.39 / 1.67 (a ratio of selling rates); the matching odds
ratio is about 2.70. Compare like with like: our logit gives odds ratios, `raw_comparison` gives
both. Their PGR / PLR are value-weighted daily rates; ours are counts on monthly periods.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats

from llm_investors.analysis.study1 import ARCHS, load
from llm_investors.measure import odean
from llm_investors.paths import ROOT

ARCH = "C(architecture, Treatment(reference='none'))"
SPECS = {
    # Same as the "Logit (robustness)" model in study1.regressions, so the two reconcile.
    "main": f"sell ~ gain * {ARCH} + ret + holding_period + drawdown + C(fe)",
    # Meeting 2 controls: dd, period, persona, seed (persona x seed are absorbed by C(fe)).
    "period FE": f"sell ~ gain * {ARCH} + ret + holding_period + drawdown + C(fe) + C(period)",
}
# A scenario's price paths are shared by every persona and architecture, so "path" is the
# coarsest level; "matched_set" (persona x scenario) is study1's default; "agent" is one run.
CLUSTERS = {
    "path": lambda d: d["scenario"],
    "matched_set": lambda d: d["persona"] + "_" + d["scenario"].astype(str),
    "agent": lambda d: d["run_id"],
}
# Barber et al. (2007), Table 3. Panel A in $TW trillion; Panel B PGR / PLR in percent.
PAPER = {
    "individual investors": {"PG": 966.46, "PL": 1639.46, "RG": 45.70, "RL": 27.82,
                             "PGR": 4.39, "PLR": 1.67},
    "all investors": {"PG": 1740.89, "PL": 2161.24, "RG": 52.55, "RL": 31.18,
                      "PGR": 2.93, "PLR": 1.42},
}
Z = stats.norm.ppf(0.975)


def prepare(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.dropna(subset=["gain"]).copy()
    d["gain"] = d["gain"].astype(int)
    d["fe"] = d["persona"] + "_" + d["scenario"].astype(str)
    return d.reset_index(drop=True)


def fit_logit(d: pd.DataFrame, formula: str, cluster: str):
    g = CLUSTERS[cluster](d).astype("category").cat.codes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.logit(formula, d).fit(cov_type="cluster", cov_kwds={"groups": g},
                                        disp=0, maxiter=200)
    if not res.mle_retvals.get("converged", True):
        raise RuntimeError("did not converge (possible separation)")
    res.n_clusters = int(g.nunique())
    return res


def _row(est: float, se: float, **kw) -> dict:
    z = est / se if se > 0 else np.nan
    return {**kw, "estimate": est, "se": se, "ci_lo": est - Z * se, "ci_hi": est + Z * se,
            "p": 2 * stats.norm.sf(abs(z))}


def _gain_terms(res, arch: str) -> np.ndarray:
    """Contrast vector picking the log-odds effect of a gain under `arch`."""
    L = pd.Series(0.0, index=res.params.index)
    L["gain"] = 1.0
    if arch != "none":
        L[f"gain:{ARCH}[T.{arch}]"] = 1.0
    return L.to_numpy()


def odds_ratios(res, archs: list[str]) -> list[dict]:
    """OR of selling a winner vs a loser per architecture, exp(b_gain [+ b_gain:arch]), with a
    95% CI from the clustered covariance; plus the ratio of ORs vs 'none', exp(b_gain:arch)."""
    b, V = res.params.to_numpy(), res.cov_params().to_numpy()
    rows = []
    for a in archs:
        L = _gain_terms(res, a)
        r = _row(float(L @ b), float(np.sqrt(L @ V @ L)), architecture=a, measure="OR")
        rows.append(r)
        if a != "none":
            L = L - _gain_terms(res, "none")
            rows.append(_row(float(L @ b), float(np.sqrt(L @ V @ L)),
                             architecture=f"{a} / none", measure="ratio of ORs"))
    for r in rows:  # log scale -> odds-ratio scale (p-values stay those of the log-odds test)
        r["log_odds"] = r["estimate"]
        r["estimate"], r["ci_lo"], r["ci_hi"] = (np.exp(r[k]) for k in
                                                ("estimate", "ci_lo", "ci_hi"))
        r["se"] = r["estimate"] * r["se"]  # delta-method SE of the OR
    return rows


def _ame_and_grad(res, d: pd.DataFrame, arch: str) -> tuple[float, np.ndarray]:
    """mean[F(x1 b) - F(x0 b)] with every row set to `arch` and gain 1 vs 0, and its gradient
    mean[f(x1 b) x1 - f(x0 b) x0]. Rebuilding the design rebuilds the interaction terms."""
    data = res.model.data  # statsmodels >= 0.15 renamed design_info to model_spec
    info = getattr(data, "model_spec", None) or data.design_info
    b = res.params.to_numpy()
    X = [np.asarray(patsy.build_design_matrices([info], d.assign(architecture=arch,
                                                                 gain=g))[0])
         for g in (1, 0)]
    xb = [x @ b for x in X]
    F = [1 / (1 + np.exp(-v)) for v in xb]
    f = [p * (1 - p) for p in F]
    ame = float(np.mean(F[0] - F[1]))
    grad = (f[0][:, None] * X[0] - f[1][:, None] * X[1]).mean(axis=0)
    return ame, grad


def marginal_effects(res, d: pd.DataFrame, archs: list[str]) -> list[dict]:
    """Average marginal effect of a paper gain on P(sell) per architecture (delta-method SEs
    from the clustered covariance) and the AME difference vs 'none'. The interaction
    coefficient alone is not this difference in a logit."""
    V = res.cov_params().to_numpy()
    est = {a: _ame_and_grad(res, d, a) for a in archs}
    rows = [_row(e, float(np.sqrt(g @ V @ g)), architecture=a, measure="AME")
            for a, (e, g) in est.items()]
    for a in archs:
        if a != "none" and "none" in est:
            e, g = est[a][0] - est["none"][0], est[a][1] - est["none"][1]
            rows.append(_row(e, float(np.sqrt(g @ V @ g)), architecture=f"{a} - none",
                             measure="AME difference"))
    return rows


def paper_benchmark() -> pd.DataFrame:
    """Barber et al. (2007) Table 3: PGR / PLR as reported, the matching odds ratio, and the
    odds ratio from the Panel A amounts (RG/PG over RL/PL)."""
    rows = []
    for who, v in PAPER.items():
        pgr, plr = v["PGR"] / 100, v["PLR"] / 100
        rows.append({"source": f"Barber et al. (2007), {who}", "PGR": pgr, "PLR": plr,
                     "DE": pgr - plr, "PGR/PLR": pgr / plr,
                     "OR": (pgr / (1 - pgr)) / (plr / (1 - plr)),
                     "OR (Panel A amounts)": (v["RG"] / v["PG"]) / (v["RL"] / v["PL"])})
    return pd.DataFrame(rows)


def raw_comparison(d: pd.DataFrame) -> pd.DataFrame:
    """Pooled PGR / PLR per architecture from the sale-day decision rows (no controls), with
    the same PGR/PLR ratio and odds ratio as the paper reports."""
    d = d.assign(cls=np.where(d["sell"] == 1, np.where(d["gain"] == 1, "RG", "RL"),
                              np.where(d["gain"] == 1, "PG", "PL")))
    rows = []
    for a in [a for a in ARCHS if a in set(d["architecture"])]:
        c = d[d["architecture"] == a]["cls"].value_counts()
        r = odean.ratios({k: c.get(k, 0) for k in ("RG", "RL", "PG", "PL")})
        ok = all(r[k] for k in ("PG", "PL", "RL")) and r["PLR"]
        rows.append({"source": f"LLM agents, {a}", "PGR": r["PGR"], "PLR": r["PLR"],
                     "DE": r["DE"], "PGR/PLR": r["PGR"] / r["PLR"] if ok else np.nan,
                     "OR": (r["RG"] / r["PG"]) / (r["RL"] / r["PL"]) if ok else np.nan,
                     "OR (Panel A amounts)": np.nan,
                     **{k: r[k] for k in ("RG", "RL", "PG", "PL")}})
    return pd.concat([paper_benchmark(), pd.DataFrame(rows)], ignore_index=True)


def analyze(d: pd.DataFrame, specs: dict = SPECS, clusters=tuple(CLUSTERS)):
    archs = [a for a in ARCHS if a in set(d["architecture"])]
    ors, ames, coefs, failed = [], [], [], []
    for spec, formula in specs.items():
        for cl in clusters:
            try:
                res = fit_logit(d, formula, cl)
            except Exception as e:  # noqa: BLE001
                failed.append({"spec": spec, "cluster": cl, "error": f"{type(e).__name__}: {e}"})
                continue
            meta = {"spec": spec, "cluster": cl, "n_clusters": res.n_clusters,
                    "N": int(res.nobs)}
            ors += [{**meta, **r} for r in odds_ratios(res, archs)]
            ames += [{**meta, **r} for r in marginal_effects(res, d, archs)]
            keep = [k for k in res.params.index if k.startswith("gain")]
            coefs += [{**meta, "term": k, "coef": res.params[k], "se": res.bse[k],
                       "p": res.pvalues[k]} for k in keep]
    return pd.DataFrame(ors), pd.DataFrame(ames), pd.DataFrame(coefs), failed


def report(ors, ames, coefs, paper, failed) -> str:
    cols = ["architecture", "estimate", "ci_lo", "ci_hi", "p"]
    out = ["Odds ratio and average marginal effect of a paper gain on P(sell), by memory "
           "architecture.\nOR > 1 (AME > 0) = winners sold more readily than losers, i.e. a "
           "disposition effect.\n"]
    for (spec, cl), g in ors.groupby(["spec", "cluster"], sort=False):
        a = ames[(ames["spec"] == spec) & (ames["cluster"] == cl)]
        out.append(f"== {spec}, clustered by {cl} (N = {g['N'].iat[0]}, clusters = "
                   f"{g['n_clusters'].iat[0]})\n{SPECS.get(spec, '')}\n"
                   f"-- odds ratios\n{g[cols].round(4).to_string(index=False)}\n"
                   f"-- average marginal effects\n{a[cols].round(4).to_string(index=False)}\n")
    wide = coefs.pivot_table(index=["spec", "term"], columns="cluster", values=["se", "p"],
                             sort=False)
    out.append(f"== Cluster comparison (same coefficients, different SEs)\n"
               f"{wide.round(4).to_string()}\n")
    show = ["source", "PGR", "PLR", "DE", "PGR/PLR", "OR", "OR (Panel A amounts)"]
    out.append("== Raw comparison with Barber et al. (2007) (no controls)\n"
               f"{paper[show].round(4).to_string(index=False)}\n"
               "The paper's 2.629 is PGR/PLR for individual investors, not an odds ratio. "
               "Paper: value-weighted, daily; agents: counts, monthly periods.\n")
    if failed:
        out.append("FAILED MODELS: " + "; ".join(f"{f['spec']} / {f['cluster']}: {f['error']}"
                                                  for f in failed) + "\n")
    return "\n".join(out)


def main(name: str = "study1"):
    _, dec = load(name)
    d = prepare(dec)
    out = ROOT / "results" / name / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    ors, ames, coefs, failed = analyze(d)
    paper = raw_comparison(d)
    ors.to_csv(out / "odds_ratios.csv", index=False)
    ames.to_csv(out / "marginal_effects.csv", index=False)
    coefs.to_csv(out / "cluster_comparison.csv", index=False)
    paper.to_csv(out / "paper_comparison.csv", index=False)
    text = report(ors, ames, coefs, paper, failed)
    (out / "odds_ratio.txt").write_text(text)
    print(text)
    print(f"-> {out}")


def cli():
    main(sys.argv[1] if len(sys.argv) > 1 else "study1")


if __name__ == "__main__":
    cli()
