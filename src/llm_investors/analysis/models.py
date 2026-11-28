"""Robustness models for the §5.2 regression: a logit next to the LPM, odds ratios, average
marginal effects (AMEs) and alternative clustering of the standard errors.

- Logit: maximum likelihood by Newton-Raphson; cluster-robust sandwich with the same CGM
  one-/two-way formula as the LPM (bread = (X'WX)^-1, scores = X(y - p), factor G/(G-1) as in
  Stata's vce(cluster)).
- Odds ratios: exp(coef). For gain:attr_k this is the factor by which attr_k multiplies the
  odds ratio of selling a gain rather than a loss.
- AMEs: averages over the estimation sample of counterfactual predictions. For a binary
  variable, P(sell | v = 1) - P(sell | v = 0); for a continuous control, dP/dx. The
  "effect on DE" of v is the average change in P(sell | gain) - P(sell | loss) when v goes
  from 0 to 1 (Ai & Norton 2003 for the logit). In the LPM it equals d_k / f_m exactly.
  SEs by the delta method with a numerical gradient and the model's clustered covariance.
- Clustering: path x profile (primary, plan §5.2), path, profile, agent run (one persona x
  memory arm x path). t / F use G_min - 1 df, as for the LPM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from llm_investors.analysis.ols import Fit, cluster_vcov, holm, ols

CLUSTERINGS: dict[str, list[str]] = {
    "path × profile": ["scenario", "persona"],
    "path": ["scenario"],
    "profile": ["persona"],
    "agent run": ["run_id"],
}
PRIMARY = "path × profile"
MODELS = ["LPM", "logit"]


@dataclass
class ModelFit:
    model: str
    clustering: str
    fit: Fit
    converged: bool = True

    def F(self, eta: np.ndarray) -> np.ndarray:
        return eta if self.model == "LPM" else expit(eta)


def logit(y: np.ndarray, X: np.ndarray, names: list[str], clusters: list[np.ndarray],
          max_iter: int = 100, tol: float = 1e-10) -> tuple[Fit, bool]:
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    converged = False
    for _ in range(max_iter):
        p = expit(X @ beta)
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.lstsq(H, X.T @ (y - p), rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    p = expit(X @ beta)
    bread = np.linalg.pinv(X.T @ (X * (p * (1 - p))[:, None]))
    V, g_min = cluster_vcov(bread, X, y - p, clusters)
    # separation shows up as huge coefficients that never settle
    converged = converged and bool(np.all(np.abs(beta) < 25))
    return Fit(list(names), beta, V, len(y), g_min), converged


def fit_all(d: pd.DataFrame, X: np.ndarray, names: list[str]) -> list[ModelFit]:
    """LPM and logit under every clustering. Coefficients do not depend on the clustering, so
    each model is estimated once and only the covariance is recomputed."""
    y = d["sold"].to_numpy(float)
    out = []
    for cname, cols in CLUSTERINGS.items():
        cl = [d[c].to_numpy() for c in cols]
        out.append(ModelFit("LPM", cname, ols(y, X, names, cl)))
        f, ok = logit(y, X, names, cl)
        out.append(ModelFit("logit", cname, f, ok))
    return out


def coef_rows(m: ModelFit) -> pd.DataFrame:
    f = m.fit
    lo, hi = f.ci()
    t = pd.DataFrame({"model": m.model, "clustering": m.clustering, "term": f.names,
                      "coef": f.beta, "se": f.se, "p": f.pvalues(), "ci_lo": lo, "ci_hi": hi,
                      "n_clusters_min": f.g_min, "converged": m.converged})
    if m.model == "logit":
        with np.errstate(over="ignore"):
            t["odds_ratio"], t["or_lo"], t["or_hi"] = np.exp(f.beta), np.exp(lo), np.exp(hi)
    dk = t["term"].str.startswith("gain:attr_")
    t["p_holm_dk"] = np.nan
    if dk.any():
        t.loc[dk, "p_holm_dk"] = holm(t.loc[dk, "p"].to_numpy())
    return t


# --------------------------------------------------------------------------- marginal effects
def _effects(d: pd.DataFrame, names: list[str], builder: Callable[[pd.DataFrame], np.ndarray],
             attrs: list[str], mems: list[str], controls: list[str]):
    """Each effect as (variable, kind, function of (F, beta) -> float). Counterfactual design
    matrices are built once here."""
    def cf(**set_):
        dd = d.copy()
        for k, v in set_.items():
            dd[k] = v
        return builder(dd)

    effs = []
    X1, X0 = cf(gain=1), cf(gain=0)
    effs.append(("gain", "AME on P(sell)", (X1, X0)))
    for a in attrs:
        if a not in names:
            continue
        effs.append((a, "AME on P(sell)", (cf(**{a: 1.0}), cf(**{a: 0.0}))))
        effs.append((a, "effect on DE", (cf(gain=1, **{a: 1.0}), cf(gain=0, **{a: 1.0}),
                                         cf(gain=1, **{a: 0.0}), cf(gain=0, **{a: 0.0}))))
    for m in mems:
        if f"mem_{m}" not in names:
            continue
        effs.append((f"mem_{m}", "AME on P(sell)",
                     (cf(architecture=m), cf(architecture="none"))))
        effs.append((f"mem_{m}", "effect on DE",
                     (cf(gain=1, architecture=m), cf(gain=0, architecture=m),
                      cf(gain=1, architecture="none"), cf(gain=0, architecture="none"))))
    for c in controls:
        if c not in names:
            continue
        h = 1e-4 * max(1e-8, float(d[c].std()))
        effs.append((c, "AME on P(sell)", (cf(**{c: d[c] + h}), cf(**{c: d[c].to_numpy()}),
                                           h)))
    return effs


def _value(F, beta, mats) -> float:
    if len(mats) == 3:                          # derivative of a continuous control
        a, b, h = mats
        return float(np.mean(F(a @ beta) - F(b @ beta)) / h)
    if len(mats) == 2:
        a, b = mats
        return float(np.mean(F(a @ beta) - F(b @ beta)))
    g1, g0, n1, n0 = mats
    return float(np.mean((F(g1 @ beta) - F(g0 @ beta)) - (F(n1 @ beta) - F(n0 @ beta))))


def marginal_effects(models: list[ModelFit], d: pd.DataFrame,
                     builder: Callable[[pd.DataFrame], np.ndarray], attrs: list[str],
                     mems: list[str], controls: list[str]) -> pd.DataFrame:
    names = models[0].fit.names
    effs = _effects(d, names, builder, attrs, mems, controls)
    rows = []
    grads: dict[tuple[str, int], np.ndarray] = {}
    for m in models:
        beta, V = m.fit.beta, m.fit.vcov
        q = stats.t.ppf(0.975, m.fit.dof)
        for i, (var, kind, mats) in enumerate(effs):
            key = (m.model, i)
            if key not in grads:                # gradient depends on beta only, not on V
                J = np.zeros(len(beta))
                for j in range(len(beta)):
                    e = np.zeros(len(beta))
                    e[j] = 1e-6 * max(1.0, abs(beta[j]))
                    J[j] = (_value(m.F, beta + e, mats) - _value(m.F, beta - e, mats)) / (2 * e[j])
                grads[key] = J
            J = grads[key]
            est = _value(m.F, beta, mats)
            se = float(np.sqrt(max(0.0, J @ V @ J)))
            p = float(2 * stats.t.sf(abs(est / se), m.fit.dof)) if se > 0 else np.nan
            rows.append({"model": m.model, "clustering": m.clustering, "effect": kind,
                         "variable": var, "estimate": est, "se": se, "p": p,
                         "ci_lo": est - q * se, "ci_hi": est + q * se,
                         "converged": m.converged})
    out = pd.DataFrame(rows)
    out["p_holm_attr"] = np.nan
    for _, g in out[(out["effect"] == "effect on DE")
                    & out["variable"].str.startswith("attr_")].groupby(["model", "clustering"]):
        out.loc[g.index, "p_holm_attr"] = holm(g["p"].to_numpy())
    return out


def robustness_summary(coefs: pd.DataFrame, models: list[ModelFit]) -> pd.DataFrame:
    """Per model x clustering: clusters, significant d_k after Holm, joint Wald tests."""
    rows = []
    for m in models:
        c = coefs[(coefs["model"] == m.model) & (coefs["clustering"] == m.clustering)]
        dk = [n for n in m.fit.names if n.startswith("gain:attr_")]
        fm = [n for n in m.fit.names if n.startswith("gain:mem_")]
        row = {"model": m.model, "clustering": m.clustering, "min_clusters": m.fit.g_min,
               "df": m.fit.dof, "converged": m.converged,
               "d_k significant (Holm)": int((c["p_holm_dk"] < 0.05).sum()) if dk else None}
        if dk:
            row["RQ1 joint p (all d_k = 0)"] = m.fit.wald(dk)["p"]
        if fm:
            row["RQ3 memory p (f_log = f_reflect = 0)"] = m.fit.wald(fm)["p"]
        rows.append(row)
    return pd.DataFrame(rows)
