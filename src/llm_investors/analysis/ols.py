"""OLS with multiway cluster-robust standard errors (Cameron, Gelbach & Miller 2011).

Used by the §5.2 regression and by the power simulation, so both use the same estimator.
Two-way clustering by path and by profile: V = V_path + V_profile - V_path×profile, each with
the small-sample factor G/(G-1) * (N-1)/(N-K). If V is not positive semi-definite, negative
eigenvalues are set to zero (CGM's fix). Inference uses t / F with G_min - 1 denominator
degrees of freedom, where G_min is the smaller number of clusters (few paths in a pilot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Fit:
    names: list[str]
    beta: np.ndarray
    vcov: np.ndarray
    n: int
    g_min: int

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0, None))

    @property
    def dof(self) -> int:
        return max(1, self.g_min - 1)

    def pvalues(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.beta / self.se
        return 2 * stats.t.sf(np.abs(t), self.dof)

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        q = stats.t.ppf(0.5 + level / 2, self.dof)
        return self.beta - q * self.se, self.beta + q * self.se

    def idx(self, names: list[str]) -> list[int]:
        return [self.names.index(n) for n in names]

    def wald(self, names: list[str]) -> dict:
        """H0: all named coefficients are 0. F(q, G_min - 1) and the chi-square p-value."""
        i = self.idx(names)
        b, V = self.beta[i], self.vcov[np.ix_(i, i)]
        w = float(b @ np.linalg.pinv(V) @ b)
        q = len(i)
        return {"chi2": w, "q": q, "p_chi2": float(stats.chi2.sf(w, q)),
                "F": w / q, "df_denom": self.dof, "p": float(stats.f.sf(w / q, q, self.dof))}


def _meat(X: np.ndarray, u: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, inv, X * u[:, None])
    return S.T @ S, G


def cluster_vcov(bread: np.ndarray, X: np.ndarray, u: np.ndarray, clusters: list[np.ndarray],
                 adj: float = 1.0) -> tuple[np.ndarray, int]:
    """Sandwich bread @ meat @ bread with one-way or two-way (CGM) clustering. Scores are
    X * u (OLS residuals, or y - p for a logit). Each term gets G/(G-1) * adj. Returns the
    PSD-corrected covariance and the smallest number of clusters."""
    k = X.shape[1]
    terms = [(c, 1) for c in clusters]
    if len(clusters) == 2:
        a, b = (np.asarray(c).astype(str) for c in clusters)
        terms.append((np.char.add(np.char.add(a, "|"), b), -1))
    V = np.zeros((k, k))
    gs = []
    for g, sign in terms:
        M, G = _meat(X, u, np.asarray(g))
        if sign > 0:
            gs.append(G)
        V += sign * (G / max(1, G - 1)) * adj * (bread @ M @ bread)
    w, Q = np.linalg.eigh((V + V.T) / 2)
    if (w < 0).any():
        V = Q @ np.diag(np.clip(w, 0, None)) @ Q.T
    return V, min(gs)


def ols(y: np.ndarray, X: np.ndarray, names: list[str], clusters: list[np.ndarray]) -> Fit:
    """clusters: one or two arrays of group labels (two-way adds their intersection)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    V, g_min = cluster_vcov(XtX_inv, X, u, clusters, adj=(n - 1) / max(1, n - k))
    return Fit(list(names), beta, V, n, g_min)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mde(se: float, alpha: float, power: float = 0.8, dof: int | None = None) -> float:
    """Minimum detectable effect for a two-sided test at `alpha` with the given power."""
    if dof:
        return float((stats.t.ppf(1 - alpha / 2, dof) + stats.t.ppf(power, dof)) * se)
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
