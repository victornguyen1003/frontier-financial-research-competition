"""Price calibration (plan §2.7): per-stock monthly σ, Student-t tails, random-walk tests.

Monthly log returns are the sum of TEJ's daily log returns (報酬率-Ln / ROI%-Ln) within the
calendar month. TEJ's returns are adjusted for dividends and splits, so ex-dates do not show
up as price drops, unlike returns from raw month-end closes. Month-end closes are still kept
for the real-against-simulated path figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from llm_investors.envs.exogenous import simulate_log_returns

MIN_MONTHS = 6            # plan: keep stocks with at least 6 monthly returns
MIN_DAYS_PER_MONTH = 10   # a stock-month counts only with this many daily returns (drops
                          # listing months and long suspensions, which would understate σ)
MIN_MARKET_DAYS = 15      # a calendar month counts only if the file has this many trading
                          # days in it (drops months the export covers only partly)


def filter_common_stocks(d: pd.DataFrame) -> pd.DataFrame:
    """TSE-listed rows with an industry, which drops funds and ETFs (industry "-")."""
    return d[(d["market"] == "TSE") & (d["industry"] != "-") & (d["industry"] != "")]


def market_days(d: pd.DataFrame) -> pd.Series:
    """Trading days per calendar month in the file, 0 for months missing inside its range."""
    md = d.groupby(d["date"].dt.to_period("M"))["date"].nunique()
    full = pd.period_range(md.index.min(), md.index.max(), freq="M")
    return md.reindex(full, fill_value=0)


def complete_months(d: pd.DataFrame) -> list:
    md = market_days(d)
    return sorted(md[md >= MIN_MARKET_DAYS].index)


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """One row per stock-month in a completely covered month: summed daily log return,
    trading days, month-end close."""
    d = d.dropna(subset=["ret_ln"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    d = d[d["month"].isin(complete_months(d))]
    g = d.groupby(["stock", "month"])
    m = pd.DataFrame({"ret": g["ret_ln"].sum(), "days": g["ret_ln"].size(),
                      "close": g["close"].last()}).reset_index()
    return m


def sigma_table(m: pd.DataFrame) -> pd.DataFrame:
    ok = m[m["days"] >= MIN_DAYS_PER_MONTH]
    s = ok.groupby("stock")["ret"].agg(["std", "mean", "size"])
    s = s[s["size"] >= MIN_MONTHS].rename(columns={"std": "sigma", "size": "n_months"})
    return s


def standardized(m: pd.DataFrame, keep) -> np.ndarray:
    ok = m[(m["days"] >= MIN_DAYS_PER_MONTH) & m["stock"].isin(keep)]
    z = ok.groupby("stock")["ret"].transform(lambda r: (r - r.mean()) / r.std(ddof=1))
    return z.dropna().to_numpy()


DF_GRID = [3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100]
DF_MAX = 100.0


def kurtosis_curve(n_months: np.ndarray, rng, reps: int = 200) -> dict[float, float]:
    """Mean pooled excess kurtosis of per-stock standardized t(df) draws, with each stock's
    own number of months, for every df in DF_GRID (the statistic does not depend on σ)."""
    out = {}
    for df in DF_GRID:
        ks = []
        for _ in range(reps):
            z = []
            for n in n_months:
                e = rng.standard_t(df, size=n)
                z.append((e - e.mean()) / e.std(ddof=1))
            ks.append(stats.kurtosis(np.concatenate(z), fisher=True))
        out[df] = float(np.mean(ks))
    return out


def fit_df(z: np.ndarray, n_months: np.ndarray, rng) -> dict:
    """Student-t degrees of freedom of the monthly shocks.

    With ~10 months per stock, standardizing by each stock's own sample sd truncates the
    tails (|z| <= (n-1)/sqrt(n)), so a direct MLE on the pooled z is biased toward a normal
    (on 1999 it returns df -> infinity). Instead df is chosen by simulated moments: the df
    whose simulated pooled standardized kurtosis, with the same months per stock, equals the
    empirical one (linear interpolation in 1/df). The MLE is recorded for reference. The
    generator needs df > 2; the fit is kept within [3, 100]."""
    target = float(stats.kurtosis(z, fisher=True))
    curve = kurtosis_curve(n_months, rng)
    dfs = np.array(DF_GRID, dtype=float)
    ks = np.array([curve[d] for d in DF_GRID])          # decreasing in df
    if target >= ks[0]:
        df, note = dfs[0], "empirical tails at least as heavy as df = 3; clamped to 3"
    elif target <= ks[-1]:
        df, note = DF_MAX, "no excess tails detected; df set to 100 (close to normal)"
    else:
        j = int(np.argmax(ks <= target))                  # first grid point below target
        x0, x1 = 1 / dfs[j - 1], 1 / dfs[j]
        w = (target - ks[j - 1]) / (ks[j] - ks[j - 1])
        df, note = 1 / (x0 + w * (x1 - x0)), "interpolated"
    mle_df, _, mle_scale = stats.t.fit(z, floc=0)
    return {"df": float(df), "method": "simulated moments on pooled standardized kurtosis",
            "target_excess_kurtosis": target,
            "curve": {str(k): v for k, v in curve.items()}, "fit_note": note,
            "df_mle_biased": float(mle_df), "scale_mle": float(mle_scale)}


def excess_kurtosis_standardized(r: np.ndarray) -> float:
    """Pooled excess kurtosis of per-series standardized returns; r is (series, months)."""
    z = (r - r.mean(axis=1, keepdims=True)) / r.std(axis=1, ddof=1, keepdims=True)
    return float(stats.kurtosis(z.ravel(), fisher=True))


# --------------------------------------------------------------------------- random walk
def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo & MacKinlay (1988) VR(q) with the heteroskedasticity-robust z*(q). r: 1-D returns."""
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4 * q:
        return np.nan, np.nan
    mu = r.mean()
    e = r - mu
    var1 = (e ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / var1
    theta = 0.0
    denom = ((e ** 2).sum()) ** 2
    for j in range(1, q):
        delta = n * ((e[j:] ** 2) * (e[:-j] ** 2)).sum() / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta / n) if theta > 0 else np.nan
    return float(vr), float(z)


def random_walk_tests(d: pd.DataFrame, keep) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    d = d[d["stock"].isin(keep)].dropna(subset=["ret_ln"])
    per_stock = []
    for s, g in d.groupby("stock"):
        r = g["ret_ln"].to_numpy()
        if len(r) < 60:
            continue
        ac = acf(r, nlags=5, fft=False)[1:]
        lb = acorr_ljungbox(r, lags=[5, 10], return_df=True)["lb_pvalue"].to_numpy()
        row = {"stock": s, **{f"ac{k}": float(ac[k - 1]) for k in range(1, 6)},
               "lb5_p": float(lb[0]), "lb10_p": float(lb[1])}
        for q in (2, 5, 10):
            row[f"vr{q}"], row[f"vr{q}_z"] = variance_ratio(r, q)
        per_stock.append(row)
    ps = pd.DataFrame(per_stock)
    # equal-weighted market index of the same stocks
    idx = d.groupby("date")["ret_ln"].mean().sort_index().to_numpy()
    idx_ac = acf(idx, nlags=5, fft=False)[1:]
    idx_lb = acorr_ljungbox(idx, lags=[5, 10], return_df=True)["lb_pvalue"].to_numpy()
    crit = stats.norm.ppf(0.975)
    out = {
        "n_stocks": int(len(ps)),
        "autocorrelation": {f"lag{k}": {"median": float(ps[f"ac{k}"].median()),
                                        "mean": float(ps[f"ac{k}"].mean()),
                                        "index": float(idx_ac[k - 1])} for k in range(1, 6)},
        "ljung_box": {f"lags{L}": {"share_p_below_0.05": float((ps[f"lb{L}_p"] < 0.05).mean()),
                                   "index_p": float(idx_lb[i])}
                      for i, L in enumerate((5, 10))},
        "variance_ratio": {},
        "note": ("Daily log returns. Per-stock statistics are summarized across stocks; "
                 "'index' is the equal-weighted average return of the same stocks. Under a "
                 "random walk, autocorrelations are near 0, VR(q) near 1 and about 5% of "
                 "stocks reject at the 5% level. Daily price limits and thin trading induce "
                 "some short-lag autocorrelation in TSE data."),
    }
    for q in (2, 5, 10):
        vr_i, z_i = variance_ratio(idx, q)
        out["variance_ratio"][f"q{q}"] = {
            "median_vr": float(ps[f"vr{q}"].median()),
            "share_reject_0.05": float((ps[f"vr{q}_z"].abs() > crit).mean()),
            "index_vr": vr_i, "index_z": z_i}
    return out


# --------------------------------------------------------------------------- main
def calibrate_prices(tej: pd.DataFrame, seed: int = 0) -> tuple[dict, dict]:
    """Return (prices.json content, extra arrays for the figures)."""
    all_rows = len(tej)
    d = filter_common_stocks(tej)
    m = monthly(d)
    sig = sigma_table(m)
    keep = set(sig.index)
    z = standardized(m, keep)
    rng = np.random.default_rng(seed)
    tfit = fit_df(z, sig["n_months"].to_numpy(), rng)
    sigmas = np.sort(sig["sigma"].to_numpy())
    n_months = int(sig["n_months"].median())
    emp_kurt = float(stats.kurtosis(z, fisher=True))
    ok = m[(m["days"] >= MIN_DAYS_PER_MONTH) & m["stock"].isin(keep)]
    raw_kurt = float(stats.kurtosis(ok["ret"], fisher=True))

    # Check the generator against the targets with the same estimator and sample length.
    sim, _ = simulate_log_returns(rng, sigmas, tfit["df"], n_stocks=20000, periods=n_months)
    sim_sig = sim.std(axis=0, ddof=1)
    sim_kurt = excess_kurtosis_standardized(sim.T)
    rw = random_walk_tests(d, keep)
    q = lambda a, p: float(np.percentile(a, p))
    prices = {
        "sigma_monthly": [round(float(s), 6) for s in sigmas],
        "sigma_summary": {"n": int(len(sigmas)), "median": q(sigmas, 50), "p25": q(sigmas, 25),
                          "p75": q(sigmas, 75), "min": float(sigmas.min()),
                          "max": float(sigmas.max())},
        "student_t": {**tfit, "pooled_excess_kurtosis_standardized": emp_kurt,
                      "pooled_excess_kurtosis_raw": raw_kurt,
                      "n_standardized_returns": int(len(z))},
        "simulated_check": {
            "n_series": 20000, "months_per_series": n_months,
            "sigma_median": q(sim_sig, 50), "sigma_p25": q(sim_sig, 25),
            "sigma_p75": q(sim_sig, 75),
            "pooled_excess_kurtosis_standardized": sim_kurt,
            "pooled_excess_kurtosis_raw": float(stats.kurtosis(sim.ravel(), fisher=True)),
            "note": "Same estimator and series length as the empirical figures."},
        "random_walk_tests": rw,
        "coverage": {
            "first_date": str(tej["date"].min().date()), "last_date": str(tej["date"].max().date()),
            "rows_total": int(all_rows), "rows_tse_common": int(len(d)),
            "stocks_total": int(tej["stock"].nunique()),
            "stocks_tse_common": int(d["stock"].nunique()),
            "stocks_used": int(len(keep)),
            "months_used": sorted({str(p) for p in m["month"].unique()}),
            "trading_days_per_month_in_file": {str(k): int(v) for k, v
                                               in market_days(d).items()},
            "months_excluded_incomplete": [str(k) for k, v in market_days(d).items()
                                           if v < MIN_MARKET_DAYS],
            "median_months_per_stock": n_months,
            # the LIMIT column holds "+" on limit days and "-" otherwise in this export
            "share_stock_days_at_limit": float((d["limit"] == "+").mean()),
            "daily_sigma_median": float(d.groupby("stock")["ret_ln"].std().median()),
            "filters": (f"Market == TSE and industry != '-' (drops funds/ETFs); calendar "
                        f"months with at least {MIN_MARKET_DAYS} trading days in the file; "
                        f"stock-months with at least {MIN_DAYS_PER_MONTH} daily returns; "
                        f"stocks with at least {MIN_MONTHS} such months"),
            "monthly_return": "sum of TEJ daily log returns (dividend- and split-adjusted)",
        },
        "generator": ("ln P(t+1) = ln P(t) + sigma_i * eps_t; sigma_i drawn per stock from "
                      "sigma_monthly; eps_t Student-t(df) scaled to unit variance; zero drift"),
    }
    # month-end closes for the path figure (rebased later)
    closes = (m[m["stock"].isin(keep)].pivot(index="month", columns="stock", values="close")
              .sort_index())
    extra = {"z": z, "sim_returns": sim[:, :2000], "closes": closes, "sigmas": sigmas,
             "sim_sigmas": sim_sig}
    return prices, extra
