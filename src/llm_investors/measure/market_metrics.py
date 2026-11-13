"""Study 2 market outcomes. Bubble measures follow Stöckl, Huber & Kirchler (2010):
RAD = mean_t |P_t - FV_t| / |mean_t FV_t|,  RD = mean_t (P_t - FV_t) / |mean_t FV_t|."""
from __future__ import annotations

import numpy as np


def bubble_measures(price, fv) -> dict:
    """RAD, RD; peak overpricing = max_t (P_t - FV_t) / mean FV (same scale as RAD/RD) and the
    period it occurs; crash = largest fall from a running price peak to a later price, as a
    fraction of that peak (maximum drawdown of the price path)."""
    p, f = np.asarray(price, float), np.asarray(fv, float)
    fbar = abs(f.mean())
    over = (p - f) / fbar
    running_peak = np.maximum.accumulate(p)
    return {"RAD": float(np.mean(np.abs(p - f)) / fbar), "RD": float(np.mean(p - f) / fbar),
            "peak_overpricing": float(over.max()), "peak_period": int(np.argmax(over)) + 1,
            "crash": float(np.max(1 - p / running_peak))}


def trading_measures(price, volume, total_shares: int) -> dict:
    p, v = np.asarray(price, float), np.asarray(volume, float)
    r = np.diff(np.log(p)) if len(p) > 1 else np.array([])
    out = {"volume_total": float(v.sum()), "turnover": float(v.sum() / total_shares),
           "volatility": float(r.std(ddof=1)) if len(r) > 2 else None,
           "ret_autocorr_1": _ac(r), "abs_ret_autocorr_1": _ac(np.abs(r)),
           "excess_kurtosis": _kurt(r)}
    return out


def _ac(x):
    x = np.asarray(x, float)
    if len(x) < 4 or np.std(x) == 0:
        return None
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _kurt(x):
    x = np.asarray(x, float)
    if len(x) < 4 or np.std(x) == 0:
        return None
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3)
