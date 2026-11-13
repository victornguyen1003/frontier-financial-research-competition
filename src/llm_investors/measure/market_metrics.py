"""Study 2 market outcomes. Bubble measures follow Stöckl, Huber & Kirchler (2010):
RAD = mean_t |P_t - FV_t| / |mean_t FV_t|,  RD = mean_t (P_t - FV_t) / |mean_t FV_t|,
with t over the periods in which shares traded (see bubble_measures)."""
from __future__ import annotations

import numpy as np


NAN = float("nan")


def _traded_mask(n: int, traded) -> np.ndarray:
    return np.ones(n, bool) if traded is None else np.asarray(traded, bool)


def bubble_measures(price, fv, traded=None) -> dict:
    """RAD, RD; peak overpricing = max_t (P_t - FV_t) / mean FV (same scale as RAD/RD) and the
    period it occurs; crash = largest fall from a running price peak to a later price, as a
    fraction of that peak (maximum drawdown of the price path).

    Only periods with a trade count (`traded`: one bool per period; None = all traded). In a
    no-trade period the logged price is the previous price carried forward, which nobody paid,
    so it is not evidence of mispricing: a frozen price above a falling FV would otherwise read
    as a large bubble. mean FV is taken over the same traded periods. With no traded period at
    all, every measure is NaN. traded_periods and no_trade_share are reported alongside."""
    p, f = np.asarray(price, float), np.asarray(fv, float)
    m = _traded_mask(len(p), traded)
    out = {"traded_periods": int(m.sum()),
           "no_trade_share": float(1 - m.mean()) if len(m) else NAN}
    if not m.any():
        return {**out, "RAD": NAN, "RD": NAN, "peak_overpricing": NAN, "peak_period": None,
                "crash": NAN}
    periods = np.arange(1, len(p) + 1)[m]
    p, f = p[m], f[m]
    fbar = abs(f.mean())
    over = (p - f) / fbar
    running_peak = np.maximum.accumulate(p)
    return {**out, "RAD": float(np.mean(np.abs(p - f)) / fbar), "RD": float(np.mean(p - f) / fbar),
            "peak_overpricing": float(over.max()), "peak_period": int(periods[np.argmax(over)]),
            "crash": float(np.max(1 - p / running_peak))}


def trading_measures(price, volume, total_shares: int, traded=None) -> dict:
    """Volume and turnover use every period (a no-trade period adds 0). Returns are log changes
    between consecutive traded prices, skipping no-trade periods, so a carried-forward price
    does not add zero returns that would understate volatility."""
    p, v = np.asarray(price, float), np.asarray(volume, float)
    p = p[_traded_mask(len(p), traded)]
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
