"""`calibrate`: write results/calibration/{prices,attributes}.json and the figures.

    calibrate                 # reads calibration.* paths from config/study1.yaml

The raw TEJ file is licensed and git-ignored; only these derived files are committed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from llm_investors.analysis import style
from llm_investors.paths import ROOT

from . import attributes, prices, sources

OUT = ROOT / "results" / "calibration"


def _figures(p: dict, extra: dict, seed: int):
    import matplotlib.pyplot as plt
    from scipy import stats
    style.apply()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    blue, orange = style.SERIES[0], style.SERIES[1]

    # 1. sigma histogram
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(extra["sigmas"], bins=40, color=blue, zorder=2)
    med = p["sigma_summary"]["median"]
    ax.axvline(med, color=style.INK_2, linewidth=1)
    ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {med:.3f}", color=style.INK_2, va="top")
    ax.set_xlabel("monthly log-return σ per stock")
    ax.set_ylabel("stocks")
    ax.set_title(f"σ of {p['sigma_summary']['n']} TSE common stocks", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "sigma_hist.png")
    plt.close(fig)

    # 2. standardized return histogram + QQ, empirical against simulated
    z = extra["z"]
    sim = extra["sim_returns"]
    zs = ((sim - sim.mean(axis=0)) / sim.std(axis=0, ddof=1)).ravel()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.linspace(-4, 4, 41)
    a1.hist(z, bins=bins, density=True, color=blue, alpha=0.75, label="TEJ", zorder=2)
    a1.hist(zs, bins=bins, density=True, histtype="step", color=orange, linewidth=2,
            label="simulated", zorder=3)
    xs = np.linspace(-4, 4, 200)
    a1.plot(xs, stats.norm.pdf(xs), color=style.INK_2, linewidth=1, linestyle="--",
            label="normal")
    a1.set_title("Standardized monthly returns", loc="left")
    a1.legend()
    qs = np.linspace(0.005, 0.995, 199)
    a2.plot(stats.norm.ppf(qs), np.quantile(z, qs), "o", ms=3, color=blue, label="TEJ")
    a2.plot(stats.norm.ppf(qs), np.quantile(zs, qs), "o", ms=3, color=orange,
            label="simulated")
    lim = [-3, 3]
    a2.plot(lim, lim, color=style.INK_2, linewidth=1, linestyle="--")
    a2.set_xlabel("normal quantile")
    a2.set_ylabel("sample quantile")
    a2.set_title(f"QQ plot (Student-t, df = {p['student_t']['df']:.1f})", loc="left")
    a2.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "returns_qq.png")
    plt.close(fig)

    # 3. real against simulated paths, rebased to 100
    closes = extra["closes"].dropna(axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(closes.columns, size=min(25, closes.shape[1]), replace=False)
    real = closes[pick].to_numpy()
    real = 100 * real / real[0]
    T = real.shape[0] - 1
    from llm_investors.envs.exogenous import simulate_log_returns
    r, _ = simulate_log_returns(rng, p["sigma_monthly"], p["student_t"]["df"], 25, T)
    simp = 100 * np.exp(np.vstack([np.zeros(25), np.cumsum(r, axis=0)]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, data, title, c in ((axes[0], real, "TEJ month-end closes", blue),
                               (axes[1], simp, "Simulated (calibrated)", orange)):
        ax.plot(range(T + 1), data, color=c, alpha=0.55, linewidth=1.2)
        ax.axhline(100, color=style.INK_2, linewidth=1)
        ax.set_title(f"{title}, 25 paths", loc="left")
        ax.set_xlabel("month")
    axes[0].set_ylabel("price (start = 100)")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_dir / "paths_real_vs_sim.png")
    plt.close(fig)


def main():
    from llm_investors.sim.common import load_config
    cfg = load_config("study1")
    cal = cfg["calibration"]
    seed = int(cal.get("seed", 0))
    OUT.mkdir(parents=True, exist_ok=True)
    tej = sources.load_tej(cal["tej_path"])
    p, extra = prices.calibrate_prices(tej, seed)
    (OUT / "prices.json").write_text(json.dumps(p, indent=2, ensure_ascii=False))
    a = attributes.calibrate_attributes(cal)
    (OUT / "attributes.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
    _figures(p, extra, seed)
    s, t, c = p["sigma_summary"], p["student_t"], p["coverage"]
    print(f"TEJ {c['first_date']} .. {c['last_date']}: {c['stocks_used']} stocks used "
          f"({c['stocks_tse_common']} TSE common, {c['stocks_total']} in file)")
    print(f"monthly σ: median {s['median']:.3f}, IQR {s['p25']:.3f}-{s['p75']:.3f}")
    print(f"Student-t df {t['df']:.2f} ({t['fit_note']}); pooled excess kurtosis of "
          f"standardized returns {t['pooled_excess_kurtosis_standardized']:.2f} (simulated "
          f"{p['simulated_check']['pooled_excess_kurtosis_standardized']:.2f}); raw "
          f"{t['pooled_excess_kurtosis_raw']:.2f} (simulated "
          f"{p['simulated_check']['pooled_excess_kurtosis_raw']:.2f})")
    if c["months_excluded_incomplete"]:
        print(f"WARNING: months only partly covered by the TEJ file, excluded: "
              f"{c['months_excluded_incomplete']} (days in file: "
              f"{c['trading_days_per_month_in_file']})")
    print(f"ages: younger {a['younger_age']} ({a['younger_age_exact']}), older "
          f"{a['older_age']} ({a['older_age_exact']})")
    if not a["tdcc_age_distribution"]["source_url"]:
        print("NOTE: calibration.tdcc_source_url is empty; add the TDCC page URL to config.")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:])
