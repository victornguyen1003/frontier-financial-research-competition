"""Power analysis for the main run (plan §2.5): `power-study1 <pilot dataset>` (free, no API).

1. Fit to the pilot: a random-effects LPM of decision-level `sold` with crossed path and
   profile effects (intercepts and gain slopes), gain and gain x memory terms. C1 is left out
   (it is the manipulation). The pilot's 8 profiles cannot identify 11 attribute effects, so
   attribute differences stay inside the profile variance: a conservative (larger) residual.
   Recorded: C0's sell rates on gains and losses, the variance components, the mean number of
   sale days per run and positions per sale day.
2. Simulate: for each (P, S), n_sims datasets with the real profile draws (config seed, so
   the simulated profiles are the ones the main run will use), every attribute effect δ on
   PGR − PLR, both memory effects δ, and C1 − C0 at the pilot's estimate. The data rows
   (gain, ret, holding period, drawdown) are resampled from the pilot's sale-day blocks.
3. Fit the §5.2 regression (same estimator as analyze-study1) and record power for each d_k
   (Holm across the 11), the joint Wald test, the memory Wald test and C1 against C0.
4. Recommend the cheapest (P, S) whose every d_k has >= 80% power and whose projected cost is
   within budget.stage_limits.full; otherwise the largest affordable design.
5. Write results/<pilot>/power/{power.csv, mde.csv, recommendation.json, power_curves.png}.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from llm_investors.agents.personas import ATTRIBUTE_KEYS, sample_profiles
from llm_investors.analysis import style
from llm_investors.analysis.ols import holm, mde, ols
from llm_investors.analysis.study1 import ALPHA, MEMS, design_matrix, load
from llm_investors.paths import ROOT
from llm_investors.sim.common import load_config

ARCHS = ["none", "log", "reflect"]
TESTS = ["d_k (Holm, min over k)", "d_k (Holm, mean over k)", "joint Wald d_k = 0",
         "memory Wald", "C1 vs C0"]


# --------------------------------------------------------------------------- 1. pilot fit
def fit_pilot(runs: pd.DataFrame, dec: pd.DataFrame) -> dict:
    d = dec.dropna(subset=["gain"]).copy()
    d["gain"] = d["gain"].astype(int)
    d["sold"] = d["sell"].astype(int)
    c0 = d[d["persona"] == "C0"]
    base = d if c0.empty else c0
    rate_gain = float(base.loc[base["gain"] == 1, "sold"].mean())
    rate_loss = float(base.loc[base["gain"] == 0, "sold"].mean())
    fitd = d[d["persona_kind"] != "C1"].copy()
    fitd["path"] = fitd["scenario"].astype(str)
    for m in MEMS:
        fitd[f"mem_{m}"] = (fitd["architecture"] == m).astype(float)
    # Variance components by the ANOVA (moments) method: robust with a pilot's few paths and
    # profiles. Crossed-effects MixedLM is recorded as a cross-check only: on pilot-sized data
    # it can report convergence with implausible components (e.g. a profile sd of 0.33).
    comps = _moments(fitd)
    method = ("ANOVA moments: between-path / between-profile variance of the loss sell rate "
              "and of the gain gap, net of binomial sampling variance")
    check = None
    try:
        import statsmodels.formula.api as smf
        vc = {"path": "0 + C(path)", "path_gain": "0 + C(path):gain",
              "profile": "0 + C(persona)", "profile_gain": "0 + C(persona):gain"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm("sold ~ gain + mem_log + mem_reflect + gain:mem_log + "
                              "gain:mem_reflect", fitd, groups=np.ones(len(fitd)),
                              vc_formula=vc).fit(reml=True)
        check = {"converged": bool(res.converged), "within_run": float(res.scale),
                 **{k: float(v) for k, v in zip(res.model.exog_vc.names, res.vcomp)}}
    except Exception as e:  # noqa: BLE001
        check = {"error": f"{type(e).__name__}: {e}"}
    # C1 - C0 difference in the gain gap (for the manipulation-check power)
    c1 = d[d["persona"] == "C1"]
    gap = lambda x: x.loc[x["gain"] == 1, "sold"].mean() - x.loc[x["gain"] == 0, "sold"].mean()
    c1_effect = float(gap(c1) - gap(base)) if not c1.empty and not c0.empty else None
    sale_days = runs["sale_days"] if "sale_days" in runs else \
        dec.groupby("run_id")["period"].nunique()
    blocks = d.groupby(["run_id", "period"]).size()
    return {"c0_sell_rate_gain": rate_gain, "c0_sell_rate_loss": rate_loss,
            "variance_components": comps, "variance_method": method,
            "mixedlm_crosscheck": check,
            "mean_sale_days_per_run": float(sale_days.mean()),
            "mean_positions_per_sale_day": float(blocks.mean()),
            "share_gain_rows": float(d["gain"].mean()),
            "c1_minus_c0_gain_gap": c1_effect, "n_rows": int(len(d)),
            "n_runs": int(runs["run_id"].nunique())}


def _moments(d: pd.DataFrame) -> dict:
    """Fallback: between-group variance of the loss sell rate (intercept) and of the gain gap
    (slope) by path and by profile, net of their binomial sampling variance."""
    out = {}
    for name, col in (("path", "path"), ("profile", "persona")):
        c = d.groupby([col, "gain"])["sold"].agg(["mean", "size"]).unstack()
        if 1 not in c["mean"] or 0 not in c["mean"] or len(c) < 2:
            out[name] = out[f"{name}_gain"] = 0.0
            continue
        p0, n0 = c["mean"][0], c["size"][0]
        p1, n1 = c["mean"][1], c["size"][1]
        sv0 = (p0 * (1 - p0) / n0).mean()
        sv1 = (p1 * (1 - p1) / n1).mean()
        out[name] = float(max(p0.var(ddof=1) - sv0, 0.0))
        out[f"{name}_gain"] = float(max((p1 - p0).var(ddof=1) - sv0 - sv1, 0.0))
    out["within_run"] = float((d["sold"] - d.groupby("gain")["sold"].transform("mean")).var())
    return out


# --------------------------------------------------------------------------- 2-3. simulate
def _blocks(dec: pd.DataFrame) -> dict:
    """The pilot's sale-day blocks (all positions of one run on one sale day), stacked."""
    d = dec.dropna(subset=["gain"]).sort_values(["run_id", "period"]).copy()
    d["gain"] = d["gain"].astype(int)
    rows = d[["gain", "ret", "holding_period", "drawdown"]].to_numpy(float)
    sizes = d.groupby(["run_id", "period"], sort=True).size().to_numpy()
    return {"rows": rows, "start": np.concatenate([[0], np.cumsum(sizes)[:-1]]), "len": sizes}


def simulate(P: int, S: int, fitp: dict, blocks: dict, profiles, delta: float, rng,
             sale_days: float) -> pd.DataFrame:
    vc = fitp["variance_components"]
    sd = {k: np.sqrt(vc.get(k, 0.0)) for k in ("path", "path_gain", "profile")}
    sd["profile_gain"] = np.sqrt(fitp["profile_gain_residual"])
    a = fitp["c0_sell_rate_loss"]
    b = fitp["c0_sell_rate_gain"] - a
    c1 = fitp["c1_minus_c0_gain_gap"]
    c1 = c1 if c1 is not None and c1 > 0 else delta
    attrs = np.array([[pr[k] for k in ATTRIBUTE_KEYS] for pr in profiles]
                     + [[0] * len(ATTRIBUTE_KEYS)] * 2, float)          # (P + 2) x 11
    # centered so the average gain gap stays at C0's (no clipping drift)
    gap_person = np.r_[delta * (attrs[:P] - 0.5).sum(axis=1), 0.0, c1]
    u_path, v_path = rng.normal(0, sd["path"], S), rng.normal(0, sd["path_gain"], S)
    u_prof, v_prof = rng.normal(0, sd["profile"], P + 2), rng.normal(0, sd["profile_gain"], P + 2)
    # one entry per run: scenario-major grid of path x persona x arm
    run_s = np.repeat(np.arange(S), (P + 2) * 3)
    run_j = np.tile(np.repeat(np.arange(P + 2), 3), S)
    run_a = np.tile(np.arange(3), S * (P + 2))
    n_days = rng.poisson(sale_days, size=len(run_s))
    day_run = np.repeat(np.arange(len(run_s)), n_days)
    choice = rng.integers(len(blocks["len"]), size=len(day_run))
    lens = blocks["len"][choice]
    row_run = np.repeat(day_run, lens)
    first = np.repeat(np.cumsum(lens) - lens, lens)
    src = np.repeat(blocks["start"][choice], lens) + (np.arange(lens.sum()) - first)
    X = blocks["rows"][src]
    g = X[:, 0]
    s_, j_, a_ = run_s[row_run], run_j[row_run], run_a[row_run]
    mem = delta * ((a_ != 0) - 2 / 3)
    p = a + u_path[s_] + u_prof[j_] + g * (b + v_path[s_] + v_prof[j_] + gap_person[j_] + mem)
    sold = (rng.random(len(g)) < np.clip(p, 0.001, 0.999)).astype(int)
    df = pd.DataFrame(X, columns=["gain", "ret", "holding_period", "drawdown"])
    df["gain"] = df["gain"].astype(int)
    df["scenario"], df["pid"], df["sold"] = s_, j_, sold
    df["architecture"] = np.array(ARCHS)[a_]
    df[[f"attr_{k}" for k in ATTRIBUTE_KEYS]] = attrs[j_]
    df["kind"] = np.where(j_ < P, "P", np.where(j_ == P, "C0", "C1"))
    return df


def _path_de(df: pd.DataFrame, pid: int) -> np.ndarray:
    g = df[df["pid"] == pid]
    c = g.groupby(["scenario", "gain"])["sold"].agg(["sum", "size"]).unstack()
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = c["sum"] / c["size"]
    return (rate.get(1) - rate.get(0)).to_numpy() if 1 in rate and 0 in rate else np.array([])


def one_sim(P, S, fitp, blocks, profiles, delta, rng, sale_days) -> dict:
    df = simulate(P, S, fitp, blocks, profiles, delta, rng, sale_days)
    d = df[df["kind"] == "P"]
    X, names = design_matrix(d)
    keep = X.std(axis=0) > 0
    keep[0] = True
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    fit = ols(d["sold"].to_numpy(float), X, names, [d["scenario"].to_numpy(),
                                                    d["pid"].to_numpy()])
    dk = [n for n in fit.names if n.startswith("gain:attr_")]
    fm = [n for n in fit.names if n.startswith("gain:mem_")]
    i = fit.idx(dk)
    p_holm = holm(fit.pvalues()[i])
    rej = p_holm < ALPHA
    c1 = _path_de(df, P + 1) - _path_de(df, P)
    c1 = c1[np.isfinite(c1)]
    try:
        p_c1 = stats.wilcoxon(c1, alternative="greater").pvalue if len(c1) > 1 and (c1 != 0).any() \
            else 1.0
    except ValueError:
        p_c1 = 1.0
    return {"rej_dk": rej, "n_dk": len(dk), "se_dk": float(fit.se[i].mean()),
            "joint": fit.wald(dk)["p"] < ALPHA, "memory": fit.wald(fm)["p"] < ALPHA,
            "c1": p_c1 < ALPHA, "se_mem": float(fit.se[fit.idx(fm)].mean()), "dof": fit.dof,
            "rows": len(d)}


# --------------------------------------------------------------------------- 4-5. outputs
def cost_per_run(name: str, runs: pd.DataFrame) -> float | None:
    proj = ROOT / "results" / name / "pilot_projection.json"
    if proj.exists():
        p = json.loads(proj.read_text())
        if p.get("cost_per_run_usd"):
            return float(p["cost_per_run_usd"])
    c = runs["cost_usd"].mean() if "cost_usd" in runs else None
    return float(c) if c and c > 0 else None


def recommend(pw: pd.DataFrame, cost: pd.DataFrame, limit: float | None) -> dict:
    t = pw.pivot(index=["P", "S"], columns="test", values="power").join(
        cost.set_index(["P", "S"]))
    t["affordable"] = True if limit is None or t["cost_usd"].isna().all() else \
        t["cost_usd"] <= limit
    ok = t[(t["d_k (Holm, min over k)"] >= 0.8) & t["affordable"]]
    if not ok.empty:
        best = ok.sort_values(["cost_usd", "runs"]).index[0]
        why = "cheapest design with >= 80% power for every d_k within the main-run budget"
    else:
        aff = t[t["affordable"]]
        best = (aff if not aff.empty else t).sort_values("runs").index[-1]
        why = ("no design reaches 80% power for every d_k within budget: largest "
               "affordable design" if not aff.empty else
               "no design is affordable at the measured cost: largest design shown")
    r = t.loc[best]
    return {"P": int(best[0]), "S": int(best[1]), "reason": why,
            "runs": int(r["runs"]), "projected_cost_usd": None if pd.isna(r["cost_usd"])
            else float(r["cost_usd"]), "budget_usd": limit,
            "power": {k: float(r[k]) for k in TESTS}}


def fig_power(pw: pd.DataFrame, mdes: pd.DataFrame, rec: dict, path):
    import matplotlib.pyplot as plt
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, test in zip(axes[:2], ["d_k (Holm, min over k)", "memory Wald"]):
        for i, (P, g) in enumerate(pw[pw["test"] == test].groupby("P")):
            ax.plot(g["S"], g["power"], marker="o", color=style.SERIES[i % 3], label=f"P = {P}")
        ax.axhline(0.8, color=style.INK_2, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("paths S")
        ax.set_ylabel("power")
        ax.set_title(f"Power: {test}", loc="left", fontsize=10)
        ax.legend(fontsize=8)
    ax = axes[2]
    for i, (P, g) in enumerate(mdes.groupby("P")):
        ax.plot(g["S"], g["mde_dk_holm"], marker="o", color=style.SERIES[i % 3], label=f"P = {P}")
    ax.axhline(0.05, color=style.INK_2, linestyle="--", linewidth=1)
    ax.axhline(0.03, color=style.GRID, linestyle=":", linewidth=1)
    ax.scatter([rec["S"]], [mdes.set_index(["P", "S"]).loc[(rec["P"], rec["S"]), "mde_dk_holm"]],
               s=120, facecolor="none", edgecolor=style.SELL, linewidth=2, zorder=4,
               label="recommended")
    ax.set_xlabel("paths S")
    ax.set_ylabel("MDE of d_k (Holm, 80% power)")
    ax.set_title("Minimum detectable effect", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(name: str, n_sims: int | None = None, quick: bool = False):
    cfg = load_config("study1")
    pc = cfg["power"]
    runs, dec, _, _ = load(name)
    if dec.empty:
        raise SystemExit(f"{name}: no decision rows (no sales in the pilot?)")
    fitp = fit_pilot(runs, dec)
    # The pilot's profiles cannot separate attribute effects from other profile-level
    # differences in the gain gap. The simulation adds attribute effects of size delta, whose
    # variance across balanced profiles is n_attr * delta^2 / 4; only the rest is residual.
    attr_var = len(ATTRIBUTE_KEYS) * pc["delta"] ** 2 / 4
    fitp["profile_gain_attribute_part"] = attr_var
    fitp["profile_gain_residual"] = max(0.0, fitp["variance_components"]["profile_gain"]
                                        - attr_var)
    blocks = _blocks(dec[dec["persona_kind"] != "C1"])
    n_sims = n_sims or pc["n_sims"]
    Ps, Ss = pc["profiles"], pc["scenarios"]
    if quick:
        Ps, Ss = Ps[:2], Ss[:2]
    delta = pc["delta"]
    rng = np.random.default_rng(pc["seed"])
    cpr = cost_per_run(name, runs)
    limit = (cfg["budget"].get("stage_limits") or {}).get("full")
    all_prof = sample_profiles(max(Ps), cfg["profiles"]["seed"])
    rows, mrows, crows = [], [], []
    for P in Ps:
        for S in Ss:
            res = [one_sim(P, S, fitp, blocks, all_prof[:P], delta, rng,
                           fitp["mean_sale_days_per_run"]) for _ in range(n_sims)]
            rej = np.array([r["rej_dk"] for r in res])            # sims x k
            per_k = rej.mean(axis=0)
            power = {"d_k (Holm, min over k)": float(per_k.min()),
                     "d_k (Holm, mean over k)": float(per_k.mean()),
                     "joint Wald d_k = 0": float(np.mean([r["joint"] for r in res])),
                     "memory Wald": float(np.mean([r["memory"] for r in res])),
                     "C1 vs C0": float(np.mean([r["c1"] for r in res]))}
            for k, v in power.items():
                rows.append({"P": P, "S": S, "test": k, "power": v, "n_sims": n_sims})
            se = float(np.mean([r["se_dk"] for r in res]))
            dof = res[0]["dof"]
            mrows.append({"P": P, "S": S, "se_dk": se,
                          "mde_dk_holm": mde(se, ALPHA / len(ATTRIBUTE_KEYS), dof=dof),
                          "mde_dk_unadjusted": mde(se, ALPHA, dof=dof),
                          "mde_memory": mde(float(np.mean([r["se_mem"] for r in res])), ALPHA,
                                            dof=dof),
                          "dof": dof, "mean_rows": float(np.mean([r["rows"] for r in res]))})
            n_runs = (P + 2) * S * len(ARCHS)
            crows.append({"P": P, "S": S, "runs": n_runs,
                          "cost_usd": n_runs * cpr if cpr else None})
            print(f"P={P:>2} S={S:>2}: power d_k(min) {power['d_k (Holm, min over k)']:.2f} "
                  f"(mean {power['d_k (Holm, mean over k)']:.2f}), joint "
                  f"{power['joint Wald d_k = 0']:.2f}, memory {power['memory Wald']:.2f}, "
                  f"C1 {power['C1 vs C0']:.2f}; MDE {mrows[-1]['mde_dk_holm']:.3f}; "
                  f"{n_runs} runs" + (f" ≈ ${crows[-1]['cost_usd']:.2f}" if cpr else ""),
                  flush=True)
    pw, mdes, cost = pd.DataFrame(rows), pd.DataFrame(mrows), pd.DataFrame(crows)
    rec = recommend(pw, cost, limit)
    rec |= {"pilot": name, "pilot_fit": fitp, "delta": delta, "n_sims": n_sims,
            "cost_per_run_usd": cpr, "mde_dk_holm": float(
                mdes.set_index(["P", "S"]).loc[(rec["P"], rec["S"]), "mde_dk_holm"])}
    out = ROOT / "results" / name / "power"
    out.mkdir(parents=True, exist_ok=True)
    pw.to_csv(out / "power.csv", index=False)
    mdes.merge(cost, on=["P", "S"]).to_csv(out / "mde.csv", index=False)
    (out / "recommendation.json").write_text(json.dumps(rec, indent=2, default=float))
    fig_power(pw, mdes, rec, out / "power_curves.png")
    print(f"\nRecommendation: P = {rec['P']}, S = {rec['S']} ({rec['reason']}); "
          f"{rec['runs']} runs" + (f" ≈ ${rec['projected_cost_usd']:.2f}"
                                   if rec["projected_cost_usd"] else "")
          + f"; MDE {rec['mde_dk_holm']:.3f}")
    print(f"-> {out}")


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="pilot dataset, e.g. study1_v3_gpt-6-luna_pilot")
    ap.add_argument("--sims", type=int, help="override power.n_sims")
    ap.add_argument("--quick", action="store_true", help="2 x 2 designs (pipeline check)")
    a = ap.parse_args()
    main(a.dataset, a.sims, a.quick)


if __name__ == "__main__":
    cli()
