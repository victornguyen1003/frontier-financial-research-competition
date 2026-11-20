import numpy as np
import pandas as pd
import pytest

from llm_investors.analysis import odds_ratio as orx


def test_paper_benchmark():
    p = orx.paper_benchmark().set_index("source")
    ind = p.loc["Barber et al. (2007), individual investors"]
    assert ind["PGR/PLR"] == pytest.approx(2.629, abs=1e-3)   # the number quoted in meeting 2
    assert ind["OR"] == pytest.approx(2.70, abs=0.01)
    assert ind["OR (Panel A amounts)"] == pytest.approx(2.79, abs=0.01)


def fake_decisions(seed=0, b_gain=-1.0, b_int=None, n_scen=40, rows=30):
    """Logit data with the decision-table columns: 2 personas x n_scen scenarios x 3 arms."""
    b_int = b_int or {"none": 0.0, "log": 0.5, "reflect": 0.8}
    rng = np.random.default_rng(seed)
    out = []
    for pers in ("young_active", "retiree"):
        for s in range(n_scen):
            u = rng.normal(0, 0.3)                       # matched-set effect
            for a in orx.ARCHS:
                gain = rng.integers(0, 2, rows)
                ret = rng.normal(0, 0.1, rows)
                xb = -0.5 + u + (b_gain + b_int[a]) * gain + 1.0 * ret
                out.append(pd.DataFrame({
                    "sell": (rng.random(rows) < 1 / (1 + np.exp(-xb))).astype(int),
                    "gain": gain.astype(float), "ret": ret,
                    "holding_period": rng.integers(1, 15, rows),
                    "drawdown": rng.uniform(0, 0.3, rows), "period": rng.integers(1, 16, rows),
                    "architecture": a, "persona": pers, "scenario": s,
                    "run_id": f"{a}_{pers}_s{s:03d}"}))
    return orx.prepare(pd.concat(out, ignore_index=True))


def test_odds_ratios_recover_truth():
    d = fake_decisions()
    res = orx.fit_logit(d, orx.SPECS["main"], "matched_set")
    r = pd.DataFrame(orx.odds_ratios(res, orx.ARCHS)).set_index("architecture")
    for a, true in (("none", -1.0), ("log", -0.5), ("reflect", -0.2)):
        assert r.loc[a, "ci_lo"] < np.exp(true) < r.loc[a, "ci_hi"]
    assert r.loc["log / none", "log_odds"] == pytest.approx(
        res.params[f"gain:{orx.ARCH}[T.log]"])


def test_ame_matches_statsmodels_margeff():
    # One arm, no interaction: our AME and delta-method SE should equal statsmodels'
    # discrete-change average marginal effect of the binary gain.
    d = fake_decisions(b_int={a: 0.0 for a in orx.ARCHS})
    dn = d[d["architecture"] == "none"]
    resn = orx.fit_logit(dn, "sell ~ gain + ret", "matched_set")
    me = resn.get_margeff(at="overall", dummy=True).summary_frame().loc["gain"]
    mine = pd.DataFrame(orx.marginal_effects(resn, dn, ["none"]))
    assert mine["estimate"].iat[0] == pytest.approx(me["dy/dx"], rel=1e-6)
    assert mine["se"].iat[0] == pytest.approx(me["Std. Err."], rel=1e-3)


def test_ame_difference_sign_and_clusters():
    d = fake_decisions()
    ors, ames, coefs, failed = orx.analyze(d, specs={"main": orx.SPECS["main"]})
    assert not failed
    n = ors.groupby("cluster")["n_clusters"].first()
    assert n["path"] == d["scenario"].nunique()
    assert n["matched_set"] == d["fe"].nunique()
    assert n["agent"] == d["run_id"].nunique()
    diff = ames[(ames["measure"] == "AME difference") & (ames["cluster"] == "path")]
    assert (diff["estimate"] > 0).all()                 # true interactions are positive
    # point estimates do not depend on the clustering
    est = ors.pivot_table(index="architecture", columns="cluster", values="estimate")
    assert np.allclose(est.min(axis=1), est.max(axis=1))


def test_raw_comparison_counts():
    d = fake_decisions()
    raw = orx.raw_comparison(d).set_index("source")
    r = raw.loc["LLM agents, none"]
    g = d[d["architecture"] == "none"]
    assert r["RG"] == ((g["sell"] == 1) & (g["gain"] == 1)).sum()
    assert r["PGR/PLR"] == pytest.approx(r["PGR"] / r["PLR"])
