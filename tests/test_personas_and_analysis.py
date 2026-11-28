"""Profiles (plan §2.2), calibration (§2.7), power (§2.5), tagging (§4.2) and stage limits."""
import json

import numpy as np
import pandas as pd
import pytest

from llm_investors.agents import personas as PZ
from llm_investors.paths import ROOT

AGES = {"younger_age": 30, "older_age": 59}


# --------------------------------------------------------------------------- personas
def test_sampling_is_deterministic_prefix_stable_and_balanced():
    a = PZ.sample_profiles(30, 7)
    assert a == PZ.sample_profiles(30, 7)
    assert a[:8] == PZ.sample_profiles(8, 7)              # pilot profiles = main run's first 8
    assert a != PZ.sample_profiles(30, 8)
    big = pd.DataFrame(PZ.sample_profiles(4000, 1))
    assert list(big.columns) == PZ.ATTRIBUTE_KEYS
    assert ((big.mean() - 0.5).abs() < 0.03).all()        # each attribute drawn with p = 1/2
    assert (big.corr().abs().to_numpy()[np.triu_indices(11, 1)] < 0.06).all()  # independent


def test_compose_persona_order_and_levels():
    c0 = PZ.compose_persona(PZ.control_profile(), AGES)
    assert c0 == ("You are an individual investor in Taiwan. You are 30 years old. You are a "
                  "man. You check your account a few times a year and rarely trade.")
    c1 = PZ.compose_persona(PZ.control_profile(), AGES, cue=True)
    assert c1 == c0 + " " + PZ.C1_CUE
    p = {k: 0 for k in PZ.ATTRIBUTE_KEYS} | {"loss_sensitivity": 1, "mental_accounting": 1,
                                              "experience": 1, "age": 1, "trading_activity": 1}
    assert PZ.compose_persona(p, AGES) == (
        "You are an individual investor in Taiwan. Losing NT$1,000 bothers you about twice as "
        "much as gaining NT$1,000 pleases you. You think of each stock as a separate account "
        "and judge it by whether it is up or down since you bought it. You have traded "
        "Taiwan-listed stocks for more than twenty years. You are 59 years old. You are a man. "
        "You check your account every day and trade several times a month.")
    with pytest.raises(ValueError):
        PZ.compose_persona(p | {"age": 2}, AGES)


def test_roster_has_profiles_then_c0_c1():
    r = PZ.roster(3, 1, AGES)
    assert [x["id"] for x in r] == ["P00", "P01", "P02", "C0", "C1"]
    assert r[3]["profile"] == r[4]["profile"] == PZ.control_profile()
    assert PZ.C1_CUE in r[4]["text"] and PZ.C1_CUE not in r[3]["text"]


# --------------------------------------------------------------------------- calibration
def _fake_tej(seed=0, n_stocks=40):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1999-01-04", "1999-12-31")
    rows = []
    for s in range(n_stocks):
        sig = rng.uniform(0.015, 0.035)
        r = rng.standard_t(5, len(days)) * sig * np.sqrt(3 / 5)
        close = 50 * np.exp(np.cumsum(r))
        ind = "-" if s == 0 else "Cement"                 # stock 0 is a fund: dropped
        for d, ri, c in zip(days, r, close):
            rows.append({"stock": f"S{s}", "date": d, "close": c, "ret_ln": ri,
                         "market": "TSE" if s < n_stocks - 1 else "OTC",
                         "industry": ind, "limit": "-"})
    return pd.DataFrame(rows)


def test_price_calibration_is_reproducible_and_filters():
    from llm_investors.calibration.prices import calibrate_prices
    t = _fake_tej()
    a, _ = calibrate_prices(t, seed=3)
    b, _ = calibrate_prices(t, seed=3)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert a["coverage"]["stocks_used"] == 38               # fund and OTC stock dropped
    assert 2.5 <= a["student_t"]["df"] <= 100
    assert a["coverage"]["months_excluded_incomplete"] == []
    # monthly sigma ~ daily sigma * sqrt(21)
    assert 0.06 < a["sigma_summary"]["median"] < 0.13


def test_incomplete_months_are_excluded():
    from llm_investors.calibration.prices import calibrate_prices
    t = _fake_tej(n_stocks=12)
    t = t[~((t["date"].dt.month == 8) | ((t["date"].dt.month == 9) & (t["date"].dt.day > 3)))]
    p, _ = calibrate_prices(t, seed=0)
    assert set(p["coverage"]["months_excluded_incomplete"]) == {"1999-08", "1999-09"}
    assert "1999-08" not in p["coverage"]["months_used"]


def test_variance_ratio_near_one_for_iid():
    from llm_investors.calibration.prices import variance_ratio
    r = np.random.default_rng(0).standard_normal(5000)
    vr, z = variance_ratio(r, 5)
    assert abs(vr - 1) < 0.1 and abs(z) < 3


def test_tdcc_ages_from_committed_table():
    from llm_investors.calibration.attributes import bracket_median, calibrate_attributes
    a = calibrate_attributes({"tdcc_age_path": "data/investor_statistics.csv",
                              "attribute_data_path": "data/attribute_data.csv"})
    assert (a["younger_age"], a["older_age"]) == (30, 59)
    assert a["groups"]["younger"]["brackets"][-1].startswith("31")
    # interpolation: uniform 0..9 and 10..19 with equal counts -> median 10
    assert bracket_median([{"lo": 0, "hi": 9, "total": 5}, {"lo": 10, "hi": 19, "total": 5}]) == 10


def test_committed_calibration_matches_its_inputs():
    """results/calibration/ is what the generator reads; it must hold what the plan needs."""
    p = json.loads((ROOT / "results" / "calibration" / "prices.json").read_text())
    a = json.loads((ROOT / "results" / "calibration" / "attributes.json").read_text())
    assert len(p["sigma_monthly"]) == p["coverage"]["stocks_used"] >= 100
    assert p["student_t"]["df"] > 2
    assert {"autocorrelation", "ljung_box", "variance_ratio"} <= set(p["random_walk_tests"])
    assert PZ.load_ages() == {"younger_age": a["younger_age"], "older_age": a["older_age"]}


# --------------------------------------------------------------------------- OLS / power
def test_two_way_cluster_ols_matches_statsmodels_one_way():
    import statsmodels.api as sm
    from llm_investors.analysis.ols import holm, ols
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(500), rng.standard_normal((500, 2))])
    g = rng.integers(0, 25, 500)
    y = X @ [1, 2, -1] + rng.standard_normal(500) + rng.standard_normal(25)[g]
    f = ols(y, X, ["c", "a", "b"], [g])
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert np.allclose(f.beta, m.params) and np.allclose(f.se, m.bse)
    f2 = ols(y, X, ["c", "a", "b"], [g, rng.integers(0, 10, 500)])
    assert f2.g_min == 10 and f2.wald(["a", "b"])["p"] < 1e-6
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def _fake_pilot_fit():
    return {"c0_sell_rate_gain": 0.35, "c0_sell_rate_loss": 0.2,
            "variance_components": {"path": 0.001, "path_gain": 0.001, "profile": 0.001,
                                    "profile_gain": 0.0, "within_run": 0.2},
            "profile_gain_residual": 0.0, "c1_minus_c0_gain_gap": 0.2,
            "mean_sale_days_per_run": 6.0}


def _fake_blocks(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 6, n)
    rows = []
    for k in sizes:
        g = rng.integers(0, 2, k)
        rows.append(np.column_stack([g, np.where(g, 1, -1) * rng.uniform(0.01, 0.3, k),
                                     rng.integers(1, 15, k), -rng.uniform(0, 0.2, k)]))
    rows = np.vstack(rows)
    return {"rows": rows.astype(float), "start": np.r_[0, np.cumsum(sizes)[:-1]], "len": sizes}


def test_power_simulation_recovers_a_known_effect():
    from llm_investors.analysis import power as PW
    from llm_investors.analysis.study1 import design_matrix
    from llm_investors.analysis.ols import ols
    rng = np.random.default_rng(0)
    prof = PZ.sample_profiles(40, 3)
    est = []
    for _ in range(8):
        df = PW.simulate(40, 15, _fake_pilot_fit(), _fake_blocks(), prof, 0.08, rng, 6.0)
        d = df[df["kind"] == "P"]
        X, names = design_matrix(d)
        f = ols(d["sold"].to_numpy(float), X, names, [d["scenario"], d["pid"]])
        est.append(f.beta[f.idx([n for n in names if n.startswith("gain:attr_")])])
    est = np.array(est)
    assert abs(est.mean() - 0.08) < 0.01                     # every d_k = delta
    r = PW.one_sim(40, 15, _fake_pilot_fit(), _fake_blocks(), prof, 0.08, rng, 6.0)
    assert r["joint"] and r["c1"] and r["rej_dk"].mean() > 0.5


def test_power_recommendation_respects_budget():
    from llm_investors.analysis.power import TESTS, recommend
    pw = pd.DataFrame([{"P": P, "S": S, "test": t, "power": 0.9 if (P, S) != (20, 10) else 0.5}
                       for P in (20, 30) for S in (10, 15) for t in TESTS])
    cost = pd.DataFrame([{"P": P, "S": S, "runs": (P + 2) * S * 3,
                          "cost_usd": (P + 2) * S * 3 * 0.005} for P in (20, 30) for S in (10, 15)])
    r = recommend(pw, cost, 6.5)
    assert (r["P"], r["S"]) == (30, 10)                       # 20x15 costs 4.95 > 30x10 4.80
    r = recommend(pw, cost, 4.0)
    assert "largest affordable" in r["reason"] or "no design" in r["reason"]


# --------------------------------------------------------------------------- tagging
def test_tag_parsing_and_dummy_labels():
    from llm_investors.agents.parsing import InvalidReply
    from llm_investors.analysis import reasoning as RZ
    batch = [(0, "Prices move randomly, so I hold."), (1, "Stock A is up 8% since I bought it.")]
    lab = RZ.parse_labels(RZ.dummy_labels(batch), [0, 1])
    assert lab[0]["randomness"] and not lab[0]["reference_point"] and lab[1]["reference_point"]
    with pytest.raises(InvalidReply, match="missing ids"):
        RZ.parse_labels('{"labels":[{"id":0,"persona":false,"reference_point":false,'
                        '"past_trades":false,"forecast":false,"randomness":true}]}', [0, 1])
    with pytest.raises(InvalidReply):
        RZ.parse_labels('{"labels":[{"id":0,"persona":"yes"}]}', [0])
    assert "json" in RZ.user_prompt(batch).lower()             # json_object mode requirement
    assert "disposition" not in RZ.CLASSIFIER_SYSTEM.lower()


def test_tag_batches_are_stable():
    from llm_investors.analysis.reasoning import batches
    t = [f"text {i}" for i in range(45)]
    a, b = batches(t, 20), batches(list(reversed(t)), 20)
    assert [len(x) for x in a] == [20, 20, 5]
    assert [[s for _, s in x] for x in a] == [[s for _, s in x] for x in b]


def test_cohen_kappa():
    from llm_investors.analysis.reasoning import cohen_kappa
    a = np.array([1, 1, 0, 0])
    assert cohen_kappa(a, a) == 1.0 and cohen_kappa(a, 1 - a) == -1.0


# --------------------------------------------------------------------------- budget
def test_stage_limit_stops_before_the_overall_cap(tmp_path):
    from llm_investors.llm.cost import BudgetExceeded, CostTracker
    c = CostTracker(10.0, {"input": 1.0, "cached_input": 1.0, "output": 1.0},
                    tmp_path / "s.json", session_cap=0.5)
    c.settle(c.reserve(0.3), {"input_tokens": 300_000})
    with pytest.raises(BudgetExceeded, match="stage limit"):
        c.reserve(0.3)
    assert c.summary()["stage_limit_usd"] == 0.5


def test_viewer_charts_build_and_drop_undefined_ratios():
    import pandas as pd
    from llm_investors.app import charts as CH
    rows = []
    for p, plr in (("C0", 0.2), ("P00", 0.1), ("P01", 0.0)):
        for a in CH.ARCHS + ["all arms"]:
            rows.append({"persona": p, "architecture": a, "runs": 5, "PGR": 0.5, "PLR": plr,
                         "DE": 0.5 - plr, "DE_lo": 0.4 - plr, "DE_hi": 0.6 - plr,
                         "Ratio": 0.5 / plr if plr else float("nan"), "Ratio_lo": 0.0,
                         "Ratio_hi": 9.0, "PGR_lo": 0.4, "PGR_hi": 0.6, "PLR_lo": 0.0,
                         "PLR_hi": 0.3})
    cells = pd.DataFrame(rows)
    assert CH.rank_by(cells)[0] == "P01"
    ch, dropped = CH.cells_chart(cells, ["C0", "P00", "P01"], ["all arms"], "Ratio")
    assert ch is not None and dropped == ["P01 (all arms)"]
    assert CH.cells_chart(cells, ["C0"], CH.ARCHS, "DE")[0].to_dict()
    assert CH.memory_chart(cells, ["C0", "P00"], change=True, background=True).to_dict()
    assert CH.persona_order(["P01", "C1", "C0"]) == ["C0", "C1", "P01"]


def _toy_decisions(seed=0, n_paths=8, n_prof=12, n=4000):
    import pandas as pd
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"scenario": rng.integers(0, n_paths, n),
                      "persona": [f"P{i:02d}" for i in rng.integers(0, n_prof, n)],
                      "gain": rng.integers(0, 2, n), "a": rng.integers(0, 2, n).astype(float),
                      "x": rng.normal(size=n)})
    d["run_id"] = d["persona"] + "_" + d["scenario"].astype(str)
    eta = -1 + 0.8 * d["gain"] + 0.3 * d["a"] + 0.9 * d["gain"] * d["a"] + 0.5 * d["x"]
    d["sold"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return d


def _toy_builder(dd):
    g, a = dd["gain"].to_numpy(float), dd["a"].to_numpy(float)
    return np.column_stack([np.ones(len(dd)), g, a, g * a, dd["x"].to_numpy(float)])


def test_logit_matches_statsmodels_with_clustered_se():
    import pandas as pd
    import statsmodels.api as sm
    from llm_investors.analysis import models as M
    d = _toy_decisions()
    X = _toy_builder(d)
    f, ok = M.logit(d["sold"].to_numpy(float), X, list("cgaix"), [d["persona"].to_numpy()])
    r = sm.Logit(d["sold"].to_numpy(float), X).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["persona"])[0],
                                              "use_correction": False})
    G = d["persona"].nunique()
    assert ok and np.allclose(f.beta, r.params, atol=1e-8)
    # statsmodels without correction = plain sandwich; ours adds G/(G-1) as Stata does
    assert np.allclose(f.se, np.sqrt(np.diag(r.cov_params()) * G / (G - 1)), rtol=1e-6)


def test_marginal_effects_lpm_equals_coefficients_and_logit_is_sane():
    from llm_investors.analysis import models as M
    d = _toy_decisions(1)
    X = _toy_builder(d)
    names = ["const", "gain", "attr_a", "gain:attr_a", "x"]
    fits = M.fit_all(d, X, names)
    me = M.marginal_effects(fits, d, lambda dd: _toy_builder(
        dd.rename(columns={"attr_a": "a"}) if "attr_a" in dd else dd),
        [], [], [])
    assert set(me["clustering"]) == set(M.CLUSTERINGS)
    lpm = fits[0].fit
    g = me[(me["model"] == "LPM") & (me["variable"] == "gain")].iloc[0]
    # LPM AME of gain = b_gain + b_gain:a * mean(a)
    assert np.isclose(g["estimate"], lpm.beta[1] + lpm.beta[3] * d["a"].mean(), atol=1e-6)
    # effect on DE of a binary variable, via the attr path of _effects
    d2 = d.rename(columns={"a": "attr_a"})
    b2 = lambda dd: _toy_builder(dd.rename(columns={"attr_a": "a"}))  # noqa: E731
    me2 = M.marginal_effects(fits[:2], d2, b2, ["attr_a"], [], ["x"])
    de = me2[(me2["effect"] == "effect on DE") & (me2["model"] == "LPM")].iloc[0]
    assert np.isclose(de["estimate"], lpm.beta[3], atol=1e-6)
    assert np.isclose(de["se"], lpm.se[3], rtol=1e-4)
    lg = me2[(me2["effect"] == "effect on DE") & (me2["model"] == "logit")].iloc[0]
    assert 0.05 < lg["estimate"] < 0.35 and lg["se"] > 0
    xx = me2[(me2["variable"] == "x") & (me2["model"] == "LPM")].iloc[0]
    assert np.isclose(xx["estimate"], lpm.beta[4], rtol=1e-4)
