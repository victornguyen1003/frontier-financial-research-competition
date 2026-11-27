"""Accounting and end-to-end checks with rule-based dummy agents (no API)."""
import copy
import hashlib
import json
import os
import threading

import numpy as np
import pytest

from llm_investors.agents import personas as PZ
from llm_investors.envs import exogenous as X
from llm_investors.sim import run_study1
from llm_investors.sim.common import ROOT, load_config


class NullBoard:
    def update(self, *a, **k): pass
    def finish(self, *a, **k): pass
    def echo(self, *a, **k): pass


class FakeLLM:
    """Dummy backend without the disk cache or cost tracker."""
    dry_run = False

    def complete(self, system, messages, dummy=None):
        return {"text": dummy(), "status": "dummy", "cached": False, "key": "x", "usage": {},
                "latency_s": 0.0, "cost_usd": 0.0, "api_retries": 0}


class RecordingLLM(FakeLLM):
    """Dummy replies, but records every prompt so the wiring can be checked."""

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, dummy=None):
        self.calls.append((system, messages))
        return super().complete(system, messages, dummy)


def v2_config() -> dict:
    """Design v2 as run in prototype 1 (three fixed personas, sigma 0.10, 3 recent months)."""
    cfg = copy.deepcopy(load_config("study1"))
    cfg["design"] = "v2"
    cfg["market"].update(generator="log_random_walk", price_history=3)
    cfg["llm"]["max_output_tokens"] = 800
    return cfg


def persona(pid="C0", cue=False, profile=None):
    prof = profile or PZ.control_profile()
    return {"id": pid, "kind": "C1" if cue else ("C0" if pid == "C0" else "profile"),
            "profile": prof, "text": PZ.compose_persona(prof, cue=cue),
            "n_active": sum(prof.values())}


def fixed(key="retiree"):
    return {"id": key, "kind": "fixed", "text": PZ.persona_text(key), "profile": None,
            "n_active": None}


# --------------------------------------------------------------------------- accounting
def test_account_sells_fund_buys_and_clip():
    acct = X.Account(0.0, {"A": X.Position(10, 100.0, 0), "B": X.Position(0, 0.0, 0)})
    px = {"A": 120.0, "B": 50.0}
    trades, rej = acct.execute([{"stock": "B", "action": "buy", "quantity": 30},
                                {"stock": "A", "action": "sell", "quantity": 15}], px, t=3)
    # sell clipped to 10 -> cash 1200 -> buy clipped to 24 shares of B
    assert [t["quantity"] for t in trades] == [10, 24]
    assert {r["stock"] for r in rej} == {"A", "B"}
    assert acct.pos["A"].shares == 0 and acct.pos["B"].shares == 24
    assert acct.pos["B"].avg_cost == 50.0 and acct.pos["B"].opened == 3
    assert abs(acct.cash - 0.0) < 1e-9


def test_average_cost():
    acct = X.Account(10_000.0, {"A": X.Position(10, 100.0, 0)})
    acct.execute([{"stock": "A", "action": "buy", "quantity": 10}], {"A": 80.0}, t=1)
    assert acct.pos["A"].avg_cost == 90.0 and acct.pos["A"].opened == 0


# --------------------------------------------------------------------------- prices
def test_v2_paths_reproducible_and_driftless():
    m = v2_config()["market"]
    a, meta = X.scenario_paths(m, 1, 7)
    b, _ = X.scenario_paths(m, 1, 7)
    assert np.array_equal(a, b) and a.shape == (m["periods"] + 1, 5) and (a[0] == 100).all()
    assert meta["generator"] == "log_random_walk" and meta["seed"] == 8
    assert m["mu"] == 0.0 and m["sigma"] == 0.10
    P = np.stack([X.scenario_paths(m, 20261113, k)[0] for k in range(100)])
    r = np.diff(np.log(P), axis=1)
    assert abs(r.mean()) < 0.01 and abs(r.std() - 0.10) < 0.01


def test_calibrated_paths_reproducible_and_seeded_per_scenario():
    m = load_config("study1")["market"]
    assert m["generator"] == "calibrated"
    a, meta = X.paths_for(m, 5, 3)
    b, _ = X.paths_for(m, 5, 3)
    c, _ = X.paths_for(m, 5, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and (a[0] == 100).all()
    assert meta["seed"] == 8 and len(meta["sigmas"]) == 5
    cal = X.load_calibration()
    assert set(meta["sigmas"]) <= {round(s, 6) for s in cal["sigma_monthly"]}


def test_simulated_sigma_and_kurtosis_match_the_targets():
    """The generator reproduces the calibration targets with the same estimator."""
    from scipy import stats
    from llm_investors.calibration.prices import excess_kurtosis_standardized
    cal = X.load_calibration()
    n = cal["simulated_check"]["months_per_series"]
    r, drawn = X.simulate_log_returns(np.random.default_rng(1), cal["sigma_monthly"],
                                      cal["student_t"]["df"], 20000, n)
    # each stock's sigma comes from the empirical distribution
    assert abs(np.median(drawn) - cal["sigma_summary"]["median"]) < 0.003
    assert abs(np.percentile(drawn, 75) - cal["sigma_summary"]["p75"]) < 0.005
    # shocks have unit variance: overall sd = sqrt(mean sigma^2)
    assert abs(r.std() / np.sqrt(np.mean(np.square(cal["sigma_monthly"]))) - 1) < 0.02
    k = excess_kurtosis_standardized(r.T)
    assert abs(k - cal["student_t"]["pooled_excess_kurtosis_standardized"]) < 0.06
    # driftless and unit-variance shocks
    z = r / drawn
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.02           # driftless, unit variance
    assert stats.kurtosis(r.ravel()) > 1                                # sigma mix -> fat tails


def test_zero_drift_calibrated_paths_have_no_winner_bias():
    m = load_config("study1")["market"]
    P = np.stack([X.paths_for(m, 20261113, k)[0] for k in range(300)])
    assert 0.42 < (P[:, -1, :] > 100).mean() < 0.58


# --------------------------------------------------------------------------- runs
class AlwaysIncompleteLLM:
    """Every request hits the incomplete limit, as the real client would raise it."""
    dry_run = False

    def complete(self, system, messages, dummy=None):
        from llm_investors.llm.client import IncompleteLimit
        raise IncompleteLimit("5 incomplete replies in a row (incomplete:max_output_tokens)")


def _events(tmp_path, run_id):
    return [json.loads(l) for l in (tmp_path / "logs" / "t" / f"run_{run_id}.jsonl")
            .read_text().splitlines()]


def test_incomplete_limit_stops_run_without_holds(tmp_path, monkeypatch):
    cfg = load_config("study1")
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    r = run_study1.run_one(cfg, AlwaysIncompleteLLM(), "t", persona("C0"), "log", 0,
                           threading.Event(), NullBoard())
    assert r["status"] == "incomplete_stop"
    ev = _events(tmp_path, "log_C0_s000")
    types = [e["type"] for e in ev]
    assert "incomplete_stop" in types and "fallback" not in types
    assert "period_end" not in types and "trade" not in types     # no hold was executed
    assert ev[-1]["type"] == "run_end" and ev[-1]["status"] == "incomplete_stop"


def test_v3_run_conserves_value_and_logs_profile_and_reasoning(tmp_path, monkeypatch):
    cfg = load_config("study1")
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    prof = PZ.sample_profiles(1, 3)[0]
    for arch in ("none", "log", "reflect"):
        r = run_study1.run_one(cfg, FakeLLM(), "t", persona("P00", profile=prof), arch, 3,
                               threading.Event(), NullBoard())
        assert r["status"] == "complete"
    ev = _events(tmp_path, "reflect_P00_s003")
    start = ev[0]
    assert start["profile"] == prof and start["persona_kind"] == "profile"
    assert start["n_active"] == sum(prof.values()) and start["design"] == "v3"
    prices, labs = np.array(start["prices"]), start["labels"]
    ends = [e for e in ev if e["type"] == "period_end"]
    for e in ends:
        t = e["period"]
        assert e["cash"] >= 0
        v = e["cash"] + sum(h["shares"] * prices[t, i] for i, h in
                            enumerate(e["holdings"][s] for s in labs))
        assert abs(v - e["value"]) < 1e-6
    assert all(e["reasoning"] for e in ends if "FALLBACK" not in str(e["orders"]))
    assert sum(e["type"] == "reflection" for e in ev) == 2   # after periods 5 and 10


def test_v3_prompts_full_history_and_memory_arms(tmp_path, monkeypatch):
    """Every arm sees average cost and the full price history; log adds the record and
    reflect adds the record plus lessons (plan §2.3, §3.2)."""
    cfg = load_config("study1")
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    users = {}
    for arch in ("none", "log", "reflect"):
        llm = RecordingLLM()
        run_study1.run_one(cfg, llm, "t", persona("C1", cue=True), arch, 0, threading.Event(),
                           NullBoard())
        users[arch] = [(s, m[0]["content"]) for s, m in llm.calls if len(m) == 1]
        dec = [u for s, u in users[arch] if "looking back" not in s]
        assert all("Price history:" in u and "average cost" in u for u in dec)
        p7 = next(u for u in dec if u.startswith("Period 7 of 15") or "\nPeriod 7 of 15" in u)
        assert "stock | month 0 | month 1 | month 2 | month 3 | month 4 | month 5 | month 6 | " \
               "today" in p7
    p7 = {a: next(u for s, u in users[a] if "Period 7 of 15." in u and "looking back" not in s)
          for a in users}
    assert "Your record so far:" not in p7["none"] and "Lessons" not in p7["none"]
    assert p7["log"].startswith("Your record so far:\nPeriod 1:") and "Lessons" not in p7["log"]
    assert p7["reflect"].startswith("Your record so far:\nPeriod 1:")
    assert "\n\nLessons from your own trading so far:\n1. " in p7["reflect"]
    # before the first reflection, reflect shows the record only, exactly like log
    p3 = next(u for s, u in users["reflect"] if "Period 3 of 15." in u)
    assert p3.startswith("Your record so far:\nPeriod 1:") and "Lessons" not in p3
    # the reflection prompt carries the persona and the record
    sysr, refl = next((s, u) for s, u in users["reflect"] if "looking back" in s)
    assert "Who you are: You are an individual investor in Taiwan." in sysr
    assert "You tend to sell stocks" in sysr and "Your record of all periods so far:" in refl


def test_v2_prompts_byte_identical(tmp_path, monkeypatch):
    """Design v2 (prototype 1) must still produce byte-identical prompts. The fingerprint was
    taken from the prototype-1 code (commit 035beaf) on a dummy v2 pilot and checked equal to
    this code's output before the v3 changes were committed."""
    cfg = v2_config()
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    for k in range(5):
        for p in cfg["agents"]["personas"]:
            for arch in cfg["agents"]["architectures"]:
                run_study1.run_one(cfg, FakeLLM(), "t", fixed(p), arch, k, threading.Event(),
                                   NullBoard())
    h = hashlib.sha256()
    for f in sorted((tmp_path / "logs" / "t").glob("run_*.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e["type"] == "llm_call":
                h.update((os.path.basename(f) + "|" + e["kind"] + "|" + str(e["period"]) + "|"
                          + str(e["attempt"]) + "|" + e["system"] + "|"
                          + json.dumps(e["messages"], ensure_ascii=False) + "\n").encode())
    assert h.hexdigest() == "e7ce7182a6b59723e297eafffddb2c6ed4e214d8316e294bf53c322dd02325aa"


class QuotaLLM:
    dry_run = False

    def complete(self, system, messages, dummy=None):
        import httpx2
        import openai
        from llm_investors.llm.client import FatalAPIError
        resp = httpx2.Response(429, request=httpx2.Request("POST", "https://x"))
        raise FatalAPIError("quota", openai.RateLimitError(
            "You exceeded your current quota", response=resp,
            body={"code": "insufficient_quota"}))


def test_quota_error_stops_everything_and_is_reported(tmp_path, monkeypatch):
    from llm_investors.sim.common import report_fatal
    cfg = load_config("study1")
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    stop = threading.Event()
    r = run_study1.run_one(cfg, QuotaLLM(), "t", persona("C0"), "none", 0, stop, NullBoard())
    assert r["status"] == "quota_stop" and stop.is_set()          # all other runs stop too
    assert r["fatal"]["code"] == "insufficient_quota"
    ev = _events(tmp_path, "none_C0_s000")
    types = [e["type"] for e in ev]
    assert "fatal_api_error" in types and "fallback" not in types and "period_end" not in types
    fe = next(e for e in ev if e["type"] == "fatal_api_error")
    assert fe["error"]["kind"] == "quota" and fe["period"] == 1
    r2 = run_study1.run_one(cfg, QuotaLLM(), "t", persona("C0"), "log", 0, stop, NullBoard())
    assert r2["status"] == "aborted"
    code = report_fatal([r, r2], tmp_path / "logs" / "t")
    rep = json.loads((tmp_path / "logs" / "t" / "FATAL_ERROR.json").read_text())
    assert code == 2 and rep["error"]["kind"] == "quota" and rep["runs_aborted"] == 1


# --------------------------------------------------------------------------- grid
def test_grid_is_scenario_major_and_matched():
    cfg = load_config("study1")
    roster = run_study1.persona_roster(cfg, 8)
    assert [r["id"] for r in roster][-2:] == ["C0", "C1"] and len(roster) == 10
    grid = run_study1.build_grid(roster, ["none", "log", "reflect"], 5)
    assert len(grid) == 150
    assert [g[2] for g in grid[:30]] == [0] * 30 and grid[30][2] == 1
    assert run_study1.variant_name(cfg, "gpt-6-luna") == "v3_gpt-6-luna"


def test_dataset_and_analysis_on_dummy_pilot(tmp_path, monkeypatch):
    """End to end on a small dummy grid: dataset columns, regression and tests run."""
    from llm_investors.analysis import study1 as A
    from llm_investors.measure import dataset
    cfg = load_config("study1")
    cfg["llm"]["backend"] = "dummy"
    for mod in (run_study1, dataset, A):
        monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "data" / "benchmarks").mkdir(parents=True)
    (tmp_path / "data" / "benchmarks" / "barber_2007_table3_panelB.csv").write_text(
        (ROOT / "data" / "benchmarks" / "barber_2007_table3_panelB.csv").read_text())
    roster = run_study1.persona_roster(cfg, 14)
    for p, arch, k in run_study1.build_grid(roster, ["none", "log", "reflect"], 3):
        run_study1.run_one(cfg, FakeLLM(), "d", p, arch, k, threading.Event(), NullBoard())
    dfs = dataset.build("d")
    runs = dfs["runs"]
    assert len(runs) == 16 * 3 * 3 and runs["complete"].all()
    assert {"attr_loss_sensitivity", "persona_kind", "n_active", "Ratio"} <= set(runs.columns)
    assert dfs["reasoning"]["reasoning"].notna().mean() > 0.9
    for name, df in dfs.items():
        dataset._save(df, tmp_path / "results" / "d", name)
    A.main("d")
    t = json.loads((tmp_path / "results" / "d" / "analysis" / "tests.json").read_text())
    assert t["regression_model"] == "full §5.2 model"          # 14 profiles: identified
    assert "rq1_joint" in t and "manipulation_check" in t and t["rq2_sameness"]
    assert len(t["rq1_paired"]["by_profile"]) == 14
