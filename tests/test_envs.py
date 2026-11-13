"""Accounting and end-to-end checks with rule-based dummy agents (no API)."""
import json
import threading

import numpy as np

from llm_investors.envs import exogenous as X
from llm_investors.sim import run_study1, run_study2
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


def test_paths_reproducible_and_start_at_100():
    m = load_config("study1")["market"]
    a, meta = X.scenario_paths(m, 1, 7)
    b, _ = X.scenario_paths(m, 1, 7)
    assert np.array_equal(a, b) and a.shape == (m["periods"] + 1, 5) and (a[0] == 100).all()
    assert meta["generator"] == "log_random_walk" and meta["seed"] == 8


def test_random_walk_is_driftless():
    m = load_config("study1")["market"]
    assert m["mu"] == 0.0 and m["sigma"] == 0.10
    P = np.stack([X.scenario_paths(m, 20261113, k)[0] for k in range(100)])
    r = np.diff(np.log(P), axis=1)
    assert abs(r.mean()) < 0.01 and abs(r.std() - 0.10) < 0.01
    assert 0.4 < (P[:, -1, :] > 100).mean() < 0.6


class AlwaysIncompleteLLM:
    """Every request hits the incomplete limit, as the real client would raise it."""
    dry_run = False

    def complete(self, system, messages, dummy=None):
        from llm_investors.llm.client import IncompleteLimit
        raise IncompleteLimit("5 incomplete replies in a row (incomplete:max_output_tokens)")


def test_incomplete_limit_stops_run_without_holds(tmp_path, monkeypatch):
    cfg = load_config("study1")
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    r = run_study1.run_one(cfg, AlwaysIncompleteLLM(), "t", "retiree", "log", 0,
                           threading.Event(), NullBoard())
    assert r["status"] == "incomplete_stop"
    ev = [json.loads(l) for l in (tmp_path / "logs" / "t" / "run_log_retiree_s000.jsonl")
          .read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert "incomplete_stop" in types and "fallback" not in types
    assert "period_end" not in types and "trade" not in types     # no hold was executed
    assert ev[-1]["type"] == "run_end" and ev[-1]["status"] == "incomplete_stop"


def test_study1_run_conserves_value(tmp_path, monkeypatch):
    cfg = load_config("study1")
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    for arch in ("none", "log", "reflect"):
        r = run_study1.run_one(cfg, FakeLLM(), "t", "retiree", arch, 3, threading.Event(),
                               NullBoard())
        assert r["status"] == "complete"
    ev = [json.loads(l) for l in (tmp_path / "logs" / "t" / "run_reflect_retiree_s003.jsonl")
          .read_text().splitlines()]
    prices = np.array(ev[0]["prices"])
    labs = ev[0]["labels"]
    for e in (e for e in ev if e["type"] == "period_end"):
        t = e["period"]
        assert e["cash"] >= 0
        v = e["cash"] + sum(h["shares"] * prices[t, i] for i, h in
                            enumerate(e["holdings"][s] for s in labs))
        assert abs(v - e["value"]) < 1e-6
    assert sum(e["type"] == "reflection" for e in ev) == 2   # after periods 5 and 10


def test_study2_market_runs_and_conserves(tmp_path, monkeypatch):
    cfg = load_config("study2")
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study2, "ROOT", tmp_path)
    for P in ({"n_agents": 12, "periods": 8},
              {"n_agents": 9, "periods": 6, "price_limit": 0.1, "transaction_tax": 0.003,
               "memory": "reflect"}):
        params = run_study2.cell_params(cfg, P)
        r = run_study2.run_market(cfg, FakeLLM(), "t", "c", params, 0, threading.Event(),
                                  NullBoard(), 4)
        assert r["status"] == "complete"   # conservation asserts ran every period
    ev = [json.loads(l) for l in (tmp_path / "logs" / "t" / "run_c_r00.jsonl").read_text()
          .splitlines()]
    fv1 = 24 * 6
    assert {a["cash"] + a["shares"] * fv1 for a in ev[0]["agents"]} == {cfg["market"]["wealth"]}
    auctions = [e for e in ev if e["type"] == "auction_cleared"]
    assert len(auctions) == 6
    for prev, cur in zip(auctions, auctions[1:]):
        lo, hi = cur["band"]
        assert lo - 1e-9 <= cur["price"] <= hi + 1e-9
        assert abs(lo - prev["price"] * 0.9) < 1e-6


def test_persona_counts():
    import random
    out = run_study2.persona_list({"a": 0.6, "b": 0.2, "c": 0.2}, 30, random.Random(0))
    assert (out.count("a"), out.count("b"), out.count("c")) == (18, 6, 6)


def test_study2_incomplete_limit_stops_market_and_is_counted(tmp_path, monkeypatch):
    from llm_investors.measure import dataset
    cfg = load_config("study2")
    monkeypatch.setattr(run_study2, "ROOT", tmp_path)
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    params = run_study2.cell_params(cfg, {"n_agents": 4, "periods": 3})
    r = run_study2.run_market(cfg, AlwaysIncompleteLLM(), "study2_t", "c", params, 0,
                              threading.Event(), NullBoard(), 2)
    assert r["status"] == "incomplete_stop"
    mk = dataset.build("study2_t")["markets"]
    assert len(mk) == 1 and not mk["complete"][0] and mk["status"][0] == "incomplete_stop"
    assert mk["incomplete_stops"][0] >= 1 and mk["periods"][0] == 0


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
    r = run_study1.run_one(cfg, QuotaLLM(), "t", "retiree", "none", 0, stop, NullBoard())
    assert r["status"] == "quota_stop" and stop.is_set()          # all other runs stop too
    assert r["fatal"]["code"] == "insufficient_quota"
    ev = [json.loads(l) for l in (tmp_path / "logs" / "t" / "run_none_retiree_s000.jsonl")
          .read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert "fatal_api_error" in types and "fallback" not in types and "period_end" not in types
    fe = next(e for e in ev if e["type"] == "fatal_api_error")
    assert fe["error"]["kind"] == "quota" and fe["period"] == 1
    # a run that starts after the stop makes no calls
    r2 = run_study1.run_one(cfg, QuotaLLM(), "t", "retiree", "log", 0, stop, NullBoard())
    assert r2["status"] == "aborted"
    code = report_fatal([r, r2], tmp_path / "logs" / "t")
    rep = json.loads((tmp_path / "logs" / "t" / "FATAL_ERROR.json").read_text())
    assert code == 2 and rep["error"]["kind"] == "quota" and rep["runs_aborted"] == 1
    assert report_fatal([{"run_id": "x", "status": "complete", "fatal": None}],
                        tmp_path / "other") == 0


class RecordingLLM(FakeLLM):
    """Dummy replies, but records every prompt so treatment wiring can be checked."""

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, dummy=None):
        self.calls.append((system, messages))
        return super().complete(system, messages, dummy)


def _market_prompts(tmp_path, monkeypatch, **cell):
    cfg = load_config("study2")
    cfg["llm"]["backend"] = "dummy"
    monkeypatch.setattr(run_study2, "ROOT", tmp_path)
    llm = RecordingLLM()
    params = run_study2.cell_params(cfg, {"n_agents": 10, "periods": 7, **cell})
    r = run_study2.run_market(cfg, llm, "w", "c", params, 0, threading.Event(), NullBoard(), 2)
    assert r["status"] == "complete"
    start = json.loads((tmp_path / "logs" / "w" / "run_c_r00.jsonl").read_text().splitlines()[0])
    return llm.calls, start


def test_study2_treatments_reach_the_prompts(tmp_path, monkeypatch):
    """The dummy policy ignores persona, memory and tax, so identical dummy outcomes cannot
    show that a treatment is wired in. Check the prompts instead (audit finding 11)."""
    from llm_investors.agents.personas import PERSONAS
    base, start = _market_prompts(tmp_path, monkeypatch)
    systems = " ".join(s for s, _ in base)
    users = [m[0]["content"] for _, m in base]
    assert "transaction tax" not in systems and "at most" not in systems
    assert any("Your record of previous periods" in u for u in users)          # baseline = log

    tax, _ = _market_prompts(tmp_path, monkeypatch, transaction_tax=0.003)
    assert all("transaction tax of 0.3% of the sale value" in s for s, _ in tax)
    lim, _ = _market_prompts(tmp_path, monkeypatch, price_limit=0.10)
    assert all("at most 10% up or down" in s for s, _ in lim)

    none, _ = _market_prompts(tmp_path, monkeypatch, memory="none")
    assert not any("Your record of previous periods" in m[0]["content"] for _, m in none)
    refl, _ = _market_prompts(tmp_path, monkeypatch, memory="reflect")
    assert any("looking back on your own trading" in s for s, _ in refl)      # reflection call
    assert any("Lessons you wrote" in m[0]["content"] for _, m in refl)       # lessons used

    _, rh = _market_prompts(tmp_path, monkeypatch, agent_mix="retail_heavy")
    personas = [a["persona"] for a in rh["agents"]]
    assert personas.count("young_active") == 6 and personas.count("retiree") == 2
    assert any(PERSONAS["young_active"] in s for s, _ in base)


def test_price_history_is_shown_in_every_arm_and_memory_differs(tmp_path, monkeypatch):
    """Price history is market information for all arms; arms differ only in own history."""
    cfg = load_config("study1")
    cfg["llm"]["backend"] = "dummy"
    assert cfg["design"] == "v2" and cfg["market"]["price_history"] == 3
    monkeypatch.setattr(run_study1, "ROOT", tmp_path)
    users = {}
    for arch in ("none", "log", "reflect"):
        llm = RecordingLLM()
        run_study1.run_one(cfg, llm, "t", "retiree", arch, 0, threading.Event(), NullBoard())
        users[arch] = [(s, m[0]["content"]) for s, m in llm.calls]
        decisions = [u for s, u in users[arch] if "looking back" not in s]
        assert all("Recent prices:" in u for u in decisions)
        assert all("no other information" not in s for s, _ in users[arch])
    p6 = {a: next(u for s, u in users[a] if "Period 6 of 15" in u) for a in users}
    assert "Your record of previous periods" in p6["log"]
    assert "Lessons you wrote" in p6["reflect"]
    assert "Your record" not in p6["none"] and "Lessons" not in p6["none"]
