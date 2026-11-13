"""LLM client: cache replay, cost cap, and retry policy, against a fake OpenAI client."""
from types import SimpleNamespace

import httpx2
import openai
import pytest

from llm_investors.llm import client as C
from llm_investors.llm.cache import DiskCache
from llm_investors.llm.cost import BudgetExceeded, CostTracker

CFG = {"backend": "openai", "model": "gpt-6-luna", "service_tier": "flex",
       "reasoning_effort": "low", "max_output_tokens": 800, "json_mode": True, "timeout_s": 5,
       "max_incomplete_retries": 5}
PRICES = {"input": 0.05, "cached_input": 0.05, "output": 0.25}


def fake_response(text='{"ok": 1}', status="completed"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200,
                            input_tokens_details=SimpleNamespace(cached_tokens=0),
                            output_tokens_details=SimpleNamespace(reasoning_tokens=120))
    details = SimpleNamespace(reason="max_output_tokens") if status == "incomplete" else None
    return SimpleNamespace(id="resp_1", status=status, incomplete_details=details,
                           output_text=text, usage=usage, service_tier="flex")


def err(cls, status, body=None):
    resp = httpx2.Response(status, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
    return cls("boom", response=resp, body=body)


class FakeResponses:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(tmp_path, script, cap=1.0):
    cl = C.LLMClient({**CFG, "backend": "dummy"}, DiskCache(tmp_path / "c"),
                     CostTracker(cap, PRICES, tmp_path / "spend.json"), echo=lambda *_: None)
    cl.backend = "openai"
    cl._client = SimpleNamespace(responses=FakeResponses(script))
    return cl


def test_request_shape_cache_and_cost(tmp_path, monkeypatch):
    cl = make(tmp_path, [fake_response()])
    msgs = [{"role": "user", "content": "json: hi"}]
    r1 = cl.complete("sys", msgs)
    kw = cl._client.responses.calls[0]
    assert kw["service_tier"] == "flex" and kw["reasoning"] == {"effort": "low"}
    assert kw["max_output_tokens"] == 800 and kw["instructions"] == "sys"
    assert kw["text"] == {"format": {"type": "json_object"}} and "temperature" not in kw
    assert not r1["cached"] and abs(r1["cost_usd"] - (1000 * 0.05 + 200 * 0.25) / 1e6) < 1e-12
    r2 = cl.complete("sys", msgs)                     # replay: no second API call, $0
    assert r2["cached"] and r2["cost_usd"] == 0 and len(cl._client.responses.calls) == 1
    assert r2["text"] == r1["text"]


def test_retries_until_success_on_429_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "https://x"))
    cl = make(tmp_path, [err(openai.RateLimitError, 429), timeout,
                         err(openai.InternalServerError, 503), fake_response()])
    r = cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert r["api_retries"] == 3 and r["text"] == '{"ok": 1}'


def err_msg(cls, status, message, code=None):
    resp = httpx2.Response(status, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
    return cls(message, response=resp, body={"code": code, "message": message} if code else None)


@pytest.mark.parametrize("make_err, expected", [
    # quota / billing / spend limit -> fatal "quota" (never retried, even as a 429)
    (lambda: err_msg(openai.RateLimitError, 429, "You exceeded your current quota",
                     "insufficient_quota"), "quota"),
    (lambda: err_msg(openai.RateLimitError, 429, "You exceeded your current quota, please "
                     "check your plan and billing details."), "quota"),
    (lambda: err_msg(openai.PermissionDeniedError, 403, "Billing hard limit has been reached",
                     "billing_hard_limit_reached"), "quota"),
    (lambda: err_msg(openai.BadRequestError, 400, "Your organization's monthly spend limit "
                     "has been reached."), "quota"),
    # retried: timeouts, rate limits, resource unavailable
    (lambda: openai.APITimeoutError(request=httpx2.Request("POST", "https://x")), "retry"),
    (lambda: err_msg(openai.RateLimitError, 429, "Rate limit reached for requests",
                     "rate_limit_exceeded"), "retry"),
    (lambda: err_msg(openai.RateLimitError, 429, "Resource unavailable, please retry",
                     "resource_unavailable"), "retry"),
    (lambda: err_msg(openai.InternalServerError, 503, "Service Unavailable"), "retry"),
    (lambda: err_msg(openai.InternalServerError, 504, "Gateway Timeout"), "retry"),
    # Cloudflare / gateway errors: timeouts or upstream unavailable -> retried
    (lambda: err_msg(openai.InternalServerError, 520, "Error code: 520"), "retry"),
    (lambda: err_msg(openai.InternalServerError, 502, "Bad Gateway"), "retry"),
    (lambda: err_msg(openai.InternalServerError, 522, "Connection timed out"), "retry"),
    (lambda: err_msg(openai.InternalServerError, 524, "A timeout occurred"), "retry"),
    # everything else is fatal
    (lambda: err_msg(openai.InternalServerError, 500, "Internal error"), "api_error"),
    (lambda: err_msg(openai.BadRequestError, 400, "Unsupported parameter"), "api_error"),
    (lambda: err_msg(openai.AuthenticationError, 401, "Incorrect API key"), "api_error"),
    (lambda: err_msg(openai.NotFoundError, 404, "The model does not exist"), "api_error"),
    (lambda: openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), "api_error"),
])
def test_classify_error(make_err, expected):
    assert C.classify_error(make_err()) == expected


def test_quota_error_is_fatal_after_one_call(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(C.time, "sleep", slept.append)
    cl = make(tmp_path, [err(openai.RateLimitError, 429, body={"code": "insufficient_quota"}),
                         fake_response()])
    with pytest.raises(C.FatalAPIError) as ei:
        cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert ei.value.kind == "quota" and ei.value.to_dict()["code"] == "insufficient_quota"
    assert len(cl._client.responses.calls) == 1 and slept == []      # no retry, no backoff


def test_other_api_errors_are_fatal_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    for e in (err(openai.InternalServerError, 500), err(openai.BadRequestError, 400)):
        cl = make(tmp_path, [e, fake_response()])
        with pytest.raises(C.FatalAPIError) as ei:
            cl.complete("sys", [{"role": "user", "content": "json: x"}])
        assert ei.value.kind == "api_error" and len(cl._client.responses.calls) == 1


def test_budget_cap_stops_before_calling(tmp_path):
    cl = make(tmp_path, [fake_response()], cap=0.00001)
    with pytest.raises(BudgetExceeded):
        cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert cl._client.responses.calls == []


def test_spend_persists_across_sessions(tmp_path):
    cl = make(tmp_path, [fake_response()])
    cl.complete("sys", [{"role": "user", "content": "json: x"}])
    again = CostTracker(1.0, PRICES, tmp_path / "spend.json")
    assert again.spent > 0


def test_malformed_json_retry_once_then_logged_hold():
    from llm_investors.agents.agent import Agent
    from llm_investors.envs import exogenous as X
    events = []
    calls = []

    class LLM:
        dry_run = False

        def complete(self, system, messages, dummy=None):
            calls.append(messages)
            return {"text": "not json", "status": "completed", "cached": False, "key": "k",
                    "usage": {}, "latency_s": 0, "cost_usd": 0, "api_retries": 0}

    ag = Agent("a", "retiree", "none", X.env_spec(5), LLM(),
               lambda t, **f: events.append((t, f)), json_retries=1)
    orders = ag.decide(1, "state")
    assert len(calls) == 2 and len(calls[1]) == 3          # retry carries the correction
    assert all(o["action"] == "hold" for o in orders)
    fb = [f for t, f in events if t == "fallback"]
    assert len(fb) == 1 and fb[0]["architecture"] == "none" and fb[0]["persona"] == "retiree"
    assert [f["valid"] for t, f in events if t == "llm_call"] == [False, False]


def incomplete():
    return fake_response('{"orders": [', status="incomplete")


def test_incomplete_is_retried_charged_and_not_cached(tmp_path):
    cl = make(tmp_path, [incomplete(), incomplete(), fake_response()])
    msgs = [{"role": "user", "content": "json: x"}]
    r = cl.complete("sys", msgs)
    one = (1000 * 0.05 + 200 * 0.25) / 1e6
    assert r["text"] == '{"ok": 1}' and r["status"] == "completed"
    assert r["incomplete_retries"] == 2 and r["incomplete_usage"]["output_tokens"] == 400
    assert abs(r["cost_usd"] - 3 * one) < 1e-12                  # all three attempts paid
    assert len(cl._client.responses.calls) == 3
    replay = cl.complete("sys", msgs)                             # only the final reply cached
    assert replay["cached"] and replay["text"] == '{"ok": 1}'
    assert replay["incomplete_retries"] == 2 and replay["cost_usd"] == 0
    assert len(cl._client.responses.calls) == 3


def test_incomplete_limit_raises(tmp_path):
    cl = make(tmp_path, [incomplete() for _ in range(5)] + [fake_response()])
    with pytest.raises(C.IncompleteLimit):
        cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert len(cl._client.responses.calls) == 5                 # stopped at the limit
    rec = cl.cache.get(cl.key("sys", [{"role": "user", "content": "json: x"}]))
    assert rec["status"] == "incomplete_limit" and rec["incomplete_retries"] == 5


def test_cached_incomplete_entry_is_ignored(tmp_path):
    cl = make(tmp_path, [fake_response()])
    msgs = [{"role": "user", "content": "json: x"}]
    cl.cache.put(cl.key("sys", msgs), {"text": "{", "status": "incomplete:max_output_tokens",
                                        "usage": {}})
    r = cl.complete("sys", msgs)
    assert not r["cached"] and r["text"] == '{"ok": 1}'


def test_budget_cap_between_incomplete_retries(tmp_path):
    # cap allows the first attempt's reservation but not a second one
    one_worst = CostTracker(1, PRICES, tmp_path / "x.json").worst_case(4, 800)
    cl = make(tmp_path, [incomplete(), fake_response()], cap=one_worst * 1.2)
    with pytest.raises(BudgetExceeded):
        cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert len(cl._client.responses.calls) == 1


def test_fatal_error_halts_all_further_calls(tmp_path):
    cl = make(tmp_path, [err(openai.RateLimitError, 429, body={"code": "insufficient_quota"}),
                         fake_response(), fake_response()])
    with pytest.raises(C.FatalAPIError):
        cl.complete("sys", [{"role": "user", "content": "json: a"}])
    for content in ("b", "c"):                       # other agents / runs: no API call at all
        with pytest.raises(C.Halted):
            cl.complete("sys", [{"role": "user", "content": content}])
    assert len(cl._client.responses.calls) == 1


def test_concurrent_identical_prompts_make_one_call(tmp_path):
    import threading as th
    gate = th.Event()

    class SlowResponses(FakeResponses):
        def create(self, **kw):
            gate.wait(2)                      # hold the first call until all threads queued
            return super().create(**kw)

    cl = make(tmp_path, [])
    cl._client = SimpleNamespace(responses=SlowResponses([fake_response(f'{{"n": {i}}}')
                                                          for i in range(4)]))
    msgs = [{"role": "user", "content": "json: same prompt"}]
    out = []
    ts = [th.Thread(target=lambda: out.append(cl.complete("sys", msgs))) for _ in range(4)]
    for t in ts:
        t.start()
    gate.set()
    for t in ts:
        t.join()
    assert len(cl._client.responses.calls) == 1             # one paid call
    assert {r["text"] for r in out} == {'{"n": 0}'}          # everyone got the same draw
    assert sum(not r["cached"] for r in out) == 1 and sum(r["cost_usd"] > 0 for r in out) == 1


def test_call_stats_retry_rate_and_tier_warning(tmp_path, capsys):
    import json as js
    from llm_investors.sim.common import call_stats, check_service_tier
    u = {"input_tokens": 1000, "output_tokens": 200, "service_tier": "flex"}
    ev = [{"type": "llm_call", "valid": False, "usage": u},
          {"type": "llm_call", "valid": True, "usage": u},
          {"type": "llm_call", "valid": True, "usage": {**u, "service_tier": "default"}}]
    (tmp_path / "run_a.jsonl").write_text("\n".join(js.dumps(e) for e in ev))
    st = call_stats(tmp_path, CostTracker(1, PRICES, tmp_path / "s.json").price)
    assert st["calls"] == 3 and st["json_retry_rate"] == 0.5
    assert st["service_tiers"] == {"flex": 2, "default": 1}
    check_service_tier(st, "flex")
    assert "1 of 3 calls were served on tier(s) {'default': 1}" in capsys.readouterr().out


def test_incomplete_limit_replays_for_free(tmp_path):
    msgs = [{"role": "user", "content": "json: x"}]
    cl = make(tmp_path, [incomplete() for _ in range(5)])
    with pytest.raises(C.IncompleteLimit):
        cl.complete("sys", msgs)
    spent = cl.cost.spent
    cl2 = make(tmp_path, [])                                      # a new session, same cache
    with pytest.raises(C.IncompleteLimit) as ei:
        cl2.complete("sys", msgs)
    assert ei.value.cached
    assert str(ei.value) == "5 incomplete replies in a row (incomplete:max_output_tokens)"
    assert cl2._client.responses.calls == [] and cl2.cost.spent == spent
    # opting in pays to try again
    cl3 = make(tmp_path, [fake_response()])
    cl3.retry_incomplete = True
    assert cl3.complete("sys", msgs)["text"] == '{"ok": 1}'
    assert len(cl3._client.responses.calls) == 1


def test_offline_replay_never_calls_the_api(tmp_path, monkeypatch):
    def no_client(**kw):
        raise AssertionError("OpenAI client must not be created during an offline replay")
    monkeypatch.setattr(openai, "OpenAI", no_client)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    msgs = [{"role": "user", "content": "json: x"}]
    seed = make(tmp_path, [fake_response()])
    seed.complete("sys", msgs)                                   # fill the cache
    cl = C.LLMClient(CFG, DiskCache(tmp_path / "c"),
                     CostTracker(1.0, PRICES, tmp_path / "spend.json"), offline=True,
                     retry_incomplete=True, echo=lambda *_: None)
    hit = cl.complete("sys", msgs)
    assert hit["cached"] and hit["cost_usd"] == 0
    with pytest.raises(C.CacheMiss):
        cl.complete("sys", [{"role": "user", "content": "json: changed prompt"}])
    assert cl._client is None


def test_client_is_created_lazily_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cl = C.LLMClient(CFG, DiskCache(tmp_path / "c"), CostTracker(1.0, PRICES, tmp_path / "s"))
    assert cl._client is None                                     # no key needed to start


def test_corrupt_cache_file_raises_instead_of_paying(tmp_path):
    from llm_investors.llm.cache import CacheCorrupt
    msgs = [{"role": "user", "content": "json: x"}]
    cl = make(tmp_path, [fake_response()])
    path = cl.cache._path(cl.key("sys", msgs))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<<<<<<< HEAD\n{"text": "a"}\n=======\n{"text": "b"}\n>>>>>>> other\n')
    with pytest.raises(CacheCorrupt, match="Sharing the cache"):
        cl.complete("sys", msgs)
    assert cl._client.responses.calls == []
    assert cl.cache.stats()["corrupt"] == 1


def test_cached_records_carry_their_cost_and_set_the_shared_budget(tmp_path):
    cl = make(tmp_path, [fake_response(), incomplete(), fake_response()])
    cl.complete("sys", [{"role": "user", "content": "json: a"}])
    cl.complete("sys", [{"role": "user", "content": "json: b"}])        # one incomplete + one complete
    one = (1000 * 0.05 + 200 * 0.25) / 1e6
    st = cl.cache.stats()
    assert st["entries"] == 2 and abs(st["paid_usd"] - 3 * one) < 1e-12
    # a collaborator with an empty local ledger starts from the shared cache total
    fresh = CostTracker(3 * one + 1e-9, PRICES, tmp_path / "other_machine.json",
                        shared_spent=st["paid_usd"])
    assert fresh.spent_before == st["paid_usd"]
    with pytest.raises(BudgetExceeded):
        fresh.reserve(1e-6)


def test_json_mode_without_json_in_input_stops_before_paying(tmp_path):
    cl = make(tmp_path, [fake_response()])
    with pytest.raises(C.FatalAPIError, match="needs the word 'json'"):
        cl.complete("Reply in JSON.", [{"role": "user", "content": "Write three lessons."}])
    assert cl._client.responses.calls == [] and cl.cost.spent == 0 and cl.halted.is_set()


def test_cloudflare_520_is_retried_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    cl = make(tmp_path, [err(openai.InternalServerError, 520), err(openai.InternalServerError, 524),
                         fake_response()])
    r = cl.complete("sys", [{"role": "user", "content": "json: x"}])
    assert r["api_retries"] == 2 and r["text"] == '{"ok": 1}' and not cl.halted.is_set()
