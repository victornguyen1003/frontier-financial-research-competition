"""Prompt rules (plan §3.4): no behavioral-finance concepts outside persona sentences and
the C1 cue, no "disposition effect" anywhere, no real tickers, dates, timestamps or run ids."""
import re

import pytest

from llm_investors.agents import personas as PZ
from llm_investors.agents import prompts
from llm_investors.agents.memory import Memory
from llm_investors.agents.parsing import (InvalidReply, parse_decision_v3, parse_lessons,
                                          parse_orders)
from llm_investors.agents.personas import PERSONAS
from llm_investors.envs import exogenous as X

AGES = {"younger_age": 30, "older_age": 59}


def neutral_texts():
    """Rules, state, memory and format blocks: must stay neutral (no behavioral wording)."""
    acct = X.Account(50000.0, {"A": X.Position(100, 100.0, 0), "B": X.Position(0, 0.0, 0)})
    import numpy as np
    prices = np.array([[100.0, 100.0], [104.2, 97.5], [98.1, 99.0]])
    mem = Memory("reflect", design="v3")
    mem.add({"period": 1, "value": 100000, "prev_value": None,
             "trades": [{"stock": "A", "action": "sell", "quantity": 5, "price": 104.2,
                         "avg_cost": 100.0}]})
    mem.lessons = ["a", "b", "c"]
    return [X.RULES_V1, X.RULES_V2, X.RULES_V3, prompts.ORDER_FORMAT_STUDY1,
            X.env_spec(5, "v3").order_format, prompts.REFLECT_FORMAT,
            prompts.reflect_user_prompt(["Period 1: ..."], ["x"]),
            prompts.correction_message("e"), mem.block(),
            X.state_block(2, 15, acct, {"A": 98.1, "B": 99.0},
                          history=X.price_history(prices, 2, ["A", "B"])),
            prompts.decision_user_prompt("m", "s")]


def persona_texts():
    out = list(PERSONAS.values())
    out += [r["text"] for r in PZ.roster(60, 1, AGES)]
    return out


def all_prompt_texts():
    texts = neutral_texts() + persona_texts()
    for p in persona_texts()[:10]:
        texts.append(prompts.system_prompt(X.RULES_V3, p, X.env_spec(5, "v3").order_format))
        texts.append(prompts.reflect_system_prompt(X.RULES_V3, p))
    for p in PERSONAS.values():
        texts.append(prompts.system_prompt(X.RULES, p, X.env_spec(5).order_format))
    return texts


def test_rendered_formats_have_single_braces():
    for f in (X.env_spec(5).order_format, X.env_spec(5, "v3").order_format):
        assert "{{" not in f and "}}" not in f and '"orders":[' in f


@pytest.mark.parametrize("text", all_prompt_texts())
def test_no_disposition_effect_and_no_dates_or_tickers(text):
    low = text.lower()
    assert "disposition" not in low
    assert not re.search(r"\b(19|20)\d{2}\b", text)
    assert not re.search(r"\b\d{4}\.TW\b", text)
    assert not re.search(r"\d{2}:\d{2}|run_id|_s\d{3}", text)     # no timestamps or run ids


@pytest.mark.parametrize("text", neutral_texts())
def test_neutral_blocks_have_no_behavioral_wording(text):
    low = text.lower()
    for term in prompts.BANNED_TERMS + ("gone up quickly", "satisfaction", "painful",
                                        "mistake", "separate account"):
        assert term not in low, term


@pytest.mark.parametrize("text", persona_texts())
def test_personas_never_name_a_theory(text):
    low = text.lower()
    for term in prompts.BANNED_TERMS:
        assert term not in low, term


def test_v3_order_format_matches_plan():
    f = X.env_spec(5, "v3").order_format
    assert f == (
        'Reply with ONLY a compact JSON object, no other text:\n'
        '{"reasoning":"<3-5 sentences: what you considered this month and why>",\n'
        ' "orders":[{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":null,\n'
        '            "reason":"<one short sentence>"}, ...]}\n'
        'Give exactly one entry for each of A, B, C, D, E. Use quantity 0 for hold.')


def test_v3_state_block_matches_plan():
    import numpy as np
    acct = X.Account(48210.0, {"A": X.Position(100, 100.0, 0), "B": X.Position(0, 0.0, 0)})
    prices = np.array([[100.0, 100.0], [96.4, 97.5], [109.8, 99.0]])
    out = X.state_block(2, 15, acct, {"A": 112.3, "B": 95.0},
                        history=X.price_history(prices, 2, ["A", "B"]))
    assert out == (
        "Period 2 of 15.\nCash: 48,210.00\n"
        "stock | shares | average cost | price today | gain/loss vs average cost\n"
        "Stock A | 100 | 100.00 | 112.30 | +12.3%\n"
        "Stock B | 0 | - | 95.00 | -\n"
        "Account value: 59,440.00\n\n"
        "Price history:\n"
        "stock | month 0 | month 1 | today\n"
        "Stock A | 100.00 | 96.40 | 112.30\n"
        "Stock B | 100.00 | 97.50 | 95.00")


def test_parse_reasoning_required_and_bounded():
    orders = ',"orders":[{"stock":"A","action":"hold","quantity":0}]}'
    d = parse_decision_v3('{"reasoning":" I hold. "' + orders, ["A"])
    assert d["reasoning"] == "I hold." and d["orders"][0]["action"] == "hold"
    for bad in ('{"reasoning":""', '{"reasoning":"   "', '{"reasoning":null',
                '{"x":1', '{"reasoning":"' + "a" * 1001 + '"'):
        with pytest.raises(InvalidReply, match="reasoning"):
            parse_decision_v3(bad + orders, ["A"])
    assert parse_decision_v3('{"reasoning":"' + "a" * 1000 + '"' + orders, ["A"])
    with pytest.raises(InvalidReply, match="missing an entry"):
        parse_decision_v3('{"reasoning":"r"' + orders, ["A", "B"])


def test_invalid_reasoning_gets_one_correction_then_fallback():
    from llm_investors.agents.agent import Agent

    class Bad:
        dry_run = False
        n = 0

        def complete(self, system, messages, dummy=None):
            Bad.n += 1
            return {"text": '{"reasoning":"","orders":[]}', "status": "completed",
                    "cached": False, "key": "k", "usage": {}, "latency_s": 0, "cost_usd": 0,
                    "api_retries": 0}
    events = []
    a = Agent("x", "C0", "none", X.env_spec(5, "v3"), Bad(), lambda t, **k: events.append(t),
              persona_prompt="You are an individual investor in Taiwan.")
    out = a.decide(1, "state")
    assert Bad.n == 2 and all(o["reason"] == "FALLBACK" for o in out)
    assert a.last_reasoning is None and "fallback" in events


def test_parse_orders_rejects_missing_and_validates():
    o = parse_orders('{"orders":[{"stock":"Stock B","action":"SELL","quantity":5,'
                     '"limit_price":null,"reason":"r"},{"stock":"A","action":"hold",'
                     '"quantity":0}]}', ["A", "B"])
    assert [x["action"] for x in o] == ["hold", "sell"] and o[1]["quantity"] == 5
    # an omitted stock is invalid (no silent hold), and the error names it for the retry
    with pytest.raises(InvalidReply, match="missing an entry for Stock A"):
        parse_orders('{"orders":[{"stock":"B","action":"sell","quantity":5}]}', ["A", "B"])
    with pytest.raises(InvalidReply):
        parse_orders('{"orders":[{"stock":"Z","action":"buy","quantity":1}]}', ["A"])
    with pytest.raises(InvalidReply):
        parse_orders('{"orders":[{"stock":"A","action":"buy","quantity":1}]}', ["A"],
                     one_order=True, require_limit=True)
    with pytest.raises(InvalidReply):
        parse_orders('{"orders": [', ["A"])


def test_parse_lessons():
    assert len(parse_lessons('{"lessons":["a","b","c"]}')) == 3
    with pytest.raises(InvalidReply):
        parse_lessons('{"lessons":["a","b"]}')


def test_default_state_prompt_unchanged_and_ablation_hides_pnl():
    acct = X.Account(50000.0, {"A": X.Position(100, 100.0, 0), "B": X.Position(0, 0.0, 0)})
    px = {"A": 108.15, "B": 95.0}
    assert X.state_block(1, 15, acct, px) == (
        "Period 1 of 15.\nCash: 50,000.00\n"
        "stock | shares | average cost | price today | gain/loss vs average cost\n"
        "Stock A | 100 | 100.00 | 108.15 | +8.2%\n"
        "Stock B | 0 | - | 95.00 | -\n"
        "Account value: 60,815.00")
    ab = X.state_block(1, 15, acct, px, show_pnl=False)
    assert "gain/loss" not in ab and "%" not in ab and "Stock A | 100 | 100.00 | 108.15" in ab
    from llm_investors.agents.memory import history_line
    rec = {"period": 2, "value": 1000, "prev_value": None,
           "trades": [{"stock": "A", "action": "sell", "quantity": 5, "price": 110.0,
                       "avg_cost": 100.0}]}
    assert "+10.0% vs your average cost" in history_line(rec)
    assert "%" not in history_line(rec, show_pnl=False)


def test_every_user_prompt_mentions_json():
    """OpenAI json_object mode requires 'json' in the input messages; the system instructions
    do not count. A reflection prompt without it stopped the first real pilot (HTTP 400)."""
    users = [prompts.decision_user_prompt(None, "state"),
             prompts.decision_user_prompt("memory", "state"),
             prompts.reflect_user_prompt(["Period 1: ..."], []),
             prompts.reflect_user_prompt(["Period 1: ..."], ["a", "b", "c"]),
             prompts.correction_message("bad")]
    for u in users:
        assert "json" in u.lower(), u


def test_v2_state_prompt_pinned_with_recent_prices():
    """Exact v2 prompt (design v2 cache keys depend on it)."""
    import numpy as np
    acct = X.Account(50000.0, {"A": X.Position(100, 100.0, 0), "B": X.Position(0, 0.0, 0)})
    prices = np.array([[100.0, 100.0], [104.2, 97.5], [98.1, 99.0], [101.5, 95.0], [108.15, 95.0]])
    px = {"A": 108.15, "B": 95.0}
    recent = X.recent_prices(prices, 4, ["A", "B"], 3)
    assert recent == {"A": [104.2, 98.1, 101.5], "B": [97.5, 99.0, 95.0]}
    assert X.state_block(4, 15, acct, px, recent=recent) == (
        "Period 4 of 15.\nCash: 50,000.00\n"
        "stock | shares | average cost | price today | gain/loss vs average cost\n"
        "Stock A | 100 | 100.00 | 108.15 | +8.2%\n"
        "Stock B | 0 | - | 95.00 | -\n"
        "Account value: 60,815.00\n\n"
        "Recent prices:\n"
        "stock | 3 months ago | 2 months ago | last month | today\n"
        "Stock A | 104.20 | 98.10 | 101.50 | 108.15\n"
        "Stock B | 97.50 | 99.00 | 95.00 | 95.00")
    # period 1 shows only the purchase month
    early = X.state_block(1, 15, acct, {"A": 104.2, "B": 97.5},
                          recent=X.recent_prices(prices, 1, ["A", "B"], 3))
    assert "stock | last month | today\nStock A | 100.00 | 104.20" in early


def test_v2_rules_drop_the_passivity_cue():
    assert "no other information" in X.RULES_V1
    assert "no other information" not in X.rules("v2")
    assert "prices in recent months" in X.rules("v2")
    assert X.env_spec(5, "v1").rules == X.RULES_V1 and X.env_spec(5).rules == X.RULES_V2
