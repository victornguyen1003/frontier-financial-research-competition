"""Study 2 environment: a Smith-Suchanek-Williams (1988) asset market with a call auction.

One risky stock (Stock A) + cash. Each period: agents submit a two-sided quote (one buy and one
sell limit order; design v2) or one limit order (design v1), a single call auction clears at
the volume-maximizing price, then each share pays a random dividend.
Fundamental value at period t = E[dividend] x remaining periods (including t).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from llm_investors.agents.agent import EnvSpec
from llm_investors.agents.parsing import InvalidReply
from llm_investors.agents.prompts import ORDER_FORMAT_STUDY2, QUOTE_FORMAT_STUDY2


# --------------------------------------------------------------------------- auction
def clear_call_auction(buys: list[dict], sells: list[dict], ref_price: float,
                       price_limit: float | None = None, min_imbalance: bool = True) -> dict:
    """buys/sells: [{"id", "quantity", "limit_price"}].

    Clearing price = the price that maximizes executed volume, among the order limits (clipped
    to the price band when a limit applies) and ref_price. Ties: smallest |demand - supply|
    (if min_imbalance), then closest to ref_price, then the lower price.
    min_imbalance=False is the proposal's literal rule (volume, then closest to ref_price). It
    leaves the price stuck at ref_price whenever a large one-sided excess crosses over a range
    that contains ref_price, so excess supply or demand never moves the price.
    All eligible orders on the long side are filled pro rata (largest-remainder rounding,
    ties by order position); the short side fills completely.
    Returns {price, volume, traded, demand, supply, imbalance, fills: {id: +/-qty}, band}.
    """
    lo, hi = 0.0, math.inf
    if price_limit:
        lo, hi = ref_price * (1 - price_limit), ref_price * (1 + price_limit)
    # Candidates: every limit (clipped to the band) plus the reference price itself, so that
    # when orders cross over a range the auction can clear exactly at the previous price.
    cands = {min(max(o["limit_price"], lo), hi) for o in buys + sells} | {ref_price}
    best = None
    for p in sorted(cands):
        d = sum(o["quantity"] for o in buys if o["limit_price"] >= p - 1e-9)
        s = sum(o["quantity"] for o in sells if o["limit_price"] <= p + 1e-9)
        v = min(d, s)
        score = (v, -abs(d - s) if min_imbalance else 0, -abs(p - ref_price), -p)
        if best is None or score > best[0]:
            best = (score, p, d, s)
    band = None if not price_limit else [round(lo, 4), round(hi, 4)]
    if best is None or best[0][0] == 0:
        return {"price": ref_price, "volume": 0, "traded": False, "demand": 0, "supply": 0,
                "imbalance": 0, "fills": {}, "band": band}
    _, p, d, s = best
    v = min(d, s)
    eb = [o for o in buys if o["limit_price"] >= p - 1e-9]
    es = [o for o in sells if o["limit_price"] <= p + 1e-9]
    fills = {}
    for side, orders, total, sign in ((eb, eb, d, 1), (es, es, s, -1)):
        alloc = _pro_rata(orders, v) if total > v else {o["id"]: o["quantity"] for o in orders}
        for i, q in alloc.items():
            if q:
                fills[i] = fills.get(i, 0) + sign * q
    return {"price": round(p, 4), "volume": v, "traded": True, "demand": d, "supply": s,
            "imbalance": d - s, "fills": fills, "band": band}


def _pro_rata(orders: list[dict], volume: int) -> dict:
    total = sum(o["quantity"] for o in orders)
    raw = [(o["id"], volume * o["quantity"] / total) for o in orders]
    alloc = {i: int(math.floor(x)) for i, x in raw}
    left = volume - sum(alloc.values())
    order = sorted(range(len(raw)), key=lambda k: (-(raw[k][1] - math.floor(raw[k][1])), k))
    for k in order[:left]:
        alloc[raw[k][0]] += 1
    return alloc


# --------------------------------------------------------------------------- traders
@dataclass
class Trader:
    cash: float
    shares: int
    avg_cost: float      # endowed shares are booked at the period-1 fundamental value
    opened: int = 0

    def value(self, price: float) -> float:
        return self.cash + self.shares * price


def fundamental_value(t: int, T: int, dividends: list[float]) -> float:
    return sum(dividends) / len(dividends) * (T - t + 1)


def validate(order: dict, tr: Trader) -> tuple[dict, dict | None]:
    """Clip an order to what the trader can afford / owns. Returns (order, rejection or None)."""
    a, q, lp = order["action"], order["quantity"], order["limit_price"]
    if a == "hold" or q == 0:
        return order, None
    feasible = min(q, tr.shares) if a == "sell" else min(q, int(tr.cash // lp))
    if feasible == q:
        return order, None
    rej = {"action": a, "requested": q, "executed": feasible, "limit_price": lp,
           "reason": "sell exceeds shares held" if a == "sell" else "buy exceeds cash at limit price"}
    new = dict(order, quantity=feasible)
    if feasible == 0:
        new.update(action="hold", limit_price=None)
    return new, rej


# --------------------------------------------------------------------------- prompts
ORDER_SENTENCE = {
    "v1": "Every period each trader may submit one order to buy or sell at a limit price.",
    "v2": ("Every period each trader submits two limit orders: a buy order at the highest price "
           "they will pay and a sell order at the lowest price they will accept."),
}


def rules(T: int, dividends: list[float], price_limit: float | None, tax: float,
          design: str = "v2") -> str:
    divs = ", ".join(f"{d:g}" for d in dividends[:-1]) + f" or {dividends[-1]:g}"
    ev = sum(dividends) / len(dividends)
    text = (f"You trade in a market with one stock, Stock A, and cash. The market lasts {T} "
            f"periods. At the end of every period each share pays a dividend of {divs}, each with "
            f"equal chance (average {ev:g}). After the dividend of period {T}, shares are worth "
            f"nothing. {ORDER_SENTENCE[design]} "
            f"All orders are collected and executed in a single auction at the one price "
            f"that lets the most shares trade; when more shares are offered than wanted at that "
            f"price (or the other way round), orders are filled in proportion to their size. "
            f"You cannot sell more shares than you own or spend more cash than you have. "
            f"Dividends are paid in cash.")
    if price_limit:
        text += (f" The price can move at most {price_limit * 100:g}% up or down from the "
                 f"previous period's price.")
    if tax:
        text += f" Sellers pay a transaction tax of {tax * 100:g}% of the sale value."
    return text


def env_spec(T: int, dividends, price_limit, tax, design: str = "v2") -> EnvSpec:
    if design == "v1":
        return EnvSpec(rules=rules(T, dividends, price_limit, tax, "v1"), labels=["A"],
                       order_format=ORDER_FORMAT_STUDY2.format(), one_order=True,
                       require_limit=True)
    return EnvSpec(rules=rules(T, dividends, price_limit, tax, "v2"), labels=["A"],
                   order_format=QUOTE_FORMAT_STUDY2, one_order=True, require_limit=True,
                   two_sided=True)


def quote_check(tr: "Trader"):
    """Account-dependent rule for a two-sided quote: each side needs quantity >= 1 unless it
    is infeasible (sell with no shares; buy when cash cannot pay one share at the buy limit).
    A quote is not a commitment to trade (an agent can price a side far from the market), but a
    side with quantity 0 would let the market go one-sided again, as in design v1."""
    def check(orders: list[dict]):
        buy, sell = orders
        if sell["quantity"] == 0 and tr.shares > 0:
            raise InvalidReply(f"sell quantity must be at least 1: you own {tr.shares} shares")
        if buy["quantity"] == 0 and tr.cash >= buy["limit_price"]:
            raise InvalidReply(f"buy quantity must be at least 1: your cash ({tr.cash:,.2f}) "
                               f"pays for at least one share at your buy limit_price")
    return check


def state_block(t: int, T: int, tr: Trader, history: list[dict], ev: float,
                last_price: float) -> str:
    fv = ev * (T - t + 1)
    lines = [f"Period {t} of {T}."]
    if history:
        lines.append("Market history:\nperiod | price | shares traded | dividend per share")
        lines += [f"{h['period']} | {h['price']:,.2f}{'' if h['traded'] else ' (no trade)'} | "
                  f"{h['volume']} | {h['dividend']:g}" for h in history]
    else:
        lines.append("This is the first period; no trades have happened yet.")
    lines.append(f"Expected value of the remaining dividends per share: {ev:g} x {T - t + 1} "
                 f"periods = {fv:,.2f}.")
    cost = f"{tr.avg_cost:,.2f}" if tr.shares else "-"
    lines.append(f"Your account: cash {tr.cash:,.2f}; shares {tr.shares}; average cost {cost}; "
                 f"value at the last price ({last_price:,.2f}): {tr.value(last_price):,.2f}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- dummy agent
def dummy_policy(agent_id: str, t: int, tr: Trader, fv: float, last_price: float,
                 two_sided: bool = True):
    calls = {"n": 0}

    def reply() -> str:
        rng = random.Random(f"{agent_id}|{t}|{calls['n']}")
        calls["n"] += 1
        if calls["n"] == 1 and rng.random() < 0.05:
            return "I think I will buy some shares."          # not JSON on purpose
        belief = fv * rng.uniform(0.8, 1.25)
        if two_sided:
            bid = round(belief * rng.uniform(0.85, 0.98), 2)
            ask = round(max(belief * rng.uniform(1.02, 1.15), bid + 0.01), 2)
            bq = rng.randint(1, 2) if tr.cash >= bid else 0
            sq = rng.randint(1, min(2, tr.shares)) if tr.shares else 0
            return json.dumps({"orders": [
                {"stock": "A", "action": "buy", "quantity": bq, "limit_price": bid,
                 "reason": "rule"},
                {"stock": "A", "action": "sell", "quantity": sq, "limit_price": ask,
                 "reason": "rule"}]})
        if belief > last_price * 1.02 and tr.cash > belief:
            o = {"action": "buy", "quantity": rng.randint(1, 2), "limit_price": round(belief, 2)}
        elif belief < last_price * 0.98 and tr.shares:
            o = {"action": "sell", "quantity": rng.randint(1, min(2, tr.shares)),
                 "limit_price": round(belief, 2)}
        else:
            o = {"action": "hold", "quantity": 0, "limit_price": None}
        return json.dumps({"orders": [{"stock": "A", **o, "reason": "rule"}]})

    def lessons() -> str:
        return json.dumps({"lessons": ["Value falls each period.", "Do not overpay late.",
                                       "Dividends add up."]})
    return reply, lessons
