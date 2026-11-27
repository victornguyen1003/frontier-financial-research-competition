"""Study 1 environment: pre-generated price paths and a cash + 5-stock brokerage account."""
from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass, field

import numpy as np

from llm_investors.agents.agent import EnvSpec
from llm_investors.agents.prompts import ORDER_FORMAT_STUDY1, ORDER_FORMAT_STUDY1_V3
from llm_investors.paths import ROOT


def labels(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


# --------------------------------------------------------------------------- paths
def scenario_paths(mcfg: dict, seed: int, k: int):
    """Driftless log-price random walk: ln P(t+1) = ln P(t) + mu + sigma * Z, Z ~ N(0, 1).

    Returns prices (periods + 1, n_stocks), every stock starting at start_price, plus metadata.
    With mu = 0 the median price stays at the start price, so a position is equally likely to
    be at a gain or a loss. Scenario k always uses seed + k.
    """
    rng = np.random.default_rng(seed + k)
    n, T, s0 = mcfg["n_stocks"], mcfg["periods"], mcfg["start_price"]
    mu, sigma = float(mcfg["mu"]), float(mcfg["sigma"])
    log_r = mu + sigma * rng.standard_normal((T, n))
    prices = np.round(s0 * np.exp(np.vstack([np.zeros(n), np.cumsum(log_r, axis=0)])), 2)
    return prices, {"scenario": k, "seed": seed + k, "generator": "log_random_walk",
                    "mu": mu, "sigma": sigma}


def simulate_log_returns(rng: np.random.Generator, sigma_pool, df: float, n_stocks: int,
                         periods: int) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated zero-drift returns (design v3), shape (periods, n_stocks), and each stock's
    sigma: each stock draws its own monthly sigma from the empirical pool, and shocks are
    Student-t(df) scaled to unit variance (needs df > 2)."""
    sig = rng.choice(np.asarray(sigma_pool, dtype=float), size=n_stocks, replace=True)
    eps = rng.standard_t(df, size=(periods, n_stocks)) * np.sqrt((df - 2) / df)
    return eps * sig, sig


@functools.lru_cache(maxsize=4)
def load_calibration(path: str = "results/calibration/prices.json") -> dict:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `calibrate` first (design v3 prices are "
                                "calibrated to TEJ data)")
    return json.loads(p.read_text())


def calibrated_paths(mcfg: dict, seed: int, k: int):
    """Design v3: ln P(t+1) = ln P(t) + sigma_i * eps_t, calibrated (plan §2.1). Scenario k
    uses seed + k. Returns prices (periods + 1, n_stocks) and metadata with each sigma_i."""
    cal = load_calibration(mcfg.get("calibration_file", "results/calibration/prices.json"))
    rng = np.random.default_rng(seed + k)
    n, T, s0 = mcfg["n_stocks"], mcfg["periods"], mcfg["start_price"]
    df = float(cal["student_t"]["df"])
    log_r, sig = simulate_log_returns(rng, cal["sigma_monthly"], df, n, T)
    prices = np.round(s0 * np.exp(np.vstack([np.zeros(n), np.cumsum(log_r, axis=0)])), 2)
    return prices, {"scenario": k, "seed": seed + k, "generator": "calibrated_t_random_walk",
                    "sigmas": [round(float(x), 6) for x in sig], "df": df}


def paths_for(mcfg: dict, seed: int, k: int):
    """Dispatch on market.generator: log_random_walk (designs v1/v2) or calibrated (v3)."""
    if mcfg.get("generator", "log_random_walk") == "calibrated":
        return calibrated_paths(mcfg, seed, k)
    return scenario_paths(mcfg, seed, k)


# --------------------------------------------------------------------------- prompts
# design v1 (first pilot): ended with "You have no other information about the companies or
# the market", and agents held almost everything (DE defined in 10/45 pilot runs).
RULES_V1 = """You manage your own brokerage account. It holds cash and five stocks, called \
Stock A, Stock B, Stock C, Stock D and Stock E. Each period is one month. Every period you see \
your account and today's prices, and for each stock you decide whether to buy, sell or hold. \
Trades execute at today's price. There are no fees or taxes. You cannot borrow or short: you \
can sell at most the shares you hold, and buy only with the cash you have. You have no other \
information about the companies or the market."""

# design v2: no passivity cue; neutral monthly review; recent prices shown in every arm.
RULES_V2 = """You manage your own brokerage account. It holds cash and five stocks, called \
Stock A, Stock B, Stock C, Stock D and Stock E. Each period is one month. Each month you see \
your account, today's price of each stock and its prices in recent months, and for each stock \
you decide whether to buy, sell or hold. Trades execute at today's price. There are no fees or \
taxes. You cannot borrow or short: you can sell at most the shares you hold, and buy only with \
the cash you have."""

# design v3 (prototype 2): average cost and the full price history since month 0.
RULES_V3 = """You manage your own brokerage account. It holds cash and five stocks, called Stock A, \
Stock B, Stock C, Stock D and Stock E. Each period is one month. Each month you see your \
account, the average cost of each position, today's price of each stock and its price history \
since month 0, and for each stock you decide whether to buy, sell or hold. Trades execute at \
today's price. There are no fees or taxes. You cannot borrow or short: you can sell at most \
the shares you hold, and buy only with the cash you have."""

RULES = RULES_V2
DESIGNS = {"v1": RULES_V1, "v2": RULES_V2, "v3": RULES_V3}


def rules(design: str = "v2") -> str:
    try:
        return DESIGNS[design]
    except KeyError:
        raise ValueError(f"unknown Study 1 design {design!r}; choose from {sorted(DESIGNS)}") \
            from None


def env_spec(n_stocks: int, design: str = "v2") -> EnvSpec:
    labs = labels(n_stocks)
    fmt = ORDER_FORMAT_STUDY1_V3 if design == "v3" else ORDER_FORMAT_STUDY1
    return EnvSpec(rules=rules(design), labels=labs, order_format=fmt.format(labels=", ".join(labs)),
                   reasoning=design == "v3", design=design)


def recent_prices(prices, t: int, labs: list[str], months: int) -> dict[str, list[float]]:
    """Each stock's prices in the `months` periods before t (fewer early on; period 0 is the
    purchase month). Market information shown to every arm, not memory."""
    lo = max(0, t - months)
    return {s: [float(prices[k, i]) for k in range(lo, t)] for i, s in enumerate(labs)}


def price_history(prices, t: int, labs: list[str]) -> dict[str, list[float]]:
    """Design v3: each stock's prices in months 0 .. t-1 (market information, every arm)."""
    return {s: [float(prices[k, i]) for k in range(t)] for i, s in enumerate(labs)}


def state_block(t: int, T: int, acct: "Account", prices_t: dict, show_pnl: bool = True,
                recent: dict[str, list[float]] | None = None,
                history: dict[str, list[float]] | None = None) -> str:
    """show_pnl=False (ablation) drops the computed gain/loss column; average cost stays.
    `recent` (design v2) adds a table of each stock's prices in recent months; `history`
    (design v3) adds the full price history since month 0."""
    rows = ["stock | shares | average cost | price today" +
            (" | gain/loss vs average cost" if show_pnl else "")]
    for s, pos in acct.pos.items():
        p = prices_t[s]
        if pos.shares > 0:
            row = f"Stock {s} | {pos.shares} | {pos.avg_cost:,.2f} | {p:,.2f}"
            rows.append(row + (f" | {(p / pos.avg_cost - 1) * 100:+.1f}%" if show_pnl else ""))
        else:
            rows.append(f"Stock {s} | 0 | - | {p:,.2f}" + (" | -" if show_pnl else ""))
    out = (f"Period {t} of {T}.\nCash: {acct.cash:,.2f}\n" + "\n".join(rows) +
           f"\nAccount value: {acct.value(prices_t):,.2f}")
    if recent:
        n = max(len(v) for v in recent.values())
        ago = ["last month" if k == 1 else f"{k} months ago" for k in range(n, 0, -1)]
        lines = ["Recent prices:", "stock | " + " | ".join(ago + ["today"])]
        for s in acct.pos:
            lines.append(f"Stock {s} | " + " | ".join(f"{p:,.2f}" for p in recent[s])
                         + f" | {prices_t[s]:,.2f}")
        out += "\n\n" + "\n".join(lines)
    if history:
        n = max(len(v) for v in history.values())
        lines = ["Price history:",
                 "stock | " + " | ".join([f"month {k}" for k in range(n)] + ["today"])]
        for s in acct.pos:
            lines.append(f"Stock {s} | " + " | ".join(f"{p:,.2f}" for p in history[s])
                         + f" | {prices_t[s]:,.2f}")
        out += "\n\n" + "\n".join(lines)
    return out


# --------------------------------------------------------------------------- account
@dataclass
class Position:
    shares: int
    avg_cost: float
    opened: int          # period the current position was opened (0 = initial endowment)


@dataclass
class Account:
    cash: float
    pos: dict[str, Position] = field(default_factory=dict)

    def value(self, prices_t: dict) -> float:
        return self.cash + sum(p.shares * prices_t[s] for s, p in self.pos.items())

    def execute(self, orders: list[dict], prices_t: dict, t: int):
        """Sells first (their cash funds buys), then buys, all at today's price.
        Infeasible quantities are clipped; returns (trades, rejections)."""
        trades, rejected = [], []
        for o in sorted(orders, key=lambda o: o["action"] != "sell"):
            s, a, q = o["stock"], o["action"], o["quantity"]
            if a == "hold" or q == 0:
                continue
            p, pos = prices_t[s], self.pos[s]
            if a == "sell":
                qe = min(q, pos.shares)
                why = "sell exceeds shares held"
            else:
                qe = min(q, int(self.cash // p))
                why = "buy exceeds available cash"
            if qe < q:
                rejected.append({"stock": s, "action": a, "requested": q, "executed": qe,
                                 "reason": why})
            if qe == 0:
                continue
            cost_before = pos.avg_cost
            if a == "sell":
                self.cash += qe * p
                pos.shares -= qe
            else:
                pos.avg_cost = (pos.shares * pos.avg_cost + qe * p) / (pos.shares + qe) \
                    if pos.shares else p
                if pos.shares == 0:
                    pos.opened = t
                pos.shares += qe
                self.cash -= qe * p
            trades.append({"stock": s, "action": a, "quantity": qe, "price": p,
                           "avg_cost": cost_before})
        assert self.cash >= -1e-6 and all(p.shares >= 0 for p in self.pos.values())
        return trades, rejected


# --------------------------------------------------------------------------- dummy agent
def dummy_policy(agent_id: str, t: int, acct: Account, prices_t: dict, persona: str):
    """Rule-based stand-in (build step 1): tends to sell winners and keep losers, and
    sometimes replies with malformed JSON so the retry/fallback path is exercised."""
    calls = {"n": 0}

    def reply() -> str:
        rng = random.Random(f"{agent_id}|{t}|{calls['n']}")
        calls["n"] += 1
        if calls["n"] == 1 and rng.random() < 0.05:
            return '{"orders": [{"stock": "A", "action": "sell"'   # truncated on purpose
        sell_gain = {"young_active": 0.5, "retiree": 0.4, "experienced": 0.3}.get(persona, 0.4)
        orders = []
        for s, pos in acct.pos.items():
            p = prices_t[s]
            ret = p / pos.avg_cost - 1 if pos.shares else 0.0
            if pos.shares and ret > 0.05 and rng.random() < sell_gain:
                orders.append({"stock": s, "action": "sell", "quantity": max(1, pos.shares // 2)})
            elif pos.shares and ret < -0.05 and rng.random() < 0.1:
                orders.append({"stock": s, "action": "sell", "quantity": pos.shares})
            elif rng.random() < 0.15:
                orders.append({"stock": s, "action": "buy", "quantity": rng.randint(5, 40)})
            else:
                orders.append({"stock": s, "action": "hold", "quantity": 0})
            orders[-1] |= {"limit_price": None, "reason": "rule"}
        return json.dumps({"orders": orders})

    def lessons() -> str:
        return json.dumps({"lessons": ["Gains can disappear.", "Patience with weak stocks.",
                                       "Keep some cash."]})
    return reply, lessons
