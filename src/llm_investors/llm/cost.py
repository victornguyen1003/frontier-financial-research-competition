"""Cost tracking from the API usage field, with a hard per-study cap.

Spend is persisted in cache/<study>/spend.json (local, not committed), so the cap covers the
pilot and the full run together. Because the cache itself is shared through git, the starting
spend is the larger of that local ledger and `shared_spent`, the total recorded in the cached
replies, so every collaborator works against the same cap. Cache replays cost nothing.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


def price_usage(prices: dict, usage: dict) -> float:
    """USD for one response's usage, with prices per 1M tokens."""
    inp, cached = usage.get("input_tokens", 0), usage.get("cached_input_tokens", 0)
    return ((inp - cached) * prices["input"] + cached * prices["cached_input"]
            + usage.get("output_tokens", 0) * prices["output"]) / 1e6


class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(self, cap_usd: float, prices: dict, ledger: str | Path,
                 shared_spent: float = 0.0):
        self.cap = float(cap_usd)
        self.prices = prices
        self.ledger = Path(ledger)
        self.lock = threading.Lock()
        self.reserved = 0.0
        prev = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
        self.local_spent = float(prev.get("spent_usd", 0.0))
        self.shared_spent = float(shared_spent)
        self.spent_before = max(self.local_spent, self.shared_spent)
        self.session = {"spent_usd": 0.0, "api_calls": 0, "cache_hits": 0, "input_tokens": 0,
                        "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    @property
    def spent(self) -> float:
        return self.spent_before + self.session["spent_usd"]

    def price(self, usage: dict) -> float:
        return price_usage(self.prices, usage)

    def worst_case(self, prompt_chars: int, max_output_tokens: int) -> float:
        # ~3 chars per token is a deliberately high token estimate.
        return (prompt_chars / 3 * self.prices["input"] + max_output_tokens * self.prices["output"]) / 1e6

    def reserve(self, amount: float) -> float:
        with self.lock:
            if self.spent + self.reserved + amount > self.cap:
                raise BudgetExceeded(
                    f"budget cap ${self.cap:.2f} reached (spent ${self.spent:.4f}, "
                    f"in flight ${self.reserved:.4f})")
            self.reserved += amount
        return amount

    def settle(self, reservation: float, usage: dict | None) -> float:
        cost = self.price(usage) if usage else 0.0
        with self.lock:
            self.reserved -= reservation
            if usage:
                s = self.session
                s["spent_usd"] += cost
                s["api_calls"] += 1
                for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
                    s[k] += usage.get(k, 0)
                self._save()
        return cost

    def hit(self):
        with self.lock:
            self.session["cache_hits"] += 1

    def _save(self):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text(json.dumps({"spent_usd": self.spent, "cap_usd": self.cap}, indent=2))
        tmp.replace(self.ledger)

    def summary(self) -> dict:
        with self.lock:
            return {**self.session, "spent_total_usd": round(self.spent, 6), "cap_usd": self.cap}
