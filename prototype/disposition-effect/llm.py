"""LLM client with an on-disk response cache keyed on the hash of the full request."""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from pathlib import Path


class LLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
        self._client = None
        if cfg.backend == "anthropic":
            import anthropic  # imported lazily so the mock backend needs no SDK
            self._client = anthropic.Anthropic()  # credentials from env / `ant auth login`

    # -- cache -------------------------------------------------------------
    def _key(self, system: str, messages: list[dict]) -> str:
        payload = {
            "backend": self.cfg.backend, "model": self.cfg.model,
            "temperature": self.cfg.temperature, "effort": self.cfg.effort,
            "max_tokens": self.cfg.max_tokens, "system": system, "messages": messages,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system: str, messages: list[dict]) -> dict:
        """Return {"text", "cached", "key", "usage"}."""
        key = self._key(system, messages)
        path = self.cache_dir / f"{key}.json"
        with self._lock:
            self.stats["calls"] += 1
        if path.exists():
            rec = json.loads(path.read_text())
            with self._lock:
                self.stats["cache_hits"] += 1
            return {**rec, "cached": True, "key": key}

        if self.cfg.backend == "mock":
            text, usage = _mock_response(system, messages, key), {}
        else:
            text, usage = self._call_anthropic(system, messages)

        rec = {"text": text, "usage": usage}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(path)
        with self._lock:
            self.stats["input_tokens"] += usage.get("input_tokens", 0)
            self.stats["output_tokens"] += usage.get("output_tokens", 0)
        return {**rec, "cached": False, "key": key}

    # -- backends ----------------------------------------------------------
    def _call_anthropic(self, system: str, messages: list[dict]) -> tuple[str, dict]:
        kwargs = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                      system=system, messages=messages)
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        # SDK already retries 429/5xx/connection errors (max_retries=2 by default).
        resp = self._client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return "", {"stop_reason": "refusal"}
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens,
                 "stop_reason": resp.stop_reason, "request_id": resp._request_id}
        return text, usage


# ---------------------------------------------------------------------------
# Mock backend: deterministic, rule-based "investor" that is deliberately
# disposition-prone (sells winners, holds losers). Lets you test the whole
# pipeline, retries and metrics without spending tokens.
# ---------------------------------------------------------------------------
def _mock_response(system: str, messages: list[dict], key: str) -> str:
    rng = random.Random(key)
    last = messages[-1]["content"]
    if "three lessons" in last.lower():
        return json.dumps({"lessons": [
            "Locking in gains early felt safe.",
            "Losing positions often recovered if I waited.",
            "Keep enough cash to buy dips.",
        ]})
    # Inject a malformed reply now and then (never on a retry) to exercise the retry path.
    if len(messages) == 1 and rng.random() < 0.1:
        return "Sure! Here is my decision: {trades: [oops"
    m = re.search(r"```json\n(.*?)\n```", messages[0]["content"], re.S)
    state = json.loads(m.group(1))
    trades = []
    for s, pos in state["positions"].items():
        pnl = pos["unrealized_pnl_pct"]
        if pos["shares"] > 0 and pnl is not None and pnl > 5 and rng.random() < 0.6:
            trades.append({"stock": s, "action": "sell", "shares": max(1, pos["shares"] // 2)})
        elif pos["shares"] > 0 and pnl is not None and pnl < -5 and rng.random() < 0.1:
            trades.append({"stock": s, "action": "sell", "shares": pos["shares"]})
        elif state["cash"] > 2 * pos["current_price"] and rng.random() < 0.15:
            trades.append({"stock": s, "action": "buy", "shares": 2})
        else:
            trades.append({"stock": s, "action": "hold", "shares": 0})
    return json.dumps({"trades": trades, "reason": "mock policy"})
