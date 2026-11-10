"""LLM investor agent with optional memory."""
from __future__ import annotations

import json
import re

MEMORY_MODES = ("none", "log", "reflect")

SYSTEM_TEMPLATE = """You are role-playing an individual investor: {persona}.
Each period you manage your own brokerage account in a market of {n} anonymous stocks \
({labels}). You only see the information given to you; there is no news and no other data.
Trades execute at the current price shown. There are no fees. You cannot short or borrow: \
you can sell at most the shares you hold and buy only with the cash you have.

Reply with ONLY a JSON object, no other text, in exactly this format:
{{"trades": [{{"stock": "A", "action": "sell" | "hold" | "buy", "shares": <int>}}, ...], \
"reason": "<one or two sentences>"}}
Include one entry for every stock. Use "shares": 0 for "hold"."""

REFLECT_SYSTEM_TEMPLATE = """You are role-playing an individual investor: {persona}.
You have been trading {n} anonymous stocks ({labels}) in your own account and are now looking back on your experience. Reply with ONLY a JSON object, no other text."""


class InvalidResponse(ValueError):
    pass


def extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise InvalidResponse("no JSON object found")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise InvalidResponse(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise InvalidResponse("top-level JSON must be an object")
    return obj


def parse_decision(text: str, labels: list[str]) -> dict:
    obj = extract_json(text)
    trades = obj.get("trades")
    if not isinstance(trades, list):
        raise InvalidResponse('"trades" must be a list')
    out, seen = [], set()
    for tr in trades:
        if not isinstance(tr, dict):
            raise InvalidResponse("each trade must be an object")
        s = str(tr.get("stock", "")).strip().upper().removeprefix("STOCK").strip()
        a = str(tr.get("action", "")).strip().lower()
        sh = tr.get("shares", 0)
        if s not in labels:
            raise InvalidResponse(f"unknown stock {tr.get('stock')!r}")
        if s in seen:
            raise InvalidResponse(f"stock {s} listed twice")
        if a not in ("sell", "hold", "buy"):
            raise InvalidResponse(f"invalid action {tr.get('action')!r}")
        if isinstance(sh, bool) or not isinstance(sh, (int, float)) or sh < 0 or sh != int(sh):
            raise InvalidResponse(f"shares must be a non-negative integer, got {sh!r}")
        seen.add(s)
        out.append({"stock": s, "action": a, "shares": int(sh) if a != "hold" else 0})
    # Stocks the model omitted are treated as "hold".
    out += [{"stock": s, "action": "hold", "shares": 0} for s in labels if s not in seen]
    return {"trades": out, "reason": str(obj.get("reason", ""))}


def parse_lessons(text: str) -> list[str]:
    obj = extract_json(text)
    les = obj.get("lessons")
    if not (isinstance(les, list) and len(les) == 3 and all(isinstance(x, str) for x in les)):
        raise InvalidResponse('"lessons" must be a list of exactly 3 strings')
    return les


class Agent:
    def __init__(self, persona: str, memory_mode: str, llm, labels: list[str],
                 reflect_every: int = 5, max_retries: int = 3, logger=None):
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        self.persona, self.memory_mode, self.llm = persona, memory_mode, llm
        self.labels, self.reflect_every, self.max_retries = labels, reflect_every, max_retries
        self.log = logger or (lambda rec: None)
        self.history: list[str] = []      # raw per-period log ("log" shows it; "reflect" summarizes it)
        self.lessons: list[str] = []      # "reflect" memory
        self.json_failures = 0
        self.system = SYSTEM_TEMPLATE.format(
            persona=persona, n=len(labels), labels=", ".join(f"Stock {s}" for s in labels))
        self.reflect_system = REFLECT_SYSTEM_TEMPLATE.format(
            persona=persona, n=len(labels), labels=", ".join(f"Stock {s}" for s in labels))

    # -- prompting ---------------------------------------------------------
    def memory_text(self) -> str | None:
        if self.memory_mode == "log" and self.history:
            return "Your record of previous periods:\n" + "\n".join(self.history)
        if self.memory_mode == "reflect" and self.lessons:
            return "Lessons you wrote from your own experience so far:\n" + \
                "\n".join(f"{i}. {x}" for i, x in enumerate(self.lessons, 1))
        return None

    def render_state(self, state: dict) -> str:
        parts = [f"Period {state['period']} of {state['total_periods']}.",
                 "Your account:\n```json\n" + json.dumps(state, indent=2) + "\n```",
                 "(purchase_price is your average cost per share; unrealized_pnl_pct is "
                 "(current_price / purchase_price - 1) * 100, null if you hold no shares.)"]
        mem = self.memory_text()
        if mem:
            parts.append(mem)
        parts.append("What do you do this period? Reply with the JSON object only.")
        return "\n\n".join(parts)

    def _ask(self, kind: str, period: int, system: str, prompt: str, parser):
        """Call the LLM, retrying with the error message appended if the reply is invalid."""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_retries + 1):
            res = self.llm.complete(system, messages)
            rec = {"type": kind, "period": period, "attempt": attempt, "system": system,
                   "messages": messages, "response": res["text"], "cached": res["cached"],
                   "cache_key": res["key"], "usage": res.get("usage", {})}
            try:
                parsed = parser(res["text"])
                self.log({**rec, "valid": True, "parsed": parsed})
                return parsed, attempt
            except InvalidResponse as e:
                self.json_failures += 1
                self.log({**rec, "valid": False, "error": str(e)})
                messages = messages + [
                    {"role": "assistant", "content": res["text"] or "(empty)"},
                    {"role": "user", "content": f"Your reply was invalid: {e}. "
                     "Reply again with ONLY the JSON object in the required format."}]
        return None, self.max_retries + 1

    def decide(self, state: dict) -> dict:
        """Return {"trades": [...], "reason": str}. Falls back to all-hold if every retry fails."""
        parsed, _ = self._ask("decision", state["period"], self.system, self.render_state(state),
                              lambda t: parse_decision(t, self.labels))
        if parsed is None:
            return {"trades": [{"stock": s, "action": "hold", "shares": 0} for s in self.labels],
                    "reason": "FALLBACK: no valid response", "fallback": True}
        return parsed

    # -- memory ------------------------------------------------------------
    def record(self, period: int, total_periods: int, executed: list[dict],
               value_before: float, value_after_prev: float | None, realized: list[str]):
        if self.memory_mode == "none":
            return
        acts = ", ".join(f"{t['action']} {t['shares']} {t['stock']} @ {t['price']:.2f}"
                         for t in executed if t["action"] != "hold") or "no trades"
        line = f"Period {period}: portfolio value {value_before:,.2f}"
        if value_after_prev is not None:
            line += f" ({(value_before / value_after_prev - 1) * 100:+.1f}% since last period)"
        line += f"; you did: {acts}"
        if realized:
            line += "; realized: " + ", ".join(realized)
        self.history.append(line + ".")
        if (self.memory_mode == "reflect" and period % self.reflect_every == 0
                and period < total_periods):
            self.reflect(period)

    def reflect(self, period: int):
        prev = ("\nYour previous lessons were:\n" +
                "\n".join(f"- {x}" for x in self.lessons)) if self.lessons else ""
        prompt = ("Here is your record of all periods so far:\n" + "\n".join(self.history) +
                  prev + "\n\nSummarize your experience as an investor into exactly three "
                  "lessons you will keep in mind for the following periods. Reply with ONLY "
                  'a JSON object: {"lessons": ["...", "...", "..."]}')
        parsed, _ = self._ask("reflection", period, self.reflect_system, prompt, parse_lessons)
        if parsed is not None:
            self.lessons = parsed
