"""Parse and validate model replies. Raises InvalidReply with a message the model can act on."""
from __future__ import annotations

import json
import math
import re


class InvalidReply(ValueError):
    pass


def extract_json(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        raise InvalidReply("empty reply")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise InvalidReply("no JSON object found") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise InvalidReply(f"invalid JSON ({e.msg})") from None
    if not isinstance(obj, dict):
        raise InvalidReply("the top-level JSON value must be an object")
    return obj


def _label(raw) -> str:
    return str(raw or "").strip().upper().removeprefix("STOCK").strip()


def _int(v, field: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
            or v < 0 or v != int(v):
        raise InvalidReply(f"{field} must be a non-negative integer, got {v!r}")
    return int(v)


def parse_orders(text: str, labels: list[str], one_order: bool = False,
                 require_limit: bool = False) -> list[dict]:
    """Return one normalized order per label: {stock, action, quantity, limit_price, reason}.

    labels = A..E: every stock needs an entry (a missing stock is an InvalidReply).
    one_order / require_limit: single-stock, limit-order variant (kept for the parser tests).
    """
    obj = extract_json(text)
    orders = obj.get("orders")
    if not isinstance(orders, list) or not orders:
        raise InvalidReply('"orders" must be a non-empty list')
    if one_order and len(orders) != 1:
        raise InvalidReply("give exactly one order")
    out, seen = [], set()
    for o in orders:
        if not isinstance(o, dict):
            raise InvalidReply("each order must be a JSON object")
        s = _label(o.get("stock")) if not one_order or o.get("stock") else labels[0]
        if s not in labels:
            raise InvalidReply(f"unknown stock {o.get('stock')!r}")
        if s in seen:
            raise InvalidReply(f"stock {s} appears twice")
        a = str(o.get("action", "")).strip().lower()
        if a not in ("buy", "sell", "hold"):
            raise InvalidReply(f"action must be buy, sell or hold, got {o.get('action')!r}")
        q = 0 if a == "hold" else _int(o.get("quantity", 0), "quantity")
        lp = o.get("limit_price")
        if a != "hold" and require_limit:
            if isinstance(lp, bool) or not isinstance(lp, (int, float)) \
                    or not math.isfinite(lp) or lp <= 0:
                raise InvalidReply(f"limit_price must be a positive number for {a}, got {lp!r}")
            lp = float(lp)
        else:
            lp = float(lp) if isinstance(lp, (int, float)) and not isinstance(lp, bool) else None
        if a != "hold" and q == 0:
            a = "hold"
        seen.add(s)
        out.append({"stock": s, "action": a, "quantity": q,
                    "limit_price": lp if a != "hold" else None,
                    "reason": str(o.get("reason", ""))[:300]})
    missing = [s for s in labels if s not in seen]
    if missing:
        # Never fill in a hold: a silent hold would inflate the disposition effect.
        raise InvalidReply("missing an entry for " + ", ".join(f"Stock {s}" for s in missing)
                           + f"; give exactly one entry for each of {', '.join(labels)}")
    order_ix = {s: i for i, s in enumerate(labels)}
    return sorted(out, key=lambda o: order_ix[o["stock"]])


MAX_REASONING_CHARS = 1000


def parse_reasoning(obj: dict) -> str:
    """Design v3: "reasoning" is required, non-empty and at most 1,000 characters."""
    r = obj.get("reasoning")
    if not isinstance(r, str) or not r.strip():
        raise InvalidReply('"reasoning" must be a non-empty string (3-5 sentences)')
    r = r.strip()
    if len(r) > MAX_REASONING_CHARS:
        raise InvalidReply(f'"reasoning" is {len(r)} characters; keep it to 3-5 sentences '
                           f"(at most {MAX_REASONING_CHARS} characters)")
    return r


def parse_decision_v3(text: str, labels: list[str]) -> dict:
    """{"reasoning": str, "orders": [...]}: the orders are validated exactly as in v1/v2."""
    obj = extract_json(text)
    reasoning = parse_reasoning(obj)
    return {"reasoning": reasoning, "orders": parse_orders(text, labels)}


def parse_lessons(text: str) -> list[str]:
    obj = extract_json(text)
    les = obj.get("lessons")
    if not (isinstance(les, list) and len(les) == 3
            and all(isinstance(x, str) and x.strip() for x in les)):
        raise InvalidReply('"lessons" must be a list of exactly 3 non-empty strings')
    return [x.strip()[:500] for x in les]


def hold(stock: str, reason: str) -> dict:
    return {"stock": stock, "action": "hold", "quantity": 0, "limit_price": None, "reason": reason}
