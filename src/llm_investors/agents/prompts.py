"""Prompt builder shared by both studies.

Layout (fixed content first so provider-side prompt caching can reuse the prefix):
  system  = environment rules + persona + reply format [+ replicate line]   (fixed per run)
  user    = memory block (append-only history or lessons) + current state     (variable)
Never put timestamps or run ids in prompts: the disk cache keys on the exact text.
"""
from __future__ import annotations

ORDER_FORMAT_STUDY1 = """Reply with ONLY a compact JSON object, no other text:
{{"orders":[{{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":null,"reason":"<one short sentence>"}}, ...]}}
Give exactly one entry for each of {labels}. Use quantity 0 for hold."""

ORDER_FORMAT_STUDY2 = """Reply with ONLY a compact JSON object, no other text:
{{"orders":[{{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":<number or null>,"reason":"<one short sentence>"}}]}}
Give exactly one entry. For buy or sell, limit_price is the highest price you will pay \
(buy) or the lowest price you will accept (sell). For hold, use quantity 0 and limit_price null."""

# Study 2 design v2: a two-sided quote every period (a buy and a sell order), so the auction
# always has both sides of the market. Design v1 (one order, ORDER_FORMAT_STUDY2) let every
# agent offer to sell with nobody bidding, and the price froze.
QUOTE_FORMAT_STUDY2 = """Reply with ONLY a compact JSON object, no other text:
{"orders":[{"stock":"A","action":"buy","quantity":<int>,"limit_price":<number>,"reason":"<one short sentence>"},{"stock":"A","action":"sell","quantity":<int>,"limit_price":<number>,"reason":"<one short sentence>"}]}
Give exactly two entries: one buy and one sell. The buy limit_price is the highest price you \
will pay; the sell limit_price is the lowest price you will accept and must be higher than \
your buy limit_price. Each quantity must be at least 1, except: use sell quantity 0 if you own \
no shares, and buy quantity 0 if your cash cannot pay for one share at your buy limit_price. \
An order executes only if the auction price is at or below your buy limit or at or above your \
sell limit."""

REFLECT_FORMAT = """Reply with ONLY a compact JSON object, no other text:
{"lessons":["<lesson 1>","<lesson 2>","<lesson 3>"]}"""


def system_prompt(env_rules: str, persona: str, order_format: str,
                  replicate_id: int | None = None) -> str:
    parts = [env_rules.strip(), "Who you are: " + persona.strip(), order_format.strip()]
    if replicate_id is not None:
        parts.append(f"(Session {replicate_id}.)")
    return "\n\n".join(parts)


def reflect_system_prompt(env_rules: str, persona: str, replicate_id: int | None = None) -> str:
    parts = [env_rules.strip(), "Who you are: " + persona.strip(),
             "You are now looking back on your own trading so far.", REFLECT_FORMAT]
    if replicate_id is not None:
        parts.append(f"(Session {replicate_id}.)")
    return "\n\n".join(parts)


def decision_user_prompt(memory_block: str | None, state_block: str) -> str:
    parts = []
    if memory_block:
        parts.append(memory_block)
    parts.append(state_block)
    parts.append("What do you do this period? Reply with the JSON object only.")
    return "\n\n".join(parts)


def reflect_user_prompt(history_lines: list[str], previous_lessons: list[str]) -> str:
    parts = ["Your record of all periods so far:\n" + "\n".join(history_lines)]
    if previous_lessons:
        parts.append("Lessons you wrote earlier:\n" +
                     "\n".join(f"- {x}" for x in previous_lessons))
    parts.append("Write exactly three lessons from your own experience that you will keep "
                 "in mind when you decide in the coming periods. Reply with the JSON object "
                 "only.")
    # The user message itself must mention JSON: OpenAI's json_object mode rejects input
    # messages without the word (the system instructions do not count).
    return "\n\n".join(parts)


def correction_message(error: str) -> str:
    return (f"Your reply could not be used: {error}. "
            "Reply again with ONLY the JSON object in the required format.")


# Phrases that must never appear in any prompt (avoid demand effects). Checked in tests.
BANNED_TERMS = ("disposition", "loss aversion", "loss-aversion", "bias", "behavioral",
                "behavioural", "prospect theory", "anchoring", "regret")
