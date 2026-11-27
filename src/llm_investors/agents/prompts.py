"""Prompt builder for Study 1 (designs v1-v3).

Layout (fixed content first so provider-side prompt caching can reuse the prefix):
  system  = environment rules + persona + reply format [+ replicate line]   (fixed per run)
  user    = memory block (append-only history or lessons) + current state     (variable)
Never put timestamps or run ids in prompts: the disk cache keys on the exact text.
"""
from __future__ import annotations

ORDER_FORMAT_STUDY1 = """Reply with ONLY a compact JSON object, no other text:
{{"orders":[{{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":null,"reason":"<one short sentence>"}}, ...]}}
Give exactly one entry for each of {labels}. Use quantity 0 for hold."""

# design v3 (prototype 2): a required free-text "reasoning" field before the orders.
ORDER_FORMAT_STUDY1_V3 = """Reply with ONLY a compact JSON object, no other text:
{{"reasoning":"<3-5 sentences: what you considered this month and why>",
 "orders":[{{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":null,
            "reason":"<one short sentence>"}}, ...]}}
Give exactly one entry for each of {labels}. Use quantity 0 for hold."""

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


# design v3 memory blocks (plan §3.2)
RECORD_HEADER_V3 = "Your record so far:"
LESSONS_HEADER_V3 = "Lessons from your own trading so far:"

# Phrases that must never appear in any prompt (avoid demand effects). Checked in tests.
BANNED_TERMS = ("disposition", "loss aversion", "loss-aversion", "bias", "behavioral",
                "behavioural", "prospect theory", "anchoring", "regret")
