"""Investor personas.

Designs v1/v2: three hand-written personas (PERSONAS), kept so those designs replay.

Design v3 (prototype 2, plan §2.2): 11 binary attributes. A profile is a dict
{attribute key: 0 or 1}; `compose_persona` turns it into the persona text. Profiles are drawn
by `sample_profiles`; C0 (all attributes at level 0) and C1 (C0 plus a bias cue) are fixed.
Never add behavioral-finance vocabulary; tests/test_prompts.py checks for it.
"""
from __future__ import annotations

import functools
import json

import numpy as np

from llm_investors.paths import ROOT

PERSONAS: dict[str, str] = {
    "young_active": (
        "You are 27 years old and work as an engineer at a technology company in Hsinchu. "
        "You opened a brokerage account two years ago and invest part of your salary every "
        "month. You check your account most days and would like to grow your savings quickly "
        "enough to buy an apartment within ten years."
    ),
    "retiree": (
        "You are 66 years old and retired after a career as a secondary-school teacher in "
        "Taichung. Your savings, together with your pension, must cover your living expenses "
        "for the rest of your life, so you want your account to hold its value and avoid large "
        "swings. You have owned a few stocks over the years but never traded often."
    ),
    "experienced": (
        "You are 45 years old and have traded Taiwan-listed stocks for more than twenty years, "
        "through several market booms and crashes. Trading is a serious part-time activity for "
        "you alongside running a small business, and your goal is to earn better long-run "
        "returns than the overall market."
    ),
}


def persona_text(key: str) -> str:
    try:
        return PERSONAS[key]
    except KeyError:
        raise KeyError(f"unknown persona {key!r}; choose from {sorted(PERSONAS)}") from None


# --------------------------------------------------------------------------- design v3
CONTROL_SENTENCE = "You are an individual investor in Taiwan."

# (key, level-1 sentence, level-0 sentence or None), in the order they are composed (1 -> 11)
ATTRIBUTES: list[tuple[str, str, str | None]] = [
    ("loss_sensitivity",
     "Losing NT$1,000 bothers you about twice as much as gaining NT$1,000 pleases you.", None),
    ("mental_accounting",
     "You think of each stock as a separate account and judge it by whether it is up or down "
     "since you bought it.", None),
    ("regret_aversion", "You strongly dislike admitting that a purchase was a mistake.", None),
    ("realization_utility",
     "Locking in a gain gives you real satisfaction, and selling at a loss feels painful.", None),
    ("financial_literacy",
     "You studied finance and understand risk, return and diversification.", None),
    ("experience", "You have traded Taiwan-listed stocks for more than twenty years.", None),
    ("professional", "You manage a stock portfolio professionally for an investment firm.", None),
    ("age", "You are {older_age} years old.", "You are {younger_age} years old."),
    ("gender", "You are a woman.", "You are a man."),
    ("market_knowledge",
     "Prices here move randomly: past price changes do not predict future changes.", None),
    ("trading_activity",
     "You check your account every day and trade several times a month.",
     "You check your account a few times a year and rarely trade."),
]
ATTRIBUTE_KEYS = [a[0] for a in ATTRIBUTES]
LABELS = {
    "loss_sensitivity": "Loss sensitivity", "mental_accounting": "Mental accounting",
    "regret_aversion": "Regret aversion", "realization_utility": "Realization utility",
    "financial_literacy": "Financial literacy", "experience": "Experience",
    "professional": "Professional", "age": "Age (older)", "gender": "Gender (woman)",
    "market_knowledge": "Market knowledge", "trading_activity": "Trading activity (active)",
}
C1_CUE = ("You tend to sell stocks that have gone up quickly and hold on to stocks that have "
          "gone down.")
FIXED = ("C0", "C1")


@functools.lru_cache(maxsize=4)
def load_ages(path: str = "results/calibration/attributes.json") -> dict[str, int]:
    """{younger_age, older_age} from the calibration output (plan §2.7)."""
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `calibrate` first (the age levels come "
                                "from the TDCC investor statistics)")
    a = json.loads(p.read_text())
    return {"younger_age": int(a["younger_age"]), "older_age": int(a["older_age"])}


def control_profile() -> dict[str, int]:
    return {k: 0 for k in ATTRIBUTE_KEYS}


def compose_persona(profile: dict[str, int], ages: dict[str, int] | None = None,
                    cue: bool = False) -> str:
    """Control sentence, then each attribute's sentence in table order (1 -> 11)."""
    ages = ages or load_ages()
    parts = [CONTROL_SENTENCE]
    for key, level1, level0 in ATTRIBUTES:
        v = profile[key]
        if v not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1, got {v!r}")
        s = level1 if v else level0
        if s:
            parts.append(s.format(**ages))
    if cue:
        parts.append(C1_CUE)
    return " ".join(parts)


def sample_profiles(n: int, seed: int) -> list[dict[str, int]]:
    """n profiles, each attribute drawn independently with probability 1/2. Deterministic for
    a given seed and prefix-stable: the first k of n profiles equal sample_profiles(k, seed),
    so the main run reuses the pilot's profiles (and replays their calls from the cache)."""
    out = []
    for i in range(n):
        draws = np.random.default_rng([seed, i]).integers(0, 2, size=len(ATTRIBUTE_KEYS))
        out.append({k: int(v) for k, v in zip(ATTRIBUTE_KEYS, draws)})
    return out


def profile_id(i: int) -> str:
    return f"P{i:02d}"


def roster(n_profiles: int, seed: int, ages: dict[str, int] | None = None) -> list[dict]:
    """[{id, kind, profile, text, n_active}] for P00.. plus C0 and C1 (plan §2.4 grid)."""
    ages = ages or load_ages()
    rows = []
    for i, prof in enumerate(sample_profiles(n_profiles, seed)):
        rows.append({"id": profile_id(i), "kind": "profile", "profile": prof,
                     "text": compose_persona(prof, ages), "n_active": sum(prof.values())})
    c0 = control_profile()
    rows.append({"id": "C0", "kind": "C0", "profile": c0, "text": compose_persona(c0, ages),
                 "n_active": 0})
    rows.append({"id": "C1", "kind": "C1", "profile": c0,
                 "text": compose_persona(c0, ages, cue=True), "n_active": 0})
    return rows
