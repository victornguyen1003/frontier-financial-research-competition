"""Investor agent. The environment supplies the rules text, the state rendering and the order
constraints; persona and memory handling are the same in every design."""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_investors.llm.client import CacheMiss, FatalAPIError, IncompleteLimit

from . import prompts
from .memory import Memory
from .parsing import InvalidReply, hold, parse_decision_v3, parse_lessons, parse_orders
from .personas import persona_text


@dataclass
class EnvSpec:
    rules: str                       # environment description (system prompt, fixed)
    labels: list[str]                # stock labels, e.g. ["A", ..., "E"] or ["A"]
    order_format: str
    one_order: bool = False
    require_limit: bool = False
    reasoning: bool = False          # design v3: replies carry a required "reasoning" field
    design: str = "v2"


@dataclass
class Agent:
    agent_id: str
    persona: str                     # persona id: a PERSONAS key (v1/v2) or a profile id (v3)
    architecture: str                # none | log | reflect
    env: EnvSpec
    llm: object
    log: object                      # callable(event_type, **fields)
    json_retries: int = 1
    reflect_after: tuple = (5, 10)
    replicate_id: int | None = None
    show_pnl: bool = True            # Study 1 ablation switch (market.show_pnl)
    persona_prompt: str | None = None  # v3: composed persona text; None = PERSONAS[persona]
    memory: Memory = field(init=False)

    def __post_init__(self):
        self.memory = Memory(self.architecture, self.show_pnl, self.env.design)
        text = self.persona_prompt if self.persona_prompt is not None else \
            persona_text(self.persona)
        self.system = prompts.system_prompt(self.env.rules, text, self.env.order_format,
                                            self.replicate_id)
        self.reflect_system = prompts.reflect_system_prompt(self.env.rules, text, self.replicate_id)
        self.n_invalid = 0
        self.n_fallback = 0
        self.n_incomplete = 0
        self.last_reasoning: str | None = None

    # ------------------------------------------------------------------
    def _ask(self, kind: str, period: int, system: str, user: str, parser, dummy):
        """Call the model; on an invalid reply, retry with a correction (json_retries times).
        Returns (parsed or None, last error)."""
        messages = [{"role": "user", "content": user}]
        err = None
        for attempt in range(self.json_retries + 1):
            try:
                res = self.llm.complete(system, messages, dummy=dummy)
            except IncompleteLimit as e:
                # Never a hold: log where it happened and let the run stop.
                self.log("incomplete_stop", agent=self.agent_id, kind=kind, period=period,
                         persona=self.persona, architecture=self.architecture, message=str(e),
                         cached=getattr(e, "cached", False))
                raise
            except CacheMiss as e:
                self.log("cache_miss", agent=self.agent_id, kind=kind, period=period,
                         message=str(e))
                raise
            except FatalAPIError as e:
                # Quota / billing / other non-retryable error: log it; the runner stops the study.
                self.log("fatal_api_error", agent=self.agent_id, kind=kind, period=period,
                         persona=self.persona, architecture=self.architecture,
                         error=e.to_dict())
                raise
            self.n_incomplete += res.get("incomplete_retries", 0)
            try:
                parsed, valid, err = parser(res["text"]), True, None
            except InvalidReply as e:
                parsed, valid, err = None, False, str(e)
                self.n_invalid += 1
            self.log("llm_call", agent=self.agent_id, kind=kind, period=period, attempt=attempt,
                     system=system, messages=messages, raw=res["text"], status=res["status"],
                     parsed=parsed, valid=valid, error=err, usage=res["usage"],
                     latency_s=res["latency_s"], cost_usd=res["cost_usd"],
                     api_retries=res["api_retries"], cached=res["cached"], cache_key=res["key"],
                     incomplete_retries=res.get("incomplete_retries", 0),
                     incomplete_usage=res.get("incomplete_usage") or {})
            if valid:
                return parsed, None
            messages = messages + [{"role": "assistant", "content": res["text"] or "(empty)"},
                                   {"role": "user", "content": prompts.correction_message(err)}]
        return None, err

    def decide(self, period: int, state_block: str, dummy=None, check=None) -> list[dict]:
        """Return one order per stock label. `check(orders)` may raise InvalidReply for
        account-dependent rules; it triggers the same correction retry. Falls back to hold
        (logged) if every reply is invalid. Design v3 also sets `last_reasoning`."""
        user = prompts.decision_user_prompt(self.memory.block(), state_block)
        self.last_reasoning = None

        def parser(t):
            if self.env.reasoning:
                out = parse_decision_v3(t, self.env.labels)
                orders = out["orders"]
            else:
                out = orders = parse_orders(t, self.env.labels, self.env.one_order,
                                            self.env.require_limit)
            if check:
                check(orders)
            return out
        parsed, err = self._ask("decision", period, self.system, user, parser, dummy)
        if parsed is not None and self.env.reasoning:
            self.last_reasoning = parsed["reasoning"]
            parsed = parsed["orders"]
        if parsed is None:
            self.n_fallback += 1
            self.log("fallback", agent=self.agent_id, kind="decision", period=period,
                     persona=self.persona, architecture=self.architecture, error=err)
            return [hold(s, "FALLBACK") for s in self.env.labels]
        return parsed

    def end_period(self, period: int, record: dict, last_period: int, dummy_lessons=None):
        """Store the factual record; run a reflection when scheduled (reflect only)."""
        self.memory.add(record)
        if (self.architecture == "reflect" and period in self.reflect_after
                and period < last_period):
            self.reflect(period, dummy_lessons)

    def reflect(self, period: int, dummy=None):
        user = prompts.reflect_user_prompt(self.memory.history, self.memory.lessons)
        parsed, err = self._ask("reflection", period, self.reflect_system, user, parse_lessons,
                                dummy)
        if parsed is None:
            self.n_fallback += 1
            self.log("fallback", agent=self.agent_id, kind="reflection", period=period,
                     persona=self.persona, architecture=self.architecture, error=err,
                     kept_previous_lessons=bool(self.memory.lessons))
            return
        self.memory.lessons = parsed
        self.log("reflection", agent=self.agent_id, period=period, persona=self.persona,
                 lessons=parsed)
