"""LLM client: OpenAI Responses API (Flex tier) + disk cache + cost cap + retries.

Failure policy (proposal §3, tightened before the pilot):
- retried with exponential backoff until success: timeouts, 429 rate limits, and "resource
  unavailable" (HTTP 503, or a resource_unavailable code/message, e.g. Flex out of capacity).
  Gateway / Cloudflare errors count as these two kinds: 504, 522 and 524 are timeouts; 502,
  520, 521 and 523 mean the upstream was unavailable or returned nothing (a 520 stopped the
  first full run 13 calls after a resume, with no OpenAI request id).
- fatal (FatalAPIError, never retried): quota / billing / spend-limit errors (kind "quota",
  e.g. insufficient_quota, even when sent as a 429) and every other API error (kind
  "api_error": bad request, auth, unknown model, other 5xx, connection drops). The caller logs
  it and stops the whole study run. A failed API call never becomes a default action.
- incomplete replies (status "incomplete", e.g. the output cap was hit): retried at once, like
  an API failure. They are charged and counted but never cached, parsed or turned into a hold.
  After `max_incomplete_retries` in a row on one request, IncompleteLimit is raised and the
  caller stops that run.
- malformed JSON (from a completed reply) is handled by the caller (agents/agent.py): retry
  once, then a logged hold.

Backends:
- openai: real calls.
- dummy:  rule-based agents (no API, no cost); the caller passes `dummy=callable`.
Dry run (`dry_run=True`): prompts are printed, nothing is sent, the dummy reply is used so the
simulation can proceed, and the would-be cost is projected from prompt length.

Replay is free: every completed reply is cached, and so is a request that hit the incomplete
limit (status "incomplete_limit"), so re-running an identical config reproduces the same stop
without paying for the attempts again (`retry_incomplete=True` pays to try those again).
`offline=True` (the `--replay` flag) guarantees $0: the OpenAI client is never created, no API
key is needed, and a cache miss raises CacheMiss instead of calling the API.
"""
from __future__ import annotations

import random
import re
import threading
import time

from .cache import DiskCache, request_key
from .cost import BudgetExceeded, CostTracker  # noqa: F401  (re-exported)

USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


class IncompleteLimit(RuntimeError):
    """Too many incomplete replies in a row for one request; the run must stop (never hold).
    `cached` is True when the stop was replayed from the cache at $0."""

    def __init__(self, message: str, cached: bool = False):
        super().__init__(message)
        self.cached = cached


class CacheMiss(RuntimeError):
    """Offline replay needed a reply that is not in the cache (the prompts changed)."""


class Halted(RuntimeError):
    """A fatal API error already stopped the study; no further API calls are made."""


class FatalAPIError(RuntimeError):
    """Non-retryable API error. kind: "quota" (quota, billing or spend limit) or "api_error"."""

    def __init__(self, kind: str, error: Exception):
        self.kind = kind
        self.error_type = type(error).__name__
        self.status = getattr(error, "status_code", None)
        self.code = _error_code(error)
        self.request_id = getattr(error, "request_id", None)
        self.detail = str(getattr(error, "message", None) or error)[:500]
        super().__init__(f"{kind}: {self.error_type} (status {self.status}, code {self.code}): "
                         f"{self.detail}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "error_type": self.error_type, "status": self.status,
                "code": self.code, "request_id": self.request_id, "message": self.detail}


QUOTA_CODES = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
               "budget_exceeded", "spend_limit_exceeded", "usage_limit_exceeded"}
QUOTA_WORDS = re.compile(r"quota|billing|spend(ing)? limit|budget|usage limit|credit balance",
                         re.I)
RETRY_CODES = {"rate_limit_exceeded", "resource_unavailable"}
# Timeouts: 408, 504, 522 (connection timed out), 524 (timeout at the gateway).
# Resource unavailable: 503, 502 (bad gateway), 520 (empty/unknown upstream response),
# 521 (upstream down), 523 (upstream unreachable). Rate limit: 429.
RETRY_STATUS = {408, 429, 502, 503, 504, 520, 521, 522, 523, 524}
UNAVAILABLE = re.compile(r"resource[ _]unavailable|resource_exhausted", re.I)


def classify_error(e: Exception) -> str:
    """'retry' | 'quota' | 'api_error'. Quota is checked first: OpenAI reports an exhausted
    quota as HTTP 429, which must not be mistaken for a rate limit."""
    import openai
    code = str(_error_code(e) or "")
    text = f"{code} {getattr(e, 'message', '') or e}"
    if code in QUOTA_CODES:
        return "quota"
    if code in RETRY_CODES:            # explicit codes beat keyword matching on the message
        return "retry"
    if isinstance(e, openai.APIStatusError) and QUOTA_WORDS.search(text):
        return "quota"
    if isinstance(e, openai.APITimeoutError):
        return "retry"
    if isinstance(e, openai.APIStatusError):
        if e.status_code in RETRY_STATUS or UNAVAILABLE.search(text):
            return "retry"
    return "api_error"


class LLMClient:
    def __init__(self, llm_cfg: dict, cache: DiskCache, cost: CostTracker,
                 dry_run: bool = False, show_prompts: int = 3, echo=print,
                 offline: bool = False, retry_incomplete: bool = False):
        self.cfg = llm_cfg
        self.backend = llm_cfg["backend"]
        self.cache, self.cost = cache, cost
        self.dry_run = dry_run
        self.show_prompts = show_prompts
        self.echo = echo
        self.lock = threading.Lock()
        self.projection = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._shown = 0
        self.halted = threading.Event()   # circuit breaker: set by the first fatal API error
        self._key_locks: dict[str, threading.Lock] = {}   # single-flight per request key
        self.offline = offline
        self.retry_incomplete = retry_incomplete
        self._client = None   # created on the first real API call, never during a replay

    def _api(self):
        with self.lock:
            if self._client is None:
                import os

                import openai
                from openai import OpenAI

                from llm_investors.env import ENV_FILE, load_env
                load_env()                      # key from the git-ignored .env, if present
                if not os.environ.get("OPENAI_API_KEY"):
                    # classified as a fatal api_error: the study stops with a clear report
                    raise openai.OpenAIError(f"OPENAI_API_KEY is not set: put it in {ENV_FILE} "
                                             "(see .env.example) or export it")
                # SDK retries are disabled: backoff is handled here, without an attempt limit.
                self._client = OpenAI(timeout=self.cfg.get("timeout_s", 900), max_retries=0)
            return self._client

    def key(self, system: str, messages: list[dict]) -> str:
        c = self.cfg
        return request_key(backend=self.backend, model=c["model"],
                           effort=c.get("reasoning_effort"), max_tokens=c["max_output_tokens"],
                           json_mode=c.get("json_mode", False), system=system, messages=messages)

    def complete(self, system: str, messages: list[dict], dummy=None) -> dict:
        """Return {text, status, cached, key, usage, latency_s, cost_usd, api_retries,
        incomplete_retries, incomplete_usage}. Only a completed reply is ever returned."""
        key = self.key(system, messages)
        base = {"key": key, "usage": {}, "latency_s": 0.0, "cost_usd": 0.0, "api_retries": 0,
                "incomplete_retries": 0, "incomplete_usage": {}}

        if self.dry_run:
            self._project(system, messages)
            return {**base, "text": dummy() if dummy else "", "status": "dry_run", "cached": False}
        if self.backend == "dummy":
            return {**base, "text": dummy(), "status": "dummy", "cached": False}

        hit = self._cached(key, base)
        if hit is not None:
            return hit
        if self.offline:
            raise CacheMiss(f"no cached reply for request {key[:12]}")
        # Single-flight: identical prompts sent at the same time (e.g. the none/log/reflect arms
        # in period 1) make ONE paid call; the others wait and read it from the cache. Without
        # this, each would pay and the last write would win, so a replay would not reproduce
        # the logs. Identical prompts therefore always share one draw (common random numbers).
        with self.lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            hit = self._cached(key, base)
            if hit is not None:
                return hit
            return self._fetch(key, system, messages)

    def _cached(self, key: str, base: dict) -> dict | None:
        hit = self.cache.get(key)
        if hit is None:
            return None
        status = str(hit.get("status", ""))
        if status == "incomplete_limit":
            # Replay the stop for free; offline replay never pays, whatever the flag says.
            if self.retry_incomplete and not self.offline:
                return None
            self.cost.hit()
            # Same message as the original stop, so a replay reproduces the logs exactly.
            raise IncompleteLimit(f"{hit.get('incomplete_retries')} incomplete replies in a row "
                                  f"({hit.get('last_status', 'incomplete')})", cached=True)
        if status.startswith("incomplete"):
            return None
        self.cost.hit()
        return {**base, **hit, "cached": True, "key": key, "cost_usd": 0.0,
                "latency_s": 0.0, "api_retries": 0}

    def _fetch(self, key: str, system: str, messages: list[dict]) -> dict:
        if self.halted.is_set():             # an earlier fatal error already stopped the study
            raise Halted("study stopped by a fatal API error")
        if self.cfg.get("json_mode") and not any("json" in m["content"].lower()
                                                 for m in messages if m["role"] == "user"):
            # OpenAI rejects this with a 400 (the instructions don't count). Catch it locally,
            # before any money is reserved or spent, with the same fatal stop and report.
            self.halted.set()
            raise FatalAPIError("api_error", ValueError(
                "json_mode needs the word 'json' in the input messages (not only in the system "
                "instructions); fix the prompt builder in agents/prompts.py"))
        prompt_chars = len(system) + sum(len(m["content"]) for m in messages)
        limit = self.cfg.get("max_incomplete_retries", 5)
        total_cost, api_retries, latency = 0.0, 0, 0.0
        incomplete, inc_usage = 0, {k: 0 for k in USAGE_KEYS}
        while True:
            if self.halted.is_set():
                raise Halted("study stopped by a fatal API error")
            # Every attempt, incomplete ones included, is reserved and charged separately;
            # BudgetExceeded propagates and ends the run (never a hold).
            reservation = self.cost.reserve(
                self.cost.worst_case(prompt_chars, self.cfg["max_output_tokens"]))
            usage = None
            try:
                t0 = time.monotonic()
                text, status, usage, retries = self._call_openai(system, messages)
                latency += time.monotonic() - t0
            finally:
                total_cost += self.cost.settle(reservation, usage)
            api_retries += retries
            if not status.startswith("incomplete"):
                break
            incomplete += 1
            for k in USAGE_KEYS:
                inc_usage[k] += usage.get(k, 0)
            if incomplete >= limit:
                # Cache the stop itself so an identical re-run does not pay for it again.
                self.cache.put(key, {"text": "", "status": "incomplete_limit", "usage": usage,
                                     "last_status": status, "incomplete_retries": incomplete,
                                     "incomplete_usage": inc_usage, "paid_usd": total_cost})
                raise IncompleteLimit(f"{incomplete} incomplete replies in a row ({status})")
            self.echo(f"[llm] {status}: retry {incomplete}")
        rec = {"text": text, "status": status, "usage": usage,
               "incomplete_retries": incomplete, "incomplete_usage": inc_usage,
               "paid_usd": total_cost}
        self.cache.put(key, rec)
        return {**rec, "cached": False, "key": key, "latency_s": round(latency, 3),
                "cost_usd": total_cost, "api_retries": api_retries}

    # -- OpenAI ------------------------------------------------------------
    def _call_openai(self, system: str, messages: list[dict]):
        import openai
        c = self.cfg
        kwargs = dict(model=c["model"], instructions=system,
                      input=[{"role": m["role"], "content": m["content"]} for m in messages],
                      max_output_tokens=c["max_output_tokens"], store=False)
        if c.get("reasoning_effort"):
            kwargs["reasoning"] = {"effort": c["reasoning_effort"]}
        if c.get("service_tier"):
            kwargs["service_tier"] = c["service_tier"]
        if c.get("json_mode"):
            kwargs["text"] = {"format": {"type": "json_object"}}
        # Temperature is deliberately never sent (API default).

        retries = 0
        while True:
            if self.halted.is_set():          # e.g. another thread hit a quota error mid-backoff
                raise Halted("study stopped by a fatal API error")
            try:
                resp = self._api().responses.create(**kwargs)
                break
            except openai.OpenAIError as e:
                kind = classify_error(e)
                if kind != "retry":
                    self.halted.set()         # stop all other threads before their next call
                    raise FatalAPIError(kind, e) from e
                err = e
            delay = min(300.0, 2.0 ** min(retries, 8)) * (0.5 + random.random())
            retries += 1
            self.echo(f"[llm] {type(err).__name__}: retry {retries} in {delay:.0f}s")
            time.sleep(delay)

        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens if u else 0,
            "cached_input_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
            "output_tokens": u.output_tokens if u else 0,
            "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0,
            "service_tier": getattr(resp, "service_tier", None),
            "response_id": resp.id,
        }
        status = resp.status
        if status == "incomplete" and resp.incomplete_details is not None:
            status = f"incomplete:{resp.incomplete_details.reason}"
        return resp.output_text or "", status, usage, retries

    # -- dry run -----------------------------------------------------------
    def _project(self, system: str, messages: list[dict]):
        chars = len(system) + sum(len(m["content"]) for m in messages)
        with self.lock:
            self.projection["calls"] += 1
            self.projection["input_tokens"] += chars // 4
            self.projection["output_tokens"] += self.cfg["max_output_tokens"]
            show = self._shown < self.show_prompts
            self._shown += show
        if show:
            self.echo("=" * 30 + " SYSTEM " + "=" * 30 + "\n" + system)
            for m in messages:
                self.echo("-" * 30 + f" {m['role'].upper()} " + "-" * 30 + "\n" + m["content"])
            self.echo("=" * 68)


def _error_code(e) -> str | None:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return body.get("code") or (body.get("error") or {}).get("code")
    return getattr(e, "code", None)
