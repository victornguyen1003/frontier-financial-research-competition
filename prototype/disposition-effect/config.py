"""Run configuration. Every knob for an experiment lives here."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- LLM ---
    backend: str = "anthropic"          # "anthropic" | "mock" (offline, rule-based, for testing the pipeline)
    model: str = "claude-haiku-5-5"     # cheap model
    # NOTE: the current Messages API (2026-10-21) rejects non-default sampling params
    # (temperature/top_p/top_k) with a 400. Leave as None unless you pin an older API
    # version / model that accepts it. It is still part of the cache key.
    temperature: float | None = None
    effort: str | None = "low"          # output_config.effort; None = model default
    max_tokens: int = 2048
    max_json_retries: int = 3           # extra attempts after an invalid response

    # --- Market ---
    seed: int = 42                      # base seed; path k uses seed + k
    n_stocks: int = 4
    T: int = 15                         # number of decision periods
    mu: float | list[float] = 0.0       # per-period drift (scalar or one per stock)
    sigma: float | list[float] = 0.08   # per-period volatility (scalar or one per stock)
    s0: float = 100.0                   # initial price for every stock

    # --- Portfolio ---
    initial_cash: float = 5000.0
    initial_shares: int = 20            # shares of every stock bought at period-0 price

    # --- Experiment grid ---
    personas: list[str] = field(default_factory=lambda: [
        "35-year-old Taiwanese retail investor, moderate risk tolerance",
    ])
    memory_modes: list[str] = field(default_factory=lambda: ["none", "log"])
    n_paths: int = 5
    reflect_every: int = 5
    workers: int = 4                    # runs executed in parallel

    # --- I/O ---
    cache_dir: Path = HERE / "cache"
    out_dir: Path = HERE / "runs"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["out_dir"] = str(self.out_dir)
        return d
