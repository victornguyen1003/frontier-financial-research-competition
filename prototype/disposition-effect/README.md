# Disposition-effect prototype

LLM investor agents trade 4 anonymous GBM stocks for 15 periods. We then measure
Odean's PGR, PLR and DE = PGR − PLR.

| file | role |
|---|---|
| `config.py` | every knob: model, temperature, effort, seed, mu/sigma, endowment, grid |
| `market.py` | GBM paths, `(T+1, N)` prices; path *k* uses seed `seed + k` |
| `agent.py` | `Agent(persona, memory_mode)`, `decide(state)`, JSON validation/retry, memory |
| `llm.py` | Anthropic client + on-disk cache (`cache/<sha256 of full request>.json`) + `mock` backend |
| `sim.py` | runs the grid, executes trades, writes `runs/<run_id>.jsonl` + `runs/summary.json` |
| `metrics.py` | PGR / PLR / DE per run and pooled; `python metrics.py runs/x.jsonl` recomputes from a log |

```bash
pip install -r requirements.txt
python sim.py --backend mock          # offline check of the pipeline (no API calls)
python sim.py                         # claude-haiku-5-5, 1 persona x {none, log} x 5 paths
python sim.py --modes none log reflect --paths 10
```

Credentials: `ANTHROPIC_API_KEY` or an `ant auth login` profile.

## Design choices
- **Endowment:** $5,000 cash plus 20 shares of each stock, bought at the period-0 price
  (100). This gives the agent gains and losses to realize from period 1 on.
- **Cost basis:** the weighted-average purchase price. Sells execute before buys at the
  current price. Orders the agent can't fill (selling more than it holds, buying more
  than its cash allows) are clipped and counted in `clipped_orders`.
- **PGR/PLR:** counted only in periods with at least one sale. A partial sale counts as
  realized. Positions where price equals cost are ignored.
- **Memory:** `log` shows the full history of past periods (value, % change, trades,
  realized P&L vs. cost). `reflect` shows only the 3 lessons from the latest reflection
  (after periods 5 and 10), so periods 1–5 match `none`.
- **Invalid JSON:** the bad reply and the error go back to the model, up to 3 retries.
  Every invalid reply adds 1 to `json_failures`. If all retries fail, that period
  becomes all-hold (`fallback_periods`).
- **Cache:** the key covers backend, model, temperature, effort, max_tokens, system and
  messages, so a re-run replays exactly, retries included. Delete `cache/` to resample.
- **Sampling:** the current Messages API rejects non-default `temperature` with a 400.
  That is why `temperature=None` is the default. `seed` controls the market paths only.
  To get independent LLM draws, vary something real (more paths, or a replicate id
  added to the prompt) rather than temperature.
