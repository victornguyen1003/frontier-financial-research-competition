# Audit 02: is replay free?

Date: 2026-11-13. Follows [audit-01](audit-01-pre-spend-2026-11-13.md).

## Question
Can a run be replayed at $0, without spending money on the OpenAI API?

## Verdict
**Yes.** A re-run of an identical configuration makes 0 API calls, and `--replay` can never spend money. Audit-01 finding 1 (the cache race) is **fixed**: identical requests sent at the same time now share one call through a lock per cache key in `LLMClient.complete` (`src/llm_investors/llm/client.py`). The gaps below don't cost money, but they affect what the logs show.

## What changed in the code since audit-01
- **One call per identical request:** concurrent identical prompts make one paid call, and the other callers read it from the cache. Identical prompts therefore always share one draw (common random numbers).
- **`--replay` (offline mode):** it never creates the OpenAI client, needs no API key, and stops on a cache miss (`CacheMiss`, exit code 3) instead of paying. It writes to `logs/<name>__replay/` and replaces the original logs only when the replay matched completely.
- **Incomplete-limit stops are cached:** a request that hit the incomplete limit replays its stop for free. `--retry-incomplete` pays to try it again.
- **Pilot logs in their own directory:** the pilot writes to `logs/<study>_pilot/`, so the full run no longer deletes the pilot logs. Both share the cache, so the full run reuses the pilot's calls.
- **Smaller changes:**
  - Study 1 runs are now ordered scenario-major.
  - The cost projection includes the JSON retry rate.
  - The pilot warns if any call was served on a tier other than Flex.
  - A `--no-pnl` ablation variant was added.

## Method
- **Fake API:** I replaced `openai.OpenAI` with a fake whose replies are random on every call, like sampling at the default temperature. Each call took 20 ms, so concurrent runs overlapped.
- **Isolation:** all outputs went to a scratchpad directory (`LLM_INVESTORS_ROOT`); the repo's `logs/`, `cache/` and `results/` were not touched.
- **Proof that replay never touches the API:** in the `--replay` checks, the API key was unset and building the client raised an error.

## Results

| Check | API calls | Result |
|---|---|---|
| Study 1 pilot (5 scenarios), first run | 615 | 90 shared prompts (log period 1, reflect periods 1–5) were paid once and read from the cache |
| Same command again | **0** | 705 cache hits |
| `--replay`, no API key, client blocked from being created | **0** | Completed; `spend.json` unchanged |
| Full stage with 10 scenarios, after the pilot | 615 | The 5 piloted scenarios were reused |
| Budget stop (cap set just above current spend, 20 scenarios) | 102 | 1 run `budget_stop`, 89 `aborted` |
| `--replay` of the stopped dataset | **0** | Exit code 3, "REPLAY INCOMPLETE"; original logs unchanged |
| Cap raised, normal run again | 1,128 | Exactly the remaining calls (1,230 − 102); nothing paid twice |
| `--replay` after the resume | **0** | Completed |
| Study 2 full (20 markets, `--workers 6`), first run | 8,586 | 594 prompts shared across cells were paid once |
| Same command again | **0** | |
| `--replay`, no API key | **0** | All 20 markets had price and volume paths identical to the paid run |

- **Free steps:** `measure/`, `analysis/` and `app/` never import the API client, so `build-dataset`, `analyze-*` and the viewer cost nothing.
- **Tests:** `pytest`: 89 passed.

## Remaining gaps (none of them cost money)
1. **Replayed logs record $0.** Every replayed `llm_call` has `cost_usd = 0`, `latency_s = 0`, `api_retries = 0` and `cached = true`; token counts are kept. A plain re-run also rewrites the logs this way. After a replay, the logs and `runs.csv` no longer show what was paid, and only `cache/<study>/spend.json` keeps the total. **Action:** back up `logs/` after each paid run, or change the replay to keep the original cost fields.
2. **`--replay` can't reproduce a run that was stopped.** A run ended by a budget stop, a fatal error or an abort has no cached reply for its next call, so `--replay` reports a miss. It does this safely, at $0 and without touching the logs. To finish such a run, use the normal command; it pays only for the calls still missing.
3. **A budget stop exits with code 0.** Only fatal API errors (code 2) and replay misses (code 3) give a non-zero exit, so a script would treat a budget stop as a success.
4. **Calls are shared only within one process.** Two terminals running the same command at the same time can still each pay for the same prompt and overwrite each other's cache entries. Don't run identical commands in parallel.
5. **Everything depends on `cache/`.** It is git-ignored, and losing it makes replays cost money again. Back it up together with `logs/`.

## Recommendation
- Use `--replay` whenever a re-run must cost $0. A plain re-run is also free if nothing changed, but if a prompt has changed it pays instead of stopping.
- Gaps 1 and 3 are small code changes worth making before the paid runs. Gaps 2, 4 and 5 need only the operating rules above.
