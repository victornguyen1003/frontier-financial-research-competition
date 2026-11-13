# Pre-spend audit: LLM investor agents (Study 1 + Study 2)

> **Status update:** findings 1 (cache race) and 2 (run order) have since been fixed. See [audit-02](audit-02-replay-2026-11-13.md).

## Context
Victor asked for an audit of the codebase and method before any paid OpenAI run, with no code changes, and asked specifically whether the findings visualization and free cache replay already exist. I read every module in `src/llm_investors/`, the configs, and `victor_proposal_2.pdf`. Then I ran things with every output sent to the scratchpad (`LLM_INVESTORS_ROOT`), so the repo is unchanged:
- `pytest`: **82 passed**.
- Study 1 dummy (900 runs) + `analyze-study1`, Study 2 dummy (20 markets) + `analyze-study2`: all complete; every figure and table is produced.
- Dry runs: Study 1 worst case is **$3.15** (cap $4); Study 2 worst case is **$2.12** (cap $6).
- `cache/` in the repo is empty, so nothing has been spent yet.
- Config prices ($0.05 / $0.25 per 1M tokens, Flex) match third-party listings. I could not confirm them on OpenAI's own page.

## Verdict
The engineering is sound. Accounting, the auction, conservation checks, the cost cap, the circuit breaker, logging and the dummy pipeline all work. **Running the Study 1 pilot is safe.** Before the full runs there is one replay bug to fix, and several method choices to decide on purpose, because each one changes the cache keys and would cost money to change later.

## Already implemented (answers to the specific questions)
- **Findings visualization: yes.**
  - Study 1 produces `de_by_architecture.png` (DE with bootstrap CIs by architecture × persona), `paths_<persona>.png` (price paths with each arm's sales marked), the regression text, `de_bootstrap.csv`, `de_differences.csv` and the Friedman test. The Findings page in `app/streamlit_app.py` shows all of them.
  - Study 2 produces `price_vs_fv.png`, `outcomes_by_cell.png`, `actions_heatmap.png`, `outcomes_by_cell.csv`, `agent_de.csv` and the stylized facts, all shown on the same page.
  - Gaps: PGR and PLR are shown as pooled point estimates with no CI (the proposal asks for CIs on PGR/PLR/DE). The action heatmap covers only the first log alphabetically (`baseline_r00`).
- **Free replay: yes.**
  - API replay: `llm/cache.py` and `LLMClient.key` cache every completed reply under a hash of model, effort, max tokens, JSON mode, system prompt and messages. A cache hit costs $0 and is not charged.
  - Viewer replay: the Live view has a period slider and a prompt/raw-response inspector, and it only reads files.
  - **One exception, finding 1 below:** replay is not guaranteed when identical prompts run at the same time.

## Findings (most important first)

### Fix before the full runs
1. **A cache race breaks "replay at $0" and makes paired draws depend on timing.** (`sim/run_study1.py:128`, `llm/client.py:128-163`)
   - Several arms send the same prompt. The `none`, `log` and `reflect` arms of the same persona × scenario all send an identical period-1 prompt, and `reflect` matches `none` through period 5. Neither the memory block nor the architecture is part of the prompt.
   - The grid is ordered persona-major, so the pilot's first 16 workers run `young_active` none/log/reflect s0–s4 *at the same time*. Each one misses the cache, pays, and writes its own reply, and the last write wins.
   - On replay, all three arms read the same reply. The arms whose reply was overwritten follow a different path from what was logged, so their later calls are new paid calls and the pilot logs cannot be reproduced.
   - In the full run, arms happen to share draws only because the grid order puts them about 200 runs apart. In Study 2 they share only because of the order with `workers: 2`, which `--workers` can break.
   - Fix (pick one): add an in-flight lock per cache key in `LLMClient.complete`, so the second caller waits and then reads the cache; or order the grid so `none` runs before the other arms. Either way, document the sharing as deliberate common random numbers.
2. **If the budget stops a run, whole cells go missing.** The Study 1 grid is persona → architecture → scenario, so a `budget_stop` aborts the *last* cells (for example `experienced × reflect`) and the paired design loses balance. Ordering the grid scenario-major means a stop removes complete matched sets instead. The risk is low at a $3.15 worst case against a $4 cap, but incompletes or a wrong price would raise it.
3. **Cluster SEs by matched set, not by run.** (`analysis/study1.py:52`)
   - The three arms of a persona × scenario share prices, and some of their replies are literally identical (finding 1), so their errors are correlated.
   - Clustering on `persona_scenario`, which contains the runs, is the honest choice. The viewer caption also says "clustered by run".

### Method decisions to make before paying (each one changes the prompts)
4. **The prompt itself shows the reference point in every arm.**
   - `envs/exogenous.py:51-56` shows a "gain/loss vs average cost" column, and the `log` history adds "% vs your average cost" on sales.
   - This is realistic, since broker apps show it. But it may push DE close to a ceiling in all arms and hide any memory effect.
   - Decide deliberately. One option is a small ablation without the column, about $0.15 at the pilot size.
5. **The final period may distort DE.** "Period t of 15" tells agents when the game ends, and selling everything in period 15 inflates both PGR and PLR. Add a robustness check that drops period 15 (analysis only, free).
6. **The `log` arm never sees price history.** History lines carry account value and own trades only (`agents/memory.py:22`). This matches the proposal's wording, but "memory" here means the agent's own trading record, not market memory. State that in the write-up.
7. **Missing stocks become silent holds.** `parse_orders` fills in a hold for any omitted stock without flagging it (`agents/parsing.py:84`). The proposal specifically warns that fake holds inflate DE. At minimum, log how often stocks are omitted per condition, or treat an omission as invalid.
8. **Dropped matched sets select the sample.** `matched_sets` drops any set where DE is undefined in *any* arm. That is fine for the dummy data (300/300 kept). Real agents that rarely sell could lose many sets in an uneven way, so report `n_sets` next to the pooled DE.
9. **The `ret` control changes what the coefficient means.** With `ret` in the model, `gain × arch` measures a jump at zero return rather than the full gap between winners and losers. Report a specification without `ret` too. The logit with 300 fixed-effect dummies may fail on separation; it is wrapped in try/except, so check that its output reads "failed" rather than disappearing silently.

### Study 2 (lower stakes, but the analysis is thin)
10. **3 runs per cell is too few for bootstrap CIs.** A percentile CI from n = 3 is close to meaningless, and nothing compares a cell against baseline. Replicates r0–r2 share dividends and endowments, but the analysis never uses that pairing. Add paired differences by replicate and present the results as descriptive.
11. **Two cells match others for the first periods.**
    - `memory_reflect` is identical to `memory_none` through period 5 (same prompts, replayed from cache), so the memory treatment acts on only 10 of 15 periods.
    - `baseline` and `memory_none` share period 1.
    - The dummy run cannot check whether persona, memory or tax treatments are wired in, because the dummy policy ignores them (four cells came out identical).
12. **In-market DE is not comparable with Study 1.** Endowed shares are booked at FV₁ = 360 while FV falls to 24, so most positions are structural paper losses. Gain/loss also uses the clearing price, which the agent had not seen when it decided. Report it, but don't line it up next to Study 1's DE as like for like.
13. **Showing FV every period may suppress bubbles.** This follows the proposal, but it may produce pricing at FV (a null result). Read the pilot's price paths before spending on 20 markets.
14. **The Study 2 cost projection leaves out JSON retries**, unlike Study 1 (`_retry_rate`). Its pilot uses T = 10 prompts, which never replay in the full run. Both effects are small.

### Operational notes (no change needed)
- Any connection drop or HTTP 500/502 stops the whole study, as you designed it. The cache makes a resume free, so expect one or two resumes over a run lasting hours.
- A client-side timeout may still be billed by OpenAI without being tracked. With a 900 s timeout this is rare; the dashboard limit covers it.
- Running `--stage full` deletes the pilot logs and results from `logs/study1`, because they share a directory. Copy them aside if the report needs pilot numbers.
- After the pilot, check that `usage.service_tier` in the logs reads `flex`, and confirm the Flex price on OpenAI's pricing page. The hard cap is computed from the config prices, so it protects you only if they are right.
- Personas still need grounding in TWSE investor statistics (an open item in the README).

## Recommended order
1. Decide on findings 4, 5 and 7, because they change the prompts or parser. Changes 1–3 and 9 are harness or analysis changes and are free to make later, apart from finding 1, which should be fixed before any paid call.
2. Run `run-study1 --stage pilot`. Inspect the incomplete and invalid rates, the omitted stocks, `service_tier`, and the projected cost.
3. Study 1 full run, then the Study 2 pilot. Read the price paths before the Study 2 full run.

## Verification after any fixes
- Run `pytest -q`, plus a new test where two threads request the same key and one API call is expected.
- Run the dummy full runs for both studies with `LLM_INVESTORS_ROOT` set to a temp directory.
- Run the pilot twice and check that the second run reports 0 API calls and logs identical to the first.

---

## Resolution (2026-11-13)

Each finding was checked against the code and confirmed. Victor decided findings 4 and 7;
everything else is a harness, analysis or documentation change. Default prompts and cache keys
are unchanged (a test pins the exact state prompt), so no paid work is invalidated.
**`pytest`: 86 passed.**

| # | Status | What changed |
|---|---|---|
| 1 | Fixed | `LLMClient.complete` is single-flight per cache key. Concurrent identical prompts make one paid call; the others wait and read the cache. Sharing a draw across arms is documented as deliberate common random numbers (README, "Replay from cache"). Test: 4 threads, 1 call. End-to-end with a fake API and 16 workers: pilot run 1 made 615 calls, run 2 made **0**, and all 45 logs were identical. |
| 2 | Fixed | The Study 1 grid is scenario-major, so a budget stop removes whole matched sets, not cells. |
| 3 | Fixed | All Study 1 regressions cluster by matched set (persona × scenario); viewer caption updated. This departs from the proposal ("by run") and is documented. |
| 4 | Decided: keep, plus ablation | The P&L column stays in the main design. `market.show_pnl: false` / `run-study1 --no-pnl` hides the gain/loss column and the "% vs cost" in log history; outputs go to `study1_nopnl*`. |
| 5 | Fixed | New regression without the final period, and pooled PGR/PLR/DE with CIs excluding it (`de_pooled.csv`). |
| 6 | Documented | `agents/memory.py` docstring and README: `log` is the agent's own trading record, not market memory. |
| 7 | Decided: invalid | A missing stock raises `InvalidReply` naming it, so the reply gets one corrected retry, then a logged fallback; never a silent hold. Counted as `omitted_stock_replies` per run and on the Observability page. |
| 8 | Fixed | `de_bootstrap.csv` adds `n_sets_dropped` and `n_runs_pooled`. |
| 9 | Fixed | Added an LPM without `ret`. A failed or non-converged logit is printed as FAILED and listed under `_failed` in `summary.json`. |
| Gap | Fixed | Pooled PGR and PLR now have cluster-bootstrap CIs. Study 2 draws one action heatmap per cell (first replicate), selectable in the viewer. |
| 10 | Fixed | Study 2 outcomes are descriptive (mean, sd, min, max; dot plot of every run). New `paired_vs_baseline.csv` gives cell minus baseline by replicate. |
| 11 | Documented + tested | Shared early periods are documented as common random numbers. A new test checks that persona mix, memory (log history, reflection calls, lessons), tax and price-limit treatments reach the prompts, since the dummy policy can't show it. |
| 12 | Fixed | The Study 1 column is renamed `study1_DE_allperiods_reference`, and a caveat (`AGENT_DE_NOTE`) appears in `summary.json`, the console and the viewer. |
| 13 | Documented | README open decision: read the Study 2 pilot's price paths before the full run. No design change. |
| 14 | Fixed | Both studies' projections use the measured JSON retry rate. The Study 2 pilot has its own directory. |
| Ops: pilot logs deleted by full run | Fixed | The pilot writes `logs/<study>_pilot/` and the full run `logs/<study>/`. They share the cache, so the full run replays the pilot's calls for free. |
| Ops: `service_tier` | Fixed | The pilot projection warns if any call was served on a tier other than the configured one. |
| Ops: Flex price, client-timeout billing, personas | Open | These need OpenAI's pricing page, the dashboard limit, and team input respectively. Listed in the README. |
