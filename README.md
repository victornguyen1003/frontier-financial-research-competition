# LLM investor agents, memory, and market outcomes

Prototype for the Frontier Financial Research Competition (theme: Generative AI and Behavioral
Finance). It implements `assets/proposals/victor_proposal_2.pdf`.

- **Study 1 (micro):** agents trade 5 anonymized stocks on exogenous price paths. Does the
  memory architecture (`none` / `log` / `reflect`) change the disposition effect (DE)?
- **Study 2 (macro):** the same agents trade one stock with each other in a call-auction market
  (Smith–Suchanek–Williams design). Do bubbles, volume and volatility change with persona mix, memory, a ±10% price limit, or a 0.3% transaction tax?

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # pinned versions used for the runs
pip install -e .                 # installs the llm_investors package and the commands below
cp .env.example .env && chmod 600 .env   # then put your key in .env (only needed for paid runs)
pytest -q                        # 96 tests, no API calls
```

**The OpenAI key** goes in `.env` at the repository root: `OPENAI_API_KEY=sk-...`.

- `.env` is git-ignored; `.env.example` is the committed template.
- A key exported in the shell takes precedence over the file.
- Paid runs check for a key before starting and stop with instructions if there isn't one.
- `--dry-run`, `--replay` and `--backend dummy` never need a key.
- The key is never printed or logged.

Also set an account spend limit in the OpenAI dashboard. The code stops itself at each
study's cap (US$4 / US$6), but the dashboard limit is the last line of defense.

## Build order and commands

| Step | Command                                             | Exit check                                                                                                                                                    |
| ---- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `run-study1 --backend dummy --stage full`         | accounting and PGR/PLR tests pass (`pytest`)                                                                                                                |
| –   | `run-study1 --dry-run`                            | prints prompts and an upper-bound cost; no API calls                                                                                                          |
| 2    | `run-study1 --stage pilot`                        | 5 scenarios × 3 × 3, written to `logs/study1_pilot/`. Prints measured tokens, the projected full cost and any service-tier mismatch, then **stops** |
| 3    | `run-study1 --stage full`                         | asks for confirmation against the pilot projection; replays the pilot's calls from cache                                                                      |
|      | `build-dataset study1 && analyze-study1`          | regressions, bootstrap CIs, figures                                                                                                                           |
| 4    | `run-study2 --backend dummy --stage full`         | auction and conservation asserts hold every period                                                                                                            |
| 5    | `run-study2 --stage pilot`, then `--stage full` | 10 agents × 10 periods (`logs/study2_v2_pilot/`), then 20 market runs. Check the pilot's `no_trade_share` and price path first (see "Study 2 order design") |
|      | `build-dataset study2_v2 && analyze-study2 study2_v2` | outcomes by treatment, figures (the name is `study2_<design>`; design v1 is plain `study2`)                                                            |

Useful flags: `--no-live` turns off the live terminal table, `--workers N` sets parallelism,
`--yes` skips the confirmation, `--scenarios N` overrides the Study 1 stage size, and
`run-study1 --no-pnl` runs the P&L-column ablation (see below).

Output directories: the full stage writes `logs/<study>/` and `results/<study>/`; the pilot
writes `<study>_pilot`, so a full run never deletes the pilot's logs. Dummy and dry runs add
`_dummy` / `_dryrun`, and the ablation adds `_nopnl` (e.g. `study1_nopnl_pilot`), so nothing
mixes with the main data.

The run commands also build the datasets themselves; `build-dataset` and `analyze-*` redo that
step and the analysis for free. Dummy datasets are named `study1_dummy` / `study2_dummy`
(e.g. `analyze-study1 study1_dummy`). Each command is also available as a module, for example
`python -m llm_investors.sim.run_study1`.

**Viewer:** `investor-viewer` (or `streamlit run src/llm_investors/app/streamlit_app.py`). It
only reads files and never calls the API.
Its pages are Research setup, Assumptions, Observability, Live view (refreshes every 2 s while a
run is going, with a period slider for finished runs) and Findings.

## Replay from cache

Every API reply is stored in `cache/<study>/` under a hash of (model, effort, max tokens, JSON
mode, system prompt, messages). Re-running an identical config replays every reply at $0 and
rewrites the logs. A request that hit the incomplete limit is cached as that stop, so a re-run
reproduces it without paying for the attempts again. `--retry-incomplete` opts in to paying for
another try.

**`--replay` guarantees $0** (`run-study1 --stage pilot --replay`, and the same for every stage
and for Study 2):

- It reads the cache only. The OpenAI client is never created, so no API key is needed and no
  request can be sent.
- It writes to a temporary `logs/<name>__replay/`. If every reply was in the cache, it swaps
  those logs in for the originals and rebuilds the results.
- If any reply is missing (because the prompts changed: config, prompt text, seed, model or
  `max_output_tokens`), it stops at the first miss, keeps the original logs untouched, reports
  the miss and exits with code 3.

A replay reproduces the logs event for event. The only exception is the order of `llm_call`
events within a Study 2 period, since agents decide in parallel; the events and all results are
identical. Without `--replay` the OpenAI client is likewise created only at the first real API
call, so a fully cached re-run also needs no key. These cost money again: any prompt edit, a change of model / effort /
`max_output_tokens`, new personas, scenarios or seed, or setting `llm.replicate_id` (which forces
fresh draws). **The cache is committed to git** so everyone can replay (see "Sharing the
cache"). Logs are not committed: `--replay` regenerates them for free.

**Identical prompts share one draw (common random numbers).** The cache key is the full
prompt, so arms that send the same prompt get the same reply:

- In Study 1, the `none`, `log` and `reflect` arms of a persona × scenario send identical
  period-1 prompts. `reflect` matches `none` through period 5, until its first lessons.
- In Study 2, replicate *r* of `memory_none` matches `memory_reflect` through period 5, and
  matches `baseline` in period 1.

This is deliberate: it removes noise from the paired comparison. Concurrent identical requests
are single-flight: one paid call, and the others wait and read it from the cache. So the sharing
doesn't depend on timing, and a re-run reproduces every log exactly. (Without that lock,
parallel arms each paid, and the last write won.)

## Sharing the cache

`cache/` is committed to the repository; it is what makes `--replay` free for every
collaborator.

- Each reply is one small JSON file named by the hash of its request
  (`cache/<study>/<2 hex>/<sha256>.json`, about 0.5 KB). A full Study 1 run is roughly 14,000
  files and 10 MB, so plain git is enough (no LFS).
- `.gitattributes` hides these files from diffs and collapses them in pull requests.
- Not committed: `cache/<study>/spend.json` (this machine's spend ledger), temporary `*.tmp`
  files, and `logs/` (regenerated by `--replay`).

**After every paid run, commit and push the cache.** The run prints a reminder when cache
files are not backed up yet, and `cache-stats` shows the state at any time:

```bash
git add cache/study1 && git commit -m "cache: study1 pilot" && git push
cache-stats          # entries, total paid, and whether everything is committed and pushed
```

**A collaborator** reproduces everything at $0, without an OpenAI key:

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
run-study1 --stage pilot --replay      # and/or --stage full; run-study2 likewise
analyze-study1 study1_pilot            # results and figures, free
investor-viewer
```

**One budget for everyone.** Each cached reply records what it cost (`paid_usd`, including
incomplete attempts). A run starts from the larger of this machine's ledger and the total in
the shared cache, so a fresh clone can't spend a second $4 or $6. The amount is printed at the
start of every paid run.

**Rules for paid runs:**

- Pull before a paid run, so you replay what others already paid for instead of paying again.
- One person runs each paid stage, then pushes the cache straight away.
- Two people who pay for the same request independently get two different draws under the
  same file name, which is a git conflict.
  - Keep the version already on GitHub (`git checkout --ours <file>` while rebasing,
    `--theirs` while merging), commit, then run your stage again with `--replay`.
  - That rebuilds your logs from the shared draws at $0.
- A cache file left with conflict markers is never treated as a miss, which would silently
  pay. The run stops with `cache_corrupt`, and `cache-stats` counts corrupt files.

## Layout

```
pyproject.toml          package metadata and the console commands
requirements.txt        pinned versions
src/llm_investors/      the code (one installable package)
  agents/               personas, memory (none/log/reflect), prompt builder, JSON parsing, Agent
  llm/                  OpenAI client (Flex tier, backoff), disk cache, cost tracker with hard cap
  envs/                 exogenous.py (Study 1 random-walk paths + account),
                        market.py (call auction, dividends)
  measure/              PGR/PLR/DE, decision-level and run-level datasets, market metrics
  analysis/             study1.py (LPM / logit, cluster bootstrap, Friedman), study2.py, style
  sim/                  run_study1.py, run_study2.py, shared plumbing (logs, live table)
  app/                  Streamlit viewer and its launcher
  paths.py              project root used for config/logs/cache/results
tests/                  accounting, PGR/PLR hand-checked example, auction, prompts, LLM client
config/                 study1.yaml, study2.yaml (model, budgets, stages, treatments, seeds)
cache/<study>/          reply cache, committed (see "Sharing the cache")
logs/<study>/run_<id>.jsonl, results/<study>/   generated at run time (logs not committed)
assets/, prototypes/    proposals, papers, and the earlier single-study prototype
```

Generated files always go to the repository root, whatever directory you run a command from.
Set `LLM_INVESTORS_ROOT` to keep config and outputs somewhere else.

Event types in the logs: `run_start`, `llm_call` (prompt, raw reply, parsed output, validity,
tokens, latency, cost, API retries, `incomplete_retries` and their token usage), `reflection`,
`fallback`, `order_rejected`, `trade`, `auction_cleared`, `period_end`, `incomplete_stop`,
`fatal_api_error`, `cache_miss` (`--replay` only), `cache_corrupt`, `budget_stop`, `run_end`.

## Study 1 prices

Each stock follows a driftless log-price random walk, `ln P(t+1) = ln P(t) + mu + sigma·Z`,
with `mu = 0` and `sigma = 0.10` per monthly period (`market.mu`, `market.sigma` in
`config/study1.yaml`). Every stock starts at 100, and scenario *k* uses seed `seed + k`. With
`mu = 0` the median price stays at 100, so a position is equally likely to be at a gain or a
loss. No real market data is used, which keeps prompts free of tickers and dates and leaves
nothing for the model to recognize.

### Prompt design (`design` in `config/study1.yaml`)

- **v1** (first pilot, `study1_pilot/`): the rules ended "You have no other information about
  the companies or the market", and agents saw only today's prices. They held about 98% of the
  time. Two personas never sold, and DE was defined in only 10 of 45 runs.
- **v2** (current): that sentence is gone, and the monthly review is described neutrally (buy,
  sell or hold, with no nudge either way). Every arm sees each stock's prices over the last
  `market.price_history` = 3 months (fewer early on). This is market information, not memory.
  The arms still differ only in the agent's own history: `log` shows past trades, portfolio
  value and realized P&L; `reflect` shows lessons.

Outputs go to `study1_v2*` (v1 keeps the plain `study1*` names). Both designs share one cache
and one budget, and v1 can still be replayed with `design: v1`.

## API errors

Only three kinds of error are retried, with exponential backoff and no attempt limit:

| Retried              | Examples                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| Timeouts             | client timeout, HTTP 408, 504                                                                                |
| Rate limits          | HTTP 429 that is not a quota error (`rate_limit_exceeded`)                                                 |
| Resource unavailable | HTTP 503, or a `resource_unavailable` code or message (Flex out of capacity)                               |
| Gateway / Cloudflare | 502, 520, 521, 523 (upstream unavailable) and 522, 524 (gateway timeouts). A 520 stopped the first full run. |

Everything else is **fatal**. The first fatal error stops the whole study run immediately:

- **Quota / billing / spend limit** (`insufficient_quota`, `billing_hard_limit_reached`, or a
  message about quota, billing or a spend limit). OpenAI sends an exhausted quota as HTTP 429,
  so quota is checked before rate limits.
- **Any other API error** (bad request, auth, unknown model, HTTP 500 from OpenAI itself,
  connection drop).

When that happens:

1. **Halt.** The client trips a circuit breaker, so no further request is sent from any
   thread, including retries waiting in backoff. A request already in flight can't be
   recalled.
2. **Log.** The agent logs a `fatal_api_error` event with the error type, HTTP status, code,
   message and request id. The run ends with status `quota_stop` or `api_error_stop`, and the
   other runs end as `aborted`. A failed call never becomes a hold.
3. **Report.** A banner is printed after the run summary and saved to
   `logs/<study>/FATAL_ERROR.json`. The Observability page shows it too. No pilot cost
   projection is written, and the command exits with code 2.
4. **Resume.** Fix the cause (e.g. the spend limit in the OpenAI dashboard), then run the same
   command again. Completed calls replay from the cache at $0.

The study's own budget cap (`budget.cap_usd`) is separate: it stops the study before a call
would exceed the cap and logs `budget_stop`.

## Incomplete replies

`max_output_tokens` is 800 in both studies, and the cap includes reasoning tokens. A reply that
comes back `incomplete` (usually because it hit the cap) is handled like an API failure, not
like malformed JSON:

- It is retried at once. It is never parsed, never cached and never turned into a hold.
- Every attempt is charged against the budget cap. The `llm_call` event records
  `incomplete_retries` and their token usage.
- After `llm.max_incomplete_retries` (5) incompletes in a row on one request, the agent logs an
  `incomplete_stop` event and that run stops with status `incomplete_stop`. A Study 2 market
  stops as a whole, because it can't clear without that agent's order. Other runs continue.
  Stopped runs are excluded from analysis and can be re-run, with completed calls replaying
  from cache.
- Counts per condition: the `incomplete` and `incomplete_stops` columns in
  `results/<study>/runs.csv` (Study 1) and `markets.csv` (Study 2), the summary printed after
  each run, and the Observability page.

## Deploying the viewer (Streamlit Community Cloud)

A deployed viewer serves a fresh git checkout, and `logs/` is git-ignored, so it needs:

- **Logs:** gzipped copies in `published/logs/<dataset>/run_*.jsonl.gz` (~12x smaller). Paid
  runs write them automatically when they finish; for older runs use `publish-logs` (all real
  datasets) or `publish-logs <dataset>`. Publishing is deterministic, so unchanged logs give no
  git diff. The viewer reads `logs/` when it exists locally and the published copy otherwise.
- **Analysis:** `build-dataset <dataset> && analyze-study1|2 <dataset>`, then commit
  `results/<dataset>/` (dummy and dry-run results stay ignored).
- **Commit and push** `published/` and `results/`, then reboot the app if it doesn't pick up
  the change.

App settings: main file `src/llm_investors/app/streamlit_app.py`, dependencies from
`requirements.txt`. The viewer never calls the API and needs no key. It opens on the main
design's full run (`study1_v2`, `study2_v2`) or, before that exists, its pilot.

## Call auction rule (Study 2)

Every period, all limit orders go into one call auction:

1. **Candidate prices:** every order's limit price (clipped to the price band if a price limit
   applies) plus the previous price.
2. **Pick the price** that maximizes traded volume. Ties are broken by the **smallest order
   imbalance** |demand − supply|, then by the price **closest to the previous price**, then by
   the lower price.
3. **Fill** the short side completely and the long side pro rata (integer shares,
   largest-remainder rounding).
4. If no orders cross, there is no trade, and the previous price carries forward (shown as
   "no trade").
5. With a price limit, the band is ±limit around the previous price, and around FV₁ in
   period 1.

**This departs from the proposal on purpose.** The proposal breaks ties only by closeness to
the previous price. Under one-sided order flow that rule keeps the price fixed: any price in
the crossing range trades the same volume, and the previous price is always in that range. In
the dummy-agent test it held the price at 360 for 15 periods while the fundamental value fell
to 24, which would inflate every bubble measure. Smallest imbalance is the usual secondary
rule in call auctions, and it lets excess supply or demand move the price.

The min-imbalance rule is the default (`market.min_imbalance_tiebreak: true` in
`config/study2.yaml`). Set it to `false` for the proposal's literal rule.
`tests/test_auction.py` covers both rules.

## Study 2 order design (v2: two-sided quotes)

**Why it changed.** In the first real pilot (design v1, `logs/study2_pilot/`, $0.011), each
agent sent one buy, sell or hold order per period. From period 2 on, 9–10 of 10 agents offered
to sell and none bid, nearly all saying the price was above the expected remaining dividends.
Nothing crossed, so the price carried forward at 240 for 9 periods while FV fell from 216 to
24. The old metrics read that frozen price as a bubble (RD = 0.82, peak overpricing 1.64),
and the in-market DE was empty because the price never left anyone's cost basis. The agents
agree with each other (same model, FV stated on screen), so one-sided order flow is the
expected outcome, and the auction can't discover a price without both sides.

**What v2 does.** Every period each agent submits a two-sided quote: one buy order (the highest
price it will pay) and one sell order (the lowest price it will accept). The auction rule
itself is unchanged.

**Assumptions made in v2** (not fixed by the proposal):

- **Buy limit < sell limit, strictly.** Then an agent can never be on both sides of one
  clearing price, so it never trades with itself (asserted every period).
- **Quantity ≥ 1 on each side unless that side is infeasible.** A sell of 0 is allowed only
  with no shares; a buy of 0 only when cash can't pay for one share at the buy limit. Without
  this rule a side could be left empty and the market could go one-sided again, as in v1.
- **A quote is not a commitment to trade.** No maximum spread is imposed: an agent that doesn't
  want to trade can price a side far from the market. No-trade periods are therefore still
  possible, and they are measured, not hidden (see below). The order format says so in one
  neutral sentence ("An order executes only if the auction price is at or below your buy limit
  or at or above your sell limit").
- **Invalid quotes** (wrong count of entries, two buys, buy limit ≥ sell limit, a missing price,
  quantity 0 on a feasible side) get the usual single retry with a correction that names the
  problem, then a logged fallback with no orders that period. Same policy as Study 1.
- **Feasibility clipping is per order and unchanged:** a buy is clipped to cash ÷ buy limit, a
  sell to shares held. Both sides can't fill in one auction, so this period's sale proceeds
  never need to fund this period's purchase.
- **Unchanged from v1:** the FV line ("Expected value of the remaining dividends …") is still
  shown every period, as the proposal specifies; market history still lists the
  carried-forward price marked "(no trade)"; the period-1 reference price is FV₁; price
  limit and tax rules are the same.
- **Executed trades, not orders,** are what the per-run heatmap (`actions_heatmap_*.png`)
  shows, since every agent now quotes both sides every period.
- **Reproducibility.** `design: v2` in `config/study2.yaml` writes to `study2_v2*`;
  `design: v1` still builds byte-identical v1 prompts (checked against all 100 logged pilot
  prompts), so the v1 pilot replays for free from the shared cache.

**Price measures use traded periods only.** In a no-trade period the logged price is the
previous price carried forward, which nobody paid, so it is not evidence of mispricing.

- RAD, RD, peak overpricing and crash are computed over traded periods; mean FV is taken over
  the same periods, and the peak period is the actual period number.
- Returns (volatility, autocorrelations, kurtosis, stylized facts) are log changes between
  consecutive traded prices; a return that spans no-trade periods counts once.
- Volume and turnover use every period (a no-trade period adds 0).
- `traded_periods` and `no_trade_share` are reported next to the measures in `markets.csv`,
  the run summary and the analysis tables. A market with no trade at all has NaN price
  measures; cell means skip NaN, so always read them with `no_trade_share`.
- The price chart in the viewer and `price_vs_fv.png` plot clearing prices at traded periods
  only.
- **In-market DE is unchanged:** in a no-trade period a position is still valued at the last
  traded price, because that is the price the agent sees.

These measures follow Stöckl, Huber & Kirchler (2010), whose lab markets trade every period. A
high `no_trade_share` makes them less comparable with lab results, and should be reported.

## Implementation choices not fixed by the proposal

- **Order handling.** Orders that exceed holdings or cash are clipped to the feasible
  quantity and logged as `order_rejected`; they are not discarded.
- **Malformed JSON.** The retry adds the model's bad reply and the error message, so the
  retry is a different prompt (an identical prompt would just replay from the cache). If the
  retry also fails, the agent holds and a `fallback` event is logged with its condition.
  API errors never become holds: they are retried or stop the study (see "API errors").
- **Study 1 endowment.** 50,000 cash plus 100 shares of each stock, bought at 100 in period 0.
- **Study 1 run order.** Runs go scenario by scenario (all personas × arms of scenario 0, then
  1, ...). If the budget cap stops the study, it loses whole matched sets at the end, never
  whole cells, so the paired design stays balanced.
- **Omitted stocks are invalid.** A Study 1 reply must give an entry for each of A–E. A missing
  stock makes the reply invalid, so it gets the usual single retry with a correction that
  names the missing stocks, then a logged fallback. No hold is ever filled in silently, since
  fake holds would inflate DE. The count per condition is `omitted_stock_replies` in
  `runs.csv`, also shown on the Observability page.
- **Reference point in the prompt.** The Study 1 state shows a "gain/loss vs average cost"
  column, and `log` history shows "% vs your average cost" on sales, as broker apps do. This may
  push DE toward a ceiling in every arm. `market.show_pnl: false` (or
  `run-study1 --no-pnl`) runs an ablation without them; average cost is still shown. The
  default prompts and cache keys are unchanged.
- **What `log` memory contains.** The agent's own trading record (account value, % change,
  trades, realized P&L vs cost), not market memory: it never repeats past prices. In Study 2
  the public price history is shown to every architecture.
- **Study 1 inference.** SEs are clustered by matched set (persona × scenario), not by run as in
  the proposal. The three arms of a set share prices and identical early prompts, so their
  errors are correlated; matched sets nest runs, so this is the more conservative choice.

  - Robustness specifications: without the `ret` control (with it, gain × arch is a jump at
    zero return, not the full winner–loser gap), without the final period (agents know the
    horizon), logit (a failure is printed as FAILED and listed in `summary.json`), and
    reflect vs none on periods 6+.
  - Pooled PGR, PLR and DE get cluster-bootstrap CIs over matched sets (`de_pooled.csv`).
  - The run-level table reports how many matched sets were dropped because DE was undefined in
    some arm.
- **Study 2 analysis is descriptive.** 3–5 runs per cell are too few for CIs, so the analysis
  reports mean, sd, min and max per cell, plus each cell minus baseline for the same
  replicate. Replicates share dividend draws and endowments across cells.
- **Study 2 tie-break.** Smallest order imbalance before closeness to the previous price
  (see "Call auction rule" above).
- **Study 2 agents and memory.** Each Study 2 agent gets a stable session tag
  (`(Session r-k.)`) so that agents with the same persona and endowment don't produce
  identical cached replies. Market history (prices, volumes, dividends) is public
  information, so every agent sees it, including under `none`. Memory controls only the
  agent's private history.
- **Study 2 DE.** It uses every period with a position, because a single stock gives no paper
  counterpart on sale days. It is **not comparable like for like with Study 1**, for three
  reasons:

  - endowed shares are booked at FV₁ = 360 while FV falls to 24, so most positions are
    structural paper losses;
  - gain/loss uses a clearing price the agent hadn't seen when it decided;
  - the sample is all periods rather than sale days.

  Study 1's all-periods DE is shown next to it only as a reference.
- **Study 2 endowments.** Endowed shares are booked at FV₁ as their purchase price.
- **Bubble measures.** Peak overpricing is max(P − FV) / mean FV, on the same scale as RAD
  and RD. Crash is the maximum drawdown of the price path. All use traded periods only (see
  "Study 2 order design").
- **Temperature** is never sent. `json_mode` sets `text.format = json_object`; replies are
  still validated.

## Open decisions (proposal §12, plus items found while building)

- **Personas.** The persona text in `agents/personas.py` still needs grounding in TWSE
  investor statistics.
- **Incomplete rate.** Check the pilot's incomplete counts per condition before the full run.
  If they're high, raise `max_output_tokens` or lower the effort. Retries cost money, and an
  uneven rate across memory modes would be worth reporting.
- **Prices on screen in Study 2.** FV is shown every period, as the proposal specifies, and was
  kept in design v2 (decision by Victor). This is a strong cue toward rational pricing and may
  keep prices near FV (a null result for bubbles). Check the v2 pilot's price path and
  `no_trade_share` before the 20-market run.
- **Cached-input price.** It is set equal to the full input price (conservative) until the
  Flex rate for cached tokens is confirmed.
- **Other §12 items.** Study 2 treatment priorities, and whether to add `gpt-6.1-sol` check
  runs.

AI tools used are logged in `AI_USE_LOG.md` (needed for the competition's AI use statement).
