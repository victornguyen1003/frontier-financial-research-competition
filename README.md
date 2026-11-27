# LLM investor personas and the disposition effect (prototype 2)

Prototype for the Frontier Financial Research Competition (theme: Generative AI and Behavioral
Finance). It implements [docs/prototype-2-plan.md](docs/prototype-2-plan.md) (Study 1,
`design: v3`); background, assumptions and limitations are in
[docs/prototype-2-research-notes.md](docs/prototype-2-research-notes.md).

LLM agents trade 5 anonymous stocks on calibrated, zero-drift price paths. Each agent gets a
persona built from 11 randomized binary attributes, plus a control persona C0 and a bias-cue
persona C1, and one of three memory arms (`none`, `log`, `reflect`). The analysis measures each
attribute's effect on the disposition effect (DE = PGR − PLR) and compares with Barber et al.
(2007).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # pinned versions used for the runs
pip install -e .                 # installs the llm_investors package and the commands below
echo 'OPENAI_API_KEY=sk-...' > .env && chmod 600 .env   # your key (only needed for paid runs)
pytest -q                        # no API calls
```

The OpenAI key goes in `.env` at the repository root (`OPENAI_API_KEY=sk-...`). `.env` is
git-ignored, a key exported in the shell wins, and `--dry-run`, `--replay` and
`--backend dummy` never need a key. Also set a spend limit in the OpenAI dashboard.

**Phone notifications (optional):** add `NTFY_TOPIC=<your topic>` to `.env` and subscribe to that
topic in the ntfy app. `run-study1` (real runs only, not dummy or dry runs), `power-study1`
and paid `tag-reasoning` then post a summary when they finish or stop. The topic stays in
`.env` because anyone who knows an ntfy topic name can read it.

## Input data

| Path | Committed | What |
| ---- | --------- | ---- |
| `data/raw/tej/` | **no** (licensed) | TEJ daily stock data, `calibration.tej_path` |
| `data/investor_statistics.csv` | yes | TDCC "Investors with Trading Accounts by Age" (age levels) |
| `data/attribute_data.csv` | yes | further hand-collected statistics (`attribute, statistic, value, year, source_url`) |
| `data/benchmarks/` | yes | Barber et al. (2007), Table 3 Panel B |
| `results/calibration/` | yes | what `calibrate` derives; the generator and persona builder read only this |

## Build order and commands

| Step | Command | Cost | Result |
| ---- | ------- | ---- | ------ |
| 0 | `calibrate` | free | `results/calibration/{prices,attributes}.json` + figures |
| 1 | `run-study1 --backend dummy --stage pilot` | free | full pipeline with rule-based agents |
| 2 | `run-study1 --dry-run --stage pilot --model <id>` | free | exact prompts and a cost projection |
| 3 | `run-study1 --stage pilot --model gpt-6-luna` | ≈ $0.35 | 150 runs → `logs/study1_v3_gpt-6-luna_pilot/`; prints measured cost and the main-run projection |
| 4 | `run-study1 --stage pilot --model gpt-6.2-sol` | see "Budget" | second pilot |
| 5 | `analyze-study1 <pilot>` and `power-study1 <pilot>` | free | §5 analysis; power curves, MDE and the recommended (P, S) |
| 6 | set `stages.full.{profiles, scenarios}` and `llm.model`, then `run-study1 --stage full` | ≤ $6.5 | main run → `study1_v3_<model>` |
| 7 | `tag-reasoning <dataset>` | ≤ $0.5 | `reasoning_tags.csv` and a 100-text hand-coding sample |
| 8 | `analyze-study1 <dataset>` | free | `results/<dataset>/analysis/` |

Every paid stage first projects its cost from its exact prompts (no API call), refuses to start
if the typical projection exceeds its stage limit (shrink with `--scenarios N`), and asks for
confirmation (`--yes` skips). Flags: `--profiles N` / `--scenarios N` override the stage size,
`--workers N`, `--no-live`, `--replay`, `--retry-incomplete`.

Names: v3 outputs are `study1_v3_<model>` (main) and `study1_v3_<model>_pilot`; dummy and dry
runs add `_dummy` / `_dryrun`. A dry run only builds the prompts and prints a cost estimate (a
paid run does the same automatically before asking), so `_dryrun` folders are not shown in the
viewer. `build-dataset <name>` and `analyze-study1 <name>` redo the
free steps.

**Viewer:** `investor-viewer` (or `streamlit run src/llm_investors/app/streamlit_app.py`). Pages:
research goal and setup, agent prompts (prompt builder), live view (refreshes every 2 s),
final dataset (filters, downloads), data analysis. It reads files only and never calls the API.

## Budget

`budget.cap_usd: 10` is a hard stop for everything in `cache/study1/` (this prototype only; the
cache was reset for it). Each stage also has its own limit (`budget.stage_limits`):

| Stage | Limit | Dry-run projection (typical / upper bound) |
| ----- | ----- | ------------------------------------------ |
| Pilot `gpt-6-luna`, 150 runs | $1.0 | $0.35 / $0.80 |
| Pilot `gpt-6.2-sol`, 150 runs | $2.0 | $14.19 / $32.11 at Standard prices (no Flex tier reported) |
| Reasoning tags | $0.5 | measured at run time |
| Main run, 1,440 runs (Luna) | $6.5 | $3.43 / $7.77 |

"Typical" assumes `llm.est_output_tokens` (450) output tokens per call; replace it with the
Luna pilot's measured value. The `gpt-6.2-sol` prices in the config are unconfirmed
third-party list prices.

## Design choices made while implementing the plan

- **Monthly returns** are the sum of TEJ's daily log returns (`ROI%-Ln`), which are dividend- and
  split-adjusted; month-end closes are kept only for the path figure. Calendar months the TEJ
  file covers with fewer than 15 trading days are excluded and listed in `prices.json`
  (`months_excluded_incomplete`): the 1999 export has 10 days in July, none in August and 3
  in September.
- **Student-t degrees of freedom** are fitted by simulated moments, not direct MLE. With ~9
  months per stock, standardizing by each stock's own sample SD truncates the tails, and the
  MLE on the pooled z-scores goes to df → ∞. The fitted df makes the simulated pooled
  standardized kurtosis equal the empirical one (same months per stock); the biased MLE is
  recorded for reference.
- **Age levels:** TDCC brackets are 0–19, 20–30, 31–40, 41–50, 51–60 and 61+, so "under 40 / 40
  and over" becomes 40 and under against 41 and over (interpolated medians 30 and 59).
- **Profile draws are prefix-stable** (`sample_profiles(30, seed)[:8] == sample_profiles(8, seed)`),
  so a same-model main run reuses the pilot's profiles and replays their calls for free.
- **Pilot regression:** 8 profiles cannot identify 11 attribute effects plus a constant. The
  analysis detects the rank deficiency and fits profile fixed effects with gain × memory terms
  only; attribute effects come from the main run.
- **Two-way clustering** (path, profile) follows Cameron, Gelbach & Miller; tests use t / F
  with (smaller number of clusters − 1) degrees of freedom.
- **Power analysis:** simulated profiles start from the sampled profiles' sell rates, not
  C0's: C0 barely trades and has a reverse DE. Variance components come from the ANOVA
  (moments) method, with MixedLM recorded only as a cross-check, because on pilot-sized data it
  reported implausible components. The residual profile heterogeneity beyond additive
  attribute effects cannot be estimated from 8 profiles, so it is an assumption
  (`power.profile_residual_sd: 0.1`), with sensitivity runs at 0 and 0.2
  (`--residual-sd`, written to `power/residual_sd_<x>/`).
- **`reflect` = record + lessons** in v3, so before the first reflection `reflect` and `log` send
  identical prompts and share one cached draw (common random numbers), like `none` and `log`
  in month 1.
- **Reasoning tags** are batched (`tagging.batch_size: 20` texts per call) to fit the $0.5
  limit; batches are formed in content-hash order so re-runs replay from cache.

## Replay and the shared cache

Every API reply is cached in `cache/study1/` under a hash of (model, effort, max tokens, JSON
mode, system prompt, messages). Re-running an identical config replays at $0. `--replay`
guarantees $0: it reads the cache only (no key needed), writes to `logs/<name>__replay/`, swaps
the logs in when every reply was cached, and otherwise stops at the first miss (exit 3) leaving
the original logs untouched. Identical concurrent requests are single-flight (one paid call).

Commit and push `cache/study1/` after every paid run (`cache-stats` shows what is not backed
up). Each cached reply records `paid_usd`, and a run starts from the larger of this machine's
ledger and the shared cache total, so collaborators share one budget. Pull before a paid run;
one person runs each paid stage.

Design v2 (prototype 1) still builds byte-identical prompts (`tests/test_envs.py` pins a
fingerprint), although its cache and results were removed for this prototype.

## API errors and incomplete replies

- **Retried** with backoff, no attempt limit: timeouts (408, 504, 522, 524), rate limits (429
  that is not quota), resource unavailable (503, Flex capacity), gateway errors (502, 520,
  521, 523).
- **Fatal**, stopping every run at once: quota / billing / spend-limit errors and every other
  API error. They are logged as `fatal_api_error`, reported in `logs/<name>/FATAL_ERROR.json`,
  and the command exits with code 2. Fix the cause and re-run; completed calls replay at $0.
- **Incomplete replies** (`max_output_tokens: 1200` includes reasoning) are retried, charged and
  never parsed or held; after 5 in a row that run stops (`incomplete_stop`).
- **Invalid replies** (bad JSON, a missing stock, empty or >1,000-character `reasoning`) get one
  correction retry, then a logged hold (`fallback`).

## Layout

```
src/llm_investors/
  agents/        personas (attribute catalog, sample_profiles, C0/C1), memory, prompts, parsing, Agent
  calibration/   sources.py (TEJ, TDCC loaders), prices.py, attributes.py, report.py (`calibrate`)
  envs/          exogenous.py: RULES_V3, calibrated generator, state block, account
  llm/           OpenAI client (Flex, backoff), disk cache, cost tracker (cap + stage limits)
  measure/       PGR/PLR/DE/Ratio, datasets (runs, positions, decisions, reasoning, lessons)
  analysis/      study1.py (§5), ols.py (two-way clustered OLS), power.py (§2.5), reasoning.py (§4.2)
  sim/           run_study1.py, shared plumbing, cache-stats, publish-logs
  app/           Streamlit viewer
config/study1.yaml   design, models and prices, budget, profiles, stages, calibration, tagging, power
data/                inputs (raw TEJ git-ignored)
results/             calibration/ (committed) and one folder per dataset
logs/                run logs (git-ignored; regenerated by --replay), published/logs/ for deployment
```

Log events: `run_start` (persona, profile, prices), `llm_call` (prompt, raw reply, parsed
output, validity, tokens, cost), `reflection`, `fallback`, `order_rejected`, `trade`,
`period_end` (holdings, orders, observations, reasoning), `incomplete_stop`,
`fatal_api_error`, `cache_miss`, `cache_corrupt`, `budget_stop`, `run_end`.

## Deploying the viewer

A deployed viewer serves a git checkout, so commit `results/<dataset>/` and the gzipped logs in
`published/logs/` (paid runs publish them automatically; otherwise `publish-logs <dataset>`).
Main file `src/llm_investors/app/streamlit_app.py`, dependencies from `requirements.txt`.
