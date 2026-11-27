# Prototype 2: implementation spec

Study 1, `design: v3`. Agents trade 5 anonymous stocks on simulated price paths. Persona
attributes and memory arms are randomized, and the output measures each attribute's effect on
the disposition effect (DE).

This file contains only what is needed to build and run the study. Rationale, assumptions,
limitations, future extensions, timeline and references are in
[prototype-2-research-notes.md](prototype-2-research-notes.md). Do not implement anything listed
there as a future extension.

## 1. Research questions (referenced by the analysis)

- **RQ1:** do persona attributes change PGR, PLR and PGR − PLR compared with the control persona
  C0?
- **RQ2:** which attributes move the DE and by how much, and which are blind spots?
- **RQ3:** does memory (`log`, `reflect`) change the DE, and does the reasoning refer to past
  trades?

Benchmark: Barber et al. (2007), Table 3 Panel B. Store it as data for the charts:

| Investor type     | PGR   | PLR   | PGR/PLR |
| ----------------- | ----- | ----- | ------- |
| Individuals       | 4.39% | 1.67% | 2.63    |
| Corporations      | 0.83% | 0.31% | 2.68    |
| Dealers           | 4.54% | 1.73% | 2.62    |
| Mutual funds      | 1.32% | 1.53% | 0.86    |
| Foreign investors | 0.46% | 0.67% | 0.69    |
| All investors     | 2.93% | 1.42% | 2.06    |

## 2. Experimental setup

### 2.1 Market environment

| Item                          | Setting |
| ----------------------------- | ------- |
| Stocks                        | 5 anonymous stocks (A–E). No tickers, no dates. |
| Prices                        | Zero-drift log random walk, ln P(t+1) = ln P(t) + σᵢ·εₜ. Each stock draws its own monthly σᵢ from the empirical σ distribution in `results/calibration/prices.json`. εₜ is Student-t with the fitted degrees of freedom, scaled to unit variance. All stocks start at 100. Scenario k uses seed `seed + k`. |
| Horizon                       | 15 periods; one period is one month |
| Endowment                     | 50,000 cash plus 100 shares of each stock, bought at 100 in period 0 |
| Trading                       | Market orders at today's price. No fees, taxes, borrowing or short sales. |
| Public information (all arms) | Account, **average cost** of each position, today's prices, and the **full price history** of each stock since month 0 (`market.price_history: all`) |
| Reference point               | Average purchase cost of each position, shown in the account table |

### 2.2 Persona attributes

11 binary attributes. Level 0 is the baseline, level 1 the alternative. For attributes 1–7 and
10, level 0 adds no sentence. Attributes 8, 9 and 11 add a sentence at either level.

| #  | Key                   | Level 1 sentence | Level 0 sentence |
| -- | --------------------- | ---------------- | ---------------- |
| 1  | `loss_sensitivity`    | Losing NT$1,000 bothers you about twice as much as gaining NT$1,000 pleases you. | — |
| 2  | `mental_accounting`   | You think of each stock as a separate account and judge it by whether it is up or down since you bought it. | — |
| 3  | `regret_aversion`     | You strongly dislike admitting that a purchase was a mistake. | — |
| 4  | `realization_utility` | Locking in a gain gives you real satisfaction, and selling at a loss feels painful. | — |
| 5  | `financial_literacy`  | You studied finance and understand risk, return and diversification. | — |
| 6  | `experience`          | You have traded Taiwan-listed stocks for more than twenty years. | — |
| 7  | `professional`        | You manage a stock portfolio professionally for an investment firm. | — |
| 8  | `age`                 | You are {older_age} years old. | You are {younger_age} years old. |
| 9  | `gender`              | You are a woman. | You are a man. |
| 10 | `market_knowledge`    | Prices here move randomly: past price changes do not predict future changes. | — |
| 11 | `trading_activity`    | You check your account every day and trade several times a month. | You check your account a few times a year and rarely trade. |

- `{older_age}` and `{younger_age}` are read from `results/calibration/attributes.json` (§2.7).
- **Sampling:** `sample_profiles(n, seed)` draws each attribute independently with probability
  ½. It must be deterministic for a given seed.
- **Composition:** `compose_persona(profile)` returns `"You are an individual investor in
  Taiwan."` followed by the attribute sentences in the table order (1 → 11).

Fixed personas:

- **C0 (control):** all attributes at level 0. The text is the control sentence plus the age,
  gender and trading-activity level-0 sentences.
- **C1 (bias cue):** C0 plus `"You tend to sell stocks that have gone up quickly and hold on to
  stocks that have gone down."`

### 2.3 Memory arms

| Arm       | Added to the user prompt beyond the public state |
| --------- | ----------------------------------------------- |
| `none`    | Nothing |
| `log`     | The agent's own record: one line per past month with account value, % change, trades, and realized gain or loss against average cost |
| `reflect` | The `log` record plus three lessons, written by a reflection call after periods 5 and 10 |

### 2.4 Run grid

- **Grid:** (P profiles + C0 + C1) × S paths × 3 arms. Every persona runs on the same S paths in
  all 3 arms.
- **Run order:** scenario-major, as now, so a budget stop removes whole matched sets.
- **Pilot:** P = 8, S = 5, giving 150 runs. It runs **twice**, once per model (§2.6).
- **Main:** P and S come from the power analysis (§2.5). Default P = 30, S = 15, giving
  1,440 runs.
- **Config keys:** `profiles.n`, `profiles.seed`, `stages.pilot.{profiles, scenarios}`,
  `stages.full.{profiles, scenarios}`.

### 2.5 Power analysis (`power-study1 <pilot dataset>`)

Run it once per model pilot.

1. **Fit to the pilot:** a random-effects model of decision-level `sold`, with a path effect, a
   profile effect, gain × attribute and gain × memory terms. Also record:
   - C0's sell rates on gains and on losses;
   - the variance components (path, profile residual, within-run);
   - the mean number of sale days per run.
2. **Simulate:** for each (P, S) in P ∈ {20, 30, 40} × S ∈ {10, 15, 20}, simulate 500 datasets
   with every attribute effect δ = 0.05 on PGR − PLR.
3. **Fit and record:** fit the §5.2 regression on each dataset and record power for:
   - each d_k, Holm-adjusted across the 11 attributes (α = 0.05);
   - the joint Wald test that all d_k = 0;
   - the memory Wald test;
   - C1 against C0.
4. **Recommend** the cheapest (P, S) with at least 80% power for each d_k whose projected cost
   is within the main-run budget (§8). If none qualifies, recommend the largest affordable
   design.
5. **Write** to `results/<pilot dataset>/power/`:
   - `power.csv`: one row per (P, S, test);
   - `mde.csv`: the minimum detectable effect per (P, S);
   - `recommendation.json`;
   - power-curve figures.

### 2.6 Model

| Setting          | Value |
| ---------------- | ----- |
| Models           | Pilots: `gpt-6-luna` and `gpt-6-sol`, run with `run-study1 --stage pilot --model <id>`. Main: one model, set in `llm.model`. |
| Tier             | Flex |
| Reasoning effort | Low |
| Temperature      | Not sent |
| Output cap       | `max_output_tokens: 1200`, including reasoning |
| Prices           | `llm.price_per_mtok` for the default model; `llm.model_prices.<id>` for other models |

`--model` already exists. It writes to `study1_v3_<model>*` and refuses a paid run if the model
has no entry in `model_prices`.

### 2.7 Calibration (`calibrate`)

| Input | Path | Notes |
| ----- | ---- | ----- |
| TEJ daily stock data | Set in `calibration.tej_path` in `config/study1.yaml` | Licensed. Never commit the raw file. Columns (Chinese headers): 公司簡稱, 年月日, 收盤價(元), 報酬率-Ln, 市場別, TSE 產業別, 漲跌停, … |
| Attribute data | `data/attribute_data.csv` | Hand-collected. Columns: `attribute, statistic, value, year, source_url` |

**Prices** (`calibration/prices.py`):

1. Keep rows with `市場別 == "TSE"` and `TSE 產業別 != "-"`. This drops funds and ETFs.
2. Compute month-end closes and monthly log returns per stock.
3. Compute each stock's σ, and keep stocks with at least 6 monthly returns.
4. Standardize returns per stock, pool them, and fit Student-t degrees of freedom.
5. Run random-walk tests on daily log returns: lag 1–5 autocorrelations, Ljung–Box and the
   Lo–MacKinlay variance ratio.
6. Write `results/calibration/prices.json` with the σ list, degrees of freedom, test results and
   sample coverage, plus figures:
   - the σ histogram;
   - a return histogram and QQ plot, empirical against simulated;
   - real paths against simulated paths, rebased to 100.

**Attributes** (`calibration/attributes.py`):

- Read `data/attribute_data.csv`.
- Derive `younger_age` and `older_age` as the median age of investors under 40 and of those 40
  and over.
- Write `results/calibration/attributes.json` with the values and their sources.

Commit `results/calibration/`. The price generator and the persona builder read only these
JSON files, never the raw data.

## 3. Agent prompts

- **System:** rules, then the persona, then the reply format. This is constant within a run.
- **User:** the memory block, then the state block, then the question.

### 3.1 System prompt

Rules (`RULES_V3` in `envs/exogenous.py`):

```
You manage your own brokerage account. It holds cash and five stocks, called Stock A,
Stock B, Stock C, Stock D and Stock E. Each period is one month. Each month you see your
account, the average cost of each position, today's price of each stock and its price history
since month 0, and for each stock you decide whether to buy, sell or hold. Trades execute at
today's price. There are no fees or taxes. You cannot borrow or short: you can sell at most
the shares you hold, and buy only with the cash you have.
```

The persona is prefixed with `Who you are: `. Example (attributes 1, 2, 6, 8 and 11 at level 1):

```
Who you are: You are an individual investor in Taiwan. Losing NT$1,000 bothers you about
twice as much as gaining NT$1,000 pleases you. You think of each stock as a separate account
and judge it by whether it is up or down since you bought it. You have traded Taiwan-listed
stocks for more than twenty years. You are {older_age} years old. You are a man. You check
your account every day and trade several times a month.
```

Reply format (`ORDER_FORMAT_STUDY1_V3`):

```
Reply with ONLY a compact JSON object, no other text:
{"reasoning":"<3-5 sentences: what you considered this month and why>",
 "orders":[{"stock":"A","action":"buy|sell|hold","quantity":<int>,"limit_price":null,
            "reason":"<one short sentence>"}, ...]}
Give exactly one entry for each of A, B, C, D, E. Use quantity 0 for hold.
```

### 3.2 User prompt

The state block is the same in every arm. Extend `state_block()` so the price-history table
covers months 0 … t−1 plus today:

```
Period 7 of 15.
Cash: 48,210.00
stock | shares | average cost | price today | gain/loss vs average cost
Stock A | 100 | 100.00 | 112.30 | +12.3%
...
Account value: 99,850.40

Price history:
stock | month 0 | month 1 | ... | month 6 | today
Stock A | 100.00 | 96.40 | ... | 109.80 | 112.30
...
What do you do this period? Reply with the JSON object only.
```

- `log` prepends `Your record so far:` and its lines.
- `reflect` prepends the record plus `Lessons from your own trading so far:` and the lessons.

### 3.3 Reflection prompt (`reflect`, after periods 5 and 10)

- **System:** the rules, the persona, `You are now looking back on your own trading so far.`,
  and this format:

  ```
  {"lessons":["<lesson 1>","<lesson 2>","<lesson 3>"]}
  ```

- **User:** the full record so far plus any earlier lessons.

### 3.4 Prompt rules (tests)

- The phrase "disposition effect" never appears in any prompt.
- Behavioral wording appears only in persona sentences and the C1 cue. Rules, state, memory and
  format blocks stay neutral. Update `tests/test_prompts.py` accordingly.
- No timestamps or run ids appear in prompts, because the cache keys on the exact text.
- `reasoning` is required. Reject a reply if it is empty or longer than 1,000 characters.
- A malformed reply or a missing stock gets one retry with a correction, then a logged fallback.
  An incomplete reply is retried and never parsed (existing policy).
- Design v2 must still replay with byte-identical prompts.

## 4. Measurement

### 4.1 Disposition effect (Odean 1998 counting, sale periods only)

- Each sold position is a realized gain (RG) or realized loss (RL), comparing the sale price
  with the average cost.
- Each position held but not sold is a paper gain (PG) or paper loss (PL), comparing today's
  price with the average cost.
- PGR = RG / (RG + PG); PLR = RL / (RL + PL); DE = PGR − PLR; Ratio = PGR / PLR.
- Compute these per run, per cell (persona × arm) and pooled.

### 4.2 Reasoning tags (`analysis/reasoning.py`)

An LLM classifier with a fixed prompt, cached like agent calls, returns one boolean per tag
for each `reasoning` text:

| Tag               | True when the text refers to |
| ----------------- | ---------------------------- |
| `persona`         | its own risk attitude, experience, profession, age or trading habits |
| `reference_point` | gain or loss against cost |
| `past_trades`     | earlier trades or lessons |
| `forecast`        | expected future prices |
| `randomness`      | prices being unpredictable |

Validation:

- Sample 100 texts to `results/<dataset>/reasoning_validation.csv` for hand coding.
- Once coded, report agreement and Cohen's κ per tag.

## 5. Analysis (`analyze-study1 <dataset>`)

### 5.1 Descriptive

- **Cell table:** RG, RL, PG, PL, PGR, PLR, DE and Ratio per persona × arm, with 95%
  cluster-bootstrap CIs that resample paths.
- **Benchmark chart:** cell Ratios against the §1 benchmark lines.

### 5.2 Main regression

Decision-level LPM, with one row per position held at the start of a sale period:

```
sold = a + b·gain + Σ_k c_k·attr_k + Σ_k d_k·(gain × attr_k)
         + Σ_m e_m·mem_m + Σ_m f_m·(gain × mem_m)
         + ret + holding_period + drawdown + ε
```

- Sample: the P profiles only. C0 and C1 are compared separately in §5.3.
- Standard errors are **two-way clustered by path and by profile**.
- Report each d_k and f_m both as a coefficient and as a change in PGR − PLR.

### 5.3 Tests

| Test | Definition |
| ---- | ---------- |
| RQ1 joint | Wald test that all d_k = 0 |
| RQ1 paired | Each profile's DE against C0, paired over paths |
| Manipulation check | DE(C1) > DE(C0), paired over paths |
| RQ2 | Each d_k with Holm-adjusted p-values. **Blind spot:** an attribute whose 95% CI lies within ±0.03. |
| RQ2 sameness | Share of identical decisions across profiles on the same path and period, with C0 against C1 as reference |
| RQ3 memory | Wald test that f_log = f_reflect = 0; `past_trades` tag rate by arm |
| RQ3 reasoning | `persona` tag rate against the number of active attributes; `reference_point` tag as a predictor of selling winners |

Write to `results/<dataset>/analysis/`:

- `cells.csv`;
- `regression.txt` and `regression.csv`;
- `tests.json`;
- `blind_spots.csv`;
- `figures/`.

### 5.4 Figures

- Cell DE with CIs by arm.
- Ratio against benchmark lines.
- A forest plot of d_k with the ±0.03 band and the MDE marked.
- Memory effects per persona.
- Tag rates by arm and attribute.
- An example path with trades marked.
- Power curves, from §2.5.

## 6. Streamlit viewer (`app/streamlit_app.py`)

The viewer reads files only and never calls the API. It has five pages.

1. **Research goal and setup**
   - The research questions and benchmark (§1), and the run grid (§2.4).
   - **Pilot comparison**, Luna against Sol: cost per run, incomplete rate, the manipulation
     check, and the spread of profile DEs around C0.
   - **Power:** power curves, MDE and the recommended (P, S), from `results/<pilot>/power/`.
   - **Market parameters** from `config/study1.yaml`.
   - **Data and calibration** from `results/calibration/`:
     - σ distribution, return histogram and QQ plot;
     - random-walk tests;
     - real against simulated paths;
     - the age levels with sources.
   - **Attribute catalog**, C0, C1 and memory arms, from `agents/personas.py`.
   - **Assumptions and limitations**, rendered from the matching sections of
     `docs/prototype-2-research-notes.md`.
   - **Budget:** cap, spend and projection.
2. **Agent prompts**
   - Prompt builder: pick attributes, or C0 or C1, plus an arm and a month, and see the exact
     system and user prompts.
   - The persona's differences from C0, highlighted.
   - The reflection prompt and reply formats.
   - Token counts per block.
3. **Live view**
   - Run monitor, refreshing every 2 s:
     - runs done, running and stopped;
     - spend against cap;
     - incomplete replies, fallbacks, omitted stocks and API errors.
   - Single run with a month slider: prices, holdings, average cost, orders, reasoning, and
     lessons for `reflect`.
4. **Final dataset**
   - The `runs`, `decisions`, `positions`, `lessons` and `reasoning_tags` tables, with filters
     by profile, attribute, arm and path.
   - A column dictionary.
   - A data-quality summary.
   - CSV downloads.
5. **Data analysis**
   - The §5 tables, tests and figures.
   - Each test shown next to its MDE.
   - Each figure with a one-line plain-language reading.

## 7. Files to change

| Area | File(s) | Change |
| ---- | ------- | ------ |
| Personas | `agents/personas.py` | Attribute catalog (§2.2); `compose_persona`; `sample_profiles`; C0 and C1. Keep the v2 personas. |
| Prompts | `agents/prompts.py`, `agents/parsing.py` | `ORDER_FORMAT_STUDY1_V3`; `reasoning` validation |
| Environment | `envs/exogenous.py` | `RULES_V3`; calibrated generator; full price history |
| Calibration | `calibration/` (new): `sources.py`, `prices.py`, `attributes.py`, `report.py`; command `calibrate` | §2.7 |
| Config | `config/study1.yaml` | `design: v3`; `profiles`; `stages`; `calibration.tej_path`; `max_output_tokens: 1200`; `budget.cap_usd: 11.56`; `model_prices.gpt-6-sol` |
| Runner | `sim/run_study1.py` | Persona × path × arm grid; structured profile fields in `run_start` |
| Dataset | `measure/dataset.py` | Profile id, attribute columns, reasoning text |
| Power | `analysis/power.py` (new); command `power-study1` | §2.5 |
| Tagging | `analysis/reasoning.py` (new) | §4.2 |
| Analysis | `analysis/study1.py` | §5 |
| Viewer | `app/streamlit_app.py` | §6 |
| Tests | `tests/` | Prompt rules; deterministic, balanced profile sampling; reasoning parsing; reproducible calibration; simulated σ and kurtosis match the targets; the power simulation recovers a known effect; v2 replay unchanged |
| Data | `.gitignore` | Ignore the raw TEJ file; commit `results/calibration/` |

Before any paid run, fix the corrupt cache file that `cache-stats` reports in `cache/study1/`.

## 8. Budget

**At most US$10 for this prototype.** `budget.cap_usd` is cumulative for Study 1, and $1.56 was
already spent on earlier runs, so set it to **11.56**.

| Stage | Limit |
| ----- | ----- |
| Pilot, `gpt-6-luna` (150 runs) | $1.0 |
| Pilot, `gpt-6-sol` (150 runs; reduce paths if the dry-run projection exceeds the limit) | $2.0 |
| Reasoning-tag classifier | $0.5 |
| Main run (size from §2.5) | $6.5 |

Every stage prints a dry-run projection before its first paid call. Shrink any stage that
would exceed its limit (fewer paths) before running it.
