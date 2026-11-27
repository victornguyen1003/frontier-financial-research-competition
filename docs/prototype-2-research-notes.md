# Prototype 2: research notes

Background for [prototype-2-plan.md](prototype-2-plan.md). This file holds the motivation, the
reasons behind design choices, assumptions, limitations, future extensions, the timeline and
references. None of it is needed to implement or run the study.

## Research goal

Financial institutions are starting to use LLM agents as simulated customers. That is only
useful if the customer profile given to the agent actually changes its behaviour. If the agent
behaves the same way whatever profile it is given, it reflects the model's own defaults, not
the customer.

This study tests that on the **disposition effect (DE)**: the tendency to sell winning stocks
sooner than losing ones (Shefrin & Statman, 1985; Odean, 1998). The DE has been measured for
Taiwanese investors by Barber, Lee, Liu & Odean (2007). We compare direction and ratio with
their figures, not levels.

## Rationale for design choices

**Zero-drift prices.** Because returns are i.i.d. with zero drift, past gains and losses carry
no information about future returns. Selling winners sooner than losers therefore has no
rational justification in this environment. Volatility and tail thickness come from data. Only
the drift is fixed at zero, by design.

**Public state against memory.** Public prices and average cost are part of the state, not
memory, so every arm can tell a gain from a loss. An arm's DE therefore can't be zero just
because it lacks that information. "Memory" means only the agent's record of its own past
decisions. The `reflect` arm adapts the reflection step of Park et al. (2023).

**Persona sentences.** Each sentence describes a preference or characteristic in plain
language without naming any theory. Literature behind each attribute:

| #  | Attribute           | Literature                                                                            |
| -- | ------------------- | ------------------------------------------------------------------------------------- |
| 1  | Loss sensitivity    | Kahneman & Tversky (1979); "about twice" from λ ≈ 2.25 in Tversky & Kahneman (1992) |
| 2  | Mental accounting   | Thaler (1985)                                                                         |
| 3  | Regret aversion     | Shefrin & Statman (1985)                                                              |
| 4  | Realization utility | Barberis & Xiong (2012)                                                               |
| 5  | Financial literacy  | Dhar & Zhu (2006)                                                                     |
| 6  | Experience          | Feng & Seasholes (2005)                                                               |
| 7  | Professional        | Barber et al. (2007): institutions show no DE                                         |
| 8  | Age                 | TDCC investor statistics                                                              |
| 9  | Gender              | Barber et al. (2007), Table 4                                                         |
| 10 | Market knowledge    | —                                                                                    |
| 11 | Trading activity    | Barber & Odean (2000); Dhar & Zhu (2006)                                              |

**Data behind the attribute levels:**

- **Age:** the typical ages of younger and older investors in Taiwan's investor age
  distribution, instead of hand-picked ages.
- **Trading activity:** the active level matches the typical turnover of Taiwanese individual
  investors, 308–630% a year in Barber et al. (2007), which is roughly one round trip a month.
- **Experience:** the thresholds follow the categories used in survey data.

**Balanced draws.** Each attribute is drawn 50/50, so every attribute's effect is estimated
with equal precision.

**C1 bias cue.** C1 is a manipulation check: it shows the largest DE the model produces when
told to. It is not evidence of persona validity.

**Two-way clustering.** Attributes are assigned per profile, so clustering by path alone would
overstate the precision of attribute effects.

**Power target.** 80% power at δ = 0.05, with a Holm adjustment across 11 attributes. The
strictest level is 0.05 / 11 ≈ 0.0045, which implies SE ≈ 0.05 / (2.84 + 0.84) ≈ 0.014 and a
95% CI half-width of about 0.027. That is just inside the ±0.03 blind-spot band, so the design
can support blind-spot claims for effects near zero. Power is not recomputed after the main run;
the CIs and the MDE are reported instead.

**Two pilots, one main model.** Each pilot measures:

- cost per run and the incomplete-reply rate;
- the manipulation check;
- how much profile DEs spread around C0;
- the variance inputs for the power analysis.

The main run uses the model that passes the manipulation check and shows more persona
sensitivity. If the two models are similar, it uses the one whose power analysis gives the
cheaper design. If `gpt-6-sol` is chosen, its higher price buys fewer runs within $6.5, and the
study reports the MDE of the largest affordable design.

**Budget assumption.** Earlier runs cost about $0.0016 per Luna run. The reasoning field and the
full price history add tokens, so the plan assumes $0.0035–0.005 per Luna run until the pilot
measures it.

## Data sources

| Data                                         | Source                                                    | Used for                                           | Status                                                                                                                        |
| -------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Daily stock data                             | TEJ (`fulldata_1995to1999.csv`)                         | σ distribution, tail thickness, random-walk tests | **Have 1999 only**: the file covers 5 Jan – 28 Dec 1999 despite its name. Request 1995–1998 to match Barber's sample. |
| Barber et al. (2007), Tables 3–4            | Published paper                                           | Benchmark ratios; PGR/PLR by gender                | Have                                                                                                                          |
| Investor age and gender distribution         | TDCC investor statistics                                  | The two age levels                                 | To collect and verify                                                                                                         |
| Financial literacy and investment experience | FSC national financial literacy survey (金融知識普及調查) | Level definitions and thresholds                   | To collect and verify                                                                                                         |
| Preference estimates                         | Tversky & Kahneman (1992) and the papers above            | Numbers in persona sentences                       | Have                                                                                                                          |

**First look at the 1999 data** (TSE common stocks, 468 firms):

| Statistic                           | Value                                          |
| ----------------------------------- | ---------------------------------------------- |
| Monthly log-return σ per stock     | median 0.126, interquartile range 0.098–0.164 |
| Pooled monthly kurtosis             | about 11, so tails are fat                     |
| Daily return σ                     | 2.9%                                           |
| Share of stock-days at the up-limit | 4.9%                                           |

The σ = 0.10 used before is below the typical stock. One year is too short to test monthly
autocorrelation, which is another reason to obtain 1995–1998.

## Assumptions

- Synthetic random-walk prices are a fair test environment: no momentum or reversal, and
  nothing for the model to recognize. Their volatility and tails match TSE data; the zero drift
  is a design choice.
- Average cost is the investor's reference point, as in Barber et al. (2007) and in broker
  apps.
- One period is one month. Agents may trade every period.
- Each agent-run is independent, and agents don't interact.
- The pilot's variance estimates are close enough to the main run's for the power analysis to
  be valid.

## Limitations

- **Levels differ from Barber.** Our periods are months, not days, and agents decide every
  period, so only signs and ratios are comparable. The benchmark is from 1995–1999 and
  aggregated.
- **Contamination.** The model may know the DE literature. Prices are synthetic, but conceptual
  knowledge cannot be removed. The C1 manipulation check and neutral rule text limit, but do
  not remove, this concern.
- **Reasoning is not a cause.** Stated reasoning may not reflect what drove the decision. The
  main evidence is the outcome difference against the control.
- **Asking for reasoning may change behaviour.** Every arm gets the field, so comparisons stay
  fair, but levels may differ from agents that don't explain themselves.
- **No stability or robustness checks in this round.** Results come from one set of model
  draws, one calibration and one specification.
- **External validity.** One model, one market design, no fees or taxes, closing prices only
  (Barber uses daily highs and lows).
- **Attribute wording.** Each attribute is one sentence. Other wordings may give other effects.
- **Calibration period.** Until 1995–1998 data are obtained, prices are calibrated on 1999
  alone. The attribute-level data come from different years and sources than Barber's sample.
- **Balanced, not representative.** Profiles are drawn 50/50 on each attribute, so pooled
  results describe the design, not Taiwan's investor population. Compare Barber with the
  matching cells (for example C0, or retail non-professional profiles), not the pooled average.
- **Long prompts late in a run.** The full price history makes the state longer each month.
  It does so equally in every arm, and token counts are reported.

## Gaps found during implementation

Found while building the prototype (2026-11-27), before any pilot was run.

**Data**

- **TEJ file has holes.** It covers 1999 only, and within 1999 it has 10 trading days in July,
  none in August and 3 in September. Those three months are excluded from the calibration and
  listed in `results/calibration/prices.json`. That leaves 9 months per stock (328 TSE common
  stocks). A complete 1995–1999 export would fix both problems and match Barber's sample.
- **Stock count differs from the first look.** The file has 345 TSE common stocks and 328 with
  enough months, not 468 as in the table above.
- **"Kurtosis about 11" in the first look is raw pooled kurtosis.** Most of it comes from
  volatility differing across stocks, which the generator reproduces by drawing a σ per stock.
  Within a stock, monthly tails are only mildly fat (fitted Student-t df ≈ 7.7).
- **Daily returns are not quite a random walk.** Lag-1 autocorrelation is about 0.11 and VR(2)
  about 1.11, likely from the 7% daily price limit and thin trading. The zero-drift i.i.d.
  generator is a design choice, not a description of 1999 TSE prices.
- **Age split.** TDCC brackets are 0–19, 20–30, 31–40, 41–50, 51–60 and 61+, so "under 40 / 40
  and over" becomes "40 and under / 41 and over". The ages are 30 and 59.
- **Age data year.** The TDCC table is from October 2026, not Barber's 1995–1999 period.
- **Missing source URL.** The TDCC page URL is not yet recorded (`calibration.tdcc_source_url`).
- **No FSC data yet.** `data/attribute_data.csv` is still empty: the FSC literacy and
  experience figures have not been collected.

**Method changes from the plan**

- **Monthly returns** sum TEJ's adjusted daily log returns instead of using raw month-end
  closes, so ex-dividend drops do not count as losses.
- **Degrees of freedom** are fitted by matching simulated kurtosis, not by MLE. With 9 months
  per stock, the MLE on per-stock standardized returns gives df → ∞.
- **The pilot cannot estimate attribute effects.** With 8 profiles, 11 attributes plus a
  constant are not identified. The pilot analysis therefore reports only memory effects, and
  attribute effects need the main run.
- **Power analysis variance.** It uses a simple moments estimate of the variance components;
  the random-effects model (MixedLM) gave implausible values on pilot-sized data. The pilot
  cannot separate attribute effects from other profile differences, so the profile variance
  left after δ-sized attribute effects is an assumption.

**Pilot checks (Luna and Sol pilots, 2026-11-27)**

- **No leakage found.** All 2,520 decision prompts show prices only up to the current month;
  no future price appears anywhere, and runs share nothing except identical prompts.
- **The baseline model has a reverse DE.** Prototype 1 (v2, 900 Luna runs) gave pooled DE ≈
  −0.15 to −0.22, and C0 in v3 gives −0.40 (Sol: −0.11). Without persona cues, the models sell
  losers more readily than winners.
- **Positive DE comes from the persona sentences.** Profiles pool to +0.38 and C1 to +0.66.
  Several attribute sentences (realization utility, mental accounting, regret aversion) describe
  the preferences that produce a DE, so a positive d_k shows that the model follows a stated
  preference, not that it has the bias on its own. Results should be framed that way.
- **Contamination.** Luna names a theory ("loss aversion") in 3.5% of its reasoning texts, when
  paraphrasing the loss-sensitivity sentence.
- **C0 rarely trades.** Its level-0 trading-activity sentence ("rarely trade") leaves only 18
  sales across 15 Luna runs, so C0's DE is imprecise. Every profile with trading activity at
  level 0 shares this.
- **Not an artifact of timing or a few runs.** Only 3–5% of profile and C1 sales fall in the
  final month, and 89% of profile runs have DE > 0.
- **Unbalanced pilot draws.** In the 8 pilot profiles, financial literacy is on in 7 and loss
  sensitivity, realization utility, professional and trading activity in 6. This is chance;
  the main run's 30 draws are closer to balanced.

**Power analysis (Luna pilot)**

- **First version unusable.** Its MDE came out at 0.30–0.53, and joint power fell as P
  increased. Two inputs caused this. Baseline sell rates came from C0, which made 18 sales
  and has a reverse DE. And all profile variance not explained by δ-sized attribute effects
  was treated as noise, although the pilot's raw attribute differences (±0.3–0.6) suggest
  most of it is attribute effects.
- **Corrected inputs.** The baseline now comes from the profiles, and the residual
  heterogeneity is an explicit assumption: SD 0.1, with sensitivity runs at 0 and 0.2.
- **Results depend on that assumption.** The pilot cannot estimate it (8 profiles against 11
  attributes is saturated). For the main design (P = 40, S = 20; 500 simulations at SD 0.1,
  200 at 0 and 0.2), the Holm MDE for d_k is 0.041 (residual 0), 0.125 (0.1) or 0.230 (0.2),
  and the minimum power for d_k at δ = 0.05 is 0.91, 0.08 or 0.02. At residual 0, 40 × 15
  would already reach 80% power.
- **δ = 0.05 is not detectable unless the residual is near 0.** The memory test and the C1
  manipulation check have power near 1.
- **Design shape barely matters at a fixed cost.** 40×20, 60×13, 80×10 and 100×8 are within
  about 0.015 of each other in MDE.
- **Main run.** Following the plan's rule (largest affordable design on the pre-specified
  grid), it uses P = 40, S = 20: 2,520 runs, about $6.0.

**Budget and models**

- **Sol pilot does not fit its limit.** `gpt-6.2-sol` (the ID in the OpenAI project) reportedly
  has no Flex tier. At Standard prices ($2 / $10 per 1M tokens, unconfirmed) a 150-run pilot
  projects to about $14, against a $2 limit; even one path (30 runs) is about $2.8.
- **Cost estimates are provisional.** They assume 450 output tokens per call until the Luna
  pilot measures it.
- **Budget cap.** The cap is $10 for this prototype alone, not $11.56, because the
  prototype-1 cache was deleted.
- **v2 replay.** Design v2 prompts are checked to be byte-identical, but v2 can no longer be
  replayed from cache: the prototype-1 cache was deleted.

## Future extensions

- **Stability.** Rerun part of the design with fresh model draws (`llm.replicate_id`) and on new
  price paths. Report:

  - the correlation and ICC of cell-level DE between replicates;
  - the share of attribute effects that keep their sign and significance;
  - a scatter of replicate 1 against replicate 2.
- **Robustness of the environment and prompts:**

  - all stocks at the 25th and the 75th percentile of the empirical σ distribution;
  - prices from an i.i.d. bootstrap of demeaned real monthly returns;
  - the state table shows the first purchase price instead of the average cost (measurement
    still uses average cost);
  - the decision format without the reasoning field;
  - other wordings of each attribute sentence;
  - other model versions.
- **Robustness of the analysis.** These need no new runs:

  - logit instead of LPM;
  - without `ret`;
  - excluding the final period;
  - clustering by matched set (profile × path);
  - an attribute × memory interaction model.
- **Population-share draws.** Collect the Taiwan population share of each attribute level:
  gender and age (TDCC), institutional against individual share of trading (TWSE 交易人結構),
  and literacy and experience (FSC survey). Then either:

  - reweight the balanced profiles, using the product of each profile's attribute-level shares
    as its weight; or
  - draw a separate sample at population shares.

  Either one gives a Taiwan-representative aggregate PGR, PLR and ratio, compared with Barber's
  all-investor (2.06) and individual (2.63) ratios.
- **Viewer additions.** A stability tab, a robustness tab, and a population tab with shares by
  attribute level and the representative aggregate against Barber.

## References

- Barber, B. M., Lee, Y.-T., Liu, Y.-J., & Odean, T. (2007). Is the aggregate investor reluctant
  to realise losses? Evidence from Taiwan. *European Financial Management*, 13(3), 423–447.
- Barber, B. M., & Odean, T. (2000). Trading is hazardous to your wealth: The common stock
  investment performance of individual investors. *Journal of Finance*, 55(2), 773–806.
- Barberis, N., & Xiong, W. (2012). Realization utility. *Journal of Financial Economics*,
  104(2), 251–271.
- Dhar, R., & Zhu, N. (2006). Up close and personal: Investor sophistication and the disposition
  effect. *Management Science*, 52(5), 726–740.
- Feng, L., & Seasholes, M. S. (2005). Do investor sophistication and trading experience
  eliminate behavioral biases in financial markets? *Review of Finance*, 9(3), 305–351.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk.
  *Econometrica*, 47(2), 263–291.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks:
  Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41–66.
- Odean, T. (1998). Are investors reluctant to realize their losses? *Journal of Finance*,
  53(5), 1775–1798.
- Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior.
  *UIST '23*.
- Shefrin, H., & Statman, M. (1985). The disposition to sell winners too early and ride losers
  too long. *Journal of Finance*, 40(3), 777–790.
- Thaler, R. (1985). Mental accounting and consumer choice. *Marketing Science*, 4(3), 199–214.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of
  uncertainty. *Journal of Risk and Uncertainty*, 5(4), 297–323.
