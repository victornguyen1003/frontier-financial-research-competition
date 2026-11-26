# Regression issue

執行日程: November 14, 2026
子專案別: Model architecture
專案人員: Junjie
關鍵進展: No

## Purpose

<aside>
<img src="https://app.notion.com/icons/star_yellow.svg" alt="https://app.notion.com/icons/star_yellow.svg" width="40px" />

Review of our current regression against Barber, Lee, Liu & Odean (2007), "Is the Aggregate Investor Reluctant to Realise Losses? Evidence from Taiwan" (the paper we want to benchmark against). Lists the issues found and the decisions the team needs to make before Victor's prototype is built further.

</aside>

Status: proposal for team discussion. Nothing here has been agreed yet. The `Memory × G` interaction term and the design suggestions below are additions, not things we decided in meeting 1.

## Current model (from meeting 1 notes)

$$
P(Sold_{it}=1)=\Lambda\left(\alpha+\beta_1\,Memory_i+\beta_2\,G_{it}+\beta_3\,(Memory_i\times G_{it})+\beta_4\,dd_{it}+\gamma'X_{it}\right)
$$

The group post used the naive version: `Sold_it = memory_i + G_it + dd_it`.

## What the paper actually measures

The paper runs no regression. It is descriptive:

- **PGR** = realised gains / (realised gains + paper gains), i.e. the daily chance that a position held at a gain gets sold.
- **PLR** = realised losses / (realised losses + paper losses).
- **Disposition effect = PGR − PLR** (or PGR / PLR).
- Gain or loss is decided by current price vs. the **average purchase price**. If the purchase price lies between the day's high and low, the position is "indeterminate" and dropped. Counting starts the day after purchase. Sold positions leave the sample.
- Significance: mean of (PGR − PLR) across investors with a t-test, plus a daily time-series version with Newey-West standard errors.

Benchmarks we could compare against (Table 3):

| Group | PGR | PLR | Ratio | Note |
| --- | --- | --- | --- | --- |
| All investors (pooled) | 2.93% | 1.42% | about 2.1x |  |
| Individuals (pooled) | 4.39% | 1.67% | about 2.6x |  |
| All investors (mean across investors) | 9.40% | 2.32% | PGR − PLR = 7.08 pts | 84% of investors have PGR above PLR |
| Foreigners / mutual funds |  |  | no disposition effect | mutual funds slightly reversed |

With only `G` in the logit and the right data unit, exp(β₂) is roughly PGR / PLR. In log-odds terms the paper implies β₂ of about 0.74 (all investors) and about 1.0 (individuals).

## Main issues

### 1. The data unit is wrong: no "positions held" denominator

The paper computes rates over positions that are still held on each day. `Sold_it` at agent × day does not say which rows count. Days after an agent has sold (or before it has bought) must not appear.

**Fix:** use a position-day panel (agent × stock × day). Start the day after purchase. Drop the position once sold. The logit then becomes a discrete-time hazard model, the same object as PGR / PLR.

### 2. "Memory" is entangled with G and dd

We currently have two definitions of memory (whiteboard):

- **(1) Prompt architecture:** no memory / observation / observation + reflection. `Memory_i` is a clean dummy. Price paths are drawn independently, so `G` and `dd` are exogenous.
- **(2) Memory = price path put in context:** `Memory_i` is the path itself, but `G_it` and `dd_it` are computed from the same path. The three cannot be separated.

The model as written uses (1) for `Memory_i` but includes `dd` as if it were (2). The coefficients cannot be interpreted until we pick one.

**Suggestion:** use (1). It matches Victor's design and our research question.

### 3. G is not defined like the paper's

The paper's reference point is the average purchase price. We need a clear purchase price. Two options:

- **Assigned holdings:** the agent is given positions at a fixed price. G is determined by the random path, so it is exogenous (cleaner identification).
- **Agent chooses when to buy:** closer to real investors, but the purchase price becomes endogenous.

**Suggestion:** assigned holdings, several stocks so that gains and losses coexist.

### 4. We are comparing against ratios, not regression coefficients

The paper gives PGR, PLR and their difference. Our logit only maps to PGR / PLR if it contains `G` alone and uses the position-day data from issue 1. With `dd`, controls and the interaction, β₂ no longer equals the raw disposition effect (with the interaction, β₂ is the effect for the baseline "no memory" group only).

Also, an LLM agent that can sell every period will have a much higher base selling rate than real investors (the paper's daily PGR is only about 3 to 9%). Levels are not comparable.

**Fix:** report raw PGR, PLR, PGR / PLR and PGR − PLR per memory group and per agent (plus the share of agents with PGR above PLR, as in the paper), then run the logit as an extension. Compare ratios and signs, not levels.

## Secondary issues

1. **Interaction in a logit.** The sign of β₃ is not the interaction effect on probability. Report odds ratios or marginal effects per memory group.
2. **dd and G are correlated** (both come from the same path). Run a G-only baseline first, then add dd.
3. **Memorisation risk.** This paper is probably in the LLM's training data. A result close to the paper's 2x could reflect recall rather than behaviour. This is what Min-K% Prob and contamination checks are for, and it should be stated in the report.
4. **Power.** About 100 agents per memory group. Check detectable effect size in a pilot before the full run.

## What is fine already

- Logit for a 0/1 outcome.
- `G` as the disposition-effect variable.
- Memory × G as the test of whether memory changes the disposition effect.
- Clustering standard errors by agent.

## Proposed minimal design (for discussion)

1. Memory definition (1): no memory / observation / observation + reflection.
2. Agents are assigned several stocks at a fixed purchase price.
3. Paths: martingale with a small constant drift, generated at random; bootstrap of real 1995 to 1999 TAIEX returns as a robustness check.
4. Position-day panel; sold positions leave the sample.
5. Report raw PGR, PLR, PGR / PLR per memory group and per agent first, then the logit with Memory × G.
6. Compare with the paper's ratios (about 2.1x pooled, about 2.6x individuals, about 0 for foreigners and funds).