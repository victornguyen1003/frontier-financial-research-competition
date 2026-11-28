# Prototype 2: findings, conclusions and implications

Main run `study1_v3_gpt-6-luna`, 2026-11-28: gpt-6-luna (Flex tier, low reasoning effort), 40
sampled profiles plus C0 and C1, 3 memory arms and 20 calibrated price paths, for 2,520 runs
and 43,142 sale-day position rows. It cost $4.68, and prototype 2 cost $6.04 in total. The
reasoning-tag step (plan §4.2) was skipped, so claims about reasoning below rest on keyword
counts, not on the validated classifier. All numbers come from
`results/study1_v3_gpt-6-luna/analysis/` and `reasoning.parquet`.

## In brief

1. **The model has no disposition effect of its own.** With no persona (C0), DE is +0.05 (95%
   CI −0.18 to +0.23); in the no-memory arm it is −0.11. The base model sells losers at least
   as readily as winners.
2. **Personas create it.** The sampled profiles pool to DE = +0.45 (CI +0.40 to +0.50), and
   PGR/PLR = 6.2, against 2.63 for Taiwanese individuals in Barber et al. (2007). The DE comes
   almost entirely from the persona text.
3. **Sentences that describe a preference raise DE; sentences that describe knowledge or
   expertise lower it; age and gender do nothing.** That matches the direction of the empirical
   literature. The size reflects how literally the model follows a stated preference.
4. **The one surprise is informative.** Telling the agent that prices are random *raises* DE by
   0.23. It does so by removing the agent's own reason for selling losers, trend extrapolation,
   which leaves the cost-based reasons to decide.
5. **The core effects hold across models.** Five attributes are significant under the LPM and
   the logit and under every clustering of the standard errors. Four others depend on the
   specification.
6. **Memory design changes the result.** A plain trading record raises DE slightly (+0.05).
   Written lessons lower it (−0.13), because the lessons mostly warn against selling gains too
   early.

## RQ1: do persona attributes change PGR, PLR and DE compared with C0?

**Yes, strongly.** The joint Wald test of all 11 attribute effects rejects (F = 27.2, p <
10⁻⁸). In the paired comparison over paths, 28 of the 40 profiles differ from C0 after the Holm
adjustment. Profile DEs range from −0.15 (P26) to +0.79 (P02), with an SD of 0.26 in their
difference from C0. The manipulation check also passes: C1, whose persona states the
mechanism, raises DE by 0.55 over C0 (one-sided p = 0.0004).

The rise in DE comes from both sides. Profiles realize more gains than C0 (PGR 0.53 vs 0.28)
and far fewer losses (PLR 0.09 vs 0.23).

## RQ2: which attributes move the DE, and which are blind spots?

The figures are regression estimates of d_k, the change in P(sell | gain) − P(sell | loss),
with all attributes and memory arms in the model. SEs are two-way clustered by path and
profile.

| Group | Attribute | d_k | 95% CI | Holm-significant |
| --- | --- | --- | --- | --- |
| Preference stated | Mental accounting | +0.26 | +0.22 to +0.30 | yes |
| | Realization utility | +0.22 | +0.16 to +0.29 | yes |
| | Regret aversion | +0.12 | +0.08 to +0.16 | yes |
| | Loss sensitivity | −0.04 | −0.08 to −0.00 | no (p_Holm = 0.16) |
| Knowledge and expertise | Market knowledge ("prices move randomly") | +0.23 | +0.16 to +0.29 | yes |
| | Professional | −0.19 | −0.24 to −0.13 | yes |
| | Financial literacy | −0.08 | −0.13 to −0.03 | yes |
| | Experience (20+ years) | +0.00 | −0.05 to +0.05 | no |
| Demographics and habits | Trading activity (active) | +0.07 | +0.03 to +0.11 | yes |
| | Age (older) | −0.03 | −0.08 to +0.02 | no |
| | Gender (woman) | −0.02 | −0.07 to +0.02 | no |

**No attribute is a blind spot** (none has its whole CI inside ±0.03). However, the Holm MDE is
about 0.095, so the four non-significant attributes are inconclusive, not shown to be zero.

### Interpretation

- **Preference sentences work as instructions.** Mental accounting, realization utility and
  regret aversion each describe, in plain words, a preference that produces a DE. The model
  follows them: profiles with mental accounting refer to the purchase price in 53% of their
  reasoning texts, against 34% without it. These are the expected signs (Shefrin & Statman
  1985; Thaler 1985; Barberis & Xiong 2012). But a positive d_k shows the model complying with
  a stated preference, not having the bias.
- **Loss sensitivity is not a DE trait, as in theory.** A sentence describing loss aversion
  lowers DE slightly. Its loss-side effect raises PLR (0.04 → 0.12 in raw terms) and does not
  hold losers. Theory agrees that loss aversion alone does not imply a DE: prospect theory needs
  the S-shaped value function and the reference-point framing as well (Kahneman & Tversky 1979;
  Barberis & Xiong 2012). The model read "losses hurt" as "avoid further losses."
- **Sophistication lowers DE, as in the field data.** Professional and financial-literacy
  sentences reduce DE, the direction found by Dhar & Zhu (2006) and Feng & Seasholes (2005).
  Professional profiles mention "random" or "future" twice as often (61% vs 29% random) and the
  purchase price less often (35% vs 47%). Experience alone does nothing, so the model links
  sophistication to training and role, not to years of trading.
- **"Prices are random" raises DE by changing beliefs, not preferences.** Without it, 59% of
  profile reasoning texts appeal to trends ("declining", "momentum", "weak"). With it, 20% do,
  and agents trade half as often (0.27 vs 0.58 sells per month). The sales that disappear are
  mostly sales of losers: PLR falls from 0.14 to 0.02, while PGR rises from 0.49 to 0.59. Once
  the agent is told that past prices carry no information, the only remaining reasons to sell
  are its stated preferences, and those favour selling at a gain. This is the most useful
  finding of the prototype. The base model's reverse DE is a belief (it extrapolates trends and
  sells what is falling), not an absence of preference. It also parallels the argument that
  much of the field DE reflects beliefs and trading responses to past returns, not a pure
  preference over realizations.
- **Raw differences mislead here.** Profiles were sampled independently, so in 39 distinct
  profiles some attributes are correlated by chance. Realization utility has a raw DE difference
  of +0.01 but a d_k of +0.22; experience has +0.10 raw but 0.00 in the regression. The
  regression is the controlled estimate, and the raw table is descriptive only.

## Robustness: logit, odds ratios, marginal effects and clustering

The plan's model is the LPM with SEs two-way clustered by path and profile. As a check, the
same regressors were also fitted as a logit, and both models were re-estimated with SEs
clustered by path only, by profile (the investor) only, and by agent run (one persona × memory
arm × path). That gives 8 model × clustering combinations. Effects are compared on the
probability scale: the logit's average marginal effect on P(sell | gain) − P(sell | loss),
which is the counterpart of the LPM's d_k.

| Attribute | LPM d_k | Logit effect on DE | Logit odds ratio (gain:attr) | Holm-significant (of 8) |
| --- | --- | --- | --- | --- |
| Realization utility | +0.22 | +0.31 | 10.4 | 8 |
| Market knowledge | +0.23 | +0.31 | 25.5 | 8 |
| Mental accounting | +0.26 | +0.27 | 5.9 | 8 |
| Regret aversion | +0.12 | +0.14 | 2.8 | 8 |
| Professional | −0.19 | −0.17 | 0.32 | 8 |
| Loss sensitivity | −0.04 | −0.07 | 0.50 | 6 |
| Financial literacy | −0.08 | −0.07 | 0.80 | 6 |
| Trading activity (active) | +0.07 | +0.04 | 0.82 | 6 |
| Gender (woman) | −0.02 | −0.07 | 0.54 | 4 |
| Age (older) | −0.03 | −0.02 | 0.98 | 1 |
| Experience | +0.00 | +0.01 | 1.15 | 0 |

- **Five effects are robust.** Realization utility, market knowledge, mental accounting, regret
  aversion and professional are significant in every combination, with the same sign and
  similar size on the probability scale.
- **Four are fragile.** Loss sensitivity is significant in the logit but not in the plan's LPM,
  and the reverse holds for financial literacy and trading activity. Gender is significant in
  the logit only. Their signs never change, so their direction is consistent, but the evidence
  depends on the model.
- **Age and experience have no detectable effect** in any combination except one (age, LPM with
  agent-run clustering).
- **Logit odds ratios overstate effects where selling is rare.** Market knowledge has the largest
  odds ratio (25.5) but the same probability effect as realization utility. With market
  knowledge, losers are almost never sold (PLR 0.02), and odds ratios are large when a baseline
  probability is near zero. Read the probability effects, not the odds ratios, for sizes.
- **Clustering choice matters little except for agent-run clustering.** Path-only, profile-only
  and two-way SEs are similar. Agent-run SEs are the smallest (down to about half the two-way
  SE), because they ignore that a persona's 60 runs share a persona and each path's
  runs share prices. They are the only clustering under which age becomes
  significant (LPM). Loss sensitivity also passes in the LPM with path-only clustering. The plan's
  two-way clustering stays the primary choice.
- **Memory is robust.** Log raises and reflect lowers the DE in both models and all clusterings
  (logit: +0.07 and −0.13; joint p ≤ 0.0001 everywhere).

## RQ3: does memory change the DE?

**Yes** (joint Wald F = 16.6, p = 0.00007).

| Arm | Profiles PGR | Profiles PLR | Profiles DE | f_m (regression) |
| --- | --- | --- | --- | --- |
| none | 0.555 | 0.090 | 0.465 | (reference) |
| log (trading record) | 0.555 | 0.049 | 0.506 | +0.05 (p = 0.013) |
| reflect (record + lessons) | 0.468 | 0.134 | 0.334 | −0.13 (p = 0.0004) |

- **The record alone makes agents hold losers slightly more.** PLR falls by about 0.04, and
  sales fall from 0.47 to 0.39 per month. Seeing their own past trades makes agents more
  reluctant to trade, and the losers are what they stop selling.
- **Lessons debias.** The lessons written after months 5 and 10 most often concern the gain
  side. In 13%, agents criticize having sold winners too early; 21% say to judge each stock by
  its prospects, not its cost; under 1% mention losers. Agents then sell fewer gains (PGR
  0.56 → 0.47) and more losses (PLR 0.09 → 0.13), and buy more (0.18 vs 0.14 buys per
  month). Reflection works as self-correction against the persona's stated preference.
- **C0 behaves differently.** Its reverse DE in the no-memory arm turns positive with memory
  (+0.16 with log, +0.15 with reflect), but C0 trades so rarely that these cells are imprecise
  (CIs about ±0.25).
- **The second half of RQ3 is unanswered.** Whether reasoning refers to past trades needed the
  tag classifier, which was skipped. Keyword matching suggests few texts do, but it is a weak
  check: 4% in log, 4% in reflect and 2% in none mention a previous trade.

## Comparison with the field benchmark

- **The ratios are much larger than Barber et al.'s, so compare signs, not levels.** Pooled
  profiles have PGR/PLR = 6.2 and C1 has 8.3, against 2.63 for Taiwanese individuals; C0 has
  1.2, between the institutional values (0.69–0.86) and the individual ones (about 2.6).
  The settings differ too much for levels to match: 5 stocks, monthly decisions, 15 months,
  sale-day counting over our own paths, and a ratio inflated whenever PLR is near 0 (seven
  profiles never sold a loser).
- **The ordering matches the field.** Profiles described as professionals or as financially
  literate have lower DE, as mutual funds and foreign investors do in Barber et al.

## Limitations

- **The model follows instructions.** Every positive effect of a preference sentence can be
  read as compliance. The study measures how faithfully an LLM enacts a described investor,
  not whether LLMs have biases.
- **Many decisions are identical.** In the same path, arm and month, 50% of profile pairs
  choose exactly the same actions. The personas shift behaviour at the margin, but a shared
  default response dominates.
- **One model, one setting.** gpt-6-luna at low reasoning effort is the only model at scale.
  The Sol pilot (C0 and C1 only, 3 paths) gave the same pattern, with C0 at −0.11
  and C1 at +0.30, but Sol at Standard prices could not be run at scale under the budget.
- **Duplicate profile.** P37 and P39 share all attributes and, through the cache, all replies.
  The analysis treats 40 clusters where there are 39 distinct profiles, so SEs are slightly too
  small. The main conclusions do not depend on it: every Holm-significant attribute has
  p < 0.005.
- **Power.** The Holm MDE (0.095) is about twice the δ = 0.05 the power analysis targeted, and
  below the 0.125 it predicted for a residual SD of 0.1. Null effects are
  inconclusive.
- **Calibration.** Price paths are calibrated to one year (1999, nine complete months), and the
  generator is a zero-drift i.i.d. random walk by design.
- **C0's trading is thin.** C0 uses the level-0 trading-activity sentence ("rarely trade"), so
  it sells little and its DE is imprecise. Contrasts against C0 are less precise than those
  between attributes.

## Implications

**For research that uses LLM agents as simulated investors**

1. **A persona-free LLM is not a retail investor.** Its default is a trend-following,
   loss-cutting trader with no disposition effect. Results from agent-based markets built on
   default LLMs should not be read as human-like behaviour.
2. **The persona is the model.** Persona sentences move DE by up to 0.26 each, and profiles span
   almost a full unit of DE. Persona wording is therefore a key modelling choice and should be
   reported in full and varied systematically, as here.
3. **Beliefs matter as much as preferences.** A single sentence about how prices behave moved
   DE as much as the strongest preference sentence. Simulations should control what agents are
   told about the price process.
4. **Memory architecture is a treatment, not plumbing.** Reflection reduced the profiles' DE by more than a quarter.
   Two otherwise identical simulations with different memory designs would reach different
   conclusions about investor bias.

**For finance**

- The model's behaviour separates two routes to a DE: a preference route (realization
  utility, mental accounting, regret) and a belief route (whether past returns are seen as
  informative). Taking away the belief that losers keep falling produced a DE from preferences
  alone. This gives a cheap, controlled testbed for comparing belief- and preference-based
  explanations. It does not replace field evidence.

## Next steps

1. Draw profiles without replacement and run one replacement for P39 (about $0.12).
2. Rewrite the preference sentences so they describe feelings without the selling rule (for
   example, "you like to feel your decisions were right"), to test whether the model *infers*
   a DE rather than following one.
3. Run the reasoning-tag classifier (≤ $0.50) to answer the second half of RQ3 and to test
   the trend-extrapolation mechanism formally.
4. Run a second model family and a higher reasoning effort on a subset, to check that the
   pattern is not specific to Luna.
5. Obtain a full 1995–1999 TEJ export, to calibrate on Barber's period.
