# Review of the v4 run (`muni_ipca_research_v4.html`, 2026-10-06)

Data: Step 3 panel 1.89M rows, 77k CUSIPs, 2026-01-05 to 2026-09-30; model panel 1.77M rows, 53k CUSIPs,
181 dates; 13.7M matched trades. Walk-forward: 27 weekly Gamma versions, OOS 2026-04-01 to 09-30.

## 1. What held from v3 (now with the fixes in place)

| Block | v3 | v4 | Read |
|---|---|---|---|
| OOS variance explained (K=3) | 0.351 | 0.354 | unchanged, factors are not the bottleneck |
| Factor shares / factor-1 gross | 50/25/25, 5.1 | 50/25/25, 3.13 | duration-ratio fix holds; leverage fell further |
| Gamma loading sd across versions | 0.14 (raw) | 0.141 raw, 0.046 aligned | Procrustes removes two thirds; the rest is a real June shift |
| Residual ACF lag 1 Pearson / Spearman | -0.06 / +0.07 | -0.054 / +0.063 | increment reverts, rank persists |
| AR(1) rank IC | negative | -0.085 (t -5.2) | AR(1) is dead as a signal |
| Trailing-20 mean rank IC | +0.09 | +0.095 (t 9.2) | level signal, every month positive (0.06 to 0.15) |
| LongConv-lite ridge IC | +0.095 | +0.089 (t 6.7), sign acc 0.54 | filter is all positive beyond lag 1, negative at lag 1 |
| Trade vs prior mark, one-day rank | FM +2.54 (t 3.40) | +2.54 (t 3.40); neutralised +2.35 (t 3.18) | identical |
| Trade vs prior mark, trailing-mean rank | -2.48 (t -3.2) | -2.48 (t -3.17) | identical |
| Trade vs algo, one-day rank | ns | -0.70 (t -1.65) | the algo already has the one-day information |
| Spread reading: abs(trade - mark) on abs(resid)/vol | +2.7 (t 8.6) | +2.68 (t 8.55) | holds |
| Evaluator revision toward trade, Q1 -> Q5 | ~50% | 53.5% -> 49%, median pass-through 0.06 -> 0.00 | the evaluator does not respond to prints same day |

## 2. What the fixes changed

**Joint Fama-MacBeth (fix 3).** One-day rank +3.46 (t 5.0) and trailing-mean rank -3.41 (t -4.1) against the
prior mark, both larger than their marginal slopes. Against the algo quote: one-day +0.48 (t 0.95, nothing),
trailing-mean -2.73 (t -4.0). The two horizons are separate signals with opposite signs, and only the slow one
survives the algo. That is the single most actionable row in the notebook: a bond that has been cheapening in
the marks for 20 days trades about 2.7 bp per rank below the algo quote.

**MSRB side convention (fix 5).** Round trip P - S median +5.8 bp, positive as it should be. By side, trade vs
mark on the one-day rank: S +6.2 (t 6.0), D +1.9 (t 2.4), P -1.2 (t -2.0). The decile chart shows where it
lives: richest-decile bonds are sold to customers 25 bp below the mark, cheapest-decile 15 bp below; the P
line is flat near +7 bp. By trailing-mean rank the S line is U-shaped (-32 bp at the rich tail, -26 at the
cheap tail, -8 in the middle), and the round trip on the one-day rank is -7.5 bp per rank (t -10): dealers
make more on rich, in-demand bonds than on cheap ones.

**Paper portfolio on all signals (fix 6).** Hedged Sharpe: AR(1) raw -0.91, AR(1) winsor -3.11, trailing mean
+0.39, LongConv +0.78, trailing mean with the sign flipped in L4/L5 +3.74 (0.24 bp per unit gross per day).
The walk-forward selector picks the flipped trailing mean every month from May and prints 7.83 over 102 days.
Two honest caveats. The "fade" is a sign flip and the choice of L4/L5 was made after seeing the v3 transaction
results on this same sample, so 3.74 is partly in-sample; the selector chooses among signals but not the fade
buckets. And the rank IC in L5 is +0.10 while the L5 portfolio Sharpe is -2.1: ranks persist, dollars revert,
which is the two-regime plot in portfolio form. The economic size is small: 0.24 bp of yield per unit gross per
day is about 3% a year on gross before costs, which is not a stat-arb book; it is a measure of skew information.

**Print features cut at signal time (fix 1).** The level-model gain over the algo fell from 4.2 bp to 1.13 bp
(MAE 11.88 vs 12.96; FM t 14.3). It is now uniform across recency, 1.1 to 1.35 bp in every bucket up to 21 days
and 0.36 bp (t 1.9) beyond 21 days, and positive in every month. That uniformity is the signature of a model
that has lost its leakage: the old ">4 bp at <=1d" was the look-ahead. Feature importance is led by
last_print_spread_bp and then dMmdSprdSide. If dMmdSprdSide is the algo's own side-adjusted spread to MMD,
the level model is a stacking correction on the algo quote rather than an independent pricer. Still useful,
since a post-processor is exactly how it would be deployed, but the claim changes. Please confirm the field.

**Model as mid with inverse-variance mark weight (fix 2).** This is the clearest negative result of the run.
Model mid MAE 12.38, mark 23.37, inverse-variance blend 15.19 (worse than the model by 2.6 bp, t -18.6),
one-parameter blend 13.72 with lambda at the grid cap of 64 in three of four months. The implied mark weight
of 0.39 is what Gaussian precision weighting says, and it is wrong, because the mark's error and the model's
error are correlated (the model uses mark-derived features) and the errors are heavy-tailed, so variance
weighting is not the MAE-optimal rule. The mark should enter as a feature only. The shrinkage-estimator framing
in the proposal should be dropped.

**Band around the model mid.** Residual-driven band 81.1% coverage at 15.3 bp half-width vs fixed band 81.3%
at 15.5 bp. No gain. The band slope on |resid|/vol decayed from 1.35 to 0.52 across months. Combined with the
spread reading, the lesson is precise: |resid|/vol measures how wrong the mark is, not how uncertain the
print is. Once the mid is the model, the residual has no width information left.

**Residual size is not monotone in uncertainty.** The middle panel of the application exhibit shows both error
curves are U-shaped in |resid|/vol: the near-zero bin has the highest RMSE (80 bp for the model, 90 for the
mark). The reason is in the panel QA: 20.5% of bond-days have an unchanged mark. A residual of zero is often a
stale mark, not a fairly priced bond. The mark-unchanged flag and days-since-mark-moved belong in the level
model and in any width rule.

## 3. What the exhibits show that the tables do not

- **Regime.** OOS variance explained is 0.15, 0.17, 0.12 in April to June and 0.50, 0.46, 0.65 in July to
  September. The weak quarter is the high-dispersion quarter (target sd 7.7 bp). The aligned factor-1 loadings
  shift once, at the start of June (extension 0.27 -> 0.41, closing yield -0.26 -> -0.19). Three days carry
  15 to 20% of bonds with |resid| > 10 bp (late April, late May, late September); those are evaluator-wide
  repricing days and they drive the Pearson lag-1 negativity and most of the "tails revert" mass. The
  AR(1) and two-regime results should be re-run excluding days where that share exceeds 10%.
- **Beta-space map on 2026-09-30.** Duration and call structure organise the space; rating does not (AA is
  everywhere, consistent with its 0.06 Gamma norm). The residual-rank panel is the important one: the whole
  short-bullet region is rich (rank below 0.2) and the mid-duration callable region is cheap. On that date
  the residual still carries a curve move the three factors missed. Global residual rank on a stress day is
  partly a missed factor; rich/cheap should be read within cluster (the C6 exhibit does this) and the FM tests
  should add a within-duration-bucket version.
- **Top mimicking weights.** The same dozen bonds (2026-27 bullets short, 2031-34 priced-to-call long) top all
  three factors with weights of 0.003 against a breadth of 5,000 to 6,000. The factor portfolios are highly
  diversified and the top-names exhibit shows only extreme betas. A weight profile by duration and call bucket
  would say more.
- **Cheap tail is mark errors.** Within-cluster cheapest residuals are +95, +78, +58 bp while the richest are
  -3 to -4 bp. The positive tail is one-off evaluator repricings. That asymmetry is why "cheap" needs a
  size cap in any signal.
- **Roll-forward.** Rolling a stale mark by the factor-implied change cuts MAE from 3.3 to 1.6 bp at one day
  and 15.8 to 6.6 bp at ten days (RMSE reduction 20% to 40%). In-panel, so an upper bound, but this is the
  most immediately useful systematic output and the one the evaluator-revision result motivates: the marks
  are slow and the factor model can move them.
- **Sample coverage.** 624k of 6.74M OOS matched trades join to a prior residual (9%), because the residual
  panel is the covered daily-mark universe. Every transaction result is about that universe.

## 4. Against the proposal and the paper

| Proposal claim | Status |
|---|---|
| IPCA residual as a rich/cheap signal | Confirmed against prints at one day (+2.5 bp/rank vs mark) but already in the algo |
| AR(1) on the residual, refit monthly | Rejected: negative IC, negative Sharpe, in every variant |
| Residual level (trailing mean) carries the information | Confirmed, with a sign flip at the slow horizon vs trades, and it survives the algo (t -4) |
| Attention/LongConv comparison | LongConv-lite matches the trailing mean on IC; the learned filter is a near-flat positive window, so it is rediscovering the mean |
| Residual magnitude as a spread overlay | Half true: it predicts the mark's error, not the print's dispersion around a good mid |
| Shrinkage of the mark toward the model | Rejected; the mark is a feature, not a prior |
| Supervised level RV (pivot) beats the algo where it is weak | Beats it by ~1.1 bp everywhere, not more where the algo is weak; depends on whether dMmdSprdSide is the algo's own spread |
| Factor roll-forward for stale marks | Strongest systematic result (60% MAE reduction at 10 days, in-panel) |

## 5. Stage

Validated-signal, pre-production design. The factor model is stable and legible, the point-in-time plumbing is
clean, the two-horizon residual signal has survived prints and the algo quote at the slow horizon, and three
application ideas (AR(1), mark shrinkage, residual-driven width) have been killed cleanly rather than left
ambiguous. What is not yet done, in order of value:

1. Confirm what dMmdSprdSide is, and refit the level model with and without it. The 1.1 bp claim depends on it.
2. Economic sizing of the slow-horizon skew: apply -2.7 bp per rank of trailing-mean to the algo quote on the
   RFQ sample and measure hit rate, adverse selection and bp captured, by side. This is the deployment test.
3. Robustness: exclude the three repricing days; within-duration-bucket FM; block bootstrap over weeks (the
   125 FM dates share 27 Gammas and the IC is autocorrelated within month).
4. Add the mark-unchanged flag and days-since-mark-moved to the level model and re-run the band.
5. Pre-register the fade: fix L4/L5 now and let the next quarter of data judge it.
6. Only one regime of out-of-sample data (five months, one dispersion cycle). The June Gamma shift says the
   model should be judged again after a full rates cycle.
