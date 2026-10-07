# Review of the v32 run (`muni_ipca_research_v32.html`, v3.1 notebook, 2026-10-07) and presentation readiness

Same data as v4/v5/v31. New in this run: baselines E and F, the last-value diagnostics, rolling conformal,
PIT beta-space clusters and the Section 12 breakdown, roll-forward by segment, pooled filter as the record.
30 figures, no errors.

## 1. What the new blocks say

**The algo's own third term is right-sized; the persistence lives elsewhere.** The error regressed on the algo's
last-value adjustment gives -0.037 (t -3.9): a slight over-reaction, zero on the S side. The correlation between
our last-error feature and that adjustment is 0.09, and 0.06 for the EWMA. The 0.59 print-to-print persistence
is therefore information the quote does not use, not a mis-scaled version of a term it already has.

**Side intercepts add nothing; the EWMA is the whole gain.**

| held out, 538k trades | MAE (bp) | gain vs algo | FM t | boot t |
|---|---|---|---|---|
| A algo quote | 12.95 | | | |
| B algo + rho(age) x last error | 13.18 | -0.23 | -4.0 | -3.3 |
| C algo + rho x EWMA error | 12.26 | +0.68 | 8.1 | 6.1 |
| E algo + side intercept | 12.85 | +0.09 | 4.9 | 3.3 |
| F = E + rho x EWMA (revised third term) | 12.29 | +0.64 | 7.0 | 5.3 |
| D level model with error features | 11.44 | +1.55 | 21.1 | 13.7 |
| D- level model without error features | 11.76 | +1.26 | 15.8 | 10.1 |

Side biases are small (P +2 to +4 bp, D -1, S +0.5) and shrinking over the sample. F is C with noise added. The
deployable two-parameter object is C: algo + 0.8 x EWMA(past errors, half-life 3 prints).

**Where the gains live (Section 12).** The universe number hides a very uneven map.

| segment | F gain (bp) | t | D gain (bp) | share of trades |
|---|---|---|---|---|
| rating A | +4.02 | 9.0 | +4.16 | 14% |
| rating AA | +0.22 | 2.6 | +1.36 | 72% |
| rating AAA | -0.45 | -5.3 | +0.16 | 12% |
| bullets | +1.40 | 10.5 | +1.63 | 37% |
| priced-to-call | +0.13 | 1.2 | +1.64 | 55% |
| duration 4-6y | +2.01 | 12.2 | +2.26 | 20% |
| duration 8-11y | +2.88 | 2.7 | +3.20 | 4% |
| duration 6-8y | -0.28 | -4.4 | +0.01 | 24% |
| duration <1y | -1.15 | -2.6 | +3.99 | 12% |
| beta3 tercile high | +1.50 | 11.5 | +1.93 | 33% |
| beta3 tercile low | +0.07 | 0.5 | +1.43 | 33% |
| cluster C0 (1y bullets) | +2.04 | 2.4 | +1.81 | 13% |
| cluster C4 (1.5y bullets) | +1.30 | 7.7 | +1.78 | 12% |

Top characteristic cells by F gain, cells with at least 2,000 trades: 8-11y callable-to-maturity A-rated
(+22.1 bp, algo MAE 39 to 15, t 6.1, 3,129 trades); 4-6y priced-to-call A-rated (+18.4 bp, 32.6 to 12.0,
t 11.5, 7,357 trades); 1-2.5y bullet A (+4.1); 2.5-4y priced-to-call A (+2.8); 1-2.5y bullet AA (+1.79, t 11.2,
60k trades); 2.5-4y bullet AA (+1.73, t 7.2, 35k). Bottom: sub-1y priced-to-call (-1.8 to -3.5), 8-11y
callable-to-maturity AA (-2.7). The pattern: the algo's error is large and persistent in single-A callables and
in short bullets; it is already well-behaved in AAA and in the 6-8y belly, where a correction only adds noise.
The sub-1y and near-call segments have algo MAEs of 40 to 120 bp, which is yield-space noise on bonds whose
yield is barely defined, and they dominate any universe-level MAE.

**The algo's error loads on the slope factor, and the correction removes it.** Fama-MacBeth of the error on
the three betas: beta1 +12.5 (t 2.7, boot 1.8), beta2 -30.8 (t -4.2, boot -3.1), beta3 0. Under F the beta2
loading is 0.8 (t 0.1) and beta1 falls to 8.8; under D it flips to +17.6 (t 3.0), an over-correction. This is
the direct tie between the two layers: the algo mis-prices along the term/extension axis the factor model
identifies, the per-bond error history carries that mis-pricing, and correcting the history corrects the
factor exposure. It also says a direct factor-beta correction (algo + b x beta2, estimated on prior months) is
a baseline we have not run and should.

**The residual signal predicts marks, not prints.** The pooled filter is the best forecaster (IC 0.083, sign
accuracy 0.546, hedged Sharpe 6.78, selected every month, 5.80 walk-forward) and has no relation with prints:
+1.15 vs the mark (t 1.3), -0.14 vs the algo (t -0.2). The joint regression with the one-day and trailing ranks
is collinear (coefficients of +6.5, -11.1, +7.2) and should not be shown. Within clusters the IC is highest in
the sub-1y clusters (0.10 to 0.15) and the within-cluster Sharpe highest in C7 (short callables, 4.9) and C3
(4y callables, 3.7). The mark-quality reading stands at +41.7 bp per unit of filtered mark noise (t 14.9,
bootstrap 11.7).

**Roll-forward by segment.** RMSE reduction at five days is 56 to 71% for 1 to 11 years and 22% below one year,
which is 40% of bond-days and drags the universe figure to 36%. Callable-to-maturity 54%, bullets 32%. The
factor model moves intermediate and long marks; the short end is idiosyncratic.

**Rolling conformal.** Coverage recovers to 77 to 78% overall and 76 to 77% in September (split conformal 61 to
65%), and coverage across width deciles is 0.73 to 0.83. But the mean half-width is 16 to 18 bp against 10.4
to 11.5 for a fixed band that covers 75%, and the top width decile averages 70 bp. The fair benchmark, a fixed
band rescaled daily on the same ten-day window, is missing from the exhibit; without it the conditional band
cannot be claimed as better than a fixed one.

## 2. Presentation readiness

**Ready to present as findings.**
1. The factor model: 13 instruments, three legible factors (level 51%, slope 40%, yield tilt 9%), aligned
   loading sd 0.02 across versions, data rules that remove the two artefacts, OOS variance explained 0.36 with
   the monthly profile, cadence closed.
2. The residual as a mark-dynamics object: state-space description (tiny near-random-walk drift, mark noise,
   white noise; phi 0.97 quiet, 0.00 active), hedged Sharpe 6.8 as the information metric, mark-noise estimate as
   a mark-quality score, roll-forward of stale marks at 56 to 71% RMSE reduction for 1 to 11 years. Presented as
   risk and marks tooling, not as a pricing signal.
3. The honest negatives: AR(1), shrinkage, residual width, skew sizing, peer signal, flow hedging, the residual
   as a print predictor. Each closed with a number.
4. Algo error correction: persistence 0.59 on both side pairs; the quote's own adjustment is right-sized; the
   EWMA rule gains 0.68 bp universe-wide with bootstrap t 6; the gains concentrate in single-A callables (+18 to
   +22 bp, 2% of trades), 4-6y bonds (+2 bp, 20%) and short bullets (+1.8 bp, 18%); the algo's error loads on
   the slope factor and the correction removes it. The level model confirms with +1.55 bp and 0.3 bp from the
   error features.

**Not ready, fix before the deck.**
1. Economic translation. Management will ask what 0.68 bp or 22 bp is worth. The notebook has par size per
   trade and DV01 per bond; a par-weighted, DV01-weighted sum of |error| removed, by segment, in dollars per
   month, is one cell of work and is the slide that matters.
2. The universe MAE is dominated by sub-1y and near-call bonds with 40 to 120 bp errors. Report headline MAE
   excluding duration under one year, or in price terms, and say so.
3. The band exhibit needs the rolling fixed benchmark. Until it is there, present the band as in progress and
   the fixed band as current practice.
4. Drop the collinear joint regression and the by-bucket filter rows from the deck; keep the pooled filter and
   the bucket parameter table.
5. Add the factor-beta correction baseline (algo + b x betas) so the slope-loading result is closed in both
   directions.
6. Say the sample: five OOS months, one dispersion cycle, September as the only stress. The error-correction
   rho rises every month as training data accumulate, so the gain is likely understated, but one quarter of
   out-of-sample is one quarter.

**Suggested storyline, twelve slides.**
1. Question: can a characteristic factor model improve muni market-making pricing, and where.
2. Data and the two rules (one chart).
3. The factor model: Gamma anatomy and stability (two charts).
4. The residual: what it is (state-space), what it predicts (marks), what it does not (prints).
5. Mark tooling: roll-forward by segment, mark-quality score.
6. The pivot: the algo's own error persists (persistence by gap and side pair).
7. The algo's error loads on the slope factor (one chart).
8. The correction: A to D table, the decile chart, rho(age).
9. Where the gains live: heatmap, cluster bars, top cells.
10. Dollars: par- and DV01-weighted error removed by segment (to build).
11. What we closed and why (the negatives, one slide).
12. Next quarter: deploy C in the identified cells, shadow-run D, band work, out-of-sample continuation.
