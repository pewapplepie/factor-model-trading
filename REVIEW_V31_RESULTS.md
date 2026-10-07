# Review of the v31 run (`muni_ipca_research_v31.html`, v3 notebook, 2026-10-07)

Same data as v4/v5. New code: data rules, refit-cadence study, signal of record, algo error correction,
conformal band. 26 figures, no errors.

## 1. Data rules did their job

- Partial-mark rule caught exactly one date, 2026-04-22, the day behind the v5 leverage spike.
- Dispersion rule (relative to the median move) flagged 3 dates: 05-27 (the day behind the v4 June Gamma step),
  09-17, 09-28. Common-move rule flagged 8 dates and kept them. The old raw-share rule would have flagged 12,
  of which 8 were common moves. The two phenomena are now separated.
- Consequence for Gamma: with the September curve days back in the fit, the full-sample anatomy is textbook:
  factor 1 is the intercept (level, share 51%), factor 2 is years-to-worst and extension (slope, 40%), factor 3
  is the yield-level tilt (9%). Aligned loading sd across the 7 monthly versions is 0.020.

## 2. Refit cadence: Gamma is stable, cadence is second order, regime dependence is absent

| scheme | folds | OOS variance explained |
|---|---|---|
| weekly | 27 | 0.3562 |
| monthly (record) | 7 | 0.3561 |
| quarterly | 3 | 0.3549 |
| frozen at 2026-03-31 | 1 | 0.3533 |
| dispersion-regime-conditional | 7 | 0.3553 |

All five sit within 0.3 points of each other in every month. The v5 swap-test result (frozen beats in-force
in July and September) does not survive the corrected data rules: in v5 the September curve days were
zero-weighted and the comparison was distorted. The regime-conditional Gamma explains 0.29 on high-dispersion
dates and 0.51 on low-dispersion dates, but a single Gamma does the same: dispersion days are harder, Gamma does
not change with them. Close the question: monthly refit, one Gamma.

## 3. Residual signal of record

| forecaster | rank IC | t | sign acc | hedged Sharpe (vol-scaled) |
|---|---|---|---|---|
| trailing-20 mean | 0.082 | 8.0 | 0.508 | 2.90 |
| ridge L=10 | 0.070 | 5.4 | 0.535 | 2.61 |
| state-space pooled | 0.083 | 7.5 | 0.546 | 6.78 |
| state-space by bucket (record) | 0.060 | 5.5 | 0.526 | 5.74 |

- The pooled filter is the best forecaster on IC, sign accuracy and Sharpe, and the walk-forward selector picks it
  every month (6.20 over 102 days in v5 for the bucket version; 5.80 here for pooled). The by-bucket version,
  which was pre-registered as the record, is worse than in v5 (IC 0.082 to 0.060). Vol scaling already handles
  the active buckets at the portfolio level, so the per-bucket parameters mainly add estimation noise, and the
  UNKNOWN-bucket fit is degenerate (idiosyncratic sd 0.14 bp, drift sd 1.6). The bucket table itself is still
  the right diagnostic: phi 0.97 to 0.98 in L1 to L3, 0.00 in L4 and L5.
- Recommendation, to pre-register now: the pooled filter becomes the record signal in the next run; the
  by-bucket parameter table stays as a diagnostic.
- Monthly IC of the pooled filter: 0.19, 0.07, 0.08, 0.03, 0.05. August, the calmest month, carries the least.

## 4. Transaction tests

| score vs target | FM beta | FM t | boot t |
|---|---|---|---|
| one-day rank vs mark | +2.10 | 2.9 | 1.75 |
| trailing-mean rank vs mark | -3.13 | -4.1 | -2.67 |
| trailing-mean rank vs algo | -2.85 | -4.6 | -3.42 |
| record signal vs mark | -2.55 | -3.3 | -2.88 |
| record signal vs algo | -0.62 | -1.0 | -0.84 |
| joint vs algo: one-day / trailing / record | +0.8 / -4.1 / -0.5 | 1.1 / -3.9 / -0.5 | |

- The record signal is real against the mark and nothing against the algo. Its sign is negative: a high
  forecast of further mark cheapening comes with prints *below* the mark. The marks keep drifting, prints do
  not follow, which is the same thing the v5 print-to-print panel said (prints ignore the residual move).
- The trailing-mean rank remains the only residual score that survives the algo quote, and v5 established it is
  worth about 0.02 bp per trade. Nothing here changes that.
- The mark-quality reading is the robust residual result: |trade - mark| rises 41.7 bp per unit of the filtered
  mark-noise estimate (FM t 14.9, bootstrap t 11.7) and 2.6 bp per unit of |residual|/vol (t 7.9).

## 5. Algo error correction: persistence confirmed, the naive rule fails, the robust rule works

| held out, 538k trades | MAE vs print (bp) | gain vs algo | FM t | boot t |
|---|---|---|---|---|
| A algo quote | 12.95 | | | |
| B algo + rho(age) x last error | 13.18 | -0.23 | -4.0 | -3.3 |
| C algo + rho x EWMA error (half-life 3 prints) | 12.26 | +0.68 | 8.1 | 6.1 |
| D level model with error features | 11.44 | +1.55 | 21.1 | 13.7 |
| D- level model without error features | 11.76 | +1.26 | 15.8 | 10.1 |

- Persistence: 0.586 (t 22.9, bootstrap 17.7). By gap it decays from 0.42 at one day to 0.31 at two to four
  weeks. Same-side pairs 0.58, opposite-side pairs 0.51: this is a per-bond bias, not bid-ask bounce.
- rho(age) is stable and monotone: 0.54 (<=1d), 0.43, 0.40, 0.35, 0.10 (>21d), and rises across months as
  training data accumulates.
- And yet B hurts. The decile chart explains it: the mean error at the next print is flat near zero across
  deciles 2 to 9 of the last error and carries over only in the extreme deciles (decile 1 -12 bp overall and
  -37 bp on the S side; decile 10 +2.5 bp). A linear slope estimated on that spreads a tail effect across the
  body and adds a noisy single print to every quote. The EWMA removes the single-print noise and gains 0.68 bp
  with bootstrap t 6; the gradient-boosted model, which can be nonlinear in the error and condition on side and
  age, gains 1.55 bp, of which 0.30 bp is attributable to the error features (D minus D-). Every month, C and D
  beat the algo.
- By side: C gains 1.6 bp on P, 0.55 on S, 0.1 on D. D gains 2.5 / 1.3 / 1.1.
- Feature importance in D: last print spread, algo side spread, then last algo error third, EWMA error seventh.

**A caveat that must be resolved before the gain is claimed.** The P and S decile lines sit at about +5 bp and
-15 bp: prints on the dealer-purchase side are above the algo quote and prints on the dealer-sale side below it,
which is the half-spread if the algo quote is a mid. Any model that knows the trade side (D and D- both do)
earns part of its gain from learning that half-spread, and a side-aware error rule earns part of its gain the
same way. The data carry a `signal_side` field; if the algo quote is already side-specific this caveat is
moot, if it is a mid then the fair comparison is against algo plus a per-side rolling median adjustment. The next
run should add that baseline (E) and report D and C against it.

## 6. Conformal band: shape calibrated, level broken by the September regime shift

| band around the level-model mid, 80% target | coverage | mean half-width (bp) |
|---|---|---|
| fixed (calibration-month quantile) | 75.3% | 10.8 |
| raw quantile model | 73.5% | 13.9 |
| split conformal | 72.7% | 13.5 |

- By month: July 77%, August 80%, September 64% for the conformal band; the fixed band is 83 / 83 / 65. The
  conformal scale s came out 0.88 to 1.02, so the calibration month confirmed the quantile model and then
  September's error scale jumped. Split conformal guarantees coverage under exchangeability, and a sell-off
  month after a calm calibration month is exactly the failure of exchangeability.
- Across its own width deciles the conformal band is flat at 0.71 to 0.77 while the fixed band runs from 0.95
  to 0.29. The shape is right; the level is wrong by one regime.
- The conformal band is also wider on average than the fixed band at lower coverage: the top width decile
  averages 59 bp and the bottom deciles 4 to 7 bp. The quantile model over-stretches in the tail and
  under-predicts in the body.
- Fix: adaptive conformal with a daily update of s from the previous days' realised coverage (so September
  re-scales within days, not after the month), and a market-dispersion feature (lagged cross-sectional sd of the
  target) in the quantile model. Until then the fixed band is the more efficient band.

## 7. Where this leaves the programme

**Closed in this run.** Refit cadence and regime-conditional Gamma (no effect; monthly, one Gamma). The by-bucket
state-space filter as the record (pooled is better). The naive last-error correction (hurts).

**Confirmed.** Data rules. Gamma stability (loading sd 0.02). Persistence of the algo's own error (0.59, both
side pairs). Mark-noise estimate as mark-quality score (t 15). Roll-forward (20% to 40% RMSE reduction).

**Open, in order of value.**
1. The side caveat on the error-correction gain: add the per-side bias baseline and confirm what the algo quote is.
2. Adaptive conformal calibration with a dispersion feature.
3. Pooled state-space filter as the record signal; drop the UNKNOWN-bucket fit.
4. The EWMA error rule as the deployable two-parameter object, with its half-life pre-registered, and the GBM as
   confirmation.
