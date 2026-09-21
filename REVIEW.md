# Review: Muni Characteristic Factor Notebook

Scope: `muni_characteristic_factor.ipynb`, all 142 cells including cached outputs.
Two lenses: (1) factor-model research, with reference to Kelly, Pruitt and Su (IPCA)
and the attention-factors paper; (2) ML engineering of the training, fitting, and
evaluation pipeline.

## Headline numbers

| Result | Value |
|---|---|
| IPCA total R², full sample, K=3 | 0.77 |
| IPCA predictive R² (mean-factor forecast) | 0.03 |
| Residual lag-1 autocorrelation | 0.175 |
| Point-in-time AR(1) daily rank IC | 0.22 |
| Break-even round-trip cost, daily residual L/S | 4.8 bps |
| MSRB Test B D10-D1 (trade vs mark) | 5.6 bps |
| Final MSRB beta on RV rank, stale quintile L1 | -26.5 bp |
| Final MSRB beta on RV rank, quintiles L2 to L5 | -0.5 to +0.6 bp |
| Charge model MAE, no adjustment vs best learned | 10.13 vs 10.25 bp |

## Verdict

The research discipline is good and the conclusion is right. The residual is a
stale-mark correction signal, not tradeable alpha, and the notebook proves that
itself through the skip-a-day, active-vs-stale, cost, and MSRB tests.

Two things do not hold up:

1. Nothing after Section 8 is truly out-of-sample, because every residual comes
   from the full-sample Gamma.
2. The notebook cannot be re-run top to bottom, so its central claims are not
   currently reproducible. A fund research review would send it back on this
   point alone.

## Factor-model review

**This is not IPCA as Kelly, Pruitt and Su use it.** All 46 characteristics are
static one-hot dummies. With time-invariant instruments, IPCA collapses to a
rank-3 compression of daily bucket-mean returns. The paper's power comes from
continuous, time-varying characteristics ranked per date. Use tenor,
years-to-call, coupon, rating score, and yield spread as continuous
rank-normalised inputs.

**Wrong return space for the use case.** Clean-price returns on bonds with very
different durations mean factor 1 is duration times the daily curve move, which
is why tenor dummies dominate Gamma. The residual is then in price-return units
while every business target from Section 20 on is in yield bps. That unit
mismatch, with no DV01 bridge, is a plausible reason the charge model fails.
Yield is already pulled. Model the change in yield minus the MMD curve, or
DV01-scaled returns.

**Leakage.** Gamma is fit on the full window and its residuals feed the "frozen
OOS" AR pipeline (9.9 to 9.12), the executable-opportunity tests (15), the
calibration (16), the attention challenger (18), and the final MSRB regression.
Ratings are taken as the latest value and applied to all history. Only the AR
layer is point-in-time.

**K selection sits inside noise.** Total R² differences across K are 0.002 to
0.008 while the fold standard deviation is 0.11. Predictive R² is negative for
every K, and the three factor return series correlate at 0.80 to 0.88, so this
is effectively one duration factor plus rotation. Select K on paired per-fold
differences with a bootstrap, or on the downstream MSRB objective.

**Smoothing explains most of what looks like alpha.** Sharpe ratios of 3.2 on
factor portfolios and 7 to 9 on residual spreads, a lag-1 autocorrelation of
0.17 that vanishes by lag 3, and persistence rising from 0.08 in active bonds
to 0.32 in stale ones are all the signature of an evaluator's partial
adjustment toward trade prints. The final MSRB regression settles it: the
effect lives entirely in the stale quintile. Never quote a Sharpe computed on
evaluated marks.

**Section 16 calibration is broken at the top.** The pooled calibrated score
has negative rank IC yet a large decile spread, which means it ranks contexts,
not bonds. The reliability table predicts +11 bps for decile 10 and realises
-46 bps, and deciles 7 to 10 are not monotone. Fit isotonic curves on
within-context percentiles with minimum bin counts and shrinkage.

**The attention challenger cannot win by construction.** Softmax over dummy
scores yields near-uniform weights with effective N around 145,000 of 225,000
bonds, so the three "attention factors" are a reweighted market portfolio,
evaluated on 29 dates. This is not the attention-factors paper's model, which
needs learned embeddings, dynamic features, sparse top-k peers, and a trained
objective. Do not treat this result as evidence either way.

## ML engineering review

**Validation is inconsistent.** Single 70/30 time splits in Sections 9.6, 9.8,
10, and 22 coexist with expanding monthly folds in 9.10. K selection uses three
folds of 15 days. Gates use hard-coded thresholds, some redundant, and were set
after seeing results. Standardise on one expanding-fold scheme and one metrics
module with bootstrap confidence intervals.

**Reproducibility is the biggest gap.** Cached outputs come from at least four
run dates. The Q1 cell (10.7) fails with a NameError because a helper is defined
in a later section. Twelve code cells have no outputs, including the core
expanding-window fit in 9.10 and the revised Section 10 robustness cells, so the
parquet those sections read is from an unknown run. The weekly walk-forward is
switched off, yet Section 18 depends on its cached Gamma. State flows through
globals, pickles, and retrain flags. Move shared helpers into a package, and make
restart-and-run-all the only accepted evidence.

**The fit is far slower than it needs to be.** The Z tensor is about 5 GB in
float32, the IPCA refit takes 11 minutes, and the row-wise call-bucket apply
takes 4 minutes. With one-hot Z, every Z'Z is a count crosstab and every Z'r is
a per-bucket sum, so the whole IPCA iteration is a groupby that runs in seconds.
That speed-up is what makes proper daily walk-forward refits affordable.

**The charge model is set up to lose.** Ridge minimises squared error on a
target with MAE 10 and RMSE 31, then is scored on MAE, so outliers drag every
prediction. The context model also includes two identical side columns. Its
rank correlation of 0.09 shows there is signal but the level is miscalibrated.
Use a quantile or Huber gradient-boosted model per side with a monotone
constraint on the RV feature, scored on MAE and pinball loss over expanding
folds.

**Minor bugs.**

- Signal-to-trade lag is measured in calendar days, so the 2-day and 6-to-20-day
  buckets are empty.
- The first observation of every bond is zero-filled and then counted as
  observed.
- The IPCA random fallback initialisation is unseeded.
- Test A has redundant economic-gate thresholds (10 bps and 20 bps).

## Recommended steps, in order

1. Fix reproducibility: shared helpers in a module, restart-and-run-all, commit
   the environment and cache signatures. Nothing else is worth defending until
   this passes.
2. Rewrite IPCA to exploit the one-hot or bucketed structure, then make the
   whole chain point-in-time: Gamma before t, residual at t, AR from earlier
   residuals only.
3. Move to yield space with the MMD curve removed, so residuals, calibration,
   and the charge target share units.
4. Replace dummies with continuous rank-normalised characteristics and re-select
   K with paired fold differences.
5. Reframe the product explicitly as a stale-mark correction score. Model the
   evaluator's adjustment process directly, and validate only on trade prices.
6. Rebuild the charge model as a robust per-side GBM with staleness and
   trade-activity features, with proper folds and intervals.
7. Only then build the attention challenger the way the paper does, on dynamic
   features, with at least three test months.
8. Extend history to two or more years so the tests span more than one rate
   regime.

## Key insights to carry forward

- Residual persistence concentrates in low-volatility, stale marks. The final
  MSRB beta is -26 bp in the stale quintile and near zero elsewhere. That is an
  evaluator-lag signal, valuable for marking and quoting, not alpha.
- Predictive R² near zero means the factor model carries no expected-return
  content on this sample. Use it as a risk and peer decomposition only.
- Factor returns correlated at 0.8 to 0.9 mean K=3 is one duration factor plus
  noise; the K choice is within fold noise.
- Any Sharpe computed on evaluated marks is inflated by smoothing.
- Price-space residuals and yield-space targets are mixed without a DV01
  bridge; this alone can explain the charge-model failure.
- The attention v0 experiment cannot win by construction and is not evidence
  against the attention-factor approach.
