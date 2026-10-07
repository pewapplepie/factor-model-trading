# Review of the v5 run (`muni_ipca_research_v5.html`, v2.3 code, 2026-10-07)

Same data as v4 (1.77M model rows, 53k CUSIPs, 181 dates; 13.7M matched trades; 27 weekly Gamma versions, OOS
2026-04-01 to 09-30). Code changes: date-weighted ALS with repricing days excluded, Gamma swap test, peer RV,
state-space and ARMA forecasters, vol-scaled portfolios, bootstrap t, print-to-print panel, skew sizing,
level-model ablations, conditional band, dealer-flow decomposition, K sensitivity.

## 1. Factor model: the June step was an artefact, and weekly refitting is too often

| Check | v4 | v5 |
|---|---|---|
| OOS variance explained (K=3) | 0.354 | 0.354 (monthly profile identical) |
| Aligned loading sd across versions | 0.046 | 0.021 (raw 0.099) |
| Factor-1 loadings path | step at 2026-06-02 | flat across all 27 versions |
| Factor shares | 50/25/25 | 45/30/26 |
| Factor-2 gross / breadth | 2.96 / 5,145 | 1.00 / 7,072 |

- Zero-weighting the repricing days removes the June step completely. Factor 2 is now the intercept alone
  (loading -0.98, gross weight 1.0, breadth 7,000): the equal-weighted market factor. Factor 1 is term against
  yield level (years-to-worst +0.72, yield -0.57, extension +0.34). Factor 3 is the yield-level/extension
  curvature. This is the cleanest identification of the three runs.
- Gamma swap test: a Gamma frozen two months earlier explains more OOS variance than the version in force in
  July (0.537 vs 0.502) and September (0.734 vs 0.659), and the first OOS Gamma (fitted on Q1 only) does as
  well or better everywhere. Mean 0.463 frozen vs 0.436 in force. Weekly refitting on an expanding window is
  not helping; the structure is stable and the refits add noise, or the Gamma is regime-specific (Q1's
  dispersed regime fits September's sell-off better than a Gamma that spent the summer learning calm markets).
- K=4: 0.361 vs 0.354; duration tilt on repricing days 0.116 vs 0.129; nothing on normal days. K=3 stands.
- Two data issues the rule did not catch. (a) The repricing rule flagged 12 dates with 40 to 90% of bonds moving
  more than 10 bp, six of them in the last week of September. Those are common moves the intercept factor
  absorbs anyway, so excluding them costs cross-sectional information for no gain; weighting is enough. The
  rule should flag dispersion after removing the day's median move, not the raw share. (b) One late-April date
  has a 20% share of |residual| > 10 bp with a Frobenius-leverage spike of 0.16 against a baseline of 0.05: a
  partial-mark day with few bonds that passed the 1,000-bond floor. A rule on bond count relative to the
  trailing median is needed.

## 2. Residual dynamics: the state-space model wins and derives the fade

| Forecaster | rank IC | t | ex repricing | hedged Sharpe (vol-scaled, ALL) |
|---|---|---|---|---|
| AR(1) raw | -0.070 | -4.3 | -0.061 | -1.17 (ex-UNKNOWN +1.37) |
| Trailing-20 mean | +0.093 | +9.3 | +0.094 | 2.87 |
| LongConv-lite ridge | +0.081 | +6.2 | +0.081 | 2.32 |
| ARMA(1,1) pooled | +0.092 | +7.1 | +0.086 | 0.71 |
| Peer RV spread | +0.012 | +0.9 | +0.021 | -0.14 |
| State-space, pooled | +0.097 | +9.2 | +0.096 | 5.11 |
| State-space, by bucket | +0.082 | +7.7 | +0.083 | 5.53 |
| Trailing mean, L4/L5 flipped | | | | 3.61 |

- Pooled parameters: phi 0.99 to 1.00, drift sd 0.003 to 0.15 bp, mark-noise sd 0.43 to 0.65 bp, idiosyncratic
  sd 1.8 to 2.3 bp. The daily increment is mostly white noise plus a mark-noise difference; the persistent
  component is tiny and near a random walk, which is why a trailing mean is nearly optimal and why ARMA(1,1)
  collapses to an EWMA (phi 0.999, theta -0.973).
- By bucket (last fold): L1 to L3 phi 0.97 to 0.98 with drift-to-noise 0.2 to 0.5; L4 phi -0.02; L5 phi 0.23
  with drift sd 1.69. In active names the "drift" is a transient and the optimal forecast flips sign. The fade
  we pre-registered is what the data imply, and the by-bucket state-space filter is now the walk-forward
  selector's choice in every month (Sharpe 6.20 over 102 days, 0.33 bp per unit gross per day).
- Vol scaling fixed the ALL-versus-bucket discrepancy: the UNKNOWN bucket was the culprit (AR(1) raw ALL -1.17,
  ex-UNKNOWN +1.37).
- The negative lag-1 autocorrelation survives excluding repricing days (-0.055). It is the mark-noise MA term,
  not the repricing days.
- The state-space mark-noise estimate predicts |trade - mark| at +37 bp per unit (FM t 14.9, bootstrap t 11.7),
  the strongest mark-quality indicator in the notebook. Days since the mark moved does not (t -1.5).

## 3. Transaction tests: statistically real, economically negligible

| Score vs target | FM beta | FM t | boot t |
|---|---|---|---|
| one-day rank vs mark | +1.84 | +2.54 | +1.55 |
| one-day rank within duration quintile vs mark | +2.44 | +3.15 | +1.83 |
| trailing-mean rank vs mark | -3.08 | -4.05 | -2.69 |
| trailing-mean rank vs algo | -2.81 | -4.60 | -3.46 |
| peer RV rank vs mark | -17.1 | -23.2 | -15.1 |
| peer RV rank vs algo | +0.99 | +1.86 | +1.71 |
| joint vs mark: one-day / trailing / peer | +2.68 / -0.74 / -16.5 | 4.3 / -0.9 / -18.9 | |
| joint vs algo: one-day / trailing / peer | +0.23 / -3.01 / +1.04 | 0.5 / -4.3 / 1.6 | |

- The one-day residual signal against the mark weakened with the new Gamma (+2.54 to +1.84) and does not
  survive the block bootstrap (t 1.55). The trailing-mean signal does, against both the mark and the algo.
- The peer RV score dominates everything against the mark: a bond that is cheap to its characteristic-space
  peers trades 17 bp per rank below its own mark. Against the algo it is nothing. The algo already prices to
  comparables; the mark does not. In the joint regression the trailing-mean reversal against the mark is
  absorbed by the peer score, but against the algo it is the only survivor.
- Economic sizing (9c) is the decisive negative. Applying the walk-forward skew to the algo quote raises MAE by
  0.10 bp (t -12) in every version, with a 50% hit rate. The slope of -3 bp per unit rank produces a skew of
  about 1 bp against 13 bp of print noise; the expected MAE gain from a signal that size is about 0.02 bp, and
  month-to-month variation in the coefficient eats it. Statistical significance over 490k trades is not
  economic significance at the trade level. The slow-horizon skew should not be sold as a quote improvement.

## 4. Print-to-print: prints follow the factor move, not the residual move; the algo's own error persists

- Pooled: next print change on factor move +0.83 (t 16), on residual move +0.01 (t 0.3), on the prior
  print-mark gap -0.38 (t -15). Prints follow the factor-implied move and ignore the residual move in the marks
  at short gaps; by 14 to 30 days the residual pass-through rises to 0.35 (t 4.5). The print-mark gap closes
  38% from the print side: the mark is more right than wrong, but it does not catch up (same-day pass-through
  remains zero in the evaluator test).
- Algo pricing error persists print to print: coefficient 0.585 (t 23, bootstrap 17.6). A bond the algo
  mis-priced at the last print is mis-priced the same way at the next one. An explicit error-correction term
  on the algo (last algo error times about 0.6, decayed by gap) is the largest untapped improvement in the
  notebook; by the usual variance argument it could take the algo MAE from 11.4 toward 9 to 10 bp, versus the
  1.1 bp the level model finds.
- Next-print MAE (bp): algo 11.4 | prior mark 17.2 | print rolled by factor 17.3 | stale print 18.7. The factor
  roll-forward is real (28.6 to 22.8 bp at 14 to 30 day gaps) but it is a fallback for when no algo quote
  exists, not a competitor.

## 5. Level model and application

- Full model gain over the algo 1.14 bp (t 13.9). Ablations: without algo-derived fields -0.32 bp (worse than
  the algo); without print features -3.4 bp; without staleness, peer and residual features +1.04 bp (unchanged).
  The level model is a stacking correction on the algo quote plus the last print. The factor-model, peer and
  staleness features contribute nothing to it.
- Mark shrinkage remains dead (ivw blend 15.3 vs model 12.4; lambda at the grid cap).
- Conditional band (10c) failed as implemented: 71% coverage at the 80% target, same mean width as the fixed
  band, top decile blown out to 72 bp by outliers, coverage rising with predicted width instead of flat. The
  quantile model is mis-calibrated under the month-to-month shift in error scale. The fix is conformal
  scaling: multiply the predicted quantile by the factor that gives target coverage on the most recent
  calibration month. Until then the fixed band is the best band.
- Dealer flow: 9% of flow-weighted variance is factor variance. Inventory risk from customer flow is
  name-specific; a three-factor hedge removes little. Mean exposure to the market factor 0.37 with net flow
  negative (dealers are net distributing).

## 6. Where this leaves the programme

**Confident.** The IPCA model as a risk model and roll-forward engine: stable, legible, a clean market factor,
residuals orthogonal, now robust to repricing days. The state-space description of the residual: tiny
persistent drift plus mark noise plus white noise, with the fade in active names derived rather than chosen.
The mark-noise estimate as a mark-quality score. Prints confirm the factor move and not the residual move.

**Closed negative, cleanly.** AR(1) as a signal. Mark shrinkage. Residual-driven width. The slow-horizon skew as
a quote adjustment (statistically real, 0.02 bp of value). Factor hedging of dealer flow. The peer RV score
against the algo (the algo already has it). The IPCA residual, staleness and peer features inside the level
model.

**Open and valuable.** (1) Algo error-correction: the 0.585 persistence of the algo's own pricing error from
print to print. (2) Gamma refit cadence and regime dependence: frozen beats weekly. (3) Conformal calibration
of the conditional band. (4) The repricing rule (dispersion after the median move) and a partial-mark-day
rule. (5) The by-bucket state-space filter as the residual signal, with the vol-scaled hedged portfolio as the
information metric (Sharpe 5.5, 0.33 bp per unit gross per day).

**Stage.** The factor model is production-grade for risk and roll-forward. The residual programme has reached
its ceiling as a pricing signal against this algo: the information is real, the economic value at the trade
level is not. The pivot the v4 review pointed to is confirmed by the data: the value is in the level layer, and
inside the level layer the biggest lever is the algo's own persistent error, not the factor model.
