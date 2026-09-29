# Review of the Revised Notebook (v1)

**Subject:** `muni_characteristic_factor_revised` (HackMD folder AlgoTrading / revised,
notes `-1` to `-4`, exported 2026-09-29).
**Relationship to other documents:** this is a *new* review of the revised notebook.
`REVIEW.md` is the original review of `muni_characteristic_factor.ipynb` and the
source of the step numbers used below. `RESEARCH_NOTE.md` is the management note
on the original notebook and has not been updated for these findings.

The four HackMD notes are one notebook export split into byte-chunks; rejoined, it
parses into 58 cells (36 code). Every code cell carries output. Cell numbers below
refer to that parse order.

---

## 1. Verdict

The revision is a strong engineering rebuild of the review's Steps 0, 1, 4, 5b and
6-7. The walk-forward IPCA machinery, the residual-maker identity, and the artifact
framework are correct and reusable.

The headline economic conclusions do not follow from the results the notebook
contains:

- The stale-quintile MSRB effect quoted in the executive summary and conclusions
  comes from a price-space run whose code is gated off in this notebook. The
  yield-space model the notebook actually runs shows no effect.
- The yield-space factor structure is dominated by a collinearity artifact between
  duration and DV01.
- Several of the new Step 5b validations are circular or mechanically guaranteed.
- A pipeline cell's recorded output does not match its code, and that stale run
  corrupted the Step 2 research panel on disk.

None of this undermines the machinery. It means the conclusions section must be
rewritten from the yield-space outputs, and three modelling fixes are needed before
the yield-space chain is evidence for anything.

---

## 2. What the revision delivers

| Review step | In the revision | Assessment |
|---|---|---|
| 0. Reproducibility | `RunConfig` dataclass, config hash, seeds, artifact manifest, cache keys with manifests, gating flags; all cells executed | Done |
| 1. Fast walk-forward IPCA | `MomentIPCA` on precomputed cross-products, including the identification rotation (QR, diagonal factor covariance, non-negative means); fits in 0.06 s; weekly refits tagged by `gamma_version` | Done, correctly |
| 2. Continuous characteristics | Lagged, per-date rank-normalised modified duration, premium/discount, closing yield, DV01 | Done; set is incomplete and collinear (Issue 2) |
| 3. Yield space | Target is daily closing-yield change in bps from OneTick marks | Done; target is quantised (Issue 3) and coverage excludes stale bonds (Issue 4) |
| 4. Frozen spec and registry | Spec JSON with hash, PIT residual score, pair score, validation registry, MSRB snapshot | Done; registry has a key bug (Issue 7) |
| 5b. Systematic path | Z/beta table, explicit mimicking weights, residual-maker, book exposure, roll-forward, beta-space peers, peer-implied yield | Done; several validations circular (Issues 5, 6) |
| 6-7. Robust charge | Fold-winsorised conditional medians, isotonic decreasing calibration, expanding weekly folds, 1-day embargo, per-fold gate, bootstrap CI | Done, and the verdict is honest |

Checks that pass exactly:

| Check | Result |
|---|---|
| Residual-maker reproduces walk-forward residual | max abs diff 0 over 463,188 rows |
| Orthogonality of residual to betas | max abs B'eps = 0 |
| Walk-forward folds | 12 weekly folds, 12 Gamma versions, train strictly before test |

---

## 3. Issues, most serious first

### Issue 1. The headline MSRB result does not come from this notebook

The executive summary (cell 0), the Step 5 economic basis (cell 36), and the
results table (cell 57) all cite a stale-quintile effect of about -22 bp per rank.
That number appears only in prose in cell 19, describing a price-space Step 1 run.
Step 1 is gated off (`RUN_STEP1_BASELINE = False`), so the run has no output here.
Cell 36 then attributes the number to the Step 3 yield-space regression, which is
incorrect.

| MSRB beta on residual rank | Quoted in conclusions | Actual yield-space output (cell 32) |
|---|---|---|
| Overall | -1.43 bp, t = -4.1 | +0.30 bp, t = +0.6 |
| Stale quintile L1 | -22.3 bp, t = -5.6 | -0.19 bp, t = -0.2 |
| Active quintile L5 | +0.16 bp, t = +0.6 | +2.28 bp, t = +1.2 |

The Indications section also cites "rank IC ~0.19", which appears in no output.

**Fix.** Rewrite cells 0, 36, and 57 from the yield-space outputs. Label the
price-space figure explicitly as a prior-run result until Step 1 is re-executed in
this notebook, or re-execute it.

### Issue 2. Factor 1 is a duration-DV01 collinearity artifact

| Evidence | Value |
|---|---|
| Correlation of duration rank and DV01 rank | 0.998 |
| Factor 1 Gamma loadings | +0.71 duration, -0.70 DV01 |
| Factor 1 share of factor variance | 99.6% |
| Factor 1 time series (smoke fit) | mean +23, std 62, max +465 bp |
| Factor 1 mimicking-portfolio gross leverage | 38.6 |

Factor 1 is the small difference between two nearly identical instruments, scaled
up to fit. The cell 25 read "duration is no longer the dominant Gamma row,
consistent with the review" is not supported: the L2-norm ranking it uses is
distorted by exactly this artifact, and the market intercept's large norm comes from
factor 2.

**Fix.** Drop DV01 as an instrument. It is approximately duration times price over
100, so it duplicates duration and premium/discount. Refit and reread Gamma, factor
variance shares, and factor correlations. `REVIEW.md` Step 2 already warned to prune
collinear characteristics with the W_beta test; run that test.

### Issue 3. The yield-space residual does not forecast out of sample

| Statistic (cells 26-29) | Value |
|---|---|
| Raw lag-1 residual rank IC | +0.084, t = +3.5 |
| Raw lag-1 pooled Pearson correlation | -0.018 |
| Walk-forward AR(1) slope | -0.044 (Aug), -0.030 (Sep) |
| AR(1) out-of-sample daily rank IC | -0.097, t = -2.8 |
| AR(1) out-of-sample R² vs zero | -0.0015 |
| AR(1) decile response | inverted (D1 +0.11 bp, D10 -0.28 bp) |

Rank and Pearson statistics disagree in sign. The target is quantised to whole basis
points (sample values 2, -4, 1, 0, 5, 3, -1; median exactly 0), but rounding itself is
too small to explain this:

| Quantity | Value |
|---|---|
| Rounding noise sd on a daily change (two levels rounded to 1 bp) | 0.41 bp |
| Share of residual variance (residual sd 3.42 bp) | 1.4% |

*Correction:* the first version of this review said rounding noise was a large share
of the signal. It is not. The more likely causes are days on which the mark did not
change at all, which create rank ties and are evaluator staleness rather than noise,
and tails that drive the OLS slope while the rank IC reflects the bulk.

**Fix.** On the existing residuals, measure the share of exactly-zero changes, and
compare the lag-1 Pearson correlation and AR slope before and after winsorising at the
1st and 99th percentiles. If winsorising reconciles the signs, fit the AR layer on
winsorised or rank-transformed residuals and carry a "mark unchanged" flag as a state
variable. Finer source marks or multi-day targets are not the binding fix. Treat the
yield-space residual as uninformative for forecasting until this is resolved.

### Issue 4. The covered universe contains almost no stale marks

The yield panel keeps only bonds with a closing mark within 7 days of the previous
one. After filtering, the model covers 25,195 CUSIPs out of about 236,000 in the
original universe, and the median gap between marks is 1.00 day in every activity
bucket (cell 30). "L1" in this notebook means low yield-change volatility, not a
stale mark.

This is the most likely reason the stale-mark effect disappears in yield space. It is
a coverage result, not a refutation of the original finding, and the notebook should
state it that way.

**Fix.** Rerun the MSRB tests on the full universe. Either keep price space and
convert with the DV01 bridge, or build a yield target for bonds without daily
closes. Define the staleness bucket from mark age or mark update rate, not from
return volatility.

### Issue 5. Several Step 5b validations are circular or mechanical

| Section | Claim | Why it does not validate |
|---|---|---|
| 11b peer-implied yield | Peer yield fits own yield at 5.0 bp vs 51.4 bp for random peers | Closing-yield rank is one of the instruments defining beta, so peers are selected on yield level. |
| 11 peer coherence | Peers are 30x closer in duration, 8x in yield | Duration and yield are inputs to beta; closeness is by construction. |
| 9 book exposure | Duration factor is monotone across duration quintiles | Beta is linear in the duration rank; monotonicity is guaranteed. |
| 10 in-panel roll-forward | Mark error falls 33% to 62% with horizon | The factor moves are estimated from the same bonds' updated marks, so this restates the same-day fit (about 51% variance explained) compounded over horizons. It does not test stale marks. |

**Fix.** For peers, remove yield level from the instruments used to find neighbours
(or use a separate peer embedding), then validate the peer-implied level out of
sample against the next MSRB print. For the roll-forward, restrict the test to bonds
whose mark did not move over the window and compare to a later print or mark
update. Drop the Section 9 monotonicity check or replace it with a comparison
against the desk's own duration report.

### Issue 6. The real roll-forward test is weak and contradicts the expectation

| MSRB Test B (cell 48) | Value |
|---|---|
| Roll-forward gap coefficient, all trades | +0.039, t = +2.3 |
| Coefficient if the gap were fully priced in | 1.0 |
| Incremental R² | 0.00002 |
| Coefficient by mark age: 1 d / 2-3 d / 4-7 d / 8-20 d | +0.47 / +0.21 / +0.24 / -0.01 |

Only about 4% of the factor-implied move shows up in the trade-vs-algo error, and the
stalest bucket, which is also the largest, shows nothing. `REVIEW.md` Step 5b expected
the effect to strengthen with mark age; it weakens. The notebook's read ("strongest at
short/moderate age") presents the contradiction as a finding.

A plausible reason: the algo signal yield may already incorporate curve moves, in
which case the trade-vs-algo error is the wrong target for a roll-forward test. The
right target is trade yield minus the stale evaluated mark.

**Fix.** Retest with target = MSRB yield minus last evaluated mark, restricted to
bonds whose mark is stale, after Issue 2 is fixed.

### Issue 7. Engineering defects

1. **Stale output and a corrupted artifact (cell 56).** The code sets
   `RUN_PIPELINE_REBUILD = False`, but the recorded output shows a rebuild ran, with
   print lines the current code does not contain. That run rewrote
   `research_panel_step2.parquet` on 2026-09-28 (736 MB to 608 MB) from a closing-marks
   audit showing one date, parsed as 1970-01-01, and 0% coverage on every field.
   Current results are unaffected because the Step 3 panel dates from 2026-09-24.
   Because `AUTO_BUILD_STEP3_YIELD_PANEL = True`, deleting the Step 3 panel would
   rebuild it from the broken Step 2 file.
2. **Registry key mismatch (cell 35).** `ar_oos_rank_ic` and `ar_oos_r2` show blank
   because the registry looks up `rank_ic_mean` and `oos_R2` while the summary uses
   `daily_rank_ic_mean` and `oos_r2_vs_zero`. The most negative metric in the
   notebook is silently missing from the registry.
3. **Look-ahead in activity buckets (cell 30).** Buckets use full-sample target
   volatility, including test weeks. Use pre-OOS data only, as Step 1 did.
4. **Price-space chain not reproducible here.** Step 1 is gated off, so the notebook
   cannot reproduce the price-space results its prose relies on.
5. **Duplicate primitives.** Step 3 redefines `MomentIPCA`, `precompute_moments`, and
   the fold builder when absent. Two copies can drift; move them into the package the
   notebook already imports (`muni_factor_pipeline`).

### Issue 8. Characteristic set is narrower than Step 2 specified

The instruments are duration, DV01, premium/discount, and yield level. Missing
relative to `REVIEW.md` Step 2: years to worst, the extension term, rating score, NR
flag, state dummies, and liquidity characteristics. With DV01 removed (Issue 2) the
model would have three characteristics plus an intercept for three factors, which is
too few to identify distinct factor structure.

---

## 4. What holds up

- **Walk-forward IPCA.** Correct estimator, correct identification, correct
  point-in-time discipline, fast enough for daily refits.
- **Residual-maker and factor weights.** The residual is exactly the return minus
  beta times the mimicking-portfolio factor, verified to machine precision.
- **Artifact framework.** Spec hash, versioned score tables, and a registry make
  model comparisons auditable once Issue 7.2 is fixed.
- **Charge model.** Honest result:

  | Model | Adjusted MAE | Improvement vs zero | Folds beating zero |
  |---|---|---|---|
  | No adjustment | 10.974 bp | 0 | 0 of 9 |
  | Side x size median | 10.868 bp | +0.106 bp | 9 of 9 |
  | Stale-gated residual median | 10.869 bp | +0.105 bp | 9 of 9 |

  The residual adds nothing over the side and size context. The notebook says so.
- **Book-exposure tooling.** Sound once the factors are fixed; the positions hook is
  ready for real desk data.

---

## 5. Recommended actions, in order

| # | Action | Resolves | Effort |
|---|---|---|---|
| 1 | Restore `research_panel_step2.parquet` from a good build; delete or fix cell 56 and turn off `AUTO_BUILD_STEP3_YIELD_PANEL` until the package is fixed | 7.1 | small |
| 2 | Rewrite the executive summary, Step 5 economic basis, and results table from yield-space outputs; label the -22 bp figure as a prior price-space run | 1 | small |
| 3 | Fix the registry keys; compute activity buckets from pre-OOS data | 7.2, 7.3 | small |
| 4 | Drop DV01; add years to worst, extension, rating score, NR flag, state, liquidity; run the W_beta pruning test; refit | 2, 8 | medium |
| 5 | Diagnose the AR sign conflict (zero-change share, winsorised vs raw slope); fit a robust or rank-based AR layer | 3 | small |
| 6 | Rerun MSRB validation on the full universe with a staleness bucket defined by mark age | 4 | medium |
| 7 | Rebuild peer and roll-forward validations without circularity; retest roll-forward against trade minus stale mark | 5, 6 | medium |
| 8 | Re-enable Step 1 or remove price-space claims; move shared primitives into the package | 7.4, 7.5 | small |

Items 1 to 3 should happen before the notebook is shared further. Items 4 to 7
determine whether the yield-space chain supports any economic conclusion.

---

## 6. Bottom line for readers of the earlier documents

- The original stale-mark finding from `REVIEW.md` and `RESEARCH_NOTE.md` is neither
  confirmed nor refuted by this revision. The yield-space test ran on a universe with
  almost no stale marks.
- The engineering recommendations of `REVIEW.md` Steps 0, 1, and 4 are implemented
  and working.
- The Step 5b tools exist, but their headline validations need to be redone before
  the roll-forward or peer-implied yield is described as a proven fair-value input.
