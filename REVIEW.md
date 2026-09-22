# Review: Muni Characteristic Factor Notebook

Scope: `muni_characteristic_factor.ipynb`, all 142 cells including cached outputs.
Two lenses: (1) factor-model research, with reference to Kelly, Pruitt and Su (IPCA)
and the attention-factors literature; (2) ML engineering of the training, fitting,
and evaluation pipeline.

Part A is the assessment. Part B is a step-by-step implementation guide. Each step
carries a **Reference** and a **Fidelity** tag:

- **Paper-faithful**: the step reproduces what the cited paper does.
- **Paper-adapted**: the paper's method, changed for the muni / RFQ setting.
- **Practice**: standard quantitative or ML practice, not from the factor papers.

Citations are listed in full at the end. They were written from knowledge; paper
hosts were not reachable from the review environment, so verify DOIs before
quoting them externally.

---

# Part A. Assessment

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
continuous, time-varying characteristics ranked per date (see Step 2).

**Wrong return space for the use case.** Clean-price returns on bonds with very
different durations mean factor 1 is duration times the daily curve move, which
is why tenor dummies dominate Gamma. The residual is then in price-return units
while every business target from Section 20 on is in yield bps. That unit
mismatch, with no DV01 bridge, is a plausible reason the charge model fails
(see Step 3).

**Leakage.** Gamma is fit on the full window and its residuals feed the "frozen
OOS" AR pipeline (9.9 to 9.12), the executable-opportunity tests (15), the
calibration (16), the attention challenger (18), and the final MSRB regression.
Ratings are taken as the latest value and applied to all history. Only the AR
layer is point-in-time (see Step 1).

**K selection sits inside noise.** Total R² differences across K are 0.002 to
0.008 while the fold standard deviation is 0.11. Predictive R² is negative for
every K, and the three factor return series correlate at 0.80 to 0.88, so this
is effectively one duration factor plus rotation (see Step 4).

**Smoothing explains most of what looks like alpha.** Sharpe ratios of 3.2 on
factor portfolios and 7 to 9 on residual spreads, a lag-1 autocorrelation of
0.17 that vanishes by lag 3, and persistence rising from 0.08 in active bonds
to 0.32 in stale ones are all the signature of an evaluator's partial
adjustment toward trade prints. The final MSRB regression settles it: the
effect lives entirely in the stale quintile. Never quote a Sharpe computed on
evaluated marks (see Step 5).

**Section 16 calibration is broken at the top.** The pooled calibrated score
has negative rank IC yet a large decile spread, which means it ranks contexts,
not bonds. The reliability table predicts +11 bps for decile 10 and realises
-46 bps, and deciles 7 to 10 are not monotone (see Step 7).

**The attention challenger cannot win by construction.** Softmax over dummy
scores yields near-uniform weights with effective N around 145,000 of 225,000
bonds, so the three "attention factors" are a reweighted market portfolio,
evaluated on 29 dates. This is not the attention-factors model, which needs
learned embeddings, dynamic features, sparse peers, and a trained objective
(see Step 8).

**The systematic path stops at diagnostics.** Gamma and the factor moves are
fitted, plotted, and turned into mimicking portfolios, then never used
downstream: every production-facing output (the RV score, the calibration, the
charge model) is built from the residual alone. That is correct for the
rich/cheap signal, since the residual is by definition the distance from fair
value, but it leaves three cheap uses of the same fit on the table: rolling
stale marks forward by beta times the factor moves, aggregating beta over
positions for book-level factor exposure and quote skew, and using beta space
as the comparables metric for RFQ pricing (see Step 5b). The attention paper
does not add a factor bet either; it combines the two paths by mapping residual
weights through the factor hedge, which Section 8.1 already derives.

## ML engineering review

**Validation is inconsistent.** Single 70/30 time splits in Sections 9.6, 9.8,
10, and 22 coexist with expanding monthly folds in 9.10. K selection uses three
folds of 15 days. Gates use hard-coded thresholds, some redundant, and were set
after seeing results (see Step 9).

**Reproducibility is the biggest gap.** Cached outputs come from at least four
run dates. The Q1 cell (10.7) fails with a NameError because a helper is defined
in a later section. Twelve code cells have no outputs, including the core
expanding-window fit in 9.10 and the revised Section 10 robustness cells, so the
parquet those sections read is from an unknown run. The weekly walk-forward is
switched off, yet Section 18 depends on its cached Gamma (see Step 0).

**The fit is far slower than it needs to be.** The Z tensor is about 5 GB in
float32, the IPCA refit takes 11 minutes, and the row-wise call-bucket apply
takes 4 minutes. With one-hot Z, every Z'Z is a count crosstab and every Z'r is
a per-bucket sum, so the whole IPCA iteration is a groupby that runs in seconds
(see Step 1).

**The charge model is set up to lose.** Ridge minimises squared error on a
target with MAE 10 and RMSE 31, then is scored on MAE, so outliers drag every
prediction. The context model also includes two identical side columns. Its
rank correlation of 0.09 shows there is signal but the level is miscalibrated
(see Step 6).

**Minor bugs.**

- Signal-to-trade lag is measured in calendar days, so the 2-day and 6-to-20-day
  buckets are empty.
- The first observation of every bond is zero-filled and then counted as
  observed.
- The IPCA random fallback initialisation is unseeded.
- Test A has redundant economic-gate thresholds (10 bps and 20 bps).

---

# Part B. Implementation guide

The steps are ordered by dependency. Steps 0 and 1 unblock everything else.

## Finding-to-step map

| Finding | Step | Reference | Fidelity |
|---|---|---|---|
| Not reproducible, broken cells, stale caches | 0 | Practice | Practice |
| Full-sample Gamma leaks into every OOS test; slow fit | 1 | KPS 2019 §2, §4 (recursive OOS) | Paper-faithful |
| Static one-hot dummies instead of characteristics | 2 | KPS 2019 §3 (rank transform); KPP 2023 (bond characteristics) | Paper-faithful |
| Price-space residual vs yield-space targets | 3 | KPP 2023 (excess returns, duration); bond math | Paper-adapted |
| K chosen inside fold noise | 4 | KPS 2019 §4 (K sweep), bootstrap tests; Diebold-Mariano 1995 | Paper-adapted |
| Smoothed marks masquerade as alpha | 5 | Getmansky-Lo-Makarov 2004; Amihud-Mendelson 1987 | Practice |
| Gamma and factor moves unused downstream | 5b | KPS 2019 §2 (beta = z Gamma); notebook §8.1 residual-maker | Paper-adapted |
| Charge model loses to zero | 6 | Koenker-Bassett 1978; Huber 1964; LightGBM | Practice |
| Calibration broken in top decile | 7 | Zadrozny-Elkan 2002; Naeini et al. 2015 | Practice |
| Attention v0 cannot win | 8 | Epstein-Yu-Pelger 2025; Guijarro-Ordonez-Pelger-Zanotti 2022 | Paper-faithful |
| Inconsistent folds, no intervals, hard gates | 9 | López de Prado 2018; Harvey-Liu-Zhu 2016; Newey-West 1987 | Practice |
| 116 days, one regime; small bugs | 10 | Harris-Piwowar 2006 (cost floors) | Practice |

---

## Step 0. Reproducibility harness

**Goal.** A single command re-executes the notebook from a clean kernel and fails
loudly if any cell errors or any cached artifact is stale.

**Reference.** Practice. **Fidelity.** Practice.

**Do this.**

1. Create a package next to the notebook and move all helpers out of cells:

   ```text
   muni_factor/
     __init__.py        # __version__ = "0.2.0"  (bump on any methodology change)
     config.py          # RunConfig dataclass: dates, K grid, tolerances, seeds, paths
     data.py            # Deephaven pulls, caches, bucket derivation (vectorised)
     ipca.py            # IPCA estimator (Step 1)
     features.py        # characteristic construction (Step 2), return space (Step 3)
     folds.py           # one fold generator used everywhere (Step 9)
     metrics.py         # R² family, IC, decile spread, bootstrap CIs, gates
     residual.py        # PIT residual chain, AR layer, smoothing model (Step 5)
     msrb.py            # trade panel, executable opportunities, calibration
     charge.py          # charge models (Step 6)
   ```

   The NameError in 10.7 disappears because `_ols1` lives in `metrics.py`.

2. Hash the config plus `__version__` plus the panel signature into every cache
   key. A code change then invalidates caches automatically; the manual
   `RETRAIN_*` flags go away.

3. Execute headless in CI or a cron job and keep the executed copy:

   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       muni_characteristic_factor.ipynb --output executed/$(git rev-parse --short HEAD).ipynb
   ```

   Fail the run on any cell error. Only an executed copy with a git hash counts
   as research evidence.

4. Seed everything once in `config.py` (`numpy`, `random`, `sklearn`,
   `lightgbm`, and the IPCA fallback initialiser).

5. Replace the row-wise `df.apply(call_bucket, axis=1)` with `pd.cut` on
   `call_days` plus a boolean mask. Store the raw panel as parquet with
   categorical dtypes instead of pickle.

**Validate.** Two consecutive clean runs on the same data produce identical
metrics tables to displayed precision.

---

## Step 1. Fast IPCA on bucketed data, then walk-forward Gamma

**Goal.** Make one IPCA fit take seconds instead of eleven minutes, then refit
Gamma every week using only past data so residuals are point-in-time.

**Reference.** Kelly, Pruitt and Su (2019), Section 2 for the ALS estimator and
identification, Section 4 for recursive out-of-sample estimation.
**Fidelity.** Paper-faithful.

**Why it is fast.** With one-hot (or bucketed) characteristics plus an intercept,
each date's cross-products depend only on per-cell counts and sums. Define a
"cell" as the full bucket combination (rating x coupon x call x state x tenor).
Let M be the C x L incidence matrix mapping cells to dummy columns, n_{c,t} the
number of bonds in cell c at date t, s_{c,t} the sum of their returns, and
q_{c,t} the sum of squared returns. Then

```text
Z_t' Z_t = M' diag(n_t) M        (L x L)
Z_t' r_t = M' s_t                (L)
r_t' r_t = sum_c q_{c,t}
```

The ALS steps are unchanged from the notebook but never touch the N x L x T
tensor:

```python
# per date: f_t = (G' A_t G)^-1 G' b_t          A_t = Z_t'Z_t, b_t = Z_t'r_t
# Gamma:    vec(G) = (sum_t f_t f_t' kron A_t)^-1 sum_t (f_t kron b_t)
# loss:     sum_t [ rr_t - 2 f_t' G' b_t + f_t' G' A_t G f_t ]
```

Precompute `A[t]`, `b[t]`, `rr[t]` once with a groupby over (date, cell). Cost
per ALS iteration is O(T (C L + L² K²)), independent of N. The 5 GB tensor is
gone.

**Identification (add what is missing).** After each Gamma step the notebook only
QR-orthonormalises. KPS also rotate so the factor covariance is diagonal with
descending entries and fix signs so each factor mean is non-negative:

```python
Q, _ = np.linalg.qr(Gamma)                     # Gamma'Gamma = I
F = solve_factors(Q)                            # T x K
W, _, _ = np.linalg.svd(F.T @ F / T)            # rotate to diagonal covariance
Gamma, F = Q @ W, F @ W
sgn = np.sign(F.mean(0)); sgn[sgn == 0] = 1     # non-negative factor means
Gamma, F = Gamma * sgn, F * sgn
```

This makes Gamma comparable across refits and makes the Gamma-instability
diagnostic meaningful.

**Walk-forward chain.** Refit weekly (daily is affordable after the speed-up):

```text
for each refit date v (weekly):
    Gamma_v  = IPCA fit on dates <= v            (warm start from Gamma_{v-1})
    for each t in (v, v + 1 week]:
        beta_t   = Z_{t-1} Gamma_v               (predetermined loadings)
        f_t      = (beta_t' beta_t)^-1 beta_t' r_t   (cross-section at t only)
        eps_t    = r_t - beta_t f_t
        store eps_t with gamma_version = v
```

The factor realisation f_t uses date-t returns only, which is the paper's
out-of-sample construction. Every downstream table (9.9 onward) must read this
residual table, keyed by `gamma_version`, and never the full-sample residual.

**Validate.** (1) On the full sample the fast estimator reproduces the current
Gamma to 1e-6. (2) A leak test: shift every return forward one day and confirm
the walk-forward residual IC collapses to zero.

---

## Step 2. Continuous, rank-normalised characteristics

**Goal.** Give IPCA instruments that vary over time and within buckets, as the
paper intends.

**Reference.** KPS (2019), Section 3: each characteristic is rank-transformed
cross-sectionally each period and mapped to [-0.5, 0.5], missing values set to
0. Kelly, Palhares and Pruitt (2023) apply the same transform to corporate bond
characteristics (duration, spread, rating, coupon, age, size, volatility,
momentum). **Fidelity.** Paper-faithful.

**Do this.**

```python
def rank_norm(x: pd.Series) -> pd.Series:
    r = x.rank(method="average")
    return (r / r.count() - 0.5).fillna(0.0)

z = (panel.groupby("date")[char_cols].transform(rank_norm))
z["const"] = 1.0
```

Characteristic set for munis, all point-in-time:

| Family | Characteristic | Source |
|---|---|---|
| Curve | years to maturity; years to call (or yield-to-worst duration) | Product |
| Structure | coupon; premium/discount (price - 100) | Product, ICEEval |
| Credit | numeric rating score (AAA=1 ... CCC=17); NR flag | algo track, PIT |
| Relative value | evaluated yield minus MMD at matched tenor | ICEEval, MMD |
| Liquidity | log days since last MSRB print; log 20-day trade count; mark update rate | MSRB, ICEEval |
| Risk | 20-day evaluated-return volatility | ICEEval |
| State | CA / NY / TX / IL dummies | Product |

Keep a few dummies; the paper's design allows mixed inputs. Drop the 46 bucket
dummies.

**Prune with the paper's test.** KPS test each characteristic with a wild
bootstrap on the residuals (their W_beta statistic): set Gamma's row for
characteristic l to zero, refit, bootstrap the change in fit. Implement it once
in `ipca.py` and drop characteristics that fail.

**Validate.** Gamma instability across weekly refits should fall, and the
loading-space nearest-neighbour distances (8.4.3) should stop being exactly zero.

---

## Step 3. Put residuals in yield space

**Goal.** Make the residual, the calibration target, and the charge target share
units, and stop the duration factor dominating Gamma.

**Reference.** KPP (2023) model bond excess returns and include duration among
the characteristics so the term factor is spanned. The bond-math identity
r ≈ -D Δy + carry is textbook. **Fidelity.** Paper-adapted.

**Do this.** Choose one target and use it everywhere:

1. **Curve-relative yield change (recommended).**
   `y_target = Δy_i,t - Δy_MMD(tenor_i, t)` in bps. The IPCA residual is then a
   bps quantity directly comparable to `algo_minus_msrb_yield_bp`.
2. **Curve-hedged excess return.** `r_i,t + D_i,t Δy_MMD(tenor_i, t)`, with
   D from yield-to-worst duration. Use if price space must be kept.

Either way, add duration (or years to maturity and call) as a characteristic per
KPP so the residual is orthogonal to the term factor by construction.

**DV01 bridge.** Where a price residual is still needed (Test A/B in Section 13),
convert with `Δprice ≈ -DV01 Δy`, DV01 from modified duration at the evaluated
yield. Never compare price residuals to yield errors without this.

**Validate.** Gamma's largest loadings should stop being the tenor row; the
first-factor share of variance should fall; Test B in bps should remain
positive after the change.

---

## Step 4. Choose K on paired fold differences

**Goal.** Replace the 0.005 tolerance rule, which cannot distinguish K on three
folds, with a test that respects fold noise and the downstream objective.

**Reference.** KPS (2019) sweep K = 1..6 and report fit; their significance
tests use a residual bootstrap. Diebold and Mariano (1995) for paired forecast
comparison; Politis and Romano (1994) for the stationary bootstrap.
**Fidelity.** Paper-adapted.

**Do this.**

1. Use weekly out-of-sample folds (Step 9) so there are 15 or more folds.
2. For each fold f compute per-date squared errors under K and K-1, then the
   per-date difference d_t. Run a Diebold-Mariano test on d_t with Newey-West
   standard errors. Accept K only if d is significantly negative.
3. Report a stationary-bootstrap confidence interval for each K's mean total R²
   over dates (block length about 5 days).
4. Add a decision-theoretic column: Test B rank IC and charge MAE of the
   residual produced under each K. Pick the smallest K that is not beaten on the
   downstream metric.
5. After Step 1's rotation, report the factor correlation matrix; if two factors
   correlate above 0.8 across refits, K is too large.

**Validate.** The chosen K should be stable across at least two non-overlapping
half-samples.

---

## Step 5. Model the evaluator, not the alpha

**Goal.** Turn "residual persistence is stale-mark echo" from a diagnosis into
the product: an estimate of where the mark should be.

**Reference.** Getmansky, Lo and Makarov (2004) MA(q) smoothing and unsmoothing;
Amihud and Mendelson (1987) partial-adjustment price model; Dimson (1979) for
stale-price bias in betas. **Fidelity.** Practice.

**Do this.**

1. **Unsmooth and measure.** Per bond (or per liquidity bucket), fit the
   observed evaluated return as `r_obs_t = Σ_{j=0..q} θ_j r_true_{t-j}` with
   Σθ = 1, q = 2, by maximum likelihood. Report the smoothing index ξ = Σθ_j²
   by liquidity quintile. Expect ξ well below 1 in L1. Any volatility or Sharpe
   on marks must use the unsmoothed series.
2. **Partial-adjustment state model.** Latent fair value V_t; observed mark
   `P_t = P_{t-1} + g (V_t - P_{t-1}) + η_t`; MSRB prints `Q_τ = V_τ + ζ_τ`
   when they occur. Estimate g per liquidity bucket and V_t by Kalman filter.
   The stale-mark correction score is `E[V_t | F_t] - P_t` in bps of yield after
   Step 3.
3. This replaces the AR(1) as the temporal layer for the mark-correction use
   case. Keep AR(1) as the benchmark it must beat on Test B.

**Validate.** Test B D10-D1 for the Kalman score versus the AR(1) score, on
identical trades, with bootstrap intervals. The filter should win most in L1 and
be neutral elsewhere.

---

## Step 5b. Put the systematic path to work

**Goal.** Use Gamma and the factor moves for the three things the residual
cannot do: roll stale marks forward, measure book-level factor exposure, and
define comparables.

**Reference.** KPS (2019), Section 2: loadings are beta_{i,t} = z_{i,t} Gamma,
so exposures aggregate linearly over positions. The notebook's own Section 8.1
derives the residual-maker W^eps = I - B W^F, which is how the attention paper
maps residual weights to asset weights. **Fidelity.** Paper-adapted.

**Do this.**

1. **Systematic roll-forward of stale marks.** For bond i whose evaluated mark
   last moved at date s < t, the factor-implied mark is

   ```text
   P_hat_{i,t} = P_{i,s} * prod_{u=s+1..t} (1 + beta_{i,u-1}' f_u)
   ```

   in price space, or the additive version in yield space after Step 3. The
   quote adjustment is `P_hat - P_mark`. Test it as a Test B variant on the
   cached MSRB panel: regress next trade-from-mark on (a) nothing, (b) the
   roll-forward gap, (c) the residual correction, (d) both, with bootstrap
   intervals. Expect (b) to matter most where mark age is largest.

2. **Book-level factor exposure.** `E_t = sum_i w_{i,t} beta_{i,t}` is a
   K-vector of the desk's duration, long-end credit, and coupon-structure
   exposures in model coordinates. Publish it daily. Use it in the quoting
   optimiser as the inventory-skew term: skew bids and offers in proportion to
   how much a fill would move E_t away from its limit. Hedge ratios against any
   liquid instrument follow from regressing that instrument's return on f_t.

3. **Comparables in beta space.** For an RFQ on bond i, take the m nearest
   bonds by Euclidean distance in standardised beta, weight them by residual
   correlation over the last 60 days, and form a peer-implied yield from their
   MSRB prints in the last k days. Report it beside the residual score. This
   formalises Section 8.4.3, which already shows peers are descriptor-coherent.

4. **Uncertainty split.** `Var(r_i) = beta_i' Sigma_f beta_i + sigma_eps_i^2`.
   Feed both terms to the sigma the production spec asks for.

Note that with one-hot Z, beta is constant within a bucket, so the roll-forward
is the bucket's cumulative mean move. It sharpens materially after Step 2.

**Validate.** Roll-forward Test B: D10-D1 and incremental R² over the raw mark,
by mark-age bucket. Exposure: reconcile E_t against the desk's existing
duration and rating reports on the same day.

---

## Step 6. Rebuild the charge model as a robust, side-specific GBM

**Goal.** Beat "no adjustment" on held-out MAE and produce a mean and an
uncertainty per RFQ.

**Reference.** Quantile regression (Koenker and Bassett 1978); Huber loss
(Huber 1964); gradient boosting with monotone constraints (Ke et al. 2017).
**Fidelity.** Practice.

**Do this.**

1. Target: `yield_error_bp`, winsorised at the 0.5 and 99.5 percentiles within
   the training fold (not clipped at 500).
2. Features, all point-in-time: RV residual in bps (Step 3), Kalman correction
   (Step 5), AR forecast, days since last customer and interdealer print, 20-day
   trade count, mark age, residual volatility, log quantity, MinuteFromSignal,
   MMD level and 1-day change, rating score, years to maturity and call, side.
   Drop the duplicate side column.
3. Fit three quantile models (0.1, 0.5, 0.9) per side (P, S, D):

   ```python
   import lightgbm as lgb
   mono = [-1 if c == "rv_bps" else 0 for c in X.columns]   # higher RV -> lower yield error
   for q in (0.1, 0.5, 0.9):
       m = lgb.LGBMRegressor(objective="quantile", alpha=q, monotone_constraints=mono,
                             n_estimators=2000, learning_rate=0.02, num_leaves=63,
                             min_child_samples=500, subsample=0.8, colsample_bytree=0.8)
       m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100)])
   ```

   The 0.5 model is `rv_mu_bps`; `(q90 - q10) / 2.56` is `rv_sigma_bps`.
4. Evaluate on expanding monthly folds with a one-day embargo: MAE and pinball
   loss versus the zero benchmark, with a Diebold-Mariano test per fold and a
   bootstrap interval on the pooled improvement. Report by side, size, and
   freshness slices with intervals.

**Validate.** Accept only if median-model MAE beats zero on every fold and the
0.1 to 0.9 band covers 80% ± 3% of held-out outcomes.

---

## Step 7. Calibrate on percentiles with shrinkage and reliability checks

**Goal.** Fix the Section 16 top-decile failure and make the calibrated bps
trustworthy.

**Reference.** Isotonic calibration (Zadrozny and Elkan 2002); calibration
evaluation with reliability diagrams and expected calibration error (Naeini,
Cooper and Hauskrecht 2015; Niculescu-Mizil and Caruana 2005).
**Fidelity.** Practice.

**Do this.**

1. Calibrate on the within-context percentile of the signal, not its raw value.
   Context = side x size bucket. This removes the scale drift that let extreme
   raw values dominate the top bin.
2. Bin percentiles into 20 equal-count bins with at least 500 observations per
   bin per context; shrink each bin mean toward the context mean with weight
   n / (n + 2000).
3. Fit isotonic regression on the shrunk bin means; refit every week on an
   expanding window; apply only to the following week.
4. Report a reliability table and expected calibration error on held-out
   weeks, with bootstrap intervals. Require realised decile means to be
   monotone before the score is used anywhere downstream.

**Validate.** Decile 10 realised minus predicted within ±3 bps; ECE below 3 bps.

---

## Step 8. Attention factors v1, built as the paper builds them

**Goal.** A challenger that can actually beat IPCA: learned peer weights from
dynamic features, trained on an objective, evaluated on identical
point-in-time folds.

**Reference.** Epstein, Yu and Pelger (2025), "Attention Factors for Statistical
Arbitrage": attention heads over characteristic embeddings define the factor
portfolios, residuals feed a time-series model, and the whole stack is trained
end-to-end. Guijarro-Ordonez, Pelger and Zanotti (2022), "Deep Learning
Statistical Arbitrage", for the residual time-series model and the Sharpe-type
training objective. **Fidelity.** Paper-faithful in architecture; objective
adapted to the RFQ target.

**Do not build this before Steps 1 to 3 exist.** The v0 result is uninformative
because its inputs were one-hot dummies.

**Architecture.**

```text
x_{i,t-1}   : Step 2 characteristics + dynamic liquidity features   (d_in)
h_{i,t-1}   = MLP(x_{i,t-1})                                         (d = 32)
a_{k,i,t-1} = softmax_i( q_k' h_{i,t-1} / sqrt(d) )   over top-m peers only
F_{k,t}     = sum_i a_{k,i,t-1} r_{i,t}                              (K heads)
beta_{i,t-1}= W h_{i,t-1}                                            (K)
eps_{i,t}   = r_{i,t} - beta_{i,t-1}' F_t
```

Use sparse top-m attention (m = 200) with no self-attention so peers are real
comparables, not the market. Add a stability penalty on day-to-day changes in
a_{k,·,t}.

**How the paper combines factors and residuals.** The trading signal is built
from residuals only. Asset weights are the residual weights mapped through the
factor hedge, `w_asset = (I - B (B'B)^-1 B') w_resid`, so the book is
factor-neutral by construction; there is no directional factor bet. For the
muni desk the same structure is: residual score sets the quote skew, and
book-level beta exposure (Step 5b) sets how the resulting inventory is hedged
or leaned against.

**Objective.**
`L = λ_fit Σ eps² + λ_rv L_MSRB(g(eps_hat_{t+1|t}), next-trade markout) + λ_stab Σ ||a_t - a_{t-1}||²`.
Keep the temporal layer fixed as point-in-time AR(1) for the first experiment
so any gain is attributable to the peer representation.

**Training protocol.** Rolling windows: 6 months train, 1 month validation for
early stopping, 1 month test; slide monthly. Baseline is IPCA on the same
inputs and folds. Compare on Test B rank IC, within rating x tenor x call rank
IC, calibration error, and peer stability, each with bootstrap intervals.

**Validate.** Effective N per head should be in the hundreds, not 145,000. The
challenger must beat IPCA on at least three of four test months.

---

## Step 9. One evaluation harness

**Goal.** Every table in the notebook comes from the same fold generator and
the same metrics code, with intervals.

**Reference.** López de Prado (2018) for purged and embargoed time-series folds;
Harvey, Liu and Zhu (2016) for the multiple-testing hurdle; Newey and West
(1987) for overlapping horizons; Fama and MacBeth (1973); Bailey and López de
Prado (2014) for the deflated Sharpe ratio. **Fidelity.** Practice.

**Do this.**

1. `folds.py` exposes one generator: expanding origin, weekly test blocks, a
   one-day embargo between train end and test start, and a horizon-length purge
   when the label spans h days. Delete the four ad-hoc 70/30 splits.
2. `metrics.py` returns every statistic with a stationary-bootstrap 90%
   interval over dates. Daily-IC t-statistics use Newey-West with lag h-1 for
   h-day horizons.
3. Because dozens of slices are tested, use a t-statistic hurdle of 3.0 for any
   claim that a slice "works", and report the number of slices examined.
4. If any Sharpe is reported, report the deflated Sharpe ratio alongside it and
   compute it on trade prices only.
5. Gates become a table produced by code, with the threshold, the statistic,
   its interval, and pass/fail. Thresholds live in `config.py` and are fixed
   before a run.

---

## Step 10. Data hygiene and history

**Goal.** Enough history to span a rate regime, and no mechanical artifacts.

**Reference.** Harris and Piwowar (2006) and Green, Hollifield and Schürhoff
(2007) for muni transaction-cost magnitudes used in the cost gates.
**Fidelity.** Practice.

**Do this.**

1. Extend the ICE panel to at least two years. The 116-day window contains one
   regime; the K choice, the persistence estimates, and the cost gates all need
   more.
2. Lag in trading days, not calendar days, so the 2-day and 6-to-20-day buckets
   populate.
3. Do not zero-fill the first observation per bond; leave it missing and let
   the mask exclude it.
4. Use a point-in-time rating history rather than the latest composite rating.
5. Set cost floors from the literature by trade size: retail-size muni
   round-trips run in the 1 to 2 percent range, institutional far lower. The
   current flat 20 bps floor is generous for small prints and conservative for
   large ones.

---

# Key insights to carry forward

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
- The residual is the right basis for the rich/cheap score, but the same fit's
  systematic path is unused: stale-mark roll-forward, book factor exposure, and
  beta-space comparables are cheap additions that reuse Gamma.

---

# References

Factor models

- Kelly, B., Pruitt, S., Su, Y. (2019). Characteristics Are Covariances: A
  Unified Model of Risk and Return. *Journal of Financial Economics* 134(3),
  501-524. [KPS 2019]
- Kelly, B., Palhares, D., Pruitt, S. (2023). Modeling Corporate Bond Returns.
  *Journal of Finance* 78(4), 1967-2008. [KPP 2023]
- Epstein, E., Yu, R., Pelger, M. (2025). Attention Factors for Statistical
  Arbitrage. arXiv preprint.
- Guijarro-Ordonez, J., Pelger, M., Zanotti, G. (2022). Deep Learning
  Statistical Arbitrage. arXiv preprint.

Smoothing and stale prices

- Getmansky, M., Lo, A., Makarov, I. (2004). An Econometric Model of Serial
  Correlation and Illiquidity in Hedge Fund Returns. *Journal of Financial
  Economics* 74(3), 529-609.
- Amihud, Y., Mendelson, H. (1987). Trading Mechanisms and Stock Returns: An
  Empirical Investigation. *Journal of Finance* 42(3), 533-553.
- Dimson, E. (1979). Risk Measurement When Shares Are Subject to Infrequent
  Trading. *Journal of Financial Economics* 7(2), 197-226.

Muni market microstructure

- Harris, L., Piwowar, M. (2006). Secondary Trading Costs in the Municipal Bond
  Market. *Journal of Finance* 61(3), 1361-1397.
- Green, R., Hollifield, B., Schürhoff, N. (2007). Financial Intermediation and
  the Costs of Trading in an Opaque Market. *Review of Financial Studies*
  20(2), 275-314.

Inference and validation

- Fama, E., MacBeth, J. (1973). Risk, Return, and Equilibrium: Empirical Tests.
  *Journal of Political Economy* 81(3), 607-636.
- Newey, W., West, K. (1987). A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica* 55(3), 703-708.
- Diebold, F., Mariano, R. (1995). Comparing Predictive Accuracy. *Journal of
  Business and Economic Statistics* 13(3), 253-263.
- Politis, D., Romano, J. (1994). The Stationary Bootstrap. *Journal of the
  American Statistical Association* 89(428), 1303-1313.
- Harvey, C., Liu, Y., Zhu, H. (2016). ... and the Cross-Section of Expected
  Returns. *Review of Financial Studies* 29(1), 5-68.
- Bailey, D., López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of
  Portfolio Management* 40(5), 94-107.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  Chapters 7 (cross-validation), 11 (backtesting dangers), 14 (statistics).

Robust regression, boosting, calibration

- Koenker, R., Bassett, G. (1978). Regression Quantiles. *Econometrica* 46(1),
  33-50.
- Huber, P. (1964). Robust Estimation of a Location Parameter. *Annals of
  Mathematical Statistics* 35(1), 73-101.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree. *NeurIPS* 30.
- Zadrozny, B., Elkan, C. (2002). Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates. *KDD*.
- Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities with
  Supervised Learning. *ICML*.
- Naeini, M. P., Cooper, G., Hauskrecht, M. (2015). Obtaining Well Calibrated
  Probabilities Using Bayesian Binning. *AAAI*.
