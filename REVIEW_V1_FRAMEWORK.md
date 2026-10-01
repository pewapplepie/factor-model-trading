# Review of `muni_characteristic_factor_v1_framework_review.md`

Date: 2026-10-01
Reviewed note: `muni_characteristic_factor_v1_framework_review.md` (spec `step3_yield_ipca_k3_v1`)
Companion documents: `REVIEW.md` (original notebook), `REVIEW_REVISED.md` (revised notebook), `RESEARCH_NOTE.md`

## Verdict

The framework and the honest reframing are good. Three things keep the note from being a v1 record you can defend:

1. a same-day residual join that leaks the close mark into the trade tests;
2. uncorrected t-statistics on pooled trades;
3. a "stale-mark story falsified" claim that the note's own next-experiment list admits is untested.

Fix those, rerun, and the +4.41 bp/rank result either survives clustered and side-controlled, or it shrinks to its true size. Either outcome is fine for a fair-value diagnostic, which is the product the note correctly describes.

## What the note gets right

- **Scope.** Yield-space IPCA as a leak-aware fair-value and risk decomposition, not a stat-arb system. That is the correct product.
- **Instrument fix.** Dropping DV01 turns factor 1 into a readable duration/term factor and cuts gross leverage from 38.6 to 10.3.
- **Target split.** Validating against `trade - evaluated mark`, and using `trade - algo` only for the quote-adjustment layer, is the right separation.
- **Engineering checks.** Residual-maker identity reproduced exactly, $B_t'\varepsilon_t = 0$ verified, every residual carrying a `gamma_version`, integrity gate on the 13-instrument spec.
- **Honest economics.** A 0.11 bp charge gain called small, and an open-risks list that covers the real ones (six months, latest ratings, covered subset, traded-name validation, materiality).

## Claim-by-claim status

| Claim in the note | Status | What settles it |
|---|---|---|
| Residual predicts `trade - mark`, +4.41 bp/rank, t = 8.9 | Not yet credible | Exclude same-day residuals, cluster by date, control for side and size |
| Stale-mark story falsified | Untested | Post-trade mark revision test (the note's next experiment 2) |
| Charge model beats no adjustment on 9/9 folds | Credible, but residual adds ~0 | Report the residual's increment over side x size alone |
| Winsorised AR rank IC +0.094, t = 2.67 | Fragile | IC as a function of threshold; thresholds from the train fold only |
| Roll-forward RMSE -34% (h=1) / -63% (h=10) | In-sample for factors | Leave-one-out factor estimate; duration x benchmark-curve baseline |
| De-circularised peers 20.0 vs 51.6 bp | Weak baseline | Compare to rule-based desk comparables, not random peers |

## What to fix before calling it v1

### 1. The same-day join is look-ahead

In the Step 4 score table, `signal_available_ts` equals the residual `date`, which is the close that produced the residual. The Step 5 join (`merge_asof`, `direction='backward'`, `allow_exact_matches=True`) then keeps `residual_age_days` in `[0, 20]`.

A trade printed on day $t$ is therefore matched to a residual computed from day $t$'s close. That close is published after the trade and, for an evaluator that uses prints, is partly formed from the trade itself. The residual and the `trade - mark` label become mechanically correlated.

Fix: require `residual_age_days >= 1`, or set `signal_available_ts` to the next business day. This applies to the transaction validation, the recency buckets, and the charge dataset. If the +4.41 bp and the 1 to 7 day concentration shrink after this, the leak was carrying them.

### 2. Plain OLS t-statistics

The OLS helper in the transaction validation uses homoskedastic standard errors over pooled trades. Trades on the same date share the factor realisation and the same cross-sectional ranks, so the effective sample is closer to the number of dates (about 53 signal dates in the charge dataset) than the number of trades (129,635).

Fix: run daily cross-sectional regressions and report a Fama-MacBeth t over dates, or cluster standard errors by date. Expect the t to fall by several times. Report the distribution of daily ICs, not only the pooled beta.

### 3. Side and size controls

`trade - mark` has a sign by side: customer buys print through the mark, customer sells print back of it. If the residual rank correlates with the algo signal side at all, a side effect shows up as a residual effect. The note's own attribution says side x size explains the charge gain.

Fix: report the residual beta within side x size cells and show the sign holds on both sides.

### 4. "Falsified" is too strong

Mark age near zero means ICE publishes a new number daily. It says nothing about whether that number carries lagged information, which is what smoothing means (Getmansky, Lo and Makarov 2004 is about information content, not publish frequency).

Two of the results offered as evidence against staleness are consistent with partial adjustment:

- active names get prints, marks adjust toward prints over several days, and the residual persists while they do;
- the `L5` lag-1 IC is also where residual dispersion is largest relative to the ~0.41 bp quantisation noise, so it is the bucket where any effect is easiest to see.

Reword to: "the mark-age axis is uninformative; trade recency is the right staleness proxy; the evaluator-revision test decides." Then run that test first, because the production use of the residual depends on it. If the residual predicts `trade - mark` but not the subsequent mark revision, the signal is quote-benchmark bias or liquidity concession, and it belongs in the quote layer, not the mark layer.

### 5. The residual's increment in the charge model is ~0

Side x size alone gives about +0.106 bp; the stale-gated residual gives about +0.111 bp. The residual's incremental contribution is on the order of 0.005 bp. Say so explicitly.

This is consistent with the weak `trade - algo` result (t = 1.2): the algo quote already conditions on recent prints, so the residual is redundant to the quote but not to the mark. The right monetisation test is on the mark side:

$$
\text{MAE}\big(|y^{trade} - (y^{mark} + \hat{\varepsilon})|\big) \quad \text{vs} \quad \text{MAE}\big(|y^{trade} - y^{mark}|\big)
$$

That is the number to put in front of the desk. It points the residual at mark-based processes (inventory valuation, risk marks, RFQ review on names the algo does not quote), not at the charge.

### 6. The sign flip under winsorisation is a finding

Raw AR IC -0.094 and winsorised AR IC +0.094 is a two-regime story: the body of the residual distribution persists and the tails revert. Treat it as a result, not a nuisance.

- Show the IC across winsorisation thresholds (1%, 2.5%, 5%).
- Confirm the thresholds are computed on the training fold only.
- Consider a sign-and-size conditional AR, or a quantile model, instead of one winsorised slope.

### 7. Use baselines the desk already has

- **Roll-forward.** Hold the bond out when estimating $f_t$ (a rank-one update of $(B_t'B_t)^{-1}$ makes this cheap), and compare against duration x benchmark-curve move, not only "leave the mark stale".
- **Peers.** Beat same-state, same-rating-bucket, same-maturity-bucket comparables, not random bonds.

### 8. Put coverage in the headline

The yield panel is 57,573 of 258,082 CUSIPs; the validation is on roughly 16,000 traded names. Every result is conditional on the active, OneTick-covered subset. That belongs in the Decision Summary, not only in Open Risks.

### Smaller points

- Ratings are latest composite, so treat the rating instrument as suspect until a point-in-time history exists.
- K = 3 and the $W_\beta$ instrument pruning are still undocumented (the note lists this).
- State the state-dummy set (`CA`, `NY`, `TX`, `FL`) as a design choice with the long tail mapped to "other".

## Productionising it cleanly

The note already names the right package layout (`contracts.py`, `transforms.py`, `orchestrator.py`). Make the notebook a thin client of that package and move every piece of logic into tested modules.

### Layers

| Layer | Responsibility | Key artifact |
|---|---|---|
| Data | Idempotent, date-partitioned ingestion per source with manifests | `data/<source>/date=YYYY-MM-DD/` + `manifest.json` |
| Feature | Step 2 / Step 3 panel builders as pure functions driven by a versioned spec | `specs/step3_yield_ipca_k3_v1.yaml` |
| Model | `fit(panel, spec) -> GammaArtifact`; `score(panel_t, gamma) -> betas, f_t, W_F, residuals` | Gamma registry table |
| Signal | Residual score, AR forecast, exposures, roll-forward, peers | Daily tables keyed `(cusip, date, spec_id, gamma_version)` |
| Validation | Standing daily/weekly report with clustered statistics | Metrics table + alerts |
| Serving | Read-only fair-value call per CUSIP and as-of date | Shadow-mode log |

### Rules per layer

- **Data.** Replace pickles with parquet partitioned by date. Each partition is written with a manifest recording the query window, row count and content hash. The existing join-coverage guard becomes a family of gates: key uniqueness, date parsing, yield units, universe drift, target clip share.
- **Feature.** The spec lists instruments, lags, transforms and the per-date rank-normalisation rule. The spec id is what every downstream row carries. Rank normalisation uses only the date-$t$ cross-section.
- **Model.** The Gamma artifact records train window, instruments, K, fit statistics and a hash, and is written to a registry. Scoring uses the latest Gamma whose train end is strictly before the scoring date. Weekly refit, daily scoring.
- **Signal.** AR thresholds come from the training fold. The availability timestamp on every row is the next business day's open.
- **Validation.** Recompute the claim table above daily and weekly with clustered statistics. Alert on coverage drop, factor-1 leverage, Gamma drift after sign alignment, residual standard deviation, daily IC, and the clustered `trade - mark` beta on a rolling four-week window.
- **Serving.** Return mark, systematic roll-forward, residual, peer-implied level and exposure contribution as separate components. Run in shadow mode next to the quoting system for four to eight weeks, logging what it would have changed with side and size context, before anyone acts on it.

### Tests

Synthetic-panel unit tests for:

- lag correctness and rank normalisation;
- Gamma orthonormality ($\Gamma'\Gamma = I$) and the residual-maker identity;
- a point-in-time assertion that no row is scored with a Gamma trained through its own date, or joined to a residual dated the same day as the trade.

### Governance

- A model card per spec version: data scope, known limitations, validation report.
- Promotion research -> candidate -> production reuses the same validation report.
- Kill criteria written down in advance: rolling clustered t below 2, sign flip, coverage loss above 20%, or factor-1 leverage above a fixed ceiling pulls the residual from the serving layer.

## Bottom line

Keep the framework and the framing. Before the note becomes the v1 record: enforce a one-day minimum residual age, cluster the t-statistics by date, control for side and size, and replace "falsified" with "untested" until the evaluator-revision test runs. Then state the residual's increment in the charge model honestly (about zero) and move its monetisation test to the mark side, where the evidence says it belongs.
