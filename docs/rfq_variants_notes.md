# RFQ factor variants: the production optimizer as the judge (variants v1.3)

Companion to `muni_rfq_factor_variants.py` / `.ipynb`. The notebook quotes every factor-derived variant through the
production optimizer, replays it on the desk's requests and reads it against production on the same four facts (win
rate, cover, crossing, P&L under the sceptical exits) in four pairs. It reads one handoff written by the lab and needs
no model code and no pipeline, so it runs in its own kernel in a few minutes once the lab's caches exist.

Regenerate the `.ipynb` from the `.py` with the percent-format converter:

```python
import re, sys
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding='utf-8').read()
parts = re.split(r'^# %%(.*)$', text, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    marker, body = parts[i].strip(), parts[i + 1].strip('\n')
    if 'markdown' in marker:
        cells.append(new_markdown_cell('\n'.join(re.sub(r'^# ?', '', ln) for ln in body.split('\n'))))
    else:
        cells.append(new_code_cell(body))
nbformat.write(new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}), dst)
```

## v1.3, the next design after the v1.2.1 run (2026-10-28)

The v1.2.1 run (90 minutes, 47 settings) said: the idiosyncratic-vol curve wins pair C on cover; the quantile-grid width
charge wins pair C on P&L and dollars and is the first variant to pass pair A, with the manual charge in; the two act on
different things and had not been combined; the width charge reaches only the 28% of requests with conditional
quantiles; everything else is flat or a clear loser. v1.3 does four things.

**Pruned, by config, not deleted.** Every retired setting can be re-enabled by one field. Retired: stretch 0.25 and
logit 0.25; the walk-forward curve picks (`curve_walk_forward`, they converged to 1 and 1.5 and quoted the same thing);
the tercile and state refits (`curve_refits`); concentration k > 1 and the hedged book term (`k_grid`, `book_k_grid`);
b(x) as a quote (`value_quote`; the adverse-selection table is still produced); the factor premium (`premium_on`); the
quantile location value and lambda 2 (`qg_forms`, `qg_lambda_grid`); the call refit, kappa by group, the common move
and the V5 sd (`call_forms`, `call_idv`). 47 settings become 18 before the stacks.

**The width charge on every request [7d].** `qg_extend`: walk-forward least squares of log calibrated width on the
request's key group, size bucket, tenor bucket, manual class and log idiosyncratic vol, over the covered rows of the
earlier months; the observed width where it exists, the model's elsewhere. Quoted as `quantile width k | ... | extended`
beside the covered version, with the width model's R2 and the covered share printed by month.

**The stacks [7f].** One requote with both a per-request curve and a per-request charge:
`argmax (x + c + b_i) p_i(x)`. Curves: logit 1.5 on the IPCA residual variance, the same tilt on the trailing residual
sd, stretch 1. Charges: the width charge at `stack_k` from the covered requests and from the extended model, for each
quantile model. Twelve stacks at the default config.

**Month by month [8b].** The top `monthly_top` candidates of pairs A and C at equal win rate (rows better on all four
valuations), the fixed references (logit 1.5, width k=2) and every stack, scored one month at a time against the
frontier comparator of that month: tables of the smallest sceptical delta, cover change and dollars by month, and a
figure with the smallest sceptical delta and the cover change by month for both pairs. `variant_matrix_monthly.parquet`.

**Caching.** The matrix [8] is cached per variant: one entry keyed on the variant's quotes (a fingerprint of the
test-row quote vector), the comparators' quotes, the frontier stage keys, the periods, the markout inputs and the
scorecard code. A run that adds or changes a few settings rescores only those; v1.2.1 rescored all 47 (33 minutes).
Also fixed: the cache key's fallback for a function without source on disk was `repr(fn)`, which carries a memory
address, so an exec'd cell or a lambda never reused its cache; the fallback is now a bytecode fingerprint. The smoke
test reuses all 38 stages on a second run.

What the smoke test checks for v1.3: every path quotes, caches and scores on the synthetic handoff, 30 settings, the
extended width covers 98% of requests, 160 variant-months in [8b], and a second run reuses every stage. The real run
decides the stacks.

## v1.2.1, revisions from the review of the v1.1 run (2026-10-28)

The Variant Lab Review page (Notion, MM Models) read the v1.1 html and raised ten points. Eight are code changes, made
before the v1.2 run so that run carries them; two are caveats the notebook already states and cannot remove.

| Review point | What changed | Where |
|---|---|---|
| The delta is a pooled mean, the t is of unweighted weekly means; signs disagree in some rows | `cluster_t`: the t of the same pooled mean with week-clustered Newey-West standard errors (weekly sums of the deviations, Bartlett kernel in week order) | [3] `cluster_t`, `score_pair` |
| The reading ignores dollars | a `dollars` axis (DV01-trimmed record dollars positive) in the reading; the headline count per pair requires cover, all four valuations and dollars | [3] `dominance`, [8] |
| The +-10 bp frontier cannot reach the variant's win rate when the manual charge is 13 bp | the grid runs by 0.5 to +-10 and by 1 to +-25 (71 widths; the frontier cell takes about twice as long and recomputes); the reading names the residual mismatch where the grid still falls short instead of dropping the row (no threshold: it is a flag on the reading) | config, [4], [8] |
| Two regimes pooled (manual charge about 1 bp May to August, 11 to 13 bp in September and October) | every pair is also scored on two periods, split at the largest month-to-month jump of the median manual charge on the test winnable requests (or `CFG.period_split`), for the frontier and levelled comparators; a `period` column, Pareto sets per period, top-15 tables per period | [4] `PERIODS`, [8] |
| The comparator width mixes sign conventions | one convention, `comparator shift (bp, + = more aggressive)`, for the frontier and the levelling (the levelling width is negated) | `score_pair`, [4], [8] |
| The walk-forward k sits at the edge of the grid; the Sharpe rewards quoting less | the pick is by mean weekly hedged dollars on the earlier months | [3] `wf_pick` |
| The frontier plot averages months whose win rates differ 20x | one curve per month, labelled | [4] |
| b(x) double counts x: the markout from the traded level contains the concession and the objective adds it again | `b(x) = E[markout - x | won at x]`, the move from the quoted mid to the exit; the realised source is also centred on the earlier months' median round trip so the resale markup is not a value of winning anything | [7] |
| The P&L sample share runs 0.79 to 0.98 | not changed: the exit of record covers 44% of winnable requests and the share is reported per row; imputing a markout would invent P&L | - |
| The risk axis rests on a covariance the lab labels uncalibrated | not changed here: the label is carried from the handoff and printed in [2]; the sd axis is read with it | - |

Two review remarks are observations, not errors: the real book is held on a small share of requests, so the real-book
and replayed-book concentration terms agree (that is the finding), and the results on this data are confirmations under
rules written after the v1 run (the setup cell says so; the pre-registered test is on data from November on).

The review's proposed candidate, stretch 1 alongside the manual charge, is not what the run supports: in pair A every
curve variant is within 0.01 bp of production at equal win rate. The curve's value shows in pair C, without the manual
charge, and the balanced choice there is logit 1.5 (96% of wins kept, the same cover saving, the higher common-markout
t). Both are frozen as the candidates for the November data.

## v1.2, the lab pieces never quoted through the optimizer (2026-10-28)

The v1.1 run settled the families it carried: the idiosyncratic-vol curve wins pair C, the concentration term is a risk
tool, b(x) from realised round trips is negative, pair B is closed. Three parts of the lab had never been quoted at all:
the factor premium (8b), the conditional quantiles in beta space (10) and the call-structure work (9). v1.2 quotes them,
each in the form its object suggests, on the same optimizer, the same replay and the same v1.1 scorecard. The lab writes
them at the request level in `models.parquet` (15a [2], lab v4.9.3); cell 2b loads it and the three families skip
themselves, saying so, when a column is absent.

| Family (cell) | Term changed | Form in x | Settings |
|---|---|---|---|
| Factor premium [7c] | `c + b_i`, `b_i = -E[dy_i]` over the record horizon (a bid gains when yields fall) | constant | as stated (slope 1); slope calibrated walk-forward by OLS of the record markout on `b_i` over production's wins in the earlier months; stated plus carry: 3 |
| Quantile grid [7d] | the markout distribution `M = -dy` from the model's quantiles, brought to the horizon by a walk-forward location slope and scale ratio on production's wins | width charge `c - gamma_k W_i` (constant); location value `c + s med_i` (constant); downside penalty `argmax p(x)[x + c + E M - lambda E(-(x + c + M))+]` (a function of x) | grid and location-scale models on the next-day residual; k in {1, 2}; lambda in {1, 2} plus walk-forward: 11 |
| Call structure [7e] | the fill curve by call group; the tilt strength by group; the tilt on the call-aware residual sd; the group's common move as value | refit: a distribution per group; kappa by group: a function of x; trailing-sd tilt: a function of x; common move: constant | refit (n0 by the previous month's LL); kappa per group picked by the group's LL; logit 1.5 on `idv_trail_v5` and on `idv_trail_v0` (the control on the same window); common move with the slope calibrated within the group: 5 |

No thresholds and no new constants: every scale is a walk-forward calibration on production's wins in the earlier months,
and the short k and lambda grids are read by the same walk-forward Sharpe pick as the concentration term. The downside
penalty is the one new optimizer path: it is solved row by row (no dedup, the term is request-specific along the grid)
in chunks of 2,000 requests, so it is the slowest of the new settings. The matrix [8] and the registry [9] pick the new
rows up unchanged; the matrix page in the review is regenerated from the same output.

What the synthetic smoke test can and cannot say about v1.2: the synthetic `models.parquet` is noise with the right
shapes and gaps, so the smoke run checks only that every path quotes, caches and scores. The real run decides.

## v1.1, after the first real run (2026-10-27)

The run of v1 on the 10-23 data (92 minutes, 60 settings) said four things: the manual charge's allocation is information and
no factor term replaces it (every pair B row negative); on top of the manual charge nothing adds (pair A); the idiosyncratic-vol
fill curve carries real information (the only sigma that improves the held-out log-likelihood; accepted in pair C under all three
sceptical valuations), which the traders already price; and the own-CUSIP inventory term cut the book's risk by two thirds at
production's win rate for 0.9 bp of cover with flat P&L, and failed a bar written for cover variants. Dollar-weighted and
count-weighted deltas disagreed, and the curve family took 54 minutes.

v1.1 changes the reading and the families accordingly. The desk's objective is: hold the win rate near production's, lower the
cover, raise the P&L under the sceptical exits. There is no threshold anywhere.

- **The comparator is production widened to the variant's win rate** through its own optimizer, month by month, on a 0.5 bp
  width grid from -10 to +10 (the residual win-rate mismatch is a column). Each variant is also read against production as it is
  and levelled to production (the v1 reading). The width grids of both regimes are cached (`var_front_x_M`, `var_front_x_0`,
  float64 so the width-0 column reproduces production bit for bit; about 430 MB each at the lab's scale).
- **Every cell is a change with its t**: win rate and its change, wins kept, new wins, wins given up, bp left on the table on
  cover wins and its change, bp saved on kept cover wins, crossing, the delta under five valuations with weekly Newey-West t and
  months positive, the smallest of the four sceptical deltas, dollars count-weighted, DV01-weighted and DV01-trimmed at the 99th
  percentile with the top-1% share, the delta on requests at the month's modal manual charge and on request-specific ones, the
  P&L sample share, the book's ex-ante sd against the comparator's, hedged dollars per dollar of sd. A sign-only **reading** says
  which axes the variant is better on (cover, P&L under all four valuations, risk), and a **Pareto set** per pair and comparator
  lists the variants nothing else beats on cover, sceptical P&L and risk at once. The v1 gates are retired.
- **Fill curve [5]**: idiosyncratic vol only; stretch and logit with the parameter picked walk-forward by cumulative held-out
  log-likelihood; the own-curve refit by vol tercile; and the state-conditioned refit by vol tercile x manual class (the month's
  modal charge or a request-specific one) x the lagged common-move tercile of the day. The stretch ratio and the logit shift are
  quantised to 0.01 so each distinct curve is built once per key and the argmax is solved once per distinct (parameter, charge).
- **Concentration term [6]**: `gamma sigma_h^2 (Q_i + u_i / 2)` with `Q_i` the real inventory, the WFTrade start-of-day DV01 in
  the CUSIP (cell 15a writes `positions.parquet`) plus the policy's incremental fills over production's in the last h days. The
  replayed-book version and the hedged book term are kept as references. The half-spread, the full book term and the combined
  curve-plus-charge are dropped (flat or dominated in v1).
- **Value b(x) [7]**: `E[markout | won at x]` per key group (coupon bin x call group) and x-quintile of production's concessions,
  walk-forward, empirical-Bayes shrinkage to the group mean, from the exit of record on production's wins and from the realised
  WFTrade round trips on the matched fills (cell 15a writes `realised.parquet`); quoted as `argmax (x + c + b(x)) p(x)`, and
  combined with the walk-forward logit curve. The b(x) tables of the last month are printed: the adverse-selection curve.
- **Signals [7b]** off by default (`CFG.signals_on`).

Objects added: `FRONT_X`, `FRONTIER` (win rate, cover and P&L by width and month), `POS_SOD`, `_real_bp`, `_man_modal`,
`_mkt_state`, `_kgrp`, `BX_TAB`, `VM['reading']`, `VM['pareto']`, `VERDICTS` (readings per pair and comparator). Registry
sections unchanged in name; `variants_value` added.

## How it connects to the lab

1. Run the lab through Section 17 and the new **Section 15a, the handoff**, which writes
   `artifacts_ipca_lab/handoff_rfq_variants/`: `requests.parquet` (the request frame with the logged quote, charges,
   outcome, thresholds, exits and signals), `arrays.npz` (betas, exposures, idiosyncratic variances, the volatility
   measures, every 14f markout, the hedge term, the 14g tilt inputs), `objects.pkl` (the production fill curves, the
   picked covariance by day, the run metadata), `holding.parquet` (the realised par-weighted median holding time by
   tenor x call bucket from Section 17), `positions.parquet` (end-of-day par by CUSIP from the WFTrade book) and
   `realised.parquet` (the matched lots' realised round trips per request row). About 1 GB at the lab's scale.
2. Run `muni_rfq_factor_variants.py` with `MUNI_LAB_ROOT` pointing at the lab's root (default the lab machine's path).
   Its outputs go to `artifacts_ipca_lab/variants/`: the cache, the figures, `variant_matrix.parquet` (every cell of
   the matrix) and `results_registry_variants.json`.

The optimizer is copied verbatim from lab 14 [1] and cell [2] asserts that it reproduces the logged optimal quantile
on at least 95 percent of requests (within 0.25 bp) and every logged win and loss before anything is scored.

## The objective, and what each family changes

Production: `x* = argmax over x of (x + c) p(x)`, `c = -(manual + liquidity + risk)`, the quote `y = m + x*/100 + adj`.
Every variant changes one term and is quoted by the same argmax on production's own curves.

| Family (cell) | Term changed | Settings |
|---|---|---|
| Fill curve by volatility [5] | `p(x)` becomes `p(x (sigma_ref / sigma_i)^a)` (stretch), `sigmoid(logit p + kappa z_i)` (logit tilt), or the desk's own curve by volatility tercile shrunk to production's (`n0` by the previous month's log-likelihood) | a in {0.5, 1}, kappa in {0.5, 1}, tercile; sigma in {systematic, idiosyncratic, total (14d), conditional scale (lab `a_risk`)}: 20 |
| Charges in the Avellaneda-Stoikov form [6] | `c - gamma s_i`: half-spread `s = sigma_h^2 u / 2`; own-CUSIP inventory `s = sigma_h^2 (Q_i + u_i / 2)`; book inventory `s_i = h (2 e'S E + e'S e + v) / (2 u)` (lab 14e), full and hedged (tenor block zeroed, hedge cost added); combined = the parametric curve form with the best held-out log-likelihood plus the book charge | `gamma_k = k / median s` on production's book in the first test month, k in {1, 2, 3, 5}; plus a walk-forward k per family: 25 |
| Model signals as charges [7] | `c + k z_i`, `z_i` the walk-forward normal score of the signal, positive = expected to richen | state space, residual level, peer gap, filtered convergence, copula conditional PIT; k in {1, 2, 3} bp per sd; plus a walk-forward k per signal: 20 |

`sigma_h^2 = sigma_5^2 h_i / 5` with `h_i` the realised holding of the request's tenor x call bucket (Section 17), else
the lab horizon. `sigma_ref` is the key's median sigma and `z_i` the standardised log sigma, both on the months before
the request's month. The fill-curve forms are validated first at the logged concession (held-out log-likelihood of the
logged outcome against production's curve; calibration by sigma decile), reported, not gating.

## The four pairs and the bars

| Pair | Variant quoted with | Production quoted with | Raw bar | Levelled bar |
|---|---|---|---|---|
| A | manual kept | manual kept | lab Section 14 bar under the record exit, the guard and (if the win rate rises) the common markout: weekly NW t >= 3, months > 0 >= min(4, N), at or above the width frontier | wins kept >= 80%, cover saved on kept cover wins >= 1 bp, hedged delta under the record exit >= 0 in >= min(4, N) months |
| B | manual removed | manual kept | as A, against production's frontier | as A levelled |
| C | manual removed | manual removed | as A, against the no-manual frontier | as A levelled |
| D | production, manual removed | production, manual kept | none: the reference | the levelled row is the "flat" control of lab 14c |

Levelled = a constant width per month, the one nearest zero that brings the variant's win rate on the month's paired
requests to its comparator's, by bisection. Every cell carries: win rate (both), wins kept, new wins, wins given up, bp
left on the table on wins with a cover (both), bp saved on kept cover wins, crossing on new wins, the levelling width,
the delta per request with weekly Newey-West t and months positive under five valuations (the exit of record, the
guard, the common markout for new wins, the hedged record exit, the lab's other-side exit), dollars per month, the
position against the width frontier, the ex-ante daily sd of the policy's own book against the comparator's, gross
DV01. Each row also carries one of four labels: accepted; positive evidence, gate not met (record t >= 2, no sign flip
under the guard or the common markout); negative evidence (t <= -2, or a sign flip with |t| >= 2); inconclusive.

Objects: `VARIANTS` (name to quotes with and without the manual charge), `Y_PM`, `Y_P0`, `FRONTIER`, `BK_BASE`,
`PAIR_D`, `CURVE_VAL`, `CURVE_CAL`, `S_MED`, `CHARGE_PICKS`, `VM` (the matrix), `VERDICTS`. Registry sections:
`variants_handoff`, `variants_references`, `variants_curves`, `variants_charges`, `variants_signals`,
`variants_matrix`, `runtime`. Figures `v4_references`, `v8_matrix`.

## Engineering notes

- The cache key binds the spec, the data dependencies, the stage's source and only the config fields the stage reads
  (`cfg=` on `cached`), so a change to `k_grid` rebuilds the charge family and nothing else.
- The weekly and monthly aggregations inside the scorecard use `np.bincount` on pre-factorised codes; the comparators'
  books are computed once (`BK_BASE`). At the lab's scale expect the curve family in 2 to 4 minutes, the charge family
  in 20 to 40 minutes on the first run (32 sequential book replays, cached after), the signals in under a minute and
  the matrix in 5 to 10 minutes.
- Every sequential replay builds the policy's own book from the first day with a covariance; days outside the test
  months quote production and only build the book, as in lab 14b.

## The synthetic smoke test (v1.1)

The harness runs Sections 17 and 15a from the repo and writes the handoff with positions and realised lots for its 7,000-request
world; the runner executes the notebook in a fresh namespace. Last run: all cells, 25 settings in both regimes, 225 matrix rows
(three comparators), both figures, every registry section, 19 seconds; the replay check exact. On the synthetic world the b(x)
tables are empty (no key group has 50 wins) and the refits choose production's curve; the numbers mean nothing for the desk.

## The synthetic smoke test (v1)

The lab's harness (session scratch, not committed) now runs Sections 17 and 15a from the repo and writes the handoff
for its 7,000-request world; a runner executes the variant notebook cell by cell in a fresh namespace on that handoff.
Result of the last run: all nine cells run, the replay check passes bit for bit (max |d| 5e-14 bp, every logged win and
loss reproduced), 65 settings are quoted in both regimes, 390 matrix rows are scored, both figures and every registry
section are produced, in 13 seconds. The synthetic world's five signals are copies of one residual matrix, so the five
signal variants coincide there; its production curve is not calibrated to its fabricated outcomes, so the fill-curve
validation shows no gain and the tercile refit chooses production's curve. None of those numbers mean anything for the
desk; they show the code paths.

## What only the real run will show

- Whether any volatility definition improves the held-out log-likelihood of production's fill curve, and by how much
  the calibration by sigma decile moves; if none does, the curve family is dead before the P&L and the combined charge
  falls back to the best parametric form anyway.
- Pair D: the manual charge's own scorecard, the number every variant is read against.
- Whether the own-CUSIP inventory term, which production does not have, lowers book risk at a cover cost below the
  book term's, and whether any charge survives in pair C, where the traders' information is taken out of both sides.
- The first run's runtime of the charge family; cut `k_grid` to (1, 3) if the sequential replays exceed an hour.
