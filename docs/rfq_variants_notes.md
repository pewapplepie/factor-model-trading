# RFQ factor variants: the production optimizer as the judge (variants v1.1)

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
