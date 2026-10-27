# RFQ factor variants: the production optimizer as the judge (variants v1)

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

## How it connects to the lab

1. Run the lab through Section 17 and the new **Section 15a, the handoff**, which writes
   `artifacts_ipca_lab/handoff_rfq_variants/`: `requests.parquet` (the request frame with the logged quote, charges,
   outcome, thresholds, exits and signals), `arrays.npz` (betas, exposures, idiosyncratic variances, the volatility
   measures, every 14f markout, the hedge term, the 14g tilt inputs), `objects.pkl` (the production fill curves, the
   picked covariance by day, the run metadata) and `holding.parquet` (the realised par-weighted median holding time by
   tenor x call bucket from Section 17). About 1 GB at the lab's scale.
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

## The synthetic smoke test

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
