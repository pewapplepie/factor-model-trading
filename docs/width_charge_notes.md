# The width charge notebook (width charge v1.0)

`muni_width_charge_research.py` is one notebook from the data to the backtest for the quoting change the variant programme (`muni_rfq_factor_variants`, v1.1 to v1.3) found worth putting live: a per-bond risk charge sized to the dispersion of the bond's own residual move, the "quantile width charge". It is written to be read by people who have not followed the lab and variant notebooks: every section says what it builds, why, and which plot to look at. It is self-contained: it reads the data pipeline directly and needs neither the lab's cache nor its handoff.

## What it does, section by section

| Section | Content | Output to look at |
|---|---|---|
| 1 | Config (`RunConfig`), stage cache keyed on the config fields each stage reads, its data and its code (bytecode when the source is not on disk), helpers | |
| 2 | Data: the evaluation panel, the MMD curve, the MSRB tape (and the trade-size characteristic from it), the RFQ log (`strategy_output` joined to `track_rfq`), WFTrade fills and the inventory book | `02_data`: covered CUSIPs, requests and win rate by month |
| 3 | The factor model: the lab's record spec, lifted unchanged (excess-over-MMD target, 20-odd rank-normalised characteristics, IPCA by alternating least squares, K = 3, monthly refit on an expanding window, Procrustes-aligned Gammas, the point-in-time residual) | `03_factor_model`: Gamma, out-of-sample fit by month |
| 4 | The residual's conditional distribution (lab Section 10): the lookup grid by beta cluster x activity bucket and the location-scale model on beta space + state, walk-forward by month; pinball skill and calibration; the bond-day quantiles of the next-day residual and total move | `04_conditional_quantiles` |
| 5 | The desk's requests through production's optimizer (lab Section 14): the pfill snapshots, the replay gate (every month within 0.25 bp on 95% of requests), the cover yields through the pricing service, the five outcome groups, the vanilla pass (every logged win and loss reproduced) | `05_requests`: the charges by month, the cover distribution, the concessions |
| 6 | What a win is worth: the comparable-bond exit of record (coupon bin x call structure x rating group x tenor bucket, mark-to-P, h = 5; the lab's circularity guard, validated here against the own print and against the realised sale), the own other-side print, the common markout, DV01; the realised WFTrade round trips (FIFO lots, the request match) | `06_exits` |
| 7 | Replay against the desk: the replay's wins matched one to one to the AM60 BID fills (both directions, by month, by trade type, the traded yield), the inventory the replay implies against WFTrade's (daily par, per-CUSIP par, CUSIP-days), and the book's P&L since April marked to the closing marks (new position, position, carry) | `07_replay_vs_desk` |
| 8 | The width charge: the band at the request (the bond-day quantiles from the last close strictly before), the scale ratio to the holding period (walk-forward on production's wins), W = half the 80% band, gamma = k / median W on the first test month, the charge `c - gamma W_i`; the variations (k, grid vs location-scale, total vs residual, every request through a width model with gamma rescaled on the extended population); a worked example on two real requests of one key | `08_width_charge` |
| 9 | The backtest: pairs A (manual kept), B (manual removed vs production with it), C (both removed), D (the manual charge itself); comparators raw, at equal win rate (production through its own optimizer on the width grid, month by month) and levelled; the paired scorecard with three sceptical valuations and the realised round trip beside, week-clustered t, DV01-trimmed dollars, gross DV01 as the risk axis; pooled and month by month, scoring cached per setting | `09_frontier_points` (the settings as points on production's own frontier, one panel per month), `09_headline_bars`, `09_heatmaps` (month x setting), `09_cumulative` (the candidate's dollars by day) |
| 10 | Where the edge comes from: by quintile of W, by key group, the attribution into common fills / new wins / wins given up, the adverse-selection table of production's own wins by W quintile | `10_edge` |
| 11 | The candidate in one table; the variations; the monthly table; the summary line; registry and runtimes | `results_registry_width_charge.json` |

## What differs from the variant notebook

- **The exit of record** is the comparable-bond bucket exit (characteristics only), not the IPCA-peer exit of lab 14f. The peer exit needs the Section 11 baskets; the bucket was its circularity guard and the two sat within a bp of MAE against the own print. The sceptical set is therefore the bucket exit, the common markout and the own other-side print; the hedged valuation is not here (no request-level covariance). The realised WFTrade round trip is reported beside, on the fills the desk actually sold.
- **The risk axis** is the book's gross DV01 over the holding window, not the ex-ante sd under the lab's covariance.
- **The extended charge** rescales gamma on the extended population's own median width (the v1.3 run scaled it on the covered bonds and over-widened).
- **The test months** are the replay months after the first full one, so every calibration has at least a month of production's wins behind it.

## Running it

```
python3 tools/py2nb.py muni_width_charge_research.py muni_width_charge_research.ipynb   # the converter in the scratch tools (percent format -> ipynb)
```

Run from the lab root (`MUNI_LAB_ROOT`, default the current directory) with `data_pipeline/` holding `muni_data_pipeline.py` and the pulled stores, and `prod_pfills/` holding the pfill snapshots. Everything it writes goes to `artifacts_width_charge/` (cache, figures, the matrix parquet, the registry). The pricing service is called for cover yields and DV01 and cached on disk; a `PRICING_OVERRIDE` global with the same signature replaces it away from the desk's network. `MUNI_WC_CFG` takes JSON overrides of `RunConfig` fields (used by the smoke run).

Expected runtime on the desk's data: the factor model walk-forward and the quantile models are the slow stages the first time (the lab's figures suggest 15 to 25 minutes together); the replay, exits and frontier grid another 25 minutes; the 14 settings a few minutes each to quote and score. A rerun with nothing changed reuses every stage.

## Pre-registered

`width_charge_v10` (Section 1): the candidate (residual band, location-scale model, k = 2, manual charge kept), the variations, the comparator of record, the valuations and the sign-only reading, fixed before the notebook ran.
