# IPCA as the desk's risk model: lab v4.9.2 (Sections 14d to 14g, 16, 17 and the 15a handoff)

Companion to `muni_ipca_factor_lab.py` / `.ipynb`. The notebook in the repo is the v4 lab reconstructed from
`muni_ipca_factor_lab_v4.html` (markdown cells re-derived from the rendered html, code cells verbatim) with the new
sections 14d to 14g, 16 and 17 inserted between 14c and 15. Nothing in cells 1 to 14c is changed, so every existing cache
on the lab machine is reused; the new cells read the objects those cells leave behind. Section 16 is the lab machine's own
cells (from `muni_ipca_factor_lab (1).html`, code verbatim); it sits before Section 15 here, not after it as in that run,
because Section 15 closes the cell log and writes the registry, and Section 17 records into both.

Regenerate the `.ipynb` from the `.py` with the percent-format converter (no jupytext metadata in the repo):

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

## Why these cells exist

The v4 run (Sections 13 to 14c) read as a test of IPCA as the book's risk model. Four things made it not that test:

1. On the winnable requests only 20 percent carried the bond's own beta; 77 percent used a tenor-by-call bucket mean.
   In the variance decomposition the factors carried 1 to 24 percent, the rate block and the idiosyncratic block the rest.
2. The core IPCA property, beta as a function of characteristics (`beta = Gamma' z`), was unused: a bond outside the
   estimation panel got a bucket mean, and a bond with a three-week-old mark got nothing fresh.
3. The expanding-window Sigma under-forecast the realised daily move of both books by about 1.5 times.
4. Every positive result in 14 and 14c lived in new wins, valued at the bond's own next other-side print, which carries
   the bid-offer and the reversion of a noisy print. The flat formation at equal win rate showing 500k dollars a month
   against production's 355k was the symptom.

## What each new cell does

| Cell | What it builds | Pre-registered bar | Objects it leaves |
|---|---|---|---|
| 14d [13] | `z` for every requested bond from the evaluation panel's last row, the request (fair value as the yield level, our bid price as the price level, DV01 as duration, tenor / call / coupon / rating buckets), the static track and the print tape; rank-normalised against the model panel on the last model date before the request; `beta = Gamma' z` with the aligned Gamma strictly before the request; a ridge of log residual sd on `z`; the fresh factor-implied move since the bond's last observation | Pearson >= 0.9 on factors 1 and 2 between the request-fields `Gamma' z` and the beta of record, on requests that have both; else everything downstream is labelled uncalibrated | `BT_BETA_GZ`, `BT_SRC_GZ`, `BT_IDV_GZ`, `BT_EXPO_GZ`, `BT_V_GZ`, `GZ_Z`, `BT['bt_level_fresh_bp']`, `GZ_LABEL` |
| 14d [14] | EWMA covariances (half-lives 10, 20, 40) of the stacked MMD tenor moves and factor realisations against the expanding window; EWMA idiosyncratic variance; scored on the vanilla and objective books by the realised over ex-ante ratio and the Gaussian log score; walk-forward pick; the 14c risk attribution redone on the new betas | the picked Sigma's ratio in [0.8, 1.25] on the vanilla book in >= 4 of 5 test months and in the last month | `BT_SIGMA_C`, `BT_IDV_C`, `BT_V_C`, `BT_VOL_C`, `bt_book_g`, `SIGMA_LABEL` |
| 14d [15] | the objective rebuilt with the selection bias on the other-side markout net of the day's common markout (`Xm`), as a leaderboard row; every variant and every 14c formation re-scored with new wins valued at (a) the record exit, (b) the common markout only, (c) the factor-implied move over the holding | the objective line is kept only if the net-b row has delta >= 0.05 bp net of common with NW t >= 2 and >= 4 months positive; a variant that raises the win rate passes only on (a) and (b) together | `BT_VARIANTS[...net b]`, `BT_OBJ_B_NET`, `bt_board` (one row added), `bt_cons`, `bt_fm_cons`, `_obj_label` |
| 14e [16] | `s_i` decomposed into rate, factor, idiosyncratic and the book cross term on vanilla's book; the 14b replay under two definitions (full; hedged = rate block sold at `HEDGE_COST_BP`, default 1 bp per fill) on vanilla, the objective and the net-b objective, k in {1, 2, 3, 5}; the k pick by the weekly Sharpe of the hedged dollar P&L; frontiers raw, net of common, hedged | a pick passes if its hedged dollars per dollar of ex-ante sd >= 1.3x its base's and hedged P&L >= 70% of its base's in >= 4 of 5 months | `BT_RISK_OPTO_GZ` (cached stage `lab_rfq_riskaware_gz`), `bt_risk_gz`, `bt_pick_gz`, `_PPG`, `BT_RISK_BOOKS_GZ` |
| 14e [17] | the manual charge regressed on the risk measures, size, signal age, call and tenor buckets, the bond's recent losses on the desk's own fills, and a month effect (pooled R2, partial R2 per block, weekly Fama-MacBeth); a reallocation that keeps the traders' monthly level and allocates it by risk, lambda multiple picked on earlier months by cover saved | keeps >= 80% of vanilla's wins, saves >= 1 bp of cover on kept cover wins against the flat reallocation, delta vs vanilla at the common-markout valuation >= 0 | `bt_manual_reg`, `bt_manual_fm`, `BT_RL_OPTO` (cached stage `lab_rfq_realloc`), `bt_rl_tab` |
| 14e [18] | the risk-adjusted exposure table (raw, net-of-common and hedged dollars; calibrated ex-ante and realised sd; hedged dollars per dollar of sd; gross DV01; factor exposures; idiosyncratic share; worst week; Sharpe raw and hedged) and the frontier figure for every policy | none; a report | `bt_exposure`, registry section `14e` |

Every bar is written into `PRE_REGISTERED` at the top of the cell that scores it, dated 2026-10-21, before any result.
Registry keys: `lab_gz_betas`, `lab_sigma_calibration`, `lab_rfq_netb`, `lab_rfq_riskaware_gz`, `lab_rfq_realloc`, `14e`.
Figures: `14d_gamma_z_betas`, `14d_sigma_calibration`, `14d_conservative_valuation`, `14e_riskaware_gz`,
`14e_manual_reallocation`, `14e_exposure_report`.

Two lines are closed in the 14d markdown on the v4 evidence: the request-level skew of Section 13 and the alpha
overlays of Section 14. No further overlay variants are built.

## 14f: the exit of record (added in v4.7, after the real run of 14d and 14e)

The real run of 14d [15] showed that every positive result in Sections 14 and 14c depended on how a *new win* was valued:
the bond's own next print on the other side of the trade, which carries the bid-offer and exists only on the liquid two
thirds of winnable requests. The backtest framework (Section 6b, agreed 2026-10-20) had specified a different exit,
comparable-bond prints anchored at the evaluated close, P or D only, mark-to-P primary, validated against realised own
prints, with a circularity guard. Section 14f builds that exit and re-scores everything under it.

| Cell | What it builds | Pre-registered rule | Objects it leaves |
|---|---|---|---|
| 14f [19] | anchors (last evaluated close before the date) for every request and every P/D print; print deviations from their own anchors; five estimators of the peer level over (t, t+5]: bucket (coupon bin x call structure x rating group x tenor bucket, characteristics only), IPCA peers (the Section 11 baskets, widened to the bucket where thin), factor / MMD, own P and own D print within 10 business days; validation of each against the realised own P print (coverage, bias, MAE, slope, by month) | exit of record = the comparable-bond mark-to-P estimator with the lowest pooled MAE and coverage >= 50% of winnable requests; the bucket mark-to-P exit is the guard | `BT_ANCHOR`, `BT_BUCKET`, `BT_EXIT_LVL`, `BT_M_EXIT` (markouts per exit in the Section 14 convention), `EX_RECORD`, `EX_GUARD`, `bt_exit_val` |
| 14f [20] | every variant, the 14c formations, the 14e reallocations and the risk-aware policies re-scored under the exit of record with the Section 14 machinery; manual requests as a segment; the hedged P&L of production's own fills under the exit of record | a variant passes if it clears the 14 bar under the exit of record and, when built on a model, also under the guard | `bt_board_f`, `bt_policies_f`, registry `lab_exit_of_record`, `lab_rfq_rescore` |

Decisions the runs resolved, written into the 14f markdown for the framework's log: settle T+2; the call-structure pfill
keys start 2026-03-27; quantity is par dollars; manual requests are 17 percent of the flow, scored as a segment.

On the synthetic smoke run the exit of record came out as the IPCA peers mark-to-P (MAE 8.3 bp against 8.8 for the bucket
and 15.2 for factor / MMD, coverage 87% / 62% / 94% of winnable requests), zero of 17 variants pass with the guard, and the
mark-to-P exits carry no bid-offer (production's fills mark 6 bp against 24 bp on the other-side exit). Those numbers only
show the code paths. Runtime on the synthetic scale: [19] 4 s, [20] 2 s; at the lab's scale expect the two merge_asof
anchors and the own-print lookup (cached stage `lab_rfq_exits_own10`) to dominate, around two to three minutes.

What only the real run will show: which comparable-bond exit validates best and with what MAE against the own print
(if none reaches 50% coverage, the bucket is the record by construction); whether any variant survives the exit of
record and the guard; and the hedge contribution under the exit of record, which no longer carries the S-side bid-offer.

## 14g: the manual charge kept and tilted (added in v4.8)

14c and 14e [17] replaced the manual charge and lost the traders' call-bucket information; the risk-aware charge of 14b and
14e [16] kept it but could only add to it. 14g keeps every request's manual charge, tilts it with a model quantity, and
removes the level effect of the tilt with a constant width set by bisection so that each month's win rate on winnable
requests equals production's (in-sample level, as the 14c calibration was). What remains is pure re-allocation at the same
number of wins.

| Cell | What it builds | Pre-registered bar | Objects it leaves |
|---|---|---|---|
| 14g [21] | five tilts: systematic vol and idiosyncratic vol ([13] betas, [14] Sigma); the signed factor covariance of the fill with production's own start-of-day book (two-sided: cheaper on a fill that offsets the book's factor exposure); the trailing 40-day correlation of the bond's residual with its 20 beta-space peers' median residual (the dependence tier the Section 12 copula models); 1 / (0.25 + years to call) on callable bonds. Four settings each: additive at 0.5, 1 and 2 times sd(manual) / sd(g), and multiplicative manual x exp(0.5 z). Requoted through the production optimizer, then levelled (cached stage `lab_rfq_tilts`) | none; a construction | `TG_TILTS`, `BT_TG_OPTO` (optimal yields with the width folded in), `bt_tg_level` |
| 14g [22] | each setting at production's win rate on the test months: vanilla wins kept, cover saved on kept cover wins, crossing on new wins, markout per fill under the exit of record (raw and hedged by the MMD move at tenor), ex-ante daily sd of the policy's own book, gross DV01, factor exposures, hedged dollars per dollar of ex-ante sd; where each tilt moves the charge against the traders by call bucket | cover saved on kept cover wins >= -0.5 bp and either ex-ante sd <= 0.9 x production's or hedged markout per fill >= production's + 1 bp, in >= 4 of the 5 test months | `bt_tg`, `bt_tg_move`, registry `lab_rfq_tilts` |

The headline metric is hedged dollars per dollar of ex-ante risk under the exit of record, which a DV01-per-fill tilt cannot
game; DV01 is a reported column. On the synthetic smoke run all twenty settings level to production's win rate within
0.15 points, none passes the bar, and the time-to-call tilt moves the charge up on the 3-to-12-month call buckets and down
on bullets, which is the direction the 14e regression said the traders already price. Runtime on the synthetic scale:
[21] 2 s for 100 requotes, [22] 2 s; at the lab's scale expect about 3 minutes for [21], cached after the first run.

## 16: the AM60 inventory book from WFTrade (lifted from the lab machine's run)

The five cells 16a to 16e call pipeline functions that are not in this repository: `mdp.load_wftrade`,
`mdp.build_inventory_book`, `mdp.book_snapshot`, `PIPE.wftrade_inventory`, and `CFG.end_date`. On a machine without
them the cells fail at the first line, and Section 17 then skips itself (its `HAS_WFTRADE` guard). The objects Section 17
reads from them: `WFTRADE_BOOK` (`trade_number`, `cusip`, `side` in {BID, ASK}, `amount`, `price`, `yield`,
`execution_timestamp`, `exec_date_ny`, `uncovered_par`), `WFTRADE_EDGE` (`trade_number`, `mark`, `mark_date`: the
evaluated clean price at the trade), `_mk` (`cusip`, `cpn`) for the carry, and `WFTRADE_END`.

Three things the lab machine's run of 16 showed that 17 is built around: the ECN link column is blank, so request-level
validation cannot come from a key; 16d's timing match (booking within 60 seconds of the RFQ clock, exact par) found 59
links; and the desk's realised P&L on complete-book months was execution edge plus carry minus a large inventory
component, with the per-trade edge against the mark positive on buys and negative on sales.

## 17: realised round trips as the validation target (added in v4.9)

Section 14 imputed every exit from the bond's next other-side print, 14f from comparable bonds' prints at a fixed
five-day horizon. For production's own fills neither is needed: WFTrade carries the sale. Section 17 pairs the buys with
the sales that close them, ties the buys to their requests without an ECN key, and uses the realised sale as the target
against which every exit estimator is judged. The fixed-horizon exit stays for counterfactual fills only, at the horizon
the desk realises. Beside P&L, every scorecard carries the logged facts that need no exit at all: the win rate, the bp
left on the table on wins with a cover, the distance to the winning print on lost requests, and the crossing on new wins.

| Cell | What it builds | Pre-registered bar | Objects it leaves |
|---|---|---|---|
| 17 [23] | FIFO round trips per CUSIP from `WFTRADE_BOOK` in execution order (an uncovered sale's par is excluded from the paired par); per buy lot the sold par, the par-weighted sale price, yield and date, the par-weighted holding time in business days, the unsold remainder; the round trip in points per 100 split exactly into buy edge, inventory and sale edge against the evaluated mark, plus carry; in yield bp as 100 (buy yield - sale yield); the request match: `algo_won` requests to AM60 BID fills on the same CUSIP and New York date, exact par first then the nearest par within 25 percent, one to one, with the yield consistency of the pairs reported; the par-weighted holding-time distribution overall and by call, tenor and size bucket; the record horizon = the par-weighted median holding of matched fills rounded to the nearest of 1, 3, 5, 10, 20, 40, 60 business days | none; a construction. The decomposition identity is asserted to 1e-6 points | `WF_PAIRS`, `WF_LOTS`, `WF_MATCH`, `RT_HOLD_TAB`, `EX_H_REALISED`, `HAS_WFTRADE`, `RT_PAR_TOL`, `RT_H_GRID`, registry `lab_wftrade_roundtrips`, figure `17_roundtrips` |
| 17 [24] | every 14f estimator (lab other side, bucket and IPCA peers mark-to-P and mark-to-D at h = 5, factor / MMD, own P and own D within 10 days) and the bucket estimators at the horizon nearest each lot's realised holding, against the realised round trip of the matched closed lots: bias, MAE, correlation, slope, pooled and by holding bucket (at least 30 lots per segment); production's scorecard by month: win rate, cover left on won-with-cover (mean and median), short of the winning print on lost (median), matched closed lots, realised round trip (bp and points), holding, buy and sale edge against the mark, inventory, and the modeled markouts on the same fills; the realised round trip by cover bucket (0-1, 1-3, 3-10, > 10 bp; won without a cover; manual) with the NW t of its mean; the Section 14 variants re-scored with the estimator of record against realised sales at the record horizon; the policy scorecard joining the 14f deltas and passes, the Section 14 cover and crossing, the 14g wins kept, hedged markout and risk, and the realised-horizon delta and pass | the estimator of record for counterfactual fills becomes the one with the lowest MAE against the realised sale among the comparable-bond estimators with coverage >= 50 percent of matched fills; a variant passes at the realised horizon on the Section 14 bar (weekly NW t >= 3, >= 4 months positive, above the width frontier) | `bt_exit_val_real`, `EX_RECORD_REALISED`, `EX_RECORD_H`, `bt_rh`, `bt_real_month`, `bt_real_cover`, `bt_policy_card`, registry `lab_wftrade_validation`, figure `17_realised_validation` |

Two readings the section is built to give. First, the adverse-selection question on real exits: if the realised round
trip falls with the bp left on the table (the 0-1 bp bucket earning more than the > 10 bp bucket), winning by too much
is costing money on the desk's own sales, not only on an imputed print. Second, whether the modeled markouts on
production's fills (14f: +5.6 bp under IPCA peers mark-to-P, +18.2 under the lab's other side) agree with what the desk
realised on the same fills; the by-month table puts them side by side, and the validation table says which estimator
to believe for the counterfactual policies.

The 14g tilts are not re-scored at the realised horizon (their hedged markout uses the MMD move at tenor over h = 5);
that is one `bt_tg`-style pass away if the realised horizon turns out far from 5 days.

## 15a: the handoff to the variant notebook (added in v4.9.2)

One cell before Section 15 writes everything `muni_rfq_factor_variants.py` reads to `artifacts_ipca_lab/handoff_rfq_variants/`:
`requests.parquet` (the request frame: logged quote, charges, outcome, thresholds, exits, signals, buckets),
`arrays.npz` (`BT_BETA_GZ`, `BT_EXPO_GZ`, `BT_V_C`, `BT_VOL_C`, every `BT_M_EXIT` markout, `_mmd5`, the 14g tilt inputs,
`_pb_u`), `objects.pkl` (`BT_CURVES`, `BT_SIGMA_C`, the run metadata: months, test months, horizons, exits of record,
labels, stage keys) and `holding.parquet` (the realised par-weighted median holding by tenor x call bucket from
`WF_MATCH`). Every object is taken with `globals().get`, so a lab run that stops before 17 still writes a handoff without
the holding table. See `docs/rfq_variants_notes.md` for what the variant notebook does with it.

## Definitions worth having in one place

- **Full book**: the 14b exposure `e_i = u_i [1_tenor, beta_i]`, nothing hedged. **Hedged book**: the tenor block is
  zero, the fill's tenor DV01 is sold on the fill day, a fill's hedged P&L is `100 (y - y_trade) + M - (MMD move at tenor
  over the holding) - HEDGE_COST_BP`, and every risk-aware policy under the hedged definition also carries the cost in
  its charge. `HEDGE_COST_BP` can be set in the kernel before the cell runs (default 1.0).
- **Marginal risk** `s_i = h (2 e_i' Sigma E_d + e_i' Sigma e_i + v_i) / (2 u_i)` as in 14b; `gamma_k = k / median(s)`
  on the vanilla book in the first month with Sigma, per definition, so `k` stays the median extra charge in bp.
- **Conservative valuation of a new win**: `100 (y - y_trade) + mkt_h` where `mkt_h` is the day's median other-side
  markout over winnable requests; **factor-implied**: `100 (y - y_trade) + BT_MKT + BT_FAC`, the 14c attribution's
  market and factor parts, the common-markout value where the attribution is missing. Common fills and wins given up
  are unchanged, so the three deltas differ only through new wins.
- **Reallocation**: `charge_i = mean manual of the test month + lambda (g_i - mean g of the month)`, `lambda = m x
  sd(manual) / sd(g)` over the earlier months, `m` in {0.5, 1, 2}. The month's level is in-sample by construction and
  is said so in the cell; the question is allocation, not level.
- **Round trip of a buy lot** (17): in points per 100, `sale price - buy price + coupon x holding days / 360`, with
  `sale price - buy price = (mark_buy - buy price) + (mark_sale - mark_buy) + (sale price - mark_sale)`, the buy edge,
  the inventory and the sale edge; all par-weighted over the sales that close the lot. In yield bp,
  `100 (buy yield - sale yield)`, the same sign convention as the Section 14 markout (positive = the bond richened).
- **Request match** (17): a won request and its WFTrade buy are the same trade on the same CUSIP and New York date with
  the same par; exact par is paired by order of appearance, then the nearest par within `RT_PAR_TOL` = 25 percent by
  mutual rank. No timestamp window and no ECN key.

## The synthetic smoke test

No real data was available here. A harness under the session scratch directory
(`lab49/smoke/harness.py`, not committed) fabricates 600 bonds over 190 business days, a model panel with the 20
instruments, six Gamma versions, a factor path, residuals of record, a ten-tenor MMD curve, a 60k-print tape and a
7,000-request `BT` with the column names and dtypes the lab uses, then executes the lab's own Section 14 cells
([1], [6], [7], 14b [9] and [10], 14c [11] and [12], verbatim from the repo notebook) and the new cells 14d to 14g on
top; for Section 17 it fabricates the Section 16 objects (1,052 AM60 buys of which 252 are the won requests at their
logged price and yield, 666 sales at 2 to 40 business days, marks from the synthetic evaluated price) and runs [23] and
[24]. Cells 62 to 65 reach the pipeline,
the pricing service and the tape, so their pure function definitions are executed from source and their frame-building
lines are replicated on the synthetic data; the vanilla outcome check (`BT_VANILLA_PASS`) holds on the synthetic frame.

Result of the last run (v4.9): every new cell runs to the end, every figure and registry section is produced, and the
registry stays JSON-serialisable. Runtimes on the synthetic scale: [13] 2 s, [14] 1 s, [15] 2 s, [16] 16 s (24
sequential replays), [17] 2 s, [18] 1 s, [19] 7 s, [20] 4 s, [21] 2 s, [22] 2 s, [23] 1 s, [24] 12 s (of which the
re-score at the realised horizon is most). The lab's own cells [1] to 14c [12] ran in 33 s on the same data.

On Section 17 the synthetic run checks the mechanics only: the decomposition identity holds to 0 points; all 252 won
requests match an AM60 buy (222 exact par, 30 nearest par) with the buy yield equal to the logged final quote yield on
100 percent of pairs, which is how the harness built them; 62 percent of lots close inside the window; the par-weighted
median holding is 12 business days, so the record horizon moves from 5 to 10; against the realised sale the IPCA peers
mark-to-P estimator has the lowest MAE among the covered comparable-bond estimators and is kept as the estimator of
record; 0 of 17 variants pass at the realised horizon, as none passed at h = 5.

On the synthetic world the bars come out as expected for noise: the request-fields `Gamma' z` correlates 0.89 / 0.95 /
0.95 with the beta of record (fails the 0.9 bar on factor 1 by design of the synthetic instruments), the EWMA
candidates do not beat the expanding window on a stationary synthetic path, the net-b objective fails, no
reallocation passes. None of those numbers mean anything for the desk; they only show the code paths.

## What only the real run will show

- The share of winnable requests with own or `Gamma' z` betas (the target is >= 95 percent) and whether the
  request-fields `z` clears the 0.9 correlation bar on factors 1 and 2. If it fails, the first thing to look at is the
  missing-instrument table: `z_liquidity_20` is always at the panel median (the raw column is not kept in `model`), and
  `trade_size_lag` / `print_freq_lag` fall back to a 60-day print window.
- Whether an EWMA half-life brings the realised over ex-ante ratio into [0.8, 1.25] in September, when the expanding
  window could not.
- Whether `b` net of the common markout stays positive. On won-with-cover fills the raw other-side markout was 14.6 bp
  and 9.9 bp net of the day's common move, so the objective's 2 to 3.5 bp of selection bias may not survive.
- How much of the manual charge is a regime level (the month effect) against request-level information, and whether
  any reallocation keeps 80 percent of wins.
- The replay runtime of [16] at 1.55M requests: 24 replays against the 10 of 14b [10]; if 14b [10] took around 100 s,
  expect 4 to 5 minutes, cached after the first run.
- Section 17 on the real book: the share of won requests that find their AM60 buy on the same day (16d found 59 with a
  60-second window and exact par; the same-day match should find most of them, and the yield-consistency line says
  whether the pairs are the same trade); the par-weighted median holding, which resets the record horizon; which
  estimator tracks the realised sale (MAE and slope), and whether the realised round trip falls with the bp left on the
  table. The re-score at the realised horizon runs `ex_peer_levels` once more at the new h over every request; at h = 10
  expect about twice the 14f [19] peer-level time.
- One item still owed from the real run of 16: the 16d link (60-second window, exact par, 59 links) should be retired
  in favour of the Section 17 match. The 14g tilts are not yet re-scored at the realised horizon.

## v4.9.1: two fixes from the real run of 14f and 14g

- **14f coverage rule.** The pick of the exit of record required coverage of at least 50 percent of winnable requests
  and tested the level's coverage (98 percent on the real run after widening to the bucket) although the markout also
  needs the anchor, which exists on 43 percent of requests. An unconditional 50 percent rule on the markout would reject
  every candidate, the fallback included, since the anchor is common to all of them. The rule now compares the estimators
  on the level's coverage of the *anchored* winnable requests, and the markout coverage of the record exit is printed
  with the pick and recorded (`markout_coverage_winnable` in `lab_exit_of_record`), with a note that every delta in 14f,
  14g and 17 is scored on that sample. The validation table carries three columns: level coverage of winnable, level
  coverage of anchored winnable, markout coverage of winnable. On the real run this changes the printed numbers and
  not the pick.
- **14g guard and other-side columns.** `bt_tg` now carries the markout per fill under the circularity guard (bucket
  mark-to-P, raw and hedged) and under the lab's other-side exit of Section 14, their differences to production, and a
  `guard agrees` flag (same sign of the hedged markout vs production under the record exit and under the guard). The
  verdict line says for how many passing settings the guard agrees. The bar is unchanged.

## Assumptions made without the data

- `BT['our_bid_price']`, `BT['dv01']`, `BT['final_comp_rating']`, `BT['tenor_bucket']`, `BT['call_bucket']`,
  `BT['coupon_bucket']` exist as the v4 cells build them; each is guarded, and a missing one sends that instrument to
  the panel median.
- `track_static` may be `None`; column names tried for a static source are `maturity` / `maturity_date`, `cpn` /
  `coupon`, `nxt_call_dt` / `next_call_date`, `state_code` / `state`, `issuer_industry` / `INDUSTRY`.
- `msrb_raw` carries `quantity` and `tradetime` (the Section 2 load); without it the print instruments sit at the
  median.
- Call-bucket labels parse as in the v4 output (`0-3m` ... `9-12m`, `1y` ... `10y`, `No Call`); tenor buckets as
  `1y` ... `21y+`.
