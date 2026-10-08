# %% [markdown]
# # Muni IPCA v4 — From Fair Mid to Markout: the Factor Model as Risk Layer, the Quote Judged in Dollars
#
# **v4.8.** The quote marked to the next real trade. The evaluation markout of Section 11 marks a would-be fill to the
# closing evaluation, which is the market's systematic view and the surface the factor model is fitted on. The real
# out-of-sample information is the next print. Section 11c builds the print tape from every MSRB print in the universe
# and marks each would-be fill to the next real print (any side, converted to a mid-equivalent by half the dealer round
# trip of its size bin estimated on prior months; inter-dealer prints unadjusted), to the next inter-dealer print, to the
# next opposite-side print (the round trip) and to the next same-side print, and to the last print within 1, 5 and 10
# business days. Every rule is re-ranked on the next-print mark, raw, with the date-demeaned and factor-hedged versions
# as checks, and the trade-based ranking is set against the evaluation-based one. The round trip of Section 11 now
# exits on the full tape rather than on the matched prints, which raises its coverage.
#
# **v4.7.** Three things. (i) Section 8b gains the benchmark it lacked: the equal-weighted universe (long every covered
# bond) in price and in yield space, and for every portfolio its beta to that market, alpha, excess return, up and
# down capture and the share of periods it beat the market; the long leg of each sort is judged against the market,
# not only against its short leg. A one-direction year makes an absolute Sharpe meaningless and a short position look
# like skill; the benchmark separates the two. (ii) Section 12d reads the real RFQ log's columns (request received
# time, logged probability, algo_won, quoted, is_final, request id, the print's own time, the final quote yield) and
# takes the algo mid from the match table when the track carries none. (iii) Runtime: the beta-space embedding is
# cached, the risk snapshot uses only the trailing window it needs, the roll-forward is vectorised.
#
# **v4.6.** The production ledger. The RFQ log (`muni_algo_trade_track`) carries what production actually did on
# every bid-side request: its optimal yield from the optimizer with the production fill curve, the pfill key it
# pooled on and the pfill it assigned, the charges, and the print if the request traded. Section 12d reads it as
# the ground truth the replica of 12c could only approximate: production's concession distribution, the
# calibration of its pooled fill curve by key and by key component, a like-for-like test of the grouping (the
# production key rebuilt, the beta-space cluster in its place, the cluster shifted by the bond's residual deviance
# against its peers, and the gradient-boosted model), all scored at production's own quote on the requests it
# received; then production's quote against the algo mid, S, SQ and the engine on the same requests (win rate,
# cover, P&L), and a production proxy for the universe built from production's own concessions by key. The
# concession grid widens to -8 bp (the replica sat on the -3 bp floor in the v45 run) and the fill models gain
# knots at -8 and -5 bp. Charge detection is numeric-only.
#
# **v4.5.** The objective test of Section 12c is read the way the desk reads it: **win rate by side** (the share of
# bid and of offer prints the quote at $x^*$ would have won), **distance to the print** (how far the quote sat from
# the MSRB level, the cover: on wins the bp given up through the print, on misses the bp short of it), and P&L, for
# the production rule on the original algo yield and for every variant, then the same by trade size, duration,
# rating, call structure, beta-space cluster and industry, side by side. Every code cell now records its runtime;
# the last cell tabulates the cells against the previous run and names the stages worth caching next.
#
# **v4.4.** Two additions and one removal. (i) The production objective is rebuilt inside the notebook: the desk
# chooses the concession $x$ on a quote to maximise $p_{\text{fill}}(x)\,(x + \text{charge})$, with the fill curve the
# trailing empirical CDF of the basis on the quote's own side and the charge the liquidity, risk and manual charges
# booked on the trade. Section 12c replicates it on the matched prints, then swaps each of its three parts for what
# this programme built (the same-side memory for the algo mid, the conditioned fill model for the pooled curve, the
# expected round-trip P&L for the booked spread) and judges every variant on realised dollars against the production
# rule. (ii) Section 8b asks the Kelly-Palhares-Pruitt question of the factor model: do the exposures, times the
# factor means estimated at each refit, rank bonds by subsequent total return? Expected-return quintile long-shorts
# at monthly and weekly rebalancing, and the model-implied tangency portfolio of the factors, on a total-return
# target from evaluated prices rather than on daily residuals. (iii) Section 14 (the roll-forward of stale marks)
# is folded into Section 8 as one table: it is a marking property of the risk layer, not a section.
#
# **v4.3.** The desk's fill-probability curve, a trailing 100-day empirical CDF of the basis between the quote and the
# print by side, is built inside the notebook and tested for what conditioning adds (size, beta-space cluster, then a
# gradient-boosted model with and without IPCA and state-space features). The level model's target moves from fair
# price to the two quantities a quote engine consumes: fill probability at a concession and the expected edge of a
# fill. Those two feed a reduced-form engine (concession per print = argmax of pfill x (edge + concession)), judged
# against S at zero on both markout metrics with the usual bar. The beta-space map gains trade size, side mix, print
# frequency and industry, and Section 13 colours it by markout. The round trip uses the raw exit where the factor
# path is missing, since exits are mostly within a day.
#
# **v4.2.** The v4.1 run answered the trend question (the hedge removed it) and left two open ones. First, the edge is
# measured against evaluations, which in munis sit near the bid, so the side split of the P&L may be a convention: a
# realised round trip (exit at the next opposite-side print in the same bond) is added as the evaluation-free check.
# Second, the bid concession optimum sat on the edge of the grid, so the grid runs to 25 bp. Two residual-side tests
# close the "thoughts 3" questions with pre-registered bars: the state-space forecast as a predictor of hedged
# markout and as a concession modifier, and the mark-noise estimate as a width scalar on the quantile grid.
#
# **v4.1.** The v4.0 run showed the raw mark-to-evaluation markout is dominated by the sample's common yield move
# (the mark moved 8 bp against bid fills and 6 bp against bid misses over five days, and 7 bp in favour of offers
# whether filled or not). v4.1 hedges it with the factor model: the mark move after a fill is replaced by the
# cumulative point-in-time residual of record, so the factor-implied part of the move is removed and the P&L is
# market-neutral, which is the P&L a market maker is paid for. The side x size intercepts use medians (the mean
# over-shifted the offer by 5 bp in v4.0), and the characteristic breakdowns are now of markout P&L, not of MAE.
#
# v4.0 moves the programme past the fair-mid question. Six versions of mid work moved the quote by 1.35 bp on a
# 13 bp error against the print, and the v51 run closed the last fair-mid idea (the interaction ladder: the
# persistence of the algo's error does not depend on the factor state). A market maker is not paid for a good mid.
# It is paid for expected P&L per quote: fill probability times edge, net of adverse selection and inventory cost.
# This version keeps the factor spine as the risk and marking layer, trims the fair-mid exhibits to what feeds the
# quote, and judges every quote rule in the currency the desk uses.
#
# 1. **Markout framework (Section 11).** For every matched print we know our quote, the print, the side and the
#    size. A print that crossed our quote is a fill we would have won; marking that fill to the evaluation one,
#    five and ten business days later gives its P&L, split into the edge at the quote and the mark move after the
#    trade. Shifting the quote by a concession traces the fill curve and the expected P&L per quote, which is a
#    one-dimensional version of the optimizer. Every mid from v3 is re-ranked under it, plus the trials v3.6
#    closed on MAE (side intercept, revised third term, factor-beta correction, age-conditioned last error) and two
#    new rules built from the v51 finding that the algo does not price size (a side x size intercept on the algo
#    quote and on the same-side memory).
# 2. **Quantile grid (Section 12).** The deliverable to the optimizer is a conditional distribution, not a number:
#    quantiles of the oriented error by side x size x beta-space cluster, shrunk toward the side x size marginal,
#    estimated on prior months. It encodes the fill curve (one minus the CDF) and the bias (the median) in one object.
# 3. **Risk model snapshot (Section 8).** Betas, factor covariance, idiosyncratic and mark-noise variance per bond,
#    exported as the inventory risk input. The factor spine (IPCA, walk-forward residuals, state-space filter) is
#    unchanged and still cached.
#
# Everything is point in time and the leakage boundaries are the v3 ones. The markout uses evaluations after the
# trade deliberately: it is an evaluation metric, not a feature. Its two caveats are stated where it is built:
# the fill proxy is optimistic (every print that crossed our quote counts as a fill, with the winner's curse that
# implies), and marks are evaluations, not executable prices. Both apply equally to every candidate, so the
# ranking is what to read, not the level.
#
# **Inputs** (from `data_pipeline/muni_data_pipeline.py`, under `./data_pipeline/`): `research_panel_step3_yield.parquet`,
# `data/closing_marks/*.parquet`, `data/algosignal_msrb/*.parquet`, `data/msrb/*.parquet`, `data/track_static/*.parquet`.
# **Outputs** go to `./artifacts_v3/`; the stage cache under `artifacts_v3/cache/` is reused across runs.
#
# **Sections**
# 1. Setup and configuration (with the pre-registered choices)
# 2. Data load and QA; the two data rules; trade size and industry characteristics
# 3. Model panel: instruments, rank-normalised per date
# 4. IPCA: date-weighted ALS, K sweep, Gamma anatomy
# 5. Walk-forward residuals: refit cadence, residuals of record
# 6. Factor-mimicking weights, leverage, beta-space map and clusters
# 7. Residual dynamics and the state-space filter (the mark-noise score)
# 8. Risk model snapshot: betas, factor covariance, idiosyncratic and mark-noise variance per bond; the factor roll-forward of stale marks; 8b factor premia as expected returns (the Kelly-Palhares-Pruitt test, monthly and weekly)
# 9. Transaction join (PIT-safe) and mark quality; 9b trade size and side; 9c where prints sit relative to our quote and to the evaluation
# 10. Quote rules: the same-side memory, the interaction ladder, the re-opened legacy trials, the side x size intercepts, the level model; the direction-aware view
# 11. Markout: would-have-filled P&L, edge and adverse selection, fill and P&L curves, the re-ranking in dollars, the cell-optimal concession, the realised round trip; 11b fill probability (the desk's curve, conditioned, modelled), the edge model, the quote engine in reduced form
# 12. Quantile grid: the conditional distribution of the oriented error for the optimizer, with its calibration; 12b the residual side: forecast as concession modifier, mark noise as width scalar; 12c the production objective pfill(x) x (x + charge), rebuilt and tested piece by piece against expected P&L; win rate, cover distance and P&L by side and by characteristic; 12d the production ledger: the RFQs we received, production's quote, pfill and key; the grouping test; the production proxy for the universe
# 13. Where the P&L lives: markout by characteristic, factor beta, cluster, side and size; the side heatmaps; the cells that carry the dollars
# 14. Results registry and summary
# 15. Robustness
#
# **Dropped in v4.4**: the stand-alone roll-forward section (its table now closes Section 8; the by-segment version
# added nothing the cluster risk table does not say).
# **Dropped in v4.1**: the MAE breakdowns by characteristic and the factor-beta loading of the pricing error (the
# factor model is the risk and marking layer, not a mid corrector, so the breakdown that matters is of P&L).
# **Dropped in v4.0** (fair-mid exhibits that no longer feed the quote): the hedged paper portfolio, the conformal
# band (replaced by the quantile grid), the achievement map, the MAE-dollar view (replaced by markout dollars), the
# per-side top-cell charts and the within-cluster signal Sharpe. Their last results are in the v3.8 paper. The
# stage cache is unchanged: walk-forward IPCA, the pooled state-space fit, the standardised print history, the level
# models and the per-side state-space memory are reused when their inputs and code are unchanged.

# %%
from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.options.display.float_format = '{:.4f}'.format

# ---- per-cell runtime log (v4.5). Every code cell starts with CELL_T('label'); the last cell tabulates the runtimes, compares
# them with the previous run (artifacts_v3/cell_runtimes.csv) and lists the cached stages each cell hit or recomputed.
_RUN_T0 = time.perf_counter(); CELL_LOG: list[dict] = []; _CELL_STATE = {'t0': _RUN_T0, 'label': '1. Setup and configuration [1]'}


def CELL_T(label: str) -> None:
    now = time.perf_counter(); CELL_LOG.append({'cell': _CELL_STATE['label'], 'seconds': now - _CELL_STATE['t0']}); _CELL_STATE['t0'] = now; _CELL_STATE['label'] = label


import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (10, 4.5), 'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})

try:
    display  # noqa: B018  (IPython provides it)
except NameError:  # plain python execution
    def display(obj):  # type: ignore
        print(obj.to_string() if hasattr(obj, 'to_string') else obj)

ROOT = Path('.').resolve()
PIPELINE_ROOT = ROOT / 'data_pipeline'
ARTIFACTS = ROOT / 'artifacts_v3'
FIGURES = ARTIFACTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PIPELINE_ROOT))
import muni_data_pipeline as mdp  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    spec_version: str = 'ipca_v3_k3'
    start_date: str = '2026-01-01'
    end_date: str = '2026-09-30'
    train_start: str = '2026-01-02'
    first_oos_date: str = '2026-04-01'
    refit_days: int = 7
    selected_k: int = 3
    candidate_k: tuple[int, ...] = (1, 2, 3, 4, 5)
    max_iter: int = 50
    tol: float = 1e-5
    ridge: float = 1e-10
    min_obs_per_bond: int = 5
    max_lag_days: int = 7
    target_clip_bp: float = 250.0
    yield_mad_k: float = 6.0          # extreme-yield rule: |y - median_date| > k * MAD_date
    ar_winsor: tuple[float, float] = (0.01, 0.99)
    longconv_lags: int = 10
    residual_min_age_days: int = 1
    min_bonds_per_date: int = 1000     # drop sparse dates (holiday / partial-mark days collapse the cross-section)
    zero_duration_cut: float = 0.25    # modified duration below this with > 1y to worst is treated as a data error
    activity_window: int = 20          # trailing observations for the point-in-time activity bucket
    duration_instrument: str = 'ratio'  # 'ratio': modified duration / years-to-worst (call-structure); 'level': modified duration
    level_signal_window: int = 20      # trailing-mean residual window (the level signal)
    band_coverage: float = 0.80
    # --- data rules (Section 2) ---
    date_weighting: str = 'inverse_var'   # each date's ALS moments weighted by 1 / its cross-sectional mean square of the target, capped
    date_weight_cap: float = 4.0
    dispersion_bp: float = 10.0           # a dispersion day: more than dispersion_share_cut of bonds move more than dispersion_bp ...
    dispersion_share_cut: float = 0.10    # ... RELATIVE TO THE DAY'S MEDIAN MOVE (an evaluator-wide reprice, not a curve move)
    common_move_bp: float = 10.0          # a common-move day: |median move| above this; flagged, kept, the intercept factor absorbs it
    exclude_dispersion_from_fit: bool = True
    partial_day_ratio: float = 0.6        # a partial-mark day: bonds priced below this share of the trailing-20-date median; dropped
    # --- refit cadence (Section 5) ---
    cadences: tuple[str, ...] = ('monthly', 'quarterly', 'frozen')   # weekly and regime-conditional closed in v31 (within 0.3 pp)
    record_cadence: str = 'monthly'       # residuals of record
    regime_cadence_days: int = 30
    # --- residual signal (Sections 7-8) ---
    portfolio_scaling: str = 'vol'
    block_days: int = 5
    n_boot: int = 500
    ssm_fit_bonds: int = 2000
    ssm_maxiter: int = 120
    record_signal: str = 'pooled'         # 'pooled' | 'bucket': which state-space filter is the signal of record (v31: pooled wins)
    # --- algo error correction and band (Sections 10-11) ---
    err_halflife_prints: float = 3.0      # EWMA of past algo errors, in prints
    age_bins_days: tuple[float, ...] = (0, 1, 3, 7, 21, 1e9)
    level_model_trees: int = 600
    conformal_calibration_months: int = 1
    conformal_window_days: int = 10       # rolling conformal: trading days of realised scores used to set the scale
    n_clusters: int = 8                   # beta-space clusters for the breakdown (fitted on the first OOS month, applied forward)
    n_min_cell: int = 2000                # minimum trades in a characteristic cell for the breakdown tables
    # --- v3.9: new characteristics and the interaction ladder ---
    trade_size_halflife_days: float = 30.0   # time-decayed mean log trade size per bond from MSRB prints strictly before the date
    industry_dummies: int = 5                # largest issuer industries as IPCA dummies; OTHER and UNKNOWN omitted
    interaction_bar_bp: float = 0.10         # a variant replaces S only if it adds this over S with bootstrap t > 3 and gains in >= 4 of 5 months
    asym_tau: float = 0.65                   # weight on the aggressive side of the error in the asymmetric loss (P and S; D symmetric)
    # v4.0 markout and quantile grid
    markout_horizons: tuple[int, ...] = (0, 1, 5, 10)   # business days after the trade at which a would-be fill is marked (0 = the trade date's close)
    markout_record_h: int = 5                            # the marking horizon of record for the re-ranking and the concession choice
    concession_grid_bp: tuple[float, float, float] = (-10.0, 26.0, 1.0)   # concession grid (start, stop, step) in bp of yield; negative = more aggressive than the mid (v4.2: to 25 bp, the v4.1 bid optimum sat on the 14 bp edge)
    rt_max_days: int = 10                                # v4.2 realised round trip: exit at the next opposite-side print in the same bond within this many calendar days
    trade_mark_horizons: tuple[int, ...] = (1, 5, 10)    # v4.8: business-day windows for the last-print-within mark, beside the next-print mark
    trade_mark_min_pairs: int = 200                      # v4.8: bond-day P/S pairs needed on prior months to estimate the half spread of a size bin
    signal_kappas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5)   # v4.2 residual-signal concession: quote shift = -kappa x oriented forecast (bp), kappa chosen on prior months
    signal_clip_bp: float = 5.0                          # the signal concession is clipped to +/- this many bp
    width_scalar_bar: float = 0.01                       # the mark-noise width scalar replaces the pooled grid if the tail calibration error falls by this much on both sides
    # v4.3 fill probability and the reduced-form quote engine
    pfill_window_days: int = 100                         # the desk's curve: trailing window of the empirical CDF of the basis, by side
    pfill_deltas: tuple[float, ...] = (-8.0, -5.0, -3.0, 0.0, 3.0, 6.0, 10.0)   # concessions at which fill probability is modelled and scored (v4.6: knots at -8 and -5 so the objective grid can reach production's range)
    pfill_trees: int = 150                               # gradient-boosted fill and edge models per fold
    pfill_max_train: int = 200_000                       # training rows per fold are subsampled to this many for speed
    pfill_min_train: int = 5_000                         # a fold needs this many training prints for the fill and edge models
    # v4.4 factor premia (Section 8b) and the production objective (Section 12c)
    premia_freqs: tuple[str, ...] = ('monthly', 'weekly')   # rebalancing frequencies of the expected-return long-short
    premia_max_abs_ret_bp: float = 3000.0                # a holding-period return beyond this (30% of price) is a data error and is dropped
    charge_units: str = 'auto'                           # units of the charge columns: 'bp' (of yield), 'pct' (percent of yield, x100), 'price' (price points, converted with duration), 'auto' (detected, printed)
    charge_max_age_days: float = 1.0                     # a per-trade charge joins a print as of the print's time within this many days, else the bond's last charge, else the side median
    objective_grid_bp: tuple[float, float, float] = (-8.0, 11.0, 1.0)   # concessions the production objective chooses from (start, stop, step); fill models are interpolated onto it (v4.6: floor -8, the replica sat on -3 in the v45 run)
    # v4.6 production ledger
    rfq_join_minutes: float = 60.0                       # an RFQ joins its print by trade id, else by cusip x side within this many minutes after the request and the same quantity
    grouping_bar: float = 0.05                           # a fill-curve grouping replaces the production key if it removes this share of the logged pfill's Brier at production's quote, in >= 4 held-out months
    proxy_min_rfqs: int = 200                            # a pfill key needs this many requests for its own median concession in the production proxy, else the side x size median
    cell_min_trades: int = 500                           # a side x size cell intercept / concession needs this many training trades, else the side value
    grid_taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    grid_shrink_k: float = 500.0                         # a cell's quantiles shrink toward the side x size marginal with weight n / (n + k)
    seed: int = 20260921


# Pre-registered choices, fixed on 2026-10-07 after the v5 review and before this notebook was run. The robustness
# section prints them so a reader can tell a choice from a fit.
PRE_REGISTERED = {
    'fixed_on': '2026-10-07 (v3.1, after the v31 review)',
    'selected_k': 3, 'record_cadence': 'monthly', 'level_signal_window': 20, 'activity_window': 20,
    'record_signal': 'pooled state-space filter (the by-bucket filter closed in v31)', 'ewma_half_life_prints': 3,
    'deployable_rule': 'same-side memory S: algo + rho_s x EWMA(past algo errors of the bond on the same side), rho_s on prior months (v41)',
    'state_space_memory': 'per (bond, side) local level with daily decay phi and diffusion q, observation noise r; ML on prints before the test month',
    'conformal_rolling_window_days': 10, 'band_benchmark': 'fixed width rescaled on the same rolling window', 'dollar_metric': '(|algo error| - |mid error|) x par x price/100 x modified duration x 1e-4, summed per month', 'breakdown_axes': 'duration, call structure, rating, state, liquidity, factor-beta terciles, 8 beta-space clusters fitted on the first OOS month',
    'dispersion_rule': 'share of |target - median_t| > 10 bp above 10% of bonds -> zero weight in the fit',
    'common_move_rule': '|median_t| > 10 bp -> flagged and kept', 'partial_day_rule': 'bonds priced < 60% of trailing-20-date median -> dropped',
    'date_weighting': 'inverse variance, cap 4x', 'duration_instrument': 'ratio',

    'portfolio_scaling': 'vol', 'band_coverage': 0.80, 'conformal': 'split conformal, calibration = the month before the test month, normalised score |err| / q_hat',
    'error_correction_reference': 'algo + rho x side-pooled EWMA of past algo errors (C), kept as the reference the same-side memory is measured against',
    'housekeeping_v38': 'closed mids removed from the run; stage cache keyed on config, data fingerprints and code',
    'v39_new_instruments': 'trade size (decayed mean log MSRB quantity, half-life 30 days, prints strictly before the date), decayed print frequency, a no-print flag, and the five largest issuer industries as dummies',
    'v39_qty_bins': 'desk bins: 0-5K, 5K-10K, 10K-15K, 15K-25K, 25K-50K, 50K-100K, 100K-250K, >250K',
    'v39_interaction_ladder': 'rho of the same-side EWMA conditioned on factor betas, beta-space cluster, |filtered residual drift|, same-side age, trade size, side x cluster, industry; all on prior months',
    'v39_bar': 'a variant replaces S only if gain - gain(S) >= 0.10 bp, bootstrap t of the daily difference > 3, and the difference is positive in >= 4 of 5 held-out months',
    'v40_markout': 'would-have-filled P&L: a print that crossed the quote is a fill at the quote, marked to the closing evaluation h business days later; P&L = s x (print - mark_h) - oriented distance + concession; record h = 5',
    'v40_fill_proxy': 'fill iff the oriented distance print - quote (P) or quote - print (S) >= the concession; inter-dealer prints excluded from P&L; optimistic by construction (winner\'s curse)',
    'v40_rerank_bar': 'a rule replaces S only if its dollars per month at h = 5 exceed S with bootstrap t of the daily difference > 3 and a positive difference in >= 4 of 5 months',
    'v40_size_intercept': 'side x quantity-bin FM mean of the error on prior months, shrunk to the side mean with weight n / (n + 500); AQ on the algo quote, SQ on the error the same-side memory leaves',
    'v41_hedged_markout': 'the mark move after a fill is the cumulative point-in-time residual of record between the trade date and the horizon (factor-implied move removed); raw and date-demeaned moves reported as checks',
    'v43_pfill': 'fill probability P(o >= delta | x) for the S quote at delta in {-3, 0, 3, 6, 10} bp; four estimators scored by Brier and log-loss on the next month: the desk curve (side, trailing 100-day empirical CDF), a side x size x cluster cell table on prior months (shrunk), a gradient-boosted classifier on trade features, and the same with IPCA and state-space features',
    'v43_edge_model': 'expected round-trip edge of a fill at delta 0, gradient-boosted on the same two feature sets, walk-forward; scored by out-of-sample R2 and realised edge by predicted decile',
    'v43_engine_bar': 'the reduced-form engine (concession per print = argmax over the delta grid of pfill(delta | x) x (expected edge + delta)) replaces S at 0 only if its round-trip P&L per print is higher by 0.10 bp with bootstrap t > 3 and positive in 4 of 5 months',
    'v48_trade_markout': 'a would-be fill is marked to the next real MSRB print in the bond within 10 calendar days: any side, converted to a mid-equivalent by half the dealer round trip of the mark\'s size bin (median P - S spread of same bond-day prints on prior months; inter-dealer prints unadjusted), and separately to the next inter-dealer, opposite-side and same-side print; and to the last print within 1, 5, 10 business days; the trade-based ranking of record is the next-print mark, raw; date-demeaned and factor-hedged shown as checks; the bar is unchanged',
    'v47_benchmark': 'the market is the equal-weighted covered universe held long over the same period (price: total return; yield space: minus the mean daily yield change); every portfolio reports beta and alpha on it, excess return, up and down capture and the share of periods above it; a portfolio that fell less with beta below one is defensive, with beta near one and positive alpha is selection, with beta below zero is a short',
    'v46_rfq_ledger': 'the RFQ log is the ground truth for the bid: production optimal yield, pfill key, pfill, charges, print; a request is scored only when it printed and joined to the matched-print frame (trade id, else cusip x side x time within rfq_join_minutes and the same quantity); win = the print crossed the quote (the logged won flag replaces it where present); cover = oriented distance of the print from the quote',
    'v46_grouping_bar': 'a fill-curve grouping replaces the production key only if, at production\'s own quote on the requests it received, it removes >= 5% of the logged pfill\'s Brier in >= 4 of the held-out months; candidates: the production-style key rebuilt (coupon bin x rating group x call group x size), the beta-space cluster in its place, the cluster shifted by the bond\'s oriented residual deviance against its cluster peers, the gradient-boosted model',
    'v46_production_proxy': 'for the universe, production is represented by the algo mid plus the median production concession of the bond\'s pfill key (key prefix from the bond\'s own requests, size token from its quantity), else the side x size median; validated on the overlap against production\'s logged quote before use',
    'v45_win_rate_cover': 'win rate = share of customer prints on a side that crossed the quote at x*; cover distance = |print - quote at x*| in bp of yield, split into bp given up through the print on wins and bp short of the print on misses; both on the same prints for every variant, with the daily difference from production bootstrapped',
    'v44_production_objective': 'concession per print x* = argmax over the grid of pfill_side(x) x (x + total charge), with pfill the trailing 100-day empirical CDF of the basis on the same side and the charge the liquidity + risk + manual charges of the trade (sign and units detected and printed); fill iff the print crossed the quote shifted by x*',
    'v44_objective_variants': 'the three parts of the objective swapped one at a time and together: the mid (algo quote -> same-side memory S), the fill curve (desk curve -> quantile grid -> gradient-boosted model with IPCA and state-space features), the value of a fill (x + charge -> expected round-trip P&L from the edge model with the selection adjustment)',
    'v44_objective_bar': 'a variant replaces the production rule only if its realised round-trip P&L per print is higher by 0.10 bp, bootstrap t of the daily difference > 3, positive in >= 4 of the held-out months',
    'v44_premia_signal': 'expected total return over the holding period = carry - modified duration x (beta . lambda) x business days, with lambda the mean daily factor realisation under the Gamma in force (in-sample on its training dates, nothing after the formation date); sorts: factor premium alone, carry alone, both, premium within duration quintile, the premium in yield space, the state-space residual signal as the reference',
    'v44_premia_portfolios': 'quintile long-short (top minus bottom), equal-weighted and DV01-balanced, rebalanced on the first residual date of each month and of each week, total return from evaluated prices with accrued coupon; the tangency portfolio of the factors with mean and covariance from the same training dates, applied to the out-of-sample factor realisations',
    'v44_premia_reading': 'with five to six monthly and about twenty-five weekly rebalances the test can only reject a large premium: the reading is weekly long-short t > 2 with a monotone quintile pattern and the same sign at monthly frequency; a Sharpe is reported, never relied on',
    'v42_round_trip': 'realised round trip: a would-be fill exits at the next opposite-side print in the same bond within 10 days, at that print\'s yield, factor-hedged over the holding period; the evaluation-independent check of the side split',
    'v42_signal_test': 'the state-space forecast, oriented by side and ranked within date x beta cluster, is tested as a predictor of the hedged markout and as a concession modifier (shift = -kappa x forecast, kappa on prior months); bar = +0.10 bp per print over S with its side concession, bootstrap t > 3, 4 of 5 months',
    'v42_width_scalar': 'quantile grid split by mark-noise tercile replaces the pooled grid if the mean absolute tail calibration error (tau 0.05, 0.10, 0.90, 0.95) falls by >= 0.01 on both sides',
    'v41_median_intercept': 'side x quantity-bin intercepts are medians of the error on prior months, shrunk to the side median with weight n / (n + 500)',
    'v40_quantile_grid': 'quantiles 5..95 of the oriented error of S by side x quantity bin x beta-space cluster, shrunk to the side x size marginal with weight n / (n + 500), on prior held-out months',
    'v39_direction': 'signed decomposition (toward / crossed by less / crossed by more / away), crossing rate by side, asymmetric pinball loss with tau 0.65 on the aggressive side',
}


CFG = RunConfig()
PIPE = mdp.Config(root=PIPELINE_ROOT, start_date=CFG.start_date, end_date=CFG.end_date)
RNG = np.random.default_rng(CFG.seed)
REGISTRY: dict = {'spec_version': CFG.spec_version, 'config': asdict(CFG), 'run_at': pd.Timestamp.now('UTC').isoformat()}


def to_ns(series: pd.Series) -> pd.Series:
    """Coerce any datetime-like column to tz-naive datetime64[ns] so merge keys always match."""
    out = pd.to_datetime(series, errors='coerce')
    try:
        if getattr(out.dt, 'tz', None) is not None:
            out = out.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return out.astype('datetime64[ns]')


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', dpi=130)
    plt.show()
    plt.close()


def record(section: str, **values) -> None:
    REGISTRY.setdefault(section, {}).update({k: (v.item() if hasattr(v, 'item') else v) for k, v in values.items()})


def flat_records(frame: pd.DataFrame, digits: int = 4) -> list[dict]:
    """to_dict(orient='records') with two-level column headers flattened to 'a | b' so the registry stays JSON-serialisable."""
    f = frame.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = [' | '.join(str(x) for x in c if str(x) != '') for c in f.columns]
    return f.round(digits).reset_index().to_dict(orient='records')


# ---------------------------------------------------------------------------------------------------------------
# Stage cache. A stage is a zero-argument function. Its key is the SHA-1 of the run configuration, the data
# fingerprints and upstream stage keys it declares as `deps`, and the source code of the function plus any helper
# functions listed in `code`. Same key -> the stored result is reused; any change to config, data or code -> the
# stage is recomputed and the previous file for that stage is removed. Nothing is switched by hand.
# ---------------------------------------------------------------------------------------------------------------
CACHE_DIR = ARTIFACTS / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STAGE_KEYS: dict[str, str] = {}
CACHE_LOG: list[dict] = []


def _stage_source(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:          # a value rather than a function: hash its repr
        return repr(obj)


def stage_key(name: str, fn, deps=(), code=()) -> str:
    payload = {'stage': name, 'spec': CFG.spec_version, 'cfg': asdict(CFG), 'deps': [str(d) for d in deps], 'code': [_stage_source(fn)] + [_stage_source(c) for c in code]}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def cached(name: str, fn, deps=(), code=()):
    key = stage_key(name, fn, deps, code); STAGE_KEYS[name] = key
    path_df, path_pk = CACHE_DIR / f'{name}_{key[:12]}.parquet', CACHE_DIR / f'{name}_{key[:12]}.pkl'
    t0 = time.perf_counter()
    if path_df.exists() or path_pk.exists():
        obj = pd.read_parquet(path_df) if path_df.exists() else pd.read_pickle(path_pk)
        p = path_df if path_df.exists() else path_pk
        print(f'[cache] {name}: reused {p.name} ({p.stat().st_size / 1e6:.0f} MB, loaded in {time.perf_counter() - t0:.1f}s)')
        CACHE_LOG.append({'stage': name, 'status': 'reused', 'seconds': time.perf_counter() - t0, 'file': p.name, 'cell': _CELL_STATE['label']}); return obj
    obj = fn(); dt = time.perf_counter() - t0
    for old in CACHE_DIR.glob(f'{name}_*'):
        old.unlink()
    if isinstance(obj, pd.DataFrame):
        try:
            obj.to_parquet(path_df, index=False); p = path_df
        except Exception as exc:        # mixed-type object columns: fall back to pickle
            print(f'[cache] {name}: parquet failed ({type(exc).__name__}), storing as pickle'); pd.to_pickle(obj, path_pk); p = path_pk
    else:
        pd.to_pickle(obj, path_pk); p = path_pk
    print(f'[cache] {name}: computed in {dt:.0f}s, saved {p.name} ({p.stat().st_size / 1e6:.0f} MB)')
    CACHE_LOG.append({'stage': name, 'status': 'computed', 'seconds': dt, 'file': p.name, 'cell': _CELL_STATE['label']}); return obj


def frame_fingerprint(frame: pd.DataFrame, cols: list[str]) -> str:
    """A cheap, deterministic fingerprint of a frame: shape plus sums and ranges of a few columns."""
    parts = [f'{len(frame)}x{frame.shape[1]}']
    for c in cols:
        if c in frame.columns:
            v = pd.to_numeric(frame[c], errors='coerce') if not np.issubdtype(frame[c].dtype, np.datetime64) else frame[c].astype('int64', errors='ignore')
            try:
                parts.append(f'{c}:{float(np.nansum(v.to_numpy(float))):.6g}:{v.min()}:{v.max()}')
            except Exception:
                parts.append(f'{c}:{frame[c].astype(str).str.len().sum()}')
    return '|'.join(parts)


# Desk quantity bins (upper edges in par) and their labels
QTY_EDGES = [5_000, 10_000, 15_000, 25_000, 50_000, 100_000, 250_000]
QTY_LABELS = ['0-5K', '5K-10K', '10K-15K', '15K-25K', '25K-50K', '50K-100K', '100K-250K', '>250K']


def qty_group(q: pd.Series) -> pd.Series:
    """Desk quantity bins: 0-5K, 5K-10K, 10K-15K, 15K-25K, 25K-50K, 50K-100K, 100K-250K, >250K."""
    v = pd.to_numeric(q, errors='coerce')
    return pd.cut(v, bins=[-np.inf] + QTY_EDGES + [np.inf], labels=QTY_LABELS, right=True).astype(str).replace('nan', 'NA')


def group_decayed_sums(frame: pd.DataFrame, keys: list[str], cols: list[str], halflife_days: float, date_col: str = 'trade_date') -> pd.DataFrame:
    """Per key group, running sums of cols decayed by 0.5^(gap_days / halflife) between the group's successive dates
    (value up to and including each date). Vectorised: one numpy step per observation index."""
    f = frame.sort_values(keys + [date_col], kind='stable').reset_index(drop=True)
    lam = 0.5 ** (1.0 / halflife_days)
    gid = f.groupby(keys, observed=True, sort=False).ngroup().to_numpy(); ng = int(gid.max()) + 1 if len(f) else 0
    step = f.groupby(keys, observed=True, sort=False).cumcount().to_numpy()
    gap = f.groupby(keys, observed=True, sort=False)[date_col].diff().dt.days.fillna(0).to_numpy(float)
    order = np.argsort(step, kind='stable'); ks = step[order]; st = np.searchsorted(ks, np.arange(ks.max() + 2)) if len(f) else np.array([0, 0])
    acc = {c: np.zeros(ng) for c in cols}; out = {c: np.empty(len(f)) for c in cols}; vals = {c: f[c].to_numpy(float) for c in cols}
    for k in range(len(st) - 1):
        rows = order[st[k]:st[k + 1]]; g = gid[rows]; w = lam ** gap[rows]
        for c in cols:
            acc[c][g] = acc[c][g] * w + vals[c][rows]; out[c][rows] = acc[c][g]
    for c in cols:
        f[f'{c}_ewm'] = out[c]
    return f


def group_ewm_by_order(values: np.ndarray, group_ids: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted mean within each group, in the row order given (rows must be sorted by group then
    time), equal to pandas' groupby(...).ewm(halflife, adjust=True, min_periods=1).mean() but vectorised across
    groups: one numpy step per observation index instead of one Python call per group."""
    values = np.asarray(values, float); group_ids = np.asarray(group_ids)
    n = len(values); out = np.full(n, np.nan)
    if n == 0:
        return out
    w = 0.5 ** (1.0 / halflife)
    # position of each row within its group (rows sorted by group)
    start = np.r_[True, group_ids[1:] != group_ids[:-1]]
    grp_start_idx = np.flatnonzero(start); grp_no = np.cumsum(start) - 1
    pos = np.arange(n) - grp_start_idx[grp_no]
    ng = len(grp_start_idx); num = np.zeros(ng); den = np.zeros(ng)
    order = np.argsort(pos, kind='stable'); ks = pos[order]; st = np.searchsorted(ks, np.arange(ks.max() + 2))
    for k in range(len(st) - 1):
        rows = order[st[k]:st[k + 1]]
        if not len(rows):
            continue
        g = grp_no[rows]; x = values[rows]; ok = np.isfinite(x)
        num[g] = num[g] * w + np.where(ok, x, 0.0); den[g] = den[g] * w + ok.astype(float)
        out[rows] = np.where(den[g] > 0, num[g] / np.where(den[g] > 0, den[g], 1.0), np.nan)
    return out


print('ROOT', ROOT)
print('Pipeline manifest:', (PIPELINE_ROOT / 'data_pipeline_manifest.json').exists())
print('Spec', CFG.spec_version, '| window', CFG.start_date, '->', CFG.end_date, '| first OOS', CFG.first_oos_date)

# %% [markdown]
# ## 2. Data load and QA
#
# The Step 3 panel is the covered yield-space modelling input written by the pipeline. We also read the
# deduplicated closing marks (for coverage) and the matched trades (for validation). Two data findings from the
# regenerated data are handled here:
#
# * `final_comp_rating` is blank for about 22% of rows while `comp_rating` is blank for about 5%. The rating
#   fallback below uses `final_comp_rating`, then `comp_rating`, then `NR`. It also treats pandas `<NA>` as blank,
#   which the previous builder did not.
# * Closing-mark yields contain a small number of extreme values. Rows whose yield sits more than `yield_mad_k`
#   median absolute deviations from that date's cross-sectional median are flagged and excluded from the fit;
#   the target is additionally clipped at ±`target_clip_bp`.

# %%
CELL_T('2. Data load and QA [1]')
t0 = time.perf_counter()
step3_path = PIPE.path('research_panel_step3_yield.parquet')
if not step3_path.exists():
    raise FileNotFoundError(f'{step3_path} missing: run mdp.build(PIPE) in the pipeline kernel first.')
raw_panel = pd.read_parquet(step3_path)
raw_panel['date'] = to_ns(raw_panel['date']).dt.normalize()
raw_panel['cusip'] = raw_panel['cusip'].astype('string')
raw_panel = raw_panel[(raw_panel['date'] >= CFG.start_date) & (raw_panel['date'] <= CFG.end_date)].reset_index(drop=True)
print(f'Step 3 panel: {len(raw_panel):,} rows | {raw_panel["cusip"].nunique():,} CUSIPs | {raw_panel["date"].min().date()} -> {raw_panel["date"].max().date()} | {time.perf_counter()-t0:.1f}s')

marks_panel, marks_audit = mdp.load_closing_marks(PIPE, CFG.start_date, CFG.end_date)
print('Closing marks (deduped):', f'{len(marks_panel):,} rows,', f'{marks_panel["CUSIP"].nunique():,} CUSIPs')

algosignal_store = PIPE.store('algosignal_msrb')
trades_raw = algosignal_store.read(categorical=False)
HAS_TRADES = trades_raw is not None and not trades_raw.empty
print('Matched trades available:', HAS_TRADES, '' if not HAS_TRADES else f'({len(trades_raw):,} rows)')
STEP3_FP = f'{step3_path.name}:{step3_path.stat().st_size}:{int(step3_path.stat().st_mtime)}'
TRADES_FP = frame_fingerprint(trades_raw, ['msrb_yield', 'algo_signal_yield', 'msrb_quantity']) if HAS_TRADES else 'none'
print('data fingerprints:', STEP3_FP, '|', TRADES_FP[:80])

# --- coverage by month ---
cov = (raw_panel.assign(month=raw_panel['date'].dt.to_period('M').astype(str))
       .groupby('month').agg(rows=('cusip', 'size'), cusips=('cusip', 'nunique'), dates=('date', 'nunique')))
display(cov)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cov['cusips'].plot.bar(ax=ax[0], color='#4C72B0'); ax[0].set_title('Covered CUSIPs per month'); ax[0].set_xlabel('')
cov['rows'].plot.bar(ax=ax[1], color='#55A868'); ax[1].set_title('Covered bond-days per month'); ax[1].set_xlabel('')
savefig('02_coverage_by_month')

# Market context: where yields went over the sample, and how dispersed daily moves were. The factor paths in
# Section 4 and the regime split in Section 5 read against this.
_lvl = raw_panel.groupby('date')['closing_yield'].quantile([0.1, 0.5, 0.9]).unstack()
_mv = raw_panel.groupby('date')['closing_yield_change_bp'].agg(sd='std', med='median')
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].fill_between(_lvl.index, _lvl[0.1], _lvl[0.9], alpha=0.25, label='10th-90th pct'); ax[0].plot(_lvl.index, _lvl[0.5], color='k', lw=1.2, label='median'); ax[0].set_title('Closing yield across the covered universe (%)'); ax[0].legend()
ax[1].plot(_mv.index, _mv['sd'], color='#C44E52', lw=1); ax[1].set_title('Cross-sectional sd of the daily yield change (bp)'); ax[1].axvline(pd.Timestamp(CFG.first_oos_date), color='k', ls='--', lw=0.8)
savefig('02_market_context')
record('data', panel_rows=len(raw_panel), panel_cusips=raw_panel['cusip'].nunique(), date_min=str(raw_panel['date'].min().date()), date_max=str(raw_panel['date'].max().date()))

# %%
CELL_T('2. Data load and QA [2]')
# --- v3.9 new characteristics: trade size from MSRB prints, issuer industry from the trade track ---
# Trade size is a bond characteristic here, not a trade feature: the time-decayed mean of log quantity over the
# bond's MSRB prints strictly before the date (half-life trade_size_halflife_days), with the decayed print count as
# a frequency measure and a flag for bonds with no print history. Industry comes from muni_algo_trade_track via the
# pipeline's static source (`track_static`); when the source is absent the industry instruments are skipped.
t0 = time.perf_counter()
msrb_store = PIPE.store('msrb')
msrb_raw = msrb_store.read(columns=['date', 'cusip', 'quantity', 'tradetime', 'tradetype', 'side', 'yield', 'price'], categorical=False)   # v4.8: the yield makes the store the print tape of Section 11c
HAS_MSRB = msrb_raw is not None and not msrb_raw.empty
if HAS_MSRB:
    prints_src = msrb_raw[['cusip', 'tradetime', 'quantity']]
elif HAS_TRADES:
    prints_src = trades_raw[['cusip', 'msrb_tradetime', 'msrb_quantity']].rename(columns={'msrb_tradetime': 'tradetime', 'msrb_quantity': 'quantity'})
    print('MSRB store empty: trade-size characteristics built from the matched trades instead')
else:
    prints_src = None


def trade_size_features(prints: pd.DataFrame, panel_rows: pd.DataFrame, halflife_days: float) -> pd.DataFrame:
    """Per bond-day: decayed mean log trade size and decayed print count over prints STRICTLY before the date."""
    p = prints.dropna(subset=['cusip', 'tradetime', 'quantity']).copy()
    p['cusip'] = p['cusip'].astype('string'); p['trade_date'] = to_ns(p['tradetime']).dt.normalize()
    p['lq'] = np.log(pd.to_numeric(p['quantity'], errors='coerce').clip(lower=1_000.0, upper=1e8))
    p = p.dropna(subset=['lq', 'trade_date'])
    daily = p.groupby(['cusip', 'trade_date'], observed=True).agg(S=('lq', 'sum'), N=('lq', 'size')).reset_index()
    dec = group_decayed_sums(daily, ['cusip'], ['S', 'N'], halflife_days, 'trade_date').rename(columns={'trade_date': 'print_date'}).sort_values('print_date')
    q = panel_rows[['cusip', 'date']].copy(); q['cusip'] = q['cusip'].astype('string'); q['_i'] = np.arange(len(q)); q = q.sort_values('date')
    j = pd.merge_asof(q, dec[['cusip', 'print_date', 'S_ewm', 'N_ewm']], left_on='date', right_on='print_date', by='cusip', direction='backward', allow_exact_matches=False)
    w = (0.5 ** (1.0 / halflife_days)) ** (j['date'] - j['print_date']).dt.days.astype(float)
    j['print_freq_lag'] = (j['N_ewm'] * w).fillna(0.0)
    j['trade_size_lag'] = np.where(j['print_freq_lag'] > 0.05, (j['S_ewm'] * w) / j['print_freq_lag'].replace(0, np.nan), np.nan)
    return j.sort_values('_i')[['trade_size_lag', 'print_freq_lag']].reset_index(drop=True)


if prints_src is not None:
    _ts = trade_size_features(prints_src, raw_panel, CFG.trade_size_halflife_days)
    raw_panel['trade_size_lag'] = _ts['trade_size_lag'].to_numpy(); raw_panel['print_freq_lag'] = _ts['print_freq_lag'].to_numpy()
    print(f'trade-size characteristics: {raw_panel["trade_size_lag"].notna().mean():.1%} of bond-days have a print history; median decayed print count {raw_panel["print_freq_lag"].median():.2f}; '
          f'median trade size {np.exp(raw_panel["trade_size_lag"].median()):,.0f} par | {time.perf_counter() - t0:.0f}s')
_ts_dir = PIPE.path(PIPE.data_dir, 'track_static')
track_static = PIPE.store('track_static').read(categorical=False) if _ts_dir.exists() and any(_ts_dir.glob('*.parquet')) else None
HAS_INDUSTRY = track_static is not None and 'issuer_industry' in track_static.columns
if HAS_INDUSTRY:
    _ind = track_static[['cusip', 'issuer_industry']].copy(); _ind['cusip'] = _ind['cusip'].astype('string'); _ind = _ind.dropna(subset=['cusip']).drop_duplicates('cusip', keep='last')
    raw_panel = raw_panel.merge(_ind, on='cusip', how='left')
    _share = raw_panel['issuer_industry'].astype('string').str.strip().replace('', pd.NA).notna().mean()
    print(f'issuer industry: {_share:.1%} of bond-days labelled; largest industries:'); display(raw_panel['issuer_industry'].astype('string').str.upper().value_counts().head(12).to_frame('bond-days'))
else:
    print('issuer industry: track_static source not found in the pipeline store; industry instruments and breakdowns are skipped (see the regeneration note in the header)')
record('data', has_msrb_store=bool(HAS_MSRB), has_industry=bool(HAS_INDUSTRY), trade_size_coverage=float(raw_panel['trade_size_lag'].notna().mean()) if 'trade_size_lag' in raw_panel.columns else None)

# %%
CELL_T('2. Data load and QA [3]')
# --- rating fallback audit ---
RATING_SCALE = {'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15, 'BBB+': 14, 'BBB': 13, 'BBB-': 12,
                'BB+': 11, 'BB': 10, 'BB-': 9, 'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1, 'D': 0}
BLANKS = {'', 'NAN', 'NONE', 'NULL', '<NA>', 'NR', 'N/A', 'WR'}


def clean_rating(series: pd.Series) -> pd.Series:
    s = series.astype('string').str.strip().str.upper()
    return s.where(s.notna() & ~s.isin(BLANKS))


def rating_with_fallback(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    final = clean_rating(frame['final_comp_rating']) if 'final_comp_rating' in frame else pd.Series(pd.NA, index=frame.index, dtype='string')
    comp = clean_rating(frame['comp_rating']) if 'comp_rating' in frame else pd.Series(pd.NA, index=frame.index, dtype='string')
    rating = final.fillna(comp)
    source = pd.Series(np.where(final.notna(), 'final', np.where(comp.notna(), 'comp_fallback', 'NR')), index=frame.index)
    score = rating.map(RATING_SCALE).astype('float64')
    return rating, score, source


_rating, _score, _source = rating_with_fallback(raw_panel)
rating_audit = pd.DataFrame({
    'final_comp_rating blank share': [float(clean_rating(raw_panel['final_comp_rating']).isna().mean())],
    'comp_rating blank share': [float(clean_rating(raw_panel['comp_rating']).isna().mean())],
    'after fallback blank share': [float(_rating.isna().mean())],
    'unmapped non-blank ratings': [int((_rating.notna() & _score.isna()).sum())],
}).T.rename(columns={0: 'value'})
display(rating_audit)
print('rating source mix:', _source.value_counts(normalize=True).round(4).to_dict())
unmapped = _rating[_rating.notna() & _score.isna()].value_counts().head(10)
if len(unmapped):
    print('Unmapped rating strings (extend RATING_SCALE if material):'); display(unmapped)
record('data', **{k.replace(' ', '_'): float(v) for k, v in rating_audit['value'].items()})

# %%
CELL_T('2. Data load and QA [4]')
# --- extreme-yield handling ---
def flag_extreme_by_date(frame: pd.DataFrame, col: str, k: float) -> pd.Series:
    g = frame.groupby('date', observed=True)[col]
    med = g.transform('median')
    mad = (frame[col] - med).abs().groupby(frame['date'], observed=True).transform('median') * 1.4826
    mad = mad.where(mad > 1e-6, 1e-6)
    return ((frame[col] - med).abs() > k * mad) | ~np.isfinite(frame[col])


ext_now = flag_extreme_by_date(raw_panel, 'closing_yield', CFG.yield_mad_k)
ext_lag = flag_extreme_by_date(raw_panel.assign(closing_yield_lag1=raw_panel['closing_yield_lag1']), 'closing_yield_lag1', CFG.yield_mad_k)
raw_panel['extreme_yield'] = (ext_now | ext_lag).astype(int)
big_move = raw_panel['closing_yield_change_bp'].abs() > CFG.target_clip_bp
print(f'extreme yield rows: {int(raw_panel["extreme_yield"].sum()):,} ({raw_panel["extreme_yield"].mean():.3%}) | |target| > {CFG.target_clip_bp:.0f} bp: {int(big_move.sum()):,} ({big_move.mean():.3%})')
q = raw_panel['closing_yield'].quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]).rename('closing_yield quantiles')
display(q.to_frame())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
clean = raw_panel.loc[raw_panel['extreme_yield'] == 0, 'closing_yield']
ax[0].hist(raw_panel['closing_yield'].clip(-5, 25), bins=120, color='#C44E52', alpha=0.6, label='raw (clipped to [-5, 25])')
ax[0].hist(clean, bins=120, color='#4C72B0', alpha=0.6, label='after MAD rule'); ax[0].legend(); ax[0].set_title('Closing yield (%)')
monthly = raw_panel.assign(m=raw_panel['date'].dt.to_period('M').astype(str))
monthly[monthly['extreme_yield'] == 0].boxplot(column='closing_yield', by='m', ax=ax[1], showfliers=False); ax[1].set_title('Closing yield by month (kept rows)'); ax[1].set_xlabel('')
tgt = raw_panel['closing_yield_change_bp']
ax[2].hist(tgt.clip(-100, 100), bins=120, color='#55A868'); ax[2].set_yscale('log'); ax[2].set_title('Daily yield change (bp, clipped ±100, log count)')
plt.suptitle(''); savefig('02_yield_distributions')
record('data', extreme_yield_rows=int(raw_panel['extreme_yield'].sum()), extreme_yield_share=float(raw_panel['extreme_yield'].mean()), target_over_clip_share=float(big_move.mean()))

# %% [markdown]
# ## 3. Model panel: instruments
#
# Thirteen instruments in v3: an intercept, seven rank-normalised continuous characteristics, the NR flag and four
# state dummies (CA, NY, TX, FL). v3.9 adds the characteristics the desk works with and the model had overlooked:
# **trade size** (decayed mean log MSRB quantity per bond, prints strictly before the date), **print frequency**
# (the decayed print count), a **no-print flag**, and **issuer industry** (the largest industries as dummies, with
# OTHER and UNKNOWN as the omitted reference). Size enters the factor model as a bond characteristic; the trade's
# own quantity is used in Section 9b and as a breakdown axis, with the desk's quantity bins. The continuous set is premium/discount, yield level, years to worst, extension,
# rating score, trailing mark-change liquidity, and a duration instrument. In the v1 spec the duration instrument
# was modified duration itself; on the regenerated data its rank correlation with years-to-worst reached 0.98 and
# the first factor became a leveraged long-short of the two (gross weight 27). With `duration_instrument='ratio'`
# the instrument is modified duration divided by years-to-worst, which is the call-structure signal (near 1 for
# bullets, well below 1 for bonds priced to call) and is close to orthogonal to term. Every other state
# is the explicit `OTHER` bucket: it is reported and used as a feature in the level model, but it is the omitted
# reference category in Z because five exhaustive dummies plus the intercept are exactly collinear. All
# continuous instruments are rank-normalised within date to [-0.5, 0.5]. The target is the daily closing-yield
# change in bp, clipped.

# %%
CELL_T('3. Model panel: instruments [1]')
STATE_DUMMIES = ['CA', 'NY', 'TX', 'FL']
TARGET = 'closing_yield_change_bp'


def rank_normalize(frame: pd.DataFrame, cols: list[str], prefix: str = 'z') -> tuple[pd.DataFrame, list[str]]:
    out = frame
    zcols = []
    for c in cols:
        r = out.groupby('date', observed=True)[c].rank(method='average', pct=True)
        out[f'{prefix}_{c}'] = (r - 0.5).fillna(0.0)
        zcols.append(f'{prefix}_{c}')
    return out, zcols


def term_characteristics(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date = frame['date']
    mat_num = pd.to_numeric(frame['maturity'], errors='coerce')
    mat = pd.to_datetime(mat_num, unit='ms', errors='coerce')
    if mat.isna().mean() > 0.5:
        mat = pd.to_datetime(frame['maturity'], errors='coerce')
    call = pd.to_datetime(frame['nxt_call_dt'], errors='coerce')
    cpn = pd.to_numeric(frame['cpn'], errors='coerce')
    is_call = frame['callable'].astype('string').str.upper().isin(['Y', 'YES', 'TRUE', '1'])
    cur_y = pd.to_numeric(frame['closing_yield_lag1'], errors='coerce')
    priced_to_call = is_call & call.notna() & (call > date) & (cpn > cur_y)
    worst = mat.where(~priced_to_call, call)
    yrs_to_mat = (mat - date).dt.days / 365.25
    yrs_to_worst = (worst - date).dt.days / 365.25
    return yrs_to_worst, (yrs_to_mat - yrs_to_worst).clip(lower=0)


def build_model_panel(panel: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, list[str]]:
    yp = panel.copy()
    yp = yp[yp['extreme_yield'] == 0]
    yp = yp[yp['closing_lag_days'].fillna(1.0) <= cfg.max_lag_days]
    yp['premium_discount_lag1'] = pd.to_numeric(yp['closing_price_lag1'], errors='coerce') - 100.0
    yp['years_to_worst'], yp['extension'] = term_characteristics(yp)
    _mat = pd.to_datetime(pd.to_numeric(yp['maturity'], errors='coerce'), unit='ms', errors='coerce')
    if _mat.isna().mean() > 0.5:
        _mat = pd.to_datetime(yp['maturity'], errors='coerce')
    yp['maturity_year'] = _mat.dt.year
    yp['is_callable'] = yp['callable'].astype('string').str.upper().isin(['Y', 'YES', 'TRUE', '1']).astype(float)
    yp['call_structure'] = np.where(yp['is_callable'] == 0, 'bullet', np.where(yp['extension'] > 0.5, 'priced-to-call', 'callable, to maturity'))
    zero_dur = (pd.to_numeric(yp['modified_duration_lag1'], errors='coerce') < cfg.zero_duration_cut) & (yp['years_to_worst'] > 1.0)
    print(f'zero-duration long-maturity rows excluded: {int(zero_dur.sum()):,} ({zero_dur.mean():.3%})')
    yp = yp[~zero_dur]
    yp['rating'], yp['rating_score'], yp['rating_source'] = rating_with_fallback(yp)
    yp['nr_flag'] = yp['rating_score'].isna().astype(float)
    state = yp['state_code'].astype('string').str.strip().str.upper()
    state = state.where(state.notna() & ~state.isin(['', 'NAN', 'NONE', 'NULL', '<NA>']), 'OTHER')
    yp['state_bucket'] = state.where(state.isin(STATE_DUMMIES), 'OTHER')
    for s in STATE_DUMMIES:
        yp[f'state_{s}'] = (state == s).astype(float)
    # Explicit reference bucket for audit, EDA and the level model. NOT an IPCA instrument: with market_fv
    # present the five exhaustive state dummies are exactly collinear, so OTHER is the omitted category in Z.
    yp['state_others'] = (yp['state_bucket'] == 'OTHER').astype(float)
    yp = yp.sort_values(['cusip', 'date'], kind='stable')
    chg = (yp.groupby('cusip', observed=True)['closing_price'].diff().abs() > 0).astype(float)
    yp['liquidity_20'] = chg.groupby(yp['cusip'], observed=True).transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
    yp['duration_ratio'] = (pd.to_numeric(yp['modified_duration_lag1'], errors='coerce') / yp['years_to_worst'].clip(lower=0.25)).clip(0, 1.5)
    dur_instr = 'duration_ratio' if cfg.duration_instrument == 'ratio' else 'modified_duration_lag1'
    base = [dur_instr, 'premium_discount_lag1', 'closing_yield_lag1']
    cont = base + ['years_to_worst', 'extension', 'rating_score', 'liquidity_20']
    # v3.9: trade size and print frequency from MSRB prints (strictly before the date), with a no-print flag
    size_cols = [c for c in ['trade_size_lag', 'print_freq_lag'] if c in yp.columns and yp[c].notna().mean() > 0.2]
    if 'trade_size_lag' in yp.columns:
        yp['no_print_flag'] = yp['trade_size_lag'].isna().astype(float)
    cont = cont + size_cols
    yp = yp.dropna(subset=base + [TARGET]).copy()
    yp['target_bp_raw'] = yp[TARGET]
    yp[TARGET] = yp[TARGET].clip(-cfg.target_clip_bp, cfg.target_clip_bp)
    yp['mark_unchanged'] = (yp['target_bp_raw'] == 0).astype(float)
    # consecutive days the evaluated mark has not moved (0 on a day it moved). A stale mark is a data state, not a fair price.
    _moved = (yp['mark_unchanged'] == 0).groupby(yp['cusip'], observed=True).cumsum()
    yp['days_since_mark_move'] = yp['mark_unchanged'].groupby([yp['cusip'], _moved], observed=True).cumsum()
    yp, zcols = rank_normalize(yp, cont)
    yp['market_fv'] = 1.0
    # v3.9: issuer industry dummies for the largest industries; OTHER and UNKNOWN are the omitted reference
    ind_cols = []
    if 'issuer_industry' in yp.columns:
        ind = yp['issuer_industry'].astype('string').str.strip().str.upper()
        ind = ind.where(ind.notna() & ~ind.isin(['', 'NAN', 'NONE', 'NULL', '<NA>']), 'UNKNOWN')
        top_ind = ind[ind != 'UNKNOWN'].value_counts().head(cfg.industry_dummies).index.tolist()
        yp['industry_bucket'] = ind.where(ind.isin(top_ind), pd.Series(np.where(ind == 'UNKNOWN', 'UNKNOWN', 'OTHER'), index=ind.index))
        for k in top_ind:
            col = 'ind_' + re.sub(r'[^A-Za-z0-9]+', '_', str(k)).strip('_')[:20]
            yp[col] = (ind == k).astype(float); ind_cols.append(col)
    chars = ['market_fv'] + zcols + ['nr_flag'] + (['no_print_flag'] if 'no_print_flag' in yp.columns else []) + [f'state_{s}' for s in STATE_DUMMIES] + ind_cols
    cnt = yp.groupby('cusip', observed=True)[TARGET].transform('count')
    yp = yp[cnt >= cfg.min_obs_per_bond].copy()
    per_date = yp.groupby('date', observed=True)['cusip'].transform('size')
    sparse = per_date < cfg.min_bonds_per_date
    if sparse.any():
        print(f'sparse dates dropped (< {cfg.min_bonds_per_date} bonds): {sorted(yp.loc[sparse, "date"].dt.date.unique())}')
        yp = yp[~sparse].copy()
    keep = ['cusip', 'date', TARGET, 'target_bp_raw', 'mark_unchanged', 'days_since_mark_move', 'closing_lag_days', 'closing_yield', 'closing_yield_lag1',
            'closing_price', 'modified_duration_lag1', 'duration_ratio', 'dv01_lag1', 'years_to_worst', 'extension', 'rating', 'rating_score', 'rating_source',
            'cpn', 'state_code', 'state_bucket', 'state_others', 'mkt_yield', 'long_comp_name', 'maturity_year', 'is_callable', 'call_structure',
            'trade_size_lag', 'print_freq_lag', 'issuer_industry', 'industry_bucket'] + chars
    keep = [c for c in dict.fromkeys(keep) if c in yp.columns]
    return yp[keep].sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True), chars


t0 = time.perf_counter()
model, CHARS = build_model_panel(raw_panel, CFG)
print(f'[{CFG.spec_version}] model rows {len(model):,} | cusips {model["cusip"].nunique():,} | dates {model["date"].nunique()} | L={len(CHARS)} | {time.perf_counter()-t0:.1f}s')
print('instruments:', CHARS)
print('mark_unchanged share:', round(float(model['mark_unchanged'].mean()), 4), '| rating source mix:', model['rating_source'].value_counts(normalize=True).round(3).to_dict())
geo = model.groupby('state_bucket', observed=True).agg(rows=('cusip', 'size'), cusips=('cusip', 'nunique'))
geo['row_share'] = geo['rows'] / geo['rows'].sum()
print('Geo buckets (OTHER is the omitted reference category in Z; state_others is kept for audit and the level model):')
display(geo.sort_values('rows', ascending=False))
top_other = model.loc[model['state_bucket'] == 'OTHER', 'state_code'].astype('string').value_counts().head(8)
print('largest states inside OTHER:', top_other.to_dict())
if 'industry_bucket' in model.columns:
    _ib = model.groupby('industry_bucket', observed=True).agg(rows=('cusip', 'size'), cusips=('cusip', 'nunique')); _ib['row_share'] = _ib['rows'] / _ib['rows'].sum()
    print('Industry buckets (OTHER and UNKNOWN are the omitted reference in Z):'); display(_ib.sort_values('rows', ascending=False))
display(model[CHARS].describe().T[['mean', 'std', 'min', 'max']])
record('model_panel', rows=len(model), cusips=model['cusip'].nunique(), dates=model['date'].nunique(), instruments=CHARS, mark_unchanged_share=float(model['mark_unchanged'].mean()), geo_buckets=geo.round(4).to_dict())

# %%
CELL_T('3. Model panel: instruments [2]')
# Instrument EDA: correlation structure and the duration vs term relationship
zc = [c for c in CHARS if c.startswith('z_')]
sample = model.sample(min(len(model), 200_000), random_state=CFG.seed)
corr = sample[zc + ['nr_flag'] + [c for c in CHARS if c in ('no_print_flag',) or c.startswith('ind_')] + [f'state_{s}' for s in STATE_DUMMIES]].corr(method='spearman')
fig, ax = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw={'width_ratios': [1.3, 1.1, 0.7]})
im = ax[0].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax[0].set_xticks(range(len(corr))); ax[0].set_xticklabels([c.replace('z_', '').replace('_lag1', '') for c in corr.columns], rotation=90)
ax[0].set_yticks(range(len(corr))); ax[0].set_yticklabels([c.replace('z_', '').replace('_lag1', '') for c in corr.columns])
ax[0].set_title('Spearman correlation of instruments'); ax[0].grid(False); plt.colorbar(im, ax=ax[0], fraction=0.046)
hb = ax[1].hexbin(sample['years_to_worst'].clip(0, 40), sample['modified_duration_lag1'].clip(0, 25), gridsize=45, cmap='viridis', mincnt=1)
ax[1].set_xlabel('years to worst'); ax[1].set_ylabel('modified duration (lag 1)'); ax[1].set_title('Duration vs term: the call structure wedge'); plt.colorbar(hb, ax=ax[1], fraction=0.046)
geo['row_share'].reindex(STATE_DUMMIES + ['OTHER']).plot.bar(ax=ax[2], color=['#4C72B0'] * 4 + ['#8C8C8C']); ax[2].set_title('Geo buckets (share of bond-days)'); ax[2].set_xlabel('')
savefig('03_instrument_eda')
print('max |corr| off-diagonal:', round(float((corr.values - np.eye(len(corr))).max()), 3), '| duration instrument:', CFG.duration_instrument)

# %%
CELL_T('3. Model panel: instruments [3]')
# The two data rules.
# (1) Dispersion days. An evaluator-wide reprice is a day where many bonds move a lot relative to the day's MEDIAN
#     move; a day where the whole curve moves 15 bp is a common move the intercept factor absorbs and must stay in
#     the fit. The v2.3 rule used the raw share of large moves and flagged the late-September sell-off. The rule is
#     now on |target - median_t|. Dispersion days get zero weight in the ALS moments; common-move days are flagged
#     and kept. Both are reported.
# (2) Partial-mark days. A date with far fewer marked bonds than usual is a partial evaluator run, not a market
#     day; the fixed 1,000-bond floor let one through in v5 (a leverage spike of 0.16 against a baseline of 0.05).
#     Dates with fewer than partial_day_ratio of the trailing-20-date median count are dropped from the panel.
_cnt = model.groupby('date', observed=True).size()
_ref = _cnt.rolling(20, min_periods=5).median().shift(1).bfill()
PARTIAL_DAYS = set(pd.DatetimeIndex(_cnt.index[_cnt < CFG.partial_day_ratio * _ref]))
if PARTIAL_DAYS:
    print(f'partial-mark days dropped (< {CFG.partial_day_ratio:.0%} of trailing median count): {sorted(d.date() for d in PARTIAL_DAYS)}')
    model = model[~model['date'].isin(PARTIAL_DAYS)].reset_index(drop=True)
_med = model.groupby('date', observed=True)['target_bp_raw'].transform('median')
_dev = (model['target_bp_raw'] - _med).abs()
_disp_share = (_dev > CFG.dispersion_bp).groupby(model['date'], observed=True).mean()
_common = model.groupby('date', observed=True)['target_bp_raw'].median()
_raw_share = (model['target_bp_raw'].abs() > CFG.dispersion_bp).groupby(model['date'], observed=True).mean()
DISPERSION_DAYS = set(pd.DatetimeIndex(_disp_share.index[_disp_share > CFG.dispersion_share_cut]))
COMMON_MOVE_DAYS = set(pd.DatetimeIndex(_common.index[_common.abs() > CFG.common_move_bp]))
model['dispersion_day'] = model['date'].isin(DISPERSION_DAYS).astype(float)
model['common_move_day'] = model['date'].isin(COMMON_MOVE_DAYS).astype(float)
print(f'dispersion days (> {CFG.dispersion_share_cut:.0%} of bonds move > {CFG.dispersion_bp:.0f} bp relative to the median): {sorted(d.date() for d in DISPERSION_DAYS)}')
print(f'common-move days (|median move| > {CFG.common_move_bp:.0f} bp, kept in the fit): {sorted(d.date() for d in COMMON_MOVE_DAYS)}')
print(f'days the old raw-share rule would have flagged: {int((_raw_share > CFG.dispersion_share_cut).sum())} | of which common-move days: {int(((_raw_share > CFG.dispersion_share_cut) & _common.abs().gt(CFG.common_move_bp)).sum())}')
fig, ax = plt.subplots(1, 3, figsize=(19, 3.9))
ax[0].plot(_cnt.index, _cnt.values, color='k', lw=1, label='bonds priced'); ax[0].plot(_ref.index, CFG.partial_day_ratio * _ref.values, color='#C44E52', ls='--', lw=0.8, label=f'{CFG.partial_day_ratio:.0%} of trailing median')
for d in PARTIAL_DAYS:
    ax[0].axvline(d, color='#C44E52', lw=1.2, alpha=0.7)
ax[0].legend(fontsize=8); ax[0].set_title('Partial-mark-day rule: bonds priced per date')
ax[1].plot(_disp_share.index, _disp_share.values, color='#8172B2', lw=1, label='share > 10 bp from the median move'); ax[1].plot(_raw_share.index, _raw_share.values, color='grey', lw=0.8, alpha=0.7, label='share > 10 bp raw (old rule)')
ax[1].axhline(CFG.dispersion_share_cut, color='k', ls='--', lw=0.8); ax[1].legend(fontsize=8); ax[1].set_title('Dispersion-day rule vs the old raw-share rule')
ax[2].bar(_common.index, _common.values, width=1.0, color=np.where(_common.abs() > CFG.common_move_bp, '#DD8452', '#4C72B0')); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Median daily yield change (bp); orange = common-move day, kept')
savefig('02_data_rules')
record('data_rules', partial_days=[str(d.date()) for d in sorted(PARTIAL_DAYS)], dispersion_days=[str(d.date()) for d in sorted(DISPERSION_DAYS)], common_move_days=[str(d.date()) for d in sorted(COMMON_MOVE_DAYS)])

# %% [markdown]
# ## 4. IPCA: K sweep, Gamma anatomy, factor paths
#
# $r_{i,t} = z_{i,t-1}'\Gamma f_t + \varepsilon_{i,t}$, estimated by alternating least squares on per-date
# cross-products. Identification: $\Gamma'\Gamma = I$, factors ordered by variance, non-negative factor means.
# The full-sample fit below is for interpretation and K selection only; everything downstream uses the
# walk-forward Gamma.

# %%
CELL_T('4. IPCA: K sweep, Gamma anatomy, factor paths [1]')
def date_weights(dates: pd.DatetimeIndex, rr: np.ndarray, nobs: np.ndarray, cfg: RunConfig, repricing: set) -> np.ndarray:
    """Per-date weights for the ALS moments. 'inverse_var' divides each date by its cross-sectional mean square of the
    target (capped), so one 20 bp day does not carry 50 normal days; dispersion days get weight zero when excluded."""
    w = np.ones(len(dates))
    if cfg.date_weighting == 'inverse_var':
        ms = rr / np.maximum(nobs, 1)
        w = np.clip(np.median(ms[ms > 0]) / np.maximum(ms, 1e-12), 0.0, cfg.date_weight_cap)
    if cfg.exclude_dispersion_from_fit and repricing:
        w = np.where(dates.isin(list(repricing)), 0.0, w)
    return w


def precompute_moments(frame: pd.DataFrame, chars: list[str], target: str, weighted: bool = True) -> dict:
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    L = len(chars)
    A = np.zeros((len(dates), L, L)); b = np.zeros((len(dates), L)); rr = np.zeros(len(dates)); nobs = np.zeros(len(dates), dtype=np.int64)
    pos = {d: i for i, d in enumerate(dates)}
    for d, g in frame.groupby('date', sort=True, observed=True):
        i = pos[pd.Timestamp(d)]
        Z = g[chars].to_numpy(float); r = g[target].to_numpy(float)
        A[i] = Z.T @ Z; b[i] = Z.T @ r; rr[i] = float(r @ r); nobs[i] = len(g)
    w = date_weights(dates, rr, nobs, CFG, DISPERSION_DAYS) if weighted else np.ones(len(dates))
    return {'dates': dates, 'A': A * w[:, None, None], 'b': b * w[:, None], 'rr': rr * w, 'nobs': nobs, 'chars': list(chars), 'w': w, 'rr_unweighted': rr}


class MomentIPCA:
    """IPCA by alternating least squares on per-date moments (Kelly, Pruitt, Su 2019)."""

    def __init__(self, n_factors: int, max_iter: int = 50, tol: float = 1e-5, ridge: float = 1e-10, seed: int = 0):
        self.K = int(n_factors); self.max_iter = max_iter; self.tol = tol; self.ridge = ridge; self.seed = seed
        self.Gamma = None; self.Factors = None; self.loss_history: list[float] = []

    def _initial_gamma(self, m, gamma_init=None):
        L = m['A'].shape[1]
        if gamma_init is not None and np.shape(gamma_init) == (L, self.K):
            return np.asarray(gamma_init, float)
        managed = m['b'] / np.maximum(m['nobs'][:, None], 1)
        u, _, _ = np.linalg.svd(managed.T @ managed)
        return u[:, :self.K]

    def _solve_factors(self, Gamma, m):
        A, b, nobs = m['A'], m['b'], m['nobs']
        F = np.zeros((A.shape[0], self.K)); eye = np.eye(self.K)
        for t in range(A.shape[0]):
            if nobs[t] >= self.K:
                F[t] = np.linalg.solve(Gamma.T @ A[t] @ Gamma + self.ridge * eye, Gamma.T @ b[t])
        return F

    def _identify(self, G, F):
        Q, R = np.linalg.qr(G)
        s = np.sign(np.diag(R)); s[s == 0] = 1
        Q = Q @ np.diag(s); R = Q.T @ G; F = F @ R.T
        vals, vecs = np.linalg.eigh(F.T @ F / max(F.shape[0], 1))
        W = vecs[:, np.argsort(vals)[::-1]]
        G = Q @ W; F = F @ W
        s = np.sign(F.mean(axis=0)); s[s == 0] = 1
        return G * s, F * s

    def _loss(self, G, F, m):
        A, b, rr, nobs = m['A'], m['b'], m['rr'], m['nobs']
        tot = sum(rr[t] - 2.0 * F[t] @ G.T @ b[t] + F[t] @ (G.T @ A[t] @ G) @ F[t] for t in range(A.shape[0]))
        return tot / max(int(nobs.sum()), 1)

    def fit(self, m, gamma_init=None, verbose=False):
        A, b = m['A'], m['b']; L = A.shape[1]; K = self.K
        G = self._initial_gamma(m, gamma_init); F = self._solve_factors(G, m); G, F = self._identify(G, F)
        eye = np.eye(L * K); self.loss_history = []
        for it in range(self.max_iter):
            F_raw = self._solve_factors(G, m)
            lhs = np.zeros((L * K, L * K)); rhs = np.zeros(L * K)
            for t in range(A.shape[0]):
                ft = F_raw[t]; lhs += np.kron(np.outer(ft, ft), A[t]); rhs += np.kron(ft, b[t])
            G, F = self._identify(np.linalg.solve(lhs + self.ridge * eye, rhs).reshape((L, K), order='F'), F_raw)
            loss = self._loss(G, F, m); self.loss_history.append(loss)
            if verbose and (it == 0 or (it + 1) % 10 == 0):
                print(f'  iter {it+1:03d} loss {loss:.6e}')
            if it > 0 and abs(self.loss_history[-2] - loss) / max(abs(self.loss_history[-2]), 1e-12) < self.tol:
                break
        self.Gamma, self.Factors = G, F
        return self

    def variance_explained(self, m) -> float:
        """1 - SSE/SST over the moments (in-sample)."""
        sse = self._loss(self.Gamma, self.Factors, m) * max(int(m['nobs'].sum()), 1)
        return 1.0 - sse / max(float(m['rr'].sum()), 1e-12)


t0 = time.perf_counter()
moments = precompute_moments(model, CHARS, TARGET)
print(f'moments: T={len(moments["dates"])}, L={len(CHARS)}, obs={int(moments["nobs"].sum()):,} | {time.perf_counter()-t0:.1f}s')
print(f'date weighting: {CFG.date_weighting} (cap {CFG.date_weight_cap}x) | zero-weight dates: {int((moments["w"] == 0).sum())} | weight range on kept dates {moments["w"][moments["w"] > 0].min():.2f}..{moments["w"][moments["w"] > 0].max():.2f}')
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(moments['dates'], moments['w'], width=1.0, color='#4C72B0'); ax.set_title('Date weights in the ALS moments (0 = repricing day excluded from the fit)')
savefig('04_date_weights')

k_rows = []
fits = {}
for k in CFG.candidate_k:
    mk = MomentIPCA(k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(moments)
    fits[k] = mk
    k_rows.append({'K': k, 'in_sample_var_explained (date-weighted)': mk.variance_explained(moments), 'iterations': len(mk.loss_history)})
k_table = pd.DataFrame(k_rows).set_index('K')
display(k_table)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k_table.index, k_table['in_sample_var_explained (date-weighted)'], marker='o'); ax.set_xlabel('K'); ax.set_ylabel('variance explained (in-sample)'); ax.set_title('K sweep (full sample, interpretation only)')
savefig('04_k_sweep')
record('ipca_full_sample', **{f'var_explained_K{k}': float(v) for k, v in k_table['in_sample_var_explained (date-weighted)'].items()})

# %%
CELL_T('4. IPCA: K sweep, Gamma anatomy, factor paths [2]')
ipca = fits[CFG.selected_k]
gamma = pd.DataFrame(ipca.Gamma, index=CHARS, columns=[f'factor{i+1}' for i in range(CFG.selected_k)])
fvar = ipca.Factors.var(axis=0); fshare = fvar / fvar.sum()
factors = pd.DataFrame(ipca.Factors, index=moments['dates'], columns=gamma.columns)
display(gamma.assign(abs_l2=np.sqrt((gamma ** 2).sum(axis=1))).sort_values('abs_l2', ascending=False))
display(pd.DataFrame({'variance': fvar, 'share': fshare, 'mean_bp': factors.mean(), 'sd_bp': factors.std()}, index=gamma.columns))

fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [1.1, 0.6, 1.6]})
im = ax[0].imshow(gamma.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax[0].set_yticks(range(len(gamma))); ax[0].set_yticklabels([c.replace('z_', '').replace('_lag1', '') for c in gamma.index]); ax[0].set_xticks(range(CFG.selected_k)); ax[0].set_xticklabels(gamma.columns)
for i in range(gamma.shape[0]):
    for j in range(gamma.shape[1]):
        ax[0].text(j, i, f'{gamma.values[i, j]:+.2f}', ha='center', va='center', fontsize=8)
ax[0].set_title('Gamma: characteristic -> factor loadings'); ax[0].grid(False); plt.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].bar(gamma.columns, fshare, color=['#4C72B0', '#55A868', '#C44E52'][:CFG.selected_k]); ax[1].set_title('Share of factor variance')
factors.cumsum().plot(ax=ax[2]); ax[2].set_title('Cumulative factor realisations (bp of yield)'); ax[2].axhline(0, color='k', lw=0.5)
savefig('04_gamma_anatomy')
record('ipca_full_sample', factor_variance_share=[float(x) for x in fshare], gamma=gamma.round(4).to_dict())

# %% [markdown]
# ## 5. Walk-forward residuals: refit cadence, regime-conditional Gamma, residuals of record
#
# Gamma is estimated on dates strictly before each test block, aligned across versions by orthogonal Procrustes,
# and applied to the test block. The v5 Gamma swap test found that a Gamma frozen two months earlier explained
# *more* out-of-sample variance than the weekly refit in every month from June on. Five schemes are therefore run
# side by side on identical test dates (`cadences`): monthly and quarterly refits on an expanding window and a Gamma
# frozen at the first OOS date. The v31 run also ran weekly and a dispersion-regime-conditional Gamma; all five sat
# within 0.3 points of variance explained in every month, so those two are closed and dropped from the default run
# (the code still supports 'weekly' and 'regime'). The residuals of record come from the pre-registered cadence.

# %%
CELL_T('5. Walk-forward residuals: refit cadence, regime-conditional [1]')
def make_folds(dates: pd.DatetimeIndex, train_start: str, first_oos: str, test_days: int | None) -> list[dict]:
    """Expanding-window folds. test_days=None gives a single fold (Gamma frozen at first_oos)."""
    dates = pd.DatetimeIndex(dates).sort_values()
    ts, start = pd.Timestamp(train_start), pd.Timestamp(first_oos)
    folds, k = [], 1
    while start <= dates.max():
        end = start + pd.Timedelta(days=test_days) if test_days else dates.max() + pd.Timedelta(days=1)
        tr = dates[(dates >= ts) & (dates < start)]; te = dates[(dates >= start) & (dates < end)]
        if len(tr) and len(te):
            folds.append({'fold': k, 'train_dates': tr, 'test_dates': te, 'train_end': tr[-1], 'test_start': te[0], 'test_end': te[-1]}); k += 1
        start = end
    return folds


def procrustes_align(G: np.ndarray, G_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate G onto G_ref within its own column span. Residuals are unchanged; labels, betas and factor paths
    become continuous across refits instead of swapping when eigenvalues cross."""
    U, _, Vt = np.linalg.svd(G.T @ G_ref)
    R = U @ Vt
    return G @ R, R


DATE_DISP = model.groupby('date', observed=True)[TARGET].std().sort_index()           # cross-sectional sd of the target by date
DATE_DISP_LAG = DATE_DISP.shift(1)                                                      # what is known at the start of a date


def walk_forward_residuals(frame: pd.DataFrame, chars: list[str], folds: list[dict], cfg: RunConfig, regime: bool = False, verbose: bool = True):
    """Returns (residual frame, aligned Gammas, raw Gammas). With regime=True each fold fits a high- and a low-dispersion
    Gamma on the training dates and each test date is scored with the Gamma of the regime its previous day indicates."""
    parts, gammas, gammas_raw = [], [], []
    inits: dict = {}; refs: dict = {}
    by_date = {d: g for d, g in frame.groupby('date', sort=True, observed=True)}
    eye = np.eye(cfg.selected_k); fcols = [f'factor{i+1}' for i in range(cfg.selected_k)]
    for fo in folds:
        t0 = time.perf_counter()
        train = frame[frame['date'].isin(fo['train_dates'])]
        groups = {'all': fo['train_dates']}
        if regime:
            thr = DATE_DISP.reindex(fo['train_dates']).median()
            hi = fo['train_dates'][DATE_DISP.reindex(fo['train_dates']).to_numpy() > thr]
            groups = {'high': hi, 'low': fo['train_dates'].difference(hi)}
        G_by = {}
        for gname, gdates in groups.items():
            sub = train[train['date'].isin(gdates)]
            if sub['date'].nunique() < 10:
                continue
            mk = MomentIPCA(cfg.selected_k, cfg.max_iter, cfg.tol, cfg.ridge, cfg.seed).fit(precompute_moments(sub, chars, TARGET), gamma_init=inits.get(gname))
            inits[gname] = mk.Gamma
            gv = pd.Timestamp(fo['train_end'])
            gammas_raw.append(pd.DataFrame(mk.Gamma, index=chars, columns=fcols).assign(gamma_version=gv, fold=fo['fold'], regime=gname))
            G_al = mk.Gamma if refs.get(gname) is None else procrustes_align(mk.Gamma, refs[gname])[0]
            refs[gname] = G_al; G_by[gname] = G_al
            gammas.append(pd.DataFrame(G_al, index=chars, columns=fcols).assign(gamma_version=gv, fold=fo['fold'], regime=gname))
        if not G_by:
            continue
        for d in fo['test_dates']:
            g = by_date.get(d)
            if g is None:
                continue
            if regime:
                lagd = DATE_DISP_LAG.get(d, np.nan); gname = 'high' if (np.isfinite(lagd) and lagd > thr and 'high' in G_by) else ('low' if 'low' in G_by else 'high')
            else:
                gname = 'all'
            G_al = G_by[gname]
            Z = g[chars].to_numpy(float); r = g[TARGET].to_numpy(float); B = Z @ G_al
            f = np.linalg.solve(B.T @ B + cfg.ridge * eye, B.T @ r)
            out = g[['cusip', 'date']].copy()
            out['gamma_version'] = pd.Timestamp(fo['train_end']); out['fold'] = fo['fold']; out['regime'] = gname
            out['target_bp'] = r; out['fitted_bp'] = B @ f; out['pit_residual'] = r - out['fitted_bp']
            for j in range(cfg.selected_k):
                out[f'beta{j+1}'] = B[:, j]; out[f'f{j+1}'] = f[j]
            parts.append(out)
        if verbose:
            print(f'  fold {fo["fold"]:02d} | train ->{fo["train_end"].date()} | test {fo["test_start"].date()}..{fo["test_end"].date()} | {"/".join(G_by)} | {time.perf_counter()-t0:.1f}s')
    return pd.concat(parts, ignore_index=True), pd.concat(gammas), pd.concat(gammas_raw)


CADENCE_DAYS = {'weekly': 7, 'monthly': 30, 'quarterly': 91, 'frozen': None, 'regime': CFG.regime_cadence_days}
PANEL_FP = frame_fingerprint(model, [TARGET, 'z_duration_ratio', 'z_closing_yield_lag1']) + f'|disp:{sorted(str(d) for d in DISPERSION_DAYS)}|partial:{sorted(str(d) for d in PARTIAL_DAYS)}'


def _walk_forward_all() -> dict:
    out: dict = {}
    for name in CFG.cadences:
        t0 = time.perf_counter()
        folds_c = make_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CADENCE_DAYS[name])
        res_c, gam_c, gam_raw_c = walk_forward_residuals(model, CHARS, folds_c, CFG, regime=(name == 'regime'), verbose=(name == CFG.record_cadence))
        out[name] = {'resid': res_c, 'gammas': gam_c, 'gammas_raw': gam_raw_c, 'folds': folds_c}
        print(f'{name:>10}: {len(folds_c):2d} folds | OOS variance explained {1 - res_c["pit_residual"].var() / res_c["target_bp"].var():.4f} | residual sd {res_c["pit_residual"].std():.2f} bp | {time.perf_counter()-t0:.0f}s')
    return out


runs = cached('walk_forward', _walk_forward_all, deps=[STEP3_FP, PANEL_FP], code=[walk_forward_residuals, make_folds, procrustes_align, MomentIPCA, precompute_moments])
for name, r in runs.items():
    print(f'{name:>10}: {len(r["folds"]):2d} folds | OOS variance explained {1 - r["resid"]["pit_residual"].var() / r["resid"]["target_bp"].var():.4f} | residual sd {r["resid"]["pit_residual"].std():.2f} bp')

cad_rows = {}
for name, r in runs.items():
    rr_ = r['resid']
    cad_rows[name] = rr_.assign(month=rr_['date'].dt.to_period('M').astype(str)).groupby('month').apply(lambda g: 1 - g['pit_residual'].var() / g['target_bp'].var(), include_groups=False)
cadence = pd.DataFrame(cad_rows); cadence.loc['ALL'] = {name: 1 - r['resid']['pit_residual'].var() / r['resid']['target_bp'].var() for name, r in runs.items()}
print('OOS variance explained by month and refit scheme (identical test dates):'); display(cadence.round(4))
if 'regime' in runs:
    _rg = runs['regime']['resid'].groupby('regime').agg(rows=('cusip', 'size'), dates=('date', 'nunique'), var_explained=('pit_residual', lambda s: np.nan))
    _rg['var_explained'] = [1 - g['pit_residual'].var() / g['target_bp'].var() for _, g in runs['regime']['resid'].groupby('regime')]
    print('Regime-conditional Gamma: test dates by regime'); display(_rg.round(4))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
cadence.drop(index='ALL').plot.bar(ax=ax[0]); ax[0].set_title('OOS variance explained by month: refit cadence and regime-conditional Gamma'); ax[0].set_xlabel(''); ax[0].legend(fontsize=8)
(cadence.drop(index='ALL').sub(cadence.drop(index='ALL')[CFG.record_cadence], axis=0)).drop(columns=CFG.record_cadence).plot(ax=ax[1], marker='o'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title(f'Difference from the {CFG.record_cadence} refit (points of variance explained)'); ax[1].set_xlabel('')
savefig('05_refit_cadence')
record('walk_forward', cadence_table=cadence.round(4).to_dict(), record_cadence=CFG.record_cadence)

# residuals of record
resid, gamma_versions, gamma_versions_raw, folds = runs[CFG.record_cadence]['resid'], runs[CFG.record_cadence]['gammas'], runs[CFG.record_cadence]['gammas_raw'], runs[CFG.record_cadence]['folds']
oos_var_expl = 1.0 - resid['pit_residual'].var() / resid['target_bp'].var()
print(f'residuals of record ({CFG.record_cadence}): {len(resid):,} rows | {resid["cusip"].nunique():,} cusips | {resid["date"].nunique()} dates | {resid["gamma_version"].nunique()} Gamma versions | OOS variance explained {oos_var_expl:.3f} | residual sd {resid["pit_residual"].std():.2f} bp')
resid.to_parquet(ARTIFACTS / 'pit_ipca_residuals_v3.parquet', index=False)
gamma_versions.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_v3.parquet', index=False)
gamma_versions_raw.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_raw_v3.parquet', index=False)
record('walk_forward', folds=len(folds), residual_rows=len(resid), cusips=resid['cusip'].nunique(), oos_dates=resid['date'].nunique(), gamma_versions=resid['gamma_version'].nunique(), oos_variance_explained=float(oos_var_expl), residual_sd_bp=float(resid['pit_residual'].std()))
del runs

# %%
CELL_T('5. Walk-forward residuals: refit cadence, regime-conditional [2]')
# Fit quality over time and Gamma stability across versions
rm = resid.assign(month=resid['date'].dt.to_period('M').astype(str)).groupby('month').apply(
    lambda g: pd.Series({'oos_var_explained': 1 - g['pit_residual'].var() / g['target_bp'].var(), 'residual_sd_bp': g['pit_residual'].std(), 'target_sd_bp': g['target_bp'].std(), 'rows': len(g)}), include_groups=False)
display(rm)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
rm['oos_var_explained'].plot.bar(ax=ax[0], color='#4C72B0'); ax[0].set_title('OOS variance explained by month'); ax[0].set_xlabel('')
rm[['residual_sd_bp', 'target_sd_bp']].plot(ax=ax[1], marker='o'); ax[1].set_title('Residual vs target sd (bp)'); ax[1].set_xlabel('')
top = gamma.assign(l2=np.sqrt((gamma ** 2).sum(axis=1))).sort_values('l2', ascending=False).index[:5]
gv_f1 = gamma_versions.reset_index().rename(columns={'index': 'instrument'}).pivot(index='gamma_version', columns='instrument', values='factor1')[top]
gv_f1.plot(ax=ax[2], marker='.'); ax[2].set_title('Factor-1 loadings by Gamma version, Procrustes-aligned'); ax[2].axhline(0, color='k', lw=0.5)
savefig('05_fit_over_time_and_gamma_stability')
fcols = [f'factor{i+1}' for i in range(CFG.selected_k)]
_gv = gamma_versions.reset_index().rename(columns={'index': 'instrument'}); _gr = gamma_versions_raw.reset_index().rename(columns={'index': 'instrument'})
sd_aligned = _gv.groupby('instrument')[fcols].std().mean(axis=1); sd_raw = _gr.groupby('instrument')[fcols].std().mean(axis=1)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
_order = gamma.assign(l2=np.sqrt((gamma ** 2).sum(axis=1))).sort_values('l2', ascending=False).index
_sdtab = pd.DataFrame({'raw (variance-ordered)': sd_raw, 'Procrustes-aligned': sd_aligned}).reindex(_order); _sdtab.index = [c.replace('z_', '').replace('_lag1', '') for c in _sdtab.index]
_sdtab.plot.bar(ax=ax[0]); ax[0].set_title('Across-version sd of loadings per instrument'); ax[0].tick_params(axis='x', rotation=60)
_gr.pivot(index='gamma_version', columns='instrument', values='factor1')[top].plot(ax=ax[1], marker='.', legend=False); ax[1].set_title('Same loadings before alignment: label swaps when eigenvalues cross'); ax[1].axhline(0, color='k', lw=0.5)
savefig('05_gamma_alignment')
drift, drift_raw = float(sd_aligned.mean()), float(sd_raw.mean())
print(f'mean across-version sd of loadings: raw {drift_raw:.4f} -> aligned {drift:.4f}')
# Daily residual dispersion: the volatility regime the later tests live in
_disp = resid.groupby('date').agg(resid_sd=('pit_residual', 'std'), target_sd=('target_bp', 'std'), big_share=('pit_residual', lambda s: (s.abs() > 10).mean()), n=('cusip', 'size'))
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
_disp[['target_sd', 'resid_sd']].plot(ax=ax[0]); ax[0].set_title('Daily cross-sectional sd: target vs residual (bp)')
ax[1].plot(_disp.index, _disp['big_share'], color='#C44E52'); ax[1].set_title('Share of bond-days with |residual| > 10 bp'); ax2 = ax[1].twinx(); ax2.plot(_disp.index, _disp['n'], color='grey', lw=0.8, alpha=0.6); ax2.set_ylabel('bonds priced', color='grey')
savefig('05_residual_dispersion')
record('walk_forward', gamma_loading_sd_across_versions=drift, gamma_loading_sd_raw=drift_raw, monthly=rm.round(4).to_dict())

# %%
CELL_T('5. Walk-forward residuals: refit cadence, regime-conditional [3]')
# Lookups used by the leverage diagnostic: the instrument matrix per date and the Gamma per version
MODEL_BY_DATE = {d: g for d, g in model.groupby('date', sort=True, observed=True)}
GAMMA_ALIGNED = {gv: g.drop(columns=['gamma_version', 'fold', 'regime']).to_numpy(float) for gv, g in gamma_versions.groupby('gamma_version', sort=True)}
GAMMA_RAW = {gv: g.drop(columns=['gamma_version', 'fold', 'regime']).to_numpy(float) for gv, g in gamma_versions_raw.groupby('gamma_version', sort=True)}
_gv_index = pd.DatetimeIndex(sorted(GAMMA_ALIGNED))

# %% [markdown]
# ## 6. Factor-mimicking weights and the residual-maker
#
# $f_t = (B_t'B_t+\lambda I)^{-1}B_t'r_t = W_t^{F\prime} r_t$ with $W_t^F = B_t(B_t'B_t+\lambda I)^{-1}$. Each column
# of $W^F$ is the portfolio of that day's bonds whose return is the factor. The residual is the projection off the
# beta span; we check $B_t'\varepsilon_t = 0$ and the gross leverage / breadth of each factor portfolio.

# %%
CELL_T('6. Factor-mimicking weights and the residual-maker [1]')
beta_cols = [f'beta{j+1}' for j in range(CFG.selected_k)]
w_rows, orth = [], []
lev_rows = []
for d, g in resid.groupby('date', sort=True, observed=True):
    B = g[beta_cols].to_numpy(float); eps = g['pit_residual'].to_numpy(float)
    BtB_inv = np.linalg.inv(B.T @ B + CFG.ridge * np.eye(CFG.selected_k))
    W = B @ BtB_inv
    orth.append(np.abs(B.T @ eps).max())
    # rotation-invariant leverage: ||W||_F = sqrt(trace((B'B)^-1)). Per-factor gross is NOT rotation-invariant, so the
    # raw (variance-ordered) Gamma is shown next to the aligned one.
    gv = g['gamma_version'].iloc[0]
    Z = MODEL_BY_DATE[pd.Timestamp(d)][CHARS].to_numpy(float); B_raw = Z @ GAMMA_RAW[gv]
    W_raw = B_raw @ np.linalg.inv(B_raw.T @ B_raw + CFG.ridge * np.eye(CFG.selected_k))
    lev_rows.append({'date': d, 'frobenius_leverage': float(np.sqrt(np.trace(BtB_inv))), 'n': len(g), 'dispersion': d in DISPERSION_DAYS})
    for j in range(CFG.selected_k):
        w = W[:, j]
        w_rows.append({'date': d, 'factor': f'factor{j+1}', 'n': len(w), 'gross': np.abs(w).sum(), 'net': w.sum(), 'breadth': (np.abs(w).sum() ** 2) / (w @ w), 'gross_raw_gamma': np.abs(W_raw[:, j]).sum()})
wdiag = pd.DataFrame(w_rows); levdiag = pd.DataFrame(lev_rows).set_index('date')
wsum = wdiag.groupby('factor')[['n', 'gross', 'gross_raw_gamma', 'net', 'breadth']].mean()
display(wsum)
print('max |B\'eps| across dates:', f'{max(orth):.2e}')
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
wdiag.pivot(index='date', columns='factor', values='gross').plot(ax=ax[0]); ax[0].set_title('Gross weight of factor-mimicking portfolios (aligned Gamma)')
wdiag.pivot(index='date', columns='factor', values='gross_raw_gamma').plot(ax=ax[1], ls='--'); ax[1].set_title('Same, raw variance-ordered Gamma (label swaps = jumps)')
ax[2].plot(levdiag.index, levdiag['frobenius_leverage'], color='k'); ax[2].set_title('Rotation-invariant leverage ||W||_F (a jump here is a real span change)')
for gv in _gv_index:
    ax[2].axvline(gv, color='grey', lw=0.4, alpha=0.5)
savefig('06_mimicking_weights')
fig, ax = plt.subplots(figsize=(9, 3.5))
wdiag.pivot(index='date', columns='factor', values='breadth').plot(ax=ax); ax.set_title('Effective breadth of the factor portfolios (bonds)')
savefig('06_mimicking_breadth')
record('mimicking_weights', frobenius_leverage_mean=float(levdiag['frobenius_leverage'].mean()), frobenius_leverage_by_month={str(k): float(v) for k, v in levdiag['frobenius_leverage'].groupby(levdiag.index.to_period('M')).mean().round(4).items()})
record('mimicking_weights', max_beta_orthogonality=float(max(orth)), **{f'{k}_{m}': float(v) for k, row in wsum.iterrows() for m, v in row.items()})

# %% [markdown]
# ## 6b. Factor space: beta-space map, clusters and within-cluster ranking
#
# The exhibits the attention-factor paper uses to make latent factors legible, adapted to munis. Bonds are
# placed in standardised beta space on the last out-of-sample date. (1) A 2-D map of that space coloured by
# rating, call structure, duration and residual rank shows which economic axes the factors organise.
# (2) K-means clusters in beta space give the peer groups the model implies, with their characteristic profile
# and named examples. (3) Inside one cluster, the residual orders bonds rich to cheap: that ordering is the
# comparables view an RFQ desk uses.

# %%
CELL_T('6b. Factor space: beta-space map, clusters and within-cluste [1]')
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

MAP_SAMPLE = 3000
N_CLUSTERS = 8
last_date = resid['date'].max()
snap = resid[resid['date'] == last_date][['cusip', 'pit_residual'] + beta_cols].merge(
    model[model['date'] == last_date][['cusip', 'long_comp_name', 'cpn', 'maturity_year', 'rating', 'rating_score', 'call_structure', 'is_callable',
                                       'modified_duration_lag1', 'years_to_worst', 'extension', 'closing_yield', 'state_bucket']], on='cusip', how='left').reset_index(drop=True)
snap['pit_rank'] = snap['pit_residual'].rank(pct=True)
snap['rating_bucket'] = pd.cut(snap['rating_score'], bins=[-1, 11.5, 14.5, 17.5, 20.5, 21.5], labels=['BB and below', 'BBB', 'A', 'AA', 'AAA']).astype(str).replace('nan', 'NR')


def bond_label(r: pd.Series, width: int = 26) -> str:
    name = r.get('long_comp_name')
    name = str(name)[:width] if isinstance(name, str) and name.strip() else str(r['cusip'])
    mat = f"{int(r['maturity_year'])}" if pd.notna(r.get('maturity_year')) else '----'
    cpn = f"{r['cpn']:.2f}%" if pd.notna(r.get('cpn')) else '--'
    rating = r['rating'] if isinstance(r.get('rating'), str) else 'NR'
    call = {'bullet': 'B', 'priced-to-call': 'C*', 'callable, to maturity': 'C'}.get(r.get('call_structure'), '?')
    return f"{name} {cpn} {mat} {rating} {call}"


snap['label'] = snap.apply(bond_label, axis=1)
Bs = snap[beta_cols].to_numpy(float); Bs = (Bs - Bs.mean(axis=0)) / np.maximum(Bs.std(axis=0), 1e-12)
idx = RNG.choice(len(snap), min(MAP_SAMPLE, len(snap)), replace=False)


def _beta_space_embedding() -> dict:
    try:
        e_ = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=CFG.seed).fit_transform(Bs[idx]); nm_ = 't-SNE'
    except Exception as exc:  # noqa: BLE001
        e_ = PCA(n_components=2, random_state=CFG.seed).fit_transform(Bs[idx]); nm_ = 'PCA'; print('t-SNE unavailable, using PCA:', exc)
    return {'emb': e_, 'name': nm_, 'cusips': snap['cusip'].astype(str).to_numpy()[idx]}


_emb = cached('beta_space_embedding', _beta_space_embedding, deps=[STAGE_KEYS.get('walk_forward'), str(last_date.date()), int(MAP_SAMPLE), frame_fingerprint(snap, beta_cols)])   # v4.7: the embedding is deterministic given the betas and the seed, and it took two minutes per run
emb, emb_name = _emb['emb'], _emb['name']
sm = snap.iloc[idx].copy(); sm['x'], sm['y'] = emb[:, 0], emb[:, 1]

fig, ax = plt.subplots(2, 2, figsize=(14, 11))
for a, (col, title, kind) in zip(ax.ravel(), [('rating_bucket', 'by rating', 'cat'), ('call_structure', 'by call structure', 'cat'),
                                               ('modified_duration_lag1', 'by modified duration', 'num'), ('pit_rank', 'by residual rank (0 rich -> 1 cheap)', 'num')]):
    if kind == 'cat':
        for j, (lvl, g) in enumerate(sm.groupby(col, observed=True)):
            a.scatter(g['x'], g['y'], s=6, alpha=0.7, label=f'{lvl} ({len(g)})', color=plt.cm.tab10(j % 10))
        a.legend(markerscale=3, fontsize=8, loc='best')
    else:
        sc = a.scatter(sm['x'], sm['y'], s=6, c=sm[col], cmap='viridis' if col != 'pit_rank' else 'RdYlGn', alpha=0.8); plt.colorbar(sc, ax=a, fraction=0.046)
    a.set_title(f'Beta space ({emb_name}), {last_date.date()}: {title}'); a.set_xticks([]); a.set_yticks([]); a.grid(False)
savefig('06b_beta_space_map')

# %%
CELL_T('6b. Factor space: beta-space map, clusters and within-cluste [2]')
# v4.3: the same map coloured by how the bond trades: decayed trade size, side mix of its prints, print frequency, industry
_tc = model[model['date'] == last_date][['cusip'] + [c for c in ['trade_size_lag', 'print_freq_lag', 'industry_bucket'] if c in model.columns]].copy(); _tc['cusip'] = _tc['cusip'].astype('string')
smt = sm.assign(cusip=sm['cusip'].astype('string')).merge(_tc, on='cusip', how='left')
if HAS_TRADES and 'msrb_side' in trades_raw.columns:
    _sd = trades_raw[['cusip', 'msrb_side']].copy(); _sd['cusip'] = _sd['cusip'].astype('string'); _sd['side'] = _sd['msrb_side'].astype('string').str.upper().str[0]
    _sd = _sd[_sd['side'].isin(['P', 'S'])]
    _mix = _sd.groupby('cusip', observed=True)['side'].agg(lambda x: float((x == 'P').mean())).rename('p_share')
    smt = smt.merge(_mix, on='cusip', how='left')
panels = [(c, t, k, cm) for c, t, k, cm in [('trade_size_lag', 'by trade size (decayed mean log par of the bond\'s prints)', 'num', 'viridis'),
                                             ('p_share', 'by side mix of the bond\'s prints (share that are dealer buys; 0.5 = balanced)', 'num', 'coolwarm'),
                                             ('print_freq_lag', 'by print frequency (decayed count of prints, log10 colour scale)', 'log', 'viridis'),
                                             ('industry_bucket', 'by issuer industry', 'cat', None)] if c in smt.columns and smt[c].notna().any()]
if panels:
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    for a, (col, title, kind, cmap) in zip(ax.ravel(), panels):
        g0 = smt.dropna(subset=[col])
        a.scatter(smt['x'], smt['y'], s=4, color='#DDDDDD', alpha=0.5)
        if kind == 'cat':
            for j, (lvl, g) in enumerate(g0.groupby(col, observed=True)):
                a.scatter(g['x'], g['y'], s=6, alpha=0.75, label=f'{lvl} ({len(g)})', color=plt.cm.tab10(j % 10))
            a.legend(markerscale=3, fontsize=8, loc='best')
        else:
            kw = {'vmin': 0.0, 'vmax': 1.0} if col == 'p_share' else {}
            cvals = np.log10(g0[col].clip(lower=0.05)) if kind == 'log' else g0[col]   # v4.4: a few very active bonds flattened the linear scale
            sc = a.scatter(g0['x'], g0['y'], s=6, c=cvals, cmap=cmap, alpha=0.85, **kw); plt.colorbar(sc, ax=a, fraction=0.046)
        a.set_title(f'Beta space ({emb_name}), {last_date.date()}: {title}', fontsize=10); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for a in ax.ravel()[len(panels):]:
        a.set_visible(False)
    fig.suptitle('How the bonds in beta space trade (grey = no print history)', fontsize=12); plt.tight_layout(); savefig('06b_beta_space_map_trading')

# %%
CELL_T('6b. Factor space: beta-space map, clusters and within-cluste [3]')
# Clusters in beta space: profile and named examples, then rich/cheap ranking inside one cluster
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=CFG.seed).fit(Bs)
snap['cluster'] = km.labels_
profile = snap.groupby('cluster').agg(bonds=('cusip', 'size'), mod_duration=('modified_duration_lag1', 'mean'), years_to_worst=('years_to_worst', 'mean'), extension=('extension', 'mean'),
                                       share_callable=('is_callable', 'mean'), rating_score=('rating_score', 'mean'), yield_pct=('closing_yield', 'mean'), resid_sd_bp=('pit_residual', 'std'))
profile['top_rating'] = snap.groupby('cluster')['rating_bucket'].agg(lambda s: s.value_counts().index[0])
profile['top_call'] = snap.groupby('cluster')['call_structure'].agg(lambda s: s.value_counts().index[0])
profile = profile.sort_values('mod_duration')
display(profile.round(3))
examples = pd.concat([pd.concat([g.nsmallest(3, 'pit_rank').assign(role='richest'), g.nlargest(3, 'pit_rank').assign(role='cheapest')]).assign(cluster=c) for c, g in snap.groupby('cluster')], ignore_index=True)
display(examples.loc[examples['cluster'].isin(profile.index[:3]), ['cluster', 'role', 'label', 'closing_yield', 'pit_residual', 'pit_rank']].round(3))

fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.3]})
sm['cluster'] = snap.loc[sm.index, 'cluster'].values
for c, g in sm.groupby('cluster'):
    ax[0].scatter(g['x'], g['y'], s=6, alpha=0.75, color=plt.cm.tab10(c % 10), label=f"C{c}: dur {profile.loc[c, 'mod_duration']:.1f}, {profile.loc[c, 'top_rating']}, {profile.loc[c, 'top_call']}")
ax[0].legend(fontsize=7, markerscale=3, loc='best'); ax[0].set_title(f'K-means clusters in beta space (k={N_CLUSTERS})'); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].grid(False)
focus = profile.index[len(profile) // 2]
fg = snap[snap['cluster'] == focus].sort_values('pit_residual')
show = pd.concat([fg.head(10), fg.tail(10)])
ax[1].barh(show['label'], show['pit_residual'], color=np.where(show['pit_residual'] >= 0, '#2CA02C', '#D62728')); ax[1].axvline(0, color='k', lw=0.6)
ax[1].set_title(f"Cluster C{focus} ({len(fg)} bonds, dur {profile.loc[focus, 'mod_duration']:.1f}): richest 10 and cheapest 10 by residual (bp)\n"
                "label = issuer, coupon, maturity, rating, B bullet / C callable / C* priced-to-call", fontsize=10); ax[1].tick_params(axis='y', labelsize=8)
savefig('06b_clusters_and_ranking')
snap.drop(columns=['label']).to_parquet(ARTIFACTS / f'beta_space_snapshot_{last_date.date()}.parquet', index=False)
record('factor_space', date=str(last_date.date()), embedding=emb_name, clusters=N_CLUSTERS, cluster_profile=profile.round(4).reset_index().to_dict(orient='records'))

# %%
CELL_T('6b. Factor space: beta-space map, clusters and within-cluste [4]')
# Point-in-time beta-space clusters for the breakdowns in Section 12: k-means fitted on the standardised betas of the
# FIRST OOS month and applied to every later bond-day with the same centroids and scaling, so membership never looks ahead.
_first_m = resid['date'].dt.to_period('M').min()
_fit = resid[resid['date'].dt.to_period('M') == _first_m]
_mu, _sd = _fit[beta_cols].mean(), _fit[beta_cols].std().replace(0, 1.0)
km_pit = KMeans(n_clusters=CFG.n_clusters, n_init=10, random_state=CFG.seed).fit(((_fit[beta_cols] - _mu) / _sd).to_numpy(float))
resid['beta_cluster'] = km_pit.predict(((resid[beta_cols] - _mu) / _sd).to_numpy(float))
_prof = resid[['cusip', 'date', 'beta_cluster']].merge(model[['cusip', 'date', 'modified_duration_lag1', 'is_callable', 'extension', 'rating_score', 'closing_yield']], on=['cusip', 'date'], how='left')
cluster_profile = _prof.groupby('beta_cluster').agg(bond_days=('cusip', 'size'), mod_duration=('modified_duration_lag1', 'mean'), share_callable=('is_callable', 'mean'), extension=('extension', 'mean'), rating_score=('rating_score', 'mean'), yield_pct=('closing_yield', 'mean')).sort_values('mod_duration')
cluster_profile['label'] = [f"C{c}: dur {r['mod_duration']:.1f}, call {r['share_callable']:.0%}, ext {r['extension']:.1f}" for c, r in cluster_profile.iterrows()]
CLUSTER_LABEL = cluster_profile['label'].to_dict()
print(f'PIT beta-space clusters (k={CFG.n_clusters}) fitted on {_first_m}, applied forward:'); display(cluster_profile.round(3))
record('factor_space', pit_clusters=cluster_profile.round(4).reset_index().to_dict(orient='records'))

# %% [markdown]
# ## 7. Residual dynamics: autocorrelation, activity buckets, the state-space filter
#
# The pooled autocorrelation function of the residual at lags 1..10, with and without dispersion days, and the
# point-in-time activity buckets (trailing-vol quintiles). One reference forecaster on expanding monthly folds,
# the trailing-20 mean (the level signal), frames the state-space filter in 7b, which is the signal of record.
# (The LongConv-lite ridge, which rediscovered the trailing mean, and the by-bucket filter, which lost to the
# pooled one, were closed in v41 and v31 and are no longer run.)

# %%
CELL_T('7. Residual dynamics: autocorrelation, activity buckets, the [1]')
resid = resid.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
grp = resid.groupby('cusip', observed=True)
resid['dispersion_day'] = resid['date'].isin(DISPERSION_DAYS)
acf_rows = []
for lag in range(1, CFG.longconv_lags + 1):
    lagged = grp['pit_residual'].shift(lag); lagged_rp = grp['dispersion_day'].shift(lag)
    ok = lagged.notna()
    x, y = lagged[ok].to_numpy(), resid.loc[ok, 'pit_residual'].to_numpy()
    ex = ok & ~resid['dispersion_day'] & ~lagged_rp.fillna(True).astype(bool)
    xe, ye = lagged[ex].to_numpy(), resid.loc[ex, 'pit_residual'].to_numpy()
    acf_rows.append({'lag': lag, 'pearson': np.corrcoef(x, y)[0, 1], 'spearman': pd.Series(x).corr(pd.Series(y), method='spearman'), 'pairs': int(ok.sum()), 'bartlett_ci': 1.96 / np.sqrt(max(int(ok.sum()), 1)),
                     'pearson_ex_dispersion': np.corrcoef(xe, ye)[0, 1] if ex.sum() > 100 else np.nan, 'spearman_ex_dispersion': pd.Series(xe).corr(pd.Series(ye), method='spearman') if ex.sum() > 100 else np.nan})
acf = pd.DataFrame(acf_rows).set_index('lag')
display(acf)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].bar(acf.index - 0.2, acf['pearson'], width=0.4, label='Pearson'); ax[0].bar(acf.index + 0.2, acf['spearman'], width=0.4, label='Spearman')
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('lag (observations)'); ax[0].set_title('Pooled residual autocorrelation (within CUSIP), all days'); ax[0].legend()
ax[1].bar(acf.index - 0.2, acf['pearson_ex_dispersion'], width=0.4, label='Pearson'); ax[1].bar(acf.index + 0.2, acf['spearman_ex_dispersion'], width=0.4, label='Spearman')
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('lag (observations)'); ax[1].set_title(f'Same, excluding pairs that touch a dispersion day ({len(DISPERSION_DAYS)} dates)'); ax[1].legend()
savefig('07_residual_acf')
record('residual_diagnostics', acf_pearson=acf['pearson'].round(4).to_dict(), acf_spearman=acf['spearman'].round(4).to_dict(), acf_pearson_ex_dispersion=acf['pearson_ex_dispersion'].round(4).to_dict())

# %%
CELL_T('7. Residual dynamics: autocorrelation, activity buckets, the [2]')
# Point-in-time activity buckets: trailing-window volatility of the target per bond, lagged one observation,
# ranked into quintiles within each date. Every bond-day gets a bucket (no UNKNOWN), and nothing looks ahead.
_m = model.sort_values(['cusip', 'date'], kind='stable')
_m['target_abs_vol'] = (_m.groupby('cusip', observed=True)[TARGET]
                          .transform(lambda s: s.rolling(CFG.activity_window, min_periods=max(5, CFG.activity_window // 2)).std().shift(1)))
_m['activity_bucket'] = _m.groupby('date', observed=True)['target_abs_vol'].transform(
    lambda s: pd.qcut(s.rank(method='first'), 5, labels=[f'L{i}' for i in range(1, 6)]).astype(str) if s.notna().sum() >= 50 else pd.Series('UNKNOWN', index=s.index))
_m.loc[_m['target_abs_vol'].isna(), 'activity_bucket'] = 'UNKNOWN'
resid = resid.merge(_m[['cusip', 'date', 'activity_bucket', 'target_abs_vol']], on=['cusip', 'date'], how='left')
resid['activity_bucket'] = resid['activity_bucket'].fillna('UNKNOWN')
print('activity bucket mix (PIT trailing vol):', resid['activity_bucket'].value_counts(normalize=True).round(3).to_dict())
resid['next_residual'] = resid.groupby('cusip', observed=True)['pit_residual'].shift(-1)
resid['next_date'] = resid.groupby('cusip', observed=True)['date'].shift(-1)
pairs = resid.dropna(subset=['next_residual'])


def daily_rank_ic(frame: pd.DataFrame, x: str, y: str, date_col: str = 'next_date', min_n: int = 50) -> pd.Series:
    out = {}
    for d, g in frame.groupby(date_col, observed=True):
        if len(g) >= min_n and g[x].std() > 0 and g[y].std() > 0:
            out[d] = g[x].corr(g[y], method='spearman')
    return pd.Series(out)


def ic_summary(ic: pd.Series) -> dict:
    return {'ic_mean': float(ic.mean()), 'ic_t': float(np.sqrt(ic.notna().sum()) * ic.mean() / ic.std()) if ic.std() > 0 else np.nan, 'dates': int(ic.notna().sum())}


bucket_rows = []
for b, g in pairs.groupby('activity_bucket', sort=True):
    s = ic_summary(daily_rank_ic(g, 'pit_residual', 'next_residual'))
    bucket_rows.append({'activity_bucket': b, 'rows': len(g), 'cusips': g['cusip'].nunique(), 'median_vol_bp': g['target_abs_vol'].median(), **s})
bucket_ic = pd.DataFrame(bucket_rows).set_index('activity_bucket')
display(bucket_ic)
from scipy import stats  # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].bar(bucket_ic.index, bucket_ic['ic_mean'], yerr=1.96 * bucket_ic['ic_mean'] / bucket_ic['ic_t'].replace(0, np.nan), color='#4C72B0', capsize=4); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('Lag-1 rank IC by activity bucket (L1 quiet -> L5 active)')
e = resid['pit_residual']
ax[1].hist(e.clip(-30, 30), bins=150, density=True, color='#55A868', alpha=0.7); xs = np.linspace(-30, 30, 300); ax[1].plot(xs, stats.norm.pdf(xs, 0, e.std()), 'k--', lw=1, label=f'normal sd={e.std():.1f}'); ax[1].set_yscale('log'); ax[1].set_ylim(bottom=1e-6); ax[1].legend(); ax[1].set_title('Residual density (bp, log scale)')
stats.probplot(e.sample(min(len(e), 50_000), random_state=CFG.seed), dist='norm', plot=ax[2]); ax[2].set_title('Residual QQ vs normal: heavy tails')
savefig('07_buckets_and_distribution')
record('residual_diagnostics', bucket_ic=bucket_ic.round(4).to_dict(), residual_kurtosis=float(stats.kurtosis(e)))

# %%
CELL_T('7. Residual dynamics: autocorrelation, activity buckets, the [3]')
# Reference forecaster on expanding monthly folds: the trailing mean (level signal)
L = CFG.longconv_lags   # kept as the number of lags shown in the ACF and the implied-tap exhibit
# Level signal: trailing mean of the residual (the bond's accumulated deviation from factor-implied fair value).
# The flat rank autocorrelation out to lag 10 says this per-bond mean, not yesterday's increment, carries the information.
W = CFG.level_signal_window
resid['resid_mean'] = grp['pit_residual'].transform(lambda s: s.rolling(W, min_periods=max(5, W // 4)).mean())
resid['resid_sd'] = grp['pit_residual'].transform(lambda s: s.rolling(W, min_periods=max(5, W // 4)).std())
pairs = resid.dropna(subset=['next_residual']).copy()
pairs['target_month'] = pairs['next_date'].dt.to_period('M')
months = sorted(pairs['target_month'].unique())


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ y)


def summarise_forecast(name: str, pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {'forecaster': name, 'rows': 0}
    ic = daily_rank_ic(pred, 'yhat', 'next_residual')
    y, yh = pred['next_residual'].to_numpy(), pred['yhat'].to_numpy()
    ex = pred[~pred['next_date'].isin(DISPERSION_DAYS) & ~pred['date'].isin(DISPERSION_DAYS)]
    return {'forecaster': name, 'rows': len(pred), 'months': pred['month'].nunique(), **ic_summary(ic), 'oos_r2_vs_zero': 1 - ((y - yh) ** 2).sum() / (y ** 2).sum(),
            'sign_acc': float((np.sign(y) == np.sign(yh))[(y != 0) & (yh != 0)].mean()), 'ic_ex_dispersion': ic_summary(daily_rank_ic(ex, 'yhat', 'next_residual'))['ic_mean'] if len(ex) > 1000 else np.nan}


def forecaster_eval(name: str, cols: list[str], winsor: bool, alpha: float = 1.0, fitter=None) -> tuple[pd.DataFrame, dict, np.ndarray | None]:
    """Expanding monthly folds. `fitter(Xtr, ytr) -> (predict_fn, coefs)` defaults to ridge; the pooled ARMA(1,1) plugs in here."""
    preds, coefs, extra = [], None, None
    for m in months[1:]:
        tr = pairs[pairs['target_month'] < m].dropna(subset=cols)
        te = pairs[pairs['target_month'] == m].dropna(subset=cols)
        if len(tr) < 5_000 or te.empty:
            continue
        Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        ytr = tr['next_residual'].to_numpy(float)
        if winsor:
            lo, hi = np.quantile(Xtr, CFG.ar_winsor, axis=0); Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
            ylo, yhi = np.quantile(ytr, CFG.ar_winsor); ytr = np.clip(ytr, ylo, yhi)
        if fitter is None:
            coefs = fit_ridge(Xtr, ytr, alpha); yhat = coefs[0] + Xte @ coefs[1:]
        else:
            predict, coefs, extra = fitter(Xtr, ytr); yhat = predict(Xte)
        p = te[['cusip', 'date', 'next_date', 'next_residual', 'activity_bucket']].copy(); p['yhat'] = yhat; p['month'] = str(m)
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    summ = summarise_forecast(name, pred)
    if extra is not None:
        summ['params'] = extra
    return pred, summ, coefs


from scipy.optimize import minimize  # noqa: E402

fc_results, fc_preds, fc_coefs = [], {}, {}
FC_SPECS = [(f'Trailing-{W} mean (level signal)', ['resid_mean'], True, None)]
for name, cols, winsor, fitter in FC_SPECS:
    t0 = time.perf_counter()
    pred, summ, coefs = forecaster_eval(name, cols, winsor, fitter=fitter)
    fc_results.append(summ); fc_preds[name] = pred; fc_coefs[name] = coefs
    print(f'{name}: {time.perf_counter()-t0:.1f}s', '' if 'params' not in summ else summ['params'])

# %% [markdown]
# ### 7b. The signal of record: state-space filter, AR drift plus mark-noise
#
# The structural reading of the residual diagnostics is two components: a slow, persistent deviation of the bond
# from factor-implied fair value (what the trailing mean picks up) and evaluated-mark noise that enters the daily
# increment as a difference (a jump followed by its correction, the negative lag-1 term). Written as a state
# space on the increment $r_t$:
#
# $r_t = m_t + \eta_t - \eta_{t-1} + \varepsilon_t, \qquad m_t = \phi\, m_{t-1} + \xi_t$
#
# with four pooled parameters $(\phi, \sigma_\xi, \sigma_\eta, \sigma_\varepsilon)$ estimated by maximising the
# Gaussian likelihood across bonds (Kalman filter, time = observation index). The one-step forecast is
# $\phi\,\hat m_t - \hat\eta_t$: the filtered drift carried forward minus the part of today's move the filter
# attributes to mark noise. The filter's implied taps over past increments are shown against the lags, and it
# gives a by-product no linear filter has: a filtered **mark-noise** estimate per bond-day, the mark-quality score.
# Increments are winsorised at the training quantiles before filtering. The pooled filter's one-step forecast is
# the residual signal of record (`record_signal`). The fit is cached on the walk-forward residuals' key.

# %%
CELL_T('7b. The signal of record: state-space filter, AR drift plus  [1]')
LOG2PI = np.log(2.0 * np.pi)


def ssm_unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.tanh(theta[0])), float(np.exp(theta[1])), float(np.exp(theta[2])), float(np.exp(theta[3]))


def ssm_filter(Y: np.ndarray, params: tuple[float, float, float, float], want_paths: bool = False):
    """Batched Kalman filter over an (N bonds, T observations) matrix with NaN padding. Returns per-bond log-likelihood
    and, if asked, the forecast of the NEXT increment made at each t, the filtered drift and the filtered mark noise."""
    phi, s_m, s_eta, s_eps = params
    N, T = Y.shape
    Tm = np.array([[phi, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]); Q = np.diag([s_m ** 2, s_eta ** 2, 0.0]); H = np.array([1.0, 1.0, -1.0]); R = s_eps ** 2
    m = np.zeros((N, 3)); P = np.tile(np.diag([s_m ** 2 / max(1.0 - phi ** 2, 1e-6), s_eta ** 2, s_eta ** 2]), (N, 1, 1))
    ll = np.zeros(N); fwd = m_path = eta_path = None
    if want_paths:
        fwd = np.full((N, T), np.nan); m_path = np.full((N, T), np.nan); eta_path = np.full((N, T), np.nan)
    for t in range(T):
        m = m @ Tm.T; P = Tm @ P @ Tm.T + Q
        yhat = m @ H; PH = P @ H; S = PH @ H + R
        y = Y[:, t]; obs = ~np.isnan(y); v = np.where(obs, y - yhat, 0.0); K = PH / S[:, None]
        ll += np.where(obs, -0.5 * (LOG2PI + np.log(S) + v ** 2 / S), 0.0)
        m = m + K * v[:, None] * obs[:, None]
        P = np.where(obs[:, None, None], P - K[:, :, None] * PH[:, None, :], P)
        if want_paths:
            fwd[:, t] = (m @ Tm.T) @ H; m_path[:, t] = m[:, 0]; eta_path[:, t] = m[:, 1]
    return ll, fwd, m_path, eta_path


def ssm_fit(Y: np.ndarray, theta0: np.ndarray | None = None, maxiter: int = 120) -> tuple[np.ndarray, float]:
    theta0 = np.array([np.arctanh(0.9), np.log(0.5), np.log(2.0), np.log(3.0)]) if theta0 is None else np.asarray(theta0, float)
    n_obs = max(int(np.isfinite(Y).sum()), 1)

    def nll(th):
        if not np.all(np.isfinite(th)) or abs(th[0]) > 6 or np.any(np.abs(th[1:]) > 12):
            return 1e12
        return -ssm_filter(Y, ssm_unpack(th))[0].sum() / n_obs

    sol = minimize(nll, theta0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-7})
    return sol.x, float(-sol.fun)


def ssm_implied_taps(params, L_: int = 12) -> np.ndarray:
    """The filter as a linear forecaster: response of the one-step forecast to a unit increment j observations ago."""
    Y = np.zeros((L_, 60 + L_))
    for j in range(L_):
        Y[j, 60 + L_ - 1 - j] = 1.0
    return ssm_filter(Y, params, want_paths=True)[1][:, -1]


def to_sequences(frame: pd.DataFrame, value_col: str) -> tuple[np.ndarray, pd.DataFrame]:
    f = frame.sort_values(['cusip', 'date'], kind='stable').copy(); f['obs_idx'] = f.groupby('cusip', observed=True).cumcount()
    wide = f.pivot(index='cusip', columns='obs_idx', values=value_col)
    return wide.to_numpy(float), f.assign(_row_bond=f['cusip'].map({c: i for i, c in enumerate(wide.index)}))


def ssm_walk_forward(name: str, by_bucket: bool) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Expanding monthly folds. Parameters are estimated on training observations only (subsampled bonds); the filter then
    runs over each bond's full history with those parameters and the forecasts for the test month are kept. With
    by_bucket, each bond is filtered with the parameters of its modal activity bucket in the training window."""
    preds, param_rows = [], []
    theta_prev: dict = {}
    base = resid[['cusip', 'date', 'pit_residual', 'next_date', 'next_residual', 'activity_bucket']].copy()
    base['target_month'] = base['next_date'].dt.to_period('M')
    for m in months[1:]:
        t0 = time.perf_counter()
        tr_mask = base['target_month'] < m
        tr = base[tr_mask]
        if len(tr) < 20_000:
            continue
        lo, hi = np.quantile(tr['pit_residual'], CFG.ar_winsor)
        base_w = base.assign(r_w=base['pit_residual'].clip(lo, hi))
        hist = base_w[base_w['date'] <= tr['date'].max()]   # everything the filter may see up to the last training date
        test_rows = base_w[base_w['target_month'] == m]
        groups = {'ALL': tr['cusip'].unique()}
        if by_bucket:
            modal = tr[tr['activity_bucket'] != 'UNKNOWN'].groupby('cusip', observed=True)['activity_bucket'].agg(lambda s: s.value_counts().index[0])
            groups = {b: modal.index[modal == b].to_numpy() for b in sorted(modal.unique())}
            groups['UNKNOWN'] = np.setdiff1d(tr['cusip'].unique(), modal.index.to_numpy())
        fitted = {}
        for gname, bonds in groups.items():
            if gname == 'UNKNOWN':
                continue   # new bonds with short histories: filtered with the mean of the bucket parameters, not fitted on their own
            fit_bonds = bonds
            if len(fit_bonds) < 50:
                continue
            sub = RNG.choice(fit_bonds, min(len(fit_bonds), CFG.ssm_fit_bonds), replace=False)
            Ytr, _ = to_sequences(tr[tr['cusip'].isin(sub)], 'pit_residual')
            Ytr = np.clip(Ytr, lo, hi)
            th, ll = ssm_fit(Ytr, theta_prev.get(gname), maxiter=CFG.ssm_maxiter); theta_prev[gname] = th
            fitted[gname] = ssm_unpack(th)
            param_rows.append({'month': str(m), 'group': gname, 'bonds_fit': len(sub), 'phi': fitted[gname][0], 's_drift': fitted[gname][1], 's_mark_noise': fitted[gname][2], 's_eps': fitted[gname][3], 'loglik_per_obs': ll})
        if by_bucket and 'UNKNOWN' in groups and 'UNKNOWN' not in fitted and fitted:
            pooled_th = np.mean([theta_prev[g] for g in fitted], axis=0); fitted['UNKNOWN'] = ssm_unpack(pooled_th)
        # filter every bond in the test month through its full history (training dates + the test month itself, PIT by construction)
        full = pd.concat([hist, test_rows[~test_rows.index.isin(hist.index)]]).sort_values(['cusip', 'date'], kind='stable')
        full = full[full['cusip'].isin(test_rows['cusip'].unique())]
        for gname, bonds in groups.items():
            params = fitted.get(gname) or fitted.get('ALL')
            if params is None:
                continue
            sel = full[full['cusip'].isin(bonds)] if gname != 'ALL' or by_bucket else full
            if sel.empty:
                continue
            Y, long = to_sequences(sel, 'r_w')
            _, fwd, m_path, eta_path = ssm_filter(Y, params, want_paths=True)
            rows = long['_row_bond'].to_numpy(); cols_ = long['obs_idx'].to_numpy()
            out = long[['cusip', 'date', 'next_date', 'next_residual', 'activity_bucket', 'target_month']].copy()
            out['yhat'] = fwd[rows, cols_]; out['m_hat'] = m_path[rows, cols_]; out['eta_hat'] = eta_path[rows, cols_]; out['group'] = gname
            preds.append(out[out['target_month'] == m])
        print(f'  {name} | {m}: groups {list(fitted)} | {time.perf_counter()-t0:.1f}s')
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred = pred.dropna(subset=['next_residual']).assign(month=lambda d: d['target_month'].astype(str)) if not pred.empty else pred
    return pred, summarise_forecast(name, pred), pd.DataFrame(param_rows)


ssm_pred, ssm_summ, ssm_params = cached('ssm_pooled', lambda: ssm_walk_forward('State-space AR drift + mark noise (pooled), winsor', by_bucket=False),
                                         deps=[STAGE_KEYS['walk_forward']], code=[ssm_walk_forward, ssm_filter, ssm_fit, ssm_unpack, to_sequences, summarise_forecast])
ssmb_pred, ssmb_params = pd.DataFrame(), pd.DataFrame()     # the by-bucket filter was closed in v31 (IC 0.061 vs 0.083 pooled)
fc_results.append(ssm_summ); fc_preds[ssm_summ['forecaster']] = ssm_pred; fc_coefs[ssm_summ['forecaster']] = None
# mark-noise estimate and filtered drift back onto the residual panel (test-month rows only: strictly out of sample)
if not ssm_pred.empty:
    resid = resid.merge(ssm_pred[['cusip', 'date', 'm_hat', 'eta_hat']].rename(columns={'m_hat': 'ssm_drift', 'eta_hat': 'ssm_mark_noise'}), on=['cusip', 'date'], how='left')
    resid['eta_abs'] = resid['ssm_mark_noise'].abs()
_rec_pred = ssm_pred if CFG.record_signal == 'pooled' else ssmb_pred
if not _rec_pred.empty:
    # the signal of record: the chosen filter's one-step forecast, strictly out of sample
    resid = resid.merge(_rec_pred[['cusip', 'date', 'yhat']].rename(columns={'yhat': 'ssm_signal'}), on=['cusip', 'date'], how='left')
fc_table = pd.DataFrame(fc_results).set_index('forecaster')
display(fc_table.drop(columns=[c for c in ['params'] if c in fc_table.columns]))
if not ssm_params.empty:
    print('State-space parameters (pooled) by fold:'); display(ssm_params.round(4))
if not ssmb_params.empty:
    lastp = ssmb_params[ssmb_params['month'] == ssmb_params['month'].max()].set_index('group')
    lastp['signal_to_noise'] = lastp['s_drift'] / lastp['s_mark_noise']
    print('State-space parameters by activity bucket (last fold): does the data imply the fade?'); display(lastp.round(4))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
if not ssm_params.empty:
    _taps = ssm_implied_taps(tuple(ssm_params.iloc[-1][['phi', 's_drift', 's_mark_noise', 's_eps']]), L)
    ax[0].bar(np.arange(1, L + 1), _taps, width=0.6, color='#C44E52', label='state-space filter (implied taps, last fold)')
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('lag (observations)'); ax[0].legend(fontsize=8); ax[0].set_title('The filter as a linear forecaster: weight on the increment j observations ago', fontsize=10)
best = fc_table['ic_mean'].idxmax()
if not fc_preds[best].empty:
    ic_m = fc_preds[best].groupby('month').apply(lambda g: ic_summary(daily_rank_ic(g, 'yhat', 'next_residual'))['ic_mean'], include_groups=False)
    ic_m.plot.bar(ax=ax[1], color='#55A868'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title(f'Monthly OOS rank IC: {best}', fontsize=10); ax[1].set_xlabel('')
if not ssm_params.empty:
    _pp = ssm_params.set_index('month')
    ax[2].bar(_pp.index, _pp['phi'], color='#8172B2', label='phi (drift persistence)'); ax2 = ax[2].twinx(); ax2.plot(_pp.index, _pp['s_mark_noise'], 'ko-', label='mark-noise sd (bp)'); ax2.plot(_pp.index, _pp['s_eps'], 's--', color='grey', label='white-noise sd (bp)'); ax2.set_ylim(bottom=0)
    ax[2].set_title('Pooled state-space parameters by fold', fontsize=10); ax[2].legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8); ax[2].tick_params(axis='x', rotation=30)
savefig('07_forecasters')
record('residual_forecast', table=fc_table.drop(columns=[c for c in ['params'] if c in fc_table.columns]).round(4).to_dict(), ssm_params=ssm_params.round(5).to_dict(orient='records'), ssm_params_by_bucket=ssmb_params.round(5).to_dict(orient='records'))

# %%
CELL_T('7b. The signal of record: state-space filter, AR drift plus  [2]')
# Two-regime check: does the body persist while the tails revert? Conditional next-residual by current-residual bin.
bins = pairs['pit_residual'].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).to_numpy()
pairs['res_bin'] = pd.cut(pairs['pit_residual'], bins=np.unique(bins), include_lowest=True)
two_regime = pairs.groupby('res_bin', observed=True).agg(n=('next_residual', 'size'), mean_now=('pit_residual', 'mean'), mean_next=('next_residual', 'mean'), median_next=('next_residual', 'median'))
two_regime['continuation_ratio'] = two_regime['mean_next'] / two_regime['mean_now']
_pex = pairs[~pairs['date'].isin(DISPERSION_DAYS) & ~pairs['next_date'].isin(DISPERSION_DAYS)]
two_regime_ex = _pex.groupby('res_bin', observed=True).agg(n=('next_residual', 'size'), mean_now=('pit_residual', 'mean'), mean_next=('next_residual', 'mean'))
two_regime['mean_next_ex_dispersion'] = two_regime_ex['mean_next']; two_regime['continuation_ex_dispersion'] = two_regime_ex['mean_next'] / two_regime_ex['mean_now']
display(two_regime)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(two_regime['mean_now'], two_regime['mean_next'], marker='o', label='all days'); ax.plot(two_regime_ex['mean_now'], two_regime_ex['mean_next'], marker='s', ls='--', label='excluding dispersion days')
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6); ax.legend()
ax.set_xlabel('current residual, bin mean (bp)'); ax.set_ylabel('next residual, bin mean (bp)'); ax.set_title('Body persists, tails revert? conditional means by current-residual quantile bin')
savefig('07_two_regime')
record('residual_diagnostics', two_regime=two_regime.reset_index().astype({'res_bin': str}).round(4).to_dict(orient='records'))

# %% [markdown]
# ## 8. Risk model snapshot: what the inventory charge needs
#
# The factor model's job in the quote engine is the risk and marking layer, not the alpha. This block packages
# what an inventory risk charge needs, per bond, on the last residual date: the three point-in-time factor betas,
# the factor covariance (sample covariance of the realised factors from the full-sample fit, bp$^2$ per day, so an
# in-sample estimate), the idiosyncratic variance (trailing variance of the residual of record) and the mark-noise
# scale from the state-space filter. The systematic share of a bond's daily variance is $\beta' \Sigma_f \beta$ over
# the total; the table shows it by beta-space cluster. The snapshot is written to `artifacts_v3/` for the optimizer.
#
# The block closes with the one marking property of the risk layer that the print-to-print panel confirmed: a stale
# mark rolled forward by its factor-implied move (beta . f, cumulated) is closer to the next mark than the stale mark
# itself, by horizon. It is an in-panel upper bound (the factor realisations are the fitted ones).

# %%
CELL_T('8. Risk model snapshot: what the inventory charge needs [1]')
last_date = resid['date'].max()
# v4.7: only the trailing window up to the last date is needed, so the rolling statistics are taken on each bond's last `activity_window` rows (a full rolling pass took two minutes)
_rs = resid.sort_values(['cusip', 'date'], kind='stable').groupby('cusip', observed=True).tail(CFG.activity_window).reset_index(drop=True)
_mn_col = 'ssm_mark_noise' if 'ssm_mark_noise' in _rs.columns else None
_g_rs = _rs.groupby('cusip', observed=True)
_idio = _g_rs['pit_residual'].agg(['var', 'count']); _idio = _idio['var'].where(_idio['count'] >= 5)
_rs['idio_var_bp2'] = _rs['cusip'].map(_idio).to_numpy()
if _mn_col:
    _mn = _rs.assign(_a=_rs[_mn_col].abs()).groupby('cusip', observed=True)['_a'].agg(['mean', 'count']); _rs['mark_noise_sd_bp'] = _rs['cusip'].map(_mn['mean'].where(_mn['count'] >= 5)).to_numpy()
fcov = factors.cov()                                                    # bp^2 per day, factors of the full-sample fit
risk_snap = _rs[_rs['date'] == last_date][['cusip', 'date'] + beta_cols + ['idio_var_bp2'] + (['mark_noise_sd_bp'] if _mn_col else []) + (['beta_cluster'] if 'beta_cluster' in _rs.columns else [])].copy()
_B = risk_snap[beta_cols].fillna(0.0).to_numpy(float)
risk_snap['sys_var_bp2'] = np.einsum('ij,jk,ik->i', _B, fcov.to_numpy(float), _B)
risk_snap['total_var_bp2'] = risk_snap['sys_var_bp2'] + risk_snap['idio_var_bp2'].fillna(risk_snap['idio_var_bp2'].median())
risk_snap['systematic_share'] = risk_snap['sys_var_bp2'] / risk_snap['total_var_bp2'].replace(0, np.nan)
risk_snap['cluster'] = risk_snap['beta_cluster'].map(CLUSTER_LABEL) if 'beta_cluster' in risk_snap.columns else 'ALL'
risk_tab = risk_snap.groupby('cluster', observed=True).agg(bonds=('cusip', 'size'), systematic_sd_bp=('sys_var_bp2', lambda x: float(np.sqrt(x.median()))), idio_sd_bp=('idio_var_bp2', lambda x: float(np.sqrt(x.median()))),
                                                           systematic_share=('systematic_share', 'median'), **({'mark_noise_sd_bp': ('mark_noise_sd_bp', 'median')} if _mn_col else {}))
print(f'Risk model snapshot on {last_date.date()}: {len(risk_snap):,} bonds. Factor covariance (bp^2 / day):'); display(fcov.round(3))
print('Daily risk by beta-space cluster (medians; systematic = beta\' Sigma beta, idiosyncratic = trailing residual variance):'); display(risk_tab.round(3))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
risk_tab[['systematic_sd_bp', 'idio_sd_bp']].plot.barh(ax=ax[0], color=['#4C72B0', '#DD8452']); ax[0].set_title('Daily yield risk per bond by cluster (bp, medians)', fontsize=10); ax[0].set_ylabel(''); ax[0].legend(['systematic (factor)', 'idiosyncratic (residual)'], fontsize=8); ax[0].tick_params(axis='y', labelsize=8)
ax[1].hist(risk_snap['systematic_share'].clip(0, 1).dropna(), bins=40, color='#55A868'); ax[1].set_title('Share of a bond\'s daily variance that is systematic', fontsize=10); ax[1].set_xlabel('systematic share'); ax[1].set_ylabel('bonds')
savefig('08_risk_model_snapshot')
risk_snap.to_parquet(ARTIFACTS / f'risk_model_snapshot_{last_date.date()}.parquet', index=False)
# the factor roll-forward of stale marks (v4.4: folded in from the former Section 14)
_rsf = resid.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True); _gsf = _rsf.groupby('cusip', observed=True); rf_rows = []
_cs_t = _gsf['target_bp'].cumsum(); _cs_r = _gsf['pit_residual'].cumsum(); _gt = _cs_t.groupby(_rsf['cusip'], observed=True); _gr = _cs_r.groupby(_rsf['cusip'], observed=True)
for h in [1, 2, 3, 5, 10]:
    # v4.7: the forward h-observation sum per bond from cumulative sums and group-wise shifts (vectorised; the per-bond rolling pass took a minute)
    cum_target = _gt.shift(-(h - 1)) - _gt.shift(1).fillna(0.0); cum_resid = _gr.shift(-(h - 1)) - _gr.shift(1).fillna(0.0)
    ok = cum_target.notna() & cum_resid.notna()
    rf_rows.append({'horizon': h, 'n': int(ok.sum()), 'stale_mae_bp': cum_target[ok].abs().mean(), 'rolled_mae_bp': cum_resid[ok].abs().mean(), 'stale_rmse_bp': np.sqrt((cum_target[ok] ** 2).mean()), 'rolled_rmse_bp': np.sqrt((cum_resid[ok] ** 2).mean())})
rollf = pd.DataFrame(rf_rows).set_index('horizon'); rollf['rmse_reduction'] = 1 - rollf['rolled_rmse_bp'] / rollf['stale_rmse_bp']
print('Factor roll-forward of a stale mark versus leaving it unchanged, by horizon in observations (in-panel upper bound):'); display(rollf.round(3))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rollf.index, rollf['stale_mae_bp'], marker='o', label='leave mark stale'); ax.plot(rollf.index, rollf['rolled_mae_bp'], marker='o', label='roll forward by beta . f'); ax.set_xlabel('horizon (observations)'); ax.set_ylabel('MAE (bp)'); ax.legend(); ax.set_title('Factor roll-forward of marks (in-panel upper bound)')
savefig('08_roll_forward')
record('risk_model', date=str(last_date.date()), factor_covariance=fcov.round(5).to_dict(), by_cluster=risk_tab.round(4).reset_index().to_dict(orient='records'), bonds=int(len(risk_snap)), roll_forward=rollf.round(4).reset_index().to_dict(orient='records'))

# %% [markdown]
# ### 8b. Factor premia as expected returns: the Kelly-Palhares-Pruitt test on a bond return target
#
# Kelly, Palhares and Pruitt ("Modeling Corporate Bond Returns") fit IPCA to bond returns and read the model two
# ways a market maker can use: the exposures times the factor means are **expected returns**, so a sort on them is a
# tradable ranking, and the factors' historical means and covariance give the **model-implied tangency portfolio**.
# This block asks both questions of our yield-space model, on a return target rather than on daily residuals.
#
# **The signal.** The model's exposures are $\beta_{i,t} = Z_{i,t}\Gamma$ in bp of yield per unit factor. Under the
# Gamma in force at date $t$, the mean daily factor realisation on its training dates is $\bar\lambda_t$ (nothing
# after $t$ enters), so the expected yield change over a holding period of $h$ business days is
# $\beta_{i,t}'\bar\lambda_t\,h$ and the expected total return, in bp of price, is
#
# $$\hat\mu_{i,t} = \underbrace{y_{i,t}\,\tfrac{\text{days}}{365}}_{\text{carry}} \;-\; D_{i,t}\,\beta_{i,t}'\bar\lambda_t\,h .$$
#
# **The target.** The realised total return from evaluated prices, $(P_{t+h} - P_t + \text{accrued})/P_t$ in bp,
# with the duration approximation $\text{carry} - D\,\Delta y$ beside it as the data check. Bonds are sorted into
# quintiles on the signal at the first residual date of every month and of every week; the long-short is top minus
# bottom, equal-weighted and DV01-balanced (weights inversely proportional to duration inside each leg, so the two
# legs carry the same rate exposure). Six sorts: the factor premium alone, carry alone, both, the premium within
# duration quintile (which removes the duration tilt a yield-space premium carries into price space), the premium in
# yield space (judged on the realised yield change), and the state-space residual signal as the reference the
# programme has used so far (the yield-space sort's predicted spread is in bp of yield, the others in bp of
# price). **The tangency portfolio** takes the factor means and covariance from the same training
# dates, $w \propto \Sigma^{-1}\mu$, and applies them to the out-of-sample daily factor realisations of record.
#
# **What this can and cannot show.** There are five to six monthly and about twenty-five weekly rebalances. The
# premium estimate itself is shown with its $t$ at every refit: if the factor means are not distinguishable from
# zero on sixty to two hundred training days, a sort on them is a bet on noise and the honest reading is "no
# measurable premium at this sample length", not "no premium". The pre-registered reading is a weekly long-short
# $t$ above 2 with a monotone quintile pattern and the same sign at monthly frequency.

# %%
CELL_T('8b. Factor premia as expected returns: the Kelly-Palhares-Pr [1]')
_fc = [f'f{j+1}' for j in range(CFG.selected_k)]
# ---- (1) the premium available at each refit: in-sample factor realisations under the fold's Gamma, on its training dates only
prem_rows, LAM, SIG = [], {}, {}
for fo in folds:
    gv = pd.Timestamp(fo['train_end']); G = GAMMA_ALIGNED.get(gv)
    if G is None:
        continue
    F = []
    for d in fo['train_dates']:
        g = MODEL_BY_DATE.get(pd.Timestamp(d))
        if g is None:
            continue
        B = g[CHARS].to_numpy(float) @ G; r = g[TARGET].to_numpy(float)
        F.append(np.linalg.solve(B.T @ B + CFG.ridge * np.eye(CFG.selected_k), B.T @ r))
    F = np.asarray(F)
    if len(F) < 20:
        continue
    LAM[fo['fold']] = F.mean(axis=0); SIG[fo['fold']] = np.cov(F.T) if len(F) > CFG.selected_k + 1 else np.eye(CFG.selected_k)
    for j in range(CFG.selected_k):
        prem_rows.append({'fold': fo['fold'], 'Gamma as of': str(gv.date()), 'training days': len(F), 'factor': f'factor{j+1}', 'mean daily realisation (bp)': float(F[:, j].mean()), 'sd (bp)': float(F[:, j].std()), 't': float(np.sqrt(len(F)) * F[:, j].mean() / F[:, j].std()) if F[:, j].std() > 0 else np.nan})
premia = pd.DataFrame(prem_rows)
if len(premia):
    _pt = premia.pivot(index='Gamma as of', columns='factor', values='mean daily realisation (bp)'); _tt = premia.pivot(index='Gamma as of', columns='factor', values='t')
    print('The factor premium as estimated at each refit: mean daily factor realisation on the training dates (bp of yield per day; t in brackets). A yield-space premium is a drift in yields; negative = yields drifting down = positive expected price return on positive-beta bonds:')
    display(pd.DataFrame({c_: [f'{v:+.3f} (t {t_:+.1f})' for v, t_ in zip(_pt[c_], _tt[c_])] for c_ in _pt.columns}, index=_pt.index))
    _nsig = int((premia['t'].abs() >= 2).sum()); print(f'{_nsig} of {len(premia)} refit x factor premium estimates have |t| >= 2.')

# ---- (2) formation dates and holding periods: the first residual date of each month / week, held to the next one
_rd = pd.DatetimeIndex(sorted(resid['date'].unique()))
FORM = {}
for fq in CFG.premia_freqs:
    firsts = pd.Series(_rd).groupby(_rd.to_period('M' if fq == 'monthly' else 'W')).first().to_numpy()
    FORM[fq] = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(firsts[:-1], firsts[1:])]
px = raw_panel[['cusip', 'date', 'closing_price', 'closing_yield', 'cpn', 'modified_duration_lag1']].copy(); px['cusip'] = px['cusip'].astype('string'); px['date'] = to_ns(px['date'])
px = px.dropna(subset=['closing_price', 'closing_yield']).drop_duplicates(['cusip', 'date']).set_index(['date', 'cusip']).sort_index()
_rsig = resid[['cusip', 'date', 'fold'] + beta_cols + (['ssm_signal'] if 'ssm_signal' in resid.columns else [])].copy(); _rsig['cusip'] = _rsig['cusip'].astype('string'); _rsig['date'] = to_ns(_rsig['date'])
SIGNALS = ['factor premium  (-D x beta.lambda x h)', 'carry only', 'factor premium + carry', 'factor premium within duration quintile', 'factor premium in yield space  (-beta.lambda)'] + (['state-space residual signal (reference)'] if 'ssm_signal' in _rsig.columns else [])


def _leg_ret(frame: pd.DataFrame, col: str, q: int, w: str) -> float:
    g = frame[frame['q'] == q]
    if g.empty:
        return np.nan
    wt = (1.0 / g['D'].clip(lower=0.25)) if w == 'dv01' else pd.Series(1.0, index=g.index)
    return float((g[col] * wt).sum() / wt.sum())


def _sort_once(t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[list[dict], pd.DataFrame]:
    u = _rsig[_rsig['date'] == t0]
    if u.empty or t0 not in px.index.get_level_values(0) or t1 not in px.index.get_level_values(0):
        return [], pd.DataFrame()
    p0 = px.xs(t0, level='date'); p1 = px.xs(t1, level='date')[['closing_price', 'closing_yield']].rename(columns={'closing_price': 'P1', 'closing_yield': 'y1'})
    f = u.merge(p0, left_on='cusip', right_index=True, how='inner').merge(p1, left_on='cusip', right_index=True, how='inner')
    days = (t1 - t0).days; bd = int(((_rd >= t0) & (_rd < t1)).sum())
    f['D'] = pd.to_numeric(f['modified_duration_lag1'], errors='coerce').fillna(f['modified_duration_lag1'].median())
    f['ret_bp'] = 1e4 * (f['P1'] - f['closing_price'] + pd.to_numeric(f['cpn'], errors='coerce').fillna(0.0) * days / 365.0) / f['closing_price']
    f['dy_bp'] = 100.0 * (f['y1'] - f['closing_yield']); f['carry_bp'] = f['closing_yield'] * days / 365.0 * 100.0
    f['dur_ret_bp'] = f['carry_bp'] - f['D'] * f['dy_bp']
    n0 = len(f); f = f[(f['closing_price'] > 1) & (f['ret_bp'].abs() <= CFG.premia_max_abs_ret_bp) & (f['dy_bp'].abs() <= 500)]
    lam = np.vstack([LAM.get(int(k_), np.full(CFG.selected_k, np.nan)) for k_ in f['fold']]) if len(f) else np.empty((0, CFG.selected_k))
    f['exp_dy_bp'] = np.einsum('ij,ij->i', f[beta_cols].to_numpy(float), lam) * bd
    f = f[np.isfinite(f['exp_dy_bp'])]
    if len(f) < 100:
        return [], pd.DataFrame()
    sig = {'factor premium  (-D x beta.lambda x h)': -f['D'] * f['exp_dy_bp'], 'carry only': f['carry_bp'], 'factor premium + carry': f['carry_bp'] - f['D'] * f['exp_dy_bp'], 'factor premium in yield space  (-beta.lambda)': -f['exp_dy_bp']}
    _dq = pd.qcut(f['D'].rank(method='first'), 5, labels=False); _fp = sig['factor premium  (-D x beta.lambda x h)']
    sig['factor premium within duration quintile'] = _fp.groupby(_dq).rank(pct=True)
    if 'ssm_signal' in f.columns and f['ssm_signal'].notna().mean() >= 0.5:
        sig['state-space residual signal (reference)'] = -f['ssm_signal']          # bonds without a signal are left out of this sort
    rows, qt = [], []
    for name in SIGNALS:
        x = sig.get(name)
        if x is None or x.std() == 0 or x.notna().sum() < 100:
            continue
        f['q'] = np.nan; f.loc[x.notna(), 'q'] = pd.qcut(x[x.notna()].rank(method='first'), 5, labels=False) + 1
        r = {'signal': name, 't0': t0, 't1': t1, 'bonds': int(f['q'].notna().sum()), 'days': days, 'bdays': bd, 'dropped share': 1 - len(f) / n0}
        r['LS equal-weight (bp)'] = _leg_ret(f, 'ret_bp', 5, 'ew') - _leg_ret(f, 'ret_bp', 1, 'ew'); r['LS DV01-balanced (bp)'] = _leg_ret(f, 'ret_bp', 5, 'dv01') - _leg_ret(f, 'ret_bp', 1, 'dv01')
        r['LS yield change, short minus long (bp)'] = _leg_ret(f, 'dy_bp', 1, 'ew') - _leg_ret(f, 'dy_bp', 5, 'ew')
        r['long leg (bp)'] = _leg_ret(f, 'ret_bp', 5, 'ew'); r['short leg (bp)'] = _leg_ret(f, 'ret_bp', 1, 'ew'); r['universe mean (bp)'] = float(f['ret_bp'].mean())
        _wd = 1.0 / f['D'].clip(lower=0.25); r['market DV01-weighted (bp)'] = float((f['ret_bp'] * _wd).sum() / _wd.sum()); r['market yield change, minus mean dy (bp)'] = float(-f['dy_bp'].mean())
        r['long leg DV01-balanced (bp)'] = _leg_ret(f, 'ret_bp', 5, 'dv01'); r['long leg yield change, minus mean dy (bp)'] = float(-_leg_ret(f, 'dy_bp', 5, 'ew'))
        r['predicted spread (bp)'] = float(x[f['q'] == 5].mean() - x[f['q'] == 1].mean()) if 'within' not in name and 'reference' not in name else np.nan
        rows.append(r)
        qt.append(f.dropna(subset=['q']).groupby('q')['ret_bp'].mean().rename(name).to_frame().T.assign(t0=t0))
    chk = pd.DataFrame({'t0': [t0], 'corr(price return, duration approx)': [float(f['ret_bp'].corr(f['dur_ret_bp']))], 'median |gap| (bp)': [float((f['ret_bp'] - f['dur_ret_bp']).abs().median())], 'bonds': [len(f)], 'dropped share': [1 - len(f) / n0]})
    return rows, (pd.concat(qt) if qt else pd.DataFrame(), chk)


LS, QT, CHK = [], [], []
for fq, pairs in FORM.items():
    for t0, t1 in pairs:
        rows, extra = _sort_once(t0, t1)
        for r in rows:
            r['freq'] = fq; LS.append(r)
        if rows:
            QT.append(extra[0].assign(freq=fq)); CHK.append(extra[1].assign(freq=fq))
ls = pd.DataFrame(LS); chk = pd.concat(CHK, ignore_index=True) if CHK else pd.DataFrame()
if len(ls):
    if len(chk):
        print('Return target check per formation date: correlation of the evaluated-price total return with the duration approximation (carry - D x dy), and the median absolute gap:'); display(chk.groupby('freq')[['corr(price return, duration approx)', 'median |gap| (bp)', 'bonds', 'dropped share']].mean().round(3))
    ann = {'monthly': np.sqrt(12.0), 'weekly': np.sqrt(52.0)}
    sum_rows = []
    for (fq, name), g in ls.groupby(['freq', 'signal'], sort=False):
        for col in ['LS equal-weight (bp)', 'LS DV01-balanced (bp)', 'LS yield change, short minus long (bp)']:
            v = g[col].dropna().to_numpy(float)
            if len(v) < 2:
                continue
            sum_rows.append({'freq': fq, 'signal': name, 'portfolio': col, 'rebalances': len(v), 'mean (bp)': v.mean(), 'sd (bp)': v.std(ddof=1), 't': np.sqrt(len(v)) * v.mean() / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan,
                             'Sharpe (annualised)': ann[fq] * v.mean() / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan, 'hit rate': float((v > 0).mean()), 'predicted spread (bp)': float(g['predicted spread (bp)'].mean()) if g['predicted spread (bp)'].notna().any() else np.nan})
    prem_tab = pd.DataFrame(sum_rows).set_index(['freq', 'signal', 'portfolio'])
    # the benchmark: the equal-weighted universe over the same periods; beta, alpha and capture of each portfolio on it, and the long leg against it
    def _vs_market(v: np.ndarray, m: np.ndarray) -> dict:
        ok = np.isfinite(v) & np.isfinite(m); v, m = v[ok], m[ok]
        if len(v) < 3 or m.std() == 0:
            return {'beta to market': np.nan, 'alpha per period (bp)': np.nan, 'excess over market (bp)': float(np.mean(v - m)) if len(v) else np.nan, 'periods above market': float(np.mean(v > m)) if len(v) else np.nan, 'up capture': np.nan, 'down capture': np.nan}
        b = float(np.cov(v, m, ddof=1)[0, 1] / np.var(m, ddof=1)); a = float(v.mean() - b * m.mean()); up = m > 0; dn = m < 0
        return {'beta to market': b, 'alpha per period (bp)': a, 'excess over market (bp)': float(np.mean(v - m)), 'periods above market': float(np.mean(v > m)), 'up capture': float(v[up].mean() / m[up].mean()) if up.sum() >= 2 and m[up].mean() != 0 else np.nan, 'down capture': float(v[dn].mean() / m[dn].mean()) if dn.sum() >= 2 and m[dn].mean() != 0 else np.nan}
    bm_rows = []
    for fq in CFG.premia_freqs:
        gf = ls[ls['freq'] == fq]
        if gf.empty:
            continue
        mk = gf.groupby('t0')[['universe mean (bp)', 'market DV01-weighted (bp)', 'market yield change, minus mean dy (bp)']].first().sort_index()
        bm_rows.append({'freq': fq, 'signal': 'MARKET: equal-weighted universe, long all bonds', 'portfolio': 'total return (bp)', 'rebalances': len(mk), 'mean (bp)': float(mk['universe mean (bp)'].mean()), 'sd (bp)': float(mk['universe mean (bp)'].std(ddof=1)), 'periods above market': np.nan, 'beta to market': 1.0, 'alpha per period (bp)': 0.0, 'excess over market (bp)': 0.0, 'up capture': 1.0, 'down capture': 1.0})
        bm_rows.append({'freq': fq, 'signal': 'MARKET: DV01-weighted universe', 'portfolio': 'total return (bp)', 'rebalances': len(mk), 'mean (bp)': float(mk['market DV01-weighted (bp)'].mean()), 'sd (bp)': float(mk['market DV01-weighted (bp)'].std(ddof=1)), **_vs_market(mk['market DV01-weighted (bp)'].to_numpy(float), mk['universe mean (bp)'].to_numpy(float))})
        for name, g in gf.groupby('signal', sort=False):
            g = g.set_index('t0').sort_index(); m_ew = g['universe mean (bp)'].to_numpy(float)
            for col, mcol, lab in [('long leg (bp)', 'universe mean (bp)', 'long leg, equal-weight (bp)'), ('long leg DV01-balanced (bp)', 'market DV01-weighted (bp)', 'long leg, DV01-balanced (bp)'), ('LS equal-weight (bp)', 'universe mean (bp)', 'LS equal-weight (bp)'), ('LS DV01-balanced (bp)', 'market DV01-weighted (bp)', 'LS DV01-balanced (bp)'), ('long leg yield change, minus mean dy (bp)', 'market yield change, minus mean dy (bp)', 'long leg, yield space (bp)')]:
                v = g[col].to_numpy(float); m = g[mcol].to_numpy(float)
                bm_rows.append({'freq': fq, 'signal': name, 'portfolio': lab, 'rebalances': int(np.isfinite(v).sum()), 'mean (bp)': float(np.nanmean(v)), 'sd (bp)': float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan, **_vs_market(v, m)})
    bench = pd.DataFrame(bm_rows).set_index(['freq', 'signal', 'portfolio'])
    for fq in CFG.premia_freqs:
        if fq in bench.index.get_level_values(0):
            print(f'Against the market, {fq}: the equal-weighted universe is the benchmark (beta, alpha and capture across rebalances; a long leg that fell less with beta < 1 is defensive, with beta near 1 and alpha > 0 is selection; a long-short is market-neutral only if its beta reads near 0):'); display(bench.loc[fq].round(3))
    for fq in CFG.premia_freqs:
        if fq in prem_tab.index.get_level_values(0):
            print(f'Expected-return quintile long-short, {fq} rebalancing (top quintile minus bottom; bp of price per holding period, yield-change row in bp of yield):'); display(prem_tab.loc[fq].round(3))
    qt_all = pd.concat(QT, ignore_index=False) if QT else pd.DataFrame()
    if len(qt_all):
        qm = qt_all[qt_all['freq'] == CFG.premia_freqs[0]].drop(columns=['t0', 'freq']).groupby(level=0).mean().reindex([n_ for n_ in SIGNALS if n_ in qt_all.index])
        print(f'Mean realised total return by predicted quintile, {CFG.premia_freqs[0]} (bp; a premium reads as a rising row from Q1 to Q5):'); display(qm.round(2))
    for fq in CFG.premia_freqs:
        _ew = ls[(ls['signal'] == 'factor premium + carry') & (ls['freq'] == fq)].set_index('t0')['LS equal-weight (bp)']
        if len(_ew):
            _ew.index = [str(d_.date()) for d_ in _ew.index]; print(f'Long-short of the full expected return (premium + carry) by {fq} rebalance date, equal-weighted (bp of price):'); display(_ew.round(1).to_frame().T)

# ---- (3) the model-implied tangency portfolio of the factors, in yield space (a factor "return" is minus its realisation: yields down = gain)
mve_rows, mve_daily = [], []
_fd = resid.groupby('date')[_fc + ['fold']].first().sort_index()
for k_, lam in LAM.items():
    te = _fd[_fd['fold'] == k_]
    if te.empty:
        continue
    mu = -lam; Sg = SIG[k_]; w_t = np.linalg.solve(Sg + 1e-9 * np.eye(len(mu)), mu); w_t = w_t / np.abs(w_t).sum()
    G_oos = -te[_fc].to_numpy(float)
    ports = {'tangency (model-implied MVE)': G_oos @ w_t, 'equal weight of factors': G_oos.mean(axis=1), **{f'factor{j+1} alone (long the factor)': G_oos[:, j] for j in range(CFG.selected_k)}}
    mve_daily.append(pd.DataFrame(ports, index=te.index).assign(fold=k_))
    mve_rows.append({'fold': k_, **{f'w{j+1}': w_t[j] for j in range(CFG.selected_k)}, 'test days': len(te)})
if mve_daily:
    mve_d = pd.concat(mve_daily); mve_w = pd.DataFrame(mve_rows).set_index('fold')
    # v4.7: the market in yield space (long every covered bond, equal-weighted: minus the mean daily yield change), and each portfolio against it
    _mkt_d = (-resid.groupby('date')['target_bp'].mean()).reindex(mve_d.index); mve_d['MARKET: equal-weighted universe (long all bonds)'] = _mkt_d.to_numpy()
    print('Tangency weights on the factors at each refit (normalised to unit gross; from the training-date mean and covariance):'); display(mve_w.round(3))
    _cols = [c_ for c_ in mve_d.columns if c_ != 'fold']
    mve = pd.DataFrame({'mean (bp/day)': mve_d[_cols].mean(), 'sd (bp/day)': mve_d[_cols].std(), 'Sharpe (annualised, daily)': np.sqrt(252.0) * mve_d[_cols].mean() / mve_d[_cols].std(), 'OOS days': len(mve_d)})
    _mm = mve_d[_cols].groupby(mve_d.index.to_period('M')).sum()
    mve['monthly hit rate'] = (_mm > 0).mean(); mve['months'] = len(_mm)
    _mk = mve_d['MARKET: equal-weighted universe (long all bonds)'].to_numpy(float); _mkm = _mm['MARKET: equal-weighted universe (long all bonds)'].to_numpy(float)
    for c_ in _cols:
        v = mve_d[c_].to_numpy(float); ok = np.isfinite(v) & np.isfinite(_mk)
        b = float(np.cov(v[ok], _mk[ok], ddof=1)[0, 1] / np.var(_mk[ok], ddof=1)) if ok.sum() > 3 and np.var(_mk[ok]) > 0 else np.nan; a = float(v[ok].mean() - b * _mk[ok].mean()) if np.isfinite(b) else np.nan
        resid_ = v[ok] - (a + b * _mk[ok]) if np.isfinite(b) else np.array([np.nan])
        mve.loc[c_, 'beta to market'] = b; mve.loc[c_, 'alpha (bp/day)'] = a; mve.loc[c_, 'alpha t'] = float(np.sqrt(ok.sum()) * a / resid_.std(ddof=1)) if np.isfinite(b) and resid_.std(ddof=1) > 0 else np.nan
        mve.loc[c_, 'information ratio (annualised)'] = float(np.sqrt(252.0) * np.mean(v[ok] - _mk[ok]) / np.std(v[ok] - _mk[ok], ddof=1)) if ok.sum() > 3 and np.std(v[ok] - _mk[ok]) > 0 else np.nan
        mve.loc[c_, 'months above market'] = float(np.mean(_mm[c_].to_numpy(float) > _mkm)); up = _mkm > 0; dn = _mkm < 0
        mve.loc[c_, 'up capture'] = float(_mm[c_].to_numpy(float)[up].mean() / _mkm[up].mean()) if up.sum() >= 2 else np.nan; mve.loc[c_, 'down capture'] = float(_mm[c_].to_numpy(float)[dn].mean() / _mkm[dn].mean()) if dn.sum() >= 2 else np.nan
    print('Out-of-sample factor portfolios in yield space (bp of yield per day) against the market (the equal-weighted universe held long): a positive mean with beta below zero is a short position in a sell-off, and only its alpha counts; the tangency portfolio is the paper\'s model-implied mean-variance portfolio:'); display(mve.round(3))
else:
    mve = pd.DataFrame(); mve_d = pd.DataFrame()

# ---- figures
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
if len(premia):
    _pt.plot.bar(ax=ax[0, 0], color=['#4C72B0', '#55A868', '#C44E52'][:CFG.selected_k]); ax[0, 0].axhline(0, color='k', lw=0.6); ax[0, 0].set_title('The premium estimate at each refit: mean daily factor realisation on the training dates (bp)', fontsize=10); ax[0, 0].set_xlabel(''); ax[0, 0].tick_params(axis='x', rotation=30, labelsize=8)
    for i_, (idx, row) in enumerate(_tt.iterrows()):
        for j_, c_ in enumerate(_tt.columns):
            ax[0, 0].text(i_ + (j_ - 1) * 0.27, _pt.loc[idx, c_], f't {row[c_]:+.1f}', ha='center', va='bottom' if _pt.loc[idx, c_] >= 0 else 'top', fontsize=7)
if len(ls):
    for name, c_ in zip(SIGNALS, ['#4C72B0', '#DD8452', '#2E8B57', '#8172B2', '#937860', '#8C8C8C']):
        g = ls[(ls['freq'] == CFG.premia_freqs[-1]) & (ls['signal'] == name)].sort_values('t0')
        if len(g):
            ax[0, 1].plot(g['t0'], g['LS equal-weight (bp)'].cumsum(), marker='.', color=c_, label=name)
    ax[0, 1].axhline(0, color='k', lw=0.6); ax[0, 1].set_title(f'Cumulative long-short total return, {CFG.premia_freqs[-1]} rebalancing, equal-weighted (bp of price)', fontsize=10); ax[0, 1].legend(fontsize=7)
    if len(qt_all):
        for name, c_ in zip(['factor premium  (-D x beta.lambda x h)', 'factor premium + carry', 'carry only'], ['#4C72B0', '#2E8B57', '#DD8452']):
            if name in qm.index:
                ax[1, 0].plot(qm.columns, qm.loc[name], marker='o', color=c_, label=name)
        _mk_q = ls[ls['freq'] == CFG.premia_freqs[0]].groupby('t0')['universe mean (bp)'].first().mean(); ax[1, 0].axhline(_mk_q, color='k', ls='--', lw=1.0, label=f'market (equal-weighted universe): {_mk_q:+.0f} bp')
        ax[1, 0].set_xlabel('predicted-return quintile (1 = lowest)'); ax[1, 0].set_ylabel('mean realised total return (bp)'); ax[1, 0].set_title(f'Realised return by predicted quintile, {CFG.premia_freqs[0]} (a premium rises left to right)', fontsize=10); ax[1, 0].legend(fontsize=8)
if len(mve_d):
    _cum = mve_d[[c_ for c_ in mve_d.columns if c_ != 'fold']].cumsum()
    for c_, col_ in zip(_cum.columns, ['k', '#8C8C8C', '#4C72B0', '#55A868', '#C44E52', '#DD8452']):
        ax[1, 1].plot(_cum.index, _cum[c_], color=col_, lw=1.6 if c_.startswith('tangency') else (1.8 if c_.startswith('MARKET') else 1.0), ls='--' if c_.startswith('MARKET') else '-', label=c_)
    ax[1, 1].axhline(0, color='k', lw=0.6); ax[1, 1].set_title('Cumulative out-of-sample factor portfolio P&L in yield space (bp; tangency in black, market dashed)', fontsize=10); ax[1, 1].legend(fontsize=7)
fig.suptitle('Factor premia as expected returns: the Kelly-Palhares-Pruitt test on evaluated-price total returns', fontsize=12); plt.tight_layout(); savefig('08b_factor_premia')
record('factor_premia', premium_estimates=premia.round(5).to_dict(orient='records'), long_short=prem_tab.round(5).reset_index().to_dict(orient='records') if len(ls) else [], by_rebalance=ls.drop(columns=[]).round(4).astype({'t0': str, 't1': str}).to_dict(orient='records') if len(ls) else [],
       benchmark=bench.round(5).reset_index().to_dict(orient='records') if len(ls) else [], quintiles=qm.round(4).reset_index().to_dict(orient='records') if len(ls) and len(qt_all) else [], return_check=chk.round(4).astype({'t0': str}).to_dict(orient='records') if len(chk) else [], tangency=mve.round(5).reset_index().to_dict(orient='records') if len(mve) else [], tangency_weights=mve_w.round(4).reset_index().to_dict(orient='records') if mve_daily else [])

# %% [markdown]
# ## 9. Transaction validation (PIT-safe)
#
# Each matched trade is joined to the latest residual dated **strictly before** the trade date. Targets:
# `e_mark = 100 (y_trade − y_prior_mark)` where the prior mark is the close on the residual date, and
# `e_algo = 100 (y_trade − y_algo)`. Inference is Fama-MacBeth over trade dates with a moving-block bootstrap t
# next to it. Three scores: the one-day residual rank, the trailing-mean rank and the state-space signal rank
# (the record). The mark-quality reading regresses |trade − mark| on the residual size and on the state-space
# mark-noise estimate.
#
# **MSRB side convention.** `S` is a dealer **sale** to a customer (the customer buys; dealer offer side, lower
# yield), `P` is a dealer **purchase** from a customer (the customer sells; dealer bid side, higher yield), `D` is
# inter-dealer. The dealer round trip is therefore `P yield − S yield` and is positive.

# %%
CELL_T('9. Transaction validation (PIT-safe) [1]')
def standardize_trades(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr.copy()
    t.columns = [str(c) for c in t.columns]
    n0 = len(t)
    if 'has_algosignal_match' in t.columns:
        flag = t['has_algosignal_match']
        t = t[flag.astype('string').str.lower().isin(['true', '1', 'y', 'yes'])] if flag.dtype == object or str(flag.dtype).startswith('string') else t[flag.astype(bool)]
    dedup_keys = [c for c in ['msrb_trade_id', 'cusip', 'msrb_tradetime', 'msrb_side', 'msrb_quantity'] if c in t.columns]
    if dedup_keys:
        t = t.drop_duplicates(subset=dedup_keys, keep='last')
    print(f'trades: {n0:,} raw -> {len(t):,} after has_algosignal_match filter and trade-id dedup')
    t['cusip'] = t['cusip'].astype('string')
    t['trade_ts'] = to_ns(t['msrb_tradetime'])
    t['signal_ts'] = to_ns(t['signal_ts']) if 'signal_ts' in t else t['trade_ts']
    t['trade_date'] = t['trade_ts'].dt.normalize()
    for c in ['msrb_yield', 'algo_signal_yield', 'msrb_quantity', 'MinuteFromSignal', 'MmdYld', 'dMmdSprdSide', 'last_value', 'msrb_price']:
        t[c] = pd.to_numeric(t[c], errors='coerce') if c in t else np.nan
    t['side'] = t['msrb_side'].astype('string').str.upper().str[0].fillna('U') if 'msrb_side' in t else 'U'
    t['size_bin'] = pd.to_numeric(t['size_bin'], errors='coerce') if 'size_bin' in t else t['msrb_quantity']
    t['size_bin'] = t['size_bin'].fillna(t['msrb_quantity'])
    t['log_size'] = np.log1p(t['msrb_quantity'].clip(lower=0).fillna(0))
    t['e_algo_bp'] = 100.0 * (t['msrb_yield'] - t['algo_signal_yield'])
    if t['signal_ts'].notna().any():
        before = len(t); t = t[(t['trade_ts'] > t['signal_ts']) | t['signal_ts'].isna()]
        print(f'trades after requiring msrb_tradetime > signal_ts: {len(t):,} (dropped {before - len(t):,})')
    t = t.dropna(subset=['cusip', 'trade_ts', 'msrb_yield']).sort_values(['cusip', 'trade_ts'], kind='stable').reset_index(drop=True)
    prev = t.groupby('cusip', observed=True)['trade_ts'].shift(1)
    t['recency_days'] = (t['trade_ts'] - prev).dt.total_seconds() / 86400.0      # bucket definition: days since the prior print before this trade
    # Print features for the level model, cut at the ALGO SIGNAL TIME so the model only sees what the algo could see:
    # the last print strictly before signal_ts (its spread to MMD and side), days since it, and prints in the 20 days before it.
    t['spread_to_mmd_bp'] = 100.0 * (t['msrb_yield'] - t['MmdYld'])
    t['_row'] = np.arange(len(t))
    prints = t[['cusip', 'trade_ts', 'spread_to_mmd_bp', 'side', 'e_algo_bp']].rename(columns={'trade_ts': 'print_ts', 'spread_to_mmd_bp': 'last_print_spread_bp', 'side': 'last_print_side', 'e_algo_bp': 'last_algo_err_bp'}).sort_values(['cusip', 'print_ts'], kind='stable')
    prints['cum_prints'] = prints.groupby('cusip', observed=True).cumcount() + 1.0
    # algo error-correction features: the EWMA of this bond's past algo errors up to and including this print (in prints),
    # side-pooled (the v3 reference C) and on the SAME SIDE of the market (the v41 deployable rule S). Both are vectorised
    # across bonds (one numpy step per print index) instead of one Python ewm call per bond.
    _gid = prints['cusip'].astype('category').cat.codes.to_numpy()
    prints['ewm_algo_err_bp'] = group_ewm_by_order(prints['last_algo_err_bp'].to_numpy(float), _gid, CFG.err_halflife_prints)
    prints = prints.sort_values(['cusip', 'last_print_side', 'print_ts'], kind='stable')
    _gid_s = (prints['cusip'].astype(str) + '|' + prints['last_print_side'].astype(str)).astype('category').cat.codes.to_numpy()
    prints['ewm_side_err_bp'] = group_ewm_by_order(prints['last_algo_err_bp'].to_numpy(float), _gid_s, CFG.err_halflife_prints)
    prints['n_side_prints'] = prints.groupby(['cusip', 'last_print_side'], observed=True).cumcount() + 1.0
    prints = prints.sort_values('print_ts', kind='stable')
    sig = t[['_row', 'cusip', 'signal_ts']].dropna(subset=['signal_ts']).sort_values('signal_ts')
    j1 = pd.merge_asof(sig, prints.drop(columns=['ewm_side_err_bp', 'n_side_prints']), left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    sig_s = t[['_row', 'cusip', 'side', 'signal_ts']].dropna(subset=['signal_ts']).sort_values('signal_ts')
    j3 = pd.merge_asof(sig_s, prints[['cusip', 'last_print_side', 'print_ts', 'ewm_side_err_bp', 'n_side_prints']].rename(columns={'last_print_side': 'side'}), left_on='signal_ts', right_on='print_ts', by=['cusip', 'side'], direction='backward', allow_exact_matches=False).set_index('_row')
    sig20 = sig.assign(signal_ts=sig['signal_ts'] - pd.Timedelta(days=20)).sort_values('signal_ts')
    j2 = pd.merge_asof(sig20, prints[['cusip', 'print_ts', 'cum_prints']], left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    j1 = j1.set_index('_row'); j2 = j2.set_index('_row')
    t = t.set_index('_row')
    t['last_print_spread_bp'] = j1['last_print_spread_bp']; t['last_print_side'] = j1['last_print_side'].fillna('NONE')
    t['last_algo_err_bp'] = j1['last_algo_err_bp']; t['ewm_algo_err_bp'] = j1['ewm_algo_err_bp']; t['n_prior_prints'] = j1['cum_prints'].fillna(0)
    t['ewm_side_err_bp'] = j3['ewm_side_err_bp']; t['n_side_prints'] = j3['n_side_prints'].fillna(0)
    t['side_err_age_days'] = (t['signal_ts'] - j3['print_ts']).dt.total_seconds() / 86400.0
    t['recency_days_sig'] = (t['signal_ts'] - j1['print_ts']).dt.total_seconds() / 86400.0
    t['prints_20d'] = (j1['cum_prints'].fillna(0) - j2['cum_prints'].fillna(0)).clip(lower=0)
    t = t.reset_index(drop=True)
    t['recency_bucket'] = pd.cut(t['recency_days'], bins=[-0.01, 1, 3, 7, 21, np.inf], labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'first_print')
    return t


def join_trades_to_residual(t: pd.DataFrame, res: pd.DataFrame, min_age: int) -> pd.DataFrame:
    extra = [c for c in ['eta_abs', 'ssm_drift', 'ssm_signal', 'fitted_bp', 'beta_cluster'] if c in res.columns]
    score = res[['cusip', 'date', 'pit_residual', 'activity_bucket', 'resid_mean', 'resid_sd', 'target_abs_vol'] + extra + beta_cols].copy()
    score['cusip'] = score['cusip'].astype('string')
    score = score.merge(model[['cusip', 'date', 'modified_duration_lag1', 'mark_unchanged', 'days_since_mark_move']], on=['cusip', 'date'], how='left')
    score['pit_rank'] = score.groupby('date', observed=True)['pit_residual'].rank(pct=True)
    score['level_rank'] = score.groupby('date', observed=True)['resid_mean'].rank(pct=True)
    if 'ssm_signal' in score.columns:
        score['ssm_rank'] = score.groupby('date', observed=True)['ssm_signal'].rank(pct=True)
    # ranks within date x duration quintile: on stress days the global rank carries a curve move the factors missed
    score['dur_cell'] = score.groupby('date', observed=True)['modified_duration_lag1'].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if s.notna().sum() >= 50 else 0)
    score['pit_rank_dur'] = score.groupby(['date', 'dur_cell'], observed=True)['pit_residual'].rank(pct=True)
    score['level_rank_dur'] = score.groupby(['date', 'dur_cell'], observed=True)['resid_mean'].rank(pct=True)
    # cumulative factor-implied and residual paths per bond (for the print-to-print panel): differences between two
    # residual dates give the factor move and the residual move between them
    score = score.sort_values(['cusip', 'date'], kind='stable')
    if 'fitted_bp' in score.columns:
        score['cum_fitted_bp'] = score.groupby('cusip', observed=True)['fitted_bp'].cumsum(); score['cum_resid_bp'] = score.groupby('cusip', observed=True)['pit_residual'].cumsum()
    score['abs_resid_z'] = (score['pit_residual'].abs() / score['target_abs_vol'].replace(0, np.nan)).fillna(score['pit_residual'].abs() / score['pit_residual'].std())
    score = score.rename(columns={'date': 'residual_date'})
    score['residual_date'] = to_ns(score['residual_date'])
    score = score.sort_values('residual_date')
    tt = t.copy()
    tt['trade_date'] = to_ns(tt['trade_date'])
    tt = tt.sort_values('trade_date')
    j = pd.merge_asof(tt, score, left_on='trade_date', right_on='residual_date', by='cusip', direction='backward', allow_exact_matches=(min_age <= 0), tolerance=pd.Timedelta(days=30))
    j = j.dropna(subset=['pit_rank'])
    j['residual_age_days'] = (j['trade_date'] - j['residual_date']).dt.days
    j = j[j['residual_age_days'] >= min_age]
    mark = model[['cusip', 'date', 'closing_yield']].rename(columns={'date': 'residual_date', 'closing_yield': 'y_prior_mark'})
    mark['residual_date'] = to_ns(mark['residual_date'])
    j = j.merge(mark, on=['cusip', 'residual_date'], how='left')
    j['e_mark_bp'] = 100.0 * (j['msrb_yield'] - j['y_prior_mark'])
    j['abs_e_mark_bp'] = j['e_mark_bp'].abs()
    # close on the trade date itself (published after the trade): used only for the evaluator-revision test
    close_td = model[['cusip', 'date', 'closing_yield']].rename(columns={'date': 'trade_date', 'closing_yield': 'y_close_trade_date'})
    close_td['trade_date'] = to_ns(close_td['trade_date'])
    j = j.merge(close_td, on=['cusip', 'trade_date'], how='left')
    j['mark_revision_bp'] = 100.0 * (j['y_close_trade_date'] - j['y_prior_mark'])
    return j.reset_index(drop=True)


def block_bootstrap_t(slopes: np.ndarray, block: int = 5, n_boot: int = 500, seed: int = 0) -> float:
    """Moving-block bootstrap over the daily slopes: the FM t assumes independent dates, but the 125 trade dates share
    27 Gammas and the daily slopes are autocorrelated within a month. Returns mean / bootstrap sd of the mean."""
    n = len(slopes)
    if n < 2 * block:
        return np.nan
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb); idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]; means[i] = slopes[idx].mean()
    sd = means.std()
    return float(slopes.mean() / sd) if sd > 0 else np.nan


def fm_slope(frame: pd.DataFrame, y: str, x: str, controls: list[str], date_col: str = 'trade_date', min_n: int = 30) -> dict:
    """Pooled OLS (descriptive) and Fama-MacBeth over dates (inference) for the slope on x."""
    f = frame.dropna(subset=[y, x]).copy()
    ctrl = [c for c in controls if c in f.columns]
    dummies = pd.get_dummies(f['side'].astype(str), prefix='side', drop_first=True).astype(float) if 'side' in f.columns else pd.DataFrame(index=f.index)
    X = pd.concat([pd.Series(1.0, index=f.index, name='const'), f[x].rename('x'), f[ctrl].apply(pd.to_numeric, errors='coerce').fillna(0.0), dummies], axis=1).to_numpy(float)
    yy = f[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = yy - X @ beta; s2 = r @ r / max(len(yy) - X.shape[1], 1); se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    slopes = []
    for d, g in f.groupby(date_col, observed=True):
        if len(g) < min_n or g[x].std() == 0:
            continue
        Xd = pd.concat([pd.Series(1.0, index=g.index), g[x], g[ctrl].apply(pd.to_numeric, errors='coerce').fillna(0.0)], axis=1).to_numpy(float)
        bd, *_ = np.linalg.lstsq(Xd, g[y].to_numpy(float), rcond=None); slopes.append(bd[1])
    slopes = np.array(slopes)
    return {'n': len(f), 'pooled_beta': beta[1], 'pooled_t': beta[1] / se[1] if se[1] > 0 else np.nan,
            'fm_beta': slopes.mean() if len(slopes) else np.nan, 'fm_t': np.sqrt(len(slopes)) * slopes.mean() / slopes.std() if len(slopes) > 2 and slopes.std() > 0 else np.nan, 'fm_dates': len(slopes),
            'boot_t': block_bootstrap_t(slopes, CFG.block_days, CFG.n_boot, CFG.seed)}


def fm_multi(frame: pd.DataFrame, y: str, xs: list[str], controls: list[str], date_col: str = 'trade_date', min_n: int = 30) -> pd.DataFrame:
    """Fama-MacBeth with several scores entered jointly: one row per score with the FM beta and t."""
    f = frame.dropna(subset=[y] + xs).copy()
    ctrl = [c for c in controls if c in f.columns]
    slopes = []
    for d, g in f.groupby(date_col, observed=True):
        if len(g) < min_n or any(g[x].std() == 0 for x in xs):
            continue
        Xd = pd.concat([pd.Series(1.0, index=g.index), g[xs], g[ctrl].apply(pd.to_numeric, errors='coerce').fillna(0.0)], axis=1).to_numpy(float)
        bd, *_ = np.linalg.lstsq(Xd, g[y].to_numpy(float), rcond=None); slopes.append(bd[1:1 + len(xs)])
    if len(slopes) < 3:
        return pd.DataFrame({'score': xs, 'fm_beta': np.nan, 'fm_t': np.nan, 'boot_t': np.nan, 'fm_dates': len(slopes), 'n': len(f)})
    S = np.array(slopes)
    return pd.DataFrame({'score': xs, 'fm_beta': S.mean(axis=0), 'fm_t': np.sqrt(len(S)) * S.mean(axis=0) / S.std(axis=0), 'boot_t': [block_bootstrap_t(S[:, j], CFG.block_days, CFG.n_boot, CFG.seed) for j in range(S.shape[1])], 'fm_dates': len(S), 'n': len(f)})


def neutralised_fm(frame: pd.DataFrame, y: str, x: str, cells: list[str], date_col: str = 'trade_date') -> dict:
    f = frame.dropna(subset=[y, x]).copy()
    for c in [y, x]:
        f[c + '_n'] = f[c] - f.groupby(cells, observed=True)[c].transform('mean')
    slopes = []
    for d, g in f.groupby(date_col, observed=True):
        if len(g) >= 30 and g[x + '_n'].std() > 0:
            slopes.append(np.polyfit(g[x + '_n'], g[y + '_n'], 1)[0])
    slopes = np.array(slopes)
    return {'n': len(f), 'fm_beta': slopes.mean() if len(slopes) else np.nan, 'fm_t': np.sqrt(len(slopes)) * slopes.mean() / slopes.std() if len(slopes) > 2 and slopes.std() > 0 else np.nan, 'fm_dates': len(slopes)}

# %%
CELL_T('9. Transaction validation (PIT-safe) [2]')
CONTROLS = ['log_size', 'MinuteFromSignal', 'dMmdSprdSide']
if HAS_TRADES:
    trades_all = cached('trades_std', lambda: standardize_trades(trades_raw), deps=[TRADES_FP], code=[standardize_trades, group_ewm_by_order])   # full matched print history
    trades = trades_all[(trades_all['trade_date'] >= pd.Timestamp(CFG.first_oos_date)) & (trades_all['trade_date'] <= pd.Timestamp(CFG.end_date))]
    tv = join_trades_to_residual(trades, resid, CFG.residual_min_age_days)
    print(f'matched trades in OOS window: {len(trades):,} | joined to a prior residual: {len(tv):,} | cusips {tv["cusip"].nunique():,} | trade dates {tv["trade_date"].nunique()} | residual age median {tv["residual_age_days"].median():.0f}d')
    SCORES = [('pit_rank', 'one-day residual rank'), ('level_rank', 'trailing-mean rank'), ('ssm_rank', 'state-space signal rank (record)')]
    rows = []
    for col, label in SCORES:
        d = tv.dropna(subset=[col])
        rows.append({'score': label, 'target': 'trade - prior mark', 'bucket': 'ALL', **fm_slope(d, 'e_mark_bp', col, CONTROLS)})
        rows.append({'score': label, 'target': 'trade - algo quote', 'bucket': 'ALL', **fm_slope(d, 'e_algo_bp', col, CONTROLS)})
    tv['size_cell'] = pd.qcut(tv['size_bin'].rank(method='first'), min(5, tv['size_bin'].nunique()), labels=False, duplicates='drop') if tv['size_bin'].notna().any() else 0
    d = tv.dropna(subset=['ssm_rank'])
    rows.append({'score': 'state-space signal rank (record)', 'target': 'trade - prior mark, neutralised date x side x size', 'bucket': 'ALL', **neutralised_fm(d, 'e_mark_bp', 'ssm_rank', ['trade_date', 'side', 'size_cell'])})
    for b, g in d.groupby('recency_bucket'):
        rows.append({'score': 'state-space signal rank (record)', 'target': 'trade - prior mark, by recency', 'bucket': b, **fm_slope(g, 'e_mark_bp', 'ssm_rank', CONTROLS)})
    for s, g in d.groupby('side'):
        rows.append({'score': 'state-space signal rank (record)', 'target': 'trade - prior mark, by side', 'bucket': f'side={s}', **fm_slope(g, 'e_mark_bp', 'ssm_rank', CONTROLS)})
    _ex = d[~d['residual_date'].isin(DISPERSION_DAYS) & ~d['trade_date'].isin(DISPERSION_DAYS)]
    rows.append({'score': 'state-space signal rank (record)', 'target': 'trade - prior mark, excluding dispersion days', 'bucket': 'ALL', **fm_slope(_ex, 'e_mark_bp', 'ssm_rank', CONTROLS)})
    rows.append({'score': 'state-space signal rank (record)', 'target': 'trade - algo quote, excluding dispersion days', 'bucket': 'ALL', **fm_slope(_ex, 'e_algo_bp', 'ssm_rank', CONTROLS)})
    tx = pd.DataFrame(rows)
    print('Fama-MacBeth over trade dates (boot_t = moving-block bootstrap, 5-day blocks):'); display(tx)
    joint_scores = ['pit_rank', 'level_rank', 'ssm_rank']
    jt = pd.concat([fm_multi(tv, 'e_mark_bp', joint_scores, CONTROLS).assign(target='trade - prior mark'), fm_multi(tv, 'e_algo_bp', joint_scores, CONTROLS).assign(target='trade - algo quote')], ignore_index=True)
    print('Joint Fama-MacBeth: the three scores entered together'); display(jt.round(4))
    # the mark-quality reading: how far prints sit from the mark, on |residual|/vol and on the state-space mark-noise estimate
    sp_rows = [{'test': '|trade - prior mark| on |residual| / vol', **fm_slope(tv, 'abs_e_mark_bp', 'abs_resid_z', CONTROLS)}]
    if 'eta_abs' in tv.columns and tv['eta_abs'].notna().mean() > 0.3:
        sp_rows.append({'test': '|trade - prior mark| on state-space mark-noise estimate |eta|', **fm_slope(tv.dropna(subset=['eta_abs']), 'abs_e_mark_bp', 'eta_abs', CONTROLS)})
    day = tv[tv['side'].isin(['P', 'S'])].groupby(['cusip', 'trade_date', 'side'], observed=True).agg(y=('msrb_yield', 'mean'), z=('abs_resid_z', 'mean'), log_size=('log_size', 'mean')).unstack('side')
    if ('y', 'P') in day.columns and ('y', 'S') in day.columns:
        rt = pd.DataFrame({'round_trip_bp': 100.0 * (day[('y', 'P')] - day[('y', 'S')]), 'abs_resid_z': day[('z', 'P')], 'log_size': day[('log_size', 'P')]}).dropna().reset_index(); rt['side'] = 'RT'
        if len(rt) > 200:
            sp_rows.append({'test': f'dealer round trip (P - S yield, same bond-day, {len(rt):,} pairs) on |residual| / vol', **fm_slope(rt, 'round_trip_bp', 'abs_resid_z', ['log_size'])})
            print(f'median dealer round trip {rt["round_trip_bp"].median():.1f} bp')
    spread_tx = pd.DataFrame(sp_rows)
    print('Mark-quality reading:'); display(spread_tx)
    record('transaction_validation', table=tx.round(4).to_dict(orient='records'), joint=jt.round(4).to_dict(orient='records'), mark_quality=spread_tx.round(4).to_dict(orient='records'), trades=len(tv), trade_dates=int(tv['trade_date'].nunique()))
else:
    tv = pd.DataFrame()
    print('No matched trades in the pipeline store; Sections 9 to 11 are skipped.')

# %% [markdown]
# ### 9b. Trade size and side: how large and small prints behave
#
# Size was missing from the framework. Here every matched print is placed in the desk's quantity bins (0-5K,
# 5K-10K, 10K-15K, 15K-25K, 25K-50K, 50K-100K, 100K-250K, >250K par) and cut by MSRB side: where the volume is,
# how far prints sit from the quote and from the prior mark (absolute and signed, daily Fama-MacBeth means), the
# dealer round trip by size, and the same by issuer industry when the pipeline carries it. Signed errors are in
# yield: on P (dealer buys) a positive error means the print's yield was above the quote's, so the quote's price
# was above the print and the bid was aggressive; on S the sign reverses.

# %%
CELL_T('9b. Trade size and side: how large and small prints behave [1]')
if HAS_TRADES and not tv.empty:
    tv['qty_group'] = qty_group(tv['msrb_quantity'])
    if 'industry_bucket' in model.columns:
        _im = model[['cusip', 'date', 'industry_bucket']].rename(columns={'date': 'residual_date'}); _im['residual_date'] = to_ns(_im['residual_date']); _im['cusip'] = _im['cusip'].astype('string')
        tv['residual_date'] = to_ns(tv['residual_date']); tv['cusip'] = tv['cusip'].astype('string')
        tv = tv.merge(_im, on=['cusip', 'residual_date'], how='left')

    def side_size_table(frame: pd.DataFrame, by: str, order: list | None = None) -> pd.DataFrame:
        g = frame.groupby([by, 'side'], observed=True)
        out = pd.DataFrame({'trades': g.size(), 'par ($mm)': g['msrb_quantity'].sum() / 1e6,
                            '|trade - algo| (bp)': g['e_algo_bp'].apply(lambda x: x.abs().mean()), '|trade - mark| (bp)': g['e_mark_bp'].apply(lambda x: x.abs().mean())})
        sg = frame.groupby([by, 'side', 'trade_date'], observed=True)['e_algo_bp'].mean().groupby(level=[0, 1])
        out['signed algo error, FM mean (bp)'] = sg.mean(); out['FM t'] = np.sqrt(sg.count()) * sg.mean() / sg.std().replace(0, np.nan)
        out['share of trades'] = out['trades'] / out['trades'].sum()
        out = out.unstack('side')
        return out.reindex(order) if order else out

    sz = side_size_table(tv, 'qty_group', QTY_LABELS)
    print('Trades by desk quantity bin and MSRB side (P dealer buys, S dealer sells, D inter-dealer):'); display(sz.round(2))
    _rt = tv[tv['side'].isin(['P', 'S'])].groupby(['cusip', 'trade_date', 'side'], observed=True).agg(y=('msrb_yield', 'mean'), q=('msrb_quantity', 'mean')).unstack('side')
    if ('y', 'P') in _rt.columns and ('y', 'S') in _rt.columns:
        _rt2 = pd.DataFrame({'round_trip_bp': 100.0 * (_rt[('y', 'P')] - _rt[('y', 'S')]), 'qty_group': qty_group(_rt[('q', 'P')])}).dropna()
        rt_size = _rt2.groupby('qty_group', observed=True)['round_trip_bp'].agg(['count', 'median', 'mean']).reindex(QTY_LABELS).dropna(how='all')
        print('Dealer round trip (P yield - S yield, same bond-day) by the quantity bin of the purchase:'); display(rt_size.round(2))
    else:
        rt_size = pd.DataFrame()
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    sz['share of trades'].reindex(QTY_LABELS).plot.bar(ax=ax[0], stacked=True, width=0.8); ax[0].set_title('Share of matched prints by quantity bin and side', fontsize=10); ax[0].set_xlabel(''); ax[0].legend(fontsize=8); ax[0].tick_params(axis='x', rotation=30)
    _hm = sz['|trade - algo| (bp)'].reindex(QTY_LABELS)[[c for c in ['P', 'S', 'D'] if c in sz['|trade - algo| (bp)'].columns]]
    im = ax[1].imshow(_hm.to_numpy(float), cmap='YlOrRd', aspect='auto'); ax[1].set_xticks(range(_hm.shape[1])); ax[1].set_xticklabels(_hm.columns); ax[1].set_yticks(range(len(_hm))); ax[1].set_yticklabels(_hm.index, fontsize=8); ax[1].set_title('|trade - algo quote| (bp) by quantity bin and side', fontsize=10); ax[1].grid(False); plt.colorbar(im, ax=ax[1], fraction=0.046)
    for i in range(_hm.shape[0]):
        for j in range(_hm.shape[1]):
            v = _hm.iloc[i, j]
            if np.isfinite(v):
                ax[1].text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8)
    _sg = sz['signed algo error, FM mean (bp)'].reindex(QTY_LABELS)
    for c in [x for x in ['P', 'S', 'D'] if x in _sg.columns]:
        ax[2].plot(range(len(_sg)), _sg[c], marker='o', label={'P': 'P dealer buys', 'S': 'S dealer sells', 'D': 'D inter-dealer'}[c])
    ax[2].axhline(0, color='k', lw=0.6); ax[2].set_xticks(range(len(_sg))); ax[2].set_xticklabels(_sg.index, rotation=30, fontsize=8); ax[2].set_title('Signed algo error by quantity bin and side (FM daily mean, bp; + = print yield above quote)', fontsize=9); ax[2].legend(fontsize=8)
    savefig('09b_size_and_side')
    if 'industry_bucket' in tv.columns and tv['industry_bucket'].notna().any():
        ind_tab = side_size_table(tv.dropna(subset=['industry_bucket']), 'industry_bucket')
        print('Trades by issuer industry and MSRB side:'); display(ind_tab.round(2))
        record('size_side_eda', by_industry=flat_records(ind_tab))
    record('size_side_eda', by_qty_group=flat_records(sz), round_trip_by_size=flat_records(rt_size) if len(rt_size) else [])
else:
    print('Section 9b skipped: needs matched trades.')

# %% [markdown]
# ### 9c. Where prints sit relative to our quote and to the evaluation
#
# Two distances per print, both oriented so that positive is good for the dealer. *Print minus quote*, oriented
# by side: positive means the print landed on the aggressive side of our quote (the customer sold below our bid
# price or bought above our offer price), so we would have won that trade at our quote. *Print minus same-day
# evaluation*, oriented: positive is the edge over the closing evaluation the dealer who did the trade earned.
# The first distance is the fill proxy Section 11 is built on; the second is the market's realised half-spread
# against the evaluation, by side and size. Read the two panels together: how often prints cross our quote, and
# how far from the evaluation the market actually trades.

# %%
CELL_T('9c. Where prints sit relative to our quote and to the evalua [1]')
if HAS_TRADES and not tv.empty and 'y_close_trade_date' in tv.columns:
    ps = tv[tv['side'].isin(['P', 'S'])].copy(); ps['s'] = np.where(ps['side'] == 'P', 1.0, -1.0)
    ps['o_quote'] = ps['s'] * ps['e_algo_bp']                                            # + = print on the aggressive side of our quote: we would have filled
    ps['o_eval'] = ps['s'] * 100.0 * (ps['msrb_yield'] - ps['y_close_trade_date'])       # + = the dealer's edge over the same-day evaluation on the actual print

    def where_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby([by, 'side'], observed=True)
        out = pd.DataFrame({'prints': g.size(), 'print - quote, oriented median (bp)': g['o_quote'].median(), 'share crossing our quote': g['o_quote'].apply(lambda x: float((x >= 0).mean())),
                            'dealer edge vs evaluation, median (bp)': g['o_eval'].median(), 'share of prints beyond the evaluation': g['o_eval'].apply(lambda x: float((x >= 0).mean()))})
        return out.unstack('side')

    wt_all = where_table(ps.assign(all='ALL'), 'all'); wt_size = where_table(ps, 'qty_group').reindex(QTY_LABELS)
    print('Where prints sit (P = dealer buys at our bid, S = dealer sells at our offer):'); display(wt_all.round(3))
    print('By desk quantity bin:'); display(wt_size.round(2))
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for i, sd_ in enumerate(['P', 'S']):
        g = ps[ps['side'] == sd_]; lab = {'P': 'P dealer buys (our bid)', 'S': 'S dealer sells (our offer)'}[sd_]
        a = ax[i, 0]; x = g['o_quote'].clip(-40, 40).dropna()
        a.hist(x, bins=120, color='#4C72B0', alpha=0.85); a.axvline(0, color='k', lw=0.8); a.axvspan(0, 40, color='#2E8B57', alpha=0.08)
        a.set_title(f'{lab}: print minus our quote, oriented (bp)', fontsize=10); a.set_xlabel('bp (positive = print crossed our quote, we would have filled)')
        a.text(0.98, 0.92, f'would have filled {float((g["o_quote"] >= 0).mean()):.0%} of prints\nmedian {g["o_quote"].median():+.1f} bp', transform=a.transAxes, ha='right', va='top', fontsize=9, bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
        a = ax[i, 1]; x = g['o_eval'].clip(-60, 60).dropna()
        a.hist(x, bins=120, color='#DD8452', alpha=0.85); a.axvline(0, color='k', lw=0.8); a.axvline(g['o_eval'].median(), color='#C44E52', lw=1.2, ls='--')
        a.set_title(f'{lab}: the dealer\'s edge over the same-day evaluation on the print (bp)', fontsize=10); a.set_xlabel('bp (positive = the dealer traded on the good side of the evaluation)')
        a.text(0.98, 0.92, f'median {g["o_eval"].median():+.1f} bp\n{float((g["o_eval"] >= 0).mean()):.0%} of prints beyond the evaluation', transform=a.transAxes, ha='right', va='top', fontsize=9, bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'})
    fig.suptitle('Where prints land: relative to our quote (left, the fill proxy) and relative to the evaluation (right, the market\'s realised half-spread)', fontsize=11); plt.tight_layout(); savefig('09c_prints_vs_quote_and_eval')
    record('prints_vs_quote_eval', overall=flat_records(wt_all), by_size=flat_records(wt_size))
else:
    print('Section 9c skipped: needs matched trades with the trade-date close.')

# %% [markdown]
# ## 10. Quote rules: the same-side memory, the ladder, the re-opened trials, the side x size intercepts, the level model
#
# The v5 print-to-print panel found that the algo's pricing error on a bond persists from one print to the next
# (coefficient 0.58), a larger lever than anything the factor residual offered against the algo quote. The v41 run
# then found that the memory is **side-specific**: an EWMA over the bond's prior prints on the same side of the
# market doubles the gain of the side-pooled EWMA and turns the short end from a loss into a gain, because the
# dealer round trip in short bonds applies a sale's error with the wrong sign to the next purchase. The universe
# side intercept, the universe-offset state-space memory, cluster pooling, within-cluster relative value and the
# per-cluster factor loading all lost to it and are closed (paper, Appendix A).
#
# *Motivation.* Consecutive matched prints of the same bond: the algo error at $t_1$ regressed on the algo error
# at $t_0$, by gap, with the factor and residual moves between them as controls.
#
# *Features, cut at the algo signal time.* For every trade: the algo error at the last matched print strictly
# before `signal_ts`, its age, its side, the EWMA of past errors (half-life `err_halflife_prints` prints) pooled
# across sides and on the trade's own side, and the number of prior prints. Nothing after the signal time is used.
#
# *Mids, walk-forward by month, every coefficient on prior months.* (A) the algo quote; (C) algo + $\rho \times$
# side-pooled EWMA, the reference; (S) algo + $\rho_s \times$ same-side EWMA, the deployable rule, falling back
# to C when the bond has no same-side history; (K) the per-side **state-space memory**: a local level per (bond,
# side) with daily decay $\phi$ and diffusion $q$ observed under noise $r$ on the bond's irregular print clock,
# parameters by maximum likelihood on prints before the test month, the posterior after the last same-side print
# decayed to the trade (S is its steady state with $\phi = 1$); (D) a gradient-boosted level model on the trade's
# spread to MMD with the print and algo-derived features plus the error features including the same-side memory;
# (D$^-$) the same without the error features. Scored against the print by error age, side and duration band with
# Fama-MacBeth and block-bootstrap t. The level models and the state-space memory are cached on the trade and
# residual keys.
#
# **v4.0 additions.** Four trials that v3.6 closed on MAE are re-opened because Section 11 re-ranks every rule in
# dollars and a rule that loses on MAE can win on P&L: (B) the last error scaled by an age-dependent persistence,
# (E) a per-side intercept, (F) the side intercept plus rho x the side-pooled EWMA (the "revised third term"), and
# (G) the direct factor-beta correction. Two rules are new, built from the v51 finding that the algo's error has a
# strong slope in trade size on every side: (AQ) the algo quote plus a side x quantity-bin intercept, and (SQ) the
# same-side memory plus that intercept estimated on the error S leaves. Intercepts are Fama-MacBeth daily means on
# prior months, shrunk to the side mean where a cell is thin. The MAE tables are kept to one scoreboard and one
# by-side table; the full breakdowns live in the registry.

# %%
CELL_T('10. Quote rules: the same-side memory, the ladder, the re-op [1]')
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

if HAS_TRADES and not tv.empty and 'cum_fitted_bp' in tv.columns:
    # ---- motivation: persistence of the algo error print to print
    _t = tv.sort_values(['cusip', 'trade_ts'], kind='stable').copy(); _g = _t.groupby('cusip', observed=True)
    for c in ['trade_ts', 'e_algo_bp', 'residual_date', 'cum_fitted_bp', 'cum_resid_bp', 'side']:
        _t[f'{c}_0'] = _g[c].shift(1)
    p2p = _t.dropna(subset=['trade_ts_0', 'cum_fitted_bp_0']).copy()
    p2p['gap_days'] = (p2p['trade_ts'] - p2p['trade_ts_0']).dt.total_seconds() / 86400.0
    p2p = p2p[(p2p['gap_days'] > 0) & (p2p['gap_days'] <= 30) & (p2p['residual_date'] > p2p['residual_date_0'])].copy()
    p2p['d_fit_bp'] = p2p['cum_fitted_bp'] - p2p['cum_fitted_bp_0']; p2p['d_res_bp'] = p2p['cum_resid_bp'] - p2p['cum_resid_bp_0']
    p2p['gap_bucket'] = pd.cut(p2p['gap_days'], bins=[0, 1, 3, 7, 14, 30], labels=['<=1d', '1-3d', '3-7d', '7-14d', '14-30d']).astype(str)
    for s_ in ['P', 'S']:
        p2p[f'side1_{s_}'] = (p2p['side'] == s_).astype(float); p2p[f'side0_{s_}'] = (p2p['side_0'] == s_).astype(float)
    P2P_CTRL = ['log_size', 'side1_P', 'side1_S', 'side0_P', 'side0_S']
    pers = fm_multi(p2p, 'e_algo_bp', ['e_algo_bp_0', 'd_fit_bp', 'd_res_bp'], P2P_CTRL)
    print(f'consecutive matched prints: {len(p2p):,} pairs, median gap {p2p["gap_days"].median():.1f} days'); print('Algo error at t1 on its own lag (persistence), the factor move and the residual move between prints:'); display(pers.round(4))
    pers_gap = pd.DataFrame([{'gap_bucket': b, 'pairs': len(g), **fm_multi(g, 'e_algo_bp', ['e_algo_bp_0'], P2P_CTRL).iloc[0][['fm_beta', 'fm_t', 'boot_t']].to_dict()} for b, g in p2p.groupby('gap_bucket') if len(g) >= 500]).set_index('gap_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-14d', '14-30d']).dropna(how='all')
    pers_same = pd.DataFrame([{'pair': k, 'pairs': len(g), **fm_multi(g, 'e_algo_bp', ['e_algo_bp_0'], ['log_size']).iloc[0][['fm_beta', 'fm_t']].to_dict()} for k, g in p2p.assign(pair=np.where(p2p['side'] == p2p['side_0'], 'same side', 'opposite side')).groupby('pair')]).set_index('pair')
    print('Persistence by gap:'); display(pers_gap.round(3)); print('Persistence when the two prints are on the same vs opposite MSRB side:'); display(pers_same.round(3))

    # ---- the walk-forward error-correction study on the trade frame
    ec = tv.dropna(subset=['e_algo_bp']).copy()
    ec['_id'] = np.arange(len(ec))
    ec['has_err'] = ec['last_algo_err_bp'].notna(); ec['has_side_err'] = ec['ewm_side_err_bp'].notna()
    ec['err_age'] = ec['recency_days_sig']
    ec['age_bucket'] = pd.cut(ec['err_age'], bins=list(CFG.age_bins_days), labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'no prior print')
    ec['month'] = ec['trade_date'].dt.to_period('M')
    ec['spread_bp'] = 100.0 * (ec['msrb_yield'] - ec['MmdYld'])
    ec['dur_band'] = pd.cut(ec['modified_duration_lag1'], bins=[-1, 1, 11, 100], labels=['<1y', '1-11y', '>11y']).astype(str)
    # v3.9 conditioning variables for the interaction ladder
    ec['qty_group'] = qty_group(ec['msrb_quantity'])
    ec['side_age_bucket'] = pd.cut(ec['side_err_age_days'], bins=list(CFG.age_bins_days), labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'no same-side print') if 'side_err_age_days' in ec.columns else 'NA'
    ec['abs_m'] = ec['ssm_drift'].abs().clip(upper=20.0) if 'ssm_drift' in ec.columns else np.nan
    ec['side_cluster'] = ec['side'].astype(str) + '|' + ec['beta_cluster'].astype('Int64').astype(str)
    print(f'trades with a prior matched print before the signal time: {ec["has_err"].mean():.1%} of {len(ec):,}; with a prior print on the same side: {ec["has_side_err"].mean():.1%}')
    feat_panel = model[['cusip', 'date', 'rating_score', 'years_to_worst', 'extension', 'cpn', 'closing_yield_lag1', 'state_others', 'call_structure', 'state_bucket', 'is_callable'] + [c for c in ['industry_bucket', 'trade_size_lag', 'print_freq_lag'] if c in model.columns] + [c for c in CHARS if c != 'market_fv']].rename(columns={'date': 'residual_date'})
    feat_panel = feat_panel[[c for c in feat_panel.columns if c in ('cusip', 'residual_date') or c not in ec.columns]]
    feat_panel['residual_date'] = to_ns(feat_panel['residual_date']); ec['residual_date'] = to_ns(ec['residual_date'])
    ec = ec.merge(feat_panel, on=['cusip', 'residual_date'], how='left')
    ec['recency_days_c'] = ec['recency_days_sig'].clip(upper=60).fillna(60)
    for s_ in ['D', 'P', 'S']:
        ec[f'side_{s_}'] = (ec['side'] == s_).astype(float); ec[f'last_side_{s_}'] = (ec['last_print_side'] == s_).astype(float)
    ec['same_side_as_last'] = (ec['side'] == ec['last_print_side']).astype(float)
    PRINT_FEATURES = ['last_print_spread_bp', 'prints_20d', 'last_side_D', 'last_side_P', 'last_side_S']
    ALGO_DERIVED = ['dMmdSprdSide', 'MinuteFromSignal']
    ERR_FEATURES = ['last_algo_err_bp', 'ewm_algo_err_bp', 'ewm_side_err_bp', 'n_side_prints', 'err_age_c', 'n_prior_prints', 'same_side_as_last']
    ec['err_age_c'] = ec['err_age'].clip(upper=60).fillna(60)
    BASE_FEATURES = PRINT_FEATURES + ALGO_DERIVED + ['rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1', 'recency_days_c', 'log_size', 'side_D', 'side_P', 'side_S', 'state_others'] + beta_cols + [c for c in CHARS if c != 'market_fv']
    BASE_FEATURES = [c for c in dict.fromkeys(BASE_FEATURES) if c in ec.columns]
    FULL_FEATURES = BASE_FEATURES + [c for c in ERR_FEATURES if c in ec.columns]
    EC_FP = frame_fingerprint(ec, ['e_algo_bp', 'ewm_side_err_bp', 'spread_bp']) + f'|{STAGE_KEYS.get("trades_std")}|{STAGE_KEYS.get("ssm_pooled")}'
    MONTHS_EC = sorted(ec['month'].unique())

    def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
        return [c for c in features if frame[c].astype(float).nunique(dropna=True) >= 2]

    def fit_gbm(tr: pd.DataFrame, te: pd.DataFrame, features: list[str], target: str, n_trees: int):
        feats = usable_features(tr, features); med = tr[feats].median()
        Xtr, Xte = tr[feats].astype(float).fillna(med), te[feats].astype(float).fillna(med)
        ytr = tr[target].clip(*tr[target].quantile([0.005, 0.995]))
        if HAS_LGB:
            mdl = lgb.LGBMRegressor(objective='l1', n_estimators=n_trees, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=CFG.seed, verbose=-1).fit(Xtr, ytr)
            imp = pd.Series(mdl.feature_importances_, index=feats)
        else:
            mdl = HistGradientBoostingRegressor(loss='absolute_error', max_iter=max(100, n_trees * 2 // 3), learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, random_state=CFG.seed).fit(Xtr, ytr); imp = None
        return mdl.predict(Xte), imp

    # Diagnostic of the algo's own third term: the quote is MMD + side-specific spread + a last-value adjustment. Regressing
    # the error on that adjustment says whether it is under- or over-sized (v31: -0.037, right-sized), and the correlation
    # with our memory features says the persistence is information the quote does not use.
    if 'last_value' in ec.columns and ec['last_value'].notna().mean() > 0.3:
        _lv = ec.dropna(subset=['last_value'])
        lv_rows = [{'bucket': 'ALL', **fm_slope(_lv, 'e_algo_bp', 'last_value', ['log_size'])}] + [{'bucket': f'side={s_}', **fm_slope(g, 'e_algo_bp', 'last_value', ['log_size'])} for s_, g in _lv.groupby('side')]
        lv_diag = pd.DataFrame(lv_rows).set_index('bucket')
        print("Algo error on the algo's own last-value adjustment (positive = under-reaction, negative = over-reaction):"); display(lv_diag[['n', 'fm_beta', 'fm_t', 'boot_t']].round(3))
        print(f"corr(same-side EWMA error, last-value adjustment) = {_lv['ewm_side_err_bp'].corr(_lv['last_value']):+.3f}; corr(side-pooled EWMA, adjustment) = {_lv['ewm_algo_err_bp'].corr(_lv['last_value']):+.3f}")
        record('error_correction', last_value_diagnostic=lv_diag.round(4).reset_index().to_dict(orient='records'), corr_side_err_last_value=float(_lv['ewm_side_err_bp'].corr(_lv['last_value'])))

    # ---- (D, D-) level models, cached: predictions for every test month keyed by _id
    def _level_models() -> pd.DataFrame:
        parts, imps = [], []
        for m in MONTHS_EC[1:]:
            tr, te = ec[ec['month'] < m], ec[ec['month'] == m]
            if len(tr) < 5_000 or te.empty:
                continue
            t0 = time.perf_counter()
            yD, impD = fit_gbm(tr, te, FULL_FEATURES, 'spread_bp', CFG.level_model_trees)
            yDm, _ = fit_gbm(tr, te, BASE_FEATURES, 'spread_bp', CFG.level_model_trees)
            parts.append(pd.DataFrame({'_id': te['_id'].to_numpy(), 'yD': yD, 'yDm': yDm}))
            if impD is not None:
                imps.append(impD.rename(str(m)))
            print(f'  level models {m}: train {len(tr):,} | test {len(te):,} | {time.perf_counter() - t0:.0f}s')
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['_id', 'yD', 'yDm'])
        if imps:
            imp = pd.concat(imps, axis=1).mean(axis=1)
            out = out.merge(pd.DataFrame({'_id': -1 - np.arange(len(imp)), 'feature': imp.index, 'importance': imp.to_numpy()}), on='_id', how='outer')   # importances ride along with negative ids
        return out

    LVL = cached('level_models', _level_models, deps=[EC_FP, FULL_FEATURES, BASE_FEATURES], code=[fit_gbm, usable_features])
    lvl_pred = LVL[LVL['_id'] >= 0].set_index('_id'); imp_mean = LVL[LVL['_id'] < 0].set_index('feature')['importance'].dropna() if 'feature' in LVL.columns else pd.Series(dtype=float)

    # ---- (K) per-side state-space memory on the full print history, cached: the posterior after the last same-side print
    #      before each trade's signal time, decayed to the trade, for every fold's parameters
    def _side_state_space_memory() -> pd.DataFrame:
        from scipy.optimize import minimize  # noqa: E402
        pr = trades_all[['cusip', 'side', 'trade_ts', 'trade_date', 'e_algo_bp']].dropna(subset=['e_algo_bp']).copy()
        pr['cusip'] = pr['cusip'].astype('string'); pr['trade_ts'] = to_ns(pr['trade_ts'])
        pr = pr.sort_values(['cusip', 'side', 'trade_ts'], kind='stable').reset_index(drop=True)
        pr['month'] = to_ns(pr['trade_date']).dt.to_period('M')
        lo, hi = pr['e_algo_bp'].quantile([0.005, 0.995]); e_kf = pr['e_algo_bp'].clip(lo, hi).to_numpy(float)
        gid = (pr['cusip'].astype(str) + '|' + pr['side'].astype(str)).astype('category').cat.codes.to_numpy(); ng = int(gid.max()) + 1
        start = np.r_[True, gid[1:] != gid[:-1]]; k = np.arange(len(pr)) - np.flatnonzero(start)[np.cumsum(start) - 1]
        gap = pr.groupby(['cusip', 'side'], observed=True)['trade_ts'].diff().dt.total_seconds().div(86400.0).fillna(0.0).clip(lower=0.0).to_numpy(float)
        pr_month = pr['month'].to_numpy()
        _ord = np.argsort(k, kind='stable')

        def layout(mask):
            o = _ord if mask is None else _ord[mask[_ord]]
            ks = k[o]; st = np.searchsorted(ks, np.arange(ks.max() + 2)) if len(ks) else np.array([0, 0]); return o, st

        full_layout = layout(None)

        def kf_pass(theta, lay=None, want=False):
            o, st = full_layout if lay is None else lay
            phi = 1.0 / (1.0 + np.exp(-theta[0])); q = np.exp(theta[1]); r = np.exp(theta[2])
            level = np.zeros(ng); var = np.full(ng, 10.0 * r); ll = 0.0
            post = np.full(len(pr), np.nan) if want else None
            for j in range(len(st) - 1):
                rows = o[st[j]:st[j + 1]]
                if not len(rows):
                    continue
                g = gid[rows]; d = gap[rows]; e = e_kf[rows]; dec = phi ** d
                m_pred = dec * level[g]; P_pred = dec * dec * var[g] + q * d; S = P_pred + r; v = e - m_pred; K = P_pred / S
                level[g] = m_pred + K * v; var[g] = (1.0 - K) * P_pred
                if want:
                    post[rows] = level[g]
                else:
                    ll += float(np.sum(-0.5 * (np.log(S) + v * v / S)))
            return post if want else -ll

        cnt = np.bincount(gid, minlength=ng); eligible = np.flatnonzero((cnt >= 5) & (cnt <= 500)); rng = np.random.default_rng(CFG.seed)
        sub = np.zeros(ng, bool); sub[rng.choice(eligible, size=min(4000, len(eligible)), replace=False)] = True
        sig_all = ec[['_id', 'cusip', 'side', 'signal_ts']].copy(); sig_all['cusip'] = sig_all['cusip'].astype('string'); sig_all['signal_ts'] = to_ns(sig_all['signal_ts']); sig_all = sig_all.dropna(subset=['signal_ts']).sort_values('signal_ts')
        parts = []
        for m in MONTHS_EC[1:]:
            t0 = time.perf_counter()
            mask = sub[gid] & (pr_month < m)
            if mask.sum() < 5_000:
                continue
            x0 = np.array([np.log(0.97 / 0.03), 0.0, np.log(max(float(np.var(e_kf[mask])), 1.0))])
            res = minimize(kf_pass, x0, args=(layout(mask), False), method='Nelder-Mead', options={'maxiter': 80, 'xatol': 1e-3, 'fatol': 1e-3})
            theta = res.x; phi = 1.0 / (1.0 + np.exp(-theta[0]))
            post = kf_pass(theta, None, True)
            pk = pd.DataFrame({'cusip': pr['cusip'], 'side': pr['side'], 'trade_ts': pr['trade_ts'], 'post': post}).sort_values('trade_ts')
            j = pd.merge_asof(sig_all, pk, left_on='signal_ts', right_on='trade_ts', by=['cusip', 'side'], direction='backward', allow_exact_matches=False)
            gap_sig = (j['signal_ts'] - j['trade_ts']).dt.total_seconds() / 86400.0
            parts.append(pd.DataFrame({'_id': j['_id'].to_numpy(), 'month': str(m), 'kf_pred_bp': (j['post'] * (phi ** gap_sig)).to_numpy(), 'phi_daily': phi, 'q': float(np.exp(theta[1])), 'r': float(np.exp(theta[2]))}))
            print(f'  state-space memory {m}: phi/day {phi:.4f} (half-life {np.log(0.5) / np.log(phi) if phi < 1 else np.inf:.0f}d), q {np.exp(theta[1]):.2f}, r {np.exp(theta[2]):.1f} | {time.perf_counter() - t0:.0f}s')
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['_id', 'month', 'kf_pred_bp', 'phi_daily', 'q', 'r'])

    KFM = cached('side_state_space_memory', _side_state_space_memory, deps=[EC_FP, TRADES_FP])
    KF_BY_MONTH = {m: g.set_index('_id') for m, g in KFM.groupby('month')} if len(KFM) else {}

    # ---- the fold loop: S and K coefficients on prior months, every mid on the test month
    parts, rho_rows = [], []; BIAS_PATH = {}
    for m in MONTHS_EC[1:]:
        tr, te = ec[ec['month'] < m], ec[ec['month'] == m].copy()
        if len(tr) < 5_000 or te.empty:
            continue
        trh = tr[tr['has_err']]
        rho_ewm = fm_slope(trh.dropna(subset=['ewm_algo_err_bp']), 'e_algo_bp', 'ewm_algo_err_bp', ['log_size'])['fm_beta']
        te['e_C_bp'] = te['e_algo_bp'] - (rho_ewm * te['ewm_algo_err_bp']).fillna(0.0)
        _ts = trh.dropna(subset=['ewm_side_err_bp'])
        rho_s = fm_slope(_ts, 'e_algo_bp', 'ewm_side_err_bp', ['log_size'])['fm_beta'] if len(_ts) >= 2_000 else rho_ewm
        te['e_S_bp'] = te['e_algo_bp'] - np.where(te['ewm_side_err_bp'].notna(), rho_s * te['ewm_side_err_bp'].fillna(0.0), rho_ewm * te['ewm_algo_err_bp'].fillna(0.0))
        row = {'month': str(m), 'rho_pooled': rho_ewm, 'rho_same_side': rho_s}
        # ---- v3.9 interaction ladder: the same-side memory with a persistence coefficient conditioned on the factor
        #      model's coordinates and on trade/microstructure state. Every coefficient on prior months; the fallback
        #      where a group is thin or a bond has no same-side history is S itself.
        trs_ = trh.dropna(subset=['ewm_side_err_bp'])
        _base_S = te['e_S_bp'].to_numpy()

        def apply_rho(rho_te) -> np.ndarray:
            r = np.asarray(rho_te, float)
            return np.where(te['ewm_side_err_bp'].notna(), te['e_algo_bp'] - r * te['ewm_side_err_bp'].fillna(0.0), _base_S)

        def rho_by_group(gcol: str, min_rows: int = 2_000):
            rho = {}
            for g_, gg in trs_.groupby(gcol, observed=True):
                if len(gg) >= min_rows:
                    v = fm_slope(gg, 'e_algo_bp', 'ewm_side_err_bp', ['log_size'])['fm_beta']
                    if np.isfinite(v):
                        rho[g_] = float(v)
            return te[gcol].map(rho).astype(float).fillna(rho_s).to_numpy(), rho

        def rho_interaction(cont_cols: list[str]):
            X = ['ewm_side_err_bp'] + [f'_x_{c}' for c in cont_cols]
            trx = trs_.copy(); tex = te.copy()
            for c in cont_cols:
                trx[f'_x_{c}'] = trx['ewm_side_err_bp'] * trx[c].astype(float).fillna(0.0); tex[f'_x_{c}'] = tex['ewm_side_err_bp'] * tex[c].astype(float).fillna(0.0)
            if len(trx) < 5_000:
                return np.full(len(te), rho_s), {}
            b = fm_multi(trx, 'e_algo_bp', X, ['log_size']).set_index('score')['fm_beta']
            r = b['ewm_side_err_bp'] + sum(b[f'_x_{c}'] * tex[c].astype(float).fillna(0.0) for c in cont_cols)
            return np.where(np.isfinite(r), r, rho_s), b.to_dict()

        r_b, b_beta = rho_interaction(beta_cols); te['e_Sb_bp'] = apply_rho(r_b); row.update({f'Sb_{k}': v for k, v in b_beta.items()})
        r_c, rho_c = rho_by_group('beta_cluster'); te['e_Sc_bp'] = apply_rho(r_c); row.update({f'Sc_{k}': v for k, v in rho_c.items()})
        if ec['abs_m'].notna().any():
            r_m, b_m = rho_interaction(['abs_m']); te['e_Sm_bp'] = apply_rho(r_m); row.update({f'Sm_{k}': v for k, v in b_m.items()})
        else:
            te['e_Sm_bp'] = _base_S
        r_a, rho_a = rho_by_group('side_age_bucket'); te['e_Sa_bp'] = apply_rho(r_a); row.update({f'Sa_{k}': v for k, v in rho_a.items()})
        r_q, rho_q = rho_by_group('qty_group'); te['e_Sq_bp'] = apply_rho(r_q); row.update({f'Sq_{k}': v for k, v in rho_q.items()})
        r_sc, rho_sc = rho_by_group('side_cluster'); te['e_Ssc_bp'] = apply_rho(r_sc)
        if 'industry_bucket' in te.columns and te['industry_bucket'].notna().any():
            r_i, rho_i = rho_by_group('industry_bucket'); te['e_Si_bp'] = apply_rho(r_i); row.update({f'Si_{k}': v for k, v in rho_i.items()})
        else:
            te['e_Si_bp'] = _base_S
        kf = KF_BY_MONTH.get(str(m))
        if kf is not None:
            te['kf_pred_bp'] = kf['kf_pred_bp'].reindex(te['_id']).to_numpy(); trk = tr.assign(kf_pred_bp=kf['kf_pred_bp'].reindex(tr['_id']).to_numpy()).dropna(subset=['kf_pred_bp'])
            rho_k = fm_slope(trk, 'e_algo_bp', 'kf_pred_bp', ['log_size'])['fm_beta'] if len(trk) >= 2_000 else 1.0
            rho_k = float(rho_k) if np.isfinite(rho_k) else 1.0
            te['e_K_bp'] = np.where(te['kf_pred_bp'].notna(), te['e_algo_bp'] - rho_k * te['kf_pred_bp'].fillna(0.0), te['e_S_bp'])
            row.update({'rho_K': rho_k, 'kf_phi_daily': float(kf['phi_daily'].iloc[0]), 'kf_half_life_days': float(np.log(0.5) / np.log(kf['phi_daily'].iloc[0])) if kf['phi_daily'].iloc[0] < 1 else np.inf, 'kf_q': float(kf['q'].iloc[0]), 'kf_r': float(kf['r'].iloc[0])})
        else:
            te['e_K_bp'] = te['e_S_bp']
        # ---- v4.0: legacy trials re-opened for the markout re-ranking (closed on MAE in v3.6)
        rho_age = {b: fm_slope(g, 'e_algo_bp', 'last_algo_err_bp', ['log_size'])['fm_beta'] for b, g in trh.groupby('age_bucket') if len(g) >= 500}
        te['e_B_bp'] = te['e_algo_bp'] - (te['age_bucket'].map(rho_age).astype(float) * te['last_algo_err_bp']).fillna(0.0)
        side_med = tr.groupby('side')['e_algo_bp'].median(); te['side_bias'] = te['side'].map(side_med).fillna(0.0)
        te['e_E_bp'] = te['e_algo_bp'] - te['side_bias']
        trs2 = trh.assign(e_sd=trh['e_algo_bp'] - trh['side'].map(side_med).fillna(0.0)).dropna(subset=['ewm_algo_err_bp'])
        rho_F = fm_slope(trs2, 'e_sd', 'ewm_algo_err_bp', ['log_size'])['fm_beta'] if len(trs2) >= 2_000 else rho_ewm
        te['e_F_bp'] = te['e_E_bp'] - (rho_F * te['ewm_algo_err_bp']).fillna(0.0)
        trb = tr.dropna(subset=beta_cols)
        bG = fm_multi(trb, 'e_algo_bp', beta_cols, ['log_size']).set_index('score')['fm_beta'].reindex(beta_cols).fillna(0.0)
        aG = float((trb['e_algo_bp'] - trb[beta_cols].to_numpy(float) @ bG.to_numpy(float)).mean())
        te['e_G_bp'] = te['e_algo_bp'] - (aG + te[beta_cols].fillna(0.0).to_numpy(float) @ bG.to_numpy(float))
        row.update({'rho_F': rho_F, 'side_bias_P': float(side_med.get('P', np.nan)), 'side_bias_S': float(side_med.get('S', np.nan)), 'side_bias_D': float(side_med.get('D', np.nan))})

        # ---- v4.0: side x size intercepts. The v51 EDA found the algo's error falls monotonically with trade size on the
        #      bid (+8 bp on odd lots to -2 bp on blocks): a level effect per side and size that no persistence rule can
        #      carry. The intercept is the Fama-MacBeth daily mean of the error by (side, quantity bin) on prior months,
        #      shrunk to the side median with weight n / (n + cell_min_trades). AQ puts it on the algo quote; SQ puts it on
        #      the same-side memory, estimated on the error S leaves on the training months.
        def cell_intercept(frame: pd.DataFrame, col: str) -> pd.Series:
            # v4.1: medians. The v4.0 mean over-shifted the offer (mean -5.9 bp against a median of -0.4 bp): the fill-relevant
            # location of the error is its median, which is also the tau = 0.5 row of the Section 12 grid.
            g = frame.groupby(['side', 'qty_group'], observed=True)[col]
            cell = g.median(); n_cell = g.size().reindex(cell.index).fillna(0.0)
            side_m = frame.groupby('side', observed=True)[col].median()
            w = n_cell / (n_cell + CFG.cell_min_trades)
            return (w * cell + (1.0 - w) * pd.Series(cell.index.get_level_values(0).map(side_m).to_numpy(), index=cell.index)).rename('bias')

        _key_te = pd.MultiIndex.from_arrays([te['side'].astype(str).to_numpy(), te['qty_group'].astype(str).to_numpy()])
        bias_A = cell_intercept(tr, 'e_algo_bp')
        te['e_AQ_bp'] = te['e_algo_bp'] - pd.Series(bias_A.reindex(_key_te).to_numpy(), index=te.index).fillna(te['side_bias'])
        tr_S = tr.assign(e_S_tr=tr['e_algo_bp'] - np.where(tr['ewm_side_err_bp'].notna(), rho_s * tr['ewm_side_err_bp'].fillna(0.0), rho_ewm * tr['ewm_algo_err_bp'].fillna(0.0)))
        bias_S = cell_intercept(tr_S, 'e_S_tr')
        te['e_SQ_bp'] = te['e_S_bp'] - pd.Series(bias_S.reindex(_key_te).to_numpy(), index=te.index).fillna(0.0)
        BIAS_PATH[str(m)] = {'A': bias_A, 'S': bias_S}
        lp = lvl_pred.reindex(te['_id'])
        te['e_D_bp'] = te['spread_bp'] - lp['yD'].to_numpy(); te['e_Dminus_bp'] = te['spread_bp'] - lp['yDm'].to_numpy(); te['y_mid_D'] = te['MmdYld'] + lp['yD'].to_numpy() / 100.0
        rho_rows.append(row); parts.append(te)
        print(f'  {m}: train {len(tr):,} | test {len(te):,} | rho pooled {rho_ewm:+.2f} | rho same-side {rho_s:+.2f}' + (f' | rho_K {row["rho_K"]:+.2f}' if 'rho_K' in row else ''))
    ecp = pd.concat(parts, ignore_index=True); rho_path = pd.DataFrame(rho_rows).set_index('month')
    for c in ['e_D_bp', 'e_Dminus_bp']:
        ecp[c] = ecp[c].fillna(ecp['e_algo_bp'])
    print('Error-correction coefficients and state-space parameters by month (estimated on prior months):'); display(rho_path.round(3))
    PREDS = {'A algo quote': 'e_algo_bp', 'C side-pooled EWMA (reference)': 'e_C_bp', 'S same-side EWMA': 'e_S_bp',
             'SQ same-side EWMA + side x size intercept': 'e_SQ_bp', 'AQ algo + side x size intercept': 'e_AQ_bp',
             'E algo + side intercept': 'e_E_bp', 'F side intercept + rho x own EWMA': 'e_F_bp', 'B algo + rho(age) x last error': 'e_B_bp', 'G algo + factor-beta correction': 'e_G_bp',
             'Sb same-side EWMA, rho by factor betas': 'e_Sb_bp', 'Sc same-side EWMA, rho by beta-space cluster': 'e_Sc_bp', 'Sm same-side EWMA, rho by |residual drift|': 'e_Sm_bp',
             'Sa same-side EWMA, rho by same-side age': 'e_Sa_bp', 'Sq same-side EWMA, rho by trade size': 'e_Sq_bp', 'Ssc same-side EWMA, rho by side x cluster': 'e_Ssc_bp', 'Si same-side EWMA, rho by industry': 'e_Si_bp',
             'K per-side state-space memory': 'e_K_bp', 'D level model with error features': 'e_D_bp', 'D- level model without error features': 'e_Dminus_bp'}
    PREDS = {k: v for k, v in PREDS.items() if v in ecp.columns}
    MID_LETTER = {k: k.split()[0] for k in PREDS}
    KEY_MIDS = [k for k in PREDS if MID_LETTER[k] in ('A', 'C', 'S', 'SQ', 'AQ', 'K', 'D')]
    # the side x size intercept of the last fold, the bias curve in one table (bp; positive = the print sits above the quote in yield)
    _lastm = sorted(BIAS_PATH)[-1]
    bias_tab = pd.DataFrame({'on the algo quote (AQ)': BIAS_PATH[_lastm]['A'], 'on the error S leaves (SQ)': BIAS_PATH[_lastm]['S']}).unstack(0)
    bias_tab = bias_tab.reindex([q for q in QTY_LABELS if q in bias_tab.index])
    print(f'Side x quantity-bin intercepts (medians) estimated for {_lastm} on the prior months (bp of yield; shrunk to the side median where a cell has < {CFG.cell_min_trades} trades):'); display(bias_tab.round(2))
    record('error_correction', size_intercepts={m_: {k_: flat_records(v_.unstack(0)) for k_, v_ in d_.items()} for m_, d_ in BIAS_PATH.items()})

    def ec_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size()})
        for k_, c in PREDS.items():
            out[f'MAE {k_.split()[0]}'] = g[c].apply(lambda s_: s_.abs().mean())
        for k_, c in list(PREDS.items())[1:]:
            d = frame.assign(gain=frame['e_algo_bp'].abs() - frame[c].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
            out[f'gain {k_.split()[0]} (bp)'] = d.mean(); out[f't {k_.split()[0]}'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
        return out

    ec_all = ec_table(ecp.assign(all='ALL'), 'all'); ec_age = ec_table(ecp, 'age_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d', 'no prior print']).dropna(how='all'); ec_side = ec_table(ecp, 'side'); ec_month = ec_table(ecp, 'month'); ec_band = ec_table(ecp, 'dur_band')
    _gd = {k_: ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp[c].abs()).groupby('trade_date')['gain'].mean().to_numpy() for k_, c in list(PREDS.items())[1:]}
    boot_gain = pd.Series({k_: block_bootstrap_t(v, CFG.block_days, CFG.n_boot, CFG.seed) for k_, v in _gd.items()}, name='bootstrap t of the daily gain')
    _o = ec_all.iloc[0]
    scoreboard = pd.DataFrame({'MAE vs print (bp)': {k_: _o[f'MAE {MID_LETTER[k_]}'] for k_ in PREDS}, 'gain over algo (bp)': {k_: _o.get(f'gain {MID_LETTER[k_]} (bp)', 0.0) for k_ in PREDS},
                               'FM t': {k_: _o.get(f't {MID_LETTER[k_]}', np.nan) for k_ in PREDS}, 'bootstrap t': {k_: boot_gain.get(k_, np.nan) for k_ in PREDS}}).sort_values('gain over algo (bp)', ascending=False)
    scoreboard['MAE rank'] = np.arange(1, len(scoreboard) + 1)
    print(f'MAE scoreboard on {len(ecp):,} held-out trades (the accuracy view; Section 11 re-ranks the same rules in dollars):'); display(scoreboard.round(3))
    side_gain = pd.DataFrame({sd_: {k_: ec_side.loc[sd_, f'gain {MID_LETTER[k_]} (bp)'] for k_ in list(PREDS)[1:]} for sd_ in ec_side.index}).rename(columns={'P': 'P dealer buys (bid)', 'S': 'S dealer sells (offer)', 'D': 'D inter-dealer'})
    print('Gain over the algo by MSRB side (bp of MAE):'); display(side_gain.round(3))
    # ---- the pre-registered bar: does a conditioned variant replace S?
    _gS = _gd['S same-side EWMA']; _mS = ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp['e_S_bp'].abs()).groupby('trade_date')['gain'].mean()
    bar_rows = []
    for k_, c in PREDS.items():
        if not k_.startswith('S') or k_ == 'S same-side EWMA':
            continue
        gv = ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp[c].abs()).groupby('trade_date')['gain'].mean()
        d = (gv - _mS).dropna(); dm = d.groupby(d.index.to_period('M')).mean()
        bar_rows.append({'variant': k_, 'gain (bp)': gv.mean(), 'gain S (bp)': _mS.mean(), 'delta vs S (bp)': d.mean(), 'boot t of delta': block_bootstrap_t(d.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed),
                         'months positive': int((dm > 0).sum()), 'months': int(len(dm)),
                         'passes bar': bool(d.mean() >= CFG.interaction_bar_bp and block_bootstrap_t(d.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) > 3 and (dm > 0).sum() >= min(4, len(dm)))})
    bar = pd.DataFrame(bar_rows).set_index('variant')
    print(f'Pre-registered bar for the interaction ladder: delta >= {CFG.interaction_bar_bp:.2f} bp over S, bootstrap t > 3, positive in >= 4 of {bar["months"].max() if len(bar) else 0} months:'); display(bar.round(3))
    # ---- direction-aware view: toward the print, crossed by less, crossed by more, away; crossing rate by side; asymmetric loss
    def oriented(e: pd.Series, side: pd.Series) -> np.ndarray:
        """Error oriented so that positive = the mid sits on the aggressive side of the print (P: our yield below the print; S: above). D is symmetric."""
        return np.where(side == 'P', e, np.where(side == 'S', -e, np.abs(e)))

    def asym_loss(e: pd.Series, side: pd.Series, tau: float) -> np.ndarray:
        o = oriented(e, side); t_ = np.where(side.isin(['P', 'S']), tau, 0.5)
        return t_ * np.maximum(o, 0.0) + (1.0 - t_) * np.maximum(-o, 0.0)

    dir_rows, cross_rows = [], []
    eA = ecp['e_algo_bp']
    for k_, c in PREDS.items():
        em = ecp[c]; same_sign = np.sign(em) == np.sign(eA); smaller = em.abs() < eA.abs()
        cat = np.where(same_sign & smaller, 'toward', np.where(~same_sign & smaller, 'crossed, closer', np.where(~same_sign & ~smaller, 'crossed, further', 'away')))
        if k_ == 'A algo quote':
            cat = np.full(len(ecp), 'unchanged')
        g_ = ecp.assign(cat=cat, gain=eA.abs() - em.abs()).groupby('cat', observed=True)['gain'].agg(['size', 'mean'])
        rrow = {'mid': k_}
        for cc in ['toward', 'crossed, closer', 'crossed, further', 'away']:
            rrow[f'share {cc}'] = float(g_['size'].get(cc, 0)) / len(ecp); rrow[f'gain {cc} (bp)'] = float(g_['mean'].get(cc, np.nan))
        la = pd.Series(asym_loss(eA, ecp['side'], CFG.asym_tau) - asym_loss(em, ecp['side'], CFG.asym_tau), index=ecp.index)
        dla = la.groupby(ecp['trade_date']).mean()
        rrow['asym loss gain (bp)'] = float(dla.mean()); rrow['asym FM t'] = float(np.sqrt(len(dla)) * dla.mean() / dla.std()) if dla.std() > 0 else np.nan
        dir_rows.append(rrow)
        o = oriented(em, ecp['side'])
        for sd_ in ['P', 'S']:
            msk = (ecp['side'] == sd_).to_numpy()
            cross_rows.append({'mid': k_, 'side': sd_, 'aggressive share': float((o[msk] > 0).mean()) if msk.any() else np.nan, 'mean aggressive bp': float(np.maximum(o[msk], 0).mean()) if msk.any() else np.nan})
    direction = pd.DataFrame(dir_rows).set_index('mid'); crossing = pd.DataFrame(cross_rows).pivot(index='mid', columns='side', values=['aggressive share', 'mean aggressive bp']).reindex(direction.index)
    print('Direction-aware view of the gain: where each mid moves the quote relative to the print (shares of trades and the mean gain inside each bucket), and the gain under an asymmetric loss (tau on the aggressive side):'); display(direction.round(3))
    print('Crossing: share of trades where the mid sits on the aggressive side of the print, by customer side (P = our bid above the print, S = our offer below it):'); display(crossing.round(3))
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2))
    _sh = direction[[f'share {cc}' for cc in ['toward', 'crossed, closer', 'crossed, further', 'away']]].drop(index='A algo quote', errors='ignore'); _sh.columns = ['toward', 'crossed, closer', 'crossed, further', 'away']
    _sh.plot.barh(ax=ax[0], stacked=True, color=['#2E8B57', '#9ACD32', '#E9967A', '#C44E52']); ax[0].set_title('Where the mid moves relative to the print (share of trades)', fontsize=10); ax[0].legend(fontsize=7, loc='lower right'); ax[0].tick_params(axis='y', labelsize=6); ax[0].invert_yaxis()
    _cr = crossing['aggressive share']; _cr.plot.barh(ax=ax[1]); ax[1].set_title('Share of trades on the aggressive side of the print, by side', fontsize=10); ax[1].tick_params(axis='y', labelsize=7); ax[1].legend(['P dealer buys', 'S dealer sells'], fontsize=8); ax[1].invert_yaxis()
    _mae_gain = pd.Series({k_: _gd[k_].mean() for k_ in _gd}); _asym = direction['asym loss gain (bp)'].reindex(_mae_gain.index)
    ax[2].scatter(_mae_gain, _asym, color='#4C72B0')
    for k_ in _mae_gain.index:
        ax[2].annotate(k_.split()[0], (_mae_gain[k_], _asym[k_]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    _lim = [min(_mae_gain.min(), _asym.min(), 0) - 0.1, max(_mae_gain.max(), _asym.max()) + 0.1]; ax[2].plot(_lim, _lim, 'k--', lw=0.6); ax[2].axhline(0, color='k', lw=0.5); ax[2].axvline(0, color='k', lw=0.5)
    ax[2].set_xlabel('MAE gain over the algo (bp)'); ax[2].set_ylabel(f'asymmetric-loss gain (bp, tau {CFG.asym_tau})'); ax[2].set_title('Does the accuracy gain survive a penalty on crossing the print?', fontsize=10)
    savefig('10_direction_aware')
    record('error_correction', interaction_bar=bar.round(4).reset_index().to_dict(orient='records'), direction=direction.round(4).reset_index().to_dict(orient='records'), crossing=flat_records(crossing))
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.5))
    ec_age[[f'MAE {MID_LETTER[k_]}' for k_ in KEY_MIDS]].plot.bar(ax=ax[0]); ax[0].set_title('MAE vs the print by age of the last algo error (bp)', fontsize=10); ax[0].set_xlabel(''); ax[0].legend([k_ for k_ in KEY_MIDS], fontsize=7)
    dd = ecp[ecp['has_side_err']].copy(); dd['decile'] = pd.qcut(dd['ewm_side_err_bp'].rank(method='first'), 10, labels=False) + 1
    for sd, g in dd.groupby('side'):
        ax[1].plot(g.groupby('decile')['e_algo_bp'].mean(), marker='o', label={'S': 'S', 'P': 'P', 'D': 'D'}.get(sd, sd))
    ax[1].plot(dd.groupby('decile')['e_algo_bp'].mean(), color='k', lw=2, label='all'); ax[1].plot(dd.groupby('decile')['ewm_side_err_bp'].mean(), color='grey', ls='--', label='same-side memory itself'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('decile of the same-side error memory'); ax[1].set_ylabel('mean algo error at this print (bp)'); ax[1].legend(fontsize=7); ax[1].set_title('Does the algo repeat its same-side error?', fontsize=10)
    rho_path[[c for c in ['rho_pooled', 'rho_same_side', 'rho_K'] if c in rho_path.columns]].plot(ax=ax[2], marker='o'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Error-correction coefficients by month (estimated on prior months)', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(fontsize=8)
    savefig('10_algo_error_correction')
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    ec_month[[f'MAE {MID_LETTER[k_]}' for k_ in KEY_MIDS]].plot(ax=ax[0], marker='o'); ax[0].set_title('Held-out MAE by month (bp)', fontsize=10); ax[0].legend([k_ for k_ in KEY_MIDS], fontsize=7); ax[0].set_xlabel('')
    if len(imp_mean):
        imp_mean.sort_values().tail(15).plot.barh(ax=ax[1], color='#4C72B0'); ax[1].set_title('Level model D: mean feature importance (error features in the list)', fontsize=10)
    else:
        ax[1].text(0.1, 0.5, 'feature importance needs LightGBM', fontsize=12); ax[1].axis('off')
    savefig('10_algo_error_correction_monthly')
    # side x duration: does the same-side memory repair the short end on every side?
    _dur_order = ['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']
    ecp['dur_bucket'] = pd.cut(ecp['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=_dur_order).astype(str)
    show_mids = [k_ for k_ in PREDS if MID_LETTER[k_] in ('C', 'S', 'SQ', 'D')]
    fig, axs = plt.subplots(1, len(show_mids), figsize=(4.8 * len(show_mids), 5.2), squeeze=False); axs = axs.ravel(); hms = {}
    for a, k_ in zip(axs, show_mids):
        c = PREDS[k_]
        hm = ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp[c].abs()).groupby(['dur_bucket', 'side', 'trade_date'], observed=True)['gain'].mean().groupby(level=[0, 1]).mean().unstack('side').reindex(_dur_order)[[x for x in ['P', 'S', 'D'] if x in ecp['side'].unique()]]
        hms[k_] = hm; v = np.nanmax(np.abs(hm.to_numpy(float))) if np.isfinite(hm.to_numpy(float)).any() else 1.0
        im = a.imshow(hm.to_numpy(float), cmap='RdYlGn', aspect='auto', vmin=-v, vmax=v); a.set_xticks(range(hm.shape[1])); a.set_xticklabels([{'P': 'P bid', 'S': 'S offer', 'D': 'D inter-dealer'}.get(x, x) for x in hm.columns], fontsize=8); a.set_yticks(range(len(hm))); a.set_yticklabels(hm.index, fontsize=8); a.set_title(k_, fontsize=9); a.grid(False)
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                x = hm.iloc[i, j]
                if np.isfinite(x):
                    a.text(j, i, f'{x:+.1f}', ha='center', va='center', fontsize=8)
    fig.suptitle('Gain over the algo by duration bucket and MSRB side (bp)', fontsize=11); plt.tight_layout(); savefig('10_side_by_duration')
    ecp.to_parquet(ARTIFACTS / 'algo_error_correction_v3.parquet', index=False)
    record('error_correction', persistence=pers.round(4).to_dict(orient='records'), persistence_by_gap=pers_gap.round(4).reset_index().to_dict(orient='records'), persistence_by_side_pair=pers_same.round(4).reset_index().to_dict(orient='records'),
           rho_by_month=rho_path.round(4).reset_index().to_dict(orient='records'), overall=ec_all.round(4).to_dict(orient='records'), by_age=ec_age.round(4).reset_index().to_dict(orient='records'), by_side=ec_side.round(4).reset_index().to_dict(orient='records'),
           by_duration_band=ec_band.round(4).reset_index().to_dict(orient='records'), bootstrap_t=boot_gain.round(4).to_dict(), features_full=FULL_FEATURES, features_base=BASE_FEATURES, side_by_duration={k_: v.round(4).reset_index().to_dict(orient='records') for k_, v in hms.items()})
else:
    print('Section 10 skipped: needs matched trades joined to residuals.')

# %% [markdown]
# ## 11. Markout: the quote judged in dollars
#
# **The objective.** A market maker's expected P&L per quote is fill probability times edge, net of what the market
# does after the fill. MAE against the print measures none of those three. This section builds the one-dimensional
# version of the optimizer from data we already have. For every matched print we know our quote, the print, the
# side and the size. Orient everything so that positive is good for the dealer: with $s = +1$ for a dealer buy
# (our bid) and $s = -1$ for a dealer sale (our offer), the *oriented distance* of a mid $m$ is
#
# $$o_m = s \cdot 100\,(y_{\text{print}} - y_m)$$
#
# and a print with $o_m \ge \delta$ is a fill we would have won at the quote $y_m + s\,\delta/100$, where $\delta$
# is a concession in bp (negative = more aggressive than the mid). Marked to the closing evaluation $h$ business
# days after the trade, the fill's P&L in bp of yield is
#
# $$\text{P\&L}_m(h, \delta) = s \cdot 100\,(y_{\text{print}} - y_{\text{mark},h}) - o_m + \delta
#   = \underbrace{s \cdot 100\,(y_{\text{quote}} - y_{\text{mark},0})}_{\text{edge at the quote}} + \underbrace{s \cdot 100\,(y_{\text{mark},0} - y_{\text{mark},h})}_{\text{mark move after the trade}}$$
#
# so the first term is the half-spread we earned against the same-day evaluation and the second is what the market
# does to the position afterwards. **v4.1 hedges the second term with the factor model.** The raw mark move
# contains the common yield move of the sample, which is inventory risk, not quote quality, and in the v4.0 run it
# dominated everything by side. The hedged version replaces it with the cumulative point-in-time residual of
# record between the trade date and the horizon,
#
# $$\text{drift}^{\,\text{hedged}}_h = -\,s \sum_{u = t+1}^{t+h} \varepsilon_{i,u}, \qquad \varepsilon_{i,u} = \Delta y_{i,u} - \beta_{i,u}' f_u,$$
#
# so the factor-implied move $\beta' f$ is removed and what remains is the bond-specific move after the fill: the
# adverse-selection term proper. The raw move and a date-demeaned move are reported beside it as checks. Dollars
# are bp x par x price / 100 x modified duration x 0.0001. Expected P&L per print is the mean of fill x P&L over *all*
# prints, which is what a quote earns per opportunity; fill share and P&L per fill are shown beside it.
#
# **What is tested.** (1) The adverse-selection picture: the residual move after the print for the prints we would
# have won against the ones we would have missed, with the raw and factor-implied moves beside it. (2) The re-ranking: every quote rule of Section 10 at $\delta = 0$ and
# the record horizon, in P&L per print and dollars per month, next to its MAE rank, with a bootstrap t of the daily
# difference against the algo quote and against S, and the pre-registered bar (dollars above S, bootstrap t > 3,
# positive in 4 of 5 months). (3) Fill and P&L curves against the concession, by side, for the algo quote, S and
# SQ. (4) The cell-optimal concession: $\delta^*$ per side x quantity bin chosen on prior months and applied
# forward, the first output the optimizer would actually consume.
#
# **Round trip (v4.2).** Edge against the evaluation depends on where the evaluator sits; in munis that is near the
# bid, which flatters offers and penalises bids. The evaluation-free version exits each would-be fill at the next
# opposite-side print in the same bond within `rt_max_days`, at that print's yield, hedged for the factor move over
# the holding period: a bid fill is sold where the next customer bought, an offer fill is covered where the next
# customer sold. Its coverage is lower (both sides must print) and it assumes we would have been the dealer on the
# exit print, so it is an upper bound of a different kind; what it is free of is the evaluator's convention.
#
# **Two caveats, stated once.** The fill proxy is optimistic: every print that crossed our quote counts as a fill,
# which ignores that the customer may have traded elsewhere and carries the winner's curse. Marks are evaluations,
# not executable prices. Both biases apply identically to every rule, so the ranking and the shape of the curves
# are the result; the dollar level is an upper bound.

# %%
CELL_T('11. Markout: the quote judged in dollars [1]')
if HAS_TRADES and 'ecp' in globals() and not ecp.empty:
    mo = ecp[ecp['side'].isin(['P', 'S'])].copy(); mo['s'] = np.where(mo['side'] == 'P', 1.0, -1.0)
    _q = pd.to_numeric(mo['msrb_quantity'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    _keep = _q.notna() & (_q > 0) & (_q <= 1e8)
    print(f'markout universe: {len(mo):,} customer-side prints (inter-dealer excluded from P&L); {int((~_keep).sum()):,} dropped for non-finite, non-positive or > $100mm par')
    mo = mo[_keep].copy(); mo['par'] = _q[_keep].astype(float)
    _px = pd.to_numeric(mo['msrb_price'], errors='coerce').replace([np.inf, -np.inf], np.nan).clip(1, 300).fillna(100.0)
    mo['dollar_per_bp'] = mo['par'] * _px / 100.0 * mo['modified_duration_lag1'].clip(lower=0).fillna(0) * 1e-4
    mo['cusip'] = mo['cusip'].astype('string'); mo['trade_date'] = to_ns(mo['trade_date'])
    # the closing evaluation h business days after the trade date (first available within 7 calendar days)
    marks = raw_panel[['cusip', 'date', 'closing_yield']].dropna().copy(); marks['cusip'] = marks['cusip'].astype('string'); marks['date'] = to_ns(marks['date']); marks = marks.sort_values('date')
    for h in CFG.markout_horizons:
        tgt = mo[['_id', 'cusip', 'trade_date']].copy(); tgt['target'] = tgt['trade_date'] + pd.offsets.BDay(h) if h > 0 else tgt['trade_date']; tgt = tgt.sort_values('target')
        j = pd.merge_asof(tgt, marks.rename(columns={'date': 'mark_date'}), left_on='target', right_on='mark_date', by='cusip', direction='forward', tolerance=pd.Timedelta(days=7)).set_index('_id')
        mo[f'y_mark_{h}'] = j['closing_yield'].reindex(mo['_id']).to_numpy()
        mo[f'pm_{h}'] = 100.0 * (mo['msrb_yield'] - mo[f'y_mark_{h}'])          # print minus mark (bp), mid-independent
    H0, HR = CFG.markout_horizons[0], CFG.markout_record_h
    # v4.1: the residual path of record per bond (cumulative point-in-time residual and factor-implied move, bp) at the same
    # horizons, so the move after a fill can be split into the factor part (hedgeable, inventory risk) and the residual part
    rp = resid[['cusip', 'date', 'pit_residual', 'fitted_bp']].dropna().copy(); rp['cusip'] = rp['cusip'].astype('string'); rp['date'] = to_ns(rp['date'])
    rp = rp.sort_values(['cusip', 'date'], kind='stable'); rp['cum_resid'] = rp.groupby('cusip', observed=True)['pit_residual'].cumsum(); rp['cum_fit'] = rp.groupby('cusip', observed=True)['fitted_bp'].cumsum(); rp = rp.sort_values('date')
    for h in CFG.markout_horizons:
        tgt = mo[['_id', 'cusip', 'trade_date']].copy(); tgt['target'] = tgt['trade_date'] + pd.offsets.BDay(h) if h > 0 else tgt['trade_date']; tgt = tgt.sort_values('target')
        j = pd.merge_asof(tgt, rp[['cusip', 'date', 'cum_resid', 'cum_fit']].rename(columns={'date': 'rdate'}), left_on='target', right_on='rdate', by='cusip', direction='forward', tolerance=pd.Timedelta(days=7)).set_index('_id')
        mo[f'cr_{h}'] = j['cum_resid'].reindex(mo['_id']).to_numpy(); mo[f'cf_{h}'] = j['cum_fit'].reindex(mo['_id']).to_numpy()
    print('mark coverage by horizon:', {f'h={h}': f'{mo[f"y_mark_{h}"].notna().mean():.1%}' for h in CFG.markout_horizons}, '| residual-path coverage:', {f'h={h}': f'{mo[f"cr_{h}"].notna().mean():.1%}' for h in CFG.markout_horizons})
    months_mo = int(mo['month'].nunique()); DELTAS = np.arange(*CFG.concession_grid_bp)
    S_ARR = mo['s'].to_numpy(float); PM = {h: mo[f'pm_{h}'].to_numpy(float) for h in CFG.markout_horizons}; DPB = mo['dollar_per_bp'].to_numpy(float)
    RD = {h: S_ARR * (PM[h] - PM[H0]) for h in CFG.markout_horizons}                                               # raw oriented mark move after the trade (bp)
    FD = {h: -S_ARR * (mo[f'cf_{h}'].to_numpy(float) - mo[f'cf_{H0}'].to_numpy(float)) for h in CFG.markout_horizons}   # factor-implied part (hedgeable)
    HD = {h: -S_ARR * (mo[f'cr_{h}'].to_numpy(float) - mo[f'cr_{H0}'].to_numpy(float)) for h in CFG.markout_horizons}   # residual part: the hedged drift of record
    _td_arr = mo['trade_date'].to_numpy()
    DD = {}
    for h in CFG.markout_horizons:                                                                                  # date-demeaned raw move: the cheap market-neutral check
        _dy = 100.0 * (mo[f'y_mark_{h}'].to_numpy(float) - mo[f'y_mark_{H0}'].to_numpy(float))
        _mkt = pd.Series(_dy).groupby(_td_arr).transform('mean').to_numpy()
        DD[h] = -S_ARR * (_dy - _mkt)

    def markout_eval(err: np.ndarray, h: int, delta: float = 0.0, drift: dict | None = None):
        """fill flag, P&L (bp) and edge at the quote (bp) for the quote implied by an error column, shifted by a concession.
        P&L = edge at the quote + the move after the trade; the move of record is the factor-hedged residual (HD)."""
        o = S_ARR * err; fill = o >= delta
        edge = S_ARR * PM[H0] - o + delta; pnl = edge + (HD if drift is None else drift)[h]
        return fill, pnl, edge

    def rerank(mask: np.ndarray, h: int, delta: float = 0.0) -> tuple[pd.DataFrame, dict]:
        rows, daily = [], {}
        td = mo['trade_date'].to_numpy()
        for k_, c in PREDS.items():
            fill, pnl, edge = markout_eval(mo[c].to_numpy(float), h, delta)
            ok = mask & np.isfinite(pnl) & np.isfinite(edge); f_ok = fill & ok
            v = np.where(f_ok, pnl, 0.0)
            d_ = pd.Series(v[ok]).groupby(td[ok]).mean(); daily[k_] = d_
            rows.append({'mid': k_, 'fill share': f_ok.sum() / max(ok.sum(), 1), 'edge at quote | fill (bp)': edge[f_ok].mean() if f_ok.any() else np.nan,
                         'residual move after | fill (bp)': (pnl - edge)[f_ok].mean() if f_ok.any() else np.nan, 'P&L | fill (bp)': pnl[f_ok].mean() if f_ok.any() else np.nan,
                         'P&L per print (bp)': float(d_.mean()), '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / months_mo)})
        return pd.DataFrame(rows).set_index('mid'), daily

    ALL = np.ones(len(mo), bool); IS_P = (mo['side'] == 'P').to_numpy(); IS_S = (mo['side'] == 'S').to_numpy()

    # ---- (1) adverse selection: the move after the print, fills vs misses, under the algo quote: raw, factor-implied, residual
    fillA, pnlA, edgeA = markout_eval(mo['e_algo_bp'].to_numpy(float), HR)
    as_rows = []
    for sd_, msk in [('P dealer buys (bid)', IS_P), ('S dealer sells (offer)', IS_S)]:
        for lab, f_ in [('would have filled', fillA), ('would have missed', ~fillA)]:
            r_ = {'side': sd_, 'prints': lab, 'share': float((msk & f_).sum() / msk.sum())}
            r_['edge at our quote (bp)'] = float(edgeA[msk & f_ & np.isfinite(edgeA)].mean())
            ok = msk & f_ & np.isfinite(RD[HR]) & np.isfinite(HD[HR])
            r_[f'raw mark move, h={HR} (bp)'] = float(RD[HR][ok].mean()); r_[f'factor-implied, h={HR} (bp)'] = float(FD[HR][ok].mean())
            for h in CFG.markout_horizons[1:]:
                ok = msk & f_ & np.isfinite(HD[h]); r_[f'residual move, h={h} (bp)'] = float(HD[h][ok].mean())
            as_rows.append(r_)
    adverse = pd.DataFrame(as_rows).set_index(['side', 'prints'])
    print('Adverse selection under the algo quote, fills vs misses (bp, oriented; negative = against the position we would hold). The raw mark move splits into the factor-implied part, which is inventory risk the desk hedges, and the residual part, which is the selection cost of the quote:'); display(adverse.round(2))
    # the trend check: the same P&L per print under the raw, the date-demeaned and the factor-hedged move, A and S by side
    tc_rows = []
    for k_ in ['A algo quote', 'S same-side EWMA']:
        for sd_, msk in [('P', IS_P), ('S', IS_S), ('ALL', ALL)]:
            r_ = {'mid': k_, 'side': sd_}
            for lab, dr in [('raw mark move', RD), ('date-demeaned move', DD), ('factor-hedged residual move', HD)]:
                fill, pnl, _ = markout_eval(mo[PREDS[k_]].to_numpy(float), HR, 0.0, dr); ok = msk & np.isfinite(pnl)
                r_[f'P&L per print, {lab} (bp)'] = float(np.where(fill & ok, pnl, 0.0)[ok].mean()) if ok.any() else np.nan
            tc_rows.append(r_)
    trend_check = pd.DataFrame(tc_rows).set_index(['mid', 'side'])
    print(f'Trend check at h = {HR}: P&L per print under three definitions of the move after the trade. A large gap between raw and hedged, with opposite signs by side, is the common yield move, not the quote:'); display(trend_check.round(2))

    # ---- (2) the re-ranking at delta = 0, record horizon, with the MAE rank beside it
    rr_all, daily_all = rerank(ALL, HR); rr_P, _ = rerank(IS_P, HR); rr_S, _ = rerank(IS_S, HR)
    rr_h1, _ = rerank(ALL, CFG.markout_horizons[1]); rr_h10, _ = rerank(ALL, CFG.markout_horizons[-1])
    _dA = daily_all['A algo quote']; _dS = daily_all['S same-side EWMA']
    for k_ in rr_all.index:
        dvA = (daily_all[k_] - _dA).dropna(); dvS = (daily_all[k_] - _dS).dropna(); dm = dvS.groupby(pd.DatetimeIndex(dvS.index).to_period('M')).mean()
        rr_all.loc[k_, 'boot t vs algo'] = block_bootstrap_t(dvA.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if k_ != 'A algo quote' else np.nan
        rr_all.loc[k_, 'boot t vs S'] = block_bootstrap_t(dvS.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if k_ != 'S same-side EWMA' else np.nan
        rr_all.loc[k_, 'months above S'] = int((dm > 0).sum()) if k_ != 'S same-side EWMA' else np.nan
    rr_all['$ vs S per month ($k)'] = rr_all['$ per month ($k)'] - rr_all.loc['S same-side EWMA', '$ per month ($k)']
    rr_all['beats S (bar)'] = (rr_all['$ vs S per month ($k)'] > 0) & (rr_all['boot t vs S'] > 3) & (rr_all['months above S'] >= min(4, months_mo))
    rr_all['MAE rank'] = scoreboard['MAE rank'].reindex(rr_all.index); rr_all['$ rank'] = rr_all['$ per month ($k)'].rank(ascending=False).astype(int)
    rr_all[f'P&L per print h={CFG.markout_horizons[1]} (bp)'] = rr_h1['P&L per print (bp)']; rr_all[f'P&L per print h={CFG.markout_horizons[-1]} (bp)'] = rr_h10['P&L per print (bp)']
    rr_all = rr_all.sort_values('$ per month ($k)', ascending=False)
    print(f'Markout re-ranking at delta = 0, factor-hedged, marked {HR} business days after the trade ({months_mo} held-out months, {len(mo):,} customer prints). Dollars are per month; the bar is dollars above S, bootstrap t > 3, positive in >= 4 months:'); display(rr_all.round(3))
    print('By side (record horizon, factor-hedged):'); display(pd.concat({'P dealer buys (bid)': rr_P[['fill share', 'edge at quote | fill (bp)', 'residual move after | fill (bp)', 'P&L per print (bp)', '$ per month ($k)']],
                                                           'S dealer sells (offer)': rr_S[['fill share', 'edge at quote | fill (bp)', 'residual move after | fill (bp)', 'P&L per print (bp)', '$ per month ($k)']]}, axis=1).reindex(rr_all.index).round(3))

    # ---- (3) fill and P&L curves against the concession, by side, for the algo quote, S and SQ
    CURVE_MIDS = [k_ for k_ in PREDS if MID_LETTER[k_] in ('A', 'S', 'SQ', 'D')]
    curve_rows = []
    for k_ in CURVE_MIDS:
        err = mo[PREDS[k_]].to_numpy(float)
        for sd_, msk in [('P', IS_P), ('S', IS_S)]:
            for dlt in DELTAS:
                fill, pnl, edge = markout_eval(err, HR, float(dlt)); ok = msk & np.isfinite(pnl); f_ok = fill & ok
                curve_rows.append({'mid': k_, 'side': sd_, 'delta (bp)': float(dlt), 'fill share': f_ok.sum() / max(ok.sum(), 1), 'P&L | fill (bp)': pnl[f_ok].mean() if f_ok.any() else np.nan,
                                   'P&L per print (bp)': float(np.where(f_ok, pnl, 0.0)[ok].mean()), '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / months_mo)})
    curves = pd.DataFrame(curve_rows)
    _cv = curves.dropna(subset=['P&L per print (bp)'])
    best_delta = _cv.loc[_cv.groupby(['mid', 'side'])['P&L per print (bp)'].idxmax()].set_index(['mid', 'side'])[['delta (bp)', 'fill share', 'P&L per print (bp)', '$ per month ($k)']]
    print('Concession that maximises P&L per print on the full held-out window (in-sample on the curve; the walk-forward version is below):'); display(best_delta.round(3))

    # ---- (4) the cell-optimal concession, walk-forward: delta* per side x quantity bin on prior months, applied to the test month, on the S quote
    # ---- realised round trip (v4.2): exit at the next opposite-side print in the same bond, hedged over the holding period
    # ---- v4.8: the print tape: every MSRB print of the universe with a yield (the msrb store), else the matched prints
    _msrb_side_col = next((c_ for c_ in ['side', 'tradetype', 'trade_type', 'msrb_tradetype'] if HAS_MSRB and c_ in msrb_raw.columns), None)   # the MSRB table carries the side as tradetype (P / S / D)
    if HAS_MSRB and _msrb_side_col and 'yield' in msrb_raw.columns and pd.to_numeric(msrb_raw['yield'], errors='coerce').notna().mean() > 0.5:
        tape = pd.DataFrame({'cusip': msrb_raw['cusip'].astype('string').to_numpy(), 'ts': to_ns(msrb_raw['tradetime']).to_numpy(), 'side': msrb_raw[_msrb_side_col].astype('string').str.upper().str.strip().str[0].to_numpy(), 'y': pd.to_numeric(msrb_raw['yield'], errors='coerce').to_numpy(float), 'q': pd.to_numeric(msrb_raw['quantity'], errors='coerce').to_numpy(float)}); TAPE_SRC = f'the full MSRB tape (side from {_msrb_side_col})'
    else:
        tape = pd.DataFrame({'cusip': trades_all['cusip'].astype('string').to_numpy(), 'ts': to_ns(trades_all['trade_ts']).to_numpy(), 'side': trades_all['side'].astype(str).to_numpy(), 'y': pd.to_numeric(trades_all['msrb_yield'], errors='coerce').to_numpy(float), 'q': pd.to_numeric(trades_all['msrb_quantity'], errors='coerce').to_numpy(float)}); TAPE_SRC = 'the matched prints (the msrb store carries no yield)'
    tape = tape.dropna(subset=['cusip', 'ts', 'y']); tape = tape[tape['side'].isin(['P', 'S', 'D']) & (tape['y'] > -5.0) & (tape['y'] < 30.0)]
    if np.nanmedian(tape['y']) > 50.0 * np.nanmedian(pd.to_numeric(mo['msrb_yield'], errors='coerce')):
        tape['y'] = tape['y'] / 100.0; print('tape yields read as bp; rescaled to percent')
    tape = tape.sort_values('ts').reset_index(drop=True); tape['cusip'] = tape['cusip'].astype('string')
    print(f'print tape: {len(tape):,} prints from {TAPE_SRC}, {tape["cusip"].nunique():,} bonds, side mix {tape["side"].value_counts(normalize=True).round(3).to_dict()}')
    _pr = tape[tape['side'].isin(['P', 'S'])].rename(columns={'side': 'exit_side', 'ts': 'exit_ts', 'y': 'y_exit'})[['cusip', 'exit_side', 'exit_ts', 'y_exit']].copy(); _pr['exit_side'] = _pr['exit_side'].astype(str); _pr['exit_date'] = _pr['exit_ts'].dt.normalize(); _pr = _pr.sort_values('exit_ts')
    _q = mo[['_id', 'cusip', 'trade_ts', 'side']].copy(); _q['trade_ts'] = to_ns(_q['trade_ts']); _q['exit_side'] = pd.Series(np.where(_q['side'] == 'P', 'S', 'P'), index=_q.index).astype(str); _q['cusip'] = _q['cusip'].astype('string'); _q = _q.sort_values('trade_ts')
    jx = pd.merge_asof(_q, _pr, left_on='trade_ts', right_on='exit_ts', by=['cusip', 'exit_side'], direction='forward', allow_exact_matches=False, tolerance=pd.Timedelta(days=CFG.rt_max_days)).set_index('_id')
    mo['y_exit'] = jx['y_exit'].reindex(mo['_id']).to_numpy(); mo['exit_date'] = jx['exit_date'].reindex(mo['_id']).to_numpy(); mo['days_to_exit'] = ((jx['exit_ts'] - jx['trade_ts']).dt.total_seconds() / 86400.0).reindex(mo['_id']).to_numpy()
    _ex = mo[['_id', 'cusip', 'exit_date']].dropna().copy(); _ex['exit_date'] = to_ns(_ex['exit_date']); _ex = _ex.sort_values('exit_date')
    jf = pd.merge_asof(_ex, rp[['cusip', 'date', 'cum_fit']].rename(columns={'date': 'rdate'}), left_on='exit_date', right_on='rdate', by='cusip', direction='forward', tolerance=pd.Timedelta(days=7)).set_index('_id')
    mo['cf_exit'] = jf['cum_fit'].reindex(mo['_id']).to_numpy()
    RT_RAW = S_ARR * 100.0 * (mo['msrb_yield'].to_numpy(float) - mo['y_exit'].to_numpy(float))                    # print-to-exit (bp), mid-independent: round-trip P&L of a mid = RT - o + delta
    RT_HDG = RT_RAW + S_ARR * (mo['cf_exit'].to_numpy(float) - mo[f'cf_{H0}'].to_numpy(float))                   # factor move over the holding period removed
    _rt_hedged_share = float(np.isfinite(RT_HDG).sum() / max(np.isfinite(RT_RAW).sum(), 1))
    RT_HDG = np.where(np.isfinite(RT_HDG), RT_HDG, RT_RAW)          # v4.3: exits are mostly within a day, so the raw round trip stands in where the factor path is missing
    _rt_ok = np.isfinite(RT_HDG)
    print(f'round trip (exits on {TAPE_SRC}): {np.isfinite(RT_RAW).mean():.1%} of customer prints have an opposite-side print in the same bond within {CFG.rt_max_days} days; {_rt_hedged_share:.0%} of those carry the factor path and are hedged, the rest use the raw exit; median days to exit {np.nanmedian(mo["days_to_exit"]):.1f}')
    BASE_EVAL = S_ARR * PM[H0] + HD[HR]                                                                            # evaluation markout base: P&L of a mid = BASE - o + delta
    BASE_RT = RT_HDG

    def rt_table(mask: np.ndarray, base: np.ndarray, mids: dict) -> pd.DataFrame:
        rows = []
        for k_, c in mids.items():
            o = S_ARR * mo[c].to_numpy(float); fill = o >= 0; pnl = base - o; ok = mask & np.isfinite(pnl); f_ok = fill & ok
            rows.append({'mid': k_, 'prints with exit': int(ok.sum()), 'fill share': f_ok.sum() / max(ok.sum(), 1), 'P&L | fill (bp)': pnl[f_ok].mean() if f_ok.any() else np.nan,
                         'P&L per print (bp)': float(np.where(f_ok, pnl, 0.0)[ok].mean()) if ok.any() else np.nan, '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / months_mo)})
        return pd.DataFrame(rows).set_index('mid')

    RT_MIDS = {k_: PREDS[k_] for k_ in PREDS if MID_LETTER[k_] in ('A', 'S', 'SQ', 'F', 'D')}
    rt_all = rt_table(ALL, BASE_RT, RT_MIDS); rt_P = rt_table(IS_P, BASE_RT, RT_MIDS); rt_S = rt_table(IS_S, BASE_RT, RT_MIDS)
    ev_P = rt_table(IS_P & _rt_ok, BASE_EVAL, RT_MIDS); ev_S = rt_table(IS_S & _rt_ok, BASE_EVAL, RT_MIDS)   # the evaluation markout on the SAME prints, for a like-for-like side split
    print('Realised round trip (factor-hedged, exit at the next opposite-side print), all customer prints with an exit:'); display(rt_all.round(3))
    side_cmp = pd.concat({'round trip, bid (P)': rt_P['P&L per print (bp)'], 'evaluation markout, bid, same prints': ev_P['P&L per print (bp)'], 'round trip, offer (S)': rt_S['P&L per print (bp)'], 'evaluation markout, offer, same prints': ev_S['P&L per print (bp)']}, axis=1)
    print('The side split under the two metrics, P&L per print (bp) on the same prints. If the evaluation sits near the bid, the bid looks worse and the offer better under the evaluation metric than under the round trip:'); display(side_cmp.round(2))
    rt_side_dollars = pd.DataFrame({'round trip $/mo ($k)': {'P dealer buys (bid)': rt_P.loc['S same-side EWMA', '$ per month ($k)'], 'S dealer sells (offer)': rt_S.loc['S same-side EWMA', '$ per month ($k)']},
                                    'evaluation $/mo ($k), same prints': {'P dealer buys (bid)': ev_P.loc['S same-side EWMA', '$ per month ($k)'], 'S dealer sells (offer)': ev_S.loc['S same-side EWMA', '$ per month ($k)']}})
    print('S same-side EWMA, dollars per month by side under the two metrics (same prints):'); display(rt_side_dollars.round(0))

    def best_delta_for(frame_mask: np.ndarray, err: np.ndarray, base_arr: np.ndarray | None = None) -> float:
        """The concession that maximises mean fill x P&L on the masked rows (evaluation markout at the record horizon by default; pass BASE_RT for the round trip)."""
        base_arr = BASE_EVAL if base_arr is None else base_arr
        o = S_ARR[frame_mask] * err[frame_mask]; base = base_arr[frame_mask] - o; ok = np.isfinite(base) & np.isfinite(o)
        if not ok.any():
            return 0.0
        o, base = o[ok], base[ok]
        vals = [float(np.where(o >= dlt, base + dlt, 0.0).mean()) for dlt in DELTAS]
        return float(DELTAS[int(np.argmax(vals))])

    MONTH_ARR = mo['month'].to_numpy(); QTY_ARR = mo['qty_group'].astype(str).to_numpy(); SIDE_ARR = mo['side'].astype(str).to_numpy()
    errS = mo['e_S_bp'].to_numpy(float); errA = mo['e_algo_bp'].to_numpy(float); errSQ = mo['e_SQ_bp'].to_numpy(float)
    delta_cell = np.zeros(len(mo)); delta_side = np.zeros(len(mo)); delta_side_rt = np.zeros(len(mo)); dstar_rows = []
    for m in sorted(mo['month'].unique())[1:]:
        trm = MONTH_ARR < m; tem = MONTH_ARR == m
        for sd_ in ['P', 'S']:
            d_side = best_delta_for(trm & (SIDE_ARR == sd_), errS); delta_side[tem & (SIDE_ARR == sd_)] = d_side
            delta_side_rt[tem & (SIDE_ARR == sd_)] = best_delta_for(trm & (SIDE_ARR == sd_), errS, BASE_RT)
            for q_ in QTY_LABELS:
                cm = trm & (SIDE_ARR == sd_) & (QTY_ARR == q_)
                d_cell = best_delta_for(cm, errS) if cm.sum() >= CFG.cell_min_trades else d_side
                delta_cell[tem & (SIDE_ARR == sd_) & (QTY_ARR == q_)] = d_cell
                dstar_rows.append({'month': str(m), 'side': sd_, 'qty_group': q_, 'delta* (bp)': d_cell, 'training trades': int(cm.sum())})
    dstar = pd.DataFrame(dstar_rows)
    tested = MONTH_ARR > sorted(mo['month'].unique())[0]
    def pnl_with(err: np.ndarray, dlt: np.ndarray | float, mask: np.ndarray, base_arr: np.ndarray | None = None) -> dict:
        base_arr = BASE_EVAL if base_arr is None else base_arr
        o = S_ARR * err; fill = o >= dlt; pnl = base_arr - o + dlt; ok = mask & np.isfinite(pnl); f_ok = fill & ok
        d_ = pd.Series(np.where(f_ok, pnl, 0.0)[ok]).groupby(mo['trade_date'].to_numpy()[ok]).mean()
        return {'fill share': f_ok.sum() / max(ok.sum(), 1), 'P&L | fill (bp)': pnl[f_ok].mean() if f_ok.any() else np.nan, 'P&L per print (bp)': float(d_.mean()), '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / max(int(pd.Series(MONTH_ARR[ok]).nunique()), 1)), '_daily': d_}
    conc_rows = {'A algo quote, delta 0': pnl_with(errA, 0.0, tested), 'S same-side EWMA, delta 0': pnl_with(errS, 0.0, tested), 'SQ same-side EWMA + size intercept, delta 0': pnl_with(errSQ, 0.0, tested),
                 'S + delta* per side (walk-forward)': pnl_with(errS, delta_side, tested), 'S + delta* per side x size cell (walk-forward)': pnl_with(errS, delta_cell, tested)}
    conc_daily = {k_: v.pop('_daily') for k_, v in conc_rows.items()}
    concession = pd.DataFrame(conc_rows).T
    _base = conc_daily['S same-side EWMA, delta 0']
    concession['boot t vs S at delta 0'] = [block_bootstrap_t((conc_daily[k_] - _base).dropna().to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if k_ != 'S same-side EWMA, delta 0' else np.nan for k_ in concession.index]
    print(f'The cell-optimal concession, walk-forward (delta* chosen on prior months, applied to the next; months after the first held-out month, h = {HR}):'); display(concession.round(3))
    dstar_last = dstar[dstar['month'] == dstar['month'].max()].pivot(index='qty_group', columns='side', values='delta* (bp)').reindex(QTY_LABELS)
    print(f'delta* by side x quantity bin for {dstar["month"].max()} (bp; positive = quote less aggressive than S by that much):'); display(dstar_last.round(1))
    # the same concession policy judged on the round trip, with delta* chosen on the round trip too
    rt_conc = {'S, delta 0': pnl_with(errS, 0.0, tested, BASE_RT), 'S + delta* per side chosen on the evaluation markout': pnl_with(errS, delta_side, tested, BASE_RT), 'S + delta* per side chosen on the round trip': pnl_with(errS, delta_side_rt, tested, BASE_RT)}
    rt_conc_daily = {k_: v.pop('_daily') for k_, v in rt_conc.items()}
    rt_concession = pd.DataFrame(rt_conc).T; _b0 = rt_conc_daily['S, delta 0']
    rt_concession['boot t vs S at delta 0'] = [block_bootstrap_t((rt_conc_daily[k_] - _b0).dropna().to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if k_ != 'S, delta 0' else np.nan for k_ in rt_concession.index]
    _dsr = pd.Series(delta_side_rt[tested]).groupby([pd.Series(SIDE_ARR[tested]), pd.Series(MONTH_ARR[tested]).astype(str)]).first().unstack(0)
    print('The side concession judged on the realised round trip (walk-forward):'); display(rt_concession.round(3))
    print('delta* per side by test month, chosen on the round trip (bp):'); display(_dsr)

    # ---- (5) size x side markout for S and the algo quote at delta 0
    def cell_markout(err: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        fill, pnl, _ = markout_eval(err, HR); ok = np.isfinite(pnl); f_ok = fill & ok
        fr = pd.DataFrame({'side': SIDE_ARR, 'qty_group': QTY_ARR, 'v': np.where(f_ok, pnl, 0.0), 'f': f_ok.astype(float), 'ok': ok})
        fr = fr[fr['ok']]; g = fr.groupby(['qty_group', 'side'], observed=True)
        return g['v'].mean().unstack('side').reindex(QTY_LABELS), g['f'].mean().unstack('side').reindex(QTY_LABELS)
    cmS, cfS = cell_markout(errS); cmA, cfA = cell_markout(errA)
    print('P&L per print (bp) by quantity bin and side at delta 0, S same-side EWMA (fill share in brackets):'); display(pd.DataFrame({c_: [f'{v:+.2f}  ({f_:.0%} filled)' if np.isfinite(v) else '' for v, f_ in zip(cmS[c_], cfS[c_])] for c_ in cmS.columns}, index=cmS.index))

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, sd_ in zip(ax, ['P dealer buys (bid)', 'S dealer sells (offer)']):
        t_ = adverse.loc[sd_]; cols = ['edge at our quote (bp)', f'raw mark move, h={HR} (bp)', f'factor-implied, h={HR} (bp)'] + [f'residual move, h={h} (bp)' for h in CFG.markout_horizons[1:]]
        x = np.arange(len(cols)); w = 0.38
        a.bar(x - w / 2, t_.loc['would have filled', cols], w, color='#2E8B57', label=f'would have filled ({t_.loc["would have filled", "share"]:.0%} of prints)')
        a.bar(x + w / 2, t_.loc['would have missed', cols], w, color='#C44E52', label=f'would have missed ({t_.loc["would have missed", "share"]:.0%})')
        a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels(['edge at quote\n(vs same-day eval)', f'raw mark move\nafter {HR}d', f'factor-implied\npart, {HR}d'] + [f'residual move\nafter {h}d' for h in CFG.markout_horizons[1:]], fontsize=8)
        a.axvline(2.5, color='#888888', lw=0.8, ls='--')
        a.set_title(f'{sd_}: edge at the quote, then the move after the trade split into factor and residual (bp)', fontsize=10); a.legend(fontsize=8); a.set_ylabel('bp, oriented (+ = good for the dealer)')
    fig.suptitle('Adverse selection under the algo quote: the prints we would have won versus the ones we would have missed (right of the dashed line = factor-hedged)', fontsize=11); plt.tight_layout(); savefig('11_markout_adverse_selection')

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.2))
    rr_plot = rr_all.sort_values('$ per month ($k)')
    cols_ = ['#2E8B57' if v >= rr_all.loc['S same-side EWMA', '$ per month ($k)'] else '#8C8C8C' for v in rr_plot['$ per month ($k)']]
    ax[0].barh(np.arange(len(rr_plot)), rr_plot['$ per month ($k)'], color=cols_); ax[0].set_yticks(np.arange(len(rr_plot))); ax[0].set_yticklabels([f'{i}   [MAE rank {int(r)}]' for i, r in zip(rr_plot.index, rr_plot['MAE rank'])], fontsize=7)
    ax[0].axvline(rr_all.loc['A algo quote', '$ per month ($k)'], color='k', ls='--', lw=0.8, label='algo quote'); ax[0].axvline(rr_all.loc['S same-side EWMA', '$ per month ($k)'], color='#4C72B0', ls=':', lw=1.0, label='S same-side EWMA')
    ax[0].set_title(f'Would-have-filled P&L per month ($k), factor-hedged, marked {HR} days after the trade; green = at or above S', fontsize=10); ax[0].set_xlabel('$k per month'); ax[0].legend(fontsize=8, loc='lower right')
    comp = pd.DataFrame({'edge at quote x fill share': rr_plot['edge at quote | fill (bp)'] * rr_plot['fill share'], 'residual move after x fill share': rr_plot['residual move after | fill (bp)'] * rr_plot['fill share']})
    comp.plot.barh(ax=ax[1], stacked=True, color=['#4C72B0', '#DD8452']); ax[1].plot(rr_plot['P&L per print (bp)'], np.arange(len(rr_plot)), 'kD', ms=4, label='P&L per print')
    ax[1].axvline(0, color='k', lw=0.6); ax[1].set_yticklabels([]); ax[1].set_title('P&L per print (bp) = edge earned at the quote + factor-hedged residual move after the trade, both times the fill share', fontsize=10); ax[1].legend(fontsize=8)
    plt.tight_layout(); savefig('11_markout_rerank')

    fig, ax = plt.subplots(2, 3, figsize=(19, 8.6))
    for i, sd_ in enumerate(['P', 'S']):
        for j, (col, ttl) in enumerate([('fill share', 'fill share'), ('P&L | fill (bp)', 'P&L per fill (bp)'), ('P&L per print (bp)', 'P&L per print (bp) = the objective')]):
            a = ax[i, j]
            for k_, col_ in zip(CURVE_MIDS, ['#4C72B0', '#2E8B57', '#DD8452', '#C44E52', '#8172B2', '#937860']):
                c_ = curves[(curves['mid'] == k_) & (curves['side'] == sd_)]
                a.plot(c_['delta (bp)'], c_[col], marker='.', color=col_, label=k_)
                if j == 2 and c_[col].notna().any():
                    bd_ = c_.loc[c_[col].idxmax()]; a.plot(bd_['delta (bp)'], bd_[col], 'o', ms=9, mfc='none', mec=col_, mew=1.5)
            a.axvline(0, color='k', lw=0.6); a.axhline(0, color='k', lw=0.4); a.set_title(f'{ {"P": "P dealer buys (bid)", "S": "S dealer sells (offer)"}[sd_] }: {ttl}', fontsize=10); a.set_xlabel('concession delta (bp; negative = more aggressive than the mid)')
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.suptitle(f'Fill curve and expected P&L against the concession, marked {HR} days after the trade (circle = best delta on the full window)', fontsize=11); plt.tight_layout(); savefig('11_markout_curves')

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for a, (tab, ftab, ttl) in zip(ax[:2], [(cmA, cfA, 'A algo quote'), (cmS, cfS, 'S same-side EWMA')]):
        v = np.nanmax(np.abs(tab.to_numpy(float))) or 1.0
        im = a.imshow(tab.to_numpy(float), cmap='RdYlGn', aspect='auto', vmin=-v, vmax=v); a.set_xticks(range(tab.shape[1])); a.set_xticklabels([{'P': 'P bid', 'S': 'S offer'}.get(c_, c_) for c_ in tab.columns]); a.set_yticks(range(len(tab))); a.set_yticklabels(tab.index, fontsize=8); a.grid(False)
        for r_ in range(tab.shape[0]):
            for c_ in range(tab.shape[1]):
                x = tab.iloc[r_, c_]; f_ = ftab.iloc[r_, c_]
                if np.isfinite(x):
                    a.text(c_, r_, f'{x:+.1f}\n({f_:.0%} filled)', ha='center', va='center', fontsize=8)
        a.set_title(f'{ttl}: P&L per print (bp) by quantity bin and side', fontsize=10)
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    dstar_last.plot.bar(ax=ax[2], color=['#4C72B0', '#DD8452']); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title(f'delta* per side x quantity bin, {dstar["month"].max()} (bp)', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(['P bid', 'S offer'], fontsize=8); ax[2].tick_params(axis='x', rotation=30, labelsize=8)
    plt.tight_layout(); savefig('11_markout_size_side')

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    _sc = side_cmp.reindex([k_ for k_ in RT_MIDS]); x = np.arange(len(_sc)); w = 0.2
    for i_, (col, c_) in enumerate(zip(_sc.columns, ['#2E8B57', '#9ACD32', '#4C72B0', '#9FB6D9'])):
        ax[0].bar(x + (i_ - 1.5) * w, _sc[col], w, color=c_, label=col)
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels([MID_LETTER[k_] for k_ in _sc.index]); ax[0].set_ylabel('P&L per print (bp)'); ax[0].legend(fontsize=7); ax[0].set_title('Side split: round trip vs evaluation markout, same prints', fontsize=10)
    for sd_, msk, c_ in [('P bid', IS_P, '#4C72B0'), ('S offer', IS_S, '#DD8452')]:
        vals_e, vals_r = [], []
        for dlt in DELTAS:
            o = S_ARR * errS; fill = o >= dlt
            pe = BASE_EVAL - o + dlt; oke = msk & _rt_ok & np.isfinite(pe); vals_e.append(float(np.where(fill & oke, pe, 0.0)[oke].mean()))
            pr_ = BASE_RT - o + dlt; okr = msk & np.isfinite(pr_); vals_r.append(float(np.where(fill & okr, pr_, 0.0)[okr].mean()))
        ax[1].plot(DELTAS, vals_e, color=c_, ls='--', marker='.', label=f'{sd_}: evaluation markout'); ax[1].plot(DELTAS, vals_r, color=c_, marker='.', label=f'{sd_}: round trip')
    ax[1].axvline(0, color='k', lw=0.6); ax[1].axhline(0, color='k', lw=0.4); ax[1].set_xlabel('concession on S (bp)'); ax[1].set_ylabel('P&L per print (bp)'); ax[1].legend(fontsize=7); ax[1].set_title('Concession curve of S under both metrics (prints with an exit)', fontsize=10)
    _dte = pd.Series(mo['days_to_exit']).dropna()
    ax[2].hist(_dte.clip(0, CFG.rt_max_days), bins=40, color='#8172B2'); ax[2].set_title(f'Days to the exit print (median {_dte.median():.1f}; {_rt_ok.mean():.0%} of prints have one)', fontsize=10); ax[2].set_xlabel('days')
    plt.tight_layout(); savefig('11_markout_round_trip')

    mo[['_id', 'cusip', 'trade_date', 'side', 'qty_group', 'par', 'dollar_per_bp', 'y_exit', 'days_to_exit', 'cf_exit'] + [f'pm_{h}' for h in CFG.markout_horizons] + [f'y_mark_{h}' for h in CFG.markout_horizons] + [f'cr_{h}' for h in CFG.markout_horizons] + [f'cf_{h}' for h in CFG.markout_horizons] + list(PREDS.values())].to_parquet(ARTIFACTS / 'markout_v4.parquet', index=False)
    record('markout', horizons=list(CFG.markout_horizons), record_h=HR, months=months_mo, prints=int(len(mo)), hedged=True, adverse_selection=flat_records(adverse.reset_index()), trend_check=flat_records(trend_check.reset_index()), rerank=rr_all.round(4).reset_index().to_dict(orient='records'),
           round_trip={'coverage': float(_rt_ok.mean()), 'median_days_to_exit': float(np.nanmedian(mo['days_to_exit'])), 'all': rt_all.round(4).reset_index().to_dict(orient='records'), 'side_split': flat_records(side_cmp.reset_index()), 'side_dollars_S': flat_records(rt_side_dollars.reset_index()), 'concession': rt_concession.round(4).reset_index().to_dict(orient='records'), 'delta_star_by_month': flat_records(_dsr.reset_index())},
           rerank_by_side={'P': rr_P.round(4).reset_index().to_dict(orient='records'), 'S': rr_S.round(4).reset_index().to_dict(orient='records')}, curves=curves.round(4).to_dict(orient='records'),
           best_delta_full_window=best_delta.round(4).reset_index().to_dict(orient='records'), concession=concession.round(4).reset_index().to_dict(orient='records'), delta_star=dstar.round(4).to_dict(orient='records'),
           size_side={'S': flat_records(cmS.reset_index()), 'A': flat_records(cmA.reset_index()), 'fill_S': flat_records(cfS.reset_index())})
else:
    print('Section 11 skipped: needs Section 10.')

# %% [markdown]
# ### 11c. Marked to the next real trade: the evaluation-free markout at every horizon
#
# The markout of Section 11 marks a would-be fill to the closing evaluation. That is the market's systematic view of
# the bond, and it is the surface the factor model is fitted on, so a ranking built on it is partly a ranking against
# the model's own world. The real out-of-sample information is the next print. This section marks every would-be fill
# to real trades only, from the **print tape**: every MSRB print in the universe with a yield, any side.
#
# **Four marks.** (i) The next print of **any side**, turned into a mid-equivalent: a dealer buy (P) sits above the
# mid and a dealer sale (S) below it by about half the dealer round trip, so the mark is $y_P - \tfrac{1}{2}\text{RT}$ or
# $y_S + \tfrac{1}{2}\text{RT}$, with the half round trip the median P minus S spread of same bond-day prints by the
# mark's size bin, estimated on prior months; an inter-dealer print (D) is used as it is. (ii) The next **inter-dealer**
# print only, the cleanest mid the tape offers. (iii) The next **opposite-side** print, which is the realised round
# trip of Section 11. (iv) The next **same-side** print, which says where the same flow printed next. Each mark is
# taken as the next print after the fill within 10 calendar days, and as the last print within 1, 5 and 10 business
# days. The P&L of a rule on a print is the oriented distance from the print to the mark less the distance the quote
# sat through the print, exactly as before, with nothing from the evaluation in it.
#
# **What is read.** Coverage by mark and horizon (a tape mark exists only where the bond traded again, and a sparse
# bond is marked late or not at all). The re-ranking of every rule on the next-print mark, raw, with the usual bar,
# beside the evaluation-based rank, and the rank agreement between the two. The P&L of S and the algo quote at every
# horizon and mark, with the date-demeaned and factor-hedged versions as checks on the common move. The noise: the
# dispersion of a fill's P&L under each mark, and the number of prints it takes to resolve a tenth of a basis point,
# because scatter is the price of real information. Then S by side and by size under trade marks.

# %%
CELL_T('11c. Marked to the next real trade [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty and 'tape' in globals():
    t0 = time.perf_counter()
    # ---- half the dealer round trip by size bin on prior months: same bond-day P and S prints on the tape
    TD = mo['trade_date'].to_numpy()   # trade dates of the customer prints (11b defines the same)
    _tp = tape.copy(); _tp['date'] = _tp['ts'].dt.normalize(); _tp['qg'] = qty_group(pd.Series(_tp['q'])).to_numpy()
    _py = _tp[_tp['side'] == 'P'].groupby(['cusip', 'date'], observed=True).agg(yP=('y', 'median'), qg=('qg', 'first')); _sy = _tp[_tp['side'] == 'S'].groupby(['cusip', 'date'], observed=True)['y'].median().rename('yS')
    _pairs = _py.join(_sy, how='inner').reset_index(); _pairs['spread'] = 100.0 * (_pairs['yP'] - _pairs['yS']); _pairs['month'] = _pairs['date'].dt.to_period('M'); _pairs = _pairs[_pairs['spread'].abs() <= 200]
    HALF: dict = {}
    for m in sorted(set(MONTH_ARR)):
        prior = _pairs[_pairs['month'] < m]
        HALF[m] = (prior.groupby('qg', observed=True)['spread'].median() / 2.0) if len(prior) >= CFG.trade_mark_min_pairs else None
    _hs_last = HALF[max(HALF)] if HALF and HALF[max(HALF)] is not None else None
    if _hs_last is not None:
        print(f'half the dealer round trip by size bin of the mark print (bp; prior months, last fold; {len(_pairs):,} same bond-day P/S pairs on the tape):'); display(_hs_last.reindex(QTY_LABELS).round(1).to_frame('half spread (bp)').T)
    else:
        print('too few same bond-day P/S pairs on the tape to estimate the half spread; any-side marks are used unadjusted')

    def half_spread_for(mark_side: np.ndarray, mark_q: np.ndarray) -> np.ndarray:
        out = np.zeros(len(mark_side)); qg_ = qty_group(pd.Series(mark_q)).to_numpy()
        for m in np.unique(MONTH_ARR):
            hs = HALF.get(m); sel = MONTH_ARR == m
            if hs is None or not sel.any():
                continue
            out[sel] = pd.Series(qg_[sel]).map(hs).fillna(float(hs.median())).to_numpy()
        return np.where(mark_side == 'D', 0.0, out)

    _qm = mo[['_id', 'cusip', 'trade_ts', 'side']].copy(); _qm['cusip'] = _qm['cusip'].astype('string'); _qm['trade_ts'] = to_ns(_qm['trade_ts']); _qm['side'] = _qm['side'].astype(str); _qm['opp'] = np.where(_qm['side'] == 'P', 'S', 'P')
    _tape_m = tape.rename(columns={'ts': 'mark_ts', 'y': 'y_mark', 'side': 'mark_side', 'q': 'q_mark'}); _tape_m['mark_side'] = _tape_m['mark_side'].astype(str)
    Y_PRINT = pd.to_numeric(mo['msrb_yield'], errors='coerce').to_numpy(float)
    TAPE_KINDS = {'any side, mid-equivalent': 'any', 'inter-dealer only': 'D', 'opposite side (round trip)': 'opp', 'same side': 'same'}
    TAPE_H = [None] + [int(h) for h in CFG.trade_mark_horizons]

    def mark_tape(kind: str, h) -> tuple[np.ndarray, np.ndarray]:
        """Oriented print-to-mark distance (bp; a rule's P&L = this - o) and days to the mark, for one mark kind and horizon (None = the next print)."""
        q = _qm.copy(); by = ['cusip']
        if kind == 'D':
            tp = _tape_m[_tape_m['mark_side'] == 'D']
        elif kind in ('opp', 'same'):
            tp = _tape_m[_tape_m['mark_side'].isin(['P', 'S'])]; q['mark_side'] = q['opp'] if kind == 'opp' else q['side']; by = ['cusip', 'mark_side']
        else:
            tp = _tape_m
        if h is None:
            j = pd.merge_asof(q.sort_values('trade_ts'), tp.sort_values('mark_ts'), left_on='trade_ts', right_on='mark_ts', by=by, direction='forward', allow_exact_matches=False, tolerance=pd.Timedelta(days=CFG.rt_max_days))
        else:
            q['target'] = q['trade_ts'].dt.normalize() + pd.offsets.BDay(h) + pd.Timedelta(hours=23, minutes=59)
            j = pd.merge_asof(q.sort_values('target'), tp.sort_values('mark_ts'), left_on='target', right_on='mark_ts', by=by, direction='backward')
            j.loc[~(j['mark_ts'] > j['trade_ts']), ['y_mark', 'mark_ts']] = np.nan
        j = j.set_index('_id').reindex(mo['_id'].to_numpy())
        y_mark = j['y_mark'].to_numpy(float); ms = j['mark_side'].astype(str).to_numpy(); qm_ = j['q_mark'].to_numpy(float)
        if kind == 'any':
            adj = half_spread_for(ms, qm_) / 100.0; y_mark = np.where(ms == 'P', y_mark - adj, np.where(ms == 'S', y_mark + adj, y_mark))
        days = ((j['mark_ts'] - pd.Series(_qm.set_index('_id')['trade_ts']).reindex(mo['_id'].to_numpy())).dt.total_seconds() / 86400.0).to_numpy(float)
        return S_ARR * 100.0 * (Y_PRINT - y_mark), days

    TM, TMD = {}, {}
    for kname, kind in TAPE_KINDS.items():
        for h in TAPE_H:
            TM[(kname, h)], TMD[(kname, h)] = mark_tape(kind, h)
    _hl = lambda h: 'next print' if h is None else f'last print within {h} bd'
    cov = pd.DataFrame({_hl(h): {kname: float(np.isfinite(TM[(kname, h)]).mean()) for kname in TAPE_KINDS} for h in TAPE_H})
    dte = pd.DataFrame({_hl(h): {kname: float(np.nanmedian(TMD[(kname, h)])) for kname in TAPE_KINDS} for h in TAPE_H})
    print(f'Coverage of the trade marks (share of customer prints with a mark; {time.perf_counter() - t0:.0f}s):'); display(cov.round(3))
    print('Median days from the fill to the mark:'); display(dte.round(1))
    TRADE_REC = ('any side, mid-equivalent', None); BASE_TRADE = TM[TRADE_REC]

    # ---- the re-ranking on the next real print, raw
    def rerank_trade(base: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
        rows, daily = [], {}
        for k_, c in PREDS.items():
            o = S_ARR * mo[c].to_numpy(float); fill = o >= 0; pnl = base - o; ok = mask & np.isfinite(pnl) & np.isfinite(o); f_ok = fill & ok
            v = np.where(f_ok, pnl, 0.0); d_ = pd.Series(v[ok]).groupby(TD[ok]).mean(); daily[k_] = d_
            rows.append({'mid': k_, 'fill share': f_ok.sum() / max(ok.sum(), 1), 'P&L | fill (bp)': float(pnl[f_ok].mean()) if f_ok.any() else np.nan, 'median P&L | fill (bp)': float(np.median(pnl[f_ok])) if f_ok.any() else np.nan,
                         'P&L per print (bp)': float(d_.mean()) if len(d_) else np.nan, '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / max(int(pd.Series(MONTH_ARR[ok]).nunique()), 1))})
        return pd.DataFrame(rows).set_index('mid'), daily

    tr_all, tr_daily = rerank_trade(BASE_TRADE, ALL); _dS_t = tr_daily['S same-side EWMA']
    for k_ in tr_all.index:
        dv = (tr_daily[k_] - _dS_t).dropna(); dm = dv.groupby(pd.DatetimeIndex(dv.index).to_period('M')).mean()
        tr_all.loc[k_, 'boot t vs S'] = block_bootstrap_t(dv.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if k_ != 'S same-side EWMA' else np.nan
        tr_all.loc[k_, 'months above S'] = int((dm > 0).sum()) if k_ != 'S same-side EWMA' else np.nan
    tr_all['beats S (bar)'] = ((tr_all['P&L per print (bp)'] - tr_all.loc['S same-side EWMA', 'P&L per print (bp)']) >= CFG.interaction_bar_bp) & (tr_all['boot t vs S'] > 3) & (tr_all['months above S'] >= min(4, months_mo))
    for kname, h in [('inter-dealer only', None), ('opposite side (round trip)', None), ('any side, mid-equivalent', 1), ('any side, mid-equivalent', 5), ('any side, mid-equivalent', 10)]:
        _t, _ = rerank_trade(TM[(kname, h)], ALL); tr_all[f'per print: {kname}, {_hl(h)}'] = _t['P&L per print (bp)']
    tr_all['evaluation $ rank'] = rr_all['$ rank'].reindex(tr_all.index); tr_all['trade $ rank'] = tr_all['$ per month ($k)'].rank(ascending=False).astype(int); tr_all['MAE rank'] = rr_all['MAE rank'].reindex(tr_all.index)
    tr_all = tr_all.sort_values('P&L per print (bp)', ascending=False)
    _rho = float(pd.Series(tr_all['$ per month ($k)']).rank().corr(pd.Series(rr_all['$ per month ($k)']).reindex(tr_all.index).rank(), method='spearman'))
    _rho_pp = float(tr_all['P&L per print (bp)'].rank().corr(rr_all['P&L per print (bp)'].reindex(tr_all.index).rank(), method='spearman'))
    print(f'Re-ranking on the next real print (any side, mid-equivalent; raw, nothing from the evaluation), {months_mo} held-out months. Bar: +{CFG.interaction_bar_bp:.2f} bp per print over S, bootstrap t > 3, positive in >= 4 months. Spearman with the evaluation ranking: dollars {_rho:+.2f}, per print {_rho_pp:+.2f}:'); display(tr_all.round(3))

    # ---- S and the algo quote at every mark and horizon; the common move: raw, date-demeaned, factor-hedged on the record mark
    hz_rows = []
    for k_ in ['S same-side EWMA', 'A algo quote']:
        o = S_ARR * mo[PREDS[k_]].to_numpy(float); fill = o >= 0
        for kname in TAPE_KINDS:
            r_ = {'mid': k_, 'mark': kname}
            for h in TAPE_H:
                pnl = TM[(kname, h)] - o; ok = np.isfinite(pnl); r_[_hl(h)] = float(np.where(fill & ok, pnl, 0.0)[ok].mean()) if ok.any() else np.nan
            hz_rows.append(r_)
        pnl_e = BASE_EVAL - o; ok = np.isfinite(pnl_e); hz_rows.append({'mid': k_, 'mark': 'evaluation (Section 11, h = 5, hedged)', 'next print': np.nan, **{_hl(h): (float(np.where(fill & ok, pnl_e, 0.0)[ok].mean()) if h == HR else np.nan) for h in TAPE_H if h is not None}})
    horizon_tab = pd.DataFrame(hz_rows).set_index(['mid', 'mark'])
    print('P&L per print (bp) by mark and horizon, S and the algo quote (raw trade marks; the evaluation row for reference):'); display(horizon_tab.round(2))
    # the common move on the record mark: the move from the print to the mark, raw, demeaned by trade date, and net of the factor-implied move to the mark date
    _mv = BASE_TRADE.copy(); _mkt = pd.Series(_mv).groupby(TD).transform('mean').to_numpy(); _dd = _mv - _mkt
    _md = _qm.set_index('_id')['trade_ts'].reindex(mo['_id'].to_numpy()) + pd.to_timedelta(np.nan_to_num(TMD[TRADE_REC], nan=0.0), unit='D')
    _ex2 = pd.DataFrame({'_id': mo['_id'].to_numpy(), 'cusip': mo['cusip'].astype('string').to_numpy(), 'mark_date': to_ns(pd.Series(_md.to_numpy())).dt.normalize().to_numpy()}).dropna(); _ex2['cusip'] = _ex2['cusip'].astype('string'); _ex2['mark_date'] = to_ns(_ex2['mark_date']); _ex2 = _ex2.sort_values('mark_date')
    jf2 = pd.merge_asof(_ex2, rp[['cusip', 'date', 'cum_fit']].rename(columns={'date': 'rdate'}), left_on='mark_date', right_on='rdate', by='cusip', direction='forward', tolerance=pd.Timedelta(days=7)).set_index('_id')
    _cfm = jf2['cum_fit'].reindex(mo['_id'].to_numpy()).to_numpy(float); _hd = _mv + S_ARR * (_cfm - mo[f'cf_{H0}'].to_numpy(float))
    cm_rows = []
    for k_ in ['S same-side EWMA', 'A algo quote']:
        o = S_ARR * mo[PREDS[k_]].to_numpy(float); fill = o >= 0
        for sd_, msk in [('P', IS_P), ('S', IS_S), ('ALL', ALL)]:
            r_ = {'mid': k_, 'side': sd_}
            for lab, base in [('raw', _mv), ('date-demeaned', _dd), ('factor-hedged', _hd)]:
                pnl = base - o; ok = msk & np.isfinite(pnl); r_[f'P&L per print, {lab} (bp)'] = float(np.where(fill & ok, pnl, 0.0)[ok].mean()) if ok.any() else np.nan
            cm_rows.append(r_)
    common_move = pd.DataFrame(cm_rows).set_index(['mid', 'side'])
    print('The common move on the next-print mark: P&L per print raw, demeaned by trade date, and net of the factor-implied move to the mark (bp). A large gap between raw and the other two, with opposite signs by side, is the market, not the quote:'); display(common_move.round(2))

    # ---- noise: the dispersion of a fill's P&L under each mark, and the prints needed to resolve a tenth of a basis point
    nz_rows = []; oS = S_ARR * errS; fS = oS >= 0
    for lab, base in [('evaluation, h = 5 (hedged)', BASE_EVAL), ('next print, any side (mid-equivalent)', BASE_TRADE), ('next inter-dealer print', TM[('inter-dealer only', None)]), ('next opposite-side print (round trip)', TM[('opposite side (round trip)', None)]), ('last print within 5 bd, any side', TM[('any side, mid-equivalent', 5)])]:
        pnl = base - oS; ok = np.isfinite(pnl); d_ = pd.Series(np.where(fS & ok, pnl, 0.0)[ok]).groupby(TD[ok]).mean()
        nz_rows.append({'mark': lab, 'coverage': float(ok.mean()), 'sd of P&L per fill (bp)': float(pnl[fS & ok].std()), 'sd of the daily mean (bp)': float(d_.std()), 'days': int(len(d_)), 'prints per day': float(ok.sum() / max(len(d_), 1)), 'prints to resolve 0.10 bp at t = 3': float((3.0 * pnl[fS & ok].std() / 0.10) ** 2) if (fS & ok).any() else np.nan})
    noise = pd.DataFrame(nz_rows).set_index('mark')
    print('Noise of each mark on S fills: a trade mark carries the bid-offer scatter of the next print, so it needs more prints for the same precision:'); display(noise.round(2))

    # ---- S by side and by size on the next-print mark
    sz_rows = []
    for q_ in QTY_LABELS:
        for sd_, msk in [('P', IS_P), ('S', IS_S)]:
            sel = msk & (QTY_ARR == q_)
            for k_ in ['S same-side EWMA', 'A algo quote']:
                o = S_ARR * mo[PREDS[k_]].to_numpy(float); pnl = BASE_TRADE - o; ok = sel & np.isfinite(pnl); f_ok = ok & (o >= 0)
                if ok.sum() >= 200:
                    sz_rows.append({'qty_group': q_, 'side': sd_, 'mid': k_, 'prints': int(ok.sum()), 'fill share': float(f_ok.sum() / ok.sum()), 'P&L per print (bp)': float(np.where(f_ok, pnl, 0.0)[ok].mean()), '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / months_mo)})
    trade_size = pd.DataFrame(sz_rows).set_index(['qty_group', 'side', 'mid']).unstack('mid') if sz_rows else pd.DataFrame()
    if len(trade_size):
        print('By quantity bin and side on the next-print mark, S and the algo quote:'); display(trade_size.round(2))

    # ---- figures
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    cov.plot.bar(ax=ax[0, 0], width=0.8); ax[0, 0].set_title('Coverage: share of customer prints with a real-trade mark', fontsize=10); ax[0, 0].tick_params(axis='x', rotation=15, labelsize=8); ax[0, 0].legend(fontsize=7); ax[0, 0].set_ylim(0, 1)
    _xh = ['next print'] + [f'{h} bd' for h in CFG.trade_mark_horizons]
    for k_, ls_ in [('S same-side EWMA', '-'), ('A algo quote', '--')]:
        for kname, c_ in zip(TAPE_KINDS, ['#4C72B0', '#2E8B57', '#DD8452', '#8C8C8C']):
            ax[0, 1].plot(_xh, horizon_tab.loc[(k_, kname)].to_numpy(float), marker='o', ls=ls_, color=c_, label=f'{k_.split()[0]}: {kname}')
    ax[0, 1].axhline(0, color='k', lw=0.5); ax[0, 1].set_ylabel('P&L per print (bp)'); ax[0, 1].set_title('S (solid) and the algo quote (dashed) by mark and horizon, raw trade marks', fontsize=10); ax[0, 1].legend(fontsize=6, ncol=2)
    _tp_ = tr_all.sort_values('P&L per print (bp)'); y_ = np.arange(len(_tp_))
    ax[0, 2].barh(y_, _tp_['P&L per print (bp)'], color=['#2E8B57' if b_ else ('#4C72B0' if i_ == 'S same-side EWMA' else '#8C8C8C') for i_, b_ in zip(_tp_.index, _tp_['beats S (bar)'])]); ax[0, 2].set_yticks(y_); ax[0, 2].set_yticklabels([f'{i_[:34]}  [eval $ rank {int(r_)}]' for i_, r_ in zip(_tp_.index, _tp_['evaluation $ rank'])], fontsize=6)
    ax[0, 2].axvline(tr_all.loc['S same-side EWMA', 'P&L per print (bp)'], color='#4C72B0', ls=':', lw=1); ax[0, 2].set_xlabel('P&L per print on the next real print (bp)'); ax[0, 2].set_title('Re-ranking on the next real trade (blue = S; green = clears the bar)', fontsize=10)
    ax[1, 0].scatter(rr_all['$ per month ($k)'].reindex(tr_all.index), tr_all['$ per month ($k)'], color='#4C72B0')
    for i_ in tr_all.index:
        ax[1, 0].annotate(MID_LETTER[i_], (rr_all.loc[i_, '$ per month ($k)'], tr_all.loc[i_, '$ per month ($k)']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[1, 0].set_xlabel('evaluation markout, dollars per month (k)'); ax[1, 0].set_ylabel('next-print markout, dollars per month (k)'); ax[1, 0].set_title(f'Evaluation vs trade marks, dollars by rule (Spearman {_rho:+.2f})', fontsize=10)
    for kname, c_ in zip(TAPE_KINDS, ['#4C72B0', '#2E8B57', '#DD8452', '#8C8C8C']):
        d_ = pd.Series(TMD[(kname, None)]).dropna()
        if len(d_):
            ax[1, 1].hist(d_.clip(0, CFG.rt_max_days), bins=40, histtype='step', lw=1.4, color=c_, label=f'{kname} (median {d_.median():.1f} d)')
    ax[1, 1].set_xlabel('days from the fill to the next print'); ax[1, 1].set_title('How soon the next real trade arrives, by mark', fontsize=10); ax[1, 1].legend(fontsize=7)
    y2 = np.arange(len(noise)); ax[1, 2].barh(y2, noise['sd of P&L per fill (bp)'], color='#937860'); ax[1, 2].set_yticks(y2); ax[1, 2].set_yticklabels([f'{i_}  (coverage {c_:.0%})' for i_, c_ in zip(noise.index, noise['coverage'])], fontsize=7); ax[1, 2].set_xlabel('sd of P&L per S fill (bp)'); ax[1, 2].set_title('The price of real information: scatter per fill by mark', fontsize=10)
    fig.suptitle('The quote marked to the next real trade instead of the evaluation', fontsize=12); plt.tight_layout(); savefig('11c_trade_markout')
    record('trade_markout', tape_source=TAPE_SRC, tape_prints=int(len(tape)), coverage=flat_records(cov.reset_index()), days_to_mark=flat_records(dte.reset_index()), half_spread_last=_hs_last.round(3).to_dict() if _hs_last is not None else {},
           rerank=tr_all.round(4).reset_index().to_dict(orient='records'), spearman_vs_evaluation={'dollars': _rho, 'per_print': _rho_pp}, by_horizon=flat_records(horizon_tab.reset_index()), common_move=flat_records(common_move.reset_index()), noise=noise.round(4).reset_index().to_dict(orient='records'),
           by_size=flat_records(trade_size.reset_index()) if len(trade_size) else [])
else:
    print('Section 11c skipped: needs Section 11.')

# %% [markdown]
# ### 11b. Fill probability: the desk's curve, conditioned, modelled; the edge model; the quote engine in reduced form
#
# **The desk's fill model.** Production estimates the probability of a fill at a concession $\delta$ as the empirical
# CDF of the basis between the algo yield and the MSRB yield over the trailing 100 days, by side: a curve of basis
# against probability. In this notebook's orientation that is $\hat p(\delta) = \Pr(o \ge \delta)$ with $o$ the
# oriented distance of the print from the quote, estimated on same-side prints strictly before the trade date. It
# is rebuilt here first as it stands, on the algo quote, and checked for calibration by size bin and beta-space
# cluster: a curve pooled across bonds predicts the same fill probability for an odd lot and a block, and Section 9c
# showed the market does not trade that way.
#
# **What conditioning adds.** Four estimators of $\Pr(o \ge \delta \mid x)$ for the S quote at $\delta \in \{-3, 0,
# 3, 6, 10\}$ bp, walk-forward by month and scored on the next month by the Brier score and the log-loss of the fill
# indicator: (i) the desk curve; (ii) a side x size x cluster table on prior months, shrunk toward side x size;
# (iii) a gradient-boosted classifier on trade and quote features (side, size, duration, rating, call, recency, the
# same-side memory); (iv) the same classifier with the factor model added: the three betas, the cluster, the residual
# of record, the state-space signal, the mark-noise estimate, the trailing residual vol. The gap between (iii) and
# (iv) is the factor model's contribution to the fill model, which is the second of the two places it can matter.
# v4.6 adds two pooled estimators that are the question the desk asked: (v) a production-style key (coupon bin x
# rating group x call group x size, the static grouping production pools on) and (vi) the beta-space cluster in its
# place, with each cell's curve shifted by the bond's oriented residual deviance against its cluster peers (the slope
# of the basis on the deviance, estimated in the cell on prior months). (v) is what production does; (vi) is the
# factor model as a dynamic grouping. Section 12d scores the same four on production's own requests.
#
# **The edge model, on the new target.** The level model D predicted fair price; here the same machinery predicts
# what a quote engine needs: the realised round-trip edge of a fill at $\delta = 0$ ($\text{RT} - o$ on the prints S
# would have won), on the same two feature sets, walk-forward. Scored by out-of-sample $R^2$ and by the realised edge
# across predicted deciles.
#
# **The engine in reduced form.** With $\hat p(\delta \mid x)$ and the expected edge $\hat e(x)$, the concession per
# print is $\delta^*(x) = \arg\max_\delta\; \hat p(\delta \mid x)\,(\hat e(x) + \delta + a_s(\delta))$ over the grid, where
# $a_s(\delta)$ is a selection adjustment estimated on prior months by side: the fills kept at a larger concession are
# the prints that were further through the quote, and their base edge is lower on average. Three versions, differing only in the fill
# model (desk, cell, GBM with the factor model), are judged on realised P&L per print on both markout metrics against
# S at zero and the side concession of Section 11, with the usual bar. All model fits are cached on the trade key.
# v4.4 adds the same fill and edge models for the **algo quote** (factor-model feature set), which Section 12c needs to
# replace the desk curve inside the production objective without changing the mid.

# %%
CELL_T('11b. Fill probability: the desks curve, conditioned, modell [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty:
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
    PF_DELTAS = [float(d) for d in CFG.pfill_deltas]; PFD = np.asarray(PF_DELTAS)
    O_S = S_ARR * errS; O_A = S_ARR * errA; TD = mo['trade_date'].to_numpy()
    CL_ARR = mo['beta_cluster'].astype('Int64').astype(str).replace('<NA>', 'NA').to_numpy() if 'beta_cluster' in mo.columns else np.full(len(mo), 'NA')
    MONTHS_MO = sorted(mo['month'].unique())

    def rolling_ecdf_pfill(h_dates: np.ndarray, h_o: np.ndarray, h_side: np.ndarray, q_dates: np.ndarray, q_side: np.ndarray, deltas: list[float], window_days: int, min_n: int = 500) -> np.ndarray:
        """The desk's curve: for each query print, the share of same-side history prints in the trailing window (strictly before the date) with o >= delta."""
        out = np.full((len(q_dates), len(deltas)), np.nan); dl = np.asarray(deltas, float)
        for sd_ in ['P', 'S']:
            hm = (h_side == sd_) & np.isfinite(h_o); hd = h_dates[hm]; ho = h_o[hm]; order = np.argsort(hd, kind='stable'); hd = hd[order]; ho = ho[order]
            qm = np.flatnonzero(q_side == sd_); qd = q_dates[qm]
            for d in np.unique(qd):
                lo = np.searchsorted(hd, d - np.timedelta64(window_days, 'D'), 'left'); hi = np.searchsorted(hd, d, 'left')
                if hi - lo < min_n:
                    continue
                w = np.sort(ho[lo:hi]); out[qm[qd == d]] = 1.0 - np.searchsorted(w, dl, 'left') / len(w)
        return out

    # ---- (1) the production curve on the algo quote, from the full print history, and its calibration at delta = 0
    _h = trades_all[['side', 'trade_date', 'e_algo_bp']].dropna(); _h = _h[_h['side'].isin(['P', 'S'])]
    desk_A = rolling_ecdf_pfill(to_ns(_h['trade_date']).to_numpy(), np.where(_h['side'].to_numpy() == 'P', 1.0, -1.0) * _h['e_algo_bp'].to_numpy(float), _h['side'].astype(str).to_numpy(), TD, SIDE_ARR, [0.0], CFG.pfill_window_days)[:, 0]
    _ok = np.isfinite(desk_A)
    _cal = pd.DataFrame({'side': SIDE_ARR, 'qty_group': QTY_ARR, 'cluster': CL_ARR, 'pred': desk_A, 'real': (O_A >= 0).astype(float)})[_ok]
    desk_cal_size = _cal.groupby(['qty_group', 'side'], observed=True).agg(predicted=('pred', 'mean'), realised=('real', 'mean'), n=('real', 'size')).unstack('side').reindex(QTY_LABELS)
    desk_cal_cl = _cal.groupby(['cluster', 'side'], observed=True).agg(predicted=('pred', 'mean'), realised=('real', 'mean'), n=('real', 'size')).unstack('side')
    desk_cal_cl.index = [CLUSTER_LABEL.get(int(c_), c_) if str(c_).lstrip('-').isdigit() else c_ for c_ in desk_cal_cl.index]
    _gap_size = (desk_cal_size['realised'] - desk_cal_size['predicted']); _gap_cl = (desk_cal_cl['realised'] - desk_cal_cl['predicted'])
    print(f'The desk curve on the algo quote (side, trailing {CFG.pfill_window_days} days) at delta = 0: predicted fill probability vs the realised crossing share, by quantity bin (coverage {_ok.mean():.0%} of customer prints):'); display(desk_cal_size.round(3))
    print('... and by beta-space cluster:'); display(desk_cal_cl.round(3))
    print(f'largest absolute calibration gap of the pooled curve: by size {np.nanmax(np.abs(_gap_size.to_numpy(float))):.3f}, by cluster {np.nanmax(np.abs(_gap_cl.to_numpy(float))):.3f} (a pooled curve cannot be off by less than its own cell dispersion)')

    # ---- (2) four estimators of pfill for the S quote, walk-forward, scored on the next month
    desk_S = rolling_ecdf_pfill(TD, O_S, SIDE_ARR, TD, SIDE_ARR, PF_DELTAS, CFG.pfill_window_days)

    def cell_pfill(tr_mask: np.ndarray, te_mask: np.ndarray, group: np.ndarray, shift: np.ndarray | None = None, o: np.ndarray | None = None, deltas: list[float] | None = None) -> np.ndarray:
        """Pooled fill curve P(o >= delta | side, size, group) from the training prints, shrunk toward side x size with weight n / (n + k).
        With `shift` (an oriented deviance d per print) each cell's curve moves with the bond: pfill(x | d) = share of training prints in the
        cell with o - b d_train >= x - b d_test, b the cell's slope of o on d (shrunk toward 0 with the same weight). Default o is the S quote."""
        o = O_S if o is None else o; deltas = PF_DELTAS if deltas is None else list(deltas); dl = np.asarray(deltas, float)
        kdf = pd.DataFrame({'side': SIDE_ARR, 'qty': QTY_ARR, 'g': np.asarray(group).astype(str)}); trk = kdf[tr_mask].reset_index(drop=True); tek = kdf[te_mask].reset_index(drop=True)
        o_tr = o[tr_mask]; okt = np.isfinite(o_tr); d_tr = shift[tr_mask] if shift is not None else None; d_te = shift[te_mask] if shift is not None else None
        out = np.full((len(tek), len(dl)), np.nan); marg_mat = np.full((len(tek), len(dl)), np.nan)
        key2 = pd.MultiIndex.from_frame(tek[['side', 'qty']])
        for j, dlt in enumerate(dl):
            t2 = trk.assign(y=(o_tr >= dlt).astype(float)); t2 = t2[okt]
            marg = t2.groupby(['side', 'qty'])['y'].mean(); sidem = t2.groupby('side')['y'].mean()
            m2 = marg.reindex(key2).to_numpy(float); ms = sidem.reindex(tek['side']).to_numpy(float); marg_mat[:, j] = np.where(np.isfinite(m2), m2, ms)
        ktr = (trk['side'] + '|' + trk['qty'] + '|' + trk['g']).to_numpy(); kte = (tek['side'] + '|' + tek['qty'] + '|' + tek['g']).to_numpy()
        tr_idx = pd.Series(np.arange(len(trk))).groupby(ktr).indices; te_idx = pd.Series(np.arange(len(tek))).groupby(kte).indices
        cell_mat = np.full((len(tek), len(dl)), np.nan); ww = np.zeros(len(tek))
        for k_, it in te_idx.items():
            itr = tr_idx.get(k_)
            if itr is None:
                continue
            itr = itr[okt[itr]]
            if len(itr) < 5:
                continue
            ww[it] = len(itr) / (len(itr) + CFG.grid_shrink_k); oc = o_tr[itr]
            if shift is None:
                cell_mat[it, :] = (oc[:, None] >= dl[None, :]).mean(axis=0)[None, :]
            else:
                dc = d_tr[itr]; vd = float(np.var(dc)); b = (float(np.cov(oc, dc)[0, 1]) / vd if vd > 1e-9 else 0.0) * (len(itr) / (len(itr) + CFG.grid_shrink_k))
                a_ = np.sort(oc - b * dc); thr = dl[None, :] - b * d_te[it][:, None]
                cell_mat[it, :] = 1.0 - np.searchsorted(a_, thr, 'left') / len(a_)
        c = np.where(np.isfinite(cell_mat), cell_mat, marg_mat)
        return ww[:, None] * c + (1.0 - ww[:, None]) * marg_mat

    # v4.6 grouping variables: the production-style key (coupon bin x rating group x call group) and the oriented residual deviance
    _cpn = pd.to_numeric(mo['cpn'], errors='coerce'); _cpnb = np.where(_cpn.isna(), 'cpn?', 'cpn' + np.clip(np.round(_cpn.fillna(4.0)), 3, 5).astype(int).astype(str))
    _rsc = pd.to_numeric(mo['rating_score'], errors='coerce'); _rg = np.where(_rsc >= 18, 'AAA-AA', np.where(_rsc >= 15, 'A', np.where(_rsc.notna(), 'BBB-', 'NR')))
    _cg = mo['call_structure'].astype(str).replace({'bullet': 'B', 'priced-to-call': 'PC', 'callable, to maturity': 'CM'}).to_numpy() if 'call_structure' in mo.columns else np.full(len(mo), 'na')
    PK_ARR = np.char.add(np.char.add(np.char.add(_cpnb.astype(str), '|'), np.char.add(_rg.astype(str), '|')), _cg.astype(str))
    DEV_ARR = np.clip(S_ARR * (pd.to_numeric(mo['pit_residual'], errors='coerce') / pd.to_numeric(mo['target_abs_vol'], errors='coerce').replace(0, np.nan)).fillna(0.0).to_numpy(float), -3.0, 3.0)
    print(f'production-style key on the universe: {len(np.unique(PK_ARR))} coupon x rating x call groups (x side x size); residual deviance d = s x residual / vol, clipped to +/- 3')
    cellp = np.full((len(mo), len(PF_DELTAS)), np.nan); cellk = np.full((len(mo), len(PF_DELTAS)), np.nan); cellshift = np.full((len(mo), len(PF_DELTAS)), np.nan)
    for m in MONTHS_MO[1:]:
        trm = MONTH_ARR < m; tem = MONTH_ARR == m
        if trm.sum() >= CFG.pfill_min_train and tem.any():
            cellp[tem] = cell_pfill(trm, tem, CL_ARR); cellk[tem] = cell_pfill(trm, tem, PK_ARR); cellshift[tem] = cell_pfill(trm, tem, CL_ARR, DEV_ARR)
    PF_BASE = [c for c in ['side_P', 'log_size', 'modified_duration_lag1', 'rating_score', 'years_to_worst', 'extension', 'is_callable', 'cpn', 'recency_days_c', 'prints_20d', 'last_print_spread_bp', 'MinuteFromSignal', 'dMmdSprdSide', 'ewm_side_err_bp', 'n_side_prints', 'side_err_age_days', 'trade_size_lag', 'print_freq_lag'] if c in mo.columns]
    PF_IPCA = PF_BASE + [c for c in beta_cols + ['beta_cluster', 'pit_residual', 'abs_resid_z', 'ssm_signal', 'ssm_drift', 'eta_abs', 'target_abs_vol', 'residual_age_days'] if c in mo.columns and c not in PF_BASE]

    def _pfill_models():
        parts, imps = [], []; rng = np.random.default_rng(CFG.seed)
        for m in MONTHS_MO[1:]:
            t0 = time.perf_counter(); trm = MONTH_ARR < m; tem = MONTH_ARR == m
            tr_idx = np.flatnonzero(trm); te_idx = np.flatnonzero(tem)
            if len(tr_idx) < CFG.pfill_min_train or not len(te_idx):
                continue
            if len(tr_idx) > CFG.pfill_max_train:
                tr_idx = np.sort(rng.choice(tr_idx, CFG.pfill_max_train, replace=False))
            out = pd.DataFrame({'_id': mo['_id'].to_numpy()[te_idx]})
            for fs_name, feats in [('trade', PF_BASE), ('ipca', PF_IPCA)]:
                feats_u = usable_features(mo.iloc[tr_idx], feats); med = mo.iloc[tr_idx][feats_u].median()
                Xtr = mo.iloc[tr_idx][feats_u].astype(float).fillna(med); Xte = mo.iloc[te_idx][feats_u].astype(float).fillna(med)
                for dlt in PF_DELTAS:
                    ytr = (O_S[tr_idx] >= dlt).astype(int)
                    if ytr.min() == ytr.max():
                        out[f'pf_{fs_name}_{dlt:g}'] = float(ytr.mean()); continue
                    if HAS_LGB:
                        mdl = lgb.LGBMClassifier(n_estimators=CFG.pfill_trees, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=CFG.seed, verbose=-1).fit(Xtr, ytr)
                        imp = pd.Series(mdl.feature_importances_, index=feats_u, dtype=float)
                    else:
                        mdl = HistGradientBoostingClassifier(max_iter=max(100, CFG.pfill_trees), learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, random_state=CFG.seed).fit(Xtr, ytr); imp = None
                    out[f'pf_{fs_name}_{dlt:g}'] = mdl.predict_proba(Xte)[:, 1]
                    if imp is not None and dlt == 0.0:
                        imps.append(imp.rename(f'pfill {fs_name}|{m}'))
                # the edge model: realised round-trip edge of a fill at delta 0 (and the evaluation edge, factor-model set only)
                fill_tr = tr_idx[O_S[tr_idx] >= 0]
                for tgt_name, base_arr in [('rt', BASE_RT), ('eval', BASE_EVAL)]:
                    if tgt_name == 'eval' and fs_name == 'trade':
                        continue
                    y_e = (base_arr - O_S)[fill_tr]; okk = np.isfinite(y_e)
                    if okk.sum() < max(500, CFG.pfill_min_train // 2):
                        out[f'edge_{fs_name}_{tgt_name}'] = np.nan; continue
                    yhat, imp_e = fit_gbm(mo.iloc[fill_tr[okk]].assign(_y=y_e[okk]), mo.iloc[te_idx], feats, '_y', CFG.pfill_trees)
                    out[f'edge_{fs_name}_{tgt_name}'] = yhat
                    if imp_e is not None and tgt_name == 'rt':
                        imps.append(imp_e.astype(float).rename(f'edge {fs_name}|{m}'))
            # v4.4: fill and edge models for the ALGO quote (factor-model feature set only), consumed by the production objective of Section 12c
            feats_u = usable_features(mo.iloc[tr_idx], PF_IPCA); med = mo.iloc[tr_idx][feats_u].median()
            Xtr = mo.iloc[tr_idx][feats_u].astype(float).fillna(med); Xte = mo.iloc[te_idx][feats_u].astype(float).fillna(med)
            for dlt in PF_DELTAS:
                ytr = (O_A[tr_idx] >= dlt).astype(int)
                if ytr.min() == ytr.max():
                    out[f'pf_ipcaA_{dlt:g}'] = float(ytr.mean()); continue
                if HAS_LGB:
                    mdl = lgb.LGBMClassifier(n_estimators=CFG.pfill_trees, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=CFG.seed, verbose=-1).fit(Xtr, ytr)
                else:
                    mdl = HistGradientBoostingClassifier(max_iter=max(100, CFG.pfill_trees), learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, random_state=CFG.seed).fit(Xtr, ytr)
                out[f'pf_ipcaA_{dlt:g}'] = mdl.predict_proba(Xte)[:, 1]
            fill_trA = tr_idx[O_A[tr_idx] >= 0]; y_eA = (BASE_RT - O_A)[fill_trA]; okA = np.isfinite(y_eA)
            out['edge_ipcaA_rt'] = fit_gbm(mo.iloc[fill_trA[okA]].assign(_y=y_eA[okA]), mo.iloc[te_idx], PF_IPCA, '_y', CFG.pfill_trees)[0] if okA.sum() >= max(500, CFG.pfill_min_train // 2) else np.nan
            parts.append(out); print(f'  fill and edge models {m}: train {len(tr_idx):,} | test {len(te_idx):,} | {time.perf_counter() - t0:.0f}s')
        pred = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['_id'])
        imp_df = pd.concat(imps, axis=1).reset_index().rename(columns={'index': 'feature'}) if imps else pd.DataFrame(columns=['feature'])
        return {'pred': pred, 'imp': imp_df}

    PF = cached('pfill_models', _pfill_models, deps=[EC_FP, PF_BASE, PF_IPCA, PF_DELTAS, CFG.pfill_trees, CFG.pfill_max_train, STAGE_KEYS.get('trades_std'), f'rt:{np.nansum(BASE_RT):.6g}:{int(np.isfinite(BASE_RT).sum())}:{TAPE_SRC}'], code=[fit_gbm, usable_features])   # v4.8: the round-trip target is a dependency (its exit source changed)
    pred = PF['pred'].set_index('_id').reindex(mo['_id'].to_numpy()) if len(PF['pred']) else pd.DataFrame(index=mo['_id'].to_numpy())
    def _mat(prefix: str) -> np.ndarray:
        cols = [f'{prefix}_{d:g}' for d in PF_DELTAS]
        return pred[cols].to_numpy(float) if all(c in pred.columns for c in cols) else np.full((len(mo), len(PF_DELTAS)), np.nan)
    METHODS = {'desk: side, trailing window': desk_S, 'cell: side x size x production-style key': cellk, 'cell: side x size x cluster, prior months': cellp, 'cell: cluster, shifted by residual deviance': cellshift, 'GBM: trade and quote features': _mat('pf_trade'), 'GBM: + IPCA and state-space': _mat('pf_ipca')}
    common = tested.copy()
    for mt in METHODS.values():
        common &= np.isfinite(mt).all(axis=1)
    if common.sum() < 200:
        print(f'Section 11b: only {int(common.sum())} prints have every fill estimator available; scoring, the edge model and the engine are skipped on this run.')
        record('pfill', deltas=PF_DELTAS, window_days=CFG.pfill_window_days, desk_calibration_by_size=flat_records(desk_cal_size.reset_index()), desk_calibration_by_cluster=flat_records(desk_cal_cl.reset_index()), scores=[], brier_relative_to_desk=[], edge_model=[], engine=[])
    else:
        sc_rows = []
        for name, mt in METHODS.items():
            for j, dlt in enumerate(PF_DELTAS):
                y = (O_S >= dlt).astype(float); p = np.clip(mt[:, j], 1e-4, 1 - 1e-4)
                for sd_, msk in [('ALL', common), ('P', common & IS_P), ('S', common & IS_S)]:
                    sc_rows.append({'method': name, 'delta (bp)': dlt, 'side': sd_, 'Brier': float(np.mean((p[msk] - y[msk]) ** 2)), 'log-loss': float(-np.mean(y[msk] * np.log(p[msk]) + (1 - y[msk]) * np.log(1 - p[msk]))), 'mean predicted': float(p[msk].mean()), 'realised': float(y[msk].mean()), 'n': int(msk.sum())})
        pf_scores = pd.DataFrame(sc_rows)
        brier_tab = pf_scores[pf_scores['side'] == 'ALL'].pivot(index='method', columns='delta (bp)', values='Brier').reindex(list(METHODS))
        brier_rel = (1.0 - brier_tab.div(brier_tab.loc['desk: side, trailing window'], axis=1))
        print(f'Fill probability for the S quote on the next month ({int(common.sum()):,} prints with every estimator available): Brier score by concession (lower is better):'); display(brier_tab.round(4))
        print('Brier improvement over the desk curve (share of its Brier removed):'); display(brier_rel.round(3))
        print('By side at delta = 0 (Brier, log-loss, mean predicted vs realised):'); display(pf_scores[(pf_scores['delta (bp)'] == 0.0) & (pf_scores['side'] != 'ALL')].set_index(['method', 'side'])[['Brier', 'log-loss', 'mean predicted', 'realised', 'n']].round(4))
        # calibration deciles at delta = 0: desk vs the factor-model GBM
        cal_rows = []
        for name in ['desk: side, trailing window', 'GBM: + IPCA and state-space']:
            p0 = METHODS[name][:, PF_DELTAS.index(0.0)]; y0 = (O_S >= 0).astype(float)
            for sd_, msk in [('P', common & IS_P), ('S', common & IS_S)]:
                dec = pd.qcut(pd.Series(p0[msk]).rank(method='first'), 10, labels=False) + 1
                g = pd.DataFrame({'dec': dec.to_numpy(), 'p': p0[msk], 'y': y0[msk]}).groupby('dec').agg(predicted=('p', 'mean'), realised=('y', 'mean'), n=('y', 'size')).reset_index()
                g['method'] = name; g['side'] = sd_; cal_rows.append(g)
        pf_cal = pd.concat(cal_rows, ignore_index=True)

        # ---- (3) the edge model: realised round-trip edge of an S fill by predicted decile, out-of-sample R2
        ed_rows, ed_dec = [], []
        for col, lab in [('edge_trade_rt', 'round-trip edge, trade features'), ('edge_ipca_rt', 'round-trip edge, + IPCA / state-space'), ('edge_ipca_eval', 'evaluation edge, + IPCA / state-space')]:
            if col not in pred.columns:
                continue
            base_arr = BASE_RT if col.endswith('_rt') else BASE_EVAL
            y = base_arr - O_S; yhat = pred[col].to_numpy(float); fill0 = O_S >= 0
            for sd_, msk in [('ALL', tested & fill0), ('P', tested & fill0 & IS_P), ('S', tested & fill0 & IS_S)]:
                ok = msk & np.isfinite(y) & np.isfinite(yhat)
                if ok.sum() < 1_000:
                    continue
                sse = float(np.sum((y[ok] - yhat[ok]) ** 2)); sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
                ic = pd.DataFrame({'d': TD[ok], 'y': y[ok], 'p': yhat[ok]}).groupby('d').apply(lambda g: g['y'].corr(g['p'], method='spearman') if len(g) > 30 else np.nan, include_groups=False).dropna()
                ed_rows.append({'model': lab, 'side': sd_, 'fills': int(ok.sum()), 'OOS R2': 1.0 - sse / sst if sst > 0 else np.nan, 'MAE (bp)': float(np.mean(np.abs(y[ok] - yhat[ok]))), 'daily rank IC': float(ic.mean()), 'IC t': float(np.sqrt(len(ic)) * ic.mean() / ic.std()) if ic.std() > 0 else np.nan})
                if sd_ != 'ALL' and col == 'edge_ipca_rt':
                    dec = pd.qcut(pd.Series(yhat[ok]).rank(method='first'), 10, labels=False) + 1
                    g = pd.DataFrame({'dec': dec.to_numpy(), 'y': y[ok], 'p': yhat[ok]}).groupby('dec').agg(predicted=('p', 'mean'), realised=('y', 'mean'), n=('y', 'size')).reset_index(); g['side'] = sd_; ed_dec.append(g)
        edge_scores = pd.DataFrame(ed_rows).set_index(['model', 'side']) if ed_rows else pd.DataFrame()
        edge_dec = pd.concat(ed_dec, ignore_index=True) if ed_dec else pd.DataFrame()
        if len(edge_scores):
            print('The edge model on the new target: realised edge of an S fill at delta 0, out of sample (R2 against the training mean; daily rank IC):'); display(edge_scores.round(3))
        if len(edge_dec):
            print('Realised round-trip edge of S fills by predicted decile (bp), factor-model feature set:'); display(edge_dec.pivot(index='dec', columns='side', values=['predicted', 'realised']).round(2))

        # ---- (4) the engine in reduced form: delta per print = argmax pfill(delta | x) x (expected edge + delta)
        e_hat = pred['edge_ipca_rt'].to_numpy(float) if 'edge_ipca_rt' in pred.columns else np.full(len(mo), np.nan)

        # selection adjustment, by side, on prior months: a larger concession keeps the prints that were further through the quote,
        # whose base edge (RT - o) is lower on average; ADJ[i, j] = mean base edge of fills kept at delta_j minus at delta 0
        ADJ = np.zeros((len(mo), len(PF_DELTAS))); _base_e = BASE_RT - O_S
        for m in MONTHS_MO[1:]:
            trm = MONTH_ARR < m; tem = MONTH_ARR == m
            for sd_, msk in [('P', IS_P), ('S', IS_S)]:
                sel0 = trm & msk & np.isfinite(_base_e)
                if (sel0 & (O_S >= 0)).sum() < max(200, CFG.pfill_min_train // 10):
                    continue
                m0 = float(_base_e[sel0 & (O_S >= 0)].mean())
                for j, dlt in enumerate(PF_DELTAS):
                    selj = sel0 & (O_S >= dlt); ADJ[tem & msk, j] = (float(_base_e[selj].mean()) - m0) if selj.sum() >= max(100, CFG.pfill_min_train // 20) else 0.0
        adj_tab = pd.DataFrame({sd_: ADJ[(MONTH_ARR == MONTHS_MO[-1]) & msk].mean(axis=0) for sd_, msk in [('P', IS_P), ('S', IS_S)]}, index=[f'delta {d:g}' for d in PF_DELTAS])
        print('Selection adjustment to the expected edge by concession and side (bp; the fills kept at a larger concession carry less base edge), last fold:'); display(adj_tab.round(2))

        def engine_delta(pf_mat: np.ndarray) -> np.ndarray:
            val = pf_mat * (e_hat[:, None] + PFD[None, :] + ADJ); val = np.where(np.isfinite(val), val, -np.inf)
            j = np.argmax(val, axis=1); return np.where(np.isfinite(val).any(axis=1), PFD[j], 0.0)

        eng_mask = tested & common & np.isfinite(e_hat)
        if eng_mask.sum() < 200:
            print(f'engine: only {int(eng_mask.sum())} prints carry every input (fill models and the edge model); the policy test is skipped on this run.')
            eng_mask = np.zeros(len(mo), bool)
        POLICIES = {'S at 0': 0.0, 'S + delta* per side (Section 11, walk-forward)': delta_side,
                    'engine: desk pfill x edge model': engine_delta(METHODS['desk: side, trailing window']), 'engine: cell pfill x edge model': engine_delta(METHODS['cell: side x size x cluster, prior months']),
                    'engine: GBM + IPCA pfill x edge model': engine_delta(METHODS['GBM: + IPCA and state-space'])}
        eng_rows, eng_daily = [], {}
        for name, dl in POLICIES.items():
            r_rt = pnl_with(errS, dl, eng_mask, BASE_RT); r_ev = pnl_with(errS, dl, eng_mask, BASE_EVAL)
            eng_daily[name] = (r_rt.pop('_daily'), r_ev.pop('_daily'))
            eng_rows.append({'policy': name, 'mean concession (bp)': float(np.mean(np.broadcast_to(dl, len(mo))[eng_mask])) if eng_mask.any() else np.nan, 'fill share': r_rt['fill share'], 'round trip: P&L per print (bp)': r_rt['P&L per print (bp)'], 'round trip: $ per month ($k)': r_rt['$ per month ($k)'],
                             'evaluation: P&L per print (bp)': r_ev['P&L per print (bp)'], 'evaluation: $ per month ($k)': r_ev['$ per month ($k)']})
        engine = pd.DataFrame(eng_rows).set_index('policy')
        _b_rt, _b_ev = eng_daily['S at 0']
        for name in engine.index:
            d_rt = (eng_daily[name][0] - _b_rt).dropna(); d_ev = (eng_daily[name][1] - _b_ev).dropna(); dm = d_rt.groupby(pd.DatetimeIndex(d_rt.index).to_period('M')).mean()
            engine.loc[name, 'round trip: boot t vs S at 0'] = block_bootstrap_t(d_rt.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if name != 'S at 0' else np.nan
            engine.loc[name, 'evaluation: boot t vs S at 0'] = block_bootstrap_t(d_ev.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if name != 'S at 0' else np.nan
            engine.loc[name, 'months above S (round trip)'] = int((dm > 0).sum()) if name != 'S at 0' else np.nan
        engine['beats S (bar, round trip)'] = ((engine['round trip: P&L per print (bp)'] - engine.loc['S at 0', 'round trip: P&L per print (bp)']) >= CFG.interaction_bar_bp) & (engine['round trip: boot t vs S at 0'] > 3) & (engine['months above S (round trip)'] >= min(4, int(pd.Series(MONTH_ARR[eng_mask]).nunique())))
        print(f'The quote engine in reduced form, walk-forward on {int(eng_mask.sum()):,} prints where every input exists; bar: +{CFG.interaction_bar_bp:.2f} bp per print on the round trip over S at 0, bootstrap t > 3, positive in >= 4 months:'); display(engine.round(3))
        _dmix = pd.DataFrame({name: pd.Series(np.broadcast_to(dl, len(mo))[eng_mask]).value_counts(normalize=True).sort_index() for name, dl in POLICIES.items() if np.ndim(dl)}).fillna(0.0)
        print('Mix of concessions chosen by each policy (share of prints):'); display(_dmix.round(3))

        # ---- figures
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        for sd_, c_ in [('P', '#4C72B0'), ('S', '#DD8452')]:
            if sd_ in desk_cal_size['predicted'].columns:
                x = np.arange(len(desk_cal_size)); ax[0, 0].plot(x, desk_cal_size[('predicted', sd_)], ls='--', marker='.', color=c_, label=f'{sd_}: desk curve predicted'); ax[0, 0].plot(x, desk_cal_size[('realised', sd_)], marker='o', color=c_, label=f'{sd_}: realised crossing share')
            if sd_ in desk_cal_cl['predicted'].columns:
                x2 = np.arange(len(desk_cal_cl)); ax[0, 1].plot(x2, desk_cal_cl[('predicted', sd_)], ls='--', marker='.', color=c_, label=f'{sd_}: predicted'); ax[0, 1].plot(x2, desk_cal_cl[('realised', sd_)], marker='o', color=c_, label=f'{sd_}: realised')
        ax[0, 0].set_xticks(np.arange(len(desk_cal_size))); ax[0, 0].set_xticklabels(desk_cal_size.index, rotation=30, fontsize=8); ax[0, 0].set_title('The desk curve on the algo quote at delta 0: pooled prediction vs realised fills, by size', fontsize=10); ax[0, 0].legend(fontsize=7); ax[0, 0].set_ylabel('fill probability')
        ax[0, 1].set_xticks(np.arange(len(desk_cal_cl))); ax[0, 1].set_xticklabels([str(i_)[:14] for i_ in desk_cal_cl.index], rotation=30, fontsize=7); ax[0, 1].set_title('... by beta-space cluster', fontsize=10); ax[0, 1].legend(fontsize=7)
        for name, c_ in zip(METHODS, ['#8C8C8C', '#C44E52', '#937860', '#8172B2', '#4C72B0', '#2E8B57']):
            ax[0, 2].plot(PF_DELTAS, brier_tab.loc[name], marker='o', color=c_, label=name)
        ax[0, 2].set_xlabel('concession delta (bp)'); ax[0, 2].set_ylabel('Brier score (next month)'); ax[0, 2].set_title('Fill-probability estimators for the S quote: Brier by concession', fontsize=10); ax[0, 2].legend(fontsize=7)
        for (name, sd_), g in pf_cal.groupby(['method', 'side']):
            ax[1, 0].plot(g['predicted'], g['realised'], marker='o', ls='-' if name.startswith('GBM') else '--', color='#4C72B0' if sd_ == 'P' else '#DD8452', label=f'{name.split(":")[0]} {sd_}')
        ax[1, 0].plot([0, 1], [0, 1], 'k:', lw=0.8); ax[1, 0].set_xlabel('predicted fill probability (decile mean)'); ax[1, 0].set_ylabel('realised'); ax[1, 0].set_title('Calibration at delta 0: desk curve (dashed) vs GBM + IPCA (solid)', fontsize=10); ax[1, 0].legend(fontsize=7)
        if len(edge_dec):
            for sd_, g in edge_dec.groupby('side'):
                ax[1, 1].plot(g['dec'], g['realised'], marker='o', color='#4C72B0' if sd_ == 'P' else '#DD8452', label=f'{sd_}: realised'); ax[1, 1].plot(g['dec'], g['predicted'], ls='--', color='#4C72B0' if sd_ == 'P' else '#DD8452', label=f'{sd_}: predicted')
            ax[1, 1].set_xlabel('predicted-edge decile'); ax[1, 1].set_ylabel('round-trip edge of S fills (bp)'); ax[1, 1].set_title('The edge model on the new target: realised edge by predicted decile', fontsize=10); ax[1, 1].legend(fontsize=7)
        else:
            ax[1, 1].text(0.5, 0.5, 'edge model unavailable', ha='center', va='center', transform=ax[1, 1].transAxes); ax[1, 1].set_axis_off()
        y_ = np.arange(len(engine)); v_ = np.nan_to_num(engine['round trip: P&L per print (bp)'].to_numpy(float))
        ax[1, 2].barh(y_, v_, color=['#2E8B57' if b_ else '#8C8C8C' for b_ in engine['beats S (bar, round trip)']]); ax[1, 2].set_yticks(y_); ax[1, 2].set_yticklabels([f'{i_}  (t {t_:+.1f})' if np.isfinite(t_) else i_ for i_, t_ in zip(engine.index, engine['round trip: boot t vs S at 0'])], fontsize=7)
        ax[1, 2].axvline(np.nan_to_num(engine.loc['S at 0', 'round trip: P&L per print (bp)']), color='k', ls='--', lw=0.8); ax[1, 2].set_xlabel('round-trip P&L per print (bp)'); ax[1, 2].set_title('The engine in reduced form vs S at 0 (green = clears the bar)', fontsize=10)
        plt.tight_layout(); savefig('11b_pfill_and_engine')
        if len(PF['imp']):
            _imp = PF['imp'].set_index('feature'); fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
            for a, pre, ttl in zip(ax, ['pfill ipca', 'edge ipca'], ['Fill model (delta 0, + IPCA / state-space): mean importance', 'Edge model (round trip, + IPCA / state-space): mean importance']):
                cols = [c for c in _imp.columns if c.startswith(pre)]
                if cols:
                    _imp[cols].mean(axis=1).sort_values().tail(15).plot.barh(ax=a, color='#4C72B0'); a.set_title(ttl, fontsize=10)
                else:
                    a.set_axis_off()
            plt.tight_layout(); savefig('11b_importances')
        record('pfill', deltas=PF_DELTAS, window_days=CFG.pfill_window_days, desk_calibration_by_size=flat_records(desk_cal_size.reset_index()), desk_calibration_by_cluster=flat_records(desk_cal_cl.reset_index()), scores=pf_scores.round(5).to_dict(orient='records'),
               brier_relative_to_desk=flat_records(brier_rel.reset_index()), calibration_deciles=pf_cal.round(4).to_dict(orient='records'), features_trade=PF_BASE, features_ipca=PF_IPCA,
               edge_model=edge_scores.round(4).reset_index().to_dict(orient='records') if len(edge_scores) else [], edge_deciles=edge_dec.round(4).to_dict(orient='records') if len(edge_dec) else [],
               engine=engine.round(4).reset_index().to_dict(orient='records'), concession_mix=flat_records(_dmix.reset_index()))
else:
    print('Section 11b skipped: needs Section 11.')

# %% [markdown]
# ## 12. Quantile grid: the conditional distribution the optimizer consumes
#
# The optimizer needs more than a point mid: for each quote it needs the distribution of where the print will land
# relative to the quote, because the fill probability at a concession $\delta$ is one minus that distribution's
# CDF at $\delta$, and the bias is its median. This section estimates that distribution as a **grid of quantiles of
# the oriented error of S** by side x desk quantity bin x beta-space cluster, on prior held-out months, shrunk
# toward the side x size marginal with weight $n / (n + k)$ where a cell is thin. It is evaluated the only way a
# quantile should be: by the realised share of test prints below each predicted quantile (calibration), by side.
# A well-calibrated grid is the fill curve of Section 11 in closed form. The final grid, estimated on every
# held-out month, is written to `artifacts_v3/quote_quantile_grid.parquet` with its cell sizes and shrinkage
# weights, together with the marginal side x size grid as the fallback.

# %%
CELL_T('12. Quantile grid: the conditional distribution the optimize [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty:
    TAUS = list(CFG.grid_taus)
    mo['cluster_id'] = mo['beta_cluster'].astype('Int64').astype(str).replace('<NA>', 'NA') if 'beta_cluster' in mo.columns else 'NA'
    mo['o_S'] = mo['s'] * mo['e_S_bp']; mo['o_A'] = mo['s'] * mo['e_algo_bp']
    CELL = ['side', 'qty_group', 'cluster_id']

    def build_grid(train: pd.DataFrame, ocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cell = train.groupby(CELL, observed=True)[ocol].quantile(TAUS).unstack(); cell.columns = TAUS
        n = train.groupby(CELL, observed=True).size().reindex(cell.index).fillna(0.0)
        marg = train.groupby(['side', 'qty_group'], observed=True)[ocol].quantile(TAUS).unstack(); marg.columns = TAUS
        w = n / (n + CFG.grid_shrink_k)
        m2 = marg.reindex(pd.MultiIndex.from_arrays([cell.index.get_level_values(0), cell.index.get_level_values(1)])); m2.index = cell.index
        g = cell.mul(w, axis=0) + m2.mul(1.0 - w, axis=0)
        g['n'] = n; g['shrink_w'] = w
        return g, marg

    def lookup(g: pd.DataFrame, marg: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([te['side'].astype(str).to_numpy(), te['qty_group'].astype(str).to_numpy(), te['cluster_id'].astype(str).to_numpy()])
        q = g.reindex(key)[TAUS].to_numpy(dtype=float, copy=True)   # copy=True: to_numpy may return a read-only view, and q is mutated below
        qm = marg.reindex(pd.MultiIndex.from_arrays([te['side'].astype(str).to_numpy(), te['qty_group'].astype(str).to_numpy()]))[TAUS].to_numpy(float)
        miss = ~np.isfinite(q[:, 0])
        q[miss] = qm[miss]
        return q

    cal_rows, fill_rows = [], []
    months_g = sorted(mo['month'].unique())
    for m in months_g[1:]:
        tr, te = mo[mo['month'] < m], mo[mo['month'] == m]
        if len(tr) < 5_000 or te.empty:
            continue
        g, marg = build_grid(tr, 'o_S'); q = lookup(g, marg, te); o = te['o_S'].to_numpy(float)
        for j, tau in enumerate(TAUS):
            for sd_ in ['P', 'S']:
                msk = (te['side'] == sd_).to_numpy() & np.isfinite(q[:, j]) & np.isfinite(o)
                if msk.sum() >= 200:
                    cal_rows.append({'month': str(m), 'side': sd_, 'tau': tau, 'realised share below q_tau': float((o[msk] <= q[msk, j]).mean()), 'n': int(msk.sum())})
    calib = pd.DataFrame(cal_rows)
    if len(calib):
        cal_tab = calib.groupby(['side', 'tau'])['realised share below q_tau'].mean().unstack('tau'); cal_tab.columns = [f'tau {t:.2f}' for t in cal_tab.columns]
        cal_month = calib.pivot_table(index='month', columns='side', values='realised share below q_tau', aggfunc=lambda x: float(np.mean(np.abs(x - calib.loc[x.index, 'tau']))))
        print('Calibration of the quantile grid on the next month (realised share of prints below the predicted quantile; a calibrated grid reads the column header):'); display(cal_tab.round(3))
        print('Mean absolute calibration error by month and side:'); display(cal_month.round(3))
    # the final grid on every held-out month, the deliverable
    g_final, marg_final = build_grid(mo, 'o_S'); g_A, marg_A = build_grid(mo, 'o_A')
    grid_long = g_final.reset_index().melt(id_vars=CELL + ['n', 'shrink_w'], value_vars=TAUS, var_name='tau', value_name='q_bp').assign(quote='S same-side EWMA')
    grid_long_A = g_A.reset_index().melt(id_vars=CELL + ['n', 'shrink_w'], value_vars=TAUS, var_name='tau', value_name='q_bp').assign(quote='A algo quote')
    pd.concat([grid_long, grid_long_A], ignore_index=True).to_parquet(ARTIFACTS / 'quote_quantile_grid.parquet', index=False)
    marg_out = pd.concat({'S same-side EWMA': marg_final, 'A algo quote': marg_A}, names=['quote']).reset_index()
    marg_out.to_parquet(ARTIFACTS / 'quote_quantile_grid_marginal.parquet', index=False)
    med = marg_final[0.5].unstack('side').reindex(QTY_LABELS); iqr = (marg_final[0.75] - marg_final[0.25]).unstack('side').reindex(QTY_LABELS)
    print('The marginal grid for S by side x quantity bin: median oriented error (bp; positive = the print lands on the aggressive side of the quote, i.e. the quote is too generous) and the interquartile width:')
    display(pd.concat({'median (bp)': med, 'IQR width (bp)': iqr}, axis=1).round(2))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    if len(calib):
        for sd_, c_ in [('P', '#4C72B0'), ('S', '#DD8452')]:
            t_ = calib[calib['side'] == sd_].groupby('tau')['realised share below q_tau'].mean()
            ax[0].plot(t_.index, t_.values, marker='o', color=c_, label={'P': 'P dealer buys (bid)', 'S': 'S dealer sells (offer)'}[sd_])
        ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8); ax[0].set_xlabel('nominal quantile tau'); ax[0].set_ylabel('realised share of next-month prints below q_tau'); ax[0].set_title('Calibration of the grid out of sample (on the diagonal = calibrated)', fontsize=10); ax[0].legend(fontsize=8)
    else:
        ax[0].text(0.5, 0.5, 'calibration needs at least two held-out months\nbefore the test month', ha='center', va='center', fontsize=10, transform=ax[0].transAxes); ax[0].set_axis_off()
    x = np.arange(len(med))
    for sd_, c_ in [('P', '#4C72B0'), ('S', '#DD8452')]:
        if sd_ in med.columns:
            q25 = marg_final[0.25].unstack('side').reindex(QTY_LABELS)[sd_]; q75 = marg_final[0.75].unstack('side').reindex(QTY_LABELS)[sd_]
            ax[1].fill_between(x, q25, q75, color=c_, alpha=0.15); ax[1].plot(x, med[sd_], marker='o', color=c_, lw=1.8, label={'P': 'P bid: median (line), 25th to 75th (band)', 'S': 'S offer: median (line), 25th to 75th (band)'}[sd_])
    ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xticks(x); ax[1].set_xticklabels(med.index, rotation=30, fontsize=8); ax[1].set_title('Oriented error of S by quantity bin and side: where the print lands relative to the quote', fontsize=10); ax[1].set_ylabel('bp (positive = print on the aggressive side of the quote)'); ax[1].legend(fontsize=8)
    _ex = g_final.reset_index(); _ex = _ex[_ex['side'] == 'P']
    _piv = _ex.pivot_table(index='qty_group', columns='cluster_id', values=0.5).reindex(QTY_LABELS)
    im = ax[2].imshow(_piv.to_numpy(float), cmap='RdYlGn_r', aspect='auto'); ax[2].set_xticks(range(_piv.shape[1])); ax[2].set_xticklabels([f'C{c_}' for c_ in _piv.columns], fontsize=8); ax[2].set_yticks(range(len(_piv))); ax[2].set_yticklabels(_piv.index, fontsize=8); ax[2].grid(False)
    for r_ in range(_piv.shape[0]):
        for c_ in range(_piv.shape[1]):
            v = _piv.iloc[r_, c_]
            if np.isfinite(v):
                ax[2].text(c_, r_, f'{v:+.1f}', ha='center', va='center', fontsize=7)
    ax[2].set_title('Median oriented error of S on the bid, by quantity bin x beta-space cluster (bp, shrunk grid)', fontsize=10); plt.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout(); savefig('12_quantile_grid')
    record('quantile_grid', taus=TAUS, cells=int(len(g_final)), calibration=cal_tab.round(4).reset_index().to_dict(orient='records') if len(calib) else [], calibration_error_by_month=cal_month.round(4).reset_index().to_dict(orient='records') if len(calib) else [],
           marginal_median=flat_records(med.reset_index()), marginal_iqr=flat_records(iqr.reset_index()))
else:
    print('Section 12 skipped: needs Section 11.')

# %% [markdown]
# ### 12b. The residual side: the forecast as a concession modifier, the mark noise as a width scalar
#
# Two pre-registered tests close the questions in "thoughts 3". **(a) Does the state-space forecast predict the
# hedged markout, and can it size a concession?** The forecast of the next residual is extended to the record
# horizon ($\hat f_h = \hat y_{t+1} + \hat m \sum_{k=2}^{h} \bar\phi^k$), oriented so that positive is favourable to
# the dealer's position ($-s\,\hat f_h$), and ranked within date x beta-space cluster. If it carries information, the
# hedged move after the print should rise across its quintiles, separately on bids and offers, and a quote shift of
# $-\kappa$ times the oriented forecast (clipped) with $\kappa$ chosen on prior months should raise fill-weighted
# P&L over S with its side concession. The bar is the usual one: +0.10 bp per print, bootstrap t > 3, four of five
# months. Three inputs are tried, as proposed: the full forecast, the drift alone, and the mark-noise magnitude as a
# pure widening. **(b) Does the mark-noise estimate scale the width of the error distribution?** The quantile grid is
# rebuilt by mark-noise tercile; if the pooled grid under-covers the tails on noisy bonds and the split grid fixes it
# by at least 0.01 of coverage on both sides, the scalar is adopted into the grid. The prior for (a) is low: the
# hedged residual move after a fill is a fraction of a basis point and the signal's rank IC is 0.08. The prior for
# (b) is higher: the mark-noise estimate explains how far prints sit from the evaluation (Section 9).

# %%
CELL_T('12b. The residual side: the forecast as a concession modifie [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty and 'ssm_signal' in mo.columns and mo['ssm_signal'].notna().mean() > 0.3:
    phi_bar = float(ssm_params['phi'].iloc[-1]) if 'ssm_params' in globals() and not ssm_params.empty and 'phi' in ssm_params.columns else 0.99
    _mult = sum(phi_bar ** k for k in range(2, HR + 1))
    mo['f_h'] = mo['ssm_signal'].astype(float) + mo['ssm_drift'].astype(float).fillna(0.0) * _mult      # cumulative residual forecast over the record horizon (bp)
    mo['f_or'] = -S_ARR * mo['f_h']                                                                     # oriented: positive = favourable to the dealer's position
    mo['f_or_drift'] = -S_ARR * mo['ssm_drift'].astype(float) * (phi_bar + _mult)
    mo['cl_key'] = mo['beta_cluster'].astype('Int64').astype(str)
    mo['f_rank'] = mo.groupby(['residual_date', 'cl_key'], observed=True)['f_or'].rank(pct=True)
    HDR = HD[HR]; fillS = (S_ARR * errS) >= 0
    # (a1) does the oriented forecast predict the hedged move? quintiles within date x cluster, by side; FM slope in bp per bp
    q_rows = []
    for sd_, msk in [('P dealer buys (bid)', IS_P), ('S dealer sells (offer)', IS_S)]:
        ok = msk & np.isfinite(HDR) & mo['f_rank'].notna().to_numpy()
        qb = pd.cut(mo.loc[ok, 'f_rank'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Q1 most adverse', 'Q2', 'Q3', 'Q4', 'Q5 most favourable'], include_lowest=True)
        g = pd.DataFrame({'q': qb.astype(str), 'f': mo.loc[ok, 'f_or'].to_numpy(), 'hd': HDR[ok], 'fillS': fillS[ok]}).groupby('q', observed=True)
        t_ = pd.DataFrame({'forecast, oriented (bp)': g['f'].mean(), 'hedged move after, all prints (bp)': g['hd'].mean(), 'hedged move after, S fills (bp)': g.apply(lambda d: d.loc[d['fillS'], 'hd'].mean()), 'S fill share': g['fillS'].mean(), 'n': g.size()})
        t_['side'] = sd_; q_rows.append(t_.reset_index())
    sig_q = pd.concat(q_rows, ignore_index=True).set_index(['side', 'q'])
    print(f'Does the oriented state-space forecast predict the hedged move after the print (h = {HR})? Quintiles of its rank within date x beta cluster:'); display(sig_q.round(3))
    _fr = mo.assign(hd=HDR)
    sig_slope = pd.concat([fm_multi(_fr[IS_P & np.isfinite(HDR)], 'hd', ['f_or'], ['log_size']).assign(side='P', input='full forecast'), fm_multi(_fr[IS_S & np.isfinite(HDR)], 'hd', ['f_or'], ['log_size']).assign(side='S', input='full forecast'),
                           fm_multi(_fr[IS_P & np.isfinite(HDR)], 'hd', ['f_or_drift'], ['log_size']).assign(side='P', input='drift only'), fm_multi(_fr[IS_S & np.isfinite(HDR)], 'hd', ['f_or_drift'], ['log_size']).assign(side='S', input='drift only')], ignore_index=True)
    print('Fama-MacBeth slope of the hedged move after the print on the oriented forecast (bp of realised move per bp of forecast; 1.0 = the forecast is right on average):'); display(sig_slope[['side', 'input', 'fm_beta', 'fm_t', 'boot_t', 'fm_dates', 'n']].round(3))
    # (a2) the concession policy, walk-forward: quote shift = -kappa x oriented forecast (clipped), kappa on prior months, on top of S and its side concession
    def policy_pnl(shift: np.ndarray, mask: np.ndarray) -> tuple[float, pd.Series]:
        o = S_ARR * errS; dl = delta_side + shift; fill = o >= dl; pnl = BASE_EVAL - o + dl; ok = mask & np.isfinite(pnl); f_ok = fill & ok
        d_ = pd.Series(np.where(f_ok, pnl, 0.0)[ok]).groupby(mo['trade_date'].to_numpy()[ok]).mean()
        return float(d_.mean()), d_
    inputs = {'full forecast': mo['f_or'].to_numpy(float), 'drift only': mo['f_or_drift'].to_numpy(float), 'mark noise |eta| (widening)': -mo['eta_abs'].astype(float).to_numpy()}
    pol_rows, pol_daily = [], {}
    base_pnl, base_daily = policy_pnl(np.zeros(len(mo)), tested); pol_daily['S + side concession'] = base_daily
    for name, x in inputs.items():
        x = np.where(np.isfinite(x), x, 0.0); shift_wf = np.zeros(len(mo)); kap_path = {}
        for m in sorted(mo['month'].unique())[1:]:
            trm = MONTH_ARR < m; tem = MONTH_ARR == m
            best_k, best_v = 0.0, -np.inf
            for kap in CFG.signal_kappas:
                v, _ = policy_pnl(np.clip(-kap * x, -CFG.signal_clip_bp, CFG.signal_clip_bp), trm)
                if v > best_v:
                    best_k, best_v = kap, v
            kap_path[str(m)] = best_k; shift_wf[tem] = np.clip(-best_k * x[tem], -CFG.signal_clip_bp, CFG.signal_clip_bp)
        v, d_ = policy_pnl(shift_wf, tested); pol_daily[name] = d_
        dv = (d_ - base_daily).dropna(); dm = dv.groupby(pd.DatetimeIndex(dv.index).to_period('M')).mean()
        bt = block_bootstrap_t(dv.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed)
        pol_rows.append({'input': name, 'P&L per print (bp)': v, 'baseline S + side concession (bp)': base_pnl, 'delta (bp)': v - base_pnl, 'boot t of delta': bt, 'months positive': int((dm > 0).sum()), 'months': int(len(dm)), 'kappa by month': kap_path,
                         'passes bar': bool((v - base_pnl) >= CFG.interaction_bar_bp and bt > 3 and (dm > 0).sum() >= min(4, len(dm)))})
    sig_policy = pd.DataFrame(pol_rows).set_index('input')
    print(f'The signal as a concession modifier on top of S and its side concession (walk-forward kappa; bar: +{CFG.interaction_bar_bp:.2f} bp per print, bootstrap t > 3, positive in >= 4 months):'); display(sig_policy.round(3))
    # (b) the mark-noise width scalar on the quantile grid: pooled vs split by |eta| tercile, tail calibration on the next month
    TAILS = [t_ for t_ in TAUS if t_ <= 0.10 or t_ >= 0.90]
    w_rows = []
    for m in months_g[1:]:
        tr, te = mo[mo['month'] < m].copy(), mo[mo['month'] == m].copy()
        if len(tr) < 5_000 or te.empty or tr['eta_abs'].notna().mean() < 0.3:
            continue
        cuts = tr['eta_abs'].quantile([1 / 3, 2 / 3]).to_numpy()
        for fr_ in (tr, te):
            fr_['eta_t'] = pd.cut(fr_['eta_abs'].astype(float), bins=[-np.inf, cuts[0], cuts[1], np.inf], labels=['low noise', 'mid', 'high noise']).astype(str)
        g_pool, marg_pool = build_grid(tr, 'o_S'); q_pool = lookup(g_pool, marg_pool, te)
        q_split = np.full_like(q_pool, np.nan)
        for lvl in ['low noise', 'mid', 'high noise']:
            trl, tel = tr[tr['eta_t'] == lvl], te['eta_t'] == lvl
            if len(trl) < 2_000 or not tel.any():
                continue
            g_l, m_l = build_grid(trl, 'o_S'); q_split[tel.to_numpy()] = lookup(g_l, m_l, te[tel])
        o = te['o_S'].to_numpy(float)
        for lvl in ['low noise', 'mid', 'high noise']:
            for sd_ in ['P', 'S']:
                msk = ((te['eta_t'] == lvl) & (te['side'] == sd_)).to_numpy() & np.isfinite(o)
                for j, tau in enumerate(TAUS):
                    if tau not in TAILS or msk.sum() < 200:
                        continue
                    for lab, qq in [('pooled', q_pool), ('split by mark noise', q_split)]:
                        okq = msk & np.isfinite(qq[:, j])
                        if okq.sum() >= 200:
                            w_rows.append({'month': str(m), 'side': sd_, 'noise tercile': lvl, 'tau': tau, 'grid': lab, 'realised': float((o[okq] <= qq[okq, j]).mean()), 'abs error': float(abs((o[okq] <= qq[okq, j]).mean() - tau))})
    width = pd.DataFrame(w_rows)
    if len(width):
        w_tab = width.pivot_table(index=['side', 'noise tercile'], columns='grid', values='abs error', aggfunc='mean')
        w_side = width.groupby(['side', 'grid'])['abs error'].mean().unstack('grid'); w_side['improvement'] = w_side['pooled'] - w_side['split by mark noise']
        g_all, _ = build_grid(mo.assign(eta_t=pd.cut(mo['eta_abs'].astype(float), bins=[-np.inf, *mo['eta_abs'].quantile([1 / 3, 2 / 3]).to_numpy(), np.inf], labels=['low noise', 'mid', 'high noise']).astype(str)), 'o_S')
        iqr_by_noise = mo.assign(eta_t=pd.cut(mo['eta_abs'].astype(float), bins=[-np.inf, *mo['eta_abs'].quantile([1 / 3, 2 / 3]).to_numpy(), np.inf], labels=['low noise', 'mid', 'high noise']).astype(str)).groupby(['side', 'eta_t'], observed=True)['o_S'].quantile([0.25, 0.75]).unstack()
        iqr_by_noise = (iqr_by_noise[0.75] - iqr_by_noise[0.25]).unstack('eta_t')[['low noise', 'mid', 'high noise']]
        width_pass = bool((w_side['improvement'] >= CFG.width_scalar_bar).all())
        print('Mark-noise width scalar: mean absolute tail calibration error (tau 0.05/0.10/0.90/0.95) on the next month, pooled grid vs grid split by |eta| tercile:'); display(w_tab.round(4))
        print(f'By side (bar: improvement >= {CFG.width_scalar_bar:.2f} on both sides) -> {"ADOPT" if width_pass else "keep the pooled grid"}:'); display(w_side.round(4))
        print('Interquartile width of the oriented error of S by side and mark-noise tercile (bp):'); display(iqr_by_noise.round(2))
    else:
        w_tab = w_side = iqr_by_noise = pd.DataFrame(); width_pass = False
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for sd_, c_ in [('P dealer buys (bid)', '#4C72B0'), ('S dealer sells (offer)', '#DD8452')]:
        t_ = sig_q.loc[sd_]
        ax[0].plot(range(len(t_)), t_['hedged move after, all prints (bp)'], marker='o', color=c_, label=f'{sd_}: all prints'); ax[0].plot(range(len(t_)), t_['hedged move after, S fills (bp)'], marker='s', ls='--', color=c_, label=f'{sd_}: S fills')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(range(5)); ax[0].set_xticklabels(['Q1\nmost adverse', 'Q2', 'Q3', 'Q4', 'Q5\nmost favourable'], fontsize=8); ax[0].set_ylabel(f'hedged move after the print, h={HR} (bp)'); ax[0].legend(fontsize=7); ax[0].set_title('Does the oriented forecast order the move after the print?', fontsize=10)
    sp = sig_policy['delta (bp)']; ax[1].barh(range(len(sp)), sp.to_numpy(float), color=['#2E8B57' if p_ else '#8C8C8C' for p_ in sig_policy['passes bar']]); ax[1].set_yticks(range(len(sp))); ax[1].set_yticklabels([f'{i_}  (boot t {b_:+.1f})' for i_, b_ in zip(sp.index, sig_policy['boot t of delta'])], fontsize=8); ax[1].axvline(0, color='k', lw=0.6); ax[1].axvline(CFG.interaction_bar_bp, color='#888888', ls='--', lw=0.8)
    ax[1].set_xlabel('P&L per print over S + side concession (bp)'); ax[1].set_title('The signal as a concession modifier (dashed = bar)', fontsize=10)
    if len(width):
        w_tab.plot.bar(ax=ax[2], color=['#8C8C8C', '#2E8B57']); ax[2].set_title('Tail calibration error by mark-noise tercile: pooled vs split grid', fontsize=10); ax[2].set_xlabel(''); ax[2].tick_params(axis='x', rotation=30, labelsize=7); ax[2].legend(fontsize=8)
    else:
        ax[2].text(0.5, 0.5, 'width-scalar test needs at least two held-out months\nbefore the test month', ha='center', va='center', fontsize=10, transform=ax[2].transAxes); ax[2].set_axis_off()
    plt.tight_layout(); savefig('12b_residual_side')
    record('residual_side', phi_bar=phi_bar, quintiles=flat_records(sig_q.reset_index()), slopes=sig_slope.round(4).to_dict(orient='records'), policy=sig_policy.drop(columns=['kappa by month']).round(4).reset_index().to_dict(orient='records'),
           kappa_paths={k_: v_ for k_, v_ in sig_policy['kappa by month'].items()}, width_scalar={'by_side': flat_records(w_side.reset_index()) if len(width) else [], 'by_tercile': flat_records(w_tab.reset_index()) if len(width) else [], 'iqr_by_noise': flat_records(iqr_by_noise.reset_index()) if len(width) else [], 'adopt': width_pass})
else:
    print('Section 12b skipped: needs the state-space signal on the trade frame.')

# %% [markdown]
# ### 12c. The production objective: pfill(x) x (x + charge), rebuilt, then replaced one piece at a time
#
# **What the desk runs.** For a quote on one side, production chooses the concession $x$ (bp of yield away from the
# algo mid) that maximises
#
# $$\max_x\; \hat p_{\text{fill}}(x)\,\big(x + \text{charge}\big),$$
#
# with $\hat p_{\text{fill}}$ the trailing empirical CDF of the basis between the algo yield and the print on the
# quote's own side (the bid uses dealer-buy prints only) and the charge the sum of the liquidity, risk and manual
# charges booked on the trade. It is a one-dimensional optimizer with three parts: a **mid** (the algo quote), a
# **fill curve** (pooled by side) and a **value of a fill** ($x$ plus the charge, which assumes the charge is earned
# in full and the market does nothing afterwards). Everything this programme has built is a candidate replacement
# for one of those parts: the same-side memory S for the mid, the quantile grid or the conditioned fill model for the
# pooled curve, and the expected round-trip P&L (the edge model with its selection adjustment) for the booked spread.
#
# **The test.** The production rule is replicated on the matched prints (charges joined as of the print, units and
# sign detected and printed), then each part is swapped alone and all together. Every variant chooses $x^*$ per
# print on its own objective (the expected-P&L objective may decline to quote where no concession on the grid has a
# positive expectation, which the production objective cannot, since $x + \text{charge}$ is positive wherever it
# looks) and is judged on the **same realised outcome**: a fill if the print crossed the quote at $x^*$, P&L from the realised round trip (and the evaluation markout beside it), in bp per print and dollars per
# month, against the production rule with the usual bar. The gap between what the production objective expects per
# print and what the round trip pays is the cost of assuming the charge is earned in full.
#
# **By side, and the cover.** The desk reads a quote rule through its win rate on the bid and on the offer and
# through where the quote sat against the print. For each variant, on the same prints: the share of P (bid) and S
# (offer) prints won, the distance $|o - x^*|$ between the quote and the MSRB print (on wins, the bp given up
# through the print; on misses, the bp short of it), the share of quotes within 2 bp of the print, and the realised
# P&L, with the daily differences from production bootstrapped. Then the same win rate and P&L by trade size,
# duration, rating, call structure, beta-space cluster and industry, bid and offer separately, for the two swaps
# that matter (the fill curve alone, and all three parts): a variant that wins where production loses, or loses
# where it wins, is a conditioning gain the pooled table hides.
#
# **Reading it.** If swapping the fill curve moves the dollars and swapping the mid does not, the desk's lever is
# the fill model; if the expected-P&L value moves them, the lever is where the quote earns its edge, not how often it
# fills. A variant that wins on its own objective but loses on realised dollars is a model that flatters itself.

# %%
CELL_T('12c. The production objective: pfill(x) x (x + charge), rebu [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty and 'METHODS' in globals() and common.sum() >= 200:
    XGa = np.arange(*CFG.objective_grid_bp).astype(float); XG = [float(x_) for x_ in XGa]
    # ---- (1) charges per print: the per-trade track store as of the print, else charge columns on the matched trades, else the static track, else zero
    def _charge_cols(frame) -> list:
        """Numeric columns whose name contains 'charge' (v4.6: comment / note / key / flag columns and non-numeric columns are left out)."""
        if frame is None:
            return []
        out = []
        for c in frame.columns:
            cl = str(c).lower()
            if 'charge' in cl and not any(t_ in cl for t_ in ('comment', 'note', 'desc', 'reason', 'key', 'flag', 'type', 'name')) and pd.to_numeric(frame[c], errors='coerce').notna().any():
                out.append(c)
        return out
    _cdir = PIPE.path(PIPE.data_dir, 'track_charges')
    trk_ch = PIPE.store('track_charges').read(categorical=False) if _cdir.exists() and any(_cdir.glob('*.parquet')) else None
    CH = pd.DataFrame(index=mo['_id'].to_numpy()); charge_src = 'none'; ccols = []
    if trk_ch is not None and _charge_cols(trk_ch) and 'cusip' in trk_ch.columns:
        ccols = _charge_cols(trk_ch)
        tcol = next((c for c in ['timestamp', 'ts', 'time', 'tradetime', 'trade_time', 'signal_ts', 'signal_time', 'quote_time', 'date'] if c in trk_ch.columns), None)
        scol = next((c for c in ['side', 'msrb_side', 'signal_side', 'quote_side'] if c in trk_ch.columns), None)
        t_ = trk_ch[['cusip'] + ([tcol] if tcol else []) + ([scol] if scol else []) + ccols].copy(); t_['cusip'] = t_['cusip'].astype('string')
        for c in ccols:
            t_[c] = pd.to_numeric(t_[c], errors='coerce')
        q_ = mo[['_id', 'cusip', 'trade_ts', 'side']].copy(); q_['cusip'] = q_['cusip'].astype('string'); q_['trade_ts'] = to_ns(q_['trade_ts']); q_['side'] = q_['side'].astype(str)
        if tcol:
            t_['_t'] = to_ns(t_[tcol]); t_ = t_.dropna(subset=['_t', 'cusip']).sort_values('_t'); q_ = q_.sort_values('trade_ts'); by = ['cusip']
            if scol:
                t_['side'] = t_[scol].astype('string').str.upper().str[0].astype(str); by = ['cusip', 'side']
            j_ = pd.merge_asof(q_, t_[by + ['_t'] + ccols], left_on='trade_ts', right_on='_t', by=by, direction='backward', tolerance=pd.Timedelta(days=CFG.charge_max_age_days)).set_index('_id')
            j2 = pd.merge_asof(q_[['_id', 'cusip', 'trade_ts']], t_[['cusip', '_t'] + ccols], left_on='trade_ts', right_on='_t', by='cusip', direction='backward').set_index('_id')
            _exact = float(np.isfinite(j_[ccols[0]].reindex(CH.index).to_numpy(float)).mean())
            for c in ccols:
                v1 = j_[c].reindex(CH.index).to_numpy(float); v2 = j2[c].reindex(CH.index).to_numpy(float); CH[c] = np.where(np.isfinite(v1), v1, v2)
            charge_src = f'track_charges store, as of the print time by {" x ".join(by)} within {CFG.charge_max_age_days:g} days ({_exact:.0%} of prints; the rest take the bond\'s last earlier charge)'
        else:
            last = t_.groupby('cusip', observed=True)[ccols].last()
            for c in ccols:
                CH[c] = last[c].reindex(mo['cusip'].astype('string')).to_numpy(float)
            charge_src = 'track_charges store, last charge per bond (the track carries no time column)'
    elif _charge_cols(mo):
        ccols = _charge_cols(mo)
        for c in ccols:
            CH[c] = pd.to_numeric(mo[c], errors='coerce').to_numpy(float)
        charge_src = 'charge columns on the matched trades (algosignal_msrb store)'
    elif 'track_static' in globals() and track_static is not None and _charge_cols(track_static):
        ccols = _charge_cols(track_static); last = track_static.assign(cusip=track_static['cusip'].astype('string')).drop_duplicates('cusip', keep='last').set_index('cusip')
        for c in ccols:
            CH[c] = pd.to_numeric(last[c], errors='coerce').reindex(mo['cusip'].astype('string')).to_numpy(float)
        charge_src = 'track_static store, one static value per bond'
    _pxa = pd.to_numeric(mo['msrb_price'], errors='coerce').clip(1, 300).fillna(100.0).to_numpy(float)
    if ccols:
        units = CFG.charge_units
        tot = CH[ccols].sum(axis=1, min_count=1).to_numpy(float)
        if units == 'auto':
            units = 'bp' if any('bp' in c.lower() for c in ccols) else ('pct' if np.nanmedian(np.abs(tot)) < 0.5 else 'bp')
        if units == 'pct':
            CH[ccols] = CH[ccols] * 100.0
        elif units == 'price':
            _conv = 1e4 / (mo['modified_duration_lag1'].clip(lower=0.25).fillna(5.0).to_numpy(float) * _pxa)
            CH[ccols] = CH[ccols].to_numpy(float) * _conv[:, None]
        tot = CH[ccols].sum(axis=1, min_count=1).to_numpy(float); flipped = False
        if np.nanmedian(tot) < 0:
            CH[ccols] = -CH[ccols]; tot = -tot; flipped = True
        print(f'Charges: {len(ccols)} column(s) {ccols} from {charge_src}; units read as {units!r}{" (sign flipped: the store books charges as negatives)" if flipped else ""}; '
              f'{float(np.isfinite(tot).mean()):.0%} of customer prints carry a charge; median total {np.nanmedian(tot):.2f} bp (P {np.nanmedian(tot[IS_P]):.2f}, S {np.nanmedian(tot[IS_S]):.2f}).')
        _ct = pd.DataFrame({'side': SIDE_ARR, 'qty_group': QTY_ARR, **{c: CH[c].to_numpy(float) for c in ccols}, 'total': tot})
        charge_tab = _ct.groupby(['qty_group', 'side'], observed=True)[ccols + ['total']].mean().unstack('side').reindex(QTY_LABELS)
        print('Mean charge by quantity bin and side (bp of yield):'); display(charge_tab.round(2))
    else:
        tot = np.full(len(mo), np.nan); units = 'none'; charge_tab = pd.DataFrame()
        print('Charges: no column containing "charge" in the track_charges, algosignal_msrb or track_static stores. The objective runs with a zero charge (max pfill(x) x x). '
              'Regenerate the pipeline with the track_charges source (see the final note of the run) to populate it.')
    CHG = tot.copy()
    for sd_, msk in [('P', IS_P), ('S', IS_S)]:
        _m = float(np.nanmedian(CHG[msk])) if np.isfinite(CHG[msk]).any() else 0.0; CHG[msk & ~np.isfinite(CHG)] = _m

    # ---- (2) fill curves on the objective grid: desk curves (A from the full history, S from the held-out months), the quantile grid (S), the GBM fill models (A and S)
    def interp_rows(P: np.ndarray, knots: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Row-wise linear interpolation of a fill curve known at the knots onto the grid, flat beyond the ends, forced non-increasing."""
        P = np.minimum.accumulate(P, axis=1); out = np.full((P.shape[0], len(grid)), np.nan)
        for g_i, x in enumerate(grid):
            if x <= knots[0]:
                out[:, g_i] = P[:, 0]
            elif x >= knots[-1]:
                out[:, g_i] = P[:, -1]
            else:
                j = int(np.searchsorted(knots, x, 'right')) - 1; w = (x - knots[j]) / (knots[j + 1] - knots[j]); out[:, g_i] = (1 - w) * P[:, j] + w * P[:, j + 1]
        return out

    _h = trades_all[['side', 'trade_date', 'e_algo_bp']].dropna(); _h = _h[_h['side'].isin(['P', 'S'])]
    desk_A_x = rolling_ecdf_pfill(to_ns(_h['trade_date']).to_numpy(), np.where(_h['side'].to_numpy() == 'P', 1.0, -1.0) * _h['e_algo_bp'].to_numpy(float), _h['side'].astype(str).to_numpy(), TD, SIDE_ARR, XG, CFG.pfill_window_days)
    desk_S_x = rolling_ecdf_pfill(TD, O_S, SIDE_ARR, TD, SIDE_ARR, XG, CFG.pfill_window_days)
    gbm_S_x = interp_rows(_mat('pf_ipca'), PFD, XGa); gbm_A_x = interp_rows(_mat('pf_ipcaA'), PFD, XGa)
    grid_S_x = np.full((len(mo), len(XG)), np.nan); TAUa = np.asarray(TAUS, float); _nt = len(TAUa)
    for m in MONTHS_MO[1:]:
        trm = MONTH_ARR < m; tem = MONTH_ARR == m
        if trm.sum() < CFG.pfill_min_train or not tem.any():
            continue
        g_, marg_ = build_grid(mo[trm], 'o_S'); q = np.maximum.accumulate(lookup(g_, marg_, mo[tem]), axis=1); rows_ = np.arange(len(q))
        for g_i, x in enumerate(XGa):
            k = (q <= x).sum(axis=1); kl = np.clip(k - 1, 0, _nt - 1); kh = np.clip(k, 0, _nt - 1)
            lo = q[rows_, kl]; hi = q[rows_, kh]; frac = np.where(hi > lo, (x - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
            F = np.where(k == 0, TAUa[0] * 0.5, np.where(k == _nt, 1.0 - (1.0 - TAUa[-1]) * 0.5, TAUa[kl] + (TAUa[kh] - TAUa[kl]) * frac))
            grid_S_x[tem, g_i] = 1.0 - F
    e_S = pred['edge_ipca_rt'].to_numpy(float) if 'edge_ipca_rt' in pred.columns else np.full(len(mo), np.nan)
    e_A = pred['edge_ipcaA_rt'].to_numpy(float) if 'edge_ipcaA_rt' in pred.columns else np.full(len(mo), np.nan)

    def adj_curve(o_mid: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Selection adjustment on the objective grid, by side on prior months: mean base edge of the fills kept at x minus at 0."""
        ADJx = np.zeros((len(mo), len(XGa))); be = base - o_mid
        for m in MONTHS_MO[1:]:
            trm = MONTH_ARR < m; tem = MONTH_ARR == m
            for sd_, msk in [('P', IS_P), ('S', IS_S)]:
                sel0 = trm & msk & np.isfinite(be)
                if (sel0 & (o_mid >= 0)).sum() < max(200, CFG.pfill_min_train // 10):
                    continue
                m0 = float(be[sel0 & (o_mid >= 0)].mean())
                for g_i, x in enumerate(XGa):
                    selj = sel0 & (o_mid >= x); ADJx[tem & msk, g_i] = (float(be[selj].mean()) - m0) if selj.sum() >= max(100, CFG.pfill_min_train // 20) else 0.0
        return ADJx

    ADJ_S = adj_curve(O_S, BASE_RT); ADJ_A = adj_curve(O_A, BASE_RT)
    V_charge = XGa[None, :] + CHG[:, None]; V_exp_S = e_S[:, None] + XGa[None, :] + ADJ_S; V_exp_A = e_A[:, None] + XGa[None, :] + ADJ_A

    def pnl_policy(err: np.ndarray, xs: np.ndarray | float, mask: np.ndarray, base_arr: np.ndarray) -> dict:
        """pnl_with for a policy that may decline to quote (x* = +inf: no fill, zero P&L on that print)."""
        o = S_ARR * err; fill = o >= xs; xe = np.where(np.isfinite(xs), xs, 0.0); pnl = base_arr - o + xe; ok = mask & np.isfinite(pnl) & np.isfinite(o); f_ok = fill & ok
        d_ = pd.Series(np.where(f_ok, pnl, 0.0)[ok]).groupby(mo['trade_date'].to_numpy()[ok]).mean()
        return {'fill share': f_ok.sum() / max(ok.sum(), 1), 'P&L per print (bp)': float(d_.mean()) if len(d_) else np.nan, '$ per month ($k)': float((pnl[f_ok] * DPB[f_ok]).sum() / 1e3 / max(int(pd.Series(MONTH_ARR[ok]).nunique()), 1)), '_daily': d_}

    def choose(pf: np.ndarray, value: np.ndarray, may_decline: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """x* per print and the objective at x*. An expected-P&L objective that is negative at every x declines to quote (x* = +inf: no fill, zero P&L);
        the production objective has no such option (x + charge is positive wherever it looks)."""
        val = np.where(np.isfinite(pf) & np.isfinite(value), pf * value, -np.inf); j = np.argmax(val, axis=1); has = np.isfinite(val).any(axis=1)
        best = np.where(has, val[np.arange(len(val)), j], np.nan); xs = np.where(has, XGa[j], np.nan)
        if may_decline:
            dec = has & (best < 0); xs = np.where(dec, np.inf, xs); best = np.where(dec, 0.0, best)
        return xs, best

    POL = {'production: A, desk curve, x + charge': (errA, O_A, desk_A_x, V_charge, False),
           'A, desk curve, expected P&L (edge model on A)': (errA, O_A, desk_A_x, V_exp_A, True),
           'A, GBM + IPCA fill model, x + charge': (errA, O_A, gbm_A_x, V_charge, False),
           'A, GBM + IPCA fill model, expected P&L': (errA, O_A, gbm_A_x, V_exp_A, True),
           'S, desk curve, x + charge': (errS, O_S, desk_S_x, V_charge, False),
           'S, quantile grid, x + charge': (errS, O_S, grid_S_x, V_charge, False),
           'S, GBM + IPCA fill model, x + charge': (errS, O_S, gbm_S_x, V_charge, False),
           'S, GBM + IPCA fill model, expected P&L (the Section 11b engine)': (errS, O_S, gbm_S_x, V_exp_S, True)}
    okm = tested & common & np.isfinite(e_S) & np.isfinite(e_A) & np.isfinite(CHG) & np.isfinite(BASE_RT)
    for _, _, pf_, _, _ in POL.values():
        okm &= np.isfinite(pf_).all(axis=1)
    if okm.sum() < 200:
        print(f'Section 12c: only {int(okm.sum())} prints carry every input (charges, the fill curves, the edge models, a round trip); the objective test is skipped on this run.')
        record('production_objective', charge_source=charge_src, charge_units=units, prints=int(okm.sum()), policies=[])
    else:
        obj_rows, obj_daily, XSTAR = [], {}, {}
        for name, (err_, o_, pf_, v_, dec_) in POL.items():
            xs, val = choose(pf_, v_, dec_); XSTAR[name] = xs
            r_rt = pnl_policy(err_, xs, okm, BASE_RT); r_ev = pnl_policy(err_, xs, okm, BASE_EVAL); obj_daily[name] = (r_rt.pop('_daily'), r_ev.pop('_daily'))
            fill_ = okm & (o_ >= xs); _fin = okm & np.isfinite(xs)
            obj_rows.append({'policy': name, 'mean x* (bp)': float(np.mean(xs[_fin])) if _fin.any() else np.nan, 'share declined': float(np.isinf(xs[okm]).mean()), 'fill share': r_rt['fill share'], 'objective value per print (expected)': float(np.nanmean(val[okm])),
                             'booked spread | fill (x + charge, bp)': float(np.mean((xs + CHG)[fill_])) if fill_.any() else np.nan, 'round trip: P&L per print (bp)': r_rt['P&L per print (bp)'], 'round trip: $ per month ($k)': r_rt['$ per month ($k)'],
                             'evaluation: P&L per print (bp)': r_ev['P&L per print (bp)'], 'evaluation: $ per month ($k)': r_ev['$ per month ($k)']})
        for name, err_ in [('A at 0 (no concession)', errA), ('S at 0 (no concession)', errS)]:
            r_rt = pnl_policy(err_, 0.0, okm, BASE_RT); r_ev = pnl_policy(err_, 0.0, okm, BASE_EVAL); obj_daily[name] = (r_rt.pop('_daily'), r_ev.pop('_daily')); XSTAR[name] = np.zeros(len(mo))
            obj_rows.append({'policy': name, 'mean x* (bp)': 0.0, 'share declined': 0.0, 'fill share': r_rt['fill share'], 'objective value per print (expected)': np.nan, 'booked spread | fill (x + charge, bp)': np.nan, 'round trip: P&L per print (bp)': r_rt['P&L per print (bp)'], 'round trip: $ per month ($k)': r_rt['$ per month ($k)'],
                             'evaluation: P&L per print (bp)': r_ev['P&L per print (bp)'], 'evaluation: $ per month ($k)': r_ev['$ per month ($k)']})
        objective = pd.DataFrame(obj_rows).set_index('policy'); PRODN = 'production: A, desk curve, x + charge'; _p_rt, _p_ev = obj_daily[PRODN]
        for name in objective.index:
            d_rt = (obj_daily[name][0] - _p_rt).dropna(); d_ev = (obj_daily[name][1] - _p_ev).dropna(); dm = d_rt.groupby(pd.DatetimeIndex(d_rt.index).to_period('M')).mean()
            objective.loc[name, 'round trip: boot t vs production'] = block_bootstrap_t(d_rt.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if name != PRODN else np.nan
            objective.loc[name, 'evaluation: boot t vs production'] = block_bootstrap_t(d_ev.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if name != PRODN else np.nan
            objective.loc[name, 'months above production (round trip)'] = int((dm > 0).sum()) if name != PRODN else np.nan
        _nm = int(pd.Series(MONTH_ARR[okm]).nunique())
        objective['beats production (bar, round trip)'] = ((objective['round trip: P&L per print (bp)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)']) >= CFG.interaction_bar_bp) & (objective['round trip: boot t vs production'] > 3) & (objective['months above production (round trip)'] >= min(4, _nm))
        print(f'The production objective and its variants on {int(okm.sum()):,} held-out customer prints with every input ({_nm} months). Each policy picks x* per print on its own objective over x in [{XGa[0]:g}, {XGa[-1]:g}] bp (the expected-P&L objective may decline to quote); all are judged on the realised round trip. Bar vs production: +{CFG.interaction_bar_bp:.2f} bp per print, bootstrap t > 3, positive in >= 4 months:'); display(objective.round(3))
        _gap = objective.loc[PRODN, 'objective value per print (expected)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)']
        print(f'Reality gap of the production objective: it expects {objective.loc[PRODN, "objective value per print (expected)"]:+.2f} bp per print (charge earned in full, no move after the fill); the round trip pays {objective.loc[PRODN, "round trip: P&L per print (bp)"]:+.2f} bp per print, a gap of {_gap:+.2f} bp.')
        comp = pd.DataFrame([{'swap': 'mid: algo quote -> same-side memory S (desk curve, x + charge)', 'delta RT P&L per print (bp)': objective.loc['S, desk curve, x + charge', 'round trip: P&L per print (bp)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)'], 'boot t': objective.loc['S, desk curve, x + charge', 'round trip: boot t vs production']},
                             {'swap': 'fill curve: desk -> GBM + IPCA (algo quote, x + charge)', 'delta RT P&L per print (bp)': objective.loc['A, GBM + IPCA fill model, x + charge', 'round trip: P&L per print (bp)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)'], 'boot t': objective.loc['A, GBM + IPCA fill model, x + charge', 'round trip: boot t vs production']},
                             {'swap': 'value of a fill: x + charge -> expected round-trip P&L (algo quote, desk curve)', 'delta RT P&L per print (bp)': objective.loc['A, desk curve, expected P&L (edge model on A)', 'round trip: P&L per print (bp)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)'], 'boot t': objective.loc['A, desk curve, expected P&L (edge model on A)', 'round trip: boot t vs production']},
                             {'swap': 'all three: S, GBM + IPCA, expected P&L', 'delta RT P&L per print (bp)': objective.loc['S, GBM + IPCA fill model, expected P&L (the Section 11b engine)', 'round trip: P&L per print (bp)'] - objective.loc[PRODN, 'round trip: P&L per print (bp)'], 'boot t': objective.loc['S, GBM + IPCA fill model, expected P&L (the Section 11b engine)', 'round trip: boot t vs production']}]).set_index('swap')
        print('What each replacement is worth on its own (round-trip P&L per print vs the production rule):'); display(comp.round(3))
        xmix = pd.DataFrame({name: pd.Series(XSTAR[name][okm]).value_counts(normalize=True).sort_index() for name in POL}).fillna(0.0); xmix.index = ['no quote' if np.isinf(i_) else f'x = {float(i_):g}' for i_ in xmix.index]
        print('Concession chosen, share of prints by policy:'); display(xmix.round(3))

        # ---- (3) by side: win rate, distance to the print (the cover), P&L; daily differences from production bootstrapped
        ALLPOL = {**{k_: v_[:2] + (XSTAR[k_],) for k_, v_ in POL.items()}, 'A at 0 (no concession)': (errA, O_A, np.zeros(len(mo))), 'S at 0 (no concession)': (errS, O_S, np.zeros(len(mo)))}
        side_rows, side_daily = [], {}
        for name, (err_, o_, xs) in ALLPOL.items():
            for sd_, msk in [('P', IS_P), ('S', IS_S)]:
                ok = okm & msk; fin = ok & np.isfinite(xs); win = fin & (o_ >= xs); miss = fin & ~(o_ >= xs); gap = o_ - xs
                pnl = BASE_RT - o_ + np.where(np.isfinite(xs), xs, 0.0); v = np.where(win, pnl, 0.0)
                d_win = pd.Series(win[ok].astype(float)).groupby(TD[ok]).mean(); d_pnl = pd.Series(v[ok]).groupby(TD[ok]).mean(); side_daily[(name, sd_)] = (d_win, d_pnl)
                side_rows.append({'policy': name, 'side': sd_, 'prints': int(ok.sum()), 'win rate': float(win.sum() / max(ok.sum(), 1)), 'declined': float((ok & ~np.isfinite(xs)).sum() / max(ok.sum(), 1)), 'mean x* (bp)': float(np.mean(xs[fin])) if fin.any() else np.nan,
                                  '|quote - print| (bp)': float(np.mean(np.abs(gap[fin]))) if fin.any() else np.nan, 'within 2 bp of the print': float((np.abs(gap[fin]) <= 2.0).mean()) if fin.any() else np.nan,
                                  'through the print | win (bp)': float(gap[win].mean()) if win.any() else np.nan, 'short of the print | miss (bp)': float(-gap[miss].mean()) if miss.any() else np.nan,
                                  'RT P&L | win (bp)': float(pnl[win].mean()) if win.any() else np.nan, 'RT P&L per print (bp)': float(d_pnl.mean()) if len(d_pnl) else np.nan, '$ per month ($k)': float((pnl[win] * DPB[win]).sum() / 1e3 / _nm)})
        by_side = pd.DataFrame(side_rows).set_index(['policy', 'side'])
        for (name, sd_) in by_side.index:
            if name == PRODN:
                continue
            dw = (side_daily[(name, sd_)][0] - side_daily[(PRODN, sd_)][0]).dropna(); dp = (side_daily[(name, sd_)][1] - side_daily[(PRODN, sd_)][1]).dropna()
            by_side.loc[(name, sd_), 'win rate boot t vs production'] = block_bootstrap_t(dw.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed)
            by_side.loc[(name, sd_), 'RT P&L boot t vs production'] = block_bootstrap_t(dp.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed)
        print('By side: win rate, distance of the quote at x* from the MSRB print (the cover), and realised P&L, on the same prints (P = bid, S = offer; bootstrap t of the daily difference from production):'); display(by_side.round(3))
        _bs = by_side.unstack('side')
        print('Win rate and P&L per print, bid vs offer, production and the two swaps that matter:')
        display(_bs.loc[[p_ for p_ in [PRODN, 'A, GBM + IPCA fill model, x + charge', 'S, GBM + IPCA fill model, expected P&L (the Section 11b engine)', 'A at 0 (no concession)'] if p_ in _bs.index], [('win rate', 'P'), ('win rate', 'S'), ('|quote - print| (bp)', 'P'), ('|quote - print| (bp)', 'S'), ('RT P&L per print (bp)', 'P'), ('RT P&L per print (bp)', 'S')]].round(3))

        # ---- (4) by characteristic x side for production and the two key swaps
        KEYP = [p_ for p_ in [PRODN, 'A, GBM + IPCA fill model, x + charge', 'S, GBM + IPCA fill model, expected P&L (the Section 11b engine)'] if p_ in POL]
        _dimv = {'trade size': QTY_ARR, 'duration bucket': pd.cut(mo['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']).astype(str).to_numpy(),
                 'rating bucket': pd.cut(mo['rating_score'].fillna(-1), bins=[-2, -0.5, 14.5, 17.5, 20.5, 21.5], labels=['NR', 'BBB and below', 'A', 'AA', 'AAA']).astype(str).to_numpy(),
                 'beta-space cluster': np.array([CLUSTER_LABEL.get(int(c_), 'NA') if str(c_).lstrip('-').isdigit() else 'NA' for c_ in CL_ARR])}
        for dname, col in [('call structure', 'call_structure'), ('industry', 'industry_bucket')]:
            if col in mo.columns and mo[col].notna().any():
                _dimv[dname] = mo[col].astype(str).to_numpy()
        _dimo = {'trade size': QTY_LABELS, 'duration bucket': ['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11'], 'rating bucket': ['NR', 'BBB and below', 'A', 'AA', 'AAA']}
        ch_rows = []
        for dname, arr in _dimv.items():
            for lvl in (_dimo.get(dname) or sorted(pd.unique(arr[okm]))):
                for sd_, msk in [('P', IS_P), ('S', IS_S)]:
                    sel = okm & msk & (arr == lvl)
                    if sel.sum() < max(100, CFG.n_min_cell // 10):
                        continue
                    for name in KEYP:
                        err_, o_, _, _, _ = POL[name]; xs = XSTAR[name]; fin = sel & np.isfinite(xs); win = fin & (o_ >= xs); pnl = BASE_RT - o_ + np.where(np.isfinite(xs), xs, 0.0)
                        ch_rows.append({'dimension': dname, 'cell': str(lvl), 'side': sd_, 'policy': name, 'prints': int(sel.sum()), 'win rate': float(win.sum() / sel.sum()), 'RT P&L per print (bp)': float(np.where(win, pnl, 0.0)[sel].mean()), '$ per month ($k)': float((pnl[win] * DPB[win]).sum() / 1e3 / _nm)})
        by_char = pd.DataFrame(ch_rows)
        _short = {PRODN: 'production', 'A, GBM + IPCA fill model, x + charge': 'fill-curve swap', 'S, GBM + IPCA fill model, expected P&L (the Section 11b engine)': 'all three'}
        char_tabs = {}
        if len(by_char):
            by_char['policy_s'] = by_char['policy'].map(_short)
            for dname, g in by_char.groupby('dimension', sort=False):
                pv = g.pivot_table(index=['cell', 'side'], columns='policy_s', values=['win rate', 'RT P&L per print (bp)'], aggfunc='first')
                order = [c_ for c_ in (_dimo.get(dname) or sorted(g['cell'].unique())) if c_ in pv.index.get_level_values(0)]
                pv = pv.reindex([(c_, sd_) for c_ in order for sd_ in ['P', 'S'] if (c_, sd_) in pv.index])
                for sw in ['fill-curve swap', 'all three']:
                    if ('RT P&L per print (bp)', sw) in pv.columns:
                        pv[('delta P&L vs production (bp)', sw)] = pv[('RT P&L per print (bp)', sw)] - pv[('RT P&L per print (bp)', 'production')]; pv[('delta win rate vs production', sw)] = pv[('win rate', sw)] - pv[('win rate', 'production')]
                pv['prints'] = g.pivot_table(index=['cell', 'side'], values='prints', aggfunc='first').reindex(pv.index)['prints']
                char_tabs[dname] = pv
                print(f'By {dname} x side: win rate and round-trip P&L per print (bp) for production and the two swaps, with the deltas vs production:'); display(pv.round(3))

        # ---- figures: by side, then by characteristic
        fig, ax = plt.subplots(1, 3, figsize=(20, 7))
        _ord = [p_ for p_ in list(POL) + ['A at 0 (no concession)', 'S at 0 (no concession)'] if p_ in _bs.index]; y_ = np.arange(len(_ord)); w = 0.38
        for a, col, ttl in zip(ax, ['win rate', '|quote - print| (bp)', 'RT P&L per print (bp)'], ['Win rate (share of prints crossed at x*)', 'Distance of the quote from the MSRB print (bp, the cover)', 'Realised round-trip P&L per print (bp)']):
            for i_, (sd_, c_) in enumerate([('P', '#4C72B0'), ('S', '#DD8452')]):
                a.barh(y_ + (i_ - 0.5) * w, _bs.loc[_ord, (col, sd_)], w, color=c_, label={'P': 'P bid', 'S': 'S offer'}[sd_])
            a.set_yticks(y_); a.set_yticklabels([p_[:48] for p_ in _ord], fontsize=7); a.set_title(ttl, fontsize=10); a.axvline(0, color='k', lw=0.5); a.invert_yaxis()
            for sd_, c_ in [('P', '#4C72B0'), ('S', '#DD8452')]:
                a.axvline(_bs.loc[PRODN, (col, sd_)], color=c_, ls=':', lw=1.0)
        ax[0].legend(fontsize=8); fig.suptitle('The objective variants by side on the same prints (dotted = production)', fontsize=12); plt.tight_layout(); savefig('12c_objective_by_side')
        if char_tabs:
            nd = len(char_tabs); fig, axs = plt.subplots(2, nd, figsize=(4.4 * nd, 11), squeeze=False)
            for j_, (dname, pv) in enumerate(char_tabs.items()):
                for i_, (blk, ttl, cmap, fmt) in enumerate([('delta P&L vs production (bp)', 'delta RT P&L per print vs production (bp)', 'RdYlGn', '{:+.1f}'), ('delta win rate vs production', 'delta win rate vs production', 'RdBu', '{:+.2f}')]):
                    cols = [c_ for c_ in pv.columns if c_[0] == blk]
                    if not cols:
                        axs[i_, j_].set_axis_off(); continue
                    tab = pv[cols].unstack('side'); tab.columns = [f'{c_[1]} | {c_[2]}' for c_ in tab.columns]; tab = tab.reindex([c_ for c_ in (_dimo.get(dname) or sorted(tab.index)) if c_ in tab.index])
                    v_ = tab.to_numpy(float); lim = np.nanmax(np.abs(v_)) if np.isfinite(v_).any() else 1.0; lim = lim if lim > 0 else 1.0
                    a = axs[i_, j_]; im = a.imshow(v_, cmap=cmap, aspect='auto', vmin=-lim, vmax=lim); a.set_xticks(range(tab.shape[1])); a.set_xticklabels(tab.columns, rotation=40, fontsize=7, ha='right'); a.set_yticks(range(len(tab))); a.set_yticklabels([str(i__)[:22] for i__ in tab.index], fontsize=7); a.grid(False)
                    for r_ in range(tab.shape[0]):
                        for c_ in range(tab.shape[1]):
                            if np.isfinite(v_[r_, c_]):
                                a.text(c_, r_, fmt.format(v_[r_, c_]), ha='center', va='center', fontsize=7)
                    a.set_title(f'{dname}: {ttl}', fontsize=9)
            fig.suptitle('Where the swaps win and lose, bid (P) and offer (S) separately: the fill-curve swap and all three parts, vs production', fontsize=11); plt.tight_layout(); savefig('12c_objective_by_characteristic')
        record('production_objective', by_side=by_side.round(4).reset_index().to_dict(orient='records'), by_characteristic=by_char.round(4).to_dict(orient='records'))

        # ---- figures
        fig, ax = plt.subplots(2, 2, figsize=(18, 11))
        if len(charge_tab):
            _ctp = charge_tab['total'] if 'total' in charge_tab.columns.get_level_values(0) else charge_tab
            x_ = np.arange(len(_ctp)); w = 0.38
            for i_, (sd_, c_) in enumerate([('P', '#4C72B0'), ('S', '#DD8452')]):
                if sd_ in _ctp.columns:
                    ax[0, 0].bar(x_ + (i_ - 0.5) * w, _ctp[sd_], w, color=c_, label={'P': 'P bid', 'S': 'S offer'}[sd_])
            ax[0, 0].set_xticks(x_); ax[0, 0].set_xticklabels(_ctp.index, rotation=30, fontsize=8); ax[0, 0].set_title(f'Total charge by quantity bin and side (bp; {", ".join(ccols)})', fontsize=10); ax[0, 0].legend(fontsize=8); ax[0, 0].set_ylabel('bp of yield')
        else:
            ax[0, 0].text(0.5, 0.5, 'no charge columns in the stores', ha='center', va='center', transform=ax[0, 0].transAxes); ax[0, 0].set_axis_off()
        for sd_, msk, c_ in [('P bid', IS_P & okm, '#4C72B0'), ('S offer', IS_S & okm, '#DD8452')]:
            if msk.any():
                pcurve = np.nanmean(desk_A_x[msk], axis=0); chm = float(np.nanmedian(CHG[msk])); val = pcurve * (XGa + chm); j = int(np.nanargmax(val))
                ax[0, 1].plot(XGa, val, marker='.', color=c_, label=f'{sd_}: desk curve x (x + median charge {chm:.1f} bp)'); ax[0, 1].plot(XGa[j], val[j], 'o', ms=9, mfc='none', mec=c_, mew=1.5)
                pg = np.nanmean(gbm_A_x[msk], axis=0); ax[0, 1].plot(XGa, pg * (XGa + chm), ls='--', color=c_, alpha=0.7, label=f'{sd_}: GBM fill model, same value')
        ax[0, 1].axvline(0, color='k', lw=0.6); ax[0, 1].set_xlabel('concession x on the algo quote (bp)'); ax[0, 1].set_ylabel('expected objective per print (bp)'); ax[0, 1].set_title('The production objective as the desk sees it: pfill(x) x (x + charge), side means (circle = its argmax)', fontsize=10); ax[0, 1].legend(fontsize=7)
        _ob = objective.sort_values('round trip: P&L per print (bp)'); y_ = np.arange(len(_ob))
        ax[1, 0].barh(y_, _ob['round trip: P&L per print (bp)'], color=['#2E8B57' if b_ else ('#4C72B0' if i_ == PRODN else '#8C8C8C') for i_, b_ in zip(_ob.index, _ob['beats production (bar, round trip)'])])
        ax[1, 0].set_yticks(y_); ax[1, 0].set_yticklabels([f'{i_}  (t {t_:+.1f})' if np.isfinite(t_) else i_ for i_, t_ in zip(_ob.index, _ob['round trip: boot t vs production'])], fontsize=7)
        ax[1, 0].axvline(objective.loc[PRODN, 'round trip: P&L per print (bp)'], color='#4C72B0', ls='--', lw=0.9); ax[1, 0].set_xlabel('realised round-trip P&L per print (bp)'); ax[1, 0].set_title('Every variant on the same realised outcome (blue = production; green = clears the bar)', fontsize=10)
        xmix.T.plot.bar(ax=ax[1, 1], stacked=True, colormap='viridis', width=0.8); ax[1, 1].set_title('Concession chosen by each policy (share of prints)', fontsize=10); ax[1, 1].set_xlabel(''); ax[1, 1].tick_params(axis='x', rotation=20, labelsize=6); ax[1, 1].legend(fontsize=6, ncol=2)
        plt.tight_layout(); savefig('12c_production_objective')
        record('production_objective', charge_source=charge_src, charge_units=units, charge_columns=ccols, prints=int(okm.sum()), months=_nm, grid=XG, charge_by_cell=flat_records(charge_tab.reset_index()) if len(charge_tab) else [],
               policies=objective.round(4).reset_index().to_dict(orient='records'), swaps=comp.round(4).reset_index().to_dict(orient='records'), concession_mix=flat_records(xmix.reset_index()), reality_gap_bp=float(_gap))
else:
    print('Section 12c skipped: needs Sections 11b and 12.')

# %% [markdown]
# ### 12d. The production ledger: the RFQs we received, production's own quote, its pfill and its key
#
# Section 12c had to replicate production from the match table, and the v45 run showed the replica is not production:
# its concession sat on the grid floor and its charges were the bond's last request, not the request's. The RFQ log
# (`muni_algo_trade_track`) removes the approximation on the bid side. For every request we received it carries the
# **optimal yield** the optimizer quoted with the production fill curve, the **pfill key** it pooled on (coupon bin x
# call group x rating group x quantity group) and the **pfill** it assigned, the charges, and the MSRB print if the
# request traded anywhere (no print = nobody traded it). Production's concession is $x_{\text{prod}} = s \cdot 100\,
# (y_{\text{opt}} - y_{\text{algo}})$.
#
# Four readings, in order. (1) **What production does**: the concession distribution by size and key, and how often a
# request prints. (2) **Is the pooled curve calibrated where it is used**: the realised crossing share at production's
# own quote against the logged pfill, by key and by key component. (3) **The grouping test** the desk asked for, at
# production's quote on the requests it received: the production-style key rebuilt from our data, the beta-space
# cluster in its place, the cluster shifted by the bond's residual deviance against its peers, the desk curve and the
# gradient-boosted model, each giving $\Pr(o_A \ge x_{\text{prod}})$ per request and scored by Brier against the logged
# pfill with the pre-registered bar. (4) **Production's quote against ours on the same requests**: the algo mid at
# zero, S, SQ, S with production's own concession, and the engine, on win rate, cover and realised round-trip P&L.
# The section closes with the **production proxy for the universe**: the algo mid plus the median production
# concession of the bond's key, validated on the overlap against the logged quote, then run through Section 12c's
# machinery on every customer print so the universe comparison no longer depends on the replica.
#
# A request is scored only when it printed and joined to a matched print (trade id, else cusip x side x time within
# `rfq_join_minutes` and the same quantity); requests that did not print are reported, never scored. The win is the
# crossing proxy of Section 11 unless the track carries a won flag, in which case the flag is the win and the proxy's
# agreement with it is reported. Column roles are detected from the track's names and can be overridden in
# `TRACK_RFQ_COLUMNS`.

# %%
CELL_T('12d. The production ledger: the RFQs we received [1]')
TRACK_RFQ_COLUMNS: dict[str, str] = {}   # optional overrides by role, e.g. {'time': 'rfq_time', 'optimal_yield': 'opt_yld', 'won': 'is_filled'}
HAS_RFQ = False
if HAS_TRADES and 'mo' in globals() and not mo.empty and 'okm' in globals() and okm.sum() >= 200:
    _rdir = PIPE.path(PIPE.data_dir, 'track_rfq')
    rfq_raw = PIPE.store('track_rfq').read(categorical=False) if _rdir.exists() and any(_rdir.glob('*.parquet')) else None
    if rfq_raw is None or rfq_raw.empty:
        print('Section 12d: no track_rfq store under the pipeline root. Pull it (mdp.pull(mdp.Config(root=..., sources=("track_rfq",)))) and rerun; the production ledger is skipped on this run.')
    else:
        CAND = {'time': ['rfq_received_time', 'rfq_time', 'rfq_ts', 'received_time', 'timestamp', 'ts', 'time', 'quote_time', 'signal_ts', 'signal_time', 'date'],
                'print_time': ['msrb_tradetime', 'msrb_event_time', 'print_time', 'tradetime', 'trade_time'],
                'req_id': ['strategy_ecn_req_id', 'ecn_req_id', 'rfq_id', 'req_id', 'request_id'],
                'is_final': ['is_final'], 'quoted': ['quoted', 'is_quoted'], 'pricing_error': ['strategy_is_pricing_error', 'is_pricing_error', 'pricing_error'], 'manual': ['manual_takeover', 'is_manual_mode', 'manual'],
                'final_quote_yield': ['final_quote_yield', 'quoted_yield', 'quote_yield_final'],
                'quantity': ['quantity', 'msrb_quantity', 'qty', 'size', 'par', 'par_amount', 'notional'],
                'algo_yield': ['algo_yield', 'algo_signal_yield', 'signal_yield', 'mid_yield', 'model_yield', 'algo_yld', 'yield_mid', 'fair_yield', 'mid'],
                'optimal_yield': ['optimal_yield', 'opt_yield', 'optimal_yld', 'opt_yld', 'quote_yield', 'our_yield', 'bid_yield', 'optimal'],
                'pfill': ['pfill', 'probability', 'p_fill', 'prob_fill', 'fill_prob', 'pfill_value', 'pfill_prob'],
                'pfill_key': ['pfill_key', 'pf_key', 'pfillkey', 'pfill_group', 'pfill_bin'],
                'msrb_yield': ['msrb_yield', 'msrb_yld', 'print_yield', 'trade_yield'],
                'msrb_trade_id': ['msrb_trade_id', 'trade_id', 'msrb_id'],
                'won': ['algo_won', 'won', 'is_won', 'filled', 'is_filled', 'traded', 'executed', 'win', 'done', 'hit'],
                'side': ['side', 'msrb_side', 'signal_side', 'quote_side', 'rfq_side', 'msrb_tradetype'],
                'cover': ['cover', 'cover_yield', 'cover_yld', 'cover_bp']}
        cols_l = {str(c).lower(): c for c in rfq_raw.columns}

        def _pick(role: str):
            if TRACK_RFQ_COLUMNS.get(role) in rfq_raw.columns:
                return TRACK_RFQ_COLUMNS[role]
            return next((cols_l[c] for c in CAND[role] if c in cols_l), None)

        RC = {r: _pick(r) for r in CAND}
        print(f'track_rfq: {len(rfq_raw):,} rows, {rfq_raw.shape[1]} columns: {list(rfq_raw.columns)}'); print('column roles detected (override with TRACK_RFQ_COLUMNS):', RC)
        _need = [r for r in ['time', 'optimal_yield'] if RC[r] is None]
        if RC['algo_yield'] is None:
            print('the track carries no algo mid yield column: the concession x_prod is measured against the match table\'s algo signal yield on the joined requests')
        if _need or 'cusip' not in rfq_raw.columns:
            print(f'Section 12d: cannot identify {_need or ["cusip"]} on the track from its column names; set TRACK_RFQ_COLUMNS and rerun. The production ledger is skipped on this run.')
        else:
            HAS_RFQ = True

# %%
CELL_T('12d. The production ledger: the RFQs we received [2]')
if HAS_RFQ:
    rq = pd.DataFrame({'cusip': rfq_raw['cusip'].astype('string'), 'rfq_ts': to_ns(rfq_raw[RC['time']])})
    for role in ['quantity', 'algo_yield', 'optimal_yield', 'pfill', 'msrb_yield', 'cover', 'final_quote_yield']:
        rq[role] = pd.to_numeric(rfq_raw[RC[role]], errors='coerce').to_numpy() if RC[role] else np.nan
    rq['print_ts'] = to_ns(rfq_raw[RC['print_time']]).to_numpy() if RC['print_time'] else pd.NaT
    def _flag(role: str):
        if not RC[role]:
            return None
        _v = rfq_raw[RC[role]]
        return (_v.astype('string').str.lower().isin(['true', '1', 'y', 'yes', 't']) if (_v.dtype == object or str(_v.dtype).startswith('string')) else pd.to_numeric(_v, errors='coerce').fillna(0) > 0).to_numpy()
    for role in ['is_final', 'quoted', 'pricing_error', 'manual']:
        _f = _flag(role); rq[role] = _f if _f is not None else np.nan
    rq['req_id'] = rfq_raw[RC['req_id']].astype('string').to_numpy() if RC['req_id'] else pd.NA
    rq['pfill_key'] = rfq_raw[RC['pfill_key']].astype('string').fillna('NA').to_numpy() if RC['pfill_key'] else 'NA'
    if RC['side']:
        _sv = rfq_raw[RC['side']].astype('string').str.upper().str.strip(); print('track side values:', _sv.value_counts(dropna=False).head(8).to_dict())
        _smap = {'P': 'P', 'B': 'P', 'BUY': 'P', 'BID': 'P', 'PURCHASE': 'P', 'DEALER BUY': 'P', 'S': 'S', 'SELL': 'S', 'OFFER': 'S', 'ASK': 'S', 'O': 'S', 'DEALER SELL': 'S'}
        rq['side'] = _sv.map(_smap).fillna('P').to_numpy()
    else:
        rq['side'] = 'P'
    _tidn = lambda v: v.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    rq['msrb_trade_id'] = _tidn(rfq_raw[RC['msrb_trade_id']]).to_numpy() if RC['msrb_trade_id'] else pd.array([pd.NA] * len(rq), dtype='string')
    if RC['won']:
        _w = rfq_raw[RC['won']]
        rq['won'] = (_w.astype('string').str.lower().isin(['true', '1', 'y', 'yes', 't', 'won', 'filled', 'done', 'win']).astype(float) if (_w.dtype == object or str(_w.dtype).startswith('string') or str(_w.dtype) == 'bool') else pd.to_numeric(_w, errors='coerce').fillna(0.0).clip(0, 1)).to_numpy(float)
    else:
        rq['won'] = np.nan
    n0 = len(rq)
    # one row per request: the final state where the track versions its rows, else the last row by time; only requests we actually quoted, without pricing errors
    if RC['req_id']:
        if RC['is_final'] and pd.Series(rq['is_final']).notna().any() and bool(pd.Series(rq['is_final']).fillna(False).astype(bool).any()):
            rq = rq[rq['is_final'].fillna(False).astype(bool)]
        rq = rq.sort_values('rfq_ts').drop_duplicates('req_id', keep='last')
    if RC['quoted']:
        _nq = len(rq); rq = rq[rq['quoted'].fillna(False).astype(bool)]; print(f'{_nq - len(rq):,} requests we did not quote dropped')
    if RC['pricing_error']:
        _ne = len(rq); rq = rq[~rq['pricing_error'].fillna(False).astype(bool)]; print(f'{_ne - len(rq):,} requests with a pricing error dropped')
    if RC['manual']:
        print(f'manual takeover on {float(rq["manual"].fillna(False).astype(bool).mean()):.1%} of the remaining requests (kept; the optimal yield is still the optimizer\'s output)')
    rq = rq.dropna(subset=['cusip', 'rfq_ts', 'optimal_yield'])
    rq = rq[(rq['rfq_ts'] >= pd.Timestamp(CFG.first_oos_date)) & (rq['rfq_ts'] < pd.Timestamp(CFG.end_date) + pd.Timedelta(days=1)) & rq['side'].isin(['P', 'S'])].reset_index(drop=True)
    rq['s'] = np.where(rq['side'] == 'P', 1.0, -1.0); rq['x_prod'] = rq['s'] * 100.0 * (rq['optimal_yield'] - rq['algo_yield'])   # NaN where the track has no algo yield: filled from the match table after the join
    _nx = len(rq); rq = rq[~(rq['x_prod'].abs() > 100.0)].reset_index(drop=True)          # a concession beyond 100 bp is a data error (NaN kept: filled after the join)
    if _nx - len(rq):
        print(f'{_nx - len(rq):,} requests dropped for |x_prod| > 100 bp (check the yield units of the optimal and algo yield columns if this is large)')
    rq['has_print'] = rq['msrb_yield'].notna(); rq['month'] = rq['rfq_ts'].dt.to_period('M'); rq['qty_group'] = qty_group(rq['quantity']).to_numpy()
    print(f'RFQ ledger: {len(rq):,} requests in the held-out window ({n0:,} rows on the track); side mix {rq["side"].value_counts(normalize=True).round(3).to_dict()}; {rq["has_print"].mean():.1%} printed; '
          f'pfill logged on {rq["pfill"].notna().mean():.0%}, key on {(rq["pfill_key"] != "NA").mean():.0%}; won flag {"present" if RC["won"] else "absent"}')
    HAS_ALGO_Y = bool(RC['algo_yield']) and rq['x_prod'].notna().mean() > 0.5
    led = rq.groupby('month').agg(requests=('cusip', 'size'), bonds=('cusip', 'nunique'), printed=('has_print', 'mean'), median_x_prod=('x_prod', 'median'), mean_pfill=('pfill', 'mean'), logged_win=('won', 'mean'))
    print('By month: requests, bonds, share that printed, production median concession (bp; positive = less aggressive than the algo mid), mean logged pfill, logged win share:'); display(led.round(3))
    def _xq_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        t_ = frame.dropna(subset=['x_prod']); xq_ = t_.groupby(['side', 'qty_group'], observed=True)['x_prod'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack(); xq_.columns = [f'q{int(round(c * 100))}' for c in xq_.columns]; xq_['n'] = t_.groupby(['side', 'qty_group'], observed=True).size()
        print(f'Production concession x_prod by side x quantity bin (bp; {label}): the range the optimizer actually uses, which the universe grid has to cover:'); display(xq_.round(2))
        _si = float(((t_['x_prod'] >= XGa[0]) & (t_['x_prod'] <= XGa[-1])).mean()); print(f'{_si:.0%} of production concessions fall inside the universe grid [{XGa[0]:g}, {XGa[-1]:g}] bp.')
        return xq_, _si
    if HAS_ALGO_Y:
        xq, _share_in = _xq_table(rq, 'all requests, against the track\'s algo yield')
    else:
        xq, _share_in = pd.DataFrame(), np.nan
    keys_top = rq.loc[rq['pfill_key'] != 'NA', 'pfill_key'].value_counts()
    if len(keys_top):
        print(f'{len(keys_top)} distinct pfill keys; the 10 most used:'); display(keys_top.head(10).to_frame('requests'))

    # ---- join printed requests to the matched-print frame: trade id first, then cusip x side x time (same quantity)
    mo_k = pd.DataFrame({'_id': mo['_id'].to_numpy(), 'cusip': mo['cusip'].astype('string').to_numpy(), 'side': mo['side'].astype(str).to_numpy(), 'trade_ts': to_ns(mo['trade_ts']).to_numpy(), 'qty_mo': pd.to_numeric(mo['msrb_quantity'], errors='coerce').to_numpy(float)})
    mo_k['cusip'] = mo_k['cusip'].astype('string')
    if 'msrb_trade_id' in mo.columns:
        mo_k['tid'] = _tidn(mo['msrb_trade_id']).to_numpy()
    rp = rq[rq['has_print']].copy().reset_index(drop=True); rp['_rid'] = np.arange(len(rp)); rp['_id'] = np.nan; how = []
    if RC['print_time'] and pd.Series(rp['print_ts']).notna().any():
        # the print's own time on the track: exact to the minute, same bond and side, same quantity where both are known
        a0 = rp[['_rid', 'cusip', 'side', 'print_ts', 'quantity']].dropna(subset=['print_ts']).copy(); a0['side'] = a0['side'].astype(str); a0['print_ts'] = to_ns(a0['print_ts']); a0 = a0.sort_values('print_ts')
        b0 = mo_k[['_id', 'cusip', 'side', 'trade_ts', 'qty_mo']].sort_values('trade_ts')
        j0 = pd.merge_asof(a0, b0, left_on='print_ts', right_on='trade_ts', by=['cusip', 'side'], direction='nearest', tolerance=pd.Timedelta(minutes=1))
        okq0 = j0['qty_mo'].isna() | j0['quantity'].isna() | ((j0['qty_mo'] - j0['quantity']).abs() <= 1.0); j0.loc[~okq0, '_id'] = np.nan
        rp['_id'] = rp['_rid'].map(j0.set_index('_rid')['_id']); how.append('the print\'s own time')
    if RC['msrb_trade_id'] and 'tid' in mo_k.columns and rp['msrb_trade_id'].notna().any() and rp['_id'].isna().any():
        _m1 = mo_k.dropna(subset=['tid']).drop_duplicates(['cusip', 'tid'])[['_id', 'cusip', 'tid']].rename(columns={'tid': 'msrb_trade_id'})
        j1 = rp[['_rid', 'cusip', 'msrb_trade_id']].merge(_m1, on=['cusip', 'msrb_trade_id'], how='left'); rp['_id'] = rp['_id'].fillna(pd.Series(j1['_id'].to_numpy(), index=rp.index)); how.append('trade id')
    _miss = rp['_id'].isna()
    if _miss.any():
        a_ = rp.loc[_miss, ['_rid', 'cusip', 'side', 'rfq_ts', 'quantity']].copy(); a_['side'] = a_['side'].astype(str); a_ = a_.sort_values('rfq_ts')
        b_ = mo_k[['_id', 'cusip', 'side', 'trade_ts', 'qty_mo']].sort_values('trade_ts')
        j2 = pd.merge_asof(a_, b_, left_on='rfq_ts', right_on='trade_ts', by=['cusip', 'side'], direction='forward', tolerance=pd.Timedelta(minutes=CFG.rfq_join_minutes))
        okq = j2['qty_mo'].isna() | j2['quantity'].isna() | ((j2['qty_mo'] - j2['quantity']).abs() <= 1.0)
        j2.loc[~okq, '_id'] = np.nan; rp['_id'] = rp['_id'].fillna(rp['_rid'].map(j2.set_index('_rid')['_id'])); how.append('cusip x side x time, same quantity')
    rp = rp.dropna(subset=['_id']).copy(); rp['_id'] = rp['_id'].astype(int)
    pos_map = pd.Series(np.arange(len(mo)), index=mo['_id'].to_numpy()); rp['pos'] = pos_map.reindex(rp['_id'].to_numpy()).to_numpy()
    rp = rp.dropna(subset=['pos']).copy(); rp['pos'] = rp['pos'].astype(int); rp = rp.drop_duplicates('pos').reset_index(drop=True)
    n_printed = int(rq['has_print'].sum()); print(f'{len(rp):,} of {n_printed:,} printed requests joined to the matched-print frame by {" then ".join(how)} ({len(rp) / max(n_printed, 1):.0%}).')
    P = rp['pos'].to_numpy(); sP = S_ARR[P]; ypP = pd.to_numeric(mo['msrb_yield'], errors='coerce').to_numpy(float)[P]; TDP = TD[P]
    _yalgo_mo = pd.to_numeric(mo['algo_signal_yield'], errors='coerce').to_numpy(float)[P]
    if not HAS_ALGO_Y:
        rp['algo_yield'] = _yalgo_mo; rp['x_prod'] = sP * 100.0 * (rp['optimal_yield'].to_numpy(float) - _yalgo_mo)
        rp = rp[~(rp['x_prod'].abs() > 100.0)].reset_index(drop=True); P = rp['pos'].to_numpy(); sP = S_ARR[P]; ypP = pd.to_numeric(mo['msrb_yield'], errors='coerce').to_numpy(float)[P]; TDP = TD[P]
        xq, _share_in = _xq_table(rp, 'joined requests, against the match table\'s algo signal yield')
        rq = rq.drop(columns=['x_prod']).merge(rp[['req_id', 'x_prod']].drop_duplicates('req_id'), on='req_id', how='left') if RC['req_id'] else rq
    o_prod = sP * 100.0 * (ypP - rp['optimal_yield'].to_numpy(float)); o_algo_t = sP * 100.0 * (ypP - rp['algo_yield'].to_numpy(float)); x_prodP = rp['x_prod'].to_numpy(float)
    _ya_gap = np.abs(rp['algo_yield'].to_numpy(float) - _yalgo_mo) * 100.0
    if HAS_ALGO_Y:
        print(f'track algo yield vs match-table algo signal yield on the joined requests: median |gap| {np.nanmedian(_ya_gap):.1f} bp, within 1 bp on {float(np.nanmean(_ya_gap <= 1.0)):.0%}')
    okr = okm[P] & np.isfinite(BASE_RT[P]) & np.isfinite(o_prod); nmr = max(int(pd.Series(MONTH_ARR[P][okr]).nunique()), 1)
    _yd = np.abs(ypP - rp['msrb_yield'].to_numpy(float)); print(f'{int(okr.sum()):,} joined requests carry every input ({nmr} months); the track print and the matched print agree on yield within 0.5 bp on {float((_yd[okr] <= 0.005).mean()):.0%} of them.')
    if okr.sum() < 200:
        print('Section 12d: too few scorable requests; the ledger analyses are skipped on this run.'); HAS_RFQ = False

# %%
CELL_T('12d. The production ledger: the RFQs we received [3]')
if HAS_RFQ:
    winP = o_prod >= 0
    # ---- (2) calibration of the logged pfill at production's own quote, by key and by key component
    g = pd.DataFrame({'key': rp['pfill_key'].astype(str).to_numpy(), 'pf': rp['pfill'].to_numpy(float), 'win': winP.astype(float), 'won': rp['won'].to_numpy(float), 'month': MONTH_ARR[P].astype(str), 'qty_group': QTY_ARR[P], 'x_prod': x_prodP})[okr]
    _parts = g['key'].str.split('_'); g['cpn_bin'] = _parts.str[0]; g['qty_tok'] = _parts.str[-1]; g['call_grp'] = np.where(_parts.str.len() >= 3, _parts.str[1], 'n/a'); g['rating_grp'] = np.where(_parts.str.len() == 4, _parts.str[2], 'n/a')
    has_pf = g['pf'].notna()
    if has_pf.mean() > 0.2:
        _bp = float(np.mean((g.loc[has_pf, 'pf'] - g.loc[has_pf, 'win']) ** 2))
        print(f'Logged pfill at production\'s quote: mean {g.loc[has_pf, "pf"].mean():.3f} vs realised crossing share {g.loc[has_pf, "win"].mean():.3f} on {int(has_pf.sum()):,} requests; Brier {_bp:.4f}' + (f'; logged win share {np.nanmean(g["won"]):.3f}, crossing proxy agrees with the won flag on {float((g["win"] == g["won"]).mean()):.0%}' if RC['won'] else ''))
        cal_key = g[has_pf].groupby('key').agg(requests=('win', 'size'), mean_pfill=('pf', 'mean'), realised=('win', 'mean'), median_x_prod=('x_prod', 'median')); cal_key['gap'] = cal_key['realised'] - cal_key['mean_pfill']
        _min_key = max(50, CFG.n_min_cell // 20); cal_key = cal_key[cal_key['requests'] >= _min_key].sort_values('requests', ascending=False)
        print(f'Calibration by pfill key (keys with >= {_min_key} scorable requests; gap = realised - logged):'); display(cal_key.head(20).round(3))
        comp_tabs = {}
        for cn in ['cpn_bin', 'call_grp', 'rating_grp', 'qty_tok']:
            t_ = g[has_pf].groupby(cn).agg(requests=('win', 'size'), mean_pfill=('pf', 'mean'), realised=('win', 'mean')); t_['gap'] = t_['realised'] - t_['mean_pfill']; comp_tabs[cn] = t_[t_['requests'] >= max(50, CFG.n_min_cell // 20)]
        print('Calibration by key component (coupon bin | call group | rating group | quantity token):'); display(pd.concat(comp_tabs, names=['component', 'level']).round(3))
    else:
        cal_key = pd.DataFrame(); comp_tabs = {}; print('The track carries no usable pfill column; the calibration of the logged curve is skipped.')

    # ---- (3) the grouping test at production's quote: Pr(o_A >= x_prod) per request from each estimator, scored against the logged pfill
    def interp_at(Pm: np.ndarray, grid: np.ndarray, x: np.ndarray) -> np.ndarray:
        xc = np.clip(x, grid[0], grid[-1]); j = np.clip(np.searchsorted(grid, xc, 'right') - 1, 0, len(grid) - 2); w = (xc - grid[j]) / (grid[j + 1] - grid[j]); r = np.arange(len(xc))
        return (1.0 - w) * Pm[r, j] + w * Pm[r, j + 1]

    cellA = {}
    for gname, grp, shf in [('cell: side x size x production-style key', PK_ARR, None), ('cell: side x size x cluster', CL_ARR, None), ('cell: cluster, shifted by residual deviance', CL_ARR, DEV_ARR)]:
        arr = np.full((len(mo), len(XG)), np.nan)
        for m in MONTHS_MO[1:]:
            trm = MONTH_ARR < m; tem = MONTH_ARR == m
            if trm.sum() >= CFG.pfill_min_train and tem.any():
                arr[tem] = cell_pfill(trm, tem, grp, shf, o=O_A, deltas=XG)
        cellA[gname] = arr
    EST = {'logged pfill (production)': rp['pfill'].to_numpy(float), 'desk curve (side, trailing window)': interp_at(desk_A_x[P], XGa, x_prodP)}
    for gname, arr in cellA.items():
        EST[gname] = interp_at(arr[P], XGa, x_prodP)
    EST['GBM: + IPCA and state-space'] = interp_at(gbm_A_x[P], XGa, x_prodP)
    okg = okr.copy()
    for v_ in EST.values():
        okg &= np.isfinite(v_)
    grp_rows = []; _mP = MONTH_ARR[P].astype(str); _base_b = None
    for name, v_ in EST.items():
        p_ = np.clip(v_, 1e-4, 1 - 1e-4); y_ = winP.astype(float)
        bm = pd.Series((p_[okg] - y_[okg]) ** 2).groupby(_mP[okg]).mean()
        if name == 'logged pfill (production)':
            _base_b = bm
        rel = (1.0 - bm / _base_b) if _base_b is not None else pd.Series(np.nan, index=bm.index)
        grp_rows.append({'estimator': name, 'requests': int(okg.sum()), 'Brier': float(np.mean((p_[okg] - y_[okg]) ** 2)), 'log-loss': float(-np.mean(y_[okg] * np.log(p_[okg]) + (1 - y_[okg]) * np.log(1 - p_[okg]))), 'mean predicted': float(p_[okg].mean()), 'realised': float(y_[okg].mean()),
                         'Brier removed vs logged': float(1.0 - np.mean((p_[okg] - y_[okg]) ** 2) / np.mean((np.clip(EST['logged pfill (production)'], 1e-4, 1 - 1e-4)[okg] - y_[okg]) ** 2)), 'months >= bar': int((rel >= CFG.grouping_bar).sum()), 'months': int(len(bm))})
    grouping = pd.DataFrame(grp_rows).set_index('estimator')
    grouping['passes bar'] = (grouping['Brier removed vs logged'] >= CFG.grouping_bar) & (grouping['months >= bar'] >= min(4, int(grouping['months'].max())))
    print(f'The grouping test at production\'s own quote on {int(okg.sum()):,} requests: Pr(o_A >= x_prod) from each estimator, scored on the realised crossing. Bar: >= {CFG.grouping_bar:.0%} of the logged pfill\'s Brier removed in >= 4 held-out months:'); display(grouping.round(4))

    # ---- (4) production's quote against ours on the same requests
    RPOL = {'production (logged optimal yield)': o_prod, 'algo mid at 0 (track algo yield)': o_algo_t, 'A at 0 (match-table algo yield)': O_A[P], 'S at 0': O_S[P], 'SQ at 0': sP * errSQ[P], 'S + production concession x_prod': O_S[P] - x_prodP}
    if RC['final_quote_yield'] and rp['final_quote_yield'].notna().mean() > 0.5:
        _ofq = sP * 100.0 * (ypP - rp['final_quote_yield'].to_numpy(float))
        if np.nanmean(np.abs(_ofq - o_prod)) > 0.05:
            RPOL['as quoted (final quote yield, after manual changes)'] = _ofq
    if RC['cover'] and rp['cover'].notna().mean() > 0.2:
        _cv = rp['cover'].to_numpy(float); _okc = okr & np.isfinite(_cv) & (o_prod >= 0)
        print(f'Logged cover on the track (units as logged): median {np.nanmedian(_cv[okr]):.3f}, 10th-90th {np.nanpercentile(_cv[okr], 10):.3f} to {np.nanpercentile(_cv[okr], 90):.3f}; correlation with our distance through the print on wins {float(np.corrcoef(_cv[_okc], o_prod[_okc])[0, 1]) if _okc.sum() > 30 else np.nan:+.2f}')
    for pname, lab in [('A, GBM + IPCA fill model, x + charge', 'A, GBM fill model, x + charge (12c)'), ('S, GBM + IPCA fill model, expected P&L (the Section 11b engine)', 'engine: S, GBM + IPCA, expected P&L (12c)')]:
        if pname in XSTAR:
            xs = XSTAR[pname][P]; base_o = O_A[P] if pname.startswith('A') else O_S[P]; RPOL[lab] = np.where(np.isfinite(xs), base_o - xs, -np.inf)

    def rfq_metrics(oq: np.ndarray, mask: np.ndarray) -> dict:
        ok = mask; quoted = ok & np.isfinite(oq); win = quoted & (oq >= 0); miss = quoted & ~(oq >= 0); pnl = np.where(win, BASE_RT[P] - oq, 0.0)
        d_ = pd.Series(pnl[ok]).groupby(TDP[ok]).mean(); dw = pd.Series(win[ok].astype(float)).groupby(TDP[ok]).mean()
        return {'requests': int(ok.sum()), 'win rate (print crossed)': float(win.sum() / max(ok.sum(), 1)), 'declined': float(1.0 - quoted.sum() / max(ok.sum(), 1)), '|quote - print| (bp)': float(np.abs(oq[quoted]).mean()) if quoted.any() else np.nan,
                'within 2 bp': float((np.abs(oq[quoted]) <= 2.0).mean()) if quoted.any() else np.nan, 'through the print | win (bp)': float(oq[win].mean()) if win.any() else np.nan, 'short of the print | miss (bp)': float(-oq[miss].mean()) if miss.any() else np.nan,
                'RT P&L | win (bp)': float((BASE_RT[P] - oq)[win].mean()) if win.any() else np.nan, 'RT P&L per request (bp)': float(d_.mean()) if len(d_) else np.nan, '$ per month ($k)': float((pnl[win] * DPB[P][win]).sum() / 1e3 / nmr), '_d': d_, '_w': dw}

    rr_rows, rr_daily = [], {}
    for name, oq in RPOL.items():
        r_ = rfq_metrics(oq, okr); rr_daily[name] = (r_.pop('_d'), r_.pop('_w')); rr_rows.append({'policy': name, **r_})
    rfq_tab = pd.DataFrame(rr_rows).set_index('policy'); _pn = 'production (logged optimal yield)'
    if RC['won']:
        rfq_tab.loc[_pn, 'logged win rate'] = float(np.nanmean(rp['won'].to_numpy(float)[okr]))
    for name in rfq_tab.index:
        if name == _pn:
            continue
        dp = (rr_daily[name][0] - rr_daily[_pn][0]).dropna(); dw = (rr_daily[name][1] - rr_daily[_pn][1]).dropna(); dm = dp.groupby(pd.DatetimeIndex(dp.index).to_period('M')).mean()
        rfq_tab.loc[name, 'RT P&L boot t vs production'] = block_bootstrap_t(dp.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed); rfq_tab.loc[name, 'win rate boot t vs production'] = block_bootstrap_t(dw.to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed); rfq_tab.loc[name, 'months above production'] = int((dm > 0).sum())
    rfq_tab['beats production (bar)'] = ((rfq_tab['RT P&L per request (bp)'] - rfq_tab.loc[_pn, 'RT P&L per request (bp)']) >= CFG.interaction_bar_bp) & (rfq_tab['RT P&L boot t vs production'] > 3) & (rfq_tab['months above production'] >= min(4, nmr))
    print(f'Production\'s logged quote against ours on the same {int(okr.sum()):,} requests (bid side unless the track carries offers): win rate, cover, realised round trip; bar vs production as before:'); display(rfq_tab.round(3))
    _gap_rfq = float(np.nanmean((rp['pfill'].to_numpy(float) * (x_prodP + CHG[P]))[okr])) if rp['pfill'].notna().mean() > 0.2 else np.nan
    if np.isfinite(_gap_rfq):
        print(f'Production\'s own expectation on these requests, pfill x (x_prod + charge): {_gap_rfq:+.2f} bp per request; the round trip paid {rfq_tab.loc[_pn, "RT P&L per request (bp)"]:+.2f}.')
    KEYR = [_pn, 'S at 0'] + [k_ for k_ in ['engine: S, GBM + IPCA, expected P&L (12c)'] if k_ in RPOL]
    by_q_rows = []
    _min_cell = max(50, CFG.n_min_cell // 20)
    for q_ in QTY_LABELS:
        sel = okr & (QTY_ARR[P] == q_)
        if sel.sum() < _min_cell:
            continue
        for name in KEYR:
            r_ = rfq_metrics(RPOL[name], sel); by_q_rows.append({'qty_group': q_, 'policy': name, 'requests': r_['requests'], 'win rate': r_['win rate (print crossed)'], '|quote - print| (bp)': r_['|quote - print| (bp)'], 'RT P&L per request (bp)': r_['RT P&L per request (bp)']})
    by_q = pd.DataFrame(by_q_rows).set_index(['qty_group', 'policy']).unstack('policy') if by_q_rows else pd.DataFrame()
    if len(by_q):
        print('By quantity bin: production vs S at 0 vs the engine on the requests (win rate, cover, P&L per request):'); display(by_q.round(3))
    ch_rows = []
    if '_dimv' in globals():
        for dname, arr in _dimv.items():
            for lvl in (_dimo.get(dname) or sorted(pd.unique(arr[P][okr]))):
                sel = okr & (arr[P] == lvl)
                if sel.sum() < _min_cell:
                    continue
                for name in KEYR:
                    r_ = rfq_metrics(RPOL[name], sel); ch_rows.append({'dimension': dname, 'cell': str(lvl), 'policy': name, 'requests': r_['requests'], 'win rate': r_['win rate (print crossed)'], 'RT P&L per request (bp)': r_['RT P&L per request (bp)']})
    rfq_char = pd.DataFrame(ch_rows)
    if len(rfq_char):
        _pv = rfq_char.pivot_table(index=['dimension', 'cell'], columns='policy', values=['win rate', 'RT P&L per request (bp)'], aggfunc='first')
        for k_ in KEYR[1:]:
            _pv[('delta P&L vs production (bp)', k_)] = _pv[('RT P&L per request (bp)', k_)] - _pv[('RT P&L per request (bp)', _pn)]
        print('By characteristic: production vs S at 0 vs the engine on the requests:'); display(_pv.round(3))

    # ---- (5) the production proxy for the universe: the algo mid plus the median production concession of the bond's key
    kk = rq[(rq['pfill_key'] != 'NA') & rq['x_prod'].notna()].copy()
    if len(kk) >= CFG.proxy_min_rfqs:
        kk['prefix'] = kk['pfill_key'].str.rsplit('_', n=1).str[0]; kk['qtok'] = kk['pfill_key'].str.rsplit('_', n=1).str[-1]
        bond_prefix = kk.groupby('cusip')['prefix'].agg(lambda x: x.value_counts().index[0]); qtok_map = kk.groupby('qty_group', observed=True)['qtok'].agg(lambda x: x.value_counts().index[0])
        med_key = kk.groupby('pfill_key')['x_prod'].agg(['median', 'size']); med_key = med_key.loc[med_key['size'] >= CFG.proxy_min_rfqs, 'median']
        u_prefix = pd.Series(mo['cusip'].astype('string').to_numpy()).map(bond_prefix); u_qtok = pd.Series(QTY_ARR).map(qtok_map); u_key = (u_prefix.astype('string') + '_' + u_qtok.astype('string'))
        x_key = u_key.map(med_key).to_numpy(float); proxy_src = 'key'
    else:
        x_key = np.full(len(mo), np.nan); print('too few requests with a pfill key for a key-level proxy; the side x size medians are used throughout')
    med_sq = rq.dropna(subset=['x_prod']).groupby(['side', 'qty_group'], observed=True)['x_prod'].median()
    _sq_side = SIDE_ARR if (rq['side'] == 'S').any() else np.full(len(mo), 'P')           # no offers on the track: the offer takes the bid's medians (flagged)
    x_sq = med_sq.reindex(pd.MultiIndex.from_arrays([_sq_side, QTY_ARR])).to_numpy(float); x_sq = np.where(np.isfinite(x_sq), x_sq, float(rq['x_prod'].dropna().median()) if rq['x_prod'].notna().any() else 0.0)
    x_proxy = np.where(np.isfinite(x_key), x_key, x_sq); _cov_key = float(np.isfinite(x_key[okm]).mean())
    # validation on the overlap: the proxy and the 12c replica against production's logged concession on the same requests
    _val = pd.DataFrame({'proxy (median x_prod by key)': x_proxy[P], 'replica (12c, desk curve x (x + charge))': XSTAR[PRODN][P] if PRODN in XSTAR else np.nan, 'logged x_prod': x_prodP})[okr]
    val_rows = []
    for c_ in ['proxy (median x_prod by key)', 'replica (12c, desk curve x (x + charge))']:
        okv = np.isfinite(_val[c_]) & np.isfinite(_val['logged x_prod'])
        val_rows.append({'representation': c_, 'requests': int(okv.sum()), 'MAE vs logged x_prod (bp)': float((_val.loc[okv, c_] - _val.loc[okv, 'logged x_prod']).abs().mean()), 'corr': float(_val.loc[okv, c_].corr(_val.loc[okv, 'logged x_prod'])) if okv.sum() > 10 else np.nan, 'mean (bp)': float(_val.loc[okv, c_].mean()), 'logged mean (bp)': float(_val.loc[okv, 'logged x_prod'].mean())})
    proxy_val = pd.DataFrame(val_rows).set_index('representation')
    print(f'The production proxy on the universe: a key-level median concession for {_cov_key:.0%} of customer prints, the side x size median for the rest{" (the offer side takes the bid medians: the track carries no offers)" if not (rq["side"] == "S").any() else ""}. Validation on the overlap against the logged concession:'); display(proxy_val.round(3))
    upol = {'production proxy: A + median x_prod by key': (errA, x_proxy), 'production replica (12c): A, desk curve, x + charge': (errA, XSTAR[PRODN]), 'A at 0': (errA, 0.0), 'S at 0': (errS, 0.0)}
    if 'S, GBM + IPCA fill model, expected P&L (the Section 11b engine)' in XSTAR:
        upol['engine: S, GBM + IPCA, expected P&L'] = (errS, XSTAR['S, GBM + IPCA fill model, expected P&L (the Section 11b engine)'])
    u_rows, u_daily = [], {}
    for name, (err_, xs) in upol.items():
        for sd_, msk in [('ALL', okm), ('P', okm & IS_P), ('S', okm & IS_S)]:
            r_ = pnl_policy(err_, xs, msk, BASE_RT); d_ = r_.pop('_daily')
            if sd_ == 'ALL':
                u_daily[name] = d_
            u_rows.append({'policy': name, 'side': sd_, 'fill share': r_['fill share'], 'RT P&L per print (bp)': r_['P&L per print (bp)'], '$ per month ($k)': r_['$ per month ($k)']})
    universe = pd.DataFrame(u_rows).set_index(['policy', 'side']).unstack('side'); _up = 'production proxy: A + median x_prod by key'
    for name in upol:
        universe.loc[name, ('boot t vs proxy', 'ALL')] = block_bootstrap_t((u_daily[name] - u_daily[_up]).dropna().to_numpy(), CFG.block_days, CFG.n_boot, CFG.seed) if name != _up else np.nan
    print(f'The universe ({int(okm.sum()):,} customer prints, both sides) with production represented by the proxy instead of the replica:'); display(universe.round(3))

    # ---- figures
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    for sd_, c_ in [('P', '#4C72B0'), ('S', '#DD8452')]:
        v_ = rq.loc[rq['side'] == sd_, 'x_prod'].dropna()
        if len(v_):
            ax[0, 0].hist(v_.clip(-30, 30), bins=60, alpha=0.6, color=c_, label=f'{sd_}: production x_prod ({len(v_):,} requests)')
    if PRODN in XSTAR:
        ax[0, 0].hist(np.clip(XSTAR[PRODN][P][okr], -30, 30), bins=60, histtype='step', color='k', lw=1.2, label='12c replica x* on the same requests')
    ax[0, 0].axvline(0, color='k', lw=0.6); ax[0, 0].set_xlabel('concession on the algo mid (bp; positive = less aggressive)'); ax[0, 0].set_title('What production actually quotes vs what the replica assumed', fontsize=10); ax[0, 0].legend(fontsize=7)
    if len(cal_key):
        ax[0, 1].scatter(cal_key['mean_pfill'], cal_key['realised'], s=np.sqrt(cal_key['requests']) * 2, alpha=0.6, color='#4C72B0'); ax[0, 1].plot([0, 1], [0, 1], 'k:', lw=0.8); ax[0, 1].set_xlabel('mean logged pfill at production\'s quote'); ax[0, 1].set_ylabel('realised crossing share'); ax[0, 1].set_title('Calibration of the production fill curve by pfill key (area = requests)', fontsize=10)
    elif comp_tabs and any(len(v_) for v_ in comp_tabs.values()):
        _ct = pd.concat({k_: v_['gap'] for k_, v_ in comp_tabs.items() if len(v_)}); _ct.index = [f'{a_}: {b_}' for a_, b_ in _ct.index]
        ax[0, 1].barh(np.arange(len(_ct)), _ct.to_numpy(float), color=['#2E8B57' if v_ >= 0 else '#C44E52' for v_ in _ct]); ax[0, 1].set_yticks(np.arange(len(_ct))); ax[0, 1].set_yticklabels(_ct.index, fontsize=7); ax[0, 1].axvline(0, color='k', lw=0.6); ax[0, 1].set_xlabel('realised crossing share - logged pfill'); ax[0, 1].set_title('Calibration gap of the production fill curve by key component', fontsize=10)
    else:
        ax[0, 1].set_axis_off()
    _gp = grouping.sort_values('Brier', ascending=False); y_ = np.arange(len(_gp))
    ax[0, 2].barh(y_, _gp['Brier'], color=['#2E8B57' if b_ else ('#4C72B0' if i_.startswith('logged') else '#8C8C8C') for i_, b_ in zip(_gp.index, _gp['passes bar'])]); ax[0, 2].set_yticks(y_); ax[0, 2].set_yticklabels([f'{i_}  ({r_:+.1%})' for i_, r_ in zip(_gp.index, _gp['Brier removed vs logged'])], fontsize=7); ax[0, 2].set_xlabel('Brier at production\'s quote'); ax[0, 2].set_title('The grouping test on production\'s own requests (blue = logged pfill; green = clears the bar)', fontsize=10)
    _rt = rfq_tab.sort_values('RT P&L per request (bp)'); y_ = np.arange(len(_rt))
    ax[1, 0].barh(y_, _rt['win rate (print crossed)'], color=['#4C72B0' if i_ == _pn else '#8C8C8C' for i_ in _rt.index]); ax[1, 0].set_yticks(y_); ax[1, 0].set_yticklabels([i_[:44] for i_ in _rt.index], fontsize=7); ax[1, 0].set_title('Win rate on the requests (blue = production)', fontsize=10); ax[1, 0].axvline(rfq_tab.loc[_pn, 'win rate (print crossed)'], color='#4C72B0', ls=':', lw=1)
    ax[1, 1].barh(y_, _rt['RT P&L per request (bp)'], color=['#2E8B57' if b_ else ('#4C72B0' if i_ == _pn else '#8C8C8C') for i_, b_ in zip(_rt.index, _rt['beats production (bar)'])]); ax[1, 1].set_yticks(y_); ax[1, 1].set_yticklabels([f'(t {t_:+.1f})' if np.isfinite(t_) else '' for t_ in _rt['RT P&L boot t vs production']], fontsize=7); ax[1, 1].set_title('Realised round-trip P&L per request (bp; green = clears the bar)', fontsize=10); ax[1, 1].axvline(rfq_tab.loc[_pn, 'RT P&L per request (bp)'], color='#4C72B0', ls=':', lw=1)
    if len(by_q):
        for name, c_ in zip(KEYR, ['#4C72B0', '#DD8452', '#2E8B57']):
            if ('RT P&L per request (bp)', name) in by_q.columns:
                ax[1, 2].plot(np.arange(len(by_q)), by_q[('RT P&L per request (bp)', name)], marker='o', color=c_, label=name[:40])
        ax[1, 2].set_xticks(np.arange(len(by_q))); ax[1, 2].set_xticklabels(by_q.index, rotation=30, fontsize=8); ax[1, 2].axhline(0, color='k', lw=0.5); ax[1, 2].set_ylabel('RT P&L per request (bp)'); ax[1, 2].set_title('By quantity bin: production vs S at 0 vs the engine', fontsize=10); ax[1, 2].legend(fontsize=7)
    else:
        ax[1, 2].set_axis_off()
    fig.suptitle('The production ledger: what production quoted, whether its fill curve is calibrated, and how its quote compares with ours on the same requests', fontsize=12); plt.tight_layout(); savefig('12d_rfq_ledger')
    record('rfq_ledger', roles=RC, requests=int(len(rq)), printed=int(n_printed), joined=int(len(rp)), scored=int(okr.sum()), months=nmr, by_month=led.round(4).reset_index().astype({'month': str}).to_dict(orient='records'), x_prod_by_cell=flat_records(xq.reset_index()) if len(xq) else [], share_x_prod_in_grid=_share_in,
           calibration_by_key=cal_key.round(4).reset_index().to_dict(orient='records') if len(cal_key) else [], calibration_by_component={k_: v_.round(4).reset_index().to_dict(orient='records') for k_, v_ in comp_tabs.items()},
           grouping=grouping.round(5).reset_index().to_dict(orient='records'), policies=rfq_tab.round(4).reset_index().to_dict(orient='records'), by_qty=flat_records(by_q.reset_index()) if len(by_q) else [], by_characteristic=rfq_char.round(4).to_dict(orient='records') if len(rfq_char) else [],
           proxy_validation=proxy_val.round(4).reset_index().to_dict(orient='records'), proxy_key_coverage=_cov_key, universe=flat_records(universe.reset_index()))
elif 'rfq_raw' in globals() and rfq_raw is not None and not rfq_raw.empty:
    print('Section 12d: the ledger could not be scored on this run (see the messages above).')

# %% [markdown]
# ## 13. Where the P&L lives: markout by characteristic, factor coordinate, side and size
#
# The same segmentation the fair-mid work used, now scored in the currency the desk uses. Every customer print
# inherits its characteristics, its three factor betas and a point-in-time beta-space cluster from its residual
# date; for each segment and each of five quote rules (the algo quote, S, SQ, F and the level model D) the table
# gives the fill share, the factor-hedged P&L per print at the record horizon, a Fama-MacBeth t of the daily P&L,
# and dollars per month. Three views: (1) one table per axis, with the S bars and the D diamond per axis in one
# figure; (2) the side x characteristic heatmaps of the S P&L per print, because a quote that earns on the bid in a
# segment and loses on the offer has to be gated by both; (3) the duration x call x rating cells that carry the
# dollars, best and worst, and how concentrated they are. The MAE versions of these tables are in the registry of
# the v3.9 run and the v3.8 paper; they are not repeated here.

# %%
CELL_T('13. Where the P&L lives: markout by characteristic, factor c [1]')
if HAS_TRADES and 'mo' in globals() and not mo.empty:
    bk = mo.copy()
    bk['dur_bucket'] = pd.cut(bk['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']).astype(str)
    bk['rating_bucket'] = pd.cut(bk['rating_score'].fillna(-1), bins=[-2, -0.5, 14.5, 17.5, 20.5, 21.5], labels=['NR', 'BBB and below', 'A', 'AA', 'AAA']).astype(str)
    bk['liq_q'] = bk.groupby('trade_date', observed=True)['z_liquidity_20'].transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1 illiquid', 'Q2', 'Q3', 'Q4', 'Q5 liquid']).astype(str) if x.notna().sum() > 50 else 'NA')
    for j in range(CFG.selected_k):
        bk[f'beta{j+1}_tercile'] = bk.groupby('trade_date', observed=True)[f'beta{j+1}'].transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['low', 'mid', 'high']).astype(str) if x.notna().sum() > 30 else 'NA')
    bk['cluster'] = bk['beta_cluster'].map(CLUSTER_LABEL).fillna('NA')
    bk['side_label'] = bk['side'].map({'P': 'P dealer buys (bid)', 'S': 'S dealer sells (offer)'}).fillna(bk['side'].astype(str))
    bk['cell'] = bk['dur_bucket'] + ' | ' + bk['call_structure'].astype(str) + ' | ' + bk['rating_bucket']
    MO_MIDS = {k_: v_ for k_, v_ in {'A algo quote': 'e_algo_bp', 'S same-side EWMA': 'e_S_bp', 'SQ same-side EWMA + size intercept': 'e_SQ_bp', 'F side intercept + rho x own EWMA': 'e_F_bp', 'D level model': 'e_D_bp'}.items() if v_ in bk.columns}
    MO_L = {k_: k_.split()[0] for k_ in MO_MIDS}
    # per-print P&L (bp) and dollars for each rule at the record horizon, NaN where the print has no mark / residual path
    for k_, c in MO_MIDS.items():
        fill, pnl, _ = markout_eval(bk[c].to_numpy(float), HR); ok = np.isfinite(pnl); L = MO_L[k_]
        bk[f'v_{L}'] = np.where(ok, np.where(fill, pnl, 0.0), np.nan); bk[f'd_{L}'] = bk[f'v_{L}'] * DPB; bk[f'f_{L}'] = np.where(ok, fill.astype(float), np.nan)

    def pnl_table(frame: pd.DataFrame, by: str, n_min: int) -> pd.DataFrame:
        g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size()})
        for k_, L in MO_L.items():
            out[f'fill {L}'] = g[f'f_{L}'].mean(); out[f'P&L/print {L} (bp)'] = g[f'v_{L}'].mean()
            d = frame.groupby([by, 'trade_date'], observed=True)[f'v_{L}'].mean().groupby(level=0)
            out[f't {L}'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
            out[f'$/mo {L} ($k)'] = g[f'd_{L}'].sum() / 1e3 / months_mo
        out['share of prints'] = out['n'] / len(frame)
        return out[out['n'] >= n_min]

    SEGMENTS = {'duration bucket': 'dur_bucket', 'call structure': 'call_structure', 'rating bucket': 'rating_bucket', 'MSRB side': 'side_label', 'trade size': 'qty_group', 'industry': 'industry_bucket', 'state bucket': 'state_bucket', 'liquidity quintile': 'liq_q', 'beta-space cluster': 'cluster', 'beta1 tercile': 'beta1_tercile', 'beta2 tercile': 'beta2_tercile', 'beta3 tercile': 'beta3_tercile'}
    ORDERS = {'duration bucket': ['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11'], 'liquidity quintile': ['Q1 illiquid', 'Q2', 'Q3', 'Q4', 'Q5 liquid'], 'MSRB side': ['P dealer buys (bid)', 'S dealer sells (offer)'], 'trade size': QTY_LABELS, 'beta1 tercile': ['low', 'mid', 'high'], 'beta2 tercile': ['low', 'mid', 'high'], 'beta3 tercile': ['low', 'mid', 'high']}
    seg_pnl = {}
    for name, col in SEGMENTS.items():
        if col in bk.columns and bk[col].notna().any():
            t_ = pnl_table(bk.dropna(subset=[col]), col, CFG.n_min_cell)
            order = [o_ for o_ in ORDERS.get(name, []) if o_ in t_.index]
            seg_pnl[name] = t_.reindex(order) if order else t_
            print(f'Factor-hedged markout by {name} (h = {HR}; P&L per print in bp, FM t of the daily P&L, dollars per month):'); display(seg_pnl[name].round(2))
    cells_pnl = pnl_table(bk, 'cell', CFG.n_min_cell).sort_values('$/mo S ($k)', ascending=False)
    print(f'Duration x call x rating cells by the dollars S earns per month (cells with >= {CFG.n_min_cell:,} prints): best ten and worst five'); display(pd.concat([cells_pnl.head(10), cells_pnl.tail(5)]).round(2))
    _pos = cells_pnl['$/mo S ($k)']
    _possum = float(_pos.clip(lower=0).sum())
    print(f'share of positive S dollars from the top 10 cells: {(_pos.head(10).clip(lower=0).sum() / _possum if _possum > 0 else float("nan")):.0%} | cells where S loses money: {int((_pos < 0).sum())} of {len(_pos)} ({bk["cell"].isin(_pos[_pos < 0].index).mean():.0%} of prints)')

    # (1) one panel per axis: S P&L per print (bars, 95% band from the FM t) with D as the diamond, labels carry $k per month
    GREEN, RED, INK = '#2E8B57', '#C44E52', '#222222'
    axes_show = [k_ for k_ in ['MSRB side', 'trade size', 'duration bucket', 'call structure', 'rating bucket', 'industry', 'beta-space cluster', 'beta1 tercile', 'beta2 tercile', 'beta3 tercile', 'liquidity quintile', 'state bucket'] if k_ in seg_pnl and len(seg_pnl[k_])]
    _nr = max(1, int(np.ceil(len(axes_show) / 3)))
    fig, axs = plt.subplots(_nr, 3, figsize=(20, 4.8 * _nr), squeeze=False); axs = axs.ravel()
    for a, k_ in zip(axs, axes_show):
        t_ = seg_pnl[k_].dropna(subset=['P&L/print S (bp)'])[::-1]
        y = np.arange(len(t_)); v = t_['P&L/print S (bp)'].to_numpy(float); se = np.minimum((v / t_['t S'].replace(0, np.nan)).abs().fillna(0.0).to_numpy(float), 3.0 * np.abs(v) + 5.0)
        a.barh(y, v, color=[GREEN if x_ >= 0 else RED for x_ in v], xerr=1.96 * se, error_kw={'ecolor': INK, 'lw': 0.8, 'capsize': 2}, alpha=0.9, label='S same-side EWMA (95% band)')
        if 'P&L/print D (bp)' in t_.columns:
            a.plot(t_['P&L/print D (bp)'].to_numpy(float), y, 'D', color=INK, ms=4, label='D level model')
        a.axvline(0, color=INK, lw=0.6); a.set_yticks(y); a.set_yticklabels([f'{i_}  ({s_:.0%}, ${d_:,.0f}k/mo)' for i_, s_, d_ in zip(t_.index, t_['share of prints'], t_['$/mo S ($k)'])], fontsize=8)
        a.set_title(f'By {k_}', fontsize=10); a.set_xlabel('factor-hedged P&L per print (bp)', fontsize=8); a.grid(axis='x', alpha=0.3)
    for a in axs[len(axes_show):]:
        a.set_visible(False)
    if axes_show:
        axs[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Where the P&L lives: factor-hedged markout per print by segment, S (bars) and D (diamonds); labels carry the share of prints and S dollars per month', fontsize=12); plt.tight_layout(rect=(0, 0, 1, 0.985)); savefig('13_pnl_by_characteristic')

    # (2) side x characteristic heatmaps of the S P&L per print
    SIDES = [x_ for x_ in ['P dealer buys (bid)', 'S dealer sells (offer)'] if x_ in bk['side_label'].unique()]
    SIDE_AXES = {'duration bucket': 'dur_bucket', 'call structure': 'call_structure', 'rating bucket': 'rating_bucket', 'trade size': 'qty_group', 'beta-space cluster': 'cluster', 'industry': 'industry_bucket'}
    side_tabs = {}
    for name, col in SIDE_AXES.items():
        if col not in bk.columns or not bk[col].notna().any():
            continue
        parts = {x_: pnl_table(bk[bk['side_label'] == x_], col, CFG.n_min_cell) for x_ in SIDES}
        g_ = pd.DataFrame({x_: p['P&L/print S (bp)'] for x_, p in parts.items()}); t_ = pd.DataFrame({x_: p['t S'] for x_, p in parts.items()}); d_ = pd.DataFrame({x_: p['$/mo S ($k)'] for x_, p in parts.items()}); f_ = pd.DataFrame({x_: p['fill S'] for x_, p in parts.items()})
        order = [o_ for o_ in ORDERS.get(name, []) if o_ in g_.index] or list(g_.index)
        side_tabs[name] = {'pnl': g_.reindex(order), 't': t_.reindex(order), '$': d_.reindex(order), 'fill': f_.reindex(order)}
        show = side_tabs[name]['pnl'].round(2).astype(str) + np.where(side_tabs[name]['t'].abs() >= 2, '*', '') + '  ($' + side_tabs[name]['$'].round(0).fillna(0).astype(int).astype(str) + 'k)'
        print(f'S factor-hedged P&L per print by {name} x side (bp; * = |FM t| >= 2; dollars per month in brackets):'); display(show)
    hm_axes = list(side_tabs)
    if hm_axes:
        fig, axs = plt.subplots(1, len(hm_axes), figsize=(5.0 * len(hm_axes), 5.6), squeeze=False); axs = axs.ravel()
        vmax = max(np.nanmax(np.abs(side_tabs[k_]['pnl'].to_numpy(float))) for k_ in hm_axes) or 1.0
        for a, k_ in zip(axs, hm_axes):
            g_ = side_tabs[k_]['pnl']; t_ = side_tabs[k_]['t']
            im = a.imshow(g_.to_numpy(float), cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
            a.set_xticks(range(g_.shape[1])); a.set_xticklabels([x_.split(' ')[0] + ' ' + x_.split('(')[-1].rstrip(')') for x_ in g_.columns], fontsize=8)
            a.set_yticks(range(g_.shape[0])); a.set_yticklabels(g_.index, fontsize=8); a.set_title(f'{k_} x side', fontsize=10); a.grid(False)
            for i_ in range(g_.shape[0]):
                for j_ in range(g_.shape[1]):
                    v = g_.iloc[i_, j_]; tt = t_.iloc[i_, j_]
                    if np.isfinite(v):
                        a.text(j_, i_, f'{v:+.1f}' + ('*' if np.isfinite(tt) and abs(tt) >= 2 else ''), ha='center', va='center', fontsize=8, fontweight='bold' if np.isfinite(tt) and abs(tt) >= 2 else 'normal')
        plt.colorbar(im, ax=axs[-1], fraction=0.046, label='S P&L per print (bp)')
        fig.suptitle('S same-side EWMA: factor-hedged P&L per print by characteristic and side (P = bid, S = offer; * = |FM t| >= 2)', fontsize=11); plt.tight_layout(); savefig('13_pnl_by_side_heatmap')

    # (3) the cells that carry the dollars
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    _tc = cells_pnl['$/mo S ($k)']; _show = pd.concat([_tc.head(10), _tc.tail(5)]); _show = _show[~_show.index.duplicated()][::-1]
    ax[0].barh(np.arange(len(_show)), _show.to_numpy(float), color=[GREEN if x_ >= 0 else RED for x_ in _show]); ax[0].set_yticks(np.arange(len(_show))); ax[0].set_yticklabels([f'{i_}  [{n_:,} prints]' for i_, n_ in zip(_show.index, cells_pnl.loc[_show.index, 'n'])], fontsize=8)
    ax[0].axvline(0, color=INK, lw=0.6); ax[0].axhline(min(4.5, len(_show) - 10.5) if len(_show) > 10 else -1, color='#888888', lw=0.8, ls='--'); ax[0].set_title('S: ten best and five worst duration x call x rating cells by dollars per month ($k)', fontsize=10)
    _cum = _tc.clip(lower=0).sort_values(ascending=False).cumsum() / max(_tc.clip(lower=0).sum(), 1e-9)
    ax[1].plot(np.arange(1, len(_cum) + 1), _cum.to_numpy(float), marker='.', color='#4C72B0'); ax[1].axhline(0.8, color='#888888', lw=0.8, ls='--'); ax[1].set_xlabel('cells, ranked by S dollars'); ax[1].set_ylabel('cumulative share of positive S dollars'); ax[1].set_title('Concentration: how many cells carry the P&L', fontsize=10); ax[1].grid(alpha=0.3)
    plt.tight_layout(); savefig('13_pnl_cells')
    # v4.3: the beta-space map of Section 6b coloured by markout: where in factor space the money is
    if 'sm' in globals() and 'v_S' in bk.columns:
        _bl = bk.groupby('cusip', observed=True).agg(pnl_S=('v_S', 'mean'), fill_S=('f_S', 'mean'), prints=('v_S', 'size')).reset_index(); _bl['cusip'] = _bl['cusip'].astype('string')
        smm = sm.assign(cusip=sm['cusip'].astype('string')).merge(_bl, on='cusip', how='left')
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
        for a, (col, ttl, cmap, lim) in zip(ax, [('pnl_S', 'S factor-hedged P&L per print (bp), bond mean', 'RdYlGn', (-15, 15)), ('fill_S', 'S fill share, bond mean', 'viridis', (0, 1)), ('prints', 'customer prints in the held-out months (log10)', 'magma', None)]):
            a.scatter(smm['x'], smm['y'], s=4, color='#DDDDDD', alpha=0.5); g0 = smm.dropna(subset=[col])
            vals = np.log10(g0[col].clip(lower=1)) if col == 'prints' else g0[col]
            sc = a.scatter(g0['x'], g0['y'], s=7, c=vals, cmap=cmap, alpha=0.85, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None); plt.colorbar(sc, ax=a, fraction=0.046)
            a.set_title(f'Beta space ({emb_name}): {ttl}', fontsize=10); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle(f'Where in factor space the P&L lives ({int(smm["pnl_S"].notna().sum()):,} of {len(smm):,} mapped bonds have customer prints; grey = none)', fontsize=11); plt.tight_layout(); savefig('13_beta_space_markout')
    record('markout_breakdown', record_h=HR, segments={k_: v_.round(4).reset_index().to_dict(orient='records') for k_, v_ in seg_pnl.items()}, cells=cells_pnl.round(4).reset_index().to_dict(orient='records'),
           by_side={k_: {m_: v_[m_].round(4).reset_index().to_dict(orient='records') for m_ in ['pnl', 't', '$', 'fill']} for k_, v_ in side_tabs.items()})
else:
    print('Section 13 skipped: needs Section 11.')

# %% [markdown]
# ## 14. Results registry and summary

# %%
CELL_T('14. Results registry and summary [1]')
score = resid[['cusip', 'date', 'gamma_version', 'fold', 'target_bp', 'fitted_bp', 'pit_residual', 'activity_bucket'] + beta_cols + [c for c in ['ssm_signal', 'ssm_drift', 'ssm_mark_noise'] if c in resid.columns]].copy()
score['pit_residual_rank'] = score.groupby('date', observed=True)['pit_residual'].rank(pct=True)
if 'ssm_signal' in score.columns:
    score['ssm_signal_rank'] = score.groupby('date', observed=True)['ssm_signal'].rank(pct=True)
score['signal_available_date'] = score['date'] + pd.offsets.BDay(1)
score['spec_version'] = CFG.spec_version
score.to_parquet(ARTIFACTS / 'pit_residual_score_v3.parquet', index=False)
(ARTIFACTS / 'results_registry.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')

summary_rows = [
    ('Model panel', f"{REGISTRY['model_panel']['rows']:,} rows, {REGISTRY['model_panel']['cusips']:,} CUSIPs, {REGISTRY['model_panel']['dates']} dates, L={len(CHARS)}"),
    ('Data rules', f"{len(PARTIAL_DAYS)} partial-mark days dropped; {len(DISPERSION_DAYS)} dispersion days zero-weighted; {len(COMMON_MOVE_DAYS)} common-move days kept"),
    ('Refit cadence (OOS var explained, ALL)', ' | '.join(f"{k} {v:.3f}" for k, v in cadence.loc['ALL'].items()) + f"; record = {CFG.record_cadence}"),
    ('Residuals of record', f"OOS variance explained {REGISTRY['walk_forward']['oos_variance_explained']:.3f}; residual sd {REGISTRY['walk_forward']['residual_sd_bp']:.2f} bp; {REGISTRY['walk_forward']['gamma_versions']} Gamma versions; aligned loading sd {REGISTRY['walk_forward']['gamma_loading_sd_across_versions']:.3f}"),
    ('Factor variance shares (full sample)', ', '.join(f'{s:.1%}' for s in REGISTRY['ipca_full_sample']['factor_variance_share'])),
    ('Residual ACF lag 1 Pearson / Spearman', f"{acf.loc[1, 'pearson']:+.3f} / {acf.loc[1, 'spearman']:+.3f} (ex dispersion days {acf.loc[1, 'pearson_ex_dispersion']:+.3f})"),
]
for nm in fc_table.index:
    summary_rows.append((f'Forecaster: {nm}', f"rank IC {fc_table.loc[nm, 'ic_mean']:+.3f} (t {fc_table.loc[nm, 'ic_t']:+.2f}); sign acc {fc_table.loc[nm, 'sign_acc']:.3f}"))
if not ssmb_params.empty:
    summary_rows.append(('State-space parameters by bucket (last fold)', '; '.join(f"{g}: phi {r['phi']:.2f}, drift/noise {r['signal_to_noise']:.2f}" for g, r in lastp.iterrows() if g != 'UNKNOWN')))
if 'risk_model' in REGISTRY:
    _rk = pd.DataFrame(REGISTRY['risk_model']['by_cluster']).set_index('cluster')
    summary_rows.append(('Risk model snapshot (median systematic share of daily variance; systematic / idio sd bp)', f"{_rk['systematic_share'].median():.0%}; {_rk['systematic_sd_bp'].median():.2f} / {_rk['idio_sd_bp'].median():.2f} bp over {REGISTRY['risk_model']['bonds']:,} bonds"))
if HAS_TRADES and not tv.empty:
    _r = tx[(tx['score'] == 'state-space signal rank (record)') & (tx['bucket'] == 'ALL') & tx['target'].isin(['trade - prior mark', 'trade - algo quote'])].set_index('target')
    summary_rows.append(('Record signal vs prior mark | vs algo quote', f"FM {_r.loc['trade - prior mark', 'fm_beta']:+.2f} bp/rank (t {_r.loc['trade - prior mark', 'fm_t']:+.2f}, boot {_r.loc['trade - prior mark', 'boot_t']:+.2f}) | {_r.loc['trade - algo quote', 'fm_beta']:+.2f} (t {_r.loc['trade - algo quote', 'fm_t']:+.2f}, boot {_r.loc['trade - algo quote', 'boot_t']:+.2f})"))
    _mq = spread_tx.set_index('test')
    summary_rows.append(('Mark-quality: |trade - mark| on |resid|/vol | on state-space |eta|', ' | '.join(f"{_mq.loc[i, 'fm_beta']:+.2f} (t {_mq.loc[i, 'fm_t']:+.1f})" for i in _mq.index[:2])))
if 'error_correction' in REGISTRY:
    _p = pers.set_index('score'); _o = ec_all.iloc[0]
    summary_rows.append(('Algo error persistence print to print', f"{_p.loc['e_algo_bp_0', 'fm_beta']:+.3f} (t {_p.loc['e_algo_bp_0', 'fm_t']:+.1f}, boot {_p.loc['e_algo_bp_0', 'boot_t']:+.1f})"))
    summary_rows.append(('Held-out MAE vs print: ' + ' | '.join(k.split()[0] for k in PREDS), ' | '.join(f"{_o[f'MAE {k.split()[0]}']:.2f}" for k in PREDS) + ' bp'))
    if 'last_value_diagnostic' in REGISTRY['error_correction']:
        _lvd = pd.DataFrame(REGISTRY['error_correction']['last_value_diagnostic']).set_index('bucket')
        summary_rows.append(("Algo error on its own last-value adjustment (ALL)", f"{_lvd.loc['ALL', 'fm_beta']:+.3f} (t {_lvd.loc['ALL', 'fm_t']:+.1f}); corr(same-side memory, adjustment) {REGISTRY['error_correction']['corr_side_err_last_value']:+.3f}"))
    summary_rows.append(('Gain over algo (bp, daily FM t, bootstrap t)', '; '.join(f"{k.split()[0]}: {_o[f'gain {k.split()[0]} (bp)']:+.2f} (t {_o[f't {k.split()[0]}']:+.1f}, boot {boot_gain[k]:+.1f})" for k in list(PREDS)[1:])))
    if 'bar' in globals() and len(bar):
        summary_rows.append(('Interaction ladder vs S (delta bp, boot t, passes bar)', '; '.join(f"{i.split()[0]}: {r['delta vs S (bp)']:+.2f} (boot {r['boot t of delta']:+.1f}) {'PASS' if r['passes bar'] else 'no'}" for i, r in bar.iterrows())))
    if 'direction' in globals():
        _dS = direction.loc['S same-side EWMA']
        summary_rows.append(('Direction of the S correction (share toward | crossed closer | crossed further | away; asym-loss gain)', f"{_dS['share toward']:.0%} | {_dS['share crossed, closer']:.0%} | {_dS['share crossed, further']:.0%} | {_dS['share away']:.0%}; {_dS['asym loss gain (bp)']:+.2f} bp (t {_dS['asym FM t']:+.1f})"))
if 'conformal_band' in REGISTRY:
    for mid_name, r in REGISTRY['conformal_band']['by_mid'].items():
        summary_rows.append((f'Band around {mid_name} ({CFG.band_coverage:.0%} target)', f"fixed {r['coverage fixed']:.1%} at {r['half-width fixed (bp)']:.1f} bp | split conformal {r['coverage conformal']:.1%} at {r['half-width conformal (bp)']:.1f} | rolling conformal {r['coverage rolling']:.1%} at {r['half-width rolling (bp)']:.1f}"))
if 'error_correction' in REGISTRY and 'ecp' in globals():
    _ex = ecp[ecp['modified_duration_lag1'] >= 1.0]
    summary_rows.append(('Headline MAE excluding duration < 1y: algo | C | S | SQ | K | D', ' | '.join(f"{_ex[c].abs().mean():.2f}" for c in ['e_algo_bp', 'e_C_bp', 'e_S_bp', 'e_SQ_bp', 'e_K_bp', 'e_D_bp']) + f' bp ({len(_ex):,} trades, {len(_ex) / len(ecp):.0%})'))
if 'markout' in REGISTRY:
    _rr = pd.DataFrame(REGISTRY['markout']['rerank']).set_index('mid'); _hr = REGISTRY['markout']['record_h']
    summary_rows.append((f'Markout $ per month at h={_hr} (would-have-filled, upper bound): algo | S | SQ | AQ | D', ' | '.join(f"{_rr.loc[k_, '$ per month ($k)']:,.0f}" for k_ in ['A algo quote', 'S same-side EWMA', 'SQ same-side EWMA + side x size intercept', 'AQ algo + side x size intercept', 'D level model with error features'] if k_ in _rr.index) + ' $k'))
    summary_rows.append((f'Markout P&L per print at h={_hr} (bp): algo | S | SQ | AQ | D', ' | '.join(f"{_rr.loc[k_, 'P&L per print (bp)']:+.2f}" for k_ in ['A algo quote', 'S same-side EWMA', 'SQ same-side EWMA + side x size intercept', 'AQ algo + side x size intercept', 'D level model with error features'] if k_ in _rr.index)))
    _tcx = pd.DataFrame(REGISTRY['markout']['trend_check']); _tcs = _tcx[_tcx['mid'] == 'S same-side EWMA'].set_index('side')
    summary_rows.append((f'Trend check, S at h={_hr}: P&L per print raw | date-demeaned | factor-hedged, P then S (bp)', '; '.join(f"{s_}: {_tcs.loc[s_, f'P&L per print, raw mark move (bp)']:+.1f} | {_tcs.loc[s_, f'P&L per print, date-demeaned move (bp)']:+.1f} | {_tcs.loc[s_, f'P&L per print, factor-hedged residual move (bp)']:+.1f}" for s_ in ['P', 'S'] if s_ in _tcs.index)))
    summary_rows.append(('Markout winner ($ rank 1) and whether it clears the bar vs S', f"{_rr.index[0]}: {_rr.iloc[0]['$ per month ($k)']:,.0f} $k/month, MAE rank {int(_rr.iloc[0]['MAE rank'])}, boot t vs S {_rr.iloc[0]['boot t vs S']:+.1f}, bar {'PASS' if bool(_rr.iloc[0]['beats S (bar)']) else 'no'}"))
    _ad = pd.DataFrame(REGISTRY['markout']['adverse_selection'])
    _adf = _ad[_ad['prints'] == 'would have filled'].set_index('side')
    summary_rows.append((f'Adverse selection on would-be fills (algo quote): edge at quote | factor-implied | residual move after {_hr}d, P / S (bp)', ' / '.join(f"{_adf.loc[s_, 'edge at our quote (bp)']:+.1f} | {_adf.loc[s_, f'factor-implied, h={_hr} (bp)']:+.1f} | {_adf.loc[s_, f'residual move, h={_hr} (bp)']:+.1f}" for s_ in _adf.index)))
if 'trade_markout' in REGISTRY and REGISTRY['trade_markout']['rerank']:
    _tr = pd.DataFrame(REGISTRY['trade_markout']['rerank']).set_index('mid'); _sp = REGISTRY['trade_markout']['spearman_vs_evaluation']
    summary_rows.append(('Marked to the next real print (any side, mid-equivalent): P&L per print algo | S | SQ | F | D (bp); winner; Spearman with the evaluation ranking ($ | per print)', ' | '.join(f"{_tr.loc[k_, 'P&L per print (bp)']:+.2f}" for k_ in ['A algo quote', 'S same-side EWMA', 'SQ same-side EWMA + side x size intercept', 'F side intercept + rho x own EWMA', 'D level model with error features'] if k_ in _tr.index) + f"; {_tr.index[0]} ({_tr.iloc[0]['P&L per print (bp)']:+.2f}, bar {'PASS' if bool(_tr.iloc[0]['beats S (bar)']) else 'no'}); {_sp['dollars']:+.2f} | {_sp['per_print']:+.2f}"))
    _cv = pd.DataFrame(REGISTRY['trade_markout']['coverage']).set_index('index') if 'index' in pd.DataFrame(REGISTRY['trade_markout']['coverage']).columns else pd.DataFrame(REGISTRY['trade_markout']['coverage'])
    _hz = pd.DataFrame(REGISTRY['trade_markout']['by_horizon']).set_index(['mid', 'mark'])
    if ('S same-side EWMA', 'any side, mid-equivalent') in _hz.index:
        _r = _hz.loc[('S same-side EWMA', 'any side, mid-equivalent')]
        summary_rows.append(('S on real-trade marks by horizon: next print | 1 bd | 5 bd | 10 bd (bp per print); inter-dealer next | opposite next', ' | '.join(f"{_r[c_]:+.2f}" for c_ in ['next print', 'last print within 1 bd', 'last print within 5 bd', 'last print within 10 bd'] if c_ in _r.index) + f"; {_hz.loc[('S same-side EWMA', 'inter-dealer only'), 'next print']:+.2f} | {_hz.loc[('S same-side EWMA', 'opposite side (round trip)'), 'next print']:+.2f}"))
    _nz = pd.DataFrame(REGISTRY['trade_markout']['noise']).set_index('mark')
    summary_rows.append(('Noise per S fill, sd (bp): evaluation h=5 | next print any side | next inter-dealer | next opposite side', ' | '.join(f"{_nz.loc[k_, 'sd of P&L per fill (bp)']:.1f}" for k_ in ['evaluation, h = 5 (hedged)', 'next print, any side (mid-equivalent)', 'next inter-dealer print', 'next opposite-side print (round trip)'] if k_ in _nz.index)))
if 'markout_breakdown' in REGISTRY:
    _mc = pd.DataFrame(REGISTRY['markout_breakdown']['cells']).set_index('cell')
    if len(_mc):
        summary_rows.append(('Markout cells (S): best and worst duration x call x rating by dollars per month', f"best {_mc.index[0]}: {_mc.iloc[0]['$/mo S ($k)']:,.0f} $k ({_mc.iloc[0]['P&L/print S (bp)']:+.1f} bp/print); worst {_mc.index[-1]}: {_mc.iloc[-1]['$/mo S ($k)']:,.0f} $k ({_mc.iloc[-1]['P&L/print S (bp)']:+.1f} bp/print)"))
    _ms = REGISTRY['markout_breakdown']['segments']
    if 'trade size' in _ms:
        _mq = pd.DataFrame(_ms['trade size']).set_index('qty_group')
        summary_rows.append(('Markout by trade size (S P&L per print, bp): smallest bin -> largest bin', ' -> '.join(f"{_mq.loc[q_, 'P&L/print S (bp)']:+.1f}" for q_ in QTY_LABELS if q_ in _mq.index)))
    _cc = pd.DataFrame(REGISTRY['markout']['concession']).set_index('index') if 'index' in pd.DataFrame(REGISTRY['markout']['concession']).columns else pd.DataFrame(REGISTRY['markout']['concession'])
    if 'round_trip' in REGISTRY['markout']:
        _rtr = REGISTRY['markout']['round_trip']; _ss = pd.DataFrame(_rtr['side_split']).set_index('mid')
        summary_rows.append((f"Round trip ({_rtr['coverage']:.0%} of prints have an exit, median {_rtr['median_days_to_exit']:.1f} days): S P&L per print bid | offer, round trip vs evaluation on the same prints (bp)", f"{_ss.loc['S same-side EWMA', 'round trip, bid (P)']:+.1f} vs {_ss.loc['S same-side EWMA', 'evaluation markout, bid, same prints']:+.1f} | {_ss.loc['S same-side EWMA', 'round trip, offer (S)']:+.1f} vs {_ss.loc['S same-side EWMA', 'evaluation markout, offer, same prints']:+.1f}"))
        _rc = pd.DataFrame(_rtr['concession']).set_index('index') if 'index' in pd.DataFrame(_rtr['concession']).columns else pd.DataFrame(_rtr['concession'])
        summary_rows.append(('Side concession on the round trip: S at 0 | delta* from evaluation | delta* from round trip (bp per print; boot t)', ' | '.join(f"{_rc.loc[k_, 'P&L per print (bp)']:+.2f}" + (f" (t {_rc.loc[k_, 'boot t vs S at delta 0']:+.1f})" if np.isfinite(_rc.loc[k_, 'boot t vs S at delta 0']) else '') for k_ in _rc.index)))
    summary_rows.append(('Cell-optimal concession, walk-forward: S at 0 | S + delta* per side | S + delta* per side x size ($k/month)', ' | '.join(f"{_cc.loc[k_, '$ per month ($k)']:,.0f}" for k_ in ['S same-side EWMA, delta 0', 'S + delta* per side (walk-forward)', 'S + delta* per side x size cell (walk-forward)'] if k_ in _cc.index)))
if 'pfill' in REGISTRY and REGISTRY['pfill']['engine']:
    _br = pd.DataFrame(REGISTRY['pfill']['brier_relative_to_desk']).set_index('method')
    summary_rows.append(('Fill model for the S quote: Brier removed vs the desk curve at delta 0 (production-style key | cluster | cluster + deviance shift | GBM trade | GBM + IPCA)', ' | '.join(f"{_br.loc[m_, '0']:+.1%}" if '0' in _br.columns else f"{_br.loc[m_, 0.0]:+.1%}" for m_ in ['cell: side x size x production-style key', 'cell: side x size x cluster, prior months', 'cell: cluster, shifted by residual deviance', 'GBM: trade and quote features', 'GBM: + IPCA and state-space'] if m_ in _br.index)))
    _dc = pd.DataFrame(REGISTRY['pfill']['desk_calibration_by_size']).set_index('qty_group')
    _gaps = [abs(_dc.loc[q_, f'realised | {s_}'] - _dc.loc[q_, f'predicted | {s_}']) for q_ in _dc.index for s_ in ['P', 'S'] if f'realised | {s_}' in _dc.columns and np.isfinite(_dc.loc[q_, f'realised | {s_}'])]
    if _gaps:
        summary_rows.append(('Desk curve on the algo quote: largest calibration gap by size bin at delta 0 (fill probability)', f'{max(_gaps):.3f}'))
    if REGISTRY['pfill']['edge_model']:
        _em = pd.DataFrame(REGISTRY['pfill']['edge_model']).set_index(['model', 'side'])
        _k = ('round-trip edge, + IPCA / state-space', 'ALL')
        if _k in _em.index:
            summary_rows.append(('Edge model (round-trip edge of S fills, + IPCA): OOS R2 | daily rank IC (t)', f"{_em.loc[_k, 'OOS R2']:.3f} | {_em.loc[_k, 'daily rank IC']:+.3f} (t {_em.loc[_k, 'IC t']:+.1f})"))
    _en = pd.DataFrame(REGISTRY['pfill']['engine']).set_index('policy')
    summary_rows.append(('Engine in reduced form vs S at 0, round-trip P&L per print (bp; boot t; bar): desk | cell | GBM + IPCA', ' | '.join(f"{_en.loc[p_, 'round trip: P&L per print (bp)']:+.2f} (t {_en.loc[p_, 'round trip: boot t vs S at 0']:+.1f}) {'PASS' if bool(_en.loc[p_, 'beats S (bar, round trip)']) else 'no'}" for p_ in ['engine: desk pfill x edge model', 'engine: cell pfill x edge model', 'engine: GBM + IPCA pfill x edge model'] if p_ in _en.index) + f"; S at 0 {_en.loc['S at 0', 'round trip: P&L per print (bp)']:+.2f}"))
if 'production_objective' in REGISTRY and REGISTRY['production_objective']['policies']:
    _po = pd.DataFrame(REGISTRY['production_objective']['policies']).set_index('policy'); _pp = 'production: A, desk curve, x + charge'
    summary_rows.append((f"Production objective (charges: {REGISTRY['production_objective']['charge_source'].split(',')[0]}, {REGISTRY['production_objective']['charge_units']}): expected | realised round trip per print (bp); mean x*; fill share", f"{_po.loc[_pp, 'objective value per print (expected)']:+.2f} | {_po.loc[_pp, 'round trip: P&L per print (bp)']:+.2f}; {_po.loc[_pp, 'mean x* (bp)']:.1f} bp; {_po.loc[_pp, 'fill share']:.0%}"))
    _sw = pd.DataFrame(REGISTRY['production_objective']['swaps']).set_index('swap')
    summary_rows.append(('Replacing one part of the production objective (delta round-trip P&L per print vs production, boot t): mid | fill curve | value of a fill | all three', ' | '.join(f"{r_['delta RT P&L per print (bp)']:+.2f} (t {r_['boot t']:+.1f})" for _, r_ in _sw.iterrows())))
    if REGISTRY['production_objective'].get('by_side'):
        _bsr = pd.DataFrame(REGISTRY['production_objective']['by_side']).set_index(['policy', 'side'])
        for _pn, _lab in [(_pp, 'production (algo yield)'), ('A, GBM + IPCA fill model, x + charge', 'fill-curve swap'), ('S, GBM + IPCA fill model, expected P&L (the Section 11b engine)', 'all three')]:
            if (_pn, 'P') in _bsr.index and (_pn, 'S') in _bsr.index:
                summary_rows.append((f'{_lab}: win rate bid | offer; |quote - print| bid | offer (bp); RT P&L per print bid | offer (bp)', f"{_bsr.loc[(_pn, 'P'), 'win rate']:.0%} | {_bsr.loc[(_pn, 'S'), 'win rate']:.0%}; {_bsr.loc[(_pn, 'P'), '|quote - print| (bp)']:.1f} | {_bsr.loc[(_pn, 'S'), '|quote - print| (bp)']:.1f}; {_bsr.loc[(_pn, 'P'), 'RT P&L per print (bp)']:+.2f} | {_bsr.loc[(_pn, 'S'), 'RT P&L per print (bp)']:+.2f}"))
    _win = _po[_po['beats production (bar, round trip)'].astype(bool)]
    summary_rows.append(('Variants that clear the bar against production (round trip)', '; '.join(f"{i_}: {r_['round trip: P&L per print (bp)']:+.2f} bp" for i_, r_ in _win.iterrows()) if len(_win) else 'none'))
if 'rfq_ledger' in REGISTRY and REGISTRY['rfq_ledger'].get('policies'):
    _rl = REGISTRY['rfq_ledger']; _rp = pd.DataFrame(_rl['policies']).set_index('policy'); _pn_ = 'production (logged optimal yield)'
    summary_rows.append(('RFQ ledger: requests | printed | joined | scored (months)', f"{_rl['requests']:,} | {_rl['printed']:,} | {_rl['joined']:,} | {_rl['scored']:,} ({_rl['months']})"))
    _gr = pd.DataFrame(_rl['grouping']).set_index('estimator')
    if 'logged pfill (production)' in _gr.index:
        summary_rows.append(('Logged pfill at production\'s quote: mean predicted | realised | Brier', f"{_gr.loc['logged pfill (production)', 'mean predicted']:.3f} | {_gr.loc['logged pfill (production)', 'realised']:.3f} | {_gr.loc['logged pfill (production)', 'Brier']:.4f}"))
    summary_rows.append(('Grouping test at production\'s quote: Brier removed vs logged pfill (desk | production-style key | cluster | cluster + deviance | GBM + IPCA); bar', ' | '.join(f"{_gr.loc[k_, 'Brier removed vs logged']:+.1%}{' PASS' if bool(_gr.loc[k_, 'passes bar']) else ''}" for k_ in ['desk curve (side, trailing window)', 'cell: side x size x production-style key', 'cell: side x size x cluster', 'cell: cluster, shifted by residual deviance', 'GBM: + IPCA and state-space'] if k_ in _gr.index)))
    summary_rows.append(('On the requests: production | S at 0 | engine: win rate; |quote - print| bp; RT P&L per request bp (boot t)', ' | '.join(f"{_rp.loc[k_, 'win rate (print crossed)']:.0%}; {_rp.loc[k_, '|quote - print| (bp)']:.1f}; {_rp.loc[k_, 'RT P&L per request (bp)']:+.2f}" + (f" (t {_rp.loc[k_, 'RT P&L boot t vs production']:+.1f})" if k_ != _pn_ and np.isfinite(_rp.loc[k_, 'RT P&L boot t vs production']) else '') for k_ in [_pn_, 'S at 0', 'engine: S, GBM + IPCA, expected P&L (12c)'] if k_ in _rp.index)))
    _pv_ = pd.DataFrame(_rl['proxy_validation']).set_index('representation')
    summary_rows.append(('Production representation vs the logged concession on the overlap, MAE bp (proxy | 12c replica)', ' | '.join(f"{_pv_.loc[k_, 'MAE vs logged x_prod (bp)']:.2f}" for k_ in _pv_.index)))
    _uv = pd.DataFrame(_rl['universe']).set_index('policy')
    _c_all = 'RT P&L per print (bp) | ALL'
    if _c_all in _uv.columns:
        summary_rows.append(('Universe with the production proxy: proxy | replica | S at 0 | engine, RT P&L per print (bp)', ' | '.join(f"{_uv.loc[k_, _c_all]:+.2f}" for k_ in ['production proxy: A + median x_prod by key', 'production replica (12c): A, desk curve, x + charge', 'S at 0', 'engine: S, GBM + IPCA, expected P&L'] if k_ in _uv.index)))
if 'residual_side' in REGISTRY:
    _sp = pd.DataFrame(REGISTRY['residual_side']['policy']).set_index('input'); _sl = pd.DataFrame(REGISTRY['residual_side']['slopes'])
    summary_rows.append(('Residual signal as concession modifier (delta bp per print over S + side concession, boot t, bar)', '; '.join(f"{i_}: {r_['delta (bp)']:+.2f} (t {r_['boot t of delta']:+.1f}) {'PASS' if r_['passes bar'] else 'no'}" for i_, r_ in _sp.iterrows())))
    summary_rows.append(('FM slope of the hedged move on the oriented forecast, full forecast P / S (bp per bp, t)', ' / '.join(f"{r_['fm_beta']:+.2f} (t {r_['fm_t']:+.1f})" for _, r_ in _sl[_sl['input'] == 'full forecast'].iterrows())))
    _ws = REGISTRY['residual_side']['width_scalar']
    if _ws['by_side']:
        _wsd = pd.DataFrame(_ws['by_side']).set_index('side')
        summary_rows.append(('Mark-noise width scalar on the grid (tail calibration error pooled -> split, P / S; adopted?)', ' / '.join(f"{_wsd.loc[s_, 'pooled']:.3f} -> {_wsd.loc[s_, 'split by mark noise']:.3f}" for s_ in _wsd.index) + f"; {'ADOPT' if _ws['adopt'] else 'keep pooled'}"))
if 'quantile_grid' in REGISTRY and REGISTRY['quantile_grid']['calibration']:
    _cg = pd.DataFrame(REGISTRY['quantile_grid']['calibration']).set_index('side')
    summary_rows.append(('Quantile grid calibration (realised share below q at tau 0.10 / 0.50 / 0.90, P then S)', '; '.join(f"{s_}: {_cg.loc[s_, 'tau 0.10']:.2f} / {_cg.loc[s_, 'tau 0.50']:.2f} / {_cg.loc[s_, 'tau 0.90']:.2f}" for s_ in _cg.index)))
if 'conformal_band' in REGISTRY and 'efficiency' in REGISTRY['conformal_band']:
    _ef = pd.DataFrame(REGISTRY['conformal_band']['efficiency']); _efD = _ef[_ef['mid'] == 'level model D'].set_index('band')
    summary_rows.append(('Band efficiency, level model D mid (half-width bp per coverage point)', '; '.join(f"{b}: {_efD.loc[b, 'width per point of coverage']:.3f} ({_efD.loc[b, 'coverage']:.0%} at {_efD.loc[b, 'half-width (bp)']:.1f} bp)" for b in ['rolling_fixed', 'rolling'] if b in _efD.index)))
if 'dollars' in REGISTRY:
    _d = REGISTRY['dollars']['overall'][0]
    summary_rows.append(('Dollar error removed per month on held-out prints ($k): C | S | K | D', ' | '.join(f"{_d[f'$ removed per month ($k) [{k}]']:,.0f}" for k in ['C side-pooled EWMA', 'S same-side memory', 'K per-side state-space memory', 'D level model']) + f" of {_d['algo |error| ($k)'] / max(len(REGISTRY['dollars']['by_month']), 1):,.0f} algo error per month"))
summary_rows.append(('Roll-forward RMSE reduction, h=1 / h=10', f"{rollf.loc[1, 'rmse_reduction']:.0%} / {rollf.loc[10, 'rmse_reduction']:.0%}"))
if 'factor_premia' in REGISTRY and REGISTRY['factor_premia']['long_short']:
    _fp = pd.DataFrame(REGISTRY['factor_premia']['long_short']).set_index(['freq', 'signal', 'portfolio'])
    for fq_ in CFG.premia_freqs:
        _k = (fq_, 'factor premium + carry', 'LS equal-weight (bp)'); _k2 = (fq_, 'factor premium  (-D x beta.lambda x h)', 'LS equal-weight (bp)')
        if _k in _fp.index:
            summary_rows.append((f'Factor premia long-short, {fq_}: premium + carry | premium alone (mean bp per period, t, hit rate; rebalances)', f"{_fp.loc[_k, 'mean (bp)']:+.1f} (t {_fp.loc[_k, 't']:+.1f}, hit {_fp.loc[_k, 'hit rate']:.0%})" + (f" | {_fp.loc[_k2, 'mean (bp)']:+.1f} (t {_fp.loc[_k2, 't']:+.1f}, hit {_fp.loc[_k2, 'hit rate']:.0%})" if _k2 in _fp.index else '') + f"; {int(_fp.loc[_k, 'rebalances'])} rebalances"))
    _pe = pd.DataFrame(REGISTRY['factor_premia']['premium_estimates'])
    if len(_pe):
        summary_rows.append(('Factor premium estimates with |t| >= 2 at the refits (of refit x factor)', f"{int((_pe['t'].abs() >= 2).sum())} of {len(_pe)}"))
    if REGISTRY['factor_premia']['tangency']:
        _tg = pd.DataFrame(REGISTRY['factor_premia']['tangency']).set_index('index')
        summary_rows.append(('Tangency portfolio of the factors, OOS (yield space): annualised Sharpe | equal weight | factor 1 alone | market', ' | '.join(f"{_tg.loc[k_, 'Sharpe (annualised, daily)']:+.2f}" for k_ in ['tangency (model-implied MVE)', 'equal weight of factors', 'factor1 alone (long the factor)', 'MARKET: equal-weighted universe (long all bonds)'] if k_ in _tg.index)))
        if 'beta to market' in _tg.columns and 'tangency (model-implied MVE)' in _tg.index:
            summary_rows.append(('Tangency portfolio against the market: beta | alpha bp/day (t) | information ratio | months above market', f"{_tg.loc['tangency (model-implied MVE)', 'beta to market']:+.2f} | {_tg.loc['tangency (model-implied MVE)', 'alpha (bp/day)']:+.3f} (t {_tg.loc['tangency (model-implied MVE)', 'alpha t']:+.1f}) | {_tg.loc['tangency (model-implied MVE)', 'information ratio (annualised)']:+.2f} | {_tg.loc['tangency (model-implied MVE)', 'months above market']:.0%}"))
    if REGISTRY['factor_premia'].get('benchmark'):
        _bm = pd.DataFrame(REGISTRY['factor_premia']['benchmark']).set_index(['freq', 'signal', 'portfolio'])
        for fq_ in CFG.premia_freqs:
            _k = (fq_, 'factor premium + carry', 'long leg, equal-weight (bp)'); _km = (fq_, 'MARKET: equal-weighted universe, long all bonds', 'total return (bp)')
            if _k in _bm.index and _km in _bm.index:
                summary_rows.append((f'Long leg of premium + carry vs the market, {fq_}: mean | market | excess (bp per period); beta; periods above market; down capture', f"{_bm.loc[_k, 'mean (bp)']:+.1f} | {_bm.loc[_km, 'mean (bp)']:+.1f} | {_bm.loc[_k, 'excess over market (bp)']:+.1f}; {_bm.loc[_k, 'beta to market']:+.2f}; {_bm.loc[_k, 'periods above market']:.0%}; {_bm.loc[_k, 'down capture']:.2f}"))
summary = pd.DataFrame(summary_rows, columns=['item', 'result']).set_index('item')
display(summary)
print('Artifacts written to', ARTIFACTS)
for p in sorted(ARTIFACTS.glob('*')):
    if p.is_file():
        print('  ', p.name, f'{p.stat().st_size/1e6:.2f} MB')
print('Figures:', len(list(FIGURES.glob('*.png'))))
_cl = pd.DataFrame(CACHE_LOG)
if len(_cl):
    print('Stage cache this run (reused stages skipped their computation; keys hash the config, data fingerprints and code):'); display(_cl.assign(seconds=_cl['seconds'].round(1)))
    record('stage_cache', log=_cl.round(2).to_dict(orient='records'), keys=STAGE_KEYS)

# %% [markdown]
# ## 15. Robustness: dispersion-day exclusions, bootstrap inference, pre-registered choices

# %%
CELL_T('15. Robustness: dispersion-day exclusions, bootstrap inferen [1]')
rob_rows = [{'check': 'Residual ACF lag 1 (Pearson)', 'all days': acf.loc[1, 'pearson'], 'ex dispersion days': acf.loc[1, 'pearson_ex_dispersion']},
            {'check': 'Residual ACF lag 1 (Spearman)', 'all days': acf.loc[1, 'spearman'], 'ex dispersion days': acf.loc[1, 'spearman_ex_dispersion']}]
for nm in fc_table.index:
    rob_rows.append({'check': f'Rank IC: {nm}', 'all days': fc_table.loc[nm, 'ic_mean'], 'ex dispersion days': fc_table.loc[nm, 'ic_ex_dispersion']})
_tail = two_regime.iloc[[0, -1]]
rob_rows.append({'check': 'Two-regime: continuation ratio, cheap tail', 'all days': _tail['continuation_ratio'].iloc[-1], 'ex dispersion days': _tail['continuation_ex_dispersion'].iloc[-1]})
rob_rows.append({'check': 'Two-regime: continuation ratio, rich tail', 'all days': _tail['continuation_ratio'].iloc[0], 'ex dispersion days': _tail['continuation_ex_dispersion'].iloc[0]})
robust = pd.DataFrame(rob_rows).set_index('check')
print('Residual results with and without dispersion days:'); display(robust.round(4))
if HAS_TRADES and not tv.empty:
    key = tx[tx['bucket'] == 'ALL'][['score', 'target', 'fm_beta', 'fm_t', 'boot_t', 'fm_dates']].copy(); key['t ratio (boot / FM)'] = key['boot_t'] / key['fm_t']
    print('Fama-MacBeth t vs moving-block bootstrap t (5-day blocks):'); display(key.round(3))
    record('robustness', fm_vs_bootstrap=key.round(4).to_dict(orient='records'))
print('Pre-registered choices (fixed before this run):'); display(pd.Series(PRE_REGISTERED).to_frame('value'))
record('robustness', residual_checks=robust.round(4).reset_index().to_dict(orient='records'), pre_registered=PRE_REGISTERED)
# ---- per-cell runtimes (v4.5): this run against the previous one, with the cached stages each cell touched
CELL_T('end')
ct = pd.DataFrame(CELL_LOG); ct['share'] = ct['seconds'] / ct['seconds'].sum(); ct['order'] = np.arange(len(ct))
_rt_path = ARTIFACTS / 'cell_runtimes.csv'
if _rt_path.exists():
    try:
        _prev = pd.read_csv(_rt_path)[['cell', 'seconds']].drop_duplicates('cell').rename(columns={'seconds': 'previous run (s)'}); ct = ct.merge(_prev, on='cell', how='left'); ct['delta (s)'] = ct['seconds'] - ct['previous run (s)']
    except Exception as exc:  # noqa: BLE001
        print('previous runtimes unreadable:', exc)
if CACHE_LOG:
    _cl = pd.DataFrame(CACHE_LOG); ct['cached stages'] = ct['cell'].map(_cl.groupby('cell').apply(lambda g: ', '.join(f'{r.stage} ({r.status}, {r.seconds:.0f}s)' for r in g.itertuples()), include_groups=False)).fillna('')
ct.to_csv(_rt_path, index=False)
_tot = ct['seconds'].sum(); _top = ct.sort_values('seconds', ascending=False).head(15)
print(f'Runtime: {_tot / 60:.1f} min over {len(ct)} cells; the 15 slowest carry {_top["seconds"].sum() / _tot:.0%} of it. Cells without a cached stage that recur near the top are the next caching candidates:')
display(_top.set_index('cell')[[c for c in ['seconds', 'previous run (s)', 'delta (s)', 'share', 'cached stages'] if c in ct.columns]].round(1))
record('runtime', total_seconds=float(_tot), cells=ct.drop(columns=['order']).round(2).to_dict(orient='records'))
(ARTIFACTS / 'results_registry.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')
print('registry updated:', ARTIFACTS / 'results_registry.json')
