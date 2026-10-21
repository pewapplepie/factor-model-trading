# muni_ipca_factor_lab.py: lab v4.8. Cells 1 to 15 reconstructed from the v4 run (muni_ipca_factor_lab_v4.html; markdown re-derived from the rendered cells, code verbatim);
# Sections 14d to 14g are new. Percent format; regenerate the .ipynb with the converter in docs/ipca_risk_model_notes.md.

# %% [markdown]
# # Muni IPCA factor lab: the factor model from data to factor premia
#
# Sections 1-8b of `muni_ipca_research_v3_latest.ipynb`, lifted verbatim; Sections 9-14 are the lab's own research so the factor model can be explored on its own,
# without the quote, markout and engine sections (9-15) that sit downstream of it.
#
# | Section | What it does |
# |---|---|
# | 1 | Setup: `RunConfig`, pre-registered choices, stage cache, registry |
# | 2 | Data load and QA: the evaluation panel, ratings, extreme yields |
# | 3 | Model panel: instruments (characteristics, ranked cross-sectionally), the excess-over-MMD target |
# | 4 | IPCA: K sweep, Gamma anatomy, factor paths, the Kelly-Palhares-Pruitt fit table |
# | 5 | Walk-forward residuals: refit cadence, Procrustes-aligned Gamma, residuals of record, the dependent-variable test |
# | 6 / 6b | Factor-mimicking weights; beta-space map, clusters, within-cluster ranking |
# | 7 / 7b | Residual dynamics; the state-space signal of record |
# | 8 / 8b | Risk model snapshot; factor premia as expected returns (KPP test on a bond return target) |
# | 9 | Call structure: the priced-to-call residual cluster; call characteristics, a restricted fourth factor, a blended-tenor hedge |
# | 10 | Conditional quantiles in beta space: the quantile grid as a model (time-series vs cross-sectional ablation) |
# | 11 | Relative value: IPCA peers, a permanent-temporary state space of the peer gap, frozen-basket and print-confirmed validation |
# | 12 | Residual dependence after the factors: a factor-structured vine copula (global, peer, bond), tail dependence and the conditional PIT |
# | 13 | RFQ P&L: the state-space, level and relative-value signals on the desk's requests; selection, a quote skew, a holding rule (`docs/rfq_*.md`) |
# | 14 | RFQ backtest: the production optimizer replayed on every bid request, then variants against it (`docs/rfq_backtest_framework.html`) |
# | 15 | Registry and runtimes |
#
# **Spec.** The record spec `ipca_v3_k3_xmmd` (K = 3, excess-over-MMD target, monthly refit). Change `spec_version`
# together with any `RunConfig` field you experiment with, so the stage cache and the registry stay labelled.
#
# **Isolation.** Everything this notebook writes goes to `./artifacts_ipca_lab/` (its own stage cache, figures,
# residuals, Gamma versions, registry). It never touches `./artifacts_v3/`, so experiments here cannot overwrite or
# evict the research notebook's caches and results. The first run recomputes every stage; later runs hit the cache.
#
# **Downstream contract.** Sections 9-15 of the research notebook consume `resid` (point-in-time residuals of record,
# `pit_residual`, `fitted_bp`, `curve_offset_bp`), the betas and clusters of 6b, the state-space signal of 7b and the
# risk snapshot of 8. Keep those names and columns if a change here is meant to flow back.

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
ARTIFACTS = ROOT / 'artifacts_ipca_lab'   # isolated from the research notebook's artifacts_v3
FIGURES = ARTIFACTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PIPELINE_ROOT))
import importlib  # noqa: E402
import muni_data_pipeline as mdp  # noqa: E402
mdp = importlib.reload(mdp)   # a kernel that imported the module before a git pull keeps the stale copy
assert hasattr(mdp, 'add_benchmark_curve_move'), f'stale muni_data_pipeline at {mdp.__file__}: git pull and restart the kernel'


@dataclass(frozen=True)
class RunConfig:
    spec_version: str = 'ipca_v3_k3_xmmd'   # record spec; relabel when experimenting
    start_date: str = '2026-01-01'
    end_date: str = '2026-09-30'
    train_start: str = '2026-01-02'
    first_oos_date: str = '2026-04-01'
    refit_days: int = 7
    selected_k: int = 3
    candidate_k: tuple[int, ...] = (1, 2, 3, 4, 5)
    mmd_factors: tuple[str, ...] = ()   # v5.1 observed MMD curve factors ('mmd_level', 'mmd_slope', 'mmd_curvature'); closed in v5.2, () = latent only
    target_mode: str = 'excess_mmd'     # v5.2 dependent variable: 'excess_mmd' = yield change minus the MMD move at the bond's tenor; 'yield_change' = raw
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
    pfill_deltas: tuple[float, ...] = (-8.0, -5.0, -3.0, 0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0)   # concessions at which fill probability is modelled and scored (v4.9: to +30 bp; production's logged concession runs to +40)
    pfill_trees: int = 150                               # gradient-boosted fill and edge models per fold
    pfill_max_train: int = 200_000                       # training rows per fold are subsampled to this many for speed
    pfill_min_train: int = 5_000                         # a fold needs this many training prints for the fill and edge models
    # v4.4 factor premia (Section 8b) and the production objective (Section 12c)
    premia_freqs: tuple[str, ...] = ('monthly', 'weekly')   # rebalancing frequencies of the expected-return long-short
    premia_max_abs_ret_bp: float = 3000.0                # a holding-period return beyond this (30% of price) is a data error and is dropped
    charge_units: str = 'auto'                           # units of the charge columns: 'bp' (of yield), 'pct' (percent of yield, x100), 'price' (price points, converted with duration), 'auto' (detected, printed)
    charge_max_age_days: float = 1.0                     # a per-trade charge joins a print as of the print's time within this many days, else the bond's last charge, else the side median
    objective_grid_bp: tuple[float, float, float] = (-8.0, 31.0, 1.0)   # concessions the production objective chooses from (start, stop, step); fill models are interpolated onto it (v4.9: to +30 bp, production's range)
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
    'v51_mmd_factors': 'observed factors g_t = daily moves (bp) of the generic MMD curve (mean of the JAN..DEC columns) from the curve in force at the previous model date to the 16:00 curve of the date: level = mean over 1Y..30Y, slope = 30Y - 2Y, curvature = 2 x 10Y - 2Y - 30Y; loadings Z Gamma_delta with Gamma_delta orthogonal to the latent Gamma (KPS 2019 pre-specified factors); K = 3 latent factors unchanged; the latent-only walk-forward is kept as the baseline (closed in v5.2: -0.03 points out of sample)',
    'v52_target': 'dependent variable = closing-yield change minus the MMD move at the bond\'s years-to-worst and worst-month column over the same close-to-close interval (16:00 curve on or before each close, linear in tenor, no roll-down; Kelly-Palhares-Pruitt rate hedge with loading one); the IPCA is fitted on it and the fit is mapped back to the yield change (fitted = MMD move + Z Gamma f); the yield-change target runs on the same folds and bond-days; a target wins on fit if the paired daily difference in residual sum of squares on the yield change has t > 3 in its favour and the same sign in >= 4 of the held-out months; otherwise the two are equal on fit and the record is the one whose residual has the smaller |slope on the MMD move|',
    'v52_k4_sensitivity': 'K = 4 run as its own spec (ipca_v3_k4_xmmd) beside the K = 3 record (ipca_v3_k3_xmmd); out of sample on the excess target K = 4 beat K = 3 in 6 of 6 months with 5-day block-bootstrap t 2.25 (below the fit bar); K = 4 replaces K = 3 only if the paired daily-dollar results of the engines that use the factor model (v50_paired_dollars, v50_ipca_contrasts) beat the K = 3 run with bootstrap t > 3 in >= 4 of the held-out months',

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
    'v50_mae_selection': 'Section 10 selects the mid on held-out MAE against the print; the selected mid is carried into the economic comparison as one policy, not as the quote of record',
    'v50_one_objective': 'the engines are compared on one common sample (prints with every mid, every fill curve, both edge models and a raw round-trip exit) and one declared objective: realised dollar P&L per month of the raw round trip (no hedge, no evaluation); the next-print mark is the secondary objective',
    'v50_paired_dollars': 'inference on dollars is the paired daily-dollar difference against S at 0 (and against the production proxy), block-bootstrapped in 5-day blocks with the month count; the dollar bar is a positive monthly difference with bootstrap t > 3 in >= 4 of the held-out months; the per-print basis-point bar is reported beside it, never in its place',
    'v50_ipca_contrasts': 'the factor model\'s economic value is read from pairs of engines that differ only in one slot: D vs D- (the mid), the GBM fill curve with vs without IPCA features, the edge model with vs without IPCA features, and both together; each pair is a paired daily-dollar difference on the common sample',
    'v49_proxy_check': 'on the requests production received, a print that crossed the logged quote is compared with the logged won flag: P(won | crossed) is the competition correction to the would-have-filled proxy and P(crossed | won) its coverage; both reported overall and by size; the universe numbers are read with that correction, not restated',
    'v49_replica_b': 'a second replica of the production objective puts the charge inside the quote: fill iff the print crossed the algo mid shifted by charge + x, value x + charge; the replica whose concession distribution is closer to the logged one on the overlap is the one the universe comparison reads',
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
print('Spec', CFG.spec_version, '| K', CFG.selected_k, '| window', CFG.start_date, '->', CFG.end_date, '| first OOS', CFG.first_oos_date, '| dependent variable', CFG.target_mode, '| MMD factors', CFG.mmd_factors or 'off')

# %% [markdown]
# ## 2. Data load and QA
#
# The Step 3 panel is the covered yield-space modelling input written by the pipeline. We also read the
# deduplicated closing marks (for coverage) and the matched trades (for validation). Two data findings from the
# regenerated data are handled here:
#
# - `final_comp_rating` is blank for about 22% of rows while `comp_rating` is blank for about 5%. The rating
# fallback below uses `final_comp_rating`, then `comp_rating`, then `NR`. It also treats pandas `<NA>` as blank,
# which the previous builder did not.
# - Closing-mark yields contain a small number of extreme values. Rows whose yield sits more than `yield_mad_k`
# median absolute deviations from that date's cross-sectional median are flagged and excluded from the fit;
# the target is additionally clipped at ±`target_clip_bp`.

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
benchmark_curve = mdp.load_benchmark_curve(PIPE, CFG.start_date, CFG.end_date)
if benchmark_curve.empty:
    raise FileNotFoundError('No benchmark_curve partitions found. Run mdp.pull_benchmark_curve(PIPE) on JupyterHub before training.')
BENCHMARK_FP = hashlib.sha1(pd.util.hash_pandas_object(
    benchmark_curve[['curve_timestamp', 'tenor_years', 'forward_month', 'benchmark_yield']], index=False
).values.tobytes()).hexdigest()
print(f'benchmark forward curve: {len(benchmark_curve):,} rows | {benchmark_curve["tenor"].nunique()} tenors | '
      f'{pd.to_datetime(benchmark_curve["date"]).min().date()} -> {pd.to_datetime(benchmark_curve["date"]).max().date()}')
record('data', has_msrb_store=bool(HAS_MSRB), has_industry=bool(HAS_INDUSTRY), benchmark_curve_fp=BENCHMARK_FP,
       benchmark_curve_rows=len(benchmark_curve), benchmark_curve_tenors=benchmark_curve['tenor'].nunique(),
       trade_size_coverage=float(raw_panel['trade_size_lag'].notna().mean()) if 'trade_size_lag' in raw_panel.columns else None)

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
# bullets, well below 1 for bonds priced to call) and is close to orthogonal to term. The benchmark curve adds two continuous instruments: the strictly lagged municipal benchmark yield interpolated to years-to-worst, and the bond yield spread over that benchmark. For a model date, the latest earlier 16:00 curve is used, the model date chooses the JAN..DEC forward column, and 1Y..30Y tenors are linearly interpolated. Every other state
# is the explicit `OTHER` bucket: it is reported and used as a feature in the level model, but it is the omitted
# reference category in Z because five exhaustive dummies plus the intercept are exactly collinear. All
# continuous instruments are rank-normalised within date to [-0.5, 0.5]. The target of record is the daily
# closing-yield change in bp, clipped.
#
# **v5.2: the dependent variable.** The factor model is fitted on the yield change in excess of the MMD move at the
# bond's own tenor, `excess_mmd_bp` = target - `mmd_move_bp`, where the move is the curve on or before the bond's
# close minus the curve on or before its previous close, both read at its years-to-worst and worst-month column (no
# roll-down). This is the Kelly-Palhares-Pruitt treatment of rates (returns hedged with a duration-matched Treasury
# portfolio before the factor model sees them). The move is contemporaneous with the close, so it adjusts the
# target and is never an instrument; the lagged curve level and spread above stay the instruments.

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


def term_characteristics(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
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
    return yrs_to_worst, (yrs_to_mat - yrs_to_worst).clip(lower=0), worst


def build_model_panel(panel: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, list[str]]:
    yp = panel.copy()
    yp = yp[yp['extreme_yield'] == 0]
    yp = yp[yp['closing_lag_days'].fillna(1.0) <= cfg.max_lag_days]
    yp['premium_discount_lag1'] = pd.to_numeric(yp['closing_price_lag1'], errors='coerce') - 100.0
    yp['years_to_worst'], yp['extension'], yp['worst_date'] = term_characteristics(yp)
    yp['worst_month'] = yp['worst_date'].dt.month
    yp, benchmark_audit = mdp.add_benchmark_curve_features(
        yp, PIPE, tenor_col='years_to_worst', forward_month_col='worst_month',
        date_col='date', yield_col='closing_yield_lag1', max_age_days=7
    )
    if benchmark_audit['feature_coverage'] < 0.95:
        raise ValueError(f'Benchmark curve feature coverage is only {benchmark_audit["feature_coverage"]:.1%}; aborting factor training.')
    print('benchmark curve feature audit:', benchmark_audit)
    # v5.2: the MMD move at the bond's tenor over its own close-to-close interval (same tenor and month column at both closes)
    yp['prev_close_date'] = yp['date'] - pd.to_timedelta(pd.to_numeric(yp['closing_lag_days'], errors='coerce'), unit='D')
    yp, move_audit = mdp.add_benchmark_curve_move(yp, PIPE, tenor_col='years_to_worst', forward_month_col='worst_month',
                                                  prev_date_col='prev_close_date', date_col='date', max_age_days=7)
    yp = yp.rename(columns={'benchmark_move_bp': 'mmd_move_bp'})
    print('MMD move at tenor audit:', move_audit)
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
    cont = base + ['years_to_worst', 'extension', 'rating_score', 'liquidity_20',
                   'benchmark_yield_lag1', 'benchmark_spread_lag1']
    # v3.9: trade size and print frequency from MSRB prints (strictly before the date), with a no-print flag
    size_cols = [c for c in ['trade_size_lag', 'print_freq_lag'] if c in yp.columns and yp[c].notna().mean() > 0.2]
    if 'trade_size_lag' in yp.columns:
        yp['no_print_flag'] = yp['trade_size_lag'].isna().astype(float)
    cont = cont + size_cols
    yp = yp.dropna(subset=base + [TARGET, 'mmd_move_bp']).copy()
    yp['target_bp_raw'] = yp[TARGET]
    yp[TARGET] = yp[TARGET].clip(-cfg.target_clip_bp, cfg.target_clip_bp)
    yp['excess_mmd_bp'] = yp[TARGET] - yp['mmd_move_bp']
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
            'trade_size_lag', 'print_freq_lag', 'issuer_industry', 'industry_bucket', 'benchmark_yield_lag1',
            'benchmark_spread_lag1', 'benchmark_tenor_clipped', 'curve_date', 'curve_age_days', 'mmd_move_bp', 'excess_mmd_bp'] + chars
    keep = [c for c in dict.fromkeys(keep) if c in yp.columns]
    return yp[keep].sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True), chars


t0 = time.perf_counter()
model, CHARS = build_model_panel(raw_panel, CFG)
FIT_TARGET = 'excess_mmd_bp' if CFG.target_mode == 'excess_mmd' else TARGET   # what the IPCA is fitted on; residuals and fits stay in TARGET space
print(f'[{CFG.spec_version}] model rows {len(model):,} | cusips {model["cusip"].nunique():,} | dates {model["date"].nunique()} | L={len(CHARS)} | {time.perf_counter()-t0:.1f}s')
print('instruments:', CHARS)
print(f'target of record: {TARGET} | IPCA fitted on: {FIT_TARGET} | mean square (bp^2): yield change {(model[TARGET] ** 2).mean():.1f}, excess over MMD {(model["excess_mmd_bp"] ** 2).mean():.1f} | corr(yield change, MMD move at tenor) {model[TARGET].corr(model["mmd_move_bp"]):.3f}')
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
record('model_panel', fit_target=FIT_TARGET, rows=len(model), cusips=model['cusip'].nunique(), dates=model['date'].nunique(), instruments=CHARS, mark_unchanged_share=float(model['mark_unchanged'].mean()), geo_buckets=geo.round(4).to_dict())

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
#
# **v5.1: MMD curve factors.** The municipal benchmark enters twice. As characteristics (Section 3) it tells the model
# where a bond sits against the curve; as **observed factors** it tells the model how the curve moved that day. With
# $g_t$ the daily MMD level, slope (30Y - 2Y) and curvature (2 x 10Y - 2Y - 30Y) moves in bp, the model is the IPCA
# with pre-specified factors of Kelly, Pruitt and Su (2019):
#
# $$r_{i,t} = z_{i,t-1}'\Gamma f_t + z_{i,t-1}'\Gamma_\delta\, g_t + \varepsilon_{i,t},$$
#
# estimated by the same alternating least squares with the stacked factor $[f_t; g_t]$. $\Gamma_\delta$ is kept
# orthogonal to $\Gamma$ (the authors' normalisation): the latent factors carry whatever part of the MMD loadings
# lies in their span, $\Gamma_\delta$ the rest. $K = 0$ is the MMD-only model; the sweep reports every $K$ with
# and without the MMD factors on the same weighted moments, and Section 5 keeps the latent-only walk-forward
# as the baseline the MMD model must beat out of sample.
#
# **v5.2: the curve moves to the left-hand side.** The v5.1 run found no out-of-sample gain from the MMD factors
# (-0.03 points), so they are off (`mmd_factors = ()`, the code path is kept). The curve is removed from the
# dependent variable instead, with a loading of one at each bond's own tenor:
#
# $$r_{i,t} - \Delta\text{MMD}_t(\tau_{i,t-1}) = z_{i,t-1}'\Gamma f_t + \varepsilon_{i,t}.$$
#
# The K sweep and the fit table below are therefore in excess space (`FIT_TARGET`); every out-of-sample result
# from Section 5 on is in total yield-change space, with the fitted move $\Delta\text{MMD} + z'\Gamma f$.

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


MMD_COLS = list(CFG.mmd_factors)


def mmd_curve_factors(curve: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Observed factors g_t in bp: level (mean move over 1Y..30Y), slope (30Y - 2Y move) and curvature (2 x 10Y - 2Y - 30Y
    move) of the generic MMD curve (mean of the JAN..DEC columns). A date's move runs from the curve in force at the
    previous model date to the date's own 16:00 curve, the close the evaluated marks are struck at, so it is a
    contemporaneous realisation like the latent factors (the instruments stay lagged)."""
    c = curve[['date', 'tenor_years', 'benchmark_yield']].copy(); c['date'] = to_ns(c['date']).dt.normalize()
    generic = c.groupby(['date', 'tenor_years'])['benchmark_yield'].mean().unstack('tenor_years').sort_index()
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    pos = generic.index.searchsorted(dates, side='right') - 1
    prev = np.r_[pos[0] - 1, pos[:-1]] if len(pos) else pos
    vals = generic.to_numpy(float)
    move = (vals[np.clip(pos, 0, None)] - vals[np.clip(prev, 0, None)]) * 100.0
    move[(pos < 0) | (prev < 0)] = np.nan
    move = pd.DataFrame(move, index=dates, columns=generic.columns.astype(float))
    t = move.columns.to_numpy(float)
    at = lambda y: move[t[np.argmin(np.abs(t - y))]]  # noqa: E731
    out = pd.DataFrame({'mmd_level': move.mean(axis=1), 'mmd_slope': at(30) - at(2), 'mmd_curvature': 2 * at(10) - at(2) - at(30)}, index=dates)
    return out[MMD_COLS]


MMD_G = mmd_curve_factors(benchmark_curve, model['date'].unique()) if MMD_COLS else pd.DataFrame(index=pd.DatetimeIndex(sorted(model['date'].unique())))
MMD_FP = frame_fingerprint(MMD_G.reset_index(), MMD_COLS) + f'|{MMD_COLS}'


def precompute_moments(frame: pd.DataFrame, chars: list[str], target: str, weighted: bool = True, observed: bool = True) -> dict:
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    L = len(chars)
    A = np.zeros((len(dates), L, L)); b = np.zeros((len(dates), L)); rr = np.zeros(len(dates)); nobs = np.zeros(len(dates), dtype=np.int64)
    pos = {d: i for i, d in enumerate(dates)}
    for d, g in frame.groupby('date', sort=True, observed=True):
        i = pos[pd.Timestamp(d)]
        Z = g[chars].to_numpy(float); r = g[target].to_numpy(float)
        A[i] = Z.T @ Z; b[i] = Z.T @ r; rr[i] = float(r @ r); nobs[i] = len(g)
    w = date_weights(dates, rr, nobs, CFG, DISPERSION_DAYS) if weighted else np.ones(len(dates))
    gobs = MMD_G.reindex(dates).fillna(0.0).to_numpy(float) if (observed and MMD_COLS) else np.zeros((len(dates), 0))
    return {'dates': dates, 'A': A * w[:, None, None], 'b': b * w[:, None], 'rr': rr * w, 'nobs': nobs, 'chars': list(chars), 'w': w, 'rr_unweighted': rr, 'g': gobs}


class MomentIPCA:
    """IPCA by alternating least squares on per-date moments (Kelly, Pruitt, Su 2019). With observed factors g_t in
    m['g'] (KPS pre-specified factors): r_t = Z_t (Gamma f_t + Gamma_delta g_t) + e_t, solved with the stacked factor
    [f_t; g_t]. Gamma_delta is kept orthogonal to Gamma (the authors' normalisation): the latent factors carry the part of
    the observed loadings inside their span, Gamma_delta the rest. K = 0 is the observed-factor-only model."""

    def __init__(self, n_factors: int, max_iter: int = 50, tol: float = 1e-5, ridge: float = 1e-10, seed: int = 0):
        self.K = int(n_factors); self.max_iter = max_iter; self.tol = tol; self.ridge = ridge; self.seed = seed
        self.Gamma = None; self.Gamma_delta = None; self.Factors = None; self.loss_history: list[float] = []

    @staticmethod
    def _g(m) -> np.ndarray:
        g = m.get('g')
        return np.zeros((m['A'].shape[0], 0)) if g is None else np.asarray(g, float)

    def _initial_gamma(self, m, gamma_init=None):
        L = m['A'].shape[1]
        if self.K == 0:
            return np.zeros((L, 0))
        if gamma_init is not None and np.shape(gamma_init) == (L, self.K):
            return np.asarray(gamma_init, float)
        managed = m['b'] / np.maximum(m['nobs'][:, None], 1)
        u, _, _ = np.linalg.svd(managed.T @ managed)
        return u[:, :self.K]

    def _solve_factors(self, Gamma, m, Gd=None):
        A, b, nobs = m['A'], m['b'], m['nobs']; g = self._g(m)
        F = np.zeros((A.shape[0], self.K)); eye = np.eye(self.K)
        if self.K == 0:
            return F
        obs = Gd is not None and g.shape[1] > 0
        for t in range(A.shape[0]):
            if nobs[t] >= self.K:
                bt = b[t] - A[t] @ (Gd @ g[t]) if obs else b[t]
                F[t] = np.linalg.solve(Gamma.T @ A[t] @ Gamma + self.ridge * eye, Gamma.T @ bt)
        return F

    def _identify(self, G, F, Gd=None, g=None):
        if self.K == 0:
            return G, F, Gd
        if Gd is not None and Gd.shape[1] > 0:
            P = np.linalg.solve(G.T @ G + self.ridge * np.eye(self.K), G.T @ Gd)
            F = F + g @ P.T; Gd = Gd - G @ P
        Q, R = np.linalg.qr(G)
        s = np.sign(np.diag(R)); s[s == 0] = 1
        Q = Q @ np.diag(s); R = Q.T @ G; F = F @ R.T
        vals, vecs = np.linalg.eigh(F.T @ F / max(F.shape[0], 1))
        W = vecs[:, np.argsort(vals)[::-1]]
        G = Q @ W; F = F @ W
        s = np.sign(F.mean(axis=0)); s[s == 0] = 1
        return G * s, F * s, Gd

    def _loss(self, G, F, m, Gd=None):
        A, b, rr, nobs = m['A'], m['b'], m['rr'], m['nobs']; g = self._g(m)
        if Gd is not None and g.shape[1] > 0:
            G, F = np.hstack([G, Gd]), np.hstack([F, g])
        tot = sum(rr[t] - 2.0 * F[t] @ G.T @ b[t] + F[t] @ (G.T @ A[t] @ G) @ F[t] for t in range(A.shape[0]))
        return tot / max(int(nobs.sum()), 1)

    def fit(self, m, gamma_init=None, verbose=False):
        A, b = m['A'], m['b']; L = A.shape[1]; K = self.K; g = self._g(m); M = g.shape[1]
        if K + M == 0:
            raise ValueError('IPCA needs at least one latent or observed factor')
        G = self._initial_gamma(m, gamma_init); Gd = np.zeros((L, M))
        F = self._solve_factors(G, m, Gd); G, F, Gd = self._identify(G, F, Gd, g)
        eye = np.eye(L * (K + M)); self.loss_history = []
        for it in range(self.max_iter):
            Ft = np.hstack([self._solve_factors(G, m, Gd), g])
            lhs = np.zeros((L * (K + M), L * (K + M))); rhs = np.zeros(L * (K + M))
            for t in range(A.shape[0]):
                ft = Ft[t]; lhs += np.kron(np.outer(ft, ft), A[t]); rhs += np.kron(ft, b[t])
            Gt = np.linalg.solve(lhs + self.ridge * eye, rhs).reshape((L, K + M), order='F')
            G, F, Gd = self._identify(Gt[:, :K], Ft[:, :K], Gt[:, K:], g)
            loss = self._loss(G, F, m, Gd); self.loss_history.append(loss)
            if verbose and (it == 0 or (it + 1) % 10 == 0):
                print(f'  iter {it+1:03d} loss {loss:.6e}')
            if K == 0 or (it > 0 and abs(self.loss_history[-2] - loss) / max(abs(self.loss_history[-2]), 1e-12) < self.tol):
                break
        self.Gamma, self.Factors, self.Gamma_delta = G, F, Gd
        return self

    def variance_explained(self, m) -> float:
        """1 - SSE/SST over the moments (in-sample)."""
        sse = self._loss(self.Gamma, self.Factors, m, self.Gamma_delta) * max(int(m['nobs'].sum()), 1)
        return 1.0 - sse / max(float(m['rr'].sum()), 1e-12)


t0 = time.perf_counter()
moments = precompute_moments(model, CHARS, FIT_TARGET)
moments_latent = {**moments, 'g': np.zeros((len(moments['dates']), 0))}
print(f'IPCA dependent variable: {FIT_TARGET} | moments: T={len(moments["dates"])}, L={len(CHARS)}, obs={int(moments["nobs"].sum()):,} | {time.perf_counter()-t0:.1f}s')
print(f'date weighting: {CFG.date_weighting} (cap {CFG.date_weight_cap}x) | zero-weight dates: {int((moments["w"] == 0).sum())} | weight range on kept dates {moments["w"][moments["w"] > 0].min():.2f}..{moments["w"][moments["w"] > 0].max():.2f}')
if MMD_COLS:
    print(f'MMD observed factors {MMD_COLS}: {int(MMD_G.notna().all(axis=1).sum())} of {len(MMD_G)} model dates with a curve move (missing -> 0) | sd (bp/day): {MMD_G.std().round(2).to_dict()}')
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(moments['dates'], moments['w'], width=1.0, color='#4C72B0'); ax.set_title('Date weights in the ALS moments (0 = repricing day excluded from the fit)')
savefig('04_date_weights')

# K sweep on the same weighted moments, with and without the MMD factors (K = 0: MMD only)
k_rows, fits, fits_latent = [], {}, {}
for k in sorted(set(CFG.candidate_k) | ({0} if MMD_COLS else set())):
    row = {'K': k}
    if MMD_COLS:
        mk = MomentIPCA(k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(moments); fits[k] = mk
        row['latent + MMD'] = mk.variance_explained(moments); row['iterations'] = len(mk.loss_history)
    if k > 0:
        ml = MomentIPCA(k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(moments_latent); fits_latent[k] = ml
        row['latent only'] = ml.variance_explained(moments_latent)
        if not MMD_COLS:
            fits[k] = ml; row['iterations'] = len(ml.loss_history)
    k_rows.append(row)
k_table = pd.DataFrame(k_rows).set_index('K')
if MMD_COLS:
    k_table['MMD gain'] = k_table['latent + MMD'] - k_table['latent only']
print(f'In-sample variance explained of {FIT_TARGET} (date-weighted), full sample:'); display(k_table.round(4))
fig, ax = plt.subplots(figsize=(7, 4))
for c, mk_ in [('latent only', 'o'), ('latent + MMD', 's')]:
    if c in k_table.columns:
        ax.plot(k_table.index, k_table[c], marker=mk_, label=c)
ax.set_xlabel('K (latent factors)'); ax.set_ylabel('variance explained (in-sample)'); ax.set_title('K sweep (full sample, interpretation only)'); ax.legend()
savefig('04_k_sweep')
record('ipca_full_sample', **{f'var_explained_K{k}_{c.replace(" ", "_").replace("+", "plus")}': float(v) for c in [x for x in ['latent only', 'latent + MMD'] if x in k_table.columns] for k, v in k_table[c].dropna().items()})

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

# %%
CELL_T('4. IPCA: K sweep, Gamma anatomy, factor paths [3]')
# v5.1 MMD factors: what they are, how they track the bond moves, and how bonds load on them
if MMD_COLS:
    _med = model.groupby('date', observed=True)[TARGET].median().rename('median_bond_move_bp')
    _mm = MMD_G.join(_med, how='inner')
    print('MMD factors (bp per day) and the daily median bond move:'); display(_mm.describe().T[['mean', 'std', 'min', 'max']].round(3))
    _corr = _mm.corr()['median_bond_move_bp'].drop('median_bond_move_bp')
    print('correlation with the daily median bond move:', _corr.round(3).to_dict())
    gamma_delta = pd.DataFrame(ipca.Gamma_delta, index=CHARS, columns=MMD_COLS)
    print(f'Gamma_delta at K={CFG.selected_k} (orthogonal to the latent Gamma: the MMD loading the latent factors cannot carry):')
    display(gamma_delta.assign(abs_l2=np.sqrt((gamma_delta ** 2).sum(axis=1))).sort_values('abs_l2', ascending=False).round(4))
    # the MMD-only model (K = 0) gives the raw loadings: a bond's yield move per bp of each MMD factor
    _s = model.sample(min(len(model), 200_000), random_state=CFG.seed)
    _raw = pd.DataFrame(_s[CHARS].to_numpy(float) @ fits[0].Gamma_delta, columns=MMD_COLS, index=_s.index)
    _raw['years_to_worst'] = pd.cut(_s['years_to_worst'], [-1, 2, 5, 10, 15, 20, 100], labels=['<2', '2-5', '5-10', '10-15', '15-20', '>20'])
    load_ytw = _raw.groupby('years_to_worst', observed=True)[MMD_COLS].mean()
    print('MMD-only model (K = 0): mean loading by years to worst (level ~1 = moves with the curve; slope rising with term):'); display(load_ytw.round(3))
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw={'width_ratios': [0.9, 1.4, 1]})
    _lim = float(np.abs(gamma_delta.values).max()) or 1.0
    im = ax[0].imshow(gamma_delta.values, cmap='RdBu_r', vmin=-_lim, vmax=_lim, aspect='auto')
    ax[0].set_yticks(range(len(CHARS))); ax[0].set_yticklabels([c.replace('z_', '').replace('_lag1', '') for c in CHARS], fontsize=7)
    ax[0].set_xticks(range(len(MMD_COLS))); ax[0].set_xticklabels([c.replace('mmd_', '') for c in MMD_COLS]); ax[0].grid(False); plt.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].set_title(f'Gamma_delta (K={CFG.selected_k}, orthogonal to Gamma)', fontsize=10)
    _mm.cumsum().plot(ax=ax[1]); ax[1].axhline(0, color='k', lw=0.5); ax[1].set_title('Cumulative MMD factors vs the cumulative median bond move (bp)', fontsize=10)
    load_ytw.plot(ax=ax[2], marker='o'); ax[2].axhline(0, color='k', lw=0.5); ax[2].axhline(1, color='grey', lw=0.5, ls=':'); ax[2].set_title('MMD-only loadings by years to worst', fontsize=10); ax[2].set_xlabel('')
    savefig('04_mmd_factors')
    record('ipca_full_sample', mmd_corr_with_median_move=_corr.round(4).to_dict(), gamma_delta=gamma_delta.round(4).to_dict(), mmd_only_loading_by_ytw=flat_records(load_ytw))
else:
    print('MMD factors off (CFG.mmd_factors = ()): latent-only IPCA.')

# %%
CELL_T('4. IPCA: K sweep, Gamma anatomy, factor paths [4]')
# The fit table of Kelly-Palhares-Pruitt (2023, corporate bonds), on the IPCA dependent variable (FIT_TARGET): total, time-series and
# cross-section R2 and the relative pricing error, for bond-days (A) and characteristic-managed portfolios x_t = Z_t'r_t/N_t (B).
# Full-sample Gammas; factors re-solved on unweighted moments so every date is scored on its own cross-section.
m_u = precompute_moments(model, CHARS, FIT_TARGET, weighted=False)
_dates = m_u['dates']
_by_date = {d: g for d, g in model.groupby('date', sort=True, observed=True)}


def fit_metrics(r, e, u, assets, dates) -> dict:
    """Total R2 = 1 - sum e^2 / sum r^2; time-series R2 = 1 - within-asset variation of e / of r; cross-section R2 =
    1 - sum e^2 / within-date variation of r; relative pricing error = RMS of each asset's mean unconditional error
    u = r - beta'lambda / RMS of its mean r (lambda = mean factor)."""
    df = pd.DataFrame({'a': assets, 't': dates, 'r': r, 'e': e, 'u': u})
    ga, gt = df.groupby('a', sort=False), df.groupby('t', sort=False)
    am = ga[['r', 'u']].mean()
    return {'Total R2': 100 * (1 - (df['e'] ** 2).sum() / (df['r'] ** 2).sum()),
            'Time Series R2': 100 * (1 - ((df['e'] - ga['e'].transform('mean')) ** 2).sum() / ((df['r'] - ga['r'].transform('mean')) ** 2).sum()),
            'Cross Section R2': 100 * (1 - (df['e'] ** 2).sum() / ((df['r'] - gt['r'].transform('mean')) ** 2).sum()),
            'Relative Pricing Error': 100 * np.sqrt((am['u'] ** 2).mean()) / np.sqrt((am['r'] ** 2).mean())}


def kpp_table_rows(spec: str, mk: MomentIPCA, m: dict) -> list[dict]:
    g = mk._g(m); obs = mk.Gamma_delta is not None and g.shape[1] > 0
    F = mk._solve_factors(mk.Gamma, m, mk.Gamma_delta if obs else None)
    lam = F.mean(axis=0); gbar = g.mean(axis=0) if obs else None
    coef_lam = mk.Gamma @ lam + (mk.Gamma_delta @ gbar if obs else 0.0)
    ind = {k_: [] for k_ in 'reuat'}; mp = {k_: [] for k_ in 'reuat'}
    for i, d in enumerate(_dates):
        gd = _by_date[d]; Z = gd[CHARS].to_numpy(float); r = gd[FIT_TARGET].to_numpy(float); n = len(r)
        fit = Z @ (mk.Gamma @ F[i] + (mk.Gamma_delta @ g[i] if obs else 0.0)); unc = Z @ coef_lam
        for dct, vals in [(ind, (r, r - fit, r - unc, gd['cusip'].to_numpy(), np.full(n, i))),
                          (mp, (Z.T @ r / n, Z.T @ (r - fit) / n, Z.T @ (r - unc) / n, np.arange(len(CHARS)), np.full(len(CHARS), i)))]:
            for k_, v_ in zip('reuat', vals):
                dct[k_].append(v_)
    out = []
    for panel, dct in [('A. Individual bond-days', ind), ('B. Managed portfolios', mp)]:
        out.append({'spec': spec, 'K': mk.K, 'panel': panel, **fit_metrics(*(np.concatenate(dct[k_]) for k_ in 'reuat'))})
    return out


t0 = time.perf_counter()
_m_lat = {**m_u, 'g': np.zeros((len(_dates), 0))}
kpp_rows = []
for k in sorted(fits_latent):
    kpp_rows += kpp_table_rows('latent only', fits_latent[k], _m_lat)
for k in (sorted(fits) if MMD_COLS else []):
    kpp_rows += kpp_table_rows('latent + MMD', fits[k], m_u)
kpp = pd.DataFrame(kpp_rows)
metrics = ['Total R2', 'Time Series R2', 'Cross Section R2', 'Relative Pricing Error']
kpp_tab = kpp.melt(id_vars=['spec', 'panel', 'K'], value_vars=metrics, var_name='metric').pivot_table(index=['spec', 'panel', 'metric'], columns='K', values='value').reindex(metrics, level='metric')
print(f'In-sample fit table, Kelly-Palhares-Pruitt layout (%, {FIT_TARGET}, {len(_dates)} dates, {model["cusip"].nunique():,} bonds, L={len(CHARS)}) | {time.perf_counter() - t0:.0f}s')
display(kpp_tab.round(1))

# Out of sample, the residuals of record (monthly refit): the same R2s on bond-days the Gamma never saw
_o = resid if 'resid' in globals() else None
if _o is not None:
    _oos = fit_metrics(_o['target_bp'].to_numpy(float), _o['pit_residual'].to_numpy(float), _o['pit_residual'].to_numpy(float), _o['cusip'].to_numpy(), _o['date'].to_numpy())
    _oos.pop('Relative Pricing Error')   # needs the premium estimated on training dates only; not defined from the residuals alone
    print(f'Out of sample, K={CFG.selected_k}, {"latent + MMD" if MMD_COLS else "latent only"} (record cadence, total yield change):', {k_: round(v_, 1) for k_, v_ in _oos.items()})
record('ipca_full_sample', kpp_fit_table=kpp.round(3).to_dict(orient='records'), kpp_fit_target=FIT_TARGET)

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
#
# **v5.2: which dependent variable.** The walk-forward is fitted on `target_mode` (the excess over the MMD move at
# tenor) and maps every fit back to the yield change: `fitted_bp` = `curve_offset_bp` (the MMD move, loading one) +
# $\beta' f$, so `target_bp`, `fitted_bp` and `pit_residual` mean what they meant before and every later section is
# unchanged. The other dependent variable runs beside it on the record cadence, on the same folds and bond-days, and
# both are scored on the same yield-change denominator: total, time-series and cross-section R2, the R2 beyond the
# MMD move itself, and the residual's slope on the MMD move (a residual that still carries the curve is not
# market-neutral, and the hedged markout of Section 11 depends on it). The pre-registered rule (`v52_target`)
# states the winner.

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


def walk_forward_residuals(frame: pd.DataFrame, chars: list[str], folds: list[dict], cfg: RunConfig, regime: bool = False, verbose: bool = True, observed: bool = True, target_mode: str | None = None):
    """Returns (residual frame, aligned Gammas, raw Gammas, Gamma_deltas). observed=True adds the MMD factors (v5.1); the
    test-date latent factors are solved after the MMD part Z Gamma_delta g_t is removed. With regime=True each fold fits a high- and a low-dispersion
    Gamma on the training dates and each test date is scored with the Gamma of the regime its previous day indicates.
    target_mode 'excess_mmd' (v5.2) fits on the yield change minus the MMD move at tenor and adds the move back with loading
    one, so target_bp, fitted_bp and pit_residual stay in yield-change space for either dependent variable."""
    parts, gammas, gammas_raw, gammas_delta = [], [], [], []
    excess = (target_mode or cfg.target_mode) == 'excess_mmd'
    fit_col = 'excess_mmd_bp' if excess else TARGET
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
        G_by, Gd_by = {}, {}
        for gname, gdates in groups.items():
            sub = train[train['date'].isin(gdates)]
            if sub['date'].nunique() < 10:
                continue
            mk = MomentIPCA(cfg.selected_k, cfg.max_iter, cfg.tol, cfg.ridge, cfg.seed).fit(precompute_moments(sub, chars, fit_col, observed=observed), gamma_init=inits.get(gname))
            inits[gname] = mk.Gamma
            gv = pd.Timestamp(fo['train_end'])
            gammas_raw.append(pd.DataFrame(mk.Gamma, index=chars, columns=fcols).assign(gamma_version=gv, fold=fo['fold'], regime=gname))
            G_al = mk.Gamma if refs.get(gname) is None else procrustes_align(mk.Gamma, refs[gname])[0]
            refs[gname] = G_al; G_by[gname] = G_al; Gd_by[gname] = mk.Gamma_delta
            if mk.Gamma_delta.shape[1]:
                gammas_delta.append(pd.DataFrame(mk.Gamma_delta, index=chars, columns=MMD_COLS).assign(gamma_version=gv, fold=fo['fold'], regime=gname))
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
            G_al, Gd = G_by[gname], Gd_by[gname]
            Z = g[chars].to_numpy(float); r = g[TARGET].to_numpy(float); B = Z @ G_al
            mv = g['mmd_move_bp'].to_numpy(float); off = mv if excess else np.zeros(len(r))
            D = Z @ Gd; gt = MMD_G.reindex([d]).fillna(0.0).to_numpy(float)[0] if Gd.shape[1] else np.zeros(0)
            mmd_fit = D @ gt if Gd.shape[1] else np.zeros(len(r))
            f = np.linalg.solve(B.T @ B + cfg.ridge * eye, B.T @ (r - off - mmd_fit))
            out = g[['cusip', 'date']].copy()
            out['gamma_version'] = pd.Timestamp(fo['train_end']); out['fold'] = fo['fold']; out['regime'] = gname
            out['target_bp'] = r; out['mmd_move_bp'] = mv; out['curve_offset_bp'] = off; out['mmd_fitted_bp'] = mmd_fit
            out['fitted_bp'] = B @ f + mmd_fit + off; out['pit_residual'] = r - out['fitted_bp']
            for j in range(cfg.selected_k):
                out[f'beta{j+1}'] = B[:, j]; out[f'f{j+1}'] = f[j]
            for j, c in enumerate(MMD_COLS if Gd.shape[1] else []):
                out[f'delta_{c}'] = D[:, j]; out[c] = gt[j]
            parts.append(out)
        if verbose:
            print(f'  fold {fo["fold"]:02d} | train ->{fo["train_end"].date()} | test {fo["test_start"].date()}..{fo["test_end"].date()} | {"/".join(G_by)} | {time.perf_counter()-t0:.1f}s')
    return pd.concat(parts, ignore_index=True), pd.concat(gammas), pd.concat(gammas_raw), (pd.concat(gammas_delta) if gammas_delta else pd.DataFrame())


CADENCE_DAYS = {'weekly': 7, 'monthly': 30, 'quarterly': 91, 'frozen': None, 'regime': CFG.regime_cadence_days}
PANEL_FP = frame_fingerprint(model, [TARGET, 'mmd_move_bp', 'z_duration_ratio', 'z_closing_yield_lag1',
                                           'z_benchmark_yield_lag1', 'z_benchmark_spread_lag1']) + f'|chars:{CHARS}|bench:{BENCHMARK_FP}|disp:{sorted(str(d) for d in DISPERSION_DAYS)}|partial:{sorted(str(d) for d in PARTIAL_DAYS)}'


def _walk_forward_all() -> dict:
    out: dict = {}
    for name in CFG.cadences:
        t0 = time.perf_counter()
        folds_c = make_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CADENCE_DAYS[name])
        res_c, gam_c, gam_raw_c, gam_d_c = walk_forward_residuals(model, CHARS, folds_c, CFG, regime=(name == 'regime'), verbose=(name == CFG.record_cadence))
        out[name] = {'resid': res_c, 'gammas': gam_c, 'gammas_raw': gam_raw_c, 'gammas_delta': gam_d_c, 'folds': folds_c}
        print(f'{name:>10}: {len(folds_c):2d} folds | OOS variance explained {1 - res_c["pit_residual"].var() / res_c["target_bp"].var():.4f} | residual sd {res_c["pit_residual"].std():.2f} bp | {time.perf_counter()-t0:.0f}s')
    if MMD_COLS:   # the latent-only baseline on the record cadence: what the MMD factors must beat out of sample
        t0 = time.perf_counter()
        folds_c = make_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CADENCE_DAYS[CFG.record_cadence])
        res_c, gam_c, gam_raw_c, _ = walk_forward_residuals(model, CHARS, folds_c, CFG, verbose=False, observed=False)
        out[LATENT_ONLY] = {'resid': res_c, 'gammas': gam_c, 'gammas_raw': gam_raw_c, 'folds': folds_c}
        print(f'{LATENT_ONLY:>10}: {len(folds_c):2d} folds | OOS variance explained {1 - res_c["pit_residual"].var() / res_c["target_bp"].var():.4f} | residual sd {res_c["pit_residual"].std():.2f} bp | {time.perf_counter()-t0:.0f}s')
    # v5.2: the other dependent variable on the record cadence, scored in the same yield-change space
    t0 = time.perf_counter()
    folds_c = make_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CADENCE_DAYS[CFG.record_cadence])
    res_c, gam_c, gam_raw_c, _ = walk_forward_residuals(model, CHARS, folds_c, CFG, verbose=False, target_mode=ALT_TARGET_MODE)
    out[ALT_TARGET] = {'resid': res_c, 'gammas': gam_c, 'gammas_raw': gam_raw_c, 'folds': folds_c}
    print(f'{ALT_TARGET:>10}: {len(folds_c):2d} folds | OOS variance explained {1 - res_c["pit_residual"].var() / res_c["target_bp"].var():.4f} | residual sd {res_c["pit_residual"].std():.2f} bp | {time.perf_counter()-t0:.0f}s')
    return out


LATENT_ONLY = f'{CFG.record_cadence}, latent only'
ALT_TARGET_MODE = 'yield_change' if CFG.target_mode == 'excess_mmd' else 'excess_mmd'
ALT_TARGET = f'{CFG.record_cadence}, {ALT_TARGET_MODE} target'
runs = cached('walk_forward', _walk_forward_all, deps=[STEP3_FP, PANEL_FP, BENCHMARK_FP, MMD_FP], code=[walk_forward_residuals, make_folds, procrustes_align, MomentIPCA, precompute_moments, mmd_curve_factors, date_weights])
for name, r in runs.items():
    print(f'{name:>10}: {len(r["folds"]):2d} folds | OOS variance explained {1 - r["resid"]["pit_residual"].var() / r["resid"]["target_bp"].var():.4f} | residual sd {r["resid"]["pit_residual"].std():.2f} bp')

cad_rows = {}
for name, r in runs.items():
    rr_ = r['resid']
    cad_rows[name] = rr_.assign(month=rr_['date'].dt.to_period('M').astype(str)).groupby('month').apply(lambda g: 1 - g['pit_residual'].var() / g['target_bp'].var(), include_groups=False)
cadence = pd.DataFrame(cad_rows); cadence.loc['ALL'] = {name: 1 - r['resid']['pit_residual'].var() / r['resid']['target_bp'].var() for name, r in runs.items()}
print('OOS variance explained of the yield change by month and refit scheme (identical test dates):'); display(cadence.round(4))
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
print(f'residuals of record ({CFG.record_cadence}, fitted on {FIT_TARGET}): {len(resid):,} rows | {resid["cusip"].nunique():,} cusips | {resid["date"].nunique()} dates | {resid["gamma_version"].nunique()} Gamma versions | OOS variance explained {oos_var_expl:.3f} | residual sd {resid["pit_residual"].std():.2f} bp')
resid.to_parquet(ARTIFACTS / 'pit_ipca_residuals_v3.parquet', index=False)
gamma_versions.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_v3.parquet', index=False)
gamma_versions_raw.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_raw_v3.parquet', index=False)
record('walk_forward', folds=len(folds), residual_rows=len(resid), cusips=resid['cusip'].nunique(), oos_dates=resid['date'].nunique(), gamma_versions=resid['gamma_version'].nunique(), oos_variance_explained=float(oos_var_expl), residual_sd_bp=float(resid['pit_residual'].std()))
gamma_versions_delta = runs[CFG.record_cadence].get('gammas_delta', pd.DataFrame())
if MMD_COLS and 'mmd_fitted_bp' in resid.columns:
    gamma_versions_delta.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_delta_v3.parquet', index=False)
    _lo = runs[LATENT_ONLY]['resid'].merge(resid[['cusip', 'date']], on=['cusip', 'date'])
    _ve = lambda e, y: 1 - e.var() / y.var()  # noqa: E731
    mmd_oos = pd.DataFrame({'latent only': _lo.groupby(_lo['date'].dt.to_period('M')).apply(lambda g: _ve(g['pit_residual'], g['target_bp']), include_groups=False),
                            'latent + MMD': resid.groupby(resid['date'].dt.to_period('M')).apply(lambda g: _ve(g['pit_residual'], g['target_bp']), include_groups=False)})
    mmd_oos.loc['ALL'] = [_ve(_lo['pit_residual'], _lo['target_bp']), oos_var_expl]; mmd_oos['MMD gain'] = mmd_oos['latent + MMD'] - mmd_oos['latent only']
    print('OOS variance explained, latent-only baseline vs latent + MMD factors (same test bond-days):'); display(mmd_oos.round(4))
    print(f'share of the OOS fit carried by the MMD part (orthogonal to the latent span): {1 - (resid["target_bp"] - resid["mmd_fitted_bp"]).var() / resid["target_bp"].var():.4f}')
    record('walk_forward', mmd_oos_vs_latent_only=mmd_oos.round(4).reset_index().astype({'date': str}).to_dict(orient='records'))

# v5.2: which dependent variable. Both walk-forwards scored on the yield change, identical bond-days, the pre-registered rule.
_keys = resid[['cusip', 'date']].merge(runs[ALT_TARGET]['resid'][['cusip', 'date']], on=['cusip', 'date'])
_tgt_runs = {f'{CFG.target_mode} (record)': resid, ALT_TARGET_MODE: runs[ALT_TARGET]['resid']}
_tgt_runs = {k_: f_.merge(_keys, on=['cusip', 'date']).sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True) for k_, f_ in _tgt_runs.items()}
_rec_k, _alt_k = list(_tgt_runs)
_base = _tgt_runs[_rec_k]
_r, _mv = _base['target_bp'].to_numpy(float), _base['mmd_move_bp'].to_numpy(float)
_tgt_e = {_rec_k: _base['pit_residual'].to_numpy(float), _alt_k: _tgt_runs[_alt_k]['pit_residual'].to_numpy(float),
          'MMD move at tenor alone (no factor model)': _r - _mv}
tgt_rows = []
for name, e in _tgt_e.items():
    m_ = fit_metrics(_r, e, e, _base['cusip'].to_numpy(), _base['date'].to_numpy()); m_.pop('Relative Pricing Error')
    tgt_rows.append({'dependent variable': name, **m_, 'R2 beyond the MMD move': 100 * (1 - (e ** 2).sum() / ((_r - _mv) ** 2).sum()),
                     'residual sd (bp)': e.std(), 'residual slope on MMD move': np.cov(e, _mv)[0, 1] / _mv.var(), 'corr(residual, MMD move)': np.corrcoef(e, _mv)[0, 1]})
target_cmp = pd.DataFrame(tgt_rows).set_index('dependent variable')
_mo = _base['date'].dt.to_period('M').astype(str).to_numpy()
_sst_m = pd.Series(_r ** 2).groupby(_mo).sum()
target_cmp_m = pd.DataFrame({k_: 100 * (1 - pd.Series(e ** 2).groupby(_mo).sum() / _sst_m) for k_, e in _tgt_e.items()})
_dd = _base['date'].to_numpy()
_d = pd.Series(_tgt_e[_alt_k] ** 2 - _tgt_e[_rec_k] ** 2).groupby(_dd).sum(); _d.index = pd.DatetimeIndex(_d.index)   # > 0: the record's residual is smaller
_dn = _d / pd.Series(_r ** 2).groupby(_dd).sum().mean()
_t = float(np.sqrt(len(_dn)) * _dn.mean() / _dn.std(ddof=1))
_dm = _d.groupby(_d.index.to_period('M')).sum(); _n_rec, _n_alt = int((_dm > 0).sum()), int((_dm < 0).sum())
target_cmp_m.loc['months favouring the record'] = np.nan; target_cmp_m.loc['months favouring the record', _rec_k] = _n_rec
if _t > 3 and _n_rec >= 4:
    target_verdict = f'{_rec_k} wins on fit (paired t {_t:+.1f}, {_n_rec} of {len(_dm)} months): keep target_mode = "{CFG.target_mode}"'
elif _t < -3 and _n_alt >= 4:
    target_verdict = f'{_alt_k} wins on fit (paired t {_t:+.1f}, {_n_alt} of {len(_dm)} months): set target_mode = "{ALT_TARGET_MODE}" with a new spec_version'
else:
    _s = target_cmp['residual slope on MMD move'].abs(); _w = _rec_k if _s[_rec_k] <= _s[_alt_k] else _alt_k
    target_verdict = (f'equal on fit (paired t {_t:+.1f}, months {_n_rec} vs {_n_alt}); the residual closer to rate-neutral is {_w} '
                      f'(|slope on MMD move| {_s[_rec_k]:.3f} vs {_s[_alt_k]:.3f})' + ('' if _w == _rec_k else f': set target_mode = "{ALT_TARGET_MODE}" with a new spec_version'))
print(f'Which dependent variable: out of sample, {CFG.record_cadence} refit, scored on the yield change on {len(_base):,} identical bond-days (R2 in %):'); display(target_cmp.round(3))
print('Total R2 of the yield change by month (%):'); display(target_cmp_m.round(2))
print(f'paired daily difference in residual sum of squares, {_alt_k} minus {_rec_k} (> 0 favours the record): mean {_dn.mean():+.4f} of the mean daily sum of squares, t {_t:+.2f}, days favouring the record {(_d > 0).mean():.0%}, months {_n_rec} of {len(_dm)}')
print('verdict (v52_target):', target_verdict)
record('walk_forward', target_comparison=flat_records(target_cmp), target_comparison_monthly=flat_records(target_cmp_m), target_paired_t=_t, target_months_favouring_record=_n_rec, target_verdict=target_verdict)
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
if CFG.target_mode == 'excess_mmd':   # v5.2: the MMD move at tenor enters with loading one; its trailing variance is systematic (covariance with the excess factors ignored)
    risk_snap['mmd_var_bp2'] = risk_snap['cusip'].map(_g_rs['curve_offset_bp'].var()).fillna(0.0).to_numpy()
    risk_snap['sys_var_bp2'] += risk_snap['mmd_var_bp2']
risk_snap['total_var_bp2'] = risk_snap['sys_var_bp2'] + risk_snap['idio_var_bp2'].fillna(risk_snap['idio_var_bp2'].median())
risk_snap['systematic_share'] = risk_snap['sys_var_bp2'] / risk_snap['total_var_bp2'].replace(0, np.nan)
risk_snap['cluster'] = risk_snap['beta_cluster'].map(CLUSTER_LABEL) if 'beta_cluster' in risk_snap.columns else 'ALL'
risk_tab = risk_snap.groupby('cluster', observed=True).agg(bonds=('cusip', 'size'), systematic_sd_bp=('sys_var_bp2', lambda x: float(np.sqrt(x.median()))), idio_sd_bp=('idio_var_bp2', lambda x: float(np.sqrt(x.median()))),
                                                           systematic_share=('systematic_share', 'median'), **({'mark_noise_sd_bp': ('mark_noise_sd_bp', 'median')} if _mn_col else {}))
print(f'Risk model snapshot on {last_date.date()}: {len(risk_snap):,} bonds. Factor covariance (bp^2 / day):'); display(fcov.round(3))
print('Daily risk by beta-space cluster (medians; systematic = beta\' Sigma beta' + (' + trailing variance of the MMD move at tenor' if CFG.target_mode == 'excess_mmd' else '') + ', idiosyncratic = trailing residual variance):'); display(risk_tab.round(3))
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
        B = g[CHARS].to_numpy(float) @ G; r = g[FIT_TARGET].to_numpy(float)   # v5.2: factors of the dependent variable the Gamma was fitted on
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
# ## 9. Call structure: the priced-to-call residual cluster, its cause, and three fixes
#
# The beta-space map of 6b shows the residual rank clumping where the priced-to-call bonds sit. A residual that clumps in
# the factor space is information the factors left behind. Three readings, each with its own fix:
#
# | Hypothesis | What it says | Fix tested |
# |---|---|---|
# | H1 missing characteristic | the instruments carry call structure only through the duration-ratio and extension ranks; the call's moneyness and the priced-to-call state are absent, and a priced-to-call bond's loadings on term and spread may differ from a bullet's | V1: a priced-to-call flag, call moneyness (coupon minus yield, ranked, callables only), and the flag interacted with years to worst, extension and the benchmark spread |
# | H2 missing factor confined to the group | priced-to-call bonds share a daily move the three factors do not span | V3: a fourth factor restricted to the group (a one-factor IPCA on the stage-1 training residual of the priced-to-call rows, solved each test date on those rows only: the restricted K = 4); V2 is the unrestricted K = 4 for reference |
# | H3 wrong hedge | the v5.2 target removes the MMD move at years to worst, which for these bonds is the call date; a callable's yield moves with a mix of the call and maturity tenors (negative convexity), so the excess target leaves a curve-slope exposure in the residual | V4: for priced-to-call bonds the hedge becomes the worst-tenor move + w x (maturity-tenor move - worst-tenor move), w estimated per fold on the training rows; V5 = V1 + V4 |
#
# **Diagnostics** on the residuals of record, by call structure: out-of-sample R2; the residual's slope on the MMD move and
# on the MMD 30Y-2Y move; the share of the residual sum of squares carried by the group's date mean, against a within-date
# random relabelling of the groups (the pure-noise reference); that date mean's persistence and its regression on the MMD
# level, slope and curvature moves (H3 predicts a slope loading); the same by call moneyness; and the rank clumping of the
# residual among each bond's 20 nearest neighbours in beta space (what the 6b map shows, as a number).
#
# **Variants** run in one harness on the record folds from per-date moments. V0 reproduces the residuals of record (printed
# as a check). All are scored on the yield change on identical bond-days. **Bar** (`lab_call_structure`, fixed before the
# run): a variant replaces the record only if the paired daily difference in residual sum of squares has moving-block
# bootstrap t > 3, favours it in at least 4 of the held-out months, and the priced-to-call common share falls.

# %%
CELL_T('9. Call structure: the priced-to-call residual cluster [1]')
from scipy.stats import rankdata  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

PRE_REGISTERED.update({
    'lab_call_structure': 'fixed 2026-10-14 before the lab run: V1 call characteristics, V2 global K = 4, V3 a priced-to-call-restricted fourth factor (two-stage), V4 a blended call/maturity-tenor MMD hedge for priced-to-call bonds, V5 = V1 + V4, all on the record folds; a variant replaces the record only if the paired daily difference in residual sum of squares of the yield change against V0 has moving-block bootstrap t > 3, favours the variant in >= 4 of the held-out months, and the priced-to-call common-component share falls',
})


def block_boot_t(x, block: int | None = None, n_boot: int | None = None, seed: int | None = None) -> float:
    """t of the mean of a time-ordered series against its moving-block bootstrap sd (CFG.block_days blocks by default)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    block = int(block or CFG.block_days); n_boot = int(n_boot or CFG.n_boot)
    if n < 2 * block or np.std(x) == 0:
        return np.nan
    rng = np.random.default_rng(CFG.seed if seed is None else seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    sd = float(x[idx].mean(axis=1).std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else np.nan


def slope(y, x) -> float:
    """OLS slope of y on x (with intercept) over the finite pairs."""
    y, x = np.asarray(y, float), np.asarray(x, float); ok = np.isfinite(y) & np.isfinite(x)
    return float(np.cov(y[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1)) if ok.sum() > 10 and np.var(x[ok]) > 0 else np.nan


def curve_moves(dates) -> pd.DataFrame:
    """Daily generic-MMD level / 30Y-2Y slope / curvature moves (bp) between successive model dates (as mmd_curve_factors)."""
    c = benchmark_curve[['date', 'tenor_years', 'benchmark_yield']].copy(); c['date'] = to_ns(c['date']).dt.normalize()
    generic = c.groupby(['date', 'tenor_years'])['benchmark_yield'].mean().unstack('tenor_years').sort_index()
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    pos = generic.index.searchsorted(dates, side='right') - 1
    prev = np.r_[pos[0] - 1, pos[:-1]]
    vals = generic.to_numpy(float)
    mv = (vals[np.clip(pos, 0, None)] - vals[np.clip(prev, 0, None)]) * 100.0
    mv[(pos < 0) | (prev < 0)] = np.nan
    mv = pd.DataFrame(mv, index=dates, columns=generic.columns.astype(float)); t = mv.columns.to_numpy(float)
    at = lambda y: mv[t[np.argmin(np.abs(t - y))]]  # noqa: E731
    return pd.DataFrame({'mmd_level': mv.mean(axis=1), 'mmd_slope': at(30) - at(2), 'mmd_curvature': 2 * at(10) - at(2) - at(30)}, index=dates)


def common_component(frame: pd.DataFrame, group_col: str, e_col: str = 'pit_residual') -> tuple[pd.DataFrame, dict]:
    """Per group: the share of its residual sum of squares carried by its date means (a daily move common to the group
    that the factors do not span), the same share after a within-date random relabelling of the groups (the pure-noise
    reference, about 1 / bonds per date), and the persistence of the group's date-mean series."""
    rng = np.random.default_rng(CFG.seed)
    f = frame[['date', group_col, e_col]].dropna().reset_index(drop=True)
    f['_placebo'] = f.groupby('date', observed=True)[group_col].transform(lambda s: rng.permutation(s.to_numpy()))
    rows, series = [], {}
    for g in f[group_col].unique():
        out = {}
        for lab, col in [('common share', group_col), ('placebo share', '_placebo')]:
            sub = f[f[col] == g]
            dm = sub.groupby('date', observed=True)[e_col].transform('mean')
            out[lab] = float((dm ** 2).sum() / (sub[e_col] ** 2).sum())
        series[g] = f[f[group_col] == g].groupby('date', observed=True)[e_col].mean()
        rows.append({group_col: g, 'bond-days': int((f[group_col] == g).sum()), **out, 'excess over placebo': out['common share'] - out['placebo share'],
                     'date-mean sd (bp)': float(series[g].std()), 'date-mean lag-1 autocorr': float(series[g].autocorr(1))})
    return pd.DataFrame(rows).set_index(group_col), series


def knn_clumping(frame: pd.DataFrame, bcols: list[str], e_col: str = 'pit_residual', group_col: str | None = 'call_structure', n_dates: int = 12, k: int = 20) -> pd.DataFrame:
    """Spatial structure of the residual in standardised beta space: on n_dates evenly spaced OOS dates, the correlation
    between a bond's residual rank and the mean residual rank of its k nearest neighbours (0 = nothing for the map to show)."""
    ds = np.sort(frame['date'].unique()); ds = ds[np.unique(np.linspace(0, len(ds) - 1, min(n_dates, len(ds))).astype(int))]
    rows = []
    for d in ds:
        g = frame[frame['date'] == d].dropna(subset=bcols + [e_col])
        if len(g) < 10 * k:
            continue
        X = g[bcols].to_numpy(float); X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
        ind = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
        er = rankdata(g[e_col].to_numpy(float)) / len(g); nb = er[ind].mean(axis=1)
        row = {'date': d, 'all': float(np.corrcoef(er, nb)[0, 1])}
        if group_col:
            lv = g[group_col].to_numpy()
            for lvl in pd.unique(lv):
                m_ = lv == lvl
                if m_.sum() >= 200:
                    row[lvl] = float(np.corrcoef(er[m_], nb[m_])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


CS_ORDER = ['bullet', 'callable, to maturity', 'priced-to-call']
cs = resid[['cusip', 'date', 'target_bp', 'pit_residual', 'mmd_move_bp'] + beta_cols].merge(
    model[['cusip', 'date', 'call_structure', 'years_to_worst', 'extension', 'cpn', 'closing_yield_lag1']], on=['cusip', 'date'], how='left')
cs['call_moneyness'] = pd.to_numeric(cs['cpn'], errors='coerce') - pd.to_numeric(cs['closing_yield_lag1'], errors='coerce')
CM = curve_moves(cs['date'].unique())
cs = cs.merge(CM, left_on='date', right_index=True, how='left')
_levels = [c for c in CS_ORDER if c in set(cs['call_structure'].dropna())]

diag_rows = []
for g in _levels:
    sub = cs[cs['call_structure'] == g]; e, r, mv = (sub[c].to_numpy(float) for c in ['pit_residual', 'target_bp', 'mmd_move_bp'])
    diag_rows.append({'call_structure': g, 'bond-days': len(sub), 'bonds': sub['cusip'].nunique(), 'OOS R2 of the yield change (%)': 100 * (1 - (e ** 2).sum() / (r ** 2).sum()),
                      'R2 beyond the MMD move (%)': 100 * (1 - (e ** 2).sum() / ((r - mv) ** 2).sum()), 'residual mean (bp)': e.mean(), 'residual sd (bp)': e.std(),
                      'slope on the MMD move at worst': slope(e, mv), 'slope on the MMD 30Y-2Y move': slope(e, sub['mmd_slope'])})
cs_diag = pd.DataFrame(diag_rows).set_index('call_structure')
cc_tab, cc_series = common_component(cs, 'call_structure'); cc_tab = cc_tab.reindex(_levels)

# the group's common daily residual against the day's curve moves: H3 predicts a slope loading for priced-to-call
cm_rows = []
for g in _levels:
    dm_ = cc_series[g]; X = CM.reindex(dm_.index).dropna(); y = dm_.reindex(X.index).to_numpy(float)
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(float)]); coef = np.linalg.lstsq(Xc, y, rcond=None)[0]
    res_ = y - Xc @ coef; se = np.sqrt(np.diag(res_ @ res_ / max(len(y) - Xc.shape[1], 1) * np.linalg.pinv(Xc.T @ Xc)))
    cm_rows.append({'call_structure': g, 'days': len(y), 'R2': 1 - (res_ @ res_) / ((y - y.mean()) ** 2).sum(),
                    **{f'{c} (bp per bp)': coef[j + 1] for j, c in enumerate(X.columns)}, **{f'{c} t': coef[j + 1] / se[j + 1] for j, c in enumerate(X.columns)}})
cm_tab = pd.DataFrame(cm_rows).set_index('call_structure')

# inside priced-to-call: does the curve exposure grow with the call's moneyness (the closer to a sure call, the less the maturity tenor should matter)?
_ptc = cs[cs['call_structure'] == 'priced-to-call'].dropna(subset=['call_moneyness'])
mon_tab = pd.DataFrame()
if len(_ptc) > 3_000:
    _ptc = _ptc.assign(tercile=pd.qcut(_ptc['call_moneyness'].rank(method='first'), 3, labels=['low (near the call strike)', 'mid', 'high (deep in the money)']))
    mon_tab = _ptc.groupby('tercile', observed=True).apply(lambda g: pd.Series({
        'bond-days': len(g), 'call moneyness (pct pts)': g['call_moneyness'].mean(), 'years to worst': g['years_to_worst'].mean(), 'extension (years)': g['extension'].mean(),
        'residual sd (bp)': g['pit_residual'].std(), 'slope on the MMD move at worst': slope(g['pit_residual'], g['mmd_move_bp']), 'slope on the MMD 30Y-2Y move': slope(g['pit_residual'], g['mmd_slope'])}), include_groups=False)

clump0 = knn_clumping(cs, beta_cols)
print('Residuals of record by call structure (out of sample, yield change):'); display(cs_diag.round(3))
print('Common daily residual of each call structure: share of the group residual sum of squares in its date mean, against a within-date random relabelling (pure noise):'); display(cc_tab.round(4))
print('That common daily residual regressed on the day\'s generic MMD level, 30Y-2Y slope and curvature moves (H3 predicts a slope loading for priced-to-call):'); display(cm_tab.round(3))
if len(mon_tab):
    print('Priced-to-call bonds by call moneyness tercile:'); display(mon_tab.round(3))
print(f'Residual-rank clumping among the 20 nearest bonds in beta space ({len(clump0)} dates; 0 = no structure):'); display(clump0.mean().to_frame('mean correlation').round(4))

fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
for g, c_ in zip(_levels, ['#4C72B0', '#55A868', '#C44E52']):
    ax[0].plot(cc_series[g].index, cc_series[g].cumsum(), color=c_, label=g)
_cm_cum = CM['mmd_slope'].reindex(cc_series[_levels[0]].index).fillna(0.0).cumsum()
ax0b = ax[0].twinx(); ax0b.plot(_cm_cum.index, _cm_cum, color='grey', ls='--', lw=1, label='MMD 30Y-2Y move, cumulative (right)'); ax0b.set_ylabel('bp', color='grey')
ax[0].axhline(0, color='k', lw=0.5); ax[0].legend(fontsize=7, loc='upper left'); ax0b.legend(fontsize=7, loc='lower left'); ax[0].set_title('Common daily residual by call structure (bp, cumulative)', fontsize=10)
cc_tab[['common share', 'placebo share']].plot.bar(ax=ax[1], color=['#C44E52', '#BBBBBB']); ax[1].set_title('Share of the residual sum of squares in the group date mean', fontsize=10); ax[1].set_xlabel(''); ax[1].tick_params(axis='x', rotation=0, labelsize=8)
clump0.mean().reindex(['all'] + _levels).plot.bar(ax=ax[2], color='#8172B2'); ax[2].axhline(0, color='k', lw=0.5); ax[2].set_title('Residual-rank clumping among the 20 nearest bonds in beta space', fontsize=10); ax[2].tick_params(axis='x', rotation=0, labelsize=8)
savefig('09_call_structure_diagnostics')
record('lab_call_structure', diagnostics=flat_records(cs_diag), common_component=flat_records(cc_tab), curve_regression=flat_records(cm_tab),
       moneyness=flat_records(mon_tab) if len(mon_tab) else [], knn_clumping=clump0.mean().round(4).to_dict())

# %%
CELL_T('9. Call structure: the priced-to-call residual cluster [2]')
assert not MMD_COLS, 'the variant harness is latent-only (CFG.mmd_factors = ())'
# ---- the variant panel: the record instruments, the call-structure candidates, and the MMD move at the maturity tenor
_keep = list(dict.fromkeys(['cusip', 'date', TARGET, 'excess_mmd_bp', 'mmd_move_bp', 'call_structure', 'is_callable', 'years_to_worst', 'extension', 'cpn', 'closing_yield_lag1', 'closing_lag_days'] + CHARS))
mcs = model[_keep].copy().reset_index(drop=True)
mcs['ptc_flag'] = (mcs['call_structure'] == 'priced-to-call').astype(float)
mcs['call_moneyness'] = pd.to_numeric(mcs['cpn'], errors='coerce') - pd.to_numeric(mcs['closing_yield_lag1'], errors='coerce')
mcs, _zm = rank_normalize(mcs, ['call_moneyness'])
mcs[_zm[0]] = mcs[_zm[0]] * mcs['is_callable']          # moneyness means something only for a callable
for c in ['z_years_to_worst', 'z_extension', 'z_benchmark_spread_lag1']:
    if c in mcs.columns:
        mcs[f'ptc_x_{c[2:]}'] = mcs['ptc_flag'] * mcs[c]
CALL_CHARS = ['ptc_flag'] + _zm + [c for c in mcs.columns if c.startswith('ptc_x_')]

_p = mcs.index[mcs['ptc_flag'] == 1]
_mt = mcs.loc[_p, ['date', 'years_to_worst', 'extension', 'closing_lag_days']].reset_index(drop=True)
_mt['years_to_mat'] = _mt['years_to_worst'] + _mt['extension']
_mt['mat_month'] = (_mt['date'] + pd.to_timedelta(_mt['years_to_mat'] * 365.25, unit='D')).dt.month
_mt['prev_close_date'] = _mt['date'] - pd.to_timedelta(pd.to_numeric(_mt['closing_lag_days'], errors='coerce'), unit='D')
_mt, _mt_audit = mdp.add_benchmark_curve_move(_mt, PIPE, tenor_col='years_to_mat', forward_month_col='mat_month', prev_date_col='prev_close_date', date_col='date', max_age_days=7)
mcs['mmd_move_mat_bp'] = mcs['mmd_move_bp']
mcs.loc[_p, 'mmd_move_mat_bp'] = _mt['benchmark_move_bp'].to_numpy()
mcs['dmove_mat_bp'] = (mcs['mmd_move_mat_bp'] - mcs['mmd_move_bp']).fillna(0.0)    # zero for bonds not priced to call
print(f'call-structure instruments: {CALL_CHARS} | priced-to-call bond-days {len(_p):,} ({len(_p) / len(mcs):.1%}) | maturity-tenor move coverage {_mt_audit["coverage"]:.1%} | '
      f'corr(worst-tenor move, maturity minus worst) on priced-to-call {np.corrcoef(mcs.loc[_p, "mmd_move_bp"], mcs.loc[_p, "dmove_mat_bp"])[0, 1]:+.3f}')


def date_moments(frame: pd.DataFrame, chars: list[str], ycols: list[str], mask_col: str | None = None) -> dict:
    """Unweighted per-date moments for several target columns: A_t = Z'Z, b_t[c] = Z'y_c and rr_t[(c1, c2)] = y_c1'y_c2,
    optionally on the rows with mask_col == 1 only (the rows a restricted factor lives on)."""
    dates = pd.DatetimeIndex(sorted(frame['date'].unique())); pos = {d: i for i, d in enumerate(dates)}
    L, T = len(chars), len(dates)
    A = np.zeros((T, L, L)); b = {c: np.zeros((T, L)) for c in ycols}; n = np.zeros(T, dtype=np.int64)
    rr = {(c1, c2): np.zeros(T) for i, c1 in enumerate(ycols) for c2 in ycols[i:]}
    for d, g in frame.groupby('date', sort=True, observed=True):
        if mask_col is not None:
            g = g[g[mask_col].to_numpy() == 1]
        i = pos[pd.Timestamp(d)]; n[i] = len(g)
        if not len(g):
            continue
        Z = g[chars].to_numpy(float); Y = g[ycols].to_numpy(float); A[i] = Z.T @ Z
        for j, c in enumerate(ycols):
            b[c][i] = Z.T @ Y[:, j]
        for (c1, c2) in rr:
            rr[(c1, c2)][i] = float(Y[:, ycols.index(c1)] @ Y[:, ycols.index(c2)])
    return {'dates': dates, 'A': A, 'b': b, 'rr': rr, 'nobs': n}


def combine_moments(dm: dict, coefs: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """b_t and rr_t of the target y = sum_c coefs[c] * y_c on the dates idx."""
    b = sum(w * dm['b'][c][idx] for c, w in coefs.items())
    rr = np.zeros(len(idx))
    for (c1, c2), v in dm['rr'].items():
        if c1 in coefs and c2 in coefs:
            rr += (1.0 if c1 == c2 else 2.0) * coefs[c1] * coefs[c2] * v[idx]
    return b, rr


def fold_moments(dm: dict, coefs: dict, idx: np.ndarray) -> dict:
    """The record's weighted ALS moments on the training dates idx (precompute_moments on the training rows, latent only)."""
    b, rr = combine_moments(dm, coefs, idx); A = dm['A'][idx]; nobs = dm['nobs'][idx]; dates = dm['dates'][idx]
    w = date_weights(dates, rr, nobs, CFG, DISPERSION_DAYS)
    return {'dates': dates, 'A': A * w[:, None, None], 'b': b * w[:, None], 'rr': rr * w, 'nobs': nobs, 'w': w, 'g': np.zeros((len(idx), 0))}


def wf_variant(frame: pd.DataFrame, chars: list[str], k: int, dm: dict, blend: bool = False, dm_group: dict | None = None, k_group: int = 0) -> tuple[pd.DataFrame, dict]:
    """The Section 5 walk-forward on the record folds, from per-date moments: Gamma fitted on each fold's training dates
    (record date weights, Gamma initialised at the previous fold's), applied to its test dates, the fit mapped back to the
    yield change. blend: the fit target is the excess over the worst-tenor MMD move minus w x (maturity-tenor move - worst-
    tenor move), w per fold by least squares on the training rows. dm_group / k_group: a k_group-factor IPCA on the stage-1
    training residual of the ptc_flag rows (moments built algebraically), solved on the test date on those rows only."""
    pos = {d: i for i, d in enumerate(dm['dates'])}; by_date = {d: g for d, g in frame.groupby('date', sort=True, observed=True)}
    parts, info, init, init_g, eye = [], {'blend_w': []}, None, None, np.eye(k)
    for fo in folds:
        idx = np.array([pos[pd.Timestamp(d)] for d in fo['train_dates'] if pd.Timestamp(d) in pos])
        w_bl = 0.0
        if blend:
            sxy = dm['rr'][('excess_mmd_bp', 'dmove_mat_bp')][idx].sum(); sxx = dm['rr'][('dmove_mat_bp', 'dmove_mat_bp')][idx].sum()
            w_bl = float(sxy / sxx) if sxx > 0 else 0.0
            info['blend_w'].append({'fold': fo['fold'], 'train_end': str(pd.Timestamp(fo['train_end']).date()), 'w': w_bl})
        coefs = {'excess_mmd_bp': 1.0, 'dmove_mat_bp': -w_bl} if blend else {'excess_mmd_bp': 1.0}
        m = fold_moments(dm, coefs, idx)
        mk = MomentIPCA(k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(m, gamma_init=init); init = G = mk.Gamma
        G2 = None
        if dm_group is not None and k_group > 0:
            A_g = dm_group['A'][idx]; b_g, rr_g = combine_moments(dm_group, coefs, idx); GF = mk.Factors @ G.T     # Gamma f_t per training date
            b_e = b_g - np.einsum('tij,tj->ti', A_g, GF)
            rr_e = rr_g - 2.0 * np.einsum('ti,ti->t', GF, b_g) + np.einsum('ti,tij,tj->t', GF, A_g, GF)
            w = m['w']
            mg = {'dates': m['dates'], 'A': A_g * w[:, None, None], 'b': b_e * w[:, None], 'rr': rr_e * w, 'nobs': dm_group['nobs'][idx], 'w': w, 'g': np.zeros((len(idx), 0))}
            mk2 = MomentIPCA(k_group, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(mg, gamma_init=init_g); init_g = G2 = mk2.Gamma
        for d in fo['test_dates']:
            g = by_date.get(pd.Timestamp(d))
            if g is None:
                continue
            Z = g[chars].to_numpy(float); r = g[TARGET].to_numpy(float)
            off = g['mmd_move_bp'].to_numpy(float) + (w_bl * g['dmove_mat_bp'].to_numpy(float) if blend else 0.0)
            B = Z @ G; f = np.linalg.solve(B.T @ B + CFG.ridge * eye, B.T @ (r - off)); e = r - off - B @ f
            out = {'cusip': g['cusip'].to_numpy(), 'date': g['date'].to_numpy(), 'target_bp': r, 'offset_bp': off}
            for j in range(k):
                out[f'b{j+1}'] = B[:, j].astype(np.float32)
            if G2 is not None:
                msk = g['ptc_flag'].to_numpy() == 1; bg = np.zeros((len(r), k_group))
                if msk.sum() > 5 * k_group:
                    B2 = Z[msk] @ G2; f2 = np.linalg.solve(B2.T @ B2 + CFG.ridge * np.eye(k_group), B2.T @ e[msk]); e[msk] = e[msk] - B2 @ f2; bg[msk] = B2
                for j in range(k_group):
                    out[f'bg{j+1}'] = bg[:, j].astype(np.float32)
            out['pit_residual'] = e
            parts.append(pd.DataFrame(out))
    return pd.concat(parts, ignore_index=True), info


_K0 = CFG.selected_k
LAB_VARIANTS = {
    'V0 record': dict(call=False, k=_K0),
    'V1 + call characteristics': dict(call=True, k=_K0),
    f'V2 K={_K0 + 1} global': dict(call=False, k=_K0 + 1),
    'V3 + PTC-restricted factor': dict(call=False, k=_K0, group=True),
    'V4 blended-tenor hedge (PTC)': dict(call=False, k=_K0, blend=True),
    'V5 = V1 + V4': dict(call=True, k=_K0, blend=True),
}
_V0 = 'V0 record'


def _call_structure_variants() -> dict:
    t0 = time.perf_counter()
    yc = ['excess_mmd_bp', 'dmove_mat_bp']
    dms = {'base': date_moments(mcs, CHARS, yc), 'call': date_moments(mcs, CHARS + CALL_CHARS, yc), 'ptc': date_moments(mcs, CHARS, yc, mask_col='ptc_flag')}
    print(f'  per-date moments: {time.perf_counter() - t0:.0f}s')
    out = {}
    for name, sp in LAB_VARIANTS.items():
        t1 = time.perf_counter()
        res_v, info = wf_variant(mcs, CHARS + CALL_CHARS if sp['call'] else CHARS, sp['k'], dms['call' if sp['call'] else 'base'], blend=sp.get('blend', False),
                                 dm_group=dms['ptc'] if sp.get('group') else None, k_group=1 if sp.get('group') else 0)
        out[name] = {'resid': res_v if name == _V0 else res_v.drop(columns=['cusip', 'target_bp']), **info}   # rows are in V0's order
        print(f'  {name}: {len(res_v):,} test bond-days | {time.perf_counter() - t1:.0f}s')
    return out


LAB_CS = cached('lab_call_structure_variants', _call_structure_variants,
                deps=[STAGE_KEYS['walk_forward'], PANEL_FP, repr(LAB_VARIANTS), repr(CALL_CHARS), frame_fingerprint(mcs, ['dmove_mat_bp', 'ptc_flag'] + _zm)],
                code=[wf_variant, date_moments, combine_moments, fold_moments, MomentIPCA, date_weights])
_chk = LAB_CS[_V0]['resid'][['cusip', 'date', 'pit_residual']].merge(resid[['cusip', 'date', 'pit_residual']], on=['cusip', 'date'], suffixes=('', '_record'))
print(f'V0 reproduces the residuals of record: {len(_chk):,} of {len(resid):,} bond-days matched | max |difference| {(_chk["pit_residual"] - _chk["pit_residual_record"]).abs().max():.2e} bp')
for name, v in LAB_CS.items():
    if v.get('blend_w'):
        print(f'{name}: blend weight on the maturity-tenor move by fold:', [round(x['w'], 3) for x in v['blend_w']])

# %%
CELL_T('9. Call structure: the priced-to-call residual cluster [3]')
_base = LAB_CS[_V0]['resid']
for name, v in LAB_CS.items():
    assert len(v['resid']) == len(_base) and (v['resid']['date'].to_numpy() == _base['date'].to_numpy()).all(), f'{name}: rows differ from V0'
_aux = _base[['cusip', 'date']].merge(mcs[['cusip', 'date', 'call_structure', 'mmd_move_bp']], on=['cusip', 'date'], how='left').merge(CM[['mmd_slope']], left_on='date', right_index=True, how='left')
_r = _base['target_bp'].to_numpy(float); _mv = _aux['mmd_move_bp'].to_numpy(float); _csv = _aux['call_structure'].to_numpy(); _ptcm = _csv == 'priced-to-call'
_dd = _base['date'].to_numpy(); _sst_d = pd.Series(_r ** 2).groupby(_dd).sum(); _e0 = _base['pit_residual'].to_numpy(float)
_ss0 = pd.Series(_e0 ** 2).groupby(_dd).sum()
rows, cum_gain = [], {}
for name, v in LAB_CS.items():
    res_v = v['resid']; e = res_v['pit_residual'].to_numpy(float)
    m_ = fit_metrics(_r, e, e, _base['cusip'].to_numpy(), _dd); m_.pop('Relative Pricing Error')
    d = (_ss0 - pd.Series(e ** 2).groupby(_dd).sum()) / _sst_d.mean(); d.index = pd.DatetimeIndex(d.index)   # > 0: the variant's residual is smaller
    dmo = d.groupby(d.index.to_period('M')).sum()
    cc, _ = common_component(pd.DataFrame({'date': _dd, 'call_structure': _csv, 'pit_residual': e}), 'call_structure')
    bcols_v = [c for c in res_v.columns if re.fullmatch(r'bg?\d+', c)]
    clump = knn_clumping(res_v.assign(call_structure=_csv), bcols_v)
    rows.append({'variant': name, **m_, 'R2 beyond the MMD move (%)': 100 * (1 - (e ** 2).sum() / ((_r - _mv) ** 2).sum()),
                 'PTC total R2 (%)': 100 * (1 - (e[_ptcm] ** 2).sum() / (_r[_ptcm] ** 2).sum()),
                 'PTC common share': float(cc.loc['priced-to-call', 'common share']) if 'priced-to-call' in cc.index else np.nan,
                 'PTC slope on MMD 30Y-2Y': slope(e[_ptcm], _aux['mmd_slope'].to_numpy(float)[_ptcm]),
                 'kNN clumping, all': float(clump['all'].mean()), 'kNN clumping, PTC': float(clump['priced-to-call'].mean()) if 'priced-to-call' in clump.columns else np.nan,
                 'paired gain (share of mean daily SS)': float(d.mean()), 'paired t (block bootstrap)': block_boot_t(d.to_numpy()) if name != _V0 else np.nan,
                 'months favouring': int((dmo > 0).sum()) if name != _V0 else np.nan, 'months': len(dmo)})
    cum_gain[name] = d.cumsum()
cs_scores = pd.DataFrame(rows).set_index('variant')
_ref = cs_scores.loc[_V0]
cs_scores['passes the bar'] = [bool(n != _V0 and row['paired t (block bootstrap)'] > 3 and row['months favouring'] >= 4 and row['PTC common share'] < _ref['PTC common share']) for n, row in cs_scores.iterrows()]
print(f'Call-structure variants, out of sample on {len(_base):,} identical bond-days (yield change; R2 in %; paired gain > 0 favours the variant):'); display(cs_scores.round(4))
_win = cs_scores.index[cs_scores['passes the bar']].tolist()
cs_verdict = (f'{_win} pass the lab_call_structure bar; carry the best into a new spec_version and re-run the research notebook' if _win
              else 'no variant passes the lab_call_structure bar: the record spec stands; read the diagnostics for which hypothesis the data favour')
print('verdict:', cs_verdict)

fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
_dr = cs_scores[['Total R2', 'Cross Section R2', 'PTC total R2 (%)']].sub(_ref[['Total R2', 'Cross Section R2', 'PTC total R2 (%)']], axis=1).drop(index=_V0)
_dr.plot.barh(ax=ax[0], color=['#4C72B0', '#55A868', '#C44E52']); ax[0].axvline(0, color='k', lw=0.6); ax[0].set_title('Gain over V0 (points of R2, yield change)', fontsize=10); ax[0].tick_params(axis='y', labelsize=8); ax[0].set_ylabel('')
for name, s in cum_gain.items():
    if name != _V0:
        ax[1].plot(s.index, s.to_numpy(), label=name)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('Cumulative paired daily gain over V0 (share of the mean daily sum of squares)', fontsize=10); ax[1].legend(fontsize=7)
cs_scores[['PTC common share', 'kNN clumping, PTC']].plot.barh(ax=ax[2], color=['#C44E52', '#8172B2']); ax[2].set_title('What the fixes do to the priced-to-call cluster', fontsize=10); ax[2].tick_params(axis='y', labelsize=8); ax[2].set_ylabel('')
savefig('09_call_structure_variants')
record('lab_call_structure', variants=flat_records(cs_scores), verdict=cs_verdict, call_chars=CALL_CHARS,
       blend_w={n: v['blend_w'] for n, v in LAB_CS.items() if v.get('blend_w')})

# %% [markdown]
# ## 10. Conditional quantiles in beta space: the quantile grid as a model
#
# The research notebook's quantile grid (its Section 12) is a lookup table: quantiles of an outcome by cell, shrunk to the
# marginal. Here the same object is estimated as a conditional model of every bond's next-day move, with the factor model
# supplying the conditioning: where the bond sits in beta space (cross-sectional) and what state the market and the bond
# are in (time series). Two outcomes, two uses:
#
# - the next-day yield change (total): inventory risk, a per-bond VaR, the width of a quote;
# - the next-day residual of record: the idiosyncratic band, the scale a relative-value signal is standardised by.
#
# **Time series or cross-sectional?** A per-bond time-series quantile regression is not identifiable on about 120 marks a
# bond (the 5% tail is six observations). The workable form is a pooled model whose features are of both kinds, with an
# ablation that says which kind carries the information:
#
# | Feature set | Features (all known at the close of t) |
# |---|---|
# | XS: beta space | the point-in-time betas, their norm, the beta-space cluster |
# | TS: regime and own history | the date's cross-sectional dispersion and its 5-day mean; the bond's trailing 20-observation sd of the yield change and of the residual; its trailing sd of the MMD move at tenor; days since its mark last moved; the filtered mark noise |
# | XS + TS | both, plus the systematic sd $\sqrt{\beta_{i,t}'\hat\Sigma_{f,t}\beta_{i,t}}$, $\hat\Sigma_{f,t}$ the trailing 60-day factor covariance |
#
# **Models**, walk-forward by month (train on the months before, test the month; quantiles 5, 10, 25, 50, 75, 90, 95):
# unconditional; Gaussian with the bond's trailing sd (what a variance risk model implies); **the grid**: quantiles by
# beta-space cluster x activity quintile, shrunk to the pooled quantiles with weight n / (n + 500); **location-scale
# linear**, $Q_\tau(x) = \mu(x) + \sigma(x)\,q_\tau$, with $\sigma$ from a regression of the absolute deviation and
# $q_\tau$ the empirical quantiles of the standardised deviation (monotone in $\tau$, and its scale coefficients say which
# features widen the distribution), on each feature set; and **gradient-boosted quantiles**, one model per $\tau$ on XS +
# TS (crossings repaired by sorting; the crossing rate is reported).
#
# **Scored** by the pinball loss averaged over the quantiles (skill against unconditional and against the grid), by
# calibration (the share of outcomes below each predicted quantile), and by the coverage and width of the 80% and 90%
# intervals. **Bar** (`lab_quantiles`, fixed before the run): a conditional model replaces the grid if its daily pinball
# beats the grid's with moving-block bootstrap t > 3, in at least 4 of the test months, with a calibration error no worse
# than the grid's + 0.01.

# %%
CELL_T('10. Conditional quantiles in beta space [1]')
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PRE_REGISTERED.update({
    'lab_quantiles': 'fixed 2026-10-14 before the lab run: conditional quantiles (5..95) of the next-day yield change and the next-day residual, walk-forward by month; a conditional model replaces the beta-cluster x activity grid if its daily mean pinball loss beats the grid with moving-block bootstrap t > 3, in >= 4 of the test months, with mean absolute calibration error no worse than the grid + 0.01',
})
QR_TAUS = np.array(CFG.grid_taus, float)
assert np.allclose(QR_TAUS, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]), 'the interval columns assume the seven grid quantiles'
QR_TARGETS = {'next_total': 'next-day yield change (total)', 'next_residual': 'next-day residual of record'}
QR_MAX_TRAIN, QR_TREES = 150_000, 150

# ---- the conditioning frame: one row per bond-day with a next observation, features known at the close of t
qf = resid[['cusip', 'date', 'next_date', 'target_bp', 'next_residual', 'curve_offset_bp', 'beta_cluster', 'activity_bucket', 'target_abs_vol', 'resid_sd'] + beta_cols
           + [c for c in ['eta_abs'] if c in resid.columns]].copy()
qf = qf.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
_gq = qf.groupby('cusip', observed=True)
qf['next_total'] = _gq['target_bp'].shift(-1)
qf['mmd_vol_20'] = _gq['curve_offset_bp'].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
qf = qf.merge(model[['cusip', 'date', 'days_since_mark_move']], on=['cusip', 'date'], how='left')
qf['date_disp'] = qf['date'].map(DATE_DISP); qf['date_disp_5d'] = qf['date'].map(DATE_DISP.rolling(5, min_periods=1).mean())
qf['beta_norm'] = np.sqrt((qf[beta_cols] ** 2).sum(axis=1))
_fac = resid.groupby('date')[[f'f{j+1}' for j in range(CFG.selected_k)]].first().sort_index(); _Fv = _fac.to_numpy(float)
_Bq = qf[beta_cols].to_numpy(float); _sys = np.full(len(qf), np.nan)
for d_, ix in qf.groupby('date').indices.items():
    i_ = _fac.index.get_loc(pd.Timestamp(d_)); w_ = _Fv[max(0, i_ - 59): i_ + 1]
    if len(w_) >= 10:
        _sys[ix] = np.sqrt(np.einsum('ij,jk,ik->i', _Bq[ix], np.cov(w_.T), _Bq[ix]))
qf['sys_sd'] = _sys
qf['beta_cluster'] = qf['beta_cluster'].astype(int)
qf['target_month'] = qf['next_date'].dt.to_period('M')
QR_XS = beta_cols + ['beta_norm', 'beta_cluster']
QR_TS = ['date_disp', 'date_disp_5d', 'target_abs_vol', 'resid_sd', 'mmd_vol_20', 'days_since_mark_move'] + (['eta_abs'] if 'eta_abs' in qf.columns else [])
QR_SETS = {'XS: beta space': QR_XS, 'TS: regime and own history': QR_TS, 'XS + TS': QR_XS + QR_TS + ['sys_sd']}
print(f'conditioning frame: {len(qf):,} bond-days | next total observed {qf["next_total"].notna().mean():.1%} | systematic sd available {qf["sys_sd"].notna().mean():.1%} | months {sorted(qf["target_month"].dropna().astype(str).unique())}')


def _q_stats(y: np.ndarray, Q: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """Per date: mean pinball over the quantiles, interval coverage and width, share of outcomes below each quantile."""
    ok = np.isfinite(y) & np.isfinite(Q).all(axis=1); y, Q, dates = y[ok], Q[ok], dates[ok]
    d = y[:, None] - Q
    f = pd.DataFrame({'date': dates, 'n': 1, 'pinball': np.maximum(QR_TAUS * d, (QR_TAUS - 1.0) * d).mean(axis=1),
                      'in90': (y >= Q[:, 0]) & (y <= Q[:, -1]), 'w90': Q[:, -1] - Q[:, 0], 'in80': (y >= Q[:, 1]) & (y <= Q[:, -2]), 'w80': Q[:, -2] - Q[:, 1]})
    for j, t in enumerate(QR_TAUS):
        f[f'below {t:.2f}'] = y <= Q[:, j]
    return f.groupby('date').agg({'n': 'sum', 'pinball': 'mean', 'in90': 'mean', 'w90': 'mean', 'in80': 'mean', 'w80': 'mean', **{f'below {t:.2f}': 'mean' for t in QR_TAUS}})


def _design(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Standardised numeric features (training medians fill gaps, clipped at 5 sd) and cluster dummies (C0 the reference)."""
    num = [f for f in feats if f != 'beta_cluster']
    med = tr[num].median()
    A = tr[num].fillna(med).fillna(0.0).to_numpy(float); Bm = te[num].fillna(med).fillna(0.0).to_numpy(float)
    mu, sd = A.mean(0), A.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    A = np.clip((A - mu) / sd, -5, 5); Bm = np.clip((Bm - mu) / sd, -5, 5); names = list(num)
    if 'beta_cluster' in feats:
        lv = list(range(1, CFG.n_clusters))
        A = np.column_stack([A] + [(tr['beta_cluster'].to_numpy() == c).astype(float) for c in lv])
        Bm = np.column_stack([Bm] + [(te['beta_cluster'].to_numpy() == c).astype(float) for c in lv]); names += [f'cluster C{c}' for c in lv]
    return A, Bm, names


def _ls_quantiles(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], y: str, lo: float, hi: float) -> tuple[np.ndarray, pd.Series]:
    """Location-scale: mu(x) by least squares on the winsorised target, sigma(x) by least squares of |y - mu|, quantiles of
    the standardised deviation on the training rows. Returns the test quantiles and the scale coefficients (bp per sd)."""
    A, Bm, names = _design(tr, te, feats)
    Xa = np.column_stack([np.ones(len(A)), A]); Xb = np.column_stack([np.ones(len(Bm)), Bm])
    yv = tr[y].to_numpy(float); yc = np.clip(yv, lo, hi)
    b_mu = np.linalg.lstsq(Xa, yc, rcond=None)[0]; dev = yc - Xa @ b_mu
    b_s = np.linalg.lstsq(Xa, np.abs(dev), rcond=None)[0]
    floor = 0.25
    u = (yv - Xa @ b_mu) / np.maximum(Xa @ b_s, floor)
    qu = np.quantile(u, QR_TAUS)
    Q = (Xb @ b_mu)[:, None] + np.maximum(Xb @ b_s, floor)[:, None] * qu[None, :]
    return Q, pd.Series(b_s[1:], index=names)


def _hgb_quantiles(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], y: str, lo: float, hi: float) -> tuple[np.ndarray, float]:
    feats = [f for f in feats if tr[f].nunique(dropna=True) >= 2]   # all-missing in the window (eta_abs before the filter's first test month) cannot be binned
    cat = np.array([f == 'beta_cluster' for f in feats]); Xa = tr[feats].to_numpy(float); Xb = te[feats].to_numpy(float); yc = np.clip(tr[y].to_numpy(float), lo, hi)
    Q = np.column_stack([HistGradientBoostingRegressor(loss='quantile', quantile=float(t), max_iter=QR_TREES, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=400,
                                                       l2_regularization=1.0, categorical_features=cat if cat.any() else None, random_state=CFG.seed).fit(Xa, yc).predict(Xb) for t in QR_TAUS])
    crossing = float((np.diff(Q, axis=1) < 0).any(axis=1).mean())
    return np.sort(Q, axis=1), crossing


def _grid_quantiles(tr: pd.DataFrame, te: pd.DataFrame, y: str, cells: tuple[str, ...] = ('beta_cluster', 'activity_bucket')) -> np.ndarray:
    cells = list(cells); pooled = np.quantile(tr[y], QR_TAUS)
    cq = tr.groupby(cells, observed=True)[y].quantile(QR_TAUS).unstack()
    n = tr.groupby(cells, observed=True).size().reindex(cq.index).to_numpy(float)
    w = (n / (n + CFG.grid_shrink_k))[:, None]
    cq = pd.DataFrame(w * cq.to_numpy(float) + (1.0 - w) * pooled[None, :], index=cq.index)
    Q = cq.reindex(pd.MultiIndex.from_frame(te[cells])).to_numpy(dtype=float, copy=True)
    miss = ~np.isfinite(Q).all(axis=1); Q[miss] = pooled
    return Q


def _conditional_quantiles() -> dict:
    stats_rows, last, crossing, coefs = [], {}, [], {}
    months_q = sorted(qf['target_month'].dropna().unique()); zq = stats.norm.ppf(QR_TAUS)
    for m in months_q[1:]:
        t0 = time.perf_counter(); te = qf[qf['target_month'] == m]
        for y, ylab in QR_TARGETS.items():
            tr = qf[(qf['target_month'] < m) & qf[y].notna()]
            if len(tr) < 20_000 or te.empty:
                continue
            trs = tr.sample(min(len(tr), QR_MAX_TRAIN), random_state=CFG.seed)
            lo, hi = np.quantile(tr[y], [0.005, 0.995]); yt = te[y].to_numpy(float); dts = te['date'].to_numpy()
            sd_col = 'target_abs_vol' if y == 'next_total' else 'resid_sd'
            Qs = {'unconditional': np.tile(np.quantile(tr[y], QR_TAUS), (len(te), 1)),
                  'Gaussian, trailing sd': te[sd_col].fillna(tr[sd_col].median()).to_numpy(float)[:, None] * zq[None, :],
                  'grid: beta cluster x activity': _grid_quantiles(tr, te, y)}
            for sname, feats in QR_SETS.items():
                Qs[f'location-scale | {sname}'], cf = _ls_quantiles(trs, te, feats, y, lo, hi)
                if sname == 'XS + TS':
                    coefs[f'{ylab} | {m}'] = cf
            Qs['boosted quantiles | XS + TS'], cr = _hgb_quantiles(trs, te, QR_SETS['XS + TS'], y, lo, hi)
            crossing.append({'target': ylab, 'month': str(m), 'crossing rate before sorting': cr})
            for name, Q in Qs.items():
                st = _q_stats(yt, Q, dts).reset_index(); st['target'] = ylab; st['model'] = name; st['month'] = str(m); stats_rows.append(st)
            if m == months_q[-1]:
                last[ylab] = {'meta': te[['cusip', 'date', 'beta_cluster']].assign(y=yt).reset_index(drop=True),
                              **{k_: Qs[k_].astype(np.float32) for k_ in ['Gaussian, trailing sd', 'grid: beta cluster x activity', 'location-scale | XS + TS', 'boosted quantiles | XS + TS']}}
        print(f'  conditional quantiles, test month {m}: {len(te):,} bond-days | {time.perf_counter() - t0:.0f}s')
    return {'stats': pd.concat(stats_rows, ignore_index=True), 'last': last, 'crossing': pd.DataFrame(crossing), 'ls_coefs': pd.DataFrame(coefs)}


QR = cached('lab_conditional_quantiles', _conditional_quantiles,
            deps=[STAGE_KEYS['walk_forward'], STAGE_KEYS.get('ssm_pooled'), repr(QR_SETS), QR_MAX_TRAIN, QR_TREES, frame_fingerprint(qf, ['next_total', 'next_residual', 'sys_sd', 'mmd_vol_20', 'resid_sd'])],
            code=[_q_stats, _design, _ls_quantiles, _hgb_quantiles, _grid_quantiles])

# %%
CELL_T('10. Conditional quantiles in beta space [2]')
qs = QR['stats']; _bel = [f'below {t:.2f}' for t in QR_TAUS]


def _qagg(g: pd.DataFrame) -> pd.Series:
    w = g['n'].to_numpy(float)
    out = {'bond-days': int(w.sum()), 'pinball (bp)': np.average(g['pinball'], weights=w),
           'coverage 90%': np.average(g['in90'], weights=w), 'width 90% (bp)': np.average(g['w90'], weights=w),
           'coverage 80%': np.average(g['in80'], weights=w), 'width 80% (bp)': np.average(g['w80'], weights=w)}
    out['calibration error'] = float(np.mean([abs(np.average(g[b], weights=w) - t) for b, t in zip(_bel, QR_TAUS)]))
    return pd.Series(out)


qr_tab = qs.groupby(['target', 'model'], sort=False).apply(_qagg, include_groups=False)
_GRID = 'grid: beta cluster x activity'
for tgt in qr_tab.index.get_level_values(0).unique():
    pv = qs[qs['target'] == tgt].pivot_table(index='date', columns='model', values='pinball').sort_index()
    for name in pv.columns:
        d = (pv[_GRID] - pv[name]).dropna(); dmo = d.groupby(pd.DatetimeIndex(d.index).to_period('M')).mean()
        qr_tab.loc[(tgt, name), 'skill vs unconditional (%)'] = 100 * (1 - qr_tab.loc[(tgt, name), 'pinball (bp)'] / qr_tab.loc[(tgt, 'unconditional'), 'pinball (bp)'])
        qr_tab.loc[(tgt, name), 'skill vs grid (%)'] = 100 * (1 - qr_tab.loc[(tgt, name), 'pinball (bp)'] / qr_tab.loc[(tgt, _GRID), 'pinball (bp)'])
        qr_tab.loc[(tgt, name), 't vs grid (block)'] = block_boot_t(d.to_numpy()) if name != _GRID else np.nan
        qr_tab.loc[(tgt, name), 'months beating grid'] = int((dmo > 0).sum()) if name != _GRID else np.nan
        qr_tab.loc[(tgt, name), 'months'] = len(dmo)
    _gc = qr_tab.loc[(tgt, _GRID), 'calibration error']
    for name in pv.columns:
        r_ = qr_tab.loc[(tgt, name)]
        qr_tab.loc[(tgt, name), 'passes the bar'] = bool(name != _GRID and r_['t vs grid (block)'] > 3 and r_['months beating grid'] >= 4 and r_['calibration error'] <= _gc + 0.01)
print('Conditional quantiles of the next-day move, out of sample (pinball averaged over the 7 quantiles; skill = % reduction in pinball; t and months: daily pinball against the grid):'); display(qr_tab.round(3))
_abl = qr_tab.loc[qr_tab.index.get_level_values(1).str.startswith('location-scale'), ['skill vs unconditional (%)', 'skill vs grid (%)', 'calibration error']]
print('Time series or cross-sectional: the location-scale model on each feature set (the ablation):'); display(_abl.round(2))
_cal = qs.groupby(['target', 'model'], sort=False).apply(lambda g: pd.Series({t: np.average(g[b], weights=g['n']) for b, t in zip(_bel, QR_TAUS)}), include_groups=False)
print('Calibration: share of outcomes below each predicted quantile (a calibrated model reads the column header):'); display(_cal.round(3))
if len(QR['crossing']):
    print('Boosted quantiles: share of bond-days with crossing quantiles before sorting:'); display(QR['crossing'].pivot(index='month', columns='target', values='crossing rate before sorting').round(4))
_lc = QR['ls_coefs']
if len(_lc):
    _lcm = pd.DataFrame({tgt: _lc[[c for c in _lc.columns if c.startswith(tgt)]].mean(axis=1) for tgt in QR_TARGETS.values()})
    print('What widens the distribution: location-scale scale coefficients on XS + TS (bp of expected absolute deviation per sd of the feature; clusters against C0), mean over months:'); display(_lcm.round(3).sort_values(QR_TARGETS['next_total'], key=np.abs, ascending=False))

# usage: last test month, the predicted 90% band of the next-day yield change by beta-space cluster, and whether it held
use_rows = []
for tgt, L_ in QR['last'].items():
    meta = L_['meta']; yv = meta['y'].to_numpy(float); ok = np.isfinite(yv)
    for name in [k_ for k_ in L_ if k_ != 'meta']:
        Q = L_[name].astype(float)
        f = pd.DataFrame({'cluster': meta['beta_cluster'].map(lambda c: CLUSTER_LABEL.get(c, f'C{c}')), 'q05': Q[:, 0], 'q50': Q[:, 3], 'q95': Q[:, -1], 'in90': (yv >= Q[:, 0]) & (yv <= Q[:, -1])})[ok]
        use_rows.append(f.groupby('cluster').agg(bonds=('q05', 'size'), q05=('q05', 'median'), q50=('q50', 'median'), q95=('q95', 'median'), coverage90=('in90', 'mean')).reset_index().assign(target=tgt, model=name))
use_tab = pd.concat(use_rows, ignore_index=True) if use_rows else pd.DataFrame()
if len(use_tab):
    print('Last test month, by beta-space cluster: median predicted 5% / 50% / 95% of the next-day move (bp) and the realised 90% coverage:')
    display(use_tab[use_tab['target'] == QR_TARGETS['next_total']].pivot_table(index='cluster', columns='model', values=['q05', 'q95', 'coverage90']).round(2))

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
_sk = qr_tab['skill vs unconditional (%)'].unstack(0).drop(index='unconditional', errors='ignore')
_sk.plot.barh(ax=ax[0, 0]); ax[0, 0].axvline(0, color='k', lw=0.6); ax[0, 0].set_title('Pinball skill vs unconditional (%)', fontsize=10); ax[0, 0].tick_params(axis='y', labelsize=8); ax[0, 0].set_ylabel('')
_t0 = QR_TARGETS['next_total']
for name, c_ in zip(['unconditional', 'Gaussian, trailing sd', _GRID, 'location-scale | XS + TS', 'boosted quantiles | XS + TS'], ['#8C8C8C', '#DD8452', '#4C72B0', '#55A868', '#C44E52']):
    if (_t0, name) in _cal.index:
        ax[0, 1].plot(QR_TAUS, _cal.loc[(_t0, name)].to_numpy(float), marker='o', color=c_, label=name)
ax[0, 1].plot([0, 1], [0, 1], 'k--', lw=0.8); ax[0, 1].set_xlabel('nominal quantile'); ax[0, 1].set_ylabel('realised share below'); ax[0, 1].set_title(f'Calibration out of sample: {_t0}', fontsize=10); ax[0, 1].legend(fontsize=7)
_pm = qs[qs['target'] == _t0].groupby(['month', 'model']).apply(lambda g: np.average(g['pinball'], weights=g['n']), include_groups=False).unstack('model')
(100 * (1 - _pm.div(_pm[_GRID], axis=0))).drop(columns=[_GRID]).plot(ax=ax[1, 0], marker='o'); ax[1, 0].axhline(0, color='k', lw=0.6); ax[1, 0].set_title(f'Monthly pinball skill vs the grid (%): {_t0}', fontsize=10); ax[1, 0].legend(fontsize=7)
if len(use_tab):
    _u = use_tab[use_tab['target'] == _t0]
    for name, c_ in zip(['Gaussian, trailing sd', _GRID, 'boosted quantiles | XS + TS'], ['#DD8452', '#4C72B0', '#C44E52']):
        g_ = _u[_u['model'] == name]
        ax[1, 1].scatter(g_['q95'] - g_['q05'], g_['coverage90'], color=c_, label=name, s=40)
    ax[1, 1].axhline(0.9, color='k', ls='--', lw=0.8); ax[1, 1].set_xlabel('median 90% band width (bp)'); ax[1, 1].set_ylabel('realised coverage'); ax[1, 1].set_title('Last test month, one point per beta-space cluster: width vs coverage', fontsize=10); ax[1, 1].legend(fontsize=7)
savefig('10_conditional_quantiles')
record('lab_quantiles', table=flat_records(qr_tab), calibration=flat_records(_cal), ablation=flat_records(_abl), usage=use_tab.round(4).to_dict(orient='records') if len(use_tab) else [],
       scale_coefficients=flat_records(_lcm) if len(_lc) else [], crossing=QR['crossing'].round(5).to_dict(orient='records'))

# %% [markdown]
# ## 11. Relative value: IPCA peers, a permanent-temporary state space, forward relative moves
#
# **The question.** Do peer-relative deviations revert, by how much and how fast, and does the reversion show up in trades
# and not only in evaluated marks? The architecture: IPCA picks the peers, a state space separates the temporary
# dislocation from the persistent premium and estimates its persistence, and forward relative moves against a frozen
# peer basket validate it. It is kept apart from the daily residual model of Sections 5-7: that model explains a day's
# move, this one asks whether a level gap is a position.
#
# **What changes from the first draft of the framework, and why**
#
# | Element | Draft | Here | Why |
# |---|---|---|---|
# | gap | yield minus peer median yield | spread to MMD at the bond's own tenor minus the peer median spread | peers close in beta space still differ in years to worst; a yield gap carries the curve between them |
# | peers | IPCA-selected | the 20 nearest bonds in standardised point-in-time beta space within the same call structure, at least 10 marked that day; re-formed weekly | the beta space is the factor model's notion of comparable; call structure is the axis it mis-sorts (Section 9) |
# | persistent premium $\mu_i$ | fixed effect | a slow local level $\mu_t = \mu_{t-1} + w_t$, diffuse at the bond's first mark, $\sigma_w$ estimated (zero nests the fixed effect) | a premium that re-prices (downgrade, call notice, new supply from the name) is not read as a dislocation for months |
# | basket changes | frozen at formation | frozen for every outcome; the daily gap series the filter sees is chain-linked across the weekly re-formations | a new basket moves the gap with no bond moving; chain-linking keeps those jumps out of the estimated dynamics |
# | persistence | group level first | $(\phi, \sigma_w, \sigma_u, \sigma_v)$ per call structure by maximum likelihood on the training months only; bond-bootstrap intervals for $\phi$ and the half-life | ~100 marks a bond cannot separate a level from a slow AR(1); the group can, with an interval |
# | signal | filtered $x$ | the model's expected convergence over the horizon, $g_t - \hat\mu_t - \phi^h\hat x_t$, filtered (never smoothed) | the noise part $\hat v_t$ reverts at once, the temporary part at rate $\phi$, the permanent part never |
#
# Per bond, on the chain-linked gap $g$ (bp):
#
# $$g_t = \mu_t + x_t + v_t, \qquad \mu_t = \mu_{t-1} + w_t, \qquad x_t = \phi\,x_{t-1} + u_t .$$
#
# For $0 < \phi < 1$ the expected temporary deviation decays geometrically with half-life $h_{1/2} = \ln 0.5 / \ln\phi$ business
# days: the half-life of the expected response to a shock, not the average time an observed gap takes to cross zero.
#
# **Signals on identical formations** (weekly, every bond with a basket, test months only): the raw gap (everything
# converges); the gap demeaned by the bond's point-in-time shrunk mean; the AR(1) forecast of that demeaned gap (within a
# call structure it ranks exactly like the demeaned gap, so it is read for magnitude); the filtered expected convergence;
# and the Section 7b increment signal, the programme's reference.
#
# **Outcomes against the frozen basket**, over 1, 5, 10 and 20 business days, bp of yield, positive = the bond richened
# against its peers: the relative convergence $-\Delta g$; its split into the bond's own spread move and the peers' move
# (convergence that comes from the peers is not the bond repricing); the bond's factor-hedged move $-\sum\varepsilon$ (what a
# dealer holding only the bond earns over the factor hedge); and the gap measured with the basket re-formed at the horizon
# (the size of the basket-change illusion).
#
# **Controls.** Moving-block bootstrap over formation dates (horizons overlap); month-sign consistency; realised against
# predicted convergence by horizon (calibrated = slope 1) beside the model's $1-\phi^h$; the information coefficient by mark
# staleness; and **transactions**: the bond's MSRB prints in the ten business days after formation, side-neutralised, give
# a print-confirmed convergence (the evaluated convergence on the print day minus the print's deviation from the evaluation),
# and the prints before formation say whether the gap itself is real. Evaluated convergence that prints do not confirm is
# the evaluator catching up, not a trade.
#
# **Bar** (`lab_rv_signal`, fixed before the run): the filtered expected convergence becomes the relative-value signal of
# record if at 5 and 10 days its information coefficient has bootstrap t > 3 and is positive in at least 4 of the test
# months, it beats the demeaned gap with paired t > 2, and its print-confirmed slope has the same sign and at least half
# the size of the evaluated slope. A short fitted half-life alone is not evidence of tradable alpha.

# %%
CELL_T('11. Relative value: peers, state space, forward moves [1]')
PRE_REGISTERED.update({
    'lab_rv_signal': 'fixed 2026-10-14 before the lab run: gap = spread to MMD at own tenor minus the median of the 20 nearest bonds in PIT beta space within the call structure (>= 10 marked), baskets re-formed weekly and frozen for outcomes; g = mu (local level) + x (AR(1)) + v per call structure, ML on training months; the filtered expected convergence g - mu - phi^h x becomes the RV signal of record if at h = 5 and 10 its IC has block-bootstrap t > 3 and is positive in >= 4 test months, its paired IC beats the PIT-demeaned gap with t > 2, and its print-confirmed slope has the same sign and >= half the evaluated slope',
})
RV_K, RV_MIN_PEERS, RV_H = 20, 10, (1, 5, 10, 20)
RV_HMAX = max(RV_H)
RV_DATES = pd.DatetimeIndex(sorted(resid['date'].unique()))
CS_NAMES = {0: 'bullet', 1: 'callable, to maturity', 2: 'priced-to-call'}; CS_CODE = {v: k for k, v in CS_NAMES.items()}

# ---- (1) spread to MMD at the bond's own tenor at each close: the strictly-earlier-curve helper run on date + 1 reads the curve on or before the close
_sp = model.loc[model['date'].isin(RV_DATES), ['cusip', 'date', 'closing_yield', 'years_to_worst', 'call_structure', 'days_since_mark_move', 'modified_duration_lag1']].reset_index(drop=True)
_sp['worst_month'] = (_sp['date'] + pd.to_timedelta(_sp['years_to_worst'] * 365.25, unit='D')).dt.month
_sp['date_p1'] = _sp['date'] + pd.Timedelta(days=1)
_sp, _sp_audit = mdp.add_benchmark_curve_features(_sp, PIPE, tenor_col='years_to_worst', forward_month_col='worst_month', date_col='date_p1', yield_col='closing_yield', max_age_days=7)
_sp['s_bp'] = 100.0 * _sp['benchmark_spread_lag1']
_sp['cs_code'] = _sp['call_structure'].map(CS_CODE)
print(f'spread to MMD at own tenor: coverage {_sp["s_bp"].notna().mean():.1%} | median curve age {_sp_audit["median_curve_age_days"]:.0f} days | median spread {_sp["s_bp"].median():+.1f} bp | IQR {_sp["s_bp"].quantile(0.75) - _sp["s_bp"].quantile(0.25):.1f} bp')

RV_BONDS = pd.Index(pd.unique(_sp['cusip']))
RV_NB, RV_ND = len(RV_BONDS), len(RV_DATES)


def rv_wide(frame: pd.DataFrame, col: str) -> np.ndarray:
    """(dates x bonds) matrix of a column on the RV date and bond grid, NaN where the bond has no row."""
    W = np.full((RV_ND, RV_NB), np.nan)
    di = RV_DATES.get_indexer(frame['date']); bi = RV_BONDS.get_indexer(frame['cusip']); ok = (di >= 0) & (bi >= 0)
    W[di[ok], bi[ok]] = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)[ok]
    return W


RV_S, RV_Y, RV_CS = rv_wide(_sp, 's_bp'), rv_wide(_sp, 'closing_yield'), rv_wide(_sp, 'cs_code')
RV_STALE, RV_DUR = rv_wide(_sp, 'days_since_mark_move'), rv_wide(_sp, 'modified_duration_lag1')
RV_CUME = np.cumsum(np.nan_to_num(rv_wide(resid, 'pit_residual')), axis=0)      # cumulative residual of record: hedged own move = minus its change
RV_SSM = rv_wide(resid, 'ssm_signal') if 'ssm_signal' in resid.columns else np.full((RV_ND, RV_NB), np.nan)
_rb_by_date = {d: g for d, g in resid[['cusip', 'date'] + beta_cols].groupby('date', sort=True)}


def rv_baskets(d0: int) -> tuple[np.ndarray, np.ndarray]:
    """Bonds formed on date index d0 and their RV_K peers: nearest in standardised beta space within the same call structure (self excluded)."""
    g = _rb_by_date[RV_DATES[d0]].dropna(subset=beta_cols)
    bi = RV_BONDS.get_indexer(g['cusip']); g = g[bi >= 0]; bi = bi[bi >= 0]
    X = g[beta_cols].to_numpy(float); X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    grp = RV_CS[d0, bi]; ids, peers = [], []
    for c in np.unique(grp[np.isfinite(grp)]):
        ix = np.flatnonzero(grp == c)
        if len(ix) <= RV_K + 1:
            continue
        ind = NearestNeighbors(n_neighbors=RV_K + 1).fit(X[ix]).kneighbors(X[ix], return_distance=False)
        ind = np.where(ind == np.arange(len(ix))[:, None], -1, ind)
        ind = np.take_along_axis(ind, np.argsort(ind == -1, axis=1, kind='stable'), axis=1)[:, :RV_K]
        ids.append(bi[ix]); peers.append(bi[ix][ind])
    return (np.concatenate(ids), np.vstack(peers)) if ids else (np.zeros(0, int), np.zeros((0, RV_K), int))


def rv_basket_gap(d: int, ids: np.ndarray, peers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gap (own spread minus the peer median spread) and the peer median on date index d, with a frozen basket."""
    sp = RV_S[d][peers]; ok = np.isfinite(sp).sum(axis=1) >= RV_MIN_PEERS
    med = np.full(len(ids), np.nan)
    if ok.any():
        med[ok] = np.nanmedian(sp[ok], axis=1)
    return RV_S[d, ids] - med, med


# ---- (2) weekly baskets; frozen-basket gaps out to RV_HMAX days; the daily gap with the current basket and its chain-linked version
RV_FORM_DAYS = [RV_DATES.get_loc(d) for d in pd.Series(RV_DATES).groupby(RV_DATES.to_period('W')).first()]
RV_FORMS, RV_GCUR, RV_CHAIN = [], np.full((RV_ND, RV_NB), np.nan), np.full((RV_ND, RV_NB), np.nan)
_cadj, _prev = np.zeros(RV_NB), None
t0 = time.perf_counter()
for fi, d0 in enumerate(RV_FORM_DAYS):
    ids, peers = rv_baskets(d0)
    if not len(ids):
        continue
    gaps = np.full((len(ids), RV_HMAX + 1), np.nan); pmed = gaps.copy()
    for h in range(min(RV_HMAX, RV_ND - 1 - d0) + 1):
        gaps[:, h], pmed[:, h] = rv_basket_gap(d0 + h, ids, peers)
    if _prev is not None:                       # chain-link: absorb the jump the new basket makes on the formation date
        _old = np.full(RV_NB, np.nan); _old[_prev[0]] = rv_basket_gap(d0, *_prev)[0]
        jump = gaps[:, 0] - _old[ids]; okj = np.isfinite(jump); _cadj[ids[okj]] += jump[okj]
    d1 = RV_FORM_DAYS[fi + 1] if fi + 1 < len(RV_FORM_DAYS) else RV_ND
    for d in range(d0, d1):
        g_ = gaps[:, d - d0] if d - d0 <= RV_HMAX else rv_basket_gap(d, ids, peers)[0]
        RV_GCUR[d, ids] = g_; RV_CHAIN[d, ids] = g_ - _cadj[ids]
    RV_FORMS.append({'t0': RV_DATES[d0], 'd0': d0, 'month': RV_DATES[d0].to_period('M'), 'ids': ids, 'peers': peers, 'gap': gaps, 'peer_med': pmed})
    _prev = (ids, peers)
_g0 = np.concatenate([F_['gap'][:, 0] for F_ in RV_FORMS])
print(f'{len(RV_FORMS)} weekly formations | {np.mean([len(F_["ids"]) for F_ in RV_FORMS]):,.0f} bonds with a basket per formation | gap at formation: '
      f'median {np.nanmedian(_g0):+.2f} bp, IQR {np.nanpercentile(_g0, 75) - np.nanpercentile(_g0, 25):.1f} bp, |gap| > 25 bp {np.nanmean(np.abs(_g0) > 25):.1%} | {time.perf_counter() - t0:.0f}s')

# %%
CELL_T('11. Relative value: peers, state space, forward moves [2]')
def rv_unpack(th: np.ndarray) -> tuple[float, float, float, float]:
    """phi in (0, 1) (logistic), and the sds of the permanent, temporary and noise shocks (exp)."""
    return float(1.0 / (1.0 + np.exp(-th[0]))), float(np.exp(th[1])), float(np.exp(th[2])), float(np.exp(th[3]))


def rv_filter(Y: np.ndarray, params: tuple, p_mu0: float, want: bool = False):
    """Kalman filter of g = mu + x + v (mu random walk, x AR(1)) batched over bonds (rows of Y, NaN = no mark). The state
    stays at its prior until the bond's first mark, where mu is diffuse (variance p_mu0); that mark's likelihood is dropped."""
    phi, s_w, s_u, s_v = params
    N, T = Y.shape
    a = np.zeros((N, 2)); P = np.zeros((N, 2, 2)); P[:, 0, 0] = p_mu0; P[:, 1, 1] = s_u ** 2 / max(1.0 - phi ** 2, 1e-6)
    Tm = np.array([[1.0, 0.0], [0.0, phi]]); Q = np.diag([s_w ** 2, s_u ** 2])
    started = np.zeros(N, bool); ll = np.zeros(N); nobs = np.zeros(N)
    MU = XH = None
    if want:
        MU = np.full((N, T), np.nan); XH = np.full((N, T), np.nan)
    for t in range(T):
        a = np.where(started[:, None], a @ Tm.T, a)
        P = np.where(started[:, None, None], Tm @ P @ Tm.T + Q, P)
        y = Y[:, t]; obs = np.isfinite(y)
        PH = P[:, :, 0] + P[:, :, 1]
        S_ = PH[:, 0] + PH[:, 1] + s_v ** 2
        v = np.where(obs, y - a[:, 0] - a[:, 1], 0.0)
        use = obs & started
        ll += np.where(use, -0.5 * (LOG2PI + np.log(S_) + v ** 2 / S_), 0.0); nobs += use
        K = PH / S_[:, None]
        a = a + K * v[:, None]
        P = np.where(obs[:, None, None], P - K[:, :, None] * PH[:, None, :], P)
        started |= obs
        if want:
            MU[:, t] = np.where(started, a[:, 0], np.nan); XH[:, t] = np.where(started, a[:, 1], np.nan)
    return (ll, nobs, MU, XH) if want else (ll, nobs)


def rv_fit(Y: np.ndarray, p_mu0: float, th0: np.ndarray | None = None, maxiter: int = 200) -> tuple[np.ndarray, float]:
    th0 = np.array([np.log(0.9 / 0.1), np.log(0.3), np.log(2.0), np.log(2.0)]) if th0 is None else np.asarray(th0, float)

    def nll(th):
        if not np.all(np.isfinite(th)) or np.any(np.abs(th) > 12):
            return 1e12
        ll, n = rv_filter(Y, rv_unpack(th), p_mu0)
        return -ll.sum() / max(n.sum(), 1.0)

    sol = minimize(nll, th0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-7})
    return sol.x, float(-sol.fun)


# point-in-time demeaned gap: chain-linked gap minus the bond's mean of earlier chain-linked gaps, shrunk n / (n + 10), >= 10 earlier marks
_cnt = np.cumsum(np.isfinite(RV_CHAIN), axis=0); _sum = np.cumsum(np.nan_to_num(RV_CHAIN), axis=0)
_cp = np.vstack([np.zeros((1, RV_NB)), _cnt[:-1]]); _spv = np.vstack([np.zeros((1, RV_NB)), _sum[:-1]])
RV_DM = RV_CHAIN - np.where(_cp >= 10, _spv / np.maximum(_cp, 1.0) * _cp / (_cp + 10.0), np.nan)
RV_MONTHS = sorted(pd.unique(RV_DATES.to_period('M')))


def _rv_state_space() -> dict:
    """Expanding monthly folds: per call structure, parameters by ML on the chain-linked gaps of the training months
    (bonds with >= 10 marks, subsampled), the AR(1) coefficient of the demeaned gap on the same rows, then every bond
    filtered through the end of the test month; each bond-day keeps the run of its own call structure that day."""
    MU = np.full((RV_ND, RV_NB), np.nan); XH = np.full((RV_ND, RV_NB), np.nan); rows, th_prev, boot = [], {}, []
    rng = np.random.default_rng(CFG.seed)
    for m in RV_MONTHS[1:]:
        t0 = time.perf_counter()
        d_start = int(RV_DATES.searchsorted(m.start_time)); d_end = int(RV_DATES.searchsorted((m + 1).start_time))
        Ytr = RV_CHAIN[:d_start]; grp_tr = pd.DataFrame(RV_CS[:d_start]).ffill().iloc[-1].to_numpy(); n_tr = np.isfinite(Ytr).sum(axis=0)
        fitted = {}
        for c, cname in CS_NAMES.items():
            cols = np.flatnonzero((grp_tr == c) & (n_tr >= 10))
            if len(cols) < 100:
                continue
            sub = rng.choice(cols, min(len(cols), CFG.ssm_fit_bonds), replace=False); Y = Ytr[:, sub].T
            p0 = 4.0 * float(np.nanvar(Y))
            th, ll = rv_fit(Y, p0, th_prev.get(c)); th_prev[c] = th; fitted[c] = (rv_unpack(th), p0)
            x_, y_ = RV_DM[:d_start - 1, cols].ravel(), RV_DM[1:d_start, cols].ravel(); ok = np.isfinite(x_) & np.isfinite(y_)
            phi_ar = float((x_[ok] @ y_[ok]) / (x_[ok] @ x_[ok])) if ok.sum() > 100 else np.nan
            phi, s_w, s_u, s_v = fitted[c][0]
            rows.append({'month': str(m), 'call_structure': cname, 'bonds fit': len(sub), 'phi': phi, 'half-life (days)': np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan,
                         'sd permanent (bp)': s_w, 'sd temporary (bp)': s_u, 'sd noise (bp)': s_v, 'temporary sd, stationary (bp)': s_u / np.sqrt(max(1 - phi ** 2, 1e-6)),
                         'phi AR(1), demeaned gap': phi_ar, 'loglik per obs': ll})
            if m == RV_MONTHS[-1]:          # bond-bootstrap interval for phi on the last fold
                for b in range(12):
                    bs = rng.choice(cols, min(len(cols), CFG.ssm_fit_bonds // 2), replace=True)
                    thb, _ = rv_fit(Ytr[:, bs].T, p0, th, maxiter=80); boot.append({'call_structure': cname, 'phi': rv_unpack(thb)[0]})
        Yall = RV_CHAIN[:d_end].T
        for c, (params, p0) in fitted.items():
            _, _, mu_, x_ = rv_filter(Yall, params, p0, want=True)
            sel = RV_CS[d_start:d_end] == c
            MU[d_start:d_end][sel] = mu_.T[d_start:d_end][sel]; XH[d_start:d_end][sel] = x_.T[d_start:d_end][sel]
        print(f'  RV state space, test month {m}: groups {[CS_NAMES[c] for c in fitted]} | {time.perf_counter() - t0:.0f}s')
    return {'mu': MU, 'x': XH, 'params': pd.DataFrame(rows), 'boot': pd.DataFrame(boot)}


RVSS = cached('lab_rv_state_space', _rv_state_space, deps=[STAGE_KEYS['walk_forward'], RV_K, RV_MIN_PEERS, float(np.nansum(RV_CHAIN)), int(np.isfinite(RV_CHAIN).sum())],
              code=[rv_filter, rv_fit, rv_unpack, rv_baskets, rv_basket_gap])
rv_params = RVSS['params']
if len(RVSS['boot']):
    _ci = RVSS['boot'].groupby('call_structure')['phi'].quantile([0.05, 0.95]).unstack()
    _ci.columns = ['phi 5%', 'phi 95%']; _ci['half-life 5% (days)'] = np.log(0.5) / np.log(_ci['phi 5%']); _ci['half-life 95% (days)'] = np.log(0.5) / np.log(_ci['phi 95%'])
else:
    _ci = pd.DataFrame()
print('Permanent-temporary state space of the peer gap, by call structure and fold (bp; half-life in business days):'); display(rv_params.round(3))
if len(_ci):
    print('Bond-bootstrap 90% interval for phi and the half-life, last fold (12 resamples):'); display(_ci.round(3))

# %%
CELL_T('11. Relative value: peers, state space, forward moves [3]')
_PHI = {(r_['month'], CS_CODE[r_['call_structure']]): r_['phi'] for _, r_ in rv_params.iterrows()}
_PHI_AR = {(r_['month'], CS_CODE[r_['call_structure']]): r_['phi AR(1), demeaned gap'] for _, r_ in rv_params.iterrows()}
_test_months = set(rv_params['month'])
ev_parts = []
for F_ in RV_FORMS:
    mstr = str(F_['month'])
    if mstr not in _test_months:
        continue
    d0, ids, gaps, pm = F_['d0'], F_['ids'], F_['gap'], F_['peer_med']
    grp = RV_CS[d0, ids]
    phi = np.array([_PHI.get((mstr, c), np.nan) if np.isfinite(c) else np.nan for c in grp]); phi_ar = np.array([_PHI_AR.get((mstr, c), np.nan) if np.isfinite(c) else np.nan for c in grp])
    mu, xh, ch = RVSS['mu'][d0, ids], RVSS['x'][d0, ids], RV_CHAIN[d0, ids]
    f = {'t0': F_['t0'], 'month': mstr, 'bi': ids, 'group': grp, 'D': RV_DUR[d0, ids], 'stale': RV_STALE[d0, ids], 'raw': gaps[:, 0], 'dm': RV_DM[d0, ids],
         'mu_hat': mu, 'x_hat': xh, 'v_hat': ch - mu - xh, 'ref': -RV_SSM[d0, ids]}
    for h in RV_H:
        f[f'ar_{h}'] = (1.0 - phi_ar ** h) * f['dm']; f[f'ssf_{h}'] = (ch - mu) - phi ** h * xh
        if d0 + h < RV_ND:
            f[f'conv_{h}'] = -(gaps[:, h] - gaps[:, 0]); f[f'own_{h}'] = -(RV_S[d0 + h, ids] - RV_S[d0, ids]); f[f'peer_{h}'] = pm[:, h] - pm[:, 0]
            f[f'hedged_{h}'] = -(RV_CUME[d0 + h, ids] - RV_CUME[d0, ids]); f[f'refresh_{h}'] = -(RV_GCUR[d0 + h, ids] - gaps[:, 0])
    ev_parts.append(pd.DataFrame(f))
EV = pd.concat(ev_parts, ignore_index=True)
EV = EV[np.isfinite(EV['raw'])].reset_index(drop=True)
print(f'evaluation sample: {EV["t0"].nunique()} formations in {sorted(_test_months)} | {len(EV):,} bond-formations | filtered signal available {EV["ssf_5"].notna().mean():.1%}')


def rv_eval(sig: str, out: str, h: int, frame: pd.DataFrame | None = None) -> tuple[dict, pd.Series]:
    """Per formation date: rank IC, slope of the outcome on the signal and the top-minus-bottom quintile outcome; their means
    with moving-block bootstrap t over formations (block = horizon in weeks, at least 2), and the months with a positive IC."""
    f = (EV if frame is None else frame)[['t0', sig, out]].dropna()
    if f['t0'].nunique() < 4:
        return {}, pd.Series(dtype=float)
    by = f.groupby('t0')
    ic = by.apply(lambda g: g[sig].corr(g[out], method='spearman') if len(g) >= 200 else np.nan, include_groups=False).dropna()
    sl = by.apply(lambda g: slope(g[out], g[sig]) if len(g) >= 200 else np.nan, include_groups=False).dropna()
    qq = by.apply(lambda g: g.loc[g[sig].rank(pct=True) > 0.8, out].mean() - g.loc[g[sig].rank(pct=True) <= 0.2, out].mean() if len(g) >= 200 else np.nan, include_groups=False).dropna()
    blk = max(2, int(np.ceil(h / 5))); icm = ic.groupby(pd.DatetimeIndex(ic.index).to_period('M')).mean()
    return {'formations': len(ic), 'IC': ic.mean(), 'IC t (block)': block_boot_t(ic.to_numpy(), block=blk), 'months IC > 0': int((icm > 0).sum()), 'months': len(icm),
            'slope (bp per bp)': sl.mean(), 'slope t (block)': block_boot_t(sl.to_numpy(), block=blk), 'Q5 - Q1 (bp)': qq.mean(), 'Q5 - Q1 t (block)': block_boot_t(qq.to_numpy(), block=blk)}, ic


RV_SIGNALS = {'raw gap': lambda h: 'raw', 'demeaned gap (PIT)': lambda h: 'dm', 'AR(1) forecast': lambda h: f'ar_{h}', 'filtered expected convergence': lambda h: f'ssf_{h}', '7b increment signal (reference)': lambda h: 'ref'}
rv_rows, rv_ics = [], {}
for h in RV_H:
    if f'conv_{h}' not in EV.columns:
        continue
    for name, col in RV_SIGNALS.items():
        res_, ic = rv_eval(col(h), f'conv_{h}', h)
        if res_:
            rv_rows.append({'h (days)': h, 'signal': name, **res_}); rv_ics[(h, name)] = ic
rv_tab = pd.DataFrame(rv_rows).set_index(['h (days)', 'signal'])
for h in RV_H:
    a_, b_ = rv_ics.get((h, 'filtered expected convergence')), rv_ics.get((h, 'demeaned gap (PIT)'))
    if a_ is not None and b_ is not None:
        d_ = (a_ - b_).dropna(); rv_tab.loc[(h, 'filtered expected convergence'), 'paired IC vs demeaned, t (block)'] = block_boot_t(d_.to_numpy(), block=max(2, int(np.ceil(h / 5))))
print('Peer-relative convergence against the frozen basket (bp of yield, positive = the bond richened against its peers); per-formation IC, slope and quintile spread:'); display(rv_tab.round(3))

# where the convergence comes from, for the filtered signal and the raw gap: own leg, peer leg, factor-hedged own move, re-formed basket
dec_rows = []
for h in RV_H:
    for name, sig in [('filtered expected convergence', f'ssf_{h}'), ('raw gap', 'raw')]:
        for lab, out in [('relative, frozen basket', f'conv_{h}'), ('own spread leg', f'own_{h}'), ('peer leg', f'peer_{h}'), ('own move, factor-hedged', f'hedged_{h}'), ('relative, basket re-formed at h', f'refresh_{h}')]:
            if out in EV.columns and sig in EV.columns:
                res_, _ = rv_eval(sig, out, h)
                if res_:
                    dec_rows.append({'h (days)': h, 'signal': name, 'outcome': lab, 'IC': res_['IC'], 'Q5 - Q1 (bp)': res_['Q5 - Q1 (bp)'], 'Q5 - Q1 t (block)': res_['Q5 - Q1 t (block)'], 'slope (bp per bp)': res_['slope (bp per bp)']})
rv_dec = pd.DataFrame(dec_rows).set_index(['h (days)', 'signal', 'outcome'])
print('Where the convergence comes from (a convergence the peers make, or one the re-formed basket manufactures, is not the bond repricing):'); display(rv_dec.round(3))

# decay calibration: realised convergence per bp of predicted convergence by horizon (1 = calibrated) beside the model's 1 - phi^h
_phi_bar = float(rv_params['phi'].mean()) if len(rv_params) else np.nan
cal_rows = []
for h in RV_H:
    if f'conv_{h}' in EV.columns:
        cal_rows.append({'h (days)': h, 'realised per bp of filtered prediction': rv_tab.loc[(h, 'filtered expected convergence'), 'slope (bp per bp)'] if (h, 'filtered expected convergence') in rv_tab.index else np.nan,
                         'realised share of the raw gap closed': rv_tab.loc[(h, 'raw gap'), 'slope (bp per bp)'] if (h, 'raw gap') in rv_tab.index else np.nan,
                         'realised per bp of AR(1) prediction': rv_tab.loc[(h, 'AR(1) forecast'), 'slope (bp per bp)'] if (h, 'AR(1) forecast') in rv_tab.index else np.nan,
                         'model: temporary share decayed, 1 - phi^h': 1 - _phi_bar ** h})
rv_cal = pd.DataFrame(cal_rows).set_index('h (days)')
print('Decay calibration (slopes of realised on predicted convergence; the filtered prediction is calibrated at 1):'); display(rv_cal.round(3))

# mark staleness: a gap on a mark that has not moved is the evaluator's, not the market's
EV['stale bucket'] = pd.cut(EV['stale'], [-0.5, 0.5, 2.5, 9.5, np.inf], labels=['moved today', '1-2 days', '3-9 days', '10+ days'])
st_rows = []
for b, g in EV.groupby('stale bucket', observed=True):
    for name, sig in [('filtered expected convergence', 'ssf_5'), ('raw gap', 'raw')]:
        res_, _ = rv_eval(sig, 'conv_5', 5, frame=g)
        if res_:
            st_rows.append({'mark staleness at formation': b, 'signal': name, 'bond-formations': len(g), 'IC': res_['IC'], 'IC t (block)': res_['IC t (block)'], 'Q5 - Q1 (bp)': res_['Q5 - Q1 (bp)']})
rv_stale = pd.DataFrame(st_rows).set_index(['mark staleness at formation', 'signal'])
print('h = 5 by mark staleness at formation:'); display(rv_stale.round(3))

# %%
CELL_T('11. Relative value: peers, state space, forward moves [4]')
# ---- transactions: MSRB prints of the bond around formation, as deviations from its evaluation that day, side-neutralised
_side_col = next((c for c in ['side', 'tradetype'] if HAS_MSRB and c in msrb_raw.columns), None)
rv_pc, rv_entry = pd.DataFrame(), pd.DataFrame()
if _side_col and 'yield' in msrb_raw.columns:
    tp = pd.DataFrame({'cusip': msrb_raw['cusip'].astype('string').to_numpy(), 'date': to_ns(msrb_raw['tradetime']).dt.normalize().to_numpy(),
                       'side': msrb_raw[_side_col].astype('string').str.upper().str.strip().str[0].to_numpy(), 'y': pd.to_numeric(msrb_raw['yield'], errors='coerce').to_numpy(float)})
    tp = tp.dropna(); tp = tp[tp['side'].isin(['P', 'S', 'D']) & (tp['y'] > -5.0) & (tp['y'] < 30.0)]
    _di = RV_DATES.get_indexer(tp['date']); _bi = RV_BONDS.get_indexer(tp['cusip']); _ok = (_di >= 0) & (_bi >= 0)
    tp = tp[_ok].assign(di=_di[_ok], bi=_bi[_ok])
    tp['dev_bp'] = 100.0 * (tp['y'].to_numpy(float) - RV_Y[tp['di'].to_numpy(), tp['bi'].to_numpy()])
    tp = tp[np.isfinite(tp['dev_bp']) & (tp['dev_bp'].abs() < 200)]
    tp['dev_bp'] = tp['dev_bp'] - tp.groupby('side')['dev_bp'].transform('median')
    _dd = tp.groupby(['di', 'bi'])['dev_bp'].mean()
    RV_DEV = np.full((RV_ND, RV_NB), np.nan); RV_DEV[_dd.index.get_level_values(0).to_numpy(), _dd.index.get_level_values(1).to_numpy()] = _dd.to_numpy()
    print(f'print tape: {len(tp):,} prints on {len(_dd):,} bond-days of the RV grid | side-neutralised deviation from the evaluation: median |dev| {tp["dev_bp"].abs().median():.1f} bp')
    pc_parts, en_parts = [], []
    for F_ in RV_FORMS:
        if str(F_['month']) not in _test_months:
            continue
        d0, ids, gaps = F_['d0'], F_['ids'], F_['gap']; H = min(10, RV_ND - 1 - d0)
        en_parts.append(pd.DataFrame({'t0': F_['t0'], 'bi': ids, 'entry print deviation (bp)': np.nanmean(RV_DEV[max(0, d0 - 4): d0 + 1][:, ids], axis=0)}))
        if H < 1:
            continue
        dev = RV_DEV[d0 + 1: d0 + H + 1][:, ids].T; conv = -(gaps[:, 1:H + 1] - gaps[:, [0]]); has = np.isfinite(dev) & np.isfinite(conv); n_ = has.sum(axis=1); ok = n_ > 0
        pc_parts.append(pd.DataFrame({'t0': F_['t0'], 'bi': ids[ok], 'print days': n_[ok],
                                      'evaluated convergence, print days': (np.where(has, conv, 0.0).sum(axis=1) / np.maximum(n_, 1))[ok],
                                      'print-confirmed convergence': (np.where(has, conv - dev, 0.0).sum(axis=1) / np.maximum(n_, 1))[ok]}))
    rv_pc = pd.concat(pc_parts, ignore_index=True).merge(EV[['t0', 'bi', 'raw', 'ssf_5', 'ssf_10']], on=['t0', 'bi'], how='left') if pc_parts else pd.DataFrame()
    rv_entry = pd.concat(en_parts, ignore_index=True).merge(EV[['t0', 'bi', 'raw', 'ssf_5']], on=['t0', 'bi'], how='left') if en_parts else pd.DataFrame()
pc_rows = []
for name, sig in [('filtered expected convergence (h = 5)', 'ssf_5'), ('filtered expected convergence (h = 10)', 'ssf_10'), ('raw gap', 'raw')]:
    for lab, out in [('evaluated', 'evaluated convergence, print days'), ('print-confirmed', 'print-confirmed convergence')]:
        if len(rv_pc):
            res_, _ = rv_eval(sig, out, 10, frame=rv_pc)
            if res_:
                pc_rows.append({'signal': name, 'outcome': lab, 'bond-formations': int(rv_pc[sig].notna().sum()), 'IC': res_['IC'], 'IC t (block)': res_['IC t (block)'], 'slope (bp per bp)': res_['slope (bp per bp)'], 'Q5 - Q1 (bp)': res_['Q5 - Q1 (bp)']})
rv_print = pd.DataFrame(pc_rows).set_index(['signal', 'outcome']) if pc_rows else pd.DataFrame()
if len(rv_print):
    print('Transactions: the same bond-formations scored on evaluated and on print-confirmed convergence over the days the bond printed (h <= 10):'); display(rv_print.round(3))
if len(rv_entry):
    _e = rv_entry.dropna(subset=['entry print deviation (bp)', 'raw'])
    _e = _e.assign(_g=_e['raw'] - _e.groupby('t0')['raw'].transform('mean'), _d=_e['entry print deviation (bp)'] - _e.groupby('t0')['entry print deviation (bp)'].transform('mean'))
    print(f'Is the gap real at formation? Prints in the 5 days before formation, deviation from the evaluation on the gap (within formation): slope {slope(_e["_d"], _e["_g"]):+.3f} bp per bp '
          f'on {len(_e):,} bond-formations (0 = prints sit where the evaluation does; negative = cheap-on-evaluation bonds print richer, the gap is partly the evaluator\'s)')

# ---- the bar
def _get(h, col):
    k_ = (h, 'filtered expected convergence')
    return rv_tab.loc[k_, col] if k_ in rv_tab.index and col in rv_tab.columns else np.nan


_bar_h = {h: bool(_get(h, 'IC t (block)') > 3 and _get(h, 'months IC > 0') >= 4 and _get(h, 'paired IC vs demeaned, t (block)') > 2) for h in (5, 10)}
_pe = rv_print.loc[('filtered expected convergence (h = 5)', 'evaluated'), 'slope (bp per bp)'] if ('filtered expected convergence (h = 5)', 'evaluated') in rv_print.index else np.nan
_pp = rv_print.loc[('filtered expected convergence (h = 5)', 'print-confirmed'), 'slope (bp per bp)'] if ('filtered expected convergence (h = 5)', 'print-confirmed') in rv_print.index else np.nan
_print_ok = bool(np.isfinite(_pe) and np.isfinite(_pp) and np.sign(_pe) == np.sign(_pp) and abs(_pp) >= 0.5 * abs(_pe))
rv_verdict = ('the filtered expected convergence passes the lab_rv_signal bar: it becomes the relative-value signal of record' if all(_bar_h.values()) and _print_ok
              else f'the filtered expected convergence does not pass the lab_rv_signal bar (IC/paired at h = 5, 10: {_bar_h}; print-confirmed slope {_pp:+.3f} vs evaluated {_pe:+.3f})')
print('verdict:', rv_verdict)

# ---- snapshot of the last formation, for the desk
_lf = RV_FORMS[-1]; _li = _lf['ids']
_last_month = rv_params['month'].max() if len(rv_params) else None
_hl = {CS_CODE[r_['call_structure']]: r_['half-life (days)'] for _, r_ in rv_params[rv_params['month'] == _last_month].iterrows()} if _last_month else {}
_phi_l = {CS_CODE[r_['call_structure']]: r_['phi'] for _, r_ in rv_params[rv_params['month'] == _last_month].iterrows()} if _last_month else {}
_grp_l = RV_CS[_lf['d0'], _li]; _phl = np.array([_phi_l.get(c, np.nan) if np.isfinite(c) else np.nan for c in _grp_l])
_mul, _xl, _chl = RVSS['mu'][_lf['d0'], _li], RVSS['x'][_lf['d0'], _li], RV_CHAIN[_lf['d0'], _li]
rv_snap = pd.DataFrame({'cusip': RV_BONDS[_li].astype(str), 'call_structure': [CS_NAMES.get(int(c), 'NA') if np.isfinite(c) else 'NA' for c in _grp_l], 'gap (bp)': _lf['gap'][:, 0],
                        'permanent mu (bp)': _mul, 'temporary x (bp)': _xl, 'expected convergence, 5d (bp)': (_chl - _mul) - _phl ** 5 * _xl, 'expected convergence, 10d (bp)': (_chl - _mul) - _phl ** 10 * _xl,
                        'group half-life (days)': [_hl.get(c, np.nan) if np.isfinite(c) else np.nan for c in _grp_l], 'peers': [','.join(RV_BONDS[p[:5]].astype(str)) for p in _lf['peers']]})
rv_snap.to_parquet(ARTIFACTS / f'rv_signal_snapshot_{_lf["t0"].date()}.parquet', index=False)
print(f'Last formation {_lf["t0"].date()}: cheapest and richest by expected 10-day convergence (bp of yield; the filtered value is only as current as the state-space fold that covers this date):')
display(pd.concat([rv_snap.nlargest(8, 'expected convergence, 10d (bp)'), rv_snap.nsmallest(8, 'expected convergence, 10d (bp)')]).round(2))

fig, ax = plt.subplots(2, 3, figsize=(21, 10))
if len(rv_params):
    _lm = rv_params['month'].max()
    for i_, (cname, g) in enumerate(rv_params.groupby('call_structure')):
        ax[0, 0].plot(g['month'], g['half-life (days)'], marker='o', color=f'C{i_}', label=cname)
        if len(_ci) and cname in _ci.index and (g['month'] == _lm).any():
            hl_ = float(g.loc[g['month'] == _lm, 'half-life (days)'].iloc[0]); lo_, hi_ = _ci.loc[cname, 'half-life 5% (days)'], _ci.loc[cname, 'half-life 95% (days)']
            if np.isfinite(hl_) and np.isfinite(lo_) and np.isfinite(hi_):
                ax[0, 0].errorbar([_lm], [hl_], yerr=[[max(hl_ - lo_, 0.0)], [max(hi_ - hl_, 0.0)]], color=f'C{i_}', capsize=4, lw=2)
    ax[0, 0].set_title('Half-life of the temporary gap by fold (business days; bars: 90% bootstrap, last fold)', fontsize=10); ax[0, 0].legend(fontsize=7)
for name, c_ in zip(RV_SIGNALS, ['#8C8C8C', '#4C72B0', '#55A868', '#C44E52', '#DD8452']):
    s_ = [rv_tab.loc[(h, name), 'IC'] if (h, name) in rv_tab.index else np.nan for h in RV_H]
    ax[0, 1].plot(RV_H, s_, marker='o', color=c_, label=name)
ax[0, 1].axhline(0, color='k', lw=0.6); ax[0, 1].set_xlabel('horizon (business days)'); ax[0, 1].set_title('Rank IC with relative convergence, frozen basket', fontsize=10); ax[0, 1].legend(fontsize=7)
if len(rv_cal):
    rv_cal.plot(ax=ax[0, 2], marker='o'); ax[0, 2].axhline(1, color='k', ls='--', lw=0.6); ax[0, 2].axhline(0, color='k', lw=0.6); ax[0, 2].set_xlabel('horizon (business days)'); ax[0, 2].set_title('Decay calibration: realised per bp of predicted convergence', fontsize=10); ax[0, 2].legend(fontsize=7)
if len(rv_dec):
    _dq = rv_dec.xs('filtered expected convergence', level='signal')['Q5 - Q1 (bp)'].unstack('outcome')
    _dq.plot(ax=ax[1, 0], marker='o'); ax[1, 0].axhline(0, color='k', lw=0.6); ax[1, 0].set_title('Filtered signal, top minus bottom quintile (bp): where the convergence comes from', fontsize=10); ax[1, 0].legend(fontsize=7)
if len(rv_print):
    rv_print['slope (bp per bp)'].unstack('outcome').plot.barh(ax=ax[1, 1], color=['#4C72B0', '#C44E52']); ax[1, 1].axvline(0, color='k', lw=0.6); ax[1, 1].set_title('Evaluated vs print-confirmed convergence: slope on the signal', fontsize=10); ax[1, 1].set_ylabel(''); ax[1, 1].tick_params(axis='y', labelsize=8)
if len(rv_stale):
    rv_stale['IC'].unstack('signal').plot.bar(ax=ax[1, 2]); ax[1, 2].axhline(0, color='k', lw=0.6); ax[1, 2].set_title('h = 5 rank IC by mark staleness at formation', fontsize=10); ax[1, 2].tick_params(axis='x', rotation=0); ax[1, 2].set_xlabel('')
savefig('11_relative_value')
record('lab_rv_signal', state_space=rv_params.round(5).to_dict(orient='records'), phi_interval=flat_records(_ci) if len(_ci) else [], signals=flat_records(rv_tab), decomposition=flat_records(rv_dec),
       calibration=flat_records(rv_cal), staleness=flat_records(rv_stale), transactions=flat_records(rv_print) if len(rv_print) else [], verdict=rv_verdict,
       design={'k_peers': RV_K, 'min_peers': RV_MIN_PEERS, 'horizons': list(RV_H), 'formations': len(RV_FORMS)})

# %% [markdown]
# ## 12. Residual dependence after the factors: a factor-structured vine copula
#
# **The idea.** IPCA answers which common factors explain the cross-section; a copula on what it leaves answers how the
# residuals still move together, including in the tails. In the factor-plus-copula form,
# $r_{i,t} = \beta_{i,t}'f_t + \varepsilon_{i,t}$, $u_{i,t} = F_i(\varepsilon_{i,t})$, and the object of interest is
# $C(u_{1,t}, \dots, u_{N,t})$: after what the common exposures explain, are the remaining moves unusually related?
#
# **What has to change to make it work on munis**
#
# | Issue | In the ETF framing | Here | Why |
# |---|---|---|---|
# | dimension | a handful of assets, one regular vine over all of them | about 9,500 bonds a day: a full vine has ~45 million pair copulas | not estimable and not needed; the factor model already says which bonds belong together |
# | structure | free | a three-node C-vine per bond-day pooled within call structure: the global residual factor $G_t$ (the day's mean standardised residual), the bond's peer factor $P_{i,t}$ (mean standardised residual of its 20 beta-space peers from Section 11, frozen weekly), and the bond $U_{i,t}$. Tree 1: $(U,G)$, $(P,G)$; tree 2: $(U,P \mid G)$ | a bi-factor copula in vine form (Krupskii-Joe): exchangeable within a call structure, so ~100 marks a bond pool into hundreds of thousands of pairs per edge |
# | margins | assumed or fitted per asset | $z = \varepsilon / \hat\sigma_{t-1}$ (the bond's trailing residual sd, strictly earlier), mapped through the empirical CDF of the training months: point in time, rank-based | the residual's kurtosis is in the hundreds and its scale moves with the regime; a copula fitted to unstandardised ranks measures volatility clustering, not dependence |
# | marks | prices | evaluated marks: bond-days whose mark did not move are dropped (a stale mark's residual is minus its fitted move, a mechanical tie to the factors), and so are dispersion days (an evaluator-wide reprice is dependence the evaluator makes) | otherwise the copula measures the evaluator |
# | families | pick one | Gaussian, Student-t, Clayton and Gumbel with their 180-degree rotations, Frank; selected per edge, call structure and fold by AIC on the training months | lower and upper tails can differ (a sell-off is not a rally); the Gaussian vine is the benchmark that says whether any of that matters |
#
# **What it is for, tested out of sample** (expanding monthly folds, parameters on earlier months only):
#
# 1. *Is the dependence non-Gaussian?* The selected vine against the Gaussian vine, by test-month log-likelihood per bond-day (paired daily, block bootstrap).
# 2. *Risk: joint tails.* $P(U < q \mid P < q)$ and $P(U > 1-q \mid P > 1-q)$ at $q$ = 5% and 1%, implied by each vine (by simulation) against the test month's empirical frequency. An inventory of a bond and its peers loses together at that rate; a Gaussian copula sets it to zero in the limit.
# 3. *Relative value: unusual given its peers.* The conditional PIT $h_{U \mid P,G}$ is how extreme today's move is given what the peers and the market did. It is tested as a reversal signal on the next 1 and 5 days of the bond's residual against the unconditional PIT and the Gaussian-vine conditional PIT. The prior is low: under any of these families the conditional PIT orders bonds almost like a linear partial residual except in the tails, so the copula's value, if any, should show in risk before it shows in ranking.
#
# **Bars** (`lab_vine_copula`, fixed before the run): *risk*: the selected vine replaces the Gaussian vine if its test log-likelihood gain has block-bootstrap t > 3 and is positive in at least 4 test months, and its mean absolute tail-dependence calibration error is smaller. *Signal*: the vine conditional PIT is adopted if its daily IC on the 5-day residual has t > 3, is positive in at least 4 months, and beats the Gaussian-vine conditional PIT with paired t > 2.

# %%
CELL_T('12. Residual dependence: a factor-structured vine copula [1]')
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402

PRE_REGISTERED.update({
    'lab_vine_copula': 'fixed 2026-10-14 before the lab run: C-vine (global residual factor G, peer factor P = mean standardised residual of the 20 Section 11 beta-space peers, bond U), tree 1 (U,G), (P,G), tree 2 (U,P|G), pooled within call structure; margins = training-month empirical CDF of the residual over its trailing sd; stale marks and dispersion days excluded; families Gaussian, t, Clayton, Gumbel (+180 rotations), Frank chosen by AIC per edge, group and fold; risk bar: test log-likelihood gain over the Gaussian vine with block-bootstrap t > 3, positive in >= 4 test months, and smaller mean absolute tail-dependence calibration error at q = 5%, 1%; signal bar: the vine conditional PIT h(U|P,G) as a 5-day residual reversal signal with daily IC t > 3, positive in >= 4 months, and paired IC over the Gaussian-vine conditional PIT with t > 2',
})
CV_FAMILIES = ('gaussian', 'student-t', 'clayton', 'survival clayton', 'gumbel', 'survival gumbel', 'frank')
CV_BOUNDS = {'clayton': (1e-3, 25.0), 'gumbel': (1.0 + 1e-4, 20.0), 'frank': (-35.0, 35.0)}
CV_MAX_FIT, CV_NSIM, CV_Q = 200_000, 100_000, (0.05, 0.01)
_EPS = 1e-10


def _cl(u):
    return np.clip(np.asarray(u, float), _EPS, 1.0 - _EPS)


def cop_logpdf(fam: str, par: tuple, u, v) -> np.ndarray:
    """Log density of a bivariate copula; 'survival X' is X rotated 180 degrees (tails swapped)."""
    u, v = _cl(u), _cl(v)
    if fam.startswith('survival '):
        return cop_logpdf(fam[9:], par, 1.0 - u, 1.0 - v)
    if fam == 'gaussian':
        r = par[0]; x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        return -0.5 * np.log(1 - r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
    if fam == 'student-t':
        r, nu = par; x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        q = (x * x + y * y - 2 * r * x * y) / (1 - r * r)
        return (gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(np.pi * nu) - 0.5 * np.log(1 - r * r) - (nu + 2) / 2 * np.log1p(q / nu)
                - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu))
    th = par[0]
    if fam == 'clayton':
        L = np.logaddexp(-th * np.log(u), -th * np.log(v)); lw = L + np.log1p(-np.exp(-L))
        return np.log1p(th) - (1 + th) * (np.log(u) + np.log(v)) - (2 + 1 / th) * lw
    if fam == 'gumbel':
        x, y = -np.log(u), -np.log(v); S = x ** th + y ** th; A = S ** (1 / th)
        return -A - np.log(u) - np.log(v) + (th - 1) * (np.log(x) + np.log(y)) + (1 / th - 2) * np.log(S) + np.log(A + th - 1)
    if fam == 'frank':
        eu, ev, e1 = np.expm1(-th * u), np.expm1(-th * v), np.expm1(-th)
        return np.log(th * -e1) - th * (u + v) - 2 * np.log(np.abs(-e1 - eu * ev))
    raise ValueError(fam)


def cop_h(fam: str, par: tuple, u, v) -> np.ndarray:
    """h(u | v) = dC(u, v) / dv: the conditional CDF of the first margin given the second."""
    u, v = _cl(u), _cl(v)
    if fam.startswith('survival '):
        return 1.0 - cop_h(fam[9:], par, 1.0 - u, 1.0 - v)
    if fam == 'gaussian':
        r = par[0]; return stats.norm.cdf((stats.norm.ppf(u) - r * stats.norm.ppf(v)) / np.sqrt(1 - r * r))
    if fam == 'student-t':
        r, nu = par; x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        return stats.t.cdf((x - r * y) / np.sqrt((nu + y * y) * (1 - r * r) / (nu + 1)), nu + 1)
    th = par[0]
    if fam == 'clayton':
        L = np.logaddexp(-th * np.log(u), -th * np.log(v)); lw = L + np.log1p(-np.exp(-L))
        return np.exp((-th - 1) * np.log(v) + (-1 - 1 / th) * lw)
    if fam == 'gumbel':
        x, y = -np.log(u), -np.log(v); S = x ** th + y ** th; A = S ** (1 / th)
        return np.exp(-A) / v * y ** (th - 1) * S ** (1 / th - 1)
    if fam == 'frank':
        eu, ev, e1 = np.expm1(-th * u), np.expm1(-th * v), np.expm1(-th)
        return (ev + 1) * eu / (e1 + eu * ev)
    raise ValueError(fam)


def cop_hinv(fam: str, par: tuple, w, v, iters: int = 34) -> np.ndarray:
    """Inverse of h in its first argument: closed form for Gaussian and t, bisection otherwise."""
    w, v = _cl(w), _cl(v)
    if fam == 'gaussian':
        r = par[0]; return stats.norm.cdf(stats.norm.ppf(w) * np.sqrt(1 - r * r) + r * stats.norm.ppf(v))
    if fam == 'student-t':
        r, nu = par; y = stats.t.ppf(v, nu)
        return stats.t.cdf(stats.t.ppf(w, nu + 1) * np.sqrt((nu + y * y) * (1 - r * r) / (nu + 1)) + r * y, nu)
    lo, hi = np.zeros_like(w), np.ones_like(w)
    for _ in range(iters):
        mid = 0.5 * (lo + hi); below = cop_h(fam, par, mid, v) < w
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def cop_fit(fam: str, u: np.ndarray, v: np.ndarray) -> tuple[tuple, float, float]:
    """Maximum likelihood (Gaussian: normal-scores correlation; t: Kendall-tau correlation, then nu); returns (par, loglik, AIC)."""
    nll = lambda par: -float(np.sum(cop_logpdf(fam, par, u, v)))  # noqa: E731
    if fam == 'gaussian':
        par = (float(np.clip(np.corrcoef(stats.norm.ppf(_cl(u)), stats.norm.ppf(_cl(v)))[0, 1], -0.99, 0.99)),)
    elif fam == 'student-t':
        tau = stats.kendalltau(u[:20_000], v[:20_000])[0]; r = float(np.clip(np.sin(np.pi * tau / 2), -0.99, 0.99))
        sol = minimize_scalar(lambda lnu: nll((r, np.exp(lnu))), bounds=(np.log(2.1), np.log(60.0)), method='bounded', options={'xatol': 1e-3})
        par = (r, float(np.exp(sol.x)))
    else:
        lo, hi = CV_BOUNDS[fam.replace('survival ', '')]
        sol = minimize_scalar(lambda th: nll((th,)) if abs(th) > 1e-6 else 1e300, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
        par = (float(sol.x),)
    ll = -nll(par)
    return par, ll, 2 * len(par) - 2 * ll


def cop_tail(fam: str, par: tuple) -> tuple[float, float]:
    """Limiting lower and upper tail dependence implied by a family."""
    if fam == 'student-t':
        r, nu = par; l_ = float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1)); return l_, l_
    th = par[0]
    return {'clayton': (2 ** (-1 / th), 0.0), 'survival clayton': (0.0, 2 ** (-1 / th)), 'gumbel': (0.0, 2 - 2 ** (1 / th)), 'survival gumbel': (2 - 2 ** (1 / th), 0.0)}.get(fam, (0.0, 0.0))


def ecdf_map(train: np.ndarray):
    """Point-in-time margin: the training sample's empirical CDF, mid-ranks, linearly interpolated, clamped inside (0, 1)."""
    xs = np.sort(np.asarray(train, float)[np.isfinite(train)]); n = len(xs); ps = (np.arange(n) + 0.5) / n
    return lambda x: np.interp(np.asarray(x, float), xs, ps, left=0.5 / n, right=1 - 0.5 / n)


# ---- standardised residual z = e / trailing sd of the bond's earlier residuals; stale marks and dispersion days excluded
_z = resid[['cusip', 'date', 'pit_residual', 'resid_sd']].merge(model[['cusip', 'date', 'mark_unchanged']], on=['cusip', 'date'], how='left').sort_values(['cusip', 'date'], kind='stable')
_z['s_prev'] = _z.groupby('cusip', observed=True)['resid_sd'].shift(1)
_z['z'] = _z['pit_residual'] / _z['s_prev'].clip(lower=0.25)
_drop = (_z['mark_unchanged'] == 1) | _z['date'].isin(DISPERSION_DAYS) | _z['s_prev'].isna()
print(f'standardised residuals: {len(_z):,} bond-days | excluded {_drop.mean():.1%} (stale mark {(_z["mark_unchanged"] == 1).mean():.1%}, dispersion day {_z["date"].isin(DISPERSION_DAYS).mean():.1%}, no trailing sd yet {_z["s_prev"].isna().mean():.1%})')
_z.loc[_drop, 'z'] = np.nan
CV_Z = rv_wide(_z, 'z')

# ---- the three nodes per bond-day: U the bond, P its frozen Section 11 peer basket (>= RV_MIN_PEERS marked), G the day's mean
_form_d0 = np.array([F_['d0'] for F_ in RV_FORMS]); _form_at = np.searchsorted(_form_d0, np.arange(RV_ND), side='right') - 1
_cv_parts = []
for d in range(RV_ND):
    if _form_at[d] < 0:
        continue
    F_ = RV_FORMS[_form_at[d]]; ids, peers = F_['ids'], F_['peers']
    zd = CV_Z[d]; zp = zd[peers]; cnt = np.isfinite(zp).sum(axis=1)
    P_ = np.full(len(ids), np.nan); okp = cnt >= RV_MIN_PEERS
    P_[okp] = np.nanmean(zp[okp], axis=1)
    U_ = zd[ids]; ok = np.isfinite(U_) & np.isfinite(P_)
    if not ok.any():
        continue
    f1 = (RV_CUME[d + 1, ids[ok]] - RV_CUME[d, ids[ok]]) if d + 1 < RV_ND else np.full(ok.sum(), np.nan)
    f5 = (RV_CUME[d + 5, ids[ok]] - RV_CUME[d, ids[ok]]) if d + 5 < RV_ND else np.full(ok.sum(), np.nan)
    _cv_parts.append(pd.DataFrame({'d': d, 'bi': ids[ok], 'zu': U_[ok], 'zp': P_[ok], 'zg': float(np.nanmean(zd)), 'group': RV_CS[d, ids[ok]], 'fwd1': f1, 'fwd5': f5}))
CV = pd.concat(_cv_parts, ignore_index=True)
CV = CV[np.isfinite(CV['group'])].reset_index(drop=True)
CV['date'] = RV_DATES[CV['d'].to_numpy()]; CV['month'] = CV['date'].dt.to_period('M')
_cvc = CV.groupby('group').apply(lambda g: pd.Series({'bond-days': len(g), 'corr(U, P)': g['zu'].corr(g['zp'], method='spearman'), 'corr(U, G)': g['zu'].corr(g['zg'], method='spearman'),
                                                       'corr(P, G)': g['zp'].corr(g['zg'], method='spearman')}), include_groups=False)
_cvc.index = [CS_NAMES[int(c)] for c in _cvc.index]
print(f'vine panel: {len(CV):,} bond-days on {CV["d"].nunique()} dates, months {sorted(CV["month"].astype(str).unique())}; rank correlations of the three nodes by call structure:'); display(_cvc.round(3))

# %%
CELL_T('12. Residual dependence: a factor-structured vine copula [2]')
def vine_eval(vine: dict, u, p, g) -> tuple[np.ndarray, np.ndarray]:
    """Log density of the C-vine (G root) per row and the bond's conditional PIT h(U | P, G)."""
    (fu, pu), (fp, pp), (f2, p2) = vine['bond - global'], vine['peers - global'], vine['bond - peers | global']
    a, b = cop_h(fu, pu, u, g), cop_h(fp, pp, p, g)
    ll = cop_logpdf(fu, pu, u, g) + cop_logpdf(fp, pp, p, g) + cop_logpdf(f2, p2, a, b)
    return ll, cop_h(f2, p2, a, b)


def vine_simulate(vine: dict, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw (U, P) from the vine: G uniform, P from h_PG inverse, the tree-2 pseudo-observation, then U from h_UG inverse."""
    (fu, pu), (fp, pp), (f2, p2) = vine['bond - global'], vine['peers - global'], vine['bond - peers | global']
    w1, w2, w3 = rng.uniform(size=(3, n))
    p = cop_hinv(fp, pp, w2, w1); a = cop_hinv(f2, p2, w3, w2)
    return cop_hinv(fu, pu, a, w1), p


def tail_dep(u: np.ndarray, p: np.ndarray, q: float) -> tuple[float, float]:
    """Finite-q tail dependence: P(U < q | P < q) and P(U > 1 - q | P > 1 - q)."""
    lo, hi = p < q, p > 1 - q
    return (float((u[lo] < q).mean()) if lo.sum() >= 20 else np.nan), (float((u[hi] > 1 - q).mean()) if hi.sum() >= 20 else np.nan)


def _vine_walk_forward() -> dict:
    rng = np.random.default_rng(CFG.seed)
    months_v = sorted(CV['month'].unique()); par_rows, ll_parts, tail_rows, cond_parts = [], [], [], []
    for m in months_v[1:]:
        t0 = time.perf_counter()
        tr_all, te_all = CV[CV['month'] < m], CV[CV['month'] == m]
        Fg = ecdf_map(tr_all.drop_duplicates('d')['zg'].to_numpy())
        for c in sorted(CV['group'].unique()):
            tr, te = tr_all[tr_all['group'] == c], te_all[te_all['group'] == c]
            if len(tr) < 20_000 or len(te) < 2_000:
                continue
            Fu, Fp = ecdf_map(tr['zu'].to_numpy()), ecdf_map(tr['zp'].to_numpy())
            sub = tr.sample(min(len(tr), CV_MAX_FIT), random_state=CFG.seed)
            U, P, Gv = Fu(sub['zu']), Fp(sub['zp']), Fg(sub['zg'])
            cname = CS_NAMES[int(c)]; vines = {'Gaussian vine': {}, 'selected vine': {}}

            def _fit_edge(edge: str, a: np.ndarray, b: np.ndarray, families=CV_FAMILIES) -> dict:
                res = {f: cop_fit(f, a, b) for f in families}; best = min(res, key=lambda f: res[f][2])
                tau = float(stats.kendalltau(a[:20_000], b[:20_000])[0])
                for f, (par, ll, aic) in res.items():
                    lt = cop_tail(f, par)
                    par_rows.append({'month': str(m), 'call_structure': cname, 'edge': edge, 'family': f, 'params': tuple(round(x, 4) for x in par), 'loglik per obs': ll / len(a),
                                     'AIC gap to best': aic - res[best][2], 'selected': f == best, 'kendall tau (data)': tau, 'lambda_L': lt[0], 'lambda_U': lt[1]})
                return {f: r_[0] for f, r_ in res.items()} | {'_best': best}

            for edge, a, b in [('bond - global', U, Gv), ('peers - global', P, Gv)]:
                fits = _fit_edge(edge, a, b)
                vines['Gaussian vine'][edge] = ('gaussian', fits['gaussian']); vines['selected vine'][edge] = (fits['_best'], fits[fits['_best']])
            for vn, fams in [('Gaussian vine', ('gaussian',)), ('selected vine', CV_FAMILIES)]:   # tree 2 on each vine's own pseudo-observations
                (fu, pu), (fp, pp) = vines[vn]['bond - global'], vines[vn]['peers - global']
                fits = _fit_edge(f'bond - peers | global ({vn})', cop_h(fu, pu, U, Gv), cop_h(fp, pp, P, Gv), fams)
                vines[vn]['bond - peers | global'] = (fits['_best'], fits[fits['_best']])
            ut, pt, gt = Fu(te['zu']), Fp(te['zp']), Fg(te['zg'])
            out = te[['d', 'bi', 'group', 'fwd1', 'fwd5']].copy(); out['u'] = ut
            for vn, vine in vines.items():
                ll, cp = vine_eval(vine, ut, pt, gt)
                ll_parts.append(pd.DataFrame({'d': te['d'].to_numpy(), 'll': ll}).groupby('d')['ll'].agg(['sum', 'size']).reset_index().assign(group=cname, vine=vn, month=str(m)))
                out[f'cpit {vn}'] = cp
                us, ps = vine_simulate(vine, CV_NSIM, rng)
                for q in CV_Q:
                    lo_, hi_ = tail_dep(us, ps, q)
                    tail_rows.append({'month': str(m), 'call_structure': cname, 'q': q, 'source': vn, 'lower': lo_, 'upper': hi_})
            for q in CV_Q:
                lo_, hi_ = tail_dep(ut, pt, q)
                tail_rows.append({'month': str(m), 'call_structure': cname, 'q': q, 'source': 'empirical (test month)', 'lower': lo_, 'upper': hi_})
            cond_parts.append(out)
            print(f'  vine {m} | {cname}: train {len(tr):,} test {len(te):,} | selected {[v_[0] for v_ in vines["selected vine"].values()]}')
        print(f'  vine, test month {m}: {time.perf_counter() - t0:.0f}s')
    return {'params': pd.DataFrame(par_rows), 'll': pd.concat(ll_parts, ignore_index=True), 'tails': pd.DataFrame(tail_rows), 'cond': pd.concat(cond_parts, ignore_index=True)}


VC = cached('lab_vine_copula', _vine_walk_forward,
            deps=[STAGE_KEYS['walk_forward'], RV_K, RV_MIN_PEERS, CV_MAX_FIT, CV_NSIM, repr(CV_Q), repr(CV_FAMILIES), len(CV), float(np.nansum(CV_Z))],
            code=[cop_logpdf, cop_h, cop_hinv, cop_fit, cop_tail, ecdf_map, vine_eval, vine_simulate, tail_dep])

# %%
CELL_T('12. Residual dependence: a factor-structured vine copula [3]')
vp = VC['params']
_sel = vp[vp['selected']].copy()
print('Families chosen by AIC per edge, call structure and fold (the Gaussian-vine tree-2 row is Gaussian by construction):')
display(_sel.pivot_table(index=['call_structure', 'edge'], columns='month', values='family', aggfunc='first'))
_last = _sel[_sel['month'] == _sel['month'].max()].set_index(['call_structure', 'edge'])[['family', 'params', 'kendall tau (data)', 'lambda_L', 'lambda_U', 'loglik per obs']]
print('Last fold, selected families: parameters, Kendall tau of the pseudo-observations, implied limiting tail dependence:'); display(_last.round(4))
_gap = vp[vp['family'] == 'gaussian'].set_index(['month', 'call_structure', 'edge'])['AIC gap to best']
print(f'AIC gap of the Gaussian family to the selected one, median over folds and groups, by edge (0 = Gaussian chosen):'); display(_gap.groupby(level='edge').median().round(0).to_frame('median AIC gap'))

# ---- (1) is the dependence non-Gaussian? test-month log-likelihood per bond-day, selected vine minus Gaussian vine, paired by date
_ll = VC['ll'].groupby(['d', 'vine'])[['sum', 'size']].sum().unstack('vine')
_gain = (_ll[('sum', 'selected vine')] - _ll[('sum', 'Gaussian vine')]) / _ll[('size', 'selected vine')]
_gain.index = RV_DATES[_gain.index.to_numpy()]
_gm = _gain.groupby(_gain.index.to_period('M')).mean()
ll_tab = pd.DataFrame({'gain per bond-day (nats)': [_gain.mean()], 'block-bootstrap t': [block_boot_t(_gain.to_numpy())], 'months positive': [int((_gm > 0).sum())], 'months': [len(_gm)], 'test days': [len(_gain)]})
_llg = VC['ll'].groupby(['group', 'vine'])[['sum', 'size']].sum(); _llg = (_llg['sum'] / _llg['size']).unstack('vine'); _llg['gain'] = _llg['selected vine'] - _llg['Gaussian vine']
print('Out of sample: test-month log-likelihood of the selected vine over the Gaussian vine:'); display(ll_tab.round(4)); display(_llg.round(4))

# ---- (2) risk: finite-q tail dependence of the bond on its peers, implied vs realised in the test month
tl = VC['tails'].melt(id_vars=['month', 'call_structure', 'q', 'source'], value_vars=['lower', 'upper'], var_name='tail').pivot_table(index=['call_structure', 'q', 'tail', 'month'], columns='source', values='value')
_emp = 'empirical (test month)'
tail_err = pd.DataFrame({vn: (tl[vn] - tl[_emp]).abs().groupby(level=['q', 'tail']).mean() for vn in ['Gaussian vine', 'selected vine']})
tail_mean = tl.groupby(level=['call_structure', 'q', 'tail']).mean()
print('P(bond in its own q-tail | peers in theirs): implied by each vine (simulated) against the test month, mean over folds:'); display(tail_mean.round(3))
print('Mean absolute tail-dependence calibration error by q and tail (lower = better):'); display(tail_err.round(4))
_risk_pass = bool(ll_tab['block-bootstrap t'].iloc[0] > 3 and ll_tab['months positive'].iloc[0] >= 4 and tail_err['selected vine'].mean() < tail_err['Gaussian vine'].mean())

# ---- (3) relative value: how unusual the bond's move is given its peers and the market, as a reversal signal
cond = VC['cond'].copy()
SIGS_V = {'unconditional PIT': 'u', 'Gaussian-vine conditional PIT': 'cpit Gaussian vine', 'selected-vine conditional PIT': 'cpit selected vine'}
for lab, col in SIGS_V.items():
    cond[lab] = 0.5 - cond[col]          # > 0: moved down in yield more than expected, expect the residual to come back up (reversal)
sig_rows, ics = [], {}
for h, out in [(1, 'fwd1'), (5, 'fwd5')]:
    for lab in SIGS_V:
        f_ = cond[['d', lab, out]].dropna()
        ic = f_.groupby('d').apply(lambda g: g[lab].corr(g[out], method='spearman') if len(g) >= 200 else np.nan, include_groups=False).dropna()
        ic.index = RV_DATES[ic.index.to_numpy()]; icm = ic.groupby(ic.index.to_period('M')).mean(); ics[(h, lab)] = ic
        sig_rows.append({'h (obs)': h, 'signal': lab, 'IC': ic.mean(), 'IC t (block)': block_boot_t(ic.to_numpy(), block=max(CFG.block_days, h)), 'months IC > 0': int((icm > 0).sum()), 'months': len(icm)})
sig_tab = pd.DataFrame(sig_rows).set_index(['h (obs)', 'signal'])
for h in (1, 5):
    d_ = (ics[(h, 'selected-vine conditional PIT')] - ics[(h, 'Gaussian-vine conditional PIT')]).dropna()
    sig_tab.loc[(h, 'selected-vine conditional PIT'), 'paired IC vs Gaussian vine, t (block)'] = block_boot_t(d_.to_numpy(), block=max(CFG.block_days, h))
print('Conditional PIT as a residual reversal signal (daily rank IC with the bond\'s next 1 and 5 residuals; positive = reversal):'); display(sig_tab.round(4))
cond['decile'] = pd.qcut(cond['cpit selected vine'].rank(method='first'), 10, labels=False) + 1
cond['decile gaussian'] = pd.qcut(cond['cpit Gaussian vine'].rank(method='first'), 10, labels=False) + 1
dec_v = pd.DataFrame({'selected vine': cond.groupby('decile')['fwd5'].mean(), 'Gaussian vine': cond.groupby('decile gaussian')['fwd5'].mean()})
print('Mean next-5 residual (bp) by decile of the conditional PIT (1 = moved down in yield most unusually given its peers):'); display(dec_v.round(3))
_s5 = sig_tab.loc[(5, 'selected-vine conditional PIT')]
_sig_pass = bool(_s5['IC t (block)'] > 3 and _s5['months IC > 0'] >= 4 and _s5.get('paired IC vs Gaussian vine, t (block)', np.nan) > 2)
vine_verdict = (f'risk bar {"PASSED" if _risk_pass else "not passed"} (log-lik gain t {ll_tab["block-bootstrap t"].iloc[0]:+.1f}, {ll_tab["months positive"].iloc[0]} of {ll_tab["months"].iloc[0]} months; '
                f'tail calibration error {tail_err["selected vine"].mean():.3f} vs Gaussian {tail_err["Gaussian vine"].mean():.3f}); '
                f'signal bar {"PASSED" if _sig_pass else "not passed"} (5-obs IC {_s5["IC"]:+.3f}, t {_s5["IC t (block)"]:+.1f}, paired vs Gaussian t {_s5.get("paired IC vs Gaussian vine, t (block)", np.nan):+.1f})')
print('verdict (lab_vine_copula):', vine_verdict)

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
_tq = tail_mean.xs(0.05, level='q')
_tq.unstack('tail').plot.barh(ax=ax[0, 0]); ax[0, 0].set_title('P(bond in its 5% tail | peers in theirs): empirical vs implied (mean over folds)', fontsize=10); ax[0, 0].tick_params(axis='y', labelsize=8); ax[0, 0].legend(fontsize=6); ax[0, 0].set_ylabel('')
ax[0, 1].plot(_gain.index, _gain.cumsum().to_numpy(), color='#C44E52'); ax[0, 1].axhline(0, color='k', lw=0.6); ax[0, 1].set_title('Cumulative out-of-sample log-likelihood gain per bond-day, selected over Gaussian vine (nats)', fontsize=10)
for lab, c_ in zip(SIGS_V, ['#8C8C8C', '#4C72B0', '#C44E52']):
    s_ = ics[(5, lab)]; ax[1, 0].plot(s_.index, s_.rolling(10, min_periods=3).mean().to_numpy(), color=c_, label=lab)
ax[1, 0].axhline(0, color='k', lw=0.6); ax[1, 0].set_title('Daily rank IC with the next-5 residual, 10-day mean', fontsize=10); ax[1, 0].legend(fontsize=7)
dec_v.plot(ax=ax[1, 1], marker='o'); ax[1, 1].axhline(0, color='k', lw=0.6); ax[1, 1].set_xlabel('conditional PIT decile'); ax[1, 1].set_ylabel('mean next-5 residual (bp)'); ax[1, 1].set_title('Reversal by conditional PIT decile', fontsize=10)
savefig('12_vine_copula')
record('lab_vine_copula', selected_families=_sel.drop(columns=['selected']).astype({'params': str}).round(5).to_dict(orient='records'), loglik_gain=flat_records(ll_tab), loglik_by_group=flat_records(_llg),
       tails=flat_records(tail_mean), tail_calibration_error=flat_records(tail_err), signal=flat_records(sig_tab), deciles=flat_records(dec_v), verdict=vine_verdict)

# %% [markdown]
# ## 13. RFQ P&L: the relative-value signal on the desk's requests
#
# **The question.** Sections 7b and 11 score the residual signals against evaluated marks on the whole covered universe.
# The desk does not trade the universe: it trades the requests it receives, at its own quote, and is filled when its
# quote is the best. This section asks whether the signals pick which of those requests make money, by how much in
# dollars, and whether the desk could act on them with a quote skew or a holding rule.
#
# **Population.** The `track_rfq` ledger (one row per request: the final state, quoted, no pricing error, held-out
# window). Column roles are detected from the names and can be overridden in `TRACK_RFQ_COLUMNS`. The *filled* sample
# is the logged won flag where the track carries one, else the crossing proxy (a print crossed our quote). Without the
# track (`HAS_RFQ = False`) the section runs on the matched customer prints as a proxy population, labelled as such;
# the skew counterfactual needs our own quote and is skipped.
#
# **Signals.** All three come from the last close strictly before the request date. Each is oriented so that positive means
# the bond is expected to richen (yield to fall against its hedge), and then multiplied by the side $s$ (+1 bid, -1 offer).
#
# | signal | definition | source |
# |---|---|---|
# | `sig_ss` | $-\hat\varepsilon_{t+1\mid t}$, the state-space forecast of the next residual | 7b, signal of record |
# | `sig_level` | $-\sum_{20}\varepsilon / (\hat\sigma_\varepsilon\sqrt{n})$, the trailing 20-day cumulative residual over its volatility | 7 (level signal), as a z-score |
# | `sig_gap` | spread to MMD at own tenor minus the median spread of the 20 nearest beta-space peers within the call structure | 11 |
#
# **P&L of a fill at the entry yield** is $s \times 100\,(y_{\text{entry}} - y_{\text{ref}})$ in bp of yield. The entry is our
# quote (the track's optimal yield), or the print in the proxy population. There are three references, each at
# $h \in \{1, 5, 10, 20\}$ business days after the request date's previous close.
#
# - **The evaluated mark.** Raw, and factor-hedged: $+\,s\times$ the move the record explains (the yield change minus the
# residual of record). For priced-to-call bonds the hedge also adds 0.9 times the maturity-minus-worst MMD move,
# which is Section 9's blended-tenor hedge.
# - **The bond's own next print.** The last exit-side customer print after the request time and before the close of day $h$.
# It is reported raw and hedged to the print's date.
# - **The peer-basket print.** The median print-implied move of the frozen Section 11 basket, using at least 3 printing
# peers. This reference is peer-hedged by construction.
#
# **Analyses**
#
# 1. *Selection.* Each week, requests are sorted by the oriented signal into quintiles. The table reports the mean
# P&L by quintile, the weekly Q5 - Q1 spread, and a Fama-MacBeth slope per one standard deviation of the signal,
# with side, log size and the concession to the mark as controls. Inference uses Newey-West t over weeks, with lags
# ceil(h / 5) + 1 because horizons overlap. A within-week permutation of the signal serves as the placebo.
# 2. *Skew counterfactual.* Our quote moves toward the customer by $k = m \times \text{clip}(\hat\beta\,(x - \bar x), \pm 5)$ bp,
# with $\hat\beta$ and $\bar x$ estimated on earlier months only, for skew multiples $m \in \{0, 0.25, 0.5, 1, 2\}$.
# A request fills if its print crosses the shifted quote, and a fill earns its P&L minus $k$.
# 3. *Inventory rule.* Hold a fill to $h$ if the oriented signal is positive, else exit at $h = 1$. The rule is
# compared with always exiting at $h = 1$.
# 4. *Segmentation.* Q5 - Q1 at $h = 5$ is broken down by side, size bin, call structure and mark staleness.
#
# **Guards.**
#
# - An assertion checks that every signal date precedes the request date.
# - Concessions beyond 100 bp are dropped as unit errors.
# - Weeks with fewer than 50 requests are dropped.
# - The placebo is required to sit near zero.
# - The own-print reference is required to agree in sign with the mark reference.
# - Dollars use DV01 = par x modified duration x 1e-4.
#
# **Bar** (`lab_rfq_pnl`, fixed before the run). The three signals are one family, and each is tested on its own:
#
# - *Selection.* A signal passes if, on the filled requests at $h = 5$ against the hedged mark, its Q5 - Q1 has
# NW t ≥ 3, is positive in ≥ 4 held-out months, and has the same sign on the own-print reference.
# - *Skew.* A skew passes if its weekly paired P&L per request against $m = 0$ has NW t ≥ 3 and is positive in
# ≥ 4 months.
# - *Inventory rule.* The same bar, measured against always exiting at $h = 1$.
#
# The live A/B design and its power calculation are in `docs/rfq_skew_experiment.md`, and the reading notes are in
# `docs/rfq_pnl_section_notes.md`.

# %%
CELL_T('13. RFQ P&L: the RV signal on the desk requests [1]')
PRE_REGISTERED.update({
    'lab_rfq_pnl': ('fixed 2026-10-14 before the lab run: requests = track_rfq final, quoted, no pricing error, held-out window (else the matched customer prints, labelled); '
                    'signals at the last close strictly before the request date, positive = expected to richen, times the side s (+1 bid, -1 offer): sig_ss = -ssm_signal (7b), '
                    'sig_level = -(20-day cumulative residual) / (its sd x sqrt(n)), sig_gap = spread to MMD at own tenor minus the median of the 20 beta-space peers (11); '
                    'P&L of a fill at the entry yield = s x 100 (entry - reference) bp at h = 1, 5, 10, 20 business days: the evaluated mark (raw; hedged = + s x (yield change - residual of record + 0.9 x maturity-minus-worst MMD move for priced-to-call)), '
                    'the last exit-side print of the bond after the request within h (raw, hedged), the peer basket print-implied move (>= 3 peers); '
                    'selection = weekly quintiles of the oriented signal and a weekly Fama-MacBeth slope (controls side, log size, concession), Newey-West t over weeks with ceil(h / 5) + 1 lags; '
                    'bar per signal (family of three): filled requests, hedged mark, h = 5: Q5 - Q1 NW t >= 3, positive in >= 4 held-out months, same sign on the hedged own print; '
                    'skew: quote moved toward the customer by m x clip(beta (x - mean x), +/- 5 bp), beta and mean from earlier months, m in {0, 0.25, 0.5, 1, 2}, fill iff the print crosses the shifted quote; '
                    'passes if the weekly paired P&L per request vs m = 0 has NW t >= 3 and is positive in >= 4 months; inventory: hold to h if the oriented signal > 0 else exit at h = 1, same bar against always exit at 1'),
})
RQ_H = (1, 5, 10, 20)
RQ_SIGS = {'x_ss': 'state space (-ssm_signal)', 'x_level': 'level (-20d cum. residual / vol)', 'x_gap': 'RV gap to peers'}
RQ_REFS = {'mark_hedged': 'evaluated mark, hedged', 'mark_raw': 'evaluated mark, raw', 'own_hedged': 'own exit print, hedged', 'own_raw': 'own exit print, raw', 'peer': 'peer basket print'}
RQ_SKEW = (0.0, 0.25, 0.5, 1.0, 2.0)
RQ_PTC_BLEND = 0.9
RQ_LEVEL_W = CFG.level_signal_window
RQ_MIN_PEER_PRINTS, RQ_MIN_WEEK, RQ_BAR_T, RQ_BAR_MONTHS = 3, 50, 3.0, 4
RQ_Z_POWER = 1.959964 + 0.841621        # two-sided 5% and 80% power


def nw_t(x, lags: int | None = None) -> float:
    """Newey-West (Bartlett kernel) t statistic of the mean of a series; lags default floor(4 (n / 100)^(2/9))."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 4:
        return np.nan
    L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else int(lags)
    u = x - x.mean(); v = float(u @ u) / n
    for l in range(1, min(L, n - 1) + 1):
        v += 2.0 * (1.0 - l / (L + 1.0)) * float(u[l:] @ u[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def rq_lags(h: int) -> int:
    return int(np.ceil(h / 5.0)) + 1


def last_obs_index(M: np.ndarray) -> np.ndarray:
    """Per column, the row index of the last finite value at or before each row (-1 before the first)."""
    idx = np.where(np.isfinite(M), np.arange(M.shape[0])[:, None], -1)
    return np.maximum.accumulate(idx, axis=0)


def rfq_signals(dq: np.ndarray, bi: np.ndarray, SSM: np.ndarray, E: np.ndarray, GAP: np.ndarray, w: int) -> dict:
    """The three signals at row dq - 1 (the last close strictly before the request date), positive = expected to richen."""
    d = dq - 1
    rows = d[:, None] - np.arange(w)[::-1][None, :]
    Ew = np.where(rows >= 0, E[np.maximum(rows, 0), bi[:, None]], np.nan)
    n = np.isfinite(Ew).sum(axis=1); cum = np.nansum(Ew, axis=1)
    sd = np.sqrt(np.nansum((Ew - (cum / np.maximum(n, 1))[:, None]) ** 2, axis=1) / np.maximum(n - 1, 1))
    level = np.where((n >= max(5, w // 2)) & (sd > 0), -cum / (np.where(sd > 0, sd, 1.0) * np.sqrt(np.maximum(n, 1))), np.nan)
    return {'sig_ss': -SSM[d, bi], 'sig_level': level, 'sig_gap': GAP[d, bi]}


def rfq_peer_matrix(dq: np.ndarray, bi: np.ndarray, forms: list[dict], nb: int, k: int) -> np.ndarray:
    """The frozen basket of the latest weekly formation on or before the request's previous close (-1 where the bond has none)."""
    P = np.full((len(dq), k), -1, dtype=np.int64)
    if not forms:
        return P
    d0s = np.array([F_['d0'] for F_ in forms]); fi = np.searchsorted(d0s, dq - 1, side='right') - 1
    for f in np.unique(fi[fi >= 0]):
        F_ = forms[f]; pos = np.full(nb, -1, dtype=np.int64); pos[F_['ids']] = np.arange(len(F_['ids']))
        r = np.flatnonzero(fi == f); row = pos[bi[r]]; ok = row >= 0
        P[r[ok]] = F_['peers'][row[ok]]
    return P


def rfq_outcomes(rq: pd.DataFrame, peers: np.ndarray, M: dict, prints: pd.DataFrame | None, H=RQ_H, blend: float = RQ_PTC_BLEND, min_peers: int = RQ_MIN_PEER_PRINTS) -> pd.DataFrame:
    """P&L (bp of yield) of a fill at y_entry, s x 100 (y_entry - reference), per horizon h = the close of panel row dq - 1 + h
    (h = 1: the request date's own close), and reference: the evaluated mark (raw; hedged = + s x the move the record explains
    from the previous close), the bond's last exit-side print after the request and before that close ends (raw; hedged to the
    print's date), and the median print-implied move of the peer basket from the previous close (peer-hedged by construction)."""
    dq = rq['dq'].to_numpy(np.int64); bi = rq['bi'].to_numpy(np.int64); s = rq['s'].to_numpy(float); ye = rq['y_entry'].to_numpy(float)
    ptc = rq['ptc'].to_numpy(float); Y, CT, CE, CD, dates = M['Y'], M['CT'], M['CE'], M['CD'], M['dates']; ND = Y.shape[0]
    d0 = dq - 1; y0 = Y[d0, bi]

    def sysmove(de: np.ndarray) -> np.ndarray:
        """The move of the bond the record explains (factor + MMD at worst tenor, + blend x maturity-minus-worst for priced-to-call), close d0 to close de."""
        de = np.clip(de, 0, ND - 1)
        return (CT[de, bi] - CT[d0, bi]) - (CE[de, bi] - CE[d0, bi]) + blend * ptc * (CD[de, bi] - CD[d0, bi])

    out = {}
    if prints is not None and len(prints):
        left = pd.DataFrame({'_i': np.arange(len(rq)), 'cusip': rq['cusip'].astype(str).to_numpy(), 'side': np.where(s > 0, 'S', 'P'), 'ts_req': rq['ts'].to_numpy().astype('datetime64[ns]')})
        right = prints[['cusip', 'side', 'ts', 'y']].astype({'ts': 'datetime64[ns]'}).sort_values('ts')   # merge_asof needs one datetime unit on both keys
    for h in H:
        de = d0 + h; okh = de < ND; dec = np.minimum(de, ND - 1)
        ym = np.where(okh, Y[dec, bi], np.nan)
        raw = s * 100.0 * (ye - ym)
        out[f'mark_raw_{h}'] = raw; out[f'mark_hedged_{h}'] = raw + s * np.where(okh, sysmove(dec), np.nan)
        if prints is not None and len(prints):
            yp = np.full(len(rq), np.nan); dp = np.full(len(rq), -1, dtype=np.int64)
            if okh.any():
                l_ = left[okh].assign(key=(dates[dec[okh]] + pd.Timedelta(days=1)).to_numpy().astype('datetime64[ns]')).sort_values('key')
                j = pd.merge_asof(l_, right, left_on='key', right_on='ts', by=['cusip', 'side'], direction='backward', allow_exact_matches=False)
                j = j[j['ts'].notna() & (j['ts'] > j['ts_req'])]; ii = j['_i'].to_numpy(np.int64)
                yp[ii] = j['y'].to_numpy(float); dp[ii] = dates.searchsorted(pd.DatetimeIndex(j['ts']).normalize(), side='right') - 1
            raw_o = s * 100.0 * (ye - yp)
            out[f'own_raw_{h}'] = raw_o; out[f'own_hedged_{h}'] = raw_o + s * np.where(dp >= 0, sysmove(np.maximum(dp, d0)), np.nan)
        if M.get('PY') is not None:
            pc = np.maximum(peers, 0); Lp = M['LASTP'][dec[:, None], pc].astype(np.int64)
            okp = (peers >= 0) & (Lp >= dq[:, None]) & okh[:, None]
            mv = np.where(okp, 100.0 * (M['PY'][np.maximum(Lp, 0), pc] - Y[d0[:, None], pc]), np.nan)
            npk = np.isfinite(mv).sum(axis=1); med = np.full(len(rq), np.nan); sel = npk >= min_peers
            if sel.any():
                med[sel] = np.nanmedian(mv[sel], axis=1)
            out[f'peer_{h}'] = s * (100.0 * (ye - y0) - med); out[f'peer_n_{h}'] = npk
    return pd.DataFrame(out, index=rq.index)


def rfq_quintiles(f: pd.DataFrame, x: str, y: str, h: int, min_n: int = RQ_MIN_WEEK) -> tuple[dict, pd.Series]:
    """Weekly quintiles of the oriented signal x: mean P&L by quintile, the weekly Q5 - Q1 spread, its NW t and month signs."""
    g = f[['week', x, y]].dropna()
    g = g[g.groupby('week')[x].transform('size') >= min_n]
    if g['week'].nunique() < 4:
        return {}, pd.Series(dtype=float)
    q = np.ceil(g.groupby('week')[x].rank(pct=True, method='first') * 5).clip(1, 5).astype(int).rename('q')
    m = g.groupby(['week', q])[y].mean().unstack('q')
    if 1 not in m.columns or 5 not in m.columns:
        return {}, pd.Series(dtype=float)
    spr = (m[5] - m[1]).dropna(); spr.index = pd.DatetimeIndex(spr.index)
    mm = spr.groupby(spr.index.to_period('M')).mean()
    res = {f'Q{i}': float(m[i].mean()) for i in range(1, 6) if i in m.columns}
    res.update({'Q5 - Q1 (bp)': float(spr.mean()), 'NW t': nw_t(spr.to_numpy(), rq_lags(h)), 'weeks': int(len(spr)), 'months +': int((mm > 0).sum()), 'months': int(len(mm)), 'requests': int(len(g))})
    return res, spr


def rfq_fm(f: pd.DataFrame, x: str, y: str, controls: list[str], min_n: int = RQ_MIN_WEEK) -> pd.Series:
    """Weekly cross-sectional OLS of P&L on the standardised oriented signal and controls: the signal's slope (bp per sd) by week."""
    g = f[['week', x, y] + controls].dropna(); out = {}
    for wk, gw in g.groupby('week', sort=True):
        xv = gw[x].to_numpy(float); sd = xv.std()
        if len(gw) < min_n or not sd > 0:
            continue
        cols = [np.ones(len(gw)), (xv - xv.mean()) / sd] + [gw[c].to_numpy(float) for c in controls if gw[c].std() > 0]
        out[wk] = float(np.linalg.lstsq(np.column_stack(cols), gw[y].to_numpy(float), rcond=None)[0][1])
    return pd.Series(out, dtype=float)


def rfq_skew_frame(f: pd.DataFrame, x: str, y: str, grid=RQ_SKEW, clip_bp: float = 5.0, min_prior: int = 500) -> pd.DataFrame:
    """Per printed request: fill and P&L at each skew multiple m. The quote moves toward the customer by k = m x clip(beta (x - c), +/- clip) bp
    (k < 0: away), beta and c by least squares of y on x over the requests of earlier months only; fill iff o + k >= 0, a fill earns y - k."""
    g = f[['week', 'month', 'o', 'dv01', x, y]].dropna().copy()
    beta = pd.Series(np.nan, index=g.index); cen = pd.Series(np.nan, index=g.index)
    for m in sorted(g['month'].unique()):
        pr = g[g['month'] < m]
        if len(pr) < min_prior:
            continue
        xv = pr[x].to_numpy(float); yv = pr[y].to_numpy(float); c = xv.mean(); v = float(((xv - c) ** 2).mean())
        if v > 0:
            sel = g['month'] == m; beta[sel] = float((xv - c) @ (yv - yv.mean())) / len(xv) / v; cen[sel] = c
    g['k1'] = (beta * (g[x] - cen)).clip(-clip_bp, clip_bp)
    g = g.dropna(subset=['k1'])
    for m_ in grid:
        k = m_ * g['k1'].to_numpy(float); fill = (g['o'].to_numpy(float) + k) >= 0
        g[f'fill_{m_:g}'] = fill.astype(float); g[f'pnl_{m_:g}'] = np.where(fill, g[y].to_numpy(float) - k, 0.0)
    return g


def rfq_skew_summary(g: pd.DataFrame, h: int, grid=RQ_SKEW) -> pd.DataFrame:
    rows = []; base = g.groupby('week')['pnl_0'].mean(); nm = max(g['month'].nunique(), 1)
    for m_ in grid:
        c = f'pnl_{m_:g}'; d = (g.groupby('week')[c].mean() - base).dropna(); d.index = pd.DatetimeIndex(d.index); dm = d.groupby(d.index.to_period('M')).mean()
        rows.append({'skew multiple m': m_, 'requests': int(len(g)), 'fill rate': float(g[f'fill_{m_:g}'].mean()), 'P&L per request (bp)': float(g[c].mean()),
                     'P&L per fill (bp)': float(g.loc[g[f'fill_{m_:g}'] > 0, c].mean()), '$ per month ($k)': float((g[c] * g['dv01']).sum() / 1e3 / nm),
                     'vs m = 0 (bp per request)': float(d.mean()), 'NW t vs m = 0': nw_t(d.to_numpy(), rq_lags(h)) if m_ else np.nan,
                     'months above m = 0': int((dm > 0).sum()) if m_ else np.nan, 'months': int(len(dm))})
    return pd.DataFrame(rows).set_index('skew multiple m')


def rfq_inventory(f: pd.DataFrame, x: str, ref: str, h_hold: int, h_exit: int = 1) -> dict:
    """Hold a fill to h_hold when the oriented signal is positive, else exit at h_exit; the gain over always exiting at h_exit."""
    y1, yH = f'{ref}_{h_exit}', f'{ref}_{h_hold}'
    g = f[['week', 'month', x, y1, yH, 'dv01']].dropna()
    if g['week'].nunique() < 4:
        return {}
    gain = np.where(g[x].to_numpy(float) > 0, (g[yH] - g[y1]).to_numpy(float), 0.0)
    wk = pd.Series(gain, index=g.index).groupby(g['week']).mean(); wk.index = pd.DatetimeIndex(wk.index); mm = wk.groupby(wk.index.to_period('M')).mean()
    return {'fills': int(len(g)), 'held share': float((g[x] > 0).mean()), 'gain per fill (bp)': float(gain.mean()), 'always hold, gain per fill (bp)': float((g[yH] - g[y1]).mean()),
            'NW t': nw_t(wk.to_numpy(), rq_lags(h_hold)), 'months +': int((mm > 0).sum()), 'months': int(len(mm)), '$ per month ($k)': float((gain * g['dv01'].to_numpy(float)).sum() / 1e3 / max(g['month'].nunique(), 1))}


def rfq_power_n(sd: float, delta: float) -> float:
    """Requests per arm for a two-arm randomised test of a mean P&L difference delta (bp) with per-request sd (bp), two-sided 5%, power 80%."""
    return float(2.0 * RQ_Z_POWER ** 2 * sd ** 2 / delta ** 2) if delta and np.isfinite(delta) and np.isfinite(sd) else np.inf

# %%
CELL_T('13. RFQ P&L: the RV signal on the desk requests [2]')
TRACK_RFQ_COLUMNS: dict[str, str] = globals().get('TRACK_RFQ_COLUMNS', {})   # optional overrides by role, e.g. {'time': 'rfq_time', 'optimal_yield': 'opt_yld'}
RQ_CAND = {'time': ['rfq_received_time', 'rfq_time', 'rfq_ts', 'received_time', 'timestamp', 'ts', 'time', 'quote_time', 'signal_ts', 'signal_time', 'date'],
           'req_id': ['strategy_ecn_req_id', 'ecn_req_id', 'rfq_id', 'req_id', 'request_id'],
           'is_final': ['is_final'], 'quoted': ['quoted', 'is_quoted'], 'pricing_error': ['strategy_is_pricing_error', 'is_pricing_error', 'pricing_error'],
           'quantity': ['quantity', 'msrb_quantity', 'qty', 'size', 'par', 'par_amount', 'notional'],
           'algo_yield': ['algo_yield', 'algo_signal_yield', 'signal_yield', 'mid_yield', 'model_yield', 'algo_yld', 'yield_mid', 'fair_yield', 'mid'],
           'optimal_yield': ['optimal_yield', 'opt_yield', 'optimal_yld', 'opt_yld', 'quote_yield', 'our_yield', 'bid_yield', 'optimal'],
           'msrb_yield': ['msrb_yield', 'msrb_yld', 'print_yield', 'trade_yield'],
           'won': ['algo_won', 'won', 'is_won', 'filled', 'is_filled', 'traded', 'executed', 'win', 'done', 'hit'],
           'side': ['side', 'msrb_side', 'signal_side', 'quote_side', 'rfq_side', 'msrb_tradetype']}
RQ_SMAP = {'P': 'P', 'B': 'P', 'BUY': 'P', 'BID': 'P', 'PURCHASE': 'P', 'DEALER BUY': 'P', 'S': 'S', 'SELL': 'S', 'OFFER': 'S', 'ASK': 'S', 'O': 'S', 'DEALER SELL': 'S'}


def _rq_flag(v: pd.Series) -> np.ndarray:
    if v.dtype == object or str(v.dtype).startswith(('string', 'bool', 'category')):
        return v.astype('string').str.lower().isin(['true', '1', 'y', 'yes', 't', 'won', 'filled', 'done', 'win']).to_numpy(bool)
    return (pd.to_numeric(v, errors='coerce').fillna(0) > 0).to_numpy(bool)


HAS_RFQ, RQ_SOURCE, RQ_RC, rq0 = False, None, {}, pd.DataFrame()
_rdir = PIPE.path(PIPE.data_dir, 'track_rfq')
if _rdir.exists() and any(_rdir.glob('*.parquet')):
    _rcols = list(PIPE.store('track_rfq').dataset().schema.names); _rl = {str(c).lower(): c for c in _rcols}
    RQ_RC = {r: (TRACK_RFQ_COLUMNS[r] if TRACK_RFQ_COLUMNS.get(r) in _rcols else next((_rl[c] for c in v if c in _rl), None)) for r, v in RQ_CAND.items()}
    print('track_rfq column roles (override with TRACK_RFQ_COLUMNS):', RQ_RC)
    if RQ_RC['time'] and RQ_RC['optimal_yield'] and 'cusip' in _rcols:
        _rr = PIPE.store('track_rfq').read(columns=['cusip'] + sorted({v for v in RQ_RC.values() if v}), categorical=False)
        rq0 = pd.DataFrame({'cusip': _rr['cusip'].astype('string').str.strip().to_numpy(), 'ts': to_ns(_rr[RQ_RC['time']]).to_numpy()})
        for role, col in [('quantity', 'qty'), ('optimal_yield', 'y_entry'), ('algo_yield', 'y_algo'), ('msrb_yield', 'y_print')]:
            rq0[col] = pd.to_numeric(_rr[RQ_RC[role]], errors='coerce').to_numpy(float) if RQ_RC[role] else np.nan
        if RQ_RC['side']:
            _sv = _rr[RQ_RC['side']].astype('string').str.upper().str.strip(); print('track side values:', _sv.value_counts(dropna=False).head(8).to_dict())
            rq0['side'] = _sv.map(RQ_SMAP).fillna('P').to_numpy()
        else:
            rq0['side'] = 'P'
        rq0['won'] = _rq_flag(_rr[RQ_RC['won']]).astype(float) if RQ_RC['won'] else np.nan
        _keep = np.ones(len(rq0), bool)
        if RQ_RC['req_id'] and RQ_RC['is_final'] and _rq_flag(_rr[RQ_RC['is_final']]).any():
            _keep &= _rq_flag(_rr[RQ_RC['is_final']])
        if RQ_RC['quoted']:
            _keep &= _rq_flag(_rr[RQ_RC['quoted']])
        if RQ_RC['pricing_error']:
            _keep &= ~_rq_flag(_rr[RQ_RC['pricing_error']])
        rq0 = rq0[_keep]
        if RQ_RC['req_id']:
            rq0 = rq0.assign(req_id=_rr.loc[_keep, RQ_RC['req_id']].astype('string').to_numpy()).sort_values('ts').drop_duplicates('req_id', keep='last')
        rq0 = rq0.dropna(subset=['cusip', 'ts', 'y_entry']).reset_index(drop=True)
        HAS_RFQ = len(rq0) > 0; RQ_SOURCE = 'track_rfq' if HAS_RFQ else None
        print(f'track_rfq: {len(_rr):,} rows -> {len(rq0):,} requests (final, quoted, no pricing error)')
    else:
        print('Section 13: the track carries no recognisable time / optimal yield / cusip column; set TRACK_RFQ_COLUMNS and rerun.')
else:
    print('Section 13: no track_rfq store under the pipeline root (mdp.pull(mdp.Config(root=..., sources=("track_rfq",))) fetches it).')
if not HAS_RFQ and HAS_TRADES:
    _t = trades_raw.drop_duplicates([c for c in ['msrb_trade_id', 'cusip', 'msrb_tradetime', 'msrb_side', 'msrb_quantity'] if c in trades_raw.columns])
    _yp = pd.to_numeric(_t['msrb_yield'], errors='coerce').to_numpy(float)
    rq0 = pd.DataFrame({'cusip': _t['cusip'].astype('string').str.strip().to_numpy(), 'ts': to_ns(_t['msrb_tradetime']).to_numpy(),
                        'side': _t['msrb_side'].astype('string').str.upper().str[0].to_numpy() if 'msrb_side' in _t.columns else 'U',
                        'y_entry': _yp, 'y_print': _yp, 'y_algo': pd.to_numeric(_t['algo_signal_yield'], errors='coerce').to_numpy(float) if 'algo_signal_yield' in _t.columns else np.nan,
                        'qty': pd.to_numeric(_t['msrb_quantity'], errors='coerce').to_numpy(float) if 'msrb_quantity' in _t.columns else np.nan, 'won': 1.0}).dropna(subset=['cusip', 'ts', 'y_entry'])
    RQ_SOURCE = 'matched customer prints (proxy population: no track_rfq; the entry is the print, every row is a fill, the skew counterfactual is skipped)'
    print(f'Section 13 runs on the proxy population: {len(rq0):,} matched customer prints')

# ---- the request frame on the RV grid: the previous close, the concession to it, the side, the fill
RQ = pd.DataFrame()
if len(rq0):
    _r = rq0[rq0['side'].isin(['P', 'S']) & (rq0['ts'] >= pd.Timestamp(CFG.first_oos_date)) & (rq0['ts'] < pd.Timestamp(CFG.end_date) + pd.Timedelta(days=1))].copy()
    _r['bi'] = pd.Index(RV_BONDS.astype(str)).get_indexer(_r['cusip'].astype(str)); _r['dq'] = RV_DATES.searchsorted(_r['ts'].dt.normalize(), side='left')
    _n0 = len(_r); _r = _r[(_r['bi'] >= 0) & (_r['dq'] >= 1) & (_r['dq'] < RV_ND)].reset_index(drop=True)
    _bi, _dq = _r['bi'].to_numpy(np.int64), _r['dq'].to_numpy(np.int64)
    assert (RV_DATES[_dq - 1] < _r['ts'].dt.normalize().to_numpy()).all(), 'a signal date is not strictly before its request date'
    _r['s'] = np.where(_r['side'] == 'P', 1.0, -1.0)
    _r['conc'] = _r['s'] * 100.0 * (_r['y_entry'] - RV_Y[_dq - 1, _bi])
    _ok = np.isfinite(_r['conc']) & (_r['conc'].abs() <= 100.0)
    print(f'{_n0:,} requests in the held-out window | {len(_r):,} on a bond of the RV grid with a previous close in it | {int((~_ok).sum()):,} dropped: no previous mark or |concession| > 100 bp')
    _r = _r[_ok].reset_index(drop=True); _bi, _dq = _r['bi'].to_numpy(np.int64), _r['dq'].to_numpy(np.int64)
    _r['x_prod'] = _r['s'] * 100.0 * (_r['y_entry'] - _r['y_algo'])
    _r['printed'] = np.isfinite(_r['y_print'].to_numpy(float))
    _r['o'] = np.where(_r['printed'], _r['s'] * 100.0 * (_r['y_print'] - _r['y_entry']), np.nan)
    _r['filled'] = np.where(np.isfinite(_r['won']), _r['won'], np.where(_r['printed'], (_r['o'] >= 0).astype(float), np.nan))
    _r['D'] = RV_DUR[_dq - 1, _bi]; _cs = RV_CS[_dq - 1, _bi]; _r['ptc'] = (_cs == 2).astype(float)
    _r['call_structure'] = [CS_NAMES.get(int(c), 'NA') if np.isfinite(c) else 'NA' for c in _cs]
    _r['staleness'] = pd.cut(RV_STALE[_dq - 1, _bi], [-0.5, 0.5, 2.5, 9.5, np.inf], labels=['moved', '1-2 days', '3-9 days', '10+ days']).astype(str)
    _r['dv01'] = (_r['qty'] * _r['D'] * 1e-4).fillna(0.0)       # dollars per bp of yield (par x modified duration, price ~ par); 0 where size or duration is missing
    _r['log_qty'] = np.log(_r['qty'].clip(lower=1000.0)); _r['qty_group'] = qty_group(_r['qty']).to_numpy()
    _r['week'] = _r['ts'].dt.to_period('W').dt.start_time; _r['month'] = _r['ts'].dt.to_period('M').astype(str)
    RQ = _r
    print(f'{len(RQ):,} requests on {RQ["cusip"].nunique():,} bonds | side mix {RQ["side"].value_counts(normalize=True).round(3).to_dict()} | printed {RQ["printed"].mean():.1%} | '
          f'filled {np.nanmean(RQ["filled"]):.1%} of the {int(np.isfinite(RQ["filled"]).sum()):,} with a known outcome | median concession to the previous mark {RQ["conc"].median():+.1f} bp')

# ---- inputs of the outcomes: residual, yield change and maturity-minus-worst MMD move on the grid; the print tape
_E = rv_wide(resid, 'pit_residual')
_CT = np.cumsum(np.nan_to_num(rv_wide(resid, 'target_bp')), axis=0)
_CD = np.cumsum(np.nan_to_num(rv_wide(mcs, 'dmove_mat_bp')), axis=0) if 'mcs' in globals() and 'dmove_mat_bp' in mcs.columns else np.zeros_like(_CT)
RQ_PRINTS, _PY, _LASTP = None, None, None
_sc = next((c for c in ['side', 'tradetype'] if HAS_MSRB and c in msrb_raw.columns), None)
if len(RQ) and _sc and {'tradetime', 'yield'}.issubset(msrb_raw.columns):
    _tp = pd.DataFrame({'cusip': msrb_raw['cusip'].astype('string').str.strip().to_numpy(), 'ts': to_ns(msrb_raw['tradetime']).to_numpy(),
                        'side': msrb_raw[_sc].astype('string').str.upper().str.strip().str[0].to_numpy(), 'y': pd.to_numeric(msrb_raw['yield'], errors='coerce').to_numpy(float)}).dropna()
    _tp = _tp[_tp['side'].isin(['P', 'S', 'D']) & (_tp['y'] > -5.0) & (_tp['y'] < 30.0)].astype({'cusip': str, 'side': str})
    RQ_PRINTS = _tp[_tp['side'].isin(['P', 'S'])].sort_values('ts').reset_index(drop=True)
    _di = RV_DATES.get_indexer(_tp['ts'].dt.normalize()); _pbi = pd.Index(RV_BONDS.astype(str)).get_indexer(_tp['cusip']); _okp = (_di >= 0) & (_pbi >= 0)
    _tq = _tp[_okp].assign(di=_di[_okp], bi=_pbi[_okp]); _tq['dev'] = 100.0 * (_tq['y'].to_numpy(float) - RV_Y[_tq['di'].to_numpy(), _tq['bi'].to_numpy()])
    _tq = _tq[np.isfinite(_tq['dev']) & (_tq['dev'].abs() < 200.0)]; _tq['dev'] = _tq['dev'] - _tq.groupby('side')['dev'].transform('median')
    _dd = _tq.groupby(['di', 'bi'])['dev'].mean()
    _DEV = np.full((RV_ND, RV_NB), np.nan, dtype=np.float32); _DEV[_dd.index.get_level_values(0).to_numpy(), _dd.index.get_level_values(1).to_numpy()] = _dd.to_numpy()
    _LASTP = last_obs_index(_DEV).astype(np.int32); _PY = np.where(np.isfinite(_DEV), RV_Y + _DEV / 100.0, np.nan).astype(np.float32)
    print(f'print tape: {len(RQ_PRINTS):,} customer prints for the own-print reference | {len(_dd):,} bond-days with a side-neutralised print for the peer reference')
else:
    print('no MSRB print tape: the own-print and peer references are skipped')


def _rfq_pnl() -> pd.DataFrame:
    t0 = time.perf_counter()
    dq, bi = RQ['dq'].to_numpy(np.int64), RQ['bi'].to_numpy(np.int64)
    sig = pd.DataFrame(rfq_signals(dq, bi, RV_SSM, _E, RV_GCUR, RQ_LEVEL_W))
    peers = rfq_peer_matrix(dq, bi, RV_FORMS, RV_NB, RV_K)
    M = {'Y': RV_Y, 'CT': _CT, 'CE': RV_CUME, 'CD': _CD, 'dates': RV_DATES, 'LASTP': _LASTP, 'PY': _PY}
    out = rfq_outcomes(RQ, peers, M, RQ_PRINTS)
    print(f'  RFQ outcomes: {len(RQ):,} requests x {len(RQ_H)} horizons | {time.perf_counter() - t0:.0f}s')
    return pd.concat([RQ.reset_index(drop=True), sig, out.reset_index(drop=True)], axis=1)


RQP = pd.DataFrame()
if len(RQ):
    RQP = cached('lab_rfq_pnl', _rfq_pnl, deps=[STAGE_KEYS['walk_forward'], RQ_SOURCE, frame_fingerprint(RQ, ['y_entry', 'qty', 'conc', 'o', 'filled']), 0 if RQ_PRINTS is None else len(RQ_PRINTS),
                                                 float(np.nansum(RV_GCUR)), float(np.nansum(_CD[-1])), RQ_H, RQ_PTC_BLEND, RQ_LEVEL_W, RQ_MIN_PEER_PRINTS, RV_K],
                 code=[rfq_signals, rfq_outcomes, rfq_peer_matrix, last_obs_index])
    for k_, c_ in [('x_ss', 'sig_ss'), ('x_level', 'sig_level'), ('x_gap', 'sig_gap')]:
        RQP[k_] = RQP['s'] * RQP[c_]
    _cov = {c_: float(RQP[c_].notna().mean()) for c_ in ['sig_ss', 'sig_level', 'sig_gap', 'mark_hedged_5', 'own_hedged_5', 'peer_5'] if c_ in RQP.columns}
    print('coverage of the request frame (share of requests):', {k_: f'{v_:.0%}' for k_, v_ in _cov.items()})
    print('signal correlations across requests (oriented by side):'); display(RQP[list(RQ_SIGS)].corr(method='spearman').round(3))
else:
    print('Section 13: no requests on the RV grid; the RFQ P&L analyses are skipped on this run.')

# %%
CELL_T('13. RFQ P&L: the RV signal on the desk requests [3]')
RQ_VERDICT, rq_sel, rq_fm_tab, rq_placebo, rq_bar = 'not run (no requests)', pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
if len(RQP):
    RQ_SAMPLES = {'filled': RQP[RQP['filled'] == 1]}
    if RQ_SOURCE == 'track_rfq':
        RQ_SAMPLES['all requests, at our quote'] = RQP
    _refs = [r_ for r_ in RQ_REFS if f'{r_}_5' in RQP.columns]
    sel_rows, RQ_SPREADS = [], {}
    for samp, F in RQ_SAMPLES.items():
        for x in RQ_SIGS:
            for ref in (_refs if samp == 'filled' else ['mark_hedged']):
                for h in RQ_H:
                    res_, spr = rfq_quintiles(F, x, f'{ref}_{h}', h)
                    if res_:
                        sel_rows.append({'sample': samp, 'signal': RQ_SIGS[x], 'reference': RQ_REFS[ref], 'h': h, **res_}); RQ_SPREADS[(samp, x, ref, h)] = spr
    rq_sel = pd.DataFrame(sel_rows).set_index(['sample', 'signal', 'reference', 'h']) if sel_rows else pd.DataFrame()
    if len(rq_sel):
        _f = rq_sel.xs('filled', level='sample')
        print(f'Selection on the filled requests ({RQ_SOURCE}): weekly Q5 - Q1 of the side-oriented signal (bp of yield per fill; NW t over weeks in brackets), by reference and horizon:')
        _pv = _f['Q5 - Q1 (bp)'].unstack('h').round(2).astype(str) + ' (' + _f['NW t'].unstack('h').round(1).astype(str) + ')'
        display(_pv)
        print('Mean P&L by signal quintile at h = 5, hedged mark (Q1 = least favourable; monotone in Q is the reading, not the end points alone):')
        display(_f.xs(('evaluated mark, hedged', 5), level=('reference', 'h'))[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5 - Q1 (bp)', 'NW t', 'months +', 'months', 'weeks', 'requests']].round(3))
        if 'all requests, at our quote' in RQ_SAMPLES:
            print('The same on every request, as if filled at our quote (the gap to the filled sample is the selection the fill itself imposes):')
            display(rq_sel.xs('all requests, at our quote', level='sample')[['Q1', 'Q5', 'Q5 - Q1 (bp)', 'NW t', 'months +', 'requests']].round(3))

    # ---- Fama-MacBeth: the signal's slope per sd, with side, size and the concession to the mark as controls
    fm_rows = []
    _ctrl = [c_ for c_ in ['s', 'log_qty', 'conc'] if c_ in RQP.columns]
    for x in RQ_SIGS:
        for h in RQ_H:
            sl = rfq_fm(RQ_SAMPLES['filled'], x, f'mark_hedged_{h}', _ctrl)
            if len(sl) >= 4:
                _m = sl.groupby(pd.DatetimeIndex(sl.index).to_period('M')).mean()
                fm_rows.append({'signal': RQ_SIGS[x], 'h': h, 'slope (bp per sd)': float(sl.mean()), 'NW t': nw_t(sl.to_numpy(), rq_lags(h)), 'weeks': len(sl), 'months +': int((_m > 0).sum())})
    rq_fm_tab = pd.DataFrame(fm_rows).set_index(['signal', 'h']) if fm_rows else pd.DataFrame()
    if len(rq_fm_tab):
        print(f'Fama-MacBeth slope of the hedged-mark P&L on the standardised signal, weekly, controls {_ctrl}:'); display(rq_fm_tab['slope (bp per sd)'].unstack('h').round(3)); display(rq_fm_tab['NW t'].unstack('h').round(2))

    # ---- placebo: the signal permuted within week must not sort P&L
    _rng = np.random.default_rng(CFG.seed); pl_rows = []
    _F = RQ_SAMPLES['filled']
    for x in RQ_SIGS:
        _sh = _F[['week', x]].copy(); _sh[x] = _sh.groupby('week')[x].transform(lambda v: v.to_numpy()[_rng.permutation(len(v))])
        res_, _ = rfq_quintiles(_F.assign(**{x: _sh[x]}), x, 'mark_hedged_5', 5)
        if res_:
            pl_rows.append({'signal': RQ_SIGS[x], 'placebo Q5 - Q1 (bp)': res_['Q5 - Q1 (bp)'], 'placebo NW t': res_['NW t']})
    rq_placebo = pd.DataFrame(pl_rows).set_index('signal') if pl_rows else pd.DataFrame()

    # ---- the bar, per signal
    bar_rows = []
    for x, name in RQ_SIGS.items():
        k_m, k_o = ('filled', name, RQ_REFS['mark_hedged'], 5), ('filled', name, RQ_REFS['own_hedged'], 5)
        if k_m not in rq_sel.index:
            continue
        r_m = rq_sel.loc[k_m]; r_o = rq_sel.loc[k_o] if k_o in rq_sel.index else None
        _same = bool(r_o is not None and np.sign(r_o['Q5 - Q1 (bp)']) == np.sign(r_m['Q5 - Q1 (bp)']))
        bar_rows.append({'signal': name, 'Q5 - Q1, hedged mark, h = 5 (bp)': r_m['Q5 - Q1 (bp)'], 'NW t': r_m['NW t'], 'months +': r_m['months +'], 'months': r_m['months'],
                         'own print Q5 - Q1 (bp)': r_o['Q5 - Q1 (bp)'] if r_o is not None else np.nan, 'own print NW t': r_o['NW t'] if r_o is not None else np.nan, 'same sign on prints': _same,
                         'placebo NW t': rq_placebo.loc[name, 'placebo NW t'] if name in rq_placebo.index else np.nan,
                         'passes': bool(r_m['NW t'] >= RQ_BAR_T and r_m['months +'] >= min(RQ_BAR_MONTHS, r_m['months']) and _same)})
    rq_bar = pd.DataFrame(bar_rows).set_index('signal') if bar_rows else pd.DataFrame()
    if len(rq_bar):
        print(f'The bar (lab_rfq_pnl): NW t >= {RQ_BAR_T:g}, positive in >= {RQ_BAR_MONTHS} months, same sign on the own exit print; the within-week placebo beside it:'); display(rq_bar.round(3))
        if len(rq_placebo) and (rq_placebo['placebo NW t'].abs() > 2.5).any():
            print('warning: a placebo sorts P&L with |NW t| > 2.5; the weekly inference is mis-sized here (clustered requests on the same bond-days), read every t in this section with that in mind')

# %%
CELL_T('13. RFQ P&L: the RV signal on the desk requests [4]')
rq_skew, rq_inv, rq_seg, rq_power = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
if len(RQP):
    # ---- (2) skew counterfactual: printed requests with our own quote (the proxy population has none)
    _sk_y = 'mark_hedged_5'
    if RQ_SOURCE == 'track_rfq' and RQP['printed'].sum() >= 1000:
        sk_parts = []
        for x, name in RQ_SIGS.items():
            g_ = rfq_skew_frame(RQP[RQP['printed']], x, _sk_y, clip_bp=CFG.signal_clip_bp)
            if len(g_) >= 500 and g_['week'].nunique() >= 4:
                sk_parts.append(rfq_skew_summary(g_, 5).assign(signal=name).reset_index())
        rq_skew = pd.concat(sk_parts, ignore_index=True).set_index(['signal', 'skew multiple m']) if sk_parts else pd.DataFrame()
        if len(rq_skew):
            rq_skew['passes'] = (rq_skew['NW t vs m = 0'] >= RQ_BAR_T) & (rq_skew['months above m = 0'] >= np.minimum(RQ_BAR_MONTHS, rq_skew['months']))
            print(f'Skew counterfactual on {int(RQP["printed"].sum()):,} printed requests: quote moved toward the customer by m x clip(beta x signal, +/- {CFG.signal_clip_bp:g} bp), beta from earlier months; '
                  'fill = the print crossed the shifted quote (optimistic: no competitor reacts); P&L = hedged mark at h = 5 minus the skew paid:'); display(rq_skew.round(3))
    else:
        print('Skew counterfactual skipped: it needs our own quote and a print per request (the track_rfq ledger).')

    # ---- (3) inventory rule on the fills
    inv_rows = []
    for x, name in RQ_SIGS.items():
        for ref in [r_ for r_ in ['mark_hedged', 'mark_raw'] if f'{r_}_1' in RQP.columns]:
            for hH in [h_ for h_ in RQ_H if h_ > 1]:
                res_ = rfq_inventory(RQ_SAMPLES['filled'], x, ref, hH)
                if res_:
                    inv_rows.append({'signal': name, 'reference': RQ_REFS[ref], 'hold to h': hH, **res_})
    rq_inv = pd.DataFrame(inv_rows).set_index(['signal', 'reference', 'hold to h']) if inv_rows else pd.DataFrame()
    if len(rq_inv):
        rq_inv['passes'] = (rq_inv['NW t'] >= RQ_BAR_T) & (rq_inv['months +'] >= np.minimum(RQ_BAR_MONTHS, rq_inv['months']))
        print('Inventory rule: hold a fill to h when the oriented signal is positive, else exit at h = 1; gain over always exiting at h = 1 (bp per fill, $ per month):'); display(rq_inv.round(3))

    # ---- (4) segmentation of the selection at h = 5, hedged mark
    seg_rows = []
    for dim in ['side', 'qty_group', 'call_structure', 'staleness']:
        for lvl, g_ in RQ_SAMPLES['filled'].groupby(dim):
            if len(g_) < 20 * RQ_MIN_WEEK:
                continue
            for x, name in RQ_SIGS.items():
                res_, _ = rfq_quintiles(g_, x, 'mark_hedged_5', 5, min_n=max(20, RQ_MIN_WEEK // 2))
                if res_:
                    seg_rows.append({'dimension': dim, 'cell': str(lvl), 'signal': name, 'requests': res_['requests'], 'Q5 - Q1 (bp)': res_['Q5 - Q1 (bp)'], 'NW t': res_['NW t']})
    rq_seg = pd.DataFrame(seg_rows)
    if len(rq_seg):
        print('Where the selection lives: Q5 - Q1 at h = 5 on the hedged mark by segment (NW t beside it):')
        display(rq_seg.pivot_table(index=['dimension', 'cell'], columns='signal', values=['Q5 - Q1 (bp)', 'NW t'], aggfunc='first').round(2))

    # ---- power for the live skew experiment (docs/rfq_skew_experiment.md)
    _F = RQ_SAMPLES['filled']; _wk_n = float(RQP.groupby('week').size().median())
    _sd = float(np.nanstd(_F['mark_hedged_5'])) if 'mark_hedged_5' in _F.columns else np.nan
    _cand = rq_skew[rq_skew.index.get_level_values(1) > 0]['vs m = 0 (bp per request)'].dropna() if len(rq_skew) else pd.Series(dtype=float)
    if len(_cand) and _cand.max() > 0:
        _best = _cand.idxmax(); _delta = float(_cand.max())
        _sd_req = float(np.nanstd(RQP.loc[RQP['printed'], _sk_y]))
    else:
        _best, _delta, _sd_req = ('assumed: the 0.10 bp interaction bar', 0.0), float(CFG.interaction_bar_bp), _sd
    _n_arm = rfq_power_n(_sd_req, _delta)
    rq_power = {'design': 'two arms, skew on / off randomised per request', 'effect (bp per request)': _delta, 'from': str(_best), 'sd per request (bp)': _sd_req,
                'requests per arm': _n_arm, 'requests per week (median)': _wk_n, 'weeks at 50/50': (2.0 * _n_arm / _wk_n) if _wk_n > 0 else np.inf}
    print('Power of a live skew A/B (two-sided 5%, 80% power, normal approximation; the observed counterfactual effect is optimistic, so this is a lower bound on the size needed):')
    display(pd.Series(rq_power).to_frame('value'))

    # ---- verdict
    _pass_sel = rq_bar.index[rq_bar['passes']].tolist() if len(rq_bar) else []
    _pass_skew = sorted({i_[0] for i_ in rq_skew.index[rq_skew['passes']]}) if len(rq_skew) else []
    _pass_inv = sorted({i_[0] for i_ in rq_inv.index[rq_inv['passes']]}) if len(rq_inv) else []
    RQ_VERDICT = (f'population: {RQ_SOURCE}; selection bar passed by {_pass_sel or "no signal"}; skew bar passed by {_pass_skew or ("no signal" if len(rq_skew) else "n/a (no ledger)")}; '
                  f'inventory bar passed by {_pass_inv or "no signal"}')
    print('VERDICT (lab_rfq_pnl):', RQ_VERDICT)

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    if len(rq_sel):
        _q = rq_sel.xs(('filled', RQ_REFS['mark_hedged'], 5), level=('sample', 'reference', 'h'))
        for name, r_ in _q.iterrows():
            ax[0].plot(range(1, 6), [r_[f'Q{i}'] for i in range(1, 6)], marker='o', label=f'{name} (t {r_["NW t"]:+.1f})')
        ax[0].set_xticks(range(1, 6)); ax[0].set_xlabel('quintile of the side-oriented signal (weekly)'); ax[0].set_ylabel('hedged mark P&L at h = 5 (bp per fill)'); ax[0].legend(fontsize=8)
        ax[0].set_title('Selection: P&L of the fills by signal quintile', fontsize=10)
        _t = rq_sel.xs('filled', level='sample')['NW t'].unstack('h')
        for (name, ref), r_ in _t.iterrows():
            if ref in (RQ_REFS['mark_hedged'], RQ_REFS['own_hedged'], RQ_REFS['peer']):
                ax[1].plot(r_.index, r_.to_numpy(float), marker='o', ls='-' if ref == RQ_REFS['mark_hedged'] else (':' if ref == RQ_REFS['peer'] else '--'), label=f'{name[:22]} | {ref}')
        ax[1].axhline(RQ_BAR_T, color='k', lw=0.7, ls=':'); ax[1].axhline(0, color='k', lw=0.5); ax[1].set_xticks(list(RQ_H)); ax[1].set_xlabel('horizon (business days)'); ax[1].set_ylabel('Q5 - Q1 NW t')
        ax[1].set_title('Selection t by horizon and reference (dotted line = bar)', fontsize=10); ax[1].legend(fontsize=6)
    if len(rq_skew):
        for name in RQ_SIGS.values():
            if name in rq_skew.index.get_level_values(0):
                _s = rq_skew.loc[name]; ax[2].plot(_s.index, _s['P&L per request (bp)'], marker='o', label=name)
        ax[2].set_xlabel('skew multiple m'); ax[2].set_ylabel('hedged P&L per printed request (bp)'); ax[2].set_title('Skew counterfactual', fontsize=10); ax[2].legend(fontsize=8)
    elif len(rq_inv):
        _i = rq_inv.xs(RQ_REFS['mark_hedged'], level='reference')['gain per fill (bp)'].unstack('hold to h')
        for name, r_ in _i.iterrows():
            ax[2].plot(r_.index, r_.to_numpy(float), marker='o', label=name)
        ax[2].axhline(0, color='k', lw=0.5); ax[2].set_xlabel('hold to h when the signal is favourable'); ax[2].set_ylabel('gain over exit at h = 1 (bp per fill)'); ax[2].set_title('Inventory rule', fontsize=10); ax[2].legend(fontsize=8)
    else:
        ax[2].set_axis_off()
    fig.suptitle(f'RFQ P&L: the residual and relative-value signals on the desk\'s requests ({RQ_SOURCE.split(" (")[0]})', fontsize=12); savefig('13_rfq_pnl')

    record('lab_rfq_pnl', source=RQ_SOURCE, roles=RQ_RC, requests=int(len(RQP)), filled=int((RQP['filled'] == 1).sum()), printed=int(RQP['printed'].sum()), verdict=RQ_VERDICT,
           selection=flat_records(rq_sel) if len(rq_sel) else [], fama_macbeth=flat_records(rq_fm_tab) if len(rq_fm_tab) else [], placebo=flat_records(rq_placebo) if len(rq_placebo) else [],
           bar=flat_records(rq_bar) if len(rq_bar) else [], skew=flat_records(rq_skew) if len(rq_skew) else [], inventory=flat_records(rq_inv) if len(rq_inv) else [],
           segments=rq_seg.round(4).to_dict(orient='records') if len(rq_seg) else [], power={k_: (v_ if not isinstance(v_, float) or np.isfinite(v_) else None) for k_, v_ in rq_power.items()})
else:
    print('Section 13 skipped: no requests (HAS_RFQ =', HAS_RFQ, ', HAS_TRADES =', HAS_TRADES, ')')

# %% [markdown]
# ## Section 14 map: RFQ backtest and the factor model on the desk
#
# | Part | Question | Cells | Headline result |
# |---|---|---|---|
# | **14. Replay and variants** | Can we replay production exactly, and does changing one ingredient (fair value, width, fill curve, objective) beat it? | [1]–[8] | Replay exact on 100% of 1.55M requests. 1 of 16 variants passes: the selection-adjusted objective (+0.171 bp per request, t 3.0). |
# | **14b. The book under IPCA** | Treat IPCA as the book's risk model (MMD tenor + IPCA factors + residual). Can a marginal-risk charge give similar P&L with less DV01? | [9]–[10] | The objective doubles book DV01. With a k = 1 risk charge it makes +$35k a month more than production with 11% less DV01 and 14% less risk. |
# | **14c. Factor charge vs the manual charge** | Keep liquidity + risk, replace the manual charge with a factor-model charge at the same win rate. Better cover? | [11]–[12] | At equal win rate, factor charges keep only about 41% of production's wins and leave more on the table: the manual charge carries request-level information the factor model lacks. They do produce a lower-DV01 book. |
#
# **How to read every comparison:**
#
# - Every variant goes through the production optimizer to its own optimal yield, and its quote = opto + the request's logged loading.
# - Wins are judged against the street threshold that actually traded.
# - P&L is the markout to the bond's next other-side print by day 5.
# - Market moves are reported separately and net out of the comparison with production.
#
# Definitions and the decision log: `docs/rfq_backtest_framework.html`. Review report: `docs/rfq_backtest_report.html`.

# %% [markdown]
# ## 14. RFQ backtest: replaying the production optimizer, then changing one ingredient at a time
#
# Every real bid request is replayed through the production optimizer (vanilla), and each variant (fair value, width, fill curve, objective) is compared with it on the same requests. Definitions, data lineage and the decision log: `docs/rfq_backtest_framework.html`.
#
# **Vanilla.** Inputs come from `muni.StrategyOutput` (pipeline source `strategy_output`): signal yield $m$, pfill key, charges. Production pfill snapshots `prod_pfills/pfills_<date>.json`, read as production does, $\text{quantile\_bp} = (\text{quantiles} \times \text{stdev} + \text{mean}) \times 100$. The optimizer is
#
# $$x^* = \arg\max_q \big(\text{quantile\_bp}_q + w + c\big)\, p_q, \qquad c = -(\text{liquidity} + \text{risk} + \text{manual}), \quad w = 0,$$
#
# and the optimal yield is $\text{opto} = m + x^*/100$, exactly as production builds it. The replayed opto equals the logged `optimalyield` on every request.
#
# **Variants.** Every variant goes through the optimizer and gets its own optimal yield:
#
# | Variant | What goes into the optimizer | Its optimal yield |
# |---|---|---|
# | Alpha overlay | signal yield $m + \alpha$ | $(m + \alpha) + x^*/100$ |
# | Width, risk tilt | $w$ in the utility | $m + x^*_w/100$ |
# | Pfill refit | its own fill curve $p$ | $m + x^*_p/100$ |
# | Objective | value of a fill $x + c + b_{\text{key}}$ | $m + x^*_b/100$ |
#
# The utility does not contain $m$. So an overlay's $x^*$ is production's, and its opto moves by exactly $\alpha$.
#
# **Quote to the street.** Each variant keeps the same downstream loading production applied to that request (carry, venue, manual): the logged `finalquoteyield − optimalyield`, so quote = opto + loading.
#
# For example, production's opto is 4.30, the loading is +0.02 and the street quote is 4.32. A variant's opto of 4.10 gives a quote of 4.12, so the variant is 20 bp richer. The quote difference always equals the opto difference, and cell [6] checks this.
#
# **Gate.** Before any variant is scored, two checks must pass. The replay must reproduce the logged `optimalQuantileBp` (within 0.25 bp) and `pfill` (within 0.01) on at least 95% of requests in every month. The vanilla quote must also reproduce every logged win and loss.
#
# **Outcome of a quote at final yield $y$.** Each request is placed in one group, and the group sets its street threshold:
#
# | Group | Threshold | A quote at $y$ wins if |
# |---|---|---|
# | Won with a cover | max(cover yield, logged yield) | $y \le$ threshold |
# | Won without a cover | logged yield | $y \le$ threshold |
# | Lost to a richer print | print yield | $y <$ threshold |
# | Lost, print not the winner | none | never |
# | No print | none | never |
#
# **Realised P&L.** A fill is valued at the level it traded, $y_{\text{trade}}$. Its markout is $M_h = 100\,(y_{\text{trade}} - y_{\text{exit}})$. The exit is taken on the other side of the trade:
#
# | Trade type | Exit prints |
# |---|---|
# | P (dealer buys from a customer) | S or D |
# | S (dealer sells to a customer) | P or D |
# | D (inter-dealer) | P or S |
#
# $y_{\text{exit}}$ is the last such print up to day $h$ (record $h = 5$; $h = 1, 3$ alongside). The markout belongs to the request, so it is the same for every variant. A variant that wins earns $100\,(y - y_{\text{trade}}) + M_h$. There is no imputation: a winnable request without an exit print is left out of the P&L sample for every variant.
#
# **What is compared.** The level of P&L moves with the market (the common markout ran from +15 bp in March to −1 bp in September). The object is the difference against vanilla. The common markout is the median $M_h$ over the day's winnable requests. Each variant is also scored net of it, which separates selection from simply winning more trades in a rising market.
#
# **Comparison.** The difference against vanilla splits exactly into three parts:
#
# - the price change on fills both quotes make, which needs no exit;
# - the P&L of new wins;
# - the P&L of wins given up.
#
# | Cell | Content |
# |---|---|
# | [1] | Library: pfill reader, file schedule, the vectorised optimizer, the replay gate |
# | [2] | Load `strategy_output` and `track_rfq`, join on (date, request id), filters, the pfill switch date, the vanilla replay and its gate |
# | [3] | Pricing service (cached), cover yields, outcome groups and thresholds, the vanilla outcome check |
# | [4] | Exit prints by type and horizon, the other-side exit, the common markout, DV01 |
# | [5] | The model signals on the requests: state space (7b), residual level (7), RV gap and filtered convergence (11), vine conditional PIT (12), risk scale (10); coverage, age, rank IC with the street error and the markout |
# | [6] | Variants, fitted walk-forward, each through the optimizer to its own opto (`bt_opto`, `bt_final`): alpha overlays (street and exit targets), the width and risk-tilt grids, a pfill refit, a selection-adjusted objective; the opto audit table |
# | [7] | Scoring: leaderboard, decomposition, robustness (net of common markout, h = 1, 3), the bar |
# | [8] | Comparison EDA: leaderboard, sign-robustness heatmap, frontiers vs vanilla under three exits, decomposition, month heatmap, cumulative dollars, market context, segments |

# %%
CELL_T('14. RFQ backtest: production optimizer replay [1]')
PRE_REGISTERED.update({
    'lab_rfq_backtest_replay': ('fixed 2026-10-20 before the lab run: requests = track_rfq final, quoted, no pricing error, joined to muni.StrategyOutput on (date, request id), '
                                'StrategyOutput side B, optimalyield > 0, trade date on or after the switch to the call-structure pfill keys (pfills_20260326); '
                                'production optimizer rebuilt from StrategyOutput signalyield, pfillKey and charges: quantile_bp = (quantiles x stdev + mean) x 100, '
                                'utility = (quantile_bp + width_level + charges_level) x probabilities, charges_level = -(manual + liquidity + risk), width_level = 0, first argmax; '
                                'pfill file by trade date: <= 2026-06-02 the 0326 file, <= 08-04 the 0602 file, <= 09-10 the 0804 file, else the 0910 file; '
                                'gate: |replayed - logged optimalQuantileBp| <= 0.25 bp and |replayed - logged pfill| <= 0.01 on >= 95% of requests in every month; '
                                'variants are compared at the optimal-yield stage, final quote = variant optimal yield + the logged finalquoteyield - optimalyield'),
})
BT_PFILL_DIR = ROOT / 'prod_pfills'
BT_PFILL_SCHEDULE = [('20260326', '2026-06-02'), ('20260602', '2026-08-04'), ('20260804', '2026-09-10'), ('20260910', None)]   # (file, last trade date it serves, inclusive)
BT_WIDTH = 0.0
BT_TOL_BP, BT_TOL_PF, BT_GATE = 0.25, 0.01, 0.95
BT_MAX_CELLS = 20_000_000          # distinct charge levels x curve points per argmax block


def bt_flag(v: pd.Series) -> np.ndarray:
    if v.dtype == object or str(v.dtype).startswith(('string', 'bool', 'category')):
        return v.astype('string').str.lower().isin(['true', '1', 'y', 'yes', 't']).to_numpy(bool)
    return (pd.to_numeric(v, errors='coerce').fillna(0) > 0).to_numpy(bool)


def bt_load_pfills(directory: Path, labels: list[str]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Production pfill snapshots as (quantile_bp, probabilities) per key, built exactly as production builds the curve."""
    out = {}
    for lab in labels:
        path = Path(directory) / f'pfills_{lab}.json'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: copy the production pfill snapshots to {directory}')
        raw = json.loads(path.read_text())
        out[lab] = {k: ((np.asarray(v['quantiles'], dtype=float) * v['stdev'] + v['mean']) * 100, np.asarray(v['probabilities'], dtype=float)) for k, v in raw.items()}
    return out


def bt_file_label(dates: pd.Series, schedule=BT_PFILL_SCHEDULE) -> np.ndarray:
    d = pd.to_datetime(dates).dt.normalize()
    lab = np.full(len(d), schedule[-1][0], dtype=object)
    for f, last in reversed(schedule[:-1]):
        lab[(d <= pd.Timestamp(last)).to_numpy()] = f
    return lab


def bt_optimize(qbp: np.ndarray, p: np.ndarray, charges_level: np.ndarray, width: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production argmax of (quantile_bp + width + charges) x p for each request, solved once per distinct charge level."""
    c = np.asarray(charges_level, dtype=float); ok = np.isfinite(c)
    ob, op, ou = np.full(len(c), np.nan), np.full(len(c), np.nan), np.full(len(c), np.nan)
    if not ok.any():
        return ob, op, ou
    uc, inv = np.unique(c[ok], return_inverse=True)
    base = qbp + width; step = max(1, BT_MAX_CELLS // max(len(qbp), 1)); best = np.empty(len(uc), dtype=np.int64)
    for s in range(0, len(uc), step):
        best[s:s + step] = np.argmax((base[None, :] + uc[s:s + step, None]) * p[None, :], axis=1)
    bi = best[inv]
    ob[ok], op[ok], ou[ok] = qbp[bi], p[bi], (base[bi] + c[ok]) * p[bi]
    return ob, op, ou


def bt_replay(frame: pd.DataFrame, curves: dict, width: float = BT_WIDTH, charges_col: str = 'charges_level') -> pd.DataFrame:
    """Optimal quantile bp, pfill and utility per request from its pfill file and key; NaN where the key is not in the file."""
    out = np.full((len(frame), 3), np.nan); ch = frame[charges_col].to_numpy(float)
    groups = pd.DataFrame({'f': frame['pfill_file'].to_numpy(), 'k': frame['pfill_key'].to_numpy()}).groupby(['f', 'k'], sort=False).indices
    for (lab, key), rows in groups.items():
        cur = curves.get(lab, {}).get(key)
        if cur is not None:
            out[rows] = np.column_stack(bt_optimize(cur[0], cur[1], ch[rows], width))
    return pd.DataFrame(out, index=frame.index, columns=['bt_opt_bp', 'bt_pfill', 'bt_util'])


def bt_gate(frame: pd.DataFrame, by: str = 'month') -> pd.DataFrame:
    d_bp = (frame['bt_opt_bp'] - frame['optimalQuantileBp']).abs(); d_pf = (frame['bt_pfill'] - frame['pfill']).abs()
    g = pd.DataFrame({by: frame[by].to_numpy(), 'key_found': frame['bt_opt_bp'].notna().to_numpy(), 'bp_ok': (d_bp <= BT_TOL_BP).to_numpy(),
                      'pf_ok': (d_pf <= BT_TOL_PF).to_numpy(), 'd_bp': d_bp.to_numpy()})
    g['both_ok'] = g['bp_ok'] & g['pf_ok']
    t = g.groupby(by).agg(requests=('both_ok', 'size'), key_found=('key_found', 'mean'), bp_match=('bp_ok', 'mean'), pfill_match=('pf_ok', 'mean'),
                          both_match=('both_ok', 'mean'), median_abs_d_bp=('d_bp', 'median'), p99_abs_d_bp=('d_bp', lambda s: s.quantile(0.99)))
    t['passes'] = t['both_match'] >= BT_GATE
    return t

# %%
CELL_T('14. RFQ backtest: production optimizer replay [2]')
BT_SO_COLS = ['date', 'ecnreqid', 'cusip', 'side', 'quantity', 'signalyield', 'optimalyield', 'optimalQuantileBp', 'pfill', 'pfillKey',
              'liquiditycharge', 'riskcharge', 'manualcharges', 'finalquoteyield']
BT_TR_COLS = ['date', 'strategy_ecn_req_id', 'cusip', 'rfq_received_time', 'is_final', 'quoted', 'strategy_is_pricing_error', 'manual_takeover', 'is_manual_mode',
              'msrb_yield', 'msrb_price', 'msrb_tradetype', 'msrb_tradetime', 'algo_won', 'cover', 'cover_price_value', 'cover_price_value_backup', 'effective_cover_price',
              'our_bid_price', 'tenor_bucket', 'qty_bucket', 'call_bucket', 'coupon_bucket', 'final_comp_rating', 'probability', 'bias_bp']
_so = PIPE.store('strategy_output').read(columns=BT_SO_COLS, categorical=False)
_tr = PIPE.store('track_rfq').read(columns=BT_TR_COLS, categorical=False)
if _so is None or _tr is None:
    raise RuntimeError('strategy_output or track_rfq is not pulled under the pipeline root: run data_pipeline_regen.ipynb')
for _f in (_so, _tr):
    _f['date'] = pd.to_datetime(_f['date']).dt.normalize()
_so['ecnreqid'] = _so['ecnreqid'].astype('string'); _tr['ecnreqid'] = _tr['strategy_ecn_req_id'].astype('string')
_n_so, _n_tr = len(_so), len(_tr)
_keep = np.ones(len(_tr), bool)
if 'is_final' in _tr and bt_flag(_tr['is_final']).any():
    _keep &= bt_flag(_tr['is_final'])
if 'quoted' in _tr:
    _keep &= bt_flag(_tr['quoted'])
if 'strategy_is_pricing_error' in _tr:
    _keep &= ~bt_flag(_tr['strategy_is_pricing_error'])
_tr = _tr[_keep].drop_duplicates(['date', 'ecnreqid'], keep='last')
BT0 = _so.merge(_tr.drop(columns=['strategy_ecn_req_id']), on=['date', 'ecnreqid'], how='inner', suffixes=('', '_trk'))
_n_join = len(BT0)
BT0 = BT0[(BT0['side'].astype(str) == 'B') & (BT0['optimalyield'] > 0) & (BT0['signalyield'] > 0)].copy()
BT0['pfill_key'] = BT0['pfillKey'].astype('string')
print(f'strategy_output {_n_so:,} rows | track_rfq {_n_tr:,} rows -> {len(_tr):,} final, quoted, no pricing error | joined on (date, request id) {_n_join:,} | '
      f'bid side with an optimal yield {len(BT0):,}')
print(f'cusip agreement between the two sources: {float((BT0["cusip"].astype(str) == BT0["cusip_trk"].astype(str)).mean()):.4f}')

# ---- the switch to the call-structure pfill keys (the 0326 file): first date from which every non-cpn3 key carries a call token
_nc = BT0[~BT0['pfill_key'].str.startswith('cpn3')]
_share = _nc.groupby('date')['pfill_key'].apply(lambda s: float(s.str.contains('call').mean()))
_ok = (_share >= 0.99).astype(int)[::-1].cummin()[::-1].astype(bool)
BT_START = pd.Timestamp(globals().get('BT_START_OVERRIDE') or (_ok[_ok].index.min() if _ok.any() else CFG.first_oos_date))
_around = _share[(_share.index >= BT_START - pd.Timedelta(days=10)) & (_share.index <= BT_START + pd.Timedelta(days=5))]
print(f'pfill key switch: call-structure keys from {BT_START.date()} (share of non-cpn3 requests with a call token, around the switch):',
      {str(k.date()): round(v, 3) for k, v in _around.items()})

BT = BT0[BT0['date'] >= BT_START].reset_index(drop=True)
BT['pfill_file'] = bt_file_label(BT['date'])
BT['charges_level'] = -(BT['manualcharges'] + BT['liquiditycharge'] + 0.0 + BT['riskcharge'])
BT['month'] = BT['date'].dt.to_period('M').astype(str)
BT_CURVES = bt_load_pfills(BT_PFILL_DIR, [f for f, _ in BT_PFILL_SCHEDULE])
print(f'replay window {BT_START.date()} .. {BT["date"].max().date()}: {len(BT):,} requests | by pfill file:', BT['pfill_file'].value_counts().sort_index().to_dict())
print('logged keys missing from their file (share of requests):',
      {f: round(float((~g['pfill_key'].isin(list(BT_CURVES[f]))).mean()), 4) for f, g in BT.groupby('pfill_file')})


def _bt_replay_stage() -> pd.DataFrame:
    t0 = time.perf_counter(); out = bt_replay(BT, BT_CURVES)
    print(f'  replay: {len(BT):,} requests, {BT.groupby(["pfill_file", "pfill_key"]).ngroups} (file, key) curves | {time.perf_counter() - t0:.0f}s')
    return out


_pf_sig = {f: (BT_PFILL_DIR / f'pfills_{f}.json').stat().st_size for f, _ in BT_PFILL_SCHEDULE}
_rep = cached('lab_rfq_replay', _bt_replay_stage, deps=[frame_fingerprint(BT, ['signalyield', 'optimalQuantileBp', 'charges_level', 'pfill']), str(BT_START.date()), BT_PFILL_SCHEDULE, _pf_sig, BT_WIDTH],
              code=[bt_optimize, bt_replay, bt_load_pfills, bt_file_label])
BT[['bt_opt_bp', 'bt_pfill', 'bt_util']] = _rep[['bt_opt_bp', 'bt_pfill', 'bt_util']].to_numpy()
BT['bt_optyield'] = BT['signalyield'] + BT['bt_opt_bp'] / 100.0
BT['adj_yield'] = BT['finalquoteyield'] - BT['optimalyield']
BT['util_logged'] = (BT['optimalQuantileBp'] + BT['charges_level']) * BT['pfill']

bt_gate_month, bt_gate_file = bt_gate(BT, 'month'), bt_gate(BT, 'pfill_file')
BT_REPLAY_PASS = bool(bt_gate_month['passes'].all())
print(f'Replay gate (|d optimalQuantileBp| <= {BT_TOL_BP} bp and |d pfill| <= {BT_TOL_PF} on >= {BT_GATE:.0%} of requests), by month:'); display(bt_gate_month.round(4))
print('... by pfill file:'); display(bt_gate_file.round(4))
_bad = BT[BT['bt_opt_bp'].notna() & ((BT['bt_opt_bp'] - BT['optimalQuantileBp']).abs() > BT_TOL_BP)]
if len(_bad):
    print(f'{len(_bad):,} requests off by more than {BT_TOL_BP} bp; the keys carrying most of them:')
    display(_bad.groupby(['pfill_file', 'pfill_key']).size().sort_values(ascending=False).head(10).to_frame('requests'))
    display(_bad[['date', 'ecnreqid', 'pfill_file', 'pfill_key', 'charges_level', 'optimalQuantileBp', 'bt_opt_bp', 'pfill', 'bt_pfill']].head(8))

# ---- outcome inputs: the cover price (cover_price_value is empty on the track; effective_cover_price is its populated form), wins, manual, matched prints
BT['cover_px'] = pd.to_numeric(BT['cover_price_value'], errors='coerce').fillna(pd.to_numeric(BT['effective_cover_price'], errors='coerce'))
BT['won_cover'] = BT['cover_px'].notna()
BT['won_logged'] = bt_flag(BT['algo_won'])
BT['manual'] = np.zeros(len(BT), bool)
for _c in ('manual_takeover', 'is_manual_mode'):
    if _c in BT:
        BT['manual'] |= bt_flag(BT[_c])
BT['print_type'] = BT['msrb_tradetype'].astype('string').str.upper().fillna('none')
print('wins: logged algo_won against a cover price present (requests):'); display(pd.crosstab(BT['won_logged'], BT['won_cover'], margins=True))
print(f'manual {BT["manual"].mean():.2%} | matched print by type {BT["print_type"].value_counts(normalize=True).round(4).to_dict()} | tenor bucket present {BT["tenor_bucket"].notna().mean():.2%}')
print(f'ex-ante expected P&L per request (optimizer utility, bp): logged {BT["util_logged"].mean():.3f}, replayed {BT["bt_util"].mean():.3f}; '
      f'finalquoteyield - optimalyield: {100 * BT["adj_yield"].median():+.2f} bp median, zero on {float((100 * BT["adj_yield"]).abs().le(0.01).mean()):.1%}')
record('lab_rfq_backtest', replay_start=str(BT_START.date()), replay_end=str(BT['date'].max().date()), requests=int(len(BT)), replay_passes=BT_REPLAY_PASS,
       gate_by_month=flat_records(bt_gate_month), gate_by_file=flat_records(bt_gate_file), won_cover_share=float(BT['won_cover'].mean()),
       won_logged_share=float(BT['won_logged'].mean()), manual_share=float(BT['manual'].mean()))

# %%
CELL_T('14. RFQ backtest: outcomes and the street level [3]')
import requests  # noqa: E402
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, USFederalHolidayCalendar  # noqa: E402

PRE_REGISTERED.update({
    'lab_rfq_backtest_outcomes': ('fixed 2026-10-20 before the lab run: bid side; prices -> yields by the desk pricing service (price2yield) at settle T+2 business days, the convention '
                                  'that reproduces the logged final quote yields from our bid prices; groups scored separately: (1) won with a cover price: threshold = '
                                  'max(cover yield, logged final yield), a quote wins iff its final yield <= threshold, d = 100 (threshold - yield) = bp left on the table; '
                                  '(2) won without a cover price (or the cover failed to price): threshold = logged final yield, scored by its shift and later against the next '
                                  'MSRB level (markout); (3) lost with a matched P/D print richer than our logged quote: a quote wins iff its final yield < print yield, '
                                  'strictly, d = bp crossed (positive) or short (negative); (4) lost with a matched print at or cheaper than our logged quote: the print is not the '
                                  'winner, no fill; (5) no matched print: no fill; the vanilla quote must reproduce the logged outcome exactly'),
})
BT_PRICING_HOST = 'http://3kmu-r-app-s01:8080'
BT_PRICING_CACHE = ARTIFACTS / 'pricing_cache.parquet'
BT_PRICING_BATCH = 250
BT_SETTLE_LAG = 2            # business days; T+1 leaves a +0.13 bp median bias against the logged yields, T+2 gives 0.0 bp (convention check below)
BT_EPS_BP = 1e-6
BT_GROUPS = ['won, cover', 'won, no cover', 'lost, printed', 'lost, print not winner', 'no print']


class _BondCalendar(AbstractHolidayCalendar):
    rules = USFederalHolidayCalendar.rules + [GoodFriday]


BT_BDAY = pd.offsets.CustomBusinessDay(calendar=_BondCalendar())


def bt_settle(dates: pd.Series, lag: int = BT_SETTLE_LAG) -> pd.Series:
    """Settle date `lag` business days after the trade date (US federal holidays and Good Friday)."""
    d = pd.to_datetime(dates).dt.normalize(); u = pd.Series(pd.DatetimeIndex(d.unique()))
    return d.map(dict(zip(u, u + BT_BDAY * lag))).dt.strftime('%Y-%m-%d')


def _bt_post(sess: requests.Session, endpoint: str, chunk: pd.DataFrame) -> list[dict]:
    """One POST of distinct cusips; on a batch failure every input is retried alone, and a failing input keeps its reason."""
    def _call(rows: pd.DataFrame) -> dict:
        r = sess.post(f'{BT_PRICING_HOST}/{endpoint}', json={'pricingInputs': [{'cusip': c, 'inputValue': float(v), 'settleDate': s} for c, v, s in zip(rows['cusip'], rows['value'], rows['settle'])]}, timeout=60)
        r.raise_for_status(); d = r.json()
        if not d.get('pricingStatus'):
            raise RuntimeError(d.get('pricingFailureReason') or 'pricingStatus false')
        return d.get('pricingResponse') or {}
    try:
        resp = _call(chunk)
    except Exception:  # noqa: BLE001
        resp = {}
        for _, row in chunk.iterrows():
            try:
                resp.update(_call(row.to_frame().T))
            except Exception as exc:  # noqa: BLE001
                resp[row['cusip']] = {'_error': str(exc)[:200]}
    out = []
    for c, v, s in zip(chunk['cusip'], chunk['value'], chunk['settle']):
        x = resp.get(c, {'_error': 'missing from response'})
        out.append({'endpoint': endpoint, 'cusip': c, 'value': v, 'settle': s, 'price': x.get('price'), 'yield': x.get('yield'), 'dv01': x.get('dv01'), 'status': x.get('_error', 'ok')})
    return out


def bt_pricing(endpoint: str, req: pd.DataFrame) -> pd.DataFrame:
    """price2yield / yield2price for (cusip, value, settle), cached on disk so a rerun never calls the service for a known input."""
    req = req.assign(cusip=req['cusip'].astype(str), value=pd.to_numeric(req['value'], errors='coerce').round(6)).dropna(subset=['value']).drop_duplicates(['cusip', 'value', 'settle'])
    cache = pd.read_parquet(BT_PRICING_CACHE) if BT_PRICING_CACHE.exists() else pd.DataFrame(columns=['endpoint', 'cusip', 'value', 'settle', 'price', 'yield', 'dv01', 'status'])
    have = cache.loc[cache['endpoint'] == endpoint, ['cusip', 'value', 'settle']]
    todo = req.merge(have, on=['cusip', 'value', 'settle'], how='left', indicator=True)
    todo = todo[todo['_merge'] == 'left_only'].drop(columns='_merge')
    if len(todo):
        t0 = time.perf_counter(); sess = requests.Session(); sess.trust_env = False; rows = []
        todo = todo.assign(_k=todo.groupby('cusip').cumcount())      # the response is keyed by cusip: one input per cusip per POST
        for _, g in todo.groupby('_k'):
            for s in range(0, len(g), BT_PRICING_BATCH):
                rows += _bt_post(sess, endpoint, g.iloc[s:s + BT_PRICING_BATCH])
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(BT_PRICING_CACHE, index=False)
        print(f'  pricing service {endpoint}: {len(rows):,} new inputs in {time.perf_counter() - t0:.0f}s, {sum(r["status"] != "ok" for r in rows):,} failed; cache {len(cache):,} rows')
    return req.merge(cache[cache['endpoint'] == endpoint].drop(columns='endpoint'), on=['cusip', 'value', 'settle'], how='left')


def bt_price_to_yield(frame: pd.DataFrame, price_col: str, mask: pd.Series) -> np.ndarray:
    """price2yield of `price_col` on the rows in `mask` at the backtest settle; NaN elsewhere and on pricing failures."""
    q = frame.loc[mask, ['cusip', price_col, 'settle']].rename(columns={price_col: 'value'}).astype({'cusip': str})
    q['value'] = pd.to_numeric(q['value'], errors='coerce').round(6)
    r = bt_pricing('price2yield', q.dropna(subset=['value']))
    out = np.full(len(frame), np.nan)
    out[np.flatnonzero(mask.to_numpy())] = q.merge(r[['cusip', 'value', 'settle', 'yield']], on=['cusip', 'value', 'settle'], how='left')['yield'].to_numpy(float)
    return out


def bt_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    """Each request's group and street threshold yield; fixed by the logged outcome, independent of any variant."""
    yl = frame['finalquoteyield'].to_numpy(float); yc = frame['cover_y'].to_numpy(float); ym = frame['msrb_yield'].to_numpy(float)
    won = frame['won_logged'].to_numpy(bool); has_cover = won & np.isfinite(yc); printed = ~won & np.isfinite(ym)
    grp = np.select([has_cover, won, printed & (ym < yl), printed], BT_GROUPS[:4], BT_GROUPS[4])
    thr = np.select([has_cover, won, printed & (ym < yl)], [np.maximum(yc, yl), yl, ym], np.nan)
    return pd.DataFrame({'group': grp, 'threshold_y': thr}, index=frame.index)


def bt_outcomes(frame: pd.DataFrame, y_final) -> pd.DataFrame:
    """Per request, for a quote at final yield y: win or not, d = 100 (threshold - y) in bp, and the shift from the logged quote in bp."""
    y = np.asarray(y_final, dtype=float); grp = frame['group'].to_numpy()
    d_bp = 100.0 * (frame['threshold_y'].to_numpy(float) - y)
    win = np.where(grp == 'lost, printed', d_bp > BT_EPS_BP, d_bp >= -BT_EPS_BP) & np.isfinite(d_bp)
    return pd.DataFrame({'group': grp, 'win': win, 'd_bp': d_bp, 'shift_bp': 100.0 * (y - frame['finalquoteyield'].to_numpy(float))}, index=frame.index)


# ---- convention check: our own bid price -> yield must reproduce the logged final quote yield
BT['settle'] = bt_settle(BT['date'])
_chk = BT.loc[BT['won_cover']].sample(min(2000, int(BT['won_cover'].sum())), random_state=CFG.seed)
_dconv = pd.Series(100.0 * (bt_price_to_yield(_chk, 'our_bid_price', pd.Series(True, index=_chk.index)) - _chk['finalquoteyield'].to_numpy(float)))
print(f'convention check on {len(_chk):,} won requests, settle T+{BT_SETTLE_LAG}: price2yield(our bid price) - logged final quote yield: median {_dconv.median():+.3f} bp, '
      f'|d| <= 0.1 bp on {(_dconv.abs() <= 0.1).mean():.1%}, |d| <= 0.5 bp on {(_dconv.abs() <= 0.5).mean():.1%}')

# ---- cover yields, groups and thresholds
BT['cover_y'] = bt_price_to_yield(BT, 'cover_px', BT['won_cover'])
BT = BT.join(bt_thresholds(BT)) if 'group' not in BT else BT.drop(columns=['group', 'threshold_y'], errors='ignore').join(bt_thresholds(BT))
print(f'cover yield priced for {np.isfinite(BT["cover_y"]).sum():,} of {int(BT["won_cover"].sum()):,} won requests with a cover price; '
      f'cover yield below our logged yield (clamped to it): {int((BT["cover_y"] < BT["finalquoteyield"]).sum()):,}')

# ---- the vanilla quote must reproduce the logged outcome exactly
BT['y_vanilla'] = BT['bt_optyield'] + BT['adj_yield']
_ov = bt_outcomes(BT, BT['y_vanilla'])
_won_rule = _ov['group'].isin(['won, cover', 'won, no cover'])
BT_VANILLA_PASS = bool((_ov['win'] == _won_rule).all() and (_won_rule == BT['won_logged']).all())
_sum = _ov.assign(print_type=np.where(_ov['group'].str.startswith('lost'), BT['print_type'], ''), manual=BT['manual']).groupby(['group', 'print_type']).agg(
    requests=('win', 'size'), vanilla_wins=('win', 'mean'), manual=('manual', 'mean'),
    d_bp_p10=('d_bp', lambda s: s.quantile(0.10)), d_bp_median=('d_bp', 'median'), d_bp_p90=('d_bp', lambda s: s.quantile(0.90)))
print('Vanilla (the production quote, replayed) by group; d = threshold yield - our final yield in bp (won, cover: left on the table; lost, printed: negative = short of the print):')
display(_sum.round(3))
print(f'vanilla reproduces the logged win / loss on every request: {BT_VANILLA_PASS} | '
      f'bp left on the table on wins with a cover: mean {_ov.loc[_ov["group"] == "won, cover", "d_bp"].mean():.2f}, median {_ov.loc[_ov["group"] == "won, cover", "d_bp"].median():.2f}')
record('lab_rfq_backtest', outcome_groups=flat_records(_sum.reset_index()), settle_lag=BT_SETTLE_LAG, convention_check_median_bp=float(_dconv.median()),
       convention_check_share_01bp=float((_dconv.abs() <= 0.1).mean()), vanilla_pass=BT_VANILLA_PASS)

# %%
CELL_T('14. RFQ backtest: exit levels, markouts and DV01 [4]')
PRE_REGISTERED.update({
    'lab_rfq_backtest_markout': ('fixed 2026-10-20 before the variants were scored, exit revised the same day on the desk rule before re-scoring: a fill is valued at the level it traded, '
                                 'y_trade (our logged final yield on a win, the matched print on a lost printed request), and its markout M_h = 100 (y_trade - y_exit) bp is a property '
                                 'of the request, the same for every variant; the exit is on the other side of the trade: a P trade exits on the bond\'s S or D prints, an S trade on P or D, '
                                 'a D trade on P or S; y_exit = the last such print after the request and after the matched trade, up to the close of the h-th business day (h = 1 is the '
                                 'request date), h in {1, 3, 5}; a request whose trade type is unknown has no exit; no imputation: a winnable request without an exit print is outside the '
                                 'P&L sample for every variant; record = h = 5; the common component of a markout = the median M_h over the winnable requests of the same request date '
                                 '(the market move plus the typical bid-offer of that day), and the markout net of it is reported alongside (the comparison with vanilla is the object; '
                                 'the market level is not); a variant\'s realised P&L on a request it wins = 100 (y_variant - y_trade) + M_h; DV01 per 100 par from the pricing service '
                                 'at the logged final yield'),
})
BT_H = (1, 3, 5)
BT_H_RECORD, BT_EXIT_RECORD = 5, 'X'
BT_EXIT_TYPES = {'P': 'P: customer sells to a dealer', 'D': 'D: inter-dealer', 'S': 'S: customer buys from a dealer'}
BT_EXIT_SIDES = {'P': ('S', 'D'), 'S': ('P', 'D'), 'D': ('P', 'S')}
BT_EXIT_LABEL = {'X': 'other side of the trade (P -> S/D, S -> P/D, D -> P/S), last print', 'Xm': 'other side, net of the day\'s common markout'}
BT_WINNABLE = ['won, cover', 'won, no cover', 'lost, printed']


def bt_tape(raw: pd.DataFrame) -> pd.DataFrame:
    """MSRB prints as (cusip, ts, type, yield), time-sorted; the Section 2 extreme-yield rule (|y - date median| > yield_mad_k x date MAD) drops unit errors."""
    t = pd.DataFrame({'cusip': raw['cusip'].astype(str).str.strip(), 'ts': to_ns(raw['tradetime']), 'type': raw['tradetype'].astype(str).str.upper().str[0],
                      'y': pd.to_numeric(raw['yield'], errors='coerce')}).dropna()
    t = t[t['type'].isin(list(BT_EXIT_TYPES))]
    day = t['ts'].dt.normalize(); med = t.groupby(day)['y'].transform('median'); mad = (t['y'] - med).abs().groupby(day).transform('median')
    keep = (t['y'] - med).abs() <= CFG.yield_mad_k * mad
    print(f'print tape: {int((~keep).sum()):,} of {len(t):,} prints beyond {CFG.yield_mad_k:g} date-MADs of the date median dropped (Section 2 rule)')
    return t[keep].sort_values('ts').reset_index(drop=True)


def bt_exits(frame: pd.DataFrame, tape: pd.DataFrame, horizons=BT_H, types=tuple(BT_EXIT_TYPES)) -> pd.DataFrame:
    """Per request, horizon and print type: the yield and time of the bond's last print of that type after max(request time, matched trade time)
    and up to the end of the h-th business day (h = 1: the request date)."""
    q = pd.DataFrame({'_i': np.arange(len(frame)), 'cusip': frame['cusip'].astype(str).to_numpy(),
                      'start': np.maximum(to_ns(frame['rfq_received_time']).fillna(frame['date']).to_numpy(), to_ns(frame['msrb_tradetime']).fillna(pd.Timestamp(0)).to_numpy())})
    d = pd.to_datetime(frame['date']).dt.normalize().to_numpy()
    out = {}
    for h in horizons:
        q['cut'] = (pd.DatetimeIndex(d) + BT_BDAY * (h - 1) + pd.Timedelta(days=1)).astype('datetime64[ns]')
        qs = q.sort_values('cut')
        for tp in types:
            j = pd.merge_asof(qs, tape.loc[tape['type'] == tp, ['cusip', 'ts', 'y']], left_on='cut', right_on='ts', by='cusip', direction='backward', allow_exact_matches=False)
            ok = (j['ts'] > j['start']).to_numpy(); ii = j['_i'].to_numpy()
            y_, t_ = np.full(len(frame), np.nan), np.full(len(frame), np.datetime64('NaT'), dtype='datetime64[ns]')
            y_[ii[ok]] = j['y'].to_numpy(float)[ok]; t_[ii[ok]] = j['ts'].to_numpy()[ok]
            out[f'yx_{h}_{tp}'] = y_; out[f'tx_{h}_{tp}'] = t_
    return pd.DataFrame(out, index=frame.index)


def bt_side_exit(frame: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yield, time and type of the last print on the other side of the request's trade (BT_EXIT_SIDES) up to day h; NaN where the trade type is unknown or nothing printed."""
    tps = list(BT_EXIT_TYPES); tt = frame['print_type'].astype(str).to_numpy()
    Y = np.column_stack([frame[f'yx_{h}_{tp}'].to_numpy(float) for tp in tps])
    T = np.column_stack([frame[f'tx_{h}_{tp}'].to_numpy('datetime64[ns]') for tp in tps])
    ok = ~np.isnat(T) & np.column_stack([np.isin(tt, [t for t, s in BT_EXIT_SIDES.items() if tp in s]) for tp in tps])
    j = np.where(ok, T.astype('int64'), np.iinfo(np.int64).min).argmax(axis=1); has = ok.any(axis=1); r = np.arange(len(frame))
    return np.where(has, Y[r, j], np.nan), np.where(has, T[r, j], np.datetime64('NaT', 'ns')), np.where(has, np.array(tps)[j], 'none')


BT_TAPE = bt_tape(msrb_raw)
print(f'print tape: {len(BT_TAPE):,} prints {BT_TAPE["ts"].min().date()} .. {BT_TAPE["ts"].max().date()} | by type {BT_TAPE["type"].value_counts().to_dict()}')
# fingerprint the two columns only: the frame's shape enters the key and BT gains columns on a rerun
_ex = cached('lab_rfq_exits', lambda: bt_exits(BT, BT_TAPE),
             deps=[frame_fingerprint(BT[['finalquoteyield', 'msrb_yield']], ['finalquoteyield', 'msrb_yield']), len(BT_TAPE), float(BT_TAPE['y'].sum()), BT_H], code=[bt_exits, bt_tape])
BT = BT.drop(columns=[c for c in _ex.columns if c in BT.columns]).join(_ex.set_index(BT.index))

BT['y_trade'] = np.where(BT['group'].isin(['won, cover', 'won, no cover']), BT['finalquoteyield'], np.where(BT['group'] == 'lost, printed', BT['msrb_yield'], np.nan))
BT['winnable'] = BT['group'].isin(BT_WINNABLE)
BT['week'] = BT['date'].dt.to_period('W').dt.start_time
for h in BT_H:
    for tp in BT_EXIT_TYPES:
        BT[f'M_{h}_{tp}'] = 100.0 * (BT['y_trade'] - BT[f'yx_{h}_{tp}'])
    BT[f'yx_{h}_X'], BT[f'tx_{h}_X'], BT[f'xtype_{h}'] = bt_side_exit(BT, h)
    BT[f'M_{h}_X'] = 100.0 * (BT['y_trade'] - BT[f'yx_{h}_X'])
    BT[f'mkt_{h}'] = BT[f'M_{h}_X'].where(BT['winnable']).groupby(BT['date']).transform('median')
    BT[f'M_{h}_Xm'] = BT[f'M_{h}_X'] - BT[f'mkt_{h}']

# ---- what the exits look like: who trades against whom, coverage and level of the markout
_w = BT[BT['winnable']]
print(f'trade type of the winnable requests (the side the exit is taken against): {_w["print_type"].value_counts().to_dict()} | '
      f'unknown trade type, no exit: {int((_w["print_type"] == "none").sum()):,}')
print(f'exit print type by trade type (h = {BT_H_RECORD}, share of requests with an exit):')
display(pd.crosstab(_w['print_type'], _w[f'xtype_{BT_H_RECORD}'], normalize='index').round(3).join(_w.groupby('print_type').size().rename('requests')))
_rows = []
for g, f in _w.groupby('group'):
    for h in BT_H:
        for tp in list(BT_EXIT_TYPES) + ['X', 'Xm']:
            m = f[f'M_{h}_{tp}']
            _rows.append({'group': g, 'h': h, 'exit': tp, 'requests': len(f), 'coverage': m.notna().mean(), 'mean (bp)': m.mean(), 'median (bp)': m.median(),
                          'p10': m.quantile(0.10), 'p90': m.quantile(0.90)})
bt_markout_tab = pd.DataFrame(_rows).set_index(['group', 'h', 'exit'])
print(f'Markout of the traded level, M = 100 (y_trade - y_exit) bp, positive = the bond richened after the trade (good for the buyer), h = {BT_H_RECORD}; '
      f'X = {BT_EXIT_LABEL["X"]}, Xm = {BT_EXIT_LABEL["Xm"]}; P, D, S = a single print type for reference:')
display(bt_markout_tab.xs(BT_H_RECORD, level='h').round(2))
_mk = _w.groupby('month').agg(**{'exit coverage': (f'M_{BT_H_RECORD}_X', lambda s: s.notna().mean()), 'common markout, mean of daily median (bp)': (f'mkt_{BT_H_RECORD}', 'mean'),
                                 'markout M (mean, bp)': (f'M_{BT_H_RECORD}_X', 'mean'), 'net of common (mean, bp)': (f'M_{BT_H_RECORD}_Xm', 'mean')})
print(f'By month (winnable requests, h = {BT_H_RECORD}): the common markout is what every trade of the day earned on the exit (market move plus the day\'s bid-offer); '
      'every variant is exposed to it, so it moves the level of P&L, not the comparison on common fills:')
display(_mk.round(3))

# ---- DV01 per 100 par at the logged final yield, one call per (bond, settle) of the winnable requests
_dq = BT.loc[BT['winnable'], ['cusip', 'settle', 'finalquoteyield']].astype({'cusip': str}).groupby(['cusip', 'settle'], as_index=False)['finalquoteyield'].median().rename(columns={'finalquoteyield': 'value'})
_dv = bt_pricing('yield2price', _dq)
_dv = _dq.assign(value=_dq['value'].round(6)).merge(_dv[['cusip', 'value', 'settle', 'dv01', 'status']], on=['cusip', 'value', 'settle'], how='left')
_dvm = BT[['cusip', 'settle']].astype({'cusip': str}).merge(_dv[['cusip', 'settle', 'dv01']], on=['cusip', 'settle'], how='left')
BT['dv01'] = pd.to_numeric(_dvm['dv01'], errors='coerce').to_numpy()
BT['usd_per_bp'] = pd.to_numeric(BT['quantity'], errors='coerce') / 100.0 * BT['dv01']
print(f'DV01: priced {int((_dv["status"] == "ok").sum()):,} of {len(_dv):,} (bond, settle) pairs | $ per bp of a fill: median {BT.loc[BT["winnable"], "usd_per_bp"].median():.2f}, '
      f'p90 {BT.loc[BT["winnable"], "usd_per_bp"].quantile(0.9):.2f} | quantity is par $ ({BT["quantity"].min():,.0f} .. {BT["quantity"].max():,.0f})')
record('lab_rfq_backtest', markout=flat_records(bt_markout_tab), markout_by_month=flat_records(_mk), exit_record=f'{BT_EXIT_LABEL[BT_EXIT_RECORD]}, h = {BT_H_RECORD}',
       exit_sides=BT_EXIT_SIDES, dv01_priced=int((_dv['status'] == 'ok').sum()))

# %%
CELL_T('14. RFQ backtest: model signals on the requests [5]')
PRE_REGISTERED.update({
    'lab_rfq_backtest_signals': ('fixed 2026-10-20 before the variants were scored: every model input is read at the bond\'s last observation on or before the last model date strictly '
                                 'before the request date (point in time; its age in model business days is kept, no age cut); signals oriented positive = the bond is expected to richen: '
                                 'ss = -state-space forecast (7b), level = -(20-day cumulative residual) / (sd sqrt n) (7), gap = spread to MMD minus the median of the 20 beta-space peers (11), '
                                 'rv = the filtered expected convergence over h = 5 of the permanent-temporary gap model, g - mu - phi^5 x, phi per month and call structure (11), '
                                 'vine = selected-vine conditional PIT h(U | P, G) - 0.5 (12); risk = trailing sd of the bond\'s daily yield change (the Section 10 Gaussian scale); '
                                 'street error e = 100 (threshold yield - signal yield) on won-with-cover and lost-printed requests, the competing level relative to production\'s fair value'),
})
BT_SIGNALS = {'ss': 'state space (7b)', 'level': 'residual level (7)', 'gap': 'RV peer gap (11)', 'rv': 'RV filtered convergence (11)', 'vine': 'vine conditional PIT (12)'}

_bi = pd.Index(RV_BONDS.astype(str)).get_indexer(BT['cusip'].astype(str))
_d0 = RV_DATES.searchsorted(BT['date'].to_numpy(), side='left') - 1
_on = (_bi >= 0) & (_d0 >= 0)


def bt_last_obs(M: np.ndarray, d0: np.ndarray, bi: np.ndarray, on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value of M at the bond's last finite observation on or before row d0, and its age in rows; NaN off the grid."""
    L = last_obs_index(M); v, age = np.full(len(d0), np.nan), np.full(len(d0), np.nan)
    li = L[d0[on], bi[on]]; ok = li >= 0; idx = np.flatnonzero(on)[ok]
    v[idx] = M[li[ok], bi[on][ok]]; age[idx] = (d0[on] - li)[ok]
    return v, age


# ---- the signal matrices on the RV grid (dates x bonds)
_rv_phi = np.full((RV_ND, RV_NB), np.nan); _rv_month = RV_DATES.to_period('M').astype(str)
for (_m, _c), _p in {(r_['month'], CS_CODE[r_['call_structure']]): r_['phi'] for _, r_ in rv_params.iterrows()}.items():
    _rv_phi[(_rv_month == _m)[:, None] & (RV_CS == _c)] = _p
_vine = np.full((RV_ND, RV_NB), np.nan); _vc = VC['cond']; _vine[_vc['d'].to_numpy(), _vc['bi'].to_numpy()] = _vc['cpit selected vine'].to_numpy(float) - 0.5
BT_SIG_MATS = {'ss': -RV_SSM, 'gap': RV_GCUR, 'rv': (RV_CHAIN - RVSS['mu']) - _rv_phi ** BT_H_RECORD * RVSS['x'], 'vine': _vine, 'risk': rv_wide(resid, 'target_abs_vol')}
for k_, M_ in BT_SIG_MATS.items():
    BT[f'a_{k_}'], BT[f'age_{k_}'] = bt_last_obs(M_, _d0, _bi, _on)
# the level signal needs the trailing residual window ending at the bond's last residual date
_lv_v, _lv_age = bt_last_obs(_E, _d0, _bi, _on)
_lvl = np.full(len(BT), np.nan); _okl = np.isfinite(_lv_age)
_lvl[_okl] = rfq_signals((_d0[_okl] - _lv_age[_okl]).astype(np.int64) + 1, _bi[_okl], RV_SSM, _E, RV_GCUR, RQ_LEVEL_W)['sig_level']
BT['a_level'], BT['age_level'] = _lvl, np.where(np.isfinite(_lvl), _lv_age, np.nan)
BT['on_model_grid'] = _on
BT['e_street'] = np.where(BT['group'].isin(['won, cover', 'lost, printed']), 100.0 * (BT['threshold_y'] - BT['signalyield']), np.nan)

# ---- coverage waterfall: how much of the desk's flow the factor model can speak to
_cov = {'requests': len(BT), 'bond in the factor-model universe': float((_bi >= 0).mean()), '... with a model date before the request': float(_on.mean())}
for k_ in list(BT_SIGNALS) + ['risk']:
    _cov[f'signal {k_} available'] = float(BT[f'a_{k_}'].notna().mean())
print('Coverage of the requests by the model inputs (share of all requests):'); display(pd.Series(_cov).to_frame('share').round(4))
_age = pd.DataFrame({k_: BT[f'age_{k_}'].describe(percentiles=[0.5, 0.9])[['count', '50%', '90%']] for k_ in list(BT_SIGNALS) + ['risk']}).T
print('Age of the signal at the request (model business days since the bond\'s last observation):'); display(_age.round(1))
print('Signal availability by outcome group and month (share of requests):')
display(BT.assign(any_signal=BT[[f'a_{k_}' for k_ in BT_SIGNALS]].notna().any(axis=1)).pivot_table(index='group', columns='month', values='any_signal', aggfunc='mean').round(3))

# ---- what each signal knows: rank IC with the street error now (e) and with the markout of the traded level (M at the record exit), by month
_ic_rows = []
for k_, name in BT_SIGNALS.items():
    for tgt, col in [('street error now, e', 'e_street'), (f'markout M (h = {BT_H_RECORD}, {BT_EXIT_RECORD})', f'M_{BT_H_RECORD}_{BT_EXIT_RECORD}')]:
        f_ = BT[['month', f'a_{k_}', col]].dropna()
        for m_, g_ in f_.groupby('month'):
            if g_[f'a_{k_}'].nunique() > 1:
                _ic_rows.append({'signal': name, 'target': tgt, 'month': m_, 'n': len(g_), 'IC': g_[f'a_{k_}'].corr(g_[col], method='spearman')})
bt_ic = pd.DataFrame(_ic_rows)
_icp = bt_ic.pivot_table(index=['target', 'signal'], columns='month', values='IC')
_icp['mean'] = _icp.mean(axis=1); _icp['months > 0'] = (_icp.drop(columns='mean') > 0).sum(axis=1); _icp['n'] = bt_ic.groupby(['target', 'signal'])['n'].sum()
print('Rank IC of each signal (positive = expected to richen) with the street error now (e = competing level - m, bp) and with the markout of the traded level (positive = the bond richened):')
display(_icp.round(3))

fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))
_cw = pd.Series(_cov).drop('requests'); ax[0].barh(_cw.index[::-1], _cw.to_numpy()[::-1], color='#4C72B0'); ax[0].set_xlim(0, 1); ax[0].set_title('Share of requests the model inputs cover', fontsize=10)
for i_, v_ in enumerate(_cw.to_numpy()[::-1]):
    ax[0].text(v_ + 0.01, i_, f'{v_:.0%}', va='center', fontsize=8)
for j_, tgt in enumerate(_icp.index.get_level_values(0).unique()):
    _h = _icp.loc[tgt].drop(columns=['mean', 'months > 0', 'n'])
    im = ax[1 + j_].imshow(_h.to_numpy(float), cmap='RdBu_r', vmin=-0.1, vmax=0.1, aspect='auto')
    _vm = float(np.nanmax(np.abs(_icp.drop(columns=['mean', 'months > 0', 'n']).to_numpy(float))))
    im = ax[1 + j_].imshow(_h.to_numpy(float), cmap='RdBu_r', vmin=-_vm, vmax=_vm, aspect='auto')
    ax[1 + j_].set_yticks(range(len(_h))); ax[1 + j_].set_yticklabels(_h.index if j_ == 0 else [], fontsize=8); ax[1 + j_].set_xticks(range(_h.shape[1])); ax[1 + j_].set_xticklabels(_h.columns, rotation=45, fontsize=8)
    for (r_, c_), v_ in np.ndenumerate(_h.to_numpy(float)):
        if np.isfinite(v_):
            ax[1 + j_].text(c_, r_, f'{v_:+.3f}', ha='center', va='center', fontsize=7)
    ax[1 + j_].set_title(f'Rank IC by month: {tgt}', fontsize=10)
fig.colorbar(im, ax=ax[2], fraction=0.03)
record('lab_rfq_backtest', signal_coverage=_cov, signal_ic=flat_records(_icp))

# %%
CELL_T('14. RFQ backtest: variants, fitted walk-forward [6]')
PRE_REGISTERED.update({
    'lab_rfq_backtest_variants': ('fixed 2026-10-20 before the variants were scored; every variant goes through the optimizer to its own optimal yield, opto = signal yield (+ alpha) '
                                  '+ optimal quantile bp / 100, and its quote to the street = opto + the request\'s logged loading (finalquoteyield - optimalyield: carry, venue, manual), '
                                  'the same loading production applied; every fitted quantity uses months strictly before the test month, and a training markout only if its '
                                  'exit print came before the test month starts. (a) alpha overlay: m\' = m + shift, shift = sd_T x sum_j beta_j z_j, z_j = normal score of signal j under '
                                  'the training sample\'s empirical CDF (0 where missing), beta = least squares of the normal score of the target on z, sd_T = 1.4826 MAD of the target; '
                                  'targets: the street error e (competing level now) and the exit error 100 (y_exit - m) (record exit); each of the five signals alone and all five; '
                                  '(b) width: w in integer bp from -Q to +Q, Q = ceil(median |charges|) of the first month; (c) risk tilt: w_i = -k (sigma_i - median sigma) / (1.4826 MAD sigma), '
                                  'same k grid, sigma = the bond\'s trailing sd of its daily yield change, scale from the first month with sigma; for (b) and (c) the walk-forward variant '
                                  'takes the grid value with the highest realised P&L per request on the earlier months; (d) pfill refit: per key, the backtest\'s own empirical fill curve '
                                  'P(win | x) on the earlier months, shrunk to the production curve with pseudo-count n0 chosen from {0, 1e1..1e5, inf} by the log-likelihood of the logged '
                                  'outcome in the last earlier month; (e) objective: value of a fill = x + c + b_key, b = realised markout - (x + adjustment) on earlier wins, '
                                  'James-Stein shrunk across keys to the pooled mean'),
})
BT_MONTHS = sorted(BT['month'].unique())
_mstart = {m: np.datetime64(pd.Period(m).start_time, 'ns') for m in BT_MONTHS}
_rec = f'{BT_H_RECORD}_{BT_EXIT_RECORD}'
_M, _TX = BT[f'M_{_rec}'].to_numpy(float), BT[f'tx_{_rec}'].to_numpy('datetime64[ns]')
_month = BT['month'].to_numpy(); _winable = BT['winnable'].to_numpy()
_m_sig, _adj, _ch = BT['signalyield'].to_numpy(float), BT['adj_yield'].to_numpy(float), BT['charges_level'].to_numpy(float)
_x_van = BT['bt_opt_bp'].to_numpy(float)


def bt_opto(m, opt_bp) -> np.ndarray:
    """Optimal yield as production builds it: signal yield + optimal quantile bp / 100."""
    return np.asarray(m, float) + np.asarray(opt_bp, float) / 100.0


def bt_final(opto) -> np.ndarray:
    """Quote to the street: the optimal yield plus the request's logged loading (carry, venue, manual), the same loading production applied."""
    return np.asarray(opto, float) + _adj


def bt_variant(family: str, opto: np.ndarray) -> dict:
    return {'family': family, 'opto': opto, 'y': bt_final(opto)}


_opt_van = bt_opto(_m_sig, _x_van)
assert np.nanmax(np.abs(100.0 * (_opt_van - BT['optimalyield'].to_numpy(float)))) < 1e-6, 'vanilla optimal yield must equal the logged optimalyield'
_y_van = bt_final(_opt_van)
assert np.array_equal(_y_van, BT['y_vanilla'].to_numpy(float), equal_nan=True), 'vanilla quote must equal the replayed final quote'


def bt_pnl(y, M: np.ndarray | None = None) -> np.ndarray:
    """Realised P&L (bp) of a final yield y per request: 100 (y - y_trade) + M on a win, 0 on a loss, NaN on a winnable request without an exit print."""
    M = _M if M is None else M; y = np.asarray(y, float)
    win = bt_outcomes(BT, y)['win'].to_numpy()
    return np.where(_winable & ~np.isfinite(M), np.nan, np.where(win, 100.0 * (y - BT['y_trade'].to_numpy(float)) + M, 0.0))


def bt_train_rows(m: str) -> np.ndarray:
    """Requests usable for fitting before test month m: earlier months, and a winnable one only if its record exit printed before m starts."""
    return (_month < m) & (~_winable | (np.isfinite(_M) & (_TX < _mstart[m])))


def bt_rank_normal(train: np.ndarray, x: np.ndarray) -> np.ndarray:
    F = ecdf_map(train); out = np.full(len(x), np.nan); ok = np.isfinite(x); out[ok] = stats.norm.ppf(F(x[ok])); return out


def bt_robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]; return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ---- (a) alpha overlays
BT_TARGETS = {'street': BT['e_street'].to_numpy(float), 'exit': 100.0 * (BT[f'yx_{_rec}'] - BT['signalyield']).to_numpy(float)}


def bt_overlay(keys: list[str], target: str) -> tuple[np.ndarray, list[dict]]:
    A = {k: BT[f'a_{k}'].to_numpy(float) for k in keys}; T = BT_TARGETS[target]; shift = np.full(len(BT), np.nan); fits = []
    for m in BT_MONTHS:
        tr = (_month < m) & np.isfinite(T) & ((_TX < _mstart[m]) if target == 'exit' else True)
        have = {k: tr & np.isfinite(A[k]) for k in keys}
        if any(h_.sum() == 0 for h_ in have.values()):
            continue
        rows = np.logical_or.reduce(list(have.values())); te = _month == m
        Ztr = np.column_stack([np.nan_to_num(bt_rank_normal(A[k][have[k]], A[k][rows])) for k in keys])
        beta = np.linalg.lstsq(Ztr, bt_rank_normal(T[rows], T[rows]), rcond=None)[0]; sd_t = bt_robust_sd(T[rows])
        Zte = np.column_stack([np.nan_to_num(bt_rank_normal(A[k][have[k]], A[k][te])) for k in keys])
        shift[te] = sd_t * Zte @ beta
        fits.append({'target': target, 'signals': '+'.join(keys), 'test month': m, 'train rows': int(rows.sum()), 'sd target (bp)': sd_t,
                     **{f'beta {k}': float(b) for k, b in zip(keys, beta)}, 'test share shifted': float((Zte != 0).any(axis=1).mean()),
                     'median |shift| when shifted (bp)': float(np.median(np.abs(shift[te][(Zte != 0).any(axis=1)]))) if (Zte != 0).any() else 0.0})
    return shift, fits


BT_VARIANTS: dict[str, dict] = {'vanilla (production replay)': bt_variant('vanilla', _opt_van)}
_fit_rows = []
for tgt in BT_TARGETS:
    for keys in [[k] for k in BT_SIGNALS] + [list(BT_SIGNALS)]:
        sh, fr = bt_overlay(keys, tgt); _fit_rows += fr
        nm = f'overlay {tgt}: {"all five" if len(keys) > 1 else keys[0]}'
        # the utility (x + w + c) p(x) does not involve m, so the optimizer on m' = m + shift returns the replayed x
        BT_VARIANTS[nm] = bt_variant(f'alpha overlay ({tgt})', bt_opto(_m_sig + sh / 100.0, _x_van))
bt_overlay_fits = pd.DataFrame(_fit_rows)
print('Alpha overlays: fitted on earlier months (beta = normal-score regression coefficient; the shift is in bp of yield, 0 where the signal is missing):')
display(bt_overlay_fits.set_index(['target', 'signals', 'test month']).round(3))

# ---- (b) width grid and (c) risk-tilt grid, every value on every request (the frontier), then the walk-forward pick; the grid holds optimal yields
_m0 = BT_MONTHS[0]
BT_Q = int(np.ceil(np.median(np.abs(_ch[_month == _m0]))))
BT_WGRID = np.arange(-BT_Q, BT_Q + 1)
_sig = BT['a_risk'].to_numpy(float); _m_risk0 = min(m for m in BT_MONTHS if np.isfinite(_sig[_month == m]).any())
_s0 = _sig[(_month == _m_risk0) & np.isfinite(_sig)]; BT_SIG_MED, BT_SIG_SCALE = float(np.median(_s0)), bt_robust_sd(_s0)
print(f'width grid: integer bp -{BT_Q} .. +{BT_Q} (Q = ceil median |charges| in {_m0}) | risk tilt: sigma median {BT_SIG_MED:.2f} bp, robust sd {BT_SIG_SCALE:.2f} bp (from {_m_risk0})')


def _bt_grids() -> pd.DataFrame:
    t0 = time.perf_counter(); out = {}
    for w in BT_WGRID:
        out[f'width {w:+d}'] = bt_opto(_m_sig, bt_replay(BT, BT_CURVES, width=float(w))['bt_opt_bp'].to_numpy())
    rr = np.isfinite(_sig); sub = BT.loc[rr, ['pfill_file', 'pfill_key', 'charges_level']].copy(); zs = (_sig[rr] - BT_SIG_MED) / BT_SIG_SCALE
    for k in BT_WGRID:
        o = _opt_van.copy(); sub['_c'] = sub['charges_level'].to_numpy() - k * zs
        o[rr] = bt_opto(_m_sig[rr], bt_replay(sub, BT_CURVES, charges_col='_c')['bt_opt_bp'].to_numpy())
        out[f'risk tilt {k:+d}'] = o   # float64: a float32 quote rounds across the exact ties at the logged yield
    print(f'  width and risk-tilt grids: {len(out)} optimal yields per request | {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out)


BT_GRID_OPTO = cached('lab_rfq_grids', _bt_grids, deps=[STAGE_KEYS['lab_rfq_replay'], list(BT_WGRID), BT_SIG_MED, BT_SIG_SCALE, frame_fingerprint(BT[['a_risk', 'charges_level']], ['a_risk', 'charges_level'])],
                      code=[bt_replay, bt_optimize, bt_opto])
assert np.array_equal(BT_GRID_OPTO['width +0'].to_numpy(float), _opt_van, equal_nan=True), 'width 0 must be vanilla, bit for bit'
BT_GRID = pd.DataFrame({c: bt_final(BT_GRID_OPTO[c].to_numpy(float)) for c in BT_GRID_OPTO.columns})   # street quotes of the grid, used by scoring and EDA
_grid_pnl = {c: bt_pnl(BT_GRID[c].to_numpy(float)) for c in BT_GRID.columns}


def bt_pick(cols: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Per test month, the grid column with the highest realised P&L per request on the earlier months (exits printed before the month); returns optimal yields."""
    o = np.full(len(BT), np.nan); picks = []
    for m in BT_MONTHS[1:]:
        tr = bt_train_rows(m); te = _month == m
        sc = {c: np.nanmean(_grid_pnl[c][tr]) for c in cols}
        best = max(sc, key=sc.get); o[te] = BT_GRID_OPTO[best].to_numpy(float)[te]
        picks.append({'test month': m, 'picked': best, 'train P&L per request (bp)': sc[best], 'vanilla on train (bp)': np.nanmean(bt_pnl(_y_van)[tr])})
    return o, picks


_wcols = [c for c in BT_GRID if c.startswith('width')]; _rcols = [c for c in BT_GRID if c.startswith('risk')]
_ow, _pw = bt_pick(_wcols); _or, _pr = bt_pick(_rcols)
_or[~np.isfinite(_sig) & np.isfinite(_or)] = _opt_van[~np.isfinite(_sig) & np.isfinite(_or)]
_or[_month < min(m for m in BT_MONTHS if m > _m_risk0)] = np.nan
BT_VARIANTS['width (walk-forward pick)'] = bt_variant('width', _ow)
BT_VARIANTS['risk tilt (walk-forward pick)'] = bt_variant('width (risk tilt, Section 10 scale)', _or)
print('Walk-forward picks on the width and risk-tilt grids:'); display(pd.DataFrame(_pw + _pr).set_index(['test month', 'picked']).round(3))

# ---- (d) pfill refit: the backtest's own fill curve per key, shrunk to production's
_thr = BT['threshold_y'].to_numpy(float); _grp = BT['group'].to_numpy()
_X = 100.0 * (_thr - _m_sig - _adj)                       # the largest concession that still wins (won: x <= X, lost printed: x < X)
_won_g, _lost_g = np.isin(_grp, ['won, cover', 'won, no cover']), _grp == 'lost, printed'
_xlog = BT['optimalQuantileBp'].to_numpy(float); _wlog = BT['won_logged'].to_numpy(bool)
_keys = BT['pfill_key'].astype(str).to_numpy(); _files = BT['pfill_file'].to_numpy()
_key_rows = pd.Series(np.arange(len(BT))).groupby(_keys).indices
BT_N0 = [0.0, 1e1, 1e2, 1e3, 1e4, 1e5, np.inf]


def bt_emp_curve(rows: np.ndarray, q: np.ndarray) -> np.ndarray:
    xw, xl = np.sort(_X[rows][_won_g[rows]]), np.sort(_X[rows][_lost_g[rows]])
    return ((len(xw) - np.searchsorted(xw, q, 'left')) + (len(xl) - np.searchsorted(xl, q, 'right'))) / max(len(rows), 1)


def bt_curve_mix(q: np.ndarray, p_prod: np.ndarray, rows: np.ndarray, n0: float) -> np.ndarray:
    n = len(rows)
    if n == 0 or n0 == np.inf:
        return p_prod
    return (n * bt_emp_curve(rows, q) + (0.0 if n0 == 0 else n0) * p_prod) / (n + (0.0 if n0 == 0 else n0))


def bt_curve_ll(n0: float, train: np.ndarray, valid: np.ndarray) -> float:
    ll = 0.0; vidx = np.flatnonzero(valid)
    for (f, k), rows in pd.DataFrame({'f': _files[valid], 'k': _keys[valid]}).groupby(['f', 'k']).indices.items():
        q, p_prod = BT_CURVES[f][k]; o = np.argsort(q); q, p_prod = q[o], p_prod[o]
        kr = _key_rows.get(k, np.zeros(0, int)); tr = kr[train[kr]]
        p = np.interp(_xlog[vidx[rows]], q, bt_curve_mix(q, p_prod, tr, n0))
        eps = 0.5 / max(len(tr), 1); p = np.clip(p, eps, 1 - eps); w = _wlog[vidx[rows]]
        ll += float(np.sum(np.where(w, np.log(p), np.log(1 - p))))
    return ll


_op = np.full(len(BT), np.nan); _pf_rows = []
for i_, m in enumerate(BT_MONTHS):
    if i_ < 2:
        continue
    m_prev = BT_MONTHS[i_ - 1]
    lls = {n0: bt_curve_ll(n0, _month < m_prev, _month == m_prev) for n0 in BT_N0}
    n0 = max(lls, key=lls.get); train = _month < m; te = _month == m
    for (f, k), rows in pd.DataFrame({'f': _files[te], 'k': _keys[te]}).groupby(['f', 'k']).indices.items():
        q, p_prod = BT_CURVES[f][k]; kr = _key_rows.get(k, np.zeros(0, int)); tr = kr[train[kr]]
        idx = np.flatnonzero(te)[rows]
        ob, _, _ = bt_optimize(q, bt_curve_mix(q, p_prod, tr, n0), _ch[idx], BT_WIDTH)
        _op[idx] = bt_opto(_m_sig[idx], ob)
    _pf_rows.append({'test month': m, 'n0 picked': n0, **{f'valid LL n0={k_:g}': v_ for k_, v_ in lls.items()}})
BT_VARIANTS['pfill refit (own curve, shrunk)'] = bt_variant('fill curve', _op)
print('Pfill refit: pseudo-count toward the production curve chosen by the previous month\'s log-likelihood (inf = production curve):'); display(pd.DataFrame(_pf_rows).set_index('test month').round(0))

# ---- (e) objective: production values a fill at x + c; the realised value adds the selection bias b, per key, shrunk across keys
_b = _M - (_xlog + 100.0 * _adj)
_oo = np.full(len(BT), np.nan); _ob_rows = []; BT_OBJ_B = np.full(len(BT), np.nan)
for m in BT_MONTHS[1:]:
    tr = bt_train_rows(m) & _wlog & np.isfinite(_b)
    if tr.sum() == 0:
        continue
    g = pd.DataFrame({'k': _keys[tr], 'b': _b[tr]}).groupby('k')['b'].agg(['mean', 'var', 'size'])
    b_bar = float(_b[tr].mean()); se2 = (g['var'].fillna(g['var'].median()) / g['size']).to_numpy()
    tau2 = max(0.0, float(np.var(g['mean'].to_numpy()) - np.mean(se2)))
    b_key = dict(zip(g.index, b_bar + (tau2 / (tau2 + se2)) * (g['mean'].to_numpy() - b_bar)))
    te = _month == m
    for (f, k), rows in pd.DataFrame({'f': _files[te], 'k': _keys[te]}).groupby(['f', 'k']).indices.items():
        q, p_prod = BT_CURVES[f][k]; idx = np.flatnonzero(te)[rows]
        BT_OBJ_B[idx] = float(b_key.get(k, b_bar))
        ob, _, _ = bt_optimize(q, p_prod, _ch[idx], BT_OBJ_B[idx][0])
        _oo[idx] = bt_opto(_m_sig[idx], ob)
    _ob_rows.append({'test month': m, 'train wins': int(tr.sum()), 'pooled b (bp)': b_bar, 'between-key sd (bp)': np.sqrt(tau2), 'keys': len(g),
                     'b per key p10 (bp)': np.percentile(list(b_key.values()), 10), 'p90': np.percentile(list(b_key.values()), 90)})
BT_VARIANTS['objective: selection-adjusted value'] = bt_variant('objective', _oo)
print('Objective: selection bias b = realised markout - production\'s assumed value, on earlier wins (record exit):'); display(pd.DataFrame(_ob_rows).set_index('test month').round(2))

# ---- each variant's own optimal yield against production's, on its test months (the street quote differs by exactly the same bp: same loading)
_aud = []
for nm, v_ in BT_VARIANTS.items():
    ok = np.isfinite(v_['opto']); d = 100.0 * (v_['opto'][ok] - _opt_van[ok]); ch_ = np.abs(d) > 1e-9
    assert np.nanmax(np.abs(100.0 * ((v_['y'][ok] - _y_van[ok]) - (v_['opto'][ok] - _opt_van[ok])))) < 1e-9, f'{nm}: quote difference must equal optimal-yield difference'
    _aud.append({'variant': nm, 'family': v_['family'], 'requests': int(ok.sum()), 'opto changed': float(ch_.mean()), 'richer (opto lower)': float((d < -1e-9).mean()),
                 'cheaper (opto higher)': float((d > 1e-9).mean()), 'shift p10 (bp)': float(np.percentile(d[ch_], 10)) if ch_.any() else 0.0,
                 'shift median (bp)': float(np.median(d[ch_])) if ch_.any() else 0.0, 'shift p90 (bp)': float(np.percentile(d[ch_], 90)) if ch_.any() else 0.0})
print('Optimal yield of each variant vs production (shift = 100 (variant opto - production opto), on requests it changes; the street quote moves by the same bp):')
display(pd.DataFrame(_aud).set_index('variant').round(3))

# %%
CELL_T('14. RFQ backtest: scoring, leaderboard and the bar [7]')
PRE_REGISTERED.update({
    'lab_rfq_backtest_bar': ('fixed 2026-10-20 before the variants were scored: each variant is scored on its own test months against vanilla on the same requests; '
                             'realised P&L per request at the record exit (other side of the trade, h = 5; revised 2026-10-20 on the desk rule before re-scoring) on the sample '
                             'where every winnable request has an exit print; the difference is '
                             'decomposed exactly into the price change on fills both make, the P&L of new wins and the P&L of wins given up; a variant passes if its weekly '
                             'paired difference has Newey-West t >= 3 and is positive in >= 4 test months (all months if fewer), and it lies above the width frontier '
                             '(vanilla with a constant width, the best grid point at or above its win rate) on the same months; the markout net of the day\'s common '
                             'markout and horizons 1 and 3 are reported as robustness'),
})
BT_BAR_T, BT_BAR_MONTHS = 3.0, 4
_usd = BT['usd_per_bp'].to_numpy(float); _week = BT['week'].to_numpy()
BT_EXIT_NAMES = {_rec: f'record (other side, h = {BT_H_RECORD})', f'{BT_H_RECORD}_Xm': f'net of common markout, h = {BT_H_RECORD}',
                 **{f'{h}_{BT_EXIT_RECORD}': f'other side, h = {h}' for h in BT_H if h != BT_H_RECORD}}


def bt_exante(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Production's own expected P&L of quoting y: (x + c) p_prod(x), x = 100 (y - adjustment - m), on the request's curve."""
    u = np.full(len(rows), np.nan)
    for (f, k), ii in pd.DataFrame({'f': _files[rows], 'k': _keys[rows]}).groupby(['f', 'k']).indices.items():
        q, p = BT_CURVES[f][k]; o = np.argsort(q); r = rows[ii]
        x = 100.0 * (y[r] - _adj[r] - _m_sig[r]); u[ii] = (x + _ch[r]) * np.interp(x, q[o], p[o])
    return u


def bt_pnl_rows(rows: np.ndarray, y: np.ndarray, win: np.ndarray, M: np.ndarray) -> np.ndarray:
    Mr = M[rows]
    return np.where(_winable[rows] & ~np.isfinite(Mr), np.nan, np.where(win, 100.0 * (y - BT['y_trade'].to_numpy(float)[rows]) + Mr, 0.0))


def bt_score(name: str, y: np.ndarray, exits: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    rows = np.flatnonzero(np.isfinite(y)); months = sorted(set(_month[rows]))
    yv, y0 = y[rows], _y_van[rows]
    ov, o0 = bt_outcomes(BT.iloc[rows], yv), bt_outcomes(BT.iloc[rows], y0)
    wv, w0 = ov['win'].to_numpy(), o0['win'].to_numpy(); grp = ov['group'].to_numpy()
    pv, p0 = bt_pnl_rows(rows, yv, wv, exits[_rec]), bt_pnl_rows(rows, y0, w0, exits[_rec])
    S = np.isfinite(pv); nS = int(S.sum()); d = pv - p0
    both, new, gone = wv & w0 & S, wv & ~w0 & S, w0 & ~wv & S
    per = pd.DataFrame({'week': _week[rows], 'month': _month[rows], 'd': d, 'S': S, 'usd_v': pv * _usd[rows], 'usd_0': p0 * _usd[rows]})
    wk = per[per['S']].groupby('week')['d'].mean(); mm = per[per['S']].groupby('month')['d'].mean()
    ex_v, ex_0 = bt_exante(y, rows), bt_exante(_y_van, rows)
    res = {'variant': name, 'family': BT_VARIANTS[name]['family'], 'test months': f'{months[0]}..{months[-1]}', 'months': len(months), 'requests': len(rows),
           'win rate': float(wv.mean()), 'win rate, vanilla': float(w0.mean()), 'new wins': int((wv & ~w0).sum()), 'wins given up': int((w0 & ~wv).sum()),
           'bp left on table (won, cover)': float(ov['d_bp'][wv & (grp == 'won, cover')].mean()), 'bp left on table, vanilla': float(o0['d_bp'][w0 & (grp == 'won, cover')].mean()),
           'crossing on new wins (bp)': float(ov['d_bp'][wv & ~w0 & (grp == 'lost, printed')].mean()) if (wv & ~w0 & (grp == 'lost, printed')).any() else np.nan,
           'ex-ante P&L per request (bp)': float(np.nanmean(ex_v)), 'ex-ante P&L, vanilla (bp)': float(np.nanmean(ex_0)),
           'P&L sample share': nS / len(rows), 'P&L per request (bp)': float(np.nanmean(pv)), '... vanilla (bp)': float(np.nanmean(p0)), 'delta per request (bp)': float(np.nanmean(d)),
           'delta: common fills price': float(np.sum(100.0 * (yv - y0)[both]) / nS), 'delta: new wins': float(np.sum(pv[new]) / nS), 'delta: wins given up': float(-np.sum(p0[gone]) / nS),
           'delta $ per month ($k)': float((np.nansum(per['usd_v']) - np.nansum(per['usd_0'])) / 1e3 / len(months)),
           'NW t (weekly)': nw_t(wk.to_numpy()), 'months > 0': int((mm > 0).sum())}
    for key, M_ in exits.items():
        a, b = bt_pnl_rows(rows, yv, wv, M_), bt_pnl_rows(rows, y0, w0, M_)
        res[f'delta, {BT_EXIT_NAMES[key]} (bp)'] = float(np.nanmean(a - b))
    res['exits with delta > 0'] = int(sum(res[f'delta, {BT_EXIT_NAMES[k]} (bp)'] > 0 for k in exits))
    return res, per.assign(pv=pv, p0=p0, win_v=wv, win_0=w0, rows=rows)


_exits = {k_: BT[f'M_{k_}'].to_numpy(float) for k_ in BT_EXIT_NAMES}
_score_rows, BT_PER = [], {}
for nm, v_ in BT_VARIANTS.items():
    r_, per_ = bt_score(nm, v_['y'], _exits); _score_rows.append(r_); BT_PER[nm] = per_

# ---- the width frontier on each variant's own months: the best constant-width P&L among grid points whose win rate is at least the variant's
_gw = {c: BT_GRID[c].to_numpy(float) for c in _wcols}
for r_ in _score_rows:
    rows = BT_PER[r_['variant']]['rows'].to_numpy()
    pts = []
    for c, yg in _gw.items():
        wg = bt_outcomes(BT.iloc[rows], yg[rows])['win'].to_numpy(); pg = bt_pnl_rows(rows, yg[rows], wg, _M)
        pts.append((float(wg.mean()), float(np.nanmean(pg))))
    feas = [p for w_, p in pts if w_ >= r_['win rate'] - 1e-12]
    r_['width frontier P&L at its win rate (bp)'] = max(feas) if feas else np.nan
    r_['above width frontier (bp)'] = r_['P&L per request (bp)'] - r_['width frontier P&L at its win rate (bp)']
bt_board = pd.DataFrame(_score_rows).set_index('variant')
bt_board['passes'] = ((bt_board['NW t (weekly)'] >= BT_BAR_T) & (bt_board['months > 0'] >= np.minimum(BT_BAR_MONTHS, bt_board['months']))
                      & (bt_board['above width frontier (bp)'] >= 0) & (bt_board.index != 'vanilla (production replay)'))
bt_board = bt_board.sort_values('NW t (weekly)', ascending=False, na_position='last')
bt_board.insert(0, 'rank', np.arange(1, len(bt_board) + 1))

_show = ['rank', 'family', 'test months', 'win rate', 'win rate, vanilla', 'new wins', 'wins given up', 'P&L per request (bp)', '... vanilla (bp)', 'delta per request (bp)',
         'delta $ per month ($k)', 'NW t (weekly)', 'months > 0', 'months', 'exits with delta > 0', 'above width frontier (bp)', 'passes']
print(f'LEADERBOARD: realised P&L at the record exit ({BT_EXIT_LABEL[BT_EXIT_RECORD]}, h = {BT_H_RECORD}), each variant against vanilla on its own test months, ranked by weekly NW t; '
      'the level of P&L carries the market, the delta against vanilla is the comparison:')
display(bt_board[_show].round(4))
print('Where the difference comes from (bp per request in the P&L sample; the three parts sum to the delta) and the quote quality:')
display(bt_board[['delta per request (bp)', 'delta: common fills price', 'delta: new wins', 'delta: wins given up', 'bp left on table (won, cover)', 'bp left on table, vanilla',
                  'crossing on new wins (bp)', 'ex-ante P&L per request (bp)', 'ex-ante P&L, vanilla (bp)', 'P&L sample share']].round(4))
print(f'Robustness: the same delta under the other exit definitions (bp per request; {len(_exits)} definitions, the count of positive ones beside):')
display(bt_board[[f'delta, {n_} (bp)' for n_ in BT_EXIT_NAMES.values()] + ['exits with delta > 0']].round(4))
_chk = bt_board.loc['vanilla (production replay)']
assert abs(_chk['delta per request (bp)']) < 1e-9 and _chk['new wins'] == 0 and _chk['wins given up'] == 0, 'vanilla must score zero against itself'
_dec = bt_board['delta: common fills price'] + bt_board['delta: new wins'] + bt_board['delta: wins given up'] - bt_board['delta per request (bp)']
assert np.nanmax(np.abs(_dec.to_numpy(float))) < 1e-9, 'the decomposition must add up'
BT_VERDICT = (f'{int(bt_board["passes"].sum())} of {len(bt_board) - 1} variants pass the bar' + (f': {bt_board.index[bt_board["passes"]].tolist()}' if bt_board['passes'].any() else '')
              + f' | best by NW t: {bt_board.index[1] if bt_board.index[0] == "vanilla (production replay)" else bt_board.index[0]}')
print('VERDICT (lab_rfq_backtest_bar):', BT_VERDICT)
record('lab_rfq_backtest', leaderboard=flat_records(bt_board), verdict=BT_VERDICT, overlay_fits=flat_records(bt_overlay_fits))

# %%
CELL_T('14. RFQ backtest: comparison EDA [8]')
_fam_col = {'vanilla': '#000000', 'alpha overlay (street)': '#4C72B0', 'alpha overlay (exit)': '#55A868', 'width': '#C44E52', 'width (risk tilt, Section 10 scale)': '#DD8452',
            'fill curve': '#8172B3', 'objective': '#937860'}
_b = bt_board.drop(index='vanilla (production replay)')

# ---- common window: the months every variant is defined, so the frontier and the heatmap compare like with like
BT_COMMON = sorted(set.intersection(*[set(_month[np.isfinite(v_['y'])]) for v_ in BT_VARIANTS.values()]))
_cw = np.flatnonzero(np.isin(_month, BT_COMMON))
print(f'common window for the frontier: {BT_COMMON[0]}..{BT_COMMON[-1]} ({len(_cw):,} requests); every number below is against vanilla on the same requests')


def _point(y: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    w = bt_outcomes(BT.iloc[_cw], y[_cw])['win'].to_numpy(); p = bt_pnl_rows(_cw, y[_cw], w, M)
    return float(w.mean()), float(np.nanmean(p))


_front = {}
for ek in [_rec, f'{BT_H_RECORD}_Xm', f'1_{BT_EXIT_RECORD}']:
    M_ = _exits[ek]; _, p0_ = _point(_y_van, M_)
    _front[ek] = {'vanilla': p0_, 'width': [(c_, w_, p_ - p0_) for c_ in _wcols for w_, p_ in [_point(BT_GRID[c_].to_numpy(float), M_)]],
                  'risk': [(c_, w_, p_ - p0_) for c_ in _rcols for w_, p_ in [_point(BT_GRID[c_].to_numpy(float), M_)]],
                  'variants': {nm: (w_, p_ - p0_) for nm, v_ in BT_VARIANTS.items() for w_, p_ in [_point(v_['y'], M_)]}}

fig, ax = plt.subplots(2, 3, figsize=(22, 12))
# (1) leaderboard: delta per request with a 2-se band from the weekly NW t
_se = (_b['delta per request (bp)'] / _b['NW t (weekly)']).abs()
_o = _b.sort_values('delta per request (bp)')
ax[0, 0].barh(_o.index, _o['delta per request (bp)'], xerr=2 * _se.reindex(_o.index), color=[_fam_col.get(f, '#999999') for f in _o['family']], alpha=0.85)
ax[0, 0].axvline(0, color='k', lw=0.8)
for i_, (nm, r_) in enumerate(_o.iterrows()):
    ax[0, 0].text(ax[0, 0].get_xlim()[1], i_, f' t {r_["NW t (weekly)"]:+.1f} | {int(r_["months > 0"])}/{int(r_["months"])} mo', va='center', fontsize=7)
ax[0, 0].tick_params(axis='y', labelsize=7); ax[0, 0].set_xlabel('realised P&L minus vanilla, bp per request (bar = 2 NW se)')
ax[0, 0].set_title(f'Leaderboard vs vanilla, record exit (other side, h = {BT_H_RECORD})', fontsize=10)
# (2) robustness: the delta under every exit definition, one row per variant in leaderboard order
_rb = _o[[f'delta, {n_} (bp)' for n_ in BT_EXIT_NAMES.values()]]; _rb.columns = list(BT_EXIT_NAMES.values())
_vr = float(np.nanmax(np.abs(_rb.to_numpy(float))))
im = ax[0, 2].imshow(_rb.to_numpy(float), cmap='RdBu', vmin=-_vr, vmax=_vr, aspect='auto')
ax[0, 2].set_yticks(range(len(_rb))); ax[0, 2].set_yticklabels(_rb.index, fontsize=7); ax[0, 2].set_xticks(range(_rb.shape[1])); ax[0, 2].set_xticklabels(_rb.columns, rotation=20, fontsize=7, ha='right')
for (r_, c_), v_ in np.ndenumerate(_rb.to_numpy(float)):
    ax[0, 2].text(c_, r_, f'{v_:+.3f}', ha='center', va='center', fontsize=6)
fig.colorbar(im, ax=ax[0, 2], fraction=0.03); ax[0, 2].set_title('Does the sign survive other exits? delta vs vanilla, bp (blue = better)', fontsize=10)
# (3-5) the frontier under three exits, as P&L minus vanilla: constant width (line), risk tilt (dashed), variants (points); vanilla at (its win rate, 0)
for a_, (ek, fr) in zip([ax[0, 1], ax[1, 0], ax[1, 1]], _front.items()):
    wx, wy = zip(*[(w_, p_) for _, w_, p_ in fr['width']]); rx, ry = zip(*[(w_, p_) for _, w_, p_ in fr['risk']])
    a_.plot(wx, wy, '-o', color='#C44E52', ms=3, label=f'constant width -{BT_Q}..+{BT_Q} bp'); a_.plot(rx, ry, '--o', color='#DD8452', ms=3, label=f'risk tilt -{BT_Q}..+{BT_Q}')
    for c_, w_, p_ in fr['width']:
        if c_ in (_wcols[0], _wcols[-1]):
            a_.annotate(c_.replace('width ', 'w '), (w_, p_), fontsize=7, xytext=(3, -9), textcoords='offset points', color='#C44E52')
    for nm, (w_, p_) in fr['variants'].items():
        fam = BT_VARIANTS[nm]['family']
        a_.scatter(w_, p_, s=90 if fam == 'vanilla' else 25, marker='*' if fam == 'vanilla' else 'o', color=_fam_col.get(fam, '#999999'), zorder=3)
        if fam in ('vanilla', 'objective', 'fill curve') or nm == _b.index[0]:
            a_.annotate(nm.split(':')[0] if fam != 'vanilla' else 'vanilla', (w_, p_), fontsize=7, xytext=(4, 4), textcoords='offset points')
    a_.set_xlabel(f'win rate ({BT_COMMON[0]}..{BT_COMMON[-1]})'); a_.set_ylabel('P&L minus vanilla, bp per request'); a_.axhline(0, color='k', lw=0.5)
    a_.set_title(f'Frontier vs vanilla, {BT_EXIT_NAMES[ek]} (vanilla level {fr["vanilla"]:+.3f} bp)', fontsize=10); a_.legend(fontsize=7)
# (6) decomposition of each variant's delta
_dc = _o[['delta: common fills price', 'delta: new wins', 'delta: wins given up']]
_left_p = np.zeros(len(_dc)); _left_n = np.zeros(len(_dc))
for c_, col_ in zip(_dc.columns, ['#4C72B0', '#55A868', '#C44E52']):
    v_ = _dc[c_].to_numpy(float)
    ax[1, 2].barh(_dc.index, v_, left=np.where(v_ >= 0, _left_p, _left_n), color=col_, label=c_.replace('delta: ', ''))
    _left_p += np.where(v_ >= 0, v_, 0); _left_n += np.where(v_ < 0, v_, 0)
ax[1, 2].axvline(0, color='k', lw=0.8); ax[1, 2].tick_params(axis='y', labelsize=7); ax[1, 2].legend(fontsize=7); ax[1, 2].set_xlabel('bp per request')
ax[1, 2].set_title('Where each variant\'s difference comes from (record exit)', fontsize=10)
savefig('14_rfq_backtest_leaderboard_frontier')

# ---- month by month, the cumulative dollars of the leaders, and the market vanilla traded in
_mm = pd.DataFrame({nm: BT_PER[nm][BT_PER[nm]['S']].groupby('month')['d'].mean() for nm in _b.index}).T
_wk = pd.DataFrame({nm: BT_PER[nm][BT_PER[nm]['S']].groupby('week').apply(lambda g: np.nansum(g['usd_v']) - np.nansum(g['usd_0']), include_groups=False) for nm in _b.index[:6]})
_pv = BT_PER['vanilla (production replay)']; _pv = _pv[_pv['S']]
_ctx = pd.DataFrame({'vanilla P&L per request (bp)': _pv.groupby('month')['p0'].mean(),
                     'common markout (bp)': BT[BT['winnable']].groupby('month')[f'mkt_{BT_H_RECORD}'].mean()})
fig, ax = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={'width_ratios': [1.3, 1, 0.8]})
_vm = float(np.nanmax(np.abs(_mm.to_numpy(float))))
im = ax[0].imshow(_mm.to_numpy(float), cmap='RdBu', vmin=-_vm, vmax=_vm, aspect='auto')
ax[0].set_yticks(range(len(_mm))); ax[0].set_yticklabels(_mm.index, fontsize=7); ax[0].set_xticks(range(_mm.shape[1])); ax[0].set_xticklabels(_mm.columns, fontsize=8)
for (r_, c_), v_ in np.ndenumerate(_mm.to_numpy(float)):
    if np.isfinite(v_):
        ax[0].text(c_, r_, f'{v_:+.3f}', ha='center', va='center', fontsize=6)
fig.colorbar(im, ax=ax[0], fraction=0.03); ax[0].set_title('P&L minus vanilla by month (bp per request; blue = better)', fontsize=10)
for nm in _wk.columns:
    ax[1].plot(_wk.index, _wk[nm].cumsum().to_numpy() / 1e3, label=nm, color=_fam_col.get(BT_VARIANTS[nm]['family'], '#999999'), lw=1.4)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_ylabel('$k, cumulative vs vanilla'); ax[1].legend(fontsize=7); ax[1].set_title('Cumulative dollars vs vanilla, the six highest-ranked variants', fontsize=10)
_xm = np.arange(len(_ctx)); ax[2].bar(_xm - 0.2, _ctx.iloc[:, 0], 0.4, color='#000000', label=_ctx.columns[0]); ax[2].bar(_xm + 0.2, _ctx.iloc[:, 1], 0.4, color='#999999', label=_ctx.columns[1])
ax[2].set_xticks(_xm); ax[2].set_xticklabels(_ctx.index, fontsize=8); ax[2].axhline(0, color='k', lw=0.6); ax[2].legend(fontsize=7)
ax[2].set_title('Context: the level vanilla earned and the day\'s common markout, by month', fontsize=10)
savefig('14_rfq_backtest_by_month')
print('Context by month (bp per request): vanilla\'s level and the common markout every variant shares; the comparison above is net of both:'); display(_ctx.round(3))

# ---- where the leaders gain or lose: segments of the request (delta per request, bp; request counts beside)
_seg_cols = {'tenor_bucket': 'tenor', 'qty_bucket': 'size', 'call_bucket': 'call', 'print_type': 'trade type', 'manual': 'manual', 'on_model_grid': 'bond in the factor model'}
_seg_rows = []
for nm in _b.index[:4]:
    per = BT_PER[nm][BT_PER[nm]['S']]; idx = per['rows'].to_numpy()
    for col, lab in _seg_cols.items():
        g = pd.DataFrame({'cell': BT[col].to_numpy()[idx].astype(str), 'd': per['d'].to_numpy()}).groupby('cell')['d'].agg(['mean', 'size'])
        for cell, r_ in g.iterrows():
            _seg_rows.append({'variant': nm, 'dimension': lab, 'cell': cell, 'delta (bp)': r_['mean'], 'requests': int(r_['size'])})
bt_seg = pd.DataFrame(_seg_rows)
print('Where the four highest-ranked variants gain or lose (delta per request vs vanilla, bp, P&L sample):')
display(bt_seg.pivot_table(index=['dimension', 'cell'], columns='variant', values='delta (bp)').round(4).join(
    bt_seg.drop_duplicates(['dimension', 'cell']).set_index(['dimension', 'cell'])['requests']))
_fr = pd.DataFrame([{'exit': BT_EXIT_NAMES[ek], 'grid': g_, 'value': c_, 'win rate': w_, 'P&L minus vanilla (bp)': p_} for ek, fr in _front.items() for g_ in ('width', 'risk') for c_, w_, p_ in fr[g_]])
print('The frontier in numbers: each constant width and risk tilt on the common window, P&L minus vanilla by exit (the best value can differ by exit):')
display(_fr.pivot_table(index=['grid', 'value'], columns='exit', values='P&L minus vanilla (bp)', sort=False).round(4).join(
    _fr[_fr['exit'] == BT_EXIT_NAMES[_rec]].set_index(['grid', 'value'])['win rate']))
record('lab_rfq_backtest', common_window=f'{BT_COMMON[0]}..{BT_COMMON[-1]}', frontier=_fr.round(5).to_dict(orient='records'), by_month=flat_records(_mm),
       context_by_month=flat_records(_ctx), segments=bt_seg.round(5).to_dict(orient='records'))

# %% [markdown]
# ### 14b. The book: IPCA as the desk's risk model, and risk-aware acceptance
#
# **Why.** IPCA (Kelly, Pruitt and Su) is a portfolio model. Returns load on a few latent factors through betas that are linear in observable characteristics, $\beta_{i,t} = \Gamma' z_{i,t}$. The paper uses it to build managed portfolios and the maximum-Sharpe (tangency) combination of factors.
#
# The desk cannot pick bonds; the RFQ flow offers them. What the desk chooses is **which requests to fill, and at what level**. The book is the set of fills still held. So the paper's idea maps to this:
#
# 1. Use IPCA, plus the MMD curve (the model's target is net of it), as the **risk model of the book**.
# 2. Price each request at its expected value **minus its marginal contribution to the book's risk**. This is the mean-variance acceptance rule, the RFQ analogue of a tangency portfolio.
#
# The aim is similar P&L with less DV01 and factor risk.
#
# **Risk model** (daily, bp of yield):
# $$\Delta y_{i} = \Delta \text{MMD}_{\tau(i)} + \beta_i' f + \varepsilon_i,\qquad e_i = u_i\,[\,\mathbb 1_{\tau(i)},\ \beta_i\,],\qquad v_i = u_i^2 \sigma_i^2$$
#
# - $u_i$ is the fill's $ per bp (par × DV01 from the pricing service).
# - $\tau(i)$ is the MMD tenor nearest the median years to maturity of the request's tenor bucket.
# - $\beta_i$ is the bond's IPCA beta at its last model date before the request. For a bond outside the model universe, it is the mean beta of in-universe requests on earlier days in the same (tenor, call) bucket.
# - $\sigma_i$ is the bond's residual sd, with the same bucket fallback (using the mean of $\sigma^2$).
# - $\Sigma$ is the covariance of $[\Delta\text{MMD}_\tau, f]$ over all days strictly before the request date. It is used only once it has at least as many days as dimensions + 1.
#
# **Book.** A fill is held for $h = 5$ business days, the record exit horizon. The book at the start of day $d$ is $E_d = \sum e_i$ over the policy's fills on days $d-h+1,\dots,d-1$, plus idiosyncratic variance $I_d = \sum v_i$.
#
# - Ex-ante daily risk: $\sqrt{E_d'\Sigma E_d + I_d}$ ($).
# - Realised risk: the sd of the book's daily rate-plus-factor move $E_d' R_d$, where $R_d = [\Delta\text{MMD}_\tau, f]$ on day $d$.
# - Net DV01: $\sum u_i$ over open fills.
#
# **Risk-aware policy.** Mean-variance gives the change in utility from a fill as $p(x)\,[(x + c + b)\,u_i - \tfrac{\gamma}{2} h\,(2 e_i'\Sigma E_d + e_i'\Sigma e_i + v_i)]$. In bp this is an extra charge
# $$\rho_i = \gamma\, s_i,\qquad s_i = \frac{h\,(2 e_i'\Sigma E_d + e_i'\Sigma e_i + v_i)}{2u_i}.$$
# The policy runs the production optimizer with $c_i - \rho_i$ to get its own optimal yield, then adds the same loading as before.
#
# - $\gamma_k = k / \text{median}(s)$, with the median taken over the vanilla book in the first month with $\Sigma$. So $k$ is the median extra charge in bp.
# - $k$ runs over $0,\dots,Q$, the width grid's range.
# - The walk-forward pick takes the $k$ with the highest weekly $ P&L Sharpe on earlier months, which is the paper's criterion.
# - It is applied on top of vanilla and on top of the objective variant.
#
# **Replay.** The replay is sequential by business day. Each policy carries its own book: its fills come from its own quotes and the same outcome rules. Requests within a day are priced against the start-of-day book.
#
# | Cell | Content |
# |---|---|
# | [9] | Risk model, request exposures, and the books of vanilla and the objective: net DV01, factor exposures, ex-ante and realised risk |
# | [10] | Risk-aware replay (both bases, every k), walk-forward pick, P&L-versus-risk table and frontier, time series |

# %%
CELL_T('14b. The book under the IPCA risk model [9]')
PRE_REGISTERED.update({
    'lab_rfq_book': ('fixed 2026-10-20 before the risk-aware variants were scored: daily yield move of a bond = MMD move at its tenor + IPCA beta\' f + residual; '
                     'tenor = the MMD tenor nearest the median years to maturity of the request\'s tenor bucket (model-universe bonds); beta = the bond\'s IPCA beta at its last '
                     'model date before the request, else the mean beta of model-universe requests on earlier days in the same (tenor bucket, call bucket), else tenor bucket, '
                     'else all; sigma^2 likewise from the bond\'s residual sd; Sigma = covariance of [MMD tenor moves, IPCA factor realisations] over all days strictly before '
                     'the request date, used once it has >= dimensions + 1 days; a fill is held h = 5 business days (the record exit horizon); exposure e = $/bp x [tenor one-hot, '
                     'beta], idiosyncratic variance v = ($/bp)^2 sigma^2; book at the start of day d = the policy\'s fills on days d-h+1..d-1'),
})
BT_HOLD = BT_H_RECORD
_pb_days = pd.DatetimeIndex(np.sort(BT['date'].unique())); _pb_di = _pb_days.get_indexer(BT['date']); _pb_nd = len(_pb_days)
_pb_u = np.nan_to_num(BT['usd_per_bp'].to_numpy(float)); _pb_u[~_winable] = 0.0

# ---- MMD curve moves (bp) by tenor year; each tenor bucket mapped to the nearest MMD tenor of its median years to maturity
_bc = benchmark_curve.assign(date=pd.to_datetime(benchmark_curve['date']).dt.normalize())
_mmd = _bc.groupby(['date', 'tenor_years'])['benchmark_yield'].median().unstack().sort_index()
_mat = pd.to_datetime(pd.to_numeric(raw_panel.assign(cusip=raw_panel['cusip'].astype(str)).groupby('cusip')['maturity'].first(), errors='coerce'), unit='ms')
_ytm = (BT['cusip'].astype(str).map(_mat) - BT['date']).dt.days / 365.25
_tyb = _ytm.groupby(BT['tenor_bucket']).median()
_ten = np.asarray(_mmd.columns, float)
BT_TENOR = {b: float(_ten[np.argmin(np.abs(_ten - y))]) for b, y in _tyb.items() if np.isfinite(y)}
_t_all = float(_ten[np.argmin(np.abs(_ten - float(_ytm.median())))])
BT_TENORS = sorted(set(BT_TENOR.values()) | {_t_all})
_pb_ti = BT['tenor_bucket'].map(BT_TENOR).fillna(_t_all).map({t: j for j, t in enumerate(BT_TENORS)}).to_numpy(int)
print('tenor bucket -> MMD tenor (median years to maturity of in-universe requests):', {b: (round(float(_tyb[b]), 1), BT_TENOR[b]) for b in BT_TENOR},
      f'| no tenor bucket: {int(BT["tenor_bucket"].isna().sum()):,} requests -> {_t_all:g}y')

# ---- the joint daily risk-factor moves and the expanding covariance strictly before each request date
_fo = resid.groupby('date')[['f1', 'f2', 'f3']].first()
BT_RISK_R = (100.0 * _mmd[BT_TENORS].diff()).join(_fo, how='inner').dropna()
_Rv = BT_RISK_R.to_numpy(float); _D = _Rv.shape[1]; _nT = len(BT_TENORS)
_cs1 = np.vstack([np.zeros(_D), np.cumsum(_Rv, 0)]); _cs2 = np.concatenate([np.zeros((1, _D, _D)), np.cumsum(_Rv[:, :, None] * _Rv[:, None, :], 0)])
_nb = BT_RISK_R.index.searchsorted(_pb_days, side='left')
BT_SIGMA = {}
for j, n in enumerate(_nb):
    if n >= _D + 1:
        mu = _cs1[n] / n; BT_SIGMA[j] = (_cs2[n] - n * np.outer(mu, mu)) / (n - 1)
_sig_first = _pb_days[min(BT_SIGMA)]
print(f'risk factors: {_nT} MMD tenors {BT_TENORS} + 3 IPCA factors, {len(BT_RISK_R)} days {BT_RISK_R.index.min().date()}..{BT_RISK_R.index.max().date()}; '
      f'Sigma available from {_sig_first.date()} ({_D + 1} days needed)')

# ---- request betas and residual variance: own (point in time), else the bucket mean over earlier days
_bi2 = pd.Index(RV_BONDS.astype(str)).get_indexer(BT['cusip'].astype(str)); _d02 = RV_DATES.searchsorted(BT['date'].to_numpy(), side='left') - 1; _on2 = (_bi2 >= 0) & (_d02 >= 0)
_own = np.column_stack([bt_last_obs(rv_wide(resid, f'beta{k}'), _d02, _bi2, _on2)[0] for k in (1, 2, 3)] + [bt_last_obs(rv_wide(resid, 'resid_sd'), _d02, _bi2, _on2)[0] ** 2])
_has = np.isfinite(_own).all(axis=1)


def bt_bucket_before(vals: np.ndarray, has: np.ndarray, keys: np.ndarray, di: np.ndarray) -> np.ndarray:
    """Per row, the mean of vals over rows with has on days strictly before the row's day and the same key; NaN where none."""
    ku, kinv = np.unique(keys, return_inverse=True)
    S = np.zeros((len(ku), _pb_nd + 1, vals.shape[1])); C = np.zeros((len(ku), _pb_nd + 1))
    np.add.at(S, (kinv[has], di[has] + 1), vals[has]); np.add.at(C, (kinv[has], di[has] + 1), 1.0)
    S, C = S.cumsum(1), C.cumsum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return S[kinv, di] / C[kinv, di][:, None]


_tb_ = BT['tenor_bucket'].astype(str).to_numpy(dtype=str); _cb_ = BT['call_bucket'].astype(str).to_numpy(dtype=str)
_fill = np.where(_has[:, None], _own, np.nan); _src = np.where(_has, 'own', '').astype(object)
for lab, keys in [('tenor x call bucket', np.char.add(np.char.add(_tb_, '|'), _cb_)), ('tenor bucket', _tb_), ('all', np.zeros(len(BT), dtype='<U1'))]:
    m_ = bt_bucket_before(_own, _has, keys, _pb_di); need = ~np.isfinite(_fill).all(axis=1) & np.isfinite(m_).all(axis=1)
    _fill[need] = m_[need]; _src[need] = lab
_src[~np.isfinite(_fill).all(axis=1)] = 'none'; _fill = np.nan_to_num(_fill)
BT_BETA, BT_IDV = _fill[:, :3], _fill[:, 3]
BT_EXPO = np.zeros((len(BT), _D)); BT_EXPO[np.arange(len(BT)), _pb_ti] = _pb_u; BT_EXPO[:, _nT:] = _pb_u[:, None] * BT_BETA
BT_V = _pb_u ** 2 * BT_IDV
print('source of beta and sigma on winnable requests:', pd.Series(_src[_winable]).value_counts(normalize=True).round(3).to_dict())


def bt_book(win: np.ndarray) -> dict:
    """Daily book of a policy's fills: exposure at the start of each day (fills on days d-h+1..d-1), ex-ante sd, realised rate+factor move, net DV01."""
    w = win & (_pb_u > 0)
    F = np.zeros((_pb_nd + 1, _D)); I = np.zeros(_pb_nd + 1); U = np.zeros(_pb_nd + 1)
    np.add.at(F, _pb_di[w] + 1, BT_EXPO[w]); np.add.at(I, _pb_di[w] + 1, BT_V[w]); np.add.at(U, _pb_di[w] + 1, _pb_u[w])
    F, I, U = F.cumsum(0), I.cumsum(0), U.cumsum(0)
    j = np.arange(_pb_nd); lo = np.maximum(0, j - (BT_HOLD - 1))
    E, Iv, Uv = F[j] - F[lo], I[j] - I[lo], U[j] - U[lo]
    sd = np.array([np.sqrt(E[k] @ BT_SIGMA[k] @ E[k] + Iv[k]) if k in BT_SIGMA else np.nan for k in j])
    ri = BT_RISK_R.index.get_indexer(_pb_days); mv = np.where(ri >= 0, np.einsum('ij,ij->i', E, _Rv[np.maximum(ri, 0)]), np.nan)
    return {'E': E, 'I': Iv, 'dv01': Uv, 'sd': sd, 'move': mv}


def bt_book_stats(bk: dict, days: np.ndarray) -> dict:
    s = lambda x: x[days][np.isfinite(x[days])]
    fe = np.abs(bk['E'][days, _nT:])
    return {'net DV01, mean ($k/bp)': s(bk['dv01']).mean() / 1e3, 'net DV01, p95 ($k/bp)': np.percentile(s(bk['dv01']), 95) / 1e3,
            'ex-ante daily sd, mean ($k)': s(bk['sd']).mean() / 1e3, 'ex-ante daily sd, p95 ($k)': np.percentile(s(bk['sd']), 95) / 1e3,
            'realised daily rate+factor move sd ($k)': s(bk['move']).std() / 1e3, **{f'|factor {k + 1} exposure|, mean ($k per unit f)': fe[:, k].mean() / 1e3 for k in range(3)}}


def bt_win_full(nm: str) -> np.ndarray:
    per = BT_PER[nm]; w = np.zeros(len(BT), bool); w[per['rows'].to_numpy()] = per['win_v'].to_numpy(); return w


_pb_win = {'vanilla': bt_win_full('vanilla (production replay)'), 'objective': bt_win_full('objective: selection-adjusted value')}
BT_BOOKS = {k: bt_book(w) for k, w in _pb_win.items()}
_pb_window = (_pb_days >= _sig_first) & (_pb_days >= pd.Timestamp(min(m for m in BT_MONTHS if np.isfinite(BT_OBJ_B[_month == m]).any())))
bt_book_tab = pd.DataFrame({k: bt_book_stats(b, _pb_window) for k, b in BT_BOOKS.items()}).T
print(f'Books of vanilla and the objective variant, {_pb_days[_pb_window][0].date()}..{_pb_days[_pb_window][-1].date()} (positions held {BT_HOLD} business days):')
display(bt_book_tab.round(2))

fig, ax = plt.subplots(1, 3, figsize=(22, 5))
for k, col in [('vanilla', '#000000'), ('objective', '#937860')]:
    ax[0].plot(_pb_days, BT_BOOKS[k]['dv01'] / 1e3, color=col, lw=1.2, label=k); ax[1].plot(_pb_days, BT_BOOKS[k]['sd'] / 1e3, color=col, lw=1.2, label=k)
ax[0].set_title('Book net DV01 ($k per bp), start of day', fontsize=10); ax[1].set_title('Book ex-ante daily sd ($k): rate + IPCA factors + idiosyncratic', fontsize=10)
for k in range(3):
    ax[2].plot(_pb_days, BT_BOOKS['objective']['E'][:, _nT + k] / 1e3, lw=1.1, label=f'factor {k + 1}')
ax[2].axhline(0, color='k', lw=0.5); ax[2].set_title('Objective book: IPCA factor exposure ($k per unit of factor)', fontsize=10)
for a_ in ax:
    a_.legend(fontsize=8)
savefig('14b_book_risk')
_dt = pd.DataFrame({'tenor': np.array(BT_TENORS)[_pb_ti], 'u': _pb_u, 'w': _pb_win['vanilla']}).query('w').groupby('tenor')['u'].sum()
print('Where vanilla\'s fill DV01 sits (share of $ per bp by MMD tenor):', (_dt / _dt.sum()).round(3).to_dict())
record('lab_rfq_book', tenors=BT_TENORS, tenor_map=BT_TENOR, sigma_from=str(_sig_first.date()), beta_source=pd.Series(_src[_winable]).value_counts(normalize=True).round(4).to_dict(),
       books=flat_records(bt_book_tab))

# %%
CELL_T('14b. Risk-aware acceptance, replayed [10]')
PRE_REGISTERED.update({
    'lab_rfq_riskaware': ('fixed 2026-10-20 before scoring: extra charge rho_i = gamma s_i, s_i = h (2 e_i\' Sigma E_d + e_i\' Sigma e_i + v_i) / (2 u_i) bp with E_d the policy\'s own '
                          'start-of-day book; the optimizer runs on the base\'s curve and value (vanilla: c; objective: c + b_key) with charges - rho, each policy gets its own optimal '
                          'yield and its quote = opto + the logged loading; gamma_k = k / median(s) on the vanilla book in the first month with Sigma, k = 0..Q (Q = the width grid '
                          'range); walk-forward pick per test month = the k with the highest weekly $ P&L Sharpe on earlier months (exits printed before the month); a pick improves '
                          'on its base if, on the test months, its weekly $ P&L Sharpe is higher and its mean ex-ante book sd is lower; the $ P&L difference is reported with its '
                          'weekly Newey-West t'),
})
_pb_rows_day = pd.Series(np.arange(len(BT))).groupby(_pb_di).indices
_y_tr = BT['y_trade'].to_numpy(float)
_pb_bases = {'vanilla': (_ch, _opt_van), 'objective': (_ch + BT_OBJ_B, BT_VARIANTS['objective: selection-adjusted value']['opto'])}


def bt_win_fast(rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The outcome rule of bt_outcomes on a subset of rows."""
    d = 100.0 * (_thr[rows] - y)
    return np.where(_grp[rows] == 'lost, printed', d > BT_EPS_BP, d >= -BT_EPS_BP) & np.isfinite(d)


def bt_requote(rows: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Production optimizer on the rows' own curves with the given charges; returns the optimal yields."""
    sub = pd.DataFrame({'pfill_file': _files[rows], 'pfill_key': _keys[rows], '_c': charges})
    return bt_opto(_m_sig[rows], bt_replay(sub, BT_CURVES, charges_col='_c')['bt_opt_bp'].to_numpy())


def bt_marginal(rows: np.ndarray, j: int, E: np.ndarray) -> np.ndarray:
    """s_i in bp per unit gamma: half the h-day marginal variance of adding fill i to book E, per $ of DV01."""
    S = BT_SIGMA[j]; e = BT_EXPO[rows]
    return BT_HOLD * (2.0 * e @ (S @ E) + np.einsum('ij,jk,ik->i', e, S, e) + BT_V[rows]) / (2.0 * _pb_u[rows])


def bt_risk_replay(base: str, gamma: float) -> np.ndarray:
    """Sequential by business day: each request is re-optimised with its marginal-risk charge against the policy's own start-of-day book."""
    c0, o0 = _pb_bases[base]; opto = o0.copy(); F = np.zeros((_pb_nd, _D))
    for j in range(_pb_nd):
        rows = _pb_rows_day.get(j, np.zeros(0, int)); act = rows[np.isfinite(o0[rows]) & (_pb_u[rows] > 0)]
        if not len(act):
            continue
        if j in BT_SIGMA:
            E = F[max(0, j - (BT_HOLD - 1)):j].sum(0)
            opto[act] = bt_requote(act, c0[act] - gamma * bt_marginal(act, j, E))
        F[j] = BT_EXPO[act][bt_win_fast(act, opto[act] + _adj[act])].sum(0)
    return opto


# ---- checks: re-optimising the objective's own charges reproduces its quotes; the gamma scale from the vanilla book in the first month with Sigma
_jt = max(BT_SIGMA); _rt = _pb_rows_day[_jt]; _rt = _rt[np.isfinite(_pb_bases['objective'][1][_rt])]
_agree = float(np.mean(np.abs(bt_requote(_rt, _pb_bases['objective'][0][_rt]) - _pb_bases['objective'][1][_rt]) < 1e-9))
print(f'objective base re-optimised with charges c + b reproduces its own optimal yield on {_agree:.2%} of {len(_rt):,} requests ({_pb_days[_jt].date()})')
_m_sig1 = _sig_first.strftime('%Y-%m'); _s_all = []
for j in sorted(BT_SIGMA):
    if _pb_days[j].strftime('%Y-%m') == _m_sig1:
        r_ = _pb_rows_day.get(j, np.zeros(0, int)); r_ = r_[_pb_u[r_] > 0]
        _s_all.append(bt_marginal(r_, j, BT_BOOKS['vanilla']['E'][j]))
_s_med = float(np.median(np.concatenate(_s_all)))
BT_GAMMAS = {k: k / _s_med for k in range(1, BT_Q + 1)}
print(f'marginal-risk scale: median s = {_s_med:.4g} bp per unit gamma ({_m_sig1}, vanilla book); gamma_k = k / median s, k = 1..{BT_Q} (median extra charge = k bp)')


def _bt_risk_runs() -> pd.DataFrame:
    out = {}
    for base in _pb_bases:
        for k, g in BT_GAMMAS.items():
            t0 = time.perf_counter(); out[f'{base} | k={k}'] = bt_risk_replay(base, g)
            print(f'  {base} k={k}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out)


BT_RISK_OPTO = cached('lab_rfq_riskaware', _bt_risk_runs, deps=[STAGE_KEYS['lab_rfq_grids'], float(np.nansum(BT_OBJ_B)), float(_pb_u.sum()), float(_Rv.sum()), float(BT_EXPO.sum()),
                                                                 BT_HOLD, list(BT_GAMMAS.values())],
                      code=[bt_risk_replay, bt_marginal, bt_requote, bt_win_fast, bt_replay, bt_optimize, bt_opto])

# ---- score every policy: P&L ($ and bp), weekly $ Sharpe, and its own book
_pols = {}
for base, (_, o0) in _pb_bases.items():
    _pols[f'{base} | k=0'] = o0
    for k in BT_GAMMAS:
        _pols[f'{base} | k={k}'] = BT_RISK_OPTO[f'{base} | k={k}'].to_numpy(float)


def bt_policy_pnl(opto: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = opto + _adj; rows = np.arange(len(BT)); win = np.zeros(len(BT), bool); ok = np.isfinite(y)
    win[ok] = bt_win_fast(rows[ok], y[ok]) & _winable[ok]
    pv = np.where(_winable & ~np.isfinite(_M), np.nan, np.where(win, 100.0 * (y - _y_tr) + _M, 0.0)); pv[~ok] = np.nan
    return win, pv, pv * np.where(_pb_u > 0, _pb_u, np.nan)


_PP = {nm: bt_policy_pnl(o) for nm, o in _pols.items()}
_risk_m0 = _m_sig1; _test_m = [m for m in BT_MONTHS if m > _risk_m0]
_wk_all = BT['week'].to_numpy()


def bt_weekly_usd(usd: np.ndarray, mask: np.ndarray) -> pd.Series:
    return pd.Series(np.where(np.isfinite(usd), usd, 0.0)[mask]).groupby(_wk_all[mask]).sum()


def bt_sharpe(s: pd.Series) -> float:
    return float(s.mean() / s.std()) if len(s) > 2 and s.std() > 0 else np.nan


# walk-forward pick of k per test month: the highest weekly $ Sharpe on earlier months (from the first month with Sigma), exits printed before the month
_picks, _pick_rows = {}, []
for base in _pb_bases:
    o = np.full(len(BT), np.nan)
    for m in _test_m:
        tr = bt_train_rows(m) & (_month >= _risk_m0) & np.isfinite(_pb_bases[base][1])
        sc = {k: bt_sharpe(bt_weekly_usd(_PP[f'{base} | k={k}'][2], tr)) for k in [0] + list(BT_GAMMAS)}
        kb = max(sc, key=lambda k: -np.inf if not np.isfinite(sc[k]) else sc[k]); te = _month == m
        o[te] = _pols[f'{base} | k={kb}'][te]
        _pick_rows.append({'base': base, 'test month': m, 'k picked': kb, **{f'train Sharpe k={k}': v for k, v in sc.items()}})
    _picks[f'{base} | risk-aware pick'] = o
for nm, o in _picks.items():
    _pols[nm] = o; _PP[nm] = bt_policy_pnl(o)
print('Walk-forward pick of the risk aversion k (weekly $ P&L Sharpe on earlier months):'); display(pd.DataFrame(_pick_rows).set_index(['base', 'test month']).round(3))

# ---- the comparison on the test months: every policy against its base, on the same requests
_tm = np.isin(_month, _test_m); _tdays = np.isin(_pb_days.strftime('%Y-%m'), _test_m)
_rows_tab, BT_RISK_BOOKS = [], {}
for nm, (win, pv, usd) in _PP.items():
    base = nm.split(' |')[0]; msk = _tm & np.isfinite(_pb_bases[base][1])
    bk = bt_book(win); BT_RISK_BOOKS[nm] = bk
    wk = bt_weekly_usd(usd, msk); wk0 = bt_weekly_usd(_PP[f'{base} | k=0'][2], msk); bs = bt_book_stats(bk, _tdays)
    _rows_tab.append({'policy': nm, 'base': base, 'win rate': float(win[msk].mean()), 'P&L per request (bp)': float(np.nanmean(pv[msk])),
                      '$ P&L per month ($k)': float(wk.sum() / len(_test_m) / 1e3), 'weekly $ Sharpe': bt_sharpe(wk), 'worst week ($k)': float(wk.min() / 1e3),
                      'net DV01, mean ($k/bp)': bs['net DV01, mean ($k/bp)'], 'net DV01, p95 ($k/bp)': bs['net DV01, p95 ($k/bp)'],
                      'ex-ante daily sd, mean ($k)': bs['ex-ante daily sd, mean ($k)'], 'realised daily move sd ($k)': bs['realised daily rate+factor move sd ($k)'],
                      'delta $ per month vs base ($k)': float((wk - wk0).sum() / len(_test_m) / 1e3), 'NW t of weekly delta $': nw_t((wk - wk0).to_numpy()) if nm != f'{base} | k=0' else np.nan})
bt_risk_tab = pd.DataFrame(_rows_tab).set_index('policy')
for base in _pb_bases:
    b0 = bt_risk_tab.loc[f'{base} | k=0']
    for c_ in ['ex-ante daily sd, mean ($k)', 'net DV01, mean ($k/bp)']:
        bt_risk_tab.loc[bt_risk_tab['base'] == base, f'{c_.split(",")[0]} vs base'] = bt_risk_tab.loc[bt_risk_tab['base'] == base, c_] / b0[c_] - 1.0
bt_risk_tab['$ P&L per $ of ex-ante sd (per month)'] = bt_risk_tab['$ P&L per month ($k)'] / bt_risk_tab['ex-ante daily sd, mean ($k)']
_verd = {}
for base in _pb_bases:
    p, b0 = bt_risk_tab.loc[f'{base} | risk-aware pick'], bt_risk_tab.loc[f'{base} | k=0']
    _verd[base] = bool(p['weekly $ Sharpe'] > b0['weekly $ Sharpe'] and p['ex-ante daily sd, mean ($k)'] < b0['ex-ante daily sd, mean ($k)'])
BT_RISK_VERDICT = {b: ('improves on its base' if v else 'does not improve on its base') for b, v in _verd.items()}
print(f'P&L against book risk on the test months {_test_m[0]}..{_test_m[-1]} (k = median extra charge in bp; k = 0 is the base; book held {BT_HOLD} business days):')
display(bt_risk_tab.drop(columns=['base']).round(3))
print('VERDICT (lab_rfq_riskaware):', BT_RISK_VERDICT)

# ---- against production itself (vanilla, k = 0) on the same test-month requests where the objective is defined
_mo = _tm & np.isfinite(_pb_bases['objective'][1]); _wv0 = bt_weekly_usd(_PP['vanilla | k=0'][2], _mo); _v0 = bt_risk_tab.loc['vanilla | k=0']
bt_vs_prod = pd.DataFrame([{'policy': nm, 'delta $ per month vs production ($k)': float((bt_weekly_usd(_PP[nm][2], _mo) - _wv0).sum() / len(_test_m) / 1e3),
                            'NW t of weekly delta $': nw_t((bt_weekly_usd(_PP[nm][2], _mo) - _wv0).to_numpy()),
                            'ex-ante sd vs production': bt_risk_tab.loc[nm, 'ex-ante daily sd, mean ($k)'] / _v0['ex-ante daily sd, mean ($k)'] - 1.0,
                            'net DV01 vs production': bt_risk_tab.loc[nm, 'net DV01, mean ($k/bp)'] / _v0['net DV01, mean ($k/bp)'] - 1.0,
                            'weekly $ Sharpe': bt_risk_tab.loc[nm, 'weekly $ Sharpe']}
                           for nm in bt_risk_tab.index if nm.startswith('objective')]).set_index('policy')
bt_vs_prod['more P&L and less risk than production'] = (bt_vs_prod['delta $ per month vs production ($k)'] > 0) & (bt_vs_prod['ex-ante sd vs production'] < 0) & (bt_vs_prod['net DV01 vs production'] < 0)
print('The objective with and without the risk charge, against production (vanilla, k = 0):'); display(bt_vs_prod.round(3))

# ---- figures: the P&L / risk frontier, the books over time, cumulative dollars
fig, ax = plt.subplots(2, 2, figsize=(20, 11))
for base, col in [('vanilla', '#000000'), ('objective', '#937860')]:
    t_ = bt_risk_tab[bt_risk_tab.index.str.startswith(f'{base} | k=')]
    ax[0, 0].plot(t_['ex-ante daily sd, mean ($k)'], t_['$ P&L per month ($k)'], '-o', color=col, label=f'{base}, k = 0..{BT_Q}')
    for nm, r_ in t_.iterrows():
        ax[0, 0].annotate(nm.split('| ')[1], (r_['ex-ante daily sd, mean ($k)'], r_['$ P&L per month ($k)']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    p_ = bt_risk_tab.loc[f'{base} | risk-aware pick']
    ax[0, 0].scatter(p_['ex-ante daily sd, mean ($k)'], p_['$ P&L per month ($k)'], marker='*', s=160, color=col, zorder=3, label=f'{base}: walk-forward pick')
ax[0, 0].set_xlabel(r'book ex-ante daily sd, mean (\$k)'); ax[0, 0].set_ylabel(r'\$ P&L per month (\$k)'); ax[0, 0].legend(fontsize=8)
ax[0, 0].set_title('P&L against book risk: each k (median extra charge, bp) and the walk-forward pick', fontsize=10)
_show_p = [('vanilla | k=0', '#000000'), ('objective | k=0', '#937860'), ('objective | risk-aware pick', '#C44E52'), ('vanilla | risk-aware pick', '#4C72B0')]
for nm, col in _show_p:
    ax[0, 1].plot(_pb_days[_tdays], BT_RISK_BOOKS[nm]['dv01'][_tdays] / 1e3, color=col, lw=1.1, label=nm)
    ax[1, 0].plot(_pb_days[_tdays], BT_RISK_BOOKS[nm]['sd'][_tdays] / 1e3, color=col, lw=1.1, label=nm)
    base = nm.split(' |')[0]; wk = bt_weekly_usd(_PP[nm][2], _tm & np.isfinite(_pb_bases[base][1]))
    ax[1, 1].plot(wk.index, wk.cumsum().to_numpy() / 1e3, color=col, lw=1.4, label=nm)
ax[0, 1].set_title(r'Book net DV01 (\$k per bp)', fontsize=10); ax[1, 0].set_title(r'Book ex-ante daily sd (\$k)', fontsize=10)
ax[1, 1].set_title(r'Cumulative \$ P&L (\$k), test months', fontsize=10); ax[1, 1].axhline(0, color='k', lw=0.5)
for a_ in ax.flat[1:]:
    a_.legend(fontsize=8)
savefig('14b_riskaware')
record('lab_rfq_riskaware', gamma_scale=_s_med, gammas=BT_GAMMAS, picks=flat_records(pd.DataFrame(_pick_rows)), table=flat_records(bt_risk_tab), verdict=BT_RISK_VERDICT,
       objective_requote_agreement=_agree)

# %% [markdown]
# ### 14c. Can the factor model replace the manual charge? Same wins, better cover
#
# **Production's charge.** Production uses the total charge:
# $$c = -(\text{manual} + \text{liquidity} + \text{risk})$$
# Over the window the manual charge averages 8.1 bp (median 4.5, range −15 to 160, non-zero on 95% of requests). Liquidity averages 3.9 bp and risk 0.03 bp, so the manual charge carries most of production's charging.
#
# **Question.** Keep the liquidity and risk charges and remove the manual charge. Put a factor-model charge built from the risk attribution of 14b in its place. Does it allocate the same overall charge better?
#
# **Formations of the factor charge.** Each one is $\lambda \cdot g_i$ (bp) and goes through the optimizer to its own optimal yield, with quote = opto + the logged loading.
#
# | Formation | $g_i$ | What it charges for |
# |---|---|---|
# | no manual | 0 | nothing; shows what the manual charge does |
# | flat | 1 | the same charge on every request; the control |
# | rate + factor | $\sqrt{h\,(\Sigma_{\tau\tau} + 2\beta'\Sigma_{f\tau} + \beta'\Sigma_{ff}\beta)}$ | the bond's systematic yield vol over the holding |
# | idiosyncratic | $\sqrt{h}\,\sigma_i$ | the bond's own residual vol |
# | total | $\sqrt{h\,(\cdots + \sigma_i^2)}$ | systematic + idiosyncratic vol |
# | book | $s_i$ of 14b | the marginal risk the fill adds to the policy's own book |
#
# **Calibration: same wins.** P&L is not the criterion. For each test month, $\lambda$ is chosen by bisection so that the formation's win rate **in that month** equals production's win rate on the same requests. This is an equal-win comparison of how the charge is allocated, not a deployable rule.
#
# A formation therefore spends the same overall charge as the manual one, allocated by risk instead of by hand. The flat formation is the control: if risk-based allocation adds nothing, it does no better than flat.
#
# The walk-forward version, with $\lambda$ from the previous month, was tried first and fails. Traders moved the manual charge from 2.6 bp in August to 17.3 bp in September, so any rule fitted on earlier months wins about 9% of requests against production's 2.65%.
#
# **What is compared, in order:**
#
# 1. **Wins:** win rate, vanilla's wins kept and lost (a quote cheaper than the cover loses a win we had), and new wins.
# 2. **Cover:** on vanilla's cover wins that the formation keeps, bp left on the table against vanilla (lower is better).
# 3. **Markout per fill:** realised P&L with its attribution. Market is the MMD move at the tenor over the holding, factor is $\beta' \Delta f$, and residual is the rest (bid-offer captured plus bond-specific).
# 4. **Book risk:** net DV01 and ex-ante sd.
# 5. **Factor breakdown:** where the formation charges more or less than the traders did, and the effect on wins and cover by tenor, by factor-2 beta and by systematic vol.
# | [12] | The formations, win rate matched to production within each month: wins, cover, markout attribution, book risk, factor breakdown, figures |
#
# | Cell | Content || [12] | The formations, win-matched walk-forward: wins, cover, markout attribution, book risk, factor breakdown, figures |
#
# |---|---|| [11] | Charges audit, vanilla's realised markout attribution (market / factor / residual), each request's factor risk attribution, and how the manual charge lines up with it |

# %%
CELL_T('14c. Charges, realised markout attribution, factor risk attribution [11]')
PRE_REGISTERED.update({
    'lab_rfq_factor_charge': ('fixed 2026-10-20 before the formations were scored: production charge = -(manual + liquidity + risk); a formation keeps liquidity + risk and replaces '
                              'manual by lambda g_i, g in {0 (no manual), 1 (flat), systematic vol, idiosyncratic vol, total vol, book marginal risk s_i of 14b}; vols are the h-day '
                              'yield vol from the 14b risk model on the request day (Sigma from earlier days only); lambda per test month by bisection so that the formation\'s win '
                              'rate in that month equals production\'s on the same requests (an equal-win comparison of allocation; the previous-month version fails because the manual '
                              'charge regime moves, 2.6 bp in August to 17.3 bp in September); each formation goes through the optimizer to its own optimal yield, '
                              'quote = opto + the logged loading; compared on the test months by win rate, vanilla wins kept / lost, new wins, bp left on the table on kept cover '
                              'wins, then markout per fill with its attribution: market = -100 (MMD at the tenor, last close before the exit date - last close before the trade '
                              'date), factor = -beta\' (sum of IPCA factor moves over the same closes), residual = markout - market - factor'),
})
_liq, _rsk, _man = (BT[c_].to_numpy(float) for c_ in ('liquiditycharge', 'riskcharge', 'manualcharges'))

# ---- charges audit: production = -(manual + liquidity + risk)
assert np.allclose(_ch, -(_man + _liq + _rsk)), 'charges_level must be minus the sum of the three charges'
_ca = pd.DataFrame({'manual': _man, 'liquidity': _liq, 'risk': _rsk}).groupby(_month).agg(['mean', 'median'])
_ca[('manual', 'non-zero share')] = pd.Series(np.abs(_man) > 1e-9).groupby(_month).mean().to_numpy()
print('Production charges by month (bp; the optimizer uses c = -(manual + liquidity + risk)):'); display(_ca.round(3))

# ---- each request's factor risk attribution over the holding: rate (MMD at its tenor), IPCA factors (with the cross term), idiosyncratic
_vr, _vf, _vi = (np.full(len(BT), np.nan) for _ in range(3))
for j, S in BT_SIGMA.items():
    r_ = _pb_rows_day.get(j)
    if r_ is None:
        continue
    t_ = _pb_ti[r_]; b_ = BT_BETA[r_]
    _vr[r_] = S[t_, t_]
    _vf[r_] = 2.0 * np.einsum('ij,ij->i', b_, S[_nT:, t_].T) + np.einsum('ij,jk,ik->i', b_, S[_nT:, _nT:], b_)
    _vi[r_] = BT_IDV[r_]
BT_VOL = {'rate + factor': np.sqrt(BT_HOLD * np.maximum(_vr + _vf, 0.0)), 'idiosyncratic': np.sqrt(BT_HOLD * _vi), 'total': np.sqrt(BT_HOLD * np.maximum(_vr + _vf + _vi, 0.0))}
_wv = _winable & np.isfinite(BT_VOL['total'])
_att = pd.DataFrame({'tenor': BT['tenor_bucket'].astype(str).to_numpy(), 'rate var': _vr, 'factor var': _vf, 'idio var': _vi, 'total vol (bp, h days)': BT_VOL['total'],
                     'manual (bp)': _man, 'liquidity (bp)': _liq})[_wv]
_ag = _att.groupby('tenor').mean()
for c_ in ['rate var', 'factor var', 'idio var']:
    _ag[f'{c_.split()[0]} share'] = _ag[c_] / _ag[['rate var', 'factor var', 'idio var']].sum(axis=1)
print(f'Factor risk attribution of winnable requests (days with Sigma), {BT_HOLD}-day yield variance by source, and the charges traders applied:')
display(_ag[['rate share', 'factor share', 'idio share', 'total vol (bp, h days)', 'manual (bp)', 'liquidity (bp)']].round(3))
_cr = pd.DataFrame({'manual': _man, 'total vol': BT_VOL['total'], 'rate + factor vol': BT_VOL['rate + factor'], 'idio vol': BT_VOL['idiosyncratic'],
                    'beta1': BT_BETA[:, 0], 'beta2': BT_BETA[:, 1], 'beta3': BT_BETA[:, 2], 'liquidity': _liq})[_wv].corr(method='spearman')['manual'].drop('manual')
print('Does the manual charge follow the factor model\'s risk? Rank correlation of manual with each risk measure (winnable requests):', _cr.round(3).to_dict())

# ---- realised markout of production's fills, attributed: market (MMD over the holding), IPCA factors, residual
_tcol = np.array([_mmd.columns.get_loc(t) for t in BT_TENORS])[_pb_ti]
_mv = _mmd.to_numpy(float); _md = _mmd.index
_xdate = pd.to_datetime(BT['tx_5_X']).dt.normalize()


def bt_prev_close(dates: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    return index.searchsorted(pd.DatetimeIndex(dates).to_numpy(), side='left') - 1


_i0, _i1 = bt_prev_close(BT['date'], _md), bt_prev_close(_xdate.fillna(BT['date']), _md)
_ok_m = (_i0 >= 0) & _xdate.notna().to_numpy()
BT_MKT = np.where(_ok_m, -100.0 * (_mv[np.maximum(_i1, 0), _tcol] - _mv[np.maximum(_i0, 0), _tcol]), np.nan)
_fcum = np.vstack([np.zeros(3), np.cumsum(_fo.to_numpy(float), 0)])
_j0, _j1 = _fo.index.searchsorted(BT['date'].to_numpy(), side='left'), _fo.index.searchsorted(_xdate.fillna(BT['date']).to_numpy(), side='left')
_ok_f = _ok_m & (BT['date'] >= _fo.index.min()).to_numpy() & (_src != 'none')
BT_FAC = np.where(_ok_f, -np.einsum('ij,ij->i', BT_BETA, _fcum[_j1] - _fcum[_j0]), np.nan)
BT_RES = _M - BT_MKT - BT_FAC


def bt_markout_attr(win: np.ndarray, y: np.ndarray, mask: np.ndarray, by=None) -> pd.DataFrame:
    """Per fill (won, with an exit print and an attribution): P&L = price vs the trade + market + factor + residual, bp; and $ per fill."""
    f = win & mask & np.isfinite(_M) & np.isfinite(BT_FAC)
    d = pd.DataFrame({'fills': 1, 'P&L per fill (bp)': 100.0 * (y - _y_tr) + _M, 'price vs trade': 100.0 * (y - _y_tr), 'markout M': _M, 'market (MMD)': BT_MKT,
                      'IPCA factors': BT_FAC, 'residual': BT_RES, '$ per fill': (100.0 * (y - _y_tr) + _M) * _pb_u})[f]
    g = d.groupby(by[f] if by is not None else np.zeros(int(f.sum())))
    out = g.mean(); out['fills'] = g['fills'].sum(); return out


_wv0 = _pb_win['vanilla']
bt_vanilla_attr = bt_markout_attr(_wv0, _y_van, np.ones(len(BT), bool), by=_month)
print('Production (vanilla) realised markout per fill, attributed (bp per fill; market = MMD at the tenor over the holding, factor = beta\' f, residual = bid-offer + bond-specific):')
display(bt_vanilla_attr.round(3))
print('MMD month-end close (%) at 2y / 6y / 12y / 23y:', _mmd[[2.0, 6.0, 12.0, 23.0]].resample('ME').last().round(3).T.to_dict(orient='index'))

fig, ax = plt.subplots(1, 3, figsize=(22, 5))
_va = bt_vanilla_attr[['market (MMD)', 'IPCA factors', 'residual']]
_bot_p = np.zeros(len(_va)); _bot_n = np.zeros(len(_va))
for c_, col in zip(_va.columns, ['#C44E52', '#4C72B0', '#55A868']):
    v_ = _va[c_].to_numpy(float); v_ = np.nan_to_num(v_)
    ax[0].bar(_va.index, v_, bottom=np.where(v_ >= 0, _bot_p, _bot_n), color=col, label=c_); _bot_p += np.maximum(v_, 0); _bot_n += np.minimum(v_, 0)
ax[0].plot(_va.index, bt_vanilla_attr['markout M'], 'ko-', label='markout M'); ax[0].axhline(0, color='k', lw=0.5); ax[0].legend(fontsize=8)
ax[0].set_title('Production fills: realised markout per fill, attributed (bp)', fontsize=10)
_bins = pd.qcut(pd.Series(BT_VOL['total'][_wv]), 10, duplicates='drop')
_mb = pd.DataFrame({'vol': BT_VOL['total'][_wv], 'manual': _man[_wv], 'liquidity': _liq[_wv]}).groupby(_bins.to_numpy(), observed=True).mean()
ax[1].plot(_mb['vol'], _mb['manual'], 'o-', color='#937860', label='manual'); ax[1].plot(_mb['vol'], _mb['liquidity'], 's-', color='#4C72B0', label='liquidity')
ax[1].set_xlabel(f'factor model total yield vol over {BT_HOLD} days (bp), decile mean'); ax[1].set_ylabel('charge (bp)'); ax[1].legend(fontsize=8)
ax[1].set_title('Charges traders applied against the factor model\'s risk', fontsize=10)
_sh = _ag[['rate share', 'factor share', 'idio share']]; _bt_ = np.zeros(len(_sh))
for c_, col in zip(_sh.columns, ['#C44E52', '#4C72B0', '#55A868']):
    ax[2].bar(_sh.index, _sh[c_], bottom=_bt_, color=col, label=c_); _bt_ += _sh[c_].to_numpy()
ax[2].legend(fontsize=8); ax[2].set_title('Risk attribution by tenor bucket (share of holding-period variance)', fontsize=10); ax[2].tick_params(axis='x', labelsize=8)
savefig('14c_charges_attribution')
record('lab_rfq_factor_charge', charges_by_month=flat_records(_ca), risk_attribution=flat_records(_ag), manual_vs_risk=_cr.round(4).to_dict(),
       vanilla_markout_attribution=flat_records(bt_vanilla_attr))

# %%
CELL_T('14c. Factor charge in place of the manual charge, win-matched [12]')
_c_base = -(_liq + _rsk)
BT_FM_G = {'no manual': None, 'flat': np.ones(len(BT)), 'rate + factor': BT_VOL['rate + factor'], 'idiosyncratic': BT_VOL['idiosyncratic'], 'total': BT_VOL['total'], 'book': 'book'}
BT_FM_ITERS = 14
_days_of = {m: [j for j in range(_pb_nd) if _pb_days[j].strftime('%Y-%m') == m] for m in BT_MONTHS}


def bt_fm_run(form: str, lam: float, days: list[int], F: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Optimal yields of a formation on the days' winnable requests with charges -(liquidity + risk) - lam g; sequential over days when g depends on the book."""
    if form != 'book':
        rows = np.concatenate([_pb_rows_day[j] for j in days]); rows = rows[_winable[rows]]
        g = 0.0 if form == 'no manual' else BT_FM_G[form][rows]
        return rows, bt_requote(rows, _c_base[rows] - lam * g), None
    F = np.zeros((_pb_nd, _D)) if F is None else F; rs, os_ = [], []
    for j in days:
        act = _pb_rows_day[j]; act = act[_winable[act]]; uok = _pb_u[act] > 0; s = np.zeros(len(act))
        s[uok] = bt_marginal(act[uok], j, F[max(0, j - (BT_HOLD - 1)):j].sum(0))
        o = bt_requote(act, _c_base[act] - lam * s)
        F[j] = BT_EXPO[act][bt_win_fast(act, o + _adj[act]) & uok].sum(0); rs.append(act); os_.append(o)
    return np.concatenate(rs), np.concatenate(os_), F


def bt_fm_calibrate(form: str, days: list[int], F0: np.ndarray | None = None) -> tuple[float, float, float]:
    """lambda by bisection so that the formation's win rate on these days equals vanilla's on the same requests; returns (lambda, target, achieved)."""
    allr = np.concatenate([_pb_rows_day[j] for j in days]); target = float(_wv0[allr].mean())

    def wr(lam: float) -> float:
        r, o, _ = bt_fm_run(form, lam, days, None if F0 is None else F0.copy())
        return float(bt_win_fast(r, o + _adj[r]).sum() / len(allr))

    if form == 'no manual' or wr(0.0) <= target:
        return 0.0, target, wr(0.0)
    g_scale = _s_med if form == 'book' else float(np.nanmean(BT_FM_G[form][allr]))
    lo, hi = 0.0, 2.0 * max(float(np.mean(_man[allr])), 1.0) / g_scale
    while wr(hi) > target:
        lo, hi = hi, 2.0 * hi
    for _ in range(BT_FM_ITERS):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if wr(mid) > target else (lo, mid)
    return hi, target, wr(hi)


def _bt_fm_runs() -> pd.DataFrame:
    out, lam_rows = {}, []
    for form in BT_FM_G:
        t0 = time.perf_counter(); o = np.full(len(BT), np.nan); F = None
        for m in _test_m:
            lam, tgt, got = bt_fm_calibrate(form, _days_of[m], F)
            rows, om, F = bt_fm_run(form, lam, _days_of[m], None if F is None else F.copy())
            o[rows] = om
            lam_rows.append({'formation': form, 'test month': m, 'lambda': lam, 'production win rate': tgt, 'formation win rate': got})
        out[form] = o
        print(f'  {form}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out).assign(_lam_json=json.dumps(lam_rows, default=float))


_fm = cached('lab_rfq_factor_charge', _bt_fm_runs, deps=[STAGE_KEYS['lab_rfq_riskaware'], float(np.nansum(BT_VOL['total'])), float(_liq.sum() + _rsk.sum()), _test_m, BT_FM_ITERS],
             code=[bt_fm_run, bt_fm_calibrate, bt_marginal, bt_requote, bt_win_fast, bt_replay, bt_optimize, bt_opto])
bt_fm_lam = pd.DataFrame(json.loads(_fm['_lam_json'].iloc[0])).set_index(['formation', 'test month'])
BT_FM_OPTO = {k: _fm[k].to_numpy(float) for k in BT_FM_G}
print('Calibration: lambda per test month so the formation\'s win rate equals production\'s on the same requests (equal-win comparison):'); display(bt_fm_lam.round(6))

# ---- score on the test months: wins first, then cover, then markout attribution and book risk
_tmw = _tm & _winable
_cov_w = (_grp == 'won, cover') & _wv0 & _tm


def bt_fm_score(nm: str, opto: np.ndarray) -> tuple[dict, np.ndarray]:
    y = opto + _adj; win = np.zeros(len(BT), bool); ok = np.isfinite(y) & _tmw
    win[ok] = bt_win_fast(np.flatnonzero(ok), y[ok])
    keep = _cov_w & win; d_v = 100.0 * (_thr - y); d_0 = 100.0 * (_thr - _y_van)
    newl = win & ~_wv0 & (_grp == 'lost, printed') & _tm
    at = bt_markout_attr(win, y, _tm).iloc[0]
    pv = np.where(win & _tm, 100.0 * (y - _y_tr) + _M, 0.0); pv[_tm & _winable & ~np.isfinite(_M)] = np.nan
    bs = bt_book_stats(bt_book(win), _tdays)
    return {'formation': nm, 'win rate': float(win[_tm].sum() / _tm.sum()), 'win rate, vanilla': float(_wv0[_tm].sum() / _tm.sum()),
            'vanilla wins kept': float((win & _wv0 & _tm).sum() / (_wv0 & _tm).sum()), 'vanilla wins lost': int((_wv0 & ~win & _tm).sum()), 'new wins': int((win & ~_wv0 & _tm).sum()),
            'cover wins kept': float(keep.sum() / _cov_w.sum()), 'bp left on kept cover wins': float(d_v[keep].mean()), '... vanilla, same requests': float(d_0[keep].mean()),
            'bp saved on kept cover wins': float((d_0 - d_v)[keep].mean()), 'crossing on new wins (bp)': float(-d_v[newl].mean()) if newl.any() else np.nan,
            **{f'per fill: {k}': float(at[k]) for k in ['P&L per fill (bp)', 'price vs trade', 'markout M', 'market (MMD)', 'IPCA factors', 'residual']},
            'P&L per request (bp)': float(np.nanmean(pv[_tm])), '$ P&L per month ($k)': float(np.nansum(pv[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3),
            'net DV01, mean ($k/bp)': bs['net DV01, mean ($k/bp)'], 'ex-ante daily sd, mean ($k)': bs['ex-ante daily sd, mean ($k)']}, win


_VAN = 'vanilla (manual + liquidity + risk)'
_fm_rows, BT_FM_WIN = [], {}
for nm, o in {_VAN: _opt_van, **BT_FM_OPTO}.items():
    r_, w_ = bt_fm_score(nm, o); _fm_rows.append(r_); BT_FM_WIN[nm] = w_
bt_fm_tab = pd.DataFrame(_fm_rows).set_index('formation')
_fl = bt_fm_tab.loc['flat']
bt_fm_tab['beats the flat control on cover'] = (bt_fm_tab['bp saved on kept cover wins'] > _fl['bp saved on kept cover wins']) & (bt_fm_tab['vanilla wins kept'] >= _fl['vanilla wins kept'])
bt_fm_tab.loc[[_VAN, 'flat', 'no manual'], 'beats the flat control on cover'] = False
_wcols_fm = ['win rate', 'win rate, vanilla', 'vanilla wins kept', 'vanilla wins lost', 'new wins', 'cover wins kept', 'bp left on kept cover wins', '... vanilla, same requests',
             'bp saved on kept cover wins', 'crossing on new wins (bp)', 'beats the flat control on cover']
print(f'1. WINS AND COVER on the test months {_test_m[0]}..{_test_m[-1]} (liquidity + risk kept; manual replaced; win rate matched to production within each month; flat = the control):')
display(bt_fm_tab[_wcols_fm].round(4))
print('2. MARKOUT PER FILL, attributed (bp per fill: price vs the trade + market (MMD) + IPCA factors + residual), and the book (secondary):')
display(bt_fm_tab[[c_ for c_ in bt_fm_tab.columns if c_.startswith('per fill')] + ['P&L per request (bp)', '$ P&L per month ($k)', 'net DV01, mean ($k/bp)', 'ex-ante daily sd, mean ($k)']].round(3))

# ---- 3. factor breakdown: where each formation charges differently from the traders, and what it does to wins and cover
_fmc = {'flat': BT_FM_G['flat'], **{k: BT_FM_G[k] for k in ['rate + factor', 'idiosyncratic', 'total']}}
_lam_t = bt_fm_lam['lambda'].unstack('test month')
_charge = {}
for k, g in _fmc.items():
    c_ = np.full(len(BT), np.nan)
    for m in _test_m:
        te = (_month == m) & _winable; c_[te] = _lam_t.loc[k, m] * g[te]
    _charge[k] = c_
_b2 = pd.qcut(pd.Series(BT_BETA[:, 1]).where(_tmw), 3, labels=['beta2 low', 'beta2 mid', 'beta2 high']).astype(str).to_numpy()
_vt = pd.qcut(pd.Series(BT_VOL['rate + factor']).where(_tmw), 3, labels=['sys vol low', 'sys vol mid', 'sys vol high']).astype(str).to_numpy()
_dims = {'tenor': BT['tenor_bucket'].astype(str).to_numpy(), 'IPCA factor-2 beta': _b2, 'systematic vol': _vt}
_br = []
for dim, lab in _dims.items():
    for cell in sorted(set(lab[_tmw]) - {'nan'}):
        msk = _tmw & (lab == cell); cv = msk & _cov_w
        row = {'dimension': dim, 'cell': cell, 'requests': int(msk.sum()), 'manual (bp)': float(_man[msk].mean()), 'vanilla win rate': float(_wv0[msk].mean())}
        for k in _fmc:
            w = BT_FM_WIN[k]; kp = cv & w
            row[f'{k}: charge - manual (bp)'] = float(np.nanmean(_charge[k][msk]) - _man[msk].mean())
            row[f'{k}: win rate - vanilla'] = float(w[msk].mean() - _wv0[msk].mean())
            row[f'{k}: bp saved on kept cover wins'] = float((100.0 * (BT_FM_OPTO[k] - _opt_van))[kp].mean()) if kp.any() else np.nan
        _br.append(row)
bt_fm_breakdown = pd.DataFrame(_br).set_index(['dimension', 'cell'])
print('3. FACTOR BREAKDOWN: how each formation re-allocates the manual charge (charge - manual, bp), and the effect on wins and cover, by tenor, factor-2 beta and systematic vol:')
display(bt_fm_breakdown.round(3))

fig, ax = plt.subplots(2, 2, figsize=(20, 11))
_t = bt_fm_tab; _xs = np.arange(len(_t))
ax[0, 0].bar(_xs - 0.2, _t['vanilla wins lost'], 0.4, color='#C44E52', label='production wins lost'); ax[0, 0].bar(_xs + 0.2, _t['new wins'], 0.4, color='#55A868', label='new wins')
ax0b = ax[0, 0].twinx(); ax0b.plot(_xs, 100 * (_t['win rate'] - _t['win rate, vanilla']), 'ko-', label='win rate - production (pp)'); ax0b.axhline(0, color='k', lw=0.5)
ax[0, 0].set_xticks(_xs); ax[0, 0].set_xticklabels(_t.index, rotation=20, fontsize=8, ha='right'); ax[0, 0].legend(fontsize=8, loc='upper left'); ax0b.legend(fontsize=8, loc='upper right')
ax[0, 0].set_title('Wins: production wins lost, new wins, and the win-rate gap', fontsize=10)
ax[0, 1].bar(_xs - 0.2, _t['... vanilla, same requests'], 0.4, color='#000000', label='production'); ax[0, 1].bar(_xs + 0.2, _t['bp left on kept cover wins'], 0.4, color='#4C72B0', label='formation')
ax[0, 1].set_xticks(_xs); ax[0, 1].set_xticklabels(_t.index, rotation=20, fontsize=8, ha='right'); ax[0, 1].legend(fontsize=8)
ax[0, 1].set_title('Cover: bp left on the table on production\'s cover wins the formation keeps (lower = better)', fontsize=10)
_pf = _t[['per fill: price vs trade', 'per fill: market (MMD)', 'per fill: IPCA factors', 'per fill: residual']].fillna(0.0)
_bp_, _bn_ = np.zeros(len(_pf)), np.zeros(len(_pf))
for c_, col in zip(_pf.columns, ['#937860', '#C44E52', '#4C72B0', '#55A868']):
    v_ = _pf[c_].to_numpy(float); ax[1, 0].bar(_xs, v_, bottom=np.where(v_ >= 0, _bp_, _bn_), color=col, label=c_.replace('per fill: ', '')); _bp_ += np.maximum(v_, 0); _bn_ += np.minimum(v_, 0)
ax[1, 0].plot(_xs, _t['per fill: P&L per fill (bp)'], 'ko-', label='P&L per fill'); ax[1, 0].axhline(0, color='k', lw=0.5)
ax[1, 0].set_xticks(_xs); ax[1, 0].set_xticklabels(_t.index, rotation=20, fontsize=8, ha='right'); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title('Markout per fill, attributed (bp)', fontsize=10)
_dec = pd.qcut(pd.Series(BT_VOL['total']).where(_tmw), 10, duplicates='drop').to_numpy()
_cd = pd.DataFrame({'vol': BT_VOL['total'], 'manual': _man, **{k: _charge[k] for k in _fmc}})[_tmw].groupby(_dec[_tmw], observed=True).mean()
for k, col in [('manual', '#937860'), ('flat', '#999999'), ('rate + factor', '#4C72B0'), ('idiosyncratic', '#55A868'), ('total', '#C44E52')]:
    ax[1, 1].plot(_cd['vol'], _cd[k], 'o-', color=col, label=k)
ax[1, 1].set_xlabel(f'factor model total yield vol over {BT_HOLD} days (bp), decile mean'); ax[1, 1].set_ylabel('charge in place of manual (bp)'); ax[1, 1].legend(fontsize=8)
ax[1, 1].set_title('How each formation allocates the charge against risk (traders\' manual in brown)', fontsize=10)
savefig('14c_factor_charge')
record('lab_rfq_factor_charge', lambdas=flat_records(bt_fm_lam.reset_index()), table=flat_records(bt_fm_tab), breakdown=flat_records(bt_fm_breakdown))

# %% [markdown]
# ## 14d. The factor model as the book's risk model, calibrated: betas from characteristics, Sigma, and the scoring fixes
#
# **Why.** 14b and 14c read as a test of IPCA as the desk's risk model, but on the winnable requests only 20 percent carried the bond's own beta; 77 percent
# used a tenor-by-call bucket mean. In the variance decomposition the factors carried 1 to 24 percent, the rate block and the idiosyncratic block the rest. The
# expanding-window Sigma under-forecast the realised daily move of both books by about 1.5 times. And every positive result in 14 and 14c lived in new wins,
# valued at the bond's own next other-side print, which carries the bid-offer and the reversion of a noisy print. This section fixes those four things, in order,
# and re-scores what 14, 14b and 14c reported. Nothing above is changed: every cell here reads the objects 14 to 14c built and leaves their caches alone.
#
# **The core IPCA property the lab had not used.** $\beta_{i,t} = \Gamma' z_{i,t}$. Any bond with characteristics has a beta, whether or not it is in the estimation
# panel, and the factor path $f_t$ gives a fresh factor-implied move for a bond whose own mark is three weeks stale. Cell [13] builds $z$ for every requested bond
# from the evaluation panel, the request itself, the static track and the print tape, ranks it against the model panel on the last model date before the request,
# and validates the result against the beta of record on the requests that have both.
#
# | Cell | Content |
# |---|---|
# | [13] | $\beta = \Gamma' z$ and $\sigma$ for every requested bond; source shares; validation against the beta of record; the fresh factor-implied level against the street error |
# | [14] | Sigma: EWMA half-lives against the expanding window, scored by the realised over ex-ante ratio and the log score on the two books; the walk-forward pick; the calibration bar; the risk attribution redone on the new betas |
# | [15] | The objective's selection bias net of the day's common markout, as a leaderboard row; every variant and formation re-scored with new wins valued at the common markout and at the factor-implied move |
#
# **Closed lines.** On the v4 evidence the request-level skew of Section 13 and the alpha overlays of Section 14 are closed: no signal passes the selection,
# skew or inventory bar on the desk's requests, every overlay loses against production (t of minus 2.0 to minus 3.4), and a live skew A/B needs 476 weeks
# of flow. No further overlay variants are built.

# %%
CELL_T('14d. Betas and sigma from characteristics for every requested bond [13]')
PRE_REGISTERED.update({
    'lab_gz_betas': ('fixed 2026-10-21 before scoring: beta_i = Gamma\' z_i for every request, Gamma = the Procrustes-aligned Gamma of the last version strictly before the '
                     'request date; z = the Section 3 instruments rebuilt for the requested bond and rank-normalised against the model panel on the last model date strictly '
                     'before the request (a missing instrument sits at the panel median, z = 0, and is counted); raw characteristics from the bond\'s last evaluation-panel row '
                     'before the request, from the request itself (production fair value as the yield level, our bid price as the price level, DV01 as duration, the tenor, '
                     'call, coupon and rating buckets), from the static track and from the print tape over the prior 60 calendar days; source of record per request: own '
                     '(the bond\'s beta of record at its last model date), else Gamma-z; validation on requests that have both: the Gamma-z beta built from request-side '
                     'characteristics only against the beta of record, Pearson and Spearman correlation and RMSE in beta-sd units per factor; the step is used downstream '
                     'only if the Pearson correlation is >= 0.9 on factors 1 and 2, otherwise everything built on it is labelled uncalibrated; sigma_i: ridge of log '
                     'residual sd on z, fitted walk-forward by month on the panel, used in a month only where its out-of-sample R2 on that month\'s in-panel requests beats '
                     'the tenor x call bucket mean; fresh level: the factor-implied yield move since the bond\'s last observation, beta\' (F(d-1) - F(t_last)) plus the MMD move '
                     'at tenor, reported against the street error by month'),
})
GZ_CORR_BAR = 0.9
GZ_PRINT_WINDOW_DAYS = 60
GZ_SIGMA_TRAIN_DATES = 20
_gz_t0 = time.perf_counter()
_gz_mdates = pd.DatetimeIndex(sorted(MODEL_BY_DATE))
_gz_dates = BT['date'].to_numpy('datetime64[ns]'); _gz_cusip = BT['cusip'].astype(str).to_numpy()
_gz_mi = _gz_mdates.searchsorted(_gz_dates, side='left') - 1                 # last model date strictly before the request
_gz_gi = _gv_index.searchsorted(_gz_dates, side='left') - 1                  # last Gamma version strictly before the request
_gz_ok = (_gz_mi >= 0) & (_gz_gi >= 0)
_ten_arr = np.asarray(_mmd.columns, float); _mmd_v = _mmd.to_numpy(float); _mmd_idx = _mmd.index


def gz_first_col(frame, names):
    if frame is None:
        return None
    for n in names:
        if n in frame.columns:
            return n
    return None


def gz_last_row_before(dates: np.ndarray, cusips: np.ndarray, source: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per request, the source's last row for the cusip strictly before the request date (NaN where none)."""
    cols = [c for c in cols if c in source.columns]
    s = source[['cusip', 'date'] + cols].copy(); s['cusip'] = s['cusip'].astype(str); s['date'] = to_ns(s['date']).dt.normalize()
    s = s.dropna(subset=['date']).sort_values('date', kind='stable')
    q = pd.DataFrame({'cusip': cusips.astype(str), 'date': to_ns(pd.Series(dates)).to_numpy(), '_i': np.arange(len(cusips))}).sort_values('date', kind='stable')
    j = pd.merge_asof(q, s, on='date', by='cusip', direction='backward', allow_exact_matches=False)
    return j.sort_values('_i').drop(columns=['_i', 'cusip', 'date']).reset_index(drop=True)


def gz_mmd_at(dates: np.ndarray, years: np.ndarray) -> np.ndarray:
    """MMD yield (percent) at the given tenor in years from the last curve close strictly before each date; NaN where none."""
    i = _mmd_idx.searchsorted(dates, side='left') - 1; ok = (i >= 0) & np.isfinite(years)
    y = np.clip(years, _ten_arr[0], _ten_arr[-1]); k = np.clip(np.searchsorted(_ten_arr, y, side='right') - 1, 0, len(_ten_arr) - 2)
    w = (y - _ten_arr[k]) / (_ten_arr[k + 1] - _ten_arr[k]); row = _mmd_v[np.maximum(i, 0)]
    v = (1 - w) * row[np.arange(len(y)), k] + w * row[np.arange(len(y)), k + 1]
    return np.where(ok, v, np.nan)


def gz_bucket_years(bucket: pd.Series) -> np.ndarray:
    """Years from a tenor or call bucket label ('0-3m', '6-9m', '1y', '5-7y', '10-15y', '21y+', 'No Call'); NaN where it cannot be read."""
    s = bucket.reset_index(drop=True).astype(str).str.strip().str.lower(); out = np.full(len(s), np.nan)
    num = s.str.extractall(r'(\d+(?:\.\d+)?)')[0].astype(float).groupby(level=0).mean()
    out[num.index.to_numpy()] = num.to_numpy()
    months = s.str.contains('m') & ~s.str.contains('y'); out[months.to_numpy()] = out[months.to_numpy()] / 12.0
    out[s.str.contains('no call').to_numpy()] = np.nan
    return out


def gz_window_prints(dates: np.ndarray, cusips: np.ndarray, tape: pd.DataFrame, window_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Print count and mean log size of the bond's prints in the window strictly before the request date."""
    t = tape.sort_values('ts', kind='stable'); c = t['cusip'].astype(str).to_numpy(); ts = t['ts'].to_numpy('datetime64[ns]')
    lq = np.log(np.maximum(pd.to_numeric(t['quantity'], errors='coerce').to_numpy(float), 1.0))
    f = pd.DataFrame({'cusip': c, 'ts': ts, 'n': 1.0, 'lq': np.nan_to_num(lq)}).sort_values(['cusip', 'ts'], kind='stable')
    f['cn'] = f.groupby('cusip')['n'].cumsum(); f['cq'] = f.groupby('cusip')['lq'].cumsum(); f = f.sort_values('ts', kind='stable'); f['ts'] = f['ts'].astype('datetime64[ns]')
    out_n, out_q = np.zeros(len(dates)), np.full(len(dates), np.nan); dates = np.asarray(dates, dtype='datetime64[ns]')
    for lab, d in [('hi', dates), ('lo', dates - np.timedelta64(window_days, 'D'))]:
        q = pd.DataFrame({'cusip': cusips.astype(str), 'ts': d.astype('datetime64[ns]'), '_i': np.arange(len(dates))}).sort_values('ts', kind='stable')
        j = pd.merge_asof(q, f[['cusip', 'ts', 'cn', 'cq']], on='ts', by='cusip', direction='backward', allow_exact_matches=False).sort_values('_i')
        if lab == 'hi':
            n_hi, q_hi = j['cn'].fillna(0).to_numpy(), j['cq'].fillna(0).to_numpy()
        else:
            n_lo, q_lo = j['cn'].fillna(0).to_numpy(), j['cq'].fillna(0).to_numpy()
    n = n_hi - n_lo; out_n = n; ok = n > 0; out_q[ok] = (q_hi - q_lo)[ok] / n[ok]
    return out_n, out_q


# ---- raw characteristics of the requested bond: evaluation panel (last row before the request), the static track, the request itself, the print tape
_gz_rp_cols = ['maturity', 'cpn', 'callable', 'nxt_call_dt', 'state_code', 'final_comp_rating', 'comp_rating', 'issuer_industry', 'closing_price', 'modified_duration_lag1',
               'trade_size_lag', 'print_freq_lag']
_gzp = gz_last_row_before(_gz_dates, _gz_cusip, raw_panel, _gz_rp_cols)
_gz_in_rp = _gzp['maturity'].notna().to_numpy() if 'maturity' in _gzp else np.zeros(len(BT), bool)
_gz_ts = globals().get('track_static')
_gz_static = None
if _gz_ts is not None:
    _cc = gz_first_col(_gz_ts, ['cusip', 'CUSIP'])
    if _cc is not None:
        _gz_static = _gz_ts.drop_duplicates(_cc, keep='last').set_index(_gz_ts[_cc].astype(str).drop_duplicates(keep='last').to_numpy())
        _gz_static = _gz_static.reindex(_gz_cusip)
_gz_src_chars = np.where(_gz_in_rp, 'panel row', 'request fields')


def gz_static(names):
    c = gz_first_col(_gz_static, names)
    return _gz_static[c].to_numpy() if c is not None else np.full(len(BT), np.nan, dtype=object)


_gz_mat = pd.to_datetime(pd.to_numeric(_gzp.get('maturity'), errors='coerce'), unit='ms', errors='coerce') if 'maturity' in _gzp else pd.Series(pd.NaT, index=BT.index)
if _gz_mat.isna().mean() > 0.5 and 'maturity' in _gzp:
    _gz_mat = pd.to_datetime(_gzp['maturity'], errors='coerce')
_gz_mat = _gz_mat.fillna(pd.to_datetime(pd.Series(gz_static(['maturity', 'maturity_date', 'MATURITY'])), errors='coerce'))
_gz_ten_y = gz_bucket_years(BT['tenor_bucket']) if 'tenor_bucket' in BT else np.full(len(BT), np.nan)
_gz_mat = _gz_mat.fillna(pd.Series(pd.to_datetime(_gz_dates) + pd.to_timedelta(np.nan_to_num(_gz_ten_y, nan=7.0) * 365.25, unit='D'), index=BT.index).where(np.isfinite(_gz_ten_y)))
_gz_cpn = pd.to_numeric(_gzp.get('cpn'), errors='coerce') if 'cpn' in _gzp else pd.Series(np.nan, index=BT.index)
_gz_cpn = _gz_cpn.fillna(pd.to_numeric(pd.Series(gz_static(['cpn', 'coupon', 'COUPON'])), errors='coerce'))
if 'coupon_bucket' in BT:
    _cb = pd.to_numeric(BT['coupon_bucket'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')
    _gz_cpn = _gz_cpn.fillna(_cb)
_gz_call = pd.to_datetime(_gzp.get('nxt_call_dt'), errors='coerce') if 'nxt_call_dt' in _gzp else pd.Series(pd.NaT, index=BT.index)
_gz_call = _gz_call.fillna(pd.to_datetime(pd.Series(gz_static(['nxt_call_dt', 'next_call_date', 'NXT_CALL_DT'])), errors='coerce'))
_gz_call_y = gz_bucket_years(BT['call_bucket']) if 'call_bucket' in BT else np.full(len(BT), np.nan)
_gz_call = _gz_call.fillna(pd.Series(pd.to_datetime(_gz_dates) + pd.to_timedelta(np.nan_to_num(_gz_call_y) * 365.25, unit='D'), index=BT.index).where(np.isfinite(_gz_call_y)))
_gz_is_call = (_gzp['callable'].astype('string').str.upper().isin(['Y', 'YES', 'TRUE', '1']) if 'callable' in _gzp else pd.Series(False, index=BT.index)).fillna(False).to_numpy(bool)
_gz_is_call = _gz_is_call | (np.isfinite(_gz_call_y) & (_gz_call_y > 0)) | _gz_call.notna().to_numpy()
_gz_y = pd.to_numeric(BT['signalyield'], errors='coerce').to_numpy(float)                         # production fair value: the yield level, fresh every day
_gz_px = pd.to_numeric(BT['our_bid_price'], errors='coerce').to_numpy(float) if 'our_bid_price' in BT else np.full(len(BT), np.nan)
_gz_px = np.where(np.isfinite(_gz_px) & (_gz_px > 20) & (_gz_px < 300), _gz_px, np.nan)
_gz_rating = (BT['final_comp_rating'].astype('string').str.strip().str.upper() if 'final_comp_rating' in BT else pd.Series(pd.NA, index=BT.index, dtype='string'))
if 'final_comp_rating' in _gzp:
    _gz_rating = _gz_rating.fillna(_gzp['final_comp_rating'].astype('string').str.strip().str.upper())
if 'comp_rating' in _gzp:
    _gz_rating = _gz_rating.fillna(_gzp['comp_rating'].astype('string').str.strip().str.upper())
_gz_rscore = _gz_rating.map(RATING_SCALE).astype(float).to_numpy() if 'RATING_SCALE' in globals() else np.full(len(BT), np.nan)
_gz_state = (_gzp['state_code'].astype('string') if 'state_code' in _gzp else pd.Series(pd.NA, index=BT.index, dtype='string'))
_gz_state = _gz_state.fillna(pd.Series(gz_static(['state_code', 'state', 'STATE']), index=BT.index).astype('string')).str.strip().str.upper().fillna('OTHER').to_numpy()
_gz_ind = (_gzp['issuer_industry'].astype('string') if 'issuer_industry' in _gzp else pd.Series(pd.NA, index=BT.index, dtype='string'))
_gz_ind = _gz_ind.fillna(pd.Series(gz_static(['issuer_industry', 'INDUSTRY']), index=BT.index).astype('string')).str.strip().str.upper().fillna('UNKNOWN')
_gz_ind_key = _gz_ind.map(lambda k: 'ind_' + re.sub(r'[^A-Za-z0-9]+', '_', str(k)).strip('_')[:20]).to_numpy()
# term characteristics, the Section 3 recipe on the request-side inputs
_gz_d = pd.to_datetime(_gz_dates)
_gz_ptc = _gz_is_call & _gz_call.notna().to_numpy() & (_gz_call > _gz_d).to_numpy() & (np.nan_to_num(_gz_cpn.to_numpy(float), nan=-1.0) > _gz_y)
_gz_worst = _gz_mat.where(~_gz_ptc, _gz_call)
_gz_ytw = ((_gz_worst - _gz_d).dt.days / 365.25).to_numpy(float); _gz_ytm = ((_gz_mat - _gz_d).dt.days / 365.25).to_numpy(float)
_gz_ext = np.clip(np.nan_to_num(_gz_ytm - _gz_ytw), 0, None); _gz_ext[~np.isfinite(_gz_ytw)] = np.nan
# duration: DV01 per 100 par at the quote where priced, else the panel's last modified duration, else a par-bond approximation at the request's yield
_gz_dv = pd.to_numeric(BT['dv01'], errors='coerce').to_numpy(float) if 'dv01' in BT else np.full(len(BT), np.nan)
_gz_dur = _gz_dv * 1e4 / np.where(np.isfinite(_gz_px), _gz_px, 100.0)
_gz_dur = np.where(np.isfinite(_gz_dur) & (_gz_dur > 0) & (_gz_dur < 40), _gz_dur, pd.to_numeric(_gzp.get('modified_duration_lag1'), errors='coerce').to_numpy(float) if 'modified_duration_lag1' in _gzp else np.nan)
_gz_yr = np.maximum(_gz_y / 100.0, 1e-4); _gz_par = (1 - (1 + _gz_yr / 2) ** (-2 * np.nan_to_num(_gz_ytw, nan=7.0))) / _gz_yr
_gz_dur = np.where(np.isfinite(_gz_dur), _gz_dur, _gz_par)
_gz_dur_src = np.where(np.isfinite(_gz_dv * 1e4), 'dv01', np.where(_gz_in_rp, 'panel', 'par approx'))
_gz_bench = gz_mmd_at(_gz_dates, _gz_ytw)
_gz_tape_q = msrb_raw[['cusip', 'tradetime', 'quantity']].rename(columns={'tradetime': 'ts'}) if ('msrb_raw' in globals() and msrb_raw is not None and 'quantity' in msrb_raw.columns) else None
if _gz_tape_q is not None:
    _gz_tape_q = _gz_tape_q.assign(ts=to_ns(_gz_tape_q['ts'])).dropna(subset=['ts'])
    _gz_pn, _gz_pq = gz_window_prints(_gz_dates, _gz_cusip, _gz_tape_q, GZ_PRINT_WINDOW_DAYS)
else:
    _gz_pn, _gz_pq = np.zeros(len(BT)), np.full(len(BT), np.nan)
GZ_RAW = pd.DataFrame({'duration_ratio': np.clip(_gz_dur / np.clip(np.nan_to_num(_gz_ytw, nan=0.25), 0.25, None), 0, 1.5), 'modified_duration_lag1': _gz_dur,
                       'premium_discount_lag1': _gz_px - 100.0, 'closing_yield_lag1': _gz_y, 'years_to_worst': _gz_ytw, 'extension': _gz_ext, 'rating_score': _gz_rscore,
                       'benchmark_yield_lag1': _gz_bench, 'benchmark_spread_lag1': _gz_y - _gz_bench, 'trade_size_lag': _gz_pq, 'print_freq_lag': np.where(_gz_pn > 0, _gz_pn, np.nan),
                       'liquidity_20': np.nan})
GZ_PANEL_RAW = {'duration_ratio': 'duration_ratio', 'modified_duration_lag1': 'modified_duration_lag1', 'premium_discount_lag1': 'closing_price', 'closing_yield_lag1': 'closing_yield_lag1',
                'years_to_worst': 'years_to_worst', 'extension': 'extension', 'rating_score': 'rating_score', 'benchmark_yield_lag1': 'benchmark_yield_lag1',
                'benchmark_spread_lag1': 'benchmark_spread_lag1', 'trade_size_lag': 'trade_size_lag', 'print_freq_lag': 'print_freq_lag'}
GZ_PANEL_SHIFT = {'premium_discount_lag1': -100.0}


def gz_rank_against_panel(raw: pd.DataFrame, mi: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """z = (panel mid-rank of the request's raw value on its model date) - 0.5 for every z instrument in CHARS; 0 and a count where the value or the panel column is missing."""
    Z = pd.DataFrame(0.0, index=raw.index, columns=[c for c in CHARS if c.startswith('z_')]); missing = {}
    groups = pd.Series(np.arange(len(mi))).groupby(mi).indices
    for zc in Z.columns:
        name = zc[2:]; col = GZ_PANEL_RAW.get(name); x = raw[name].to_numpy(float) if name in raw else np.full(len(raw), np.nan)
        z = np.zeros(len(raw)); n_missing = 0
        for d_i, rows in groups.items():
            if d_i < 0:
                n_missing += len(rows); continue
            g = MODEL_BY_DATE[_gz_mdates[d_i]]
            if col is None or col not in g.columns:
                n_missing += len(rows); continue
            v = np.sort(pd.to_numeric(g[col], errors='coerce').to_numpy(float) + GZ_PANEL_SHIFT.get(name, 0.0)); v = v[np.isfinite(v)]
            if len(v) < 10:
                n_missing += len(rows); continue
            xr = x[rows]; ok = np.isfinite(xr)
            r = 0.5 * (np.searchsorted(v, xr[ok], 'left') + np.searchsorted(v, xr[ok], 'right')) / len(v)
            z[rows[ok]] = r - 0.5; n_missing += int((~ok).sum())
        Z[zc] = z; missing[zc] = n_missing / len(raw)
    return Z, missing


GZ_Z, _gz_missing = gz_rank_against_panel(GZ_RAW, _gz_mi)
for c in CHARS:
    if c == 'market_fv':
        GZ_Z[c] = 1.0
    elif c == 'nr_flag':
        GZ_Z[c] = (~np.isfinite(_gz_rscore)).astype(float)
    elif c == 'no_print_flag':
        GZ_Z[c] = (_gz_pn <= 0).astype(float)
    elif c.startswith('state_'):
        GZ_Z[c] = (_gz_state == c[6:]).astype(float)
    elif c.startswith('ind_'):
        GZ_Z[c] = (_gz_ind_key == c).astype(float)
    elif c not in GZ_Z:
        GZ_Z[c] = 0.0; _gz_missing[c] = 1.0
GZ_Z = GZ_Z[CHARS]
print(f'request-side instruments built for {len(BT):,} requests ({time.perf_counter() - _gz_t0:.0f}s): raw characteristics from {pd.Series(_gz_src_chars).value_counts(normalize=True).round(3).to_dict()}; '
      f'duration from {pd.Series(_gz_dur_src).value_counts(normalize=True).round(3).to_dict()}')
print('share of requests with the instrument at the panel median (missing):', {k: round(v, 3) for k, v in _gz_missing.items() if v > 0.0})


def gz_apply_gamma(Z: np.ndarray, gi: np.ndarray) -> np.ndarray:
    """beta = Z Gamma with the Gamma version of each row; NaN where there is no version yet."""
    B = np.full((len(Z), CFG.selected_k), np.nan)
    for g, rows in pd.Series(np.arange(len(gi))).groupby(gi).indices.items():
        if g >= 0:
            B[rows] = Z[rows] @ GAMMA_ALIGNED[_gv_index[g]]
    return B


BT_BETA_RQ = gz_apply_gamma(GZ_Z.to_numpy(float), _gz_gi)                     # from request-side characteristics only
# the same with the bond's own last panel instruments (a stale z, the current Gamma): what Gamma drift alone does
_gzz = gz_last_row_before(_gz_dates, _gz_cusip, model[['cusip', 'date'] + CHARS], CHARS)
_gz_has_z = _gzz[CHARS[0]].notna().to_numpy()
BT_BETA_PZ = np.full_like(BT_BETA_RQ, np.nan); BT_BETA_PZ[_gz_has_z] = gz_apply_gamma(_gzz.loc[_gz_has_z, CHARS].to_numpy(float), _gz_gi[_gz_has_z])

# ---- validation on requests with the beta of record (14b: own, point in time)
_gz_own = _own[:, :3]; _gz_v = _has & np.isfinite(BT_BETA_RQ).all(axis=1)
_val_rows = []
for k in range(3):
    sd = float(np.nanstd(_gz_own[_gz_v, k]))
    for lab, B in [('request fields', BT_BETA_RQ), ('panel instruments', BT_BETA_PZ)]:
        ok = _gz_v & np.isfinite(B[:, k])
        _val_rows.append({'factor': k + 1, 'z source': lab, 'requests': int(ok.sum()), 'Pearson': float(np.corrcoef(B[ok, k], _gz_own[ok, k])[0, 1]) if ok.sum() > 10 else np.nan,
                          'Spearman': float(stats.spearmanr(B[ok, k], _gz_own[ok, k])[0]) if ok.sum() > 10 else np.nan,
                          'RMSE / sd of beta': float(np.sqrt(np.mean((B[ok, k] - _gz_own[ok, k]) ** 2)) / sd) if ok.sum() > 10 and sd > 0 else np.nan})
bt_gz_valid = pd.DataFrame(_val_rows).set_index(['z source', 'factor'])
_gz_pass = bool(np.all([bt_gz_valid.loc[('request fields', k), 'Pearson'] >= GZ_CORR_BAR for k in (1, 2)]))
GZ_LABEL = 'calibrated' if _gz_pass else 'uncalibrated'
print(f'Validation of beta = Gamma\' z against the beta of record on the {int(_gz_v.sum()):,} requests that have both (bar: Pearson >= {GZ_CORR_BAR} on factors 1 and 2 for the request-fields z):')
display(bt_gz_valid.round(3))
print(f'VERDICT (lab_gz_betas): {"passes" if _gz_pass else "fails"} the correlation bar; downstream results built on Gamma-z betas are labelled {GZ_LABEL}')

# ---- beta of record for this section: own where the bond has one, else Gamma-z from request-side characteristics
BT_BETA_GZ = np.where(_has[:, None], _gz_own, BT_BETA_RQ)
BT_SRC_GZ = np.where(_has, 'own', np.where(np.isfinite(BT_BETA_RQ).all(axis=1), np.where(_gz_in_rp, 'Gamma-z (panel row)', 'Gamma-z (request fields)'), 'none')).astype(object)
_gz_none = ~np.isfinite(BT_BETA_GZ).all(axis=1)
BT_BETA_GZ[_gz_none] = BT_BETA[_gz_none]; BT_SRC_GZ[_gz_none & (_src != 'none')] = 'bucket'
BT_BETA_GZ = np.nan_to_num(BT_BETA_GZ)
_gz_src_share = pd.Series(BT_SRC_GZ[_winable]).value_counts(normalize=True).round(4)
print('source of beta on winnable requests (14b was own 20%, bucket 77%):', _gz_src_share.to_dict())

# ---- sigma from characteristics: ridge of log residual sd on z, walk-forward by month; used where it beats the bucket mean out of sample
_gz_sd = resid.loc[np.isfinite(resid['resid_sd']) & (resid['resid_sd'] > 0), ['cusip', 'date', 'resid_sd']].merge(model[['cusip', 'date'] + CHARS], on=['cusip', 'date'], how='inner')
_gz_sd_dates = pd.DatetimeIndex(sorted(_gz_sd['date'].unique()))
_gz_Zuse = np.where(_gz_has_z[:, None], _gzz[CHARS].to_numpy(float), GZ_Z.to_numpy(float)); _gz_Zuse = np.nan_to_num(_gz_Zuse)
_gz_own_var = _own[:, 3]; _gz_bucket_var = np.where(_has, np.nan, BT_IDV)       # the 14b bucket mean, where it was the source
_gz_pred_var = np.full(len(BT), np.nan); _sig_rows = []; _gz_use_model = {}
for m in BT_MONTHS:
    tr_d = _gz_sd_dates[_gz_sd_dates < _mstart[m]]
    if len(tr_d) < 5:
        continue
    tr = _gz_sd[_gz_sd['date'].isin(tr_d[-GZ_SIGMA_TRAIN_DATES:])]
    X = tr[CHARS].to_numpy(float); y = np.log(tr['resid_sd'].to_numpy(float)); lam = 1e-3 * len(X)
    coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    te = _month == m; _gz_pred_var[te] = np.exp(_gz_Zuse[te] @ coef) ** 2
    v = te & _has & np.isfinite(_gz_own_var) & (_gz_own_var > 0)
    # the bucket control on the same requests: the tenor x call mean of earlier days (bt_bucket_before), the 14b fallback
    bm = bt_bucket_before(_own[:, 3:4], _has, np.char.add(np.char.add(_tb_, '|'), _cb_), _pb_di)[:, 0]
    yv = np.log(_gz_own_var[v]) / 2
    r2 = lambda p: float(1 - np.nansum((yv - p) ** 2) / np.sum((yv - yv.mean()) ** 2)) if v.sum() > 20 else np.nan  # noqa: E731
    r2_m, r2_b = r2(np.log(_gz_pred_var[v]) / 2), r2(np.log(np.where(bm[v] > 0, bm[v], np.nan)) / 2)
    _gz_use_model[m] = bool(np.isfinite(r2_m) and (not np.isfinite(r2_b) or r2_m > r2_b))
    _sig_rows.append({'test month': m, 'train rows': len(tr), 'train dates': len(tr_d[-GZ_SIGMA_TRAIN_DATES:]), 'validation requests': int(v.sum()), 'R2 characteristic model': r2_m,
                      'R2 bucket mean': r2_b, 'used': 'characteristic model' if _gz_use_model[m] else 'bucket mean'})
bt_gz_sigma = pd.DataFrame(_sig_rows).set_index('test month')
print('sigma from characteristics: out-of-sample R2 of log residual sd on the month\'s in-panel requests, against the tenor x call bucket mean:'); display(bt_gz_sigma.round(3))
_gz_model_m = np.isin(_month, [m for m, u in _gz_use_model.items() if u])
BT_IDV_GZ = np.where(_has & np.isfinite(_gz_own_var), _gz_own_var, np.where(_gz_model_m & np.isfinite(_gz_pred_var), _gz_pred_var, BT_IDV))
BT_IDV_GZ = np.where(np.isfinite(BT_IDV_GZ) & (BT_IDV_GZ > 0), BT_IDV_GZ, float(np.nanmedian(_gz_own_var)))
BT_EXPO_GZ = np.zeros((len(BT), _D)); BT_EXPO_GZ[np.arange(len(BT)), _pb_ti] = _pb_u; BT_EXPO_GZ[:, _nT:] = _pb_u[:, None] * BT_BETA_GZ
BT_V_GZ = _pb_u ** 2 * BT_IDV_GZ

# ---- the fresh factor-implied level: what the factors and the curve did since the bond's last observation
_gz_age = bt_last_obs(rv_wide(resid, 'beta1'), _d02, _bi2, _on2)[1]
_gz_tl = (_d02 - np.nan_to_num(_gz_age)).astype(int); _gz_fresh_ok = _has & np.isfinite(_gz_age) & (_d02 >= 0)
_fresh_f = np.einsum('ij,ij->i', _gz_own, _fcum[np.clip(_d02 + 1, 0, len(_fcum) - 1)] - _fcum[np.clip(_gz_tl + 1, 0, len(_fcum) - 1)])
_i_prev = _mmd_idx.searchsorted(RV_DATES[np.clip(_d02, 0, len(RV_DATES) - 1)].to_numpy(), side='right') - 1
_i_last = _mmd_idx.searchsorted(RV_DATES[np.clip(_gz_tl, 0, len(RV_DATES) - 1)].to_numpy(), side='right') - 1
_fresh_m = 100.0 * (_mmd_v[np.maximum(_i_prev, 0), _tcol] - _mmd_v[np.maximum(_i_last, 0), _tcol])
BT['bt_level_fresh_bp'] = np.where(_gz_fresh_ok & (_i_prev >= 0) & (_i_last >= 0), _fresh_f + _fresh_m, np.nan)
_fr = BT.loc[np.isfinite(BT['bt_level_fresh_bp']) & np.isfinite(BT['e_street']), ['month', 'bt_level_fresh_bp', 'e_street']]
_fr_rows = []
for m_, g_ in _fr.groupby('month'):
    if len(g_) > 50 and g_['bt_level_fresh_bp'].std() > 0:
        sl = np.polyfit(g_['bt_level_fresh_bp'], g_['e_street'], 1)[0]
        _fr_rows.append({'month': m_, 'requests': len(g_), 'median |fresh move| (bp)': float(g_['bt_level_fresh_bp'].abs().median()), 'share |fresh| > 5 bp': float((g_['bt_level_fresh_bp'].abs() > 5).mean()),
                         'Spearman with the street error': float(g_['bt_level_fresh_bp'].corr(g_['e_street'], method='spearman')), 'slope of the street error on the fresh move': float(sl)})
bt_fresh_tab = pd.DataFrame(_fr_rows).set_index('month') if _fr_rows else pd.DataFrame()
print('The fresh factor-implied level (beta\' dF + MMD move at tenor since the bond\'s last observation) against the street error e = 100 (competing level - m); a positive slope says the stale mark is part of the error:')
display(bt_fresh_tab.round(3))

fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
_smp = np.flatnonzero(_gz_v); _smp = RNG.choice(_smp, min(20000, len(_smp)), replace=False) if len(_smp) else _smp
for k in range(3):
    ax[k].scatter(_gz_own[_smp, k], BT_BETA_RQ[_smp, k], s=3, alpha=0.3, color='#4C72B0'); lo, hi = np.nanpercentile(_gz_own[_smp, k], [1, 99]) if len(_smp) else (0, 1)
    ax[k].plot([lo, hi], [lo, hi], 'k--', lw=0.8); ax[k].set_xlabel(f'beta {k + 1} of record'); ax[k].set_ylabel('Gamma\' z, request fields')
    ax[k].set_title(f'factor {k + 1}: Pearson {bt_gz_valid.loc[("request fields", k + 1), "Pearson"]:.3f}', fontsize=10)
_gz_src_share.plot.barh(ax=ax[3], color='#55A868'); ax[3].set_title('Source of beta on winnable requests', fontsize=10); ax[3].set_xlim(0, 1)
savefig('14d_gamma_z_betas')
record('lab_gz_betas', validation=flat_records(bt_gz_valid), passes=_gz_pass, label=GZ_LABEL, beta_source=_gz_src_share.to_dict(), instruments_missing=_gz_missing,
       sigma_model=flat_records(bt_gz_sigma), fresh_level=flat_records(bt_fresh_tab) if len(bt_fresh_tab) else [], seconds=time.perf_counter() - _gz_t0)
print(f'[13] {time.perf_counter() - _gz_t0:.0f}s')

# %%
CELL_T('14d. Sigma calibration: EWMA against the expanding window [14]')
PRE_REGISTERED.update({
    'lab_sigma_calibration': ('fixed 2026-10-21 before scoring: candidates for the risk-factor covariance = the 14b expanding window (control) and EWMA covariances of the stacked MMD '
                              'tenor moves and IPCA factor realisations with half-lives 10, 20 and 40 business days (zero mean, adjust-weighted, days strictly before the request '
                              'date, available from the same first day as 14b); idiosyncratic variance = the EWMA of the bond\'s squared residual with the same half-life where '
                              'the bond has >= 10 residual observations, else the [13] sigma; scored on the books of vanilla and the objective with the [13] exposures: by month, '
                              'the realised over ex-ante ratio (RMS of the book\'s daily rate-plus-factor move over the RMS of the ex-ante systematic sd) and the Gaussian log score; '
                              'walk-forward pick per month = the candidate with the highest summed log score on the earlier months with Sigma (control in the first such month); '
                              'bar: the picked Sigma\'s ratio lies in [0.8, 1.25] on the vanilla book in >= 4 of the 5 test months and in the last month; if the bar fails the '
                              'best candidate is still used downstream, labelled uncalibrated'),
})
GZ_HALFLIVES = (10, 20, 40)
GZ_RATIO_LO, GZ_RATIO_HI, GZ_MIN_IDIO_OBS = 0.8, 1.25, 10
_sc_t0 = time.perf_counter()


def bt_ewma_cov_path(R: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA covariance after each row (index n = rows < n), zero mean, weights normalised (adjust=True)."""
    lam = 0.5 ** (1.0 / halflife); T, D = R.shape; out = np.zeros((T + 1, D, D)); S = np.zeros((D, D)); w = 0.0
    for t in range(T):
        S = lam * S + np.outer(R[t], R[t]); w = lam * w + 1.0; out[t + 1] = S / w
    return out


def bt_sigma_by_day(path: np.ndarray) -> dict:
    return {j: path[n] for j, n in enumerate(_nb) if n >= _D + 1}


BT_SIGMA_CAND = {'expanding': BT_SIGMA, **{f'EWMA hl={h}': bt_sigma_by_day(bt_ewma_cov_path(_Rv, h)) for h in GZ_HALFLIVES}}

# ---- idiosyncratic variance per candidate: the EWMA of the bond's squared residual at its last observation before the request
_rs_ = resid[['cusip', 'date', 'pit_residual']].dropna().sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
_rs_gid = _rs_.groupby('cusip', sort=False).ngroup().to_numpy(); _rs_['n_obs'] = _rs_.groupby('cusip', sort=False).cumcount() + 1
_gz_nobs = bt_last_obs(rv_wide(_rs_, 'n_obs'), _d02, _bi2, _on2)[0]
BT_IDV_CAND = {'expanding': BT_IDV_GZ}
for h in GZ_HALFLIVES:
    _rs_['v'] = group_ewm_by_order(_rs_['pit_residual'].to_numpy(float) ** 2, _rs_gid, h)
    own_v = bt_last_obs(rv_wide(_rs_, 'v'), _d02, _bi2, _on2)[0]
    BT_IDV_CAND[f'EWMA hl={h}'] = np.where(np.isfinite(own_v) & (own_v > 0) & (np.nan_to_num(_gz_nobs) >= GZ_MIN_IDIO_OBS), own_v, BT_IDV_GZ)


def bt_book_g(win: np.ndarray, expo: np.ndarray, V: np.ndarray, sigma: dict) -> dict:
    """The 14b book with any exposure, idiosyncratic variance and covariance: start-of-day exposure, ex-ante sd (systematic and total), realised rate+factor move, gross DV01."""
    w = win & (_pb_u > 0)
    F = np.zeros((_pb_nd + 1, _D)); I = np.zeros(_pb_nd + 1); U = np.zeros(_pb_nd + 1)
    np.add.at(F, _pb_di[w] + 1, expo[w]); np.add.at(I, _pb_di[w] + 1, V[w]); np.add.at(U, _pb_di[w] + 1, _pb_u[w])
    F, I, U = F.cumsum(0), I.cumsum(0), U.cumsum(0)
    j = np.arange(_pb_nd); lo = np.maximum(0, j - (BT_HOLD - 1)); E, Iv, Uv = F[j] - F[lo], I[j] - I[lo], U[j] - U[lo]
    sys_ = np.array([E[k] @ sigma[k] @ E[k] if k in sigma else np.nan for k in j])
    ri = BT_RISK_R.index.get_indexer(_pb_days); mv = np.where(ri >= 0, np.einsum('ij,ij->i', E, _Rv[np.maximum(ri, 0)]), np.nan)
    return {'E': E, 'I': Iv, 'dv01': Uv, 'sd_sys': np.sqrt(np.maximum(sys_, 0)), 'sd': np.sqrt(np.maximum(sys_, 0) + Iv), 'move': mv, 'idio share': Iv / np.where(np.maximum(sys_, 0) + Iv > 0, np.maximum(sys_, 0) + Iv, np.nan)}


_pb_month = np.array(_pb_days.strftime('%Y-%m')); _sig_months = sorted(set(_pb_month[list(BT_SIGMA)]))
_cal_rows = []
for cand, S_ in BT_SIGMA_CAND.items():
    for bname, w_ in _pb_win.items():
        bk = bt_book_g(w_, BT_EXPO_GZ, BT_IDV_CAND[cand] * _pb_u ** 2, S_)
        for m in _sig_months:
            d = (_pb_month == m) & np.isfinite(bk['sd_sys']) & np.isfinite(bk['move']) & (bk['sd_sys'] > 0)
            if d.sum() < 5:
                continue
            s2, mv = bk['sd_sys'][d] ** 2, bk['move'][d]
            _cal_rows.append({'candidate': cand, 'book': bname, 'month': m, 'days': int(d.sum()), 'realised / ex-ante': float(np.sqrt(np.mean(mv ** 2) / np.mean(s2))),
                              'log score': float(np.mean(-0.5 * np.log(2 * np.pi * s2) - mv ** 2 / (2 * s2)))})
bt_sigma_cal = pd.DataFrame(_cal_rows)
_ratio_tab = bt_sigma_cal.pivot_table(index=['book', 'candidate'], columns='month', values='realised / ex-ante')
print('Realised over ex-ante ratio of the book\'s daily rate-plus-factor move by month (1 = calibrated; the 14b expanding window is the control):'); display(_ratio_tab.round(3))
_ls = bt_sigma_cal.groupby(['candidate', 'month'])['log score'].sum().unstack('month')
# walk-forward pick: the candidate with the highest summed log score on the earlier months with Sigma; the control in the first month
_pick_sig, _pick_rows_s = {}, []
for i_, m in enumerate(_sig_months):
    if i_ == 0:
        _pick_sig[m] = 'expanding'; _pick_rows_s.append({'month': m, 'picked': 'expanding', 'reason': 'first month with Sigma'}); continue
    sc = _ls[[x for x in _sig_months[:i_] if x in _ls.columns]].sum(axis=1)
    _pick_sig[m] = str(sc.idxmax()); _pick_rows_s.append({'month': m, 'picked': _pick_sig[m], **{f'log score {c}': float(v) for c, v in sc.items()}})
bt_sigma_pick = pd.DataFrame(_pick_rows_s).set_index('month')
print('Walk-forward pick of the covariance by month (summed log score over the earlier months, both books):'); display(bt_sigma_pick.round(2))
BT_SIGMA_C = {j: BT_SIGMA_CAND[_pick_sig[_pb_month[j]]][j] for j in BT_SIGMA if _pb_month[j] in _pick_sig}
_req_pick = pd.Series(_month).map(_pick_sig).fillna('expanding').to_numpy()
BT_IDV_C = BT_IDV_GZ.copy()
for cand in BT_IDV_CAND:
    sel = _req_pick == cand; BT_IDV_C[sel] = BT_IDV_CAND[cand][sel]
BT_V_C = _pb_u ** 2 * BT_IDV_C
_picked_ratio = {m: float(_ratio_tab.loc[('vanilla', _pick_sig[m]), m]) if ('vanilla', _pick_sig[m]) in _ratio_tab.index and m in _ratio_tab.columns else np.nan for m in _sig_months}
_test_sig = [m for m in _sig_months if m in _test_m]
_in_band = {m: bool(np.isfinite(_picked_ratio[m]) and GZ_RATIO_LO <= _picked_ratio[m] <= GZ_RATIO_HI) for m in _test_sig}
_sig_pass = bool(sum(_in_band.values()) >= min(4, len(_test_sig)) and (_in_band.get(_test_sig[-1], False) if _test_sig else False))
SIGMA_LABEL = 'calibrated' if _sig_pass else 'uncalibrated'
print(f'picked Sigma, realised / ex-ante on the vanilla book by test month: {dict((m, round(v, 3)) for m, v in _picked_ratio.items() if m in _test_sig)} | in [{GZ_RATIO_LO}, {GZ_RATIO_HI}]: {_in_band}')
print(f'VERDICT (lab_sigma_calibration): {"passes" if _sig_pass else "fails"} the calibration bar; the picked covariance is labelled {SIGMA_LABEL}')

# ---- the 14c risk attribution redone with the [13] betas and the picked Sigma: does the factor share move?
_vr_c, _vf_c, _vi_c = (np.full(len(BT), np.nan) for _ in range(3))
for j, S in BT_SIGMA_C.items():
    r_ = _pb_rows_day.get(j)
    if r_ is None:
        continue
    t_ = _pb_ti[r_]; b_ = BT_BETA_GZ[r_]
    _vr_c[r_] = S[t_, t_]; _vf_c[r_] = 2.0 * np.einsum('ij,ij->i', b_, S[_nT:, t_].T) + np.einsum('ij,jk,ik->i', b_, S[_nT:, _nT:], b_); _vi_c[r_] = BT_IDV_C[r_]
BT_VOL_C = {'rate + factor': np.sqrt(BT_HOLD * np.maximum(_vr_c + _vf_c, 0.0)), 'idiosyncratic': np.sqrt(BT_HOLD * _vi_c), 'total': np.sqrt(BT_HOLD * np.maximum(_vr_c + _vf_c + _vi_c, 0.0))}
_wv_c = _winable & np.isfinite(BT_VOL_C['total'])
_att_c = pd.DataFrame({'tenor': BT['tenor_bucket'].astype(str).to_numpy(), 'rate var': _vr_c, 'factor var': _vf_c, 'idio var': _vi_c, 'total vol (bp, h days)': BT_VOL_C['total']})[_wv_c].groupby('tenor').mean()
for c_ in ['rate var', 'factor var', 'idio var']:
    _att_c[f'{c_.split()[0]} share'] = _att_c[c_] / _att_c[['rate var', 'factor var', 'idio var']].sum(axis=1)
_att_cmp = _att_c[['rate share', 'factor share', 'idio share', 'total vol (bp, h days)']].join(_ag[['rate share', 'factor share', 'idio share', 'total vol (bp, h days)']], lsuffix=' [13]+[14]', rsuffix=' 14c')
print(f'Risk attribution of winnable requests ({BT_HOLD}-day variance) with the [13] betas and the picked Sigma, beside the 14c version:'); display(_att_cmp.round(3))

fig, ax = plt.subplots(1, 3, figsize=(22, 5))
for cand, g_ in _ratio_tab.loc['vanilla'].iterrows():
    ax[0].plot(g_.index, g_.to_numpy(float), 'o-', lw=1.1, label=cand)
ax[0].axhspan(GZ_RATIO_LO, GZ_RATIO_HI, color='#55A868', alpha=0.15); ax[0].axhline(1, color='k', lw=0.6); ax[0].legend(fontsize=8); ax[0].set_title('Realised / ex-ante by month, vanilla book (band = the bar)', fontsize=10)
_bk_e = bt_book_g(_pb_win['vanilla'], BT_EXPO_GZ, BT_V_C, BT_SIGMA); _bk_p = bt_book_g(_pb_win['vanilla'], BT_EXPO_GZ, BT_V_C, BT_SIGMA_C)
ax[1].plot(_pb_days, np.abs(_bk_e['move']) / 1e3, color='#999999', lw=0.8, label='|realised daily move|'); ax[1].plot(_pb_days, _bk_e['sd_sys'] / 1e3, color='#000000', lw=1.1, label='ex-ante, expanding')
ax[1].plot(_pb_days, _bk_p['sd_sys'] / 1e3, color='#C44E52', lw=1.1, label='ex-ante, picked'); ax[1].legend(fontsize=8); ax[1].set_title(r'Vanilla book: systematic ex-ante sd vs realised (\$k)', fontsize=10)
_sh = _att_cmp[['rate share [13]+[14]', 'factor share [13]+[14]', 'idio share [13]+[14]']]; _b_ = np.zeros(len(_sh))
for c_, col in zip(_sh.columns, ['#C44E52', '#4C72B0', '#55A868']):
    ax[2].bar(_sh.index, _sh[c_], bottom=_b_, color=col, label=c_.replace(' [13]+[14]', '')); _b_ += _sh[c_].to_numpy()
ax[2].legend(fontsize=8); ax[2].set_title('Risk attribution by tenor with Gamma-z betas and the picked Sigma', fontsize=10); ax[2].tick_params(axis='x', labelsize=8)
savefig('14d_sigma_calibration')
record('lab_sigma_calibration', ratios=flat_records(_ratio_tab.reset_index()), picks=flat_records(bt_sigma_pick.reset_index()), picked_ratio=_picked_ratio, passes=_sig_pass, label=SIGMA_LABEL,
       attribution=flat_records(_att_cmp), seconds=time.perf_counter() - _sc_t0)
print(f'[14] {time.perf_counter() - _sc_t0:.0f}s')

# %%
CELL_T('14d. The objective net of the market, and new wins valued conservatively [15]')
PRE_REGISTERED.update({
    'lab_rfq_netb': ('fixed 2026-10-21 before scoring: (a) the objective variant rebuilt with the selection bias on the other-side markout net of the day\'s common markout '
                     '(Xm), same James-Stein shrinkage across keys and the same walk-forward, added to the leaderboard as its own row and scored by the 14 bar; the objective '
                     'line is kept only if the net-b row has delta >= 0.05 bp per request net of the common markout with weekly NW t >= 2 and >= 4 of its test months positive, '
                     'else the original objective is recorded as market beta, not selection; (b) every variant and every 14c formation re-scored with new wins (the variant wins, '
                     'vanilla does not) valued at 100 (y - y_trade) plus the day\'s common markout only, and separately plus the factor-implied move over the holding (MMD at tenor '
                     'plus beta\' f, the 14c attribution; the common-markout value where the attribution is missing); common fills and wins given up are unchanged; a variant '
                     'that raises the win rate passes only if it passes the 14 bar on the record valuation and on the common-markout valuation'),
})
_nb_t0 = time.perf_counter()
NETB_MIN_DELTA, NETB_MIN_T, NETB_MIN_MONTHS = 0.05, 2.0, 4
_rec_xm = f'{BT_H_RECORD}_Xm'
_b_net = BT[f'M_{_rec_xm}'].to_numpy(float) - (_xlog + 100.0 * _adj)
_oo_net = np.full(len(BT), np.nan); _ob_net_rows = []; BT_OBJ_B_NET = np.full(len(BT), np.nan)
for m in BT_MONTHS[1:]:
    tr = bt_train_rows(m) & _wlog & np.isfinite(_b_net)
    if tr.sum() == 0:
        continue
    g = pd.DataFrame({'k': _keys[tr], 'b': _b_net[tr]}).groupby('k')['b'].agg(['mean', 'var', 'size'])
    b_bar = float(_b_net[tr].mean()); se2 = (g['var'].fillna(g['var'].median()) / g['size']).to_numpy()
    tau2 = max(0.0, float(np.var(g['mean'].to_numpy()) - np.mean(se2)))
    b_key = dict(zip(g.index, b_bar + (tau2 / (tau2 + se2)) * (g['mean'].to_numpy() - b_bar)))
    te = _month == m
    for (f, k), rows in pd.DataFrame({'f': _files[te], 'k': _keys[te]}).groupby(['f', 'k']).indices.items():
        q, p_prod = BT_CURVES[f][k]; idx = np.flatnonzero(te)[rows]
        BT_OBJ_B_NET[idx] = float(b_key.get(k, b_bar))
        ob, _, _ = bt_optimize(q, p_prod, _ch[idx], BT_OBJ_B_NET[idx][0])
        _oo_net[idx] = bt_opto(_m_sig[idx], ob)
    _ob_net_rows.append({'test month': m, 'train wins': int(tr.sum()), 'pooled b, net of common (bp)': b_bar, 'between-key sd (bp)': np.sqrt(tau2), 'keys': len(g)})
NETB_NAME = 'objective: selection-adjusted value, net b'
BT_VARIANTS[NETB_NAME] = bt_variant('objective', _oo_net)
_obn = pd.DataFrame(_ob_net_rows).set_index('test month')
print('Selection bias net of the day\'s common markout (b on Xm), beside the raw b of 14 [6]:')
display(_obn.join(pd.DataFrame(_ob_rows).set_index('test month')[['pooled b (bp)']].rename(columns={'pooled b (bp)': 'pooled b, raw (bp)'})).round(2))
_r_net, _per_net = bt_score(NETB_NAME, BT_VARIANTS[NETB_NAME]['y'], _exits); BT_PER[NETB_NAME] = _per_net
rows_ = _per_net['rows'].to_numpy(); pts = []
for c, yg in _gw.items():
    wg = bt_outcomes(BT.iloc[rows_], yg[rows_])['win'].to_numpy(); pg = bt_pnl_rows(rows_, yg[rows_], wg, _M); pts.append((float(wg.mean()), float(np.nanmean(pg))))
feas = [p for w_, p in pts if w_ >= _r_net['win rate'] - 1e-12]
_r_net['width frontier P&L at its win rate (bp)'] = max(feas) if feas else np.nan; _r_net['above width frontier (bp)'] = _r_net['P&L per request (bp)'] - _r_net['width frontier P&L at its win rate (bp)']
_r_net['passes'] = bool(_r_net['NW t (weekly)'] >= BT_BAR_T and _r_net['months > 0'] >= min(BT_BAR_MONTHS, _r_net['months']) and _r_net['above width frontier (bp)'] >= 0)
bt_board = pd.concat([bt_board.drop(columns='rank'), pd.DataFrame([_r_net]).set_index('variant')]).sort_values('NW t (weekly)', ascending=False, na_position='last')
bt_board.insert(0, 'rank', np.arange(1, len(bt_board) + 1))
_net_col = f'delta, {BT_EXIT_NAMES[_rec_xm]} (bp)'
_netb_keep = bool(_r_net[_net_col] >= NETB_MIN_DELTA and _r_net['NW t (weekly)'] >= NETB_MIN_T and _r_net['months > 0'] >= min(NETB_MIN_MONTHS, _r_net['months']))
_obj_label = 'selection (net-b row passes)' if _netb_keep else 'market beta, not selection (net-b row fails)'
print('Leaderboard with the net-b objective (record exit; the delta net of the common markout beside it):')
display(bt_board.loc[[NETB_NAME, 'objective: selection-adjusted value', 'width (walk-forward pick)', 'vanilla (production replay)'],
                     ['rank', 'win rate', 'delta per request (bp)', _net_col, 'NW t (weekly)', 'months > 0', 'months', 'delta $ per month ($k)', 'above width frontier (bp)', 'passes']].round(4))
print(f'VERDICT (lab_rfq_netb, objective line): {_obj_label}')


def bt_score_cons(name: str) -> dict:
    """The 14 delta against vanilla under three valuations of a new win: the record exit, the day's common markout only, the factor-implied move over the holding."""
    per = BT_PER[name]; rows = per['rows'].to_numpy(); wv, w0 = per['win_v'].to_numpy(bool), per['win_0'].to_numpy(bool); pv, p0, S = per['pv'].to_numpy(float), per['p0'].to_numpy(float), per['S'].to_numpy(bool)
    y = BT_VARIANTS[name]['y'][rows]; px = 100.0 * (y - _y_tr[rows]); new = wv & ~w0
    mk = BT[f'mkt_{BT_H_RECORD}'].to_numpy(float)[rows]; sysm = (BT_MKT + BT_FAC)[rows]
    pvA = np.where(new, px + mk, pv); pvB = np.where(new & np.isfinite(sysm), px + sysm, pvA)
    out = {'variant': name, 'new wins share of requests': float(new.mean())}
    for lab, p in [('record', pv), ('common markout', pvA), ('factor-implied move', pvB)]:
        d = p - p0; wk = pd.Series(d[S]).groupby(per['week'].to_numpy()[S]).mean(); mm = pd.Series(d[S]).groupby(per['month'].to_numpy()[S]).mean()
        out[f'delta, {lab} (bp)'] = float(np.nanmean(d[S])); out[f'NW t, {lab}'] = nw_t(wk.to_numpy()); out[f'months > 0, {lab}'] = int((mm > 0).sum())
    out['factor-implied fallback share'] = float((new & ~np.isfinite(sysm)).sum() / max(int(new.sum()), 1))
    return out


bt_cons = pd.DataFrame([bt_score_cons(nm) for nm in BT_VARIANTS]).set_index('variant')
bt_cons['passes, record'] = bt_board['passes'].reindex(bt_cons.index)
bt_cons['passes, record and common markout'] = bt_cons['passes, record'] & (bt_cons['NW t, common markout'] >= BT_BAR_T) & (bt_cons['months > 0, common markout'] >= np.minimum(BT_BAR_MONTHS, bt_board['months'].reindex(bt_cons.index)))
bt_cons = bt_cons.loc[bt_board.index]
print('Every variant under the three valuations of a new win (the delta against vanilla, bp per request; common fills and wins given up are identical across the three):')
display(bt_cons.round(4))
BT_CONS_VERDICT = f'{int(bt_cons["passes, record and common markout"].sum())} of {len(bt_cons) - 1} variants pass on the record and the common-markout valuation' + \
    (f': {bt_cons.index[bt_cons["passes, record and common markout"]].tolist()}' if bt_cons['passes, record and common markout'].any() else '')
print('VERDICT (lab_rfq_netb, conservative bar):', BT_CONS_VERDICT)

# ---- the 14c formations under the same three valuations, on the test months
_fm_cons_rows = []
for nm, o in {_VAN: _opt_van, **BT_FM_OPTO}.items():
    y = o + _adj; win = np.zeros(len(BT), bool); ok = np.isfinite(y) & _tmw; win[ok] = bt_win_fast(np.flatnonzero(ok), y[ok])
    px = 100.0 * (y - _y_tr); new = win & ~_wv0; mk = BT[f'mkt_{BT_H_RECORD}'].to_numpy(float); sysm = BT_MKT + BT_FAC
    pv = np.where(win & _tm, px + _M, 0.0); pv[_tm & _winable & ~np.isfinite(_M)] = np.nan
    pvA = np.where(new & _tm, px + mk, pv); pvB = np.where(new & _tm & np.isfinite(sysm), px + sysm, pvA)
    row = {'formation': nm, 'win rate': float(win[_tm].sum() / _tm.sum()), 'new wins': int((new & _tm).sum())}
    for lab, p in [('record', pv), ('common markout', pvA), ('factor-implied move', pvB)]:
        row[f'P&L per request, {lab} (bp)'] = float(np.nanmean(p[_tm])); row[f'$ per month, {lab} ($k)'] = float(np.nansum(p[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3)
    _fm_cons_rows.append(row)
bt_fm_cons = pd.DataFrame(_fm_cons_rows).set_index('formation')
for lab in ['record', 'common markout', 'factor-implied move']:
    bt_fm_cons[f'delta vs vanilla, {lab} (bp)'] = bt_fm_cons[f'P&L per request, {lab} (bp)'] - bt_fm_cons.loc[_VAN, f'P&L per request, {lab} (bp)']
print('The 14c formations (win rate matched to production within the month) under the three valuations, test months:'); display(bt_fm_cons.round(4))

fig, ax = plt.subplots(1, 2, figsize=(20, 5.5))
_top = [n for n in bt_board.index if n != 'vanilla (production replay)'][:8]; _xs = np.arange(len(_top))
for k_, (lab, col) in enumerate([('record', '#000000'), ('common markout', '#4C72B0'), ('factor-implied move', '#55A868')]):
    ax[0].bar(_xs + (k_ - 1) * 0.27, bt_cons.loc[_top, f'delta, {lab} (bp)'], 0.27, color=col, label=lab)
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(_xs); ax[0].set_xticklabels(_top, rotation=25, ha='right', fontsize=8); ax[0].legend(fontsize=8)
ax[0].set_title('Delta vs vanilla (bp per request) under the three valuations of a new win, top variants', fontsize=10)
_fx = np.arange(len(bt_fm_cons))
for k_, (lab, col) in enumerate([('record', '#000000'), ('common markout', '#4C72B0'), ('factor-implied move', '#55A868')]):
    ax[1].bar(_fx + (k_ - 1) * 0.27, bt_fm_cons[f'delta vs vanilla, {lab} (bp)'], 0.27, color=col, label=lab)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xticks(_fx); ax[1].set_xticklabels(bt_fm_cons.index, rotation=25, ha='right', fontsize=8); ax[1].legend(fontsize=8)
ax[1].set_title('The 14c formations: delta vs vanilla under the three valuations', fontsize=10)
savefig('14d_conservative_valuation')
record('lab_rfq_netb', net_b=flat_records(_obn.reset_index()), netb_row={k: (v.item() if hasattr(v, 'item') else v) for k, v in _r_net.items() if not isinstance(v, (list, dict))}, objective_verdict=_obj_label,
       conservative=flat_records(bt_cons), conservative_verdict=BT_CONS_VERDICT, formations=flat_records(bt_fm_cons), leaderboard=flat_records(bt_board), seconds=time.perf_counter() - _nb_t0)
print(f'[15] {time.perf_counter() - _nb_t0:.0f}s')

# %% [markdown]
# ## 14e. Risk-adjusted exposure: the hedged book, the charge decomposed, the manual charge reallocated, and the report
#
# **The question 14b asked, asked properly.** With betas on every request and a calibrated Sigma, what does a marginal-risk charge buy, and what is it
# charging for? Two definitions of the book are scored side by side, because which one is the desk's reality is a desk decision, not a modelling one:
#
# - **full**: the 14b book, the rate block of the MMD curve included, nothing hedged;
# - **hedged**: the fill's tenor DV01 is sold on the fill day at a cost `HEDGE_COST_BP` (1 bp of yield per fill by default), so the book carries IPCA factor and
#   idiosyncratic risk only, and a fill's P&L is its markout minus the MMD move at its tenor minus the hedge cost.
#
# Under either definition the extra charge is $\rho_i = \gamma s_i$ with $s_i$ the 14b marginal risk, now on the [13] exposures and the [14] covariance, and
# $k$ = the median extra charge in bp. The walk-forward pick of $k$ uses the weekly Sharpe of the **hedged** dollar P&L, so that it cannot select market beta.
#
# | Cell | Content |
# |---|---|
# | [16] | $s_i$ decomposed into rate, factor, idiosyncratic and book cross term; the risk-aware replay under both definitions on vanilla, the objective and the net-b objective; the hedged $k$ pick; the frontier three ways; the bar |
# | [17] | What the manual charge knows: a regression on the risk measures, size, signal age, buckets, the bond's recent losses and a month effect; a reallocation that keeps the traders' monthly level and allocates it by risk |
# | [18] | The risk-adjusted exposure table and frontier for every policy |

# %%
CELL_T('14e. The charge decomposed; the risk-aware replay, full and hedged, with the hedged k pick [16]')
PRE_REGISTERED.update({
    'lab_rfq_riskaware_gz': ('fixed 2026-10-21 before scoring: exposures from [13] (own beta else Gamma-z), idiosyncratic variance and Sigma from [14]; two definitions: full = the 14b '
                             'exposure (tenor one-hot and beta), hedged = the rate block removed, a hedge cost HEDGE_COST_BP (default 1 bp per fill) added to the charge of every '
                             'risk-aware policy and to the P&L of every hedged policy, and the fill\'s hedged P&L = 100 (y - y_trade) + M - the MMD move at its tenor over the '
                             'holding - the cost; rho_i = gamma s_i with gamma_k = k / median s on the vanilla book in the first month with Sigma under each definition, k in '
                             '{1, 2, 3, 5} within the 14 width range; bases = vanilla, the objective, the net-b objective; walk-forward pick per test month = the k with the '
                             'highest weekly Sharpe of the hedged dollar P&L on earlier months; a pick passes if, on the hedged series over the test months, its dollars per '
                             'dollar of ex-ante sd is >= 1.3 x its base\'s and its hedged P&L >= 70% of its base\'s in >= 4 of the 5 test months; frontiers reported raw, net of '
                             'the common markout, and hedged'),
})
_ra_t0 = time.perf_counter()
GZ_HEDGE_COST_BP = float(globals().get('HEDGE_COST_BP', 1.0))
GZ_K_GRID = [k for k in BT_GAMMAS if k in (1, 2, 3, 5)] or list(BT_GAMMAS)
GZ_PASS_RATIO, GZ_PASS_PNL, GZ_PASS_MONTHS = 1.3, 0.7, 4
BT_EXPO_H = BT_EXPO_GZ.copy(); BT_EXPO_H[:, :_nT] = 0.0
GZ_DEFS = {'full': (BT_EXPO_GZ, 0.0), 'hedged': (BT_EXPO_H, GZ_HEDGE_COST_BP)}
_gz_bases = {'vanilla': (_ch, _opt_van), 'objective': _pb_bases['objective'], 'objective net b': (_ch + BT_OBJ_B_NET, _oo_net)}
_mkt_rec = BT[f'mkt_{BT_H_RECORD}'].to_numpy(float)


def bt_marginal_g(rows: np.ndarray, j: int, E: np.ndarray, expo: np.ndarray, V: np.ndarray, sigma: dict) -> np.ndarray:
    S = sigma[j]; e = expo[rows]
    return BT_HOLD * (2.0 * e @ (S @ E) + np.einsum('ij,jk,ik->i', e, S, e) + V[rows]) / (2.0 * _pb_u[rows])


def bt_marginal_parts(rows: np.ndarray, j: int, E: np.ndarray, expo: np.ndarray, V: np.ndarray, sigma: dict) -> dict:
    """s_i split into the fill's own rate variance, its factor variance (with the rate-factor cross term), its idiosyncratic variance and the book cross term, bp per unit gamma."""
    S = sigma[j]; e = expo[rows]; er = e.copy(); er[:, _nT:] = 0.0; u2 = 2.0 * _pb_u[rows]
    own = np.einsum('ij,jk,ik->i', e, S, e); rate = np.einsum('ij,jk,ik->i', er, S, er)
    return {'rate': BT_HOLD * rate / u2, 'factor': BT_HOLD * (own - rate) / u2, 'idiosyncratic': BT_HOLD * V[rows] / u2, 'book cross term': BT_HOLD * 2.0 * (e @ (S @ E)) / u2}


def bt_risk_replay_g(base: str, gamma: float, expo: np.ndarray, V: np.ndarray, sigma: dict, extra_bp: float) -> np.ndarray:
    """The 14b sequential replay with any exposure, idiosyncratic variance, covariance and a fixed extra charge (the hedge cost)."""
    c0, o0 = _gz_bases[base]; opto = o0.copy(); F = np.zeros((_pb_nd, _D))
    for j in range(_pb_nd):
        rows = _pb_rows_day.get(j, np.zeros(0, int)); act = rows[np.isfinite(o0[rows]) & (_pb_u[rows] > 0)]
        if not len(act):
            continue
        if j in sigma:
            E = F[max(0, j - (BT_HOLD - 1)):j].sum(0)
            opto[act] = bt_requote(act, c0[act] - extra_bp - gamma * bt_marginal_g(act, j, E, expo, V, sigma))
        F[j] = expo[act][bt_win_fast(act, opto[act] + _adj[act])].sum(0)
    return opto


# ---- what the charge is charging for: s_i decomposed on vanilla's book (full definition, [13] exposures, picked Sigma)
_bk_van_c = bt_book_g(_pb_win['vanilla'], BT_EXPO_GZ, BT_V_C, BT_SIGMA_C)
_parts = {k: np.full(len(BT), np.nan) for k in ['rate', 'factor', 'idiosyncratic', 'book cross term']}
for j in sorted(BT_SIGMA_C):
    r_ = _pb_rows_day.get(j, np.zeros(0, int)); r_ = r_[_winable[r_] & (_pb_u[r_] > 0)]
    if len(r_):
        for k, v in bt_marginal_parts(r_, j, _bk_van_c['E'][j], BT_EXPO_GZ, BT_V_C, BT_SIGMA_C).items():
            _parts[k][r_] = v
_pd = pd.DataFrame({**_parts, 'tenor': BT['tenor_bucket'].astype(str).to_numpy(), 'month': _month})
_pd = _pd[np.isfinite(_pd['rate'])]; _pd['s_i'] = _pd[list(_parts)].sum(axis=1)
_dec_ten = _pd.groupby('tenor')[list(_parts) + ['s_i']].mean(); _dec_mon = _pd.groupby('month')[list(_parts) + ['s_i']].mean()
for t_ in (_dec_ten, _dec_mon):
    for k in _parts:
        t_[f'{k} share'] = t_[k] / t_['s_i']
_s_scale = {}
for dname, (expo_, cost_) in GZ_DEFS.items():
    bk0 = bt_book_g(_pb_win['vanilla'], expo_, BT_V_C, BT_SIGMA_C); vals = []
    for j in sorted(BT_SIGMA_C):
        if _pb_month[j] == _m_sig1:
            r_ = _pb_rows_day.get(j, np.zeros(0, int)); r_ = r_[_pb_u[r_] > 0]
            if len(r_):
                vals.append(bt_marginal_g(r_, j, bk0['E'][j], expo_, BT_V_C, BT_SIGMA_C))
    _s_scale[dname] = float(np.median(np.concatenate(vals))) if vals else _s_med
print(f'Marginal risk s_i on vanilla\'s book, decomposed (bp per unit gamma, mean over winnable requests on days with Sigma; share of the total beside), by tenor bucket:')
display(_dec_ten.round(3))
print('... by month:'); display(_dec_mon.round(3))
print(f'median s on the vanilla book in {_m_sig1}: full {_s_scale["full"]:.4g}, hedged {_s_scale["hedged"]:.4g} bp per unit gamma (14b, own/bucket betas and the expanding Sigma: {_s_med:.4g}); '
      f'gamma_k = k / median s per definition, k in {GZ_K_GRID}; hedge cost {GZ_HEDGE_COST_BP} bp per fill')


def _bt_risk_runs_gz() -> pd.DataFrame:
    out = {}
    for dname, (expo_, cost_) in GZ_DEFS.items():
        for base in _gz_bases:
            for k in GZ_K_GRID:
                t0 = time.perf_counter(); out[f'{base} | {dname} | k={k}'] = bt_risk_replay_g(base, k / _s_scale[dname], expo_, BT_V_C, BT_SIGMA_C, cost_)
                print(f'  {base} | {dname} | k={k}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out)


BT_RISK_OPTO_GZ = cached('lab_rfq_riskaware_gz', _bt_risk_runs_gz,
                         deps=[STAGE_KEYS['lab_rfq_grids'], float(np.nansum(BT_OBJ_B)), float(np.nansum(BT_OBJ_B_NET)), float(_pb_u.sum()), float(BT_EXPO_GZ.sum()), float(BT_V_C.sum()),
                               float(sum(S.sum() for S in BT_SIGMA_C.values())), BT_HOLD, GZ_K_GRID, GZ_HEDGE_COST_BP, _s_scale],
                         code=[bt_risk_replay_g, bt_marginal_g, bt_requote, bt_win_fast, bt_replay, bt_optimize, bt_opto])

# ---- score every policy: raw, net of common markout and hedged P&L; the book under its own definition; the hedged k pick
_pols_gz = {}
for dname in GZ_DEFS:
    for base, (_, o0) in _gz_bases.items():
        _pols_gz[f'{base} | {dname} | k=0'] = o0
        for k in GZ_K_GRID:
            _pols_gz[f'{base} | {dname} | k={k}'] = BT_RISK_OPTO_GZ[f'{base} | {dname} | k={k}'].to_numpy(float)


def bt_policy_pnl_g(opto: np.ndarray, hedged: bool, cost: float) -> dict:
    """Per request: win flag; raw, net-of-common and hedged P&L in bp and dollars (NaN on a winnable request without the needed exit or attribution)."""
    y = opto + _adj; rows = np.arange(len(BT)); win = np.zeros(len(BT), bool); ok = np.isfinite(y)
    win[ok] = bt_win_fast(rows[ok], y[ok]) & _winable[ok]
    px = 100.0 * (y - _y_tr)
    raw = np.where(_winable & ~np.isfinite(_M), np.nan, np.where(win, px + _M, 0.0)); raw[~ok] = np.nan
    net = np.where(_winable & ~np.isfinite(_M), np.nan, np.where(win, px + _M - np.nan_to_num(_mkt_rec), 0.0)); net[~ok] = np.nan
    hed = np.where(_winable & ~(np.isfinite(_M) & np.isfinite(BT_MKT)), np.nan, np.where(win, px + _M - np.nan_to_num(BT_MKT) - (cost if hedged else 0.0), 0.0)); hed[~ok] = np.nan
    u = np.where(_pb_u > 0, _pb_u, np.nan)
    return {'win': win, 'raw': raw, 'net': net, 'hedged': hed, 'usd_raw': raw * u, 'usd_net': net * u, 'usd_hedged': hed * u}


_PPG = {nm: bt_policy_pnl_g(o, nm.split(' | ')[1] == 'hedged', GZ_HEDGE_COST_BP) for nm, o in _pols_gz.items()}
_picks_gz, _pick_rows_gz = {}, []
for dname in GZ_DEFS:
    for base in _gz_bases:
        o = np.full(len(BT), np.nan)
        for m in _test_m:
            tr = bt_train_rows(m) & (_month >= _risk_m0) & np.isfinite(_gz_bases[base][1])
            sc = {k: bt_sharpe(bt_weekly_usd(_PPG[f'{base} | {dname} | k={k}']['usd_hedged'], tr)) for k in [0] + GZ_K_GRID}
            kb = max(sc, key=lambda k: -np.inf if not np.isfinite(sc[k]) else sc[k]); te = _month == m
            o[te] = _pols_gz[f'{base} | {dname} | k={kb}'][te]
            _pick_rows_gz.append({'definition': dname, 'base': base, 'test month': m, 'k picked': kb, **{f'train hedged Sharpe k={k}': v for k, v in sc.items()}})
        nm = f'{base} | {dname} | hedged-Sharpe pick'; _pols_gz[nm] = o; _PPG[nm] = bt_policy_pnl_g(o, dname == 'hedged', GZ_HEDGE_COST_BP)
bt_pick_gz = pd.DataFrame(_pick_rows_gz).set_index(['definition', 'base', 'test month'])
print('Walk-forward pick of k by the weekly Sharpe of the hedged dollar P&L on earlier months:'); display(bt_pick_gz.round(3))

_rows_gz, BT_RISK_BOOKS_GZ = [], {}
for nm, P in _PPG.items():
    base, dname, kl = nm.split(' | '); msk = _tm & np.isfinite(_gz_bases[base][1]); expo_ = GZ_DEFS[dname][0]
    bk = bt_book_g(P['win'], expo_, BT_V_C, BT_SIGMA_C); BT_RISK_BOOKS_GZ[nm] = bk
    wk = {k: bt_weekly_usd(P[f'usd_{k}'], msk) for k in ('raw', 'net', 'hedged')}
    base_nm = f'{base} | {dname} | k=0'; wk0 = bt_weekly_usd(_PPG[base_nm]['usd_hedged'], msk)
    sdv = bk['sd'][_tdays]; sdv = sdv[np.isfinite(sdv)]; mvv = bk['move'][_tdays]; mvv = mvv[np.isfinite(mvv)]
    fe = np.abs(bk['E'][_tdays][:, _nT:]); ish = bk['idio share'][_tdays]
    _rows_gz.append({'policy': nm, 'base': base, 'definition': dname, 'k': kl, 'win rate': float(P['win'][msk].mean()),
                     '$ P&L per month, raw ($k)': float(wk['raw'].sum() / len(_test_m) / 1e3), '$ P&L per month, net of common ($k)': float(wk['net'].sum() / len(_test_m) / 1e3),
                     '$ P&L per month, hedged ($k)': float(wk['hedged'].sum() / len(_test_m) / 1e3), 'weekly Sharpe, raw': bt_sharpe(wk['raw']), 'weekly Sharpe, hedged': bt_sharpe(wk['hedged']),
                     'worst week, hedged ($k)': float(wk['hedged'].min() / 1e3), 'ex-ante daily sd, mean ($k)': float(sdv.mean() / 1e3) if len(sdv) else np.nan,
                     'realised daily move sd ($k)': float(mvv.std() / 1e3) if len(mvv) else np.nan, 'gross DV01, mean ($k/bp)': float(np.nanmean(bk['dv01'][_tdays]) / 1e3),
                     **{f'|factor {k + 1} exposure|, mean ($k)': float(fe[:, k].mean() / 1e3) for k in range(3)}, 'idio variance share, mean': float(np.nanmean(ish)),
                     'hedged $ per $ of ex-ante sd': float(wk['hedged'].sum() / len(_test_m) / sdv.mean()) if len(sdv) and sdv.mean() > 0 else np.nan,
                     'delta hedged $ per month vs base ($k)': float((wk['hedged'] - wk0).sum() / len(_test_m) / 1e3), 'NW t of weekly hedged delta': nw_t((wk['hedged'] - wk0).to_numpy()) if kl != 'k=0' else np.nan})
bt_risk_gz = pd.DataFrame(_rows_gz).set_index('policy')
# the bar, month by month, for each pick against its base
_verd_gz = {}
for dname in GZ_DEFS:
    for base in _gz_bases:
        nm, b0 = f'{base} | {dname} | hedged-Sharpe pick', f'{base} | {dname} | k=0'; okm = 0
        for m in _test_m:
            mm = (_month == m) & np.isfinite(_gz_bases[base][1]); dm = _tdays & (_pb_month == m)
            p_, p0_ = float(np.nansum(_PPG[nm]['usd_hedged'][mm])), float(np.nansum(_PPG[b0]['usd_hedged'][mm]))
            s_, s0_ = float(np.nanmean(BT_RISK_BOOKS_GZ[nm]['sd'][dm])), float(np.nanmean(BT_RISK_BOOKS_GZ[b0]['sd'][dm]))
            okm += int(np.isfinite([p_, p0_, s_, s0_]).all() and s_ > 0 and s0_ > 0 and p0_ > 0 and (p_ / s_) >= GZ_PASS_RATIO * (p0_ / s0_) and p_ >= GZ_PASS_PNL * p0_)
        _verd_gz[nm] = {'months passing': okm, 'passes': bool(okm >= min(GZ_PASS_MONTHS, len(_test_m)))}
bt_risk_gz['months passing the bar'] = pd.Series({k: v['months passing'] for k, v in _verd_gz.items()}).reindex(bt_risk_gz.index)
bt_risk_gz['passes'] = pd.Series({k: v['passes'] for k, v in _verd_gz.items()}).reindex(bt_risk_gz.index).fillna(False).astype(bool)
print(f'Risk-aware policies on the test months {_test_m[0]}..{_test_m[-1]} with the [13] betas and the [14] Sigma ({GZ_LABEL} betas, {SIGMA_LABEL} Sigma); hedged = rate block sold at {GZ_HEDGE_COST_BP} bp per fill:')
display(bt_risk_gz.drop(columns=['base', 'definition', 'k']).round(3))
BT_RISK_GZ_VERDICT = {k: ('passes' if v['passes'] else 'fails') for k, v in _verd_gz.items()}
print('VERDICT (lab_rfq_riskaware_gz):', BT_RISK_GZ_VERDICT)

fig, ax = plt.subplots(2, 2, figsize=(20, 11))
_cols_b = {'vanilla': '#000000', 'objective': '#937860', 'objective net b': '#C44E52'}
for i_, dname in enumerate(GZ_DEFS):
    for base, col in _cols_b.items():
        t_ = bt_risk_gz[(bt_risk_gz['base'] == base) & (bt_risk_gz['definition'] == dname) & bt_risk_gz['k'].str.startswith('k=')]
        ax[0, i_].plot(t_['ex-ante daily sd, mean ($k)'], t_['$ P&L per month, hedged ($k)'], '-o', color=col, label=base)
        for nm, r_ in t_.iterrows():
            ax[0, i_].annotate(r_['k'], (r_['ex-ante daily sd, mean ($k)'], r_['$ P&L per month, hedged ($k)']), fontsize=7, xytext=(3, 3), textcoords='offset points')
        p_ = bt_risk_gz.loc[f'{base} | {dname} | hedged-Sharpe pick']
        ax[0, i_].scatter(p_['ex-ante daily sd, mean ($k)'], p_['$ P&L per month, hedged ($k)'], marker='*', s=160, color=col, zorder=3)
    ax[0, i_].set_xlabel(r'book ex-ante daily sd, mean (\$k)'); ax[0, i_].set_ylabel(r'hedged \$ P&L per month (\$k)'); ax[0, i_].legend(fontsize=8)
    ax[0, i_].set_title(f'{dname} book: hedged P&L against calibrated ex-ante risk, k = 0..{max(GZ_K_GRID)} (star = hedged-Sharpe pick)', fontsize=10)
for base, col in _cols_b.items():
    for dname, ls in [('full', '-'), ('hedged', '--')]:
        nm = f'{base} | {dname} | hedged-Sharpe pick'; msk = _tm & np.isfinite(_gz_bases[base][1]); wk = bt_weekly_usd(_PPG[nm]['usd_hedged'], msk)
        ax[1, 0].plot(wk.index, wk.cumsum().to_numpy() / 1e3, color=col, ls=ls, lw=1.3, label=nm)
        ax[1, 1].plot(_pb_days[_tdays], BT_RISK_BOOKS_GZ[nm]['sd'][_tdays] / 1e3, color=col, ls=ls, lw=1.0, label=nm)
ax[1, 0].axhline(0, color='k', lw=0.5); ax[1, 0].legend(fontsize=7); ax[1, 0].set_title(r'Cumulative hedged \$ P&L of the picks (\$k)', fontsize=10)
ax[1, 1].legend(fontsize=7); ax[1, 1].set_title(r'Book ex-ante daily sd of the picks (\$k, picked Sigma)', fontsize=10)
savefig('14e_riskaware_gz')
record('lab_rfq_riskaware_gz', decomposition_by_tenor=flat_records(_dec_ten), decomposition_by_month=flat_records(_dec_mon), gamma_scale=_s_scale, k_grid=GZ_K_GRID, hedge_cost_bp=GZ_HEDGE_COST_BP,
       picks=flat_records(bt_pick_gz.reset_index()), table=flat_records(bt_risk_gz), verdict=BT_RISK_GZ_VERDICT, labels={'betas': GZ_LABEL, 'sigma': SIGMA_LABEL}, seconds=time.perf_counter() - _ra_t0)
print(f'[16] {time.perf_counter() - _ra_t0:.0f}s')

# %%
CELL_T('14e. What the manual charge knows, and a reallocation that keeps its level [17]')
PRE_REGISTERED.update({
    'lab_rfq_realloc': ('fixed 2026-10-21 before scoring: (a) the manual charge (bp) on winnable requests on days with Sigma regressed on log total, systematic and idiosyncratic '
                        'vol ([13], [14] scale), log quantity, the age of the state-space signal (a missing-age flag), call and tenor bucket dummies, the bond\'s realised loss on '
                        'the desk\'s own fills over the prior 20 business days (exit printed before the request date), and a month effect; pooled R2 with and without the month '
                        'effect, the partial R2 of each block with the month effect kept, and weekly Fama-MacBeth slopes on standardised regressors; (b) reallocation: charge = '
                        'the traders\' mean manual charge of the test month + lambda (g_i - mean g of the month), g = flat (control), systematic vol, idiosyncratic vol, total vol; '
                        'lambda = a multiple m in {0.5, 1, 2} of sd(manual) / sd(g) on the earlier months, m picked per test month by the bp of cover saved on kept cover wins '
                        'over the earlier months subject to keeping >= 80% of vanilla\'s wins; liquidity and risk charges kept; a reallocation passes if on the test months it '
                        'keeps >= 80% of vanilla\'s wins, saves >= 1 bp of cover on kept cover wins against the flat reallocation, and its delta against vanilla with new '
                        'wins at the common markout is >= 0'),
})
_rl_t0 = time.perf_counter()
RL_MULTS, RL_KEEP, RL_COVER_BP, RL_LOSS_WINDOW = (0.5, 1.0, 2.0), 0.8, 1.0, 20

# ---- (a) what the manual charge knows
_rg = _winable & np.isfinite(BT_VOL_C['total']) & (BT_VOL_C['total'] > 0) & (BT_VOL_C['idiosyncratic'] > 0) & (_month >= _risk_m0)
# the bond's realised loss on the desk's own (vanilla) fills whose exit printed in the prior 20 business days
_pv_van = BT_PER['vanilla (production replay)']; _pv_full = np.full(len(BT), np.nan); _pv_full[_pv_van['rows'].to_numpy()] = _pv_van['pv'].to_numpy(float)
_loss = np.where(_wv0 & np.isfinite(_pv_full) & (_pv_full < 0), _pv_full, 0.0)
_xd = _pb_days.searchsorted(pd.to_datetime(_TX).normalize().to_numpy(), side='left'); _xd_ok = ~np.isnat(_TX) & (_xd < _pb_nd)
_bond_idx, _bonds = pd.factorize(_gz_cusip)
_L = np.zeros((len(_bonds), _pb_nd + 1)); np.add.at(_L, (_bond_idx[_xd_ok & (_loss < 0)], _xd[_xd_ok & (_loss < 0)] + 1), _loss[_xd_ok & (_loss < 0)]); _L = _L.cumsum(1)
_own_loss = _L[_bond_idx, _pb_di] - _L[_bond_idx, np.maximum(_pb_di - RL_LOSS_WINDOW, 0)]
_age = BT['age_ss'].to_numpy(float) if 'age_ss' in BT else np.full(len(BT), np.nan)
_X_blocks = {'risk': np.column_stack([np.log(BT_VOL_C['total']), np.log(np.maximum(BT_VOL_C['rate + factor'], 1e-6)), np.log(BT_VOL_C['idiosyncratic'])]),
             'size': np.log(np.maximum(pd.to_numeric(BT['quantity'], errors='coerce').to_numpy(float), 1.0))[:, None],
             'signal age': np.column_stack([np.where(np.isfinite(_age), _age, np.nanmax(_age) if np.isfinite(_age).any() else 0.0), (~np.isfinite(_age)).astype(float)]),
             'call bucket': pd.get_dummies(BT['call_bucket'].astype(str), drop_first=True).to_numpy(float) if 'call_bucket' in BT else np.zeros((len(BT), 0)),
             'tenor bucket': pd.get_dummies(BT['tenor_bucket'].astype(str), drop_first=True).to_numpy(float) if 'tenor_bucket' in BT else np.zeros((len(BT), 0)),
             'own recent loss': _own_loss[:, None]}
_FE = pd.get_dummies(pd.Series(_month), drop_first=True).to_numpy(float)
_yman = _man[_rg]


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    return float(1 - e @ e / np.sum((y - y.mean()) ** 2))


_Xall = np.column_stack([v[_rg] for v in _X_blocks.values()])
_r2_fe, _r2_nofe = _ols_r2(np.column_stack([_Xall, _FE[_rg]]), _yman), _ols_r2(_Xall, _yman); _r2_feonly = _ols_r2(_FE[_rg], _yman)
_partial = {}
for k in _X_blocks:
    Xk = np.column_stack([v[_rg] for kk, v in _X_blocks.items() if kk != k] + [_FE[_rg]]); _partial[k] = _r2_fe - _ols_r2(Xk, _yman)
bt_manual_reg = pd.DataFrame({'R2': {'all blocks + month effect': _r2_fe, 'all blocks, no month effect': _r2_nofe, 'month effect only': _r2_feonly, **{f'partial: {k}': v for k, v in _partial.items()}}})
print(f'What the manual charge knows ({int(_rg.sum()):,} winnable requests on days with Sigma): R2 of the manual charge, and the partial R2 of each block with the month effect kept:')
display(bt_manual_reg.round(4))
# weekly Fama-MacBeth on standardised regressors (no month effect inside a week)
_fm_rows = []
_wk_r = _week[_rg]; _Xs = _Xall.copy(); _mu, _sd = np.nanmean(_Xs, 0), np.nanstd(_Xs, 0); _Xs = (_Xs - _mu) / np.where(_sd > 0, _sd, 1.0)
_names = [f'{k} {i + 1}' if v.shape[1] > 1 else k for k, v in _X_blocks.items() for i in range(v.shape[1])]
for w_ in sorted(set(_wk_r)):
    r_ = _wk_r == w_
    if r_.sum() > 5 * _Xs.shape[1] + 20:
        Xw = np.column_stack([np.ones(int(r_.sum())), _Xs[r_]]); _fm_rows.append(np.linalg.lstsq(Xw, _yman[r_], rcond=None)[0][1:])
_fm = np.array(_fm_rows) if _fm_rows else np.zeros((0, _Xs.shape[1]))
bt_manual_fm = pd.DataFrame({'mean slope (bp per sd)': _fm.mean(0) if len(_fm) else np.nan, 'NW t': [nw_t(_fm[:, i]) for i in range(_Xs.shape[1])] if len(_fm) else np.nan, 'weeks': len(_fm)}, index=_names)
print('Weekly Fama-MacBeth slopes of the manual charge on the standardised regressors (bp per sd):'); display(bt_manual_fm.round(3))

# ---- (b) reallocation: the traders' monthly level, allocated by risk
RL_G = {'flat': np.ones(len(BT)), 'rate + factor': BT_VOL_C['rate + factor'], 'idiosyncratic': BT_VOL_C['idiosyncratic'], 'total': BT_VOL_C['total']}
_rl_months = [m for m in BT_MONTHS if m >= _risk_m0]
_rl_level = {m: float(np.mean(_man[(_month == m) & _winable])) for m in _rl_months}


def _rl_lambda(form: str, mult: float, m: str) -> float:
    """lambda = mult x sd(manual) / sd(g) over the earlier months (the month itself for the first), on winnable requests with g."""
    earlier = [x for x in _rl_months if x < m] or [m]; r_ = np.isin(_month, earlier) & _winable & np.isfinite(RL_G[form])
    sg = float(np.nanstd(RL_G[form][r_])); return mult * float(np.std(_man[r_])) / sg if sg > 0 else 0.0


def _rl_charge(form: str, mult: float, m: str) -> tuple[np.ndarray, np.ndarray]:
    r_ = np.flatnonzero((_month == m) & _winable); g = RL_G[form][r_]; gbar = float(np.nanmean(g)); lam = 0.0 if form == 'flat' else _rl_lambda(form, mult, m)
    return r_, _rl_level[m] + lam * np.nan_to_num(g - gbar)


def _bt_realloc_runs() -> pd.DataFrame:
    out = {}
    for form in RL_G:
        for mult in (RL_MULTS if form != 'flat' else (1.0,)):
            t0 = time.perf_counter(); o = np.full(len(BT), np.nan)
            for m in _rl_months:
                r_, c_ = _rl_charge(form, mult, m); o[r_] = bt_requote(r_, _c_base[r_] - c_)
            out[f'{form} | m={mult:g}'] = o; print(f'  {form} m={mult:g}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out)


BT_RL_OPTO = cached('lab_rfq_realloc', _bt_realloc_runs, deps=[STAGE_KEYS['lab_rfq_replay'], float(np.nansum(BT_VOL_C['total'])), float(_liq.sum() + _rsk.sum() + _man.sum()), _rl_months, RL_MULTS,
                                                              float(sum(S.sum() for S in BT_SIGMA_C.values()))],
                    code=[_rl_charge, _rl_lambda, bt_requote, bt_replay, bt_optimize, bt_opto])


def _rl_cover(opto: np.ndarray, msk: np.ndarray) -> tuple[float, float]:
    """(share of vanilla's wins kept, bp of cover saved on kept cover wins) on the rows in msk."""
    y = opto + _adj; win = np.zeros(len(BT), bool); ok = np.isfinite(y) & msk & _winable; win[ok] = bt_win_fast(np.flatnonzero(ok), y[ok])
    keep = (_grp == 'won, cover') & _wv0 & msk & win; kept = float((win & _wv0 & msk).sum() / max(int((_wv0 & msk).sum()), 1))
    return kept, float((100.0 * (opto - _opt_van))[keep].mean()) if keep.any() else np.nan


_rl_pick_rows, _rl_pol = [], {}
for form in RL_G:
    o = np.full(len(BT), np.nan)
    for m in _test_m:
        earlier = [x for x in _rl_months if x < m]; best, best_sc, best_kept, best_saved = 1.0, -np.inf, np.nan, np.nan
        if form != 'flat':
            for mult in RL_MULTS:
                col = BT_RL_OPTO[f'{form} | m={mult:g}'].to_numpy(float); msk = np.isin(_month, earlier)
                kept, saved = _rl_cover(col, msk) if earlier else (np.nan, np.nan)
                sc = saved if (np.isfinite(saved) and kept >= RL_KEEP) else (-np.inf if not np.isfinite(saved) else saved - 1e6 * (RL_KEEP - kept))
                if sc > best_sc:
                    best, best_sc, best_kept, best_saved = mult, sc, kept, saved
        te = _month == m; o[te] = BT_RL_OPTO[f'{form} | m={best:g}'].to_numpy(float)[te]
        _rl_pick_rows.append({'formation': form, 'test month': m, 'multiple picked': best, 'train wins kept': best_kept, 'train cover saved (bp)': best_saved, 'keeps 80% on train': bool(np.isfinite(best_kept) and best_kept >= RL_KEEP),
                              'traders\' level (bp)': _rl_level[m]})
    _rl_pol[f'reallocation: {form}'] = o
bt_rl_pick = pd.DataFrame(_rl_pick_rows).set_index(['formation', 'test month'])
print('Reallocation: the traders\' mean manual charge of the month as the level, lambda multiple picked on the earlier months by cover saved (subject to keeping 80% of wins):'); display(bt_rl_pick.round(3))

_rl_rows = []
for nm, o in {_VAN: _opt_van, **_rl_pol}.items():
    r_, w_ = bt_fm_score(nm, o)
    y = o + _adj; px = 100.0 * (y - _y_tr); new = w_ & ~_wv0; mk = _mkt_rec; sysm = BT_MKT + BT_FAC
    pv = np.where(w_ & _tm, px + _M, 0.0); pv[_tm & _winable & ~np.isfinite(_M)] = np.nan
    pvA = np.where(new & _tm, px + mk, pv); pvB = np.where(new & _tm & np.isfinite(sysm), px + sysm, pvA)
    hed = np.where(w_ & _tm, px + _M - np.nan_to_num(BT_MKT), 0.0); hed[_tm & _winable & ~(np.isfinite(_M) & np.isfinite(BT_MKT))] = np.nan
    bk = bt_book_g(w_, BT_EXPO_GZ, BT_V_C, BT_SIGMA_C); sdv = bk['sd'][_tdays]; sdv = sdv[np.isfinite(sdv)]
    _rl_rows.append({**{k: r_[k] for k in ['formation', 'win rate', 'vanilla wins kept', 'vanilla wins lost', 'new wins', 'bp left on kept cover wins', '... vanilla, same requests', 'bp saved on kept cover wins',
                                            'crossing on new wins (bp)', 'per fill: P&L per fill (bp)', 'per fill: market (MMD)', 'per fill: IPCA factors', 'per fill: residual', 'P&L per request (bp)', '$ P&L per month ($k)']},
                     'P&L per request, common markout (bp)': float(np.nanmean(pvA[_tm])), 'P&L per request, factor-implied (bp)': float(np.nanmean(pvB[_tm])),
                     '$ per month, hedged ($k)': float(np.nansum(hed[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3), 'ex-ante daily sd, mean ($k)': float(sdv.mean() / 1e3) if len(sdv) else np.nan,
                     'gross DV01, mean ($k/bp)': float(np.nanmean(bk['dv01'][_tdays]) / 1e3)})
bt_rl_tab = pd.DataFrame(_rl_rows).set_index('formation')
for lab in ['common markout', 'factor-implied']:
    bt_rl_tab[f'delta vs vanilla, {lab} (bp)'] = bt_rl_tab[f'P&L per request, {lab} (bp)'] - bt_rl_tab.loc[_VAN, f'P&L per request, {lab} (bp)']
_flat_saved = bt_rl_tab.loc['reallocation: flat', 'bp saved on kept cover wins']
bt_rl_tab['passes'] = (bt_rl_tab['vanilla wins kept'] >= RL_KEEP) & (bt_rl_tab['bp saved on kept cover wins'] - _flat_saved >= RL_COVER_BP) & (bt_rl_tab['delta vs vanilla, common markout (bp)'] >= 0)
bt_rl_tab.loc[[_VAN, 'reallocation: flat'], 'passes'] = False
print(f'Reallocation on the test months {_test_m[0]}..{_test_m[-1]} (liquidity + risk kept; the manual level kept; flat = the control):'); display(bt_rl_tab.round(4))
BT_RL_VERDICT = f'{int(bt_rl_tab["passes"].sum())} reallocation(s) pass the bar' + (f': {bt_rl_tab.index[bt_rl_tab["passes"]].tolist()}' if bt_rl_tab['passes'].any() else '')
print('VERDICT (lab_rfq_realloc):', BT_RL_VERDICT)

fig, ax = plt.subplots(1, 3, figsize=(22, 5))
_pr = bt_manual_reg.loc[[i for i in bt_manual_reg.index if i.startswith('partial')]]['R2']
ax[0].barh([i.replace('partial: ', '') for i in _pr.index], _pr.to_numpy(), color='#4C72B0'); ax[0].set_title(f'Partial R2 of each block for the manual charge (total R2 {_r2_fe:.3f}, month effect alone {_r2_feonly:.3f})', fontsize=10)
_xs = np.arange(len(bt_rl_tab))
ax[1].bar(_xs - 0.2, bt_rl_tab['... vanilla, same requests'], 0.4, color='#000000', label='production'); ax[1].bar(_xs + 0.2, bt_rl_tab['bp left on kept cover wins'], 0.4, color='#4C72B0', label='reallocation')
ax[1].set_xticks(_xs); ax[1].set_xticklabels(bt_rl_tab.index, rotation=20, ha='right', fontsize=8); ax[1].legend(fontsize=8); ax[1].set_title('Cover: bp left on the table on kept cover wins', fontsize=10)
for k_, (lab, col) in enumerate([('record', '#000000'), ('common markout', '#4C72B0')]):
    c_ = 'P&L per request (bp)' if lab == 'record' else 'P&L per request, common markout (bp)'
    ax[2].bar(_xs + (k_ - 0.5) * 0.4, bt_rl_tab[c_] - bt_rl_tab.loc[_VAN, c_], 0.4, color=col, label=lab)
ax[2].axhline(0, color='k', lw=0.6); ax[2].set_xticks(_xs); ax[2].set_xticklabels(bt_rl_tab.index, rotation=20, ha='right', fontsize=8); ax[2].legend(fontsize=8)
ax[2].set_title('Delta vs vanilla, bp per request (record and common-markout valuation)', fontsize=10)
savefig('14e_manual_reallocation')
record('lab_rfq_realloc', regression=bt_manual_reg['R2'].round(5).to_dict(), fama_macbeth=flat_records(bt_manual_fm), picks=flat_records(bt_rl_pick.reset_index()), table=flat_records(bt_rl_tab),
       verdict=BT_RL_VERDICT, seconds=time.perf_counter() - _rl_t0)
print(f'[17] {time.perf_counter() - _rl_t0:.0f}s')

# %%
CELL_T('14e. The risk-adjusted exposure report [18]')
_rp_rows = []
for nm in bt_risk_gz.index:
    base, dname, kl = nm.split(' | ')
    if kl in ('k=0', 'hedged-Sharpe pick'):
        r_ = bt_risk_gz.loc[nm]
        _rp_rows.append({'policy': f'{base}, {dname}, {"production quote" if kl == "k=0" else "risk-aware pick"}', 'win rate': r_['win rate'],
                         '$ per month, raw ($k)': r_['$ P&L per month, raw ($k)'], '$ per month, net of common ($k)': r_['$ P&L per month, net of common ($k)'], '$ per month, hedged ($k)': r_['$ P&L per month, hedged ($k)'],
                         'ex-ante daily sd ($k)': r_['ex-ante daily sd, mean ($k)'], 'realised daily move sd ($k)': r_['realised daily move sd ($k)'], 'hedged $ per $ of sd': r_['hedged $ per $ of ex-ante sd'],
                         'gross DV01 ($k/bp)': r_['gross DV01, mean ($k/bp)'], **{f'|factor {k + 1}| ($k)': r_[f'|factor {k + 1} exposure|, mean ($k)'] for k in range(3)},
                         'idio variance share': r_['idio variance share, mean'], 'worst week, hedged ($k)': r_['worst week, hedged ($k)'], 'Sharpe raw': r_['weekly Sharpe, raw'], 'Sharpe hedged': r_['weekly Sharpe, hedged'],
                         'passes its bar': bool(r_['passes']) if kl != 'k=0' else np.nan})
_best_rl = bt_rl_tab[bt_rl_tab['passes']].sort_values('bp saved on kept cover wins', ascending=False).index[0] if bt_rl_tab['passes'].any() else None
if _best_rl is not None:
    r_ = bt_rl_tab.loc[_best_rl]
    _rp_rows.append({'policy': f'{_best_rl}, full', 'win rate': r_['win rate'], '$ per month, raw ($k)': r_['$ P&L per month ($k)'], '$ per month, hedged ($k)': r_['$ per month, hedged ($k)'],
                     'ex-ante daily sd ($k)': r_['ex-ante daily sd, mean ($k)'], 'gross DV01 ($k/bp)': r_['gross DV01, mean ($k/bp)'], 'passes its bar': True})
bt_exposure = pd.DataFrame(_rp_rows).set_index('policy')
print(f'RISK-ADJUSTED EXPOSURE, test months {_test_m[0]}..{_test_m[-1]}: betas {GZ_LABEL}, Sigma {SIGMA_LABEL}, objective line = {_obj_label}; hedge cost {GZ_HEDGE_COST_BP} bp per fill:')
display(bt_exposure.round(3))
_head = {'vanilla full k=0 hedged $ per $ sd': float(bt_risk_gz.loc['vanilla | full | k=0', 'hedged $ per $ of ex-ante sd']),
         'vanilla full pick hedged $ per $ sd': float(bt_risk_gz.loc['vanilla | full | hedged-Sharpe pick', 'hedged $ per $ of ex-ante sd']),
         'vanilla hedged pick hedged $ per $ sd': float(bt_risk_gz.loc['vanilla | hedged | hedged-Sharpe pick', 'hedged $ per $ of ex-ante sd'])}
print('headline:', {k: round(v, 3) for k, v in _head.items()})

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
for dname, mk_ in [('full', 'o'), ('hedged', 's')]:
    for base, col in _cols_b.items():
        t_ = bt_risk_gz[(bt_risk_gz['base'] == base) & (bt_risk_gz['definition'] == dname)]
        ax[0].scatter(t_['ex-ante daily sd, mean ($k)'], t_['$ P&L per month, hedged ($k)'], marker=mk_, color=col, s=28, label=f'{base}, {dname}')
if _best_rl is not None:
    ax[0].scatter(bt_rl_tab.loc[_best_rl, 'ex-ante daily sd, mean ($k)'], bt_rl_tab.loc[_best_rl, '$ per month, hedged ($k)'], marker='^', color='#55A868', s=80, label=_best_rl)
_v0 = bt_risk_gz.loc['vanilla | full | k=0']; ax[0].scatter(_v0['ex-ante daily sd, mean ($k)'], _v0['$ P&L per month, hedged ($k)'], marker='*', s=220, color='#000000', zorder=4, label='production')
ax[0].set_xlabel(r'book ex-ante daily sd, mean (\$k, calibrated)'); ax[0].set_ylabel(r'hedged \$ P&L per month (\$k)'); ax[0].legend(fontsize=7); ax[0].set_title('Every policy: hedged P&L against calibrated risk', fontsize=10)
_e = bt_exposure[['$ per month, hedged ($k)', 'ex-ante daily sd ($k)']].dropna(); _xs = np.arange(len(_e))
ax[1].bar(_xs - 0.2, _e['$ per month, hedged ($k)'], 0.4, color='#4C72B0', label=r'hedged \$ per month (\$k)'); ax[1].bar(_xs + 0.2, _e['ex-ante daily sd ($k)'], 0.4, color='#C44E52', label=r'ex-ante daily sd (\$k)')
ax[1].set_xticks(_xs); ax[1].set_xticklabels(_e.index, rotation=25, ha='right', fontsize=7); ax[1].legend(fontsize=8); ax[1].set_title('P&L and risk by policy', fontsize=10)
savefig('14e_exposure_report')
record('14e', exposure=flat_records(bt_exposure), headline=_head, best_reallocation=_best_rl, labels={'betas': GZ_LABEL, 'sigma': SIGMA_LABEL, 'objective': _obj_label})

# %% [markdown]
# ## 14f. The exit of record: comparable-bond markouts, validated against real prints, and everything re-scored
#
# **Why.** The backtest framework (`docs/rfq_backtest_framework.html`, Section 6b, agreed with the desk on 2026-10-20) fixes the exit as the level
# where *comparable bonds* print after the request, anchored at each bond's own evaluated close: exit prints P or D only, mark-to-P primary, every
# estimator validated against realised own prints before any variant is scored, and a circularity guard (a variant built on a model must also win under
# an exit that does not use that model). Section 14 did not build that exit. It used the bond's own next print on the *other side* of the trade (a P fill
# exits on S or D prints), which carries the bid-offer every fill earns and exists only on the liquid two thirds of winnable requests. 14d [15] showed
# that every positive result in 14 and 14c depended on it. This section builds the framework's exit, validates it, fixes the exit of record by a
# pre-registered rule, and re-scores every variant, formation, reallocation and risk-aware policy under it.
#
# **The exit.** For bond $i$ requested at $t$, anchor $E_i$ = its evaluated close strictly before $t$. For any P (or D) print of a comparable bond $j$ on
# day $\tau \in (t, t+h]$, the deviation from its own anchor is $d_{j\tau} = 100\,(y_{j\tau} - E_j)$. The peer level is $L_h$ = the median of $d_{j\tau}$
# over the peer set's prints in the window, at least three prints. A fill at $y_q$ is then worth $100\,(y_q - E_i) - L_h$: the entry concession to the
# anchor minus where the peers print against theirs. No level at entry is needed, and the P/D basis to evaluations is carried by $L_h$ itself.
#
# | Estimator | Peer set | Role |
# |---|---|---|
# | bucket, mark-to-P (and mark-to-D) | coupon bin x call structure x rating group x tenor bucket, from characteristics only | model-free; exit-of-record candidate and the circularity guard |
# | IPCA peers, mark-to-P (and mark-to-D) | the 20 nearest bonds in point-in-time beta space within call structure (Section 11 baskets) | model-based |
# | factor / MMD | $\beta' \Delta f$ + the MMD move at the bond's tenor over the holding, from evaluations | full coverage; evaluator-driven |
# | own print, P (and D) | the bond's own next P (D) print within 10 business days | ground truth on the subset that prints; the validation target |
# | lab other side (14 [4]) | the bond's own last S or D print by day 5 | the Section 14 record, kept for continuity |
#
# | Cell | Content |
# |---|---|
# | [19] | anchors, print deviations, the five estimators at h = 5 (1 and 3 for the bucket), validation against the own P print, the exit of record |
# | [20] | the leaderboard, the 14c formations, the 14e reallocations and the risk-aware policies re-scored under the exit of record, the guard, the manual segment |
#
# **Decisions resolved by the runs, written back here for the framework's log:** settle is T+2 (the convention check reproduces logged yields at T+2,
# not T+1); the call-structure pfill keys start 2026-03-27; quantity is par dollars; manual requests are 17 percent of the flow and are scored as a segment.

# %%
CELL_T('14f. Comparable-bond exits, validated; the exit of record [19]')
PRE_REGISTERED.update({
    'lab_exit_of_record': ('fixed 2026-10-21 before any re-scoring: anchor = the bond\'s last evaluated close strictly before the request date (Step 3 panel closing yield); print '
                           'deviation = 100 (print yield - the printing bond\'s own anchor), P and D prints only, |d| <= 300 bp; peer level L_h = median deviation over the peer '
                           'set\'s prints in (t, t + h business days], >= 3 prints; estimators: bucket (coupon bin x call structure x rating group x tenor bucket from characteristics '
                           'only; mark-to-P and mark-to-D), IPCA peers (the Section 11 frozen baskets of the latest weekly formation on or before the previous close, widened to the '
                           'bucket where fewer than 3 peer prints; mark-to-P and mark-to-D), factor / MMD (Gamma-z beta\' dF + MMD move at tenor over h days), own P and own D print '
                           'within 10 business days; validation = on winnable requests with an anchor and an own P print within 10 business days, each estimator\'s predicted deviation '
                           'against the realised own-print deviation: coverage of winnable requests, bias, MAE, slope, by month and pooled, h = 5; exit of record = among the three '
                           'comparable-bond estimators at mark-to-P, the lowest pooled MAE with coverage >= 50% of winnable requests; the bucket mark-to-P exit is the circularity '
                           'guard; the Section 14 other-side own-print exit is reported, not record'),
})
_ef_t0 = time.perf_counter()
EX_H, EX_H_OWN, EX_MIN_PRINTS, EX_DEV_CLIP, EX_MIN_COVER = (1, 3, 5), 10, 3, 300.0, 0.5
EX_BINS = np.arange(-EX_DEV_CLIP, EX_DEV_CLIP + 1.0, 1.0); EX_NBIN = len(EX_BINS) - 1
_ex_dates = BT['date'].to_numpy('datetime64[ns]'); _ex_cusip = BT['cusip'].astype(str).to_numpy()
_TEN_EDGES, _TEN_LABELS = [0, 1.5, 2.5, 3.5, 4.5, 7.5, 9.5, 15.5, 20.5, np.inf], ['1y', '2y', '3y', '4y', '5-7y', '8-9y', '10-15y', '16-20y', '21y+']


def ex_rating_group(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, float)
    return np.where(s >= 21, 'AAAs', np.where(s >= 18, 'AAs', np.where(s >= 15, 'As', np.where(np.isfinite(s), 'BBBs', 'NR'))))


def ex_coupon_bin(cpn: np.ndarray) -> np.ndarray:
    c = np.asarray(cpn, float)
    return np.where(c < 4, 'cpn3', np.where(c < 5, 'cpn4', np.where(np.isfinite(c), 'cpn5', 'cpnNA')))


def ex_call_structure(is_call: np.ndarray, call_date, mat_date, cpn: np.ndarray, yld: np.ndarray, date) -> np.ndarray:
    """The Section 3 rule: priced-to-call if callable, a future call date and the coupon above the yield; else callable to maturity; else bullet."""
    ptc = np.asarray(is_call, bool) & pd.notna(call_date).to_numpy() & (pd.to_datetime(call_date) > pd.to_datetime(date)).to_numpy() & (np.nan_to_num(np.asarray(cpn, float), nan=-1.0) > np.asarray(yld, float))
    ext = ((pd.to_datetime(mat_date) - pd.to_datetime(call_date)).dt.days / 365.25).to_numpy(float)
    ptc = ptc & (np.nan_to_num(ext) > 0.5)
    return np.where(ptc, 'priced-to-call', np.where(np.asarray(is_call, bool), 'callable, to maturity', 'bullet'))


def ex_tenor_bucket(years: np.ndarray) -> np.ndarray:
    return pd.cut(pd.Series(np.asarray(years, float)), _TEN_EDGES, labels=_TEN_LABELS).astype(str).replace('nan', 'NA').to_numpy()


# ---- anchors: the request's bond and every printing bond at their last evaluated close strictly before the date
_rp_cols = ['closing_yield', 'maturity', 'cpn', 'callable', 'nxt_call_dt', 'final_comp_rating', 'comp_rating']
_ex_req = gz_last_row_before(_ex_dates, _ex_cusip, raw_panel, _rp_cols)
BT_ANCHOR = pd.to_numeric(_ex_req['closing_yield'], errors='coerce').to_numpy(float) if 'closing_yield' in _ex_req else np.full(len(BT), np.nan)
_tape = BT_TAPE[BT_TAPE['type'].isin(['P', 'D'])].reset_index(drop=True); _tape['date'] = _tape['ts'].dt.normalize().astype('datetime64[ns]')
_tp_req = gz_last_row_before(_tape['date'].to_numpy('datetime64[ns]'), _tape['cusip'].astype(str).to_numpy(), raw_panel, _rp_cols)
_tape['anchor'] = pd.to_numeric(_tp_req['closing_yield'], errors='coerce').to_numpy(float) if 'closing_yield' in _tp_req else np.nan
_tape['dev'] = 100.0 * (_tape['y'].to_numpy(float) - _tape['anchor'].to_numpy(float))
_tape['di'] = _pb_days.searchsorted(_tape['date'].to_numpy('datetime64[ns]'), side='left')
_tp_keep = (np.isfinite(_tape['dev']) & (_tape['dev'].abs() <= EX_DEV_CLIP) & (_tape['di'] < _pb_nd)).to_numpy() & (np.asarray(_pb_days)[np.minimum(_tape['di'].to_numpy(), _pb_nd - 1)] == _tape['date'].to_numpy())
_tape = _tape[_tp_keep].reset_index(drop=True); _tp_req = _tp_req[_tp_keep].reset_index(drop=True)
print(f'print tape for the exits: {len(_tape):,} P/D prints with an anchor ({(_tape["type"] == "P").mean():.0%} P) on {_tape["cusip"].nunique():,} bonds; request anchors on {np.isfinite(BT_ANCHOR).mean():.1%} of requests, '
      f'{np.isfinite(BT_ANCHOR[_winable]).mean():.1%} of winnable')
# characteristics of the printing bonds for the bucket (the same rule as the request side)
_tp_mat = pd.to_datetime(pd.to_numeric(_tp_req['maturity'], errors='coerce'), unit='ms', errors='coerce')
if _tp_mat.isna().mean() > 0.5:
    _tp_mat = pd.to_datetime(_tp_req['maturity'], errors='coerce')
_tp_call = pd.to_datetime(_tp_req['nxt_call_dt'], errors='coerce'); _tp_is_call = _tp_req['callable'].astype('string').str.upper().isin(['Y', 'YES', 'TRUE', '1']).fillna(False).to_numpy(bool)
_tp_cpn = pd.to_numeric(_tp_req['cpn'], errors='coerce').to_numpy(float)
_tp_score = clean_rating(_tp_req['final_comp_rating']).fillna(clean_rating(_tp_req['comp_rating'])).map(RATING_SCALE).astype(float).to_numpy() if 'final_comp_rating' in _tp_req else np.full(len(_tape), np.nan)
_tape['bucket'] = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(ex_coupon_bin(_tp_cpn).astype(str), '|'), ex_call_structure(_tp_is_call, _tp_call, _tp_mat, _tp_cpn, _tape['anchor'].to_numpy(), _tape['date']).astype(str)), '|'),
                                          np.char.add(ex_rating_group(_tp_score).astype(str), '|')), ex_tenor_bucket(((_tp_mat - _tape['date']).dt.days / 365.25).to_numpy()).astype(str))
# the request's bucket: coupon bin from the key, rating group and call structure from [13], tenor bucket from the track
_rq_cb = BT['pfill_key'].astype(str).str.extract(r'^(cpn\d)')[0].fillna(pd.Series(ex_coupon_bin(_gz_cpn.to_numpy(float)), index=BT.index)).to_numpy(str)
_rq_cs = np.where(_gz_ptc, 'priced-to-call', np.where(_gz_is_call, 'callable, to maturity', 'bullet'))
BT_BUCKET = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(_rq_cb, '|'), _rq_cs.astype(str)), '|'), np.char.add(ex_rating_group(_gz_rscore).astype(str), '|')), BT['tenor_bucket'].astype(str).to_numpy(str))
_bk_ids, _bk_names = pd.factorize(np.concatenate([_tape['bucket'].to_numpy(), BT_BUCKET]))
_tape['bk'] = _bk_ids[:len(_tape)]; _rq_bk = _bk_ids[len(_tape):]
print(f'buckets: {len(_bk_names):,} cells of coupon bin x call structure x rating group x tenor bucket; prints per bucket median {_tape.groupby("bk").size().median():,.0f}; '
      f'requests whose bucket has prints: {np.isin(_rq_bk, _tape["bk"].unique()).mean():.1%}')


def ex_bucket_levels(tape: pd.DataFrame, rq_bk: np.ndarray, rq_di: np.ndarray, h: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Median deviation of the bucket's prints of one side in (t, t + h], and the print count, by cumulative 1-bp histograms per bucket and day."""
    t = tape[tape['type'] == side]
    b = np.digitize(t['dev'].to_numpy(float), EX_BINS) - 1; b = np.clip(b, 0, EX_NBIN - 1)
    H = np.zeros((len(_bk_names), _pb_nd + 1, EX_NBIN), dtype=np.int32); np.add.at(H, (t['bk'].to_numpy(), t['di'].to_numpy() + 1, b), 1); H = H.cumsum(1)
    lvl, cnt = np.full(len(rq_bk), np.nan), np.zeros(len(rq_bk))
    lo, hi = rq_di + 1, np.minimum(rq_di + h, _pb_nd - 1) + 1; ok = (rq_bk >= 0) & (lo <= _pb_nd - 1)
    for s in range(0, len(rq_bk), 100_000):
        r = np.flatnonzero(ok[s:s + 100_000]) + s
        if not len(r):
            continue
        W = H[rq_bk[r], hi[r]] - H[rq_bk[r], lo[r]]; n = W.sum(1); C = W.cumsum(1)
        med = np.argmax(C >= np.ceil(n / 2.0)[:, None], axis=1); lvl[r] = np.where(n >= EX_MIN_PRINTS, EX_BINS[med] + 0.5, np.nan); cnt[r] = n
    return lvl, cnt


def ex_bond_daily(tape: pd.DataFrame, side: str) -> np.ndarray:
    """(days x RV bonds) median deviation of each bond's prints of one side per day; NaN where none."""
    t = tape[tape['type'] == side]; bi = pd.Index(RV_BONDS.astype(str)).get_indexer(t['cusip'].astype(str)); ok = bi >= 0
    g = pd.DataFrame({'di': t['di'].to_numpy()[ok], 'bi': bi[ok], 'dev': t['dev'].to_numpy(float)[ok]}).groupby(['di', 'bi'])['dev'].median()
    D = np.full((_pb_nd, RV_NB), np.nan, dtype=np.float32); D[g.index.get_level_values(0).to_numpy(), g.index.get_level_values(1).to_numpy()] = g.to_numpy(np.float32)
    return D


def ex_peer_levels(D: np.ndarray, peers: np.ndarray, rq_di: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Median over the peers' daily deviations in (t, t + h]; the count of (bond, day) values; NaN where fewer than EX_MIN_PRINTS."""
    lvl, cnt = np.full(len(rq_di), np.nan), np.zeros(len(rq_di)); has = (peers >= 0).any(axis=1)
    days = np.arange(1, h + 1)
    for s in range(0, len(rq_di), 50_000):
        r = np.flatnonzero(has[s:s + 50_000]) + s
        if not len(r):
            continue
        dd = np.minimum(rq_di[r][:, None] + days[None, :], _pb_nd - 1); valid_d = (rq_di[r][:, None] + days[None, :]) <= _pb_nd - 1
        P = peers[r]; pm = P >= 0
        V = D[dd[:, :, None], np.maximum(P, 0)[:, None, :]]                       # (n, h, k)
        V = np.where(pm[:, None, :] & valid_d[:, :, None], V, np.nan).reshape(len(r), -1)
        n = np.isfinite(V).sum(axis=1); cnt[r] = n
        m = np.nanmedian(np.where(np.isfinite(V), V, np.nan), axis=1) if V.size else np.full(len(r), np.nan)
        lvl[r] = np.where(n >= EX_MIN_PRINTS, m, np.nan)
    return lvl, cnt


# ---- the estimators at h = 5 (bucket also at 1 and 3)
_rq_di = _pb_di
BT_EXIT_LVL, BT_EXIT_CNT = {}, {}
for side in ('P', 'D'):
    for h in EX_H:
        if h != BT_H_RECORD and side == 'D':
            continue
        BT_EXIT_LVL[f'bucket {side} h={h}'], BT_EXIT_CNT[f'bucket {side} h={h}'] = ex_bucket_levels(_tape, _rq_bk, _rq_di, h, side)
_dq_rq = _d02 + 1
_peers_rq = rfq_peer_matrix(np.where(_on2, _dq_rq, 0), np.where(_on2, _bi2, 0), RV_FORMS, RV_NB, RV_K) if 'RV_FORMS' in globals() and len(RV_FORMS) else np.full((len(BT), RV_K), -1, dtype=np.int64)
_peers_rq[~_on2] = -1
for side in ('P', 'D'):
    D_ = ex_bond_daily(_tape, side); lvl, cnt = ex_peer_levels(D_, _peers_rq, _rq_di, BT_H_RECORD); own_ok = np.isfinite(lvl)
    wide = ~own_ok & np.isfinite(BT_EXIT_LVL[f'bucket {side} h={BT_H_RECORD}'])
    lvl[wide] = BT_EXIT_LVL[f'bucket {side} h={BT_H_RECORD}'][wide]                # widened to the bucket where the basket is thin
    BT_EXIT_LVL[f'IPCA peers {side} h={BT_H_RECORD}'], BT_EXIT_CNT[f'IPCA peers {side} h={BT_H_RECORD}'] = lvl, cnt
    print(f'IPCA peers mark-to-{side}: basket level on {own_ok[_winable].mean():.1%} of winnable requests, {wide[_winable].mean():.1%} widened to the bucket')
# factor / MMD over h business days from the evaluation: beta' dF + the MMD move at tenor (yield change, bp)
_ex_i0 = np.clip(_d02, 0, len(RV_DATES) - 1); _ex_d5 = _pb_days[np.minimum(_rq_di + BT_H_RECORD, _pb_nd - 1)]
_ex_j0 = _fo.index.searchsorted(RV_DATES[_ex_i0].to_numpy(), side='right'); _ex_j1 = _fo.index.searchsorted(_ex_d5.to_numpy(), side='right')
_ex_m0 = _mmd_idx.searchsorted(RV_DATES[_ex_i0].to_numpy(), side='right') - 1; _ex_m1 = _mmd_idx.searchsorted(_ex_d5.to_numpy(), side='right') - 1
_ex_fac = np.einsum('ij,ij->i', BT_BETA_GZ, _fcum[np.clip(_ex_j1, 0, len(_fcum) - 1)] - _fcum[np.clip(_ex_j0, 0, len(_fcum) - 1)])
_ex_mmd = 100.0 * (_mmd_v[np.maximum(_ex_m1, 0), _tcol] - _mmd_v[np.maximum(_ex_m0, 0), _tcol])
_ex_fok = (_d02 >= 0) & (_ex_m0 >= 0) & (_ex_m1 >= 0) & (_rq_di + BT_H_RECORD <= _pb_nd - 1)
BT_EXIT_LVL[f'factor / MMD h={BT_H_RECORD}'] = np.where(_ex_fok, _ex_fac + _ex_mmd, np.nan)
# own prints within 10 business days, P and D separately
_own10 = cached('lab_rfq_exits_own10', lambda: bt_exits(BT, BT_TAPE, horizons=(EX_H_OWN,), types=('P', 'D')),
                deps=[STAGE_KEYS['lab_rfq_exits'], EX_H_OWN], code=[bt_exits])
BT_OWN_P10, BT_OWN_D10 = _own10[f'yx_{EX_H_OWN}_P'].to_numpy(float), _own10[f'yx_{EX_H_OWN}_D'].to_numpy(float)
_own_devP, _own_devD = 100.0 * (BT_OWN_P10 - BT_ANCHOR), 100.0 * (BT_OWN_D10 - BT_ANCHOR)

# ---- validation against the own P print (and the own D print) on winnable requests with an anchor
_est_names = {f'bucket P h={BT_H_RECORD}': 'bucket, mark-to-P', f'IPCA peers P h={BT_H_RECORD}': 'IPCA peers, mark-to-P', f'factor / MMD h={BT_H_RECORD}': 'factor / MMD',
              f'bucket D h={BT_H_RECORD}': 'bucket, mark-to-D', f'IPCA peers D h={BT_H_RECORD}': 'IPCA peers, mark-to-D'}
_val_base = _winable & np.isfinite(BT_ANCHOR)
_val_rows = []
for key, nm in _est_names.items():
    pred = BT_EXIT_LVL[key]
    for tgt_nm, tgt in [('own P print', _own_devP), ('own D print', _own_devD)]:
        ok = _val_base & np.isfinite(pred) & np.isfinite(tgt) & (np.abs(tgt) <= EX_DEV_CLIP)
        for m_ in ['ALL'] + BT_MONTHS:
            r = ok if m_ == 'ALL' else ok & (_month == m_)
            if r.sum() < 50:
                continue
            e = pred[r] - tgt[r]; sl = np.polyfit(pred[r], tgt[r], 1)[0] if np.std(pred[r]) > 0 else np.nan
            _val_rows.append({'estimator': nm, 'target': tgt_nm, 'month': m_, 'requests': int(r.sum()), 'coverage of winnable': float(np.isfinite(pred[_winable]).mean()),
                              'bias (bp)': float(e.mean()), 'MAE (bp)': float(np.abs(e).mean()), 'slope': float(sl), 'corr': float(np.corrcoef(pred[r], tgt[r])[0, 1])})
bt_exit_val = pd.DataFrame(_val_rows).set_index(['target', 'estimator', 'month'])
_pooled = bt_exit_val.xs('own P print', level='target').xs('ALL', level='month')
print(f'Validation at h = {BT_H_RECORD} against the realised own P print within {EX_H_OWN} business days (winnable requests with an anchor; MAE and bias in bp of the deviation from the anchor):')
display(_pooled.round(3))
print('... by month, MAE against the own P print:'); display(bt_exit_val.xs('own P print', level='target')['MAE (bp)'].unstack('month').round(2))
_cands = [k for k in [f'bucket P h={BT_H_RECORD}', f'IPCA peers P h={BT_H_RECORD}', f'factor / MMD h={BT_H_RECORD}'] if _est_names[k] in _pooled.index and _pooled.loc[_est_names[k], 'coverage of winnable'] >= EX_MIN_COVER]
EX_RECORD_KEY = min(_cands, key=lambda k: _pooled.loc[_est_names[k], 'MAE (bp)']) if _cands else f'bucket P h={BT_H_RECORD}'
EX_GUARD_KEY = f'bucket P h={BT_H_RECORD}'
print(f'EXIT OF RECORD (lab_exit_of_record): {_est_names[EX_RECORD_KEY]} (lowest pooled MAE among comparable-bond mark-to-P estimators with coverage >= {EX_MIN_COVER:.0%}); '
      f'circularity guard: {_est_names[EX_GUARD_KEY]}')

# ---- markouts per estimator in the Section 14 convention: M = 100 (y_trade - exit level), positive = the bond richened after the trade
BT_M_EXIT = {'lab other side (14)': _M}
for key, nm in _est_names.items():
    BT_M_EXIT[nm] = 100.0 * (_y_tr - BT_ANCHOR) - BT_EXIT_LVL[key]
BT_M_EXIT['own P within 10d'] = 100.0 * (_y_tr - BT_OWN_P10); BT_M_EXIT['own D within 10d'] = 100.0 * (_y_tr - BT_OWN_D10)
for h in (1, 3):
    BT_M_EXIT[f'bucket, mark-to-P h={h}'] = 100.0 * (_y_tr - BT_ANCHOR) - BT_EXIT_LVL[f'bucket P h={h}']
EX_RECORD, EX_GUARD = _est_names[EX_RECORD_KEY], _est_names[EX_GUARD_KEY]
_cov_tab = pd.DataFrame({'coverage of winnable requests': {k: float(np.isfinite(v[_winable]).mean()) for k, v in BT_M_EXIT.items()},
                         'mean markout on vanilla fills (bp)': {k: float(np.nanmean(v[_wv0])) for k, v in BT_M_EXIT.items()},
                         'median markout on vanilla fills (bp)': {k: float(np.nanmedian(v[_wv0])) for k, v in BT_M_EXIT.items()}})
print('Every exit: coverage and the markout of production\'s own fills (the lab\'s other-side exit carries the bid-offer; mark-to-P does not):'); display(_cov_tab.round(3))

fig, ax = plt.subplots(1, 3, figsize=(22, 5.5))
_mae = bt_exit_val.xs('own P print', level='target')['MAE (bp)'].unstack('month').drop(columns='ALL', errors='ignore')
for nm, row in _mae.iterrows():
    ax[0].plot(row.index, row.to_numpy(float), 'o-', lw=1.1, label=nm)
ax[0].legend(fontsize=8); ax[0].set_title(f'MAE against the own P print by month (bp), h = {BT_H_RECORD}', fontsize=10)
_sm = np.flatnonzero(_val_base & np.isfinite(BT_EXIT_LVL[EX_RECORD_KEY]) & np.isfinite(_own_devP)); _sm = RNG.choice(_sm, min(20000, len(_sm)), replace=False) if len(_sm) else _sm
ax[1].scatter(BT_EXIT_LVL[EX_RECORD_KEY][_sm], _own_devP[_sm], s=3, alpha=0.3, color='#4C72B0'); ax[1].plot([-60, 60], [-60, 60], 'k--', lw=0.8)
ax[1].set_xlim(-80, 80); ax[1].set_ylim(-150, 150); ax[1].set_xlabel(f'{EX_RECORD}: predicted deviation (bp)'); ax[1].set_ylabel('realised own P print deviation (bp)'); ax[1].set_title('Exit of record against the realised own print', fontsize=10)
_cv = _cov_tab['coverage of winnable requests']; ax[2].barh(_cv.index, _cv.to_numpy(), color='#55A868'); ax[2].set_xlim(0, 1); ax[2].set_title('Coverage of winnable requests by exit', fontsize=10); ax[2].tick_params(axis='y', labelsize=8)
savefig('14f_exit_validation')
record('lab_exit_of_record', validation=flat_records(bt_exit_val.reset_index()), exit_of_record=EX_RECORD, guard=EX_GUARD, coverage=flat_records(_cov_tab), buckets=int(len(_bk_names)),
       anchor_coverage_winnable=float(np.isfinite(BT_ANCHOR[_winable]).mean()), seconds=time.perf_counter() - _ef_t0)
print(f'[19] {time.perf_counter() - _ef_t0:.0f}s')

# %%
CELL_T('14f. Everything re-scored under the exit of record, with the circularity guard [20]')
PRE_REGISTERED.update({
    'lab_rfq_rescore': ('fixed 2026-10-21 before re-scoring: every variant of 14 and 14d, every 14c formation, every 14e reallocation and every 14e risk-aware policy is scored '
                        'under the exit of record from [19] with the Section 14 machinery (paired against vanilla on the same requests, weekly NW t, months positive, the width '
                        'frontier at its win rate); the P&L sample is the winnable requests where the exit of record exists; a variant passes if NW t >= 3, >= 4 test months '
                        'positive and above the width frontier under the exit of record AND, for a variant built on a model (alpha overlays, the objective, the fill-curve refit, '
                        'the risk-aware policies, the model-based reallocations), the same under the bucket mark-to-P exit (the guard); the lab\'s other-side exit and the own-print '
                        'exits are reported beside; manual requests are a reported segment'),
})
_rs_t0 = time.perf_counter()
_MREC, _MGUARD = BT_M_EXIT[EX_RECORD], BT_M_EXIT[EX_GUARD]
_mmd5 = np.where(_ex_fok, _ex_mmd, np.nan)                                            # the MMD move at tenor over the h-day holding, for the hedged P&L
_report_exits = {k: v for k, v in BT_M_EXIT.items() if k in (EX_RECORD, EX_GUARD, 'lab other side (14)', 'IPCA peers, mark-to-P', 'bucket, mark-to-D', 'own P within 10d', 'factor / MMD')}
MODEL_BUILT = ('alpha overlay (street)', 'alpha overlay (exit)', 'fill curve', 'objective')


def bt_score_f(name: str, y: np.ndarray, M_rec: np.ndarray, exits: dict, family: str) -> tuple[dict, pd.DataFrame]:
    rows = np.flatnonzero(np.isfinite(y)); months = sorted(set(_month[rows])); yv, y0 = y[rows], _y_van[rows]
    ov, o0 = bt_outcomes(BT.iloc[rows], yv), bt_outcomes(BT.iloc[rows], y0); wv, w0 = ov['win'].to_numpy(), o0['win'].to_numpy(); grp = ov['group'].to_numpy()
    pv, p0 = bt_pnl_rows(rows, yv, wv, M_rec), bt_pnl_rows(rows, y0, w0, M_rec); S = np.isfinite(pv); nS = max(int(S.sum()), 1); d = pv - p0
    both, new, gone = wv & w0 & S, wv & ~w0 & S, w0 & ~wv & S
    per = pd.DataFrame({'week': _week[rows], 'month': _month[rows], 'd': d, 'S': S, 'usd_v': pv * _usd[rows], 'usd_0': p0 * _usd[rows], 'manual': BT['manual'].to_numpy(bool)[rows]})
    wk = per[per['S']].groupby('week')['d'].mean(); mm = per[per['S']].groupby('month')['d'].mean()
    res = {'variant': name, 'family': family, 'months': len(months), 'requests': len(rows), 'win rate': float(wv.mean()), 'win rate, vanilla': float(w0.mean()),
           'new wins': int((wv & ~w0).sum()), 'wins given up': int((w0 & ~wv).sum()), 'P&L sample share': nS / len(rows),
           'P&L per request (bp)': float(np.nanmean(pv)), '... vanilla (bp)': float(np.nanmean(p0)), 'delta per request (bp)': float(np.nanmean(d)),
           'delta: common fills price': float(np.sum(100.0 * (yv - y0)[both]) / nS), 'delta: new wins': float(np.sum(pv[new]) / nS), 'delta: wins given up': float(-np.sum(p0[gone]) / nS),
           'delta $ per month ($k)': float((np.nansum(per['usd_v']) - np.nansum(per['usd_0'])) / 1e3 / len(months)), 'NW t (weekly)': nw_t(wk.to_numpy()), 'months > 0': int((mm > 0).sum()),
           'delta, manual requests (bp)': float(np.nanmean(d[S & per['manual'].to_numpy()])) if (S & per['manual'].to_numpy()).any() else np.nan,
           'delta, algo requests (bp)': float(np.nanmean(d[S & ~per['manual'].to_numpy()])) if (S & ~per['manual'].to_numpy()).any() else np.nan}
    for key, M_ in exits.items():
        a, b = bt_pnl_rows(rows, yv, wv, M_), bt_pnl_rows(rows, y0, w0, M_); da = a - b; Sa = np.isfinite(da)
        wka = pd.Series(da[Sa]).groupby(_week[rows][Sa]).mean(); mma = pd.Series(da[Sa]).groupby(_month[rows][Sa]).mean()
        res[f'delta | {key} (bp)'] = float(np.nanmean(da)); res[f'NW t | {key}'] = nw_t(wka.to_numpy()); res[f'months > 0 | {key}'] = int((mma > 0).sum())
    # the width frontier at the variant's win rate under the record exit
    pts = []
    for c, yg in _gw.items():
        wg = bt_outcomes(BT.iloc[rows], yg[rows])['win'].to_numpy(); pg = bt_pnl_rows(rows, yg[rows], wg, M_rec); pts.append((float(wg.mean()), float(np.nanmean(pg))))
    feas = [p for w_, p in pts if w_ >= res['win rate'] - 1e-12]
    res['above width frontier (bp)'] = res['P&L per request (bp)'] - (max(feas) if feas else np.nan)
    return res, per.assign(pv=pv, p0=p0, win_v=wv, win_0=w0, rows=rows)


_rs_rows, BT_PER_F = [], {}
for nm, v_ in BT_VARIANTS.items():
    r_, per_ = bt_score_f(nm, v_['y'], _MREC, _report_exits, v_['family']); _rs_rows.append(r_); BT_PER_F[nm] = per_
bt_board_f = pd.DataFrame(_rs_rows).set_index('variant')
_bar_rec = (bt_board_f['NW t (weekly)'] >= BT_BAR_T) & (bt_board_f['months > 0'] >= np.minimum(BT_BAR_MONTHS, bt_board_f['months'])) & (bt_board_f['above width frontier (bp)'] >= 0)
_bar_guard = (bt_board_f[f'NW t | {EX_GUARD}'] >= BT_BAR_T) & (bt_board_f[f'months > 0 | {EX_GUARD}'] >= np.minimum(BT_BAR_MONTHS, bt_board_f['months']))
_is_model = bt_board_f['family'].isin(MODEL_BUILT)
bt_board_f['passes, exit of record'] = _bar_rec & (bt_board_f.index != 'vanilla (production replay)')
bt_board_f['passes, with the guard'] = bt_board_f['passes, exit of record'] & (~_is_model | _bar_guard)
bt_board_f = bt_board_f.sort_values('NW t (weekly)', ascending=False, na_position='last'); bt_board_f.insert(0, 'rank', np.arange(1, len(bt_board_f) + 1))
_show_f = ['rank', 'family', 'win rate', 'new wins', 'wins given up', 'P&L sample share', 'delta per request (bp)', 'delta: common fills price', 'delta: new wins', 'delta: wins given up',
           'delta $ per month ($k)', 'NW t (weekly)', 'months > 0', 'above width frontier (bp)', f'delta | {EX_GUARD} (bp)', f'NW t | {EX_GUARD}', 'passes, exit of record', 'passes, with the guard']
print(f'LEADERBOARD under the exit of record ({EX_RECORD}, h = {BT_H_RECORD}); the guard column is the bucket mark-to-P exit:'); display(bt_board_f[_show_f].round(4))
print('The same delta under every reported exit (bp per request; the lab\'s other-side exit of Section 14 beside the comparable-bond exits):')
display(bt_board_f[[f'delta | {k} (bp)' for k in _report_exits]].round(4))
print('Manual requests as a segment (delta vs vanilla, bp per request, under the exit of record):'); display(bt_board_f[['delta, manual requests (bp)', 'delta, algo requests (bp)']].round(4))
BT_F_VERDICT = (f'{int(bt_board_f["passes, with the guard"].sum())} of {len(bt_board_f) - 1} variants pass under the exit of record with the guard'
                + (f': {bt_board_f.index[bt_board_f["passes, with the guard"]].tolist()}' if bt_board_f['passes, with the guard'].any() else ''))
print('VERDICT (lab_rfq_rescore, variants):', BT_F_VERDICT)


def bt_policy_rows_f(name: str, opto: np.ndarray, hedged_cost: float = 0.0) -> dict:
    """A policy scored on the test months under the exit of record: P&L per request and $ per month, raw and hedged (MMD at tenor over the holding), and under the guard."""
    y = opto + _adj; win = np.zeros(len(BT), bool); ok = np.isfinite(y) & _tm; win[ok] = bt_win_fast(np.flatnonzero(ok), y[ok]) & _winable[ok]
    px = 100.0 * (y - _y_tr); out = {'policy': name, 'win rate': float(win[_tm].sum() / _tm.sum())}
    for lab, M_ in [('exit of record', _MREC), ('guard', _MGUARD), ('lab other side', _M)]:
        pv = np.where(win & _tm, px + M_, 0.0); pv[_tm & _winable & ~np.isfinite(M_)] = np.nan
        out[f'P&L per request, {lab} (bp)'] = float(np.nanmean(pv[_tm])); out[f'$ per month, {lab} ($k)'] = float(np.nansum(pv[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3)
    hed = np.where(win & _tm, px + _MREC + np.nan_to_num(_mmd5) - hedged_cost, 0.0); hed[_tm & _winable & ~(np.isfinite(_MREC) & np.isfinite(_mmd5))] = np.nan
    out['$ per month, hedged, exit of record ($k)'] = float(np.nansum(hed[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3)
    return out


_pol_rows = [bt_policy_rows_f('production (vanilla)', _opt_van)]
_pol_rows += [bt_policy_rows_f(f'14c formation: {k}', o) for k, o in BT_FM_OPTO.items()]
_pol_rows += [bt_policy_rows_f(k, o) for k, o in _rl_pol.items()]
for nm in [f'{b} | {d} | hedged-Sharpe pick' for d in GZ_DEFS for b in _gz_bases] + [f'{b} | {d} | k=1' for d in GZ_DEFS for b in ('vanilla',)]:
    _pol_rows.append(bt_policy_rows_f(f'risk-aware: {nm}', _pols_gz[nm], GZ_HEDGE_COST_BP if nm.split(' | ')[1] == 'hedged' else 0.0))
bt_policies_f = pd.DataFrame(_pol_rows).set_index('policy')
for lab in ['exit of record', 'guard', 'lab other side']:
    bt_policies_f[f'delta vs production, {lab} (bp)'] = bt_policies_f[f'P&L per request, {lab} (bp)'] - bt_policies_f.loc['production (vanilla)', f'P&L per request, {lab} (bp)']
print(f'The 14c formations, the 14e reallocations and the risk-aware policies on the test months {_test_m[0]}..{_test_m[-1]}, under the exit of record, the guard and the lab\'s other-side exit:')
display(bt_policies_f.round(4))
_hedge_gain = float(bt_policies_f.loc['production (vanilla)', '$ per month, hedged, exit of record ($k)'] - bt_policies_f.loc['production (vanilla)', '$ per month, exit of record ($k)'])
print(f'production\'s own fills under the exit of record: ${bt_policies_f.loc["production (vanilla)", "$ per month, exit of record ($k)"]:,.0f}k a month raw, '
      f'${bt_policies_f.loc["production (vanilla)", "$ per month, hedged, exit of record ($k)"]:,.0f}k hedged (MMD at tenor over {BT_H_RECORD} days): hedge contribution ${_hedge_gain:+,.0f}k a month')

fig, ax = plt.subplots(1, 3, figsize=(22, 6))
_top = [n for n in bt_board_f.index if n != 'vanilla (production replay)'][:8]; _xs = np.arange(len(_top))
for k_, (lab, col) in enumerate([('lab other side (14)', '#999999'), (EX_RECORD, '#000000'), (EX_GUARD, '#4C72B0')]):
    ax[0].bar(_xs + (k_ - 1) * 0.27, bt_board_f.loc[_top, f'delta | {lab} (bp)'], 0.27, color=col, label=lab)
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(_xs); ax[0].set_xticklabels(_top, rotation=25, ha='right', fontsize=8); ax[0].legend(fontsize=8); ax[0].set_title('Delta vs vanilla by exit, top variants (bp per request)', fontsize=10)
_hm = bt_board_f[[f'delta | {k} (bp)' for k in _report_exits]].drop(index='vanilla (production replay)', errors='ignore'); _vm = float(np.nanmax(np.abs(_hm.to_numpy(float)))) or 1.0
im = ax[1].imshow(_hm.to_numpy(float), cmap='RdBu', vmin=-_vm, vmax=_vm, aspect='auto'); ax[1].set_yticks(range(len(_hm))); ax[1].set_yticklabels(_hm.index, fontsize=7); ax[1].set_xticks(range(_hm.shape[1])); ax[1].set_xticklabels(list(_report_exits), rotation=30, ha='right', fontsize=7)
ax[1].set_title('Does the sign survive the exit? delta vs vanilla (blue = better)', fontsize=10); fig.colorbar(im, ax=ax[1], fraction=0.03)
_pp = bt_policies_f['delta vs production, exit of record (bp)'].drop(index='production (vanilla)'); ax[2].barh(_pp.index, _pp.to_numpy(float), color='#937860'); ax[2].axvline(0, color='k', lw=0.6)
ax[2].set_title('Formations, reallocations and risk-aware policies: delta vs production under the exit of record (bp per request)', fontsize=9); ax[2].tick_params(axis='y', labelsize=7)
savefig('14f_rescore')
record('lab_rfq_rescore', exit_of_record=EX_RECORD, guard=EX_GUARD, leaderboard=flat_records(bt_board_f), policies=flat_records(bt_policies_f), verdict=BT_F_VERDICT, hedge_contribution_k=_hedge_gain,
       seconds=time.perf_counter() - _rs_t0)
print(f'[20] {time.perf_counter() - _rs_t0:.0f}s')

# %% [markdown]
# ## 14g. The manual charge kept and tilted: can the model re-allocate what the traders charge, at the same win rate?
#
# **Why.** 14c and 14e [17] replaced the manual charge, and the 14e regression showed why that failed: the traders' per-request information is the
# call bucket, and both tests threw it away. The risk-aware charge of 14b and 14e [16] kept the manual charge but could only add to it, because a
# bid-only book is always long. This section keeps every request's manual charge and tilts it with a model quantity, then removes the level effect of
# the tilt with a constant width so that each month's win rate equals production's. What is left is pure re-allocation: the same number of wins,
# placed by the model's risk measure instead of only the traders' call-bucket view, scored on cover, book risk and markouts under the exit of record.
#
# **Tilts** $g_i$ (standardised on the earlier months), additive $c_i = \text{manual}_i + \lambda (g_i - \bar g)$ with $\lambda$ a multiple of
# $\text{sd(manual)}/\text{sd}(g)$, and multiplicative $c_i = \text{manual}_i \exp(\kappa z_i)$; liquidity and risk charges kept throughout:
#
# | tilt | $g_i$ | what it charges for |
# |---|---|---|
# | systematic vol | 5-day rate-plus-factor vol from the [14] Sigma and the [13] betas | the bond's factor and curve exposure |
# | idiosyncratic vol | 5-day residual vol ([14]) | bond-specific noise |
# | signed factor exposure vs the book | $h\,\beta_i' \Sigma_{ff} E^f_d$, the fill's factor covariance with production's own start-of-day book | two-sided: cheaper on a fill that offsets the book's factor exposure, dearer on one that adds to it |
# | peer co-movement | trailing 40-day correlation of the bond's residual with its 20 beta-space peers' median residual (the dependence tier the Section 12 copula models) | how much of the bond's residual is shared with its peers after the factors |
# | time to call | $1 / (0.25 + \text{years to call})$ on callable bonds, 0 on bullets | what the manual charge already prices, sharpened within the call bucket |
#
# **Level.** After the tilt, a constant width $w$ (bp) on the final yield is set by bisection so that the month's win rate on winnable requests equals
# production's, within 0.05 points. $w$ is in-sample by construction, as the 14c calibration was; the object is allocation, not level.
#
# | Cell | Content |
# |---|---|
# | [21] | the five tilts, the four settings, the requotes and the equal-win widths (cached stage `lab_rfq_tilts`) |
# | [22] | scoring at equal win rate: wins kept, cover, markout per fill under the exit of record (raw and hedged), ex-ante book risk, hedged dollars per dollar of risk; the bar; figures |

# %%
CELL_T('14g. Tilts on the manual charge, requoted and levelled to equal win rate [21]')
PRE_REGISTERED.update({
    'lab_rfq_tilts': ('fixed 2026-10-21 before scoring: charge = manual_i + liquidity + risk tilted by one of five model quantities, additively (lambda = m x sd(manual) / sd(g) over the '
                      'earlier months, m in {0.5, 1, 2}, g centred at its mean over the month\'s winnable requests) or multiplicatively (manual x exp(kappa z), z the standardised g, '
                      'kappa = 0.5); tilts: systematic vol, idiosyncratic vol ([13] betas, [14] Sigma), the signed factor covariance of the fill with production\'s own start-of-day '
                      'book, the trailing 40-day correlation of the bond\'s residual with its 20 beta-space peers\' median residual, and 1 / (0.25 + years to call) on callable bonds; '
                      'every setting goes through the production optimizer with the tilted charge, then a constant width on the final yield is set by bisection so that the month\'s '
                      'win rate on winnable requests equals production\'s within 0.05 points (in-sample level, as in 14c); scored on the test months at equal win rate: vanilla wins '
                      'kept, bp of cover saved on kept cover wins, crossing on new wins, markout per fill under the exit of record (raw and hedged by the MMD move at tenor), ex-ante '
                      'daily sd of the policy\'s own book ([13] exposures, [14] Sigma), gross DV01, hedged dollars per dollar of ex-ante sd; a tilt passes if cover saved on kept '
                      'cover wins >= -0.5 bp and either ex-ante sd <= 0.9 x production\'s or hedged markout per fill >= production\'s + 1 bp, in >= 4 of the 5 test months'),
})
_tg_t0 = time.perf_counter()
TG_MULTS, TG_KAPPA, TG_WIN_TOL, TG_W_RANGE, TG_ITERS, TG_CORR_WINDOW = (0.5, 1.0, 2.0), 0.5, 0.0005, 40.0, 16, 40

# ---- the five tilts
_tg_fac_book = np.full(len(BT), np.nan)                                     # signed: h x beta_i' Sigma_ff E^f_d on production's own book
for j in sorted(BT_SIGMA_C):
    r_ = _pb_rows_day.get(j)
    if r_ is None:
        continue
    S_ff = BT_SIGMA_C[j][_nT:, _nT:]; E_f = _bk_van_c['E'][j][_nT:]
    _tg_fac_book[r_] = BT_HOLD * (BT_BETA_GZ[r_] @ (S_ff @ E_f)) / 1e3           # $k per bp of the fill's own DV01-free beta, a scale the lambda absorbs
# peer co-movement: trailing correlation of the bond's residual with its peers' median residual, on the RV grid
_pm = np.full((RV_ND, RV_NB), np.nan, dtype=np.float32)
if 'RV_FORMS' in globals() and len(RV_FORMS):
    _form_days = np.array([F_['d0'] for F_ in RV_FORMS])
    for fi, F_ in enumerate(RV_FORMS):
        d_lo, d_hi = F_['d0'], (_form_days[fi + 1] if fi + 1 < len(_form_days) else RV_ND)
        ids, peers = F_['ids'], F_['peers']
        for d in range(d_lo, d_hi):
            _pm[d, ids] = np.nanmedian(_E[d][peers], axis=1)
_x, _y = _E, _pm.astype(float); _ok = np.isfinite(_x) & np.isfinite(_y); _xz, _yz = np.where(_ok, _x, 0.0), np.where(_ok, _y, 0.0)
_cs = lambda a: np.vstack([np.zeros((1, RV_NB)), np.cumsum(a, axis=0)])  # noqa: E731
_Sn, _Sx, _Sy, _Sxy, _Sxx, _Syy = _cs(_ok.astype(float)), _cs(_xz), _cs(_yz), _cs(_xz * _yz), _cs(_xz ** 2), _cs(_yz ** 2)
_W = TG_CORR_WINDOW; _hi = np.arange(1, RV_ND + 1); _lo = np.maximum(_hi - _W, 0)
_n = _Sn[_hi] - _Sn[_lo]; _sx = _Sx[_hi] - _Sx[_lo]; _sy = _Sy[_hi] - _Sy[_lo]; _sxy = _Sxy[_hi] - _Sxy[_lo]; _sxx = _Sxx[_hi] - _Sxx[_lo]; _syy = _Syy[_hi] - _Syy[_lo]
with np.errstate(invalid='ignore', divide='ignore'):
    _cov = _sxy - _sx * _sy / np.maximum(_n, 1); _vx = _sxx - _sx ** 2 / np.maximum(_n, 1); _vy = _syy - _sy ** 2 / np.maximum(_n, 1)
    _PCORR = np.where((_n >= 10) & (_vx > 0) & (_vy > 0), _cov / np.sqrt(_vx * _vy), np.nan)
_tg_peer = bt_last_obs(_PCORR, _d02, _bi2, _on2)[0]
_tg_ytc = ((_gz_call - pd.to_datetime(_gz_dates)).dt.days / 365.25).to_numpy(float)
_tg_call = np.where(_gz_is_call & np.isfinite(_tg_ytc) & (_tg_ytc > 0), 1.0 / (0.25 + np.nan_to_num(_tg_ytc, nan=1.0)), 0.0)
TG_TILTS = {'systematic vol': BT_VOL_C['rate + factor'], 'idiosyncratic vol': BT_VOL_C['idiosyncratic'], 'signed factor exposure vs book': _tg_fac_book, 'peer co-movement': _tg_peer, 'time to call': _tg_call}
print('tilt coverage on winnable requests (missing sits at the month mean, no tilt):', {k: round(float(np.isfinite(v[_winable]).mean()), 3) for k, v in TG_TILTS.items()})
print('tilt correlations with the manual charge (winnable requests, Spearman):',
      {k: round(float(pd.Series(v[_winable]).corr(pd.Series(_man[_winable]), method='spearman')), 3) for k, v in TG_TILTS.items()})
TG_SETTINGS = [('additive', m) for m in TG_MULTS] + [('multiplicative', TG_KAPPA)]


def tg_charge(tilt: str, kind: str, par: float, m: str) -> tuple[np.ndarray, np.ndarray]:
    """Winnable rows of month m and their tilted charge level (production's convention, negative = a cost)."""
    r_ = np.flatnonzero((_month == m) & _winable); g = TG_TILTS[tilt][r_]
    earlier = [x for x in BT_MONTHS if x < m] or [m]; e_ = np.isin(_month, earlier) & _winable & np.isfinite(TG_TILTS[tilt])
    mu_e, sd_e = float(np.nanmean(TG_TILTS[tilt][e_])), float(np.nanstd(TG_TILTS[tilt][e_])); sd_man = float(np.std(_man[e_]))
    gbar = float(np.nanmean(g)) if np.isfinite(g).any() else 0.0
    if kind == 'additive':
        lam = par * sd_man / sd_e if sd_e > 0 else 0.0; man_t = _man[r_] + lam * np.nan_to_num(g - gbar)
    else:
        z = np.where(np.isfinite(g), (g - mu_e) / (sd_e if sd_e > 0 else 1.0), 0.0); man_t = _man[r_] * np.exp(par * np.clip(z, -3, 3))
    return r_, -(np.maximum(man_t, 0.0) + _liq[r_] + _rsk[r_])


def tg_level(r_: np.ndarray, opto: np.ndarray, target: float) -> tuple[float, float]:
    """The constant width (bp) that brings the month's win rate to production's; returns (width, achieved win rate)."""
    def wr(w: float) -> float:
        return float(bt_win_fast(r_, opto + _adj[r_] + w / 100.0).mean())
    lo, hi = -TG_W_RANGE, TG_W_RANGE
    if wr(lo) < target:
        return lo, wr(lo)
    if wr(hi) > target:
        return hi, wr(hi)
    for _ in range(TG_ITERS):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if wr(mid) > target else (lo, mid)
    return hi, wr(hi)


def _bt_tilt_runs() -> pd.DataFrame:
    out, rows = {}, []
    for tilt in TG_TILTS:
        for kind, par in TG_SETTINGS:
            t0 = time.perf_counter(); o = np.full(len(BT), np.nan); nm = f'{tilt} | {kind} {par:g}'
            for m in _test_m:
                r_, c_ = tg_charge(tilt, kind, par, m); om = bt_requote(r_, c_); target = float(_wv0[r_].mean())
                w, got = tg_level(r_, om, target); o[r_] = om + w / 100.0
                rows.append({'tilt': tilt, 'setting': f'{kind} {par:g}', 'test month': m, 'width (bp)': w, 'production win rate': target, 'win rate': got,
                             'mean |charge - manual| (bp)': float(np.mean(np.abs(-c_ - _liq[r_] - _rsk[r_] - _man[r_])))})
            out[nm] = o; print(f'  {nm}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out).assign(_rows_json=json.dumps(rows, default=float))


BT_TG = cached('lab_rfq_tilts', _bt_tilt_runs, deps=[STAGE_KEYS['lab_rfq_replay'], float(np.nansum(BT_VOL_C['total'])), float(np.nansum(_tg_fac_book)), float(np.nansum(_tg_peer)), float(_tg_call.sum()),
                                                      _test_m, TG_MULTS, TG_KAPPA, TG_W_RANGE, TG_ITERS, float(_man.sum())],
               code=[tg_charge, tg_level, bt_requote, bt_replay, bt_optimize, bt_opto, bt_win_fast])
bt_tg_level = pd.DataFrame(json.loads(BT_TG['_rows_json'].iloc[0])).set_index(['tilt', 'setting', 'test month'])
BT_TG_OPTO = {c: BT_TG[c].to_numpy(float) for c in BT_TG.columns if c != '_rows_json'}   # note: the width is folded into these optimal yields
print('Equal-win levelling: the constant width that returns each month to production\'s win rate after the tilt (positive = cheaper), and the size of the tilt:')
display(bt_tg_level.round(4))
record('lab_rfq_tilts', levels=flat_records(bt_tg_level.reset_index()), tilt_coverage={k: float(np.isfinite(v[_winable]).mean()) for k, v in TG_TILTS.items()}, seconds=time.perf_counter() - _tg_t0)
print(f'[21] {time.perf_counter() - _tg_t0:.0f}s')

# %%
CELL_T('14g. Scoring at equal win rate: cover, risk, markouts under the exit of record [22]')
_ts_t0 = time.perf_counter()
TG_COVER_BP, TG_SD_RATIO, TG_MARKOUT_BP, TG_MONTHS = -0.5, 0.9, 1.0, 4
_hed_rec = _MREC + np.nan_to_num(_mmd5)                                      # hedged markout under the exit of record: + the MMD move at tenor over the holding
_hed_rec[~(np.isfinite(_MREC) & np.isfinite(_mmd5))] = np.nan


def tg_score(name: str, opto_final: np.ndarray, is_vanilla: bool = False) -> tuple[dict, dict]:
    """A policy at (near) production's win rate on the test months: wins, cover, markouts under the exit of record, the book; plus the per-month pieces the bar needs."""
    y = (_y_van if is_vanilla else opto_final + _adj); win = np.zeros(len(BT), bool); ok = np.isfinite(y) & _tmw; win[ok] = bt_win_fast(np.flatnonzero(ok), y[ok])
    keep = _cov_w & win; d_v = 100.0 * (_thr - y); d_0 = 100.0 * (_thr - _y_van); newl = win & ~_wv0 & (_grp == 'lost, printed') & _tm; px = 100.0 * (y - _y_tr)
    pv = np.where(win & _tm, px + _MREC, 0.0); pv[_tm & _winable & ~np.isfinite(_MREC)] = np.nan
    hv = np.where(win & _tm, px + _hed_rec, 0.0); hv[_tm & _winable & ~np.isfinite(_hed_rec)] = np.nan
    bk = bt_book_g(win, BT_EXPO_GZ, BT_V_C, BT_SIGMA_C); sdv = bk['sd'][_tdays]; sdv = sdv[np.isfinite(sdv)]
    f = win & _tm & np.isfinite(_MREC)
    res = {'policy': name, 'win rate': float(win[_tm].sum() / _tm.sum()), 'vanilla wins kept': float((win & _wv0 & _tm).sum() / max(int((_wv0 & _tm).sum()), 1)), 'new wins': int((win & ~_wv0 & _tm).sum()),
           'bp left on kept cover wins': float(d_v[keep].mean()) if keep.any() else np.nan, 'bp saved on kept cover wins': float((d_0 - d_v)[keep].mean()) if keep.any() else np.nan,
           'crossing on new wins (bp)': float(-d_v[newl].mean()) if newl.any() else np.nan, 'markout per fill, exit of record (bp)': float(np.nanmean((px + _MREC)[f])),
           'hedged markout per fill (bp)': float(np.nanmean((px + _hed_rec)[f & np.isfinite(_hed_rec)])), '$ per month, exit of record ($k)': float(np.nansum(pv[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3),
           '$ per month, hedged ($k)': float(np.nansum(hv[_tm] * _pb_u[_tm]) / len(_test_m) / 1e3), 'ex-ante daily sd, mean ($k)': float(sdv.mean() / 1e3) if len(sdv) else np.nan,
           'gross DV01, mean ($k/bp)': float(np.nanmean(bk['dv01'][_tdays]) / 1e3), **{f'|factor {k + 1}| ($k)': float(np.abs(bk['E'][_tdays][:, _nT + k]).mean() / 1e3) for k in range(3)}}
    res['hedged $ per $ of ex-ante sd'] = res['$ per month, hedged ($k)'] / res['ex-ante daily sd, mean ($k)'] if res['ex-ante daily sd, mean ($k)'] else np.nan
    monthly = {}
    for m in _test_m:
        mm = _tm & (_month == m); km = keep & mm; fm = f & mm; dm = _tdays & (_pb_month == m); s_ = bk['sd'][dm]; s_ = s_[np.isfinite(s_)]
        monthly[m] = {'cover saved': float((d_0 - d_v)[km].mean()) if km.any() else np.nan, 'hedged markout': float(np.nanmean((px + _hed_rec)[fm & np.isfinite(_hed_rec)])) if fm.any() else np.nan,
                      'sd': float(s_.mean()) if len(s_) else np.nan}
    return res, monthly


_tg_rows, _tg_monthly = [], {}
r0, m0 = tg_score('production (vanilla)', _opt_van, is_vanilla=True); _tg_rows.append(r0); _tg_monthly['production (vanilla)'] = m0
for nm, o in BT_TG_OPTO.items():
    r_, mm_ = tg_score(nm, o); _tg_rows.append(r_); _tg_monthly[nm] = mm_
bt_tg = pd.DataFrame(_tg_rows).set_index('policy')
_pass_rows = {}
for nm, mm_ in _tg_monthly.items():
    n_ok = 0
    for m in _test_m:
        a, b = mm_[m], m0[m]
        c1 = np.isfinite(a['cover saved']) and a['cover saved'] >= TG_COVER_BP
        c2 = (np.isfinite(a['sd']) and np.isfinite(b['sd']) and a['sd'] <= TG_SD_RATIO * b['sd']) or (np.isfinite(a['hedged markout']) and np.isfinite(b['hedged markout']) and a['hedged markout'] >= b['hedged markout'] + TG_MARKOUT_BP)
        n_ok += int(c1 and c2)
    _pass_rows[nm] = n_ok
bt_tg['months passing'] = pd.Series(_pass_rows); bt_tg['passes'] = (bt_tg['months passing'] >= min(TG_MONTHS, len(_test_m))) & (bt_tg.index != 'production (vanilla)')
bt_tg['ex-ante sd vs production'] = bt_tg['ex-ante daily sd, mean ($k)'] / r0['ex-ante daily sd, mean ($k)'] - 1.0
bt_tg['hedged markout vs production (bp)'] = bt_tg['hedged markout per fill (bp)'] - r0['hedged markout per fill (bp)']
_show_tg = ['win rate', 'vanilla wins kept', 'new wins', 'bp saved on kept cover wins', 'crossing on new wins (bp)', 'markout per fill, exit of record (bp)', 'hedged markout per fill (bp)', 'hedged markout vs production (bp)',
            '$ per month, exit of record ($k)', '$ per month, hedged ($k)', 'ex-ante daily sd, mean ($k)', 'ex-ante sd vs production', 'gross DV01, mean ($k/bp)', 'hedged $ per $ of ex-ante sd', 'months passing', 'passes']
print(f'TILTS ON THE MANUAL CHARGE at production\'s win rate, test months {_test_m[0]}..{_test_m[-1]}, exit of record = {EX_RECORD} ({GZ_LABEL} betas, {SIGMA_LABEL} Sigma):')
display(bt_tg[_show_tg].round(4))
_best_tg = bt_tg[bt_tg['passes']].sort_values('hedged $ per $ of ex-ante sd', ascending=False)
BT_TG_VERDICT = f'{int(bt_tg["passes"].sum())} of {len(bt_tg) - 1} tilt settings pass the bar' + (f'; best by hedged $ per $ of risk: {_best_tg.index[0]}' if len(_best_tg) else '')
print('VERDICT (lab_rfq_tilts):', BT_TG_VERDICT)
# where a tilt moves the charge: by call bucket and by tenor, the mean charge minus manual for the m = 1 additive settings
_cb_ = BT['call_bucket'].astype(str).to_numpy() if 'call_bucket' in BT else np.full(len(BT), 'NA')
_mv_rows = []
for tilt in TG_TILTS:
    for m in _test_m:
        r_, c_ = tg_charge(tilt, 'additive', 1.0, m); d_ = (-c_ - _liq[r_] - _rsk[r_]) - _man[r_]
        for cell in np.unique(_cb_[r_]):
            s_ = _cb_[r_] == cell; _mv_rows.append({'tilt': tilt, 'call bucket': cell, 'month': m, 'charge - manual (bp)': float(d_[s_].mean()), 'requests': int(s_.sum())})
bt_tg_move = pd.DataFrame(_mv_rows).groupby(['tilt', 'call bucket'])['charge - manual (bp)'].mean().unstack('tilt')
print('Where the additive m = 1 tilts move the charge against the traders (mean charge minus manual, bp, by call bucket):'); display(bt_tg_move.round(2))

fig, ax = plt.subplots(1, 3, figsize=(22, 6))
_cols_t = dict(zip(TG_TILTS, ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860']))
for nm, r_ in bt_tg.drop(index='production (vanilla)').iterrows():
    tilt, setting = nm.split(' | '); mk = 'o' if setting.startswith('additive') else 's'
    ax[0].scatter(r_['ex-ante daily sd, mean ($k)'], r_['bp saved on kept cover wins'], marker=mk, color=_cols_t[tilt], s=40, label=f'{tilt}' if setting == 'additive 1' else None)
    ax[1].scatter(r_['ex-ante daily sd, mean ($k)'], r_['hedged markout per fill (bp)'], marker=mk, color=_cols_t[tilt], s=40, label=f'{tilt}' if setting == 'additive 1' else None)
for a_, yv_, lab in [(ax[0], r0['bp saved on kept cover wins'], 'cover saved on kept cover wins (bp)'), (ax[1], r0['hedged markout per fill (bp)'], 'hedged markout per fill, exit of record (bp)')]:
    a_.scatter(r0['ex-ante daily sd, mean ($k)'], yv_, marker='*', s=220, color='#000000', zorder=4, label='production'); a_.axvline(TG_SD_RATIO * r0['ex-ante daily sd, mean ($k)'], color='#999999', ls='--', lw=0.8)
    a_.set_xlabel(r'book ex-ante daily sd, mean (\$k)'); a_.set_ylabel(lab); a_.legend(fontsize=7)
ax[0].set_title('Equal win rate: cover against book risk (circles additive, squares multiplicative; dashed = 0.9 x production)', fontsize=9); ax[1].set_title('Equal win rate: hedged markout against book risk', fontsize=10)
_hr = bt_tg['hedged $ per $ of ex-ante sd']; ax[2].barh(_hr.index, _hr.to_numpy(float), color=['#000000' if i == 'production (vanilla)' else ('#55A868' if p else '#999999') for i, p in zip(_hr.index, bt_tg['passes'])])
ax[2].set_title(r'Hedged \$ per \$ of ex-ante risk (green = passes the bar)', fontsize=10); ax[2].tick_params(axis='y', labelsize=7)
savefig('14g_tilts')
record('lab_rfq_tilts', table=flat_records(bt_tg), verdict=BT_TG_VERDICT, moves=flat_records(bt_tg_move.reset_index()), seconds=time.perf_counter() - _ts_t0)
print(f'[22] {time.perf_counter() - _ts_t0:.0f}s')

# %% [markdown]
# ## 15. Registry and runtimes
#
# Writes `artifacts_ipca_lab/results_registry.json` and the per-cell runtime table.

# %%
CELL_T('15. Registry and runtimes [1]')
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
