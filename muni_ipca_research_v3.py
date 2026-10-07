# %% [markdown]
# # Muni IPCA v3 — Factor Model of Record, State-Space Residual Signal, Algo Error Correction
#
# Clean rebuild after the v5 review. The validated spine is kept (point-in-time IPCA on 13 instruments, aligned
# Gamma versions, factor-mimicking weights, beta-space exhibits, the state-space description of the residual),
# the closed experiments are dropped (AR(1) and ARMA signals, mark shrinkage, residual-driven width, quote skew
# sizing, peer RV as a signal, factor hedging of flow, K sensitivity), and four new blocks carry the programme
# forward:
#
# 1. **Two data rules** that the v5 run showed were missing: a *dispersion-day* rule (evaluator-wide reprices are
#    days where many bonds move a lot *relative to the day's median move*, not days where the whole curve moves)
#    and a *partial-mark-day* rule (dates with far fewer bonds than usual are dropped, not just dates under a fixed floor).
# 2. **Refit cadence and regime-conditional Gamma**: the Gamma swap test said a Gamma frozen two months earlier
#    beat the weekly refit out of sample. Weekly, monthly, quarterly, frozen and a dispersion-regime-conditional
#    Gamma are run side by side and the residuals of record come from the pre-registered cadence.
# 3. **The pooled state-space filter as the residual signal of record**: AR drift plus mark-noise MA, with the
#    filtered drift as the signal and the filtered mark noise as the mark-quality score (the by-bucket filter was
#    closed in v31 and the LongConv ridge in v41).
# 4. **Algo error correction**: the v5 print-to-print panel found the algo's own pricing error persists 58% from
#    one print to the next. The algo quote is MMD yield + side-specific MMD spread + a last-value adjustment, so
#    that persistence says the existing adjustment is under-sized. It is turned into point-in-time features (last
#    matched print's algo error, its age, an exponentially weighted history **on the same side of the market**),
#    a diagnostic of the algo's own last-value term, the same-side memory (the deployable rule, v41), a per-side
#    state-space memory as its generalisation, the side-pooled EWMA as the reference, and a level model with and
#    without the error features. The band around the chosen mid is calibrated by conformal scaling, split by
#    month and rolling over the last ten trading days, with a lagged market-dispersion feature.
# 5. **Where the gains live**: every result is cut by the model's own axes (duration, call structure, rating,
#    state, liquidity, factor-beta terciles, point-in-time beta-space clusters). The aim is not to improve the
#    whole universe but to find the characteristic cells and factor exposures where the improvement concentrates,
#    and to ask whether the algo's error loads on the factor betas at all.
#
# **Inputs** (from `data_pipeline/muni_data_pipeline.py`, under `./data_pipeline/`): `research_panel_step3_yield.parquet`,
# `data/closing_marks/*.parquet`, `data/algosignal_msrb/*.parquet`. **Outputs** go to `./artifacts_v3/`.
#
# **Sections**
# 1. Setup and configuration (with the pre-registered choices)
# 2. Data load and QA; the two data rules
# 3. Model panel: 13 instruments, rank-normalised per date
# 4. IPCA: date-weighted ALS, K sweep, Gamma anatomy
# 5. Walk-forward residuals: refit cadence study, regime-conditional Gamma, residuals of record
# 6. Factor-mimicking weights, leverage, beta-space map and clusters
# 7. Residual dynamics: ACF, activity buckets, the state-space filter (signal of record), two-regime check
# 8. Hedged paper portfolio (vol-scaled) as the information metric
# 9. Transaction validation of the signal of record (Fama-MacBeth with block bootstrap); 9b trade size and side
# 10. Algo error correction: the same-side memory, the interaction ladder (rho conditioned on factor state, cluster, residual drift, age, size, industry) with its pre-registered bar, the direction-aware view, the level model
# 11. Conformal band around the chosen mid: split (monthly) and rolling (10-day) calibration
# 12. Where the gains live: breakdown by characteristics, factor betas and beta-space clusters; 12b what they are worth in dollars
# 13. Systematic path: factor roll-forward of stale marks, by segment
# 14. Results registry and summary
# 15. Robustness: dispersion-day exclusions, bootstrap inference, pre-registered choices
#
# **v3.8 housekeeping.** The closed mids are gone (single-print rule, side intercepts, universe and local factor
# loadings, universe-offset state-space memory, cluster pooling, within-cluster relative value, LongConv ridge,
# by-bucket filter); their numbers live in the paper's closed-experiments appendix. The expensive stages
# (walk-forward IPCA, the pooled state-space fit, the standardised print history with its memories, the per-side
# state-space memory, the level models, the band models) run through `cached(...)`: each result is stored under
# `artifacts_v3/cache/` with a key hashed from the configuration, the data fingerprints and the source code of
# the stage, so a re-run with the same inputs and code reuses it automatically and any change to either recomputes it.
# Nothing is toggled by hand; stale cache files for a stage are removed when a new key is written.
#
# **v3.9 additions.** Two characteristics the desk works with and the model had overlooked: *trade size* (decayed
# mean of MSRB print size and print frequency, strictly before the date, with a no-print flag) and *issuer industry*
# (from the trade-track static table; dummies for the largest industries). Both enter the IPCA instrument set, the
# EDA (Section 9b: size x side, with the desk's quantity bins) and the Section 12 breakdowns. The algo correction
# gets an *interaction ladder*: the same-side memory with its persistence coefficient conditioned on the factor
# betas, the beta-space cluster, the filtered residual drift, the same-side age, the trade size, side x cluster and
# industry, judged against a pre-registered bar (>= 0.10 bp over S, bootstrap t > 3, positive in 4 of 5 months), and a
# *direction-aware* view (signed decomposition, crossing rate by side, pinball loss) beside the absolute gains.
#
# Every leakage boundary is explicit: Gamma is refit on past dates only, instruments are lagged and rank-normalised
# within date, every trade is matched to a residual dated strictly before the trade date, every print-derived
# feature is cut at the algo signal time, and every calibration uses the month before the test month.

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
    seed: int = 20260921


# Pre-registered choices, fixed on 2026-10-07 after the v5 review and before this notebook was run. Section 14
# prints them so a reader can tell a choice from a fit.
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
        CACHE_LOG.append({'stage': name, 'status': 'reused', 'seconds': time.perf_counter() - t0, 'file': p.name}); return obj
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
    CACHE_LOG.append({'stage': name, 'status': 'computed', 'seconds': dt, 'file': p.name}); return obj


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
# --- v3.9 new characteristics: trade size from MSRB prints, issuer industry from the trade track ---
# Trade size is a bond characteristic here, not a trade feature: the time-decayed mean of log quantity over the
# bond's MSRB prints strictly before the date (half-life trade_size_halflife_days), with the decayed print count as
# a frequency measure and a flag for bonds with no print history. Industry comes from muni_algo_trade_track via the
# pipeline's static source (`track_static`); when the source is absent the industry instruments are skipped.
t0 = time.perf_counter()
msrb_store = PIPE.store('msrb')
msrb_raw = msrb_store.read(columns=['date', 'cusip', 'quantity', 'tradetime', 'tradetype', 'side'], categorical=False)
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
try:
    emb = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=CFG.seed).fit_transform(Bs[idx]); emb_name = 't-SNE'
except Exception as exc:  # noqa: BLE001
    emb = PCA(n_components=2, random_state=CFG.seed).fit_transform(Bs[idx]); emb_name = 'PCA'
    print('t-SNE unavailable, using PCA:', exc)
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
# ## 8. Paper portfolio: is the residual path harvestable?
#
# Every forecaster from Section 7 is run as a signal. Forecasts are divided by the bond's trailing vol, cross-sectionally
# demeaned, scaled to unit gross, and projected off the beta span with the day's $B_t$, so each portfolio has zero
# factor exposure. A walk-forward selector trades, each month, the signal with the best
# hedged Sharpe over the prior months, so the headline is not an in-sample pick.
# P&L is realised on the next day's yield change (bp of yield captured per unit gross). Sharpe is annualised with
# $\sqrt{252}$. This is the number a stat-arb paper would report; a market maker reads it as "how much skew
# information is in the residual" rather than as a tradable strategy.

# %%
nxt = resid[['cusip', 'date', 'target_bp']].rename(columns={'date': 'next_date', 'target_bp': 'next_target_bp'})


def prep_portfolio_frame(pred: pd.DataFrame) -> pd.DataFrame:
    pp = pred.merge(resid[['cusip', 'date', 'target_abs_vol'] + beta_cols], on=['cusip', 'date'], how='left')
    return pp.merge(nxt, on=['cusip', 'next_date'], how='left').dropna(subset=['next_target_bp'] + beta_cols)


def portfolio_pnl(frame: pd.DataFrame, hedge: bool = True, signal_col: str = 'yhat', scaling: str | None = None) -> pd.Series:
    """scaling: 'raw' uses the forecast in bp as the weight (concentrates gross in volatile names and takes cross-bucket
    bets); 'vol' divides by the bond's trailing vol (a z-score); 'rank' uses the within-date rank. Default from config."""
    scaling = CFG.portfolio_scaling if scaling is None else scaling
    out = {}
    for d, g in frame.groupby('date', observed=True):
        if len(g) < 50:
            continue
        s_ = g[signal_col].to_numpy(float)
        if scaling == 'vol':
            v = g['target_abs_vol'].to_numpy(float); v = np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 1.0); s_ = s_ / np.maximum(v, 0.5)
        elif scaling == 'rank':
            s_ = pd.Series(s_).rank(pct=True).to_numpy() - 0.5
        w = s_ - s_.mean()
        if hedge:
            B = g[beta_cols].to_numpy(float); w = w - B @ np.linalg.solve(B.T @ B + 1e-8 * np.eye(B.shape[1]), B.T @ w)
        gross = np.abs(w).sum()
        if gross <= 0:
            continue
        out[d] = float((w / gross) @ g['next_target_bp'].to_numpy(float))
    return pd.Series(out).sort_index()


def sharpe(x: pd.Series) -> float:
    return float(np.sqrt(252) * x.mean() / x.std()) if len(x) > 2 and x.std() > 0 else np.nan


# Candidate signals: every forecaster from Section 7 (the trailing mean and the pooled state-space filter)
signals = {k: prep_portfolio_frame(v) for k, v in fc_preds.items() if not v.empty}

pnl_by_signal = {k: portfolio_pnl(v) for k, v in signals.items()}
universes = ['ALL'] + sorted(b for b in resid['activity_bucket'].unique() if b != 'UNKNOWN')
rows = []
for k, v in signals.items():
    row = {'forecaster': k, 'ALL': sharpe(pnl_by_signal[k]), 'ALL unhedged': sharpe(portfolio_pnl(v, hedge=False)), 'mean bp/day': pnl_by_signal[k].mean(), 'days': len(pnl_by_signal[k]),
           'ALL ex-UNKNOWN': sharpe(portfolio_pnl(v[v['activity_bucket'] != 'UNKNOWN'])), 'ALL raw-scaled': sharpe(portfolio_pnl(v, scaling='raw')), 'ALL rank-scaled': sharpe(portfolio_pnl(v, scaling='rank'))}
    for b in universes[1:]:
        row[b] = sharpe(portfolio_pnl(v[v['activity_bucket'] == b]))
    rows.append(row)
paper = pd.DataFrame(rows).set_index('forecaster')
print(f'Hedged Sharpe by forecaster and universe (weights {CFG.portfolio_scaling}-scaled; raw and rank scalings and the ex-UNKNOWN universe shown for the ALL column):'); display(paper.round(2))

# Walk-forward selection: each month, trade the forecaster with the best hedged Sharpe over the prior months.
pnl_df = pd.DataFrame(pnl_by_signal).dropna(how='all')
pnl_df['month'] = pnl_df.index.to_period('M')
sel_parts, choices = [], {}
for m in sorted(pnl_df['month'].unique())[1:]:
    prior = pnl_df[pnl_df['month'] < m].drop(columns='month')
    if len(prior) < 15:
        continue
    pick = prior.apply(sharpe).idxmax(); choices[str(m)] = pick
    sel_parts.append(pnl_df.loc[pnl_df['month'] == m, pick].rename('selected'))
pnl_selected = pd.concat(sel_parts) if sel_parts else pd.Series(dtype=float)
print('walk-forward selection by prior-month Sharpe:', choices)
print(f'selected-signal hedged Sharpe: {sharpe(pnl_selected):.2f} over {len(pnl_selected)} days')

fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
for k, v in pnl_by_signal.items():
    v.cumsum().plot(ax=ax[0], label=f'{k} ({sharpe(v):.2f})', lw=1.2)
if len(pnl_selected):
    pnl_selected.cumsum().plot(ax=ax[0], color='k', lw=2, label=f'walk-forward selected ({sharpe(pnl_selected):.2f})')
ax[0].legend(fontsize=7); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('Hedged paper portfolio: cumulative bp captured per unit gross, by signal')
paper[universes[1:]].T.plot.bar(ax=ax[1]); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('Hedged Sharpe by activity bucket and signal'); ax[1].legend(fontsize=7); ax[1].set_xlabel('')
savefig('08_paper_portfolio')
best_portfolio = paper['ALL'].idxmax()
record('paper_portfolio', table=paper.round(4).reset_index().to_dict(orient='records'), walk_forward_choices=choices, selected_sharpe=float(sharpe(pnl_selected)) if len(pnl_selected) else None, best_by_all=best_portfolio)

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
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.4))
    for a, (col, title) in zip(ax[:2], [('ssm_rank', 'state-space signal rank'), ('level_rank', f'trailing-{CFG.level_signal_window} mean rank')]):
        dd = tv.dropna(subset=[col]).copy(); dd['decile'] = pd.qcut(dd[col].rank(method='first'), 10, labels=False) + 1
        for sd, g in dd.groupby('side'):
            if len(g) > 2000:
                a.plot(g.groupby('decile')['e_mark_bp'].mean(), marker='o', label={'S': 'S: dealer sale (customer buys)', 'P': 'P: dealer purchase (customer sells)', 'D': 'D: inter-dealer'}.get(sd, sd))
        a.plot(dd.groupby('decile')['e_mark_bp'].mean(), color='k', lw=2, label='all'); a.axhline(0, color='k', lw=0.6); a.set_xlabel('score decile (1 = forecast cheapening low -> 10 high)'); a.set_ylabel('mean trade - prior mark (bp)'); a.set_title(f'Trade vs prior mark by {title}', fontsize=10); a.legend(fontsize=7)
    rec_order = ['<=1d', '1-3d', '3-7d', '7-21d', '>21d']
    rec = tx[tx['target'] == 'trade - prior mark, by recency'].set_index('bucket').reindex(rec_order)
    ax[2].bar(rec.index, rec['fm_beta'], yerr=1.96 * (rec['fm_beta'] / rec['fm_t']).abs(), capsize=4, color='#4C72B0'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('FM beta of trade - mark on the record signal, by recency (bp/rank)', fontsize=10)
    savefig('09_transaction_validation')
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
# ## 10. Algo error correction
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

# %%
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
    parts, rho_rows = [], []
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
        lp = lvl_pred.reindex(te['_id'])
        te['e_D_bp'] = te['spread_bp'] - lp['yD'].to_numpy(); te['e_Dminus_bp'] = te['spread_bp'] - lp['yDm'].to_numpy(); te['y_mid_D'] = te['MmdYld'] + lp['yD'].to_numpy() / 100.0
        rho_rows.append(row); parts.append(te)
        print(f'  {m}: train {len(tr):,} | test {len(te):,} | rho pooled {rho_ewm:+.2f} | rho same-side {rho_s:+.2f}' + (f' | rho_K {row["rho_K"]:+.2f}' if 'rho_K' in row else ''))
    ecp = pd.concat(parts, ignore_index=True); rho_path = pd.DataFrame(rho_rows).set_index('month')
    for c in ['e_D_bp', 'e_Dminus_bp']:
        ecp[c] = ecp[c].fillna(ecp['e_algo_bp'])
    print('Error-correction coefficients and state-space parameters by month (estimated on prior months):'); display(rho_path.round(3))
    PREDS = {'A algo quote': 'e_algo_bp', 'C side-pooled EWMA (reference)': 'e_C_bp', 'S same-side EWMA': 'e_S_bp',
             'Sb same-side EWMA, rho by factor betas': 'e_Sb_bp', 'Sc same-side EWMA, rho by beta-space cluster': 'e_Sc_bp', 'Sm same-side EWMA, rho by |residual drift|': 'e_Sm_bp',
             'Sa same-side EWMA, rho by same-side age': 'e_Sa_bp', 'Sq same-side EWMA, rho by trade size': 'e_Sq_bp', 'Ssc same-side EWMA, rho by side x cluster': 'e_Ssc_bp', 'Si same-side EWMA, rho by industry': 'e_Si_bp',
             'K per-side state-space memory': 'e_K_bp', 'D level model with error features': 'e_D_bp', 'D- level model without error features': 'e_Dminus_bp'}
    PREDS = {k: v for k, v in PREDS.items() if v in ecp.columns}

    def ec_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size()})
        for k_, c in PREDS.items():
            out[f'MAE {k_.split()[0]}'] = g[c].apply(lambda s_: s_.abs().mean())
        for k_, c in list(PREDS.items())[1:]:
            d = frame.assign(gain=frame['e_algo_bp'].abs() - frame[c].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
            out[f'gain {k_.split()[0]} (bp)'] = d.mean(); out[f't {k_.split()[0]}'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
        return out

    ec_all = ec_table(ecp.assign(all='ALL'), 'all'); ec_age = ec_table(ecp, 'age_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d', 'no prior print']).dropna(how='all'); ec_side = ec_table(ecp, 'side'); ec_month = ec_table(ecp, 'month'); ec_band = ec_table(ecp, 'dur_band')
    print('Held-out MAE against the print (bp) and daily-FM gain over the algo quote:'); display(ec_all.T.round(3)); display(ec_age.round(3)); display(ec_side.round(3)); display(ec_band.round(3))
    _gd = {k_: ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp[c].abs()).groupby('trade_date')['gain'].mean().to_numpy() for k_, c in list(PREDS.items())[1:]}
    boot_gain = pd.Series({k_: block_bootstrap_t(v, CFG.block_days, CFG.n_boot, CFG.seed) for k_, v in _gd.items()}, name='bootstrap t of the daily gain'); display(boot_gain.round(2).to_frame())
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
    _sh.plot.barh(ax=ax[0], stacked=True, color=['#2E8B57', '#9ACD32', '#E9967A', '#C44E52']); ax[0].set_title('Where the mid moves relative to the print (share of trades)', fontsize=10); ax[0].legend(fontsize=7, loc='lower right'); ax[0].tick_params(axis='y', labelsize=7); ax[0].invert_yaxis()
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
    ec_age[[f'MAE {k_.split()[0]}' for k_ in PREDS]].plot.bar(ax=ax[0]); ax[0].set_title('MAE vs the print by age of the last algo error (bp)', fontsize=10); ax[0].set_xlabel(''); ax[0].legend([k_ for k_ in PREDS], fontsize=7)
    dd = ecp[ecp['has_side_err']].copy(); dd['decile'] = pd.qcut(dd['ewm_side_err_bp'].rank(method='first'), 10, labels=False) + 1
    for sd, g in dd.groupby('side'):
        ax[1].plot(g.groupby('decile')['e_algo_bp'].mean(), marker='o', label={'S': 'S', 'P': 'P', 'D': 'D'}.get(sd, sd))
    ax[1].plot(dd.groupby('decile')['e_algo_bp'].mean(), color='k', lw=2, label='all'); ax[1].plot(dd.groupby('decile')['ewm_side_err_bp'].mean(), color='grey', ls='--', label='same-side memory itself'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('decile of the same-side error memory'); ax[1].set_ylabel('mean algo error at this print (bp)'); ax[1].legend(fontsize=7); ax[1].set_title('Does the algo repeat its same-side error?', fontsize=10)
    rho_path[[c for c in ['rho_pooled', 'rho_same_side', 'rho_K'] if c in rho_path.columns]].plot(ax=ax[2], marker='o'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Error-correction coefficients by month (estimated on prior months)', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(fontsize=8)
    savefig('10_algo_error_correction')
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    ec_month[[f'MAE {k_.split()[0]}' for k_ in PREDS]].plot(ax=ax[0], marker='o'); ax[0].set_title('Held-out MAE by month (bp)', fontsize=10); ax[0].legend([k_ for k_ in PREDS], fontsize=7); ax[0].set_xlabel('')
    if len(imp_mean):
        imp_mean.sort_values().tail(15).plot.barh(ax=ax[1], color='#4C72B0'); ax[1].set_title('Level model D: mean feature importance (error features in the list)', fontsize=10)
    else:
        ax[1].text(0.1, 0.5, 'feature importance needs LightGBM', fontsize=12); ax[1].axis('off')
    savefig('10_algo_error_correction_monthly')
    # side x duration: does the same-side memory repair the short end on every side?
    _dur_order = ['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']
    ecp['dur_bucket'] = pd.cut(ecp['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=_dur_order).astype(str)
    show_mids = [k_ for k_ in PREDS if k_.split()[0] in ('C', 'S', 'K', 'D')]
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
# ## 11. Conformal band around the chosen mid: split and rolling calibration
#
# v2.3 fitted a quantile model of |print − mid| and got 71% coverage at an 80% target: the quantile model is
# mis-calibrated under the month-to-month shift in error scale. **Split conformal** fixes this without changing
# the model. For test month $m$, the quantile model is trained on months before $m-1$, the calibration month
# $m-1$ gives the scaling constant $s$ = the target quantile of the normalised score $|e| / \hat q$, and the band
# in month $m$ is $s\,\hat q$. Coverage is then guaranteed on average under exchangeability, and the width still
# adapts to the trade. Three bands at the same target: fixed (calibration-month quantile of $|e|$), raw quantile
# model, and conformalised quantile model, each around three mids: the algo quote, the same-side memory S and
# the level model D. The v31 run showed the monthly split-conformal scale cannot follow a regime shift inside the
# test month (September covered 64%), so a **rolling** variant is added: the scale for each test date is the target
# quantile of the normalised scores over the previous ten trading days, and the quantile model gets the lagged
# cross-sectional dispersion of the market as a feature. The fair benchmark for any conditional band is a fixed
# width rescaled on the same window ("rolling fixed"); the efficiency table reports half-width per point of coverage.
# The quantile models are cached on the error-correction key.

# %%
if HAS_TRADES and 'ecp' in globals() and not ecp.empty:
    ecp['market_disp_lag'] = ecp['trade_date'].map(DATE_DISP_LAG)          # cross-sectional sd of the target on the previous panel date: known at the open
    ecp['market_disp_lag'] = ecp['market_disp_lag'].fillna(ecp['market_disp_lag'].median())
    BAND_FEATURES = [c for c in ['log_size', 'side_D', 'side_P', 'side_S', 'recency_days_c', 'prints_20d', 'MinuteFromSignal', 'last_print_spread_bp', 'last_algo_err_bp', 'ewm_algo_err_bp', 'abs_resid_z', 'eta_abs', 'modified_duration_lag1', 'rating_score', 'target_abs_vol', 'dMmdSprdSide', 'market_disp_lag'] if c in ecp.columns]
    BAND_FEATURES = BAND_FEATURES + [c for c in ['ewm_side_err_bp', 'n_side_prints'] if c in ecp.columns and c not in BAND_FEATURES]
    months_b = sorted(ecp['month'].unique())

    def _band_models() -> pd.DataFrame:
        bands, bimp = [], []
        for mid_name, err_col in [('algo quote', 'e_algo_bp'), ('same-side memory S', 'e_S_bp'), ('level model D', 'e_D_bp')]:
            for i in range(2, len(months_b)):
                m, cal_m = months_b[i], months_b[i - CFG.conformal_calibration_months]
                tr = ecp[ecp['month'] < cal_m]; cal = ecp[ecp['month'] == cal_m]; te = ecp[ecp['month'] == m].copy()
                if len(tr) < 5_000 or cal.empty or te.empty:
                    continue
                feats = usable_features(tr, BAND_FEATURES); med = tr[feats].median()
                ytr = tr[err_col].abs()
                if HAS_LGB:
                    qm = lgb.LGBMRegressor(objective='quantile', alpha=CFG.band_coverage, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, random_state=CFG.seed, verbose=-1).fit(tr[feats].astype(float).fillna(med), ytr)
                    bimp.append(pd.Series(qm.feature_importances_, index=feats))
                else:
                    qm = HistGradientBoostingRegressor(loss='quantile', quantile=CFG.band_coverage, max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=CFG.seed).fit(tr[feats].astype(float).fillna(med), ytr)
                q_cal = np.maximum(qm.predict(cal[feats].astype(float).fillna(med)), 0.5); q_te = np.maximum(qm.predict(te[feats].astype(float).fillna(med)), 0.5)
                s = float(np.quantile(cal[err_col].abs().to_numpy() / q_cal, CFG.band_coverage))
                fixed = float(np.quantile(cal[err_col].abs(), CFG.band_coverage))
                te['mid'] = mid_name; te['abs_err'] = te[err_col].abs(); te['hw_fixed'] = fixed; te['hw_quantile'] = q_te; te['hw_conformal'] = s * q_te; te['s_conformal'] = s
                # rolling conformal: for each test date, the scale is the target quantile of the normalised scores over the previous
                # conformal_window_days trading days (calibration month + earlier test days), so a regime shift re-scales within days
                hist_dates = pd.concat([cal.assign(_q=q_cal), te.assign(_q=q_te)])[['trade_date', err_col, '_q']].assign(score=lambda d: d[err_col].abs() / d['_q'])
                daily_scores = {d: g['score'].to_numpy() for d, g in hist_dates.groupby('trade_date')}
                all_days = sorted(daily_scores); s_roll = {}
                for d in sorted(te['trade_date'].unique()):
                    prev = [x for x in all_days if x < d][-CFG.conformal_window_days:]
                    pool = np.concatenate([daily_scores[x] for x in prev]) if prev else np.array([s])
                    s_roll[d] = float(np.quantile(pool, CFG.band_coverage)) if len(pool) > 50 else s
                te['s_rolling'] = te['trade_date'].map(s_roll); te['hw_rolling'] = te['s_rolling'] * q_te
                # the fair benchmark for the conditional band: a FIXED width rescaled on the same rolling window (the target quantile of
                # |error| over the previous conformal_window_days trading days)
                daily_abs = {d: g[err_col].abs().to_numpy() for d, g in hist_dates.groupby('trade_date')}
                hw_rf = {}
                for d in sorted(te['trade_date'].unique()):
                    prev = [x for x in all_days if x < d][-CFG.conformal_window_days:]
                    pool = np.concatenate([daily_abs[x] for x in prev]) if prev else np.array([fixed])
                    hw_rf[d] = float(np.quantile(pool, CFG.band_coverage)) if len(pool) > 50 else fixed
                te['hw_rolling_fixed'] = te['trade_date'].map(hw_rf)
                bands.append(te[['cusip', 'trade_date', 'side', 'age_bucket', 'recency_bucket', 'month', 'mid', 'abs_err', 'hw_fixed', 'hw_rolling_fixed', 'hw_quantile', 'hw_conformal', 'hw_rolling', 's_conformal', 's_rolling']])
        _bd = pd.concat(bands, ignore_index=True) if bands else pd.DataFrame()
        if bimp:
            _imp = pd.concat(bimp, axis=1).mean(axis=1); _bd = pd.concat([_bd, pd.DataFrame({'mid': '__importance__', 'cusip': _imp.index, 'abs_err': _imp.to_numpy()})], ignore_index=True)
        return _bd

    BD_ALL = cached('band_models', _band_models, deps=[EC_FP, STAGE_KEYS.get('level_models'), BAND_FEATURES])
    bands = [BD_ALL[BD_ALL['mid'] != '__importance__']] if len(BD_ALL) else []
    if bands:
        bd = pd.concat(bands, ignore_index=True); bd['month'] = bd['month'].astype(str)
        BANDS = ['fixed', 'rolling_fixed', 'quantile', 'conformal', 'rolling']
        for k in BANDS:
            bd[f'cov_{k}'] = (bd['abs_err'] <= bd[f'hw_{k}']).astype(float)

        def band_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
            g = frame.groupby(by, observed=True)
            return pd.DataFrame({'n': g.size(), **{f'coverage {k}': g[f'cov_{k}'].mean() for k in BANDS}, **{f'half-width {k} (bp)': g[f'hw_{k}'].mean() for k in BANDS}})

        for mid_name, g in bd.groupby('mid'):
            print(f'Band around the {mid_name}, target {CFG.band_coverage:.0%} (held-out months; conformal scale s by month: {g.groupby("month")["s_conformal"].first().round(2).to_dict()})')
            display(band_table(g.assign(all='ALL'), 'all').round(3)); display(band_table(g, 'month').round(3)); display(band_table(g, 'side').round(3))
        gD = bd[bd['mid'] == 'level model D'].copy(); gD['width_decile'] = pd.qcut(gD['hw_rolling'].rank(method='first'), 10, labels=False) + 1
        dec = band_table(gD, 'width_decile')
        print('Calibration by rolling-conformal width decile, level model D mid (flat at the target = calibrated):'); display(dec[['n', 'coverage fixed', 'coverage rolling_fixed', 'coverage quantile', 'coverage conformal', 'coverage rolling', 'half-width rolling_fixed (bp)', 'half-width rolling (bp)']].round(3))
        _eff = bd.groupby('mid')[[f'cov_{k}' for k in BANDS] + [f'hw_{k}' for k in BANDS]].mean()
        eff = pd.DataFrame({k: {'coverage': _eff[f'cov_{k}'], 'half-width (bp)': _eff[f'hw_{k}']} for k in BANDS}).T if False else pd.concat({k: pd.DataFrame({'coverage': _eff[f'cov_{k}'], 'half-width (bp)': _eff[f'hw_{k}']}) for k in BANDS}, names=['band', 'mid'])
        eff['width per point of coverage'] = eff['half-width (bp)'] / (100 * eff['coverage'])
        print('Band efficiency by mid: half-width per point of coverage (lower is better; the rolling fixed band is the fair benchmark for the conditional bands)'); display(eff.round(3))
        s_path = bd[bd['mid'] == 'level model D'].groupby('trade_date')[['s_conformal', 's_rolling']].first()
        fig, ax = plt.subplots(1, 3, figsize=(19, 4.3))
        for k, st in [('fixed', 's--'), ('rolling_fixed', 'x--'), ('quantile', '^-.'), ('conformal', 'o-'), ('rolling', 'D-')]:
            ax[0].plot(dec.index, dec[f'coverage {k}'], st, label=k.replace('_', ' '))
        ax[0].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8); ax[0].set_ylim(0.4, 1.0); ax[0].set_xlabel('rolling-conformal half-width decile'); ax[0].legend(fontsize=8); ax[0].set_title('Coverage by predicted-width decile (level model D mid)', fontsize=10)
        dcov = bd[bd['mid'] == 'level model D'].groupby('trade_date')[['cov_fixed', 'cov_rolling_fixed', 'cov_conformal', 'cov_rolling']].mean().rolling(5).mean()
        for k, c in [('fixed', 'grey'), ('rolling_fixed', 'k'), ('conformal', '#4C72B0'), ('rolling', '#C44E52')]:
            ax[1].plot(dcov.index, dcov[f'cov_{k}'], color=c, lw=1.2, label=k.replace('_', ' '))
        ax[1].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8); ax[1].legend(fontsize=8); ax[1].set_title('Daily coverage (5-day mean), level model D mid: monthly vs rolling calibration', fontsize=10)
        ax2 = ax[1].twinx(); ax2.plot(s_path.index, s_path['s_rolling'], color='#C44E52', ls=':', lw=0.8); ax2.set_ylabel('rolling scale s', color='#C44E52')
        bw = bd.groupby(['mid', 'side'])[['hw_rolling_fixed', 'hw_rolling']].mean()
        bw.plot.bar(ax=ax[2]); ax[2].set_title('Mean half-width by mid and side (bp): rolling fixed vs rolling conformal', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(['rolling fixed', 'rolling conformal'], fontsize=8); ax[2].tick_params(axis='x', labelsize=7)
        savefig('11_conformal_band')
        bd.to_parquet(ARTIFACTS / 'conformal_band_v3.parquet', index=False)
        record('conformal_band', features=BAND_FEATURES, by_mid={mid_name: band_table(g.assign(all='ALL'), 'all').round(4).to_dict(orient='records')[0] for mid_name, g in bd.groupby('mid')}, by_width_decile=dec.round(4).reset_index().to_dict(orient='records'), efficiency=eff.round(4).reset_index().to_dict(orient='records'))
    else:
        print('Section 11: not enough months for train / calibration / test.')
else:
    print('Section 11 skipped: needs Section 10.')

# %% [markdown]
# ## 12. Where the gains live: breakdown by characteristics, factor betas and beta-space clusters
#
# The research question is not whether the whole universe improves but *where* the improvement concentrates, in
# the model's own coordinates. Every bond-day carries its characteristics, its three factor betas and a
# point-in-time beta-space cluster; every trade inherits them from its residual date. Four views, all on the same
# segmentation: (1) the error-correction gain of the same-side memory (S) and the level model (D) over the algo
# by segment, with a duration-by-call heatmap and the top characteristic cells; (2) whether the algo's error
# loads on the factor betas at all, before and after correction, which is the direct tie between the pricing
# layer and the factor model; (3) the record residual signal within each cluster (hedged Sharpe, IC, trade test);
# (4) the factor roll-forward by segment, in Section 13.

# %%
if HAS_TRADES and 'ecp' in globals() and not ecp.empty:
    bk = ecp.copy()
    bk['dur_bucket'] = pd.cut(bk['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']).astype(str)
    bk['rating_bucket'] = pd.cut(bk['rating_score'].fillna(-1), bins=[-2, -0.5, 14.5, 17.5, 20.5, 21.5], labels=['NR', 'BBB and below', 'A', 'AA', 'AAA']).astype(str)
    bk['liq_q'] = bk.groupby('trade_date', observed=True)['z_liquidity_20'].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=['Q1 illiquid', 'Q2', 'Q3', 'Q4', 'Q5 liquid']).astype(str) if s.notna().sum() > 50 else 'NA')
    for j in range(CFG.selected_k):
        bk[f'beta{j+1}_tercile'] = bk.groupby('trade_date', observed=True)[f'beta{j+1}'].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'mid', 'high']).astype(str) if s.notna().sum() > 30 else 'NA')
    bk['cluster'] = bk['beta_cluster'].map(CLUSTER_LABEL).fillna('NA')
    bk['side_label'] = bk['side'].map({'P': 'P dealer buys (bid)', 'S': 'S dealer sells (offer)', 'D': 'D inter-dealer'}).fillna(bk['side'].astype(str))
    MIDS_BK = {'A algo quote': 'e_algo_bp', 'C side-pooled EWMA': 'e_C_bp', 'S same-side memory': 'e_S_bp', 'K per-side state-space memory': 'e_K_bp', 'D level model': 'e_D_bp'}

    def gain_table(frame: pd.DataFrame, by: str, n_min: int) -> pd.DataFrame:
        g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size()})
        for k, c in MIDS_BK.items():
            out[f'MAE {k.split()[0]}'] = g[c].apply(lambda s: s.abs().mean())
        for k, c in list(MIDS_BK.items())[1:]:
            d = frame.assign(gain=frame['e_algo_bp'].abs() - frame[c].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
            out[f'gain {k.split()[0]} (bp)'] = d.mean(); out[f't {k.split()[0]}'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
        out['share of trades'] = out['n'] / len(frame)
        return out[out['n'] >= n_min]

    SEGMENTS = {'duration bucket': 'dur_bucket', 'call structure': 'call_structure', 'rating bucket': 'rating_bucket', 'state bucket': 'state_bucket', 'liquidity quintile': 'liq_q', 'beta-space cluster': 'cluster', 'beta1 tercile': 'beta1_tercile', 'beta2 tercile': 'beta2_tercile', 'beta3 tercile': 'beta3_tercile', 'MSRB side': 'side_label', 'trade size': 'qty_group', 'industry': 'industry_bucket'}
    seg_tables = {}
    for name, col in SEGMENTS.items():
        if col in bk.columns and bk[col].notna().any():
            seg_tables[name] = gain_table(bk.dropna(subset=[col]), col, CFG.n_min_cell)
            print(f'Gain over the algo by {name}:'); display(seg_tables[name].round(3))
    # characteristic cells: duration x call x rating, ranked by the gain of the same-side memory
    bk['cell'] = bk['dur_bucket'] + ' | ' + bk['call_structure'].astype(str) + ' | ' + bk['rating_bucket']
    cells = gain_table(bk, 'cell', CFG.n_min_cell).sort_values('gain S (bp)', ascending=False)
    print(f'Top and bottom characteristic cells by the gain of the same-side memory (cells with >= {CFG.n_min_cell:,} trades):'); display(pd.concat([cells.head(10), cells.tail(5)]).round(3))
    # (2) does the algo error load on the factor betas? FM of the error on the three betas (plus side), before and after correction
    load_rows = []
    for k, c in MIDS_BK.items():
        r = fm_multi(bk.dropna(subset=beta_cols), c, beta_cols, ['log_size', 'side_P', 'side_S']).assign(mid=k)
        load_rows.append(r)
    beta_loading = pd.concat(load_rows, ignore_index=True)
    print('Fama-MacBeth: pricing error on the factor betas (does the quote miss a factor axis?), by mid'); display(beta_loading.round(4))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    hm = bk.groupby(['dur_bucket', 'call_structure'], observed=True).apply(lambda g: (g['e_algo_bp'].abs() - g['e_S_bp'].abs()).mean() if len(g) >= CFG.n_min_cell else np.nan, include_groups=False).unstack('call_structure').reindex(['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11'])
    im = ax[0].imshow(hm.to_numpy(float), cmap='RdYlGn', aspect='auto', vmin=-np.nanmax(np.abs(hm.to_numpy(float))), vmax=np.nanmax(np.abs(hm.to_numpy(float))))
    ax[0].set_xticks(range(hm.shape[1])); ax[0].set_xticklabels(hm.columns, rotation=20, fontsize=8); ax[0].set_yticks(range(hm.shape[0])); ax[0].set_yticklabels(hm.index); ax[0].set_ylabel('modified duration bucket')
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iloc[i, j]
            if np.isfinite(v):
                ax[0].text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=8)
    ax[0].set_title('Gain of the same-side memory over the algo (bp): duration x call structure', fontsize=10); ax[0].grid(False); plt.colorbar(im, ax=ax[0], fraction=0.046)
    cl = seg_tables.get('beta-space cluster')
    if cl is not None and len(cl):
        cl[['gain S (bp)', 'gain D (bp)']].plot.barh(ax=ax[1]); ax[1].axvline(0, color='k', lw=0.6); ax[1].set_title('Gain over the algo by beta-space cluster (bp)', fontsize=10); ax[1].set_ylabel(''); ax[1].legend(['S same-side memory', 'D level model'], fontsize=8); ax[1].tick_params(axis='y', labelsize=8)
    bl = beta_loading[beta_loading['mid'].isin(['A algo quote', 'S same-side memory', 'D level model'])].pivot(index='score', columns='mid', values='fm_beta').reindex(beta_cols)
    bl.plot.bar(ax=ax[2]); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('FM loading of the pricing error on each factor beta (bp per unit beta)', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(fontsize=8)
    savefig('12_where_the_gains_live')
    # (3) the record residual signal within each cluster
    rec_name = next(k for k in signals if (k.startswith('State-space') and ('pooled' in k) == (CFG.record_signal == 'pooled')))
    sig = signals[rec_name].merge(resid[['cusip', 'date', 'beta_cluster']], on=['cusip', 'date'], how='left')
    sig['cluster'] = sig['beta_cluster'].map(CLUSTER_LABEL)
    sig_rows = []
    for c, g in sig.groupby('cluster'):
        if len(g) < CFG.n_min_cell:
            continue
        ic = ic_summary(daily_rank_ic(g, 'yhat', 'next_residual'))
        sig_rows.append({'cluster': c, 'bond_days': len(g), 'rank IC': ic['ic_mean'], 'IC t': ic['ic_t'], 'hedged Sharpe (within cluster)': sharpe(portfolio_pnl(g))})
    sig_cluster = pd.DataFrame(sig_rows, columns=['cluster', 'bond_days', 'rank IC', 'IC t', 'hedged Sharpe (within cluster)']).set_index('cluster')
    if 'ssm_rank' in tv.columns and len(sig_cluster):
        tvc = tv.copy(); tvc['cluster'] = tvc['beta_cluster'].map(CLUSTER_LABEL)   # beta_cluster already rides on the trade frame from the residual join
        fmc = {c: fm_slope(g, 'e_mark_bp', 'ssm_rank', CONTROLS) for c, g in tvc.dropna(subset=['ssm_rank']).groupby('cluster') if len(g) >= CFG.n_min_cell}
        sig_cluster['trade - mark on signal (FM bp/rank)'] = pd.Series({c: v['fm_beta'] for c, v in fmc.items()}); sig_cluster['FM t'] = pd.Series({c: v['fm_t'] for c, v in fmc.items()})
    print(f'Record residual signal ({rec_name}) within beta-space clusters:'); display(sig_cluster.round(3))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    if len(sig_cluster):
        sig_cluster['hedged Sharpe (within cluster)'].plot.barh(ax=ax[0], color='#4C72B0'); ax[0].axvline(0, color='k', lw=0.6); ax[0].set_title('Record signal: hedged Sharpe within each beta-space cluster', fontsize=10); ax[0].set_ylabel(''); ax[0].tick_params(axis='y', labelsize=8)
        sig_cluster['rank IC'].plot.barh(ax=ax[1], color='#55A868')
    ax[1].axvline(0, color='k', lw=0.6); ax[1].set_title('Record signal: next-day rank IC within each cluster', fontsize=10); ax[1].set_ylabel(''); ax[1].tick_params(axis='y', labelsize=8)
    savefig('12_signal_by_cluster')
    record('breakdown', segments={k: v.round(4).reset_index().to_dict(orient='records') for k, v in seg_tables.items()}, top_cells=cells.head(10).round(4).reset_index().to_dict(orient='records'), beta_loading=beta_loading.round(4).to_dict(orient='records'), signal_by_cluster=sig_cluster.round(4).reset_index().to_dict(orient='records'))
else:
    print('Section 12 skipped: needs Section 10.')

# %% [markdown]
# ### 12a. The achievement map: top gains by characteristic, factor and cluster
#
# The tables above carry every column; this block is the view for a slide. Bars are the gain of the revised third
# term F over the algo quote (bp of MAE against the print, daily Fama-MacBeth mean) with a 95% band from the FM t;
# the diamond is the level model D, the ceiling. Labels carry the share of held-out trades. Three views: the ten
# best segments across every axis next to the ten best and five worst characteristic cells; one panel per
# characteristic axis; and the gains in the model's own coordinates, the factor-beta terciles. The scorecard at
# the end is the one-table version: best and worst segment on each axis.

# %%
if HAS_TRADES and 'seg_tables' in globals() and seg_tables:
    GREEN, RED, INK = '#2E8B57', '#C44E52', '#222222'

    def _se(tab: pd.DataFrame, col: str = 'S') -> pd.Series:
        t = tab[f't {col}'].replace(0, np.nan)
        return (tab[f'gain {col} (bp)'] / t).abs().fillna(0.0)

    def gain_bars(ax, tab: pd.DataFrame, title: str, order: list | None = None, show_d: bool = True, label_share: bool = True) -> pd.DataFrame:
        t = tab.reindex(order) if order is not None else tab.sort_values('gain S (bp)')
        t = t.dropna(subset=['gain S (bp)'])
        y = np.arange(len(t)); g = t['gain S (bp)'].to_numpy(float); se = _se(t).to_numpy(float)
        ax.barh(y, g, color=[GREEN if v >= 0 else RED for v in g], xerr=1.96 * se, error_kw={'ecolor': INK, 'lw': 0.8, 'capsize': 2}, alpha=0.9, label='S same-side memory (95% band)')
        if show_d and 'gain D (bp)' in t.columns:
            ax.plot(t['gain D (bp)'].to_numpy(float), y, 'D', color=INK, ms=4, label='D level model (ceiling)')
        ax.axvline(0, color=INK, lw=0.6); ax.set_yticks(y)
        ax.set_yticklabels([f'{i}  ({s:.0%})' if label_share else str(i) for i, s in zip(t.index, t['share of trades'])], fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel('gain over the algo quote, bp of MAE against the print', fontsize=8); ax.grid(axis='x', alpha=0.3)
        span = max(np.nanmax(np.abs(g) + 1.96 * se), 1e-9)
        for yi, v, s in zip(y, g, se):
            ax.text(v + (1.96 * s + 0.03 * span) * (1 if v >= 0 else -1), yi, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
        return t

    # (1) the ten best segments across every axis, and the ten best / five worst characteristic cells
    AX_LABEL = {'duration bucket': 'dur', 'call structure': 'call', 'rating bucket': 'rating', 'state bucket': 'state', 'liquidity quintile': 'liq', 'beta-space cluster': 'cluster', 'beta1 tercile': 'beta1', 'beta2 tercile': 'beta2', 'beta3 tercile': 'beta3', 'MSRB side': 'side'}
    allseg = pd.concat([v.set_index(v.index.map(lambda i, k=k: f'{AX_LABEL.get(k, k)}: {i}')) for k, v in seg_tables.items() if len(v)])
    top_seg = allseg.sort_values('gain S (bp)', ascending=False).head(10)
    cells_show = pd.concat([cells.head(10), cells.tail(5)])
    cells_show.index = [f'{i}   [MAE {a:.1f} -> {f:.1f}]' for i, a, f in zip(cells_show.index, cells_show['MAE A'], cells_show['MAE S'])]
    fig, ax = plt.subplots(1, 2, figsize=(19, 6.4))
    gain_bars(ax[0], top_seg, 'Ten best segments across every axis'); ax[0].legend(fontsize=8, loc='lower right')
    gain_bars(ax[1], cells_show, f'Ten best and five worst duration x call x rating cells (>= {CFG.n_min_cell:,} trades)', order=list(cells_show.index[::-1]))
    ax[1].axhline(4.5, color='#888888', lw=0.8, ls='--')
    fig.suptitle('Where the error-correction gains live (share of held-out trades in brackets)', fontsize=12); plt.tight_layout(); savefig('12a_top_gains')

    # (2) one panel per characteristic axis
    axes_show = [k for k in ['duration bucket', 'call structure', 'rating bucket', 'MSRB side', 'trade size', 'industry', 'state bucket', 'liquidity quintile', 'beta-space cluster'] if k in seg_tables and len(seg_tables[k])]
    ORDERS = {'duration bucket': ['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11'], 'liquidity quintile': ['Q1 illiquid', 'Q2', 'Q3', 'Q4', 'Q5 liquid'], 'MSRB side': ['P dealer buys (bid)', 'S dealer sells (offer)', 'D inter-dealer'], 'trade size': QTY_LABELS}
    _nr = max(1, int(np.ceil(len(axes_show) / 3)))
    fig, axs = plt.subplots(_nr, 3, figsize=(20, 5 * _nr), squeeze=False); axs = axs.ravel()
    for a, k in zip(axs, axes_show):
        order = [o for o in ORDERS.get(k, []) if o in seg_tables[k].index]
        gain_bars(a, seg_tables[k], f'By {k}', order=order[::-1] if order else None)
    for a in axs[len(axes_show):]:
        a.set_visible(False)
    if axes_show:
        axs[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Gain of the same-side memory over the algo quote by characteristic (bars, 95% band) and the level model (diamond)', fontsize=12); plt.tight_layout(); savefig('12a_gain_by_characteristic')

    # (3) the model's own coordinates: factor-beta terciles
    terc = [k for k in ['beta1 tercile', 'beta2 tercile', 'beta3 tercile'] if k in seg_tables and len(seg_tables[k])]
    if terc:
        FACTOR_NAME = {'beta1 tercile': 'Level beta (beta1) tercile', 'beta2 tercile': 'Slope beta (beta2) tercile', 'beta3 tercile': 'Yield-tilt beta (beta3) tercile'}
        fig, axs = plt.subplots(1, len(terc), figsize=(6.4 * len(terc), 4.6), squeeze=False); axs = axs.ravel()
        for a, k in zip(axs, terc):
            tab = seg_tables[k].reindex([o for o in ['low', 'mid', 'high'] if o in seg_tables[k].index])
            x = np.arange(len(tab)); w = 0.36
            a.bar(x - w / 2, tab['gain S (bp)'], w, yerr=1.96 * _se(tab), color=[GREEN if v >= 0 else RED for v in tab['gain S (bp)']], capsize=3, label='S same-side memory')
            a.bar(x + w / 2, tab['gain D (bp)'], w, color='#8C8C8C', label='D level model')
            a.axhline(0, color=INK, lw=0.6); a.set_xticks(x); a.set_xticklabels([f'{i}\n(algo MAE {m:.1f} bp, {s:.0%})' for i, m, s in zip(tab.index, tab['MAE A'], tab['share of trades'])], fontsize=8)
            a.set_title(FACTOR_NAME.get(k, k), fontsize=10); a.set_ylabel('gain over the algo (bp)', fontsize=8); a.grid(axis='y', alpha=0.3)
            for xi, v in zip(x, tab['gain S (bp)']):
                a.text(xi - w / 2, v, f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        axs[0].legend(fontsize=8)
        fig.suptitle('Gains in factor coordinates: factor-beta terciles, point-in-time within each trade date', fontsize=11); plt.tight_layout(); savefig('12a_gain_by_factor')

    # scorecard: best and worst segment on each axis, the one-table version for a slide
    rows = []
    for k, v in seg_tables.items():
        if v.empty or v['gain S (bp)'].isna().all():
            continue
        b, w = v['gain S (bp)'].idxmax(), v['gain S (bp)'].idxmin()
        rows.append({'axis': k, 'best segment': b, 'gain S (bp)': v.loc[b, 'gain S (bp)'], 't': v.loc[b, 't S'], 'share': v.loc[b, 'share of trades'], 'algo MAE': v.loc[b, 'MAE A'], 'gain D (bp)': v.loc[b, 'gain D (bp)'],
                     'worst segment': w, 'worst gain S (bp)': v.loc[w, 'gain S (bp)'], 'worst t': v.loc[w, 't S'], 'worst share': v.loc[w, 'share of trades']})
    scorecard = pd.DataFrame(rows).set_index('axis')
    print('Scorecard: best and worst segment on each axis by the gain of the same-side memory (the slide table):'); display(scorecard.round(2))
    record('achievement_map', top_segments=top_seg.round(4).reset_index().to_dict(orient='records'), scorecard=scorecard.round(4).reset_index().to_dict(orient='records'))
else:
    print('Section 12a skipped: needs Section 12.')

# %% [markdown]
# #### By MSRB side: bid, offer and inter-dealer
#
# The same gains split by the side of the print: P is a dealer purchase from a customer (our bid), S a dealer
# sale to a customer (our offer), D an inter-dealer trade. The side is also an axis in the tables and charts
# above; here it is crossed with the characteristics, because a correction that helps on the bid in one
# segment and hurts on the offer in another has to be gated by both. Cells are the gain of the revised third
# term F in bp; an asterisk marks |FM t| >= 2. The last figure is the five best cells on each side.

# %%
if HAS_TRADES and 'seg_tables' in globals() and seg_tables and 'side_label' in bk.columns:
    SIDES = [s for s in ['P dealer buys (bid)', 'S dealer sells (offer)', 'D inter-dealer'] if s in bk['side_label'].unique()]
    SIDE_AXES = {'duration bucket': 'dur_bucket', 'call structure': 'call_structure', 'rating bucket': 'rating_bucket', 'beta-space cluster': 'cluster', 'liquidity quintile': 'liq_q', 'trade size': 'qty_group', 'industry': 'industry_bucket'}
    side_tabs = {}
    for name, col in SIDE_AXES.items():
        if col not in bk.columns or not bk[col].notna().any():
            continue
        parts = {s: gain_table(bk[bk['side_label'] == s], col, CFG.n_min_cell) for s in SIDES}
        g = pd.DataFrame({s: p['gain S (bp)'] for s, p in parts.items()}); t = pd.DataFrame({s: p['t S'] for s, p in parts.items()})
        d = pd.DataFrame({s: p['gain D (bp)'] for s, p in parts.items()}); n = pd.DataFrame({s: p['n'] for s, p in parts.items()})
        order = [o for o in ORDERS.get(name, []) if o in g.index] or list(g.index)
        side_tabs[name] = {'gain S': g.reindex(order), 't S': t.reindex(order), 'gain D': d.reindex(order), 'n': n.reindex(order)}
        show = side_tabs[name]['gain S'].round(2).astype(str) + np.where(side_tabs[name]['t S'].abs() >= 2, '*', '') + '  (D ' + side_tabs[name]['gain D'].round(2).astype(str) + ')'
        print(f'Gain over the algo by {name} x MSRB side: S same-side memory (* = |FM t| >= 2), D level model in brackets'); display(show)
    # side-level summary across every mid, with the algo MAE
    side_all = gain_table(bk, 'side_label', CFG.n_min_cell).reindex(SIDES)
    print('Gain over the algo by MSRB side, every mid:'); display(side_all.round(3))
    # heatmaps: rows = segment, columns = side
    hm_axes = [k for k in ['duration bucket', 'call structure', 'rating bucket', 'trade size', 'beta-space cluster', 'industry'] if k in side_tabs]
    if hm_axes:
        fig, axs = plt.subplots(1, len(hm_axes), figsize=(5.2 * len(hm_axes), 5.6), squeeze=False); axs = axs.ravel()
        vmax = max(np.nanmax(np.abs(side_tabs[k]['gain S'].to_numpy(float))) for k in hm_axes) or 1.0
        for a, k in zip(axs, hm_axes):
            g = side_tabs[k]['gain S']; t = side_tabs[k]['t S']
            im = a.imshow(g.to_numpy(float), cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
            a.set_xticks(range(g.shape[1])); a.set_xticklabels([s.split(' ')[0] + ' ' + s.split('(')[-1].rstrip(')') if '(' in s else s for s in g.columns], fontsize=8)
            a.set_yticks(range(g.shape[0])); a.set_yticklabels(g.index, fontsize=8); a.set_title(f'{k} x side', fontsize=10); a.grid(False)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    v = g.iloc[i, j]; tt = t.iloc[i, j]
                    if np.isfinite(v):
                        a.text(j, i, f'{v:+.2f}' + ('*' if np.isfinite(tt) and abs(tt) >= 2 else ''), ha='center', va='center', fontsize=8, fontweight='bold' if np.isfinite(tt) and abs(tt) >= 2 else 'normal')
        plt.colorbar(im, ax=axs[-1], fraction=0.046, label='gain S over the algo (bp)')
        fig.suptitle('Gain of the same-side memory by characteristic and MSRB side (P = bid, S = offer, D = inter-dealer; * = |FM t| >= 2)', fontsize=11); plt.tight_layout(); savefig('12a_gain_by_side_heatmap')
    # the five best cells on each side
    side_cells = {}
    fig, axs = plt.subplots(1, len(SIDES), figsize=(6.6 * len(SIDES), 4.8), squeeze=False); axs = axs.ravel()
    for a, s in zip(axs, SIDES):
        cs = gain_table(bk[bk['side_label'] == s], 'cell', CFG.n_min_cell).sort_values('gain S (bp)', ascending=False)
        side_cells[s] = cs
        top = cs.head(5).copy(); top.index = [f'{i}  [MAE {x:.1f} -> {y:.1f}]' for i, x, y in zip(top.index, top['MAE A'], top['MAE S'])]
        if len(top):
            gain_bars(a, top, f'{s}: five best cells', order=list(top.index[::-1]))
        else:
            a.set_title(f'{s}: no cell with >= {CFG.n_min_cell:,} trades', fontsize=10)
            a.set_axis_off()
    axs[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Five best duration x call x rating cells on each side (bars F with 95% band, diamond D)', fontsize=11); plt.tight_layout(); savefig('12a_top_cells_by_side')
    for s, cs in side_cells.items():
        print(f'Top five and bottom three cells, {s}:'); display(pd.concat([cs.head(5), cs.tail(3)])[['n', 'MAE A', 'MAE S', 'MAE D', 'gain S (bp)', 't S', 'gain D (bp)', 't D', 'share of trades']].round(3))
    record('achievement_by_side', side_summary=side_all.round(4).reset_index().to_dict(orient='records'),
           by_axis={k: {m: v[m].round(4).reset_index().to_dict(orient='records') for m in ['gain S', 't S', 'gain D', 'n']} for k, v in side_tabs.items()},
           top_cells={s: cs.head(5).round(4).reset_index().to_dict(orient='records') for s, cs in side_cells.items()})
else:
    print('Side breakdown skipped: needs Section 12a.')

# %% [markdown]
# ### 12b. What the gains are worth: par- and DV01-weighted error removed
#
# A basis point of yield error is not a dollar until it is multiplied by the trade's par and the bond's dollar
# duration. For every held-out trade the dollar value of one basis point is par x price / 100 x modified duration
# x 0.0001, and the dollar error removed by a mid is (|algo error| - |mid error|) x that value. Summed by month and
# by segment this is the mis-pricing the correction removes, in dollars, on the trades that actually printed. It is
# an accuracy measure, not a P&L: how much of it a desk captures depends on which quotes are hit, which is the
# fill model this notebook does not have. The par-weighted gain in bp is reported next to it so the two views can
# be compared.

# %%
if HAS_TRADES and 'bk' in globals() and 'msrb_price' in bk.columns:
    # par must be finite and plausible: the raw MSRB quantity field carries a few inf / absurd values that would swamp every sum
    _q = pd.to_numeric(bk['msrb_quantity'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    _bad = _q.isna() | (_q <= 0) | (_q > 1e8)
    print(f'par filter for the dollar view: {int(_bad.sum()):,} of {len(bk):,} trades dropped (non-finite, non-positive or above $100mm par)')
    bk = bk[~_bad].copy(); bk['msrb_quantity'] = _q[~_bad].astype(float)
    _px = pd.to_numeric(bk['msrb_price'], errors='coerce').replace([np.inf, -np.inf], np.nan).clip(1, 300).fillna(100.0)
    bk['dollar_per_bp'] = (bk['msrb_quantity'] * _px / 100.0 * bk['modified_duration_lag1'].clip(lower=0).fillna(0) * 1e-4)
    DOL = {'C side-pooled EWMA': 'e_C_bp', 'S same-side memory': 'e_S_bp', 'K per-side state-space memory': 'e_K_bp', 'D level model': 'e_D_bp'}
    for k, c in DOL.items():
        bk[f'$ removed [{k}]'] = (bk['e_algo_bp'].abs() - bk[c].abs()) * bk['dollar_per_bp']
        bk[f'par-bp [{k}]'] = (bk['e_algo_bp'].abs() - bk[c].abs()) * bk['msrb_quantity'].clip(lower=0).fillna(0)
    months_held = bk['month'].nunique()

    def dollar_table(frame: pd.DataFrame, by: str, n_min: int = 0) -> pd.DataFrame:
        g = frame.groupby(by, observed=True)
        out = pd.DataFrame({'trades': g.size(), 'par traded ($mm)': g['msrb_quantity'].sum() / 1e6, 'algo |error| ($k)': (frame['e_algo_bp'].abs() * frame['dollar_per_bp']).groupby(frame[by], observed=True).sum() / 1e3})
        for k in DOL:
            out[f'$ removed per month ($k) [{k}]'] = g[f'$ removed [{k}]'].sum() / 1e3 / months_held
            out[f'par-weighted gain (bp) [{k}]'] = g[f'par-bp [{k}]'].sum() / g['msrb_quantity'].sum().replace(0, np.nan)
        return out[out['trades'] >= n_min]

    dol_month = dollar_table(bk, 'month'); dol_month.index = dol_month.index.astype(str)
    dol_all = dollar_table(bk.assign(all='ALL'), 'all')
    print(f'Dollar error removed on held-out trades ({months_held} months, {len(bk):,} trades, {bk["msrb_quantity"].sum()/1e9:.1f} bn par):'); display(dol_all.T.round(2))
    print('By month:'); display(dol_month[['trades', 'par traded ($mm)'] + [c for c in dol_month.columns if c.startswith('$ removed')]].round(1))
    dol_seg = {name: dollar_table(bk, col, CFG.n_min_cell) for name, col in [('rating bucket', 'rating_bucket'), ('duration bucket', 'dur_bucket'), ('call structure', 'call_structure'), ('beta-space cluster', 'cluster'), ('trade size', 'qty_group')] + ([('industry', 'industry_bucket')] if 'industry_bucket' in bk.columns and bk['industry_bucket'].notna().any() else [])}
    for name, tbl in dol_seg.items():
        print(f'By {name}:'); display(tbl[['trades', 'par traded ($mm)', 'algo |error| ($k)'] + [c for c in tbl.columns if '$ removed' in c]].round(1))
    dol_cells = dollar_table(bk, 'cell', CFG.n_min_cell).sort_values('$ removed per month ($k) [S same-side memory]', ascending=False)
    print('Top characteristic cells by dollars removed per month, same-side memory:'); display(dol_cells.head(10)[['trades', 'par traded ($mm)', 'algo |error| ($k)', '$ removed per month ($k) [S same-side memory]', 'par-weighted gain (bp) [S same-side memory]', '$ removed per month ($k) [D level model]']].round(1))
    _pos = bk.groupby('cell', observed=True)['$ removed [S same-side memory]'].sum(); _parc = bk.groupby('cell', observed=True)['msrb_quantity'].sum()
    print(f'share of par traded in cells where the same-side memory removes error: {_parc[_pos > 0].sum() / _parc.sum():.1%} | share of dollars removed coming from the top 10 cells: {dol_cells.head(10)["$ removed per month ($k) [S same-side memory]"].sum() / max(dol_cells["$ removed per month ($k) [S same-side memory]"].clip(lower=0).sum(), 1e-9):.1%}')
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    dol_month[[c for c in dol_month.columns if c.startswith('$ removed')]].rename(columns=lambda c: c.split('[')[1].rstrip(']')).plot.bar(ax=ax[0]); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('Dollar error removed per month ($k), by mid', fontsize=10); ax[0].set_xlabel(''); ax[0].legend(fontsize=8)
    _dc = dol_seg['beta-space cluster'][[c for c in dol_seg['beta-space cluster'].columns if c.startswith('$ removed')]].rename(columns=lambda c: c.split('[')[1].rstrip(']'))
    _dc.plot.barh(ax=ax[1]); ax[1].axvline(0, color='k', lw=0.6); ax[1].set_title('Dollar error removed per month ($k) by beta-space cluster', fontsize=10); ax[1].set_ylabel(''); ax[1].legend(fontsize=7); ax[1].tick_params(axis='y', labelsize=8)
    _tc = dol_cells.head(10)['$ removed per month ($k) [S same-side memory]'][::-1]
    _tc.plot.barh(ax=ax[2], color='#55A868'); ax[2].set_title('Top 10 characteristic cells: $k removed per month, same-side memory', fontsize=10); ax[2].set_ylabel(''); ax[2].tick_params(axis='y', labelsize=7)
    savefig('12b_dollars_removed')
    record('dollars', months=months_held, overall=dol_all.round(3).to_dict(orient='records'), by_month=dol_month.round(3).reset_index().to_dict(orient='records'), by_segment={k: v.round(3).reset_index().to_dict(orient='records') for k, v in dol_seg.items()}, top_cells=dol_cells.head(10).round(3).reset_index().to_dict(orient='records'))
else:
    print('Section 12b skipped: needs trade par and price.')

# %% [markdown]
# ## 13. Systematic path: factor roll-forward of stale marks, by segment
#
# The one systematic use of the factor model that the v5 print-to-print panel confirmed out of panel: the next
# print follows the factor-implied move (coefficient 0.83) and ignores the residual move in the marks. Here the
# in-panel version: moving a mark by the cumulative factor-implied change versus leaving it unchanged, by horizon.

# %%
r_sorted = resid.sort_values(['cusip', 'date'], kind='stable')
g = r_sorted.groupby('cusip', observed=True)
rf_rows = []
for h in [1, 2, 3, 5, 10]:
    cum_target = g['target_bp'].transform(lambda s: s.rolling(h).sum().shift(-h + 1))
    cum_resid = g['pit_residual'].transform(lambda s: s.rolling(h).sum().shift(-h + 1))
    ok = cum_target.notna() & cum_resid.notna()
    rf_rows.append({'horizon': h, 'n': int(ok.sum()), 'stale_mae_bp': cum_target[ok].abs().mean(), 'rolled_mae_bp': cum_resid[ok].abs().mean(), 'stale_rmse_bp': np.sqrt((cum_target[ok] ** 2).mean()), 'rolled_rmse_bp': np.sqrt((cum_resid[ok] ** 2).mean())})
rollf = pd.DataFrame(rf_rows).set_index('horizon'); rollf['rmse_reduction'] = 1 - rollf['rolled_rmse_bp'] / rollf['stale_rmse_bp']
display(rollf.round(3))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rollf.index, rollf['stale_mae_bp'], marker='o', label='leave mark stale'); ax.plot(rollf.index, rollf['rolled_mae_bp'], marker='o', label='roll forward by beta . f'); ax.set_xlabel('horizon (observations)'); ax.set_ylabel('MAE (bp)'); ax.legend(); ax.set_title('Factor roll-forward of marks (in-panel upper bound)')
savefig('13_roll_forward')
record('systematic_path', roll_forward=rollf.round(4).reset_index().to_dict(orient='records'))

# %%
# roll-forward by segment: where does the factor model move stale marks most usefully?
h = 5
_rs = resid.sort_values(['cusip', 'date'], kind='stable'); _g = _rs.groupby('cusip', observed=True)
_rs = _rs.assign(cum_target=_g['target_bp'].transform(lambda s: s.rolling(h).sum().shift(-h + 1)), cum_resid=_g['pit_residual'].transform(lambda s: s.rolling(h).sum().shift(-h + 1))).dropna(subset=['cum_target', 'cum_resid'])
_rs = _rs.merge(model[['cusip', 'date', 'modified_duration_lag1', 'call_structure']], on=['cusip', 'date'], how='left')
_rs['dur_bucket'] = pd.cut(_rs['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=['<1', '1-2.5', '2.5-4', '4-6', '6-8', '8-11', '>11']).astype(str)
_rs['cluster'] = _rs['beta_cluster'].map(CLUSTER_LABEL)
rf_seg = {}
for name, col in [('duration bucket', 'dur_bucket'), ('call structure', 'call_structure'), ('beta-space cluster', 'cluster')]:
    g = _rs.groupby(col, observed=True)
    t = pd.DataFrame({'bond_days': g.size(), 'stale_rmse_bp': g['cum_target'].apply(lambda s: np.sqrt((s ** 2).mean())), 'rolled_rmse_bp': g['cum_resid'].apply(lambda s: np.sqrt((s ** 2).mean()))}); t['rmse_reduction'] = 1 - t['rolled_rmse_bp'] / t['stale_rmse_bp']
    rf_seg[name] = t; print(f'Roll-forward at h={h} by {name}:'); display(t.round(3))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
for a, (name, t) in zip(ax, rf_seg.items()):
    t['rmse_reduction'].plot.bar(ax=a, color='#8172B2'); a.set_title(f'Roll-forward RMSE reduction (h={h}) by {name}', fontsize=10); a.set_xlabel(''); a.tick_params(axis='x', rotation=30, labelsize=8)
savefig('13_roll_forward_by_segment')
record('systematic_path', roll_forward_by_segment={k: v.round(4).reset_index().to_dict(orient='records') for k, v in rf_seg.items()})

# %% [markdown]
# ## 14. Results registry and summary

# %%
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
summary_rows.append(('Paper portfolio hedged Sharpe (vol-scaled, ALL)', '; '.join(f"{k}: {v:.2f}" for k, v in paper['ALL'].items()) + (f"; walk-forward selected {sharpe(pnl_selected):.2f}" if len(pnl_selected) else '')))
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
if 'breakdown' in REGISTRY:
    _bl = pd.DataFrame(REGISTRY['breakdown']['beta_loading']); _a = _bl[_bl['mid'] == 'A algo quote'].set_index('score')
    summary_rows.append(('Algo error loading on factor betas (FM bp per unit beta, t)', '; '.join(f"{b}: {_a.loc[b, 'fm_beta']:+.2f} (t {_a.loc[b, 'fm_t']:+.1f})" for b in beta_cols)))
    _tc = pd.DataFrame(REGISTRY['breakdown']['top_cells'])
    if len(_tc):
        summary_rows.append(('Top characteristic cell by gain of the same-side memory', f"{_tc.iloc[0]['cell']}: {_tc.iloc[0]['gain S (bp)']:+.2f} bp over {int(_tc.iloc[0]['n']):,} trades"))
if 'error_correction' in REGISTRY and 'ecp' in globals():
    _ex = ecp[ecp['modified_duration_lag1'] >= 1.0]
    summary_rows.append(('Headline MAE excluding duration < 1y: algo | C | S | K | D', ' | '.join(f"{_ex[c].abs().mean():.2f}" for c in ['e_algo_bp', 'e_C_bp', 'e_S_bp', 'e_K_bp', 'e_D_bp']) + f' bp ({len(_ex):,} trades, {len(_ex) / len(ecp):.0%})'))
if 'conformal_band' in REGISTRY and 'efficiency' in REGISTRY['conformal_band']:
    _ef = pd.DataFrame(REGISTRY['conformal_band']['efficiency']); _efD = _ef[_ef['mid'] == 'level model D'].set_index('band')
    summary_rows.append(('Band efficiency, level model D mid (half-width bp per coverage point)', '; '.join(f"{b}: {_efD.loc[b, 'width per point of coverage']:.3f} ({_efD.loc[b, 'coverage']:.0%} at {_efD.loc[b, 'half-width (bp)']:.1f} bp)" for b in ['rolling_fixed', 'rolling'] if b in _efD.index)))
if 'dollars' in REGISTRY:
    _d = REGISTRY['dollars']['overall'][0]
    summary_rows.append(('Dollar error removed per month on held-out prints ($k): C | S | K | D', ' | '.join(f"{_d[f'$ removed per month ($k) [{k}]']:,.0f}" for k in ['C side-pooled EWMA', 'S same-side memory', 'K per-side state-space memory', 'D level model']) + f" of {_d['algo |error| ($k)'] / max(len(REGISTRY['dollars']['by_month']), 1):,.0f} algo error per month"))
summary_rows.append(('Roll-forward RMSE reduction, h=1 / h=10', f"{rollf.loc[1, 'rmse_reduction']:.0%} / {rollf.loc[10, 'rmse_reduction']:.0%}"))
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
(ARTIFACTS / 'results_registry.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')
print('registry updated:', ARTIFACTS / 'results_registry.json')
