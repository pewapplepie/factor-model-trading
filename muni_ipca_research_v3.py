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
# 3. **The by-bucket state-space filter as the residual signal of record**: AR drift plus mark-noise MA, parameters
#    by activity bucket, with the filtered drift as the signal and the filtered mark noise as the mark-quality score.
# 4. **Algo error correction**: the v5 print-to-print panel found the algo's own pricing error persists 58% from
#    one print to the next. That persistence is turned into point-in-time features (last matched print's algo
#    error, its age, an exponentially weighted history), a two-parameter baseline (algo + rho(age) x last error),
#    and a level model with and without the error features. The band around the chosen mid is then calibrated
#    by **split conformal** scaling, which fixes the under-coverage of the raw quantile model.
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
# 9. Transaction validation of the signal of record (Fama-MacBeth with block bootstrap)
# 10. Algo error correction: persistence, point-in-time error features, baselines, level model with and without them
# 11. Conformal band around the chosen mid
# 12. Systematic path: factor roll-forward of stale marks
# 13. Results registry and summary
# 14. Robustness: dispersion-day exclusions, bootstrap inference, pre-registered choices
#
# Every leakage boundary is explicit: Gamma is refit on past dates only, instruments are lagged and rank-normalised
# within date, every trade is matched to a residual dated strictly before the trade date, every print-derived
# feature is cut at the algo signal time, and every calibration uses the month before the test month.

# %%
from __future__ import annotations

import json
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
    cadences: tuple[str, ...] = ('weekly', 'monthly', 'quarterly', 'frozen', 'regime')
    record_cadence: str = 'monthly'       # residuals of record (pre-registered after the v5 swap test)
    regime_cadence_days: int = 30
    # --- residual signal (Sections 7-8) ---
    portfolio_scaling: str = 'vol'
    block_days: int = 5
    n_boot: int = 500
    ssm_fit_bonds: int = 2000
    ssm_maxiter: int = 120
    # --- algo error correction and band (Sections 10-11) ---
    err_halflife_prints: float = 3.0      # EWMA of past algo errors, in prints
    age_bins_days: tuple[float, ...] = (0, 1, 3, 7, 21, 1e9)
    level_model_trees: int = 600
    conformal_calibration_months: int = 1
    seed: int = 20260921

# Pre-registered choices, fixed on 2026-10-07 after the v5 review and before this notebook was run. Section 14
# prints them so a reader can tell a choice from a fit.
PRE_REGISTERED = {
    'fixed_on': '2026-10-07',
    'selected_k': 3, 'record_cadence': 'monthly', 'level_signal_window': 20, 'activity_window': 20,
    'dispersion_rule': 'share of |target - median_t| > 10 bp above 10% of bonds -> zero weight in the fit',
    'common_move_rule': '|median_t| > 10 bp -> flagged and kept', 'partial_day_rule': 'bonds priced < 60% of trailing-20-date median -> dropped',
    'date_weighting': 'inverse variance, cap 4x', 'duration_instrument': 'ratio',
    'residual_signal_of_record': 'state-space AR drift + mark-noise MA, parameters by activity bucket, winsorised increments',
    'portfolio_scaling': 'vol', 'band_coverage': 0.80, 'conformal': 'split conformal, calibration = the month before the test month, normalised score |err| / q_hat',
    'error_correction_baseline': 'algo + rho(age bucket) x last matched print algo error, rho estimated on prior months',
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
# Thirteen instruments: an intercept, seven rank-normalised continuous characteristics, the NR flag and four
# state dummies (CA, NY, TX, FL). The continuous set is premium/discount, yield level, years to worst, extension,
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
    yp = yp.dropna(subset=base + [TARGET]).copy()
    yp['target_bp_raw'] = yp[TARGET]
    yp[TARGET] = yp[TARGET].clip(-cfg.target_clip_bp, cfg.target_clip_bp)
    yp['mark_unchanged'] = (yp['target_bp_raw'] == 0).astype(float)
    # consecutive days the evaluated mark has not moved (0 on a day it moved). A stale mark is a data state, not a fair price.
    _moved = (yp['mark_unchanged'] == 0).groupby(yp['cusip'], observed=True).cumsum()
    yp['days_since_mark_move'] = yp['mark_unchanged'].groupby([yp['cusip'], _moved], observed=True).cumsum()
    yp, zcols = rank_normalize(yp, cont)
    yp['market_fv'] = 1.0
    chars = ['market_fv'] + zcols + ['nr_flag'] + [f'state_{s}' for s in STATE_DUMMIES]
    cnt = yp.groupby('cusip', observed=True)[TARGET].transform('count')
    yp = yp[cnt >= cfg.min_obs_per_bond].copy()
    per_date = yp.groupby('date', observed=True)['cusip'].transform('size')
    sparse = per_date < cfg.min_bonds_per_date
    if sparse.any():
        print(f'sparse dates dropped (< {cfg.min_bonds_per_date} bonds): {sorted(yp.loc[sparse, "date"].dt.date.unique())}')
        yp = yp[~sparse].copy()
    keep = ['cusip', 'date', TARGET, 'target_bp_raw', 'mark_unchanged', 'days_since_mark_move', 'closing_lag_days', 'closing_yield', 'closing_yield_lag1',
            'closing_price', 'modified_duration_lag1', 'duration_ratio', 'dv01_lag1', 'years_to_worst', 'extension', 'rating', 'rating_score', 'rating_source',
            'cpn', 'state_code', 'state_bucket', 'state_others', 'mkt_yield', 'long_comp_name', 'maturity_year', 'is_callable', 'call_structure'] + chars
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
display(model[CHARS].describe().T[['mean', 'std', 'min', 'max']])
record('model_panel', rows=len(model), cusips=model['cusip'].nunique(), dates=model['date'].nunique(), instruments=CHARS, mark_unchanged_share=float(model['mark_unchanged'].mean()), geo_buckets=geo.round(4).to_dict())

# %%
# Instrument EDA: correlation structure and the duration vs term relationship
zc = [c for c in CHARS if c.startswith('z_')]
sample = model.sample(min(len(model), 200_000), random_state=CFG.seed)
corr = sample[zc + ['nr_flag'] + [f'state_{s}' for s in STATE_DUMMIES]].corr(method='spearman')
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
# side by side on identical test dates: weekly, monthly and quarterly refits on an expanding window, a Gamma
# frozen at the first OOS date, and a **regime-conditional** Gamma, where two Gammas are fitted on training dates
# split by cross-sectional dispersion (above or below the training median of the daily target sd) and each test
# date uses the Gamma of the regime indicated by the *previous* day's dispersion. The residuals of record come
# from the pre-registered cadence (`record_cadence`), not from the winner of this table.

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
runs: dict = {}
for name in CFG.cadences:
    t0 = time.perf_counter()
    folds_c = make_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CADENCE_DAYS[name])
    res_c, gam_c, gam_raw_c = walk_forward_residuals(model, CHARS, folds_c, CFG, regime=(name == 'regime'), verbose=(name == CFG.record_cadence))
    runs[name] = {'resid': res_c, 'gammas': gam_c, 'gammas_raw': gam_raw_c, 'folds': folds_c}
    print(f'{name:>10}: {len(folds_c):2d} folds | OOS variance explained {1 - res_c["pit_residual"].var() / res_c["target_bp"].var():.4f} | residual sd {res_c["pit_residual"].std():.2f} bp | {time.perf_counter()-t0:.0f}s')

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
(cadence.drop(index='ALL').sub(cadence.drop(index='ALL')['weekly'], axis=0)).drop(columns='weekly').plot(ax=ax[1], marker='o'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('Gain over the weekly refit (points of variance explained)'); ax[1].set_xlabel('')
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

# %% [markdown]
# ## 7. Residual dynamics: autocorrelation, activity buckets, the state-space filter
#
# The pooled autocorrelation function of the residual at lags 1..10, with and without dispersion days, and the
# point-in-time activity buckets (trailing-vol quintiles). Two reference forecasters on expanding monthly folds,
# the trailing-20 mean (the level signal) and the pooled ridge filter on 10 lags, frame the state-space filter in
# 7b, which is the signal of record.

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
# Reference forecasters on expanding monthly folds: the trailing mean and the pooled ridge filter
L = CFG.longconv_lags
for lag in range(1, L + 1):
    resid[f'lag{lag}'] = grp['pit_residual'].shift(lag - 1)  # lag1 == current residual
lag_cols = [f'lag{i}' for i in range(1, L + 1)]
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
FC_SPECS = [(f'Trailing-{W} mean (level signal)', ['resid_mean'], True, None), (f'LongConv-lite ridge L={L} winsor', lag_cols, True, None)]
for name, cols, winsor, fitter in FC_SPECS:
    t0 = time.perf_counter()
    pred, summ, coefs = forecaster_eval(name, cols, winsor, fitter=fitter)
    fc_results.append(summ); fc_preds[name] = pred; fc_coefs[name] = coefs
    print(f'{name}: {time.perf_counter()-t0:.1f}s', '' if 'params' not in summ else summ['params'])

# %% [markdown]
# ### 7b. The signal of record: state-space filter, AR drift plus mark-noise, by activity bucket
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
# attributes to mark noise. This is the model the LongConv-lite ridge approximates, with the weights derived
# rather than fitted, and it gives two by-products the ridge cannot: a filtered **mark-noise** estimate per
# bond-day (a mark-quality score) and parameters per activity bucket, which say whether the hand-coded fade in
# active names is what the data imply (the v5 run found phi near 1 in quiet buckets and near 0 in active ones).
# Increments are winsorised at the training quantiles before filtering. The by-bucket filter's one-step forecast
# is the residual signal of record and is carried into Sections 8, 9 and 13.

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
            fit_bonds = bonds if gname != 'UNKNOWN' else groups.get('ALL', bonds)
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

t0 = time.perf_counter()
ssm_pred, ssm_summ, ssm_params = ssm_walk_forward('State-space AR drift + mark noise (pooled), winsor', by_bucket=False)
ssmb_pred, ssmb_summ, ssmb_params = ssm_walk_forward('State-space AR drift + mark noise (by activity bucket), winsor', by_bucket=True)
print(f'state-space forecasters: {time.perf_counter()-t0:.0f}s')
for nm, pr, sm in [(ssm_summ['forecaster'], ssm_pred, ssm_summ), (ssmb_summ['forecaster'], ssmb_pred, ssmb_summ)]:
    fc_results.append(sm); fc_preds[nm] = pr; fc_coefs[nm] = None
# mark-noise estimate and filtered drift back onto the residual panel (test-month rows only: strictly out of sample)
if not ssm_pred.empty:
    resid = resid.merge(ssm_pred[['cusip', 'date', 'm_hat', 'eta_hat']].rename(columns={'m_hat': 'ssm_drift', 'eta_hat': 'ssm_mark_noise'}), on=['cusip', 'date'], how='left')
    resid['eta_abs'] = resid['ssm_mark_noise'].abs()
if not ssmb_pred.empty:
    # the signal of record: the by-bucket filter's one-step forecast, strictly out of sample
    resid = resid.merge(ssmb_pred[['cusip', 'date', 'yhat']].rename(columns={'yhat': 'ssm_signal'}), on=['cusip', 'date'], how='left')
fc_table = pd.DataFrame(fc_results).set_index('forecaster')
display(fc_table.drop(columns=[c for c in ['params'] if c in fc_table.columns]))
if not ssm_params.empty:
    print('State-space parameters (pooled) by fold:'); display(ssm_params.round(4))
if not ssmb_params.empty:
    lastp = ssmb_params[ssmb_params['month'] == ssmb_params['month'].max()].set_index('group')
    lastp['signal_to_noise'] = lastp['s_drift'] / lastp['s_mark_noise']
    print('State-space parameters by activity bucket (last fold): does the data imply the fade?'); display(lastp.round(4))
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
if fc_coefs[f'LongConv-lite ridge L={L} winsor'] is not None:
    ax[0].bar(np.arange(1, L + 1) - 0.2, fc_coefs[f'LongConv-lite ridge L={L} winsor'][1:], width=0.4, color='#4C72B0', label='ridge (fitted)')
if not ssm_params.empty:
    _taps = ssm_implied_taps(tuple(ssm_params.iloc[-1][['phi', 's_drift', 's_mark_noise', 's_eps']]), L)
    ax[0].bar(np.arange(1, L + 1) + 0.2, _taps, width=0.4, color='#C44E52', label='state-space (implied)')
if fc_coefs.get('ARMA(1,1) pooled, winsor') is not None:
    ax[0].plot(np.arange(1, L + 1), fc_coefs['ARMA(1,1) pooled, winsor'][1:], 'ko-', ms=3, lw=0.8, label='ARMA(1,1) (implied)')
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('lag'); ax[0].legend(fontsize=8); ax[0].set_title('Linear filters over the residual path: fitted vs implied taps', fontsize=10)
best = fc_table['ic_mean'].idxmax()
if not fc_preds[best].empty:
    ic_m = fc_preds[best].groupby('month').apply(lambda g: ic_summary(daily_rank_ic(g, 'yhat', 'next_residual'))['ic_mean'], include_groups=False)
    ic_m.plot.bar(ax=ax[1], color='#55A868'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title(f'Monthly OOS rank IC: {best}', fontsize=10); ax[1].set_xlabel('')
if not ssmb_params.empty:
    _lp = lastp.drop(index=[g for g in ['UNKNOWN', 'ALL'] if g in lastp.index], errors='ignore')
    ax[2].bar(_lp.index, _lp['phi'], color='#8172B2', label='phi (drift persistence)'); ax2 = ax[2].twinx(); ax2.plot(_lp.index, _lp['signal_to_noise'], 'ko-', label='drift sd / mark-noise sd'); ax2.set_ylim(bottom=0)
    ax[2].set_title('State-space parameters by activity bucket (last fold)', fontsize=10); ax[2].legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
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

# Candidate signals: every forecaster from Section 7 (the by-bucket state-space filter learns its own sign per bucket)
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
    # algo error-correction features: the EWMA of this bond's past algo errors up to and including this print (in prints)
    prints['ewm_algo_err_bp'] = prints.groupby('cusip', observed=True)['last_algo_err_bp'].transform(lambda s: s.ewm(halflife=CFG.err_halflife_prints, min_periods=1).mean())
    prints = prints.sort_values('print_ts', kind='stable')
    sig = t[['_row', 'cusip', 'signal_ts']].dropna(subset=['signal_ts']).sort_values('signal_ts')
    j1 = pd.merge_asof(sig, prints, left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    sig20 = sig.assign(signal_ts=sig['signal_ts'] - pd.Timedelta(days=20)).sort_values('signal_ts')
    j2 = pd.merge_asof(sig20, prints[['cusip', 'print_ts', 'cum_prints']], left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    j1 = j1.set_index('_row'); j2 = j2.set_index('_row')
    t = t.set_index('_row')
    t['last_print_spread_bp'] = j1['last_print_spread_bp']; t['last_print_side'] = j1['last_print_side'].fillna('NONE')
    t['last_algo_err_bp'] = j1['last_algo_err_bp']; t['ewm_algo_err_bp'] = j1['ewm_algo_err_bp']; t['n_prior_prints'] = j1['cum_prints'].fillna(0)
    t['recency_days_sig'] = (t['signal_ts'] - j1['print_ts']).dt.total_seconds() / 86400.0
    t['prints_20d'] = (j1['cum_prints'].fillna(0) - j2['cum_prints'].fillna(0)).clip(lower=0)
    t = t.reset_index(drop=True)
    t['recency_bucket'] = pd.cut(t['recency_days'], bins=[-0.01, 1, 3, 7, 21, np.inf], labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'first_print')
    return t

def join_trades_to_residual(t: pd.DataFrame, res: pd.DataFrame, min_age: int) -> pd.DataFrame:
    extra = [c for c in ['eta_abs', 'ssm_drift', 'ssm_signal', 'fitted_bp'] if c in res.columns]
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
    trades = standardize_trades(trades_raw)
    trades = trades[(trades['trade_date'] >= pd.Timestamp(CFG.first_oos_date)) & (trades['trade_date'] <= pd.Timestamp(CFG.end_date))]
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
# ## 10. Algo error correction
#
# The v5 print-to-print panel found that the algo's pricing error on a bond persists from one print to the next
# (coefficient 0.58), a larger lever than anything the factor residual offered against the algo quote. This
# section builds it properly.
#
# *Motivation.* Consecutive matched prints of the same bond: the algo error at $t_1$ regressed on the algo error
# at $t_0$, by gap, with the factor and residual moves between them as controls.
#
# *Features, cut at the algo signal time.* For every trade: the algo error at the last matched print strictly
# before `signal_ts`, its age in days, its side, an exponentially weighted mean of past algo errors (half-life
# `err_halflife_prints` prints) and the number of prior prints. Nothing after the signal time is used.
#
# *Baselines and model, walk-forward by month.* (A) the algo quote; (B) algo + $\rho(\text{age})\times$ last
# error, with $\rho$ per age bucket estimated on prior months; (C) algo + $\rho \times$ EWMA error; (D) a
# gradient-boosted level model on the trade's spread to MMD with the algo-derived and print features of v2.3 plus
# the error features; (D$^-$) the same without the error features. All are scored against the print they were
# matched to, by error age and by side, with Fama-MacBeth and block-bootstrap t-statistics.

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
    ec['has_err'] = ec['last_algo_err_bp'].notna()
    ec['err_age'] = ec['recency_days_sig']
    ec['age_bucket'] = pd.cut(ec['err_age'], bins=list(CFG.age_bins_days), labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'no prior print')
    ec['month'] = ec['trade_date'].dt.to_period('M')
    ec['spread_bp'] = 100.0 * (ec['msrb_yield'] - ec['MmdYld'])
    print(f'trades with a prior matched print before the signal time: {ec["has_err"].mean():.1%} of {len(ec):,}')
    feat_panel = model[['cusip', 'date', 'rating_score', 'years_to_worst', 'extension', 'cpn', 'closing_yield_lag1', 'state_others'] + [c for c in CHARS if c != 'market_fv']].rename(columns={'date': 'residual_date'})
    feat_panel = feat_panel[[c for c in feat_panel.columns if c in ('cusip', 'residual_date') or c not in ec.columns]]
    feat_panel['residual_date'] = to_ns(feat_panel['residual_date']); ec['residual_date'] = to_ns(ec['residual_date'])
    ec = ec.merge(feat_panel, on=['cusip', 'residual_date'], how='left')
    ec['recency_days_c'] = ec['recency_days_sig'].clip(upper=60).fillna(60)
    for s in ['D', 'P', 'S']:
        ec[f'side_{s}'] = (ec['side'] == s).astype(float); ec[f'last_side_{s}'] = (ec['last_print_side'] == s).astype(float)
    ec['same_side_as_last'] = (ec['side'] == ec['last_print_side']).astype(float)
    PRINT_FEATURES = ['last_print_spread_bp', 'prints_20d', 'last_side_D', 'last_side_P', 'last_side_S']
    ALGO_DERIVED = ['dMmdSprdSide', 'MinuteFromSignal']
    ERR_FEATURES = ['last_algo_err_bp', 'ewm_algo_err_bp', 'err_age_c', 'n_prior_prints', 'same_side_as_last']
    ec['err_age_c'] = ec['err_age'].clip(upper=60).fillna(60)
    BASE_FEATURES = PRINT_FEATURES + ALGO_DERIVED + ['rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1', 'recency_days_c', 'log_size', 'side_D', 'side_P', 'side_S', 'state_others'] + beta_cols + [c for c in CHARS if c != 'market_fv']
    BASE_FEATURES = [c for c in dict.fromkeys(BASE_FEATURES) if c in ec.columns]
    FULL_FEATURES = BASE_FEATURES + [c for c in ERR_FEATURES if c in ec.columns]

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

    parts, rho_rows, imps = [], [], []
    for m in sorted(ec['month'].unique())[1:]:
        t0 = time.perf_counter()
        tr, te = ec[ec['month'] < m], ec[ec['month'] == m].copy()
        if len(tr) < 5_000 or te.empty:
            continue
        # (B) rho by age bucket, (C) rho for the EWMA error, both on prior months, Fama-MacBeth slopes
        trh = tr[tr['has_err']]
        rho_age = {b: fm_slope(g, 'e_algo_bp', 'last_algo_err_bp', ['log_size'])['fm_beta'] for b, g in trh.groupby('age_bucket') if len(g) >= 500}
        rho_ewm = fm_slope(trh.dropna(subset=['ewm_algo_err_bp']), 'e_algo_bp', 'ewm_algo_err_bp', ['log_size'])['fm_beta']
        rho_rows.append({'month': str(m), **{f'rho[{b}]': v for b, v in rho_age.items()}, 'rho_ewm': rho_ewm})
        te['rho_age'] = te['age_bucket'].map(rho_age).astype(float)
        te['e_B_bp'] = te['e_algo_bp'] - (te['rho_age'] * te['last_algo_err_bp']).fillna(0.0)
        te['e_C_bp'] = te['e_algo_bp'] - (rho_ewm * te['ewm_algo_err_bp']).fillna(0.0)
        # (D) and (D-) level models on the spread to MMD
        yD, impD = fit_gbm(tr, te, FULL_FEATURES, 'spread_bp', CFG.level_model_trees); te['e_D_bp'] = te['spread_bp'] - yD
        yDm, _ = fit_gbm(tr, te, BASE_FEATURES, 'spread_bp', CFG.level_model_trees); te['e_Dminus_bp'] = te['spread_bp'] - yDm
        te['y_mid_D'] = te['MmdYld'] + yD / 100.0
        if impD is not None:
            imps.append(impD)
        parts.append(te); print(f'  {m}: train {len(tr):,} | test {len(te):,} | rho by age {{{", ".join(f"{k}: {v:+.2f}" for k, v in rho_age.items())}}} | rho_ewm {rho_ewm:+.2f} | {time.perf_counter()-t0:.0f}s')
    ecp = pd.concat(parts, ignore_index=True); rho_path = pd.DataFrame(rho_rows).set_index('month')
    print('Error-correction coefficients by month (estimated on prior months):'); display(rho_path.round(3))
    PREDS = {'A algo quote': 'e_algo_bp', 'B algo + rho(age) x last error': 'e_B_bp', 'C algo + rho x EWMA error': 'e_C_bp', 'D level model with error features': 'e_D_bp', 'D- level model without error features': 'e_Dminus_bp'}

    def ec_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size()})
        for k, c in PREDS.items():
            out[f'MAE {k}'] = g[c].apply(lambda s: s.abs().mean())
        for k, c in list(PREDS.items())[1:]:
            d = frame.assign(gain=frame['e_algo_bp'].abs() - frame[c].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
            out[f'gain {k.split()[0]} (bp)'] = d.mean(); out[f't {k.split()[0]}'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
        return out

    ec_all = ec_table(ecp.assign(all='ALL'), 'all'); ec_age = ec_table(ecp, 'age_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d', 'no prior print']).dropna(how='all'); ec_side = ec_table(ecp, 'side'); ec_month = ec_table(ecp, 'month')
    print('Held-out MAE against the print (bp) and daily-FM gain over the algo quote:'); display(ec_all.T.round(3)); display(ec_age.round(3)); display(ec_side.round(3))
    # bootstrap t for the headline gains
    _gd = {k: ecp.assign(gain=ecp['e_algo_bp'].abs() - ecp[c].abs()).groupby('trade_date')['gain'].mean().to_numpy() for k, c in list(PREDS.items())[1:]}
    boot_gain = pd.Series({k: block_bootstrap_t(v, CFG.block_days, CFG.n_boot, CFG.seed) for k, v in _gd.items()}, name='bootstrap t of the daily gain'); display(boot_gain.round(2).to_frame())
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.5))
    ec_age[[f'MAE {k}' for k in PREDS]].plot.bar(ax=ax[0]); ax[0].set_title('MAE vs the print by age of the last algo error (bp)', fontsize=10); ax[0].set_xlabel(''); ax[0].legend([k for k in PREDS], fontsize=7)
    dd = ecp[ecp['has_err']].copy(); dd['decile'] = pd.qcut(dd['last_algo_err_bp'].rank(method='first'), 10, labels=False) + 1
    for sd, g in dd.groupby('side'):
        ax[1].plot(g.groupby('decile')['e_algo_bp'].mean(), marker='o', label={'S': 'S', 'P': 'P', 'D': 'D'}.get(sd, sd))
    ax[1].plot(dd.groupby('decile')['e_algo_bp'].mean(), color='k', lw=2, label='all'); ax[1].plot(dd.groupby('decile')['last_algo_err_bp'].mean(), color='grey', ls='--', label='last error itself'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('decile of the last algo error'); ax[1].set_ylabel('mean algo error at this print (bp)'); ax[1].legend(fontsize=7); ax[1].set_title('Does the algo repeat its last error?', fontsize=10)
    _rc = [c for c in rho_path.columns if c.startswith('rho[')]
    rho_path[_rc].mean().rename(index=lambda s: s[4:-1]).reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).plot.bar(ax=ax[2], color='#8172B2', yerr=rho_path[_rc].std().rename(index=lambda s: s[4:-1]).reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d']), capsize=3)
    ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('rho(age): error-correction coefficient by age of the last print (mean +/- sd across months)', fontsize=10); ax[2].set_xlabel('')
    savefig('10_algo_error_correction')
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    ec_month[[f'MAE {k}' for k in PREDS]].plot(ax=ax[0], marker='o'); ax[0].set_title('Held-out MAE by month (bp)', fontsize=10); ax[0].legend([k for k in PREDS], fontsize=7); ax[0].set_xlabel('')
    if imps:
        pd.concat(imps, axis=1).mean(axis=1).sort_values().tail(15).plot.barh(ax=ax[1], color='#4C72B0'); ax[1].set_title('Level model D: mean feature importance (error features in the list)', fontsize=10)
    else:
        ax[1].text(0.1, 0.5, 'feature importance needs LightGBM', fontsize=12); ax[1].axis('off')
    savefig('10_algo_error_correction_monthly')
    ecp.to_parquet(ARTIFACTS / 'algo_error_correction_v3.parquet', index=False)
    record('error_correction', persistence=pers.round(4).to_dict(orient='records'), persistence_by_gap=pers_gap.round(4).reset_index().to_dict(orient='records'), persistence_by_side_pair=pers_same.round(4).reset_index().to_dict(orient='records'),
           rho_by_month=rho_path.round(4).reset_index().to_dict(orient='records'), overall=ec_all.round(4).to_dict(orient='records'), by_age=ec_age.round(4).reset_index().to_dict(orient='records'), by_side=ec_side.round(4).reset_index().to_dict(orient='records'), bootstrap_t=boot_gain.round(4).to_dict(), features_full=FULL_FEATURES)
else:
    print('Section 10 skipped: needs matched trades joined to residuals.')

# %% [markdown]
# ## 11. Conformal band around the chosen mid
#
# v2.3 fitted a quantile model of |print − mid| and got 71% coverage at an 80% target: the quantile model is
# mis-calibrated under the month-to-month shift in error scale. **Split conformal** fixes this without changing
# the model. For test month $m$, the quantile model is trained on months before $m-1$, the calibration month
# $m-1$ gives the scaling constant $s$ = the target quantile of the normalised score $|e| / \hat q$, and the band
# in month $m$ is $s\,\hat q$. Coverage is then guaranteed on average under exchangeability, and the width still
# adapts to the trade. Three bands at the same target: fixed (calibration-month quantile of $|e|$), raw quantile
# model, and conformalised quantile model, each around two mids: the algo quote and the level model D.

# %%
if HAS_TRADES and 'ecp' in globals() and not ecp.empty:
    BAND_FEATURES = [c for c in ['log_size', 'side_D', 'side_P', 'side_S', 'recency_days_c', 'prints_20d', 'MinuteFromSignal', 'last_print_spread_bp', 'last_algo_err_bp', 'ewm_algo_err_bp', 'abs_resid_z', 'eta_abs', 'modified_duration_lag1', 'rating_score', 'target_abs_vol', 'dMmdSprdSide'] if c in ecp.columns]
    bands, bimp = [], []
    months_b = sorted(ecp['month'].unique())
    for mid_name, err_col in [('algo quote', 'e_algo_bp'), ('level model D', 'e_D_bp')]:
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
            bands.append(te[['cusip', 'trade_date', 'side', 'age_bucket', 'recency_bucket', 'month', 'mid', 'abs_err', 'hw_fixed', 'hw_quantile', 'hw_conformal', 's_conformal']])
    if bands:
        bd = pd.concat(bands, ignore_index=True); bd['month'] = bd['month'].astype(str)
        for k in ['fixed', 'quantile', 'conformal']:
            bd[f'cov_{k}'] = (bd['abs_err'] <= bd[f'hw_{k}']).astype(float)

        def band_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
            g = frame.groupby(by, observed=True)
            return pd.DataFrame({'n': g.size(), **{f'coverage {k}': g[f'cov_{k}'].mean() for k in ['fixed', 'quantile', 'conformal']}, **{f'half-width {k} (bp)': g[f'hw_{k}'].mean() for k in ['fixed', 'quantile', 'conformal']}})

        for mid_name, g in bd.groupby('mid'):
            print(f'Band around the {mid_name}, target {CFG.band_coverage:.0%} (held-out months; conformal scale s by month: {g.groupby("month")["s_conformal"].first().round(2).to_dict()})')
            display(band_table(g.assign(all='ALL'), 'all').round(3)); display(band_table(g, 'month').round(3)); display(band_table(g, 'side').round(3))
        gD = bd[bd['mid'] == 'level model D'].copy(); gD['width_decile'] = pd.qcut(gD['hw_conformal'].rank(method='first'), 10, labels=False) + 1
        dec = band_table(gD, 'width_decile')
        print('Calibration by conformal-width decile, level model D mid (flat at the target = calibrated):'); display(dec[['n', 'coverage fixed', 'coverage quantile', 'coverage conformal', 'half-width conformal (bp)']].round(3))
        fig, ax = plt.subplots(1, 3, figsize=(19, 4.3))
        for k, st in [('fixed', 's--'), ('quantile', '^-.'), ('conformal', 'o-')]:
            ax[0].plot(dec.index, dec[f'coverage {k}'], st, label=k)
        ax[0].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8); ax[0].set_ylim(0.4, 1.0); ax[0].set_xlabel('conformal half-width decile'); ax[0].legend(fontsize=8); ax[0].set_title('Coverage by predicted-width decile (level model D mid)', fontsize=10)
        bm = bd.groupby(['mid', 'month'])[['cov_fixed', 'cov_quantile', 'cov_conformal']].mean()
        for mid_name, st in [('algo quote', '--'), ('level model D', '-')]:
            for k, c in [('fixed', 'grey'), ('conformal', '#4C72B0')]:
                ax[1].plot(bm.loc[mid_name].index, bm.loc[mid_name][f'cov_{k}'], st, color=c, marker='o', label=f'{k}, {mid_name}')
        ax[1].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8); ax[1].legend(fontsize=7); ax[1].set_title('Coverage by month: fixed vs conformal, two mids', fontsize=10)
        bw = bd.groupby(['mid', 'side'])[['hw_fixed', 'hw_conformal']].mean()
        bw.plot.bar(ax=ax[2]); ax[2].set_title('Mean half-width by mid and side (bp)', fontsize=10); ax[2].set_xlabel(''); ax[2].legend(['fixed', 'conformal'], fontsize=8)
        savefig('11_conformal_band')
        bd.to_parquet(ARTIFACTS / 'conformal_band_v3.parquet', index=False)
        record('conformal_band', features=BAND_FEATURES, by_mid={mid_name: band_table(g.assign(all='ALL'), 'all').round(4).to_dict(orient='records')[0] for mid_name, g in bd.groupby('mid')}, by_width_decile=dec.round(4).reset_index().to_dict(orient='records'))
    else:
        print('Section 11: not enough months for train / calibration / test.')
else:
    print('Section 11 skipped: needs Section 10.')

# %% [markdown]
# ## 12. Systematic path: factor roll-forward of stale marks
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
savefig('12_roll_forward')
record('systematic_path', roll_forward=rollf.round(4).reset_index().to_dict(orient='records'))

# %% [markdown]
# ## 13. Results registry and summary

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
    summary_rows.append(('Held-out MAE vs print: A algo | B rho(age) | C EWMA | D level+err | D- level', ' | '.join(f"{_o[f'MAE {k}']:.2f}" for k in PREDS) + ' bp'))
    summary_rows.append(('Gain over algo (bp, daily FM t, bootstrap t)', '; '.join(f"{k.split()[0]}: {_o[f'gain {k.split()[0]} (bp)']:+.2f} (t {_o[f't {k.split()[0]}']:+.1f}, boot {boot_gain[k]:+.1f})" for k in list(PREDS)[1:])))
if 'conformal_band' in REGISTRY:
    for mid_name, r in REGISTRY['conformal_band']['by_mid'].items():
        summary_rows.append((f'Band around {mid_name} ({CFG.band_coverage:.0%} target)', f"fixed {r['coverage fixed']:.1%} at {r['half-width fixed (bp)']:.1f} bp | quantile {r['coverage quantile']:.1%} at {r['half-width quantile (bp)']:.1f} | conformal {r['coverage conformal']:.1%} at {r['half-width conformal (bp)']:.1f}"))
summary_rows.append(('Roll-forward RMSE reduction, h=1 / h=10', f"{rollf.loc[1, 'rmse_reduction']:.0%} / {rollf.loc[10, 'rmse_reduction']:.0%}"))
summary = pd.DataFrame(summary_rows, columns=['item', 'result']).set_index('item')
display(summary)
print('Artifacts written to', ARTIFACTS)
for p in sorted(ARTIFACTS.glob('*')):
    if p.is_file():
        print('  ', p.name, f'{p.stat().st_size/1e6:.2f} MB')
print('Figures:', len(list(FIGURES.glob('*.png'))))

# %% [markdown]
# ## 14. Robustness: dispersion-day exclusions, bootstrap inference, pre-registered choices

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
