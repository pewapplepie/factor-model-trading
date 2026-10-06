# %% [markdown]
# # Muni Characteristic Factor Model v2.3 — IPCA Residuals, Level Relative Value, Market-Making Signals
#
# Clean rebuild of the research chain on the regenerated pipeline data (2026-01 to 2026-09).
#
# **Inputs** (produced by `data_pipeline/muni_data_pipeline.py`, all under `./data_pipeline/`):
#
# | Artifact | Use |
# |---|---|
# | `data/panel/*.parquet` | Full-universe evaluated marks + static fields (read with pushdown) |
# | `data/closing_marks/*.parquet` | OneTick close marks, deduplicated at read |
# | `data/algosignal_msrb/*.parquet` | AlgoSignal ↔ MSRB matched trades (validation + level model) |
# | `research_panel_step3_yield.parquet` | Covered yield-space panel: the modelling input |
#
# **Outputs** go to `./artifacts_v2/` (residuals, scores, Gamma per version, figures, `results_registry.json`).
#
# **Sections**
# 1. Setup and configuration
# 2. Data load and QA (coverage, rating fallback, extreme-yield handling)
# 3. Model panel: 13 instruments, rank-normalised per date
# 4. IPCA: K sweep, Gamma anatomy, factor paths
# 5. Walk-forward (point-in-time) residuals and Gamma stability
# 6. Factor-mimicking weights and the residual-maker identity
# 7. Residual diagnostics: ACF, activity buckets, AR(1) vs LongConv-lite, two-regime check
# 8. Paper portfolio: is the residual path harvestable as a hedged portfolio?
# 9. Transaction validation (PIT-safe, Fama-MacBeth, recency, side, neutralised)
# 10. Level relative value: walk-forward GBM on trade spreads vs algo quote and prior mark
# 10b. Application: shrinkage mid, residual-driven half-width, concession (walk-forward)
# 6c. Peer relative value: point-in-time comparables in characteristic space
# 7. (cont.) State-space and ARMA forecasters, implied Kalman gains by bucket
# 9b. Print-to-print error correction: does the next print follow the factor move, the residual, or neither?
# 9c. Economic sizing: the slow-horizon skew applied to the algo quote, walk-forward
# 10c. Conditional half-width: a quantile model of print dispersion around the model mid
# 11. Systematic path: factor roll-forward and de-circularised beta-space comparables
# 11b. Factor exposure of dealer flow: how much inventory risk is factor risk
# 12. Results registry and summary
# 13. Robustness: K sensitivity, repricing-day exclusions, bootstrap inference, pre-registered choices
#
# v2.3 changes (after the v4 run review): date-weighted IPCA with repricing days excluded from the fit and a
# Gamma swap test; rotation-invariant leverage diagnostic; point-in-time peer relative value in characteristic
# space (6c); pooled state-space forecaster (AR drift + mark-noise MA) and pooled ARMA(1,1) baseline alongside
# the ridge filter, with implied Kalman gains by activity bucket (7); vol- and rank-scaled paper portfolios and
# an ex-UNKNOWN universe (8); block-bootstrap t-stats, within-duration ranks and peer-RV scores in the
# transaction tests (9); print-to-print error-correction panel scoring the roll-forward against prints (9b);
# walk-forward economic sizing of the slow-horizon skew on the algo quote (9c); level-model ablations and
# mark-staleness / peer / mark-noise features (10); conditional half-width from a quantile model (10c);
# factor exposure of dealer flow (11b); robustness section with K sensitivity, repricing-day exclusions and the
# pre-registered choices (13).
# v2.2 changes: print features cut at the algo signal time; Procrustes-aligned Gamma versions; joint
# one-day / trailing-mean regression; MSRB side convention (S = dealer sale to customer, P = dealer purchase
# from customer, D = inter-dealer); paper portfolio run on every forecaster with walk-forward selection and a
# fade variant; model-as-mid application with inverse-variance mark weight; market-context, residual-dispersion,
# decile and calibration exhibits.
# v2.1 changes: duration enters as the ratio of modified duration to years-to-worst (breaks the 0.98 rank
# correlation between the two level instruments that produced a single leveraged factor); a trailing-mean
# residual is tested as the level signal; the transaction section adds the spread reading (absolute gap,
# dealer-to-customer round trip) and the evaluator-revision test; the level model gets print features; and
# Section 10b turns the findings into the three quote outputs.
#
# Every leakage boundary is explicit: Gamma is refit weekly on past dates only, instruments are lagged and
# rank-normalised within date, residual thresholds come from training folds, and every trade is matched to a
# residual dated strictly before the trade date.

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
ARTIFACTS = ROOT / 'artifacts_v2'
FIGURES = ARTIFACTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PIPELINE_ROOT))
import muni_data_pipeline as mdp  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    spec_version: str = 'step3_yield_ipca_k3_v3'
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
    shrink_lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    fade_buckets: tuple[str, ...] = ('L4', 'L5')   # activity buckets where the trailing-mean signal is faded
    band_coverage: float = 0.80
    # --- v2.3 ---
    date_weighting: str = 'inverse_var'   # 'none' | 'inverse_var': each date's moments weighted by 1 / its cross-sectional mean square of the target
    date_weight_cap: float = 4.0          # a date can weigh at most this multiple of the median date
    repricing_bp: float = 10.0            # a repricing day is one where this share of bonds ...
    repricing_share_cut: float = 0.10     # ... moves by more than repricing_bp (evaluator-wide reprice, not a market move)
    exclude_repricing_from_fit: bool = True
    portfolio_scaling: str = 'vol'        # 'raw' | 'vol' | 'rank': how a forecast becomes a weight (Section 8)
    block_days: int = 5                   # moving-block bootstrap over trade dates for Fama-MacBeth inference
    n_boot: int = 500
    peer_k: int = 20                      # neighbours for the point-in-time peer RV (Section 6c)
    ssm_fit_bonds: int = 2000             # bonds subsampled per state-space fit
    ssm_maxiter: int = 120
    level_model_ablation: bool = True
    ablation_trees: int = 300
    robustness_k: tuple[int, ...] = (4,)  # extra walk-forward runs for the K sensitivity check (Section 13)
    seed: int = 20260921


# Pre-registered choices. These were fixed on 2026-10-06 after the v4 run and before the v2.3 run; they are not
# tuned on the data this notebook scores. Section 13 reports them so the next reader can tell a choice from a fit.
PRE_REGISTERED = {
    'fixed_on': '2026-10-06',
    'selected_k': 3, 'refit_days': 7, 'level_signal_window': 20, 'activity_window': 20, 'longconv_lags': 10,
    'fade_buckets': ['L4', 'L5'], 'repricing_rule': 'share of |target| > 10 bp above 10% of bonds', 'date_weighting': 'inverse variance, cap 4x',
    'duration_instrument': 'ratio', 'band_coverage': 0.80, 'peer_k': 20, 'portfolio_scaling': 'vol',
    'economic_sizing_rule': 'skew = prior-month FM slope x (trailing-mean rank - 0.5), applied to the algo quote',
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
# Repricing days. On a handful of dates a large share of the universe moves by more than 10 bp at once: an evaluator-wide
# reprice (curve rebase, month-end), not a cross-section of bond-specific news. Those dates dominate unweighted ALS
# moments and the negative lag-1 autocorrelation. They are flagged here, excluded from the fit (configurable), and
# every residual test is repeated without them in Section 13.
_rp = model.groupby('date', observed=True)['target_bp_raw'].apply(lambda s: float((s.abs() > CFG.repricing_bp).mean()))
REPRICING_DATES = set(pd.DatetimeIndex(_rp[_rp > CFG.repricing_share_cut].index))
model['repricing_day'] = model['date'].isin(REPRICING_DATES).astype(float)
print(f'repricing days (> {CFG.repricing_share_cut:.0%} of bonds move > {CFG.repricing_bp:.0f} bp): {sorted(d.date() for d in REPRICING_DATES)}')
print(f'rows on repricing days: {model["repricing_day"].mean():.2%} | mean |target| on those days {model.loc[model["repricing_day"] == 1, "target_bp_raw"].abs().mean():.1f} bp vs {model.loc[model["repricing_day"] == 0, "target_bp_raw"].abs().mean():.1f} bp otherwise')
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].plot(_rp.index, _rp.values, color='#C44E52', lw=1); ax[0].axhline(CFG.repricing_share_cut, color='k', ls='--', lw=0.8); ax[0].set_title(f'Share of bonds with |daily yield change| > {CFG.repricing_bp:.0f} bp'); ax[0].axvline(pd.Timestamp(CFG.first_oos_date), color='grey', ls=':', lw=0.8)
_ms = model.groupby('date', observed=True)['target_bp_raw'].apply(lambda s: float((s ** 2).mean()))
ax[1].semilogy(_ms.index, _ms.values, color='#4C72B0', lw=1); ax[1].set_title('Cross-sectional mean square of the target (bp²): the weight an unweighted ALS gives a date')
for d in REPRICING_DATES:
    ax[1].axvline(d, color='#C44E52', lw=0.6, alpha=0.6)
savefig('03_repricing_days')
record('model_panel', repricing_days=[str(d.date()) for d in sorted(REPRICING_DATES)], repricing_row_share=float(model['repricing_day'].mean()))

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
    target (capped), so one 20 bp day does not carry 50 normal days; repricing days get weight zero when excluded."""
    w = np.ones(len(dates))
    if cfg.date_weighting == 'inverse_var':
        ms = rr / np.maximum(nobs, 1)
        w = np.clip(np.median(ms[ms > 0]) / np.maximum(ms, 1e-12), 0.0, cfg.date_weight_cap)
    if cfg.exclude_repricing_from_fit and repricing:
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
    w = date_weights(dates, rr, nobs, CFG, REPRICING_DATES) if weighted else np.ones(len(dates))
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
# ## 5. Walk-forward (point-in-time) residuals
#
# Gamma is refit every `refit_days` on dates strictly before each test week, starting from `first_oos_date`.
# Each residual carries the `gamma_version` (train end) that produced it. Everything downstream uses these rows.

# %%
def make_weekly_folds(dates: pd.DatetimeIndex, train_start: str, first_oos: str, test_days: int) -> list[dict]:
    dates = pd.DatetimeIndex(dates).sort_values()
    ts, start = pd.Timestamp(train_start), pd.Timestamp(first_oos)
    folds = []
    k = 1
    while start <= dates.max():
        end = start + pd.Timedelta(days=test_days)
        tr = dates[(dates >= ts) & (dates < start)]; te = dates[(dates >= start) & (dates < end)]
        if len(tr) and len(te):
            folds.append({'fold': k, 'train_dates': tr, 'test_dates': te, 'train_end': tr[-1], 'test_start': te[0], 'test_end': te[-1]}); k += 1
        start = end
    return folds


def procrustes_align(G: np.ndarray, G_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate G onto G_ref within its own column span (orthogonal Procrustes). Residuals are unchanged; factor
    labels, betas and factor realisations become continuous across refits instead of swapping when eigenvalues cross."""
    U, _, Vt = np.linalg.svd(G.T @ G_ref)
    R = U @ Vt
    return G @ R, R


def walk_forward_residuals(frame: pd.DataFrame, chars: list[str], folds: list[dict], cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts, gammas, gammas_raw = [], [], []
    gamma_init = None
    G_ref = None
    by_date = {d: g for d, g in frame.groupby('date', sort=True, observed=True)}
    eye = np.eye(cfg.selected_k)
    fcols = [f'factor{i+1}' for i in range(cfg.selected_k)]
    for fo in folds:
        t0 = time.perf_counter()
        train = frame[frame['date'].isin(fo['train_dates'])]
        mk = MomentIPCA(cfg.selected_k, cfg.max_iter, cfg.tol, cfg.ridge, cfg.seed).fit(precompute_moments(train, chars, TARGET), gamma_init=gamma_init)
        gamma_init = mk.Gamma
        gv = pd.Timestamp(fo['train_end'])
        gammas_raw.append(pd.DataFrame(mk.Gamma, index=chars, columns=fcols).assign(gamma_version=gv, fold=fo['fold']))
        G_al = mk.Gamma if G_ref is None else procrustes_align(mk.Gamma, G_ref)[0]
        G_ref = G_al
        gammas.append(pd.DataFrame(G_al, index=chars, columns=fcols).assign(gamma_version=gv, fold=fo['fold']))
        for d in fo['test_dates']:
            g = by_date.get(d)
            if g is None:
                continue
            Z = g[chars].to_numpy(float); r = g[TARGET].to_numpy(float)
            B = Z @ G_al
            f = np.linalg.solve(B.T @ B + cfg.ridge * eye, B.T @ r)
            out = g[['cusip', 'date']].copy()
            out['gamma_version'] = gv; out['fold'] = fo['fold']; out['target_bp'] = r; out['fitted_bp'] = B @ f; out['pit_residual'] = r - out['fitted_bp']
            for j in range(cfg.selected_k):
                out[f'beta{j+1}'] = B[:, j]; out[f'f{j+1}'] = f[j]
            parts.append(out)
        print(f'fold {fo["fold"]:02d} | train ->{fo["train_end"].date()} | test {fo["test_start"].date()}..{fo["test_end"].date()} | {time.perf_counter()-t0:.1f}s')
    return pd.concat(parts, ignore_index=True), pd.concat(gammas), pd.concat(gammas_raw)


folds = make_weekly_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CFG.refit_days)
print(f'{len(folds)} walk-forward folds; first test {folds[0]["test_start"].date()}, last test {folds[-1]["test_end"].date()}')
t0 = time.perf_counter()
resid, gamma_versions, gamma_versions_raw = walk_forward_residuals(model, CHARS, folds, CFG)
print(f'PIT residuals: {len(resid):,} rows | {resid["cusip"].nunique():,} cusips | {resid["date"].nunique()} dates | {resid["gamma_version"].nunique()} gamma versions | {time.perf_counter()-t0:.1f}s')
oos_var_expl = 1.0 - resid['pit_residual'].var() / resid['target_bp'].var()
print(f'OOS variance explained {oos_var_expl:.3f} | residual sd {resid["pit_residual"].std():.2f} bp')
resid.to_parquet(ARTIFACTS / 'pit_ipca_residuals_v2.parquet', index=False)
gamma_versions.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_v2.parquet', index=False)
gamma_versions_raw.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_raw_v2.parquet', index=False)
record('walk_forward', folds=len(folds), residual_rows=len(resid), cusips=resid['cusip'].nunique(), oos_dates=resid['date'].nunique(), gamma_versions=resid['gamma_version'].nunique(), oos_variance_explained=float(oos_var_expl), residual_sd_bp=float(resid['pit_residual'].std()))

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
# Gamma swap test. The aligned loadings move once, at the first version whose training window contains a repricing
# day. Was that a better Gamma for the months that followed, or an artefact? Score each OOS month under (a) the
# version in force (what the residuals use), (b) a Gamma frozen two months earlier, (c) the first OOS Gamma.
MODEL_BY_DATE = {d: g for d, g in model.groupby('date', sort=True, observed=True)}
GAMMA_ALIGNED = {gv: g.drop(columns=['gamma_version', 'fold']).to_numpy(float) for gv, g in gamma_versions.groupby('gamma_version', sort=True)}
GAMMA_RAW = {gv: g.drop(columns=['gamma_version', 'fold']).to_numpy(float) for gv, g in gamma_versions_raw.groupby('gamma_version', sort=True)}
_gv_index = pd.DatetimeIndex(sorted(GAMMA_ALIGNED))


def var_explained_under(G: np.ndarray, dates) -> float:
    sse = sst = 0.0
    for d in dates:
        g = MODEL_BY_DATE.get(pd.Timestamp(d))
        if g is None:
            continue
        Z = g[CHARS].to_numpy(float); r = g[TARGET].to_numpy(float); B = Z @ G
        f = np.linalg.solve(B.T @ B + CFG.ridge * np.eye(B.shape[1]), B.T @ r); e = r - B @ f
        sse += float(e @ e); sst += float(r @ r)
    return 1.0 - sse / max(sst, 1e-12)


swap_rows = []
for mth, g in resid.groupby(resid['date'].dt.to_period('M')):
    dts = sorted(g['date'].unique())
    frozen_2m = _gv_index[_gv_index <= pd.Timestamp(dts[0]) - pd.Timedelta(days=60)]
    row = {'month': str(mth), 'in force': 1 - g['pit_residual'].var() / g['target_bp'].var(), 'first OOS Gamma': var_explained_under(GAMMA_ALIGNED[_gv_index[0]], dts)}
    row['frozen 2 months earlier'] = var_explained_under(GAMMA_ALIGNED[frozen_2m[-1]], dts) if len(frozen_2m) else np.nan
    swap_rows.append(row)
gswap = pd.DataFrame(swap_rows).set_index('month')
print('Gamma swap test: OOS variance explained by month under the Gamma in force vs frozen versions'); display(gswap.round(3))
fig, ax = plt.subplots(figsize=(9, 3.8))
gswap.plot.bar(ax=ax); ax.set_title('OOS variance explained: Gamma in force vs frozen Gammas (a gap says the refit mattered)'); ax.set_xlabel(''); ax.legend(fontsize=8)
savefig('05_gamma_swap_test')
record('walk_forward', gamma_swap_test=gswap.round(4).reset_index().to_dict(orient='records'))

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
    lev_rows.append({'date': d, 'frobenius_leverage': float(np.sqrt(np.trace(BtB_inv))), 'n': len(g), 'repricing': d in REPRICING_DATES})
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
# ## 6b. Factor space: clusters, top loadings and within-cluster ranking
#
# The exhibits the attention-factor paper uses to make latent factors legible, adapted to munis. Bonds are
# placed in standardised beta space on the last out-of-sample date. (1) A 2-D map of that space coloured by
# rating, call structure, duration and residual rank shows which economic axes the factors organise.
# (2) The bonds with the largest weight in each factor-mimicking portfolio, labelled by issuer, coupon,
# maturity, rating and call type rather than CUSIP, show what each factor is made of. (3) K-means clusters in
# beta space give the peer groups the model implies, with their characteristic profile and named examples.
# (4) Inside one cluster, the residual orders bonds rich to cheap: that ordering is the comparables view an
# RFQ desk uses.

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
# Top loadings per factor: the bonds carrying the largest factor-mimicking weight, named not numbered
B_last = snap[beta_cols].to_numpy(float)
W_last = B_last @ np.linalg.inv(B_last.T @ B_last + CFG.ridge * np.eye(CFG.selected_k))
fig, ax = plt.subplots(1, CFG.selected_k, figsize=(6.5 * CFG.selected_k, 6))
top_tables = {}
for j in range(CFG.selected_k):
    w = pd.Series(W_last[:, j], index=snap.index)
    top = w.reindex(w.abs().sort_values(ascending=False).index[:12]).sort_values()
    t = snap.loc[top.index, ['label', 'modified_duration_lag1', 'years_to_worst', 'rating', 'call_structure', 'closing_yield']].assign(weight=top.values, beta=snap.loc[top.index, beta_cols[j]].values)
    top_tables[f'factor{j+1}'] = t.reset_index(drop=True)
    a = ax[j] if CFG.selected_k > 1 else ax
    a.barh(t['label'], t['weight'], color=np.where(t['weight'] >= 0, '#2CA02C', '#D62728')); a.axvline(0, color='k', lw=0.6)
    load = gamma[f'factor{j+1}'].abs().sort_values(ascending=False).index[:3]
    a.set_title(f'factor{j+1}: top mimicking weights\n(Gamma: ' + ', '.join(f"{c.replace('z_', '').replace('_lag1', '')} {gamma.loc[c, f'factor{j+1}']:+.2f}" for c in load) + ')', fontsize=10)
    a.tick_params(axis='y', labelsize=8)
savefig('06b_top_loadings_per_factor')
for k, t in top_tables.items():
    print(k); display(t[['label', 'weight', 'beta', 'modified_duration_lag1', 'years_to_worst', 'closing_yield']].round(4))

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
record('factor_space', date=str(last_date.date()), embedding=emb_name, clusters=N_CLUSTERS, cluster_profile=profile.round(4).reset_index().to_dict(orient='records'),
       top_loadings={k: t[['label', 'weight', 'beta']].round(4).to_dict(orient='records') for k, t in top_tables.items()})

# %% [markdown]
# ## 6c. Peer relative value: point-in-time comparables in characteristic space
#
# The muni relative-value literature prices a bond off its comparables: bonds that are alike in the
# characteristics that drive spread, with the fair level read from how those peers trade or are marked. The
# supervised version learns the similarity from a spread model; the version here is its unsupervised baseline,
# computed point-in-time on every OOS date: nearest neighbours in the standardised instrument space **without**
# the yield-level and premium/discount instruments (so peers are not close in yield by construction), and the
# peer-implied yield is the distance-weighted mean of their closing yields. The peer spread, own yield minus
# peer-implied yield, is a *level* relative-value score that does not depend on the factor model at all. It is
# used three ways: as a forecaster in Section 7, as a score in the transaction tests, and as a feature in the
# level model.

# %%
from sklearn.neighbors import NearestNeighbors  # noqa: E402

PEER_CHARS = [c for c in CHARS if c not in ('market_fv', 'z_closing_yield_lag1', 'z_premium_discount_lag1')] + ['state_others']


def peer_rv_by_date(frame: pd.DataFrame, k: int, cols: list[str]) -> pd.DataFrame:
    parts = []
    for d, g in frame.groupby('date', sort=True, observed=True):
        if len(g) < k + 5:
            continue
        X = g[cols].to_numpy(float); X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X); dist, idx = dist[:, 1:], idx[:, 1:]
        w = 1.0 / (dist + 1e-6); w = w / w.sum(axis=1, keepdims=True)
        y = g['closing_yield'].to_numpy(float); peer = (y[idx] * w).sum(axis=1)
        rand = y[RNG.integers(0, len(g), size=idx.shape)].mean(axis=1)
        parts.append(pd.DataFrame({'cusip': g['cusip'].to_numpy(), 'date': d, 'peer_yield': peer, 'peer_spread_bp': 100.0 * (y - peer), 'peer_dist': dist.mean(axis=1), 'random_peer_abs_err_bp': 100.0 * np.abs(y - rand)}))
    out = pd.concat(parts, ignore_index=True); out['cusip'] = out['cusip'].astype('string')
    return out


t0 = time.perf_counter()
peer_pit = peer_rv_by_date(model[model['date'] >= pd.Timestamp(CFG.first_oos_date)].dropna(subset=['closing_yield']), CFG.peer_k, PEER_CHARS)
print(f'peer RV: {len(peer_pit):,} bond-days on {peer_pit["date"].nunique()} OOS dates, k={CFG.peer_k}, {len(PEER_CHARS)} characteristics | {time.perf_counter()-t0:.1f}s')
resid = resid.merge(peer_pit[['cusip', 'date', 'peer_yield', 'peer_spread_bp', 'peer_dist']], on=['cusip', 'date'], how='left')
resid['peer_rank'] = resid.groupby('date', observed=True)['peer_spread_bp'].rank(pct=True)
_pq = peer_pit.groupby('date').agg(peer_mae_bp=('peer_spread_bp', lambda s: s.abs().mean()), random_mae_bp=('random_peer_abs_err_bp', 'mean'), n=('cusip', 'size'))
_pc = resid.dropna(subset=['peer_spread_bp']).groupby('date').apply(lambda g: g['peer_spread_bp'].corr(g['pit_residual'], method='spearman'), include_groups=False)
print(f'peer-implied yield MAE {_pq["peer_mae_bp"].mean():.1f} bp vs random peers {_pq["random_mae_bp"].mean():.1f} bp | mean daily Spearman(peer spread, one-day residual) {_pc.mean():+.3f}')
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
_pq[['peer_mae_bp', 'random_mae_bp']].plot(ax=ax[0]); ax[0].set_title('Replication quality by date: |own - peer-implied yield| (bp)'); ax[0].set_xlabel('')
_last = resid[resid['date'] == resid['date'].max()].dropna(subset=['peer_yield'])
_own = model.loc[model['date'] == resid['date'].max(), ['cusip', 'closing_yield']].merge(_last[['cusip', 'peer_yield']], on='cusip')
_lim = [np.nanpercentile(_own['closing_yield'], 1), np.nanpercentile(_own['closing_yield'], 99)]
hb = ax[1].hexbin(_own['peer_yield'], _own['closing_yield'], gridsize=45, cmap='viridis', mincnt=1, bins='log'); ax[1].plot(_lim, _lim, 'k--', lw=1); ax[1].set_xlabel('peer-implied yield (%)'); ax[1].set_ylabel('own closing yield (%)'); ax[1].set_title(f'Characteristic-space peers, {resid["date"].max().date()}')
ax[2].plot(_pc.index, _pc.values, color='#8172B2'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Daily Spearman: peer spread vs one-day residual'); ax[2].set_xlabel('')
savefig('06c_peer_rv')
record('peer_rv', k=CFG.peer_k, chars=PEER_CHARS, peer_mae_bp=float(_pq['peer_mae_bp'].mean()), random_mae_bp=float(_pq['random_mae_bp'].mean()), corr_with_residual=float(_pc.mean()))

# %% [markdown]
# ## 7. Residual diagnostics: autocorrelation, activity buckets, AR(1) vs LongConv-lite
#
# The pooled autocorrelation function of the residual at lags 1..10 is the first look at whether the residual
# path carries information. Activity buckets are pre-OOS volatility quintiles (no look-ahead). Three forecasters
# are compared on expanding monthly folds: raw AR(1), winsorised AR(1), and a pooled ridge regression on the
# last `longconv_lags` residuals (a long convolution without the nonlinearity).

# %%
resid = resid.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
grp = resid.groupby('cusip', observed=True)
resid['repricing_day'] = resid['date'].isin(REPRICING_DATES)
acf_rows = []
for lag in range(1, CFG.longconv_lags + 1):
    lagged = grp['pit_residual'].shift(lag); lagged_rp = grp['repricing_day'].shift(lag)
    ok = lagged.notna()
    x, y = lagged[ok].to_numpy(), resid.loc[ok, 'pit_residual'].to_numpy()
    ex = ok & ~resid['repricing_day'] & ~lagged_rp.fillna(True).astype(bool)
    xe, ye = lagged[ex].to_numpy(), resid.loc[ex, 'pit_residual'].to_numpy()
    acf_rows.append({'lag': lag, 'pearson': np.corrcoef(x, y)[0, 1], 'spearman': pd.Series(x).corr(pd.Series(y), method='spearman'), 'pairs': int(ok.sum()), 'bartlett_ci': 1.96 / np.sqrt(max(int(ok.sum()), 1)),
                     'pearson_ex_repricing': np.corrcoef(xe, ye)[0, 1] if ex.sum() > 100 else np.nan, 'spearman_ex_repricing': pd.Series(xe).corr(pd.Series(ye), method='spearman') if ex.sum() > 100 else np.nan})
acf = pd.DataFrame(acf_rows).set_index('lag')
display(acf)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].bar(acf.index - 0.2, acf['pearson'], width=0.4, label='Pearson'); ax[0].bar(acf.index + 0.2, acf['spearman'], width=0.4, label='Spearman')
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('lag (observations)'); ax[0].set_title('Pooled residual autocorrelation (within CUSIP), all days'); ax[0].legend()
ax[1].bar(acf.index - 0.2, acf['pearson_ex_repricing'], width=0.4, label='Pearson'); ax[1].bar(acf.index + 0.2, acf['spearman_ex_repricing'], width=0.4, label='Spearman')
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('lag (observations)'); ax[1].set_title(f'Same, excluding pairs that touch a repricing day ({len(REPRICING_DATES)} dates)'); ax[1].legend()
savefig('07_residual_acf')
record('residual_diagnostics', acf_pearson=acf['pearson'].round(4).to_dict(), acf_spearman=acf['spearman'].round(4).to_dict(), acf_pearson_ex_repricing=acf['pearson_ex_repricing'].round(4).to_dict())

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
# Forecasters on expanding monthly folds: AR(1) raw, AR(1) winsorised, LongConv-lite (ridge on L lags, winsorised)
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
    ex = pred[~pred['next_date'].isin(REPRICING_DATES) & ~pred['date'].isin(REPRICING_DATES)]
    return {'forecaster': name, 'rows': len(pred), 'months': pred['month'].nunique(), **ic_summary(ic), 'oos_r2_vs_zero': 1 - ((y - yh) ** 2).sum() / (y ** 2).sum(),
            'sign_acc': float((np.sign(y) == np.sign(yh))[(y != 0) & (yh != 0)].mean()), 'ic_ex_repricing': ic_summary(daily_rank_ic(ex, 'yhat', 'next_residual'))['ic_mean'] if len(ex) > 1000 else np.nan}


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


def fit_arma11_taps(X: np.ndarray, y: np.ndarray, max_rows: int = 300_000):
    """Pooled ARMA(1,1) for the increment through its truncated AR(inf) taps pi_j = (phi + theta)(-theta)^(j-1), j = 1..L.
    Two parameters on the same lag design the ridge uses. Note the shape it can express: geometric taps of one sign or
    alternating sign; it cannot produce 'negative at lag 1, flat positive after', which is why it is a baseline."""
    L_ = X.shape[1]; sub = RNG.choice(len(X), min(len(X), max_rows), replace=False); Xs, ys = X[sub], y[sub]; jj = np.arange(L_)

    def taps(p):
        return (p[0] + p[1]) * ((-p[1]) ** jj)

    def resfn(p):
        t = taps(p); return Xs @ t + (ys.mean() - Xs.mean(axis=0) @ t) - ys

    best = None
    for x0 in ([0.9, -0.85], [0.5, -0.3], [0.1, 0.1], [-0.3, 0.2]):
        sol = least_squares(resfn, x0=x0, bounds=([-0.999, -0.999], [0.999, 0.999]))
        if best is None or sol.cost < best.cost:
            best = sol
    t = taps(best.x); c = float(y.mean() - X.mean(axis=0) @ t)
    return (lambda Xn: c + Xn @ t), np.r_[c, t], {'phi': float(best.x[0]), 'theta': float(best.x[1])}


from scipy.optimize import least_squares, minimize  # noqa: E402

fc_results, fc_preds, fc_coefs = [], {}, {}
FC_SPECS = [('AR(1) raw', ['lag1'], False, None), ('AR(1) winsor', ['lag1'], True, None), (f'Trailing-{W} mean (level signal)', ['resid_mean'], True, None),
            (f'LongConv-lite ridge L={L} winsor', lag_cols, True, None), ('ARMA(1,1) pooled, winsor', lag_cols, True, fit_arma11_taps),
            ('Peer RV spread (chars-space kNN)', ['peer_spread_bp'], True, None)]
for name, cols, winsor, fitter in FC_SPECS:
    t0 = time.perf_counter()
    pred, summ, coefs = forecaster_eval(name, cols, winsor, fitter=fitter)
    fc_results.append(summ); fc_preds[name] = pred; fc_coefs[name] = coefs
    print(f'{name}: {time.perf_counter()-t0:.1f}s', '' if 'params' not in summ else summ['params'])

# %% [markdown]
# ### 7b. State-space forecaster: AR drift plus mark-noise, pooled Kalman filter
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
# active names is what the data imply. Increments are winsorised at the training quantiles before filtering.

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
_pex = pairs[~pairs['date'].isin(REPRICING_DATES) & ~pairs['next_date'].isin(REPRICING_DATES)]
two_regime_ex = _pex.groupby('res_bin', observed=True).agg(n=('next_residual', 'size'), mean_now=('pit_residual', 'mean'), mean_next=('next_residual', 'mean'))
two_regime['mean_next_ex_repricing'] = two_regime_ex['mean_next']; two_regime['continuation_ex_repricing'] = two_regime_ex['mean_next'] / two_regime_ex['mean_now']
display(two_regime)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(two_regime['mean_now'], two_regime['mean_next'], marker='o', label='all days'); ax.plot(two_regime_ex['mean_now'], two_regime_ex['mean_next'], marker='s', ls='--', label='excluding repricing days')
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6); ax.legend()
ax.set_xlabel('current residual, bin mean (bp)'); ax.set_ylabel('next residual, bin mean (bp)'); ax.set_title('Body persists, tails revert? conditional means by current-residual quantile bin')
savefig('07_two_regime')
record('residual_diagnostics', two_regime=two_regime.reset_index().astype({'res_bin': str}).round(4).to_dict(orient='records'))

# %% [markdown]
# ## 8. Paper portfolio: is the residual path harvestable?
#
# Every forecaster from Section 7 is run as a signal, plus the trailing mean faded in the active buckets. Weights
# are cross-sectionally demeaned, scaled to unit gross, and projected off the beta span with the day's $B_t$, so
# each portfolio has zero factor exposure. A walk-forward selector trades, each month, the signal with the best
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


# Candidate signals: every forecaster from Section 7, plus the trailing mean faded in the active buckets
# (the transaction tests show the accumulated residual reverts, and the reversal is concentrated in active names).
mean_name = next(k for k in fc_preds if k.startswith('Trailing'))
signals = {k: prep_portfolio_frame(v) for k, v in fc_preds.items() if not v.empty}
fade = signals[mean_name].copy(); fade['yhat'] = np.where(fade['activity_bucket'].isin(CFG.fade_buckets), -fade['yhat'], fade['yhat'])
signals[f'{mean_name}, faded in {"/".join(CFG.fade_buckets)}'] = fade

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
print('Note: the L4/L5 fade was fixed before this run (see PRE_REGISTERED); the walk-forward selector chooses among signals, not fade buckets.')

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
# `e_algo = 100 (y_trade − y_algo)`. Inference is Fama-MacBeth over trade dates; pooled OLS is descriptive. We
# also report the slope after neutralising date × side × size, by trade recency and by side, a joint regression
# on the one-day residual rank and the trailing-mean rank, the spread reading, and the evaluator-revision test.
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
    prints = t[['cusip', 'trade_ts', 'spread_to_mmd_bp', 'side']].rename(columns={'trade_ts': 'print_ts', 'spread_to_mmd_bp': 'last_print_spread_bp', 'side': 'last_print_side'}).sort_values('print_ts')
    prints['cum_prints'] = prints.groupby('cusip', observed=True).cumcount() + 1.0
    sig = t[['_row', 'cusip', 'signal_ts']].dropna(subset=['signal_ts']).sort_values('signal_ts')
    j1 = pd.merge_asof(sig, prints, left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    sig20 = sig.assign(signal_ts=sig['signal_ts'] - pd.Timedelta(days=20)).sort_values('signal_ts')
    j2 = pd.merge_asof(sig20, prints[['cusip', 'print_ts', 'cum_prints']], left_on='signal_ts', right_on='print_ts', by='cusip', direction='backward', allow_exact_matches=False)
    j1 = j1.set_index('_row'); j2 = j2.set_index('_row')
    t = t.set_index('_row')
    t['last_print_spread_bp'] = j1['last_print_spread_bp']; t['last_print_side'] = j1['last_print_side'].fillna('NONE')
    t['recency_days_sig'] = (t['signal_ts'] - j1['print_ts']).dt.total_seconds() / 86400.0
    t['prints_20d'] = (j1['cum_prints'].fillna(0) - j2['cum_prints'].fillna(0)).clip(lower=0)
    t = t.reset_index(drop=True)
    t['recency_bucket'] = pd.cut(t['recency_days'], bins=[-0.01, 1, 3, 7, 21, np.inf], labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'first_print')
    return t


def join_trades_to_residual(t: pd.DataFrame, res: pd.DataFrame, min_age: int) -> pd.DataFrame:
    extra = [c for c in ['peer_spread_bp', 'peer_rank', 'peer_dist', 'eta_abs', 'ssm_drift', 'fitted_bp'] if c in res.columns]
    score = res[['cusip', 'date', 'pit_residual', 'activity_bucket', 'resid_mean', 'resid_sd', 'target_abs_vol'] + extra + beta_cols].copy()
    score['cusip'] = score['cusip'].astype('string')
    score = score.merge(model[['cusip', 'date', 'modified_duration_lag1', 'mark_unchanged', 'days_since_mark_move']], on=['cusip', 'date'], how='left')
    score['pit_rank'] = score.groupby('date', observed=True)['pit_residual'].rank(pct=True)
    score['level_rank'] = score.groupby('date', observed=True)['resid_mean'].rank(pct=True)
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


CONTROLS = ['log_size', 'MinuteFromSignal', 'dMmdSprdSide']
if HAS_TRADES:
    trades = standardize_trades(trades_raw)
    trades = trades[(trades['trade_date'] >= pd.Timestamp(CFG.first_oos_date)) & (trades['trade_date'] <= pd.Timestamp(CFG.end_date))]
    tv = join_trades_to_residual(trades, resid, CFG.residual_min_age_days)
    print(f'matched trades in OOS window: {len(trades):,} | joined to a prior residual: {len(tv):,} | cusips {tv["cusip"].nunique():,} | trade dates {tv["trade_date"].nunique()} | residual age median {tv["residual_age_days"].median():.0f}d')
    rows = [{'target': 'e vs algo quote', 'bucket': 'ALL', **fm_slope(tv, 'e_algo_bp', 'pit_rank', CONTROLS)},
            {'target': 'e vs prior mark', 'bucket': 'ALL', **fm_slope(tv, 'e_mark_bp', 'pit_rank', CONTROLS)}]
    for b, g in tv.groupby('recency_bucket'):
        rows.append({'target': 'e vs prior mark by recency', 'bucket': b, **fm_slope(g, 'e_mark_bp', 'pit_rank', CONTROLS)})
    for s, g in tv.groupby('side'):
        rows.append({'target': 'e vs prior mark by side', 'bucket': f'side={s}', **fm_slope(g, 'e_mark_bp', 'pit_rank', CONTROLS)})
    tv['size_cell'] = pd.qcut(tv['size_bin'].rank(method='first'), min(5, tv['size_bin'].nunique()), labels=False, duplicates='drop') if tv['size_bin'].notna().any() else 0
    neut = neutralised_fm(tv, 'e_mark_bp', 'pit_rank', ['trade_date', 'side', 'size_cell'])
    rows.append({'target': 'e vs prior mark, neutralised date x side x size', 'bucket': 'ALL', **neut})
    # the level signal as the score
    if tv['level_rank'].notna().any():
        rows.append({'target': 'e vs prior mark, score = trailing-mean residual rank', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['level_rank']), 'e_mark_bp', 'level_rank', CONTROLS)})
        rows.append({'target': 'e vs algo quote, score = trailing-mean residual rank', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['level_rank']), 'e_algo_bp', 'level_rank', CONTROLS)})
    # ranks within duration quintile (stress-day robustness) and the peer RV score
    rows.append({'target': 'e vs prior mark, one-day rank within duration quintile', 'bucket': 'ALL', **fm_slope(tv, 'e_mark_bp', 'pit_rank_dur', CONTROLS)})
    rows.append({'target': 'e vs algo quote, one-day rank within duration quintile', 'bucket': 'ALL', **fm_slope(tv, 'e_algo_bp', 'pit_rank_dur', CONTROLS)})
    rows.append({'target': 'e vs prior mark, trailing-mean rank within duration quintile', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['level_rank_dur']), 'e_mark_bp', 'level_rank_dur', CONTROLS)})
    rows.append({'target': 'e vs algo quote, trailing-mean rank within duration quintile', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['level_rank_dur']), 'e_algo_bp', 'level_rank_dur', CONTROLS)})
    if 'peer_rank' in tv.columns and tv['peer_rank'].notna().any():
        rows.append({'target': 'e vs prior mark, score = peer RV rank (cheap to peers = high)', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['peer_rank']), 'e_mark_bp', 'peer_rank', CONTROLS)})
        rows.append({'target': 'e vs algo quote, score = peer RV rank', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['peer_rank']), 'e_algo_bp', 'peer_rank', CONTROLS)})
    # excluding repricing days (the residual on the day before the trade, or the trade date, is a repricing day)
    _ex = tv[~tv['residual_date'].isin(REPRICING_DATES) & ~tv['trade_date'].isin(REPRICING_DATES)]
    rows.append({'target': 'e vs prior mark, excluding repricing days', 'bucket': 'ALL', **fm_slope(_ex, 'e_mark_bp', 'pit_rank', CONTROLS)})
    rows.append({'target': 'e vs prior mark, trailing-mean rank, excluding repricing days', 'bucket': 'ALL', **fm_slope(_ex.dropna(subset=['level_rank']), 'e_mark_bp', 'level_rank', CONTROLS)})
    tx = pd.DataFrame(rows)
    display(tx)
    # Joint regression: are the one-day continuation, the trailing-mean reversal and the peer RV score independent?
    joint_scores = ['pit_rank', 'level_rank'] + (['peer_rank'] if 'peer_rank' in tv.columns and tv['peer_rank'].notna().mean() > 0.5 else [])
    jt = pd.concat([fm_multi(tv, 'e_mark_bp', joint_scores, CONTROLS).assign(target='e vs prior mark'),
                    fm_multi(tv, 'e_algo_bp', joint_scores, CONTROLS).assign(target='e vs algo quote')], ignore_index=True)
    print('Joint Fama-MacBeth: one-day residual rank, trailing-mean rank' + (' and peer RV rank' if len(joint_scores) == 3 else '') + ' entered together (boot_t = moving-block bootstrap over weeks)'); display(jt.round(4))
    record('transaction_validation', joint=jt.round(4).to_dict(orient='records'))

    # Decile charts: mean trade-vs-mark by score decile, by MSRB side (S dealer sale / P dealer purchase / D inter-dealer)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, (col, title) in zip(ax, [('pit_rank', 'one-day residual rank'), ('level_rank', f'trailing-{CFG.level_signal_window} mean residual rank')]):
        d = tv.dropna(subset=[col]).copy(); d['decile'] = pd.qcut(d[col].rank(method='first'), 10, labels=False) + 1
        for sd, g in d.groupby('side'):
            if len(g) > 2000:
                a.plot(g.groupby('decile')['e_mark_bp'].mean(), marker='o', label={'S': 'S: dealer sale (customer buys)', 'P': 'P: dealer purchase (customer sells)', 'D': 'D: inter-dealer'}.get(sd, sd))
        a.plot(d.groupby('decile')['e_mark_bp'].mean(), color='k', lw=2, label='all'); a.axhline(0, color='k', lw=0.6); a.set_xlabel('score decile (1 rich -> 10 cheap)'); a.set_ylabel('mean trade - prior mark (bp)'); a.set_title(f'Trade vs prior mark by {title}'); a.legend(fontsize=8)
    savefig('09_decile_charts')

    rec_order = ['<=1d', '1-3d', '3-7d', '7-21d', '>21d']
    # ---- the spread reading: does the residual predict how far from the mark trades print, regardless of direction?
    sp_rows = [{'test': '|trade - prior mark| on residual rank', 'bucket': 'ALL', **fm_slope(tv, 'abs_e_mark_bp', 'pit_rank', CONTROLS)},
               {'test': '|trade - prior mark| on |residual| / vol', 'bucket': 'ALL', **fm_slope(tv, 'abs_e_mark_bp', 'abs_resid_z', CONTROLS)},
               {'test': '|trade - prior mark| on days since the mark last moved', 'bucket': 'ALL', **fm_slope(tv.assign(dsm=tv['days_since_mark_move'].clip(upper=20)), 'abs_e_mark_bp', 'dsm', CONTROLS)}]
    if 'eta_abs' in tv.columns and tv['eta_abs'].notna().mean() > 0.3:
        sp_rows.append({'test': '|trade - prior mark| on state-space mark-noise estimate |eta|', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['eta_abs']), 'abs_e_mark_bp', 'eta_abs', CONTROLS)})
        sp_rows.append({'test': 'trade - prior mark on filtered drift (state-space)', 'bucket': 'ALL', **fm_slope(tv.dropna(subset=['ssm_drift']), 'e_mark_bp', 'ssm_drift', CONTROLS)})
    for b, g in tv.groupby('recency_bucket'):
        sp_rows.append({'test': '|trade - prior mark| on residual rank, by recency', 'bucket': b, **fm_slope(g, 'abs_e_mark_bp', 'pit_rank', CONTROLS)})
    # dealer round trip: same bond, same day, both a dealer purchase (P, customer sells, higher yield) and a dealer sale (S, customer buys, lower yield)
    day = tv[tv['side'].isin(['P', 'S'])].groupby(['cusip', 'trade_date', 'side'], observed=True).agg(y=('msrb_yield', 'mean'), rank=('pit_rank', 'mean'), z=('abs_resid_z', 'mean'), log_size=('log_size', 'mean')).unstack('side')
    if ('y', 'P') in day.columns and ('y', 'S') in day.columns:
        rt = pd.DataFrame({'round_trip_bp': 100.0 * (day[('y', 'P')] - day[('y', 'S')]), 'pit_rank': day[('rank', 'P')], 'abs_resid_z': day[('z', 'P')], 'log_size': day[('log_size', 'P')]}).dropna().reset_index()
        rt['side'] = 'RT'
        if len(rt) > 200:
            sp_rows.append({'test': 'dealer round trip (P yield - S yield, same bond-day) on residual rank', 'bucket': f'{len(rt):,} pairs', **fm_slope(rt, 'round_trip_bp', 'pit_rank', ['log_size'])})
            sp_rows.append({'test': 'dealer round trip on |residual| / vol', 'bucket': f'{len(rt):,} pairs', **fm_slope(rt, 'round_trip_bp', 'abs_resid_z', ['log_size'])})
            print(f'round-trip pairs: {len(rt):,} | median dealer round trip {rt["round_trip_bp"].median():.1f} bp (positive expected under the MSRB convention)')
    spread_tx = pd.DataFrame(sp_rows)
    print('Spread reading (positive = wider around the mark when the residual is large):'); display(spread_tx)

    # ---- evaluator-revision test: after a trade, does the same-day close move toward the print, and more so when the residual is large?
    ev = tv.dropna(subset=['mark_revision_bp', 'e_mark_bp']).copy()
    ev = ev[(ev['e_mark_bp'].abs() > 0.5)]
    ev['toward_trade'] = (np.sign(ev['mark_revision_bp']) == np.sign(ev['e_mark_bp'])).astype(float)
    ev['pass_through'] = (ev['mark_revision_bp'] / ev['e_mark_bp']).clip(-2, 2)
    ev['abs_resid_q'] = pd.qcut(ev['abs_resid_z'].rank(method='first'), 5, labels=['Q1 small', 'Q2', 'Q3', 'Q4', 'Q5 large'])
    evt = ev.groupby('abs_resid_q', observed=True).agg(trades=('e_mark_bp', 'size'), abs_gap_bp=('e_mark_bp', lambda s: s.abs().median()), share_revised_toward_trade=('toward_trade', 'mean'), median_pass_through=('pass_through', 'median'), abs_revision_bp=('mark_revision_bp', lambda s: s.abs().median()))
    print('Evaluator-revision test (same-day close vs prior mark, by |residual| quintile):'); display(evt.round(3))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    sr = spread_tx[spread_tx['test'].str.contains('by recency')].set_index('bucket').reindex(rec_order)
    ax[0].bar(sr.index, sr['fm_beta'], yerr=1.96 * (sr['fm_beta'] / sr['fm_t']).abs(), capsize=4, color='#8172B2'); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('Spread reading: |trade - mark| on residual rank, by recency (bp/rank)', fontsize=10)
    ax[1].bar(evt.index.astype(str), evt['share_revised_toward_trade'], color='#64B5CD'); ax[1].axhline(0.5, color='k', lw=0.6, ls='--'); ax[1].set_ylim(0.3, 1.0); ax[1].set_title('Share of same-day mark revisions toward the trade, by |residual| quintile', fontsize=10)
    ax[2].bar(evt.index.astype(str), evt['median_pass_through'], color='#CCB974'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Median pass-through of trade - mark into the same-day close', fontsize=10)
    savefig('09_spread_and_revision')
    record('transaction_validation', spread_table=spread_tx.round(4).to_dict(orient='records'), evaluator_revision=evt.round(4).reset_index().astype({'abs_resid_q': str}).to_dict(orient='records'))
    rec_order = ['<=1d', '1-3d', '3-7d', '7-21d', '>21d']
    rec = tx[tx['target'] == 'e vs prior mark by recency'].set_index('bucket').reindex(rec_order)
    side = tx[tx['target'] == 'e vs prior mark by side'].set_index('bucket')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(rec.index, rec['fm_beta'], yerr=1.96 * (rec['fm_beta'] / rec['fm_t']).abs(), capsize=4, color='#4C72B0'); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('FM beta of trade-vs-prior-mark on residual rank, by trade recency (bp per rank)')
    ax[1].bar(side.index, side['fm_beta'], yerr=1.96 * (side['fm_beta'] / side['fm_t']).abs(), capsize=4, color='#55A868'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('by MSRB side')
    savefig('09_transaction_validation')
    record('transaction_validation', table=tx.round(4).to_dict(orient='records'), trades=len(tv), trade_dates=int(tv['trade_date'].nunique()))
else:
    tv = pd.DataFrame()
    print('No matched trades in the pipeline store; Sections 9 and 10 are skipped.')

# %% [markdown]
# ## 9b. Print-to-print error correction
#
# The direct test of the pricing-error story. Take consecutive prints of the same bond at $t_0$ and $t_1$ (gap up
# to 30 days). Between them the factor model produces a cumulative factor-implied move $\Delta F$ and a cumulative
# residual move $\Delta\varepsilon$ in the evaluated marks (both known before $t_1$). The print-to-print yield change
# is regressed on both, plus the pricing error of the mark at $t_0$:
#
# $\Delta y^{print} = a + b_F\,\Delta F + b_\varepsilon\,\Delta\varepsilon + c\,(y^{print}_0 - mark_0) + \ldots$
#
# $b_F \approx 1$ says prints follow the factor move in full (the roll-forward is real, not just in-panel);
# $b_\varepsilon$ says how much of the residual move in the marks the next print confirms; $c < 0$ says the gap
# between the print and the mark closes from the print side (the mark was right), $c \approx 0$ that the mark
# catches up. The same panel scores five predictors of the next print by gap length, which is the roll-forward
# scored out of panel, and asks whether the algo's own pricing error persists from one print to the next.

# %%
if HAS_TRADES and not tv.empty and 'cum_fitted_bp' in tv.columns:
    _t = tv.sort_values(['cusip', 'trade_ts'], kind='stable').copy()
    _g = _t.groupby('cusip', observed=True)
    for c in ['trade_ts', 'msrb_yield', 'e_algo_bp', 'e_mark_bp', 'residual_date', 'cum_fitted_bp', 'cum_resid_bp', 'y_prior_mark', 'algo_signal_yield', 'side']:
        _t[f'{c}_0'] = _g[c].shift(1)
    p2p = _t.dropna(subset=['trade_ts_0', 'cum_fitted_bp_0', 'cum_fitted_bp']).copy()
    p2p['gap_days'] = (p2p['trade_ts'] - p2p['trade_ts_0']).dt.total_seconds() / 86400.0
    p2p = p2p[(p2p['gap_days'] > 0) & (p2p['gap_days'] <= 30) & (p2p['residual_date'] > p2p['residual_date_0'])].copy()
    p2p['dy_print_bp'] = 100.0 * (p2p['msrb_yield'] - p2p['msrb_yield_0'])
    p2p['d_fit_bp'] = p2p['cum_fitted_bp'] - p2p['cum_fitted_bp_0']; p2p['d_res_bp'] = p2p['cum_resid_bp'] - p2p['cum_resid_bp_0']
    p2p['d_mark_bp'] = 100.0 * (p2p['y_prior_mark'] - p2p['y_prior_mark_0'])
    p2p['gap_bucket'] = pd.cut(p2p['gap_days'], bins=[0, 1, 3, 7, 14, 30], labels=['<=1d', '1-3d', '3-7d', '7-14d', '14-30d']).astype(str)
    for s_ in ['P', 'S']:
        p2p[f'side1_{s_}'] = (p2p['side'] == s_).astype(float); p2p[f'side0_{s_}'] = (p2p['side_0'] == s_).astype(float)
    P2P_CTRL = ['log_size', 'side1_P', 'side1_S', 'side0_P', 'side0_S']
    print(f'print-to-print pairs: {len(p2p):,} | median gap {p2p["gap_days"].median():.1f} days | corr(mark move, factor + residual move) {p2p["d_mark_bp"].corr(p2p["d_fit_bp"] + p2p["d_res_bp"]):.3f}')
    r1 = fm_multi(p2p, 'dy_print_bp', ['d_fit_bp', 'd_res_bp', 'e_mark_bp_0'], P2P_CTRL).assign(regression='print change on factor move, residual move, prior print-mark gap')
    r2 = fm_multi(p2p, 'e_algo_bp', ['e_algo_bp_0', 'd_fit_bp', 'd_res_bp'], P2P_CTRL).assign(regression='algo pricing error at t1 on its own lag, factor move, residual move')
    r3 = fm_multi(p2p, 'dy_print_bp', ['d_mark_bp', 'e_mark_bp_0'], P2P_CTRL).assign(regression='print change on total mark move and prior gap')
    p2p_reg = pd.concat([r1, r2, r3], ignore_index=True)
    print('Print-to-print regressions (Fama-MacBeth over t1 dates; boot_t = moving-block bootstrap):'); display(p2p_reg.round(4))
    by_gap = []
    for b, g in p2p.groupby('gap_bucket'):
        if len(g) >= 500:
            rr = fm_multi(g, 'dy_print_bp', ['d_fit_bp', 'd_res_bp', 'e_mark_bp_0'], P2P_CTRL).set_index('score')
            by_gap.append({'gap_bucket': b, 'pairs': len(g), 'b_factor': rr.loc['d_fit_bp', 'fm_beta'], 't_factor': rr.loc['d_fit_bp', 'fm_t'], 'b_residual': rr.loc['d_res_bp', 'fm_beta'], 't_residual': rr.loc['d_res_bp', 'fm_t'], 'c_gap': rr.loc['e_mark_bp_0', 'fm_beta'], 't_gap': rr.loc['e_mark_bp_0', 'fm_t']})
    p2p_gap = pd.DataFrame(by_gap).set_index('gap_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-14d', '14-30d']).dropna(how='all')
    display(p2p_gap.round(3))
    # Predictors of the next print, scored against the print: the roll-forward out of panel
    cand = {'stale print': p2p['msrb_yield_0'], 'print + factor move': p2p['msrb_yield_0'] + p2p['d_fit_bp'] / 100.0, 'print + factor + residual move': p2p['msrb_yield_0'] + (p2p['d_fit_bp'] + p2p['d_res_bp']) / 100.0,
            'prior mark': p2p['y_prior_mark'], 'algo quote': p2p['algo_signal_yield']}
    mae_rows = []
    for b, g in p2p.groupby('gap_bucket'):
        row = {'gap_bucket': b, 'pairs': len(g)}
        for k, v in cand.items():
            row[k] = (100.0 * (g['msrb_yield'] - v.loc[g.index])).abs().mean()
        mae_rows.append(row)
    rowa = {'gap_bucket': 'ALL', 'pairs': len(p2p), **{k: (100.0 * (p2p['msrb_yield'] - v)).abs().mean() for k, v in cand.items()}}
    p2p_mae = pd.DataFrame(mae_rows + [rowa]).set_index('gap_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-14d', '14-30d', 'ALL']).dropna(how='all')
    print('MAE of next-print predictors by gap (bp): the roll-forward scored against prints'); display(p2p_mae.round(2))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
    p2p_mae.drop(index='ALL', errors='ignore')[list(cand)].plot.bar(ax=ax[0]); ax[0].set_title('MAE vs next print by gap between prints (bp)', fontsize=10); ax[0].set_xlabel(''); ax[0].legend(fontsize=7)
    xx = np.arange(len(p2p_gap)); ax[1].bar(xx - 0.2, p2p_gap['b_factor'], width=0.4, yerr=1.96 * (p2p_gap['b_factor'] / p2p_gap['t_factor']).abs(), capsize=3, label='b_factor (1 = prints follow the factor move)'); ax[1].bar(xx + 0.2, p2p_gap['b_residual'], width=0.4, yerr=1.96 * (p2p_gap['b_residual'] / p2p_gap['t_residual']).abs(), capsize=3, label='b_residual (share of the mark residual move confirmed)')
    ax[1].set_xticks(xx); ax[1].set_xticklabels(p2p_gap.index); ax[1].axhline(1, color='k', ls='--', lw=0.6); ax[1].axhline(0, color='k', lw=0.6); ax[1].legend(fontsize=7); ax[1].set_title('Print-to-print pass-through of the two mark components', fontsize=10)
    ax[2].bar(p2p_gap.index, p2p_gap['c_gap'], yerr=1.96 * (p2p_gap['c_gap'] / p2p_gap['t_gap']).abs(), capsize=3, color='#8172B2'); ax[2].axhline(0, color='k', lw=0.6); ax[2].axhline(-1, color='k', ls='--', lw=0.6); ax[2].set_title('c: prior print-mark gap into the next print change (-1 = print reverts to mark, 0 = mark was wrong)', fontsize=8)
    savefig('09b_print_to_print')
    record('print_to_print', pairs=len(p2p), regressions=p2p_reg.round(4).to_dict(orient='records'), by_gap=p2p_gap.round(4).reset_index().to_dict(orient='records'), mae=p2p_mae.round(3).reset_index().to_dict(orient='records'))
else:
    print('Section 9b skipped: needs matched trades joined to residuals.')

# %% [markdown]
# ## 9c. Economic sizing: the slow-horizon skew on the algo quote, walk-forward
#
# The one row that survived the algo quote is the trailing-mean rank. This cell turns it into a quote adjustment
# the way it would be deployed: each month, the skew coefficient is the Fama-MacBeth slope of (print − algo) on the
# score over the *prior* months, the adjusted quote is algo + skew, and the adjusted quote is scored against the
# print it was matched to. Three versions: trailing-mean rank alone, jointly with the one-day rank, and jointly
# with the peer RV rank. Reported by side, because a market maker is paid on the side that is hit.

# %%
if HAS_TRADES and not tv.empty:
    es = tv.dropna(subset=['level_rank', 'e_algo_bp', 'pit_rank']).copy()
    es['month'] = es['trade_date'].dt.to_period('M')
    has_peer = 'peer_rank' in es.columns and es['peer_rank'].notna().mean() > 0.5
    versions = {'trailing-mean rank': ['level_rank'], 'one-day + trailing-mean ranks': ['pit_rank', 'level_rank']}
    if has_peer:
        versions['one-day + trailing-mean + peer RV ranks'] = ['pit_rank', 'level_rank', 'peer_rank']
    es_parts, coef_rows = [], []
    for m in sorted(es['month'].unique())[1:]:
        tr, te = es[es['month'] < m], es[es['month'] == m].copy()
        if len(tr) < 5_000 or te.empty:
            continue
        for vname, xs in versions.items():
            sub = tr.dropna(subset=xs)
            co = fm_multi(sub, 'e_algo_bp', xs, []).set_index('score')['fm_beta']
            skew = sum(co[x] * (te[x] - 0.5) for x in xs)
            te[f'skew_bp[{vname}]'] = skew; te[f'e_adj_bp[{vname}]'] = te['e_algo_bp'] - skew
            coef_rows.append({'month': str(m), 'version': vname, **{f'k_{x}': float(co[x]) for x in xs}})
        es_parts.append(te)
    if es_parts:
        esp = pd.concat(es_parts, ignore_index=True); kpath = pd.DataFrame(coef_rows)
        print('Skew coefficients by month (bp per unit rank, estimated on prior months):'); display(kpath.round(3))

        def sizing_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
            g = frame.groupby(by, observed=True); out = pd.DataFrame({'n': g.size(), 'MAE algo (bp)': g['e_algo_bp'].apply(lambda s: s.abs().mean())})
            for vname in versions:
                e = f'e_adj_bp[{vname}]'; sk = f'skew_bp[{vname}]'
                out[f'MAE adj [{vname}]'] = g[e].apply(lambda s: s.abs().mean())
                d = frame.assign(gain=frame['e_algo_bp'].abs() - frame[e].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
                out[f'gain [{vname}] bp'] = d.mean(); out[f'gain t [{vname}]'] = np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)
                act = frame[frame[sk].abs() > 1.0]
                out[f'hit rate [{vname}]'] = act.assign(hit=(np.sign(act[sk]) == np.sign(act['e_algo_bp'])).astype(float)).groupby(by, observed=True)['hit'].mean()
                out[f'mean |skew| [{vname}] bp'] = g[sk].apply(lambda s: s.abs().mean())
            return out

        es_all = sizing_table(esp.assign(all='ALL'), 'all'); es_side = sizing_table(esp, 'side'); es_rec = sizing_table(esp, 'recency_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).dropna(how='all')
        print('Economic sizing of the skew on the algo quote (held-out months; hit rate among |skew| > 1 bp):'); display(es_all.T.round(3)); display(es_side.round(3)); display(es_rec.round(3))
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
        v0 = list(versions)[-1]
        gm = esp.assign(gain=esp['e_algo_bp'].abs() - esp[f'e_adj_bp[{v0}]'].abs()).groupby('month', observed=True)['gain'].mean()
        gm.index = gm.index.astype(str); gm.plot.bar(ax=ax[0], color='#4C72B0'); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title(f'Monthly gain vs algo quote, bp per trade [{v0}]', fontsize=10); ax[0].set_xlabel('')
        es_side[[f'gain [{v}] bp' for v in versions]].plot.bar(ax=ax[1]); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('Gain by MSRB side (bp per trade)', fontsize=10); ax[1].legend([v for v in versions], fontsize=7); ax[1].set_xlabel('')
        _k = kpath[kpath['version'] == 'trailing-mean rank'].set_index('month')['k_level_rank']; ax[2].plot(_k.index, _k.values, 'o-'); ax[2].axhline(0, color='k', lw=0.6); ax[2].set_title('Skew coefficient on the trailing-mean rank by month (bp per unit rank)', fontsize=10)
        savefig('09c_economic_sizing')
        esp.to_parquet(ARTIFACTS / 'quote_skew_v2.parquet', index=False)
        record('economic_sizing', versions={k: v for k, v in versions.items()}, overall=es_all.round(4).to_dict(orient='records'), by_side=es_side.round(4).reset_index().to_dict(orient='records'), by_recency=es_rec.round(4).reset_index().to_dict(orient='records'), coefficients=kpath.round(4).to_dict(orient='records'))
    else:
        print('Section 9c: not enough prior-month trades to estimate the skew.')
else:
    print('Section 9c skipped: needs matched trades.')

# %% [markdown]
# ## 10. Level relative value: walk-forward GBM on trade spreads
#
# The pivot experiment. Target: the trade's spread to the MMD curve in bp. Features: the bond's instruments and
# betas on the residual date, plus side, size, recency and timing. Walk-forward by month (train on prior months).
# For each test trade we compare the absolute error of three yield predictions: the algo quote, the prior
# evaluated mark, and the model-implied yield (MMD + predicted spread). The question is whether the level model
# beats the algo where the algo is weakest: names with no recent print. v2.1 adds the print features the algo
# has and the first run lacked: the bond's last print spread to MMD, that print's side, and prints in 20 days.

# %%
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

if HAS_TRADES and not tv.empty and tv['MmdYld'].notna().mean() > 0.5 and len(tv) >= 4_000:
    lv = tv.copy()
    lv['spread_bp'] = 100.0 * (lv['msrb_yield'] - lv['MmdYld'])
    feat_panel = model[['cusip', 'date', 'rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1', 'state_others'] + [c for c in CHARS if c != 'market_fv']].rename(columns={'date': 'residual_date'})
    feat_panel['residual_date'] = to_ns(feat_panel['residual_date'])
    lv['residual_date'] = to_ns(lv['residual_date'])
    feat_panel = feat_panel[[c for c in feat_panel.columns if c in ('cusip', 'residual_date') or c not in lv.columns]]   # the residual join already carries some of these
    lv = lv.merge(feat_panel, on=['cusip', 'residual_date'], how='left')
    lv['recency_days_c'] = lv['recency_days_sig'].clip(upper=60).fillna(60)   # days from the last print to the SIGNAL, not to the trade
    for s in ['D', 'P', 'S']:
        lv[f'side_{s}'] = (lv['side'] == s).astype(float)
        lv[f'last_side_{s}'] = (lv['last_print_side'] == s).astype(float)
    PRINT_FEATURES = ['last_print_spread_bp', 'prints_20d', 'last_side_D', 'last_side_P', 'last_side_S']
    ALGO_DERIVED = ['dMmdSprdSide', 'MinuteFromSignal']            # fields that come with the algo signal itself
    STALENESS_FEATURES = [c for c in ['mark_unchanged', 'days_since_mark_move', 'eta_abs', 'ssm_drift'] if c in lv.columns]
    PEER_FEATURES = [c for c in ['peer_spread_bp', 'peer_dist'] if c in lv.columns]
    RESIDUAL_FEATURES = ['pit_residual', 'pit_rank']
    FEATURES = PRINT_FEATURES + ['rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1', 'recency_days_c', 'log_size', 'side_D', 'side_P', 'side_S', 'state_others'] + RESIDUAL_FEATURES + ALGO_DERIVED + STALENESS_FEATURES + PEER_FEATURES + beta_cols + [c for c in CHARS if c != 'market_fv']
    FEATURES = [c for c in dict.fromkeys(FEATURES) if c in lv.columns]
    lv = lv.dropna(subset=['spread_bp', 'e_mark_bp']).copy()
    lv['month'] = lv['trade_date'].dt.to_period('M')

    def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
        # drop columns that are constant or all-missing in the training window (a one-value column breaks the histogram binner)
        return [c for c in features if frame[c].astype(float).nunique(dropna=True) >= 2]

    def fit_level_model(frame: pd.DataFrame, features: list[str], n_trees: int) -> tuple[pd.DataFrame, list[pd.Series]]:
        preds, importances = [], []
        for m in sorted(frame['month'].unique())[1:]:
            tr, te = frame[frame['month'] < m], frame[frame['month'] == m]
            if len(tr) < 2_000 or te.empty:
                continue
            feats = usable_features(tr, features)
            med = tr[feats].median()
            Xtr, Xte = tr[feats].astype(float).fillna(med), te[feats].astype(float).fillna(med)
            ytr = tr['spread_bp'].clip(*tr['spread_bp'].quantile([0.005, 0.995]))
            if HAS_LGB:
                mdl = lgb.LGBMRegressor(objective='l1', n_estimators=n_trees, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=CFG.seed, verbose=-1).fit(Xtr, ytr)
                importances.append(pd.Series(mdl.feature_importances_, index=feats))
            else:
                mdl = HistGradientBoostingRegressor(loss='absolute_error', max_iter=max(100, n_trees * 2 // 3), learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, random_state=CFG.seed).fit(Xtr, ytr)
            p = te[['cusip', 'trade_date', 'trade_ts', 'recency_bucket', 'side', 'size_cell', 'msrb_yield', 'MmdYld', 'e_algo_bp', 'e_mark_bp', 'y_prior_mark', 'pit_residual', 'abs_resid_z', 'residual_date']].copy()
            p['spread_hat'] = mdl.predict(Xte); p['e_model_bp'] = 100.0 * (p['msrb_yield'] - (p['MmdYld'] + p['spread_hat'] / 100.0)); p['month'] = str(m)
            preds.append(p)
        if not preds:
            raise RuntimeError('level model: no held-out month had at least 2,000 training trades; widen the window or lower the threshold')
        return pd.concat(preds, ignore_index=True), importances

    t0 = time.perf_counter()
    lvp, importances = fit_level_model(lv, FEATURES, 600)
    print(f'level model ({"LightGBM" if HAS_LGB else "sklearn HistGBM"}): {len(lvp):,} held-out trades over {lvp["month"].nunique()} months, {len(FEATURES)} features (incl. {len(PRINT_FEATURES)} print, {len(ALGO_DERIVED)} algo-derived, {len(STALENESS_FEATURES)} staleness, {len(PEER_FEATURES)} peer) | {time.perf_counter()-t0:.0f}s')

    def mae_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby(by)
        out = pd.DataFrame({'n': g.size(), 'MAE algo quote': g['e_algo_bp'].apply(lambda s: s.abs().mean()), 'MAE prior mark': g['e_mark_bp'].apply(lambda s: s.abs().mean()), 'MAE level model': g['e_model_bp'].apply(lambda s: s.abs().mean())})
        # daily FM t of (|e_algo| - |e_model|)
        d = frame.assign(gain=frame['e_algo_bp'].abs() - frame['e_model_bp'].abs()).groupby([by, 'trade_date'])['gain'].mean().groupby(level=0)
        out['gain vs algo (bp)'] = d.mean(); out['gain t (FM)'] = np.sqrt(d.count()) * d.mean() / d.std()
        return out

    overall = mae_table(lvp.assign(all='ALL'), 'all')
    by_rec = mae_table(lvp, 'recency_bucket').reindex(['<=1d', '1-3d', '3-7d', '7-21d', '>21d', 'first_print'])
    by_side = mae_table(lvp, 'side')
    display(overall); display(by_rec); display(by_side)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    by_rec[['MAE algo quote', 'MAE prior mark', 'MAE level model']].plot.bar(ax=ax[0]); ax[0].set_title('Absolute yield error of three predictions by trade recency (bp)'); ax[0].set_xlabel('')
    if importances:
        imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=True).tail(15)
        imp.plot.barh(ax=ax[1], color='#4C72B0'); ax[1].set_title('Level model: mean feature importance')
    else:
        ax[1].text(0.1, 0.5, 'feature importance needs LightGBM', fontsize=12); ax[1].axis('off')
    savefig('10_level_model')
    # Calibration and stability: predicted vs realised spread, and monthly MAE of algo vs model
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    _sp = 100.0 * (lvp['msrb_yield'] - lvp['MmdYld'])
    lim = [np.nanpercentile(_sp, 1), np.nanpercentile(_sp, 99)]
    hb = ax[0].hexbin(lvp['spread_hat'].clip(*lim), _sp.clip(*lim), gridsize=50, cmap='viridis', mincnt=1, bins='log'); ax[0].plot(lim, lim, 'k--', lw=1); ax[0].set_xlabel('predicted spread to MMD (bp)'); ax[0].set_ylabel('realised trade spread to MMD (bp)'); ax[0].set_title('Level model calibration on held-out trades'); plt.colorbar(hb, ax=ax[0], fraction=0.046)
    mm_ = lvp.groupby('month').agg(algo=('e_algo_bp', lambda s: s.abs().mean()), mark=('e_mark_bp', lambda s: s.abs().mean()), model=('e_model_bp', lambda s: s.abs().mean()))
    mm_.plot(ax=ax[1], marker='o'); ax[1].set_title('Held-out MAE by month (bp): algo quote, prior mark, level model'); ax[1].set_xlabel('')
    savefig('10_level_model_calibration')
    lvp.to_parquet(ARTIFACTS / 'level_model_predictions_v2.parquet', index=False)
    record('level_model', backend='lightgbm' if HAS_LGB else 'sklearn_histgbm', features=FEATURES, overall=overall.round(4).to_dict(orient='records'), by_recency=by_rec.round(4).reset_index().to_dict(orient='records'), by_side=by_side.round(4).reset_index().to_dict(orient='records'))
    # Ablations: what the gain over the algo depends on. Dropping the algo-derived fields answers whether the model is an
    # independent pricer or a correction on the algo quote; dropping print features isolates the mark/characteristic view.
    if CFG.level_model_ablation:
        abl_sets = {'full': FEATURES, 'without algo-derived fields': [c for c in FEATURES if c not in ALGO_DERIVED], 'without print features': [c for c in FEATURES if c not in PRINT_FEATURES],
                    'without algo-derived and print features': [c for c in FEATURES if c not in ALGO_DERIVED + PRINT_FEATURES], 'without staleness / peer / residual features': [c for c in FEATURES if c not in STALENESS_FEATURES + PEER_FEATURES + RESIDUAL_FEATURES]}
        abl_rows = []
        for aname, feats in abl_sets.items():
            t0 = time.perf_counter()
            ap, _ = fit_level_model(lv, feats, CFG.ablation_trees)
            at = mae_table(ap.assign(all='ALL'), 'all').iloc[0]; ar = mae_table(ap, 'recency_bucket')
            abl_rows.append({'variant': aname, 'features': len(feats), 'MAE algo': at['MAE algo quote'], 'MAE model': at['MAE level model'], 'gain vs algo (bp)': at['gain vs algo (bp)'], 'gain t (FM)': at['gain t (FM)'],
                             'gain <=1d': ar['gain vs algo (bp)'].get('<=1d', np.nan), 'gain >21d': ar['gain vs algo (bp)'].get('>21d', np.nan), 'seconds': time.perf_counter() - t0})
        ablation = pd.DataFrame(abl_rows).set_index('variant')
        print(f'Level-model ablations ({CFG.ablation_trees} trees each; the full model above uses 600):'); display(ablation.round(3))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ablation['gain vs algo (bp)'].plot.barh(ax=ax, color='#4C72B0'); ax.axvline(0, color='k', lw=0.6); ax.set_title('Level model: gain over the algo quote by feature set (bp of MAE)'); ax.set_ylabel('')
        savefig('10_level_model_ablation')
        record('level_model', ablation=ablation.round(4).reset_index().to_dict(orient='records'))
else:
    print('Level model skipped: needs at least 4,000 matched trades with MmdYld.')

# %% [markdown]
# ## 10b. Application: shrinkage mid, residual-driven half-width, concession
#
# The level model is the **mid**. The evaluated mark enters only through an **inverse-variance weight** estimated
# on the training months within bins of residual size (|residual| / bond volatility): in each bin the weight on
# the mark is its error precision against the next print relative to the model's, so the data decides how much
# mark to keep and the one-parameter curve $1/(1+\lambda z)$ is kept only as a check with a grid up to 64. The
# **half-width** is fitted around the model mid as a base term plus a residual-size term, scaled on the training
# months to cover 80% of prints, and compared with a fixed band at the same target. The **concession** is that
# half-width on the dealer's side: bid (P, we buy from the customer) = mid + h, offer (S, we sell) = mid − h.
# Everything is walk-forward by month.

# %%
if HAS_TRADES and not tv.empty:
    app = tv.dropna(subset=['y_prior_mark', 'msrb_yield']).copy()
    # rule-based cohort value on the residual date: median closing yield of bonds in the same duration x rating x call cell
    coh_src = model[['cusip', 'date', 'closing_yield', 'modified_duration_lag1', 'rating_score', 'call_structure']].copy()
    coh_src['dur_cell'] = pd.cut(coh_src['modified_duration_lag1'], bins=[-1, 1, 2.5, 4, 6, 8, 11, 100], labels=False)
    coh_src['rat_cell'] = pd.cut(coh_src['rating_score'].fillna(-1), bins=[-2, 11.5, 14.5, 17.5, 20.5, 21.5], labels=False)
    cells = ['date', 'dur_cell', 'rat_cell', 'call_structure']
    coh_src['n_cell'] = coh_src.groupby(cells, observed=True)['closing_yield'].transform('size')
    coh_src['y_cohort'] = coh_src.groupby(cells, observed=True)['closing_yield'].transform('median')
    coh_src['y_cohort_loo'] = np.where(coh_src['n_cell'] > 1, coh_src['y_cohort'], np.nan)
    coh = coh_src[['cusip', 'date', 'y_cohort_loo']].rename(columns={'date': 'residual_date', 'y_cohort_loo': 'y_cohort'})
    coh['residual_date'] = to_ns(coh['residual_date']); app['residual_date'] = to_ns(app['residual_date'])
    app = app.merge(coh, on=['cusip', 'residual_date'], how='left')
    if 'lvp' in globals() and not lvp.empty:
        lm = lvp[['cusip', 'trade_ts', 'spread_hat', 'MmdYld']].copy(); lm['y_model_lm'] = lm['MmdYld'] + lm['spread_hat'] / 100.0
        app = app.merge(lm[['cusip', 'trade_ts', 'y_model_lm']], on=['cusip', 'trade_ts'], how='left')
        app['y_model'] = app['y_model_lm'].fillna(app['y_cohort']); app['model_src'] = np.where(app['y_model_lm'].notna(), 'level_model', 'cohort'); model_source = 'level model where available, cohort median otherwise'
    else:
        app['y_model'] = app['y_cohort']; app['model_src'] = 'cohort'; model_source = 'cohort median (level model not run)'
    app = app.dropna(subset=['y_model']).copy()
    app['z'] = app['abs_resid_z'].clip(0, 10).fillna(1.0)
    app['month'] = app['trade_date'].dt.to_period('M')
    app['err_mark_bp'] = 100.0 * (app['msrb_yield'] - app['y_prior_mark']); app['err_model_bp'] = 100.0 * (app['msrb_yield'] - app['y_model'])
    print(f'application sample: {len(app):,} trades | model value: {model_source}')

    def blend_lambda(frame: pd.DataFrame, lam: float) -> pd.Series:
        w = 1.0 / (1.0 + lam * frame['z'])
        return w * frame['y_prior_mark'] + (1.0 - w) * frame['y_model']

    out_rows, per_trade, weight_curves = [], [], []
    months = sorted(app['month'].unique())
    for m in months[1:]:
        tr, te = app[app['month'] < m], app[app['month'] == m]
        if len(tr) < 2_000 or te.empty:
            continue
        # (a) one-parameter shrinkage as a check (grid now extends to 64)
        lam_mae = {lam: (100.0 * (tr['msrb_yield'] - blend_lambda(tr, lam))).abs().mean() for lam in CFG.shrink_lambda_grid}
        lam_star = min(lam_mae, key=lam_mae.get)
        # (b) inverse-variance weight on the mark by residual-size bin, and (c) the half-width around the model mid, both
        # estimated on training rows WITH THE SAME MODEL SOURCE as the test row (level-model months vs cohort-only months
        # have very different error scales; mixing them inflates the band).
        w_te = pd.Series(np.nan, index=te.index); half_te = pd.Series(np.nan, index=te.index); fixed_te = pd.Series(np.nan, index=te.index)
        a = b = s_scale = fixed_half = np.nan
        for src in te['model_src'].unique():
            tes_idx = te.index[te['model_src'] == src]
            if (tr['model_src'] == src).sum() < 500:
                continue   # no same-source training history yet (e.g. the first level-model month): leave these rows unweighted and unbanded
            trs = tr[tr['model_src'] == src]
            edges = np.unique(np.quantile(trs['z'], np.linspace(0, 1, 11))); edges[0], edges[-1] = -np.inf, np.inf
            tr_bin = pd.cut(trs['z'], edges, labels=False); te_bin = pd.cut(te.loc[tes_idx, 'z'], edges, labels=False)
            mse_mark = (trs['err_mark_bp'] ** 2).groupby(tr_bin).mean(); mse_model = (trs['err_model_bp'] ** 2).groupby(tr_bin).mean()
            w_bin = (1.0 / mse_mark) / (1.0 / mse_mark + 1.0 / mse_model)
            w_te.loc[tes_idx] = te_bin.map(w_bin).astype(float).fillna(float(w_bin.mean())).to_numpy()
            tr_err = trs['err_model_bp'].abs()
            bins = pd.DataFrame({'z': trs['z'].groupby(tr_bin).median(), 'err': tr_err.groupby(tr_bin).median()}).dropna()
            b_, a_ = np.polyfit(bins['z'], bins['err'], 1) if len(bins) > 2 else (0.0, float(tr_err.median())); a_, b_ = max(a_, 0.1), max(b_, 0.0)
            s_ = np.quantile(tr_err / (a_ + b_ * trs['z']), CFG.band_coverage); f_ = np.quantile(tr_err, CFG.band_coverage)
            half_te.loc[tes_idx] = (s_ * (a_ + b_ * te.loc[tes_idx, 'z'])).to_numpy(); fixed_te.loc[tes_idx] = f_
            weight_curves.append(pd.DataFrame({'month': str(m), 'model_src': src, 'z_bin': w_bin.index, 'z_mid': trs['z'].groupby(tr_bin).median().values, 'w_mark': w_bin.values, 'rmse_mark': np.sqrt(mse_mark.values), 'rmse_model': np.sqrt(mse_model.values)}))
            if src == te['model_src'].mode().iloc[0]:
                a, b, s_scale, fixed_half = a_, b_, s_, f_
        y_blend = w_te * te['y_prior_mark'] + (1.0 - w_te) * te['y_model']
        if w_te.isna().all():
            print(f'{m}: no same-source training rows yet; month skipped for the blend and band'); continue
        rec = te[['cusip', 'trade_date', 'trade_ts', 'side', 'recency_bucket', 'size_cell', 'msrb_yield', 'y_prior_mark', 'y_model', 'z', 'err_mark_bp', 'err_model_bp']].copy()
        rec['w_mark'] = w_te; rec['y_blend'] = y_blend; rec['err_blend_bp'] = 100.0 * (rec['msrb_yield'] - rec['y_blend'])
        rec['err_lambda_bp'] = 100.0 * (rec['msrb_yield'] - blend_lambda(te, lam_star)); rec['lambda'] = lam_star
        rec['half_width_bp'] = half_te; rec['fixed_half_bp'] = fixed_te; rec['model_src'] = te['model_src']
        rec['covered'] = (rec['err_model_bp'].abs() <= rec['half_width_bp']).astype(float); rec['covered_fixed'] = (rec['err_model_bp'].abs() <= rec['fixed_half_bp']).astype(float)
        # concession in yield terms around the model mid: dealer bid (P side, we buy) = mid + h, dealer offer (S side, we sell) = mid - h
        rec['bid_yield'] = rec['y_model'] + rec['half_width_bp'] / 100.0; rec['offer_yield'] = rec['y_model'] - rec['half_width_bp'] / 100.0
        rec['month'] = str(m)
        per_trade.append(rec)
        out_rows.append({'month': str(m), 'train': len(tr), 'test': len(te), 'lambda*': lam_star, 'mean w_mark (ivw)': float(w_te.mean()), 'band a': a, 'band b per z': b, 'band scale': s_scale,
                         'MAE mark': rec['err_mark_bp'].abs().mean(), 'MAE model': rec['err_model_bp'].abs().mean(), 'MAE blend ivw': rec['err_blend_bp'].abs().mean(), 'MAE blend lambda': rec['err_lambda_bp'].abs().mean(),
                         'coverage resid band': rec['covered'].mean(), 'half-width resid': rec['half_width_bp'].mean(), 'coverage fixed': rec['covered_fixed'].mean(), 'fixed half-width': rec['fixed_half_bp'].mean(), 'share level-model mid': float((te['model_src'] == 'level_model').mean())})
    appf = pd.DataFrame(out_rows).set_index('month'); apt = pd.concat(per_trade, ignore_index=True); wcurve = pd.concat(weight_curves, ignore_index=True)
    display(appf.round(3))

    def app_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        g = frame.groupby(by, observed=True)
        t = pd.DataFrame({'n': g.size(), 'MAE mark': g['err_mark_bp'].apply(lambda x: x.abs().mean()), 'MAE model': g['err_model_bp'].apply(lambda x: x.abs().mean()),
                          'MAE blend ivw': g['err_blend_bp'].apply(lambda x: x.abs().mean()), 'MAE blend lambda': g['err_lambda_bp'].apply(lambda x: x.abs().mean()), 'mean w_mark': g['w_mark'].mean(),
                          'coverage resid band': g['covered'].mean(), 'half-width resid (bp)': g['half_width_bp'].mean(), 'coverage fixed': g['covered_fixed'].mean(), 'fixed half-width (bp)': g['fixed_half_bp'].mean()})
        d = frame.assign(gain=frame['err_model_bp'].abs() - frame['err_blend_bp'].abs()).groupby([by, 'trade_date'], observed=True)['gain'].mean().groupby(level=0)
        t['blend gain vs model (bp)'] = d.mean(); t['gain t (FM)'] = (np.sqrt(d.count()) * d.mean() / d.std().replace(0, np.nan)).fillna(0.0)
        return t

    app_all = app_table(apt.assign(all='ALL'), 'all'); app_rec = app_table(apt, 'recency_bucket').reindex(rec_order + ['first_print']); app_side = app_table(apt, 'side'); app_src = app_table(apt, 'model_src')
    display(app_all.round(3)); display(app_src.round(3)); display(app_rec.round(3)); display(app_side.round(3))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    app_rec[['MAE mark', 'MAE model', 'MAE blend ivw']].plot.bar(ax=ax[0]); ax[0].set_title('Abs. error vs next print by recency: mark, model mid, ivw blend (bp)', fontsize=10); ax[0].set_xlabel('')
    wlast = wcurve[(wcurve['month'] == wcurve['month'].max()) & (wcurve['model_src'] == wcurve.loc[wcurve['month'] == wcurve['month'].max(), 'model_src'].mode().iloc[0])]
    ax[1].plot(wlast['z_mid'], wlast['rmse_mark'], marker='o', label='RMSE of mark vs print'); ax[1].plot(wlast['z_mid'], wlast['rmse_model'], marker='s', label='RMSE of model mid vs print'); ax[1].set_xlabel('|residual| / bond vol (bin medians, last training set)'); ax[1].set_ylabel('bp'); ax[1].legend(loc='upper left', fontsize=8); ax[1].set_title('Error variances by residual size and the implied trust weight', fontsize=10)
    ax2 = ax[1].twinx(); ax2.plot(wlast['z_mid'], wlast['w_mark'], color='grey', ls='--', label='inverse-variance weight on mark'); ax2.set_ylim(0, 1.05); ax2.legend(loc='lower right', fontsize=8)
    ax[2].bar(['resid band', 'fixed band'], [apt['covered'].mean(), apt['covered_fixed'].mean()], color=['#4C72B0', '#8C8C8C']); ax[2].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8)
    ax[2].set_ylim(0.5, 1.0); ax[2].set_title(f"Coverage around the model mid, target {CFG.band_coverage:.0%}: half-width {apt['half_width_bp'].mean():.1f} vs {apt['fixed_half_bp'].mean():.1f} bp", fontsize=10)
    savefig('10b_shrinkage_mid_and_band')
    sample_cols = ['cusip', 'trade_date', 'side', 'recency_bucket', 'y_prior_mark', 'y_model', 'w_mark', 'y_blend', 'half_width_bp', 'bid_yield', 'offer_yield', 'msrb_yield']
    print('Per-RFQ outputs (sample; bid = dealer buys from customer, offer = dealer sells to customer):'); display(apt.sample(min(8, len(apt)), random_state=CFG.seed)[sample_cols].round(4))
    apt.to_parquet(ARTIFACTS / 'quote_components_v2.parquet', index=False)
    record('application', model_source=model_source, by_month=appf.round(4).reset_index().to_dict(orient='records'), overall=app_all.round(4).to_dict(orient='records'),
           by_recency=app_rec.round(4).reset_index().to_dict(orient='records'), by_side=app_side.round(4).reset_index().to_dict(orient='records'), lambda_path=appf['lambda*'].tolist(),
           weight_curve_last=wlast.round(4).to_dict(orient='records'))
else:
    print('Section 10b skipped: needs matched trades.')

# %% [markdown]
# ## 10c. Conditional half-width: a quantile model of print dispersion around the model mid
#
# Section 10b showed that residual size does not widen the band once the mid is the level model: the residual
# measures how wrong the *mark* is, which the model mid has already corrected. What is left around the mid is
# print dispersion, and its drivers are the trade's own attributes (size, side, how long since the last print,
# how many prints), the mark's staleness and the bond's volatility. The half-width is therefore fitted directly:
# a gradient-boosted quantile regression of |print − model mid| at the target coverage, walk-forward by month,
# compared with the fixed band and the residual band at the same target. A well-calibrated conditional band has
# flat coverage across its own width deciles and a smaller mean width than the fixed band.

# %%
if HAS_TRADES and 'apt' in globals() and 'lv' in globals() and not apt.empty:
    bf = apt.merge(lv[['cusip', 'trade_ts', 'log_size', 'recency_days_c', 'prints_20d', 'MinuteFromSignal', 'last_print_spread_bp', 'modified_duration_lag1', 'rating_score', 'target_abs_vol', 'side_D', 'side_P', 'side_S'] + STALENESS_FEATURES + PEER_FEATURES].drop_duplicates(['cusip', 'trade_ts']), on=['cusip', 'trade_ts'], how='left')
    BAND_FEATURES = [c for c in ['log_size', 'side_D', 'side_P', 'side_S', 'recency_days_c', 'prints_20d', 'MinuteFromSignal', 'last_print_spread_bp', 'z', 'modified_duration_lag1', 'rating_score', 'target_abs_vol'] + STALENESS_FEATURES + PEER_FEATURES if c in bf.columns]
    bf['abs_err_model'] = bf['err_model_bp'].abs(); bf['month_p'] = pd.to_datetime(bf['trade_date']).dt.to_period('M')
    qparts, qimp = [], []
    for m in sorted(bf['month_p'].unique())[1:]:
        tr, te = bf[bf['month_p'] < m], bf[bf['month_p'] == m]
        if len(tr) < 2_000 or te.empty:
            continue
        bfeats = usable_features(tr, BAND_FEATURES)
        med = tr[bfeats].median(); Xtr, Xte = tr[bfeats].astype(float).fillna(med), te[bfeats].astype(float).fillna(med)
        if HAS_LGB:
            qm = lgb.LGBMRegressor(objective='quantile', alpha=CFG.band_coverage, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, random_state=CFG.seed, verbose=-1).fit(Xtr, tr['abs_err_model'])
            qimp.append(pd.Series(qm.feature_importances_, index=bfeats))
        else:
            qm = HistGradientBoostingRegressor(loss='quantile', quantile=CFG.band_coverage, max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=CFG.seed).fit(Xtr, tr['abs_err_model'])
        q = te[['cusip', 'trade_date', 'side', 'recency_bucket', 'abs_err_model', 'half_width_bp', 'fixed_half_bp', 'covered', 'covered_fixed']].copy()
        q['half_width_q'] = np.maximum(qm.predict(Xte), 0.5); q['covered_q'] = (q['abs_err_model'] <= q['half_width_q']).astype(float); q['month'] = str(m)
        qparts.append(q)
    if qparts:
        qb = pd.concat(qparts, ignore_index=True)
        qb['width_decile'] = pd.qcut(qb['half_width_q'].rank(method='first'), 10, labels=False) + 1

        def band_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
            g = frame.groupby(by, observed=True)
            return pd.DataFrame({'n': g.size(), 'coverage conditional': g['covered_q'].mean(), 'half-width conditional (bp)': g['half_width_q'].mean(), 'coverage resid band': g['covered'].mean(), 'half-width resid (bp)': g['half_width_bp'].mean(), 'coverage fixed': g['covered_fixed'].mean(), 'half-width fixed (bp)': g['fixed_half_bp'].mean()})

        qb_all = band_table(qb.assign(all='ALL'), 'all'); qb_side = band_table(qb, 'side'); qb_rec = band_table(qb, 'recency_bucket').reindex(rec_order).dropna(how='all'); qb_dec = band_table(qb, 'width_decile')
        print(f'Conditional band at target {CFG.band_coverage:.0%}: coverage and mean half-width vs the fixed and residual bands (held-out months)'); display(qb_all.round(3)); display(qb_side.round(3)); display(qb_rec.round(3))
        print('Calibration by predicted-width decile (coverage should be flat at the target):'); display(qb_dec[['n', 'coverage conditional', 'half-width conditional (bp)', 'coverage fixed']].round(3))
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
        ax[0].plot(qb_dec.index, qb_dec['coverage conditional'], 'o-', label='conditional band'); ax[0].plot(qb_dec.index, qb_dec['coverage fixed'], 's--', color='grey', label='fixed band'); ax[0].axhline(CFG.band_coverage, color='k', ls='--', lw=0.8); ax[0].set_xlabel('predicted half-width decile'); ax[0].set_ylim(0.5, 1.0); ax[0].legend(fontsize=8); ax[0].set_title('Coverage by predicted-width decile: calibration', fontsize=10)
        ax[1].bar(qb_dec.index, qb_dec['half-width conditional (bp)'], color='#4C72B0'); ax[1].axhline(qb['fixed_half_bp'].mean(), color='grey', ls='--', label='fixed half-width'); ax[1].legend(fontsize=8); ax[1].set_xlabel('predicted half-width decile'); ax[1].set_title('Mean conditional half-width by decile (bp)', fontsize=10)
        if qimp:
            pd.concat(qimp, axis=1).mean(axis=1).sort_values().tail(12).plot.barh(ax=ax[2], color='#55A868'); ax[2].set_title('What drives print dispersion around the mid: quantile-model importance', fontsize=10)
        savefig('10c_conditional_band')
        record('conditional_band', features=BAND_FEATURES, overall=qb_all.round(4).to_dict(orient='records'), by_side=qb_side.round(4).reset_index().to_dict(orient='records'), by_recency=qb_rec.round(4).reset_index().to_dict(orient='records'), by_width_decile=qb_dec.round(4).reset_index().to_dict(orient='records'))
    else:
        print('Section 10c: not enough training months for the quantile band.')
else:
    print('Section 10c skipped: needs the level model and the 10b per-trade frame.')

# %% [markdown]
# ## 11. Systematic path: roll-forward and de-circularised comparables
#
# *Roll-forward*: moving a mark by the cumulative factor-implied change versus leaving it unchanged, by horizon.
# In-panel, so an upper bound. *Comparables*: nearest neighbours in standardised beta space where Gamma is refit
# **without** the yield-level and premium/discount instruments, so peers are not close in yield by construction;
# the peer-implied yield level is compared with random peers.

# %%
r_sorted = resid.sort_values(['cusip', 'date'], kind='stable')
g = r_sorted.groupby('cusip', observed=True)
rf_rows = []
for h in [1, 2, 3, 5, 10]:
    cum_target = g['target_bp'].transform(lambda s: s.rolling(h).sum().shift(-h + 1))   # sum over t+1..t+h aligned at t+1 start
    cum_resid = g['pit_residual'].transform(lambda s: s.rolling(h).sum().shift(-h + 1))
    ok = cum_target.notna() & cum_resid.notna()
    rf_rows.append({'horizon': h, 'n': int(ok.sum()), 'stale_mae_bp': cum_target[ok].abs().mean(), 'rolled_mae_bp': cum_resid[ok].abs().mean(), 'stale_rmse_bp': np.sqrt((cum_target[ok] ** 2).mean()), 'rolled_rmse_bp': np.sqrt((cum_resid[ok] ** 2).mean())})
rollf = pd.DataFrame(rf_rows).set_index('horizon'); rollf['rmse_reduction'] = 1 - rollf['rolled_rmse_bp'] / rollf['stale_rmse_bp']
display(rollf)

# de-circularised peers on the last OOS date
level_instr = {'z_closing_yield_lag1', 'z_premium_discount_lag1'}
chars_nl = [c for c in CHARS if c not in level_instr]
last_fold = folds[-1]
train_nl = model[model['date'].isin(last_fold['train_dates'])]
ipca_nl = MomentIPCA(CFG.selected_k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(precompute_moments(train_nl, chars_nl, TARGET))
last_date = resid['date'].max()
snap = model[model['date'] == last_date].dropna(subset=['closing_yield']).reset_index(drop=True)
B_nl = snap[chars_nl].to_numpy(float) @ ipca_nl.Gamma
B_std = (B_nl - B_nl.mean(axis=0)) / np.maximum(B_nl.std(axis=0), 1e-12)
from sklearn.neighbors import NearestNeighbors  # noqa: E402

K_PEERS = 20
n_q = min(len(snap), 5_000)
q_idx = RNG.choice(len(snap), n_q, replace=False)
nn = NearestNeighbors(n_neighbors=K_PEERS + 1).fit(B_std)
dist, idx = nn.kneighbors(B_std[q_idx])
dist, idx = dist[:, 1:], idx[:, 1:]
wts = 1.0 / (dist + 1e-6); wts = wts / wts.sum(axis=1, keepdims=True)
own = snap.loc[q_idx, 'closing_yield'].to_numpy(float)
peer_level = (snap['closing_yield'].to_numpy(float)[idx] * wts).sum(axis=1)
rand_idx = RNG.integers(0, len(snap), size=idx.shape)
rand_level = snap['closing_yield'].to_numpy(float)[rand_idx].mean(axis=1)
dur = snap['modified_duration_lag1'].to_numpy(float)
peer_dur_gap = np.abs(dur[idx] - dur[q_idx, None]).mean(); rand_dur_gap = np.abs(dur[rand_idx] - dur[q_idx, None]).mean()
peers = pd.Series({'date': str(last_date.date()), 'bonds_priced': n_q, 'peers': K_PEERS, 'instruments_in_embedding': len(chars_nl),
                   'peer_implied_mae_bp': 100 * np.abs(own - peer_level).mean(), 'random_peer_mae_bp': 100 * np.abs(own - rand_level).mean(),
                   'peer_duration_gap_yrs': peer_dur_gap, 'random_duration_gap_yrs': rand_dur_gap, 'peer_spread_sd_bp': 100 * np.std(own - peer_level)})
display(peers.to_frame('value'))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(rollf.index, rollf['stale_mae_bp'], marker='o', label='leave mark stale'); ax[0].plot(rollf.index, rollf['rolled_mae_bp'], marker='o', label='roll forward by beta·f'); ax[0].set_xlabel('horizon (observations)'); ax[0].set_ylabel('MAE (bp)'); ax[0].legend(); ax[0].set_title('Factor roll-forward of marks (in-panel upper bound)')
ax[1].scatter(peer_level, own, s=4, alpha=0.3); lim = [np.nanpercentile(own, 1), np.nanpercentile(own, 99)]; ax[1].plot(lim, lim, 'k--', lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel('peer-implied yield (%)'); ax[1].set_ylabel('own closing yield (%)'); ax[1].set_title(f'De-circularised beta-space peers, {last_date.date()}')
savefig('11_systematic_path')
record('systematic_path', roll_forward=rollf.round(4).reset_index().to_dict(orient='records'), peers={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in peers.items()})

# %% [markdown]
# ## 11b. Factor exposure of dealer flow
#
# The factor-mimicking portfolios of Section 6 are hedge instruments. This cell asks how much of the risk a dealer
# takes on from customer flow is factor risk. Daily net dealer flow per bond is par bought from customers (P) minus
# par sold to customers (S). The flow-weighted yield change that day decomposes exactly into a factor part
# ($\sum_i w_i \beta_i' f_t$) and a residual part; the share of its variance that is factor variance is what a
# three-factor hedge would remove, and the day's exposure vector $\sum_i w_i \beta_i$ is what the hedge would be
# sized on. In-panel and contemporaneous: a risk decomposition, not a forecast.

# %%
if HAS_TRADES and not tv.empty:
    fl = trades[trades['side'].isin(['P', 'S'])].copy()
    fl['signed_par'] = np.where(fl['side'] == 'P', 1.0, -1.0) * fl['msrb_quantity'].fillna(0).clip(lower=0)
    flow = fl.groupby(['cusip', 'trade_date'], observed=True)['signed_par'].sum().reset_index().rename(columns={'trade_date': 'date'})
    flow['date'] = to_ns(flow['date'])
    fx = flow.merge(resid[['cusip', 'date', 'target_bp', 'fitted_bp', 'pit_residual'] + beta_cols], on=['cusip', 'date'], how='inner')
    fx = fx[fx['signed_par'] != 0]
    rows_fx = []
    for d, g in fx.groupby('date', observed=True):
        if len(g) < 30:
            continue
        w = g['signed_par'].to_numpy(float); gross = np.abs(w).sum(); w = w / gross
        B = g[beta_cols].to_numpy(float)
        rows_fx.append({'date': d, 'bonds': len(g), 'gross_par': gross, 'net_share': float(w.sum()), 'flow_pnl_bp': float(w @ g['target_bp'].to_numpy(float)), 'factor_part_bp': float(w @ g['fitted_bp'].to_numpy(float)), 'resid_part_bp': float(w @ g['pit_residual'].to_numpy(float)), **{f'exposure_{j+1}': float(w @ B[:, j]) for j in range(CFG.selected_k)}})
    fxd = pd.DataFrame(rows_fx).set_index('date')
    if len(fxd) > 10:
        r2 = 1.0 - fxd['resid_part_bp'].var() / fxd['flow_pnl_bp'].var()
        fx_summary = pd.Series({'days': len(fxd), 'bonds per day': fxd['bonds'].mean(), 'mean net flow share (P - S over gross)': fxd['net_share'].mean(), 'sd of flow-weighted yield change (bp)': fxd['flow_pnl_bp'].std(), 'sd of factor part (bp)': fxd['factor_part_bp'].std(), 'sd of residual part (bp)': fxd['resid_part_bp'].std(), 'share of flow variance that is factor variance': r2, **{f'mean |exposure_{j+1}|': fxd[f'exposure_{j+1}'].abs().mean() for j in range(CFG.selected_k)}})
        display(fx_summary.to_frame('value').round(4))
        fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
        fxd[[f'exposure_{j+1}' for j in range(CFG.selected_k)]].plot(ax=ax[0]); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_title('Factor exposure of the day\'s net dealer flow (per unit gross par)', fontsize=10); ax[0].set_xlabel('')
        ax[1].plot(fxd.index, fxd['flow_pnl_bp'].cumsum(), label='flow-weighted yield change'); ax[1].plot(fxd.index, fxd['factor_part_bp'].cumsum(), label='factor part'); ax[1].plot(fxd.index, fxd['resid_part_bp'].cumsum(), label='residual part (what a factor hedge leaves)'); ax[1].legend(fontsize=8); ax[1].set_title(f'Cumulative decomposition (bp per unit gross); factor share of variance {r2:.0%}', fontsize=10); ax[1].set_xlabel('')
        savefig('11b_flow_factor_exposure')
        record('flow_exposure', **{k.replace(' ', '_'): float(v) for k, v in fx_summary.items()})
    else:
        print('Section 11b: too few days with flow matched to residuals.')
else:
    print('Section 11b skipped: needs matched trades.')

# %% [markdown]
# ## 12. Results registry and summary

# %%
score = resid[['cusip', 'date', 'gamma_version', 'fold', 'target_bp', 'fitted_bp', 'pit_residual', 'activity_bucket'] + beta_cols].copy()
score['pit_residual_rank'] = score.groupby('date', observed=True)['pit_residual'].rank(pct=True)
score['signal_available_date'] = score['date'] + pd.offsets.BDay(1)
score['spec_version'] = CFG.spec_version
score.to_parquet(ARTIFACTS / 'pit_residual_score_v2.parquet', index=False)
(ARTIFACTS / 'results_registry.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')

summary_rows = [
    ('Model panel', f"{REGISTRY['model_panel']['rows']:,} rows, {REGISTRY['model_panel']['cusips']:,} CUSIPs, {REGISTRY['model_panel']['dates']} dates, L={len(CHARS)}"),
    ('Extreme-yield rows excluded', f"{REGISTRY['data']['extreme_yield_rows']:,} ({REGISTRY['data']['extreme_yield_share']:.3%})"),
    ('Rating blank after fallback', f"{REGISTRY['data']['after_fallback_blank_share']:.1%} (final only: {REGISTRY['data']['final_comp_rating_blank_share']:.1%})"),
    ('OOS variance explained (K=3)', f"{REGISTRY['walk_forward']['oos_variance_explained']:.3f}; residual sd {REGISTRY['walk_forward']['residual_sd_bp']:.2f} bp over {REGISTRY['walk_forward']['gamma_versions']} Gamma versions"),
    ('Factor variance shares', ', '.join(f'{s:.1%}' for s in REGISTRY['ipca_full_sample']['factor_variance_share'])),
    ('Factor-1 gross leverage', f"{REGISTRY['mimicking_weights'].get('factor1_gross', float('nan')):.2f}"),
    ('Residual ACF lag 1 / 5 (Pearson)', f"{acf.loc[1, 'pearson']:+.3f} / {acf.loc[min(5, L), 'pearson']:+.3f}"),
    ('Best forecaster', f"{best}: rank IC {fc_table.loc[best, 'ic_mean']:+.3f} (t {fc_table.loc[best, 'ic_t']:+.2f})"),
    ('Paper portfolio hedged Sharpe (ALL)', '; '.join(f"{k}: {v:.2f}" for k, v in paper['ALL'].items()) + (f"; walk-forward selected {sharpe(pnl_selected):.2f}" if len(pnl_selected) else '')),
]
if HAS_TRADES and not tv.empty:
    tm = tx[(tx['target'] == 'e vs prior mark') & (tx['bucket'] == 'ALL')].iloc[0]; ta = tx[(tx['target'] == 'e vs algo quote')].iloc[0]; tn = tx[tx['target'].str.startswith('e vs prior mark, neutralised')].iloc[0]
    summary_rows += [('Trade vs prior mark on residual rank', f"FM {tm['fm_beta']:+.2f} bp/rank (t {tm['fm_t']:+.2f}); neutralised {tn['fm_beta']:+.2f} (t {tn['fm_t']:+.2f})"),
                     ('Trade vs algo quote on residual rank', f"FM {ta['fm_beta']:+.2f} bp/rank (t {ta['fm_t']:+.2f})")]
if 'level_model' in REGISTRY:
    o = REGISTRY['level_model']['overall'][0]
    summary_rows.append(('Level model vs algo quote (all trades)', f"MAE algo {o['MAE algo quote']:.2f} | mark {o['MAE prior mark']:.2f} | model {o['MAE level model']:.2f} bp; gain {o['gain vs algo (bp)']:+.2f} (t {o['gain t (FM)']:+.2f})"))
if HAS_TRADES and not tv.empty:
    sp = spread_tx.iloc[0]
    summary_rows.append(('Spread reading: |trade - mark| on residual rank', f"FM {sp['fm_beta']:+.2f} bp/rank (t {sp['fm_t']:+.2f})"))
    summary_rows.append(('Evaluator revision toward trade, small vs large |residual|', f"{evt['share_revised_toward_trade'].iloc[0]:.0%} -> {evt['share_revised_toward_trade'].iloc[-1]:.0%}; pass-through {evt['median_pass_through'].iloc[0]:.2f} -> {evt['median_pass_through'].iloc[-1]:.2f}"))
if HAS_TRADES and not tv.empty:
    jm = jt[jt['target'] == 'e vs prior mark'].set_index('score')
    summary_rows.append(('Joint FM on trade vs mark: one-day rank | trailing-mean rank', f"{jm.loc['pit_rank', 'fm_beta']:+.2f} (t {jm.loc['pit_rank', 'fm_t']:+.2f}) | {jm.loc['level_rank', 'fm_beta']:+.2f} (t {jm.loc['level_rank', 'fm_t']:+.2f})"))
if 'application' in REGISTRY:
    a = REGISTRY['application']['overall'][0]
    summary_rows.append(('Mid vs next print (held out)', f"MAE mark {a['MAE mark']:.2f} | model mid {a['MAE model']:.2f} | ivw blend {a['MAE blend ivw']:.2f} | lambda blend {a['MAE blend lambda']:.2f} bp; mean mark weight {a['mean w_mark']:.2f}; blend gain vs model {a['blend gain vs model (bp)']:+.2f} (t {a['gain t (FM)']:+.2f})"))
    summary_rows.append(('Band around model mid: residual vs fixed', f"coverage {a['coverage resid band']:.1%} at half-width {a['half-width resid (bp)']:.1f} bp vs {a['coverage fixed']:.1%} at {a['fixed half-width (bp)']:.1f} bp"))
if 'gamma_swap_test' in REGISTRY['walk_forward']:
    _gs = gswap.dropna(); summary_rows.append(('Gamma swap test (mean OOS var explained)', f"in force {_gs['in force'].mean():.3f} | frozen 2m earlier {_gs['frozen 2 months earlier'].mean():.3f} | first OOS Gamma {_gs['first OOS Gamma'].mean():.3f}"))
summary_rows.append(('Repricing days excluded from the fit', f"{len(REPRICING_DATES)} dates; residual ACF lag 1 Pearson ex repricing {acf.loc[1, 'pearson_ex_repricing']:+.3f}"))
if 'peer_rv' in REGISTRY:
    summary_rows.append(('Peer RV (chars-space kNN, PIT)', f"peer-implied MAE {REGISTRY['peer_rv']['peer_mae_bp']:.1f} bp vs random {REGISTRY['peer_rv']['random_mae_bp']:.1f}; corr with one-day residual {REGISTRY['peer_rv']['corr_with_residual']:+.3f}"))
for nm in [k for k in fc_table.index if k.startswith('State-space') or k.startswith('ARMA') or k.startswith('Peer RV')]:
    summary_rows.append((f'Forecaster: {nm}', f"rank IC {fc_table.loc[nm, 'ic_mean']:+.3f} (t {fc_table.loc[nm, 'ic_t']:+.2f}); ex repricing {fc_table.loc[nm, 'ic_ex_repricing']:+.3f}"))
if not ssmb_params.empty:
    summary_rows.append(('State-space parameters by bucket (last fold)', '; '.join(f"{g}: phi {r['phi']:.2f}, drift/noise {r['signal_to_noise']:.2f}" for g, r in lastp.iterrows() if g not in ('UNKNOWN',))))
if 'print_to_print' in REGISTRY:
    _r1 = p2p_reg[p2p_reg['regression'].str.startswith('print change on factor')].set_index('score')
    summary_rows.append(('Print-to-print: next print on factor move | residual move | prior gap', f"{_r1.loc['d_fit_bp', 'fm_beta']:+.2f} (t {_r1.loc['d_fit_bp', 'fm_t']:+.1f}) | {_r1.loc['d_res_bp', 'fm_beta']:+.2f} (t {_r1.loc['d_res_bp', 'fm_t']:+.1f}) | {_r1.loc['e_mark_bp_0', 'fm_beta']:+.2f} (t {_r1.loc['e_mark_bp_0', 'fm_t']:+.1f})"))
    summary_rows.append(('Next-print MAE: stale print | print + factor | prior mark | algo', ' | '.join(f"{p2p_mae.loc['ALL', c]:.1f}" for c in ['stale print', 'print + factor move', 'prior mark', 'algo quote']) + ' bp'))
if 'economic_sizing' in REGISTRY:
    _e = es_all.iloc[0]; _v = list(versions)[-1]
    summary_rows.append(('Skew on algo quote (walk-forward)', f"MAE algo {_e['MAE algo (bp)']:.2f} -> adjusted {_e[f'MAE adj [{_v}]']:.2f} bp; gain {_e[f'gain [{_v}] bp']:+.2f} (t {_e[f'gain t [{_v}]']:+.1f}); hit rate {_e[f'hit rate [{_v}]']:.1%} [{_v}]"))
if 'level_model' in REGISTRY and 'ablation' in REGISTRY['level_model']:
    _a = ablation['gain vs algo (bp)']
    summary_rows.append(('Level-model gain vs algo by feature set', '; '.join(f"{k}: {v:+.2f}" for k, v in _a.items())))
if 'conditional_band' in REGISTRY:
    _q = qb_all.iloc[0]
    summary_rows.append(('Conditional band vs fixed (same target)', f"coverage {_q['coverage conditional']:.1%} at {_q['half-width conditional (bp)']:.1f} bp vs fixed {_q['coverage fixed']:.1%} at {_q['half-width fixed (bp)']:.1f} bp"))
if 'flow_exposure' in REGISTRY:
    summary_rows.append(('Dealer flow: share of flow-weighted variance that is factor variance', f"{REGISTRY['flow_exposure']['share_of_flow_variance_that_is_factor_variance']:.0%}"))
summary = pd.DataFrame(summary_rows, columns=['item', 'result']).set_index('item')
display(summary)
print('Artifacts written to', ARTIFACTS)
for p in sorted(ARTIFACTS.glob('*')):
    if p.is_file():
        print('  ', p.name, f'{p.stat().st_size/1e6:.2f} MB')
print('Figures:', len(list(FIGURES.glob('*.png'))))


# %% [markdown]
# ## 13. Robustness: K sensitivity, repricing-day exclusions, bootstrap inference, pre-registered choices
#
# Three questions a referee would ask. (1) Does a fourth factor absorb the curve move the residual still carries on
# stress days? The walk-forward is rerun with K from `robustness_k` and the daily Spearman correlation of the
# residual with duration (the "duration tilt" of the residual) is compared on normal and repricing days. (2) Do the
# residual results survive dropping the repricing days? (3) Do the Fama-MacBeth t-statistics survive dependence
# across dates? The block-bootstrap t is shown next to the FM t for every key row. The pre-registered choices
# close the section.

# %%
import dataclasses  # noqa: E402

dur_map = model[['cusip', 'date', 'modified_duration_lag1']]


def duration_tilt(res_frame: pd.DataFrame) -> pd.Series:
    f = res_frame[['cusip', 'date', 'pit_residual']].merge(dur_map, on=['cusip', 'date'], how='left').dropna()
    return f.groupby('date', observed=True).apply(lambda g: g['pit_residual'].corr(g['modified_duration_lag1'], method='spearman') if len(g) > 100 else np.nan, include_groups=False)


tilt = {f'K={CFG.selected_k}': duration_tilt(resid)}
k_month = {f'K={CFG.selected_k}': rm['oos_var_explained']}
for k in CFG.robustness_k:
    if k == CFG.selected_k:
        continue
    t0 = time.perf_counter()
    cfg_k = dataclasses.replace(CFG, selected_k=k)
    resid_k, _, _ = walk_forward_residuals(model, CHARS, folds, cfg_k)
    print(f'K={k}: OOS variance explained {1 - resid_k["pit_residual"].var() / resid_k["target_bp"].var():.3f} | {time.perf_counter()-t0:.0f}s')
    k_month[f'K={k}'] = resid_k.assign(month=resid_k['date'].dt.to_period('M').astype(str)).groupby('month').apply(lambda g: 1 - g['pit_residual'].var() / g['target_bp'].var(), include_groups=False)
    tilt[f'K={k}'] = duration_tilt(resid_k)
    del resid_k
k_tab = pd.DataFrame(k_month); tilt_df = pd.DataFrame(tilt)
tilt_df['repricing'] = tilt_df.index.isin(list(REPRICING_DATES))
tilt_tab = tilt_df.groupby('repricing').agg(lambda s: s.abs().mean()).rename(index={False: 'normal days', True: 'repricing days'})
print('OOS variance explained by month and K:'); display(k_tab.round(3))
print('Mean |Spearman(residual, duration)| by date type and K (a curve move the factors missed shows up here):'); display(tilt_tab.round(3))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
k_tab.plot.bar(ax=ax[0]); ax[0].set_title('OOS variance explained by month: K sensitivity'); ax[0].set_xlabel('')
for c in [c for c in tilt_df.columns if c.startswith('K=')]:
    ax[1].plot(tilt_df.index, tilt_df[c], lw=1, label=c)
for d in REPRICING_DATES:
    ax[1].axvline(d, color='#C44E52', lw=0.6, alpha=0.5)
ax[1].axhline(0, color='k', lw=0.6); ax[1].legend(fontsize=8); ax[1].set_title('Daily duration tilt of the residual (Spearman); red = repricing days'); ax[1].set_xlabel('')
savefig('13_k_sensitivity_and_tilt')
record('robustness', k_by_month=k_tab.round(4).to_dict(), duration_tilt=tilt_tab.round(4).to_dict())

# %%
# Inference robustness: FM t vs block-bootstrap t for the key transaction rows; residual results with and without repricing days
rob_rows = [{'check': 'Residual ACF lag 1 (Pearson)', 'all days': acf.loc[1, 'pearson'], 'ex repricing days': acf.loc[1, 'pearson_ex_repricing']},
            {'check': 'Residual ACF lag 1 (Spearman)', 'all days': acf.loc[1, 'spearman'], 'ex repricing days': acf.loc[1, 'spearman_ex_repricing']}]
for nm in fc_table.index:
    rob_rows.append({'check': f'Rank IC: {nm}', 'all days': fc_table.loc[nm, 'ic_mean'], 'ex repricing days': fc_table.loc[nm, 'ic_ex_repricing']})
_tail = two_regime.iloc[[0, -1]]
rob_rows.append({'check': 'Two-regime: continuation ratio, cheap tail', 'all days': _tail['continuation_ratio'].iloc[-1], 'ex repricing days': _tail['continuation_ex_repricing'].iloc[-1]})
rob_rows.append({'check': 'Two-regime: continuation ratio, rich tail', 'all days': _tail['continuation_ratio'].iloc[0], 'ex repricing days': _tail['continuation_ex_repricing'].iloc[0]})
robust = pd.DataFrame(rob_rows).set_index('check')
print('Residual results with and without repricing days:'); display(robust.round(4))
if HAS_TRADES and not tv.empty:
    key = tx[tx['bucket'] == 'ALL'][['target', 'fm_beta', 'fm_t', 'boot_t', 'fm_dates']].set_index('target')
    key['t ratio (boot / FM)'] = key['boot_t'] / key['fm_t']
    print('Fama-MacBeth t vs moving-block bootstrap t (5-day blocks) for the ALL rows:'); display(key.round(3))
    record('robustness', fm_vs_bootstrap=key.round(4).reset_index().to_dict(orient='records'))
print('Pre-registered choices (fixed before this run):'); display(pd.Series(PRE_REGISTERED).to_frame('value'))
record('robustness', residual_checks=robust.round(4).reset_index().to_dict(orient='records'), pre_registered=PRE_REGISTERED)
(ARTIFACTS / 'results_registry.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')
print('registry updated:', ARTIFACTS / 'results_registry.json')
