# %% [markdown]
# # Muni Characteristic Factor Model v2 — IPCA Residuals, Level Relative Value, Market-Making Signals
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
# 11. Systematic path: factor roll-forward and de-circularised beta-space comparables
# 12. Results registry and summary
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
    spec_version: str = 'step3_yield_ipca_k3_v2'
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
    seed: int = 20260921


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
# Thirteen instruments, identical to the v1 spec (DV01 dropped): an intercept, seven rank-normalised continuous
# characteristics (lagged modified duration, premium/discount, yield level, years to worst, extension, rating
# score, trailing mark-change liquidity), the NR flag and four state dummies. All continuous instruments are
# rank-normalised within date to [-0.5, 0.5]. The target is the daily closing-yield change in bp, clipped.

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
    yp['rating'], yp['rating_score'], yp['rating_source'] = rating_with_fallback(yp)
    yp['nr_flag'] = yp['rating_score'].isna().astype(float)
    state = yp['state_code'].astype('string').str.strip().str.upper().fillna('OTHER')
    for s in STATE_DUMMIES:
        yp[f'state_{s}'] = (state == s).astype(float)
    yp = yp.sort_values(['cusip', 'date'], kind='stable')
    chg = (yp.groupby('cusip', observed=True)['closing_price'].diff().abs() > 0).astype(float)
    yp['liquidity_20'] = chg.groupby(yp['cusip'], observed=True).transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
    base = ['modified_duration_lag1', 'premium_discount_lag1', 'closing_yield_lag1']
    cont = base + ['years_to_worst', 'extension', 'rating_score', 'liquidity_20']
    yp = yp.dropna(subset=base + [TARGET]).copy()
    yp['target_bp_raw'] = yp[TARGET]
    yp[TARGET] = yp[TARGET].clip(-cfg.target_clip_bp, cfg.target_clip_bp)
    yp['mark_unchanged'] = (yp['target_bp_raw'] == 0).astype(float)
    yp, zcols = rank_normalize(yp, cont)
    yp['market_fv'] = 1.0
    chars = ['market_fv'] + zcols + ['nr_flag'] + [f'state_{s}' for s in STATE_DUMMIES]
    cnt = yp.groupby('cusip', observed=True)[TARGET].transform('count')
    yp = yp[cnt >= cfg.min_obs_per_bond].copy()
    keep = ['cusip', 'date', TARGET, 'target_bp_raw', 'mark_unchanged', 'closing_lag_days', 'closing_yield', 'closing_yield_lag1',
            'closing_price', 'modified_duration_lag1', 'dv01_lag1', 'years_to_worst', 'extension', 'rating', 'rating_score', 'rating_source',
            'cpn', 'state_code', 'mkt_yield'] + chars
    keep = [c for c in dict.fromkeys(keep) if c in yp.columns]
    return yp[keep].sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True), chars


t0 = time.perf_counter()
model, CHARS = build_model_panel(raw_panel, CFG)
print(f'[{CFG.spec_version}] model rows {len(model):,} | cusips {model["cusip"].nunique():,} | dates {model["date"].nunique()} | L={len(CHARS)} | {time.perf_counter()-t0:.1f}s')
print('instruments:', CHARS)
print('mark_unchanged share:', round(float(model['mark_unchanged'].mean()), 4), '| rating source mix:', model['rating_source'].value_counts(normalize=True).round(3).to_dict())
display(model[CHARS].describe().T[['mean', 'std', 'min', 'max']])
record('model_panel', rows=len(model), cusips=model['cusip'].nunique(), dates=model['date'].nunique(), instruments=CHARS, mark_unchanged_share=float(model['mark_unchanged'].mean()))

# %%
# Instrument EDA: correlation structure and the duration vs term relationship
zc = [c for c in CHARS if c.startswith('z_')]
sample = model.sample(min(len(model), 200_000), random_state=CFG.seed)
corr = sample[zc + ['nr_flag'] + [f'state_{s}' for s in STATE_DUMMIES]].corr(method='spearman')
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
im = ax[0].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax[0].set_xticks(range(len(corr))); ax[0].set_xticklabels([c.replace('z_', '').replace('_lag1', '') for c in corr.columns], rotation=90)
ax[0].set_yticks(range(len(corr))); ax[0].set_yticklabels([c.replace('z_', '').replace('_lag1', '') for c in corr.columns])
ax[0].set_title('Spearman correlation of instruments'); ax[0].grid(False); plt.colorbar(im, ax=ax[0], fraction=0.046)
hb = ax[1].hexbin(sample['years_to_worst'].clip(0, 40), sample['modified_duration_lag1'].clip(0, 25), gridsize=45, cmap='viridis', mincnt=1)
ax[1].set_xlabel('years to worst'); ax[1].set_ylabel('modified duration (lag 1)'); ax[1].set_title('Duration vs term: the call structure wedge'); plt.colorbar(hb, ax=ax[1], fraction=0.046)
savefig('03_instrument_eda')
print('max |corr| off-diagonal:', round(float((corr.values - np.eye(len(corr))).max()), 3))

# %% [markdown]
# ## 4. IPCA: K sweep, Gamma anatomy, factor paths
#
# $r_{i,t} = z_{i,t-1}'\Gamma f_t + \varepsilon_{i,t}$, estimated by alternating least squares on per-date
# cross-products. Identification: $\Gamma'\Gamma = I$, factors ordered by variance, non-negative factor means.
# The full-sample fit below is for interpretation and K selection only; everything downstream uses the
# walk-forward Gamma.

# %%
def precompute_moments(frame: pd.DataFrame, chars: list[str], target: str) -> dict:
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    L = len(chars)
    A = np.zeros((len(dates), L, L)); b = np.zeros((len(dates), L)); rr = np.zeros(len(dates)); nobs = np.zeros(len(dates), dtype=np.int64)
    pos = {d: i for i, d in enumerate(dates)}
    for d, g in frame.groupby('date', sort=True, observed=True):
        i = pos[pd.Timestamp(d)]
        Z = g[chars].to_numpy(float); r = g[target].to_numpy(float)
        A[i] = Z.T @ Z; b[i] = Z.T @ r; rr[i] = float(r @ r); nobs[i] = len(g)
    return {'dates': dates, 'A': A, 'b': b, 'rr': rr, 'nobs': nobs, 'chars': list(chars)}


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

k_rows = []
fits = {}
for k in CFG.candidate_k:
    mk = MomentIPCA(k, CFG.max_iter, CFG.tol, CFG.ridge, CFG.seed).fit(moments)
    fits[k] = mk
    k_rows.append({'K': k, 'in_sample_var_explained': mk.variance_explained(moments), 'iterations': len(mk.loss_history)})
k_table = pd.DataFrame(k_rows).set_index('K')
display(k_table)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k_table.index, k_table['in_sample_var_explained'], marker='o'); ax.set_xlabel('K'); ax.set_ylabel('variance explained (in-sample)'); ax.set_title('K sweep (full sample, interpretation only)')
savefig('04_k_sweep')
record('ipca_full_sample', **{f'var_explained_K{k}': float(v) for k, v in k_table['in_sample_var_explained'].items()})

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


def walk_forward_residuals(frame: pd.DataFrame, chars: list[str], folds: list[dict], cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts, gammas = [], []
    gamma_init = None
    by_date = {d: g for d, g in frame.groupby('date', sort=True, observed=True)}
    eye = np.eye(cfg.selected_k)
    for fo in folds:
        t0 = time.perf_counter()
        train = frame[frame['date'].isin(fo['train_dates'])]
        mk = MomentIPCA(cfg.selected_k, cfg.max_iter, cfg.tol, cfg.ridge, cfg.seed).fit(precompute_moments(train, chars, TARGET), gamma_init=gamma_init)
        gamma_init = mk.Gamma
        gv = pd.Timestamp(fo['train_end'])
        gammas.append(pd.DataFrame(mk.Gamma, index=chars, columns=[f'factor{i+1}' for i in range(cfg.selected_k)]).assign(gamma_version=gv, fold=fo['fold']))
        for d in fo['test_dates']:
            g = by_date.get(d)
            if g is None:
                continue
            Z = g[chars].to_numpy(float); r = g[TARGET].to_numpy(float)
            B = Z @ mk.Gamma
            f = np.linalg.solve(B.T @ B + cfg.ridge * eye, B.T @ r)
            out = g[['cusip', 'date']].copy()
            out['gamma_version'] = gv; out['fold'] = fo['fold']; out['target_bp'] = r; out['fitted_bp'] = B @ f; out['pit_residual'] = r - out['fitted_bp']
            for j in range(cfg.selected_k):
                out[f'beta{j+1}'] = B[:, j]; out[f'f{j+1}'] = f[j]
            parts.append(out)
        print(f'fold {fo["fold"]:02d} | train ->{fo["train_end"].date()} | test {fo["test_start"].date()}..{fo["test_end"].date()} | {time.perf_counter()-t0:.1f}s')
    return pd.concat(parts, ignore_index=True), pd.concat(gammas)


folds = make_weekly_folds(moments['dates'], CFG.train_start, CFG.first_oos_date, CFG.refit_days)
print(f'{len(folds)} walk-forward folds; first test {folds[0]["test_start"].date()}, last test {folds[-1]["test_end"].date()}')
t0 = time.perf_counter()
resid, gamma_versions = walk_forward_residuals(model, CHARS, folds, CFG)
print(f'PIT residuals: {len(resid):,} rows | {resid["cusip"].nunique():,} cusips | {resid["date"].nunique()} dates | {resid["gamma_version"].nunique()} gamma versions | {time.perf_counter()-t0:.1f}s')
oos_var_expl = 1.0 - resid['pit_residual'].var() / resid['target_bp'].var()
print(f'OOS variance explained {oos_var_expl:.3f} | residual sd {resid["pit_residual"].std():.2f} bp')
resid.to_parquet(ARTIFACTS / 'pit_ipca_residuals_v2.parquet', index=False)
gamma_versions.reset_index().rename(columns={'index': 'instrument'}).to_parquet(ARTIFACTS / 'gamma_versions_v2.parquet', index=False)
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
gv_f1.plot(ax=ax[2], marker='.'); ax[2].set_title('Factor-1 loadings by Gamma version (top instruments)'); ax[2].axhline(0, color='k', lw=0.5)
savefig('05_fit_over_time_and_gamma_stability')
drift = gamma_versions.reset_index().rename(columns={'index': 'instrument'}).groupby('instrument')[['factor1', 'factor2', 'factor3'][:CFG.selected_k]].std().mean().mean()
print('mean across-version sd of loadings:', round(float(drift), 4))
record('walk_forward', gamma_loading_sd_across_versions=float(drift), monthly=rm.round(4).to_dict())

# %% [markdown]
# ## 6. Factor-mimicking weights and the residual-maker
#
# $f_t = (B_t'B_t+\lambda I)^{-1}B_t'r_t = W_t^{F\prime} r_t$ with $W_t^F = B_t(B_t'B_t+\lambda I)^{-1}$. Each column
# of $W^F$ is the portfolio of that day's bonds whose return is the factor. The residual is the projection off the
# beta span; we check $B_t'\varepsilon_t = 0$ and the gross leverage / breadth of each factor portfolio.

# %%
beta_cols = [f'beta{j+1}' for j in range(CFG.selected_k)]
w_rows, orth = [], []
for d, g in resid.groupby('date', sort=True, observed=True):
    B = g[beta_cols].to_numpy(float); eps = g['pit_residual'].to_numpy(float)
    W = B @ np.linalg.inv(B.T @ B + CFG.ridge * np.eye(CFG.selected_k))
    orth.append(np.abs(B.T @ eps).max())
    for j in range(CFG.selected_k):
        w = W[:, j]
        w_rows.append({'date': d, 'factor': f'factor{j+1}', 'n': len(w), 'gross': np.abs(w).sum(), 'net': w.sum(), 'breadth': (np.abs(w).sum() ** 2) / (w @ w)})
wdiag = pd.DataFrame(w_rows)
wsum = wdiag.groupby('factor')[['n', 'gross', 'net', 'breadth']].mean()
display(wsum)
print('max |B\'eps| across dates:', f'{max(orth):.2e}')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
wdiag.pivot(index='date', columns='factor', values='gross').plot(ax=ax[0]); ax[0].set_title('Gross weight (leverage) of factor-mimicking portfolios')
wdiag.pivot(index='date', columns='factor', values='breadth').plot(ax=ax[1]); ax[1].set_title('Effective breadth (bonds)')
savefig('06_mimicking_weights')
record('mimicking_weights', max_beta_orthogonality=float(max(orth)), **{f'{k}_{m}': float(v) for k, row in wsum.iterrows() for m, v in row.items()})

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
acf_rows = []
for lag in range(1, CFG.longconv_lags + 1):
    lagged = grp['pit_residual'].shift(lag)
    ok = lagged.notna()
    x, y = lagged[ok].to_numpy(), resid.loc[ok, 'pit_residual'].to_numpy()
    acf_rows.append({'lag': lag, 'pearson': np.corrcoef(x, y)[0, 1], 'spearman': pd.Series(x).corr(pd.Series(y), method='spearman'), 'pairs': int(ok.sum()), 'bartlett_ci': 1.96 / np.sqrt(max(int(ok.sum()), 1))})
acf = pd.DataFrame(acf_rows).set_index('lag')
display(acf)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(acf.index - 0.2, acf['pearson'], width=0.4, label='Pearson'); ax.bar(acf.index + 0.2, acf['spearman'], width=0.4, label='Spearman')
ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('lag (observations)'); ax.set_title('Pooled residual autocorrelation (within CUSIP)'); ax.legend()
savefig('07_residual_acf')
record('residual_diagnostics', acf_pearson=acf['pearson'].round(4).to_dict(), acf_spearman=acf['spearman'].round(4).to_dict())

# %%
# Activity buckets (pre-OOS volatility quintiles) and lag-1 rank IC by bucket; distribution and QQ of the residual
pre = model[model['date'] < CFG.first_oos_date].groupby('cusip', observed=True)[TARGET].agg(target_abs_vol='std', obs='count').reset_index()
pre['activity_bucket'] = pd.qcut(pre['target_abs_vol'].fillna(0).rank(method='first'), 5, labels=[f'L{i}' for i in range(1, 6)]).astype(str)
resid = resid.merge(pre[['cusip', 'activity_bucket', 'target_abs_vol']], on='cusip', how='left')
resid['activity_bucket'] = resid['activity_bucket'].fillna('UNKNOWN')
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
pairs = resid.dropna(subset=['next_residual']).copy()
pairs['target_month'] = pairs['next_date'].dt.to_period('M')
months = sorted(pairs['target_month'].unique())


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ y)


def forecaster_eval(name: str, cols: list[str], winsor: bool, alpha: float = 1.0) -> tuple[pd.DataFrame, dict, np.ndarray | None]:
    preds, coefs = [], None
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
        coefs = fit_ridge(Xtr, ytr, alpha)
        p = te[['cusip', 'date', 'next_date', 'next_residual', 'activity_bucket']].copy(); p['yhat'] = coefs[0] + Xte @ coefs[1:]; p['month'] = str(m)
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if pred.empty:
        return pred, {'forecaster': name, 'rows': 0}, coefs
    ic = daily_rank_ic(pred, 'yhat', 'next_residual')
    y, yh = pred['next_residual'].to_numpy(), pred['yhat'].to_numpy()
    summ = {'forecaster': name, 'rows': len(pred), 'months': pred['month'].nunique(), **ic_summary(ic), 'oos_r2_vs_zero': 1 - ((y - yh) ** 2).sum() / (y ** 2).sum(), 'sign_acc': float((np.sign(y) == np.sign(yh))[(y != 0) & (yh != 0)].mean())}
    return pred, summ, coefs


fc_results, fc_preds, fc_coefs = [], {}, {}
for name, cols, winsor in [('AR(1) raw', ['lag1'], False), ('AR(1) winsor', ['lag1'], True), (f'LongConv-lite ridge L={L} winsor', lag_cols, True)]:
    pred, summ, coefs = forecaster_eval(name, cols, winsor)
    fc_results.append(summ); fc_preds[name] = pred; fc_coefs[name] = coefs
fc_table = pd.DataFrame(fc_results).set_index('forecaster')
display(fc_table)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
if fc_coefs[f'LongConv-lite ridge L={L} winsor'] is not None:
    ax[0].bar(range(1, L + 1), fc_coefs[f'LongConv-lite ridge L={L} winsor'][1:], color='#4C72B0'); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('lag'); ax[0].set_title('LongConv-lite: learned linear filter over the residual path (last fold)')
best = fc_table['ic_mean'].idxmax()
if not fc_preds[best].empty:
    ic_m = fc_preds[best].groupby('month').apply(lambda g: ic_summary(daily_rank_ic(g, 'yhat', 'next_residual'))['ic_mean'], include_groups=False)
    ic_m.plot.bar(ax=ax[1], color='#55A868'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title(f'Monthly OOS rank IC: {best}'); ax[1].set_xlabel('')
savefig('07_forecasters')
record('residual_forecast', table=fc_table.round(4).to_dict())

# %%
# Two-regime check: does the body persist while the tails revert? Conditional next-residual by current-residual bin.
bins = pairs['pit_residual'].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).to_numpy()
pairs['res_bin'] = pd.cut(pairs['pit_residual'], bins=np.unique(bins), include_lowest=True)
two_regime = pairs.groupby('res_bin', observed=True).agg(n=('next_residual', 'size'), mean_now=('pit_residual', 'mean'), mean_next=('next_residual', 'mean'), median_next=('next_residual', 'median'))
two_regime['continuation_ratio'] = two_regime['mean_next'] / two_regime['mean_now']
display(two_regime)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(two_regime['mean_now'], two_regime['mean_next'], marker='o'); ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('current residual, bin mean (bp)'); ax.set_ylabel('next residual, bin mean (bp)'); ax.set_title('Body persists, tails revert? conditional means by current-residual quantile bin')
savefig('07_two_regime')
record('residual_diagnostics', two_regime=two_regime.reset_index().astype({'res_bin': str}).round(4).to_dict(orient='records'))

# %% [markdown]
# ## 8. Paper portfolio: is the residual path harvestable?
#
# Signal: the best forecaster's prediction of tomorrow's residual. Weights are cross-sectionally demeaned, scaled
# to unit gross, and projected off the beta span with the day's $B_t$, so the portfolio has zero factor exposure.
# P&L is realised on the next day's yield change (bp of yield captured per unit gross). Sharpe is annualised with
# $\sqrt{252}$. This is the number a stat-arb paper would report; a market maker reads it as "how much skew
# information is in the residual" rather than as a tradable strategy.

# %%
best_pred = fc_preds[best].copy()
pp = best_pred.merge(resid[['cusip', 'date'] + beta_cols], on=['cusip', 'date'], how='left')
nxt = resid[['cusip', 'date', 'target_bp', 'pit_residual']].rename(columns={'date': 'next_date', 'target_bp': 'next_target_bp', 'pit_residual': 'next_resid_chk'})
pp = pp.merge(nxt, on=['cusip', 'next_date'], how='left').dropna(subset=['next_target_bp'] + beta_cols)


def portfolio_pnl(frame: pd.DataFrame, hedge: bool) -> pd.Series:
    out = {}
    for d, g in frame.groupby('date', observed=True):
        if len(g) < 50:
            continue
        s = g['yhat'].to_numpy(float); w = s - s.mean()
        if hedge:
            B = g[beta_cols].to_numpy(float); w = w - B @ np.linalg.solve(B.T @ B + 1e-8 * np.eye(B.shape[1]), B.T @ w)
        gross = np.abs(w).sum()
        if gross <= 0:
            continue
        w = w / gross
        out[d] = float(w @ g['next_target_bp'].to_numpy(float))
    return pd.Series(out).sort_index()


def sharpe(x: pd.Series) -> float:
    return float(np.sqrt(252) * x.mean() / x.std()) if x.std() > 0 else np.nan


pnl_h, pnl_u = portfolio_pnl(pp, True), portfolio_pnl(pp, False)
pp_rows = [{'universe': 'ALL', 'hedged_sharpe': sharpe(pnl_h), 'unhedged_sharpe': sharpe(pnl_u), 'mean_bp_per_day': pnl_h.mean(), 'days': len(pnl_h)}]
for b, g in pp.groupby('activity_bucket'):
    s = portfolio_pnl(g, True)
    pp_rows.append({'universe': b, 'hedged_sharpe': sharpe(s), 'unhedged_sharpe': sharpe(portfolio_pnl(g, False)), 'mean_bp_per_day': s.mean(), 'days': len(s)})
paper = pd.DataFrame(pp_rows).set_index('universe')
display(paper)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
pnl_h.cumsum().plot(ax=ax[0], label=f'hedged, Sharpe {sharpe(pnl_h):.2f}'); pnl_u.cumsum().plot(ax=ax[0], label=f'unhedged, Sharpe {sharpe(pnl_u):.2f}'); ax[0].legend(); ax[0].set_title('Paper portfolio: cumulative bp captured per unit gross'); ax[0].axhline(0, color='k', lw=0.6)
paper['hedged_sharpe'].drop('ALL').plot.bar(ax=ax[1], color='#4C72B0'); ax[1].axhline(0, color='k', lw=0.6); ax[1].set_title('Hedged Sharpe by activity bucket'); ax[1].set_xlabel('')
savefig('08_paper_portfolio')
record('paper_portfolio', signal=best, **{f'{i}_{k}': (float(v) if pd.notna(v) else None) for i, row in paper.iterrows() for k, v in row.items()})

# %% [markdown]
# ## 9. Transaction validation (PIT-safe)
#
# Each matched trade is joined to the latest residual dated **strictly before** the trade date. Targets:
# `e_mark = 100 (y_trade − y_prior_mark)` where the prior mark is the close on the residual date, and
# `e_algo = 100 (y_trade − y_algo)`. Inference is Fama-MacBeth over trade dates; pooled OLS is descriptive. We
# also report the slope after neutralising date × side × size, by trade recency and by side.

# %%
def standardize_trades(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr.copy()
    t.columns = [str(c) for c in t.columns]
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
    t = t.dropna(subset=['cusip', 'trade_ts', 'msrb_yield']).sort_values(['cusip', 'trade_ts'], kind='stable').reset_index(drop=True)
    prev = t.groupby('cusip', observed=True)['trade_ts'].shift(1)
    t['recency_days'] = (t['trade_ts'] - prev).dt.total_seconds() / 86400.0
    t['recency_bucket'] = pd.cut(t['recency_days'], bins=[-0.01, 1, 3, 7, 21, np.inf], labels=['<=1d', '1-3d', '3-7d', '7-21d', '>21d']).astype(str).replace('nan', 'first_print')
    return t


def join_trades_to_residual(t: pd.DataFrame, res: pd.DataFrame, min_age: int) -> pd.DataFrame:
    score = res[['cusip', 'date', 'pit_residual', 'activity_bucket'] + beta_cols].copy()
    score['pit_rank'] = score.groupby('date', observed=True)['pit_residual'].rank(pct=True)
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
    return j.reset_index(drop=True)


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
            'fm_beta': slopes.mean() if len(slopes) else np.nan, 'fm_t': np.sqrt(len(slopes)) * slopes.mean() / slopes.std() if len(slopes) > 2 and slopes.std() > 0 else np.nan, 'fm_dates': len(slopes)}


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
    tx = pd.DataFrame(rows)
    display(tx)
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
# ## 10. Level relative value: walk-forward GBM on trade spreads
#
# The pivot experiment. Target: the trade's spread to the MMD curve in bp. Features: the bond's instruments and
# betas on the residual date, plus side, size, recency and timing. Walk-forward by month (train on prior months).
# For each test trade we compare the absolute error of three yield predictions: the algo quote, the prior
# evaluated mark, and the model-implied yield (MMD + predicted spread). The question is whether the level model
# beats the algo where the algo is weakest: names with no recent print.

# %%
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

if HAS_TRADES and not tv.empty and tv['MmdYld'].notna().mean() > 0.5:
    lv = tv.copy()
    lv['spread_bp'] = 100.0 * (lv['msrb_yield'] - lv['MmdYld'])
    feat_panel = model[['cusip', 'date', 'rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1'] + [c for c in CHARS if c != 'market_fv']].rename(columns={'date': 'residual_date'})
    feat_panel['residual_date'] = to_ns(feat_panel['residual_date'])
    lv['residual_date'] = to_ns(lv['residual_date'])
    lv = lv.merge(feat_panel, on=['cusip', 'residual_date'], how='left')
    lv['recency_days_c'] = lv['recency_days'].clip(upper=60).fillna(60)
    for s in ['D', 'P', 'S']:
        lv[f'side_{s}'] = (lv['side'] == s).astype(float)
    FEATURES = ['rating_score', 'years_to_worst', 'extension', 'cpn', 'modified_duration_lag1', 'closing_yield_lag1', 'pit_residual', 'pit_rank', 'recency_days_c', 'log_size', 'MinuteFromSignal', 'dMmdSprdSide', 'side_D', 'side_P', 'side_S'] + beta_cols + [c for c in CHARS if c != 'market_fv']
    FEATURES = [c for c in dict.fromkeys(FEATURES) if c in lv.columns]
    lv = lv.dropna(subset=['spread_bp', 'e_mark_bp']).copy()
    lv['month'] = lv['trade_date'].dt.to_period('M')
    preds = []
    importances = []
    for m in sorted(lv['month'].unique())[1:]:
        tr, te = lv[lv['month'] < m], lv[lv['month'] == m]
        if len(tr) < 2_000 or te.empty:
            continue
        Xtr, Xte = tr[FEATURES].astype(float).fillna(tr[FEATURES].median()), te[FEATURES].astype(float).fillna(tr[FEATURES].median())
        ytr = tr['spread_bp'].clip(*tr['spread_bp'].quantile([0.005, 0.995]))
        if HAS_LGB:
            mdl = lgb.LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=CFG.seed, verbose=-1).fit(Xtr, ytr)
            importances.append(pd.Series(mdl.feature_importances_, index=FEATURES))
        else:
            mdl = HistGradientBoostingRegressor(loss='absolute_error', max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, random_state=CFG.seed).fit(Xtr, ytr)
        p = te[['cusip', 'trade_date', 'recency_bucket', 'side', 'msrb_yield', 'MmdYld', 'e_algo_bp', 'e_mark_bp']].copy()
        p['spread_hat'] = mdl.predict(Xte); p['e_model_bp'] = 100.0 * (p['msrb_yield'] - (p['MmdYld'] + p['spread_hat'] / 100.0)); p['month'] = str(m)
        preds.append(p)
    lvp = pd.concat(preds, ignore_index=True)
    print(f'level model ({"LightGBM" if HAS_LGB else "sklearn HistGBM"}): {len(lvp):,} held-out trades over {lvp["month"].nunique()} months, {len(FEATURES)} features')

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
    lvp.to_parquet(ARTIFACTS / 'level_model_predictions_v2.parquet', index=False)
    record('level_model', backend='lightgbm' if HAS_LGB else 'sklearn_histgbm', features=FEATURES, overall=overall.round(4).to_dict(orient='records'), by_recency=by_rec.round(4).reset_index().to_dict(orient='records'), by_side=by_side.round(4).reset_index().to_dict(orient='records'))
else:
    print('Level model skipped: needs matched trades with MmdYld.')

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
    ('Paper portfolio hedged Sharpe', f"{paper.loc['ALL', 'hedged_sharpe']:.2f} (ALL); best bucket {paper['hedged_sharpe'].drop('ALL').idxmax()} {paper['hedged_sharpe'].drop('ALL').max():.2f}"),
]
if HAS_TRADES and not tv.empty:
    tm = tx[(tx['target'] == 'e vs prior mark') & (tx['bucket'] == 'ALL')].iloc[0]; ta = tx[(tx['target'] == 'e vs algo quote')].iloc[0]; tn = tx[tx['target'].str.startswith('e vs prior mark, neutralised')].iloc[0]
    summary_rows += [('Trade vs prior mark on residual rank', f"FM {tm['fm_beta']:+.2f} bp/rank (t {tm['fm_t']:+.2f}); neutralised {tn['fm_beta']:+.2f} (t {tn['fm_t']:+.2f})"),
                     ('Trade vs algo quote on residual rank', f"FM {ta['fm_beta']:+.2f} bp/rank (t {ta['fm_t']:+.2f})")]
if 'level_model' in REGISTRY:
    o = REGISTRY['level_model']['overall'][0]
    summary_rows.append(('Level model vs algo quote (all trades)', f"MAE algo {o['MAE algo quote']:.2f} | mark {o['MAE prior mark']:.2f} | model {o['MAE level model']:.2f} bp; gain {o['gain vs algo (bp)']:+.2f} (t {o['gain t (FM)']:+.2f})"))
summary = pd.DataFrame(summary_rows, columns=['item', 'result']).set_index('item')
display(summary)
print('Artifacts written to', ARTIFACTS)
for p in sorted(ARTIFACTS.glob('*')):
    if p.is_file():
        print('  ', p.name, f'{p.stat().st_size/1e6:.2f} MB')
print('Figures:', len(list(FIGURES.glob('*.png'))))
