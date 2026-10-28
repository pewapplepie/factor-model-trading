# muni_rfq_factor_variants.py: variants v1.1. Every factor-derived quoting variant through the production optimizer, replayed on the desk's requests and read
# against production on the same four facts (win rate, cover, crossing, P&L under the sceptical exits), in four pairs. Reads the handoff written by
# Section 15a of muni_ipca_factor_lab.py; needs no model code and no pipeline. Percent format; regenerate the .ipynb with the converter in docs/rfq_variants_notes.md.

# %% [markdown]
# # RFQ factor variants: the production optimizer as the judge
#
# **Why a separate notebook.** The lab builds the factor model and tests its pieces on their own terms (fit, calibration, information coefficients).
# None of that decides whether a piece helps the desk quote. The only test that does is the production optimizer replayed on every request with the
# piece in it, paired against production on the same requests, with the win decided by the logged cover or winning print. This notebook does only that.
# Every variant is a change to one term of production's objective, `argmax over x of (x + c) p(x)`, quoted by production's own argmax on
# production's own fill curves, and nothing is gated by the model-level verdicts: the replay is the source of truth on relative performance.
#
# **Inputs.** The handoff of lab Section 15a: the request frame with the logged quote, charges, outcome, thresholds and exits; the production fill
# curves; the betas, exposures, idiosyncratic variances and the picked covariance on the requests (14d); every exit estimator's markout (14f);
# the model signals at the request (14 [5]); the realised holding time by bucket (17).
#
# **The four pairs.** Each variant is quoted twice, with the manual charge in `c` and with it removed, and so is production:
#
# | Pair | Variant quoted with | Production quoted with | Question |
# |---|---|---|---|
# | A | manual kept | manual kept | does the factor term add to what the traders charge |
# | B | manual removed | manual kept | can the factor term replace the manual charge |
# | C | manual removed | manual removed | what the factor term is worth on its own |
# | D | production, manual removed | production, manual kept | the reference: what the manual charge itself is worth |
#
# Each pair is read raw (the frontier point) and levelled (a constant width per month so the variant's win rate equals its comparator's, so cover
# and wins kept are compared at equal wins). Every cell carries the same columns, paired on identical requests.
#
# **What v1.1 changed, after the first real run.** The objective is the desk's: hold the win rate near production's, lower the cover, raise the P&L
# under the sceptical exits. So the comparator is production itself widened through its own optimizer to the variant's win rate, month by month,
# and there is no threshold anywhere: every cell is a change against that comparator with its t, and a sign-only reading of the axes it is better
# on. The families follow the v1 evidence: the fill curve on idiosyncratic volatility (the one sigma with information), now also as a state-conditioned
# curve; the own-CUSIP concentration term, now on the real WFTrade book; and a new value family, the expected markout as a function of the
# concession x, from the exit of record and from the desk's realised sales. The signal charges are off.
#
# | Cell | Content |
# |---|---|
# | [1] | setup: config, cache keyed on the fields each stage reads, registry, pre-registered bars |
# | [2] | the handoff, the arrays, the replay check (production reproduced bit for bit) |
# | [3] | machinery: the optimizer (verbatim), requote by charge and by curve, the win rule, levelling, the sequential book, the paired scorecard |
# | [4] | references: production with and without the manual charge, the 0.5 bp width frontiers of both regimes, pair D |
# | [5] | the fill curve as a conditional distribution: idiosyncratic vol, parameters picked walk-forward, the state-conditioned refit; validated by log-likelihood |
# | [6] | the concentration term on the real book (and the replayed-book and hedged-book references) |
# | [7] | the value of a win as a function of x: b(x) from the record exit and from realised sales; with the curve |
# | [7b] | the model signals as charges, off by default |
# | [8] | the matrix: pairs A, B, C against production, against the frontier at equal win rate, and levelled; readings; Pareto sets |
# | [9] | registry and runtimes |

# %%
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)
try:
    from IPython.display import display  # noqa: F401
    get_ipython()  # type: ignore[name-defined]  # noqa: F821
    _INTERACTIVE = True
except Exception:  # noqa: BLE001
    _INTERACTIVE = False

    def display(obj):  # noqa: D103
        with pd.option_context('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 120):
            print(obj)
import matplotlib
if not _INTERACTIVE:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)


@dataclass(frozen=True)
class VariantConfig:
    spec_version: str = 'rfq_variants_v1_1'
    seed: int = 7
    sigmas: tuple[str, ...] = ('idiosyncratic',)              # v1: systematic and total worsened the fill-curve likelihood; the conditional scale covers 18 percent
    stretch_exponents: tuple[float, ...] = (0.25, 0.5, 1.0)
    logit_kappas: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
    n0_grid: tuple[float, ...] = (0.0, 1e1, 1e2, 1e3, 1e4, 1e5, float('inf'))
    k_grid: tuple[int, ...] = (1, 2, 3, 5)                    # concentration term: median extra charge in bp
    book_k_grid: tuple[int, ...] = (1, 2)                     # the hedged book term, kept as the 14e reference
    signal_k: tuple[float, ...] = (1.0, 2.0, 3.0)
    signals_on: bool = False
    hedge_cost_bp: float = 1.0
    frontier_widths: tuple[float, ...] = tuple(w / 2.0 for w in range(-20, 21))   # -10 .. +10 bp by 0.5
    x_bins: int = 5
    dv01_trim_q: float = 0.99
    level_range_bp: float = 40.0
    level_iters: int = 16
    chunk: int = 20_000
    quant_step: float = 0.01
    test_months: tuple[str, ...] = ()

CFG = VariantConfig()
ROOT = Path(os.environ.get('MUNI_LAB_ROOT', '/home/k178619/muni/Algos/factor_model'))
ARTIFACTS = ROOT / 'artifacts_ipca_lab'; HANDOFF = ARTIFACTS / 'handoff_rfq_variants'
OUT = ARTIFACTS / 'variants'; CACHE_DIR = OUT / 'cache'; FIGURES = OUT / 'figures'
for _p in (CACHE_DIR, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(CFG.seed)
REGISTRY: dict = {'spec_version': CFG.spec_version, 'config': asdict(CFG), 'run_at': pd.Timestamp.now('UTC').isoformat()}
STAGE_KEYS: dict[str, str] = {}; CACHE_LOG: list[dict] = []; CELL_LOG: list[dict] = []
_RUN_T0 = time.perf_counter(); _CELL_STATE = {'t0': _RUN_T0, 'label': '1. Setup [1]'}

PRE_REGISTERED: dict[str, str] = {
    'variants_v11_reading': ('written 2026-10-28 after the v1 run of 2026-10-27 with its results in view, so on this data a confirmation under revised rules and a pre-registered test '
                             'only on later data. The objective: hold the win rate near production\'s, lower the cover, raise the P&L under the sceptical exits. No threshold anywhere. '
                             'Comparators, per pair: production as it is; production widened through its own optimizer to the variant\'s win rate month by month (the nearest width on a '
                             '0.5 bp grid from -10 to +10, the residual mismatch reported), the primary reading; the variant levelled to production (v1). Pairs as v1: A manual kept vs '
                             'production; B manual removed vs production; C both removed; D production without the manual charge, the reference. Every cell: win rate and its change, wins '
                             'kept, new wins, wins given up, bp left on the table on cover wins and its change, bp saved on kept cover wins, crossing on new wins, the delta per request '
                             'under the exit of record, the guard, the common markout (new wins), the hedged record exit and the lab\'s other side with weekly Newey-West t and months '
                             'positive, the smallest of the four sceptical deltas, dollars per month count-weighted, DV01-weighted and DV01-trimmed at the 99th percentile with the top-1% '
                             'share, the delta on the requests at the month\'s modal manual charge and on the request-specific ones, the P&L sample share, the book\'s ex-ante sd against '
                             'the comparator\'s and the hedged dollars per dollar of sd; a sign-only reading (better on cover / P&L under all four valuations / risk) and the Pareto set per '
                             'pair on (cover change, smallest sceptical delta, sd change). The v1 gates are retired, not replaced'),
    'variants_v11_families': ('fixed 2026-10-28: (5) the fill curve on idiosyncratic volatility: stretch a in {0.25, 0.5, 1} and logit kappa in {0.25, 0.5, 1, 1.5}, each also with the '
                              'parameter picked walk-forward by the cumulative held-out log-likelihood gain on the earlier months; the own-curve refit by vol tercile and the state-conditioned '
                              'refit by vol tercile x manual class (the month\'s modal charge or request-specific) x the lagged common-move tercile of the day, n0 by the previous month\'s '
                              'log-likelihood; validated at the logged concession, reported by manual class, not gating. (6) the concentration term gamma sigma_h^2 (Q_i + u_i / 2) with Q_i '
                              'the WFTrade start-of-day DV01 in the CUSIP plus the policy\'s incremental fills over production\'s in the last h days; the replayed-book version and the hedged '
                              'book term as references; gamma_k = k / median s on production\'s book in the first test month. (7) b(x) = E[markout | won at x] per key group x x-quintile, '
                              'walk-forward with empirical-Bayes shrinkage to the group mean, from the exit of record on production\'s wins and from the realised WFTrade round trips on the '
                              'matched fills; quoted as argmax (x + c + b(x)) p(x); also combined with the walk-forward logit curve. (7b) signals off'),
}


def CELL_T(label: str) -> None:
    now = time.perf_counter(); CELL_LOG.append({'cell': _CELL_STATE['label'], 'seconds': now - _CELL_STATE['t0']}); _CELL_STATE.update(t0=now, label=label)
    print(f'\n=== {label} | elapsed {(now - _RUN_T0) / 60:.1f} min ===')


def record(section: str, **values) -> None:
    REGISTRY.setdefault(section, {}).update({k: (v.item() if hasattr(v, 'item') else v) for k, v in values.items()})
    json.dumps(REGISTRY[section], default=str)


def savefig(name: str) -> None:
    plt.tight_layout(); plt.savefig(FIGURES / f'{name}.png', dpi=110)
    if _INTERACTIVE:
        plt.show()
    plt.close('all')


def flat_records(frame: pd.DataFrame, digits: int = 4) -> list[dict]:
    f = frame.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = [' | '.join(str(x) for x in c if str(x) != '') for c in f.columns]
    return f.round(digits).reset_index().to_dict(orient='records')


def nw_t(x, lags: int | None = None) -> float:
    """Newey-West (Bartlett kernel) t statistic of the mean; lags default floor(4 (n / 100)^(2/9)). Verbatim from the lab."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 4:
        return np.nan
    L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else int(lags)
    u = x - x.mean(); v = float(u @ u) / n
    for l in range(1, min(L, n - 1) + 1):
        v += 2.0 * (1.0 - l / (L + 1.0)) * float(u[l:] @ u[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return {'shape': list(x.shape), 'sum': float(np.nansum(x.astype(float))) if x.dtype.kind in 'fiub' else str(x[:5])}
    if hasattr(x, 'item'):
        return x.item()
    return x


def _stage_source(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return repr(obj)


def _fingerprint(payload) -> str:
    return hashlib.sha1(json.dumps(_jsonable(payload), sort_keys=True, default=str).encode()).hexdigest()[:16]


def frame_fingerprint(frame: pd.DataFrame, cols: list[str] | None = None) -> str:
    f = frame if cols is None else frame[cols]
    h = pd.util.hash_pandas_object(f, index=True).to_numpy(np.uint64)
    return _fingerprint({'shape': f.shape, 'columns': list(f.columns), 'xor': int(np.bitwise_xor.reduce(h)) if len(h) else 0, 'sum': int(h.sum())})


def _stage_files(name: str) -> list[Path]:
    import re
    pat = re.compile(rf'^{re.escape(name)}_[0-9a-f]{{16}}\.(pkl|parquet)$')
    return [p for p in CACHE_DIR.glob(f'{name}_*') if pat.match(p.name)]


def cached(name: str, fn, deps=(), code=(), cfg: tuple[str, ...] = (), force: bool = False):
    """Run or load a stage. The key binds the spec, ONLY the config fields the stage reads (cfg), its data dependencies and its source, so a change
    to a field another stage reads does not rebuild this one. DataFrames go to parquet, else pickle; temporary file then rename; unreadable cache rebuilt."""
    sub = {k: getattr(CFG, k) for k in cfg}
    key = _fingerprint({'spec': CFG.spec_version, 'cfg': sub, 'deps': deps, 'code': [_stage_source(fn)] + [_stage_source(x) for x in code]})
    STAGE_KEYS[name] = key
    path_pq, path_pk = CACHE_DIR / f'{name}_{key}.parquet', CACHE_DIR / f'{name}_{key}.pkl'
    path = path_pq if path_pq.exists() else path_pk; t0 = time.perf_counter(); status = 'computed'; out = None
    if path.exists() and not force:
        try:
            out = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_pickle(path); status = 'reused'
        except (EOFError, pickle.UnpicklingError, OSError, ValueError, AttributeError, ModuleNotFoundError) as exc:
            print(f'{name}: cache file {path.name} unreadable ({type(exc).__name__}); rebuilding'); path.unlink(missing_ok=True)
    if status != 'reused':
        out = fn()
        if isinstance(out, pd.DataFrame):
            tmp = path_pq.with_name(f'{path_pq.name}.tmp{os.getpid()}')
            try:
                out.to_parquet(tmp); os.replace(tmp, path_pq); path = path_pq
            except Exception as exc:  # noqa: BLE001
                tmp.unlink(missing_ok=True); print(f'  parquet failed ({type(exc).__name__}); stored as pickle'); tmp = path_pk.with_name(f'{path_pk.name}.tmp{os.getpid()}'); pd.to_pickle(out, tmp); os.replace(tmp, path_pk); path = path_pk
        else:
            tmp = path_pk.with_name(f'{path_pk.name}.tmp{os.getpid()}'); pd.to_pickle(out, tmp); os.replace(tmp, path_pk); path = path_pk
        for old in _stage_files(name):
            if old != path:
                old.unlink(missing_ok=True)
    dt = time.perf_counter() - t0
    CACHE_LOG.append({'stage': name, 'key': key, 'status': status, 'seconds': dt, 'file': path.name, 'cell': _CELL_STATE['label']})
    print(f'{name}: {"cache" if status == "reused" else status} {dt:.1f}s [{key}] ({path.stat().st_size / 1e6:,.0f} MB)')
    return out


for _tmp in CACHE_DIR.glob('*.tmp*'):
    _tmp.unlink(missing_ok=True)
print(f'variants {CFG.spec_version} | root {ROOT} | handoff {HANDOFF} | sigmas {CFG.sigmas} | k grid {CFG.k_grid} | hedge cost {CFG.hedge_cost_bp} bp | frontier widths {CFG.frontier_widths[0]:+.1f}..{CFG.frontier_widths[-1]:+.1f} by 0.5')

# %% [markdown]
# ## 2. The handoff and the replay check
#
# The request frame, the curves and the arrays as the lab left them. Production is re-quoted here by the optimizer copied verbatim from the lab and
# must reproduce the logged optimal quantile and every logged win and loss; nothing is scored if it does not.

# %%
CELL_T('2. The handoff and the replay check [2]')
_h_t0 = time.perf_counter()
assert HANDOFF.exists(), f'{HANDOFF} not found: run lab Section 15a first'
BT = pd.read_parquet(HANDOFF / 'requests.parquet'); _A = dict(np.load(HANDOFF / 'arrays.npz')); _O = pd.read_pickle(HANDOFF / 'objects.pkl')
META, BT_CURVES, SIGMA = _O['meta'], _O['curves'], {int(k): v for k, v in _O['sigma'].items()}
HOLD_TAB = pd.read_parquet(HANDOFF / 'holding.parquet') if (HANDOFF / 'holding.parquet').exists() else None
N = len(BT); BT['date'] = pd.to_datetime(BT['date']).dt.tz_localize(None) if getattr(BT['date'].dt, 'tz', None) is not None else pd.to_datetime(BT['date'])
BT_EPS_BP, BT_H_RECORD, BT_HOLD = float(META['eps_bp']), int(META['h_record']), int(META['hold'])
BT_MONTHS = sorted(BT['month'].unique()); TEST_M = list(CFG.test_months) or list(META.get('test_months') or [])
if not TEST_M:
    TEST_M = BT_MONTHS[1:]
_month = BT['month'].to_numpy(); _week = BT['week'].to_numpy(); _winable = BT['winnable'].to_numpy(bool)
_thr = BT['threshold_y'].to_numpy(float); _grp = BT['group'].astype(str).to_numpy()
_m_sig, _adj, _ch = BT['signalyield'].to_numpy(float), BT['adj_yield'].to_numpy(float), BT['charges_level'].to_numpy(float)
_man, _liq, _rsk = (BT[c].to_numpy(float) for c in ('manualcharges', 'liquiditycharge', 'riskcharge'))
_c_base = -(_liq + _rsk)                                   # production's charge with the manual charge removed
_y_tr = BT['y_trade'].to_numpy(float); _usd = BT['usd_per_bp'].to_numpy(float)
_keys = BT['pfill_key'].astype(str).to_numpy(); _files = BT['pfill_file'].astype(str).to_numpy()
_xlog = BT['optimalQuantileBp'].to_numpy(float); _wlog = BT['won_logged'].to_numpy(bool); _y_van = BT['y_vanilla'].to_numpy(float)
_pb_days = pd.DatetimeIndex(pd.to_datetime(META['pb_days'])); _pb_di = _pb_days.get_indexer(BT['date'].dt.normalize()); _pb_nd = len(_pb_days)
assert (_pb_di >= 0).all(), 'request dates outside the handoff day index'
_pb_rows_day = pd.Series(np.arange(N)).groupby(_pb_di).indices
_pb_u = _A['pb_u'] if 'pb_u' in _A else np.where(_winable, np.nan_to_num(_usd), 0.0)
_pb_month = np.array(_pb_days.strftime('%Y-%m'))
_tm = np.isin(_month, TEST_M); _tmw = _tm & _winable; _tdays = np.isin(_pb_month, TEST_M)
_nT, _D = int(META['nT']), int(META['D'])
EXPO, V_IDIO = _A['BT_EXPO_GZ'], _A['BT_V_C']; EXPO_H = EXPO.copy(); EXPO_H[:, :_nT] = 0.0
VOL = {k[4:]: v for k, v in _A.items() if k.startswith('VOL_')}; MEXIT = {k[6:]: v for k, v in _A.items() if k.startswith('MEXIT_')}
EX_RECORD, EX_GUARD = META['exit_record'], META['exit_guard']
assert EX_RECORD in MEXIT and EX_GUARD in MEXIT, f'exits {EX_RECORD} / {EX_GUARD} missing from the handoff'
_M_REC, _M_GUARD, _M_OTHER = MEXIT[EX_RECORD], MEXIT[EX_GUARD], BT[f'M_{BT_H_RECORD}_X'].to_numpy(float)
_mmd5 = _A['mmd5'] if 'mmd5' in _A else np.full(N, np.nan)
_cm_rec = pd.Series(np.where(_winable, _M_REC, np.nan)).groupby(BT['date'].to_numpy()).transform('median').to_numpy(float)   # the day's common markout of the record exit
SIG_NAMES = dict(META['signals']); SIG = {k: BT[f'a_{k}'].to_numpy(float) for k in SIG_NAMES if f'a_{k}' in BT}
_a_risk = BT['a_risk'].to_numpy(float) if 'a_risk' in BT else np.full(N, np.nan)
_sig_days = sorted(SIGMA); _ci = pd.factorize(BT['cusip'].astype(str))[0]; _n_cusip = int(_ci.max()) + 1
_wk_code, _wk_levels = pd.factorize(pd.Series(_week), sort=True); _mo_code, _mo_levels = pd.factorize(pd.Series(_month), sort=True)


def _group_mean(d: np.ndarray, codes: np.ndarray, n_levels: int) -> np.ndarray:
    """Mean of the finite values of d per code (NaN where a code has none); the weekly and monthly aggregation without pandas."""
    ok = np.isfinite(d); num = np.bincount(codes[ok], weights=d[ok], minlength=n_levels); den = np.bincount(codes[ok], minlength=n_levels)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


# ---- the optimizer, verbatim from lab Section 14 [1]; the win rule from 14 [3]
def bt_optimize(qbp: np.ndarray, p: np.ndarray, charges_level: np.ndarray, width: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production argmax of (quantile_bp + width + charges) x p for each request, solved once per distinct charge level."""
    c = np.asarray(charges_level, dtype=float); ok = np.isfinite(c)
    ob, op, ou = np.full(len(c), np.nan), np.full(len(c), np.nan), np.full(len(c), np.nan)
    if not ok.any():
        return ob, op, ou
    uc, inv = np.unique(c[ok], return_inverse=True)
    base = qbp + width; step = max(1, 2_000_000 // max(len(qbp), 1)); best = np.empty(len(uc), dtype=np.int64)
    for s in range(0, len(uc), step):
        best[s:s + step] = np.argmax((base[None, :] + uc[s:s + step, None]) * p[None, :], axis=1)
    bi = best[inv]
    ob[ok], op[ok], ou[ok] = qbp[bi], p[bi], (base[bi] + c[ok]) * p[bi]
    return ob, op, ou


def bt_opto(m, opt_bp) -> np.ndarray:
    return np.asarray(m, float) + np.asarray(opt_bp, float) / 100.0


def bt_final(opto, rows: np.ndarray | None = None) -> np.ndarray:
    """The street quote: the optimal yield plus the request's logged loading, as production applied it."""
    return np.asarray(opto, float) + (_adj if rows is None else _adj[rows])


def bt_win_fast(rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The outcome rule of lab 14 [3] on a subset of rows: a quote richer than the winning print wins, a quote cheaper than our cover loses."""
    d = 100.0 * (_thr[rows] - y)
    return np.where(_grp[rows] == 'lost, printed', d > BT_EPS_BP, d >= -BT_EPS_BP) & np.isfinite(d)


def requote(rows: np.ndarray, charges: np.ndarray, width: float = 0.0) -> np.ndarray:
    """Production optimizer on the rows' own curves with the given charge levels; optimal yields (NaN where the key is not in the file)."""
    out = np.full(len(rows), np.nan); ch = np.asarray(charges, float)
    for (f, k), ii in pd.DataFrame({'f': _files[rows], 'k': _keys[rows]}).groupby(['f', 'k'], sort=False).indices.items():
        cur = BT_CURVES.get(f, {}).get(k)
        if cur is not None:
            out[ii] = bt_optimize(cur[0], cur[1], ch[ii], width)[0]
    return bt_opto(_m_sig[rows], out)


_all = np.arange(N); _opt_rep = requote(_all, _ch)
_rep_bp = 100.0 * (_opt_rep - _m_sig) - _xlog
REPLAY_BP = float(np.nanmax(np.abs(_rep_bp))); REPLAY_SHARE = float(np.mean(np.abs(np.nan_to_num(_rep_bp, nan=1e9)) <= 0.25))
_y_rep = bt_final(_opt_rep); _w_rep = bt_win_fast(_all, _y_rep)
REPLAY_PASS = bool(REPLAY_SHARE >= 0.95 and (_w_rep == _wlog).all() and np.allclose(_y_rep, _y_van, atol=1e-9, equal_nan=True))
_y_van = _y_rep
print(f'handoff {META["spec"]} written {META["run_at"][:19]}: {N:,} requests {BT_MONTHS[0]}..{BT_MONTHS[-1]}, winnable {_winable.mean():.1%}, logged win rate {_wlog.mean():.2%}, '
      f'manual requests {BT["manual"].astype(bool).mean():.1%} | test months {TEST_M[0]}..{TEST_M[-1]} ({len(TEST_M)}): {_tm.sum():,} requests, {_tmw.sum():,} winnable')
print(f'exits: record = {EX_RECORD}, guard = {EX_GUARD}; markout coverage of winnable requests record {np.isfinite(_M_REC[_winable]).mean():.1%}, guard {np.isfinite(_M_GUARD[_winable]).mean():.1%}, '
      f'other side {np.isfinite(_M_OTHER[_winable]).mean():.1%}, hedge term {np.isfinite(_mmd5[_winable]).mean():.1%} | covariance on {len(_sig_days)} of {_pb_nd} days ({META.get("sigma_label")}, betas {META.get("gz_label")})')
print('signal coverage of winnable requests:', {SIG_NAMES.get(k, k): round(float(np.isfinite(v[_winable]).mean()), 3) for k, v in SIG.items()}, '| conditional scale', round(float(np.isfinite(_a_risk[_winable]).mean()), 3))
print(f'REPLAY CHECK: optimal quantile reproduced within 0.25 bp on {REPLAY_SHARE:.2%} (max |d| {REPLAY_BP:.3g} bp); every logged win and loss reproduced: {(_w_rep == _wlog).all()}; '
      f'lab vanilla pass {META.get("vanilla_pass")} -> {"PASS" if REPLAY_PASS else "FAIL"}')
assert REPLAY_PASS, 'production not reproduced: nothing below is comparable'
_hold_req = np.full(N, float(BT_HOLD))
if HOLD_TAB is not None and 'tenor_bucket' in BT and 'call_bucket' in BT:
    _hk = HOLD_TAB[HOLD_TAB['tenor_bucket'] != 'ALL'].set_index(['tenor_bucket', 'call_bucket'])['hold_bdays']
    _hr = pd.MultiIndex.from_arrays([BT['tenor_bucket'].astype(str), BT['call_bucket'].astype(str)]).map(_hk.to_dict()).to_numpy(float)
    _hall = float(HOLD_TAB.loc[HOLD_TAB['tenor_bucket'] == 'ALL', 'hold_bdays'].iloc[0]) if (HOLD_TAB['tenor_bucket'] == 'ALL').any() else float(BT_HOLD)
    _hold_req = np.where(np.isfinite(_hr), _hr, _hall); _hold_req = np.clip(_hold_req, 1.0, 60.0)
    print(f'realised holding by tenor x call bucket (17): median over requests {np.median(_hold_req[_winable]):.1f} business days (lab horizon {BT_HOLD}); the charge family scales variance by h_i / {BT_HOLD}')
else:
    print(f'no realised holding table in the handoff: the charge family uses the lab horizon h = {BT_HOLD}')
# ---- v1.1 inputs: the real position by CUSIP and day, the realised round trips per request, the manual class, the market state, the key group
POS = pd.read_parquet(HANDOFF / 'positions.parquet') if (HANDOFF / 'positions.parquet').exists() else None
POS_SOD = np.zeros((_n_cusip, _pb_nd))                      # start-of-day position in $ DV01 per bp, by CUSIP code and day index
_dv01_100 = np.full(_n_cusip, np.nan); _dq = pd.Series(np.where(BT['quantity'].to_numpy(float) > 0, _usd / BT['quantity'].to_numpy(float) * 100.0, np.nan)).groupby(_ci).median(); _dv01_100[_dq.index.to_numpy()] = _dq.to_numpy()
if POS is not None and len(POS):
    _pc = pd.Index(pd.factorize(BT['cusip'].astype(str))[1]).get_indexer(POS['cusip'].astype(str)); _pd_ = _pb_days.searchsorted(pd.to_datetime(POS['date']).dt.normalize().to_numpy(), side='right') - 1
    ok_ = (_pc >= 0) & (_pd_ >= 0); eod = np.full((_n_cusip, _pb_nd), np.nan); eod[_pc[ok_], _pd_[ok_]] = POS['pos_eod_par'].to_numpy(float)[ok_]
    eod = pd.DataFrame(eod).ffill(axis=1).fillna(0.0).to_numpy(); POS_SOD[:, 1:] = eod[:, :-1] / 100.0 * np.nan_to_num(_dv01_100)[:, None]
    print(f'real book: {len(POS):,} CUSIP-days of end-of-day positions; a position is held at the request on {np.mean(POS_SOD[_ci[_tmw], _pb_di[_tmw]] > 0):.1%} of test winnable requests')
else:
    print('no positions in the handoff: the concentration term falls back to the replayed book')
REAL = pd.read_parquet(HANDOFF / 'realised.parquet') if (HANDOFF / 'realised.parquet').exists() else None
_real_bp = np.full(N, np.nan)
if REAL is not None and len(REAL):
    _rs = REAL[(REAL['sold_par'] > 0) & np.isfinite(REAL['roundtrip_bp'])]; _real_bp[_rs['bt_row'].to_numpy(int)] = _rs['roundtrip_bp'].to_numpy(float)
    print(f'realised round trips on {np.isfinite(_real_bp).sum():,} request rows (mean {np.nanmean(_real_bp):+.1f} bp)')
_man_r = np.round(_man * 2.0) / 2.0; _man_mode = {m: float(pd.Series(_man_r[(_month == m) & _winable]).mode().iloc[0]) for m in BT_MONTHS}
_man_modal = np.abs(_man_r - pd.Series(_month).map(_man_mode).to_numpy(float)) < 0.25
_cm_by_day = pd.Series(np.where(_winable, _M_REC, np.nan)).groupby(_pb_di).median().reindex(range(_pb_nd)).to_numpy(float)
_cm_lag = np.full(_pb_nd, np.nan); _cm_lag[BT_HOLD:] = _cm_by_day[:-BT_HOLD]       # the common markout of requests BT_HOLD days earlier: realised by the day
_cm_edges = np.nanquantile(_cm_lag[(_pb_month < TEST_M[0]) & np.isfinite(_cm_lag)], [1 / 3, 2 / 3]) if np.isfinite(_cm_lag[_pb_month < TEST_M[0]]).sum() > 5 else np.nanquantile(_cm_lag[np.isfinite(_cm_lag)], [1 / 3, 2 / 3])
_mkt_state = np.where(np.isfinite(_cm_lag[_pb_di]), np.searchsorted(_cm_edges, np.nan_to_num(_cm_lag[_pb_di])), 1).astype(int)
_kgrp = np.array([(k.split('_')[0] if k.startswith('cpn3') else '_'.join(k.split('_')[:2])) for k in _keys], dtype=object)
print(f'manual class: {_man_modal[_tmw].mean():.1%} of test winnable requests carry the month\'s modal manual charge | market state terciles of the lagged common move at {np.round(_cm_edges, 2)} bp | key groups {len(set(_kgrp[_tmw]))}')
record('variants_handoff', spec=META['spec'], requests=int(N), test_months=TEST_M, replay_pass=REPLAY_PASS, replay_max_bp=REPLAY_BP, exit_record=EX_RECORD, exit_guard=EX_GUARD,
       markout_coverage_record=float(np.isfinite(_M_REC[_winable]).mean()), holding_median=float(np.median(_hold_req[_winable])), seconds=time.perf_counter() - _h_t0)

# %% [markdown]
# ## 3. Machinery
#
# Quoting by charge (`requote`), by curve (`requote_curve`: a per-request fill curve on the key's grid, the parameters quantised so the argmax is solved
# once per distinct (parameter, charge) pair as production solves it once per charge) and by value (`requote_value`: a value term `b(x)` on the key's
# grid). The sequential book for the inventory terms, with the real start-of-day position. The frontier comparator: production widened through its
# own optimizer to the variant's win rate, month by month. One paired scorecard, with no threshold anywhere: every comparison is reported as the
# change in win rate, cover, crossing, P&L under five valuations, dollars count- and DV01-weighted, and book risk, with the Newey-West t beside it.

# %%
CELL_T('3. Machinery [3]')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _argmax_dedup(q: np.ndarray, P_of: dict, par_code: np.ndarray, c: np.ndarray, vadd: np.ndarray | None = None) -> np.ndarray:
    """argmax over the grid of (q + c + vadd) x P[par_code] for each row, solved once per distinct (par_code, c) pair."""
    pair = np.unique(np.column_stack([par_code, c]), axis=0, return_inverse=True); uniq, inv = pair[0], pair[1].ravel()
    best = np.empty(len(uniq), dtype=np.int64); base = q if vadd is None else q + vadd
    for pc in np.unique(uniq[:, 0]):
        sel = np.flatnonzero(uniq[:, 0] == pc); P = P_of[pc]
        U = (base[None, :] + uniq[sel, 1][:, None]) * P[None, :]; best[sel] = np.argmax(U, axis=1)
    return q[best[inv]]


def requote_curve(rows: np.ndarray, charges: np.ndarray, ratio: np.ndarray | None = None, shift: np.ndarray | None = None, curves: dict | None = None,
                  curve_of: np.ndarray | None = None, vadd: dict | None = None) -> np.ndarray:
    """Production's argmax with a per-request fill curve: p(q x ratio_i) (stretch), sigmoid(logit p + shift_i) (logit tilt), curves[(f, k, code_i)] (a refit),
    and optionally a value term vadd[(f, k)] on the grid. ratio, shift and curves all None reproduces requote. The stretch ratio and the shift are quantised
    to CFG.quant_step so each distinct curve is built once per key."""
    out = np.full(len(rows), np.nan); ch = np.asarray(charges, float)
    for (f, k), ii in pd.DataFrame({'f': _files[rows], 'k': _keys[rows]}).groupby(['f', 'k'], sort=False).indices.items():
        cur = BT_CURVES.get(f, {}).get(k)
        if cur is None:
            continue
        q, p = cur; ok = np.isfinite(ch[ii]); jj = ii[ok]
        if not len(jj):
            continue
        if ratio is not None:
            par = np.round(ratio[jj] / CFG.quant_step) * CFG.quant_step; codes, inv = np.unique(par, return_inverse=True)
            P_of = {ci: np.interp(q * codes[ci], q, p) for ci in range(len(codes))}
        elif shift is not None:
            par = np.round(shift[jj] / CFG.quant_step) * CFG.quant_step; codes, inv = np.unique(par, return_inverse=True); lp = _logit(p)
            P_of = {ci: _sigmoid(lp + codes[ci]) for ci in range(len(codes))}
        elif curves is not None:
            codes, inv = np.unique(curve_of[jj], return_inverse=True); P_of = {ci: curves.get((f, k, int(codes[ci])), p) for ci in range(len(codes))}
        else:
            inv = np.zeros(len(jj), dtype=np.int64); P_of = {0: p}
        out[jj] = _argmax_dedup(q, P_of, inv.ravel(), ch[jj], None if vadd is None else vadd.get((f, k)))
    return bt_opto(_m_sig[rows], out)


def requote_value(rows: np.ndarray, charges: np.ndarray, vadd: dict) -> np.ndarray:
    """Production's argmax of (x + c + b_key(x)) p(x): the value term on the key's grid."""
    return requote_curve(rows, charges, vadd=vadd)


def p_at(rows: np.ndarray, x: np.ndarray, ratio: np.ndarray | None = None, shift: np.ndarray | None = None, curves: dict | None = None, curve_of: np.ndarray | None = None) -> np.ndarray:
    """The (reshaped) fill probability at concession x for each row; the validation of a curve form at the logged concession."""
    out = np.full(len(rows), np.nan)
    for (f, k), ii in pd.DataFrame({'f': _files[rows], 'k': _keys[rows]}).groupby(['f', 'k'], sort=False).indices.items():
        cur = BT_CURVES.get(f, {}).get(k)
        if cur is None:
            continue
        q, p = cur; xi = x[ii]
        if ratio is not None:
            out[ii] = np.interp(xi * ratio[ii], q, p)
        elif shift is not None:
            out[ii] = _sigmoid(_logit(np.interp(xi, q, p)) + shift[ii])
        elif curves is not None:
            for t in np.unique(curve_of[ii]):
                sel = curve_of[ii] == t; out[ii[sel]] = np.interp(xi[sel], q, curves.get((f, k, int(t)), p))
        else:
            out[ii] = np.interp(xi, q, p)
    return out


def level_to(y_v: np.ndarray, y_b: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """A constant width per month (bp, positive = cheaper) added to y_v so its win rate on the month's paired rows equals y_b's: the width nearest zero that does it, by bisection."""
    y = y_v.copy(); widths = {}
    for m in TEST_M:
        r = rows[_month[rows] == m]
        if not len(r):
            continue
        target = float(bt_win_fast(r, y_b[r]).mean())

        def wr(w: float) -> float:
            return float(bt_win_fast(r, y_v[r] + w / 100.0).mean())

        w0 = wr(0.0)
        if abs(w0 - target) <= 1e-12:
            w = 0.0
        elif w0 > target:
            lo, hi = 0.0, CFG.level_range_bp
            if wr(hi) > target:
                w = hi
            else:
                for _ in range(CFG.level_iters):
                    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if wr(mid) > target else (lo, mid)
                w = hi
        else:
            lo, hi = -CFG.level_range_bp, 0.0
            if wr(lo) < target:
                w = lo
            else:
                for _ in range(CFG.level_iters):
                    mid = 0.5 * (lo + hi); lo, hi = (lo, mid) if wr(mid) < target else (mid, hi)
                w = lo
        y[r] = y_v[r] + w / 100.0; widths[m] = w
    return y, widths


def book_sd(win: np.ndarray) -> dict:
    """Ex-ante daily sd of the policy's own book on the test days (fills held BT_HOLD days) and gross DV01: lab bt_book_g without the realised part."""
    w = win & (_pb_u > 0); di = _pb_di[w] + 1
    F = np.column_stack([np.bincount(di, weights=EXPO[w, c], minlength=_pb_nd + 1) for c in range(_D)]); I = np.bincount(di, weights=V_IDIO[w], minlength=_pb_nd + 1); U = np.bincount(di, weights=_pb_u[w], minlength=_pb_nd + 1)
    F, I, U = F.cumsum(0), I.cumsum(0), U.cumsum(0); j = np.arange(_pb_nd); lo = np.maximum(0, j - (BT_HOLD - 1)); E, Iv, Uv = F[j] - F[lo], I[j] - I[lo], U[j] - U[lo]
    sd = np.array([np.sqrt(max(E[k] @ SIGMA[k] @ E[k], 0.0) + Iv[k]) if k in SIGMA else np.nan for k in j])
    d = _tdays & np.isfinite(sd)
    return {'sd': float(np.mean(sd[d])) if d.any() else np.nan, 'dv01': float(np.mean(Uv[_tdays])) if _tdays.any() else np.nan}


def s_book(rows: np.ndarray, j: int, E: np.ndarray, expo: np.ndarray) -> np.ndarray:
    """Marginal risk of each fill against book exposure E on day j (lab 14e bt_marginal_g): BT_HOLD (2 e'S E + e'S e + v) / (2 u), bp per unit gamma."""
    S = SIGMA[j]; e = expo[rows]
    return BT_HOLD * (2.0 * e @ (S @ E) + np.einsum('ij,jk,ik->i', e, S, e) + V_IDIO[rows]) / (2.0 * _pb_u[rows])


_vol2 = VOL['total'] ** 2 * (_hold_req / BT_HOLD)


def book_replay(c0: np.ndarray, o0: np.ndarray, kind: str, gamma: float, extra_bp: float = 0.0, curve: dict | None = None) -> np.ndarray:
    """Sequential by business day from the first day with a covariance. kind: 'own real' = the own-CUSIP concentration term on the real book, Q_i = the
    WFTrade start-of-day DV01 in the CUSIP plus the policy's incremental fills over production's in the last BT_HOLD days; 'own replayed' = the v1
    definition, the policy's own replayed fills only; 'hedged' = the book term with the tenor block zeroed and the hedge cost. Test-month requests are
    re-quoted with the charge; earlier days quote the base and only build the book."""
    opto = o0.copy(); F = np.zeros((_pb_nd, _D)); Qp = np.zeros((_pb_nd, _n_cusip)); Qv = np.zeros((_pb_nd, _n_cusip)); expo = EXPO_H
    for j in range(_pb_nd):
        rows = _pb_rows_day.get(j, np.zeros(0, int)); act = rows[np.isfinite(o0[rows]) & (_pb_u[rows] > 0)]
        if not len(act):
            continue
        if j in SIGMA and _tdays[j]:
            lo = max(0, j - (BT_HOLD - 1)); ci = _ci[act]
            if kind == 'own real':
                Qi = np.maximum(POS_SOD[ci, j] + Qp[lo:j].sum(0)[ci] - Qv[lo:j].sum(0)[ci], 0.0); s = _vol2[act] * (Qi + _pb_u[act] / 2.0)
            elif kind == 'own replayed':
                Qi = Qp[lo:j].sum(0)[ci]; s = _vol2[act] * (Qi + _pb_u[act] / 2.0)
            else:
                s = s_book(act, j, F[lo:j].sum(0), expo)
            c_ = c0[act] - extra_bp - gamma * s
            opto[act] = requote(act, c_) if curve is None else requote_curve(act, c_, **{k: (v[act] if isinstance(v, np.ndarray) else v) for k, v in curve.items()})
        w = bt_win_fast(act, opto[act] + _adj[act]); wv = _w_van_all[act]
        F[j] = expo[act][w].sum(0); np.add.at(Qp[j], _ci[act][w], _pb_u[act][w]); np.add.at(Qv[j], _ci[act][wv], _pb_u[act][wv])
    return opto


def frontier_quote(regime: str, width_by_month: dict) -> np.ndarray:
    """Production widened by a (possibly different) width per month, from the cached width grid of the regime."""
    y = np.full(N, np.nan); X = FRONT_X[regime]
    for m, w in width_by_month.items():
        col = f'{w:+.1f}'
        if col not in X.columns:
            continue
        r = X.index.to_numpy()[(_month[X.index.to_numpy()] == m)]; y[r] = bt_opto(_m_sig[r], X.loc[r, col].to_numpy(float)) + _adj[r]
    return y


def match_frontier(y_v: np.ndarray, regime: str, rows: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """The width per month whose win rate on the month's rows is nearest the variant's: the comparator at equal win rate. Returns the quote, the widths and the residual win-rate mismatch."""
    widths, mism = {}, {}; X = FRONT_X[regime]
    for m in TEST_M:
        r = rows[_month[rows] == m]
        if not len(r):
            continue
        target = float(bt_win_fast(r, y_v[r]).mean()); best, bw, bm = None, None, np.inf
        for col in X.columns:
            yw = bt_opto(_m_sig[r], X.loc[r, col].to_numpy(float)) + _adj[r]; wr = float(bt_win_fast(r, yw).mean())
            if abs(wr - target) < bm:
                best, bw, bm = col, float(col), abs(wr - target)
        widths[m] = bw; mism[m] = (float(bt_win_fast(r, bt_opto(_m_sig[r], X.loc[r, best].to_numpy(float)) + _adj[r]).mean()) - target)
    return frontier_quote(regime, widths), widths, mism


VALUATIONS = ('record', 'guard', 'common markout', 'hedged', 'other side')
_usd_cap = float(np.nanquantile(_usd[_tmw], CFG.dv01_trim_q)); _usd_trim = np.minimum(np.nan_to_num(_usd), _usd_cap); _usd_top = np.nan_to_num(_usd) > _usd_cap


def score_pair(name: str, y_v: np.ndarray, y_b: np.ndarray, family: str = '', setting: str = '', regime: str = '', pair: str = '', mode: str = 'raw',
               widths: dict | None = None, mismatch: dict | None = None, bk_base: dict | None = None) -> dict:
    """The paired scorecard: variant quote y_v against comparator y_b on the test-month rows where both exist. No thresholds: every column is a change against the comparator with its t."""
    rows = np.flatnonzero(_tmw & np.isfinite(y_v) & np.isfinite(y_b)); yv, yb = y_v[rows], y_b[rows]
    wv, wb = bt_win_fast(rows, yv), bt_win_fast(rows, yb); g = _grp[rows]; dv, db = 100.0 * (_thr[rows] - yv), 100.0 * (_thr[rows] - yb)
    px_v, px_b = 100.0 * (yv - _y_tr[rows]), 100.0 * (yb - _y_tr[rows]); kept = wv & wb; new = wv & ~wb; gone = wb & ~wv; keptc = kept & (g == 'won, cover')
    cv, cb = wv & (g == 'won, cover'), wb & (g == 'won, cover')
    res = {'variant': name, 'family': family, 'setting': setting, 'regime': regime, 'pair': pair, 'mode': mode, 'months': len(set(_month[rows])), 'requests': len(rows),
           'win rate': float(wv.mean()), 'win rate, base': float(wb.mean()), 'win rate change': float(wv.mean() - wb.mean()), 'wins kept': float(kept.sum() / max(wb.sum(), 1)),
           'new wins': int(new.sum()), 'wins given up': int(gone.sum()),
           'bp left on table (won, cover)': float(dv[cv].mean()) if cv.any() else np.nan, 'bp left on table, base': float(db[cb].mean()) if cb.any() else np.nan,
           'bp saved on kept cover wins': float((db - dv)[keptc].mean()) if keptc.any() else np.nan,
           'crossing on new wins (bp)': float(-dv[new & (g == 'lost, printed')].mean()) if (new & (g == 'lost, printed')).any() else np.nan,
           'comparator width (bp)': float(np.mean(list(widths.values()))) if widths else 0.0, 'win rate mismatch': float(np.mean(np.abs(list(mismatch.values())))) if mismatch else 0.0}
    res['cover change (bp)'] = res['bp left on table (won, cover)'] - res['bp left on table, base']
    vals = {'record': (_M_REC[rows], _M_REC[rows]), 'guard': (_M_GUARD[rows], _M_GUARD[rows]), 'common markout': (np.where(new, _cm_rec[rows], _M_REC[rows]), _M_REC[rows]),
            'hedged': (_M_REC[rows] + _mmd5[rows] - CFG.hedge_cost_bp, _M_REC[rows] + _mmd5[rows] - CFG.hedge_cost_bp), 'other side': (_M_OTHER[rows], _M_OTHER[rows])}
    for key, (Mv, Mb) in vals.items():
        pv = np.where(wv, px_v + Mv, 0.0); pb = np.where(wb, px_b + Mb, 0.0); S = np.isfinite(pv) & np.isfinite(pb) & ~((wv & ~np.isfinite(Mv)) | (wb & ~np.isfinite(Mb)))
        d = np.where(S, pv - pb, np.nan); wk = _group_mean(d, _wk_code[rows], len(_wk_levels)); mm = _group_mean(d, _mo_code[rows], len(_mo_levels))
        res[f'delta | {key} (bp)'] = float(np.nanmean(d)) if S.any() else np.nan; res[f'NW t | {key}'] = nw_t(wk); res[f'months > 0 | {key}'] = int(np.nansum(mm > 0))
        if key in ('record', 'hedged'):
            u, ut = _usd[rows], _usd_trim[rows]; dd = np.where(S, (pv - pb), 0.0)
            res[f'delta $ per month | {key} ($k)'] = float(np.nansum(dd * u) / 1e3 / max(res['months'], 1)); res[f'delta $ per month, trimmed | {key} ($k)'] = float(np.nansum(dd * ut) / 1e3 / max(res['months'], 1))
            tot = np.nansum(dd * u); res[f'top 1% DV01 share of delta $ | {key}'] = float(np.nansum((dd * u)[_usd_top[rows]]) / tot) if tot != 0 else np.nan
        if key == 'record':
            res['P&L sample share'] = float(S.mean()); res['P&L per request (bp)'] = float(np.nanmean(pv[S])); res['P&L per request, base (bp)'] = float(np.nanmean(pb[S]))
            res['delta: common fills price'] = float(np.sum((px_v - px_b)[kept & S]) / max(S.sum(), 1)); res['delta: new wins'] = float(np.sum(pv[new & S]) / max(S.sum(), 1))
            res['delta: wins given up'] = float(-np.sum(pb[gone & S]) / max(S.sum(), 1))
            for cls, msk in (('manual modal', _man_modal[rows]), ('manual specific', ~_man_modal[rows])):
                dm = np.where(S & msk, pv - pb, np.nan); res[f'delta | record, {cls} (bp)'] = float(np.nanmean(dm)) if (S & msk).any() else np.nan
                res[f'NW t | record, {cls}'] = nw_t(_group_mean(dm, _wk_code[rows], len(_wk_levels)))
        if key == 'hedged':
            res['hedged $ per month, variant ($k)'] = float(np.nansum(np.where(S, pv, 0.0) * _usd[rows]) / 1e3 / max(res['months'], 1)); res['hedged $ per month, base ($k)'] = float(np.nansum(np.where(S, pb, 0.0) * _usd[rows]) / 1e3 / max(res['months'], 1))
    full_v = np.zeros(N, bool); full_v[rows] = wv; bk = book_sd(full_v)
    if bk_base is None:
        full_b = np.zeros(N, bool); full_b[rows] = wb; bk_base = book_sd(full_b)
    res['ex-ante daily sd ($k)'] = bk['sd'] / 1e3; res['ex-ante sd, base ($k)'] = bk_base['sd'] / 1e3; res['ex-ante sd vs base'] = bk['sd'] / bk_base['sd'] - 1.0 if bk_base['sd'] and np.isfinite(bk_base['sd']) else np.nan
    res['gross DV01 ($k/bp)'] = bk['dv01'] / 1e3
    res['hedged $ per $ of sd, variant'] = res['hedged $ per month, variant ($k)'] / res['ex-ante daily sd ($k)'] if res['ex-ante daily sd ($k)'] else np.nan
    res['hedged $ per $ of sd, base'] = res['hedged $ per month, base ($k)'] / res['ex-ante sd, base ($k)'] if res['ex-ante sd, base ($k)'] else np.nan
    res['min delta, sceptical (bp)'] = float(np.nanmin([res['delta | record (bp)'], res['delta | guard (bp)'], res['delta | common markout (bp)'], res['delta | hedged (bp)']]))
    return res


def dominance(r: dict) -> str:
    """Sign-only reading against the comparator: cover (bp left on cover wins lower), P&L (the delta positive under the record exit, the guard, the common markout and the hedged exit), risk (book sd not higher)."""
    cov = r['cover change (bp)'] < 0; pnl_all = r['min delta, sceptical (bp)'] > 0; pnl_rec = r['delta | record (bp)'] > 0; rsk = (r['ex-ante sd vs base'] <= 0) if pd.notna(r['ex-ante sd vs base']) else False
    parts = [n for n, ok in (('cover', cov), ('P&L under all four valuations', pnl_all), ('P&L under the record exit only', pnl_rec and not pnl_all), ('risk', rsk)) if ok]
    return 'better on ' + ', '.join(parts) if parts else 'worse on every axis'


def pareto(frame: pd.DataFrame) -> pd.Series:
    """Non-dominated rows on (cover change lower, min sceptical delta higher, sd change lower)."""
    a = frame[['cover change (bp)', 'min delta, sceptical (bp)', 'ex-ante sd vs base']].to_numpy(float).copy(); a[:, 1] = -a[:, 1]; a = np.nan_to_num(a, nan=np.inf)
    keep = np.ones(len(a), bool)
    for i in range(len(a)):
        keep[i] = not np.any(np.all(a <= a[i], axis=1) & np.any(a < a[i], axis=1))
    return pd.Series(keep, index=frame.index)


def wf_pick(settings: dict[str, tuple[np.ndarray, np.ndarray]], order: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Walk-forward pick among settings per test month by the hedged weekly $ Sharpe of the manual-kept quote on the earlier test months; the first month takes order[0]."""
    def sharpe(y: np.ndarray, months: list[str]) -> float:
        r = np.flatnonzero(_tmw & np.isfinite(y) & np.isin(_month, months))
        if not len(r):
            return np.nan
        w = bt_win_fast(r, y[r]); pnl = np.where(w, (100.0 * (y[r] - _y_tr[r]) + _M_REC[r] + _mmd5[r] - CFG.hedge_cost_bp) * _usd[r], 0.0); ok = np.isfinite(pnl)
        wk = np.bincount(_wk_code[r][ok], weights=pnl[ok], minlength=len(_wk_levels)); has = np.bincount(_wk_code[r][ok], minlength=len(_wk_levels)) > 0; wk = wk[has]
        return float(wk.mean() / wk.std(ddof=1)) if len(wk) > 2 and wk.std(ddof=1) > 0 else np.nan

    yM, y0, rows = np.full(N, np.nan), np.full(N, np.nan), []
    for i, m in enumerate(TEST_M):
        earlier = TEST_M[:i]
        if earlier:
            sc = {k: sharpe(v[0], earlier) for k, v in settings.items()}; best = max(sc, key=lambda k: (np.nan_to_num(sc[k], nan=-np.inf)))
        else:
            sc, best = {}, order[0]
        te = _month == m; yM[te] = settings[best][0][te]; y0[te] = settings[best][1][te]; rows.append({'test month': m, 'picked': best, **{f'train Sharpe {k}': v for k, v in sc.items()}})
    return yM, y0, rows


VARIANTS: dict[str, dict] = {}
print('machinery ready: requote by charge, curve and value; the sequential book with the real position; the frontier comparator; the scorecard without thresholds')

# %% [markdown]
# ## 4. References: production with and without the manual charge, the width frontiers, pair D
#
# Production quoted by the optimizer with its logged charge and with the manual charge removed, then widened on a 0.5 bp grid from -10 to +10 bp in
# both regimes. Those grids are the comparators of Section 8: for any variant, the production width whose win rate matches the variant's, month by
# month. Pair D reads the manual charge itself against production and against its own frontier at 23 percent.

# %%
CELL_T('4. References, the frontiers and pair D [4]')
_r_t0 = time.perf_counter()
_tm_rows = np.flatnonzero(_tmw)
Y_PM = _y_van.copy()
Y_P0 = np.full(N, np.nan); Y_P0[_tm_rows] = bt_final(requote(_tm_rows, _c_base[_tm_rows]), _tm_rows)
_w_van_all = bt_win_fast(_all, Y_PM)


def _width_grid_x(c: np.ndarray) -> pd.DataFrame:
    """Optimal quantile bp on the test-month winnable rows for every width of the grid (float64, so the quote is reproduced bit for bit)."""
    out = {}
    for w in CFG.frontier_widths:
        t0 = time.perf_counter(); out[f'{w:+.1f}'] = 100.0 * (requote(_tm_rows, c[_tm_rows], float(w)) - _m_sig[_tm_rows]); print(f'  width {w:+.1f}: {time.perf_counter() - t0:.0f}s')
    return pd.DataFrame(out, index=_tm_rows)


FRONT_X = {'M': cached('var_front_x_M', lambda: _width_grid_x(_ch), deps=[frame_fingerprint(BT, ['charges_level', 'threshold_y']), TEST_M], code=[requote, bt_optimize], cfg=('frontier_widths',)),
           '0': cached('var_front_x_0', lambda: _width_grid_x(_c_base), deps=[frame_fingerprint(BT, ['liquiditycharge', 'riskcharge', 'threshold_y']), TEST_M], code=[requote, bt_optimize], cfg=('frontier_widths',))}
assert np.allclose(bt_opto(_m_sig[_tm_rows], FRONT_X['M']['+0.0'].to_numpy(float)) + _adj[_tm_rows], Y_PM[_tm_rows], atol=1e-9, equal_nan=True), 'width 0 must reproduce production'
_fr_rows = []
for reg in ('M', '0'):
    for col in FRONT_X[reg].columns:
        yw = bt_opto(_m_sig[_tm_rows], FRONT_X[reg][col].to_numpy(float)) + _adj[_tm_rows]; ww = bt_win_fast(_tm_rows, yw)
        for m in TEST_M:
            r = _month[_tm_rows] == m; cvr = ww & r & (_grp[_tm_rows] == 'won, cover'); pv = np.where(ww & r, 100.0 * (yw - _y_tr[_tm_rows]) + _M_REC[_tm_rows], np.nan)
            _fr_rows.append({'regime': reg, 'width': float(col), 'month': m, 'win rate': float(ww[r].mean()), 'bp left on cover wins': float((100.0 * (_thr[_tm_rows] - yw))[cvr].mean()) if cvr.any() else np.nan,
                             'P&L per request, record (bp)': float(np.nanmean(np.where(r, np.where(ww, pv, 0.0), np.nan)))})
FRONTIER = pd.DataFrame(_fr_rows)
_fr_show = FRONTIER[FRONTIER['width'].isin([-10, -5, -2, -1, 0, 1, 2, 5, 10])].pivot_table(index=['regime', 'width'], columns='month', values='win rate')
print('Production\'s width frontier, win rate by month (manual kept = M, manual removed = 0; the full grid is 0.5 bp from -10 to +10):'); display(_fr_show.round(4))
_wbM, _wb0 = np.zeros(N, bool), np.zeros(N, bool); _wbM[_tm_rows] = bt_win_fast(_tm_rows, Y_PM[_tm_rows]); _wb0[_tm_rows] = bt_win_fast(_tm_rows, Y_P0[_tm_rows])
BK_BASE = {'M': book_sd(_wbM), '0': book_sd(_wb0)}
_d_rows = [score_pair('production, manual removed', Y_P0, Y_PM, family='reference', setting='manual removed', regime='0', pair='D', mode='vs production', bk_base=BK_BASE['M'])]
_yf, _wf_, _mm_ = match_frontier(Y_P0, 'M', _tm_rows)
_d_rows.append(score_pair('production, manual removed', Y_P0, _yf, family='reference', setting='manual removed', regime='0', pair='D', mode='vs frontier at equal win rate', widths=_wf_, mismatch=_mm_))
_yl, _wd = level_to(Y_P0, Y_PM, _tm_rows)
_d_rows.append(score_pair('production, manual removed', _yl, Y_PM, family='reference', setting='manual removed', regime='0', pair='D', mode='levelled to production', widths=_wd, bk_base=BK_BASE['M']))
PAIR_D = pd.DataFrame(_d_rows).set_index('mode'); PAIR_D['reading'] = [dominance(r) for _, r in PAIR_D.iterrows()]
SHOW = ['win rate', 'win rate, base', 'wins kept', 'new wins', 'wins given up', 'bp left on table (won, cover)', 'bp left on table, base', 'cover change (bp)', 'bp saved on kept cover wins', 'crossing on new wins (bp)',
        'comparator width (bp)', 'win rate mismatch', 'delta | record (bp)', 'NW t | record', 'months > 0 | record', 'delta | guard (bp)', 'NW t | guard', 'delta | common markout (bp)', 'NW t | common markout',
        'delta | hedged (bp)', 'NW t | hedged', 'min delta, sceptical (bp)', 'delta | other side (bp)', 'delta $ per month | record ($k)', 'delta $ per month, trimmed | record ($k)', 'top 1% DV01 share of delta $ | record',
        'delta | record, manual modal (bp)', 'NW t | record, manual modal', 'delta | record, manual specific (bp)', 'NW t | record, manual specific', 'P&L sample share',
        'ex-ante daily sd ($k)', 'ex-ante sd vs base', 'gross DV01 ($k/bp)', 'hedged $ per $ of sd, variant', 'hedged $ per $ of sd, base', 'reading']
_man_m = pd.Series(_man[_tmw]).groupby(_month[_tmw]).agg(['mean', 'median']); _man_m.columns = ['manual charge, mean (bp)', 'median (bp)']
_wr_m = pd.DataFrame({'win rate, manual kept': pd.Series(_wbM[_tm_rows]).groupby(_month[_tm_rows]).mean(), 'win rate, manual removed': pd.Series(_wb0[_tm_rows]).groupby(_month[_tm_rows]).mean(),
                      'share of requests at the month\'s modal manual charge': pd.Series(_man_modal[_tm_rows]).groupby(_month[_tm_rows]).mean()})
print('The manual charge by test month, production\'s win rate with and without it, and the share of requests carrying the month\'s modal charge (no request-specific trader input):'); display(_man_m.join(_wr_m).round(4))
print('PAIR D (reference): production without the manual charge, read three ways:'); display(PAIR_D[SHOW].T)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for reg, col in (('M', '#4C72B0'), ('0', '#C44E52')):
    f_ = FRONTIER[FRONTIER['regime'] == reg].groupby('width')[['win rate', 'P&L per request, record (bp)']].mean()
    ax[0].plot(f_['win rate'], f_['P&L per request, record (bp)'], '-', color=col, lw=1.2, label='manual kept' if reg == 'M' else 'manual removed')
ax[0].set_xlabel('win rate'); ax[0].set_ylabel('P&L per request, record exit (bp)'); ax[0].set_title('Width frontiers of the two regimes (0.5 bp grid, mean over months)', fontsize=10); ax[0].legend(fontsize=8)
ax[1].bar(np.arange(len(_wr_m)), _wr_m['win rate, manual kept'], width=0.4, color='#4C72B0', label='manual kept'); ax[1].bar(np.arange(len(_wr_m)) + 0.4, _wr_m['win rate, manual removed'], width=0.4, color='#C44E52', label='manual removed')
ax[1].set_xticks(np.arange(len(_wr_m)) + 0.2); ax[1].set_xticklabels(_wr_m.index, fontsize=8); ax[1].set_title('Production win rate by month, with and without the manual charge', fontsize=10); ax[1].legend(fontsize=8)
savefig('v4_references')
record('variants_references', frontier=flat_records(FRONTIER), pair_D=flat_records(PAIR_D.drop(columns=['reading'])), pair_D_reading=PAIR_D['reading'].to_dict(), manual_by_month=flat_records(_man_m.join(_wr_m)), seconds=time.perf_counter() - _r_t0)
print(f'[4] {time.perf_counter() - _r_t0:.0f}s')

# %% [markdown]
# ## 5. The fill curve as a conditional distribution of the competing bid
#
# v1 established idiosyncratic volatility as a state of the fill probability (the only sigma that improves the held-out log-likelihood of production's
# curve) and that the traders already price it. This cell keeps the two parametric forms on idiosyncratic volatility with the parameter picked
# walk-forward by log-likelihood, the own-curve refit by volatility tercile, and adds the state-conditioned curve: the desk's own curve per key by
# idiosyncratic-vol tercile x manual-charge class (the month's modal charge, or a request-specific one) x the lagged common-move tercile of the day,
# shrunk to production's curve. Every form is validated at the logged concession and reported by manual class; nothing gates.

# %%
CELL_T('5. The fill curve as a conditional distribution [5]')
_c_t0 = time.perf_counter()
SIGMAS = {k: v for k, v in {'systematic': VOL.get('rate + factor'), 'idiosyncratic': VOL.get('idiosyncratic'), 'total': VOL.get('total'), 'conditional scale': _a_risk}.items() if k in CFG.sigmas and v is not None}
_X = 100.0 * (_thr - _m_sig - _adj); _won_g, _lost_g = np.isin(_grp, ['won, cover', 'won, no cover']), _grp == 'lost, printed'


def _ref_by_key(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref, mu, sd = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan); ls = np.log(np.where(sig > 0, sig, np.nan))
    for m in BT_MONTHS[1:]:
        tr = (_month < m) & _winable & np.isfinite(ls); te = _month == m
        if tr.sum() < 100:
            continue
        med = pd.Series(sig[tr]).groupby(_keys[tr]).median(); ref[te] = pd.Series(_keys[te]).map(med).to_numpy(float)
        ref[te & ~np.isfinite(ref)] = float(np.median(sig[tr])); mu[te], sd[te] = float(np.mean(ls[tr])), float(np.std(ls[tr]))
    return ref, mu, sd


def _state_codes(sig: np.ndarray, train: np.ndarray, with_state: bool) -> np.ndarray:
    """Per request: idio-vol tercile (edges on the training rows) x manual class x lagged common-move tercile; terciles only when with_state is False."""
    ls = np.log(np.where(sig > 0, sig, np.nan)); q1, q2 = np.nanquantile(ls[train], [1 / 3, 2 / 3]); terc = np.where(ls <= q1, 0, np.where(ls <= q2, 1, 2)); terc[~np.isfinite(ls)] = 1
    return terc if not with_state else terc * 6 + _man_modal.astype(int) * 3 + _mkt_state


def _state_refit(sig: np.ndarray, n0: float, train: np.ndarray, with_state: bool) -> tuple[dict, np.ndarray]:
    """Per (file, key, state): the desk's own curve on the training rows shrunk to production with pseudo-count n0."""
    codes = _state_codes(sig, train, with_state); curves = {}
    if n0 == np.inf:
        return curves, codes
    for (f, k, t), ii in pd.DataFrame({'f': _files[train], 'k': _keys[train], 't': codes[train]}).groupby(['f', 'k', 't'], sort=False).indices.items():
        cur = BT_CURVES.get(f, {}).get(k)
        if cur is None:
            continue
        q, p = cur; rows = np.flatnonzero(train)[ii]; xw, xl = np.sort(_X[rows][_won_g[rows]]), np.sort(_X[rows][_lost_g[rows]]); n = len(rows)
        emp = ((len(xw) - np.searchsorted(xw, q, 'left')) + (len(xl) - np.searchsorted(xl, q, 'right'))) / max(n, 1)
        curves[(f, k, int(t))] = (n * emp + (0.0 if n0 == 0 else n0) * p) / (n + (0.0 if n0 == 0 else n0))
    return curves, codes


def _ll(p: np.ndarray, won: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.where(won, np.log(p), np.log(1 - p))


def _curve_family() -> dict:
    out, val_rows, cal_rows, llm = {}, [], [], {}
    p_prod = p_at(_all, _xlog); ll0 = _ll(p_prod, _wlog)

    def finish(nm, form, sname, p_hat, yM, y0, kw, t0):
        ev = np.flatnonzero(np.isfinite(_xlog) & np.isfinite(p_hat) & (_month >= BT_MONTHS[1])); gain = _ll(p_hat[ev], _wlog[ev]) - ll0[ev]
        mm = _group_mean(gain, _mo_code[ev], len(_mo_levels)); llm[nm] = dict(zip(_mo_levels, mm))
        sig = SIGMAS[sname] if sname in SIGMAS else np.full(N, np.nan); ok = np.isfinite(sig[ev]) & (sig[ev] > 0); dec = np.full(len(ev), np.nan)
        if ok.sum() > 100:
            dec[ok] = pd.qcut(pd.Series(sig[ev][ok]).rank(method='first'), 10, labels=False).to_numpy()
        cal = pd.DataFrame({'decile': dec, 'p_hat': p_hat[ev], 'p_prod': p_prod[ev], 'won': _wlog[ev].astype(float)}).dropna().groupby('decile').mean()
        val_rows.append({'variant': nm, 'sigma': sname, 'form': form, 'requests': int(len(ev)), 'LL gain per request vs production': float(gain.mean()), 'months with LL gain > 0': f'{int(np.nansum(mm > 0))}/{int(np.isfinite(mm).sum())}',
                         'LL gain, manual modal': float(gain[_man_modal[ev]].mean()) if _man_modal[ev].any() else np.nan, 'LL gain, manual specific': float(gain[~_man_modal[ev]].mean()) if (~_man_modal[ev]).any() else np.nan,
                         'calibration: max |p_hat - won| by sigma decile': float((cal['p_hat'] - cal['won']).abs().max()) if len(cal) else np.nan,
                         'calibration: max |p_prod - won| by sigma decile': float((cal['p_prod'] - cal['won']).abs().max()) if len(cal) else np.nan, 'quoted test requests': int(np.isfinite(yM).sum()), 'seconds': time.perf_counter() - t0})
        cal_rows.extend([{'variant': nm, 'decile': int(d), **r.to_dict()} for d, r in cal.iterrows()])
        out[nm] = {'family': 'fill curve', 'setting': nm, 'yM': bt_final(yM), 'y0': bt_final(y0), 'kw': kw, 'sigma': sname}
        print(f'  {nm}: LL gain {val_rows[-1]["LL gain per request vs production"]:+.5f} per request, {val_rows[-1]["months with LL gain > 0"]} months | {time.perf_counter() - t0:.0f}s')

    for sname, sig in SIGMAS.items():
        ref, mu, sd = _ref_by_key(sig); z = np.nan_to_num(np.clip((np.log(np.where(sig > 0, sig, np.nan)) - mu) / np.where(sd > 0, sd, np.nan), -3, 3))
        fixed = {}
        for form, grid in (('stretch', CFG.stretch_exponents), ('logit', CFG.logit_kappas)):
            for par in grid:
                t0 = time.perf_counter(); nm = f'{form} {par:g} | {sname}'; yM, y0, p_hat = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
                if form == 'stretch':
                    prm = np.where(np.isfinite(ref) & (sig > 0), (ref / np.where(sig > 0, sig, np.nan)) ** par, 1.0); kw = {'ratio': prm}
                else:
                    prm = par * z; kw = {'shift': prm}
                yM[_tm_rows] = requote_curve(_tm_rows, _ch[_tm_rows], **{k: v[_tm_rows] for k, v in kw.items()}); y0[_tm_rows] = requote_curve(_tm_rows, _c_base[_tm_rows], **{k: v[_tm_rows] for k, v in kw.items()})
                ev = np.flatnonzero(np.isfinite(ref)); p_hat[ev] = p_at(ev, _xlog[ev], **{k: v[ev] for k, v in kw.items()})
                finish(nm, form, sname, p_hat, yM, y0, kw, t0); fixed[(form, par)] = nm
            # the walk-forward parameter: per test month, the grid value with the highest cumulative LL gain on the earlier months
            t0 = time.perf_counter(); nm = f'{form} walk-forward | {sname}'; yM, y0, p_hat = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan); picks = {}
            kw_wf = {'ratio' if form == 'stretch' else 'shift': np.full(N, np.nan)}
            for i, m in enumerate(BT_MONTHS[1:]):
                earlier = [x for x in BT_MONTHS[1:i + 1]]; sc = {par: float(np.nansum([llm[fixed[(form, par)]].get(x, np.nan) for x in earlier])) for par in grid} if earlier else {par: 0.0 for par in grid}
                par = max(sc, key=sc.get) if earlier else grid[0]; picks[m] = par; te = _month == m; src = out[fixed[(form, par)]]
                yM[te] = src['yM'][te] - _adj[te]; y0[te] = src['y0'][te] - _adj[te]
                prm = src['kw']['ratio' if form == 'stretch' else 'shift']; kw_wf['ratio' if form == 'stretch' else 'shift'][te] = prm[te]
                tev = np.flatnonzero(te & np.isfinite(ref)); p_hat[tev] = p_at(tev, _xlog[tev], **{k: v[tev] for k, v in src['kw'].items()})
            finish(nm, form, sname, p_hat, yM, y0, kw_wf, t0); out[nm]['picks'] = picks
        for with_state, label in ((False, 'tercile refit'), (True, 'state refit')):
            t0 = time.perf_counter(); nm = f'{label} | {sname}'; yM, y0, p_hat = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan); n0_rows = []
            for i, m in enumerate(BT_MONTHS[1:], start=1):
                train = (_month < m) & np.isfinite(_X) & np.isfinite(_xlog); te = _month == m
                if train.sum() < 100:
                    continue
                prev = BT_MONTHS[i - 1]; tr_prev = (_month < prev) & np.isfinite(_X) & np.isfinite(_xlog); v_prev = (_month == prev) & np.isfinite(_xlog); lls = {}
                for n0 in CFG.n0_grid:
                    if tr_prev.sum() < 100:
                        lls[n0] = -np.inf if n0 != np.inf else 0.0; continue
                    cv, tc = _state_refit(sig, n0, tr_prev, with_state); vr = np.flatnonzero(v_prev); lls[n0] = float(np.sum(_ll(p_at(vr, _xlog[vr], curves=cv, curve_of=tc), _wlog[vr])))
                n0 = max(lls, key=lls.get); cv, tc = _state_refit(sig, n0, train, with_state); ter = np.flatnonzero(te); p_hat[ter] = p_at(ter, _xlog[ter], curves=cv, curve_of=tc)
                if m in TEST_M:
                    tw = ter[_winable[ter]]; yM[tw] = requote_curve(tw, _ch[tw], curves=cv, curve_of=tc); y0[tw] = requote_curve(tw, _c_base[tw], curves=cv, curve_of=tc)
                n0_rows.append({'month': m, 'n0': n0})
            finish(nm, label, sname, p_hat, yM, y0, None, t0); out[nm]['n0'] = n0_rows
    return {'variants': out, 'validation': pd.DataFrame(val_rows).set_index('variant'), 'calibration': pd.DataFrame(cal_rows), 'll_by_month': llm}


_cf = cached('var_curves', _curve_family, deps=[frame_fingerprint(BT, ['charges_level', 'threshold_y', 'optimalQuantileBp']), TEST_M, {k: float(np.nansum(v)) for k, v in SIGMAS.items()}, float(_man_modal.sum()), float(_mkt_state.sum())],
             code=[requote_curve, _argmax_dedup, p_at, _ref_by_key, _state_refit, _state_codes, bt_optimize], cfg=('sigmas', 'stretch_exponents', 'logit_kappas', 'n0_grid', 'chunk', 'quant_step'))
CURVE_VAL, CURVE_CAL = _cf['validation'], _cf['calibration']
for k_, v_ in _cf['variants'].items():
    VARIANTS[k_] = v_
print('Fill-curve validation at the logged concession (LL gain > 0 = the reshaped curve predicts the logged outcome better than production\'s; by manual class beside):'); display(CURVE_VAL.drop(columns=['seconds']).round(5))
print('Walk-forward parameter picks:', {k: v['picks'] for k, v in _cf['variants'].items() if 'picks' in v})
print('Shrinkage pseudo-count n0 of the refits by month:', {k: {r['month']: r['n0'] for r in v['n0']} for k, v in _cf['variants'].items() if 'n0' in v})
_best_form = CURVE_VAL['LL gain per request vs production'].idxmax() if len(CURVE_VAL) else None
if _best_form is not None:
    print(f'Calibration by sigma decile of the best form ({_best_form}):'); display(CURVE_CAL[CURVE_CAL['variant'] == _best_form].set_index('decile')[['p_hat', 'p_prod', 'won']].round(4).T)
record('variants_curves', validation=flat_records(CURVE_VAL), calibration=flat_records(CURVE_CAL), settings=list(_cf['variants']), seconds=time.perf_counter() - _c_t0)
print(f'[5] {time.perf_counter() - _c_t0:.0f}s')

# %% [markdown]
# ## 6. The concentration term on the real book
#
# v1's own-CUSIP term was the one charge that moved the book: at production's win rate it cut ex-ante risk by two thirds for 0.9 bp of cover, with
# P&L flat. It was built on the policy's own replayed fills. Here `Q_i` is the real inventory: the WFTrade start-of-day DV01 in the CUSIP plus the
# policy's incremental fills over production's in the last h days, so for production the term sees the desk's actual book. The replayed-book version
# is kept as the v1 reference and the hedged book term as the 14e reference. `gamma_k = k / median s` on production's book in the first test month.

# %%
CELL_T('6. The concentration term on the real book [6]')
_g_t0 = time.perf_counter()
_m1 = TEST_M[0]; _r1 = np.flatnonzero((_month == _m1) & _winable & (_pb_u > 0))


def _s_on_vanilla(kind: str) -> np.ndarray:
    """s on production's book for the first test month's requests: the scale of gamma."""
    F = np.zeros((_pb_nd, _D)); Qv = np.zeros((_pb_nd, _n_cusip)); s = np.full(len(_r1), np.nan); pos = pd.Series(np.arange(len(_r1)), index=_r1)
    for j in range(_pb_nd):
        rows = _pb_rows_day.get(j, np.zeros(0, int)); act = rows[(_pb_u[rows] > 0)]
        if not len(act):
            continue
        if _pb_month[j] == _m1 and j in SIGMA:
            lo = max(0, j - (BT_HOLD - 1)); a1 = act[np.isin(act, _r1)]
            if len(a1):
                ci = _ci[a1]
                if kind == 'own real':
                    s[pos[a1].to_numpy()] = _vol2[a1] * (np.maximum(POS_SOD[ci, j], 0.0) + _pb_u[a1] / 2.0)
                elif kind == 'own replayed':
                    s[pos[a1].to_numpy()] = _vol2[a1] * (Qv[lo:j].sum(0)[ci] + _pb_u[a1] / 2.0)
                else:
                    s[pos[a1].to_numpy()] = s_book(a1, j, F[lo:j].sum(0), EXPO_H)
        w = _w_van_all[act]; F[j] = EXPO_H[act][w].sum(0); np.add.at(Qv[j], _ci[act][w], _pb_u[act][w])
        if _pb_month[j] > _m1:
            break
    return s


S_MED = {k: float(np.nanmedian(_s_on_vanilla(k))) for k in ('own real', 'own replayed', 'hedged')}
_s_real = _s_on_vanilla('own real'); print('gamma scale: median s on production\'s book in', _m1, {k: f'{v:.3g}' for k, v in S_MED.items()},
                                           f'| real position held at the request: {np.mean(POS_SOD[_ci[_r1], _pb_di[_r1]] > 0):.1%} of first-month requests, median DV01 when held ${np.median(POS_SOD[_ci[_r1], _pb_di[_r1]][POS_SOD[_ci[_r1], _pb_di[_r1]] > 0]) if (POS_SOD[_ci[_r1], _pb_di[_r1]] > 0).any() else 0:,.0f} per bp')


def _charge_family() -> dict:
    out = {}
    o_M_all = requote(_all, _ch); o_0_all = o_M_all.copy(); o_0_all[_tm_rows] = requote(_tm_rows, _c_base[_tm_rows])
    for kind, grid in (('own real', CFG.k_grid), ('own replayed', CFG.k_grid), ('hedged', CFG.book_k_grid)):
        for k in grid:
            t0 = time.perf_counter(); g = k / S_MED[kind]; nm = f'{kind} | k={k}'; extra = CFG.hedge_cost_bp if kind == 'hedged' else 0.0
            yM = book_replay(_ch, o_M_all, kind, g, extra); y0 = book_replay(_c_base, o_0_all, kind, g, extra); yM[~_tmw] = np.nan; y0[~_tmw] = np.nan
            out[nm] = {'family': {'own real': 'concentration, real book', 'own replayed': 'concentration, replayed book', 'hedged': 'book inventory, hedged'}[kind], 'setting': f'k={k}', 'yM': bt_final(yM), 'y0': bt_final(y0)}
            print(f'  {nm}: {time.perf_counter() - t0:.0f}s')
    return out


_gf = cached('var_charges', _charge_family, deps=[frame_fingerprint(BT, ['charges_level', 'threshold_y']), TEST_M, float(np.nansum(_vol2)), float(np.nansum(EXPO)), S_MED, float(POS_SOD.sum()), len(SIGMA)],
             code=[book_replay, s_book, requote, bt_optimize], cfg=('k_grid', 'book_k_grid', 'hedge_cost_bp'))
for k_, v_ in _gf.items():
    VARIANTS[k_] = v_
CHARGE_PICKS = {}
for kind, fam in (('own real', 'concentration, real book'), ('own replayed', 'concentration, replayed book')):
    sets = {f'{kind} | k={k}': (VARIANTS[f'{kind} | k={k}']['yM'], VARIANTS[f'{kind} | k={k}']['y0']) for k in CFG.k_grid if f'{kind} | k={k}' in VARIANTS}
    if sets:
        yM, y0, rows = wf_pick(sets, list(sets)); VARIANTS[f'{kind} | walk-forward k'] = {'family': fam, 'setting': 'walk-forward k', 'yM': yM, 'y0': y0}
        t_ = pd.DataFrame(rows).set_index('test month'); t_.columns = [c.replace(f'train Sharpe {kind} | ', 'Sharpe ') for c in t_.columns]; t_['picked'] = t_['picked'].str.replace(f'{kind} | ', ''); CHARGE_PICKS[kind] = t_
print('Walk-forward k (hedged weekly $ Sharpe of the manual-kept quote on the earlier test months; the first test month takes k = 1):'); display(pd.concat(CHARGE_PICKS, names=['family']).round(3))
record('variants_charges', s_median=S_MED, picks={k: flat_records(v) for k, v in CHARGE_PICKS.items()}, seconds=time.perf_counter() - _g_t0)
print(f'[6] {time.perf_counter() - _g_t0:.0f}s')

# %% [markdown]
# ## 7. The value of a win as a function of the concession: b(x)
#
# Production values every win at the same key at `x + c` wherever on the curve it was won. The post-trade state differs with x: a win far inside the
# cover is more often a weak cover, a stale mark or a bond nobody else wanted. `b(x) = E[markout | won at x]` is estimated per key group (coupon bin x
# call group) and x-quintile, walk-forward, from two sources: the exit of record on production's wins, and the desk's realised round trips from
# WFTrade on the matched fills. Empirical-Bayes shrinkage to the group mean; a step function on the key's grid; quoted as `argmax (x + c + b(x)) p(x)`.
# The combination with the walk-forward logit curve is the first variant to change both the distribution and the value along x.

# %%
CELL_T('7. The value of a win as a function of x [7]')
_b_t0 = time.perf_counter()
_won_rec = _w_van_all & np.isfinite(_M_REC); _won_real = _w_van_all & np.isfinite(_real_bp)


def _bx_tables(target: np.ndarray, have: np.ndarray, m: str) -> tuple[dict, dict, pd.DataFrame]:
    """Per key group: x-quintile edges on the earlier months' wins and the shrunk mean of the target per bin; returns edges, b per (group, bin) and a table."""
    tr = (_month < m) & have & np.isfinite(_xlog); edges, bvals, rows_ = {}, {}, []
    for grp_, ii in pd.Series(np.flatnonzero(tr)).groupby(_kgrp[tr]).indices.items():
        r = np.flatnonzero(tr)[ii]
        if len(r) < 50:
            continue
        e = np.quantile(_xlog[r], np.linspace(0, 1, CFG.x_bins + 1)[1:-1]); b = np.searchsorted(e, _xlog[r], side='right'); y = target[r]
        g = pd.DataFrame({'b': b, 'y': y}).groupby('b')['y'].agg(['mean', 'var', 'size']); ybar = float(y.mean())
        se2 = (g['var'].fillna(g['var'].median()) / g['size']).to_numpy(); tau2 = max(0.0, float(np.var(g['mean'].to_numpy()) - np.mean(se2)))
        shr = ybar + (tau2 / (tau2 + se2)) * (g['mean'].to_numpy() - ybar); edges[grp_] = e
        bvals[grp_] = np.array([float(shr[list(g.index).index(k)]) if k in g.index else ybar for k in range(CFG.x_bins)])
        rows_.extend([{'month': m, 'group': grp_, 'x bin': int(k), 'wins': int(g.loc[k, 'size']) if k in g.index else 0, 'raw mean (bp)': float(g.loc[k, 'mean']) if k in g.index else np.nan, 'b(x) (bp)': bvals[grp_][k]} for k in range(CFG.x_bins)])
    return edges, bvals, pd.DataFrame(rows_)


def _vadd_for(edges: dict, bvals: dict) -> dict:
    """The value term per (file, key) on the key's grid: b of the group's bin that each grid point falls in."""
    vadd = {}
    for f, ks in BT_CURVES.items():
        for k, (q, p) in ks.items():
            grp_ = _kgrp_of_key(k)
            if grp_ in edges:
                vadd[(f, k)] = bvals[grp_][np.searchsorted(edges[grp_], q, side='right')]
    return vadd


def _value_family() -> dict:
    out, tabs = {}, []
    srcs = [('record', _M_REC, _won_rec)] + ([('realised', _real_bp, _won_real)] if np.isfinite(_real_bp).sum() > 1000 else [])
    logit_wf = next((k for k in VARIANTS if k.startswith('logit walk-forward')), None)
    for sname, target, have in srcs:
        for combo in (False, True):
            if combo and logit_wf is None:
                continue
            t0 = time.perf_counter(); nm = f'b(x) {sname}' + (f' + {logit_wf}' if combo else ''); yM, y0 = np.full(N, np.nan), np.full(N, np.nan)
            for m in TEST_M:
                edges, bvals, tab = _bx_tables(target, have, m)
                if not combo:
                    tabs.append(tab.assign(source=sname))
                vadd = _vadd_for(edges, bvals); te = np.flatnonzero((_month == m) & _winable)
                kw = {} if not combo else {'shift': VARIANTS[logit_wf]['kw']['shift'][te]}
                yM[te] = requote_curve(te, _ch[te], vadd=vadd, **kw); y0[te] = requote_curve(te, _c_base[te], vadd=vadd, **kw)
            out[nm] = {'family': 'value b(x)' + (' + curve' if combo else ''), 'setting': nm, 'yM': bt_final(yM), 'y0': bt_final(y0)}
            print(f'  {nm}: {time.perf_counter() - t0:.0f}s')
    return {'variants': out, 'tables': pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()}


_kgrp_of_key = lambda k: (k.split('_')[0] if k.startswith('cpn3') else '_'.join(k.split('_')[:2]))  # noqa: E731
_vf = cached('var_value', _value_family, deps=[frame_fingerprint(BT, ['charges_level', 'threshold_y', 'optimalQuantileBp']), TEST_M, float(np.nansum(_M_REC[_won_rec])), float(np.nansum(_real_bp[_won_real])), [k for k in VARIANTS if k.startswith('logit walk-forward')]],
             code=[requote_curve, _argmax_dedup, _bx_tables, _vadd_for, bt_optimize], cfg=('x_bins', 'quant_step'))
for k_, v_ in _vf['variants'].items():
    VARIANTS[k_] = v_
BX_TAB = _vf['tables']
if len(BX_TAB):
    _last = BX_TAB[BX_TAB['month'] == TEST_M[-1]]
    print(f'b(x) on the wins before {TEST_M[-1]}: the expected markout by x-quintile and key group (bin 0 = the least aggressive fifth of production\'s concessions, {CFG.x_bins - 1} = the most aggressive):')
    display(_last.pivot_table(index=['source', 'group'], columns='x bin', values='b(x) (bp)').round(2))
    print('... wins behind each cell:'); display(_last.pivot_table(index=['source', 'group'], columns='x bin', values='wins').astype(int))
record('variants_value', tables=flat_records(BX_TAB), settings=list(_vf['variants']), seconds=time.perf_counter() - _b_t0)
print(f'[7] {time.perf_counter() - _b_t0:.0f}s')

# %% [markdown]
# ## 7b. The model signals as charges (off by default)
#
# Closed on the v1 run: nothing anywhere at 12 to 26 percent coverage. Kept behind `CFG.signals_on` so the line can be reopened with a new reason.

# %%
CELL_T('7b. The model signals as charges [7b]')
_s_t0 = time.perf_counter()


def _z_walk(sig: np.ndarray) -> np.ndarray:
    z = np.zeros(N)
    for m in BT_MONTHS[1:]:
        tr = (_month < m) & _winable & np.isfinite(sig); te = (_month == m) & np.isfinite(sig)
        if tr.sum() < 100:
            continue
        xs = np.sort(sig[tr]); n = len(xs); F = np.interp(sig[te], xs, (np.arange(n) + 0.5) / n, left=0.5 / n, right=1 - 0.5 / n); z[te] = np.clip(stats.norm.ppf(F), -3, 3)
    return z


def _signal_family() -> dict:
    out = {}
    for s, sig in SIG.items():
        z = _z_walk(sig)
        for k in CFG.signal_k:
            nm = f'{SIG_NAMES.get(s, s)} | k={k:g}'; yM, y0 = np.full(N, np.nan), np.full(N, np.nan)
            yM[_tm_rows] = requote(_tm_rows, _ch[_tm_rows] + k * z[_tm_rows]); y0[_tm_rows] = requote(_tm_rows, _c_base[_tm_rows] + k * z[_tm_rows])
            out[nm] = {'family': 'signal charge', 'setting': f'{SIG_NAMES.get(s, s)} k={k:g}', 'yM': bt_final(yM), 'y0': bt_final(y0), 'coverage': float(np.isfinite(sig[_tmw]).mean())}
    return out


if CFG.signals_on:
    _sf = cached('var_signals', _signal_family, deps=[frame_fingerprint(BT, ['charges_level', 'threshold_y']), TEST_M, {k: float(np.nansum(v)) for k, v in SIG.items()}], code=[requote, _z_walk, bt_optimize], cfg=('signal_k',))
    for k_, v_ in _sf.items():
        VARIANTS[k_] = v_
    print(f'{len(_sf)} signal settings quoted')
else:
    print('signals off (CFG.signals_on = False): closed on the v1 run')
print(f'[7b] {time.perf_counter() - _s_t0:.0f}s')

# %% [markdown]
# ## 8. The matrix
#
# Every variant in pairs A, B and C against three comparators: production as it is; production widened through its own optimizer to the variant's
# win rate (the equal-win-rate reading, the primary one); and the variant levelled to production (the v1 reading). No thresholds: each row carries
# the change in win rate, cover, crossing, the P&L delta under five valuations with its t, dollars count- and DV01-weighted, the manual-class split,
# and book risk, and a sign-only reading of which axes it is better on. The Pareto set per pair is the set of variants no other variant beats on
# cover, sceptical P&L and risk at once.

# %%
CELL_T('8. The matrix: pairs A, B, C against production, the frontier and levelled [8]')
_x_t0 = time.perf_counter(); _rows_vm = []
for nm, v_ in VARIANTS.items():
    for pair, yv, yb, reg, bk in (('A', v_['yM'], Y_PM, 'M', BK_BASE['M']), ('B', v_['y0'], Y_PM, 'M', BK_BASE['M']), ('C', v_['y0'], Y_P0, '0', BK_BASE['0'])):
        if not np.isfinite(yv).any():
            continue
        rows = np.flatnonzero(_tmw & np.isfinite(yv) & np.isfinite(yb))
        _rows_vm.append(score_pair(nm, yv, yb, family=v_['family'], setting=v_['setting'], regime=reg, pair=pair, mode='vs production', bk_base=bk))
        yf, wf_, mm_ = match_frontier(yv, reg, rows)
        _rows_vm.append(score_pair(nm, yv, yf, family=v_['family'], setting=v_['setting'], regime=reg, pair=pair, mode='vs frontier at equal win rate', widths=wf_, mismatch=mm_))
        yl, wd = level_to(yv, yb, rows)
        _rows_vm.append(score_pair(nm, yl, yb, family=v_['family'], setting=v_['setting'], regime=reg, pair=pair, mode='levelled to production', widths=wd, bk_base=bk))
VM = pd.DataFrame(_rows_vm); VM['reading'] = [dominance(r) for _, r in VM.iterrows()]
VM['pareto'] = False
for (pair, mode), idx in VM.groupby(['pair', 'mode']).indices.items():
    VM.loc[VM.index[idx], 'pareto'] = pareto(VM.iloc[idx]).to_numpy()
VERDICTS = {}
for pair, q in (('A', 'variant with the manual charge, against production'), ('B', 'variant without the manual charge, against production'), ('C', 'both without the manual charge')):
    for mode in ('vs frontier at equal win rate', 'vs production', 'levelled to production'):
        t = VM[(VM['pair'] == pair) & (VM['mode'] == mode)].set_index('variant').sort_values('min delta, sceptical (bp)', ascending=False, na_position='last')
        better = t.index[t['reading'].str.contains('P&L under all four') & t['reading'].str.contains('cover')].tolist()
        VERDICTS[f'{pair} | {mode}'] = (f'{len(better)} of {len(t)} better than the comparator on cover and on P&L under all four valuations' + (f': {better}' if better else '')
                                        + f'; Pareto set {t.index[t["pareto"]].tolist()}')
        _top = t.index[:25].union(t.index[t['pareto']], sort=False)
        print(f'\nPAIR {pair}, {mode.upper()} ({q}; ranked by the smallest delta over the four sceptical valuations; {len(_top)} of {len(t)} rows shown):'); display(t.loc[_top, SHOW + ['pareto']].round(4))
        print(f'READING (pair {pair}, {mode}): {VERDICTS[f"{pair} | {mode}"]}')
print('\nThe reference pair D (Section 4):', {m: f'{r["reading"]}; win rate {r["win rate"]:.3f} vs {r["win rate, base"]:.3f}, cover change {r["cover change (bp)"]:+.2f} bp, min sceptical delta {r["min delta, sceptical (bp)"]:+.3f} bp' for m, r in PAIR_D.iterrows()})
_fc = dict(zip(sorted(VM['family'].unique()), ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860', '#DD8452', '#64B5CD', '#8C8C8C', '#CCB974']))
fig, ax = plt.subplots(1, 3, figsize=(22, 6))
for a_, pair in ((ax[0], 'A'), (ax[1], 'C')):
    t = VM[(VM['pair'] == pair) & (VM['mode'] == 'vs frontier at equal win rate')]
    for fam, g in t.groupby('family'):
        a_.scatter(g['cover change (bp)'], g['min delta, sceptical (bp)'], s=20 + 200 * np.clip(-g['ex-ante sd vs base'].fillna(0), 0, 1), color=_fc.get(fam, '#999999'), label=fam, alpha=0.8, edgecolors=np.where(g['pareto'], 'k', 'none'))
    a_.axhline(0, color='k', lw=0.5); a_.axvline(0, color='k', lw=0.5); a_.set_xlabel('cover change vs production at equal win rate (bp, negative = less left on the table)'); a_.set_ylabel('smallest delta over the four sceptical valuations (bp)')
    a_.set_title(f'Pair {pair} at equal win rate: the true win is the lower-left of cover and upper of P&L (size = risk cut, black edge = Pareto)', fontsize=9); a_.legend(fontsize=7)
_fam = VM[(VM['pair'] == 'A') & (VM['mode'] == 'vs frontier at equal win rate')].groupby('family')['min delta, sceptical (bp)'].agg(['max', 'median'])
ax[2].barh(_fam.index, _fam['max'], color='#55A868', label='best setting'); ax[2].barh(_fam.index, _fam['median'], color='#937860', label='median setting'); ax[2].axvline(0, color='k', lw=0.6)
ax[2].set_title('Pair A at equal win rate: smallest sceptical delta by family (bp per request)', fontsize=9); ax[2].legend(fontsize=8); ax[2].tick_params(axis='y', labelsize=8)
savefig('v8_matrix')
record('variants_matrix', table=flat_records(VM.set_index(['variant', 'pair', 'mode'])), verdicts=VERDICTS, seconds=time.perf_counter() - _x_t0)
VM.to_parquet(OUT / 'variant_matrix.parquet')
print(f'matrix written: {OUT / "variant_matrix.parquet"} ({len(VM)} rows) | [8] {time.perf_counter() - _x_t0:.0f}s')

# %% [markdown]
# ## 9. Registry and runtimes

# %%
CELL_T('9. Registry and runtimes [9]')
CELL_T('end')
ct = pd.DataFrame(CELL_LOG); ct['share'] = ct['seconds'] / ct['seconds'].sum()
if CACHE_LOG:
    _cl = pd.DataFrame(CACHE_LOG); ct['cached stages'] = ct['cell'].map(_cl.groupby('cell').apply(lambda g: ', '.join(f'{r.stage} ({r.status}, {r.seconds:.0f}s)' for r in g.itertuples()), include_groups=False)).fillna('')
print(f'Runtime: {ct["seconds"].sum() / 60:.1f} min over {len(ct)} cells'); display(ct.set_index('cell').round(1))
record('runtime', total_seconds=float(ct['seconds'].sum()), cells=ct.round(2).to_dict(orient='records'))
(OUT / 'results_registry_variants.json').write_text(json.dumps(REGISTRY, indent=2, default=str), encoding='utf-8')
print('registry updated:', OUT / 'results_registry_variants.json')
