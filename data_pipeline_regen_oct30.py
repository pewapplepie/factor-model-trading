# %% [markdown]
# # Data pipeline regeneration: the window to 2026-10-30, the settle date and the carry charge on the RFQ log
#
# Run on the JupyterHub kernel from the lab root (`factor_model/`) or from `data_pipeline/`. Each cell says what it pulls and why; the pulls
# are resumable (re-run a cell and only unfinished months go again), the checks are read-only.
#
# **Why a plain re-run is not enough.** The store skips a partition it considers complete, which means written after its window closed. October
# was pulled on the 27th with a window that ended on the 23rd, so it reads as complete and would be skipped even with the new end date. And the
# RFQ log (`track_rfq`, every column of `muni_algo_trade_track`) was pulled before its settle column existed, so every month of it needs to go
# again: a dataset takes its schema from its first file, and a column missing there is silently dropped on read. Both need `force=True`.
#
# **What the width-charge notebook (v1.0.2) reads from here.** The settle date production priced with, resolved on `track_rfq` (any of
# `settle_date`, `settlement_date`, `settle_dt`); the carry finance charge, resolved on `strategy_output` first (`carryfinancecharges`, logged with
# the quote, `carryadjustedyield` beside it) and on `track_rfq` as the fallback. The carry is therefore already on disk from the October 20 pull;
# the settle date is what the re-pull is for.

# %%
import importlib
import sys
from pathlib import Path

import pandas as pd

# The lab reads the stores under data_pipeline/data, so pull there whether this runs from factor_model or data_pipeline.
PIPELINE_ROOT = Path('data_pipeline').resolve() if Path('data_pipeline/muni_data_pipeline.py').exists() else Path('.').resolve()
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import muni_data_pipeline as mdp  # noqa: E402
importlib.reload(mdp)
pd.set_option('display.max_columns', None); pd.set_option('display.width', 220)

START, END = '2026-01-01', '2026-10-30'
print('pipeline root:', PIPELINE_ROOT)
print('sources:', mdp.Config.sources)
for s_ in ('strategy_output', 'track_rfq', 'wftrade'):
    assert s_ in mdp.Config.sources, f'old module loaded ({s_} missing): git pull on the hub, then restart the kernel'

# %% [markdown]
# ## 1. What is on disk now, and what a plain pull would do
#
# Read-only. The plan shows, per source and month, whether the partition exists and reads as complete; `will_pull` is what `pull()` would do
# without `force`. Expect October to show `complete = True` for every source pulled on the 27th, which is the problem described above. The
# second block lists the columns the stores hold today that look like a settle date or a carry charge.

# %%
cfg_all = mdp.Config(root=PIPELINE_ROOT, start_date=START, end_date=END)
mdp.setup_logging(cfg_all)
plan = mdp.plan(cfg_all)
print(plan[plan['partition'] >= '2026-09'].to_string(index=False))


def columns_like(source: str, *needles: str) -> list[str]:
    d = cfg_all.store(source).dataset()
    if d is None:
        return ['(store empty)']
    return [c for c in d.schema.names if any(n in c.lower() for n in needles)]


for src in ('track_rfq', 'strategy_output'):
    print(f'{src}: settle-like {columns_like(src, "settl")} | carry-like {columns_like(src, "carry")}')
    s = cfg_all.store(src).summary()
    print(f'  partitions {s["partition"].min() if len(s) else None} .. {s["partition"].max() if len(s) else None}, last written {s["written_at"].max() if len(s) else None}')

# %% [markdown]
# ## 2. The RFQ log, every month, forced
#
# `track_rfq` takes every column of `muni_algo_trade_track`, so a forced re-pull of all ten months brings the settle date (and anything else
# added to the table since October 20) into every partition. `track_charges` rides along: it is built from every column whose name contains
# "charge", so the carry charge appears there too. About the size of the `strategy_output` backfill (four minutes for nine months on October 20).

# %%
cfg_rfq = mdp.Config(root=PIPELINE_ROOT, start_date=START, end_date=END, sources=('track_rfq', 'track_charges'))
print(mdp.plan(cfg_rfq, force=True).to_string(index=False))
report = mdp.pull(cfg_rfq, force=True)
print('RFQ log re-pull done in', report.get('elapsed_s'), 's')
print({k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in report.get('deephaven', {}).items()})

# %% [markdown]
# ## 3. October, every other source, forced
#
# The window is October 1 to October 30. The start date inside October means only the October partition of each source is touched; `force`
# overrides the "complete" reading. `track_static` is pulled once and is not here. If the desk's module pulls the MMD curve through its own
# function rather than as a source, run `mdp.pull_benchmark_curve(cfg_oct)` after this cell; the lab's error message names it.

# %%
cfg_oct = mdp.Config(root=PIPELINE_ROOT, start_date='2026-10-01', end_date=END,
                     sources=tuple(s for s in mdp.Config.sources if s not in mdp.STATIC_SOURCES and s not in ('track_rfq', 'track_charges')))
print('sources:', cfg_oct.sources)
print(mdp.plan(cfg_oct, force=True).to_string(index=False))
report = mdp.pull(cfg_oct, force=True)
print('October re-pull done in', report.get('elapsed_s'), 's')
for k, v in report.items():
    if isinstance(v, dict):
        print(k, {kk: {kkk: (len(vvv) if isinstance(vvv, list) else vvv) for kkk, vvv in vv.items()} if isinstance(vv, dict) else vv for kk, vv in v.items()})

# %% [markdown]
# ## 4. Rebuild the panel over the full window
#
# Step 2 and Step 3 (`research_panel_step3_yield.parquet`, the file the notebooks read) are rebuilt from January 1 to October 30 with the same
# settings as before (covered universe, point-in-time ratings). The build validates itself at the end.

# %%
cfg_build = mdp.Config(root=PIPELINE_ROOT, start_date=START, end_date=END, covered_only=True, pit_ratings=True)
report = mdp.build(cfg_build)
print('build done in', report.get('elapsed_s'), 's | validation ok:', {k: v.get('ok') for k, v in report.get('validation', {}).items() if isinstance(v, dict)})

# %% [markdown]
# ## 5. Checks before the notebook runs
#
# Read-only. For every source: the October partition's rows, window end and write time (the window end must now be October 31, the write time
# today). Then the two columns the notebook needs: the settle date on `track_rfq` (its name, how it reads as a date, and the share lying between
# the trade date and T+2 business days, which is what the notebook will keep), and the carry charge on `strategy_output` (its unit, by its
# agreement with the loading production applied after the optimizer). Then the last date in each request store and in the panel.

# %%
print('=== partitions written today ===')
for src in cfg_all.sources:
    s = cfg_all.store(src).summary()
    if s.empty:
        print(f'{src}: (none)'); continue
    s = s[s['partition'] >= '2026-10']
    print(f'{src}: ' + ' | '.join(f'{r.partition} rows {r.rows:,} window_end {str(r.window_end)[:10]} written {str(r.written_at)[:16]}' for r in s.itertuples()))

print('\n=== the settle date on track_rfq ===')
tr_cols = columns_like('track_rfq', 'settl')
print('settle-like columns:', tr_cols)
if tr_cols:
    col = tr_cols[0]
    tr = cfg_all.store('track_rfq').read(columns=['date', col], categorical=False)
    sd = pd.to_datetime(tr[col], errors='coerce'); d0 = pd.to_datetime(tr['date']).dt.normalize()
    lag = pd.Series(pd.to_numeric((sd.dt.normalize() - d0).dt.days, errors='coerce'))
    print(f'{col}: dtype {tr[col].dtype} | non-null {tr[col].notna().mean():.1%} | reads as a date {sd.notna().mean():.1%} | e.g. {tr[col].dropna().astype(str).head(3).tolist()}')
    print('calendar days from trade to settle (share):', lag.value_counts(normalize=True).sort_index().round(4).head(10).to_dict())
    print(f'before the trade date {float((lag < 0).mean()):.2%} | more than 4 calendar days after (beyond T+2 on most weeks) {float((lag > 4).mean()):.2%}')
else:
    print('NO settle column on track_rfq after the forced re-pull: the table does not carry it; the notebook prices every request at T+2 and says so')

print('\n=== the carry charge on strategy_output ===')
so_cols = columns_like('strategy_output', 'carry')
print('carry-like columns:', so_cols)
so = cfg_all.store('strategy_output').read(columns=['date', 'signalyield', 'optimalyield', 'finalquoteyield'] + so_cols, categorical=False)
adj = so['finalquoteyield'] - so['optimalyield']
for c in so_cols:
    x = pd.to_numeric(so[c], errors='coerce')
    if 'charge' in c.lower():
        print(f'{c}: non-null {x.notna().mean():.1%} | median {x.median():.6f} | p90 {x.quantile(0.9):.6f} | loading = carry if read as yield {float(((adj - x).abs() <= 1e-6).mean()):.1%}, as bp {float(((adj - x / 100.0).abs() <= 1e-6).mean()):.1%}')
    else:
        print(f'{c}: = signal + carry on {float(((x - so["signalyield"] - pd.to_numeric(so[so_cols[0]], errors="coerce")).abs() <= 1e-6).mean()):.1%}, = signal on {float(((x - so["signalyield"]).abs() <= 1e-9).mean()):.1%}')

print('\n=== last dates ===')
for src in ('strategy_output', 'track_rfq', 'msrb', 'wftrade'):
    f = cfg_all.store(src).read(columns=['date'] if src != 'wftrade' else None, categorical=False)
    if f is None or f.empty:
        print(f'{src}: empty'); continue
    dc = 'date' if 'date' in f.columns else next((c for c in f.columns if 'exec_date' in c or 'date' in c.lower()), None)
    print(f'{src}: {pd.to_datetime(f[dc]).max().date() if dc else "?"} ({len(f):,} rows)')
p3 = pd.read_parquet(cfg_all.path('research_panel_step3_yield.parquet'), columns=['date', 'cusip'])
print(f'panel step 3: {p3["date"].max().date()} ({len(p3):,} rows, {p3["cusip"].nunique():,} CUSIPs)')
print('\n=== validate() ===')
print({k: v.get('ok') for k, v in mdp.validate(cfg_all).items() if isinstance(v, dict)})

# %% [markdown]
# ## 6. Then
#
# 1. If production wrote a pfill snapshot after `pfills_20261008.json`, copy it to `prod_pfills/` and add it to `pfill_schedule` in the
#    width-charge notebook's config, with the last date of the previous file.
# 2. Run `muni_width_charge_research.ipynb` from the top. Every stage rebuilds on the new data fingerprints, about an hour.
# 3. Read its Section 5 audit lines first: the carry unit and match share, the settle sources, the cover and print audits, the vanilla pass.
#    Then Section 7's tangent table by call structure. Upload the HTML as before.
