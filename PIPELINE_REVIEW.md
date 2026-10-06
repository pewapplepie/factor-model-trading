# Data Pipeline Review and Rebuild: `datapipeline-pyversion` → `muni_data_pipeline.py`

Date: 2026-10-06
Reviewed: HackMD `AlgoTrading/pyfold/datapipeline-pyversion` (notebook exported to a `# %%` script, 1,808 lines)
Replacement: `muni_data_pipeline.py` in this repo (one importable module, also a CLI)
Target: backfill to 2026-01-01 and extend to 2026-09-30 in under one hour, resumable after any crash

> Tested here: the storage, planning, parsing, dedupe, Step 2 and Step 3 layers, on synthetic data (see "Verification"). Not tested here: the live Deephaven and OneTick calls, which only exist on the JupyterHub kernel. Section 6 lists the three API assumptions to confirm on the first dry run.

---

## 1. Why the current run takes ten hours and crashes

The data is not large. The yield panel the model uses is 1.2M rows. The ten hours come from moving and re-moving a 30M-row full-universe panel through the slowest possible path, several times, as a pickle of Python objects.

| # | Where | What happens | Cost |
|---|---|---|---|
| 1 | Deephaven panel pull | One `open_table('panel_slim').to_arrow().to_pandas()` of the whole window: 29.7M rows x 12 columns in a single gRPC snapshot | The step most likely to hang or OOM. There is no partial result if it dies. |
| 2 | Panel pull, twice | The "Product/RDI name enrichment" cell and the following "panel cache merge" cell **both** pull `panel_slim` and both rewrite the pickle | The heaviest step runs twice. |
| 3 | `panel_slim_cache.pkl` | 2.7 GB pickle. Read in full just to find the max date (cutoff cell), read again to merge, read again for the universe cell, read again for Step 2 | Four full deserialisations of 2.7 GB. Each inflates to 15 to 25 GB in memory because `cusip`, `state_code`, ratings and `long_comp_name` are Python object strings. |
| 4 | Cache merge | `pd.concat([old 30M, new 30M])` then `drop_duplicates(['date','cusip'], keep='last')` then `sort_values` on object strings | Minutes per pass, doubles peak memory. |
| 5 | OneTick pull | All 258k algo-universe CUSIPs passed as the symbol list, 25 sequential batches of ~10k symbols over the full window. Only ~57k of them exist in the closing-marks DB | 78% of symbol work returns nothing. Sequential. |
| 6 | OneTick result conversion | `pf()` builds one DataFrame per symbol with `Series.apply`, then `pd.concat([ret, pf(r)])` inside the batch loop | Quadratic concat; Python loop over 10k symbols per batch. |
| 7 | OneTick append | Read entire cache, `concat`, `drop_duplicates()` over **all columns** (hashes floats), rewrite | Slow and wrong: a re-pull with a different `PUBLISHTIME` creates a duplicate rather than replacing. |
| 8 | Step 2 build | Left join of the 30M-row panel to 1.2M marks, then `groupby('cusip')` diffs, lags and six per-date `rank_normalize` transforms on all 30M rows | 96% of those rows have no closing mark. Step 3 then throws them away. The `rank_*` columns are not consumed by the research notebook, which rank-normalises its own instruments. |
| 9 | Validation | `validate_source_cache('closing_marks')` sorts and dedupes the full raw marks cache just to report; Step 2 does it again | Duplicate work. |
| 10 | Incremental logic | `cutoff = max(cached_max − 1 day, start_date)` | **Backfill is impossible.** With a cache from April, setting `start_date='2026-01-01'` still pulls from September. January to March would never arrive. |
| 11 | Resume | Nothing is written until each stage finishes | A crash at hour nine loses everything. |

Items 1 to 4 explain the crash. Items 5 to 9 explain most of the wall clock. Item 10 blocks the thing you want to do next.

Smaller defects, fixed in passing:

- `normalize_yield_to_percent` is called inside the numeric-cast loop six times (harmless, but sloppy).
- `jupyter_tools.deephaven` is star-imported in three places, each of which can open a session.
- Three near-identical `_dh_cols` helpers inside server-side scripts.
- `drop_duplicates(keep='last')` on MSRB and AlgoSignal caches has no deterministic sort before it.
- The final two audit cells are copies of each other.

---

## 2. What the rebuild does differently

### 2.1 Partitioned, resumable storage

Every source is a directory of monthly parquet files plus a manifest:

```
data/
  panel/            2026-01.parquet ... 2026-09.parquet  _manifest.json
  closing_marks/    2026-01.parquet ... _manifest.json
  msrb/             ...
  algosignal_msrb/  ...
research_panel_step2.parquet
research_panel_step3_yield.parquet
universe_cusips_cache.parquet
product_name_cache.parquet
data_pipeline_manifest.json
pipeline.log
```

A partition is **complete** when it exists and was written after its window closed. `plan()` pulls only partitions that are missing or still open (the current month). Widening the window to January pulls January, February and March and nothing else. A crash loses at most one partition. Writes are atomic (temp file, then rename).

Strings are dictionary-encoded on disk and loaded as categoricals. The full-universe panel for nine months is roughly 45M rows and about 1 GB on disk. `load_panel(columns=..., start=..., end=..., cusips=...)` pushes all three filters into the parquet reader, so the research notebook never deserialises the whole thing again.

### 2.2 Deephaven: one script per partition, Arrow in, no pickles

- A prelude script runs once per session and builds the server-side tables that do not change across partitions: the algo universe, `Product.last_by(cusip)`, the latest ratings.
- Each partition is one short script: filter `ICEEval` to `[start, end)` **before** `where_in` and `last_by`, project to the needed columns, then two `natural_join`s. The result is fetched as one Arrow table of roughly 5M rows, normalised, and written. The server variable is released immediately.
- Filtering by date first and projecting before the joins is the main server-side saving; the old code did `update_view` then `where_in` on the full history, then joined every column.
- **Point-in-time ratings** (`pit_ratings=True`, default): the rating join for partition `[a, b)` uses `muni_algo_trade_track.where(date < b).last_by(cusip)` when the track table has a `date` column, so the rating is as of the partition end rather than today. This is the cheap fix for the "latest composite rating" leakage noted in every review. It falls back to latest if the table has no date.
- MSRB and AlgoSignal follow the same per-partition pattern.

### 2.3 OneTick: discover symbols, then (partition × batch) on a thread pool

- `FindDbSymbols` on the closing-marks DB returns the symbols that actually exist (about 57k), optionally intersected with the universe. The 258k-symbol list is gone.
- Tasks are (month, batch of 2,000 symbols). With 57k symbols and 9 months that is about 29 × 9 = 261 tasks on 6 worker threads. Each task's result is normalised and buffered; a month is written as soon as its last batch lands.
- Result conversion is one `pd.concat` per task, not a quadratic loop.
- Dedup to one row per `(CUSIP, DATE)` happens at **read** time with a deterministic order (`SOURCESYSTEM, PUBLISHTIME, OMDSEQ`, keep last), so a re-pull replaces rather than duplicates.
- `apply_times_daily`, `batch_size`, `Presort` concurrency, timezone and context are config fields, defaulting to what the old code used.

### 2.4 Both sources at once

Deephaven and OneTick are different servers. `pull()` runs the Deephaven partition loop and the OneTick pool concurrently. Local CPU is not the bottleneck for either; the wall clock is the slower of the two, not the sum.

### 2.5 Step 2 / Step 3 on the covered population

`build_step2` joins the panel to the deduplicated marks **inner** on the OneTick-covered CUSIPs by default, which is the only population the yield-space model uses. That is about 1.5M rows for nine months instead of 45M, and the lag features take seconds. The ranked `rank_*` columns are dropped since nothing consumes them. `--full-universe` keeps a left-join Step 2 written partition by partition for anyone who needs it. Step 3 is the same logic as before with typed columns and a stable sort.

### 2.6 Robustness

- Retries with exponential backoff on every remote call (3 attempts, 5s base).
- Structured logging to stdout and `pipeline.log` with thread names, so a slow run shows which partition is running.
- Validation gates in the manifest: missing partitions, empty partitions whose window has started, dedupe audit.
- `--dry-run` prints the plan and the OneTick task count without pulling.

---

## 3. Expected timing for 2026-01-01 to 2026-09-30

These are estimates, not measurements. They assume the Deephaven server does the heavy `last_by` work at roughly the rate the old single pull achieved per row, and that OneTick serves 2,000-symbol monthly queries in tens of seconds.

| Stage | Old | New (estimate) | Why |
|---|---|---|---|
| Deephaven panel | hours, often crashes | 9 partitions × 2 to 4 min ≈ 20 to 35 min | Date filter first, projection before joins, 5M-row Arrow fetches |
| OneTick closing marks | 1 to 2 h sequential | 261 tasks / 6 workers ≈ 10 to 20 min, overlapping with Deephaven | Real symbol list, parallel, no quadratic concat |
| MSRB + AlgoSignal | 20 to 40 min | 5 to 10 min, in the Deephaven loop | Per-partition scripts |
| Cache merge | 30 to 60 min | 0 | No merge; partitions are the cache |
| Step 2 + Step 3 | 1 to 2 h | under 2 min | 1.5M covered rows, categoricals |
| **Total** | **10 h+** | **35 to 50 min**, bounded by the Deephaven loop | |

If the Deephaven loop is the long pole, two things cut it further: run two partitions concurrently from two sessions, or restrict the panel pull to the covered CUSIP set after the first OneTick month lands (`load_panel(cusips=...)` already supports it on the read side; a `panel_universe='covered'` option on the pull side is a ten-line change).

---

## 4. How to run it on JupyterHub

Copy `muni_data_pipeline.py` next to the notebooks. In a kernel that has Deephaven and OneTick:

```python
import muni_data_pipeline as mdp

cfg = mdp.Config(
    root='.',
    start_date='2026-01-01',
    end_date='2026-09-30',
    ot_workers=6,            # OneTick threads
    pit_ratings=True,        # rating as of partition end
    covered_only=True,       # Step 2/3 on the OneTick-covered rows
)

# 1. Look before you pull: which partitions, how many OneTick tasks
print(mdp.plan(cfg).to_string(index=False))
mdp.pull(cfg, dry_run=True)

# 2. Pull both sources concurrently (resumable; rerun after any failure)
report = mdp.pull(cfg)

# 3. Build Step 2 / Step 3 and write the manifest
build = mdp.build(cfg)
```

Or from a terminal on the same host:

```bash
python muni_data_pipeline.py plan  --root . --start 2026-01-01 --end 2026-09-30
python muni_data_pipeline.py pull  --root . --start 2026-01-01 --end 2026-09-30 --dry-run
python muni_data_pipeline.py all   --root . --start 2026-01-01 --end 2026-09-30
```

Daily refresh is the same command with the same window. Only the current month is re-pulled. Extending `end_date` into October adds one partition.

### Research notebook changes

Two reads change. Everything else is untouched.

```python
# before
panel = pd.read_pickle('panel_slim_cache.pkl')
# after: column, date and cusip pushdown; categoricals in memory
panel = mdp.load_panel(cfg, columns=['date', 'cusip', 'price', 'mkt_yield'], start='2026-04-01')

# Step 3 panel path is unchanged
step3 = pd.read_parquet('research_panel_step3_yield.parquet')
```

If something still needs the pickle, `mdp.export_legacy_pickle(cfg)` or `--legacy-pickle` writes it, but that reintroduces the 2.7 GB object-dtype problem and should be a stopgap.

---

## 5. Verification done here

Synthetic end-to-end run on the local layers (3,000 CUSIPs, 600 covered, Jan to mid-March, duplicate marks from a second source system, yields in decimal units, `DATE` as integer `YYYYMMDD`):

| Check | Result |
|---|---|
| Monthly partitions and half-open windows | correct, last partition ends the day after `end_date` |
| Completed vs open partition logic | closed windows complete; a partition whose window includes today is re-pulled |
| Backfill plan when `start_date` moves earlier | only the new months are planned |
| `load_panel` pushdown by date and cusip | 10 CUSIPs × February only, categorical dtype |
| Closing-mark dedupe | one row per (CUSIP, DATE); higher-precedence source kept; yields scaled to percent |
| Step 2 inner (covered) and full left-join paths | row counts match expectations |
| Step 3 | per-CUSIP dates monotone; `closing_yield_change_bp` equals 100 × diff to 1e-9 |
| CLI `plan` | runs |

---

## 6. Assumptions to confirm on the first live dry run

1. **`where_in` with a renamed column.** The prelude builds `mdp_universe_cins = universe.view(['cusip_cins = cusip'])` and filters `ICEEval.where_in(mdp_universe_cins, cols=['cusip_cins'])`. If your Deephaven version wants the `'A = B'` match form instead, change that one line.
2. **`FindDbSymbols` output.** The code strips the `MUNI_CLOSING_MARKS_DB::` prefix from `SYMBOL_NAME`. If symbols come back bare, the strip is a no-op.
3. **`otq.run` from worker threads.** The OneTick Python client is normally safe to call concurrently. If you see server-side throttling, lower `ot_workers` to 3; if a single `otq.run` is not thread-safe in your build, set `ot_workers=1` and the design still works, just slower.
4. **Session calls off the main thread.** `pull()` runs the Deephaven loop in a worker thread so OneTick can run alongside. If `jupyter_tools.deephaven` objects to that, pass `parallel_sources=False` and the two sources run back to back.
5. **`muni_algo_trade_track` has a `date` column.** Needed for point-in-time ratings; the prelude detects it and falls back to latest ratings if absent.

---

## 7. What was deliberately left out

- No DuckDB or Polars dependency. Pyarrow dataset pushdown plus categoricals is enough at this size and is already on the kernel.
- No per-date rank normalisation in Step 2. The research notebook does its own on its own instrument set.
- No ProductRDI name backfill round trip. Names come from the panel pull; the old code had this off by default because it hung the session.
- No scheduler. The script is idempotent, so cron or a notebook cell on a timer is enough.
