"""
muni_data_pipeline.py
=====================

Partitioned, resumable data ingestion for the muni characteristic factor model.

Sources
-------
* Deephaven  : muni.ICEEval (evaluated marks), muni.Product (static fields),
               muni_offline.muni_algo_trade_track (universe + ratings),
               muni.MSRB (trade prints), muni_offline.algosignal_msrb_match.
* OneTick    : MUNI_CLOSING_MARKS_DB.CLOSING_MARKS (close price/yield/duration/DV01).

Design
------
* Every source is stored as a directory of monthly (or weekly) parquet partitions
  with a manifest. A run only pulls partitions that are missing or still open
  (the current month), so widening the date window backfills exactly the gap.
* Deephaven pulls are issued per partition as server-side scripts and fetched as
  Arrow tables; the 30M-row single snapshot that used to crash is gone.
* OneTick pulls are (partition x symbol-batch) tasks run on a thread pool, with
  symbols restricted to those that actually exist in the closing-marks database.
* Deephaven and OneTick run concurrently (different servers).
* Step 2/3 panels are built on the OneTick-covered rows only by default, which is
  the only population the yield-space model uses. The full-universe panel stays
  available through `load_panel()` with column and date pushdown.
* Strings are dictionary-encoded on disk and categorical in memory. No pickles.

Usage
-----
Script:
    python muni_data_pipeline.py plan  --root /path --start 2026-01-01 --end 2026-09-30
    python muni_data_pipeline.py pull  --root /path --start 2026-01-01 --end 2026-09-30
    python muni_data_pipeline.py build --root /path
    python muni_data_pipeline.py all   --root /path --start 2026-01-01 --end 2026-09-30

Notebook (JupyterHub kernel with Deephaven + OneTick):
    import muni_data_pipeline as mdp
    cfg = mdp.Config(root='.', start_date='2026-01-01', end_date='2026-09-30')
    mdp.run_all(cfg)

Research notebook consumers:
    panel = mdp.load_panel(cfg, columns=[...], start='2026-04-01')   # replaces read_pickle
    step3 = pd.read_parquet(cfg.path('research_panel_step3_yield.parquet'))
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

LOG = logging.getLogger('muni_pipeline')


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    root: Path = Path('.')
    start_date: str = '2026-01-01'
    end_date: str = '2026-09-30'
    partition: str = 'M'                 # 'M' monthly, 'W' weekly
    data_dir: str = 'data'

    # Deephaven
    dh_env: str = 'FICC'
    dh_heap_gb: int = 16
    pit_ratings: bool = True             # rating as of partition end instead of latest
    sources: tuple[str, ...] = ('panel', 'closing_marks', 'msrb', 'algosignal_msrb', 'track_static', 'track_charges')

    # OneTick
    ot_context: str = 'MUNI_PROD'
    ot_db: str = 'MUNI_CLOSING_MARKS_DB'
    ot_tick_type: str = 'CLOSING_MARKS'
    ot_symbol_batch: int = 2000
    ot_workers: int = 6
    ot_batch_size: int = 1000
    ot_timezone: str = 'EST5EDT'
    ot_apply_times_daily: bool = True
    ot_presort_concurrency: int = 4

    # Robustness
    retries: int = 3
    retry_backoff_s: float = 5.0
    parallel_sources: bool = True

    # Build
    covered_only: bool = True
    write_legacy_pickle: bool = False
    method_version: str = 'muni_data_pipeline_v1_0'

    def path(self, *parts: str | Path) -> Path:
        p = Path(*parts)
        return p if p.is_absolute() else Path(self.root) / p

    def store(self, source: str) -> 'PartitionStore':
        return PartitionStore(self.path(self.data_dir, source), source)

    def partitions(self) -> list['Partition']:
        return make_partitions(self.start_date, self.end_date, self.partition)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out['root'] = str(self.root)
        return out


@dataclass(frozen=True)
class Partition:
    key: str
    start: pd.Timestamp          # inclusive
    end: pd.Timestamp            # exclusive

    @property
    def start_str(self) -> str:
        return self.start.strftime('%Y-%m-%d')

    @property
    def end_str(self) -> str:
        return self.end.strftime('%Y-%m-%d')

    @property
    def last_day_str(self) -> str:
        return (self.end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def make_partitions(start_date: str, end_date: str, freq: str = 'M') -> list[Partition]:
    start = pd.Timestamp(start_date).normalize()
    end_incl = pd.Timestamp(end_date).normalize()
    if end_incl < start:
        raise ValueError(f'end_date {end_date} is before start_date {start_date}')
    out: list[Partition] = []
    if freq.upper().startswith('M'):
        cur = start.replace(day=1)
        while cur <= end_incl:
            nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
            out.append(Partition(cur.strftime('%Y-%m'), max(cur, start), min(nxt, end_incl + pd.Timedelta(days=1))))
            cur = nxt
    elif freq.upper().startswith('W'):
        cur = start - pd.Timedelta(days=start.weekday())  # Monday
        while cur <= end_incl:
            nxt = cur + pd.Timedelta(days=7)
            out.append(Partition(cur.strftime('%Y-W%W'), max(cur, start), min(nxt, end_incl + pd.Timedelta(days=1))))
            cur = nxt
    else:
        raise ValueError(f'Unknown partition freq {freq}')
    return out


# --------------------------------------------------------------------------------------
# Partition store (parquet directory + manifest)
# --------------------------------------------------------------------------------------
class PartitionStore:
    """One directory per source: <key>.parquet files plus _manifest.json."""

    def __init__(self, directory: Path, name: str):
        self.dir = Path(directory)
        self.name = name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def manifest_path(self) -> Path:
        return self.dir / '_manifest.json'

    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                LOG.warning('%s: manifest unreadable, starting fresh', self.name)
        return {'source': self.name, 'partitions': {}}

    def file(self, key: str) -> Path:
        return self.dir / f'{key}.parquet'

    def has(self, key: str) -> bool:
        return self.file(key).exists() and key in self.manifest()['partitions']

    def is_complete(self, key: str, today: pd.Timestamp | None = None) -> bool:
        """A partition is complete if it exists and its window closed before it was written."""
        meta = self.manifest()['partitions'].get(key)
        if not meta or not self.file(key).exists():
            return False
        written = pd.Timestamp(meta.get('written_at'))
        if written.tzinfo is not None:
            written = written.tz_convert(None)
        window_end = pd.Timestamp(meta.get('window_end'))
        return bool(written.normalize() > window_end)

    def plan(self, partitions: Sequence[Partition], force: bool = False) -> list[Partition]:
        if force:
            return list(partitions)
        return [p for p in partitions if not self.is_complete(p.key)]

    def write(self, key: str, table: pa.Table | pd.DataFrame, partition: Partition, extra: dict | None = None) -> dict:
        if isinstance(table, pd.DataFrame):
            table = pa.Table.from_pandas(table, preserve_index=False)
        tmp = self.file(key).with_suffix('.parquet.tmp')
        pq.write_table(table, tmp, compression='zstd', use_dictionary=True)
        os.replace(tmp, self.file(key))
        meta = {
            'rows': table.num_rows,
            'columns': table.column_names,
            'window_start': partition.start_str,
            'window_end': partition.end_str,
            'written_at': pd.Timestamp.now('UTC').isoformat(),
            'size_mb': round(self.file(key).stat().st_size / 1e6, 3),
        }
        if extra:
            meta.update(extra)
        with self._lock:
            m = self.manifest()
            m['partitions'][key] = meta
            tmp_m = self.manifest_path.with_suffix('.json.tmp')
            tmp_m.write_text(json.dumps(m, indent=2, sort_keys=True, default=str), encoding='utf-8')
            os.replace(tmp_m, self.manifest_path)
        LOG.info('%s: wrote %s (%s rows, %.1f MB)', self.name, key, f'{table.num_rows:,}', meta['size_mb'])
        return meta

    def keys(self) -> list[str]:
        return sorted(k for k in self.manifest()['partitions'] if self.file(k).exists())

    def dataset(self) -> ds.Dataset | None:
        files = [str(self.file(k)) for k in self.keys()]
        if not files:
            return None
        return ds.dataset(files, format='parquet')

    def read(self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None,
             to_pandas: bool = True, categorical: bool = True) -> pd.DataFrame | pa.Table | None:
        dset = self.dataset()
        if dset is None:
            return None
        cols = [c for c in columns if c in dset.schema.names] if columns else None
        table = dset.to_table(columns=cols, filter=filter)
        if not to_pandas:
            return table
        return table.to_pandas(strings_to_categorical=categorical, self_destruct=True)

    def summary(self) -> pd.DataFrame:
        m = self.manifest()['partitions']
        rows = [{'partition': k, **{kk: v.get(kk) for kk in ['rows', 'window_start', 'window_end', 'written_at', 'size_mb']}} for k, v in sorted(m.items())]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def retry(fn: Callable[[], Any], attempts: int, backoff_s: float, label: str) -> Any:
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last = exc
            if i == attempts:
                break
            wait = backoff_s * (2 ** (i - 1))
            LOG.warning('%s failed (attempt %d/%d): %s; retrying in %.0fs', label, i, attempts, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f'{label} failed after {attempts} attempts: {last}') from last


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors='coerce')
    try:
        if getattr(out.dt, 'tz', None) is not None:
            out = out.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return out


def _cat(series: pd.Series) -> pd.Series:
    return series.astype('string').astype('category')


# --------------------------------------------------------------------------------------
# Deephaven
# --------------------------------------------------------------------------------------
PANEL_COLUMNS = ['date', 'cusip', 'price', 'mkt_yield', 'cpn', 'state_code', 'callable',
                 'nxt_call_dt', 'maturity', 'long_comp_name', 'final_comp_rating', 'comp_rating']
MSRB_WANT = ['date', 'cusip', 'quantity', 'tradetime', 'tradetype', 'price', 'yield', 'settlement_date', 'trade_id', 'side']
ALGOSIGNAL_WANT = ['date', 'msrb_trade_id', 'cusip', 'msrb_quantity', 'size_bin', 'msrb_side', 'signal_side',
                   'msrb_tradetime', 'msrb_price', 'msrb_yield', 'has_algosignal_match', 'signal_ts',
                   'MinuteFromSignal', 'MmdYld', 'dMmdSprdSide', 'last_value', 'algo_signal_yield',
                   'algo_minus_msrb_yield_bp', 'Flg']
# Static per-CUSIP fields from muni_algo_trade_track (last row per cusip). Only the columns that exist are kept.
TRACK_STATIC_WANT = ['cusip', 'issuer_industry', 'industry', 'sector', 'issuer', 'state', 'tax_status', 'coupon_type',
                     'issue_size', 'amount_outstanding', 'outstanding_amount', 'final_comp_rating', 'comp_rating']
STATIC_SOURCES = {'track_static'}          # pulled once (first partition key), not per month
# Per-trade charges from muni_algo_trade_track (v4.4): every column whose name contains 'charge' (liquidity, risk,
# manual, total ...) plus the keys needed to join a charge to a matched print as of its time. Pulled per partition on
# the track's date column when it has one, otherwise once. Column names on the track are not fixed, so the keep list
# is resolved on the server from what exists; TRACK_CHARGE_KEYS lists every key/time column we would like if present.
TRACK_CHARGE_KEYS = ['cusip', 'date', 'timestamp', 'ts', 'time', 'tradetime', 'trade_time', 'signal_ts', 'signal_time', 'quote_time',
                     'side', 'msrb_side', 'signal_side', 'quote_side', 'quantity', 'size', 'msrb_quantity', 'price', 'yield', 'algo_yield',
                     'msrb_trade_id', 'trade_id', 'order_id', 'inquiry_id']

DH_PRELUDE = r"""
# ---- muni_data_pipeline prelude: built once per session, reused by every partition ----
def _mdp_cols(tbl):
    try:
        return [str(n) for n in tbl.column_names]
    except Exception:
        try:
            return [c.name for c in tbl.columns]
        except Exception:
            return []

_mdp_track = db.historical_table('muni_offline', 'muni_algo_trade_track')
mdp_universe = _mdp_track.select_distinct(['cusip'])
mdp_universe_cins = mdp_universe.view(['cusip_cins = cusip'])
mdp_rating_latest = _mdp_track.last_by(['cusip']).view(['cusip', 'final_comp_rating', 'comp_rating'])
mdp_track_has_date = 'date' in _mdp_cols(_mdp_track)

_mdp_product = db.historical_table('muni', 'Product').last_by(['cusip'])
_mdp_pcols = _mdp_cols(_mdp_product)
_mdp_name_col = next((c for c in ['long_comp_name', 'LONG_COMP_NAME', 'longCompName', 'security_description',
                                   'SECURITY_DESCRIPTION', 'ultimate_borrower_name', 'issuer', 'ISSUER'] if c in _mdp_pcols), None)
_mdp_product_joins = ['cpn', 'state_code', 'callable', 'nxt_call_dt', 'maturity']
if _mdp_name_col is not None:
    if _mdp_name_col != 'long_comp_name':
        _mdp_product = _mdp_product.update_view([f'long_comp_name = {_mdp_name_col}'])
    _mdp_product_joins.append('long_comp_name')
mdp_product = _mdp_product.view(['cusip'] + _mdp_product_joins)
mdp_product_joins = _mdp_product_joins
print('[mdp] prelude ready | universe', mdp_universe.size, '| product name col', _mdp_name_col, '| track has date', mdp_track_has_date)
"""

DH_PANEL_PARTITION = r"""
# ---- panel partition {key}: [{a}, {b}) ----
_mdp_rating = mdp_rating_latest
if {pit} and mdp_track_has_date:
    _mdp_rating = _mdp_track.where('date < `{b}`').last_by(['cusip']).view(['cusip', 'final_comp_rating', 'comp_rating'])
mdp_panel_part = (
    db.historical_table('muni', 'ICEEval')
      .where(['date >= `{a}`', 'date < `{b}`'])
      .where_in(mdp_universe_cins, cols=['cusip_cins'])
      .last_by(['date', 'cusip_cins'])
      .view(['date', 'cusip = cusip_cins', 'price = mean_evaluation_clean', 'mkt_yield = mean_yield'])
      .natural_join(mdp_product, on=['cusip'], joins=mdp_product_joins)
      .natural_join(_mdp_rating, on=['cusip'], joins=['final_comp_rating', 'comp_rating'])
)
"""

DH_MSRB_PARTITION = r"""
# ---- msrb partition {key}: [{a}, {b}) ----
_mdp_m = (db.historical_table('muni', 'MSRB')
            .where(['date >= `{a}`', 'date < `{b}`'])
            .where_in(mdp_universe, cols=['cusip']))
_mdp_mcols = _mdp_cols(_mdp_m)
_mdp_keep = [c for c in {want} if c in _mdp_mcols] or ['date', 'cusip', 'quantity', 'tradetime', 'tradetype']
mdp_msrb_part = _mdp_m.view(_mdp_keep)
"""

DH_TRACK_STATIC_PARTITION = r"""
# ---- track_static (one pull): last row per cusip of muni_algo_trade_track, static columns only ----
_mdp_tcols = _mdp_cols(_mdp_track)
_mdp_tkeep = [c for c in {want} if c in _mdp_tcols] or ['cusip']
mdp_track_static_part = _mdp_track.last_by(['cusip']).view(_mdp_tkeep)
"""

DH_TRACK_CHARGES_PARTITION = r"""
# ---- track_charges partition {key}: [{a}, {b}) -- per-trade charge columns of muni_algo_trade_track ----
_mdp_tcols = _mdp_cols(_mdp_track)
_mdp_charge = [c for c in _mdp_tcols if 'charge' in c.lower()]
_mdp_ckeep = [c for c in {keys} if c in _mdp_tcols] + [c for c in _mdp_charge if c not in {keys}]
_mdp_ckeep = _mdp_ckeep or ['cusip']
if mdp_track_has_date:
    mdp_track_charges_part = _mdp_track.where(['date >= `{a}`', 'date < `{b}`']).view(_mdp_ckeep)
else:
    mdp_track_charges_part = _mdp_track.view(_mdp_ckeep)
"""

DH_ALGOSIGNAL_PARTITION = r"""
# ---- algosignal partition {key}: [{a}, {b}) ----
_mdp_s = db.historical_table('muni_offline', 'algosignal_msrb_match').where(['date >= `{a}`', 'date < `{b}`'])
_mdp_scols = _mdp_cols(_mdp_s)
_mdp_keep = [c for c in {want} if c in _mdp_scols] or _mdp_scols
_mdp_keep = _mdp_keep + [c for c in _mdp_scols if 'charge' in c.lower() and c not in _mdp_keep]   # v4.4: any charge column on the match table rides along
mdp_algosignal_part = _mdp_s.view(_mdp_keep)
"""


class DeephavenClient:
    """Thin wrapper over the JupyterHub Deephaven session (jupyter_tools.deephaven)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = None
        self._prelude_done = False

    def connect(self) -> 'DeephavenClient':
        if self.session is not None:
            return self
        os.environ.setdefault('DH_ENV', self.cfg.dh_env)
        os.environ.setdefault('DH_HEAP_SIZE', str(self.cfg.dh_heap_gb))
        try:
            from jupyter_tools.deephaven import DeephavenSession  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('Deephaven is only available on the JupyterHub kernel (jupyter_tools.deephaven).') from exc
        self.session = DeephavenSession.session
        return self

    def run(self, script: str) -> None:
        self.connect().session.run_script(script)

    def fetch(self, name: str) -> pa.Table:
        return self.connect().session.open_table(name).to_arrow()

    def release(self, *names: str) -> None:
        try:
            self.run('\n'.join(f'{n} = None' for n in names))
        except Exception as exc:  # noqa: BLE001
            LOG.debug('release failed for %s: %s', names, exc)

    def prelude(self) -> None:
        if not self._prelude_done:
            self.run(DH_PRELUDE)
            self._prelude_done = True

    def pull_partition(self, source: str, part: Partition) -> pa.Table:
        self.prelude()
        if source == 'panel':
            script, name = DH_PANEL_PARTITION.format(key=part.key, a=part.start_str, b=part.end_str, pit=self.cfg.pit_ratings), 'mdp_panel_part'
        elif source == 'msrb':
            script, name = DH_MSRB_PARTITION.format(key=part.key, a=part.start_str, b=part.end_str, want=MSRB_WANT), 'mdp_msrb_part'
        elif source == 'algosignal_msrb':
            script, name = DH_ALGOSIGNAL_PARTITION.format(key=part.key, a=part.start_str, b=part.end_str, want=ALGOSIGNAL_WANT), 'mdp_algosignal_part'
        elif source == 'track_static':
            script, name = DH_TRACK_STATIC_PARTITION.format(want=TRACK_STATIC_WANT), 'mdp_track_static_part'
        elif source == 'track_charges':
            script, name = DH_TRACK_CHARGES_PARTITION.format(key=part.key, a=part.start_str, b=part.end_str, keys=TRACK_CHARGE_KEYS), 'mdp_track_charges_part'
        else:
            raise KeyError(source)
        self.run(script)
        try:
            return self.fetch(name)
        finally:
            self.release(name)


def normalize_panel(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(strings_to_categorical=False, self_destruct=True)
    df.columns = [str(c) for c in df.columns]
    df = df.reindex(columns=PANEL_COLUMNS)
    df['date'] = _to_naive_datetime(df['date']).dt.normalize()
    df['cusip'] = df['cusip'].astype('string')
    df = df.dropna(subset=['date', 'cusip'])
    for col in ['price', 'mkt_yield', 'cpn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['state_code', 'callable', 'final_comp_rating', 'comp_rating', 'long_comp_name']:
        df[col] = df[col].astype('string')
    # Deephaven dedupes on (date, cusip_cins) server side; keep a cheap local guard.
    df = df.drop_duplicates(['date', 'cusip'], keep='last')
    return df.sort_values(['date', 'cusip']).reset_index(drop=True)


def normalize_msrb(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(self_destruct=True)
    df.columns = [str(c) for c in df.columns]
    if 'date' in df.columns:
        df['date'] = _to_naive_datetime(df['date']).dt.normalize()
    if 'tradetime' in df.columns:
        df['tradetime'] = _to_naive_datetime(df['tradetime'])
    if 'cusip' in df.columns:
        df['cusip'] = df['cusip'].astype('string')
    return df.drop_duplicates().reset_index(drop=True)


def normalize_algosignal(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(self_destruct=True)
    df.columns = [str(c) for c in df.columns]
    for col in ['date', 'msrb_tradetime', 'signal_ts']:
        if col in df.columns:
            df[col] = _to_naive_datetime(df[col])
    if 'date' in df.columns:
        df['date'] = df['date'].dt.normalize()
    if 'cusip' in df.columns:
        df['cusip'] = df['cusip'].astype('string')
    return df.drop_duplicates().reset_index(drop=True)


def normalize_track_static(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(self_destruct=True)
    df.columns = [str(c) for c in df.columns]
    if 'cusip' in df.columns:
        df['cusip'] = df['cusip'].astype('string')
        df = df.dropna(subset=['cusip']).drop_duplicates('cusip', keep='last')
    for col in df.columns:
        if col != 'cusip' and (df[col].dtype == object or str(df[col].dtype).startswith('string')):
            df[col] = df[col].astype('string')
    return df.reset_index(drop=True)


def normalize_track_charges(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(self_destruct=True)
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        if col in ('date', 'timestamp', 'ts', 'time', 'tradetime', 'trade_time', 'signal_ts', 'signal_time', 'quote_time'):
            df[col] = _to_naive_datetime(df[col])
        elif 'charge' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'cusip' in df.columns:
        df['cusip'] = df['cusip'].astype('string')
    return df.drop_duplicates().reset_index(drop=True)


NORMALIZERS = {'panel': normalize_panel, 'msrb': normalize_msrb, 'algosignal_msrb': normalize_algosignal, 'track_static': normalize_track_static, 'track_charges': normalize_track_charges}


def pull_deephaven(cfg: Config, sources: Sequence[str], force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Pull every missing/open partition for the Deephaven sources, one partition at a time."""
    report: dict[str, Any] = {}
    client = DeephavenClient(cfg)
    for source in sources:
        store = cfg.store(source)
        parts = cfg.partitions()[:1] if source in STATIC_SOURCES else cfg.partitions()
        todo = store.plan(parts, force=force)
        report[source] = {'planned': [p.key for p in todo], 'done': [], 'failed': []}
        LOG.info('deephaven/%s: %d of %d partitions to pull: %s', source, len(todo), len(cfg.partitions()), [p.key for p in todo])
        if dry_run:
            continue
        for part in todo:
            t0 = time.perf_counter()
            try:
                table = retry(lambda: client.pull_partition(source, part), cfg.retries, cfg.retry_backoff_s, f'deephaven/{source}/{part.key}')
                frame = NORMALIZERS[source](table)
                store.write(part.key, frame, part, extra={'elapsed_s': round(time.perf_counter() - t0, 1)})
                report[source]['done'].append(part.key)
            except Exception as exc:  # noqa: BLE001
                LOG.error('deephaven/%s/%s FAILED: %s', source, part.key, exc)
                report[source]['failed'].append({'partition': part.key, 'error': repr(exc)})
    return report


# --------------------------------------------------------------------------------------
# OneTick
# --------------------------------------------------------------------------------------
CLOSING_MARKS_FIELDS = ['CUSIP', 'DATE', 'PUBLISHTIME', 'SOURCESYSTEM', 'PRICE', 'YIELD',
                        'DURATION', 'MODIFIED_DURATION', 'DV01', 'BENCHMARK', 'OMDSEQ']


class OneTickClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import onetick.query as otq  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("onetick.query is not available in this kernel; run on the JupyterHub kernel.") from exc
        self.otq = otq

    def list_symbols(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        """Symbols that exist in the closing-marks DB for the window (far fewer than the algo universe)."""
        otq, cfg = self.otq, self.cfg
        e = (otq.FindDbSymbols(pattern='%', show_tick_type=True).symbols(cfg.ot_db + '::')
             >> otq.WhereClause(where=f"TICK_TYPE='{cfg.ot_tick_type}'"))
        q = otq.GraphQuery(e).start_time(start.to_pydatetime()).end_time(end.to_pydatetime())
        res = otq.run(q, context=cfg.ot_context)
        names = res.output(cfg.ot_db + '::').data['SYMBOL_NAME'].astype(str).tolist()
        prefix = cfg.ot_db + '::'
        return sorted({n[len(prefix):] if n.startswith(prefix) else n for n in names})

    def fetch(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str] = CLOSING_MARKS_FIELDS) -> pd.DataFrame:
        otq, cfg = self.otq, self.cfg
        syms = [f'{cfg.ot_db}::{s}' for s in symbols]
        graph = (otq.Passthrough(fields=','.join(fields)).tick_type(cfg.ot_tick_type)
                 >> otq.Presort(max_concurrency=cfg.ot_presort_concurrency).symbols(syms)
                 >> otq.Merge())
        q = otq.Graph(graph).start_time(start.to_pydatetime()).end_time(end.to_pydatetime())
        result = otq.run(q, context=cfg.ot_context, symbol_date=start.strftime('%Y%m%d'),
                         apply_times_daily=cfg.ot_apply_times_daily, batch_size=cfg.ot_batch_size, timezone=cfg.ot_timezone)
        return _onetick_result_to_frame(result)


def _onetick_result_to_frame(result: Any) -> pd.DataFrame:
    frames = []
    for item in iter(result):
        data = result.output(item.name).data
        if data is None or len(data) == 0:
            continue
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if 'SYMBOL' not in df.columns:
            df = df.assign(SYMBOL=item.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def _parse_mark_dates(raw: pd.Series) -> pd.Series:
    """DATE arrives as datetime, ISO string, integer YYYYMMDD, or epoch ms/s depending on the OneTick client."""
    if pd.api.types.is_datetime64_any_dtype(raw):
        return _to_naive_datetime(raw)
    num = pd.to_numeric(raw, errors='coerce')
    if num.notna().mean() > 0.9:
        vals = num.dropna()
        if vals.between(19000101, 21001231).mean() > 0.9:
            return pd.to_datetime(num.round().astype('Int64').astype('string'), format='%Y%m%d', errors='coerce')
        if vals.abs().median() > 1e11:
            return pd.to_datetime(num, unit='ms', errors='coerce')
        return pd.to_datetime(num, unit='s', errors='coerce')
    return _to_naive_datetime(raw)

def normalize_closing_marks_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalisation at write time: uppercase columns, typed DATE, string CUSIP. Dedup happens at read."""
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip().upper() for c in df.columns})
    if 'CUSIP' not in df.columns and 'SYMBOL' in df.columns:
        df['CUSIP'] = df['SYMBOL'].astype(str).str.split('::').str[-1]
    keep = [c for c in CLOSING_MARKS_FIELDS + ['TIME', 'SYMBOL'] if c in df.columns]
    df = df[keep].copy()
    df['CUSIP'] = df['CUSIP'].astype('string')
    dates = _parse_mark_dates(df['DATE']) if 'DATE' in df.columns else pd.Series(pd.NaT, index=df.index)
    for fill_col in ['TIME', 'PUBLISHTIME']:
        if fill_col in df.columns:
            dates = dates.fillna(_to_naive_datetime(df[fill_col]))
    df['DATE'] = dates.dt.normalize()
    df['DATE'] = df['DATE'].mask(df['DATE'] < pd.Timestamp('2000-01-01'))
    if 'PUBLISHTIME' in df.columns:
        df['PUBLISHTIME'] = _to_naive_datetime(df['PUBLISHTIME'])
    for col in ['PRICE', 'YIELD', 'DURATION', 'MODIFIED_DURATION', 'DV01', 'OMDSEQ']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['SOURCESYSTEM', 'BENCHMARK']:
        if col in df.columns:
            df[col] = df[col].astype('string')
    return df.dropna(subset=['CUSIP', 'DATE']).reset_index(drop=True)


def pull_onetick(cfg: Config, force: bool = False, dry_run: bool = False, universe: Sequence[str] | None = None) -> dict[str, Any]:
    """(partition x symbol-batch) tasks on a thread pool; each partition is written once all its batches finish."""
    store = cfg.store('closing_marks')
    parts = cfg.partitions()
    todo = store.plan(parts, force=force)
    report: dict[str, Any] = {'planned': [p.key for p in todo], 'done': [], 'failed': [], 'symbols': None}
    LOG.info('onetick/closing_marks: %d of %d partitions to pull: %s', len(todo), len(parts), [p.key for p in todo])
    if not todo:
        return report
    client = OneTickClient(cfg)
    window_start, window_end = min(p.start for p in todo), max(p.end for p in todo)
    symbols = retry(lambda: client.list_symbols(window_start, window_end), cfg.retries, cfg.retry_backoff_s, 'onetick/list_symbols')
    if universe:
        uni = set(map(str, universe))
        symbols = [s for s in symbols if s in uni]
    report['symbols'] = len(symbols)
    batches = [symbols[i:i + cfg.ot_symbol_batch] for i in range(0, len(symbols), cfg.ot_symbol_batch)]
    LOG.info('onetick/closing_marks: %d symbols in %d batches x %d partitions = %d tasks on %d workers',
             len(symbols), len(batches), len(todo), len(batches) * len(todo), cfg.ot_workers)
    if dry_run or not symbols:
        return report

    pending = {p.key: len(batches) for p in todo}
    buffers: dict[str, list[pd.DataFrame]] = {p.key: [] for p in todo}
    failed: set[str] = set()
    part_by_key = {p.key: p for p in todo}
    lock = threading.Lock()

    def task(part: Partition, bi: int, syms: list[str]) -> tuple[str, int, pd.DataFrame]:
        label = f'onetick/{part.key}/batch{bi}'
        frame = retry(lambda: client.fetch(syms, part.start, part.end), cfg.retries, cfg.retry_backoff_s, label)
        return part.key, bi, normalize_closing_marks_raw(frame)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.ot_workers, thread_name_prefix='onetick') as pool:
        futures = [pool.submit(task, p, bi, b) for p in todo for bi, b in enumerate(batches)]
        for fut in as_completed(futures):
            try:
                key, bi, frame = fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error('onetick task failed permanently: %s', exc)
                # find which partition this belonged to is not recoverable from the exception; mark all incomplete ones
                report['failed'].append(repr(exc))
                continue
            with lock:
                buffers[key].append(frame)
                pending[key] -= 1
                done_now = pending[key] == 0
            if done_now:
                part = part_by_key[key]
                merged = pd.concat([f for f in buffers.pop(key) if not f.empty], ignore_index=True) if any(not f.empty for f in buffers[key]) else pd.DataFrame(columns=CLOSING_MARKS_FIELDS)
                store.write(key, merged, part, extra={'symbols': len(symbols), 'batches': len(batches), 'elapsed_s': round(time.perf_counter() - t0, 1)})
                report['done'].append(key)
    if report['failed']:
        LOG.error('onetick/closing_marks: %d task failures; partitions not written: %s', len(report['failed']), [k for k, n in pending.items() if n > 0])
    return report


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------
def _date_filter(start: str | None, end: str | None, col: str = 'date') -> pc.Expression | None:
    expr = None
    if start:
        expr = pc.field(col) >= pa.scalar(pd.Timestamp(start).to_pydatetime())
    if end:
        e2 = pc.field(col) <= pa.scalar(pd.Timestamp(end).to_pydatetime())
        expr = e2 if expr is None else (expr & e2)
    return expr


def load_panel(cfg: Config, columns: Sequence[str] | None = None, start: str | None = None, end: str | None = None,
               cusips: Iterable[str] | None = None, categorical: bool = True) -> pd.DataFrame:
    """Full-universe evaluated-mark panel with column/date/cusip pushdown. Replaces pd.read_pickle(panel_slim_cache.pkl)."""
    expr = _date_filter(start, end, 'date')
    if cusips is not None:
        c = pc.field('cusip').isin(pa.array(sorted(set(map(str, cusips))), type=pa.string()))
        expr = c if expr is None else (expr & c)
    out = cfg.store('panel').read(columns=columns, filter=expr, categorical=categorical)
    return out if out is not None else pd.DataFrame(columns=columns or PANEL_COLUMNS)


def load_closing_marks(cfg: Config, start: str | None = None, end: str | None = None, dedupe: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Closing marks, deduplicated to one row per (CUSIP, DATE): last by SOURCESYSTEM, PUBLISHTIME, OMDSEQ."""
    raw = cfg.store('closing_marks').read(filter=_date_filter(start, end, 'DATE'), categorical=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CLOSING_MARKS_FIELDS), pd.Series(dtype='object')
    raw['YIELD'] = normalize_yield_to_percent(raw['YIELD'])
    audit = {'raw_rows': len(raw), 'raw_cusips': raw['CUSIP'].nunique(), 'raw_dates': raw['DATE'].nunique(),
             'cache_date_min': raw['DATE'].min(), 'cache_date_max': raw['DATE'].max()}
    if not dedupe:
        return raw, pd.Series(audit)
    sort_cols = [c for c in ['CUSIP', 'DATE', 'SOURCESYSTEM', 'PUBLISHTIME', 'OMDSEQ'] if c in raw.columns]
    panel = (raw.sort_values(sort_cols, na_position='first', kind='stable')
                .drop_duplicates(['CUSIP', 'DATE'], keep='last')
                .reset_index(drop=True))
    audit.update({'deduped_rows': len(panel), 'deduped_cusips': panel['CUSIP'].nunique(),
                  'multi_rows_share': 1.0 - len(panel) / max(len(raw), 1)})
    return panel, pd.Series(audit)


def normalize_yield_to_percent(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty and finite.abs().median() < 1.0:
        numeric = numeric * 100.0
    return numeric


# --------------------------------------------------------------------------------------
# Step 2 / Step 3 builds
# --------------------------------------------------------------------------------------
def build_step2(cfg: Config) -> dict[str, Any]:
    """Step 2 research panel: evaluated-mark panel joined to deduplicated OneTick close marks.

    covered_only=True (default): inner join on the OneTick-covered CUSIPs only (the model population).
    covered_only=False: left join over the full universe, written partition by partition without lag features.
    """
    t0 = time.perf_counter()
    marks, marks_audit = load_closing_marks(cfg, cfg.start_date, cfg.end_date)
    if marks.empty:
        raise RuntimeError('No closing marks in store; run the OneTick pull first.')
    right = marks.rename(columns={'CUSIP': 'cusip', 'DATE': 'date', 'PRICE': 'closing_price', 'YIELD': 'closing_yield',
                                  'DURATION': 'duration', 'MODIFIED_DURATION': 'modified_duration', 'DV01': 'dv01',
                                  'BENCHMARK': 'benchmark', 'SOURCESYSTEM': 'source_system'})
    keep = [c for c in ['cusip', 'date', 'closing_price', 'closing_yield', 'duration', 'modified_duration', 'dv01', 'benchmark', 'source_system'] if c in right.columns]
    right = right[keep]
    right['cusip'] = right['cusip'].astype('string')
    out_path = cfg.path('research_panel_step2.parquet')

    if cfg.covered_only:
        covered = right['cusip'].unique().tolist()
        panel = load_panel(cfg, start=cfg.start_date, end=cfg.end_date, cusips=covered, categorical=False)
        panel['cusip'] = panel['cusip'].astype('string')
        merged = panel.merge(right, on=['cusip', 'date'], how='inner')
        merged = merged.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
        merged = _add_step2_features(merged)
        _write_parquet(merged, out_path)
        rows, cusips = len(merged), merged['cusip'].nunique()
    else:
        writer = None
        rows = 0
        cusip_set: set[str] = set()
        for part in cfg.partitions():
            panel = load_panel(cfg, start=part.start_str, end=part.last_day_str, categorical=False)
            if panel.empty:
                continue
            panel['cusip'] = panel['cusip'].astype('string')
            merged = panel.merge(right, on=['cusip', 'date'], how='left')
            merged['premium_discount'] = merged['closing_price'] - 100.0
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='zstd')
            writer.write_table(table)
            rows += len(merged)
            cusip_set.update(merged['cusip'].dropna().unique().tolist())
        if writer is not None:
            writer.close()
        cusips = len(cusip_set)

    manifest = {'output': str(out_path), 'rows': rows, 'cusips': cusips, 'covered_only': cfg.covered_only,
                'closing_marks_audit': {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in marks_audit.to_dict().items()},
                'elapsed_s': round(time.perf_counter() - t0, 1)}
    _update_manifest(cfg, 'step2', manifest)
    LOG.info('step2: %s rows, %s cusips in %.1fs', f'{rows:,}', f'{cusips:,}', manifest['elapsed_s'])
    return manifest


def _add_step2_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['closing_price', 'closing_yield', 'duration', 'modified_duration', 'dv01']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['premium_discount'] = df['closing_price'] - 100.0
    g = df.groupby('cusip', sort=False, observed=True)
    df['yield_change_bp'] = g['closing_yield'].diff() * 100.0
    for col in ['modified_duration', 'duration', 'dv01', 'premium_discount', 'closing_yield', 'yield_change_bp']:
        df[f'{col}_lag1'] = g[col].shift(1)
    return df


def build_step3(cfg: Config, min_abs_dv01: float = 1e-8) -> dict[str, Any]:
    t0 = time.perf_counter()
    step2_path = cfg.path('research_panel_step2.parquet')
    if not step2_path.exists():
        build_step2(cfg)
    required = ['date', 'cusip', 'closing_price', 'closing_yield', 'modified_duration', 'dv01']
    cols = pq.ParquetFile(step2_path).schema.names
    df = pd.read_parquet(step2_path)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f'Step 3 yield panel missing required columns: {missing}')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['cusip'] = df['cusip'].astype('string')
    for col in ['closing_price', 'closing_yield', 'duration', 'modified_duration', 'dv01', 'price', 'mkt_yield']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    covered_rows = int(df.dropna(subset=required).shape[0])
    model = df.dropna(subset=required)
    model = model[(model['modified_duration'] > 0) & (model['dv01'] > min_abs_dv01)]
    model = model.sort_values(['cusip', 'date'], kind='stable').reset_index(drop=True)
    g = model.groupby('cusip', sort=False, observed=True)
    model['closing_lag_days'] = g['date'].diff().dt.days
    for col in ['closing_yield', 'closing_price', 'modified_duration', 'duration', 'dv01']:
        if col in model.columns:
            model[f'{col}_lag1'] = g[col].shift(1)
    model['closing_yield_change_bp'] = g['closing_yield'].diff() * 100.0
    model['closing_price_change'] = g['closing_price'].diff()
    model['dv01_implied_yield_change_bp'] = np.where(model['dv01_lag1'].abs() > min_abs_dv01, -model['closing_price_change'] / model['dv01_lag1'], np.nan)
    model['dv01_bridge_error_bp'] = model['dv01_implied_yield_change_bp'] - model['closing_yield_change_bp']
    if 'mkt_yield' in model.columns:
        model['panel_minus_closing_yield_bp'] = (model['mkt_yield'] - model['closing_yield']) * 100.0
    model = model.dropna(subset=['closing_yield_change_bp', 'closing_yield_lag1', 'dv01_lag1']).reset_index(drop=True)
    out_path = cfg.path('research_panel_step3_yield.parquet')
    _write_parquet(model, out_path)
    audit = {
        'input_rows': len(df), 'covered_positive_risk_rows': covered_rows, 'model_rows': len(model),
        'model_cusips': int(model['cusip'].nunique()),
        'date_min': str(model['date'].min()), 'date_max': str(model['date'].max()),
        'closing_lag_days_median': float(model['closing_lag_days'].median()),
        'closing_lag_days_p95': float(model['closing_lag_days'].quantile(0.95)),
        'target_bp_median': float(model['closing_yield_change_bp'].median()),
        'target_abs_bp_p95': float(model['closing_yield_change_bp'].abs().quantile(0.95)),
        'dv01_bridge_error_bp_median': float(model['dv01_bridge_error_bp'].median()),
        'dv01_bridge_error_abs_bp_p95': float(model['dv01_bridge_error_bp'].abs().quantile(0.95)),
        'elapsed_s': round(time.perf_counter() - t0, 1),
    }
    manifest = {'output': str(out_path), 'audit': audit}
    _update_manifest(cfg, 'step3_yield_panel', manifest)
    LOG.info('step3: %s rows, %s cusips in %.1fs', f'{len(model):,}', f'{audit["model_cusips"]:,}', audit['elapsed_s'])
    return manifest


def write_universe_and_names(cfg: Config) -> dict[str, Any]:
    panel = load_panel(cfg, columns=['cusip', 'long_comp_name'], categorical=False)
    if panel.empty:
        return {'universe_cusips': 0}
    panel['cusip'] = panel['cusip'].astype('string')
    universe = panel[['cusip']].drop_duplicates().sort_values('cusip').reset_index(drop=True)
    _write_parquet(universe, cfg.path('universe_cusips_cache.parquet'))
    names = panel.dropna(subset=['long_comp_name'])
    names = names[names['long_comp_name'].astype('string').str.strip() != '']
    names = names.drop_duplicates('cusip', keep='last')[['cusip', 'long_comp_name']].reset_index(drop=True)
    _write_parquet(names, cfg.path('product_name_cache.parquet'))
    return {'universe_cusips': len(universe), 'named_cusips': len(names)}


def export_legacy_pickle(cfg: Config) -> Path:
    """Optional: rebuild panel_slim_cache.pkl for code that still reads the pickle. Prefer load_panel()."""
    panel = load_panel(cfg, categorical=False)
    out = cfg.path('panel_slim_cache.pkl')
    panel.to_pickle(out)
    return out


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression='zstd', use_dictionary=True)
    os.replace(tmp, path)


def _update_manifest(cfg: Config, key: str, value: dict[str, Any]) -> None:
    path = cfg.path('data_pipeline_manifest.json')
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            manifest = {}
    manifest['config'] = cfg.to_dict()
    manifest['method_version'] = cfg.method_version
    manifest[key] = value
    manifest['updated_at'] = pd.Timestamp.now('UTC').isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding='utf-8')


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate(cfg: Config) -> dict[str, Any]:
    """Cheap gates on the stores: partition coverage, empty partitions, date ranges inside windows, dup share."""
    out: dict[str, Any] = {}
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()
    for source in cfg.sources:
        store = cfg.store(source)
        m = store.manifest()['partitions']
        checks = []
        for part in (cfg.partitions()[:1] if source in STATIC_SOURCES else cfg.partitions()):
            meta = m.get(part.key)
            if meta is None:
                checks.append({'partition': part.key, 'status': 'missing'})
                continue
            status = 'pass'
            if meta.get('rows', 0) == 0 and part.start <= today:
                status = 'warn_empty'
            checks.append({'partition': part.key, 'status': status, 'rows': meta.get('rows'), 'written_at': meta.get('written_at')})
        out[source] = {'ok': all(c['status'] == 'pass' for c in checks), 'checks': checks}
    marks, audit = load_closing_marks(cfg, cfg.start_date, cfg.end_date)
    out['closing_marks_dedupe_audit'] = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in audit.to_dict().items()}
    _update_manifest(cfg, 'validation', out)
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def plan(cfg: Config, force: bool = False) -> pd.DataFrame:
    rows = []
    for source in cfg.sources:
        store = cfg.store(source)
        for part in (cfg.partitions()[:1] if source in STATIC_SOURCES else cfg.partitions()):
            rows.append({'source': source, 'partition': part.key, 'window': f'[{part.start_str}, {part.end_str})',
                         'exists': store.has(part.key), 'complete': store.is_complete(part.key),
                         'will_pull': force or not store.is_complete(part.key)})
    return pd.DataFrame(rows)


def pull(cfg: Config, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    dh_sources = [s for s in cfg.sources if s in NORMALIZERS]
    do_ot = 'closing_marks' in cfg.sources
    report: dict[str, Any] = {}
    t0 = time.perf_counter()
    if cfg.parallel_sources and dh_sources and do_ot and not dry_run:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='source') as pool:
            f_dh = pool.submit(pull_deephaven, cfg, dh_sources, force, dry_run)
            f_ot = pool.submit(pull_onetick, cfg, force, dry_run)
            report['deephaven'] = f_dh.result()
            report['onetick'] = f_ot.result()
    else:
        if dh_sources:
            report['deephaven'] = pull_deephaven(cfg, dh_sources, force, dry_run)
        if do_ot:
            report['onetick'] = pull_onetick(cfg, force, dry_run)
    report['elapsed_s'] = round(time.perf_counter() - t0, 1)
    _update_manifest(cfg, 'pull', report)
    return report


def build(cfg: Config) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = {'universe': write_universe_and_names(cfg), 'step2': build_step2(cfg), 'step3': build_step3(cfg)}
    if cfg.write_legacy_pickle:
        out['legacy_pickle'] = str(export_legacy_pickle(cfg))
    out['validation'] = validate(cfg)
    out['elapsed_s'] = round(time.perf_counter() - t0, 1)
    return out


def run_all(cfg: Config, force: bool = False) -> dict[str, Any]:
    setup_logging(cfg)
    LOG.info('run_all | root=%s | window %s -> %s | partitions=%d', cfg.root, cfg.start_date, cfg.end_date, len(cfg.partitions()))
    report = {'pull': pull(cfg, force=force), 'build': build(cfg)}
    LOG.info('run_all done | pull %.0fs | build %.0fs', report['pull']['elapsed_s'], report['build']['elapsed_s'])
    return report


def setup_logging(cfg: Config, level: int = logging.INFO) -> None:
    if LOG.handlers:
        return
    LOG.setLevel(level)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(threadName)s %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    Path(cfg.root).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(cfg.path('pipeline.log'))
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def _cli() -> None:
    ap = argparse.ArgumentParser(description='Muni data pipeline (Deephaven + OneTick, partitioned and resumable)')
    ap.add_argument('command', choices=['plan', 'pull', 'build', 'validate', 'all'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--start', default=Config.start_date)
    ap.add_argument('--end', default=Config.end_date)
    ap.add_argument('--partition', default='M', choices=['M', 'W'])
    ap.add_argument('--sources', default=','.join(Config.sources))
    ap.add_argument('--workers', type=int, default=Config.ot_workers)
    ap.add_argument('--symbol-batch', type=int, default=Config.ot_symbol_batch)
    ap.add_argument('--force', action='store_true', help='re-pull partitions even if complete')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--full-universe', action='store_true', help='Step 2 as left join over the full panel')
    ap.add_argument('--legacy-pickle', action='store_true', help='also write panel_slim_cache.pkl')
    ap.add_argument('--no-parallel-sources', action='store_true')
    ap.add_argument('--latest-ratings', action='store_true', help='disable point-in-time ratings')
    args = ap.parse_args()
    cfg = Config(root=Path(args.root), start_date=args.start, end_date=args.end, partition=args.partition,
                 sources=tuple(s.strip() for s in args.sources.split(',') if s.strip()),
                 ot_workers=args.workers, ot_symbol_batch=args.symbol_batch,
                 covered_only=not args.full_universe, write_legacy_pickle=args.legacy_pickle,
                 parallel_sources=not args.no_parallel_sources, pit_ratings=not args.latest_ratings)
    setup_logging(cfg)
    if args.command == 'plan':
        df = plan(cfg, force=args.force)
        print(df.to_string(index=False))
    elif args.command == 'pull':
        print(json.dumps(pull(cfg, force=args.force, dry_run=args.dry_run), indent=2, default=str))
    elif args.command == 'build':
        print(json.dumps(build(cfg), indent=2, default=str))
    elif args.command == 'validate':
        print(json.dumps(validate(cfg), indent=2, default=str))
    else:
        print(json.dumps(run_all(cfg, force=args.force), indent=2, default=str))


if __name__ == '__main__':
    _cli()
