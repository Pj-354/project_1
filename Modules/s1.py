# ibkr_incremental_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np
from glob import glob 
from pathlib import Path
import os 
import re 
from datetime import datetime, timedelta
from typing import Iterable, Optional, Dict, List
pd.set_option('future.no_silent_downcasting', True)
import datetime as dt
from pandas.tseries.offsets import BDay
from scipy.stats import zscore
from typing import Literal, Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
from ib_insync import IB, util, Stock

BarSize = Literal["5m", "1d"]
DATASETS_ROOT = Path(
    os.environ.get("MOON2_DATASETS","/Users/phillip/Desktop/Moon2/data/datasets")
).expanduser().resolve()
DAILY_ROOT = Path(
    os.environ.get("MOON2_DATASETS", "/Users/phillip/Desktop/Moon2/data/daily")
).expanduser().resolve()

def _ib_bar_size(bar_size: BarSize) -> str:
    return {"5m": "5 mins", "1d": "1 day"}[bar_size]

def _ib_what_to_show(bar_size: BarSize, what_override: Optional[str]) -> str:
    if what_override:
        return what_override
    # Sensible defaults for US equities:
    return "TRADES" if bar_size == "5m" else "ADJUSTED_LAST"

def _resolve_paths(ticker: str, bar_size: BarSize) -> Tuple[Path, str]:
    if bar_size == "1d":
        return (DAILY_ROOT / f"{ticker}_daily.csv", "csv")
    else:
        return (DATASETS_ROOT / f"{ticker}_5minute.parquet", "parquet")

def _read_last_timestamp(path: Path, kind: str) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    if kind == "csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(path)
    if df.empty:
        return None
    return pd.to_datetime(df["timestamp"].max())

def _append_save(path: Path, kind: str, df_new: pd.DataFrame) -> None:
    if df_new.empty:
        return
    # standardize columns and ordering
    cols = ["timestamp","open","high","low","close","volume"]
    df_new = df_new[cols].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if path.exists():
        if kind == "csv":
            df_old = pd.read_csv(path, parse_dates=["timestamp"])
            df = (pd.concat([df_old, df_new], ignore_index=True)
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last"))
            df.to_csv(path, index=False)
        else:
            df_old = pd.read_parquet(path)
            df = (pd.concat([df_old, df_new], ignore_index=True)
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last"))
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, path)
    else:
        if kind == "csv":
            df_new.to_csv(path, index=False)
        else:
            table = pa.Table.from_pandas(df_new, preserve_index=False)
            pq.write_table(table, path)

def _throttle_needed(reqs_per_10min: int) -> float:
    # crude pacing helper; keep well under 60/10min budget.
    # e.g. at 30/10min, sleep ~20s every 30 requests. You can replace with a token bucket if you prefer.
    return 0.0

def _chunk_duration_for(bar_size: BarSize) -> str:
    # Keep conservative to avoid server-side chunking limits.
    return "30 D" if bar_size == "5m" else "2 Y"

def _bars_to_df(bars, bar_size: BarSize) -> pd.DataFrame:
    # ib_insync BarData has fields: date, open, high, low, close, volume, etc.
    recs = [{
        "timestamp": pd.to_datetime(b.date),
        "open": b.open, "high": b.high, "low": b.low, "close": b.close,
        "volume": getattr(b, "volume", 0)
    } for b in bars]
    df = pd.DataFrame.from_records(recs)
    # Normalize to UTC and (optionally) floor to 5-minute grid if you want strict alignment
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT", errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    if bar_size == "1d":
        # daily bars are session dates; keep as date-aligned midnight UTC
        df["timestamp"] = df["timestamp"].dt.normalize()
    return df

class IBKRLoader:
    def __init__(self, host: str="127.0.0.1", port: int=7497, client_id: int=1, readonly: bool=True):
        util.startLoop()
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id, readonly=readonly)

    def _fetch_chunk(self, contract, end_dt: dt.datetime, duration: str,
                     bar_size_str: str, what_to_show: str, use_rth: bool):
        return self.ib.reqHistoricalData(
            contract=contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size_str,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,   # get epoch/ISO datetime
            keepUpToDate=False
        )

    def update_local(
        self,
        ticker: str,
        bar_size: BarSize,
        exchange: str="SMART",
        currency: str="USD",
        use_rth: bool=False,
        what_override: Optional[str]=None,
        primary_exchange: Optional[str]=None
    ) -> Path:
        path, kind = _resolve_paths(ticker, bar_size)
        last_ts = _read_last_timestamp(path, kind)
        # define contract
        c = Stock(ticker, exchange, currency, primaryExchange=primary_exchange)
        self.ib.qualifyContracts(c)

        # If no local data, pull a modest seed; if local exists, pull from last_ts forward.
        bar_size_str = _ib_bar_size(bar_size)
        what_to_show = _ib_what_to_show(bar_size, what_override)

        end_dt = dt.datetime.utcnow()
        # loop in chunks backward from now until we cover [last_ts, now]
        to_concat = []
        duration = _chunk_duration_for(bar_size)
        while True:
            bars = self._fetch_chunk(c, end_dt, duration, bar_size_str, what_to_show, use_rth)
            df = _bars_to_df(bars, bar_size)
            if df.empty:
                break
            # keep only rows strictly greater than last_ts (if present)
            if last_ts is not None:
                df = df[df["timestamp"] > last_ts]
            if df.empty:
                break
            to_concat.append(df)
            # continue until we reached last_ts or we decide it’s enough
            earliest = df["timestamp"].min()
            if last_ts is None or earliest <= last_ts + pd.Timedelta(seconds=1):
                break
            # next leg: end at earliest - 1 second
            end_dt = earliest.to_pydatetime() - dt.timedelta(seconds=1)

        if to_concat:
            new_df = pd.concat(to_concat, ignore_index=True)
            _append_save(path, kind, new_df)

        return path
