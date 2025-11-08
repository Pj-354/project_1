from asyncio import events
from matplotlib import ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
from pathlib import Path
import os 
import re 
from datetime import datetime, timedelta
from typing import Iterable, Optional, Dict
pd.set_option('future.no_silent_downcasting', True)

from polygon import RESTClient
import datetime as dt
from pandas.tseries.offsets import BDay
import requests
import time
import json
from matplotlib.dates import MonthLocator, AutoDateFormatter, AutoDateFormatter
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression



API_key = 'tt2gOLH0fHAmPX70a4QURLFy59PRCZr3'
client = RESTClient(API_key, trace=True)
base_url = "https://api.polygon.io/vX/reference/financials"
headers = {"Authorization": f"Bearer {API_key}"}

BINS = [-np.inf, -2, -1, 1, 2, np.inf]
LABELS = ["Large Negative", "Moderate Negative", "Normal", "Moderate Positive", "Large Positive"]


def rate_limited_request(url, headers, params):
    
    response = requests.get(url, headers=headers, params=params)

    return response


def calc_log_rets(df: pd.DataFrame):
    px = df.astype(float)
    prev = px.shift(1)

    # valid only where both prices > 0
    valid = (px > 0) & (prev > 0)

    ratio = (px / prev).where(valid)          # NaN where invalid (zero/negative/first row)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.log(ratio)                  # log(p_t / p_{t-1})
    return rets.dropna(how="all")


def get_top_N_US_stocks(N):

    URL = "https://www.slickcharts.com/sp500"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()  # will raise if still forbidden

    tables = pd.read_html(resp.text)
    spx_df = tables[0]  # first table = current constituents
    tickers = spx_df["Symbol"].astype(str).str.upper().str.replace(r"[^\w\.-]", "", regex=True).tolist()

    return tickers[:N+1]

def handle_missing_fundamental_data(df, *, eps=0.0):
    """
    Backfill missing share-counts and EPS (quarterly vs yearly handled separately).
    Zero-division safe: any 0/NaN (or |den|<=eps) denominator yields NaN.
    """
    df = df.copy()

    def safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        ok = den.notna() & (den.abs() > eps)
        out = pd.Series(np.nan, index=num.index, dtype=float)
        out.loc[ok] = num.loc[ok] / den.loc[ok]
        return out

    q = df["quarterly"].fillna(False).astype(bool)

    # -- Quarter-only: back out avg shares from quarterly earnings / EPS
    m = df["basic_avg_shares"].isna() & q
    df.loc[m, "basic_avg_shares"] = safe_div(df.loc[m, "earnings_quarterly"], df.loc[m, "basic_eps"])

    m = df["diluted_avg_shares"].isna() & q
    df.loc[m, "diluted_avg_shares"] = safe_div(df.loc[m, "earnings_quarterly"], df.loc[m, "diluted_eps"])

    # -- Yearly: rough approx avg shares from yearly earnings / yearly EPS
    m = df["basic_avg_shares"].isna() & ~q
    df.loc[m, "basic_avg_shares"] = safe_div(df.loc[m, "earnings"], df.loc[m, "basic_eps_"])

    m = df["diluted_avg_shares"].isna() & ~q
    df.loc[m, "diluted_avg_shares"] = safe_div(df.loc[m, "earnings"], df.loc[m, "diluted_eps_"])

    # -- Fill missing yearly EPS from quarterly earnings and computed shares
    m = df["basic_eps"].isna() & ~q
    df.loc[m, "basic_eps"] = safe_div(df.loc[m, "earnings_quarterly"], df.loc[m, "basic_avg_shares"])

    m = df["diluted_eps"].isna() & ~q
    df.loc[m, "diluted_eps"] = safe_div(df.loc[m, "earnings_quarterly"], df.loc[m, "diluted_avg_shares"])

    return df

from pathlib import Path
import re, pandas as pd, numpy as np, datetime as dt

def get_minute_level_data(
    ticker: str,
    n: int,
    period: str,
    start_date: str,
    base_dir: str = "Datasets",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load the EARLIEST parquet for filenames like:
        {ticker}_5minute_{YYYY-MM}_minute_level_data.parquet
    - Recursive search under base_dir (and module-anchored fallback)
    - Normalize tz-aware index, reindex to n-minute grid in US/Eastern
    - FFill OHLC only
    """
    # --- where are we searching? ---
    cwd = Path.cwd()
    primary_base = Path(base_dir)
    fallback_base = Path(__file__).resolve().parent.parent / base_dir  # e.g., project_root/Datasets

    searched = []
    candidates = []

    def _collect(base: Path):
        searched.append(str(base.resolve()))
        if base.exists():
            candidates.extend(base.rglob(f"{ticker}_*minute_*_minute_level_data.parquet"))

    _collect(primary_base)
    if fallback_base != primary_base:
        _collect(fallback_base)

    if verbose:
        print(f"[get_minute_level_data] CWD: {cwd}")
        print(f"[get_minute_level_data] Searched: {searched}")
        print(f"[get_minute_level_data] Found: {[p.name for p in candidates]}")

    if not candidates:
        raise FileNotFoundError(f"No files found for {ticker} under: {searched}")

    # enforce exact pattern and pick earliest YYYY-MM
    rx = re.compile(
        rf"^{re.escape(ticker)}_(\d+)minute_(\d{{4}}-\d{{2}})_minute_level_data\.parquet$"
    )
    dated = []
    for p in candidates:
        m = rx.match(p.name)
        if not m:
            continue
        mins = int(m.group(1))
        ym = pd.to_datetime(m.group(2), format="%Y-%m")
        if mins == 5:  # keep only 5-minute files
            dated.append((ym, p))
    if not dated:
        raise FileNotFoundError(
            f"Found parquet files, but none matched '{ticker}_5minute_YYYY-MM_minute_level_data.parquet'."
        )

    path = min(dated, key=lambda t: t[0])[1]
    if verbose:
        print(f"[get_minute_level_data] Selected: {path.name}")

    # --- load & normalize datetime index (tz-aware) ---
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.drop(columns=["Date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx = idx.dt.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    else:
        raise ValueError("Expected 'Date' column or DatetimeIndex in parquet.")

    df.index = idx.dt.tz_convert("US/Eastern")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if df.empty:
        return df

    # --- full regular n-minute grid ---
    full_idx = pd.date_range(df.index[0], df.index[-1], freq=f"{n}min", tz="US/Eastern")
    df = df.reindex(full_idx)

    # forward-fill PRICES only
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if price_cols:
        df[price_cols] = df[price_cols].ffill()

    # session labels
    t = df.index.time
    df["session"] = np.select(
        [
            (t >= dt.time(4, 0))  & (t < dt.time(9, 30)),
            (t >= dt.time(9, 30)) & (t < dt.time(16, 0)),
            (t >= dt.time(16, 0)) & (t < dt.time(21, 0)),
        ],
        ["Pre-Market", "Regular", "Post-Market"],
        default="closed",
    )

    if verbose:
        print(f"[get_minute_level_data] Loaded {len(df)} rows | last={df.index.max()}")
    return df


def map_earnings_to_fwd_rets(fwd_df, earnings_with_prices):

    earnings_with_prices['earnings flag']       = True
    df_earnings                     = pd.merge_asof(fwd_df,
                                                    earnings_with_prices['earnings flag'],
                                                    left_index = True,
                                                    right_index= True,
                                                    direction  = 'forward',
                                                    tolerance  = pd.Timedelta('1D')
                                                     )
    return df_earnings

def fetch_in_six_month_chunks(
    ticker_obj,
    ticker : str,
    period: str = 'minute',
    n: int = 5,
    months_per_chunk: int = 3,
    cooldown_sec: int = 12,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    tz: str = "Europe/London",
):
    """
    Calls `ticker_obj.get_historical_prices(start_date=..., end_date=...)` in 6M chunks,
    concatenates `ticker_obj.minute_level_data` from each call, and saves to CSV.

    - If start_date/end_date are omitted:
        end_date = (today in tz) - 1 day
        start_date = (today in tz) - 2 years
    - Cooldown of `cooldown_sec` seconds between chunk requests.
    - Deduplicates and sorts by index before saving.
    """
    # Figure out date window
    today = pd.Timestamp.now(tz).normalize()
    if end_date is None:
        end_dt = (today - pd.Timedelta(days=1)).date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    if start_date is None:
        start_dt = (today - pd.DateOffset(years=2)).date()
    else:
        start_dt = pd.to_datetime(start_date).date()

    if start_dt > end_dt:
        raise ValueError(f"start_date {start_dt} must be <= end_date {end_dt}")

    # Build 6-month chunks
    chunks = []
    cur = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    step = pd.DateOffset(days=np.floor(months_per_chunk*30))

    while cur <= end_ts:
        next_start = cur + step
        chunk_end = min(end_ts, next_start)
        chunks.append((cur.date().isoformat(), chunk_end.date().isoformat()))
        cur = next_start + pd.Timedelta(days=1)  # move to the day after this chunk
    
    # Call API per chunk and collect data
    frames = []
    for i, (s, e) in enumerate(chunks):
        print(i, s,e)
        ticker_obj.get_historical_prices(period = period, n=n, start_date=s, end_date=e, limit = 5000_000)
        df_chunk = ticker_obj.minute_level_prices.copy()
        frames.append(df_chunk)
        time.sleep(cooldown_sec)

    combined = pd.concat(frames, axis=0, sort=False)      # row-wise stitch
    # If index is a DateTimeIndex (as typical for minute bars), this will keep last duplicate
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    # Save & return
    print('Success')
    start_dt = pd.to_datetime(start_dt).strftime('%Y-%m')
    combined.index = pd.to_datetime(combined['timestamp'], unit='ms').tz_localize('UTC').tz_convert('US/Eastern')
    combined.to_csv(f'data/datasets/{ticker}_{n}{period}_{start_dt}_minute_level_data.csv')

    return combined

def load_benchmark(ticker, start):
    t = TickerData(ticker, filing_date_gte=start)
    t.get_historical_prices()
    t.calc_zscore_and_rets(earnings=False)
    # Use 'close' (or 'adj_close' if you have it) and drop NAs
    px = t.historical_prices['close'].dropna()
    px.index = px.index.tz_localize(None) if hasattr(px.index, "tz") and px.index.tz is not None else px.index
    return px, t  

def _safe_div_series(num, den, eps=0.0):
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=n.index, dtype=float)
    ok = d.notna() & (d.abs() > eps)
    out.loc[ok] = n.loc[ok] / d.loc[ok]
    return out

def latest_daily_file(glob_obj: Iterable[Path]) -> Optional[Path]:
    """
    Pick the newest file among matches like  ANYPREFIX_*_daily.csv
    where * is a date/time string (e.g. YYYY-MM-DD, YYYYMMDD, ...).
    Returns a Path or None.
    """
    candidates = [p for p in glob_obj if p.is_file()]
    if not candidates:
        return None

    def parse_dt(p: Path) -> datetime:
        # Expect '<prefix>_<datepart>_daily.csv'
        m = re.match(r'^(.+?)_(.+?)_daily\.csv$', p.name)
        if not m:
            return datetime.min
        s = m.group(2)  # the datepart between underscores

        # Try common formats
        for fmt in (
            "%Y-%m-%d", "%Y%m%d",
            "%Y-%m-%d_%H%M%S", "%Y%m%d_%H%M%S",
            "%Y-%m-%dT%H%M%S", "%Y%m%dT%H%M%S",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass

        # Fallback: strip to digits and try again
        digits = re.sub(r"\D", "", s)
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(digits, fmt)
            except ValueError:
                pass

        return datetime.min

    return max(candidates, key=parse_dt, default=None)




class TickerData:

    def __init__(self, ticker: str, filing_date_gte: str):
        self.ticker = ticker
        self.filing_date_gte = filing_date_gte
        self.forecast_date = (pd.to_datetime(filing_date_gte) - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        date_today = (pd.Timestamp.today(tz="UTC").tz_convert("US/Eastern")).date()
        self.date_today = pd.Timestamp(date_today).strftime("%Y-%m-%d")

        # populated later
        self.historical_prices: Optional[pd.DataFrame] = None
        self.minute_level_prices: Optional[pd.DataFrame] = None
        self.no_call_needed: bool = False

    def get_historical_prices(
            self,
            *,
            date_updated: bool = False,
            period: str = "day",
            n: int = 1,
            start_date: str | bool = False,
            end_date: str | bool = False,
            limit: int = 50000,
            fetch_in_chunks: bool = False,
            chunksize: int = 90,
            cooldown_sec: int = 12,
            time_delta: int = 0
        ) -> None:
            """
            Router: daily vs minute. Keeps your original signature so callers don't change.
            - day: n is always forced to 1
            - minute: n is always forced to 5
            """
            if not start_date:
                start_date = self.filing_date_gte
            if not end_date:
                end_date = (pd.to_datetime(self.date_today) - pd.Timedelta(days=time_delta)).strftime("%Y-%m-%d")

            if period == "day":
                n = 1  # enforce daily bars
                self._sync_daily(
                    start_date=start_date,
                    end_date=end_date,
                    date_updated=date_updated,
                    limit=limit,
                    fetch_in_chunks=fetch_in_chunks,
                    chunksize=chunksize,
                    cooldown_sec=cooldown_sec,
                )
            else:
                n = 5  # enforce 5-minute bars
                self._sync_minute(
                    n=n,
                    start_date=start_date,
                    end_date=end_date,
                    date_updated=date_updated,
                    limit=limit,
                    fetch_in_chunks=fetch_in_chunks,
                    chunksize=chunksize,
                    cooldown_sec=cooldown_sec,
                    period=period,
                )

    def _sync_daily(
            self,
            *,
            start_date: str,
            end_date: str,
            date_updated: bool,
            limit: int,
            fetch_in_chunks: bool,
            chunksize: int,
            cooldown_sec: int,
        ) -> None:
            """
            Daily: load existing file if present; if date_updated=True, fetch tail and merge; always save Daily/{TICKER}_{end_date}_daily.csv
            """
            daily_path = self.get_daily_price_data()
            have_file = daily_path is not None
            if end_date is None:
                end_date = self.date_today

            if have_file:
                existing = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            else:
                existing = None

            if have_file and not date_updated:
                # Use file only
                self.historical_prices = existing
                self.no_call_needed = True
                return

            # Determine API range
            api_start = start_date
            if have_file and date_updated:
                # parse the last cached day from the CSV index
                last_idx = pd.to_datetime(existing.index).max()
                api_start = last_idx.strftime("%Y-%m-%d")  # inclusive; dedupe later

            # fetch
            fresh = self._fetch_aggs(
                n=1,
                period="day",
                start_date=api_start,
                end_date=end_date,
                limit=limit,
                fetch_in_chunks=fetch_in_chunks,
                chunksize=chunksize,
                cooldown_sec=cooldown_sec,
            )
            fresh = self._normalize_aggs(fresh, period="day")

            # merge
            if have_file:
                df = pd.concat([existing, fresh], axis=0)
            else:
                df = fresh

            # de-dup by 'timestamp' then sort by 'timestamp'
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            df.index = pd.to_datetime(df.index)

            self.historical_prices = df
            self.no_call_needed = False

            # Save one file per run (your convention)
            out = Path("data/daily") / f"{self.ticker}_{end_date}_daily.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            self.historical_prices.to_csv(out)

        # --------------------------- Minute flow -----------------------------

    def _sync_minute(
        self,
        *,
        n: int,
        start_date: str,
        end_date: str,
        date_updated: bool,
        limit: int,
        fetch_in_chunks: bool,
        chunksize: int,
        cooldown_sec: int,
        period: str,
    ) -> None:
        """
        Minute: try to load the file named by oldest month (derived from the provided start_date month),
        then optionally append tail and overwrite the CSV named by the *true* oldest month in the merged data.
        """
        # Try to load a pre-existing month-named CSV (based on the *requested* start_date's month)
        try:
            df_file = get_minute_level_data(self.ticker, n, period, start_date)  # index tz-aware Eastern
            have_file = True
        except Exception as e:
            print(f"[{self.ticker}] No minute CSV found: {e}")
            df_file = None
            have_file = False

        if have_file and not date_updated:
            # Use file only
            self.minute_level_prices = df_file.sort_index()
            self.no_call_needed = True
            # Ensure we save back using the month of the true oldest entry
            oldest_month = pd.to_datetime(self.minute_level_prices.index.min()).strftime("%Y-%m")
            print(f'saving because we have the file and we do not want to update date : {self.minute_level_prices.index[-1]}')
            self._sav(n=n, period=period, month_tag=oldest_month)
            return

        # When we must fetch (either file missing, or date_updated=True)
        if have_file and date_updated:
            api_start = pd.to_datetime(df_file.index.max()).strftime("%Y-%m-%d")  # inclusive
        else:
            api_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")

        fresh = self._fetch_aggs(
            n=n,
            period=period,
            start_date=api_start,
            end_date=end_date,
            limit=limit,
            fetch_in_chunks=fetch_in_chunks,
            chunksize=chunksize,
            cooldown_sec=cooldown_sec,
        )
        fresh = self._normalize_aggs(fresh, period="minute")  # tz-aware Eastern index

        if have_file:
            print('merging old and new')
            merged = pd.concat([df_file, fresh], axis=0)
        else:
            print('just have new')
            merged = fresh

        # de-dup by 'timestamp' then sort by 'timestamp'
        if "timestamp" in merged.columns:
            merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        self.minute_level_prices = merged.sort_index()
        self.no_call_needed = False

        # Save to the month derived from the **true oldest entry** in the merged frame
        oldest_month = pd.to_datetime(self.minute_level_prices.index.min()).strftime("%Y-%m")
        self._save_minute_pq(n=n, period=period, month_tag=oldest_month)

    # ----------------------------- Utilities -----------------------------

    def _fetch_aggs(
        self,
        *,
        n: int,
        period: str,
        start_date: str,
        end_date: str,
        limit: int,
        fetch_in_chunks: bool,
        chunksize: int,
        cooldown_sec: int,
    ) -> pd.DataFrame:
        """Unified Polygon fetch, supports optional chunking. Dates are YYYY-MM-DD strings."""
        start_s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_s = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        def _fetch_chunk(s_ts: pd.Timestamp, e_ts: pd.Timestamp) -> pd.DataFrame:
            raw = []
            for a in client.list_aggs(
                ticker=self.ticker,
                multiplier=n,
                timespan=period,
                from_=s_ts.strftime("%Y-%m-%d"),
                to=e_ts.strftime("%Y-%m-%d"),
                limit=limit,
            ):
                raw.append(a)
            return pd.DataFrame(raw) if raw else pd.DataFrame()

        if not fetch_in_chunks:
            raw = []
            for a in client.list_aggs(
                ticker=self.ticker,
                multiplier=n,
                timespan=period,
                from_=start_s,
                to=end_s,
                limit=limit,
            ):
                raw.append(a)
            return pd.DataFrame(raw)

        # chunked
        frames = []
        cur_start = pd.Timestamp(start_s)
        end_ts = pd.Timestamp(end_s)
        delta = timedelta(days=chunksize)
        while cur_start < end_ts:
            cur_end = min(cur_start + delta, end_ts)
            frames.append(_fetch_chunk(cur_start, cur_end))
            cur_start = cur_end
            if cur_start < end_ts:
                time.sleep(cooldown_sec)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _normalize_aggs(self, df: pd.DataFrame, *, period: str) -> pd.DataFrame:
        """Standardize the frame returned by Polygon into our canonical index and columns."""
        if df.empty:
            return df

        if period == "minute":
            # make Eastern tz-aware index
            df = df.copy()
            df["Date"] = (
                pd.to_datetime(df["timestamp"], unit="ms")
                .dt.tz_localize("UTC")
                .dt.tz_convert("US/Eastern")
            )  # tz-aware Eastern index. tz_localize then tz_convert is the standard path. :contentReference[oaicite:4]{index=4}
            df.set_index("Date", inplace=True)
            return df

        # daily
        df = df.copy()
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
        df.set_index("Date", inplace=True)
        return df

    def _save_minute_pq(self, *, n: int, period: str, month_tag: str) -> None:
        out = Path("Datasets") / f"{self.ticker}_{n}{period}_{month_tag}_minute_level_data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(self.minute_level_prices.index[-1], self.ticker)
        # writes pandas DataFrame to Parquet; stores tz-aware datetimes via pyarrow metadata
        self.minute_level_prices.to_parquet(out, engine="pyarrow", compression="snappy", index=True)

    def get_daily_price_data(self) -> Optional[Path]:
        """Return the most recent daily CSV Path (or None)."""
        path_to_daily = Path("data/daily")
        pattern = f"{self.ticker}_*_daily.csv"
        latest_path = latest_daily_file(path_to_daily.glob(pattern))
        return latest_path

    def latest_daily_file(glob_obj: Iterable[Path]) -> Optional[Path]:
        """
        Pick the newest file among matches like ANYPREFIX_*_daily.csv
        where * is a date/time string (YYYY-MM-DD, YYYYMMDD, etc.).
        """
        candidates = [p for p in glob_obj if p.is_file()]
        if not candidates:
            return None

        def parse_dt(p: Path) -> datetime:
            m = re.match(r"^(.+?)_(.+?)_daily\.csv$", p.name)
            if not m:
                return datetime.min
            s = m.group(2)

            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d_%H%M%S", "%Y%m%d_%H%M%S", "%Y-%m-%dT%H%M%S", "%Y%m%dT%H%M%S"):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    pass
            digits = re.sub(r"\D", "", s)
            for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
                try:
                    return datetime.strptime(digits, fmt)
                except ValueError:
                    pass
            return datetime.min

        return max(candidates, key=parse_dt, default=None)

    def get_latest_ratios_data(self):
        """ 
        Checks to see if we have already downloaded the data, so we can add to it and save it
        """
        path_to_daily       = Path('Daily') / Path('Fundamentals')
        path_to_ratios      = Path('Daily') / Path('Earnings')
        pattern             = f"{self.ticker}_*.csv"
        file_path_ratios    = latest_daily_file(path_to_daily.glob(pattern))
        file_path_earnings  = latest_daily_file(path_to_ratios.glob(pattern))

        return file_path_ratios, file_path_earnings

    def get_fundamental_data(self, test : bool = False):

        fundamentals = {}

        # Build your API calls manually

        params = {
            "ticker": self.ticker,
            "filing_date.gte": self.forecast_date,
            "limit": 50  # Adjust as needed
                 }

        raw_df = rate_limited_request(base_url, headers, params)
        json_raw_df = raw_df.json()

        # Get important data to calculate metrics
        number_of_periods = len(json_raw_df.get('results'))

        for i in range(0, number_of_periods):
            raw_df_period                   = json_raw_df.get('results', np.nan)[i]
            income_statement                = raw_df_period.get('financials').get('income_statement', np.nan)
            end_date                        = raw_df_period.get('end_date')

            if raw_df_period.get('timeframe') != 'quarterly':
                quarterly        = False
            else:
                quarterly        = True 
            
            
            try:
                basic_avg_shares = income_statement.get('basic_average_shares', {})
                basic_avg_shares = basic_avg_shares['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: basic_avg_shares missing")
                basic_avg_shares = np.nan

            try:
                diluted_avg_shares = income_statement.get('diluted_average_shares', {})
                diluted_avg_shares = diluted_avg_shares['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: diluted_avg_shares missing")
                diluted_avg_shares = np.nan

            try:
                diluted_eps = income_statement.get('diluted_earnings_per_share', {})
                diluted_eps = diluted_eps['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: diluted_eps missing")
                diluted_eps = np.nan

            try:
                basic_eps = income_statement.get('basic_earnings_per_share', {})
                basic_eps = basic_eps['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: basic_eps missing")
                basic_eps = np.nan

            try:
                earnings = income_statement.get('net_income_loss_available_to_common_stockholders_basic', {})
                earnings = earnings['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: earnings missing")
                earnings = np.nan

            try:
                revenue = income_statement.get('revenues', {})
                revenue = revenue['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: revenue missing")
                revenue = np.nan

            try:
                r_d = income_statement.get('research_and_development', {})
                r_d = r_d['value']
            except Exception:
                r_d = np.nan

            file_date = raw_df_period.get('filing_date', np.nan)
            if file_date != np.nan:
                file_date = pd.to_datetime(file_date).date()
                file_date = (file_date - BDay(1)).date()


            fundamentals[end_date] = pd.Series([
                                    basic_avg_shares,
                                    basic_eps, 
                                    diluted_avg_shares,
                                    diluted_eps,
                                    earnings,
                                    r_d,
                                    revenue,
                                    file_date,
                                    quarterly
                                    ]
                                )   
        fundamentals_df = pd.DataFrame(fundamentals).T
        fundamentals_df.rename({
                0 : 'basic_avg_shares',
                1 : 'basic_eps', 
                2 : 'diluted_avg_shares',
                3 : 'diluted_eps',
                4 : 'earnings',
                5 : 'r&d',
                6 : 'revenue',
                7 : 'earnings tag',
                8 : 'quarterly'

            }, axis = 1, inplace = True)

        fundamentals_df.index           = pd.to_datetime(fundamentals_df.index).strftime('%Y-%m-%d')
        fundamentals_df.index           = pd.to_datetime(fundamentals_df.index)
        fundamentals_df.sort_index(ascending=True,inplace=True)
            
        # Test : returns json file
        if test:
            self.json_raw_df                = json_raw_df


        def get_quarterly_fundamentals(df):
            """ 
            Some stocks list their 4th quarter (or whatever quarter) earnings 
            as sum of the total year, so must back out what that quarter earnigns are

            used to calculate TTM EPS and TTM P/E
            
            """
            # ensure sorted by date
            df = df.sort_index()

            # sum of prior 3 quarters' earnings
            prior3          = df['earnings'].shift(1).rolling(3, min_periods=3).sum()
            prior3_rev      = df['revenue'].shift(1).rolling(3, min_periods=3).sum()
            prior3_rd       = df['r&d'].shift(1).rolling(3, min_periods=3).sum()

            # flag annual rows per your rule
            is_annual = df['quarterly'] == False

            # Placeholder for data processing
            df['basic_eps_']               = df['basic_eps'].copy()
            df['diluted_eps_']             = df['diluted_eps'].copy()
            # adjusted quarterly earnings:
            df['earnings_quarterly']       = np.where(is_annual, df['earnings'] - prior3, df['earnings'])
            df['basic_eps']                = np.where(is_annual, df['earnings_quarterly'] / df['basic_avg_shares'],
                                                                 df['basic_eps']) \
                                                                 if ~(df['basic_avg_shares'].isna() | df['basic_avg_shares'] == 0).any() else np.nan
            
            df['diluted_eps']              = np.where(is_annual, df['earnings_quarterly'] / df['diluted_avg_shares'],
                                                                 df['diluted_eps']) \
                                                                 if ~(df['diluted_avg_shares'].isna() | df['diluted_avg_shares'] == 0).any() else np.nan
            
            df['revenue']                  = np.where(is_annual, df['revenue'] - prior3_rev, df['revenue'])
            df['r&d']                      = np.where(is_annual, df['r&d'] - prior3_rd, df['r&d'])


            # Backfill the eps that we do have but set to np.nan because of zero division error
            df['basic_eps']                = np.where(df['basic_eps'] == np.nan,
                                                      df['basic_eps_'],
                                                      df['basic_eps'])

            df['diluted_eps']              = np.where(df['diluted_eps'] == np.nan,
                                                      df['diluted_eps_'],
                                                      df['diluted_eps'])
            return df
    
    
        # Process for missing values (share count and eps)
        df = get_quarterly_fundamentals(fundamentals_df)

        df = handle_missing_fundamental_data(df)
        self.fundamentals = df
    
    def get_ratios(self, date_updated = False):
        """  
        Calculate P/E Ratios and TTM Earnings
        """
        file_path_ratios, file_path_earnings = self.get_latest_ratios_data()
            
        if file_path_ratios != None and file_path_earnings != None:
            ratios_data               = pd.read_csv(file_path_ratios, index_col = ['Date'], parse_dates=True)
            earnings_dates            = pd.read_csv(file_path_earnings, index_col = [0])
            earnings_dates            = earnings_dates.dropna()
            earnings_dates.columns    = ['earnings_date']
            earnings_dates['earnings_date'] = pd.to_datetime(earnings_dates['earnings_date'])
   
            if date_updated == False:
                self.summary_data = ratios_data
                self.earning_dates = earnings_dates

                return None

        if not hasattr(self, 'fundamentals'):
            self.get_fundamental_data()

        if not hasattr(self, 'historical_prices'):
            self.get_historical_prices()

        quarterly_fundamentals                 = self.fundamentals
        historical_prices                      = self.historical_prices

        quarterly_fundamentals['TTM earnings'] = quarterly_fundamentals['earnings_quarterly'].rolling(window=4, 
                                                                                                      min_periods=4) \
                                                                                                      .sum()
        quarterly_fundamentals['TTM eps']      = quarterly_fundamentals['TTM earnings'] \
                                               / quarterly_fundamentals['diluted_avg_shares'] \
                                               if ~(quarterly_fundamentals['diluted_avg_shares'].isna() \
                                               | quarterly_fundamentals['diluted_avg_shares'] == 0) \
                                               .any() else np.nan
        
        historical_prices.index = pd.to_datetime(historical_prices.index)

        merged = pd.merge_asof(left                 = historical_prices,
                                right               = quarterly_fundamentals,
                                left_index          = True,
                                right_index         = True,
                                allow_exact_matches = True,
                                direction           = 'backward',
                                tolerance           = None
                                )
        EPS = 1e-12  

        merged['TTM P/E']               = _safe_div_series(merged['vwap'], merged['TTM eps'],      eps=EPS)
        merged['quarterly P/E diluted'] = _safe_div_series(merged['vwap'], merged['diluted_eps'],  eps=EPS) * 0.25
        merged['quarterly P/E basic']   = _safe_div_series(merged['vwap'], merged['basic_eps'],    eps=EPS) * 0.25
        
        keep = ['open', 'high', 'low', 'close', 'vwap', 'quarterly P/E diluted', 'quarterly P/E basic', 'TTM P/E', 'TTM eps', 'TTM earnings']

        self.merged_test = merged
        # negative p/e ratio makes no sense
        merged['TTM P/E']                  = np.where(merged['TTM P/E'] < 0, 0, merged['TTM P/E'])
        merged['quarterly P/E diluted']    = np.where(merged['quarterly P/E diluted'] < 0, 0, merged['quarterly P/E diluted'])
        merged['quarterly P/E basic']      = np.where(merged['quarterly P/E basic'] < 0, 0, merged['quarterly P/E basic'])
        merged['earnings tag']             = pd.to_datetime(merged['earnings tag']).dt.strftime('%Y-%m-%d')
        self.raw                           = merged
        self.summary_data                  = merged[keep]            
        self.earning_dates                 = pd.Series(merged['earnings tag'].unique()).dropna()
        self.summary_data.to_csv(rf'data/daily/Fundamentals/{self.ticker}_{self.date_today}.csv')
        self.earning_dates.to_csv(rf'data/daily/Earnings/{self.ticker}_{self.date_today}.csv')

    def map_earnings_to_prices(self,
                               method: str = "backward",
                               tolerance = dt.timedelta(5)) -> pd.DataFrame:
        """
        Maps the share price on the day closest to the earnings date so you can see on the time series plots
            - make sure it has :
                1. Historical prices
                2. Earning dates
        """
        
        prices                                      = self.historical_prices
        prices.index                                = pd.to_datetime(prices.index)
        self.earning_dates.dropna(inplace=True)
        self.earning_dates.columns                  = ['earnings_date']
        self.earning_dates['earnings_date'] = pd.to_datetime(self.earning_dates['earnings_date'])
        events                                      = self.earning_dates.copy()


        merged = pd.merge_asof(
            left=events,
            right=prices,
            left_on="earnings_date",
            right_index=True,
            direction=method,                
            tolerance=pd.Timedelta(tolerance) if tolerance else None
        )

        merged.set_index('earnings_date', inplace= True)
        merged.index = pd.to_datetime(merged.index)
        self.earning_dates_with_prices = merged

    def plot_single_timeseries(self, plot=True, earnings_show=True):
        """ 
        Plots the time series with earning dates highlighted. Dependencies
            - Needs map_earnings_to_prices to get the share price on earnigns day
            - Needs historical price chart
            - Needs Ratios (it is where the earnings logic is done)
        """
        if not hasattr(self, 'historical_prices'):
            self.get_historical_prices()

        if not hasattr(self, 'earning_dates') and earnings_show == True:
            self.get_fundamental_data()
            self.get_ratios()
        
        if not hasattr(self, 'earning_dates_with_prices') and earnings_show == True:
            self.map_earnings_to_prices()

        if earnings_show == True:
            df = self.summary_data
            earnings_dates = self.earning_dates_with_prices
        else:
            df = self.historical_prices

        if plot == True:                    
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

            if earnings_show == True:
                axs[0].scatter(earnings_dates.index, earnings_dates['close'],marker='x', color='tab:red', label='Earnings Date')
                axs[1].plot(df['quarterly P/E diluted'], label='Quarterly P/E', alpha = 0.5, linestyle='--')
                axs[1].plot(df['TTM P/E'], label='TTM P/E', color='tab:blue')

                df.loc[:,'negative TTM P/E'] = np.where(df['TTM P/E'] < 1, df['TTM P/E'], np.nan)
                df.loc[:,'negative Q P/E'] = np.where(df['quarterly P/E diluted'] < 1, df['quarterly P/E diluted'], np.nan)


                axs[1].plot(df['negative TTM P/E'], color = 'lightgray', linewidth=3)
                axs[1].plot(df['negative Q P/E'], color = 'lightgray',linewidth=3)
                axs[1].grid()
                axs[1].legend()
                axs[1].set_title('P/E Ratios')

            axs[0].plot(df['close'], label='Close Price')
            axs[0].legend()
            axs[0].set_title(f'Share Price of {self.ticker}')
            axs[0].set_ylabel('$ Dollars')
            axs[0].grid()

    def calc_log_rets(self, returns):
        rets = self.summary_data[returns].pct_change().dropna()
        rets = np.log(1 + rets).to_frame('returns')
        self.returns = rets

    def plot_returns_distribution(self,
                                  xlabel: str = "Returns",
                                  bins: int | int = 25,
                                  kde: bool = True,
                                  percent_axis: bool = True,
                                  show_stats: bool = True,
                                  ax: plt.Axes | None = None,
                                  returns : str | str = 'intraday' 
                                  ):
        """
        Plot a histogram + KDE for returns.
        """
        # sanitize
        if not hasattr(self, 'summary_returns_data'):
            try:
                self.calc_zscore_and_rets()
            except Exception as e:
                r = self.summary_data.apply(calc_log_rets)[returns].dropna().to_frame('returns')

        r = self.summary_returns_data[returns].dropna().to_frame('returns') 

        if r.empty:
            raise ValueError("No non-NaN returns to plot.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5.5))

        # Use density so the KDE is on the same scale as the histogram
        sns.histplot(data = r, x = 'returns', bins=bins, stat="density", kde=kde, ax=ax)

        # Labels / title
        
        ax.set_title(f'{returns} Distribution')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")

        # Optional percent axis (for decimal returns like 0.01 = 1%)
        if percent_axis:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))

        # Optional mean/median lines
        if show_stats:
            mu, med             = r['returns'].mean() , r['returns'].median() 
            stD, mad            = (r['returns'].std() * np.sqrt(252) * 100).round(2), (np.mean(np.abs(r['returns'] - mu)) * np.sqrt(252) * 100).round(2)
            stD_30day           = ((r['returns'].iloc[-31:-1]).std() * np.sqrt(252) * 100).round(2)

            ax.axvline(mu, linestyle="--", linewidth=1.2, label=f"Mean: {mu:.4f}")
            ax.axvline(med, linestyle=":", linewidth=1.2, label=f"Median: {med:.4f}")

            ax.annotate(
                f'Realized Vol : {stD}%\nMAD : {mad}%\n30-day Realized Vol: {stD_30day}%', # Text for Annotation
                xy = (0.025, 10), # Location of the annotation
                xytext =(r['returns'].max() - 1.5*r['returns'].std(), 15), # Location of the text
            )
            
            ax.legend()

        plt.tight_layout()
        return ax

    def compute_rolling_return_zscores(
            self,
            window: int = 30,
            *,
            columns=("open", "high", "low", "close", "vwap"),
            log_returns: bool = True,
            min_periods: int | None = None,
            ddof: int = 1,
            returns_prefix: str = "ret_",
            zscore_prefix: str = "zscore_",
            bins = BINS,
            labels = LABELS
        ):
            """
            From self.prices (daily OHLC), compute same-field returns and their rolling z-scores.

            return_t = log(price_t / price_{t-1}) if log_returns else (price_t / price_{t-1} - 1)

            zscore = (return - rolling_mean) / rolling_std

            Adds no columns to self.prices. Instead, returns and persists a new DataFrame at self.<store_attr>.
            """
            df = self.historical_prices.copy()

            # ---- returns (same-field, t vs t-1)
            P = df.loc[:, columns].astype(float)

            ratio = P.div(P.shift(1))
            if log_returns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    R = np.log(ratio)
            else:
                R = ratio - 1.0

            # Clean non-finite values (e.g., 0 or negative prices, first row)
            R = R.where(np.isfinite(R))

            # ---- rolling z-scores of returns
            if min_periods is None:
                min_periods = window

            roll = R.rolling(window=window, min_periods=min_periods)
            mu = roll.mean()
            sigma = roll.std(ddof=ddof)

            Z = (R - mu) / sigma
            Z = Z.where(sigma.ne(0))  # avoid +/-inf when std == 0

            # ---- name columns & assemble output
            R.columns = [f"{returns_prefix}{c}" for c in columns]
            Z.columns = [f"{zscore_prefix}{c}" for c in columns]

            out = pd.concat([R, Z], axis=1)

            # NEW: add rolling std of returns used for z-scores
            retstd = sigma.copy() * np.sqrt(252)  # annualized
            retstd.columns = [f"retstd_{c}" for c in columns]
            out = pd.concat([out, retstd], axis=1)

            for c in columns:
                out[f"zbin_{c}"] = pd.cut(out[f"zscore_{c}"], bins=bins, labels=labels, include_lowest=True, ordered=True)

            self.summary_returns_data = out.dropna(how='all')

    def calc_forward_log_return(self, window : int = 1):
        """
        Compute forward cumulative log return over 'window' days.
        Log returns add, so just take rolling sum
        """
        # Reverse the return series
        if not hasattr(self, 'summary_returns_data'):
            self.calc_zscore_and_rets()

        cols         = ['open', 'high', 'low', 'intraday', 'close', 'vwap', 'overnight']
        df           = self.summary_returns_data.copy()
        df_          = df[cols].copy()
        reversed_    = df_.iloc[::-1]

        for i in cols:
            df_[f'{i} forward {window} day log rets']                 = reversed_[i].shift(1).rolling(window).sum()
            df_[f'{i} forward {window} day arithmetic rets']          = np.exp(df_[f'{i} forward {window} day log rets']) - 1
            
        setattr(self, f'fwd_rets_{window}', df_.dropna(how='any'))

    def plot_fwd_returns_distribution_by_return_z(  self,
                                                    bucket_class    : str = 'Normal',
                                                    rets_type       : str = 'vwap',
                                                    fwd_rets_type   : str = 'intraday',
                                                    window          : int = 5,
                                                    filter_earnings : bool = True,
                                                    number_of_bins  : int = 25,
                                                    return_data     : bool = False):
        """
        Need an input of the summary data with OHLC 
            - calculates zscore and returns then bucket it by zscore
            - calculates forward returns
            - maps earning dates 
        For a given type of 
            rets_type : e.g., vwap, intraday, open
        Sees what the effect is on the
            fwd_rets_type e.g., how does big overnight returns affect 
                                the following day's market returns (intraday)
        For a given interval (by zscore)
            bucket_class : e.g., Large Negative

        This aims to understand what the distribution of returns in the following N days 
        Can filter out days with earnings
        """
        if not hasattr(self, 'summary_returns_data'):
            self.calc_zscore_and_rets()

        if not hasattr(self, f'fwd_rets_{window}'):
            self.calc_forward_log_return(window)
        
        if not hasattr(self, 'earning_dates_with_prices'):
            self.map_earnings_to_prices()

        normalized_rets     = self.summary_returns_data
        fwd_rets            = getattr(self, f'fwd_rets_{window}')
        earnings            = self.earning_dates_with_prices
        fwd_rets_earnings   = map_earnings_to_fwd_rets(fwd_rets, earnings)
        
        # Get bucket for rets_type (the returns that you are using as predictor)
        bucket                                             = normalized_rets[f'bucket {rets_type}']
        fwd_rets_earnings[f'bucket {rets_type}']           = bucket

        # Get the forward returns of the fwd_rets_type and combine it with returns of rets_type
        if filter_earnings:
            rets_earnings_filtered = fwd_rets_earnings[~(fwd_rets_earnings['earnings flag'] == True)]
            x = rets_earnings_filtered.filter(like=fwd_rets_type)
            y = rets_earnings_filtered.filter(like=rets_type)
        else:
            x = fwd_rets_earnings.filter(like= fwd_rets_type)
            y = fwd_rets_earnings.filter(like= rets_type)
        
        x = pd.concat([x,y], axis =1)
        x = x.copy()
        x = x[x[f'bucket {rets_type}'] == bucket_class]

        # Summary Statistics
        mean = x[f'{fwd_rets_type} forward {window} day arithmetic rets'].mean()
        std  = x[f'{fwd_rets_type} forward {window} day arithmetic rets'].std()
        skew = x[f'{fwd_rets_type} forward {window} day arithmetic rets'].skew()
        mdn  = x[f'{fwd_rets_type} forward {window} day arithmetic rets'].median()


        plt.figure(figsize=(12,6))
        sns.histplot(data = x,
                    x    = f'{rets_type} forward {window} day arithmetic rets',
                    bins = number_of_bins,
                    kde  = True,
        )

        plt.title(f'{self.ticker}:{bucket_class} Returns on {fwd_rets_type} for {window} day forward Dist\nFiltered For Earnings = {filter_earnings}\nBased on preceding {rets_type} returns')
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.annotate(
            text = f'Avg : {(100 *mean).round(2)}%\nStD : {(100*std).round(2)}%\nSkew : {(100*skew).round(2)}\nMedian : {(100*mdn).round(2)}%',
            xy= (0.01, 1)
        )
        plt.grid()
        if return_data:
            return x
        else:
            print('set return_data to True to get the dataset')

    def calc_minute_level_rets(self, bins : list[int] = BINS, labels : list[str] = LABELS):
            """
            Take a dataframe with OHLC, vwap data and calculate log rets and zscores and buckets them
            And overnight returns
            Also computes intra day returns
            """
            df = self.minute_level_prices.copy()

            intraday_rets                                      = np.log(1 + (df['close'] - df['open']) / df['open'])
            # Todays open minus Yesterdays close (shift pushes older entries forward)
            overnight_rets                                     = np.log(1+(df['open'] - df['close'].shift()) / df['close'].shift())

            df.loc[:,['open', 'high', 'low', 'close', 'vwap']] = df.loc[:,['open', 'high', 'low', 'close','vwap']].apply(calc_log_rets)
            df['intraday']                                     = intraday_rets
            df['overnight']                                    = overnight_rets
            df                                                 = df.dropna(how='any', subset=['open', 'overnight'])
            zscores                                            = df[['open', 'high', 'low', 'close','vwap', 'intraday', 'overnight']].apply(zscore)
            
            # Cut into bins
            df[['z open', 'z high', 'z low', 'z close','z vwap', 'z intraday', 'z overnight']] = zscores
            
            rets    = []
            for i in ['open', 'high', 'low', 'close','vwap', 'intraday', 'overnight']:
                returns                                  = df[f'{i}']
                returns_zscore                           = df[f'z {i}']                                            
                returns_bucket                           = pd.cut(df[f'z {i}'],
                                                                bins=bins,
                                                                labels=labels)
                rets_filtered                            = pd.DataFrame({
                    f'{i}'        : returns,
                    f'z {i}'      : returns_zscore,
                    f'bucket {i}' : returns_bucket
                })
                rets.append(rets_filtered)

            self.summary_returns_data = pd.concat(rets, axis=1)





class TickerComparison():
    from typing import Dict, List, Optional
    """
    Build a multi-ticker view by orchestrating many TickerData objects.

    - period='day'  -> always 1-day bars (TickerData enforces n=1)
    - period='minute' -> always 5-minute bars (TickerData enforces n=5)
    - date_updated=False -> NEVER call API, only load existing CSVs via TickerData
    - date_updated=True  -> append tail from last cached timestamp (inclusive), then de-dup & save
    """

    def __init__(
        self,
        tickers: List[str],
        filing_date_gte: str,
        *,
        period: str = "day",
        date_updated: bool = False,
        fetch_in_chunks: bool = False,
        waiting_time: int = 15,
        # (optional) you can pass through a global start/end, or let TickerData use its defaults
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunksize : int = 200
    ):
        self.tickers = tickers
        self.filing_date_gte = filing_date_gte
        self.period = period
        self.date_updated = date_updated
        self.fetch_in_chunks = fetch_in_chunks
        self.waiting_time = waiting_time
        self.start_date = start_date
        self.end_date = end_date
        self.chunksize = chunksize

        prices_by_ticker: Dict[str, pd.DataFrame] = {}
        returns_by_ticker: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        # Throttle only when we actually made an API call in prior iteration
        last_call_hit_api = False

        for idx, tkr in enumerate(tickers, start=1):
            try:
                print(f"[{idx}/{len(tickers)}] {tkr}: start")

                td = TickerData(tkr, self.filing_date_gte)

                # Route to TickerData; it enforces n=1 (day) / n=5 (minute)
                td.get_historical_prices(
                    period=self.period,
                    date_updated=self.date_updated,
                    fetch_in_chunks=self.fetch_in_chunks,
                    cooldown_sec=self.waiting_time,
                    start_date=self.start_date if self.start_date else None,
                    end_date=self.end_date   if self.end_date   else None,
                    chunksize = self.chunksize,
                    limit = 500000                
                    )

                # Decide if we hit the API (TickerData sets this)
                last_call_hit_api = not getattr(td, "no_call_needed", False)

                # Prices (always)
                if self.period == "minute":
                    if td.minute_level_prices is None or td.minute_level_prices.empty:
                        raise ValueError("minute_level_prices is empty")
                    prices_by_ticker[tkr] = td.minute_level_prices.copy(deep=True)
                else:
                    if td.historical_prices is None or td.historical_prices.empty:
                        raise ValueError("historical_prices is empty")
                    prices_by_ticker[tkr] = td.historical_prices.copy(deep=True)

                    # Daily returns/z-scores (if available)
                    if hasattr(td, "summary_returns_data") and td.summary_returns_data is not None:
                        returns_by_ticker[tkr] = td.summary_returns_data.copy(deep=True)

                print(f"[{idx}/{len(tickers)}] {tkr}: done")

            except Exception as e:
                print(f"[{tkr}] failed: {e}")
                failed.append(tkr)
                # Don’t sleep on failures, and don’t mark as API call
                last_call_hit_api = False
                time.sleep(waiting_time)
                continue

        # --- Collate outputs (outer join by index for different trading calendars/timestamps) ---
        # Prices
        if prices_by_ticker:
            # axis=1 concat keeps each ticker as a column block; outer join by index is default.
            # This is the standard pattern for combining aligned time series. :contentReference[oaicite:0]{index=0}
            self.tickers_stocks_prices = pd.concat(prices_by_ticker, axis=1)
        else:
            self.tickers_stocks_prices = pd.DataFrame()

        # Daily returns/z-scores (not for minute)
        if self.period != "minute" and returns_by_ticker:
            self.tickers_time_series_returns = pd.concat(returns_by_ticker, axis=1)
        else:
            self.tickers_time_series_returns = pd.DataFrame()

        self.failed_tickers = failed



    def analyse_correlations(self, ticker_X, ticker_y, plot : bool = True, returns : str = 'close'):
        """
        Given two tickers, compute the betas to each other, correlation, and plot scatter
        """
        X = self.tickers_time_series.loc[:, (ticker_X, returns)].to_numpy().reshape(-1,1)
        y = self.tickers_time_series.loc[:, (ticker_y, returns)]

        model       = LinearRegression(fit_intercept=False).fit(X,y)
        beta_coef   = model.coef_[0]
        r2          = model.score(X,y)

        if plot == True:
            plt.title(f'Slope : {np.round(beta_coef,2)}\nR^2 {np.round(r2,2)}')

            plt.scatter(X,y)
            plt.xlabel(f"{ticker_X}")
            plt.ylabel(f'{ticker_y}')
            plt.legend()
            plt.show()
    
    def get_ratios(self):
        """ 
        For each ticker tries to get earnings
        """
        failed_earnings           = []
        earnings_with_prices_dict = {}
        stock_prices_with_ratios  = {}
        for ticker in self.tickers:
            
            try: 
                ticker_obj      = TickerData(ticker, filing_date_gte=self.filing_date_gte)
                ticker_obj.get_ratios()
                ticker_obj.plot_single_timeseries(plot=False)
                stock_prices_with_ratios[ticker]     = ticker_obj.summary_data
                earnings_with_prices_dict[ticker]    = ticker_obj.earning_dates_with_prices.copy(deep=True)
            except KeyError as e:
                print(ticker, ' failed to get earnings')
                failed_earnings.append(ticker)
            except TypeError as e:
                print(ticker, 'failed due to', e)
                failed_earnings.append(ticker)

        self.earnings_with_prices   = pd.concat(earnings_with_prices_dict, axis = 1)
        self.prices_with_ratios     = pd.concat(stock_prices_with_ratios, axis =1)
        self.failed_earnings        = failed_earnings

    def plot_time_series(self, earnings_show=True, plot=True):
            """ 
            Specify if we want to get earnings -> will try 
            """
            if earnings_show == True:
                earnings_dates = self.earnings_with_prices
                df_earnings    = self.prices_with_ratios.copy()
                df_wo_earnings = self.tickers_stocks_prices.copy()

                stock_with_earnings = df_earnings.columns.get_level_values(0)
                df_wo_earnings      = df_wo_earnings.drop(columns=stock_with_earnings)
            if earnings_show == False:
                df_wo_earnings = self.tickers_stocks_prices

            if plot == True:                    
                fig, axs = plt.subplots(2, 1, figsize=(12, 10))

                for t in earnings_dates.columns.get_level_values(0).unique():
                    earnings_to_plot = earnings_dates[t]['close'].dropna().copy()
                    stock_prices     = df_earnings[t].copy()

                    # Normalize to first price
                    base = stock_prices['close'].iloc[0]
                    earnings_to_plot      = earnings_to_plot / base
                    stock_prices['close'] = stock_prices['close'] / base

                    # 1) draw price line, capture its color
                    (line,) = axs[0].plot(stock_prices.index, stock_prices['close'], label=t)
                    col = line.get_color()

                    # 2) reuse that color for earnings marks
                    axs[0].scatter(earnings_to_plot.index, earnings_to_plot, marker='x', s=35, color='black', zorder=3)

                    # --- P/E panel (optional: keep same base color for this ticker) ---
                    # axs[1].plot(stock_prices['quarterly P/E diluted'], label=f'{t} Q', alpha=0.5, linestyle='--', color=col)
                    axs[1].plot(stock_prices['TTM P/E'], label=f'{t} TTM', color=col)

                    neg_ttm = stock_prices['TTM P/E'].where(stock_prices['TTM P/E'] < 1)
                    # neg_q   = stock_prices['quarterly P/E diluted'].where(stock_prices['quarterly P/E diluted'] < 1)
                    axs[1].plot(neg_ttm, alpha=0.6, linewidth=3, color=col, label='_nolegend_')
                    # axs[1].plot(neg_q,   alpha=0.6, linewidth=3, color=col, label='_nolegend_')

                # Group without earnings
                for t in df_wo_earnings.columns.get_level_values(0).unique():
                    df = df_wo_earnings[t].copy()
                    axs[0].plot(df.index, df['close'] / df['close'].iloc[0], label=t)  # will use cycle colors

                axs[1].grid(); axs[1].legend(); axs[1].set_title('P/E Ratios')
                axs[0].legend(); axs[0].set_title('Share Price Group Comparison')
                axs[0].set_ylabel('Normalized to $1 Dollar'); axs[0].grid()
        


if __name__ == "__main__":


    ALAB = TickerData('ALAB', filing_date_gte='2023-09-10')
    ALAB.get_historical_prices()
    ALAB.get_fundamental_data()
    ALAB.get_ratios()
