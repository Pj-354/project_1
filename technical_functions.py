import pandas as pd
import numpy as np
from math import isnan
import time 
import datetime as dt
from sqlalchemy import update
import matplotlib.pyplot as plt

from stock_data_functions import TickerData
from read_in_data_functions import get_sub_sectors_tickers, get_sub_sectors_names
from stock_data_functions import calc_log_rets

from fredapi import Fred
fred_api = Fred('e48d0413b1cd0a3b30b58d42225373de')
filing_date = dt.datetime.today().date() - pd.Timedelta(days=365*2 -1)

pd.set_option('future.no_silent_downcasting', True)


    
class TA:

    def __init__(self, ticker : str, date = dt.datetime.today(), date_updated=True):
        """ 
        Create a object using the TickerData class
        """
        self.ticker         = ticker 
        self.filing_date    = date - dt.timedelta(days=365*2 - 1)
        ticker_obj          = TickerData(ticker, filing_date)
        ticker_obj.get_historical_prices(date_updated=date_updated)
        self.prices         = ticker_obj.historical_prices

        try:
            ticker_obj.get_ratios(date_updated)
            print('Ratios successful')
            ticker_obj.plot_single_timeseries(plot=False, earnings_show=True)
            ticker_obj.map_earnings_to_prices()

            self.fundamentals   = ticker_obj.summary_data
            self.earning_dates  = ticker_obj.earning_dates
            
        except Exception as e:
            print(f"Failed to get earning dates because of : {e}")


    def track_bullish_trend(
        self,
        *,
        kind: str = "price",        # "price" or "returns"
        slope_window: int = 63,     # lookback bars for linear trend on log-price
        ma_window: int = 50,        # moving average window
        confirm_with_ma: bool = True,
        dd_max: float | None = None,# e.g. 0.2 to cap drawdown at -20%
        slope_thresh: float = 0.0   # min growth per bar to call it bullish
    ) -> pd.DataFrame:
        """
        Compute rolling trend features and a boolean 'bullish' regime flag.
        """
        s = self.prices['close']
        s = s.dropna().astype(float).copy()
        if s.empty:
            return pd.DataFrame(index=s.index)

        # Ensure a price series
        if kind.lower() == "returns":
            p = (1.0 + s.fillna(0.0)).cumprod()
        elif kind.lower() == "price":
            p = s.copy()
        else:
            raise ValueError("kind must be 'price' or 'returns'")

        p = p.replace([np.inf, -np.inf], np.nan).dropna()
        lp = np.log(p)
        t = pd.Series(np.arange(len(lp)), index=lp.index, dtype=float)

        # Rolling linear trend of log-price on time
        cov_lt = lp.rolling(slope_window, min_periods=slope_window).cov(t)
        var_t  = t.rolling(slope_window, min_periods=slope_window).var()
        slope  = cov_lt / var_t                               # log-price slope per bar
        growth_per_bar = np.exp(slope) - 1                    # implied % growth per bar
        corr_lt = lp.rolling(slope_window, min_periods=slope_window).corr(t)
        r2 = corr_lt**2

        # Confirmation features
        ma = p.rolling(ma_window, min_periods=ma_window).mean()
        above_ma = p > ma

        cummax = p.cummax()
        drawdown = p / cummax - 1.0

        bullish = (growth_per_bar > slope_thresh)
        if confirm_with_ma:
            bullish = bullish & above_ma
        if dd_max is not None:
            bullish = bullish & (drawdown >= -float(dd_max))
        bullish = bullish.fillna(False)

        # Run-length of bullish regime
        group_id = (bullish != bullish.shift()).cumsum()
        run_len = bullish.groupby(group_id).cumcount() + 1
        run_len = run_len.where(bullish, 0)

        out = pd.DataFrame({
            "price": p,
            "log_price": lp,
            "trend_slope_log": slope,
            "trend_growth_per_bar": growth_per_bar,
            "trend_r2": r2,
            "ma": ma,
            "above_ma": above_ma,
            "drawdown": drawdown,
            "bullish": bullish,
            "bullish_run_length": run_len,
        })

        out["bullish_start"] = bullish & ~bullish.shift(1, fill_value=False)
        out["bullish_end"]   = ~bullish & bullish.shift(1, fill_value=False)
        
        self.ta_indicators1 = out 
        
        # Call the plotting function
        self.plot_bullish_visuals()

    def _bullish_intervals(self, col_name : str):
        """
        List of (start_ts, end_ts) intervals where b==True
        """
        b = self.ta_indicators1['bullish']
        if b.empty:
            return []
        on = b.fillna(False)
        flip = on.ne(on.shift(fill_value=False))
        starts = on.index[flip & on]
        ends   = on.index[flip.shift(-1, fill_value=False) & ~on.shift(-1, fill_value=False)]
        if len(starts) and (len(ends) == 0 or starts[-1] > ends[-1]):
            ends = ends.append(pd.Index([on.index[-1]]))

        self.bullish_intervals =  list(zip(starts, ends))

        return list(zip(starts, ends))

    def plot_bullish_visuals(self, *, slope_thresh: float = 0.0):
        """
        Three simple figures:
         1.  price+MA with shaded bullish regimes
         2. trend growth
         3. drawdown.
        """
        title_prefix = self.ticker 
        out          = self.ta_indicators1

        # 1) Price & MA with bullish shading
        plt.figure(figsize=(12, 4.5))
        plt.plot(out.index, out["price"], label="Price")
        plt.plot(out.index, out["ma"], label="MA")
        for start, end in self._bullish_intervals('bullish'):
            plt.axvspan(start, end, alpha=0.15)  # shaded bullish windows
        plt.title(f"{title_prefix}Price with Bullish Regimes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 2) Trend growth per bar with threshold
        plt.figure(figsize=(12, 3.5))
        plt.plot(out.index, out["trend_growth_per_bar"], label="Trend growth per bar")
        plt.axhline(y=slope_thresh, linestyle="--", label="Threshold")
        plt.title(f"{title_prefix}Trend Growth per Bar")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 3) Drawdown
        plt.figure(figsize=(12, 3.5))
        plt.plot(out.index, out["drawdown"], label="Drawdown")
        plt.title(f"{title_prefix}Drawdown")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def metric_glossary() -> dict[str, str]:
        """Concise explanations for each output column."""
        return {
            "price": "Input price (or cumprod of returns).",
            "log_price": "Natural log of price (linearizes compounding).",
            "trend_slope_log": "Rolling regression slope of log-price vs time (log-return per bar).",
            "trend_growth_per_bar": "exp(slope)−1; implied % growth per bar from the trend.",
            "trend_r2": "Squared correlation of log-price with time; straightness of trend (0–1).",
            "ma": "Moving average used as confirmation.",
            "above_ma": "True when price > MA.",
            "drawdown": "Price / running high − 1; depth below peak.",
            "bullish": "Final regime flag after filters (trend, MA, drawdown).",
            "bullish_run_length": "Consecutive bars in the current bullish regime.",
            "bullish_start": "Marker for the first bar of a bullish regime.",
            "bullish_end": "Marker for the first bar after a bullish regime ends.",
        }

    def _compute_indicators(
        self,
        *,
        sma: tuple[int, int] = (20, 50),
        ema: tuple[int, int] = (12, 26),
        bb_window: int = 20,
        bb_k: float = 2.0,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_window: int = 14,
        stoch_smooth: int = 3,
    ) -> pd.DataFrame:
        """
        Compute the minimal indicator set needed for the two plots.
        Works with whatever columns exist in self.prices.
        """
        df = self.prices.copy()
        if "close" not in df:
            raise ValueError("self.prices must include a 'close' column.")

        c = df["close"].astype(float)

        # --- SMAs / EMAs ---
        s_short, s_long = sma
        e_short, e_long = ema
        df[f"SMA_{s_short}"] = c.rolling(s_short, min_periods=s_short).mean()
        df[f"SMA_{s_long}"]  = c.rolling(s_long,  min_periods=s_long).mean()
        df[f"EMA_{e_short}"] = c.ewm(span=e_short, adjust=False, min_periods=e_short).mean()
        df[f"EMA_{e_long}"]  = c.ewm(span=e_long,  adjust=False, min_periods=e_long).mean()

        # --- Bollinger Bands (on close) ---
        bb_mid = c.rolling(bb_window, min_periods=bb_window).mean()
        bb_std = c.rolling(bb_window, min_periods=bb_window).std()
        df[f"BB_mid_{bb_window}"]   = bb_mid
        df[f"BB_upper_{bb_window}"] = bb_mid + bb_k * bb_std
        df[f"BB_lower_{bb_window}"] = bb_mid - bb_k * bb_std

        # --- RSI (Wilder) ---
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI_{rsi_window}"] = 100 - (100 / (1 + rs))

        # --- MACD ---
        ema_fast = c.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean()
        ema_slow = c.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        macd_sig  = macd_line.ewm(span=macd_signal, adjust=False, min_periods=macd_signal).mean()
        df[f"MACD_{macd_fast}_{macd_slow}"] = macd_line
        df[f"MACDsig_{macd_signal}"]        = macd_sig
        df["MACD_hist"]                     = macd_line - macd_sig

        # --- Stochastic %K/%D (needs high/low) ---
        if {"high", "low"}.issubset(df.columns):
            ll = df["low"].rolling(stoch_window, min_periods=stoch_window).min()
            hh = df["high"].rolling(stoch_window, min_periods=stoch_window).max()
            k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
            df[f"STOCH_%K_{stoch_window}"] = k.rolling(stoch_smooth, min_periods=stoch_smooth).mean()
            df[f"STOCH_%D_{stoch_smooth}"] = (
                df[f"STOCH_%K_{stoch_window}"].rolling(stoch_smooth, min_periods=stoch_smooth).mean()
            )

        return df

    def _get_earnings_dates(self) -> pd.DatetimeIndex:
        """
        Return unique earnings dates as a DatetimeIndex. Robust to:
        - self.earning_dates / self.earnings_dates / self.earnings_date
        - DataFrame with 'earnings_date' column, Series/list/scalar, or missing.
        """
        possible_attrs = ("earning_dates", "earnings_dates", "earnings_date")
        for name in possible_attrs:
            if hasattr(self, name):
                obj = getattr(self, name)
                try:
                    if isinstance(obj, pd.DataFrame):
                        col = "earnings_date" if "earnings_date" in obj.columns else next(
                            (c for c in obj.columns if "earn" in str(c).lower()), None
                        )
                        vals = obj[col] if col is not None else pd.Series([], dtype="datetime64[ns]")
                    elif isinstance(obj, (pd.Series, list, tuple, np.ndarray)):
                        vals = obj
                    else:  # single value
                        vals = [obj]
                    ts = pd.to_datetime(pd.Series(vals).dropna(), errors="coerce").dropna()
                    return pd.DatetimeIndex(ts.unique().sort_values())
                except Exception:
                    pass
        return pd.DatetimeIndex([])

    def _align_earnings_to_index(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Map earnings dates to the nearest timestamps in idx (handles non-trading days).
        Returns a DatetimeIndex subset of idx (unique).
        """
        if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
            return pd.DatetimeIndex([])
        ed = self._get_earnings_dates()
        if ed.empty:
            return pd.DatetimeIndex([])
        # nearest trading bars
        pos = idx.get_indexer(ed, method="nearest")
        pos = pos[pos >= 0]
        return pd.DatetimeIndex(idx[pos]).unique().sort_values()

    def _draw_price_chart(self, df, *, sma, ema, bb_window, bb_k, show_legend, title_tag):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        # --- Price panel ---
        ax = axs[0]
        ax.plot(df.index, df["close"], label="Close")
        ax.plot(df.index, df[f"SMA_{sma[0]}"], label=f"SMA {sma[0]}")
        ax.plot(df.index, df[f"SMA_{sma[1]}"], label=f"SMA {sma[1]}")
        ax.plot(df.index, df[f"EMA_{ema[0]}"], label=f"EMA {ema[0]}", linestyle="--", alpha=0.8)
        ax.plot(df.index, df[f"EMA_{ema[1]}"], label=f"EMA {ema[1]}", linestyle="--", alpha=0.8)

        mid = f"BB_mid_{bb_window}"
        up  = f"BB_upper_{bb_window}"
        lo  = f"BB_lower_{bb_window}"
        ax.fill_between(df.index, df[lo], df[up], alpha=0.12, label=f"BBands {bb_window}")

        # Earnings crosses (on price)
        edx = self._align_earnings_to_index(df.index)
        if len(edx) > 0:
            y_price = df.loc[edx, "close"]
            ax.scatter(edx, y_price, marker="x", s=60, zorder=5, label="Earnings")

        ax.set_title(f"{self.ticker} • Price / BB / SMA / EMA {title_tag}")
        ax.grid(True)
        if show_legend:
            ax.legend(loc="upper left")

        # --- Volume panel ---
        axv = axs[1]
        if "volume" in df.columns:
            axv.bar(df.index, df["volume"].fillna(0.0), width=1.0, label="Volume", alpha=0.8)
            # Earnings crosses on volume (optional)
            if len(edx) > 0:
                yv = df.loc[edx, "volume"].fillna(0.0)
                axv.scatter(edx, yv, marker="x", s=45, zorder=5, label="Earnings")
            axv.set_title("Volume")
            if show_legend:
                axv.legend(loc="upper left")
        else:
            axv.text(0.5, 0.5, "No 'volume' column", ha="center", va="center", transform=axv.transAxes)
            axv.set_title("Volume (missing)")
        axv.grid(True)

        plt.tight_layout()
        return fig, axs

    def plot_price_chart(
        self,
        *,
        sma: tuple[int, int] = (20, 50),
        ema: tuple[int, int] = (12, 26),
        bb_window: int = 20,
        bb_k: float = 2.0,
        lookback: int | None = None,
        show_legend: bool = True,
    ):
        """
        Draw two price charts:
        1) Full dataset
        2) (Optional) Zoomed: last `lookback` rows (trading days)
        Returns (fig_full, axs_full) if lookback is None,
                else (fig_full, axs_full, fig_zoom, axs_zoom).
        """
        df = self._compute_indicators(sma=sma, ema=ema, bb_window=bb_window, bb_k=bb_k)

        # Full
        fig_full, axs_full = self._draw_price_chart(
            df, sma=sma, ema=ema, bb_window=bb_window, bb_k=bb_k,
            show_legend=show_legend, title_tag="(Full)"
        )

        if not lookback:
            return (fig_full, axs_full)

        # Zoomed (last N trading bars)
        df_zoom = df.tail(int(lookback))
        fig_zoom, axs_zoom = self._draw_price_chart(
            df_zoom, sma=sma, ema=ema, bb_window=bb_window, bb_k=bb_k,
            show_legend=show_legend, title_tag=f"(Zoom: last {int(lookback)} bars)"
        )
        return (fig_full, axs_full, fig_zoom, axs_zoom)

    def _draw_mom_panels(self, df, *, rsi_window, macd_fast, macd_slow, macd_signal,
                        include_stoch, stoch_window, stoch_smooth, show_legend, title_tag):
        import matplotlib.pyplot as plt

        panels = ["rsi", "macd"]
        has_stoch = include_stoch and (f"STOCH_%K_{stoch_window}" in df) and (f"STOCH_%D_{stoch_smooth}" in df)
        if has_stoch:
            panels.append("stoch")

        fig, axs = plt.subplots(len(panels), 1, figsize=(12, 2.8 * len(panels)), sharex=True)
        if len(panels) == 1:
            axs = [axs]
        i = 0

        edx = self._align_earnings_to_index(df.index)

        # --- RSI ---
        rsi_col = f"RSI_{rsi_window}"
        ax = axs[i]; i += 1
        ax.plot(df.index, df[rsi_col], label=rsi_col)
        if len(edx) > 0:
            y = df.loc[edx, rsi_col]
            y = y[y.notna()]
            if not y.empty:
                ax.scatter(y.index, y.values, marker="x", s=45, zorder=5, label="Earnings")
        ax.axhline(70, linestyle="--", linewidth=1)
        ax.axhline(30, linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_title(f"{self.ticker} • RSI {title_tag}")
        ax.grid(True)
        if show_legend: ax.legend(loc="upper left")

        # --- MACD ---
        macd_col = f"MACD_{macd_fast}_{macd_slow}"
        sig_col  = f"MACDsig_{macd_signal}"
        hist_col = "MACD_hist"
        ax = axs[i]; i += 1
        ax.plot(df.index, df[macd_col], label="MACD")
        ax.plot(df.index, df[sig_col], label="Signal", linestyle="--")
        if hist_col in df:
            ax.bar(df.index, df[hist_col], width=1.0, alpha=0.3, label="Hist")
        if len(edx) > 0:
            y = df.loc[edx, macd_col]
            y = y[y.notna()]
            if not y.empty:
                ax.scatter(y.index, y.values, marker="x", s=45, zorder=5, label="Earnings")
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{self.ticker} • MACD {title_tag}")
        ax.grid(True)
        if show_legend: ax.legend(loc="upper left")

        # --- Stochastic (optional) ---
        if has_stoch:
            k_col = f"STOCH_%K_{stoch_window}"
            d_col = f"STOCH_%D_{stoch_smooth}"
            ax = axs[i]
            ax.plot(df.index, df[k_col], label="%K")
            ax.plot(df.index, df[d_col], label="%D", linestyle="--")
            if len(edx) > 0:
                y = df.loc[edx, k_col]
                y = y[y.notna()]
                if not y.empty:
                    ax.scatter(y.index, y.values, marker="x", s=45, zorder=5, label="Earnings")
            ax.axhline(80, linestyle="--", linewidth=1)
            ax.axhline(20, linestyle="--", linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_title(f"{self.ticker} • Stochastic {title_tag}")
            ax.grid(True)
            if show_legend: ax.legend(loc="upper left")

        plt.tight_layout()
        return fig, axs

    def plot_mom_indicators(
        self,
        *,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        include_stoch: bool = True,
        stoch_window: int = 14,
        stoch_smooth: int = 3,
        lookback: int | None = None,
        show_legend: bool = True,
    ):
        """
        Draw momentum indicator charts:
        1) Full dataset
        2) (Optional) Zoomed: last `lookback` rows (trading days)
        Returns (fig_full, axs_full) if lookback is None,
                else (fig_full, axs_full, fig_zoom, axs_zoom).
        """
        df = self._compute_indicators(
            rsi_window=rsi_window,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            stoch_window=stoch_window, stoch_smooth=stoch_smooth
        )

        # Full
        fig_full, axs_full = self._draw_mom_panels(
            df,
            rsi_window=rsi_window, macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            include_stoch=include_stoch, stoch_window=stoch_window, stoch_smooth=stoch_smooth,
            show_legend=show_legend, title_tag="(Full)"
        )

        if not lookback:
            return (fig_full, axs_full)

        # Zoomed
        df_zoom = df.tail(int(lookback))
        fig_zoom, axs_zoom = self._draw_mom_panels(
            df_zoom,
            rsi_window=rsi_window, macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            include_stoch=include_stoch, stoch_window=stoch_window, stoch_smooth=stoch_smooth,
            show_legend=show_legend, title_tag=f"(Zoom: last {int(lookback)} bars)"
        )
        return (fig_full, axs_full, fig_zoom, axs_zoom)


class MultipleTA:
    """
    Manage multiple TA objects for cross-ticker visualisations:
      - Overlapping bullish regimes (background shading by overlap count)
      - Dynamic conditional rolling correlation
      - Momentum indicators (RSI & MACD) in one figure
    """

    def __init__(self, tickers: list[str], *, date=None, date_updated=False, bench_default: str = "SPY"):
        """
        tickers: list of ticker strings
        date:    end date (defaults to today)
        date_updated: pass through to TA
        bench_default: used if a comparison ticker isn't specified
        """
        self.tickers = list(tickers)

        if date is None:
            date = dt.datetime.today()
        self.ta = {t: TA(t, date=date, date_updated=date_updated) for t in self.tickers}

        if isinstance(bench_default, str):
            if bench_default == "SPX" or bench_default == "SPY":
                self.bench_default = self.get_benchmark_series("SPX")
            if bench_default == "Q" or bench_default == "NDQ":
                self.bench_default = self.get_benchmark_series("Q")

        # convenience cache: closing prices aligned on inner dates
        self._close = self._build_close_df(self.ta)

    @staticmethod
    def get_benchmark_series(ticker : str):
        if ticker == "Q":
            NDQ = fred_api.get_series('NASDAQ100', observation_start = filing_date, observation_end= dt.datetime.today()).to_frame('Q')

            return NDQ
        else:
            SPX = fred_api.get_series('SP500', observation_start = filing_date, observation_end=dt.datetime.today()).to_frame('SPX')
            return SPX
    @staticmethod
    def _log_ret(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return np.log(s).diff()

    @staticmethod
    def _build_close_df(ta_dict: dict[str, "TA"]) -> pd.DataFrame:
        """Inner-join close prices across all tickers; columns named by ticker."""
        df_list = []
        for t, obj in ta_dict.items():
            if "close" not in obj.prices.columns:
                continue
            c = obj.prices["close"].astype(float).rename(t)
            df_list.append(c)
        if not df_list:
            raise ValueError("No valid 'close' series found.")
        close = pd.concat(df_list, axis=1, join="inner").dropna(how="all")
        return close

    @staticmethod
    def _bullish_from_series(
        s: pd.Series,
        *,
        kind: str = "price",
        slope_window: int = 63,
        ma_window: int = 50,
        confirm_with_ma: bool = True,
        dd_max: float | None = None,
        slope_thresh: float = 0.0
    ) -> pd.Series:
        """Compute bullish boolean from a single price series (same logic as your TA)."""
        s = s.dropna().astype(float)
        if kind == "returns":
            p = (1.0 + s.fillna(0.0)).cumprod()
        elif kind == "price":
            p = s.copy()
        else:
            raise ValueError("kind must be 'price' or 'returns'")
        p = p.replace([np.inf, -np.inf], np.nan).dropna()

        lp = np.log(p)
        t = pd.Series(np.arange(len(lp)), index=lp.index, dtype=float)
        cov_lt = lp.rolling(slope_window, min_periods=slope_window).cov(t)
        var_t  = t.rolling(slope_window, min_periods=slope_window).var()
        slope  = cov_lt / var_t
        growth_per_bar = np.exp(slope) - 1
        ma = p.rolling(ma_window, min_periods=ma_window).mean()
        above_ma = p > ma
        drawdown = p / p.cummax() - 1.0

        bull = (growth_per_bar > slope_thresh)
        if confirm_with_ma:
            bull = bull & above_ma
        if dd_max is not None:
            bull = bull & (drawdown >= -float(dd_max))
        return bull.reindex(p.index).fillna(False)

    def _ensure_close_and_index(self, lookback: int | None = None) -> pd.DataFrame:
        """Return inner-joined close prices; optionally last `lookback` rows."""
        close = self._close.copy()
        if lookback:
            close = close.tail(int(lookback))
        return close

    # ---------- public plots ----------

    def plot_bullish_trends(
        self,
        *,
        slope_window: int = 63,
        ma_window: int = 50,
        confirm_with_ma: bool = True,
        dd_max: float | None = None,
        slope_thresh: float = 0.0,
        normalize: bool = True,
        lookback: int | None = None,
        show_legend: bool = True,
    ):
        """
        Overlay all tickers' prices and shade background by how many tickers are bullish.
        """
        close = self._ensure_close_and_index(lookback=lookback)

        # Compute bullish boolean per ticker on aligned index
        bull_df = pd.DataFrame(index=close.index)
        for t in close.columns:
            bull_df[t] = self._bullish_from_series(
                close[t],
                kind="price",
                slope_window=slope_window,
                ma_window=ma_window,
                confirm_with_ma=confirm_with_ma,
                dd_max=dd_max,
                slope_thresh=slope_thresh,
            )

        # Count overlaps
        overlap = bull_df.sum(axis=1)  # 0..N
        nmax = int(overlap.max())

        # Prepare plotting prices
        if normalize:
            price_to_plot = close / close.iloc[0]
        else:
            price_to_plot = close

        fig, ax = plt.subplots(figsize=(13, 6))
        # Price lines per ticker
        for t in price_to_plot.columns:
            ax.plot(price_to_plot.index, price_to_plot[t], label=t)

        # Shade by overlap intensity (discrete colormap steps)
        if nmax > 0:
            cmap = plt.cm.get_cmap("Greens", nmax + 1)  # 0..nmax
            # Walk through runs of constant overlap
            run_id = (overlap != overlap.shift()).cumsum()
            for gid, seg in overlap.groupby(run_id):
                k = int(seg.iloc[0])
                if k <= 0:
                    continue
                start, end = seg.index[0], seg.index[-1]
                ax.axvspan(start, end, color=cmap(k), alpha=0.20, label=None)

            # Add a small legend patch for overlap counts
            from matplotlib.patches import Patch
            patches = [Patch(facecolor=plt.cm.get_cmap("Greens", nmax + 1)(k), alpha=0.20,
                             label=f"{k} bullish") for k in range(1, nmax + 1)]
            # Avoid duplicate legend entries with same labels as tickers
            handles, labels = ax.get_legend_handles_labels()
            if show_legend:
                ax.legend(handles + patches, labels + [p.get_label() for p in patches],
                          loc="upper left", ncol=2)

        ax.set_title("Bullish Overlaps (shading = number of tickers bullish)")
        ax.grid(True)
        plt.tight_layout()
        return fig, ax

    def plot_dynamic_conditional_correlation(
        self,
        ticker_a: str,
        ticker_b: str | None = None,
        *,
        price : str = 'close',
        window: int = 63,
        ref: str = "b",       # which series' sign to condition on: "a" or "b"
        lookback: int | None = None,
        min_points: int = 10  # minimum points inside condition to compute corr
    ):
        """
        Rolling correlation of log-returns, split conditionally by sign of ref's return.
        Plots three lines:
        - Unconditional rolling correlation
        - Conditional (ref>0) rolling correlation
        - Conditional (ref<0) rolling correlation
        Default comparison is against 'SPY' if ticker_b is None.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if ticker_b is None:
            ticker_b = self.bench_default.columns[0]
            df = pd.concat(
                [
                    self.ta[ticker_a].prices[price].rename(ticker_a).astype(float),
                    self.bench_default[ticker_b].rename(ticker_b).astype(float),
                ],
                axis=1,
                join="inner",
            ).dropna()

        else:
            df = pd.concat(
                [
                    self.ta[ticker_a].prices[price].rename(ticker_a).astype(float),
                    self.ta[ticker_b].prices[price].rename(ticker_b).astype(float),
                ],
                axis=1,
                join="inner",
            ).dropna()

        if lookback:
            df = df.tail(int(lookback))

        # Log returns
        rA = self._log_ret(df[ticker_a]).dropna()
        rB = self._log_ret(df[ticker_b]).dropna()
        ret = pd.concat([rA, rB], axis=1, join="inner").dropna()
        ret.columns = ["A", "B"]

        if len(ret) < window + 5:
            raise ValueError("Not enough data for the chosen window.")

        ref_col = "A" if ref.lower().startswith("a") else "B"

        # --- Unconditional rolling correlation (fast, built-in) ---
        rc = ret["A"].rolling(window).corr(ret["B"])

        # --- Conditional rolling correlation helpers ---
        def _roll_apply_df(df2: pd.DataFrame, win: int, func) -> pd.Series:
            out = pd.Series(index=df2.index, dtype=float)
            if len(df2) < win:
                return out
            # iterate windows; assign result at window end
            for i in range(win - 1, len(df2)):
                wdf = df2.iloc[i - win + 1 : i + 1]
                out.iloc[i] = func(wdf)
            return out

        def _cond_corr(wdf: pd.DataFrame, sign: int) -> float:
            # sign=+1 -> ref>0; sign=-1 -> ref<0
            cond = wdf[ref_col] > 0 if sign > 0 else wdf[ref_col] < 0
            xs = wdf.loc[cond]
            if len(xs) < min_points:
                return np.nan
            return xs["A"].corr(xs["B"])

        rcp = _roll_apply_df(ret, window, lambda w: _cond_corr(w, +1))  # ref>0
        rcn = _roll_apply_df(ret, window, lambda w: _cond_corr(w, -1))  # ref<0

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rc.index, rc.values, label=f"Rolling corr ({window})")
        ax.plot(rcp.index, rcp.values, label=f"Cond corr ({ref_col}>0)")
        ax.plot(rcn.index, rcn.values, label=f"Cond corr ({ref_col}<0)")
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_title(f"Dynamic Conditional Correlation: {ticker_a} vs {ticker_b}\n on {price} data")
        ax.grid(True)
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig, ax

    def plot_momentum_indicators(
        self,
        *,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback: int | None = None,
        include_stoch: bool = False,   # keep off by default to avoid clutter
        stoch_window: int = 14,
        stoch_smooth: int = 3,
        show_legend: bool = True,
    ):
        """
        Single figure; top = RSI overlay, bottom = MACD overlay for all tickers.
        (Optional Stochastic overlay if include_stoch=True and high/low exist.)
        """
        # Build per-ticker indicator frames and align on common index
        rsi_map = {}
        macd_map = {}
        stochK_map = {}
        for t, obj in self.ta.items():
            df_ind = obj._compute_indicators(
                rsi_window=rsi_window,
                macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                stoch_window=stoch_window, stoch_smooth=stoch_smooth
            )
            if lookback:
                df_ind = df_ind.tail(int(lookback))
            rsi_col = f"RSI_{rsi_window}"
            macd_col = f"MACD_{macd_fast}_{macd_slow}"
            rsi_map[t] = df_ind[rsi_col]
            macd_map[t] = df_ind[macd_col]
            if include_stoch and f"STOCH_%K_{stoch_window}" in df_ind:
                stochK_map[t] = df_ind[f"STOCH_%K_{stoch_window}"]

        # Align by inner join on index
        def align_map(d):
            if not d:
                return pd.DataFrame()
            return pd.concat(d.values(), axis=1, join="inner").set_axis(list(d.keys()), axis=1)

        rsi_df = align_map(rsi_map)
        macd_df = align_map(macd_map)
        stoch_df = align_map(stochK_map) if include_stoch and stochK_map else None

        nrows = 3 if (stoch_df is not None and not stoch_df.empty) else 2
        fig, axs = plt.subplots(nrows, 1, figsize=(12, 2.8*nrows), sharex=True)

        # RSI
        ax = axs[0]
        for t in rsi_df.columns:
            ax.plot(rsi_df.index, rsi_df[t], label=t)
        ax.axhline(70, linestyle="--", linewidth=1)
        ax.axhline(30, linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_title("RSI overlay")
        ax.grid(True)
        if show_legend:
            ax.legend(loc="upper left", ncol=2)

        # MACD
        ax = axs[1]
        for t in macd_df.columns:
            ax.plot(macd_df.index, macd_df[t], label=t)
        ax.axhline(0, linewidth=1)
        ax.set_title("MACD overlay")
        ax.grid(True)

        # Stochastic (optional)
        if nrows == 3:
            ax = axs[2]
            for t in stoch_df.columns:
                ax.plot(stoch_df.index, stoch_df[t], label=t)
            ax.axhline(80, linestyle="--", linewidth=1)
            ax.axhline(20, linestyle="--", linewidth=1)
            ax.set_ylim(0, 100)
            ax.set_title("Stochastic %K overlay")
            ax.grid(True)

        plt.tight_layout()
        return fig, axs



if __name__ == "__main__":

    # Build for a few tickers
    mta = MultipleTA(["NVDA", "CRDO", "AMD"])

    mta.plot_bullish_trends(normalize=True)
    mta.plot_bullish_trends(normalize=True, lookback=90)
    mta.plot_dynamic_conditional_correlation("CRDO", window=63, ref="b", lookback=252)

    # # Or explicitly vs MSFT, conditioning on AAPL's sign:
    mta.plot_dynamic_conditional_correlation("CRDO", "NVDA", window=63, ref="a", lookback=252)

    # 3) Momentum indicators for all tickers (full), with Stochastic too
    mta.plot_momentum_indicators(rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9,
                              include_stoch=True, lookback=180)

    
