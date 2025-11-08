from stock_data_functions import TickerComparison, TickerData, calc_log_rets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Optional, Dict, Any, Sequence, Tuple
import math
from typing import Optional, List, Dict, Any, Tuple

############################# Helper Functions #############################

def _align_asset_and_factors(asset_ret: pd.Series, factor_ret) -> Tuple[pd.Series, pd.DataFrame]:
    """Align asset and factor returns; coerce factors to DataFrame; drop any rows with NaN across all."""
    if isinstance(factor_ret, pd.Series):
        F = factor_ret.to_frame(name=factor_ret.name or "factor")
    elif isinstance(factor_ret, pd.DataFrame):
        F = factor_ret.copy()
    else:
        raise TypeError("factor_ret must be a pandas Series or DataFrame.")
    df = pd.concat([asset_ret.rename("x"), F], axis=1, join="inner").dropna(how="any")
    if df.empty:
        return asset_ret.iloc[0:0], F.iloc[0:0]
    x = df["x"]
    F = df.drop(columns=["x"])
    return x, F

def _orthogonalize_static(F: pd.DataFrame, order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Static sequential orthogonalization: residualize each factor against all previous ones (no intercept).
    Keeps the same columns (and order if 'order' provided). Suitable for interpretable 'incremental' betas.
    """
    if order is None:
        order = list(F.columns)
    else:
        missing = [c for c in order if c not in F.columns]
        if missing:
            raise ValueError(f"orthogonalize order has missing columns: {missing}")
        F = F[order]
    R = pd.DataFrame(index=F.index, columns=F.columns, dtype=float)
    prev_cols: List[str] = []
    for col in F.columns:
        y = F[col]
        if not prev_cols:
            R[col] = y.astype(float)
            prev_cols = [col]
            continue
        X = F[prev_cols]
        M = pd.concat([X, y], axis=1).dropna()
        if M.empty or X.shape[1] == 0:
            R[col] = np.nan
            prev_cols.append(col)
            continue
        Xn = M[prev_cols].to_numpy()
        yn = M[col].to_numpy()
        beta_base, *_ = np.linalg.lstsq(Xn, yn, rcond=None)  # OLS, no intercept
        res = y.copy().astype(float)
        fitted = pd.Series(index=M.index, data=Xn @ beta_base)
        res.loc[M.index] = yn - fitted.values
        R[col] = res
        prev_cols.append(col)
    return R

def _drop_weekends_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        # Your invariants use US/Eastern; make it explicit if naive
        idx_e = idx.tz_localize("US/Eastern")
    else:
        idx_e = idx.tz_convert("US/Eastern")

    mask = idx_e.weekday < 5   # 0=Mon, ..., 6=Sun
    return idx[mask]

def _select_ext_hours_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # US/Eastern 04:30 ≤ time < 21:00 (post-market close per your convention)
    t = idx.tz_convert("US/Eastern").time
    return idx[(t >= dt.time(4,30)) & (t < dt.time(20,0))]

def _coerce_betas_dict_to_df(optimal_betas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Each value is a daily-indexed DataFrame of betas with columns=selected factor tickers
    for that stock. We concatenate to a single wide DataFrame with MultiIndex columns:
    (stock, factor). This tends to play well with downstream alignment code.
    """
    pieces = []
    for tkr, df in optimal_betas.items():
        tmp = df.copy()
        # ensure columns are strings (factor tickers) and create a (stock, factor) MultiIndex
        tmp.columns = pd.MultiIndex.from_product([[tkr], list(map(str, tmp.columns))])
        pieces.append(tmp)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).sort_index()
    # if you prefer a flat column index: out.columns = [f"{s}|{f}" for s, f in out.columns]
    return out

def get_session_and_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'session' and 'trade_date' columns to df based on index timestamps.
    This is for close to close
    """
    df = df.copy() 
    t = df.index

    if t.tz is None:
        t = t.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        t = t.tz_convert('US/Eastern')
    tt = t.time
    df['session'] = np.select(
        [
            (tt >= dt.time(4, 0))  & (tt <= dt.time(9, 30)),   # Pre-Market
            (tt > dt.time(9, 30)) & (tt <= dt.time(16, 0)),   # Regular
            (tt > dt.time(16, 0)) & (tt < dt.time(20, 0)),   # Post-Market
        ],
        ['Pre-Market', 'Regular', 'Post-Market'],
        default='closed'
    )
    et_date          = pd.Series(t.date,index=df.index)
    df['trade_date'] = np.where(df['session'] == 'Post-Market',
                                (t + pd.Timedelta(days=1)).date,
                                et_date)
    
    return df

################################### pipeline #####################################

def ewma_beta(
    asset_ret: pd.Series,
    factor_ret: pd.Series | pd.DataFrame,
    lam: float = 0.94,
    method: str = "ridge",                 # "ridge" | "orthogonalize"
    ridge: float = 0.0,                    # default 0.0 => exact legacy behavior for single-factor
    orth_order: Optional[List[str]] = None # only used if method="orthogonalize"
) -> pd.Series | pd.DataFrame:
    """
    Exponentially-weighted rolling beta (RiskMetrics-style), now supporting multiple factors and two methods:
      - method="ridge": EWMA multi-variate beta with small l2 stabilization (keep original factors).
      - method="orthogonalize": sequentially residualize factors for incremental 'excess-over-base' exposures,
                                then run the same EWMA on the residualized factors.

    Returns:
      - Series if a single factor is passed (backward compatible),
      - DataFrame (columns = factor names) if multiple factors are passed.
    """
    # 0) align
    x, F = _align_asset_and_factors(asset_ret, factor_ret)
    if F.empty or x.empty:
        if isinstance(factor_ret, pd.Series):
            return pd.Series(dtype=float, index=asset_ret.index, name="beta_ewma")
        else:
            return pd.DataFrame(index=asset_ret.index, columns=getattr(factor_ret, "columns", None), dtype=float)

    idx = x.index
    cols = F.columns.tolist()

    # 1) optional orthogonalization
    if method.lower() in ("orthogonalize", "orthogonalise", "ortho"):
        F_use = _orthogonalize_static(F, order=orth_order)
    elif method.lower() == "ridge":
        F_use = F
    else:
        raise ValueError("method must be 'ridge' or 'orthogonalize'.")

    T, K = F_use.shape

    # 2) single-factor fast path
    if K == 1:
        f = F_use.iloc[:, 0].to_numpy()
        xn = x.to_numpy()
        cov_vals = np.zeros(T); var_vals = np.zeros(T)
        beta_vals = np.full(T, np.nan)
        lam1 = 1.0 - lam
        cov_vals[0] = lam1 * xn[0] * f[0]
        var_vals[0] = lam1 * f[0] * f[0]
        beta_vals[0] = cov_vals[0] / (var_vals[0] + ridge) if (var_vals[0] + ridge) != 0 else np.nan
        for t in range(1, T):
            xt, ft = xn[t], f[t]
            cov_vals[t] = lam * cov_vals[t-1] + lam1 * (xt * ft)
            var_vals[t] = lam * var_vals[t-1] + lam1 * (ft * ft)
            denom = var_vals[t] + ridge
            beta_vals[t] = cov_vals[t] / denom if denom != 0 else np.nan
        out = pd.Series(beta_vals, index=idx, name="beta_ewma")

    # 3) multi-factor recursion
    else:
        X = x.to_numpy()
        Fm = F_use.to_numpy()           # T x K
        Q = np.zeros((K, K), float)     # EWMA factor covariance
        c = np.zeros(K, float)          # EWMA cross-cov with asset
        betas = np.full((T, K), np.nan, float)
        lam1 = 1.0 - lam
        I = np.eye(K)
        for t in range(T):
            f_t = Fm[t, :]
            x_t = X[t]
            Q = lam * Q + lam1 * np.outer(f_t, f_t)
            c = lam * c + lam1 * (f_t * x_t)
            A = Q + ridge * I
            try:
                beta_t = np.linalg.solve(A, c)
            except np.linalg.LinAlgError:
                beta_t = np.full(K, np.nan)
            betas[t, :] = beta_t
        out = pd.DataFrame(betas, index=idx, columns=cols)

    # 4) reindex to original asset_ret index (backward compatible shape)
    return out.reindex(asset_ret.index)

def run_full_pipeline(
    *,
    filing_date_gte: str,
    dt_updated : bool = False,
    dt_updated_reg : bool = True,
    regressor: List[str] = None,          # candidate regressors universe (e.g., ["SPY","SMH","I:NDX"])
    stock: List[str] = None,
    # minute fetch knobs
    minute_waiting_time: int = 60,
    minute_chunksize: int = 200,
    minute_fetch_in_chunks: bool = False,
    # optional windows
    daily_start_date: Optional[str] = None,
    daily_end_date: Optional[str] = None,
    minute_start_date: Optional[str] = None,
    minute_end_date: Optional[str] = None,
    # --- NEW knobs ---
    use_optimal_betas: bool = True,
    optimal_beta_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    This function is to pull data and get optimal betas then passed for through tot he other functions.
    """
    from ewma_beta_tuning import get_optimal_betas
    # =========================
    # BUILD DATA OBJECTS
    # =========================
    stock_object_minute = TickerComparison(
        stock,
        filing_date_gte=filing_date_gte,
        period="minute",
        date_updated=dt_updated,
        fetch_in_chunks=minute_fetch_in_chunks,
        waiting_time=minute_waiting_time,
        chunksize=minute_chunksize,
        start_date=minute_start_date,
        end_date=minute_end_date,
    )
    regressor_object_minute = TickerComparison(
        regressor,
        filing_date_gte=filing_date_gte,
        period="minute",
        date_updated=dt_updated_reg,
        fetch_in_chunks=minute_fetch_in_chunks,
        waiting_time=minute_waiting_time,
        start_date=minute_start_date,
        end_date=minute_end_date,
    )

    stock_object_daily = TickerComparison(
        stock,
        filing_date_gte=filing_date_gte,
        period="day",
        date_updated=dt_updated,
        fetch_in_chunks=False,
        waiting_time=0,
        start_date=daily_start_date,
        end_date=daily_end_date,
    )
    regressor_object_daily = TickerComparison(
        regressor,
        filing_date_gte=filing_date_gte,
        period="day",
        date_updated=dt_updated_reg,
        fetch_in_chunks=False,
        waiting_time=0,
        start_date=daily_start_date,
        end_date=daily_end_date,
    )

    # =========================
    # DAILY & MINUTE RETURNS
    # =========================
    stock_returns = (
        stock_object_daily.tickers_stocks_prices
        .loc[:, (slice(None), "close")]
        .apply(calc_log_rets)
        .droplevel(1, axis=1)
    )
    regressor_returns = (
        regressor_object_daily.tickers_stocks_prices
        .loc[:, (slice(None), "close")]
        .apply(calc_log_rets)
        .droplevel(1, axis=1)
    )

    stock_returns_min = (
        stock_object_minute.tickers_stocks_prices
        .loc[:, (slice(None), "close")]
        .apply(calc_log_rets)
        .droplevel(1, axis=1)
    )
    regressor_returns_min = (
        regressor_object_minute.tickers_stocks_prices
        .loc[:, (slice(None), "close")]
        .apply(calc_log_rets)
        .droplevel(1, axis=1)
    )


    # =========================
    # BETAS (DAILY)
    # =========================
    best_params = None
    optimal_betas = None

    if use_optimal_betas:
        if optimal_beta_kwargs is None:
            raise ValueError("use_optimal_betas=True but optimal_beta_kwargs is None. "
                             "Provide factor_sets, ridge_grid, lambdas, thresh, date_updated*, and daily_start_date.")
        
        optimal_betas, best_params = get_optimal_betas(
            pass_tickercomparison_obj=True,
            stock_obj=stock_object_daily,
            regressor_obj=regressor_object_daily,
            tickers=stock,
            regressors=regressor,
            **optimal_beta_kwargs,
        )
        betas_df = _coerce_betas_dict_to_df(optimal_betas)
    else:
        return "Have Not Completed the Code for Non Optimal Betas"



    out = {
        "stock_object_minute": stock_object_minute,
        "regressor_object_minute": regressor_object_minute,
        "stock_object_daily": stock_object_daily,
        "regressor_object_daily": regressor_object_daily,
        "stock_returns_daily": stock_returns,
        "regressor_returns_daily": regressor_returns,
        "stock_returns_minute": stock_returns_min,
        "regressor_returns_minute": regressor_returns_min,
        "betas_df": betas_df,
    }
    if use_optimal_betas:
        out["best_params"] = best_params
        out["optimal_betas"] = optimal_betas
    return out

def run_full_pipeline_multifactor_r2(
    *,
    filing_date_gte: str,
    stock: Sequence[str],
    regressor: Sequence[str],
    lookback_short: int = 20,
    lookback_long: int = 60,
    # pull-through knobs
    dt_updated: bool = False,
    dt_updated_reg: bool = True,
    minute_waiting_time: int = 60,
    minute_chunksize: int = 200,
    minute_fetch_in_chunks: bool = False,
    daily_start_date: str | None = None,
    daily_end_date: str | None = None,
    minute_start_date: str | None = None,
    minute_end_date: str | None = None,
    use_optimal_betas: bool = True,
    optimal_beta_kwargs: Dict[str, any] | None = None,
) -> Dict[str, any]:

    pipe = run_full_pipeline(
        filing_date_gte=filing_date_gte,
        stock=list(stock),
        regressor=list(regressor),
        dt_updated=dt_updated,
        dt_updated_reg=dt_updated_reg,
        minute_waiting_time=minute_waiting_time,
        minute_chunksize=minute_chunksize,
        minute_fetch_in_chunks=minute_fetch_in_chunks,
        daily_start_date=daily_start_date,
        daily_end_date=daily_end_date,
        minute_start_date=minute_start_date,
        minute_end_date=minute_end_date,
        use_optimal_betas=use_optimal_betas,
        optimal_beta_kwargs=optimal_beta_kwargs,
    )

    # Build stock 5m returns with a 'trade_date' column
    stock_rets_min = pipe["stock_returns_minute"]
    labelled_stock_min = get_session_and_trade_date(stock_rets_min)   # adds 'trade_date'

    factor_5m = pipe["regressor_returns_minute"]
    betas = pipe.get("optimal_betas")  # dict[ticker]->DataFrame of daily betas (multi-factor)

    # Compute short & long rolling R² (daily index, columns = tickers)
    r2_short = rolling_r2_intraday_multifactor_perstock(
        stock_5m=labelled_stock_min, factor_5m=factor_5m, beta_daily=betas, lookback_days=lookback_short
    )
    r2_long = rolling_r2_intraday_multifactor_perstock(
        stock_5m=labelled_stock_min, factor_5m=factor_5m, beta_daily=betas, lookback_days=lookback_long
    )

    pipe["r2_short_daily"] = r2_short
    pipe["r2_long_daily"]  = r2_long
    pipe["labelled_stock_returns_minute"] = labelled_stock_min
    return pipe

def rolling_r2_intraday_multifactor_perstock(
    stock_5m: pd.DataFrame,                  # columns: 'trade_date' + tickers (5m log returns)
    factor_5m: pd.DataFrame,                 # columns: factor names (5m log returns)
    beta_daily: Dict[str, pd.DataFrame],     # {ticker -> DataFrame(index=day, cols=factors used by that ticker)}
    lookback_days: int,
    day_col: str = "trade_date",
    min_pairs: int = 120,
    dtype: str = "float64",
) -> pd.DataFrame:
    if day_col not in stock_5m.columns:
        raise KeyError(f"'{day_col}' not found in stock_5m")

    day_label = stock_5m[day_col]
    day_label = pd.to_datetime(day_label).dt.date
    tickers = [c for c in stock_5m.columns if c != day_col and c in beta_daily]
    if not tickers:
        return pd.DataFrame(index=pd.Index([], name="trading_day"))

    # distinct trading days taken from the stock panel
    unique_days = pd.Index(day_label.dropna().unique()).sort_values()
    if len(unique_days) == 0:
        return pd.DataFrame(index=pd.Index([], name="trading_day"))

    r2_by_ticker = {tic: pd.Series(index=unique_days, dtype=dtype) for tic in tickers}

    for tic in tickers:
        y = stock_5m[tic].astype(dtype)

        B_daily = beta_daily[tic]
        B_daily.index = pd.to_datetime(B_daily.index)
        cols = [f for f in B_daily.columns if f in factor_5m.columns]
        if not cols:
            continue

        # map daily betas to each minute along THIS stock's day labels
        B_5m = pd.DataFrame({f: day_label.map(B_daily[f]) for f in cols},
                            index=stock_5m.index, dtype=dtype)

        # align factor returns to THIS stock's minute grid
        F_5m = factor_5m[cols].reindex(stock_5m.index).astype(dtype)

        # prediction requires ALL factor legs present at that minute
        yhat = (F_5m * B_5m).sum(axis=1, min_count=len(cols))
        valid = y.notna() & yhat.notna()

        for i, this_day in enumerate(unique_days):
            start_i = max(0, i - lookback_days + 1)
            window_days = unique_days[start_i:i+1]
            in_window = valid & day_label.isin(window_days)

            n = int(in_window.sum())
            if n < min_pairs:
                r2_by_ticker[tic].loc[this_day] = np.nan
                continue

            yw, yhatw = y.loc[in_window], yhat.loc[in_window]
            ybar = float(yw.mean())
            sse = float(((yw - yhatw) ** 2).sum())
            sst = float(((yw - ybar)  ** 2).sum())
            r2_by_ticker[tic].loc[this_day] = np.float32(1.0 - (sse / (sst if sst != 0.0 else np.nan)))

    out = pd.DataFrame(r2_by_ticker)
    out.index.name = "trading_day"
    return out

def setup_clean_experiment(
    *,
    filing_date_gte: str,
    stock: List[str],
    regressor: List[str],                 # e.g., ["SPY"] or ["SPY","I:NDX"]; we use regressor[0] as the factor
    dt_updated: bool = False,
    dt_updated_reg: bool = False,
    # regime lookbacks
    short_lookback: int = 20,
    long_lookback: int = 60,
    # minute fetch knobs (pass-through to your pipeline)
    minute_waiting_time: int = 60,
    minute_chunksize: int = 200,
    minute_fetch_in_chunks: bool = False,
    # optional date bounds
    daily_start_date: Optional[str] = None,
    daily_end_date: Optional[str] = None,
    minute_start_date: Optional[str] = None,
    minute_end_date: Optional[str] = None,
    use_optimal_betas: bool = True,
    optimal_beta_kwargs: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Sets up and runs a clean experiment for multifactor intraday R² analysis.
    """
    # ----------------------- 0) Run your pipeline once -----------------------
    pipe = run_full_pipeline_multifactor_r2(
        filing_date_gte=filing_date_gte,
        dt_updated=dt_updated,
        dt_updated_reg=dt_updated_reg,
        stock=stock,
        regressor=regressor,
        lookback_long=long_lookback, 
        lookback_short=short_lookback, 
        minute_waiting_time=minute_waiting_time,
        minute_chunksize=minute_chunksize,
        minute_fetch_in_chunks=minute_fetch_in_chunks,
        daily_start_date=daily_start_date,
        minute_start_date=minute_start_date,
        daily_end_date=daily_end_date,
        minute_end_date=minute_end_date,
        use_optimal_betas = use_optimal_betas,
        optimal_beta_kwargs = optimal_beta_kwargs
    )

    # Convenience handles
    stock_min_prices = pipe["stock_object_minute"].tickers_stocks_prices     # MultiIndex: (ticker, field)
    reg_min_prices   = pipe["regressor_object_minute"].tickers_stocks_prices
    stock_rets_min   = pipe["stock_returns_minute"]                          # minute close log returns
    reg_rets_min     = pipe["regressor_returns_minute"]
    stock_rets_day   = pipe["stock_returns_daily"]                           # daily close log returns
    reg_rets_day     = pipe["regressor_returns_daily"]
    beta_daily       = pipe["betas_df"]                                      # EWMA betas (daily) vs factor_ticker

    # Minute factor return (Series)
    factor_min_ret = reg_rets_min.copy()

    # ----------------------- 1) Minute meta: trade_date & time-of-day -----------------------
    # Ensure tz-aware US/Eastern minute index
    min_index = stock_rets_min.index
    if getattr(min_index, "tz", None) is None:
        # if not tz-aware, assume US/Eastern per your invariants
        min_index = min_index.tz_localize("US/Eastern")
        stock_rets_min.index = min_index
        reg_rets_min.index   = min_index
        factor_min_ret.index = min_index

    td_series = pd.Series(min_index.tz_convert("US/Eastern").date, index=min_index, name="trade_date")
    tod_series = pd.Series(min_index.tz_convert("US/Eastern").strftime("%H:%M"), index=min_index, name="tod")

    return {'stock_min_prices' : stock_min_prices,
            'reg_min_prices'   : reg_min_prices,
            'stock_rets_min'   : stock_rets_min,
            'reg_rets_min'     : reg_rets_min,
            'stock_rets_day'   : stock_rets_day,
            'reg_rets_day'     : reg_rets_day,
            'beta_daily'       : beta_daily,
            'factor_min_ret'   : factor_min_ret,
            'td_series'        : td_series,
            'tod_series'       : tod_series,
            'min_index'        : min_index,
            'pipe'             : pipe}

#### Beta and Residuals
def map_lagged_betas(data, stock):
    """
    Map lagged betas from daily to minute frequency for all stocks.
    """
    beta_daily       = data['beta_daily'].copy() 
    stock_min_rets   = data['stock_rets_min'].copy()
    td_series        = data['td_series'].copy()
    lagged_betas     = beta_daily.shift(1)
    lagged_betas_min = {}

    for s in stock:
        cols                = lagged_betas[s].columns
        beta_daily_lag1_s   = lagged_betas[s].copy()

        beta_lagged_temp    = pd.DataFrame(index=stock_min_rets.index,
                                           columns=cols, dtype=float)
        for i in cols:
            # Assign each regressor to lagged trade date
            m = td_series.map(beta_daily_lag1_s[i])
            beta_lagged_temp[i] = m.values
        lagged_betas_min[s] = beta_lagged_temp
    return lagged_betas_min

#### Calculate residuals
def calc_idio_rets(s, *, stock_rets_min, factor_min_ret, beta_minute_lag1, dtype='float32'):
    # stock returns: drop missing upfront
    Sr    = stock_rets_min[s].dropna().astype(dtype)

    # betas: drop any row with any missing leg
    B_all = beta_minute_lag1[s]
    B     = B_all.dropna(how='any').astype(dtype)

    # factors: use only the columns this stock actually uses, then align to B's minutes and drop rows with any NaN
    cols = B.columns.intersection(factor_min_ret.columns)
    F    = factor_min_ret[cols].reindex(B.index).dropna(how='any')

    # final time index = minutes present (and fully observed) in B, F, and Sr
    idx = B.index.intersection(F.index, sort=False).intersection(Sr.index, sort=False)

    B = B.loc[idx, cols].astype(dtype)
    F = F.loc[idx, cols].astype(dtype)
    Sr = Sr.loc[idx].astype(dtype)
    # row-wise dot for factor contribution; no fills anywhere
    fac = np.einsum('ij,ij->i', B.to_numpy(copy=False), F.to_numpy(copy=False), optimize=True)
    fac = pd.Series(fac, index=idx, dtype=dtype)
    resid = Sr - fac
    return resid, fac

def get_resid_fac(data, lagged_betas_min, stock = None):
    """ 
    Calculate factor returns and idiosyncratic returns for all stocks.
    Uses calc_idio_rets function.
    """

    stock_rets_min  = data['stock_rets_min'].copy()
    factor_min_ret  = data['factor_min_ret'].copy()

    if stock is None:
        stock = stock_rets_min.columns.tolist()
    results = {s: calc_idio_rets(s,
                                stock_rets_min=stock_rets_min,
                                factor_min_ret=factor_min_ret,
                                beta_minute_lag1=lagged_betas_min)
            for s in stock}

    residual_1m = pd.DataFrame({s: r[0] for s, r in results.items()})
    fac_rets    = pd.DataFrame({s: r[1] for s, r in results.items()})

    return residual_1m, fac_rets

#### Backtest

def _tod_sigma_lagged(
        res : pd.Series, 
        td_series : pd.Series,
        tod : pd.Series,
        window : int,
    ) -> pd.DataFrame:

    df = pd.DataFrame({
        'resid_rolling' : res,
        'trade_date' : td_series,
        'tod'       : tod
    }, 
    index = res.index)


    blocks = []


    for i_tod, grp in df.groupby('tod'):
        # Reindex 
        reindex_days = pd.date_range(grp['resid_rolling'].index.min(), grp['resid_rolling'].index.max(), freq='D', tz='US/Eastern')
        reindex_days = _drop_weekends_index(reindex_days)
        grp = grp.reindex(reindex_days)

        # Calculating rolling std and fallback
        s = grp['resid_rolling'].copy()
        valid_s                 = s.dropna()  
        valid_std_prior         = valid_s.shift(1).rolling(window=window, min_periods=window).std().rename('rolling_std')
        valid_std_fallback      = valid_std_prior.rolling(window=window, min_periods=window).quantile(0.2).rename('fallback_std')
        valid_std_fallback2     = valid_std_prior.expanding(min_periods=window).mean().rename('final_fallback_std')

        valid_stds              = pd.concat([valid_std_prior, valid_std_fallback, valid_std_fallback2], axis=1)
        # merge 
        left_df     = pd.DataFrame({'idx' : s.index})
        right_df    = valid_stds.reset_index().rename(
                                    columns={'index': 'idx'}
                                )
        merged = pd.merge_asof(
            left_df.sort_values('idx'),
            right_df.sort_values('idx'),
            on='idx',
            direction='backward'
            )

        merged.set_index('idx', inplace=True)
        merged = merged.merge(grp, left_index=True, right_index=True, how='inner')

        merged['sigma'] = np.where(merged['rolling_std'] >= merged['fallback_std'],
                                   merged['rolling_std'],
                                   merged['fallback_std'])

        merged['sigma'] = np.where(merged['sigma'] < merged['final_fallback_std'],
                                   merged['final_fallback_std'],
                                   merged['sigma'])
        
        
        cap_min  = merged['sigma'].quantile(0.1)
        cap_repl = merged['sigma'].quantile(0.1)

        merged['sigma'] = np.where(merged['sigma'] < cap_min,
                                   cap_repl, merged['sigma'])
        blocks.append(merged)

    sig_df = pd.concat(blocks, ignore_index=False)

    return sig_df

def build_sigma_and_z_from_tod(
    *,
    residual_m: Dict[int, pd.DataFrame],     # {m_bars -> DataFrame(index=minutes, cols=stocks)} m-bar idio residual sums
    td_series: pd.Series,                    # per-minute trade_date (same master minute index)
    tod_series: pd.Series,                   # per-minute "HH:MM" strings (same master minute index)
    stocks: List[str],                       # list of tickers to include
    window_days: int,                        # prior-day lookback used inside _tod_sigma_lagged
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    For each event window m (bars), compute:
      - sigma_tod_m[m]: wide DataFrame of per-minute σ using _tod_sigma_lagged(...)[ 'sigma' ] for each stock
      - z_m[m]:         wide DataFrame of z = residual_m[m] / sigma_tod_m[m]
    Ensures σ is aligned to the *same* timestamps and trade_dates as the residuals being divided.
    """
    sigma_tod_m: Dict[int, pd.DataFrame] = {}
    z_m: Dict[int, pd.DataFrame] = {}

    # Pre-align helpers to avoid repeated work in the inner loop
    def _aligned_meta(ix: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
        td_aligned  = td_series.reindex(ix)
        tod_aligned = tod_series.reindex(ix)
        return td_aligned, tod_aligned

    for m, rm_df in residual_m.items():
        # keep only requested stocks that are present
        cols = [s for s in stocks if s in rm_df.columns]
        idx  = rm_df.index

        td_aligned, tod_aligned = _aligned_meta(idx)

        # Build sigma wide DF for this m
        sig_wide = pd.DataFrame(index=idx, columns=cols, dtype=float)

        for s in cols:
            # 1) run your TOD-lagged sigma estimator on the *m-bar residual series* for this stock
            sig_raw = _tod_sigma_lagged(
                res=rm_df[s],                    # pass the m-bar residual sum for this stock
                td_series=td_aligned,            # trade_date aligned to *this exact* minute index
                tod=tod_aligned,                 # "HH:MM" aligned to *this exact* minute index
                window=window_days,
            )

            # 2) extract only the 'sigma' column and align to the residual index
            #    (_tod_sigma_lagged returns extra columns; we only keep σ)
            sig_series = sig_raw['sigma'].reindex(idx)
            sig_wide[s] = sig_series.values

        # Store sigma_tod and compute z with strict row-wise alignment
        sigma_tod_m[m] = sig_wide
        z_m[m] = rm_df[cols].divide(sig_wide[cols]) 

    return sigma_tod_m, z_m

def sanity_check_sigma_and_z(
    *,
    sigma_tod_m: Dict[int, pd.DataFrame],   # {m -> DataFrame(index=minutes, cols=tickers) of sigma}
    z_m: Dict[int, pd.DataFrame],           # {m -> DataFrame(index=minutes, cols=tickers) of z}
    thresholds: Tuple[float, ...] = (1.0, 2.0, 3.0),
    max_acf_lag: int = 5,
    tz: str = "US/Eastern",
    z_std_bounds: Tuple[float, float] = (0.8, 1.25),   # acceptable z std band
    z_mean_tol: float = 0.05,                          # acceptable |mean(z)|
    hod_cv_tol: float = 0.30                           # acceptable CV of by-hour stds
) -> Dict[int, Dict[str, Any]]:
    """
    Sanity-check σ and z for each event window key in the dictionaries.
    Returns a dict keyed by window, each containing:
      - 'alignment': info about index/column alignment actions
      - 'sigma_summary': per-ticker stats on sigma (NaN%, <=0%, q10/median/q90)
      - 'z_summary': per-ticker calibration stats (mean/std/skew/kurtosis, exceedances, ACF)
      - 'z_by_hour': per-ticker DataFrame of by-hour mean/std
      - 'flags': per-ticker booleans highlighting potential issues
    """

    def _hour_of_day(idx: pd.DatetimeIndex) -> pd.Series:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
        return pd.Series(idx.hour, index=idx, name="hod")

    def _acf(series: pd.Series, lag: int) -> float:
        try:
            return float(series.autocorr(lag=lag))
        except Exception:
            return np.nan

    # P(|Z|>t) under N(0,1) via erfc identity (no SciPy dependency)
    def _p_norm_abs_gt(t: float) -> float:
        return float(math.erfc(float(t) / math.sqrt(2)))

    out: Dict[int, Dict[str, Any]] = {}

    # iterate windows present in both dicts
    common_windows = sorted(set(sigma_tod_m).intersection(set(z_m)))
    for m in common_windows:
        sig = sigma_tod_m[m].copy()
        Z   = z_m[m].copy()

        # --- alignment: ensure same index/columns (conservative: intersect)
        align_info = {"aligned_index": False, "aligned_columns": False}
        if not sig.index.equals(Z.index):
            common_index = sig.index.intersection(Z.index)
            sig = sig.reindex(common_index)
            Z   = Z.reindex(common_index)
            align_info["aligned_index"] = True
        if list(sig.columns) != list(Z.columns):
            common_cols = [c for c in sig.columns if c in Z.columns]
            sig = sig[common_cols]
            Z   = Z[common_cols]
            align_info["aligned_columns"] = True

        cols = list(Z.columns)
        idx  = Z.index
        hod  = _hour_of_day(idx)

        # --- sigma summary
        sig_rows = []
        for s in cols:
            x = sig[s]
            n = len(x)
            nan_rate   = float(x.isna().mean()) if n else np.nan
            nonpos_rate= float((x <= 0).mean()) if n else np.nan
            q10 = float(x.quantile(0.10)) if n else np.nan
            med = float(x.median())       if n else np.nan
            q90 = float(x.quantile(0.90)) if n else np.nan
            sig_rows.append({
                "ticker": s,
                "nan_rate": nan_rate,
                "nonpos_rate": nonpos_rate,
                "q10": q10, "median": med, "q90": q90
            })
        sigma_summary = (pd.DataFrame(sig_rows)
                         .set_index("ticker")
                         if sig_rows else pd.DataFrame())

        # --- z summary & by-hour
        z_rows = []
        by_hour: Dict[str, pd.DataFrame] = {}
        flags_rows = []
        for s in cols:
            z = Z[s].dropna()
            if z.empty:
                # still record a row with NaNs to preserve output shape
                z_rows.append({
                    "ticker": s, "mean": np.nan, "std": np.nan, "skew": np.nan, "kurtosis": np.nan,
                    **{f"p_emp(|z|>{t})": np.nan for t in thresholds},
                    **{f"p_norm(|z|>{t})": _p_norm_abs_gt(t) for t in thresholds},
                    **{f"acf_z_{k}": np.nan for k in range(1, max_acf_lag+1)},
                    **{f"acf_z2_{k}": np.nan for k in range(1, max_acf_lag+1)}
                })
                flags_rows.append({"ticker": s, "flag_mean": True, "flag_std": True, "flag_hod_cv": True})
                continue

            mean = float(z.mean())
            std  = float(z.std(ddof=0))
            skew = float(z.skew())
            kurt = float(z.kurt())  # pandas returns excess kurtosis (Fisher)

            exc_emp = {f"p_emp(|z|>{t})": float((z.abs() > t).mean()) for t in thresholds}
            exc_the = {f"p_norm(|z|>{t})": _p_norm_abs_gt(t) for t in thresholds}

            g = pd.DataFrame({"z": z, "hod": hod.reindex(z.index)}).dropna()
            per_h = g.groupby("hod")["z"].agg(["mean", "std"])
            by_hour[s] = per_h
            std_cv = (float(per_h["std"].std() / per_h["std"].mean())
                      if per_h["std"].mean() and per_h["std"].mean() > 0 else np.nan)

            ac_z  = {lag: _acf(z, lag) for lag in range(1, max_acf_lag + 1)}
            ac_z2 = {lag: _acf(z.pow(2), lag) for lag in range(1, max_acf_lag + 1)}

            z_rows.append({
                "ticker": s, "mean": mean, "std": std, "skew": skew, "kurtosis": kurt,
                "std_cv_by_hour": std_cv, **exc_emp, **exc_the,
                **{f"acf_z_{k}": v for k, v in ac_z.items()},
                **{f"acf_z2_{k}": v for k, v in ac_z2.items()},
            })

            # simple flags against tolerances
            flag_mean = (abs(mean) > z_mean_tol)
            flag_std  = not (z_std_bounds[0] <= std <= z_std_bounds[1])
            flag_hod  = (std_cv > hod_cv_tol) if not np.isnan(std_cv) else True
            flags_rows.append({"ticker": s, "flag_mean": flag_mean, "flag_std": flag_std, "flag_hod_cv": flag_hod})

        z_summary = (pd.DataFrame(z_rows).set_index("ticker") if z_rows else pd.DataFrame())
        flags_df  = (pd.DataFrame(flags_rows).set_index("ticker") if flags_rows else pd.DataFrame())

        out[m] = {
            "alignment": align_info,
            "sigma_summary": sigma_summary,
            "z_summary": z_summary,
            "z_by_hour": by_hour,         # dict[ticker] -> DataFrame with index=hour 0..23
            "flags": flags_df
        }

    return out

def z_exceedance_diagnostic(
    *,
    sigma_tod_m: Dict[int, pd.DataFrame],   # {m -> sigma DataFrame (minutes x tickers)}
    z_m: Dict[int, pd.DataFrame],           # {m -> z DataFrame (minutes x tickers)}
    residual_m: Dict[int, pd.DataFrame],    # {m -> residual DataFrame (minutes x tickers)}
    m: int,                                  # event window (bars)
    Z: float = 3.0,                          # exceedance threshold on |z|
    top_k: int = 10,
    tz: str = "US/Eastern",
    integrity_rtol: float = 1e-10,           # kept for signature compatibility; not used here
    integrity_atol: float = 1e-12,           # kept for signature compatibility; not used here
    return_plot: bool = True,
    clip_mult: float = 3.0,                  # clip |z| at clip_mult * Z for plotting
    s_min: float = 6.0,                      # min marker size
    s_max: float = 22.0                      # max marker size
) -> Dict[str, Any]:
    """
    Returns:
      - 'summary_wide': MultiIndex-columns DataFrame (ticker → ['date','session','z_avg','z_count'])
      - 'reversion_wide': MultiIndex-columns DataFrame (ticker → ['date','session','spread','count_above_spread','z_count'])
      - 'events_long': long table with per-bar exceedances
      - 'fig': Plotly figure (legend = 'Ticker × Session'); None if return_plot=False
    """

    # ---------- pull & validate ----------
    if m not in sigma_tod_m or m not in z_m or m not in residual_m:
        raise ValueError(f"Missing window m={m} in one of the inputs.")
    sig = sigma_tod_m[m].copy()
    Zdf = z_m[m].copy()
    Rdf = residual_m[m].copy()

    if not (sig.index.equals(Zdf.index) and Zdf.index.equals(Rdf.index)):
        raise ValueError(f"Index mismatch across sigma/z/residual for m={m}. Exact alignment required.")
    if list(sig.columns) != list(Zdf.columns) or list(Zdf.columns) != list(Rdf.columns):
        raise ValueError(f"Column mismatch across sigma/z/residual for m={m}. Exact ticker sets required.")

    def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is None:
            raise ValueError("Timestamps must be tz-aware. Found naive index.")
        return idx.tz_convert(tz)

    idx_et = _to_et(Zdf.index)
    sig.index = idx_et; Zdf.index = idx_et; Rdf.index = idx_et

    # ---------- sessions & trade dates ----------
    # Sessions: pre [04:30,09:30), regular [09:30,16:00), post [16:00,20:00)
    def _session_label(ts: pd.Timestamp) -> Optional[str]:
        hhmm = ts.tz_convert(tz).strftime("%H:%M")
        if "04:30" <= hhmm < "09:30": return "pre"
        if "09:30" <= hhmm < "16:00": return "regular"
        if "16:00" <= hhmm < "20:00": return "post"
        return None

    sess = pd.Series([_session_label(ts) for ts in Zdf.index], index=Zdf.index, name="session")
    keep = sess.notna()
    if not keep.any():
        raise ValueError("No timestamps fall within defined sessions (pre/regular/post).")

    sig = sig[keep]; Zdf = Zdf[keep]; Rdf = Rdf[keep]; sess = sess[keep]
    trade_date = pd.Series(Zdf.index.tz_convert(tz).date, index=Zdf.index, name="date")

    # ---------- long exceedances ----------
    z_long = Zdf.stack().rename("z").to_frame()
    s_long = sig.stack().rename("sigma").to_frame()
    r_long = Rdf.stack().rename("resid").to_frame()

    events = z_long.join(s_long).join(r_long)  # index = (timestamp, ticker)
    events.index.names = ["timestamp", "ticker"]
    ts_level = events.index.get_level_values("timestamp")
    events["session"] = sess.reindex(ts_level).to_numpy()
    events["date"]    = trade_date.reindex(ts_level).to_numpy()
    events["abs_z"]   = events["z"].abs()
    events["sign"]    = np.sign(events["z"])

    exceed = events[events["abs_z"] >= Z].copy()

    if exceed.empty:
        empty_wide = pd.DataFrame()
        return {
            "summary_wide": empty_wide,
            "reversion_wide": empty_wide,
            "events_long": exceed.reset_index()[["ticker","timestamp","date","session","z","sigma","resid","abs_z","sign"]],
            "fig": None
        }
    exceed = exceed.reset_index()[["ticker","timestamp","date","session","z","sigma","resid","abs_z","sign"]]

    # ---------- per-ticker (date × session) summaries ----------
    def _agg_group(g: pd.DataFrame) -> pd.Series:
        z_count   = int(g.shape[0])
        z_avg     = float(g["z"].mean())
        z_med     = float(g["z"].median())
        pct_pos   = float((g["z"] > 0).sum() / z_count) * 100
        max_abs_z = float(g["abs_z"].max())
        # signed z at the time of the maximum absolute z
        i_ext         = g["abs_z"].to_numpy().argmax()
        max_signed_z  = float(g["z"].iloc[i_ext])
        time_of_max   = g["timestamp"].iloc[i_ext]
        return pd.Series({
            "z_count": z_count,
            "z_avg": z_avg,
            "z_med": z_med,
            "pct_pos": pct_pos,
            "max_abs_z": max_abs_z,
            "max_signed_z": max_signed_z,
            "time_of_max": time_of_max
        })
    grp = exceed.groupby(["ticker","date","session"], sort=False).apply(_agg_group).reset_index()

    def _topk_for_ticker(df_t: pd.DataFrame, k: int) -> pd.DataFrame:
        sortd = df_t.sort_values(["z_avg", "z_count"], ascending=[False, False])
        out = sortd[["date","session","z_avg", "max_signed_z","z_med","pct_pos","z_count"]].head(k).reset_index(drop=True)
        if out.shape[0] < k:
            pad = pd.DataFrame(index=range(out.shape[0], k), columns=out.columns, dtype=object)
            out = pd.concat([out, pad], ignore_index=True)
        return out

    tickers = grp["ticker"].unique().tolist()
    per_ticker_tables = {t: _topk_for_ticker(grp[grp["ticker"] == t], top_k) for t in tickers}

    def _stack_wide(blocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cols_lv2 = ["date","session","z_avg","z_count", "z_med", "pct_pos"]
        pieces = []
        for t, df_t in blocks.items():
            df_t = df_t[cols_lv2]
            df_t.columns = pd.MultiIndex.from_product([[t], cols_lv2])
            pieces.append(df_t)
        return pd.concat(pieces, axis=1) if pieces else pd.DataFrame()

    summary_wide = _stack_wide(per_ticker_tables)

    # ---------- reversion table ----------
    def _reversion_metrics(g: pd.DataFrame) -> Optional[pd.Series]:
        if (g["z"] > 0).any() and (g["z"] < 0).any():
            spread = float(abs(g["z"].max() - g["z"].min()))
            count_above_spread = int(Z)  # retained name per your spec
            return pd.Series({"spread": spread, "count_above_thresh": count_above_spread, "z_count": int(g.shape[0])})
        return None

    rev_rows = []
    for (tkr, d, s), gslice in exceed.groupby(["ticker","date","session"], sort=False):
        met = _reversion_metrics(gslice)
        if met is not None:
            row = {"ticker": tkr, "date": d, "session": s}
            row.update(met.to_dict()); rev_rows.append(row)

    if rev_rows:
        rev_df = pd.DataFrame(rev_rows)
        rev_blocks = {}
        for t in rev_df["ticker"].unique():
            df_t = rev_df[rev_df["ticker"] == t].sort_values(["z_count","spread"], ascending=[False, False]).head(top_k)
            out_t = df_t[["date","session","spread","count_above_thresh","z_count"]].reset_index(drop=True)
            if out_t.shape[0] < top_k:
                pad = pd.DataFrame(index=range(out_t.shape[0], top_k), columns=out_t.columns, dtype=object)
                out_t = pd.concat([out_t, pad], ignore_index=True)
            out_t.columns = pd.MultiIndex.from_product([[t], ["date","session","spread","count_above_thresh","z_count"]])
            rev_blocks[t] = out_t
        reversion_wide = pd.concat(list(rev_blocks.values()), axis=1)
    else:
        reversion_wide = pd.DataFrame()

    # ---------- Plotly: legend entries = 'Ticker × Session' ----------
    fig = None
    if return_plot:
        # order tickers by activity
        counts_by_tkr = exceed.groupby("ticker").size().sort_values(ascending=False)
        ticker_order = counts_by_tkr.index.tolist() if not counts_by_tkr.empty else sorted(Zdf.columns.tolist())
        top3 = set(ticker_order[:3])

        z_cap = max(Z, clip_mult * Z)
        def _sizes(z_series: pd.Series) -> np.ndarray:
            mag = np.minimum(np.abs(z_series.values), z_cap)
            if z_cap == Z:
                return np.full_like(mag, (s_min + s_max) / 2.0, dtype=float)
            frac = (mag - Z) / (z_cap - Z)
            frac = np.clip(frac, 0.0, 1.0)
            return s_min + frac * (s_max - s_min)

        sessions = ["pre","regular","post"]
        traces = []

        for sess_name in sessions:
            e_sess = exceed[exceed["session"] == sess_name]
            for tkr in ticker_order:
                e_t = e_sess[e_sess["ticker"] == tkr]
                if e_t.empty:
                    # still make an empty trace so the legend shows all combos
                    traces.append(go.Scatter(
                        x=[], y=[],
                        mode="markers",
                        name=f"{tkr} × {sess_name.capitalize()}",
                        legendgroup=f"{tkr}_{sess_name}",
                        showlegend=True,
                        visible="legendonly" if not (sess_name == "regular" and tkr in top3) else True,
                        marker=dict(symbol="circle", size=8, opacity=0.6),
                        hovertemplate="No points<extra></extra>"
                    ))
                    continue

                # per-point marker symbol by sign
                symbols = ["triangle-up" if val > 0 else "triangle-down" for val in e_t["z"].values]
                sizes = _sizes(e_t["z"])

                traces.append(go.Scatter(
                    x=e_t["timestamp"],
                    y=[tkr]*len(e_t),
                    mode="markers",
                    name=f"{tkr} × {sess_name.capitalize()}",
                    legendgroup=f"{tkr}_{sess_name}",
                    showlegend=True,
                    visible="legendonly" if not (sess_name == "regular" and tkr in top3) else True,
                    marker=dict(symbol=symbols, size=sizes, opacity=0.7),
                    hovertemplate=("Ticker=%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M %Z}"
                                   "<br>Session=%{customdata[1]}"
                                   "<br>z=%{customdata[2]:.2f} (clipped at "+f"{clip_mult:g}×"+")"
                                   "<br>σ=%{customdata[3]:.4g}"
                                   "<br>resid=%{customdata[4]:.4g}<extra></extra>"),
                    customdata=np.stack([
                        e_t["ticker"].values,
                        e_t["session"].values,
                        np.clip(np.abs(e_t["z"].values), a_min=None, a_max=z_cap) * np.sign(e_t["z"].values),
                        e_t["sigma"].values,
                        e_t["resid"].values
                    ], axis=-1)
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Z-Score Exceedances (|z| ≥ {Z:g}, clipped at {clip_mult:g}×)",
            xaxis_title="Time (ET)",
            yaxis_title="Ticker",
            yaxis=dict(categoryorder="array", categoryarray=ticker_order),
            hovermode="closest",
            legend_title="Toggle: Ticker × Session",
            template="plotly_white"
        )
        fig.update_xaxes(rangeslider_visible=True)

    return {
        "summary_wide": summary_wide,
        "reversion_wide": reversion_wide,
        "events_long": exceed,   # per-bar exceedances with z, sigma, resid
        "fig": fig
    }

def plot_hourly_net_resid_for_exceedance_hours(
    *,
    sigma_tod_m: Dict[int, pd.DataFrame],   # not used; kept for signature symmetry
    z_m: Dict[int, pd.DataFrame],           # {m -> z DataFrame} (minute x tickers)
    residual_m: Dict[int, pd.DataFrame],    # {m -> residual DataFrame}
    m: int,
    Z: float = 3.0,
    tickers: Optional[List[str]] = None,    # up to 2; if None, auto-pick top-2 by exceedance hours
    session: Optional[str] = None,          # 'pre' | 'regular' | 'post' | None
    tz: str = "US/Eastern",
) -> Dict[str, Any]:
    """
    Show hourly net idiosyncratic returns (sum of residual_m within the hour, signed),
    but only for hours that contain at least one |z| >= Z bar.
    Returns {'fig': plotly.Figure, 'hourly_tables': {ticker: DataFrame}}
    """
    if m not in z_m or m not in residual_m:
        raise ValueError(f"Missing window m={m} in inputs.")
    Zdf = z_m[m].copy()
    Rdf = residual_m[m].copy()

    if not Zdf.index.equals(Rdf.index):
        raise ValueError("Index mismatch between z_m[m] and residual_m[m].")
    if list(Zdf.columns) != list(Rdf.columns):
        raise ValueError("Column mismatch between z_m[m] and residual_m[m].")

    # ensure ET tz
    def _to_et(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is None: raise ValueError("Timestamps must be tz-aware.")
        return idx.tz_convert(tz)
    idx_et = _to_et(Zdf.index)
    Zdf.index = idx_et; Rdf.index = idx_et

    # session labeling
    def _session_label(ts: pd.Timestamp) -> Optional[str]:
        hhmm = ts.tz_convert(tz).strftime("%H:%M")
        if "04:30" <= hhmm < "09:30": return "pre"
        if "09:30" <= hhmm < "16:00": return "regular"
        if "16:00" <= hhmm < "20:00": return "post"
        return None
    sess = pd.Series([_session_label(ts) for ts in Zdf.index], index=Zdf.index, name="session")
    keep = sess.notna()
    if not keep.any():
        raise ValueError("No timestamps fall within (pre/regular/post) sessions.")
    Zdf, Rdf, sess = Zdf[keep], Rdf[keep], sess[keep]

    if session is not None:
        session = session.lower()
        if session not in {"pre","regular","post"}:
            raise ValueError("session must be one of: 'pre','regular','post', or None.")
        mask = (sess == session)
        Zdf, Rdf, sess = Zdf[mask], Rdf[mask], sess[mask]

    # long table & exceedance mask
    z_long = Zdf.stack().rename("z")
    r_long = Rdf.stack().rename("resid")
    df = pd.concat([z_long, r_long], axis=1)
    df.index.names = ["timestamp","ticker"]
    df["abs_z"] = df["z"].abs()
    df = df.reset_index()

    # hour key
    df["hour"] = df["timestamp"].dt.tz_convert(tz).dt.floor("h")

    # choose tickers (max 2) by # of hours that contain at least one exceedance
    hours_with_exceed = (df.assign(exceed=df["abs_z"]>=Z)
                           .groupby(["ticker","hour"])["exceed"].max()
                           .rename("has_exceed")
                           .reset_index())
    if tickers is None or len(tickers)==0:
        counts = (hours_with_exceed.groupby("ticker")["has_exceed"]
                  .sum().sort_values(ascending=False))
        chosen = counts.index.tolist()[:2]
    else:
        chosen = [t for t in tickers if t in Zdf.columns][:2]
    if not chosen:
        raise ValueError("No valid tickers after filtering.")

    # build hourly net signed sums, but only for hours that have at least one exceedance
    hourly_tables: Dict[str, pd.DataFrame] = {}
    for t in chosen:
        dft = df[df["ticker"]==t].copy()
        hx  = hours_with_exceed[hours_with_exceed["ticker"]==t]
        good_hours = set(hx.loc[hx["has_exceed"], "hour"])
        if not good_hours:
            hourly_tables[t] = pd.DataFrame(columns=["hour","net_resid","max_abs_z"])
            continue
        g = (dft.groupby("hour").agg(
                net_resid=("resid", "sum"),
                max_abs_z=("abs_z", "max"))
             .reset_index())
        g = g[g["hour"].isin(good_hours)].sort_values("hour")
        hourly_tables[t] = g

    # plotly
    fig = go.Figure()
    for t in chosen:
        g = hourly_tables[t]
        fig.add_trace(go.Scatter(
            x=g["hour"], y=g["net_resid"],
            mode="lines+markers",
            name=t,
            hovertemplate=("Ticker="+t+
                           "<br>%{x|%Y-%m-%d %H:%M %Z}"
                           "<br>Hourly net resid=%{y:.4g}"
                           "<br>max |z| in hour=%{customdata[0]:.2f}<extra></extra>"),
            customdata=np.stack([g["max_abs_z"].to_numpy()], axis=-1) if not g.empty else None
        ))
    ttl = [f"Hourly Net Idiosyncratic Return (show hours with ≥1 bar |z|≥{Z:g})", f"m={m}"]
    if session: ttl.append(session.capitalize())
    fig.update_layout(
        title=" — ".join(ttl),
        xaxis_title="Time (ET) — hourly bins",
        yaxis_title="Hourly net residual (sum of residual_m within the hour)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="#777", zerolinewidth=1)
    fig.update_xaxes(rangeslider_visible=True)

    return {"fig": fig, "hourly_tables": hourly_tables}

def find_max_min_trade_date(df):
    """ 
    Find the max high and min low for a trade date DataFrame.
    """
    df.dropna(subset=['high', 'low'], how='any', inplace=True)
    max = df.nlargest(n=1, columns=['high'])[['high', 'transactions', 'volume']]
    max.columns = ['high', 'high_transactions', 'high_volume']
    min = df.nsmallest(n=1, columns=['low'])[['low', 'transactions', 'volume']]
    min.columns = ['low', 'low_transactions', 'low_volume']
    
    increase = True if max.index.time < min.index.time else False

    max['increase'] = increase
    min['low_time'] = min.index.time
    max['high_time'] = max.index.time
    min.reset_index(inplace=True)
    max.reset_index(inplace=True)
    
    min_max_for_trade_date = pd.concat([max,min], axis=1)
    return min_max_for_trade_date[['increase', 'high', 'low', 'high_time', 'low_time', 'high_volume', 'low_volume', 'high_transactions', 'low_transactions']]


################ Regime Analysis

def build_regime_per_stock(
    combined_lab: pd.DataFrame,
    factor_rets: pd.DataFrame,
    beta_daily_lag1: Dict[str, pd.DataFrame],
    short_lookback: int,
    long_lookback: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build regime detection DataFrames per stock using rolling R².

    Args:
        combined_lab (pd.DataFrame): Combined labels DataFrame.
        factor_rets (pd.DataFrame): Factor returns DataFrame.
        beta_daily_lag1 (Dict[str, pd.DataFrame]): Daily lagged betas per stock.
        short_lookback (int): Short lookback period for rolling R².
        long_lookback (int): Long lookback period for rolling R².

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing short and long lookback regime DataFrames.
    """
    lagged_r2_short = rolling_r2_intraday_multifactor_perstock(
        combined_lab, factor_rets, beta_daily_lag1, short_lookback
    )
    lagged_r2_long = rolling_r2_intraday_multifactor_perstock(
        combined_lab, factor_rets, beta_daily_lag1, long_lookback
    )
    return lagged_r2_short, lagged_r2_long

def define_regime_per_stock(
        lagged_r2_long: int,
        lagged_r2_short: int,
        delta : float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build regime detection DataFrames per stock using rolling R².

    Args:
        combined_lab (pd.DataFrame): Combined labels DataFrame.
        factor_rets (pd.DataFrame): Factor returns DataFrame.
        beta_daily_lag1 (Dict[str, pd.DataFrame]): Daily lagged betas per stock.
        short_lookback (int): Short lookback period for rolling R².
        long_lookback (int): Long lookback period for rolling R².

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing short and long lookback regime DataFrames.
    """
    r2_valid_dict_diff = {}

    for i in lagged_r2_long.columns:
        mask                          = (lagged_r2_long[i].isna() == False) & (lagged_r2_short[i].isna() == False)
        temp                          = (lagged_r2_short[i][mask] - lagged_r2_long[i][mask]).to_frame(f'{i}_r2_diff')
        temp[f'c_idio_regime_d{delta}'] = np.where(temp[f'{i}_r2_diff'] > delta, -1, 0)
        temp[f'c_idio_regime_d{delta}'] = np.where(temp[f'{i}_r2_diff'] < -delta, 1, temp[f'c_idio_regime_d{delta}'])

        r2_long_                      = lagged_r2_long[i][mask].rename('long_r2')
        r2_short_                     = lagged_r2_short[i][mask].rename('short_r2')
        temp                          = pd.concat([temp, r2_short_, r2_long_], axis=1)

        r2_valid_dict_diff[i] = temp

    return r2_valid_dict_diff

def build_regime_df(residual_1m : dict[str, pd.DataFrame],
                    combined_lab : pd.DataFrame,
                    r2_valid_dict_diff : dict[str, pd.DataFrame],
                    lower_q : float = 0.3,
                    upper_q : float = 0.7,
                    delta : float = 0.3,
                    ):
    regime_dfs = {}

    for i in residual_1m.columns:

        res = residual_1m[i]
        res.name = 'idio'

        df = pd.concat([combined_lab[[i, 'trade_date', 'session']], res], axis=1).dropna()

        for j in r2_valid_dict_diff[i].columns:
            df[j] = df['trade_date'].map(r2_valid_dict_diff[i][j])

        r2_cap_idio = df['short_r2'].quantile(lower_q)
        r2_cap_macro = df['short_r2'].quantile(upper_q)
        df[f'idio_regime_d{delta}'] = df[f'c_idio_regime_d{delta}'].copy()
        df[f'idio_regime_d{delta}'] = np.where(df['short_r2'] < r2_cap_idio, 1, df[f'idio_regime_d{delta}'])
        df[f'idio_regime_d{delta}'] = np.where(df['short_r2'] > r2_cap_macro, -1, df[f'idio_regime_d{delta}'])
        
        regime_dfs[i] = df
    
    return regime_dfs

def enforce_runs(s, runlen):
    out = s.copy()

    # If Series, operate directly
    if isinstance(s, pd.Series):
        keep_idio = s.eq( 1).rolling(runlen, min_periods=runlen).sum().ge(runlen)
        keep_coup = s.eq(-1).rolling(runlen, min_periods=runlen).sum().ge(runlen)
        out.loc[s.eq( 1) & ~keep_idio] = 0
        out.loc[s.eq(-1) & ~keep_coup] = 0
        return out

    # If DataFrame, do it column by column
    for col in s.columns:
        if pd.api.types.is_numeric_dtype(s[col]):
            keep_idio = s[col].eq( 1).rolling(runlen, min_periods=runlen).sum().ge(runlen)
            keep_coup = s[col].eq(-1).rolling(runlen, min_periods=runlen).sum().ge(runlen)
            out.loc[s[col].eq( 1) & ~keep_idio, col] = 0
            out.loc[s[col].eq(-1) & ~keep_coup, col] = 0
    return out

def enforce_regimes(regime_dfs, runlen : int = 3, delta : float = 0.3):
    for s, df in regime_dfs.items():
        
        regime = df[['trade_date', f'idio_regime_d{delta}']].groupby('trade_date').first()
        regime_enforced = enforce_runs(regime[f'idio_regime_d{delta}'], runlen)
        df[f'idio_regime_d{delta}'] = df['trade_date'].map(regime_enforced)
    return regime_dfs

def regime_pipeline(
        data, stock,
        short_lookback : int,
        long_lookback  : int,
        runlen : int, delta : float
)-> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """ 
    Uses lagged betas to calculate r2 (long and short lookback windows)
    per stock each day.
    """
    # We need stock returns and factor returns
    stock_rets_min  = data['stock_rets_min'].copy()
    factor_min_ret  = data['factor_min_ret'].copy()
    beta_daily      = data['beta_daily'].copy()
    # We also need the betas
    if stock is None:
        stock = stock_rets_min.columns

    lagged_betas_min        = map_lagged_betas(data, stock)
    residual_1m, fac_rets   = get_resid_fac(data, lagged_betas_min, stock)
    # Get labelled returns data for r2 
    combined_lab            = get_session_and_trade_date(stock_rets_min)
    # Get lagged r2 values
    lagged_r2_short, lagged_r2_long = build_regime_per_stock(
        combined_lab,
        factor_min_ret,
        beta_daily,
        short_lookback,
        long_lookback
    )
    # Assigns back the r2 data to the stock data
    r2_valid_dict_diff = define_regime_per_stock(
        lagged_r2_long,
        lagged_r2_short,
        delta
    )
    regime_dfs = build_regime_df(
        residual_1m, combined_lab,
        r2_valid_dict_diff, delta=delta
    )
    # Ensure a run length
    regime_dfs = enforce_regimes(regime_dfs, runlen=runlen, delta=delta)
    
    return regime_dfs, residual_1m, fac_rets

import plotly.graph_objects as go
import plotly.io as pio

def plot_interactive_r2_dashboard_multifactor(
    *,
    filing_date_gte: str,
    stock: Sequence[str],
    regressor: Sequence[str],
    short_lookback: int = 20,
    long_lookback: int = 60,
    # fetch knobs
    dt_updated: bool = False,
    dt_updated_reg: bool = False,
    minute_waiting_time: int = 60,
    minute_chunksize: int = 200,
    minute_fetch_in_chunks: bool = False,
    daily_start_date: str | None = None,
    daily_end_date: str | None = None,
    minute_start_date: str | None = None,
    minute_end_date: str | None = None,
    # presentation
    init_stock: str | None = None,
    title: str | None = None,
    optimal_beta_kwargs: Dict[str, Any] = None,
):
    # ---------- build data ----------
    pipe = run_full_pipeline_multifactor_r2(
        filing_date_gte=filing_date_gte,
        stock=stock,
        regressor=regressor,
        lookback_short=short_lookback,
        lookback_long=long_lookback,
        dt_updated=dt_updated,
        dt_updated_reg=dt_updated_reg,
        minute_waiting_time=minute_waiting_time,
        minute_chunksize=minute_chunksize,
        minute_fetch_in_chunks=minute_fetch_in_chunks,
        daily_start_date=daily_start_date,
        daily_end_date=daily_end_date,
        minute_start_date=minute_start_date,
        minute_end_date=minute_end_date,
        use_optimal_betas=True,
        optimal_beta_kwargs=optimal_beta_kwargs,
    )

    r2s = pipe["r2_short_daily"]   # index: trading_day, cols: tickers
    r2l = pipe["r2_long_daily"]

    # ensure aligned index across both frames
    idx = r2s.index.union(r2l.index).sort_values()
    r2s = r2s.reindex(idx)
    r2l = r2l.reindex(idx)

    # ---------- figure ----------
    fig = go.Figure()
    tickers = list(r2s.columns)
    init = init_stock or (tickers[0] if tickers else None)
    if init is None:
        return go.Figure()

    # add 2 traces per stock, default hidden; show only init
    for t in tickers:
        fig.add_trace(go.Scatter(
            x=idx, y=r2s[t], mode="lines", name=f"{t} R² (short {short_lookback}d)",
            hovertemplate="%{x|%Y-%m-%d}<br>R²=%{y:.3f}<extra></extra>",
            visible=(t == init)
        ))
        fig.add_trace(go.Scatter(
            x=idx, y=r2l[t], mode="lines", name=f"{t} R² (long {long_lookback}d)",
            line=dict(dash="dash"),  # dashed line for long lookback
            hovertemplate="%{x|%Y-%m-%d}<br>R²=%{y:.3f}<extra></extra>",
            visible=(t == init)
        ))

    # dropdown to toggle stocks: flip visibility for the pair of traces belonging to that ticker
    buttons = []
    total = len(fig.data)
    traces_per_stock = 2
    for i, t in enumerate(tickers):
        vis = [False] * total
        base = i * traces_per_stock
        vis[base] = True
        vis[base + 1] = True
        buttons.append(dict(
            label=t,
            method="update",
            args=[{"visible": vis},
                  {"title": f"{t} — Rolling multi-factor R² (short vs long)"}],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down", x=1.0, xanchor="right", y=1.12, yanchor="top",
            buttons=buttons, showactive=True, bgcolor="white", bordercolor="#ccc", pad=dict(r=8, t=2, b=2, l=2),
        )],
        xaxis=dict(title="Trading day"),
        yaxis=dict(title="R²", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        margin=dict(l=70, r=70, t=70, b=60),
        title=title or f"{init} — Rolling multi-factor R² (short vs long)"
    )
    pio.templates.default = "plotly_white"
    return fig
