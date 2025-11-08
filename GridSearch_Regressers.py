# ===================== end_to_end_factor_selector.py =====================
from __future__ import annotations
from typing import Sequence, Optional, Tuple, Dict, Any, List
import itertools as it
import numpy as np
import pandas as pd
from fredapi import Fred
import datetime as dt
from typing import Union, Optional
from Modules.stock_data_functions import TickerComparison, calc_log_rets


# --------------------------- utils ---------------------------
def calc_zscore(
    series: pd.Series,
    winsor: Optional[Union[float, Tuple[float, float]]] = None,
    ddof: int = 1,
) -> pd.Series:
    """
    Convert returns to z-scores after optional two-sided winsorization.

    winsor:
      - None: no winsorization
      - float a in [0, 0.5): symmetric winsor at (a, 1-a)
      - tuple (low_q, high_q): explicit quantiles, e.g. (0.01, 0.99)
    ddof:
      - degrees of freedom for std (pandas std defaults to 1)
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)

    if winsor is not None:
        if isinstance(winsor, (int, float)):
            a = float(winsor)
            if not (0 <= a < 0.5):
                raise ValueError("winsor float must be in [0, 0.5).")
            low_q, high_q = a, 1 - a
        else:
            if len(winsor) != 2:
                raise ValueError("winsor tuple must be (low_q, high_q).")
            low_q, high_q = winsor

        if not (0 <= low_q < high_q <= 1):
            raise ValueError("quantiles must satisfy 0 ≤ low < high ≤ 1.")

        q_low, q_high = s.quantile([low_q, high_q])
        s = s.clip(lower=q_low, upper=q_high)

    mu = s.mean()
    sigma = s.std(ddof=ddof)

    # Safe for constant/NA windows: return zeros where data exist, keep NaNs as NaNs.
    z = (s - mu) / sigma if (np.isfinite(sigma) and sigma != 0) else s * 0.0
    z.name = series.name
    return z

############### Reduce ETF Universe ##########################

def choose_etf_representatives(
    X: pd.DataFrame,
    k_max: Optional[int] = None,
    corr_cap: float = 0.90,
    min_var: float = 1e-12,
) -> List[str]:
    """
    Unsupervised ETF de-duplication via QR-like column pivoting + correlation cap.
    - Input X: DataFrame of (already winsorized/z-scored) ETF returns, aligned in time.
    - k_max: max number of ETFs to keep (default: automatic from effective rank).
    - corr_cap: don't admit a candidate if |corr| with any chosen ETF exceeds this.
    - min_var: skip columns with near-zero variance.

    Returns: list of selected ETF column names in importance order.
    """
    # 0) Clean: drop all-NaN cols, keep numeric, filter near-constant
    X = X.copy()
    X = X.select_dtypes(include=[np.number])
    variances = X.var(skipna=True)
    cols = [c for c in X.columns if np.isfinite(variances.get(c, np.nan)) and variances[c] > min_var]
    X = X[cols]
    if X.empty:
        return []

    # Precompute correlation matrix for corr_cap screening (pairwise complete obs)
    corr = X.corr().abs()

    # 1) Center (columns already standardized is fine; centering helps numerics)
    Xc = X.subtract(X.mean(axis=0), axis=1).to_numpy()
    n, p = Xc.shape

    # 2) Gram–Schmidt with pivoting by residual column norms
    selected = []
    residual = Xc.copy()
    remaining = list(range(p))

    # Auto k_max: limit by p and a loose n/10 rule to keep models parsimonious later
    if k_max is None:
        k_max = min(p, max(1, n // 10))

    # Track an orthonormal basis of chosen columns to orthogonalize candidates
    Q = []

    while remaining and len(selected) < k_max:
        # Correlation-cap mask against *original* features
        mask = []
        for j in remaining:
            ok = True
            for s in selected:
                # if any selected is too correlated with candidate, drop candidate
                if corr.iloc[s, j] > corr_cap:
                    ok = False
                    break
            mask.append(ok)
        cand_idx = [r for r, ok in zip(remaining, mask) if ok]
        if not cand_idx:
            break

        # Compute residual norms for current candidates
        norms = []
        for j in cand_idx:
            col = residual[:, j]
            # handle NaNs by zeroing them out for norm computation
            norms.append(np.nan_to_num(col).dot(np.nan_to_num(col)))
        # choose pivot with maximum residual norm
        jstar = cand_idx[int(np.argmax(norms))]
        selected.append(jstar)

        # Update orthonormal basis with the newly selected column (Modified GS)
        v = residual[:, jstar].copy()
        for q in Q:
            v -= np.nansum(v * q) * q
        denom = np.sqrt(np.nansum(v * v))
        if denom == 0 or not np.isfinite(denom):
            # if degenerate, remove and continue
            selected.pop()
            remaining.remove(jstar)
            continue
        q_new = v / denom
        Q.append(q_new)

        # Orthogonalize all columns against q_new (one MGS step)
        for j in remaining:
            residual[:, j] -= np.nansum(residual[:, j] * q_new) * q_new

        remaining.remove(jstar)

    # Map indices back to column names
    return [X.columns[i] for i in selected]


##################### panel_quick_pick_per_ticker ##########################
import numpy as np
import pandas as pd

# ---------------------------
# Core small, safe utilities
# ---------------------------

def _oos_r2(y_true, y_pred, y_bench):
    """Campbell–Thompson OOS R^2 vs a mean benchmark; NA-safe."""
    v = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_bench)
    if v.sum() == 0:
        return np.nan
    e  = y_true[v] - y_pred[v]
    eb = y_true[v] - y_bench[v]
    den = np.sum(eb**2)
    return 1 - (np.sum(e**2) / den if den > 0 else np.nan)

def _is_expanding(win):
    """True if window means expanding (None/NaN/'expanding')."""
    return (win is None) or (isinstance(win, float) and np.isnan(win)) or pd.isna(win) \
           or (isinstance(win, str) and win.lower() == "expanding")

def _normalize_windows(windows):
    """
    Turn a windows spec into a list of labels:
      - None or 'expanding' -> 'expanding'
      - numeric-like -> int(window)
    """
    norm = []
    for w in windows:
        if _is_expanding(w):
            norm.append("expanding")
        else:
            # accept ints/floats/str-digits
            if isinstance(w, (int, np.integer)):
                norm.append(int(w))
            elif isinstance(w, float) and np.isfinite(w):
                norm.append(int(round(w)))
            elif isinstance(w, str) and w.isdigit():
                norm.append(int(w))
            else:
                # unknown token -> skip
                continue
    # de-duplicate while preserving order
    seen, out = set(), []
    for w in norm:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def _window_to_int_or_none(window_label):
    """'expanding' -> None; numeric label -> int."""
    return None if _is_expanding(window_label) else int(window_label)

# ---------------------------
# Feature selection (y-aware)
# ---------------------------

def _forward_path_safe(X, y, k_max=8, min_rows=5):
    """
    Greedy forward selection on residuals (OMP-style) with NA/constant guards.
    Returns a list of column NAMES in import order.
    """
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if df.shape[0] < min_rows:
        return []
    # drop constants (std==0)
    cols = [c for c in X.columns if df[c].std(ddof=1) > 0]
    if not cols:
        return []
    chosen, r = [], df["__y__"].to_numpy()

    for _ in range(min(k_max, len(cols))):
        remaining = [c for c in cols if c not in chosen]
        if not remaining:
            break
        # correlation with residuals (auto-aligned; NA-safe)
        corrs = df[remaining].corrwith(pd.Series(r, index=df.index)).abs()
        corrs = corrs.replace([np.inf, -np.inf], np.nan).dropna()
        if corrs.empty:
            break
        jname = corrs.idxmax()
        chosen.append(jname)
        # update residual on same rows
        Xsel = df[chosen].to_numpy()
        beta, *_ = np.linalg.lstsq(Xsel, df["__y__"].to_numpy(), rcond=None)
        r = df["__y__"].to_numpy() - Xsel @ beta
    return chosen

# ---------------------------
# Single-ticker quick model
# ---------------------------

def quick_pick_model_single(
    y_series: pd.Series,
    X: pd.DataFrame,
    n_folds=3,
    k_max=8,
    windows=(None, 120, 252),
    test_frac=0.15,
    min_clean_obs=100,
):
    """
    Per ticker:
      1) dropna on y, align X by index, dropna jointly,
      2) build y-aware path on first training slice,
      3) score a tiny grid across k and windows,
      4) refit winner and report test OOS R^2.
    Returns (winner_dict, grid_df)
    """
    # 1) Per-ticker clean
    y_t = y_series.dropna()
    if y_t.empty:
        raise ValueError("Ticker has no non-NaN observations.")

    X_t = X.reindex(y_t.index)
    df_all = pd.concat([y_t.rename("__y__"), X_t], axis=1).dropna()
    if df_all.shape[0] < min_clean_obs:
        raise ValueError("Not enough clean observations after alignment and dropna().")

    y_all = df_all["__y__"]
    X_all = df_all.drop(columns="__y__")

    # 2) Walk-forward split bounds
    T = len(y_all)
    test_n = int(np.floor(T * test_frac))
    trainval_end = T - test_n
    if trainval_end <= 0:
        raise ValueError("Test fraction too large for series length.")
    # start folds after ~50% of history
    fold_edges = np.linspace(int(T * 0.5), trainval_end, n_folds + 1, dtype=int)

    # 3) Fix the feature order on the first training slice
    path_cols = _forward_path_safe(X_all.iloc[:fold_edges[0]], y_all.iloc[:fold_edges[0]], k_max=k_max)
    if not path_cols:
        raise ValueError("No usable predictors after cleaning (all NaN/constant?).")

    # 4) Evaluate grid
    win_specs = _normalize_windows(windows)
    results = []

    for k in range(1, min(k_max, len(path_cols)) + 1):
        cols = path_cols[:k]
        for Wlab in win_specs:
            preds, bench, truth = [], [], []
            for i in range(n_folds):
                tr_end, va_end = fold_edges[i], fold_edges[i+1]
                y_tr, X_tr = y_all.iloc[:tr_end], X_all.iloc[:tr_end][cols]
                y_va, X_va = y_all.iloc[tr_end:va_end], X_all.iloc[tr_end:va_end][cols]

                if _is_expanding(Wlab):
                    # expanding OLS
                    df_tr = pd.concat([y_tr.rename("__y__"), X_tr], axis=1).dropna()
                    if df_tr.shape[0] >= len(cols) + 2:
                        beta, *_ = np.linalg.lstsq(df_tr[cols].to_numpy(), df_tr["__y__"].to_numpy(), rcond=None)
                        y_hat = (X_va @ beta).to_numpy()
                    else:
                        y_hat = np.full(len(y_va), np.nan)
                else:
                    # rolling OLS with fixed window
                    Wint = _window_to_int_or_none(Wlab)
                    y_hat = np.full(len(y_va), np.nan)
                    for t in range(len(y_va)):
                        end_idx = tr_end + t
                        start_idx = max(0, end_idx - Wint)
                        y_win = y_all.iloc[start_idx:end_idx]
                        X_win = X_all.iloc[start_idx:end_idx][cols]
                        df_win = pd.concat([y_win.rename("__y__"), X_win], axis=1).dropna()
                        if df_win.shape[0] >= len(cols) + 2:
                            beta_t, *_ = np.linalg.lstsq(df_win[cols].to_numpy(), df_win["__y__"].to_numpy(), rcond=None)
                            y_hat[t] = float(X_va.iloc[t].to_numpy() @ beta_t)

                # expanding-mean benchmark up to tr_end-1
                m = y_all.iloc[:tr_end].expanding().mean().reindex(y_va.index, method="pad").to_numpy()
                preds.append(y_hat); bench.append(m); truth.append(y_va.to_numpy())

            y_pred  = np.concatenate(preds)
            y_bench = np.concatenate(bench)
            y_true  = np.concatenate(truth)
            score = _oos_r2(y_true, y_pred, y_bench)

            results.append({"k": int(k), "window": Wlab, "oos_r2": score, "cols": cols})

    df_res = pd.DataFrame(results).sort_values(["oos_r2", "k"], ascending=[False, True]).reset_index(drop=True)
    winner = df_res.iloc[0].to_dict()

    # 5) Final test read-out
    cols = winner["cols"]
    y_tv, X_tv = y_all.iloc[:trainval_end], X_all.iloc[:trainval_end][cols]
    y_te, X_te = y_all.iloc[trainval_end:], X_all.iloc[trainval_end:][cols]

    if _is_expanding(winner["window"]):
        df_tv = pd.concat([y_tv.rename("__y__"), X_tv], axis=1).dropna()
        if df_tv.shape[0] >= len(cols) + 2:
            beta, *_ = np.linalg.lstsq(df_tv[cols].to_numpy(), df_tv["__y__"].to_numpy(), rcond=None)
            y_hat_te = (X_te @ beta).to_numpy()
        else:
            y_hat_te = np.full(len(y_te), np.nan)
    else:
        Wint = _window_to_int_or_none(winner["window"])  # guaranteed int here
        y_hat_te = np.full(len(y_te), np.nan)
        for t in range(len(y_te)):
            end_idx = trainval_end + t
            start_idx = max(0, end_idx - Wint)
            y_win = y_all.iloc[start_idx:end_idx]
            X_win = X_all.iloc[start_idx:end_idx][cols]
            df_win = pd.concat([y_win.rename("__y__"), X_win], axis=1).dropna()
            if df_win.shape[0] >= len(cols) + 2:
                beta_t, *_ = np.linalg.lstsq(df_win[cols].to_numpy(), df_win["__y__"].to_numpy(), rcond=None)
                y_hat_te[t] = float(X_te.iloc[t].to_numpy() @ beta_t)

    bench_te = y_all.iloc[:trainval_end].expanding().mean().reindex(y_te.index, method="pad").to_numpy()
    winner.update({"test_oos_r2": float(_oos_r2(y_te.to_numpy(), y_hat_te, bench_te))})
    return winner, df_res

# ---------------------------
# Panel wrapper (per ticker)
# ---------------------------

def panel_quick_pick_per_ticker(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    n_folds=3,
    k_max=8,
    windows=(None, 120, 252),
    test_frac=0.15,
    store_grids=False,
    quiet=False,
    min_clean_obs=100,
):
    summaries, details = [], {}
    for ticker in Y.columns:
        try:
            winner, grid = quick_pick_model_single(
                Y[ticker], X,
                n_folds=n_folds, k_max=k_max,
                windows=windows, test_frac=test_frac,
                min_clean_obs=min_clean_obs,
            )
            best_oos = float(grid.iloc[0]["oos_r2"])
            win_label = winner.get("window")  # already 'expanding' or int

            summaries.append({
                "ticker": ticker,
                "oos_r2_best_cv": best_oos,
                "k": int(winner["k"]),
                "window": win_label,  # <-- never cast blindly
                "test_oos_r2": float(winner.get("test_oos_r2", float("nan"))),
                "features": ",".join(winner["cols"]),
            })

            details[ticker] = {"winner": winner, "grid": grid if store_grids else None}
            if not quiet:
                print(f"{ticker}: CV OOS R\u00b2={best_oos:.4f} | test OOS R\u00b2={winner['test_oos_r2']:.4f} "
                      f"| k={winner['k']} | window={win_label}")
        except Exception as e:
            summaries.append({
                "ticker": ticker, "oos_r2_best_cv": np.nan,
                "k": None, "window": None, "test_oos_r2": np.nan,
                "features": "", "error": str(e)
            })
            details[ticker] = {"error": str(e)}
            if not quiet:
                print(f"{ticker}: ERROR -> {e}")

    summary_df = pd.DataFrame(summaries).set_index("ticker").sort_values("oos_r2_best_cv", ascending=False)
    return summary_df, details







###################### Script Run ##############################

def run_script():
    from Modules.ticker_config import STOCK_TICKERS, ETF_TICKERS
    from Modules.stock_data_functions import TickerComparison, calc_log_rets
    stock_obj = TickerComparison(STOCK_TICKERS, '2023-09-11', period='day',
                                 date_updated=False, fetch_in_chunks=False,
                                 waiting_time=1, end_date='2025-11-05')
    reg_obj   = TickerComparison(ETF_TICKERS, '2023-09-11', period='day',
                                 date_updated=False, fetch_in_chunks=False,
                                 waiting_time=1, end_date='2025-11-05')

    stock_prices = stock_obj.tickers_stocks_prices.loc[:, (slice(None), 'vwap')].copy()
    stock_prices = stock_prices.droplevel(1, axis=1)
    stock_rets   = stock_prices.apply(calc_log_rets)

    reg_p        = reg_obj.tickers_stocks_prices.loc[:, (slice(None), 'vwap')].copy().apply(calc_log_rets)
    reg_rets     = reg_p.droplevel(1, axis=1)

    not_present_reg = []
    for i in reg_rets.columns:
        if reg_rets.first_valid_index() > dt.datetime(2023, 11, 15):
            not_present_reg.append(i)

    not_present = []
    for i in STOCK_TICKERS:
        if stock_rets.first_valid_index() > dt.datetime(2023, 11, 15):
            not_present.append(i)

    reg_rets    = reg_rets.drop(columns=not_present_reg)
    stock_rets  = stock_rets.drop(columns=not_present)

    # Just drops when there are more than 22 missing entries
    null_stocks = stock_rets.isnull().sum()
    good_stocks = null_stocks[null_stocks < 22].index.tolist()
    stock_rets  = stock_rets[good_stocks].copy()
    return reg_rets, stock_rets

if __name__ == "__main__":


    # fred_api = Fred('e48d0413b1cd0a3b30b58d42225373de')

    # stock_obj = TickerComparison(STOCK_TICKERS, '2023-09-11', period='day',
    #                              date_updated=False, fetch_in_chunks=False,
    #                              waiting_time=1, end_date='2025-11-05')
    # reg_obj   = TickerComparison(ETF_TICKERS, '2023-09-11', period='day',
    #                              date_updated=False, fetch_in_chunks=False,
    #                              waiting_time=1, end_date='2025-11-05')

    # stock_prices = stock_obj.tickers_stocks_prices.loc[:, (slice(None), 'vwap')].copy()
    # stock_prices = stock_prices.droplevel(1, axis=1)
    # stock_rets   = stock_prices.apply(calc_log_rets)

    # reg_p        = reg_obj.tickers_stocks_prices.loc[:, (slice(None), 'vwap')].copy().apply(calc_log_rets)
    # reg_rets     = reg_p.droplevel(1, axis=1)
    
    # yield_spread = fred_api.get_series('T10Y3M', observation_start='2023-09-11').dropna().rename('T10Y3m')
    # vix          = fred_api.get_series('VIXCLS', observation_start='2023-09-11').dropna().rename('VIX')
    # move         = fred_api.get_series('BAMLC0A0CM', observation_start='2023-09-11').dropna().rename('MOVE')
    # dollar       = fred_api.get_series('DTWEXBGS', observation_start='2023-09-11').dropna().rename('DXY')
    # btc          = fred_api.get_series('CBBTCUSD', observation_start='2023-09-11').dropna().rename('BTC')
    # btc_rets     = calc_log_rets(btc)
    # dollar_rets  = calc_log_rets(dollar)

    # reg_rets = pd.concat([reg_rets, vix, move, yield_spread, btc_rets, dollar_rets], axis=1)

    reg_rets, stock_rets = run_script()
    
    # Z score everything
    X_full           = reg_rets.apply(calc_zscore, winsor=0.01)
    y                = stock_rets.apply(calc_zscore, winsor=0.01)
    # Choose regressors
    regressor_chosen = choose_etf_representatives(X_full,20, corr_cap=0.95)
    X_reduced        = X_full[regressor_chosen].copy()

    summary, details = panel_quick_pick_per_ticker(y, X_reduced, n_folds=3, k_max=8, windows=(None, 60, 120, 252), store_grids=True)

