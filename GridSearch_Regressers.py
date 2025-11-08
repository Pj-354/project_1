# ===================== end_to_end_factor_selector.py =====================
from __future__ import annotations
from typing import Sequence, Optional, Tuple, Dict, Any, List
import itertools as it
import numpy as np
import pandas as pd
from fredapi import Fred
import datetime as dt
from typing import Union, Optional
import Modules.stock_data_functions
import Modules.regression_functions
from Modules.stock_data_functions import TickerComparison, calc_log_rets
from Modules.regression_functions import fit_returns_ols, orthogonalize_factors, rolling_ols
from pathlib import Path
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

####################### Step 2 : Selecting winners and finding models ##############################
def select_winner_from_grid(grid_df: pd.DataFrame,
                            similarity_tol: float = 0.005,
                            prefer_longer_window: bool = True,
                            verbose: bool = False) -> dict:
    """
    Fast, deterministic chooser:
      - Find best by oos_r2.
      - Keep configs within `similarity_tol` of best (e.g., 0.5 pp).
      - Among those, pick smallest k; tie-break by window:
          * if prefer_longer_window: expanding < 120 < 252 < 504 (longer is 'simpler' to operate)
          * else: expanding is preferred, then shorter numeric window.
    Returns dict(row) for the chosen config.
    """
    if grid_df is None or grid_df.empty:
        raise ValueError("Empty grid for this ticker.")

    # Ensure correct types
    g = grid_df.copy()
    if "oos_r2" not in g.columns or "k" not in g.columns or "window" not in g.columns or "cols" not in g.columns:
        raise ValueError("Grid missing required columns: ['oos_r2','k','window','cols'].")

    # Clean window labels
    g["window"] = g["window"].apply(lambda w: "expanding" if _is_expanding(w) else int(w))

    # Best score and candidate set
    g = g.sort_values(["oos_r2", "k"], ascending=[False, True]).reset_index(drop=True)
    best = g.iloc[0]
    cutoff = best["oos_r2"] - similarity_tol
    candidates = g[g["oos_r2"] >= cutoff].copy()

    # Simplest k
    min_k = candidates["k"].min()
    candidates = candidates[candidates["k"] == min_k]

    # Window preference ordering
    def win_rank(w):
        if w == "expanding": return 0
        # numeric
        return (w if prefer_longer_window else -w)

    candidates["__wrank"] = candidates["window"].apply(win_rank)
    chosen = candidates.sort_values(["__wrank"]).iloc[0].drop(labels="__wrank")

    if verbose:
        print(f"[select] best={best['oos_r2']:.4f}, tol={similarity_tol:.4f}, "
              f"picked k={int(chosen['k'])}, window={chosen['window']}, "
              f"score={chosen['oos_r2']:.4f}")

    return chosen.to_dict()

# ----------------------------------------------------
# 2) Refit & test (train+val) for a chosen config
# ----------------------------------------------------
def refit_and_test_ticker(y_series: pd.Series,
                          X: pd.DataFrame,
                          config: dict,
                          n_folds: int = 3,
                          test_frac: float = 0.15,
                          verbose: bool = False) -> dict:
    """
    Re-aligns per ticker, uses the SAME split logic, refits winner on train+val,
    and computes test OOS R^2 + returns final coefficients.
    """
    # Per-ticker clean & align
    y_t = y_series.dropna()
    X_t = X.reindex(y_t.index)
    df_all = pd.concat([y_t.rename("__y__"), X_t], axis=1).dropna()
    if df_all.shape[0] < 100:
        raise ValueError("Not enough clean observations after alignment and dropna().")

    y_all = df_all["__y__"]; X_all = df_all.drop(columns="__y__")

    T = len(y_all)
    test_n = int(np.floor(T * test_frac))
    trainval_end = T - test_n
    if trainval_end <= 0:
        raise ValueError("Test fraction too large for series length.")
    fold_edges = np.linspace(int(T * 0.5), trainval_end, n_folds + 1, dtype=int)

    cols = config["cols"]
    win  = config["window"]
    win_label = "expanding" if _is_expanding(win) else int(win)

    # Final coefficients (at deploy time)
    if _is_expanding(win_label):
        df_tv = pd.concat([y_all.iloc[:trainval_end].rename("__y__"),
                           X_all.iloc[:trainval_end][cols]], axis=1).dropna()
        if df_tv.shape[0] < len(cols) + 2:
            raise ValueError("Too few rows to fit expanding OLS.")
        beta, *_ = np.linalg.lstsq(df_tv[cols].to_numpy(), df_tv["__y__"].to_numpy(), rcond=None)
        coefs = pd.Series(beta, index=cols)
        # Predict test with frozen coefs
        y_hat_te = (X_all.iloc[trainval_end:][cols] @ coefs).to_numpy()
    else:
        Wint = _window_to_int_or_none(win_label)
        # Take the last rolling-window fit ending at trainval_end
        start_idx = max(0, trainval_end - Wint)
        df_win = pd.concat([y_all.iloc[start_idx:trainval_end].rename("__y__"),
                            X_all.iloc[start_idx:trainval_end][cols]], axis=1).dropna()
        if df_win.shape[0] < len(cols) + 2:
            raise ValueError("Too few rows to fit rolling OLS.")
        beta_tv, *_ = np.linalg.lstsq(df_win[cols].to_numpy(), df_win["__y__"].to_numpy(), rcond=None)
        coefs = pd.Series(beta_tv, index=cols)

        # Rolling refit during test (as in your CV scoring)
        y_hat_te = np.full(test_n, np.nan)
        for t in range(test_n):
            end_idx = trainval_end + t
            s_idx   = max(0, end_idx - Wint)
            df_win = pd.concat([y_all.iloc[s_idx:end_idx].rename("__y__"),
                                X_all.iloc[s_idx:end_idx][cols]], axis=1).dropna()
            if df_win.shape[0] >= len(cols) + 2:
                beta_t, *_ = np.linalg.lstsq(df_win[cols].to_numpy(), df_win["__y__"].to_numpy(), rcond=None)
                y_hat_te[t] = float(X_all.iloc[trainval_end + t][cols].to_numpy() @ beta_t)

    # Benchmark & test OOS R^2
    bench_te = y_all.iloc[:trainval_end].expanding().mean().reindex(y_all.iloc[trainval_end:].index, method="pad").to_numpy()
    test_r2 = _oos_r2(y_all.iloc[trainval_end:].to_numpy(), y_hat_te, bench_te)

    if verbose:
        print(f"[refit] window={win_label}, k={len(cols)}, test OOS R^2={test_r2:.4f}")

    return {
        "window": win_label,
        "k": int(len(cols)),
        "cols": list(cols),
        "coefs": coefs,           # pandas Series indexed by factor names
        "test_oos_r2": float(test_r2),
    }

# ----------------------------------------------------
# 3) Panel: choose winner, refit, summarize
# ----------------------------------------------------
def finalize_models_panel(Y: pd.DataFrame,
                          X: pd.DataFrame,
                          details: dict,
                          similarity_tol: float = 0.005,
                          prefer_longer_window: bool = True,
                          n_folds: int = 3,
                          test_frac: float = 0.15,
                          verbose: bool = True):
    """
    For each ticker in `details`, pick the winner using a tolerance rule,
    refit, and collect final coefficients + test OOS R^2.
    """
    final_summary = []
    final_models  = {}
    for ticker, obj in details.items():
        try:
            grid = obj.get("grid")
            if grid is None or grid.empty:
                raise ValueError("Missing/empty grid for ticker.")
            chosen = select_winner_from_grid(grid,
                                             similarity_tol=similarity_tol,
                                             prefer_longer_window=prefer_longer_window,
                                             verbose=verbose)
            # Refit & test
            result = refit_and_test_ticker(Y[ticker], X, chosen, n_folds=n_folds, test_frac=test_frac, verbose=verbose)
            final_models[ticker] = result
            final_summary.append({
                "ticker": ticker,
                "chosen_oos_r2_cv": float(chosen["oos_r2"]),
                "k": int(result["k"]),
                "window": result["window"],
                "test_oos_r2": float(result["test_oos_r2"]),
                "features": ",".join(result["cols"]),
            })
            if verbose:
                print(f"[done] {ticker}: chosen CV={chosen['oos_r2']:.4f}, "
                      f"test={result['test_oos_r2']:.4f}, k={result['k']}, window={result['window']}")
        except Exception as e:
            if verbose:
                print(f"[error] {ticker}: {e}")
            final_summary.append({
                "ticker": ticker, "chosen_oos_r2_cv": np.nan, "k": None,
                "window": None, "test_oos_r2": np.nan, "features": "", "error": str(e)
            })
            final_models[ticker] = {"error": str(e)}

    summary_df = pd.DataFrame(final_summary).set_index("ticker").sort_values("chosen_oos_r2_cv", ascending=False)
    return summary_df, final_models

# ----------------------------------------------------
# 4) Consistency tests (sanity checks)
# ----------------------------------------------------
def run_consistency_checks(summary_df: pd.DataFrame,
                           final_models: dict,
                           X: pd.DataFrame,
                           tol_k_match: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Basic assertions to flag misconfigurations:
      - k equals number of features
      - window label valid
      - all selected features exist in X
      - test_oos_r2 is finite if CV score was finite
    """
    rows = []
    for ticker, row in summary_df.iterrows():
        checks = {"ticker": ticker, "ok": True, "messages": []}
        fm = final_models.get(ticker, {})
        cols = fm.get("cols", [])
        k = fm.get("k", None)
        win = fm.get("window", None)
        test_r2 = fm.get("test_oos_r2", np.nan)
        cv_r2 = row.get("chosen_oos_r2_cv", np.nan)

        # k vs features
        if k is not None and (k != len(cols)):
            checks["ok"] = False; checks["messages"].append(f"k={k} != len(cols)={len(cols)}")

        # window valid
        if not (_is_expanding(win) or isinstance(win, int)):
            checks["ok"] = False; checks["messages"].append(f"invalid window label: {win}")

        # features existence
        missing = [c for c in cols if c not in X.columns]
        if missing:
            checks["ok"] = False; checks["messages"].append(f"missing factors in X: {missing}")

        # test R^2 finite if CV finite
        if np.isfinite(cv_r2) and not np.isfinite(test_r2):
            checks["ok"] = False; checks["messages"].append("finite CV but NaN test_oos_r2")

        rows.append(checks)
        if verbose and (not checks["ok"]):
            print(f"[check] {ticker} FAILED: {' | '.join(checks['messages'])}")

    return pd.DataFrame(rows).set_index("ticker")

############### Regressor Reduction Diagnostics ###############

def summarise_reduction(chosen_regressors, reg_rets, winsor=(0.01, 0.99)):
    def _aligned_rows(X_full: pd.DataFrame, X_reduced: pd.DataFrame) -> pd.Index:
        idx = X_full.index.intersection(X_reduced.index)
        m1 = X_full.loc[idx].notna().all(axis=1)
        m2 = X_reduced.loc[idx].notna().all(axis=1)
        return idx[m1 & m2]
    def spanning_r2_scores(X_full: pd.DataFrame, X_reduced: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
        rows = _aligned_rows(X_full, X_reduced)
        Xf, Xr = X_full.loc[rows], X_reduced.loc[rows]
        r2 = {}
        for col in Xf.columns:
            Xr_loo = Xr.drop(columns=[col], errors="ignore")
            if Xr_loo.shape[1] == 0:
                r2[col] = np.nan; continue
            df = pd.concat([Xf[[col]], Xr_loo], axis=1).dropna()
            if df.shape[0] < Xr_loo.shape[1] + 2:
                r2[col] = np.nan; continue
            yv = df.iloc[:, 0].values
            Xv = np.c_[np.ones(len(df)), df.iloc[:, 1:].values]
            beta, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
            yhat = Xv @ beta
            ss_res = np.sum((yv - yhat)**2)
            ss_tot = np.sum((yv - yv.mean())**2)
            r2[col] = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        r2s = pd.Series(r2).sort_values(ascending=False)
        summary = {
            "median_R2": float(r2s.median(skipna=True)),
            "p5_R2": float(r2s.quantile(0.05)),
            "min_R2": float(r2s.min(skipna=True)),
            "coverage": int(r2s.notna().sum()),
            "total": int(len(r2s)),
        }
        return r2s, summary
    def principal_angles_summary(X_full: pd.DataFrame, X_reduced: pd.DataFrame, var_thresh: float = 0.95) -> Dict[str, float]:
        rows = _aligned_rows(X_full, X_reduced)
        A = X_full.loc[rows].to_numpy()
        B = X_reduced.loc[rows].to_numpy()
        # center columns (does not change subspaces up to translation)
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)

        # SVD of A to get top-k observation-space PCs (U_k)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        var = S**2
        k = max(1, int(np.searchsorted(np.cumsum(var)/var.sum(), var_thresh) + 1))
        Uk = U[:, :k]                     # orthonormal basis for A's top-k subspace (in R^n)

        # Orthonormal basis for column space of B (in R^n)
        Qb, _ = np.linalg.qr(B, mode="reduced")

        # Principal angles via SVD of Uk^T Qb  (angles' cosines are the singular values)
        s = np.clip(np.linalg.svd(Uk.T @ Qb, compute_uv=False), -1.0, 1.0)
        ang = np.degrees(np.arccos(s))
        return {
            "max_angle_deg": float(ang.max()) if ang.size else np.nan,
            "mean_angle_deg": float(ang.mean()) if ang.size else np.nan,
            "min_cos": float(s.min()) if s.size else np.nan,
            "mean_cos": float(s.mean()) if s.size else np.nan,
            "k_pc": int(k),
        }
    # 3) Effective rank (entropy of spectrum), aligned
    def _effective_rank_from_corr_aligned(X: pd.DataFrame) -> float:
        Z = X.to_numpy()
        C = np.corrcoef(Z, rowvar=False)
        w = np.linalg.eigvalsh((C + C.T) / 2.0)
        w = np.clip(w, 0, None)
        if w.sum() <= 0: return 0.0
        p = w / w.sum()
        H = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
        return float(np.exp(H))
    def effective_rank_compare(X_full: pd.DataFrame, X_reduced: pd.DataFrame) -> Dict[str, float]:
        rows = _aligned_rows(X_full, X_reduced)
        Xf, Xr = X_full.loc[rows], X_reduced.loc[rows]
        er_full = _effective_rank_from_corr_aligned(Xf)
        er_red  = _effective_rank_from_corr_aligned(Xr)
        return {
            "erank_full": er_full,
            "erank_reduced": er_red,
            "ratio_reduced_over_full": float(er_red/er_full) if er_full > 0 else np.nan,
        }
    # 4) VIF & condition indices (use the SAME aligned rows)
    def vif_series(X: pd.DataFrame, rows: pd.Index = None) -> pd.Series:
        Z = X.loc[rows] if rows is not None else X.dropna(how="any")
        C = np.corrcoef(Z.values, rowvar=False)
        C = (C + C.T) / 2.0
        C_inv = np.linalg.pinv(C)
        return pd.Series(np.diag(C_inv), index=Z.columns, name="VIF")
    def condition_indices(X: pd.DataFrame, rows: pd.Index = None) -> pd.Series:
        Z = X.loc[rows] if rows is not None else X.dropna(how="any")
        C = np.corrcoef(Z.values, rowvar=False)
        C = (C + C.T) / 2.0
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        w = np.clip(w, 1e-12, None)
        ci = np.sqrt(w[0] / w)
        return pd.Series(ci, index=[f"CI_{i+1}" for i in range(len(ci))], name="ConditionIndex")
    
    X_full = reg_rets.apply(calc_zscore, winsor=winsor)
    X_reduced = X_full[chosen_regressors].copy()
    rows            = _aligned_rows(X_full, X_reduced)
    r2s, r2_summary = spanning_r2_scores(X_full, X_reduced)
    angles          = principal_angles_summary(X_full, X_reduced, var_thresh=0.95)
    er              = effective_rank_compare(X_full, X_reduced)
    vifs            = vif_series(X_reduced, rows=rows)          # or X_full, same rows
    cis             = condition_indices(X_reduced, rows=rows)

    return r2_summary, angles, er, vifs, cis

###################### Script Run ##############################
def collect_model_comps(rets, reg_rets, final_models, stocks, winsor_grid, window,
                        out_dir="results_model_test", verbose=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _wlab(w):
        return f"{w[0]:.4f}-{w[1]:.4f}" if isinstance(w, (list, tuple)) and len(w) == 2 else str(w)

    def _safe_parquet(df: pd.DataFrame, path: Path, note: str):
        try:
            df.to_parquet(path)
            if verbose: print(f"[save] {note} → {path.name}")
        except Exception as e:
            print(f"[warn] parquet save skipped ({note}): {e}")

    def _save_diag_dict(diag: dict, base_path: Path, tag: str):
        """Split a diagnostics dict: save any DataFrame/Series under their own files; pack scalars to one row."""
        scalars = {}
        skipped = []
        for k, v in diag.items():
            try:
                if isinstance(v, pd.DataFrame):
                    _safe_parquet(v, base_path.with_name(f"{base_path.stem}__{tag}_{k}.parquet"), f"{tag}.{k}")
                elif isinstance(v, pd.Series):
                    _safe_parquet(v.to_frame("value"), base_path.with_name(f"{base_path.stem}__{tag}_{k}.parquet"), f"{tag}.{k}")
                elif np.isscalar(v) or isinstance(v, (str, bytes, bool)):
                    scalars[k] = v
                elif isinstance(v, (list, tuple, np.ndarray)) and (len(v) == 0 or np.isscalar(v[0]) or isinstance(v[0], (str, bytes, bool))):
                    _safe_parquet(pd.DataFrame({k: list(v)}), base_path.with_name(f"{base_path.stem}__{tag}_{k}.parquet"), f"{tag}.{k}")
                else:
                    skipped.append(f"{k}<{type(v).__name__}>")
            except Exception as e:
                skipped.append(f"{k}<{type(v).__name__}>: {e}")
        if scalars:
            _safe_parquet(pd.DataFrame([scalars]), base_path.with_name(f"{base_path.stem}__{tag}_scalars.parquet"), f"{tag}.scalars")
        if skipped and verbose:
            print(f"[info] skipped (not parquet-serializable) in {tag}: {skipped}")

    for s in stocks:
        # get regressors for this ticker
        try:
            fm = final_models.get(s, {})
            regressors = fm.get("cols") or fm.get("features")
            if regressors is None:
                raise ValueError("no regressors in final_models")
            if isinstance(regressors, str):
                regressors = [r.strip() for r in regressors.split(",") if r.strip()]

            y_s = rets[s].copy()
            X_s = reg_rets[regressors].copy()
            tmp = pd.concat([y_s.rename("y"), X_s], axis=1, join="inner").dropna()
            if tmp.shape[0] < max(20, len(regressors) + 5):
                raise ValueError("not enough overlapping rows after alignment")
            y_s = tmp["y"]; X_s = tmp[regressors]
            if verbose: print(f"[{s}] rows={len(y_s)} k={len(regressors)}")
        except Exception as e:
            if verbose: print(f"[skip] {s}: {e}")
            continue

        for winsor in winsor_grid:
            wlab = _wlab(winsor)

            # ----- SIMPLE OLS -----
            try:
                res, coef_tbl, diag, summ_txt = fit_returns_ols(y_s, X_s, winsorize_quantiles=winsor)
                if isinstance(coef_tbl, pd.Series): coef_tbl = coef_tbl.to_frame("value")
                if isinstance(coef_tbl, pd.DataFrame):
                    _safe_parquet(coef_tbl, Path(out_dir, f"{s}__winsor_{wlab}__simple_coef.parquet"), f"{s} simple_ols coef")
                else:
                    if verbose: print(f"[info] {s} simple_ols: coef_table not DataFrame/Series; skipped")

                # diagnostics may contain nested DataFrames (e.g., VIF) → save separately
                if isinstance(diag, dict):
                    _save_diag_dict(diag, Path(out_dir, f"{s}__winsor_{wlab}__simple_diag.parquet"), "simple_diag")
                elif isinstance(diag, pd.DataFrame):
                    _safe_parquet(diag, Path(out_dir, f"{s}__winsor_{wlab}__simple_diag.parquet"), f"{s} simple_ols diagnostics")
                elif isinstance(diag, pd.Series):
                    _safe_parquet(diag.to_frame("value"), Path(out_dir, f"{s}__winsor_{wlab}__simple_diag.parquet"), f"{s} simple_ols diagnostics")
                else:
                    if verbose: print(f"[info] {s} simple_ols: diagnostics not parquet-serializable; skipped")

                _safe_parquet(pd.DataFrame({"summary_text": [str(summ_txt)]}),
                              Path(out_dir, f"{s}__winsor_{wlab}__simple_summary.parquet"), f"{s} simple_ols summary")
            except Exception as e:
                print(f"[error] {s} simple_ols winsor={wlab}: {e}")

            # ----- ORTHO OLS -----
            try:
                ortho_out = orthogonalize_factors(X_s, normalize="match_original_std")
                X_ortho = ortho_out[0] if isinstance(ortho_out, (list, tuple)) else ortho_out
                res_o, coef_tbl_o, diag_o, summ_txt_o = fit_returns_ols(y_s, X_ortho, winsorize_quantiles=winsor)

                if isinstance(coef_tbl_o, pd.Series): coef_tbl_o = coef_tbl_o.to_frame("value")
                if isinstance(coef_tbl_o, pd.DataFrame):
                    _safe_parquet(coef_tbl_o, Path(out_dir, f"{s}__winsor_{wlab}__ortho_coef.parquet"), f"{s} ortho_ols coef")
                else:
                    if verbose: print(f"[info] {s} ortho_ols: coef_table not DataFrame/Series; skipped")

                if isinstance(diag_o, dict):
                    _save_diag_dict(diag_o, Path(out_dir, f"{s}__winsor_{wlab}__ortho_diag.parquet"), "ortho_diag")
                elif isinstance(diag_o, pd.DataFrame):
                    _safe_parquet(diag_o, Path(out_dir, f"{s}__winsor_{wlab}__ortho_diag.parquet"), f"{s} ortho_ols diagnostics")
                elif isinstance(diag_o, pd.Series):
                    _safe_parquet(diag_o.to_frame("value"), Path(out_dir, f"{s}__winsor_{wlab}__ortho_diag.parquet"), f"{s} ortho_ols diagnostics")
                else:
                    if verbose: print(f"[info] {s} ortho_ols: diagnostics not parquet-serializable; skipped")

                _safe_parquet(pd.DataFrame({"summary_text": [str(summ_txt_o)]}),
                              Path(out_dir, f"{s}__winsor_{wlab}__ortho_summary.parquet"), f"{s} ortho_ols summary")
            except Exception as e:
                print(f"[error] {s} ortho_ols winsor={wlab}: {e}")

            # ----- ROLLING OLS -----
            try:
                # use orthogonalized X for rolling (as in your prior code)
                ortho_out = orthogonalize_factors(X_s, normalize="match_original_std")
                X_ortho = ortho_out[0] if isinstance(ortho_out, (list, tuple)) else ortho_out

                out = rolling_ols(y_s, X_ortho, window=window, winsorize_quantiles=winsor, oos_predict_next=True)

                if isinstance(out.get("betas"), pd.DataFrame):
                    _safe_parquet(out["betas"], Path(out_dir, f"{s}__winsor_{wlab}__rolling_betas.parquet"), f"{s} rolling betas")
                if isinstance(out.get("r2_adj"), pd.DataFrame):
                    _safe_parquet(out["r2_adj"], Path(out_dir, f"{s}__winsor_{wlab}__rolling_r2adj.parquet"), f"{s} rolling r2_adj")
                elif isinstance(out.get("r2_adj"), pd.Series):
                    _safe_parquet(out["r2_adj"].to_frame("r2_adj"), Path(out_dir, f"{s}__winsor_{wlab}__rolling_r2adj.parquet"), f"{s} rolling r2_adj")
                if isinstance(out.get("yhat_next"), (pd.Series, pd.DataFrame)):
                    yhat = out["yhat_next"] if isinstance(out["yhat_next"], pd.Series) else out["yhat_next"].iloc[:, 0]
                    err = pd.concat([y_s.rename("y"), yhat.rename("pred")], axis=1, join="inner").dropna()
                    err["error"] = err["y"] - err["pred"]
                    _safe_parquet(err, Path(out_dir, f"{s}__winsor_{wlab}__rolling_errors.parquet"), f"{s} rolling errors")
                    oos_val = _oos_r2(err["y"].to_numpy(), err["pred"].to_numpy(), np.full(len(err), y_s.mean()))
                    _safe_parquet(pd.DataFrame([{"ticker": s, "winsor": _wlab(winsor), "window": int(window), "oos_r2": oos_val}]),
                                  Path(out_dir, f"{s}__winsor_{wlab}__rolling_metrics.parquet"), f"{s} rolling metrics")
            except Exception as e:
                print(f"[error] {s} rolling_ols winsor={wlab}: {e}")

    if verbose:
        print(f"[done] wrote artifacts to: {Path(out_dir).resolve()}")

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

    reg_rets, stock_rets = run_script()
    
    # Z score everything
    X_full           = reg_rets.apply(calc_zscore, winsor=0.01)
    y                = stock_rets.apply(calc_zscore, winsor=0.01)
    # Choose regressors
    regressor_chosen = choose_etf_representatives(X_full,20, corr_cap=0.95)
    X_reduced        = reg_rets[regressor_chosen].copy()
    r2_summary, angles, er, vifs, cis = summarise_reduction(regressor_chosen, reg_rets, winsor=(0.01, 0.09))

    # Get the chosen models for each ticker
    summary, details = panel_quick_pick_per_ticker(y, X_reduced, n_folds=3, k_max=8,
                                                   windows=(None, 60, 120, 252), store_grids=True)


    # Running finalize model to get best model parameters for each
    final_summary, final_models = finalize_models_panel(
                                    y, X_reduced, details,
                                    similarity_tol=0.005,  # treat within 0.5pp as "similar"
                                    prefer_longer_window=True,
                                    n_folds=3, test_frac=0.15,
                                    verbose=True)
    

    

