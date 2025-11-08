# pipeline_regression.py (verbose build)
from __future__ import annotations
import os, json, itertools, hashlib
import pandas as pd
from typing import Iterable, Sequence, Tuple, Dict, Any, Optional
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from regression_functions import (
    clean_and_align_returns,
    fit_returns_ols,
    orthogonalize_factors,
    rolling_ols,
    rolling_vif,
)

def _vprint(verbose: bool, *args):
    if verbose:
        print(*args)

# ------------- Utilities -------------
def _reg_key(regressors: Sequence[str]) -> str:
    return "|".join(regressors)

def _param_key(ticker: str,
               regressors: Sequence[str],
               window: Optional[int],
               intercept: bool,
               winsorize_quantiles: Tuple[float,float],
               robust: str,
               hac_lags: Optional[int]) -> str:
    d = {
        "ticker": ticker,
        "regressors": list(regressors),
        "window": None if window is None else int(window),
        "intercept": bool(intercept),
        "winsor": tuple(map(float, winsorize_quantiles)) if winsorize_quantiles else None,
        "robust": str(robust),
        "hac_lags": None if hac_lags is None else int(hac_lags),
    }
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _serialize_regressors(regressors: Sequence[str]) -> str:
    return ",".join(regressors)

def _append_parquet(df: pd.DataFrame, path: str, *, verbose: bool=False) -> None:
    """
    Simple, robust 'append': read-if-exists, concat, drop duplicates on a strong key, write back.
    Keeps everything per ticker in ONE parquet file.
    """
    if df is None or df.empty:
        _vprint(verbose, f"[append] Nothing to append -> {path}")
        return

    _vprint(verbose, f"[append] Target path: {path}")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        _vprint(verbose, f"[append] Existing rows: {len(old):,}  Incoming rows: {len(df):,}")
        key_cols = [
            "kind","ticker","model","regressors","window","intercept",
            "winsor_lo","winsor_hi","robust","hac_lags","date","factor","metric"
        ]
        missing = [c for c in old.columns if c not in df.columns]
        for c in missing: df[c] = pd.NA
        missing_old = [c for c in df.columns if c not in old.columns]
        for c in missing_old: old[c] = pd.NA
        both = pd.concat([old[df.columns], df], axis=0, ignore_index=True)
        before = len(both)
        both = both.drop_duplicates(subset=key_cols, keep="last")
        after = len(both)
        _vprint(verbose, f"[append] Deduped rows: {before:,} -> {after:,}")
        both.to_parquet(path, index=False)
        _vprint(verbose, f"[append] Saved -> {path}")
    else:
        _vprint(verbose, f"[append] New file. Writing {len(df):,} rows -> {path}")
        df.to_parquet(path, index=False)

# ------------- Flatteners for saving as one tidy table -------------
def _static_to_records(ticker: str,
                       model_name: str,
                       regressors: Sequence[str],
                       intercept: bool,
                       winsorize_quantiles: Tuple[float,float]|None,
                       robust: str,
                       hac_lags: Optional[int],
                       y_index: pd.Index,
                       coef_table: pd.DataFrame,
                       diagnostics: Dict[str, Any],
                       *,
                       verbose: bool=False) -> pd.DataFrame:
    reg_str = _serialize_regressors(regressors)
    wins_lo = winsorize_quantiles[0] if winsorize_quantiles else None
    wins_hi = winsorize_quantiles[1] if winsorize_quantiles else None
    start_dt = pd.to_datetime(y_index.min()) if len(y_index) else pd.NaT
    end_dt   = pd.to_datetime(y_index.max()) if len(y_index) else pd.NaT

    rows = []

    _vprint(verbose, f"[static_to_records:{model_name}] coef_table columns: {list(coef_table.columns)}  "
                     f"index (factors): {list(coef_table.index)}")

    # Coefficients (one row per metric per factor)
    metric_cols = ["coef","std_err","t","pval","ci_low","ci_high"]
    for fac, row in coef_table.iterrows():
        for metric in metric_cols:
            val = row[metric] if metric in row.index else pd.NA
            rows.append({
                "kind": "static",
                "ticker": ticker,
                "model": model_name,
                "regressors": reg_str,
                "window": None,
                "intercept": intercept,
                "winsor_lo": wins_lo,
                "winsor_hi": wins_hi,
                "robust": robust,
                "hac_lags": hac_lags,
                "date": pd.NaT,
                "factor": fac,
                "metric": metric,
                "value": (float(val) if (pd.notna(val)) else pd.NA),
                "sample_start": start_dt,
                "sample_end": end_dt,
            })

    # Diagnostics (scalars + static VIF if present)
    for k, v in diagnostics.items():
        if k == "vif" and isinstance(v, pd.DataFrame):
            _vprint(verbose, f"[static_to_records:{model_name}] static VIF columns: {list(v.columns)} index: {list(v.index)}")
            for fac, vif_val in v["VIF"].items():
                rows.append({
                    "kind": "vif_static",
                    "ticker": ticker,
                    "model": model_name,
                    "regressors": reg_str,
                    "window": None,
                    "intercept": intercept,
                    "winsor_lo": wins_lo,
                    "winsor_hi": wins_hi,
                    "robust": robust,
                    "hac_lags": hac_lags,
                    "date": pd.NaT,
                    "factor": fac,
                    "metric": "vif",
                    "value": float(vif_val),
                    "sample_start": start_dt,
                    "sample_end": end_dt,
                })
            continue
        if isinstance(v, (int, float)) and pd.notna(v):
            rows.append({
                "kind": "static_diag",
                "ticker": ticker,
                "model": model_name,
                "regressors": reg_str,
                "window": None,
                "intercept": intercept,
                "winsor_lo": wins_lo,
                "winsor_hi": wins_hi,
                "robust": robust,
                "hac_lags": hac_lags,
                "date": pd.NaT,
                "factor": None,
                "metric": k,
                "value": float(v),
                "sample_start": start_dt,
                "sample_end": end_dt,
            })

    out = pd.DataFrame.from_records(rows)
    _vprint(verbose, f"[static_to_records:{model_name}] rows built: {len(out):,}")
    return out

def _rolling_to_records(ticker: str,
                        regressors: Sequence[str],
                        window: int,
                        intercept: bool,
                        winsorize_quantiles: Tuple[float,float]|None,
                        robust: str,
                        hac_lags: Optional[int],
                        ols_out: Dict[str, pd.Series|pd.DataFrame],
                        *,
                        verbose: bool=False) -> pd.DataFrame:
    reg_str = _serialize_regressors(regressors)
    wins_lo = winsorize_quantiles[0] if winsorize_quantiles else None
    wins_hi = winsorize_quantiles[1] if winsorize_quantiles else None

    rows = []
    betas: pd.DataFrame = ols_out.get("betas", pd.DataFrame())
    _vprint(verbose, f"[rolling_to_records] betas shape: {betas.shape}")

    # Betas
    for dt, row in betas.iterrows():
        for fac, val in row.items():
            rows.append({
                "kind": "rolling",
                "ticker": ticker,
                "model": "rolling",
                "regressors": reg_str,
                "window": int(window),
                "intercept": intercept,
                "winsor_lo": wins_lo,
                "winsor_hi": wins_hi,
                "robust": robust,
                "hac_lags": hac_lags,
                "date": pd.to_datetime(dt),
                "factor": fac,
                "metric": "beta",
                "value": (float(val) if pd.notna(val) else pd.NA),
                "sample_start": pd.NaT,
                "sample_end": pd.NaT,
            })

    # R2 / R2_adj
    for metric_name in ["r2","r2_adj"]:
        ser = ols_out.get(metric_name, None)
        if isinstance(ser, pd.Series):
            _vprint(verbose, f"[rolling_to_records] {metric_name} len={len(ser)}  "
                             f"head={ser.head(3).to_dict() if len(ser)>0 else {}}")
            for dt, val in ser.items():
                rows.append({
                    "kind": "rolling",
                    "ticker": ticker,
                    "model": "rolling",
                    "regressors": reg_str,
                    "window": int(window),
                    "intercept": intercept,
                    "winsor_lo": wins_lo,
                    "winsor_hi": wins_hi,
                    "robust": robust,
                    "hac_lags": hac_lags,
                    "date": pd.to_datetime(dt),
                    "factor": None,
                    "metric": metric_name,
                    "value": (float(val) if pd.notna(val) else pd.NA),
                    "sample_start": pd.NaT,
                    "sample_end": pd.NaT,
                })
    out = pd.DataFrame.from_records(rows)
    _vprint(verbose, f"[rolling_to_records] rows built: {len(out):,}")
    return out

def _rolling_vif_to_records(ticker: str,
                            regressors: Sequence[str],
                            window: int,
                            vif_df: pd.DataFrame,
                            *,
                            verbose: bool=False) -> pd.DataFrame:
    reg_str = _serialize_regressors(regressors)
    rows = []
    _vprint(verbose, f"[vif_to_records] rolling VIF shape: {vif_df.shape}")
    for dt, row in vif_df.iterrows():
        for fac, val in row.items():
            rows.append({
                "kind": "vif_rolling",
                "ticker": ticker,
                "model": "rolling",
                "regressors": reg_str,
                "window": int(window),
                "intercept": pd.NA,
                "winsor_lo": pd.NA,
                "winsor_hi": pd.NA,
                "robust": pd.NA,
                "hac_lags": pd.NA,
                "date": pd.to_datetime(dt),
                "factor": fac,
                "metric": "vif",
                "value": (float(val) if pd.notna(val) else pd.NA),
                "sample_start": pd.NaT,
                "sample_end": pd.NaT,
            })
    out = pd.DataFrame.from_records(rows)
    _vprint(verbose, f"[vif_to_records] rows built: {len(out):,}")
    return out

# ------------- Public API -------------

def run_single_regression(
    *,
    rets: pd.DataFrame,
    reg_rets: pd.DataFrame,
    ticker: str,
    regressors: Sequence[str],
    window: int,
    intercept: bool = True,
    winsorize_quantiles: Tuple[float,float] = (0.025, 0.975),
    robust: str = "hac",
    hac_lags: Optional[int] = None,    # if None -> window
    save_dir: str = "regression_results",
    save: bool = True,
    compute_rolling_vif: bool = True,
    vif_sample_by: str = "days",
    vif_sample_freq: str | int = "5B",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the three models (normal OLS, orthogonalised OLS, rolling OLS) for one ticker+regressor set.
    Returns an 'out' dict with keys ['normal','ortho','rolling'].
    Also appends results to a SINGLE parquet per ticker at save_dir/{ticker}.parquet.
    """
    _ensure_dir(save_dir)
    parquet_path = os.path.join(save_dir, f"{ticker}.parquet")
    hac_lags = window if (hac_lags is None) else hac_lags

    _vprint(verbose, f"\n=== RUN: ticker={ticker} regs={regressors} window={window} "
                     f"intercept={intercept} winsor={winsorize_quantiles} robust={robust} hac_lags={hac_lags} ===")
    _vprint(verbose, f"[paths] save_dir={save_dir} parquet_path={parquet_path}")

    # Align PER TICKER and PER REGRESSOR SET
    X_raw = reg_rets[list(regressors)]
    y_raw = rets[ticker]
    _vprint(verbose, f"[input] X_raw shape={X_raw.shape} y_raw len={len(y_raw)} "
                     f"X_raw NA={X_raw.isna().sum().sum()} y_raw NA={y_raw.isna().sum()}")

    X_raw = reg_rets[list(regressors)]
    y_raw = rets[ticker]
    X_aligned, y_aligned = clean_and_align_returns(X_raw, y_raw)

    _vprint('Y : ',type(y_aligned),' X : ', type(X_aligned))
    _vprint(verbose, f"[align] X_aligned shape={X_aligned.shape} y_aligned len={len(y_aligned)} "
                     f"NA: X={X_aligned.isna().sum().sum()} y={y_aligned.isna().sum()}")
    if len(y_aligned):
        _vprint(verbose, f"[align] y stats: mean={y_aligned.mean()} std={y_aligned.std()}")
        _vprint(verbose, f"[align] X first rows:\n{X_aligned.head(3)}")

    # --- normal OLS ---
    res, coef_table, diagnostics, summary_text = fit_returns_ols(
        y=y_aligned, X=X_aligned,
        add_intercept=intercept,
        winsorize_quantiles=winsorize_quantiles,
        robust=robust,
        hac_lags=hac_lags,
    )
    _vprint(verbose, f"[normal] coef_table head:\n{coef_table.head()}")
    _vprint(verbose, f"[normal] diagnostics keys={list(diagnostics.keys())}")
    if "r2_adj" in diagnostics:
        _vprint(verbose, f"[normal] R2={diagnostics.get('r2')}  R2_adj={diagnostics.get('r2_adj')}  n={diagnostics.get('n_obs')}")

    normal_records = _static_to_records(
        ticker, "normal", regressors, intercept, winsorize_quantiles, robust, hac_lags,
        y_index=y_aligned.index,
        coef_table=coef_table, diagnostics=diagnostics,
        verbose=verbose
    )

    # --- orthogonalised OLS ---
    X_orth = orthogonalize_factors(X_aligned, order=list(regressors),
                                   demean=True, normalize="match_original_std")
    # sanity check: correlation structure change?
    try:
        corr_before = X_aligned.corr().round(3)
        corr_after  = X_orth.corr().round(3)
        _vprint(verbose, f"[ortho] corr before:\n{corr_before}\n[ortho] corr after:\n{corr_after}")
    except Exception as e:
        _vprint(verbose, f"[ortho] correlation check failed: {e}")

    res_o, coef_table_o, diagnostics_o, summary_text_o = fit_returns_ols(
        y=y_aligned, X=X_orth,
        add_intercept=intercept,
        winsorize_quantiles=winsorize_quantiles,
        robust=robust,
        hac_lags=hac_lags,
    )
    _vprint(verbose, f"[ortho] coef_table head:\n{coef_table_o.head()}")
    if "r2_adj" in diagnostics_o:
        _vprint(verbose, f"[ortho] R2={diagnostics_o.get('r2')}  R2_adj={diagnostics_o.get('r2_adj')}  n={diagnostics_o.get('n_obs')}")

    ortho_records = _static_to_records(
        ticker, "ortho", regressors, intercept, winsorize_quantiles, robust, hac_lags,
        y_index=y_aligned.index,
        coef_table=coef_table_o, diagnostics=diagnostics_o,
        verbose=verbose
    )

    # --- rolling OLS (uses ORTHO X) ---
    ols_out = rolling_ols(
        y=y_aligned, X=X_orth,
        window=int(window), step=1,
        add_intercept=intercept,
        winsorize_quantiles=winsorize_quantiles,
        robust=robust, hac_lags=hac_lags,
        oos_predict_next=False
    )
    # peek
    betas_ts = ols_out.get("betas", pd.DataFrame())
    r2_ts = ols_out.get("r2", pd.Series(dtype=float))
    r2a_ts = ols_out.get("r2_adj", pd.Series(dtype=float))
    _vprint(verbose, f"[rolling] betas shape={betas_ts.shape}  r2 len={len(r2_ts)}  r2_adj len={len(r2a_ts)}")
    if isinstance(r2_ts, pd.Series) and len(r2_ts):
        _vprint(verbose, f"[rolling] r2 describe:\n{r2_ts.describe().round(4)}")
    if not betas_ts.empty:
        _vprint(verbose, f"[rolling] betas head:\n{betas_ts.head()}")

    rolling_records = _rolling_to_records(
        ticker, regressors, window, intercept, winsorize_quantiles, robust, hac_lags, ols_out,
        verbose=verbose
    )

    # --- rolling VIF on RAW X ---
    vif_records = pd.DataFrame()
    if compute_rolling_vif:
        vif_df = rolling_vif(X_aligned, window=int(window), sample_by=vif_sample_by,
                             sample_freq=vif_sample_freq, add_intercept=True)
        _vprint(verbose, f"[vif] rolling_vif shape={vif_df.shape}")
        vif_records = _rolling_vif_to_records(ticker, regressors, window, vif_df, verbose=verbose)

    # Save all in ONE parquet per ticker
    if save:
        frames = [normal_records, ortho_records, rolling_records, vif_records]
        frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        counts = {name: len(df) for name, df in zip(
            ["normal_records","ortho_records","rolling_records","vif_records"],
            [normal_records, ortho_records, rolling_records, vif_records]
        )}
        _vprint(verbose, f"[save] row counts -> {counts}")
        if frames:
            to_save = pd.concat(frames, axis=0, ignore_index=True)
            _vprint(verbose, f"[save] concatenated rows: {len(to_save):,}")
            _append_parquet(to_save, parquet_path, verbose=verbose)
        else:
            _vprint(verbose, "[save] Nothing to save for this run.")

        # Post-save verification (sample)
        try:
            verify = pd.read_parquet(parquet_path)
            _vprint(verbose, f"[verify] file rows now: {len(verify):,}. "
                             f"Sample of last 5 rows:\n{verify.tail(5)}")
        except Exception as e:
            _vprint(verbose, f"[verify] Could not read back parquet: {e}")

    out = {
        "normal": {
            "result": res,
            "coef_table": coef_table,
            "diagnostics": diagnostics,
            "summary_text": summary_text,
        },
        "ortho": {
            "result": res_o,
            "coef_table": coef_table_o,
            "diagnostics": diagnostics_o,
            "summary_text": summary_text_o,
        },
        "rolling": ols_out,
    }
    return out

def run_grid_pipeline(
    *,
    rets: pd.DataFrame,
    reg_rets: pd.DataFrame,
    tickers: Sequence[str],
    regressor_sets: Sequence[Sequence[str]],
    windows: Sequence[int],
    intercept_opts: Sequence[bool] = (True,),
    winsorize_opts: Sequence[Tuple[float,float]] = ((0.025, 0.975),),
    robust: str = "hac",
    save_dir: str = "regression_results",   # <= unified default
    compute_rolling_vif: bool = True,
    vif_sample_by: str = "days",
    vif_sample_freq: str | int = "5B",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Sequential (no joblib) grid over tickers × regressor_sets × windows × intercept × winsor.
    Writes/extends a SINGLE parquet per ticker. Returns an index DataFrame of all runs.
    """
    _ensure_dir(save_dir)
    _vprint(verbose, f"[grid] save_dir={save_dir}")
    index_rows = []
    for ticker in tickers:
        for regs in regressor_sets:
            for window in windows:
                for intercept in intercept_opts:
                    for wins in winsorize_opts:
                        _vprint(verbose, f"[grid] RUN -> {ticker} | regs={regs} | window={window} | "
                                         f"intercept={intercept} | winsor={wins}")
                        out = run_single_regression(
                            rets=rets,
                            reg_rets=reg_rets,
                            ticker=ticker,
                            regressors=list(regs),
                            window=int(window),
                            intercept=bool(intercept),
                            winsorize_quantiles=wins,
                            robust=robust,
                            hac_lags=window,
                            save_dir=save_dir,
                            save=True,
                            compute_rolling_vif=compute_rolling_vif,
                            vif_sample_by=vif_sample_by,
                            vif_sample_freq=vif_sample_freq,
                            verbose=verbose,
                        )
                        index_rows.append({
                            "ticker": ticker,
                            "regressors": _serialize_regressors(regs),
                            "window": int(window),
                            "intercept": bool(intercept),
                            "winsor_lo": wins[0], "winsor_hi": wins[1],
                            "robust": robust,
                            "hac_lags": int(window),
                            "has_results": True,
                        })
    idx = pd.DataFrame(index_rows)
    _vprint(verbose, f"[grid] completed. runs={len(idx)}")
    return idx

# --------- Loader helpers (easy access) ---------

def load_ticker_table(ticker: str, base_dir: str = "regression_results", *, verbose: bool=False) -> pd.DataFrame:
    path = os.path.join(base_dir, f"{ticker}.parquet")
    _vprint(verbose, f"[load] loading {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved results for {ticker} at {path}")
    df = pd.read_parquet(path)
    _vprint(verbose, f"[load] rows={len(df):,} cols={len(df.columns)}. kinds={df['kind'].value_counts().to_dict()}")
    return df

def fetch_results(
    ticker: str,
    *,
    regressors: Sequence[str],
    window: Optional[int] = None,       # None -> static only
    intercept: Optional[bool] = None,   # if None, don't filter by intercept
    winsorize_quantiles: Optional[Tuple[float,float]] = None,
    robust: Optional[str] = None,
    base_dir: str = "regression_results",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reconstructs out['normal'], out['ortho'], out['rolling'] from the SINGLE parquet for this ticker,
    filtered to the given parameter set.
    """
    df = load_ticker_table(ticker, base_dir, verbose=verbose)
    reg_str = _serialize_regressors(regressors)

    mask = (df["regressors"] == reg_str)
    if window is None:
        mask &= df["window"].isna()
    else:
        mask &= (df["window"] == int(window))
    if intercept is not None:
        mask &= (df["intercept"] == bool(intercept))
    if winsorize_quantiles is not None:
        mask &= (df["winsor_lo"] == winsorize_quantiles[0]) & (df["winsor_hi"] == winsorize_quantiles[1])
    if robust is not None:
        mask &= (df["robust"] == robust)

    sub = df[mask].copy()
    _vprint(verbose, f"[fetch] filter -> ticker={ticker} regs={regressors} window={window} "
                     f"intercept={intercept} winsor={winsorize_quantiles} robust={robust}")
    _vprint(verbose, f"[fetch] matched rows={len(sub):,} "
                     f"(static={len(sub[sub['kind']=='static'])}, rolling={len(sub[sub['kind']=='rolling'])})")

    out: Dict[str, Any] = {"normal": {}, "ortho": {}, "rolling": {}}

    # ---- normal & ortho (static) ----
    for model_name in ["normal", "ortho"]:
        stat = sub[(sub["kind"] == "static") & (sub["model"] == model_name)]
        diag = sub[(sub["kind"] == "static_diag") & (sub["model"] == model_name)]
        vif  = sub[(sub["kind"] == "vif_static") & (sub["model"] == model_name)]

        _vprint(verbose, f"[fetch:{model_name}] rows -> coef={len(stat)} diag={len(diag)} vif={len(vif)}")

        if not stat.empty:
            coef = (
                stat.pivot_table(index="factor", columns="metric", values="value", aggfunc="last")
                .sort_index()
            )
            _vprint(verbose, f"[fetch:{model_name}] coef head:\n{coef.head()}")
            out[model_name]["coef_table"] = coef

        if not diag.empty:
            diag_tbl = (
                diag.pivot_table(index="metric", values="value", aggfunc="last")
                .rename_axis(None)
            )
            d = diag_tbl["value"].to_dict()
            _vprint(verbose, f"[fetch:{model_name}] diagnostics -> {d}")
            out[model_name]["diagnostics"] = d

        if not vif.empty:
            out[model_name]["vif"] = (
                vif.pivot_table(index="factor", values="value", aggfunc="last")
                .rename(columns={"value":"VIF"})
            )

    # ---- rolling ----
    roll = sub[sub["kind"] == "rolling"]
    _vprint(verbose, f"[fetch:rolling] rows={len(roll)}")
    if not roll.empty:
        betas = roll[roll["metric"] == "beta"]
        if not betas.empty:
            betas_tbl = (
                betas.pivot_table(index="date", columns="factor", values="value", aggfunc="last")
                .sort_index()
            )
            _vprint(verbose, f"[fetch:rolling] betas shape={betas_tbl.shape} head:\n{betas_tbl.head()}")
            out["rolling"]["betas"] = betas_tbl
        for m in ["r2","r2_adj"]:
            s = roll[roll["metric"] == m][["date","value"]].dropna()
            if not s.empty:
                ser = s.set_index("date")["value"].sort_index()
                _vprint(verbose, f"[fetch:rolling] {m} len={len(ser)} head:\n{ser.head()}")
                out["rolling"][m] = ser

    # rolling VIF
    vif_roll = sub[sub["kind"] == "vif_rolling"]
    if not vif_roll.empty:
        out["rolling"]["vif"] = (
            vif_roll.pivot_table(index="date", columns="factor", values="value", aggfunc="last")
            .sort_index()
        )
    return out
