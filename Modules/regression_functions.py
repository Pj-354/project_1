from .stock_data_functions import TickerData, TickerComparison, calc_log_rets
import numpy as np
import pandas as pd
from statsmodels.stats import diagnostic as smd
from statsmodels.stats import stattools as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.tseries.frequencies import to_offset
import datetime as dt
import importlib
import warnings
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot

from statsmodels.stats import diagnostic as smd
from statsmodels.stats import stattools as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.tseries.frequencies import to_offset


###################### Helper functions ########################
def clean_and_align_returns(X : pd.DataFrame, y: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and align returns DataFrame X and Series y.
    Returns aligned X and y with NaNs dropped.
    """
    y = y.copy()
    X = X.copy()
    df = pd.concat({"y": y, "X": X}, axis=1).dropna()
    return df["X"], df["y"]

def orthogonalize_factors(
    X: pd.DataFrame,
    *,
    order: list[str] | None = None,
    method: str = "sequential",          # {"sequential","qr"}
    demean: bool = True,                 # subtract mean of each factor first
    normalize: str | None = None,        # {None,"unit_norm","unit_var","match_original_std"}
    verify: bool = False,                # return a correlation check if True
):
    """
    Orthogonalize columns (factors) of X.

    Parameters
    ----------
    X : pd.DataFrame
        Aligned factor return series (rows=time, cols=factors).
    order : list[str] or None
        Processing order of factors. Defaults to X.columns order.
        (Order matters for 'sequential' method.)
    method : {"sequential","qr"}
        - "sequential": residualize each factor on previously orthogonalized ones
                        (modified Gram–Schmidt). Interpretability-friendly.
        - "qr":         numpy QR decomposition; very stable, yields orthonormal basis
                        (before optional rescaling).
    demean : bool
        If True, subtract column means before orthogonalizing (recommended).
    normalize : None or {"unit_norm","unit_var","match_original_std"}
        Optional rescaling of the orthogonalized columns:
        - None: no rescale (keep residual scale).
        - "unit_norm": L2 norm == 1 for each column.
        - "unit_var":  sample variance == 1 for each column.
        - "match_original_std": match each column's std to its original std.
    verify : bool
        If True, returns (X_ortho, corr_offdiag_max) where corr_offdiag_max is the
        maximum absolute off-diagonal element of the correlation matrix of X_ortho.

    Returns
    -------
    X_ortho : pd.DataFrame
        Orthogonalized factors (same index and columns).
    (optional) corr_offdiag_max : float
        Only if verify=True.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    cols = list(X.columns) if order is None else list(order)
    X_use = X[cols].astype(float).copy()

    if demean:
        X_use = X_use - X_use.mean()

    if method not in {"sequential", "qr"}:
        raise ValueError("method must be one of {'sequential','qr'}")

    if method == "sequential":
        # Modified Gram–Schmidt via projections on already-orthogonalized columns
        Xo = pd.DataFrame(index=X_use.index, columns=cols, dtype=float)
        Q_prev = []  # store numpy arrays for speed

        for c in cols:
            v = X_use[c].to_numpy(copy=True)
            for q in Q_prev:
                denom = np.dot(q, q)
                if denom > 0.0:
                    v = v - (np.dot(q, v) / denom) * q

            # handle near-linear dependence
            if np.allclose(v, 0.0):
                warnings.warn(f"Factor '{c}' is (near) linear combo of previous; column set to zeros.")
                Xo[c] = 0.0
                Q_prev.append(np.zeros_like(v))
            else:
                Xo[c] = v
                Q_prev.append(v)

    else:  # method == "qr"
        # Economic QR gives Q (n x k) with orthonormal columns; R is k x k upper-triangular
        Q, R = np.linalg.qr(X_use.to_numpy(), mode="reduced")
        # map back to DataFrame
        Xo = pd.DataFrame(Q, index=X_use.index, columns=cols)

    # Optional normalization / rescaling
    if normalize is not None:
        if normalize not in {"unit_norm", "unit_var", "match_original_std"}:
            raise ValueError("normalize must be one of {None,'unit_norm','unit_var','match_original_std'}")

        if normalize == "unit_norm":
            for c in cols:
                nrm = np.linalg.norm(Xo[c].to_numpy())
                if nrm > 0:
                    Xo[c] /= nrm

        elif normalize == "unit_var":
            for c in cols:
                sd = Xo[c].std(ddof=1)
                if sd > 0:
                    Xo[c] /= sd

        elif normalize == "match_original_std":
            orig_sd = X_use.std(ddof=1)
            for c in cols:
                sd_new = Xo[c].std(ddof=1)
                if sd_new > 0 and orig_sd[c] > 0:
                    Xo[c] *= (orig_sd[c] / sd_new)

    if verify:
        corr = Xo.corr()
        # zero-out diagonal and take max |off-diagonal|
        np.fill_diagonal(corr.values, 0.0)
        return Xo, float(np.abs(corr.values).max())

    return Xo

def _auto_newey_west_lags(n: int) -> int:
    """
    Automatic HAC lag length (Newey-West) using the common
    rule-of-thumb: floor(4 * (n/100)^(2/9)).
    """
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))

def _winsorize_block(y: pd.Series, X: pd.DataFrame, q: tuple[float, float]):
    lo, hi = map(float, q)
    ylo, yhi = y.quantile([lo, hi]).to_numpy()
    y2 = y.clip(ylo, yhi)
    Xlo = X.quantile(lo)
    Xhi = X.quantile(hi)
    X2 = X.clip(lower=Xlo, upper=Xhi, axis=1)
    return y2, X2

def _need_min_obs(p: int, add_intercept: bool) -> int:
    """A conservative minimum nobs per window."""
    k = p + (1 if add_intercept else 0)
    return max(10, k + 2)

def _winsorize_block(y: pd.Series, X: pd.DataFrame, q):
    lo, hi = map(float, q)
    ylo, yhi = y.quantile([lo, hi]).to_numpy()
    y2 = y.clip(ylo, yhi)
    Xlo = X.quantile(lo)
    Xhi = X.quantile(hi)
    X2 = X.clip(lower=Xlo, upper=Xhi, axis=1)
    return y2, X2

def _coerce_y(y) -> pd.Series:
    """Ensure y is a float Series named 'y'."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        elif "y" in y.columns:
            y = y["y"]
        else:
            raise ValueError("y is a DataFrame with multiple columns; pass a Series or a 1-col DataFrame.")
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return y.astype(float).rename("y")

def compute_vif_matrix(X: pd.DataFrame, add_intercept: bool = True) -> pd.DataFrame:
    Xp = X.astype(float).copy()
    if add_intercept:
        Xp = sm.add_constant(Xp, has_constant="add")
    cols = list(Xp.columns)
    vifs = []
    for i, c in enumerate(cols):
        if c == "const":
            vifs.append(np.nan)
        else:
            vifs.append(variance_inflation_factor(Xp.values, i))
    out = pd.DataFrame({"VIF": vifs}, index=cols)
    return out.drop(index="const", errors="ignore")

def rolling_vif(
    X: pd.DataFrame,
    *,
    window: int,
    sample_by: str = "rows",       # {"rows","days"}
    sample_freq: int | str = 5,    # rows: int; days: e.g. 5, "5B", "7D"
    add_intercept: bool = True,
    missing: str = "drop"
) -> pd.DataFrame:
    if sample_by not in {"rows", "days"}:
        raise ValueError("sample_by must be 'rows' or 'days'.")

    n = len(X)
    p = X.shape[1]
    min_obs = max(10, p + (1 if add_intercept else 0) + 2)

    # ----- Build list of endpoint indices to evaluate -----
    if sample_by == "rows":
        if not isinstance(sample_freq, int) or sample_freq <= 0:
            raise ValueError("For sample_by='rows', sample_freq must be a positive int.")
        endpoints = list(range(window - 1, n, sample_freq))

    else:
        # sample_by == "days"
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("For sample_by='days', X.index must be a DatetimeIndex.")

        # Normalize offset input
        if isinstance(sample_freq, str):
            off = to_offset(sample_freq)  # e.g. "5B", "7D", "2W"
        elif isinstance(sample_freq, int) and sample_freq > 0:
            off = to_offset(f"{int(sample_freq)}D")  # integer -> calendar days
        else:
            raise ValueError("For sample_by='days', sample_freq must be a positive int or a pandas offset alias string.")

        # --- Option A: Vectorized (fast & clean) ---
        # Generate a schedule of target timestamps, then map each to the first index >= target
        start_ts = X.index[window - 1]
        end_ts   = X.index[-1]
        cuts = pd.date_range(start=start_ts, end=end_ts, freq=off)

        # searchsorted gives insertion points; keep those inside range
        pos = X.index.searchsorted(cuts, side="left")
        endpoints = [i for i in pos if window - 1 <= i < n]

        # --- Option B: Iterative (also correct; uncomment if preferred) ---
        # endpoints = []
        # next_cut = X.index[window - 1]
        # for i in range(window - 1, n):
        #     t = X.index[i]
        #     if t >= next_cut:
#                 endpoints.append(i)
        #         next_cut = next_cut + off

    # ----- Compute VIFs on sampled windows -----
    out_rows, out_idx = [], []

    for end in endpoints:
        start = end - window + 1
        block = X.iloc[start:end+1]

        if missing == "drop":
            block = block.dropna()
        elif missing == "raise" and block.isna().any().any():
            raise ValueError(f"NaNs in window ending at {X.index[end]}")

        if len(block) < min_obs:
            continue

        vifs = compute_vif_matrix(block, add_intercept=add_intercept)["VIF"]
        out_rows.append(vifs)
        out_idx.append(X.index[end])

    if not out_rows:
        raise ValueError("No VIFs computed — check window length, sampling frequency, and missing handling.")

    return pd.DataFrame(out_rows, index=pd.Index(out_idx, name="window_end")).sort_index()

################### VIF #####################
def compute_vif_matrix(X: pd.DataFrame, add_intercept: bool = True) -> pd.DataFrame:
    """Classic unweighted VIFs for a single design matrix X (no y needed)."""
    Xp = X.astype(float).copy()
    if add_intercept:
        Xp = sm.add_constant(Xp, has_constant="add")
    cols = list(Xp.columns)
    vifs = []
    for i, c in enumerate(cols):
        if c == "const":
            vifs.append(np.nan)  # we don't report a VIF for the constant
        else:
            vifs.append(variance_inflation_factor(Xp.values, i))
    out = pd.DataFrame({"VIF": vifs}, index=cols)
    return out.drop(index="const", errors="ignore")

def rolling_vif(
    X: pd.DataFrame,
    *,
    window: int,
    sample_by: str = "rows",       # {"rows","days"}
    sample_freq: int | str = 5,    # rows: int; days: e.g. 5, "5B", "7D"
    add_intercept: bool = True,
    missing: str = "drop"
) -> pd.DataFrame:
    """Compute VIFs on rolling windows, sampled by rows or calendar frequency."""
    if sample_by not in {"rows", "days"}:
        raise ValueError("sample_by must be 'rows' or 'days'.")

    n = len(X)
    p = X.shape[1]
    min_obs = max(10, p + (1 if add_intercept else 0) + 2)

    # --- determine endpoints to evaluate ---
    if sample_by == "rows":
        if not isinstance(sample_freq, int) or sample_freq <= 0:
            raise ValueError("For sample_by='rows', sample_freq must be a positive int.")
        endpoints = list(range(window - 1, n, sample_freq))
    else:
        # calendar sampling
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("For sample_by='days', X.index must be a DatetimeIndex.")
        off = (to_offset(sample_freq) if isinstance(sample_freq, str)
               else to_offset(f"{int(sample_freq)}D"))
        # vectorized schedule → map to nearest index ≥ cut
        start_ts = X.index[window - 1]
        end_ts   = X.index[-1]
        cuts = pd.date_range(start=start_ts, end=end_ts, freq=off)  # supports "5B", "7D", etc.
        pos = X.index.searchsorted(cuts, side="left")
        endpoints = [i for i in pos if window - 1 <= i < n]

    # --- compute VIFs on sampled windows ---
    out_rows, out_idx = [], []
    for end in endpoints:
        start = end - window + 1
        block = X.iloc[start:end+1]
        if missing == "drop":
            block = block.dropna()
        elif missing == "raise" and block.isna().any().any():
            raise ValueError(f"NaNs in window ending at {X.index[end]}")
        if len(block) < min_obs:
            continue
        vifs = compute_vif_matrix(block, add_intercept=add_intercept)["VIF"]
        out_rows.append(vifs); out_idx.append(X.index[end])

    if not out_rows:
        raise ValueError("No VIFs computed — check window length, sampling, and missing handling.")
    return pd.DataFrame(out_rows, index=pd.Index(out_idx, name="window_end")).sort_index()

################### Regression #####################
def fit_returns_ols(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    add_intercept: bool = True,
    winsorize_quantiles: tuple | None = None,  # e.g. (0.01, 0.99)
    robust: str = "auto",                      # {"auto","hc3","hac","none"}
    hac_lags: int | None = None,
    lb_lags: int = 10,                         # Ljung-Box on residuals
    arch_lags: int = 5,                        # ARCH LM test lags
    missing: str = "drop"                      # {"drop","raise","none"}
):
    

    y_clean = y.copy()
    X_clean = X.copy()
    # 2) Optional winsorization to tame outliers (symmetric clipping)
    if winsorize_quantiles is not None:
        lo, hi = winsorize_quantiles
        lo = float(lo); hi = float(hi)
        y_q = y_clean.quantile([lo, hi]).to_numpy()
        y_clean = y_clean.clip(y_q[0], y_q[1])
        X_q_lo = X_clean.quantile(lo)
        X_q_hi = X_clean.quantile(hi)
        X_clean = X_clean.clip(lower=X_q_lo, upper=X_q_hi, axis=1)

    # 3) Add intercept if requested
    if add_intercept:
        X_exog = sm.add_constant(X_clean, has_constant="add")
    else:
        X_exog = X_clean

    # 4) Fit classic OLS
    ols = sm.OLS(y_clean.values, X_exog.values, missing=missing)
    base_res = ols.fit()
    # 5) Choose robust covariance, if any
    n = int(base_res.nobs)
    cov_choice = robust.lower()
    if cov_choice == "auto":
        cov_choice = "hac" if n >= 80 else "hc3"

    if cov_choice == "hc3":
        res = base_res.get_robustcov_results(cov_type="HC3")
        cov_used = "HC3 (heteroskedasticity-robust)"
    elif cov_choice == "hac":
        lags = _auto_newey_west_lags(n) if hac_lags is None else int(hac_lags)
        res = base_res.get_robustcov_results(
            cov_type="HAC",
            maxlags=lags,
            use_correction=True
        )
        cov_used = f"HAC (Newey–West), lags={lags}"
    elif cov_choice == "none":
        res = base_res
        cov_used = "Classic OLS"
    else:
        raise ValueError("robust must be one of {'auto','hc3','hac','none'}")

    # 6) Coefficient table (with robust SEs)
    params = pd.Series(res.params, index=X_exog.columns if hasattr(X_exog, "columns") else range(len(res.params)))
    bse    = pd.Series(res.bse,    index=params.index)
    tvals  = pd.Series(res.tvalues, index=params.index)
    pvals  = pd.Series(res.pvalues, index=params.index)
    ci     = pd.DataFrame(res.conf_int(), columns=["ci_low", "ci_high"])
    ci.index = params.index

    coef_table = pd.concat(
        [params.rename("coef"), bse.rename("std_err"), tvals.rename("t"), pvals.rename("pval"), ci],
        axis=1
    )

    # 7) Diagnostics
    resid = pd.Series(res.resid, index=X_clean.index)
    dw = sms.durbin_watson(resid)

    # Jarque–Bera normality
    jb_stat, jb_p, skew, kurt = sms.jarque_bera(resid)

    # Breusch–Pagan and White tests (need exog incl constant if present)
    # Use original (non-NW-adjusted) residuals per usual practice
    bp_stat, bp_p, _, _ = smd.het_breuschpagan(base_res.resid, X_exog)
    white_stat, white_p, _, _ = smd.het_white(base_res.resid, X_exog)

    # Ljung–Box for residual autocorr
    lb = smd.acorr_ljungbox(resid, lags=[lb_lags], return_df=True).iloc[0]
    lb_stat, lb_p = float(lb["lb_stat"]), float(lb["lb_pvalue"])

    # ARCH LM test for conditional heteroskedasticity
    arch_stat, arch_p, _, _ = smd.het_arch(resid, nlags=arch_lags)

    # Condition number for multicollinearity
    # (use exog with constant if present)
    try:
        # Using singular values of X'X
        _, svals, _ = np.linalg.svd(np.asarray(X_exog), full_matrices=False)
        cond_no = float(svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
    except Exception:
        cond_no = np.nan

    # VIFs (exclude intercept if present)
    vif_df = None
    try:
        if add_intercept and "const" in (X_exog.columns if hasattr(X_exog, "columns") else []):
            X_vif = X_exog.drop(columns=["const"])
        else:
            X_vif = X_exog.copy()

        if hasattr(X_vif, "values") and X_vif.shape[1] > 0:
            vifs = []
            cols = list(X_vif.columns)
            for i in range(len(cols)):
                vifs.append(variance_inflation_factor(X_vif.values, i))
            vif_df = pd.DataFrame({"feature": cols, "VIF": vifs}).set_index("feature")
    except Exception:
        pass  # keep vif_df=None if computation fails

    diagnostics = {
        "nobs": n,
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "f_statistic": float(getattr(res, "fvalue", np.nan)),
        "f_pvalue": float(getattr(res, "f_pvalue", np.nan)),
        "stderr_type": cov_used,
        "durbin_watson": float(dw),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_pvalue": float(bp_p),
        "white_stat": float(white_stat),
        "white_pvalue": float(white_p),
        "ljung_box_stat": float(lb_stat),
        "ljung_box_pvalue": float(lb_p),
        "arch_lm_stat": float(arch_stat),
        "arch_lm_pvalue": float(arch_p),
        "condition_number": float(cond_no),
        "vif": vif_df  # DataFrame or None
    }

    # 8) Text summary (already uses robust cov if applied)
    # Use summary2 for a compact, readable output
    try:
        summary_text = res.summary2(title=f"OLS Returns Regression [{cov_used}]").as_text()
    except Exception:
        summary_text = str(res.summary())

    return res, coef_table, diagnostics, summary_text


    k = p + (1 if add_intercept else 0)
    return max(10, k + 2)

def rolling_ols(
    y: pd.Series | pd.DataFrame,
    X: pd.DataFrame,
    *,
    window: int,
    step: int = 1,
    add_intercept: bool = True,
    winsorize_quantiles: tuple[float, float] | None = None,
    robust: str = "auto",            # {"auto","hc3","hac","none"}
    hac_lags: int | None = None,
    lb_lags: int = 10,
    arch_lags: int = 5,
    missing: str = "drop",           # {"drop","raise","none"}
    oos_predict_next: bool = False
):
    y = _coerce_y(y)
    X = X.astype(float).copy()
    df = pd.concat([y.to_frame(), X], axis=1, join="inner")
    cols = X.columns.tolist()

    beta_list, se_list, t_list, p_list = [], [], [], []
    r2_list, r2a_list, aic_list, bic_list = [], [], [], []
    f_list, fp_list, stderr_list, nobs_list = [], [], [], []
    dw_list, jb_p_list, lb_p_list, arch_p_list = [], [], [], []
    idx_list, yhat_next = [], []

    min_obs = _need_min_obs(p=len(cols), add_intercept=add_intercept)

    for end in range(window - 1, len(df), step):
        start = end - window + 1
        block = df.iloc[start:end+1]
        if missing == "drop":
            block = block.dropna()
        elif missing == "raise" and block.isna().any().any():
            raise ValueError(f"NaNs present in window ending {df.index[end]}")
        if len(block) < min_obs:
            continue

        yw = block["y"].astype(float).copy()
        Xw = block[cols].astype(float).copy()
        if winsorize_quantiles is not None:
            yw, Xw = _winsorize_block(yw, Xw, winsorize_quantiles)

        X_exog = sm.add_constant(Xw, has_constant="add") if add_intercept else Xw
        base_res = sm.OLS(yw.values, X_exog.values, missing=missing).fit()

        n = int(base_res.nobs)
        cov_choice = robust.lower()
        if cov_choice == "auto":
            cov_choice = "hac" if n >= 80 else "hc3"

        if cov_choice == "hc3":
            res = base_res.get_robustcov_results(cov_type="HC3")
            cov_used = "HC3"
        elif cov_choice == "hac":
            lags = _auto_newey_west_lags(n) if hac_lags is None else int(hac_lags)
            lags = max(1, min(lags, max(1, n // 4)))
            res = base_res.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
            cov_used = f"HAC({lags})"
        elif cov_choice == "none":
            res = base_res
            cov_used = "OLS"
        else:
            raise ValueError("robust must be one of {'auto','hc3','hac','none'}")

        exog_cols = (["const"] if add_intercept else []) + cols
        params = pd.Series(res.params, index=exog_cols)
        bse    = pd.Series(res.bse,    index=exog_cols)
        tvals  = pd.Series(res.tvalues, index=exog_cols)
        pvals  = pd.Series(res.pvalues, index=exog_cols)

        resid = pd.Series(res.resid, index=block.index)
        dw = sms.durbin_watson(resid)
        _, jb_p, _, _ = sms.jarque_bera(resid)
        lb = smd.acorr_ljungbox(resid, lags=[lb_lags], return_df=True).iloc[0]
        lb_p = float(lb["lb_pvalue"])
        _, arch_p, _, _ = smd.het_arch(resid, nlags=arch_lags)

        beta_list.append(params); se_list.append(bse)
        t_list.append(tvals);     p_list.append(pvals)
        r2_list.append(float(res.rsquared));      r2a_list.append(float(res.rsquared_adj))
        aic_list.append(float(res.aic));          bic_list.append(float(res.bic))
        f_list.append(float(getattr(res, "fvalue", np.nan)))
        fp_list.append(float(getattr(res, "f_pvalue", np.nan)))
        stderr_list.append(cov_used);             nobs_list.append(n)
        dw_list.append(float(dw));                jb_p_list.append(float(jb_p))
        lb_p_list.append(lb_p);                   arch_p_list.append(float(arch_p))
        idx_list.append(block.index[-1])

        if oos_predict_next and end + 1 < len(df):
            next_row = df.iloc[end + 1:end + 2][cols]
            next_exog = (np.concatenate(([1.0], next_row.values.flatten()))
                         if add_intercept else next_row.values.flatten())
            yhat_next.append((df.index[end + 1], float(np.dot(params.values, next_exog))))

    if not idx_list:
        raise ValueError("No rolling windows were fit; check window size vs data length and missing handling.")

    index = pd.Index(idx_list, name="window_end")
    out = {
        "betas":   pd.DataFrame(beta_list, index=index),
        "std_err": pd.DataFrame(se_list,   index=index),
        "tvalues": pd.DataFrame(t_list,    index=index),
        "pvalues": pd.DataFrame(p_list,    index=index),
        "r2":      pd.Series(r2_list,  index=index, name="r2"),
        "r2_adj":  pd.Series(r2a_list, index=index, name="r2_adj"),
        "aic":     pd.Series(aic_list, index=index, name="aic"),
        "bic":     pd.Series(bic_list, index=index, name="bic"),
        "f_statistic": pd.Series(f_list,   index=index, name="f_statistic"),
        "f_pvalue":    pd.Series(fp_list,  index=index, name="f_pvalue"),
        "stderr_type": pd.Series(stderr_list, index=index, name="stderr_type"),
        "nobs":        pd.Series(nobs_list,   index=index, name="nobs"),
        "durbin_watson":        pd.Series(dw_list,   index=index, name="durbin_watson"),
        "jarque_bera_pvalue":   pd.Series(jb_p_list, index=index, name="jarque_bera_pvalue"),
        "ljung_box_pvalue":     pd.Series(lb_p_list, index=index, name="ljung_box_pvalue"),
        "arch_lm_pvalue":       pd.Series(arch_p_list,index=index, name="arch_lm_pvalue"),
    }
    if oos_predict_next and yhat_next:
        out["yhat_next"] = pd.Series(dict(yhat_next)).sort_index()
    return out

def _r2(y_true: pd.Series, y_pred: pd.Series, centered: bool) -> float:
    mask = (~y_true.isna()) & (~y_pred.isna())
    y_t = y_true[mask]; y_p = y_pred[mask]
    ssr = float(((y_t - y_p)**2).sum())
    if centered:
        tss = float(((y_t - y_t.mean())**2).sum())
    else:
        tss = float((y_t**2).sum())
    return np.nan if tss <= 0 else 1.0 - ssr/tss

def r2_from_betas_or_fit_v2(
    y: pd.Series | pd.DataFrame,
    X: pd.DataFrame,
    *,
    betas: pd.Series | None = None,       # static betas (Series with optional "const")
    add_intercept: bool = True,           # used only when fitting (betas=None)
    ols_start: str | pd.Timestamp | None = None,
    ols_end:   str | pd.Timestamp | None = None,
    winsorize_quantiles: tuple[float,float] | None = None,
    strict_cols: bool = True              # enforce exact match of factors vs betas index (ex-const)
):
    """
    If betas is provided: compute in-window R^2 identical to statsmodels (same rows, same centering).
    If betas is None: fit an OLS on [ols_start:ols_end] and return R^2 from that fit (and betas).
    """
    y = _coerce_y(y)
    X = X.astype(float).copy()
    df = pd.concat([y.to_frame(), X], axis=1, join="inner")

    def _winsorize_df(y: pd.Series, X: pd.DataFrame, q):
        lo, hi = map(float, q)
        ylo, yhi = y.quantile([lo,hi]).to_numpy()
        y2 = y.clip(ylo, yhi)
        Xlo, Xhi = X.quantile(lo), X.quantile(hi)
        X2 = X.clip(lower=Xlo, upper=Xhi, axis=1)
        return y2, X2

    # --- CASE A: use provided betas (what you're doing) ---
    if isinstance(betas, pd.Series):
        params = betas.astype(float).copy()
        const = float(params.pop("const"))  # remove const if present
        # remove any const-like column in X to avoid double-counting
        X_use = X.copy()
        const_cols = [c for c in X_use.columns if c.lower() in {"const","intercept"}]
        if const_cols:
            X_use = X_use.drop(columns=const_cols)

        # same window as the fit
        block = df.loc[ols_start:ols_end].dropna(how="any")   # matches OLS row-dropping
        y_fit = block["y"]
        X_fit = X_use.reindex(block.index)

        # optional winsorization to mirror fit preprocessing
        if winsorize_quantiles is not None:
            y_fit, X_fit = _winsorize_df(y_fit, X_fit, winsorize_quantiles)

        # column check & alignment
        if strict_cols:
            miss_in_X  = [c for c in params.index if c not in X_fit.columns]
            extra_in_X = [c for c in X_fit.columns if c not in params.index]
            if miss_in_X or extra_in_X:
                raise ValueError(f"Column mismatch. Missing in X: {miss_in_X}; Extra in X: {extra_in_X}")
        beta_vec = params.reindex(X_fit.columns).fillna(0.0)

        yhat = const + X_fit.dot(beta_vec)
        # statsmodels rule: centered R^2 iff model has an intercept
        centered = True if const != 0.0 else False
        r2 = _r2(y_fit, yhat, centered=centered)
        return {"r2_eval": r2, "params_used": pd.concat([pd.Series({"const": const}), beta_vec])}

    # --- CASE B: fit OLS on the window (for exact parity with statsmodels) ---
    if ols_start is None or ols_end is None:
        raise ValueError("When betas=None, provide ols_start and ols_end.")

    block = df.loc[ols_start:ols_end].dropna(how="any")
    if block.empty:
        raise ValueError("No data in the OLS window after dropping NaNs.")
    y_fit = block["y"]; X_fit = X.reindex(block.index)
    if winsorize_quantiles is not None:
        y_fit, X_fit = _winsorize_df(y_fit, X_fit, winsorize_quantiles)

    X_ex = sm.add_constant(X_fit, has_constant="add") if add_intercept else X_fit
    res = sm.OLS(y_fit.values, X_ex.values).fit()

    exog_cols = (["const"] if add_intercept else []) + list(X_fit.columns)
    params = pd.Series(res.params, index=exog_cols)

    # replicate statsmodels in-sample R^2 exactly
    if add_intercept:
        yhat = params["const"] + X_fit.dot(params.drop("const"))
        centered = True
    else:
        yhat = X_fit.dot(params)
        centered = False
    r2 = _r2(y_fit, yhat, centered=centered)
    return {"r2_in_sample": r2, "params_used": params}

################### Regime Variable Detection #####################
def transform_r2_logit(r2: pd.Series | pd.DataFrame, eps: float = 1e-8) -> pd.Series | pd.DataFrame:
    r2 = r2.astype(float).copy()
    # keep pandas structure
    return r2.clip(lower=eps, upper=1 - eps).applymap(lambda v: np.log(v / (1 - v))) if isinstance(r2, pd.DataFrame) \
           else np.log(r2.clip(lower=eps, upper=1 - eps) / (1 - r2.clip(lower=eps, upper=1 - eps)))

def lag_regimes(regimes: pd.DataFrame | pd.Series, lags: int = 1, add_const: bool = True) -> pd.DataFrame:
    Z = regimes.to_frame() if isinstance(regimes, pd.Series) else regimes.copy()
    Z = Z.astype(float).shift(int(lags))
    return sm.add_constant(Z, has_constant="add") if add_const else Z

def _coerce_single_col(y: pd.Series | pd.DataFrame, name: str = "y") -> tuple[pd.Series, str]:
    if isinstance(y, pd.Series): 
        return y.rename(name), name
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        return y.iloc[:, 0].rename(y.columns[0]), y.columns[0]
    raise ValueError("y must be a Series or a 1-column DataFrame.")

def fit_hac_ols(y: pd.Series | pd.DataFrame, X: pd.DataFrame, window: int, use_correction: bool = True):
    # y can be Series or 1-col DataFrame
    y_ser, col_name = _coerce_single_col(y)
    df = pd.concat([y_ser, X], axis=1).dropna()
    if df.empty:
        raise ValueError("No observations after aligning and dropping NaNs.")
    yv = df[col_name].values
    Xv = df.drop(columns=[col_name]).values
    base = sm.OLS(yv, Xv).fit()
    L = max(1, int(window) - 1)
    res = base.get_robustcov_results(cov_type="HAC", maxlags=L, use_correction=bool(use_correction))
    cols = df.drop(columns=[col_name]).columns
    table = pd.DataFrame({"coef": res.params, "std_err": res.bse, "t": res.tvalues, "pval": res.pvalues}, index=cols)
    return res, table

def fit_fractional_logit_hac(r2: pd.Series | pd.DataFrame, X: pd.DataFrame, window: int, eps: float = 1e-8):
    # expects raw R^2 in (0,1)
    y_ser, _ = _coerce_single_col(r2)
    y_ser = y_ser.clip(lower=eps, upper=1 - eps)  # keep pandas structure
    df = pd.concat([y_ser.rename("y"), X], axis=1).dropna()
    if df.empty:
        raise ValueError("No observations after aligning and dropping NaNs.")
    fam = sm.families.Binomial(link=sm.families.links.logit())  # logit link
    glm = sm.GLM(df["y"].values, df.drop(columns=["y"]).values, family=fam)
    L = max(1, int(window) - 1)
    res = glm.fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
    cols = df.drop(columns=["y"]).columns
    table = pd.DataFrame({"coef": res.params, "std_err": res.bse,
                          "z": getattr(res, "tvalues", res.params / res.bse), "pval": res.pvalues}, index=cols)
    return res, table
