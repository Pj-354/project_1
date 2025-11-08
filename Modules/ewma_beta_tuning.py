
# ewma_beta_tuning.py
# -----------------------------------------------------------------------------
# Grid search over EWMA beta settings (lambda, ridge, method, factor set)
# and choose the best parameters per ticker based on pseudo-OOS MSE, then
# beta roughness, NaN fraction, and factor condition number.
#
# Expects your environment to provide:
#   from your_module import ewma_beta_diagnostic
#
# Inputs:
#   - asset_ret_by_ticker: dict[str, pd.Series] of daily log returns for each ticker
#   - factor_ret_all: pd.DataFrame of daily log returns for *all* candidate factors
#   - factor_sets: list[list[str]] specifying which subsets of factors to consider
#   - lambdas: list[float] EWMA decay params (e.g., [0.90,0.94,0.96,0.98])
#   - ridge_grid: list[float] ridge strengths (e.g., [1e-8,1e-6,1e-4,1e-3])
#   - methods: iterable of {"ridge","orthogonalize"}
#   - orth_orders: optional dict mapping a tuple(factors) -> explicit orth order list; if None, use factors order
#
# Returns:
#   best_params: pd.DataFrame (index=ticker) with chosen method, lambda, ridge, factor_set, metrics
#   full_grid:   pd.DataFrame with one row per (ticker × combo) for auditing
#
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from stock_data_functions import TickerComparison, TickerData, calc_log_rets
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd
import itertools
import numpy as np
import pandas as pd


@dataclass
class RankingWeights:
    # Primary objective is always pseudo-OOS MSE (min). These weights only matter if you decide to
    # produce a combined score; by default we use lexicographic ranking.
    mse: float = 1.0
    roughness: float = 0.0
    nan_frac: float = 0.0
    cond_num: float = 0.0


# ---------- helpers ----------

def _align(asset: pd.Series, factors: pd.DataFrame, min_obs: int = 60) -> Tuple[pd.Series, pd.DataFrame]:
    """Inner-join asset & factor returns on dates and drop rows with any NaNs."""
    df = pd.concat([asset.rename("y"), factors], axis=1).dropna()
    if df.shape[0] < min_obs or df.shape[1] < 2:
        return pd.Series(dtype=float), pd.DataFrame()
    return df["y"], df.drop(columns=["y"])

def _lower_dict(d: dict) -> dict:
    return {str(k).lower(): v for k, v in d.items()}

def _get_metric(row_like, *, names: Sequence[str], substr: Sequence[str] = (), square_if_rmse: bool = False) -> float:
    """
    Robustly fetch a metric from a row/dict:
      - tries exact (case-insensitive) names in order,
      - then tries any key containing ALL substrings in 'substr'.
      - if 'square_if_rmse' and the chosen key name suggests RMSE, it squares it.
    """
    if hasattr(row_like, "to_dict"):
        d = row_like.to_dict()
    elif isinstance(row_like, dict):
        d = row_like
    else:
        try:
            d = dict(row_like)
        except Exception:
            return np.nan

    dl = _lower_dict(d)
    chosen_key = None
    for nm in names:
        if nm.lower() in dl:
            chosen_key = nm.lower()
            break
    if chosen_key is None and substr:
        for k in dl.keys():
            if all(s in k for s in substr):
                chosen_key = k
                break
    if chosen_key is None:
        return np.nan

    try:
        val = float(dl[chosen_key])
    except Exception:
        return np.nan

    if square_if_rmse and ("rmse" in chosen_key or chosen_key.endswith("_rms")):
        val = val * val
    return val

# ---------- main tuner ----------
def ewma_beta_diagnostic(
    asset_ret: pd.Series,
    factor_ret: pd.Series | pd.DataFrame,
    lam: float = 0.94,
    ridge: float = 1e-6,
    orth_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute both methods and summarize: pseudo-OOS MSE (using β_{t-1}), β roughness, NaN fraction, cond#."""
    from factor_analysis_functions import ewma_beta
    betas_ridge = ewma_beta(asset_ret, factor_ret, lam=lam, method="ridge", ridge=ridge)
    betas_ortho = ewma_beta(asset_ret, factor_ret, lam=lam, method="orthogonalize", ridge=0.0, orth_order=orth_order)

    def _align(asset_ret, factor_ret):
        if isinstance(factor_ret, pd.Series):
            F = factor_ret.to_frame(name=factor_ret.name or "factor")
        else:
            F = factor_ret
        df = pd.concat([asset_ret.rename("x"), F], axis=1, join="inner").dropna(how="any")
        return df["x"], df.drop(columns=["x"])

    x, F = _align(asset_ret, factor_ret)

    def _pseudo_oos_mse(betas, x: pd.Series, F: pd.DataFrame) -> float:
        idx = x.index.intersection(betas.index).intersection(F.index)
        if isinstance(betas, pd.Series):
            B = betas.reindex(idx).to_frame("b")
            Fsub = F.iloc[:, :1].rename(columns={F.columns[0]:"f"}).reindex(idx)
            yhat = (B.shift(1)["b"] * Fsub["f"])
        else:
            B = betas.reindex(idx)
            Fsub = F.reindex(idx)[B.columns]
            yhat = (B.shift(1).values * Fsub.values).sum(axis=1)
            yhat = pd.Series(yhat, index=idx)
        err = x.reindex(idx) - yhat
        return float(np.nanmean(err.to_numpy()**2))

    def _beta_roughness(betas) -> float:
        if isinstance(betas, pd.Series):
            d = betas.diff().dropna().to_numpy()
            return float(np.nanmean(d**2))
        d = betas.diff().dropna().to_numpy()
        return float(np.nanmean(np.sum(d**2, axis=1)))

    def _median_cond(F: pd.DataFrame) -> float:
        if F.empty: return float("nan")
        S = F.to_numpy()
        cov = (S.T @ S) / max(1, S.shape[0])
        vals = np.linalg.eigvalsh(cov + 1e-12*np.eye(cov.shape[0]))
        return float(np.max(vals) / np.min(vals)) if np.min(vals) > 0 else float("inf")

    summary = pd.DataFrame({
        "ridge": {
            "pseudo_OOS_MSE": _pseudo_oos_mse(betas_ridge, x, F),
            "beta_roughness": _beta_roughness(betas_ridge),
            "nan_frac": float(pd.isna(betas_ridge).mean().mean() if isinstance(betas_ridge, pd.DataFrame) else pd.isna(betas_ridge).mean()),
            "factor_cond_num": _median_cond(F),
        },
        "orthogonalize": {
            "pseudo_OOS_MSE": _pseudo_oos_mse(betas_ortho, x, F if isinstance(betas_ortho, pd.Series) else F[betas_ortho.columns]),
            "beta_roughness": _beta_roughness(betas_ortho),
            "nan_frac": float(pd.isna(betas_ortho).mean().mean() if isinstance(betas_ortho, pd.DataFrame) else pd.isna(betas_ortho).mean()),
            "factor_cond_num": _median_cond(F),
        }
    }).T

    return {"betas": {"ridge": betas_ridge, "orthogonalize": betas_ortho}, "summary": summary}

def tune_betas_for_tickers(
    asset_ret_by_ticker: Dict[str, pd.Series],
    factor_ret_all: pd.DataFrame,
    factor_sets: List[List[str]],
    lambdas: Sequence[float] = (0.90, 0.92, 0.94, 0.96, 0.98),
    ridge_grid: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-3),
    methods: Sequence[str] = ("ridge", "orthogonalize"),
    orth_orders: Optional[Dict[Tuple[str, ...], List[str]]] = None,
    min_obs: int = 60,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid search per ticker using ewma_beta_diagnostic(...).

    Selection rule (per ticker):
      1) Minimize pseudo_oos_mse
      2) Tie-break: minimize beta_roughness
      3) Tie-break: minimize nan_frac
      4) Tie-break: minimize factor_cond_num

    Note: We *do not* favor coverage in ranking. We only report pre-alignment missingness and coverage for audit.
    """
    results = []

    for tkr, asset in asset_ret_by_ticker.items():
        if verbose:
            print(f"[tune] {tkr} ...")

        for fset in factor_sets:
            fset = [f for f in fset if f in factor_ret_all.columns]
            if not fset:
                continue

            fac = factor_ret_all[fset]

            # ---- audit missingness over a common calendar (NOT used in selection) ----
            cal = asset.index.union(fac.index)
            asset_on_cal = asset.reindex(cal)
            fac_on_cal   = fac.reindex(cal)

            pre_nan_frac_asset        = float(asset_on_cal.isna().mean())                    # scalar
            pre_nan_frac_factors_mean = float(fac_on_cal.isna().mean().mean())              # avg across factors
            pre_nan_frac_factors_max  = float(fac_on_cal.isna().mean().max())               # worst factor

            # ---- align for estimation (inner-join + dropna) ----
            y, X = _align(asset, fac, min_obs=min_obs)
            n_obs = int(len(y))
            post_align_coverage = (n_obs / len(cal)) if len(cal) else 0.0

            if y.empty or X.empty:
                if verbose:
                    print(f"  - skip {tkr} {tuple(fset)}: insufficient overlap")
                continue

            for lam in lambdas:
                for rg in ridge_grid:
                    for method in methods:
                        # orthogonalization order if required
                        ortho_order = None
                        if method == "orthogonalize":
                            key = tuple(fset)
                            if orth_orders and key in orth_orders:
                                ortho_order = orth_orders[key]
                            else:
                                ortho_order = list(fset)

                        try:
                            diag = ewma_beta_diagnostic(y, X, lam=lam, ridge=rg, orth_order=ortho_order)
                            # Expect either a dict with 'summary' or a single-row structure
                            summary = diag.get("summary", None) if isinstance(diag, dict) else None
                            if summary is None:
                                # Coerce to DataFrame; assume it's just one method
                                summary = pd.DataFrame([diag], index=[method])

                            # If both methods are present as rows, index by 'method' if needed
                            if "method" in summary.columns and method not in summary.index:
                                summary = summary.set_index("method")

                            # Pick the row
                            if method in summary.index:
                                row_like = summary.loc[method]
                            elif summary.shape[0] == 1:
                                row_like = summary.iloc[0]  # best effort
                            else:
                                # Can't identify the intended row; skip
                                continue

                            # ---- robust metric extraction ----
                            pseudo_oos_mse = _get_metric(
                                row_like,
                                names=["pseudo_oos_mse", "pseudo_OOS_MSE", "oos_mse", "oos_rmse", "oos_mse_pseudo"],
                                substr=("oos","mse"),
                                square_if_rmse=True,  # if we only have RMSE, compare on MSE by squaring
                            )
                            beta_roughness = _get_metric(
                                row_like,
                                names=["beta_roughness", "roughness", "beta_tv", "total_variation"],
                                substr=("rough",),
                            )
                            nan_frac = _get_metric(
                                row_like,
                                names=["nan_frac", "nan_fraction", "nan_rate", "nanrate"],
                                substr=("nan",),
                            )
                            factor_cond_num = _get_metric(
                                row_like,
                                names=["factor_cond_num", "condition_number", "cond_num", "kappa"],
                                substr=("cond",),
                            )

                            results.append({
                                "ticker": tkr,
                                "factors": tuple(fset),
                                "method": method,
                                "lam": float(lam),
                                "ridge": float(rg),
                                "pseudo_oos_mse": pseudo_oos_mse,
                                "beta_roughness": beta_roughness,
                                "nan_frac": nan_frac,
                                "factor_cond_num": factor_cond_num,
                                # ---- audit-only fields (not used in sorting) ----
                                "pre_nan_frac_asset": pre_nan_frac_asset,
                                "pre_nan_frac_factors_mean": pre_nan_frac_factors_mean,
                                "pre_nan_frac_factors_max": pre_nan_frac_factors_max,
                                "post_align_coverage": post_align_coverage,
                                "n_obs": n_obs,
                            })

                        except Exception as ex:
                            # Penalize failures so they're never selected; still keep audit fields
                            results.append({
                                "ticker": tkr,
                                "factors": tuple(fset),
                                "method": method,
                                "lam": float(lam),
                                "ridge": float(rg),
                                "pseudo_oos_mse": np.inf,
                                "beta_roughness": np.inf,
                                "nan_frac": 1.0,
                                "factor_cond_num": np.inf,
                                "pre_nan_frac_asset": pre_nan_frac_asset,
                                "pre_nan_frac_factors_mean": pre_nan_frac_factors_mean,
                                "pre_nan_frac_factors_max": pre_nan_frac_factors_max,
                                "post_align_coverage": post_align_coverage,
                                "n_obs": n_obs,
                            })
                            if verbose:
                                print(f"  ! combo failed for {tkr}, {tuple(fset)}, {method}, lam={lam}, ridge={rg}: {ex}")

    if not results:
        raise ValueError("No valid combinations evaluated. Check inputs and factor sets.")

    full_grid = pd.DataFrame(results)

    # Selection: unchanged (no explicit favoring of coverage or pre-missingness)
    sort_cols = ["pseudo_oos_mse", "beta_roughness", "nan_frac", "factor_cond_num"]
    best_rows = (
        full_grid
        .sort_values(sort_cols, ascending=[True, True, True, True])
        .groupby("ticker", as_index=False)
        .first()
    )

    ordered = ["ticker", "method", "lam", "ridge", "factors"] + sort_cols + [
        "n_obs", "post_align_coverage", "pre_nan_frac_asset", "pre_nan_frac_factors_mean", "pre_nan_frac_factors_max"
    ]
    best_params = best_rows[ordered].set_index("ticker")

    return best_params, full_grid

def make_factor_sets(all_factors: Sequence[str], sizes: Sequence[int] = (2,)) -> List[List[str]]:
    """Helper to enumerate factor subsets (avoid large sizes to prevent explosion)."""
    sets: List[List[str]] = []
    for k in sizes:
        for comb in itertools.combinations(all_factors, k):
            sets.append(list(comb))
    return sets

############### picking the best parameters ########################

def _pick_best_method_with_band(df: pd.DataFrame,
                               methods=("ridge", "orthogonalize"),
                               rel_impr_threshold=0.1) -> pd.DataFrame:
    """
    df: rows = metrics (e.g., 'pseudo_oos_mse','beta_roughness','factor_cond_num'),
        cols = methods (e.g., 'ridge','orthogonalize'). Lower is better for all.
    rel_impr_threshold: winner must improve by >= this fraction vs the worse method.
                       Example: 0.50 means the better value must be at least 50% lower.
    Returns: DataFrame with per-metric winner/tie and the relative improvement.
    """
    out_rows = []
    for metric in df.index:
        row = df.loc[metric]
        vals = {m: float(row.get(m, np.nan)) for m in methods}

        # Handle missing/inf robustly
        fin = {m: v for m, v in vals.items() if np.isfinite(v)}
        if len(fin) == 0:
            winner, rel_impr = np.nan, np.nan
        elif len(fin) == 1:
            winner = next(iter(fin.keys()))
            rel_impr = np.inf  # only one valid value, it wins by default
        else:
            # Lower is better
            best_m = min(fin, key=fin.get)
            worst_m = max(fin, key=fin.get)
            best_v, worst_v = fin[best_m], fin[worst_m]
            # Relative improvement: (worst - best) / worst
            rel_impr = (worst_v - best_v) / worst_v if worst_v != 0 else np.inf
            winner = best_m if rel_impr >= rel_impr_threshold else np.nan

        out_rows.append({
            "metric": metric,
            methods[0]: vals.get(methods[0], np.nan),
            methods[1]: vals.get(methods[1], np.nan),
            "relative_improvement": rel_impr,
            "winner_or_tie": winner,
        })

    return pd.DataFrame(out_rows).set_index("metric")

def pick_best_factor_set_and_ridge(
    table: pd.DataFrame,
    *,
    mse_row: str = "pseudo_oos_mse",
    kappa_row: str = "factor_cond_num",
    band: float = 0.25  # 25% similarity band (relative)
):
    """
    table: rows include mse_row, kappa_row; columns are a 2-level MultiIndex:
           level 0 = factors (tuple-like, e.g. ('SPY','SMH')), level 1 = ridge (float)
    band:  relative similarity tolerance. A column c is 'in-band' if
           MSE[c] <= min_MSE / (1 - band).  (e.g., band=0.25 => within ~33.3% of the best)

    Returns:
      winner: dict with {factors, ridge, mse, kappa}
      equivalents: list of (factors, ridge) columns considered tied with the winner
    """
    if not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels != 2:
        raise ValueError("Expect MultiIndex columns (level0=factors, level1=ridge).")

    # 1) Pull metrics and keep finite MSEs
    if mse_row not in table.index:
        raise ValueError(f"Row '{mse_row}' not found.")
    mse = table.loc[mse_row].astype(float)
    valid = np.isfinite(mse.values)
    if not valid.any():
        raise ValueError("No finite pseudo_oos_mse values.")
    mse = mse[valid]

    # 2) Build the similarity band around the best MSE
    min_mse = float(mse.min())
    cutoff = min_mse / (1.0 - band)  # e.g., band=0.25 -> cutoff=1.333*min
    in_band = mse[mse <= cutoff]

    # 3) If only one in-band, done
    if len(in_band) == 1:
        col = in_band.idxmin()  # returns (factors, ridge) for MultiIndex columns
        winner = {
            "factors": col[0],
            "ridge": col[1],
            "mse": float(table.loc[mse_row, col]),
            "kappa": float(table.loc[kappa_row, col]) if kappa_row in table.index else np.nan,
        }
        return winner, [col]

    # 4) Tie-break by factor condition number (lower is better)
    contenders = in_band.index
    if kappa_row in table.index:
        kappa = table.loc[kappa_row, contenders].astype(float)
        kmin = float(kappa.min())
        contenders = kappa.index[kappa <= kmin + 0]  # exact min; keep all exact ties

    # 5) If still tied, prefer fewer factors; then smaller ridge
    if len(contenders) > 1:
        # number of factors in the tuple (('SPY','SMH'),) -> 2 ; ('SPY',) -> 1
        nf = pd.Series({c: (len(c[0]) if isinstance(c[0], (tuple, list)) else 1) for c in contenders})
        contenders = nf.index[nf == nf.min()]

    if len(contenders) > 1:
        # prefer smallest ridge among remaining ties
        ridges = pd.Series({c: float(c[1]) for c in contenders})
        contenders = ridges.index[ridges == ridges.min()]

    # pick final winner (deterministic)
    col = sorted(contenders, key=lambda c: (len(c[0]) if isinstance(c[0], (tuple, list)) else 1, c[1]))[0]
    winner = {
        "factors": col[0],
        "ridge": col[1],
        "mse": float(table.loc[mse_row, col]),
        "kappa": float(table.loc[kappa_row, col]) if kappa_row in table.index else np.nan,
    }
    return winner, list(in_band.index)

def _choose_best_factor_set(table : pd.DataFrame, factor_cond_thresh = 30, factors = None, table_return = False):

    if factors == None:
        factors == table.columns.to_list()

    res_filtered = table.loc['factor_cond_num'] < factor_cond_thresh
    if res_filtered.sum() == 0:
        print('Factor Condition All Too Large')
        print('Doubling')
        res_filtered = table.loc['factor_cond_num'] < factor_cond_thresh * 2

    res_filtered = table.loc[:, res_filtered]
    res_filtered = res_filtered.drop(labels=['factor_cond_num'], axis=0)

    best_factors = list(set(res_filtered.idxmin(axis=1).values))
    if table_return == False:
        return best_factors
    if table_return:
        return res_filtered, best_factors

def _choose_best_ridge_factors(table : pd.DataFrame, thresh):
    """ 
    Given multi column table (level 0 : lambda, level 1 : factor sets, level 2 : ridge)
    Choose the best factor and best ridge
    """
    df = table.unstack().unstack()

    best_mse = df['pseudo_oos_mse'].min()
    mse_thresh = best_mse / (1 - thresh)

    df = df[df['pseudo_oos_mse'] <= mse_thresh]
    best = df.sort_values(by='beta_roughness').head(1)
    best_factor, best_ridge = best.index[0][1:]

    return best_factor, best_ridge

def best_beta_method_per_stock(grid : pd.DataFrame, stocks : List[str], thresh = 0.05, factor_cond_thresh = 30):
    results = {}
    for stock in stocks:
        # Choose between ridge or orthogonalize based on lowest pseudo_oos_mse
        table = grid[grid['ticker'] == stock].groupby(('method'))[['pseudo_oos_mse', 'beta_roughness', 'factor_cond_num']].mean().copy().T
        best_method = _pick_best_method_with_band(table, rel_impr_threshold=thresh)['winner_or_tie'].mode().values[0]
        
        # Choose best lambda
        table = (grid[
                    (grid['ticker'] == stock)
                  & (grid['method'] == best_method)
                  ].groupby('lam')[
                      ['pseudo_oos_mse', 'beta_roughness']]
                    .mean().T
        )

        best_lam = _pick_best_method_with_band(
                        table, methods=table.columns.tolist(), rel_impr_threshold=thresh
                    )['winner_or_tie'].dropna().mode().values[0]

        # Choose best factor_sets
        table = grid[(grid['ticker'] == stock)
                   & (grid['method'] == best_method)
                   & (grid['lam']    == best_lam)
                    ].groupby(['factors'])[
                        ['pseudo_oos_mse', 'beta_roughness', 'factor_cond_num']
                    ].mean().copy().T
        
        best_factors = _choose_best_factor_set(table, factor_cond_thresh = factor_cond_thresh)
        
        # Choose from set of best factors and best ridge combination
        table = grid[(grid['ticker'] == stock)
                    & (grid['method'] == best_method)
                    & (grid['factors'].isin(best_factors))
                    & (grid['lam'] == best_lam)
                        ].groupby(['lam', 'factors', 'ridge'])[
                        ['pseudo_oos_mse', 'beta_roughness', 'factor_cond_num']
                        ].mean().copy().T
        
        best_factor, best_ridge = _choose_best_ridge_factors(table, thresh)

        results[stock] = {'best_method': best_method,
                          'best_factor': best_factor,
                          'best_lam'   : best_lam,
                          'best_ridge' : best_ridge}
    return results

def get_optimal_betas(tickers, regressors,
                      factor_sets,
                      ridge_grid,
                      lambdas,
                      thresh ,
                      date_updated,
                      date_updated_regressor,
                      daily_start_date,
                      filing_date_gte = '2023-09-11',
                      pass_tickercomparison_obj = True,
                      stock_obj = None,
                      regressor_obj = None):
    
    # Build factor and stock returns
    prep = prepare_ewma_inputs_from_tickers(
        stock=tickers,                   
        regressors=regressors,             
        filing_date_gte=filing_date_gte,
        date_updated=date_updated,
        date_updated_regressor=date_updated_regressor,
        daily_start_date=daily_start_date,
        pass_tickercomparison_obj=pass_tickercomparison_obj,
        stock_obj=stock_obj,
        regressor_obj=regressor_obj
    )
    # Get all the param combinations    
    best, grid = tune_betas_for_tickers(
        asset_ret_by_ticker=prep['stock_returns_daily'],
        factor_ret_all=prep['factor_returns_daily'],
        factor_sets=factor_sets,
        lambdas=lambdas,
        ridge_grid=ridge_grid,
        methods=("ridge","orthogonalize"),
        min_obs=60,
        verbose=True,
    )
    # Find the optimal combinations
    best_params = best_beta_method_per_stock(grid, tickers, thresh)

    optimal_betas = {}
    for i in best_params.keys():
        beta_df = ewma_betas_from_tickers(
            stock = [i],
            regressors = best_params[i]['best_factor'],
            filing_date_gte=daily_start_date,
            date_updated=date_updated,
            date_updated_regressor=date_updated_regressor,
            daily_start_date=daily_start_date,
            lam=best_params[i]['best_lam'],
            method=best_params[i]['best_method'],
            ridge=best_params[i]['best_ridge'],
            orth_order=best_params[i]['best_factor'],
        )

        optimal_betas[i] = beta_df['betas_by_stock'][i]
    
    return optimal_betas, best_params

def _extract_daily_close_rets(tc: TickerComparison) -> pd.DataFrame:
    """
    From a TickerComparison(period='day'), extract ('ticker','close') prices -> log returns.
    Returns a DataFrame [date x tickers] of daily log returns.
    """
    prices = (
        tc.tickers_stocks_prices
          .loc[:, (slice(None), "close")]
          .droplevel(1, axis=1)
          .copy()
    )
    rets = prices.apply(calc_log_rets)
    # Daily already; drop leading NaNs from first diff/log
    return rets.dropna(how="all")

def prepare_ewma_inputs_from_tickers(
    *,
    stock: List[str],
    regressors: List[str],
    filing_date_gte: str,
    # fetch/update knobs (match your style in run_full_pipeline)
    date_updated: bool = False,
    date_updated_regressor: bool = True,
    daily_start_date: Optional[str] = None,
    daily_end_date: Optional[str] = None,
    pass_tickercomparison_obj : bool = True,
    stock_obj : TickerComparison = None,
    regressor_obj : TickerComparison = None,
) -> Dict[str, Any]:
    """
    Build daily returns for stocks & factors using your TickerComparison objects, aligned and ready for EWMA beta.
    Returns:
      {
        'stock_object_daily': TickerComparison,
        'regressor_object_daily': TickerComparison,
        'stock_returns_daily': DataFrame[date x stock],
        'factor_returns_daily': DataFrame[date x regressors],
        'aligned_index': DatetimeIndex (inner-joined)
      }
    """
    # 1) Construct daily TickerComparison objects (same as run_full_pipeline)
    if pass_tickercomparison_obj:
        stock_daily = stock_obj
        reg_daily   = regressor_obj

    else:
        stock_daily = TickerComparison(
            stock,
            filing_date_gte=filing_date_gte,
            period="day",
            date_updated=date_updated,
            fetch_in_chunks=False,
            waiting_time=0,
            start_date=daily_start_date,
            end_date=daily_end_date,
        )
        reg_daily = TickerComparison(
            regressors,
            filing_date_gte=filing_date_gte,
            period="day",
            date_updated=date_updated_regressor,
            fetch_in_chunks=False,
            waiting_time=0,
            start_date=daily_start_date,
            end_date=daily_end_date,
        )

    # 2) Daily log returns (close→close), aligned
    stock_rets_daily  = _extract_daily_close_rets(stock_daily)
    factor_rets_daily = _extract_daily_close_rets(reg_daily)

    # 3) Align on intersection of dates (inner join)
    aligned_idx = stock_rets_daily.index.intersection(factor_rets_daily.index)
    stock_rets_daily  = stock_rets_daily.reindex(aligned_idx)
    factor_rets_daily = factor_rets_daily.reindex(aligned_idx)

    return {
        "stock_object_daily": stock_daily,
        "regressor_object_daily": reg_daily,
        "stock_returns_daily": stock_rets_daily,
        "factor_returns_daily": factor_rets_daily,
        "aligned_index": aligned_idx,
    }

def compute_ewma_betas_from_inputs(
    *,
    stock_returns_daily: pd.DataFrame,    # [date x stock]
    factor_returns_daily: pd.DataFrame,   # [date x factors]
    lam: float = 0.94,
    method: str = "ridge",                # "ridge" | "orthogonalize"
    ridge: float = 1e-6,                  # small stabilizer (0.0 to match pure OLS)
    orth_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    For each stock column, run your ewma_beta against the factor set.
    Returns:
      {
        'betas_by_stock': Dict[stock -> DataFrame(date x factors)],
        'betas_panel': DataFrame(date x MultiIndex(stock,factor))
      }
    """
    # local import to use your updated ewma_beta (supports multi-factor & method switch)
    from factor_analysis_functions import ewma_beta

    # Ensure factors are in a stable column order (esp. for orthogonalize)
    factor_cols = list(factor_returns_daily.columns)
    F = factor_returns_daily[factor_cols]

    betas_by_stock: Dict[str, pd.DataFrame] = {}
    wide_blocks = []

    # Iterate stocks and compute betas against the full factor set
    for s in stock_returns_daily.columns:
        x = stock_returns_daily[s].dropna()
        # Align x with F
        df_xy = pd.concat([x.rename("x"), F], axis=1, join="inner").dropna(how="any")
        if df_xy.empty:
            betas_by_stock[s] = pd.DataFrame(index=stock_returns_daily.index, columns=factor_cols, dtype=float)
            continue

        b = ewma_beta(
            asset_ret=df_xy["x"],
            factor_ret=df_xy[factor_cols],
            lam=lam,
            method=method,
            ridge=ridge,
            orth_order=orth_order
        )
        # Guarantee DataFrame shape (Series if single factor)
        if isinstance(b, pd.Series):
            b = b.to_frame(factor_cols[0])
        betas_by_stock[s] = b.reindex(stock_returns_daily.index)

        # build panel block with MultiIndex (stock,factor)
        blk = b.copy()
        blk.columns = pd.MultiIndex.from_product([[s], blk.columns])
        wide_blocks.append(blk)

    betas_panel = pd.concat(wide_blocks, axis=1) if wide_blocks else pd.DataFrame(index=stock_returns_daily.index)
    return {"betas_by_stock": betas_by_stock, "betas_panel": betas_panel}

def ewma_betas_from_tickers(
    *,
    stock: List[str],
    regressors: List[str],
    filing_date_gte: str,
    date_updated: bool = False,
    date_updated_regressor: bool = True,
    daily_start_date: Optional[str] = None,
    daily_end_date: Optional[str] = None,
    lam: float = 0.94,
    method: str = "ridge",                # "ridge" | "orthogonalize"
    ridge: float = 1e-6,
    orth_order: Optional[List[str]] = None,
    pass_tickercomparison_obj = False,
    stock_obj = None,
    regressor_obj = None
) -> Dict[str, Any]:
    """
    One-call convenience:
      - builds daily TickerComparison for stocks & regressors,
      - computes daily log returns,
      - runs EWMA betas (multi-factor) per stock, with your chosen method,
      - returns both per-stock betas and a panel (MultiIndex columns).
    """
    prep = prepare_ewma_inputs_from_tickers(
        stock=stock,
        regressors=regressors,
        filing_date_gte=filing_date_gte,
        date_updated=date_updated,
        date_updated_regressor=date_updated_regressor,
        daily_start_date=daily_start_date,
        daily_end_date=daily_end_date,
        pass_tickercomparison_obj=pass_tickercomparison_obj,
        stock_obj=stock_obj,
        regressor_obj=regressor_obj
    )
    out = compute_ewma_betas_from_inputs(
        stock_returns_daily=prep["stock_returns_daily"],
        factor_returns_daily=prep["factor_returns_daily"],
        lam=lam,
        method=method,
        ridge=ridge,
        orth_order=orth_order
    )
    # bubble up the useful bits
    out.update({
        "stock_object_daily": prep["stock_object_daily"],
        "regressor_object_daily": prep["regressor_object_daily"],
        "stock_returns_daily": prep["stock_returns_daily"],
        "factor_returns_daily": prep["factor_returns_daily"],
    })
    return out
