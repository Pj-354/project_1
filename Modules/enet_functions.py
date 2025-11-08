import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Optional, List

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd

def cluster_pick_representatives(X, thr=0.80, pick="lowest_mean_corr", keep_first=None):
    """
    Hierarchical clustering on distance = 1 - |corr|.
    Pick one column per cluster.
    pick options: "lowest_mean_corr" (default) or "highest_vol"
    """
    keep_first = set(keep_first or [])
    X = X.copy()

    # Correlation (abs), robust to NaNs
    C = X.corr().abs()
    np.fill_diagonal(C.values, 1.0)
    C = C.fillna(0.0)  # treat unknown corr as 0 so they won't cluster tightly

    # Distances and clustering
    D = 1 - C
    Z = linkage(squareform(D.values, checks=False), method="average")
    clusters = fcluster(Z, t=1 - thr, criterion="distance")

    # Group columns by cluster id
    groups = {}
    for col, g in zip(C.columns, clusters):
        groups.setdefault(g, []).append(col)

    chosen = []
    for cols in groups.values():
        # 1) If cluster has only one member, take it
        if len(cols) == 1:
            winner = cols[0]

        # 2) Honor keep_first if present in this cluster
        elif keep_first & set(cols):
            # choose the first keep_first ticker encountered
            winner = next(c for c in cols if c in keep_first)

        else:
            if pick == "highest_vol":
                # highest standard deviation in this cluster
                winner = X[cols].std(skipna=True).idxmax()
            else:
                # lowest mean |corr| to ALL others (more robust than intra-cluster)
                mean_global = {}
                for c in cols:
                    s = C.loc[c].drop(c)              # corr to everyone else
                    val = s.replace([np.inf, -np.inf], np.nan).mean()
                    mean_global[c] = val
                ser = pd.Series(mean_global).dropna()
                if ser.empty:
                    # fallback: if all NaN for some reason, pick the first (sorted for determinism)
                    winner = sorted(cols)[0]
                else:
                    winner = ser.idxmin()

        chosen.append(winner)

    # Final safety: drop any accidental NaNs (shouldn't happen now)
    chosen = [c for c in chosen if isinstance(c, str) and c in X.columns]
    return X[chosen].copy(), chosen
# -----------------------------
# Purged + Embargo splitter (for inner CV)
# -----------------------------
class PurgedEmbargoSplit:
    """
    Time-series CV with purge and embargo.
    Splits indices 0..n-1 into K folds in order; each fold uses:
      - train: [0 : test_start - embargo) with the last `purge` obs removed
      - test:  [test_start : test_end)
    """
    def __init__(self, n_splits: int = 3, purge: int = 5, embargo: int = 2, test_size: int = 20):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.test_size = test_size

    def split(self, X, y=None, groups=None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_ends = np.linspace(self.test_size, n, self.n_splits, dtype=int)
        for end in fold_ends:
            test_end = end
            test_start = max(test_end - self.test_size, 0)
            train_end = max(test_start - self.embargo, 0)
            train_idx = np.arange(0, max(train_end - self.purge, 0))
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) > 20 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

# -----------------------------
# Helper: recover original-scale coefficients from a fitted pipeline
# (X was scaled by StandardScaler, y left as-is)
# -----------------------------
def recover_original_coefs(pipe: Pipeline, X_fit: pd.DataFrame) -> Tuple[pd.Series, float]:
    scaler: StandardScaler = pipe.named_steps['scale']
    model: ElasticNet = pipe.named_steps['model']

    std = pd.Series(scaler.scale_, index=X_fit.columns)
    mu  = pd.Series(scaler.mean_,  index=X_fit.columns)
    coef_scaled = pd.Series(model.coef_, index=X_fit.columns)

    beta_orig = coef_scaled / std
    intercept_orig = float(model.intercept_) - float((mu / std * coef_scaled).sum())
    return beta_orig, intercept_orig

# -----------------------------
# Rolling Elastic Net backtester
# -----------------------------
@dataclass
class ENetRollingParams:
    train_window: int = 126
    test_horizon: int = 1
    step: int = 1
    purge: int = 5
    embargo: int = 2
    inner_cv_splits: int = 3
    inner_cv_test_size: int = 20
    alphas: Optional[np.ndarray] = None
    l1_ratios: Optional[List[float]] = None
    n_jobs: int = -1
    max_iter: int = 50000
    tol: float = 1e-4
    intercept: bool = True        # <<< new flag

def rolling_enet_pipeline(
    y: pd.Series,
    X: pd.DataFrame,
    params: ENetRollingParams = ENetRollingParams(),
) -> Dict[str, object]:

    # --- align & sanitize ---
    y = y.dropna()
    X = X.dropna(how="all", axis=1).loc[y.index].dropna()
    y = y.loc[X.index]
    idx = X.index
    n = len(idx)

    alphas = params.alphas if params.alphas is not None else np.logspace(-4, 0, 9)
    l1s    = params.l1_ratios if params.l1_ratios is not None else [0.1, 0.3, 0.5, 0.7, 0.9]

    betas = pd.DataFrame(np.nan, index=idx, columns=X.columns)
    intercept = pd.Series(np.nan, index=idx, name='intercept')
    y_hat = pd.Series(np.nan, index=idx, name='y_hat')
    chosen_hp = []

    # --- helpers ---
    def recover_betas_no_intercept(pipe: Pipeline, X_fit: pd.DataFrame) -> pd.Series:
        scaler: StandardScaler = pipe.named_steps['scale']
        coef   = pd.Series(pipe.named_steps['model'].coef_, index=X_fit.columns)
        std    = pd.Series(scaler.scale_, index=X_fit.columns)
        return coef / std

    def recover_original_coefs(pipe: Pipeline, X_fit: pd.DataFrame) -> Tuple[pd.Series, float]:
        scaler: StandardScaler = pipe.named_steps['scale']
        model: ElasticNet = pipe.named_steps['model']
        std = pd.Series(scaler.scale_, index=X_fit.columns)
        mu  = pd.Series(scaler.mean_,  index=X_fit.columns)
        coef_scaled = pd.Series(model.coef_, index=X_fit.columns)
        beta_orig = coef_scaled / std
        intercept_orig = float(model.intercept_) - float((mu / std * coef_scaled).sum())
        return beta_orig, intercept_orig

    def build_purged_cv_splits(n_train, n_splits, test_size, purge, embargo):
        splits = []
        max_blocks = max(0, (n_train - purge - embargo) // test_size)
        use_blocks = min(n_splits, max_blocks)
        for b in range(use_blocks, 0, -1):
            val_end   = n_train - (b-1)*test_size
            val_start = val_end - test_size
            tr_end    = max(val_start - embargo, 0)
            tr = np.arange(0, max(tr_end - purge, 0))
            va = np.arange(val_start, val_end)
            if len(tr) > 10 and len(va) > 0:
                splits.append((tr, va))
        return splits

    t = params.train_window
    while (t + params.embargo + params.test_horizon) <= n:
        tr_start = t - params.train_window
        tr_end   = t
        te_start = t + params.embargo
        te_end   = t + params.embargo + params.test_horizon

        tr_idx = np.arange(tr_start, tr_end - params.purge)
        te_idx = np.arange(te_start, te_end)

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te       = X.iloc[te_idx]

        # CV splits for this train block
        cv_splits = build_purged_cv_splits(
            n_train=len(X_tr),
            n_splits=params.inner_cv_splits,
            test_size=params.inner_cv_test_size,
            purge=params.purge,
            embargo=params.embargo,
        )
        if len(cv_splits) == 0:
            val = min(params.inner_cv_test_size, max(5, len(X_tr)//5))
            tr = np.arange(0, max(len(X_tr) - val - params.purge, 1))
            va = np.arange(len(X_tr) - val, len(X_tr))
            cv_splits = [(tr, va)]

        # Branch on intercept
        if params.intercept:
            # WITH intercept: no centering of y; model estimates intercept.
            pipe = Pipeline([
                ('scale', StandardScaler(with_mean=True, with_std=True)),
                ('model', ElasticNet(fit_intercept=True, max_iter=params.max_iter, tol=params.tol, selection='random'))
            ])
            grid = {'model__alpha': alphas, 'model__l1_ratio': l1s}
            gs = GridSearchCV(pipe, grid, scoring='r2', cv=cv_splits, n_jobs=params.n_jobs, refit=True)
            gs.fit(X_tr, y_tr)

            best_pipe: Pipeline = gs.best_estimator_
            pred = best_pipe.predict(X_te)
            y_hat.iloc[te_idx] = pred

            t_store = te_idx[-1]
            b, c = recover_original_coefs(best_pipe, X_tr)
            betas.iloc[t_store] = b.values
            intercept.iloc[t_store] = c

        else:
            # NO intercept: center y on TRAIN; add mean back to predictions.
            y_tr_mean = y_tr.mean()
            y_tr_c    = y_tr - y_tr_mean

            pipe = Pipeline([
                ('scale', StandardScaler(with_mean=True, with_std=True)),
                ('model', ElasticNet(fit_intercept=False, max_iter=params.max_iter, tol=params.tol, selection='random'))
            ])
            grid = {'model__alpha': alphas, 'model__l1_ratio': l1s}
            gs = GridSearchCV(pipe, grid, scoring='r2', cv=cv_splits, n_jobs=params.n_jobs, refit=True)
            gs.fit(X_tr, y_tr_c)

            best_pipe: Pipeline = gs.best_estimator_
            pred = best_pipe.predict(X_te) + y_tr_mean
            y_hat.iloc[te_idx] = pred

            t_store = te_idx[-1]
            b = recover_betas_no_intercept(best_pipe, X_tr)
            betas.iloc[t_store] = b.values
            intercept.iloc[t_store] = 0.0  # by design for no-intercept spec

        chosen_hp.append((idx[t_store], gs.best_params_['model__alpha'], gs.best_params_['model__l1_ratio']))
        t += params.step

    betas = betas.ffill()
    intercept = intercept.ffill().fillna(0.0)

    resid = y - y_hat
    common = resid.dropna().index
    oos_r2  = r2_score(y.loc[common], y_hat.loc[common]) if len(common) else np.nan
    oos_rmse = root_mean_squared_error(y.loc[common], y_hat.loc[common]) if len(common) else np.nan

    chosen_hp = pd.DataFrame(chosen_hp, columns=['date','alpha','l1_ratio']).set_index('date')

    return {
        'betas': betas,
        'intercept': intercept,
        'y_hat': y_hat,
        'resid': resid,
        'oos_scores': {'R2': oos_r2, 'RMSE': oos_rmse},
        'chosen_hp': chosen_hp
    }