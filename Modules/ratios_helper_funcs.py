from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import pandas as pd
import numpy as np
from IPython.display import display
from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer, BUCKET_METRIC_MAP


__all__ = [
    "check_ncf",
    "get_rawfields_finnonly",
    "_load_SEC_from_parquet",
    "identify_overlapping_alias_mapping_SEC",
    "_pivot_SEC",
    "_select_cols_from_SEC",
    "diff_sign_check",
    "check_missing_raw_poly_finn",
    "show_debt_liab_cash",
    "show_operating",
    "fetch",
    "filter_dates",
    "_nan_cols",
    "filter_financials_quarterised_zeros",
    "get_metric_components",
    "assess_years_and_poly_quarters",
    "overwrite_with_selected",

]

DEFAULT_QUARTERISED_DIR = Path(__file__).resolve().parent.parent / "data" / "quarterised_fundamentals"

EXCLUDE = [
    "bs_market_cap_diluted",
    "bs_market_cap_basic",
    "earnings_release_date",
    "accepted_date",
    "close",
    "earnings_release_session",
    "form",
    "period_key",
]

CHECK_COLS = ['is_revenues',
              'is_cost_of_revenue',
              'is_gross_profit',
              'is_operating_expenses',
              'is_operating_income_loss',
              'is_research_and_development',
              'is_selling_general_and_administrative_expenses',
              'is_depreciation_and_amortization',
              'is_net_interest_expense',
              'is_income_tax_expense_benefit',
              'is_net_income_loss',
              'bs_assets',
              'bs_equity',
              'bs_property_plant_and_equipment_gross',
              'bs_net_property_plant_and_equipment',
              'bs_current_assets',
              'bs_cash_and_cash_equivalents',
              'bs_short_term_investments',
              'bs_inventory',
              'bs_current_liabilities',
              'bs_accounts_payable',
              'bs_deferred_revenue',
              'bs_noncurrent_liabilities',
              'bs_interest_bearing_debt',
              'bs_total_liabilities',
              'cf_net_cash_flow_from_operating_activities',
              'cf_capital_expenditures',
              'cf_net_cash_flow_from_investing_activities',
              'cf_net_cash_flow_from_financing_activities',
              'cf_net_cash_flow',
              'cf_effect_of_exchange_rate_on_cash_and_cash_equivalents',
              'bs_marketable_securities',
              'is_basic_average_shares',
              'is_diluted_average_shares',
              'cf_interest_paid_net']
def fetch(
    tickers: Union[str, Sequence[str]],
    start_year: Union[int, str] = "2020",
    quarter: Union[int, str] = "1",
    base_dir: Union[str, Path, None] = None,
) -> Union[pd.DataFrame, Tuple[Dict[str, pd.DataFrame], List[str]]]:
    """
    Load quarterised fundamentals parquet(s) for one or many tickers and filter from a start quarter.

    Args:
        tickers: Ticker symbol or iterable of symbols.
        start_year: Calendar year to start from (inclusive).
        quarter: Quarter number (1-4) to start from (inclusive).
        base_dir: Override root directory containing per-ticker parquet folders.

    Returns:
        - If `tickers` is a string: a single DataFrame filtered from the requested quarter.
        - If `tickers` is a sequence: (dict of ticker -> DataFrame, list of tickers that failed).
    """
    base_path = Path(base_dir) if base_dir is not None else DEFAULT_QUARTERISED_DIR
    try:
        start_dt = f"{int(start_year):04d}Q{int(quarter)}"
    except (TypeError, ValueError) as exc:
        raise ValueError("start_year and quarter must be convertible to integers") from exc

    def _load_single(ticker: str) -> pd.DataFrame:
        symbol = ticker.strip().upper()
        parquet_path = base_path / symbol / f"{symbol}_quarterised.parquet"
        df = pd.read_parquet(parquet_path)
        df = df.sort_index()
        # Robust filtering: boolean mask avoids KeyError when start_dt is not an index label.
        mask = df.index.astype(str) >= start_dt
        return df.loc[mask]

    if isinstance(tickers, str):
        return _load_single(tickers)

    if not isinstance(tickers, Sequence):
        raise TypeError("tickers must be a string or a sequence of strings")

    results: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for ticker in tickers:
        if not isinstance(ticker, str):
            failed.append(str(ticker))
            continue
        try:
            results[ticker.upper().strip()] = _load_single(ticker)
        except FileNotFoundError:
            failed.append(ticker)
        except Exception:
            failed.append(ticker)

    return results, failed


def filter_dates(
    df: pd.DataFrame,
    start_date,
    end_date,
    date_col: str = "end_date",
) -> pd.DataFrame:
    """
    Return rows between start_date and end_date (inclusive) based on `date_col`.
    Dates are coerced with `pd.to_datetime`; rows with invalid dates are dropped.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = df[date_col].notna() & (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df.loc[mask].sort_values(by=date_col)


def _nan_cols(
    df: pd.DataFrame,
    thresh: Union[int, float],
    exclude: Optional[Sequence[str]] = None,
    meta_columns: Optional[Sequence[str]] = None,
    analyzer: Optional["TTMRatiosAnalyzer"] = None,
) -> List[str]:
    """
    Return columns whose NaN count exceeds `thresh`.

    Args:
        df: DataFrame to inspect.
        thresh: Threshold for NaN counts. If 0 < thresh < 1, treated as a fraction of rows.
        exclude: Columns to skip (defaults to EXCLUDE).
        meta_columns: Additional columns to skip (e.g., metadata).
        analyzer: Optional analyzer instance; if provided, its `_META_DROP_COLUMNS` are also skipped.
    """
    if df.empty:
        return []

    exclude_set = set(exclude or EXCLUDE)
    meta_set = set(meta_columns or [])
    if analyzer is not None:
        meta_set.update(getattr(analyzer, "_META_DROP_COLUMNS", ()))

    cols_to_check = [c for c in df.columns if c not in exclude_set and c not in meta_set]
    if not cols_to_check:
        return []

    nan_counts = df[cols_to_check].isna().sum()
    if 0 < thresh < 1:
        threshold_count = int(len(df) * thresh)
    else:
        threshold_count = int(thresh)

    return [col for col, count in nan_counts.items() if count > threshold_count]


def check_ncf(financials_quarterised: pd.DataFrame) -> None:
    """
    Quick reconciliation helper: confirms net cash flow equals the sum of operating,
    investing, and financing cash flows (after FX). Prints the discrepancy in billions.
    """
    net_cashflows = (
        -financials_quarterised["cf_effect_of_exchange_rate_on_cash_and_cash_equivalents"]
        + financials_quarterised["cf_net_cash_flow"]
    )
    sum_cashflows = financials_quarterised[
        [
            "cf_net_cash_flow_from_financing_activities",
            "cf_net_cash_flow_from_investing_activities",
            "cf_net_cash_flow_from_operating_activities",
        ]
    ].sum(axis=1)
    print((net_cashflows - sum_cashflows) / 1e9)


def get_rawfields_finnonly(
    ticker: str,
    freq: str,
    raw_api,
    num: int,
    keyword,
):
    """
    Convenience extractor for Finnhub raw API payloads.
    - If raw_api is None, fetches and returns the raw JSON data list.
    - Otherwise, returns a DataFrame of concept/value pairs for the filing at index `num`,
      optionally filtered by keyword substring(s).
    """
    from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer  # lazy import to avoid cycles

    if raw_api is None:
        raw_api = (
            TTMRatiosAnalyzer(verbose=False, interactive=False, debug=True)
            ._fetch_finnhub_raw(ticker, freq)
            .get("data")
        )
        return raw_api

    print(str(raw_api[num].get("year")) + "Q" + str(raw_api[num].get("quarter")))
    raw_api_data = (
        [i for i in raw_api[num].get("report").get("ic")]
        + [i for i in raw_api[num].get("report").get("bs")]
        + [i for i in raw_api[num].get("report").get("cf")]
    )

    if keyword == "all":
        dict_ = {i["concept"]: i["value"] for i in raw_api_data}
    else:
        dict_ = {
            i["concept"]: i["value"]
            for i in raw_api_data
            if any(kw.lower() in i["concept"].lower() for kw in keyword)
        }

    df2 = pd.DataFrame(data=dict_.values(), index=dict_.keys(), columns=["value"])

    return df2


def _load_SEC_from_parquet(
    ticker: str, analyzer: Optional["TTMRatiosAnalyzer"] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load saved SEC raw parquet, map GAAP tags to canonical, and split into flow/stock DataFrames.
    Useful for quick inspection or re-pivoting without hitting the SEC API again.
    """
    if analyzer is None:
        from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer  # lazy import

        analyzer = TTMRatiosAnalyzer(verbose=False, interactive=False, debug=False, force_update=False)

    try:
        raw_parquet = pd.read_parquet(f"/Users/phillip/Desktop/Moon2/data/SEC_fundamentals/{ticker}_SEC_raw.parquet")
    except FileNotFoundError:
        analyzer._load_sec_financials(ticker, save=True)
        raw_parquet = pd.read_parquet(f"/Users/phillip/Desktop/Moon2/data/SEC_fundamentals/{ticker}_SEC_raw.parquet")
    canon_map = {
        concept: analyzer._finnhub_field_to_canon(concept, ticker, False)
        for concept in raw_parquet["concept_raw"].unique()
    }
    base = (
        raw_parquet.rename(
            columns={
                "concept_raw": "raw_concept",
                "value": "val",
                "fiscal_year": "fy",
                "fiscal_period": "fp",
                "start": "start_date",
                "end": "end_date",
                "filed": "filing_date",
            }
        )
        .loc[lambda d: d["val"].notna()]
        .assign(canonical=lambda d: d["raw_concept"].map(canon_map))
    )

    stock_raw_df = (
        base[base["start_date"].isna()]
        .rename(columns={"time": "end_date"})
        [["raw_concept", "canonical", "unit", "val", "fy", "fp", "end_date", "filing_date", "ticker"]]
        .assign(prefix="bs_")
        .reset_index(drop=True)
    )

    flow_raw_df = (
        base[base["start_date"].notna()]
        .rename(columns={"start": "start_date", "end": "end_date"})
        [["raw_concept", "canonical", "unit", "val", "fy", "fp", "start_date", "end_date", "filing_date", "ticker"]]
        .reset_index(drop=True)
    )
    return flow_raw_df, stock_raw_df


def identify_overlapping_alias_mapping_SEC(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify SEC canonical fields that have multiple raw concepts mapped to them.
    Handy for spotting ambiguous alias mappings.
    """
    df = df.copy()
    mask = df.groupby("canonical")["raw_concept"].transform("nunique") > 1
    return df[mask][["canonical", "raw_concept"]].drop_duplicates()


def _pivot_SEC(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot SEC long-form raw data back to wide format.
    Returns start/end for flows or time for stocks along with canonical columns.
    """
    df.rename(columns={"start": "start_date",
                       "end": "end_date",
                       "filing" : "filing_date",
                       "fy" : "fiscal_year",
                       "fp": "fiscal_period"}, inplace=True)
    if "start_date" in df.columns and "end_date" in df.columns:
        return (
            df.pivot(
                index=["start_date", "end_date", "filing_date", "fiscal_year", "fiscal_period", "ticker"],
                columns="canonical",
                values="val",
            )
            .reset_index()
        )
    else:
        return (
            df.pivot(
                index=["time", "fy", "fp", "filing_date", "ticker"],
                columns="canonical",
                values="val",
            )
            .reset_index()
            .rename(columns={"time": "end_date", "fy": "fiscal_year", "fp": "fiscal_period"})
        )


def _select_cols_from_SEC(
    long_sec_df: pd.DataFrame,
    selected_cols: List[str],
    analyzer: Optional["TTMRatiosAnalyzer"] = None,
) -> pd.DataFrame:
    """
    Filter long-form SEC data for selected canonical fields and return a wide, sorted DataFrame.
    Useful to inspect a subset of metrics without reprocessing full SEC payloads.
    """
    if analyzer is None:
        from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer  # lazy import

        analyzer = TTMRatiosAnalyzer(verbose=False, interactive=False, debug=False, force_update=False)
    available_cols = [analyzer._strip_statement_prefix(c) for c in selected_cols]
    keep_cols = (
        ["canonical", "val", "fy", "fp", "start", "end", "filing_date", "ticker"]
        if "start" in long_sec_df.columns
        else ["canonical", "val", "fy", "fp", "time", "filing_date", "ticker"]
    )
    _select_df = long_sec_df[long_sec_df["canonical"].isin(available_cols)][keep_cols]
    out = _pivot_SEC(_select_df)
    sort_cols = ["end_date"] if "start_date" not in out.columns else ["start_date", "end_date"]
    return out.sort_values(by=sort_cols)


def diff_sign_check(
    financials_quarterised: pd.DataFrame,
    left_terms: dict,
    right_terms: dict,
) -> pd.Series:
    """
    Compute the signed difference between two linear combinations of fields.
    `left_terms` and `right_terms` map column names -> sign/multiplier.
    Returns a Series indexed like `financials_quarterised`, where:
        diff = sum(left_terms * values) - sum(right_terms * values)

    Example:
        diff_sign_check(fin_q, {'bs_total_assets': 1.0}, {'bs_equity': 1.0, 'bs_total_liabilities': 1.0})
        yields assets - (equity + liabilities) per period.
    """
    df = financials_quarterised

    def _linear_comb(terms: dict) -> pd.Series:
        cols = []
        for col, sign in (terms or {}).items():
            if col in df.columns:
                cols.append(df[col].astype(float) * float(sign))
            else:
                cols.append(pd.Series(0.0, index=df.index))
        return sum(cols) if cols else pd.Series(0.0, index=df.index)

    left_sum = _linear_comb(left_terms)
    right_sum = _linear_comb(right_terms)
    return left_sum - right_sum

def check_missing_raw_poly_finn(
    ticker: str,
    calendar_year_slice: Tuple[int, int] | None = None,
    metrics: List[str] | None = None,
    analyzer = None,
) -> pd.DataFrame:
    """
    Check for missing raw fields in the loaded merged dataframe (using the period key).
    Pass an existing analyzer if you want to reuse config/session; otherwise a fresh one is created.
    """
    if analyzer is None:
        from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer  # lazy import to avoid cycles

        analyzer = TTMRatiosAnalyzer(
            verbose=False,
            interactive=False,
            debug=False,
            force_update=False,
            register_unrecognized_fields=False,
        )
    _, missing, raw = analyzer._load_financials_merged(ticker=ticker, return_raw=True)
    raw = raw.sort_values(by=['start_date', 'end_date'])
    if calendar_year_slice is not None:
        start_fy, end_fy = calendar_year_slice
        start_fy = pd.to_datetime(f"{start_fy}-01-01").strftime("%Y-%m-%d")
        end_fy = pd.to_datetime(f"{end_fy}-12-31").strftime("%Y-%m-%d")
        raw = raw[(raw['end_date'] >= start_fy) & (raw['end_date'] <= end_fy)]
    if metrics is not None:
        cols = ['start_date', 'end_date', 'period_key'] + metrics
        raw = raw[cols]

    return raw, missing

def show_debt_liab_cash(quarterised_financials, n=10):
    display(quarterised_financials.filter(like='liab').tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials.filter(like='debt').tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials.filter(like='cash').tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))

def show_operating(quarterised_financials, n=10):
    display(quarterised_financials[['is_research_and_development', 'is_selling_general_and_administrative_expenses', "is_depreciation_and_amortization", 'is_operating_expense_residual']].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials[['is_operating_expenses','is_operating_expense_alt', 'is_operating_income_loss']].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials[['is_revenues', 'is_gross_profit', 'is_cost_of_revenue', "is_net_income_loss", "eps_basic","eps_diluted"]].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))

def filter_financials_quarterised_zeros(quarterised_financials):
    quarterised_financials = quarterised_financials.copy()
    STRICT_POSITIVE = [
            "bs_assets",
            "bs_current_assets",
            "bs_current_liabilities",
            "bs_total_liabilities",
            "bs_noncurrent_liabilities",
            "bs_equity",
            "bs_goodwill",
            "bs_cash_and_cash_equivalents",
            "bs_short_term_investments",
            "bs_cash_and_equivalents_short_term_investments",
            "bs_property_plant_and_equipment_gross",
            "bs_property_plant_and_equipment_net",
            "bs_deferred_revenue",
            "bs_interest_bearing_debt",
            "bs_noncontrolling_interest_equity",
    
            "bs_market_cap_basic",
            "bs_market_cap_diluted",
            "is_revenues",
            "is_gross_profit",
            "is_basic_average_shares",
            "is_diluted_average_shares",
            # CapEx columns (positive outflows)
            "cf_capital_expenditures",
            "cf_adjusted_capex",
        ]

    # Filter to only columns that exist in the dataframe
    cols = [col for col in STRICT_POSITIVE if col in quarterised_financials.columns]
    
    if not cols:
        print("No STRICT_POSITIVE columns found in dataframe.")
        return quarterised_financials
    
    # Find negative values
    negative_mask = quarterised_financials.loc[:, cols] < 0
    
    # Count negative values per column
    negative_counts = negative_mask.sum()
    
    # Print diagnostics - only columns with negative values
    print("=" * 70)
    print("Negative Values in STRICT_POSITIVE Columns:\n")
    cols_with_negatives = negative_counts[negative_counts > 0]
    if len(cols_with_negatives) > 0:
        for col, count in cols_with_negatives.sort_values(ascending=False).items():
            print(f"  {col}: {count} negative value(s)")
    else:
        print("  No negative values found in any STRICT_POSITIVE columns.")
    print("=" * 70)
    
    # Replace negative values with NaN
    quarterised_financials.loc[:, cols] = quarterised_financials.loc[:, cols].where(
        ~negative_mask, np.nan
    )
    
    return quarterised_financials

def get_metric_components(
    metric_values: Optional[List[str]] = None,
    metric_buckets_path: Optional[Path] = None,
    metric_components_path: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get constituent components for each metric based on metric buckets and components.
    
    Parameters
    ----------
    metric_values : List[str] or None
        List of metric values (e.g., ["is_depreciation_and_amortization", "is_revenues"]).
        These are the values from metric_buckets.json (1-1 mapping with bucket keys).
        If None, returns all metrics.
    metric_buckets_path : Path, optional
        Path to metric_buckets.json. Defaults to data/metric_buckets.json.
    metric_components_path : Path, optional
        Path to metric_components2.json. Defaults to data/metric_components2.json.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping metric names to their components:
        {
            "metric_name": {
                "bucket": "bucket_name",
                "components": {
                    "component_name": {"sign": 1.0},
                    ...
                }
            },
            ...
        }
    """
    import json
    
    # Set default paths
    if metric_buckets_path is None:
        metric_buckets_path = Path(__file__).resolve().parent.parent / "data" / "metric_buckets.json"
    if metric_components_path is None:
        metric_components_path = Path(__file__).resolve().parent.parent / "data" / "metric_components2.json"
    
    # Load metric buckets
    with metric_buckets_path.open() as f:
        buckets_data = json.load(f)
    
    # Load metric components
    with metric_components_path.open() as f:
        components_data = json.load(f)
    
    buckets = buckets_data.get("buckets", {})
    components_buckets = components_data.get("buckets", {})
    
    # Build reverse mapping: metric -> bucket_key
    metric_to_bucket = {}
    for bucket_key, bucket_info in buckets.items():
        metric = bucket_info.get("metric")
        if metric is not None:
            metric_to_bucket[metric] = bucket_key
    
    # If None, get all metric values
    if metric_values is None:
        metric_values = list(metric_to_bucket.keys())
    
    result = {}
    
    for metric_value in metric_values:
        # Find the bucket for this metric
        bucket_key = metric_to_bucket.get(metric_value)
        
        if bucket_key is None:
            # Metric not found in mapping, skip
            continue
        
        # Get components for this bucket
        bucket_components = components_buckets.get(bucket_key, {})
        components = bucket_components.get("components", {})
        
        result[metric_value] = {
            "bucket": bucket_key,
            "components": components
        }
    
    return result


def assess_years_and_poly_quarters(df: pd.DataFrame):
    """
    Given a quarterly financials DataFrame, return:
      1) year_quality: per-fiscal-year status (good / bad + issues)
      2) quarters_to_replace: fiscal years/quarters where poly data should be replaced

    Assumptions:
      - df has columns:
          ['effective_fiscal_year', 'fiscal_period', 'period_key'] + metrics
      - fiscal_period is like 'Q1', 'Q2', 'Q3', 'Q4'
      - provider can be inferred from period_key: contains 'finn' or 'poly'
    """

    df = df.copy()
    meta_cols = [
        'ticker','provider', 'effective_fiscal_year', 'period_type', 'fiscal_period',
        'fiscal_year', 'quarter', 'start_date', 'end_date', 'filing_date', 'form', 'timeframe',
        'accepted_date', 'access_number', 'period_key'
        ]
    metrics = [c for c in df.columns if c not in meta_cols]
    # --- infer provider from period_key --------------------------------------
    df['provider'] = df['period_key'].astype('string').str.extract(r'(finn|poly)', expand=False)

    # --- per-year quality assessment -----------------------------------------
    def _assess_year(g: pd.DataFrame) -> pd.Series:
        # quarters present
        quarters = set(g['fiscal_period'].dropna())
        has_all_quarters = {'Q1', 'Q2', 'Q3', 'Q4'}.issubset(quarters)

        # all required metrics non-null in all quarters
        all_metrics_non_null = g[metrics].notna().all(axis=1).all()

        # provider family: ignore NaN provider (e.g. derived rows)
        provs = g['provider'].dropna().unique()
        single_provider_family = len(provs) <= 1

        status = 'good' if (has_all_quarters and
                            all_metrics_non_null and
                            single_provider_family) else 'bad'

        issues = []
        if not has_all_quarters:
            issues.append('missing_quarter')
        if not all_metrics_non_null:
            issues.append('null_metrics')
        if not single_provider_family:
            issues.append('mixed_providers')

        return pd.Series({
            'status': status,
            'issues': ','.join(issues)
        })

    year_quality = (
        df.groupby('effective_fiscal_year')
          .apply(_assess_year)
          .reset_index()
    )

    # --- identify poly rows that should be replaced --------------------------
    # rows that have *any* metric missing
    df['has_missing_metrics'] = df[list(metrics)].isna().any(axis=1)

    # per-year, do we have at least one non-poly row with metrics present?
    non_poly_with_metrics = df[
        (~df['provider'].eq('poly')) &
        df[list(metrics)].notna().any(axis=1)
    ]

    year_has_alt_provider = (
        non_poly_with_metrics
        .groupby('effective_fiscal_year')
        .size()
        .gt(0)
        .to_dict()
    )

    df['year_has_alt_provider'] = df['effective_fiscal_year'].map(year_has_alt_provider).fillna(False)

    # a row needs replacement if:
    #   - provider is poly
    #   - at least one of the metrics is NaN
    #   - the same fiscal year has a non-poly provider with metrics
    replace_mask = (
        df['provider'].eq('poly') &
        df['has_missing_metrics'] &
        df['year_has_alt_provider']
    )

    quarters_to_replace = (
        df.loc[replace_mask, ['effective_fiscal_year', 'fiscal_period']]
          .drop_duplicates()
          .sort_values(['effective_fiscal_year', 'fiscal_period'])
          .reset_index(drop=True)
    )
    quarters_to_replace["fiscal_period"] = quarters_to_replace["fiscal_period"].str.replace("Q0", "Q4")
    return year_quality, quarters_to_replace

def overwrite_with_selected(
    base_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    keep_unchanged: bool = True,
    ticker_col: str = "ticker",
    year_col: str = "effective_fiscal_year",
    period_col: str = "fiscal_period",
    period_type_col: str = "period_type",
    period_key_col: str = "period_key",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Overwrite rows in base_df with values from selected_df for matching
    (ticker, effective_fiscal_year, fiscal_period), only where selected_df
    has non-null values.

    Additional behaviour:
      - Any columns present in selected_df but missing in base_df are
        added to the output and populated only for the overwritten rows
        (NaN elsewhere).
      - Optionally blank out columns not present in selected_df, for the
        overwritten rows, if keep_unchanged=False.

    Also:
      - Updates period_type from selected_df.
      - Rebuilds period_key as:
          f"{ticker}_{effective_fiscal_year}_{quarter_number}_sec"
      - Returns a second DataFrame of rows where start/end date deltas
        between base and selected exceed 20 days.

    Parameters
    ----------
    base_df : DataFrame
        Original panel (Polygon/Finnhub merged).
    selected_df : DataFrame
        SEC-selected rows; must include ticker_col, year_col,
        period_col, period_type_col.
    keep_unchanged : bool, default True
        If True, columns that do not exist in selected_df are left unchanged
        in base_df for overwritten rows.
        If False, all non-meta columns that are not in selected_df are set
        to NaN for overwritten rows.
    ticker_col, year_col, period_col, period_type_col, period_key_col : str
        Column names for ticker, effective fiscal year, fiscal period,
        period type, and period key.

    Returns
    -------
    updated_df : DataFrame
        base_df with SEC values overwritten where applicable.
    date_mismatch_flags : DataFrame
        Rows (with keys and dates) where start/end date deltas exceed 20 days.
    """

    # Work on copies
    updated = base_df.copy()
    sel = selected_df.copy()

    # Ensure key columns are present in selected_df
    for col in (ticker_col, year_col, period_col, period_type_col):
        if col not in sel.columns:
            raise ValueError(f"selected_df is missing required column '{col}'")

    # Ensure dates are datetime where present
    for df_ in (updated, sel):
        for dcol in ("start_date", "end_date"):
            if dcol in df_.columns:
                df_[dcol] = pd.to_datetime(df_[dcol], errors="coerce")

    # Build MultiIndex views for alignment
    key_cols = [ticker_col, year_col, period_col]
    base_idxed = updated.set_index(key_cols)
    sel_idxed = sel.set_index(key_cols)

    # --- NEW: ensure all selected columns exist in base ---------------------
    # For any column in selected_df that's missing in base_df, add it
    missing_in_base = [c for c in sel_idxed.columns if c not in base_idxed.columns]
    if missing_in_base:
        # Create empty dataframe with NaN values for missing columns
        missing_df = pd.DataFrame(
            np.nan,
            index=base_idxed.index,
            columns=missing_in_base
        )
        # Concat to add missing columns (avoids fragmented dataframe warnings)
        base_idxed = pd.concat([base_idxed, missing_df], axis=1)

    # Keys to overwrite: intersection of (ticker, year, period)
    common_index = base_idxed.index.intersection(sel_idxed.index)
    if len(common_index) == 0:
        # nothing to do
        return base_idxed.reset_index(), pd.DataFrame()

    # Date mismatch checks
    date_mismatch_flags = pd.DataFrame()
    if {"start_date", "end_date"}.issubset(base_idxed.columns) and \
       {"start_date", "end_date"}.issubset(sel_idxed.columns):

        orig_dates = base_idxed.loc[common_index, ["start_date", "end_date"]].copy()
        new_dates = sel_idxed.loc[common_index, ["start_date", "end_date"]].copy()

        delta_start = (new_dates["start_date"] - orig_dates["start_date"]).dt.days.abs()
        delta_end = (new_dates["end_date"] - orig_dates["end_date"]).dt.days.abs()

        flags = pd.concat(
            [
                orig_dates.add_suffix("_orig"),
                new_dates.add_suffix("_new"),
                delta_start.rename("delta_start_days"),
                delta_end.rename("delta_end_days"),
            ],
            axis=1,
        )
        flags = flags[(flags["delta_start_days"] > 20) | (flags["delta_end_days"] > 20)]
        date_mismatch_flags = flags.reset_index()  # brings ticker/year/period back as columns

    # Meta columns: do not blank these even if keep_unchanged=False
    meta_cols = {
        ticker_col,
        year_col,
        period_col,
        period_type_col,
        period_key_col,
        "start_date",
        "end_date",
        "filing_date",
        "fiscal_year",
        "timeframe",
        "quarter",
        "form",
        "accepted_date",
        "access_number",
        "provider",
    }
    meta_cols = [c for c in base_idxed.columns if c in meta_cols]

    # Columns common to both base and selected: these can be overwritten
    common_cols = [c for c in base_idxed.columns if c in sel_idxed.columns]

    # Combine: selected overrides base where selected is non-null
    base_sub = base_idxed.loc[common_index, common_cols]
    sel_sub = sel_idxed.loc[common_index, common_cols]
    combined = sel_sub.combine_first(base_sub)  # non-null in selected win

    base_idxed.loc[common_index, common_cols] = combined

    # For columns missing in selected_df: possibly blank out metrics
    if not keep_unchanged:
        missing_cols = [c for c in base_idxed.columns if c not in sel_idxed.columns]
        for col in missing_cols:
            if col not in meta_cols:
                base_idxed.loc[common_index, col] = np.nan

    # Reset index back to ordinary columns
    updated = base_idxed.reset_index()

    # Update period_type explicitly (in case selected had a different value)
    if period_type_col in sel_idxed.columns:
        pt_vals = sel_idxed.loc[common_index, period_type_col]
        pt_df = pt_vals.reset_index()
        updated = updated.merge(
            pt_df,
            on=key_cols,
            how="left",
            suffixes=("", "_sec"),
        )
        mask_pt = updated[f"{period_type_col}_sec"].notna()
        updated.loc[mask_pt, period_type_col] = updated.loc[mask_pt, f"{period_type_col}_sec"]
        updated = updated.drop(columns=[f"{period_type_col}_sec"])

    # Rebuild period_key for overwritten rows
    if period_key_col in updated.columns:
        common_index_df = pd.MultiIndex.from_tuples(common_index, names=key_cols)
        mask_overwritten = updated.set_index(key_cols).index.isin(common_index_df)

        fp_str = updated.loc[mask_overwritten, period_col].astype(str)
        qnum = fp_str.str.extract(r"(\d+)", expand=False)

        updated.loc[mask_overwritten, period_key_col] = (
            updated.loc[mask_overwritten, ticker_col].astype(str)
            + "_"
            + updated.loc[mask_overwritten, year_col].astype("Int64").astype(str)
            + "_"
            + qnum
            + "_sec"
        )

    return updated, date_mismatch_flags

def classify_period_type(df: pd.DataFrame, start_col="start_date", end_col="end_date",
                         fy_threshold=300, ytd_threshold=120) -> pd.DataFrame:
    """Add period_type ∈ {FY, YTD_Q, Q} based on span days."""
    out = df.copy()
    start = pd.to_datetime(out[start_col], errors="coerce")
    end = pd.to_datetime(out[end_col], errors="coerce")
    days = (end - start).dt.days

    out["period_type"] = pd.NA
    out.loc[days >= fy_threshold, "period_type"] = "FY"
    out.loc[(days >= ytd_threshold) & (days < fy_threshold), "period_type"] = "YTD_Q"
    out.loc[days < ytd_threshold, "period_type"] = "Q"
    return out

def classify_fiscal_from_cache(df: pd.DataFrame,
                               fy_cache,
                               ticker_col: str = "ticker",
                               start_col: str = "start_date",
                               end_col: str = "end_date",
                               period_type_col: str = "period_type"):
    """
    Given:
      - df: SEC raw dataframe with at least
            [ticker, start_date, end_date, period_type]
      - fy_cache: either
          * dict keyed by (ticker, fiscal_year) -> approx fiscal-year start date
            e.g. {("MSFT", 2023): "2022-07-01", ...}
          * OR a DataFrame with columns:
              [ticker, effective_fiscal_year, fy_start]
    Returns df with new columns:
      - effective_fiscal_year  (int)
      - effective_fiscal_period  ('Q1'..'Q4')
    Classification is based on end_date via merge_asof onto fy_start per ticker.
    """
    out = df.copy()
    out[end_col] = pd.to_datetime(out[end_col])
    out[start_col] = pd.to_datetime(out[start_col])

    # --- build fiscal-year table from cache ---------------------------------
    if isinstance(fy_cache, dict):
        rows = []
        for (ticker, fy), dt in fy_cache.items():
            rows.append({
                ticker_col: ticker,
                "effective_fiscal_year": int(fy),
                "fy_start": pd.to_datetime(dt),
            })
        fy_table = pd.DataFrame(rows)
    else:
        fy_table = fy_cache.copy()
        # normalise column names to [ticker, effective_fiscal_year, fy_start]
        rename_map = {}
        for col in fy_table.columns:
            lc = col.lower()
            if lc == "ticker":
                rename_map[col] = ticker_col
            elif "year" in lc:
                rename_map[col] = "effective_fiscal_year"
            elif "start" in lc:
                rename_map[col] = "fy_start"
        fy_table = fy_table.rename(columns=rename_map)

    if fy_table.empty:
        out["effective_fiscal_year"] = pd.NA
        out["effective_fiscal_period"] = pd.NA
        return out

    fy_table["fy_start"] = pd.to_datetime(fy_table["fy_start"])
    fy_table = fy_table.sort_values([ticker_col, "fy_start"]).reset_index(drop=True)

    # --- attach fiscal year via merge_asof on end_date ----------------------
    out_sorted = out.sort_values([ticker_col, end_col]).reset_index(drop=True)
    merged = pd.merge_asof(
        out_sorted,
        fy_table.sort_values([ticker_col, "fy_start"]),
        left_on=end_col,
        right_on="fy_start",
        by=ticker_col,
        direction="backward",
        tolerance=pd.Timedelta(days=500),  # generous to allow 52/53-week drift
    )

    # --- infer fiscal quarter based on end_date vs fy_start -----------------
    def _infer_quarter(row):
        end = row[end_col]
        fy_start = row["fy_start"]
        if pd.isna(end) or pd.isna(fy_start):
            return pd.NA
        # months difference from fy_start to end
        diff_months = (end.year - fy_start.year) * 12 + (end.month - fy_start.month)
        q = diff_months // 3 + 1
        if q < 1:
            q = 1
        if q > 4:
            q = 4
        return f"Q{int(q)}"

    merged["effective_fiscal_period"] = merged.apply(_infer_quarter, axis=1)

    # restore original row order
    merged = merged.sort_index()
    return merged

def select_preferred_period_rows(
    df: pd.DataFrame,
    keys: pd.DataFrame,
    ticker_col: str = "ticker",
    year_col: str = "effective_fiscal_year",
    period_col: str = "fiscal_period",
    period_type_col: str = "period_type",
):
    """
    Select rows from df whose (year_col, period_col) are in keys, resolving
    duplicates and prioritising period_type.

    Logic:
      1) Inner-join df to keys on (year_col, period_col).
      2) For each (ticker, year_col, period_col, period_type_col):
           - If multiple rows exist, choose the row with the most non-null
             metric values.
           - Then fill NaNs in that chosen row from the other rows in the
             same group (same period_type only).
      3) For each (ticker, year_col, period_col) choose one period_type
         with priority: 'Q' < 'YTD_Q' < 'FY' < others.

    Parameters
    ----------
    df : DataFrame
        Source SEC dataframe; must have ticker_col, year_col, period_col,
        period_type_col plus metric columns.
    keys : DataFrame
        DataFrame with at least [year_col, period_col] specifying fiscal
        year + period pairs to select.
    ticker_col, year_col, period_col, period_type_col : str
        Column names as described.

    Returns
    -------
    DataFrame
        One row per (ticker, year_col, period_col) in keys, with the best
        period_type and de-duplicated / NaN-filled metrics.
    """

    # 1) Restrict to requested (year, period) combinations
    merged = df.merge(
        keys[[year_col, period_col]].drop_duplicates(),
        on=[year_col, period_col],
        how="inner",
    )

    if merged.empty:
        return merged

    # Identify meta vs metric columns
    # Meta columns: anything structural; metrics = everything else
    candidate_meta = {
        ticker_col,
        year_col,
        period_col,
        period_type_col,
        "start_date",
        "end_date",
        "filing_date",
        "timeframe",
        "fiscal_year",
        "form",
    }
    meta_cols = [c for c in merged.columns if c in candidate_meta]
    metric_cols = [c for c in merged.columns if c not in meta_cols]

    # 2) Collapse duplicates within (ticker, year, period, period_type)
    group_cols = [ticker_col, year_col, period_col, period_type_col]

    def _collapse_group(g: pd.DataFrame) -> pd.Series:
        if len(g) == 1:
            return g.iloc[0]

        # pick row with most non-null metric values
        nn = g[metric_cols].notna().sum(axis=1)
        best_idx = nn.idxmax()
        base = g.loc[best_idx].copy()

        # fill NaNs in base from other rows in same group (same period_type)
        for idx, row in g.iterrows():
            if idx == best_idx:
                continue
            mask = base[metric_cols].isna() & row[metric_cols].notna()
            if mask.any():
                base.loc[metric_cols] = base[metric_cols].where(~mask, row[metric_cols])

        return base

    collapsed = (
        merged.groupby(group_cols, as_index=False, group_keys=False)
              .apply(_collapse_group)
              .reset_index(drop=True)
    )

    # 3) Prioritise period_type per (ticker, year, period)
    priority_map = {"Q": 0, "YTD_Q": 1, "FY": 2}
    collapsed["_pt_priority"] = (
        collapsed[period_type_col].map(priority_map).fillna(3).astype(int)
    )

    collapsed = (
        collapsed.sort_values([ticker_col, year_col, period_col, "_pt_priority"])
                 .drop_duplicates(subset=[ticker_col, year_col, period_col], keep="first")
                 .drop(columns=["_pt_priority"])
    )

    return collapsed



# def build_quarterised_with_sec(ticker: str, analyzer: TTMRatiosAnalyzer,
#                                start_dt: str = "2020-01-01",
#                                end_dt: str = "2025-12-31",
#                                return_raw : bool = True,
#                                ) -> pd.DataFrame:
#     """
#     Concise version: Build quarterised financials with SEC replacements.
    
#     Steps:
#     1. Load merged financials (Polygon + Finnhub)
#     2. Fetch and classify SEC data
#     3. Identify bad quarters and replace with SEC
#     4. Apply canonical mapping and bucket aggregation
#     5. Quarterise, fix outliers, and back out periods
#     """
#     # Load merged data
#     analyzer._capture_bucket_components = True
#     quarterised_financials, _, financials = analyzer._load_financials_merged(
#                                                     ticker, start_dt, end_dt, return_raw=True
#                                                 )
    
#     # Get fiscal year cache
#     fy_cache = analyzer._fy_cache_
    
#     # Fetch SEC data
#     sec_raw = analyzer.fetch_sec_financials(ticker, force_update=True)
#     if sec_raw.empty:
#         # No SEC data - fallback to regular quarterisation
#         quarterised = analyzer._quarterize_flows(financials)
#         return quarterised
    
#     # Classify and filter SEC periods
#     start_offset = pd.to_datetime(start_dt) - pd.DateOffset(months=6)
#     classified_sec = classify_period_type(sec_raw)
#     filtered_sec = filter_dates(classified_sec, start_offset, end_dt, date_col="end_date")
    
#     # Identify bad quarters in merged data
#     _, quarters_to_replace = assess_years_and_poly_quarters(financials)
#     if return_raw:
#         finnhub_poly_data = financials.copy()
#     if quarters_to_replace.empty:
#         # No bad quarters - use merged data as-is
#         print("No bad quarters found, passing through to build quarterised bucket views")
#         return analyzer._build_quarterized_bucket_views(quarterised_financials, ticker)
    
#     # Tag SEC data and select preferred rows
#     tagged_sec = classify_fiscal_from_cache(filtered_sec, fy_cache=fy_cache).sort_values(by=["fy_start", "effective_fiscal_period"])
#     selected = select_preferred_period_rows(tagged_sec, quarters_to_replace)
#     if return_raw:
#         selected_preferred_rows_from_sec = selected.copy()
#     # Overwrite bad rows with SEC data
#     df_overwrite = overwrite_with_selected(financials, selected)[0].sort_values(by=['effective_fiscal_year', 'quarter'])

#     # Apply canonical mapping and bucket aggregation
#     analyzer._map_fields_to_buckets(df_overwrite.columns, ticker=ticker)
#     canonical = analyzer._prepare_canonical_financials(df_overwrite)
#     analyzer._apply_bucket_aggregates(canonical)
    
#     # Replace zeros with NaN for metrics
#     metric_cols = [c for c in BUCKET_METRIC_MAP.values() if c in canonical.columns]
#     canonical[metric_cols] = canonical[metric_cols].replace(0.0, np.nan)
    
#     # Quarterise, fix outliers, back out periods
#     quarterised = analyzer._quarterize_flows(canonical)
#     quarterised = analyzer._fix_share_count_outliers(quarterised)
    
#     if return_raw == True:
#         return (analyzer._back_out_periods(quarterised, False, True, fy_cache),
#                 df_overwrite, fy_cache, finnhub_poly_data,
#                 metric_cols, selected_preferred_rows_from_sec,
#                 canonical, tagged_sec)
#     else:
#         return analyzer._back_out_periods(quarterised, False, True, fy_cache)

# def aggregate_single_bucket(df: pd.DataFrame, bucket_key: str,
#                             analyzer: TTMRatiosAnalyzer, return_details: bool = False):
#     """
#     Replicate bucket aggregation for a single metric bucket.
    
#     Args:
#         df: Pre-aggregation dataframe (raw financials with component columns)
#         bucket_key: Bucket name from metric_components2.json (e.g., "cash_and_cash_equivalents")
#         analyzer: TTMRatiosAnalyzer instance with loaded bucket components
#         return_details: If True, return (sum, components_df) tuple with component series
    
#     Returns:
#         If return_details=False: Series with bucket sum
#         If return_details=True: (Series with bucket sum, DataFrame with each component as a column)
#     """
#     # Ensure bucket components are loaded
#     analyzer._ensure_bucket_components_loaded()
#     df = df.copy()
#     df = analyzer._back_out_periods(df)
#     # Get component definitions for this bucket {canonical_name: sign}
#     components = analyzer._bucket_components.get(bucket_key, {})
#     if not components:
#         result = pd.Series(np.nan, index=df.index)
#         return (result, pd.DataFrame(index=df.index)) if return_details else result
    
#     # Sum components with their signs and collect component series
#     result = pd.Series(0.0, index=df.index)
#     mask = pd.Series(False, index=df.index)
#     component_data = {}
    
#     for canonical, sign in components.items():
#         col = analyzer._resolve_column_for_canonical(df, canonical)
#         if col:
#             series = pd.to_numeric(df[col], errors='coerce')
#             result += sign * series.fillna(0)
#             mask |= series.notna()
#             # Store with sign indicator in column name
#             component_data[col] = series
    
#     result = result.where(mask)  # NaN where all components are NaN
    
#     if return_details:
#         components_df = pd.DataFrame(component_data, index=df.index)
#         components_df['__bucket_sum__'] = result
#         return result, components_df
#     return result

# def debug_aggregate_single_bucket_sec(ticker, analyzer, metric):
#     quarterised_financials, raw, fy_cache = build_quarterised_with_sec(ticker,analyzer)
#     bucket_sum, components_df = aggregate_single_bucket(raw, metric, analyzer, return_details=True)
#     components = pd.Series(components_df.columns)
#     components = components[components != '__bucket_sum__'].apply(
#         lambda col : analyzer._strip_statement_prefix(col)
#         )
#     sec_raw_stock, sec_raw_flow = _load_SEC_from_parquet(ticker, analyzer)
#     sec_long = pd.concat([sec_raw_stock, sec_raw_flow])
#     identified_sec_long = sec_long[sec_long["canonical"].isin(components)].copy()
#     start_date = pd.to_datetime(identified_sec_long["end_date"]) - pd.DateOffset(months=3)
#     identified_sec_long["start_date"] = identified_sec_long["start_date"].fillna(start_date)
#     df = classify_fiscal_from_cache(identified_sec_long, fy_cache=fy_cache).sort_values(by=["fy_start", "effective_fiscal_period"])
#     pivot = pd.pivot_table(df, index=["start_date", "end_date","effective_fiscal_period", "effective_fiscal_year"], columns='canonical', values=['val', "raw_concept"], aggfunc='first')
#     pivot = pivot.swaplevel(0,1, axis=1).reset_index()
#     pivot["effective_fiscal_year"] = pivot["effective_fiscal_year"].astype(int)
#     pivot["period"] = pivot["effective_fiscal_year"].astype(str) + pivot["effective_fiscal_period"].astype(str)
#     pivot = pivot.set_index("period")

#         # Flatten pivot's multi-level columns first
#     pivot_flat = pivot.copy()
#     pivot_flat.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
#                         for col in pivot_flat.columns]

#     # Reset indices
#     components_with_period = components_df.reset_index()
#     pivot_with_period = pivot_flat.reset_index()

#     # Inner merge on 'period' column - only keeps rows that exist in components_df
#     combined = pd.merge(
#         components_with_period,
#         pivot_with_period,
#         on="period",
#         how="inner",
#         suffixes=("", "_sec")
#     )

#     return combined, components_df.sort_index(), bucket_sum, fy_cache, quarterised_financials


# def debug_aggregate_single_bucket_finn(ticker, analyzer, metric_name, fy_cache):
#     """
#     Debug function to analyze Finnhub raw data for a specific metric bucket.
    
#     Args:
#         ticker: Stock ticker symbol
#         analyzer: TTMRatiosAnalyzer instance
#         metric_name: Name of the metric bucket (e.g., "cash_and_cash_equivalents")
#         fy_cache: Fiscal year cache dictionary
    
#     Returns:
#         DataFrame: Pivoted data with period as index and multi-level columns
#     """
#     metric_buckets = analyzer._load_metric_buckets().get("buckets")
#     buckets = {k: v["metric"] for k, v in metric_buckets.items()}

#     # Get the prefixed metric name
#     if metric_name not in buckets:
#         raise KeyError(f"Metric '{metric_name}' not found in bucket definitions")
    
#     metric_with_prefix = buckets[metric_name]
    
#     # Get bucket components and clean metadata
#     buckets_can = analyzer.get_bucket_component_series(metric_with_prefix)
#     buckets_can = analyzer._back_out_periods(buckets_can)
#     meta_cols = [col for col in analyzer._META_DROP_COLUMNS if col in buckets_can.columns]
    
#     # Get raw Finnhub data
#     raw_finnhub = analyzer.analyze_raw_finnhub()
#     buckets_can = buckets_can.drop(columns=meta_cols).drop(metric_with_prefix, axis=1)
#     finnhub_canon_cols = buckets_can.columns.tolist()
    
#     # Filter for relevant components
#     raw_finn = raw_finnhub[raw_finnhub["canonical_mapped"].isin(finnhub_canon_cols)][
#         ["start_date", "end_date", "quarter", "concept_raw", "canonical_mapped", "value"]
#     ].copy()
#     raw_finn["ticker"] = ticker
    
#     # Classify fiscal periods
#     df = classify_fiscal_from_cache(
#         raw_finn, 
#         fy_cache=fy_cache
#     ).sort_values(by=["fy_start", "effective_fiscal_period"]).rename(
#         columns={"value": "val", "concept_raw": "raw_concept", "canonical_mapped": "canonical"}
#     )
    
#     # Pivot to wide format
#     pivot = pd.pivot_table(
#         df, 
#         index=["start_date", "end_date", "effective_fiscal_period", "effective_fiscal_year"], 
#         columns='canonical', 
#         values=['val', "raw_concept"], 
#         aggfunc='first'
#     )
#     pivot = pivot.swaplevel(0, 1, axis=1).reset_index()
#     pivot["effective_fiscal_year"] = pivot["effective_fiscal_year"].astype(int)
#     pivot["period"] = pivot["effective_fiscal_year"].astype(str) + pivot["effective_fiscal_period"].astype(str)
#     pivot = pivot.set_index("period")

#     return pivot


# def merged_financials_with_debug_full_pipeline(
#         ticker: str, analyzer, metric_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict, pd.DataFrame]:
#     """
#     Compare SEC and Finnhub/Polygon data sources for a specific metric bucket.
    
#     This wrapper function calls both debug_aggregate_single_bucket_sec and 
#     debug_aggregate_single_bucket_finn, then merges their outputs into a 
#     multi-level column DataFrame where:
#     - Level 0 indicates the data source: 'SEC' or 'Finnhub/Polygon'
#     - Level 1 contains the actual column names
#     - Metadata columns (start_date, end_date, etc.) are shared (no source label)
#     """
#     # Get SEC data
#     combined, components_df, bucket_sum, fy_cache, quarterised_financials = debug_aggregate_single_bucket_sec(
#         ticker, analyzer, metric_name
#     )
    
#     # Get Finnhub/Polygon data
#     pivot = debug_aggregate_single_bucket_finn(
#         ticker, analyzer, metric_name, fy_cache
#     )
    
#     # Flatten pivot's multi-level columns for merging
#     pivot_flat = pivot.copy()
#     pivot_flat.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else col 
#                           for col in pivot_flat.columns]
#     pivot_flat = pivot_flat.reset_index()
    
#     # Prepare combined for merge (already flat)
#     combined_clean = combined.copy()
    
#     # Merge on period
#     merged = pd.merge(
#         combined_clean,
#         pivot_flat,
#         on='period',
#         how='outer',
#         suffixes=('_sec', '_finn')
#     )
    
#     # Define metadata columns
#     metadata_cols = ['start_date', 'end_date', 'effective_fiscal_period', 'effective_fiscal_year']
    
#     # Identify source-specific columns
#     sec_component_cols = [col for col in combined.columns 
#                           if col not in ['period'] + metadata_cols]
    
#     pivot_original_cols = [col for col in pivot_flat.columns 
#                            if col not in ['period'] + metadata_cols]
    
#     # Build multi-level column dictionary
#     multi_level_dict = {}
    
#     # Add period as shared metadata (no source label)
#     multi_level_dict[('', 'period')] = merged['period']
    
#     # Add metadata columns as shared (no source label) - prefer SEC, else Finnhub
#     for col in metadata_cols:
#         sec_col = f"{col}_sec"
#         finn_col = f"{col}_finn"
        
#         if sec_col in merged.columns:
#             multi_level_dict[('', col)] = merged[sec_col]
#         elif finn_col in merged.columns:
#             multi_level_dict[('', col)] = merged[finn_col]
#         elif col in merged.columns:
#             multi_level_dict[('', col)] = merged[col]
    
#     # Add SEC component columns
#     for col in sec_component_cols:
#         if col in merged.columns:
#             multi_level_dict[('SEC', col)] = merged[col]
#         elif f"{col}_sec" in merged.columns:
#             multi_level_dict[('SEC', col)] = merged[f"{col}_sec"]
    
#     # Add Finnhub/Polygon columns
#     for col in pivot_original_cols:
#         if col in merged.columns:
#             multi_level_dict[('Finnhub/Polygon', col)] = merged[col]
#         elif f"{col}_finn" in merged.columns:
#             multi_level_dict[('Finnhub/Polygon', col)] = merged[f"{col}_finn"]
    
#     # Create multi-level dataframe
#     merged_multi = pd.DataFrame(multi_level_dict)
    
#     # Set period as index
#     merged_multi = merged_multi.set_index(('', 'period'))
#     merged_multi.index.name = 'period'
    
#     # Sort by period
#     merged_multi = merged_multi.sort_index()
#     print("[RETURNING] 1. Merged Dataframe of Metric Components from SEC and Finnhub/Polygon\n2." \
#           "Component DataFrame from SEC\n3. Bucket Sum Series from SEC\n" \
#           "4. Fiscal Year Cache\n5. Quarterised Financials DataFrame")
#     return merged_multi, components_df, bucket_sum, fy_cache, quarterised_financials


def overwrite_quarterised_rows(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    keys: pd.DataFrame,
    keep_unchanged: bool = True,
    year_col: str = "effective_fiscal_year",
    period_col: str = "fiscal_period"
) -> pd.DataFrame:
    """Overwrite rows in base_df with source_df for (year_col, period_col) in keys."""
    updated = base_df.copy()
    key_cols = [year_col, period_col]
    
    # Create MultiIndex from keys
    keys_idx = pd.MultiIndex.from_frame(keys[key_cols].drop_duplicates())
    
    # Set index for alignment
    base_idxed = updated.set_index(key_cols)
    source_idxed = source_df.set_index(key_cols)
    
    # Find matching rows (intersection of keys and available in both dataframes)
    common_idx = base_idxed.index.intersection(source_idxed.index).intersection(keys_idx)
    
    if len(common_idx) == 0:
        return updated
    
    # Overwrite common columns (source non-null wins)
    common_cols = [c for c in base_idxed.columns if c in source_idxed.columns]
    source_vals = source_idxed.loc[common_idx, common_cols]
    base_vals = base_idxed.loc[common_idx, common_cols]
    base_idxed.loc[common_idx, common_cols] = source_vals.combine_first(base_vals)
    
    # Handle missing columns
    if not keep_unchanged:
        missing_cols = [c for c in base_idxed.columns if c not in source_idxed.columns]
        base_idxed.loc[common_idx, missing_cols] = np.nan
    
    return base_idxed.reset_index()



_NEW_FUNCTIONS_ = [
    "_get_sec_data_for_ticker",
    "_process_sec_long_into_wide",
    "_aggregate_single_bucket",
    "_debug_aggregate_single_bucket_sec",
    "_debug_aggregate_single_bucket_finn",
    "_merged_financials_with_debug_full_pipeline",
    "_overwrite_quarterised_rows",
    "get_bucket_components",
    "map_complex_cols",
    "classify_effective_fiscal_quarters",
    "classify_effective_fiscal_quarters_stock",
    "pick_quarters",
    "quarterize_per_metric_sec_flows",
    "get_all_metric_components_in_array",
    "get_raw_to_canon_from_sec",
    "set_period_as_index",
]

def get_sec_data_for_ticker(ticker, analyzer, start_dt : str="2020-01-01"):

    # Load SEC raw data (if it exists)
    try:
        sec_raw = _load_SEC_from_parquet(ticker, analyzer)
    except Exception as e:
        print(f"[FETCHING] No Existing Paraquet for {ticker}")
        analyzer._load_sec_financials(ticker, start_dt, save=True, force_update=True)
        sec_raw = _load_SEC_from_parquet(ticker, analyzer)

    # Combine flow and stock data and backfill start_dates (3months offset)
    sec_long       = pd.concat([sec_raw[0], sec_raw[1]])
    start_date_ser = pd.to_datetime(sec_long["end_date"]) - pd.DateOffset(months=3)
    sec_long["start_date"] = sec_long["start_date"].fillna(start_date_ser)

    return sec_long

def process_sec_long_into_wide(ticker, sec_long, fy_cache, start_year : int = 2020, raw_concept_or_canonical = "canonical"):
    """ 
    Takes the combined flow + stock data => correctly classify period type and years 
    Also pivots the long df to a wide df
    """
    sec_long = classify_fiscal_from_cache(sec_long, fy_cache)
    sec_long = sec_long[sec_long["effective_fiscal_year"] >= 2020].copy()

    idx_cols = ["start_date", "end_date", "fy", "fp", "effective_fiscal_year", "fy_start", "effective_fiscal_period"]
    for c in ["start_date", "end_date"]: 
        sec_long[c] = pd.to_datetime(sec_long[c], errors="coerce")
    
    return (sec_long
            .pivot_table(
                index=idx_cols,
                columns=raw_concept_or_canonical,
                values="val",
                aggfunc="first"
            )).reset_index().sort_values(by=["start_date", "end_date"]
             ).set_index(["effective_fiscal_year", "effective_fiscal_period"])

def collapse_sec_wide_df(wide_df):
    grouped = wide_df.groupby(level=["effective_fiscal_year", "effective_fiscal_period"]).first().dropna(axis=1, how='all')
    return grouped.reset_index()

def apply_sum_buckets_sec(grouped, analyzer):
    analyzer._map_fields_to_buckets(grouped.columns, ticker="MSFT")
    canonical_df = analyzer._prepare_canonical_financials(grouped)
    analyzer._apply_bucket_aggregates(canonical_df)
    return canonical_df

def apply_sum_buckets_sec_simple(grouped, analyzer, complex_metrics):
    from Modules.ttm_ratios_analyzer import BUCKET_METRIC_MAP
    MAPPING = {
        analyzer._strip_statement_prefix(v): v
        for k, v in BUCKET_METRIC_MAP.items()
        if v not in complex_metrics.keys()
                        }

    keep_cols = [c for c in grouped.columns if c in MAPPING.keys()] 
    meta_cols =[c for c in ["effective_fiscal_year", "effective_fiscal_period", "start_date", "end_date"] if c in grouped.columns]
    grouped2 = grouped[keep_cols + meta_cols]
    grouped2 = grouped2.set_index(["effective_fiscal_year", "effective_fiscal_period"])
    grouped2 = grouped2.rename(columns=MAPPING)

    return grouped2

def get_bucket_components(canonical, grouped, analyzer):

    components = analyzer._load_metric_components()["buckets"].get(canonical)
    components_bucket = components["components"]
    cols = []
    for comp, sign in components_bucket.items():
        
        column_name = analyzer._resolve_column_for_canonical(grouped, comp)
        if column_name:
            cols.append(column_name)

    return cols

def map_complex_cols(
    df: pd.DataFrame,
    merged_quarterised: pd.DataFrame,
    col_list: List[str]
) -> pd.DataFrame:
    """
    Fill NaN values in df with values from merged_quarterised for specified columns.
    Aligns by index to ensure correct mapping, only fills NaN values (does not overwrite existing values).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Base dataframe to update
    merged_quarterised : pd.DataFrame
        Source dataframe with values to fill
    col_list : List[str]
        List of column names to fill from merged_quarterised
        
    Returns:
    --------
    pd.DataFrame
        Updated dataframe with NaN values filled from merged_quarterised
    """
    updated = df.copy()
    
    # Filter to only columns that exist in merged_quarterised
    available_cols = [col for col in col_list if col in merged_quarterised.columns]
    
    if not available_cols:
        return updated
    
    # Check if we can align by index
    common_idx = df.index.intersection(merged_quarterised.index)
    
    if len(common_idx) > 0:
        # Use combine_first to fill NaNs: updated values take precedence, then fill from merged_quarterised
        for col in available_cols:
            if col not in updated.columns:
                # Create column if it doesn't exist
                updated[col] = np.nan
            
            # Get source values aligned to common index
            source_values = merged_quarterised.loc[common_idx, col]
            
            # Use combine_first: fills NaN in updated with values from source
            updated.loc[common_idx, col] = updated.loc[common_idx, col].combine_first(source_values)
    else:
        # If no common index, try to align by key columns
        key_cols = ["ticker", "effective_fiscal_year", "quarter"]
        if all(col in df.columns and col in merged_quarterised.columns for col in key_cols):
            # Merge on key columns to align data (left merge preserves updated's index)
            merged_aligned = updated.merge(
                merged_quarterised[key_cols + available_cols],
                on=key_cols,
                how="left",
                suffixes=("", "_source")
            )
            
            # Fill NaN values using combine_first for each column
            for col in available_cols:
                source_col = f"{col}_source"
                if source_col in merged_aligned.columns:
                    if col not in updated.columns:
                        updated[col] = np.nan
                    # Use combine_first: fills NaN in original with values from source
                    # merged_aligned has the same index as updated (left merge)
                    updated[col] = updated[col].combine_first(merged_aligned[source_col])
    
    return updated

def classify_effective_fiscal_quarters(
    sec_long: pd.DataFrame,
    fy_cache: Dict[Tuple[str, int], Any],
    analyzer: TTMRatiosAnalyzer = None,
    ticker: str = None,
    raw_concept_or_canonical: str = "concept",
    metric : str = "revenues",
    min_fy: int = 2020,
    max_fy: Optional[int] = None,
) -> pd.DataFrame:
    """
    Classify effective fiscal quarters for a given ticker and raw_concept
    using fy_cache and (start_date, end_date).

    Logic (per fiscal year):
      1) Get fiscal-year start (from fy_cache) and fiscal-year end
         (next year's start - 1 day; if missing, assume +365.25 days).
      2) Define quarter "target" end dates:
         Q1: fy_start + 91.25 days
         Q2: fy_start + 182.5 days
         Q3: fy_start + 273.75 days
         Q4: fy_end
      3) Filter SEC rows whose (start_date, end_date) fall inside this FY,
         with span < 400 days.
      4) Use merge_asof to map each row's end_date to the nearest quarter end.
      5) Classify period_type by span:
           span > 315  -> 'FY'
           span < 120  -> 'Q'
           else        -> 'YTD'
      6) Attach effective_fiscal_year and effective_fiscal_period (Q1–Q4).
    All fiscal years from min_fy up to max_fy are concatenated into a
    single output DataFrame.
    """
    if raw_concept_or_canonical == "raw_concept":
        metric_canon = analyzer._finnhub_field_to_canon(metric, ticker)
    else:
        metric_canon = metric
    

    # ------------- 1) Filter to this ticker + concept and prep base df -------------
    df = sec_long[(sec_long["ticker"] == ticker) & (sec_long[raw_concept_or_canonical] == metric)].copy()

    # Drop labels we don't trust / don't need (if present)
    for col in ["fp", "fy", "unit", "filing_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure datetime
    for col in ["start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Optional: backfill missing start_date with end_date - 3 months
    mask_missing_start = df["start_date"].isna() & df["end_date"].notna()
    df.loc[mask_missing_start, "start_date"] = (
        df.loc[mask_missing_start, "end_date"] - pd.DateOffset(months=3)
    )

    # Compute span
    df["span"] = (df["end_date"] - df["start_date"]).dt.days

    # ------------- 2) Determine fiscal years to process for this ticker -------------
    # Collect FYs for this ticker from fy_cache
    fy_years_for_ticker: List[int] = sorted(
        fy for (t, fy) in fy_cache.keys() if t == ticker
    )

    if max_fy is not None:
        fy_years_for_ticker = [fy for fy in fy_years_for_ticker if min_fy <= fy <= max_fy]
    else:
        fy_years_for_ticker = [fy for fy in fy_years_for_ticker if fy >= min_fy]

    out_frames: List[pd.DataFrame] = []

    # ------------- 3) Iterate through fiscal years and classify quarters -------------
    for fy_year in fy_years_for_ticker:
        # Get FY start from fy_cache
        fy_start = fy_cache.get((ticker, fy_year), None)
        if fy_start is None:
            continue  # skip if no mapping

        fy_start = pd.to_datetime(fy_start)

        # Determine FY end: next fy_start - 1 day if available, else assume +365.25 days
        next_start = fy_cache.get((ticker, fy_year + 1), None)
        if next_start is not None:
            fy_end = pd.to_datetime(next_start) - pd.DateOffset(days=1)
        else:
            fy_end = fy_start + pd.DateOffset(days=365)  # simple fallback

        # Quarter end approximations (same as your code)
        q1_end_date = fy_start + pd.DateOffset(days=91.25)
        q2_end_date = fy_start + pd.DateOffset(days=182.5)
        q3_end_date = fy_start + pd.DateOffset(days=273.75)
        q4_end_date = fy_end

        date_buckets = pd.Series([q1_end_date, q2_end_date, q3_end_date, q4_end_date])
        spans = pd.Series([91.25, 182.5, 273.75, 365.25])
        labels = pd.Series(["Q1", "Q2", "Q3", "Q4"])

        date_map = pd.concat([date_buckets, spans, labels], axis=1).reset_index(drop=True)
        date_map = date_map.rename(columns={0: "end_date", 1: "span", 2: "quarter"})

        # Filter "year-like" periods (span < 400) and within this fiscal-year window
        tmp = df[(df["span"] < 400)].copy().drop_duplicates()

        mask = (
            (tmp["start_date"] >= fy_start) &
            (tmp["end_date"] <= fy_end)
        )
        tmp = tmp[mask].copy().sort_values(["end_date"])

        if tmp.empty:
            continue

        # merge_asof requires both sides sorted by key
        date_map = date_map.sort_values("end_date")
        tmp = pd.merge_asof(
            left=tmp.sort_values("end_date"),
            right=date_map,
            on="end_date",
            direction="nearest",
            suffixes=("", "_map"),
        )

        # Classify period_type by actual span, as in your logic
        tmp["period_type"] = np.where(tmp["span"] > 315, "FY", "YTD_Q")
        tmp["period_type"] = np.where(tmp["span"] < 120, "Q", tmp["period_type"])

        # Attach effective fiscal year and effective fiscal quarter label
        tmp["effective_fiscal_year"] = fy_year
        tmp["effective_fiscal_period"] = tmp["quarter"]  # Q1/Q2/Q3/Q4

        out_frames.append(tmp)

    # ------------- 4) Concatenate all fiscal-year slices into one DataFrame -------------
    if not out_frames:
        return pd.DataFrame()

    result = pd.concat(out_frames, ignore_index=True)

    # Optional: reorder columns for readability
    cols_order = [
        "ticker",
        "raw_concept",
        "canonical",
        "start_date",
        "end_date",
        "span",
        "effective_fiscal_year",
        "effective_fiscal_period",
        "period_type",
        "quarter",       # from date_map
        "span_target",   # target quarter span used for mapping
    ]
    # Only keep columns that actually exist
    cols_order = [c for c in cols_order if c in result.columns] + [
        c for c in result.columns if c not in cols_order
    ]
    result = result[cols_order]

    return result

def classify_effective_fiscal_quarters_stock(
    sec_long: pd.DataFrame,
    fy_cache: Dict[Tuple[str, int], Any],
    analyzer: TTMRatiosAnalyzer = None,
    ticker: str = None,
    raw_concept_or_canonical: str = "canonical",
    metric: str = "cash_and_cash_equivalents",
    min_fy: int = 2020,
    max_fy: Optional[int] = None,
) -> pd.DataFrame:
    """
    Classify effective fiscal quarters for STOCK-based columns (point-in-time balances).
    Similar to classify_effective_fiscal_quarters but adapted for stock columns which
    don't have start_date or span - they are snapshots at end_date.

    Logic (per fiscal year):
      1) Get fiscal-year start (from fy_cache) and fiscal-year end
         (next year's start - 1 day; if missing, assume +365 days).
      2) Define quarter "target" end dates:
         Q1: fy_start + 91.25 days
         Q2: fy_start + 182.5 days
         Q3: fy_start + 273.75 days
         Q4: fy_end
      3) Filter SEC rows whose end_date falls inside this FY.
      4) Use merge_asof to map each row's end_date to the nearest quarter end.
      5) Attach effective_fiscal_year and effective_fiscal_period (Q1–Q4).
      6) Set period_type to "STOCK" (point-in-time balance).
    All fiscal years from min_fy up to max_fy are concatenated into a
    single output DataFrame.
    """
    if raw_concept_or_canonical == "raw_concept":
        if analyzer is None or ticker is None:
            raise ValueError("analyzer and ticker required when raw_concept_or_canonical='raw_concept'")
        metric_canon = analyzer._finnhub_field_to_canon(metric, ticker)
    else:
        metric_canon = metric

    # ------------- 1) Filter to this ticker + concept and prep base df -------------
    df = sec_long[(sec_long["ticker"] == ticker) & (sec_long[raw_concept_or_canonical] == metric_canon)].copy()

    if df.empty:
        return pd.DataFrame()

    # Drop labels we don't trust / don't need (if present)
    for col in ["fp", "fy", "unit", "filing_date", "start_date"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure datetime for end_date (stock columns only have end_date)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    
    # Drop rows with invalid end_date
    df = df[df["end_date"].notna()].copy()
    
    if df.empty:
        return pd.DataFrame()

    # ------------- 2) Determine fiscal years to process for this ticker -------------
    # Collect FYs for this ticker from fy_cache
    fy_years_for_ticker: List[int] = sorted(
        fy for (t, fy) in fy_cache.keys() if t == ticker
    )

    if max_fy is not None:
        fy_years_for_ticker = [fy for fy in fy_years_for_ticker if min_fy <= fy <= max_fy]
    else:
        fy_years_for_ticker = [fy for fy in fy_years_for_ticker if fy >= min_fy]

    out_frames: List[pd.DataFrame] = []

    # ------------- 3) Iterate through fiscal years and classify quarters -------------
    for fy_year in fy_years_for_ticker:
        # Get FY start from fy_cache
        fy_start = fy_cache.get((ticker, fy_year), None)
        if fy_start is None:
            continue  # skip if no mapping

        fy_start = pd.to_datetime(fy_start)

        # Determine FY end: next fy_start - 1 day if available, else assume +365 days
        next_start = fy_cache.get((ticker, fy_year + 1), None)
        if next_start is not None:
            fy_end = pd.to_datetime(next_start) - pd.DateOffset(days=1)
        else:
            fy_end = fy_start + pd.DateOffset(days=365)  # simple fallback

        # Quarter end approximations
        q1_end_date = fy_start + pd.DateOffset(days=91.25)
        q2_end_date = fy_start + pd.DateOffset(days=182.5)
        q3_end_date = fy_start + pd.DateOffset(days=273.75)
        q4_end_date = fy_end

        date_buckets = pd.Series([q1_end_date, q2_end_date, q3_end_date, q4_end_date])
        spans = pd.Series([91.25, 182.5, 273.75, 365.25])
        labels = pd.Series(["Q1", "Q2", "Q3", "Q4"])

        date_map = pd.concat([date_buckets, spans, labels], axis=1).reset_index(drop=True)
        date_map = date_map.rename(columns={0: "end_date", 1: "span", 2: "quarter"})

        # Filter rows within this fiscal-year window (stock columns: only check end_date)
        mask = (
            (df["end_date"] >= fy_start) &
            (df["end_date"] <= fy_end)
        )
        tmp = df[mask].copy().drop_duplicates().sort_values("end_date")

        if tmp.empty:
            continue

        # merge_asof requires both sides sorted by key
        date_map = date_map.sort_values("end_date")
        tmp = pd.merge_asof(
            left=tmp.sort_values("end_date"),
            right=date_map,
            on="end_date",
            direction="nearest",
            suffixes=("", "_map"),
        )

        # Stock columns are point-in-time balances, so period_type is always "STOCK"
        tmp["period_type"] = "Q"

        # Attach effective fiscal year and effective fiscal quarter label
        tmp["effective_fiscal_year"] = fy_year
        tmp["effective_fiscal_period"] = tmp["quarter"]  # Q1/Q2/Q3/Q4

        out_frames.append(tmp)

    # ------------- 4) Concatenate all fiscal-year slices into one DataFrame -------------
    if not out_frames:
        return pd.DataFrame()

    result = pd.concat(out_frames, ignore_index=True)

    # Optional: reorder columns for readability
    cols_order = [
        "ticker",
        "raw_concept",
        "canonical",
        "end_date",
        "effective_fiscal_year",
        "effective_fiscal_period",
        "period_type",
        "quarter",       # from date_map
        "span",          # target quarter span from date_map
    ]
    # Only keep columns that actually exist
    cols_order = [c for c in cols_order if c in result.columns] + [
        c for c in result.columns if c not in cols_order
    ]
    result = result[cols_order]

    return result

def pick_quarters(df):
    priority_map = {"Q": 0, "YTD_Q": 1, "FY": 2}
    ranks = df["period_type"].map(priority_map)
    idx = ranks.groupby(
[        df["effective_fiscal_year"], df["effective_fiscal_period"]]
    ).idxmin()
    return df.loc[idx].sort_values(["effective_fiscal_year", "effective_fiscal_period"])

def quarterize_per_metric_sec_flows(
    df: pd.DataFrame,
    value_col: str = "val",
    ticker_col: str = "ticker",
    year_col: str = "effective_fiscal_year",
    quarter_col: str = "quarter",              # Int 1–4, or will be created
    period_type_col: str = "period_type",
    metric_cols = ("canonical", "raw_concept") # id for the metric
) -> pd.DataFrame:
    """
    Convert flows reported as Q / YTD_Q / FY into pure quarterly flows.

    Assumes each row is one metric (e.g. Revenues) at a given date,
    with a single numeric column `value_col`.
    """

    df = df.copy()

    if df[quarter_col].str.contains("Q").any():
        df.drop(columns=[quarter_col], inplace=True)
    # Ensure we have a numeric quarter column
    if quarter_col not in df.columns and "effective_fiscal_period" in df.columns:
        df[quarter_col] = (
            df["effective_fiscal_period"]
            .astype(str)
            .str.extract(r"Q(\d)", expand=False)
            .astype("Int64")
        )

    # Normalise period_type priority so that if you have multiple rows
    # for the same (ticker, metric, year, quarter), we pick:
    # Q > YTD_Q > FY.
    priority = {"Q": 0, "YTD_Q": 1, "FY": 2}
    df["_priority"] = df[period_type_col].map(priority).fillna(99)

    group_keys = [ticker_col, *metric_cols, year_col]

    # Choose a single record per quarter according to priority
    idx = (
        df.groupby(group_keys + [quarter_col], dropna=False)["_priority"]
          .idxmin()
          .dropna()
          .astype(int)
    )
    base = df.loc[idx].sort_values(group_keys + [quarter_col])

    def _quarterize_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(quarter_col)
        last_ytd = np.nan          # last YTD value seen
        cum_quarter = np.nan       # sum of quarterised flows so far

        out = []

        for _, row in g.iterrows():
            val = row[value_col]
            if pd.isna(val):
                out.append(np.nan)
                continue

            pt = str(row[period_type_col])
            q  = row[quarter_col]

            # Treat NaN cumulative as 0 for arithmetic
            base_cum = 0.0 if pd.isna(cum_quarter) else float(cum_quarter)

            if pt == "Q":
                q_val = float(val)
                cum_quarter = base_cum + q_val
                last_ytd = np.nan   # reset, we now trust pure quarters

            elif pt == "YTD_Q":
                # Use previous YTD if available, otherwise use sum of prior quarters
                if not pd.isna(last_ytd):
                    base = float(last_ytd)
                else:
                    base = base_cum
                q_val = float(val) - base
                cum_quarter = base_cum + q_val
                last_ytd = float(val)

            elif pt == "FY":
                # Normally quarter 4; if we already know prior quarters,
                # back out Q4 as FY - sum(Q1..Q3). Otherwise just use val.
                if q == 4 and base_cum != 0:
                    q_val = float(val) - base_cum
                else:
                    q_val = float(val)
                cum_quarter = base_cum + q_val

            else:
                # Fallback: treat as a pure quarter
                q_val = float(val)
                cum_quarter = base_cum + q_val

            out.append(q_val)

        g[value_col + "_quarter"] = out
        # All output rows are now quarterly flows
        g[period_type_col] = "Q"
        return g

    result = (
        base.groupby(group_keys, group_keys=False, dropna=False)
            .apply(_quarterize_group)
            .drop(columns="_priority")
    )

    return result

def get_all_metric_components_in_array():
    import json 
    from pathlib import Path
    metric_components_path = Path("../data/metric_components2.json")
    with open(metric_components_path, "r") as f:
        metric_components = json.load(f)

    # Extract all component names from all buckets
    all_components = []
    buckets = metric_components.get("buckets", {})
    for bucket_name, bucket_data in buckets.items():
        components = bucket_data.get("components", {})
        if components:
            # Add all component names from this bucket
            all_components.extend(components.keys())

    # Convert to numpy array (unique components only)
    all_components_array = pd.Series(sorted(set(all_components)))
    return all_components_array

def get_raw_to_canon_from_sec(ticker, analyzer):
    sec_raw = _load_SEC_from_parquet(ticker, analyzer)
    df = pd.concat(sec_raw)
    df["canonical_finnhub_to_canon"] = analyzer._finnhub_field_series_to_canon(df["raw_concept"])
    out = df[["raw_concept", "canonical_finnhub_to_canon", "canonical"]].drop_duplicates()

    # check if it is in metric_components2
    metric_components = get_all_metric_components_in_array()
    out["mapped_in_metric_components"] = np.where(out["canonical_finnhub_to_canon"].isin(metric_components), True, False)
    return out

def set_period_as_index(df):
    period_index = df["effective_fiscal_year"].astype(str) + df["effective_fiscal_period"]
    df = df.set_index(period_index)
    df = df.rename(columns={"effective_fiscal_period": "fiscal_period"})
    print("[RENAMED] effective_fiscal_period -> fiscal_period")
    return df

