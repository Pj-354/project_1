from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import pandas as pd
from IPython.display import display

if TYPE_CHECKING:  # Avoid runtime import cycles
    from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer


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
        .rename(columns={"end_date": "time"})
        [["raw_concept", "canonical", "unit", "val", "fy", "fp", "time", "filing_date", "ticker"]]
        .assign(prefix="bs_")
        .reset_index(drop=True)
    )

    flow_raw_df = (
        base[base["start_date"].notna()]
        .rename(columns={"start_date": "start", "end_date": "end"})
        [["raw_concept", "canonical", "unit", "val", "fy", "fp", "start", "end", "filing_date", "ticker"]]
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
    if "start" and "end" in df.columns:
        return (
            df.pivot(
                index=["start", "end", "filing_date", "fy", "fp", "ticker"],
                columns="canonical",
                values="val",
            )
            .reset_index()
            .rename(columns={"start": "start_date", "end": "end_date", "fy": "fiscal_year", "fp": "fiscal_period"})
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
    display(quarterised_financials[['is_research_and_development', 'is_selling_general_and_administrative_expenses', "is_depreciation_and_amortization", 'is_operating_expense_residual', "is_operating_expenses_ex_da_components", "is_operating_expenses_components_full"]].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials[['is_operating_expenses','is_operating_expense_alt', 'is_operating_income_loss']].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))
    display(quarterised_financials[['is_revenues', 'is_gross_profit', 'is_cost_of_revenue', "is_net_income_loss", "eps_basic","eps_diluted"]].tail(n).T.loc[:, ::-1].sort_values(by=['2025Q3'], ascending=False))