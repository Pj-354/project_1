
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Sequence, Tuple, Dict, Any
from Modules.ttm_ratios_analyzer import TTMRatiosAnalyzer, BUCKET_METRIC_MAP

# ------------------------ #
# Local fast helper clones #
# ------------------------ #

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
              'bs_treasury_stock_value',
              'cf_net_cash_flow_from_operating_activities',
              'cf_capital_expenditures',
              'cf_net_cash_flow_from_financing_activities',
              'cf_net_cash_flow_from_investing_activities',
              'cf_net_cash_flow',
              'cf_effect_of_exchange_rate_on_cash_and_cash_equivalents',
              'is_basic_average_shares',
              'is_diluted_average_shares',
              'is_basic_earnings_per_share',
              'is_diluted_earnings_per_share',
              'bs_noncontrolling_interest_equity',
              'is_noncontrolling_interest_income',
              'cf_noncontrolling_interest_cash_flow',
              'cf_equity_method_cash_flow',
              'is_equity_method_income',
              'bs_equity_method_investments',
              'other_comprehensive_income_loss',
              'bs_crypto_assets',
              'is_crypto_asset_gain_loss']


def classify_period_type_fast(
    df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
    fy_threshold: int = 300,
    ytd_threshold: int = 120,
) -> pd.DataFrame:
    out = df.copy()
    start = pd.to_datetime(out[start_col], errors="coerce")
    end = pd.to_datetime(out[end_col], errors="coerce")
    days = (end - start).dt.days
    out["period_type"] = pd.NA
    out.loc[days >= fy_threshold, "period_type"] = "FY"
    out.loc[(days >= ytd_threshold) & (days < fy_threshold), "period_type"] = "YTD_Q"
    out.loc[days < ytd_threshold, "period_type"] = "Q"
    return out


def classify_fiscal_from_cache_fast(
    df: pd.DataFrame,
    fy_cache,
    ticker_col: str = "ticker",
    start_col: str = "start_date",
    end_col: str = "end_date",
    period_type_col: str = "period_type",
) -> pd.DataFrame:
    out = df.copy()
    out[end_col] = pd.to_datetime(out[end_col])
    out[start_col] = pd.to_datetime(out[start_col])
    if isinstance(fy_cache, dict):
        rows = [
            {ticker_col: t, "effective_fiscal_year": int(fy), "fy_start": pd.to_datetime(dt)}
            for (t, fy), dt in fy_cache.items()
        ]
        fy_table = pd.DataFrame(rows)
    else:
        fy_table = fy_cache.copy()
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

    out_sorted = out.sort_values([ticker_col, end_col]).reset_index(drop=True)
    merged = pd.merge_asof(
        out_sorted,
        fy_table.sort_values([ticker_col, "fy_start"]),
        left_on=end_col,
        right_on="fy_start",
        by=ticker_col,
        direction="backward",
        tolerance=pd.Timedelta(days=500),
    )

    def _infer_quarter(end: pd.Timestamp, fy_start: pd.Timestamp) -> Any:
        if pd.isna(end) or pd.isna(fy_start):
            return pd.NA
        diff_months = (end.year - fy_start.year) * 12 + (end.month - fy_start.month)
        q = diff_months // 3 + 1
        q = 1 if q < 1 else 4 if q > 4 else q
        return f"Q{int(q)}"

    merged["effective_fiscal_period"] = [
        _infer_quarter(e, s) for e, s in zip(merged[end_col], merged["fy_start"])
    ]
    merged = merged.sort_index()
    return merged


def assess_years_and_poly_quarters_fast(
    df: pd.DataFrame,
    metrics: Sequence[str] = CHECK_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    metrics = [c for c in metrics if c in df.columns]
    df["provider"] = df["period_key"].astype("string").str.extract(r"(finn|poly)", expand=False)
    quarters_present = df.groupby("effective_fiscal_year")["fiscal_period"].apply(lambda s: set(s.dropna()))
    has_all_quarters = quarters_present.apply(lambda s: {"Q1", "Q2", "Q3", "Q4"}.issubset(s))
    all_metrics_non_null = (
        df.groupby("effective_fiscal_year")[metrics].apply(lambda g: g.notna().all(axis=1).all())
    )
    provider_count = df.groupby("effective_fiscal_year")["provider"].apply(lambda s: len(s.dropna().unique()))
    status = (has_all_quarters & all_metrics_non_null & (provider_count <= 1)).map({True: "good", False: "bad"})

    issues = []
    issues.append(np.where(has_all_quarters, "", "missing_quarter"))
    issues.append(np.where(all_metrics_non_null, "", "null_metrics"))
    issues.append(np.where(provider_count <= 1, "", "mixed_providers"))
    issues_combined = pd.Series(
        [";".join(filter(None, vals)) for vals in zip(*issues)],
        index=status.index,
    )
    year_quality = pd.DataFrame(
        {"effective_fiscal_year": status.index, "status": status.values, "issues": issues_combined.values}
    )

    df["has_missing_metrics"] = df[metrics].isna().any(axis=1)
    non_poly_with_metrics = df[(~df["provider"].eq("poly")) & df[metrics].notna().any(axis=1)]
    year_has_alt_provider = (
        non_poly_with_metrics.groupby("effective_fiscal_year").size().gt(0).to_dict()
    )
    df["year_has_alt_provider"] = df["effective_fiscal_year"].map(year_has_alt_provider).fillna(False)
    replace_mask = df["provider"].eq("poly") & df["has_missing_metrics"] & df["year_has_alt_provider"]
    quarters_to_replace = (
        df.loc[replace_mask, ["effective_fiscal_year", "fiscal_period"]]
        .drop_duplicates()
        .sort_values(["effective_fiscal_year", "fiscal_period"])
        .reset_index(drop=True)
    )
    quarters_to_replace["fiscal_period"] = quarters_to_replace["fiscal_period"].str.replace("Q0", "Q4")
    return year_quality, quarters_to_replace


def select_preferred_period_rows_fast(
    df: pd.DataFrame,
    keys: pd.DataFrame,
    ticker_col: str = "ticker",
    year_col: str = "effective_fiscal_year",
    period_col: str = "fiscal_period",
    period_type_col: str = "period_type",
) -> pd.DataFrame:
    merged = df.merge(keys[[year_col, period_col]].drop_duplicates(), on=[year_col, period_col], how="inner")
    if merged.empty:
        return merged

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

    group_cols = [ticker_col, year_col, period_col, period_type_col]

    def _collapse_group(g: pd.DataFrame) -> pd.Series:
        if len(g) == 1:
            return g.iloc[0]
        nn = g[metric_cols].notna().sum(axis=1)
        best_idx = nn.idxmax()
        base = g.loc[best_idx].copy()
        others = g.drop(index=best_idx)
        for _, row in others.iterrows():
            mask = base[metric_cols].isna() & row[metric_cols].notna()
            if mask.any():
                base.loc[metric_cols] = base[metric_cols].where(~mask, row[metric_cols])
        return base

    collapsed = (
        merged.groupby(group_cols, as_index=False, group_keys=False)
        .apply(_collapse_group)
        .reset_index(drop=True)
    )
    priority_map = {"Q": 0, "YTD_Q": 1, "FY": 2}
    collapsed["_pt_priority"] = collapsed[period_type_col].map(priority_map).fillna(3).astype(int)
    collapsed = (
        collapsed.sort_values([ticker_col, year_col, period_col, "_pt_priority"])
        .drop_duplicates(subset=[ticker_col, year_col, period_col], keep="first")
        .drop(columns=["_pt_priority"])
    )
    return collapsed


def overwrite_with_selected_fast(
    base_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    keep_unchanged: bool = False,
    ticker_col: str = "ticker",
    year_col: str = "effective_fiscal_year",
    period_col: str = "fiscal_period",
    period_type_col: str = "period_type",
    period_key_col: str = "period_key",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    updated = base_df.copy()
    sel = selected_df.copy()
    for col in (ticker_col, year_col, period_col, period_type_col):
        if col not in sel.columns:
            raise ValueError(f"selected_df is missing required column '{col}'")
    for df_ in (updated, sel):
        for dcol in ("start_date", "end_date"):
            if dcol in df_.columns:
                df_[dcol] = pd.to_datetime(df_[dcol], errors="coerce")

    key_cols = [ticker_col, year_col, period_col]
    base_idxed = updated.set_index(key_cols)
    sel_idxed = sel.set_index(key_cols)

    missing_in_base = [c for c in sel_idxed.columns if c not in base_idxed.columns]
    if missing_in_base:
        base_idxed.loc[:, missing_in_base] = np.nan

    common_index = base_idxed.index.intersection(sel_idxed.index)
    if len(common_index) == 0:
        return base_idxed.reset_index(), pd.DataFrame()

    date_mismatch_flags = pd.DataFrame()
    if {"start_date", "end_date"}.issubset(base_idxed.columns) and {"start_date", "end_date"}.issubset(
        sel_idxed.columns
    ):
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
        date_mismatch_flags = flags.reset_index()

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
    common_cols = [c for c in base_idxed.columns if c in sel_idxed.columns]

    base_sub = base_idxed.loc[common_index, common_cols]
    sel_sub = sel_idxed.loc[common_index, common_cols]
    combined = sel_sub.combine_first(base_sub)
    base_idxed.loc[common_index, common_cols] = combined

    if not keep_unchanged:
        missing_cols = [c for c in base_idxed.columns if c not in sel_idxed.columns]
        for col in missing_cols:
            if col not in meta_cols:
                base_idxed.loc[common_index, col] = np.nan

    updated_reset = base_idxed.reset_index()
    if period_type_col in sel_idxed.columns:
        pt_vals = sel_idxed.loc[common_index, period_type_col]
        pt_df = pt_vals.reset_index()
        updated_reset = updated_reset.merge(pt_df, on=key_cols, how="left", suffixes=("", "_sec"))
        mask_pt = updated_reset[f"{period_type_col}_sec"].notna()
        updated_reset.loc[mask_pt, period_type_col] = updated_reset.loc[mask_pt, f"{period_type_col}_sec"]
        updated_reset = updated_reset.drop(columns=[f"{period_type_col}_sec"])

    if period_key_col in updated_reset.columns:
        common_index_df = pd.MultiIndex.from_tuples(common_index, names=key_cols)
        mask_overwritten = updated_reset.set_index(key_cols).index.isin(common_index_df)
        fp_str = updated_reset.loc[mask_overwritten, period_col].astype(str)
        qnum = fp_str.str.extract(r"(\\d+)", expand=False)
        updated_reset.loc[mask_overwritten, period_key_col] = (
            updated_reset.loc[mask_overwritten, ticker_col].astype(str)
            + "_"
            + updated_reset.loc[mask_overwritten, year_col].astype("Int64").astype(str)
            + "_"
            + qnum
            + "_sec"
        )
    return updated_reset, date_mismatch_flags

def build_quarterised_with_sec_fast(
    ticker: str,
    analyzer: TTMRatiosAnalyzer,
    start_dt: str = "2020-01-01",
    end_dt: str = "2025-12-31",
    return_raw: bool = True,
    keep_unchanged : bool = True
):
    """
    Drop-in replacement for build_quarterised_with_sec (same inputs/outputs/logic)
    with reduced redundant work.
    """
    analyzer._capture_bucket_components = True
    quarterised_financials, _, financials = analyzer._load_financials_merged(
        ticker, start_dt, end_dt, return_raw=True
    )
    fy_cache = getattr(analyzer, "_fy_cache_", None)

    sec_raw = analyzer.fetch_sec_financials(ticker, force_update=True)
    if sec_raw.empty:
        q = analyzer._quarterize_flows(financials)
        if return_raw:
            return analyzer._back_out_periods(q, False, True, fy_cache), financials, fy_cache
        return analyzer._back_out_periods(q, False, True, fy_cache)
    # Fetch 
    start_offset = pd.to_datetime(start_dt) - pd.DateOffset(months=6)
    classified_sec = classify_period_type_fast(sec_raw)
    end_dt_series = pd.to_datetime(classified_sec["end_date"], errors="coerce")
    mask = (end_dt_series >= start_offset) & (end_dt_series <= pd.to_datetime(end_dt))
    filtered_sec = classified_sec.loc[mask].copy()

    _, quarters_to_replace = assess_years_and_poly_quarters_fast(financials)
    if quarters_to_replace.empty:
        return analyzer._build_quarterized_bucket_views(quarterised_financials, ticker)

    tagged_sec = classify_fiscal_from_cache_fast(filtered_sec, fy_cache=fy_cache).sort_values(
        by=["fy_start", "effective_fiscal_period"]
    )
    selected = select_preferred_period_rows_fast(tagged_sec, quarters_to_replace)
    df_overwrite = overwrite_with_selected_fast(financials, selected, keep_unchanged)[0].sort_values(
        by=["effective_fiscal_year", "quarter"]
    )

    analyzer._map_fields_to_buckets(df_overwrite.columns, ticker=ticker)
    canonical = analyzer._prepare_canonical_financials(df_overwrite)
    analyzer._apply_bucket_aggregates(canonical)
    metric_cols = [c for c in BUCKET_METRIC_MAP.values() if c in canonical.columns]
    canonical[metric_cols] = canonical[metric_cols].replace(0.0, pd.NA)

    quarterised = analyzer._quarterize_flows(canonical)
    quarterised = analyzer._fix_share_count_outliers(quarterised)
    backed = analyzer._back_out_periods(quarterised, False, True, fy_cache)

    if return_raw:
        return backed, df_overwrite, fy_cache
    return backed
