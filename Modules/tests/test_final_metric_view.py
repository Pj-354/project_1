import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer


def _base_analyzer():
    return TTMRatiosAnalyzer(tickers=[], interactive=False, force_update=False)


def _base_rows():
    start = datetime(2023, 1, 1)
    rows = []
    for i in range(4):
        end = start + timedelta(days=89 * (i + 1))
        rows.append(
            {
                "ticker": "ZZZ",
                "start_date": start,
                "end_date": end,
                "filing_date": end + timedelta(days=15),
                "fiscal_year": 2023,
                "fiscal_period": f"Q{i + 1}",
                "quarter": i + 1,
                "period_type": "Q",
                "price": 100 + i,
            }
        )
    return pd.DataFrame(rows)


def test_final_metric_view_populates_sources():
    analyzer = _base_analyzer()
    base = _base_rows().iloc[:2].copy()
    base["is_revenue_from_contract_with_customer_excl_tax"] = [100.0, 120.0]
    base["is_cost_of_revenue"] = [60.0, 80.0]
    base["is_operating_income_loss"] = [20.0, 22.0]
    base["is_net_income_loss"] = [15.0, 17.0]
    base["bs_assets"] = [200.0, 210.0]
    base["bs_equity"] = [120.0, 130.0]
    base["bs_cash_and_cash_equivalents"] = [50.0, 55.0]
    base["bs_current_assets"] = [90.0, 95.0]
    base["bs_current_liabilities"] = [70.0, 75.0]
    base["bs_inventory"] = [20.0, 25.0]
    base["cf_net_cash_flow_from_operating_activities"] = [30.0, 32.0]
    base["capital_expenditures"] = [-10.0, -12.0]

    canonical_cols = [
        c
        for c in base.columns
        if isinstance(c, str) and c.startswith(("is_", "bs_", "cf_", "ci_", "fh_"))
    ]
    analyzer._map_fields_to_buckets(canonical_cols)

    prepared = analyzer._prepare_canonical_financials(base)
    final_view = analyzer._build_final_metric_view(prepared)

    required_cols = {
        "is_revenues",
        "is_cost_of_revenue",
        "is_operating_income_loss",
        "bs_assets",
        "bs_equity",
    }
    assert required_cols.issubset(final_view.columns)

    first_row = final_view.iloc[0]
    assert first_row["is_revenues"] == 100.0
    assert first_row["is_revenues__source"].startswith("bucket")
    assert final_view["bs_assets__source"].notna().all()


def test_final_metric_view_feeds_ttm_columns():
    analyzer = _base_analyzer()
    base = _base_rows()
    base["is_revenues"] = [100.0, 110.0, 120.0, 130.0]
    base["is_cost_of_revenue"] = [60.0, 65.0, 70.0, 75.0]
    base["is_operating_income_loss"] = [20.0, 21.0, 22.0, 23.0]
    base["is_net_income_loss"] = [15.0, 16.0, 17.0, 18.0]
    base["is_depreciation_and_amortization"] = [5.0, 5.5, 6.0, 6.5]
    base["bs_assets"] = [200.0, 205.0, 210.0, 215.0]
    base["bs_current_assets"] = [90.0, 92.0, 94.0, 96.0]
    base["bs_current_liabilities"] = [70.0, 72.0, 74.0, 76.0]
    base["bs_inventory"] = [20.0, 21.0, 22.0, 23.0]
    base["bs_equity"] = [120.0, 122.0, 124.0, 126.0]
    base["bs_cash_and_cash_equivalents"] = [50.0, 51.0, 52.0, 53.0]
    base["cf_net_cash_flow_from_operating_activities"] = [30.0, 31.0, 32.0, 33.0]
    base["cf_net_cash_flow_from_investing_activities"] = [-12.0, -13.0, -14.0, -15.0]
    base["capital_expenditures"] = [-8.0, -8.5, -9.0, -9.5]

    flows = analyzer._compute_ttm_flows(base)
    last = flows.iloc[-1]

    assert np.isclose(last["is_revenues_ttm"], sum([100.0, 110.0, 120.0, 130.0]))
    assert "bs_assets_avg_4q" in flows.columns
    assert "capital_expenditures_ttm" in flows.columns

