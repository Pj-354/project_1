import sys
import types

import numpy as np
import pandas as pd

requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer


def _base_analyzer():
    return TTMRatiosAnalyzer(tickers=[], interactive=False, force_update=False)


def test_quarterize_ytd_and_fy_backfill():
    analyzer = _base_analyzer()
    data = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "fiscal_year": 2024,
                "quarter": 1,
                "form": "10-Q",
                "start_date": "2024-01-01",
                "end_date": "2024-03-31",
                "is_revenues": 100,
            },
            {
                "ticker": "AAA",
                "fiscal_year": 2024,
                "quarter": 2,
                "form": "10-Q",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "is_revenues": 250,
            },
            {
                "ticker": "AAA",
                "fiscal_year": 2024,
                "quarter": 4,
                "form": "10-K",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "is_revenues": 520,
            },
        ]
    )
    annotated = analyzer._annotate_period_types(data)
    quarterized = analyzer._quarterize_flows(annotated)
    q = quarterized.sort_values(["fiscal_year", "quarter"])
    q1 = q[(q["quarter"] == 1)].iloc[0]
    q2 = q[(q["quarter"] == 2)].iloc[0]
    q3 = q[(q["quarter"] == 3)].iloc[0]
    q4 = q[(q["quarter"] == 4)].iloc[0]
    assert q1["is_revenues"] == 100
    assert q2["is_revenues"] == 150  # 250 - 100
    assert pd.isna(q3["is_revenues"])
    assert q4["is_revenues"] == 270  # 520 - (100 + 150)


def test_quarterize_mixed_quarter_and_ytd():
    analyzer = _base_analyzer()
    data = pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "fiscal_year": 2025,
                "quarter": 1,
                "form": "10-Q",
                "start_date": "2025-01-02",
                "end_date": "2025-03-31",
                "is_revenues": 100,
            },
            {
                "ticker": "BBB",
                "fiscal_year": 2025,
                "quarter": 2,
                "form": "10-Q",
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
                "is_revenues": 300,
            },
        ]
    )
    annotated = analyzer._annotate_period_types(data)
    qdata = analyzer._quarterize_flows(annotated)
    qdata = qdata.sort_values(["fiscal_year", "quarter"])
    q1 = qdata[qdata["quarter"] == 1].iloc[0]
    q2 = qdata[qdata["quarter"] == 2].iloc[0]
    assert q1["is_revenues"] == 100
    assert q2["is_revenues"] == 200  # 300 - 100


def test_non_calendar_year_ytd_detection():
    analyzer = _base_analyzer()
    data = pd.DataFrame(
        [
            {
                "ticker": "FFF",
                "fiscal_year": 2024,
                "quarter": 1,
                "form": "10-Q",
                "start_date": "2023-07-01",
                "end_date": "2023-09-30",
            },
            {
                "ticker": "FFF",
                "fiscal_year": 2024,
                "quarter": 2,
                "form": "10-Q",
                "start_date": "2023-07-01",
                "end_date": "2023-12-31",
            },
        ]
    )
    annotated = analyzer._annotate_period_types(data)
    q1 = annotated[annotated["quarter"] == 1].iloc[0]
    q2 = annotated[annotated["quarter"] == 2].iloc[0]
    assert q1["period_type"] == "Q"
    assert q2["period_type"] == "YTD_Q"


def test_bucket_and_ttm_after_quarterization():
    analyzer = _base_analyzer()
    periods = [
        ("2023-01-01", "2023-03-31"),
        ("2023-04-01", "2023-06-30"),
        ("2023-07-01", "2023-09-30"),
        ("2023-10-01", "2023-12-31"),
    ]
    values = [100.0, 150.0, 200.0, 250.0]
    rows = []
    for q, (start, end), val in zip([1, 2, 3, 4], periods, values):
        rows.append(
            {
                "ticker": "CCC",
                "fiscal_year": 2023,
                "quarter": q,
                "form": "10-Q",
                "start_date": start,
                "end_date": end,
                "is_revenue_from_contract_with_customer_excl_tax": val,
            }
        )
    data = pd.DataFrame(rows)
    annotated = analyzer._annotate_period_types(data)
    qdata = analyzer._quarterize_flows(annotated)
    canonical_cols = [
        c
        for c in qdata.columns
        if isinstance(c, str) and c.startswith(("is_", "bs_", "cf_", "ci_", "fh_"))
    ]
    analyzer._map_fields_to_buckets(canonical_cols)
    prepared = analyzer._prepare_canonical_financials(qdata)
    final_view = analyzer._build_final_metric_view(prepared)
    flows = analyzer._compute_ttm_flows(final_view)
    last = flows.iloc[-1]
    assert last["is_revenues"] == 250.0
    assert last["is_revenues_ttm"] == 700.0  # 100+150+200+250

