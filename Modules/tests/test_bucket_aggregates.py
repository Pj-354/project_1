import sys
import types

import numpy as np
import pandas as pd

requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

from Modules.ttm_ratios_analyzer_backup import TTMRatiosAnalyzer, BUCKET_METRIC_MAP


def test_apply_bucket_aggregates_respects_direct_values():
    analyzer = TTMRatiosAnalyzer(tickers=[], interactive=False, force_update=False)

    analyzer._bucket_components = {
        "revenues": {
            "is_contract_revenue": 1.0,
            "custom_sales": 1.0,
        },
    }

    df = pd.DataFrame(
        {
            "is_contract_revenue": [100.0, np.nan],
            "custom_sales": [20.0, 40.0],
            "is_revenues": [np.nan, 200.0],
        }
    )

    analyzer._apply_bucket_aggregates(df)

    assert df["is_revenues"].iloc[0] == 120.0
    assert df["is_revenues"].iloc[1] == 200.0

def test_usage_tracking_records_sources_for_ebitda():
    analyzer = TTMRatiosAnalyzer(tickers=[], interactive=False, force_update=False)
    analyzer._bucket_components = {
        "operating_income": {
            "custom_operating": 1.0,
        },
    }
    analyzer._init_metric_usage_tracking({"ebitda_ttm"})

    df = pd.DataFrame(
        {
            "custom_operating": [15.0],
            "is_operating_income_loss": [np.nan],
            "is_depreciation_and_amortization": [7.5],
        }
    )

    analyzer._apply_bucket_aggregates(df)
    analyzer._record_direct_metric_usage(df)
    analyzer._finalize_metric_usage()

    assert analyzer.last_metric_usage_summary == {
        "ebitda_ttm": [
            "is_depreciation_and_amortization",
            "is_operating_income_loss",
        ]
    }

def test_normalize_audit_metrics_filters_invalid():
    analyzer = TTMRatiosAnalyzer(tickers=[], interactive=False, force_update=False)
    result = analyzer._normalize_audit_metrics(["ebitda_ttm", "invalid_metric"])
    assert result == {"ebitda_ttm"}

