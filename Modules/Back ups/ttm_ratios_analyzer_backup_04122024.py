from __future__ import annotations
import requests
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Literal, Iterable, Sequence, Set, ClassVar, Callable
from datetime import datetime, date
from difflib import get_close_matches
from ._fundamental_calculator import METRIC_RULES, compute_metric
from math import isnan
import math
import warnings

# ------------------------------------------------------------------ #
# Name normalisation, mandatory-field contract, completeness checks
# ------------------------------------------------------------------ #

FINNHUB_EXPLICIT_MAP: Dict[str, str] = {
    # ------------------------------------------------------------
    # BALANCE SHEET – core items your pipeline uses
    # ------------------------------------------------------------
    # Assets
    "Assets":                       "assets",
    "AssetsCurrent":                "current_assets",
    "AssetsNoncurrent":             "noncurrent_assets",
    "AssetsHeldForSaleAtCarryingValue": "assets_held_for_sale",
    "AssetsHeldForSaleCurrent":     "assets_held_for_sale_current",

    # Liabilities
   
    "LiabilitiesAndStockholdersEquity": "liabilities_and_equity",
    "LiabilitiesAndStockholdersEquityIncludingPortionAttributableToNoncontrollingInterest":
                                     "liabilities_and_equity_including_noncontrolling_interest",

    # Equity / temporary equity / minority interest
    "StockholdersEquity":           "equity",
    "AdditionalPaidInCapital":      "additional_paid_in_capital",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings_accumulated_deficit",
    "RetainedEarningsAccumulatedDeficitAndAccumulatedOtherComprehensiveIncomeLossNetOfTax":
                                     "retained_earnings_and_aoci",
    "MinorityInterest":             "equity_attributable_to_noncontrolling_interest",
    "OtherMinorityInterests":       "equity_attributable_to_noncontrolling_interest",
    "RedeemableNoncontrollingInterestEquityCarryingAmount":
                                     "redeemable_non_controlling_interest",
    "TemporaryEquityCarryingAmountAttributableToParent":
                                     "temporary_equity_attributable_to_parent",

    # Cash and equivalents / restricted cash
    "CashAndCashEquivalentsAtCarryingValue":    "cash_and_cash_equivalents",
    "CashCashEquivalentsAndShortTermInvestments":
                                     "cash_and_cash_equivalents_short_term_investments",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents":
                                     "cash_and_cash_equivalents_and_restricted",
    "RestrictedCash":               "restricted_cash",
    "RestrictedCashCurrent":        "restricted_cash_current",
    "RestrictedCashNoncurrent":     "restricted_cash_noncurrent",
    "RestrictedCashAndCashEquivalents": "restricted_cash_and_cash_equivalents",
    "RestrictedCashAndCashEquivalentsAtCarryingValue":
                                     "restricted_cash_and_cash_equivalents",
    "RestrictedCashAndCashEquivalentsNoncurrent":
                                     "restricted_cash_and_cash_equivalents_noncurrent",

    # Investments

    "DebtSecuritiesAvailableForSaleAndHeldToMaturity":
                                     "debt_securities",
    "AvailableForSaleSecuritiesCurrent":
                                     "available_for_sale_securities_current",
    "AvailableForSaleSecuritiesNoncurrent":
                                     "available_for_sale_securities_noncurrent",

    # Accounts receivable
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "ReceivablesNetCurrent":        "accounts_receivable",
    "NontradeReceivablesCurrent":   "other_receivables_current",

    # Inventory
    "InventoryNet":                 "inventory",
    "InventoryFinishedGoods":       "inventory_finished_goods",
    "InventoryRawMaterialsAndSupplies": "inventory_raw_materials",
    "InventoryWorkInProcess":       "inventory_work_in_process",
    "InventoryFinishedGoodsAndWorkInProcess":
                                     "inventory_finished_goods_and_wip",

    # Prepaid and other current assets
    "PrepaidExpenseCurrent":        "prepaid_expenses",
    "PrepaidExpenseAndOtherAssetsCurrent":
                                     "prepaid_expenses_and_other_current_assets",
    "PrepaidExpensesAndOtherReceivables":
                                     "prepaid_expenses_and_other_current_assets",
    "PrepaidRevenueShareExpensesAndOtherAssetsCurrent":
                                     "prepaid_expenses_and_other_current_assets",
    "OtherAssetsCurrent":           "other_current_assets",

    # PP&E and intangibles
    "PropertyPlantAndEquipmentGross":    "property_plant_and_equipment_gross",
    "PropertyPlantAndEquipmentNet":      "net_property_plant_and_equipment",
    "PropertyPlantAndEquipmentAndCapitalizedSoftwareNet":
                                     "property_plant_and_equipment_and_capitalized_software",
    "BuildingsAndImprovementsGross": "buildings_and_improvements_gross",
    "MachineryAndEquipmentGross":    "machinery_and_equipment_gross",
    "Goodwill":                      "goodwill",
    "FiniteLivedIntangibleAssetsNet": "intangible_assets",
    "OtherIntangibleAssetsNet":      "intangible_assets_other",

    # Current payables / accrued
    "AccountsPayableCurrent":        "accounts_payable",
    "AccountsPayableAndAccruedLiabilitiesCurrent":
                                     "accounts_payable_and_accrued_liabilities",
    "AccountsPayableAndOtherLiabilities":
                                     "accounts_payable_and_other_liabilities",
    "AccruedLiabilitiesCurrent":     "accrued_liabilities_current",
    "AccruedAndOtherCurrentLiabilities":
                                     "accrued_and_other_current_liabilities",
    "OtherLiabilitiesCurrent":       "other_current_liabilities",
    "AccruedIncomeTaxesCurrent":     "accrued_income_taxes_current",
    "DividendsPayableCurrent":       "dividends_payable",

    # Non-current liabilities
    "OtherLiabilitiesNoncurrent":   "other_noncurrent_liabilities",
    "DeferredIncomeTaxLiabilitiesNet":
                                     "deferred_tax_liabilities",
    "DeferredTaxLiabilitiesNoncurrent":
                                     "deferred_tax_liabilities_noncurrent",
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent":
                                     "pension_and_postretirement_liabilities",

    # Debt (short and long)
    "DebtCurrent":                  "current_debt",
    "ShortTermBorrowings":         "short_term_borrowings",
    "OtherShortTermBorrowings":    "short_term_borrowings",
    "LongTermDebt":                "long_term_debt",

    # Leases
    "FinanceLeaseLiabilityCurrent":  "finance_lease_liability_current",
    "FinanceLeaseLiabilityNoncurrent": "finance_lease_liability_noncurrent",
    "OperatingLeaseLiabilityCurrent": "operating_lease_liability_current",
    "OperatingLeaseLiabilityNoncurrent": "operating_lease_liability_noncurrent",

    # ------------------------------------------------------------
    # INCOME STATEMENT – key items for TTM + Builder
    # ------------------------------------------------------------
    # Revenue
    "Revenues":                    "revenues",
    "SalesRevenueFromContractWithCustomerExclTax":
                                   "revenues_from_contract_with_customer_excl_tax",
    "SalesRevenueFromGoods":
                                   "revenues_from_goods",
    "SalesRevenueFromServices":
                                   "revenues_from_services",
    "SalesRevenueFromServicesAndOther":
                                   "revenues_from_services_and_other",
    # Gross profit
    "GrossProfit":                 "gross_profit",

    # Operating expenses and operating income
    "OperatingExpenses":           "operating_expenses",
    "SellingGeneralAndAdministrativeExpense":
                                   "selling_general_and_administrative_expenses",
    "GeneralAndAdministrativeExpense":
                                   "general_and_administrative",
    "MarketingAndAdvertisingExpense":
                                   "marketing_and_advertising",
    "ResearchAndDevelopmentExpense":
                                   "research_and_development",
    "FulfillmentExpense":          "fulfillment_expense",
    "TechnologyAndInfrastructureExpense":
                                   "technology_and_infrastructure_expense",
    "OperatingIncomeLoss":         "operating_income_loss",
    "NonoperatingIncomeExpense":   "non_operating_income_loss",

    # Net income and components
    "NetIncomeLoss":               "net_income_loss",
    "NetIncomeLossAttributableToNoncontrollingInterest":
                                   "net_income_loss_attributable_to_noncontrolling_interest",
    "NetIncomeLossAvailableToCommonStockholdersBasic":
                                   "net_income_loss_available_to_common_stockholders_basic",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes":
                                   "income_loss_from_continuing_operations_before_tax",
    "IncomeTaxExpenseBenefit":     "income_tax_expense_benefit",
    "IncomeLossFromDiscontinuedOperationsNetOfTax":
                                   "income_loss_from_discontinued_operations_after_tax",

    # Interest + D&A
    "DepreciationDepletionAndAmortization":
                                   "depreciation_depletion_and_amortization",
    "DepreciationAndAmortization": "depreciation_and_amortization",
    "AmortizationOfIntangibleAssets":
                                   "amortization_of_intangible_assets",
    "InterestExpense":             "interest_expense",
    "InterestExpenseOperating":    "interest_expense_operating",
    "InterestExpenseNonoperating": "interest_expense_nonoperating",
    "InterestIncomeExpenseNet":    "net_interest_expense",

    # EPS and share counts
    "EarningsPerShareBasic":       "basic_earnings_per_share",
    "EarningsPerShareDiluted":     "diluted_earnings_per_share",
    "EarningsPerShareBasicAndDiluted":
                                   "basic_earnings_per_share",  # fallback
    "WeightedAverageNumberOfSharesOutstandingBasic":
                                   "basic_average_shares",
    "WeightedAverageNumberOfDilutedSharesOutstanding":
                                   "diluted_average_shares",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted":
                                   "basic_average_shares",  # fallback

    # Provision, loan losses (for banks)
    "ProvisionForLoanLeaseAndOtherLosses":
                                   "provision_for_loan_lease_and_other_losses",

    # ------------------------------------------------------------
    # CASH FLOW STATEMENT – key items you actually use
    # ------------------------------------------------------------
    "NetCashProvidedByUsedInOperatingActivities":
                                   "net_cash_flow_from_operating_activities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations":
                                   "net_cash_flow_from_operating_activities_continuing",
    "NetCashProvidedByUsedInInvestingActivities":
                                   "net_cash_flow_from_investing_activities",
    "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations":
                                   "net_cash_flow_from_investing_activities_continuing",
    "NetCashProvidedByUsedInFinancingActivities":
                                   "net_cash_flow_from_financing_activities",
    "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations":
                                   "net_cash_flow_from_financing_activities_continuing",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect":
                                   "net_cash_flow",
    # Capex proxies
    "ProceedsFromSaleOfPropertyPlantAndEquipment":
                                   "proceeds_from_sale_of_pp&e",

}

STATEMENT_PREFIXES: Tuple[str, ...] = ("is_", "bs_", "cf_", "ci_", "fh_")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC_BUCKETS_PATH = PROJECT_ROOT / "data/metric_buckets.json"
DEFAULT_METRIC_COMPONENTS_PATH = PROJECT_ROOT / "data/metric_components2.json"

FLOW_PREFIXES_FOR_BUCKETS = ("is_", "cf_", "ci_")
STOCK_PREFIXES_FOR_BUCKETS = ("bs_",)
STOCK_METRIC_OVERRIDES = {"is_basic_average_shares", "is_diluted_average_shares"}
STOCK_BUCKET_OVERRIDES = {"short_term_debt", "long_term_debt", "treasury_stock", "crypto_assets"}


def _strip_bucket_statement_prefix(name: str) -> str:
    if not isinstance(name, str):
        return name
    for prefix in STOCK_PREFIXES_FOR_BUCKETS + FLOW_PREFIXES_FOR_BUCKETS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _build_bucket_classification(
    targets: Dict[str, Optional[str]],
    components: Dict[str, Dict[str, Dict[str, float]]],
) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
]:
    flow_dict: Dict[str, Dict[str, List[str]]] = {}
    stock_dict: Dict[str, Dict[str, List[str]]] = {}

    for bucket, target in targets.items():
        if not target:
            continue
        canonical = target.strip()
        base_name = _strip_bucket_statement_prefix(canonical)
        payload = {
            "metric": base_name,
            "components": list((components.get(bucket) or {}).keys()),
        }

        lower = canonical.lower()
        if (
            bucket in STOCK_BUCKET_OVERRIDES
            or lower in STOCK_METRIC_OVERRIDES
            or canonical.startswith(STOCK_PREFIXES_FOR_BUCKETS)
        ):
            stock_dict[bucket] = payload
        else:
            flow_dict[bucket] = payload

    return flow_dict, stock_dict


def _flatten_metric_dict(d: Dict[str, Dict[str, Any]]) -> List[str]:
    flat: Set[str] = set()
    for bucket_name, cfg in d.items():
        flat.add(bucket_name)
        metric = cfg.get("metric")
        if isinstance(metric, str):
            flat.add(metric)
        components = cfg.get("components", [])
        for comp in components:
            if isinstance(comp, str):
                flat.add(comp)
    return sorted(flat)

# ------------------------------------------------------------------ #
# TTM flow utilities
# ------------------------------------------------------------------ #

FLOW_PREFIXES: Tuple[str, ...] = ("is_", "cf_")
STOCK_PREFIXES: Tuple[str, ...] = ("bs_",)
META_COLUMNS: Set[str] = {
    "ticker",
    "start_date",
    "end_date",
    "period_key",
    "px_close",
    "close",
    "volume",
}


def _is_flow_column(col: str) -> bool:
    """Return True if column is a flow (income or cash-flow) and not meta."""
    if col in META_COLUMNS:
        return False
    return isinstance(col, str) and col.startswith(FLOW_PREFIXES)


def get_flow_columns(df: pd.DataFrame) -> List[str]:
    """Infer flow columns from naming conventions."""
    return [c for c in df.columns if _is_flow_column(c)]


def compute_ttm_flows(
    df: pd.DataFrame,
    entity_col: str = "ticker",
    date_col: str = "end_date",
    flow_cols: Optional[Sequence[str]] = None,
    window: int = 4,
    min_periods: int = 4,
    inplace: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compute TTM (rolling window) sums for flow columns in a quarterised panel.

    - Detects flow columns by naming (prefix is_/cf_) if `flow_cols` is None.
    - Groups by `entity_col`, sorted by `date_col`, and applies a rolling sum.
    - Adds `<col>_ttm` columns; skips existing unless overwrite=True.
    """
    if entity_col not in df.columns or date_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{entity_col}' and '{date_col}' columns.")

    work = df if inplace else df.copy()
    orig_index = work.index

    candidate_flows = list(flow_cols) if flow_cols is not None else get_flow_columns(work)
    # Drop missing/non-existent
    candidate_flows = [c for c in candidate_flows if c in work.columns]
    if not candidate_flows:
        return work

    # Keep only numeric flow columns
    numeric_flows: List[str] = []
    for c in candidate_flows:
        if pd.api.types.is_numeric_dtype(work[c]):
            numeric_flows.append(c)
        else:
            try:
                work[c] = pd.to_numeric(work[c], errors="coerce")
                numeric_flows.append(c)
            except Exception:
                continue
    if not numeric_flows:
        return work

    # Sort for rolling computation
    work_sorted = work.sort_values([entity_col, date_col])

    new_cols: List[str] = []
    for col in numeric_flows:
        ttm_col = f"{col}_ttm"
        if (ttm_col in work_sorted.columns) and not overwrite:
            continue
        ttm_series = (
            work_sorted.groupby(entity_col, group_keys=False)[col]
            .rolling(window=window, min_periods=min_periods)
            .sum()
        )
        # rolling returns aligned with sorted index
        work_sorted[ttm_col] = ttm_series.to_numpy()
        new_cols.append(ttm_col)

    # Restore original order
    work_sorted = work_sorted.loc[orig_index]
    if inplace:
        for c in new_cols:
            work[c] = work_sorted[c]
        return work
    return work_sorted


def _load_bucket_sources() -> Tuple[
    Dict[str, Optional[str]],
    Dict[str, Dict[str, Dict[str, float]]],
]:
    try:
        bucket_data = json.loads(DEFAULT_METRIC_BUCKETS_PATH.read_text())
        component_data = json.loads(DEFAULT_METRIC_COMPONENTS_PATH.read_text())
    except Exception:
        return {}, {}

    bucket_targets = {
        bucket: (meta or {}).get("metric")
        for bucket, meta in (bucket_data.get("buckets") or {}).items()
    }
    component_buckets = component_data.get("buckets") or {}
    bucket_components = {
        bucket: (component_buckets.get(bucket, {}).get("components") or {})
        for bucket in bucket_targets
    }
    return bucket_targets, bucket_components


_bucket_targets: Dict[str, Optional[str]]
_bucket_components: Dict[str, Dict[str, Dict[str, float]]]
try:
    _bucket_targets, _bucket_components = _load_bucket_sources()
except Exception:
    _bucket_targets, _bucket_components = {}, {}

_FLOW_COLS: List[str] = []
_STOCK_COLS: List[str] = []
try:
    _flow_dict, _stock_dict = _build_bucket_classification(_bucket_targets, _bucket_components)
    _FLOW_COLS = _flatten_metric_dict(_flow_dict)
    _STOCK_COLS = _flatten_metric_dict(_stock_dict)
except Exception:
    _FLOW_COLS = []
    _STOCK_COLS = []


_dynamic_bucket_map: Dict[str, str] = {
    bucket: metric for bucket, metric in (_bucket_targets or {}).items() if isinstance(metric, str) and metric
}

if not _dynamic_bucket_map:
    warnings.warn(
        "metric_buckets.json did not yield any bucket mappings; falling back to built-in defaults.",
        RuntimeWarning,
    )
    _dynamic_bucket_map = {
        "revenues": "is_revenues",
        "cost_of_revenue": "is_cost_of_revenue",
        "gross_profit": "is_gross_profit",
        "operating_expenses": "is_operating_expenses",
        "operating_income": "is_operating_income_loss",
        "depreciation_and_amortization": "is_depreciation_and_amortization",
        "net_interest_expense": "is_net_interest_expense",
        "income_tax_expense": "is_income_tax_expense_benefit",
        "net_income": "is_net_income_loss",
        "total_assets": "bs_assets",
        "book_equity": "bs_equity",
        "cash_and_equivalents": "bs_cash_and_cash_equivalents_short_term_investments",
        "treasury_stock": "bs_treasury_stock_value",
        "operating_cash_flow": "cf_net_cash_flow_from_operating_activities",
        "capital_expenditures": "cf_capital_expenditures",
        "basic_average_shares": "is_basic_average_shares",
        "diluted_average_shares": "is_diluted_average_shares",
        "basic_earnings_per_share": "is_basic_earnings_per_share",
        "diluted_earnings_per_share": "is_diluted_earnings_per_share",
        "noncontrolling_interest_equity": "bs_noncontrolling_interest_equity",
        "noncontrolling_interest_income": "is_noncontrolling_interest_income",
        "noncontrolling_interest_cash_flow": "cf_noncontrolling_interest_cash_flow",
        "equity_method_cash_flow": "cf_equity_method_cash_flow",
        "equity_method_income": "is_equity_method_income",
        "equity_method_investments": "bs_equity_method_investments",
        "oci": "other_comprehensive_income_loss",
        "crypto_assets": "bs_crypto_assets",
        "crypto_pnl": "is_crypto_asset_gain_loss",
    }

BUCKET_METRIC_MAP: Dict[str, str] = dict(_dynamic_bucket_map)

METRIC_TO_BUCKET_MAP: Dict[str, str] = {
    metric: bucket for bucket, metric in BUCKET_METRIC_MAP.items()
}

API_KEY = 'tt2gOLH0fHAmPX70a4QURLFy59PRCZr3'
FINN_API_KEY = 'd4c967hr01qudf6h82c0d4c967hr01qudf6h82cg'

@dataclass
class TTMRatiosAnalyzer:
    """
    TTM (Trailing Twelve Months) financial ratios analyzer with aliasing and optional interactivity.
    """
    api_key: str = API_KEY
    finn_api_key: str = FINN_API_KEY
    tickers: List[str] = field(default_factory=list)
    fundamentals_dir: Optional[str] = "/Users/phillip/Desktop/Moon2/data/polygon_fundamentals"
    finnhub_fundamentals_dir: Optional[str] = "/Users/phillip/Desktop/Moon2/data/finnhub_fundamentals"
    sec_cik_cache_path: str = "/Users/phillip/Desktop/Moon2/data/sec_cik_map.json"
    sec_user_agent: str = "Moon2/1.0 (contact: research@moon2.local)"
    ratios_dir: str = "/Users/phillip/Desktop/Moon2/data/ttm_ratios"
    quarterised_fundamentals_dir: str = "/Users/phillip/Desktop/Moon2/data/quarterised_fundamentals"
    local_price_dir: Optional[str] = "/Users/phillip/Desktop/Moon2/data/daily2/stocks"
    min_end_date: Optional[str] = None
    max_end_date: Optional[str] = None
    verbose: bool = False
    force_update: bool = True
    debug: bool = False
    register_unrecognized_fields: bool = False
    base_url: str = "https://api.polygon.io"
    poly_alias_path: str = "/Users/phillip/Desktop/Moon2/data/merged_alias.json"
    finn_alias_path: str = "/Users/phillip/Desktop/Moon2/data/merged_alias.json"
    metric_buckets_path: str = "/Users/phillip/Desktop/Moon2/data/metric_buckets.json"
    metric_components_path: str = "/Users/phillip/Desktop/Moon2/data/metric_components2.json"
    interactive: bool = False
    default_financials_provider: Literal["polygon", "finnhub"] = "polygon"
    _MANDATORY_BASE_FIELDS: Dict[str, str] = field(
    default_factory=lambda: {
        # income statement
        "is_revenues": "income",
        "is_net_income_loss": "income",
        "is_operating_income_loss": "income",
        "is_depreciation_and_amortization": "income",
        "is_basic_earnings_per_share": "income",
        "is_diluted_earnings_per_share": "income",
        "is_basic_average_shares": "income",
        "is_diluted_average_shares": "income",
        "is_selling_general_and_administrative_expenses": "income",
        "is_gross_profit": "income",
        "is_research_and_development": "income",
        "is_cost_of_revenue": "income",
        "is_income_loss_before_equity_method_investments": "income",
        "is_net_income_loss_available_to_common_stockholders_basic": "income",
        "is_interest_expense_operating": "income",
        "is_income_loss_from_equity_method_investments": "income",
        # balance sheet
        "bs_equity": "balance",
        "bs_equity_attributable_to_parent": "balance",
        "bs_cash": "balance",
        "bs_long_term_debt": "balance",
        "bs_assets": "balance",
        "bs_current_assets": "balance",
        "bs_current_liabilities": "balance",
        "bs_inventory": "balance",
        # cash flow
        "cf_net_cash_flow_from_investing_activities": "cashflow"})
    _alias_map: Dict[str, List[str]] = field(default_factory=dict)
    _reverse_map: Dict[str, str] = field(default_factory=dict)
    _session_cache: Dict[str, str] = field(default_factory=dict)
    _current_provider: str = field(default="poly", init=False)
    unrecognised_fields: Dict[str, str] = field(default_factory=dict)
    _bucket_components: Optional[Dict[str, Dict[str, float]]] = field(default=None, init=False, repr=False)
    calc_view: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _capture_bucket_components: bool = field(default=False, init=False, repr=False)
    _bucket_component_store: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _fiscal_year_start_cache: Dict[Tuple[str, int], pd.Timestamp] = field(
        default_factory=dict, init=False, repr=False
    )
    _fy_cache_: Optional[Dict[Tuple[str, int], pd.Timestamp]] = field(
        default=None, init=False, repr=False
    )
    _reverse_map_cache: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)
    _alias_map_cache: Dict[str, Dict[str, List[str]]] = field(default_factory=dict, init=False, repr=False)
    _loaded_financials: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _quarterized_financials: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _filled_with_annuals: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # --------------------------------------------------------------------- #
    # Init + alias management
    # --------------------------------------------------------------------- #

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key is required")

        if self.fundamentals_dir:
            Path(self.fundamentals_dir).mkdir(parents=True, exist_ok=True)
        if self.finnhub_fundamentals_dir:
            Path(self.finnhub_fundamentals_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ratios_dir).mkdir(parents=True, exist_ok=True)
        Path(self.quarterised_fundamentals_dir).mkdir(parents=True, exist_ok=True)

        self._load_aliases()

    def _log(self, msg: str, level: str = "info"):
        level_norm = (level or "").lower()
        if not self.verbose and level_norm not in {"warning", "error"}:
            return
        print(msg)

    def _load_aliases(self, provider: str = "poly"):
        """
        Load aliases for specified provider.
        
        Parameters:
        -----------
        provider : str
            Either "poly" (or "polygon") for Polygon aliases,
            or "finn" (or "finnhub") for Finnhub aliases
        """
        # Normalize provider name
        provider = provider.lower()
        if provider in ["finn", "finnhub"]:
            alias_path = Path(self.finn_alias_path)
        elif provider in ["poly", "polygon"]:
            alias_path = Path(self.poly_alias_path)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'poly' or 'finn'")
        
        if alias_path.exists():
            with open(alias_path, "r") as f:
                data = json.load(f)
            self._alias_map = data.get("aliases", {})
            self._current_provider = provider
            self._alias_map_cache[provider] = self._alias_map
            self._log(
                f"Loaded {len(self._alias_map)} canonical fields from {alias_path} ({provider})",
                level="debug",
            )
        else:
            print('Initialising with default alias map')
            # No file exists, initialize with defaults (for Polygon)
            self._current_provider = provider
            # Canonical names are WITHOUT prefix; prefixes are "is_", "bs_", "cf_", "ci_"
            self._alias_map = {
                # Income statement
                "revenues": ["revenues", "revenue", "total_revenue", "total_revenues", "sales"],
                "net_income_loss": ["net_income_loss", "net_income", "net_loss", "net_earnings"],
                "operating_income_loss": ["operating_income_loss", "operating_income", "operating_loss", "ebit"],
                "depreciation_and_amortization": [
                    "depreciation_and_amortization",
                    "depreciation_amortization",
                    "depreciation_and_amortization_expense",
                ],
                "basic_earnings_per_share": [
                    "basic_earnings_per_share",
                    "eps_basic",
                    "basic_eps",
                    "earnings_per_share_basic",
                ],
                "diluted_earnings_per_share": [
                    "diluted_earnings_per_share",
                    "eps_diluted",
                    "diluted_eps",
                    "earnings_per_share_diluted",
                ],
                "basic_average_shares": [
                    "basic_average_shares",
                    "weighted_average_shares_basic",
                    "weighted_average_shares_outstanding_basic",
                ],
                "diluted_average_shares": [
                    "diluted_average_shares",
                    "weighted_average_shares_diluted",
                    "weighted_average_shares_outstanding_diluted",
                ],
                "selling_general_and_administrative_expenses": [
                    "selling_general_and_administrative_expenses",
                    "selling_general_and_administrative",
                    "sga_expense",
                    "sg&a_expenses",
                ],
                "gross_profit": ["gross_profit", "gross_income"],
                "research_and_development": [
                    "research_and_development",
                    "research_and_development_expense",
                    "r&d_expense",
                    "rnd_expense",
                ],
                "income_loss_before_equity_method_investments": [
                    "income_loss_before_equity_method_investments",
                    "income_before_equity_method_investments",
                ],
                "net_income_loss_available_to_common_stockholders_basic": [
                    "net_income_loss_available_to_common_stockholders_basic",
                    "net_income_available_to_common_stockholders",
                ],
                "income_loss_from_equity_method_investments": [
                    "income_loss_from_equity_method_investments",
                    "equity_method_investment_income",
                ],
                # Balance sheet
                "equity": [
                    "equity",
                    "total_equity",
                    "stockholders_equity",
                    "shareholders_equity",
                    "total_stockholders_equity",
                ],
                "equity_attributable_to_parent": [
                    "equity_attributable_to_parent",
                    "equity_attributable_to_shareholders",
                    "stockholders_equity_attributable_to_parent",
                ],
                "assets": ["assets", "total_assets"],
                "current_assets": ["current_assets", "total_current_assets"],
                "current_liabilities": ["current_liabilities", "total_current_liabilities"],
                "inventory": ["inventory", "inventories", "total_inventory"],
                # Cash flow
                "net_cash_flow_from_investing_activities": [
                    "net_cash_flow_from_investing_activities",
                    "cash_flow_from_investing_activities",
                    "investing_cash_flow",
                    "net_cash_flow_investing_activities",
                ],
            }
            self._alias_map_cache[provider] = self._alias_map
            self._save_aliases()

        self._build_reverse_map()

    def _save_aliases(self, provider: Optional[str] = None):
        """
        Save aliases to the appropriate JSON file based on provider.
        
        Parameters:
        -----------
        provider : str, optional
            Either "poly" or "finn". If None, uses self._current_provider
        """
        if provider is None:
            provider = self._current_provider
        
        # Normalize provider name
        provider = provider.lower()
        if provider in ["finn", "finnhub"]:
            alias_path = Path(self.finn_alias_path)
        elif provider in ["poly", "polygon"]:
            alias_path = Path(self.poly_alias_path)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'poly' or 'finn'")
        
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "provider": provider,
            "aliases": self._alias_map,
            "last_updated": datetime.now().isoformat(),
            "total_canonical_fields": len(self._alias_map),
            "total_variants": sum(len(v) for v in self._alias_map.values()),
        }
        with open(alias_path, "w") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"Saved alias map to {alias_path} ({provider})")

    def _build_reverse_map(self):
        """
        Build normalized variant -> canonical lookup for fast matching
        """
        self._reverse_map = {}
        for canonical, variants in self._alias_map.items():
            norm_canon = self._normalize_name(canonical)
            self._reverse_map[norm_canon] = canonical
            for variant in variants:
                norm_var = self._normalize_name(variant)
                self._reverse_map[norm_var] = canonical
        if hasattr(self, "_alias_map_cache"):
            self._reverse_map_cache[self._current_provider] = dict(self._reverse_map)

    def _read_alias_file(self, provider: str) -> Dict[str, Any]:
        provider = provider.lower()
        if provider in ("poly", "polygon"):
            path = Path(self.poly_alias_path)
        elif provider in ("finn", "finnhub"):
            path = Path(self.finn_alias_path)
        else:
            raise ValueError(f"Unknown provider '{provider}'")
        if not path.exists():
            return {}
        with open(path, "r") as handle:
            return json.load(handle)

    def compare_canonical_aliases(self) -> Dict[str, Any]:
        """
        Compare canonical field coverage between Polygon and Finnhub alias maps.
        Returns a dict with intersections and provider-only sets.
        """
        poly_aliases = self._read_alias_file("polygon").get("aliases", {})
        finn_aliases = self._read_alias_file("finnhub").get("aliases", {})
        poly_set = set(poly_aliases.keys())
        finn_set = set(finn_aliases.keys())
        overlap = sorted(poly_set & finn_set)
        only_poly = sorted(poly_set - finn_set)
        only_finn = sorted(finn_set - poly_set)
        return {
            "intersection": overlap,
            "polygon_only": only_poly,
            "finnhub_only": only_finn,
            "polygon_count": len(poly_set),
            "finnhub_count": len(finn_set),
        }

    # --------------------------------------------------------------------- #
    # Alias resolution
    # --------------------------------------------------------------------- #

    _TICKER_PREFIX_OVERRIDES: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "GOOGL": ("GOOGL", "GOOG"),
        "GOOG": ("GOOG", "GOOGL"),
    }

    def _canonical_field(
        self,
        raw_key: str,
        section_prefix: str,
        ticker: Optional[str] = None,
    ) -> str:
        """
        Map raw field name to canonical base name.
        Pipeline:
          1) normalize
          2) session cache
          3) exact alias/reverse_map
          4) high-confidence fuzzy match
          5) interactive prompt (optional)
        """
        raw_value = raw_key if isinstance(raw_key, str) else str(raw_key)
        raw_value = raw_value.strip()

        provider = (getattr(self, "_current_provider", "") or "").lower()
        working_key = raw_value

        if ticker:
            working_key = self._strip_us_gaap_tag(working_key, ticker=ticker)

        stripped = self._strip_statement_prefix(working_key)
        if stripped:
            working_key = stripped

        norm_raw = self._normalize_name(working_key)

        # 1. session cache
        if norm_raw in self._session_cache:
            return self._session_cache[norm_raw]

        # 2. direct reverse_map
        if norm_raw in self._reverse_map:
            canon = self._reverse_map[norm_raw]
            self._session_cache[norm_raw] = canon
            return canon

        # 3. fuzzy auto-match
        canon_fuzzy = self._auto_fuzzy_match(norm_raw, threshold=0.99)
        if canon_fuzzy is not None:
            self._session_cache[norm_raw] = canon_fuzzy
            if self.verbose:
                print(f"Auto-mapped '{raw_key}' → '{canon_fuzzy}' via fuzzy match")
            return canon_fuzzy

        # 4. non-interactive mode: keep as-is
        if not self.interactive:
            self._session_cache[norm_raw] = raw_value
            if self.verbose:
                print(f"Unknown field '{raw_value}' kept as-is (non-interactive mode).")
            return raw_value

        # 5. interactive classification (simplified)
        print("\n" + "=" * 70)
        print("UNKNOWN FIELD DETECTED")
        print("=" * 70)
        print(f"Section prefix: {section_prefix}")
        print(f"Raw field name: '{raw_value}'")
        print("=" * 70)
        print("Options:")
        print("  - Type an EXISTING canonical name to map to it")
        print("  - Type a NEW canonical name to create")
        print("  - Type 'list' to see all canonical names")
        print("  - Type 'skip' to keep as raw name")
        print("  - Type 'stop' to disable interactive mode")
        print("=" * 70)

        while True:
            resp = input("Your choice: ").strip()
            low = resp.lower()

            if low == "stop":
                self.interactive = False
                self._session_cache[norm_raw] = raw_value
                print("Interactive mode disabled; keeping unknown fields as-is.")
                return raw_value

            if low == "skip":
                self._session_cache[norm_raw] = raw_value
                print(f"Keeping field as '{raw_value}'")
                return raw_value

            if low == "list":
                print("\nExisting canonical fields:")
                for c in sorted(self._alias_map.keys()):
                    variants = self._alias_map[c]
                    ex = variants[0] if variants else c
                    more = f" (+{len(variants)-1})" if len(variants) > 1 else ""
                    print(f"  • {c}  e.g. '{ex}'{more}")
                print()
                continue

            if resp:
                canonical_input = resp.strip()
                canonical = self._normalize_canonical_name(canonical_input)
                # existing canonical (accept exact or normalized key)
                if canonical in self._alias_map:
                    if working_key not in self._alias_map[canonical]:
                        self._alias_map[canonical].append(working_key)
                        self._save_aliases()
                    self._reverse_map[self._normalize_name(working_key)] = canonical
                    self._session_cache[norm_raw] = canonical
                    print(f"Mapped '{raw_value}' → '{canonical}'")
                    return canonical
                if canonical_input in self._alias_map and canonical_input != canonical:
                    canonical = canonical_input
                    if working_key not in self._alias_map[canonical]:
                        self._alias_map[canonical].append(working_key)
                        self._save_aliases()
                    self._reverse_map[self._normalize_name(working_key)] = canonical
                    self._session_cache[norm_raw] = canonical
                    print(f"Mapped '{raw_value}' → '{canonical}'")
                    return canonical
                # new canonical
                else:
                    canonical_norm = canonical
                    self._alias_map[canonical_norm] = [working_key]
                    self._reverse_map[self._normalize_name(working_key)] = canonical_norm
                    self._reverse_map[self._normalize_name(canonical_norm)] = canonical_norm
                    self._session_cache[norm_raw] = canonical_norm
                    self._save_aliases()
                    print(f"Created canonical '{canonical_norm}' for '{raw_value}'")
                    return canonical_norm

            print("Enter canonical name, 'list', 'skip', or 'stop'.")
    def _flatten_section(
        self,
        sec: Optional[Dict[str, Any]],
        prefix: str,
        ticker: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Flatten one section dict -> prefix+canonical_name: float.

        Uses Polygon's 'label' field (if present) as the raw name passed
        into the canonical alias machinery, rather than the JSON key.
        """
        out: Dict[str, float] = {}
        if not isinstance(sec, dict):
            return out

        # Optional debug structure; remove if you don't need it
        raw_sources: Dict[str, List[Tuple[str, float]]] = {}

        for k, v in sec.items():
            # v is usually a dict like:
            #   {'value': 18373000.0, 'unit': 'shares', 'label': 'Basic Average Shares', ...}
            if isinstance(v, dict):
                val = v.get("value")
                raw_label = v.get("label") or k  # fall back to key if label missing
            else:
                val = v
                raw_label = k

            if isinstance(val, (int, float)):
                # Use the human-readable label as the input to the alias system
                canon = self._canonical_field(raw_label, prefix, ticker=ticker)
                field_name = f"{prefix}{canon}"

                # Debug: track which JSON keys and labels produced this field
                if field_name not in raw_sources:
                    raw_sources[field_name] = []
                raw_sources[field_name].append((f"{k} | {raw_label}", float(val)))

                if field_name in out and abs(out[field_name] - float(val)) > 0.01:
                    if self.verbose:
                        self._log(
                            f"Warning: duplicate field '{field_name}' "
                            f"with differing values. "
                            f"Existing={out[field_name]} new={float(val)}"
                        )
                        self.raw_sources = raw_sources
                else:
                    out[field_name] = float(val)
        
        return out
    def _normalize_name(self, raw: str) -> str:
        """Normalize a raw field name for matching."""
        s = raw.lower()
        for ch in [",", "(", ")", "/", "-", " ", "."]:
            s = s.replace(ch, "_")
        while "__" in s:
            s = s.replace("__", "_")
        s = s.strip("_")

        # simple token synonym mapping
        synonym_map = {
            "earnings": "income",
            "expenses": "expense",
            "expense": "expense",
            "sales": "revenue",
            "revenues": "revenue",
            "rev": "revenue",
            "int": "interest",
            "net": "net",
        }
        tokens = s.split("_")
        norm_tokens = [synonym_map.get(tok, tok) for tok in tokens]
        return "_".join(norm_tokens)
    def _auto_fuzzy_match(self, norm_key: str, threshold: float = 0.999) -> Optional[str]:
        """High-confidence fuzzy match from normalized key to canonical name."""
        if not self._reverse_map:
            return None
        candidates = list(self._reverse_map.keys())
        matches = get_close_matches(norm_key, candidates, n=1, cutoff=threshold)
        if not matches:
            return None
        best = matches[0]
        return self._reverse_map.get(best)
    

    
    def _register_unrecognized_field(self, canonical: str, raw_label: str) -> None:
        """Add a newly discovered Finnhub field to aliases + tracking dict."""
        if not self.register_unrecognized_fields:
            return
        canonical = self._normalize_canonical_name(canonical.strip("_"))
        print('[REGISTERING UNRECOGNIZED FIELD] Creating alias for:', canonical, '<--', raw_label)
        if not canonical or not raw_label:
            return

        variants = self._alias_map.setdefault(canonical, [])
        if raw_label not in variants:
            variants.append(raw_label)
            # Persist to Finnhub alias file and refresh reverse map
            self._save_aliases(provider="finn")
            self._build_reverse_map()
        self._ensure_component_tracking(canonical)

    @staticmethod
    def _strip_statement_prefix(field: str) -> str:
        for prefix in ("is_", "bs_", "cf_", "ci_"):
            if field.startswith(prefix):
                return field[len(prefix):]
        return field

    @staticmethod
    def _to_snake_case(name: str) -> str:
        if not isinstance(name, str):
            return name
        tokens = re.split(r"[^A-Za-z0-9]+", name)
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return name.lower()
        words: List[str] = []
        for tok in tokens:
            pieces = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z0-9]+",
                tok,
            )
            words.extend(pieces or [tok])
        return "_".join(word.lower() for word in words)

    @classmethod
    def _normalize_component_name(cls, field: str) -> str:
        stripped = cls._strip_statement_prefix(field or "")
        return cls._to_snake_case(stripped)

    @classmethod
    def _normalize_canonical_name(cls, name: str) -> str:
        """
        Ensure new canonical names are snake_case with underscores between words.
        Used when adding unknown fields into merged_alias.
        """
        return cls._to_snake_case(name or "")

    def _canonical_to_prefixed_metric(self, canonical: str) -> str:
        """
        Map a base canonical name to a prefixed metric (is_/bs_/cf_/ci_) when possible.
        Falls back to the input if no mapping is found.
        """
        if not isinstance(canonical, str):
            return canonical
        if canonical.startswith(STATEMENT_PREFIXES):
            return canonical

        normalized = self._normalize_component_name(canonical)

        # Direct match against known prefixed metrics
        for metric_name in METRIC_TO_BUCKET_MAP.keys():
            if self._normalize_component_name(metric_name) == normalized:
                return metric_name

        # Component lookup -> bucket -> metric
        try:
            component_lookup = self._get_component_lookup()
            entry = component_lookup.get(normalized)
            if entry:
                bucket = entry.get("bucket")
                metric = BUCKET_METRIC_MAP.get(bucket)
                if metric:
                    return metric
        except Exception:
            pass

        return canonical

    @classmethod
    def _ticker_prefix_variants(cls, ticker: Optional[str]) -> List[str]:
        if not ticker:
            return []
        base = str(ticker).upper()
        variants = list(cls._TICKER_PREFIX_OVERRIDES.get(base, (base,)))
        ordered: List[str] = []
        seen: Set[str] = set()
        for variant in variants:
            upper_variant = variant.upper()
            if upper_variant not in seen:
                seen.add(upper_variant)
                ordered.append(upper_variant)
        return ordered

    @classmethod
    def _strip_us_gaap_tag(cls, concept: Optional[str], ticker: Optional[str] = None) -> str:
        if not isinstance(concept, str):
            return ""
        # Handle colon format (e.g., "ADI:NonCashOperatingLeaseCosts" or "ADI:us-gaap_NonCash...")
        text = concept.split(":", 1)[-1]
        
        # Remove ticker prefix if ticker is provided
        if ticker:
            for variant in cls._ticker_prefix_variants(ticker):
                lower_variant = variant.lower()
                if text.startswith(variant + ":") or text.startswith(lower_variant + ":"):
                    text = text.split(":", 1)[-1]
                ticker_pattern = rf"^(?:{re.escape(variant)}|{re.escape(lower_variant)})_"
                text = re.sub(ticker_pattern, '', text, flags=re.IGNORECASE)
        
        # Remove us-gaap/us_gaap prefixes
        if text.lower().startswith("us-gaap") or text.lower().startswith("us_gaap"):
            text = text.split("_", 1)[-1] if "_" in text else text[len("us-gaap") :]
        return text

    def _load_alias_canonical_set(self, path: Union[str, Path]) -> Set[str]:
        data = json.loads(Path(path).read_text())
        aliases = data.get("aliases", {}) or {}
        normalized: Set[str] = set()
        for canonical, variants in aliases.items():
            normalized.add(self._normalize_component_name(canonical))
            for variant in variants:
                normalized.add(self._normalize_component_name(variant))
        return normalized

    def _component_keys(self, canonical: str) -> List[str]:
        stripped = self._strip_statement_prefix(canonical)
        snake = self._to_snake_case(stripped)
        keys: List[str] = []
        for value in (canonical, stripped, snake):
            if isinstance(value, str) and value and value not in keys:
                keys.append(value)
        return keys

    def _load_metric_buckets(self) -> Dict[str, Any]:
        path = Path(self.metric_buckets_path)
        if not path.exists():
            raise FileNotFoundError(f"Bucket schema not found at {path}")
        with open(path, "r") as handle:
            return json.load(handle)

    def _get_component_lookup(self) -> Dict[str, Dict[str, str]]:
        """
        Build (and cache) a lookup from normalized component names to
        their metric_components entries.
        """
        lookup = getattr(self, "_component_lookup", None)
        if lookup is not None:
            return lookup

        components_doc = self._load_metric_components()
        buckets = components_doc.get("buckets", {}) or {}
        flattened: Dict[str, Dict[str, str]] = {}
        for bucket_name, meta in buckets.items():
            components = (meta.get("components") or {}).keys()
            for component in components:
                normalized = self._normalize_component_name(component)
                if normalized not in flattened:
                    flattened[normalized] = {
                        "component": component,
                        "bucket": bucket_name,
                    }
        self._component_lookup = flattened
        return flattened

    def _canonical_candidates_from_raw(
        self,
        raw_field: str,
        ticker: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a prioritized list of canonical candidates for a raw field.
        """
        candidates: List[str] = []
        seen: Set[str] = set()

        def _add(value: Optional[str]) -> None:
            if not value:
                return
            candidate = str(value).strip()
            if not candidate:
                return
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        if not raw_field or not isinstance(raw_field, str):
            return candidates

        raw = raw_field.strip()

        # Primary attempt: use alias mapping helper
        try:
            canonical = self._finnhub_field_to_canon(raw, ticker=ticker)
            _add(canonical)
        except Exception:
            pass

        stripped = self._strip_us_gaap_tag(raw, ticker=ticker)
        _add(stripped)

        normalized_raw = self._normalize_name(stripped)
        canonical_from_reverse = self._reverse_map.get(normalized_raw)
        _add(canonical_from_reverse)

        normalized_component = self._normalize_component_name(stripped)
        _add(normalized_component)

        if canonical_from_reverse:
            _add(self._normalize_component_name(canonical_from_reverse))

        _add(self._strip_statement_prefix(normalized_component))

        return candidates

    def map_fields_to_components(
        self,
        raw_fields: Sequence[str],
        ticker: Optional[str],
        provider: str = "finn",
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Map raw provider-specific field names to canonical components defined in metric_components.json.

        For each raw field the method:
          1. Strips provider/ticker/us-gaap prefixes.
          2. Uses the merged alias map to resolve to a canonical field.
          3. Normalizes to snake_case without statement prefixes.
          4. Looks up the matching component (and bucket) in metric_components.json.

        Returns a dictionary keyed by the original raw field, with values:
            {
              "canonical": <canonical_field or None>,
              "component": <metric_components component name or None>,
              "bucket": <bucket name or None>,
            }
        """
        if not raw_fields:
            return {}

        provider_normalized = self._normalize_provider(provider)
        self._load_aliases(provider_normalized)
        self._build_reverse_map()

        component_lookup = self._get_component_lookup()
        results: Dict[str, Dict[str, Optional[str]]] = {}

        for raw in raw_fields:
            canonical: Optional[str] = None
            component_entry: Optional[Dict[str, str]] = None

            if isinstance(raw, str) and raw.strip():
                candidates = self._canonical_candidates_from_raw(raw, ticker=ticker)
                if candidates:
                    canonical = candidates[0]
                for candidate in candidates:
                    normalized = self._normalize_component_name(candidate)
                    entry = component_lookup.get(normalized)
                    if entry:
                        component_entry = entry
                        canonical = entry.get("component", canonical)
                        break

            results[raw] = {
                "canonical": canonical,
                "component": component_entry.get("component") if component_entry else None,
                "bucket": component_entry.get("bucket") if component_entry else None,
            }

        return results

    def get_bucket_for_raw_field(
        self,
        raw_field: str,
        ticker: Optional[str] = None,
        provider: str = "finn",
    ) -> Optional[str]:
        """
        Map a raw field name to its bucket in metric_components.json.
        
        Process:
        1. Strip ticker/us-gaap prefixes
        2. Map through merged alias JSON to get canonical name
        3. Normalize to snake_case
        4. Look up in metric_components.json to find which bucket it belongs to
        
        Parameters
        ----------
        raw_field : str
            Raw field name from provider (e.g., "ADI:NetIncome" or "adi_net_income")
        ticker : str, optional
            Ticker symbol for prefix stripping (e.g., "ADI")
        provider : str
            Provider name ("finn" or "poly")
            
        Returns
        -------
        str or None
            Bucket name if found in metric_components.json, None otherwise
        """
        if not raw_field or not isinstance(raw_field, str):
            return None
        
        try:
            provider_normalized = self._normalize_provider(provider)
            self._load_aliases(provider_normalized)
            self._build_reverse_map()

            candidates = self._canonical_candidates_from_raw(raw_field, ticker=ticker)
            component_lookup = self._get_component_lookup()

            for candidate in candidates:
                normalized = self._normalize_component_name(candidate)
                component_entry = component_lookup.get(normalized)
                if component_entry:
                    return component_entry.get("bucket")

            # Fall back to metric bucket mapping (top-level metrics)
            metric_buckets = self._load_metric_buckets()
            bucket_entries = metric_buckets.get("buckets", {})
            for bucket_name, meta in bucket_entries.items():
                metric = meta.get("metric")
                if metric:
                    metric_normalized = self._normalize_component_name(metric)
                    for candidate in candidates:
                        normalized_candidate = self._normalize_component_name(candidate)
                        if normalized_candidate == metric_normalized:
                            return bucket_name
                # Also compare bucket names directly
                if bucket_name:
                    bucket_normalized = self._normalize_component_name(bucket_name)
                    for candidate in candidates:
                        normalized_candidate = self._normalize_component_name(candidate)
                        if normalized_candidate == bucket_normalized:
                            return bucket_name

            return None
        except Exception:
            return None


    def analyze_bucket_components(
        self,
        bucket_name: str,
    ) -> Dict[str, Any]:
        """
        Analyze a bucket from metric_components.json to see which components:
        - Exist in the merged alias JSON
        - Map to metrics in metric_buckets.json
        
        Parameters
        ----------
        bucket_name : str
            Name of the bucket to analyze (e.g., "revenues", "operating_expenses")
            
        Returns
        -------
        Dict with:
            - "bucket_name": The bucket name
            - "components": List of all components in the bucket
            - "components_with_aliases": Dict mapping component -> list of aliases
            - "components_without_aliases": List of components not in merged alias
            - "metric_bucket_mapping": The metric bucket entry if it exists
            - "metric_name": The metric name from metric_buckets.json if it exists
        """
        components_doc = self._load_metric_components()
        buckets = components_doc.get("buckets", {})
        
        if bucket_name not in buckets:
            return {
                "bucket_name": bucket_name,
                "error": f"Bucket '{bucket_name}' not found in metric_components.json",
                "available_buckets": list(buckets.keys()),
            }
        
        bucket_meta = buckets[bucket_name]
        bucket_components = bucket_meta.get("components", {})
        component_names = list(bucket_components.keys())
        
        # Load merged alias JSON
        alias_path = Path(self.finn_alias_path)
        if not alias_path.exists():
            return {
                "bucket_name": bucket_name,
                "error": f"Alias file not found at {alias_path}",
            }
        
        alias_data = json.loads(alias_path.read_text())
        aliases = alias_data.get("aliases", {})
        
        # Check which components have aliases
        components_with_aliases: Dict[str, List[str]] = {}
        components_without_aliases: List[str] = []
        
        for component in component_names:
            normalized = self._normalize_component_name(component)
            if component in aliases:
                components_with_aliases[component] = aliases[component]
            elif normalized in aliases:
                components_with_aliases[component] = aliases[normalized]
            else:
                components_without_aliases.append(component)
        
        # Check metric_buckets.json
        metric_buckets = self._load_metric_buckets()
        bucket_entries = metric_buckets.get("buckets", {})
        metric_bucket_entry = bucket_entries.get(bucket_name)
        metric_name = metric_bucket_entry.get("metric") if metric_bucket_entry else None
        
        return {
            "bucket_name": bucket_name,
            "components": component_names,
            "component_count": len(component_names),
            "components_with_aliases": components_with_aliases,
            "components_without_aliases": components_without_aliases,
            "alias_coverage": len(components_with_aliases) / len(component_names) if component_names else 0.0,
            "metric_bucket_mapping": metric_bucket_entry,
            "metric_name": metric_name,
        }

    def find_dead_components(
        self,
    ) -> Dict[str, Any]:
        """
        Find "dead" sub-components that exist in metric_components.json but:
        - Don't exist as keys in the merged alias JSON
        - Don't exist as bucket names in metric_buckets.json
        
        Returns
        -------
        Dict with:
            - "dead_components": List of components not in aliases or metric_buckets
            - "dead_by_bucket": Dict mapping bucket -> list of dead components
            - "total_dead": Count of dead components
            - "total_components": Total components across all buckets
        """
        components_doc = self._load_metric_components()
        buckets = components_doc.get("buckets", {})
        
        # Load merged alias JSON
        alias_path = Path(self.finn_alias_path)
        if not alias_path.exists():
            return {
                "error": f"Alias file not found at {alias_path}",
            }
        
        alias_data = json.loads(alias_path.read_text())
        aliases = alias_data.get("aliases", {})
        alias_keys = set(aliases.keys())
        # Also check normalized versions
        for key in aliases.keys():
            alias_keys.add(self._normalize_component_name(key))
        
        # Load metric_buckets.json
        metric_buckets = self._load_metric_buckets()
        bucket_entries = metric_buckets.get("buckets", {})
        metric_bucket_names = set(bucket_entries.keys())
        # Also check metric values
        for entry in bucket_entries.values():
            metric = entry.get("metric")
            if metric:
                metric_bucket_names.add(metric)
                # Strip statement prefixes for comparison
                metric_bucket_names.add(self._normalize_component_name(metric))
        
        dead_components: List[str] = []
        dead_by_bucket: Dict[str, List[str]] = {}
        total_components = 0
        
        for bucket_name, bucket_meta in buckets.items():
            bucket_components = bucket_meta.get("components", {})
            bucket_dead: List[str] = []
            
            for component in bucket_components.keys():
                total_components += 1
                normalized = self._normalize_component_name(component)
                
                # Check if component exists in aliases (exact or normalized)
                in_aliases = (
                    component in alias_keys
                    or normalized in alias_keys
                )
                
                # Check if component exists in metric_buckets (as bucket name or metric)
                in_metric_buckets = (
                    component in metric_bucket_names
                    or normalized in metric_bucket_names
                    or bucket_name in metric_bucket_names
                )
                
                if not in_aliases and not in_metric_buckets:
                    dead_components.append(component)
                    bucket_dead.append(component)
            
            if bucket_dead:
                dead_by_bucket[bucket_name] = bucket_dead
        
        return {
            "dead_components": sorted(dead_components),
            "dead_by_bucket": dead_by_bucket,
            "total_dead": len(dead_components),
            "total_components": total_components,
            "dead_percentage": len(dead_components) / total_components if total_components > 0 else 0.0,
        }

    def _load_metric_components(self) -> Dict[str, Any]:
        path = Path(self.metric_components_path)
        if not path.exists():
            raise FileNotFoundError(f"Metric components file not found at {path}")
        with open(path, "r") as handle:
            return json.load(handle)

    def get_bucket_components_map(self) -> Dict[str, List[str]]:
        """
        Return a dict of bucket -> list of component names.
        If a bucket has no components, map it to [bucket] itself.
        """
        components_doc = self._load_metric_components()
        buckets = components_doc.get("buckets", {}) or {}
        out: Dict[str, List[str]] = {}
        for bucket_name, meta in buckets.items():
            comps = list((meta.get("components") or {}).keys())
            if comps:
                out[bucket_name] = comps
            else:
                out[bucket_name] = [bucket_name]
        return out

    def _save_metric_components(self, components: Dict[str, Any]) -> None:
        path = Path(self.metric_components_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(components, handle, indent=2)

    def _ensure_component_tracking(self, canonical: str) -> None:
        """Ensure every canonical field appears in metric_components (default -> other)."""
        try:
            components = self._load_metric_components()
        except FileNotFoundError:
            return

        for meta in components.get("buckets", {}).values():
            keys = meta.get("components", {})
            if any(key in keys for key in self._component_keys(canonical)):
                return

        other_bucket = components.setdefault("buckets", {}).setdefault(
            "other", {"components": {}}
        )
        other_components = other_bucket.setdefault("components", {})
        for key in self._component_keys(canonical):
            if key not in other_components:
                other_components[key] = {"sign": 1.0}
            self._save_metric_components(components)

    def _find_bucket_for_field(
        self, canonical: str, components: Dict[str, Any]
    ) -> Optional[str]:
        for bucket_key, meta in components.get("buckets", {}).items():
            keys = meta.get("components", {})
            for candidate in self._component_keys(canonical):
                if candidate in keys:
                    return bucket_key
        return None

    def _prompt_bucket_assignment(
        self,
        canonical: str,
        components: Dict[str, Any],
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        keys = self._component_keys(canonical)
        display_key = keys[-1]
        buckets = list(components.get("buckets", {}).keys())
        friendly = display_name or canonical
        self._log(
            f"[Input] Map '{friendly}' (raw '{canonical}', stored as '{display_key}') to bucket "
            f"(options: {', '.join(f'{i}:{b}' for i, b in enumerate(buckets))})"
        )
        while True:
            bucket_choice = input("Bucket name or index (blank=skip): ").strip()
            if not bucket_choice:
                self._log(f"Skipping bucket assignment for {canonical}")
                return components
            if bucket_choice.isdigit():
                idx = int(bucket_choice)
                if 0 <= idx < len(buckets):
                    bucket_choice = buckets[idx]
            if bucket_choice not in components["buckets"]:
                print("Invalid bucket. Try again.")
                continue
            sign_choice = input("Sign (+/-, default +): ").strip() or "+"
            sign = 1.0 if sign_choice != "-" else -1.0
            for meta in components["buckets"].values():
                comps = meta.setdefault("components", {})
                for candidate in keys:
                    comps.pop(candidate, None)
            target = components["buckets"][bucket_choice].setdefault(
                "components", {}
            )
            target[display_key] = {"sign": sign}
            self._save_metric_components(components)
            print(f"Mapped {canonical} -> {bucket_choice} (sign={sign:+})\n")
            return components

    def _map_fields_to_buckets(
        self,
        canonical_fields: Iterable[str],
        ticker: Optional[str] = None,
    ) -> None:
        try:
            components = self._load_metric_components()
        except FileNotFoundError:
            self._log("metric_components.json missing; skipping bucket mapping.")
            return

        unmapped_details: List[Tuple[str, str]] = []
        for field in canonical_fields:
            if self._find_bucket_for_field(field, components) is None:
                display = self._describe_unmapped_field(field, ticker=ticker)
                unmapped_details.append((field, display))

        if not unmapped_details:
            return

        if not self.interactive:
            example_display = [
                (display or raw) for raw, display in unmapped_details[:5]
            ]
            self._log(
                f"{len(unmapped_details)} field(s) lack bucket mapping "
                f"(e.g. {example_display}). Run bucket mapper to classify them."
            )
            return

        try:
            _ = self._load_metric_buckets()
        except FileNotFoundError as exc:
            self._log(str(exc))
            return

        for raw_field, display in unmapped_details:
            components = self._prompt_bucket_assignment(
                raw_field, components, display_name=display
            )

    def _find_bucket_for_metric(self, metric_name: str) -> Optional[str]:
        return METRIC_TO_BUCKET_MAP.get(metric_name)

    def _describe_unmapped_field(
        self,
        field: Any,
        ticker: Optional[str] = None,
    ) -> str:
        """
        Return a human-friendly representation of an unmapped field by stripping
        ticker prefixes, us-gaap tags, and statement prefixes (is_/bs_/cf_/ci_).
        """
        if not isinstance(field, str):
            return str(field)

        candidate = field.strip()
        candidate = self._strip_us_gaap_tag(candidate, ticker=ticker)

        if ticker:
            pattern = re.compile(rf"^{re.escape(ticker)}[_:]", re.IGNORECASE)
            candidate = re.sub(pattern, "", candidate, count=1)

        candidate = self._strip_statement_prefix(candidate or "")
        candidate = candidate.strip("_")
        return candidate or field

    def _prepare_canonical_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all final metrics exist, are numeric, and benefit from bucket aggregation + rules.
        """
        if df.empty:
            return df

        df = df.copy()
        required_cols: Set[str] = set()
        required_cols.update(BUCKET_METRIC_MAP.values())

        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = self._num(df[col])

        self._apply_bucket_aggregates(df)

        metrics_to_rebuild = [m for m in required_cols if m in METRIC_RULES]
        if metrics_to_rebuild:
            for metric in metrics_to_rebuild:
                series = df.apply(
                    lambda row: compute_metric(metric, row.to_dict(), METRIC_RULES)[0],
                    axis=1,
                )
                existing = self._num(df[metric])
                df[metric] = existing.combine_first(self._num(series))

        return df

    

    def _ensure_bucket_components_loaded(self) -> None:
        """
        Lazily load metric_components.json into self._bucket_components.

        Structure:
            self._bucket_components: Dict[bucket_key, Dict[canonical_name, float]]
        """
        if getattr(self, "_bucket_components", None) is not None:
            return

        path = Path(self.metric_components_path)
        if not path.exists():
            self._log(f"[buckets] No metric_components.json found at {path}", level="warning")
            self._bucket_components = {}
            return

        raw = json.loads(path.read_text())
        buckets = raw.get("buckets", {}) or {}
        out: Dict[str, Dict[str, float]] = {}

        for bucket_key, meta in buckets.items():
            comps = meta.get("components", {}) or {}
            clean: Dict[str, float] = {}
            for canonical, meta2 in comps.items():
                if not isinstance(canonical, str):
                    continue
                try:
                    sign = float(meta2.get("sign", 1.0))
                except Exception:
                    sign = 1.0
                clean[canonical] = sign
            out[bucket_key] = clean

        self._bucket_components = out
        self._log(f"[buckets] Loaded components for {len(out)} bucket(s).")

    def _resolve_column_for_canonical(self, df: pd.DataFrame, canonical: str) -> Optional[str]:
        """
        Given a canonical name from metric_components.json, find the actual
        column name in df.

        Resolution order:
          1. Exact match (canonical).
          2. Any STATEMENT_PREFIXES + canonical (e.g. is_revenues, bs_assets, cf_*).
          3. If canonical already starts with a prefix, also try stripping it and
             re-adding each prefix.
        """
        cols = list(df.columns)
        normalized = self._normalize_component_name(canonical)
        candidates = []
        if canonical not in candidates:
            candidates.append(canonical)
        if normalized not in candidates:
            candidates.append(normalized)

        for name in candidates:
            if name in cols:
                return name

        for prefix in STATEMENT_PREFIXES:
            for name in candidates:
                candidate = f"{prefix}{name}"
                if candidate in cols:
                    return candidate

        for col in cols:
            if self._normalize_component_name(col) == normalized:
                return col

        return None

    def _get_unmapped_fields(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "quarterly",
    ) -> Dict[str, Dict[str, List[str]]]:
        """Collect Finnhub concepts missing from alias maps or metric components."""
        try:
            self.fetch_finnhub_statements(
                ticker,
                start_date=start_date,
                end_date=end_date,
                freq=freq,
                finn_jason_save=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Finnhub statements for {ticker}: {exc}") from exc

        api_response = getattr(self, "api_response_finn", None)
        if not api_response:
            raise ValueError("Finnhub API response unavailable; ensure finn_jason_save=True was used.")

        alias_set = self._load_alias_canonical_set(self.poly_alias_path)
        alias_set.update(self._load_alias_canonical_set(self.finn_alias_path))
        if FINNHUB_EXPLICIT_MAP:
            alias_set.update({self._normalize_component_name(name) for name in FINNHUB_EXPLICIT_MAP.values()})

        components_doc = self._load_metric_components()
        component_names = {
            self._normalize_component_name(component)
            for meta in (components_doc.get("buckets") or {}).values()
            for component in (meta.get("components") or {}).keys()
        }

        buckets = {
            "ic": "income_statement",
            "bs": "balance_sheet",
            "cf": "cash_flow",
        }
        summary: Dict[str, Dict[str, Set[str]]] = {
            label: {"missing_alias": set(), "missing_components": set()}
            for label in buckets.values()
        }

        for filing in api_response.get("data", []):
            report = filing.get("report") or {}
            filing_ticker = filing.get("symbol") or ticker
            for stmt_key, label in buckets.items():
                for entry in report.get(stmt_key, []) or []:
                    concept = self._strip_us_gaap_tag(entry.get("concept"), ticker=filing_ticker)
                    normalized = self._normalize_component_name(concept)
                    if not normalized:
                        continue
                    if normalized not in alias_set:
                        summary[label]["missing_alias"].add(normalized)
                    if normalized not in component_names:
                        summary[label]["missing_components"].add(normalized)

        return {
            label: {
                "missing_alias": sorted(values["missing_alias"]),
                "missing_components": sorted(values["missing_components"]),
            }
            for label, values in summary.items()
        }

    def _sum_bucket(self, df: pd.DataFrame, bucket_key: str) -> pd.Series:
        """
        Sum all components for a bucket, respecting signs and NaNs.

        Returns a Series aligned to df.index, with NaN where all components are NaN.
        """
        self._ensure_bucket_components_loaded()
        entries = self._bucket_components.get(bucket_key, {})
        if not entries:
            return pd.Series(np.nan, index=df.index)

        values = pd.Series(0.0, index=df.index, dtype="float64")
        mask = pd.Series(False, index=df.index)

        resolved_cols: Dict[str, float] = {}
        missing: List[str] = []

        for canonical, sign in entries.items():
            column_name = self._resolve_column_for_canonical(df, canonical)
            if column_name is None:
                missing.append(canonical)
                continue
            resolved_cols[column_name] = sign

        if missing:
            self._log(
                f"[buckets:{bucket_key}] {len(missing)} component(s) not present in frame: "
                + ", ".join(sorted(missing)),
                level="debug",
            )

        if not resolved_cols:
            return pd.Series(np.nan, index=df.index)

        for column_name, sign in resolved_cols.items():
            series = self._num(df[column_name])
            values = values.add(sign * series.fillna(0.0), fill_value=0.0)
            mask = mask | series.notna()

        values = values.where(mask)
        return values

    
    def get_bucket_component_series(self, metric_name: str) -> pd.DataFrame:
        """
        Return the signed component contributions captured during the most
        recent `get_financial_statement_quarterised(..., see_bucket=True)` call.

        The returned DataFrame now carries metadata that lists the exact
        column names `_quarterize_flows` needs in order to flow-quarterize
        this metric. Access them via:

            frame = analyzer.get_bucket_component_series("is_revenues")
            frame.attrs["quarterize_flow_columns"]  # present in frame
            frame.attrs["quarterize_flow_columns_missing"]  # canonical names still unresolved
        """
        if not self._bucket_component_store:
            raise RuntimeError(
                "Bucket components were not captured. Call get_financial_statement_quarterised(..., see_bucket=True) first."
            )
        if metric_name not in self._bucket_component_store:
            raise KeyError(f"No bucket component data stored for metric '{metric_name}'")
        return self._bucket_component_store[metric_name].copy()

    def run_data_quality_checks(self, df: pd.DataFrame, null_threshold: int = 50) -> pd.DataFrame:
        """
        Run data-integrity checks on quarterly financial data for a single entity.

        Assumptions:
        - df.index is the period identifier (e.g. '2022Q1', '2023Q4'), type string.
        - df contains one entity (ticker) only.
        - df may have multiple quarters; will be sorted by index.

        Returns a DataFrame with columns:
            - metric:  name of metric or relationship
            - quarter: index value from the input df (e.g. '2022Q1')
            - error:   error category string
            - value:   numeric value that triggered the rule (residual, pct change, etc.)
            - details: short text specifying the identity or change (e.g. QoQ/YoY)
        """

        # Ensure chronological order by index
        df = df.sort_index().copy()
        null_df = df.isnull().sum(axis=1)
        if null_threshold:
            df = df[null_df < null_threshold]
        # -------------------------------------------------------------------------
        # Helpers
        # -------------------------------------------------------------------------
        def safe_div(a, b):
            """
            Elementwise safe division on 1D arrays / Series.
            Returns a numpy array of floats with NaN where division is not possible.
            """
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            out = np.full_like(a_arr, np.nan, dtype=float)
            mask = (b_arr != 0) & (~np.isnan(b_arr))
            out[mask] = a_arr[mask] / b_arr[mask]
            return out

        issues = []

        def add_issue(metric, quarter, error, value, details=None):
            issues.append(
                {
                    "metric": metric,
                    "quarter": quarter,
                    "error": error,
                    "value": value,
                    "details": details,
                }
            )

        # -------------------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------------------

        # Identity tolerances (relative, e.g. 0.01 = 1%)
        IDENTITY_TOL = {
            "bs_assets_eq_liab_plus_equity": 0.025,  # 1% of assets
            "bs_total_liabilities_split": 0.025,     # 1% of total liabilities
            "is_gross_profit_identity": 0.01,       # 1% of revenue
            "is_operating_income_identity": 0.02,   # 2% of revenue
            "is_net_income_identity": 0.05,         # 5% of revenue
            "cf_total_net_cash_flow": 0.10,         # 2% of |net cash flow|
            "cf_fcf_identity": 0.05,                # 5% of |FCF|
            "cash_balance_linkage": 0.05,           # 5% of |delta cash|
            "mkt_cap_linkage": 0.10,                # 10% of market cap
        }

        # Q/Q and Y/Y outlier thresholds (absolute percentage change, e.g. 10 = 1000%)
        QOQ_ABS_CHANGE_THRESHOLD = 10.0
        YOY_ABS_CHANGE_THRESHOLD = 10.0

        # Metrics expected to be non-negative (including CapEx, if stored as positive outflow)
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

        # Metrics expected to be non-positive (<= 0)
        STRICT_NONPOSITIVE = [
            "treasury_stock_value",
        ]

        CORE_MUST_HAVE = [
            "bs_assets",
            "bs_total_liabilities",
            "bs_equity",
            "is_revenues",
            "is_net_income_loss",
            "cf_net_cash_flow_from_operating_activities",
            "cf_net_cash_flow",
        ]

        NAN_CHECK_COLUMNS = [
            "is_revenues",
            "is_cost_of_revenue",
            "is_operating_expenses",
            "is_research_and_development",
            "is_selling_general_and_administrative_expenses",
            "is_net_income_loss",
            "bs_market_cap_diluted",
            "bs_market_cap_basic",
            "is_operating_income_loss",
            "eps_basic",
            "eps_diluted",
            "bs_net_debt",
            "cf_free_cash_flow",
            "cf_net_cash_flow_from_operating_activities",
            "bs_cash_and_equivalents_short_term_investments",
            "cf_capital_expenditures",
            "bs_net_property_plant_and_equipment",
        ]

        VOL_METRICS = [
            "bs_assets",
            "bs_equity",
            "bs_total_liabilities",
            "bs_cash_and_cash_equivalents",
            "bs_property_plant_and_equipment_net",
            "bs_goodwill",
            "bs_interest_bearing_debt",
            "bs_market_cap_diluted",
            "is_revenues",
            "is_gross_profit",
            "is_operating_income_loss",
            "is_net_income_loss",
            "is_income_tax_expense_benefit",
            "cf_net_cash_flow_from_operating_activities",
            "cf_net_cash_flow",
            "cf_capital_expenditures",
            "cf_free_cash_flow",
        ]

        quarters = df.index

        # -------------------------------------------------------------------------
        # 1) Sign checks & missingness
        # -------------------------------------------------------------------------

        total_rows = len(df)
        if total_rows:
            for col in NAN_CHECK_COLUMNS:
                if col not in df.columns:
                    continue
                nan_count = df[col].isna().sum()
                if nan_count == 0:
                    continue
                if nan_count == total_rows:
                    add_issue(
                        col,
                        "ALL",
                        "ALL_NANS",
                        nan_count,
                        details=f"{nan_count}/{total_rows} rows are NaN",
                    )
                else:
                    add_issue(
                        col,
                        "ALL",
                        "CONTAINS_NANS",
                        nan_count,
                        details=f"{nan_count}/{total_rows} rows are NaN",
                    )

        # Strictly non-negative
        for col in STRICT_POSITIVE:
            if col in df.columns:
                vals = np.asarray(df[col], dtype=float)
                bad_mask = vals < 0
                for q, bad, v in zip(quarters, bad_mask, vals):
                    if bad:
                        add_issue(col, q, "SIGN_VIOLATION_STRICT_POSITIVE", v,
                                  details="Expected >= 0")

        # Non-positive (<= 0)
        for col in STRICT_NONPOSITIVE:
            if col in df.columns:
                vals = np.asarray(df[col], dtype=float)
                bad_mask = vals > 0
                for q, bad, v in zip(quarters, bad_mask, vals):
                    if bad:
                        add_issue(col, q, "SIGN_VIOLATION_STRICT_NONPOSITIVE", v,
                                  details="Expected <= 0")

        # Missing core metrics
        for col in CORE_MUST_HAVE:
            if col in df.columns:
                bad_mask = df[col].isna().to_numpy()
                for q, bad in zip(quarters, bad_mask):
                    if bad:
                        add_issue(col, q, "MISSING_EXPECTED_VALUE", np.nan,
                                  details="Core metric missing")

        # -------------------------------------------------------------------------
        # 2) Accounting identities (within a quarter)
        # -------------------------------------------------------------------------

        # 2.1 Assets ≈ Liabilities + Equity
        if {"bs_assets", "bs_total_liabilities", "bs_equity"}.issubset(df.columns):
            residual = df["bs_assets"] - (df["bs_total_liabilities"] + df["bs_equity"])
            base = np.abs(np.asarray(df["bs_assets"], dtype=float))
            res_arr = np.asarray(residual, dtype=float)
            tol = IDENTITY_TOL["bs_assets_eq_liab_plus_equity"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "bs_assets",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details="Assets != Liabilities + Equity"
                    )

            # Proportion: current assets <= total assets
            if "bs_current_assets" in df.columns:
                curr = np.asarray(df["bs_current_assets"], dtype=float)
                total = np.asarray(df["bs_assets"], dtype=float)
                bad_mask = curr > total
                for q, bad, c, t in zip(quarters, bad_mask, curr, total):
                    if bad and t != 0:
                        add_issue(
                            "bs_current_assets",
                            q,
                            "PROPORTION_OUT_OF_RANGE",
                            c / t,
                            details="Current assets > total assets (ratio)"
                        )

        # 2.2 Total liabilities ≈ current + noncurrent (alt)
        if {"bs_total_liabilities", "bs_current_liabilities", "bs_noncurrent_liabilities_alt"}.issubset(df.columns):
            residual = (
                df["bs_total_liabilities"]
                - (df["bs_current_liabilities"] + df["bs_noncurrent_liabilities_alt"])
            )
            base = np.abs(np.asarray(df["bs_total_liabilities"], dtype=float))
            res_arr = np.asarray(residual, dtype=float)
            tol = IDENTITY_TOL["bs_total_liabilities_split"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "bs_total_liabilities",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details="Total liabilities != current + noncurrent (alt)"
                    )

            # Proportion: current liabilities <= total liabilities
            curr = np.asarray(df["bs_current_liabilities"], dtype=float)
            total = np.asarray(df["bs_total_liabilities"], dtype=float)
            bad_mask = curr > total
            for q, bad, c, t in zip(quarters, bad_mask, curr, total):
                if bad and t != 0:
                    add_issue(
                        "bs_current_liabilities",
                        q,
                        "PROPORTION_OUT_OF_RANGE",
                        c / t,
                        details="Current liabilities > total liabilities (ratio)"
                    )

        # 2.2b Noncurrent liabilities consistency: reported vs alt
        if {"bs_noncurrent_liabilities", "bs_noncurrent_liabilities_alt"}.issubset(df.columns):
            diff = df["bs_noncurrent_liabilities"] - df["bs_noncurrent_liabilities_alt"]
            base = np.abs(np.asarray(df["bs_noncurrent_liabilities_alt"], dtype=float))
            res_arr = np.asarray(diff, dtype=float)
            tol = IDENTITY_TOL["bs_total_liabilities_split"]

            bad_mask = ((base > 0) & (np.abs(res_arr) > tol * base)) | ((base == 0) & (np.abs(res_arr) > 1e-6))
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "bs_noncurrent_liabilities_alt",
                        q,
                        "ACCOUNTING_IDENTITY_ERROR",
                        r,
                        details="Noncurrent liabilities vs alt differ",
                    )

        # 2.3 PP&E: net <= gross
        if {
            "bs_property_plant_and_equipment_gross",
            "bs_property_plant_and_equipment_net",
        }.issubset(df.columns):
            net = np.asarray(df["bs_property_plant_and_equipment_net"], dtype=float)
            gross = np.asarray(df["bs_property_plant_and_equipment_gross"], dtype=float)
            bad_mask = net > gross
            for q, bad, n, g in zip(quarters, bad_mask, net, gross):
                if bad:
                    add_issue(
                        "bs_property_plant_and_equipment_net",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        n - g,
                        details="Net PP&E > gross PP&E"
                    )

        # 2.4 Income statement identities

        # Gross profit: revenue - COGS ≈ gross profit
        if {"is_revenues", "is_cost_of_revenue", "is_gross_profit"}.issubset(df.columns):
            residual = df["is_revenues"] - df["is_cost_of_revenue"] - df["is_gross_profit"]
            res_arr = np.asarray(residual, dtype=float)
            base = np.abs(np.asarray(df["is_revenues"], dtype=float))
            tol = IDENTITY_TOL["is_gross_profit_identity"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "is_gross_profit",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details="Gross profit != Revenue - COGS"
                    )

        # Operating income: gross profit - op ex ≈ operating income
        if {
            "is_gross_profit",
            "is_operating_expenses",
            "is_operating_income_loss",
        }.issubset(df.columns):
            residual = (
                df["is_gross_profit"]
                - df["is_operating_expenses"]
                - df["is_operating_income_loss"]
            )
            res_arr = np.asarray(residual, dtype=float)
            if "is_revenues" in df.columns:
                base = np.abs(np.asarray(df["is_revenues"], dtype=float))
            else:
                base = np.abs(np.asarray(df["is_gross_profit"], dtype=float))
            tol = IDENTITY_TOL["is_operating_income_identity"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "is_operating_income_loss",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details="Operating income != Gross profit - Operating expenses"
                    )

        # Net income (simplified)
        required = {
            "is_net_income_loss",
            "is_operating_income_loss",
            "is_net_interest_expense",
            "is_income_tax_expense_benefit",
        }
        if required.issubset(df.columns):
            approx = (
                df["is_operating_income_loss"]
                - df["is_net_interest_expense"]
                - df["is_income_tax_expense_benefit"]
            )
            if "is_noncontrolling_interest_income" in df.columns:
                approx = approx - df["is_noncontrolling_interest_income"]
            if "is_equity_method_income" in df.columns:
                approx = approx + df["is_equity_method_income"]
            if "crypto_asset_gain_loss" in df.columns:
                approx = approx + df["crypto_asset_gain_loss"]

            residual = df["is_net_income_loss"] - approx
            res_arr = np.asarray(residual, dtype=float)
            if "is_revenues" in df.columns:
                base = np.abs(np.asarray(df["is_revenues"], dtype=float))
            else:
                base = np.abs(np.asarray(df["is_net_income_loss"], dtype=float))
            tol = IDENTITY_TOL["is_net_income_identity"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "is_net_income_loss",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details="Net income != Operating income - interest - tax (+/- other)"
                    )

        # 2.5 Cash-flow identities
        # Total net cash flow (including FX effect) = CFO + CFI + CFF
        cf_cols = {
            "cf_net_cash_flow_from_operating_activities",
            "cf_net_cash_flow_from_investing_activities",
            "cf_net_cash_flow_from_financing_activities",
            "cf_net_cash_flow",
        }
        has_fx = "cf_effect_of_exchange_rate_on_cash_and_cash_equivalents" in df.columns

        if cf_cols.issubset(df.columns):
            cfo = df["cf_net_cash_flow_from_operating_activities"]
            cfi = df["cf_net_cash_flow_from_investing_activities"]
            cff = df["cf_net_cash_flow_from_financing_activities"]

            # CFO + CFI + CFF
            sum_cashflows = cfo + cfi + cff

            if has_fx:
                # Net cash flows after including FX effect on cash
                fx = df["cf_effect_of_exchange_rate_on_cash_and_cash_equivalents"]
                net_cashflows = -fx + df["cf_net_cash_flow"]

                residual = sum_cashflows - net_cashflows
                base = np.abs(np.asarray(net_cashflows, dtype=float))
                details_text = "Net cash flow + FX effect != CFO + CFI + CFF"
            else:
                # Fallback to legacy identity if FX column is missing
                residual = sum_cashflows - df["cf_net_cash_flow"]
                base = np.abs(np.asarray(df["cf_net_cash_flow"], dtype=float))
                details_text = "Net cash flow != CFO + CFI + CFF"

            res_arr = np.asarray(residual, dtype=float)
            tol = IDENTITY_TOL["cf_total_net_cash_flow"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "cf_net_cash_flow",
                        q,
                        "ACCOUNTING_IDENTITY_MISMATCH",
                        r,
                        details=details_text,
                    )

        # FCF ≈ CFO - Capex
        if {
            "cf_free_cash_flow",
            "cf_net_cash_flow_from_operating_activities",
            "cf_capital_expenditures",
        }.issubset(df.columns):
            residual = (
                df["cf_net_cash_flow_from_operating_activities"]
                - df["cf_capital_expenditures"]
                - df["cf_free_cash_flow"]
            )
            res_arr = np.asarray(residual, dtype=float)
            base = np.abs(np.asarray(df["cf_free_cash_flow"], dtype=float))
            tol = IDENTITY_TOL["cf_fcf_identity"]

            bad_mask = (base > 0) & (np.abs(res_arr) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, res_arr):
                if bad:
                    add_issue(
                        "cf_free_cash_flow",
                        q,
                        "STATEMENT_LINKAGE_MISMATCH",
                        r,
                        details="FCF != CFO - Capex"
                    )

        # Cash balance linkage: Δcash ≈ net cash flow (approx)
        if {"bs_cash_and_cash_equivalents", "cf_net_cash_flow"}.issubset(df.columns):
            cash = np.asarray(df["bs_cash_and_cash_equivalents"], dtype=float)
            delta_cash = cash - np.roll(cash, 1)
            delta_cash[0] = np.nan  # first period has no prior
            residual = delta_cash - np.asarray(df["cf_net_cash_flow"], dtype=float)
            base = np.abs(delta_cash)
            tol = IDENTITY_TOL["cash_balance_linkage"]

            bad_mask = (base > 0) & (np.abs(residual) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, residual):
                if bad:
                    add_issue(
                        "bs_cash_and_cash_equivalents",
                        q,
                        "STATEMENT_LINKAGE_MISMATCH",
                        r,
                        details="ΔCash != net cash flow"
                    )

        # 2.6 Market cap / price / shares

        # Diluted >= basic
        if {"is_basic_average_shares", "is_diluted_average_shares"}.issubset(df.columns):
            basic = np.asarray(df["is_basic_average_shares"], dtype=float)
            diluted = np.asarray(df["is_diluted_average_shares"], dtype=float)
            bad_mask = diluted < basic
            for q, bad, b, d in zip(quarters, bad_mask, basic, diluted):
                if bad:
                    add_issue(
                        "is_diluted_average_shares",
                        q,
                        "STRUCTURAL_RELATION_VIOLATION",
                        d - b,
                        details="Diluted shares < basic shares"
                    )

        # Basic market cap ≈ price * basic shares
        if {"close", "is_basic_average_shares", "bs_market_cap_basic"}.issubset(df.columns):
            price = np.asarray(df["close"], dtype=float)
            shares = np.asarray(df["is_basic_average_shares"], dtype=float)
            mktcap = np.asarray(df["bs_market_cap_basic"], dtype=float)
            implied = price * shares
            residual = mktcap - implied
            base = np.abs(mktcap)
            tol = IDENTITY_TOL["mkt_cap_linkage"]

            bad_mask = (base > 0) & (np.abs(residual) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, residual):
                if bad:
                    add_issue(
                        "bs_market_cap_basic",
                        q,
                        "STATEMENT_LINKAGE_MISMATCH",
                        r,
                        details="Market cap (basic) != price × basic shares"
                    )

        # Diluted market cap ≈ price * diluted shares
        if {"close", "is_diluted_average_shares", "bs_market_cap_diluted"}.issubset(df.columns):
            price = np.asarray(df["close"], dtype=float)
            shares = np.asarray(df["is_diluted_average_shares"], dtype=float)
            mktcap = np.asarray(df["bs_market_cap_diluted"], dtype=float)
            implied = price * shares
            residual = mktcap - implied
            base = np.abs(mktcap)
            tol = IDENTITY_TOL["mkt_cap_linkage"]

            bad_mask = (base > 0) & (np.abs(residual) > tol * base)
            for q, bad, r in zip(quarters, bad_mask, residual):
                if bad:
                    add_issue(
                        "bs_market_cap_diluted",
                        q,
                        "STATEMENT_LINKAGE_MISMATCH",
                        r,
                        details="Market cap (diluted) != price × diluted shares"
                    )

        # -------------------------------------------------------------------------
        # 3) Q/Q and Y/Y outlier checks (with actual QoQ/YoY in details)
        # -------------------------------------------------------------------------

        for col in VOL_METRICS:
            if col not in df.columns:
                continue

            series = np.asarray(df[col], dtype=float)

            # Q/Q percentage change
            prev = np.roll(series, 1)
            prev[0] = np.nan
            pct_qoq = safe_div(series - prev, np.abs(prev))

            bad_mask = np.abs(pct_qoq) > QOQ_ABS_CHANGE_THRESHOLD
            for q, bad, val, cur, p in zip(quarters, bad_mask, pct_qoq, series, prev):
                if bad and not np.isnan(val):
                    details = f"qoq_change={val}, current={cur}, prev={p}"
                    add_issue(col, q, "OUTLIER_QOQ_CHANGE", val, details=details)

            # Y/Y percentage change (t vs t-4)
            prev_y = np.roll(series, 4)
            prev_y[:4] = np.nan
            pct_yoy = safe_div(series - prev_y, np.abs(prev_y))

            bad_mask = np.abs(pct_yoy) > YOY_ABS_CHANGE_THRESHOLD
            for q, bad, val, cur, p in zip(quarters, bad_mask, pct_yoy, series, prev_y):
                if bad and not np.isnan(val):
                    details = f"yoy_change={val}, current={cur}, prev_year={p}"
                    add_issue(col, q, "OUTLIER_YOY_CHANGE", val, details=details)

        # -------------------------------------------------------------------------
        # Output
        # -------------------------------------------------------------------------

        errors_df = pd.DataFrame(
            issues,
            columns=["metric", "quarter", "error", "value", "details"]
        )
        return errors_df

    def extract_error_context(
        self,
        errors_df: pd.DataFrame,
        financials_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Given:
          - errors_df with columns ['metric', 'quarter', 'error', 'value', 'details']
          - financials_df indexed by quarter (e.g. '2022Q1'), with raw financial columns

        Return a long-form DataFrame where each row corresponds to an error and
        includes the key financial fields to inspect for that error.
        """
        nan_errors = errors_df[errors_df["error"].isin(['ALL_NANS', 'CONTAINS_NANS'])]
        errors_df  = errors_df[~errors_df["error"].isin(['ALL_NANS', 'CONTAINS_NANS'])]
        # Map (metric, error) -> list of financial columns to show
        CONTEXT_MAP = {
            # Balance sheet identities
            ("bs_assets", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "bs_assets", "bs_total_liabilities", "bs_equity"
            ],
            ("bs_total_liabilities", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "bs_total_liabilities", "bs_current_liabilities", "bs_noncurrent_liabilities_alt"
            ],
            ("bs_noncurrent_liabilities_alt", "ACCOUNTING_IDENTITY_ERROR"): [
                "bs_noncurrent_liabilities", "bs_noncurrent_liabilities_alt"
            ],
            ("bs_property_plant_and_equipment_net", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "bs_property_plant_and_equipment_net", "bs_property_plant_and_equipment_gross"
            ],

            # Income statement identities
            ("is_gross_profit", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "is_revenues", "is_cost_of_revenue", "is_gross_profit"
            ],
            ("is_operating_income_loss", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "is_gross_profit", "is_operating_expenses", "is_operating_income_loss"
            ],
            ("is_net_income_loss", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "is_net_income_loss",
                "is_operating_income_loss",
                "is_net_interest_expense",
                "is_income_tax_expense_benefit",
                "is_noncontrolling_interest_income",
                "is_equity_method_income",
                "crypto_asset_gain_loss",
            ],

            # Cash flow identities
                        ("cf_net_cash_flow", "ACCOUNTING_IDENTITY_MISMATCH"): [
                "cf_net_cash_flow_from_operating_activities",
                "cf_net_cash_flow_from_investing_activities",
                "cf_net_cash_flow_from_financing_activities",
                "cf_net_cash_flow",
                "cf_effect_of_exchange_rate_on_cash_and_cash_equivalents",
            ],
            ("cf_free_cash_flow", "STATEMENT_LINKAGE_MISMATCH"): [
                "cf_net_cash_flow_from_operating_activities",
                "cf_capital_expenditures",
                "cf_free_cash_flow",
            ],
            ("bs_cash_and_cash_equivalents", "STATEMENT_LINKAGE_MISMATCH"): [
                "bs_cash_and_cash_equivalents",
                "cf_net_cash_flow",
            ],

            # Market cap / shares
            ("bs_market_cap_basic", "STATEMENT_LINKAGE_MISMATCH"): [
                "bs_market_cap_basic", "close", "is_basic_average_shares"
            ],
            ("bs_market_cap_diluted", "STATEMENT_LINKAGE_MISMATCH"): [
                "bs_market_cap_diluted", "close", "is_diluted_average_shares"
            ],
            ("is_diluted_average_shares", "STRUCTURAL_RELATION_VIOLATION"): [
                "is_basic_average_shares", "is_diluted_average_shares"
            ],
        }

        # Default columns when there is no specific mapping
        DEFAULT_CONTEXT_COLS = {
            # sign / missingness / ratio anomalies: show the metric itself plus a few anchors
            "SIGN_VIOLATION_STRICT_POSITIVE": lambda m: [m],
            "SIGN_VIOLATION_STRICT_NONPOSITIVE": lambda m: [m],
            "MISSING_EXPECTED_VALUE": lambda m: [m],
            "OUTLIER_QOQ_CHANGE": lambda m: [m],
            "OUTLIER_YOY_CHANGE": lambda m: [m],
            "PROPORTION_OUT_OF_RANGE": lambda m: [m],
            "STRUCTURAL_RELATION_VIOLATION": lambda m: [m],
            "STATEMENT_LINKAGE_MISMATCH": lambda m: [m],
        }

        records = []

        # Make sure financials_df can be indexed by 'quarter' from errors_df
        # Expect financials_df.index to be the same strings like '2022Q1'
        for _, err in errors_df.iterrows():
            q = err["quarter"]
            metric = err["metric"]
            error = err["error"]
            details = err.get("details", None)

            # Skip if quarter is not in financials
            if q not in financials_df.index:
                continue

            # Determine which columns to show
            key = (metric, error)
            if key in CONTEXT_MAP:
                cols = CONTEXT_MAP[key]
            else:
                # fall back to default mapping based on error type
                if error in DEFAULT_CONTEXT_COLS:
                    cols = DEFAULT_CONTEXT_COLS[error](metric)
                else:
                    # last fallback: just show the metric itself
                    cols = [metric]

            # Keep only columns that actually exist
            available_cols = [c for c in cols if c in financials_df.columns]

            # Extract the row for that quarter
            row = financials_df.loc[q, available_cols]

            record = {
                "quarter": q,
                "metric": metric,
                "error": error,
                "details": details,
            }

            # Add the financial values to the record
            for c in available_cols:
                record[c] = row[c]

            records.append(record)

        if not records:
            return pd.DataFrame(columns=["quarter", "metric", "error", "details"]), nan_errors

        # Build resulting DataFrame
        context_df = pd.DataFrame(records)

        # Optional: sort by quarter then error
        context_df = context_df.sort_values(["quarter", "metric", "error"]).reset_index(drop=True)

        return context_df, nan_errors

    def _apply_bucket_aggregates(self, df: pd.DataFrame) -> None:
        """
        For each bucket in BUCKET_METRIC_MAP, compute the signed sum of its
        components and merge into df[metric_name] using combine_first.

        Directly reported metrics always take precedence; the bucket sum fills gaps.
        """
        self._ensure_bucket_components_loaded()
        if not getattr(self, "_bucket_components", None):
            return

        capturing = bool(self._capture_bucket_components)
        self._bucket_component_store = {}

        # E.g., is_revenues, is_revenues_from_contract_with_customer_excl_tax
        for bucket_key, metric_name in BUCKET_METRIC_MAP.items():
            comps = self._bucket_components.get(bucket_key, {})
            if not comps: # if it doesn't exist, skip
                if capturing:
                    self._bucket_component_store[metric_name] = pd.DataFrame(index=df.index)
                continue
            component_frames: Dict[str, pd.Series] = {}
            component_sources: Dict[str, Optional[str]] = {}
            for canonical, sign in comps.items():
                column_name = self._resolve_column_for_canonical(df, canonical)
                component_sources[canonical] = column_name
                if column_name is None:
                    continue
                series = self._num(df[column_name])
                component_frames[canonical] = sign * series

            bucket_series = self._sum_bucket(df, bucket_key)
            if capturing:
                meta_priority = list(self._RAW_META_COLUMNS) + list(self._MERGED_META_COLUMNS) + [
                    "quarter",
                    "fiscal_year",
                    "effective_fiscal_year",
                ]
                seen_meta: Set[str] = set()
                meta_cols = []
                for col in meta_priority:
                    if col in df.columns and col not in seen_meta:
                        seen_meta.add(col)
                        meta_cols.append(col)
                if metric_name in df.columns and metric_name not in meta_cols:
                    meta_cols.append(metric_name)

                base_meta = (
                    df[meta_cols].copy()
                    if meta_cols
                    else pd.DataFrame(index=df.index)
                )
                components_frame = (
                    pd.DataFrame(component_frames, index=df.index)
                    if component_frames
                    else pd.DataFrame(index=df.index)
                )
                comp_df = pd.concat([base_meta, components_frame], axis=1)

                quarterize_meta = self._describe_quarterize_flow_columns(
                    metric_name, component_sources, meta_cols
                )
                comp_df.attrs["quarterize_flow_columns"] = quarterize_meta["present"]
                comp_df.attrs["quarterize_flow_columns_missing"] = quarterize_meta["missing"]
                self._bucket_component_store[metric_name] = comp_df

            if metric_name in df.columns:
                existing = self._num(df[metric_name])
                combined = existing.combine_first(bucket_series)
                df[metric_name] = combined

                self._log(
                    f"[bucket:{bucket_key}] metric={metric_name} "
                    f"direct_rows={(existing.notna()).sum()} "
                    f"bucket_rows={(combined.notna() & existing.isna()).sum()}",
                    level="debug",
                )
            else:
                df[metric_name] = bucket_series

                self._log(
                    f"[bucket:{bucket_key}] metric={metric_name} "
                    f"(no direct column) bucket_rows={(bucket_series.notna()).sum()}",
                    level="debug",
                )

    #### For Finnhub : clean the concept field, then map it

    def _describe_quarterize_flow_columns(
        self,
        metric_name: str,
        component_sources: Dict[str, Optional[str]],
        meta_columns: Sequence[str],
    ) -> Dict[str, List[str]]:
        """
        Build a summary of the concrete columns `_quarterize_flows` must see
        for the provided metric. Includes the metric itself plus each resolved
        component column. Any canonical component that failed to resolve is
        reported under "missing".
        """
        present: List[str] = []
        missing: List[str] = []
        seen: Set[str] = set()

        def _add(name: Optional[str]) -> None:
            if not name:
                return
            if name in seen:
                return
            seen.add(name)
            present.append(name)

        for meta in meta_columns:
            _add(meta)
        _add(metric_name)
        for canonical, resolved in component_sources.items():
            if resolved:
                _add(resolved)
            else:
                missing.append(canonical)

        return {
            "present": present,
            "missing": sorted(set(missing)),
        }

    # ------------------------------------------------------------------ #
    # Fast alias helpers
    # ------------------------------------------------------------------ #

    def _ensure_reverse_map_cached(self, provider: str) -> Dict[str, str]:
        """
        Ensure reverse map is loaded and cached for provider without
        repeatedly hitting the filesystem.
        """
        prov = provider.lower()
        if prov in self._reverse_map_cache:
            self._current_provider = prov
            self._alias_map = self._alias_map_cache.get(prov, self._alias_map)
            self._reverse_map = self._reverse_map_cache[prov]
            return self._reverse_map

        self._load_aliases(prov)
        self._build_reverse_map()
        return self._reverse_map

    def _finnhub_field_series_to_canon(
        self,
        concepts: pd.Series,
        ticker: Optional[str] = None,
        register_unknown: bool = True,
    ) -> pd.Series:
        """
        Vectorized canonical mapping for Finnhub concept Series.
        Mirrors `_finnhub_field_to_canon` logic but in batch.
        """
        import re

        self._ensure_reverse_map_cached("finn")
        reverse_map = self._reverse_map

        ticker_upper = (ticker or "").upper()
        ticker_lower = ticker_upper.lower()

        def _normalize_finnhub_field_name(field_name: str) -> str:
            pattern_colon = r"^[a-zA-Z]+:"
            cleaned = re.sub(pattern_colon, "", field_name)
            if ticker_upper:
                if cleaned.startswith(ticker_upper + ":") or cleaned.startswith(ticker_lower + ":"):
                    cleaned = cleaned.split(":", 1)[-1]
                cleaned = re.sub(
                    f"^{re.escape(ticker_upper)}_|^{re.escape(ticker_lower)}_",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
            cleaned = re.sub(r"^us[-_]gaap_", "", cleaned, flags=re.IGNORECASE)
            return cleaned

        ser = concepts.fillna("").astype(str)
        clean = ser.map(_normalize_finnhub_field_name)

        explicit = clean.map(FINNHUB_EXPLICIT_MAP)
        out = explicit.copy()

        norm_raw = clean.map(self._normalize_name)
        out = out.fillna(norm_raw.map(reverse_map))

        snake = clean.map(self._to_snake_case)
        norm_snake = snake.map(self._normalize_name)
        out = out.fillna(norm_snake.map(reverse_map))

        unknown_mask = out.isna()
        out.loc[unknown_mask] = snake.loc[unknown_mask]

        if register_unknown and self.register_unrecognized_fields:
            if not hasattr(self, "unrecognised_fields"):
                self.unrecognised_fields = {}
            unique_unknowns = norm_snake.loc[unknown_mask].unique().tolist()
            for unk in unique_unknowns:
                raw_candidates = clean.loc[norm_snake == unk]
                raw_first = raw_candidates.iloc[0] if not raw_candidates.empty else unk
                try:
                    self._register_unrecognized_field(unk, raw_first)
                except Exception:
                    pass
                placeholder = snake.loc[norm_snake == unk].iloc[0]
                self.unrecognised_fields[placeholder] = placeholder

        return out

    def _finnhub_field_to_canon(
        self,
        concept: str,
        ticker: Optional[str] = None,
        register_unknown: bool = True,
    ):
        import re
        self._ensure_reverse_map_cached("finn")
        def normalize_finnhub_field_name(field_name: str) -> str:
            """Remove ticker prefix from field name (both : and _ formats)."""
            # Handle colon format (e.g., "ADI:NonCashOperatingLeaseCosts")
            pattern_colon = r'^[a-zA-Z]+:'
            cleaned = re.sub(pattern_colon, '', field_name)
            
            # Remove ticker prefix if ticker is provided (more precise than regex)
            if ticker:
                ticker_upper = ticker.upper()
                ticker_lower = ticker.lower()
                # Remove ticker prefix with colon
                if cleaned.startswith(ticker_upper + ":") or cleaned.startswith(ticker_lower + ":"):
                    cleaned = cleaned.split(":", 1)[-1]
            # Remove ticker prefix with underscore (case-insensitive)
            ticker_pattern = f"^{re.escape(ticker_upper)}_|^{re.escape(ticker_lower)}_"
            cleaned = re.sub(ticker_pattern, '', cleaned, flags=re.IGNORECASE)
            # Remove us-gaap/us_gaap prefixes (case-insensitive)
            cleaned = re.sub(r'^us[-_]gaap_', '', cleaned, flags=re.IGNORECASE)
            return cleaned

        def _map_finnhub_field(raw: str) -> str:
     
            # 1) direct explicit map shortcut (exact GAAP tag -> canonical)
            if raw in FINNHUB_EXPLICIT_MAP:
                self._log(f"{raw} is in FINNHUB_EXPLICIT_MAP", level="debug")
                return FINNHUB_EXPLICIT_MAP[raw]

            # 2) normalized lookups against reverse_map
            norm_raw = self._normalize_name(raw)
            if norm_raw in self._reverse_map:
                self._log(f"{raw} is in reverse map (normalized raw)", level="debug")
                return self._reverse_map[norm_raw]

            snake = self._to_snake_case(raw)
            norm_snake = self._normalize_name(snake)
            if norm_snake in self._reverse_map:
                self._log(f"{norm_snake} is in reverse map (normalized snake)", level="debug")
                return self._reverse_map[norm_snake]

            # 3) fallback: mark unrecognized and return snake_case placeholder
            placeholder = f"{snake}"
            self._log(f"{placeholder} is not recognised, registering unrecognized field", level="debug")
            if register_unknown and self.register_unrecognized_fields:
                self._register_unrecognized_field(norm_snake, raw)
                self.unrecognised_fields[placeholder] = snake
            return placeholder

        clean = normalize_finnhub_field_name(concept)
        canon = _map_finnhub_field(clean)

        return canon
    
    # --------------------------------------------------------------------- #
    # Quarterly Financials Builder
    # --------------------------------------------------------------------- #
    _FY_FORMS = {"10-K", "20-F", "40-F"}
    _Q_FORMS  = {"10-Q", "6-K"}

    _FULL_YEAR_MIN_DAYS = 300   # ~10+ months
    _YTD_MIN_DAYS       = 120   # >= ~4 months
    _FY_START_TOLERANCE = 7     # days

    
    # --------------------------------------------------------------------- #
    # Low-level helpers
    # --------------------------------------------------------------------- #

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")
    
    @staticmethod
    def _statement_from_quarterised(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract all financial statements from quarterised financials dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Quarterised financials dataframe with period index
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple of (is_statement, balance_sheet, cashflow, misc_df)
            All statements are transposed with metrics as rows and periods as columns
        """
        df = df.copy()
        
        def create_is_from_quarterised(df: pd.DataFrame) -> pd.DataFrame:
            is_statement_cols = [
                "end_date",
                "is_revenues",
                "is_sales_revenue_net",
                
                "is_cost_of_revenue",
                "is_gross_profit",

                "is_research_and_development",
                "is_selling_general_and_administrative_expenses",
                "is_depreciation_and_amortization",
                "is_operating_expense_residual",
                "is_operating_expenses",
                "is_operating_expense_alt",
                "is_operating_income_loss",

                "is_net_interest_expense",
                "cf_net_interest_paid",

                "is_income_tax_expense_benefit",
                "is_net_income_loss",
                
                "is_diluted_average_shares",
                "is_basic_average_shares",
                
                "eps_basic",
                "eps_diluted",
                "bs_market_cap_diluted",
                "bs_market_cap_basic",
                
                'cf_capital_expenditures',
                'cf_adjusted_capex',
                'cf_free_cash_flow',
            ]
            existing_cols = [c for c in is_statement_cols if c in df.columns]
            is_statement = df[existing_cols].T
            return is_statement
        
        def create_balsheet_from_quarterised(df: pd.DataFrame) -> pd.DataFrame:
            b_cols = ['end_date', 'earnings_release_date',
                    'bs_assets', 'bs_total_liabilities',
                    'bs_equity', 'bs_current_liabilities',
                    "bs_noncurrent_liabilities",
                    "bs_noncurrent_liabilities_alt",
                    'bs_accounts_payable', 
                    'bs_current_assets', 'bs_goodwill',
                    "bs_marketable_securities",
                    'bs_property_plant_and_equipment',
                    'bs_cash_and_cash_equivalents', 'bs_short_term_investments',
                    'bs_cash_and_equivalents_short_term_investments',
                    'bs_cash_and_cash_equivalents_restricted',
                    'bs_deferred_revenue',
                    "bs_inventory",
                    "bs_finance_lease_liability",
                    "bs_operating_lease_liability",
                    'bs_noncurrent_liabilities',
                    'bs_interest_bearing_debt',
                    'bs_net_debt',
                    'bs_net_property_plant_and_equipment',
                    'bs_noncontrolling_interest_equity']
            cols = [c for c in b_cols if c in df.columns]
            df = df[cols].copy()
            return df
        
        def create_cashflow_from_quarterised(df: pd.DataFrame) -> pd.DataFrame:
            cf_cols = [
                    'cf_net_cash_flow',
                    'cf_net_cash_flow_from_operating_activities',
                    'cf_net_cash_flow_from_financing_activities',
                    'cf_net_cash_flow_from_investing_activities',
                    'cf_noncontrolling_interest_cash_flow',
                    'cf_equity_method_cash_flow',
                    'cf_capital_expenditures',
                    'is_depreciation_and_amortization',
                    "cf_net_interest_paid",
                    'cf_free_cash_flow',
                    "cf_net_cash_flow_including_restricted_cash_and_cash_equivalents",
                    'cf_effect_of_exchange_rate_on_cash_and_cash_equivalents',
                    "cf_effect_of_exchange_rate_on_cash_cash_equivalents_restricted_cash_and_restricted_cash_equivalents"
                    ]
            cf_cols_ = [c for c in cf_cols if c in df.columns]
            cashflow = df[cf_cols_].copy()
            cashflow = cashflow.T
            return cashflow

        balance_sheet = create_balsheet_from_quarterised(df).T
        cashflow = create_cashflow_from_quarterised(df)
        is_statement = create_is_from_quarterised(df)

        cols_used = balance_sheet.index.tolist() + cashflow.index.tolist() + is_statement.index.tolist()
        cols_leftover = df.columns.difference(cols_used)
        misc_df = df[cols_leftover].copy().dropna(axis=1, how='all')
        return is_statement, balance_sheet, cashflow, misc_df
    
    def _normalize_provider(self, provider: Optional[str]) -> str:
        """Return canonical provider key."""
        key = (provider or self.default_financials_provider or "polygon").lower()
        if key in ("polygon", "poly"):
            return "polygon"
        if key in ("finnhub", "finn"):
            return "finnhub"
        if key in ("sec", "sec.gov"):
            return "sec"
        raise ValueError(f"Unknown financials provider '{provider}'. Use 'polygon', 'finnhub', or 'sec'.")

    def _get_fundamentals_dir_for_provider(self, provider: str) -> Optional[str]:
        if provider == "polygon":
            return self.fundamentals_dir
        if provider == "finnhub":
            return self.finnhub_fundamentals_dir
        if provider == "sec":
            return None
        return None

    def _load_financials_for_provider(self, ticker: str, provider: str) -> pd.DataFrame:
        provider = self._normalize_provider(provider)
        if provider == "polygon":
            return self._load_polygon_financials(ticker)
        if provider == "finnhub":
            return self._load_finnhub_financials(ticker)
        raise ValueError(f"Unsupported provider '{provider}'")

    def _load_financials_for_provider_multi(
        self,
        primary_ticker: str,
        provider: str,
        alias_tickers: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Load fundamentals for a primary ticker plus any equivalent tickers,
        standardize them to a single ticker symbol, and merge into one frame.
        """
        aliases = [str(t).strip().upper() for t in (alias_tickers or []) if str(t).strip()]
        base = str(primary_ticker).strip().upper()
        tickers = [base] + [t for t in aliases if t and t != base]

        frames: List[pd.DataFrame] = []
        for tk in tickers:
            try:
                df = self._load_financials_for_provider(tk, provider=provider)
            except Exception as exc:
                self._log(f"{provider} load failed for {tk}: {exc}", level="warning")
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["source_ticker"] = df.get("ticker", tk)
            df["ticker"] = base
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        sort_cols = [col for col in ("end_date", "fiscal_period", "filing_date") if col in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, na_position="last")
        dedup_cols = [col for col in ("ticker", "source_ticker", "end_date", "fiscal_period", "timeframe") if col in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        return combined.reset_index(drop=True)

    def _load_polygon_financials(self, ticker: str) -> pd.DataFrame:
        cached = None
        if self.fundamentals_dir:
            fpath = Path(self.fundamentals_dir) / f"{ticker}_fundamentals.parquet"
            if fpath.exists() and not self.force_update:
                cached = pd.read_parquet(fpath)
        if cached is not None:
            return cached
        return self.fetch_quarterly_financials(ticker)

    def _load_finnhub_financials(self, ticker: str) -> pd.DataFrame:
        return self.get_finnhub_financials_unified(
            ticker,
            start_date=self.min_end_date,
            end_date=self.max_end_date,
            wide=True,
        )

    def _fetch_finnhub_annual_financials(self, ticker: str) -> pd.DataFrame:
        annual_dict = self.fetch_finnhub_statements(
            ticker,
            start_date=self.min_end_date,
            end_date=self.max_end_date,
            freq="annual",
        )
        merged = self._merge_finnhub_statements(annual_dict)
        if merged is None or merged.empty:
            return pd.DataFrame()
        standardized = self._standardize_finnhub_meta(merged, timeframe="annual")
        return standardized

    # --------------------------------------------------------------------- #
    # Financials fetch : SEC Company Facts
    # --------------------------------------------------------------------- #

    def _sec_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _load_sec_cik_cache(self) -> Dict[str, int]:
        path = Path(self.sec_cik_cache_path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return {k.upper(): int(v) for k, v in data.items()}
            except Exception:
                return {}
        return {}

    def _save_sec_cik_cache(self, mapping: Dict[str, int]) -> None:
        path = Path(self.sec_cik_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = {k: int(v) for k, v in mapping.items()}
            path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def _lookup_cik(self, ticker: str) -> Optional[int]:
        cache = self._load_sec_cik_cache()
        t = (ticker or "").upper().strip()
        if not t:
            return None
        if t in cache:
            return cache[t]
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            data = self._get_json(url, headers=self._sec_headers())
        except Exception as exc:
            self._log(f"SEC ticker lookup failed for {t}: {exc}", level="warning")
            return None
        mapping: Dict[str, int] = {}
        for entry in (data or {}).values():
            tick = str(entry.get("ticker", "")).upper()
            cik_str = entry.get("cik_str")
            try:
                cik_val = int(cik_str)
            except Exception:
                continue
            if tick:
                mapping[tick] = cik_val
        if mapping:
            cache.update(mapping)
            self._save_sec_cik_cache(cache)
        return cache.get(t)

    def _fetch_sec_company_facts(self, ticker: str) -> Dict[str, Any]:
        cik = self._lookup_cik(ticker)
        if not cik:
            return {}
        cik_str = f"{int(cik):010d}"
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        try:
            return self._get_json(url, headers=self._sec_headers())
        except Exception as exc:
            self._log(f"SEC company facts fetch failed for {ticker}: {exc}", level="warning")
            return {}

    def fetch_sec_financials(
        self,
        ticker: str,
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
        save: bool = True,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch SEC company facts, normalize to canonical field names, and return a wide frame
        compatible with the Polygon/Finnhub loaders (meta + metric columns).
        """
        ticker = (ticker or "").upper()
        if not ticker:
            return pd.DataFrame()

        base_dir = Path(self.finnhub_fundamentals_dir or self.fundamentals_dir or ".").parent
        sec_dir = Path(base_dir) / "SEC_fundamentals"
        sec_dir.mkdir(parents=True, exist_ok=True)
        raw_path = sec_dir / f"{ticker}_SEC_raw.parquet"
        cached_raw_rows: List[Dict[str, Any]] = []

        if not force_update and raw_path.exists():
            try:
                cached_raw_rows = pd.read_parquet(raw_path).to_dict(orient="records")
            except Exception as exc:
                self._log(f"Failed to load cached SEC raw parquet for {ticker}: {exc}", level="warning")

        raw_rows: List[Dict[str, Any]] = []
        if cached_raw_rows:
            raw_rows = cached_raw_rows
        else:
            raw = self._fetch_sec_company_facts(ticker)
            facts = (raw.get("facts") or {}).get("us-gaap", {})
            if not facts:
                return pd.DataFrame()
            for concept, payload in facts.items():
                units = payload.get("units") or {}
                for unit, filings in units.items():
                    for entry in filings:
                        # Keep the full SEC payload for caching; entry may contain extra keys (accn, frame, segment, etc.)
                        entry_copy = dict(entry)
                        entry_copy.update(
                            {
                                "ticker": ticker,
                                "concept_raw": concept,
                                "unit": unit,
                            }
                        )
                        raw_rows.append(entry_copy)
            try:
                pd.json_normalize(raw_rows, sep="__").to_parquet(raw_path, index=False)
            except Exception as exc:
                self._log(f"Failed to save SEC raw parquet for {ticker}: {exc}", level="warning")

        records: List[Dict[str, Any]] = []
        raw_records: List[Dict[str, Any]] = []
        for entry in raw_rows:
            concept = entry.get("concept_raw")
            unit = entry.get("unit")
            val = entry.get("val")
            if val is None:
                continue
            fy = entry.get("fy")
            fp = entry.get("fp")
            end_date = entry.get("end")
            filing_date = entry.get("filed") or entry.get("filing_date")
            if start_dt and end_dt and end_date:
                try:
                    end_ts = pd.to_datetime(end_date)
                    if (pd.to_datetime(start_dt) > end_ts) or (pd.to_datetime(end_dt) < end_ts):
                        continue
                except Exception:
                    pass
            prefixed = self._canonical_to_prefixed_metric(
                self._normalize_canonical_name(
                    self._finnhub_field_to_canon(concept, ticker=ticker, register_unknown=False)
                )
            )
            raw_records.append(
                {
                    "ticker": ticker,
                    "concept_raw": concept,
                    "unit": unit,
                    "value": val,
                    "start_date": entry.get("start"),
                    "end_date": end_date,
                    "filing_date": filing_date,
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "form": entry.get("form"),
                }
            )
            records.append(
                {
                    "ticker": ticker,
                    "concept_raw": concept,
                    "concept": prefixed,
                    "unit": unit,
                    "value": val,
                    "start_date": entry.get("start"),
                    "end_date": end_date,
                    "filing_date": filing_date,
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "form": entry.get("form"),
                }
            )

        if not records:
            return pd.DataFrame()

        df_long = pd.DataFrame(records)
        df_long["end_date"] = pd.to_datetime(df_long["end_date"], errors="coerce")
        df_long["start_date"] = pd.to_datetime(df_long["start_date"], errors="coerce")
        df_long["filing_date"] = pd.to_datetime(df_long["filing_date"], errors="coerce")
        df_long["fiscal_year"] = pd.to_numeric(df_long["fiscal_year"], errors="coerce").astype("Int64")

        # Determine timeframe and clean fiscal period
        df_long["fiscal_period"] = df_long["fiscal_period"].fillna("")
        df_long["fiscal_period"] = df_long["fiscal_period"].str.replace("FY", "Q4")
        df_long["timeframe"] = np.where(
            df_long["fiscal_period"].str.upper().eq("Q4"),
            "annual",
            "quarterly",
        )

        meta_cols = [
            "ticker",
            "start_date",
            "end_date",
            "filing_date",
            "fiscal_year",
            "fiscal_period",
            "timeframe",
            "form",
        ]
        pivot = df_long.pivot_table(
            index=meta_cols,
            columns="concept",
            values="value",
            aggfunc="first",
        ).reset_index()

        metric_cols = [c for c in pivot.columns if c not in meta_cols]
        pivot[metric_cols] = pivot[metric_cols].apply(pd.to_numeric, errors="coerce")
        pivot = pivot.sort_values("end_date").reset_index(drop=True)

        if save:
            try:
                base_dir = Path(self.finnhub_fundamentals_dir or self.fundamentals_dir or ".").parent
                sec_dir = Path(base_dir) / "SEC_fundamentals"
                sec_dir.mkdir(parents=True, exist_ok=True)
                raw_df = pd.DataFrame(raw_records if raw_records else records)
                raw_df.to_parquet(sec_dir / f"{ticker.upper()}_SEC.parquet", index=False)
            except Exception as exc:
                self._log(f"Failed to save SEC raw parquet for {ticker}: {exc}", level="warning")

        return pivot

    def _load_sec_financials(self, ticker: str, save: bool = False) -> pd.DataFrame:
        return self.fetch_sec_financials(
            ticker,
            start_dt=self.min_end_date,
            end_dt=self.max_end_date,
            save=save,
        )

    # --------------------------------------------------------------------- #
    # Financials fetch (quarterly + annual) : Poly
    # --------------------------------------------------------------------- #

    def fetch_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        """Fetch all quarterly financials and flatten sections."""
        self._load_aliases(provider="poly")
        url = f"{self.base_url}/vX/reference/financials"
        params = {
            "ticker": ticker,
            "timeframe": "quarterly",
            "sort": "period_of_report_date",
            "order": "asc",
            "limit": 100,
            "apiKey": self.api_key,
        }
        rows: List[Dict[str, Any]] = []
        first = True
        while url:
            print("Fetching")
            data = self._get_json(url, params=params if first else None)
            first = False
            for item in data.get("results", []):
                fin = item.get("financials", {}) or {}
                row: Dict[str, Any] = {
                    "ticker": (item.get("tickers") or [ticker])[0],
                    "fiscal_year": item.get("fiscal_year"),
                    "fiscal_period": item.get("fiscal_period"),
                    "timeframe": item.get("timeframe"),
                    "filing_date": item.get("filing_date"),
                    "start_date": item.get("start_date"),
                    "end_date": item.get("end_date"),
                }
                isec = fin.get("income_statement") or {}
                bsec = fin.get("balance_sheet") or {}
                cfsec = fin.get("cash_flow_statement") or {}
                cisec = fin.get("comprehensive_income") or {}

                ticker_value = row.get("ticker")
                is_flat = self._flatten_section(isec, "is_", ticker=ticker_value)
                bs_flat = self._flatten_section(bsec, "bs_", ticker=ticker_value)
                cf_flat = self._flatten_section(cfsec, "cf_", ticker=ticker_value)
                ci_flat = self._flatten_section(cisec, "ci_", ticker=ticker_value)

                if is_flat: row.update(is_flat)
                if bs_flat: row.update(bs_flat)
                if cf_flat: row.update(cf_flat)
                if ci_flat: row.update(ci_flat)
                rows.append(row)

            url = data.get("next_url")

        if not rows:
            return pd.DataFrame()
        self.json_data = data
        df = pd.DataFrame(rows)
        self.metrics_raw = df
        for col in ["filing_date", "start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        stmt_cols = [c for c in df.columns if c.startswith(("is_", "bs_", "cf_", "ci_"))]
        df[stmt_cols] = df[stmt_cols].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("end_date").reset_index(drop=True)
        return df

    def _fetch_annual_financials(self, ticker: str) -> pd.DataFrame:
        """Fetch annual financials (used only for Q4 filing_date patching)."""
        self._load_aliases(provider="poly")
        url = f"{self.base_url}/vX/reference/financials"
        params = {
            "ticker": ticker,
            "timeframe": "annual",
            "sort": "period_of_report_date",
            "order": "asc",
            "limit": 100,
            "apiKey": self.api_key,
        }
        rows: List[Dict[str, Any]] = []
        first = True
        while url:
            data = self._get_json(url, params=params if first else None)
            first = False
            for item in data.get("results", []):
                fin = item.get("financials", {}) or {}
                row: Dict[str, Any] = {
                    "ticker": (item.get("tickers") or [ticker])[0],
                    "fiscal_year": item.get("fiscal_year"),
                    "fiscal_period": item.get("fiscal_period"),
                    "timeframe": item.get("timeframe"),
                    "filing_date": item.get("filing_date"),
                    "start_date": item.get("start_date"),
                    "end_date": item.get("end_date"),
                }
                isec = fin.get("income_statement") or {}
                bsec = fin.get("balance_sheet") or {}
                cfsec = fin.get("cash_flow_statement") or {}
                cisec = fin.get("comprehensive_income") or {}

                ticker_value = row.get("ticker")
                is_flat = self._flatten_section(isec, "is_", ticker=ticker_value)
                bs_flat = self._flatten_section(bsec, "bs_", ticker=ticker_value)
                cf_flat = self._flatten_section(cfsec, "cf_", ticker=ticker_value)
                ci_flat = self._flatten_section(cisec, "ci_", ticker=ticker_value)
                
                if is_flat: row.update(is_flat)
                if bs_flat: row.update(bs_flat)
                if cf_flat: row.update(cf_flat)
                if ci_flat: row.update(ci_flat)
                rows.append(row)
            url = data.get("next_url")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ["filing_date", "start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        stmt_cols = [c for c in df.columns if c.startswith(("is_", "bs_", "cf_", "ci_"))]
        df[stmt_cols] = df[stmt_cols].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("end_date").reset_index(drop=True)
        return df

    def _fill_q4_filing_from_annual(self, fins: pd.DataFrame, ticker: str, provider: str = "polygon") -> pd.DataFrame:
        """If Q4 quarterly rows have NaT filing_date, patch from annual filings for the chosen provider."""
        if fins.empty:
            print(f"{ticker}: Returning fins - Input dataframe is empty")
            return fins

        q4_mask = (fins["fiscal_period"] == "Q4") 
        missing = q4_mask & fins["filing_date"].isna()
        if not missing.any():
            print(f"{ticker}: Returning fins - No Q4 rows with missing filing_date found")
            return fins

        provider_key = self._normalize_provider(provider)
        self._log(f"{ticker}: Q4 quarterly rows with missing filing_date → fetch annual ({provider_key})")
        if provider_key == "polygon":
            annual = self._fetch_annual_financials(ticker)
        else:
            annual = self._fetch_finnhub_annual_financials(ticker)
        if annual.empty:
            self._log(f"{ticker}: annual financials empty, cannot patch Q4 filing_date")
            print(f"{ticker}: Returning fins - Annual financials are empty, cannot patch Q4 filing_date")
            return fins

        for idx in fins[missing].index:
            fy = fins.loc[idx, "fiscal_year"]
            fins.loc[idx, "timeframe"] = "annual"  # mark as annual after patch
            cand = annual[annual["fiscal_year"] == fy]
            if cand.empty:
                end = fins.loc[idx, "end_date"]
                cand = annual.iloc[[annual["end_date"].sub(end).abs().argmin()]]
            fins.at[idx, "filing_date"] = cand["filing_date"].iloc[0]

        print(f"{ticker}: Returning fins - Successfully patched {missing.sum()} Q4 filing_date(s) from annual data")
        return fins

    # --------------------------------------------------------------------- #
    # Financials fetch (quarterly + annual) : Finnhub
    # --------------------------------------------------------------------- #

    def _parse_finnhub_statement(
        self, 
        filing: Dict[str, Any], 
        statement_type: str = 'ic', 
        pivot: bool = True
    ) -> pd.DataFrame:
        """
        Parse a single statement type from a Finnhub filing.
        
        Parameters:
        -----------
        filing : dict
            Single filing from Finnhub API response
        statement_type : str
            'ic' (income), 'bs' (balance sheet), or 'cf' (cash flow)
        pivot : bool
            If True, return wide format with concepts as columns
        
        Returns:
        --------
        pd.DataFrame : Parsed statement data
        """
        # Extract metadata
        metadata = {
            'symbol': filing.get('symbol'),
            'year': filing.get('year'),
            'quarter': filing.get('quarter'),
            'form': filing.get('form'),
            'start_date': filing.get('startDate'),
            'end_date': filing.get('endDate'),
            'filing_date': filing.get('filedDate'),
            'accepted_date': filing.get('acceptedDate'),
            'access_number': filing.get('accessNumber')
        }
        
        # Get the report data
        report = filing.get('report', {})
        statement_data = report.get(statement_type, {})
        
        if not statement_data:
            # Return empty DataFrame with metadata columns
            return pd.DataFrame([metadata])
        
        # Parse line items
        rows = []
        if statement_type == 'ic':
            prefix = "is"
        else:
            prefix = statement_type
        ticker = metadata.get('symbol')
        fast_alias = getattr(self, "_fast_fetch", False)
        if fast_alias:
            lines_df = pd.DataFrame(statement_data)
            if lines_df.empty:
                return pd.DataFrame([metadata])

            raw_concepts = lines_df.get('concept', pd.Series(dtype=str)).copy()
            canon = self._finnhub_field_series_to_canon(
                raw_concepts,
                ticker=ticker,
                register_unknown=self.register_unrecognized_fields,
            )
            lines_df = lines_df.copy()
            lines_df['concept'] = f"{prefix}_" + canon.astype(str)
            if self.debug:
                lines_df['concept_raw'] = raw_concepts
            meta_df = pd.DataFrame([metadata] * len(lines_df))
            cols = ['label', 'concept', 'unit', 'value']
            if self.debug:
                cols.append('concept_raw')
            df = pd.concat(
                [meta_df.reset_index(drop=True), lines_df.reindex(columns=cols).reset_index(drop=True)],
                axis=1,
            )
        else:
            for line_item in statement_data:
                row = metadata.copy()
                row['label'] = line_item.get('label')
                row['concept'] = f'{prefix}_' + self._finnhub_field_to_canon(line_item.get('concept'), ticker=ticker)
                row['unit'] = line_item.get('unit')
                row['value'] = line_item.get('value')
                if self.debug:
                    row['concept_raw'] = line_item.get('concept')
                rows.append(row)
            df = pd.DataFrame(rows)

        if pivot and not df.empty:
            # Pivot to wide format: one row per filing, columns are concepts
            df_pivot = df.pivot_table(
                index=['symbol', 'end_date', 'filing_date', 'year', 'quarter', 
                       'form', 'start_date', 'accepted_date', 'access_number'],
                columns='concept',
                values='value',
                aggfunc='first'
            ).reset_index()
            if self.debug:
                if hasattr(self, 'raw_finnhub'):
                    self.raw_finnhub.append(df)
                else:
                    self.raw_finnhub = []
                    self.raw_finnhub.append(df)
            return df_pivot
        
        return df

    def _parse_finnhub_financials(
        self, 
        api_response: Dict[str, Any], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        pivot: bool = True
    ) -> Dict[str, Any]:
        """
        Parse Finnhub financials API response into 3 separate DataFrames.
        Returns:
        --------
        dict : {
            'income_statement': pd.DataFrame,
            'balance_sheet': pd.DataFrame,
            'cash_flow': pd.DataFrame,
            'metadata': dict with summary info
        }
        """
        data = api_response.get('data', [])
        
        if not data:
            print('Data not found : Returning empty dataframe')
            empty_df = pd.DataFrame()
            return {
                'income_statement': empty_df,
                'balance_sheet': empty_df,
                'cash_flow': empty_df,
                'metadata': {'count': 0, 'symbol': None}
            }
        
        # Parse each statement type
        ic_dfs = []
        bs_dfs = []
        cf_dfs = []
        
        for filing in data:
            # Apply date filters
            filing_end_date = filing.get('filedDate', '')
            if start_date and filing_end_date < start_date:
                continue
            if end_date and filing_end_date > end_date:
                continue
            # Parse each statement type
            ic_df = self._parse_finnhub_statement(filing, 'ic', pivot=pivot)
            bs_df = self._parse_finnhub_statement(filing, 'bs', pivot=pivot)
            cf_df = self._parse_finnhub_statement(filing, 'cf', pivot=pivot)
            
            if not ic_df.empty:
                ic_dfs.append(ic_df)
            if not bs_df.empty:
                bs_dfs.append(bs_df)
            if not cf_df.empty:
                cf_dfs.append(cf_df)
        
        # Combine all filings
        income_statement = pd.concat(ic_dfs, ignore_index=True) if ic_dfs else pd.DataFrame()
        balance_sheet = pd.concat(bs_dfs, ignore_index=True) if bs_dfs else pd.DataFrame()
        cash_flow = pd.concat(cf_dfs, ignore_index=True) if cf_dfs else pd.DataFrame()
        
        # Sort by end_date
        for df in [income_statement, balance_sheet, cash_flow]:
            if not df.empty and 'end_date' in df.columns:
                df.sort_values('end_date', inplace=True)
                df.reset_index(drop=True, inplace=True)
        
        return {
            'income_statement': income_statement,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'metadata': {
                'count': len(ic_dfs),
                'symbol': data[0].get('symbol') if data else None,
                'date_range': {
                    'start': income_statement['end_date'].min() if not income_statement.empty else None,
                    'end': income_statement['end_date'].max() if not income_statement.empty else None
                }
            }
        }

    def _finnhub_raw_cache_path(self, symbol: str) -> Optional[Path]:
        if not self.finnhub_fundamentals_dir:
            return None
        base = Path(self.finnhub_fundamentals_dir)
        base.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.upper().replace("/", "_")
        return base / f"{safe_symbol}_raw.parquet"

    def _legacy_finnhub_raw_cache_path(self, symbol: str, freq: str) -> Optional[Path]:
        if not self.finnhub_fundamentals_dir:
            return None
        base = Path(self.finnhub_fundamentals_dir)
        base.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.upper().replace("/", "_")
        return base / f"{safe_symbol}_{freq}_raw.parquet"

    def _load_finnhub_raw_cache(self, symbol: str, freq: str) -> Optional[Dict[str, Any]]:
        cache_path = self._finnhub_raw_cache_path(symbol)
        if cache_path and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty and {"freq", "raw_json"}.issubset(df.columns):
                    match = df[df["freq"] == freq]
                    if not match.empty:
                        if "fetched_at" in match.columns:
                            match = match.sort_values("fetched_at")
                        return json.loads(match.iloc[-1]["raw_json"])
            except Exception as exc:
                self._log(f"Failed to read Finnhub cache for {symbol} ({freq}): {exc}", level="warning")
        legacy_path = self._legacy_finnhub_raw_cache_path(symbol, freq)
        if legacy_path and legacy_path.exists():
            try:
                df = pd.read_parquet(legacy_path)
                if not df.empty and "raw_json" in df.columns:
                    return json.loads(df.iloc[0]["raw_json"])
            except Exception as exc:
                self._log(
                    f"Failed to read legacy Finnhub cache for {symbol} ({freq}): {exc}",
                    level="warning",
                )
        return None

    def _save_finnhub_raw_cache(self, symbol: str, freq: str, payload: Dict[str, Any]) -> None:
        cache_path = self._finnhub_raw_cache_path(symbol)
        if not cache_path:
            return
        try:
            columns = ["freq", "raw_json", "fetched_at"]
            existing = pd.DataFrame(columns=columns)
            if cache_path.exists():
                existing = pd.read_parquet(cache_path)
                if columns[0] not in existing.columns:
                    existing = pd.DataFrame(columns=columns)
            if not existing.empty:
                existing = existing[existing["freq"] != freq]
            new_row = pd.DataFrame(
                [
                    {
                        "freq": freq,
                        "raw_json": json.dumps(payload),
                        "fetched_at": pd.Timestamp.utcnow(),
                    }
                ]
            )
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined.to_parquet(cache_path, index=False)
            legacy_path = self._legacy_finnhub_raw_cache_path(symbol, freq)
            if legacy_path and legacy_path.exists():
                legacy_path.unlink(missing_ok=True)
            self._log(f"Cached Finnhub {freq} payload for {symbol} -> {cache_path}")
        except Exception as exc:
            self._log(f"Failed to write Finnhub cache for {symbol} ({freq}): {exc}", level="warning")

    def _fetch_finnhub_raw(self, symbol: str, freq: str) -> Dict[str, Any]:
        url = "https://finnhub.io/api/v1/stock/financials-reported"
        params = {
            "symbol": symbol,
            "freq": freq,
            "token": self.finn_api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_finnhub_raw_response(self, symbol: str, freq: str) -> Dict[str, Any]:
        raw: Optional[Dict[str, Any]] = None
        if not self.force_update:
            raw = self._load_finnhub_raw_cache(symbol, freq)
        if raw is None:
            raw = self._fetch_finnhub_raw(symbol, freq)
            self._save_finnhub_raw_cache(symbol, freq, raw)
        self.api_response_finn = raw
        return raw

    def fetch_finnhub_statements(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "quarterly",
        finn_jason_save : bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch and parse Finnhub financials, returning dict with 3 DataFrames.
        Finnhub API doesn't support date filtering at API level.
        All filings are fetched, then filtered by start_date/end_date in parsing.
        """
        api_response = self._get_finnhub_raw_response(symbol, freq)
        self.api_response_finn = api_response if finn_jason_save else None
        # Parse into structured DataFrames (filtering happens here)
        parsed = self._parse_finnhub_financials(api_response, start_date, end_date, pivot=True)
        
        return parsed

    # ------------------------------
    # Finnhub unified helpers
    # ------------------------------
    def _merge_finnhub_statements(self, finnhub_statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge Finnhub income, balance, and cash flow wide frames on shared metadata."""
        is_df = finnhub_statements.get('income_statement', pd.DataFrame())
        bs_df = finnhub_statements.get('balance_sheet', pd.DataFrame())
        cf_df = finnhub_statements.get('cash_flow', pd.DataFrame())

        if is_df.empty and bs_df.empty and cf_df.empty:
            return pd.DataFrame()

        merge_cols = [
            'symbol', 'end_date', 'filing_date', 'year', 'quarter', 'form',
            'start_date', 'accepted_date', 'access_number'
        ]

        base = None
        for part in (is_df, bs_df, cf_df):
            if part is not None and not part.empty:
                base = part if base is None else base.merge(part, on=merge_cols, how='outer')
        return base if base is not None else pd.DataFrame()

    def _standardize_finnhub_meta(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Make Finnhub meta columns consistent with Polygon naming."""
        if df.empty:
            return df

        out = df.copy()
        # Rename to Polygon-style
        rename_map = {
            'symbol': 'ticker',
            'year': 'fiscal_year',
        }
        out = out.rename(columns=rename_map)

        # Fiscal period formatting
        if 'quarter' in out.columns:
            # quarter may be NaN for annual
            qp = out['quarter']
            out['fiscal_period'] = np.where(qp.notna(), 'Q' + qp.astype('Int64').astype(str), 'FY')
        else:
            out['fiscal_period'] = 'FY' if timeframe == 'annual' else None

        out['timeframe'] = timeframe

        # Ensure expected Polygon meta columns exist
        for col in ['ticker', 'start_date', 'end_date', 'filing_date', 'fiscal_year', 'fiscal_period', 'timeframe']:
            if col not in out.columns:
                out[col] = np.nan

        # Sort chronologically by end_date
        if 'end_date' in out.columns:
            out['end_date'] = pd.to_datetime(out['end_date'], errors='coerce')
            out = out.sort_values('end_date').reset_index(drop=True)

        return out

    def get_finnhub_financials_unified(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        wide: bool = True
    ) -> pd.DataFrame:
        """
                Fetch Finnhub quarterly and annual, merge IS/BS/CF for each, standardize meta to Polygon style,
                and return a combined DataFrame sorted by end_date.

                By default returns WIDE format (one row per filing, metrics as columns).
                Set wide=False to return LONG format with columns [...meta..., 'concept', 'value'].

                Columns (wide) include Polygon-style meta:
                    ['ticker','start_date','end_date','filing_date','fiscal_year','fiscal_period','timeframe', ... is_/bs_/cf_ ...]
        """
        frames: List[pd.DataFrame] = []
        for freq in ("quarterly", "annual"):
            raw_payload = self._get_finnhub_raw_response(symbol, freq)
            parsed = self._parse_finnhub_financials(raw_payload, start_date, end_date, pivot=True)
            merged  = self._merge_finnhub_statements(parsed)
            if merged is None or merged.empty:
                continue
            standardized = self._standardize_finnhub_meta(merged, timeframe=freq)
            frames.append(standardized)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True, sort=False)

        # Order columns: meta first
        meta_first = ['ticker','start_date','end_date','filing_date','fiscal_year','fiscal_period','timeframe']
        other = [c for c in out.columns if c not in meta_first]
        out = out[meta_first + other]

        # Sort final chronologically
        if 'end_date' in out.columns:
            out = out.sort_values('end_date').reset_index(drop=True)

        if wide:
            return out

        # Convert to LONG format: keep all non-metric columns as id_vars, melt only metric columns
        metric_prefixes = ("is_", "bs_", "cf_", "ci_")
        value_vars = [c for c in out.columns if any(c.startswith(p) for p in metric_prefixes)]
        id_vars = [c for c in out.columns if c not in value_vars]
        if not value_vars:
            # Nothing to melt; return as-is
            return out
        long_df = out.melt(id_vars=id_vars, value_vars=value_vars, var_name="concept", value_name="value")
        # Drop rows where value is all NaN to reduce size
        long_df = long_df.dropna(subset=["value"], how="all")
        # Keep chronological ordering within id_vars context
        if 'end_date' in long_df.columns:
            long_df = long_df.sort_values('end_date').reset_index(drop=True)
        return long_df
    
    def collect_unrecognized_unified_fields(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Loop through tickers, fetch unified Finnhub statements, capture any columns that contain
        '_UNREC' (matches both _UNRECOGNIZED and _UNRECOGNISED), and persist the long-form results.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        meta_cols = [
            'ticker',
            'timeframe',
            'fiscal_year',
            'fiscal_period',
            'start_date',
            'end_date',
            'filing_date',
        ]

        collected: List[pd.DataFrame] = []
        discovered_fields: set[str] = set()

        for ticker in tickers:
            try:
                fins = self.get_finnhub_financials_unified(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    wide=True,
                )
            except Exception as exc:
                self._log(f"{ticker}: failed to fetch unified fins ({exc})")
                continue

            if fins.empty:
                continue

            unrec_cols = [
                c for c in fins.columns
                if isinstance(c, str) and "_UNREC" in c.upper()
            ]
            if not unrec_cols:
                continue

            discovered_fields.update(unrec_cols)
            keep_cols = [c for c in meta_cols if c in fins.columns] + unrec_cols
            subset = fins[keep_cols].copy()
            melted = subset.melt(
                id_vars=[c for c in meta_cols if c in subset.columns],
                value_vars=unrec_cols,
                var_name="field",
                value_name="value",
            )
            melted = melted.dropna(subset=["value"], how="all")
            if melted.empty:
                continue
            melted["source_ticker"] = ticker
            collected.append(melted)

        if not collected:
            empty_cols = ['ticker', 'timeframe', 'fiscal_year', 'fiscal_period', 'start_date',
                          'end_date', 'filing_date', 'field', 'value', 'source_ticker']
            result = pd.DataFrame(columns=empty_cols)
        else:
            result = pd.concat(collected, ignore_index=True)

        unique_fields = sorted(result['field'].unique())
        self.unrecognised_fields = {field: field for field in unique_fields}
        return unique_fields
    
    # --------------------------------------------------------------------- #
    # Prices: local + Polygon fallback, aligned to filing_date
    # --------------------------------------------------------------------- #

    def _load_local_price_series(self, ticker: str) -> Optional[pd.Series]:
        """Load merged close series from local_price_dir if available."""
        if not self.local_price_dir:
            return None
        pattern = f"{ticker.upper()}_*.parquet"
        files = list(Path(self.local_price_dir).glob(pattern))
        if not files:
            return None
        dfs = []
        for p in files:
            df = pd.read_parquet(p)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.normalize()
            if "close" not in df.columns:
                continue
            dfs.append(df[["close"]])
        if not dfs:
            return None
        merged = pd.concat(dfs).sort_index()
        return merged["close"]

    def _align_prices_to_filing_dates(self, ticker: str, date_col : str, fins: pd.DataFrame) -> pd.Series:
        """
        Align prices to filing_date (or end_date if filing_date is NaT):
        price = last close on or before that date.
        """
        # Try local first
        s = self._load_local_price_series(ticker)
        # If no local, fallback to Polygon
        if s is None:
            from pandas.tseries.offsets import BDay

            fdates = pd.to_datetime(fins[date_col], errors="coerce")
            if fdates.isna().all():
                # Fallback: use end_date
                fdates = pd.to_datetime(fins["end_date"], errors="coerce")
            start = (fdates.min() - BDay(5)).normalize()
            end = (fdates.max() + BDay(5)).normalize()
            prices = self.fetch_daily_prices(ticker, start, end)
            if prices.empty:
                raise ValueError(f"{ticker}: no price data {start}–{end}")
            s = prices.set_index("date")["close"].sort_index()

        # Normalize index to date
        if isinstance(s.index, pd.DatetimeIndex):
            s.index = s.index.normalize()

        fdates = pd.to_datetime(fins[date_col], errors="coerce")
        fdates = fdates.dt.normalize()
        aligned = s.reindex(fdates, method="ffill")
        return aligned

    def fetch_daily_prices(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, adjusted: bool = True
    ) -> pd.DataFrame:
        """Fetch OHLC daily prices from Polygon."""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        data = self._get_json(url, params=params)
        results = data.get("results") or []
        if not results:
            return pd.DataFrame(columns=["date", "close"])
        recs = []
        for r in results:
            ts = pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert("US/Eastern")
            recs.append({"date": ts.normalize(), "close": float(r["c"])})
        df = (
            pd.DataFrame(recs)
            .groupby("date", as_index=False)["close"]
            .last()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return df

    # --------------------------------------------------------------------- #
    # Core TTM flows + ratios
    # --------------------------------------------------------------------- #

    # def _compute_ttm_flows(self, final_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Compute all TTM flows (4-quarter rolling) and balance-sheet derived metrics from the final metric view.
    #     """
    #     if final_df.empty:
    #         return final_df

    #     if "end_date" not in final_df.columns:
    #         raise KeyError("Final metrics dataframe must include 'end_date'")

    #     df = final_df.copy().sort_values("end_date").reset_index(drop=True)

    #     needed_cols: Set[str] = set(self._final_metric_target_columns())
    #     needed_cols.update(
    #         {
    #             "is_basic_earnings_per_share",
    #             "is_diluted_earnings_per_share",
    #             "cf_net_cash_flow_from_operating_activities",
    #             "cf_net_cash_flow_from_investing_activities",
    #             "capital_expenditures",
    #             "bs_total_debt",
    #             "short_term_debt",
    #             "bs_long_term_debt",
    #         }
    #     )

    #     for col in needed_cols:
    #         if col not in df.columns:
    #             df[col] = np.nan
    #         df[col] = self._num(df[col])

    #     df["price"] = self._num(df.get("price"))

    #     df["ebitda_q"] = (
    #         df["is_operating_income_loss"].fillna(0.0)
    #         + df["is_depreciation_and_amortization"].fillna(0.0)
    #     )

    #     def roll4(col: str) -> pd.Series:
    #         return self._num(df[col]).rolling(window=4, min_periods=4).sum()

    #     df["is_revenues_ttm"] = roll4("is_revenues")
    #     df["is_net_income_loss_ttm"] = roll4("is_net_income_loss")
    #     df["ebitda_ttm"] = roll4("ebitda_q")
    #     df["is_operating_income_loss_ttm"] = roll4("is_operating_income_loss")
    #     df["is_depreciation_and_amortization_ttm"] = roll4("is_depreciation_and_amortization")
    #     df["is_basic_earnings_per_share_ttm"] = roll4("is_basic_earnings_per_share")
    #     df["is_diluted_earnings_per_share_ttm"] = roll4("is_diluted_earnings_per_share")
    #     df["is_gross_profit_ttm"] = roll4("is_gross_profit")
    #     df["is_cost_of_revenue_ttm"] = roll4("is_cost_of_revenue")
    #     df["is_selling_general_and_administrative_expenses_ttm"] = roll4(
    #         "is_selling_general_and_administrative_expenses"
    #     )
    #     df["is_research_and_development_ttm"] = roll4("is_research_and_development")
    #     df["is_income_loss_from_equity_method_investments_ttm"] = roll4(
    #         "is_income_loss_from_equity_method_investments"
    #     )
    #     df["is_income_loss_before_equity_method_investments_ttm"] = roll4(
    #         "is_income_loss_before_equity_method_investments"
    #     )
    #     df["is_net_income_loss_available_to_common_stockholders_basic_ttm"] = roll4(
    #         "is_net_income_loss_available_to_common_stockholders_basic"
    #     )
    #     df["cf_net_cash_flow_from_operating_activities_ttm"] = roll4(
    #         "cf_net_cash_flow_from_operating_activities"
    #     )
    #     df["cf_net_cash_flow_from_investing_activities_ttm"] = roll4(
    #         "cf_net_cash_flow_from_investing_activities"
    #     )
    #     capex_series = roll4("capital_expenditures")
    #     df["capital_expenditures_ttm"] = (-capex_series).where(capex_series < 0)
    #     df["capital_expenditures_ttm"] = df["capital_expenditures_ttm"].combine_first(capex_series)
    #     if df["capital_expenditures_ttm"].isna().all():
    #         investing_ttm = df["cf_net_cash_flow_from_investing_activities_ttm"]
    #         df["capital_expenditures_ttm"] = (-investing_ttm).where(investing_ttm < 0)

    #     # Balance sheet and stock metrics
    #     bs_equity_parent = self._num(df["bs_equity_attributable_to_parent"])
    #     bs_equity_total = self._num(df["bs_equity"])
    #     df["bs_equity_effective"] = bs_equity_parent.combine_first(bs_equity_total)

    #     df["cash"] = self._num(df["bs_cash_and_cash_equivalents"])
    #     df["total_debt"] = self._num(df.get("bs_total_debt"))

    #     if df["total_debt"].isna().all():
    #         short_term = self._num(df.get("short_term_debt"))
    #         long_term = self._num(df.get("bs_long_term_debt"))
    #         df["total_debt"] = short_term.fillna(0.0) + long_term.fillna(0.0)

    #     df["net_working_capital"] = df["bs_current_assets"] - df["bs_current_liabilities"]
    #     df["capital_employed"] = df["bs_assets"] - df["bs_current_liabilities"]

    #     df["net_working_capital_avg_4q"] = (
    #         df["net_working_capital"].rolling(window=4, min_periods=2).mean()
    #     )
    #     df["capital_employed_avg_4q"] = (
    #         df["capital_employed"].rolling(window=4, min_periods=2).mean()
    #     )
    #     df["bs_assets_avg_4q"] = df["bs_assets"].rolling(window=4, min_periods=2).mean()
    #     df["bs_inventory_avg_4q"] = df["bs_inventory"].rolling(window=4, min_periods=2).mean()
    #     df["bs_equity_avg_4q"] = df["bs_equity_effective"].rolling(window=4, min_periods=2).mean()

    #     # Aliases retained for backwards compatibility
    #     df["equity"] = df["bs_equity_effective"]
    #     df["assets"] = df["bs_assets"]
    #     df["current_assets"] = df["bs_current_assets"]
    #     df["current_liabilities"] = df["bs_current_liabilities"]
    #     df["inventory"] = df["bs_inventory"]
    #     df["nwc"] = df["net_working_capital"]
    #     df["nwc_avg_4q"] = df["net_working_capital_avg_4q"]
    #     df["assets_avg_4q"] = df["bs_assets_avg_4q"]
    #     df["inventory_avg_4q"] = df["bs_inventory_avg_4q"]
    #     df["shares_diluted"] = df["is_diluted_average_shares"]
    #     df["shares_basic"] = df["is_basic_average_shares"]

    #     # Interest expense tracking
    #     df["interest_expense_q"] = self._num(df["is_interest_expense_operating"])
    #     df["interest_expense_ttm"] = (
    #         df["interest_expense_q"].rolling(window=4, min_periods=4).sum().abs()
    #     )

    #     # Equity affiliates fallback (if needed)
    #     df["equity_affiliates_income_ttm"] = df[
    #         "is_income_loss_from_equity_method_investments_ttm"
    #     ]
    #     missing = df["equity_affiliates_income_ttm"].isna()
    #     df.loc[missing, "equity_affiliates_income_ttm"] = (
    #         df["is_income_loss_before_equity_method_investments_ttm"]
    #         - df["is_net_income_loss_available_to_common_stockholders_basic_ttm"]
    #     )

    #     return df

    # def _compute_ttm_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Compute all ratios from TTM flows and prices."""
    #     def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    #         d = d.replace(0, np.nan)
    #         return n / d

    #     # price, market values
    #     df["price"] = self._num(df["price"])
    #     df["is_diluted_average_shares"] = self._num(df["is_diluted_average_shares"])
    #     df["market_cap"] = df["price"] * df["is_diluted_average_shares"]
        
    #     def _ev_row(row):
    #         data = row.to_dict()
    #         ev, _src = compute_metric("enterprise_value", data, METRIC_RULES)
    #         if ev is None:
    #             # very last fallback: same as your old formula
    #             td = data.get("total_debt")
    #             cash = data.get("cash")
    #             if td is not None and not isnan(td) and cash is not None and not isnan(cash):
    #                 ev = data["market_cap"] + td - cash
    #         return ev

    #     df["enterprise_value"] = df.apply(_ev_row, axis=1)
    #     # Valuation ratios
    #     df["pe_ttm"] = safe_div(df["price"], df["is_basic_earnings_per_share_ttm"])
    #     df["pe_diluted_ttm"] = safe_div(df["price"], df["is_diluted_earnings_per_share_ttm"])
    #     df["ps_ttm"] = safe_div(df["market_cap"], df["is_revenues_ttm"])
    #     df["pb_ttm"] = safe_div(df["market_cap"], df["bs_equity_effective"])
    #     df["ev_ebitda_ttm"] = safe_div(df["enterprise_value"], df["ebitda_ttm"])
    #     df["ev_operating_income_ttm"] = safe_div(
    #         df["enterprise_value"], df["is_operating_income_loss_ttm"]
    #     )

    #     # Margins
    #     df["net_margin_ttm"] = safe_div(df["is_net_income_loss_ttm"], df["is_revenues_ttm"])
    #     df["ebitda_margin_ttm"] = safe_div(df["ebitda_ttm"], df["is_revenues_ttm"])
    #     df["operating_margin_ttm"] = safe_div(
    #         df["is_operating_income_loss_ttm"], df["is_revenues_ttm"]
    #     )

    #     # % of revenue
    #     df["equity_affiliates_performance_pct_rev_ttm"] = safe_div(
    #         df["is_income_loss_from_equity_method_investments_ttm"], df["is_revenues_ttm"]
    #     )
    #     df["sga_pct_rev_ttm"] = safe_div(
    #         df["is_selling_general_and_administrative_expenses_ttm"], df["is_revenues_ttm"]
    #     )
    #     df["gross_margin_additional_ttm"] = safe_div(
    #         df["is_gross_profit_ttm"], df["is_revenues_ttm"]
    #     )
    #     df["rnd_pct_rev_ttm"] = safe_div(
    #         df["is_research_and_development_ttm"], df["is_revenues_ttm"]
    #     )

    #     # Capex / DA, ROCE, ROA, inventory turnover, interest coverage
    #     df["capex_to_da_ttm"] = safe_div(
    #         df["capital_expenditures_ttm"], df["is_depreciation_and_amortization_ttm"]
    #     )
    #     df["roce_ttm"] = safe_div(
    #         df["is_operating_income_loss_ttm"], df["capital_employed_avg_4q"]
    #     )
    #     df["roa_ttm"] = safe_div(df["is_net_income_loss_ttm"], df["bs_assets_avg_4q"])
    #     df["inventory_turnover_ttm"] = safe_div(
    #         df["is_cost_of_revenue_ttm"], df["bs_inventory_avg_4q"]
    #     )
    #     df["interest_coverage_ttm"] = safe_div(
    #         df["is_operating_income_loss_ttm"], df["interest_expense_ttm"]
    #     )

    #     return df

    # # --------------------------------------------------------------------- #
    # # End-to-end for one ticker
    # # --------------------------------------------------------------------- #

    # def build_ttm_ratios_from_financials(
    #     self,
    #     ticker: str,
    #     return_separate: bool = False,
    #     provider: Optional[str] = None,
    #     audit_trail: Optional[Union[str, Iterable[str]]] = None,
    #     return_fin: bool = False,
    # ) -> pd.DataFrame:
    #     """Build TTM ratios for a single ticker and save fundamentals + ratios from Polygon or Finnhub data."""
    #     t = ticker.strip().upper()
    #     provider_key = self._normalize_provider(provider)
    #     if audit_trail:
    #         self._log("audit_trail parameter is deprecated and will be ignored.", level="debug")
    #     self._log(f"Start {t} [{provider_key}]")

    #     fins = self._load_financials_for_provider(t, provider_key)
    #     self._loaded_financials = fins
    #     if fins.empty:
    #         raise ValueError(f"{t}: no quarterly financials returned")

    #     fins = fins.copy()

    #     # Clean / filter
    #     if "end_date" in fins.columns:
    #         fins["end_date"] = pd.to_datetime(fins["end_date"], errors="coerce")
    #     if "filing_date" in fins.columns:
    #         fins["filing_date"] = pd.to_datetime(fins["filing_date"], errors="coerce")

    #     fins = self._annotate_period_types(fins)
    #     fins = self._quarterize_flows(fins)
    #     fins = fins.sort_values("end_date").reset_index(drop=True)
    #     self._quarterized_financials = fins
    #     if self.min_end_date:
    #         fins = fins[fins["end_date"] >= pd.to_datetime(self.min_end_date)]
    #     if self.max_end_date:
    #         fins = fins[fins["end_date"] <= pd.to_datetime(self.max_end_date)]

    #     fins = fins.sort_values("end_date").reset_index(drop=True)

    #     if fins.empty:
    #         raise ValueError(f"{t}: no quarterly financials in requested range")

    #     # Patch Q4 filing_date from annual if NaT
    #     fins = self._fill_q4_filing_from_annual(fins, t, provider=provider_key)
    #     self._filled_with_annuals = fins
    #     # Ensure newly seen fields are assigned to metric buckets
    #     canonical_cols = [
    #         c
    #         for c in fins.columns
    #         if isinstance(c, str) and c.startswith(("is_", "bs_", "cf_", "ci_", "fh_"))
    #     ]
    #     self._map_fields_to_buckets(canonical_cols)

    #     # Align prices to filing_date (fallback: end_date)
    #     prices_ser = self._align_prices_to_filing_dates(t, fins)

    #     # Align the time frames 
    #     avaliable_dates = pd.to_datetime(fins['filing_date'])
    #     min_end_date=self.min_end_date; max_end_date=self.max_end_date
        
    #     if min_end_date is not None:
    #         avaliable_dates = avaliable_dates[avaliable_dates >= pd.to_datetime(min_end_date)]
    #         prices_ser      = prices_ser[prices_ser.index >= pd.to_datetime(min_end_date)]
    #     if max_end_date is not None:
    #         avaliable_dates = avaliable_dates[avaliable_dates <= pd.to_datetime(max_end_date)]
    #         prices_ser      = prices_ser[prices_ser.index <= pd.to_datetime(max_end_date)]

    #     fins = fins[fins['filing_date'].isin(avaliable_dates)]
    #     fins =         pd.merge_asof(
    #                         left   = fins,
    #                         right  = prices_ser,
    #                         left_on='filing_date',
    #                         right_index=True,
    #                         direction='backward',
    #                         tolerance=pd.Timedelta('3D')
    #                     ).rename({'key_0': 'close_price_date', 'close': 'price'}, axis=1)

    #     # Save fundamentals snapshot (raw + price) per provider
    #     fundamentals_dir = self._get_fundamentals_dir_for_provider(provider_key)
    #     if fundamentals_dir is not None:
    #         fpath = Path(fundamentals_dir) / f"{t}_fundamentals.parquet"
    #         fins.to_parquet(fpath, index=False)

    #     if return_fin:
    #         return fins
    #     # Prepare canonical metrics and final view
    #     fins = fins.sort_values("end_date").reset_index(drop=True)
    #     canonical = self._prepare_canonical_financials(fins)
    #     calc_df = self._build_final_metric_view(canonical)
    #     if "ticker" not in calc_df.columns:
    #         calc_df["ticker"] = t
    #     else:
    #         calc_df["ticker"] = calc_df["ticker"].fillna(t)
    #     self.calc_view = calc_df.copy()

    #     # Compute TTM flows
    #     flows = self._compute_ttm_flows(calc_df)
    #     flows["ticker"] = t
    #     self.fundamentals_df = flows
    #     # Compute ratios
    #     metrics = self._compute_ttm_ratios(flows)
    #     identifier_cols = [
    #             "ticker",
    #             "start_date",
    #             "end_date",
    #             "filing_date",
    #             "fiscal_year",
    #             "fiscal_period",
    #             "period_type",
    #             "price"]
    #         # Category 2: Income Statement Items (TTM flows)
    #     income_statement_cols = [
    #             "is_revenues_ttm",
    #             "is_net_income_loss_ttm",
    #             "ebitda_ttm",
    #             "is_operating_income_loss_ttm",
    #             "is_gross_profit_ttm",
    #             "is_cost_of_revenue_ttm",
    #             "is_selling_general_and_administrative_expenses_ttm",
    #             "is_research_and_development_ttm",
    #             "is_income_loss_from_equity_method_investments_ttm",
    #             "is_income_loss_before_equity_method_investments_ttm",
    #             "is_net_income_loss_available_to_common_stockholders_basic_ttm",
    #             "is_depreciation_and_amortization_ttm",
    #             "capital_expenditures_ttm",
    #             "cf_net_cash_flow_from_operating_activities_ttm",
    #             "cf_net_cash_flow_from_investing_activities_ttm",
    #         ]
            
    #         # Category 3: Balance Sheet Items (point-in-time)
    #     balance_sheet_cols = [
    #             "bs_equity",
    #             "bs_equity_attributable_to_parent",
    #             "bs_equity_effective",
    #             "bs_cash_and_cash_equivalents",
    #             "cash",
    #             "total_debt",
    #             "bs_assets",
    #             "bs_current_assets",
    #             "bs_current_liabilities",
    #             "bs_inventory",
    #             "net_working_capital",
    #             "net_working_capital_avg_4q",
    #             "capital_employed",
    #             "capital_employed_avg_4q",
    #             "bs_assets_avg_4q",
    #             "bs_inventory_avg_4q",
    #             "bs_equity_avg_4q",
    #         ]
            
    #         # Category 4: Valuation & Per-Share Metrics
    #     valuation_cols = [
    #             "is_basic_average_shares",
    #             "is_diluted_average_shares",
    #             "market_cap",
    #             "enterprise_value",
    #             "is_basic_earnings_per_share_ttm",
    #             "is_diluted_earnings_per_share_ttm",
    #             "pe_ttm",
    #             "pe_diluted_ttm",
    #             "ps_ttm",
    #             "pb_ttm",
    #             "ev_ebitda_ttm",
    #             "ev_operating_income_ttm",
    #         ]
            
    #         # Category 5: Margins, Returns & Efficiency Ratios
    #     ratio_cols = [
    #             "net_margin_ttm",
    #             "ebitda_margin_ttm",
    #             "operating_margin_ttm",
    #             "gross_margin_additional_ttm",
    #             "equity_affiliates_performance_pct_rev_ttm",
    #             "sga_pct_rev_ttm",
    #             "rnd_pct_rev_ttm",
    #             "capex_to_da_ttm",
    #             "roce_ttm",
    #             "roa_ttm",
    #             "inventory_turnover_ttm",
    #             "interest_coverage_ttm",
    #         ]
    #     self.fins = fins
    #     if return_separate:                    
    #         return (metrics[identifier_cols],
    #                 metrics[income_statement_cols],
    #                 metrics[balance_sheet_cols],
    #                 metrics[valuation_cols], 
    #                 metrics[ratio_cols])
        
    #     # Combine all categories
    #     keep_cols = (
    #         identifier_cols +
    #         income_statement_cols +
    #         balance_sheet_cols +
    #         valuation_cols +
    #         ratio_cols
    #     )

    #     metrics = metrics[keep_cols].sort_values("end_date").reset_index(drop=True)

    #     # Save ratios
    #     rpath = Path(self.ratios_dir) / f"{t}_ratios.parquet"
    #     if rpath.exists() and not self.force_update:
    #         existing = pd.read_parquet(rpath)
    #         combined = pd.concat([existing, metrics], ignore_index=True)
    #         combined["end_date"] = pd.to_datetime(combined["end_date"], errors="coerce")
    #         combined = combined.drop_duplicates(subset=["ticker", "end_date"], keep="last")
    #         combined = combined.sort_values(["ticker", "end_date"]).reset_index(drop=True)
    #         combined.to_parquet(rpath, index=False)
    #     else:
    #         metrics.to_parquet(rpath, index=False)

    #     self._log(f"Finished {t} ({len(metrics)} rows)")

    #     return metrics


    ###### Unified Polygon and Finnhub Data #########
    _FY_FORMS: ClassVar[Set[str]] = {"10-K", "20-F", "40-F"}
    _Q_FORMS: ClassVar[Set[str]] = {"10-Q", "6-K"}
    _PERIOD_MAP: ClassVar[Dict[str, int]] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    _FULL_YEAR_MIN_DAYS: ClassVar[int] = 300  # ~10+ months
    _YTD_MIN_DAYS: ClassVar[int] = 120        # >= ~4 months
    _FY_START_TOLERANCE: ClassVar[int] = 7    # days
    _QUARTER_SPACING_TOLERANCE: ClassVar[int] = 10  # day tolerance around 3-month spacing
    _FISCAL_YEAR_SPAN_TOLERANCE: ClassVar[int] = 5  # Q1 start to Q4 end span tolerance
    _CROSS_YEAR_ALIGNMENT_TOLERANCE: ClassVar[int] = 10  # day tolerance YoY same-quarter alignment
    _EARNINGS_RELEASE_MAX_LAG: ClassVar[int] = 80  # end_date must be within 50d before release
    _EARNINGS_RELEASE_SPACING_BOUNDS: ClassVar[Tuple[int, int]] = (60, 130)  # 3-4 months
    _EARNINGS_FILING_TOLERANCE: ClassVar[int] = 80  # filing date should trail release by <=15d

    _RAW_META_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "period_type",
        "effective_fiscal_year",
        "ticker",
        "start_date",
        "end_date",
        "filing_date",
        "form",
        "fiscal_period",
        "timeframe",
        "accepted_date",
        "period_key",
    )
    _MERGED_META_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "ticker",
        "provider",
        "effective_fiscal_year",
        "period_type",
        "fiscal_period",
        "quarter",
        "start_date",
        "end_date",
        "filing_date",
        "period_key",
    )
    _META_DROP_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "ticker",
        "provider",
        "effective_fiscal_year",
        "period_type",
        "fiscal_period",
        "fiscal_year",
        "quarter",
        "start_date",
        "end_date",
        "filing_date",
        "form",
        "timeframe",
        "accepted_date",
        "access_number",
        "period_key",
    )
    _FLOW_PREFIXES: ClassVar[Tuple[str, ...]] = ("is_", "cf_", "ci_")
    _STOCK_PREFIXES: ClassVar[Tuple[str, ...]] = ("bs_",)
    _NON_ADDITIVE_FLOW_COLS: ClassVar[Set[str]] = {
        "is_basic_earnings_per_share",
        "is_diluted_earnings_per_share",
        "basic_average_shares",
        "diluted_average_shares",
    }

    @staticmethod
    def _attach_effective_fiscal_year(
        fins: pd.DataFrame,
        ticker_col: str = "ticker",
        fiscal_year_col: str = "fiscal_year",
        start_col: str = "start_date",
        end_col: str = "end_date",
        form_col: str = "form",
        timeframe_col: str = "timeframe",
        fiscal_period_col: str = "fiscal_period",
    ) -> pd.DataFrame:
        """Derive an effective fiscal year for each row based on annual filings (robust to 52/53-week)."""
        if fins.empty:
            fins["effective_fiscal_year"] = pd.NA
            return fins

        df = fins.copy()
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

        tf = df.get(timeframe_col)
        fp = df.get(fiscal_period_col)
        fm = df.get(form_col)
        df["__timeframe_norm"] = tf.astype(str).str.lower() if tf is not None else ""
        df["__fiscal_period_norm"] = fp.astype(str).str.upper() if fp is not None else ""
        df["__form_norm"] = fm.astype(str).str.upper() if fm is not None else ""

        annual_like = (
            (df["__timeframe_norm"] == "annual")
            | (df["__fiscal_period_norm"].isin({"FY", "Q0", "Y"}))
            | (df["__form_norm"].isin(TTMRatiosAnalyzer._FY_FORMS))
        )

        df["effective_fiscal_year"] = pd.NA
        if ticker_col not in df.columns:
            raise KeyError(f"Expected '{ticker_col}' column in dataframe")

        def _infer_fy_pattern(annual_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
            """Return (end_month, end_day_or_None) using annual rows."""
            if annual_df.empty or annual_df[end_col].isna().all():
                return None, None
            ends = pd.to_datetime(annual_df[end_col], errors="coerce").dropna()
            if ends.empty:
                return None, None
            end_md = ends.apply(lambda d: (d.month, d.day))
            end_month_mode = end_md.map(lambda x: x[0]).mode()
            end_day_mode = end_md.mode()
            if len(end_month_mode) == 1:
                m = int(end_month_mode.iloc[0])
            else:
                m = None
            if len(end_day_mode) == 1:
                md = end_day_mode.iloc[0]
                if m is not None and md[0] == m:
                    d = int(md[1])
                else:
                    d = None
            else:
                d = None
            return m, d

        def _label_fy(end_dt: pd.Timestamp, fp_str: str, fy_end_month: Optional[int], fy_end_day: Optional[int]) -> int:
            """Consistent FY label based on end date and inferred FY end pattern."""
            if pd.isna(end_dt):
                return pd.NA
            end_year = end_dt.year
            if fy_end_month is None:
                return end_year
            if fy_end_day is not None:
                before_or_on = (end_dt.month, end_dt.day) <= (fy_end_month, fy_end_day)
            else:
                before_or_on = end_dt.month <= fy_end_month
            if fy_end_month <= 2 and before_or_on:
                return end_year - 1
            return end_year

        for ticker, tgrp in df.groupby(ticker_col, dropna=False):
            tgrp = tgrp.copy()
            idx = tgrp.index
            annual = tgrp[annual_like.loc[idx] & tgrp[end_col].notna()].copy()
            fy_end_month, fy_end_day = _infer_fy_pattern(annual)

            for ridx, rrow in tgrp.iterrows():
                end_dt = rrow[end_col]
                fp_val = str(rrow.get(fiscal_period_col, "")).upper()
                fy_tag = rrow.get(fiscal_year_col)

                fy_label = _label_fy(end_dt, fp_val, fy_end_month, fy_end_day)
                if pd.notna(fy_tag):
                    try:
                        fy_tag_int = int(fy_tag)
                        if fy_label is pd.NA or abs(fy_tag_int - fy_label) <= 1:
                            fy_label = fy_tag_int
                    except Exception:
                        pass

                df.at[ridx, "effective_fiscal_year"] = fy_label

        return df.drop(columns=["__timeframe_norm", "__fiscal_period_norm", "__form_norm"], errors="ignore")


    @staticmethod
    def _infer_fiscal_year_start_cache(
        fins: pd.DataFrame,
        ticker_col: str = "ticker",
        fiscal_year_col: str = "effective_fiscal_year",
        start_col: str = "start_date",
        end_col: str = "end_date",
        timeframe_col: str = "timeframe",
        fiscal_period_col: str = "fiscal_period",
        form_col: str = "form",
        max_span_days: int = 400,
    ) -> Dict[tuple[str, int], pd.Timestamp]:
        """Infer fiscal-year start dates for each (ticker, fiscal year) using annual rows first, then spans."""
        if fins.empty:
            return {}

        df = fins.copy()
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
        df["__period_days"] = (df[end_col] - df[start_col]).dt.days

        tf = df.get(timeframe_col)
        fp = df.get(fiscal_period_col)
        fm = df.get(form_col)
        df["__timeframe_norm"] = tf.astype(str).str.lower() if tf is not None else ""
        df["__fiscal_period_norm"] = fp.astype(str).str.upper() if fp is not None else ""
        df["__form_norm"] = fm.astype(str).str.upper() if fm is not None else ""

        annual_like = (
            (df["__timeframe_norm"] == "annual")
            | (df["__fiscal_period_norm"].isin({"FY", "Q0", "Y"}))
            | (df["__form_norm"].isin(TTMRatiosAnalyzer._FY_FORMS))
            | (df["__period_days"] >= TTMRatiosAnalyzer._FULL_YEAR_MIN_DAYS)
        )
        df["__is_annual_like"] = annual_like

        group_cols = [ticker_col, fiscal_year_col]
        if any(col not in df.columns for col in group_cols):
            raise KeyError(f"Expected columns {group_cols} in dataframe")

        cache: Dict[tuple[str, int], pd.Timestamp] = {}
        for (tick, fy), grp in df.groupby(group_cols, dropna=False):
            if tick is None or pd.isna(fy):
                continue
            try:
                fy_int = int(fy)
            except (TypeError, ValueError):
                continue

            ticker_key = str(tick).upper()
            grp = grp.copy()
            annual_grp = grp[grp["__is_annual_like"] & grp[start_col].notna() & grp[end_col].notna()]
            if not annual_grp.empty:
                fy_start = annual_grp[start_col].min().normalize()
                cache[(ticker_key, fy_int)] = fy_start
                continue

            q_grp = grp[grp[start_col].notna()]
            if q_grp.empty:
                continue

            latest_end = q_grp[end_col].max()
            trimmed = q_grp[
                (q_grp[end_col].notna())
                & (q_grp[start_col] >= latest_end - pd.Timedelta(days=max_span_days))
            ]
            if trimmed.empty:
                fy_start = q_grp[start_col].min().normalize()
            else:
                fy_start = trimmed[start_col].min().normalize()
            cache[(ticker_key, fy_int)] = fy_start

        return cache

    @staticmethod
    def _get_fiscal_year_start_for_row(
        row: pd.Series,
        fiscal_start_cache: Dict[tuple[str, int], pd.Timestamp],
        ticker_col: str = "ticker",
        fiscal_year_col: str = "effective_fiscal_year",
    ) -> Optional[pd.Timestamp]:
        ticker = row.get(ticker_col) or row.get("symbol")
        fy = row.get(fiscal_year_col) or row.get("year")
        if ticker is None or pd.isna(fy):
            return None
        try:
            fy_int = int(fy)
        except (TypeError, ValueError):
            return None

        key = (str(ticker).upper(), fy_int)
        ts = fiscal_start_cache.get(key)
        if ts is None or pd.isna(ts):
            return None
        return pd.to_datetime(ts).normalize().date()

    @staticmethod
    def _classify_period_type(
        row: pd.Series,
        fiscal_start_cache: Dict[tuple[str, int], pd.Timestamp],
    ) -> str:
        form = str(row.get("form") or "").strip().upper()

        def _parse_date(val: Any) -> Optional[date]:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            try:
                return pd.to_datetime(val).date()
            except Exception:
                return None

        start = _parse_date(row.get("start_date") or row.get("startDate"))
        end = _parse_date(row.get("end_date") or row.get("endDate"))

        if start is None or end is None:
            if form in TTMRatiosAnalyzer._FY_FORMS:
                return "FY"
            if form in TTMRatiosAnalyzer._Q_FORMS:
                return "Q"
            return "Q"

        period_days = (end - start).days
        if form in TTMRatiosAnalyzer._FY_FORMS or period_days >= TTMRatiosAnalyzer._FULL_YEAR_MIN_DAYS:
            return "FY"

        fiscal_year_start = TTMRatiosAnalyzer._get_fiscal_year_start_for_row(row, fiscal_start_cache)
        if (
            fiscal_year_start is not None
            and abs((start - fiscal_year_start).days) <= TTMRatiosAnalyzer._FY_START_TOLERANCE
            and period_days >= TTMRatiosAnalyzer._YTD_MIN_DAYS
        ):
            return "YTD_Q"

        return "Q"

    @staticmethod
    def _classify_period_type_vectorized(
        df: pd.DataFrame,
        fiscal_start_cache: Dict[tuple[str, int], pd.Timestamp],
    ) -> pd.Series:
        """
        Vectorized period-type classifier using FY start cache.
        """
        if df.empty:
            return pd.Series(dtype=object, index=df.index)

        form_series = df["form"] if "form" in df.columns else pd.Series(index=df.index, dtype=object)
        form = form_series.astype(str).str.upper()
        start_col = df["start_date"] if "start_date" in df.columns else df.get("startDate")
        end_col = df["end_date"] if "end_date" in df.columns else df.get("endDate")
        fy_col = df["effective_fiscal_year"] if "effective_fiscal_year" in df.columns else df.get("year")
        ticker_col = df["ticker"] if "ticker" in df.columns else df.get("symbol")
        fp_col = df["fiscal_period"] if "fiscal_period" in df.columns else df.get("period")

        if start_col is None:
            start_col = pd.Series(pd.NaT, index=df.index)
        if end_col is None:
            end_col = pd.Series(pd.NaT, index=df.index)
        if fy_col is None:
            fy_col = pd.Series(pd.NA, index=df.index, dtype="Int64")
        if ticker_col is None:
            ticker_col = pd.Series(index=df.index, dtype=object)
        if fp_col is None:
            fp_col = pd.Series("", index=df.index)

        start = pd.to_datetime(start_col, errors="coerce")
        end = pd.to_datetime(end_col, errors="coerce")
        fy = pd.to_numeric(fy_col, errors="coerce").astype("Int64")
        ticker = ticker_col.astype(str).str.upper()
        fp = fp_col.astype(str).str.upper()

        result = pd.Series("Q", index=df.index)

        mask_missing_dates = start.isna() | end.isna()
        result.loc[mask_missing_dates & form.isin(TTMRatiosAnalyzer._FY_FORMS)] = "FY"
        result.loc[mask_missing_dates & form.isin(TTMRatiosAnalyzer._Q_FORMS)] = "Q"

        period_days = (end - start).dt.days
        is_fy = (
            form.isin(TTMRatiosAnalyzer._FY_FORMS)
            | fp.isin({"FY", "Q0", "Y"})
            | (period_days >= TTMRatiosAnalyzer._FULL_YEAR_MIN_DAYS)
        )
        result.loc[~mask_missing_dates & is_fy] = "FY"

        key_series = pd.Series(list(zip(ticker, fy)), index=df.index)
        fy_start_series = key_series.map(fiscal_start_cache)
        fy_delta = (start.dt.normalize() - pd.to_datetime(fy_start_series, errors="coerce")).dt.days

        is_ytd = (
            ~mask_missing_dates
            & result.eq("Q")
            & fy_start_series.notna()
            & fy_delta.abs().le(TTMRatiosAnalyzer._FY_START_TOLERANCE)
            & (period_days >= TTMRatiosAnalyzer._YTD_MIN_DAYS)
        )
        result.loc[is_ytd] = "YTD_Q"
        return result

    @staticmethod
    def _choose_canonical_raw_row(
        df: pd.DataFrame,
        provider_preferences: Sequence[str] = ("finn", "poly"),
    ) -> pd.DataFrame:
        required = ["ticker", "effective_fiscal_year", "period_type", "quarter", "provider"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        out = df.copy()
        out["quarter"] = out["quarter"].astype("Int64")
        provider_order = [p.lower() for p in provider_preferences]
        key_cols = ["ticker", "effective_fiscal_year", "period_type", "quarter"]

        def _sort_by_date(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            if "filing_date" in g.columns and g["filing_date"].notna().all():
                return g.sort_values("filing_date")
            if "end_date" in g.columns and g["end_date"].notna().all():
                return g.sort_values("end_date")
            return g

        def _choose_one(group: pd.DataFrame) -> pd.Series:
            g = group.copy()
            providers_here = list(g["provider"].dropna().unique())
            for prov in provider_order:
                if prov not in providers_here:
                    continue
                gprov = g[g["provider"] == prov]
                if gprov.empty:
                    continue
                chosen = _sort_by_date(gprov).iloc[-1].copy()
                chosen["providers_available"] = providers_here
                return chosen

            chosen = _sort_by_date(g).iloc[-1].copy()
            chosen["providers_available"] = providers_here
            return chosen

        canonical = (
            out.groupby(key_cols, dropna=False, group_keys=False)
            .apply(_choose_one)
            .reset_index(drop=True)
        )
        return canonical

    @staticmethod
    def _ensure_quarterly_coverage(
        df: pd.DataFrame,
        preferred_providers: Sequence[str] = ("finn", "poly"),
        year_col: str = "effective_fiscal_year",
        period_col: str = "fiscal_period",
        provider_col: str = "provider",
        period_type_col: str = "period_type",
        end_date_col: str = "end_date",
        filing_date_col: str = "filing_date",
    ) -> Tuple[pd.DataFrame, Dict[tuple[Any, Any], List[str]]]:
        """Ensure each fiscal year has one row per quarter, preferring providers with flow data."""
        if df.empty:
            return df.copy(), {}

        data = df.copy()
        quarter_alias = {"Q0": "Q4", "FY": "Q4", "Y": "Q4", "4Q": "Q4", "3Q": "Q3", "2Q": "Q2", "1Q": "Q1"}

        def normalize_quarter(value: Any) -> Optional[str]:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            s = str(value).strip().upper()
            s = quarter_alias.get(s, s)
            if s.startswith("Q") and len(s) == 2 and s[1] in "1234":
                return s
            return None

        data["_quarter_key"] = data[period_col].apply(normalize_quarter)
        data = data[data["_quarter_key"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
        if data.empty:
            return data.drop(columns=["_quarter_key"], errors="ignore"), {}

        if "quarter" not in data.columns:
            data["quarter"] = data["_quarter_key"].map(TTMRatiosAnalyzer._PERIOD_MAP).astype("Int64")

        providers = [p.lower() for p in preferred_providers]
        for provider in data[provider_col].dropna().str.lower().unique():
            if provider not in providers:
                providers.append(provider)
        provider_rank = {provider: idx for idx, provider in enumerate(providers)}

        period_rank = {"Q": 0, "YTD_Q": 1, "FY": 2}
        data["_provider_rank"] = data[provider_col].str.lower().map(provider_rank).fillna(len(providers))
        data["_type_rank"] = data[period_type_col].map(period_rank).fillna(len(period_rank))
        data["_end_sort"] = pd.to_datetime(data.get(end_date_col), errors="coerce").fillna(pd.Timestamp.min)
        data["_filing_sort"] = (
            pd.to_datetime(data.get(filing_date_col), errors="coerce").fillna(pd.Timestamp.min)
        )

        flow_cols = [c for c in data.columns if c in getattr(TTMRatiosAnalyzer, "_FLOW_COLS", [])]
        if not flow_cols:
            flow_cols = [c for c in data.columns if c.startswith(TTMRatiosAnalyzer._FLOW_PREFIXES)]

        def _row_has_flow(row: pd.Series) -> bool:
            if not flow_cols:
                return True
            return row[flow_cols].notna().any()

        selected_idx: List[int] = []
        missing: Dict[tuple[Any, Any], List[str]] = {}

        for (ticker, year), group in data.groupby(["ticker", year_col], dropna=False):
            for q_label in ("Q1", "Q2", "Q3", "Q4"):
                subset = group[group["_quarter_key"] == q_label]
                if subset.empty:
                    missing.setdefault((ticker, year), []).append(q_label)
                    continue
                finn_mask = subset[provider_col].str.lower() == "finn"
                finn_candidates = subset[finn_mask]
                if not finn_candidates.empty and flow_cols:
                    finn_candidates = finn_candidates.loc[
                        [idx for idx, row in finn_candidates.iterrows() if _row_has_flow(row)]
                    ]

                if not finn_candidates.empty:
                    subset_to_sort = finn_candidates
                else:
                    subset_to_sort = subset[~finn_mask] if not subset[~finn_mask].empty else subset

                subset_sorted = subset_to_sort.sort_values(
                    by=["_provider_rank", "_type_rank", "_end_sort", "_filing_sort"],
                    ascending=[True, True, False, False],
                )
                selected_idx.append(subset_sorted.index[0])

        selected = data.loc[selected_idx].copy()
        selected = selected.sort_values(["ticker", year_col, "quarter"])
        selected = selected.drop(
            columns=["_provider_rank", "_type_rank", "_end_sort", "_filing_sort"], errors="ignore"
        )
        return selected.reset_index(drop=True), {k: v for k, v in missing.items() if v}


    @staticmethod
    def _quarterize_flows(
        fins: pd.DataFrame,
        ticker_col: str = "ticker",
        fiscal_year_col: str = "effective_fiscal_year",
        period_type_col: str = "period_type",
        fiscal_period_col: str = "fiscal_period",
    ) -> pd.DataFrame:
        if fins.empty:
            return fins

        df = fins.copy()
        if fiscal_period_col in df.columns:
            df[fiscal_period_col] = df[fiscal_period_col].str.replace("Q0", "Q4")

        if ticker_col not in df.columns:
            if "symbol" in df.columns:
                ticker_col = "symbol"
            else:
                raise KeyError("Expected a ticker or symbol column for quarterisation")
        if fiscal_year_col not in df.columns:
            raise KeyError(f"Missing fiscal year column '{fiscal_year_col}'")
        if period_type_col not in df.columns:
            raise KeyError(f"Missing period type column '{period_type_col}'")

        if "quarter" not in df.columns:
            df["quarter"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        else:
            df["quarter"] = df["quarter"].astype("Int64")

        missing_quarter = df["quarter"].isna()
        if missing_quarter.any() and fiscal_period_col in df.columns:
            extracted = (
                df.loc[missing_quarter, fiscal_period_col]
                .astype(str)
                .str.extract(r"Q(\d)")
            )
            df.loc[missing_quarter, "quarter"] = extracted[0].astype("Int64")
            missing_quarter = df["quarter"].isna()

        if missing_quarter.any() and "end_date" in df.columns:
            df.loc[missing_quarter, "quarter"] = (
                pd.to_datetime(df.loc[missing_quarter, "end_date"], errors="coerce")
                .dt.quarter.astype("Int64")
            )
            missing_quarter = df["quarter"].isna()

        if period_type_col in df.columns:
            fy_mask = df["quarter"].isna() & (
                df[period_type_col].astype(str).str.upper() == "FY"
            )
            if fy_mask.any():
                df.loc[fy_mask, "quarter"] = pd.Series(
                    4, index=df.loc[fy_mask].index, dtype="Int64"
                )
                if fiscal_period_col in df.columns:
                    df.loc[fy_mask, fiscal_period_col] = df.loc[
                        fy_mask, fiscal_period_col
                    ].fillna("Q4")
                TTMRatiosAnalyzer._validate_fy_quarter_sequence(
                    df,
                    fy_mask,
                    ticker_col=ticker_col,
                    fiscal_year_col=fiscal_year_col,
                )

        def _normalized(col: Any) -> str:
            if not isinstance(col, str):
                return ""
            return TTMRatiosAnalyzer._strip_statement_prefix(col)

        flow_cols = [c for c in df.columns if _normalized(c) in _FLOW_COLS]
        stock_cols = [c for c in df.columns if _normalized(c) in _STOCK_COLS]

        if not flow_cols:
            flow_cols = [
                c for c in df.columns if isinstance(c, str) and c.startswith(TTMRatiosAnalyzer._FLOW_PREFIXES)
            ]
        if not stock_cols:
            stock_cols = [
                c for c in df.columns if isinstance(c, str) and c.startswith(TTMRatiosAnalyzer._STOCK_PREFIXES)
            ]

        for col in ("is_basic_average_shares", "is_diluted_average_shares"):
            if col in flow_cols:
                flow_cols.remove(col)
                stock_cols.append(col)

        meta_cols = [c for c in df.columns if c not in (*flow_cols, *stock_cols)]

        group_cols = [ticker_col, fiscal_year_col]
        records: List[Dict[str, Any]] = []
        for (ticker, fy), grp in df.groupby(group_cols, dropna=False):
            grp = grp.copy()
            sort_cols = ["quarter"]
            if "end_date" in grp.columns:
                sort_cols.append("end_date")
            if "filing_date" in grp.columns:
                sort_cols.append("filing_date")
            grp = grp.sort_values(sort_cols)

            quarter_rows: Dict[int, pd.Series] = {}
            fy_row: Optional[pd.Series] = None

            for _, row in grp.iterrows():
                period_type = str(row.get(period_type_col) or "Q")
                q_val = row.get("quarter")
                try:
                    q_int = int(q_val)
                except Exception:
                    q_int = None
                if period_type == "FY":
                    fy_row = row
                    continue
                if q_int is None or not (1 <= q_int <= 4):
                    continue
                quarter_rows[q_int] = row

            flow_arrays = {col: [np.nan, np.nan, np.nan, np.nan] for col in flow_cols}
            cumulative_flow = {col: np.nan for col in flow_cols}
            ytd_tracker = {col: np.nan for col in flow_cols}
            fy_values = {col: np.nan for col in flow_cols}

            if fy_row is not None:
                for col in flow_cols:
                    val = fy_row.get(col)
                    if not pd.isna(val):
                        fy_values[col] = val

            for q in range(1, 5):
                row = quarter_rows.get(q)
                if row is None:
                    continue
                period_type = str(row.get(period_type_col) or "Q")
                for col in flow_cols:
                    val = row.get(col)
                    if pd.isna(val):
                        continue
                    if period_type == "Q":
                        flow_arrays[col][q - 1] = val
                        prev_cum = cumulative_flow[col]
                        cumulative_flow[col] = val if pd.isna(prev_cum) else prev_cum + val
                    elif period_type == "YTD_Q":
                        prev_ytd = ytd_tracker[col]
                        prev_cum = cumulative_flow[col]
                        if not pd.isna(prev_ytd):
                            incremental = val - prev_ytd
                        elif not pd.isna(prev_cum):
                            incremental = val - prev_cum
                        else:
                            incremental = val
                        flow_arrays[col][q - 1] = incremental
                        if not pd.isna(incremental):
                            cumulative_flow[col] = (
                                incremental if pd.isna(prev_cum) else prev_cum + incremental
                            )
                        ytd_tracker[col] = val

            for col in flow_cols:
                if pd.isna(flow_arrays[col][3]) and not pd.isna(fy_values[col]):
                    first_three = [
                        flow_arrays[col][i] for i in range(3) if not pd.isna(flow_arrays[col][i])
                    ]
                    if first_three:
                        known_sum = sum(first_three)
                        fy_val = fy_values[col]    
                        flow_arrays[col][3] = fy_val - known_sum

            for q in range(1, 5):
                row = quarter_rows.get(q)
                record: Dict[str, Any] = {}
                if row is not None:
                    record.update({col: row.get(col) for col in meta_cols})
                else:
                    for col in meta_cols:
                        record[col] = pd.NA
                    record[ticker_col] = ticker
                    record[fiscal_year_col] = fy
                    record["quarter"] = q
                    record[fiscal_period_col] = f"Q{q}"
                    record["timeframe"] = "quarterly"

                record[ticker_col] = ticker
                record[fiscal_year_col] = fy
                record["quarter"] = q
                record["timeframe"] = "quarterly"
                record[period_type_col] = "Q"

                for col in flow_cols:
                    record[col] = flow_arrays[col][q - 1]

                for col in stock_cols:
                    value = None
                    if row is not None:
                        value = row.get(col)
                    if (value is None or pd.isna(value)) and fy_row is not None and q == 4:
                        value = fy_row.get(col)
                    record[col] = value

                if "end_date" in df.columns and pd.isna(record.get("end_date")):
                    try:
                        period = pd.Period(year=int(fy), quarter=q)
                        record["end_date"] = period.end_time
                    except Exception:
                        record["end_date"] = pd.NaT

                if "start_date" in df.columns and pd.isna(record.get("start_date")):
                    try:
                        period = pd.Period(year=int(fy), quarter=q)
                        record["start_date"] = period.start_time
                    except Exception:
                        record["start_date"] = pd.NaT

                if "filing_date" in df.columns and pd.isna(record.get("filing_date")):
                    record["filing_date"] = record.get("end_date")

                records.append(record)

        fins_q = pd.DataFrame(records)
        order_cols = list(df.columns)
        for col in fins_q.columns:
            if col not in order_cols:
                order_cols.append(col)
        fins_q = fins_q[order_cols]
        return fins_q

    @staticmethod
    def _fix_share_count_outliers(
        df: pd.DataFrame,
        share_cols: Optional[List[str]] = None,
        threshold_ratio: float = 0.1,
        nan_neighbor_min_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """
        Identify and fix outliers in share count columns using linear interpolation.

        Two types of fixes:
          1) Outlier spikes/drops: value is much smaller than both neighbors.
          2) NaN gaps: a run of NaNs between two valid points whose endpoints
             are not too different (future share count within a factor of
             1 / nan_neighbor_min_ratio to nan_neighbor_min_ratio of the past).

        Parameters
        ----------
        df : pd.DataFrame
            Quarterised dataframe with period index or with columns like
            'effective_fiscal_year' and 'quarter'.

        share_cols : Optional[List[str]]
            List of column names to check for outliers. Defaults to basic and diluted shares.

        threshold_ratio : float
            Minimum ratio of current value to both neighbors to avoid being flagged as outlier.
            Default 0.1 means value must be at least 10% of both neighbors.

        nan_neighbor_min_ratio : float
            Minimum ratio between the two non-NaN endpoints around a NaN gap in
            order to interpolate between them. For example, 0.5 means the later
            share count must be >= 50% and <= 200% of the earlier one.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers and suitable NaNs replaced by linear interpolation,
            preserving original index labels.

        """
        if df.empty:
            return df

        if share_cols is None:
            share_cols = ["is_basic_average_shares", "is_diluted_average_shares"]

        df = df.copy()
        original_index = df.index.copy()

        # Determine sorting method based on dataframe structure
        if df.index.name == "period" or (
            len(df) > 0 and isinstance(df.index[0], str) and "Q" in str(df.index[0])
        ):
            # Period index - sort by period
            df_sorted = df.sort_index()
            use_index = True
            index_mapping = {i: idx for i, idx in enumerate(df_sorted.index)}
        elif "effective_fiscal_year" in df.columns and "quarter" in df.columns:
            # Has fiscal year and quarter columns - sort by these
            df_sorted = df.sort_values(["effective_fiscal_year", "quarter"]).reset_index(
                drop=True
            )
            use_index = False
            sorted_df = df.sort_values(["effective_fiscal_year", "quarter"])
            index_mapping = {i: sorted_df.index[i] for i in range(len(sorted_df))}
        else:
            # Fallback: sort by index
            df_sorted = df.sort_index()
            use_index = True
            index_mapping = {i: idx for i, idx in enumerate(df_sorted.index)}

        for col in share_cols:
            if col not in df_sorted.columns:
                continue

            # Work on a numeric copy
            series = pd.to_numeric(df_sorted[col].copy(), errors="coerce")

            # -------------------------------------------------
            # 1) Detect and fix "downward spike" outliers
            # -------------------------------------------------
            outliers: List[int] = []

            for i in range(1, len(series) - 1):
                current = series.iloc[i]
                prev = series.iloc[i - 1]
                next_val = series.iloc[i + 1]

                # Skip if any value is NaN
                if pd.isna(current) or pd.isna(prev) or pd.isna(next_val):
                    continue

                # Check if current is significantly smaller than both neighbors
                ratio_to_prev = current / prev if prev != 0 else np.inf
                ratio_to_next = current / next_val if next_val != 0 else np.inf

                # If current is less than threshold_ratio of both neighbors, it's an outlier
                if ratio_to_prev < threshold_ratio and ratio_to_next < threshold_ratio:
                    outliers.append(i)

            # Fix outliers using linear interpolation
            if outliers:
                for idx in outliers:
                    prev_val = series.iloc[idx - 1]
                    next_val = series.iloc[idx + 1]

                    interpolated = (prev_val + next_val) / 2.0

                    df_sorted.iloc[idx, df_sorted.columns.get_loc(col)] = interpolated

            # Refresh series after outlier fixes
            series = pd.to_numeric(df_sorted[col].copy(), errors="coerce").values

            n = len(series)

            # -------------------------------------------------
            # 2) Fill NaN gaps by linear interpolation
            #    when endpoints are not too far apart
            # -------------------------------------------------
            i = 0

            while i < n:
                if not np.isnan(series[i]):
                    i += 1
                    continue

                # Start of a NaN run
                start = i
                while i < n and np.isnan(series[i]):
                    i += 1
                end = i - 1  # inclusive index of last NaN

                left_idx = start - 1
                right_idx = end + 1

                # Only interpolate if we have valid neighbors on both sides
                if left_idx >= 0 and right_idx < n:
                    left_val = series[left_idx]
                    right_val = series[right_idx]

                    if (not np.isnan(left_val)) and (not np.isnan(right_val)):
                        # Shares should be positive; still guard against zero
                        if left_val > 0 and right_val > 0:
                            ratio = right_val / left_val

                            # Require future within [nan_neighbor_min_ratio, 1 / nan_neighbor_min_ratio] of past
                            if (
                                ratio >= nan_neighbor_min_ratio
                                and ratio <= 1.0 / nan_neighbor_min_ratio
                            ):
                                # Perform linear interpolation across [left_idx, right_idx]
                                num_points = right_idx - left_idx + 1
                                interp_vals = np.linspace(left_val, right_val, num_points)

                                # Fill only the NaN segment (start..end inclusive)
                                # interp_vals[0] corresponds to left_idx, interp_vals[-1] to right_idx
                                # We want positions start to end (inclusive), which is interp_vals[1:-1]
                                col_idx = df_sorted.columns.get_loc(col)
                                interp_to_assign = interp_vals[1:-1]
                                
                                # Ensure we have the right number of values
                                num_nans = end - start + 1
                                if len(interp_to_assign) == num_nans:
                                    df_sorted.iloc[start:end + 1, col_idx] = interp_to_assign
                                else:
                                    # Fallback: assign one by one if lengths don't match
                                    for i, pos in enumerate(range(start, end + 1)):
                                        if i < len(interp_to_assign):
                                            df_sorted.iloc[pos, col_idx] = interp_to_assign[i]

            # end loop over NaN segments for this column

        # Restore original index structure
        if use_index:
            # Already has correct index from sort_index()
            return df_sorted
        else:
            df_sorted.index = [index_mapping[i] for i in range(len(df_sorted))]
            return df_sorted

    @staticmethod
    def _validate_fy_quarter_sequence(
        df: pd.DataFrame,
        fy_mask: pd.Series,
        ticker_col: str,
        fiscal_year_col: str,
    ) -> None:
        """
        When an FY row is coerced into a Q4 snapshot, make sure the surrounding
        fiscal year still has lower-quarter context (Q1/Q2) so consumers can
        interpret the period ordering safely.
        """
        if not fy_mask.any():
            return

        fy_indices = set(df.loc[fy_mask].index)
        if not fy_indices:
            return

        numeric_quarters = pd.to_numeric(df.get("quarter"), errors="coerce")
        df = df.copy()
        df["__quarter_numeric"] = numeric_quarters

        for (ticker, fy), grp in df.groupby([ticker_col, fiscal_year_col], dropna=False):
            grp_indices = set(grp.index)
            if not (grp_indices & fy_indices):
                continue

            quarters = {
                q for q in grp["__quarter_numeric"].tolist() if pd.notna(q)
            }
            if 4 not in quarters:
                continue
            if len(grp) <= 1:
                continue

            has_q1 = any(abs(q - 1) < 1e-9 for q in quarters)
            has_q2 = any(abs(q - 2) < 1e-9 for q in quarters)
            missing = []
            if not has_q1:
                missing.append("Q1")
            if not has_q2:
                missing.append("Q2")
            if missing:
                message = (
                    f"[quarterize] FY-derived Q4 for {ticker} FY{fy} is missing "
                    f"{' and '.join(missing)} context."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _back_out_periods(
        df: pd.DataFrame,
        drop_filing_date: bool = True,
        keep_dates: bool = True,
        fy_cache: Optional[Dict[tuple[str, int], pd.Timestamp]] = None,
        logger: Optional[Callable[[str, str], None]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        working = df.copy()
        fiscal_years = pd.to_numeric(working.get("effective_fiscal_year"), errors="coerce")
        quarters = pd.to_numeric(working.get("quarter"), errors="coerce")

        fy_int = fiscal_years.astype("Int64")
        q_int = quarters.astype("Int64")
        working["effective_fiscal_year"] = fy_int
        working["quarter"] = q_int
        working["period"] = fy_int.astype("string").str.cat("Q" + q_int.astype("string"))

        if keep_dates:
            fiscal_cache = (
                fy_cache
                if fy_cache is not None
                else TTMRatiosAnalyzer._infer_fiscal_year_start_cache(working)
            )
            working = TTMRatiosAnalyzer._ensure_quarter_date_alignment(
                working,
                fiscal_cache=fiscal_cache,
                logger=logger,
            )
        TTMRatiosAnalyzer._fiscal_cache = fiscal_cache
        meta_columns = list(TTMRatiosAnalyzer._META_DROP_COLUMNS)
        if keep_dates:
            for col in ("start_date", "end_date", "filing_date"):
                if col in meta_columns:
                    meta_columns.remove(col)
        elif not drop_filing_date and "filing_date" in meta_columns:
            meta_columns.remove("filing_date")

        working = working.set_index("period").sort_index()
        return working

    @staticmethod
    def _ensure_quarter_date_alignment(
        df: pd.DataFrame,
        fiscal_cache: Optional[Dict[tuple[str, int], pd.Timestamp]] = None,
        quarter_gap_tolerance: Optional[int] = None,
        fiscal_span_tolerance: Optional[int] = None,
        cross_year_tolerance: Optional[int] = None,
        logger: Optional[Callable[[str, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Keep start/end/filing dates while enforcing fiscal year boundary rules.
        """
        if df.empty:
            return df

        def _log(msg: str, level: str = "warning") -> None:
            if logger:
                logger(msg, level=level)
            else:
                # Only issue warnings for warning level and above, not debug
                if level != "debug":
                    warnings.warn(msg, RuntimeWarning, stacklevel=3)

        required = {"ticker", "effective_fiscal_year", "quarter"}
        missing = [col for col in required if col not in df.columns]
        if missing:
            _log(
                f"Cannot enforce quarter date alignment because columns {missing} are missing.",
                level="warning",
            )
            return df

        working = df.copy()

        def _normalize(series: pd.Series) -> pd.Series:
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            return dt.dt.tz_convert(None).dt.normalize()

        for col in ("start_date", "end_date", "filing_date"):
            if col not in working.columns:
                working[col] = pd.NaT
            working[col] = _normalize(working[col])

        working["effective_fiscal_year"] = pd.to_numeric(
            working["effective_fiscal_year"], errors="coerce"
        ).astype("Int64")
        working["quarter"] = pd.to_numeric(working["quarter"], errors="coerce").astype("Int64")

        start_original = working["start_date"].copy()
        end_original = working["end_date"].copy()

        if quarter_gap_tolerance is None:
            quarter_gap_tolerance = TTMRatiosAnalyzer._QUARTER_SPACING_TOLERANCE
        if fiscal_span_tolerance is None:
            fiscal_span_tolerance = TTMRatiosAnalyzer._FISCAL_YEAR_SPAN_TOLERANCE
        if cross_year_tolerance is None:
            cross_year_tolerance = TTMRatiosAnalyzer._CROSS_YEAR_ALIGNMENT_TOLERANCE

        fiscal_cache = (
            fiscal_cache
            if fiscal_cache is not None
            else TTMRatiosAnalyzer._infer_fiscal_year_start_cache(working)
        )
        if not fiscal_cache:
            _log(
                "Fiscal year cache was empty; returning normalized date columns without adjustments.",
                level="warning",
            )
            for col in ("start_date", "end_date", "filing_date"):
                working[col] = working[col].dt.date
            return working

        def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            try:
                ts = pd.Timestamp(value)
            except Exception:
                return None
            if pd.isna(ts):
                return None
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts.normalize()

        def _days_diff(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> Optional[int]:
            if a is None or b is None or pd.isna(a) or pd.isna(b):
                return None
            return abs((a - b).days)

        def _warn(message: str) -> None:
            _log(message, level="debug")

        for (ticker, fy), grp in working.groupby(["ticker", "effective_fiscal_year"], dropna=False):
            if ticker is None or pd.isna(fy):
                continue
            try:
                fy_int = int(fy)
            except (TypeError, ValueError):
                continue
            ticker_key = str(ticker).upper()
            fy_start = _to_timestamp(fiscal_cache.get((ticker_key, fy_int)))
            if fy_start is None:
                continue
            next_start = _to_timestamp(fiscal_cache.get((ticker_key, fy_int + 1)))
            if next_start is None:
                next_start = (fy_start + pd.DateOffset(months=12)).normalize()

            expected_starts = {
                q: (fy_start + pd.DateOffset(months=3 * (q - 1))).normalize() for q in range(1, 5)
            }
            expected_ends: Dict[int, pd.Timestamp] = {}
            for q in range(1, 4):
                expected_ends[q] = (expected_starts[q + 1] - pd.Timedelta(days=1)).normalize()
            expected_ends[4] = (next_start - pd.Timedelta(days=1)).normalize()

            actual_start_map: Dict[int, Optional[pd.Timestamp]] = {}
            actual_end_map: Dict[int, Optional[pd.Timestamp]] = {}

            for q in range(1, 5):
                q_rows = grp[grp["quarter"] == q]
                if q_rows.empty:
                    continue
                idx = q_rows.index
                actual_start = start_original.loc[idx[0]]
                actual_end = end_original.loc[idx[0]]
                actual_start_map[q] = actual_start
                actual_end_map[q] = actual_end

                expected_start = expected_starts.get(q)
                expected_end = expected_ends.get(q)

                if expected_start is not None:
                    diff = _days_diff(actual_start, expected_start)
                    if diff is not None and diff > quarter_gap_tolerance and pd.notna(actual_start):
                        _warn(
                            f"start_date misalignment for {ticker_key} FY{fy_int} Q{q}: "
                            f"{actual_start.date()} vs expected {expected_start.date()} (>{quarter_gap_tolerance}d)"
                        )
                    working.loc[idx, "start_date"] = expected_start

                if expected_end is not None:
                    diff = _days_diff(actual_end, expected_end)
                    if diff is not None and diff > quarter_gap_tolerance and pd.notna(actual_end):
                        _warn(
                            f"end_date misalignment for {ticker_key} FY{fy_int} Q{q}: "
                            f"{actual_end.date()} vs expected {expected_end.date()} (>{quarter_gap_tolerance}d)"
                        )
                    working.loc[idx, "end_date"] = expected_end

            for q in range(1, 4):
                exp_start_gap = _days_diff(expected_starts.get(q + 1), expected_starts.get(q))
                s_now = actual_start_map.get(q)
                s_next = actual_start_map.get(q + 1)
                if exp_start_gap is not None and s_now is not None and s_next is not None:
                    gap = abs((s_next - s_now).days)
                    if abs(gap - exp_start_gap) > quarter_gap_tolerance:
                        _warn(
                            f"Quarter start spacing off for {ticker_key} FY{fy_int} between Q{q} and Q{q+1}: "
                            f"{gap}d vs expected {exp_start_gap}d"
                        )

                exp_end_gap = _days_diff(expected_ends.get(q + 1), expected_ends.get(q))
                e_now = actual_end_map.get(q)
                e_next = actual_end_map.get(q + 1)
                if exp_end_gap is not None and e_now is not None and e_next is not None:
                    gap = abs((e_next - e_now).days)
                    if abs(gap - exp_end_gap) > quarter_gap_tolerance:
                        _warn(
                            f"Quarter end spacing off for {ticker_key} FY{fy_int} between Q{q} and Q{q+1}: "
                            f"{gap}d vs expected {exp_end_gap}d"
                        )

            q1_actual = actual_start_map.get(1)
            q4_actual_end = actual_end_map.get(4)
            expected_year_span = _days_diff(next_start, fy_start)
            if (
                expected_year_span is not None
                and q1_actual is not None
                and q4_actual_end is not None
            ):
                actual_span = abs((q4_actual_end - q1_actual).days)
                if abs(actual_span - expected_year_span) > fiscal_span_tolerance:
                    _warn(
                        f"Fiscal span mismatch for {ticker_key} FY{fy_int}: "
                        f"Q1 start to Q4 end is {actual_span}d vs expected {expected_year_span}d"
                    )

            if q4_actual_end is not None and next_start is not None:
                boundary_diff = _days_diff(q4_actual_end + pd.Timedelta(days=1), next_start)
                if boundary_diff is not None and boundary_diff > fiscal_span_tolerance:
                    _warn(
                        f"Q4 boundary mismatch for {ticker_key} FY{fy_int}: "
                        f"Q4 end + 1d differs from next FY start by {boundary_diff}d"
                    )

        def _warn_year_over_year(series: pd.Series, label: str) -> None:
            payload = pd.DataFrame(
                {
                    "ticker": working["ticker"],
                    "quarter": working["quarter"],
                    "year": working["effective_fiscal_year"],
                    label: series,
                }
            ).dropna(subset=[label, "quarter", "year"])

            if payload.empty:
                return

            payload = payload.sort_values(["ticker", "quarter", "year"])
            for (ticker, quarter), g in payload.groupby(["ticker", "quarter"]):
                prev_val: Optional[pd.Timestamp] = None
                prev_year: Optional[int] = None
                for _, row in g.iterrows():
                    ts = row[label]
                    year = row["year"]
                    if prev_val is None:
                        prev_val = ts
                        prev_year = year
                        continue
                    year_gap = int(year) - int(prev_year)
                    if year_gap <= 0:
                        prev_val = ts
                        prev_year = year
                        continue
                    diff = abs((ts - prev_val).days)
                    expected = 365 * year_gap
                    if abs(diff - expected) > cross_year_tolerance:
                        _warn(
                            f"Year-over-year {label} drift for {ticker} Q{int(quarter)}: "
                            f"{diff}d vs expected {expected}d"
                        )
                    prev_val = ts
                    prev_year = year

        _warn_year_over_year(start_original, "start_date")
        _warn_year_over_year(end_original, "end_date")

        for col in ("start_date", "end_date", "filing_date"):
            working[col] = working[col].dt.date

        return working

    @staticmethod
    def _fetch_earnings_calendar(
        tickers: Union[str, Sequence[str]],
        owner: str = "post-no-preference",
        database: str = "earnings",
    ) -> pd.DataFrame:
        """
        Fetch earnings release dates for the requested tickers via the DoltHub SQL API.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        clean: List[str] = []
        for ticker in tickers:
            if ticker is None:
                continue
            tick = str(ticker).strip().upper()
            if tick:
                clean.append(tick)

        clean = sorted(set(clean))
        if not clean:
            return pd.DataFrame(columns=["ticker", "session", "earnings_release_date"])

        escaped = [t.replace("'", "''") for t in clean]
        in_clause = ", ".join(f"'{t}'" for t in escaped)
        query = f"""
        SELECT
          act_symbol AS ticker,
          `when` AS session,
          date AS earnings_release_date
        FROM earnings_calendar
        WHERE act_symbol IN ({in_clause})
        ORDER BY date;
        """

        response = requests.get(
            f"https://www.dolthub.com/api/v1alpha1/{owner}/{database}",
            params={"q": query},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        if not rows:
            return pd.DataFrame(columns=["ticker", "session", "earnings_release_date"])

        df = pd.DataFrame(rows)
        df = df.rename(
            columns={
                "act_symbol": "ticker",
                "when": "session",
                "date": "earnings_release_date",
            }
        )
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["earnings_release_date"] = pd.to_datetime(
            df["earnings_release_date"], errors="coerce"
        )
        df = df.dropna(subset=["earnings_release_date"])
        df = df.sort_values("earnings_release_date").reset_index(drop=True)
        return df

    def _apply_earnings_release_alignment(
        self,
        ticker: str,
        df: pd.DataFrame,
        owner: str = "post-no-preference",
        database: str = "earnings",
    ) -> pd.DataFrame:
        """
        Attach earnings release dates (from DoltHub) to each quarter without overwriting filing_date.
        """
        if df is None or df.empty:
            return df

        required = {"ticker", "effective_fiscal_year", "quarter", "end_date"}
        missing = [col for col in required if col not in df.columns]
        if missing:
            self._log(
                f"Cannot align earnings dates because columns {missing} are missing.",
                level="warning",
            )
            return df

        working = df.copy()

        for col in ("start_date", "end_date", "filing_date"):
            if col not in working.columns:
                working[col] = pd.NaT
            working[col] = pd.to_datetime(working[col], errors="coerce").dt.normalize()

        if "earnings_release_date" not in working.columns:
            working["earnings_release_date"] = pd.NaT
        working["earnings_release_date"] = pd.to_datetime(
            working["earnings_release_date"], errors="coerce"
        ).dt.normalize()
        
        if "earnings_release_session" not in working.columns:
            working["earnings_release_session"] = pd.NA

        ticker_norm_col = "_ticker_norm"
        working[ticker_norm_col] = working["ticker"].apply(
            lambda v: str(v).strip().upper() if pd.notna(v) else None
        )

        tickers = sorted({t for t in working[ticker_norm_col].dropna().unique() if t})
        if not tickers:
            working.drop(columns=[ticker_norm_col], inplace=True)
            for col in ("start_date", "end_date", "filing_date", "earnings_release_date"):
                working[col] = working[col].dt.date
            return working

        try:
            earnings = TTMRatiosAnalyzer._fetch_earnings_calendar(
                tickers, owner=owner, database=database
            )
        except Exception as exc:
            self._log(
                f"Unable to fetch earnings calendar: {exc}",
                level="warning",
            )
            working.drop(columns=[ticker_norm_col], inplace=True)
            for col in ("start_date", "end_date", "filing_date", "earnings_release_date"):
                working[col] = working[col].dt.date
            return working

        if earnings.empty:
            self._log(
                "Earnings calendar returned no rows; earnings release dates were not adjusted.",
                level="warning",
            )
            working.drop(columns=[ticker_norm_col], inplace=True)
            for col in ("start_date", "end_date", "filing_date", "earnings_release_date"):
                working[col] = working[col].dt.date
            return working

        earnings["earnings_release_date"] = pd.to_datetime(
            earnings["earnings_release_date"], errors="coerce"
        ).dt.normalize()
        earnings = earnings.dropna(subset=["earnings_release_date"])

        if earnings.empty:
            self._log(
                "All earnings release dates were NaT after parsing; earnings dates unchanged.",
                level="warning",
            )
            working.drop(columns=[ticker_norm_col], inplace=True)
            for col in ("start_date", "end_date", "filing_date", "earnings_release_date"):
                working[col] = working[col].dt.date
            return working

        max_lag = TTMRatiosAnalyzer._EARNINGS_RELEASE_MAX_LAG
        min_spacing, max_spacing = TTMRatiosAnalyzer._EARNINGS_RELEASE_SPACING_BOUNDS

        def _warn(msg: str) -> None:
            self._log(msg, level="warning")

        assignments: Dict[Any, pd.Timestamp] = {}

        for ticker_key, group in working.groupby(ticker_norm_col):
            if ticker_key is None:
                continue

            ticker_releases = (
                earnings[earnings["ticker"] == ticker_key]
                .sort_values("earnings_release_date")
                .reset_index(drop=True)
            )
            if ticker_releases.empty:
                self._log(
                    f"No earnings releases available for {ticker_key}; earnings dates left unchanged.",
                    level="debug",
                )
                continue

            releases = list(ticker_releases["earnings_release_date"])
            used: Set[int] = set()

            ordered_group = group.sort_values(["effective_fiscal_year", "quarter"])

            for idx, row in ordered_group.iterrows():
                end_date = row["end_date"]
                if pd.isna(end_date):
                    continue

                candidate_idx: Optional[int] = None
                candidate_date: Optional[pd.Timestamp] = None

                for j, rd in enumerate(releases):
                    if j in used:
                        continue
                    lag = (rd - end_date).days
                    if lag < 0 or lag > max_lag:
                        continue
                    candidate_idx = j
                    candidate_date = rd
                    break

                if candidate_date is None:
                    self._log(
                        f"Unable to align earnings release for {ticker_key} "
                        f"FY{row['effective_fiscal_year']} Q{row['quarter']}.",
                        level="debug",
                    )
                    continue

                used.add(candidate_idx)

                existing_filing = working.at[idx, "filing_date"]
                if pd.notna(existing_filing):
                    delta = (existing_filing - candidate_date).days
                    if delta < 0:
                        _warn(
                            f"Existing filing date {existing_filing.date()} precedes "
                            f"earnings release {candidate_date.date()} for "
                            f"{ticker_key} FY{row['effective_fiscal_year']} Q{row['quarter']}."
                        )
                    elif delta > TTMRatiosAnalyzer._EARNINGS_FILING_TOLERANCE:
                        _warn(
                            f"Filing date {existing_filing.date()} is {delta}d after "
                            f"earnings release {candidate_date.date()} for "
                            f"{ticker_key} FY{row['effective_fiscal_year']} Q{row['quarter']}."
                        )

                working.at[idx, "earnings_release_date"] = candidate_date
                if "session" in ticker_releases.columns and candidate_idx is not None:
                    session_value = ticker_releases.iloc[candidate_idx]["session"]
                    working.at[idx, "earnings_release_session"] = session_value
                assignments[idx] = candidate_date

            for fy, fy_group in ordered_group.groupby("effective_fiscal_year"):
                fy_releases = (
                    working.loc[fy_group.index, "earnings_release_date"]
                    .dropna()
                    .sort_values()
                )
                if fy_releases.empty:
                    continue
                gaps = (
                    fy_releases.iloc[1:].values - fy_releases.iloc[:-1].values
                ).astype("timedelta64[D]").astype(int)
                for gap in gaps:
                    if gap < min_spacing or gap > max_spacing:
                        _warn(
                            f"Earnings releases for {ticker_key} FY{fy} are {gap}d apart; "
                            f"expected {min_spacing}-{max_spacing}d."
                        )
        if "filing_date" in working.columns and "earnings_release_date" in working.columns:
            mask = working["filing_date"].isna() & working["earnings_release_date"].notna()
            if mask.any():
                self._log(
                    f"Filling {mask.sum()} NaT filing_date values with earnings_release_date.",
                    level="debug",
                )
                working.loc[mask, "filing_date"] = working.loc[mask, "earnings_release_date"]
        working.drop(columns=[ticker_norm_col], inplace=True)
        
        for col in ("start_date", "end_date", "filing_date", "earnings_release_date"):
            working[col] = working[col].dt.date

        if not assignments:
            self._log("No earnings release dates were matched to any periods.", level="debug")
        else:
            unmatched = set(working.index) - set(assignments.keys())
            if unmatched:
                self._log(
                    f"{len(unmatched)} period rows did not receive an earnings release match.",
                    level="debug",
                )

        if "earnings_release_date" in working.columns:
            self._log(
                "[apply_earnings_release_alignment] DataFrame has earnings_release_date column",
                level="debug",
            )
        return working

    def _build_quarterized_bucket_views(
        self,
        raw: Optional[pd.DataFrame] = None,
        provider: str = "finn",
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        working = raw.copy()
        self._map_fields_to_buckets(working.columns, ticker=ticker)
        canonical = self._prepare_canonical_financials(working)
        self._apply_bucket_aggregates(canonical)
        bucket_eff = self._attach_effective_fiscal_year(canonical)
        fy_cache = self._infer_fiscal_year_start_cache(bucket_eff)
        self._fy_cache_ = fy_cache
        bucket_eff["period_type"] = self._classify_period_type_vectorized(bucket_eff, fy_cache)

        meta_cols = [c for c in self._RAW_META_COLUMNS if c in bucket_eff.columns]
        metric_cols = [c for c in BUCKET_METRIC_MAP.values() if c in bucket_eff.columns]
        cols = meta_cols + metric_cols
        if provider == "poly":
            cols = [c for c in cols if c not in {"access_number", "quarter", "form", "accepted_date"}]
        bucket_eff = bucket_eff.loc[:, cols]
        if metric_cols:
            bucket_eff[metric_cols] = bucket_eff[metric_cols].replace(0.0, np.nan)
        else:
            bucket_eff = bucket_eff.replace(0.0, np.nan)
        self.bucket_eff = bucket_eff.copy()
        bucket_q = self._quarterize_flows(bucket_eff)
        # Fix share count outliers after quarterisation
        bucket_q = self._fix_share_count_outliers(bucket_q)
        quarterised = self._back_out_periods(
            bucket_q, drop_filing_date=False, keep_dates=True, fy_cache=fy_cache, logger=self._log
        )
        return quarterised

    def _load_financials_merged(
        self,
        ticker: str,
        start_dt: str = "2020-01-01",
        end_dt: str = "2025-11-23",
        provider_preferred: Sequence[str] = ("finn", "poly"),
        equivalent_tickers: Optional[Sequence[str]] = None,
        fast_fetch: bool = True,
        return_raw : bool = False,
    ) -> Tuple[pd.DataFrame, Dict[tuple[Any, Any], List[str]]]:
        """
        Load financials from Finnhub and Polygon, merging and quarterising them.
        fast_fetch uses vectorized classification and cached alias maps.
        """
        print('Set [RETURN_FALSE] to True to get raw dataframe')
        base_ticker = str(ticker).strip().upper()
        equivalents = tuple(
            str(t).strip().upper() for t in (equivalent_tickers or []) if str(t).strip()
        )

        def _fetch_provider(name: str) -> pd.DataFrame:
            if equivalents:
                return self._load_financials_for_provider_multi(
                    base_ticker, provider=name, alias_tickers=equivalents
                )
            return self._load_financials_for_provider(base_ticker, provider=name)

        provider_alias = {
            "finn": "finn",
            "finnhub": "finn",
            "poly": "poly",
            "polygon": "poly",
        }
        preferred = [provider_alias.get(p.lower(), p.lower()) for p in provider_preferred]
        preferred = [p for p in preferred if p in {"finn", "poly"}]
        if not preferred:
            preferred = ["finn", "poly"]

        provider_frames: Dict[str, pd.DataFrame] = {}
        prev_fast = getattr(self, "_fast_fetch", False)
        self._fast_fetch = bool(fast_fetch)
        try:
            for name in {"finn", "poly"}:
                 provider_frames[name] = _fetch_provider(name)

            finn = provider_frames.get("finn", pd.DataFrame())
            poly = provider_frames.get("poly", pd.DataFrame())

            def _attach_and_infer_effective_fiscal_year(df: pd.DataFrame, provider: str) -> pd.DataFrame:
                df_fin = self._attach_effective_fiscal_year(df)
                fy_cache = self._infer_fiscal_year_start_cache(df_fin)
                self._fy_cache_ = fy_cache
                if fast_fetch:
                    df_fin["period_type"] = self._classify_period_type_vectorized(df_fin, fy_cache)
                else:
                    df_fin["period_type"] = df_fin.apply(
                        lambda row: self._classify_period_type(row, fy_cache), axis=1
                    )
                df_fin["provider"] = provider
                return df_fin

            def clean_combined_raw(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                df["fiscal_period"] = df["fiscal_period"].str.replace("Q0", "Q4")
                df["quarter"] = df["fiscal_period"].map(TTMRatiosAnalyzer._PERIOD_MAP).astype("Int64")
                df["period_key"] = (
                    df["ticker"].astype(str)
                    + "_"
                    + df["effective_fiscal_year"].astype(str)
                    + "_"
                    + df["quarter"].astype(str)
                    + "_"
                    + df["period_type"]
                    + "_"
                    + df["provider"]
                )
                return df

            eff_frames: List[pd.DataFrame] = []
            if not finn.empty:
                eff_frames.append(_attach_and_infer_effective_fiscal_year(finn, "finn"))
            if not poly.empty:
                eff_frames.append(_attach_and_infer_effective_fiscal_year(poly, "poly"))

            if not eff_frames:
                return pd.DataFrame(), {}

            eff_all = pd.concat(eff_frames, ignore_index=True)
            eff_all = clean_combined_raw(eff_all)

            meta_cols = [col for col in self._MERGED_META_COLUMNS if col in eff_all.columns]
            eff_meta = eff_all[meta_cols].copy()

            if "filing_date" in eff_meta.columns:
                filing_dates = pd.to_datetime(eff_meta["filing_date"], errors="coerce")
                start_ts = pd.to_datetime(start_dt)
                end_ts = pd.to_datetime(end_dt)
                mask = ((filing_dates >= start_ts) & (filing_dates <= end_ts)) | filing_dates.isna()
                if mask.any():
                    eff_meta = eff_meta.loc[mask].copy()

            merged, missing = self._ensure_quarterly_coverage(
                eff_meta, preferred_providers=tuple(p for p in preferred if p != "sec")
            )
            financials = eff_all[eff_all["period_key"].isin(merged["period_key"])].copy()

            financials_quartised = self._build_quarterized_bucket_views(
                financials, provider=preferred[0], ticker=base_ticker
            )
            if return_raw:
                return financials_quartised, missing, financials
            return financials_quartised, missing
        finally:
            self._fast_fetch = prev_fast

    # ---------------------------- #
    # SEC helper utilities (_sec)
    # ---------------------------- #
    def finalize_quarterised_financials(
        self,
        financials_quarterised: pd.DataFrame,
        *,
        ticker: str,
        start_dt: str = "2020-01-01",
        end_dt: str = "2025-11-23",
        earnings_ticker: Optional[str] = None,
        scale_output: bool = True,
        scale_units: Literal["dollars", "thousands", "millions", "billions"] = "millions",
        round_output: bool = False,
        round_sigfigs: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the downstream processing normally performed inside
        `get_financial_statement_quarterised` starting from an already quarterised
        dataframe (post `_quarterize_flows` / `_back_out_periods`).
        """
        if financials_quarterised is None or financials_quarterised.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty, empty
        
        if ("effective_fiscal_year" not in financials_quarterised.columns 
            and "period_key" in financials_quarterised.columns
            and "ticker" not in financials_quarterised.columns):
            raise ValueError("effective_fiscal_year and ticker columns are required")

        working = financials_quarterised.copy()

        if earnings_ticker:
            override = str(earnings_ticker).strip().upper()
            working_align = working.copy()
            working_align["_original_ticker"] = working_align.get("ticker")
            working_align["ticker"] = override
            aligned = self._apply_earnings_release_alignment(df=working_align, ticker=earnings_ticker)
            aligned["ticker"] = aligned["_original_ticker"]
            working = aligned.drop(columns=["_original_ticker"])
        else:
            working = self._apply_earnings_release_alignment(df=working, ticker=ticker)

        def _compute_ratio(numer_col: str, denom_col: str, target_col: str, precision: Optional[int] = None) -> None:
            if numer_col in working.columns and denom_col in working.columns:
                res = (working[numer_col] / working[denom_col]).replace([np.inf, -np.inf], np.nan)
                if precision is not None:
                    res = res.round(precision)
                working[target_col] = res

        _compute_ratio("is_net_income_loss", "is_basic_average_shares", "eps_basic", precision=4)
        _compute_ratio("is_net_income_loss", "is_diluted_average_shares", "eps_diluted", precision=4)

        if (
            "bs_total_liabilities" not in working.columns
            or working["bs_total_liabilities"].isna().all()
        ):
            if (
                "bs_current_liabilities" in working.columns
                and "bs_noncurrent_liabilities" in working.columns
                and "bs_interest_bearing_debt" in working.columns
            ):
                working["bs_total_liabilities"] = (
                    working["bs_current_liabilities"].fillna(0)
                    + working["bs_noncurrent_liabilities"].fillna(0)
                ).replace([np.inf, -np.inf, 0], np.nan).round(2)

        if (
            "bs_cash_and_cash_equivalents" in working.columns
            and "bs_short_term_investments" in working.columns
        ):
            working["bs_cash_and_equivalents_short_term_investments"] = (
                working["bs_cash_and_cash_equivalents"].fillna(0)
                + working["bs_short_term_investments"].fillna(0)
            ).replace([np.inf, -np.inf, 0], np.nan).round(2)

        if (
            "bs_interest_bearing_debt" in working.columns
            and "bs_cash_and_equivalents_short_term_investments" in working.columns
        ):
            working["bs_net_debt"] = (
                working["bs_interest_bearing_debt"].fillna(0.0)
                - working["bs_cash_and_equivalents_short_term_investments"].fillna(0.0)
            ).replace([np.inf, -np.inf], np.nan).round(2)

        if (
            "cf_net_cash_flow_from_operating_activities" in working.columns
            and "cf_capital_expenditures" in working.columns
        ):
            working["cf_free_cash_flow"] = (
                working["cf_net_cash_flow_from_operating_activities"]
                - working["cf_capital_expenditures"]
            ).replace([np.inf, -np.inf], np.nan).round(2)

        if (
            "earnings_release_session" in working.columns
            and "earnings_release_date" in working.columns
        ):
            period_index = working.index
            working = working.reset_index()
            price_lookup = str(earnings_ticker).strip().upper() if earnings_ticker else ticker
            price_ser = self._align_prices_to_filing_dates(
                ticker=price_lookup,
                date_col="earnings_release_date",
                fins=working,
            )
            price_ser = price_ser.reset_index().dropna(axis=0)
            working["earnings_release_date"] = pd.to_datetime(
                working["earnings_release_date"]
            )
            working = pd.merge(
                left=working,
                right=price_ser,
                left_on="earnings_release_date",
                right_on="earnings_release_date",
                how="left",
            )
            if "period" in working.columns:
                working = working.set_index("period")
            elif len(working) == len(period_index):
                working.index = period_index

        if (
            "is_diluted_average_shares" in working.columns
            and "close" in working.columns
        ):
            working["bs_market_cap_diluted"] = (
                working["is_diluted_average_shares"] * working["close"]
            )
        if "is_basic_average_shares" in working.columns and "close" in working.columns:
            working["bs_market_cap_basic"] = (
                working["is_basic_average_shares"] * working["close"]
            )
        if ("is_gross_profit" in working.columns 
            and "is_operating_income_loss" in working.columns
            ):
            working["is_operating_expense_alt"] = (
                working["is_gross_profit"] - working["is_operating_income_loss"]
            )
        if ("is_research_and_development" in working.columns
            and "is_selling_general_and_administrative_expenses" in working.columns
            and "is_operating_expense_alt" in working.columns
            and "is_depreciation_and_amortization" in working.columns
            ):
            working["is_operating_expense_residual"] = (
                working["is_operating_expense_alt"]
                - working["is_research_and_development"].fillna(0)
                - working["is_selling_general_and_administrative_expenses"].fillna(0)
                - working["is_depreciation_and_amortization"].fillna(0)
            )
        if ("bs_current_liabilities" in working.columns
            and "bs_total_liabilities" in working.columns
            ):
            working["bs_noncurrent_liabilities_alt"] = (
                working["bs_total_liabilities"]
                - working["bs_current_liabilities"]
            )
            working["bs_noncurrent_liabilities"] = working["bs_noncurrent_liabilities"].fillna(working["bs_noncurrent_liabilities_alt"])
        bal_cols = ["bs_assets", "bs_total_liabilities", "bs_equity"]
        if all(c in working.columns for c in bal_cols):
            a = working["bs_assets"]
            l = working["bs_total_liabilities"]
            e = working["bs_equity"]

            mask_missing_assets = a.isna() & l.notna() & e.notna()
            working.loc[mask_missing_assets, "bs_assets"] = (
                l[mask_missing_assets] + e[mask_missing_assets]
            ).round(2)

            mask_missing_liab = l.isna() & a.notna() & e.notna()
            working.loc[mask_missing_liab, "bs_total_liabilities"] = (
                a[mask_missing_liab] - e[mask_missing_liab]
            ).round(2)

            mask_missing_equity = e.isna() & a.notna() & l.notna()
            working.loc[mask_missing_equity, "bs_equity"] = (
                a[mask_missing_equity] - l[mask_missing_equity]
            ).round(2)

        if ("bs_noncontrolling_interest_equity" in working.columns
            and working["bs_noncontrolling_interest_equity"].isna().any()
            and "bs_equity" in working.columns):
            if "bs_stockholders_equity_including_portion_attributable_to_noncontrolling_interest" in working.columns:
                working["bs_noncontrolling_interest_equity"] = np.where(
                    working["bs_noncontrolling_interest_equity"].isna(),
                    working["bs_stockholders_equity_including_portion_attributable_to_noncontrolling_interest"] - working["bs_equity"],                   
                    working["bs_noncontrolling_interest_equity"]
                )
            elif "bs_total_liabilities" in working.columns and "bs_liabilities_and_equity" in working.columns:
                working["bs_noncontrolling_interest_equity"] = np.where(
                    working["bs_noncontrolling_interest_equity"].isna(),
                    working["bs_liabilities_and_equity"] - working["bs_total_liabilities"] - working["bs_equity"],
                    working["bs_noncontrolling_interest_equity"]
                )
            else:
                working["bs_noncontrolling_interest_equity"] = np.nan 



        if ("is_depreciation_and_amortization" in working.columns
            and "cf_depreciation_and_amortization" in working.columns
            ):
            working["is_depreciation_and_amortization"] = np.where(working["is_depreciation_and_amortization"].isna(),
                                                                   working["cf_depreciation_and_amortization"],
                                                                   working["is_depreciation_and_amortization"])
        if "start_date" in working.columns:
            start_ts = pd.to_datetime(start_dt)
            end_ts = pd.to_datetime(end_dt)
            start_dates = pd.to_datetime(working["start_date"], errors="coerce")
            date_mask = (start_dates >= start_ts) & (start_dates <= end_ts)
            working_filtered = working.loc[date_mask].copy()
        else:
            working_filtered = working
        




        is_statement, balance_sheet, cashflow, misc_df = self._statement_from_quarterised(
            working_filtered
        )

        drop_cols = [
            "filing_date",
            "effective_fiscal_year",
            "is_basic_earnings_per_share",
            "is_diluted_earnings_per_share",
            "period_type",
            "ticker",
            "accepted_date",
            "fiscal_period",
            "form",
            "timeframe",
            "end_date", 
            "start_date",
            "close",
            "earnings_release_date",
            "earnings_release_session"
        ]
        drop_cols_ = [c for c in drop_cols if c in working_filtered.columns]
        final_financials = working_filtered.drop(columns=drop_cols_, errors="ignore")

        def _scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
            """Scale numeric columns by the requested unit while leaving meta/eps/share/price columns unchanged."""
            if frame is None or frame.empty or not scale_output:
                return frame

            factor_map = {
                "dollars": 1.0,
                "thousands": 1e3,
                "millions": 1e6,
                "billions": 1e9,
            }
            factor = factor_map.get(scale_units, 1.0)
            if factor == 0:
                factor = 1.0

            # Columns to skip scaling
            meta_cols = set(self._META_DROP_COLUMNS) | set(self._MERGED_META_COLUMNS) | set(self._RAW_META_COLUMNS)
            skip_prefixes = ("eps",)  # EPS
            skip_contains = ("earnings_per_share", "shares", "date", "form","ticker")
            skip_exact = {"close", "price", "start_date", "end_date", "filing_date", "earnings_release_date", "accepted_date"}

            scaled = frame.copy()
            to_scale: List[str] = []
            for col in scaled.columns:
                if col in meta_cols:
                    continue
                if any(col.startswith(p) for p in skip_prefixes):
                    continue
                if any(s in col for s in skip_contains):
                    continue
                if col in skip_exact:
                    continue
                to_scale.append(col)
            if not to_scale:
                return scaled

            row_index = scaled.index.astype(str)
            row_skip_mask = (
                row_index.isin(skip_exact)
                | row_index.str.startswith(skip_prefixes)
                | row_index.str.contains("|".join(skip_contains))
                | row_index.isin(meta_cols)
            )

            for col in to_scale:
                ser_orig = scaled[col]
                ser_num = pd.to_numeric(ser_orig, errors="coerce")
                mask = ser_num.notna() & (~row_skip_mask)
                if mask.any():
                    ser_scaled = ser_orig.copy()
                    ser_scaled.loc[mask] = ser_num.loc[mask] / factor
                    scaled[col] = ser_scaled
            return scaled

        def _round_sigfig_frame(frame: pd.DataFrame) -> pd.DataFrame:
            """Round numeric columns to the requested significant figures."""
            if frame is None or frame.empty or not round_output:
                return frame
            rounded = frame.copy()
            meta_cols = set(self._META_DROP_COLUMNS) | set(self._MERGED_META_COLUMNS) | set(self._RAW_META_COLUMNS)
            skip_prefixes = ("eps",)
            skip_contains = ("earnings_per_share", "shares", "date")
            skip_exact = {"close", "price", "start_date", "end_date", "filing_date", "earnings_release_date", "accepted_date"}
            def _round_series(s: pd.Series) -> pd.Series:
                arr = pd.to_numeric(s, errors="coerce")
                def _round_val(x: float) -> float:
                    if pd.isna(x):
                        return np.nan
                    if x == 0:
                        return 0.0
                    order = math.floor(math.log10(abs(x)))
                    return round(x, int(round_sigfigs - order - 1))
                return arr.apply(_round_val)
            for col in rounded.columns:
                if col in meta_cols:
                    continue
                if any(col.startswith(p) for p in skip_prefixes):
                    continue
                if any(s in col for s in skip_contains):
                    continue
                if col in skip_exact:
                    continue
                ser = pd.to_numeric(rounded[col], errors="coerce")
                if ser.notna().any():
                    rounded[col] = _round_series(rounded[col])
            return rounded

        is_statement = _scale_frame(is_statement)
        balance_sheet = _scale_frame(balance_sheet)
        cashflow = _scale_frame(cashflow)
        misc_df = _scale_frame(misc_df)
        is_statement = _round_sigfig_frame(is_statement)
        balance_sheet = _round_sigfig_frame(balance_sheet)
        cashflow = _round_sigfig_frame(cashflow)
        misc_df = _round_sigfig_frame(misc_df)

        # Keep final_financials unscaled for downstream persistence/parquet
        return is_statement.loc[:, ::-1], balance_sheet.loc[:, ::-1], cashflow.loc[:, ::-1], misc_df.loc[:, ::-1], working_filtered

    def get_financial_statement_quarterised(
        self,
        ticker: str,
        start_dt: str = "2020-01-01",
        end_dt: str = "2025-11-27",
        see_bucket: bool = False,
        equivalent_tickers: Optional[Sequence[str]] = None,
        earnings_ticker: Optional[str] = None,
        fast_fetch: bool = True,
        *,
        scale_output: bool = True,
        scale_units: Literal["dollars", "thousands", "millions", "billions"] = "millions",
        round_output: bool = False,
        round_sigfigs: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        previous_capture = self._capture_bucket_components
        self._capture_bucket_components = bool(see_bucket)
        if not self._capture_bucket_components:
            self._bucket_component_store = {}
        financials_quarterised, missing = self._load_financials_merged(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            equivalent_tickers=equivalent_tickers,
            fast_fetch=fast_fetch,
        )
        self._capture_bucket_components = previous_capture
        if financials_quarterised.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty, financials_quarterised

        return self.finalize_quarterised_financials(
            financials_quarterised=financials_quarterised,
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            earnings_ticker=earnings_ticker,
            scale_output=scale_output,
            scale_units=scale_units,
            round_output=round_output,
            round_sigfigs=round_sigfigs,
        )

    def get_financial_statement_quarterised_unified(
        self,
        ticker: str,
        equivalent_tickers: Sequence[str],
        start_dt: str = "2020-01-01",
        end_dt: str = "2025-11-23",
        see_bucket: bool = False,
        earnings_ticker: Optional[str] = None,
        fast_fetch: bool = True,
        *,
        scale_output: bool = True,
        scale_units: Literal["dollars", "thousands", "millions", "billions"] = "millions",
        round_output: bool = False,
        round_sigfigs: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convenience wrapper that merges filings across equivalent tickers
        (e.g., GOOG + GOOGL) and returns a unified quarterised dataset.
        """
        return self.get_financial_statement_quarterised(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            see_bucket=see_bucket,
            equivalent_tickers=equivalent_tickers,
            earnings_ticker=earnings_ticker,
            fast_fetch=fast_fetch,
            scale_output=scale_output,
            scale_units=scale_units,
            round_output=round_output,
            round_sigfigs=round_sigfigs,
        )

    def save_financials_quarterised(
        self,
        tickers: List[str],
        output_dir: Optional[str] = "/Users/phillip/Desktop/Moon2/data/quarterised_fundamentals",
        start_dt: str = "2020-01-01",
        end_dt: str = "2025-11-23",
        equivalent_tickers_map: Optional[Dict[str, Sequence[str]]] = None,
        earnings_ticker_map: Optional[Dict[str, str]] = None,
        fast_fetch: bool = True,
        *,
        scale_output: bool = True,
        scale_units: Literal["dollars", "thousands", "millions", "billions"] = "millions",
        round_output: bool = False,
        round_sigfigs: int = 3,
    ) -> Dict[str, bool]:
        """
        Save quarterised financials dataframe to parquet files, one per ticker.
        
        This method processes a list of tickers, calls get_financial_statement_quarterised
        for each, and saves only the financials_quarterised dataframe to a parquet file
        in a folder per ticker. The method is robust to failures and continues processing
        even if individual tickers fail.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to process
        output_dir : Optional[str]
            Base directory for saving parquet files. If None, defaults to
            quarterised_fundamentals_dir
        start_dt : str
            Start date for financial data (default: "2020-01-01")
        end_dt : str
            End date for financial data (default: "2025-11-23")
        equivalent_tickers_map : Optional[Dict[str, Sequence[str]]]
            Optional mapping of ticker to equivalent tickers (e.g., {"GOOG": ["GOOGL"]})
        earnings_ticker_map : Optional[Dict[str, str]]
            Optional mapping of ticker to earnings ticker (e.g., {"GOOG": "GOOGL"})
        
        Returns:
        --------
        Dict[str, bool]
            Dictionary mapping ticker to success status (True if saved successfully, False otherwise)
        """
        if output_dir is None:
            output_dir = self.quarterised_fundamentals_dir
        
        output_path = Path(output_dir)
        
        results: Dict[str, bool] = {}
        
        for ticker in tickers:
            ticker_upper = ticker.upper().strip()
            try:
                self._log(f"Processing {ticker_upper}...", level="info")
                
                # Get equivalent tickers and earnings ticker if provided
                equivalent_tickers = None
                if equivalent_tickers_map and ticker_upper in equivalent_tickers_map:
                    equivalent_tickers = equivalent_tickers_map[ticker_upper]
                
                earnings_ticker = None
                if earnings_ticker_map and ticker_upper in earnings_ticker_map:
                    earnings_ticker = earnings_ticker_map[ticker_upper]
                
                # Call get_financial_statement_quarterised
                _, _, _, _, financials_quarterised = self.get_financial_statement_quarterised(
                    ticker=ticker_upper,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    equivalent_tickers=equivalent_tickers,
                    earnings_ticker=earnings_ticker,
                    fast_fetch=fast_fetch,
                    scale_output=scale_output,
                    scale_units=scale_units,
                    round_output=round_output,
                    round_sigfigs=round_sigfigs,
                )
                
                # Check if we got data
                if financials_quarterised.empty:
                    self._log(f"  Warning: {ticker_upper} returned empty dataframe", level="warning")
                    results[ticker_upper] = False
                    continue
                
                # Create ticker-specific folder
                ticker_folder = output_path / ticker_upper
                ticker_folder.mkdir(parents=True, exist_ok=True)
                
                # Save to parquet
                parquet_path = ticker_folder / f"{ticker_upper}_quarterised.parquet"
                financials_quarterised.to_parquet(parquet_path, index=True)
                
                self._log(f"  ✓ Saved {ticker_upper} to {parquet_path}", level="info")
                results[ticker_upper] = True
                
            except Exception as e:
                self._log(f"  ✗ Error processing {ticker_upper}: {str(e)}", level="error")
                results[ticker_upper] = False
                continue
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        self._log(f"\nSummary: {successful}/{total} tickers saved successfully", level="info")
        
        return results
    
    def analyze_raw_finnhub(self, year = None, metric = None):
        if not hasattr(self, 'raw_finnhub'):
            print("No raw_finnhub attribute found.")
            return None
        else:
            df = pd.concat(self.raw_finnhub).copy()
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Build raw -> canonical mapping and bucket memberships
            try:
                self._ensure_reverse_map_cached("finn")
                reverse_map = self._reverse_map
            except Exception:
                reverse_map = {}

            # Build canonical -> buckets map (may have multiple buckets)
            try:
                components_doc = self._load_metric_components()
                buckets_doc = components_doc.get("buckets", {}) or {}
                bucket_map: Dict[str, Set[str]] = {}
                for bucket_name, meta in buckets_doc.items():
                    metric = meta.get("metric")
                    comps = (meta.get("components") or {}).keys()
                    names = [metric] if metric else []
                    names.extend(comps)
                    for name in names:
                        if not name:
                            continue
                        norm = self._normalize_component_name(name)
                        bucket_map.setdefault(norm, set()).add(bucket_name)
            except Exception:
                bucket_map = {}

            # Normalize concept_raw similar to finnhub field normalization
            def _normalize_finnhub_field(field_name: str, ticker: Optional[str]) -> str:
                import re
                cleaned = str(field_name or "")
                ticker_upper = (ticker or "").upper()
                ticker_lower = ticker_upper.lower()
                pattern_colon = r"^[a-zA-Z]+:"
                cleaned = re.sub(pattern_colon, "", cleaned)
                if ticker_upper:
                    if cleaned.startswith(ticker_upper + ":") or cleaned.startswith(ticker_lower + ":"):
                        cleaned = cleaned.split(":", 1)[-1]
                    cleaned = re.sub(
                        f"^{re.escape(ticker_upper)}_|^{re.escape(ticker_lower)}_",
                        "",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                cleaned = re.sub(r"^us[-_]gaap_", "", cleaned, flags=re.IGNORECASE)
                return cleaned

            def _normalize_name(raw: str) -> str:
                s = str(raw or "").lower()
                for ch in [",", "(", ")", "/", "-", " ", "."]:
                    s = s.replace(ch, "_")
                while "__" in s:
                    s = s.replace("__", "_")
                return s.strip("_")

            def _canonical_from_raw(raw: str, ticker: Optional[str]) -> str:
                cleaned = _normalize_finnhub_field(raw, ticker)
                # Try normalized version first
                normalized = _normalize_name(cleaned)
                if normalized in reverse_map:
                    return reverse_map[normalized]
                # Try snake_case conversion (handles camelCase like CapitalLeaseObligationsCurrent)
                snake = self._to_snake_case(cleaned)
                norm_snake = _normalize_name(snake)
                if norm_snake in reverse_map:
                    return reverse_map[norm_snake]
                # Fallback to normalized version
                return normalized

            # Bucket lookup
            def _buckets_for_canonical(canonical: str) -> List[str]:
                norm = self._normalize_component_name(canonical)
                return sorted(bucket_map.get(norm, []))

            df["canonical_mapped"] = df.apply(
                lambda row: _canonical_from_raw(row.get("concept_raw"), row.get("symbol")), axis=1
            )
            df["bucket_list"] = df["canonical_mapped"].apply(_buckets_for_canonical)
            df["bucket"] = df["bucket_list"].apply(lambda lst: ",".join(lst) if lst else "")

            ticker = df['symbol'].unique()
            if len(ticker) == 1:
                df.to_parquet(f'/Users/phillip/Desktop/Moon2/data/finnhub_fundamentals/{ticker[0]}_raw_finnhub.parquet')
            if year != None:
                mask2 = df['year'] == year 
                df = df[mask2]
           
            if metric != None:
                mask = df['concept'].str.contains(metric, case=False, na=False) | df['concept_raw'].str.contains(metric, case=False, na=False)
                df = df[mask]
     
            return df

  
#### New Functions ####










#### End New Functions ####