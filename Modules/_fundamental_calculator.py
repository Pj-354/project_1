# _fundamental_calculator.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import numpy as np

Number = Optional[float]


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------
def _sum_non_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(vals)) if vals else None

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not math.isnan(float(x))


def _to_num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _sum_zeros(row: Dict[str, Any], fields: List[str]) -> Optional[float]:
    """
    Sum components, treating missing as 0.
    If *all* components are missing (None), return None (no info).
    """
    vals: List[float] = []
    seen_any = False
    for f in fields:
        v = _to_num(row.get(f))
        if v is not None:
            vals.append(v)
            seen_any = True
        else:
            # missing component -> 0
            vals.append(0.0)
    if not seen_any:
        return None
    return float(sum(vals))


def _first_non_null(row: Dict[str, Any], fields: List[str]) -> Optional[float]:
    """Return the first non-null numeric value in fields, or None."""
    for f in fields:
        v = _to_num(row.get(f))
        if v is not None:
            return v
    return None


def _enforce_sign(val: Optional[float], sign: str) -> Optional[float]:
    """
    Enforce sign conventions where economically unambiguous.
    sign = 'nonpositive' => result <= 0
    sign = 'nonnegative' => result >= 0
    """
    if val is None:
        return None
    if sign == "nonpositive" and val > 0:
        return -abs(val)
    if sign == "nonnegative" and val < 0:
        return abs(val)
    return val


# ------------------------------------------------------------
# Rule representation
# ------------------------------------------------------------

@dataclass
class FormulaRule:
    """
    One candidate way to compute a metric.

    - expr: string expression in terms of other fields, e.g.
        "is_revenue_from_goods + is_revenue_from_services"
      OR a special keyword "direct" meaning "just take the raw field".
    - fields: optional explicit dependency list (only needed if expr="direct").
    - require_all: if True -> if any dependency is None -> rule fails.
      if False -> missing are treated as 0 in sums, but NaNs propagate where appropriate.
    - enforce_sign: 'nonpositive', 'nonnegative', or None
    """
    expr: str = ""
    require_all: bool = False
    enforce_sign: Optional[str] = None
    func: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None


@dataclass
class MetricRule:
    """
    A metric can have multiple alternative formulas tried in order.
    We also store a human-readable description for debugging.
    """
    name: str
    formulas: List[FormulaRule] = field(default_factory=list)
    description: str = ""


# ------------------------------------------------------------
# Safe expression evaluation
# ------------------------------------------------------------

_ALLOWED_FUNCS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "max": max,
    "min": min,
}

def _eval_expr(expr: str, row: Dict[str, Any]) -> Optional[float]:
    """
    Evaluate a small arithmetic expression in terms of row fields.

    - Variables must be valid Python identifiers and correspond to keys in `row`.
    - Missing fields -> 0 (your "missing = economically zero" convention) in arithmetic.
    """
    # Build local environment: variable name -> numeric value (0 if missing)
    local_env: Dict[str, Any] = {}
    for k, v in row.items():
        if not isinstance(k, str):
            continue
        # Only allow identifiers
        if not k.replace("_", "").isalnum():
            continue
        val = _to_num(v)
        local_env[k] = 0.0 if val is None else val

    # Add allowed functions
    local_env.update(_ALLOWED_FUNCS)

    try:
        val = eval(expr, {"__builtins__": {}}, local_env)
    except Exception:
        return None

    return _to_num(val)


# ------------------------------------------------------------
# Core compute_metric
# ------------------------------------------------------------

def compute_metric(
    name: str,
    row: Dict[str, Any],
    rules: Dict[str, MetricRule],
) -> Tuple[Optional[float], str]:
    """
    Compute a metric using METRIC_RULES.

    Supports three kinds of formulas, tried in order for each metric:

      1) Callable rule: FormulaRule(func=callable)
         -> func(row_dict) must return Optional[float].

      2) "direct" rule: FormulaRule(expr="direct")
         -> just use the raw column value.

      3) Expression rule: FormulaRule(expr="a + b - c")
         -> evaluated via _eval_expr with missing treated as 0.

    Returns:
        (value, source_description)
    """
    rule = rules.get(name)
    raw_val = _to_num(row.get(name))

    # No explicit rule: return raw
    if rule is None:
        return raw_val, "raw"

    for fr in rule.formulas:
        val: Optional[float] = None

        # 1) Callable rule, if present
        func = getattr(fr, "func", None)
        if func is not None:
            try:
                val = func(row)
            except Exception:
                val = None

        # 2) "direct" rule
        elif fr.expr == "direct":
            if raw_val is None:
                continue
            val = raw_val

        # 3) Expression rule
        else:
            val = _eval_expr(fr.expr, row)

        # If require_all and evaluation failed -> skip this formula
        if fr.require_all and (val is None):
            continue

        # Enforce sign convention if requested
        enforce_sign = getattr(fr, "enforce_sign", None)
        if enforce_sign:
            val = _enforce_sign(val, enforce_sign)

        if val is not None:
            # Build a useful source tag
            if func is not None:
                src_expr = func.__name__
            else:
                src_expr = fr.expr or "<expr>"
            return val, f"rule:{name}:{src_expr}"

    # Fallback: raw value
    return raw_val, "raw"


# ------------------------------------------------------------
# METRIC_RULES: economic definitions
# ------------------------------------------------------------

METRIC_RULES: Dict[str, MetricRule] = {}

def _add_rule(name: str, description: str, formulas: List[FormulaRule]):
    METRIC_RULES[name] = MetricRule(name=name, description=description, formulas=formulas)


# ============ INCOME STATEMENT BUILDERS ============
# Gross profit
_add_rule(
    "is_gross_profit",
    "Gross profit = revenue - cost_of_revenue, fallback to reported gross profit.",
    [
        # Construct from revenue and cost
        FormulaRule(expr="is_revenues - is_cost_of_revenue", require_all=False),
        # Fallback: direct field
        FormulaRule(expr="direct"),
    ],
)

# Operating expenses (broad catch-all)


def compute_operating_expenses_from_components(row: Dict[str, Any]) -> Optional[float]:
    def num(key: str) -> Optional[float]:
        return _to_num(row.get(key))

    def first_non_null(*keys: str) -> Optional[float]:
        for k in keys:
            v = num(k)
            if v is not None:
                return v
        return None

    def sum_keys(keys: List[str]) -> Optional[float]:
        return _sum_non_none([num(k) for k in keys])

    # 1) Direct
    direct = first_non_null("operating_expenses")
    if direct is not None:
        return direct

    # 2) Aggregates vs parts (avoid SG&A double count)
    sgna = first_non_null(
        "selling_general_and_administrative_expenses",
        "selling_general_and_administrative",
        "selling_general_and_administrative_expenses",  # duplicate spelling included
    )
    marketing = sum_keys([
        "marketing_and_advertising_expense",
        "marketing_expense",
        "marketingexpense",
        "selling_and_marketing_expense",
    ])
    gna = num("general_and_administrative")

    # 3) Other buckets from metric_components2
    # R&D: check all alternative field names (first non-null wins)
    rd = first_non_null(
        "research_and_development",
        "research_and_development_expense_excluding_acquired_in_process_cost",
        "research_and_development_in_process",
        "technology_and_development_expense",
    )
    tech = sum_keys([
        "technology_and_content_expense",
        "technology_and_infrastructure_expense",
        "technology_communications_and_equipment_expense",
        "information_technology_and_data_processing",
    ])
    fulfillment = num("fulfillment_expense")
    labor = num("labor_and_related_expense")
    da = num("depreciation_and_amortization")
    contract_amort = num("capitalized_contract_cost_amortization")

    impairment = sum_keys([
        "asset_impairment_charges",
        "other_asset_impairment_charges",
        "goodwill_impairment_loss",
    ])
    restructuring = sum_keys([
        "restructuring_charges",
        "restructuring_and_other_expenses",
        "restructuringcostsandassetimpairmentcharges",
        "restructuring_costs_and_asset_impairment_charges",
        "restructuring_settlement_and_impairment_provisions",
        "restructuringcharges_noncash",
        "share_based_payment_arrangement_expense_restructuring",
    ])
    provisions = sum_keys([
        "provision_for_doubtful_accounts",
        "provision_for_loan_lease_and_other_losses",
        "provision_for_loan_lease_and_other_losses_excl_policy_conformity",
    ])
    legal = sum_keys([
        "litigationsettlementexpense",
        "litigation_settlement_expense",
        "litigation_settlement_gross",
    ])
    other_ops = sum_keys([
        "other_operating_income_expenses",
        "other_cost_and_expense_operating",
        "other_costandexpenseoperating",
    ])
    misc = sum_keys([
        "professional_and_contract_services_expense",
        "costmethod_investments_other_than_temporary_impairment",
        "stock_option_plan_expense",
        "labor_and_related_expense",
    ])

    parts: List[Optional[float]] = []
    if sgna is not None:
        parts.append(sgna)
    else:
        if marketing is not None:
            parts.append(marketing)
        if gna is not None:
            parts.append(gna)

    for v in (
        rd, tech, fulfillment, labor, da, contract_amort,
        impairment, restructuring, provisions, legal, other_ops, misc
    ):
        if v is not None:
            parts.append(v)

    total = _sum_non_none(parts)
    if total is not None:
        return total

    # 4) Last-resort broad aggregate (can include COGS; use only if nothing else)
    return first_non_null("costs_and_expenses")


# Replace the existing operating-expenses rule with:
_add_rule(
    "is_operating_expenses",
    "Operating expenses with granular fallback built from metric_components2; "
    "avoids double counting by preferring SG&A aggregates over their parts.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(func=compute_operating_expenses_from_components),
        FormulaRule(expr="gross_profit - operating_income_loss", require_all=False),
    ],
)


# Operating income / loss
_add_rule(
    "is_operating_income_loss",
    "Operating income; enforce sign convention (loss <= 0, profit >= 0).",
    [
        # Prefer direct field; allow negative when loss
        FormulaRule(expr="direct"),
        # Fallback: gross profit - operating expenses
        FormulaRule(expr="is_gross_profit - is_operating_expenses", require_all=False),
    ],
)

# Interest expense – operating
_add_rule(
    "is_interest_expense_operating",
    "Interest expense from operations; enforce as non-positive.",
    [
        FormulaRule(expr="direct", enforce_sign="nonpositive"),
        # Fallback: total interest expense if only generic field present
        FormulaRule(expr="interest_expense", enforce_sign="nonpositive"),
    ],
)


# ============ NET INTEREST EXPENSE BUILDER ============

def compute_net_interest_expense(row: Dict[str, Any]) -> Optional[float]:
    """
    Net interest expense = interest expense - interest income.

    Precedence (short-circuiting to avoid double counting):

      1) If 'net_interest_expense' is present, use it directly.
         This is the most aggregate field representing net interest.

      2) If 'interest_and_other_net' is present, use it.
         Some companies report a combined interest/other line.

      3) If 'interest_expense' is present:
         - Subtract 'investment_income_interest' if available to get net.
         - Otherwise, use interest_expense as the net (assumes no material income).

      4) Build from operating + non-operating components:
         - Prefer net versions (net_interest_expense_operating, net_interest_expense_nonoperating)
         - Fall back to gross versions minus income

      5) Build from granular components (finance lease, LT debt interest, etc.)
         but only if no aggregate exists.

    Sign convention: positive = net expense (cash outflow), negative = net income.
    """
    # ---------- 1) Direct net_interest_expense aggregate ----------
    net_int = _to_num(row.get("net_interest_expense"))
    if net_int is not None:
        return net_int

    # ---------- 2) Combined interest_and_other_net ----------
    int_and_other = _to_num(row.get("interest_and_other_net"))
    if int_and_other is not None:
        return int_and_other

    # ---------- 3) interest_expense (possibly netted with income) ----------
    int_expense = _to_num(row.get("interest_expense"))
    if int_expense is not None:
        int_income = _to_num(row.get("investment_income_interest"))
        if int_income is not None:
            return int_expense - int_income
        return int_expense

    # ---------- 4) Operating + non-operating split ----------
    # Prefer net versions if available
    net_op = _to_num(row.get("net_interest_expense_operating"))
    net_nonop = _to_num(row.get("net_interest_expense_nonoperating"))

    if net_op is not None or net_nonop is not None:
        # Use net versions, sum them
        return _sum_non_none([net_op, net_nonop])

    # Try gross operating + gross non-operating
    gross_op = _to_num(row.get("interest_expense_operating"))
    gross_nonop = _to_num(row.get("interest_expense_nonoperating"))

    if gross_op is not None or gross_nonop is not None:
        total_gross = _sum_non_none([gross_op, gross_nonop])
        if total_gross is not None:
            # Subtract interest income if available
            int_income = _to_num(row.get("investment_income_interest"))
            if int_income is not None:
                return total_gross - int_income
            return total_gross

    # ---------- 5) Granular components (last resort) ----------
    # These are specific line items that should not overlap
    granular_components = [
        "interest_paid_on_long_term_debt",
        "finance_lease_interest_expense",
        "interest_paid_financing_obligations",
    ]
    granular_sum = _sum_present(row, granular_components)

    if granular_sum is not None:
        # Subtract interest income if available
        int_income = _to_num(row.get("investment_income_interest"))
        if int_income is not None:
            return granular_sum - int_income
        return granular_sum

    return None


_add_rule(
    "is_net_interest_expense",
    "Net interest expense (interest expense minus interest income). "
    "Prefers aggregate net_interest_expense, falls back to constructing from components.",
    [
        FormulaRule(func=compute_net_interest_expense),
        FormulaRule(expr="direct"),
    ],
)


def compute_restructuring_and_other_opex(row: Dict[str, Any]) -> Optional[float]:
    """
    Compute restructuring_and_other_opex from various alternative fields.
    All fields are alternatives (first non-null wins).
    """
    # Priority list of alternative fields (first non-null wins)
    alternatives = [
        "restructuring_settlement_and_impairment_provisions",
        "restructuring_and_other",
        "restructuring_costs_and_other",
        "restructuring_costs_and_asset_impairment_charges",
        "impairment_integration_and_restructuring_expenses",
        "impairment_and_restructuring_expenses",
        "restructuringand_other_special_chargesnet",
        "restructuringand_other_special_charges",
        "other_cost_and_expense_operating",
        "other_nonoperating_expense",
    ]
    
    for field in alternatives:
        val = _to_num(row.get(field))
        if val is not None:
            return val
    
    return None


_add_rule(
    "is_restructuring_and_other_opex",
    "Restructuring, impairment, and other operating expenses. "
    "First non-null from various alternative fields.",
    [
        FormulaRule(func=compute_restructuring_and_other_opex),
        FormulaRule(expr="direct"),
    ],
)


# Income before equity method
_add_rule(
    "is_income_loss_before_equity_method_investments",
    "Income before equity-method investments.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(
            expr="is_net_income_loss + is_income_loss_from_equity_method_investments",
            require_all=False,
        ),
    ],
)

# Equity-method income
_add_rule(
    "is_income_loss_from_equity_method_investments",
    "Equity method income / loss.",
    [
        FormulaRule(expr="direct"),
    ],
)

# Net income available to common (basic)
_add_rule(
    "is_net_income_loss_available_to_common_stockholders_basic",
    "Net income available to common (basic).",
    [
        FormulaRule(expr="direct"),
        FormulaRule(
            expr="is_net_income_loss - preferred_stock_dividends_and_other_adjustments",
            require_all=False,
        ),
    ],
)

# EPS (basic / diluted)
_add_rule(
    "is_basic_earnings_per_share",
    "Basic EPS.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(
            expr="is_net_income_loss_available_to_common_stockholders_basic / is_basic_average_shares",
            require_all=True,
        ),
    ],
)

_add_rule(
    "is_diluted_earnings_per_share",
    "Diluted EPS.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(
            expr="is_net_income_loss_available_to_common_stockholders_basic / is_diluted_average_shares",
            require_all=True,
        ),
    ],
)


# ============ DEPRECIATION & AMORTIZATION BUILDER ============

# ============ DEPRECIATION / AMORTISATION BUILDERS ============

def compute_depreciation_only(row: Dict[str, Any]) -> Optional[float]:
    """
    Depreciation component (excluding amortisation where possible).
    Very simple: prefer a single 'depreciation' field.
    """
    dep = _first_non_null(row, ["depreciation"])
    return dep


def compute_amortization_only(row: Dict[str, Any]) -> Optional[float]:
    """
    Amortisation component (intangibles, ROU assets, acquisition-related,
    inventory step-up), avoiding double counting when multiple levels exist.
    """

    # ---- 1) Main intangible/ROU/goodwill amortisation family ----
    # Take the most aggregate field, if any, and ignore other members of this family.
    main_family = [
        "amortization_of_intangible_assets_and_right_of_use_assets_and_goodwill_and_other_intangibles",
        "amortization_of_intangible_assets_and_right_of_use_assets_and_goodwill",
        "amortization_of_intangible_assets_and_right_of_use_assets_and_other_intangibles",
        "amortization_of_intangible_assets_and_right_of_use_assets",
        "amortization_of_intangible_and_right_of_use_assets",
        "amortization_of_intangible_assets",
    ]
    main_amort = _first_non_null(row, main_family)

    # ---- 2) Acquisition-related intangible amortisation family ----
    acq_agg_candidates = [
        "amortization_of_acquired_intangible_assets",
        "amortization_of_acquisition_related_intangibles_and_costs",
        "amortizationof_acquisition_related_intangible_assets",
    ]
    acq_amort = _first_non_null(row, acq_agg_candidates)

    # If no aggregate, sum detailed acquisition-related components
    if acq_amort is None:
        acq_detail = [
            "amortization_of_acquisition_related_intangibles_cogs",
            "amortization_of_acquisition_related_intangibles_opex",
            "amortization_of_acquisition_costs",
            "amortizationofacquisitionrelatedintangibleassetsoperatingexpenses",
            "amortizationofacquisitionrelatedintangibleassetscostofproductssold",
        ]
        acq_amort = _sum_present(row, acq_detail)

    # ---- 3) Inventory valuation step-up amortisation ----
    inv_step_amort = _first_non_null(
        row,
        [
            "amortization_of_inventory_valuation_step_up",
            "amortizationofinventoryvaluationstepup",
        ],
    )

    # ---- 4) Catch-all "other amortisation" if present ----
    other_da = _to_num(row.get("other_depreciation_and_amortization"))
    # You might choose to exclude this if you suspect double-counting; keep as separate knob.

    return _sum_non_none([main_amort, acq_amort, inv_step_amort, other_da])

def compute_depreciation_and_amortization(row: Dict[str, Any]) -> Optional[float]:
    """
    Comprehensive depreciation & amortization metric.

    Precedence:
      1) Use any aggregate D&A / DDA field if available.
      2) Else: depreciation_only + amortization_only (with family-level precedence).
    """

    # 1) Aggregate D&A / DDA style candidates (first non-null wins)
    aggregate_candidates = [
        "depreciation_and_amortization",
        "depreciation_depletion_and_amortization",
        "cost_depreciation_amortization_and_depletion",
        "depreciation_amortization_and_impairment",
        "depreciation_amortization_and_other_noncash_items",
        "depreciation_amortization_and_accretion_net",
        "depreciation_amortization_and_other",
    ]
    agg_val = _first_non_null(row, aggregate_candidates)
    if agg_val is not None:
        return agg_val

    # 2) Build from depreciation + amortisation sub-metrics
    dep = compute_depreciation_only(row)
    amort = compute_amortization_only(row)

    return _sum_non_none([dep, amort])

def compute_net_property_plant_and_equipment(row: Dict[str, Any]) -> Optional[float]:
    """
    Net property, plant and equipment (PPE), avoiding double counting when
    both aggregate and component fields are present.

    Precedence (short-circuiting):

      1) property_plant_and_equipment_and_finance_lease_right_of_use_asset_after_accumulated_depreciation_and_amortization
         -> treat as the most aggregated total (PPE + finance-lease ROU assets).

      2) net_property_plant_and_equipment
         -> fallback: direct net PPE total (assumed to include all PPE classes).

      3) property_plant_and_equipment_gross
         - accumulated_depreciation_depletion_and_amortization_property_plant_and_equipment
         -> derive net PPE from gross and accumulated.

      4) If none of the above exist, fall back to any standalone net-like
         sub-buckets (capitalized computer software, other PPE), summed
         together. This is a best-effort partial PPE, but still avoids
         double counting.
    """

    # ---------- 1) Combined PPE + finance-lease ROU assets (most aggregate) ----------
    combined_ppe_rou = _to_num(
        row.get(
            "property_plant_and_equipment_and_finance_lease_right_of_use_asset_after_accumulated_depreciation_and_amortization"
        )
    )
    if combined_ppe_rou is not None:
        return combined_ppe_rou

    # ---------- 2) Direct net PPE aggregate ----------
    direct_net = _to_num(row.get("net_property_plant_and_equipment"))
    if direct_net is not None:
        return direct_net

    # ---------- 3) Derive from gross – accumulated ----------
    gross = _to_num(row.get("property_plant_and_equipment_gross"))
    accum = _to_num(
        row.get(
            "accumulated_depreciation_depletion_and_amortization_property_plant_and_equipment"
        )
    )

    core_net: Optional[float] = None
    if (gross is not None) and (accum is not None):
        core_net = gross - accum

    if core_net is not None:
        # IMPORTANT: do NOT add subcomponents (software, "other PPE") on top of
        # this, because in most filers those are already included in the gross
        # PPE balance that we just netted out via accumulated depreciation.
        return core_net

    # ---------- 4) Fallback: sum standalone net-ish sub-buckets ----------
    # Use net capitalized software if available; otherwise (as a last resort)
    # fall back to its gross value. This is imperfect but better than returning None.
    software_net = _to_num(row.get("capitalized_computer_software_net"))
    if software_net is None:
        software_net = _to_num(row.get("capitalized_computer_software_gross"))

    # "Other PPE" – often a separate line; treat it as net-only fallback.
    other_ppe = _to_num(row.get("property_plant_and_equipment_other"))

    fallback = _sum_non_none([software_net, other_ppe])
    return fallback


_add_rule(
    "is_depreciation_and_amortization",
    "Total depreciation and amortization; prefer aggregate fields, otherwise sum granular components.",
    [
        # Primary: custom Python logic
        FormulaRule(func=compute_depreciation_and_amortization),
        # Fallback: use the direct mapped field if present
        FormulaRule(expr="direct"),
    ],
)


# ============ BALANCE SHEET BUILDERS ============

# Total assets
_add_rule(
    "bs_assets",
    "Total assets.",
    [
        FormulaRule(expr="direct"),
    ],
)

# Current assets
_add_rule(
    "bs_current_assets",
    "Total current assets.",
    [
        FormulaRule(expr="direct"),
    ],
)

# Current liabilities
_add_rule(
    "bs_current_liabilities",
    "Total current liabilities.",
    [
        FormulaRule(expr="direct"),
    ],
)


# Noncurrent liabilities
_add_rule(
    "bs_noncurrent_liabilities",
    "Total noncurrent liabilities.",
    [
        FormulaRule(expr="direct"),
    ],
)

_add_rule(
    "bs_total_liabilities",
    "Total liabilities with fallbacks: current + noncurrent, then assets minus equity.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(expr="bs_current_liabilities + bs_noncurrent_liabilities", require_all=False),
        FormulaRule(expr="bs_assets - bs_equity", require_all=True),
        FormulaRule(expr="bs_liabilities_and_equity - bs_equity", require_all=True),
    ],
)


# Equity (total) and equity attributable to parent
_add_rule(
    "bs_equity",
    "Total equity (may include non-controlling).",
    [
        FormulaRule(expr="direct"),  # from StockholdersEquity
    ],
)

_add_rule(
    "bs_equity_attributable_to_parent",
    "Equity attributable to parent/shareholders.",
    [
        FormulaRule(expr="direct"),
    ],
)


# Noncontrolling interest equity
def compute_noncontrolling_interest_equity(row: Dict[str, Any]) -> Optional[float]:
    """
    Noncontrolling interest (minority interest) in equity, avoiding double counting.

    Precedence (short-circuiting):
      1) If 'equity_attributable_to_noncontrolling_interest' is present, use it directly.
         This is the most direct representation of NCI.
      2) If both 'stockholders_equity_including_portion_attributable_to_noncontrolling_interest'
         (total equity including NCI) and 'equity' (parent equity) are present,
         compute NCI as: total_with_nci - parent_equity.
      3) Fallback to 'minority_interest' if present (older terminology).
      4) Return None if no data available.
    """
    # 1) Direct NCI field (preferred)
    nci_direct = _to_num(row.get("equity_attributable_to_noncontrolling_interest"))
    if nci_direct is not None:
        return nci_direct

    # 2) Derive from total equity (including NCI) minus parent equity
    total_with_nci = _to_num(row.get("stockholders_equity_including_portion_attributable_to_noncontrolling_interest"))
    parent_equity = _to_num(row.get("equity"))
    if total_with_nci is not None and parent_equity is not None:
        return total_with_nci - parent_equity

    # 3) Fallback to minority_interest (older terminology)
    minority = _to_num(row.get("minority_interest"))
    if minority is not None:
        return minority

    return None


_add_rule(
    "bs_noncontrolling_interest_equity",
    "Noncontrolling interest (minority interest) in equity. Prefers direct NCI field, "
    "falls back to deriving from total equity minus parent equity.",
    [
        FormulaRule(func=compute_noncontrolling_interest_equity),
        FormulaRule(expr="direct"),
    ],
)


# Long-term debt (core bucket)
_add_rule(
    "bs_long_term_debt",
    "Core long-term debt (ex leases where possible).",
    [
        FormulaRule(expr="direct"),
    ],
)

# PPE
_add_rule(
    "bs_net_property_plant_and_equipment",
    "Net property, plant and equipment (including finance-lease ROU assets "
    "when only a combined total is reported), with precedence on aggregate "
    "fields and no double counting of components.",
    [
        FormulaRule(func=compute_net_property_plant_and_equipment),
        # Fallback to any directly mapped bs_net_property_plant_and_equipment field
        FormulaRule(expr="direct"),
    ],
)

# ============ CASH FLOW BUILDERS ============

# Net cash flow from investing (used for capex TTM)
_add_rule(
    "cf_net_cash_flow_from_investing_activities",
    "Net cash flow from investing activities.",
    [
        FormulaRule(expr="direct"),
    ],
)

# ============ INTEREST BEARING DEBT BUILDERS ============

def _sum_present(row: Dict[str, Any], names: List[str]) -> Optional[float]:
    """Sum only non-NaN components; return None if all missing."""
    vals = []
    for n in names:
        v = _to_num(row.get(n))
        if v is not None:
            vals.append(v)
    return float(sum(vals)) if vals else None


def compute_interest_bearing_debt(row: Dict[str, Any]) -> Optional[float]:
    """
    Debt-only interest-bearing obligations.

    Definition:
      - Includes bond/loan-style debt (short- and long-term).
      - Excludes all lease liabilities (finance + operating).
      - Avoids double-counting current vs noncurrent when a total exists.

    Expected canonical fields (debt side):
      - current_debt
      - short_term_borrowings
      - other_short_term_borrowings
      - convertible_debt_current
      - convertible_notes_payable_current
      - long_term_debt              (may already include current portion)
      - long_term_debt_noncurrent
      - long_term_debt_current
      - long_term_notes_and_loans
      - secured_long_term_debt
      - long_term_loans_from_bank
      - convertible_long_term_notes_payable
      - convertible_debt_noncurrent
    """
    lt_total = _to_num(row.get("long_term_debt"))
    if lt_total is not None:
        return lt_total
    # ---------- 1) SHORT-TERM DEBT ONLY (NO LEASES) ----------
    short_term_components = [
        "current_debt",
        "short_term_borrowings",
        "other_short_term_borrowings",
        "convertible_debt_current",
        "convertible_notes_payable_current",
        # If you have a dedicated current-notes-payable field, include it here:
        "notes_payable_current",
    ]
    short_term_debt = _sum_present(row, short_term_components)

    # ---------- 2) LONG-TERM DEBT (NO LEASES) ----------

    # Prefer a single aggregate long_term_debt field if present
    # (For MSFT this equals noncurrent LT debt + current portion. )

    if lt_total is None:
        # If no aggregate, build from components:
        # - explicitly sum current + noncurrent LT debt
        # - add other bond/loan LT components
        lt_basic = _sum_present(
            row,
            [
            "long_term_debt_noncurrent",
            "long_term_debt_current",
            ],
        )

        lt_other = _sum_present(
            row,
            [
            "long_term_notes_and_loans",
            "secured_long_term_debt",
            "long_term_loans_from_bank",
            "convertible_long_term_notes_payable",
            "convertible_debt_noncurrent",
            ],
        )

        lt_total = _sum_non_none([lt_basic, lt_other])

    # IMPORTANT: we deliberately DO NOT use any of:
    #   long_term_debt_and_capital_leases_incl_current
    #   long_term_debt_and_capital_leases
    #   long_term_debt_and_finance_lease_obligations_noncurrent
    # because they embed lease liabilities by design.

    # ---------- 3) COMBINE SHORT- AND LONG-TERM ----------
    if short_term_debt is None and lt_total is None:
        return None
    return (short_term_debt or 0.0) + (lt_total or 0.0)


_add_rule(
    "bs_interest_bearing_debt",
    "Comprehensive interest-bearing debt metric.",
    [
        FormulaRule(
            func=compute_interest_bearing_debt,
        ),
    ],
)

def compute_finance_lease_liabilities(row: Dict[str, Any]) -> Optional[float]:
    """
    Total finance lease liabilities (current + noncurrent).

    Precedence:
      1) If a total 'finance_lease_liability' field is present (as in MSFT),
         return it directly.
      2) Otherwise, compute as current + noncurrent pieces using whatever
         canonical names are available.
    """

    # 1) Direct total, if present (e.g. MSFT's total finance lease liabilities).
    total = _to_num(row.get("finance_lease_liability"))
    if total is not None:
        return total

    # 2) Build from components (current + noncurrent).
    # List all canonical variants you might have.
    noncurrent = _sum_present(
        row,
        [
            "finance_lease_liability_noncurrent",
            "finance_lease_liabilities_noncurrent",
            "lease_liability_noncurrent",      # if this is mapped as finance-only in your system
        ],
    )

    current = _sum_present(
        row,
        [
            "finance_lease_liability_current",
            "finance_lease_liabilities_current",
            "lease_liability_current",         # as above, if mapped that way
        ],
    )

    return _sum_non_none([current, noncurrent])

_add_rule(
    "bs_finance_lease_liability",
    "Total finance lease liabilities (current + noncurrent).",
    [
        FormulaRule(func=compute_finance_lease_liabilities),
    ],
)

def compute_operating_lease_liabilities(row: Dict[str, Any]) -> Optional[float]:
    """
    Total operating lease liabilities (current + noncurrent).

    Precedence:
      1) If a total 'operating_lease_liability' field is present, return it.
      2) Otherwise, compute as current + noncurrent components.
    """

    # 1) Direct total, if present (e.g. MSFT's total operating lease liabilities).
    total = _to_num(row.get("operating_lease_liability"))
    if total is not None:
        return total

    # 2) Build from components.
    noncurrent = _sum_present(
        row,
        [
            "operating_lease_liability_noncurrent",
        ],
    )

    current = _sum_present(
        row,
        [
            "operating_lease_liability_current",
        ],
    )

    return _sum_non_none([current, noncurrent])

_add_rule(
    "bs_operating_lease_liability",
    "Total operating lease liabilities (current + noncurrent).",
    [
        FormulaRule(func=compute_operating_lease_liabilities),
        FormulaRule(expr="direct"),
    ],
)


# ============ MARKETABLE SECURITIES BUILDER ============

def compute_marketable_securities(row: Dict[str, Any]) -> Optional[float]:
    """
    Total marketable securities (current + noncurrent) without double counting.

    Precedence (short-circuiting):
      1) If 'marketable_securities' (a total) is present, return it.
      2) Else, if BOTH 'marketable_securities_current' and
         'marketable_securities_noncurrent' are present, sum them.
      3) Else, if 'debt_securities_available_for_sale_excl_accrued_interest'
         (a total AFS debt balance) is present, return it.
      4) Else, if 'available_for_sale_debt_securities' (total AFS debt) is present,
         return it.
      5) Else, build a total from current/noncurrent AFS securities and AFS
         debt securities, choosing ONE representative series for current and ONE
         for noncurrent to avoid double counting.
    """
    # ---------- 1) Direct marketable_securities total ----------
    total_ms = _to_num(row.get("marketable_securities"))
    if total_ms is not None:
        return total_ms

    # ---------- 2) Sum marketable_securities_current + noncurrent if both present ----------
    ms_curr = _to_num(row.get("marketable_securities_current"))
    ms_noncurr = _to_num(row.get("marketable_securities_noncurrent"))
    if (ms_curr is not None) and (ms_noncurr is not None):
        return ms_curr + ms_noncurr

    # ---------- 3) Total AFS debt: debt_securities_available_for_sale_excl_accrued_interest ----------
    # This is designed as a total for AFS debt securities (no accrued interest).
    afs_debt_excl = _to_num(row.get("debt_securities_available_for_sale_excl_accrued_interest"))
    if afs_debt_excl is not None:
        return afs_debt_excl

    # ---------- 4) Total AFS debt: available_for_sale_debt_securities ----------
    afs_debt_total = _to_num(row.get("available_for_sale_debt_securities"))
    if afs_debt_total is not None:
        return afs_debt_total

    # ---------- 5) Build current + noncurrent buckets without double counting ----------
    # Current bucket: prefer generic AFS securities if present, otherwise AFS debt.
    # Only take ONE of these so we don't double count.
    curr_candidates = [
        "available_for_sale_securities_current",      # broader: overall AFS securities (debt + possibly equity)
        "available_for_sale_debt_securities_current", # narrower: AFS debt only
    ]
    curr_val = None
    for name in curr_candidates:
        v = _to_num(row.get(name))
        if v is not None:
            curr_val = v
            break

    # Noncurrent bucket:
    # - Prefer generic AFS securities (overall)
    # - If not present, use AFS debt noncurrent
    # - If that's missing, fall back to any equity-specific AFS noncurrent bucket
    noncurr_candidates = [
        "available_for_sale_securities_noncurrent",
        "available_for_sale_debt_securities_noncurrent",
        "available_for_sale_securities_equity_securities_noncurrent",
    ]
    noncurr_val = None
    for name in noncurr_candidates:
        v = _to_num(row.get(name))
        if v is not None:
            noncurr_val = v
            break

    # If we *still* have nothing, but marketable_securities_current exists alone,
    # use that as the best available proxy.
    if curr_val is None and noncurr_val is None:
        if ms_curr is not None:
            return ms_curr
        if ms_noncurr is not None:
            return ms_noncurr
        # Nothing to go on
        return None

    return (curr_val or 0.0) + (noncurr_val or 0.0)


_add_rule(
    "bs_marketable_securities",
    "Total marketable securities (current + noncurrent) without double counting.",
    [
        FormulaRule(func=compute_marketable_securities),
        FormulaRule(expr="direct"),
    ],
)


# ============ SHORT-TERM INVESTMENTS BUILDER ============

def compute_short_term_investments(row: Dict[str, Any]) -> Optional[float]:
    """
    Short-term investments metric with the following precedence:

      1) Direct short_term_investments (us-gaap:ShortTermInvestments).
      2) Cash + short-term investments total minus cash-only total.
      3) Cash + cash equivalents + restricted cash + short-term investments
         minus cash + cash equivalents + restricted cash (+ restricted cash equivalents).
      4) Direct 'marketable securities / short-term investments' current tags.
      5) Sum of component buckets (AFS, HTM, trading, other current marketable
         securities) with internal de-duplication where possible.
    """
    # ---------- 1) DIRECT SHORT-TERM INVESTMENTS ----------
    direct_st = _to_num(row.get("short_term_investments"))
    if direct_st is not None:
        return direct_st

    # ---------- 2) CASH + STI TOTAL MINUS CASH ----------
    total_cash_st = _first_non_null(
        row,
        [
            "cash_cash_equivalents_and_short_term_investments",
            # include possible alias if your canonicalization differs
            "cash_and_cash_equivalents_and_short_term_investments",
        ],
    )
    if total_cash_st is not None:
        cash_only = _first_non_null(
            row,
            [
                "cash_and_cash_equivalents",
                "cash_and_cash_equivalents_at_carrying_value",
            ],
        )
        if cash_only is not None:
            diff = total_cash_st - cash_only
            # basic sanity: STI should not be negative
            if diff >= 0:
                return diff

    # ---------- 3) CASH + RESTRICTED CASH + STI MINUS CASH + RESTRICTED CASH ----------
    total_cash_restr_st = _first_non_null(
        row,
        [
            "cash_cash_equivalents_restricted_cash_and_short_term_investments",
            # in case your canonicalizer produces a slightly different form
            "cash_cash_equivalents_restricted_cash_restricted_cash_equivalents_and_short_term_investments",
        ],
    )
    if total_cash_restr_st is not None:
        cash_restr_only = _first_non_null(
            row,
            [
                "cash_cash_equivalents_restricted_cash_and_restricted_cash_equivalents",
                "cash_and_cash_equivalents_restricted_cash_and_restricted_cash_equivalents",
            ],
        )
        if cash_restr_only is not None:
            diff = total_cash_restr_st - cash_restr_only
            if diff >= 0:
                return diff

    # ---------- 4) DIRECT MARKETABLE/SHORT-TERM INVESTMENT TAGS ----------
    # Treat these as total short-term investments if the more canonical measures are absent.
    direct_marketables = _first_non_null(
        row,
        [
            "marketable_securities_current",                        # us-gaap:MarketableSecuritiesCurrent
            "available_for_sale_securities_short_term_investments_amortized_cost",
            # deprecated but still seen: us-gaap:AvailableForSaleSecuritiesShortTermInvestmentsAmortizedCost 
        ],
    )
    if direct_marketables is not None:
        return direct_marketables

    # ---------- 5) BUILD FROM COMPONENT BUCKETS (NO DOUBLE COUNTING WITHIN BUCKETS) ----------
    # 5a) Available-for-sale (AFS) short-term bucket
    # Prefer the rollup if present; otherwise sum the components.
    afs_rollup = _to_num(row.get("available_for_sale_securities_current"))
    if afs_rollup is not None:
        afs_current = afs_rollup
    else:
        afs_current = _sum_present(
            row,
            [
                "available_for_sale_securities_debt_securities_current",
                "available_for_sale_securities_equity_securities_current",
                "available_for_sale_securities_short_term_investments_amortized_cost",
                "available_for_sale_debt_securities_current",
                "debt_securities_available_for_sale_excl_accrued_interest",
            ],
        )

    # 5b) Held-to-maturity (HTM) short-term bucket
    htm_current = _sum_present(
        row,
        [
            "held_to_maturity_securities_current",
        ]
    )

    # 5c) Trading securities current bucket
    trading_current = _sum_present(
        row,
        [
            "trading_securities_current",
            # if you use a more generic current trading tag, add it here
        ]
    )

    # 5d) Other current marketable securities buckets (if not captured above)
    other_current = _sum_present(
        row,
        [
            "marketable_securities_fixed_maturities_current",
            # add other explicit "current" marketable security tags here if you see them
        ]
    )

    buckets = [
        v for v in [afs_current, htm_current, trading_current, other_current] if v is not None
    ]
    if not buckets:
        return None

    return float(sum(buckets))


_add_rule(
    "bs_short_term_investments",
    "Short-term investments with precedence: direct field, cash+STI minus cash, component buckets.",
    [
        FormulaRule(func=compute_short_term_investments),
        FormulaRule(expr="direct"),
    ],
)


# ============ SG&A BUILDER ============

def compute_sg_and_a(row: Dict[str, Any]) -> Optional[float]:
    """
    Selling, general and administrative expenses (SG&A).

    Components used (your canonical names):
      - selling_general_and_administrative_expenses  (aggregate if present)
      - general_and_administrative
      - selling_and_marketing_expense
      - marketing_and_advertising
      - other_selling_general_and_administrative_expense

    Precedence:
      1) If 'selling_general_and_administrative_expenses' is present, return it.
      2) Else, if BOTH 'general_and_administrative' and 'selling_and_marketing_expense'
         exist, use their sum (core SG&A).
      3) Else, sum whatever subset of the granular SG&A components is available.
    """

    # 1) Direct SG&A aggregate (us-gaap:SellingGeneralAndAdministrativeExpense)
    direct = _to_num(row.get("selling_general_and_administrative_expenses"))
    if direct is not None:
        return direct

    # 2) Core split: G&A + Selling & Marketing
    ga = _to_num(row.get("general_and_administrative"))
    sm = _to_num(row.get("selling_and_marketing_expense"))
    if ga is not None and sm is not None:
        return ga + sm

    # 3) Build from any available granular components
    #    (no aggregate present; sum subset of components).
    components: List[str] = [
        "general_and_administrative",
        "selling_and_marketing_expense",
        "marketing_and_advertising",
        "other_selling_general_and_administrative_expense",
    ]
    sga = _sum_present(row, components)
    return sga


_add_rule(
    "is_selling_general_and_administrative_expenses",
    "Selling, general and administrative expenses (SG&A).",
    [
        FormulaRule(func=compute_sg_and_a),
        FormulaRule(expr="direct"),
    ],
)


# ============ CAPEX BUILDER ============

def compute_capital_expenditures(row: Dict[str, Any]) -> Optional[float]:
    """
    Capital expenditures (Capex).

    Components (your canonical names):
      - capital_expenditures (aggregate if present)
      - payments_for_financed_property_plant_and_equipment_and_intangible_assets_financing_activities
      - payments_for_financed_property_plant_and_equipment_financing_activities
      - payments_for_proceeds_from_productive_assets
      - payments_for_solar_energy_systems
      - payments_for_solar_energy_systems_leased_and_to_be_leased
      - payments_for_solar_energy_systems_net_of_sales
      - payments_on_equipment_purchase_contracts
      - payments_to_acquire_assets_investing_activities
      - payments_to_acquire_property_plant_and_equipment
      - payments_to_acquire_other_productive_assets
      - payments_for_deposits_on_real_estate_acquisitions
      - proceedsfrom_deposit_paymentsfor_withdrawalfor_purchaseof_landand_building
      - payments_to_acquire_intangible_assets
      - payments_to_acquire_other_indefinite_lived_intangible_assets
      - payments_to_develop_software
      - payments_for_software
      - payments_to_acquire_indefinite_lived_intangible_crypto_assets
      - paymentsforrecordedunconditionalpurchaseobligation
      - share_based_payment_arrangement_amount_capitalized_in_software_development_costs

    Precedence:
      1) If 'capital_expenditures' is present, return it (assumed to be total capex).
      2) Else, sum all available capex-related components.
    """

    # 1) Direct capex aggregate (if your vendor gives a total)
    direct = _to_num(row.get("capital_expenditures"))
    if direct is not None:
        return direct

    # 2) Sum all component cash outflows that represent capex,
    #    using your component list.
    component_fields: List[str] = [
        "payments_for_financed_property_plant_and_equipment_and_intangible_assets_financing_activities",
        "payments_for_financed_property_plant_and_equipment_financing_activities",
        "payments_for_proceeds_from_productive_assets",
        "payments_for_solar_energy_systems",
        "payments_for_solar_energy_systems_leased_and_to_be_leased",
        "payments_for_solar_energy_systems_net_of_sales",
        "payments_on_equipment_purchase_contracts",
        "payments_to_acquire_assets_investing_activities",
        "payments_to_acquire_property_plant_and_equipment",
        "payments_to_acquire_other_productive_assets",
        "payments_for_deposits_on_real_estate_acquisitions",
        "proceedsfrom_deposit_paymentsfor_withdrawalfor_purchaseof_landand_building",
        "payments_to_acquire_intangible_assets",
        "payments_to_acquire_other_indefinite_lived_intangible_assets",
        "payments_to_develop_software",
        "payments_for_software",
        "payments_to_acquire_indefinite_lived_intangible_crypto_assets",
        "paymentsforrecordedunconditionalpurchaseobligation",
        "share_based_payment_arrangement_amount_capitalized_in_software_development_costs",
    ]

    capex = _sum_present(row, component_fields)
    return capex


_add_rule(
    "cf_capital_expenditures",
    "Capital expenditures (investing and financed capex cash flows).",
    [
        FormulaRule(func=compute_capital_expenditures),
        FormulaRule(expr="direct"),
    ],
)


def compute_cash_and_cash_equivalents(row: Dict[str, Any]) -> Optional[float]:
    """
    Cash and cash equivalents INCLUDING restricted cash, avoiding double counting.

    Precedence (short-circuiting):
      1) If a total "cash + cash equivalents + restricted cash" tag exists,
         return it directly.
      2) Else, take the best available total "cash + cash equivalents"
         (no restricted cash) and add a restricted-cash-only bucket.
      3) Else, if no total exists, build "cash + cash equivalents" from
         the separate cash and cash-equivalents fields, then add restricted
         cash if present.
    """
    # ---------- 1) Direct total including restricted cash ----------
    total_with_restricted = _first_non_null(
        row,
        [
            # Common "cash + CE + restricted cash (+ restricted CE)" style tags
            "cash_and_cash_equivalents_restricted_cash_and_restricted_cash_equivalents",
            "cash_cash_equivalents_restricted_cash_and_restricted_cash_equivalents",
            "cash_and_cash_equivalents_and_restricted_cash",
            "cash_and_cash_equivalents_and_restricted",
            "cash_cash_equivalents_and_restricted_cash",
        ],
    )
    if total_with_restricted is not None:
        # Already includes restricted cash; nothing else to add.
        return total_with_restricted

    # ---------- 2) Base cash + cash equivalents (excluding restricted) ----------
    base_agg = _first_non_null(
        row,
        [
            # Canonical aggregate "cash + cash equivalents"
            "cash_and_cash_equivalents",
            "cash_and_cash_equivalents_at_carrying_value",
            # Some vendors give "excluding time deposits" as a total; use as fallback.
            "cash_and_cash_equivalents_excluding_time_deposits",
        ],
    )

    if base_agg is not None:
        base_cce = base_agg
    else:
        # No aggregate: build from components
        base_cce = _sum_present(
            row,
            [
                "cash",
                "cash_equivalents_at_carrying_value",
            ],
        )

    # ---------- 3) Restricted-cash-only bucket ----------
    # Prefer a single aggregate if present, otherwise sum current + noncurrent.
    restricted_total = _first_non_null(
        row,
        [
            "restricted_cash_and_cash_equivalents",
            "restricted_cash_and_cash_equivalents_current",
            "restricted_cash_and_cash_equivalents_noncurrent",
        ],
    )

    if restricted_total is None:
        restricted_total = _sum_present(
            row,
            [
                "restricted_cash",
                "restricted_cash_current",
                "restricted_cash_noncurrent",
            ],
        )

    # ---------- 4) Combine ----------
    if base_cce is None and restricted_total is None:
        return None

    return (base_cce or 0.0) + (restricted_total or 0.0)


_add_rule(
    "bs_cash_and_cash_equivalents",
    "Cash and cash equivalents including restricted cash where possible, "
    "giving precedence to aggregate tags and avoiding double counting.",
    [
        FormulaRule(func=compute_cash_and_cash_equivalents),
        # Fallback if you ever have a direct bs_cash_and_cash_equivalents column
        FormulaRule(expr="direct"),
    ],
)
