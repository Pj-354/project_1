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
    expr: str
    require_all: bool = False
    enforce_sign: Optional[str] = None


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

    Returns:
        (value, source_description)
    """
    # 1. If no explicit rule, just return the raw value
    rule = rules.get(name)
    raw_val = _to_num(row.get(name))

    if rule is None:
        return raw_val, "raw"

    # 2. Try each formula in order
    for fr in rule.formulas:
        # Special case: "direct" => use raw_val, possibly with sign enforcement
        if fr.expr == "direct":
            if raw_val is None:
                continue
            val = raw_val
        else:
            val = _eval_expr(fr.expr, row)

        # If require_all and any dependency was missing -> treat as failure
        if fr.require_all and (val is None):
            continue

        # Enforce sign convention if requested
        val = _enforce_sign(val, fr.enforce_sign) if fr.enforce_sign else val

        if val is not None:
            return val, f"rule:{name}:{fr.expr}"

    # 3. Fallback: return raw value if nothing worked
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
_add_rule(
    "is_operating_expenses",
    "Total operating expenses; if missing, derive as gross profit - operating income.",
    [
        FormulaRule(expr="direct"),
        FormulaRule(expr="is_gross_profit - is_operating_income_loss", require_all=False),
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

# Long-term debt (core bucket)
_add_rule(
    "bs_long_term_debt",
    "Core long-term debt (ex leases where possible).",
    [
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

