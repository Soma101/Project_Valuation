"""
Capital Project Evaluator
=========================
A Streamlit app that builds pro forma cash flows for a capital project and
returns NPV, IRR, and payback period with an accept / reject read on each.

Run with:
    pip install streamlit pandas
    streamlit run capital_project_model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants / sane bounds
# ---------------------------------------------------------------------------

MAX_TERM = 50
MAX_MONEY = 1e12          # $1 trillion, a hard ceiling on any dollar input
MAX_UNITS = 1e12
EPS = 1e-9

DEPRECIATION_METHODS = [
    "Straight-line",
    "Double-declining balance (with straight-line switch)",
    "Sum-of-years'-digits",
    "None (CapEx not depreciated)",
]


# ---------------------------------------------------------------------------
# Validation plumbing
# ---------------------------------------------------------------------------

@dataclass
class Validation:
    """Collects blocking errors and non-blocking warnings for the input set."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def is_number(x) -> bool:
    """True only for real, finite numbers. Catches None, NaN, inf, and text."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def require_number(v: Validation, value, label: str) -> bool:
    """Guard every numeric field: reject blanks, text, NaN, and infinities."""
    if not is_number(value):
        v.error(f"{label} needs a number. Enter a value.")
        return False
    return True


def check_range(
    v: Validation,
    value,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    allow_min: bool = True,
    units: str = "",
) -> bool:
    """Range-check a field that has already passed require_number()."""
    if not is_number(value):
        return False
    x = float(value)
    if minimum is not None:
        if allow_min and x < minimum:
            v.error(f"{label} can't be below {minimum:,.4g}{units}.")
            return False
        if not allow_min and x <= minimum:
            v.error(f"{label} must be greater than {minimum:,.4g}{units}.")
            return False
    if maximum is not None and x > maximum:
        v.error(f"{label} can't be above {maximum:,.4g}{units}.")
        return False
    return True


# ---------------------------------------------------------------------------
# Finance math
# ---------------------------------------------------------------------------

def depreciation_schedule(
    capex: float, life: int, term: int, method: str
) -> List[float]:
    """Annual depreciation for years 1..term. Never exceeds CapEx in total."""
    sched = [0.0] * term
    if capex <= 0 or life <= 0 or method.startswith("None"):
        return sched

    if method.startswith("Straight-line"):
        annual = capex / life
        for t in range(min(life, term)):
            sched[t] = annual

    elif method.startswith("Double-declining"):
        book = capex
        rate = 2.0 / life
        for t in range(min(life, term)):
            years_left = life - t
            ddb = book * rate
            sl_of_remainder = book / years_left if years_left > 0 else book
            annual = max(ddb, sl_of_remainder)   # standard SL switch-over
            annual = min(annual, book)
            sched[t] = annual
            book -= annual

    elif method.startswith("Sum-of-years"):
        denom = life * (life + 1) / 2.0
        for t in range(min(life, term)):
            sched[t] = capex * (life - t) / denom

    # Numerical hygiene: never depreciate more than the asset cost.
    total = sum(sched)
    if total > capex + EPS:
        scale = capex / total
        sched = [d * scale for d in sched]
    return sched


def npv(rate: float, cash_flows: List[float]) -> float:
    """NPV where cash_flows[0] sits at t=0."""
    if rate <= -1.0:
        return float("nan")
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cash_flows))


def irr(cash_flows: List[float]) -> Tuple[Optional[float], int]:
    """
    Find IRRs by scanning for sign changes in NPV, then bisecting.
    Returns (lowest root, number of roots found). No numpy_financial needed.
    """
    if not cash_flows or all(abs(cf) < EPS for cf in cash_flows):
        return None, 0
    if min(cash_flows) >= 0 or max(cash_flows) <= 0:
        return None, 0  # all one sign: no rate of return exists

    grid = [-0.9999 + i * 0.0025 for i in range(int((20.0 + 0.9999) / 0.0025) + 1)]
    roots: List[float] = []
    prev_r = grid[0]
    prev_v = npv(prev_r, cash_flows)

    for r in grid[1:]:
        v = npv(r, cash_flows)
        if not math.isfinite(v):
            prev_r, prev_v = r, v
            continue
        if abs(v) < EPS:
            roots.append(r)
        elif math.isfinite(prev_v) and prev_v * v < 0:
            lo, hi = prev_r, r
            for _ in range(200):
                mid = (lo + hi) / 2.0
                vm = npv(mid, cash_flows)
                if not math.isfinite(vm):
                    break
                if npv(lo, cash_flows) * vm <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2.0)
        prev_r, prev_v = r, v

    # De-duplicate roots that the grid found twice.
    unique: List[float] = []
    for r in sorted(roots):
        if not unique or abs(r - unique[-1]) > 1e-4:
            unique.append(r)

    if not unique:
        return None, 0
    return unique[0], len(unique)


def payback_period(cash_flows: List[float]) -> Optional[float]:
    """
    Years until cumulative cash flow turns non-negative, interpolated within
    the crossing year. cash_flows[0] is t=0. None means never recovered.
    """
    cum = cash_flows[0]
    if cum >= 0:
        return 0.0
    for t in range(1, len(cash_flows)):
        cf = cash_flows[t]
        if cum + cf >= 0:
            if cf <= 0:
                continue
            return (t - 1) + (-cum / cf)
        cum += cf
    return None


def discounted_payback_period(cash_flows: List[float], rate: float) -> Optional[float]:
    if rate <= -1.0:
        return None
    discounted = [cf / (1.0 + rate) ** t for t, cf in enumerate(cash_flows)]
    return payback_period(discounted)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    term: int
    target_payback: float
    revenue_mode: str            # "Total revenue" or "Unit economics"
    initial_revenue: float
    volume: float
    price_per_unit: float
    cost_per_unit: float
    growth_rate: float           # decimal
    capex: float
    dep_method: str
    dep_life: int
    gross_margin: float          # decimal
    base_opex: float
    cond_threshold: float
    cond_premium: float
    premium_is_pct: bool
    nwc_pct: float               # decimal, % of revenue
    tax_rate: float              # decimal
    wacc: float                  # decimal
    loss_carryforward: bool


def build_pro_forma(i: Inputs) -> pd.DataFrame:
    """Returns a DataFrame with one column per year, 0..term."""
    years = list(range(0, i.term + 1))
    dep = depreciation_schedule(i.capex, i.dep_life, i.term, i.dep_method)

    rows = {
        k: [0.0] * len(years)
        for k in (
            "Units sold",
            "Revenue",
            "COGS",
            "Gross profit",
            "Base OpEx",
            "Conditional OpEx premium",
            "Total OpEx",
            "EBITDA",
            "Depreciation",
            "EBIT",
            "Taxes",
            "NOPAT",
            "Add back depreciation",
            "Net working capital (balance)",
            "Change in NWC",
            "CapEx",
            "Free cash flow",
            "Cumulative FCF",
            "Discounted FCF",
        )
    }

    nwc_prev = 0.0
    nol = 0.0  # net operating loss carryforward

    for t in years:
        if t == 0:
            revenue = 0.0
            units = 0.0
        else:
            g = (1.0 + i.growth_rate) ** (t - 1)
            if i.revenue_mode == "Unit economics":
                units = i.volume * g
                revenue = units * i.price_per_unit
            else:
                units = 0.0
                revenue = i.initial_revenue * g

        if t == 0:
            cogs = 0.0
        elif i.revenue_mode == "Unit economics":
            cogs = units * i.cost_per_unit
        else:
            cogs = revenue * (1.0 - i.gross_margin)

        gross_profit = revenue - cogs

        base_opex = i.base_opex if t >= 1 else 0.0
        premium = 0.0
        if t >= 1 and revenue > i.cond_threshold:
            premium = revenue * i.cond_premium if i.premium_is_pct else i.cond_premium
        total_opex = base_opex + premium

        ebitda = gross_profit - total_opex
        d = dep[t - 1] if t >= 1 else 0.0
        ebit = ebitda - d

        if i.loss_carryforward:
            if ebit > 0:
                shield = min(nol, ebit)
                nol -= shield
                taxes = (ebit - shield) * i.tax_rate
            else:
                nol += -ebit
                taxes = 0.0
        else:
            taxes = ebit * i.tax_rate if ebit > 0 else 0.0
        nopat = ebit - taxes

        # NWC scales with next year's revenue and is fully released at exit.
        if t == i.term:
            nwc_balance = 0.0
        else:
            g_next = (1.0 + i.growth_rate) ** t
            next_revenue = (
                i.volume * g_next * i.price_per_unit
                if i.revenue_mode == "Unit economics"
                else i.initial_revenue * g_next
            )
            nwc_balance = max(0.0, next_revenue * i.nwc_pct)
        delta_nwc = nwc_balance - nwc_prev
        nwc_prev = nwc_balance

        capex_flow = i.capex if t == 0 else 0.0
        fcf = nopat + d - delta_nwc - capex_flow

        vals = {
            "Units sold": units,
            "Revenue": revenue,
            "COGS": -cogs,
            "Gross profit": gross_profit,
            "Base OpEx": -base_opex,
            "Conditional OpEx premium": -premium,
            "Total OpEx": -total_opex,
            "EBITDA": ebitda,
            "Depreciation": -d,
            "EBIT": ebit,
            "Taxes": -taxes,
            "NOPAT": nopat,
            "Add back depreciation": d,
            "Net working capital (balance)": nwc_balance,
            "Change in NWC": -delta_nwc,
            "CapEx": -capex_flow,
            "Free cash flow": fcf,
        }
        for k, val in vals.items():
            rows[k][t] = 0.0 if val == 0 else val

    cum = 0.0
    for t in years:
        cum += rows["Free cash flow"][t]
        rows["Cumulative FCF"][t] = cum
        rows["Discounted FCF"][t] = (
            rows["Free cash flow"][t] / (1.0 + i.wacc) ** t
            if i.wacc > -1.0
            else float("nan")
        )

    df = pd.DataFrame(rows, index=[f"Year {t}" for t in years]).T
    if i.revenue_mode != "Unit economics":
        df = df.drop(index=["Units sold"])
    return df


# ---------------------------------------------------------------------------
# Input collection + validation
# ---------------------------------------------------------------------------

def collect_inputs() -> Tuple[Inputs, Validation]:
    v = Validation()
    sb = st.sidebar
    sb.header("Assumptions")

    sb.subheader("Horizon")
    term = sb.number_input(
        "Project term (years)", min_value=1, max_value=MAX_TERM, value=5, step=1,
        help="Whole years of operations, 1 to 50.",
    )
    target_payback = sb.number_input(
        "Target payback period (years)", min_value=0.0, max_value=float(MAX_TERM),
        value=3.0, step=0.5, format="%.2f",
        help="The hurdle the project must beat to pass the payback test.",
    )

    sb.subheader("Revenue")
    revenue_mode = sb.radio(
        "Revenue input",
        ["Total revenue", "Unit economics"],
        help="Unit economics derives revenue and COGS from volume, price, and unit cost.",
    )

    initial_revenue = volume = price_per_unit = cost_per_unit = 0.0
    gross_margin_pct = 0.0

    if revenue_mode == "Total revenue":
        initial_revenue = sb.number_input(
            "Initial revenue, Year 1 ($)", min_value=0.0, max_value=MAX_MONEY,
            value=5_000_000.0, step=100_000.0, format="%.2f",
        )
    else:
        volume = sb.number_input(
            "Sales volume, Year 1 (units)", min_value=0.0, max_value=MAX_UNITS,
            value=500_000.0, step=10_000.0, format="%.2f",
        )
        price_per_unit = sb.number_input(
            "Price per unit ($)", min_value=0.0, max_value=MAX_MONEY,
            value=10.0, step=0.25, format="%.4f",
        )
        cost_per_unit = sb.number_input(
            "Cost per unit ($)", min_value=0.0, max_value=MAX_MONEY,
            value=6.0, step=0.25, format="%.4f",
        )

    growth_pct = sb.number_input(
        "Annual sales growth rate (%)", min_value=-100.0, max_value=500.0,
        value=5.0, step=0.5, format="%.2f",
        help="Applied to volume in unit-economics mode, to revenue otherwise. "
             "Negative values model decline.",
    )

    if revenue_mode == "Total revenue":
        gross_margin_pct = sb.number_input(
            "Gross margin (%)", min_value=0.0, max_value=100.0,
            value=40.0, step=1.0, format="%.2f",
        )
    else:
        sb.caption("Gross margin is derived from price and cost per unit.")

    sb.subheader("Capital and depreciation")
    capex = sb.number_input(
        "Initial capital expenditure ($)", min_value=0.0, max_value=MAX_MONEY,
        value=3_000_000.0, step=50_000.0, format="%.2f",
        help="Spent at Year 0.",
    )
    dep_method = sb.selectbox("Depreciation method", DEPRECIATION_METHODS, index=0)
    dep_life = sb.number_input(
        "Depreciable life (years)", min_value=1, max_value=MAX_TERM,
        value=int(term), step=1,
        help="Defaults to the project term.",
    )

    sb.subheader("Operating expenses")
    base_opex = sb.number_input(
        "Base OpEx per year ($)", min_value=0.0, max_value=MAX_MONEY,
        value=800_000.0, step=10_000.0, format="%.2f",
    )
    cond_threshold = sb.number_input(
        "Conditional OpEx threshold ($ of revenue)", min_value=0.0, max_value=MAX_MONEY,
        value=6_000_000.0, step=100_000.0, format="%.2f",
        help="Above this revenue level, the premium below is added to OpEx.",
    )
    premium_unit = sb.radio(
        "Conditional OpEx premium is", ["% of revenue", "Fixed $ per year"],
        horizontal=True,
    )
    if premium_unit == "% of revenue":
        cond_premium_val = sb.number_input(
            "Conditional OpEx premium (% of revenue)", min_value=0.0, max_value=100.0,
            value=2.0, step=0.25, format="%.2f",
        )
    else:
        cond_premium_val = sb.number_input(
            "Conditional OpEx premium ($ per year)", min_value=0.0, max_value=MAX_MONEY,
            value=100_000.0, step=10_000.0, format="%.2f",
        )

    sb.subheader("Working capital, tax, discounting")
    nwc_pct = sb.number_input(
        "NWC requirement (% of next year's revenue)", min_value=0.0, max_value=100.0,
        value=10.0, step=1.0, format="%.2f",
        help="Invested ahead of the revenue it supports and released at exit.",
    )
    tax_pct = sb.number_input(
        "Corporate tax rate (%)", min_value=0.0, max_value=100.0,
        value=21.0, step=1.0, format="%.2f",
    )
    loss_carryforward = sb.checkbox(
        "Carry operating losses forward against future tax", value=True
    )
    wacc_pct = sb.number_input(
        "Discount rate / WACC (%)", min_value=0.0, max_value=200.0,
        value=10.0, step=0.25, format="%.2f",
    )

    # --- Type and finiteness guards -------------------------------------
    numeric_fields = [
        (term, "Project term"),
        (target_payback, "Target payback period"),
        (growth_pct, "Annual sales growth rate"),
        (capex, "Initial capital expenditure"),
        (dep_life, "Depreciable life"),
        (base_opex, "Base OpEx"),
        (cond_threshold, "Conditional OpEx threshold"),
        (cond_premium_val, "Conditional OpEx premium"),
        (nwc_pct, "NWC requirement"),
        (tax_pct, "Corporate tax rate"),
        (wacc_pct, "Discount rate (WACC)"),
    ]
    if revenue_mode == "Total revenue":
        numeric_fields += [
            (initial_revenue, "Initial revenue"),
            (gross_margin_pct, "Gross margin"),
        ]
    else:
        numeric_fields += [
            (volume, "Sales volume"),
            (price_per_unit, "Price per unit"),
            (cost_per_unit, "Cost per unit"),
        ]
    for value, label in numeric_fields:
        require_number(v, value, label)

    # --- Range and sign checks ------------------------------------------
    check_range(v, term, "Project term", minimum=1, maximum=MAX_TERM, units=" years")
    check_range(v, target_payback, "Target payback period", minimum=0,
                allow_min=False, maximum=float(MAX_TERM), units=" years")
    check_range(v, capex, "Initial capital expenditure", minimum=0, maximum=MAX_MONEY)
    check_range(v, base_opex, "Base OpEx", minimum=0, maximum=MAX_MONEY)
    check_range(v, cond_threshold, "Conditional OpEx threshold", minimum=0,
                maximum=MAX_MONEY)
    check_range(v, cond_premium_val, "Conditional OpEx premium", minimum=0)
    check_range(v, nwc_pct, "NWC requirement", minimum=0, maximum=100, units="%")
    check_range(v, tax_pct, "Corporate tax rate", minimum=0, maximum=100, units="%")
    check_range(v, wacc_pct, "Discount rate (WACC)", minimum=0, maximum=200, units="%")
    check_range(v, growth_pct, "Annual sales growth rate", minimum=-100, maximum=500,
                units="%")

    if revenue_mode == "Total revenue":
        check_range(v, initial_revenue, "Initial revenue", minimum=0, allow_min=False,
                    maximum=MAX_MONEY)
        check_range(v, gross_margin_pct, "Gross margin", minimum=0, maximum=100,
                    units="%")
    else:
        check_range(v, volume, "Sales volume", minimum=0, allow_min=False,
                    maximum=MAX_UNITS)
        check_range(v, price_per_unit, "Price per unit", minimum=0, allow_min=False,
                    maximum=MAX_MONEY)
        check_range(v, cost_per_unit, "Cost per unit", minimum=0, maximum=MAX_MONEY)

    # --- Cross-field checks: warnings the model can still run through ---
    if is_number(target_payback) and is_number(term) and target_payback > term:
        v.warn(
            f"Target payback of {target_payback:,.2f} years is longer than the "
            f"{int(term)}-year term. The payback test can't be met inside the model."
        )
    if is_number(dep_life) and is_number(term) and dep_life > term and capex > 0 \
            and not dep_method.startswith("None"):
        v.warn(
            f"Depreciable life ({int(dep_life)} yrs) runs past the term "
            f"({int(term)} yrs). Undepreciated book value is left on the table — "
            "the model assumes no salvage or terminal value."
        )
    if capex == 0:
        v.warn("CapEx is zero. Payback is immediate and NPV reflects operations only.")
    if revenue_mode == "Unit economics" and is_number(price_per_unit) \
            and is_number(cost_per_unit) and cost_per_unit >= price_per_unit > 0:
        v.warn(
            f"Cost per unit (${cost_per_unit:,.4g}) is at or above price per unit "
            f"(${price_per_unit:,.4g}). Every unit sold loses money."
        )
    if revenue_mode == "Total revenue" and gross_margin_pct == 0:
        v.warn("Gross margin is 0%. COGS consumes all revenue.")
    if abs(growth_pct) > 50:
        v.warn(f"A {growth_pct:,.2f}% annual growth rate is aggressive. "
               "Check it compounds to something you believe over the full term.")
    if growth_pct <= -100 + EPS:
        v.warn("Growth of -100% zeroes out revenue after Year 1.")
    if wacc_pct == 0:
        v.warn("A 0% discount rate means NPV equals undiscounted cash. "
               "The NPV and IRR tests lose their meaning.")
    elif wacc_pct > 40:
        v.warn(f"A {wacc_pct:,.2f}% WACC is very high. Confirm it isn't entered "
               "as a decimal by mistake.")
    if tax_pct > 60:
        v.warn(f"A {tax_pct:,.2f}% tax rate is above any major jurisdiction.")
    if nwc_pct > 50:
        v.warn(f"NWC at {nwc_pct:,.2f}% of revenue is unusually heavy.")

    yr1_rev = (volume * price_per_unit if revenue_mode == "Unit economics"
               else initial_revenue)
    if is_number(cond_threshold) and cond_premium_val > 0 and yr1_rev > 0:
        if cond_threshold <= yr1_rev:
            v.warn("The conditional OpEx threshold is at or below Year 1 revenue, "
                   "so the premium applies in every year.")
        else:
            g = 1.0 + growth_pct / 100.0
            peak = max(
                yr1_rev * (g ** t) for t in range(int(term if is_number(term) else 1))
            ) if is_number(term) else yr1_rev
            if cond_threshold > peak:
                v.warn("Revenue never reaches the conditional OpEx threshold, "
                       "so the premium never applies.")

    inputs = Inputs(
        term=int(term) if is_number(term) else 1,
        target_payback=float(target_payback) if is_number(target_payback) else 0.0,
        revenue_mode=revenue_mode,
        initial_revenue=float(initial_revenue),
        volume=float(volume),
        price_per_unit=float(price_per_unit),
        cost_per_unit=float(cost_per_unit),
        growth_rate=float(growth_pct) / 100.0,
        capex=float(capex),
        dep_method=dep_method,
        dep_life=int(dep_life) if is_number(dep_life) else 1,
        gross_margin=float(gross_margin_pct) / 100.0,
        base_opex=float(base_opex),
        cond_threshold=float(cond_threshold),
        cond_premium=(float(cond_premium_val) / 100.0
                      if premium_unit == "% of revenue" else float(cond_premium_val)),
        premium_is_pct=(premium_unit == "% of revenue"),
        nwc_pct=float(nwc_pct) / 100.0,
        tax_rate=float(tax_pct) / 100.0,
        wacc=float(wacc_pct) / 100.0,
        loss_carryforward=bool(loss_carryforward),
    )
    return inputs, v


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def money(x: float) -> str:
    if not is_number(x):
        return "—"
    return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"


def show_results(i: Inputs, df: pd.DataFrame) -> None:
    fcf = df.loc["Free cash flow"].tolist()
    project_npv = npv(i.wacc, fcf)
    project_irr, n_roots = irr(fcf)
    pb = payback_period(fcf)
    dpb = discounted_payback_period(fcf, i.wacc)

    st.subheader("Decision metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("NPV", money(project_npv), help=f"Discounted at {i.wacc:.2%}")
    c2.metric("IRR", f"{project_irr:.2%}" if project_irr is not None else "None",
              help="Rate where NPV equals zero")
    c3.metric("Payback", f"{pb:,.2f} yrs" if pb is not None else "Never",
              delta=f"target {i.target_payback:,.2f} yrs", delta_color="off")

    verdicts = []

    st.subheader("Recommendation by metric")

    if project_npv > 0:
        st.success(f"**NPV — accept.** Adds {money(project_npv)} of value at a "
                   f"{i.wacc:.2%} cost of capital.")
        verdicts.append(True)
    elif abs(project_npv) < 1:
        st.warning("**NPV — indifferent.** Value creation is essentially zero; "
                   "the project just clears its cost of capital.")
        verdicts.append(False)
    else:
        st.error(f"**NPV — reject.** Destroys {money(abs(project_npv))} of value at a "
                 f"{i.wacc:.2%} cost of capital.")
        verdicts.append(False)

    if project_irr is None:
        st.warning("**IRR — no decision.** Cash flows never change sign, so no "
                   "internal rate of return exists.")
        verdicts.append(False)
    else:
        if n_roots > 1:
            st.warning(f"Cash flows change sign more than once, producing {n_roots} "
                       "IRRs. Weight the NPV test more heavily here.")
        if project_irr > i.wacc:
            st.success(f"**IRR — accept.** {project_irr:.2%} return beats the "
                       f"{i.wacc:.2%} hurdle by "
                       f"{(project_irr - i.wacc) * 100:,.2f} points.")
            verdicts.append(True)
        else:
            st.error(f"**IRR — reject.** {project_irr:.2%} return falls short of the "
                     f"{i.wacc:.2%} hurdle.")
            verdicts.append(False)

    if pb is None:
        st.error(f"**Payback — reject.** The project never recovers its investment "
                 f"inside {i.term} years.")
        verdicts.append(False)
    elif pb <= i.target_payback:
        st.success(f"**Payback — accept.** Capital returns in {pb:,.2f} years, "
                   f"inside the {i.target_payback:,.2f}-year target.")
        verdicts.append(True)
    else:
        st.error(f"**Payback — reject.** Capital returns in {pb:,.2f} years, "
                 f"past the {i.target_payback:,.2f}-year target.")
        verdicts.append(False)

    st.caption(
        "Discounted payback: "
        + (f"{dpb:,.2f} years" if dpb is not None else "not achieved within the term")
    )

    passed = sum(verdicts)
    st.subheader("Overall")
    if passed == 3:
        st.success("All three tests pass. **Proceed.**")
    elif passed == 0:
        st.error("All three tests fail. **Do not proceed.**")
    else:
        st.warning(
            f"{passed} of 3 tests pass. Mixed signal — NPV is the value-maximizing "
            "criterion, so treat it as the tiebreaker and read payback as a "
            "liquidity-risk constraint rather than a value test."
        )

    st.subheader("Pro forma cash flows")
    st.dataframe(df.style.format("{:,.0f}"), use_container_width=True)
    st.caption(
        "Outflows are shown negative. FCF = NOPAT + depreciation − ΔNWC − CapEx. "
        "Working capital is funded a year ahead of the revenue it supports and "
        "released in the final year."
    )

    st.download_button(
        "Download pro forma as CSV",
        df.to_csv().encode("utf-8"),
        file_name="pro_forma_cash_flows.csv",
        mime="text/csv",
    )

    chart = pd.DataFrame(
        {
            "Free cash flow": df.loc["Free cash flow"].values,
            "Cumulative FCF": df.loc["Cumulative FCF"].values,
        },
        index=[int(c.split()[1]) for c in df.columns],
    )
    chart.index.name = "Year"
    st.subheader("Cash flow profile")
    st.bar_chart(chart[["Free cash flow"]])
    st.line_chart(chart[["Cumulative FCF"]])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Capital Project Evaluator", layout="wide")
    st.title("Capital Project Evaluator")
    st.write(
        "Set the assumptions in the sidebar. The model builds pro forma cash flows "
        "from revenue down to free cash flow, then tests the project on NPV, IRR, "
        "and payback."
    )

    inputs, v = collect_inputs()

    for w in v.warnings:
        st.warning(w)

    if not v.ok:
        st.error("Fix these before the model will run:")
        for e in v.errors:
            st.markdown(f"- {e}")
        st.stop()

    try:
        df = build_pro_forma(inputs)
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        st.error(
            "The assumptions produced numbers the model can't handle "
            f"({type(exc).__name__}). Reduce the growth rate, term, or dollar "
            "magnitudes and try again."
        )
        st.stop()

    if not df.replace([float("inf"), float("-inf")], float("nan")).notna().all().all():
        st.error(
            "The assumptions overflowed into infinite or undefined values. "
            "Lower the growth rate or the project term."
        )
        st.stop()

    show_results(inputs, df)


if __name__ == "__main__":
    main()
