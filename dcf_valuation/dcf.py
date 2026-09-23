"""Assumptions-driven discounted cash flow valuation using yfinance."""

from __future__ import annotations

import sys
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "AAPL"

ASSUMPTIONS = {
    "revenue_growth": [0.08, 0.07, 0.06, 0.05, 0.04],
    "ebit_margin": None,
    "tax_rate": None,
    "da_pct_rev": None,
    "capex_pct_rev": None,
    "nwc_pct_rev": None,
    "wacc": 0.09,
    "terminal_growth": 0.025,
    "forecast_years": 5,
    "risk_free_rate": 0.045,
    "market_risk_premium": 0.05,
    "cost_of_debt": 0.05,
}


def get_row(statement: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return the first matching statement row, allowing minor label differences."""
    labels = {str(label).strip().casefold(): label for label in statement.index}
    for name in candidates:
        if name in statement.index:
            return statement.loc[name]
        match = labels.get(name.strip().casefold())
        if match is not None:
            return statement.loc[match]
    raise KeyError(
        f"None of {list(candidates)} found. Available rows: {list(statement.index)}"
    )


def latest_value(statement: pd.DataFrame, candidates: Iterable[str], position: int = 0) -> float:
    """Return a finite numeric value from a statement row."""
    values = pd.to_numeric(get_row(statement, candidates), errors="coerce")
    if len(values) <= position or pd.isna(values.iloc[position]):
        raise ValueError(f"No usable value found for {list(candidates)}")
    return float(values.iloc[position])


def latest_value_from_statements(
    statements: Iterable[pd.DataFrame], candidates: Iterable[str], default: float | None = None
) -> float:
    """Return a value from the first statement containing a usable candidate row."""
    for statement in statements:
        try:
            return latest_value(statement, candidates)
        except (KeyError, ValueError):
            continue
    if default is not None:
        return default
    raise KeyError(f"None of {list(candidates)} found in the supplied statements")


def first_info_value(info: dict, keys: Iterable[str], default: float | None = None) -> float | None:
    """Return the first finite numeric value from a yfinance info dictionary."""
    for key in keys:
        value = info.get(key)
        if value is not None and pd.notna(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return default


def load_inputs(ticker: str) -> dict[str, float | dict]:
    """Download statements and convert the latest reported values into model inputs."""
    stock = yf.Ticker(ticker)
    income = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow
    info = stock.info

    if income.empty or balance.empty or cashflow.empty:
        raise ValueError(f"Incomplete financial statements returned for {ticker}")

    revenue = latest_value(income, ["Total Revenue", "Operating Revenue"])
    ebit = latest_value(income, ["EBIT", "Operating Income"])
    pretax = latest_value(income, ["Pretax Income", "Income Before Tax"])
    tax_provision = latest_value(income, ["Tax Provision", "Income Tax Expense"])
    dep = abs(
        latest_value_from_statements(
            [cashflow, income],
            [
                "Depreciation And Amortization",
                "Depreciation",
                "Reconciled Depreciation",
                "Depreciation Amortization Depletion Income Statement",
                "Depreciation And Amortization In Income Statement",
            ],
            default=0.0,
        )
    )
    capex = abs(latest_value(cashflow, ["Capital Expenditure", "Capital Expenditures"]))

    current_assets = latest_value(balance, ["Total Current Assets", "Current Assets"])
    previous_assets = latest_value(balance, ["Total Current Assets", "Current Assets"], 1)
    cash_now = latest_value(
        balance, ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"]
    )
    cash_previous = latest_value(
        balance,
        ["Cash And Cash Equivalents", "Cash", "Cash Cash Equivalents And Short Term Investments"],
        1,
    )
    delta_nwc = (current_assets - cash_now) - (previous_assets - cash_previous)

    shares_out = first_info_value(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    market_cap = first_info_value(info, ["marketCap"])
    current_price = first_info_value(info, ["currentPrice", "regularMarketPrice", "previousClose"])
    debt = first_info_value(info, ["totalDebt"], 0.0) or 0.0
    cash = first_info_value(info, ["totalCash", "cash"], 0.0) or 0.0
    beta = first_info_value(info, ["beta"], 1.0) or 1.0

    if shares_out is None:
        raise ValueError(f"Could not determine shares outstanding for {ticker}")
    if market_cap is None and current_price is not None:
        market_cap = shares_out * current_price
    if market_cap is None:
        raise ValueError(f"Could not determine market capitalization for {ticker}")
    if current_price is None:
        raise ValueError(f"Could not determine current price for {ticker}")
    if pretax == 0:
        raise ValueError("Pretax income is zero; the tax rate cannot be estimated")

    tax_rate = max(tax_provision / pretax, 0.0)
    tax_rate = min(tax_rate, 0.35)
    assumptions = ASSUMPTIONS.copy()
    assumptions.update(
        {
            "ebit_margin": ebit / revenue,
            "tax_rate": tax_rate,
            "da_pct_rev": dep / revenue,
            "capex_pct_rev": capex / revenue,
            "nwc_pct_rev": delta_nwc / revenue,
        }
    )

    return {
        "revenue": revenue,
        "assumptions": assumptions,
        "shares_out": shares_out,
        "market_cap": market_cap,
        "debt": debt,
        "cash": cash,
        "net_debt": debt - cash,
        "beta": beta,
        "current_price": current_price,
    }


def project_free_cash_flows(revenue: float, assumptions: dict) -> pd.DataFrame:
    """Project revenue, operating profit, and free cash flow for each forecast year."""
    rows = []
    projected_revenue = revenue
    for year, growth in enumerate(assumptions["revenue_growth"], start=1):
        projected_revenue *= 1 + growth
        ebit = projected_revenue * assumptions["ebit_margin"]
        nopat = ebit * (1 - assumptions["tax_rate"])
        da = projected_revenue * assumptions["da_pct_rev"]
        capex = projected_revenue * assumptions["capex_pct_rev"]
        delta_nwc = projected_revenue * assumptions["nwc_pct_rev"]
        fcf = nopat + da - capex - delta_nwc
        rows.append(
            {
                "year": year,
                "revenue": projected_revenue,
                "ebit": ebit,
                "nopat": nopat,
                "da": da,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "fcf": fcf,
            }
        )
    return pd.DataFrame(rows)


def calculate_wacc(assumptions: dict, market_cap: float, debt: float, beta: float) -> tuple[float, float]:
    """Calculate CAPM cost of equity and capital-weighted WACC."""
    risk_free = assumptions["risk_free_rate"]
    mrp = assumptions["market_risk_premium"]
    cost_equity = risk_free + beta * mrp
    total_capital = market_cap + debt
    if total_capital <= 0:
        raise ValueError("Market capitalization plus debt must be positive")
    wacc = (market_cap / total_capital) * cost_equity + (debt / total_capital) * assumptions["cost_of_debt"] * (1 - assumptions["tax_rate"])
    return cost_equity, wacc


def valuation(projection: pd.DataFrame, assumptions: dict, net_debt: float, shares_out: float) -> tuple[dict, pd.DataFrame]:
    """Calculate DCF value and a WACC/terminal-growth sensitivity table."""
    wacc = assumptions["wacc"]
    terminal_growth = assumptions["terminal_growth"]
    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth")

    years = projection["year"].to_numpy()
    discounts = (1 + wacc) ** years
    pv_fcf = float((projection["fcf"] / discounts).sum())
    terminal_value = float(projection["fcf"].iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth))
    pv_terminal = terminal_value / discounts[-1]
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt
    fair_price = equity_value / shares_out

    waccs = np.linspace(wacc - 0.02, wacc + 0.02, 5)
    growth_rates = np.linspace(terminal_growth - 0.01, terminal_growth + 0.01, 5)
    sensitivity = pd.DataFrame(
        index=[f"{value:.1%}" for value in waccs],
        columns=[f"{value:.1%}" for value in growth_rates],
        dtype=float,
    )
    for wacc_value in waccs:
        for growth_rate in growth_rates:
            if wacc_value <= growth_rate:
                sensitivity.loc[f"{wacc_value:.1%}", f"{growth_rate:.1%}"] = np.nan
                continue
            discount_factors = (1 + wacc_value) ** years
            pv_forecast = float((projection["fcf"] / discount_factors).sum())
            terminal = projection["fcf"].iloc[-1] * (1 + growth_rate) / (wacc_value - growth_rate)
            pv_terminal_sensitivity = terminal / discount_factors[-1]
            sensitivity.loc[f"{wacc_value:.1%}", f"{growth_rate:.1%}"] = (pv_forecast + pv_terminal_sensitivity - net_debt) / shares_out

    return {
        "pv_fcf": pv_fcf,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_price": fair_price,
    }, sensitivity


def format_billions(value: float) -> str:
    return f"${value / 1e9:,.1f}B"


def main(ticker: str = TICKER) -> None:
    inputs = load_inputs(ticker.upper())
    assumptions = inputs["assumptions"]
    cost_equity, wacc = calculate_wacc(assumptions, inputs["market_cap"], inputs["debt"], inputs["beta"])
    assumptions["wacc"] = wacc
    projection = project_free_cash_flows(inputs["revenue"], assumptions)
    results, sensitivity = valuation(projection, assumptions, inputs["net_debt"], inputs["shares_out"])

    print(f"\n{ticker.upper()} DCF valuation")
    print("\nProjection (USD):")
    print(projection.round(0).to_string(index=False))
    print(f"\nCost of equity: {cost_equity:.2%} | WACC: {wacc:.2%}")
    print(f"Enterprise value: {format_billions(results['enterprise_value'])}")
    print(f"Equity value:     {format_billions(results['equity_value'])}")
    print(f"Fair price:       ${results['fair_price']:.2f} vs market ${inputs['current_price']:.2f}")
    print("\nSensitivity: implied fair value per share")
    print(sensitivity.round(2).to_string())


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else TICKER)
    except Exception as error:
        raise SystemExit(f"DCF valuation failed: {error}") from error
