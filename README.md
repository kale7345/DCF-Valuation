# DCF Valuation Model

A Python discounted cash flow model that downloads a public company's annual financial statements with `yfinance`, projects five years of free cash flow, calculates CAPM/WACC, values terminal cash flows, and prints a WACC/terminal-growth sensitivity table.

## Run it

```powershell
py -m pip install yfinance pandas numpy
py dcf_valuation/dcf.py AAPL
```

Pass another ticker as the first argument:

```powershell
py dcf_valuation/dcf.py MSFT
```

The model uses the latest reported financials for revenue, EBIT, taxes, depreciation, capital expenditure, and non-cash working capital. It derives the operating assumptions from those values, while the forecast growth path, risk-free rate, equity risk premium, cost of debt, and terminal growth are kept in the `ASSUMPTIONS` dictionary in [dcf_valuation/dcf.py](dcf_valuation/dcf.py).

## Interpretation

The output's fair price is the model's implied value per share after subtracting net debt. The sensitivity grid shows how much that value changes when WACC and terminal growth move together; lower WACC and higher terminal growth increase the result because they reduce discounting and extend the value of cash flows beyond the forecast period. A fair price above the current market price implies the market may be assuming slower growth, lower margins, or higher risk than this model. Because terminal value is usually a large share of enterprise value, the sensitivity table should be treated as a range, not a precise target.

## Sensitivity table

Run the command above and capture the printed `Sensitivity: implied fair value per share` table for a dated model output or report. Results change as market data and company filings update.
