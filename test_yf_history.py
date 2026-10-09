import yfinance as yf
import pandas as pd

t = yf.Ticker("AAPL")

# Try to get income stmt
inc = t.quarterly_income_stmt
print("Income Statement columns:", inc.columns if inc is not None else "None")

# Try to get balance sheet
bs = t.quarterly_balance_sheet
print("Balance Sheet columns:", bs.columns if bs is not None else "None")

