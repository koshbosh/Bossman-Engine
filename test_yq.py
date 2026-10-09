from yahooquery import Ticker
t = Ticker('AAPL')
inc = t.income_statement(frequency='q')
if isinstance(inc, str):
    print("Error:", inc)
else:
    print("YahooQuery Income Statement dates:", inc['asOfDate'].unique() if 'asOfDate' in inc.columns else inc.columns)
