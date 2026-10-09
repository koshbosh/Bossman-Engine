import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import os
import warnings

from regression import NumPyOLS
from score_backtest import fetch_wikipedia_sp500, get_point_in_time_sp500, build_features_for_month

warnings.simplefilter(action='ignore', category=FutureWarning)

FACTORS_TO_TEST = [
    'momentum_1mo', 'momentum_3mo', 'momentum_6mo', 'momentum_1y',
    'ma50_proximity', 'ma200_proximity', 'rsi', 'annualized_vol'
]

def generate_factor_html_report(factor_stats, n_obs):
    sorted_factors = sorted(factor_stats.items(), key=lambda x: abs(x[1]['corr']), reverse=True)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Factor-by-Factor Attribution Analysis</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #58a6ff; }}
            .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #30363d; padding: 12px; text-align: left; }}
            th {{ background-color: #21262d; font-weight: 600; }}
            .positive {{ color: #3fb950; }}
            .negative {{ color: #f85149; }}
            .signal {{ color: #a371f7; font-weight: bold; }}
            .noise {{ color: #8b949e; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Factor-by-Factor Attribution Analysis</h1>
            <p><strong>N = {n_obs:,} observations</strong> | Data excludes survivorship and look-ahead biases.</p>
            
            <div class="card">
                <h2>Factor Leaderboard (Sorted by Signal Strength)</h2>
                <p>Factors are evaluated independently for their ability to predict 1-month forward returns.</p>
                <table>
                    <tr>
                        <th>Factor</th>
                        <th>Spearman Corr</th>
                        <th>OLS p-value</th>
                        <th>Q1 (Highest) Return</th>
                        <th>Q5 (Lowest) Return</th>
                        <th>Spread (Q1 - Q5)</th>
                        <th>Verdict</th>
                    </tr>
    """
    
    for factor, stats in sorted_factors:
        is_significant = stats['p_value'] < 0.05
        verdict = "<span class='signal'>SIGNAL</span>" if is_significant else "<span class='noise'>NOISE</span>"
        
        html += f"""
                    <tr>
                        <td><strong>{factor}</strong></td>
                        <td class="{'positive' if stats['corr'] > 0 else 'negative'}">{stats['corr']:.4f}</td>
                        <td class="{'positive' if is_significant else ''}">{stats['p_value']:.4e}</td>
                        <td class="{'positive' if stats['q1_ret'] > 0 else 'negative'}">{(stats['q1_ret']*100):.2f}%</td>
                        <td class="{'positive' if stats['q5_ret'] > 0 else 'negative'}">{(stats['q5_ret']*100):.2f}%</td>
                        <td class="{'positive' if stats['spread'] > 0 else 'negative'}">{(stats['spread']*100):.2f}%</td>
                        <td>{verdict}</td>
                    </tr>
        """
        
    html += """
                </table>
                <p><em>* Q1 represents stocks with the highest values for the factor (e.g. highest RSI, highest momentum). Q5 represents the lowest.</em></p>
                <p><em>* A Signal verdict implies the OLS p-value is statistically significant (< 0.05).</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("factor_attribution_results.html", "w") as f:
        f.write(html)
    print("Report saved to factor_attribution_results.html")

def run_factor_backtest():
    if os.path.exists("factor_raw_data.csv"):
        print("Loading cached results from 'factor_raw_data.csv'...")
        results_df = pd.read_csv("factor_raw_data.csv")
    else:
        current_tickers, sector_map, changes_df = fetch_wikipedia_sp500()
        
        all_historical_tickers = set(current_tickers)
        for _, row in changes_df.iterrows():
            added = str(row.get('added', '')).strip()
            removed = str(row.get('removed', '')).strip()
            if added and added.upper() != 'NAN':
                all_historical_tickers.add(added)
            if removed and removed.upper() != 'NAN':
                all_historical_tickers.add(removed)
                
        all_historical_tickers = list(all_historical_tickers)
        print(f"Total unique tickers to fetch (including historical): {len(all_historical_tickers)}")
        
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(years=10)
        fetch_start = start_date - relativedelta(years=2) 
        
        print(f"Downloading daily prices from {fetch_start} to {end_date}...")
        prices = yf.download(all_historical_tickers + ['SPY'], start=fetch_start, end=end_date, progress=False, group_by='ticker')
        
        close_prices_dict = {}
        for t in all_historical_tickers + ['SPY']:
            if t in prices.columns.levels[0]:
                if 'Close' in prices[t].columns:
                    close_prices_dict[t] = prices[t]['Close']
        close_prices = pd.DataFrame(close_prices_dict)
                
        if 'SPY' not in close_prices.columns:
            print("Failed to download SPY.")
            return
            
        monthly_prices = close_prices.resample('ME').last()
        spy_rets = close_prices['SPY'].pct_change().dropna()
        
        valid_dates = [d for d in monthly_prices.index if start_date <= d.date() < monthly_prices.index[-1].date()]
        print(f"Running monthly backtest loop across {len(valid_dates)} months...")
        
        results = []
        
        for i, date in enumerate(valid_dates):
            print(f"Processing month: {date.strftime('%Y-%m')}...", end='\r')
            
            pit_universe = get_point_in_time_sp500(date, current_tickers, changes_df)
            hist_p = close_prices.loc[:date]
            if hist_p.empty: continue
                
            stocks_df = build_features_for_month(date, hist_p, spy_rets, pit_universe, sector_map)
            if stocks_df.empty: continue
                
            next_date = monthly_prices.index[monthly_prices.index.get_loc(date) + 1]
            
            for _, row in stocks_df.iterrows():
                t = row['ticker']
                try:
                    p_now = monthly_prices.loc[date, t]
                    p_next = monthly_prices.loc[next_date, t]
                    if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                        fwd_ret = (p_next / p_now) - 1
                        
                        record = {
                            'date': date,
                            'ticker': t,
                            'forward_return': fwd_ret
                        }
                        for f in FACTORS_TO_TEST:
                            record[f] = row[f]
                            
                        results.append(record)
                except KeyError:
                    pass
        
        print("\nBacktest loop complete.")
        results_df = pd.DataFrame(results)
        results_df.to_csv("factor_raw_data.csv", index=False)
        print("Saved raw backtest results to 'factor_raw_data.csv'.")
        
    print("Evaluating individual factors...")
    factor_stats = {}
    
    for factor in FACTORS_TO_TEST:
        clean_df = results_df[['forward_return', factor]].dropna().copy()
        if clean_df.empty: continue
            
        X = clean_df[[factor]].values
        y = clean_df['forward_return'].values
        
        ols = NumPyOLS()
        ols.fit(X, y, [factor])
        
        # Pearson rank correlation (Spearman substitute)
        spearman_corr = clean_df[factor].rank().corr(clean_df['forward_return'].rank(), method='pearson')
        
        # Quintiles
        clean_df['quintile'] = pd.qcut(clean_df[factor], 5, labels=['Q5_Lowest', 'Q4', 'Q3', 'Q2', 'Q1_Highest'], duplicates='drop')
        q_rets = clean_df.groupby('quintile', observed=False)['forward_return'].mean() * 12
        
        q1_ret = q_rets.get('Q1_Highest', float('nan'))
        q5_ret = q_rets.get('Q5_Lowest', float('nan'))
        
        factor_stats[factor] = {
            'coef': float(ols.beta[1]),
            't_stat': float(ols.t_stats[1]),
            'p_value': float(ols.p_values[1]),
            'corr': float(spearman_corr),
            'q1_ret': float(q1_ret),
            'q5_ret': float(q5_ret),
            'spread': float(q1_ret - q5_ret)
        }
        
    generate_factor_html_report(factor_stats, len(results_df))

if __name__ == "__main__":
    run_factor_backtest()
