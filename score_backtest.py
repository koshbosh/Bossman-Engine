import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import datetime
from dateutil.relativedelta import relativedelta
import warnings
import json

from regression import NumPyOLS
from recommender import (
    generate_recommendations,
    RISK_LEVELS,
    SECTOR_PE_BENCHMARKS
)

warnings.simplefilter(action='ignore', category=FutureWarning)

HTML_REPORT_PATH = "score_backtest_results.html"

# ─────────────────────────────────────────────
# 1. WIKIPEDIA HISTORICAL CONSTITUENTS
# ─────────────────────────────────────────────

def fetch_wikipedia_sp500():
    print("Fetching S&P 500 constituents and history from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    
    # Table 0: Current Constituents
    current_df = tables[0].copy()
    current_df['Symbol'] = current_df['Symbol'].str.replace('.', '-', regex=False)
    current_tickers = current_df['Symbol'].tolist()
    sector_map = dict(zip(current_df['Symbol'], current_df['GICS Sector']))
    
    # Table 1: Historical Changes
    changes_df = tables[1].copy()
    
    # Flatten multi-index columns if present
    if isinstance(changes_df.columns, pd.MultiIndex):
        changes_df.columns = [
            f"{col[0]}_{col[1]}" if col[0] != col[1] else col[0] 
            for col in changes_df.columns
        ]
    
    # Normalize column names depending on Wikipedia's current format
    col_map = {}
    for col in changes_df.columns:
        c = col.lower()
        if 'date' in c: col_map[col] = 'date'
        elif 'added_ticker' in c: col_map[col] = 'added'
        elif 'removed_ticker' in c: col_map[col] = 'removed'
        
    changes_df = changes_df.rename(columns=col_map)
    changes_df['date'] = pd.to_datetime(changes_df['date'], errors='coerce')
    changes_df = changes_df.dropna(subset=['date']).sort_values('date', ascending=False)
    
    # Clean tickers
    if 'added' in changes_df.columns:
        changes_df['added'] = changes_df['added'].astype(str).str.replace('.', '-', regex=False).replace('nan', '')
    if 'removed' in changes_df.columns:
        changes_df['removed'] = changes_df['removed'].astype(str).str.replace('.', '-', regex=False).replace('nan', '')
        
    return current_tickers, sector_map, changes_df

def get_point_in_time_sp500(target_date, current_tickers, changes_df):
    """
    Reconstructs the S&P 500 universe at `target_date` by rolling backwards from today.
    """
    tickers = set(current_tickers)
    
    # Get all changes that happened AFTER our target date
    # Because we are rolling backwards, we undo these changes.
    recent_changes = changes_df[changes_df['date'] > pd.Timestamp(target_date)]
    
    for _, row in recent_changes.iterrows():
        added = str(row.get('added', '')).strip()
        removed = str(row.get('removed', '')).strip()
        
        # Undo addition: it was added after target_date, so it wasn't in the index yet
        if added and added in tickers:
            tickers.remove(added)
            
        # Undo removal: it was removed after target_date, so it was still in the index
        if removed and removed.upper() != 'NAN':
            tickers.add(removed)
            
    return list(tickers)

# ─────────────────────────────────────────────
# 2. FEATURE GENERATION (No Fundamentals)
# ─────────────────────────────────────────────

def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 20 or len(market_returns) < 20: return 1.0
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 20: return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var != 0 else 1.0

def build_features_for_month(target_date, hist_p, spy_rets, universe, sector_map):
    """
    Builds the feature dataframe exactly as generate_recommendations expects,
    but neutralizes P/E and D/E to avoid look-ahead bias.
    """
    stocks_data = []
    
    # Use trailing 252 days for market returns
    if len(spy_rets.loc[:target_date]) < 252:
        return pd.DataFrame()
    market_rets = spy_rets.loc[:target_date].iloc[-252:]
    
    for t in universe:
        if t not in hist_p.columns: continue
        s = hist_p[t].dropna()
        if len(s) < 252: continue
        
        p_now = float(s.iloc[-1])
        if p_now <= 0: continue
            
        s_rets = s.iloc[-252:].pct_change().dropna()
        
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        ma_50_val = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")
        ma_200_val = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")
        
        # Calculate proximities inline instead of importing calculation to avoid circular dependencies
        ma50_prox = (p_now - ma_50_val) / ma_50_val if ma_50_val > 0 else 0
        ma200_prox = (p_now - ma_200_val) / ma_200_val if ma_200_val > 0 else 0
        
        sector = sector_map.get(t, "N/A")
        neutral_pe = SECTOR_PE_BENCHMARKS.get(sector, 20.0)
        
        stocks_data.append({
            "ticker": t,
            "price": p_now,
            "momentum_1mo": (p_now / s.iloc[-21] - 1) if len(s) >= 21 else 0,
            "momentum_3mo": (p_now / s.iloc[-63] - 1) if len(s) >= 63 else 0,
            "momentum_6mo": (p_now / s.iloc[-126] - 1) if len(s) >= 126 else 0,
            "momentum_1y": (p_now / s.iloc[-252] - 1),
            "annualized_vol": s_rets.std() * np.sqrt(252),
            "rsi": rsi,
            "beta": calculate_beta(s_rets, market_rets),
            "perf_1w": (p_now / s.iloc[-5] - 1) if len(s) >= 5 else 0,
            "pe_ratio": neutral_pe,       # Neutralized (No look-ahead bias)
            "debt_to_equity": 0.0,        # Neutralized
            "sector": sector,
            "ma_50": ma_50_val,
            "ma_200": ma_200_val,
            "ma50_proximity": ma50_prox,
            "ma200_proximity": ma200_prox,
            "short_percent": 0.0,
            "current_volume": 1000000,    # Neutralized Volume
            "avg_volume": 1000000
        })
        
    return pd.DataFrame(stocks_data)


# ─────────────────────────────────────────────
# 3. HTML REPORT GENERATION
# ─────────────────────────────────────────────

def generate_html_report(results_df, ols_results):
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Score vs Return Correlation - Basak Bossman Engine</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #58a6ff; }}
            .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            .chart-container {{ position: relative; height: 400px; width: 100%; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
            th {{ background-color: #21262d; }}
            .positive {{ color: #3fb950; }}
            .negative {{ color: #f85149; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .metric-box {{ background-color: #21262d; border-radius: 6px; padding: 15px; text-align: center; border: 1px solid #30363d; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; color: #58a6ff; }}
            .profile-section {{ border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 20px; }}
            .profile-section:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Score vs. Forward Return Correlation (10-Year Backtest)</h1>
            <p><strong>N = {len(results_df):,} observations</strong> | Data excludes survivorship bias.</p>
            
            <div class="card">
                <h2>Statistical Significance per Profile</h2>
                <p><em>OLS Regression: Forward 1M Return ~ Recommendation Score</em></p>
    """
    
    for profile, stats in ols_results.items():
        html += f"""
                <div class="profile-section">
                    <h3>{profile}</h3>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div>OLS Coef</div>
                            <div class="metric-value {'positive' if stats['coef'] > 0 else 'negative'}">
                                {stats['coef']:.6f}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div>t-statistic</div>
                            <div class="metric-value {'positive' if stats['t_stat'] > 2 else ('negative' if stats['t_stat'] < -2 else '')}">
                                {stats['t_stat']:.2f}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div>p-value</div>
                            <div class="metric-value">
                                {stats['p_value']:.4e}
                            </div>
                        </div>
                        <div class="metric-box">
                            <div>Spearman Corr</div>
                            <div class="metric-value {'positive' if stats['corr'] > 0 else 'negative'}">
                                {stats['corr']:.4f}
                            </div>
                        </div>
                    </div>
                </div>
        """
    
    html += """
            </div>
    """
    
    # Process Quintiles for all profiles
    html += """<div class="card"><h2>Quintile Analysis (Annualized Returns)</h2>"""
    
    profiles = results_df['profile'].unique()
    quintile_data_js = {}
    
    for profile in profiles:
        prof_df = results_df[results_df['profile'] == profile].copy()
        try:
            # Q5 is highest score (since qcut bins ascending)
            prof_df['quintile'] = pd.qcut(prof_df['recommendation_score'], 5, labels=['Q5_Lowest', 'Q4', 'Q3', 'Q2', 'Q1_Highest'])
            
            # Group by quintile, mean 1M return, annualize it
            q_rets = prof_df.groupby('quintile')['forward_return'].mean() * 12
            
            # Sort so Q1_Highest is first
            q_rets = q_rets.sort_index(ascending=False)
            
            quintile_data_js[profile] = q_rets.tolist()
            
            html += f"<h3>{profile}</h3><table><tr><th>Quintile</th><th>Mean Annualized Return</th></tr>"
            for q, ret in q_rets.items():
                html += f"<tr><td>{q}</td><td class='{'positive' if ret > 0 else 'negative'}'>{(ret*100):.2f}%</td></tr>"
            html += "</table>"
            
        except Exception as e:
            html += f"<p>Error computing quintiles for {profile}: {e}</p>"
            
    html += f"""
                <div class="chart-container">
                    <canvas id="quintileChart"></canvas>
                </div>
            </div>
            
            <script>
                const ctx = document.getElementById('quintileChart').getContext('2d');
                const chartData = {json.dumps(quintile_data_js)};
                
                // Use All-Weather Core for the main chart, or the first available
                const activeProfile = 'All-Weather Core' in chartData ? 'All-Weather Core' : Object.keys(chartData)[0];
                const data = chartData[activeProfile];
                
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: ['Q1 (Highest Score)', 'Q2', 'Q3', 'Q4', 'Q5 (Lowest Score)'],
                        datasets: [{{
                            label: `Annualized Return (${{activeProfile}})`,
                            data: [data[0]*100, data[1]*100, data[2]*100, data[3]*100, data[4]*100],
                            backgroundColor: [
                                '#3fb950', '#2ea043', '#8b949e', '#f85149', '#da3633'
                            ],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ labels: {{ color: '#c9d1d9' }} }}
                        }},
                        scales: {{
                            y: {{
                                title: {{ display: true, text: 'Annualized Return (%)', color: '#c9d1d9' }},
                                ticks: {{ color: '#c9d1d9' }},
                                grid: {{ color: '#30363d' }}
                            }},
                            x: {{
                                ticks: {{ color: '#c9d1d9' }},
                                grid: {{ color: '#30363d' }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
    </body>
    </html>
    """
    
    with open(HTML_REPORT_PATH, 'w') as f:
        f.write(html)
    print(f"Interactive report saved to: {HTML_REPORT_PATH}")


# ─────────────────────────────────────────────
# 4. MAIN BACKTEST LOOP
# ─────────────────────────────────────────────

def run_backtest():
    import os
    if os.path.exists("score_backtest_raw_data.csv"):
        print("Loading cached results from 'score_backtest_raw_data.csv'...")
        results_df = pd.read_csv("score_backtest_raw_data.csv")
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
            
            for profile in RISK_LEVELS.keys():
                scored_df = generate_recommendations(stocks_df, user_profile=profile, top_n=None, pe_benchmarks=SECTOR_PE_BENCHMARKS)
                
                for _, row in scored_df.iterrows():
                    t = row['ticker']
                    score = row['recommendation_score']
                    
                    try:
                        p_now = monthly_prices.loc[date, t]
                        p_next = monthly_prices.loc[next_date, t]
                        if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                            fwd_ret = (p_next / p_now) - 1
                            results.append({
                                'date': date,
                                'ticker': t,
                                'profile': profile,
                                'recommendation_score': score,
                                'forward_return': fwd_ret
                            })
                    except KeyError:
                        pass
        
        print("\nBacktest loop complete.")
        results_df = pd.DataFrame(results)
        results_df.to_csv("score_backtest_raw_data.csv", index=False)
        print("Saved raw backtest results to 'score_backtest_raw_data.csv'.")
    
    # 5. OLS Regression Analysis per Profile
    print("Running OLS Regression and Correlation Analysis per profile...")
    ols_results = {}
    
    for profile in results_df['profile'].unique():
        prof_df = results_df[results_df['profile'] == profile].copy()
        
        X = prof_df[['recommendation_score']].values
        y = prof_df['forward_return'].values
        
        ols = NumPyOLS()
        ols.fit(X, y, ['recommendation_score'])
        
        # Calculate Spearman correlation manually using ranks since scipy is not installed
        spearman_corr = prof_df['recommendation_score'].rank().corr(prof_df['forward_return'].rank(), method='pearson')
        
        ols_results[profile] = {
            'coef': float(ols.beta[1]),
            't_stat': float(ols.t_stats[1]),
            'p_value': float(ols.p_values[1]),
            'r2': float(ols.r2),
            'corr': float(spearman_corr)
        }
        
    generate_html_report(results_df, ols_results)
    
if __name__ == "__main__":
    run_backtest()
