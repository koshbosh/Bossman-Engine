import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from recommender import (
    generate_recommendations, normalize, risk_penalty, 
    RISK_LEVELS, SECTOR_SAFE_LIMITS, SECTOR_PE_BENCHMARKS, calculate_value_score
)
import requests
from io import StringIO
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_sp500_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
    return df['Symbol'].tolist(), sector_map

def fetch_fundamentals(tickers):
    """Fetch current P/E ratios for the entire universe"""
    print(f"Fetching current P/E ratios for {len(tickers)} tickers...")
    pe_map = {}
    
    def get_pe(t):
        try:
            return t, yf.Ticker(t).info.get("trailingPE", 20.0)
        except:
            return t, 20.0

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(get_pe, tickers))
        for t, pe in results:
            pe_map[t] = pe
    return pe_map

def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 20 or len(market_returns) < 20: return 1.0
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 20: return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var != 0 else 1.0

def score_robust(stocks_df, profile_name, top_n=25):
    df = stocks_df.copy()
    cfg = RISK_LEVELS[profile_name]
    
    # ── Normalize MA proximity features (primary scoring signals) ───
    for col in ["ma50_proximity", "ma200_proximity"]:
        if col in df.columns:
            df[col + "_norm"] = normalize(df[col])

    # ── Normalize momentum features (used by gear-shift + penalties) ─
    for col in ["momentum_1mo", "momentum_3mo", "momentum_6mo", "momentum_1y"]:
        if col in df.columns:
            df[col + "_norm"] = normalize(df[col])
    
    scores = []
    use_conditional = profile_name in ["All-Weather Core", "Dip Hunter", "Alpha Bull"]
    use_robustness = cfg["robustness_active"]
    
    base_ma50_wt = cfg["ma50_proximity_weight"]
    orig_ma200_wt = cfg["ma200_proximity_weight"]
    base_3mo_wt = orig_ma200_wt / 2
    base_ma200_wt = orig_ma200_wt / 2
    base_1y_wt = cfg["momentum_1y_weight"]
    
    for _, row in df.iterrows():
        if use_conditional:
            if row.get("momentum_3mo", 0) > row.get("momentum_1mo", 0):
                w_ma50, w_ma200 = base_ma50_wt - 0.05, base_ma200_wt + 0.05
            else:
                w_ma50, w_ma200 = base_ma50_wt + 0.05, base_ma200_wt - 0.05
            w_3mo = base_3mo_wt
        else:
            w_ma50, w_3mo, w_ma200 = base_ma50_wt, 0.0, orig_ma200_wt

        risk_s = risk_penalty(row["annualized_vol"], cfg["vol_limit"])
        value_s = calculate_value_score(row.get("pe_ratio", 20.0), row["sector"])
        
        base_score = (
            w_ma50 * row.get("ma50_proximity_norm", 0.5) +
            w_3mo * row.get("momentum_3mo_norm", 0) +
            w_ma200 * row.get("ma200_proximity_norm", 0.5) +
            base_1y_wt * row["momentum_1y_norm"] +
            cfg["risk_weight"] * risk_s +
            cfg["value_weight"] * value_s
        )
        
        final_penalty = 1.0
        if use_robustness:
            sector_limit = SECTOR_SAFE_LIMITS.get(row["sector"], 0.08)
            safe_limit = sector_limit * (1 + max(0, row.get("beta", 1.0)))
            if row.get("perf_1w", 0) > safe_limit:
                final_penalty *= np.clip(1.0 - ((row["perf_1w"] - safe_limit) * 2.5), 0.5, 1.0)
            final_penalty *= np.clip(1.0 - (max(0, row.get("rsi", 50) - 70) / 100), 0.7, 1.0)
            
        scores.append(base_score * final_penalty)
    
    df["recommendation_score"] = scores
    return df.sort_values("recommendation_score", ascending=False).head(top_n)

def run_backtest(name, tickers, pe_map, sector_map, start_date_str, end_date_str, top_n=25):
    print(f"\nRUNNING BACKTEST: {name}")
    all_tickers = tickers + ["SPY"]
    fetch_start = datetime.strptime(start_date_str, '%Y-%m-%d') - relativedelta(years=1, months=6)
    prices = yf.download(all_tickers, start=fetch_start, end=end_date_str, progress=False)['Close']
    spy_rets = prices["SPY"].pct_change().dropna()
    monthly_prices = prices.resample('ME').last()
    valid_dates = [d for d in monthly_prices.index if datetime.strptime(start_date_str, '%Y-%m-%d') <= d < monthly_prices.index[-1]]
    
    profiles = list(RISK_LEVELS.keys())
    portfolio_returns = {p: [] for p in profiles}
    benchmark_returns = []
    
    for date in valid_dates:
        hist_p = prices.loc[:date]
        market_rets = spy_rets.loc[:date].iloc[-252:]
        stocks_data = []
        for t in tickers:
            if t not in hist_p.columns: continue
            s = hist_p[t].dropna()
            if len(s) < 252: continue
            p_now = s.iloc[-1]
            s_rets = s.iloc[-252:].pct_change().dropna()
            
            delta = s.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            
            # Simulated Historical P/E (Adjusting current PE by price change)
            current_p = prices[t].iloc[-1]
            sim_pe = pe_map.get(t, 20.0) * (p_now / current_p) if current_p > 0 else 20.0
            
            # ── MA Proximity Scores ──────────────────────────────────────
            ma_50_val = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")
            ma_200_val = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")

            from recommender import calculate_ma_proximity


            ma50_prox = calculate_ma_proximity(p_now, ma_50_val)


            ma200_prox = calculate_ma_proximity(p_now, ma_200_val)

            stocks_data.append({
                "ticker": t,
                "momentum_1mo": (p_now / s.iloc[-21] - 1) if len(s) >= 21 else 0,
                "momentum_3mo": (p_now / s.iloc[-63] - 1) if len(s) >= 63 else 0,
                "momentum_6mo": (p_now / s.iloc[-126] - 1) if len(s) >= 126 else 0,
                "momentum_1y": (p_now / s.iloc[-252] - 1),
                "annualized_vol": s_rets.std() * np.sqrt(252),
                "rsi": rsi,
                "beta": calculate_beta(s_rets, market_rets),
                "perf_1w": (p_now / s.iloc[-5] - 1) if len(s) >= 5 else 0,
                "pe_ratio": sim_pe,
                "sector": sector_map.get(t, "N/A"),
                "ma_50": ma_50_val,
                "ma_200": ma_200_val,
                "ma50_proximity": ma50_prox,
                "ma200_proximity": ma200_prox,
            })
            
        if not stocks_data: continue
        stocks_df = pd.DataFrame(stocks_data)
        next_date = monthly_prices.index[monthly_prices.index.get_loc(date) + 1]
        
        for p in profiles:
            recs = score_robust(stocks_df, p, top_n=top_n)
            rets = []
            for t in recs['ticker'].tolist():
                p1, p2 = monthly_prices.loc[date, t], monthly_prices.loc[next_date, t]
                if pd.notna(p1) and pd.notna(p2) and p1 > 0: rets.append((p2 / p1) - 1)
            portfolio_returns[p].append(np.mean(rets) if rets else 0)
        
        bench = []
        for t in tickers:
            if t in monthly_prices.columns:
                p1, p2 = monthly_prices.loc[date, t], monthly_prices.loc[next_date, t]
                if pd.notna(p1) and pd.notna(p2) and p1 > 0: bench.append((p2 / p1) - 1)
        benchmark_returns.append(np.mean(bench) if bench else 0)

    print(f"\n--- RESULTS: {name} ---")
    for p in profiles:
        cum = np.prod([1 + r for r in portfolio_returns[p]]) - 1
        print(f"  {p:20}: {cum:>8.2%}")
    print(f"  {'Benchmark':20}: {np.prod([1+r for r in benchmark_returns])-1:>8.2%}")

if __name__ == "__main__":
    tickers, sector_map = get_sp500_data()
    pe_map = fetch_fundamentals(tickers)
    
    # 1. Past 12 Months
    run_backtest("Past 12 Months", tickers, pe_map, sector_map, (datetime.today() - relativedelta(months=12)).strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d'))
    # 2. Stagflation 2022
    run_backtest("Stagflation 2022", tickers, pe_map, sector_map, '2021-12-01', '2022-12-31')
    # 3. COVID Recovery 2020
    run_backtest("COVID Recovery 2020", tickers, pe_map, sector_map, '2019-12-01', '2020-12-31')
    # 4. Financial Crisis 2008
    run_backtest("Financial Crisis 2008", tickers, pe_map, sector_map, '2007-12-01', '2008-12-31')
