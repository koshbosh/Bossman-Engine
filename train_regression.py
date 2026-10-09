"""
======================================================================
BASAK BOSSMAN ENGINE: 25-YEAR REGRESSION MODEL TRAINING SCRIPT
======================================================================
Fetches 25 years of S&P 500 price data and current fundamentals,
simulates historical features monthly to avoid survivorship bias,
and trains a NumPy-based OLS model to predict forward 1-month returns.

Saves the trained weights and scaling parameters to 'regression_model.json'.
======================================================================
"""
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
import requests
import json
import os
import sys
import warnings
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from regression import NumPyOLS
from recommender import calculate_ma_proximity, SECTOR_PE_BENCHMARKS

warnings.simplefilter(action='ignore', category=FutureWarning)

HISTORICAL_CONSTITUENTS_FILE = "sp500_historical_constituents.csv"

# ─────────────────────────────────────────────
# STEP 1: CONSTITUENT DATA & UNIVERSE LOADER
# ─────────────────────────────────────────────

def load_historical_constituents():
    try:
        df = pd.read_csv(HISTORICAL_CONSTITUENTS_FILE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        print(f"Loaded historical constituents: {len(df)} snapshots.")
        return df
    except FileNotFoundError:
        print(f"Warning: '{HISTORICAL_CONSTITUENTS_FILE}' not found. Survivorship bias will be present.")
        return None

def get_constituents_for_date(hist_df, date, fallback_tickers):
    if hist_df is None:
        return fallback_tickers

    valid = hist_df[hist_df.index <= date]
    if valid.empty:
        return fallback_tickers

    snapshot = valid.iloc[-1]
    if "tickers" in snapshot.index:
        tickers_str = snapshot["tickers"]
        if isinstance(tickers_str, str):
            active = [t.strip() for t in tickers_str.split(",") if t.strip()]
        else:
            active = []
    else:
        active = [col for col in snapshot.index if snapshot[col] == 1]

    active = [t for t in active if t in set(fallback_tickers)]
    return active if active else fallback_tickers

def get_sp500_data():
    print("Fetching current S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
    return df['Symbol'].tolist(), sector_map

# ─────────────────────────────────────────────
# STEP 2: MULTITHREADED CURRENT FUNDAMENTALS FETCH
# ─────────────────────────────────────────────

def fetch_single_fundamental(t):
    try:
        info = yf.Ticker(t).info
        return t, {
            "pe_ratio": info.get("trailingPE", None),
            "debt_to_equity": info.get("debtToEquity", 0) or 0
        }
    except Exception:
        return t, {"pe_ratio": None, "debt_to_equity": 0}

def fetch_fundamentals(tickers):
    print(f"Fetching current fundamentals for {len(tickers)} tickers using multithreading...")
    fund_data = {}
    total = len(tickers)
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_ticker = {executor.submit(fetch_single_fundamental, t): t for t in tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            t, res = future.result()
            fund_data[t] = res
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"  {completed}/{total} fetched...")
                
    return fund_data

# ─────────────────────────────────────────────
# STEP 3: BETA CALCULATION
# ─────────────────────────────────────────────

def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 60 or len(market_returns) < 60:
        return 1.0
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var != 0 else 1.0

# ─────────────────────────────────────────────
# STEP 4: MAIN TRAINING WORKFLOW
# ─────────────────────────────────────────────

def main():
    # Load universe
    all_tickers, sector_map = get_sp500_data()
    hist_constituents = load_historical_constituents()
    
    # Fetch fundamentals (used to simulate historical P/E and constant D/E)
    current_funds = fetch_fundamentals(all_tickers)
    
    # Dates setup (25 years: 2001 to present)
    start_date = "2000-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Downloading 25 years of daily price data ({start_date} -> {end_date})...")
    price_data = yf.download(
        all_tickers + ["SPY"], start=start_date, end=end_date, progress=True
    )['Close']
    
    # Resample to monthly to identify training snapshots
    monthly_prices = price_data.resample('ME').last()
    
    # Build SPY returns for beta calculation
    spy_rets = price_data["SPY"].pct_change().dropna()
    
    # We skip the first 2 years of daily data to allow feature windows (e.g. 200MA, 1y momentum, 1y beta)
    train_dates = [d for d in monthly_prices.index if d.year >= 2002 and d < monthly_prices.index[-1]]
    
    print(f"Sampling training snapshots over {len(train_dates)} months...")
    
    training_rows = []
    
    # Loop over historical dates
    for idx_date, date in enumerate(train_dates):
        # We need the next date to compute forward 1-month return
        next_date = monthly_prices.index[monthly_prices.index.get_loc(date) + 1]
        
        # Historical price slice up to date t
        hist_p = price_data.loc[:date]
        market_rets = spy_rets.loc[:date].iloc[-252:]
        
        # Get active constituents on this date
        active_tickers = get_constituents_for_date(hist_constituents, date, all_tickers)
        
        # Spot log progress every 24 months
        if idx_date % 24 == 0 or idx_date == len(train_dates) - 1:
            print(f"  Processing date {date.strftime('%Y-%m-%d')} ({len(active_tickers)} active stocks)...")
            
        for t in active_tickers:
            if t not in hist_p.columns or t == "SPY":
                continue
                
            s = hist_p[t].dropna()
            if len(s) < 252:
                continue
                
            # Price at current snapshot date
            p_now = s.iloc[-1]
            
            # Forward return over the next month
            p_next = monthly_prices.loc[next_date, t]
            if pd.isna(p_now) or pd.isna(p_next) or p_now <= 0 or p_next <= 0:
                continue
            fwd_return = (p_next / p_now) - 1.0
            
            # Feature calculations
            s_rets = s.iloc[-252:].pct_change().dropna()
            
            # Momentum
            m1mo = (p_now / s.iloc[-21] - 1) if len(s) >= 21 else 0
            m3mo = (p_now / s.iloc[-63] - 1) if len(s) >= 63 else 0
            m6mo = (p_now / s.iloc[-126] - 1) if len(s) >= 126 else 0
            m1y  = (p_now / s.iloc[-252] - 1)
            
            # RSI
            delta = s.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            
            # Vol & Beta
            vol = s_rets.std() * np.sqrt(252)
            beta = calculate_beta(s_rets, market_rets)
            
            # MA values
            ma_50 = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")
            ma_200 = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")
            if np.isnan(ma_50) or np.isnan(ma_200):
                continue
                
            ma50_prox = calculate_ma_proximity(p_now, ma_50)
            ma200_prox = calculate_ma_proximity(p_now, ma_200)
            
            # Fundamentals
            funds = current_funds.get(t, {})
            current_pe = funds.get("pe_ratio")
            de = funds.get("debt_to_equity", 0)
            
            # Estimate simulated historical P/E using price ratio
            price_current = price_data[t].iloc[-1]
            if current_pe is not None and price_current > 0 and p_now > 0:
                sim_pe = current_pe * (p_now / price_current)
            else:
                sector = sector_map.get(t, "N/A")
                sim_pe = SECTOR_PE_BENCHMARKS.get(sector, 23.5)
                
            # Clean extremes/outliers
            if sim_pe <= 0 or np.isnan(sim_pe):
                sector = sector_map.get(t, "N/A")
                sim_pe = SECTOR_PE_BENCHMARKS.get(sector, 23.5)
            sim_pe = np.clip(sim_pe, 2.0, 150.0)
            de = np.clip(de, 0.0, 500.0)
            
            training_rows.append({
                "forward_return": fwd_return,
                "pe_ratio": sim_pe,
                "debt_to_equity": de,
                "ma50_proximity": ma50_prox,
                "ma200_proximity": ma200_prox,
                "momentum_1y": m1y,
                "rsi": rsi,
                "annualized_vol": vol,
                "beta": beta
            })
            
    df_train = pd.DataFrame(training_rows)
    print(f"\nCreated training dataset with {len(df_train)} observations.")
    
    # Preprocess and drop NaN or infinite values
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Cleaned dataset has {len(df_train)} valid observations.")
    
    # Filter out extreme outlier returns (e.g. forward return > 50% or < -50% in a month)
    # to avoid OLS distortion from extreme noise
    df_train = df_train[(df_train['forward_return'] >= -0.50) & (df_train['forward_return'] <= 0.50)]
    print(f"Removed forward return outliers: {len(df_train)} observations remaining.")
    
    # Features matrix X and target y
    features = [
        "pe_ratio", "debt_to_equity", "ma50_proximity", "ma200_proximity",
        "momentum_1y", "rsi", "annualized_vol", "beta"
    ]
    X = df_train[features].values
    y = df_train["forward_return"].values
    
    # Fit OLS
    print("\nFitting Ordinary Least Squares regression...")
    ols = NumPyOLS()
    ols.fit(X, y, features)
    
    # Print summary
    print(ols.summary())
    
    # Save model weights to regression_model.json
    ols.save_model("regression_model.json")

if __name__ == "__main__":
    main()
