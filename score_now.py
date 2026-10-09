"""
======================================================================
BASAK BOSSMAN ENGINE: QUICK TICKER SCORER
======================================================================
Calculates the current recommendation scores for specific stocks
relative to the S&P 500 universe.

This script is optimized for speed:
- Fetches S&P 500 price data for a stable normalization baseline.
- Fetches full fundamentals (P/E, D/E) only for the requested tickers.
- Uses sector defaults for the rest of the universe.

Usage:
    python score_now.py AAPL TSLA MSFT
    python score_now.py --profile high NVDA AMD
======================================================================
"""
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
import requests
import argparse
import sys
from io import StringIO
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from recommender import (
    generate_recommendations, 
    RISK_LEVELS, 
    SECTOR_PE_BENCHMARKS,
    SECTOR_DE_THRESHOLDS
)

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_sp500_tickers():
    """Fetches S&P 500 ticker list and sector map."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
        return df['Symbol'].tolist(), sector_map
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return [], {}

import os
from yahooquery import Ticker

def fetch_data(target_tickers, baseline_tickers):
    """
    Downloads price data for target + baseline.
    Downloads fundamentals ONLY for target using yahooquery.
    """
    all_tickers = list(set(target_tickers + baseline_tickers + ["SPY"]))
    
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(
        all_tickers,
        period="2y",
        auto_adjust=True,
        progress=False
    )['Close']
    
    print(f"Fetching live fundamentals for {len(target_tickers)} target stocks using yahooquery...")
    fund_data = {}
    
    if target_tickers:
        t = Ticker(target_tickers)
        sum_det = t.summary_detail
        fin_data = t.financial_data
        key_stats = t.key_stats
        prof = t.summary_profile
        
        for tick in target_tickers:
            sd = sum_det.get(tick, {}) if isinstance(sum_det, dict) else {}
            fd = fin_data.get(tick, {}) if isinstance(fin_data, dict) else {}
            ks = key_stats.get(tick, {}) if isinstance(key_stats, dict) else {}
            pr = prof.get(tick, {}) if isinstance(prof, dict) else {}
            
            if isinstance(sd, str): sd = {}
            if isinstance(fd, str): fd = {}
            if isinstance(ks, str): ks = {}
            if isinstance(pr, str): pr = {}
            
            fund_data[tick] = {
                "pe_ratio": sd.get("trailingPE"),
                "debt_to_equity": fd.get("debtToEquity", 0) or 0,
                "sector": pr.get("sector", "N/A"),
                "ma_200": sd.get("twoHundredDayAverage"),
                "peg_ratio": ks.get("pegRatio"),
            }
            
    return data, fund_data


def calculate_beta(stock_returns, market_returns):
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    var = np.var(m)
    return np.cov(s, m)[0][1] / var if var != 0 else 1.0

def build_feature_table(tickers, data, fund_data, sector_map, pe_benchmarks=None):
    if pe_benchmarks is None:
        pe_benchmarks = SECTOR_PE_BENCHMARKS
        
    spy_rets = data["SPY"].pct_change().dropna()
    records = []
    
    for t in tickers:
        if t not in data.columns:
            continue
            
        s = data[t].dropna()
        if len(s) < 252:
            continue
            
        p_now = s.iloc[-1]
        s_rets = s.pct_change().dropna()
        
        # Momentum (still needed for gear-shift, penalties, sector bonus)
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
        ann_vol = s_rets.iloc[-252:].std() * np.sqrt(252)
        beta = calculate_beta(s_rets.iloc[-252:], spy_rets)
        
        # Fundamentals
        # If target, use fund_data. Otherwise use defaults.
        if t in fund_data:
            sector = fund_data[t].get("sector", "N/A")
            pe = fund_data[t].get("pe_ratio", None)
            de = fund_data[t].get("debt_to_equity", 0)
            ma_200_val = fund_data[t].get("ma_200", None)
            peg = fund_data[t].get("peg_ratio", None)
        else:
            sector = sector_map.get(t, "N/A")
            pe = None
            de = 0
            ma_200_val = None
            peg = None
            
        if pe is None or pe <= 0 or pe != pe:
            pe = pe_benchmarks.get(sector, 23.5)

        # Fall back to computing 200MA from price history if yfinance didn't provide it
        if ma_200_val is None or (isinstance(ma_200_val, float) and np.isnan(ma_200_val)):
            ma_200_val = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")

        # ── 50-day Moving Average ──────────────────────────────────────
        ma_50_val = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")

        # ── MA Proximity Scores ────────────────────────────────────────
        from recommender import calculate_ma_proximity
        ma50_prox = calculate_ma_proximity(p_now, ma_50_val)
        ma200_prox = calculate_ma_proximity(p_now, ma_200_val)

        records.append({
            "ticker": t,
            "sector": sector,
            "price": round(p_now, 2),
            "momentum_1mo": m1mo,
            "momentum_3mo": m3mo,
            "momentum_6mo": m6mo,
            "momentum_1y": m1y,
            "annualized_vol": ann_vol,
            "rsi": round(rsi, 1),
            "beta": round(beta, 2),
            "perf_1w": (p_now / s.iloc[-5] - 1) if len(s) >= 5 else 0,
            "pe_ratio": pe,
            "peg_ratio": peg,
            "debt_to_equity": de,
            "ma_200": round(ma_200_val, 2) if not np.isnan(ma_200_val) else None,
            "ma_50": round(ma_50_val, 2) if not np.isnan(ma_50_val) else None,
            "ma50_proximity": ma50_prox,
            "ma200_proximity": ma200_prox,
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Basak Bossman Quick Scorer")
    parser.add_argument("tickers", nargs="*", help="Tickers to score")
    parser.add_argument("--profile", type=str, default="Dip Hunter",
                        choices=list(RISK_LEVELS.keys()),
                        help="Risk profile to use for scoring")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Show scores for all risk profiles")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top picks to show when scanning the S&P 500")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Skip live fundamentals fetch and use sector benchmarks (much faster)")
    args = parser.parse_args()
    
    # 1. Baseline Universe
    sp500_tickers, sp500_sectors = get_sp500_tickers()
    
    tickers = args.tickers
    is_full_scan = False
    if not tickers:
        is_full_scan = True
        tickers = sp500_tickers
        print(f"No tickers specified. Performing a full S&P 500 scan ({len(tickers)} stocks)...")
    
    target_tickers = [t.upper() for t in tickers]
    
    if not is_full_scan:
        print(f"Target Tickers: {', '.join(target_tickers)}")
    
    # 2. Fetch Data
    baseline_tickers = sp500_tickers if is_full_scan else []
    
    if args.skip_fundamentals:
        print("Skipping live fundamentals fetch (using sector defaults for P/E & D/E)...")
        all_tickers = list(set(target_tickers + baseline_tickers + ["SPY"]))
        print(f"Downloading price data for {len(all_tickers)} tickers...")
        data = yf.download(
            all_tickers,
            period="2y",
            auto_adjust=True,
            progress=False
        )['Close']
        fund_data = {}
    else:
        data, fund_data = fetch_data(target_tickers, baseline_tickers)
    
    # 3. Build Features & Dynamic Benchmarks
    dynamic_pe_benchmarks = None
    if is_full_scan and not args.skip_fundamentals:
        # We only compute dynamic P/E if we actually fetched the full S&P 500 universe
        sector_pes = {}
        for t, funds in fund_data.items():
            pe = funds.get("pe_ratio")
            if pe is not None and pe > 0:
                sect = funds.get("sector", sp500_sectors.get(t, "N/A"))
                sector_pes.setdefault(sect, []).append(pe)
        dynamic_pe_benchmarks = {s: np.median(pes) for s, pes in sector_pes.items()}
        print("Computed dynamic P/E sector medians from full scan.")
        
    all_scan_tickers = list(set(target_tickers + baseline_tickers))
    stocks_df = build_feature_table(all_scan_tickers, data, fund_data, sp500_sectors, dynamic_pe_benchmarks)
    
    # 4. Regression model
    from regression import get_trained_ols_model
    ols_model = get_trained_ols_model(all_scan_tickers, data, fund_data, sp500_sectors)
    features_list = ["pe_ratio", "debt_to_equity", "ma50_proximity", "ma200_proximity", "momentum_1y", "rsi", "annualized_vol", "beta"]

    # 5. Score
    profiles_to_run = list(RISK_LEVELS.keys()) if args.all_profiles else [args.profile]
    
    for profile in profiles_to_run:
        scored_df = generate_recommendations(stocks_df, user_profile=profile, top_n=len(stocks_df), pe_benchmarks=dynamic_pe_benchmarks)
        results = scored_df[scored_df['ticker'].isin(target_tickers)].copy()
        
        # Predict return for results using OLS regression model
        if len(results) > 0:
            X_pred = results[features_list].values
            y_pred, se_pred = ols_model.predict(X_pred)
            results["pred_return"] = y_pred
            results["pred_se"] = se_pred
            results["prob_positive"] = [ols_model.normal_cdf(y / se) * 100 for y, se in zip(y_pred, se_pred)]
            results["conf_lower"] = y_pred - 1.96 * se_pred
            results["conf_upper"] = y_pred + 1.96 * se_pred

        if is_full_scan:
            results = results.head(args.top)
            print("\n" + "="*80)
            print(f"BASAK BOSSMAN ENGINE — S&P 500 SCAN: TOP {args.top} ({profile.upper()})")
            print("="*80)
        else:
            print("\n" + "="*80)
            print(f"BASAK BOSSMAN ENGINE — QUICK SCORE: {profile.upper()}")
            print("="*80)
        
        display = results[[
            "ticker", "sector", "price", "ma_50", "ma_200",
            "momentum_1mo", "momentum_6mo", "momentum_1y",
            "rsi", "pe_ratio", "peg_ratio"
        ]].copy()
        
        display["momentum_1mo"] = display["momentum_1mo"].map(lambda x: f"{x*100:+.1f}%")
        display["momentum_6mo"] = display["momentum_6mo"].map(lambda x: f"{x*100:+.1f}%")
        display["momentum_1y"]  = display["momentum_1y"].map(lambda x: f"{x*100:+.1f}%")
        display["pe_ratio"]     = display["pe_ratio"].map(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        display["peg_ratio"]    = display["peg_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        if "pred_return" in results.columns:
            display["Exp Return"] = results["pred_return"].map(lambda x: f"{x*100:+.2f}%")
            display["95% CI"] = results.apply(lambda r: f"[{r['conf_lower']*100:+.1f}%, {r['conf_upper']*100:+.1f}%]", axis=1)
            display["Prob>0"] = results["prob_positive"].map(lambda x: f"{x:.1f}%")
            
        display["score"] = results["recommendation_score"].map(lambda x: f"{x:.4f}")
        
        # Show whether stock is above or below 50MA and 200MA
        def ma_label(price_val, ma_val):
            try:
                p, ma = float(price_val), float(ma_val)
                pct = (p - ma) / ma * 100
                return f"{pct:+.1f}%"
            except Exception:
                return "N/A"
        display["vs50MA"]  = display.apply(lambda row: ma_label(row["price"], row["ma_50"]), axis=1)
        display["vs200MA"] = display.apply(lambda row: ma_label(row["price"], row["ma_200"]), axis=1)
        
        display = display.rename(columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "price":  "Price",
            "momentum_1mo": "1mo",
            "momentum_6mo": "6mo",
            "momentum_1y":  "1yr",
            "rsi":          "RSI",
            "pe_ratio":     "P/E",
            "peg_ratio":    "PEG",
            "score":        "Score",
            "vs50MA":       "vs50MA",
            "vs200MA":      "vs200MA",
        })
        
        # Select and print specific columns
        cols_to_print = ["Ticker", "Price", "1yr", "RSI", "P/E", "Score", "vs50MA", "vs200MA"]
        if "Exp Return" in display.columns:
            cols_to_print += ["Exp Return", "95% CI", "Prob>0"]
            
        print(display[cols_to_print].to_string(index=False))
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
