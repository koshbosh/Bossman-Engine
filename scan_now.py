"""
======================================================================
BASAK BOSSMAN ENGINE: LIVE S&P 500 SCANNER
======================================================================
Fetches current market data for all S&P 500 stocks and runs the
scoring engine to surface today's top picks per risk profile.

Unlike the backtests (which loop over historical dates), this script
runs the model ONCE on live data — giving you actionable picks now.

Key difference from backtest:
- Uses REAL current P/E ratios from yfinance (not neutralised).
  This means the value factor is live for low/low-medium profiles.
- News sentiment still defaults to neutral (no live feed wired in).

Usage:
    python scan_now.py

    # Scan a specific profile only:
    python scan_now.py --profile low-medium

    # Change number of picks:
    python scan_now.py --top 10

Output:
- Prints ranked picks per profile to terminal
- Saves full scored universe to: scan_results_YYYYMMDD.csv
======================================================================
"""
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
import requests
import argparse
from io import StringIO
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from recommender import generate_recommendations, RISK_LEVELS, SECTOR_PE_BENCHMARKS

warnings.simplefilter(action='ignore', category=FutureWarning)

TOP_N = 10
PROFILES = list(RISK_LEVELS.keys())


# ─────────────────────────────────────────────
# STEP 1: GET S&P 500 TICKERS
# ─────────────────────────────────────────────

def get_sp500_tickers():
    print("Fetching S&P 500 ticker list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
    tickers = df['Symbol'].tolist()
    print(f"  Found {len(tickers)} tickers.\n")
    return tickers, sector_map


# ─────────────────────────────────────────────
# STEP 2: DOWNLOAD PRICE DATA
# ─────────────────────────────────────────────

def fetch_price_data(tickers):
    """
    Downloads 2 years of daily price data for all tickers + SPY.
    2 years gives us enough history for 1y momentum + beta calculation.
    """
    print("Downloading price data (this takes ~2-3 minutes)...")
    all_tickers = list(set(tickers + ["SPY"]))
    data = yf.download(
        all_tickers,
        period="2y",
        auto_adjust=True,
        progress=True
    )['Close']
    print(f"  Downloaded. Shape: {data.shape}\n")
    return data


# ─────────────────────────────────────────────
# STEP 3: FETCH FUNDAMENTALS (P/E, D/E)
# ─────────────────────────────────────────────

def fetch_single_fundamental(t):
    try:
        info = yf.Ticker(t).info
        return t, {
            "pe_ratio": info.get("trailingPE", None),
            "debt_to_equity": info.get("debtToEquity", 0) or 0,
            "ma_200": info.get("twoHundredDayAverage", None),
        }
    except Exception:
        return t, {"pe_ratio": None, "debt_to_equity": 0, "ma_200": None}


def fetch_fundamentals(tickers):
    """
    Fetches current P/E and D/E ratios from yfinance for each ticker.
    Uses multi-threading to speed up the fetching process significantly.
    """
    print("Fetching fundamentals (P/E, D/E) using multithreading...")
    print("  This should take less than a minute with concurrent requests.\n")

    fund_data = {}
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=25) as executor:
        future_to_ticker = {executor.submit(fetch_single_fundamental, t): t for t in tickers}
        
        completed = 0
        for future in as_completed(future_to_ticker):
            t, res = future.result()
            fund_data[t] = res
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  {completed}/{total} fetched...")

    print(f"\n  Fundamentals fetched for {len(fund_data)} tickers.\n")
    return fund_data


# ─────────────────────────────────────────────
# STEP 4: CALCULATE BETA
# ─────────────────────────────────────────────

def calculate_beta(stock_returns, market_returns):
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    var = np.var(m)
    return np.cov(s, m)[0][1] / var if var != 0 else 1.0


# ─────────────────────────────────────────────
# STEP 5: BUILD STOCK FEATURE TABLE
# ─────────────────────────────────────────────

def build_stock_features(tickers, data, fund_data, sector_map, pe_benchmarks=None):
    """
    For each ticker, computes all the features the scoring engine needs:
    MA proximity (50d/200d), momentum (1mo/3mo/6mo/1y), volatility, RSI,
    beta, P/E, D/E, sector.
    """
    if pe_benchmarks is None:
        pe_benchmarks = SECTOR_PE_BENCHMARKS
        
    print("Computing features for each stock...")

    spy_rets = data["SPY"].pct_change().dropna()
    records = []
    skipped = 0

    for t in tickers:
        if t not in data.columns:
            skipped += 1
            continue

        s = data[t].dropna()
        if len(s) < 252:   # need at least 1 year of data
            skipped += 1
            continue

        p_now = s.iloc[-1]
        s_rets = s.pct_change().dropna()

        # ── Momentum (still needed for gear-shift, penalties, sector bonus) ─
        m1mo  = (p_now / s.iloc[-21]  - 1) if len(s) >= 21  else 0
        m3mo  = (p_now / s.iloc[-63]  - 1) if len(s) >= 63  else 0
        m6mo  = (p_now / s.iloc[-126] - 1) if len(s) >= 126 else 0
        m1y   = (p_now / s.iloc[-252] - 1)

        # ── RSI (14-day) ──────────────────────────────────────────
        delta = s.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi   = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        # ── Volatility & Beta ─────────────────────────────────────
        ann_vol = s_rets.iloc[-252:].std() * np.sqrt(252)
        beta    = calculate_beta(s_rets.iloc[-252:], spy_rets)

        # ── Fundamentals ─────────────────────────────────────────
        sector = sector_map.get(t, "N/A")
        funds  = fund_data.get(t, {})
        pe     = funds.get("pe_ratio", None)

        # If P/E is missing or invalid, fall back to dynamic or static sector median
        if pe is None or pe <= 0 or pe != pe:  # NaN check
            pe = pe_benchmarks.get(sector, 23.5)

        # ── 200-day Moving Average ────────────────────────────────────
        funds  = fund_data.get(t, {})
        ma_200_val = funds.get("ma_200", None)
        if ma_200_val is None or (isinstance(ma_200_val, float) and np.isnan(ma_200_val)):
            ma_200_val = float(s.iloc[-200:].mean()) if len(s) >= 200 else float("nan")

        # ── 50-day Moving Average ─────────────────────────────────────
        ma_50_val = float(s.iloc[-50:].mean()) if len(s) >= 50 else float("nan")

        # ── MA Proximity Scores ──────────────────────────────────────
        from recommender import calculate_ma_proximity
        ma50_prox = calculate_ma_proximity(p_now, ma_50_val)
        ma200_prox = calculate_ma_proximity(p_now, ma_200_val)

        records.append({
            "ticker":         t,
            "sector":         sector,
            "price":          round(p_now, 2),
            "momentum_1mo":   m1mo,
            "momentum_3mo":   m3mo,
            "momentum_6mo":   m6mo,
            "momentum_1y":    m1y,
            "annualized_vol": ann_vol,
            "rsi":            round(rsi, 1),
            "beta":           round(beta, 2),
            "perf_1w":        (p_now / s.iloc[-5] - 1) if len(s) >= 5 else 0,
            "pe_ratio":       pe,
            "debt_to_equity": funds.get("debt_to_equity", 0),
            "ma_200":         round(ma_200_val, 2) if not np.isnan(ma_200_val) else None,
            "ma_50":          round(ma_50_val, 2) if not np.isnan(ma_50_val) else None,
            "ma50_proximity": ma50_prox,
            "ma200_proximity": ma200_prox,
        })

    df = pd.DataFrame(records)
    print(f"  Built features for {len(df)} stocks. ({skipped} skipped — insufficient data)\n")
    return df


# ─────────────────────────────────────────────
# STEP 6: SCORE & DISPLAY
# ─────────────────────────────────────────────

def run_scan(stocks_df, profiles, top_n, pe_benchmarks=None):
    results = {}
    for profile in profiles:
        recs = generate_recommendations(stocks_df, user_profile=profile, top_n=top_n, pe_benchmarks=pe_benchmarks)
        results[profile] = recs
    return results


def display_results(results, top_n):
    today = datetime.today().strftime('%B %d, %Y')

    print("\n" + "=" * 80)
    print(f"BASAK BOSSMAN ENGINE — LIVE SCAN RESULTS")
    print(f"Date   : {today}")
    print(f"Top N  : {top_n} per profile")
    print("=" * 80)

    for profile, df in results.items():
        print(f"\n{'─'*80}")
        print(f"  PROFILE: {profile.upper()}")
        print(f"{'─'*80}")

        display = df[[
            "ticker", "sector", "price", "ma_50", "ma_200",
            "momentum_1mo", "momentum_6mo", "momentum_1y",
            "rsi", "pe_ratio", "recommendation_score"
        ]].copy()

        display["momentum_1mo"] = display["momentum_1mo"].map(lambda x: f"{x*100:+.1f}%")
        display["momentum_6mo"] = display["momentum_6mo"].map(lambda x: f"{x*100:+.1f}%")
        display["momentum_1y"]  = display["momentum_1y"].map(lambda x: f"{x*100:+.1f}%")
        display["pe_ratio"]     = display["pe_ratio"].map(lambda x: f"{x:.1f}x")
        display["score"]        = df["recommendation_score"].map(lambda x: f"{x:.4f}")
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
        
        # OLS Predictions display if available
        if "pred_return" in df.columns:
            display["Exp Return"] = df["pred_return"].map(lambda x: f"{x*100:+.2f}%")
            display["95% CI"] = df.apply(lambda r: f"[{r['conf_lower']*100:+.1f}%, {r['conf_upper']*100:+.1f}%]", axis=1)
            display["Prob>0"] = df["prob_positive"].map(lambda x: f"{x:.1f}%")
            
        display = display.drop(columns=["recommendation_score", "ma_200", "ma_50"])
        
        rename_dict = {
            "ticker": "Ticker",
            "sector": "Sector",
            "price":  "Price",
            "momentum_1mo": "1mo",
            "momentum_6mo": "6mo",
            "momentum_1y":  "1yr",
            "rsi":          "RSI",
            "pe_ratio":     "P/E",
            "score":        "Score",
            "vs50MA":       "vs50MA",
            "vs200MA":      "vs200MA",
        }
        display = display.rename(columns=rename_dict)
        display.index = range(1, len(display) + 1)
        
        # Reorder columns to display statistical forecast values clearly
        cols_to_print = ["Ticker", "Sector", "Price", "1yr", "RSI", "P/E", "Score", "vs50MA", "vs200MA"]
        if "Exp Return" in display.columns:
            cols_to_print += ["Exp Return", "95% CI", "Prob>0"]
            
        print(display[cols_to_print].to_string())

    print("\n" + "=" * 80)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Basak Bossman Live Scanner")
    parser.add_argument("--profile", type=str, default=None,
                        help="Run a single profile only (e.g. --profile medium)")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help="Number of picks to show (default: 10)")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Skip P/E/D/E fetch (faster, uses sector defaults)")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else PROFILES
    top_n    = args.top

    # ── Step 1: Universe ────────────────────────────────────────
    tickers, sector_map = get_sp500_tickers()

    # ── Step 2: Prices ──────────────────────────────────────────
    data = fetch_price_data(tickers)

    # ── Step 3: Fundamentals & Dynamic Benchmarks ───────────────
    dynamic_pe_benchmarks = None
    if args.skip_fundamentals:
        print("Skipping fundamentals fetch (using sector defaults for P/E).\n")
        fund_data = {t: {"pe_ratio": None, "debt_to_equity": 0} for t in tickers}
    else:
        fund_data = fetch_fundamentals(tickers)
        
        # Compute dynamic P/E medians by sector
        sector_pes = {}
        for t in tickers:
            pe = fund_data.get(t, {}).get("pe_ratio")
            if pe is not None and pe > 0:
                sect = sector_map.get(t, "N/A")
                sector_pes.setdefault(sect, []).append(pe)
        
        dynamic_pe_benchmarks = {s: np.median(pes) for s, pes in sector_pes.items()}
        print("  Computed dynamic P/E sector medians.")

    # ── Step 4: Feature table ───────────────────────────────────
    stocks_df = build_stock_features(tickers, data, fund_data, sector_map, dynamic_pe_benchmarks)

    # ── Step 5: Score ───────────────────────────────────────────
    print(f"Scoring {len(stocks_df)} stocks across profiles: {profiles}...")
    results = run_scan(stocks_df, profiles, top_n, dynamic_pe_benchmarks)

    # ── Step 5.5: Run Return Prediction Regression ──────────────
    from regression import get_trained_ols_model
    ols_model = get_trained_ols_model(tickers, data, fund_data, sector_map)

    features_list = ["pe_ratio", "debt_to_equity", "ma50_proximity", "ma200_proximity", "momentum_1y", "rsi", "annualized_vol", "beta"]
    
    # Predict for each profile's top N picks
    for profile in profiles:
        df = results[profile]
        if len(df) > 0:
            X_pred = df[features_list].values
            y_pred, se_pred = ols_model.predict(X_pred)
            df["pred_return"] = y_pred
            df["pred_se"] = se_pred
            df["prob_positive"] = [ols_model.normal_cdf(y / se) * 100 for y, se in zip(y_pred, se_pred)]
            df["conf_lower"] = y_pred - 1.96 * se_pred
            df["conf_upper"] = y_pred + 1.96 * se_pred

    # Predict for the entire universe (to save in the CSV)
    if len(stocks_df) > 0:
        X_all = stocks_df[features_list].values
        y_all, se_all = ols_model.predict(X_all)
        stocks_df["pred_return"] = y_all
        stocks_df["pred_se"] = se_all
        stocks_df["prob_positive"] = [ols_model.normal_cdf(y / se) * 100 for y, se in zip(y_all, se_all)]
        stocks_df["conf_lower"] = y_all - 1.96 * se_all
        stocks_df["conf_upper"] = y_all + 1.96 * se_all

    # ── Step 6: Display ─────────────────────────────────────────
    display_results(results, top_n)

    # ── Step 7: Save full scored universe ───────────────────────
    date_str = datetime.today().strftime('%Y%m%d')
    output_file = f"scan_results_{date_str}.csv"
    stocks_df.to_csv(output_file, index=False)
    print(f"Full scored universe saved to: {output_file}\n")


if __name__ == "__main__":
    main()
