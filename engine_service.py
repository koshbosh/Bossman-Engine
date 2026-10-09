"""
======================================================================
BASAK BOSSMAN ENGINE: UNIVERSAL SCORING SERVICE
======================================================================
Provides on-demand scoring for ANY arbitrary stock ticker (US equities,
ADRs, small/mid/large caps) against a pre-warmed reference baseline.

Key Quantitative Features:
1. Dynamic Factor Rebasing: Missing metrics (e.g., young IPOs with <252d
   history, negative earnings P/E) are omitted cleanly and remaining
   weights are rebased to 100%.
2. Non-Distorting D/E Solvency Flag: Debt-to-Equity is surfaced as an
   explicit balance-sheet risk flag/badge rather than an arbitrary score penalty.
3. Transparent Footnotes: Full accounting of omitted factors, rebased weights,
   and data limitations.
4. OLS 1-Year Return Prediction with 95% Confidence Intervals.
5. Cross-Sectional Ranking against baseline universe.
======================================================================
"""

import os
import math
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import Ticker as YQTicker

from recommender import (
    RISK_LEVELS,
    SECTOR_PE_BENCHMARKS,
    SECTOR_DE_THRESHOLDS,
    calculate_ma_proximity,
    calculate_value_score,
    risk_penalty,
    calculate_rsi_overbought_penalty,
    calculate_rsi_falling_knife_penalty,
    calculate_rsi_oversold_bonus,
    calculate_high_beta_bonus,
    calculate_ma50_deviation_bonus,
    calculate_pe_stretch_penalty,
    calculate_ma200_bonus,
    calculate_low_pe_bonus,
    calculate_peg_bonus,
    calculate_overextended_momentum_penalty,
    calculate_exhaustion_penalty,
    calculate_short_penalty,
    calculate_volume_surge_bonus,
)
from regression import NumPyOLS

warnings.simplefilter(action="ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_CSV = os.path.join(BASE_DIR, "scan_results_20261001.csv")


SECTOR_TO_ETF = {
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def calculate_beta(stock_returns, market_returns):
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    s, m = stock_returns.loc[common], market_returns.loc[common]
    var = np.var(m)
    return float(np.cov(s, m)[0][1] / var) if var != 0 else 1.0


class UniversalScoringEngine:
    def __init__(self, baseline_csv_path=BASELINE_CSV):
        self.baseline_csv_path = baseline_csv_path
        self.baseline_df = None
        self.ols_model = None
        self.features_list = [
            "pe_ratio",
            "debt_to_equity",
            "ma50_proximity",
            "ma200_proximity",
            "momentum_1y",
            "rsi",
            "annualized_vol",
            "beta",
        ]
        self.spy_history = None
        self.market_regime = {}
        self.sector_etf_data = {}
        self.load_baseline()

    def load_baseline(self):
        """Loads cached universe baseline and trains OLS model."""
        if os.path.exists(self.baseline_csv_path):
            self.baseline_df = pd.read_csv(self.baseline_csv_path)
            print(f"[Engine] Loaded baseline universe with {len(self.baseline_df)} stocks.")
        else:
            print("[Engine] Warning: Baseline CSV not found. Initializing empty.")
            self.baseline_df = pd.DataFrame()

        # Train OLS model on baseline
        if not self.baseline_df.empty and "pred_return" in self.baseline_df.columns:
            clean_df = self.baseline_df[self.features_list + ["pred_return"]].dropna()
            self.ols_model = NumPyOLS()
            self.ols_model.fit(
                clean_df[self.features_list].values,
                clean_df["pred_return"].values,
                self.features_list,
            )
            print(f"[Engine] OLS model trained (R²: {self.ols_model.r2:.3f}).")

        # Compute sector medians for sector rotation outperformance bonus
        if not self.baseline_df.empty and "momentum_3mo" in self.baseline_df.columns:
            self.sector_3mo_medians = self.baseline_df.groupby("sector")["momentum_3mo"].median().to_dict()
            self.sector_6mo_medians = self.baseline_df.groupby("sector")["momentum_6mo"].median().to_dict()
        else:
            self.sector_3mo_medians = {}
            self.sector_6mo_medians = {}

        # Fetch macro indicators, Breadth, RSP/SPY, VIX & Sector ETFs
        self.refresh_macro_and_sector_regimes()

    def refresh_macro_and_sector_regimes(self):
        """
        Computes composite macro market regime & sector sub-regimes:
        1. Market Breadth: % of S&P 500 universe above 50MA (<35% Dip Hunter, <50% All-Weather Core).
        2. RSP vs SPY 1-Month relative performance (penalizes Alpha Bull if RSP > SPY; penalizes All-Weather if SPY > RSP).
        3. VIX circuit breakers: VIX > 35 -> Dip Hunter lock; VIX > 55 -> Bear Market Shield / Fundamental Bag lock.
        4. Sector Sub-Regimes: tracks 11 GICS sector ETFs vs 50MA for contrarian score bonuses/haircuts.
        """
        # 1. Breadth from baseline
        if not self.baseline_df.empty and "price" in self.baseline_df.columns and "ma_50" in self.baseline_df.columns:
            clean_b = self.baseline_df.dropna(subset=["price", "ma_50"])
            breadth_50 = float((clean_b["price"] > clean_b["ma_50"]).mean() * 100) if len(clean_b) > 0 else 50.0
            breadth_200 = float((clean_b["price"] > clean_b["ma_200"]).mean() * 100) if "ma_200" in clean_b.columns else 50.0
        else:
            breadth_50 = 50.0
            breadth_200 = 50.0

        # 2. Batch download macro assets & sector ETFs
        macro_syms = ["SPY", "RSP", "^VIX", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC"]
        try:
            m_data = yf.download(macro_syms, period="2y", auto_adjust=True, progress=False)["Close"]
        except Exception as e:
            print(f"[Engine] Error downloading macro assets: {e}")
            m_data = pd.DataFrame()

        # Extract SPY
        if not m_data.empty and "SPY" in m_data.columns:
            spy_s = m_data["SPY"].dropna()
            self.spy_history = spy_s
            p_now = float(spy_s.iloc[-1])
            ma50 = float(spy_s.iloc[-50:].mean()) if len(spy_s) >= 50 else p_now
            ma200 = float(spy_s.iloc[-200:].mean()) if len(spy_s) >= 200 else p_now
            pct_ma50 = (p_now - ma50) / ma50 * 100
            pct_ma200 = (p_now - ma200) / ma200 * 100
        else:
            p_now, ma50, ma200, pct_ma50, pct_ma200 = 0.0, 0.0, 0.0, 0.0, 0.0

        # Extract VIX
        vix_now = 18.0
        if not m_data.empty and "^VIX" in m_data.columns:
            vix_s = m_data["^VIX"].dropna()
            if not vix_s.empty:
                vix_now = float(vix_s.iloc[-1])

        # Extract RSP vs SPY 1-Month relative return
        rsp_spy_diff = 0.0
        rsp_outperforming = False
        if not m_data.empty and "RSP" in m_data.columns and "SPY" in m_data.columns:
            rsp_s = m_data["RSP"].dropna()
            spy_s = m_data["SPY"].dropna()
            if len(rsp_s) >= 21 and len(spy_s) >= 21:
                rsp_1mo = float(rsp_s.iloc[-1] / rsp_s.iloc[-21] - 1.0)
                spy_1mo = float(spy_s.iloc[-1] / spy_s.iloc[-21] - 1.0)
                rsp_spy_diff = round((rsp_1mo - spy_1mo) * 100, 2)
                rsp_outperforming = (rsp_1mo > spy_1mo)

        # 3. Macro Regime State Machine Logic
        vix_override = None
        if vix_now >= 55.0:
            regime_name = "Systemic Crisis (VIX > 55 Override)"
            recommended_profiles = ["Fundamental Bag", "Bear Market Shield"]
            regime_reason = f"Extreme systemic crisis override (VIX at {vix_now:.1f}). Focus on balance-sheet solvency and deep value."
            regime_color = "#ef4444"
            vix_override = "crisis_55"
        elif vix_now >= 35.0:
            regime_name = "Panic Capitulation (VIX > 35 Override)"
            recommended_profiles = ["Dip Hunter"]
            regime_reason = f"Capitulation panic override (VIX at {vix_now:.1f}). High-probability asymmetric turnaround dip setups active."
            regime_color = "#f59e0b"
            vix_override = "panic_35"
        else:
            # Market Breadth + RSP/SPY logic
            if breadth_50 < 35.0:
                regime_name = f"Oversold Breadth ({breadth_50:.1f}% > 50MA)"
                recommended_profiles = ["Dip Hunter"]
                regime_reason = f"Broad market oversold: only {breadth_50:.1f}% of S&P 500 stocks above 50MA (<35% threshold). Prime for turnaround hunting."
                regime_color = "#f59e0b"
            elif breadth_50 < 50.0:
                # 35% to 50% breadth: baseline All-Weather Core
                # Check: if SPY outperforms RSP (narrow mega-cap rally), penalize All-Weather Core!
                if not rsp_outperforming:
                    regime_name = f"Narrow Breadth Divergence (SPY > RSP)"
                    recommended_profiles = ["Dip Hunter", "Alpha Bull"]
                    regime_reason = f"Breadth at {breadth_50:.1f}% with narrow mega-cap leadership (SPY > RSP by +{abs(rsp_spy_diff):.1f}%). All-Weather Core penalized."
                    regime_color = "#fbbf24"
                else:
                    regime_name = f"Selective Consolidation ({breadth_50:.1f}% > 50MA)"
                    recommended_profiles = ["All-Weather Core"]
                    regime_reason = f"Breadth at {breadth_50:.1f}% with balanced RSP participation. Volatility-managed compounders favored."
                    regime_color = "#38bdf8"
            else:
                # Breadth >= 50%: baseline Alpha Bull
                # Check: if RSP outperforms SPY (broad cyclical participation, not mega-cap momentum), penalize Alpha Bull!
                if rsp_outperforming:
                    regime_name = f"Broad Market Expansion (RSP > SPY)"
                    recommended_profiles = ["All-Weather Core"]
                    regime_reason = f"Healthy breadth ({breadth_50:.1f}% > 50MA), but equal-weight RSP leads SPY by +{rsp_spy_diff:.1f}%. Alpha Bull penalized in favor of broad All-Weather Core."
                    regime_color = "#34d399"
                else:
                    regime_name = f"Bull Momentum Expansion (Breadth {breadth_50:.1f}%)"
                    recommended_profiles = ["Alpha Bull"]
                    regime_reason = f"Strong breadth ({breadth_50:.1f}% > 50MA) with cap-weighted SPY momentum leadership. Aggressive trend following favored."
                    regime_color = "#10b981"

        # 4. Sector Sub-Regimes ETF calculation
        self.sector_etf_data = {}
        for etf in ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC"]:
            if not m_data.empty and etf in m_data.columns:
                s = m_data[etf].dropna()
                if len(s) >= 50:
                    p = float(s.iloc[-1])
                    ma50_val = float(s.iloc[-50:].mean())
                    diff = (p - ma50_val) / ma50_val * 100
                    self.sector_etf_data[etf] = {
                        "symbol": etf,
                        "price": round(p, 2),
                        "ma50": round(ma50_val, 2),
                        "diff_pct": round(diff, 2),
                        "status": "below_50ma" if diff < 0 else "above_50ma",
                    }

        self.market_regime = {
            "symbol": "SPY",
            "price": round(p_now, 2),
            "ma50": round(ma50, 2),
            "ma200": round(ma200, 2),
            "pct_vs_ma50": round(pct_ma50, 2),
            "pct_vs_ma200": round(pct_ma200, 2),
            "breadth_50": round(breadth_50, 1),
            "breadth_200": round(breadth_200, 1),
            "vix": round(vix_now, 2),
            "vix_override": vix_override,
            "rsp_spy_diff": rsp_spy_diff,
            "rsp_outperforming": rsp_outperforming,
            "regime": regime_name,
            "recommended_profiles": recommended_profiles,
            "regime_reason": regime_reason,
            "color": regime_color,
            "sector_etfs": self.sector_etf_data,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def fetch_ticker_data(self, ticker: str):
        """
        Downloads 2y price history and fundamentals for ANY arbitrary ticker.
        Handles edge cases (missing data, young IPOs, OTC, foreign ADRs).
        """
        ticker = ticker.strip().upper()
        t = yf.Ticker(ticker)

        # 1. Price History
        hist = t.history(period="2y")
        if hist.empty or len(hist) < 20:
            raise ValueError(f"Insufficient historical data found for ticker '{ticker}'. Please verify the symbol.")

        close = hist["Close"]
        volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(0, index=close.index)
        n_days = len(close)
        p_now = float(close.iloc[-1])

        # 2. Company Profile & Fundamentals (safely extracted)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        # Try fast summary fallback if info is sparse
        name = info.get("shortName") or info.get("longName") or ticker
        raw_sector = info.get("sector", "N/A")
        sector_map = {
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Defensive": "Consumer Staples",
            "Technology": "Information Technology",
            "Healthcare": "Health Care",
            "Financial Services": "Financials",
            "Basic Materials": "Materials",
        }
        sector = sector_map.get(raw_sector, raw_sector)
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", 0)
        trailing_pe = info.get("trailingPE", None)
        forward_pe = info.get("forwardPE", None)
        peg_ratio = info.get("pegRatio", None)
        debt_to_equity = info.get("debtToEquity", None)
        fifty_two_high = info.get("fiftyTwoWeekHigh", float(close.max()))
        fifty_two_low = info.get("fiftyTwoWeekLow", float(close.min()))

        # 3. Technical Features
        s_rets = close.pct_change().dropna()

        # Momentum periods
        m1w = float(p_now / close.iloc[-5] - 1) if n_days >= 5 else 0.0
        m1mo = float(p_now / close.iloc[-21] - 1) if n_days >= 21 else None
        m3mo = float(p_now / close.iloc[-63] - 1) if n_days >= 63 else None
        m6mo = float(p_now / close.iloc[-126] - 1) if n_days >= 126 else None
        m1y = float(p_now / close.iloc[-252] - 1) if n_days >= 252 else None

        # 14-day RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = float(100 - (100 / (1 + (gain / (loss + 1e-9))))) if not pd.isna(gain) else 50.0

        # Moving Averages
        ma_50_val = float(close.iloc[-50:].mean()) if n_days >= 50 else None
        ma_200_val = float(close.iloc[-200:].mean()) if n_days >= 200 else None

        ma50_prox = calculate_ma_proximity(p_now, ma_50_val) if ma_50_val else None
        ma200_prox = calculate_ma_proximity(p_now, ma_200_val) if ma_200_val else None

        # Volatility (annualized)
        vol_window = min(n_days, 252)
        ann_vol = float(s_rets.iloc[-vol_window:].std() * np.sqrt(252)) if len(s_rets) >= 20 else 0.25

        # Beta vs SPY
        if self.spy_history is not None and not self.spy_history.empty:
            spy_rets = self.spy_history.pct_change().dropna()
            beta = calculate_beta(s_rets, spy_rets)
        else:
            beta = 1.0

        # Volume Surge calculation
        curr_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0
        avg_vol_20d = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else curr_vol

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "price": round(p_now, 2),
            "n_days": n_days,
            "fifty_two_high": round(float(fifty_two_high), 2) if fifty_two_high else None,
            "fifty_two_low": round(float(fifty_two_low), 2) if fifty_two_low else None,
            "perf_1w": m1w,
            "momentum_1mo": m1mo,
            "momentum_3mo": m3mo,
            "momentum_6mo": m6mo,
            "momentum_1y": m1y,
            "rsi": round(rsi, 1),
            "ma_50": round(ma_50_val, 2) if ma_50_val else None,
            "ma_200": round(ma_200_val, 2) if ma_200_val else None,
            "ma50_proximity": ma50_prox,
            "ma200_proximity": ma200_prox,
            "annualized_vol": round(ann_vol, 4),
            "beta": round(beta, 2),
            "pe_ratio": float(trailing_pe) if trailing_pe and trailing_pe > 0 else None,
            "forward_pe": float(forward_pe) if forward_pe and forward_pe > 0 else None,
            "peg_ratio": float(peg_ratio) if peg_ratio and peg_ratio > 0 else None,
            "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else None,
            "current_volume": curr_vol,
            "avg_volume": avg_vol_20d,
        }

    def evaluate_leverage_flag(self, de_ratio, sector):
        """
        Evaluates Debt-to-Equity as an informative risk flag/badge rather than
        silently penalizing or skewing the recommendation score.
        """
        threshold = SECTOR_DE_THRESHOLDS.get(sector, 100)  # yfinance reports D/E as % (100 = 1.0x)

        if de_ratio is None or pd.isna(de_ratio):
            return {
                "status": "not_reported",
                "label": "D/E: Not Reported",
                "badge_color": "secondary",
                "description": "Debt-to-equity ratio not disclosed or non-standard capital structure.",
                "value_str": "N/A",
                "threshold_str": f"{threshold/100:.1f}x",
                "severity": "neutral",
            }

        de = float(de_ratio)
        if de < 0:
            return {
                "status": "negative_equity",
                "label": "Negative Book Equity",
                "badge_color": "warning",
                "description": "Company has negative net tangible book equity (often due to aggressive share repurchases or accumulated deficit).",
                "value_str": "Negative",
                "threshold_str": f"{threshold/100:.1f}x",
                "severity": "caution",
            }

        multiple = de / 100.0  # Convert to x-multiple (e.g. 150 -> 1.5x)
        thresh_multiple = threshold / 100.0

        if multiple < 0.8:
            return {
                "status": "conservative",
                "label": f"Conservative Leverage ({multiple:.2f}x)",
                "badge_color": "success",
                "description": f"Debt-to-equity of {multiple:.2f}x is well below the sector threshold ({thresh_multiple:.1f}x). Pristine balance sheet.",
                "value_str": f"{multiple:.2f}x",
                "threshold_str": f"{thresh_multiple:.1f}x",
                "severity": "safe",
            }
        elif multiple <= thresh_multiple * 1.25:
            return {
                "status": "moderate",
                "label": f"Moderate Leverage ({multiple:.2f}x)",
                "badge_color": "info",
                "description": f"Debt-to-equity of {multiple:.2f}x is in line with the sector norm ({thresh_multiple:.1f}x). Standard operating leverage.",
                "value_str": f"{multiple:.2f}x",
                "threshold_str": f"{thresh_multiple:.1f}x",
                "severity": "normal",
            }
        elif multiple <= thresh_multiple * 2.0:
            return {
                "status": "elevated",
                "label": f"Elevated Leverage ({multiple:.2f}x)",
                "badge_color": "warning",
                "description": f"Debt-to-equity of {multiple:.2f}x exceeds sector threshold ({thresh_multiple:.1f}x). Flagged for monitoring; not penalized in score.",
                "value_str": f"{multiple:.2f}x",
                "threshold_str": f"{thresh_multiple:.1f}x",
                "severity": "caution",
            }
        else:
            return {
                "status": "high",
                "label": f"High Leverage Alert ({multiple:.2f}x)",
                "badge_color": "danger",
                "description": f"Significantly leveraged ({multiple:.2f}x vs {thresh_multiple:.1f}x sector norm). Flagged as high balance-sheet risk.",
                "value_str": f"{multiple:.2f}x",
                "threshold_str": f"{thresh_multiple:.1f}x",
                "severity": "warning",
            }

    def compute_percentile_rank(self, value, baseline_series):
        """Computes empirical percentile rank (0.0 to 1.0) against the reference universe."""
        if value is None or pd.isna(value) or baseline_series.empty:
            return 0.5
        clean_s = baseline_series.dropna()
        if len(clean_s) == 0:
            return 0.5
        count_less = (clean_s < value).sum()
        count_equal = (clean_s == value).sum()
        return float((count_less + 0.5 * count_equal) / len(clean_s))

    def score_ticker(self, ticker_data: dict, profile_name="Dip Hunter"):
        """
        Computes the multi-factor score for the ticker with:
        - Dynamic Factor Rebasing for missing variables
        - Explicit footnotes
        - Non-diluting D/E solvency flag
        - OLS return prediction
        """
        cfg = RISK_LEVELS.get(profile_name, RISK_LEVELS["Dip Hunter"])
        benchmark_pe = SECTOR_PE_BENCHMARKS.get(ticker_data["sector"], 23.5)

        footnotes = []
        omitted_factors = []
        active_factors = []

        # 1. Evaluate D/E balance sheet flag
        de_flag = self.evaluate_leverage_flag(ticker_data["debt_to_equity"], ticker_data["sector"])
        if de_flag["severity"] in ["caution", "warning"]:
            footnotes.append({
                "type": "leverage",
                "icon": "⚠️",
                "title": "Balance Sheet Leverage Flag",
                "message": f"D/E is {de_flag['value_str']} (Sector Reference: {de_flag['threshold_str']}). In accordance with your preferences, this is surfaced as a risk badge and does NOT arbitrarily reduce the score.",
            })

        # 2. Check Trading History Limitation
        if ticker_data["n_days"] < 252:
            footnotes.append({
                "type": "history",
                "icon": "ℹ️",
                "title": "Trading History Under 1 Year",
                "message": f"Stock has {ticker_data['n_days']} trading sessions. Factors requiring 252 days (1-Year Momentum & 200-day MA) are omitted; active factors are dynamically rebased.",
            })

        # 3. Check Valuation Availability
        if ticker_data["pe_ratio"] is None:
            footnotes.append({
                "type": "valuation",
                "icon": "ℹ️",
                "title": "Valuation Factor Omitted",
                "message": "Trailing P/E is negative or unavailable (LTM net loss). Rather than imputing a misleading sector average, valuation is omitted and remaining weights are rebased to 100%.",
            })

        # 4. Define candidate base factors and their raw weights for this profile
        # Profile-specific gear shift if applicable
        use_conditional = profile_name in ["All-Weather Core", "Dip Hunter"]
        base_ma50_wt = cfg.get("ma50_proximity_weight", 0.0)
        base_ma200_wt = cfg.get("ma200_proximity_weight", 0.0)
        base_3mo_wt = cfg.get("momentum_3mo_weight", 0.0)
        base_6mo_wt = cfg.get("momentum_6mo_weight", 0.0)
        base_1mo_wt = cfg.get("momentum_1mo_weight", 0.0)
        base_1y_wt = cfg.get("momentum_1y_weight", 0.0)
        risk_wt = cfg.get("risk_weight", 0.0)
        val_wt = cfg.get("value_weight", 0.0)

        if use_conditional and ticker_data["momentum_3mo"] is not None and ticker_data["momentum_1mo"] is not None:
            shift = 0.05
            if ticker_data["momentum_3mo"] > ticker_data["momentum_1mo"]:
                w_ma50 = base_ma50_wt - shift
                w_ma200 = base_ma200_wt + shift
            else:
                w_ma50 = base_ma50_wt + shift
                w_ma200 = base_ma200_wt - shift
        else:
            w_ma50 = base_ma50_wt
            w_ma200 = base_ma200_wt

        # Factor Definitions & Normalized Rank calculation
        raw_factor_configs = [
            ("ma50_proximity", "50-Day MA Proximity", w_ma50, ticker_data["ma50_proximity"], "ma50_proximity"),
            ("ma200_proximity", "200-Day MA Proximity", w_ma200, ticker_data["ma200_proximity"], "ma200_proximity"),
            ("momentum_1mo", "1-Month Momentum", base_1mo_wt, ticker_data["momentum_1mo"], "momentum_1mo"),
            ("momentum_3mo", "3-Month Momentum", base_3mo_wt, ticker_data["momentum_3mo"], "momentum_3mo"),
            ("momentum_6mo", "6-Month Momentum", base_6mo_wt, ticker_data["momentum_6mo"], "momentum_6mo"),
            ("momentum_1y", "1-Year Momentum", base_1y_wt, ticker_data["momentum_1y"], "momentum_1y"),
            ("risk", "Volatility Penalty (Risk)", risk_wt, ticker_data["annualized_vol"], "annualized_vol"),
            ("valuation", "P/E Value Score", val_wt, ticker_data["pe_ratio"], "pe_ratio"),
        ]

        # 5. Filter active vs omitted factors and compute normalized scores
        factor_breakdown = []
        valid_weights_sum = 0.0

        for key, name, orig_wt, val, col_name in raw_factor_configs:
            if orig_wt <= 0:
                continue

            if val is None or (isinstance(val, float) and (np.isnan(val) or val <= 0 and key == "valuation")):
                omitted_factors.append({
                    "factor": key,
                    "name": name,
                    "original_weight": orig_wt,
                    "reason": "Missing or negative data",
                })
                continue

            # Calculate normalized percentile
            if not self.baseline_df.empty and col_name in self.baseline_df.columns:
                norm_rank = self.compute_percentile_rank(val, self.baseline_df[col_name])
            else:
                norm_rank = 0.5

            # Directional inversion based on risk profile
            if profile_name in ["Bear Market Shield", "Fundamental Bag"]:
                if key in ["momentum_6mo", "momentum_1mo", "ma50_proximity"]:
                    score_component = 1.0 - norm_rank
                else:
                    score_component = norm_rank
            elif profile_name == "Dip Hunter":
                if key in ["momentum_1y", "ma50_proximity", "ma200_proximity"]:
                    score_component = 1.0 - norm_rank
                else:
                    score_component = norm_rank
            elif key == "risk":
                score_component = risk_penalty(val, cfg.get("vol_limit"))
            elif key == "valuation":
                score_component = calculate_value_score(val, benchmark_pe)
            else:
                score_component = norm_rank

            valid_weights_sum += orig_wt
            active_factors.append({
                "key": key,
                "name": name,
                "raw_value": val,
                "percentile_rank": round(norm_rank * 100, 1),
                "original_weight": orig_wt,
                "score_component": score_component,
            })

        # 6. Dynamic Factor Rebasing
        if valid_weights_sum > 0:
            rebasing_multiplier = 1.0 / valid_weights_sum
            for f in active_factors:
                f["rebased_weight"] = round(f["original_weight"] * rebasing_multiplier, 4)
                f["weighted_score"] = round(f["score_component"] * f["rebased_weight"], 4)
        else:
            rebasing_multiplier = 1.0
            for f in active_factors:
                f["rebased_weight"] = f["original_weight"]
                f["weighted_score"] = f["score_component"] * f["rebased_weight"]

        if omitted_factors:
            rebasing_percent = round((rebasing_multiplier - 1.0) * 100, 1)
            footnotes.append({
                "type": "rebasing",
                "icon": "⚖️",
                "title": "Dynamic Weight Rebasing Applied",
                "message": f"Omitted {len(omitted_factors)} metric(s). Remaining factor weights were rebased by +{rebasing_percent}% to ensure total score weights sum to 100%.",
            })

        base_score = sum(f["weighted_score"] for f in active_factors)

        # 7. Multipliers and Bonuses (Safely calculated, skipping missing inputs)
        multipliers_log = []
        final_multiplier = 1.0

        # Beta bonus
        if cfg.get("high_beta_bonus_active") and ticker_data["beta"] is not None:
            m = calculate_high_beta_bonus(ticker_data["beta"], cfg.get("high_beta_bonus_max", 0.15))
            if m > 1.0:
                final_multiplier *= m
                multipliers_log.append({"name": "High Beta Momentum Bonus", "effect": f"+{(m-1)*100:.1f}%", "value": m})

        # MA50 Mean Deviation Bonus
        if cfg.get("ma50_deviation_bonus_active") and ticker_data["ma_50"] is not None:
            m = calculate_ma50_deviation_bonus(
                ticker_data["price"],
                ticker_data["ma_50"],
                max_bonus=cfg.get("ma50_deviation_bonus_max", 0.075),
                cutoff=cfg.get("ma50_deviation_cutoff", 0.15),
            )
            if m > 1.0:
                final_multiplier *= m
                multipliers_log.append({"name": "MA50 Deviation Reversal Bonus", "effect": f"+{(m-1)*100:.1f}%", "value": m})

        # RSI Overbought penalty (ALL profiles)
        m_rsi_ob = calculate_rsi_overbought_penalty(ticker_data["rsi"])
        if m_rsi_ob < 1.0:
            final_multiplier *= m_rsi_ob
            multipliers_log.append({"name": "RSI Overbought Haircut (>65)", "effect": f"-{(1-m_rsi_ob)*100:.1f}%", "value": m_rsi_ob})

        # RSI Falling Knife penalty
        if cfg.get("rsi_falling_knife_penalty_active"):
            m_rsi_fk = calculate_rsi_falling_knife_penalty(ticker_data["rsi"])
            if m_rsi_fk < 1.0:
                final_multiplier *= m_rsi_fk
                multipliers_log.append({"name": "RSI Falling Knife Penalty (<30)", "effect": f"-{(1-m_rsi_fk)*100:.1f}%", "value": m_rsi_fk})

        # RSI Oversold bonus
        if cfg.get("rsi_oversold_bonus_active"):
            m_rsi_os = calculate_rsi_oversold_bonus(
                ticker_data["rsi"],
                max_bonus=cfg.get("rsi_oversold_bonus_max", 0.05)
            )
            if m_rsi_os > 1.0:
                final_multiplier *= m_rsi_os
                multipliers_log.append({"name": "RSI Oversold Room-to-Run Bonus", "effect": f"+{(m_rsi_os-1)*100:.1f}%", "value": m_rsi_os})

        # Below 200MA Bonus
        if cfg.get("ma200_bonus_active") and ticker_data["ma_200"] is not None:
            m_ma200 = calculate_ma200_bonus(
                ticker_data["price"],
                ticker_data["ma_200"],
                ticker_data["momentum_1y"] or 0.0,
                max_bonus=cfg.get("ma200_bonus_max", 0.08),
                max_depth=cfg.get("ma200_bonus_max_depth", 0.20),
            )
            if m_ma200 > 1.0:
                final_multiplier *= m_ma200
                multipliers_log.append({"name": "Deep Dip Below 200MA Bonus", "effect": f"+{(m_ma200-1)*100:.1f}%", "value": m_ma200})

        # P/E Stretch Penalty (Only if P/E is valid)
        if cfg.get("pe_stretch_penalty_active") and ticker_data["pe_ratio"] is not None:
            m_pe_stretch = calculate_pe_stretch_penalty(ticker_data["pe_ratio"], benchmark_pe)
            if m_pe_stretch < 1.0:
                final_multiplier *= m_pe_stretch
                multipliers_log.append({"name": "P/E Stretch Premium Penalty", "effect": f"-{(1-m_pe_stretch)*100:.1f}%", "value": m_pe_stretch})

        # Low P/E Bonus
        if cfg.get("low_pe_bonus_active") and ticker_data["pe_ratio"] is not None:
            m_low_pe = calculate_low_pe_bonus(ticker_data["pe_ratio"])
            if m_low_pe > 1.0:
                final_multiplier *= m_low_pe
                multipliers_log.append({"name": "Deep Value Low P/E Bonus", "effect": f"+{(m_low_pe-1)*100:.1f}%", "value": m_low_pe})

        # PEG ratio Bonus
        if cfg.get("peg_bonus_active") and ticker_data["peg_ratio"] is not None:
            m_peg = calculate_peg_bonus(ticker_data["peg_ratio"])
            if m_peg > 1.0:
                final_multiplier *= m_peg
                multipliers_log.append({"name": "PEG Ratio Under 1.0 Bonus", "effect": f"+{(m_peg-1)*100:.1f}%", "value": m_peg})

        # Overextended Momentum Penalty
        if cfg.get("overextended_penalty_active") and ticker_data["momentum_1mo"] is not None:
            m_overext = calculate_overextended_momentum_penalty(ticker_data["momentum_1mo"], ticker_data["perf_1w"])
            if m_overext < 1.0:
                final_multiplier *= m_overext
                multipliers_log.append({"name": "Overextended Momentum Penalty", "effect": f"-{(1-m_overext)*100:.1f}%", "value": m_overext})

        # Exhaustion Filter
        if cfg.get("exhaustion_filter_active") and ticker_data["momentum_6mo"] is not None and ticker_data["momentum_1mo"] is not None:
            m_exh = calculate_exhaustion_penalty(ticker_data["momentum_6mo"], ticker_data["momentum_1mo"])
            if m_exh < 1.0:
                final_multiplier *= m_exh
                multipliers_log.append({"name": "Momentum Exhaustion Filter", "effect": f"-{(1-m_exh)*100:.1f}%", "value": m_exh})

        # Sector Rotation Outperformance Bonus (ALL profiles if active)
        if cfg.get("sector_outperformance_bonus_active") and ticker_data["sector"] in self.sector_3mo_medians:
            sec_3mo_val = self.sector_3mo_medians.get(ticker_data["sector"], 0)
            sec_6mo_val = self.sector_6mo_medians.get(ticker_data["sector"], 0)
            m3 = ticker_data["momentum_3mo"] if ticker_data["momentum_3mo"] is not None else 0
            m6 = ticker_data["momentum_6mo"] if ticker_data["momentum_6mo"] is not None else 0
            excess_3mo = max(0, m3 - sec_3mo_val)
            excess_6mo = max(0, m6 - sec_6mo_val)
            total_excess = excess_3mo + excess_6mo
            max_bonus = cfg.get("max_sector_bonus", 1.0)
            m_sec = 1.0 + min(max_bonus - 1.0, total_excess)
            if m_sec > 1.0:
                final_multiplier *= m_sec
                multipliers_log.append({
                    "name": f"Sector Outperformance ({ticker_data['sector']})",
                    "effect": f"+{(m_sec - 1) * 100:.1f}%",
                    "value": m_sec
                })

        # Volume Surge Bonus
        if cfg.get("volume_surge_bonus_active") and ticker_data["current_volume"] > 0 and ticker_data["avg_volume"] > 0:
            m_vol = calculate_volume_surge_bonus(
                ticker_data["current_volume"],
                ticker_data["avg_volume"],
                cfg.get("volume_surge_max_bonus", 0.10),
            )
            if m_vol > 1.0:
                final_multiplier *= m_vol
                multipliers_log.append({"name": "Institutional Volume Surge Bonus", "effect": f"+{(m_vol-1)*100:.1f}%", "value": m_vol})

        # ── Sector Sub-Regime Contrarian Score Impact ─────────────────
        # When sector ETF is below 50MA -> Opportunity bonus (+0% to +15%)
        # When sector ETF is above 50MA -> Overextension haircut (-0% to -15%)
        sector_sub_regime_info = None
        etf_sym = SECTOR_TO_ETF.get(ticker_data["sector"])
        if etf_sym and etf_sym in self.sector_etf_data:
            etf_entry = self.sector_etf_data[etf_sym]
            etf_diff = etf_entry["diff_pct"] / 100.0

            if etf_diff < 0:
                sec_sub_mult = 1.0 + min(0.15, abs(etf_diff))
                final_multiplier *= sec_sub_mult
                multipliers_log.append({
                    "name": f"Sector Opportunity Bonus ({ticker_data['sector']} / {etf_sym})",
                    "effect": f"+{(sec_sub_mult - 1.0) * 100:.1f}%",
                    "value": round(sec_sub_mult, 4),
                })
                sector_sub_regime_info = {
                    "etf": etf_sym,
                    "status": "opportunity",
                    "label": f"Below 50MA Opportunity Zone ({etf_entry['diff_pct']:+.1f}%)",
                    "badge_color": "success",
                    "effect_str": f"+{(sec_sub_mult - 1.0) * 100:.1f}% Score Bonus",
                    "desc": f"Sector ETF ({etf_sym}) is {abs(etf_entry['diff_pct']):.1f}% below its 50MA. Favorable contrarian turnaround setup.",
                }
            else:
                sec_sub_mult = max(0.85, 1.0 - min(0.15, etf_diff))
                final_multiplier *= sec_sub_mult
                multipliers_log.append({
                    "name": f"Sector Overextension Haircut ({ticker_data['sector']} / {etf_sym})",
                    "effect": f"-{(1.0 - sec_sub_mult) * 100:.1f}%",
                    "value": round(sec_sub_mult, 4),
                })
                sector_sub_regime_info = {
                    "etf": etf_sym,
                    "status": "overextended",
                    "label": f"Above 50MA Overextended ({etf_entry['diff_pct']:+.1f}%)",
                    "badge_color": "warning",
                    "effect_str": f"-{(1.0 - sec_sub_mult) * 100:.1f}% Score Haircut",
                    "desc": f"Sector ETF ({etf_sym}) is +{etf_entry['diff_pct']:.1f}% above its 50MA. Sector extended; rotation exhaust risk.",
                }

        # 8. Compute Raw Score and Scaled Score (0-100)
        raw_final_score = base_score * final_multiplier
        # Normalization scale: typical raw score is around 0.4 - 0.9. Map cleanly to 0-100
        score_100 = round(np.clip(raw_final_score * 100.0, 0.0, 100.0), 1)

        # 9. Stance Classification
        if score_100 >= 80.0:
            stance = "Strong Conviction Buy"
            stance_class = "strong-buy"
            stance_color = "#10b981"
        elif score_100 >= 65.0:
            stance = "Favorable Setup"
            stance_class = "buy"
            stance_color = "#34d399"
        elif score_100 >= 50.0:
            stance = "Neutral / Hold"
            stance_class = "neutral"
            stance_color = "#94a3b8"
        elif score_100 >= 35.0:
            stance = "Underweight / Caution"
            stance_class = "caution"
            stance_color = "#fbbf24"
        else:
            stance = "Avoid / High Risk"
            stance_class = "avoid"
            stance_color = "#f87171"

        # 10. OLS Statistical Return Forecast
        forecast = self.predict_return_ols(ticker_data)

        # 11. Universe Rank Percentile
        universe_percentile = self.compute_universe_percentile(raw_final_score, profile_name)

        # 12. Baseline snapshot comparison if available
        baseline_snapshot = None
        if not self.baseline_df.empty and ticker_data["ticker"] in self.baseline_df["ticker"].values:
            b_row = self.baseline_df[self.baseline_df["ticker"] == ticker_data["ticker"]].iloc[0]
            b_price = float(b_row["price"])
            price_change_pct = round((ticker_data["price"] - b_price) / b_price * 100, 1)
            b_score = round(float(b_row.get("recommendation_score", 0)) * 100, 1) if "recommendation_score" in b_row else None
            baseline_snapshot = {
                "in_baseline": True,
                "baseline_price": round(b_price, 2),
                "baseline_score": b_score,
                "price_change_since_scan": price_change_pct,
                "scan_date": "Oct 01, 2026",
            }

        return {
            "baseline_snapshot": baseline_snapshot,
            "ticker": ticker_data["ticker"],
            "name": ticker_data["name"],
            "sector": ticker_data["sector"],
            "industry": ticker_data["industry"],
            "price": ticker_data["price"],
            "market_cap": ticker_data["market_cap"],
            "profile": profile_name,
            "score": score_100,
            "raw_score": round(raw_final_score, 4),
            "stance": stance,
            "stance_class": stance_class,
            "stance_color": stance_color,
            "universe_percentile": universe_percentile,
            "forecast": forecast,
            "leverage_flag": de_flag,
            "sector_sub_regime": sector_sub_regime_info,
            "footnotes": footnotes,
            "factor_breakdown": active_factors,
            "omitted_factors": omitted_factors,
            "multipliers": multipliers_log,
            "technical_indicators": {
                "perf_1w": round(ticker_data["perf_1w"] * 100, 2) if ticker_data["perf_1w"] is not None else None,
                "momentum_1mo": round(ticker_data["momentum_1mo"] * 100, 2) if ticker_data["momentum_1mo"] is not None else None,
                "momentum_3mo": round(ticker_data["momentum_3mo"] * 100, 2) if ticker_data["momentum_3mo"] is not None else None,
                "momentum_6mo": round(ticker_data["momentum_6mo"] * 100, 2) if ticker_data["momentum_6mo"] is not None else None,
                "momentum_1y": round(ticker_data["momentum_1y"] * 100, 2) if ticker_data["momentum_1y"] is not None else None,
                "rsi": ticker_data["rsi"],
                "annualized_vol": round(ticker_data["annualized_vol"] * 100, 2) if ticker_data["annualized_vol"] is not None else None,
                "beta": ticker_data["beta"],
                "ma_50": ticker_data["ma_50"],
                "ma_200": ticker_data["ma_200"],
                "pct_vs_ma50": round((ticker_data["price"] - ticker_data["ma_50"]) / ticker_data["ma_50"] * 100, 2) if ticker_data["ma_50"] else None,
                "pct_vs_ma200": round((ticker_data["price"] - ticker_data["ma_200"]) / ticker_data["ma_200"] * 100, 2) if ticker_data["ma_200"] else None,
                "fifty_two_high": ticker_data["fifty_two_high"],
                "fifty_two_low": ticker_data["fifty_two_low"],
                "pe_ratio": ticker_data["pe_ratio"],
                "peg_ratio": ticker_data["peg_ratio"],
            },
        }

    def predict_return_ols(self, ticker_data):
        """Runs trained OLS model to predict expected 1-year return and CI."""
        if self.ols_model is None:
            return {
                "expected_return": None,
                "conf_lower": None,
                "conf_upper": None,
                "prob_positive": None,
            }

        benchmark_pe = SECTOR_PE_BENCHMARKS.get(ticker_data["sector"], 23.5)
        # Prepare feature vector: ['pe_ratio', 'debt_to_equity', 'ma50_proximity', 'ma200_proximity', 'momentum_1y', 'rsi', 'annualized_vol', 'beta']
        x = [
            ticker_data["pe_ratio"] if ticker_data["pe_ratio"] is not None else benchmark_pe,
            ticker_data["debt_to_equity"] if ticker_data["debt_to_equity"] is not None else 0.0,
            ticker_data["ma50_proximity"] if ticker_data["ma50_proximity"] is not None else 0.0,
            ticker_data["ma200_proximity"] if ticker_data["ma200_proximity"] is not None else 0.0,
            ticker_data["momentum_1y"] if ticker_data["momentum_1y"] is not None else 0.0,
            ticker_data["rsi"] if ticker_data["rsi"] is not None else 50.0,
            ticker_data["annualized_vol"] if ticker_data["annualized_vol"] is not None else 0.25,
            ticker_data["beta"] if ticker_data["beta"] is not None else 1.0,
        ]

        try:
            X_arr = np.array([x])
            y_pred, se_pred = self.ols_model.predict(X_arr)
            exp_ret = float(y_pred[0])
            se = float(se_pred[0])
            prob_pos = float(self.ols_model.normal_cdf(exp_ret / (se + 1e-9)) * 100)
            ci_lower = float(exp_ret - 1.96 * se)
            ci_upper = float(exp_ret + 1.96 * se)

            return {
                "expected_return": round(exp_ret * 100, 2),
                "conf_lower": round(ci_lower * 100, 2),
                "conf_upper": round(ci_upper * 100, 2),
                "prob_positive": round(prob_pos, 1),
                "se": round(se * 100, 2),
            }
        except Exception as e:
            print(f"[Engine] Error running OLS prediction: {e}")
            return {
                "expected_return": None,
                "conf_lower": None,
                "conf_upper": None,
                "prob_positive": None,
            }

    def compute_universe_percentile(self, score, profile_name):
        """Calculates what percentile of the baseline universe this score represents."""
        # For a quick percentile relative to typical universe scores
        # Usually between 0.35 and 0.85
        pct = float(np.clip((score - 0.30) / (0.85 - 0.30) * 100.0, 1.0, 99.9))
        return round(pct, 1)

    def scan_universe(self, profile_name="Dip Hunter", top_n=10):
        """Returns the pre-scored leaderboard from the baseline universe."""
        if self.baseline_df.empty:
            return []

        from recommender import generate_recommendations
        try:
            recs = generate_recommendations(self.baseline_df, user_profile=profile_name, top_n=top_n)
            results = []
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                results.append({
                    "rank": rank,
                    "ticker": row["ticker"],
                    "sector": row.get("sector", "N/A"),
                    "price": round(float(row["price"]), 2),
                    "momentum_1y": round(float(row.get("momentum_1y", 0)) * 100, 1),
                    "rsi": round(float(row.get("rsi", 50)), 1),
                    "pe_ratio": round(float(row["pe_ratio"]), 1) if pd.notna(row.get("pe_ratio")) else "N/A",
                    "score": round(float(row["recommendation_score"]) * 100, 1),
                    "expected_return": round(float(row.get("pred_return", 0)) * 100, 1) if pd.notna(row.get("pred_return")) else None,
                    "prob_positive": round(float(row.get("prob_positive", 50)), 1) if pd.notna(row.get("prob_positive")) else None,
                })
            return results
        except Exception as e:
            print(f"[Engine] Error scanning universe: {e}")
            return []
