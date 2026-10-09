"""
======================================================================
BASAK BOSSMAN ENGINE: HYBRID V2.3
======================================================================
A sophisticated, adaptive multi-factor stock backtesting engine.
Combines MA-Proximity Trend-Following with Targeted Fundamental
Value Shields, plus anti-bandwagon controls.

Final Architecture:
- MA Proximity Scoring: Replaces raw 1mo and 6mo momentum with
  50-day and 200-day moving average proximity scores. Rewards
  stocks trading near their trend lines; penalises only positive
  overextension (>15% above MA). Stocks below their MA are NOT
  penalised — this preserves alignment with undervalued-stock
  bonuses elsewhere in the model.
- P/E Value Bonus: Active for Low/Low-Med profiles only.
- Debt Penalty: Soft penalty for Low/Low-Med profiles only.
  High D/E is penalised conservatively — companies that operate
  bullishly with high leverage (Medium and above) are unaffected.
- Adaptive Gear Shift: Dynamic 0.05 shift based on 3mo vs 1mo trend.
- Robustness: Beta speed limits active for conservative tiers.
- RSI > 70 Penalty: Active for ALL profiles. Prevents chasing
  overbought stocks regardless of risk appetite.
- RSI ≤ 50 Bonus: Active for Low, Low-Med, Medium profiles only.
  Rewards stocks with room to run before becoming overbought.
- P/E Stretch Penalty: Active for Low, Low-Med profiles only.
  Penalises stocks trading at ≥1.5× their sector average P/E.
- Below 200MA Bonus: Active for ALL profiles. Rewards stocks
  trading >10% below their 200-day moving average. For Medium-High
  and High profiles, the bonus scales with depth (capped at 30%
  below) to reward aggressive dip-buying.

CHANGELOG v2.3:
- CHANGE: Replaced raw 1-month momentum with 50-day MA proximity.
- CHANGE: Replaced raw 6-month momentum with 200-day MA proximity.
- CHANGE: MA proximity is one-sided — only penalises stocks >15%
  ABOVE their moving average. Stocks at or below MA score equally.
- CHANGE: 200MA bonus redesigned — only activates >10% below.
  Conservative profiles: small fixed bonus (max +8%).
  Aggressive profiles: scaling bonus (max +15%, capped at 30% below).
- Renamed weight keys: momentum_1moo_weight → ma50_proximity_weight,
  momentum_6mo_weight → ma200_proximity_weight.

CHANGELOG v2.2:
- ADD: RSI > 70 penalty extended to ALL profiles (was robustness-only).
- ADD: RSI ≤ 50 bonus for Low/Low-Med/Medium to reward room-to-run.
- ADD: P/E stretch penalty (≥1.5× sector avg) for Low/Low-Med profiles.
- ADD: Below-200MA bonus for all profiles as a value-hunting signal.
  Requires 'price' and 'ma_200' fields in the feature table.

CHANGELOG v2.1:
- FIX: Beta robustness direction corrected. High-beta stocks now
  receive a TIGHTER speed limit (safe_limit divided by beta), not a
  looser one. Previously multiplying by (1 + beta) was rewarding
  volatile stocks with more room before penalty, which is backwards.
- ADD: calculate_solvency_penalty() restored. Soft, gradual D/E
  penalty only for 'Bear Market Shield' and 'Fundamental Bag' profiles. Uses sector-
  specific thresholds (Utilities and Financials get higher limits
  since leverage is structurally normal for them). Floors at 0.7
  so it never catastrophically tanks a stock's score.
======================================================================"""
import requests
from io import StringIO
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

RISK_LEVELS = {
    "Bear Market Shield": {
        "momentum_6mo_weight": 0.5455,        # 54.55% of base score (inverted rank to reward worst momentum)
        "ma50_proximity_weight": 0.2727,      # 27.27% of base score (symmetric 50MA scale, half of 6mo wt)
        "momentum_1mo_weight": 0.1818,        # 18.18% of base score (inverted rank, 1/3 of 6mo wt)
        "ma200_proximity_weight": 0.00,
        "momentum_1y_weight": 0.00,
        "value_weight": 0.00,
        "risk_weight": 0.00,

        "vol_limit": 0.40,
        "robustness_active": False,
        "solvency_active": True,      # Soft D/E penalty ON for conservative profiles
        "rsi_oversold_bonus_active": True,   # Bonus for RSI ≤ 50
        "rsi_oversold_bonus_max": 0.10,       # Matches Volume Surge (+10.0%)
        "pe_stretch_penalty_active": True,   # Penalty for P/E ≥ 1.5× sector avg
        "ma200_bonus_active": True,          # Bonus for price below 200-day MA
        "ma200_bonus_max": 0.12,             # 1.2x of RSI/Volume surge (+12.0%)
        "ma200_bonus_max_depth": 0.30,       # Full bonus at 30% below
        "sector_outperformance_bonus_active": True,
        "max_sector_bonus": 1.05,
        "low_pe_bonus_active": True,
        "peg_bonus_active": True,
        "overextended_penalty_active": True,
        "exhaustion_filter_active": True,
        "short_penalty_active": True,
        "short_penalty_severity": 2.0,
        "volume_surge_bonus_active": True,
        "volume_surge_max_bonus": 0.10,      # Base V (+10.0%)
    },
    "Fundamental Bag": {
        "momentum_6mo_weight": 0.5455,        # 54.55% of base score (inverted rank to reward worst momentum)
        "ma50_proximity_weight": 0.2727,      # 27.27% of base score (symmetric 50MA scale, half of 6mo wt)
        "momentum_1mo_weight": 0.1818,        # 18.18% of base score (inverted rank, 1/3 of 6mo wt)
        "ma200_proximity_weight": 0.00,
        "momentum_1y_weight": 0.00,
        "value_weight": 0.00,
        "risk_weight": 0.00,

        "vol_limit": 0.40,
        "robustness_active": False,
        "solvency_active": True,      # Soft D/E penalty ON for conservative profiles
        "rsi_oversold_bonus_active": True,
        "rsi_oversold_bonus_max": 0.10,       # Matches Volume Surge (+10.0%)
        "pe_stretch_penalty_active": True,
        "ma200_bonus_active": True,
        "ma200_bonus_max": 0.12,             # 1.2x of RSI/Volume surge (+12.0%)
        "ma200_bonus_max_depth": 0.30,       # Full bonus at 30% below
        "sector_outperformance_bonus_active": True,
        "max_sector_bonus": 1.10,
        "low_pe_bonus_active": True,
        "peg_bonus_active": True,
        "overextended_penalty_active": True,
        "exhaustion_filter_active": True,
        "short_penalty_active": True,
        "short_penalty_severity": 1.0,
        "volume_surge_bonus_active": True,
        "volume_surge_max_bonus": 0.10,      # Base V (+10.0%)
    },
    "All-Weather Core": {
        "risk_weight": 0.463,                 # 46.3% of base score (annualized_vol t-stat = 25.40)
        "ma50_proximity_weight": 0.236,       # 23.6% of base score (ma50_proximity t-stat = 12.91)
        "momentum_6mo_weight": 0.151,         # 15.1% of base score (momentum_6mo t-stat = 8.29)
        "ma200_proximity_weight": 0.150,      # 15.0% of base score (ma200_proximity t-stat = 8.21)
        "momentum_1y_weight": 0.00,           # Removed
        "value_weight": 0.00,

        "vol_limit": 0.40,
        "robustness_active": False,
        "solvency_active": False,     # Bullish/leveraged operators not penalised
        "rsi_oversold_bonus_active": False,   # Removed RSI oversold bonus
        "pe_stretch_penalty_active": False,  # Not valuation-constrained
        "ma200_bonus_active": True,
        "ma200_bonus_max": 0.08,             # Max +8% bonus
        "ma200_bonus_max_depth": 0.20,       # Full bonus at 20% below
        "sector_outperformance_bonus_active": True,
        "max_sector_bonus": 1.15,
        "low_pe_bonus_active": True,
        "peg_bonus_active": True,
        "overextended_penalty_active": True,
        "exhaustion_filter_active": True,
        "short_penalty_active": True,
        "short_penalty_severity": 0.5,
        "volume_surge_bonus_active": True,
        "volume_surge_max_bonus": 0.05,
    },
    "Dip Hunter": {
        "momentum_3mo_weight": 0.255,        # 25.5% of base score (t-stat = +7.94)
        "momentum_1y_weight": 0.215,         # 21.5% of base score (inverted rank, t-stat = -6.70)
        "ma50_proximity_weight": 0.204,       # 20.4% of base score (inverted rank, t-stat = -6.35)
        "momentum_6mo_weight": 0.188,        # 18.8% of base score (t-stat = +5.85)
        "ma200_proximity_weight": 0.138,      # 13.8% of base score (inverted rank, t-stat = -4.29)
        "value_weight": 0.00,
        "risk_weight": 0.00,

        "vol_limit": 0.30,
        "robustness_active": False,
        "solvency_active": False,
        "rsi_oversold_bonus_active": False,   # Cleansed: No RSI bonus (t-stat = -0.78, noise)
        "rsi_oversold_bonus_max": 0.00,
        "rsi_falling_knife_penalty_active": True, # Active: Penalty for RSI < 30
        "pe_stretch_penalty_active": False,
        "ma200_bonus_active": True,
        "ma200_bonus_max": 0.15,             # Max +15% bonus for below 200MA
        "ma200_bonus_max_depth": 0.30,       # Full bonus at 30% below
        "sector_outperformance_bonus_active": True,
        "max_sector_bonus": 1.175,
        "low_pe_bonus_active": True,
        "peg_bonus_active": True,
        "overextended_penalty_active": True,
        "exhaustion_filter_active": True,
        "short_penalty_active": False,
        "volume_surge_bonus_active": True,
        "volume_surge_max_bonus": 0.10,
    },
    "Alpha Bull": {
        "ma50_proximity_weight": 0.642,       # 64.2% of base score (multivariate t-stat ratio)
        "ma200_proximity_weight": 0.358,      # 35.8% of base score (multivariate t-stat ratio)
        "momentum_1y_weight": 0.00,           # Removed
        "value_weight": 0.0,
        "risk_weight": 0.00,

        "vol_limit": None,                    # Volatility limit removed
        "robustness_active": False,
        "solvency_active": False,
        "rsi_oversold_bonus_active": True,
        "rsi_oversold_bonus_max": 0.05,       # 1/3 of high beta bonus (+5.0%)
        "pe_stretch_penalty_active": False,
        "high_beta_bonus_active": True,
        "high_beta_bonus_max": 0.15,          # Reference B_beta (+15.0%)
        "ma50_deviation_bonus_active": True,
        "ma50_deviation_bonus_max": 0.075,    # 1/2 of high beta bonus (+7.5%)
        "ma50_deviation_cutoff": 0.15,        # Turns OFF if > 15% above MA50
        "ma200_bonus_active": True,
        "ma200_bonus_max": 0.15,             # Scaling bonus, max +15%
        "ma200_bonus_max_depth": 0.30,       # Full bonus at 30% below
        "sector_outperformance_bonus_active": True,
        "max_sector_bonus": 1.20,
        "low_pe_bonus_active": True,
        "peg_bonus_active": True,
        "overextended_penalty_active": True,
        "exhaustion_filter_active": True,
        "short_penalty_active": False,
        "volume_surge_bonus_active": True,
        "volume_surge_max_bonus": 0.20,
    }
}

# Prevailing Sector-Specific P/E Benchmarks
SECTOR_PE_BENCHMARKS = {
    "Information Technology": 31.4,
    "Semiconductors": 34.0,
    "Health Care": 19.2,
    "Financials": 15.8,
    "Utilities": 17.5,
    "Energy": 11.8,
    "Consumer Staples": 20.6,
    "Consumer Discretionary": 26.2,
    "Communication Services": 19.8,
    "Industrials": 21.4,
    "Materials": 19.4,
    "Real Estate": 18.2,
    "N/A": 23.5
}

# Sector Safe Limits for robustness speed checks (weekly return caps)
SECTOR_SAFE_LIMITS = {
    "Information Technology": 0.12,
    "Health Care": 0.08,
    "Financials": 0.08,
    "Consumer Discretionary": 0.10,
    "Communication Services": 0.10,
    "Industrials": 0.07,
    "Consumer Staples": 0.05,
    "Energy": 0.15,
    "Utilities": 0.05,
    "Real Estate": 0.06,
    "Materials": 0.07,
    "N/A": 0.08
}

# Sector D/E thresholds (as yfinance percentage, e.g. 150 = 1.5x) above which
# the soft solvency penalty activates. Sectors like Utilities, REITs and
# Financials carry structural leverage so their thresholds are set higher.
# Recalibrated using Damodaran's Market D/E (adjusted for leases) medians * 1.5x.
SECTOR_DE_THRESHOLDS = {
    "Information Technology": 12,
    "Health Care": 22,
    "Financials": 250,      # Special case to accommodate structural leverage
    "Consumer Discretionary": 34,
    "Communication Services": 60,
    "Industrials": 38,
    "Consumer Staples": 65,
    "Energy": 33,
    "Utilities": 117,
    "Real Estate": 127,
    "Materials": 50,
    "N/A": 100
}


def normalize(series):
    if series.max() == series.min():
        return series * 0.0
    return series.rank(pct=True)


def calculate_ma_proximity(price, ma):
    """
    Two-sided moving average proximity penalty.
    Penalizes stocks >15% ABOVE the MA (overextended) AND >15% BELOW the MA (severe downtrend).
    """
    if pd.isna(ma) or ma <= 0:
        return 0.0
    pct_diff = (price - ma) / ma
    if pct_diff > 0.15:
        return -(pct_diff - 0.15)
    return 0.0


def risk_penalty(vol, limit):
    if limit is None or pd.isna(limit):
        return 1.0
    if vol <= limit:
        return 1.0
    return max(0, 1.0 - (vol - limit) * 2.0)


def calculate_value_score(pe, benchmark):
    try:
        pe = float(pe)
    except (ValueError, TypeError):
        return 0.0
    if pd.isna(pe) or pe <= 0:
        return 0.0
    return max(0.0, 1.0 - (pe / benchmark))


def calculate_solvency_penalty(de_ratio, sector):
    """
    Soft D/E penalty — applied only for low and low-medium profiles.

    Only activates above sector-specific thresholds. The penalty is
    gradual (15% reduction per full threshold breached) and floors at
    0.7, so genuinely leveraged-but-growing companies take a nudge
    rather than a knockout.

    yfinance reports D/E as a percentage (150 = 1.5x debt-to-equity).
    """
    try:
        de = float(de_ratio)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(de) or de < 0:
        return 1.0

    threshold = SECTOR_DE_THRESHOLDS.get(sector, 100)
    if de <= threshold:
        return 1.0

    excess_ratio = (de - threshold) / threshold
    return max(0.6, 1.0 - (excess_ratio * 0.15))


def calculate_rsi_overbought_penalty(rsi):
    """
    Penalises overbought stocks (RSI > 70) for ALL risk profiles.

    The engine was too reactive/bandwagon-prone because it kept
    rewarding recent momentum without checking if a stock was
    already overstretched. This is a hard multiplier: stocks with
    RSI well above 70 take a meaningful hit.

    - RSI ≤ 70  → no penalty (multiplier = 1.0)
    - RSI = 80  → ~10% reduction
    - RSI = 90  → ~20% reduction
    - Floors at 0.70 so even extreme RSIs don't catastrophically
      collapse the score.
    """
    try:
        rsi = float(rsi)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(rsi) or rsi <= 65:
        return 1.0
    return max(0.65, 1.0 - (rsi - 65) / 100)


def calculate_rsi_falling_knife_penalty(rsi):
    """
    Penalises extreme oversold / falling knives (RSI < 30).
    1% reduction per point below 30, floored at 0.70 (max 30% haircut).
    - RSI >= 30  → no penalty (multiplier = 1.0)
    - RSI = 20   → 10% reduction (multiplier = 0.90)
    - RSI <= 0   → 30% reduction (multiplier = 0.70, floor)
    """
    try:
        rsi = float(rsi)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(rsi) or rsi >= 30:
        return 1.0
    return max(0.70, 1.0 - (30 - rsi) / 100)


def calculate_rsi_oversold_bonus(rsi, max_bonus=0.15):
    """
    Rewards stocks with RSI between 30 and 50 for profiles where RSI bonus is active.
    - RSI = 50  → +0% bonus (multiplier = 1.0)
    - RSI = 30  → max_bonus (e.g. +15% or +5%)
    - RSI < 30 or RSI > 50 → 1.0 (no bonus)
    """
    try:
        rsi = float(rsi)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(rsi) or rsi > 50 or rsi < 30:
        return 1.0
        
    # Scale: 0 bonus at RSI=50, growing to max_bonus at RSI=30
    bonus = ((50 - rsi) / 20.0) * max_bonus
    return 1.0 + bonus


def calculate_high_beta_bonus(beta, max_bonus=0.15):
    """
    Rewards high beta stocks.
    - Beta <= 1.0  → no bonus (multiplier = 1.0)
    - Beta >= 1.5  → full max_bonus (multiplier = 1.0 + max_bonus)
    - Scales linearly between Beta 1.0 and 1.5.
    """
    try:
        b = float(beta)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(b) or b <= 1.0:
        return 1.0
    scale = min(1.0, (b - 1.0) / 0.5)
    return 1.0 + (max_bonus * scale)


def calculate_ma50_deviation_bonus(price, ma_50, max_bonus=0.075, cutoff=0.15):
    """
    Rewards mean deviation from the 50-day MA at both extremes,
    UNLESS price is > 15% (cutoff) above MA50, in which case bonus is turned OFF (1.0).
    
    - If price > ma_50 * (1 + cutoff) → 1.0 (turned off)
    - Otherwise, absolute percent deviation |price - ma_50| / ma_50:
      scales linearly from 0% deviation up to cutoff deviation (full max_bonus).
    """
    try:
        p = float(price)
        ma = float(ma_50)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(p) or pd.isna(ma) or ma <= 0:
        return 1.0
    
    pct_diff = (p - ma) / ma
    if pct_diff > cutoff:
        return 1.0  # Turned off if > 15% above MA50
        
    abs_dev = abs(pct_diff)
    scale = min(1.0, abs_dev / cutoff)
    return 1.0 + (max_bonus * scale)


def calculate_pe_stretch_penalty(pe, benchmark):
    """
    Penalises stocks whose P/E is at least 1.5× their sector average.
    Active only for Low and Low-Med profiles.

    A stock trading at a significant premium to its sector peers is a
    bandwagon risk — you may be buying the hype, not the value. The
    penalty is graded: sharper the stretch, the bigger the haircut.

    - P/E < 1.5× sector avg  → no penalty (multiplier = 1.0)
    - P/E = 1.5× sector avg  → ~10% reduction (multiplier = 0.90)
    - P/E = 2.0× sector avg  → ~20% reduction (multiplier = 0.80)
    - Floors at 0.65 to prevent a knockout on high-growth names.
    """
    try:
        pe = float(pe)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(pe) or pe <= 0:
        return 1.0
    stretch = pe / benchmark
    if stretch < 1.5:
        return 1.0
    # 20% penalty per full stretch multiple above 1.5×, floored at 0.65
    return max(0.65, 1.0 - (stretch - 1.5) * 0.20)


def calculate_ma200_bonus(price, ma_200, momentum_1y, max_bonus=0.08, max_depth=0.20):
    """
    Rewards stocks trading more than 10% below their 200-day MA,
    UNLESS their 1-year momentum is extremely negative (Falling Knife check).

    Only activates when the stock is meaningfully beaten down (>10%
    below the 200MA). This filters out normal pullbacks and targets
    genuine dip opportunities.

    The max_bonus and max_depth parameters are profile-aware:
    - Conservative profiles (low/low-med/medium): max_bonus=0.08,
      max_depth=0.20 — small, capped bonus for beaten-down names.
    - Aggressive profiles (medium-high/high): max_bonus=0.12-0.15,
      max_depth=0.30 — scaling bonus that rewards deeper dips, capping
      at 30% below to avoid distressed-stock traps.

    - Price within 10% of 200MA → no bonus (multiplier = 1.0)
    - Price = 15% below 200MA   → partial bonus (scaled linearly)
    - Price = max_depth below    → full max_bonus
    - Bonus does not increase beyond max_depth below.
    """
    try:
        price  = float(price)
        ma_200 = float(ma_200)
        momentum_1y = float(momentum_1y)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(price) or pd.isna(ma_200) or ma_200 <= 0 or price >= ma_200:
        return 1.0
        
    # Falling Knife check: revoke bonus if stock is dying
    if not pd.isna(momentum_1y) and momentum_1y < -0.30:
        return 1.0
        
    pct_below = (ma_200 - price) / ma_200  # e.g. 0.15 = 15% below
    if pct_below <= 0.10:
        return 1.0
    # Scale linearly from 10% below (0 bonus) to max_depth below (full bonus)
    depth_range = max_depth - 0.10
    if depth_range <= 0:
        return 1.0 + max_bonus  # edge case: max_depth == 0.10
    scale = min(1.0, (pct_below - 0.10) / depth_range)
    return 1.0 + (max_bonus * scale)


def calculate_low_pe_bonus(pe):
    """
    Rewards stocks with a P/E ratio below 10 for ALL profiles.
    - P/E >= 10   → no bonus (multiplier = 1.0)
    - P/E < 10    → progressive bonus up to +5% as P/E approaches 0.
    """
    try:
        pe = float(pe)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(pe) or pe <= 0 or pe >= 10:
        return 1.0
    return 1.0 + (10 - pe) / 10 * 0.05

def calculate_peg_bonus(peg):
    """
    Rewards stocks with a PEG ratio below 1.0.
    - PEG >= 1.0  → no bonus (multiplier = 1.0)
    - PEG < 1.0   → progressive bonus up to +5% as PEG approaches 0.
    """
    try:
        peg = float(peg)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(peg) or peg <= 0 or peg >= 1.0:
        return 1.0
    return 1.0 + (1.0 - peg) * 0.05


def calculate_overextended_momentum_penalty(momentum_1mo, perf_1w):
    """
    Applies a hard penalty of 0.50 if recent performance is overextended:
    - 1-month momentum > 30% OR 1-week performance > 15%.
    """
    try:
        m1 = float(momentum_1mo)
        w1 = float(perf_1w)
    except (ValueError, TypeError):
        return 1.0
    if (not pd.isna(m1) and m1 > 0.40) or (not pd.isna(w1) and w1 > 0.10):
        return 0.50
    return 1.0


def calculate_exhaustion_penalty(momentum_6mo, momentum_1mo):
    """
    Scaled penalty for momentum exhaustion: fires when the gap between
    6-month and 1-month momentum exceeds 40%.

    - diff ≤ 0.40  → no penalty (multiplier = 1.0)
    - diff = 0.50  → −10% (multiplier = 0.90)
    - diff = 0.70  → −30% (multiplier = 0.70)
    - diff ≥ 0.90  → −50% (multiplier = 0.50, floor)
    """
    try:
        m6 = float(momentum_6mo)
        m1 = float(momentum_1mo)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(m6) or pd.isna(m1):
        return 1.0
    diff = m6 - m1
    if diff <= 0.40:
        return 1.0
    return max(0.50, 1.0 - (diff - 0.40))


def calculate_short_penalty(short_percent, severity):
    """
    Penalizes highly shorted stocks.
    short_percent is e.g. 0.10 for 10%
    severity scales the penalty. 
    A severity of 2.0 means 10% short interest -> 20% score penalty (multiplier = 0.80).
    Floored at 0.50.
    """
    try:
        sp = float(short_percent)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(sp) or sp <= 0.05:  # Ignore normal background shorting (<= 5%)
        return 1.0
    
    penalty = (sp - 0.05) * severity
    return max(0.50, 1.0 - penalty)


def calculate_volume_surge_bonus(current_vol, avg_vol, max_bonus):
    """
    Rewards stocks breaking out on unusually high volume.
    Requires current_vol > 1.5 * avg_vol to activate.
    Scales linearly up to 3.0 * avg_vol for max_bonus.
    """
    try:
        cv = float(current_vol)
        av = float(avg_vol)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(cv) or pd.isna(av) or av <= 0:
        return 1.0
    
    ratio = cv / av
    if ratio <= 1.5:
        return 1.0
    
    # Scale from 1.5x (0 bonus) to 3.0x (max_bonus)
    scale = min(1.0, (ratio - 1.5) / 1.5)
    return 1.0 + (max_bonus * scale)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df = pd.read_html(StringIO(requests.get(url, headers=headers).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    sector_map = dict(zip(df['Symbol'], df['GICS Sector']))
    return df['Symbol'].tolist(), sector_map


def generate_recommendations(stocks_df, user_profile="Dip Hunter", top_n=10, pe_benchmarks=None):
    df = stocks_df.copy()
    cfg = RISK_LEVELS.get(user_profile, RISK_LEVELS["Dip Hunter"])

    # Fallback to static benchmarks if none provided (e.g. for historical backtests)
    if pe_benchmarks is None:
        pe_benchmarks = SECTOR_PE_BENCHMARKS

    # ── Normalize MA proximity features (primary scoring signals) ───
    for col in ["ma50_proximity", "ma200_proximity"]:
        if col in df.columns:
            df[col + "_norm"] = normalize(df[col])

    # ── Normalize momentum features (used by gear-shift + penalties) ─
    for col in ["momentum_1mo", "momentum_3mo", "momentum_6mo", "momentum_1y"]:
        if col in df.columns:
            df[col + "_norm"] = normalize(df[col])

    # Calculate sector medians for outperformance bonus
    sector_3mo = df.groupby('sector')['momentum_3mo'].median().to_dict() if 'momentum_3mo' in df.columns else {}
    sector_6mo = df.groupby('sector')['momentum_6mo'].median().to_dict() if 'momentum_6mo' in df.columns else {}

    scores = []
    use_conditional = user_profile in ["All-Weather Core", "Dip Hunter"]
    use_robustness = cfg["robustness_active"]
    use_solvency = cfg["solvency_active"]
    use_rsi_oversold_bonus = cfg.get("rsi_oversold_bonus_active", False)
    rsi_oversold_bonus_max = cfg.get("rsi_oversold_bonus_max", 0.15)
    use_rsi_falling_knife = cfg.get("rsi_falling_knife_penalty_active", True)
    use_pe_stretch_penalty = cfg.get("pe_stretch_penalty_active", False)
    use_high_beta_bonus = cfg.get("high_beta_bonus_active", False)
    high_beta_bonus_max = cfg.get("high_beta_bonus_max", 0.15)
    use_ma50_dev_bonus = cfg.get("ma50_deviation_bonus_active", False)
    ma50_dev_bonus_max = cfg.get("ma50_deviation_bonus_max", 0.075)
    ma50_dev_cutoff = cfg.get("ma50_deviation_cutoff", 0.15)
    use_ma200_bonus = cfg.get("ma200_bonus_active", False)
    ma200_bonus_max = cfg.get("ma200_bonus_max", 0.08)
    ma200_bonus_max_depth = cfg.get("ma200_bonus_max_depth", 0.20)
    use_sector_bonus = cfg.get("sector_outperformance_bonus_active", False)
    max_sector_bonus = cfg.get("max_sector_bonus", 1.0)
    use_low_pe_bonus = cfg.get("low_pe_bonus_active", False)
    use_peg_bonus = cfg.get("peg_bonus_active", False)
    use_overextended_penalty = cfg.get("overextended_penalty_active", False)
    use_exhaustion_filter = cfg.get("exhaustion_filter_active", False)
    use_short_penalty = cfg.get("short_penalty_active", False)
    short_penalty_severity = cfg.get("short_penalty_severity", 1.0)
    use_volume_surge = cfg.get("volume_surge_bonus_active", False)
    volume_surge_max_bonus = cfg.get("volume_surge_max_bonus", 0.10)

    # ── Weight keys for base score factors ─────────────────────────
    base_3mo_wt = cfg.get("momentum_3mo_weight", 0.0)
    base_6mo_wt = cfg.get("momentum_6mo_weight", 0.0)
    base_1mo_wt = cfg.get("momentum_1mo_weight", 0.0)
    base_ma50_wt = cfg.get("ma50_proximity_weight", 0.0)
    base_ma200_wt = cfg.get("ma200_proximity_weight", 0.0)
    base_1y_wt = cfg.get("momentum_1y_weight", 0.0)

    for _, row in df.iterrows():
        # Adaptive gear shift (only if use_conditional is True)
        # Dynamically shifts weight between MA50 (turnaround) and MA200 (deep dip)
        # without leaking base_3mo_wt or disturbing total factor weight budget.
        if use_conditional:
            shift = 0.05
            if row.get("momentum_3mo", 0) > row.get("momentum_1mo", 0):
                w_ma50 = base_ma50_wt - shift
                w_ma200 = base_ma200_wt + shift
            else:
                w_ma50 = base_ma50_wt + shift
                w_ma200 = base_ma200_wt - shift
            w_3mo = base_3mo_wt
        else:
            w_ma50, w_3mo, w_ma200 = base_ma50_wt, base_3mo_wt, base_ma200_wt

        benchmark = pe_benchmarks.get(row["sector"], 23.5)

        risk_s = risk_penalty(row["annualized_vol"], cfg.get("vol_limit"))
        value_s = calculate_value_score(row.get("pe_ratio", 20.0), benchmark)
        
        # Profile-specific feature direction (inversion for dip/bear signals)
        if user_profile in ["Bear Market Shield", "Fundamental Bag"]:
            m6_score = 1.0 - row.get("momentum_6mo_norm", 0.5)
            m1_score = 1.0 - row.get("momentum_1mo_norm", 0.5)
            ma50_score = 1.0 - row.get("ma50_proximity_norm", 0.5)
            m3_score = row.get("momentum_3mo_norm", 0.5)
            ma200_score = row.get("ma200_proximity_norm", 0.5)
            m1y_score = row.get("momentum_1y_norm", 0.5)
        elif user_profile == "Dip Hunter":
            m3_score = row.get("momentum_3mo_norm", 0.5)
            m6_score = row.get("momentum_6mo_norm", 0.5)
            m1y_score = 1.0 - row.get("momentum_1y_norm", 0.5)       # Inverted rank for -6.70 t-stat
            ma50_score = 1.0 - row.get("ma50_proximity_norm", 0.5)     # Inverted rank for -6.35 t-stat
            ma200_score = 1.0 - row.get("ma200_proximity_norm", 0.5)    # Inverted rank for -4.29 t-stat
            m1_score = row.get("momentum_1mo_norm", 0.5)
        else:
            m3_score = row.get("momentum_3mo_norm", 0.5)
            m6_score = row.get("momentum_6mo_norm", 0.5)
            m1_score = row.get("momentum_1mo_norm", 0.5)
            m1y_score = row.get("momentum_1y_norm", 0.5)
            ma50_score = row.get("ma50_proximity_norm", 0.5)
            ma200_score = row.get("ma200_proximity_norm", 0.5)

        base_score = (
            base_6mo_wt * m6_score +
            base_1mo_wt * m1_score +
            w_ma50 * ma50_score +
            w_3mo * m3_score +
            w_ma200 * ma200_score +
            base_1y_wt * m1y_score +
            cfg["risk_weight"] * risk_s +
            cfg["value_weight"] * value_s
        )

        final_multiplier = 1.0

        # ── High Beta Bonus ───────────────────────────────────────────
        if use_high_beta_bonus:
            final_multiplier *= calculate_high_beta_bonus(
                row.get("beta", 1.0), max_bonus=high_beta_bonus_max
            )

        # ── MA50 Mean Deviation Bonus ─────────────────────────────────
        if use_ma50_dev_bonus:
            final_multiplier *= calculate_ma50_deviation_bonus(
                row.get("price", 0), row.get("ma_50", float("nan")),
                max_bonus=ma50_dev_bonus_max, cutoff=ma50_dev_cutoff
            )

        # ── Soft solvency penalty — Deactivated per mandate: D/E is surfaced as a non-punitive balance sheet flag
        # if use_solvency:
        #     final_multiplier *= calculate_solvency_penalty(
        #         row.get("debt_to_equity", 0), row["sector"]
        #     )

        if use_robustness:
            sector_limit = SECTOR_SAFE_LIMITS.get(row["sector"], 0.08)

            # FIX v2.1: Beta now TIGHTENS the speed limit, not loosens it.
            # A high-beta stock should be flagged sooner for unusual weekly moves,
            # not given more room. Clamped at 0.5 to prevent extreme tightening.
            beta = max(0.5, row.get("beta", 1.0))
            safe_limit = sector_limit / beta

            if row.get("perf_1w", 0) > safe_limit:
                excess = row["perf_1w"] - safe_limit
                final_multiplier *= np.clip(1.0 - (excess * 2.5), 0.5, 1.0)

        # ── RSI > 65 overbought penalty — ALL profiles ─────────────────
        final_multiplier *= calculate_rsi_overbought_penalty(row.get("rsi", 50))

        # ── RSI < 30 falling knife penalty ─────────────────────────────
        if use_rsi_falling_knife:
            final_multiplier *= calculate_rsi_falling_knife_penalty(row.get("rsi", 50))

        # ── RSI 30..50 oversold bonus — active profiles only ───────────
        if use_rsi_oversold_bonus:
            final_multiplier *= calculate_rsi_oversold_bonus(row.get("rsi", 50), max_bonus=rsi_oversold_bonus_max)

        # ── P/E stretch penalty — Low / Low-Med only ───────────────────
        if use_pe_stretch_penalty:
            final_multiplier *= calculate_pe_stretch_penalty(
                row.get("pe_ratio", 0), benchmark
            )

        # ── Below 200MA bonus — ALL profiles (profile-aware scaling) ───
        if use_ma200_bonus:
            final_multiplier *= calculate_ma200_bonus(
                row.get("price", 0), row.get("ma_200", float("nan")),
                row.get("momentum_1y", 0.0),
                max_bonus=ma200_bonus_max, max_depth=ma200_bonus_max_depth
            )

        # ── Low P/E bonus — ALL profiles ──────────────────────────────
        if use_low_pe_bonus:
            final_multiplier *= calculate_low_pe_bonus(row.get("pe_ratio", 20.0))

        # ── PEG ratio bonus — ALL profiles ────────────────────────────
        if use_peg_bonus:
            final_multiplier *= calculate_peg_bonus(row.get("peg_ratio", 1.0))

        # ── Overextended momentum penalty — ALL profiles ──────────────
        if use_overextended_penalty:
            final_multiplier *= calculate_overextended_momentum_penalty(
                row.get("momentum_1mo", 0), row.get("perf_1w", 0)
            )

        # ── Momentum exhaustion filter — ALL profiles ─────────────────
        if use_exhaustion_filter:
            final_multiplier *= calculate_exhaustion_penalty(
                row.get("momentum_6mo", 0), row.get("momentum_1mo", 0)
            )

        # ── Sector Rotation Outperformance Bonus — ALL profiles ────────
        if use_sector_bonus:
            sec_3mo_val = sector_3mo.get(row["sector"], 0)
            sec_6mo_val = sector_6mo.get(row["sector"], 0)
            excess_3mo = max(0, row.get("momentum_3mo", 0) - sec_3mo_val)
            excess_6mo = max(0, row.get("momentum_6mo", 0) - sec_6mo_val)
            
            # Dynamic scaling: 1% outperformance = 1% bonus, capped per profile
            total_excess = excess_3mo + excess_6mo
            sector_bonus_mult = 1.0 + min(max_sector_bonus - 1.0, total_excess)
            final_multiplier *= sector_bonus_mult

        # ── Shorting Penalty ──────────────────────────────────────────
        if use_short_penalty:
            final_multiplier *= calculate_short_penalty(
                row.get("short_percent", 0), short_penalty_severity
            )

        # ── Volume Surge Bonus ────────────────────────────────────────
        if use_volume_surge:
            final_multiplier *= calculate_volume_surge_bonus(
                row.get("current_volume", 0), row.get("avg_volume", 0), volume_surge_max_bonus
            )

        scores.append(base_score * final_multiplier)

    df["recommendation_score"] = scores
    df["profile"] = user_profile
    df = df.sort_values(
        "recommendation_score", 
        ascending=False
    )

    if top_n is None:
        return df

    return df.head(top_n)

    df ["rebalance_date"] = pd.Timestamp.today().normalize()