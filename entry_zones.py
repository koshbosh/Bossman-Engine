"""Advisory price zones based on volatility and the production scoring equation."""

import math

import numpy as np

from recommender import calculate_ma_proximity


TRADING_DAYS_PER_YEAR = 252
TOUCH_HORIZON_DAYS = 21
ZONE_SIGMA_CENTERS = (0.5, 1.0, 1.5, 2.0)
MIN_RETURNS_FOR_TOUCH_ESTIMATE = 60


def hypothetical_ticker_data(ticker_data, close_history, price):
    """Rebuild price-sensitive inputs while holding other inputs fixed."""
    closes = np.asarray(close_history, dtype=float).copy()
    if closes.size < 20 or not np.isfinite(price) or price <= 0:
        raise ValueError("At least 20 valid closes and a positive hypothetical price are required.")

    closes[-1] = price
    n_days = len(closes)
    result = dict(ticker_data)
    result["price"] = float(price)
    result["perf_1w"] = float(price / closes[-5] - 1) if n_days >= 5 else 0.0
    result["momentum_1mo"] = float(price / closes[-21] - 1) if n_days >= 21 else None
    result["momentum_3mo"] = float(price / closes[-63] - 1) if n_days >= 63 else None
    result["momentum_6mo"] = float(price / closes[-126] - 1) if n_days >= 126 else None
    result["momentum_1y"] = float(price / closes[-252] - 1) if n_days >= 252 else None

    delta = np.diff(closes)
    if delta.size >= 14:
        recent_delta = delta[-14:]
        gain = np.maximum(recent_delta, 0).mean()
        loss = np.maximum(-recent_delta, 0).mean()
        result["rsi"] = float(100 - (100 / (1 + gain / (loss + 1e-9))))

    result["ma_50"] = float(closes[-50:].mean()) if n_days >= 50 else None
    result["ma_200"] = float(closes[-200:].mean()) if n_days >= 200 else None
    result["ma50_proximity"] = (
        calculate_ma_proximity(price, result["ma_50"]) if result["ma_50"] else None
    )
    result["ma200_proximity"] = (
        calculate_ma_proximity(price, result["ma_200"]) if result["ma_200"] else None
    )
    result["fifty_two_high"] = float(max(ticker_data.get("fifty_two_high") or closes.max(), price))
    result["fifty_two_low"] = float(min(ticker_data.get("fifty_two_low") or closes.min(), price))
    return result


def _simulate_zone_touch_probabilities(
    price,
    annualized_vol,
    daily_returns,
    daily_high_to_close,
    daily_low_to_close,
    simulations,
    seed,
):
    returns = np.asarray(daily_returns, dtype=float)
    high_ratios = np.asarray(daily_high_to_close, dtype=float)
    low_ratios = np.asarray(daily_low_to_close, dtype=float)
    valid = (
        np.isfinite(returns)
        & np.isfinite(high_ratios)
        & np.isfinite(low_ratios)
        & (high_ratios >= 1.0)
        & (low_ratios > 0.0)
        & (low_ratios <= 1.0)
    )
    returns = returns[valid]
    high_ratios = high_ratios[valid]
    low_ratios = low_ratios[valid]
    if returns.size < MIN_RETURNS_FOR_TOUCH_ESTIMATE:
        return None, "At least 60 daily returns are needed to estimate touch probabilities."

    sigma = annualized_vol * math.sqrt(TOUCH_HORIZON_DAYS / TRADING_DAYS_PER_YEAR)
    if not math.isfinite(sigma) or sigma <= 0:
        return None, "Volatility is unavailable or non-positive."

    rng = np.random.default_rng(seed)
    trailing_returns = returns[-252:]
    trailing_high_ratios = high_ratios[-252:]
    trailing_low_ratios = low_ratios[-252:]
    sampled_indexes = rng.integers(
        0,
        len(trailing_returns),
        size=(simulations, TOUCH_HORIZON_DAYS),
    )
    sampled_returns = trailing_returns[sampled_indexes] - trailing_returns.mean()
    paths = price * np.exp(np.cumsum(sampled_returns, axis=1))
    path_highs = paths * trailing_high_ratios[sampled_indexes]
    path_lows = paths * trailing_low_ratios[sampled_indexes]

    probabilities = {}
    for center in ZONE_SIGMA_CENTERS:
        low = price * math.exp(-(center + 0.25) * sigma)
        high = price * math.exp(-(center - 0.25) * sigma)
        touched = np.any((path_highs >= low) & (path_lows <= high), axis=1)
        probabilities[center] = float(touched.mean())
    return probabilities, None


def build_entry_zone_analysis(engine, ticker_data, profile_name, simulations=10000):
    """Build a separate advisory analysis; never mutates production score inputs."""
    close_history = ticker_data.get("_close_history")
    if not close_history or len(close_history) < 20:
        raise ValueError("Price history is unavailable for entry-zone analysis.")

    price = float(ticker_data["price"])
    annualized_vol = float(ticker_data["annualized_vol"])
    sigma_21d = annualized_vol * math.sqrt(TOUCH_HORIZON_DAYS / TRADING_DAYS_PER_YEAR)
    if not math.isfinite(sigma_21d) or sigma_21d <= 0:
        raise ValueError("A positive annualized volatility estimate is required.")

    base_score = engine.score_ticker(ticker_data, profile_name=profile_name)["score"]
    closes = np.asarray(close_history, dtype=float)
    high_history = ticker_data.get("_high_history")
    low_history = ticker_data.get("_low_history")
    if high_history is None or low_history is None:
        probabilities, probability_note = (
            None,
            "Adjusted daily high/low history is required to identify intraday band touches.",
        )
    else:
        highs = np.asarray(high_history, dtype=float)
        lows = np.asarray(low_history, dtype=float)
        if len(highs) != len(closes) or len(lows) != len(closes):
            probabilities, probability_note = (
                None,
                "Aligned adjusted daily OHLC history is unavailable.",
            )
        else:
            daily_returns = np.diff(np.log(closes))
            daily_high_to_close = highs[1:] / closes[1:]
            daily_low_to_close = lows[1:] / closes[1:]
            probabilities, probability_note = _simulate_zone_touch_probabilities(
                price,
                annualized_vol,
                daily_returns,
                daily_high_to_close,
                daily_low_to_close,
                simulations,
                seed=sum(ord(char) for char in ticker_data["ticker"]),
            )

    zones = []
    for center in ZONE_SIGMA_CENTERS:
        lower_sigma = center + 0.25
        upper_sigma = center - 0.25
        lower_price = price * math.exp(-lower_sigma * sigma_21d)
        upper_price = price * math.exp(-upper_sigma * sigma_21d)
        hypothetical_data = hypothetical_ticker_data(
            ticker_data,
            close_history,
            price * math.exp(-center * sigma_21d),
        )
        hypothetical_score = engine.score_ticker(
            hypothetical_data,
            profile_name=profile_name,
        )["score"]
        zones.append({
            "center_sigma": center,
            "lower_price": round(lower_price, 2),
            "upper_price": round(upper_price, 2),
            "reference_price": round(hypothetical_data["price"], 2),
            "touch_probability": (
                probabilities[center] if probabilities is not None else None
            ),
            "score": hypothetical_score,
            "score_change": round(hypothetical_score - base_score, 2),
            "expected_return": None,
            "combined_rank_value": None,
            "expected_return_rank": None,
            "combined_rank": None,
        })

    touchable_zones = [zone for zone in zones if zone["touch_probability"] is not None]
    for rank, zone in enumerate(
        sorted(touchable_zones, key=lambda item: item["touch_probability"], reverse=True),
        start=1,
    ):
        zone["touch_probability_rank"] = rank
    for rank, zone in enumerate(
        sorted(zones, key=lambda item: item["score"], reverse=True),
        start=1,
    ):
        zone["score_rank"] = rank
    for zone in zones:
        zone.setdefault("touch_probability_rank", None)

    best_zone = max(zones, key=lambda zone: zone["score"])
    return {
        "ticker": ticker_data["ticker"],
        "profile": profile_name,
        "spot_price": price,
        "current_score": base_score,
        "horizon_days": TOUCH_HORIZON_DAYS,
        "annualized_volatility": annualized_vol,
        "one_month_sigma_pct": round((math.exp(sigma_21d) - 1) * 100, 2),
        "zones": zones,
        "near_max_score_zone_sigma": best_zone["center_sigma"],
        "touch_probability_status": (
            "available" if probabilities is not None else "unavailable"
        ),
        "touch_probability_note": probability_note,
        "touch_probability_method": (
            "Bootstrap of aligned trailing adjusted daily returns and high/low "
            "ranges with zero drift; a touch is counted when a simulated daily "
            "bar intersects the band."
        ),
        "calibration": {
            "status": "unavailable",
            "reason": (
                "No profile-specific walk-forward score-to-return calibration is "
                "available. Expected return and combined ranking are intentionally "
                "not estimated."
            ),
            "candidate_horizons_days": [21, 63, 126, 252],
            "horizon_selection_method": (
                "Walk-forward mean monthly Spearman IC, with top-minus-bottom "
                "quintile spread as a secondary check."
            ),
        },
        "disclaimer": (
            "Advisory model-score and volatility zones only; not fair value, "
            "a guarantee of a fill, or a recommendation."
        ),
    }
