"""Historical limit-order entry analysis for the Dip Hunter profile."""

import math

import numpy as np

from recommender import calculate_ma_proximity


TRADING_DAYS_PER_YEAR = 252
ORDER_OPEN_DAYS = 5
FORWARD_HORIZONS = (21, 42, 63, 90)
VOLATILITY_LOOKBACK_DAYS = 252
ZONE_SIGMA_CENTERS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
MIN_FILLED_SAMPLES_FOR_RANK = 30
ROUND_TRIP_COST_BPS = 10


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


def _historical_limit_order_estimates(
    closes,
    opens,
    highs,
    lows,
    zone_centers=ZONE_SIGMA_CENTERS,
    horizons=FORWARD_HORIZONS,
    order_open_days=ORDER_OPEN_DAYS,
    min_filled_samples=MIN_FILLED_SAMPLES_FOR_RANK,
    round_trip_cost_bps=ROUND_TRIP_COST_BPS,
):
    """Replay fixed-sigma entries using only preceding data and subsequent OHLC."""
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    if not (len(closes) == len(opens) == len(highs) == len(lows)):
        return None, "Aligned adjusted OHLC history is required."

    valid_prices = (
        np.isfinite(closes)
        & np.isfinite(opens)
        & np.isfinite(highs)
        & np.isfinite(lows)
        & (closes > 0)
        & (opens > 0)
        & (highs > 0)
        & (lows > 0)
    )
    if not valid_prices.all():
        return None, "Price history contains invalid OHLC observations."

    max_horizon = max(horizons)
    first_anchor = VOLATILITY_LOOKBACK_DAYS
    anchor_stop = len(closes) - order_open_days - max_horizon + 1
    if anchor_stop <= first_anchor:
        return None, (
            f"At least {VOLATILITY_LOOKBACK_DAYS + order_open_days + max_horizon + 1} "
            "sessions are required for the longest forward horizon."
        )

    opportunity_counts = {center: 0 for center in zone_centers}
    fills = {center: [] for center in zone_centers}
    for anchor in range(first_anchor, anchor_stop):
        prior_closes = closes[anchor - VOLATILITY_LOOKBACK_DAYS : anchor + 1]
        daily_returns = np.diff(np.log(prior_closes))
        annualized_volatility = float(np.std(daily_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if not math.isfinite(annualized_volatility) or annualized_volatility <= 0:
            continue

        sigma_21d = annualized_volatility * math.sqrt(21 / TRADING_DAYS_PER_YEAR)
        anchor_price = closes[anchor]
        for center in zone_centers:
            opportunity_counts[center] += 1
            trigger_price = anchor_price * math.exp(-center * sigma_21d)
            first_fill = None
            if center == 0:
                first_fill = (anchor + 1, opens[anchor + 1])
            elif center > 0:
                for fill_index in range(anchor + 1, anchor + order_open_days + 1):
                    if lows[fill_index] <= trigger_price:
                        fill_price = min(trigger_price, opens[fill_index])
                        first_fill = (fill_index, fill_price)
                        break
            else:
                for fill_index in range(anchor + 1, anchor + order_open_days + 1):
                    if opens[fill_index] >= trigger_price:
                        first_fill = (fill_index, opens[fill_index])
                        break
                    if highs[fill_index] >= trigger_price:
                        first_fill = (fill_index, trigger_price)
                        break
            if first_fill is not None:
                fills[center].append(first_fill)

    cost_fraction = round_trip_cost_bps / 10_000
    estimates = {}
    for center in zone_centers:
        opportunities = opportunity_counts[center]
        center_fills = fills[center]
        forward = {}
        for horizon in horizons:
            returns = np.asarray(
                [
                    closes[fill_index + horizon] / fill_price - 1
                    for fill_index, fill_price in center_fills
                ],
                dtype=float,
            )
            sufficient = len(returns) >= min_filled_samples
            if sufficient:
                mean_return = float(returns.mean())
                standard_error = (
                    float(returns.std(ddof=1) / math.sqrt(len(returns)))
                    if len(returns) > 1
                    else None
                )
                net_return = mean_return - cost_fraction
                fill_probability = len(center_fills) / opportunities if opportunities else None
                expected_value = fill_probability * net_return if fill_probability is not None else None
            else:
                mean_return = standard_error = net_return = expected_value = None
                fill_probability = len(center_fills) / opportunities if opportunities else None
            forward[horizon] = {
                "expected_return": round(mean_return * 100, 2) if mean_return is not None else None,
                "net_expected_return": round(net_return * 100, 2) if net_return is not None else None,
                "expected_value": round(expected_value * 100, 2) if expected_value is not None else None,
                "standard_error": round(standard_error * 100, 2) if standard_error is not None else None,
                "rank": None,
                "filled_samples": len(returns),
                "opportunity_count": opportunities,
                "fill_probability": round(fill_probability, 4) if fill_probability is not None else None,
                "status": "available" if sufficient else "insufficient_samples",
            }
        estimates[center] = forward

    for horizon in horizons:
        rankable = [
            (center, estimates[center][horizon]["expected_value"])
            for center in zone_centers
            if estimates[center][horizon]["expected_value"] is not None
        ]
        if len(rankable) >= 2:
            for rank, (center, _) in enumerate(
                sorted(rankable, key=lambda item: (-item[1], item[0])),
                start=1,
            ):
                estimates[center][horizon]["rank"] = rank

    return estimates, None


def build_entry_zone_analysis(engine, ticker_data, profile_name):
    """Build a per-ticker historical fill study; it does not alter the live score."""
    if profile_name != "Dip Hunter":
        raise ValueError("Historical entry estimates are currently available for Dip Hunter only.")

    close_history = ticker_data.get("_close_history")
    open_history = ticker_data.get("_open_history")
    high_history = ticker_data.get("_high_history")
    low_history = ticker_data.get("_low_history")
    if close_history is None or len(close_history) < 20:
        raise ValueError("Price history is unavailable for entry-zone analysis.")
    if open_history is None or high_history is None or low_history is None:
        raise ValueError("Aligned adjusted daily OHLC history is required for entry-zone analysis.")
    if not (len(close_history) == len(open_history) == len(high_history) == len(low_history)):
        raise ValueError("Aligned adjusted daily OHLC history is unavailable.")

    price = float(ticker_data["price"])
    annualized_vol = float(ticker_data["annualized_vol"])
    sigma_21d = annualized_vol * math.sqrt(21 / TRADING_DAYS_PER_YEAR)
    if not math.isfinite(sigma_21d) or sigma_21d <= 0:
        raise ValueError("A positive annualized volatility estimate is required.")

    historical_estimates, estimate_note = _historical_limit_order_estimates(
        close_history,
        open_history,
        high_history,
        low_history,
    )
    base_score = engine.score_ticker(ticker_data, profile_name=profile_name)["score"]
    zones = []
    for center in ZONE_SIGMA_CENTERS:
        band_edges = (
            price * math.exp(-(center + 0.25) * sigma_21d),
            price * math.exp(-(center - 0.25) * sigma_21d),
        )
        lower_price, upper_price = sorted(band_edges)
        reference_price = price * math.exp(-center * sigma_21d)
        hypothetical_data = hypothetical_ticker_data(
            ticker_data,
            close_history,
            reference_price,
        )
        hypothetical_score = engine.score_ticker(
            hypothetical_data,
            profile_name=profile_name,
        )["score"]
        horizons = (
            historical_estimates[center]
            if historical_estimates is not None
            else {
                horizon: {
                    "expected_return": None,
                    "net_expected_return": None,
                    "expected_value": None,
                    "standard_error": None,
                    "rank": None,
                    "filled_samples": 0,
                    "opportunity_count": 0,
                    "fill_probability": None,
                    "status": "unavailable",
                }
                for horizon in FORWARD_HORIZONS
            }
        )
        zones.append({
            "center_sigma": center,
            "entry_type": (
                "Buy stop" if center < 0
                else "Market at next open" if center == 0
                else "Limit buy"
            ),
            "lower_price": round(lower_price, 2),
            "upper_price": round(upper_price, 2),
            "reference_price": round(reference_price, 2),
            "score": hypothetical_score,
            "score_change": round(hypothetical_score - base_score, 2),
            "score_rank": None,
            "horizons": [
                {"horizon_days": horizon, **horizons[horizon]}
                for horizon in FORWARD_HORIZONS
            ],
        })

    for rank, zone in enumerate(
        sorted(zones, key=lambda item: (-item["score"], item["center_sigma"])),
        start=1,
    ):
        zone["score_rank"] = rank

    has_estimates = any(
        result["expected_return"] is not None
        for zone in zones
        for result in zone["horizons"]
    )
    calibration_reason = estimate_note or (
        "No zone has enough historical fills for a return estimate. "
        if not has_estimates
        else ""
    )
    calibration_reason += (
        "Estimates use this ticker's trailing history only and have not been "
        "validated in a separate walk-forward test. Daily order windows overlap, "
        "so the displayed standard errors are naive and are not confidence intervals. "
        "Horizon ranks are reported separately; no single horizon is selected as optimal."
    )
    return {
        "ticker": ticker_data["ticker"],
        "profile": profile_name,
        "spot_price": price,
        "current_score": base_score,
        "order_open_days": ORDER_OPEN_DAYS,
        "horizons_days": list(FORWARD_HORIZONS),
        "annualized_volatility": annualized_vol,
        "one_month_sigma_pct": round((math.exp(sigma_21d) - 1) * 100, 2),
        "zones": zones,
        "calibration": {
            "status": (
                "exploratory"
                if has_estimates
                else "insufficient_samples" if historical_estimates is not None
                else "insufficient_history"
            ),
            "scope": "single_ticker_historical_replay",
            "minimum_fills_for_estimate": MIN_FILLED_SAMPLES_FOR_RANK,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "reason": calibration_reason,
        },
        "method": (
            "Historical replay of candidate entries set at each sigma-center price. "
            "Below-spot limit buys fill at the limit or a better opening price after a "
            "downward gap. Above-spot buy stops fill at the trigger when the daily high "
            "reaches it, or at the opening price after an upward gap. The at-spot candidate "
            "is a market-at-next-open order. Limit and stop orders expire after five "
            "trading sessions. Returns are measured from the fill price to the adjusted "
            "close 21, 42, 63, or 90 sessions later. E(x) is the arithmetic mean gross "
            "return after fills; EV is fill probability times net E(x), less an assumed "
            "10 bp round-trip cost, with unfilled orders counted as zero."
        ),
        "disclaimer": (
            "Exploratory historical estimates, not a validated forecast or recommendation. "
            "Returns are based on one ticker's history and do not guarantee future fills "
            "or performance."
        ),
    }
