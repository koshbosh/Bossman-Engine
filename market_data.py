"""Validation helpers for live daily market-price histories."""

import numpy as np
import pandas as pd


MAX_PRICE_AGE_DAYS = 7


def prepare_daily_price_history(
    history,
    ticker,
    minimum_rows=20,
    max_age_days=MAX_PRICE_AGE_DAYS,
    now=None,
):
    """Return clean, ordered daily bars and reject missing or stale closes."""
    if history is None or history.empty or "Close" not in history.columns:
        raise ValueError(f"No daily closing-price history returned for '{ticker}'.")

    frame = history.copy()
    try:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid daily price dates returned for '{ticker}'.") from error

    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    valid_close = close.notna() & np.isfinite(close) & close.gt(0)
    frame = frame.loc[valid_close].copy()
    frame["Close"] = close.loc[valid_close]
    for column in ("Open", "High", "Low"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values.where(np.isfinite(values) & values.gt(0))
    if "Volume" in frame.columns:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        frame["Volume"] = volume.where(np.isfinite(volume) & volume.ge(0))

    if len(frame) < minimum_rows:
        raise ValueError(
            f"Insufficient valid daily prices for '{ticker}': "
            f"{len(frame)} rows; at least {minimum_rows} required."
        )

    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    latest_date = frame.index[-1].normalize()
    age_days = (current_time.normalize() - latest_date).days
    if age_days < -1:
        raise ValueError(
            f"Daily price history for '{ticker}' is dated in the future: "
            f"{latest_date.date().isoformat()}."
        )
    if age_days > max_age_days:
        raise ValueError(
            f"Latest daily close for '{ticker}' is stale: "
            f"{latest_date.date().isoformat()} ({age_days} days old)."
        )

    return frame
