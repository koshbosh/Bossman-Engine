"""Run a point-in-time, regime-conditioned factor backtest.

Usage:
    python regime_factor_backtest.py

The default run starts in March 2007 and tests each month's factor ranks
against the following month's adjusted close-to-close returns. Historical
constituents are read from archived revisions of Wikipedia's S&P 500 list.
The archived page does not show ticker symbols in January or February 2007.
"""

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from factor_attribution import FACTORS_TO_TEST
from score_backtest import build_features_for_month


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REVISION_URL = "https://en.wikipedia.org/w/index.php?oldid={revision_id}"
USER_AGENT = "BossmanEngineResearch/1.0 (historical backtest)"
DEFAULT_START = "2007-03"
PRICE_LOOKBACK_START = "2006-01-01"
PRICE_CHUNK_SIZE = 100
PRICE_CACHE_DIR = Path.home() / ".cache" / "bossman_engine_regime_backtest"
OUTPUT_DIR = Path("backtest_results")
REGIME_MONTHLY_PATH = OUTPUT_DIR / "regime_factor_regimes_monthly.csv"
FACTOR_MONTHLY_PATH = OUTPUT_DIR / "regime_factor_factor_monthly.csv"
SUMMARY_PATH = OUTPUT_DIR / "regime_factor_summary.csv"
METADATA_PATH = OUTPUT_DIR / "regime_factor_metadata.json"


def classify_regime(breadth_50, vix, rsp_outperforming):
    """Match the live regime state machine's thresholds and branch order."""
    if vix >= 55.0:
        return "Systemic Crisis (VIX > 55 Override)"
    if vix >= 35.0:
        return "Panic Capitulation (VIX > 35 Override)"
    if breadth_50 < 35.0:
        return "Oversold Breadth"
    if breadth_50 < 50.0:
        if rsp_outperforming:
            return "Selective Consolidation"
        return "Narrow Breadth Divergence"
    if rsp_outperforming:
        return "Broad Market Expansion"
    return "Bull Momentum Expansion"


def _cache_path(cache_dir, month):
    return cache_dir / f"constituents_{month}.json"


def _request_wikipedia_json(session, params):
    for attempt in range(6):
        response = session.get(WIKIPEDIA_API, params=params, timeout=30)
        if response.status_code in {429, 503} and attempt < 5:
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2**attempt)
            continue
        response.raise_for_status()
        payload = response.json()
        api_error = payload.get("error")
        if not api_error:
            time.sleep(1.0)
            return payload
        if api_error.get("code") not in {"maxlag", "ratelimited"} or attempt == 5:
            raise RuntimeError(f"Wikipedia API error: {api_error}")
        retry_after = response.headers.get("Retry-After")
        time.sleep(float(retry_after) if retry_after else 2**attempt)
    raise RuntimeError("Wikipedia API retry limit exceeded.")


def parse_constituent_tickers(wikitext):
    """Extract tickers from archived table markup across Wikipedia formats."""
    symbols = set(
        re.findall(
            r"\{\{\s*[A-Za-z0-9 _-]*Symbol\s*\|\s*([A-Z0-9.-]+)",
            wikitext,
            flags=re.IGNORECASE,
        )
    )
    if len(symbols) < 400:
        for line in wikitext.splitlines():
            if not line.startswith("|") or "||" not in line:
                continue
            cells = line.split("||")
            for cell in cells[:2]:
                ticker = cell.strip().lstrip("|").strip()
                if re.fullmatch(r"[A-Z0-9.-]+", ticker):
                    symbols.add(ticker)

    tickers = sorted(
        {
            symbol.upper().replace(".", "-")
            for symbol in symbols
            if re.fullmatch(r"[A-Z0-9.-]+", symbol, flags=re.IGNORECASE)
        }
    )
    return tickers


def _fetch_revision_catalog(months, session):
    first_month_start = months[0].start_time.normalize()
    last_month_end = months[-1].end_time.normalize()
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": "List of S&P 500 companies",
        "rvprop": "ids|timestamp",
        "rvstart": f"{last_month_end:%Y-%m-%d}T23:59:59Z",
        "rvend": f"{first_month_start:%Y-%m-%d}T00:00:00Z",
        "rvlimit": 500,
        "rvdir": "older",
        "format": "json",
        "maxlag": 5,
    }
    revisions = []
    while True:
        payload = _request_wikipedia_json(session, params)
        page = next(iter(payload.get("query", {}).get("pages", {}).values()), {})
        revisions.extend(page.get("revisions", []))
        continuation = payload.get("continue", {}).get("rvcontinue")
        if not continuation:
            break
        params["rvcontinue"] = continuation

    revisions.sort(key=lambda revision: revision["timestamp"])
    return revisions


def _fetch_revision_wikitext(revision_id, session):
    payload = _request_wikipedia_json(
        session,
        {
            "action": "query",
            "prop": "revisions",
            "revids": str(revision_id),
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "maxlag": 5,
        },
    )
    for page in payload.get("query", {}).get("pages", {}).values():
        for revision in page.get("revisions", []):
            if revision["revid"] == revision_id:
                slots = revision.get("slots", {}).get("main", {})
                return slots.get("*", slots.get("content", ""))
    return ""


def fetch_constituent_history(months, cache_dir, force_refresh=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    snapshots = {}
    missing_months = []
    for period in months:
        month = str(period)
        path = _cache_path(cache_dir, month)
        if path.exists() and not force_refresh:
            snapshots[month] = json.loads(path.read_text())
        else:
            missing_months.append(period)

    if not missing_months:
        return snapshots

    revisions = _fetch_revision_catalog(missing_months, session)
    monthly_revision = {}
    for period in missing_months:
        month_end = period.end_time.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        eligible = [
            revision
            for revision in revisions
            if pd.Timestamp(revision["timestamp"]).tz_localize(None) <= month_end
        ]
        if not eligible:
            raise RuntimeError(f"No archived S&P 500 revision found for {period}.")
        monthly_revision[str(period)] = eligible[-1]

    unique_revisions = {
        revision["revid"]: revision for revision in monthly_revision.values()
    }
    contents = {}
    revision_ids = list(unique_revisions)
    for start in range(0, len(revision_ids), 50):
        batch = revision_ids[start : start + 50]
        payload = _request_wikipedia_json(
            session,
            {
                "action": "query",
                "prop": "revisions",
                "revids": "|".join(map(str, batch)),
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "format": "json",
                "maxlag": 5,
            },
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            for revision in page.get("revisions", []):
                slots = revision.get("slots", {}).get("main", {})
                contents[revision["revid"]] = slots.get("*", slots.get("content", ""))

    for index, period in enumerate(missing_months, start=1):
        month = str(period)
        revision = monthly_revision[month]
        tickers = parse_constituent_tickers(contents.get(revision["revid"], ""))
        if len(tickers) < 400:
            tickers = parse_constituent_tickers(
                _fetch_revision_wikitext(revision["revid"], session)
            )
        if len(tickers) < 400:
            raise RuntimeError(
                f"Archived S&P 500 revision {revision['revid']} for {month} has only "
                f"{len(tickers)} tickers; "
                "the page did not contain a usable ticker table. The archived list "
                "first provides ticker symbols in March 2007; start there or later."
            )
        snapshot = {
            "month": month,
            "revision_id": revision["revid"],
            "revision_timestamp": revision["timestamp"],
            "tickers": tickers,
        }
        _cache_path(cache_dir, month).write_text(json.dumps(snapshot))
        snapshots[month] = snapshot
        print(
            f"Loaded constituent snapshot {index}/{len(missing_months)}: {month} "
            f"({len(snapshots[month]['tickers'])} tickers, "
            f"revision {snapshots[month]['revision_id']})"
        )
    return snapshots


def download_close_prices(tickers, start_date, end_date, cache_dir, force_refresh=False):
    """Download adjusted close prices in batches and cache the combined panel."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"adjusted_close_{start_date}_{end_date}.pkl"
    symbols = sorted(set(tickers) | {"SPY", "RSP", "^VIX"})
    if cache_path.exists() and not force_refresh:
        print(f"Loading cached adjusted closes from {cache_path}")
        close_prices = pd.read_pickle(cache_path)
        return close_prices, sorted(set(symbols) - set(close_prices.columns))

    close_series = {}
    failed_symbols = []
    total_chunks = math.ceil(len(symbols) / PRICE_CHUNK_SIZE)

    for chunk_index, start in enumerate(range(0, len(symbols), PRICE_CHUNK_SIZE), start=1):
        chunk = symbols[start : start + PRICE_CHUNK_SIZE]
        print(f"Downloading price chunk {chunk_index}/{total_chunks} ({len(chunk)} symbols)")
        try:
            prices = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            failed_symbols.extend(chunk)
            print(f"Price download failed for chunk {chunk_index}: {exc}")
            continue

        if prices.empty:
            failed_symbols.extend(chunk)
            continue

        for symbol in chunk:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    series = prices[symbol]["Close"]
                elif len(chunk) == 1 and "Close" in prices.columns:
                    series = prices["Close"]
                else:
                    failed_symbols.append(symbol)
                    continue
                series = series.dropna()
                if series.empty:
                    failed_symbols.append(symbol)
                else:
                    close_series[symbol] = series.rename(symbol)
            except (KeyError, TypeError):
                failed_symbols.append(symbol)

    if not {"SPY", "RSP", "^VIX"}.issubset(close_series):
        missing = sorted({"SPY", "RSP", "^VIX"} - set(close_series))
        raise RuntimeError(f"Required regime price history is missing: {', '.join(missing)}")

    close_prices = pd.concat(close_series.values(), axis=1).sort_index()
    close_prices = close_prices[~close_prices.index.duplicated(keep="last")]
    close_prices.to_pickle(cache_path)
    return close_prices, sorted(set(failed_symbols))


def calculate_regime_inputs(date, point_in_time_tickers, close_prices):
    """Compute breadth, VIX, and RSP/SPY using observations available by date."""
    history = close_prices.loc[:date]
    breadth_values = []
    for ticker in point_in_time_tickers:
        if ticker not in history.columns:
            continue
        prices = history[ticker].dropna()
        if len(prices) >= 50:
            breadth_values.append(float(prices.iloc[-1] > prices.iloc[-50:].mean()))

    if not breadth_values:
        return None
    breadth_50 = float(np.mean(breadth_values) * 100)

    vix_history = history["^VIX"].dropna()
    if vix_history.empty:
        return None
    vix = float(vix_history.iloc[-1])

    common_macro_dates = history[["RSP", "SPY"]].dropna()
    if len(common_macro_dates) < 21:
        return None
    rsp_return = float(
        common_macro_dates["RSP"].iloc[-1] / common_macro_dates["RSP"].iloc[-21] - 1
    )
    spy_return = float(
        common_macro_dates["SPY"].iloc[-1] / common_macro_dates["SPY"].iloc[-21] - 1
    )
    rsp_spy_diff = (rsp_return - spy_return) * 100
    regime = classify_regime(breadth_50, vix, rsp_return > spy_return)
    return {
        "breadth_50": breadth_50,
        "breadth_50_coverage": len(breadth_values),
        "vix": vix,
        "rsp_spy_diff_pct": rsp_spy_diff,
        "regime": regime,
    }


def score_month_factors(
    date,
    next_date,
    point_in_time_tickers,
    close_prices,
    monthly_prices,
    spy_returns,
):
    """Calculate equal-weight next-month quintile returns for each raw factor."""
    history = close_prices.loc[:date]
    features = build_features_for_month(
        date, history, spy_returns, point_in_time_tickers, sector_map={}
    )
    if features.empty:
        return []

    current_prices = monthly_prices.loc[date]
    next_prices = monthly_prices.loc[next_date]
    records = []
    for factor in FACTORS_TO_TEST:
        if factor not in features.columns:
            continue
        factor_data = features[["ticker", factor]].copy()
        factor_data["forward_return"] = factor_data["ticker"].map(
            lambda ticker: (
                next_prices.get(ticker, np.nan)
                / current_prices.get(ticker, np.nan)
                - 1
                if ticker in close_prices.columns
                and pd.notna(next_prices.get(ticker, np.nan))
                and pd.notna(current_prices.get(ticker, np.nan))
                and current_prices.get(ticker, 0) > 0
                else np.nan
            )
        )
        factor_data = factor_data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[factor, "forward_return"]
        )
        if len(factor_data) < 5 or factor_data[factor].nunique() < 5:
            continue
        factor_data["bin"] = pd.qcut(
            factor_data[factor], q=5, labels=False, duplicates="drop"
        )
        if factor_data["bin"].nunique() != 5:
            continue

        quintile_returns = factor_data.groupby("bin")["forward_return"].mean()
        if len(quintile_returns) != 5:
            continue
        q5_lowest = float(quintile_returns.loc[0])
        q4 = float(quintile_returns.loc[1])
        q3 = float(quintile_returns.loc[2])
        q2 = float(quintile_returns.loc[3])
        q1_highest = float(quintile_returns.loc[4])
        records.append(
            {
                "month": date.strftime("%Y-%m"),
                "factor": factor,
                "n_stocks": int(len(factor_data)),
                "q1_highest_return": q1_highest,
                "q2_return": q2,
                "q3_return": q3,
                "q4_return": q4,
                "q5_lowest_return": q5_lowest,
                "spread_q1_minus_q5": q1_highest - q5_lowest,
                "spearman_ic": float(
                    factor_data[factor].rank().corr(
                        factor_data["forward_return"].rank(), method="pearson"
                    )
                ),
            }
        )
    return records


def _monotonicity(quintile_means):
    low_to_high = [
        quintile_means["q5_lowest_return"],
        quintile_means["q4_return"],
        quintile_means["q3_return"],
        quintile_means["q2_return"],
        quintile_means["q1_highest_return"],
    ]
    if all(left < right for left, right in zip(low_to_high, low_to_high[1:])):
        return "Strictly increasing"
    if all(left > right for left, right in zip(low_to_high, low_to_high[1:])):
        return "Strictly decreasing"
    if low_to_high[-1] > low_to_high[0]:
        return "Generally increasing"
    if low_to_high[-1] < low_to_high[0]:
        return "Generally decreasing"
    return "Mixed or flat"


def summarize_results(factor_monthly, regime_month_counts):
    if factor_monthly.empty:
        return pd.DataFrame()
    quintile_columns = [
        "q1_highest_return",
        "q2_return",
        "q3_return",
        "q4_return",
        "q5_lowest_return",
    ]
    summaries = []
    for (regime, factor), group in factor_monthly.groupby(["regime", "factor"]):
        means = group[quintile_columns].mean()
        spread = group["spread_q1_minus_q5"]
        spread_std = spread.std(ddof=1)
        spread_t = (
            float(spread.mean() / (spread_std / np.sqrt(len(spread))))
            if len(spread) > 1 and spread_std > 0
            else np.nan
        )
        summaries.append(
            {
                "regime": regime,
                "factor": factor,
                "regime_months": int(regime_month_counts.get(regime, 0)),
                "months_with_factor_data": int(len(group)),
                "mean_q1_highest_return": float(means["q1_highest_return"]),
                "mean_q2_return": float(means["q2_return"]),
                "mean_q3_return": float(means["q3_return"]),
                "mean_q4_return": float(means["q4_return"]),
                "mean_q5_lowest_return": float(means["q5_lowest_return"]),
                "mean_q1_minus_q5_spread": float(spread.mean()),
                "median_monthly_spread": float(spread.median()),
                "mean_monthly_spearman_ic": float(group["spearman_ic"].mean()),
                "positive_spread_months": int((spread > 0).sum()),
                "spread_t_stat_unadjusted": spread_t,
                "mean_quintile_monotonicity": _monotonicity(means),
            }
        )
    return pd.DataFrame(summaries).sort_values(["regime", "factor"])


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="First month to score (YYYY-MM); defaults to January 2007.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload archived constituents and price data instead of using local caches.",
    )
    return parser.parse_args()


def run_backtest(start_month=DEFAULT_START, force_refresh=False):
    start_period = pd.Period(start_month, freq="M")
    last_complete_return_period = pd.Timestamp.now().to_period("M") - 1
    last_score_period = last_complete_return_period - 1
    if start_period > last_score_period:
        raise ValueError(
            f"Start month {start_period} is later than the last month with a completed "
            "following-month return."
        )

    periods = pd.period_range(start_period, last_score_period, freq="M")
    last_data_end = (last_complete_return_period + 1).start_time.strftime("%Y-%m-%d")
    snapshots = fetch_constituent_history(
        periods, PRICE_CACHE_DIR / "constituents", force_refresh=force_refresh
    )
    all_tickers = {
        ticker for snapshot in snapshots.values() for ticker in snapshot["tickers"]
    }
    prices, failed_symbols = download_close_prices(
        all_tickers,
        PRICE_LOOKBACK_START,
        last_data_end,
        PRICE_CACHE_DIR,
        force_refresh=force_refresh,
    )
    monthly_prices = prices.resample("ME").last()
    spy_returns = prices["SPY"].pct_change().dropna()

    regime_monthly_records = []
    factor_monthly_records = []
    regime_month_counts = {}
    for period in periods:
        month = str(period)
        date = monthly_prices.index[monthly_prices.index.to_period("M") == period]
        next_period = period + 1
        next_date = monthly_prices.index[monthly_prices.index.to_period("M") == next_period]
        if len(date) != 1 or len(next_date) != 1:
            print(f"Skipping {month}: month-end price data unavailable.")
            continue
        date = date[0]
        next_date = next_date[0]
        snapshot = snapshots[month]
        point_in_time_tickers = snapshot["tickers"]
        regime_inputs = calculate_regime_inputs(date, point_in_time_tickers, prices)
        if regime_inputs is None:
            print(f"Skipping {month}: required regime inputs are unavailable.")
            continue

        history = prices.loc[:date]
        breadth_coverage_ratio = (
            regime_inputs["breadth_50_coverage"] / len(point_in_time_tickers)
        )
        regime_monthly_records.append(
            {
                "month": month,
                **regime_inputs,
                "constituent_count": len(point_in_time_tickers),
                "breadth_50_coverage_ratio": breadth_coverage_ratio,
                "price_coverage_count": sum(
                    ticker in history.columns and history[ticker].dropna().shape[0] >= 50
                    for ticker in point_in_time_tickers
                ),
                "wikipedia_revision_id": snapshot["revision_id"],
                "wikipedia_revision_timestamp": snapshot["revision_timestamp"],
            }
        )
        regime_month_counts[regime_inputs["regime"]] = (
            regime_month_counts.get(regime_inputs["regime"], 0) + 1
        )

        monthly_factors = score_month_factors(
            date,
            next_date,
            point_in_time_tickers,
            prices,
            monthly_prices,
            spy_returns,
        )
        for record in monthly_factors:
            factor_monthly_records.append(
                {"regime": regime_inputs["regime"], **record}
            )
        print(
            f"Scored {month}: {regime_inputs['regime']}; "
            f"{len(point_in_time_tickers)} constituents, "
            f"{sum(record['n_stocks'] for record in monthly_factors)} "
            "factor-stock observations"
        )

    regime_monthly = pd.DataFrame(regime_monthly_records)
    factor_monthly = pd.DataFrame(factor_monthly_records)
    summary = summarize_results(factor_monthly, regime_month_counts)
    if regime_monthly.empty or factor_monthly.empty or summary.empty:
        raise RuntimeError("Backtest produced no usable regime/factor results.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regime_monthly.to_csv(REGIME_MONTHLY_PATH, index=False)
    factor_monthly.to_csv(FACTOR_MONTHLY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_month_requested": str(start_period),
        "last_scoring_month": str(periods[-1]),
        "last_completed_return_month": str(last_complete_return_period),
        "factor_names": FACTORS_TO_TEST,
        "regime_rules": {
            "source": "engine_service.EngineService.refresh_macro_and_sector_regimes",
            "vix_crisis_threshold": 55.0,
            "vix_panic_threshold": 35.0,
            "breadth_oversold_threshold_pct": 35.0,
            "breadth_consolidation_threshold_pct": 50.0,
            "breadth_method": "point-in-time constituent close above trailing 50-session mean",
            "rsp_spy_method": "trailing 21 common daily adjusted closes; RSP > SPY",
        },
        "return_method": "next calendar month's adjusted month-end close / current adjusted month-end close - 1",
        "constituent_source": "archived Wikipedia List of S&P 500 companies revisions",
        "first_month_with_archived_ticker_symbols": "2007-03",
        "price_source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "adjusted_prices": True,
        "months_with_regime_data": int(len(regime_monthly)),
        "unique_historical_constituent_tickers": len(all_tickers),
        "breadth_coverage_ratio_minimum": float(
            regime_monthly["breadth_50_coverage_ratio"].min()
        ),
        "breadth_coverage_ratio_median": float(
            regime_monthly["breadth_50_coverage_ratio"].median()
        ),
        "monthly_factor_universe_size_minimum": int(
            factor_monthly.groupby("month")["n_stocks"].min().min()
        ),
        "monthly_factor_universe_size_median": int(
            factor_monthly.groupby("month")["n_stocks"].min().median()
        ),
        "months_with_factor_data": {
            f"{regime}|{factor}": int(len(group))
            for (regime, factor), group in factor_monthly.groupby(["regime", "factor"])
        },
        "missing_price_symbols_count": len(failed_symbols),
        "missing_price_symbols_sample": failed_symbols[:100],
        "outputs": [
            str(REGIME_MONTHLY_PATH),
            str(FACTOR_MONTHLY_PATH),
            str(SUMMARY_PATH),
        ],
        "limitations": [
            "Archived Wikipedia membership snapshots are a historical proxy, not an official point-in-time index file.",
            "Yahoo Finance may omit or have incomplete adjusted-price histories for delisted or renamed constituents.",
            "Historical constituent price coverage varies over time; early-period breadth can be biased by unavailable delisted-stock prices.",
            "The unadjusted spread t-statistic is descriptive and does not correct for serial dependence or multiple testing.",
            "This is factor-level association analysis, not a portfolio simulation or causal test.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Summary saved to {SUMMARY_PATH}")
    print(f"Monthly factor results saved to {FACTOR_MONTHLY_PATH}")
    print(f"Monthly regime inputs saved to {REGIME_MONTHLY_PATH}")
    print(f"Run metadata saved to {METADATA_PATH}")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(start_month=args.start, force_refresh=args.force_refresh)
