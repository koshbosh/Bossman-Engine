"""Audit historical S&P 500 tickers absent from the cached price panel."""

import argparse
import json
from pathlib import Path

import pandas as pd

from regime_factor_backtest import PRICE_CACHE_DIR, OUTPUT_DIR


PRICE_CACHE_NAME = "adjusted_close_2006-01-01_2026-10-01.pkl"
CONSTITUENT_CACHE_DIR = PRICE_CACHE_DIR / "constituents"
AUDIT_PATH = OUTPUT_DIR / "regime_factor_missing_ticker_audit.csv"
BENCHMARK_SYMBOLS = {"SPY", "RSP", "^VIX"}

# These are data-provider symbol spellings to verify, not automatic mappings.
VERIFIED_FORMAT_CANDIDATES = {
    "BFB": "BF-B",
    "BRKB": "BRK-B",
}


def classify_missing_ticker(ticker, is_latest_constituent):
    if ticker in VERIFIED_FORMAT_CANDIDATES:
        return (
            "Potential ticker-format repair",
            VERIFIED_FORMAT_CANDIDATES[ticker],
        )
    if is_latest_constituent:
        return "Latest constituent; provider/history gap needs investigation", ""
    return "No returned series; likely delisted/renamed or provider coverage gap", ""


def build_missing_ticker_audit(cache_dir=PRICE_CACHE_DIR):
    price_path = cache_dir / PRICE_CACHE_NAME
    constituent_dir = cache_dir / "constituents"
    if not price_path.exists():
        raise FileNotFoundError(f"Price cache not found: {price_path}")
    if not constituent_dir.exists():
        raise FileNotFoundError(f"Constituent cache not found: {constituent_dir}")

    prices = pd.read_pickle(price_path)
    constituent_files = sorted(constituent_dir.glob("constituents_*.json"))
    if not constituent_files:
        raise FileNotFoundError(f"No constituent snapshots found in {constituent_dir}")

    first_seen = {}
    last_seen = {}
    membership_months = {}
    latest_tickers = set()
    latest_month = ""
    for path in constituent_files:
        snapshot = json.loads(path.read_text())
        month = snapshot["month"]
        if month > latest_month:
            latest_month = month
            latest_tickers = set(snapshot["tickers"])
        for ticker in set(snapshot["tickers"]):
            membership_months[ticker] = membership_months.get(ticker, 0) + 1
            first_seen.setdefault(ticker, month)
            last_seen[ticker] = month

    available = set(prices.columns) - BENCHMARK_SYMBOLS
    records = []
    for ticker in sorted(set(membership_months) - available):
        is_latest = ticker in latest_tickers
        classification, candidate = classify_missing_ticker(ticker, is_latest)
        records.append(
            {
                "ticker": ticker,
                "membership_months": membership_months[ticker],
                "first_seen_month": first_seen[ticker],
                "last_seen_month": last_seen[ticker],
                "in_latest_archived_constituents": is_latest,
                "price_series_available": False,
                "classification": classification,
                "candidate_provider_symbol_to_verify": candidate,
            }
        )
    return pd.DataFrame(records), latest_month, len(available)


def run_audit(cache_dir=PRICE_CACHE_DIR, output_path=AUDIT_PATH):
    audit, latest_month, priced_symbol_count = build_missing_ticker_audit(cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    print(f"Latest archived constituent snapshot: {latest_month}")
    print(f"Historical constituent symbols with price series: {priced_symbol_count}")
    print(f"Historical constituent symbols without price series: {len(audit)}")
    if audit.empty:
        print("No missing constituent price series found.")
        return audit

    print("\nMissing-symbol classification:")
    print(audit["classification"].value_counts().to_string())
    print("\nMost persistent missing constituents:")
    print(
        audit.sort_values("membership_months", ascending=False)
        .head(20)[
            [
                "ticker",
                "membership_months",
                "first_seen_month",
                "last_seen_month",
                "in_latest_archived_constituents",
                "classification",
                "candidate_provider_symbol_to_verify",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nDetailed audit saved to {output_path}")
    return audit


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PRICE_CACHE_DIR,
        help="Backtest cache directory (defaults to the research cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_PATH,
        help="Path for the generated missing-ticker CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_audit(cache_dir=args.cache_dir, output_path=args.output)
