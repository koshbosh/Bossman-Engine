"""Download point-in-time constituent adjusted OHLC into the local research cache.

Usage:
    python acquire_historical_ohlc.py
    python acquire_historical_ohlc.py --start 2006-01-01 --end 2026-10-01

Data is stored under ~/.cache/bossman_engine_regime_backtest, not in the
repository. Yahoo Finance availability for delisted or renamed tickers is not
guaranteed; the output manifest reports observed coverage and missing symbols.
"""

import argparse
import hashlib
import json
import time

import pandas as pd
import yfinance as yf

from regime_factor_backtest import PRICE_CACHE_DIR


DEFAULT_START = "2006-01-01"
DEFAULT_END = "2026-10-01"
DEFAULT_CHUNK_SIZE = 50
PROVIDER_SYMBOL_ALIASES = {
    "BFB": "BF-B",
    "BRKB": "BRK-B",
}
OHLC_FIELDS = ("Open", "High", "Low", "Close")
CONSTITUENT_CACHE_DIR = PRICE_CACHE_DIR / "constituents"
OHLC_CACHE_DIR = PRICE_CACHE_DIR / "historical_ohlc"


def load_historical_tickers(constituent_cache_dir=CONSTITUENT_CACHE_DIR):
    """Return the union of tickers from the cached archived constituent sets."""
    paths = sorted(constituent_cache_dir.glob("constituents_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No archived constituent snapshots found in {constituent_cache_dir}."
        )

    tickers = set()
    for path in paths:
        snapshot = json.loads(path.read_text())
        tickers.update(str(ticker).strip().upper() for ticker in snapshot["tickers"])
    return sorted(tickers)


def _cache_key(start, end, tickers, chunk_size):
    payload = json.dumps(
        {
            "start": start,
            "end": end,
            "tickers": tickers,
            "chunk_size": chunk_size,
            "provider_symbol_aliases": PROVIDER_SYMBOL_ALIASES,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalize_downloaded_batch(downloaded, requested_tickers):
    """Normalize yfinance's ticker-first MultiIndex output into ticker panels."""
    panels = {}
    if downloaded.empty:
        return panels
    if not isinstance(downloaded.columns, pd.MultiIndex):
        if len(requested_tickers) != 1:
            return panels
        ticker = requested_tickers[0]
        available = set(downloaded.columns)
        fields = [field for field in OHLC_FIELDS if field in available]
        if len(fields) == len(OHLC_FIELDS):
            panels[ticker] = downloaded[fields].copy()
        return panels

    first_level = downloaded.columns.get_level_values(0)
    second_level = downloaded.columns.get_level_values(1)
    for ticker in requested_tickers:
        if ticker in first_level:
            panel = downloaded[ticker]
        elif ticker in second_level:
            panel = downloaded.xs(ticker, axis=1, level=1)
        else:
            continue
        if not all(field in panel.columns for field in OHLC_FIELDS):
            continue
        panel = panel.loc[:, list(OHLC_FIELDS)].copy()
        panel = panel.dropna(how="all")
        if not panel.empty:
            panels[ticker] = panel
    return panels


def download_adjusted_ohlc(
    tickers,
    start=DEFAULT_START,
    end=DEFAULT_END,
    chunk_size=DEFAULT_CHUNK_SIZE,
    cache_dir=OHLC_CACHE_DIR,
    force_refresh=False,
):
    """Fetch adjusted OHLC in resumable chunks and return the joined panel."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        ticker: PROVIDER_SYMBOL_ALIASES.get(ticker, ticker)
        for ticker in tickers
    }
    requests = sorted(set(aliases.values()) | {"SPY", "RSP", "^VIX"})
    key = _cache_key(start, end, tickers, chunk_size)
    chunk_dir = cache_dir / key
    chunk_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    panels = {}
    batches = [
        requests[offset:offset + chunk_size]
        for offset in range(0, len(requests), chunk_size)
    ]

    for batch_number, provider_symbols in enumerate(batches, start=1):
        chunk_path = chunk_dir / f"chunk_{batch_number:03d}.pkl"
        if chunk_path.exists() and not force_refresh:
            batch_panels = pd.read_pickle(chunk_path)
            print(f"Using cached OHLC chunk {batch_number}/{len(batches)}")
        else:
            print(
                f"Downloading adjusted OHLC chunk {batch_number}/{len(batches)} "
                f"({len(provider_symbols)} symbols)"
            )
            downloaded = yf.download(
                provider_symbols,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=False,
            )
            batch_panels = _normalize_downloaded_batch(downloaded, provider_symbols)
            pd.to_pickle(batch_panels, chunk_path)
            time.sleep(0.5)

        panels.update(batch_panels)
        failures.extend(symbol for symbol in provider_symbols if symbol not in batch_panels)

    original_panels = {}
    for original, provider in aliases.items():
        if provider in panels:
            original_panels[original] = panels[provider]
    for symbol in ("SPY", "RSP", "^VIX"):
        if symbol in panels:
            original_panels[symbol] = panels[symbol]
    if not original_panels:
        raise RuntimeError("Yahoo Finance returned no adjusted OHLC series.")

    panel = pd.concat(
        original_panels,
        axis=1,
        names=["ticker", "field"],
        sort=True,
    ).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel.to_pickle(cache_dir / f"adjusted_ohlc_{start}_{end}.pkl")

    missing_original = [
        ticker
        for ticker, provider in aliases.items()
        if provider not in panels
    ]
    coverage_by_year = {}
    for year, year_panel in panel.groupby(panel.index.year):
        ticker_counts = (
            year_panel.notna()
            .T.groupby(level="ticker")
            .any()
            .T.sum(axis=1)
        )
        coverage_by_year[str(year)] = {
            "sessions": int(len(year_panel)),
            "median_symbols_with_ohlc": int(ticker_counts.median()),
            "minimum_symbols_with_ohlc": int(ticker_counts.min()),
        }

    manifest = {
        "source": "Yahoo Finance via yfinance",
        "adjusted": True,
        "start_date": start,
        "end_date_exclusive": end,
        "historical_constituent_ticker_count": len(tickers),
        "historical_constituent_series_found": len(tickers) - len(missing_original),
        "historical_constituent_series_missing": len(missing_original),
        "missing_tickers": missing_original,
        "provider_symbol_aliases": {
            original: provider
            for original, provider in PROVIDER_SYMBOL_ALIASES.items()
            if original in tickers
        },
        "panel_path": str(cache_dir / f"adjusted_ohlc_{start}_{end}.pkl"),
        "panel_rows": int(len(panel)),
        "panel_tickers": int(panel.columns.get_level_values("ticker").nunique()),
        "panel_fields": list(OHLC_FIELDS),
        "coverage_by_year": coverage_by_year,
        "non_historical_ticker_failures": sorted(
            symbol for symbol in failures if symbol not in set(tickers)
        ),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Saved {len(original_panels)} ticker panels / {len(panel)} sessions. "
        f"Historical tickers missing OHLC: {len(missing_original)}."
    )
    print(f"Manifest: {cache_dir / 'manifest.json'}")
    return panel, manifest


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    tickers = load_historical_tickers()
    download_adjusted_ohlc(
        tickers,
        start=args.start,
        end=args.end,
        chunk_size=args.chunk_size,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
