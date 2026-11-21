"""Acquire SEC company facts for historical-universe symbols with current CIK links.

The SEC contact email is supplied at run time and is not written to source or
metadata:
    SEC_CONTACT_EMAIL=you@example.com python acquire_sec_company_facts.py

Raw CompanyFacts JSON is gzip-compressed under the local research cache. The
SEC's ticker map covers current symbols only; it cannot identify every
delisted historical constituent.
"""

import argparse
import gzip
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

from acquire_historical_ohlc import (
    PROVIDER_SYMBOL_ALIASES,
    load_historical_tickers,
)
from regime_factor_backtest import PRICE_CACHE_DIR


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_CACHE_DIR = PRICE_CACHE_DIR / "sec_companyfacts"
FACTS_DIR = SEC_CACHE_DIR / "facts"
MIN_REQUEST_INTERVAL_SECONDS = 0.15


def _headers(contact_email):
    return {
        "User-Agent": f"Bossman Engine historical research contact: {contact_email}",
        "Accept-Encoding": "gzip, deflate",
    }


def _request_json(session, url, headers):
    for attempt in range(5):
        response = session.get(url, headers=headers, timeout=45)
        if response.status_code in {429, 503} and attempt < 4:
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2**attempt)
            continue
        response.raise_for_status()
        return response.json(), response.content
    raise RuntimeError(f"SEC request retry limit exceeded for {url}.")


def build_current_ticker_cik_map(sec_records):
    """Normalize the SEC's current symbol-to-CIK mapping."""
    ticker_map = {}
    for record in sec_records:
        ticker = str(record.get("ticker", "")).strip().upper().replace(".", "-")
        cik = record.get("cik_str")
        if ticker and cik is not None:
            ticker_map[ticker] = str(cik).zfill(10)
    return ticker_map


def _validate_company_facts(payload, expected_cik):
    if str(payload.get("cik", "")).zfill(10) != expected_cik:
        raise ValueError(f"SEC CompanyFacts response CIK mismatch for {expected_cik}.")
    if not isinstance(payload.get("facts"), dict):
        raise ValueError(f"SEC CompanyFacts response has no facts object for {expected_cik}.")


def _normalize_additional_ciks(ciks):
    normalized = set()
    for cik in ciks or ():
        value = str(cik).strip()
        if not re.fullmatch(r"\d{1,10}", value):
            raise ValueError(f"Invalid historical SEC CIK: {value!r}.")
        normalized.add(value.zfill(10))
    return normalized


def _summarize_facts(payload):
    filing_dates = []
    fact_count = 0
    namespaces = payload.get("facts", {})
    for namespace_facts in namespaces.values():
        for fact in namespace_facts.values():
            for observations in fact.get("units", {}).values():
                fact_count += len(observations)
                filing_dates.extend(
                    observation["filed"]
                    for observation in observations
                    if observation.get("filed")
                )
    return {
        "entity_name": payload.get("entityName"),
        "namespace_count": len(namespaces),
        "fact_observation_count": fact_count,
        "earliest_filing_date": min(filing_dates) if filing_dates else None,
        "latest_filing_date": max(filing_dates) if filing_dates else None,
    }


def acquire_company_facts(
    contact_email,
    constituent_tickers=None,
    cache_dir=SEC_CACHE_DIR,
    force_refresh=False,
    additional_ciks=None,
):
    if not contact_email or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", contact_email):
        raise ValueError("A valid SEC contact email is required.")

    constituent_tickers = constituent_tickers or load_historical_tickers()
    ticker_map_path = cache_dir / "company_tickers.json"
    facts_dir = cache_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    headers = _headers(contact_email)
    session = requests.Session()
    if ticker_map_path.exists() and not force_refresh:
        ticker_payload = json.loads(ticker_map_path.read_text())
    else:
        ticker_payload, _ = _request_json(session, SEC_TICKER_MAP_URL, headers)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ticker_map_path.write_text(json.dumps(ticker_payload))
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS)

    ticker_map = build_current_ticker_cik_map(ticker_payload.values())
    matched_tickers = {
        ticker: ticker_map[provider_symbol]
        for ticker in constituent_tickers
        if (provider_symbol := PROVIDER_SYMBOL_ALIASES.get(ticker, ticker)) in ticker_map
    }
    cik_tickers = {}
    for ticker, cik in matched_tickers.items():
        cik_tickers.setdefault(cik, []).append(ticker)
    current_ciks = set(cik_tickers)
    historical_ciks = _normalize_additional_ciks(additional_ciks)
    for cik in historical_ciks:
        cik_tickers.setdefault(cik, [])

    facts_summaries = {}
    failed_ciks = []
    for index, cik in enumerate(sorted(cik_tickers), start=1):
        path = facts_dir / f"CIK{cik}.json.gz"
        if path.exists() and not force_refresh:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            try:
                payload, response_content = _request_json(
                    session,
                    SEC_COMPANY_FACTS_URL.format(cik=cik),
                    headers,
                )
                _validate_company_facts(payload, cik)
                with gzip.open(path, "wb") as stream:
                    stream.write(response_content)
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 403:
                    raise RuntimeError(
                        "SEC returned HTTP 403. Stopping to avoid further requests; "
                        "confirm the SEC User-Agent contact or wait for rate limits."
                    ) from error
                failed_ciks.append({"cik": cik, "error": str(error)})
                print(f"SEC facts unavailable for CIK {cik}: {error}")
            except (requests.RequestException, ValueError) as error:
                failed_ciks.append({"cik": cik, "error": str(error)})
                print(f"SEC facts unavailable for CIK {cik}: {error}")
            finally:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS)
            if failed_ciks and failed_ciks[-1]["cik"] == cik:
                continue

        _validate_company_facts(payload, cik)
        facts_summaries[cik] = _summarize_facts(payload)
        if index % 50 == 0 or index == len(cik_tickers):
            print(
                f"Processed SEC CompanyFacts {index}/{len(cik_tickers)} "
                f"({len(facts_summaries)} available)"
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SEC EDGAR CompanyFacts API",
        "point_in_time_rule": (
            "Each observation includes its SEC filing date; consumers must "
            "exclude observations filed after the historical scoring date."
        ),
        "historical_constituent_symbols": len(constituent_tickers),
        "symbols_mapped_by_current_sec_ticker_file": len(matched_tickers),
        "historical_symbols_without_current_cik_mapping": sorted(
            set(constituent_tickers) - set(matched_tickers)
        ),
        "unique_current_ciks_requested": len(current_ciks),
        "historical_ciks_from_archived_revisions": len(historical_ciks),
        "historical_ciks_not_in_current_ticker_file": sorted(
            historical_ciks - current_ciks
        ),
        "unique_companyfacts_ciks_requested": len(cik_tickers),
        "companyfacts_available": len(facts_summaries),
        "companyfacts_failed": failed_ciks,
        "facts_cache_directory": str(facts_dir),
        "mapping_source": SEC_TICKER_MAP_URL,
        "companyfacts_source": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        "facts_summary": facts_summaries,
        "limitations": [
            "SEC's published ticker file maps current symbols and does not provide a complete historical ticker-to-CIK map.",
            "CompanyFacts observations must be filtered by filed date to avoid look-ahead; availability before mandatory XBRL filing coverage is limited.",
            "CompanyFacts does not itself supply historical GICS classifications or historical scoring-code versions.",
        ],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Mapped {len(matched_tickers)}/{len(constituent_tickers)} historical symbols; "
        f"SEC CompanyFacts available for {len(facts_summaries)}/{len(cik_tickers)} CIKs."
    )
    print(f"Manifest: {cache_dir / 'manifest.json'}")
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Redownload the SEC ticker map and all mapped CompanyFacts records.",
    )
    parser.add_argument(
        "--historical-metadata",
        help=(
            "Gzipped JSON output from acquire_historical_constituent_metadata.py; "
            "download archived CIKs in addition to current ticker-map matches."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    contact_email = os.environ.get("SEC_CONTACT_EMAIL", "")
    additional_ciks = ()
    if args.historical_metadata:
        with gzip.open(args.historical_metadata, "rt", encoding="utf-8") as stream:
            snapshots = json.load(stream)
        additional_ciks = {
            constituent["cik"]
            for snapshot in snapshots
            for constituent in snapshot["constituents"]
            if constituent.get("cik")
        }
    acquire_company_facts(
        contact_email,
        force_refresh=args.force_refresh,
        additional_ciks=additional_ciks,
    )


if __name__ == "__main__":
    main()
