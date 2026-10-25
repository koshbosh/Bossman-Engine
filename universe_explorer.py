"""Free public-source company universe collection for the industry explorer."""

import re
import threading
import time
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
RUSSELL2000_URL = "https://stoxray.com/markets/russell-2000"
RUSSELL_PROXY_NAME = "Russell 2000 (Stoxray proxy)"
GICS_CLASSIFICATION = "GICS (source-provided)"
GICS_SECTOR_ORDER = (
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
)
CACHE_TTL_SECONDS = 6 * 60 * 60
REQUEST_TIMEOUT = (5, 30)
USER_AGENT = "BossmanEngineIndustryExplorer/1.0 (public-data research)"


def _clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()

_cache_lock = threading.Lock()
_cached_universe = None
_cached_at = 0.0
_cached_gics_universe = None
_cached_gics_at = 0.0


def normalize_ticker(value):
    """Normalize common share-class punctuation without guessing symbols."""
    if value is None or pd.isna(value):
        return None
    ticker = str(value).strip().upper().replace(".", "-")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,9}", ticker):
        return None
    return ticker


def _get(session, url):
    response = session.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def fetch_sp500(session):
    response = _get(session, SP500_URL)
    frame = pd.read_csv(StringIO(response.text))
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    if not required.issubset(frame.columns):
        raise ValueError("S&P 500 CSV is missing expected constituent/classification columns")

    records = []
    for row in frame.to_dict("records"):
        ticker = normalize_ticker(row.get("Symbol"))
        if not ticker:
            continue
        records.append({
            "ticker": ticker,
            "name": _clean_text(row.get("Security")) or ticker,
            "sector": _clean_text(row.get("GICS Sector")),
            "industry": _clean_text(row.get("GICS Sub-Industry")),
            "classification_system": "GICS (source-provided)",
            "classification_source": "DataHub S&P 500 CSV (Wikipedia-derived)",
            "classification_url": SP500_URL,
            "membership": "S&P 500",
        })
    if len(records) < 450:
        raise ValueError(f"Unexpectedly low S&P 500 constituent count: {len(records)}")
    return records


def fetch_nasdaq100(session):
    response = _get(session, NASDAQ100_URL)
    tables = pd.read_html(StringIO(response.text))
    frame = next(
        (
            table for table in tables
            if "Ticker" in table.columns
            and any("ICB Industry" in str(column) for column in table.columns)
            and any("ICB Subsector" in str(column) for column in table.columns)
        ),
        None,
    )
    if frame is None:
        raise ValueError("Nasdaq-100 company table with ICB classifications was not found")

    industry_column = next(column for column in frame.columns if "ICB Industry" in str(column))
    subsector_column = next(column for column in frame.columns if "ICB Subsector" in str(column))
    company_column = next(
        (column for column in frame.columns if str(column).strip().lower() in {"company", "security"}),
        None,
    )
    records = []
    for row in frame.to_dict("records"):
        ticker = normalize_ticker(row.get("Ticker"))
        if not ticker:
            continue
        company = _clean_text(row.get(company_column)) if company_column else ticker
        records.append({
            "ticker": ticker,
            "name": company or ticker,
            "sector": _clean_text(row.get(industry_column)),
            "industry": _clean_text(row.get(subsector_column)),
            "classification_system": "ICB (Wikipedia table)",
            "classification_source": "Nasdaq-100 Wikipedia company list",
            "classification_url": NASDAQ100_URL,
            "membership": "Nasdaq-100",
        })
    if len(records) < 80:
        raise ValueError(f"Unexpectedly low Nasdaq-100 constituent count: {len(records)}")
    return records


def _parse_russell_page(html):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 6:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        ticker_text = values[1].split()[0] if values[1].split() else ""
        ticker = normalize_ticker(ticker_text)
        if not ticker or ticker.lower() in {"ticker", "symbol"}:
            continue
        industry = values[-1].strip()
        records.append({
            "ticker": ticker,
            "name": values[0].strip() or ticker,
            "sector": "",
            "industry": industry,
            "classification_system": "Stoxray industry label",
            "classification_source": "Stoxray Russell 2000 constituent page",
            "classification_url": RUSSELL2000_URL,
            "membership": RUSSELL_PROXY_NAME,
        })
    return records


def fetch_russell2000_proxy(session):
    response = _get(session, RUSSELL2000_URL)
    first_html = response.text
    pages = [first_html]
    soup = BeautifulSoup(first_html, "html.parser")
    last_page = 1
    for link in soup.find_all("a", href=True):
        if link.get_text(" ", strip=True).lower() == "last":
            match = re.search(r"[?&]page=(\d+)", link["href"])
            if match:
                last_page = min(int(match.group(1)), 45)
                break

    for page_number in range(2, last_page + 1):
        page_url = f"{RUSSELL2000_URL}?page={page_number}"
        pages.append(_get(session, page_url).text)
        time.sleep(0.12)

    records = []
    for html in pages:
        records.extend(_parse_russell_page(html))
    unique = {record["ticker"]: record for record in records}
    if len(unique) < 1000:
        raise ValueError(
            f"Stoxray returned only {len(unique)} unique symbols; "
            "the Russell 2000 proxy is too incomplete to include"
        )
    return list(unique.values())


def _combine_sources(source_records):
    priority = {
        "S&P 500": 3,
        "Nasdaq-100": 2,
        RUSSELL_PROXY_NAME: 1,
    }
    companies = {}
    for records in source_records:
        for record in records:
            ticker = record["ticker"]
            current = companies.get(ticker)
            if current is None:
                companies[ticker] = {
                    **record,
                    "memberships": [record["membership"]],
                    "_priority": priority[record["membership"]],
                }
                continue
            if record["membership"] not in current["memberships"]:
                current["memberships"].append(record["membership"])
            if priority[record["membership"]] > current["_priority"]:
                current.update({
                    key: record.get(key, "")
                    for key in (
                        "name", "sector", "industry",
                        "classification_system", "classification_source",
                        "classification_url",
                    )
                })
                current["_priority"] = priority[record["membership"]]

    result = []
    for company in companies.values():
        company.pop("membership", None)
        company.pop("_priority", None)
        company["sector"] = company["sector"] or "Not provided by source"
        company["industry"] = company["industry"] or "Unclassified"
        company["classification_system"] = (
            company["classification_system"] or "Unclassified"
        )
        company["classification_source"] = (
            company["classification_source"] or "No classification source"
        )
        company["classification_url"] = company.get("classification_url", "")
        company["group_key"] = "|".join((
            company["classification_system"],
            company["sector"],
            company["industry"],
        ))
        result.append(company)
    return sorted(result, key=lambda company: company["ticker"])


def _collect_universe():
    session = requests.Session()
    source_definitions = [
        ("S&P 500 · DataHub / Wikipedia-derived", fetch_sp500, SP500_URL, False),
        ("Nasdaq-100 · Wikipedia / ICB", fetch_nasdaq100, NASDAQ100_URL, False),
        (RUSSELL_PROXY_NAME, fetch_russell2000_proxy, RUSSELL2000_URL, True),
    ]
    source_records = []
    source_status = []
    for name, fetcher, url, is_proxy in source_definitions:
        try:
            records = fetcher(session)
            source_records.append(records)
            source_status.append({
                "name": name,
                "url": url,
                "status": "partial" if is_proxy else "available",
                "count": len(records),
                "coverage_note": (
                    "Unofficial proxy; paginated results overlap and do not verify a complete Russell 2000 roster."
                    if is_proxy else None
                ),
            })
        except (requests.RequestException, ValueError, KeyError, StopIteration) as error:
            source_status.append({
                "name": name,
                "url": url,
                "status": "unavailable",
                "count": 0,
                "message": str(error),
            })
    companies = _combine_sources(source_records)
    if not companies:
        details = "; ".join(
            f"{source['name']}: {source.get('message', 'no data')}"
            for source in source_status
        )
        raise RuntimeError(f"No public constituent source was available. {details}")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "companies": companies,
        "sources": source_status,
        "fetched_at": fetched_at,
    }


def get_market_universe(force_refresh=False):
    """Return a process-cached union; refreshes public sources every six hours."""
    global _cached_universe, _cached_at
    with _cache_lock:
        is_fresh = _cached_universe is not None and time.time() - _cached_at < CACHE_TTL_SECONDS
        if is_fresh and not force_refresh:
            return _cached_universe
        try:
            collected = _collect_universe()
        except RuntimeError:
            if _cached_universe is None:
                raise
            return {
                **_cached_universe,
                "refresh_error": "Refresh failed; showing the last successfully collected universe.",
            }
        _cached_universe = collected
        _cached_at = time.time()
        return collected


def build_industry_groups(companies):
    groups = {}
    for company in companies:
        key = company["group_key"]
        if key not in groups:
            groups[key] = {
                "key": key,
                "sector": company["sector"],
                "industry": company["industry"],
                "classification_system": company["classification_system"],
                "classification_source": company["classification_source"],
                "classification_url": company["classification_url"],
                "count": 0,
            }
        groups[key]["count"] += 1
    return sorted(
        groups.values(),
        key=lambda group: (
            group["sector"].casefold(),
            group["industry"].casefold(),
            group["classification_system"].casefold(),
        ),
    )


def build_gics_sectors(companies):
    counts = {}
    for company in companies:
        if company["classification_system"] != GICS_CLASSIFICATION:
            continue
        sector = company["sector"]
        if sector and sector != "Not provided by source":
            counts[sector] = counts.get(sector, 0) + 1
    return [
        {"name": name, "company_count": count}
        for name, count in sorted(
            counts.items(),
            key=lambda item: (
                GICS_SECTOR_ORDER.index(item[0])
                if item[0] in GICS_SECTOR_ORDER
                else len(GICS_SECTOR_ORDER),
                item[0].casefold(),
            ),
        )
    ]


def build_gics_subindustries(companies, sector):
    counts = {}
    for company in companies:
        if (
            company["classification_system"] != GICS_CLASSIFICATION
            or company["sector"] != sector
            or company["industry"] == "Unclassified"
        ):
            continue
        counts[company["industry"]] = counts.get(company["industry"], 0) + 1
    return [
        {"name": name, "company_count": count}
        for name, count in sorted(counts.items(), key=lambda item: item[0].casefold())
    ]


def get_gics_universe(force_refresh=False):
    """Return just the GICS-classified S&P list used by the sector explorer."""
    global _cached_gics_universe, _cached_gics_at
    with _cache_lock:
        is_fresh = (
            _cached_gics_universe is not None
            and time.time() - _cached_gics_at < CACHE_TTL_SECONDS
        )
        if is_fresh and not force_refresh:
            return _cached_gics_universe

        try:
            with requests.Session() as session:
                companies = fetch_sp500(session)
        except (requests.RequestException, ValueError) as error:
            if _cached_gics_universe is None:
                raise RuntimeError(f"GICS constituent source unavailable: {error}") from error
            return {
                **_cached_gics_universe,
                "refresh_error": "GICS source refresh failed; showing the last successful list.",
            }

        result = {
            "companies": companies,
            "sources": [{
                "name": "S&P 500 · DataHub / Wikipedia-derived GICS",
                "url": SP500_URL,
                "status": "available",
                "count": len(companies),
            }],
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _cached_gics_universe = result
        _cached_gics_at = time.time()
        return result
