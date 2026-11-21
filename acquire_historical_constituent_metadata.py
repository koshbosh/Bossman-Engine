"""Recover point-in-time company, CIK, and GICS labels from archived Wikipedia revisions.

Usage:
    python acquire_historical_constituent_metadata.py

Revision text and extracted monthly metadata are cached under
~/.cache/bossman_engine_regime_backtest, outside the repository.
"""

import argparse
import gzip
import json
import re

import requests

from acquire_historical_ohlc import CONSTITUENT_CACHE_DIR
from regime_factor_backtest import (
    PRICE_CACHE_DIR,
    USER_AGENT,
    _request_wikipedia_json,
)


METADATA_CACHE_DIR = PRICE_CACHE_DIR / "wikipedia_historical_metadata"
REVISION_CACHE_DIR = METADATA_CACHE_DIR / "revisions"
METADATA_PATH = METADATA_CACHE_DIR / "monthly_constituent_metadata.json.gz"
MANIFEST_PATH = METADATA_CACHE_DIR / "manifest.json"
REVISION_BATCH_SIZE = 50


def _plain_cell_text(value):
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    value = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(
        r"\[\[([^|\]]+)\|([^\]]+)\]\]",
        lambda match: match.group(2),
        value,
    )
    value = re.sub(r"\[\[([^\]]+)\]\]", lambda match: match.group(1), value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _split_table_cells(line, marker):
    value = line.strip()[1:].strip()
    cells = [cell.strip().lstrip("|").strip() for cell in value.split(marker)]
    if cells and not cells[0]:
        cells.pop(0)
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _header_name(value):
    return _plain_cell_text(value).casefold()


def _ticker_from_cell(value):
    template = re.search(
        r"\{\{\s*[A-Za-z0-9 _-]*(?:symbol|nyse|nasdaq)\s*\|\s*([A-Z0-9.-]+)",
        value,
        flags=re.IGNORECASE,
    )
    candidate = template.group(1) if template else _plain_cell_text(value)
    candidate = candidate.strip().upper().replace(".", "-")
    return candidate if re.fullmatch(r"[A-Z0-9-]+", candidate) else ""


def _find_component_table(wikitext):
    section = re.search(
        r"^==+\s*S&P 500 component stocks\s*==+\s*$",
        wikitext,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if section:
        following_section = re.search(
            r"^==[^=].*?==\s*$",
            wikitext[section.end() :],
            re.MULTILINE,
        )
        end = (
            section.end() + following_section.start()
            if following_section
            else len(wikitext)
        )
        section_text = wikitext[section.end() : end]
    else:
        section_text = wikitext
    tables = re.findall(r"\{\|.*?\|\}", section_text, flags=re.DOTALL)
    candidates = []
    for table in tables:
        rows = _parse_component_table(table)
        if rows:
            candidates.append(rows)
    if not candidates:
        raise ValueError("No constituent table with ticker symbols was found.")
    return max(candidates, key=len)


def _parse_component_table(table):
    headers = []
    rows = []
    current_row = []
    reading_rows = False

    def save_row():
        if not headers or not current_row:
            return
        header_indexes = {
            "ticker": next(
                (i for i, name in enumerate(headers) if "ticker" in name or "symbol" in name),
                None,
            ),
            "company": next(
                (i for i, name in enumerate(headers) if "security" in name or "company" in name),
                None,
            ),
            "cik": next(
                (i for i, name in enumerate(headers) if re.search(r"\bcik\b", name)),
                None,
            ),
            "sector": next(
                (i for i, name in enumerate(headers) if "sector" in name),
                None,
            ),
        }
        ticker_index = header_indexes["ticker"]
        if ticker_index is None or ticker_index >= len(current_row):
            return

        ticker = _ticker_from_cell(current_row[ticker_index])
        if not ticker:
            return

        def cell_text(key):
            index = header_indexes[key]
            return (
                _plain_cell_text(current_row[index])
                if index is not None and index < len(current_row)
                else ""
            )

        cik_value = cell_text("cik")
        cik_match = re.search(r"(?<!\d)(\d{8,10})(?!\d)", cik_value)
        rows.append(
            {
                "ticker": ticker,
                "company": cell_text("company") or None,
                "cik": cik_match.group(1).zfill(10) if cik_match else None,
                "gics_sector": cell_text("sector") or None,
            }
        )

    for line in table.splitlines():
        stripped = line.strip()
        if stripped.startswith("|-"):
            save_row()
            current_row = []
            reading_rows = True
        elif stripped.startswith("|}"):
            save_row()
            current_row = []
            break
        elif stripped.startswith("!") and not rows:
            headers.extend(
                _header_name(header)
                for header in _split_table_cells(stripped, "!!")
            )
        elif reading_rows and stripped.startswith("|") and not stripped.startswith("|+"):
            current_row.extend(_split_table_cells(stripped, "||"))

    if not rows:
        return []

    # Prefer rows with distinct symbols and keep the first occurrence in the archived table.
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault(row["ticker"], row)
    return list(unique_rows.values())


def parse_historical_constituent_metadata(wikitext):
    """Parse company metadata from one archived S&P 500 constituent table."""
    return _find_component_table(wikitext)


def _load_cached_revision(revision_id, revision_cache_dir):
    path = revision_cache_dir / f"{revision_id}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("revision_id") != revision_id or not payload.get("wikitext"):
        raise ValueError(f"Invalid cached Wikipedia revision {revision_id}.")
    return payload["wikitext"]


def _fetch_revision_texts(revision_ids, session, revision_cache_dir):
    revision_cache_dir.mkdir(parents=True, exist_ok=True)
    content = {}
    uncached = []
    for revision_id in revision_ids:
        cached = _load_cached_revision(revision_id, revision_cache_dir)
        if cached is None:
            uncached.append(revision_id)
        else:
            content[revision_id] = cached

    for start in range(0, len(uncached), REVISION_BATCH_SIZE):
        batch = uncached[start : start + REVISION_BATCH_SIZE]
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
        found = {}
        for page in payload.get("query", {}).get("pages", {}).values():
            for revision in page.get("revisions", []):
                revision_id = revision["revid"]
                slots = revision.get("slots", {}).get("main", {})
                text = slots.get("*", slots.get("content", ""))
                if text:
                    found[revision_id] = text
        missing = set(batch) - set(found)
        if missing:
            raise RuntimeError(
                "Wikipedia did not return archived revision(s): "
                + ", ".join(map(str, sorted(missing)))
            )
        for revision_id, text in found.items():
            cache_path = revision_cache_dir / f"{revision_id}.json.gz"
            with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
                json.dump({"revision_id": revision_id, "wikitext": text}, stream)
            content[revision_id] = text
    return content


def acquire_historical_constituent_metadata(
    constituent_cache_dir=CONSTITUENT_CACHE_DIR,
    metadata_cache_dir=METADATA_CACHE_DIR,
):
    snapshot_paths = sorted(constituent_cache_dir.glob("constituents_*.json"))
    if not snapshot_paths:
        raise FileNotFoundError(
            f"No archived constituent snapshots found in {constituent_cache_dir}."
        )

    snapshots = [json.loads(path.read_text()) for path in snapshot_paths]
    revisions = {
        int(snapshot["revision_id"])
        for snapshot in snapshots
    }
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    metadata_cache_dir.mkdir(parents=True, exist_ok=True)
    revision_cache_dir = metadata_cache_dir / "revisions"
    metadata_path = metadata_cache_dir / "monthly_constituent_metadata.json.gz"
    manifest_path = metadata_cache_dir / "manifest.json"
    revision_texts = _fetch_revision_texts(
        sorted(revisions),
        session,
        revision_cache_dir,
    )

    monthly_records = []
    mismatches = []
    for snapshot in snapshots:
        month = snapshot["month"]
        rows = parse_historical_constituent_metadata(
            revision_texts[int(snapshot["revision_id"])]
        )
        expected = set(snapshot["tickers"])
        observed = {row["ticker"] for row in rows}
        if expected != observed:
            mismatches.append(
                {
                    "month": month,
                    "missing_from_table": sorted(expected - observed),
                    "not_in_cached_snapshot": sorted(observed - expected),
                }
            )
        monthly_records.append(
            {
                "month": month,
                "revision_id": snapshot["revision_id"],
                "revision_timestamp": snapshot["revision_timestamp"],
                "constituents": rows,
            }
        )

    total_symbols = sum(len(item["constituents"]) for item in monthly_records)
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        json.dump(monthly_records, stream)

    by_year = {}
    for year in sorted({item["month"][:4] for item in monthly_records}):
        year_records = [
            item for item in monthly_records if item["month"].startswith(year)
        ]
        symbol_rows = [
            row for item in year_records for row in item["constituents"]
        ]
        by_year[year] = {
            "months": len(year_records),
            "symbols": len(symbol_rows),
            "with_cik": sum(row["cik"] is not None for row in symbol_rows),
            "with_gics_sector": sum(row["gics_sector"] is not None for row in symbol_rows),
        }

    manifest = {
        "source": "Archived Wikipedia revisions of List of S&P 500 companies",
        "generated_metadata_path": str(metadata_path),
        "monthly_snapshots": len(monthly_records),
        "unique_wikipedia_revisions": len(revisions),
        "constituent_rows": total_symbols,
        "rows_with_cik": sum(
            row["cik"] is not None
            for item in monthly_records
            for row in item["constituents"]
        ),
        "rows_with_gics_sector": sum(
            row["gics_sector"] is not None
            for item in monthly_records
            for row in item["constituents"]
        ),
        "ticker_set_mismatches": mismatches,
        "coverage_by_year": by_year,
        "limitations": [
            "Wikipedia is a historical proxy, not an official S&P Dow Jones Indices constituent archive.",
            "Archived revisions before the CIK column was added do not provide direct point-in-time ticker-to-CIK mappings.",
            "Some early revisions have an Industry column rather than a GICS Sector column; no sector is inferred from it.",
            "GICS labels reproduce the text in each archived revision and have not been independently validated against licensed GICS history.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = acquire_historical_constituent_metadata()
    print(
        f"Processed {manifest['monthly_snapshots']} monthly snapshots "
        f"({manifest['unique_wikipedia_revisions']} unique revisions); "
        f"{manifest['rows_with_cik']}/{manifest['constituent_rows']} rows have CIK "
        f"and {manifest['rows_with_gics_sector']}/{manifest['constituent_rows']} "
        "have GICS sector data."
    )
    if manifest["ticker_set_mismatches"]:
        print(
            f"WARNING: {len(manifest['ticker_set_mismatches'])} ticker-set "
            "mismatches are recorded in the manifest."
        )
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
