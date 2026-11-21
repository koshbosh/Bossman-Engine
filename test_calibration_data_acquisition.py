import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import acquire_sec_company_facts
from acquire_sec_company_facts import (
    _normalize_additional_ciks,
    _summarize_facts,
    _validate_company_facts,
    build_current_ticker_cik_map,
    acquire_company_facts,
)
from acquire_historical_ohlc import _normalize_downloaded_batch
import pandas as pd


class CalibrationDataAcquisitionTests(unittest.TestCase):
    def test_archived_ciks_are_normalized_and_validated(self):
        self.assertEqual(
            _normalize_additional_ciks(["1067983", "0000320193", 1067983]),
            {"0001067983", "0000320193"},
        )
        with self.assertRaisesRegex(ValueError, "Invalid historical SEC CIK"):
            _normalize_additional_ciks(["not-a-cik"])

    def test_company_facts_acquisition_includes_archived_ciks(self):
        def fake_request(*request_args):
            url = request_args[1]
            if url == acquire_sec_company_facts.SEC_TICKER_MAP_URL:
                return (
                    {"0": {"ticker": "AAPL", "cik_str": 320193}},
                    b"{}",
                )
            cik = url.rsplit("CIK", 1)[1].removesuffix(".json")
            payload = {"cik": int(cik), "facts": {}}
            return payload, json.dumps(payload).encode("utf-8")

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                acquire_sec_company_facts,
                "_request_json",
                side_effect=fake_request,
            ),
            patch.object(acquire_sec_company_facts.time, "sleep"),
        ):
            manifest = acquire_company_facts(
                "research@example.com",
                constituent_tickers=["AAPL"],
                cache_dir=Path(temp_dir),
                additional_ciks=["9999999"],
            )

        self.assertEqual(manifest["unique_current_ciks_requested"], 1)
        self.assertEqual(manifest["historical_ciks_from_archived_revisions"], 1)
        self.assertEqual(manifest["unique_companyfacts_ciks_requested"], 2)
        self.assertEqual(manifest["companyfacts_available"], 2)

    def test_current_sec_symbol_map_normalizes_tickers_and_ciks(self):
        mapping = build_current_ticker_cik_map([
            {"ticker": "BRK.B", "cik_str": 1067983},
            {"ticker": "AAPL", "cik_str": 320193},
        ])

        self.assertEqual(mapping["BRK-B"], "0001067983")
        self.assertEqual(mapping["AAPL"], "0000320193")

    def test_company_facts_validation_requires_matching_cik(self):
        _validate_company_facts(
            {"cik": 320193, "facts": {"us-gaap": {}}},
            "0000320193",
        )
        with self.assertRaisesRegex(ValueError, "CIK mismatch"):
            _validate_company_facts(
                {"cik": 320193, "facts": {"us-gaap": {}}},
                "0000789019",
            )

    def test_fact_summary_reports_source_filing_dates(self):
        summary = _summarize_facts({
            "entityName": "Example Corp",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {
                            "USD": [
                                {"val": 1, "filed": "2010-02-01"},
                                {"val": 2, "filed": "2011-02-01"},
                            ]
                        }
                    }
                }
            },
        })

        self.assertEqual(summary["entity_name"], "Example Corp")
        self.assertEqual(summary["fact_observation_count"], 2)
        self.assertEqual(summary["earliest_filing_date"], "2010-02-01")
        self.assertEqual(summary["latest_filing_date"], "2011-02-01")

    def test_adjusted_ohlc_normalizer_handles_ticker_first_columns(self):
        columns = pd.MultiIndex.from_product(
            [["AAPL", "MMM"], ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Price"],
        )
        source = pd.DataFrame(
            [[1, 2, 0.5, 1.5, 10] * 2],
            index=pd.date_range("2020-01-02", periods=1),
            columns=columns,
        )

        panels = _normalize_downloaded_batch(source, ["AAPL", "MMM"])

        self.assertEqual(set(panels), {"AAPL", "MMM"})
        self.assertEqual(panels["AAPL"].columns.tolist(), ["Open", "High", "Low", "Close"])
        self.assertEqual(panels["MMM"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
