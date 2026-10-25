import unittest
from unittest.mock import patch

import requests

import universe_explorer
from universe_explorer import (
    RUSSELL_PROXY_NAME,
    _combine_sources,
    _parse_russell_page,
    build_gics_sectors,
    build_gics_subindustries,
    build_industry_groups,
    normalize_ticker,
)


class UniverseExplorerTests(unittest.TestCase):
    def test_normalizes_common_share_classes_and_rejects_invalid_symbols(self):
        self.assertEqual(normalize_ticker("brk.b"), "BRK-B")
        self.assertEqual(normalize_ticker(" AAPL "), "AAPL")
        self.assertIsNone(normalize_ticker("not a ticker"))
        self.assertIsNone(normalize_ticker(""))

    def test_russell_page_parser_extracts_company_ticker_and_source_industry(self):
        html = """
        <table>
          <tr><th>Company</th><th>Ticker</th><th>Last</th><th>Cur</th>
              <th>Mkt Cap</th><th>Industry</th></tr>
          <tr><td>Example Systems Inc</td><td>EXMP XNAS</td><td>12.34</td>
              <td>USD</td><td>120</td><td>Electronic Components</td></tr>
        </table>
        """
        records = _parse_russell_page(html)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ticker"], "EXMP")
        self.assertEqual(records[0]["name"], "Example Systems Inc")
        self.assertEqual(records[0]["industry"], "Electronic Components")
        self.assertEqual(records[0]["membership"], RUSSELL_PROXY_NAME)

    def test_combines_memberships_without_overwriting_preferred_classification(self):
        sp500 = [{
            "ticker": "EXMP",
            "name": "Example Inc",
            "sector": "Industrials",
            "industry": "Industrial Machinery",
            "classification_system": "GICS (source-provided)",
            "classification_source": "S&P source",
            "membership": "S&P 500",
        }]
        nasdaq = [{
            "ticker": "EXMP",
            "name": "Example Corporation",
            "sector": "Technology",
            "industry": "Software",
            "classification_system": "ICB (Wikipedia table)",
            "classification_source": "Nasdaq source",
            "membership": "Nasdaq-100",
        }]
        russell = [{
            "ticker": "EXMP",
            "name": "Example",
            "sector": "",
            "industry": "Applications Software",
            "classification_system": "Stoxray industry label",
            "classification_source": "Russell source",
            "membership": RUSSELL_PROXY_NAME,
        }]

        result = _combine_sources([russell, nasdaq, sp500])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sector"], "Industrials")
        self.assertEqual(result[0]["industry"], "Industrial Machinery")
        self.assertEqual(
            set(result[0]["memberships"]),
            {"S&P 500", "Nasdaq-100", RUSSELL_PROXY_NAME},
        )

    def test_missing_industry_is_explicit_and_group_keys_keep_taxonomies_separate(self):
        records = [
            {
                "ticker": "AAA", "name": "A", "sector": "", "industry": "",
                "classification_system": "", "classification_source": "",
                "membership": "S&P 500",
            },
            {
                "ticker": "BBB", "name": "B", "sector": "Not provided by source",
                "industry": "Software", "classification_system": "ICB",
                "classification_source": "Nasdaq list", "membership": "Nasdaq-100",
            },
        ]
        companies = _combine_sources([records])
        self.assertEqual(companies[0]["industry"], "Unclassified")
        self.assertNotEqual(companies[0]["group_key"], companies[1]["group_key"])
        groups = build_industry_groups(companies)
        self.assertEqual(sum(group["count"] for group in groups), 2)
        self.assertEqual(len(groups), 2)

    def test_collection_keeps_available_sources_and_marks_failed_sources(self):
        nasdaq_row = {
            "ticker": "AAA", "name": "A", "sector": "Technology",
            "industry": "Software", "classification_system": "ICB",
            "classification_source": "Nasdaq source", "membership": "Nasdaq-100",
        }
        russell_row = {
            "ticker": "BBB", "name": "B", "sector": "",
            "industry": "Software", "classification_system": "Stoxray",
            "classification_source": "Russell source",
            "membership": RUSSELL_PROXY_NAME,
        }
        with (
            patch.object(universe_explorer, "fetch_sp500", side_effect=requests.Timeout("offline")),
            patch.object(universe_explorer, "fetch_nasdaq100", return_value=[nasdaq_row]),
            patch.object(universe_explorer, "fetch_russell2000_proxy", return_value=[russell_row]),
        ):
            result = universe_explorer._collect_universe()

        self.assertEqual(len(result["companies"]), 2)
        self.assertEqual(result["sources"][0]["status"], "unavailable")
        self.assertEqual(result["sources"][1]["status"], "available")
        self.assertEqual(result["sources"][2]["status"], "partial")
        self.assertIn("do not verify a complete", result["sources"][2]["coverage_note"])

    def test_gics_hierarchy_groups_only_gics_companies_and_orders_sectors(self):
        companies = [
            {"classification_system": "GICS (source-provided)", "sector": "Industrials", "industry": "Machinery"},
            {"classification_system": "GICS (source-provided)", "sector": "Information Technology", "industry": "Semiconductors"},
            {"classification_system": "GICS (source-provided)", "sector": "Information Technology", "industry": "Software"},
            {"classification_system": "ICB (Wikipedia table)", "sector": "Technology", "industry": "Software"},
        ]
        sectors = build_gics_sectors(companies)
        self.assertEqual(
            [sector["name"] for sector in sectors],
            ["Information Technology", "Industrials"],
        )
        self.assertEqual(sectors[0]["company_count"], 2)
        subindustries = build_gics_subindustries(companies, "Information Technology")
        self.assertEqual(
            [subindustry["name"] for subindustry in subindustries],
            ["Semiconductors", "Software"],
        )
        self.assertEqual(sum(row["company_count"] for row in subindustries), 2)


if __name__ == "__main__":
    unittest.main()
