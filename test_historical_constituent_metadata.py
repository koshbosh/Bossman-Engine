import unittest

from acquire_historical_constituent_metadata import (
    parse_historical_constituent_metadata,
)


class HistoricalConstituentMetadataTests(unittest.TestCase):
    def test_parses_legacy_industry_table_without_inventing_sector_or_cik(self):
        wikitext = """== S&P 500 component stocks ==
{| class="wikitable"
! Company !! [[Ticker symbol]] !! [[SEC filing|SEC filings]] !! Industry
|-
| [[3M Company]] || MMM || [http://www.sec.gov/cgi-bin/browse-edgar?CIK=MMM reports] || Industrial Conglomerates
|}
"""
        self.assertEqual(
            parse_historical_constituent_metadata(wikitext),
            [
                {
                    "ticker": "MMM",
                    "company": "3M Company",
                    "cik": None,
                    "gics_sector": None,
                }
            ],
        )

    def test_parses_sector_from_template_ticker_table(self):
        wikitext = """== S&P 500 component stocks ==
{| class="wikitable"
! [[Ticker symbol]] !! Company !! SEC filings !! GICS Sector
|-
| {{NyseSymbol|MMM}} || [[3M Company]] || reports || Industrials
|}
"""
        self.assertEqual(
            parse_historical_constituent_metadata(wikitext),
            [
                {
                    "ticker": "MMM",
                    "company": "3M Company",
                    "cik": None,
                    "gics_sector": "Industrials",
                }
            ],
        )

    def test_parses_point_in_time_cik_column(self):
        wikitext = """== S&P 500 component stocks ==
{| class="wikitable"
! Ticker symbol !! Security !! GICS Sector !! [[Central Index Key|CIK]]
|-
| {{NasdaqSymbol|ADBE}} || [[Adobe Inc.]] || Information Technology || 0000796343
|}
"""
        self.assertEqual(
            parse_historical_constituent_metadata(wikitext),
            [
                {
                    "ticker": "ADBE",
                    "company": "Adobe Inc.",
                    "cik": "0000796343",
                    "gics_sector": "Information Technology",
                }
            ],
        )

    def test_parses_one_cell_per_line_layout_and_normalizes_ticker(self):
        wikitext = """== S&P 500 component stocks ==
{| class="wikitable"
! Symbol
! Security !! GICS Sector !! [[Central Index Key|CIK]]
|-
|| {{NyseSymbol|BRK.B}}
|| [[Berkshire Hathaway]]
|| Financials
|| 0001067983
|}
"""
        self.assertEqual(
            parse_historical_constituent_metadata(wikitext),
            [
                {
                    "ticker": "BRK-B",
                    "company": "Berkshire Hathaway",
                    "cik": "0001067983",
                    "gics_sector": "Financials",
                }
            ],
        )

    def test_rejects_wikitext_without_the_component_section(self):
        with self.assertRaisesRegex(ValueError, "constituent table"):
            parse_historical_constituent_metadata("== History ==\nNo table here.")


if __name__ == "__main__":
    unittest.main()
