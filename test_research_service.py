import json
import os
import unittest
from unittest.mock import patch

import research_service
from research_service import (
    ResearchConfigurationError,
    _extract_filing_sections,
    _filing_url,
    _format_fact_evidence,
    normalize_ticker,
    answer_company_question,
    consume_research_rate_limit,
    get_research_status,
    is_valid_research_access_token,
)


class ResearchServiceTests(unittest.TestCase):
    def test_ticker_normalization_and_validation(self):
        self.assertEqual(normalize_ticker(" brk.b "), "BRK-B")
        with self.assertRaisesRegex(ValueError, "valid US stock ticker"):
            normalize_ticker("AAPL/../../")

    def test_status_reports_missing_secrets_without_exposing_them(self):
        with patch.dict(os.environ, {}, clear=True):
            status = get_research_status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["missing"], [
            "SEC_CONTACT_EMAIL (valid monitored email)",
            "OPENAI_API_KEY",
            "RESEARCH_ACCESS_TOKEN",
        ])
        self.assertNotIn("api_key", status)

    def test_configured_access_token_is_compared_without_exposure(self):
        with patch.dict(os.environ, {"RESEARCH_ACCESS_TOKEN": "private-token"}):
            self.assertTrue(is_valid_research_access_token("private-token"))
            self.assertFalse(is_valid_research_access_token("wrong-token"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_valid_research_access_token("private-token"))

    def test_filing_url_uses_only_sec_archive_paths(self):
        url = _filing_url("0000320193", "0000320193-24-000001", "aapl-20240928.htm")
        self.assertEqual(
            url,
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000001/aapl-20240928.htm",
        )
        self.assertIsNone(_filing_url("320193", "accession", "../outside.htm"))

    def test_sec_requests_use_uncompressed_data(self):
        with patch.dict(os.environ, {"SEC_CONTACT_EMAIL": "research@example.com"}):
            headers = research_service._sec_headers()
        self.assertEqual(headers["Accept-Encoding"], "identity")

    def test_extracts_standard_filing_sections_and_skips_script_text(self):
        markup = """
        <html><script>Item 1. Business fake injected text</script>
        <h2>Item 1. Business</h2><p>We sell software to businesses.</p>
        <h2>Item 1A. Risk Factors</h2><p>Competition may reduce growth.</p>
        <h2>Item 7. Management's Discussion and Analysis</h2>
        <p>Revenue increased during the period.</p><h2>Item 8. Financial Statements</h2>
        </html>
        """
        sections = _extract_filing_sections(markup)
        self.assertEqual([label for label, _ in sections], [
            "Business", "Risk factors", "Management discussion",
        ])
        self.assertIn("We sell software", sections[0][1])
        self.assertNotIn("fake injected text", " ".join(text for _, text in sections))
        self.assertNotIn("Financial Statements", sections[2][1])

    def test_formats_source_periods_and_filing_dates(self):
        lines = _format_fact_evidence({
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "val": 12500000000,
                                    "end": "2025-12-31",
                                    "filed": "2026-02-01",
                                    "form": "10-K",
                                    "fp": "FY",
                                },
                                {
                                    "val": 1000000000,
                                    "end": "2025-12-31",
                                    "filed": "2026-01-01",
                                    "form": "8-K",
                                    "fp": "FY",
                                },
                            ]
                        }
                    }
                }
            }
        })
        self.assertEqual(len(lines), 1)
        self.assertIn("$12.50B", lines[0])
        self.assertIn("filed 2026-02-01", lines[0])

    def test_answer_rejects_empty_or_oversized_question_before_fetching(self):
        with patch.object(research_service, "_get_company_evidence") as evidence:
            with self.assertRaisesRegex(ValueError, "Enter a research question"):
                answer_company_question("MSFT", "  ")
            with self.assertRaisesRegex(ValueError, "1,200 characters"):
                answer_company_question("MSFT", "x" * 1201)
        evidence.assert_not_called()

    def test_missing_model_key_is_an_explicit_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ResearchConfigurationError, "OPENAI_API_KEY"):
                research_service._ask_model("What are the risks?", {"name": "Example"}, "evidence")

    def test_model_request_keeps_provider_key_server_side_and_sends_evidence(self):
        response_body = json.dumps({
            "choices": [{"message": {"content": "Supported answer [S1]."}}]
        }).encode()
        with (
            patch.dict(os.environ, {
                "OPENAI_API_KEY": "test-secret-key",
                "OPENAI_MODEL": "test-model",
            }),
            patch("research_service.urlopen") as mock_urlopen,
        ):
            mock_urlopen.return_value.__enter__.return_value.read.return_value = response_body
            answer = research_service._ask_model(
                "Analyze revenue",
                {"name": "Example", "ticker": "EXM"},
                "[S1] Reported revenue was $1B.",
            )
        request = mock_urlopen.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertEqual(answer, "Supported answer [S1].")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret-key")
        self.assertIs(mock_urlopen.call_args.kwargs["context"], research_service.TLS_CONTEXT)
        self.assertIn("[S1] Reported revenue was $1B.", request_body["messages"][1]["content"])

    def test_answer_returns_model_text_with_primary_source_metadata(self):
        evidence = {
            "company": {"ticker": "MSFT", "name": "Microsoft Corporation"},
            "sources": [{"id": "S1", "title": "CompanyFacts", "url": "https://www.sec.gov/example"}],
            "evidence": "[S1] Revenue was reported.",
            "retrievedAt": "2026-11-07T10:00:00Z",
        }
        with (
            patch.object(research_service, "_get_company_evidence", return_value=evidence),
            patch.object(research_service, "_ask_model", return_value="Revenue grew [S1]."),
        ):
            result = answer_company_question("MSFT", "Summarize revenue.")
        self.assertEqual(result["answer"], "Revenue grew [S1].")
        self.assertEqual(result["sources"][0]["id"], "S1")
        self.assertEqual(result["ticker"], "MSFT")

    def test_entry_question_requests_market_analysis_and_cites_it(self):
        self.assertTrue(research_service.requires_market_analysis("Find possible entry zones"))
        self.assertFalse(research_service.requires_market_analysis("Summarize disclosed risks"))
        analysis = {
            "ticker": "MSFT",
            "price_as_of": "2026-11-06",
            "spot_price": 500.0,
            "annualized_volatility": 0.25,
            "horizon_days": 21,
            "one_month_sigma_pct": 7.1,
            "zones": [{
                "center_sigma": 0.5,
                "lower_price": 480.0,
                "upper_price": 490.0,
                "touch_probability": 0.6,
            }],
            "disclaimer": "Advisory only.",
        }
        with (
            patch.object(research_service, "_get_company_evidence", return_value={
                "company": {"ticker": "MSFT", "name": "Microsoft Corporation"},
                "sources": [{"id": "S1", "title": "CompanyFacts"}],
                "evidence": "[S1] SEC facts.",
                "retrievedAt": "2026-11-07T10:00:00Z",
            }),
            patch.object(research_service, "_ask_model", return_value="See [S2].") as ask_model,
        ):
            result = answer_company_question(
                "MSFT",
                "Find possible entry zones",
                market_analysis=analysis,
            )
        self.assertEqual(result["sources"][1]["id"], "S2")
        self.assertIn("Market data", result["sources"][1]["type"])
        self.assertIn("$500.00", ask_model.call_args.args[2])
        self.assertIn("modeled touch estimate 60.0%", ask_model.call_args.args[2])

    def test_research_rate_limit_caps_a_client(self):
        client_id = "test-client-rate-limit"
        self.assertTrue(consume_research_rate_limit(client_id))
        for _ in range(research_service.MAX_RESEARCH_REQUESTS_PER_CLIENT_PER_HOUR - 1):
            self.assertTrue(consume_research_rate_limit(client_id))
        self.assertFalse(consume_research_rate_limit(client_id))


if __name__ == "__main__":
    unittest.main()
