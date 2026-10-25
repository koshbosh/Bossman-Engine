import time
import unittest

import numpy as np
import pandas as pd

from market_scanner import (
    AutomatedMarketScanner,
    _extract_ticker_history,
    build_live_feature_row,
)
from recommender import calculate_ma200_bonus, normalize


class MarketScannerTests(unittest.TestCase):
    def test_extracts_ticker_first_and_field_first_batch_history(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        history = pd.DataFrame({"Close": [10, 11, 12]}, index=dates)
        ticker_first = pd.concat({"AAA": history}, axis=1)
        field_first = ticker_first.swaplevel(axis=1)

        pd.testing.assert_frame_equal(
            _extract_ticker_history(ticker_first, "AAA"),
            history,
        )
        pd.testing.assert_frame_equal(
            _extract_ticker_history(field_first, "AAA"),
            history,
        )
        self.assertTrue(_extract_ticker_history(ticker_first, "BBB").empty)

    def test_short_history_keeps_unavailable_factors_missing(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        history = pd.DataFrame(
            {
                "Close": np.linspace(20, 30, len(dates)),
                "Volume": np.full(len(dates), 1000),
            },
            index=dates,
        )

        features = build_live_feature_row("AAA", history, {"name": "Example"})

        self.assertTrue(np.isnan(features["momentum_3mo"]))
        self.assertTrue(np.isnan(features["momentum_1y"]))
        self.assertTrue(np.isnan(features["ma50_proximity"]))
        self.assertTrue(np.isnan(features["ma200_proximity"]))
        self.assertEqual(features["price_as_of"], dates[-1].isoformat())

    def test_normalization_assigns_neutral_values_to_missing_factors(self):
        normalized = normalize(pd.Series([1.0, np.nan, 3.0]))
        self.assertEqual(normalized.iloc[1], 0.5)
        self.assertTrue(np.isfinite(normalized).all())
        all_missing = normalize(pd.Series([np.nan, np.nan]))
        self.assertEqual(all_missing.tolist(), [0.5, 0.5])
        constant = normalize(pd.Series([2.0, 2.0, np.nan]))
        self.assertEqual(constant.tolist(), [0.5, 0.5, 0.5])
        self.assertEqual(calculate_ma200_bonus(70, 100, np.nan), 1.0)

    def test_scans_live_batch_data_tracks_failures_and_reranks_cached_features(self):
        dates = pd.date_range("2025-08-01", periods=300, freq="B")
        histories = {}
        for ticker, trend in (("AAA", 0.0007), ("BBB", -0.0002)):
            close = 50 * (1 + trend) ** np.arange(len(dates))
            histories[ticker] = pd.DataFrame(
                {"Close": close, "Volume": np.full(len(dates), 2000)},
                index=dates,
            )

        class FakeEngine:
            spy_history = pd.Series(
                np.linspace(100, 120, len(dates)),
                index=dates,
            )

            def predict_return_ols(self, features):
                return {"expected_return": 8.0, "prob_positive": 60.0}

        downloads = []

        def download(tickers):
            downloads.append(list(tickers))
            return pd.concat(
                {ticker: histories[ticker] for ticker in tickers if ticker in histories},
                axis=1,
            )

        scanner = AutomatedMarketScanner(
            FakeEngine(),
            universe_loader=lambda: {
                "companies": [
                    {"ticker": "AAA", "name": "A", "sector": "Industrials"},
                    {"ticker": "BBB", "name": "B", "sector": "Health Care"},
                    {"ticker": "CCC", "name": "C", "sector": "Industrials"},
                    {"ticker": "aaa", "name": "Duplicate", "sector": "Industrials"},
                ],
                "sources": [{
                    "name": "Russell proxy",
                    "count": 3,
                    "status": "partial",
                    "coverage_note": "Incomplete proxy roster",
                }],
                "fetched_at": "2026-10-26T12:00:00+00:00",
            },
            history_downloader=download,
        )

        initial = scanner.get_scan("Dip Hunter", force_refresh=True)
        self.assertEqual(initial["status"], "running")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = scanner.get_scan("Dip Hunter")
            if result["status"] != "running":
                break
            time.sleep(0.01)

        self.assertEqual(result["status"], "complete", result["message"])
        self.assertEqual(result["total_in_universe"], 3)
        self.assertEqual(result["processed_count"], 3)
        self.assertEqual(result["scoreable_count"], 2)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(len(result["leaderboard"]), 2)
        self.assertTrue(all(np.isfinite(row["score"]) for row in result["leaderboard"]))
        self.assertTrue(result["price_data_as_of"])
        self.assertIn("incomplete", result["source_counts"][0]["coverage_note"].lower())

        reranked = scanner.get_scan("Alpha Bull")
        self.assertEqual(reranked["profile"], "Alpha Bull")
        self.assertEqual(len(downloads), 1)

        scanner.get_scan("Dip Hunter", force_refresh=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            refreshed = scanner.get_scan("Dip Hunter")
            if refreshed["status"] != "running":
                break
            time.sleep(0.01)
        self.assertEqual(refreshed["status"], "complete", refreshed["message"])
        self.assertEqual(len(downloads), 2)

    def test_rejects_unknown_profile_before_fetching_universe(self):
        scanner = AutomatedMarketScanner(
            object(),
            universe_loader=lambda: self.fail("Unknown profile should fail before loading."),
        )
        with self.assertRaisesRegex(ValueError, "Unknown scoring profile"):
            scanner.get_scan("Unknown")


if __name__ == "__main__":
    unittest.main()
