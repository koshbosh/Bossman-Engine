from datetime import date, timedelta
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_scanner import (
    AutomatedMarketScanner,
    _extract_ticker_history,
    build_live_feature_row,
)
from engine_service import UniversalScoringEngine, apply_gics_classification
from market_data import prepare_daily_price_history
from recommender import (
    calculate_ma200_bonus,
    calculate_ma_proximity,
    calculate_overextended_momentum_penalty,
    generate_recommendations,
    normalize,
)


class MarketScannerTests(unittest.TestCase):
    def test_ma_proximity_only_penalizes_material_upside_extension(self):
        self.assertEqual(calculate_ma_proximity(79.24, 168.19), 0.0)
        self.assertEqual(calculate_ma_proximity(100, 100), 0.0)
        self.assertEqual(calculate_ma_proximity(110, 100), 0.0)
        self.assertAlmostEqual(calculate_ma_proximity(120, 100), -0.05)

    def test_weekly_overextension_is_checked_when_monthly_history_is_missing(self):
        self.assertEqual(
            calculate_overextended_momentum_penalty(None, 0.11),
            0.5,
        )

    def test_daily_history_is_sorted_deduplicated_and_keeps_only_valid_closes(self):
        now = pd.Timestamp("2026-03-16")
        dates = pd.date_range(end=now, periods=22, freq="B")
        history = pd.DataFrame(
            {"Close": np.arange(22, dtype=float) + 10},
            index=dates,
        )
        history.iloc[2, history.columns.get_loc("Close")] = np.nan
        history = pd.concat([history, history.iloc[[-1]]]).iloc[::-1]

        cleaned = prepare_daily_price_history(
            history,
            "AAA",
            now=now,
        )

        self.assertEqual(len(cleaned), 21)
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        self.assertTrue(cleaned.index.is_unique)
        self.assertTrue((cleaned["Close"] > 0).all())

    def test_daily_history_rejects_stale_and_future_closes(self):
        now = pd.Timestamp("2026-03-16")
        stale = pd.DataFrame(
            {"Close": np.arange(20, dtype=float) + 10},
            index=pd.date_range(end=now - timedelta(days=8), periods=20, freq="B"),
        )
        future = pd.DataFrame(
            {"Close": np.arange(20, dtype=float) + 10},
            index=pd.date_range(end=now + timedelta(days=3), periods=20, freq="B"),
        )

        with self.assertRaisesRegex(ValueError, "stale"):
            prepare_daily_price_history(stale, "AAA", now=now)
        with self.assertRaisesRegex(ValueError, "future"):
            prepare_daily_price_history(future, "AAA", now=now)

    def test_baseline_ma_proximity_is_rebuilt_from_price_and_averages(self):
        baseline = pd.DataFrame([{
            "ticker": "SNDK",
            "sector": "Information Technology",
            "price": 1728.43,
            "ma_50": 1536.29,
            "ma_200": 1130.35,
            "ma50_proximity": 0.0,
            "ma200_proximity": 0.0,
            "momentum_3mo": -0.57,
            "momentum_6mo": -0.60,
        }])

        with tempfile.NamedTemporaryFile(suffix=".csv") as file:
            baseline.to_csv(file.name, index=False)
            engine = UniversalScoringEngine.__new__(UniversalScoringEngine)
            engine.baseline_csv_path = file.name
            engine.features_list = []
            with patch.object(engine, "refresh_macro_and_sector_regimes"):
                engine.load_baseline()

        row = engine.baseline_df.iloc[0]
        self.assertEqual(row["ma50_proximity"], 0.0)
        self.assertAlmostEqual(
            row["ma200_proximity"],
            -((1728.43 - 1130.35) / 1130.35 - 0.15),
        )

    def test_gics_classification_supplies_sector_etf_context_for_deep_dive(self):
        ticker_data = {"ticker": "APH", "sector": "N/A", "industry": "N/A"}

        classified = apply_gics_classification(
            ticker_data,
            "Information Technology",
            "Electronic Equipment & Instruments",
        )

        self.assertEqual(classified["sector"], "Information Technology")
        self.assertEqual(classified["industry"], "Electronic Equipment & Instruments")
        self.assertEqual(ticker_data["sector"], "N/A")
        with self.assertRaisesRegex(ValueError, "Unknown GICS sector"):
            apply_gics_classification(ticker_data, "Unknown sector", "Unknown industry")

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
        dates = pd.bdate_range(
            end=date.today() - timedelta(days=1),
            periods=30,
        )
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
        self.assertAlmostEqual(
            features["momentum_1mo"],
            features["price"] / history["Close"].iloc[-22] - 1,
        )
        self.assertAlmostEqual(
            features["perf_1w"],
            features["price"] / history["Close"].iloc[-6] - 1,
        )
        self.assertEqual(features["price_as_of"], dates[-1].isoformat())

    def test_volume_surge_baseline_excludes_the_current_session(self):
        dates = pd.bdate_range(
            end=date.today() - timedelta(days=1),
            periods=30,
        )
        volume = np.full(len(dates), 1000.0)
        volume[-1] = 4000.0
        history = pd.DataFrame(
            {
                "Close": np.linspace(20, 30, len(dates)),
                "Volume": volume,
            },
            index=dates,
        )

        features = build_live_feature_row("AAA", history, {"name": "Example"})

        self.assertEqual(features["current_volume"], 4000.0)
        self.assertEqual(features["avg_volume"], 1000.0)

    def test_normalization_preserves_missing_factors(self):
        normalized = normalize(pd.Series([1.0, np.nan, 3.0]))
        self.assertTrue(np.isnan(normalized.iloc[1]))
        self.assertTrue(np.isfinite(normalized.dropna()).all())
        all_missing = normalize(pd.Series([np.nan, np.nan]))
        self.assertTrue(all_missing.isna().all())
        constant = normalize(pd.Series([2.0, 2.0, np.nan]))
        self.assertEqual(constant.iloc[:2].tolist(), [0.5, 0.5])
        self.assertTrue(np.isnan(constant.iloc[2]))
        self.assertEqual(calculate_ma200_bonus(70, 100, np.nan), 1.0)

    def test_scans_live_batch_data_tracks_failures_and_reranks_cached_features(self):
        dates = pd.bdate_range(
            end=date.today() - timedelta(days=1),
            periods=300,
        )
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

            def refresh_macro_data(self):
                return True

            def update_live_market_context(self, feature_frame):
                self.live_market_context = feature_frame.copy()

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
        self.assertEqual(len(scanner.engine.live_market_context), 2)

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

    def test_scanner_recommendations_do_not_assign_neutral_scores_to_missing_rows(self):
        features = pd.DataFrame([
            {
                "ticker": "AAA",
                "sector": "Industrials",
                "price": 100.0,
                "momentum_3mo": 0.2,
                "momentum_1mo": 0.1,
                "momentum_6mo": np.nan,
                "momentum_1y": np.nan,
                "ma50_proximity": np.nan,
                "ma200_proximity": np.nan,
                "annualized_vol": np.nan,
                "rsi": np.nan,
                "beta": np.nan,
                "ma_50": np.nan,
                "ma_200": np.nan,
                "perf_1w": np.nan,
                "current_volume": np.nan,
                "avg_volume": np.nan,
            },
            {
                "ticker": "BBB",
                "sector": "Industrials",
                "price": 100.0,
                "momentum_3mo": np.nan,
                "momentum_1mo": np.nan,
                "momentum_6mo": np.nan,
                "momentum_1y": np.nan,
                "ma50_proximity": np.nan,
                "ma200_proximity": np.nan,
                "annualized_vol": np.nan,
                "rsi": np.nan,
                "beta": np.nan,
                "ma_50": np.nan,
                "ma_200": np.nan,
                "perf_1w": np.nan,
                "current_volume": np.nan,
                "avg_volume": np.nan,
            },
        ])

        ranked = generate_recommendations(
            features,
            user_profile="Dip Hunter",
            top_n=None,
        )

        scores = ranked.set_index("ticker")["recommendation_score"]
        self.assertTrue(np.isfinite(scores["AAA"]))
        self.assertTrue(np.isnan(scores["BBB"]))

    def test_macro_regime_is_unavailable_without_complete_fresh_inputs(self):
        engine = UniversalScoringEngine.__new__(UniversalScoringEngine)
        engine.baseline_df = pd.DataFrame({
            "price": [120.0],
            "ma_50": [100.0],
            "ma_200": [90.0],
        })
        engine._macro_close_data = pd.DataFrame()
        engine.macro_data_as_of = None
        engine.macro_data_error = "macro feed unavailable"
        engine.spy_history = None
        engine.sector_etf_data = {}

        engine.refresh_macro_and_sector_regimes(refresh_macro=False)

        self.assertEqual(engine.market_regime["regime"], "Market data unavailable")
        self.assertEqual(engine.market_regime["recommended_profiles"], [])
        self.assertIsNone(engine.market_regime["price"])
        self.assertIsNone(engine.market_regime["breadth_50"])
        self.assertIsNone(engine.market_regime["vix"])

    def test_engine_startup_does_not_fetch_macro_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = f"{temp_dir}/missing-baseline.csv"
            with patch("engine_service.yf.download") as download:
                engine = UniversalScoringEngine(baseline_csv_path=baseline_path)

            download.assert_not_called()
            self.assertEqual(engine.market_regime["regime"], "Market data unavailable")

    def test_ols_proxy_is_unavailable_instead_of_imputing_missing_inputs(self):
        class NoCallModel:
            def predict(self, _features):
                raise AssertionError("The model must not receive imputed values.")

        engine = UniversalScoringEngine.__new__(UniversalScoringEngine)
        engine.features_list = [
            "pe_ratio",
            "debt_to_equity",
            "ma50_proximity",
            "ma200_proximity",
            "momentum_1y",
            "rsi",
            "annualized_vol",
            "beta",
        ]
        engine.ols_model = NoCallModel()

        forecast = engine.predict_return_ols({"pe_ratio": 12.0})

        self.assertIsNone(forecast["expected_return"])
        self.assertEqual(forecast["validation_status"], "unavailable")
        self.assertIn("debt_to_equity", forecast["reason"])

    def test_completed_scan_updates_live_breadth_and_scoring_reference(self):
        engine = UniversalScoringEngine.__new__(UniversalScoringEngine)
        engine.baseline_df = pd.DataFrame()
        engine._macro_close_data = pd.DataFrame()
        engine.macro_data_as_of = None
        engine.macro_data_error = None
        engine.spy_history = None
        engine.sector_etf_data = {}

        features = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Tech", "Tech", "Health", "Health"],
            "price": [11.0, 9.0, 12.0, 8.0],
            "ma_50": [10.0, 10.0, 10.0, 10.0],
            "ma_200": [10.0, 10.0, 10.0, 10.0],
            "momentum_3mo": [0.2, 0.0, 0.1, -0.1],
            "momentum_6mo": [0.3, 0.1, 0.2, 0.0],
            "price_as_of": ["2026-11-12"] * 4,
        })

        engine.update_live_market_context(features)

        self.assertEqual(engine.market_regime["breadth_50"], 50.0)
        self.assertEqual(engine.market_regime["breadth_200"], 50.0)
        self.assertEqual(engine.market_regime["breadth_source"], "Live scanner universe")
        self.assertEqual(engine.market_regime["breadth_universe_count"], 4)
        self.assertTrue(engine.market_regime["breadth_data_as_of"].startswith("2026-11-12"))
        self.assertEqual(engine.scoring_reference_source, "Live scanner universe")
        self.assertEqual(engine.sector_3mo_medians["Tech"], 0.1)

    def test_rejects_unknown_profile_before_fetching_universe(self):
        scanner = AutomatedMarketScanner(
            object(),
            universe_loader=lambda: self.fail("Unknown profile should fail before loading."),
        )
        with self.assertRaisesRegex(ValueError, "Unknown scoring profile"):
            scanner.get_scan("Unknown")


if __name__ == "__main__":
    unittest.main()
