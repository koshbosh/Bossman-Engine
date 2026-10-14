import unittest

import numpy as np

from entry_zones import (
    _simulate_zone_touch_probabilities,
    build_entry_zone_analysis,
    hypothetical_ticker_data,
)


class StubEngine:
    def score_ticker(self, ticker_data, profile_name):
        return {"score": round(ticker_data["price"], 2)}


class EntryZoneTests(unittest.TestCase):
    def setUp(self):
        self.closes = np.linspace(80, 100, 260).tolist()
        self.highs = (np.asarray(self.closes) * 1.01).tolist()
        self.lows = (np.asarray(self.closes) * 0.99).tolist()
        self.ticker_data = {
            "ticker": "TEST",
            "price": 100.0,
            "annualized_vol": 0.25,
            "n_days": len(self.closes),
            "momentum_1mo": 0.01,
            "momentum_3mo": 0.02,
            "momentum_6mo": 0.03,
            "momentum_1y": 0.04,
            "rsi": 50.0,
            "ma_50": 98.0,
            "ma_200": 92.0,
            "ma50_proximity": 0.02,
            "ma200_proximity": 0.08,
            "fifty_two_high": 100.0,
            "fifty_two_low": 80.0,
        }

    def test_hypothetical_price_recalculates_price_features_without_mutating_input(self):
        original_momentum = self.ticker_data["momentum_1mo"]
        adjusted = hypothetical_ticker_data(self.ticker_data, self.closes, 90.0)

        self.assertEqual(adjusted["price"], 90.0)
        self.assertNotEqual(adjusted["momentum_1mo"], original_momentum)
        self.assertNotEqual(adjusted["ma_50"], self.ticker_data["ma_50"])
        self.assertEqual(self.ticker_data["price"], 100.0)
        self.assertEqual(self.ticker_data["momentum_1mo"], original_momentum)

    def test_entry_zone_output_keeps_uncalibrated_expectancy_unavailable(self):
        data = dict(
            self.ticker_data,
            _close_history=self.closes,
            _high_history=self.highs,
            _low_history=self.lows,
        )
        result = build_entry_zone_analysis(StubEngine(), data, "Dip Hunter", simulations=1000)

        self.assertEqual(len(result["zones"]), 4)
        self.assertEqual(result["calibration"]["status"], "unavailable")
        self.assertTrue(all(zone["expected_return"] is None for zone in result["zones"]))
        self.assertTrue(all(zone["combined_rank_value"] is None for zone in result["zones"]))
        self.assertEqual(sorted(zone["score_rank"] for zone in result["zones"]), [1, 2, 3, 4])
        self.assertEqual(
            sorted(zone["touch_probability_rank"] for zone in result["zones"]),
            [1, 2, 3, 4],
        )
        self.assertTrue(all(zone["lower_price"] < zone["upper_price"] < 100 for zone in result["zones"]))

    def test_touch_probability_simulation_is_reproducible_and_bounded(self):
        returns = np.random.default_rng(3).normal(0, 0.01, size=252)
        high_ratios = np.full(252, 1.01)
        low_ratios = np.full(252, 0.99)
        first, note = _simulate_zone_touch_probabilities(
            100.0, 0.25, returns, high_ratios, low_ratios, simulations=2000, seed=17
        )
        second, _ = _simulate_zone_touch_probabilities(
            100.0, 0.25, returns, high_ratios, low_ratios, simulations=2000, seed=17
        )

        self.assertIsNone(note)
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= probability <= 1 for probability in first.values()))

    def test_touch_probability_requires_sufficient_price_history(self):
        probabilities, note = _simulate_zone_touch_probabilities(
            100.0,
            0.25,
            np.zeros(20),
            np.ones(20),
            np.ones(20),
            simulations=100,
            seed=1,
        )
        self.assertIsNone(probabilities)
        self.assertIn("60 daily returns", note)

    def test_touch_probability_counts_intraday_band_intersection(self):
        returns = np.zeros(252)
        high_ratios = np.full(252, 1.01)
        low_ratios = np.full(252, 0.75)
        probabilities, note = _simulate_zone_touch_probabilities(
            100.0,
            0.25,
            returns,
            high_ratios,
            low_ratios,
            simulations=100,
            seed=1,
        )

        self.assertIsNone(note)
        self.assertEqual(set(probabilities.values()), {1.0})

    def test_entry_zone_touch_probability_unavailable_without_daily_ohlc(self):
        data = dict(self.ticker_data, _close_history=self.closes)
        result = build_entry_zone_analysis(StubEngine(), data, "Dip Hunter", simulations=100)

        self.assertEqual(result["touch_probability_status"], "unavailable")
        self.assertTrue(all(zone["touch_probability"] is None for zone in result["zones"]))
        self.assertIn("high/low history", result["touch_probability_note"])


if __name__ == "__main__":
    unittest.main()
