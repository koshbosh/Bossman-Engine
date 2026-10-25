import unittest

import numpy as np

from entry_zones import (
    FORWARD_HORIZONS,
    _historical_limit_order_estimates,
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
        self.opens = self.closes.copy()
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

    def test_historical_replay_estimates_and_ranks_filled_orders(self):
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0004, 0.01, 700)
        closes = 100 * np.exp(np.cumsum(returns))
        opens = np.r_[closes[0], closes[:-1]]
        highs = closes * 2
        lows = closes * 0.5

        estimates, note = _historical_limit_order_estimates(closes, opens, highs, lows)

        self.assertIsNone(note)
        self.assertIsNotNone(estimates)
        for horizon in FORWARD_HORIZONS:
            results = [estimates[center][horizon] for center in estimates]
            self.assertTrue(all(result["fill_probability"] == 1 for result in results))
            self.assertTrue(all(result["expected_return"] is not None for result in results))
            self.assertEqual(sorted(result["rank"] for result in results), list(range(1, 10)))
            self.assertTrue(all(result["expected_value"] is not None for result in results))

    def test_orders_not_filled_within_five_sessions_have_zero_fill_probability(self):
        sessions = np.arange(400)
        closes = 100 * np.exp(0.0002 * sessions + 0.00001 * np.sin(sessions))
        opens = np.r_[closes[0], closes[:-1]]
        highs = closes * 1.000001
        lows = closes * 0.99999

        estimates, note = _historical_limit_order_estimates(
            closes,
            opens,
            highs,
            lows,
            zone_centers=(0.5,),
            horizons=(21,),
            min_filled_samples=1,
        )

        self.assertIsNone(note)
        result = estimates[0.5][21]
        self.assertEqual(result["fill_probability"], 0)
        self.assertIsNone(result["expected_return"])
        self.assertEqual(result["filled_samples"], 0)

    def test_limit_order_expires_after_five_sessions(self):
        rng = np.random.default_rng(9)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 253)))
        anchor_price = closes[-1]
        post_anchor = np.full(94, anchor_price)
        post_anchor[:5] = anchor_price * 1.1
        post_anchor[5] = anchor_price * 0.5
        closes = np.r_[closes, post_anchor]
        opens = np.r_[closes[0], closes[:-1]]
        highs = closes * 1.1
        lows = closes * 0.99

        estimates, note = _historical_limit_order_estimates(
            closes,
            opens,
            highs,
            lows,
            zone_centers=(0.5,),
            horizons=(90,),
            min_filled_samples=1,
        )

        self.assertIsNone(note)
        self.assertEqual(estimates[0.5][90]["opportunity_count"], 1)
        self.assertEqual(estimates[0.5][90]["filled_samples"], 0)

    def test_above_spot_buy_stop_fills_on_high_crossing_and_measures_from_trigger(self):
        closes = 100 * np.exp(0.0001 * np.arange(253) + 0.005 * np.sin(np.arange(253) / 4))
        anchor_price = closes[-1]
        trigger = anchor_price * np.exp(
            0.5 * np.std(np.diff(np.log(closes[-253:])), ddof=1) * np.sqrt(21)
        )
        future_closes = np.full(25, anchor_price)
        future_closes[0] = anchor_price
        future_closes[1] = trigger
        future_closes[2:] = trigger * 1.10
        closes = np.r_[closes, future_closes]
        opens = np.r_[closes[0], closes[:-1]]
        highs = closes.copy()
        highs[254] = trigger * 1.001
        lows = closes * 0.999

        estimates, note = _historical_limit_order_estimates(
            closes,
            opens,
            highs,
            lows,
            zone_centers=(-0.5,),
            horizons=(21,),
            min_filled_samples=1,
        )

        self.assertIsNone(note)
        estimate = estimates[-0.5][21]
        self.assertEqual(estimate["filled_samples"], 1)
        self.assertAlmostEqual(estimate["expected_return"], 10.0, places=5)

    def test_historical_replay_requires_full_lookback_and_forward_window(self):
        estimates, note = _historical_limit_order_estimates(
            np.ones(300),
            np.ones(300),
            np.ones(300),
            np.ones(300),
        )

        self.assertIsNone(estimates)
        self.assertIn("sessions are required", note)

    def test_entry_zone_output_uses_historical_results_and_marks_scope(self):
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0004, 0.01, 700)
        closes = 100 * np.exp(np.cumsum(returns))
        opens = np.r_[closes[0], closes[:-1]]
        highs = closes * 2
        lows = closes * 0.5
        data = dict(
            self.ticker_data,
            price=float(closes[-1]),
            annualized_vol=float(np.std(returns[-252:]) * np.sqrt(252)),
            _open_history=opens.tolist(),
            _close_history=closes.tolist(),
            _high_history=highs.tolist(),
            _low_history=lows.tolist(),
        )
        result = build_entry_zone_analysis(StubEngine(), data, "Dip Hunter")

        self.assertEqual(len(result["zones"]), 9)
        self.assertEqual(result["calibration"]["status"], "exploratory")
        self.assertEqual(result["calibration"]["scope"], "single_ticker_historical_replay")
        self.assertEqual(result["order_open_days"], 5)
        self.assertEqual(result["horizons_days"], [21, 42, 63, 90])
        self.assertTrue(all(zone["horizons"][0]["expected_return"] is not None for zone in result["zones"]))
        self.assertTrue(all(zone["lower_price"] < zone["upper_price"] for zone in result["zones"]))
        self.assertEqual(
            {zone["entry_type"] for zone in result["zones"]},
            {"Buy stop", "Market at next open", "Limit buy"},
        )

    def test_entry_zone_analysis_rejects_unimplemented_profiles(self):
        with self.assertRaisesRegex(ValueError, "Dip Hunter only"):
            build_entry_zone_analysis(StubEngine(), self.ticker_data, "All-Weather Core")


if __name__ == "__main__":
    unittest.main()
