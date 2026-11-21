import unittest
from unittest.mock import patch

import pandas as pd

import regime_factor_backtest
import recommender
from alpha_bull_weight_experiment import (
    VARIANTS,
    _score_metrics,
    _score_with_weights,
    summarize_comparison,
)
from regime_factor_backtest import (
    _monotonicity,
    classify_regime,
    parse_constituent_tickers,
    score_month_factors,
)
from audit_regime_backtest_coverage import classify_missing_ticker


class RegimeFactorBacktestTests(unittest.TestCase):
    def test_crisis_vix_override_takes_precedence_over_breadth(self):
        self.assertEqual(
            classify_regime(80, 55, True),
            "Systemic Crisis (VIX > 55 Override)",
        )

    def test_panic_vix_override_takes_precedence_over_breadth(self):
        self.assertEqual(
            classify_regime(80, 35, False),
            "Panic Capitulation (VIX > 35 Override)",
        )

    def test_breadth_and_equal_weight_relative_strength_select_live_states(self):
        self.assertEqual(classify_regime(34.9, 20, False), "Oversold Breadth")
        self.assertEqual(classify_regime(40, 20, False), "Narrow Breadth Divergence")
        self.assertEqual(classify_regime(40, 20, True), "Selective Consolidation")
        self.assertEqual(classify_regime(50, 20, True), "Broad Market Expansion")
        self.assertEqual(classify_regime(50, 20, False), "Bull Momentum Expansion")

    def test_monotonicity_checks_low_to_high_factor_bins(self):
        increasing = pd.Series(
            {
                "q1_highest_return": 0.05,
                "q2_return": 0.04,
                "q3_return": 0.03,
                "q4_return": 0.02,
                "q5_lowest_return": 0.01,
            }
        )
        self.assertEqual(_monotonicity(increasing), "Strictly increasing")

        mixed = increasing.copy()
        mixed["q2_return"] = 0.0
        self.assertEqual(_monotonicity(mixed), "Generally increasing")

    def test_parses_legacy_plain_ticker_cells(self):
        wikitext = "| [[3M Company]] || MMM || reports || Industrials"
        self.assertEqual(parse_constituent_tickers(wikitext), ["MMM"])

    def test_parses_ticker_before_company_cell(self):
        wikitext = "| MMM || [[3M Company]] || reports || Industrials"
        self.assertEqual(parse_constituent_tickers(wikitext), ["MMM"])

    def test_parses_ticker_templates_in_archived_table_formats(self):
        wikitext = (
            "| {{NyseSymbol|MMM}} || [[3M Company]] || Industrials\n"
            "|| {{NasdaqSymbol|ABCD}}"
        )
        self.assertEqual(parse_constituent_tickers(wikitext), ["ABCD", "MMM"])

    def test_missing_ticker_audit_only_marks_verified_format_candidates(self):
        self.assertEqual(
            classify_missing_ticker("BRKB", False),
            ("Potential ticker-format repair", "BRK-B"),
        )
        self.assertEqual(
            classify_missing_ticker("WBD", True),
            ("Latest constituent; provider/history gap needs investigation", ""),
        )
        self.assertEqual(
            classify_missing_ticker("ABC", False),
            ("No returned series; likely delisted/renamed or provider coverage gap", ""),
        )

    def test_monthly_quintile_returns_use_month_end_prices(self):
        month_end = pd.Timestamp("2024-01-31")
        next_month_end = pd.Timestamp("2024-02-29")
        tickers = ["A", "B", "C", "D", "E"]
        features = pd.DataFrame(
            {"ticker": tickers, "factor": [1, 2, 3, 4, 5]}
        )
        daily_prices = pd.DataFrame(
            {"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0], "E": [10.0]},
            index=[pd.Timestamp("2024-01-30")],
        )
        month_end_prices = pd.DataFrame(
            {
                "A": [10.0, 10.0],
                "B": [10.0, 11.0],
                "C": [10.0, 12.0],
                "D": [10.0, 13.0],
                "E": [10.0, 14.0],
            },
            index=[month_end, next_month_end],
        )
        spy_returns = pd.Series(dtype=float)

        with (
            patch.object(regime_factor_backtest, "FACTORS_TO_TEST", ["factor"]),
            patch.object(
                regime_factor_backtest,
                "build_features_for_month",
                return_value=features,
            ),
        ):
            result = score_month_factors(
                month_end,
                next_month_end,
                tickers,
                daily_prices,
                month_end_prices,
                spy_returns,
            )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["q5_lowest_return"], 0.0)
        self.assertAlmostEqual(result[0]["q1_highest_return"], 0.4)
        self.assertAlmostEqual(result[0]["spread_q1_minus_q5"], 0.4)

    def test_alpha_bull_weight_variants_preserve_one_hundred_percent_budget(self):
        for weights in VARIANTS.values():
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_experimental_scoring_does_not_mutate_production_weights(self):
        tickers = [f"T{i}" for i in range(10)]
        features = pd.DataFrame(
            {
                "ticker": tickers,
                "sector": ["N/A"] * 10,
                "price": [100 + i for i in range(10)],
                "ma_50": [100.0] * 10,
                "ma_200": [100.0] * 10,
                "ma50_proximity": [i / 10 for i in range(10)],
                "ma200_proximity": [i / 10 for i in range(10)],
                "momentum_1mo": [i / 20 for i in range(10)],
                "momentum_3mo": [i / 15 for i in range(10)],
                "momentum_6mo": [i / 12 for i in range(10)],
                "momentum_1y": [i / 10 for i in range(10)],
                "annualized_vol": [0.2] * 10,
                "rsi": [50.0] * 10,
                "beta": [1.0] * 10,
                "perf_1w": [0.0] * 10,
                "pe_ratio": [20.0] * 10,
                "debt_to_equity": [0.0] * 10,
                "short_percent": [0.0] * 10,
                "current_volume": [1_000_000] * 10,
                "avg_volume": [1_000_000] * 10,
            }
        )
        production_weights = {
            key: recommender.RISK_LEVELS["Alpha Bull"].get(key)
            for key in ("ma50_proximity_weight", "ma200_proximity_weight", "momentum_1y_weight")
        }

        scored = _score_with_weights(features, VARIANTS["Momentum +10pp"])

        self.assertEqual(len(scored), len(features))
        self.assertEqual(
            production_weights,
            {
                key: recommender.RISK_LEVELS["Alpha Bull"].get(key)
                for key in production_weights
            },
        )

    def test_score_metrics_uses_highest_and_lowest_score_quintiles(self):
        scores = pd.DataFrame(
            {
                "recommendation_score": range(10),
                "forward_return": [0.01 * value for value in range(10)],
            }
        )
        metrics = _score_metrics(scores)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["n_stocks"], 10)
        self.assertAlmostEqual(
            metrics["spread_q1_minus_q5"], metrics["q1_highest_return"] - metrics["q5_lowest_return"]
        )


if __name__ == "__main__":
    unittest.main()
