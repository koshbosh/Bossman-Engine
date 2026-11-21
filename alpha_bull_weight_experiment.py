"""Compare offline Alpha Bull base-weight alternatives in Bull Momentum Expansion.

Usage:
    python alpha_bull_weight_experiment.py

No production scoring configuration is changed. The run compares the current
Alpha Bull base weights with pre-specified 10- and 20-point transfers from
50-day MA proximity to 1-year momentum.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import recommender
from regime_factor_backtest import PRICE_CACHE_DIR, OUTPUT_DIR
from score_backtest import build_features_for_month


REGIME_MONTHLY_PATH = OUTPUT_DIR / "regime_factor_regimes_monthly.csv"
MONTHLY_COMPARISON_PATH = OUTPUT_DIR / "alpha_bull_weight_monthly.csv"
SUMMARY_PATH = OUTPUT_DIR / "alpha_bull_weight_summary.csv"
METADATA_PATH = OUTPUT_DIR / "alpha_bull_weight_metadata.json"
PRICE_CACHE_PATH = PRICE_CACHE_DIR / "adjusted_close_2006-01-01_2026-10-01.pkl"
CONSTITUENT_CACHE_DIR = PRICE_CACHE_DIR / "constituents"
TARGET_REGIME = "Bull Momentum Expansion"

VARIANTS = {
    "Baseline": {
        "ma50_proximity_weight": 0.642,
        "ma200_proximity_weight": 0.358,
        "momentum_1y_weight": 0.000,
    },
    "Momentum +10pp": {
        "ma50_proximity_weight": 0.542,
        "ma200_proximity_weight": 0.358,
        "momentum_1y_weight": 0.100,
    },
    "Momentum +20pp": {
        "ma50_proximity_weight": 0.442,
        "ma200_proximity_weight": 0.358,
        "momentum_1y_weight": 0.200,
    },
}

TIME_SPLITS = {
    "Exploration 2007-2016": (2007, 2016),
    "Validation 2017-2021": (2017, 2021),
    "Holdout 2022-2026": (2022, 2026),
}


def _score_with_weights(features, weights):
    """Run the production recommendation function with an isolated config copy."""
    original_levels = recommender.RISK_LEVELS
    experimental_levels = copy.deepcopy(original_levels)
    config = experimental_levels["Alpha Bull"]
    config.update(weights)

    # Historical sector and fundamental data are not available in this panel.
    config["sector_outperformance_bonus_active"] = False
    config["low_pe_bonus_active"] = False
    config["peg_bonus_active"] = False

    try:
        recommender.RISK_LEVELS = experimental_levels
        return recommender.generate_recommendations(
            features,
            user_profile="Alpha Bull",
            top_n=None,
            pe_benchmarks={"N/A": 20.0},
        )
    finally:
        recommender.RISK_LEVELS = original_levels


def _score_metrics(scores):
    valid = scores.dropna(subset=["recommendation_score", "forward_return"]).copy()
    if len(valid) < 5 or valid["recommendation_score"].nunique() < 5:
        return None
    valid["quintile"] = pd.qcut(
        valid["recommendation_score"],
        q=5,
        labels=False,
        duplicates="drop",
    )
    if valid["quintile"].nunique() != 5:
        return None
    quintile_returns = valid.groupby("quintile")["forward_return"].mean()
    if len(quintile_returns) != 5:
        return None
    return {
        "n_stocks": int(len(valid)),
        "q1_highest_return": float(quintile_returns.loc[4]),
        "q2_return": float(quintile_returns.loc[3]),
        "q3_return": float(quintile_returns.loc[2]),
        "q4_return": float(quintile_returns.loc[1]),
        "q5_lowest_return": float(quintile_returns.loc[0]),
        "spread_q1_minus_q5": float(
            quintile_returns.loc[4] - quintile_returns.loc[0]
        ),
        "spearman_ic": float(
            valid["recommendation_score"].rank().corr(
                valid["forward_return"].rank(), method="pearson"
            )
        ),
    }


def _strict_monotonicity(summary_row):
    values = [
        summary_row["mean_q5_lowest_return"],
        summary_row["mean_q4_return"],
        summary_row["mean_q3_return"],
        summary_row["mean_q2_return"],
        summary_row["mean_q1_highest_return"],
    ]
    if all(left < right for left, right in zip(values, values[1:])):
        return "Strictly increasing"
    if all(left > right for left, right in zip(values, values[1:])):
        return "Strictly decreasing"
    if values[-1] > values[0]:
        return "Generally increasing"
    if values[-1] < values[0]:
        return "Generally decreasing"
    return "Mixed or flat"


def summarize_comparison(monthly_results):
    if monthly_results.empty:
        return pd.DataFrame()

    summary_rows = []
    metric_columns = [
        "q1_highest_return",
        "q2_return",
        "q3_return",
        "q4_return",
        "q5_lowest_return",
        "spread_q1_minus_q5",
        "spearman_ic",
    ]
    filtered = monthly_results.copy()
    filtered["year"] = filtered["month"].str[:4].astype(int)

    for split_name, (start_year, end_year) in TIME_SPLITS.items():
        split_data = filtered[filtered["year"].between(start_year, end_year)]
        for variant, group in split_data.groupby("variant", sort=False):
            means = group[metric_columns].mean()
            spread = group["spread_q1_minus_q5"]
            row = {
                "time_split": split_name,
                "variant": variant,
                "start_year": start_year,
                "end_year": end_year,
                "months": int(len(group)),
                "median_stocks_per_month": int(group["n_stocks"].median()),
                "mean_q1_highest_return": float(means["q1_highest_return"]),
                "mean_q2_return": float(means["q2_return"]),
                "mean_q3_return": float(means["q3_return"]),
                "mean_q4_return": float(means["q4_return"]),
                "mean_q5_lowest_return": float(means["q5_lowest_return"]),
                "mean_q1_minus_q5_spread": float(means["spread_q1_minus_q5"]),
                "median_monthly_spread": float(spread.median()),
                "mean_monthly_spearman_ic": float(means["spearman_ic"]),
                "positive_spread_months": int((spread > 0).sum()),
                "positive_spread_fraction": float((spread > 0).mean()),
            }
            row["mean_quintile_monotonicity"] = _strict_monotonicity(
                {
                    "mean_q1_highest_return": row["mean_q1_highest_return"],
                    "mean_q2_return": row["mean_q2_return"],
                    "mean_q3_return": row["mean_q3_return"],
                    "mean_q4_return": row["mean_q4_return"],
                    "mean_q5_lowest_return": row["mean_q5_lowest_return"],
                }
            )
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["variant"] == "Baseline"][
        ["time_split", "mean_q1_minus_q5_spread", "mean_monthly_spearman_ic"]
    ].rename(
        columns={
            "mean_q1_minus_q5_spread": "baseline_mean_spread",
            "mean_monthly_spearman_ic": "baseline_mean_ic",
        }
    )
    summary = summary.merge(baseline, on="time_split", how="left", validate="many_to_one")
    summary["spread_change_vs_baseline"] = (
        summary["mean_q1_minus_q5_spread"] - summary["baseline_mean_spread"]
    )
    summary["ic_change_vs_baseline"] = (
        summary["mean_monthly_spearman_ic"] - summary["baseline_mean_ic"]
    )
    return summary.drop(columns=["baseline_mean_spread", "baseline_mean_ic"])


def run_experiment():
    for path in (REGIME_MONTHLY_PATH, PRICE_CACHE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required backtest input is missing: {path}")

    regimes = pd.read_csv(REGIME_MONTHLY_PATH)
    bull_months = regimes.loc[
        regimes["regime"] == TARGET_REGIME, "month"
    ].drop_duplicates()
    if bull_months.empty:
        raise RuntimeError(f"No months found for target regime {TARGET_REGIME}.")

    prices = pd.read_pickle(PRICE_CACHE_PATH)
    monthly_prices = prices.resample("ME").last()
    spy_returns = prices["SPY"].pct_change().dropna()
    original_levels = recommender.RISK_LEVELS

    month_results = []
    try:
        for month in bull_months:
            period = pd.Period(month, freq="M")
            current_dates = monthly_prices.index[
                monthly_prices.index.to_period("M") == period
            ]
            next_dates = monthly_prices.index[
                monthly_prices.index.to_period("M") == period + 1
            ]
            snapshot_path = CONSTITUENT_CACHE_DIR / f"constituents_{month}.json"
            if len(current_dates) != 1 or len(next_dates) != 1:
                print(f"Skipping {month}: month-end price data unavailable.")
                continue
            if not snapshot_path.exists():
                raise FileNotFoundError(f"Constituent snapshot is missing: {snapshot_path}")

            snapshot = json.loads(snapshot_path.read_text())
            date = current_dates[0]
            next_date = next_dates[0]
            history = prices.loc[:date]
            features = build_features_for_month(
                date,
                history,
                spy_returns,
                snapshot["tickers"],
                sector_map={},
            )
            if features.empty:
                print(f"Skipping {month}: feature history unavailable.")
                continue

            current_prices = monthly_prices.loc[date]
            next_prices = monthly_prices.loc[next_date]
            features = features.copy()
            features["forward_return"] = features["ticker"].map(
                lambda ticker: (
                    next_prices.get(ticker, np.nan)
                    / current_prices.get(ticker, np.nan)
                    - 1
                    if pd.notna(current_prices.get(ticker, np.nan))
                    and current_prices.get(ticker, 0) > 0
                    and pd.notna(next_prices.get(ticker, np.nan))
                    else np.nan
                )
            )
            features = features.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["forward_return"]
            )
            if len(features) < 25:
                print(f"Skipping {month}: only {len(features)} stocks have returns.")
                continue

            for variant, weights in VARIANTS.items():
                scored = _score_with_weights(features, weights)
                scored = scored[["ticker", "recommendation_score"]].merge(
                    features[["ticker", "forward_return"]],
                    on="ticker",
                    how="inner",
                    validate="one_to_one",
                )
                metrics = _score_metrics(scored)
                if metrics is None:
                    print(f"Skipping {month}/{variant}: cannot form five score quintiles.")
                    continue
                month_results.append(
                    {
                        "month": month,
                        "regime": TARGET_REGIME,
                        "variant": variant,
                        **metrics,
                    }
                )
            print(f"Compared all Alpha Bull variants for {month} ({len(features)} stocks)")
    finally:
        recommender.RISK_LEVELS = original_levels

    monthly = pd.DataFrame(month_results)
    if monthly.empty:
        raise RuntimeError("The Alpha Bull experiment produced no usable results.")
    summary = summarize_comparison(monthly)
    if summary.empty:
        raise RuntimeError("No time-split summaries could be calculated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(MONTHLY_COMPARISON_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    metadata = {
        "experiment": "Alpha Bull base-weight sensitivity in Bull Momentum Expansion",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_regime": TARGET_REGIME,
        "score_function": "recommender.generate_recommendations",
        "production_profile_weights": VARIANTS["Baseline"],
        "variants": VARIANTS,
        "method": (
            "For each point-in-time S&P 500 constituent month, score with the "
            "existing Alpha Bull scorer and compute following-month adjusted "
            "month-end return by recommendation-score quintile."
        ),
        "common_adjustments": [
            "Historical sector classifications and fundamentals are unavailable; "
            "sector, low-PE, and PEG bonuses are disabled identically for all variants.",
            "Other existing Alpha Bull price-based scoring adjustments are kept the same.",
        ],
        "time_splits": {
            name: {"start_year": years[0], "end_year": years[1]}
            for name, years in TIME_SPLITS.items()
        },
        "months_tested": int(monthly["month"].nunique()),
        "latest_tested_month": str(monthly["month"].max()),
        "price_coverage_warning": (
            "This experiment reuses the incomplete historical Yahoo Finance "
            "price panel. Results are exploratory and do not justify live weight changes."
        ),
        "interpretation_warning": (
            "The 2022-2026 holdout is small and should not be used to repeatedly "
            "select weights; all reported comparisons are descriptive."
        ),
        "outputs": [
            str(MONTHLY_COMPARISON_PATH),
            str(SUMMARY_PATH),
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Monthly comparison saved to {MONTHLY_COMPARISON_PATH}")
    print(f"Time-split summary saved to {SUMMARY_PATH}")
    print(f"Experiment metadata saved to {METADATA_PATH}")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_experiment()
