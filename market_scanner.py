"""Live technical-data scanner for the configured public constituent universe."""

import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from recommender import RISK_LEVELS, calculate_ma_proximity, generate_recommendations
from universe_explorer import get_market_universe, normalize_ticker

HISTORY_PERIOD = "2y"
DOWNLOAD_CHUNK_SIZE = 80
SCAN_CACHE_TTL_SECONDS = 15 * 60
LEADERBOARD_LIMIT = 15


def _extract_ticker_history(download, ticker):
    if download is None or download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        for level in range(download.columns.nlevels):
            if ticker in download.columns.get_level_values(level):
                try:
                    history = download.xs(ticker, axis=1, level=level, drop_level=True)
                    if isinstance(history.columns, pd.MultiIndex):
                        history.columns = history.columns.get_level_values(-1)
                    return history
                except KeyError:
                    continue
        return pd.DataFrame()
    return download


def _calculate_beta(stock_returns, market_returns):
    if market_returns is None or market_returns.empty:
        return 1.0
    common = stock_returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return 1.0
    stock = stock_returns.loc[common].to_numpy(dtype=float)
    market = market_returns.loc[common].to_numpy(dtype=float)
    variance = np.var(market)
    return float(np.cov(stock, market)[0, 1] / variance) if variance > 0 else 1.0


def build_live_feature_row(ticker, history, company, spy_history=None):
    """Build current scanner features from adjusted OHLCV history."""
    if history is None or history.empty or "Close" not in history:
        return None

    close = pd.to_numeric(history["Close"], errors="coerce")
    close = close[np.isfinite(close) & (close > 0)]
    if len(close) < 20:
        return None

    history = history.loc[close.index]
    price = float(close.iloc[-1])
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volume = (
        pd.to_numeric(history["Volume"], errors="coerce").fillna(0)
        if "Volume" in history.columns
        else pd.Series(0, index=close.index)
    )
    momentum = {}
    for key, sessions in (
        ("momentum_1mo", 21),
        ("momentum_3mo", 63),
        ("momentum_6mo", 126),
        ("momentum_1y", 252),
    ):
        momentum[key] = (
            float(price / close.iloc[-sessions] - 1)
            if len(close) >= sessions
            else np.nan
        )

    ma_50 = float(close.tail(50).mean()) if len(close) >= 50 else np.nan
    ma_200 = float(close.tail(200).mean()) if len(close) >= 200 else np.nan
    ma50_proximity = (
        calculate_ma_proximity(price, ma_50) if pd.notna(ma_50) else np.nan
    )
    ma200_proximity = (
        calculate_ma_proximity(price, ma_200) if pd.notna(ma_200) else np.nan
    )

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rsi = (
        float(100 - (100 / (1 + (gain / (loss + 1e-9)))))
        if pd.notna(gain)
        else 50.0
    )
    vol_window = min(len(returns), 252)
    annualized_vol = (
        float(returns.tail(vol_window).std() * np.sqrt(252))
        if len(returns) >= 20
        else np.nan
    )
    spy_returns = (
        spy_history.pct_change().dropna()
        if spy_history is not None and not spy_history.empty
        else None
    )
    beta = _calculate_beta(returns, spy_returns)
    current_volume = float(volume.iloc[-1]) if len(volume) else 0.0
    average_volume = float(volume.tail(20).mean()) if len(volume) >= 20 else current_volume
    perf_1w = float(price / close.iloc[-5] - 1) if len(close) >= 5 else np.nan

    return {
        "ticker": ticker,
        "name": company.get("name") or ticker,
        "sector": company.get("sector") or "N/A",
        "price_as_of": close.index[-1].isoformat(),
        "price": price,
        **momentum,
        "annualized_vol": annualized_vol,
        "rsi": rsi,
        "beta": beta,
        "perf_1w": perf_1w,
        "pe_ratio": np.nan,
        "peg_ratio": np.nan,
        "debt_to_equity": np.nan,
        "ma_200": ma_200,
        "ma_50": ma_50,
        "ma50_proximity": ma50_proximity,
        "ma200_proximity": ma200_proximity,
        "current_volume": current_volume,
        "avg_volume": average_volume,
    }


class AutomatedMarketScanner:
    """Runs one background batch scan and caches fresh features for profile reranking."""

    def __init__(
        self,
        engine,
        universe_loader=get_market_universe,
        history_downloader=None,
        cache_ttl_seconds=SCAN_CACHE_TTL_SECONDS,
    ):
        self.engine = engine
        self.universe_loader = universe_loader
        self.history_downloader = history_downloader or self._download_history
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.RLock()
        self._state = {
            "job_id": None,
            "status": "idle",
            "phase": "idle",
            "message": "No live market scan has been run yet.",
            "profile": None,
            "processed_count": 0,
            "scoreable_count": 0,
            "failed_count": 0,
            "total_in_universe": 0,
            "leaderboard": [],
            "last_updated": None,
            "source_fetched_at": None,
            "price_data_as_of": None,
            "source_counts": [],
            "score_baseline_source": None,
            "failures": [],
        }
        self._feature_frame = None
        self._universe_count = 0
        self._source_fetched_at = None
        self._feature_frame_updated_at = 0.0
        self._source_counts = []

    @staticmethod
    def _download_history(tickers):
        return yf.download(
            tickers=tickers,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
            timeout=30,
        )

    def get_scan(self, profile, force_refresh=False):
        if profile not in RISK_LEVELS:
            raise ValueError(f"Unknown scoring profile: {profile}")
        with self._lock:
            if self._state["status"] == "running":
                return self._public_state()
            if self._state["status"] == "error" and not force_refresh:
                return self._public_state()
            cache_is_fresh = (
                self._feature_frame is not None
                and time.time() - self._feature_frame_updated_at < self.cache_ttl_seconds
            )
            if force_refresh or not cache_is_fresh:
                self._start_scan(profile)
                return self._public_state()
            if self._state["profile"] != profile:
                self._rank_features(profile)
            return self._public_state()

    def _start_scan(self, profile):
        self._state = {
            **self._state,
            "job_id": str(uuid.uuid4()),
            "status": "running",
            "phase": "loading_universe",
            "message": "Loading the configured public constituent universe…",
            "profile": profile,
            "processed_count": 0,
            "scoreable_count": 0,
            "failed_count": 0,
            "total_in_universe": 0,
            "leaderboard": [],
            "last_updated": None,
            "source_fetched_at": None,
            "price_data_as_of": None,
            "source_counts": [],
            "score_baseline_source": None,
            "failures": [],
        }
        thread = threading.Thread(
            target=self._run_scan,
            args=(self._state["job_id"], profile),
            daemon=True,
            name="market-universe-scan",
        )
        thread.start()

    def _run_scan(self, job_id, profile):
        try:
            universe = self.universe_loader()
            companies = universe["companies"]
            deduplicated = {}
            for company in companies:
                ticker = normalize_ticker(company.get("ticker"))
                if ticker:
                    deduplicated.setdefault(ticker, company)
            tickers = sorted(deduplicated)
            source_counts = [
                {
                    "name": source["name"],
                    "count": source.get("count", 0),
                    "status": source.get("status", "available"),
                    "coverage_note": source.get("coverage_note"),
                }
                for source in universe.get("sources", [])
            ]
            with self._lock:
                if self._state["job_id"] != job_id:
                    return
                self._universe_count = len(tickers)
                self._source_counts = source_counts
                self._source_fetched_at = universe.get("fetched_at")
                self._state.update({
                    "phase": "downloading_prices",
                    "message": f"Downloading fresh {HISTORY_PERIOD} daily price history in batches…",
                    "total_in_universe": len(tickers),
                    "source_fetched_at": self._source_fetched_at,
                    "source_counts": source_counts,
                })

            spy_history = self.engine.spy_history
            rows = []
            failures = []
            processed = 0
            for start in range(0, len(tickers), DOWNLOAD_CHUNK_SIZE):
                batch = tickers[start:start + DOWNLOAD_CHUNK_SIZE]
                try:
                    downloaded = self.history_downloader(batch)
                except Exception as error:
                    failures.extend({"ticker": ticker, "error": str(error)} for ticker in batch)
                    processed += len(batch)
                    self._update_progress(job_id, processed, len(rows), len(failures), failures)
                    continue

                for ticker in batch:
                    company = deduplicated[ticker]
                    try:
                        history = _extract_ticker_history(downloaded, ticker)
                        feature_row = build_live_feature_row(
                            ticker,
                            history,
                            company,
                            spy_history=spy_history,
                        )
                        if feature_row is None:
                            failures.append({
                                "ticker": ticker,
                                "error": "No usable two-year price history",
                            })
                        else:
                            rows.append(feature_row)
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        failures.append({"ticker": ticker, "error": str(error)})
                processed += len(batch)
                self._update_progress(job_id, processed, len(rows), len(failures), failures)

            if not rows:
                raise RuntimeError(
                    "No tickers returned usable price history. Check Yahoo Finance availability and try again."
                )

            feature_frame = pd.DataFrame(rows)
            with self._lock:
                if self._state["job_id"] != job_id:
                    return
                self._feature_frame = feature_frame
                self._feature_frame_updated_at = time.time()
                self._universe_count = len(tickers)
                self._source_fetched_at = universe.get("fetched_at")
                self._source_counts = source_counts
                self._rank_features(profile)
                self._state.update({
                    "status": "complete",
                    "phase": "complete",
                    "message": (
                        f"Fresh technical scan complete: {len(rows):,} of {len(tickers):,} "
                        "universe tickers ranked."
                    ),
                    "processed_count": processed,
                    "failed_count": len(failures),
                    "failures": failures[:20],
                    "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
        except Exception as error:
            with self._lock:
                if self._state["job_id"] == job_id:
                    self._state.update({
                        "status": "error",
                        "phase": "error",
                        "message": str(error),
                        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })

    def _update_progress(self, job_id, processed, scoreable, failed, failures):
        with self._lock:
            if self._state["job_id"] != job_id:
                return
            total = self._state["total_in_universe"]
            self._state.update({
                "processed_count": processed,
                "scoreable_count": scoreable,
                "failed_count": failed,
                "failures": failures[:20],
                "message": f"Refreshing prices and indicators: {processed:,} of {total:,} tickers checked…",
            })

    def _rank_features(self, profile):
        if self._feature_frame is None or self._feature_frame.empty:
            return
        ranked = generate_recommendations(
            self._feature_frame,
            user_profile=profile,
            top_n=None,
        )
        ranked = ranked.sort_values(
            ["recommendation_score", "ticker"],
            ascending=[False, True],
            kind="stable",
        )
        results = []
        for rank, (_, row) in enumerate(ranked.head(LEADERBOARD_LIMIT).iterrows(), 1):
            forecast_features = {
                "sector": row.get("sector", "N/A"),
                "pe_ratio": None,
                "debt_to_equity": None,
            }
            for feature in (
                "ma50_proximity",
                "ma200_proximity",
                "momentum_1y",
                "rsi",
                "annualized_vol",
                "beta",
            ):
                value = row.get(feature)
                forecast_features[feature] = (
                    None if pd.isna(value) else float(value)
                )
            forecast = self.engine.predict_return_ols(forecast_features)
            expected_return = forecast.get("expected_return")
            if expected_return is not None and not np.isfinite(expected_return):
                expected_return = None
            prob_positive = forecast.get("prob_positive")
            if prob_positive is not None and not np.isfinite(prob_positive):
                prob_positive = None
            results.append({
                "rank": rank,
                "ticker": row["ticker"],
                "name": row["name"],
                "sector": row["sector"],
                "price": round(float(row["price"]), 2),
                "momentum_1y": (
                    round(float(row["momentum_1y"]) * 100, 1)
                    if pd.notna(row["momentum_1y"]) else None
                ),
                "rsi": round(float(row["rsi"]), 1) if pd.notna(row["rsi"]) else None,
                "pe_ratio": None,
                "score": round(float(row["recommendation_score"]) * 100, 1),
                "expected_return": expected_return,
                "prob_positive": prob_positive,
            })
        price_dates = pd.to_datetime(
            self._feature_frame["price_as_of"],
            errors="coerce",
            utc=True,
        ).dropna()
        self._state.update({
            "profile": profile,
            "leaderboard": results,
            "scoreable_count": len(ranked),
            "total_in_universe": self._universe_count,
            "source_fetched_at": self._source_fetched_at,
            "source_counts": self._source_counts,
            "score_baseline_source": "Live cross-sectional technical features",
            "price_data_as_of": (
                price_dates.max().isoformat(timespec="seconds")
                if not price_dates.empty else None
            ),
        })

    def _public_state(self):
        return {**self._state}
