"""
======================================================================
BASAK BOSSMAN ENGINE: WEB SERVER & REST API
======================================================================
High-performance multithreaded HTTP server serving the interactive
dashboard and REST APIs. Zero external framework dependencies;
runs on standard Python 3.
======================================================================
"""

import os
import sys
import json
import time
import mimetypes
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime

from engine_service import UniversalScoringEngine, apply_gics_classification
from market_scanner import AutomatedMarketScanner

from universe_explorer import (
    build_gics_sectors,
    build_gics_subindustries,
    get_gics_universe,
    normalize_ticker as normalize_universe_ticker,
)
from research_service import (
    ResearchConfigurationError,
    ResearchSourceError,
    answer_company_question,
    consume_research_rate_limit,
    get_research_status,
    is_valid_research_access_token,
    MAX_QUESTION_LENGTH,
    normalize_ticker,
    requires_market_analysis,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Singleton scoring engine instance
print("[Server] Initializing Universal Scoring Engine...")
ENGINE = UniversalScoringEngine()
print("[Server] Engine initialized successfully.")
MARKET_SCANNER = AutomatedMarketScanner(ENGINE)

# In-memory LRU cache for fetched ticker data (15-minute TTL)
TICKER_CACHE = {}
CACHE_TTL = 900  # 15 minutes in seconds
INDUSTRY_SCORE_CACHE = {}


def has_live_scoring_reference():
    reference_is_current = (
        ENGINE.scoring_reference_source == "Live scanner universe"
        and ENGINE.scoring_reference_df is not None
        and not ENGINE.scoring_reference_df.empty
        and ENGINE.scoring_reference_updated_at is not None
    )
    if not reference_is_current:
        return False
    return MARKET_SCANNER.has_fresh_scoring_reference()


class EngineRequestHandler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": True, "message": message}, status=status)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/research":
            return self.send_error_json("Not Found", status=404)

        content_length = self.headers.get("Content-Length", "")
        try:
            length = int(content_length)
        except ValueError:
            return self.send_error_json("A valid Content-Length header is required.")
        if length < 1 or length > 8192:
            return self.send_error_json("Research request must be between 1 and 8192 bytes.")

        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self.send_error_json("Request body must be valid JSON.")
        if not isinstance(payload, dict):
            return self.send_error_json("Request body must be a JSON object.")

        try:
            ticker = normalize_ticker(payload.get("ticker", ""))
            question = payload.get("question", "")
            status = get_research_status()
            if not status["ready"]:
                missing = ", ".join(status["missing"])
                raise ResearchConfigurationError(
                    f"Research is not configured. Set {missing} on the server."
                )
            if not consume_research_rate_limit(self.client_address[0]):
                return self.send_error_json(
                    "Research request limit reached. Try again after the hourly limit resets.",
                    status=429,
                )
            authorization = self.headers.get("Authorization", "")
            token = authorization.removeprefix("Bearer ").strip()
            if not is_valid_research_access_token(token):
                return self.send_error_json(
                    "A valid research access token is required for this request.",
                    status=401,
                )
            if not isinstance(question, str) or not question.strip():
                return self.send_error_json("Enter a research question.")
            if len(question) > MAX_QUESTION_LENGTH:
                return self.send_error_json(
                    f"Research questions must be {MAX_QUESTION_LENGTH:,} characters or fewer."
                )
            market_analysis = None
            market_error = None
            if requires_market_analysis(question):
                try:
                    now = time.time()
                    cached = TICKER_CACHE.get(ticker)
                    if cached and (now - cached["timestamp"]) < CACHE_TTL:
                        ticker_data = cached["data"]
                    else:
                        ticker_data = ENGINE.fetch_ticker_data(ticker)
                        TICKER_CACHE[ticker] = {
                            "data": ticker_data,
                            "timestamp": now,
                        }
                    market_analysis = ENGINE.analyze_entry_zones(
                        ticker_data,
                        profile_name="All-Weather Core",
                        simulations=2500,
                    )
                    market_analysis["price_as_of"] = ticker_data.get("price_as_of")
                except Exception as error:
                    sys.stderr.write(
                        f"[Research] Market/entry analysis unavailable for {ticker}: {error}\n"
                    )
                    market_error = (
                        "The market data provider or entry-zone calculation was unavailable."
                    )
            result = answer_company_question(
                ticker,
                question,
                market_analysis=market_analysis,
                market_error=market_error,
            )
            return self.send_json(result)
        except ValueError as error:
            return self.send_error_json(str(error))
        except ResearchConfigurationError as error:
            return self.send_error_json(str(error), status=503)
        except ResearchSourceError as error:
            return self.send_error_json(str(error), status=502)
        except Exception as error:
            sys.stderr.write(f"[Research] Unexpected research error: {error}\n")
            return self.send_error_json("Research failed unexpectedly. Check the server log.", status=500)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/research/status":
            return self.send_json(get_research_status())

        # ── API: Score Single Ticker ──────────────────────────────────
        if path == "/api/score":
            ticker = params.get("ticker", [""])[0].strip().upper()
            profile = params.get("profile", ["Dip Hunter"])[0].strip()
            force_refresh = params.get("refresh", ["0"])[0] == "1"
            gics_sector = params.get("gics_sector", [""])[0].strip()
            gics_subindustry = params.get("gics_subindustry", [""])[0].strip()

            if not ticker:
                return self.send_error_json("Missing 'ticker' parameter.")
            if profile not in RISK_LEVELS:
                return self.send_error_json(f"Unknown scoring profile: {profile}")
            if not has_live_scoring_reference():
                return self.send_error_json(
                    "A completed live universe scan is required before scoring. "
                    "Refresh the market scan and try again.",
                    status=503,
                )

            try:
                # Check cache for raw ticker metrics
                now = time.time()
                cached = TICKER_CACHE.get(ticker)
                if (
                    not force_refresh
                    and cached
                    and (now - cached["timestamp"]) < CACHE_TTL
                ):
                    ticker_data = cached["data"]
                else:
                    ticker_data = ENGINE.fetch_ticker_data(ticker)
                    TICKER_CACHE[ticker] = {"data": ticker_data, "timestamp": now}

                ticker_data = apply_gics_classification(
                    ticker_data,
                    gics_sector,
                    gics_subindustry,
                )
                result = ENGINE.score_ticker(ticker_data, profile_name=profile)
                result["market_regime"] = ENGINE.market_regime
                return self.send_json(result)
            except ValueError as error:
                return self.send_error_json(str(error), status=422)
            except Exception as e:
                return self.send_error_json(f"Failed to score '{ticker}': {str(e)}", status=500)

        # ── API: Historical Dip Hunter Entry Analysis ──────────────────
        elif path == "/api/entry-zones":
            ticker = params.get("ticker", [""])[0].strip().upper()
            profile = params.get("profile", ["Dip Hunter"])[0].strip()

            if not ticker:
                return self.send_error_json("Missing 'ticker' parameter.")
            if profile != "Dip Hunter":
                return self.send_error_json(
                    "Historical entry analysis is currently available for Dip Hunter only."
                )
            if not has_live_scoring_reference():
                return self.send_error_json(
                    "A completed live universe scan is required before entry analysis.",
                    status=503,
                )

            try:
                now = time.time()
                cached = TICKER_CACHE.get(ticker)
                if cached and (now - cached["timestamp"]) < CACHE_TTL:
                    ticker_data = cached["data"]
                else:
                    ticker_data = ENGINE.fetch_ticker_data(ticker)
                    TICKER_CACHE[ticker] = {"data": ticker_data, "timestamp": now}

                return self.send_json(ENGINE.analyze_entry_zones(ticker_data, profile))
            except Exception as e:
                return self.send_error_json(
                    f"Failed to analyze entry zones for '{ticker}': {str(e)}",
                    status=500,
                )

        # ── API: Multi-Ticker Compare ─────────────────────────────────
        elif path == "/api/compare":
            tickers_raw = params.get("tickers", [""])[0]
            profile = params.get("profile", ["Dip Hunter"])[0].strip()

            tickers_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            if not tickers_list:
                return self.send_error_json("Please provide at least one ticker in 'tickers'.")
            if profile not in RISK_LEVELS:
                return self.send_error_json(f"Unknown scoring profile: {profile}")
            if not has_live_scoring_reference():
                return self.send_error_json(
                    "A completed live universe scan is required before comparison.",
                    status=503,
                )

            results = []
            now = time.time()
            for t in tickers_list[:6]:  # Limit to max 6 tickers for clean comparison
                try:
                    cached = TICKER_CACHE.get(t)
                    if cached and (now - cached["timestamp"]) < CACHE_TTL:
                        t_data = cached["data"]
                    else:
                        t_data = ENGINE.fetch_ticker_data(t)
                        TICKER_CACHE[t] = {"data": t_data, "timestamp": now}

                    scored = ENGINE.score_ticker(t_data, profile_name=profile)
                    results.append(scored)
                except Exception as e:
                    results.append({"ticker": t, "error": str(e)})

            return self.send_json({"profile": profile, "comparison": results})

        # ── API: Universe Scan Leaderboard ────────────────────────────
        elif path == "/api/scan":
            profile = params.get("profile", ["Dip Hunter"])[0].strip()
            if profile not in RISK_LEVELS:
                return self.send_error_json(f"Unknown scoring profile: {profile}")
            refresh = params.get("refresh", ["false"])[0].lower() == "true"
            try:
                scan = MARKET_SCANNER.get_scan(profile, force_refresh=refresh)
            except ValueError as error:
                return self.send_error_json(str(error))
            scan["market_regime"] = ENGINE.market_regime
            return self.send_json(scan)

        # ── API: Free Public-Source Industry Explorer ─────────────────
        elif path == "/api/industry-explorer":
            profile = params.get("profile", ["Dip Hunter"])[0].strip()
            requested_sector = params.get("sector", [""])[0]
            requested_subindustry = params.get("subindustry", [""])[0]
            company_search = params.get("search", [""])[0].strip()[:100]
            force_refresh = params.get("refresh", [""])[0].lower() == "true"
            if profile not in RISK_LEVELS:
                return self.send_error_json(f"Unknown scoring profile: {profile}")
            if not has_live_scoring_reference():
                return self.send_error_json(
                    "A completed live universe scan is required before showing live company scores.",
                    status=503,
                )

            try:
                universe = get_gics_universe(force_refresh=force_refresh)
                gics_companies = universe["companies"]
                sectors = build_gics_sectors(gics_companies)
                selected_sector = next(
                    (sector["name"] for sector in sectors if sector["name"] == requested_sector),
                    None,
                )
                if requested_sector and selected_sector is None:
                    return self.send_error_json("The selected GICS sector is no longer available.")
                if selected_sector is None and sectors:
                    selected_sector = sectors[0]["name"]
                subindustries = (
                    build_gics_subindustries(gics_companies, selected_sector)
                    if selected_sector else []
                )
                selected_subindustry = next(
                    (
                        industry["name"] for industry in subindustries
                        if industry["name"] == requested_subindustry
                    ),
                    None,
                )
                if requested_subindustry and selected_subindustry is None:
                    return self.send_error_json(
                        "The selected GICS sub-industry is not available in this sector."
                    )

                cache_key = (profile, ENGINE.scoring_reference_updated_at)
                if cache_key not in INDUSTRY_SCORE_CACHE:
                    score_rows = ENGINE.scan_universe(
                        profile_name=profile,
                        top_n=len(ENGINE.scoring_reference_df),
                    )
                    INDUSTRY_SCORE_CACHE[cache_key] = {
                        normalize_ticker(row["ticker"]): {
                            "score": row["score"],
                            "rank": row["rank"],
                        }
                        for row in score_rows
                        if normalize_ticker(row.get("ticker"))
                    }
                scores = INDUSTRY_SCORE_CACHE[cache_key]
                if company_search:
                        from universe_explorer import search_gics_companies

                        matching_companies = search_gics_companies(
                            gics_companies, company_search
                        )
                else:
                    matching_companies = [
                        company for company in gics_companies
                        if company["sector"] == selected_sector
                        and selected_subindustry is not None
                        and company["industry"] == selected_subindustry
                    ]
                selected_companies = [
                    {
                        **company,
                        **scores.get(company["ticker"], {}),
                    }
                    for company in matching_companies
                ]
                selected_companies.sort(
                    key=lambda company: (
                        company.get("rank") is None,
                        company.get("rank", float("inf")),
                        company["name"].casefold(),
                    )
                )

                return self.send_json({
                    "profile": profile,
                    "sectors": sectors,
                    "selected_sector": selected_sector,
                    "subindustries": subindustries,
                    "selected_subindustry": selected_subindustry,
                    "companies": selected_companies,
                    "search_query": company_search or None,
                    "search_result_count": len(selected_companies) if company_search else None,
                    "unique_company_count": len(gics_companies),
                    "selected_sector_company_count": sum(
                        company["sector"] == selected_sector for company in gics_companies
                    ),
                    "score_baseline_count": len(scores),
                    "score_company_count": sum(
                        company["ticker"] in scores for company in selected_companies
                    ),
                    "score_baseline_source": ENGINE.scoring_reference_source,
                    "sources": universe["sources"],
                    "fetched_at": universe["fetched_at"],
                    "refresh_error": universe.get("refresh_error"),
                })
            except RuntimeError as error:
                return self.send_error_json(
                    f"Could not load the public company universe: {error}",
                    status=503,
                )

        # ── API: Market Regime & SPY Status ───────────────────────────
        elif path == "/api/market-regime":
            return self.send_json(ENGINE.market_regime)

        # ── API: Profile Metadata ─────────────────────────────────────
        elif path == "/api/profiles":
            from recommender import RISK_LEVELS
            profiles_meta = {
                "Dip Hunter": {
                    "tagline": "Turnaround & Mean Reversion",
                    "description": "Rewards deeply discounted stocks, 3-month momentum turnarounds, and oversold pullbacks below 200MA.",
                    "focus": "Inverted 1y momentum, Deep Dip MA200 bonus, Falling knife guard",
                },
                "Alpha Bull": {
                    "tagline": "Aggressive Trend Momentum",
                    "description": "High-conviction trend follower riding above 50MA and 200MA with high beta amplification.",
                    "focus": "MA50 & MA200 proximity, High Beta bonus, Room-to-run RSI",
                },
                "All-Weather Core": {
                    "tagline": "Risk-Adjusted Compounders",
                    "description": "Volatility-penalized core strategy targeting steady outperforming compounders with low downside risk.",
                    "focus": "46.3% Risk Weight, Balanced MA trend, Dynamic gear shift",
                },
                "Bear Market Shield": {
                    "tagline": "Capital Preservation",
                    "description": "Defensive posture prioritizing low volatility, inverted momentum, and strict valuation ceilings.",
                    "focus": "Capital protection, Inverted momentum, Solvency guard",
                },
                "Fundamental Bag": {
                    "tagline": "Deep Value Discipline",
                    "description": "Value-oriented strategy rewarding disciplined P/E multiples, PEG ratios, and low structural debt.",
                    "focus": "Valuation shields, PEG under 1.0, Solvency discipline",
                },
            }
            return self.send_json(profiles_meta)

        # ── Static File Serving ───────────────────────────────────────
        if path == "/" or path == "/index.html":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            rel_path = path.lstrip("/")
            file_path = os.path.join(STATIC_DIR, rel_path)

        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error_json(f"Error reading file: {e}", status=500)
        else:
            self.send_error_json("Not Found", status=404)

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}\n")


def run_server(port=8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, EngineRequestHandler)
    print(f"\n{'='*70}")
    print(f"🚀 BASAK BOSSMAN ENGINE WEB APP RUNNING")
    print(f"📡 URL: http://localhost:{port}")
    print(f"{'='*70}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
