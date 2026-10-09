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

from engine_service import UniversalScoringEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Singleton scoring engine instance
print("[Server] Initializing Universal Scoring Engine...")
ENGINE = UniversalScoringEngine()
print("[Server] Engine initialized successfully.")

# In-memory LRU cache for fetched ticker data (15-minute TTL)
TICKER_CACHE = {}
CACHE_TTL = 900  # 15 minutes in seconds


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

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # ── API: Score Single Ticker ──────────────────────────────────
        if path == "/api/score":
            ticker = params.get("ticker", [""])[0].strip().upper()
            profile = params.get("profile", ["Dip Hunter"])[0].strip()

            if not ticker:
                return self.send_error_json("Missing 'ticker' parameter.")

            try:
                # Check cache for raw ticker metrics
                now = time.time()
                cached = TICKER_CACHE.get(ticker)
                if cached and (now - cached["timestamp"]) < CACHE_TTL:
                    ticker_data = cached["data"]
                else:
                    ticker_data = ENGINE.fetch_ticker_data(ticker)
                    TICKER_CACHE[ticker] = {"data": ticker_data, "timestamp": now}

                result = ENGINE.score_ticker(ticker_data, profile_name=profile)
                result["market_regime"] = ENGINE.market_regime
                return self.send_json(result)
            except Exception as e:
                return self.send_error_json(f"Failed to score '{ticker}': {str(e)}", status=500)

        # ── API: Multi-Ticker Compare ─────────────────────────────────
        elif path == "/api/compare":
            tickers_raw = params.get("tickers", [""])[0]
            profile = params.get("profile", ["Dip Hunter"])[0].strip()

            tickers_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            if not tickers_list:
                return self.send_error_json("Please provide at least one ticker in 'tickers'.")

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
            limit = int(params.get("limit", ["15"])[0])
            leaderboard = ENGINE.scan_universe(profile_name=profile, top_n=limit)
            return self.send_json({
                "profile": profile,
                "leaderboard": leaderboard,
                "total_in_universe": len(ENGINE.baseline_df),
                "market_regime": ENGINE.market_regime,
            })

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
