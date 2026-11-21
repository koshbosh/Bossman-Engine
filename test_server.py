import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import server


class ServerApiTests(unittest.TestCase):
    def test_scan_endpoint_returns_json_status(self):
        scan_state = {
            "status": "idle",
            "message": "No live market scan has been run yet.",
            "leaderboard": [],
        }
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.EngineRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        try:
            with patch.object(server.MARKET_SCANNER, "get_scan", return_value=scan_state):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", httpd.server_address[1], timeout=5
                )
                connection.request("GET", "/api/scan?profile=Dip%20Hunter")
                response = connection.getresponse()
                body = response.read()
                connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(body)["status"], "idle")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
