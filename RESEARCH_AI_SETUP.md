# Research Desk AI setup

The research assistant uses public SEC company facts and recent 10-K/10-Q
filings, then sends the question and retrieved public excerpts to an
OpenAI-compatible chat-completions API. The browser never receives the API key.
Retrieved source excerpts are cached in server memory for up to 30 minutes;
answers and user questions are not persisted by the server. Analyst notes and
thesis scenarios remain in the browser's local storage.

1. Copy `.env.example` to `.env`.
2. Set `SEC_CONTACT_EMAIL` to a monitored email address, and set
   `OPENAI_API_KEY` to a key for your chosen provider. Set
   `RESEARCH_ACCESS_TOKEN` to a long, random private value that you will enter
   in the Research Desk. Optionally adjust `OPENAI_MODEL` and `OPENAI_BASE_URL`
   for an OpenAI-compatible provider.
3. Export the values in the terminal before starting the server:

   ```sh
   set -a
   . ./.env
   set +a
   python3 server.py
   ```

   The project's existing virtual environment can be used as
   `.venv/bin/python server.py`.

The SEC asks API users to identify themselves with a descriptive User-Agent;
the server uses `SEC_CONTACT_EMAIL` for that purpose. Keep `.env` private.
For deployment, configure the same values through the hosting provider's
environment-variable settings instead of uploading `.env`.
The research endpoint has per-client and global hourly limits, but these are
not user authentication. The required shared access token is a lightweight
single-user gate, not a substitute for sign-in and per-user authorization. Put
a public deployment behind an access-control layer before adding a paid
provider key.

The first version only covers companies in the SEC's current ticker map.
Financial data and filing text can be incomplete or non-standard; verify
claims against the linked filings. For questions about prices or entries, the
app also uses the existing Yahoo Finance daily-price feed to calculate
volatility bands and historical-bar touch estimates. These are not real-time
quotes, fair value, expected returns, or recommendations. This is research
support, not investment advice.
