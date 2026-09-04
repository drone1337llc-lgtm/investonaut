# Investonaut — Alpaca AI Trading Agents Hackathon

**A CrewAI multi-agent crypto trader on the Alpaca MCP server.**

Three LLM agents — **bull**, **bear**, and **manager** — deliberate the market
every 15 minutes, 24/7, and their verdict **directly moves per-symbol
allocations and the risk cap** behind hard risk rails. Trades an 11-symbol
crypto + equity universe around the clock.

## Run the judge demo (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs **fully offline** with a deterministic mock crew — no API keys
needed. To connect the **Live Status** page to the real Alpaca paper account,
set these environment variables before launching:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
streamlit run app.py
```

## Live mode (real Alpaca + real CrewAI crew)

The app runs **fully offline** with a deterministic mock crew — no API keys needed.
To enable **live mode** (real Alpaca paper account + real CrewAI→ollama crew), set
these secrets. On **Streamlit Cloud**: open the app → **⋮ → Settings → Secrets**
and paste the TOML below. Locally: copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in (it is gitignored).

```toml
# Alpaca paper account (Live Status page)
ALPACA_API_KEY = "paste-paper-api-key"
ALPACA_SECRET_KEY = "paste-paper-secret-key"

# ollama-cloud (real CrewAI crew on the Universe & Leans + Live Pipeline pages)
OLLAMA_CLOUD_URL = "https://ollama.com/v1"
OLLAMA_CLOUD_KEY = "paste-ollama-cloud-key"

# Optional crew model overrides (defaults match the live fleet)
# PROTONAUT_MODEL_BULL = "gpt-oss:120b"
# PROTONAUT_MODEL_BEAR = "nemotron-3-nano:30b"
# PROTONAUT_MODEL_MGR = "gemma4:31b"
```

> **Never commit real values.** `.streamlit/secrets.toml` is gitignored; only the
> `.example` template is committed. The app reads `st.secrets` first, then env vars.

## What's in the demo

| Page | What it shows |
|------|---------------|
| **Overview** | Cover + one-line pitch |
| **The Crew** | Bull / bear / manager agents and their models |
| **Universe & Leans** | 11-symbol universe with crew lean multipliers |
| **Risk Rails** | Cap factors, exposure cap, circuit breaker, cash reserve |
| **Live Status** | Offline audit log, or live Alpaca paper account |
| **Submission Deck** | Demo reel, 9-slide PDF, 6 video scenes |

## The real system

The live trading loop runs on the fleet (cudacuda) against Alpaca paper account
**PA39I1R4BNYL**, watchdog-supervised, deliberating every 15 minutes. Every
verdict, order, and rejection is logged to a SQLite audit log (`decisions.db`).

## Repo

- **Submission repo:** `drone1337llc-lgtm/riskfirst`
- **Hackathon:** Alpaca AI Trading Agents, Sep 4 2026 15:00 UTC
