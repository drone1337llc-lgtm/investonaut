# Investonaut — Alpaca AI Trading Agents Hackathon
# Judge-facing Streamlit demo.
#
# Modes:
#   OFFLINE (default)  — deterministic mock crew, no keys needed.
#   LIVE               — real Alpaca paper account + real CrewAI->ollama crew,
#                        enabled when the secrets below are set (Streamlit Cloud
#                        dashboard -> Secrets, or a local .streamlit/secrets.toml).
#
# SECRETS (never commit real values; see .streamlit/secrets.toml.example):
#   ALPACA_API_KEY     Alpaca paper API key
#   ALPACA_SECRET_KEY  Alpaca paper secret key
#   OLLAMA_CLOUD_URL   ollama-cloud base URL (default https://ollama.com/v1)
#   OLLAMA_CLOUD_KEY   ollama-cloud API key
#   PROTONAUT_MODEL_BULL / _BEAR / _MGR   crew model ids (optional overrides)

import os
import json
import random
import time
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(
    page_title="Investonaut — Alpaca AI Trading Agents",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants (mirror the live config on cudacuda)
# ---------------------------------------------------------------------------
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "HYPE", "NVDA", "AMD", "OKTA", "SKHY", "SNDK", "MU"]
CREW_LEAN_MIN, CREW_LEAN_MAX = 0.7, 1.3
CAP_BULL, CAP_BEAR, CAP_NEUTRAL = 1.10, 0.85, 1.0
EXPOSURE_CAP = 0.95
CASH_RESERVE = 0.10
DEFAULT_MODELS = {
    "bull": "gpt-oss:120b",
    "bear": "nemotron-3-nano:30b",
    "manager": "gemma4:31b",
}

# Deterministic seed so the offline demo is reproducible for judges.
random.seed(20260904)


# ---------------------------------------------------------------------------
# Secret access — Streamlit secrets first, then env vars. Never printed.
# ---------------------------------------------------------------------------
def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _models() -> dict:
    return {
        "bull": _get_secret("PROTONAUT_MODEL_BULL", DEFAULT_MODELS["bull"]),
        "bear": _get_secret("PROTONAUT_MODEL_BEAR", DEFAULT_MODELS["bear"]),
        "manager": _get_secret("PROTONAUT_MODEL_MGR", DEFAULT_MODELS["manager"]),
    }


def _live_available() -> bool:
    return bool(_get_secret("ALPACA_API_KEY") and _get_secret("ALPACA_SECRET_KEY"))


def _crew_available() -> bool:
    return bool(_get_secret("OLLAMA_CLOUD_KEY"))


# ---------------------------------------------------------------------------
# Mock market + crew (offline, deterministic)
# ---------------------------------------------------------------------------
def _mock_prices():
    """Deterministic pseudo-prices so the demo looks alive but is stable."""
    base = {
        "BTC": 64000, "ETH": 3200, "SOL": 145, "XRP": 0.62, "HYPE": 0.28,
        "NVDA": 128, "AMD": 165, "OKTA": 88, "SKHY": 12.4, "SNDK": 76, "MU": 118,
    }
    drift = {s: random.uniform(-0.02, 0.02) for s in UNIVERSE}
    return {s: round(base[s] * (1 + drift[s]), 4) for s in UNIVERSE}


def _mock_verdict(prices):
    """Simulate the 3-agent crew debate -> a single verdict.

    Mirrors crewai_advisor.py: bull leans into strength, bear trims
    overextension, manager synthesizes. Deterministic given the price set.
    """
    ranked = sorted(prices.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    bull_over = {s for s, _ in ranked[: n // 3]}
    bear_under = {s for s, _ in ranked[-n // 3:]}
    leans = {}
    for s in UNIVERSE:
        if s in bull_over and s not in bear_under:
            leans[s] = "overweight"
        elif s in bear_under and s not in bull_over:
            leans[s] = "underweight"
        else:
            leans[s] = "neutral"
    bull_cnt = sum(1 for v in leans.values() if v == "overweight")
    bear_cnt = sum(1 for v in leans.values() if v == "underweight")
    stance = "bull" if bull_cnt > bear_cnt else ("bear" if bear_cnt > bull_cnt else "neutral")
    cap_factor = {"bull": CAP_BULL, "bear": CAP_BEAR}.get(stance, CAP_NEUTRAL)
    mults = {
        s: (CREW_LEAN_MAX if v == "overweight" else CREW_LEAN_MIN if v == "underweight" else 1.0)
        for s, v in leans.items()
    }
    return {"stance": stance, "cap_factor": cap_factor, "leans": leans, "multipliers": mults}


def _mock_audit_log(verdict, prices):
    """Build a small, honest audit-log table for the demo."""
    rows = []
    for s in UNIVERSE:
        rows.append({
            "symbol": s,
            "price": prices[s],
            "lean": verdict["leans"][s],
            "multiplier": verdict["multipliers"][s],
            "target_alloc": round(0.09 * verdict["multipliers"][s], 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Live CrewAI -> ollama crew (real, mirrors crewai_advisor.py on cudacuda)
# ---------------------------------------------------------------------------
def _live_crew_verdict(prices: dict) -> dict:
    """Run the real 3-agent crew against ollama-cloud. Returns a verdict dict.

    Falls back to the mock on any error so the demo never breaks.
    """
    try:
        from crewai import LLM, Agent, Task, Crew, Process

        base = _get_secret("OLLAMA_CLOUD_URL", "https://ollama.com/v1")
        key = _get_secret("OLLAMA_CLOUD_KEY")
        models = _models()

        def _llm(model: str):
            return LLM(
                model=f"openai/{model}",
                base_url=base,
                api_key=key,
                temperature=0.3,
                max_tokens=2000,
            )

        snap_lines = ["MARKET SNAPSHOT:"]
        for sym, px in sorted(prices.items()):
            snap_lines.append(f"  {sym}: price={px:.4f}")
        snap = "\n".join(snap_lines)

        prompt = (
            "You are a market analyst for an automated trading bot. Given the "
            "snapshot below, decide for EACH symbol whether to overweight, hold "
            "neutral, or underweight it, and give an overall market stance "
            "(bull/neutral/bear). Respond ONLY as JSON:\n"
            '{"stance":"bull|neutral|bear","leans":{"SYM":"overweight|neutral|underweight",...}}\n'
            "Be decisive and specific. Snapshot:\n" + snap
        )

        agent_bull = Agent(
            role="Momentum Analyst",
            goal="Find the strongest momentum names to overweight.",
            backstory="Aggressive momentum trader; leans into strength.",
            llm=_llm(models["bull"]), verbose=False, allow_delegation=False,
        )
        agent_bear = Agent(
            role="Risk Analyst",
            goal="Flag overextended names to underweight and protect capital.",
            backstory="Defensive risk manager; trims into strength, buys weakness.",
            llm=_llm(models["bear"]), verbose=False, allow_delegation=False,
        )
        manager = Agent(
            role="Portfolio Manager",
            goal="Synthesize the two analysts into one decisive verdict.",
            backstory="Senior PM who balances momentum and risk.",
            llm=_llm(models["manager"]), verbose=False, allow_delegation=False,
        )

        t_bull = Task(description=prompt, expected_output="JSON verdict", agent=agent_bull)
        t_bear = Task(description=prompt, expected_output="JSON verdict", agent=agent_bear)
        t_mgr = Task(
            description=("Given the two analyst verdicts, produce the final JSON "
                         "verdict with stance and per-symbol leans."),
            expected_output="JSON verdict", agent=manager,
        )

        crew = Crew(
            agents=[agent_bull, agent_bear, manager],
            tasks=[t_bull, t_bear, t_mgr],
            process=Process.sequential,
            verbose=False,
        )
        raw = str(crew.kickoff())

        # parse JSON from the manager's output
        t = raw.strip()
        if t.startswith("```"):
            t = t.split("```")[1] if "```" in t[3:] else t
            t = t.lstrip("json").strip()
        verdict = {}
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            verdict = json.loads(t[start:end + 1])

        stance = str(verdict.get("stance", "neutral")).lower()
        cap_factor = {"bull": CAP_BULL, "bear": CAP_BEAR}.get(stance, CAP_NEUTRAL)
        leans = verdict.get("leans", {}) if isinstance(verdict.get("leans"), dict) else {}
        mults = {}
        for s in UNIVERSE:
            lean = str(leans.get(s, "neutral")).lower()
            if lean in ("overweight", "over", "buy", "long"):
                mults[s] = CREW_LEAN_MAX
            elif lean in ("underweight", "under", "sell", "short"):
                mults[s] = CREW_LEAN_MIN
            else:
                mults[s] = 1.0
        return {"stance": stance, "cap_factor": cap_factor, "leans": leans,
                "multipliers": mults, "live": True}
    except Exception as e:
        # fall back to mock, but flag it
        v = _mock_verdict(prices)
        v["live"] = False
        v["error"] = str(e)[:200]
        return v


def _get_verdict(prices: dict) -> dict:
    """Return the live crew verdict if configured, else the mock."""
    if _crew_available():
        return _live_crew_verdict(prices)
    return _mock_verdict(prices)


# ---------------------------------------------------------------------------
# Live Alpaca (optional) — only used if keys are present
# ---------------------------------------------------------------------------
def _live_account():
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            _get_secret("ALPACA_API_KEY"),
            _get_secret("ALPACA_SECRET_KEY"),
            paper=True,
        )
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": acct.status,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _css():
    st.markdown("""
    <style>
      .block-container { padding-top: 1.5rem; }
      .crew-card {
        background: #121826; border: 1px solid #1f2a3d; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: .8rem;
      }
      .crew-card h4 { margin: 0 0 .3rem 0; }
      .pill {
        display: inline-block; border-radius: 999px; padding: .15rem .7rem;
        font-size: .78rem; font-weight: 600; margin-right: .3rem;
      }
      .pill-bull { background: rgba(61,220,151,.12); color: #3ddc97; border: 1px solid rgba(61,220,151,.35); }
      .pill-bear { background: rgba(255,107,107,.12); color: #ff6b6b; border: 1px solid rgba(255,107,107,.35); }
      .pill-neutral { background: rgba(245,192,76,.12); color: #f5c04c; border: 1px solid rgba(245,192,76,.35); }
      .big-num { font-size: 2.2rem; font-weight: 700; }
      .lbl { color: #8b97ab; font-size: .8rem; }
    </style>
    """, unsafe_allow_html=True)


def _pill(lean):
    cls = {"overweight": "pill-bull", "underweight": "pill-bear"}.get(lean, "pill-neutral")
    return f'<span class="pill {cls}">{lean}</span>'


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview():
    st.image("assets/cover.png", use_container_width=True)
    st.markdown("""
    **Investonaut** is a CrewAI multi-agent crypto trader on the Alpaca MCP server.
    Three LLM agents — **bull**, **bear**, and **manager** — deliberate the market
    every 15 minutes, 24/7, and their verdict **directly moves per-symbol
    allocations and the risk cap** behind hard risk rails.

    Use the sidebar to explore the crew, the universe, the risk framework, the
    live status, and the full submission deck.
    """)


def page_crew():
    st.header("The Crew — Three Minds, One Book")
    st.caption("Each agent independently reads the market snapshot and returns a per-symbol lean + a risk stance. The manager synthesizes a single verdict that actually trades.")
    models = _models()
    for role, model, color, desc in [
        ("Bull", models["bull"], "green", "Proposes overweights on momentum leaders, raises the risk cap when it reads strength."),
        ("Bear", models["bear"], "red", "Flags overextended names to underweight, lowers the cap when it reads fragility."),
        ("Manager", models["manager"], "amber", "Synthesizes the debate into the verdict — the allocation multipliers and cap factor that trade."),
    ]:
        st.markdown(
            f'<div class="crew-card"><h4 style="color:{color}">{role} · <code>{model}</code></h4>'
            f'<p style="color:#8b97ab">{desc}</p></div>',
            unsafe_allow_html=True,
        )
    st.info("**Not advisory-only.** The crew verdict moves allocations within a 0.7–1.3x band and adjusts the risk cap — it has real influence on the book.")


def page_universe():
    st.header("Universe & Crew Leans")
    st.caption("11-symbol crypto + equity universe. The crew leans on all of them — crypto trades 24/7, equities during market hours.")
    prices = _mock_prices()
    verdict = _get_verdict(prices)
    rows = _mock_audit_log(verdict, prices)
    mode = "LIVE crew" if verdict.get("live") else "offline mock"
    st.markdown(f"**Crew stance:** `{verdict['stance']}` · **cap factor:** `{verdict['cap_factor']:.2f}` · *({mode})*")
    if verdict.get("error"):
        st.caption(f"Live crew unavailable, fell back to mock: {verdict['error']}")
    for r in rows:
        mult = r["multiplier"]
        color = "#3ddc97" if mult > 1 else ("#ff6b6b" if mult < 1 else "#f5c04c")
        st.markdown(
            f"**{r['symbol']}**  ${r['price']:,.2f}  {_pill(r['lean'])}  "
            f"<span style='color:{color};font-weight:700'>{mult:.2f}x</span>  "
            f"target {r['target_alloc']*100:.1f}%",
            unsafe_allow_html=True,
        )
        st.progress(min(mult / CREW_LEAN_MAX, 1.0))


def page_risk():
    st.header("Risk Rails — Enforced Pre-Broker")
    st.caption("These fire before any order ships. They are policy, not suggestion.")
    cols = st.columns(4)
    for col, (num, lbl) in zip(cols, [
        ("1.10", "cap factor · bull"),
        ("0.85", "cap factor · bear"),
        ("95%", "exposure cap"),
        ("0", "naked shorts"),
    ]):
        col.markdown(f'<div class="big-num">{num}</div><div class="lbl">{lbl}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    - **Exposure scaling** — when total exposure exceeds the cap, de-leveraging orders scale it back (confirmed live).
    - **Circuit breaker** — trips → the book goes flat until regime confirmation.
    - **Cash reserve** — 10% buffer held back.
    - **Audit log** — every verdict, order, and rejection logged to SQLite (`decisions.db`). A P&L narrative, not a black box.
    """)


def page_live():
    st.header("Live Status")
    if _live_available():
        acct = _live_account()
        if "error" in acct:
            st.error(f"Could not reach Alpaca: {acct['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity", f"${acct['equity']:,.2f}")
            c2.metric("Cash", f"${acct['cash']:,.2f}")
            c3.metric("Buying Power", f"${acct['buying_power']:,.2f}")
            st.success(f"Connected to Alpaca paper account (status: {acct['status']}).")
    else:
        st.warning("No Alpaca keys set — showing the deterministic offline demo. "
                   "Set `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in Streamlit secrets to connect to the live paper account.")
        prices = _mock_prices()
        verdict = _get_verdict(prices)
        st.markdown(f"**Crew verdict:** stance `{verdict['stance']}`, cap factor `{verdict['cap_factor']:.2f}`")
        st.markdown("**Audit log (offline demo):**")
        st.dataframe(_mock_audit_log(verdict, prices), use_container_width=True)


def page_pipeline():
    """Realtime visual of the full decision pipeline — market -> crew -> risk -> orders.

    Each stage renders the ACTUAL simulated data for that stage (mock prices,
    bull/bear leans, manager verdict, risk-rail math, allocation deltas, orders)
    so it reads as a live simulation, not a static slideshow.
    """
    st.header("Live Pipeline — Watch It Think")
    st.caption("Every 15 minutes the crew runs this exact sequence. Hit **Run a cycle** to step through it once, or toggle **Auto-advance** to loop.")

    # ---- controls ----
    c1, c2, c3 = st.columns([1, 1, 2])
    run = c1.button("▶ Run a cycle", type="primary")
    auto = c2.toggle("Auto-advance", value=False)
    speed = c3.slider("Step speed (s)", 0.3, 3.0, 1.0, key="pipe_speed")

    # ---- state: one cycle's worth of data ----
    if "pipe_cycle" not in st.session_state:
        prices = _mock_prices()
        verdict = _get_verdict(prices)
        rows = _mock_audit_log(verdict, prices)
        st.session_state["pipe_cycle"] = {
            "prices": prices, "verdict": verdict, "rows": rows,
            "idx": 0, "running": False,
        }

    cyc = st.session_state["pipe_cycle"]
    prices = cyc["prices"]
    verdict = cyc["verdict"]
    rows = cyc["rows"]

    # ---- start / stop logic ----
    if run:
        cyc["idx"] = 0
        cyc["running"] = True
    if auto:
        cyc["running"] = True
    elif not run:
        cyc["running"] = False

    animate = auto or cyc["running"]

    # ---- per-stage simulated content ----
    def _stage_market():
        st.markdown("### 📡 Market snapshot")
        st.caption(f"{len(UNIVERSE)} symbols read — crypto + equities. Prices are the live mock feed.")
        cols = st.columns(4)
        for i, s in enumerate(UNIVERSE):
            with cols[i % 4]:
                st.metric(s, f"${prices[s]:,.4f}")

    def _stage_bull():
        st.markdown("### 🐂 Bull reads strength")
        st.caption(f"Momentum Analyst ({_models()['bull']}) proposes overweights on the strongest names.")
        ranked = sorted(prices.items(), key=lambda kv: kv[1], reverse=True)
        n = len(ranked)
        over = [s for s, _ in ranked[: n // 3]]
        st.markdown("**Proposes overweight:**")
        for s in over:
            st.markdown(f"- **{s}** — ${prices[s]:,.4f} (top momentum)")
        st.markdown("**Holds neutral:**")
        st.markdown(", ".join(s for s in UNIVERSE if s not in over))

    def _stage_bear():
        st.markdown("### 🐻 Bear flags risk")
        st.caption(f"Risk Analyst ({_models()['bear']}) trims overextension and protects capital.")
        ranked = sorted(prices.items(), key=lambda kv: kv[1])
        n = len(ranked)
        under = [s for s, _ in ranked[: n // 3]]
        st.markdown("**Flags underweight:**")
        for s in under:
            st.markdown(f"- **{s}** — ${prices[s]:,.4f} (weakest momentum)")
        st.markdown("**Holds neutral:**")
        st.markdown(", ".join(s for s in UNIVERSE if s not in under))

    def _stage_manager():
        st.markdown("### 🧠 Manager synthesizes")
        st.caption(f"PM ({_models()['manager']}) balances the debate into one verdict.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Stance", verdict["stance"].upper())
        m2.metric("Cap factor", f"{verdict['cap_factor']:.2f}")
        m3.metric("Mode", "LIVE crew" if verdict.get("live") else "offline mock")
        st.markdown("**Per-symbol lean:**")
        for r in rows:
            st.markdown(f"- **{r['symbol']}**  {_pill(r['lean'])}", unsafe_allow_html=True)

    def _stage_risk():
        st.markdown("### 🛡️ Risk rails check")
        st.caption("Policy, not suggestion — fires before any order ships.")
        total_exposure = sum(r["target_alloc"] for r in rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total exposure", f"{total_exposure:.0%}")
        c2.metric("Exposure cap", f"{EXPOSURE_CAP:.0%}")
        c3.metric("Cash reserve", f"{CASH_RESERVE:.0%}")
        c4.metric("Naked shorts", "0")
        ok = total_exposure <= EXPOSURE_CAP
        if ok:
            st.success(f"Exposure {total_exposure:.0%} ≤ cap {EXPOSURE_CAP:.0%} — within rails. Orders proceed.")
        else:
            st.error(f"Exposure {total_exposure:.0%} > cap {EXPOSURE_CAP:.0%} — de-leveraging orders would scale it back.")

    def _stage_alloc():
        st.markdown("### ⚖️ Allocations move")
        st.caption("Per-symbol multipliers 0.7–1.3x applied to the book.")
        for r in rows:
            mult = r["multiplier"]
            color = "#3ddc97" if mult > 1 else ("#ff6b6b" if mult < 1 else "#f5c04c")
            st.markdown(
                f"**{r['symbol']}**  {_pill(r['lean'])}  "
                f"<span style='color:{color};font-weight:700'>{mult:.2f}x</span>  "
                f"target {r['target_alloc']*100:.1f}%", unsafe_allow_html=True)
            st.progress(min(mult / CREW_LEAN_MAX, 1.0))

    def _stage_orders():
        st.markdown("### 🚀 Orders to Alpaca MCP")
        st.caption("Paper account PA39I1R4BNYL — every step audited to SQLite.")
        st.markdown("**Order batch (simulated):**")
        for r in rows:
            if r["multiplier"] > 1:
                action = "BUY"
            elif r["multiplier"] < 1:
                action = "SELL"
            else:
                action = "HOLD"
            st.markdown(f"- `{action}` **{r['symbol']}** — target {r['target_alloc']*100:.1f}% of book")
        st.markdown("**Audit log:** every verdict, order, and rejection logged to `decisions.db`.")

    stages = [
        ("1 · Market snapshot", "#4da3ff", "📡", _stage_market),
        ("2 · Bull reads strength", "#3ddc97", "🐂", _stage_bull),
        ("3 · Bear flags risk", "#ff6b6b", "🐻", _stage_bear),
        ("4 · Manager synthesizes", "#f5c04c", "🧠", _stage_manager),
        ("5 · Risk rails check", "#ff6b6b", "🛡️", _stage_risk),
        ("6 · Allocations move", "#4da3ff", "⚖️", _stage_alloc),
        ("7 · Orders to Alpaca MCP", "#3ddc97", "🚀", _stage_orders),
    ]

    # ---- render (fragment re-runs every `speed` while animating) ----
    @st.fragment(run_every=speed if animate else None)
    def _render():
        if cyc["running"]:
            if cyc["idx"] < len(stages) - 1:
                cyc["idx"] += 1
            elif auto:
                cyc["idx"] = 0  # auto-advance loops
            else:
                cyc["running"] = False  # Run a cycle stops at the end

        # pipeline diagram (always shows all 7 stages, highlights current)
        st.markdown("### Decision pipeline")
        cols = st.columns(len(stages))
        for i, (title, color, icon, _) in enumerate(stages):
            active = (i == cyc["idx"])
            done = (i < cyc["idx"])
            with cols[i]:
                border = color if active else ("#1f2a3d" if not done else "#2a3a55")
                bg = color if active else "#121826"
                st.markdown(
                    f'<div style="border:2px solid {border};background:{bg};border-radius:10px;'
                    f'padding:.6rem .3rem;text-align:center;min-height:86px">'
                    f'<div style="font-size:1.4rem">{icon}</div>'
                    f'<div style="font-size:.72rem;font-weight:600;color:{"#0b0e14" if active else "#e8edf5"}">{title.split("·")[1].strip()}</div>'
                    f'</div>', unsafe_allow_html=True)

        # current stage detail — the actual simulated data
        st.markdown("---")
        title, color, icon, render_fn = stages[cyc["idx"]]
        render_fn()

    _render()


def page_deck():
    st.header("Submission Deck")
    tab_video, tab_slides, tab_scenes = st.tabs(["Demo Reel", "Slides (PDF)", "Video Scenes"])
    with tab_video:
        st.video("assets/demo-reel.mp4")
    with tab_slides:
        st.markdown("Download the 9-slide deck:")
        with open("assets/slides.pdf", "rb") as f:
            st.download_button("Download slides.pdf", f, file_name="investonaut-slides.pdf", mime="application/pdf")
    with tab_scenes:
        st.caption("6 scene images for the Artlist.ai video render.")
        cols = st.columns(3)
        for i, scene in enumerate(sorted(os.listdir("scenes"))):
            with cols[i % 3]:
                st.image(f"scenes/{scene}", use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------
def main():
    _css()
    st.sidebar.image("assets/cover.png", use_container_width=True)
    st.sidebar.markdown("## Investonaut")
    st.sidebar.caption("Alpaca AI Trading Agents · Sep 4, 2026")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Live Pipeline", "The Crew", "Universe & Leans", "Risk Rails", "Live Status", "Submission Deck"],
    )
    st.sidebar.markdown("---")
    live = _live_available()
    crew = _crew_available()
    mode = "LIVE" if (live or crew) else "OFFLINE (deterministic)"
    st.sidebar.caption(f"Alpaca: {'LIVE' if live else 'offline'} · Crew: {'LIVE' if crew else 'mock'}")
    st.sidebar.caption(f"Mode: {mode}")
    st.sidebar.caption(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    pages = {
        "Overview": page_overview,
        "Live Pipeline": page_pipeline,
        "The Crew": page_crew,
        "Universe & Leans": page_universe,
        "Risk Rails": page_risk,
        "Live Status": page_live,
        "Submission Deck": page_deck,
    }
    pages[page]()


if __name__ == "__main__":
    main()
