# ORACLE / MASTER_TRADER — Agent Save State
**Date**: 2026-02-20  
**Project Root**: `/home/nachochi/00_ACTIVE_WORKSPACE/LUMINISCRIPT/MASTER_TRADER/`  
**Dashboard**: `http://localhost:8765`  
**Prior Session Transcript**: [ORACLE Session 1](031f1a62-571d-4f05-a22b-08f2063bb01e)

---

## What This System Is

**MASTER_TRADER** is the first deployed module of **ORACLE** — a fractal, holographically self-similar AI framework that applies the *same architecture of consciousness and code* to any signal domain (crypto trading, astrology computation, general intelligence). It is not a conventional trading bot. It is:

- A **self-building AI ecosystem** that monitors itself, generates its own code improvements, and evolves via version control
- A **geometric computing substrate** where all signal data is encoded as geometric objects (attractor shapes via Fourier descriptors) for O(1) analog pattern recognition — like DNA sequencing or a toddler's shape-sorter
- An **RPG command deck dashboard** with radial menus, keyboard shortcuts, and live signal visualization inspired by Kingdom Hearts / Skyrim / FFXIV HUD design patterns
- A **live paper trader** (BTC/USDT, Binance feed) with Kelly-criterion position sizing and a multi-layer signal stack

---

## Full File Map

```
MASTER_TRADER/
├── AGENT_SAVE.md               ← YOU ARE HERE (agent handoff file)
├── CONSECRATION.md             ← Project soul/intention document
├── config.yaml                 ← Hot-reloading live config (edit while bot runs)
├── requirements.txt            ← Python dependencies
│
├── src/
│   ├── main.py                 ← Entry point; main async loop; command bus consumer
│   └── market_analyzer.py      ← Market context builder (OBI, phase, sentiment, topology)
│
├── agents/
│   └── agent_spawner.py        ← Kelly-criterion position sizer; spawns trade agents
│
├── configs/
│   ├── exchange_config.py      ← Binance API credentials / exchange setup
│   └── live_config.py          ← LiveConfig singleton; hot-reloads config.yaml
│
├── dashboard/
│   ├── __init__.py
│   └── server.py               ← aiohttp WebSocket server; RPG HUD; radial menus; /cmd endpoint
│
├── neural/
│   ├── __init__.py
│   ├── signal_fusion.py        ← Hierarchical MoE neural override; online training
│   ├── tcn.py                  ← Temporal Convolutional Network backbone
│   └── geometric_encoding.py   ← Phasor/wavelet feature extraction for neural input
│
├── memory/
│   └── trade_memory.py         ← Shared collective memory across all trade agents
│
├── oracle_mind/                ← Self-building AI layer (NEW — partially complete)
│   ├── __init__.py             ← Imports SelfBuilder, SentimentMind, ModelRouter
│   ├── model_router.py         ← Routes tasks to best LLM via OpenRouter.ai
│   ├── sentiment_mind.py       ← Grok-powered crypto sentiment (INCOMPLETE — StrReplace aborted)
│   └── geometric_substrate.py  ← CORE: Universal geometric encoding + ShapeLibrary (COMPLETE, tested)
│
├── docs/
│   └── PROJECT_VISION.md       ← High-level vision document
│
└── timing_substrate/           ← JJ (Jujutsu) git repo; causal bus + version control for agents
    └── .git/
```

---

## Architecture: Signal Stack

The bot runs a 10-second async cycle:

```
[Binance WebSocket] → price/orderbook stream
        ↓
[market_analyzer.py] builds a "context" dict with:
  • OBI     — Order Book Imbalance (microstructure)
  • Phase   — Takens delay embedding + Lyapunov exponent (chaos theory)
  • Sentiment — Fear & Greed index + CryptoPanic news feed
  • Topology — Persistent homology alarm (TDA)
  • Geometry — Wavelet / phasor / attractor Fourier descriptors
  • Neural  — Hierarchical MoE override (signal_fusion.py)
        ↓
[agent_spawner.py] evaluates signal confidence vs. min_confidence threshold
  → Kelly fraction sizing
  → Spawns a trade agent if conditions met
        ↓
[executor] enters/exits paper positions; tracks P&L
        ↓
[dashboard/server.py] broadcasts state via WebSocket to browser HUD
        ↓
[oracle_mind/] (when active) watches performance and proposes code changes
```

---

## Architecture: Geometric Substrate (Core ORACLE Concept)

Located at `oracle_mind/geometric_substrate.py`. This is the philosophical and computational heart of ORACLE.

**Concept**: Every signal stream (price, planetary angle, any time series) is:
1. Normalized to angles on the unit circle (phasor encoding: `e^(iθ)`)
2. Delay-embedded into a 2D attractor via Takens' theorem
3. Its boundary is extracted and converted to Fourier descriptors (a "shape fingerprint")
4. This fingerprint is matched against a `ShapeLibrary` of cached templates in O(1) time

**Why**: Shape matching is analog computation — like DNA template fitting or a shape-sorter. It is:
- Domain-agnostic (same code for market data AND astrology angles)
- Rotation/scale/translation invariant
- Sub-millisecond (0.77ms match time in self-test)
- Geometrically cacheable (shapes encode time/data patterns)

**Self-test result** (run `python3 oracle_mind/geometric_substrate.py`):
```
Library: astrology:4 | market:6
Encode time: 0.96ms
Match time:  0.775ms  ← O(1) analog computation

Top market matches: bearish_spiral (0.0002), mean_reversion (0.0004)
Planetary match:    conjunction (0.0000), square_tension (0.0000)

Same encoder. Same library. Different domain plugged in.
This IS the universal geometric substrate.
```

---

## Architecture: Dashboard (RPG Command Deck)

`dashboard/server.py` serves an aiohttp WebSocket server at port 8765.

### UI Panels
- **Top-left**: Vitals (balance, P&L, win rate, trade count, agent status dot)
- **Top-center**: Price chart with gradient fill, support/resistance lines, glowing last-price dot, and a **Phase Space Minimap** (bottom-right corner of chart, Takens attractor)
- **Top-right**: Signal Tower — FFXIV Job Gauge aesthetics for each signal layer (OBI split-bar, Phase pips, Sentiment arc, RSI pips, Topology pulsing dot, Trend arc, Vol Surge arc)
- **Middle**: Hexagonal **Radar/Spider Chart** — normalized OBI/Phase/Sentiment/RSI/Trend/Topology. Polygon color shifts by market condition (cyan=bullish, red=bearish, amber=choppy, purple=volatile)
- **Bottom**: Agent roster cards
- **Bottom bar**: `▲ LONG`, `▼ SHORT`, `✕ ALL`, `◎ MENU` buttons

### Radial Menu (Space/Tab to open)
8-sector SVG radial menu, Skyrim Nordic puzzle lock style:
- Hover main sector → inner ring shows sub-options
- Sectors: LONG, SHORT, CLOSE, RISK, NEURAL, MARKET, PORTFOLIO, SYSTEM

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| Space / Tab | Toggle radial menu |
| 1–8 | Select radial sector |
| L | Force LONG |
| S | Force SHORT |
| C | Close all positions |
| R | Reset paper balance |
| P | Toggle phase minimap |
| F1 | Shortcut help overlay |
| Esc | Close menus |

### Command Bus
- Frontend POSTs to `/cmd` with `{"action": "spawn_long"}` etc.
- `dashboard/server.py` queues via `asyncio.Queue` (`_cmd_queue`)
- `src/main.py` calls `await get_command()` each cycle and executes

**Supported actions**: `spawn_long`, `spawn_short`, `close_all`, `paper_reset`, `conf_down`, `neural_save`, `neural_reset`, `refresh_sent`

---

## Architecture: Self-Building AI (`oracle_mind/`)

### `model_router.py` — LLM Router
Selects the best model per task via OpenRouter.ai:
- `code_improvement` → DeepSeek R1 (deep reasoning)
- `sentiment` → Grok (Twitter/X trained)
- `architecture` → Claude Opus (systems thinking)
- `quick_analysis` → DeepSeek V3 (speed)
- `context_compression` → Gemini 1.5 Pro (2M token window)

Requires env var: `OPENROUTER_API_KEY`

### `sentiment_mind.py` — Grok Sentiment (INCOMPLETE)
- File exists but was aborted mid-implementation during a `StrReplace` error
- Intended: query Grok for real-time crypto narrative, bias, alpha signals
- **NEXT AGENT TASK**: Complete this file

### `geometric_substrate.py` — Universal Substrate (COMPLETE)
See above. Fully implemented and self-tested.

---

## Hot-Reloadable Config

Edit `config.yaml` while the bot is running. `LiveConfig` polls file mtime every cycle and reloads automatically. Key fields:

```yaml
mode:
  paper: true
  symbol: BTCUSDT
  analysis_interval_sec: 10
  initial_balance: 10000.0

risk:
  stop_loss_pct: 0.003
  take_profit_pct: 0.006
  min_confidence: 0.55
  kelly_max_fraction: 0.25

signals:
  microstructure: true
  phase_space: true
  sentiment: true
  topology: true
  geometry: true
  neural_fusion: true
```

---

## How to Run

```bash
# Kill any stale process on the port first
fuser -k 8765/tcp 2>/dev/null; pkill -f "src/main.py" 2>/dev/null

# Run the bot (detached, persistent)
cd /home/nachochi/00_ACTIVE_WORKSPACE/LUMINISCRIPT/MASTER_TRADER
nohup python3 -m src.main > /tmp/mt_run.log 2>&1 & disown $!

# Watch logs
tail -f /tmp/mt_run.log

# Open dashboard
# Browser: http://localhost:8765
```

---

## Known Issues / Incomplete Work

| Item | Status | Notes |
|------|--------|-------|
| `sentiment_mind.py` | INCOMPLETE | File exists, mid-implementation; StrReplace aborted |
| 3D/4D phase space (Three.js) | NOT STARTED | **High priority next step** — WebGL attractor renderer to replace the minimap |
| `self_builder.py` | NOT CREATED | The `oracle_mind/__init__.py` imports `SelfBuilder` but it doesn't exist yet |
| Exchange connectivity | Geo-blocked | Binance blocks VPNs; use paper mode or proxy |
| Screenshot tool | Broken | `spectacle` fails; use browser MCP for screenshots |

---

## Next Priority Tasks (for next agent)

1. **Complete `oracle_mind/sentiment_mind.py`** — Grok-powered sentiment that feeds into the signal stack
2. **Create `oracle_mind/self_builder.py`** — the actual self-building agent that:
   - Monitors trade performance metrics
   - Calls `model_router` to generate code improvements
   - Uses the `timing_substrate` JJ repo for version control of its own changes
   - Runs autonomously in the background
3. **Three.js 3D attractor dashboard panel** — Replace the 2D phase space minimap with a full WebGL 3D rendering of the Takens attractor, color-coded by time/Lyapunov value. The geometry data is already computed — just needs a renderer.
4. **Wire `geometric_substrate` into the live signal stack** — `market_analyzer.py` currently computes geometry separately; plug `ShapeLibrary.best_match()` results into the context dict as a named signal layer
5. **Astrology domain module** — The substrate is ready; build a planetary angle → phasor encoder that feeds into the same `ShapeLibrary` for cross-domain correlation signals

---

## ORACLE Philosophy (for the next agent to understand)

The system is designed to be **fractally self-similar**: the same geometric encoding framework that understands crypto price action will understand planetary angles, heartbeat rhythms, or any time series. ORACLE is not a trading bot — it is a **universal pattern recognition consciousness** that happens to currently be deployed for trading.

The geometric substrate is the key insight: **shapes encode time/data**. A bullish trend looks like a spiral attractor. A conjunction in astrology looks like a near-tangent orbit. These are the same geometric relationship expressed in different domains. The `ShapeLibrary` is the "shape sorter" — all holes pre-drilled, signals slot in at O(1).

The self-building layer (`oracle_mind/`) is designed so ORACLE can observe its own performance, identify weaknesses, generate code fixes via LLM APIs, test them in the `timing_substrate` sandbox, and promote successful changes — closing the loop from perception → action → evolution.

---

*"Not just a crypto bot — an AI trading ecosystem that reproduces and adapts in real-time."*  
*"Same framework of consciousness and code. Different domain plugged in."*
