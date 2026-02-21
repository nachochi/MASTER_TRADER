#!/usr/bin/env python3
"""
MASTER_TRADER — Agent Spawner
Reads MarketContext → creates the right specialized trading agent for that exact moment.
Dynamic agent creation: no pre-built strategies, context drives everything.

Each agent is a self-contained decision unit with:
  - Custom instructions based on current market condition
  - Risk parameters tuned to current volatility
  - A specific objective (breakout ride, mean reversion, fade the crowd, etc.)
  - Memory of its own performance feeding back to collective intelligence
"""

import asyncio
import json
import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market_analyzer import MarketContext, MarketCondition
from configs.exchange_config import CONFIG, RiskConfig
from memory.trade_memory import TradeMemory


# ─────────────────────────────────────────────
# AGENT DEFINITIONS
# ─────────────────────────────────────────────

@dataclass
class TradingAgent:
    """
    A spawned trading agent. Each instance is purpose-built for one market condition.
    Lives until its trade is closed or it's stopped out.
    """
    agent_id:     str
    agent_name:   str
    condition:    MarketCondition
    strategy:     str
    direction:    str             # LONG / SHORT / WAIT
    entry_trigger: Dict           # conditions to enter
    exit_rules:   Dict            # stop loss, take profit, time limit
    size_usd:     float
    instructions: str
    spawned_at:   str = field(default_factory=lambda: datetime.utcnow().isoformat())
    active:       bool = True
    trade_id:     Optional[int] = None
    signal_id:    Optional[int] = None


# ─────────────────────────────────────────────
# STRATEGY BLUEPRINTS  (one per condition type)
# ─────────────────────────────────────────────

STRATEGY_MAP: Dict[str, Dict] = {

    MarketCondition.BULLISH_BREAKOUT.value: {
        "name": "BreakoutRider",
        "direction": "LONG",
        "instructions": (
            "Market has confirmed bullish breakout with volume surge and positive sentiment. "
            "Ride the momentum. Enter on the first pullback to breakout level. "
            "Trail stop below the breakout candle low. Take partial profit at 2x stop distance, "
            "let rest run with trailing stop."
        ),
        "stop_mult": 1.0,
        "tp_mult": 2.5,
    },

    MarketCondition.BEARISH_BREAKDOWN.value: {
        "name": "BreakdownShort",
        "direction": "SHORT",
        "instructions": (
            "Confirmed bearish breakdown with volume and negative sentiment. "
            "Short on any bounce to the breakdown level. "
            "Stop above the breakdown candle high. Target 2x the breakdown range."
        ),
        "stop_mult": 1.0,
        "tp_mult": 2.5,
    },

    MarketCondition.MEAN_REVERSION_OVERSOLD.value: {
        "name": "OversoldReversion",
        "direction": "LONG",
        "instructions": (
            "RSI below 32 with price extended below moving average. Classic mean reversion setup. "
            "Enter near support with tight stop just below. "
            "Target: return to 20 EMA. If price doesn't recover within 4 hours, exit."
        ),
        "stop_mult": 0.7,
        "tp_mult": 1.5,
    },

    MarketCondition.MEAN_REVERSION_OVERBOUGHT.value: {
        "name": "OverboughtReversion",
        "direction": "SHORT",
        "instructions": (
            "RSI above 68 with price extended above moving average. "
            "Short on the first sign of exhaustion (decreasing volume on up candles). "
            "Target: return to 20 EMA. Tight stop above recent high."
        ),
        "stop_mult": 0.7,
        "tp_mult": 1.5,
    },

    # The contrarian play — this is the second-order edge
    "contrarian_buy": {
        "name": "ContrarianFader",
        "direction": "LONG",
        "instructions": (
            "Order book shows extreme sell-side pressure — crowd is maximally short. "
            "Research shows: fade extreme OBI. The crowd is about to get squeezed. "
            "Enter long with tight stop below current support. "
            "Expect sharp snap-back. Exit quickly — this is a scalp."
        ),
        "stop_mult": 0.5,
        "tp_mult": 1.2,
    },

    "contrarian_sell": {
        "name": "ContrarianFader",
        "direction": "SHORT",
        "instructions": (
            "Order book shows extreme buy-side pressure — crowd is maximally long. "
            "Fade the extreme OBI. Enter short with tight stop above resistance. "
            "Sharp snap-back expected. Quick scalp exit."
        ),
        "stop_mult": 0.5,
        "tp_mult": 1.2,
    },

    MarketCondition.STRONG_UPTREND.value: {
        "name": "TrendFollower",
        "direction": "LONG",
        "instructions": (
            "Strong uptrend confirmed by EMA alignment and trend strength score. "
            "Buy pullbacks to 20 EMA. Trend is your friend. "
            "Hold until trend strength drops below 0.4 or RSI hits 75."
        ),
        "stop_mult": 1.2,
        "tp_mult": 3.0,
    },

    MarketCondition.STRONG_DOWNTREND.value: {
        "name": "TrendFollower",
        "direction": "SHORT",
        "instructions": (
            "Strong downtrend. Short bounces to 20 EMA. "
            "Trend is your friend. Hold until trend strength drops or RSI hits 25."
        ),
        "stop_mult": 1.2,
        "tp_mult": 3.0,
    },

    MarketCondition.PHASE_TRANSITION.value: {
        "name": "RegimeWatcher",
        "direction": "WAIT",
        "instructions": (
            "Phase space attractor transition detected — market is switching regimes. "
            "DO NOT TRADE. Watch for first confirmed direction post-transition. "
            "Next signal will clarify direction. Patience is the edge here."
        ),
        "stop_mult": 0.0,
        "tp_mult": 0.0,
    },

    MarketCondition.BUBBLE_WARNING.value: {
        "name": "BubbleGuard",
        "direction": "SHORT",
        "instructions": (
            "Topology alarm: persistence norm above 0.75 with bullish sentiment. "
            "Classic bubble topology. Take small short position against the crowd. "
            "Tight stop — timing is uncertain. This is a probabilistic hedge."
        ),
        "stop_mult": 0.5,
        "tp_mult": 4.0,
    },

    MarketCondition.CRASH_WARNING.value: {
        "name": "CrashGuard",
        "direction": "SHORT",
        "instructions": (
            "Topology crash warning. High complexity topology without bubble sentiment. "
            "Short with small size. Keep stop tight — false alarms exist. "
            "If confirmed, add size. Research shows topology norm peaks before crashes."
        ),
        "stop_mult": 0.5,
        "tp_mult": 5.0,
    },

    MarketCondition.SIDEWAYS_CONSOLIDATION.value: {
        "name": "RangeBound",
        "direction": "WAIT",
        "instructions": (
            "Consolidation phase. No clear directional edge. "
            "Wait for breakout or wait for another signal layer to activate. "
            "Do not force trades in ranging markets — that is how accounts bleed."
        ),
        "stop_mult": 0.0,
        "tp_mult": 0.0,
    },

    MarketCondition.HIGH_VOLATILITY.value: {
        "name": "VolatilityManager",
        "direction": "WAIT",
        "instructions": (
            "Abnormal volatility detected. Spread widens, slippage increases, "
            "stops get hunted. Reduce size by 50% if trading at all. "
            "Prefer waiting for volatility to normalize."
        ),
        "stop_mult": 0.0,
        "tp_mult": 0.0,
    },

    MarketCondition.NEWS_DRIVEN_SPIKE.value: {
        "name": "NewsFader",
        "direction": "WAIT",
        "instructions": (
            "News-driven price spike. Initial move is usually overextended. "
            "Wait 15 minutes for the dust to settle. "
            "Then fade the initial reaction if it was extreme."
        ),
        "stop_mult": 0.0,
        "tp_mult": 0.0,
    },
}


# ─────────────────────────────────────────────
# AGENT SPAWNER
# ─────────────────────────────────────────────

class AgentSpawner:
    """
    Reads MarketContext → selects strategy blueprint → spawns specialized agent.
    Logs spawn to memory. Routes to paper executor.
    """

    def __init__(self, memory: Optional[TradeMemory] = None):
        self.memory = memory or TradeMemory()
        self.active_agents: Dict[str, TradingAgent] = {}
        self.risk = CONFIG.risk

    def _get_blueprint(self, context: MarketContext) -> Dict:
        """
        Select the strategy blueprint for this market context.
        Contrarian OBI signal overrides standard condition when extreme.
        """
        # Second-order edge: contrarian OBI overrides everything except alarms
        topology_alarm = context.topology_alarm > 0.75
        if context.contrarian_signal and not topology_alarm:
            obi_label = context.context_data.get("obi_label", "")
            if "sell_pressure" in obi_label:
                return STRATEGY_MAP["contrarian_buy"]
            elif "buy_pressure" in obi_label:
                return STRATEGY_MAP["contrarian_sell"]

        # Check learned pattern performance — if this condition has a bad track record, wait
        best = self.memory.get_best_conditions(min_samples=10)
        condition_stats = {p["condition"]: p for p in best}
        stats = condition_stats.get(context.condition.value)
        if stats and stats.get("win_rate", 1.0) < 0.35:
            # This condition has lost money historically — override to WAIT
            return STRATEGY_MAP.get(
                MarketCondition.SIDEWAYS_CONSOLIDATION.value,
                {"name": "HistoricalLoser", "direction": "WAIT",
                 "instructions": "Pattern has negative historical win rate. Waiting.", 
                 "stop_mult": 0, "tp_mult": 0}
            )

        return STRATEGY_MAP.get(
            context.condition.value,
            STRATEGY_MAP[MarketCondition.SIDEWAYS_CONSOLIDATION.value]
        )

    def _kelly_fraction(self) -> float:
        """
        True Kelly Criterion from live performance history.

        Formula: f* = (p * b - q) / b
            p = historical win rate
            q = 1 - p (loss rate)
            b = avg_win / avg_loss (payoff ratio)

        Half-Kelly applied: f = f* / 2  (standard safety practice)
        Falls back to min fraction when not enough history.
        """
        stats = self.memory.portfolio_summary()
        n_trades = stats.get("total_trades", 0)

        if n_trades < 10:
            # Not enough history — use conservative flat fraction
            return self.risk.kelly_min_fraction

        win_rate  = stats.get("win_rate", 0.5)
        avg_win   = abs(stats.get("avg_win_usd", 1.0)) or 1.0
        avg_loss  = abs(stats.get("avg_loss_usd", 1.0)) or 1.0
        b         = avg_win / avg_loss          # payoff ratio
        p         = win_rate
        q         = 1.0 - p

        kelly_full = (p * b - q) / b

        if kelly_full <= 0:
            # Negative edge — the math says don't bet
            return 0.0

        half_kelly = kelly_full / 2.0           # half-Kelly for safety
        return float(min(half_kelly, self.risk.kelly_max_fraction))

    def _compute_size(self, context: MarketContext, account_balance: float) -> float:
        """
        Kelly-sized position that scales with the compounding account balance.

        Each win grows the balance → next bet is larger → exponential growth curve.
        Risk level and confidence modulate the Kelly fraction further.
        """
        kelly_f = self._kelly_fraction()

        if kelly_f <= 0:
            return 0.0

        risk_factor = {"HIGH": 0.4, "MEDIUM": 0.7, "LOW": 1.0}[context.risk_level]
        confidence_factor = max(0.5, context.confidence)   # floor at 0.5 so we still trade

        adjusted_f = kelly_f * risk_factor * confidence_factor
        size = account_balance * adjusted_f

        # Apply hard ceiling — never lose the account on one trade
        size = min(size, self.risk.max_position_usd)
        size = max(size, 1.0)   # always at least $1 so system is never idle
        return round(size, 2)

    def spawn(self, context: MarketContext, signal_id: int, account_balance: float = 10_000.0) -> Optional[TradingAgent]:
        """
        Main spawn function. Returns a TradingAgent or None (if WAIT condition).
        """
        # Check gate conditions
        if context.confidence < self.risk.min_confidence:
            print(f"[Spawner] Confidence {context.confidence:.2%} below threshold — no spawn")
            return None

        if context.opportunity_score < self.risk.min_opportunity:
            print(f"[Spawner] Opportunity {context.opportunity_score:.2%} below threshold — no spawn")
            return None

        if len(self.active_agents) >= self.risk.max_open_trades:
            print(f"[Spawner] Max open trades ({self.risk.max_open_trades}) reached — no spawn")
            return None

        blueprint = self._get_blueprint(context)

        if blueprint["direction"] == "WAIT":
            print(f"[Spawner] {blueprint['name']} says WAIT — {context.condition.value}")
            return None

        size = self._compute_size(context, account_balance)
        agent_id = str(uuid.uuid4())[:8]
        agent_name = f"{blueprint['name']}_{agent_id}"

        current_price = context.key_levels.get("current", 0)
        stop_pct = self.risk.stop_loss_pct * blueprint["stop_mult"]
        tp_pct   = self.risk.take_profit_pct * blueprint["tp_mult"]

        direction = blueprint["direction"]
        if direction == "LONG":
            stop_price = current_price * (1 - stop_pct)
            tp_price   = current_price * (1 + tp_pct)
        else:
            stop_price = current_price * (1 + stop_pct)
            tp_price   = current_price * (1 - tp_pct)

        agent = TradingAgent(
            agent_id=agent_id,
            agent_name=agent_name,
            condition=context.condition,
            strategy=blueprint["name"],
            direction=direction,
            entry_trigger={"price": current_price, "condition": context.condition.value},
            exit_rules={
                "stop_loss": stop_price,
                "take_profit": tp_price,
                "max_hours": 24,
            },
            size_usd=size,
            instructions=blueprint["instructions"],
            signal_id=signal_id,
        )

        # Log to memory
        agent_db_id = self.memory.log_agent_spawn(
            agent_name=agent_name,
            condition=context.condition.value,
            strategy=blueprint["name"],
            instructions=blueprint["instructions"]
        )

        self.active_agents[agent_id] = agent

        print(f"\n[Spawner] ✨ SPAWNED: {agent_name}")
        print(f"          Strategy:  {blueprint['name']}")
        print(f"          Direction: {direction}")
        print(f"          Entry:     ${current_price:,.2f}")
        print(f"          Stop:      ${stop_price:,.4f} ({stop_pct:.2%})")
        print(f"          Target:    ${tp_price:,.4f} ({tp_pct:.2%})")
        print(f"          Size:      ${size:.2f}")
        print(f"          Logic:     {blueprint['instructions'][:80]}...")

        return agent

    def deactivate(self, agent_id: str, outcome: str, pnl_usd: float):
        """Remove agent from active pool and record outcome."""
        agent = self.active_agents.pop(agent_id, None)
        if agent:
            print(f"[Spawner] Agent {agent.agent_name} deactivated | {outcome} | ${pnl_usd:+.2f}")

    def list_active(self) -> List[Dict]:
        return [
            {
                "id": a.agent_id,
                "name": a.agent_name,
                "strategy": a.strategy,
                "direction": a.direction,
                "spawned": a.spawned_at,
                "size": a.size_usd,
            }
            for a in self.active_agents.values()
        ]


# ─────────────────────────────────────────────
# PAPER EXECUTOR
# ─────────────────────────────────────────────

class PaperExecutor:
    """
    Simulates trade execution for paper trading.
    Tracks virtual portfolio. No real exchange calls.
    """

    def __init__(self, memory: TradeMemory, starting_balance: float = 10_000.0):
        self.memory  = memory
        self.balance = starting_balance
        self.open_trades: Dict[str, Dict] = {}

    def execute_entry(self, agent: TradingAgent, current_price: float) -> int:
        """Simulate order fill at current market price."""
        trade_id = self.memory.open_trade(
            signal_id=agent.signal_id or 0,
            symbol="BTCUSDT",
            direction=agent.direction,
            entry_price=current_price,
            size_usd=agent.size_usd,
            agent_name=agent.agent_name,
            entry_reason=agent.instructions[:100],
            paper=True
        )
        agent.trade_id = trade_id
        self.open_trades[agent.agent_id] = {
            "trade_id": trade_id,
            "entry": current_price,
            "stop":  agent.exit_rules["stop_loss"],
            "tp":    agent.exit_rules["take_profit"],
            "direction": agent.direction,
            "size":  agent.size_usd,
        }
        return trade_id

    def check_exits(self, current_price: float, spawner: AgentSpawner):
        """Check if any open agents should be closed."""
        to_close = []
        for agent_id, trade_info in self.open_trades.items():
            direction = trade_info["direction"]
            stop = trade_info["stop"]
            tp   = trade_info["tp"]

            hit_stop = (direction == "LONG" and current_price <= stop) or \
                       (direction == "SHORT" and current_price >= stop)
            hit_tp   = (direction == "LONG" and current_price >= tp) or \
                       (direction == "SHORT" and current_price <= tp)

            if hit_stop or hit_tp:
                reason = "TAKE_PROFIT" if hit_tp else "STOP_LOSS"
                result = self.memory.close_trade(trade_info["trade_id"], current_price, reason)
                pnl = result.get("pnl_usd", 0)
                self.balance += pnl
                spawner.deactivate(agent_id, reason, pnl)
                to_close.append(agent_id)

        for aid in to_close:
            del self.open_trades[aid]


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mem = TradeMemory()
    spawner = AgentSpawner(memory=mem)

    print("[AgentSpawner] Initialized.")
    print(f"  Active agents: {len(spawner.active_agents)}")
    print(f"  Strategy map:  {len(STRATEGY_MAP)} conditions covered")
    print(f"  Risk config:   max ${CONFIG.risk.max_position_usd:.0f} per trade")
