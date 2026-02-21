#!/usr/bin/env python3
"""
MASTER_TRADER — Trade Memory
SQLite-backed persistent memory. Every signal, decision, trade, and outcome stored.
This is what makes the system learn across sessions — the collective intelligence.

Schema:
  signals  — every MarketContext snapshot
  trades   — every paper/real trade with entry, exit, PnL
  patterns — aggregated pattern performance (what works, what doesn't)
  agents   — agent spawn log and their outcomes
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
from dataclasses import asdict
from pathlib import Path

# Memory lives next to this file
MEMORY_DB = Path(__file__).parent / "master_trader_memory.db"


class TradeMemory:
    """
    Persistent SQLite memory for MASTER_TRADER.
    Thread-safe via connection-per-call pattern.
    """

    def __init__(self, db_path: Path = MEMORY_DB):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create schema if not exists."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    condition       TEXT NOT NULL,
                    confidence      REAL,
                    risk_level      TEXT,
                    opportunity     REAL,
                    sentiment_score REAL,
                    contrarian      INTEGER,
                    phase_regime    TEXT,
                    topology_alarm  REAL,
                    volume_profile  TEXT,
                    current_price   REAL,
                    support         REAL,
                    resistance      REAL,
                    context_json    TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id       INTEGER REFERENCES signals(id),
                    agent_name      TEXT,
                    symbol          TEXT NOT NULL,
                    direction       TEXT NOT NULL,   -- LONG / SHORT
                    entry_price     REAL NOT NULL,
                    exit_price      REAL,
                    size_usd        REAL NOT NULL,
                    pnl_usd         REAL,
                    pnl_pct         REAL,
                    status          TEXT DEFAULT 'OPEN',  -- OPEN / CLOSED / STOPPED
                    entry_time      TEXT NOT NULL,
                    exit_time       TEXT,
                    entry_reason    TEXT,
                    exit_reason     TEXT,
                    paper_trade     INTEGER DEFAULT 1     -- 1 = paper, 0 = real
                );

                CREATE TABLE IF NOT EXISTS patterns (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition       TEXT NOT NULL,
                    timeframe       TEXT,
                    sample_count    INTEGER DEFAULT 0,
                    win_count       INTEGER DEFAULT 0,
                    loss_count      INTEGER DEFAULT 0,
                    avg_pnl_pct     REAL DEFAULT 0.0,
                    avg_win_pct     REAL DEFAULT 0.0,
                    avg_loss_pct    REAL DEFAULT 0.0,
                    best_pnl_pct    REAL DEFAULT 0.0,
                    worst_pnl_pct   REAL DEFAULT 0.0,
                    last_updated    TEXT,
                    UNIQUE(condition, timeframe)
                );

                CREATE TABLE IF NOT EXISTS agents (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name      TEXT NOT NULL,
                    spawn_time      TEXT NOT NULL,
                    condition       TEXT,
                    strategy        TEXT,
                    instructions    TEXT,
                    outcome         TEXT,   -- WIN / LOSS / STOPPED / RUNNING
                    pnl_usd         REAL,
                    active          INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_signals_ts     ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_cond   ON signals(condition);
                CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol  ON trades(symbol);
            """)

    # ─────────────────────────────────────────
    # SIGNALS
    # ─────────────────────────────────────────

    def save_signal(self, symbol: str, context) -> int:
        """
        Persist a MarketContext snapshot.
        Returns the row ID for linking to trades.
        """
        kl = context.key_levels
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO signals (
                    timestamp, symbol, condition, confidence, risk_level,
                    opportunity, sentiment_score, contrarian, phase_regime,
                    topology_alarm, volume_profile, current_price, support,
                    resistance, context_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                context.timestamp,
                symbol,
                context.condition.value,
                context.confidence,
                context.risk_level,
                context.opportunity_score,
                context.sentiment_score,
                int(context.contrarian_signal),
                context.phase_space_regime,
                context.topology_alarm,
                context.volume_profile,
                kl.get("current"),
                kl.get("support"),
                kl.get("resistance"),
                json.dumps(context.context_data, default=str)
            ))
            return cursor.lastrowid

    def get_recent_signals(self, symbol: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT ?
            """, (symbol, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_signal_accuracy(self, condition: str, lookahead_minutes: int = 60) -> Dict:
        """
        Check how accurate past signals of this type were.
        Compares signal price to price lookahead_minutes later.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s1.id, s1.condition, s1.current_price, s1.timestamp,
                       s2.current_price as future_price
                FROM signals s1
                JOIN signals s2 ON s2.symbol = s1.symbol
                    AND s2.timestamp > s1.timestamp
                    AND s2.timestamp <= datetime(s1.timestamp, ? || ' minutes')
                WHERE s1.condition = ?
                ORDER BY s1.id DESC
                LIMIT 100
            """, (str(lookahead_minutes), condition)).fetchall()

        if not rows:
            return {"condition": condition, "sample_count": 0}

        bullish_conditions = {
            "bullish_breakout", "strong_uptrend", "mean_reversion_oversold", "phase_transition"
        }
        bearish_conditions = {
            "bearish_breakdown", "strong_downtrend", "mean_reversion_overbought",
            "crash_warning"
        }

        correct = 0
        total = 0
        pnl_list = []

        for r in rows:
            if r["future_price"] and r["current_price"]:
                move_pct = (r["future_price"] - r["current_price"]) / r["current_price"]
                pnl_list.append(move_pct)
                if condition in bullish_conditions and move_pct > 0:
                    correct += 1
                elif condition in bearish_conditions and move_pct < 0:
                    correct += 1
                total += 1

        import numpy as np
        return {
            "condition": condition,
            "sample_count": total,
            "accuracy": correct / total if total else 0.0,
            "avg_move_pct": float(np.mean(pnl_list)) if pnl_list else 0.0,
            "lookahead_minutes": lookahead_minutes
        }

    # ─────────────────────────────────────────
    # TRADES
    # ─────────────────────────────────────────

    def open_trade(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        agent_name: str = "manual",
        entry_reason: str = "",
        paper: bool = True
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    signal_id, agent_name, symbol, direction, entry_price,
                    size_usd, status, entry_time, entry_reason, paper_trade
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_id, agent_name, symbol, direction,
                entry_price, size_usd, "OPEN",
                datetime.utcnow().isoformat(), entry_reason, int(paper)
            ))
            trade_id = cursor.lastrowid
        print(f"[TradeMemory] OPENED trade #{trade_id} | {direction} {symbol} @ ${entry_price:,.2f} | ${size_usd:.2f}")
        return trade_id

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str = ""
    ) -> Dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id=?", (trade_id,)
            ).fetchone()

            if not row:
                return {"error": "trade_not_found"}

            direction   = row["direction"]
            entry_price = row["entry_price"]
            size_usd    = row["size_usd"]

            if direction == "LONG":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl_usd = size_usd * pnl_pct

            conn.execute("""
                UPDATE trades SET
                    exit_price=?, pnl_usd=?, pnl_pct=?,
                    status='CLOSED', exit_time=?, exit_reason=?
                WHERE id=?
            """, (
                exit_price, pnl_usd, pnl_pct,
                datetime.utcnow().isoformat(), exit_reason, trade_id
            ))

        self._update_pattern(row["signal_id"], pnl_pct)
        result = {
            "trade_id": trade_id, "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct, "exit_price": exit_price
        }
        emoji = "✅" if pnl_usd >= 0 else "❌"
        print(f"[TradeMemory] {emoji} CLOSED trade #{trade_id} | PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2%})")
        return result

    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM trades WHERE status='OPEN'"
        params: tuple = ()
        if symbol:
            query += " AND symbol=?"
            params = (symbol,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────
    # PATTERNS  (collective learning)
    # ─────────────────────────────────────────

    def _update_pattern(self, signal_id: Optional[int], pnl_pct: float):
        """Update rolling stats for the pattern that generated this trade."""
        if not signal_id:
            return

        with self._connect() as conn:
            sig = conn.execute(
                "SELECT condition, timestamp FROM signals WHERE id=?", (signal_id,)
            ).fetchone()

            if not sig:
                return

            condition = sig["condition"]
            # Upsert pattern stats
            conn.execute("""
                INSERT INTO patterns (condition, timeframe, sample_count, win_count, loss_count,
                    avg_pnl_pct, best_pnl_pct, worst_pnl_pct, last_updated)
                VALUES (?, '1m', 1,
                    CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                    CASE WHEN ? < 0 THEN 1 ELSE 0 END,
                    ?, ?, ?, ?)
                ON CONFLICT(condition, timeframe) DO UPDATE SET
                    sample_count = sample_count + 1,
                    win_count    = win_count + CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                    loss_count   = loss_count + CASE WHEN ? < 0 THEN 1 ELSE 0 END,
                    avg_pnl_pct  = (avg_pnl_pct * sample_count + ?) / (sample_count + 1),
                    best_pnl_pct  = MAX(best_pnl_pct, ?),
                    worst_pnl_pct = MIN(worst_pnl_pct, ?),
                    last_updated  = ?
            """, (
                condition,
                pnl_pct, pnl_pct, pnl_pct, pnl_pct, pnl_pct, datetime.utcnow().isoformat(),
                pnl_pct, pnl_pct,
                pnl_pct, pnl_pct, pnl_pct, datetime.utcnow().isoformat()
            ))

    def get_pattern_stats(self, condition: Optional[str] = None) -> List[Dict]:
        """Get learned pattern performance — what conditions produce profits."""
        query = "SELECT * FROM patterns"
        params: tuple = ()
        if condition:
            query += " WHERE condition=?"
            params = (condition,)
        query += " ORDER BY avg_pnl_pct DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_best_conditions(self, min_samples: int = 5) -> List[Dict]:
        """Return conditions ranked by win rate — the system's learned wisdom."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT *,
                    CAST(win_count AS REAL) / NULLIF(sample_count, 0) as win_rate
                FROM patterns
                WHERE sample_count >= ?
                ORDER BY win_rate DESC, avg_pnl_pct DESC
            """, (min_samples,)).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────
    # AGENTS
    # ─────────────────────────────────────────

    def log_agent_spawn(
        self,
        agent_name: str,
        condition: str,
        strategy: str,
        instructions: str
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO agents (agent_name, spawn_time, condition, strategy, instructions, active)
                VALUES (?,?,?,?,?,1)
            """, (agent_name, datetime.utcnow().isoformat(), condition, strategy, instructions))
            return cursor.lastrowid

    def close_agent(self, agent_id: int, outcome: str, pnl_usd: float):
        with self._connect() as conn:
            conn.execute("""
                UPDATE agents SET active=0, outcome=?, pnl_usd=? WHERE id=?
            """, (outcome, pnl_usd, agent_id))

    # ─────────────────────────────────────────
    # PORTFOLIO SNAPSHOT
    # ─────────────────────────────────────────

    def get_last_closed_trade(self) -> Optional[Dict]:
        """Return the most recently closed trade — used for neural training signal."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM trades WHERE status='CLOSED'
                ORDER BY exit_time DESC LIMIT 1
            """).fetchone()
        return dict(row) if row else None

    def _get_recent_prices(self, limit: int = 100) -> list:
        """Return recent close prices from signal history for dashboard attractor viz."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT current_price FROM signals
                WHERE current_price > 0
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        prices = [r["current_price"] for r in reversed(rows)]
        return prices

    def portfolio_summary(self) -> Dict:
        """
        Current state of the paper portfolio.
        Includes avg_win_usd and avg_loss_usd for Kelly Criterion sizing.
        """
        with self._connect() as conn:
            totals = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END) as closed,
                    SUM(CASE WHEN status='OPEN'   THEN 1 ELSE 0 END) as open,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usd)  as total_pnl,
                    AVG(pnl_pct)  as avg_pnl_pct,
                    MAX(pnl_pct)  as best_trade_pct,
                    MIN(pnl_pct)  as worst_trade_pct,
                    AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END) as avg_win_usd,
                    AVG(CASE WHEN pnl_usd < 0 THEN ABS(pnl_usd) END) as avg_loss_usd
                FROM trades
                WHERE paper_trade=1 AND status='CLOSED'
            """).fetchone()

        d = dict(totals)
        closed = d.get("closed") or 1
        wins   = d.get("wins") or 0
        d["win_rate"]    = wins / closed
        d["avg_win_usd"] = d.get("avg_win_usd") or 1.0
        d["avg_loss_usd"]= d.get("avg_loss_usd") or 1.0
        return d


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    mem = TradeMemory()
    summary = mem.portfolio_summary()
    print("[TradeMemory] Database initialized.")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Open:         {summary['open']}")
    print(f"  Closed:       {summary['closed']}")
    print(f"  Win rate:     {summary['win_rate']:.1%}")
    print(f"  Total PnL:    ${(summary['total_pnl'] or 0):+.2f}")
    print(f"  DB path:      {MEMORY_DB}")
