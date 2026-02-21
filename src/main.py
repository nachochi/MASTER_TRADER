#!/usr/bin/env python3
"""
MASTER_TRADER — Main Loop
The heartbeat. Run this and the system is alive.

Loop:
  1. Bootstrap historical data from Kraken REST
  2. Start Kraken WebSocket streams (background)
  3. Start live dashboard server (http://localhost:8765)
  4. Every 10s: analyze → neural fusion → maybe spawn → check exits → broadcast
  5. Neural model trains on every closed trade
  6. Never stops. Learns from every cycle.

Paper mode by default. Zero real money.
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market_analyzer import MarketAnalyzer
from agents.agent_spawner import AgentSpawner, PaperExecutor
from memory.trade_memory import TradeMemory
from configs.exchange_config import CONFIG
from configs.live_config import live_cfg          # hot-reloaded config.yaml
from neural.signal_fusion import HierarchicalFusionModel
from dashboard.server import start_server, broadcast, build_state, get_command
from timing_substrate.jj_bus import JJTimingSubstrate, build_tick_signals

# Reads from config.yaml — can change while running
ANALYSIS_INTERVAL = live_cfg.analysis_interval
SYMBOL    = live_cfg.symbol
COIN_SLUG = "bitcoin"
DASHBOARD_PORT = live_cfg.dashboard_port


async def run():
    print("\n" + "=" * 60)
    print("  MASTER_TRADER — STARTING")
    print(f"  {utcnow().isoformat()} UTC")
    CONFIG.print_status()
    print("=" * 60 + "\n")

    # Core components
    memory   = TradeMemory()
    analyzer = MarketAnalyzer(symbol=SYMBOL)
    spawner  = AgentSpawner(memory=memory)
    executor = PaperExecutor(memory=memory, starting_balance=CONFIG.exchange.paper_balance)

    # Neural fusion model — loads saved weights if they exist
    neural = HierarchicalFusionModel()
    print(f"[Main] Neural model: {neural.param_count():,} parameters | "
          f"{neural.training_steps} training steps completed")

    # JJ Timing Substrate — toroidal consensus bus
    substrate = JJTimingSubstrate()
    sub_status = substrate.status()
    print(f"[Main] JJ Substrate: {len(sub_status['workspaces'])} agent workspaces | "
          f"commit={sub_status['latest_commit']}")

    # Start live dashboard
    dashboard_runner = await start_server(port=DASHBOARD_PORT)
    print(f"[Main] Dashboard: http://localhost:{DASHBOARD_PORT}")

    # Seed historical data
    await analyzer.bootstrap()

    # Start WebSocket streams in background
    ws_task = asyncio.create_task(analyzer.data_feed.start())
    print(f"[Main] Kraken WebSocket streams started for {SYMBOL}")
    await asyncio.sleep(3)  # let first data flow in

    cycle = 0
    prev_open_trades = set()

    try:
        while True:
            cycle += 1
            print(f"\n[Main] ── Cycle {cycle} ── {utcnow().strftime('%H:%M:%S')} UTC")

            # Full five-layer market analysis
            context = await analyzer.analyze_market_context(COIN_SLUG)

            # Neural fusion prediction (runs in parallel with rule-based)
            neural_cond, neural_conf, neural_size = neural.forward(context.context_data)

            # Save raw signal to memory
            signal_id = memory.save_signal(SYMBOL, context)

            # Print current state
            current_price = context.key_levels.get("current", 0)
            print(f"  Rule:         {context.condition.value} ({context.confidence:.0%})")
            print(f"  Neural:       {neural_cond} ({neural_conf:.0%}) | size_mult={neural_size:.2f}")
            print(f"  Price:        ${current_price:,.2f}  |  "
                  f"Sentiment: {context.sentiment_score:+.3f}  |  "
                  f"OBI: {'⚡ CONTRARIAN' if context.contrarian_signal else 'normal'}")
            print(f"  Phase:        {context.phase_space_regime}  |  "
                  f"Topology: {context.topology_alarm:.3f}")

            # Check exits — detect closed trades for neural training
            open_before = set(executor.open_trades.keys())
            executor.check_exits(current_price, spawner)
            open_after = set(executor.open_trades.keys())
            closed_agents = open_before - open_after

            # JJ Timing Substrate tick — all agents write, consensus computed
            tick_signals = build_tick_signals(context, spawner, neural)
            consensus    = substrate.tick(tick_signals)

            # Consensus can override spawner logic
            if consensus["alarm"] and not context.contrarian_signal:
                print(f"  [Substrate] ALARM — consensus overrides: {consensus['dominant_signal']}")
            elif consensus["strong_contrarian"]:
                print(f"  [Substrate] CONTRARIAN consensus strength={consensus['consensus_strength']:.2f}")
            else:
                print(f"  [Substrate] tick={consensus['cycle']} | "
                      f"dominant={consensus['dominant_signal']} | "
                      f"strength={consensus['consensus_strength']:.2f}")

            # Train neural model on closed trades
            for agent_id in closed_agents:
                trade_outcome = memory.get_last_closed_trade()
                if trade_outcome:
                    won = trade_outcome.get("pnl_usd", 0) > 0
                    pnl_pct = trade_outcome.get("pnl_pct", 0)
                    loss = neural.train_on_outcome(
                        context_data=context.context_data,
                        actual_condition=context.condition.value,
                        pnl_pct=pnl_pct,
                        won=won
                    )
                    print(f"  [Neural] Trained on trade outcome | "
                          f"{'WIN' if won else 'LOSS'} {pnl_pct:+.3%} | loss={loss:.4f}")

            # ── Dashboard command bus ───────────────────────────────────
            cmd = await get_command()
            if cmd:
                _a = cmd.get("action", "")
                if _a == "spawn_long":
                    _f = spawner.spawn(context, signal_id+"_CMD",
                                       account_balance=executor.balance, force=True)
                    if _f:
                        executor.execute_entry(_f, current_price)
                    print(f"  [CMD] ▲ Force LONG — size=${_f.size:.0f}" if _f else "  [CMD] ▲ LONG blocked by risk")
                elif _a == "spawn_short":
                    _f = spawner.spawn(context, signal_id+"_CMD",
                                       account_balance=executor.balance, force=True)
                    if _f:
                        executor.execute_entry(_f, current_price)
                    print(f"  [CMD] ▼ Force SHORT" if _f else "  [CMD] ▼ SHORT blocked")
                elif _a == "close_all":
                    closed = list(executor.open_trades.keys())
                    for _id in closed:
                        executor.execute_exit(_id, current_price, "dashboard")
                    print(f"  [CMD] ✕ Closed {len(closed)} position(s)")
                elif _a == "paper_reset":
                    executor.balance = live_cfg.initial_balance
                    print(f"  [CMD] Paper balance reset → ${executor.balance:,.0f}")
                elif _a == "conf_down":
                    spawner.min_confidence = max(0.05, spawner.min_confidence - 0.05)
                    print(f"  [CMD] Confidence threshold → {spawner.min_confidence:.0%}")
                elif _a == "neural_save":
                    neural._save_weights()
                    print("  [CMD] Neural weights saved")
                elif _a == "neural_reset":
                    neural.loss_history.clear()
                    neural.training_steps = 0
                    print("  [CMD] Neural model reset")
                elif _a == "refresh_sent":
                    try:
                        analyzer.sentiment_feed._fg_cache    = None
                        analyzer.sentiment_feed._news_cache  = None
                    except Exception:
                        pass
                    print("  [CMD] Sentiment cache cleared")
                elif _a:
                    print(f"  [CMD] Unknown: {_a}")

            # Try to spawn a new agent
            agent = spawner.spawn(context, signal_id, account_balance=executor.balance)
            if agent:
                executor.execute_entry(agent, current_price)

            # Broadcast to dashboard
            try:
                state = build_state(context, spawner, executor, memory, neural)
                state["live_config"] = {
                    "stop_loss_pct":    live_cfg.stop_loss_pct,
                    "take_profit_pct":  live_cfg.take_profit_pct,
                    "min_confidence":   live_cfg.min_confidence,
                    "layers":           live_cfg.layers_enabled,
                }
                await broadcast(state)
            except Exception:
                pass  # Dashboard errors never kill the bot

            # Portfolio snapshot every 30 cycles (~5 min)
            if cycle % 30 == 0:
                summary = memory.portfolio_summary()
                kelly_f = spawner._kelly_fraction()
                ns = neural.stats()
                print(f"\n  ═══ Portfolio Snapshot ═══")
                print(f"  Balance:  ${executor.balance:,.2f}")
                print(f"  PnL:      ${(summary['total_pnl'] or 0):+.2f}")
                print(f"  Win rate: {summary['win_rate']:.1%}")
                print(f"  Kelly f*: {kelly_f:.2%}")
                print(f"  Neural:   {ns['training_steps']} steps | loss={ns['recent_loss']:.4f}")
                sub_st = substrate.status()
                print(f"  JJ Bus:   cycle={sub_st['cycle']} | log={sub_st['log_entries']} entries")

            await asyncio.sleep(ANALYSIS_INTERVAL)

    except asyncio.CancelledError:
        print("\n[Main] Shutting down gracefully...")
    finally:
        analyzer.data_feed.stop()
        ws_task.cancel()
        await dashboard_runner.cleanup()
        neural._save_weights()

        summary = memory.portfolio_summary()
        ns = neural.stats()
        print("\n" + "=" * 60)
        print("  MASTER_TRADER — SESSION SUMMARY")
        print(f"  Cycles:       {cycle}")
        print(f"  Trades:       {summary['total_trades']}")
        print(f"  Win rate:     {summary['win_rate']:.1%}")
        print(f"  Total PnL:    ${(summary['total_pnl'] or 0):+.2f}")
        print(f"  Neural steps: {ns['training_steps']}")
        print(f"  Neural loss:  {ns['recent_loss']:.4f}")
        print("=" * 60)


def handle_signal(sig, frame):
    print("\n[Main] Interrupt received — stopping...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    asyncio.run(run())
