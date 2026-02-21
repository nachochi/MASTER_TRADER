#!/usr/bin/env python3
"""
MASTER_TRADER — Exchange Config
Paper trading by default. Zero real money until validated.
Secrets loaded from environment — never hardcoded.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Load secrets from the ORACLE nexus secrets file
SECRETS_FILE = Path.home() / ".config" / "ai" / "nexus" / "secrets.env"

def _load_secrets():
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

_load_secrets()


@dataclass
class RiskConfig:
    """
    Scalping-optimized risk parameters.

    Philosophy: tight stops, fast exits, high frequency, compound every win.
    Kelly Criterion drives position sizing — size scales with account balance.
    Half-Kelly applied for safety (never bet full Kelly in practice).

    Compounding math:
        Day 1:  $10,000 × 0.3% avg gain × 20 trades = $600 → $10,600
        Day 30: exponential curve kicks in as balance grows
    """
    # Scalp exit parameters — tight and fast
    stop_loss_pct:       float = 0.003    # 0.3% stop (scalp-tight)
    take_profit_pct:     float = 0.006    # 0.6% TP → 2:1 R:R maintained

    # Kelly-based sizing constraints
    kelly_max_fraction:  float = 0.25     # never bet more than 25% of balance (half-Kelly cap)
    kelly_min_fraction:  float = 0.01     # floor: always bet at least 1% of balance
    max_portfolio_risk:  float = 0.02     # 2% total portfolio at risk at once
    max_drawdown_pct:    float = 0.08     # halt if portfolio drops 8%

    # Legacy flat cap — used as hard ceiling only
    max_position_usd:    float = 500.0    # hard cap per trade regardless of Kelly

    # Throughput
    max_open_trades:     int   = 5        # more concurrent scalp positions
    min_confidence:      float = 0.55     # slightly lower threshold = more trades
    min_opportunity:     float = 0.35     # lower bar = more signals captured


@dataclass
class ExchangeConfig:
    """
    Exchange connection config.
    paper=True  → simulate trades locally, NO exchange calls
    paper=False → live trades (only after 30-day validation)
    """
    name:          str   = "binance"
    paper:         bool  = True           # ALWAYS start paper
    testnet:       bool  = True           # Binance testnet when paper=False

    # API keys — from environment only
    api_key:       str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret:    str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet_key:   str = field(default_factory=lambda: os.getenv("BINANCE_TESTNET_API_KEY", ""))
    testnet_secret:str = field(default_factory=lambda: os.getenv("BINANCE_TESTNET_SECRET", ""))

    # WebSocket
    ws_base:       str = "wss://stream.binance.com:9443/ws"
    rest_base:     str = "https://api.binance.com"
    testnet_ws:    str = "wss://testnet.binance.vision/ws"
    testnet_rest:  str = "https://testnet.binance.vision"

    # Targets
    symbols:       List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    base_currency: str = "USDT"

    # Paper portfolio
    paper_balance: float = 10_000.0      # starting paper balance

    def effective_ws(self) -> str:
        return self.testnet_ws if (not self.paper and self.testnet) else self.ws_base

    def effective_rest(self) -> str:
        return self.testnet_rest if (not self.paper and self.testnet) else self.rest_base

    def effective_key(self) -> str:
        return self.testnet_key if self.testnet else self.api_key

    def effective_secret(self) -> str:
        return self.testnet_secret if self.testnet else self.api_secret

    def status(self) -> str:
        if self.paper:
            return "PAPER TRADING (simulated, no real money)"
        if self.testnet:
            return "TESTNET (real API, fake money)"
        return "⚠️  LIVE TRADING — REAL MONEY"


@dataclass
class SentimentConfig:
    santiment_key:  str = field(default_factory=lambda: os.getenv("SANTIMENT_API_KEY", ""))
    cryptopanic_key:str = field(default_factory=lambda: os.getenv("CRYPTOPANIC_API_KEY", ""))
    lead_hours:     int = 24    # confirmed sentiment lead time


@dataclass
class MasterConfig:
    exchange:   ExchangeConfig  = field(default_factory=ExchangeConfig)
    risk:       RiskConfig      = field(default_factory=RiskConfig)
    sentiment:  SentimentConfig = field(default_factory=SentimentConfig)
    log_dir:    Path = Path(__file__).parent.parent / "logs"
    memory_dir: Path = Path(__file__).parent.parent / "memory"

    def __post_init__(self):
        self.log_dir.mkdir(exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)

    def validate(self) -> List[str]:
        """Return list of warnings about missing config."""
        warnings = []
        if not self.exchange.paper and not self.exchange.api_key:
            warnings.append("BINANCE_API_KEY not set — required for live trading")
        if not self.sentiment.santiment_key:
            warnings.append("SANTIMENT_API_KEY not set — sentiment layer will use fallback")
        return warnings

    def print_status(self):
        print("=" * 50)
        print("  MASTER_TRADER Configuration")
        print("=" * 50)
        print(f"  Mode:         {self.exchange.status()}")
        print(f"  Symbols:      {', '.join(self.exchange.symbols)}")
        print(f"  Paper bal:    ${self.exchange.paper_balance:,.2f}")
        print(f"  Max pos:      ${self.risk.max_position_usd:.0f}")
        print(f"  Stop loss:    {self.risk.stop_loss_pct:.1%}")
        print(f"  Take profit:  {self.risk.take_profit_pct:.1%}")
        print(f"  Min conf:     {self.risk.min_confidence:.0%}")
        print(f"  Log dir:      {self.log_dir}")
        warnings = self.validate()
        if warnings:
            print("\n  ⚠️  Warnings:")
            for w in warnings:
                print(f"     - {w}")
        print("=" * 50)


# Singleton config instance
CONFIG = MasterConfig()


if __name__ == "__main__":
    CONFIG.print_status()
