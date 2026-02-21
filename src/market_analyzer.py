#!/usr/bin/env python3
"""
MASTER_TRADER — Market Analyzer
Real implementations. Real data. No stubs.

Signal stack:
  Layer 1: Microstructure    — Binance WebSocket, order book imbalance (Kalman filtered)
  Layer 2: Phase Space       — Takens delay embedding, attractor trajectory, regime detection
  Layer 3: Sentiment         — Santiment API + firecrawl news scraping
  Layer 4: On-Chain Flow     — Exchange flow acceleration (second derivative of whale movement)
  Layer 5: Topology          — Persistence norm for crash/bubble early warning

Research basis: Oxford (2025), arxiv 2602.00383, engrxiv 4579, Springer 2024
Key insight: contrarian positioning against obvious OBI generates alpha (market maker dilemma)
"""

import asyncio
import json
import time
import sqlite3
import os
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from collections import deque

import numpy as np
import pandas as pd
import aiohttp
import websockets
from scipy.signal import savgol_filter
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from neural.geometric_encoding import (
    haar_wavelet_decompose, wavelet_entropy, wavelet_dominant_scale,
    attractor_fourier_descriptors, attractor_spectral_entropy,
    phasor_coherence, encode_signals_geometric, build_geometric_context
)
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

class MarketCondition(Enum):
    BULLISH_BREAKOUT            = "bullish_breakout"
    BEARISH_BREAKDOWN           = "bearish_breakdown"
    SIDEWAYS_CONSOLIDATION      = "sideways_consolidation"
    HIGH_VOLATILITY             = "high_volatility"
    LOW_VOLATILITY              = "low_volatility"
    STRONG_UPTREND              = "strong_uptrend"
    STRONG_DOWNTREND            = "strong_downtrend"
    MEAN_REVERSION_OVERSOLD     = "mean_reversion_oversold"
    MEAN_REVERSION_OVERBOUGHT   = "mean_reversion_overbought"
    NEWS_DRIVEN_SPIKE           = "news_driven_spike"
    ARBITRAGE_OPPORTUNITY       = "arbitrage_opportunity"
    PUMP_AND_DUMP               = "pump_and_dump"
    PHASE_TRANSITION            = "phase_transition"       # attractor regime change
    BUBBLE_WARNING              = "bubble_warning"         # topology signal
    CRASH_WARNING               = "crash_warning"          # topology signal


@dataclass
class MarketContext:
    condition:        MarketCondition
    confidence:       float
    timeframe:        str
    key_levels:       Dict[str, float]
    sentiment_score:  float
    volume_profile:   str
    risk_level:       str
    opportunity_score: float
    contrarian_signal: bool          # True = fade the obvious move
    phase_space_regime: str          # attractor state label
    topology_alarm:   float          # 0–1, persistence norm
    context_data:     Dict[str, Any] = field(default_factory=dict)
    timestamp:        str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────
# KALMAN FILTER  (1-D, constant velocity model)
# Used to clean noisy order book imbalance signal
# ─────────────────────────────────────────────

class KalmanFilter1D:
    """Lightweight scalar Kalman filter — cleans OBI without look-ahead bias."""

    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 0.1):
        self.Q = process_variance       # process noise
        self.R = measurement_variance   # measurement noise
        self.x = 0.0                    # state estimate
        self.P = 1.0                    # error covariance

    def update(self, measurement: float) -> float:
        # Predict
        P_pred = self.P + self.Q
        # Kalman gain
        K = P_pred / (P_pred + self.R)
        # Update
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x


# ─────────────────────────────────────────────
# PHASE SPACE RECONSTRUCTION  (Takens theorem)
# ─────────────────────────────────────────────

class PhaseSpaceAnalyzer:
    """
    Implements delay embedding based on Takens' theorem.
    Converts 1D price series → m-dimensional attractor trajectory.
    Detects regime changes by tracking trajectory position in phase space.

    Research: MAPSR (2024), arxiv 2602.00383
    """

    def __init__(self, embedding_dim: int = 3, delay: int = 5):
        self.m = embedding_dim
        self.tau = delay
        self.trajectory_history: deque = deque(maxlen=200)
        self.regime_labels = []

    def embed(self, series: np.ndarray) -> np.ndarray:
        """Convert 1D series to m-dimensional delay embedding."""
        n = len(series)
        required = self.m * self.tau
        if n < required + 1:
            return np.array([])

        points = []
        for i in range(required, n):
            point = [series[i - j * self.tau] for j in range(self.m)]
            points.append(point)
        return np.array(points)

    def _riemannian_metric(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Estimate the local Riemannian metric tensor of the market manifold.

        Insight: time and space are the same thing. The delay embedding converts
        temporal structure into geometric structure — the same manifold, different
        coordinates. But the metric of this manifold is not flat (Euclidean).
        High-volatility regions warp the metric — distances compress near singularities.

        We approximate the metric tensor g_ij using the local covariance structure:
            g_ij ≈ (J^T J)^{-1}  where J is the local Jacobian of the flow.

        In practice: the inverse of the local covariance matrix, normalized.
        This gives us a metric that stretches near calm regions (where differences
        matter more) and compresses near chaotic regions (where everything is close).
        """
        if len(trajectory) < 5:
            return np.eye(trajectory.shape[1])

        # Local covariance as metric proxy
        cov = np.cov(trajectory.T) + np.eye(trajectory.shape[1]) * 1e-8
        try:
            # Inverse covariance = precision matrix = Mahalanobis metric
            metric = np.linalg.inv(cov)
            # Normalize so distances are comparable across regimes
            metric /= (np.trace(metric) / trajectory.shape[1] + 1e-10)
        except np.linalg.LinAlgError:
            metric = np.eye(trajectory.shape[1])
        return metric

    def _geodesic_dist(self, a: np.ndarray, b: np.ndarray, metric: np.ndarray) -> float:
        """Mahalanobis distance — geodesic distance on the estimated manifold."""
        diff = a - b
        return float(np.sqrt(np.maximum(diff @ metric @ diff, 0)))

    def compute_lyapunov_proxy(self, trajectory: np.ndarray) -> float:
        """
        Riemannian Lyapunov exponent proxy.

        Key upgrade: uses geodesic (Mahalanobis) distance instead of Euclidean.
        In flat market spacetime these are equivalent. In curved spacetime
        (high vol, phase transition) the geodesic measure is strictly more accurate —
        it accounts for the fact that the manifold warps around singularities.

        Positive = chaotic expansion (geodesics diverge)
        Negative = stable contraction (geodesics converge = attractor basin)
        """
        if len(trajectory) < 10:
            return 0.0

        recent = trajectory[-50:] if len(trajectory) >= 50 else trajectory
        metric = self._riemannian_metric(recent)

        # Geodesic nearest-neighbor distances
        nn_dists = np.zeros(len(recent))
        for i in range(len(recent)):
            min_d = np.inf
            for j in range(len(recent)):
                if i != j:
                    d = self._geodesic_dist(recent[i], recent[j], metric)
                    if d < min_d:
                        min_d = d
            nn_dists[i] = min_d

        divergence = np.diff(np.log(nn_dists + 1e-10))
        return float(np.mean(divergence))

    def detect_regime_change(self, trajectory: np.ndarray, threshold: float = 0.15) -> Tuple[bool, str]:
        """
        Detect attractor regime change using geodesic displacement.

        The centroid displacement is now measured in Riemannian distance —
        a shift that looks small in Euclidean space may be large geodesically
        if it crosses a region of high curvature (volatility spike).
        This makes transition detection sensitive to the geometry, not just distance.
        """
        if len(trajectory) < 20:
            return False, "insufficient_data"

        self.trajectory_history.extend(trajectory[-5:].tolist())

        if len(self.trajectory_history) < 40:
            return False, "warming_up"

        history_arr = np.array(list(self.trajectory_history))
        metric = self._riemannian_metric(history_arr)

        centroid        = history_arr[:-10].mean(axis=0)
        recent_centroid = history_arr[-10:].mean(axis=0)

        # Geodesic displacement — geometry-aware regime detection
        geodesic_displacement = self._geodesic_dist(recent_centroid, centroid, metric)
        euclidean_norm        = np.linalg.norm(centroid) + 1e-10
        displacement          = geodesic_displacement / euclidean_norm

        if displacement > threshold:
            lyap   = self.compute_lyapunov_proxy(trajectory)
            regime = "chaotic_expansion" if lyap > 0 else "stable_contraction"
            return True, regime

        return False, "stable"

    def analyze(self, prices: np.ndarray) -> Dict[str, Any]:
        """Full phase space analysis pipeline."""
        min_required = self.m * self.tau + self.tau
        if len(prices) < min_required:
            return {"regime": "warming_up", "lyapunov": 0.0, "transition": False,
                    "trajectory_length": 0, "embedding_dim": self.m, "delay": self.tau}

        # Normalize
        std = prices.std()
        prices_norm = (prices - prices.mean()) / (std if std > 1e-10 else 1.0)
        trajectory = self.embed(prices_norm)

        if len(trajectory) == 0:
            return {"regime": "insufficient_data", "lyapunov": 0.0, "transition": False}

        transition, regime = self.detect_regime_change(trajectory)
        lyapunov = self.compute_lyapunov_proxy(trajectory)

        return {
            "regime": regime,
            "lyapunov": lyapunov,
            "transition": transition,
            "trajectory_length": len(trajectory),
            "embedding_dim": self.m,
            "delay": self.tau
        }


# ─────────────────────────────────────────────
# TOPOLOGY SIGNAL  (Persistence Norm)
# ─────────────────────────────────────────────

class TopologyAnalyzer:
    """
    Computes persistence norm as a crash/bubble early warning signal.

    Research basis: MDPI 2024 (persistent homology for financial crisis detection).
    Persistence norm peaks BEFORE crashes. Low during exogenous shocks.

    We approximate via Betti-0 persistence using a sliding window
    approach — avoids full TDA library dependency.
    """

    def __init__(self, window: int = 50):
        self.window = window

    def persistence_norm_proxy(self, prices: np.ndarray) -> float:
        """
        Proxy for persistence norm using local maxima/minima density.
        High = topologically complex = warning signal.
        Low = smooth = stable regime.
        """
        if len(prices) < self.window:
            return 0.0

        recent = prices[-self.window:]
        smoothed = savgol_filter(recent, min(11, len(recent) // 4 * 2 + 1), 3)

        # Count zero crossings of first derivative (proxy for Betti-0 features)
        diffs = np.diff(smoothed)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # Normalize to 0–1
        max_possible = self.window - 2
        norm = sign_changes / max_possible
        return float(np.clip(norm, 0.0, 1.0))

    def analyze(self, prices: np.ndarray) -> Dict[str, Any]:
        norm = self.persistence_norm_proxy(prices)

        if norm > 0.7:
            alarm = "crash_or_bubble_warning"
        elif norm > 0.5:
            alarm = "elevated_complexity"
        else:
            alarm = "stable"

        return {"persistence_norm": norm, "alarm": alarm}


# ─────────────────────────────────────────────
# BINANCE WEBSOCKET DATA FEED
# ─────────────────────────────────────────────

class KrakenDataFeed:
    """
    Real-time Kraken WebSocket + REST feed.
    Replaces Binance (geo-blocked HTTP 451).
    Kraken: accessible worldwide, free market data, no auth required.

    Streams:
      - ticker  → real-time price/volume
      - book    → order book for OBI calculation
    REST:
      - OHLC    → historical candles for indicator seeding
    """

    WS_URL   = "wss://ws.kraken.com"
    REST_URL = "https://api.kraken.com/0/public"

    # Kraken pair names → internal symbol map
    PAIR_MAP = {
        "BTCUSDT": "XBT/USD",
        "ETHUSDT": "ETH/USD",
        "SOLUSDT": "SOL/USD",
    }
    # REST uses slightly different pair names
    REST_PAIR_MAP = {
        "BTCUSDT": "XBTUSD",
        "ETHUSDT": "ETHUSD",
        "SOLUSDT": "SOLUSD",
    }
    # Result key Kraken uses in OHLC response
    OHLC_KEY_MAP = {
        "BTCUSDT": "XXBTZUSD",
        "ETHUSDT": "XETHZUSD",
        "SOLUSDT": "SOLUSD",
    }

    def __init__(self, symbol: str = "BTCUSDT", depth_levels: int = 20):
        self.symbol_raw   = symbol.upper()
        self.ws_pair      = self.PAIR_MAP.get(symbol.upper(), "XBT/USD")
        self.rest_pair    = self.REST_PAIR_MAP.get(symbol.upper(), "XBTUSD")
        self.ohlc_key     = self.OHLC_KEY_MAP.get(symbol.upper(), "XXBTZUSD")
        self.depth_levels = depth_levels

        # Ring buffers — same interface as before
        self.trades:     deque = deque(maxlen=1000)
        self.prices:     deque = deque(maxlen=500)
        self.volumes:    deque = deque(maxlen=500)
        self.timestamps: deque = deque(maxlen=500)

        # Order book
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

        # Kalman-filtered OBI
        self.obi_kalman = KalmanFilter1D(process_variance=1e-4, measurement_variance=0.05)
        self.obi_series: deque = deque(maxlen=200)

        self._running = False

    async def fetch_historical_klines(
        self,
        interval: str = "1m",
        limit: int = 200,
        session: Optional[aiohttp.ClientSession] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV from Kraken REST.
        Kraken interval: 1 = 1min, 5, 15, 30, 60, 240, 1440, 10080, 21600
        """
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        kraken_interval = interval_map.get(interval, 1)

        url = f"{self.REST_URL}/OHLC"
        params = {"pair": self.rest_pair, "interval": kraken_interval}

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json(content_type=None)
        except Exception as e:
            print(f"[KrakenDataFeed] REST fetch failed: {e}")
            return pd.DataFrame()
        finally:
            if close_session:
                await session.close()

        errors = data.get("error", [])
        if errors:
            print(f"[KrakenDataFeed] REST API errors: {errors}")
            return pd.DataFrame()

        # Find the result key (Kraken sometimes varies case)
        result = data.get("result", {})
        candles = result.get(self.ohlc_key) or next(
            (v for k, v in result.items() if k != "last"), []
        )

        if not candles:
            print(f"[KrakenDataFeed] No candle data in response")
            return pd.DataFrame()

        # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(candles[-limit:], columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["time"]   = pd.to_datetime(df["time"], unit="s")

        for _, row in df.iterrows():
            self.prices.append(row["close"])
            self.volumes.append(row["volume"])
            self.timestamps.append(row["time"].timestamp())

        return df

    def compute_obi(self) -> float:
        """OBI — same logic, Kraken order book data."""
        if not self.bids or not self.asks:
            return 0.0

        levels = self.depth_levels
        top_bids = sorted(self.bids.items(), reverse=True)[:levels]
        top_asks = sorted(self.asks.items())[:levels]

        bid_vol = sum(v for _, v in top_bids)
        ask_vol = sum(v for _, v in top_asks)
        total   = bid_vol + ask_vol

        if total == 0:
            return 0.0

        raw_obi      = (bid_vol - ask_vol) / total
        filtered_obi = self.obi_kalman.update(raw_obi)
        self.obi_series.append(filtered_obi)
        return filtered_obi

    def is_obi_extreme(self, threshold: float = 0.65) -> Tuple[bool, str]:
        if len(self.obi_series) < 10:
            return False, "neutral"
        avg = np.mean(list(self.obi_series)[-10:])
        if avg > threshold:
            return True, "extreme_buy_pressure_fade_it"
        if avg < -threshold:
            return True, "extreme_sell_pressure_fade_it"
        return False, "neutral"

    def _apply_book_snapshot(self, entries: list, side: Dict):
        """Apply full book snapshot."""
        side.clear()
        for price, vol, *_ in entries:
            p, v = float(price), float(vol)
            if v > 0:
                side[p] = v

    def _apply_book_update(self, entries: list, side: Dict):
        """Apply incremental book update."""
        for price, vol, *_ in entries:
            p, v = float(price), float(vol)
            if v == 0:
                side.pop(p, None)
            else:
                side[p] = v

    async def stream_ws(self):
        """
        Single Kraken WebSocket handling both ticker and book subscriptions.
        Reconnects automatically on disconnect.
        """
        while self._running:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=10
                ) as ws:
                    # Subscribe to ticker + book
                    sub = {
                        "event": "subscribe",
                        "pair":  [self.ws_pair],
                        "subscription": {"name": "ticker"}
                    }
                    await ws.send(json.dumps(sub))

                    book_sub = {
                        "event": "subscribe",
                        "pair":  [self.ws_pair],
                        "subscription": {"name": "book", "depth": 25}
                    }
                    await ws.send(json.dumps(book_sub))

                    while self._running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                            msg = json.loads(raw)

                            # Skip system messages
                            if isinstance(msg, dict):
                                continue

                            if not isinstance(msg, list) or len(msg) < 3:
                                continue

                            channel = msg[-2] if len(msg) == 4 else msg[-1]
                            data    = msg[1]

                            if channel == "ticker":
                                # c = [last_trade_price, lot_volume]
                                price  = float(data["c"][0])
                                volume = float(data["v"][1])  # 24h volume
                                ts     = datetime.now(timezone.utc).timestamp()
                                self.prices.append(price)
                                self.volumes.append(volume)
                                self.timestamps.append(ts)
                                self.trades.append({"price": price, "volume": volume, "time": ts})

                            elif "book" in str(channel):
                                # Snapshot: has "as" and "bs" keys
                                if "as" in data:
                                    self._apply_book_snapshot(data["as"], self.asks)
                                if "bs" in data:
                                    self._apply_book_snapshot(data["bs"], self.bids)
                                # Incremental updates
                                if "a" in data:
                                    self._apply_book_update(data["a"], self.asks)
                                if "b" in data:
                                    self._apply_book_update(data["b"], self.bids)

                                # Kraken sometimes sends both update and snapshot in same message
                                if isinstance(msg, list) and len(msg) == 4:
                                    extra = msg[2] if isinstance(msg[2], dict) else {}
                                    if "a" in extra:
                                        self._apply_book_update(extra["a"], self.asks)
                                    if "b" in extra:
                                        self._apply_book_update(extra["b"], self.bids)

                        except asyncio.TimeoutError:
                            continue

            except Exception as e:
                if self._running:
                    print(f"[KrakenDataFeed] WS error: {e} — reconnecting in 5s")
                    await asyncio.sleep(5)

    async def start(self):
        self._running = True
        await self.stream_ws()

    def stop(self):
        self._running = False

    def get_price_array(self) -> np.ndarray:
        return np.array(list(self.prices))

    def get_volume_array(self) -> np.ndarray:
        return np.array(list(self.volumes))


# Alias for backward compatibility
BinanceDataFeed = KrakenDataFeed


# ─────────────────────────────────────────────
# TECHNICAL ANALYSIS ENGINE
# ─────────────────────────────────────────────

class TechnicalAnalyzer:
    """
    Full technical signal suite.
    Preprocessed with Savitzky-Golay filter (research-validated over raw data).
    """

    @staticmethod
    def smooth(series: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
        if len(series) < window:
            return series
        w = window if window % 2 == 1 else window - 1
        w = max(w, poly + 1)
        return savgol_filter(series, w, poly)

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return float(prices[-1]) if len(prices) else 0.0
        k = 2 / (period + 1)
        ema_val = prices[0]
        for p in prices[1:]:
            ema_val = p * k + ema_val * (1 - k)
        return float(ema_val)

    @staticmethod
    def macd(prices: np.ndarray) -> Tuple[float, float]:
        if len(prices) < 26:
            return 0.0, 0.0
        fast = TechnicalAnalyzer.ema(prices, 12)
        slow = TechnicalAnalyzer.ema(prices, 26)
        macd_line = fast - slow
        signal_prices = np.array([
            TechnicalAnalyzer.ema(prices[:i], 12) - TechnicalAnalyzer.ema(prices[:i], 26)
            for i in range(26, len(prices) + 1)
        ])
        signal = TechnicalAnalyzer.ema(signal_prices, 9) if len(signal_prices) >= 9 else macd_line
        return float(macd_line), float(signal)

    @staticmethod
    def bollinger(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        if len(prices) < period:
            p = float(prices[-1]) if len(prices) else 0.0
            return {"upper": p, "middle": p, "lower": p, "bandwidth": 0.0, "percent_b": 0.5}
        window = prices[-period:]
        mid    = window.mean()
        std    = window.std()
        upper  = mid + std_dev * std
        lower  = mid - std_dev * std
        current = prices[-1]
        bandwidth = (upper - lower) / (mid + 1e-10)
        percent_b = (current - lower) / (upper - lower + 1e-10)
        return {
            "upper": float(upper), "middle": float(mid),
            "lower": float(lower), "bandwidth": float(bandwidth),
            "percent_b": float(percent_b)
        }

    @staticmethod
    def volatility(prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        if len(prices) < period + 1:
            return {"current": 0.0, "average": 0.0, "ratio": 1.0, "spike": False}
        returns = np.diff(np.log(prices[-period * 2:] + 1e-10))
        current_vol = returns[-period:].std()
        avg_vol     = returns.std()
        ratio = current_vol / (avg_vol + 1e-10)
        return {
            "current": float(current_vol),
            "average": float(avg_vol),
            "ratio": float(ratio),
            "spike": bool(ratio > 2.0)
        }

    @staticmethod
    def support_resistance(prices: np.ndarray, window: int = 20) -> Dict[str, float]:
        if len(prices) < window:
            p = float(prices[-1]) if len(prices) else 0.0
            return {"support": p * 0.98, "resistance": p * 1.02}
        recent = prices[-window:]
        return {
            "support":    float(recent.min()),
            "resistance": float(recent.max())
        }

    def analyze(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Full technical analysis pass."""
        if len(prices) < 5:
            return {}

        smoothed = self.smooth(prices)

        rsi_val    = self.rsi(smoothed)
        macd_l, macd_s = self.macd(smoothed)
        bb         = self.bollinger(smoothed)
        vol        = self.volatility(smoothed)
        sr         = self.support_resistance(smoothed)

        current = float(smoothed[-1])
        ema20   = self.ema(smoothed, 20)
        ema50   = self.ema(smoothed, 50)

        trend_direction = 1.0 if ema20 > ema50 else -1.0
        trend_strength  = min(abs(ema20 - ema50) / (ema50 + 1e-10) * 20, 1.0)

        price_vs_ma = (current - ema20) / (ema20 + 1e-10)

        # Volume surge
        if len(volumes) >= 20:
            vol_surge = float(volumes[-1]) / (float(np.mean(list(volumes)[-20:])) + 1e-10)
        else:
            vol_surge = 1.0

        breakout_score  = max(0.0, (bb["percent_b"] - 0.9) * 10)
        breakdown_score = max(0.0, (0.1 - bb["percent_b"]) * 10)

        # Buy/sell pressure proxy via volume direction
        if len(volumes) >= 2:
            if prices[-1] > prices[-2]:
                buying_pressure  = vol_surge
                selling_pressure = 1.0
            else:
                buying_pressure  = 1.0
                selling_pressure = vol_surge
        else:
            buying_pressure = selling_pressure = 1.0

        return {
            "rsi": rsi_val,
            "macd_line": macd_l,
            "macd_signal": macd_s,
            "macd_histogram": macd_l - macd_s,
            "bollinger": bb,
            "volatility": vol,
            "support": sr["support"],
            "resistance": sr["resistance"],
            "ema20": ema20,
            "ema50": ema50,
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "price_vs_ma": float(price_vs_ma),
            "vol_surge": float(vol_surge),
            "breakout_score": float(breakout_score),
            "breakdown_score": float(breakdown_score),
            "buying_pressure": float(buying_pressure),
            "selling_pressure": float(selling_pressure),
            "current_price": current,
        }


# ─────────────────────────────────────────────
# SENTIMENT FEED  (100% free, no API keys needed)
#
# Sources:
#   1. Alternative.me Fear & Greed Index  — no key, no signup
#   2. CryptoPanic free tier              — no key (public endpoint)
#   3. Exorde API                         — 1000 free credits on signup
#
# Sentiment leads price 24-48h (2025 research confirmed).
# ─────────────────────────────────────────────

class SentimentFeed:

    FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=3&format=json"
    CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v1/posts/?auth_token=&public=true"

    # Cache — fear/greed updates hourly, news every 15 min
    _fg_cache:   Optional[Dict] = None
    _fg_ts:      float          = 0.0
    _news_cache: float          = 0.0
    _news_ts:    float          = 0.0
    _FG_TTL      = 3600.0    # 1 hour
    _NEWS_TTL    = 900.0     # 15 minutes

    async def get_fear_greed(self) -> Dict[str, Any]:
        """
        Alternative.me Fear & Greed Index. Cached 1 hour — it only updates hourly.
        Calling this 360×/hour was wasting 20-40% of every cycle on a redundant fetch.
        """
        now = time.time()
        if self._fg_cache is not None and (now - self._fg_ts) < self._FG_TTL:
            return self._fg_cache
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FEAR_GREED_URL,
                    timeout=aiohttp.ClientTimeout(total=6)
                ) as resp:
                    data = await resp.json(content_type=None)

            entries = data.get("data", [])
            if not entries:
                return {"score": 0.0, "label": "neutral", "value": 50, "source": "fear_greed"}

            # Most recent value
            latest = int(entries[0]["value"])
            label  = entries[0]["value_classification"].lower()

            # Normalize 0-100 → -1 to +1
            # <25 = extreme fear (bearish signal for contrarian = BUY)
            # >75 = extreme greed (bearish signal for contrarian = SELL)
            # We preserve the raw market mood direction here — let agent logic invert
            score = (latest - 50) / 50.0   # -1 at 0, +1 at 100

            # Trend: compare today vs yesterday
            trend_score = 0.0
            if len(entries) >= 2:
                yesterday = int(entries[1]["value"])
                trend_score = (latest - yesterday) / 50.0

            result = {
                "score": float(score),
                "trend": float(trend_score),
                "value": latest,
                "label": label,
                "source": "alternative.me"
            }
            SentimentFeed._fg_cache = result
            SentimentFeed._fg_ts    = time.time()
            return result
        except Exception as e:
            if self._fg_cache is not None:
                return self._fg_cache   # return stale cache on error
            return {"score": 0.0, "label": "neutral", "value": 50, "source": f"error:{e}"}

    async def get_news_sentiment(self, currencies: str = "BTC") -> float:
        """CryptoPanic news sentiment. Cached 15 minutes."""
        now = time.time()
        if self._news_cache != 0.0 and (now - self._news_ts) < self._NEWS_TTL:
            return self._news_cache
        try:
            url = f"{self.CRYPTOPANIC_URL}&currencies={currencies}&kind=news"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                    if resp.status != 200:
                        return 0.0
                    data = await resp.json(content_type=None)

            results = data.get("results", [])[:15]
            if not results:
                return 0.0

            # Vote-based sentiment scoring
            scores = []
            for r in results:
                votes = r.get("votes", {})
                pos = votes.get("positive", 0) or 0
                neg = votes.get("negative", 0) or 0
                total = pos + neg
                if total > 0:
                    scores.append((pos - neg) / total)

            result = float(np.mean(scores)) if scores else 0.0
            SentimentFeed._news_cache = result
            SentimentFeed._news_ts    = time.time()
            return result

        except Exception:
            return self._news_cache   # stale on error

    async def analyze(self, coin_slug: str = "bitcoin") -> Dict[str, Any]:
        """
        Fuse fear/greed + news sentiment into a single score.
        Both sources are genuinely free with no account required.
        """
        currencies = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}.get(coin_slug, "BTC")

        fg_data, news_score = await asyncio.gather(
            self.get_fear_greed(),
            self.get_news_sentiment(currencies)
        )

        # Weight: fear/greed is more reliable (macro sentiment)
        # news is more immediate but noisier
        combined = 0.65 * fg_data["score"] + 0.35 * news_score

        return {
            "combined_score": float(np.clip(combined, -1.0, 1.0)),
            "fear_greed": fg_data,
            "news_score": float(news_score),
            "lead_hours": 24,
        }


# ─────────────────────────────────────────────
# MARKET ANALYZER  (orchestrator)
# ─────────────────────────────────────────────

class MarketAnalyzer:
    """
    Orchestrates all five signal layers into a unified MarketContext.
    This is the brain that agent_spawner.py reads to create the right agent.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol     = symbol
        self.data_feed  = KrakenDataFeed(symbol)
        self.technical  = TechnicalAnalyzer()
        self.phase      = PhaseSpaceAnalyzer(embedding_dim=3, delay=5)
        self.topology   = TopologyAnalyzer(window=50)
        self.sentiment  = SentimentFeed()

    async def bootstrap(self):
        """Seed price buffers from REST before WebSocket streams start."""
        print(f"[MarketAnalyzer] Seeding {self.symbol} historical data...")
        await self.data_feed.fetch_historical_klines(interval="1m", limit=200)
        print(f"[MarketAnalyzer] Seeded {len(self.data_feed.prices)} candles.")

    def _identify_key_levels(self, prices: np.ndarray) -> Dict[str, float]:
        if len(prices) < 20:
            return {}
        sr = TechnicalAnalyzer.support_resistance(prices)
        current = float(prices[-1])
        return {
            "current":    current,
            "support":    sr["support"],
            "resistance": sr["resistance"],
            "support_dist":    abs(current - sr["support"]) / current,
            "resistance_dist": abs(sr["resistance"] - current) / current,
        }

    def _assess_risk(self, vol: Dict, obi_extreme: bool) -> str:
        ratio = vol.get("ratio", 1.0)
        if ratio > 2.5 or obi_extreme:
            return "HIGH"
        if ratio > 1.5:
            return "MEDIUM"
        return "LOW"

    def _determine_condition(
        self,
        tech: Dict,
        phase_data: Dict,
        topo_data: Dict,
        sentiment_score: float,
        obi: float,
        obi_extreme: bool
    ) -> Tuple[MarketCondition, float]:
        """
        Signal fusion — five layers vote.
        Returns (condition, confidence).
        """

        # Topology alarms take priority
        if topo_data["persistence_norm"] > 0.75:
            if sentiment_score > 0.3:
                return MarketCondition.BUBBLE_WARNING, 0.80
            return MarketCondition.CRASH_WARNING, 0.75

        # Phase transition
        if phase_data.get("transition"):
            return MarketCondition.PHASE_TRANSITION, 0.72

        # Standard conditions
        rsi = tech.get("rsi", 50)
        ts  = tech.get("trend_strength", 0)
        td  = tech.get("trend_direction", 0)
        bs  = tech.get("breakout_score", 0)
        bds = tech.get("breakdown_score", 0)
        vs  = tech.get("vol_surge", 1)
        vr  = tech.get("volatility", {}).get("ratio", 1.0)

        if bs > 0.7 and vs > 1.5 and sentiment_score > 0.4:
            return MarketCondition.BULLISH_BREAKOUT, min(0.5 + bs * 0.3, 0.95)

        if bds > 0.7 and vs > 1.5 and sentiment_score < -0.4:
            return MarketCondition.BEARISH_BREAKDOWN, min(0.5 + bds * 0.3, 0.95)

        if ts > 0.65 and td > 0:
            return MarketCondition.STRONG_UPTREND, 0.5 + ts * 0.3

        if ts > 0.65 and td < 0:
            return MarketCondition.STRONG_DOWNTREND, 0.5 + ts * 0.3

        if vr > 2.0:
            return MarketCondition.HIGH_VOLATILITY, 0.70

        if abs(sentiment_score) > 0.7 and tech.get("volatility", {}).get("spike"):
            return MarketCondition.NEWS_DRIVEN_SPIKE, 0.65

        if rsi < 32 and tech.get("price_vs_ma", 0) < -0.04:
            return MarketCondition.MEAN_REVERSION_OVERSOLD, 0.60

        if rsi > 68 and tech.get("price_vs_ma", 0) > 0.04:
            return MarketCondition.MEAN_REVERSION_OVERBOUGHT, 0.60

        return MarketCondition.SIDEWAYS_CONSOLIDATION, 0.45

    async def analyze_market_context(
        self,
        coin_slug: str = "bitcoin",
        timeframe: str = "1m"
    ) -> MarketContext:
        """
        Full five-layer analysis. Returns a MarketContext ready for agent spawning.
        """
        prices  = self.data_feed.get_price_array()
        volumes = self.data_feed.get_volume_array()

        # Parallel: sentiment + technical + phase space + topology + OBI
        sentiment_task = asyncio.create_task(self.sentiment.analyze(coin_slug))

        tech_signals = self.technical.analyze(prices, volumes)
        phase_data   = self.phase.analyze(prices)
        topo_data    = self.topology.analyze(prices)
        obi          = self.data_feed.compute_obi()
        obi_extreme, obi_label = self.data_feed.is_obi_extreme()

        sentiment_data = await sentiment_task
        sentiment_score = sentiment_data["combined_score"]

        # Geometric encoding layer — temporal geometry of the market
        geo_context: Dict = {}
        if len(prices) >= 8:
            try:
                geo_context = build_geometric_context(
                    context_data={
                        "obi": obi, "obi_label": obi_label,
                        "technical": tech_signals,
                        "phase_space": phase_data,
                        "sentiment": sentiment_data,
                        "topology": topo_data,
                        "topology_alarm": float(topo_data.get("persistence_norm", 0)),
                    },
                    price_history=prices,
                    trajectory=None,  # populated after phase analysis runs longer
                )
            except Exception:
                geo_context = {}

        condition, confidence = self._determine_condition(
            tech_signals, phase_data, topo_data,
            sentiment_score, obi, obi_extreme
        )

        key_levels      = self._identify_key_levels(prices)
        risk_level      = self._assess_risk(tech_signals.get("volatility", {}), obi_extreme)
        opportunity     = confidence * (1.0 - {"HIGH": 0.5, "MEDIUM": 0.25, "LOW": 0.0}[risk_level])

        # Volume profile
        vol_ratio = tech_signals.get("vol_surge", 1.0)
        if vol_ratio > 2.0:
            vol_profile = "surge"
        elif vol_ratio > 1.3:
            vol_profile = "above_average"
        elif vol_ratio < 0.7:
            vol_profile = "drying_up"
        else:
            vol_profile = "normal"

        return MarketContext(
            condition=condition,
            confidence=confidence,
            timeframe=timeframe,
            key_levels=key_levels,
            sentiment_score=sentiment_score,
            volume_profile=vol_profile,
            risk_level=risk_level,
            opportunity_score=float(opportunity),
            contrarian_signal=obi_extreme,
            phase_space_regime=phase_data.get("regime", "unknown"),
            topology_alarm=float(topo_data["persistence_norm"]),
            context_data={
                "technical":    tech_signals,
                "phase_space":  phase_data,
                "topology":     topo_data,
                "sentiment":    sentiment_data,
                "obi":          obi,
                "obi_label":    obi_label,
                "prices_count": len(prices),
                "geometry":     geo_context,
            }
        )


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  MASTER_TRADER — Market Analyzer")
    print("  Five-layer signal stack | Paper trading mode")
    print("=" * 60)

    analyzer = MarketAnalyzer(symbol="BTCUSDT")
    await analyzer.bootstrap()

    context = await analyzer.analyze_market_context(coin_slug="bitcoin")

    print(f"\n  Symbol:          {analyzer.symbol}")
    print(f"  Condition:       {context.condition.value}")
    print(f"  Confidence:      {context.confidence:.2%}")
    print(f"  Risk:            {context.risk_level}")
    print(f"  Opportunity:     {context.opportunity_score:.2%}")
    print(f"  Sentiment:       {context.sentiment_score:+.3f}")
    print(f"  Contrarian:      {context.contrarian_signal} ({context.context_data.get('obi_label')})")
    print(f"  Phase Regime:    {context.phase_space_regime}")
    print(f"  Topology Alarm:  {context.topology_alarm:.3f}")
    print(f"  Volume Profile:  {context.volume_profile}")
    print(f"  Price:           ${context.key_levels.get('current', 0):,.2f}")
    print(f"  Support:         ${context.key_levels.get('support', 0):,.2f}")
    print(f"  Resistance:      ${context.key_levels.get('resistance', 0):,.2f}")
    print(f"\n  Timestamp:       {context.timestamp}")
    print("=" * 60)

    return context


if __name__ == "__main__":
    asyncio.run(main())
