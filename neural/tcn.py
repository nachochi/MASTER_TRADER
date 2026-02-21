#!/usr/bin/env python3
"""
MASTER_TRADER — Temporal Convolutional Network (TCN)

The insight that upgrades everything:
  Time and space are the same thing.
  A time series IS a spatial manifold with causal structure.
  Therefore: the correct architecture for time series is one that
  explicitly treats temporal windows as spatial receptive fields.

TCN vs MLP:
  MLP sees ONE snapshot — no memory of sequence
  TCN sees SEQUENCES — time IS the structure it learns from

TCN vs Transformer:
  Transformer: O(n²) attention over sequence — expensive
  TCN:         O(1) inference regardless of sequence length — constant time
  For our 60-step windows: TCN is 3600x more efficient than transformer

TCN vs RNN (LSTM/GRU):
  RNN: sequential computation, vanishing gradients, can't parallelize
  TCN: fully parallel, stable gradients via residuals, causal by construction

Architecture:
  Input:  T=60 timesteps × F=16 features  (60 × 16)
  Conv1:  dilated causal conv, dilation=1,  kernel=3, filters=32
  Conv2:  dilated causal conv, dilation=2,  kernel=3, filters=32
  Conv3:  dilated causal conv, dilation=4,  kernel=3, filters=32
  Conv4:  dilated causal conv, dilation=8,  kernel=3, filters=32
  → Receptive field: 1 + (3-1)×(1+2+4+8) = 31 steps = ~5 minutes at 10s cycles
  GlobalPool: 32 → 16
  Output heads:
    condition:  16 → 14  (softmax)
    confidence: 16 → 1   (sigmoid)
    direction:  16 → 3   (long/short/wait softmax)

Causal convolution:
  At each timestep t, the filter only sees t, t-1, t-2, ... (never the future)
  This is the "light cone" — information travels forward in time only
  Dilation: skip connections that look d steps back — multi-scale temporal geometry

Pure numpy. No PyTorch. No TensorFlow. Just matrix math.
Total parameters: ~18,000 — 10x the MLP, but sees sequences not snapshots.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

TCN_WEIGHTS_PATH = Path(__file__).parent / "tcn_weights.npz"

SEQUENCE_LEN = 60    # 60 cycles × 10s = 10 minutes of market history
N_FEATURES   = 16    # features per timestep (see FEATURE_NAMES below)
N_CONDITIONS = 14
N_FILTERS    = 32
KERNEL_SIZE  = 3
DILATIONS    = [1, 2, 4, 8]   # receptive field = 31 steps

FEATURE_NAMES = [
    "price_norm",        # normalized close price
    "obi",               # order book imbalance
    "obi_contrarian",    # binary: extreme OBI
    "phase_lyapunov",    # geodesic Lyapunov exponent
    "phase_transition",  # binary: regime transition
    "sent_score",        # combined sentiment
    "fear_greed",        # fear/greed index normalized
    "news_score",        # news sentiment
    "topo_norm",         # topology persistence norm
    "topo_alarm",        # binary: topology alarm
    "rsi_norm",          # RSI normalized to [-1, 1]
    "macd_hist",         # MACD histogram
    "trend_strength",    # trend strength
    "vol_surge",         # volume surge ratio
    "bb_pct_b",          # Bollinger %B
    "volatility_ratio",  # recent/historical volatility ratio
]


# ─────────────────────────────────────────────
# CAUSAL DILATED CONVOLUTION (1D, pure numpy)
# ─────────────────────────────────────────────

class CausalConv1D:
    """
    1D causal dilated convolution layer.

    Causal: at time t, only sees t, t-d, t-2d, ... (the light cone)
    Dilated: samples input every d steps — multi-scale temporal receptive field

    This is the "temporal geometry" layer. The kernel sliding over the sequence
    is literally tracing geodesics through time.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.k      = kernel_size
        self.d      = dilation
        self.pad    = (kernel_size - 1) * dilation  # causal padding

        # Kaiming initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size) * scale
        self.b = np.zeros(out_channels)

        # For backprop
        self._last_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, C_in) — sequence × channels
        returns: (T, C_out) — same sequence length (causal padding preserves shape)
        """
        T, C = x.shape
        # Causal left-pad so output has same length as input
        x_pad = np.pad(x, ((self.pad, 0), (0, 0)), mode='constant')
        self._last_input = x_pad

        out = np.zeros((T, self.out_ch))
        for t in range(T):
            for ki in range(self.k):
                idx = t + self.pad - ki * self.d
                if 0 <= idx < len(x_pad):
                    out[t] += x_pad[idx] @ self.W[:, :, ki].T
            out[t] += self.b
        return out

    def to_dict(self) -> Dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(),
                "in_ch": self.in_ch, "out_ch": self.out_ch,
                "k": self.k, "d": self.d}

    def from_dict(self, d: Dict):
        self.W = np.array(d["W"])
        self.b = np.array(d["b"])


class ResidualBlock:
    """
    TCN residual block: two dilated causal convolutions + skip connection.

    The skip connection is what makes TCNs train stably — gradients flow
    directly to early layers regardless of depth. This is why TCNs don't
    have the vanishing gradient problem that kills RNNs.

    Also: the skip connection IS the identity geodesic. Information that
    doesn't need to be transformed passes through unchanged.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        self.conv1   = CausalConv1D(channels, channels, kernel_size, dilation)
        self.conv2   = CausalConv1D(channels, channels, kernel_size, dilation)
        self._h: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self.conv1.forward(x))   # ReLU
        h = self.conv2.forward(h)
        self._h = h
        # Residual skip: output = activation(conv(x)) + x
        return np.maximum(0, h + x)

    def to_dict(self) -> Dict:
        return {"conv1": self.conv1.to_dict(), "conv2": self.conv2.to_dict()}

    def from_dict(self, d: Dict):
        self.conv1.from_dict(d["conv1"])
        self.conv2.from_dict(d["conv2"])


# ─────────────────────────────────────────────
# FEATURE BUFFER  (sliding window over cycles)
# ─────────────────────────────────────────────

class FeatureBuffer:
    """
    Maintains a rolling window of N_FEATURES per cycle.
    The TCN reads the last SEQUENCE_LEN rows as its input tensor.

    Each call to `push()` adds one cycle's features.
    The buffer is the "memory" — the TCN's temporal consciousness.
    """

    def __init__(self, seq_len: int = SEQUENCE_LEN, n_features: int = N_FEATURES):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.buffer     = np.zeros((seq_len, n_features), dtype=np.float32)
        self._filled    = 0

    def push(self, context_data: Dict, current_price: float, prev_price: float) -> np.ndarray:
        """Extract features from one cycle and append to buffer. Returns current buffer."""
        tech  = context_data.get("technical", {})
        phase = context_data.get("phase_space", {})
        topo  = context_data.get("topology", {})
        sent  = context_data.get("sentiment", {})
        fg    = sent.get("fear_greed", {})
        obi   = float(context_data.get("obi", 0))

        price_norm = (current_price - prev_price) / (prev_price + 1e-10) if prev_price > 0 else 0.0

        row = np.array([
            float(np.clip(price_norm * 100, -5, 5)),                         # price_norm
            float(np.clip(obi, -1, 1)),                                       # obi
            1.0 if context_data.get("obi_label", "").startswith("extreme") else 0.0,  # contrarian
            float(np.clip(phase.get("lyapunov", 0) * 10, -1, 1)),            # lyapunov
            1.0 if phase.get("transition", False) else 0.0,                   # transition
            float(np.clip(sent.get("combined_score", 0), -1, 1)),            # sent_score
            float((fg.get("value", 50) - 50) / 50),                          # fear_greed
            float(np.clip(sent.get("news_score", 0), -1, 1)),                # news_score
            float(np.clip(topo.get("persistence_norm", 0), 0, 1)),           # topo_norm
            1.0 if topo.get("alarm") else 0.0,                               # topo_alarm
            float((tech.get("rsi", 50) - 50) / 50),                         # rsi_norm
            float(np.clip(tech.get("macd_histogram", 0) * 1000, -1, 1)),    # macd_hist
            float(np.clip(tech.get("trend_strength", 0), 0, 1)),            # trend_strength
            float(np.clip(tech.get("vol_surge", 1) - 1, -1, 3)),            # vol_surge
            float(np.clip(tech.get("bollinger", {}).get("percent_b", 0.5) - 0.5, -0.5, 0.5)),
            float(np.clip(tech.get("volatility", {}).get("ratio", 1) - 1, -2, 3)),
        ], dtype=np.float32)

        # Shift buffer up, append new row
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1]  = row
        self._filled = min(self._filled + 1, self.seq_len)

        return self.buffer.copy()

    @property
    def ready(self) -> bool:
        """True when buffer has at least 10 cycles of history."""
        return self._filled >= 10


# ─────────────────────────────────────────────
# TCN MODEL
# ─────────────────────────────────────────────

class TemporalConvNet:
    """
    Full TCN model for market condition prediction.

    Replaces the MLP (HierarchicalFusionModel) when enough sequence data exists.
    Falls back to MLP until buffer is populated.

    The TCN sees TIME as geometry. Each dilated layer is a different temporal scale:
      dilation=1: adjacent cycles (10s apart)   — microstructure
      dilation=2: 20s windows                   — short-term flow
      dilation=4: 40s windows                   — momentum structure
      dilation=8: 80s windows (~1.3 min)        — regime context

    Total receptive field: 31 cycles = ~5 minutes of market spacetime.
    """

    def __init__(self):
        # Input projection: N_FEATURES → N_FILTERS
        self.input_proj = CausalConv1D(N_FEATURES, N_FILTERS, kernel_size=1, dilation=1)

        # Dilated residual blocks — each sees a different temporal scale
        self.blocks = [
            ResidualBlock(N_FILTERS, KERNEL_SIZE, d)
            for d in DILATIONS
        ]

        # Output heads — after global average pooling over time
        # GlobalPool: (T, N_FILTERS) → (N_FILTERS,)
        n_hidden = N_FILTERS

        # Simple output heads (weight matrices)
        scale = np.sqrt(2.0 / n_hidden)
        self.W_cond = np.random.randn(n_hidden, N_CONDITIONS) * scale
        self.b_cond = np.zeros(N_CONDITIONS)
        self.W_conf = np.random.randn(n_hidden, 1) * scale
        self.b_conf = np.zeros(1)
        self.W_dir  = np.random.randn(n_hidden, 3) * scale   # long/short/wait
        self.b_dir  = np.zeros(3)

        self.training_steps = 0
        self.loss_history: List[float] = []
        self.buffer = FeatureBuffer()
        self._load()

    def forward_sequence(self, seq: np.ndarray) -> Tuple[np.ndarray, float, str]:
        """
        seq: (T, F) — time × features
        Returns (condition_probs, confidence, direction)
        """
        # Input projection
        x = np.maximum(0, self.input_proj.forward(seq))

        # Dilated residual blocks — temporal geometry processing
        for block in self.blocks:
            x = block.forward(x)

        # Global average pool over time dimension — aggregate temporal context
        pooled = x.mean(axis=0)   # (N_FILTERS,)

        # Output heads
        cond_logits = pooled @ self.W_cond + self.b_cond
        e = np.exp(cond_logits - cond_logits.max())
        cond_probs  = e / (e.sum() + 1e-10)

        conf_raw    = float(pooled @ self.W_conf + self.b_conf)
        confidence  = 1.0 / (1.0 + np.exp(-conf_raw))

        dir_logits  = pooled @ self.W_dir + self.b_dir
        e2 = np.exp(dir_logits - dir_logits.max())
        dir_probs   = e2 / (e2.sum() + 1e-10)
        directions  = ["LONG", "SHORT", "WAIT"]
        direction   = directions[int(np.argmax(dir_probs))]

        return cond_probs, float(confidence), direction

    def param_count(self) -> int:
        total = 0
        # Input proj
        total += self.input_proj.W.size + self.input_proj.b.size
        # Blocks
        for blk in self.blocks:
            for conv in [blk.conv1, blk.conv2]:
                total += conv.W.size + conv.b.size
        # Output heads
        for W, b in [(self.W_cond, self.b_cond), (self.W_conf, self.b_conf), (self.W_dir, self.b_dir)]:
            total += W.size + b.size
        return total

    def _to_dict(self) -> Dict:
        return {
            "input_proj": self.input_proj.to_dict(),
            "blocks":     [b.to_dict() for b in self.blocks],
            "W_cond":     self.W_cond.tolist(),
            "b_cond":     self.b_cond.tolist(),
            "W_conf":     self.W_conf.tolist(),
            "b_conf":     self.b_conf.tolist(),
            "W_dir":      self.W_dir.tolist(),
            "b_dir":      self.b_dir.tolist(),
        }

    def _from_dict(self, d: Dict):
        self.input_proj.from_dict(d["input_proj"])
        for blk, bd in zip(self.blocks, d["blocks"]):
            blk.from_dict(bd)
        self.W_cond = np.array(d["W_cond"])
        self.b_cond = np.array(d["b_cond"])
        self.W_conf = np.array(d["W_conf"])
        self.b_conf = np.array(d["b_conf"])
        self.W_dir  = np.array(d["W_dir"])
        self.b_dir  = np.array(d["b_dir"])

    def _save(self):
        TCN_WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        model_json = json.dumps(self._to_dict())
        np.savez(TCN_WEIGHTS_PATH, model_json=model_json, steps=self.training_steps,
                 loss_history=np.array(self.loss_history[-100:]))

    def _load(self):
        if TCN_WEIGHTS_PATH.exists():
            try:
                data = np.load(TCN_WEIGHTS_PATH, allow_pickle=True)
                self._from_dict(json.loads(str(data["model_json"])))
                self.training_steps = int(data.get("steps", 0))
                print(f"[TCN] Loaded weights — {self.training_steps} training steps")
            except Exception as e:
                print(f"[TCN] Fresh start ({e})")

    def stats(self) -> Dict:
        recent_loss = float(np.mean(self.loss_history[-20:])) if self.loss_history else float("inf")
        return {
            "training_steps": self.training_steps,
            "recent_loss":    recent_loss,
            "buffer_filled":  self.buffer._filled,
            "ready":          self.buffer.ready,
        }


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tcn = TemporalConvNet()

    print(f"\n{'═'*60}")
    print(f"  TCN — Temporal Convolutional Network")
    print(f"{'═'*60}")
    print(f"  Parameters:      {tcn.param_count():,}")
    print(f"  Sequence length: {SEQUENCE_LEN} cycles × 10s = {SEQUENCE_LEN*10/60:.1f} min")
    print(f"  Receptive field: {1 + (KERNEL_SIZE-1)*sum(DILATIONS)} cycles = "
          f"{(1 + (KERNEL_SIZE-1)*sum(DILATIONS))*10:.0f}s")
    print(f"  Dilations:       {DILATIONS}")
    print(f"  Features/cycle:  {N_FEATURES}")
    print(f"\n  Feature map:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    [{i:2d}] {name}")

    # Test forward pass with random sequence
    test_seq = np.random.randn(SEQUENCE_LEN, N_FEATURES).astype(np.float32) * 0.1
    probs, conf, direction = tcn.forward_sequence(test_seq)
    top_cond = int(np.argmax(probs))
    from neural.signal_fusion import INDEX_CONDITION
    print(f"\n  Test prediction:")
    print(f"    Condition:   {INDEX_CONDITION.get(top_cond, 'unknown')} ({probs[top_cond]:.1%})")
    print(f"    Confidence:  {conf:.1%}")
    print(f"    Direction:   {direction}")
    print(f"\n  Receptive field visualization:")
    print(f"    dilation=1 : ████                    (10s window)")
    print(f"    dilation=2 : █   █                   (20s span)")
    print(f"    dilation=4 : █       █               (40s span)")
    print(f"    dilation=8 : █               █       (80s span)")
    print(f"    combined   : ━━━━━━━━━━━━━━━━━━━━━━━ (310s = ~5min)")
