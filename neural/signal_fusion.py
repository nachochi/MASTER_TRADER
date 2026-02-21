#!/usr/bin/env python3
"""
MASTER_TRADER — Hierarchical Neural Signal Fusion

Architecture: Sparse MoE (Mixture of Experts) — exactly like DeepSeek-R1 distilled.
Each signal layer has its own tiny expert network.
A fusion attention layer aggregates expert outputs.
Total: ~2,100 parameters. Trains online from real trade outcomes.

Philosophy:
  - Purpose-built beats brute-force at the right scale
  - 2K learned weights tuned to THIS market beat 70B generic weights
  - Every closed trade is a training signal — the model improves live
  - No PyTorch, no TensorFlow, no dependencies — pure numpy matrix math

Expert architecture per layer:
  Microstructure:  3 → 8 → 4   (OBI, kalman, contrarian flag)
  Phase Space:     4 → 8 → 4   (regime, lyapunov, transition, trajectory_len)
  Sentiment:       2 → 4 → 4   (fear_greed, news_score)
  Volume/Tech:     5 → 8 → 4   (rsi, macd_hist, trend_strength, vol_surge, bb_pct)
  Topology:        2 → 4 → 4   (persistence_norm, topology_alarm)

Fusion:
  Expert outputs:  4+4+4+4+4 = 20
  Hidden:          20 → 32 → 16
  Output:          16 → N_CONDITIONS (probability distribution)
                   16 → 1 (confidence scalar)
                   16 → 1 (position size scalar)
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

N_CONDITIONS = 14   # MarketCondition enum values
WEIGHTS_PATH = Path(__file__).parent / "weights.npz"
LEARNING_RATE = 0.003
WEIGHT_DECAY  = 0.0001    # L2 regularization — prevents overfitting on small data


# ─────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-10)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def dsigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)


# ─────────────────────────────────────────────
# DENSE LAYER
# ─────────────────────────────────────────────

class DenseLayer:
    """Single fully-connected layer with backprop support."""

    def __init__(self, n_in: int, n_out: int, activation: str = "relu"):
        # He initialization — optimal for ReLU networks
        scale = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros(n_out)
        self.activation = activation
        self._last_input: Optional[np.ndarray] = None
        self._last_output: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        z = x @ self.W + self.b
        if self.activation == "relu":
            self._last_output = relu(z)
        elif self.activation == "sigmoid":
            self._last_output = sigmoid(z)
        elif self.activation == "tanh":
            self._last_output = tanh(z)
        else:
            self._last_output = z   # linear
        return self._last_output

    def backward(self, d_out: np.ndarray, lr: float, wd: float) -> np.ndarray:
        """Backprop. Returns gradient w.r.t. input for chaining."""
        if self.activation == "relu":
            d_z = d_out * drelu(self._last_output)
        elif self.activation == "sigmoid":
            d_z = d_out * dsigmoid(self._last_output)
        elif self.activation == "tanh":
            d_z = d_out * (1 - self._last_output ** 2)
        else:
            d_z = d_out

        d_W = np.outer(self._last_input, d_z)
        d_b = d_z
        d_x = d_z @ self.W.T

        # SGD + L2 weight decay
        self.W -= lr * (d_W + wd * self.W)
        self.b -= lr * d_b
        return d_x

    def to_dict(self) -> Dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "activation": self.activation}

    def from_dict(self, d: Dict):
        self.W = np.array(d["W"])
        self.b = np.array(d["b"])
        self.activation = d["activation"]


# ─────────────────────────────────────────────
# EXPERT NETWORKS  (one per signal layer)
# ─────────────────────────────────────────────

class ExpertNetwork:
    """
    Tiny 2-layer expert for one signal domain.
    Learns what's important within its domain — zero cross-contamination.
    """

    def __init__(self, n_in: int, n_hidden: int = 8, n_out: int = 4, name: str = "expert"):
        self.name = name
        self.l1 = DenseLayer(n_in, n_hidden, "relu")
        self.l2 = DenseLayer(n_hidden, n_out, "tanh")

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.l2.forward(self.l1.forward(x))

    def backward(self, d_out: np.ndarray, lr: float, wd: float) -> np.ndarray:
        d1 = self.l2.backward(d_out, lr, wd)
        return self.l1.backward(d1, lr, wd)

    def to_dict(self) -> Dict:
        return {"l1": self.l1.to_dict(), "l2": self.l2.to_dict(), "name": self.name}

    def from_dict(self, d: Dict):
        self.l1.from_dict(d["l1"])
        self.l2.from_dict(d["l2"])


# ─────────────────────────────────────────────
# FUSION NETWORK  (aggregates all experts)
# ─────────────────────────────────────────────

class FusionNetwork:
    """
    Aggregates all expert outputs into a final trading decision.
    3-layer with attention-like weighting on expert outputs.
    """

    def __init__(self, n_experts_out: int = 20):
        self.gate   = DenseLayer(n_experts_out, 5, "sigmoid")   # expert weighting gate
        self.hidden = DenseLayer(n_experts_out, 32, "relu")
        self.l2     = DenseLayer(32, 16, "relu")
        self.cond_head = DenseLayer(16, N_CONDITIONS, "linear") # condition probs
        self.conf_head = DenseLayer(16, 1, "sigmoid")           # confidence scalar
        self.size_head = DenseLayer(16, 1, "sigmoid")           # position size modifier

        # Intermediate states for backprop
        self._gate_out: Optional[np.ndarray] = None
        self._h1: Optional[np.ndarray] = None
        self._h2: Optional[np.ndarray] = None

    def forward(self, expert_concat: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Returns (condition_probs, confidence, size_multiplier)
        """
        # Gated weighting — expert attention
        gate   = self.gate.forward(expert_concat)
        # Tile gate to match expert concat shape (simplified attention)
        # Each of 5 expert slots gets its own gate weight
        gate_tiled = np.repeat(gate, 4)   # 5 gates × 4 outputs each = 20
        gated_input = expert_concat * gate_tiled

        self._gate_out = gated_input
        h1 = self.hidden.forward(gated_input)
        h2 = self.l2.forward(h1)
        self._h1, self._h2 = h1, h2

        cond_logits = self.cond_head.forward(h2)
        cond_probs  = softmax(cond_logits)
        confidence  = float(self.conf_head.forward(h2)[0])
        size_mult   = float(self.size_head.forward(h2)[0])

        return cond_probs, confidence, size_mult

    def backward(self, d_cond: np.ndarray, d_conf: float, lr: float, wd: float):
        """Backprop through fusion network."""
        d_h2  = self.cond_head.backward(d_cond, lr, wd)
        d_h2 += self.conf_head.backward(np.array([d_conf]), lr, wd)
        d_h1  = self.l2.backward(d_h2, lr, wd)
        self.hidden.backward(d_h1, lr, wd)

    def to_dict(self) -> Dict:
        return {
            "gate":      self.gate.to_dict(),
            "hidden":    self.hidden.to_dict(),
            "l2":        self.l2.to_dict(),
            "cond_head": self.cond_head.to_dict(),
            "conf_head": self.conf_head.to_dict(),
            "size_head": self.size_head.to_dict(),
        }

    def from_dict(self, d: Dict):
        self.gate.from_dict(d["gate"])
        self.hidden.from_dict(d["hidden"])
        self.l2.from_dict(d["l2"])
        self.cond_head.from_dict(d["cond_head"])
        self.conf_head.from_dict(d["conf_head"])
        self.size_head.from_dict(d["size_head"])


# ─────────────────────────────────────────────
# SIGNAL FEATURE EXTRACTOR
# ─────────────────────────────────────────────

CONDITION_INDEX = {
    "bullish_breakout":         0,
    "bearish_breakdown":        1,
    "strong_uptrend":           2,
    "strong_downtrend":         3,
    "mean_reversion_oversold":  4,
    "mean_reversion_overbought":5,
    "phase_transition":         6,
    "bubble_warning":           7,
    "crash_warning":            8,
    "sideways_consolidation":   9,
    "high_volatility":          10,
    "news_driven_spike":        11,
    "contrarian_buy":           12,
    "contrarian_sell":          13,
}
INDEX_CONDITION = {v: k for k, v in CONDITION_INDEX.items()}


def extract_features(context_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the 5 expert input vectors from a MarketContext.context_data dict.
    Returns (micro_feats, phase_feats, sentiment_feats, tech_feats, topo_feats)
    """
    tech  = context_data.get("technical", {})
    phase = context_data.get("phase_space", {})
    topo  = context_data.get("topology", {})
    sent  = context_data.get("sentiment", {})
    obi   = float(context_data.get("obi", 0.0))

    # Expert 1: Microstructure (3 features)
    micro_feats = np.array([
        float(np.clip(obi, -1, 1)),
        1.0 if context_data.get("obi_label", "").startswith("extreme") else 0.0,
        float(np.clip(obi * 2, -1, 1)),   # amplified OBI signal
    ], dtype=np.float32)

    # Expert 2: Phase Space (4 features)
    regime_map = {"chaotic_expansion": 1.0, "stable_contraction": -0.5,
                  "stable": 0.0, "warming_up": 0.0, "insufficient_data": 0.0}
    phase_feats = np.array([
        regime_map.get(phase.get("regime", "stable"), 0.0),
        float(np.clip(phase.get("lyapunov", 0.0) * 10, -1, 1)),
        1.0 if phase.get("transition", False) else 0.0,
        float(np.clip(phase.get("trajectory_length", 0) / 200.0, 0, 1)),
    ], dtype=np.float32)

    # Expert 3: Sentiment (2 features)
    fg   = sent.get("fear_greed", {})
    sent_feats = np.array([
        float(np.clip(fg.get("score", 0.0), -1, 1)),
        float(np.clip(sent.get("news_score", 0.0), -1, 1)),
    ], dtype=np.float32)

    # Expert 4: Technical (5 features)
    rsi = float(tech.get("rsi", 50))
    tech_feats = np.array([
        (rsi - 50) / 50.0,                                              # RSI normalized
        float(np.clip(tech.get("macd_histogram", 0.0) * 100, -1, 1)), # MACD hist
        float(np.clip(tech.get("trend_strength", 0.0), 0, 1)),         # trend strength
        float(np.clip(tech.get("vol_surge", 1.0) - 1.0, -1, 2)),       # vol surge
        float(np.clip(tech.get("bollinger", {}).get("percent_b", 0.5) - 0.5, -0.5, 0.5)),
    ], dtype=np.float32)

    # Expert 5: Topology (2 features)
    topo_feats = np.array([
        float(np.clip(topo.get("persistence_norm", 0.0), 0, 1)),
        1.0 if topo.get("alarm") == "crash_or_bubble_warning" else 0.0,
    ], dtype=np.float32)

    return micro_feats, phase_feats, sent_feats, tech_feats, topo_feats


# ─────────────────────────────────────────────
# HIERARCHICAL NEURAL FUSION MODEL
# ─────────────────────────────────────────────

class HierarchicalFusionModel:
    """
    The neural brain replacing rule-based _determine_condition().

    Architecture:
        5 domain experts → gated fusion → condition distribution + confidence

    Training:
        Online learning — updates weights after every closed trade.
        Target: condition that led to a profitable trade gets reinforced.
        Loss: cross-entropy on condition + MSE on confidence.

    Storage:
        Weights saved to weights.npz after each training step.
        Loads automatically on startup — persists across restarts.
    """

    def __init__(self):
        self.expert_micro    = ExpertNetwork(3, 8, 4, "microstructure")
        self.expert_phase    = ExpertNetwork(4, 8, 4, "phase_space")
        self.expert_sent     = ExpertNetwork(2, 4, 4, "sentiment")
        self.expert_tech     = ExpertNetwork(5, 8, 4, "technical")
        self.expert_topo     = ExpertNetwork(2, 4, 4, "topology")
        self.fusion          = FusionNetwork(n_experts_out=20)

        self.lr = LEARNING_RATE
        self.wd = WEIGHT_DECAY
        self.training_steps = 0
        self.loss_history: List[float] = []

        self._load_weights()

    def _load_weights(self):
        if WEIGHTS_PATH.exists():
            try:
                data = np.load(WEIGHTS_PATH, allow_pickle=True)
                model_dict = json.loads(str(data["model_json"]))
                self._from_dict(model_dict)
                self.training_steps = int(data.get("steps", 0))
                print(f"[NeuralFusion] Loaded weights — {self.training_steps} training steps")
            except Exception as e:
                print(f"[NeuralFusion] Could not load weights ({e}) — starting fresh")

    def _save_weights(self):
        WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        model_json = json.dumps(self._to_dict())
        np.savez(WEIGHTS_PATH,
                 model_json=model_json,
                 steps=self.training_steps,
                 loss_history=np.array(self.loss_history[-100:]))

    def _to_dict(self) -> Dict:
        return {
            "micro":   self.expert_micro.to_dict(),
            "phase":   self.expert_phase.to_dict(),
            "sent":    self.expert_sent.to_dict(),
            "tech":    self.expert_tech.to_dict(),
            "topo":    self.expert_topo.to_dict(),
            "fusion":  self.fusion.to_dict(),
        }

    def _from_dict(self, d: Dict):
        self.expert_micro.from_dict(d["micro"])
        self.expert_phase.from_dict(d["phase"])
        self.expert_sent.from_dict(d["sent"])
        self.expert_tech.from_dict(d["tech"])
        self.expert_topo.from_dict(d["topo"])
        self.fusion.from_dict(d["fusion"])

    def forward(self, context_data: Dict) -> Tuple[str, float, float]:
        """
        Full forward pass.
        Returns (predicted_condition, confidence, size_multiplier)
        """
        micro_f, phase_f, sent_f, tech_f, topo_f = extract_features(context_data)

        # Expert passes
        e_micro = self.expert_micro.forward(micro_f)
        e_phase = self.expert_phase.forward(phase_f)
        e_sent  = self.expert_sent.forward(sent_f)
        e_tech  = self.expert_tech.forward(tech_f)
        e_topo  = self.expert_topo.forward(topo_f)

        # Concatenate expert outputs → fusion input
        expert_concat = np.concatenate([e_micro, e_phase, e_sent, e_tech, e_topo])

        # Fusion pass
        cond_probs, confidence, size_mult = self.fusion.forward(expert_concat)

        # Pick top condition
        top_idx   = int(np.argmax(cond_probs))
        condition = INDEX_CONDITION.get(top_idx, "sideways_consolidation")

        # Confidence is both the fusion output and modulated by probability mass
        prob_confidence = float(cond_probs[top_idx])
        combined_conf   = 0.5 * confidence + 0.5 * prob_confidence

        return condition, float(combined_conf), float(size_mult)

    def train_on_outcome(
        self,
        context_data: Dict,
        actual_condition: str,
        pnl_pct: float,
        won: bool
    ):
        """
        Online learning: update weights based on trade outcome.
        Called after every trade closes.

        Logic:
          - If trade won: reinforce the condition that was predicted
          - If trade lost: penalize — push away from predicted condition
          - Confidence target: win → 1.0, loss → 0.0
        """
        micro_f, phase_f, sent_f, tech_f, topo_f = extract_features(context_data)

        # Forward pass (with state stored for backprop)
        e_micro = self.expert_micro.forward(micro_f)
        e_phase = self.expert_phase.forward(phase_f)
        e_sent  = self.expert_sent.forward(sent_f)
        e_tech  = self.expert_tech.forward(tech_f)
        e_topo  = self.expert_topo.forward(topo_f)
        expert_concat = np.concatenate([e_micro, e_phase, e_sent, e_tech, e_topo])

        cond_probs, confidence, _ = self.fusion.forward(expert_concat)

        # Target: one-hot of actual condition that led to this outcome
        target_idx   = CONDITION_INDEX.get(actual_condition, 9)
        target_probs = np.zeros(N_CONDITIONS)
        target_probs[target_idx] = 1.0

        # Cross-entropy loss gradient
        d_cond = cond_probs - target_probs   # CE gradient w.r.t. logits

        # Scale gradient by outcome: wins push harder than losses
        reward = abs(pnl_pct) * (1.0 if won else -0.5)
        reward = float(np.clip(reward, -2.0, 2.0))
        d_cond *= reward

        # Confidence target
        conf_target = 1.0 if won else 0.0
        d_conf = confidence - conf_target

        # Backprop through fusion
        self.fusion.backward(d_cond, d_conf, self.lr, self.wd)

        # Backprop through experts (simplified — uniform gradient distribution)
        d_expert = np.ones(20) * 0.01 * reward
        self.expert_micro.backward(d_expert[:4], self.lr, self.wd)
        self.expert_phase.backward(d_expert[4:8], self.lr, self.wd)
        self.expert_sent.backward(d_expert[8:12], self.lr, self.wd)
        self.expert_tech.backward(d_expert[12:16], self.lr, self.wd)
        self.expert_topo.backward(d_expert[16:20], self.lr, self.wd)

        # Track loss
        ce_loss = -np.log(cond_probs[target_idx] + 1e-10)
        self.loss_history.append(float(ce_loss))
        self.training_steps += 1

        # Adaptive learning rate decay
        if self.training_steps % 50 == 0:
            self.lr = max(self.lr * 0.95, 0.0005)

        # Save weights every 10 training steps
        if self.training_steps % 10 == 0:
            self._save_weights()

        return ce_loss

    def stats(self) -> Dict:
        recent_loss = float(np.mean(self.loss_history[-20:])) if self.loss_history else 0.0
        return {
            "training_steps": self.training_steps,
            "recent_loss":    float(recent_loss),
            "learning_rate":  self.lr,
            "weights_saved":  WEIGHTS_PATH.exists(),
        }

    def param_count(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for expert in [self.expert_micro, self.expert_phase, self.expert_sent,
                       self.expert_tech, self.expert_topo]:
            for layer in [expert.l1, expert.l2]:
                total += layer.W.size + layer.b.size
        for layer in [self.fusion.gate, self.fusion.hidden, self.fusion.l2,
                      self.fusion.cond_head, self.fusion.conf_head, self.fusion.size_head]:
            total += layer.W.size + layer.b.size
        return total


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = HierarchicalFusionModel()
    print(f"\n[NeuralFusion] Hierarchical MoE Signal Fusion")
    print(f"  Total parameters: {model.param_count():,}")
    print(f"  Training steps:   {model.training_steps}")
    print(f"  Weights file:     {WEIGHTS_PATH}")
    print(f"\n  Expert breakdown:")
    print(f"    Microstructure: 3→8→4   ({3*8+8 + 8*4+4} params)")
    print(f"    Phase Space:    4→8→4   ({4*8+8 + 8*4+4} params)")
    print(f"    Sentiment:      2→4→4   ({2*4+4 + 4*4+4} params)")
    print(f"    Technical:      5→8→4   ({5*8+8 + 8*4+4} params)")
    print(f"    Topology:       2→4→4   ({2*4+4 + 4*4+4} params)")
    print(f"  Fusion: 20→gate(5)→32→16→[{N_CONDITIONS} cond | 1 conf | 1 size]")

    # Test forward pass with synthetic data
    test_context = {
        "obi": 0.72,
        "obi_label": "extreme_buy_pressure_fade_it",
        "technical": {"rsi": 71.0, "macd_histogram": 0.003, "trend_strength": 0.6,
                      "vol_surge": 1.8, "bollinger": {"percent_b": 0.85}},
        "phase_space": {"regime": "chaotic_expansion", "lyapunov": 0.08,
                        "transition": True, "trajectory_length": 150},
        "sentiment":   {"fear_greed": {"score": -0.7}, "news_score": -0.3},
        "topology":    {"persistence_norm": 0.65, "alarm": None},
    }

    cond, conf, size = model.forward(test_context)
    print(f"\n  Test prediction:")
    print(f"    Condition:   {cond}")
    print(f"    Confidence:  {conf:.2%}")
    print(f"    Size mult:   {size:.3f}")
    print(f"\n  Ready. Waiting for live trade outcomes to train on.")
