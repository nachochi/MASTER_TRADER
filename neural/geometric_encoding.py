#!/usr/bin/env python3
"""
GEOMETRIC SIGNAL ENCODING
─────────────────────────
Vision: every signal encoded as a geometric object (angle + magnitude).
Angles encode phase relationships. Shapes encode frequency content.
Time and space are the same manifold — temporal geometry IS spatial geometry.

Three layers of geometric encoding:

1. COMPLEX PHASOR ENCODING
   Each scalar signal → r·e^(iθ) on the unit circle
   Preserves: magnitude (r) + phase relationship (θ)
   Two in-phase signals → small angle → constructive interference
   Two anti-phase signals → 180° → destructive cancellation
   The neural network sees GEOMETRY not just numbers

2. WAVELET DECOMPOSITION
   Price time series → multi-scale frequency content
   At each scale τ: "what frequency is dominant right now?"
   This IS temporal geometry — the frequency spectrum of the price manifold
   Scales: 4, 8, 16, 32, 64 cycles (40s, 80s, 2.5min, 5min, 10min)
   Each scale = one octave of market temporal geometry

3. SPECTRAL ATTRACTOR ANALYSIS
   Phase space trajectory → frequency decomposition of the attractor shape
   The shape's Fourier descriptors = the market's hidden oscillation modes
   A regime change = a shift in dominant spectral mode
   A crash = emergence of a new low-frequency mode (the "ground state" collapses)

All three encode the same truth from different angles (literally):
  - Phasors: inter-signal phase relationships
  - Wavelets: intra-price temporal frequencies
  - Spectral attractor: the geometry of the hidden state space

Combined feature vector: 48 real values from 16 original scalars
(3x expansion by encoding the geometric relationships, not losing information)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# 1. COMPLEX PHASOR ENCODING
# ─────────────────────────────────────────────

def to_phasor(value: float, scale: float = np.pi) -> Tuple[float, float]:
    """
    Map a scalar in [-1, 1] to a point on the unit circle.
    value=0   → angle=0       → (1, 0)   neutral
    value=1   → angle=π       → (-1, 0)  maximum positive
    value=-1  → angle=-π      → (-1, 0)  maximum negative (same position, opposite rotation)

    We use the full circle to encode direction AND magnitude:
      angle = value × π  (maps [-1,1] → [-π, π])
      magnitude = |value|  (how confident/strong the signal is)

    Returns (real, imag) = (magnitude × cos(angle), magnitude × sin(angle))
    """
    v = float(np.clip(value, -1, 1))
    angle = v * scale
    mag   = abs(v)
    return (mag * np.cos(angle), mag * np.sin(angle))


def encode_signals_geometric(signals: Dict[str, float]) -> np.ndarray:
    """
    Convert a dict of scalar signals to a geometric phasor vector.

    Each signal becomes (real, imag) = 2 values encoding magnitude + phase.
    The resulting vector lives on the product of unit circles — a torus.
    Neural network operates on this toroidal geometry directly.

    The angle between any two phasors = their phase relationship.
    Parallel phasors = aligned signals = high confidence.
    Orthogonal phasors = uncorrelated signals = uncertainty.
    Anti-parallel phasors = opposing signals = high tension, likely reversal.
    """
    phasors = []
    for key, val in signals.items():
        r, i = to_phasor(val)
        phasors.extend([r, i])
    return np.array(phasors, dtype=np.float32)


def phasor_coherence(phasors: np.ndarray) -> float:
    """
    Measure coherence of all signals — how aligned are they?

    High coherence = all signals pointing same direction = high conviction
    Low coherence = signals scattered around circle = market indecision

    Formula: |mean(phasors as complex numbers)| ∈ [0, 1]
    This is the Kuramoto order parameter — used in physics to measure
    synchronization of coupled oscillators. Our signals ARE coupled oscillators.
    """
    n = len(phasors) // 2
    if n == 0:
        return 0.0
    complex_phasors = phasors[:n*2:2] + 1j * phasors[1:n*2:2]
    return float(abs(np.mean(complex_phasors)))


def phase_angle_between(a: float, b: float) -> float:
    """
    The angle between two signals in phasor space.
    0° = perfectly aligned (both bullish or both bearish)
    90° = orthogonal (uncorrelated)
    180° = opposing (one bullish, one bearish)

    This is the geometric measure of signal agreement.
    """
    pa = np.array(to_phasor(a))
    pb = np.array(to_phasor(b))
    cos_angle = np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


# ─────────────────────────────────────────────
# 2. MULTI-SCALE WAVELET DECOMPOSITION
# ─────────────────────────────────────────────

def haar_wavelet_decompose(signal: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    Haar wavelet decomposition — the simplest wavelet, most efficient.

    At each level: split signal into low-frequency (average) and
    high-frequency (difference) components.

    This gives us the temporal geometry of the price:
      Level 1: 2-cycle oscillations  (~20s)   — tick noise
      Level 2: 4-cycle oscillations  (~40s)   — microstructure
      Level 3: 8-cycle oscillations  (~80s)   — short momentum
      Level 4: 16-cycle oscillations (~2.5m)  — flow regime
      Level 5: 32-cycle oscillations (~5m)    — macro structure

    Each level's energy = "how much of the price movement lives at this scale"
    A regime change = energy redistribution across scales.
    A crash = sudden energy burst at ALL scales simultaneously.

    Returns: array of (approximation, detail) energies per level
    """
    if len(signal) < 2**levels:
        levels = max(1, int(np.log2(len(signal))))

    coeff_energies = []
    s = signal.copy().astype(float)

    for _ in range(levels):
        if len(s) < 2:
            break
        # Haar: pair-wise average and difference
        n = len(s) // 2 * 2   # ensure even
        approx = (s[:n:2] + s[1:n:2]) / np.sqrt(2)
        detail = (s[:n:2] - s[1:n:2]) / np.sqrt(2)

        # Energy at this scale = RMS of detail coefficients
        energy = float(np.sqrt(np.mean(detail**2)) if len(detail) > 0 else 0.0)
        coeff_energies.append(energy)

        s = approx  # recurse on approximation

    # Final approximation energy (lowest frequency component)
    coeff_energies.append(float(np.sqrt(np.mean(s**2)) if len(s) > 0 else 0.0))

    # Normalize so energies sum to 1 (relative frequency content)
    total = sum(coeff_energies) + 1e-10
    return np.array([e / total for e in coeff_energies], dtype=np.float32)


def wavelet_dominant_scale(energies: np.ndarray) -> int:
    """Which temporal scale carries the most energy right now?"""
    return int(np.argmax(energies))


def wavelet_entropy(energies: np.ndarray) -> float:
    """
    Shannon entropy of the wavelet energy distribution.
    Low entropy = energy concentrated at one scale = clear regime
    High entropy = energy spread across all scales = chaotic/transition
    """
    e = energies + 1e-10
    return float(-np.sum(e * np.log2(e)))


# ─────────────────────────────────────────────
# 3. SPECTRAL ATTRACTOR ANALYSIS
# ─────────────────────────────────────────────

def attractor_fourier_descriptors(trajectory: np.ndarray, n_descriptors: int = 8) -> np.ndarray:
    """
    Compute Fourier descriptors of the phase space attractor shape.

    The attractor projected to 2D (first two delay dimensions) forms a curve.
    Its Fourier descriptors encode the SHAPE of that curve as frequency components.
    Each descriptor = how much of the shape "lives at" frequency k.

    Key insight: the dominant Fourier descriptor tells you the attractor's
    fundamental oscillation mode — the market's "natural frequency."

    A limit cycle (periodic market) → one dominant descriptor
    A strange attractor (chaotic) → many descriptors with similar energy
    A fixed point (flat market) → descriptor[0] dominates

    Returns: array of n_descriptors normalized energy values
    """
    if len(trajectory) < 4 or trajectory.shape[1] < 2:
        return np.zeros(n_descriptors, dtype=np.float32)

    # Project to 2D plane (first two delay dimensions)
    x = trajectory[:, 0].astype(complex)
    y = trajectory[:, 1].astype(complex)
    z = x + 1j * y   # Complex representation of 2D curve

    # FFT of the complex curve = Fourier descriptors
    fft = np.fft.fft(z)
    descriptors = np.abs(fft[:n_descriptors])

    # Normalize (make translation and scale invariant)
    if descriptors[0] > 1e-10:
        descriptors = descriptors / descriptors[0]

    # Energy normalization
    total = np.sum(descriptors**2) + 1e-10
    return (descriptors**2 / total).astype(np.float32)


def attractor_spectral_entropy(descriptors: np.ndarray) -> float:
    """
    Entropy of attractor shape's frequency content.
    Low = simple shape (limit cycle, periodic market)
    High = complex shape (strange attractor, chaotic market)
    """
    d = descriptors + 1e-10
    return float(-np.sum(d * np.log2(d)))


def dominant_oscillation_period(descriptors: np.ndarray, cycle_seconds: int = 10) -> float:
    """
    The market's dominant oscillation period in seconds.
    Derived from which Fourier descriptor has the most energy (after DC).
    """
    if len(descriptors) < 2:
        return float("inf")
    dominant_k = int(np.argmax(descriptors[1:]) + 1)   # skip DC (k=0)
    if dominant_k == 0:
        return float("inf")
    # Period = total_length / dominant_frequency × cycle_seconds
    return float(len(descriptors) / dominant_k * cycle_seconds)


# ─────────────────────────────────────────────
# UNIFIED GEOMETRIC CONTEXT VECTOR
# ─────────────────────────────────────────────

def build_geometric_context(
    context_data: Dict,
    price_history: np.ndarray,
    trajectory: Optional[np.ndarray] = None,
) -> Dict:
    """
    Build the full geometric context from market data.
    Returns a dict of geometric measures ready for neural processing.

    This is the "reading the geometry" function — it converts market state
    into the language of shapes, angles, and frequencies.
    """
    tech  = context_data.get("technical", {})
    phase = context_data.get("phase_space", {})
    sent  = context_data.get("sentiment", {})
    fg    = sent.get("fear_greed", {})
    obi   = float(context_data.get("obi", 0))

    # ── Phasor encoding of key signals ──
    signal_scalars = {
        "obi":           obi,
        "sentiment":     float(sent.get("combined_score", 0)),
        "rsi":           (float(tech.get("rsi", 50)) - 50) / 50,
        "macd":          float(np.clip(tech.get("macd_histogram", 0) * 1000, -1, 1)),
        "trend":         float(tech.get("trend_direction", 0)) * float(tech.get("trend_strength", 0)),
        "fear_greed":    (float(fg.get("value", 50)) - 50) / 50,
        "topo":          float(context_data.get("topology_alarm", 0)) * 2 - 1,
        "lyapunov":      float(np.clip(phase.get("lyapunov", 0) * 10, -1, 1)),
    }

    phasors = encode_signals_geometric(signal_scalars)
    coherence = phasor_coherence(phasors)

    # Key phase angles — geometric relationships between signals
    obi_sent_angle = phase_angle_between(obi, sent.get("combined_score", 0))
    rsi_macd_angle = phase_angle_between(
        (tech.get("rsi", 50) - 50) / 50,
        np.clip(tech.get("macd_histogram", 0) * 1000, -1, 1)
    )

    # ── Wavelet decomposition of price ──
    wavelet_energies  = haar_wavelet_decompose(price_history, levels=5)
    dominant_scale    = wavelet_dominant_scale(wavelet_energies)
    wav_entropy       = wavelet_entropy(wavelet_energies)

    # ── Spectral attractor analysis ──
    spectral_result = {}
    if trajectory is not None and len(trajectory) >= 4:
        descriptors  = attractor_fourier_descriptors(trajectory)
        spec_entropy = attractor_spectral_entropy(descriptors)
        dom_period   = dominant_oscillation_period(descriptors)
        spectral_result = {
            "fourier_descriptors": descriptors.tolist(),
            "spectral_entropy":    spec_entropy,
            "dominant_period_s":   dom_period,
            "attractor_mode":      "periodic" if spec_entropy < 1.5 else
                                   "chaotic" if spec_entropy > 2.5 else "transitional",
        }

    return {
        # Phasor geometry
        "phasors":        phasors.tolist(),
        "coherence":      coherence,
        "obi_sent_angle": obi_sent_angle,
        "rsi_macd_angle": rsi_macd_angle,

        # Interpretation
        "signal_alignment": "aligned" if coherence > 0.6
                            else "scattered" if coherence < 0.2
                            else "mixed",
        "phase_tension":    "high" if obi_sent_angle > 120 else
                            "low" if obi_sent_angle < 30 else "moderate",

        # Wavelet temporal geometry
        "wavelet_energies":  wavelet_energies.tolist(),
        "dominant_scale":    dominant_scale,
        "wavelet_entropy":   wav_entropy,
        "time_structure":    "microstructure" if dominant_scale <= 1 else
                             "momentum" if dominant_scale <= 3 else "macro",

        # Spectral attractor
        **spectral_result,
    }


# ─────────────────────────────────────────────
# ENTRYPOINT — demonstrate the encoding
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    print(f"\n{'═'*60}")
    print(f"  GEOMETRIC SIGNAL ENCODING")
    print(f"{'═'*60}")

    # ── Phasor demo ──
    print("\n[1] COMPLEX PHASOR ENCODING")
    cases = [
        ("Aligned bullish",   {"obi": 0.7, "sentiment": 0.6, "rsi": 0.5}),
        ("Aligned bearish",   {"obi": -0.7, "sentiment": -0.6, "rsi": -0.5}),
        ("Contradictory",     {"obi": 0.8, "sentiment": -0.7, "rsi": 0.3}),
        ("Neutral scattered", {"obi": 0.1, "sentiment": -0.1, "rsi": 0.05}),
    ]
    for name, signals in cases:
        p = encode_signals_geometric(signals)
        c = phasor_coherence(p)
        print(f"  {name:<22} coherence={c:.3f}  {'HIGH CONVICTION' if c > 0.6 else 'LOW CONVICTION'}")

    # ── Wavelet demo ──
    print("\n[2] WAVELET TEMPORAL GEOMETRY")
    # Simulate different market regimes
    t = np.linspace(0, 4*np.pi, 64)
    regimes = {
        "Trending (low freq dominant)":  np.cumsum(np.random.randn(64) * 0.3) + t,
        "Oscillating (mid freq)":        np.sin(t * 3) + np.random.randn(64) * 0.1,
        "Chaotic (all freqs)":           np.random.randn(64),
        "Flash crash (energy burst)":    np.concatenate([np.zeros(48), np.random.randn(16)*3]),
    }
    for name, signal in regimes.items():
        energies = haar_wavelet_decompose(signal)
        dom = wavelet_dominant_scale(energies)
        ent = wavelet_entropy(energies)
        print(f"  {name:<38} dominant_scale={dom}  entropy={ent:.2f}")

    # ── Attractor spectral demo ──
    print("\n[3] SPECTRAL ATTRACTOR ANALYSIS")
    # Simulate attractor trajectories
    # Limit cycle (periodic market)
    theta = np.linspace(0, 4*np.pi, 100)
    limit_cycle = np.column_stack([np.cos(theta), np.sin(theta)])
    # Strange attractor proxy (chaotic)
    chaotic = np.random.randn(100, 2)
    chaotic = np.cumsum(chaotic * 0.1, axis=0)

    for name, traj in [("Limit cycle (periodic)", limit_cycle),
                        ("Chaotic trajectory", chaotic)]:
        desc = attractor_fourier_descriptors(traj)
        ent  = attractor_spectral_entropy(desc)
        per  = dominant_oscillation_period(desc)
        mode = "periodic" if ent < 1.5 else "chaotic" if ent > 2.5 else "transitional"
        print(f"  {name:<28} entropy={ent:.2f}  period={per:.0f}s  mode={mode}")

    # ── Phase angle relationships ──
    print("\n[4] INTER-SIGNAL PHASE ANGLES")
    angle_cases = [
        ("OBI=+0.8, Sentiment=+0.7", 0.8,  0.7),
        ("OBI=+0.8, Sentiment=-0.7", 0.8, -0.7),
        ("OBI=+0.5, Sentiment=0.0",  0.5,  0.0),
    ]
    for desc, a, b in angle_cases:
        angle = phase_angle_between(a, b)
        interp = "IN PHASE (aligned)" if angle < 45 else \
                 "QUADRATURE (orthogonal)" if 45 <= angle < 135 else \
                 "ANTI-PHASE (opposing)"
        print(f"  {desc:<35} {angle:5.1f}°  {interp}")

    print(f"\n  The shape of the market IS its temporal structure.")
    print(f"  Geometry and time are the same manifold, different coordinates.")
