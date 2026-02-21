#!/usr/bin/env python3
"""
ORACLE — Universal Geometric Substrate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The core insight: shapes encode time/data.

Like a toddler's shape sorter — the circle doesn't CALCULATE if it fits,
it just DOES. O(1) analog computation. No algebra at query time.

DNA does this: adenine is shaped so only thymine slots in. The molecular
geometry IS the rule. That's why biology is compute-efficient.

Takens' theorem (1981) proves this for dynamical systems:
  The geometry of the delay-embedded attractor contains ALL information
  about the underlying system — including hidden variables.
  The shape IS the knowledge.

This module is the universal encoder/matcher:
  1. Any signal stream → phasor (complex plane encoding)
  2. Phasors → attractor shape (delay embedding)
  3. Shape → cached template library (the "shape sorter holes")
  4. New shape matches template → O(1) pattern recognition

Works for: price, planetary angles, any periodic/quasi-periodic signal.
Same code. Same geometry. Different domains plugged in.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


# ─────────────────────────────────────────────────────────────────────────────
# Core Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometricObject:
    """
    A cached shape — the 'hole in the shape sorter'.
    Encodes a pattern as Fourier descriptors of its attractor.

    Fourier descriptors = the shape's frequency fingerprint.
    Two shapes are similar if their frequency fingerprints are close.
    This makes comparison O(n_descriptors) ≈ O(1) for fixed n.
    """
    name:         str
    domain:       str              # "market", "astrology", "neural", "custom"
    descriptors:  np.ndarray      # Fourier descriptors of the attractor boundary
    centroid:     complex          # center of mass in complex plane
    scale:        float            # characteristic size (for normalization)
    phase_offset: float            # rotation invariance offset
    metadata:     Dict = field(default_factory=dict)

    def distance_to(self, other: "GeometricObject") -> float:
        """
        Geometric distance between two shapes.
        Rotation-invariant, scale-invariant, translation-invariant.
        Like checking if the shape fits the hole — regardless of orientation.
        """
        n = min(len(self.descriptors), len(other.descriptors))
        if n == 0:
            return 1.0
        a = self.descriptors[:n] / (np.linalg.norm(self.descriptors[:n]) + 1e-9)
        b = other.descriptors[:n] / (np.linalg.norm(other.descriptors[:n]) + 1e-9)
        # Phase-align (rotation invariance) then compare magnitudes
        mag_a = np.abs(a)
        mag_b = np.abs(b)
        return float(np.mean((mag_a - mag_b) ** 2))

    def fits(self, other: "GeometricObject", threshold: float = 0.08) -> bool:
        """Does this shape fit that hole? O(1) analog computation."""
        return self.distance_to(other) < threshold


# ─────────────────────────────────────────────────────────────────────────────
# Universal Signal Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GeometricEncoder:
    """
    Converts any signal stream into a geometric object (attractor shape).

    Signal → phasor field → delay embedding → boundary → Fourier descriptors

    The phasor step is the key bridge:
      - A scalar value maps to a point on the complex unit circle
      - Angle = the value's position in its own range
      - Magnitude = normalized intensity
    This is the same as how sinusoidal signals naturally live in the complex plane.
    It's why Fourier analysis works — everything IS rotation in phase space.
    """

    def __init__(self, tau: int = 5, embedding_dim: int = 3,
                 n_descriptors: int = 32) -> None:
        self.tau           = tau            # delay parameter (Takens)
        self.embedding_dim = embedding_dim  # phase space dimension
        self.n_descriptors = n_descriptors  # number of Fourier descriptors to keep

    def encode(self, signal: np.ndarray, name: str = "unnamed",
               domain: str = "generic") -> Optional[GeometricObject]:
        """
        Main entry: signal array → GeometricObject.
        Works for any domain — price, angle, intensity, anything.
        """
        if len(signal) < self.tau * self.embedding_dim + 4:
            return None

        # Step 1: normalize to [0, 2π] — map the signal to angles on the unit circle
        sig_norm = self._normalize_to_circle(signal)

        # Step 2: convert to phasor field — each sample becomes a complex number
        phasors = np.exp(1j * sig_norm)                      # e^(iθ) — Euler's formula

        # Step 3: delay embedding (Takens) — unfold the attractor
        embedded = self._delay_embed(np.real(phasors))       # 2D projection: Re(phasor)

        # Step 4: extract the attractor boundary as a closed curve
        boundary = self._attractor_boundary(embedded)

        if len(boundary) < 8:
            return None

        # Step 5: Fourier descriptors of the boundary shape
        #   These are the "shape fingerprint" — rotation/scale/translation invariant
        descriptors, centroid, scale = self._fourier_descriptors(boundary)

        return GeometricObject(
            name=name,
            domain=domain,
            descriptors=descriptors,
            centroid=centroid,
            scale=scale,
            phase_offset=0.0,
            metadata={"signal_len": len(signal), "tau": self.tau},
        )

    # ── Signal encoding primitives ──────────────────────────────────────────

    def _normalize_to_circle(self, sig: np.ndarray) -> np.ndarray:
        """Map any real signal to [0, 2π] — onto the unit circle."""
        mn, mx = sig.min(), sig.max()
        if mx == mn:
            return np.zeros_like(sig)
        return ((sig - mn) / (mx - mn)) * 2 * np.pi

    def _delay_embed(self, sig: np.ndarray) -> np.ndarray:
        """
        Takens delay embedding: build phase space vectors.
        Each row = [x(t), x(t-τ), x(t-2τ), ...]
        Returns shape (N, embedding_dim).
        """
        n = len(sig) - (self.embedding_dim - 1) * self.tau
        if n <= 0:
            return np.zeros((1, 2))
        rows = np.stack(
            [sig[i * self.tau: i * self.tau + n] for i in range(self.embedding_dim)],
            axis=1
        )
        return rows  # shape: (N, embedding_dim)

    def _attractor_boundary(self, embedded: np.ndarray) -> np.ndarray:
        """
        Extract the 2D boundary of the attractor.
        Uses the first two dimensions (PCA-like projection via first 2 coords).
        Returns complex array representing the boundary curve.
        """
        x = embedded[:, 0]
        y = embedded[:, 1] if embedded.shape[1] > 1 else np.zeros_like(x)

        # Sort points by angle around centroid → closed boundary curve
        cx, cy = x.mean(), y.mean()
        angles = np.arctan2(y - cy, x - cx)
        order  = np.argsort(angles)

        # Subsample to fixed number of boundary points
        n_pts  = min(256, len(order))
        idx    = np.linspace(0, len(order) - 1, n_pts, dtype=int)
        ox     = x[order[idx]] - cx
        oy     = y[order[idx]] - cy

        return ox + 1j * oy    # complex representation of the boundary

    def _fourier_descriptors(
        self, boundary: np.ndarray
    ) -> Tuple[np.ndarray, complex, float]:
        """
        Compute Fourier descriptors of the boundary curve.
        These are invariant to: rotation, scale, translation, starting point.
        They're the 'shape fingerprint'.

        The first descriptor encodes the overall scale.
        Higher descriptors encode finer shape details.
        """
        fft    = np.fft.fft(boundary)
        centroid = fft[0] / len(boundary)        # DC component = centroid

        # Normalize: skip DC, normalize by magnitude of first harmonic
        desc   = fft[1: self.n_descriptors + 1]
        scale  = np.abs(desc[0]) + 1e-9
        desc   = desc / scale                    # scale invariant
        desc   = desc / np.exp(1j * np.angle(desc[0]))  # rotation invariant

        return desc, centroid, float(scale)


# ─────────────────────────────────────────────────────────────────────────────
# Shape Library (the "shape sorter" — the holes)
# ─────────────────────────────────────────────────────────────────────────────

class ShapeLibrary:
    """
    Persistent cache of named geometric patterns.
    Works across ALL domains — market patterns, planetary configurations, anything.

    This is the 'shape sorter'. You add shapes (the holes), then check
    if any incoming shape fits. Pattern recognition is now geometric comparison.

    The library persists to disk as JSON so patterns accumulate over time.
    The bot literally learns new shape templates from its own experience.
    """

    LIBRARY_PATH = Path(__file__).parent.parent / "memory" / "shape_library.json"

    def __init__(self, encoder: Optional[GeometricEncoder] = None) -> None:
        self.encoder  = encoder or GeometricEncoder()
        self.shapes:  Dict[str, GeometricObject] = {}
        self._load()

        # ── Pre-load canonical market shapes ──────────────────────────────
        self._seed_market_shapes()

    # ── Core operations ───────────────────────────────────────────────────

    def add(self, obj: GeometricObject) -> None:
        """Add a shape template to the library."""
        self.shapes[f"{obj.domain}/{obj.name}"] = obj
        self._save()

    def learn_from_signal(self, signal: np.ndarray, name: str,
                          domain: str = "market") -> Optional[GeometricObject]:
        """Encode a signal and add it to the library."""
        obj = self.encoder.encode(signal, name, domain)
        if obj:
            self.add(obj)
        return obj

    def match(
        self,
        query: GeometricObject,
        domain_filter: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[str, float, GeometricObject]]:
        """
        Find the closest shapes to the query.
        Returns list of (name, distance, shape) sorted by closeness.

        This is the 'does the shape fit the hole?' check.
        O(n_library × n_descriptors) ≈ O(n_library) for fixed descriptor size.
        """
        candidates = [
            (key, shape)
            for key, shape in self.shapes.items()
            if domain_filter is None or shape.domain == domain_filter
        ]
        if not candidates:
            return []

        results = [
            (key, query.distance_to(shape), shape)
            for key, shape in candidates
        ]
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def best_match(
        self,
        signal: np.ndarray,
        domain_filter: Optional[str] = None,
        threshold: float = 0.12,
    ) -> Optional[Tuple[str, float]]:
        """
        Encode a signal and return the best matching template name + distance.
        Returns None if no shape fits within threshold.
        """
        obj = self.encoder.encode(signal)
        if obj is None:
            return None
        matches = self.match(obj, domain_filter=domain_filter, top_k=1)
        if not matches:
            return None
        name, dist, _ = matches[0]
        if dist > threshold:
            return None
        return (name, dist)

    # ── Canonical shape seeds ──────────────────────────────────────────────

    def _seed_market_shapes(self) -> None:
        """
        Pre-build idealized market pattern shapes.
        These are mathematical archetypes — the 'holes in the sorter'.
        Real price data gets compared to these templates.
        """
        if len(self.shapes) > 5:
            return   # already seeded

        enc = self.encoder
        t   = np.linspace(0, 4 * np.pi, 200)

        # Bullish spiral — expanding helix, rising price
        bullish = np.sin(t) * (1 + t / (4 * np.pi)) + np.linspace(0, 1, 200)
        self._add_canonical(bullish, "bullish_spiral", "market")

        # Bearish spiral — contracting, falling
        bearish = np.sin(t) * (1 + t / (4 * np.pi)) + np.linspace(1, 0, 200)
        self._add_canonical(bearish, "bearish_spiral", "market")

        # Chaotic expansion — Lorenz-like divergence
        chaotic = np.sin(t) * np.exp(t / (8 * np.pi))
        self._add_canonical(chaotic, "chaotic_expansion", "market")

        # Stable attractor — limit cycle, sideways
        stable = np.sin(t) + 0.1 * np.sin(3 * t)
        self._add_canonical(stable, "stable_cycle", "market")

        # Mean reversion — damped oscillation returning to center
        mean_rev = np.sin(t) * np.exp(-t / (4 * np.pi))
        self._add_canonical(mean_rev, "mean_reversion", "market")

        # Breakout — sudden departure from stable attractor
        breakout = np.concatenate([
            np.sin(t[:100]),
            np.sin(t[100:]) + np.linspace(0, 2, 100)
        ])
        self._add_canonical(breakout, "breakout_up", "market")

        # ── Planetary archetypes (logarithmic orbital geometry) ───────────
        # Conjunction: two planets at same angle → constructive interference
        conj = np.cos(t) + np.cos(t + 0.1)
        self._add_canonical(conj, "conjunction", "astrology")

        # Opposition: planets 180° apart → destructive interference
        opp = np.cos(t) + np.cos(t + np.pi)
        self._add_canonical(opp, "opposition", "astrology")

        # Trine: 120° — harmonic resonance (3rd harmonic)
        trine = np.cos(t) + np.cos(t + 2 * np.pi / 3)
        self._add_canonical(trine, "trine", "astrology")

        # Square: 90° — tension (4th harmonic)
        square = np.cos(t) + np.cos(t + np.pi / 2)
        self._add_canonical(square, "square_tension", "astrology")

    def _add_canonical(self, signal: np.ndarray, name: str, domain: str) -> None:
        key = f"{domain}/{name}"
        if key not in self.shapes:
            obj = self.encoder.encode(signal, name, domain)
            if obj:
                self.shapes[key] = obj

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self.LIBRARY_PATH.parent.mkdir(exist_ok=True)
            data = {}
            for key, obj in self.shapes.items():
                data[key] = {
                    "name":         obj.name,
                    "domain":       obj.domain,
                    "descriptors":  [{"re": float(d.real), "im": float(d.imag)}
                                     for d in obj.descriptors],
                    "centroid":     {"re": float(obj.centroid.real),
                                     "im": float(obj.centroid.imag)},
                    "scale":        float(obj.scale),
                    "phase_offset": float(obj.phase_offset),
                    "metadata":     obj.metadata,
                }
            self.LIBRARY_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[ShapeLibrary] Save error: {e}")

    def _load(self) -> None:
        try:
            if not self.LIBRARY_PATH.exists():
                return
            data = json.loads(self.LIBRARY_PATH.read_text())
            for key, d in data.items():
                desc = np.array(
                    [x["re"] + 1j * x["im"] for x in d["descriptors"]]
                )
                cent = d["centroid"]["re"] + 1j * d["centroid"]["im"]
                self.shapes[key] = GeometricObject(
                    name=d["name"], domain=d["domain"],
                    descriptors=desc, centroid=cent,
                    scale=d["scale"], phase_offset=d["phase_offset"],
                    metadata=d.get("metadata", {}),
                )
            print(f"[ShapeLibrary] Loaded {len(self.shapes)} cached geometric objects")
        except Exception as e:
            print(f"[ShapeLibrary] Load error: {e}")

    def summary(self) -> str:
        by_domain: Dict[str, int] = {}
        for obj in self.shapes.values():
            by_domain[obj.domain] = by_domain.get(obj.domain, 0) + 1
        return " | ".join(f"{d}:{n}" for d, n in sorted(by_domain.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("═" * 60)
    print("  ORACLE Geometric Substrate — Self-Test")
    print("═" * 60)

    lib = ShapeLibrary()
    enc = lib.encoder

    print(f"\n  Library: {lib.summary()}")
    print(f"  Shapes:  {len(lib.shapes)} cached templates\n")

    # Simulate a bullish price series
    t       = np.linspace(0, 4 * np.pi, 300)
    price   = 67000 + 500 * np.sin(t) * (1 + t / (4 * np.pi)) + np.linspace(0, 1200, 300)
    noise   = np.random.randn(300) * 80
    price  += noise

    t0      = time.perf_counter()
    obj     = enc.encode(price, "live_btc", "market")
    enc_ms  = (time.perf_counter() - t0) * 1000

    t0      = time.perf_counter()
    matches = lib.match(obj, domain_filter="market", top_k=3)
    match_ms= (time.perf_counter() - t0) * 1000

    print(f"  Encode time:  {enc_ms:.2f}ms")
    print(f"  Match time:   {match_ms:.3f}ms  ← O(1) analog computation")
    print(f"\n  Top market pattern matches:")
    for name, dist, shape in matches:
        fit = "✓ FITS" if dist < 0.12 else "  distant"
        print(f"    {fit}  {name:<35}  distance={dist:.4f}")

    # Test astrology: conjunction vs opposition
    t2      = np.linspace(0, 4 * np.pi, 200)
    conj_live = np.cos(t2) + np.cos(t2 + 0.05)   # nearly conjunct
    obj2      = enc.encode(conj_live, "mars_sun", "astrology")
    matches2  = lib.match(obj2, domain_filter="astrology", top_k=3)
    print(f"\n  Planetary pattern (near-conjunction):")
    for name, dist, shape in matches2:
        fit = "✓ FITS" if dist < 0.12 else "  distant"
        print(f"    {fit}  {name:<35}  distance={dist:.4f}")

    print("\n  Same encoder. Same library. Different domain plugged in.")
    print("  This is the universal geometric substrate.\n")
