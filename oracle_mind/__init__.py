"""
ORACLE MIND — Self-building AI layer for MASTER_TRADER.

Architecture:
  SelfBuilder   — reads performance metrics, proposes + applies code improvements
  SentimentMind — Grok API for real-time Twitter/X crypto sentiment
  ModelRouter   — OpenRouter meta-layer: picks cheapest/fastest model per task
  negative_compute — inversion-first response (cast into mold)
"""
try:
    from .self_builder import SelfBuilder
except ImportError:
    SelfBuilder = None
try:
    from .sentiment_mind import SentimentMind
except ImportError:
    SentimentMind = None
from .model_router import ModelRouter, MODELS
from . import negative_compute
