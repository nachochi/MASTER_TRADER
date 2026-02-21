"""
ORACLE MIND — Self-building AI layer for MASTER_TRADER.

Architecture:
  SelfBuilder   — reads performance metrics, proposes + applies code improvements
  SentimentMind — Grok API for real-time Twitter/X crypto sentiment
  ModelRouter   — OpenRouter meta-layer: picks cheapest/fastest model per task
"""
from .self_builder import SelfBuilder
from .sentiment_mind import SentimentMind
from .model_router import ModelRouter, MODELS
