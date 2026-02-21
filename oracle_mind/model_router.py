#!/usr/bin/env python3
"""
ORACLE MIND — Model Router
One key (OpenRouter) → access to every frontier model.
Automatically picks the right model for each task type.

Get your key: https://openrouter.ai  (free tier available)
Set: OPENROUTER_API_KEY in ~/.config/ai/nexus/secrets.env

Task routing:
  code_improvement  → DeepSeek R1 (best code reasoning, cheapest)
  sentiment_read    → Grok (Twitter-trained, crypto native)
  architecture      → Claude Opus (most reliable for big decisions)
  quick_analysis    → DeepSeek V3 (fast, cheap, very capable)
  codebase_scan     → Gemini 1.5 Pro (1M context = reads everything)
"""

import os
import json
import asyncio
from typing import Any, Dict, Optional

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


# ── Model registry — update as better models release ─────────────────────────
MODELS = {
    # Task → (openrouter model id, max_tokens, approx $/1M input)
    "code_improvement": ("deepseek/deepseek-r1",          4096,  0.55),
    "quick_analysis":   ("deepseek/deepseek-chat",        2048,  0.27),
    "sentiment_read":   ("x-ai/grok-2-1212",              2048,  2.00),
    "architecture":     ("anthropic/claude-opus-4-5",     4096, 15.00),
    "codebase_scan":    ("google/gemini-pro-1.5",         8192,  1.25),
    "fast_cheap":       ("meta-llama/llama-3.3-70b-instruct:nitro", 2048, 0.20),
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ModelRouter:
    """
    Thin async wrapper around OpenRouter.
    Automatically selects the best model per task.
    Falls back gracefully if no API key is set.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            print("[ModelRouter] No OPENROUTER_API_KEY — AI self-building disabled.")
            print("[ModelRouter] Get a free key at https://openrouter.ai")

    async def ask(
        self,
        task: str,
        prompt: str,
        system: str = "You are ORACLE MIND, an expert AI that improves trading systems.",
        temperature: float = 0.3,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a prompt to the best model for this task.
        Returns the response text or None on failure.
        """
        if not self.enabled or not _HAS_AIOHTTP:
            return None

        model_id, max_tok, _ = MODELS.get(task, MODELS["quick_analysis"])
        if model_override:
            model_id = model_override

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/oracle-master-trader",
            "X-Title": "ORACLE MASTER_TRADER",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            "max_tokens": max_tok,
            "temperature": temperature,
        }

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status != 200:
                        txt = await r.text()
                        print(f"[ModelRouter] HTTP {r.status}: {txt[:120]}")
                        return None
                    data = await r.json()
                    return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            print(f"[ModelRouter] Timeout on task={task}")
            return None
        except Exception as e:
            print(f"[ModelRouter] Error: {e}")
            return None

    def cost_estimate(self, task: str, input_tokens: int, output_tokens: int) -> float:
        """Return estimated cost in USD for a call."""
        _, _, cost_per_m = MODELS.get(task, MODELS["quick_analysis"])
        return (input_tokens + output_tokens) / 1_000_000 * cost_per_m
