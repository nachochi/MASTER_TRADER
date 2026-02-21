"""
ORACLE Negative Computing — Inversion Before Generation
=========================================================
Don't multiply (generate). Divide (look up). The response already exists;
we find which cavity the query fits. Like casting: the mold is the library.

Usage:
  r = respond("what is the current market shape?")  # invert first; Ollama on miss
  invert("hello")  -> None or cached response
  store("hello", "Hi. ORACLE here.")  -> next invert("hello") returns that
"""

import numpy as np
from pathlib import Path
from typing import Optional

# Min signal length for geometric encoder (tau*embedding_dim+4)
_MIN_LEN = 5 * 3 + 4  # 19

_CACHE_PATH = Path(__file__).parent.parent / "memory" / "negative_compute_cache.json"


def _text_to_signal(text: str) -> np.ndarray:
    """Turn message into 1D signal (bytes → float). Pad/tile to min length."""
    raw = text.strip().lower().encode("utf-8")
    if not raw:
        raw = b"\x00"
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
    while len(arr) < _MIN_LEN:
        arr = np.tile(arr, 2)
    return arr[: 256]  # cap so encode is fast


# Response cavities: normalized query text -> response. (Shape-keyed cavities later.)
_response_lib: dict[str, str] = {}


def _load_cache():
    global _response_lib
    if _CACHE_PATH.exists():
        import json
        try:
            _response_lib = json.loads(_CACHE_PATH.read_text())
        except Exception:
            _response_lib = {}


def _save_cache():
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    _CACHE_PATH.write_text(json.dumps(_response_lib, indent=0))


def invert(message: str, distance_threshold: float = 0.15) -> Optional[str]:
    """
    Matrix division: find which stored response the query fits.
    Returns cached response if query (exact normalized text) is in the mold; else None.
    """
    if not message or not message.strip():
        return None
    _load_cache()
    key = message.strip().lower()
    return _response_lib.get(key)


def store(message: str, response: str) -> None:
    """Add a cavity: this query maps to this response. Next invert(message) returns response."""
    _load_cache()
    key = message.strip().lower()
    _response_lib[key] = response.strip()
    _save_cache()


def respond(message: str, ollama_if_miss: bool = True) -> str:
    """
    One entrypoint: invert first. On miss, call Ollama and store for next time.
    """
    hit = invert(message)
    if hit is not None:
        return hit
    if not ollama_if_miss:
        return ""
    try:
        import urllib.request
        import json
        url = "http://localhost:11435/v1/chat/completions"
        key = "v5Jv8iDCBtJoFUqpTdMyOJPE2ZBVJN67Tgbj0Azp27k"
        payload = {
            "model": "qwen2.5-coder:7b-instruct",
            "messages": [
                {"role": "system", "content": "You are ORACLE. Answer briefly. One or two sentences."},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "temperature": 0.3,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        response = (out.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if response:
            store(message, response)
        return response or "(no response)"
    except Exception as e:
        return f"(oracle unavailable: {e})"


# Seed a few cavities so inversion works immediately
def _seed():
    _load_cache()
    if _response_lib:
        return
    _response_lib["what is the current market shape?"] = "The market is encoded as a shape in phase space. Check the dashboard spiral — that is the current shape."
    _response_lib["who are you"] = "ORACLE. Universal pattern recognition. Same geometry across every domain."
    _response_lib["oracle"] = "I am ORACLE. The substrate. The mold. You are talking to the cavity that your query fits."
    _save_cache()


_seed()
