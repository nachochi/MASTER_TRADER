#!/usr/bin/env python3
"""
MASTER_TRADER — Live Config Loader
Watches config.yaml and hot-reloads whenever the file changes.
All subsystems read through this singleton — no restart required.
"""

import time
import threading
from pathlib import Path
from typing import Any, Dict

# Try PyYAML, fall back to a minimal YAML-ish parser for simple key: value
try:
    import yaml as _yaml
    def _parse(text: str) -> Dict:
        return _yaml.safe_load(text)
except ImportError:
    def _parse(text: str) -> Dict:
        """Minimal parser: handles nested key: value and # comments."""
        result: Dict = {}
        stack = [result]
        indent_levels = [0]
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith('#'):
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Pop stack to correct level
            while len(indent_levels) > 1 and indent <= indent_levels[-1]:
                indent_levels.pop()
                stack.pop()
            if ':' not in stripped:
                continue
            key, _, val = stripped.partition(':')
            key = key.strip()
            val = val.split('#')[0].strip()
            if not val:                          # nested block
                new_dict: Dict = {}
                stack[-1][key] = new_dict
                stack.append(new_dict)
                indent_levels.append(indent)
            else:
                # coerce types
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass  # keep as string
                stack[-1][key] = val
        return result


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ── Defaults (used if YAML parsing fails) ─────────────────────────────────
_DEFAULTS: Dict = {
    "mode":      {"paper": True, "symbol": "BTCUSDT",
                  "analysis_interval_sec": 10, "initial_balance": 10000.0},
    "risk":      {"stop_loss_pct": 0.003, "take_profit_pct": 0.006,
                  "max_open_trades": 5, "max_position_usd": 500.0,
                  "max_drawdown_pct": 0.08, "min_confidence": 0.55,
                  "kelly_max_fraction": 0.25, "kelly_min_fraction": 0.01},
    "signals":   {"microstructure": True, "phase_space": True,
                  "sentiment": True, "topology": True,
                  "geometry": True, "neural_fusion": True},
    "neural":    {"learning_rate": 0.001, "online_training": True,
                  "save_every_n_steps": 50},
    "dashboard": {"port": 8765, "broadcast_every_n_cycles": 1,
                  "phase_minimap": True, "radar_chart": True},
    "substrate": {"enabled": True, "log_consensus": True},
    "sentiment": {"fear_greed_ttl_sec": 3600, "news_ttl_sec": 900},
}


class LiveConfig:
    """
    Singleton that hot-reloads config.yaml every 10 seconds.
    Access via `live_cfg.get("risk", "stop_loss_pct")` or dict-style.
    """

    def __init__(self) -> None:
        self._data: Dict = {}
        self._mtime: float = 0.0
        self._lock = threading.Lock()
        self._load()
        self._start_watcher()

    def _load(self) -> None:
        try:
            text = CONFIG_PATH.read_text()
            parsed = _parse(text)
            if parsed:
                with self._lock:
                    self._data = parsed
                    self._mtime = CONFIG_PATH.stat().st_mtime
        except Exception as e:
            if not self._data:          # first load failure → use defaults
                with self._lock:
                    self._data = dict(_DEFAULTS)
            print(f"[LiveConfig] Parse error ({e}) — keeping previous values")

    def _watcher(self) -> None:
        while True:
            time.sleep(10)
            try:
                mtime = CONFIG_PATH.stat().st_mtime
                if mtime != self._mtime:
                    self._load()
                    print(f"[LiveConfig] ♻  config.yaml reloaded")
            except Exception:
                pass

    def _start_watcher(self) -> None:
        t = threading.Thread(target=self._watcher, daemon=True)
        t.start()

    # ── Access helpers ────────────────────────────────────────────────────
    def section(self, sec: str) -> Dict:
        with self._lock:
            return dict(self._data.get(sec, _DEFAULTS.get(sec, {})))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    def __getitem__(self, section: str) -> Dict:
        return self.section(section)

    # ── Convenience properties ─────────────────────────────────────────────
    @property
    def stop_loss_pct(self) -> float:
        return float(self.get("risk", "stop_loss_pct", 0.003))

    @property
    def take_profit_pct(self) -> float:
        return float(self.get("risk", "take_profit_pct", 0.006))

    @property
    def min_confidence(self) -> float:
        return float(self.get("risk", "min_confidence", 0.55))

    @property
    def max_open_trades(self) -> int:
        return int(self.get("risk", "max_open_trades", 5))

    @property
    def initial_balance(self) -> float:
        return float(self.get("mode", "initial_balance", 10000.0))

    @property
    def analysis_interval(self) -> int:
        return int(self.get("mode", "analysis_interval_sec", 10))

    @property
    def symbol(self) -> str:
        return str(self.get("mode", "symbol", "BTCUSDT"))

    @property
    def layers_enabled(self) -> Dict[str, bool]:
        return {k: bool(v) for k, v in self.section("signals").items()}

    @property
    def dashboard_port(self) -> int:
        return int(self.get("dashboard", "port", 8765))

    def print_summary(self) -> None:
        r = self.section("risk")
        m = self.section("mode")
        s = self.section("signals")
        print(f"  config.yaml  stop={r.get('stop_loss_pct',.003):.1%}  "
              f"tp={r.get('take_profit_pct',.006):.1%}  "
              f"conf≥{r.get('min_confidence',.55):.0%}  "
              f"layers={[k for k,v in s.items() if v]}")


# ── Singleton ─────────────────────────────────────────────────────────────
live_cfg = LiveConfig()
