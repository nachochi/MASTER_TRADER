#!/usr/bin/env python3
"""
MASTER_TRADER — RPG Command Deck Dashboard v2
Interaction design inspired by:
  Kingdom Hearts (command deck + action buttons)
  Mass Effect (radial power wheel)
  Skyrim (favorites / Nordic puzzle lock nested radial)
  Dead Space (diegetic color language)
  FFXIV (per-signal job gauges)
  Pokémon (stat radar/spider chart)
  Every open-world game (minimap corner overlay)

Access: http://localhost:8765
Commands sent via POST /cmd → consumed by main.py
"""

import asyncio
import json
import sys
import math as _math
from pathlib import Path
from datetime import datetime, timezone
from typing import Set

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Command queue — browser → bot
# ─────────────────────────────────────────────────────────────────────────────
_cmd_queue: asyncio.Queue = asyncio.Queue()


async def get_command():
    """Called by main.py each cycle to consume one pending command."""
    if not _cmd_queue.empty():
        return await _cmd_queue.get()
    return None

# ─────────────────────────────────────────────────────────────────────────────
# HTML — Command Deck v2
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ORACLE // MASTER_TRADER</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #030312; --bg2: #070720; --bg3: #0a0a26;
  --border:  #1a2060;
  --cyan:    #00ccff; --cyan2: #00aadd;
  --purple:  #8844ff; --green: #00ff88;
  --amber:   #ffaa00; --red:   #ff2244;
  --gold:    #ffcc44; --dim:   #334466;
  --text:    #c8d8f0; --textdim: #445577;
}

body {
  background: var(--bg); color: var(--text);
  font-family: 'Courier New', monospace; font-size: 12px;
  height: 100vh; display: flex; flex-direction: column;
  overflow: hidden; user-select: none;
}
body::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 9000;
  background: repeating-linear-gradient(0deg,transparent,transparent 2px,
    rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px);
}

/* ── HEADER ─────────────────────────────────────────────────────────────── */
#hdr {
  display: flex; align-items: center; gap: 12px;
  padding: 6px 18px; flex-shrink: 0;
  background: linear-gradient(90deg,#050520,#09092e,#050520);
  border-bottom: 1px solid var(--border);
}
#logo { font-size:13px; font-weight:bold; letter-spacing:4px; color:var(--cyan);
  text-shadow: 0 0 16px var(--cyan),0 0 32px rgba(0,204,255,.25); }
#logo .sl { color:var(--border); }
#cdot { width:7px;height:7px;border-radius:50%;background:var(--dim);transition:all .4s;flex-shrink:0; }
#cdot.live { background:var(--green);box-shadow:0 0 8px var(--green);animation:dp 2s ease-in-out infinite; }
@keyframes dp{0%,100%{opacity:1}50%{opacity:.5}}
#ctext{color:var(--textdim);font-size:9px;letter-spacing:2px;}
#clk{margin-left:auto;color:var(--textdim);font-size:10px;letter-spacing:2px;font-variant-numeric:tabular-nums;}
#mbadge{padding:2px 10px;border-radius:2px;font-size:9px;font-weight:bold;letter-spacing:2px;
  background:rgba(255,204,68,.1);color:var(--gold);border:1px solid rgba(255,204,68,.25);}
#kb-hint{font-size:8px;color:var(--dim);letter-spacing:1px;}

/* ── MAIN GRID ───────────────────────────────────────────────────────────── */
#main {
  flex:1; display:grid;
  grid-template-columns: 175px 1fr 175px;
  grid-template-rows: 55% 45%;
  gap:1px; background:var(--border);
  overflow:hidden; min-height:0;
}

/* ── PANELS ─────────────────────────────────────────────────────────────── */
.panel {
  background:var(--bg2); padding:8px 11px;
  overflow:hidden; display:flex; flex-direction:column; gap:4px;
  position:relative;
}
.ptitle {
  font-size:8px;letter-spacing:3px;color:var(--textdim);
  text-transform:uppercase;border-bottom:1px solid var(--border);
  padding-bottom:3px;flex-shrink:0;
}

/* ── VITALS ─────────────────────────────────────────────────────────────── */
#vitals{grid-column:1;grid-row:1;}
.sb{margin-bottom:6px;}
.sl2{font-size:8px;letter-spacing:2px;color:var(--textdim);text-transform:uppercase;
  margin-bottom:2px;display:flex;justify-content:space-between;align-items:center;}
.sbadge{font-size:7px;padding:1px 5px;border-radius:2px;font-weight:bold;}
.sv{font-size:16px;font-weight:bold;font-variant-numeric:tabular-nums;line-height:1;margin-bottom:3px;}
.bw{height:7px;background:rgba(255,255,255,.04);border-radius:1px;overflow:hidden;position:relative;}
.bf{height:100%;border-radius:1px;transition:width .5s ease,background .5s ease;}
.bf::after{content:'';position:absolute;top:0;right:0;width:30%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.3));
  animation:sh 2.5s ease-in-out infinite;}
@keyframes sh{0%,100%{opacity:0}50%{opacity:1}}
.hp-hi{background:var(--green);box-shadow:0 0 7px rgba(0,255,136,.5);}
.hp-md{background:var(--amber);box-shadow:0 0 7px rgba(255,170,0,.5);}
.hp-lo{background:var(--red);  box-shadow:0 0 7px rgba(255,34,68,.5);}
.mp-f{background:var(--purple);box-shadow:0 0 6px rgba(136,68,255,.5);}
.xp-f{background:var(--cyan);  box-shadow:0 0 5px rgba(0,204,255,.4);}
.xpw{height:4px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden;}
.pnl{font-size:18px;font-weight:bold;font-variant-numeric:tabular-nums;}
.pnl.pos{color:var(--green);text-shadow:0 0 10px rgba(0,255,136,.4);}
.pnl.neg{color:var(--red);  text-shadow:0 0 10px rgba(255,34,68,.4);}
.pnl.neu{color:var(--textdim);}

/* ── PRICE CHART PANEL ───────────────────────────────────────────────────── */
#chart-panel{grid-column:2;grid-row:1;position:relative;}
#ph{display:flex;align-items:baseline;gap:10px;flex-shrink:0;}
#pbig{font-size:24px;font-weight:bold;color:var(--cyan);
  text-shadow:0 0 16px rgba(0,204,255,.5);font-variant-numeric:tabular-nums;}
#pdelta{font-size:12px;font-variant-numeric:tabular-nums;}
#pdelta.pos{color:var(--green);}#pdelta.neg{color:var(--red);}#pdelta.neu{color:var(--textdim);}

/* Minimap phase overlay — bottom-right corner of chart like every open world game */
#phase-mini{
  position:absolute; bottom:4px; right:4px;
  width:110px; height:90px;
  background:rgba(7,7,32,.88);
  border:1px solid rgba(136,68,255,.4);
  border-radius:3px; overflow:hidden;
}
#phase-mini-label{
  position:absolute;top:2px;left:4px;
  font-size:7px;letter-spacing:1px;color:rgba(136,68,255,.7);
  text-transform:uppercase; pointer-events:none;
}

canvas{display:block;flex:1;min-height:0;}

/* ── SIGNAL RADAR PANEL ──────────────────────────────────────────────────── */
#radar-panel{grid-column:2;grid-row:2;}
#radar-info{display:flex;gap:12px;font-size:8px;color:var(--textdim);flex-shrink:0;letter-spacing:1px;}
#radar-info span{color:var(--cyan);}

/* ── SIGNAL TOWER ────────────────────────────────────────────────────────── */
#signals{grid-column:3;grid-row:1/3;overflow-y:auto;}
#signals::-webkit-scrollbar{width:3px;}
#signals::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.gb{margin-bottom:8px;flex-shrink:0;}
.gn{font-size:7px;letter-spacing:2px;color:var(--textdim);text-transform:uppercase;
  margin-bottom:3px;display:flex;justify-content:space-between;}
.gr{font-size:10px;font-variant-numeric:tabular-nums;}

/* OBI split bar */
.ob{height:9px;background:rgba(255,255,255,.04);border-radius:1px;position:relative;overflow:hidden;}
.ob-b{position:absolute;left:50%;top:0;height:100%;background:var(--green);
  box-shadow:0 0 5px var(--green);transition:width .4s;}
.ob-s{position:absolute;right:50%;top:0;height:100%;background:var(--red);
  box-shadow:0 0 5px var(--red);transition:width .4s;}
.ob-c{position:absolute;left:50%;top:0;width:1px;height:100%;
  background:rgba(255,255,255,.25);transform:translateX(-50%);}

/* Pip row */
.pr{display:flex;gap:2px;}
.pp{flex:1;height:8px;border-radius:1px;background:rgba(255,255,255,.05);
  transition:background .25s,box-shadow .25s;}

/* Arc bar */
.ab{height:6px;background:rgba(255,255,255,.05);border-radius:6px;overflow:hidden;}
.af{height:100%;border-radius:6px;transition:width .5s,background .4s;}

/* Topology */
.tr2{display:flex;align-items:center;gap:5px;}
.td{width:8px;height:8px;border-radius:50%;background:var(--dim);
  flex-shrink:0;transition:all .3s;}
.td.alarmed{background:var(--red);box-shadow:0 0 10px var(--red);
  animation:tb .6s ease-in-out infinite;}
@keyframes tb{0%,100%{transform:scale(1)}50%{transform:scale(1.5);opacity:.6}}
.tbw{flex:1;height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden;}
.tbf{height:100%;background:var(--red);border-radius:3px;transition:width .4s;}

.sdiv{border:none;border-top:1px solid var(--border);margin:6px 0;}
.sr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px;}
.sl3{font-size:7px;color:var(--textdim);letter-spacing:1px;text-transform:uppercase;}
.sv2{font-size:10px;font-variant-numeric:tabular-nums;}

/* ── AGENTS ──────────────────────────────────────────────────────────────── */
#agents{grid-column:1;grid-row:2;overflow-y:auto;}
.ac{background:rgba(255,255,255,.02);border:1px solid var(--border);
  border-radius:3px;padding:5px 7px;margin-bottom:4px;}
.ac.long{border-left:2px solid var(--green);}
.ac.short{border-left:2px solid var(--red);}
.at2{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;}
.an{font-size:9px;color:var(--text);}
.ad{font-size:8px;font-weight:bold;}
.ad.long{color:var(--green);}
.ad.short{color:var(--red);}
.am{display:flex;justify-content:space-between;font-size:9px;margin-bottom:3px;}
.ahb{height:3px;background:rgba(255,255,255,.05);border-radius:2px;overflow:hidden;}
.ahf{height:100%;border-radius:2px;transition:width .4s;}
.nop{color:var(--dim);font-size:9px;letter-spacing:2px;text-align:center;padding:20px 0;text-transform:uppercase;}

/* ── COMMAND DECK ────────────────────────────────────────────────────────── */
#deck {
  flex-shrink:0;
  background:linear-gradient(0deg,#040418,#07072a);
  border-top:1px solid var(--border);
  padding:8px 18px;
  display:grid;
  grid-template-columns: auto 1fr auto auto auto auto auto;
  gap:14px; align-items:center;
  position:relative;
}
#deck::before{content:'';position:absolute;top:0;left:8%;right:8%;height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.3;}

/* KH-style action buttons */
.kh-btn {
  display:flex; align-items:center; gap:5px;
  padding:5px 12px; border-radius:3px; border:1px solid;
  font-family:'Courier New',monospace; font-size:10px; font-weight:bold;
  letter-spacing:2px; cursor:pointer; text-transform:uppercase;
  transition:all .2s; background:transparent; color:inherit;
}
.kh-btn:hover { transform:translateY(-1px); }
.kh-btn:active { transform:translateY(1px); }
.kh-btn.long-btn {
  color:var(--green); border-color:rgba(0,255,136,.35);
  background:rgba(0,255,136,.06);
}
.kh-btn.long-btn:hover{background:rgba(0,255,136,.15);box-shadow:0 0 12px rgba(0,255,136,.3);}
.kh-btn.short-btn {
  color:var(--red); border-color:rgba(255,34,68,.35);
  background:rgba(255,34,68,.06);
}
.kh-btn.short-btn:hover{background:rgba(255,34,68,.15);box-shadow:0 0 12px rgba(255,34,68,.3);}
.kh-btn.menu-btn {
  color:var(--cyan); border-color:rgba(0,204,255,.35);
  background:rgba(0,204,255,.06);
}
.kh-btn.menu-btn:hover{background:rgba(0,204,255,.15);box-shadow:0 0 12px rgba(0,204,255,.3);}
.kh-btn.close-btn {
  color:var(--amber); border-color:rgba(255,170,0,.35);
  background:rgba(255,170,0,.06);
}
.kh-btn.close-btn:hover{background:rgba(255,170,0,.15);box-shadow:0 0 12px rgba(255,170,0,.3);}

/* Condition display */
#cond-area{display:flex;flex-direction:column;gap:3px;min-width:220px;}
#cond-lbl{font-size:7px;letter-spacing:3px;color:var(--textdim);text-transform:uppercase;}
#cond-name{font-size:14px;font-weight:bold;letter-spacing:3px;
  text-transform:uppercase;transition:color .4s,text-shadow .4s;}
#cond-bw{height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;}
#cond-bf{height:100%;border-radius:2px;transition:width .6s,background .4s;}

.ds{display:flex;flex-direction:column;align-items:center;gap:1px;}
.dsl{font-size:7px;letter-spacing:2px;color:var(--textdim);text-transform:uppercase;white-space:nowrap;}
.dsv{font-size:13px;font-weight:bold;font-variant-numeric:tabular-nums;white-space:nowrap;}
.dsv.pos{color:var(--green);}
.dsv.neg{color:var(--red);}

/* ── RADIAL MENU OVERLAY ──────────────────────────────────────────────────── */
#radial-overlay {
  position:fixed;inset:0;z-index:1000;
  display:none;align-items:center;justify-content:center;
  background:rgba(3,3,18,.75);
  backdrop-filter:blur(4px);
}
#radial-overlay.open{display:flex;}

#radial-svg { overflow:visible; }

.radial-sector {
  cursor:pointer;
  transition:filter .15s;
}
.radial-sector:hover { filter: brightness(1.6); }
.radial-sector.selected { filter: brightness(2); }

#radial-label-box {
  position:absolute;
  bottom:calc(50% - 220px);
  left:50%; transform:translateX(-50%);
  text-align:center;
  font-size:11px;letter-spacing:3px;color:var(--text);
  text-transform:uppercase;
  pointer-events:none;
}
#radial-hint{font-size:8px;color:var(--textdim);letter-spacing:2px;margin-top:3px;}

/* Sub-menu ring items */
.sub-sector { cursor:pointer; }
.sub-sector:hover path { filter:brightness(1.8); }

/* Key shortcut badges on sectors */
.key-badge { font-size:9px; fill:rgba(200,216,240,.6); font-family:'Courier New'; }

/* ── NOTIFICATION TOAST ───────────────────────────────────────────────────── */
#toast {
  position:fixed;top:60px;right:20px;z-index:2000;
  padding:8px 16px;border-radius:3px;font-size:11px;
  letter-spacing:2px;text-transform:uppercase;
  background:rgba(0,204,255,.15);border:1px solid rgba(0,204,255,.4);
  color:var(--cyan);
  opacity:0;transition:opacity .3s;
  pointer-events:none;
}
#toast.show{opacity:1;}

/* ── KEYBOARD SHORTCUT OVERLAY ────────────────────────────────────────────── */
#kb-overlay{
  position:fixed;inset:0;z-index:999;
  background:rgba(3,3,18,.9);
  display:none;align-items:center;justify-content:center;
  backdrop-filter:blur(6px);
}
#kb-overlay.open{display:flex;}
#kb-box{
  border:1px solid var(--border);background:var(--bg3);
  padding:24px 32px;border-radius:4px;min-width:320px;
}
#kb-box h2{font-size:11px;letter-spacing:4px;color:var(--cyan);
  text-transform:uppercase;margin-bottom:16px;
  text-shadow:0 0 12px var(--cyan);}
.kb-row{display:flex;justify-content:space-between;gap:24px;
  margin-bottom:8px;font-size:10px;}
.kb-key{color:var(--gold);letter-spacing:2px;white-space:nowrap;}
.kb-desc{color:var(--textdim);}
</style>
</head>
<body>

<!-- HEADER -->
<div id="hdr">
  <div id="logo">⬡ ORACLE <span class="sl">//</span> MASTER_TRADER</div>
  <div id="cdot"></div>
  <div id="ctext">CONNECTING</div>
  <span style="color:var(--cyan);font-size:11px;letter-spacing:2px">BTC/USD</span>
  <div id="clk">--:--:-- UTC</div>
  <div id="mbadge">PAPER</div>
  <div id="kb-hint">[ F1 HELP ]  [ SPACE MENU ]</div>
</div>

<!-- MAIN GRID -->
<div id="main">

  <!-- VITALS -->
  <div class="panel" id="vitals">
    <div class="ptitle">Vitals</div>
    <div class="sb">
      <div class="sl2"><span>Balance</span>
        <span class="sbadge" style="background:rgba(0,255,136,.08);color:var(--green);border:1px solid rgba(0,255,136,.2)">HP</span>
      </div>
      <div class="sv" id="bal-v">$10,000.00</div>
      <div class="bw"><div class="bf hp-hi" id="hp-f" style="width:100%"></div></div>
    </div>
    <div class="sb">
      <div class="sl2"><span>Kelly Fraction</span>
        <span class="sbadge" style="background:rgba(136,68,255,.08);color:var(--purple);border:1px solid rgba(136,68,255,.2)">MP</span>
      </div>
      <div class="sv" style="font-size:13px" id="kelly-v">--.--%</div>
      <div class="bw"><div class="bf mp-f" id="mp-f" style="width:0%"></div></div>
    </div>
    <div class="sb">
      <div class="sl2"><span>Win Rate</span>
        <span class="sbadge" style="background:rgba(0,204,255,.08);color:var(--cyan);border:1px solid rgba(0,204,255,.2)">XP</span>
      </div>
      <div class="sv" style="font-size:13px" id="wr-v">--%</div>
      <div class="xpw"><div class="bf xp-f" id="xp-f" style="width:0%"></div></div>
    </div>
    <div class="sb">
      <div class="sl2"><span>PnL</span></div>
      <div class="pnl neu" id="pnl-v">$+0.00</div>
    </div>
    <div class="sb">
      <div class="sl2"><span>Trades</span></div>
      <div style="display:flex;gap:6px;align-items:baseline">
        <span style="font-size:13px;font-variant-numeric:tabular-nums" id="t-tot">0</span>
        <span style="font-size:8px;color:var(--textdim)">total</span>
        <span style="font-size:13px;color:var(--cyan);font-variant-numeric:tabular-nums" id="t-open">0</span>
        <span style="font-size:8px;color:var(--textdim)">open</span>
      </div>
    </div>
  </div>

  <!-- PRICE CHART (with minimap phase in corner) -->
  <div class="panel" id="chart-panel">
    <div class="ptitle">Live Price — BTC/USD • Kraken</div>
    <div id="ph">
      <div id="pbig">$--,---</div>
      <div id="pdelta" class="neu">+0.000%</div>
    </div>
    <canvas id="price-cvs"></canvas>
    <!-- Phase space minimap — bottom-right corner like open-world minimap -->
    <div id="phase-mini">
      <canvas id="mini-cvs" style="width:100%;height:100%;display:block;"></canvas>
    </div>
    <div id="phase-mini-label">PHASE τ=5</div>
  </div>

  <!-- SIGNAL TOWER (spans both rows) -->
  <div class="panel" id="signals">
    <div class="ptitle">Signal Tower</div>

    <div class="gb">
      <div class="gn"><span>Microstructure OBI</span><span class="gr" id="obi-r">0.000</span></div>
      <div class="ob">
        <div class="ob-b" id="ob-b" style="width:0%"></div>
        <div class="ob-s" id="ob-s" style="width:0%"></div>
        <div class="ob-c"></div>
      </div>
    </div>

    <div class="gb">
      <div class="gn"><span>Phase Space</span><span class="gr" id="ph-r" style="color:var(--purple)">stable</span></div>
      <div class="pr" id="ph-pips"></div>
    </div>

    <div class="gb">
      <div class="gn"><span>Sentiment F&amp;G</span><span class="gr" id="sn-r">50</span></div>
      <div class="ab"><div class="af" id="sn-f" style="width:50%"></div></div>
    </div>

    <div class="gb">
      <div class="gn"><span>RSI</span><span class="gr" id="rsi-r" style="color:var(--amber)">50</span></div>
      <div class="pr" id="rsi-pips"></div>
    </div>

    <div class="gb">
      <div class="gn"><span>Topology Alarm</span><span class="gr" id="tp-r" style="color:var(--red)">0.0000</span></div>
      <div class="tr2">
        <div class="td" id="tp-dot"></div>
        <div class="tbw"><div class="tbf" id="tp-f" style="width:0%"></div></div>
      </div>
    </div>

    <div class="gb">
      <div class="gn"><span>Trend Strength</span><span class="gr" id="tr-r">0.000</span></div>
      <div class="ab"><div class="af" id="tr-f" style="width:0%;background:var(--cyan)"></div></div>
    </div>

    <div class="gb">
      <div class="gn"><span>Vol Surge</span><span class="gr" id="vs-r" style="color:var(--gold)">1.00x</span></div>
      <div class="ab"><div class="af" id="vs-f" style="width:10%;background:var(--gold)"></div></div>
    </div>

    <hr class="sdiv">
    <div style="font-size:7px;letter-spacing:2px;color:var(--textdim);margin-bottom:5px">Temporal Geometry</div>
    <div class="sr"><span class="sl3">Wavelet Entropy</span><span class="sv2" id="g-wv" style="color:var(--gold)">--</span></div>
    <div class="sr"><span class="sl3">Phasor Coherence</span><span class="sv2" id="g-co" style="color:var(--gold)">--</span></div>
    <div class="sr"><span class="sl3">Dominant Period</span><span class="sv2" id="g-dp" style="color:var(--gold)">--s</span></div>
    <div class="sr"><span class="sl3">Spectral Entropy</span><span class="sv2" id="g-se" style="color:var(--gold)">--</span></div>

    <hr class="sdiv">
    <div style="font-size:7px;letter-spacing:2px;color:var(--textdim);margin-bottom:5px">Neural Model</div>
    <div class="sr"><span class="sl3">Parameters</span><span class="sv2" id="n-p" style="color:var(--cyan)">--</span></div>
    <div class="sr"><span class="sl3">Training Steps</span><span class="sv2" id="n-s" style="color:var(--cyan)">0</span></div>
    <div class="sr"><span class="sl3">Recent Loss</span><span class="sv2" id="n-l" style="color:var(--cyan)">--</span></div>
  </div>

  <!-- AGENTS -->
  <div class="panel" id="agents">
    <div class="ptitle">Active Party</div>
    <div id="ag-list"><div class="nop">⬡ No Active Positions</div></div>
  </div>

  <!-- RADAR / SPIDER CHART — Pokémon stats + Destiny guardian stats -->
  <div class="panel" id="radar-panel">
    <div class="ptitle">Signal Radar — Live Multi-Axis</div>
    <div id="radar-info">
      REGIME:&nbsp;<span id="ri-reg">--</span>&nbsp;&nbsp;
      LYAPUNOV:&nbsp;<span id="ri-lyap">--</span>&nbsp;&nbsp;
      PHASE PTS:&nbsp;<span id="ri-pts">0</span>
    </div>
    <canvas id="radar-cvs"></canvas>
  </div>

</div>

<!-- COMMAND DECK — Kingdom Hearts action bar -->
<div id="deck">

  <!-- KH action buttons — left side -->
  <div style="display:flex;gap:6px;">
    <button class="kh-btn long-btn" onclick="sendCmd('spawn_long')" title="Force spawn LONG agent [L]">▲ LONG</button>
    <button class="kh-btn short-btn" onclick="sendCmd('spawn_short')" title="Force spawn SHORT agent [S]">▼ SHORT</button>
    <button class="kh-btn close-btn" onclick="sendCmd('close_all')" title="Close all positions [C]">✕ ALL</button>
    <button class="kh-btn menu-btn" onclick="openRadial()" title="Open command radial [Space]">◎ MENU</button>
  </div>

  <!-- Condition -->
  <div id="cond-area">
    <div id="cond-lbl">Condition</div>
    <div id="cond-name" style="color:var(--textdim)">SCANNING</div>
    <div id="cond-bw"><div id="cond-bf" style="width:0%;background:var(--textdim)"></div></div>
  </div>

  <div class="ds"><div class="dsl">Confidence</div><div class="dsv" id="dk-cf">--%</div></div>
  <div class="ds"><div class="dsl">Balance</div><div class="dsv" id="dk-bl">$--</div></div>
  <div class="ds"><div class="dsl">Kelly %</div><div class="dsv" id="dk-kl" style="color:var(--purple)">--%</div></div>
  <div class="ds"><div class="dsl">Support</div><div class="dsv" id="dk-sp" style="font-size:11px;color:var(--green)">--</div></div>
  <div class="ds"><div class="dsl">Resist</div><div class="dsv" id="dk-rs" style="font-size:11px;color:var(--red)">--</div></div>
</div>

<!-- RADIAL MENU OVERLAY — Mass Effect Power Wheel + Skyrim Nordic nested lock -->
<div id="radial-overlay" onclick="if(event.target===this)closeRadial()">
  <svg id="radial-svg" width="480" height="480" viewBox="-240 -240 480 480">
    <!-- Outer ring decorations -->
    <circle cx="0" cy="0" r="230" fill="none" stroke="rgba(26,32,96,.6)" stroke-width="1"/>
    <circle cx="0" cy="0" r="80"  fill="none" stroke="rgba(26,32,96,.6)" stroke-width="1"/>
    <circle cx="0" cy="0" r="40"  fill="rgba(3,3,18,.8)" stroke="rgba(0,204,255,.2)" stroke-width="1"/>
    <!-- Center ORACLE mark -->
    <text x="0" y="5" text-anchor="middle" fill="rgba(0,204,255,.7)" font-size="9"
      font-family="Courier New" letter-spacing="2">ORACLE</text>

    <!-- Sectors built by JS -->
    <g id="radial-sectors"></g>
    <!-- Sub-menu ring built by JS -->
    <g id="sub-ring"></g>
    <!-- Key labels -->
    <g id="key-labels"></g>
  </svg>
  <div id="radial-label-box">
    <div id="r-name" style="font-size:13px;letter-spacing:4px;color:var(--cyan)"></div>
    <div id="r-desc" style="font-size:9px;color:var(--textdim);margin-top:2px"></div>
    <div id="radial-hint">CLICK TO SELECT • ESC TO CLOSE • 1-8 TO JUMP</div>
  </div>
</div>

<!-- KEYBOARD SHORTCUT OVERLAY -->
<div id="kb-overlay" onclick="if(event.target===this)closeKb()">
  <div id="kb-box">
    <h2>⬡ Command Reference</h2>
    <div class="kb-row"><span class="kb-key">SPACE / TAB</span><span class="kb-desc">Open radial command menu</span></div>
    <div class="kb-row"><span class="kb-key">1 – 8</span><span class="kb-desc">Quick select radial sector</span></div>
    <div class="kb-row"><span class="kb-key">L</span><span class="kb-desc">Force spawn LONG agent</span></div>
    <div class="kb-row"><span class="kb-key">S</span><span class="kb-desc">Force spawn SHORT agent</span></div>
    <div class="kb-row"><span class="kb-key">C</span><span class="kb-desc">Close all positions</span></div>
    <div class="kb-row"><span class="kb-key">R</span><span class="kb-desc">Reset paper balance</span></div>
    <div class="kb-row"><span class="kb-key">P</span><span class="kb-desc">Toggle phase space minimap</span></div>
    <div class="kb-row"><span class="kb-key">ESC</span><span class="kb-desc">Close any menu</span></div>
    <div class="kb-row"><span class="kb-key">F1</span><span class="kb-desc">This help screen</span></div>
  </div>
</div>

<!-- TOAST -->
<div id="toast"></div>

<script>
'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const priceHist  = [];
const phaseHist  = [];
let prevPrice    = null;
let radarData    = { obi:0.5, phase:0.5, sentiment:0.5, rsi:0.5, topology:0, trend:0.5 };

// ── Canvas refs ────────────────────────────────────────────────────────────
const priceCvs = document.getElementById('price-cvs');
const pCtx     = priceCvs.getContext('2d');
const miniCvs  = document.getElementById('mini-cvs');
const mCtx     = miniCvs.getContext('2d');
const radarCvs = document.getElementById('radar-cvs');
const rCtx     = radarCvs.getContext('2d');

// ── Condition colors (Dead Space color language) ───────────────────────────
const CC = {
  bullish_breakout:'#00ff88', mean_reversion_oversold:'#00ff88',
  strong_uptrend:'#00ccff',   bearish_breakdown:'#ff2244',
  mean_reversion_overbought:'#ff2244', strong_downtrend:'#ff4455',
  phase_transition:'#8844ff', crash_warning:'#ff0000',
  bubble_warning:'#ff6600',   high_volatility:'#ffcc44',
  sideways_consolidation:'#334466', scanning:'#334466',
};
const PC = {
  chaotic_expansion:'#ff2244', stable_contraction:'#8844ff',
  stable:'#00ff88', warming_up:'#334466',
};

// ── Build pips ─────────────────────────────────────────────────────────────
function buildPips(id, n) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  for (let i = 0; i < n; i++) { const p = document.createElement('div'); p.className='pp'; el.appendChild(p); }
}
function setPips(id, frac, col) {
  const pips = document.getElementById(id).querySelectorAll('.pp');
  const lit  = Math.round(Math.max(0,Math.min(1,frac)) * pips.length);
  pips.forEach((p,i) => {
    if (i < lit) { p.style.background=col; p.style.boxShadow=`0 0 5px ${col}`; }
    else         { p.style.background='rgba(255,255,255,0.05)'; p.style.boxShadow='none'; }
  });
}

// ── Resize ─────────────────────────────────────────────────────────────────
function resizeAll() {
  const cp = document.getElementById('chart-panel');
  priceCvs.width  = cp.clientWidth  - 22;
  priceCvs.height = Math.max(60, cp.clientHeight - 68);

  miniCvs.width  = 110; miniCvs.height = 90;

  const rp = document.getElementById('radar-panel');
  radarCvs.width  = rp.clientWidth  - 22;
  radarCvs.height = Math.max(60, rp.clientHeight - 48);
}

// ── Price chart ────────────────────────────────────────────────────────────
function drawPrice() {
  const W=priceCvs.width, H=priceCvs.height;
  if (W<=0||H<=0||priceHist.length<2) { pCtx.clearRect(0,0,W,H); return; }
  pCtx.clearRect(0,0,W,H);
  const prices = priceHist.map(p=>p.v);
  const mn=Math.min(...prices), mx=Math.max(...prices), range=mx-mn||1;
  const pad=12, tp=18;
  const xi = i => pad+(i/(priceHist.length-1))*(W-2*pad);
  const yi = v => tp+(1-(v-mn)/range)*(H-tp-pad);

  // Grid
  pCtx.strokeStyle='rgba(26,32,96,0.4)'; pCtx.lineWidth=1;
  [0.25,0.5,0.75].forEach(t=>{
    const y=tp+t*(H-tp-pad);
    pCtx.beginPath(); pCtx.moveTo(pad,y); pCtx.lineTo(W-pad,y); pCtx.stroke();
    pCtx.fillStyle='rgba(68,85,119,0.6)'; pCtx.font='8px Courier New';
    pCtx.fillText('$'+(mn+(1-t)*range).toLocaleString('en-US',{maximumFractionDigits:0}), pad+2, y-2);
  });

  // Support/resistance
  if (window._sup>0) {
    pCtx.setLineDash([4,6]); pCtx.strokeStyle='rgba(0,255,136,0.3)'; pCtx.lineWidth=1;
    const sy=yi(window._sup); pCtx.beginPath(); pCtx.moveTo(pad,sy); pCtx.lineTo(W-pad,sy); pCtx.stroke();
  }
  if (window._res>0) {
    pCtx.strokeStyle='rgba(255,34,68,0.3)';
    const ry=yi(window._res); pCtx.beginPath(); pCtx.moveTo(pad,ry); pCtx.lineTo(W-pad,ry); pCtx.stroke();
  }
  pCtx.setLineDash([]);

  // Area fill
  const grad=pCtx.createLinearGradient(0,tp,0,H);
  grad.addColorStop(0,'rgba(0,204,255,0.18)'); grad.addColorStop(1,'rgba(0,204,255,0)');
  pCtx.beginPath(); pCtx.moveTo(xi(0),yi(prices[0]));
  for (let i=1;i<prices.length;i++) pCtx.lineTo(xi(i),yi(prices[i]));
  pCtx.lineTo(xi(prices.length-1),H); pCtx.lineTo(xi(0),H);
  pCtx.closePath(); pCtx.fillStyle=grad; pCtx.fill();

  // Line
  pCtx.beginPath(); pCtx.strokeStyle='#00ccff'; pCtx.lineWidth=1.5;
  pCtx.shadowBlur=6; pCtx.shadowColor='rgba(0,204,255,0.5)';
  pCtx.moveTo(xi(0),yi(prices[0]));
  for (let i=1;i<prices.length;i++) pCtx.lineTo(xi(i),yi(prices[i]));
  pCtx.stroke(); pCtx.shadowBlur=0;

  // Last dot
  const lx=xi(prices.length-1), ly=yi(prices[prices.length-1]);
  pCtx.beginPath(); pCtx.arc(lx,ly,3.5,0,Math.PI*2);
  pCtx.fillStyle='#00ccff'; pCtx.shadowBlur=14; pCtx.shadowColor='#00ccff';
  pCtx.fill(); pCtx.shadowBlur=0;
}

// ── Phase minimap (corner overlay — like BotW minimap) ────────────────────
function drawMini() {
  const W=miniCvs.width, H=miniCvs.height;
  // Persistence fade — creates trail without clearing
  mCtx.fillStyle='rgba(7,7,32,0.15)'; mCtx.fillRect(0,0,W,H);
  if (phaseHist.length<3) {
    mCtx.fillStyle='#070720'; mCtx.fillRect(0,0,W,H);
    mCtx.fillStyle='rgba(68,85,119,0.5)'; mCtx.font='8px Courier New';
    mCtx.textAlign='center'; mCtx.fillText('WARMING',W/2,H/2-6); mCtx.fillText('UP',W/2,H/2+6); mCtx.textAlign='left';
    return;
  }
  const xs=phaseHist.map(p=>p[0]), ys=phaseHist.map(p=>p[1]);
  const xmn=Math.min(...xs),xmx=Math.max(...xs),xr=(xmx-xmn)||1;
  const ymn=Math.min(...ys),ymx=Math.max(...ys),yr=(ymx-ymn)||1;
  const pad=4;
  const px=v=>pad+((v-xmn)/xr)*(W-2*pad);
  const py=v=>pad+((v-ymn)/yr)*(H-2*pad);

  for (let i=1;i<phaseHist.length;i++) {
    const t=i/phaseHist.length;
    const hue=260+t*80, lum=30+t*35;
    mCtx.beginPath();
    mCtx.strokeStyle=`hsla(${hue},80%,${lum}%,${0.2+t*0.8})`;
    mCtx.lineWidth=t>0.85?1.5:0.8;
    mCtx.moveTo(px(phaseHist[i-1][0]),py(phaseHist[i-1][1]));
    mCtx.lineTo(px(phaseHist[i][0]),  py(phaseHist[i][1]));
    mCtx.stroke();
  }
  const lx=px(xs[xs.length-1]), ly=py(ys[ys.length-1]);
  mCtx.beginPath(); mCtx.arc(lx,ly,3,0,Math.PI*2);
  mCtx.fillStyle='#fff'; mCtx.shadowBlur=10; mCtx.shadowColor='#8844ff';
  mCtx.fill(); mCtx.shadowBlur=0;
}

// ── Radar/Spider chart — Pokémon stats + Destiny guardian ─────────────────
const AXES = [
  {k:'obi',       label:'OBI',       angle:-Math.PI/2},
  {k:'phase',     label:'PHASE',     angle:-Math.PI/2+Math.PI/3},
  {k:'sentiment', label:'SENTIMENT', angle:-Math.PI/2+2*Math.PI/3},
  {k:'rsi',       label:'RSI',       angle:-Math.PI/2+Math.PI},
  {k:'trend',     label:'TREND',     angle:-Math.PI/2+4*Math.PI/3},
  {k:'topology',  label:'TOPOLOGY',  angle:-Math.PI/2+5*Math.PI/3},
];

function drawRadar() {
  const W=radarCvs.width, H=radarCvs.height;
  if (W<=0||H<=0) return;
  rCtx.clearRect(0,0,W,H);
  const cx=W/2, cy=H/2, R=Math.min(cx,cy)-28;
  const n=AXES.length;

  // Spider web rings (concentric hexagons)
  [0.25,0.5,0.75,1.0].forEach(t=>{
    rCtx.beginPath();
    AXES.forEach((a,i)=>{
      const x=cx+Math.cos(a.angle)*R*t;
      const y=cy+Math.sin(a.angle)*R*t;
      i===0?rCtx.moveTo(x,y):rCtx.lineTo(x,y);
    });
    rCtx.closePath();
    rCtx.strokeStyle=t===1?'rgba(26,32,96,0.8)':'rgba(26,32,96,0.35)';
    rCtx.lineWidth=t===1?1:0.5; rCtx.stroke();
  });

  // Axis lines
  AXES.forEach(a=>{
    rCtx.beginPath();
    rCtx.moveTo(cx,cy);
    rCtx.lineTo(cx+Math.cos(a.angle)*R, cy+Math.sin(a.angle)*R);
    rCtx.strokeStyle='rgba(26,32,96,0.6)'; rCtx.lineWidth=1; rCtx.stroke();
  });

  // Data polygon — filled with glow
  const vals = AXES.map(a=>Math.max(0,Math.min(1,radarData[a.k]||0)));
  const condColor = window._condColor || '#00ccff';

  rCtx.beginPath();
  vals.forEach((v,i)=>{
    const x=cx+Math.cos(AXES[i].angle)*R*v;
    const y=cy+Math.sin(AXES[i].angle)*R*v;
    i===0?rCtx.moveTo(x,y):rCtx.lineTo(x,y);
  });
  rCtx.closePath();

  // Gradient fill
  const grad=rCtx.createRadialGradient(cx,cy,0,cx,cy,R);
  grad.addColorStop(0, condColor+'44');
  grad.addColorStop(1, condColor+'11');
  rCtx.fillStyle=grad; rCtx.fill();

  // Outline with glow
  rCtx.strokeStyle=condColor; rCtx.lineWidth=1.5;
  rCtx.shadowBlur=8; rCtx.shadowColor=condColor; rCtx.stroke(); rCtx.shadowBlur=0;

  // Dots at vertices
  vals.forEach((v,i)=>{
    const x=cx+Math.cos(AXES[i].angle)*R*v;
    const y=cy+Math.sin(AXES[i].angle)*R*v;
    rCtx.beginPath(); rCtx.arc(x,y,3,0,Math.PI*2);
    rCtx.fillStyle=condColor; rCtx.shadowBlur=8; rCtx.shadowColor=condColor;
    rCtx.fill(); rCtx.shadowBlur=0;
  });

  // Labels
  rCtx.font='8px Courier New'; rCtx.fillStyle='rgba(68,85,119,0.9)'; rCtx.textAlign='center';
  AXES.forEach(a=>{
    const lx=cx+Math.cos(a.angle)*(R+16);
    const ly=cy+Math.sin(a.angle)*(R+16)+3;
    rCtx.fillText(a.label, lx, ly);
  });
  rCtx.textAlign='left';
}

// Continuous animation loop for radar + mini
let animRunning=true;
function animLoop() {
  if (!animRunning) return;
  drawMini(); drawRadar();
  requestAnimationFrame(animLoop);
}

// ── Radial menu — Mass Effect power wheel + Skyrim Nordic nested lock ──────
const SECTORS = [
  { label:'LONG',     desc:'Spawn a LONG agent',        color:'#00ff88', key:'1', cmd:'spawn_long',
    subs:[{label:'FORCE',desc:'Force spawn now',cmd:'spawn_long'},{label:'CONF+',desc:'Lower threshold',cmd:'conf_down'}]},
  { label:'SHORT',    desc:'Spawn a SHORT agent',       color:'#ff2244', key:'2', cmd:'spawn_short',
    subs:[{label:'FORCE',desc:'Force spawn now',cmd:'spawn_short'},{label:'CONF+',desc:'Lower threshold',cmd:'conf_down'}]},
  { label:'CLOSE',    desc:'Close all positions',       color:'#ffaa00', key:'3', cmd:'close_all',
    subs:[{label:'ALL',desc:'Close all agents',cmd:'close_all'},{label:'OLDEST',desc:'Close oldest',cmd:'close_oldest'}]},
  { label:'RISK',     desc:'Risk management',           color:'#ff6600', key:'4', cmd:null,
    subs:[{label:'SL-',desc:'Tighten stop loss',cmd:'sl_tighten'},{label:'TP+',desc:'Extend take profit',cmd:'tp_extend'}]},
  { label:'NEURAL',   desc:'Neural model controls',     color:'#ffcc44', key:'5', cmd:null,
    subs:[{label:'SAVE',desc:'Save weights',cmd:'neural_save'},{label:'RESET',desc:'Reset weights',cmd:'neural_reset'}]},
  { label:'MARKET',   desc:'Market data controls',      color:'#00ccff', key:'6', cmd:null,
    subs:[{label:'REFRESH',desc:'Refresh sentiment',cmd:'refresh_sent'},{label:'HISTORY',desc:'Re-seed history',cmd:'reseed'}]},
  { label:'PORTFOLIO',desc:'Portfolio & paper trading', color:'#8844ff', key:'7', cmd:null,
    subs:[{label:'RESET',desc:'Reset paper balance',cmd:'paper_reset'},{label:'REPORT',desc:'Show summary',cmd:'report'}]},
  { label:'SYSTEM',   desc:'System settings',           color:'#445577', key:'8', cmd:null,
    subs:[{label:'LOGS',desc:'Dump to console',cmd:'dump_logs'},{label:'STATUS',desc:'Show substrate',cmd:'substrate_status'}]},
];

let radialOpen   = false;
let hoveredSector= null;
const R_OUTER=210, R_INNER=85, R_SUB_OUT=80, R_SUB_IN=44;
const SECTOR_GAP = 0.04;   // radians gap between sectors

function sectorPath(cx,cy,r1,r2,startAngle,endAngle) {
  const a1=startAngle+SECTOR_GAP/2, a2=endAngle-SECTOR_GAP/2;
  return [
    `M ${cx+r1*Math.cos(a1)} ${cy+r1*Math.sin(a1)}`,
    `L ${cx+r2*Math.cos(a1)} ${cy+r2*Math.sin(a1)}`,
    `A ${r2} ${r2} 0 0 1 ${cx+r2*Math.cos(a2)} ${cy+r2*Math.sin(a2)}`,
    `L ${cx+r1*Math.cos(a2)} ${cy+r1*Math.sin(a2)}`,
    `A ${r1} ${r1} 0 0 0 ${cx+r1*Math.cos(a1)} ${cy+r1*Math.sin(a1)}`,
    'Z'
  ].join(' ');
}

function buildRadial() {
  const g    = document.getElementById('radial-sectors');
  const kl   = document.getElementById('key-labels');
  g.innerHTML=''; kl.innerHTML='';

  SECTORS.forEach((s,i)=>{
    const span  = (Math.PI*2)/SECTORS.length;
    const start = -Math.PI/2 + i*span;
    const end   = start+span;
    const mid   = (start+end)/2;

    // Sector path
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d', sectorPath(0,0,R_INNER,R_OUTER,start,end));
    path.setAttribute('fill', s.color+'22');
    path.setAttribute('stroke', s.color+'66');
    path.setAttribute('stroke-width','1');
    path.setAttribute('class','radial-sector');
    path.dataset.idx=i;

    path.addEventListener('mouseenter',()=>hoverSector(i));
    path.addEventListener('mouseleave',()=>hoverSector(null));
    path.addEventListener('click',()=>selectSector(i));
    g.appendChild(path);

    // Label
    const lx=(R_INNER+R_OUTER)/2*Math.cos(mid);
    const ly=(R_INNER+R_OUTER)/2*Math.sin(mid);
    const txt=document.createElementNS('http://www.w3.org/2000/svg','text');
    txt.setAttribute('x',lx); txt.setAttribute('y',ly);
    txt.setAttribute('text-anchor','middle'); txt.setAttribute('dominant-baseline','middle');
    txt.setAttribute('fill',s.color); txt.setAttribute('font-size','9');
    txt.setAttribute('font-family','Courier New'); txt.setAttribute('letter-spacing','2');
    txt.setAttribute('pointer-events','none');
    txt.textContent=s.label;
    g.appendChild(txt);

    // Key badge
    const kx=(R_OUTER+12)*Math.cos(mid), ky=(R_OUTER+12)*Math.sin(mid);
    const kb=document.createElementNS('http://www.w3.org/2000/svg','text');
    kb.setAttribute('x',kx); kb.setAttribute('y',ky);
    kb.setAttribute('text-anchor','middle'); kb.setAttribute('dominant-baseline','middle');
    kb.setAttribute('fill','rgba(200,216,240,0.4)'); kb.setAttribute('font-size','8');
    kb.setAttribute('font-family','Courier New'); kb.setAttribute('pointer-events','none');
    kb.textContent=s.key;
    kl.appendChild(kb);
  });
}

function hoverSector(idx) {
  hoveredSector=idx;
  // Update sub-ring
  const sg  = document.getElementById('sub-ring');
  sg.innerHTML='';
  document.getElementById('r-name').textContent='';
  document.getElementById('r-desc').textContent='';
  if (idx===null) return;
  const s   = SECTORS[idx];
  const span= (Math.PI*2)/SECTORS.length;
  const mid = -Math.PI/2 + idx*span + span/2;
  document.getElementById('r-name').textContent=s.label;
  document.getElementById('r-desc').textContent=s.desc;

  // Highlight main sector
  document.querySelectorAll('.radial-sector').forEach((p,i)=>{
    p.setAttribute('fill', i===idx ? SECTORS[i].color+'44' : SECTORS[i].color+'22');
    p.setAttribute('stroke',i===idx ? SECTORS[i].color+'cc' : SECTORS[i].color+'66');
  });

  // Build sub-ring (inner ring — the Skyrim Nordic nested lock)
  const subSpan=(Math.PI*2)/s.subs.length;
  s.subs.forEach((sub,j)=>{
    const sa=-Math.PI/2+j*subSpan, se=sa+subSpan;
    const sm=(sa+se)/2;
    const grp=document.createElementNS('http://www.w3.org/2000/svg','g');
    grp.setAttribute('class','sub-sector');

    const p=document.createElementNS('http://www.w3.org/2000/svg','path');
    p.setAttribute('d',sectorPath(0,0,R_SUB_IN,R_SUB_OUT,sa,se));
    p.setAttribute('fill',s.color+'33'); p.setAttribute('stroke',s.color+'88');
    p.setAttribute('stroke-width','1');
    p.addEventListener('click',e=>{e.stopPropagation();sendCmd(sub.cmd);closeRadial();toast(sub.label);});

    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',(R_SUB_IN+R_SUB_OUT)/2*Math.cos(sm));
    t.setAttribute('y',(R_SUB_IN+R_SUB_OUT)/2*Math.sin(sm));
    t.setAttribute('text-anchor','middle'); t.setAttribute('dominant-baseline','middle');
    t.setAttribute('fill',s.color); t.setAttribute('font-size','7');
    t.setAttribute('font-family','Courier New'); t.setAttribute('pointer-events','none');
    t.textContent=sub.label;

    grp.appendChild(p); grp.appendChild(t);
    sg.appendChild(grp);
  });
}

function selectSector(idx) {
  const s=SECTORS[idx];
  if (s.cmd) { sendCmd(s.cmd); toast(s.label); closeRadial(); }
}

function openRadial() {
  radialOpen=true;
  document.getElementById('radial-overlay').classList.add('open');
  buildRadial();
}
function closeRadial() {
  radialOpen=false;
  document.getElementById('radial-overlay').classList.remove('open');
  hoveredSector=null;
}

// ── Send command to bot via POST /cmd ─────────────────────────────────────
async function sendCmd(action) {
  if (!action) return;
  try {
    const r=await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action})});
    const j=await r.json();
    toast(action.toUpperCase().replace(/_/g,' ') + (j.ok?' ✓':' ✗'));
  } catch(e) { toast('ERROR: ' + e.message); }
}

// ── Toast notification ────────────────────────────────────────────────────
let _toastTimer=null;
function toast(msg) {
  const el=document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>el.classList.remove('show'),2200);
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{
  if (e.target.tagName==='INPUT') return;
  switch(e.key) {
    case ' ': case 'Tab': e.preventDefault(); radialOpen?closeRadial():openRadial(); break;
    case 'Escape': closeRadial(); closeKb(); break;
    case 'F1': e.preventDefault(); toggleKb(); break;
    case 'l': case 'L': sendCmd('spawn_long');  toast('▲ LONG QUEUED');  break;
    case 's': case 'S': sendCmd('spawn_short'); toast('▼ SHORT QUEUED'); break;
    case 'c': case 'C': sendCmd('close_all');   toast('✕ CLOSE ALL QUEUED'); break;
    case 'r': case 'R': sendCmd('paper_reset'); toast('PAPER RESET QUEUED'); break;
    case 'p': case 'P': document.getElementById('phase-mini').style.display=
      document.getElementById('phase-mini').style.display==='none'?'block':'none'; break;
    case '1':case '2':case '3':case '4':
    case '5':case '6':case '7':case '8':
      const idx=parseInt(e.key)-1;
      if (!radialOpen) openRadial();
      setTimeout(()=>hoverSector(idx),50); break;
  }
});

function toggleKb(){document.getElementById('kb-overlay').classList.toggle('open');}
function closeKb(){document.getElementById('kb-overlay').classList.remove('open');}

// ── Update UI ──────────────────────────────────────────────────────────────
function hpClass(p){return p>.65?'bf hp-hi':p>.35?'bf hp-md':'bf hp-lo';}

function updateUI(d) {
  // Price
  if (d.price>0) {
    const fmt=p=>'$'+p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    document.getElementById('pbig').textContent=fmt(d.price);
    if (prevPrice!==null) {
      const chg=((d.price-prevPrice)/prevPrice)*100;
      const el=document.getElementById('pdelta');
      el.textContent=(chg>=0?'+':'')+chg.toFixed(3)+'%';
      el.className=chg>0.0005?'pos':chg<-0.0005?'neg':'neu';
    }
    prevPrice=d.price;
    priceHist.push({v:d.price});
    if (priceHist.length>200) priceHist.shift();
    window._sup=d.support>0?d.support:null;
    window._res=d.resistance>0?d.resistance:null;
    drawPrice();
  }

  // Vitals
  const port=d.portfolio||{}, pnl=port.total_pnl||0, wr=(port.win_rate||0)*100, bal=d.balance||10000;
  const kelly=d.kelly_fraction||0;
  document.getElementById('bal-v').textContent='$'+bal.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  const balPct=Math.max(0,Math.min(1,bal/10000));
  const hpEl=document.getElementById('hp-f');
  hpEl.style.width=(balPct*100)+'%'; hpEl.className=hpClass(balPct);
  document.getElementById('kelly-v').textContent=(kelly*100).toFixed(2)+'%';
  document.getElementById('mp-f').style.width=Math.min(100,kelly*100*5)+'%';
  document.getElementById('wr-v').textContent=wr.toFixed(0)+'%';
  document.getElementById('xp-f').style.width=wr+'%';
  const pnlEl=document.getElementById('pnl-v');
  pnlEl.textContent='$'+(pnl>=0?'+':'')+pnl.toFixed(2);
  pnlEl.className='pnl '+(pnl>0?'pos':pnl<0?'neg':'neu');
  document.getElementById('t-tot').textContent=port.total_trades||0;
  document.getElementById('t-open').textContent=d.open_trades||0;

  // Condition
  const cond=d.condition||'scanning', conf=d.confidence||0;
  const color=CC[cond]||'#334466';
  window._condColor=color;
  const cn=document.getElementById('cond-name');
  cn.textContent=cond.replace(/_/g,' ').toUpperCase();
  cn.style.color=color; cn.style.textShadow=`0 0 16px ${color}`;
  const cf=document.getElementById('cond-bf');
  cf.style.width=(conf*100)+'%'; cf.style.background=color;
  document.getElementById('dk-cf').textContent=(conf*100).toFixed(1)+'%';
  document.getElementById('dk-cf').style.color=color;
  document.getElementById('dk-bl').textContent='$'+Math.round(bal).toLocaleString();
  document.getElementById('dk-kl').textContent=(kelly*100).toFixed(2)+'%';
  document.getElementById('dk-sp').textContent=d.support>0?'$'+Math.round(d.support).toLocaleString():'--';
  document.getElementById('dk-rs').textContent=d.resistance>0?'$'+Math.round(d.resistance).toLocaleString():'--';

  // Signal tower
  const obi=d.obi||0;
  document.getElementById('obi-r').textContent=obi.toFixed(3);
  document.getElementById('obi-r').style.color=obi>0?'var(--green)':obi<0?'var(--red)':'var(--textdim)';
  document.getElementById('ob-b').style.width=obi>0?Math.min(50,obi*50)+'%':'0%';
  document.getElementById('ob-s').style.width=obi<0?Math.min(50,-obi*50)+'%':'0%';

  const pr=d.phase_regime||'warming_up';
  const pc=PC[pr]||'#334466';
  const pf={chaotic_expansion:1,stable_contraction:.37,stable:.55,warming_up:.12}[pr]||.5;
  setPips('ph-pips',pf,pc);
  const phr=document.getElementById('ph-r'); phr.textContent=pr.replace(/_/g,' '); phr.style.color=pc;

  const fgv=d.fear_greed?.value||50, fgl=d.fear_greed?.label||'neutral';
  document.getElementById('sn-r').textContent=fgv+' — '+fgl;
  const snf=document.getElementById('sn-f');
  snf.style.width=fgv+'%'; snf.style.background=fgv>60?'var(--green)':fgv<40?'var(--red)':'var(--amber)';

  const rsi=d.rsi||50;
  document.getElementById('rsi-r').textContent=Math.round(rsi);
  document.getElementById('rsi-r').style.color=rsi>70?'var(--red)':rsi<30?'var(--green)':'var(--amber)';
  setPips('rsi-pips',rsi/100,rsi>70?'var(--red)':rsi<30?'var(--green)':'var(--amber)');

  const topo=d.topology||0;
  document.getElementById('tp-r').textContent=topo.toFixed(4);
  document.getElementById('tp-f').style.width=Math.min(100,topo*100)+'%';
  topo>.3?document.getElementById('tp-dot').classList.add('alarmed')
        :document.getElementById('tp-dot').classList.remove('alarmed');

  const trend=d.trend_strength||0;
  document.getElementById('tr-r').textContent=trend.toFixed(3);
  document.getElementById('tr-r').style.color=trend>0?'var(--cyan)':'var(--red)';
  const trf=document.getElementById('tr-f');
  trf.style.width=Math.min(100,Math.abs(trend)*100)+'%';
  trf.style.background=trend>=0?'var(--cyan)':'var(--red)';

  const vs=d.vol_surge||1;
  document.getElementById('vs-r').textContent=vs.toFixed(2)+'x';
  document.getElementById('vs-f').style.width=Math.min(100,(vs-.5)*66.7)+'%';

  // Geometry
  const geo=d.geometry||{};
  const gf=(v,dp)=>(v!=null&&!isNaN(v))?Number(v).toFixed(dp):'--';
  document.getElementById('g-wv').textContent=gf(geo.wavelet_entropy,4);
  document.getElementById('g-co').textContent=gf(geo.phasor_coherence,4);
  document.getElementById('g-dp').textContent=geo.dominant_period!=null?gf(geo.dominant_period,1)+'s':'--s';
  document.getElementById('g-se').textContent=gf(geo.attractor_spectral_entropy,4);

  // Neural
  if (d.neural) {
    document.getElementById('n-p').textContent=(d.neural.param_count||0).toLocaleString();
    document.getElementById('n-s').textContent=d.neural.training_steps||0;
    document.getElementById('n-l').textContent=(d.neural.recent_loss||0).toFixed(5);
  }

  // Phase points → minimap
  if (d.phase_points?.length>0) {
    d.phase_points.forEach(p=>phaseHist.push(p));
    while(phaseHist.length>500) phaseHist.shift();
    document.getElementById('ri-reg').textContent=pr.replace(/_/g,' ');
    document.getElementById('ri-lyap').textContent=(d.lyapunov||0).toFixed(5);
    document.getElementById('ri-pts').textContent=phaseHist.length;
  }

  // Radar data update
  radarData = {
    obi:        Math.max(0,Math.min(1,(obi+1)/2)),
    phase:      pf,
    sentiment:  fgv/100,
    rsi:        rsi/100,
    trend:      Math.max(0,Math.min(1,(trend+1)/2)),
    topology:   Math.min(1,topo),
  };

  // Agents
  renderAgents(d.agents||[]);
}

function renderAgents(agents) {
  const el=document.getElementById('ag-list');
  if (!agents.length) { el.innerHTML='<div class="nop">⬡ No Active Positions</div>'; return; }
  el.innerHTML=agents.map(a=>{
    const iL=a.direction==='LONG';
    const hpPct=Math.max(0,Math.min(100,50+(a.pnl/Math.max(1,a.size*.006))*50));
    const hpc=hpPct>60?'var(--green)':hpPct>30?'var(--amber)':'var(--red)';
    const dc=iL?'long':'short';
    return `<div class="ac ${dc}">
      <div class="at2">
        <div class="an">${a.name}</div>
        <div class="ad ${dc}">${iL?'▲ LONG':'▼ SHORT'}</div>
      </div>
      <div class="am">
        <span style="color:var(--textdim)">$${(a.size||0).toFixed(0)}</span>
        <span style="color:${a.pnl>=0?'var(--green)':'var(--red)'}">${a.pnl>=0?'+':''}$${(a.pnl||0).toFixed(2)}</span>
      </div>
      <div class="ahb"><div class="ahf" style="width:${hpPct}%;background:${hpc};box-shadow:0 0 4px ${hpc}"></div></div>
    </div>`;
  }).join('');
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  const ws=new WebSocket('ws://'+location.hostname+':8765/ws');
  const dot=document.getElementById('cdot'), txt=document.getElementById('ctext');
  ws.onopen=()=>{ dot.className='live'; txt.textContent='LIVE — KRAKEN'; };
  ws.onmessage=e=>{ try{updateUI(JSON.parse(e.data));}catch(err){console.debug('[ORACLE] parse error',err);} };
  ws.onclose=()=>{ dot.className=''; txt.textContent='RECONNECTING...'; setTimeout(connect,2000); };
}

// ── Boot ───────────────────────────────────────────────────────────────────
window.addEventListener('load',()=>{
  buildPips('ph-pips',8); buildPips('rsi-pips',10);
  resizeAll();
  window.addEventListener('resize',resizeAll);
  setInterval(()=>{
    const t=new Date();
    document.getElementById('clk').textContent=t.toUTCString().slice(17,25)+' UTC';
  },1000);
  animLoop();
  connect();
  // Show hint briefly on load
  setTimeout(()=>toast('SPACE to open radial menu • F1 for shortcuts'),1500);
});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket + Command endpoint
# ─────────────────────────────────────────────────────────────────────────────

_clients: Set[web.WebSocketResponse] = set()
_latest_payload: str = ""


import math as _m

def _safe_json(obj):
    """Recursively sanitize: Infinity/NaN/numpy scalars → 0.0."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        return 0.0 if (_m.isnan(obj) or _m.isinf(obj)) else obj
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            f = float(obj)
            return 0.0 if (_m.isnan(f) or _m.isinf(f)) else f
    except ImportError:
        pass
    if hasattr(obj, '__float__') and not isinstance(obj, (str, bytes, bool, int)):
        try:
            f = float(obj)
            return 0.0 if (_m.isnan(f) or _m.isinf(f)) else f
        except Exception:
            pass
    return obj


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _clients.add(ws)
    if _latest_payload:
        await ws.send_str(_latest_payload)
    try:
        async for _ in ws:
            pass
    finally:
        _clients.discard(ws)
    return ws


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html")


async def cmd_handler(request: web.Request) -> web.Response:
    """Receive commands from the browser and enqueue for main.py."""
    try:
        data = await request.json()
        action = data.get("action", "")
        if action:
            await _cmd_queue.put({"action": action, **{k: v for k, v in data.items() if k != "action"}})
        return web.json_response({"ok": True, "action": action})
    except Exception as e:
        return web.Response(status=400, text=str(e))


async def broadcast(state: dict) -> None:
    global _latest_payload
    payload = json.dumps(_safe_json(state))
    _latest_payload = payload
    if not _clients:
        return
    dead: Set[web.WebSocketResponse] = set()
    for ws in list(_clients):
        try:
            await ws.send_str(payload)
        except Exception:
            dead.add(ws)
    _clients -= dead


def build_state(context, spawner, executor, memory, neural_model=None) -> dict:
    kl    = context.key_levels
    tech  = context.context_data.get("technical",   {})
    phase = context.context_data.get("phase_space",  {})
    sent  = context.context_data.get("sentiment",   {})
    geo   = context.context_data.get("geometry",    {})
    fg    = sent.get("fear_greed", {})

    prices    = list(executor.memory._get_recent_prices()) if hasattr(executor.memory, "_get_recent_prices") else []
    phase_pts = []
    if len(prices) >= 10:
        for i in range(5, len(prices)):
            phase_pts.append([float(prices[i]), float(prices[i - 5])])
        phase_pts = phase_pts[-60:]

    agents = []
    for a_id, ti in executor.open_trades.items():
        cur = float(kl.get("current", ti["entry"]))
        pnl = ((cur - ti["entry"]) / ti["entry"] * ti["size"]
               if ti["direction"] == "LONG"
               else (ti["entry"] - cur) / ti["entry"] * ti["size"])
        name = getattr(spawner.active_agents.get(a_id), "agent_name", a_id)
        agents.append({"name": name, "direction": ti["direction"],
                        "size": float(ti["size"]), "pnl": float(pnl)})

    portfolio = memory.portfolio_summary()

    kelly_f = 0.0
    try:
        kelly_f = float(spawner._kelly_fraction())
        if _m.isnan(kelly_f) or _m.isinf(kelly_f):
            kelly_f = 0.0
    except Exception:
        pass

    state: dict = {
        "ts":             datetime.now(timezone.utc).isoformat(),
        "price":          float(kl.get("current", 0)),
        "condition":      context.condition.value,
        "confidence":     float(context.confidence),
        "sentiment":      float(context.sentiment_score),
        "obi":            float(context.context_data.get("obi", 0)),
        "obi_label":      context.context_data.get("obi_label", "normal"),
        "phase_regime":   context.phase_space_regime,
        "lyapunov":       float(phase.get("lyapunov", 0)),
        "topology":       float(context.topology_alarm),
        "rsi":            float(tech.get("rsi", 50)),
        "trend_strength": float(tech.get("trend_strength", 0)),
        "vol_surge":      float(tech.get("vol_surge", 1)),
        "support":        float(kl.get("support", 0)),
        "resistance":     float(kl.get("resistance", 0)),
        "fear_greed":     {"value": fg.get("value", 50), "label": fg.get("label", "neutral")},
        "open_trades":    len(executor.open_trades),
        "balance":        float(getattr(executor, "balance", 10000)),
        "kelly_fraction": kelly_f,
        "agents":         agents,
        "phase_points":   phase_pts,
        "geometry": {
            "wavelet_entropy":            geo.get("wavelet_entropy"),
            "phasor_coherence":           geo.get("phasor_coherence"),
            "dominant_period":            geo.get("dominant_oscillation_period"),
            "attractor_spectral_entropy": geo.get("attractor_spectral_entropy"),
        },
        "portfolio": {
            "total_pnl":    float(portfolio.get("total_pnl")    or 0),
            "win_rate":     float(portfolio.get("win_rate")     or 0),
            "total_trades": int(portfolio.get("total_trades")   or 0),
        },
    }

    if neural_model is not None:
        ns = neural_model.stats()
        def _c(v):
            try:
                f = float(v)
                return 0.0 if (_m.isnan(f) or _m.isinf(f)) else f
            except Exception:
                return 0.0
        try:
            cond, conf, _ = neural_model.forward(context.context_data)
        except Exception:
            cond, conf = context.condition.value, context.confidence
        state["neural"] = {
            "param_count":    int(neural_model.param_count()),
            "training_steps": int(ns.get("training_steps", 0)),
            "recent_loss":    _c(ns.get("recent_loss", 0.0)),
            "learning_rate":  _c(ns.get("learning_rate", 0.0)),
            "condition":      cond,
            "confidence":     _c(conf),
        }

    return state


async def start_server(host: str = "0.0.0.0", port: int = 8765) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/",    index_handler)
    app.router.add_get("/ws",  ws_handler)
    app.router.add_post("/cmd", cmd_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"[Dashboard] ⬡ Command Deck v2 live at http://localhost:{port}")
    return runner


if __name__ == "__main__":
    asyncio.run(start_server())
